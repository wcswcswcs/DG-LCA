# DG-KAN v9.2.7 FC-PureKAN Full-Pass、Purity Audit 与 Functional Gate 完整实验计划

> 本计划基于 v9.2.6 `FC-PureKAN PrimitiveRedesign` 的真实结果制定。  
> v9.2.6 是一个关键推进：第一次出现了 **FC-only PureKAN near-pass primitive**。  
> 但是，本轮仍不能宣称 “PureKAN 已经比肩或超越 MLP”。  
> 当前最准确的判断是：
>
> $$
> \boxed{
> \text{FC-PureKAN 已从 system/blocker 阶段推进到 near-pass trainability 阶段，但还没有达到 full pass，更没有达到 Beyond-MLP。}
> }
> $$
>
> v9.2.7 的目标不是继续做小修小补，也不是提前研究 PureKANConv / PureKANFormer。  
> 本轮只做 FC-PureKAN：先确认 `LinearLiftQuadraticEdgeBasis` 是否真正满足 strict PureKAN contract，再把 `LQ-t2-h256` 从 near-pass 推进到 robust full-pass，最后才决定 functional update 是否允许打开。

---

## 0. 最新状态与独立判断

### 0.1 v9.2.6 的核心事实

v9.2.6 的 terminal route 是：

```text
route = R3-FCPrimitiveTrainabilityNearPass
best_candidate = LQ-t2-h256
primitive_family = LinearLiftQuadraticEdgeBasis
success_v926_fc_primitive_p4 = true
success_v926_trainability_reentry = true
success_v926_functional_opened = false
```

最终 artifact：

```text
results/real_rerun_20260506/v926_fc_purekan_primitive_redesign_liftquad_compact_p5_20260509T174500Z/
```

关键数据：

```text
synthetic pairwise R2 = 0.9742757678

P4:
  forward ratio = 1.086593
  backward ratio = 0.735494
  step ratio = 1.104636
  compact memory ratio = 0.969501
  conservative memory ratio = 1.265113
  P4 pass = 1 under compact/recompute live-set

P5:
  datasets = MNIST,Fashion-MNIST,KMNIST
  seeds = 0,1,2
  rows = 9
  near-pass = 8/9
  macro delta vs MLP-match = -0.004000
  P5 near-pass = 1
  P5 full pass = 0
```

P5 中唯一没有 near-pass 的关键 row 是：

```text
KMNIST seed0:
  KAN acc = 0.818500
  MLP-match acc = 0.831000
  delta = -0.012500
  min pass = 0
```

其余 rows 均满足 `Acc_KAN >= Acc_MLP - 0.01`。

### 0.2 是否达到目标

没有达到终极目标，也没有达到 full pass。

必须区分三个层级：

```text
P4 kernel-native pass:
  达成。LQ-t2-h256 在 compact/recompute live-set 下通过 forward/backward/step/memory gate。

P5 near-pass:
  达成。8/9 rows near-pass，macro delta = -0.004000。

P5 full pass / Beyond-MLP:
  未达成。macro delta 仍为负，KMNIST seed0 超过 -0.01 tolerance。
```

因此当前不能写：

```text
Clean FullEdge / FC-PureKAN 已经完全比过 MLP；
PureKAN 已经可以进入 Conv / Former 扩展；
functional update 已经证明有独特优势；
external fair validation 已经可以声明成功。
```

当前能写的是：

$$
\boxed{
\text{v9.2.6 首次得到 FC-only PureKAN near-pass primitive，但仍需 full-pass 修复和严格 contract 审计。}
}
$$

### 0.3 为什么这次进展很重要

v9.2.5 之前的状态是：

```text
FC-D2:
  保留 pairwise interaction，但 P4 不闭合。

单层 GEMM-native edge-basis:
  系统很快，但 pairwise R2 为负，退化成 additive edge model。
```

v9.2.6 的 `LinearLiftQuadraticEdgeBasis` 改变了这个局面：

```text
1. 通过 linear lift 把输入映射到 lifted coordinates；
2. 在 lifted coordinates 上施加 quadratic edge basis；
3. 用 T2 / quadratic channel 恢复 pairwise interaction；
4. 使 backward 主要变成 GEMM + pointwise derivative；
5. 避免旧 D2 lowrank FullEdge 的 heavy coefficient reduction。
```

这说明之前我们把问题理解成“必须做昂贵 D2 FullEdge composition 才有 interaction”并不完全正确。更准确是：

$$
\boxed{
\text{需要 interaction，但不一定需要旧 D2 lowrank coefficient-reduction 形式。}
}
$$

`identity lift -> quadratic edge-basis` 是更便宜的 interaction route。

### 0.4 当前最关键的风险

v9.2.6 也暴露了三个必须严肃处理的风险。

第一，`LinearLiftQuadraticEdgeBasis` 的 PureKAN 等价性必须重新审计。  
如果 linear lift 是普通 trainable linear hidden layer，再接 quadratic activation，那么它可能退化成：

$$
h=xW,\quad y=B_2(h)V,
$$

这更像 MLP-with-quadratic-activation，而不是严格 FullEdge edge-function system。  
如果它可以被等价写成两层 FullEdge：

$$
h_j=\sum_i\phi^{0}_{ij}(x_i),
$$

$$
y_c=\sum_j\phi^{1}_{jc}(h_j),
$$

其中第一层 $\phi^0$ 是 identity edge-basis，第二层 $\phi^1$ 是 quadratic edge-basis，则它可以保留为 FC-PureKAN。  
所以 v9.2.7 第一任务不是继续刷 P5，而是做 **PureKAN equivalence audit**。

第二，P4 pass 依赖 compact/recompute live-set。  
官方 compact memory ratio 是 `0.969501`，但 conservative memory ratio 是 `1.265113`。这不是失败，但必须规范：compact/recompute 是否公平，MLP-match 是否也允许同等 recompute/checkpoint policy，哪些 tensor 算 model live-set，哪些 tensor 算 input batch / temporary / recompute。若 memory 口径不清，P4 pass 会被质疑。

第三，P5 只是 near-pass。  
当前 macro delta 为 `-0.004000`，不是 full pass。唯一 miss 是 KMNIST seed0，但不能简单把它当作偶然。必须做 robust confirmation，判断这是 seed/split noise、KMNIST-specific hard mode、margin/logit tail、lift conditioning，还是 basis capacity 边界。

---

## 1. v9.2.7 总体目标

v9.2.7 的总体目标是：

$$
\boxed{
\text{把 LQ-t2-h256 从 FC-PureKAN near-pass 推进到可信 full-pass 或明确边界。}
}
$$

这个目标包含四个子目标。

### 1.1 Strict FC-PureKAN validity

证明 `LinearLiftQuadraticEdgeBasis` 不是普通 MLP hidden path，而是合法 FC-PureKAN composition 或合法 edge-basis factorization。

必须满足：

```text
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
NoOrdinaryMLPPathPass = 1
NoTrainablePreprocessorPass = 1
NoLossBackwardPass = 1
NoTeacherNoLossModificationPass = 1
```

### 1.2 P4 fairness validity

证明 P4 pass 的 compact/recompute memory accounting 是公平可复现的，而不是因选择性记账通过。

必须同时记录：

```text
compact memory ratio
conservative memory ratio
MLP-match compact/checkpoint counterpart memory ratio
activation/cache/temp breakdown
recompute overhead
```

official route 使用 compact memory 之前，必须证明 compact policy 是明确实现策略，而不是删除失败数据。

### 1.3 P5 robust full-pass attempt

在不改 loss、不用 teacher、不加 sampler/class-weight、不打开 functional update 的前提下，修复或证伪 AdamW-only full pass。

P5 near-pass 已有：

$$
\Delta Acc_{\text{macro}}=-0.004000.
$$

v9.2.7 要目标化处理：

```text
KMNIST seed0 miss
macro delta negative
seed/split robustness
margin/logit tail
lift conditioning
basis/channel utilization
```

### 1.4 Functional update gate decision

Functional update 不得倒灌成 AdamW-only success。  
v9.2.7 可以在 P5 robust near-pass 保持成立后，做 gated functional decision；但 official AdamW full-pass 仍必须单独报告。

Functional update 只能回答：

```text
是否改善 curvature / ECE / NLL / robustness；
是否不伤 task；
是否系统成本可控。
```

不能回答：

```text
AdamW-only 是否成功。
```

---

## 2. 硬约束

### 2.1 Clean training contract

所有 official candidate 必须满足：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
distillation_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

训练目标：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

P5 AdamW-only 阶段只允许：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{\text{AdamW-equivalent}}.
$$

P5 不允许：

$$
\Delta\theta_{\text{functional}}.
$$

### 2.2 FC-PureKAN equivalence contract

每一层必须满足：

$$
h^{\ell+1}_j=\sum_i\phi^\ell_{ij}(h^\ell_i).
$$

每条 edge function 必须满足：

$$
\phi^\ell_{ij}(x)=\sum_k c^\ell_{ij,k}B^\ell_k(\tilde{x}).
$$

对于 `LinearLiftQuadraticEdgeBasis`，必须证明它能写成：

$$
h_j=\sum_i c^{0}_{ij}B_0(x_i),
$$

其中：

$$
B_0(x)=x.
$$

然后：

$$
y_c=\sum_j\left(c^{1}_{jc,0}h_j+c^{1}_{jc,2}T_2(h_j)\right).
$$

如果代码实现实际等价于：

$$
h=xW,\quad z=\sigma(h),\quad y=zV,
$$

且 $\sigma$ 作为普通 node activation / MLP hidden path，而不是 edge-basis function，则不能作为 strict FC-PureKAN official success。

### 2.3 Conv / Former deferred rule

本轮仍必须记录：

```text
PureKANConv_status = deferred_until_FC_PureKAN_P5_pass_or_explicit_user_unlock
PureKANFormer_status = deferred_until_FC_PureKAN_P5_pass_or_explicit_user_unlock
```

不得产生 PureKANConv / PureKANFormer measured candidate。

### 2.4 Comparison rule

Official 主比较必须是同参数 MLP-match：

$$
\left|
\frac{Params_{\text{KAN}}-Params_{\text{MLP-match}}}
{Params_{\text{MLP-match}}}
\right|\leq0.05.
$$

必须同时记录 same-shape diagnostic，但 same-shape 不能用于 official success：

```text
official_fair_comparison_vs_mlp_match.csv
same_shape_diagnostic_vs_mlp.csv
```

---

## 3. 核心假设

### H1：LQ primitive 的成功来自合法 FC-PureKAN composition，而不是偷偷回到 MLP hidden path

H1 是最重要的纯度假设。  
如果 LQ 只是普通 linear layer + quadratic activation + linear output，那么虽然它能 train，但不能作为 strict PureKAN 成功。  
如果 LQ 可以被证明为两层 FullEdge composition，那么它是合法的 FC-PureKAN primitive。

H1 成立标准：

```text
FullEdgeEquivalencePass = 1
ordinary_mlp_hidden_path_used = 0
trainable_preprocessor_used = 0
external_residual_shortcut_used = 0
lowrank_edge_factorization_pass = 1 if factorized
```

并且需要落盘：

```text
symbolic_formula_lq_as_full_edge.md
contract_equivalence_audit_v927.csv
```

### H2：v9.2.6 的 P5 miss 主要是稳定性 / margin / conditioning 问题，而不是 LQ 表达力根本不足

H2 假设：`8/9 near-pass` 与 macro `-0.004000` 说明 LQ 的表达力已经基本到达 MLP-match，剩余缺口来自某些 seed / dataset 的训练稳定性、margin tail 或 lifted feature conditioning。

H2 成立标准：

在 10-seed robust confirmation 中：

$$
\Delta Acc_{\text{macro}}\geq-0.01,
$$

且 near-pass rows：

$$
\frac{\#\{Acc_{\text{KAN}}\geq Acc_{\text{MLP}}-0.01\}}{\#\text{rows}}\geq0.80.
$$

若 full-pass repair 后：

$$
\Delta Acc_{\text{macro}}\geq0,
$$

则 H2 强成立。

### H3：KMNIST seed0 miss 不是偶然，要用 margin / CE tail / confusion / hard-sample attribution 判断

H3 成立标准：

若 KMNIST seed0 仍失败，必须能定位到至少一种机制：

```text
CE tail: CE_p99 much larger than MLP
Margin: margin_p10 < 0 or much lower than MLP
Class mode: classwise gap concentrated in specific classes
Lift conditioning: lifted feature condition number high
Feature rank: effective rank lower than MLP-match
Update scale: update_over_param outlier
```

如果无法定位机制，不能继续做修复 sweep。

### H4：Compact memory pass 必须经受 fair memory accounting

H4 假设：compact/recompute live-set 是合法系统策略，但必须与 MLP-match fair policy 对齐。

H4 成立标准：

```text
official compact memory ratio <= 1.05
conservative memory ratio recorded
MLP-match compact/checkpoint counterpart recorded
recompute overhead recorded
no hidden basis-cache deletion without accounting
```

并且：

$$
T_{\text{step,compact}}\leq1.50T_{\text{step,MLP-match}}.
$$

### H5：Full-pass 修复应优先从 parameterization / conditioning 出发，而不是 lr sweep

H5 假设：继续大规模 lr sweep 不是正路。v9.2.7 只能使用小型预注册修复：

```text
lift initialization
lift feature normalization
output scale
T2 vs T2T3 vs Legendre23
hidden 224/256/288 narrow bracketing
compact memory parity
```

H5 成立标准：

若修复有效，必须同时改善：

$$
\Delta Acc_{\text{macro}},
$$

以及至少一个机制指标：

```text
margin_p10
CE_p99
lift_condition_number
effective_rank
basis_usage_entropy
```

不能只报告 final accuracy。

### H6：Functional update 可作为 gated secondary branch，但不能替代 AdamW-only full pass

H6 成立标准：

Functional branch 若打开，必须满足：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}},
$$

并且系统成本：

$$
T_{\text{step,functional}}\leq1.50T_{\text{MLP-match}}.
$$

但即使 functional 通过，也只能声明：

```text
functional advantage on near-pass base
```

不能声明：

```text
AdamW-only full pass
```

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-MLP-match:
  同参数 MLP-match，CE-only，AdamW。

B1-MLP-same-shape:
  同输入/输出/hidden shape diagnostic，不作为 official success。

K0-S3-B2-P4-F0:
  v9.2 单层 active-k reference，P4 pass 但 P5 fail。

D2-R7-reference:
  v9.2.5 interaction-retaining D2 reference，interaction pass 但 P4 fail。

LQ0-LQ-t2-h256:
  v9.2.6 best candidate，P4 pass，P5 near-pass。
```

### 4.2 LQ purity variants

```text
EQ0-LQ-current:
  当前实现。

EQ1-LQ-explicit-two-layer-FullEdge:
  显式拆成 identity edge-basis lift layer + quadratic edge-basis output layer。

EQ2-LQ-factorization-audit:
  保持当前实现，但落盘 coefficient factorization equivalence proof。

EQ3-LQ-negative-MLP-hidden-control:
  故意使用普通 linear + node quadratic activation，作为 forbidden diagnostic，不参与 official route。
```

### 4.3 Full-pass repair candidates

这些候选必须全部保持 FC-PureKAN contract。

```text
R0-LQ-t2-h256-current:
  当前 reference。

R1-LQ-t2-h256-orthogonal-lift-init:
  lift 权重正交 / QR 初始化，降低 lifted feature condition number。

R2-LQ-t2-h256-fanin-output-scale:
  fan-in calibrated output scale，目标是减少 CE tail。

R3-LQ-t2-h256-lift-feature-normalized:
  lifted coordinate 内部 fixed normalization / edge-owned affine normalization，必须证明不是 LayerNorm/MLP hidden path。

R4-LQ-t2t3-h256:
  增加 T3 edge-basis channel，但必须保持 P4。

R5-LQ-legendre23-h256:
  替换 basis family，检查 Chebyshev 特异性。

R6-LQ-t2-h224:
  narrow hidden bracket，检查 h256 是否过宽或过拟合。

R7-LQ-t2-h288:
  narrow hidden bracket，检查 h256 是否容量不足。

R8-LQ-t2-h256-margin-scale-init:
  只改初始化 / output scale，不改 loss，不改 sampler。

R9-LQ-t2-h256-longer-budget:
  训练 budget diagnostic，epochs = 40，不作为直接 official unless P5-20epoch also reported。
```

### 4.4 Functional candidates

只有 P5 robust near-pass 或 P5 full pass 后才打开：

```text
F0-AdamW-only-reference
F1-EdgeCorrectionFunctional
F2-HighCurvatureLiftFunctional
F3-NoOpMatchedOverhead
F4-RandomDirection
```

Functional update 默认只作用于 quadratic edge-basis coefficient，不作用于 identity lift channel，除非单独证明不伤 task。

---

## 5. 实验阶段

## P0：route recap、contract 与 deferred 架构审计

### 目标

确认 v9.2.7 没有把 PureKANConv / PureKANFormer 带回 active route，并复现 v9.2.6 的关键 boundary。

### 必须记录

```text
candidate_id
candidate_family
measured
eligible_for_route
purekanconv_status
purekanformer_status
loss_type
label_smoothing
teacher_used
distillation_used
uses_loss_backward
fake_data_used
proxy_row_used
full_edge_equivalence_pass
ordinary_mlp_hidden_path_used
trainable_preprocessor_used
external_residual_shortcut_used
```

### 判断标准

```text
purekanconv_status = deferred
purekanformer_status = deferred
no measured Conv / Former candidate
loss_type = CE
label_smoothing = 0
teacher_used = 0
uses_loss_backward = 0
```

### 可视化

```text
p0_contract_heatmap.svg
p0_deferred_arch_registry.md
p0_route_gate_diagram.svg
```

---

## P1：LQ PureKAN equivalence audit

### 目标

审计 `LinearLiftQuadraticEdgeBasis` 是否是真正 FC-PureKAN，而不是普通 MLP hidden path。P1 是 v9.2.7 的第一硬 gate。P1 不过，后续 P5 full-pass 不能作为 official PureKAN success。

### 方法

对 `EQ0/EQ1/EQ2/EQ3` 执行：

```text
symbolic formula expansion
parameter ownership audit
edge-function equivalence check
negative control comparison
manual backward path check
non-KAN path audit
```

对于 LQ，需要落盘其公式：

$$
h_j=\sum_i c^{0}_{ij}x_i,
$$

$$
y_c=\sum_j c^{1}_{jc,0}h_j+\sum_j c^{1}_{jc,2}T_2(h_j).
$$

并说明第一层是 identity edge-basis FullEdge layer，第二层是 quadratic edge-basis FullEdge layer。若代码实现无法被这种形式覆盖，则 candidate 降级为 diagnostic。

### 必须记录

```text
candidate_id
formula_form
layer_count
edge_basis_channels
trainable_params_total
edge_owned_params
non_edge_params
ordinary_mlp_hidden_path_used
node_activation_used
full_edge_equivalence_pass
manual_forward
manual_backward
manual_update
GradRelErrMax
GradCosMin
OutputAbsDiffMax
equivalent_formula_file
```

### 判断标准

P1 pass：

```text
full_edge_equivalence_pass = 1
ordinary_mlp_hidden_path_used = 0
node_activation_used = 0 unless represented as edge-basis layer
non_edge_params = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
GradRelErrMax <= 1e-4
GradCosMin >= 0.999
```

### 可视化

```text
p1_lq_equivalence_diagram.svg
p1_param_ownership_bar.svg
p1_gradcheck_lollipop.svg
```

---

## P2：v9.2.6 P4/P5 reproduction and memory accounting audit

### 目标

复现 `LQ-t2-h256` 的 P4/P5 boundary，并规范 compact vs conservative memory accounting。

### 设置

```text
candidate = LQ-t2-h256
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
official_memory_mode = compact
also_record_conservative_memory = 1
also_record_mlp_recompute_counterpart = 1
```

### 必须记录

P4：

```text
forward_ratio
backward_ratio
step_ratio
compact_memory_ratio
conservative_memory_ratio
mlp_compact_memory_ratio
mlp_conservative_memory_ratio
recompute_time_overhead
basis_cache_MB
activation_cache_MB
temporary_MB
kernel_count
```

P5：

```text
dataset
seed
KAN_acc
MLP_match_acc
delta
min_pass
train_loss
test_loss
CE_p50
CE_p90
CE_p99
margin_mean
margin_p10
wrong_confidence_p95
ECE
NLL
```

### 判断标准

P4 reproduction pass：

$$
|forward_{\text{repeat}}-1.086593|\leq0.10,
$$

$$
|backward_{\text{repeat}}-0.735494|\leq0.10,
$$

$$
|step_{\text{repeat}}-1.104636|\leq0.10,
$$

$$
compact\_memory_{\text{repeat}}\leq1.05.
$$

P5 reproduction pass：

$$
p5\_near\_pass\_count\geq8/9,
$$

$$
|\Delta Acc_{\text{macro,repeat}}-(-0.004000)|\leq0.01.
$$

### 可视化

```text
p2_p4_memory_accounting_bar.svg
p2_p5_reproduction_gap_bar.svg
p2_compact_vs_conservative_memory.svg
p2_ce_margin_reproduction.svg
```

---

## P3：robust near-pass / full-pass confirmation

### 目标

判断 v9.2.6 的 near-pass 是否稳健，以及 full-pass 差距是否只是 3-seed sample noise。

### 设置

```text
candidate = LQ-t2-h256
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
epochs = 20
train_size = 9984
test_size = 2000
baseline = same-parameter MLP-match
functional_update = off
```

如果资源有限，先跑：

```text
priority seeds:
  previous miss row KMNIST seed0 repeat x3
  all datasets seeds 0,1,2 repeat
  then full 10-seed
```

但最终 route 只能基于完整预注册 run。

### 必须记录

```text
dataset
seed
repeat_id
KAN_acc
MLP_match_acc
delta_vs_mlp
min_pass
train_acc
train_loss
test_loss
ECE
NLL
CE_p99
margin_p10
wrong_confidence_p95
lift_condition_number
lift_effective_rank
basis_usage_entropy
forward_ratio
backward_ratio
step_ratio
compact_memory_ratio
```

### 判断标准

Robust near-pass：

$$
\frac{\#\text{near-pass rows}}{\#\text{rows}}\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Full pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

Strong pass candidate：

$$
\Delta Acc_{\text{macro}}\geq0.005.
$$

### 可视化

```text
p3_seedwise_delta_stripplot.svg
p3_dataset_macro_gap_bar.svg
p3_near_pass_matrix.svg
p3_margin_ce_tail_by_dataset.svg
p3_lift_condition_vs_delta.svg
p3_task_system_pareto.svg
```

---

## P4：failure attribution for miss rows

### 目标

若 P3 中仍存在 miss row，尤其 KMNIST seed0 或其他 KMNIST rows，定位问题来源，不允许盲修。

### 必须记录

```text
dataset
seed
class_id
classwise_acc_KAN
classwise_acc_MLP
classwise_gap
confusion_matrix
hard_sample_ids
hard_sample_loss
hard_sample_margin
CE_p99
margin_p10
logit_norm_mean
logit_norm_p95
lift_feature_mean
lift_feature_std
lift_condition_number
lift_effective_rank
dead_lift_dim_fraction
dominant_lift_dim_fraction
basis_usage_entropy
grad_norm_lift
grad_norm_quadratic
update_over_param_lift
update_over_param_quadratic
```

### 判断标准

Attribution pass 要求 miss row 至少归入一种机制：

```text
M1-margin_tail:
  CE_p99 / margin_p10 明显差于 MLP。

M2-class_mode:
  classwise gap 集中在少数类别，且 confusion matrix 明确。

M3-lift_conditioning:
  lift condition number 或 dominant dim 过高。

M4-basis_underuse:
  basis usage entropy 低，dominant basis fraction 高。

M5-update_scale:
  update_over_param outlier 或 bad update rate 高。

M6-data_variance:
  repeated seed/split variance 解释了 miss。
```

如果无法归因，则不得进入 P5 repair sweep。

### 可视化

```text
p4_confusion_matrix_miss_rows.svg
p4_classwise_gap_heatmap.svg
p4_hard_sample_gallery.md
p4_lift_feature_spectrum.svg
p4_update_scale_trace.svg
p4_basis_usage_heatmap.svg
```

---

## P5：pre-registered full-pass repair

### 目标

在不改变 loss / teacher / sampler / class weight / functional update 的前提下，把 LQ 从 near-pass 推向 full-pass。P5 不是大 sweep，只允许根据 P4 attribution 选择少量预注册机制修复。

### 候选与适用条件

```text
R1-orthogonal-lift-init:
  适用于 M3-lift_conditioning。

R2-fanin-output-scale:
  适用于 M1-margin_tail 或 M5-update_scale。

R3-lift-feature-normalized:
  适用于 M3-lift_conditioning，但必须通过 PureKAN equivalence audit。

R4-t2t3-h256:
  适用于 M4-basis_underuse 或 expressive boundary。

R5-legendre23-h256:
  适用于 Chebyshev-specific conditioning issue。

R6-h224:
  适用于 over-capacity / unstable lift。

R7-h288:
  适用于 under-capacity / rank bottleneck。

R8-margin-scale-init:
  适用于 CE tail / logit scale issue。
```

### 必须记录

```text
candidate_id
repair_hypothesis
dataset
seed
KAN_acc
MLP_match_acc
delta_vs_mlp
P4_forward_ratio
P4_backward_ratio
P4_step_ratio
P4_compact_memory_ratio
P4_conservative_memory_ratio
CE_p99
margin_p10
lift_condition_number
effective_rank
basis_usage_entropy
GradRelErrMax
GradCosMin
full_edge_equivalence_pass
```

### 判断标准

Repair candidate promotion：

$$
\Delta Acc_{\text{macro,candidate}}\geq\Delta Acc_{\text{macro,LQ0}}+0.003.
$$

Full-pass repair：

$$
\Delta Acc_{\text{macro,candidate}}\geq0.
$$

P4 must remain pass：

$$
forward\leq1.25,\quad backward\leq1.50,\quad step\leq1.50,\quad compact\_memory\leq1.05.
$$

No mechanism regression：

```text
GradPass = 1
FullEdgeEquivalencePass = 1
R2_pairwise >= 0.95
```

### 可视化

```text
p5_repair_macro_delta.svg
p5_repair_mechanism_metric.svg
p5_repair_p4_pareto.svg
p5_before_after_kmnist_seed0.svg
```

---

## P6：survivor confirmation

### 目标

对 P5 survivor 做最终 10-seed confirmation，避免小修只在局部 seed 有效。

### 设置

```text
candidate = best P5 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
epochs = 20
baseline = MLP-match
functional_update = off
```

### 必须记录

```text
all P3 metrics
plus:
bootstrap_CI95_macro_delta
Holm_p
seed_win_rate
dataset_win_rate
P4_pass_count
conservative_memory_summary
```

### 判断标准

Official P5 near-pass：

$$
\Delta Acc_{\text{macro}}\geq-0.01,
$$

$$
near\_pass\_rate\geq0.80.
$$

Official P5 pass：

$$
\Delta Acc_{\text{macro}}\geq0,
$$

$$
CI^{low}_{95\%,macro}>-0.005.
$$

Strong task pass：

$$
\Delta Acc_{\text{macro}}\geq0.005,
$$

$$
CI^{low}_{95\%,macro}>0.
$$

### 可视化

```text
p6_macro_delta_ci.svg
p6_seed_win_matrix.svg
p6_dataset_win_bar.svg
p6_p4_task_joint_scorecard.svg
```

---

## P7：gated functional open decision and diagnostic

### 目标

只有 P6 near-pass 成立后，才允许 functional update diagnostic。P7 不得影响 AdamW-only P5 success 结论。

### Functional candidates

```text
F0-AdamW-only
F1-QuadraticCoeffFunctional
F2-HighCurvatureLiftFunctional
F3-NoOpMatchedOverhead
F4-RandomDirection
```

### 必须记录

```text
functional_mode
dataset
seed
test_acc
delta_vs_adamw
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
ECE_delta
NLL_delta
robustness_proxy
bad_step_rate
holdout_descent_ratio
functional_update_time_ratio
step_ratio
memory_ratio
```

### 判断标准

Functional causality：

$$
Curvature_{\text{Functional}}<Curvature_{\text{NoOp}},
$$

$$
Curvature_{\text{Functional}}<Curvature_{\text{Random}},
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Functional useful：

$$
Curvature_{\text{Functional}}\leq0.90Curvature_{\text{AdamW}},
$$

and at least one of:

$$
ECE_{\text{Functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{Functional}}\leq NLL_{\text{AdamW}},
$$

$$
RobustnessDrop_{\text{Functional}}\leq RobustnessDrop_{\text{AdamW}}.
$$

System gate：

$$
StepRatio_{\text{Functional}}\leq1.50,
$$

$$
MemoryRatio_{\text{Functional}}\leq1.05.
$$

### 可视化

```text
p7_functional_task_geometry_pareto.svg
p7_noop_random_control_matrix.svg
p7_curvature_vs_accuracy.svg
p7_functional_event_timeline.svg
```

---

## P8：route decision

### Route cases

```text
R1-LQStrictPureKANFullPass:
  LQ candidate passes PureKAN equivalence, P4, and P5 full pass.

R2-LQNearPassOnly:
  LQ candidate passes PureKAN equivalence and P4, reaches robust near-pass but not full pass.

R3-LQPurityFail:
  LQ trainability works, but equivalence audit shows ordinary MLP hidden path.

R4-LQMemoryAccountingUnfair:
  compact memory pass cannot be justified under fair accounting.

R5-LQFullPassRepairFail:
  attribution and repair fail to move macro delta to >= 0.

R6-FunctionalUsefulOnNearPassBase:
  AdamW remains near-pass, but functional gives valid geometry/calibration/robustness benefit.

R7-FunctionalNotOpened:
  P5 robust near-pass fails.

R8-FCPureKANPrimitiveNeedsRedesign:
  LQ fails purity or robust near-pass; next required is new FC-PureKAN primitive, not Conv/Former.
```

### route_decision.json 必须记录

```text
route
best_candidate
best_basis
best_hidden_dim
primitive_family
purekan_equivalence_pass
p4_pass
p5_near_pass
p5_pass
p5_macro_delta
p5_ci95_low
p5_seed_win_rate
compact_memory_ratio
conservative_memory_ratio
memory_accounting_pass
functional_open_allowed
functional_pass
primary_blocker
next_required_implementation
success_v927_purekan_equivalence
success_v927_p4
success_v927_p5_nearpass
success_v927_p5_fullpass
success_v927_functional
purekanconv_deferred
purekanformer_deferred
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v927.csv
deferred_architecture_registry_v927.csv
p1_lq_purekan_equivalence_audit.csv
p1_lq_symbolic_formula.md
p2_v926_reproduction_memory_accounting.csv
p2_p5_reproduction_rows.csv
p3_robust_nearpass_confirmation.csv
p4_miss_row_failure_attribution.csv
p5_fullpass_repair_candidates.csv
p6_survivor_confirmation.csv
p7_functional_open_diagnostic.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_purekan_equivalence_fail
F3_ordinary_mlp_path_detected
F4_memory_accounting_fail
F5_reproduction_fail
F6_robust_nearpass_fail
F7_kmnist_seed0_unresolved
F8_margin_tail_fail
F9_lift_conditioning_fail
F10_basis_underuse_fail
F11_repair_no_macro_gain
F12_p5_fullpass_fail
F13_functional_not_opened
F14_functional_causality_fail
F15_fake_or_proxy_violation
F16_artifact_missing
```

---

## 7. 第一轮执行顺序

```text
Step 1:
  P0 route recap and contract audit。
  明确 Conv / Former deferred。

Step 2:
  P1 LQ PureKAN equivalence audit。
  如果 P1 不过，LQ 不得作为 official PureKAN success。

Step 3:
  P2 v9.2.6 reproduction and memory accounting audit。
  复现 P4/P5，并规范 compact vs conservative memory。

Step 4:
  P3 robust 10-seed near-pass/full-pass confirmation。
  判断 near-pass 是否稳健。

Step 5:
  P4 miss row attribution。
  重点分析 KMNIST seed0 和新增 miss rows。

Step 6:
  P5 pre-registered full-pass repair。
  只允许机制驱动修复，不做大 sweep。

Step 7:
  P6 survivor confirmation。
  对 P5 survivor 做 10-seed confirmation。

Step 8:
  只有 P6 near-pass 成立，才打开 P7 functional diagnostic。
```

---

## 8. 停止条件

### 成功停止

Minimum v9.2.7 success：

```text
PureKAN equivalence pass
P4 pass
P5 robust near-pass
Conv/Former deferred
no fake/proxy
```

Full-pass success：

```text
PureKAN equivalence pass
P4 pass
P5 macro delta >= 0
CI95 macro low > -0.005
```

Strong success：

```text
Full-pass success
+
functional useful or ECE/NLL/robustness advantage
```

### 失败停止

```text
1. LQ fails PureKAN equivalence audit；
2. compact memory accounting cannot be justified；
3. v9.2.6 P4/P5 cannot be reproduced；
4. robust 10-seed near-pass fails；
5. miss rows cannot be attributed；
6. all pre-registered repairs fail to improve macro delta by >= 0.003；
7. P5 full-pass remains fail；
8. functional diagnostic harms task or exceeds system gate；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 9. 最终解释规则

### Case A：LQ reaches full pass

可以声明：

```text
FC-PureKAN reached AdamW-only full pass against same-parameter MLP-match.
```

但仍不能声明 broad Beyond-MLP 或 Conv / Former success，除非后续 external fair / functional / scaling 通过。

### Case B：LQ remains near-pass

必须声明：

```text
FC-PureKAN has reached robust near-pass, but not full pass. Current blocker is task margin/stability, not P4 system.
```

### Case C：LQ purity fails

必须声明：

```text
LQ is useful as a diagnostic primitive, but not an official strict PureKAN success.
```

### Case D：P4 memory accounting fails

必须声明：

```text
LQ task signal is real, but system fairness is not established under memory accounting audit.
```

### Case E：functional helps but AdamW full-pass fails

必须声明：

```text
Functional update shows secondary geometry benefit on a near-pass base, but AdamW-only full pass is still not established.
```

---

## 10. 最终建议

v9.2.7 的一句话策略是：

$$
\boxed{
\text{先审计 LQ 是否真 PureKAN，再把 near-pass 做成 robust full-pass；functional 只能作为 gated secondary evidence。}
}
$$

当前最重要的不是继续研究 Conv / Former，也不是马上开 functional，而是这四件事：

```text
1. LQ-t2-h256 是否严格等价于 FC-PureKAN composition；
2. compact memory P4 pass 是否公平；
3. 8/9 near-pass 是否能在 10 seeds 下保持；
4. KMNIST seed0 和 macro -0.004 能否通过机制性修复推到 full pass。
```
