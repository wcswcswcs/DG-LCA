# DG-KAN v3.7：PureKAN-UFULL 优先修复、代码审计与实验计划

> 目标：优先让 `PureKAN-UFULL` 全面超过 `PureKAN-AdamW`、`Hybrid-DGKAN-UFULL` 和 `MLP-AdamW`。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`。  
> 当前结论前提：v3.6 已证明 PureKAN 架构本身可训练，`PureKAN-AdamW` 表现强；但当前 `PureKAN-UFULL` 明显落后，说明必须优先检查实现、配置和 PureKAN 专用 functional update 设计。

---

## 0. 本轮为什么必须优先 PureKAN-UFULL

当前最严重的问题不是 Hybrid-DGKAN 是否还能工作，而是：

$$
\boxed{
\text{如果 PureKAN-UFULL 不能超过 PureKAN-AdamW，}
\text{那么我们不能宣称 U-FULL 是完整 KAN 网络的 optimizer。}
}
$$

v3.6 结果显示：

```text
PureKAN-AdamW 可训练，并且在 Fashion/KMNIST/MNIST 上表现不弱；
PureKAN-UFULL 在三个数据集上 accuracy 和 validation-loss AUC 都明显落后。
```

这意味着当前叙事必须收紧：

```text
旧叙事：U-FULL 是通用 KAN functional optimizer。
新问题：U-FULL 目前只在 Hybrid-DGKAN branch setting 中成立，
       还没有证明能训练完整 PureKAN 网络。
```

本轮目标不是继续给 Hybrid 找更好 profile，而是优先解决：

$$
\boxed{
\text{PureKAN-UFULL 为什么输给 PureKAN-AdamW？}
}
$$

并把它推进到：

$$
\boxed{
\text{PureKAN-UFULL} \geq
\max(\text{PureKAN-AdamW},\ \text{Hybrid-DGKAN-UFULL},\ \text{MLP-AdamW})
}
$$

---

## 1. 代码审计：目前发现的高优先级风险

### 1.1 风险一：`diagwarmup` 很可能没有真正生效

当前 `PureKAN-UFULL` 配置里有：

```text
v3_phase_mode = hard
v3_metric_active = full_sobolev_gram
v3_metric_transition = full_sobolev_gram
v3_metric_geometry = full_sobolev_gram
branch_max_active_frac = 0.0
branch_boost = 1.0
branch_final_scale = 1.0
```

训练 loop 中，在每个 step 前有逻辑：

```text
if progress >= branch_max_active_frac:
    switch_branch_state(...)
```

因为 `branch_max_active_frac = 0.0`，所以在 step 0 就会直接触发 switch，进入 `GEOMETRY` phase。这样会导致：

```text
ACTIVE phase 基本不存在；
v3_metric_active 不会真正参与训练；
如果 diagwarmup 只是改 active metric，它会和 U-FULL 完全一样。
```

v3.6 中 `PureKAN-UFULL` 和 `PureKAN-UFULL-diagwarmup` 三个数据集结果完全相同，这非常像配置没有生效，而不是方法自然得到完全相同曲线。

因此本轮第一修复项是：

```text
PureKAN-UFULL-full:
  从 step 0 使用 full_sobolev_gram。

PureKAN-UFULL-diagwarmup:
  必须保证 early steps 真正使用 basis_diag_gram 或 identity/diag。
  branch_max_active_frac 不能是 0。
```

建议新增字段：

```text
pure_metric_schedule = full_from_start | diag_warmup | identity_warmup | layerwise
pure_warmup_frac = 0.15 或 0.25
```

并要求每个 run 输出：

```text
phase_trace
metric_mode_trace
metric_mix_trace
active_steps_actual
geometry_steps_actual
```

---

### 1.2 风险二：PureKAN 的 `alphaFixed1` 标签可能不等于 block alpha 真的为 1

当前 `PureKANClassifier` 的 block alpha 来自：

```text
PureResidualKANBlock(..., alpha=alpha_init)
```

而不是从 `alpha_mode=fixed1` 派生。也就是说，如果全局 `alpha_init=1.5`，那么 PureKAN block 的实际残差系数可能是：

$$
\alpha = 1.5
$$

即使方法名叫 `alphaFixed1`。这会造成两个问题：

```text
1. 实验标签和真实模型不一致；
2. PureKAN-AdamW 与 PureKAN-UFULL 都可能在一个不是预期的 alpha setting 下比较。
```

本轮必须修复：

```text
PureKANClassifier 接受 alpha_mode。
如果 alpha_mode = fixed1，则 block alpha = 1.0。
如果 alpha_mode = fixed_init，则 block alpha = alpha_init。
如果 alpha_mode = learnable，则 PureKAN 是否允许 learnable alpha 要单独显式开启。
```

并记录：

```text
pure_alpha_mode
pure_alpha_layer_0
pure_alpha_layer_1
pure_alpha_mean
pure_alpha_trainable
```

---

### 1.3 风险三：PureKAN 的 no-KAN / contribution audit 现在不完整

当前 `supports_disable_kan` 不包含 `PureKANClassifier`。因此 PureKAN 的 `no_kan_drop` 很可能被写成 0 或等于 full model 对照，不能用于判断 PureKAN 的层贡献。

但是 PureKAN 没有普通 MLP stem/head，所谓 “disable KAN” 本身不再有同样含义。正确做法不是简单 no-KAN，而是做 layer ablation：

```text
input_kan ablation: 不建议直接置零，因为会破坏输入通道；可做 frozen/random/control。
block_kan ablation: disable residual KAN blocks。
output_kan ablation: 替换为 linear probe 或 frozen output KAN。
```

本轮必须新增 PureKAN 专用 contribution metrics：

```text
pure/block_disable_acc
pure/block_disable_loss
pure/block_no_residual_drop
pure/output_randomized_acc
pure/input_frozen_acc
pure/layerwise_logit_delta_norm
pure/layerwise_margin_contribution
```

---

### 1.4 风险四：PureKAN 所有层共用同一个 metric / lr，可能是根本不合适

PureKAN 有三类 KAN 层：

```text
input_kan: raw input -> hidden
block_kan: hidden -> hidden residual branch
output_kan: hidden -> logits
```

当前 U-FULL 对所有 RBFDense 层用同一个：

```text
metric = full_sobolev_gram
coeff_lr = same value
trust_radius = same value
```

这很可能不合理。因为三类层的作用不同：

```text
input_kan 需要快速形成表征；
block_kan 需要稳定 residual transformation；
output_kan 需要快速对齐分类 margin。
```

本轮必须把 PureKAN-UFULL 改成 role-aware：

```text
pure_input_metric
pure_block_metric
pure_output_metric
pure_input_lr_mult
pure_block_lr_mult
pure_output_lr_mult
pure_input_trust_radius
pure_block_trust_radius
pure_output_trust_radius
```

如果不做 role-aware，PureKAN-UFULL 很可能继续输给 AdamW。

---

### 1.5 风险五：full Sobolev from start 可能对 PureKAN 全网络过度平滑

Hybrid-DGKAN 中，U-FULL 只优化 residual KAN branch，non-KAN stem/head 由 AdamW 提供稳定表征与分类通道。PureKAN 中，input 和 output 都是 KAN：

$$
h_0 = \operatorname{KAN}_{in}(x)
$$

$$
h_{l+1} = h_l + \operatorname{KAN}_l(\operatorname{FixedNorm}(h_l))
$$

$$
z = \operatorname{KAN}_{out}(\operatorname{FixedNorm}(h_L))
$$

如果所有层从 step 0 都用 full Sobolev，可能会出现：

```text
input representation 学得太慢；
output logits 对齐太慢；
validation loss AUC 明显落后；
accuracy 后期也追不上 AdamW。
```

因此 PureKAN 需要优先测试：

```text
full_from_start 是否真的适合所有层；
diag/identity warmup 是否必要；
input/output 是否应该用 diag 或 identity，而 block 用 full Sobolev。
```

---

## 2. 本轮核心假设

本轮只验证一个主命题：

$$
\boxed{
\text{PureKAN-UFULL 能否通过实现修复和 role-aware functional update，}
\text{全面超过 PureKAN-AdamW、Hybrid-DGKAN-UFULL 和 MLP-AdamW？}
}
$$

这里 “全面超过” 定义为：

### 2.1 Accuracy

对每个 dataset：

$$
\operatorname{Acc}_{PureKAN-UFULL}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{Hybrid-UFULL},
\operatorname{Acc}_{MLP-AdamW}
)
$$

正式 5/10-seed confirm 中允许 paired CI 边界非常小，但 mean 必须不低于最强 baseline。

### 2.2 Convergence

用 validation loss AUC：

$$
\operatorname{AUCImprove}_{PureKAN-UFULL\ vs\ best\ baseline}>0
$$

其中：

$$
\operatorname{AUCImprove}
=
\frac{
\operatorname{AUC}_{baseline}-\operatorname{AUC}_{method}
}{
\operatorname{AUC}_{baseline}+\epsilon
}.
$$

### 2.3 Geometry

PureKAN-UFULL 必须比 PureKAN-AdamW 有更好函数几何：

$$
\phi'_{p95,U FULL}<\phi'_{p95,AdamW}
$$

$$
\kappa(J)_{U FULL}<\kappa(J)_{AdamW}
$$

### 2.4 Calibration

$$
\operatorname{ECE}_{U FULL}<\operatorname{ECE}_{AdamW}
$$

### 2.5 Functional legitimacy

PureKAN-UFULL 必须满足：

```text
learnable_nonKAN_params = 0
coeff_params_seen_by_functional_step = all learnable params
AdamW optimizer state = 0 for PureKAN-UFULL
```

否则不能叫 PureKAN-UFULL。

---

## 3. 必须先修的实现项

### Fix 1：PureKAN alpha 真实 fixed1

修改：

```python
class PureResidualKANBlock(nn.Module):
    def __init__(self, dim, basis_count, alpha_init=1.0, alpha_mode="fixed1"):
        if alpha_mode == "fixed1":
            self.register_buffer("alpha", torch.tensor(1.0))
        elif alpha_mode == "fixed_init":
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        elif alpha_mode == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
```

`PureKANClassifier` 也传入 `alpha_mode`。

必须 assert：

```text
method contains alphaFixed1 -> all pure alpha = 1.0
```

---

### Fix 2：PureKAN diagwarmup 不允许 step0 switch

新增 PureKAN-specific schedule：

```text
PureKAN-UFULL-full:
  active metric = full
  branch_max_active_frac = 0.0

PureKAN-UFULL-diagwarmup:
  active metric = basis_diag_gram
  geometry metric = full_sobolev_gram
  branch_max_active_frac = 0.20
  v3_phase_mode = hard or smooth

PureKAN-UFULL-identitywarmup:
  active metric = identity
  geometry metric = full_sobolev_gram
  branch_max_active_frac = 0.10 or 0.20
```

必须记录并检查：

```text
active_steps_actual > 0 for diagwarmup/identitywarmup
metric_active_seen = basis_diag_gram or identity
metric_geometry_seen = full_sobolev_gram
```

---

### Fix 3：PureKAN layer-role specific metric / LR

新增 config：

```text
pure_input_metric = identity | basis_diag_gram | full_sobolev_gram
pure_block_metric = identity | basis_diag_gram | full_sobolev_gram
pure_output_metric = identity | basis_diag_gram | full_sobolev_gram

pure_input_lr_mult
pure_block_lr_mult
pure_output_lr_mult
```

最小实现可以在 `functional_coeff_step` 中按参数名判断：

```text
input_kan.coeff -> role=input
blocks.*.kan.coeff -> role=block
output_kan.coeff -> role=output
```

然后选择 metric 和 lr multiplier。

---

### Fix 4：PureKAN update coverage audit

每个 run 输出：

```text
pure/learnable_total_params
pure/learnable_nonkan_params
pure/coeff_param_count
pure/coeff_param_numel_total
pure/coeff_param_seen_ratio
pure/input_update_norm
pure/block_update_norm_mean
pure/output_update_norm
pure/input_update_over_param
pure/block_update_over_param_mean
pure/output_update_over_param
pure/input_raw_grad_norm
pure/block_raw_grad_norm_mean
pure/output_raw_grad_norm
pure/input_precond_norm
pure/block_precond_norm_mean
pure/output_precond_norm
```

通过条件：

$$
\text{coeff\_param\_seen\_ratio}=1.0
$$

$$
\text{learnable\_nonKAN\_params}=0
$$

---

### Fix 5：PureKAN layer contribution audit

新增 PureKAN 专用 ablation：

```text
pure/block_disable_acc
pure/block_disable_drop
pure/input_frozen_probe_acc
pure/output_frozen_probe_acc
pure/layerwise_logit_delta_norm
pure/layerwise_margin_contribution
```

尤其要记录：

$$
\Delta_{block}
=
\operatorname{Acc}_{full}-\operatorname{Acc}_{disable\ residual\ blocks}
$$

这比普通 no-KAN 更适合 PureKAN。

---

## 4. 实验路线总览

```text
P0: Code and config audit smoke
P1: One-batch shadow-step diagnosis
P2: Metric schedule repair test
P3: Layer-role PureKAN-UFULL sweep
P4: Capacity and epoch budget check
P5: 3-seed candidate selection
P6: 5-seed confirm against all baselines
P7: 10-seed final confirm
P8: Failure analysis if P6/P7 fail
```

---

## 5. P0：代码与配置 smoke

### 5.1 目标

确认 PureKAN-UFULL 真的是 PureKAN，且所有 learnable 参数都被 functional update 覆盖。

### 5.2 数据集

```text
Fashion-MNIST
KMNIST
MNIST
```

### 5.3 方法

```text
PureKAN-AdamW
PureKAN-UFULL-full
PureKAN-UFULL-diagwarmup
PureKAN-UFULL-identitywarmup
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

### 5.4 设置

```text
train/val/test = 512/128/128
epochs = 1
seed = 0
hidden_dim = 96
basis_count = 24
depth = 4
alpha_mode = fixed1
```

### 5.5 必须记录

```text
learnable_total_params
learnable_kan_params
learnable_nonkan_params
purekan_nonkan_param_count
purekan_has_linear
purekan_has_layernorm_params
purekan_has_bias_params
coeff_param_count
coeff_param_names
coeff_param_seen_ratio
pure_alpha_mean
pure_alpha_layer_k
metric_trace_first_20_steps
phase_trace_first_20_steps
input_update_norm
block_update_norm_mean
output_update_norm
input_metric_mode_seen
block_metric_mode_seen
output_metric_mode_seen
trust_clip_rate
run_failed
nan_count
```

### 5.6 通过条件

```text
PureKAN-UFULL:
  learnable_nonkan_params = 0
  purekan_has_linear = 0
  purekan_has_layernorm_params = 0
  purekan_has_bias_params = 0
  coeff_param_seen_ratio = 1.0
  alpha = 1.0 when alphaFixed1
  diagwarmup trace differs from full trace
  all key update norms finite
  trust_clip_rate < 0.05
```

如果 P0 不过，不允许进入 P1。

### 5.7 可视化

```text
config audit table
coefficient coverage bar
metric phase trace for first 20 steps
per-layer update norm bar
```

---

## 6. P1：one-batch shadow-step diagnosis

### 6.1 目标

判断 PureKAN-UFULL 失败是：

```text
step 太小；
direction 错；
full Sobolev 太平滑；
input/output 层被错误 precondition；
trust clipping 压制；
还是 schedule 没生效。
```

### 6.2 做法

对同一个初始化和同一个 minibatch，计算不同 candidate update 的 shadow step。不要真的训练，只复制参数后试一步：

```text
raw SGD / identity functional
basis_diag_gram
full_sobolev_gram
diag_to_full
layerwise: input diag, block full, output diag
layerwise: input identity, block full, output identity
AdamW one-step reference
```

### 6.3 记录

```text
shadow/train_loss_before
shadow/train_loss_after
shadow/val_loss_before
shadow/val_loss_after
shadow/predicted_descent
shadow/actual_descent
shadow/bad_step_bool
shadow/raw_grad_norm_by_layer
shadow/precond_grad_norm_by_layer
shadow/update_norm_by_layer
shadow/update_over_param_by_layer
shadow/cos_raw_precond_by_layer
shadow/metric_condition_by_layer
shadow/trust_clip_by_layer
```

### 6.4 判定

如果：

$$
\Delta L_{actual}<0
$$

表示这个 update 是 descent step。

如果 full Sobolev 的 actual descent 比 diag 差很多，说明：

```text
full metric 对 PureKAN 早期不适合。
```

如果 output layer 的 preconditioned direction 最差，说明：

```text
output_kan 不能用同一套 full Sobolev。
```

### 6.5 可视化

```text
predicted vs actual descent scatter
per-layer cos(raw, precond) heatmap
per-layer update_over_param heatmap
metric condition by layer bar
bad step rate by metric mode
```

---

## 7. P2：metric schedule repair test

### 7.1 目标

验证修复后的 diagwarmup / identitywarmup 是否真正改变 PureKAN-UFULL trajectory。

### 7.2 方法

```text
PureKAN-AdamW
PureKAN-UFULL-full
PureKAN-UFULL-diagwarmup-0.10
PureKAN-UFULL-diagwarmup-0.20
PureKAN-UFULL-identitywarmup-0.10
PureKAN-UFULL-identitywarmup-0.20
PureKAN-UFULL-diag-to-full-smooth-0.20
```

### 7.3 设置

```text
datasets = Fashion-MNIST, KMNIST, MNIST
seeds = 0,1,2
epochs = existing PureKAN setting
hidden_dim = 96
basis_count = 24
depth = 4
```

### 7.4 记录

```text
acc
val_loss_auc
val_acc_auc
ECE
phi_prime_p95
jacobian_condition
curvature_energy
basis_dead_frac
out_of_grid_frac
input_update_norm
block_update_norm
output_update_norm
metric_phase_auc
active_steps_actual
transition_loss_jump
```

### 7.5 通过条件

P2 的目标不是最终超过 AdamW，而是必须证明：

```text
diagwarmup / identitywarmup 和 full_from_start 曲线不同；
至少一个 schedule 在 validation-loss AUC 上优于 PureKAN-UFULL-full；
至少一个 schedule 在 accuracy 上缩小 PureKAN-UFULL vs PureKAN-AdamW gap。
```

---

## 8. P3：layer-role PureKAN-UFULL sweep

### 8.1 目标

找到真正适合 PureKAN 的 layer-wise functional update。

### 8.2 候选

#### Candidate A：all-full

```text
input = full
block = full
output = full
```

这是当前 U-FULL baseline。

#### Candidate B：input/output diag, block full

```text
input = basis_diag_gram
block = full_sobolev_gram
output = basis_diag_gram
```

假设：input/output 需要更快的 task fitting，block 负责 geometry。

#### Candidate C：input identity, block full, output diag

```text
input = identity
block = full_sobolev_gram
output = basis_diag_gram
```

假设：input layer 不应被 Sobolev 过度平滑。

#### Candidate D：input diag, block full, output identity

```text
input = basis_diag_gram
block = full_sobolev_gram
output = identity
```

假设：output classifier 需要最快对齐 margin。

#### Candidate E：all-diag warmup then rolewise

```text
early:
  input = identity or diag
  block = diag
  output = identity or diag
late:
  input = diag
  block = full
  output = diag
```

### 8.3 LR multipliers

每个 candidate 先用小网格：

```text
input_lr_mult in {1.0, 2.0}
block_lr_mult in {1.0}
output_lr_mult in {1.0, 2.0, 4.0}
```

不要先大扫。

### 8.4 数据集与 seeds

```text
Fashion-MNIST, KMNIST, MNIST
seeds = 0,1,2
```

### 8.5 记录

```text
acc
val_loss_auc
ECE
phi_prime_p95
jacobian_condition
input_update_over_param
block_update_over_param
output_update_over_param
input_cos_raw_precond
block_cos_raw_precond
output_cos_raw_precond
input_metric_condition
block_metric_condition
output_metric_condition
logit_margin
classwise_acc
classwise_margin
pure_block_disable_drop
```

### 8.6 判定

进入 P5 的候选必须满足：

```text
acc gap vs PureKAN-AdamW < 1.0% on all three datasets
AUC improvement vs PureKAN-UFULL-full > 0
geometry better than PureKAN-AdamW
ECE better than PureKAN-AdamW or not worse by > 5%
```

---

## 9. P4：capacity and epoch budget check

### 9.1 目标

排除 PureKAN-UFULL 只是训练预算不足。

### 9.2 方法

只拿 P3 最好的 2 个候选，测试：

```text
epochs = base, 1.5x, 2x
hidden_dim = 96, 128
basis_count = 24, 32
```

### 9.3 判定

如果 PureKAN-UFULL 只需要更长训练即可超过 AdamW，则：

```text
AUC 会逐渐转正；
final acc 会追上；
update norm 不会异常小。
```

如果更长训练仍然 AUC 负，则是 update direction 问题，不是预算问题。

---

## 10. P5：3-seed candidate selection

### 10.1 Baselines

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
PureKAN-UFULL-full
```

### 10.2 Candidates

从 P3/P4 选最多 3 个：

```text
PureKAN-UFULL-rolewise-best1
PureKAN-UFULL-rolewise-best2
PureKAN-UFULL-schedule-best
```

### 10.3 Gate

进入 P6 的候选必须：

```text
Fashion acc >= max baselines - 0.5%
KMNIST acc >= max baselines - 0.5%
MNIST acc >= max baselines - 0.5%
AUC better than PureKAN-AdamW on at least 2/3 datasets
geometry better than PureKAN-AdamW on 3/3 datasets
ECE not worse than PureKAN-AdamW by > 5%
```

如果没有候选通过 P5，则不允许跑 P6/P7，必须回到 P1/P3 诊断。

---

## 11. P6：5-seed confirm against all baselines

### 11.1 数据集

```text
Fashion-MNIST
KMNIST
MNIST
```

### 11.2 Methods

```text
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
PureKAN-AdamW
PureKAN-UFULL-final-candidate
```

### 11.3 Seeds

```text
seeds = 0..4
```

### 11.4 成功条件

#### Hard pass

PureKAN-UFULL 在 3 个数据集上满足：

$$
\operatorname{Acc}_{U FULL}
\geq
\max(\operatorname{Acc}_{PureAdamW},\operatorname{Acc}_{Hybrid},\operatorname{Acc}_{MLP})
$$

并且：

$$
\operatorname{AUC}_{U FULL}<\operatorname{AUC}_{PureAdamW}
$$

$$
\operatorname{ECE}_{U FULL}<\operatorname{ECE}_{PureAdamW}
$$

$$
\phi'_{U FULL}<\phi'_{PureAdamW}
$$

#### Medium pass

如果 accuracy 只在某个数据集差不超过 `0.5%`，但 AUC/ECE/geometry 明显优于 AdamW，可以进入 P7，但不能写全面超过。

---

## 12. P7：10-seed final confirm

只在 P6 hard pass 或非常接近 hard pass 时运行。

```text
seeds = 0..9
methods = same as P6
```

最终报告 paired seed delta：

```text
PureKAN-UFULL vs PureKAN-AdamW
PureKAN-UFULL vs Hybrid-DGKAN-UFULL
PureKAN-UFULL vs MLP-AdamW
```

记录：

```text
acc delta mean / CI
AUC delta mean / CI
ECE delta mean / CI
phi/J delta mean / CI
```

---

## 13. P8：如果 PureKAN-UFULL 仍失败，必须输出 failure diagnosis

如果 P5/P6 失败，不允许只写 “PureKAN-UFULL failed”。必须输出失败类型。

### Failure type A：step too small

证据：

```text
update_over_param 很小
actual descent 为正但太小
longer training 可追上
```

### Failure type B：direction wrong

证据：

```text
cos(raw, precond) 很低
shadow actual descent 差
AUC 持续为负
```

### Failure type C：output layer bottleneck

证据：

```text
output_update_norm 低
output margin contribution 低
output identity/diag 比 full 好
```

### Failure type D：input layer bottleneck

证据：

```text
input_update_norm 低
input representation rank 低
input diag/identity 比 full 好
```

### Failure type E：over-smoothing

证据：

```text
phi/curvature 远低于 AdamW
accuracy 低
longer training 仍不上升
```

---

## 14. 统一 dashboard 必须包含的图

### Page 1：Final dominance scorecard

```text
method
dataset
acc mean/std
acc delta vs PureKAN-AdamW
acc delta vs Hybrid
acc delta vs MLP
AUC delta
ECE delta
phi/J reduction
pass status
```

### Page 2：Loss / accuracy curves

```text
train_loss vs epoch
val_loss vs epoch
val_acc vs epoch
```

每条曲线显示 mean ± std。

### Page 3：Pure layer update audit

```text
input_update_over_param vs epoch
block_update_over_param vs epoch
output_update_over_param vs epoch
input_raw_grad_norm / precond_norm
block_raw_grad_norm / precond_norm
output_raw_grad_norm / precond_norm
```

### Page 4：Direction audit

```text
cos(raw, precond) heatmap by layer role
predicted vs actual descent scatter
bad shadow step rate by metric
```

### Page 5：Metric condition

```text
input metric condition
block metric condition
output metric condition
metric eig_min/eig_max
trust clip rate
```

### Page 6：Representation and basis

```text
basis occupancy by layer
basis dead fraction
out-of-grid fraction
representation effective rank
feature norm distribution
```

### Page 7：Margin and classwise behavior

```text
logit margin distribution
classwise accuracy
classwise margin delta
block_disable_drop by class
```

---

## 15. 最终写法预案

### 情况 1：PureKAN-UFULL hard pass

可以写：

```text
After correcting PureKAN-specific phase scheduling and role-wise functional metrics,
PureKAN-UFULL outperforms PureKAN-AdamW, Hybrid-DGKAN-UFULL, and MLP-AdamW.
This supports U-FULL as a genuine optimizer for complete KAN networks.
```

### 情况 2：PureKAN-UFULL medium pass

可以写：

```text
PureKAN-UFULL becomes competitive with PureKAN-AdamW while improving geometry/calibration,
but does not yet dominate all baselines.
```

### 情况 3：PureKAN-UFULL fails after implementation fixes

必须写：

```text
PureKAN is expressive and trainable under AdamW, but current Sobolev functional update
is not sufficient as a full-network optimizer. U-FULL remains valid as a branch-level
functional optimizer inside Hybrid-DGKAN, but full PureKAN optimization remains open.
```

---

## 16. 本轮最重要的停止规则

不允许继续盲目大扫。

如果 P0 显示实现问题：

```text
先修代码，不跑实验。
```

如果 P1 显示 full Sobolev direction 非 descent：

```text
优先做 layerwise metric，不做 lr sweep。
```

如果 P3 显示 output layer 是瓶颈：

```text
只改 output metric / lr，不改 input/block。
```

如果 P5 没有候选接近 AdamW：

```text
停止 10-seed confirm，回到 functional metric 设计。
```

---

## 17. 最终优先级

```text
1. 修 PureKAN alpha 和 diagwarmup schedule。
2. 确认所有 PureKAN 参数都被 functional update 覆盖。
3. 做 one-batch shadow-step，判断 full Sobolev 是否 direction 错。
4. 做 role-wise metric：input/output 不再强制 full Sobolev。
5. 只在 3-seed 接近 PureKAN-AdamW 后做 5/10-seed。
```

本轮成功后，项目叙事才能从：

```text
Hybrid branch functional optimizer
```

推进到：

```text
Pure KAN full-network functional optimizer
```
