# DG-KAN Optimizer v3.2 下一步详细实验计划：从 true-Gram 3-seed 信号到 clean default

> 日期：2026-05-01  
> 目标：在 v3.1 transition 语义修复和 true-Gram 3-seed 正信号之后，停止大范围搜索，把实验推进到 **可复现确认、机制解释、alpha 模式决策、wall-clock 收敛判断、以及后续 scaling 决策**。  
> 公式格式：Typora 友好，全文只使用 `$...$` 与 `$$...$$`。  
> 本计划不新增 data metric、SNR、FSAM、prox/spectral、target solve、naive Adam moment 等组件。它们继续保留为后续 profile，不进入 v3.2 主线。

---

## 0. 当前状态与下一步总判断

这一轮 v3.1 实验已经把问题从“v3 是否有希望”推进到了“如何把 true-Gram v3.1 固定成 clean default”。关键事实如下。

第一，`diag_to_full_sobolev` 的 transition 语义已经修正。旧语义是：

```text
旧语义: diag-fast -> full-fast
```

新语义是：

```text
新语义: diag-fast -> full-geo
```

R0 deterministic checks 中，learnable-alpha 与 fixed-alpha 两个版本都通过了端点检查：

```text
rel0 = 0
rel1 = 0
cond_fast ~= 6906
cond_geo ~= 6545
mix_first = 0.001
pass = True
```

这说明当 `mix=0` 时，direction 完全等于 `diag-fast direction`；当 `mix=1` 时，direction 完全等于 `full-geo direction`。因此，后续 v3.1 / v3.2 的结果不再被上一轮 transition 语义错位污染。

第二，Fashion-MNIST 是当前最清晰的成功点。learnable-alpha 下，`F-V3-HARD-base30` 在 3 seeds 上达到：

```text
acc = 0.8487 ± 0.0135
AUC improve = 10.33%
phi red = 34.2%
J red = 93.1%
branch/A = 0.665
ECE red = 24.4%
```

它相比 v2 bothcos 的 `AUC improve = 5.93%` 明显更强，同时 accuracy、geometry、branch utilization、ECE 都没有牺牲。因此，Fashion 当前默认候选应是：

```text
Fashion default candidate = F-V3-HARD-base30
```

`F-V3-HARD-f085` 虽然 accuracy 和 AUC 略高，但 Jacobian reduction 只有 `18.3%`，不满足“几何好”的主目标，所以不能作为 clean default。

第三，KMNIST 的 true-Gram 信号也成立，但最清楚的是 learnable-alpha track。learnable-alpha 下，`K-V3-FULL-restcos07` 和 `K-V3-FULL-bothcos07` 都形成了 3-seed 正结果：

```text
K-V3-FULL-restcos07:
  acc = 0.7803 ± 0.0135
  AUC improve = 9.33%
  phi red = 28.4%
  J red = 93.8%
  branch/A = 0.595
  ECE red = 23.8%

K-V3-FULL-bothcos07:
  acc = 0.7787 ± 0.0052
  AUC improve = 10.80%
  phi red = 28.5%
  J red = 98.3%
  branch/A = 0.597
  ECE red = 20.8%
```

这说明 `full_sobolev_gram from start` 已经不是 seed0 偶然信号，而是可以扩到 5 seeds 的候选。当前建议是：

```text
KMNIST accuracy/ECE candidate = K-V3-FULL-restcos07
KMNIST convergence/geometry candidate = K-V3-FULL-bothcos07
```

第四，fixed-alpha 不是简单全面更好。Fashion fixed-alpha 很强，尤其 hard true-Gram 的 AUC/J 更干净；但 KMNIST fixed-alpha 下 AdamW baseline 自身变强，导致 `K-V3-FULL-base` 虽然 AUC / phi / J 好，但 accuracy 没有超过 AdamW。当前代码状态又已经把 `ResidualKANBlock.alpha` 固定为不可学习的 `1.0`。因此，下一步必须把 `alpha_mode` 做成显式配置，否则后续实验会混淆“optimizer 改进”和“模型族改变”。

基于以上判断，v3.2 的总策略是：

$$
\boxed{
\text{不再大范围扫 optimizer 组件，转向 clean confirm + mechanism audit + alpha-mode decision。}
}
$$

更具体地说，下一步不是继续发明新 optimizer，而是确认下面两个 dataset-specific default：

$$
\boxed{
\text{Fashion default candidate: hard diag-to-full true Gram}
}
$$

$$
\boxed{
\text{KMNIST default candidate: full Sobolev Gram from start}
}
$$

同时必须单独解决：

$$
\boxed{
\text{alpha 是 learnable 还是 fixed，不应再通过手改代码决定。}
}
$$

---

## 1. v3.2 的实验原则

v3.2 的核心原则是 **把搜索空间变窄，把结论变硬**。当前 true-Gram 已经有明确正信号，如果继续扫 metric、transition、branch scale、coeff LR、rest LR、alpha、smooth/hard 等所有组合，实验会再次变成全家桶式探索。v3.2 要避免这种情况。

### 1.1 默认允许 dataset-specific default

现在的证据已经表明，Fashion 和 KMNIST 对 metric phase 的偏好不同。

Fashion 更像需要：

$$
\text{early diag activation} \rightarrow \text{late full Gram geometry control}.
$$

KMNIST 更像需要：

$$
\text{full Sobolev Gram from start}.
$$

所以 v3.2 不强求一个统一 method label 在所有数据集上最优。当前阶段可以接受：

```text
Fashion: F-V3-HARD-base30
KMNIST: K-V3-FULL-restcos07 或 K-V3-FULL-bothcos07
```

真正要统一的是方法思想：

$$
\boxed{
\text{true shared Sobolev Gram improves functional update direction。}
}
$$

而不是要求每个数据集都使用同一个 transition mode。

### 1.2 alpha 模式必须显式化

当前结果中同时存在 learnable-alpha 和 fixed-alpha。它们不是同一个模型的轻微变体，而是不同的 residual branch amplitude 参数化。

learnable-alpha 模型是：

$$
h_{k+1}=h_k+\alpha_k b_t\operatorname{KAN}_k(\operatorname{LN}(h_k)).
$$

fixed-alpha 模型是：

$$
h_{k+1}=h_k+1.0\cdot b_t\operatorname{KAN}_k(\operatorname{LN}(h_k)).
$$

这会影响 branch utilization、effective coefficient step、geometry audit 和 AdamW baseline。因此 v3.2 必须实现：

```text
alpha_mode = learnable
alpha_mode = fixed1
alpha_mode = fixed_init_optional
```

其中 `fixed1` 表示 `alpha=1.0` 且不可学习；`learnable` 表示按照 `alpha_init` 初始化并由 AdamW 优化；`fixed_init_optional` 只作为后续 ablation，不进入 clean confirm。

每个 run 的 metadata 必须记录：

```text
model/alpha_mode
model/alpha_init
model/alpha_trainable
model/alpha_final_layer_0
model/alpha_final_layer_1
...
```

如果不这样做，后续无法解释 KMNIST fixed-alpha 为什么 AdamW baseline 变强，也无法判断 Fashion fixed-alpha 的 AUC/J 改善到底来自 optimizer 还是 residual amplitude 改变。

### 1.3 smooth transition 暂停作为默认

修复后 smooth transition 已经不再是“明显坏实现”。但是当前 3-seed 成功点不是 smooth，而是：

```text
Fashion: hard diag-to-full
KMNIST: full-from-start
```

所以 v3.2 中 smooth transition 的地位是：

```text
已修复，可复查；
不是默认；
不进入 5-seed clean confirm 主线。
```

只有当 hard/full 候选在 5-seed confirm 中失败，或者 direction audit 显示 hard switch 有明显 loss jump，才重新打开 smooth transition。

### 1.4 先确认，再解释，再扩展

v3.2 的实验顺序必须是：

```text
实现可复现性修复
-> 5-seed clean confirm
-> 机制审计
-> target-matched / wall-clock 收敛
-> 10-seed confirm
-> Rational / CIFAR scaling
```

不能先跳到 Rational / CIFAR / DG-KANFormer，否则基础 optimizer 还没有 clean default，后续 scaling 的失败或成功都难解释。

---

## 2. v3.2 的核心假设

v3.2 要验证的假设不是泛泛的“v3 更好”，而是四个可以被证伪的具体假设。

### 2.1 H1：Fashion 的 clean default 是 hard diag-to-full true Gram

假设：Fashion-MNIST 需要 early diag metric 激活表达力，再切到 full Sobolev Gram 收住几何。hard switch 足够好，不需要 smooth transition。

候选：

```text
F-V3-HARD-base30
```

成功标准：5 seeds 上同时满足：

$$
\operatorname{Acc}_{F\text{-}V3}\geq\operatorname{Acc}_{AdamW}-0.003,
$$

$$
\operatorname{AUCImprove}>0.10,
$$

$$
\operatorname{PhiRed}>0.30,
$$

$$
\operatorname{JRed}>0.80,
$$

$$
0.55<\operatorname{branch/AdamW}<0.75,
$$

$$
\operatorname{ECERed}>0.15.
$$

如果 `F-V3-HARD-f085` accuracy 更高但 J reduction 仍显著低于 `base30`，则它只能作为 accuracy profile，不作为 clean default。

### 2.2 H2：KMNIST 的 clean default 是 full Sobolev Gram from start

假设：KMNIST 不需要 early diag activation，或者 early diag activation 会带来不稳定/无效方向；从一开始使用 full Sobolev Gram 更能改善 validation-loss trajectory。

候选：

```text
K-V3-FULL-restcos07
K-V3-FULL-bothcos07
```

成功标准：5 seeds 上至少一个满足：

$$
\operatorname{AccGap}_{vs AdamW}<0.005,
$$

$$
\operatorname{AUCImprove}>0.09,
$$

$$
\operatorname{PhiRed}>0.25,
$$

$$
\operatorname{JRed}>0.90,
$$

$$
0.55<\operatorname{branch/AdamW}<0.70,
$$

$$
\operatorname{ECERed}>0.15.
$$

如果 `bothcos07` 的 AUC/J 最强但 acc 比 `restcos07` 低，则默认选 `restcos07`，把 `bothcos07` 定位为 convergence/geometry profile。

### 2.3 H3：fixed-alpha 需要单独决策，不能继承 learnable-alpha profile

假设：fixed-alpha 对 Fashion 是安全甚至有利的，但对 KMNIST 需要重新调 coeff/rest/scale，不能直接沿用 learnable-alpha 的 full-from-start profile。

成功标准：如果要把 fixed-alpha 作为新代码默认，则 fixed-alpha 必须至少在 Fashion 和 KMNIST 都达到 medium candidate；否则主线仍保留 learnable-alpha，fixed-alpha 作为 ablation。

### 2.4 H4：true Gram 的收益来自 update direction，而不只是步长变化

假设：true shared Sobolev Gram 改善的是 functional-space direction，而不是偶然改变 update norm 或等价于调低/调高 LR。

需要通过 direction audit 证明：

$$
\cos(d_{diag-fast},d_{full-geo})
$$

与 AUC / geometry 的变化存在可解释关系，并且 full Gram 的 update metric norm 没有异常放大。

如果发现 true Gram 只是改变了 step norm，则需要加入 norm-matched control：

```text
basis_diag_gram + lr rescaled to match full_gram update_norm
```

### 2.5 H5：v3.1 的 branch 真的参与任务，而不是只让 branch norm 合理

假设：当前 branch/A 在 `0.59-0.67` 之间是健康的，但这只是 utilization proxy。要证明高表达力，需要 no-KAN drop 和 logit/margin contribution。

机制成功要求：

$$
\Delta_{noKAN}^{v3}>0.7\Delta_{noKAN}^{AdamW},
$$

并且：

$$
\operatorname{KANLogitDeltaNorm}^{v3}>0,
$$

$$
\operatorname{KANMarginContribution}^{v3}>0.
$$

---

## 3. 必须先做的代码与日志修复

v3.2 的第一阶段不是训练，而是把实验可复现性修干净。没有这些修复，后续 5-seed / 10-seed 的解释会很困难。

### 3.1 实现 `alpha_mode`

当前代码后半轮已经把：

```python
self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
```

改成了：

```python
self.register_buffer("alpha", torch.tensor(1.0))
```

v3.2 不应再通过手改代码切换。建议实现：

```python
class ResidualKANBlock(nn.Module):
    def __init__(self, dim, basis_count, alpha_init, alpha_mode="learnable"):
        ...
        if alpha_mode == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        elif alpha_mode == "fixed1":
            self.register_buffer("alpha", torch.tensor(1.0))
        elif alpha_mode == "fixed_init":
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        else:
            raise ValueError(alpha_mode)
```

`TrainConfig` 增加：

```text
alpha_mode: str = "learnable"
```

CLI 增加：

```text
--alpha-mode learnable|fixed1|fixed_init
```

每个 run 写入：

```text
alpha_mode
alpha_trainable
alpha_init
alpha_final_mean
alpha_final_std
alpha_layer_0_final
alpha_layer_1_final
...
```

对于 fixed-alpha，`alpha_final_*` 仍记录为 `1.0`，用于聚合表一致。

### 3.2 method label 必须包含 alpha 信息

后续结果表中禁止只写：

```text
K-V3-FULL-restcos07
```

而要写成：

```text
K-V3-FULL-restcos07-alphaLearn
K-V3-FULL-restcos07-alphaFixed1
```

或者在 summary group 中明确按：

```text
dataset, method, alpha_mode, epochs, train_size
```

聚合。

### 3.3 direction audit 字段要从 summary 扩展到 curve-level

当前已经新增了一些：

```text
v3_direction_*
v3_diagfast_*
v3_fullgeo_*
```

v3.2 需要把它们放到每个 epoch 或固定 step interval 的 curve 文件里。建议每个 run 输出：

```text
runs/{run_id}/curves.csv
runs/{run_id}/direction_audit.csv
runs/{run_id}/branch_audit.csv
```

`direction_audit.csv` 至少包含：

```text
step
epoch
phase
metric_mode
metric_mix
raw_grad_norm
diagfast_direction_norm
fullgeo_direction_norm
current_direction_norm
cos_raw_diagfast
cos_raw_fullgeo
cos_diagfast_fullgeo
cos_current_fullgeo
norm_fullgeo_over_diagfast
norm_current_over_diagfast
metric_norm_diagfast
metric_norm_fullgeo
metric_norm_current
update_over_coeff_norm
predicted_descent_proxy
actual_shadow_descent_optional
```

### 3.4 transition loss jump 要显式记录

尤其是 Fashion hard switch，需要记录 switch 前后 validation loss 是否跳变。

定义：

$$
\Delta L_{switch}=L_{val}(e_{switch}+1)-L_{val}(e_{switch}).
$$

如果 hard switch 的 `AUC` 很好但有明显 `transition_loss_jump`，则 smooth 还有复查价值。如果 hard switch 没有 jump，就不需要再打开 smooth。

### 3.5 保存 no-KAN 和 branch 机制审计

每个正式 run 至少在以下时刻做 no-KAN audit：

```text
epoch 0
epoch switch_epoch if exists
epoch mid
epoch final
best_val_epoch
```

记录：

```text
ablation/no_kan_val_acc
ablation/no_kan_test_acc
ablation/no_kan_val_loss
ablation/no_kan_test_loss
ablation/no_kan_acc_drop
ablation/no_kan_loss_increase
ablation/kan_logit_delta_norm
ablation/kan_margin_contribution
branch/final_global_ratio
branch/final_layer_k_ratio
branch/train_mean_ratio
branch/final_over_train_mean
```

否则只有 `branch/A`，不足以证明表达力。

---

## 4. 实验包总览

v3.2 分成十二个实验包。它们不是平行扫，而是按依赖顺序推进。

```text
V3.2-P0: reproducibility and alpha-mode implementation check
V3.2-P1: alpha-mode smoke and R0/R1 rerun
V3.2-P2: Fashion 5-seed clean confirm
V3.2-P3: KMNIST 5-seed clean confirm
V3.2-P4: alpha-mode decision
V3.2-P5: 10-seed final confirm
V3.2-P6: branch expression and no-KAN mechanism audit
V3.2-P7: true-Gram direction audit
V3.2-P8: target-matched and wall-clock convergence
V3.2-P9: fixed-alpha KMNIST repair, conditional only
V3.2-P10: smooth transition revisit, conditional only
V3.2-P11: Rational / KAT scaling precheck, after default freeze
V3.2-P12: CIFAR-small readiness, after P5-P8 pass
```

每个包都必须输出：

```text
runs.csv
summary_by_method.csv
summary_by_dataset.csv
failure_table.csv
aggregate_summary.json
plots/
```

---

## 5. V3.2-P0：reproducibility and alpha-mode implementation check

### 5.1 目标

P0 的目标是确认新代码没有改变旧结果语义，并且 `alpha_mode` 可以在同一份代码中切换。

这一步不做方法结论。它只回答：

```text
learnable-alpha 和 fixed-alpha 是否都能复现 R0/R1 smoke？
transition endpoint 是否仍然正确？
method label / metadata 是否完整？
```

### 5.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
model = h32 / depth2 / basis8
alpha_mode = learnable, fixed1
methods = AdamW, GA-FU-v2-current, GA-FU-v3-smoke-fixed-transition
```

其中 v3 smoke 使用：

```text
v3_metric_active = basis_diag_gram
v3_metric_transition = diag_to_full_sobolev
v3_metric_geometry = full_sobolev_gram
v3_phase_mode = smooth
```

### 5.3 必须记录

```text
run_failed
nan_or_inf_count
alpha_mode
alpha_trainable
transition_endpoint_rel0
transition_endpoint_rel1
v3_metric_condition_active
v3_metric_condition_geometry
v3_metric_eig_min_active
v3_metric_eig_max_active
v3_metric_eig_min_geometry
v3_metric_eig_max_geometry
trust_clip_rate
v3_phase_final
v3_metric_mix_auc
```

### 5.4 通过条件

P0 通过要求：

$$
\text{run failures}=0.
$$

$$
\text{all key metrics finite}.
$$

$$
\operatorname{rel0}=0,\quad \operatorname{rel1}=0.
$$

$$
1<\operatorname{cond}_{active}<10^4,
\quad
1<\operatorname{cond}_{geo}<10^4.
$$

$$
\operatorname{trust\_clip\_rate}<0.02.
$$

如果这里失败，不能进入 5-seed confirm。

### 5.5 可视化

P0 只需要简单 dashboard：

```text
bar: metric condition active / geometry by alpha_mode
bar: test acc by method / alpha_mode
bar: trust clip rate by method
line: metric mix over step for v3 smoke
```

---

## 6. V3.2-P1：alpha-mode sanity rerun

### 6.1 目标

P1 是一个小规模正式训练 sanity，用于确认 `alpha_mode` 开关没有引入新的结果偏差。它不是最终结论，但要帮助决定后续 P2/P3 是否同时跑 learnable 和 fixed。

### 6.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 6000 / 1000 / 1000
model = h96 / depth4 / basis24
epochs = Fashion 30, KMNIST 20
seeds = 0
alpha_mode = learnable, fixed1
```

### 6.3 方法

Fashion：

```text
AdamW
GA-FU-v2-tuned-bothcos
F-V3-HARD-base30
F-V3-HARD-c006-softcoeff
```

KMNIST：

```text
AdamW
GA-FU-v2-tuned
K-V3-FULL-base
K-V3-FULL-restcos07
K-V3-FULL-bothcos07
```

### 6.4 判定

P1 不根据单 seed 选最终方法，只检查：

```text
1. seed0 是否与上一轮同 alpha_mode 的趋势一致；
2. fixed1 是否继续出现 KMNIST AdamW baseline 更强的现象；
3. learnable-alpha 是否仍给出 KMNIST full-from-start 的强 AUC/J 信号；
4. Fashion hard true-Gram 是否仍强。
```

如果 P1 与上一轮严重不一致，优先检查：

```text
alpha_mode label 是否混乱
seeds 是否一致
data split 是否一致
epochs 是否一致
fixed-alpha 是否误加进 optimizer param group
method override 是否被覆盖
```

---

## 7. V3.2-P2：Fashion 5-seed clean confirm

### 7.1 目标

P2 是 Fashion 的正式 confirm。目标是把 3-seed 成功点扩到 5 seeds，并选择 Fashion 的 clean default。

Fashion 现在不需要大扫，只需要确认以下问题：

```text
F-V3-HARD-base30 是否稳定强于 v2 bothcos？
F-V3-HARD-c006-softcoeff 是否只是 AUC profile，还是可作为 fixed-alpha 默认？
F-V3-FULL-f085 是否需要保留？
```

### 7.2 设置

```text
dataset = Fashion-MNIST
train / val / test = 6000 / 1000 / 1000
model = h96 / depth4 / basis24
epochs = 30
seeds = 0,1,2,3,4
batch_size = 256
eval_batch_size = 512
audit_batch_size = 256
```

### 7.3 方法

主 confirm 只跑以下方法：

```text
AdamW-alphaLearn
GA-FU-v2-bothcos-alphaLearn
F-V3-HARD-base30-alphaLearn

AdamW-alphaFixed1
GA-FU-v2-bothcos-alphaFixed1
F-V3-HARD-base30-alphaFixed1
F-V3-HARD-c006-softcoeff-alphaFixed1
```

是否跑 `F-V3-HARD-c006-softcoeff-alphaLearn` 取决于资源。如果资源足够，可以作为 AUC profile 加入；如果资源有限，先不跑，因为它在 learnable-alpha 3-seed 中 J reduction 较弱。

`F-V3-FULL-f085` 暂时不进 P2 主 confirm，因为它 Fashion 3-seed accuracy 低于 hard-base，作用主要是 full-from-start 对照。若需要保留一个 full-from-start 对照，可以只跑 seeds `0,1,2`，不扩 5。

### 7.4 记录指标

任务性能：

```text
train_acc_final
val_acc_final
test_acc_final
best_val_acc
best_val_epoch
train_loss_final
val_loss_final
test_loss_final
```

收敛：

```text
val_loss_curve
val_loss_auc
val_auc_improvement_vs_adamw
val_loss_auc_first_half
val_loss_auc_second_half
loss_at_10pct_steps
loss_at_25pct_steps
loss_at_50pct_steps
loss_at_75pct_steps
```

几何：

```text
phi_prime_p95
phi_prime_max
phi_prime_reduction_vs_adamw
jac_condition_max
jac_reduction_vs_adamw
curvature_energy
credit_amplification_p95
```

branch 表达：

```text
branch_output_norm_ratio_train_mean
branch_output_norm_ratio_final
branch_over_adamw
branch_layer_0_final
branch_layer_1_final
branch_layer_2_final
branch_layer_3_final
no_kan_test_acc
no_kan_acc_drop
no_kan_drop_over_adamw
kan_logit_delta_norm
kan_margin_contribution
```

transition：

```text
switch_step
switch_epoch
switch_reason
transition_loss_jump
phase_final
branch_scale_curve
coeff_lr_curve
metric_mode_curve
```

compute：

```text
step_time_ms
epoch_time_sec
total_train_time_sec
precond_solve_time_ms
metric_build_time_ms
memory_peak_allocated_mb
```

calibration：

```text
ece
nll
confidence_correct_mean
confidence_wrong_mean
ece_reduction_vs_adamw
```

### 7.5 可视化

P2 必须生成以下图。

**Fashion scorecard**：

```text
x = method
y = test_acc_mean, error = std
旁边标注 AUC improve / phi red / J red / ECE red
```

**Validation loss trajectory**：

```text
x = epoch
y = val_loss
line = seed mean
band = seed std
vertical line = switch epoch mean for hard methods
```

**AUC vs geometry Pareto**：

```text
x = val_loss_auc
y = test_acc
color = method
size = J_reduction
```

**Branch expression panel**：

```text
plot 1: branch_over_adamw by method
plot 2: no_kan_drop by method
plot 3: branch_ratio vs no_kan_drop scatter
plot 4: branch_ratio vs test_acc scatter
```

**Transition behavior**：

```text
branch_scale over epoch
coeff_lr_multiplier over epoch
transition_loss_jump bar
switch_reason count
```

**Calibration**：

```text
ECE bar by method
confidence histogram correct vs wrong
reliability diagram optional
```

### 7.6 Fashion decision rule

Fashion clean default 首选规则：

```text
首选 F-V3-HARD-base30-alphaLearn，除非 fixed1 在 5 seeds 上同时满足更高 accuracy、更高 AUC、同等 geometry。
```

具体 gate：

$$
\operatorname{AccGap}_{vs AdamW}<0.003
\quad\text{or}\quad
\operatorname{Acc}_{method}\geq\operatorname{Acc}_{AdamW}.
$$

$$
\operatorname{AUCImprove}>0.10.
$$

$$
\operatorname{PhiRed}>0.30.
$$

$$
\operatorname{JRed}>0.80.
$$

$$
0.55<\operatorname{branch/AdamW}<0.75.
$$

$$
\operatorname{ECERed}>0.15.
$$

如果 fixed-alpha `F-V3-HARD-c006-softcoeff` AUC 最好但 accuracy 或 branch/no-KAN 弱，则它定位为：

```text
Fashion fixed-alpha AUC profile
```

不是 clean default。

---

## 8. V3.2-P3：KMNIST 5-seed clean confirm

### 8.1 目标

P3 是 KMNIST 的正式 confirm。它要回答三个问题：

```text
1. learnable-alpha full-from-start 是否稳定优于 v2 tuned？
2. restcos07 和 bothcos07 哪个更适合作为默认？
3. fixed-alpha 是否需要 repair，还是直接降级为 ablation？
```

### 8.2 设置

```text
dataset = KMNIST
train / val / test = 6000 / 1000 / 1000
model = h96 / depth4 / basis24
epochs = 20
seeds = 0,1,2,3,4
batch_size = 256
eval_batch_size = 512
audit_batch_size = 256
```

### 8.3 方法

learnable-alpha 主线：

```text
AdamW-alphaLearn
GA-FU-v2-tuned-alphaLearn
K-V3-FULL-restcos07-alphaLearn
K-V3-FULL-bothcos07-alphaLearn
```

fixed-alpha 判断线：

```text
AdamW-alphaFixed1
GA-FU-v2-tuned-alphaFixed1
K-V3-FULL-base-alphaFixed1
```

fixed-alpha 的 `restcos07 / bothcos07` 在上一轮 3-seed acc gap 较大，不建议直接扩 5 seeds。除非 P1 seed0 显示修复后的代码有明显变化，否则它们进入 P9 repair，而不是 P3 confirm。

### 8.4 记录指标

与 P2 相同，但 KMNIST 要额外强调 seed stability：

```text
test_acc_std
paired_acc_delta_vs_adamw
paired_auc_delta_vs_adamw
paired_phi_delta_vs_adamw
paired_j_delta_vs_adamw
bootstrap_ci_95_acc_delta
bootstrap_ci_95_auc_delta
```

因为 KMNIST 的 std 通常比 Fashion 大，单纯 mean 不够。

### 8.5 可视化

**KMNIST scorecard**：

```text
method | acc | acc gap | AUC improve | phi red | J red | ECE red | branch/A | pass level
```

**Paired seed plot**：

```text
x = seed
y = test_acc_method - test_acc_AdamW
line = method
```

**AUC/J tradeoff**：

```text
x = AUC improve
y = J reduction
color = method
size = test_acc
```

**restcos07 vs bothcos07 comparison**：

```text
bar 1: test acc mean ± std
bar 2: AUC improve
bar 3: J reduction
bar 4: ECE reduction
```

### 8.6 KMNIST decision rule

如果 `K-V3-FULL-restcos07-alphaLearn` 满足：

$$
\operatorname{AccGap}_{vs AdamW}<0.005,
$$

$$
\operatorname{AUCImprove}>0.09,
$$

$$
\operatorname{PhiRed}>0.25,
$$

$$
\operatorname{JRed}>0.90,
$$

$$
\operatorname{ECERed}>0.15,
$$

则它成为 KMNIST clean default。

如果 `bothcos07` 的 AUC/J 显著更好，但 acc 略低于 `restcos07`，则：

```text
KMNIST default = restcos07
KMNIST convergence/geometry profile = bothcos07
```

如果两个 learnable-alpha 候选在 5 seeds 上都不满足 acc gap，但 AUC/J/ECE 很强，则 v3.1 定位为：

```text
KMNIST geometry/convergence Pareto optimizer
```

并进入 P9 或模型容量调优，而不是继续调 transition。

---

## 9. V3.2-P4：alpha-mode decision

### 9.1 目标

P4 不跑新实验，除非 P2/P3 显示 alpha 模式结果矛盾。它是一个决策分析包，用于回答：

```text
Stage I /后续代码默认是否使用 learnable-alpha 还是 fixed-alpha？
```

### 9.2 分析方式

对 Fashion 和 KMNIST 分别做以下比较。

#### AdamW baseline shift

记录：

$$
\Delta Acc_{AdamW}^{fixed-learn}
=
Acc_{AdamW,fixed1}-\Acc_{AdamW,learnable}.
$$

如果 fixed-alpha 让 AdamW baseline 明显升高，但 v3 没同步升高，就说明 fixed-alpha 改变了问题难度和相对优势。

#### v3 相对优势

记录：

$$
\Delta Acc_{v3-AdamW}
=
Acc_{v3}-Acc_{AdamW}.
$$

$$
\Delta AUC_{v3-AdamW}
=
\frac{AUC_{AdamW}-AUC_{v3}}{AUC_{AdamW}+\epsilon}.
$$

#### geometry preservation

比较：

$$
\operatorname{PhiRed}_{learnable}
\quad\text{vs}\quad
\operatorname{PhiRed}_{fixed1}.
$$

$$
\operatorname{JRed}_{learnable}
\quad\text{vs}\quad
\operatorname{JRed}_{fixed1}.
$$

### 9.3 决策规则

如果 fixed-alpha 在 Fashion 和 KMNIST 都满足：

$$
\operatorname{AccGap}_{vs AdamW}<0.005,
$$

$$
\operatorname{AUCImprove}>0.09,
$$

$$
\operatorname{PhiRed}>0.25,
$$

$$
\operatorname{JRed}>0.20,
$$

则 fixed-alpha 可以成为新默认架构。

如果 fixed-alpha 只在 Fashion 强，但 KMNIST acc gap 明显，则：

```text
默认架构 = learnable-alpha
fixed-alpha = Fashion / geometry ablation
```

如果 learnable-alpha 在 KMNIST 明显更稳，而 Fashion 两者都强，则优先保留 learnable-alpha，因为它跨数据集更干净。

---

## 10. V3.2-P5：10-seed final confirm

### 10.1 触发条件

只有 P2/P3 至少一个候选通过 5-seed medium gate，才进入 P5。

### 10.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model = h96 / depth4 / basis24
Fashion epochs = 30
KMNIST epochs = 20
```

### 10.3 方法

最多保留 4 个方法：

```text
AdamW-final-alphaModeSelected
GA-FU-v2-final-alphaModeSelected
Fashion: F-V3-HARD-base30-final
KMNIST: K-V3-FULL-selected-final
```

如果 alpha-mode selected 是 learnable，则不再跑 fixed-alpha；如果 fixed-alpha 成为新默认，才反过来不跑 learnable。P5 不是比较 alpha 的阶段。

### 10.4 统计

必须输出：

```text
mean
std
median
min
max
95% bootstrap CI
paired seed delta vs AdamW
paired seed delta vs v2
```

对每个关键指标都做 paired delta：

$$
\Delta_s^{metric}=Metric_{v3,s}-Metric_{baseline,s}.
$$

报告：

$$
\bar\Delta,
\quad
\operatorname{std}(\Delta),
\quad
CI_{95\%}(\Delta).
$$

### 10.5 通过条件

P5 通过后才能写：

```text
v3.2 true-Gram GA-FU is the DG-KAN default optimizer for Fashion/KMNIST small vision.
```

如果只有 Fashion 通过，写：

```text
v3.2 true-Gram GA-FU is confirmed on Fashion-MNIST; KMNIST remains a geometry/convergence candidate.
```

如果 KMNIST AUC/J/ECE 强但 acc gap 不够，不能叫 clean default，只能叫 Pareto optimizer。

---

## 11. V3.2-P6：branch expression and no-KAN mechanism audit

### 11.1 目标

P6 解释 v3.1/v3.2 是否真的释放了 DG-KAN 的表达力。

当前 branch/A 是健康的，但这只是 norm proxy。P6 要证明：

```text
KAN branch 不只是变大，而是真的贡献分类决策。
```

### 11.2 设置

使用 P5 或 P2/P3 选出的候选，不新增方法。

```text
methods = AdamW, v2, v3-selected
seeds = P2/P3 已有 seeds
checkpoints = final + best_val + switch_epoch optional
```

### 11.3 指标

#### No-KAN drop

$$
\Delta_{noKAN}
=
Acc_{full}-Acc_{disableKAN}.
$$

记录：

```text
no_kan_val_acc
no_kan_test_acc
no_kan_acc_drop
no_kan_loss_increase
no_kan_drop_over_adamw
```

#### KAN logit contribution

定义 full logits：

$$
z_{full}=f(x).
$$

disable KAN logits：

$$
z_{noKAN}=f_{disableKAN}(x).
$$

KAN logit delta：

$$
\Delta z_{KAN}=z_{full}-z_{noKAN}.
$$

记录：

```text
kan_logit_delta_norm_mean
kan_logit_delta_norm_p95
kan_correct_class_delta_mean
kan_margin_contribution_mean
kan_margin_contribution_p95
```

Margin contribution 定义：

$$
\Delta margin_{KAN}
=
margin(z_{full})-margin(z_{noKAN}).
$$

其中：

$$
margin(z)=z_y-\max_{c\ne y}z_c.
$$

#### Per-layer branch contribution

记录：

```text
branch_layer_k_ratio_final
branch_layer_k_ratio_train_mean
branch_layer_k_no_kan_drop_if_ablatable_optional
branch_layer_k_phi_prime_p95
branch_layer_k_curvature
```

### 11.4 可视化

**Branch/no-KAN scatter**：

```text
x = branch_over_adamw
y = no_kan_drop_over_adamw
color = method
```

健康区域是：

$$
0.55<\operatorname{branch/AdamW}<0.75
$$

且：

$$
\Delta_{noKAN}^{v3}>0.7\Delta_{noKAN}^{AdamW}.
$$

**KAN margin contribution histogram**：

```text
x = Delta margin_KAN
split by correct / incorrect samples
```

如果 KAN contribution 主要出现在错误样本或负 margin，需要进一步诊断。

**Per-layer branch heatmap**：

```text
x = layer
y = method
value = final branch ratio
```

### 11.5 机制成功标准

P6 通过要求：

$$
\Delta_{noKAN}^{v3}>0.7\Delta_{noKAN}^{AdamW}.
$$

$$
\operatorname{branch/AdamW}_{final}>0.55.
$$

$$
\operatorname{KANMarginContribution}_{mean}>0.
$$

同时：

$$
\operatorname{PhiRed}>0.25,
\quad
\operatorname{JRed}>0.20.
$$

如果 branch/A 合理但 no-KAN drop 很小，则 v3.2 不能 claim high expression；只能 claim geometry/convergence improvement。

---

## 12. V3.2-P7：true-Gram direction audit

### 12.1 目标

P7 解释 true Gram 为什么有效。当前我们知道 true Gram 有用，但还不知道它的收益来自：

```text
1. 更好的方向；
2. 更合适的步长；
3. 更强的正则；
4. seed 偶然；
5. 与 rest/head schedule 的交互。
```

P7 要把这些拆开。

### 12.2 Direction definitions

同一个 coefficient gradient $g$ 下，定义：

$$
d_{raw}=-g.
$$

$$
d_{diag-fast}=-\operatorname{diag}(M_{fast}+\rho I)^{-1}g.
$$

$$
d_{full-geo}=-(M_{geo}+\rho I)^{-1}g.
$$

transition 中当前方向：

$$
d_{current}=(1-\lambda)d_{diag-fast}+\lambda d_{full-geo}.
$$

### 12.3 必须记录

```text
cos_raw_diagfast
cos_raw_fullgeo
cos_diagfast_fullgeo
cos_current_fullgeo
norm_raw
norm_diagfast
norm_fullgeo
norm_current
norm_fullgeo_over_diagfast
metric_norm_diagfast
metric_norm_fullgeo
metric_norm_current
update_over_coeff_norm
predicted_descent_proxy
actual_shadow_descent_optional
```

`predicted_descent_proxy` 可定义为：

$$
\operatorname{PredDesc}(d)=g^\top d.
$$

因为 $d$ 是 descent direction，健康情况下应该：

$$
g^\top d<0.
$$

### 12.4 Norm-matched control

如果发现 full Gram 的 `norm_fullgeo_over_diagfast` 明显偏小或偏大，需要做 norm-matched control：

$$
\tilde d_{diag}=d_{diag-fast}\cdot \frac{\|d_{full-geo}\|}{\|d_{diag-fast}\|+\epsilon}.
$$

然后比较：

```text
basis_diag_normmatched
full_sobolev_gram
```

如果 full Gram 仍然 AUC/J 更好，说明 direction 真的重要；否则收益可能主要来自 step norm。

### 12.5 可视化

**Direction cosine heatmap**：

```text
x = phase / epoch
y = direction pair
value = cosine
```

**Norm ratio over training**：

```text
x = epoch
y = norm_fullgeo_over_diagfast
line = method
```

**Direction vs outcome scatter**：

```text
x = mean cos_diagfast_fullgeo
y = AUC improvement
size = J reduction
color = dataset
```

**Metric condition dashboard**：

```text
bar: eig_min, eig_max, condition by dataset/method
line: update_metric_norm over epoch
```

### 12.6 解释目标

P7 最后要能回答：

```text
Fashion 为什么 hard diag-to-full 更好？
KMNIST 为什么 full-from-start 更好？
```

当前工作假设是：

$$
\boxed{
\text{Fashion 需要 early diag 释放表达力，再切 full Gram 收几何。}
}
$$

$$
\boxed{
\text{KMNIST 更需要从一开始就有稳定的 function-space direction。}
}
$$

P7 要用 direction cosine、norm ratio、transition loss jump 和 no-KAN drop 去支持或推翻这个解释。

---

## 13. V3.2-P8：target-matched and wall-clock convergence

### 13.1 目标

当前 AUC improvement 很强，但不能直接等价于 “训练更快”。因为 true Gram 有 Cholesky solve 和额外计算，step time 可能比 AdamW 慢。P8 要判断能否 claim：

```text
faster convergence
```

还是只能 claim：

```text
better validation-loss trajectory
```

### 13.2 Target definitions

对每个 dataset 和 seed，AdamW final val loss：

$$
L^*_{A,s}=L_{val,A,s}^{final}.
$$

AdamW final val accuracy：

$$
Acc^*_{A,s}=Acc_{val,A,s}^{final}.
$$

定义 relaxed loss target：

$$
L^{relaxed}_s=1.05L^*_{A,s}.
$$

定义 relaxed acc target：

$$
Acc^{relaxed}_s=Acc^*_{A,s}-0.005.
$$

记录每个 method 首次达到这些 target 的 step 和 wall-clock。

### 13.3 指标

```text
target/reached_adamw_final_loss
target/steps_to_adamw_final_loss
target/time_to_adamw_final_loss_sec
target/reached_relaxed_loss
target/steps_to_relaxed_loss
target/time_to_relaxed_loss_sec
target/reached_adamw_final_acc
target/steps_to_adamw_final_acc
target/time_to_adamw_final_acc_sec
target/reached_relaxed_acc
target/steps_to_relaxed_acc
target/time_to_relaxed_acc_sec
```

Compute accounting：

```text
step_time_ms_mean
step_time_ms_p50
step_time_ms_p95
epoch_time_sec
total_train_time_sec
precond_solve_time_ms_mean
metric_build_time_ms_total
samples_per_sec
memory_peak_allocated_mb
```

### 13.4 可视化

**Target reach curve**：

```text
x = step or wall-clock time
y = fraction of seeds that reached target
line = method
```

**Steps-to-target paired bars**：

```text
x = method
y = steps_to_relaxed_loss
paired by seed
```

**Wall-clock Pareto plot**：

```text
x = total_train_time_sec
y = final test_acc
size = AUC improvement
color = method
```

**AUC vs time-to-target scatter**：

```text
x = val_loss_auc
y = time_to_relaxed_loss
color = method
```

### 13.5 判定

Fast convergence claim 成立需要：

$$
\operatorname{time\_to\_target}_{v3}
<
\operatorname{time\_to\_target}_{AdamW}
$$

或至少：

$$
\operatorname{steps\_to\_target}_{v3}
<
\operatorname{steps\_to\_target}_{AdamW}
$$

且：

$$
\operatorname{step\_time\_ratio}_{v3/AdamW}<1.2.
$$

如果 AUC 好但 time-to-target 不好，结论写成：

```text
v3 improves validation-loss trajectory and calibration under matched steps, but does not yet prove wall-clock faster convergence.
```

---

## 14. V3.2-P9：fixed-alpha KMNIST repair, conditional only

### 14.1 触发条件

只有当你决定当前代码必须以 fixed-alpha 为默认架构时，才运行 P9。否则 P9 暂停。

触发条件：

```text
alpha_mode=fixed1 将作为后续 scaling / Rational / CIFAR 的默认代码状态。
```

如果后续愿意恢复 learnable-alpha，则 P9 不需要。

### 14.2 目标

fixed-alpha KMNIST 目前的问题是：

```text
AUC / phi / J 好，但 AdamW baseline 变强，v3 accuracy 没过 AdamW。
```

P9 只做很小的 repair sweep，目标不是找全局最优，而是找一个 fixed-alpha 下不输 AdamW 的 full-Gram profile。

### 14.3 设置

```text
dataset = KMNIST
alpha_mode = fixed1
model = h96 / depth4 / basis24
epochs = 20
seeds = 0,1,2
```

### 14.4 Sweep matrix

固定：

```text
metric = full_sobolev_gram from start
branch_boost = 1.2
geometry_min_epochs = 0
v3_alpha_geo = 0.15
v3_beta_geo = 0.02
```

扫：

```text
coeff_lr in {0.12, 0.14, 0.16, 0.18}
branch_final_scale in {0.8, 0.85, 0.9}
rest_schedule in {none, restcos07}
coeff_schedule in {none, coeffcos07 only if needed}
```

为了控制规模，先做 seed0：

```text
4 * 3 * 2 = 24 runs + AdamW/v2 baselines
```

选 top 3 后扩 seed0/1/2。

### 14.5 选择指标

固定-alpha KMNIST repair 的第一目标是 accuracy 不输，第二目标才是 AUC。

选择 top candidate 的 score：

$$
Score
=
2.0\cdot AccGain
+1.0\cdot AUCImprove
+0.5\cdot PhiRed
+0.5\cdot JRed
+0.3\cdot ECERed
-0.5\cdot BranchPenalty.
$$

其中：

$$
BranchPenalty=\max(0,0.55-r)+\max(0,r-0.75).
$$

### 14.6 成功标准

P9 成功要求：

$$
\operatorname{AccGap}_{vs AdamW}<0.005,
$$

$$
\operatorname{AUCImprove}>0.08,
$$

$$
\operatorname{PhiRed}>0.25,
$$

$$
\operatorname{JRed}>0.20.
$$

如果 P9 找不到，则 fixed-alpha 不能作为 KMNIST clean default。

---

## 15. V3.2-P10：smooth transition revisit, conditional only

### 15.1 触发条件

P10 只有在下面任一情况出现时才运行：

```text
1. Fashion hard switch 在 P2 5-seed 中出现明显 transition_loss_jump；
2. Fashion hard 的 J reduction 不稳定；
3. KMNIST full-from-start 5-seed accuracy 不稳，需要更温和 transition；
4. direction audit 显示 hard switch 前后 direction cosine 断裂很大。
```

### 15.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
alpha_mode = selected alpha mode from P4
```

### 15.3 方法

Fashion：

```text
F-V3-HARD-base30
F-V3-SMOOTH-fixed-transition-tr005
F-V3-SMOOTH-fixed-transition-tr010
F-V3-SMOOTH-fixed-transition-longtr
```

KMNIST：

```text
K-V3-FULL-selected
K-V3-DIAG-TO-FULL-SMOOTH-tr010
K-V3-DIAG-TO-FULL-SMOOTH-longtr
```

### 15.4 记录额外指标

```text
transition_length_steps
metric_mix_curve
transition_loss_jump
transition_acc_jump
cos_current_fullgeo_before_after_switch
branch_scale_slope
coeff_lr_slope
```

### 15.5 判定

Smooth 只有在满足以下条件时才有资格替代 hard/full：

$$
\operatorname{AUCImprove}_{smooth}>
\operatorname{AUCImprove}_{current}+0.01,
$$

$$
\operatorname{Acc}_{smooth}\geq
\operatorname{Acc}_{current}-0.003,
$$

$$
\operatorname{JRed}_{smooth}\geq
\operatorname{JRed}_{current}-0.10.
$$

否则 smooth 继续保留为 ablation。

---

## 16. V3.2-P11：Rational / KAT scaling precheck, after default freeze

### 16.1 为什么现在不能立刻切 Rational

Rational / KAT 的参数和计算效率更好，但它改变了 DG-KAN 的 primitive。RBF-DGKAN 是 edge-wise function：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t).
$$

Rational / KAT 更像：

```text
group rational activation -> Linear
```

它不再是每条 edge 一个独立函数，而是 group-wise activation function。它可能更适合 scaling，但会改变 functional update、geometry audit 和 branch expression 的含义。

所以 v3.2 的原则是：

```text
先冻结 RBF v3.2 default；
再做 Rational scaling precheck；
不要用 Rational 逃避 optimizer clean confirm。
```

### 16.2 触发条件

只有 P5-P8 至少给出一个 clean default 或明确 Pareto optimizer 后，才运行 P11。

### 16.3 最小 precheck 设置

```text
datasets = Fashion-MNIST, KMNIST
model size = param-matched and width-matched
methods = MLP, RBF-DGKAN-v3.2, Rational-DGKAN-AdamW, Rational-DGKAN-GA-FU-compatible
seeds = 0,1,2
```

### 16.4 Rational 必须记录的额外指标

```text
rational/denominator_min
rational/denominator_p01
rational/r_prime_p95
rational/r_prime_max
rational/r_double_prime_p95
rational/group_function_diversity
rational/group_activation_norm
rational/group_saturation_rate
```

### 16.5 决策 gate

Rational 只有在满足以下条件时，才进入 Stage I scaling default：

$$
\operatorname{AccGap}_{Rational\ vs\ RBF}<0.01,
$$

$$
\operatorname{Memory}_{Rational}<0.7\operatorname{Memory}_{RBF},
$$

$$
\operatorname{StepTime}_{Rational}<0.8\operatorname{StepTime}_{RBF},
$$

$$
\operatorname{GeometryWorse}_{Rational}<0.20.
$$

否则，Rational 只作为 engineering ablation，不替代 RBF 主线。

---

## 17. V3.2-P12：CIFAR-small readiness

### 17.1 触发条件

P12 只有在 P5-P8 至少确认 Fashion/KMNIST 上的 v3.2 default 或 Pareto optimizer 后运行。

### 17.2 目标

P12 不追求 SOTA，只检查 v3.2 是否具备从灰度小图扩展到自然图像的基本条件。

### 17.3 设置

```text
dataset = CIFAR-10 small
train / val / test = configurable, e.g. 10000 / 2000 / 2000
model = ConvStem + DGKAN blocks
methods = AdamW, RBF-DGKAN-AdamW, RBF-DGKAN-v3.2
seeds = 0,1,2
```

### 17.4 必须记录

```text
accuracy
val_loss_auc
branch_over_adamw
no_kan_drop
phi_prime_p95
jac_condition
ECE
step_time
memory
```

### 17.5 通过条件

CIFAR-small readiness 不要求超过 AdamW，但要求：

$$
\operatorname{AccGap}_{v3}<0.03,
$$

$$
\operatorname{AUCImprove}>0,
$$

$$
\operatorname{PhiRed}>0.10,
$$

$$
\operatorname{branch/AdamW}>0.45.
$$

如果 branch under-active 或 memory 爆炸，则回到 Rational / grouped primitive，不继续堆 optimizer。

---

## 18. 全局指标规范

所有正式 run 必须记录以下指标。缺任何一类，不能进入最终报告。

### 18.1 Task metrics

```text
train_loss_final
val_loss_final
test_loss_final
train_acc_final
val_acc_final
test_acc_final
best_val_acc
best_val_epoch
```

### 18.2 Convergence metrics

```text
val_loss_auc
val_acc_auc
train_loss_auc
AUC_improvement_vs_adamw
AUC_improvement_vs_v2
loss_at_10pct_steps
loss_at_25pct_steps
loss_at_50pct_steps
loss_at_75pct_steps
steps_to_relaxed_loss
time_to_relaxed_loss
```

### 18.3 Geometry metrics

```text
phi_prime_p95
phi_prime_max
phi_prime_reduction_vs_adamw
jac_condition_mean
jac_condition_max
jac_reduction_vs_adamw
curvature_energy
credit_amplification_p95
```

### 18.4 Branch / expression metrics

```text
branch_output_norm_ratio_train_mean
branch_output_norm_ratio_final
branch_over_adamw
branch_layer_k_ratio_final
no_kan_acc_drop
no_kan_drop_over_adamw
kan_logit_delta_norm
kan_margin_contribution
active_basis_fraction
dead_basis_fraction
basis_occupancy_entropy
```

### 18.5 v3 metric / direction metrics

```text
v3_metric_condition_active
v3_metric_condition_geometry
v3_metric_eig_min_active
v3_metric_eig_max_active
v3_metric_eig_min_geometry
v3_metric_eig_max_geometry
cos_diagfast_fullgeo
norm_fullgeo_over_diagfast
update_metric_norm
update_over_coeff_norm
precond_direction_norm
raw_grad_norm
```

### 18.6 Schedule metrics

```text
phase_final
switch_step
switch_epoch
switch_reason
active_phase_fraction_actual
transition_length_steps
metric_mix_auc
branch_scale_curve
coeff_lr_multiplier_curve
transition_loss_jump
shrink_events
```

### 18.7 Trust / numerical metrics

```text
trust_clip_count
trust_clip_rate
nan_or_inf_count
bad_step_count
metric_cholesky_success
metric_build_time_ms
precond_solve_time_ms
```

### 18.8 Calibration metrics

```text
ece
nll
confidence_correct_mean
confidence_wrong_mean
ece_reduction_vs_adamw
```

### 18.9 Compute / memory metrics

```text
step_time_ms_mean
step_time_ms_p95
epoch_time_sec
total_train_time_sec
samples_per_sec
memory_peak_allocated_mb
memory_reserved_mb
```

---

## 19. 全局可视化规范

每个实验包至少输出以下图。图可以通过 W&B dashboard 或本地 `plots/` 生成。

### 19.1 Scorecard table

列：

```text
dataset
method
alpha_mode
test_acc_mean ± std
acc_gap_vs_adamw
AUC_improve
phi_red
J_red
branch/A
no_kan_drop
ECE_red
step_time_ratio
pass_level
```

### 19.2 Loss curves

```text
x = epoch
y = val_loss
line = method mean
band = std
vertical line = switch epoch
```

### 19.3 Accuracy curves

```text
x = epoch
y = val_acc
line = method mean
band = std
```

### 19.4 Pareto plots

```text
x = val_loss_auc
y = test_acc
size = J_reduction
color = method
```

```text
x = phi_prime_p95
y = test_acc
size = branch/A
color = method
```

```text
x = step_time_ms
y = test_acc
size = AUC_improvement
color = method
```

### 19.5 Branch mechanism plots

```text
branch/A over epoch
no_kan_drop over epoch
branch/A vs no_kan_drop scatter
branch/A vs phi_prime_p95 scatter
per-layer branch heatmap
```

### 19.6 Direction audit plots

```text
cos_diagfast_fullgeo over epoch
norm_fullgeo_over_diagfast over epoch
metric_condition by method
update_metric_norm over epoch
```

### 19.7 Alpha-mode comparison plots

```text
AdamW acc by alpha_mode
v3 acc by alpha_mode
AUC improvement by alpha_mode
branch/A by alpha_mode
J reduction by alpha_mode
```

### 19.8 Failure plots

```text
failure type counts by method
failure type counts by dataset
accuracy_gap vs branch_underactive scatter
geometry_failure vs branch_overactive scatter
```

---

## 20. Failure table 规则

每个 run 都要打 failure tags。一个 run 可以有多个 tag。

| failure type | 触发条件 | 含义 |
|---|---|---|
| `accuracy_gap_too_large` | acc gap vs AdamW > threshold | 精度不够 |
| `no_convergence_gain` | AUC improvement <= 0 | 没有收敛轨迹收益 |
| `weak_convergence_gain` | AUC improvement < target | AUC 不够 |
| `branch_underactive` | branch/A < 0.45 | KAN branch 没用起来 |
| `branch_overactive` | branch/A > 0.95 | branch 过强 |
| `no_kan_no_contribution` | no-KAN drop < 0.5 AdamW | branch 没有任务贡献 |
| `geometry_not_preserved` | phi/J reduction 不达标 | 几何优势不足 |
| `jacobian_spike` | J cond > 1.2 AdamW | Jacobian 失控 |
| `phi_prime_spike` | phi p95 > 1.2 AdamW | derivative 失控 |
| `transition_jump` | transition_loss_jump > 0.02 | switch 有损失跳变 |
| `trust_not_safety` | trust_clip_rate > 0.05 | trust 实质参与优化 |
| `wall_clock_no_gain` | time-to-target 不优 | 不能 claim 快收敛 |
| `seed_unstable` | acc std > 0.008 | seed 不稳定 |
| `alpha_mode_confounded` | alpha metadata missing | alpha 结果不可解释 |
| `metric_condition_high` | condition > 1e4 | Gram 数值风险 |

Failure aggregation 输出：

```text
failure_table.csv
failure_summary_by_method.csv
failure_summary_by_dataset.csv
failure_summary_by_alpha_mode.csv
```

---

## 21. 最终决策规则

### 21.1 v3.2 clean default

如果 Fashion 和 KMNIST 都通过 5-seed medium，并且 P6/P8 没有否定表达力和 wall-clock 结论，则：

```text
DG-KAN clean default optimizer = GA-FU-v3.2 true-Gram
```

论文/报告中可以写：

```text
true shared Sobolev Gram improves functional update trajectory while preserving branch utilization and geometry.
```

### 21.2 dataset-specific default

如果 Fashion 强通过，KMNIST 只达到 Pareto/medium 边缘，则：

```text
Fashion default = F-V3-HARD-base30
KMNIST default candidate = K-V3-FULL-restcos07 or bothcos07
```

结论写成：

```text
v3.2 is confirmed as a dataset-specific true-Gram optimizer candidate, with Fashion clean-default evidence and KMNIST convergence/geometry evidence.
```

### 21.3 v3.2 Pareto optimizer

如果 accuracy 没完全过，但 AUC/J/ECE 很强，则：

```text
v3.2 = geometry/convergence Pareto optimizer
AdamW = accuracy baseline
```

不能写 clean default。

### 21.4 回到模型容量，而不是继续调 optimizer

如果 P2/P3 显示：

```text
branch/A 合理
no-KAN drop 合理
geometry 合理
但 accuracy 仍差
```

则问题可能不是 optimizer，而是模型容量 / primitive 参数效率。下一步应转到：

```text
hidden_dim / basis_count / depth
Rational / grouped primitive
ConvStem / CIFAR architecture
```

而不是继续调 branch_scale 或 transition。

### 21.5 fixed-alpha 决策

如果 fixed-alpha 只在 Fashion 好，在 KMNIST 不好，则：

```text
alpha_mode = learnable remains default
fixed-alpha = Fashion / geometry ablation
```

如果 fixed-alpha 在 P9 后 KMNIST 也过，则：

```text
alpha_mode = fixed1 can be considered for scaling due to simpler geometry/accounting
```

---

## 22. 推荐执行顺序

最小推荐执行顺序如下。

第一步，修代码并跑 P0/P1：

```text
P0: alpha_mode + transition endpoint + smoke
P1: seed0 sanity, both alpha modes
```

第二步，直接确认当前最强候选：

```text
P2 Fashion 5-seed:
  F-V3-HARD-base30-alphaLearn
  F-V3-HARD-base30-alphaFixed1
  F-V3-HARD-c006-softcoeff-alphaFixed1
  baselines

P3 KMNIST 5-seed:
  K-V3-FULL-restcos07-alphaLearn
  K-V3-FULL-bothcos07-alphaLearn
  K-V3-FULL-base-alphaFixed1
  baselines
```

第三步，做 P4 alpha 决策。如果 fixed-alpha 不是必须默认，就不要跑 P9。

第四步，对选出的候选做 P6/P7/P8：

```text
P6 branch/no-KAN 表达力审计
P7 direction audit
P8 target-matched / wall-clock convergence
```

第五步，若 5-seed 通过，进入 P5 10-seed final confirm。

第六步，只有 clean default 冻结后，才运行：

```text
P11 Rational / KAT scaling precheck
P12 CIFAR-small readiness
```

---

## 23. 当前我会立即执行的最小实验清单

如果只允许跑最小必要实验，我建议按下面顺序执行。

### 23.1 必跑 A：alpha_mode implementation + smoke

```text
alpha_mode = learnable, fixed1
datasets = Fashion-MNIST, KMNIST
methods = AdamW, GA-FU-v3-smoke
seeds = 0
epochs = 1
```

目标：确保代码切换干净。

### 23.2 必跑 B：Fashion 5-seed

```text
AdamW-alphaLearn
v2-bothcos-alphaLearn
F-V3-HARD-base30-alphaLearn
AdamW-alphaFixed1
v2-bothcos-alphaFixed1
F-V3-HARD-base30-alphaFixed1
F-V3-HARD-c006-softcoeff-alphaFixed1
```

目标：确认 Fashion clean default。

### 23.3 必跑 C：KMNIST 5-seed

```text
AdamW-alphaLearn
v2-tuned-alphaLearn
K-V3-FULL-restcos07-alphaLearn
K-V3-FULL-bothcos07-alphaLearn
AdamW-alphaFixed1
K-V3-FULL-base-alphaFixed1
```

目标：确认 KMNIST learnable-alpha default，并决定 fixed-alpha 是否需要 repair。

### 23.4 必跑 D：机制审计

只对最终候选跑：

```text
no-KAN drop
KAN logit delta
KAN margin contribution
per-layer branch
true-Gram direction cosine
target-matched convergence
```

目标：把“效果好”解释成“为什么好”。

---

## 24. 预期结果与可能分叉

### 24.1 最理想结果

```text
Fashion F-V3-HARD-base30 5-seed strong。
KMNIST K-V3-FULL-restcos07 5-seed medium，bothcos07 convergence profile strong。
learnable-alpha 跨数据集更稳。
P6 no-KAN drop 证明 branch 有任务贡献。
P8 steps-to-target 优于 AdamW，wall-clock 持平或略差但可接受。
```

结论：v3.2 成为 DG-KAN clean default。

### 24.2 次优但很有价值

```text
Fashion strong。
KMNIST AUC/J/ECE strong，但 acc gap 仍略大。
P6 branch 有任务贡献。
P8 wall-clock 不一定更快。
```

结论：v3.2 是 Fashion clean default + KMNIST Pareto optimizer。下一步不是调 optimizer，而是看 model capacity / Rational / ConvStem。

### 24.3 需要回退的情况

```text
5-seed 后 Fashion hard-base J reduction 不稳。
KMNIST full-from-start seed variance 大。
no-KAN drop 很小。
```

此时不应直接进入 10 seeds。应先跑：

```text
P7 direction audit
P10 smooth transition revisit
P9 fixed-alpha repair if relevant
```

### 24.4 最坏情况

```text
true-Gram 3-seed 信号在 5 seeds 消失。
AUC/J/ECE 不稳定。
branch/no-KAN 不支持表达力。
```

结论：v3.1 只是 seed / small-run artifact。回到 v2 tuned 作为 stable baseline，并把 true Gram 降级为 ablation。

---

## 25. 本计划的最终产物

v3.2 完成后应该产出以下文件：

```text
results/gafu_v3_2_p0_smoke/
results/gafu_v3_2_p2_fashion_confirm/
results/gafu_v3_2_p3_kmnist_confirm/
results/gafu_v3_2_p6_mechanism/
results/gafu_v3_2_p7_direction_audit/
results/gafu_v3_2_p8_target_matched/
```

以及一份总结文档：

```text
DG-KAN_Optimizer_v3.2_结果复盘.md
```

其中必须包含：

```text
1. final selected method per dataset
2. alpha_mode decision
3. 5-seed / 10-seed scorecard
4. branch/no-KAN mechanism evidence
5. direction audit explanation
6. target-matched convergence result
7. failure table
8. next scaling decision: RBF continue / Rational precheck / CIFAR-small
```

---

## 26. 一句话结论

v3.2 的下一步不是继续探索更多 optimizer 组件，而是把当前 true-Gram 信号做成硬结论：

$$
\boxed{
\text{Fashion hard diag-to-full 与 KMNIST full-from-start 进入 5-seed clean confirm。}
}
$$

同时必须把 alpha 模式显式化，并补齐 no-KAN、direction audit、target-matched convergence。只有这些通过后，才应该进入 Rational / KAT 或 CIFAR-small scaling。
