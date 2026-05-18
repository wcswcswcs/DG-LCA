# DG-KAN v9.7.1 Exact Transfer Gate / Core Expansion / Direct Small Update 完整实验计划

> 本计划基于 v9.7.0 `双线验证 / ExistingAction 与 TransferPrinciple` 的真实结果制定。  
> v9.7.1 不继续小修 `R8A`、`TransferLCB`、`B3`、`C16G` 或 `APGU`。  
> 本轮要回答一个更底层的问题：**训练当下能不能用“跨样本有效”这个简单原则，稳定选出或直接求出一个 functional update 动作？**

---

## 0. 一句话目标

v9.7.1 的目标不是再堆一个复杂人工 score，而是验证一个更简单的命题：

$$
\boxed{
\text{一个额外参数改动，如果真的有用，应该能帮助没有参与生成它的样本。}
}
$$

更直白地说：

```text
如果一个动作只对提出它的那几个样本有用，
但对另外一批样本没用，
它很可能是在记噪声，不能作为 functional update。

如果一个动作对另一批样本也有帮助，
并且不伤旧知识、不制造 long-risk、计算不太慢，
它才值得进入 controller。
```

v9.7.1 因此采用双线并行：

```text
A 线：继续推进 existing-action route。
     目标是看旧 AP0 动作里那 77 个干净核心动作，能不能用更少人工设计的方法补到 87 个，
     同时解决 LDO，也就是 leave-dataset-out 稳定性。

B 线：验证 exact cross-sample transfer。
     目标是补齐 v9.7.0 缺失的 exact per-sample gradient / exact apply checkpoint，
     直接测 transfer principle 本身，而不是只测 proxy。
```

如果 B 线成立，再开 C 线：

```text
C 线：直接求一个 small transfer-optimal update。
     不再盲目生成 APG/APGU 小变体，
     而是在一个很小的 KAN 参数子空间里求能跨样本 transfer 的更新。
```

---

## 1. 对 v9.7.0 的独立判断

### 1.1 v9.7.0 有进展，但不是能力成功

v9.7.0 的 route 是：

```text
route = R4-TransferPrincipleFail
base_candidate = LQ-t2-h256
success_v9700_strict_purekan_functional = False
success_v9700_full_functional = False
success_v9700_external_ready = False
```

这不是 functional success。controller、selected runtime、paired replay、short/full training 都没有打开。

但 v9.7.0 不是空转。它完成了三件重要事情：

```text
1. 同时验证了 existing-action continuation 和 cross-sample transfer principle；
2. 落盘了完整 transfer diagnostic ledger：2876 个 AP0 actions 全部有 transfer proxy row；
3. 证明当前 landed proxy 版 transfer 不能替代旧 rank。
```

关键数据：

```text
P1 transfer ledger:
  AP0 action rows = 2876
  transfer rows = 2876
  missing required field = 0
  NaN/Inf = 0
  exact_per_sample_gradient_available = 0
```

这说明 v9.7.0 并不是没跑起来，而是诚实地证明：**当前只有 proxy，没有 exact transfer objective**。

---

### 1.2 旧 rank 仍然质量高，但 LDO 不稳

v9.7.0 的 best old rule 仍是：

```text
R0-raw-R8A-transported-value-rank
accepted = 87
GradeAB precision = 0.8735632183908046
V_integrated LCB = 0.14111334880346277
longrisk UCB = 0.0
bad/null/memory/offdiag UCB = 0 / 0 / 0 / 0
LDO drop = 0.37931034482758624
```

这组数字说明：

```text
旧 rank 能选到一批很好的动作；
这批动作不是 value 差，也不是 longrisk 高；
真正的问题是 leave-dataset-out 不稳。
```

所以 old rank route 还没有死，但它不能 official。

---

### 1.3 TransferLCB proxy 降低 LDO，但把动作质量打坏了

v9.7.0 的 best transfer rule：

```text
R7-TransferLCB-global-conformal-threshold
precision = 0.17525773195876287
V LCB = -0.21558700787971152
longrisk UCB = 0.10979531263193916
LDO drop = 0.07216494845360824
```

`R2-TransferLCB-only` 更直观：

```text
TopK87 precision = 0.09195402298850575
V LCB = -0.24289252733948727
longrisk UCB = 0.12221253967001162
LDO drop = 0.04597701149425287
```

这说明：

```text
TransferLCB proxy 确实更 group-stable；
但它选出来的动作质量很差。
```

所以 v9.7.0 不能说 “transfer principle 彻底失败”。更准确是：

$$
\boxed{
\text{当前 proxy transfer 失败；exact transfer 还没有被真正测到。}
}
$$

---

### 1.4 Core + Expansion 仍然卡在 coverage 与 LDO

Core-only 仍然是最干净的区域：

```text
B0-T3.1-CoreOnly:
  accepted = 77
  coverage = 0.026773296244784424
  GradeAB = 1.0
  V LCB = 0.16402807605399972
  risk/bad/null/memory/offdiag UCB = 0 / 0 / 0 / 0 / 0
  LDO = 0.39080459770114945
```

但 coverage 下限约为 87 个动作，所以 77 个不够。

transfer 扩展补到 87 后：

```text
B3-CorePlus10TransferLCB:
  accepted = 87
  coverage = 0.030250347705146036
  GradeAB = 0.8850574712643678
  V LCB = 0.11911876658700193
  risk/bad/null/memory/offdiag UCB = 0 / 0 / 0 / 0 / 0
  LDO = 0.39080459770114945
```

这说明 transfer expansion 能补到数量，也能保持风险干净，但没有解决 LDO，且 V LCB 比旧 expansion 低。

---

### 1.5 Dataset shift 没被 no-tuning transport 解决

v9.7.0 的 dataset diagnostic 显示：

```text
raw_score PSI mean = 0.31313509863308747
TransferLCB PSI mean = 0.4096172222597585
```

per-dataset raw score diagnostic：

| dataset | PSI | TopK87 count / precision | V LCB | longrisk UCB | core action count |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0.3477132255617457 | 34 / 1.0 | 0.2185522558920942 | 0.0 | 34 |
| KMNIST | 0.3472940184541398 | 36 / 0.7777777777777778 | 0.053201019604910284 | 0.0 | 28 |
| MNIST | 0.244398051883377 | 17 / 0.8235294117647058 | 0.05691790232751366 | 0.0 | 15 |

这里最重要的不是哪个数据集好，而是：

```text
不同 dataset 的 score 分布、core 动作数量、precision 都不同；
但 official controller 又不能按 dataset 写规则。
```

因此下一步需要的是 dataset-agnostic 的 score transport / split-invariant ranking，不是 dataset-specific tuning。

---

### 1.6 Exact microprobe 和 direct solved update 都还没真正打开

v9.7.0 的 P5：

```text
status = not_run
reason = exact_per_sample_apply_checkpoint_not_landed
exact_per_sample_gradient_available = 0
microprobe_pass = 0
```

P6：

```text
generated_action_count = 0
branch_horizon_rows_actual = 0
direct_solved_weak_pass = 0
direct_solved_strong_pass = 0
reason = new_exact_transfer_objective_not_available_from_landed_artifacts
```

所以 v9.7.0 没有真正验证：

$$
\Delta^* = \arg\max_{\Delta} LCB_{transfer}(\Delta).
$$

它只验证了 landed proxy 版 transfer。这个区别非常重要。

---

## 2. 当前真正卡在哪里

### 卡点 1：old rank 有质量，但跨数据集不稳

旧 rank 能找出 87 个质量很高的动作：precision 高、V 正、longrisk 低。但 LDO drop 是 `0.3793`。这说明它可能抓住了某些 dataset / score scale / action distribution 的局部规律，而不是通用规律。

### 卡点 2：proxy transfer 稳，但没质量

TransferLCB proxy 能降低 LDO drop，但 precision、V、longrisk 全面变差。它没有把“跨样本有效”变成好动作选择。

### 卡点 3：77 个 core 动作非常干净，但数量不够

Core-only 只有 77 个动作，低于 87 左右的最低 coverage 门槛。补 10 个动作是现在 existing-action route 的关键难点。

### 卡点 4：exact transfer 所需 artifact 没落地

没有 exact per-sample gradient，也没有 exact apply checkpoint，所以不能判断 transfer principle 本身。现在不能因为 proxy 失败就放弃 transfer，也不能因为 transfer 理论好听就打开 generated route。

### 卡点 5：generated route 没有新 objective，应继续停止

APGH/APGL/APGT 连续失败，APGU 停止是正确的。没有 exact transfer objective，就不应该生成新的 APG/APGU rows。

---

## 3. 为什么感觉非常慢

这个感觉是对的。最近几轮一直在排除假成功：

```text
强 rank signal -> LDO 不稳；
core target 很干净 -> 数量不够；
扩展 target 数量够 -> LDO 或 null/offdiag 出问题；
score transport 降低 LDO -> precision/value 崩；
TransferLCB 更稳 -> 质量崩；
generated primitive 工程能跑 -> value-negative / high-longrisk；
exact transfer 还没 artifact -> direct update 不能打开。
```

这不是没进度，但它不是“能力上涨式进度”。它是排雷式进度。

为了加快，v9.7.1 必须同时跑：

```text
1. exact transfer artifact 落地；
2. exact transfer selector；
3. core + expansion 最小补 10 动作；
4. dataset shift transport；
5. direct transfer-solved update；
6. generated-route stop/continue decision。
```

不能再一轮只确认一个 blocker。

---

## 4. v9.7.1 的核心思想

v9.7.1 不再把 score 设计得很复杂。主原则只有一个：

$$
\boxed{
\text{动作必须帮助没有参与生成它的样本。}
}
$$

给一个候选动作 $\Delta$，对一批检查样本 $B$，定义每个样本的线性化收益：

$$
r_i(\Delta)=-g_i^\top\Delta.
$$

如果做 exact apply，则定义：

$$
r_i^{exact}(\Delta)=CE_i(\theta)-CE_i(\theta+\epsilon\Delta).
$$

transfer 下界：

$$
LCB_{transfer}(\Delta)=\mu_B(\Delta)-z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}.
$$

最小接受条件：

$$
LCB_{transfer}(\Delta)>0.
$$

但是只靠这个还不够，所以加少数硬门，不加复杂权重：

$$
ApplyOK(\Delta)=1,
$$

$$
Size(\Delta)\le \epsilon,
$$

$$
MemoryHarm(\Delta)\le \tau_M,
$$

$$
LongRiskProxy(\Delta)\le \tau_L,
$$

$$
Cost(\Delta)\le C_{max}.
$$

注意：这些是 hard gate，不是人工加权 score。

---

# v9.7.1 实验计划

## 5. 总体目标

v9.7.1 的总目标是：

$$
\boxed{
\text{判断 exact cross-sample transfer 是否能成为 functional update 的第一性判据。}
}
$$

并行回答三个问题：

```text
Q1. Existing-action route：
    旧 AP0 动作里，能不能用 exact transfer 帮助 core 77 扩到 87，并降低 LDO？

Q2. Transfer principle route：
    exact transfer 是否比 proxy transfer 更接近 GradeAB / V / low longrisk？

Q3. Direct update route：
    如果 exact transfer 有效，能否直接在小 KAN 子空间里求出一个 transfer-positive update？
```

---

## 6. 硬约束

继续保持：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / auxiliary loss
no sampler / class weight
no fake / proxy rows official
no CPU offload official
KAN path 不使用 PyTorch loss.backward graph
functional update 是 update rule，不是 loss trick
official controller 不使用 dataset_name 分支
official controller 不使用 validation/test/future outcome
official controller 不使用 branch-horizon outcome-derived fields
old outcome table 不进入 official
```

允许：

```text
per-dataset diagnostics；
leave-dataset-out / leave-stratum-out / leave-template-out；
exact per-example gradient from current train batch / train-memory buffer；
exact apply microprobe on train-memory buffer；
small-subspace direct solve；
negative controls；
runtime preflight；
Base-Acc Sentinel；
```

禁止：

```text
按 dataset 写 threshold；
按 dataset 写 TopK；
把 proxy transfer 当 exact transfer；
把 direct solved update 的 formula estimate 当 branch-horizon outcome；
在 exact transfer 未落地时恢复 blind APGU；
```

---

## 7. 关键名词解释

### Existing action

已经在 canonical AP0 表里的旧动作。它们有完整历史 outcome，可以用来验证新选择规则。

### Core actions

当前最干净的一批动作。v9.7.0 里是 77 个：GradeAB = 1.0，V 正，longrisk/bad/null/memory/offdiag 都为 0，但数量不够。

### Expansion actions

为了达到 coverage 下限，需要从 core 外面再补的动作。现在至少要补约 10 个。

### Transfer

一个动作对没参与生成它的样本是否也有帮助。它是一个比“当前 batch loss 降不降”更接近泛化的判断。

### Exact transfer

直接用 per-example gradient 或 exact apply checkpoint 计算的 transfer，不是 proxy。

### Proxy transfer

v9.7.0 使用的 `loo_transfer_proxy/action_projection_signal/grad_mean_sq_group/grad_var_trace_group`。它可以诊断，但不能代表 exact transfer。

### Direct small update

不是从 APG/APGU 生成器里挑动作，而是在一个很小的参数子空间里直接求一个 transfer-positive 更新。

---

# 8. 实验阶段

---

## P0. Boundary reproduction / legality / no-fake audit

### 目标

确认 v9.7.0 边界被复现，且 v9.7.1 没有偷用 forbidden 字段。

### 假设

v9.7.1 只能在 v9.7.0 基础上新增 exact transfer artifact，不能绕过 v9.7.0 的 gate。

### 必须记录

```text
source_route_v9700
system_legal_controller_pass_v9700
generated_stop_v9700
field_green_count
field_yellow_count
field_red_count
uses_dataset_name_for_controller
uses_outcome_derived_field
uses_validation_or_test
fake_row_count
proxy_official_count
cpu_offload_used
manual_forward_pass
manual_backward_pass
manual_update_pass
```

### 判断标准

通过条件：

```text
red official field count = 0
fake/proxy official rows = 0
CPU offload official = 0
dataset name not used for selector/controller
v9.7.0 boundary reproduced
```

失败后停止所有下游。

### 可视化

```text
field legality Sankey：green / yellow / red 字段流向；
boundary reproduction table；
```

---

## P1. Exact transfer artifact landing

### 目标

补齐 v9.7.0 缺失的 exact per-sample gradient / exact apply checkpoint。

### 假设

如果 exact transfer 不能落地，v9.7.1 不能判断 transfer principle。

### 两种实现方式

#### 方式 A：per-example gradient inner product

对每个 action $\Delta_a$ 和检查样本 $i$，记录：

$$
r_i(a)=-g_i^\top\Delta_a.
$$

#### 方式 B：exact apply microprobe

对少量 action，真实 apply 一小步：

$$
r_i^{exact}(a)=CE_i(\theta)-CE_i(\theta+\epsilon\Delta_a).
$$

方式 A 更快，方式 B 更准。P1 必须至少落地方式 A；P2/P3 对关键 TopK 需要方式 B 交叉验证。

### 必须记录

```text
action_id
event_id
dataset_id_for_diagnostic_only
seed
family
step_bucket
template_id
payload_hash
batch_split_id
producer_sample_count
checker_sample_count
memory_sample_count
per_example_response_mean
per_example_response_std
per_example_response_lcb
positive_response_fraction
negative_response_fraction
hard_tail_response_mean
old_family_response_mean
old_stratum_response_mean
action_norm
AdamW_cosine
exact_apply_epsilon
exact_apply_response_mean
exact_apply_response_lcb
exact_vs_linear_corr
exact_vs_linear_abs_error_p95
compute_ms
```

### 判断标准

Strong pass：

```text
all 2876 AP0 actions have exact linear transfer rows；
missing required field = 0；
NaN/Inf = 0；
exact-vs-linear correlation on audit subset >= 0.80；
exact-vs-linear p95 abs error <= predefined tolerance；
feature compute q90 <= allowed budget；
```

Weak pass：

```text
all 2876 AP0 actions have linear transfer rows；
exact apply audit subset >= 256 actions；
exact-vs-linear correlation >= 0.60；
```

Fail：

```text
exact gradient unavailable；
exact apply checkpoint unavailable；
exact-vs-linear correlation < 0.60；
```

### 可视化

```text
linear transfer vs exact apply scatter；
per-example response distribution violin；
response mean vs response std plot；
per-dataset response scale histogram；
```

---

## P2. Exact transfer selector on existing AP0 actions

### 目标

判断 exact transfer 是否比 v9.7.0 的 proxy transfer 更有用。

### 假设

如果 transfer principle 是对的，那么 exact transfer 的 TopK 应该至少接近旧 R8A 的质量，同时比旧 R8A 更 LDO 稳。

### 候选规则

```text
E0-old-R8A-baseline
E1-exact-transfer-LCB-only
E2-exact-transfer-positive-fraction
E3-exact-transfer-SNR
E4-exact-transfer-LCB + size hard gate
E5-exact-transfer-LCB + memory hard gate
E6-exact-transfer-LCB + old-family hard gate
E7-exact-transfer-LCB + risk hard gate
E8-exact-transfer-LCB + all hard gates
```

注意：这些不是加权 score。它们只改变 hard gate。

### 必须记录

```text
rule_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
max_dataset_share
max_stratum_share
max_template_share
feature_cost_q90
```

### Strong pass

```text
accepted_count >= 87
coverage >= 0.03
GradeAB_precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_UCB <= 0.05
offdiag_UCB <= 0.05
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
```

### Weak pass

```text
accepted_count >= 87
GradeAB_precision >= 0.70
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
LDO_drop <= 0.20
```

### Fail

```text
exact transfer quality similar to v9.7.0 proxy:
precision < 0.30 or V_LCB <= 0；
or exact transfer has good quality but LDO still >= 0.30；
```

### 可视化

```text
rule comparison table；
TopK precision vs LDO tradeoff curve；
V LCB vs longrisk UCB scatter；
accepted action distribution by dataset / template / step；
```

---

## P3. Core 77 + exact-transfer expansion

### 目标

解决最具体的 existing-action 问题：77 个 core 动作很干净，但 coverage 不够。P3 只问：能否用 exact transfer 找到额外 10 个 expansion 动作。

### 假设

如果 exact transfer 有用，它不一定要重排全部动作，只要能从 core 外找到 10 个安全动作即可。

### 候选 expansion

```text
X0-core-only-77
X1-core-plus-10-old-R8A
X2-core-plus-10-proxy-TransferLCB
X3-core-plus-10-exact-TransferLCB
X4-core-plus-10-exact-TransferLCB-memory-gate
X5-core-plus-10-exact-TransferLCB-old-family-gate
X6-core-plus-10-exact-TransferLCB-risk-gate
X7-core-plus-10-exact-TransferLCB-all-hard-gates
```

### 必须记录

```text
core_count
expansion_count
accepted_count
coverage
expansion_action_ids
expansion_dataset_distribution
expansion_template_distribution
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
```

### Pass 标准

```text
accepted_count >= 87
GradeAB_precision >= 0.90
V_integrated_LCB >= 0.12
h240_longrisk_UCB = 0 or <= 0.03
bad_UCB = 0 or <= 0.03
null_UCB <= 0.10
memory_UCB <= 0.05
offdiag_UCB <= 0.05
LDO_drop <= 0.15 strong; <= 0.20 weak
```

### 可视化

```text
core vs expansion action quality bar chart；
expansion nearest-neighbor map；
per-dataset expansion count and precision；
LDO before/after expansion plot；
```

---

## P4. Dataset shift without dataset-specific tuning

### 目标

诊断不同数据集上的 score shift，但不能按数据集调 controller。

### 假设

如果 exact transfer 是更本质的信号，它的 score distribution 应该比 old rank 更容易做 dataset-agnostic transport。

### 允许的 transport

允许：

```text
global conformal threshold；
global rank percentile；
leave-one-dataset calibration；
feature standardization from training stream statistics；
```

禁止：

```text
if dataset == Fashion-MNIST then threshold = ...；
per-dataset TopK；
per-dataset manually chosen veto；
```

### 必须记录

```text
score_name
PSI_mean
PSI_max
per_dataset_score_mean
per_dataset_score_std
per_dataset_target_density
per_dataset_core_count
per_dataset_precision
per_dataset_V_LCB
per_dataset_longrisk_UCB
transport_rule_id
transport_LDO_drop
transport_precision
transport_V_LCB
transport_longrisk_UCB
```

### Pass 标准

```text
transported rule passes P2 or P3 weak/strong gate；
PSI_mean decreases vs raw score；
no dataset-specific branch used；
```

### 可视化

```text
score histogram by dataset；
QQ plot raw vs transported score；
threshold stability plot；
per-dataset accepted count vs precision；
```

---

## P5. Direct transfer-solved update in small KAN subspaces

### 目标

如果 exact transfer 有用，不再从人工 APG family 里挑动作，而是在一个小参数子空间里直接求更新。

### 假设

存在一个小更新 $\Delta^*$，它能最大化 transfer LCB，同时不破坏 memory / risk / cost。

### 子空间

```text
D1-last-edge-coefficients-only
D2-final-KAN-basis-block-only
D3-low-rank-edge-residual-direction
D4-AdamW-orthogonal-residual-direction
D5-memory-gradient-orthogonal-residual-direction
D6-hard-tail-edge-local-direction
```

### 求解形式

线性形式：

$$
\Delta^* = \arg\max_{\Delta\in\mathcal{A}} \mu_B(\Delta)-z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}
$$

约束：

$$
\|\Delta\|\le\epsilon,
$$

$$
MemoryHarm(\Delta)\le\tau_M,
$$

$$
Cost(\Delta)\le C_{max}.
$$

### 必须记录

```text
subspace_id
solver_status
generated_action_count
payload_hash_missing
action_apply_error_linf
exact_transfer_LCB
exact_transfer_mean
exact_transfer_std
memory_harm_proxy
longrisk_proxy
branch_horizon_rows
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
LDO_drop
runtime_payload_apply_ms
```

### Open condition

P5 只能在 P1 exact transfer weak pass 后打开。

### Pass 标准

```text
generated_action_count >= 64
branch_horizon_rows complete
GradeAB_precision >= 0.50 for smoke weak
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
negative control fails
```

如果 P5 smoke pass，再进入 P8 controller candidate。否则 generated route 继续 stop。

### 可视化

```text
subspace transfer objective curve；
exact transfer vs branch-horizon V scatter；
negative control comparison；
subspace cost comparison；
```

---

## P6. Proxy vs exact transfer failure taxonomy

### 目标

解释 v9.7.0 proxy transfer 为什么失败。

### 必须分类

```text
F1 proxy sign wrong：proxy transfer 与 exact transfer 符号相反；
F2 proxy scale wrong：排序相关但尺度不稳；
F3 exact transfer itself weak：exact 也选不出好动作；
F4 exact transfer value-positive but longrisk high；
F5 exact transfer dataset stable but low precision；
F6 exact transfer precise but coverage low；
F7 exact transfer useful only as expansion feature；
```

### 必须记录

```text
proxy_score
exact_score
actual_GradeAB
actual_V
actual_longrisk
spearman_proxy_exact
spearman_exact_actual
sign_match_rate
per_dataset_corr
per_template_corr
failure_class
```

### 可视化

```text
proxy vs exact scatter；
exact vs actual V scatter；
failure class stacked bar；
per-dataset correlation heatmap；
```

---

## P7. Existing-action minimal controller

### 目标

只有 P2 或 P3 过 weak gate，才冻结一个最小 controller。

### Controller 形式

不允许复杂人工加权 score。允许：

```text
rank by exact_transfer_LCB；
core action whitelist learned from legal training split；
hard gates for size / memory / longrisk / cost；
fixed TopK count or conformal threshold；
```

### 必须记录

```text
controller_id
calibration_split
heldout_split
accepted_count_cal
accepted_count_heldout
coverage_heldout
GradeAB_precision_heldout
V_LCB_heldout
longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
LDO_drop
LSO_drop
LTO_drop
feature_compute_ms_q90
payload_apply_ms_q90
```

### Pass 标准

```text
accepted_count >= 87
coverage >= 0.03
GradeAB_precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO/LSO/LTO drop <= 0.10 strong, <= 0.20 weak
```

---

## P8. Selected runtime

### 目标

只有 P7 pass 才测 selected controller runtime。

### 必须记录

```text
step_count
active_step_count
selected_action_count
feature_compute_ms_q50/q90/q99
exact_transfer_compute_ms_q50/q90/q99
payload_apply_ms_q50/q90/q99
kernel_count
sync_count
step_ratio_q50/q90/q99
peak_memory_ratio
```

### Pass 标准

```text
step_ratio_q90 <= 1.50
peak_memory_ratio <= 1.05
no CPU offload
no fake/proxy runtime rows
```

---

## P9. Generated-route decision

### 目标

决定 generated route 是否继续停止，还是因为 exact transfer objective 出现而重新打开。

### 规则

继续停止，如果：

```text
P1 exact transfer fail；
P2/P3 exact transfer selector fail；
P5 direct transfer-solved update not_run or fail；
```

重新打开，如果：

```text
P5 exact direct solved smoke weak pass；
negative controls fail；
branch-horizon real outcomes show positive V and low longrisk；
```

### 必须记录

```text
generated_route_status
reason
new_objective_evidence_present
generated_action_count
branch_horizon_rows
best_generated_precision
best_generated_V_LCB
best_generated_longrisk_UCB
```

---

## P10. Official system boundary

### 目标

只在 P7 controller + P8 runtime 过线后打开。

### Pass 标准

```text
controller_pass = 1
runtime_pass = 1
contract_audit_pass = 1
no_fake_no_proxy_pass = 1
system_legal_controller_pass = 1
```

否则 paired replay / short-full 全部 gate-block。

---

## P11. Paired replay boundary

### 只在 P10 pass 后打开

必须比较：

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
```

必须记录：

```text
final_test_acc_delta
CE_AUC_delta
NLL_delta
ECE_delta
CEp99_delta
hard_tail_acc_delta
runtime_delta
memory_delta
```

Pass 标准：

```text
RealFunctional beats AdamWParallel and bestLR on primary metrics；
shuffled payload fails；
NoOp fails；
runtime within envelope；
no dataset-specific tuning；
```

---

## P12. Base-Acc Sentinel continuation

继续跑，但不用于 controller。

必须记录：

```text
MNIST / Fashion-MNIST / KMNIST
seeds 0..9
LQ-t2-h256
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
mean test acc
std test acc
train trace
catastrophic fail flag
base_acc_used_for_controller = 0
```

---

# 9. 并行执行结构

v9.7.1 必须并行，不再一轮只查一个 blocker。

```text
Lane A: Exact transfer artifact
  P1, P6

Lane B: Existing-action selection
  P2, P3, P4, P7

Lane C: Direct solved update
  P5, P9

Lane D: Runtime/system boundary
  P8, P10, P11

Lane E: Sentinel/no-fake/contract
  P0, P12
```

执行顺序：

```text
第一批并行：P0, P1, P4, P12
第二批并行：P2, P3, P6
第三批条件并行：P5, P7
第四批条件并行：P8, P10, P11
```

---

# 10. 预期 route 决策

## R1-ExactTransferArtifactMissing

```text
P1 fail。
不能继续 transfer route；existing-action route 只能继续 old-rank diagnostics；generated route stop。
```

## R2-ProxyTransferFailedButExactTransferWorks

```text
P1 pass，P2/P3 exact transfer pass。
继续 existing-action controller。
```

## R3-TransferPrincipleFalseForExistingActions

```text
P1 pass，但 exact transfer precision/value/risk 仍失败。
停止 transfer-as-selector route。
```

## R4-TransferUsefulOnlyForExpansion

```text
P2 full ranking fail，但 P3 core+exact-expansion pass。
继续 minimal controller，只把 transfer 用作 expansion selector。
```

## R5-DirectTransferSolvedUpdatePromising

```text
P5 pass。
generated route 以 direct solved update 形式重新打开。
```

## R6-ExistingActionControllerPassRuntimeBlocked

```text
P7 pass，P8 fail。
下一轮只修 selected runtime，不再修 selector。
```

## R7-SystemLegalLocalControllerPass

```text
P7 + P8 + P10 pass。
打开 paired replay boundary。
```

---

# 11. Stop 条件

### Stop transfer selector

如果 exact transfer artifact pass 后仍满足：

```text
TopK87 precision < 0.30
V_LCB <= 0
longrisk_UCB > 0.10
```

则停止 transfer-as-ranker route。

### Stop generated route

如果 P5 不打开或 P5 fail，则继续：

```text
generated_route_status = stopped_no_new_objective
```

禁止盲跑 APGV/APGW。

### Stop existing-action controller route

如果 P2/P3/P7 都 fail，且 exact transfer 不能改善 core expansion，则 existing-action route 进入：

```text
existing_action_route = diagnostic_only
```

下一步转向更底层的 update-rule math，而不是继续调 rank/certificate。

---

# 12. 本轮最终判断标准

v9.7.1 最低有效进展不是 system success，而是至少完成一个明确判断：

```text
1. exact transfer 是否真的比 proxy transfer 更有用；
2. exact transfer 是否能补 core 77 -> 87；
3. exact transfer 是否能降低 LDO 且保持 precision/value/risk；
4. direct transfer-solved update 是否值得重新打开 generated route；
5. 如果这些都失败，existing-action + transfer 路线应停止小修，回到更基础的数学判据或 KAN 子空间设计。
```

强成功标准：

```text
P7 controller pass + P8 runtime pass + P10 system pass。
```

弱成功标准：

```text
P3 core+exact-expansion weak pass；
或 P5 direct solved update weak pass；
```

失败标准：

```text
P1 exact artifact fail；
或 exact transfer 与 proxy 一样失败；
或 exact transfer only reduces LDO but destroys precision/value；
```

---

# 13. 为什么这不是小修小补

v9.7.1 不再问：

```text
R8A threshold 怎么调？
TransferLCB threshold 怎么调？
CorePlus10 怎么调？
APGU 要不要换个名字？
```

它问的是：

$$
\boxed{
\text{functional update 的第一性判据是否应该是 cross-sample transfer？}
}
$$

如果答案是否定的，就停止 transfer route。  
如果答案是肯定的，就直接进入最小 controller 或 direct update。  
这比继续堆人工 score 更硬，也更能加快项目。

