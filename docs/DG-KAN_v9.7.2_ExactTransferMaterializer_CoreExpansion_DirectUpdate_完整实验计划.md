# DG-KAN v9.7.2 Exact Transfer Materializer / Core Expansion / Direct Update 完整实验计划

> 本计划基于 v9.7.1 的真实结果制定。v9.7.1 的核心 route 是：
>
> ```text
> route = R1-ExactTransferArtifactMissing
> strict_purekan_functional = False
> full_functional = False
> external_ready = False
> ```
>
> v9.7.1 没有验证 exact transfer 原理本身。它只验证了一个前提：exact per-sample gradient 和 exact apply checkpoint 没有落盘，所以 exact transfer 实验对象不存在。v9.7.2 的目标不是再调 proxy，也不是再给 existing-action rank 加一个阈值，而是把 exact transfer 这个对象真实造出来，并用它同时验证两条路线。

---

# 0. 这轮一句话目标

v9.7.2 要回答一个很具体的问题：

$$
\boxed{
\text{训练当下能否用“跨样本有效”的证据，稳定挑出或直接生成一个小的 functional update？}
}
$$

这里“跨样本有效”不是抽象词。意思是：

```text
一个动作不是只让提出它的那批样本变好，
而是让另一批没有参与构造它的样本也变好。
```

如果一个动作只对生成它的样本有效，它更像记噪声；如果它对另一批样本也有效，它更可能是真正有泛化价值的更新方向。

---

# 1. v9.7.1 的独立判断

## 1.1 v9.7.1 有进展，但进展非常有限

v9.7.1 做对了一件事：没有把 proxy transfer 冒充 exact transfer，也没有编造 fake rows。

真实结果是：

```text
AP0 action count = 2876
exact linear transfer rows = 0
exact apply audit subset = 0
exact_per_sample_gradient_available = 0
exact_apply_checkpoint_available = 0
matched exact artifact count = 0
proxy_promoted_to_official = 0
```

所以 v9.7.1 的科学结论不是：

```text
cross-sample transfer 原理失败。
```

更准确是：

$$
\boxed{
\text{exact transfer 尚未被真正测试；当前失败是 artifact/materializer 失败。}
}
$$

## 1.2 v9.7.1 为什么让人感觉非常慢

这轮确实非常像“没进度”，因为它没有打开：

```text
existing-action controller；
selected runtime；
system controller；
paired replay；
short/full training；
direct transfer-solved update。
```

它主要确认了：

```text
exact transfer rows = 0；
exact apply checkpoint = 0；
direct update generated actions = 0；
generated route 继续 stopped_no_new_objective。
```

这类问题不应该等一个完整 runner 跑完才发现。v9.7.2 必须把 exact artifact preflight 提前成第一分钟就完成的 gate。

## 1.3 v9.7.1 没有推翻 existing-action 路线

旧的 existing-action rank 仍然很强，但跨 dataset 不稳：

```text
old R8A precision = 0.8735632183908046
old R8A V LCB = 0.14111334880346277
old R8A LDO drop = 0.37931034482758624
```

proxy transfer 的质量很差，但 LDO 更稳：

```text
proxy transfer precision = 0.17525773195876287
proxy transfer V LCB = -0.21558700787971152
proxy transfer LDO drop = 0.07216494845360824
```

这说明：

```text
旧 rank 能找到好动作，但跨 dataset 不稳；
proxy transfer 跨 dataset 稳一些，但动作质量太差；
exact transfer 是否能同时兼顾两者，尚未验证。
```

## 1.4 v9.7.1 暴露的真正管理问题

v9.7.1 的最大反思不是 transfer 理论，而是实验流程：

```text
如果 exact_per_sample_gradient_available = 0，
那么 P2/P3/P4/P5/P6/P7 不应该继续作为完整实验阶段运行，
应该直接进入 artifact-build route。
```

v9.7.2 必须加入分层 preflight：

```text
1 action exact gradient preflight；
8 action exact gradient preflight；
64 action exact gradient preflight；
2876 action full exact transfer ledger。
```

这样才能避免“一轮发现一个基础缺失”的慢节奏。

---

# 2. 本轮总体路线：双线验证

v9.7.2 不应该放弃当前路线，也不能只做新理论。应该双线并行。

## 2.1 A 线：继续验证 existing-action route

A 线要问：

```text
canonical AP0 里已经存在的动作，能否用 exact transfer 补足 core 77 -> 87，
并且降低 LDO drop？
```

当前已知：

```text
core-only 动作数 = 77
coverage = 0.02677
GradeAB precision = 1.0
V LCB = 0.164
longrisk/bad/null/memory/offdiag = 0
```

但 coverage 下限约需要：

$$
N_{min}=\lceil 0.03 \times 2876 \rceil = 87.
$$

所以缺口是：

$$
87-77=10.
$$

A 线的目标不是重新找 87 个，而是只解决这个缺口：

```text
保住 77 个 clean core；
只用 exact transfer 选择额外 10 个 expansion actions；
检查这 87 个动作是否同时过 precision / value / risk / LDO。
```

## 2.2 B 线：验证 exact cross-sample transfer principle

B 线要问：

```text
如果一个动作对没参与构造它的样本也有效，
它是否更可能是好 functional update？
```

对动作 $\Delta$，对检查样本 $i$，定义线性化收益：

$$
r_i(\Delta)=-g_i^\top \Delta.
$$

对检查集合 $B$，定义均值和方差：

$$
\mu_B(\Delta)=\frac{1}{|B|}\sum_{i\in B}r_i(\Delta),
$$

$$
\sigma_B^2(\Delta)=\frac{1}{|B|-1}\sum_{i\in B}(r_i(\Delta)-\mu_B(\Delta))^2.
$$

保守下界：

$$
LCB_{transfer}(\Delta)
=
\mu_B(\Delta)
-
z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}.
$$

最简单的接受条件是：

$$
LCB_{transfer}(\Delta)>0.
$$

这里不再手工加很多项。memory、longrisk、cost 先只作为硬门，不做复杂加权。

## 2.3 C 线：direct transfer-solved update

C 线只在 B 线 exact transfer 成立后打开。

它不再从 APG/APGU 这种手工 primitive 家族里造动作，而是在一个小 KAN 子空间里直接求：

$$
\Delta^*=
\arg\max_{\Delta\in\mathcal A}
LCB_{transfer}(\Delta)
$$

并满足：

$$
\|\Delta\|\le \epsilon,
$$

$$
Cost(\Delta)\le C_{max}.
$$

这里 $\mathcal A$ 是一个很小的允许空间，例如：

```text
最后一层 edge coefficient 子空间；
低秩 edge 子空间；
AdamW 正交小残差子空间；
KAN basis block 子空间。
```

如果 B 线不过，C 线不运行。

## 2.4 D 线：generated route 继续停止

APGH/APGL/APGT 已经连续 value-negative / high-longrisk。v9.6.8 和 v9.7.0 都已经决定 generated route stopped_no_new_objective。v9.7.2 继续遵守：

```text
没有 exact transfer objective 前，不跑 APGU/APGV/APGW blind variant；
没有 direct transfer-solved update 前，不恢复 generated-action route；
没有 branch-horizon outcome 前，不写 generated frontier pass。
```

---

# 3. 本轮硬约束

继续遵守：

```text
no teacher；
no self-teacher；
no distillation；
no loss modification；
no auxiliary loss；
no dataset-specific selector；
no validation/test feature at commit；
no future outcome feature；
no outcome-derived field in official score；
no old table official；
no fake rows；
no proxy rows；
no CPU offload；
KAN path 不使用 PyTorch loss.backward graph；
functional update 是 update rule，不是 loss trick。
```

允许：

```text
per-example gradient materialization；
train-memory buffer microprobe；
linearized transfer；
small exact apply audit subset；
leave-dataset-out diagnostic；
leave-stratum-out diagnostic；
leave-template-out diagnostic；
proxy transfer only as diagnostic baseline；
Base-Acc Sentinel as health monitor only。
```

---

# 4. 关键术语说明

## 4.1 action

一个 action 是一次候选额外参数改动。它代表：

```text
在普通 AdamW 更新之外，额外给 KAN 参数加一小步。
```

写成公式：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{func}.
$$

## 4.2 existing-action

existing-action 指已经在 canonical AP0 action table 里的历史动作。它们不是新生成的，而是已有的 2876 个动作。

## 4.3 exact transfer

exact transfer 指对一个动作 $\Delta$，用真实 per-example gradient 或真实小步 apply，测它对另一批样本是否有帮助。

它不是 proxy，不是 action_projection_signal，也不是 loo_transfer_proxy。

## 4.4 proxy transfer

proxy transfer 是 v9.7.0 已经落盘的近似信号，例如：

```text
loo_transfer_proxy；
action_projection_signal；
grad_mean_sq_group；
grad_var_trace_group。
```

v9.7.0 已证明 proxy 不够好，不能 official。

## 4.5 core actions

core actions 是当前最干净的 77 个动作：

```text
GradeAB = 1；
V LCB > 0；
longrisk = 0；
bad = 0；
null = 0；
memory/offdiag = 0；
但数量只有 77，不够 coverage。
```

## 4.6 expansion actions

expansion actions 是为了从 77 补到 87 而添加的动作。它们必须尽量接近 core 质量。

## 4.7 LDO

LDO 是 leave-dataset-out。意思是：

```text
在不看某个 dataset 的结果时训练/选择规则，
再看这个规则在被留出的 dataset 上是否仍然有效。
```

它用于防止规则只适合 MNIST / Fashion-MNIST / KMNIST 中某一个数据集。

## 4.8 LSO

LSO 是 leave-stratum-out。意思是：

```text
留出某类样本/事件/难度区域，检查规则是否仍然有效。
```

## 4.9 LTO

LTO 是 leave-template-out。意思是：

```text
留出某种 action template，检查规则是否仍然有效。
```

---

# 5. 阶段设计

# P0. Boundary reproduction and fail-fast preflight

## 目标

确认 v9.7.1 boundary，并在 1 分钟内判断 exact artifact 是否真的可生成。不能再等完整 runner 跑完才发现 exact rows = 0。

## 假设

H0：v9.7.1 的失败不是 transfer 原理失败，而是 exact artifact 缺失。

## 执行

读取 v9.7.1 artifact，确认：

```text
source_route = R1-ExactTransferArtifactMissing；
exact_linear_transfer_row_count = 0；
exact_per_sample_gradient_available = 0；
exact_apply_checkpoint_available = 0；
generated_route_status = stopped_no_new_objective。
```

然后执行 1-action exact preflight：

```text
选 1 个 canonical AP0 action；
选 8 个 train-check samples；
计算 per-example gradient；
计算 r_i = -g_i^T Δ；
写 exact_transfer_preflight_1action.csv；
检查字段完整、NaN/Inf、hash、cost。
```

## 记录指标

```text
preflight_action_count
preflight_sample_count
per_example_gradient_available
r_i_count
missing_gradient_count
nan_count
inf_count
gradient_compute_ms_q50/q90
memory_mb_peak
exact_transfer_preflight_pass
```

## 成立标准

```text
per_example_gradient_available = 1；
r_i_count = action_count * sample_count；
missing_gradient_count = 0；
nan_count = 0；
inf_count = 0；
gradient_compute_ms_q90 <= pre-registered budget。
```

若 P0 fail，v9.7.2 不进入 P1-P12，直接 route：

```text
R0-ExactTransferPreflightMissing
```

## 可视化

```text
1-action r_i histogram；
per-example gradient norm histogram；
gradient compute time bar；
missing field checklist。
```

---

# P1. Exact per-example gradient materializer

## 目标

把 exact transfer 的核心材料落盘：每个 action 对一组检查样本的 per-example linear response。

## 假设

H1：exact per-example gradient materializer 可以在不使用 fake/proxy/CPU offload 的情况下，为 2876 个 AP0 actions 生成完整 response ledger。

## 执行

分三档运行：

```text
P1a: 8 actions × 32 samples；
P1b: 128 actions × 64 samples；
P1c: 2876 actions × 64 samples。
```

每个 action 记录：

```text
action_id；
event_id；
dataset；
seed；
template_id；
family；
step_bucket；
payload_hash；
payload_norm；
check_sample_id；
check_sample_group；
grad_hash；
grad_norm；
response_linear = -g_i^T Δ；
response_sign；
response_abs；
compute_ms；
memory_mb。
```

## 记录指标

```text
exact_linear_transfer_row_count
expected_row_count
completion_rate
action_completion_rate
sample_completion_rate
missing_gradient_count
missing_payload_count
nan_count
inf_count
duplicate_row_count
response_linear_mean/std/p10/p50/p90
per_dataset_response_mean
per_template_response_mean
compute_ms_q50/q90
memory_mb_peak
```

## 判断标准

强通过：

```text
completion_rate = 1.0；
action_completion_rate = 1.0；
missing_gradient_count = 0；
nan_count = 0；
inf_count = 0；
duplicate_row_count = 0。
```

弱通过：

```text
completion_rate >= 0.95；
missing rows 有完整 retry manifest；
缺失不是集中于某 dataset / template / family。
```

失败：

```text
completion_rate < 0.95；
或者 exact rows 仍为 0；
或者需要 proxy 才能填表。
```

## 可视化

```text
response_linear histogram；
response by dataset violin；
response by template violin；
action completion heatmap；
compute time by action size scatter；
response mean vs payload norm scatter。
```

---

# P2. Exact apply audit subset

## 目标

验证线性化 response 与真实 apply 后的 loss change 是否一致。否则 exact linear transfer 可能只是理论上好看。

## 假设

H2：在小步动作上，$-g_i^T\Delta$ 能预测真实 apply 后的 per-example CE 改善。

## 执行

选取：

```text
64 actions：
  16 core actions；
  16 old R8A high-score actions；
  16 proxy-transfer high-score actions；
  16 random/low-score controls。

每个 action 选 64 check samples。
```

对每个 action/sample，记录：

```text
CE_before；
CE_after_exact_apply；
response_apply = CE_before - CE_after；
response_linear = -g_i^T Δ；
linear_apply_error = response_apply - response_linear。
```

## 记录指标

```text
exact_apply_action_count
exact_apply_sample_count
linear_apply_correlation
linear_apply_spearman
linear_apply_mae
linear_apply_sign_match_rate
apply_response_mean/std
per_dataset_correlation
per_template_correlation
exact_apply_compute_ms_q90
```

## 判断标准

强通过：

```text
linear_apply_correlation >= 0.70；
linear_apply_sign_match_rate >= 0.70；
per_dataset_correlation >= 0.50 for all datasets。
```

弱通过：

```text
linear_apply_correlation >= 0.50；
linear_apply_sign_match_rate >= 0.60；
至少 2/3 dataset 通过。
```

失败：

```text
linear_apply_correlation < 0.50；
或 sign_match_rate < 0.60。
```

## 可视化

```text
response_linear vs response_apply scatter；
sign match confusion matrix；
correlation by dataset bar；
apply response histogram for core / rank / random groups。
```

---

# P3. Exact transfer score definitions

## 目标

用最少人工设计定义 transfer 分数，避免重新变成 GeometryScore patching。

## 候选规则

### T0：mean transfer

$$
Score_{T0}(\Delta)=\mu_B(\Delta).
$$

### T1：LCB transfer

$$
Score_{T1}(\Delta)=\mu_B(\Delta)-z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}.
$$

### T2：SNR transfer

$$
Score_{T2}(\Delta)=\frac{\mu_B(\Delta)^2}{\sigma_B^2(\Delta)+\epsilon}.
$$

### T3：sign-agreement transfer

$$
Score_{T3}(\Delta)=\frac{1}{|B|}\sum_{i\in B}\mathbf 1[r_i(\Delta)>0].
$$

### T4：core-safe transfer

T4 不加权，只做硬门：

```text
T1 rank top；
longrisk proxy veto；
memory/offdiag veto；
cost veto。
```

## 记录指标

对每个 score 记录：

```text
TopK64/TopK77/TopK87/TopK97 precision
V_integrated LCB
longrisk UCB
bad UCB
null UCB
memory UCB
offdiag UCB
LDO drop
LSO drop
LTO drop
max dataset share
max template share
accepted count
coverage
PSI by dataset
score mean/std by dataset
```

## 判断标准

一个 transfer score 只有同时满足以下条件才可进入 controller candidate：

$$
N_{accept}\ge 87,
$$

$$
Precision_{GradeAB}\ge0.75,
$$

$$
LCB(V_{integrated})>0,
$$

$$
UCB(LongRisk)\le0.05,
$$

$$
UCB(Bad)\le0.05,
$$

$$
UCB(Null)\le0.15,
$$

$$
LDO_{drop}\le0.10,
$$

$$
LSO_{drop}\le0.10,
$$

$$
LTO_{drop}\le0.10.
$$

## 可视化

```text
score distribution by dataset；
score vs V scatter；
score vs longrisk scatter；
TopK composition stacked bar；
LDO drop waterfall；
precision-coverage curve；
V-risk frontier curve。
```

---

# P4. A 线：Core 77 + Exact Transfer Expansion

## 目标

用 exact transfer 只补足最小缺口：从 77 个 core actions 补到 87 个 accepted actions。

## 假设

H4：core 77 是稳定干净区域，exact transfer 可以比 proxy / old rank 更好地选择 10 个 expansion actions。

## 执行

固定 core：

```text
core_set = T3.1 / T2.3 / MemoryOffdiagCore 的 77 个 clean actions。
```

候选 expansion pool：

```text
从非 core actions 中选择。
排除：longrisk proxy fail；memory/offdiag fail；bad/null high-risk。
按 T1/T2/T3 transfer score 排名。
选 10 个。
```

形成：

```text
accepted_set = core_77 + expansion_10。
```

## 记录指标

```text
core_count
expansion_count
accepted_count
expansion_action_ids
expansion_dataset_distribution
expansion_template_distribution
expansion_transfer_score_mean
accepted GradeAB precision
accepted V LCB
accepted longrisk UCB
accepted bad UCB
accepted null UCB
accepted memory/offdiag UCB
LDO/LSO/LTO drop
```

## 判断标准

强通过：

```text
accepted_count >= 87；
GradeAB precision >= 0.75；
V LCB > 0；
longrisk/bad UCB <= 0.05；
null UCB <= 0.15；
LDO/LSO/LTO drop <= 0.10。
```

弱通过：

```text
accepted_count >= 87；
GradeAB precision >= 0.75；
V LCB > 0；
longrisk = 0；
LDO drop <= 0.20。
```

失败：

```text
补 10 个后 LDO 仍约 0.39；
或者 V LCB 变负；
或者 null/offdiag/bad 超线。
```

## 可视化

```text
core vs expansion quality table；
expansion action transfer score bar；
per-dataset accepted count；
per-dataset precision；
LDO failure matrix；
core-only vs core+expansion frontier plot。
```

---

# P5. B 线：Exact transfer vs old rank vs proxy transfer

## 目标

直接比较三类方法：

```text
old rank：旧 R8A/R5B/RANK 类规则；
proxy transfer：v9.7.0 的近似 transfer；
exact transfer：本轮新落盘的 exact response。
```

## 假设

H5：exact transfer 应该至少比 proxy transfer 更接近 old rank 的 precision，同时比 old rank 更稳。

## 记录表

每个方法、每个 TopK：

```text
method_id
TopK
precision
V LCB
longrisk UCB
bad UCB
null UCB
memory/offdiag UCB
LDO drop
LSO drop
LTO drop
PSI mean
max dataset share
max template share
```

## 判断标准

Exact transfer strong pass：

```text
TopK87 precision >= 0.75；
V LCB > 0；
longrisk UCB <= 0.05；
LDO drop <= 0.10。
```

Exact transfer useful diagnostic：

```text
TopK87 precision >= 0.60；
V LCB > 0；
LDO drop <= old_rank_LDO_drop - 0.15。
```

Exact transfer fail：

```text
precision <= proxy transfer precision + 0.10；
或者 V LCB < 0；
或者 LDO drop 与 old rank 同样高。
```

## 可视化

```text
method comparison bar；
precision vs LDO scatter；
V vs longrisk scatter；
per-dataset score calibration plot；
TopK overlap Venn diagram。
```

---

# P6. Exact transfer dataset shift audit

## 目标

判断 exact transfer 是否真正缓解 dataset shift，还是像 proxy 一样只换了一个不稳定分数。

## 执行

对 MNIST / Fashion-MNIST / KMNIST 记录：

```text
score mean；
score std；
TopK accepted count；
TopK precision；
V LCB；
longrisk UCB；
target density；
core coverage；
expansion coverage。
```

不允许按 dataset 调 threshold，只做诊断。

## 判断标准

Dataset-shift solved：

```text
per-dataset precision >= 0.70；
per-dataset V LCB >= 0 或 macro V LCB > 0 且 no dataset negative severe；
LDO drop <= 0.10；
score PSI mean <= 0.15。
```

Dataset-shift unresolved：

```text
某 dataset precision drop > 0.20；
或者 score PSI mean > 0.30；
或者 exact score 在某 dataset 上整体尺度偏移严重。
```

## 可视化

```text
score density by dataset；
per-dataset TopK precision bar；
PSI heatmap；
heldout dataset drop waterfall；
core/expansion density by dataset。
```

---

# P7. Direct transfer-solved small update

## 前置条件

只有 P2 和 P3 至少 weak pass 才运行。

## 目标

不再手工设计 APG/APGU，而是在小 KAN 子空间里直接求一个 transfer-positive 更新。

## 子空间

```text
S1-last-layer-edge-coefficients；
S2-low-rank-edge-block；
S3-AdamW-orthogonal-residual；
S4-high-SNR-basis-block；
S5-memory-safe-basis-block；
S6-control-transfer-positive-block。
```

## 优化目标

基础版本：

$$
\Delta^*
=
\arg\max_{\Delta\in S}
\left(
\mu_B(\Delta)-z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}
\right)
$$

约束：

$$
\|\Delta\|\le\epsilon,
$$

$$
Cost(\Delta)\le C_{max}.
$$

安全门：

```text
memory proxy 不得明显变坏；
longrisk proxy 不得明显变坏；
payload apply cost 不得超过 budget。
```

## 执行

每个子空间生成：

```text
32 direct updates；
共最多 192 actions；
真实 payload；
真实 certificate；
真实 action apply；
真实 branch-horizon outcome。
```

## 记录指标

```text
generated_action_count
subspace_id
transfer objective value
payload norm
AdamW cosine
branch-horizon rows
GradeAB precision
V_integrated LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
new positive rate
longrisk created rate
rows/sec
payload_apply_ms
```

## 判断标准

Generated direct update pass：

```text
at least one subspace:
  accepted_count >= 32；
  GradeAB precision >= 0.50；
  V LCB > 0；
  longrisk UCB <= 0.10；
  longrisk created rate <= 0.10。
```

Strong pass：

```text
accepted_count >= 64；
GradeAB precision >= 0.75；
V LCB > 0；
longrisk/bad/null all pass；
LDO/LSO/LTO weak pass。
```

Fail：

```text
best generated V LCB < 0；
longrisk created > 0.50；
negative control comparable to best。
```

## 可视化

```text
subspace frontier plot；
transfer objective vs actual V scatter；
longrisk created by subspace；
new positive rate by subspace；
AdamW cosine vs outcome scatter。
```

---

# P8. Minimal existing-action controller candidate

## 前置条件

P4 or P5 至少 weak pass。

## 目标

将 exact transfer / core-expansion 规则冻结成一个最小 controller。

## Controller 形式

避免复杂人工 score。controller 只允许：

```text
core membership rule；
exact transfer rank；
hard veto：longrisk proxy / memory offdiag / cost；
fixed accepted count K=87 或 fixed threshold from calibration。
```

不允许：

```text
dataset-specific threshold；
outcome-derived feature；
validation/test metric；
proxy-only score official；
manual per-dataset correction。
```

## 记录指标

```text
controller_id
calibration split
heldout split
accepted count
coverage
precision
V LCB
longrisk/bad/null UCB
memory/offdiag UCB
LDO/LSO/LTO drop
max dataset share
max template share
feature compute ms
exact transfer compute ms
payload apply ms
```

## 判断标准

Controller pass：

$$
N_{accept}\ge87,
$$

$$
Precision\ge0.75,
$$

$$
LCB(V)>0,
$$

$$
UCB(LongRisk)\le0.05,
$$

$$
UCB(Bad)\le0.05,
$$

$$
UCB(Null)\le0.15,
$$

$$
LDO/LSO/LTO\ drop\le0.10.
$$

---

# P9. Selected runtime preflight

## 前置条件

P8 controller pass。

## 目标

测 selected controller 的真实 online runtime，不允许用 offline materializer runtime 替代。

## 执行

Timed path 包含：

```text
feature compute；
exact transfer compute or cached exact transfer lookup；
rank / veto；
payload apply；
base AdamW step。
```

Timed path 不包含：

```text
offline branch-horizon materializer；
outcome label lookup；
future replay；
training report IO。
```

## 指标

```text
step_ratio_q50/q90/q99
feature_compute_ms_q90
transfer_compute_ms_q90
rank_veto_ms_q90
payload_apply_ms_q90
memory_ratio
active_step_count
accepted_step_count
zero_event_step_launch_count
kernel_launch_count
sync_count
```

## 通过标准

```text
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05；
zero_event_step_launch_count = 0；
no CPU offload；
no offline materializer in timed path。
```

---

# P10. System boundary

## 目标

只有 controller 和 runtime 同时过线，才允许进入 paired replay / short-full。

## Pass 条件

```text
P8 controller pass = 1；
P9 runtime pass = 1；
field legality pass = 1；
no fake/proxy pass = 1；
manual training contract pass = 1；
secondary/control outcome ready = 1。
```

若任意失败：

```text
system_legal_controller_pass = 0；
paired replay not_run；
short/full not_run。
```

---

# P11. Official paired replay boundary

## 前置条件

P10 pass。

## 目标

证明 selected functional update 不是只在 diagnostic 表里好，而是真的打过 controls。

## Controls

```text
AdamWOnly；
AdamWParallel；
bestLR；
NoOp；
Random；
ShuffledFunctionalPayload。
```

## 指标

```text
RealFunctional vs best control V20/V80/V240
beats_AdamWParallel
beats_bestLR
beats_NoOp
beats_Random
shuffle_control_fail
bad/null/longrisk
ECE/NLL/CEp99/margin
```

## Pass 条件

```text
RealFunctional beats best control in primary value；
shuffled payload does not pass；
bad/null/longrisk all within gate；
LDO/LSO/LTO stable。
```

---

# P12. Short/full training boundary

## 前置条件

P11 pass。

## 目标

只在 system + paired replay 都过线后，才进入真实训练比较。

## 指标

```text
train/val/test acc
CE
NLL
ECE
CEp99
margin_p10
steps_to_target
time_to_target
sample efficiency
robustness noise sweep
continual retained accuracy
forgetting score
step time
memory
```

## Baselines

```text
MatchedMLP；
AdamWStrongLRGridMLP；
QuadraticFeatureMLP；
LQ base without functional update；
selected functional KAN。
```

## Pass 条件

不能只看 acc。需要：

```text
functional KAN 不比 LQ base 差；
至少一个非-acc 维度明显改善；
与 strong MLP baseline 公平比较；
runtime envelope 仍过；
不按 dataset 调参。
```

---

# 6. 并行执行安排

为了避免继续“一轮发现一个 blocker”，v9.7.2 按并行包执行。

## Batch A：artifact build

```text
P0 exact preflight；
P1 exact linear transfer materializer；
P2 exact apply subset。
```

如果 P0 fail，Batch A 停止，不跑后续大阶段。

## Batch B：existing-action analysis

```text
P3 transfer score definitions；
P4 core + expansion；
P5 rank comparison；
P6 dataset shift audit。
```

Batch B 只在 P1 至少 weak pass 后运行。

## Batch C：direct update

```text
P7 direct transfer-solved update。
```

只在 P2/P3 weak pass 后运行。

## Batch D：controller/runtime

```text
P8 minimal controller；
P9 selected runtime；
P10 system boundary。
```

只在 P4/P5 pass 后运行。

---

# 7. Route decision

## R0：Exact transfer preflight missing

```text
P0 exact preflight fails。
```

动作：停止全轮，修 artifact，不跑 controller。

## R1：Exact transfer materializer incomplete

```text
P1 completion < 0.95。
```

动作：修 materializer / memory / batching。

## R2：Linear transfer not predictive

```text
P2 linear_apply_correlation < 0.50。
```

动作：linear transfer 不能作为主线；需要 exact apply response 或二阶近似。

## R3：Exact transfer not better than proxy

```text
P5 exact transfer precision/value 不优于 proxy。
```

动作：停止 transfer controller 主线，保留为 diagnostic。

## R4：Exact transfer improves stability but quality too low

```text
LDO 下降，但 precision/V 崩。
```

动作：transfer 只能做 veto/regularizer，不能做 rank。

## R5：Core expansion solved, controller candidate ready

```text
P4 pass。
```

动作：进入 P8/P9。

## R6：Direct transfer-solved update promising

```text
P7 generated update weak/strong pass。
```

动作：打开 generated route，但只限 transfer-solved actions。

## R7：System pass

```text
P8 + P9 + P10 pass。
```

动作：进入 paired replay。

---

# 8. 本轮关键反思

v9.7.1 说明我们不能继续做这种实验：

```text
计划里写 exact transfer；
runner 里发现 exact artifact 不存在；
然后整轮停在 missing artifact。
```

v9.7.2 的核心改动是：

```text
先把 exact transfer 作为实验对象造出来；
再比较它是否比旧 rank 和 proxy transfer 有价值；
再决定是否进入 controller/runtime。
```

同时继续保留 existing-action route，因为它仍然有强信号；也继续停止 generated route，因为没有新目标前继续盲造只会重复 value-negative / high-longrisk。

---

# 9. 预期结论形式

v9.7.2 不要求一定 success。它必须至少给出一个清楚结论：

```text
Case A:
  exact transfer artifact 仍无法落地。
  结论：当前理论线还没进入可实验阶段。

Case B:
  exact transfer artifact 落地，但 linear response 不预测 true apply。
  结论：需要 exact apply / 二阶 / local Jacobian，不可用一阶 transfer。

Case C:
  exact transfer 可预测，但不能选好动作。
  结论：cross-sample transfer 原理不够，existing-action route 继续依赖旧 rank。

Case D:
  exact transfer 能补 core 77 -> 87，并降低 LDO。
  结论：进入 minimal controller / runtime。

Case E:
  direct transfer-solved update 产生新好动作。
  结论：generated route 用 transfer objective 重新打开。
```

---

# 10. 成功的最低定义

v9.7.2 的最低有效成功不是 full functional success，而是：

```text
1. exact transfer rows 不再是 0；
2. exact apply subset 能校验 transfer；
3. exact transfer 至少解释 existing-action core/expansion 的一部分；
4. 给出明确 route：继续 exact transfer、改 exact apply、还是停止 transfer route。
```

v9.7.2 的强成功是：

```text
core 77 + exact expansion 10 过线；
minimal controller 过 heldout/LDO/LSO/LTO；
selected runtime 过 step_ratio_q90 <= 1.50；
进入 paired replay boundary。
```

