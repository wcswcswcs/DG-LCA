# DG-KAN v9.8.1 Future-Trajectory Geometry-Adaptive Optimizer 完整实验计划

> 本计划基于 v9.8.0 `Geometry-Adaptive Optimizer` 的真实结果制定。  
> v9.8.1 不继续小修 `E4 / OldRank / WT80 / ExactTransfer / GeometryScore / APG*`。  
> 本轮要回答一个更高层的问题：
>
> $$
> \boxed{\text{优化器能否根据网络的未来训练轨迹和函数空间几何，自适应地产生或选择一个真正有用的小参数改动？}}
> $$
>
> 这里的“函数空间几何”不限制为 KAN，也不限制为当前 edge basis。KAN 的 basis rank、cover entropy、edge locality 可以作为诊断，但不能成为唯一理论基础。我们要找的是更通用的训练更新原则：它应该适用于 KAN、MLP、Transformer 或其他模型，只是 DG-KAN 当前作为最严格的实验平台。

---

# 0. 本轮一句话判断

v9.8.0 有进展，但不是能力成功。它第一次把 v9.7.7 的 `Core77 / support-backfill / natural density` 讨论推进到 `future trajectory` 和 `geometry-adaptive generated update` 层面；但它没有证明 future-trajectory 理论，也没有形成 controller。

v9.8.0 最重要的数据是：

```text
route = RouteD-FutureTrajectoryUnsupported
primary_blocker = future_trajectory_theory_not_supported
system_legal_controller_pass = 0
generated_route_status = stopped_no_official_generated_branch_horizon_pass
```

关键实测：

```text
P1 LDO_raw / LDO_quality / LDO_support / LDO_backfill
= 0.441558 / 0.092605 / 0.0 / 0.348953
backfill_share = 0.790276

P2 completed_panel_count = 1
P2 not_run_panel_count = 3
reason = extension_panels_blocked_by_missing_labeled_materializer

P3 future_trajectory_pass = 0
Core77 V20 LCB = 1.573549
OldOnly V20 LCB = 1.276864
ExactOnly V20 LCB = 0.490586

P4 geometry_diagnostic_pass = 0
TopK87 precision = 0.068966
TopK87 V = -0.934885
TopK87 LongRisk = 0.848541
TopK87 LDO = 0.011494

P5 generated_preflight_pass = 0
generated_rows = 256
real future h20 LCB = 0.158977

P6 not_run
reason = P5_generated_preflight_not_strong_pass

no_fake/proxy/cpu = 0 / 0 / 0
```

我的独立判断是：

$$
\boxed{\text{v9.8.0 没有证伪“未来轨迹”思想；它证伪的是当前这版 future-trajectory / geometry score / generated preflight 还不能成为 controller。}}
$$

原因是：

1. P3 只列出了 h20 的强正信号，h1/h5 没有 landed rows；如果没有完整 h1/h5/h20/h80/h240 轨迹，就不能说 future trajectory 理论被充分检验。  
2. Core77 和 OldOnly 的 h20 V LCB 很强，ExactOnly 也为正但明显更弱；这更像“有未来轨迹信号，但当前判据没形成可部署规则”。  
3. P4 的 geometry score 失败非常严重，说明当前 architecture-agnostic geometry diagnostic 不是好 selector。  
4. P5 generated preflight 已经出现 h20 正 LCB，这是 generated route 比过去多轮 value-negative 的一个新信号；但它没有通过强门槛，不能直接重开 generated route。  
5. Natural AP0 labeled stream extension 仍没落地，因此 Core77 density 到底够不够还没有回答。

---

# 1. v9.8.0 的真实进展

## 1.1 LDO 分解被再次确认

v9.8.0 继续支持 v9.7.7 的判断：Core77 的 raw LDO 大，主要不是 Core77 本身质量跨数据集崩了，而是 support/backfill 语义造成的惩罚。

$$
LDO_{raw}=0.441558
$$

但：

$$
LDO_{quality}=0.092605
$$

$$
LDO_{support}=0
$$

$$
LDO_{backfill}=0.348953
$$

且：

$$
BackfillShare=0.790276
$$

解释：raw LDO 把三件事混在一起了：

```text
1. Core77 自身质量有没有跨 dataset 下降；
2. 某个 dataset 留出后，Core77 支持数是否不足；
3. 为了凑 coverage 被迫用外部低质动作回填。
```

v9.8.0 继续说明：Core77 的质量下降不是主要问题，最大问题是“好动作数量不够 + 回填动作差”。

## 1.2 natural stream extension 仍然没回答

P2 只有一个 panel 完成，另外三个 panel 未运行，原因是缺少可审计 labeled materializer。

这意味着我们还不知道：如果把自然 AP0 action stream 从 2876 扩到 5k / 10k / 20k，Core-like 好动作密度是否会超过 0.03。

现在只能说：

```text
已有 2876 actions 中，Core-like 动作数量附近仍不确定；
不能用当前样本直接宣称 density sufficient；
也不能说 existing-action route 已经没希望。
```

## 1.3 future trajectory 结果有信号，但不够完整

P3 的 h20 数据：

```text
Core77 V20 LCB = 1.573549
OldOnly V20 LCB = 1.276864
ExactOnly V20 LCB = 0.490586
```

这说明：

```text
Core77 和 OldOnly 在 h20 上确实比 ExactOnly 更强；
ExactOnly 不是完全没 h20 value，但明显弱；
这支持“好动作不是 immediate transfer，而是未来路径效应”的一部分。
```

但是 P3 pass = 0，且 h1/h5 不可用。我的判断是：P3 不能被解读成“未来轨迹理论失败”，只能说：

$$
\boxed{\text{未来轨迹理论还没有被充分支持，也没有形成可训练当下选择的判据。}}
$$

## 1.4 当前 architecture-agnostic geometry score 失败得很清楚

P4 的 TopK87：

```text
precision = 0.068966
V = -0.934885
LongRisk = 0.848541
LDO = 0.011494
```

这是一种典型失败：它几乎能降低 LDO，但选出来的动作质量崩了、value 为负、long-risk 极高。

这说明：

```text
当前 geometry score 可能在捕捉“跨 group 低差异”的区域，
但这个区域不是好动作区域。
```

因此不能继续小修 P4 分数。P4 的失败不是阈值问题，而是“这个几何读数没有对准好训练轨迹”。

## 1.5 generated preflight 有弱正信号，但不能重开 full generated route

P5 generated preflight：

```text
generated_rows = 256
real future h20 LCB = 0.158977
P5 pass = 0
```

这比过去多轮 APG/APGL/APGT/APGU 的 value-negative / high-longrisk 结果更有意思。它说明新的 function-space update smoke 至少不是完全没方向。

但 P5 没有 strong pass，P6 branch-horizon official smoke 未运行。所以不能说 generated route 重新成功。更准确是：

$$
\boxed{\text{generated route 出现 h20 正信号，但还没有 h80/h240、risk、memory、offdiag、runtime 的完整证据。}}
$$

---

# 2. 现在真正卡在哪里

## 2.1 卡在“未来轨迹”没有被完整测量

我们现在说“好动作是把后续训练带到更好轨迹上”，但 v9.8.0 仍主要依赖 canonical h20/h80/h240 outcome rows；h1/h5 没有 landed rows，训练路径中的中间状态也不完整。

真正应该测的是一条曲线：

$$
\Gamma_a(h)=Metric(\Phi_h(\theta_t+\Delta_a))-Metric(\Phi_h(\theta_t))
$$

其中：

```text
h = 1, 5, 20, 80, 240
Metric = CE, NLL, margin, memory loss, hard-tail CEp99, offdiag risk, cover entropy, basis rank, longrisk
```

现在只有少量端点，无法判断“动作把训练轨迹推向更好 basin”还是“动作只在 h20 端点偶然好”。

## 2.2 卡在“几何读数”没有对准 value

P4 说明当前几何读数选出来的动作极差。这个结果很重要：它提醒我们，不能把“architecture-agnostic geometry”变成另一个人工 score。

更高层地说，我们缺的是：

```text
能从函数空间响应中解释 Core77 / OldOnly 为什么好，
而不是用一堆几何 proxy 去重新排序。
```

## 2.3 卡在 natural stream density

Core77 / Core-like 只有 77 个左右，最低 coverage 需要约 87 个。我们一直在补 10 个动作，但补出来的动作会引入 raw LDO、LFO、null、offdiag 或 value 问题。

如果自然 AP0 action stream 扩大后 clean density 能稳定超过 0.03，那么 existing-action route 可以继续。

如果自然 stream 扩大后仍低于 0.03，那么问题不是 selector，而是 action source 本身不够；就必须走 generated-update route。

## 2.4 卡在 generated route 缺完整 horizon evidence

P5 h20 LCB 为正，但 generated route 不能只看 h20。我们以前已经多次看到 h20 好、h240 longrisk 高的情况。

下一步 generated route 必须检查：

```text
h20 value;
h80 not worse;
h240 longrisk;
bad/null;
memory/offdiag;
cover/basis;
runtime cost.
```

如果这些不补齐，P5 h20 正信号只能是 smoke，不是 route reopening。

---

# 3. 本轮反思：不要把“未来轨迹”变成新口号

v9.8.0 最大的风险是：我们把“未来轨迹”当成一个新标签，但没有真正建立数学对象。

应该区分三层：

## 3.1 端点 outcome

```text
h20/h80/h240 最终 outcome 好不好。
```

这是必要的，但它只是结果。

## 3.2 路径形状

```text
从 h1 到 h240，loss、memory、cover、risk 是怎么变化的。
```

这才接近“训练轨迹”。

## 3.3 训练动力学机制

```text
这个动作为什么改变后续路径？
它是否改变了后续 AdamW 的有效方向？
它是否降低了 hard-tail？
它是否保住了旧 family？
它是否让局部函数区域更稳定？
```

只有第三层才可能变成通用优化原则。

所以 v9.8.1 不应该继续问：

```text
P3 future_trajectory_pass 能不能调过？
P4 geometry score threshold 能不能调过？
P5 generated h20 能不能再好一点？
```

而应该问：

```text
未来训练轨迹到底由哪些可训练当下测量的函数空间响应决定？
这些响应是否跨 dataset / template / family 稳定？
如果稳定，能否直接生成满足这些响应的更新？
```

---

# 4. v9.8.1 总体目标

v9.8.1 的总目标是：

$$
\boxed{\text{把“未来轨迹改善”从一句解释，变成可测量、可验证、可生成、可部署的 functional update 原则。}}
$$

本轮不是小修。

本轮并行推进四条线：

```text
A 线：Future Trajectory Causal Audit
  查 Core77 / OldOnly / ExactOnly / Random / Generated 在完整 h1/h5/h20/h80/h240 路径上的差异。

B 线：Architecture-Agnostic Function Geometry
  不依赖 KAN basis，构造函数空间响应指标，判断它们是否能解释好动作。

C 线：Natural AP0 Stream Density
  真正落地 labeled natural stream extension，判断 existing-action route 是否有足够动作密度。

D 线：Geometry-Adaptive Generated Update
  只有在 A/B 给出机制线索后，有限重开 generated route；否则继续停止 blind primitive。
```

---

# 5. 核心假设

## H1：Core77 / OldOnly 的优势来自未来训练路径，而不是 immediate CE response

直观解释：好动作不一定让当前一步 loss 降最多，但它会让后面训练更稳、更安全、更有收益。

数学形式：

$$
AUV(a)=\sum_{h\in\{1,5,20,80,240\}} w_h V_h(a)
$$

若 H1 成立，应看到：

```text
Core77 / OldOnly 的 AUV LCB > 0；
ExactOnly / RandomMatched 的 AUV LCB <= 0；
Core77 / OldOnly 的 longrisk / memory / offdiag 更低；
Core77 / OldOnly 不一定在 h1 最强，但 h20/h80/h240 更稳定。
```

## H2：当前 P4 几何读数失败，是因为它没有测函数空间传播，而不是因为几何方向无意义

若 H2 成立，应看到：

```text
function-space response features 能解释 Core77 / OldOnly；
KAN-specific basis features 只是辅助，不是必要条件；
这些指标对 MLP sentinel 也能计算，至少不依赖 KAN basis 名称。
```

## H3：existing-action route 的根本边界取决于自然动作密度

若 H3 成立，扩展 action stream 后：

```text
Core-like rate 稳定 >= 0.03；
或者 Core-like count 足够支撑 accepted_count >= 87；
且 support-aware/raw gate 冲突减弱。
```

若 H3 失败，说明当前 action source 不足，应停止 existing-action route 作为主线。

## H4：generated route 只有在 h20/h80/h240 全路径上过线才允许重开

P5 h20 正信号不够。generated update 必须满足：

$$
V_{20}^{LCB}>0
$$

$$
V_{80}^{LCB}\ge -\epsilon
$$

$$
LongRisk_{240}^{UCB}\le 0.05
$$

$$
Bad^{UCB}\le 0.05
$$

$$
Null^{UCB}\le 0.15
$$

$$
MemoryFail^{UCB}\le 0.05
$$

$$
OffdiagFail^{UCB}\le 0.05
$$

否则仍然停止 generated route。

---

# 6. 详细实验计划

## P0. v9.8.0 边界复现与字段合法性审计

### 目标

确认本轮没有跳过 v9.8.0 的失败边界。

### 必须复现

```text
route_v9800 = RouteD-FutureTrajectoryUnsupported
system_legal_controller_pass = 0
generated_route_status = stopped_no_official_generated_branch_horizon_pass
no_fake/proxy/cpu = 0 / 0 / 0
```

### 记录指标

```text
source_route
field_green_count
field_yellow_count
field_red_count
old_table_used
future_outcome_used_in_controller
proxy_row_count
fake_row_count
cpu_offload_count
```

### 通过标准

```text
boundary_reproduced = 1
field_red_count = 0
no_fake = 1
no_proxy = 1
controller_not_open_before_gate = 1
```

---

## P1. Natural AP0 Labeled Stream Extension Materializer

### 目标

解决 v9.8.0 / v9.7.7 反复卡住的问题：自然动作扩流没有可审计 labeled materializer。

### 实验设计

构造三个扩展 panel：

```text
Panel B: 5000 natural AP0 actions
Panel C: 10000 natural AP0 actions
Panel D: 20000 natural AP0 actions
```

每个 action 必须有：

```text
action_id
event_id
dataset
seed
template_id
family
step_bucket
payload_hash
action_apply_error
branch_horizon outcomes for h20/h80/h240
RealFunctional branch
AdamWOnly branch
AdamWParallel branch
bestLR branch
NoOp branch
Random branch
ShuffledPayload branch
memory/offdiag labels
bad/null/longrisk labels
```

### 记录指标

```text
panel_action_count
panel_branch_horizon_row_count
completion_rate
rows_per_sec
label_exclusivity_violations
missing_hash_count
duplicate_action_id_count
CoreLike_count
CoreLike_rate
CoreLike_Wilson_LCB/UCB
GradeAB_count/rate
ValuePositiveNoLongRisk_count/rate
MemoryOffdiagCore_count/rate
per_dataset_core_rate
per_template_core_rate
```

### 判断标准

Strong pass：

$$
CoreLikeRate^{LCB} \ge 0.03
$$

且：

```text
completion_rate = 1.0
quality_audit_pass = 1
per_dataset_core_rate_min >= 0.02
```

Weak pass：

$$
CoreLikeRate^{mean} \ge 0.03
$$

且 Wilson CI 不强烈低于 0.03。

Fail：

$$
CoreLikeRate^{UCB}<0.03
$$

或 materializer 不完整。

### 可视化

```text
CoreLike rate vs panel size 曲线
Wilson CI 带状图
per-dataset density 条形图
per-template density heatmap
CoreLike / GradeAB / ValuePositiveNoLongRisk density 对比图
```

---

## P2. Future Trajectory Full Path Materializer

### 目标

真正测量“这个动作是否把后续训练带到更好路径”，而不是只看 h20/h80/h240 的端点。

### 动作组

```text
G1 Core77
G2 OldOnly
G3 ExactOnly
G4 RandomMatched
G5 E4-DatasetBlindScoreNormalizedTop10 / full87 diagnostic
G6 v9.8.0 generated preflight actions
```

RandomMatched 必须按以下字段匹配：

```text
dataset
seed
family
step_bucket
action_norm_bucket
payload_norm_bucket
template_id if available
```

### horizons

```text
h = 1, 5, 20, 80, 240
```

### 每个 horizon 记录

```text
CE_delta
NLL_delta
margin_delta
V_ctrl
bad_event
null_event
longrisk
memory_buffer_loss_delta
old_family_loss_delta
old_stratum_margin_delta
hard_tail_CEp99_delta
offdiag_fail
cover_entropy_delta
basis_effective_rank_delta
jacobian_proxy_delta
payload_apply_cost
step_runtime_delta
```

### 核心统计

定义：

$$
AUV(a)=\sum_h w_h V_h(a)
$$

默认：

$$
w_{1}=0.05,\quad w_{5}=0.10,\quad w_{20}=0.25,\quad w_{80}=0.30,\quad w_{240}=0.30
$$

也记录不加权版本，避免人为权重造成假象。

### 判断标准

Future trajectory 支持成立，当：

```text
Core77 AUV_LCB > 0
OldOnly AUV_LCB > 0
ExactOnly AUV_LCB <= 0 or clearly lower than OldOnly
RandomMatched AUV_LCB <= 0
Core77 / OldOnly h240 longrisk UCB <= 0.05
Core77 / OldOnly memory/offdiag UCB <= 0.05
OldRank score 与 AUV 的 Spearman > ExactTransfer score 与 AUV 的 Spearman
```

若 Core77/OldOnly 只在 h20 好，h80/h240 不稳，则 future trajectory hypothesis fail。

### 可视化

```text
group × horizon V 曲线
horizon risk 曲线
AUV 分布 violin plot
OldOnly vs ExactOnly response scatter
OldRank score vs AUV scatter
ExactTransfer score vs AUV scatter
memory/offdiag fail by horizon
cover entropy / basis rank by horizon
```

---

## P3. Future Trajectory Mechanism Decomposition

### 目标

如果 P2 显示 Core77 / OldOnly 未来轨迹好，P3 必须解释“为什么好”。

### 机制候选

```text
M1 后续 AdamW alignment 改善
M2 hard-tail 下降
M3 old-family memory 保持
M4 offdiag risk 下降
M5 cover entropy 不塌
M6 basis rank 不塌
M7 action norm 小但方向有效
M8 function displacement 平滑
M9 gradient conflict relief
M10 exact transfer 不强但 future path strong
```

### 记录指标

```text
mechanism_id
matched_pair_count
effect_size
AUC_AUV_positive
TopK87_precision
TopK87_AUV_LCB
TopK87_longrisk_UCB
LDO/LSO/LTO/LFO_drop
per_dataset_effect_size
per_template_effect_size
```

### 判断标准

机制可用，当：

```text
effect_size >= 0.8
TopK87 precision >= 0.75
AUV_LCB > 0
longrisk_UCB <= 0.05
max leaveout drop <= 0.10
使用字段全部 commit-time legal
```

若只在 pooled 上成立、leaveout 崩，则保留 diagnostic，不转 controller。

---

## P4. Architecture-Agnostic Function Geometry Ledger v2

### 目标

构建不依赖 KAN 当前 basis 的函数空间几何指标。

### 核心原则

不要把 KAN basis 当作唯一理论基础。对任意模型，都可以定义：

```text
output response
Jacobian-vector response
function displacement
memory response
hard-tail response
trajectory displacement
cost
```

### 每个 action 记录

```text
function_displacement_norm_current_batch
function_displacement_norm_memory
function_displacement_norm_hard_tail
per_example_response_mean
per_example_response_std
response_SNR
Jacobian_vector_norm
AdamW_alignment
gradient_conflict_score
memory_response_LCB
hard_tail_response_LCB
future_path_response_proxy
payload_apply_ms
feature_compute_ms
```

KAN-only 诊断作为附加列：

```text
basis_effective_rank_delta
cover_entropy_delta
edge_block_sparsity
edge_locality
basis_activation_entropy
```

这些列不能作为 architecture-agnostic score 的必要条件，只能用于解释 KAN-specific 行为。

### 判断标准

Architecture-agnostic geometry pass：

```text
至少一个不依赖 KAN-specific basis 的 score 满足：
TopK87 precision >= 0.75
AUV_LCB > 0
longrisk_UCB <= 0.05
LDO/LSO/LTO drop <= 0.10
```

如果 KAN-only 指标能解释，而 architecture-agnostic 指标完全不能，则说明当前机制可能依赖 KAN basis；这不是坏事，但它不能作为跨架构 optimizer 原则。

### 可视化

```text
function displacement vs AUV scatter
response SNR vs AUV scatter
memory response vs longrisk scatter
Jacobian-vector norm vs value/risk scatter
KAN basis rank vs AUV side-by-side
architecture-agnostic score 与 KAN-specific score 对比
```

---

## P5. Existing-Action Minimal Controller Boundary

### 目标

只有 P1/P2/P3/P4 给出足够证据时，才尝试 existing-action minimal controller。

### Controller 不允许使用

```text
dataset_name
future outcome
GradeAB label
V_integrated label
risk_score derived from future branch-horizon
old table label
validation/test metric
```

### Controller 允许使用

```text
commit-time function-space response
memory response
hard-tail response
AdamW alignment
action norm/cost
template/family only as balance/audit, not dispatch
```

### 必须通过

$$
N_{accept}\ge 87
$$

$$
Precision_{GradeAB}\ge 0.75
$$

$$
AUV^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05
$$

$$
Bad^{UCB}\le 0.05
$$

$$
Null^{UCB}\le 0.15
$$

$$
MemoryFail^{UCB}\le 0.05
$$

$$
OffdiagFail^{UCB}\le 0.05
$$

$$
\max(LDO,LSO,LTO,LFO)\le0.10
$$

若 support-aware pass 但 raw gate fail，必须进入 P8，不得直接 official。

---

## P6. Geometry-Adaptive Generated Update Preflight v2

### 目标

检验 v9.8.0 P5 的 h20 正信号是否能扩展成真正可用的 generated update。

### 前提

只有 P2/P3 给出 future trajectory 机制线索时运行。否则 generated route 继续 stopped。

### 生成方式限制

不再命名一堆 APG 小变体。只允许两类生成目标：

```text
G-Traj：直接最大化 future trajectory proxy；
G-Safe：在 memory/offdiag/longrisk hard gate 下最小改变函数空间。
```

形式：

$$
\Delta^*=\arg\max_{\Delta\in\mathcal{S}} \widehat{AUV}(\Delta)
$$

约束：

$$
MemoryHarm(\Delta)\le \tau_M
$$

$$
OffdiagRisk(\Delta)\le \tau_O
$$

$$
LongRiskProxy(\Delta)\le \tau_L
$$

$$
Cost(\Delta)\le C_{max}
$$

其中 $\mathcal{S}$ 是小更新子空间，可以是：

```text
last-layer function-space subspace
low-rank output-response subspace
edge/basis block subspace
AdamW-orthogonal residual subspace
```

注意：这些是子空间约束，不是 dataset-specific 规则。

### Preflight 阶段

```text
1 action preflight
8 action preflight
32 action preflight
256 action smoke
512 action official smoke only if 256 pass
```

### 256-action pass 标准

```text
V20_LCB > 0
V80_LCB >= -0.02
h240 longrisk_UCB <= 0.10
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.10
new_positive_rate >= 0.10
longrisk_created_rate <= 0.10
```

若不满足，不跑 P7 generated official branch-horizon。

---

## P7. Generated Branch-Horizon Official Smoke

### 目标

如果 P6 过，真实 materialize generated branch-horizon outcomes。

### 记录

```text
generated_action_count
branch_horizon_rows
completion_rate
rows_per_sec
V20/V80/V240
AUV
longrisk
bad/null
memory/offdiag
controls beaten rate
shuffled payload control
negative control
runtime cost
```

### 通过标准

同 P5，但增加 shuffled control：

```text
RealGenerated beats ShuffledGenerated
RealGenerated beats NoOp / Random
RealGenerated beats AdamWParallel or bestLR at sufficient rate
```

---

## P8. Raw LDO vs Support-Aware Gate Resolution

### 目标

解决 v9.7.7 / v9.8.0 反复出现的 gate 冲突。

### 问题

如果 online controller 不会接受 backfill 动作，那么 raw LDO 中 backfill 惩罚可能不是 online failure。  
但如果为了 coverage 必须 backfill，那么 raw LDO 就是动作密度不足的真实失败。

### 实验

构造三种 evaluation：

```text
E_raw: legacy raw LDO，允许回填；
E_no_backfill: 只评估 controller 自己会接受的动作，不回填；
E_density_required: 要求每个 leaveout split 中 accepted_count >= 87，不足则 fail。
```

### 记录

```text
accepted_count_per_split
precision_per_split
V_LCB_per_split
longrisk_per_split
backfill_count
backfill_precision
support_adjusted_LDO
raw_LDO
no_backfill_LDO
density_fail_count
```

### 判断

允许进入 official 修改 gate 讨论，必须满足：

```text
E_no_backfill pass = 1
E_density_required pass = 1
backfill_precision = 0 or backfill shown irrelevant to online controller
natural stream density pass = 1
```

否则 raw gate 仍保留。

---

## P9. Runtime Boundary

只有 P5 或 P7 选中 controller / generated update 后运行。

### 记录

```text
feature_compute_ms_q50/q90
controller_score_ms_q50/q90
payload_apply_ms_q50/q90
kernel_launch_count
sync_count
step_ratio_q90
memory_ratio
active_step_count
empty_step_launch_count
```

### 通过标准

$$
StepRatio_{q90}\le1.50
$$

$$
MemoryRatio\le1.05
$$

且：

```text
no CPU offload
no fake/proxy row
no offline materializer in timed path
```

---

## P10. Official Paired Replay Boundary

只有 P5/P7 + P9 过后打开。

### branches

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

### pass 标准

```text
RealFunctional beats AdamWParallel rate high
RealFunctional beats bestLR rate high
RealFunctional beats NoOp/Random/ShuffledPayload
AUV_LCB > 0
longrisk/bad/null controlled
leaveout stable
```

---

## P11. Short/Full Training Boundary

只有 P10 过后打开。

### 记录

```text
test_acc
val_loss_auc_step
val_loss_auc_time
time_to_target
steps_to_target
NLL
ECE
CEp99
margin_p10
hard-stratum acc
memory retained acc
forgetting
forward transfer
backward transfer
step_ratio_q90
memory_ratio
```

这些不能用于调 controller，只能用于最终验证。

---

# 7. 并行执行策略

v9.8.1 必须并行，不允许一轮只发现一个 blocker。

```text
Lane A: P1 natural stream extension
Lane B: P2/P3 future trajectory audit
Lane C: P4 architecture-agnostic geometry ledger
Lane D: P6/P7 generated update, only if P2/P3 provide evidence
Lane E: P8 LDO gate resolution
Lane F: P0/no-fake/contract/Base-Acc Sentinel
```

执行顺序：

```text
Day 0 preflight:
  P0 + P1 100-action preflight + P2 small trajectory preflight + P4 feature preflight

Day 1 main:
  P1 full panels + P2 full groups + P4 full ledger

Day 2 conditional:
  P3 mechanism + P6 generated preflight + P8 gate resolution

Day 3 only if gates pass:
  P5/P7 controller + P9 runtime
```

---

# 8. 快速停止规则

## Stop future trajectory route

如果：

```text
Core77/OldOnly AUV_LCB <= 0
or ExactOnly AUV >= OldOnly AUV
or h240 longrisk high for Core77/OldOnly
```

则停止“future trajectory explains good actions”路线，回到 density / outcome-only boundary。

## Stop architecture-agnostic geometry score

如果 P4 出现：

```text
TopK87 precision < 0.30
or V_LCB < 0
or longrisk_UCB > 0.30
```

则不再调这个 score，必须重做函数空间特征定义。

## Stop generated route

如果 P6 256-action smoke 满足任一：

```text
V20_LCB <= 0
or V80_LCB < -0.05
or h240 longrisk_UCB > 0.20
or longrisk_created_rate > 0.20
```

则 generated route 继续 stopped，不跑 P7。

## Stop existing-action route

如果 P1 显示：

$$
CoreLikeRate^{UCB}<0.03
$$

且 P4/P6 都没有可用机制，则 existing-action route 暂停，转向更底层 update-rule design。

---

# 9. 最终 route 决策

## R1-NaturalDensitySufficient_ControllerCandidate

```text
P1 density pass
P5 controller pass
P9 runtime pass
```

下一步：paired replay。

## R2-FutureTrajectoryMechanismFound

```text
P2/P3 support future trajectory hypothesis
P4 finds legal architecture-agnostic signal
```

下一步：minimal controller or generated solver。

## R3-FutureTrajectoryEndpointOnly_NoMechanism

```text
Core77/OldOnly endpoints good
but path/mechanism not explained
```

下一步：trajectory path instrumentation redesign。

## R4-ArchitectureAgnosticGeometryFail

```text
P4 score TopK low-quality/high-risk
```

下一步：stop geometry score patching。

## R5-GeneratedH20PositiveButLongHorizonFail

```text
P6 h20 positive
but h80/h240/risk fail
```

下一步：do not reopen generated route; redesign objective。

## R6-ActionDensityInsufficient

```text
P1 expanded natural stream density still below 0.03
```

下一步：existing-action route cannot be main controller path。

## R7-SystemPass

```text
P5/P7 pass
P9 pass
contract/no-fake pass
```

下一步：P10 paired replay。

---

# 10. v9.8.1 成功标准

最低有效成功不是 full functional success，而是至少满足一个：

## Success A: natural density closure

```text
P1 materializer pass
CoreLikeRate LCB >= 0.03
per-dataset min density acceptable
```

## Success B: future trajectory mechanism closure

```text
Core77/OldOnly future path clearly better than ExactOnly/Random
mechanism feature legal and leaveout stable
```

## Success C: generated route reopening evidence

```text
generated update h20/h80/h240 pass
longrisk/bad/null/memory/offdiag controlled
```

## Success D: actual controller candidate

```text
accepted_count >= 87
precision >= 0.75
AUV_LCB > 0
longrisk/bad/null/memory/offdiag controlled
max leaveout drop <= 0.10
```

若以上都不成立，v9.8.1 仍然不是失败回退，但必须明确输出：

```text
future trajectory theory unsupported;
or existing action density insufficient;
or geometry score misaligned;
or generated update objective insufficient.
```

---

# 11. 本轮必须落盘 artifacts

```text
p0_boundary_reproduction_v9810.csv
p0_field_legality_audit_v9810.csv
p1_natural_ap0_stream_extension_materializer_v9810.csv
p1_density_curve_v9810.csv
p2_future_trajectory_full_path_v9810.csv
p2_group_trajectory_summary_v9810.csv
p3_future_trajectory_mechanism_decomposition_v9810.csv
p4_architecture_agnostic_function_geometry_ledger_v9810.csv
p4_geometry_score_evaluation_v9810.csv
p5_existing_action_minimal_controller_boundary_v9810.csv
p6_generated_update_preflight_v2_v9810.csv
p7_generated_branch_horizon_smoke_v9810.csv
p8_ldo_gate_resolution_v9810.csv
p9_runtime_boundary_v9810.csv
p10_paired_replay_boundary_v9810.csv
p11_short_full_boundary_v9810.csv
base_acc_sentinel_v9810.csv
no_fake_audit_v9810.csv
contract_audit_v9810.csv
route_decision_v9810.json
run_manifest_v9810.json
failure_taxonomy_v9810.csv
```

---

# 12. 最终判断

v9.8.0 的结果告诉我们：

```text
1. Core77 raw LDO 仍主要是 support/backfill 问题；
2. natural labeled stream extension 仍是未落地硬 blocker；
3. future trajectory 有 h20 正信号，但不够完整；
4. architecture-agnostic geometry score 当前严重选错；
5. generated update 有 h20 正信号，但没有长程安全证据；
6. controller/runtime/paired replay/short-full 仍不能打开。
```

因此 v9.8.1 不能继续做人工 score patch，也不能把 P5 h20 正信号夸大成 generated route pass。

下一轮真正要做的是：

$$
\boxed{\text{完整测量未来训练路径，建立不依赖 KAN basis 的函数空间几何读数，补齐自然动作密度，并只在机制证据成立时重开 generated update。}}
$$

这才对应项目更高层的目标：

```text
不是在当前数据集上找动作；
不是围绕 KAN 当前 basis 调规则；
而是让优化器自适应理解网络函数空间的几何，选择或生成能把后续训练带到更好轨迹的小参数改动。
```
