# DG-KAN v9.8.3 四线并行：未来轨迹机制、自然动作扩流、合法机制发现、几何自适应更新

> 本文件基于 v9.8.2 的真实执行结果制定。目标不是继续小修分数、阈值、APG 变体或单个 target，而是把当前最关键的问题拆成四条可以并行推进、互相验证、互相否定的实验线。
>
> 本计划公式全部使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 0. 一句话判断

v9.8.2 有进展，但不是 functional success。

它真正完成的是：

```text
h1 / h5 / h20 / h80 / h240 的完整未来路径，终于在 selected groups 上真实落盘。
```

它真正暴露的是：

```text
好动作的未来路径确实比坏动作更像“有用训练轨迹”，
但我们还没有训练当下可用的机制去解释、识别或生成这种动作。
```

所以当前项目状态不是：

```text
没有好动作；
没有未来轨迹信号；
KAN base 已经失败；
functional update 已经被证伪。
```

更准确是：

$$
\boxed{
\text{局部好动作存在，未来路径信号存在；但合法机制、自然动作密度、生成规则和 controller 都还没闭合。}
}
$$

---

# 1. 先解释本轮四条线分别是什么

从 v9.8.0 开始，我们不再只问“哪个动作事后 outcome 好”，而是分成四条线并行验证。

## 1.1 A 线：未来训练轨迹

这条线问：

```text
一个动作加进去以后，后面训练路是不是变好了？
```

不是只看当前一步 loss，而是看：

```text
h = 1, 5, 20, 80, 240
```

这些未来时间点上，动作是否让模型更稳、更低风险、更不忘旧知识。

本轮对应 v9.8.2 的 P1。

## 1.2 B 线：训练当下能不能解释好动作

这条线问：

```text
既然 Core77 / OldOnly 这些动作未来表现好，
训练当下有没有合法字段能解释它们为什么好？
```

这里的“合法字段”指：

```text
训练当下能看到的信息；
不能用未来 outcome；
不能用 dataset name 做分支；
不能用 test / validation；
不能用事后标签。
```

本轮对应 v9.8.2 的 P2 和 P5。

## 1.3 C 线：自然动作池够不够大

这条线问：

```text
现有 2876 个自然 AP0 动作里只有 77 个 Core-like 好动作，
那如果自然动作池扩大到 5000 / 10000 / 20000，
好动作比例会不会稳定超过最低 coverage 线？
```

如果自然动作池里好动作密度够，existing-action 路线还有希望。

如果自然动作池扩大后好动作密度仍低，就说明问题不是 selector，而是动作来源本身不够。

本轮对应 v9.8.2 的 P3。

## 1.4 D 线：能不能直接生成好动作

这条线问：

```text
如果自然动作池不够，能不能让优化器直接生成一种小参数改动，
让模型后续训练轨迹变好？
```

注意，D 线不能再盲目做 APG/APGT/APGU 小变体。只有当 A/B 线说明“好动作的机制是什么”之后，D 线才允许重开。

本轮对应 v9.8.2 的 P7。

---

# 2. v9.8.2 的四线进展

## 2.1 A 线：未来轨迹终于真实补齐，这是本轮最大进展

v9.8.2 真实 materialize 了 selected groups 的 full future path：

```text
selected actions = 242
expected rows = 7260
actual rows = 7260
horizons = 1, 5, 20, 80, 240
missing horizon = 0
NaN / Inf = 0 / 0
no-transform sanity = 1
```

这是很实的推进。之前 h1/h5 经常 unavailable，现在不是 unavailable，而是真实 replay rows。

四组动作的 AUV LCB 是：

```text
Core77        = 2.80616189380043
OldOnly       = 0.9993033070897774
ExactOnly     = 1.5125686938611076
RandomMatched = 1.7813876735867753
```

这里不能只看 AUV 大小。要看完整路径形状。

Core77 的未来路径最干净：

```text
h1   V LCB = 0.0882, longrisk = 0, memory = 0, offdiag = 0
h5   V LCB = 0.4709, longrisk = 0, memory = 0, offdiag = 0
h20  V LCB = 1.5735, longrisk = 0, memory = 0, offdiag = 0
h80  V LCB = 3.1200, longrisk = 0, memory = 0, offdiag = 0
h240 V LCB = 4.7509, longrisk UCB = 0.0383, memory = 0, offdiag = 0
```

这个形状非常重要。它说明 Core77 不是只在某个 horizon 上偶然好，而是从 h1 到 h240 持续改善，并且长期风险接近 gate 线以内。

ExactOnly 也不完全坏：

```text
h240 V LCB = 2.9221
longrisk = 0
memory = 0
offdiag = 0
```

但它的问题是：它在之前 outcome 上并不是高质量动作集合，且与 OldRank / Core77 目标不一致。它可能是“安全但不够对应我们真正想要的 value/control 优势”的动作。

RandomMatched 的 V 也会增长，但它的风险很高：

```text
h240 V LCB = 3.5923
h240 longrisk UCB = 0.8485
memory UCB = 0.9434
offdiag UCB = 0.9605
```

这说明一个关键事实：

$$
\boxed{
\text{V 或 AUV 大，不等于好动作；必须同时看 longrisk、memory、offdiag。}
}
$$

A 线结论：

```text
未来路径 materializer 成功；
Core77 的完整路径强且干净；
但单纯未来路径强弱还不能直接变成 controller。
```

---

## 2.2 B 线：没有找到训练当下合法机制，这是本轮主 blocker

v9.8.2 的 P2 做了 mechanism contrast。结果很关键。

最强的 `M_signal`：

```text
TopK87 precision = 0.8735632183908046
V LCB = 0.14061570456586386
AUV LCB = 2.7227479106902694
longrisk UCB = 0.0
legal = 0
pass = 0
```

这说明它能复现强好动作区域，但它不是训练当下可用的 official 机制。换句话说：

```text
它像“答案附近的信号”；
不是在线优化器可以合法使用的信号。
```

`M_memory` 更合法：

```text
legal = 1
TopK87 precision = 0.41379310344827586
V LCB = -0.11427822648412846
AUV LCB = 2.716555586178205
longrisk UCB = 0.0
pass = 0
```

它能压 longrisk，但 value 选不出来。这个结果延续了之前多轮结论：

```text
memory/offdiag 更像安全边界；
不是直接产生正收益的充分机制。
```

P5 把几何机制拆成几类：

```text
G_func      precision = 0.4598, V LCB = -0.4021, longrisk = 0.3950, legal = 0
G_path      precision = 0.4483, V LCB = -0.4358, longrisk = 0.3698, legal = 0
G_memory    precision = 0.4138, V LCB = -0.1143, longrisk = 0.0,    legal = 1
G_stability precision = 0.4598, V LCB = -0.3758, longrisk = 0.3570, legal = 0
G_cost      precision = 0.2299, V LCB = -0.2198, longrisk = 0.2104, legal = 1
```

没有一个 legal family 同时满足 precision、value、risk。说明：

$$
\boxed{
\text{我们还没有找到“训练当下可见”的好动作机制。}
}
$$

这不是小修阈值的问题。它说明目前的合法几何读数仍然没有捕捉到 Core77 / OldOnly 的真正原因。

---

## 2.3 C 线：自然动作扩流仍然没落地，这是第二 blocker

P3 仍然只有现有 2876 panel：

```text
Panel-target-2876:
  labeled = 2876
  CoreLike rate = 0.026773296244784424
  LCB = 0.021475240123524635
  UCB = 0.033333885489495195
```

但是：

```text
Panel-target-5000   not_run
Panel-target-10000  not_run
Panel-target-20000  not_run
reason = no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9820
```

这意味着我们仍然不知道：

```text
如果自然动作池扩大，好动作密度会不会稳定超过 0.03？
```

这是非常关键的问题。因为现在 Core-like rate 的 Wilson interval 跨过 0.03：

$$
0.0215 \le p_{CoreLike} \le 0.0333.
$$

这说明当前 2876 个动作不足以下定论。

如果自然扩流后 $p_{CoreLike}$ 稳定超过 0.03，existing-action route 可能继续。

如果自然扩流后 $p_{CoreLike}$ 稳定低于 0.03，那么就算 selector 再好，也没有足够动作支撑 coverage。那就必须转向生成新动作。

C 线结论：

```text
这是下一轮必须优先落地的工程 blocker；
不能再用 2876 panel 反复推断 density。
```

---

## 2.4 D 线：generated route 继续停止是正确的

v9.8.2 的 generated route 是：

```text
generated_route_status = stopped_no_legal_mechanism_or_generated_branch_horizon
P7 generated = not_run
reason = P1_P2_P5_mechanism_not_established_generated_route_stopped
```

这个选择是对的。

因为 P2 没找到合法机制，P5 没有 legal geometry family pass。如果这个时候重开 APG/APGU/APGX，只会重新进入：

```text
生成 512 个动作；
branch-horizon 落盘；
然后 value-negative / high-longrisk；
再换名字继续。
```

D 线结论：

```text
不要盲目生成；
只有 A/B 线找到机制，或 C 线证明自然动作密度不足但机制明确，才允许 generated route 有限重开。
```

---

# 3. 为什么你会感觉非常慢

你的感觉是对的。

因为到 v9.8.2 为止，还是没有打开：

```text
controller
selected runtime
official paired replay
short/full training
```

从“系统能力上涨”的角度看，确实非常慢。

但这几轮不是没有推进，而是在拆假希望：

```text
StableAccept 强行修补 -> 失败；
outcome-derived certificate -> 非法；
payload pocket -> 混杂；
exact one-step transfer -> 准但不选好动作；
windowed transfer -> 选不出好动作；
architecture-agnostic geometry score -> 选到高风险动作；
future path endpoint -> 有信号但机制不合法；
Core77 support-aware LDO -> 有希望但 density/gate 未闭合。
```

慢的根本原因是：

$$
\boxed{
\text{我们已经能事后看到好动作，但还没有训练当下可解释、可生成、可部署的机制。}
}
$$

这不是再调一个 threshold 能解决的问题。

---

# 4. 当前最重要的发现

## 4.1 好动作不是当前一步 loss 下降

v9.7.2 已经证明 exact one-step transfer 测得很准，但选不出好动作。v9.8.2 又进一步说明，真正好的 Core77 动作在未来路径上持续改善，而不只是 h1 立即最大。

因此：

$$
\boxed{
\text{functional update 不是 one-step loss descent，而是 future trajectory steering。}
}
$$

直白说：

```text
不是找当前这一步 loss 降最多的动作；
是找能把后续训练带到更好路线上的小参数改动。
```

## 4.2 大函数响应不是好几何

v9.8.1 已经显示 function response diagnostic AUV 很高但 TopK precision 很低、longrisk 很高。v9.8.2 的 RandomMatched 也说明：V/AUV 可以增长，但 memory/offdiag/longrisk 全坏。

所以：

$$
\boxed{
\text{函数输出变动大，不等于训练几何变好。}
}
$$

## 4.3 Memory / offdiag 是安全条件，不是充分机制

Memory/offdiag 多轮都能压风险，但单独拿来选动作会造成 value 负或 precision 不够。

因此：

```text
memory/offdiag 应该作为硬安全门；
不能作为唯一 score。
```

## 4.4 好动作可能是“未来路径稳定区”，不是“某个可见单特征”

Core77 路径显示：

```text
从 h1 到 h240 持续变好；
长期风险低；
memory/offdiag 不坏。
```

但 P2/P5 找不到合法机制。说明好动作可能是多个因素共同作用，不容易被单个字段解释。

下一步不能继续找单特征，而要做机制分解：

```text
它改变了后续 AdamW 方向吗？
它减少了 hard-tail 吗？
它保护了旧 family 吗？
它让局部函数片区更稳定了吗？
它让未来 gradient variance 更低了吗？
```

---

# 5. v9.8.3 总体目标

v9.8.3 不再小修 v9.8.2 的 P2/P5 分数，也不重开盲生成。

本轮总体目标是：

$$
\boxed{
\text{把“未来路径好”拆成可验证的训练动力学机制，并同时补上自然动作扩流。}
}
$$

四条线并行。

```text
A 线：未来路径机制分解
B 线：训练当下合法机制发现
C 线：自然 AP0 动作扩流
D 线：生成路线是否允许重开
```

---

# 6. v9.8.3 实验计划

---

# P0：复现 v9.8.2 边界

## 目标

确认 v9.8.2 的 full future path、mechanism fail、density missing、generated stop、no-fake boundary 没有回退。

## 必须记录

```text
source_route_v9820
P1_full_future_path_rows_expected
P1_full_future_path_rows_actual
P1_weak_pass
P2_mechanism_pass
P3_density_strong_pass
P5_geometry_weak_pass
controller_pass
generated_pass
runtime_pass
fake/proxy/cpu
```

## 通过标准

```text
boundary_reproduced = 1
fake/proxy/cpu = 0 / 0 / 0
```

如果 P0 不过，停止所有后续阶段。

---

# P1：未来路径机制分解 v2

## 目标

不是再证明 Core77 好，而是解释 Core77 为什么好。

本阶段比较四组：

```text
Core77：最干净的 existing-action 好动作。
OldOnly：OldRank 喜欢、ExactTransfer 不喜欢，但历史上结果好的动作。
ExactOnly：ExactTransfer 喜欢、OldRank 不喜欢，但历史上结果差的动作。
RandomMatched：按 dataset / family / template / step 匹配的随机动作。
```

## 核心假设

$$
H_{P1}:
\text{Core77 的优势来自未来训练路径的稳定改善，而不是 h1 immediate loss 下降。}
$$

## 必须记录

每个 action、每个 horizon 记录：

```text
action_id
group
dataset_id_for_diagnostic_only
seed
family
template_id
step_bucket
horizon
V
Vctrl
CE_delta
NLL_delta
margin_delta
CEp99_delta
hard_tail_loss_delta
old_family_loss_delta
old_stratum_loss_delta
memory_buffer_loss_delta
longrisk
bad_event
null_event
memory_fail
offdiag_fail
cover_entropy_delta
basis_effective_rank_delta
curvature_proxy_delta
AdamW_alignment
action_norm
payload_norm
OldRank_score
ExactTransfer_score
WT80_score
```

## 新增路径指标

对每个 action 定义：

$$
AUV(a)=\sum_{h\in\{1,5,20,80,240\}} w_h V_h(a)
$$

其中默认：

$$
w_1=0.05,
\quad
w_5=0.10,
\quad
w_{20}=0.25,
\quad
w_{80}=0.30,
\quad
w_{240}=0.30.
$$

定义路径稳定度：

$$
Stability(a)=
\mathbb{1}[V_{20}>0]
\cdot
\mathbb{1}[V_{80}>0]
\cdot
\mathbb{1}[LongRisk_{240}=0]
\cdot
\mathbb{1}[MemoryFail=0]
\cdot
\mathbb{1}[OffdiagFail=0].
$$

定义 immediate mismatch：

$$
Mismatch(a)=
\mathbb{1}[V_1(a)\le 0]
\cdot
\mathbb{1}[AUV(a)>0].
$$

这个指标用于识别：

```text
当前一步不强，但未来路径强的动作。
```

## 判断标准

P1 weak pass：

```text
Core77 AUV LCB > ExactOnly AUV LCB + 0.25
Core77 h240 longrisk UCB <= 0.05
Core77 memory/offdiag UCB <= 0.05
```

P1 strong pass：

```text
Core77 AUV LCB > RandomMatched AUV LCB
Core77 h240 longrisk UCB <= 0.05
Core77 memory/offdiag UCB <= 0.05
OldOnly 或 Core77 至少一个 group 显示 future-path-positive / immediate-not-dominant 模式
```

## 可视化

```text
1. group × horizon 的 V 曲线；
2. group × horizon 的 longrisk 曲线；
3. group × horizon 的 memory/offdiag 曲线；
4. AUV vs h1 V 散点图；
5. AUV vs ExactTransfer_score 散点图；
6. Core77 / OldOnly / ExactOnly 的路径雷达图；
7. action-level trajectory spaghetti plot。
```

## 失败解释

如果 Core77 只在 h20/h80/h240 好，而 h1 不解释它：

```text
说明好动作不是 immediate descent，下一步必须估计 future-path mechanism。
```

如果 Core77 与 RandomMatched 的 AUV 差异不显著，但风险差异显著：

```text
说明好动作的核心不是 value 大，而是风险/记忆/稳定性好。
```

---

# P2：合法机制发现 v2

## 目标

把 P1 的未来路径现象压缩成训练当下可用的机制。

这里不允许使用：

```text
future outcome
AUV label
Grade label
V_integrated
risk_score
dataset name branch
validation/test metric
```

## 核心假设

$$
H_{P2}:
\text{存在少量训练当下合法变量，可以解释 Core77/OldOnly 的 future-path advantage。}
$$

## 候选机制族

本轮不再做一个大 score，而是做机制族诊断。

### 机制 1：未来 AdamW 可优化性

问：这个动作是否让后续 AdamW 更容易走？

记录：

```text
post_action_gradient_norm_proxy
post_action_gradient_variance_proxy
AdamW_alignment_after_action
AdamW_conflict_reduction
```

### 机制 2：硬尾样本缓解

问：这个动作是否减少 hard-tail 样本的压力？

记录：

```text
hard_tail_CE_delta_proxy
hard_tail_margin_delta_proxy
hard_tail_gradient_agreement
```

### 机制 3：旧知识保持

问：这个动作是否保护 old family / old stratum？

记录：

```text
old_family_probe_loss_delta
old_family_margin_delta
memory_buffer_CE_delta
memory_buffer_gradient_agreement
```

### 机制 4：局部函数片区稳定

问：这个动作是否没有破坏局部函数空间？

记录：

```text
function_displacement_norm
local_jacobian_response_norm
output_response_rank
cover_entropy_proxy
basis_effective_rank_proxy
```

注意：如果某个指标只能在 KAN basis 上定义，就标记为 architecture-specific；如果可以在任意网络上定义，就标记为 architecture-agnostic。

## 必须记录

```text
mechanism_id
feature_names
feature_legality = green / yellow / red
architecture_specific = 0 / 1
TopK87_precision
V_LCB
AUV_LCB
longrisk_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
feature_cost_q90
```

## 判断标准

P2 weak pass：

```text
TopK87 precision >= 0.75
AUV_LCB > 0
longrisk_UCB <= 0.05
memory_UCB <= 0.05
offdiag_UCB <= 0.05
feature_legality contains no red field
```

P2 strong pass：

```text
P2 weak pass
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
feature_cost_q90 <= budget
```

## 可视化

```text
1. mechanism forest plot；
2. legal vs diagnostic mechanism scatter；
3. TopK precision vs AUV LCB；
4. longrisk UCB vs memory/offdiag UCB；
5. LDO/LSO/LTO drop waterfall；
6. feature cost vs quality plot。
```

## 失败解释

如果 only diagnostic mechanisms pass：

```text
说明我们能事后解释，但不能 online 使用。
```

如果 legal memory mechanism low-risk but value-negative：

```text
说明 memory 是安全门，不是收益源。
```

如果 legal function-response mechanism high-value but high-risk：

```text
说明大函数响应不是好几何，需要加入未来路径约束。
```

---

# P3：自然 AP0 动作扩流 materializer

## 目标

解决持续多轮未解决的问题：自然动作池扩大后，好动作密度到底够不够。

这一步优先级极高。没有它，existing-action route 一直卡在 77 vs 87 的争论上。

## 核心假设

$$
H_{P3}:
\text{如果自然 AP0 动作池扩大，Core-like 动作密度会稳定接近或超过 }0.03.
$$

## 实验 panel

```text
Panel A: 2876 actions, already labeled
Panel B: 5000 actions
Panel C: 10000 actions
Panel D: 20000 actions
```

每个 panel 必须是真实 labeled rows，不允许 fake / proxy。

## Preflight

在跑 5000/10000/20000 前，必须跑：

```text
single-action labeled materializer preflight
16-action labeled materializer preflight
256-action labeled materializer preflight
```

只有三个 preflight 都通过，才跑 panel B/C/D。

## 必须记录

```text
panel_id
target_action_count
actual_labeled_action_count
branch_horizon_expected_rows
branch_horizon_actual_rows
missing_label_count
unresolved_exception_count
rows_per_sec
CoreLike_count
CoreLike_rate
CoreLike_LCB
CoreLike_UCB
GradeAB_count
ValuePositiveNoLongRisk_count
MemoryOffdiagCore_count
per_dataset_rate
per_template_rate
per_family_rate
```

## 判断标准

P3 weak pass：

```text
Panel B completed
CoreLike_LCB >= 0.025
```

P3 strong pass：

```text
Panel C or D completed
CoreLike_LCB >= 0.03
min_per_dataset_CoreLike_LCB >= 0.02
missing_label_count = 0
fake/proxy = 0
```

## 可视化

```text
1. CoreLike density vs action count；
2. Wilson CI density curve；
3. per-dataset density bar；
4. per-family density heatmap；
5. throughput rows/sec vs panel size；
6. failure reason Pareto。
```

## 决策

如果 P3 strong pass：

```text
existing-action route remains viable；
下一步做 controller / runtime。
```

如果 P3 fail because density < 0.03：

```text
自然动作池本身不够；
必须设计 generated update。
```

如果 P3 fail because materializer missing：

```text
v9.8.4 仍优先修 materializer；
不允许继续 controller fantasy。
```

---

# P4：Gate semantics 与 official 口径

## 目标

继续拆解 raw LDO、support-aware LDO、density gate 的关系，但不擅自修改 official gate。

## 核心假设

$$
H_{P4}:
\text{如果 Core-like density 在自然扩流中足够，则 raw LDO 的 backfill 惩罚可以被新的 density-aware gate 替代。}
$$

## 必须记录

```text
raw_LDO
quality_LDO
support_LDO
backfill_LDO
backfill_share
raw_gate_pass
support_gate_pass
density_gate_pass
official_gate_modification_allowed
```

## 判断标准

允许提出 official gate modification 需要同时满足：

```text
P3 strong pass
quality_LDO <= 0.10
support_LDO <= 0.10
backfill_share explains >= 0.70 raw LDO
density_LCB >= 0.03
```

否则：

```text
official raw gate 保持；
不能用 support-aware diagnostic 写 system pass。
```

---

# P5：几何机制不再做大 score，而做“判别矩阵”

## 目标

当前 architecture-agnostic geometry score 已失败。本阶段不再调它，而是构建判别矩阵：哪些机制能分开 Core77 和 RandomMatched、OldOnly 和 ExactOnly。

## 对照组

```text
Core77 vs RandomMatched
OldOnly vs ExactOnly
Core77 vs ExactOnly
Core77 vs OldOnly
```

## 必须记录

对每个机制变量记录：

```text
feature_name
legality
architecture_specific
effect_size_Core_vs_Random
effect_size_Old_vs_Exact
effect_size_Core_vs_Exact
feature_cost_q90
TopK87_precision
V_LCB
AUV_LCB
longrisk_UCB
memory_UCB
offdiag_UCB
```

## 判断标准

机制 weak：

```text
effect_size_Core_vs_Random >= 0.5
TopK87_precision >= 0.60
longrisk_UCB <= 0.10
```

机制 strong：

```text
effect_size_Core_vs_Random >= 0.8
effect_size_Old_vs_Exact >= 0.5
TopK87_precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
LDO_drop <= 0.10
```

---

# P6：Existing-action minimal controller boundary

## 目标

只有当 P2 或 P5 strong pass，且 P3 weak/strong pass 给出足够 density，才允许构造 controller。

## Controller 约束

```text
no dataset_name branch
no outcome-derived fields
no future outcome
no test/validation outcome
no old table
feature groups <= 6
thresholds frozen
```

## 必须记录

```text
controller_id
feature_names
thresholds
calibration_split_hash
heldout_split_hash
accepted_count
coverage
precision
V_LCB
AUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
feature_cost_q90
```

## Pass

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
AUV_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_UCB <= 0.05
offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

---

# P7：Generated update reopen gate

## 目标

判断是否允许重开 generated route。

## 允许条件

Generated route 只有在以下任意条件成立时才能重开：

```text
Condition A:
  P2/P5 找到 legal mechanism strong pass，且该机制可转成 generation objective。

Condition B:
  P3 strong pass fail because density < 0.03，但 P1/P2 证明 good-action mechanism 明确。

Condition C:
  P6 existing-action controller pass，但想进一步扩展 action density。
```

否则：

```text
generated_route_status = stopped
```

## 如果重开，生成目标

不允许写 APG 小变体。必须直接基于机制 objective：

$$
\Delta^* = \arg\max_{\Delta \in \mathcal{A}} FuturePathGain(\Delta)
$$

subject to：

$$
MemoryHarm(\Delta) \le \tau_M,
$$

$$
OffdiagRisk(\Delta) \le \tau_O,
$$

$$
LongRiskProxy(\Delta) \le \tau_L,
$$

$$
Cost(\Delta) \le C_{max}.
$$

## 必须记录

```text
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_error_linf
branch_horizon_rows_expected
branch_horizon_rows_actual
h1/h5/h20/h80/h240 V
AUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
source-to-generated damage
new_positive_created_rate
longrisk_created_rate
```

## Pass

```text
generated_count >= 256
precision >= 0.50 weak / 0.75 strong
AUV_LCB > 0
h240 longrisk_UCB <= 0.05
memory/offdiag_UCB <= 0.05
new_positive_created_rate >= 0.10
longrisk_created_rate <= 0.05
```

---

# P8：Selected runtime boundary

## 目标

只有 P6 controller pass 或 P7 generated pass 后，才测 selected runtime。

## 必须记录

```text
feature_compute_ms_q90
mechanism_compute_ms_q90
certificate_compute_ms_q90
payload_apply_ms_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
empty_step_kernel_count
active_step_count
selected_actions_per_active_step
```

## Pass

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
empty_step_kernel_count = 0
```

---

# P9：Official paired replay boundary

## 目标

只有 P8 pass 才能打开。

## 必须记录

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

每个 branch 记录：

```text
final CE
final acc
NLL
ECE
CEp99
margin_p10
hard-tail metrics
memory metrics
runtime
```

## Pass

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
ShuffledPayload does not pass
LDO/LSO/LTO stable
runtime pass
```

---

# P10：Short/full training boundary

只有 P9 pass 才能打开。

记录：

```text
train/val/test acc
CE/NLL/ECE
steps_to_target
time_to_target
sample efficiency
robustness
continual retention
forgetting
MLP / StrongLRGridMLP / QuadraticFeatureMLP controls
```

不允许按 dataset 调 controller。

---

# 7. v9.8.3 预期 route

## Route A：Future mechanism found, density enough

```text
P1 strong
P2 or P5 strong
P3 strong
P6 controller pass
```

下一步进入 runtime。

## Route B：Future path exists, mechanism absent

```text
P1 strong
P2/P5 fail
```

说明我们知道哪些动作未来好，但训练当下仍看不见。下一轮继续机制发现，不开 controller。

## Route C：Mechanism exists, density insufficient

```text
P2/P5 strong
P3 fail density < 0.03
```

说明 selector 可行但自然动作池不够，下一步重开 generated update。

## Route D：Natural stream materializer missing

```text
P3 not_run because no materializer
```

下一轮优先修 materializer。不能继续靠 2876 panel 推断。

## Route E：Generated route reopened but fails

```text
P7 generated actions fail value/risk
```

说明机制不能直接生成动作，回到 objective 设计。

---

# 8. 本轮不做什么

v9.8.3 明确不做：

```text
不调 v9.8.2 P2 M_signal threshold；
不调 P5 geometry family threshold；
不只看 AUV；
不只看 h20；
不在 P3 缺失时修改 official density gate；
不重开 APG/APGU 盲生成；
不把 support-aware pass 写成 raw official pass；
不按 dataset 调参；
不使用 future outcome 做 controller feature；
不在 controller 未过时打开 runtime / paired replay / short-full。
```

---

# 9. 我对路线的最终判断

v9.8.2 之后，项目应该停止把问题理解成：

```text
再找一个更好分数；
再补 10 个动作；
再调一个几何 threshold；
再造一个 APG 变体。
```

真正问题是：

$$
\boxed{
\text{我们要找到一种训练当下可用的机制，能解释并复现“未来训练轨迹变好”的动作。}
}
$$

而且这个机制不能局限于 KAN 当前基函数。KAN 的 basis / cover 可以是诊断读数，但优化器最终应理解的是更通用的函数空间几何：

```text
函数响应是否稳定；
未来路径是否改善；
记忆是否保持；
hard-tail 是否缓解；
局部函数片区是否不塌；
long-risk 是否被压住；
计算成本是否可承受。
```

v9.8.3 的成败，不在于能不能马上 system pass，而在于能不能把这个问题从“事后知道好动作”推进到：

```text
训练当下知道为什么它好，
并能稳定选出或生成它。
```
