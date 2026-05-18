# DG-KAN v9.7.7 结果解读与 v9.8.0 Geometry-Adaptive Optimizer 并行验证计划

> 本文件基于 v9.7.7 的真实实验结果制定。  
> 目标不是继续调一个阈值，也不是继续换一个 APG 变体。  
> 本轮要把项目从“在已有动作里找好动作”，推进到“定义并验证一种能自适应理解网络训练几何的优化器”。

---

# 0. 一句话结论

v9.7.7 有进展，但仍然不是能力成功。

这轮最重要的发现是：

```text
Core77 本身质量很高；
raw LDO 很大部分来自 leaveout 后的外部低质动作回填；
support-aware gate 看起来接近可用；
但 raw official gate、LFO、动作密度、OldOnly 机制解释、自然动作流扩展都还没过。
```

更深一层，我认为 v9.7.7 说明：

$$
\boxed{
\text{我们已经知道存在高质量动作，但还没有掌握一种通用的、训练当下可解释的、跨任务稳定的几何自适应更新规则。}
}
$$

也就是说，当前问题不是再找一个更巧的经验筛选器，而是要回答：

```text
什么样的小参数改动，能让网络后续训练轨迹更好？
这个判断能否不依赖 MNIST / Fashion-MNIST / KMNIST 这些具体数据集？
这个判断能否不依赖 KAN 当前某个基函数或某个 action template？
优化器能否自己从模型的函数空间响应、记忆保持、局部稳定性中读懂几何？
```

v9.8.0 因此是一个路线重置：

```text
existing-action 路线继续作为诊断线；
transfer 路线升级为 future trajectory 线；
generated-action 路线继续停止，除非出现新的几何自适应目标；
主线改为 architecture-agnostic Geometry-Adaptive Optimizer。
```

---

# 1. v9.7.7 的独立数据判断

## 1.1 v9.7.7 不是没进展，但不是 system progress

v9.7.7 的 route 是：

```text
R4-CoreExpansionPassButRawGateConflict
```

它没有达到：

```text
strict PureKAN functional success；
full functional success；
external ready；
selected controller；
selected runtime；
official paired replay；
short/full training。
```

所以从“系统能力”角度看，仍然没有进入正反馈链路：

```text
selected controller
-> selected runtime
-> paired replay
-> short/full training
```

但 v9.7.7 做了一件很重要的事：它把 Core77 的 LDO 问题拆开了。

关键数字：

```text
LDO_raw = 0.4415584415584416
LDO_quality = 0.09260529551331587
LDO_support = 0.0
LDO_backfill = 0.34895314604512573
precision-only LDO = 0.0
value-only LDO = 0.09260529551331587
support-adjusted LDO = 0.09260529551331587
equal-count LDO = 0.0773703535816084
```

这说明 raw LDO 不是一个纯粹的“动作质量跨数据集崩了”的指标。它混入了：

```text
1. 质量下降；
2. 支持度不足；
3. leaveout 后为了补数量而引入的低质动作回填。
```

其中主要问题是第三项。

Fashion-MNIST / KMNIST / MNIST 的 backfill count 是：

```text
34 / 28 / 15
```

并且这些 backfill 的 precision 都是：

```text
0.0
```

所以更准确的判断是：

$$
\boxed{
\text{Core77 的已知核心动作质量稳定；raw gate 失败主要来自数量不足和低质回填。}
}
$$

---

## 1.2 Core77 的本质：干净但稀疏

当前 Panel A 仍只是已有 AP0 universe：

```text
AP0 action count = 2876
Core-like count = 77
Core-like rate = 0.026773296244784424
min per-dataset core-like rate = 0.01824817518248175
```

如果 official coverage 下限仍取 $0.03$，2876 个动作里最低需要：

$$
N_{min}=\lceil 0.03\times 2876\rceil=87.
$$

Core77 只有 77 个，差 10 个。

v9.7.7 的 density CI 是：

$$
[0.021475240123524635, 0.033333885489495195].
$$

它跨过 $0.03$，因此不能得出“密度一定够”或“密度一定不够”。

这意味着：

```text
我们还不知道自然动作流扩大后，Core-like 动作密度是否会稳定超过 coverage 下限。
```

而 P2 的 natural AP0 stream extension 没打开：

```text
materializer_entrypoint_found = 0
Panel B/C/D = not_run
reason = no_landed_natural_AP0_stream_extension_materializer_for_v9770
```

所以现在最重要的工程 blocker 之一是：

$$
\boxed{
\text{我们无法判断好动作密度随自然动作流扩展后的真实趋势。}
}
$$

---

## 1.3 E4 是强诊断，但不能 official

v9.7.7 中最接近的是：

```text
E4-DatasetBlindScoreNormalizedTop10
accepted = 87
precision = 0.8850574712643678
V LCB = 0.14120469391749882
risk/bad/null/memory/offdiag UCB = 0
support-adjusted LDO = 0.08877827643026047
```

如果只看这些，E4 很强。

但它不能 official，因为：

```text
raw LDO = 0.39080459770114945
LFO = 0.16091954022988508
raw official pass count = 0
```

这说明：

```text
support-aware gate 认为它接近可用；
legacy raw official gate 仍拒绝；
leave-family-out 也不够稳定。
```

因此 v9.7.7 的核心冲突不是“动作质量差”，而是：

$$
\boxed{
\text{support-aware 解释与 raw official gate 之间存在评价口径冲突。}
}
$$

这个冲突不能靠悄悄改 gate 解决。必须搞清楚：

```text
raw LDO 是否在惩罚真实部署风险？
还是在惩罚一个评估流程中的低质回填机制？
如果实际在线 controller 不会接受那些 backfill，raw LDO 是否过度悲观？
如果 online 系统必须维持 coverage，低质 backfill 又是否说明动作密度不足？
```

也就是说，v9.7.7 把问题变成了一个更高层的决策：

```text
是修改评估定义，还是增加动作密度，还是重新设计更新规则？
```

---

## 1.4 OldOnly 仍未解释

OldOnly 是一组非常关键的动作：

```text
OldRank 选中；
ExactTransfer 没选中；
但真实结果很好。
```

v9.7.7 继续没有解释它。

P5 结果：

```text
matched pair count = 294
mechanism pass count = 0
best feature = WT80_LCB
TopK87 precision = 0.47126436781609193
V LCB = -0.10386482729963209
```

一些 effect-size 较强的特征也不够：

```text
memory_score_inverse / old_family_margin_delta / old_family_probe_loss_inverse
TopK87 precision = 0.45977011494252873
V LCB = -0.033718338838361464
```

这说明：

$$
\boxed{
\text{OldRank 能找到高质量动作，但我们仍然不知道它在训练当下到底抓住了什么。}
}
$$

只要 OldOnly 机制不清楚，OldRank 仍然只能是 diagnostic，而不是理论上可解释的 optimizer rule。

---

## 1.5 Generated-action route 继续停止是正确的

v9.7.7 中：

```text
new objective evidence = 0
APGU/APGV/APGW/APGX/APGY/APGZ run = 0
```

这和前面多轮一致。APG/APGL/APGH/APGT 等路线的共同模式是：

```text
工程闭合；
branch-horizon replay 完整；
但生成动作 value-negative / high-longrisk。
```

因此继续盲目新增 APG 变体是不科学的。

必须等到有新的数学目标，才允许重新打开 generated route。

---

# 2. 更深层判断：现在的问题不是 KAN 基函数问题，而是优化器如何读懂网络几何

用户提出的关键修正是：

```text
优化器应该能自适应了解网络的几何，
不应该拘泥于 KAN 或当前使用的基函数。
```

我认为这个判断是对的，而且应该成为下一阶段的核心原则。

当前我们一直围绕 KAN action、basis cover、payload、candidate template 做诊断。这些是必要的，因为当前实验对象是 KAN。但如果最终目标是 Beyond-MLP 的训练系统，那么优化器不能写死为：

```text
只适用于某个 KAN basis；
只适用于某个 AP0 action template；
只适用于 MNIST/Fashion/KMNIST；
只适用于某个 edge coefficient block。
```

更高层应该是：

$$
\boxed{
\text{优化器通过函数空间响应、记忆保持、局部稳定性和未来轨迹变化，自适应地读懂当前网络的几何。}
}
$$

这意味着：

```text
KAN-specific basis rank / cover entropy 可以作为诊断；
但 official 更新规则应尽量建立在 architecture-agnostic function-space quantities 上。
```

例如：

```text
per-example output response；
train-memory response；
hard-tail response；
Jacobian-vector response；
function displacement norm；
future trajectory value；
old-family harm；
long-risk proxy；
runtime cost。
```

这些东西可以在 KAN、MLP、Transformer 上都定义。

---

# 3. 现在我们真正应该追求的数学对象

以前我们问：

```text
这个动作是不是让当前一步 loss 降？
```

v9.7.2-v9.7.4 已经说明这不够。ExactTransfer 测得很准，但选不到真正好动作。

更正确的对象是未来训练轨迹。

给定当前参数 $\theta_t$，基础优化器 AdamW 的未来 $h$ 步训练算子记为：

$$
\Phi_h(\theta_t).
$$

一个 functional update 是一个小参数改动 $\Delta$，它不应该只优化当前一步，而应该让：

$$
\Phi_h(\theta_t+\Delta)
$$

比：

$$
\Phi_h(\theta_t)
$$

进入更好的训练路径。

定义未来轨迹收益：

$$
FT_h(\Delta)
=
M(\Phi_h(\theta_t))
-
M(\Phi_h(\theta_t+\Delta)),
$$

其中 $M$ 不是单一 acc，而是可以取：

```text
CE；
NLL；
margin；
hard-tail CEp99；
old-family loss；
memory-buffer loss；
long-risk；
bad/null；
cover collapse；
function displacement stability。
```

一个好更新不是满足：

$$
-g_t^\top \Delta > 0.
$$

而是满足：

$$
LCB(FT_{20}(\Delta))>0,
$$

$$
LCB(FT_{80}(\Delta))\ge -\epsilon,
$$

$$
UCB(LongRisk_{240}(\Delta))\le \tau_L,
$$

$$
UCB(MemoryHarm(\Delta))\le \tau_M.
$$

这才对应：

```text
不是找当前这一步降 loss 最多的动作；
而是找能把后续训练带到更好轨迹上的小参数改动。
```

---

# 4. v9.8.0 的路线重置

v9.8.0 不应该继续只是：

```text
Core77 + expansion；
OldRank threshold；
WT80 threshold；
APG 变体；
support-aware gate 调参。
```

v9.8.0 应该并行做四条线：

```text
A 线：Future Trajectory Effect Audit
B 线：Architecture-Agnostic Geometry Probe
C 线：Existing-Action Density and Gate Conflict
D 线：Geometry-Adaptive Update Generation
```

这四条线不是妥协，而是一个清晰的层级：

```text
A 线回答：好动作是不是未来轨迹改善？
B 线回答：优化器能否不依赖 KAN basis 读懂几何？
C 线回答：已有动作流是否足够支撑 controller？
D 线回答：如果 C 不够，能否直接生成几何自适应更新？
```

---

# 5. A 线：Future Trajectory Effect Audit

## 5.1 目标

验证当前高质量动作的本质是否真的是“未来训练轨迹改善”，而不是当前一步 response。

要比较四组动作：

```text
Core77:
  当前最干净的 77 个动作。

OldOnly:
  OldRank 选中、ExactTransfer 没选中，但真实结果好的动作。

ExactOnly:
  ExactTransfer 选中、OldRank 没选中，但真实结果差的动作。

RandomMatched:
  按 dataset / family / template / step / action norm 匹配的随机动作。
```

## 5.2 核心假设

$$
H_A:
\text{Core77 / OldOnly 的优势来自未来训练轨迹改善，而不是 immediate CE transfer。}
$$

如果 $H_A$ 成立，我们应该看到：

```text
Core77 / OldOnly 在 h1 不一定最好；
但在 h20 / h80 / h240 的 V 更稳；
longrisk 更低；
memory/offdiag 更安全；
cover 不塌；
old-family 不忘。
```

## 5.3 必须记录的指标

每个 action，每个 horizon $h \in \{1,5,20,80,240\}$ 记录：

```text
action_id
group_id = Core77 / OldOnly / ExactOnly / RandomMatched
dataset_id only for diagnostics
seed
family
template_id
step_bucket
action_norm
payload_norm

CE_delta_h
NLL_delta_h
margin_delta_h
V_integrated_h
CEp99_delta_h
hard_tail_loss_delta_h

old_family_loss_delta_h
old_family_margin_delta_h
old_stratum_loss_delta_h
memory_buffer_loss_delta_h

longrisk_h
bad_event_h
null_event_h
memory_fail_h
offdiag_fail_h

function_displacement_norm_h
Jacobian_response_norm_h
local_curvature_proxy_h
cover_entropy_delta_h
basis_effective_rank_delta_h

OldRank_score
ExactTransfer_score
WT80_score
support_adjusted_score
```

其中 KAN-specific 的 `cover_entropy_delta` 和 `basis_effective_rank_delta` 只能作为诊断，不作为通用 optimizer 的唯一依据。

## 5.4 判断标准

A 线通过需要同时满足：

$$
LCB(V_{20}^{Core77})>0,
$$

$$
LCB(V_{80}^{Core77})\ge -\epsilon,
$$

$$
UCB(LongRisk_{240}^{Core77})\le 0.05,
$$

$$
LCB(V_{20}^{OldOnly})>0,
$$

$$
UCB(LongRisk_{240}^{OldOnly})\le 0.05,
$$

并且：

$$
LCB(V_{20}^{ExactOnly})\le 0
$$

或：

$$
UCB(LongRisk_{240}^{ExactOnly})>0.15.
$$

此外，需要满足：

```text
OldRank_score 与 future V 的相关性 > ExactTransfer_score 与 future V 的相关性；
Core77 / OldOnly 的 memory/offdiag fail 明显低于 ExactOnly；
这个结论不能只在某一个 dataset 上成立。
```

## 5.5 可视化

必须画：

```text
1. group × horizon 的 V 曲线；
2. group × horizon 的 longrisk 曲线；
3. OldOnly vs ExactOnly 的 future trajectory spider plot；
4. ExactTransfer score vs V20/V80/V240 散点；
5. OldRank score vs V20/V80/V240 散点；
6. Core77 per-dataset support/value heatmap；
7. memory/offdiag fail over horizon 曲线；
8. cover entropy / basis rank over horizon 曲线。
```

## 5.6 结果解释

如果 A 线通过：

```text
证明好动作不是 immediate descent，
而是未来训练轨迹改善。
```

如果 A 线不通过：

```text
说明 Core77 / OldOnly 只是 outcome 标签上的局部好动作，
我们仍没有机制解释。
```

---

# 6. B 线：Architecture-Agnostic Geometry Probe

## 6.1 目标

构建不依赖 KAN 当前 basis 的几何读数。

这条线的目标不是说 KAN 不重要，而是：

```text
官方 optimizer 的核心判断应该尽量在 function space 中成立；
KAN basis 相关指标只作为解释和诊断。
```

## 6.2 通用几何量

对每个动作 $\Delta$，定义以下架构无关响应：

### 6.2.1 当前样本函数位移

$$
d_B(\Delta)=J_B\Delta,
$$

其中 $J_B$ 是当前 batch 输出对参数的 Jacobian。

记录：

```text
mean |d_B|
max |d_B|
class-logit direction alignment
margin response
CE response
```

### 6.2.2 记忆样本函数位移

$$
d_M(\Delta)=J_M\Delta,
$$

其中 $M$ 是 train-memory buffer / old-family / old-stratum 样本集合。

记录：

```text
memory loss change；
old-family margin change；
old-stratum CE change；
function displacement norm on memory。
```

### 6.2.3 多样本一致性

$$
\mu_\Delta = \mathbb{E}_i[-g_i^\top\Delta],
$$

$$
\sigma_\Delta^2 = Var_i[-g_i^\top\Delta].
$$

记录：

$$
SNR_\Delta = \frac{\mu_\Delta^2}{\sigma_\Delta^2+\epsilon}.
$$

但这次 SNR 只作为一个读数，不再当作主 selector。

### 6.2.4 动态放大风险

用小扰动估计动作是否会让未来训练轨迹发散：

```text
h1/h5/h20 的 function displacement growth；
CEp99 growth；
margin tail collapse；
local curvature proxy；
Jacobian response growth。
```

## 6.3 KAN 诊断量

这些只作为 KAN-specific 解释，不作为跨架构必要条件：

```text
basis effective rank delta；
cover entropy delta；
edge block activation spread；
hard-tail cover entropy；
basis locality concentration；
edge-function update sparsity。
```

如果这些指标解释 Core77 / OldOnly，很有价值；但不能把它们写成只能适用于当前基函数的唯一 rule。

## 6.4 判断标准

B 线通过需要找到一组 architecture-agnostic 几何量 $Z_{geom}$，满足：

$$
P(GradeAB=1\mid Z_{geom}, D=d)
$$

在不同 dataset $d$ 上稳定。

具体 gate：

```text
TopK87 precision >= 0.75；
V LCB > 0；
LongRisk UCB <= 0.05；
Bad UCB <= 0.05；
Null UCB <= 0.15；
LDO / LSO / LTO <= 0.10；
不使用 dataset_name；
不使用 outcome-derived fields。
```

如果 B 线失败，则说明 current commit-time geometry probes 还不够，需要 future trajectory probe 或 generated update。

---

# 7. C 线：Existing-Action Density and Gate Conflict

## 7.1 目标

继续回答 v9.7.7 暴露的问题：

```text
Core77 干净但只有 77 个；
E4 支持度口径下接近可用；
raw official gate 与 support-aware gate 冲突；
natural AP0 stream extension 没有 materialize。
```

C 线不再做更多阈值小修，而是做三个硬判断。

---

## 7.2 C1：LDO 分解的 official 解释

继续拆：

$$
LDO_{raw}=LDO_{quality}+LDO_{support}+LDO_{backfill}.
$$

必须记录：

```text
per dataset accepted core count；
per dataset precision；
per dataset V LCB；
per dataset backfill count；
per dataset backfill precision；
per dataset support-adjusted precision；
per dataset equal-count precision；
per family / template / stratum support。
```

判断：

```text
如果 raw LDO 主要来自 backfill，且 online controller 不会接受 backfill，
则 raw LDO gate 需要单独标注为 density failure，而不是 quality failure。

如果 raw LDO 主要来自 core quality drop，
则 Core77 不是 stable route。
```

---

## 7.3 C2：Natural AP0 stream extension materializer

必须实现自然动作流扩展，而不是继续在 2876 个动作里挤 10 个。

目标规模：

```text
Panel A: 2876 actions, existing baseline；
Panel B: 5000 actions；
Panel C: 10000 actions；
Panel D: 20000 actions。
```

每个 panel 必须记录：

```text
Core-like count；
Core-like rate；
GradeAB count/rate；
ValuePositiveNoLongRisk count/rate；
per-dataset rate；
per-template rate；
per-family rate；
Wilson CI；
actions needed for 87 core；
actions needed for 0.03 coverage；
materializer rows/sec；
quality violations；
CUDA memory / wallclock。
```

判断：

```text
如果 Core-like rate 稳定 >= 0.03，existing-action route 继续；
如果 Core-like rate 稳定 < 0.03，existing-action route 不能成为主路线；
如果 rate 随 action count 上升，说明当前 action universe 太小；
如果 rate 随 action count 下降，说明 Core77 可能是小样本好区域。
```

---

## 7.4 C3：Core77 + expansion 最小补齐

只允许在自然扩展流上做最小补齐。

目标：

```text
Core set >= 87；
或者 Core77 + expansion10，但 expansion 必须来自 architecture-agnostic future trajectory / geometry criteria。
```

不允许：

```text
只用 OldRank top10；
只用 dataset-specific threshold；
只用 outcome-derived rank；
只用 exact transfer threshold。
```

判断标准：

```text
accepted >= 87；
precision >= 0.75；
V LCB > 0；
longrisk/bad/null/memory/offdiag UCB within gate；
raw LDO <= 0.10 或明确 route 到 density-only failure；
LFO <= 0.10；
LTO <= 0.10；
LSO <= 0.10。
```

---

# 8. D 线：Geometry-Adaptive Update Generation

## 8.1 原则

D 线不是继续 APG/APGU 变体。它要从函数空间目标直接生成一个小更新。

当前基础优化器是 AdamW：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}.
$$

functional update 是：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{func}.
$$

新的目标不是“造一个 KAN basis 变体”，而是：

$$
\Delta\theta_{func}
=
\text{一个能改善未来函数空间轨迹、同时不伤 memory / cover / runtime 的小扰动。}
$$

---

## 8.2 D1：Function-Space Least-Squares Update

在当前 batch $B$ 上，用输出 Jacobian $J_B$ 构造最小函数空间修正：

$$
\Delta_{out}
=
-J_B^T(J_BJ_B^T+\lambda I)^{-1}r_B,
$$

其中 $r_B$ 是当前输出残差方向。

解释：

```text
这不是 KAN-specific；
它适用于任意可求 JVP/VJP 的网络；
它问的是：在函数空间里，用最小参数扰动修正输出。
```

需要记录：

```text
function displacement norm；
parameter norm；
CE response on current batch；
CE response on memory buffer；
old-family harm；
exact apply error；
runtime cost。
```

---

## 8.3 D2：Memory-Projected Update

防止伤害旧知识。用 memory samples 的 Jacobian $J_M$ 构造投影：

$$
P_M
=
I-J_M^T(J_MJ_M^T+\lambda I)^{-1}J_M.
$$

然后：

$$
\Delta_{mem}=P_M\Delta_{out}.
$$

解释：

```text
这不是把 memory 加进 loss；
这是一个更新空间约束：
只允许在尽量不移动 memory outputs 的方向上更新。
```

判断：

```text
memory harm UCB <= threshold；
old-family margin not worse；
V20/V80/V240 不明显损失；
runtime cost 可接受。
```

---

## 8.4 D3：Future-Checked Update

对 $\Delta_{out}$ 和 $\Delta_{mem}$ 做小规模未来轨迹检查。

流程：

```text
1. apply candidate update；
2. run h=5/20 short unroll with same AdamW schedule；
3. evaluate check samples / memory / hard-tail；
4. only keep update if future trajectory gate passes。
```

未来轨迹 gate：

$$
LCB(FT_{20})>0,
$$

$$
UCB(MemoryHarm_{20})\le \tau_M,
$$

$$
UCB(LongRiskProxy_{20})\le \tau_L.
$$

注意：这一步可能贵，所以 v9.8.0 只做 smoke / diagnostic，不立刻 official runtime。

---

## 8.5 D4：Generated Update Branch-Horizon Smoke

如果 D1-D3 preflight 过，才允许生成 actions。

规模：

```text
D1: 64 actions；
D2: 64 actions；
D3: 64 actions；
negative control: 64 shuffled / random projected actions。
```

每个 action 跑：

```text
branches = RealFunctional / AdamWParallel / bestLR / NoOp / Random / shuffled payload；
horizons = 20 / 80 / 240。
```

判断标准：

```text
GradeAB precision >= 0.50 for smoke；
V LCB > 0；
longrisk UCB <= 0.10；
bad/null <= threshold；
negative control not pass。
```

强 pass 才进入 controller。

---

# 9. v9.8.0 详细实验阶段

## P0. Boundary reproduction and audit

目标：复现 v9.7.7 的所有关键边界，确认本轮不是建立在错误 artifact 上。

必须记录：

```text
route_v9770；
Core-like count/rate；
LDO decomposition；
E4 metrics；
OldOnly mechanism pass count；
natural stream materializer status；
generated route status；
field red count；
Base-Acc Sentinel health。
```

通过标准：

```text
v9.7.7 boundary reproduced；
no fake / no proxy；
no dataset-specific controller；
no outcome-derived field in official feature；
manual training contract preserved。
```

可视化：

```text
v9.7.7 route waterfall；
Core77 LDO decomposition bar；
E4 raw vs support-aware metrics table。
```

---

## P1. Core77 LDO semantic decomposition v2

目标：判断 raw LDO gate 是否混合了质量、支持度、回填。

假设：

$$
H_{P1}:
LDO_{raw}\text{ 的主要成分是 backfill，不是 core quality failure。}
$$

记录指标：

```text
LDO_raw；
LDO_quality；
LDO_support；
LDO_backfill；
precision_only_LDO；
value_only_LDO；
support_adjusted_LDO；
equal_count_LDO；
per-dataset backfill precision；
per-family backfill precision；
backfill source template/family/step distribution。
```

判断标准：

```text
如果 LDO_backfill / LDO_raw >= 0.70 且 precision-only LDO <= 0.05：
  route = density/support failure, not quality failure。

如果 LDO_quality >= 0.20：
  route = true cross-dataset quality failure。
```

可视化：

```text
LDO composition stacked bar；
backfill precision heatmap；
Core vs Backfill value distribution；
per-dataset support plot。
```

---

## P2. Natural AP0 stream extension materializer

目标：落地自然动作流扩展，判断好动作密度是不是足够。

假设：

$$
H_{P2}:
\text{Core-like action density 随自然 action stream 扩大后稳定接近或超过 }0.03.
$$

实施：

```text
Panel A: existing 2876 actions；
Panel B: 5000 natural actions；
Panel C: 10000 natural actions；
Panel D: 20000 natural actions。
```

每个 panel 记录：

```text
Core-like count/rate；
GradeAB count/rate；
ValuePositiveNoLongRisk count/rate；
T5-like count/rate；
per-dataset count/rate；
per-template count/rate；
per-family count/rate；
Wilson CI；
quality audit；
rows/sec；
GPU memory；
wallclock；
missing rows；
label exclusivity violations。
```

判断标准：

```text
Strong pass:
  Core-like rate LCB >= 0.03
  min per-dataset Core-like rate >= 0.02
  quality audit pass

Weak pass:
  Core-like CI includes 0.03 and Core-like count >= 87 in Panel C or D

Fail:
  Core-like rate UCB < 0.03 after Panel D
```

可视化：

```text
Core-like density vs action count with Wilson CI；
per-dataset density curves；
family/template diversity curves；
materializer throughput curve。
```

---

## P3. Future trajectory effect audit

目标：验证好动作是否改善未来训练轨迹。

动作组：

```text
Core77；
OldOnly；
ExactOnly；
E4Expansion10；
RandomMatched。
```

horizons：

```text
1, 5, 20, 80, 240
```

记录：

```text
CE_delta_h；
NLL_delta_h；
margin_delta_h；
V_integrated_h；
CEp99_delta_h；
hard_tail_loss_delta_h；
old_family_loss_delta_h；
memory_buffer_loss_delta_h；
longrisk_h；
bad_event_h；
null_event_h；
memory_fail_h；
offdiag_fail_h；
function_displacement_norm_h；
cover_entropy_delta_h；
basis_rank_delta_h。
```

判断标准：

```text
Core77 and OldOnly:
  V20 LCB > 0
  V80 LCB >= -epsilon
  LongRisk240 UCB <= 0.05
  MemoryFail UCB <= 0.05

ExactOnly:
  either V20 LCB <= 0 or LongRisk240 UCB > 0.15

OldRank correlation with future V > ExactTransfer correlation with future V
```

可视化：

```text
trajectory curves by group；
OldOnly vs ExactOnly future value contrast；
exact transfer vs future V scatter；
OldRank vs future V scatter；
memory/offdiag over horizon；
cover/rank over horizon。
```

---

## P4. Architecture-agnostic geometry probe ledger

目标：构建不依赖 KAN basis 的几何读数。

记录：

```text
J_B Delta response；
J_M Delta response；
per-example response mean/variance；
function displacement norm；
output margin displacement；
memory displacement；
old-family displacement；
hard-tail displacement；
local response growth；
cost q90。
```

另外记录 KAN-only diagnostic：

```text
basis rank delta；
cover entropy delta；
edge block concentration；
basis activation entropy。
```

判断：

```text
architecture-agnostic probe TopK87:
  precision >= 0.75
  V LCB > 0
  longrisk UCB <= 0.05
  LDO/LSO/LTO <= 0.10

KAN-only diagnostic:
  只能解释，不可单独 official。
```

可视化：

```text
function-space response heatmap；
KAN-only vs architecture-agnostic feature correlation；
per-dataset geometry score distribution；
feature importance under leaveout。
```

---

## P5. Geometry-adaptive update preflight

目标：从函数空间目标直接生成少量候选更新，而不是 APG 变体。

候选：

```text
D1 Function-Space Least-Squares Update；
D2 Memory-Projected Update；
D3 Future-Checked Update；
Negative controls。
```

记录：

```text
generated_action_count；
function displacement norm；
parameter norm；
action apply error；
payload hash；
memory harm；
preflight CE response；
future h5/h20 response；
runtime cost。
```

判断：

```text
preflight pass only if:
  action apply error = 0 within tolerance；
  memory harm UCB <= threshold；
  future h20 value LCB > 0 for smoke subset；
  negative controls diverge or fail。
```

---

## P6. Geometry-adaptive update branch-horizon smoke

目标：真实跑 generated updates 的 branch-horizon outcome。

规模：

```text
D1 64 actions；
D2 64 actions；
D3 64 actions；
negative 64 actions。
```

branches：

```text
RealFunctional；
AdamWParallel；
bestLR；
NoOp；
Random；
ShuffledFunctionalPayload。
```

horizons：

```text
20, 80, 240
```

判断：

```text
Weak generated pass:
  GradeAB precision >= 0.50
  V LCB > 0
  longrisk UCB <= 0.10
  negative controls fail

Strong generated pass:
  GradeAB precision >= 0.75
  V LCB > 0
  longrisk UCB <= 0.05
  bad/null UCB within gate
  LDO/LSO smoke stable
```

---

## P7. Existing-action minimal controller boundary

目标：只在 P1-P4 给出足够 evidence 后，尝试 minimal controller。

controller 不能使用：

```text
dataset_name；
outcome-derived fields；
old table labels；
future branch outcomes；
KAN-only basis feature as sole rule。
```

controller 可以使用：

```text
function-space geometry probes；
memory/offdiag hard veto；
future-trajectory-derived training-time proxy；
Core-density support rule；
cost gate。
```

通过标准：

```text
accepted >= 87；
precision >= 0.75；
V LCB > 0；
longrisk/bad/null/memory/offdiag UCB within gate；
LDO/LSO/LTO/LFO <= 0.10；
template/family support sufficient；
no dataset dispatch。
```

---

## P8. Runtime boundary

只有 P7 或 P6 过线，才测 selected runtime。

记录：

```text
feature compute q90；
probe compute q90；
certificate compute q90；
payload apply q90；
step_ratio_q90；
peak memory ratio；
active step launch count；
empty step launch count。
```

通过标准：

$$
StepRatio_{q90}\le 1.50,
$$

$$
MemoryRatio\le 1.05.
$$

如果 geometry-adaptive probes 太贵，则 route 到：

```text
geometry signal exists but runtime illegal
```

不能写成 system pass。

---

## P9. Paired replay / short-full boundary

只有 P7 + P8 均过，才打开。

记录：

```text
RealFunctional beats AdamWParallel；
RealFunctional beats bestLR；
RealFunctional beats NoOp；
RealFunctional beats Random；
shuffled payload fails；
leave-dataset-out；
leave-stratum-out；
leave-template-out；
short-run acc/loss/calibration；
full-run sample efficiency；
continual / anti-forgetting。
```

这一步不是 v9.8.0 的默认目标，只有上游 pass 才运行。

---

# 10. 关键 stop rules

## 10.1 Existing-action route stop rule

如果 Panel D natural stream 后：

$$
UCB(p_{Core-like}) < 0.03,
$$

则 existing-action route 不能作为主路线。

结论应写成：

```text
clean actions exist but density insufficient under current action source。
```

## 10.2 Transfer route stop rule

如果 future trajectory audit 显示：

```text
OldOnly / Core77 不比 ExactOnly / RandomMatched 稳定；
或 future trajectory metrics 无法解释 outcome；
```

则 stop transfer-style theory route。

## 10.3 Geometry-adaptive generated route stop rule

如果 D1/D2/D3 全部：

```text
V LCB <= 0；
longrisk UCB high；
negative control 不低于 real update；
```

则 generated route 继续停止，不再新增 APG* 名称。

## 10.4 Runtime stop rule

如果 selected controller 需要 expensive unroll 才能工作，且：

$$
StepRatio_{q90}>1.50,
$$

则不能 official。需要 distill cheaper proxy，但 proxy 必须重新过 P7。

---

# 11. 最终判断逻辑

v9.8.0 结束时，必须给出以下几种 route 之一。

## Route A：Existing-action route viable

条件：

```text
natural action stream density >= 0.03；
minimal controller pass；
runtime pass。
```

含义：

```text
当前自然 AP0 action source 足够，只需要更稳定的 controller。
```

## Route B：Geometry-adaptive generated route viable

条件：

```text
D1/D2/D3 generated updates produce GradeAB-positive, low-risk actions；
minimal generated controller pass；
runtime pass。
```

含义：

```text
从函数空间几何直接生成动作是正确路线。
```

## Route C：Good actions exist but density insufficient

条件：

```text
Core-like quality stable；
但 natural stream density < 0.03。
```

含义：

```text
不是 selector 失败，而是 action source 不够。
```

## Route D：Future trajectory theory unsupported

条件：

```text
Core77/OldOnly future trajectory 没有结构性优势。
```

含义：

```text
当前高质量动作可能只是 outcome-level coincidence；需要重写理论。
```

## Route E：Geometry signal exists but runtime illegal

条件：

```text
controller metrics pass；
但 step_ratio_q90 > 1.50。
```

含义：

```text
理论有希望，但系统不可用。
```

---

# 12. 本轮最重要的反思

v9.7.7 之后，项目不能继续主要依赖经验筛选。

我们现在知道：

```text
1. 高质量动作存在；
2. Core77 的质量跨数据集不一定真的崩；
3. raw LDO 大量来自支持度和回填惩罚；
4. OldOnly 仍未被解释；
5. exact/windowed transfer 当前解释不了好动作；
6. generated-action 变体多轮失败；
7. 当前 action source 可能密度不足。
```

所以 v9.8.0 的核心不是“补一个阈值”，而是：

$$
\boxed{
\text{把 functional update 从经验动作筛选，提升为几何自适应优化器问题。}
}
$$

这个优化器不应拘泥于 KAN 当前基函数。KAN 是当前实验模型，但好优化器应该能读懂任意网络的函数空间几何：

```text
这个小更新是否改变未来训练轨迹？
是否保住旧知识？
是否避免长期风险？
是否提升或维持局部表达能力？
是否跨数据分布稳定？
是否 runtime 可承受？
```

如果 v9.8.0 能回答这些问题，项目才会从“很多人工实验”走向“有数学核心的 functional training system”。

