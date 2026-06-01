# DG-KAN v9.7.4 Existing-Action LDO 拆解 / Transfer 失配复盘 / Horizon-Transfer 验证 完整实验计划

> 本计划基于 v9.7.3 的真实执行结果制定。v9.7.4 不继续小修 `R8A`、`T4`、`WT20`、`E1` 或 APG/APGU/APGV 小变体。  
> 本轮目标是把 v9.7.3 的两个主事实拆清楚：
>
> 1. 旧的 existing-action rank 仍然能选到高质量动作，但跨数据集不稳；
> 2. exact/windowed transfer 在当前形式下没有帮助，甚至会排掉大量高价值旧 rank 动作。
>
> v9.7.4 的目标不是马上追求 full functional success，而是决定下一步到底应继续 existing-action 路线、重写 transfer 判据，还是彻底暂停 generated-action 路线。

---

## 0. 本轮一句话目标

v9.7.4 要回答的问题是：

$$
\boxed{
\text{已有好动作为什么跨数据集不稳？exact/windowed transfer 为什么没有识别它们？}
}
$$

更直白地说：

```text
我们已经知道：
  有一批旧 rank 能选中的动作很强；
  有 77 个 core 动作非常干净；
  补到 87 个动作后质量还不错；
  但 LDO 仍失败。

我们也知道：
  exact one-step transfer 真实落地了；
  linear/apply audit 很准；
  但 exact transfer 排名很差；
  windowed transfer subset 也很差。

现在必须查清：
  旧 rank 为什么好？
  旧 rank 为什么跨数据集不稳？
  transfer 为什么排掉高价值动作？
  one-step/windowed CE transfer 是否根本不是我们要的好动作判据？
```

---

## 1. 背景解释：本轮涉及的几个名字是什么意思

### 1.1 existing-action

`existing-action` 指已经在 canonical AP0 universe 里存在的 2876 个候选动作。每个动作可以理解为：

```text
在某个训练 step，除了普通 AdamW 更新之外，额外给 KAN 参数加一小块变化。
```

这条路线不是重新造动作，而是在已有 2876 个动作里挑一批。

### 1.2 old rank / R8A / R5B

`old rank` 指过去几轮里表现最强的合法排序规则。它可以把一批动作排到前面。它的问题不是质量差，而是跨数据集不稳。

v9.7.3 里它仍然很强：

```text
S0-old-rank precision = 0.873563
V_integrated LCB = 0.141113
longrisk UCB = 0.0
LDO drop = 0.379310
```

这说明它能找到好动作，但离开某个数据集之后掉得很厉害。

### 1.3 Core77

`Core77` 指最干净的 77 个动作集合。它们几乎都很好：

```text
accepted = 77
GradeAB = 1.0
V_integrated LCB > 0
longrisk / bad / null / memory / offdiag 都为 0
```

问题是 official coverage 下限大约需要 87 个动作，所以它少了约 10 个。

### 1.4 expansion10

`expansion10` 指从 core 之外再补 10 个动作，把 77 个补到 87 个。v9.7.3 里 best expansion 是：

```text
E1-old-rank-top10
accepted = 87
precision = 0.885057
V LCB = 0.141205
longrisk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.390805
```

这说明补出来的 87 个动作质量很好，但仍然跨数据集不稳。

### 1.5 exact transfer

`exact transfer` 指真正计算一个动作对没有参与构造它的样本是否有帮助。

对动作 $\Delta$ 和检查样本 $i$，线性化收益可以写成：

$$
r_i(\Delta)=-g_i^\top\Delta.
$$

如果 $r_i(\Delta)>0$，表示这个动作预计会降低样本 $i$ 的 loss。v9.7.2 已经证明 exact transfer 的测量是准确的：linear/apply Pearson 接近 1，Spearman 也接近 1。

但 v9.7.3 显示 exact transfer 不是好动作选择器：

```text
ExactT4Top87 precision = 0.206897
OldRankTop87 precision = 0.873563
```

### 1.6 windowed transfer

`windowed transfer` 指不只看一步，而是看窗口 1、5、20 的 transfer 表现。v9.7.3 做了 subset materializer：

```text
subset actions = 384
windows = 1,5,20
check samples/action = 32
materializer pass = 1
```

但 best `WT20-LCB` 很差：

```text
TopK87 precision = 0.080460
V LCB = -1.024359
longrisk UCB = 0.786870
```

所以当前 windowed transfer 也没有成为好动作规则。

### 1.7 LDO

`LDO` 是 leave-dataset-out。意思是：训练/校准时不看某一个数据集，然后测试在这个被留出的数据集上是否还有效。

它不是为了按数据集调参。恰好相反，它是为了防止 controller 偷学某个数据集特例。

v9.7.3 最大 blocker 仍然是 LDO：好动作在 pooled 上很强，但跨数据集不稳。

---

## 2. v9.7.3 的独立判断

### 2.1 v9.7.3 有进展，但不是能力成功

v9.7.3 没有 strict PureKAN functional success，也没有 full functional success，更没有 external-ready。controller、selected runtime、paired replay、short/full training 都没有打开。

但 v9.7.3 不是空跑。它做了三件实事：

```text
1. exact transfer failure autopsy：证明 exact transfer 排掉了大量旧 rank 高价值动作；
2. Core77 + expansion10：证明用 old rank 补 10 个动作仍能保持很高质量，但 LDO 不过；
3. windowed transfer subset：证明当前 windowed transfer 分数也没有成为好动作选择器。
```

### 2.2 最重要的发现：exact transfer 会杀掉高价值旧动作

v9.7.3 的 P1 数据非常关键：

```text
OldRankTop87 precision = 0.873563
ExactT4Top87 precision = 0.206897
OldExactIntersection count = 18
OldExactIntersection precision = 0.944444
ExactOnly count = 69
ExactOnly precision = 0.014493
ExactOnly V LCB = -0.303714
OldOnly count = 69
OldOnly precision = 0.855072
failure mode = exact_transfer_kills_high_value_actions
```

这组数说明：

```text
old rank 和 exact transfer 的交集很小，但交集质量很高；
exact transfer 单独选出来的 69 个动作几乎没用；
old rank 单独选出来、exact transfer 排掉的 69 个动作反而非常好。
```

所以现在不能说：

```text
exact transfer 是更干净的理论判据，只是阈值没调好。
```

更准确是：

$$
\boxed{
\text{one-step exact transfer 与我们真正想要的 h20/h80/h240 好动作之间存在目标失配。}
}
$$

### 2.3 Core77 + old-rank expansion 是现有最强线索

v9.7.3 的 P2 显示：

```text
best = E1-old-rank-top10
accepted = 87
precision = 0.885057
V LCB = 0.141205
longrisk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.390805
```

这说明 existing-action 路线还没有死。事实上，如果不看 LDO，它已经非常接近可用。

真正的问题是：

$$
\boxed{
\text{这 87 个动作在 pooled 上很好，但没有跨数据集稳定。}
}
$$

所以 v9.7.4 不应该继续调 accepted count，而要问：

```text
这 87 个动作为什么在某些 dataset 上好，在 leave-dataset-out 时不稳？
core 77 和 expansion 10 分别贡献了什么？
不稳定主要来自 core，还是来自补进去的 10 个？
```

### 2.4 windowed transfer 当前也失败

v9.7.3 的 P3 显示 windowed transfer materializer 真实运行：

```text
subset actions = 384
windows = 1,5,20
check samples/action = 32
windowed rows = 384
materializer pass = 1
```

但 best score 仍很差：

```text
best = WT20-LCB
TopK87 precision = 0.080460
V LCB = -1.024359
longrisk UCB = 0.786870
memory/offdiag UCB = 0.906962 / 0.916296
```

这说明不是把 one-step transfer 改成 20-step window 就自然能解决问题。

但这还不能彻底否定 transfer 思路。因为本轮 windowed transfer 只在 subset 上跑，且 scoring 方式可能仍然在看 CE transfer，而不是看 functional update 真正在 h20/h80/h240 上的 geometry effect。

### 2.5 dataset-shift stabilization 仍然失败

v9.7.3 的 P4 继续确认：

```text
best = S0-old-rank
precision = 0.873563
V LCB = 0.141113
LDO drop = 0.379310
```

低 LDO 策略可以降低 drop，但会让 precision/value 崩。这和 v9.6.8、v9.7.0、v9.7.2 一致：

```text
质量强 -> LDO 差；
LDO 低 -> 质量差。
```

这不是简单 score transport 能解决的问题。

---

## 3. 当前真正卡在哪里

### 3.1 卡点一：旧 rank 好，但我们还不知道它为什么好

旧 rank 能选出高质量动作。这个事实多轮稳定出现。

但我们还没有解释：

```text
它选中的动作共同结构是什么？
它是不是学到某个 dataset / family / stratum 的局部规律？
它是不是隐含利用了未来效果的 proxy？
它是不是在某些训练阶段特别有效？
它选中的动作为什么 exact transfer 不喜欢？
```

如果不解释旧 rank，直接调阈值就是小修小补。

### 3.2 卡点二：exact transfer 衡量的是即时 CE，而目标是长期训练形状

exact transfer 测得很准，但它问的问题可能太短：

```text
加这个动作后，当前 check sample 的 CE 是否马上下降？
```

而我们真正要的是：

```text
h20 是否有收益；
h80 是否不坏；
h240 是否不出 longrisk；
old family 是否不忘；
cover/offdiag 是否不坏；
是否打过 AdamWParallel / bestLR / NoOp / Random。
```

这两者不一定一致。v9.7.3 的 OldOnly vs ExactOnly 数据已经说明：one-step transfer 单独看，会选错动作。

### 3.3 卡点三：Core77 到 87 的缺口仍然没有跨数据集稳定解

Core77 很干净，但数量不足。old-rank top10 可以补到 87，并且质量也不错，但 LDO 不过。

这个问题不能再写成：

```text
再找一个 expansion threshold。
```

应该写成：

```text
补进去的 10 个动作要满足跨数据集稳定，不能只在 pooled 上好。
```

### 3.4 卡点四：generated route 仍没有新目标

APGH/APGL/APGT 多轮失败，APGU/APGV/APGW 没有运行是正确的。

v9.7.3 也没有出现新的 direct update objective。windowed transfer subset 没过，所以不能重新打开 generated route。

现在 generated route 的状态应继续是：

```text
stopped_no_new_objective
```

除非 v9.7.4 能证明某种 horizon-coupled transfer 或 geometry rule 可以作为新生成目标。

---

## 4. v9.7.4 的总体策略

v9.7.4 不再做单线推进，而是三线并行。

### A 线：existing-action 路线继续推进

目标是查清旧 rank 为什么好，以及如何解决 LDO。

核心问题：

$$
\boxed{
\text{能否从 Core77 + expansion10 中得到跨数据集稳定的 accepted region？}
}
$$

### B 线：transfer 路线改成“失配解释”，不是继续调阈值

目标不是再调 TransferLCB，而是解释为什么 exact/windowed transfer 会排掉高价值旧动作。

核心问题：

$$
\boxed{
\text{one-step/windowed CE transfer 与 h20/h80/h240 好动作之间到底差在哪里？}
}
$$

### C 线：generated route 继续停止，除非 B 线找到新目标

目标不是继续造 APG 小变体，而是设硬 stop-rule。

核心问题：

$$
\boxed{
\text{是否存在一个比 APGH/APGL/APGT 更清楚的新生成目标？}
}
$$

若没有，generated route 不运行。

---

## 5. 本轮核心假设

### H1：旧 rank 高价值动作不是靠 one-step CE transfer 解释的

依据：v9.7.3 中 OldOnly count = 69，precision = 0.855072；ExactOnly count = 69，precision = 0.014493。

H1 认为：

```text
旧 rank 选中的高价值动作，可能通过延迟收益、几何稳定、memory/offdiag 安全、horizon-safe 路径起作用；
而 exact transfer 只看一步 CE，因此会错杀这些动作。
```

成立标准：

```text
OldOnly actions 在 one-step exact transfer 上低分，
但在 h20/h80/h240 outcome、memory/offdiag、GradeAB 上明显优于 ExactOnly。

并且 OldOnly 的优势不能被 dataset-name、template collapse、payload bucket 等 forbidden/degenerate axis 单独解释。
```

失败标准：

```text
OldOnly 只是某个 group pocket；
或者 OldOnly 与 ExactOnly 的差异可以被非法/退化字段解释；
或者 OldOnly 在 stricter leaveout 下质量也消失。
```

### H2：Core77 + expansion10 的 LDO 失败主要来自 expansion10 或 score-scale shift

H2 认为 Core77 可能本身稳定，补进去的 10 个动作或 score scale transport 造成 LDO fail。

成立标准：

```text
Core77 的 LDO drop 明显小于 Core77+E1；
或者 expansion10 在某个 heldout dataset 的 bad/null/offdiag 或 false positive 显著上升；
或者 expansion10 的 score distribution 在 Fashion-MNIST / KMNIST / MNIST 之间发生明显尺度漂移。
```

失败标准：

```text
Core77 自身也 LDO fail；
expansion10 不是主因；
LDO failure 是全体 accepted region 的系统性问题。
```

### H3：windowed transfer 的失败是因为窗口/样本/指标没有对齐真正目标

H3 认为 v9.7.3 的 WT20 失败不代表所有 transfer 思路失败。可能问题是：

```text
只看 CE transfer；
subset 没覆盖关键 cohort；
没有按 OldOnly / Intersection / ExactOnly 分层；
没有把 transfer 与 memory/offdiag/horizon outcome 联合分析。
```

成立标准：

```text
分层后某些窗口或样本群能解释 OldOnly 的 h20/h80/h240 优势；
或者 horizon-coupled transfer 与 GradeAB / V_integrated / longrisk 有明显关系。
```

失败标准：

```text
window 1/5/20/80 的 transfer 与好动作均无稳定关系；
transfer-positive 仍大量 value-negative 或 longrisk-high；
transfer 不能改善 Core77 expansion 的 LDO。
```

### H4：generated route 不应重启，除非 transfer 或 geometry rule 给出新生成目标

成立标准：

```text
若 H1-H3 没有产生新目标，APGU/APGV/APGW 继续 not_run。
```

失败标准：

```text
如果 horizon-coupled transfer 找到稳定正例机制，
则允许 direct transfer-solved update sandbox，
但仍不能 blind run APG family。
```

---

## 6. P0：边界复现与 artifact 合法性审计

### 目标

确认 v9.7.3 的关键边界真实存在，并且本轮不混入非法字段、不使用 dataset-specific controller。

### 必须读取的 artifact

```text
v9.7.3 route_decision
v9.7.3 exact_transfer_failure_autopsy
v9.7.3 core77_expansion10 table
v9.7.3 windowed_transfer_subset table
v9.7.3 dataset_shift_stabilization table
v9.7.2 exact transfer materializer
canonical AP0 outcome universe
Base-Acc Sentinel manifest
```

### 必须记录的指标

```text
source_route_v9730
OldRankTop87_precision
ExactT4Top87_precision
OldExactIntersection_count
OldOnly_count
ExactOnly_count
Core77_count
E1_accepted_count
E1_precision
E1_V_LCB
E1_LDO_drop
WT20_precision
WT20_V_LCB
generated_route_status
field_green_count
field_yellow_count
field_red_count
dataset_name_used_in_controller
outcome_derived_field_used_in_controller
fake_proxy_row_count
```

### 判断标准

P0 pass 当且仅当：

```text
v9.7.3 boundary 可复现；
field red count = 0；
dataset_name_used_in_controller = 0；
outcome_derived_field_used_in_controller = 0；
fake_proxy_row_count = 0。
```

---

## 7. P1：OldRank vs ExactTransfer 失配拆解

### 目标

把 87 个 OldRankTop 动作、87 个 ExactT4Top 动作拆成三组：

```text
Intersection：OldRank 和 ExactTransfer 都喜欢；
OldOnly：OldRank 喜欢，但 ExactTransfer 不喜欢；
ExactOnly：ExactTransfer 喜欢，但 OldRank 不喜欢。
```

v9.7.3 的基础数据是：

```text
Intersection count = 18, precision = 0.944444
OldOnly count = 69, precision = 0.855072
ExactOnly count = 69, precision = 0.014493
```

### 要记录的字段

每个 action 记录：

```text
action_id
event_id
dataset_id 仅用于 diagnostic / leaveout，不进入 controller
seed
family
stratum
step_bucket
candidate_template_id
payload_hash
old_rank_score
exact_T4_score
exact_transfer_mean
exact_transfer_std
exact_transfer_LCB
window1_score
window5_score
window20_score
V_integrated
GradeAB
h20_value
h80_value
h240_value
h240_longrisk
bad_event
null_event
memory_fail
offdiag_fail
cover_collapse
payload_norm
action_norm
action_AdamW_cosine
```

### 分析方式

构造三张对比表：

```text
Intersection vs OldOnly
Intersection vs ExactOnly
OldOnly vs ExactOnly
```

每张表报告：

```text
count
GradeAB precision
V_integrated LCB
h20/h80/h240 value mean and LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
per-dataset count and precision
per-family count and precision
per-template count and precision
score distribution
```

### 可视化

```text
1. old_rank_score vs exact_transfer_LCB scatter，点颜色为 GradeAB / longrisk；
2. 三组 action 的 V_integrated violin plot；
3. 三组 action 的 h240 longrisk bar；
4. 三组 action 的 per-dataset precision heatmap；
5. OldOnly actions 的 exact transfer 分布与 h20/h80/h240 outcome 对照图；
6. ExactOnly actions 的 high transfer / low value 失败案例表。
```

### 假设成立标准

H1 pass 当且仅当：

```text
OldOnly precision >= 0.75；
OldOnly V_integrated LCB > 0；
OldOnly longrisk UCB <= 0.05；
ExactOnly precision <= 0.25 或 V_integrated LCB <= 0；
OldOnly 优势不能被单个 forbidden/degenerate group 解释；
OldOnly 至少覆盖 2 个 dataset 和 >= 16 个 templates。
```

若 H1 pass：

```text
one-step exact transfer 不再作为主 rank；
进入 P4 查 delayed / horizon-coupled transfer。
```

若 H1 fail：

```text
old rank 可能也是 pocket 或非法 proxy；
existing-action route 降级，不能继续靠旧 rank。
```

---

## 8. P2：Core77 与 expansion10 的 LDO 责任拆解

### 目标

确定 LDO 失败来自哪里：

```text
Core77 本身不稳？
expansion10 不稳？
score scale shift？
某个 dataset 的 target density 少？
某个 dataset 的 null/offdiag 增加？
```

### 要比较的集合

```text
C0-Core77
E1-old-rank-top10
C0+E1-87
C0+ExactT4-top10
C0+Intersection-first10
C0+random-safe-diagnostic10
```

### 记录指标

```text
accepted_count
coverage
GradeAB precision
V_integrated LCB
h20/h80/h240 V LCB
longrisk/bad/null/memory/offdiag UCB
LDO drop
LSO drop
LTO drop
per-dataset accepted count
per-dataset precision
per-dataset V LCB
per-dataset null/bad/offdiag
per-dataset score mean/std/quantile
core_vs_expansion contribution
```

### 可视化

```text
1. Core77 和 expansion10 在三个 dataset 上的 action count 堆叠图；
2. Core77 vs expansion10 的 precision / V / risk 对比图；
3. leave-dataset-out 时被丢失的动作列表；
4. score distribution 按 dataset 分面；
5. expansion action 的 failure reason waterfall。
```

### 判断标准

如果以下条件成立，说明 Core77 本身稳定，问题主要在 expansion：

```text
Core77 LDO drop <= 0.10；
Core77 per-dataset precision >= 0.75；
Core77 V LCB > 0；
Core77 longrisk/bad/null UCB 低；
Core77+E1 的 LDO drop 明显高于 Core77。
```

如果 Core77 自身也 LDO fail：

```text
existing-action route 需要重新定义 core，不只是补 10 个动作。
```

---

## 9. P3：dataset shift 的非调参诊断

### 目标

诊断 dataset shift，但不按 dataset 调 controller。

允许：

```text
per-dataset diagnostics；
leave-dataset-out；
score distribution comparison；
dataset-blind quantile transform；
calibration split worst-group analysis。
```

禁止：

```text
if dataset == MNIST then threshold = ...；
per-dataset topK；
per-dataset manual dispatch；
dataset name as score feature。
```

### 要测试的非调参方法

#### S0：raw old rank

保留 v9.7.3 best baseline。

#### S1：global quantile rank

把 score 变成全局分位数，不使用 dataset name。

#### S2：batch-distribution normalized rank

用当前 batch 的无标签统计做归一化：

```text
current batch CE mean/std
current batch margin mean/std
hard-tail fraction
memory/offdiag proxy distribution
```

不使用 dataset name。

#### S3：core-anchor rank

用 Core77 的 score distribution 作为 anchor，判断新动作是否靠近 core 区域。

#### S4：worst-group calibrated threshold

在 calibration split 中，用 leaveout axes 找 worst-group threshold，但 final rule 只有一个全局阈值，不按 dataset 分支。

### 记录指标

```text
TopK87 precision
V_integrated LCB
longrisk/bad/null/memory/offdiag UCB
LDO/LSO/LTO drop
per-dataset score PSI
per-dataset score mean/std
accepted group share
coverage
```

### 判断标准

P3 pass 当且仅当至少一个方法满足：

$$
N_{accept}\ge87,
$$

$$
Precision_{GradeAB}\ge0.75,
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
LDO\_drop\le0.10,
$$

$$
LSO\_drop\le0.10,
$$

$$
LTO\_drop\le0.10.
$$

---

## 10. P4：Horizon-Coupled Transfer 失配验证

### 目标

验证 transfer 失败是不是因为只看 one-step CE，而没有看 h20/h80/h240 的真实轨迹。

### 样本设计

从以下 cohort 各取动作：

```text
Intersection：18 个全取；
OldOnly：最多 128 个，若不足全取；
ExactOnly：最多 128 个，若不足全取；
Core77：77 个全取；
Expansion10：10 个全取；
RandomLegalSafeDiagnostic：64 个。
```

### 运行窗口

```text
window = 1, 5, 20, 80
```

window 80 是新增重点。如果资源不足，先跑 cohort-balanced 版本，再决定是否 full materialize。

### 记录指标

每个 action / window 记录：

```text
window_transfer_mean
window_transfer_std
window_transfer_LCB
check_sample_CE_delta
check_sample_margin_delta
memory_sample_CE_delta
old_family_CE_delta
hard_tail_CE_delta
horizon_state_hash
apply_error
V_integrated
GradeAB
h240_longrisk
bad/null/memory/offdiag
```

### 可视化

```text
1. window 1/5/20/80 transfer 与 V_integrated 的相关图；
2. window transfer 与 h240 longrisk 的相关图；
3. OldOnly/ExactOnly/Intersection 的 window transfer 曲线；
4. transfer LCB vs old rank score 的 2D 图；
5. window 80 是否恢复 OldOnly 高价值动作的 bar plot。
```

### 判断标准

P4 weak pass：

```text
某个 window score 对 GradeAB 的 TopK87 precision >= 0.50；
且 V LCB > 0；
且 longrisk UCB <= 0.15。
```

P4 strong pass：

```text
TopK87 precision >= 0.75；
V LCB > 0；
longrisk/bad/null/memory/offdiag 全部过线；
LDO/LSO/LTO drop <= 0.10。
```

若 P4 fail：

```text
transfer principle 继续降级为诊断，不进入 controller，也不打开 generated route。
```

---

## 11. P5：旧 rank 机制解释，而不是继续信任黑箱

### 目标

解释旧 rank 为什么能选到高质量动作。

### 拆解对象

```text
old rank score components
memory/offdiag proxy
value proxy
risk veto
hard-tail proxy
score transport component
template / family / step features
```

### 需要回答

```text
旧 rank 是否依赖某个 group pocket？
旧 rank 是否隐含 dataset shift？
旧 rank 是否主要在 memory/offdiag-safe 区域有效？
旧 rank 是否其实在检测 h20/h80/h240 之外的几何稳定特征？
旧 rank 与 exact/windowed transfer 的冲突来自哪里？
```

### 记录指标

```text
component ablation precision
component ablation V LCB
component ablation longrisk UCB
component ablation LDO/LSO/LTO
single-component TopK precision
leave-component-out drop
field legality color
```

### 判断标准

P5 pass 当且仅当：

```text
能解释 >= 80% OldOnly high-value actions 的共同机制；
该机制不使用 red fields；
该机制不是 base share = 1 的退化轴；
该机制跨至少 2 个 dataset 和 >= 16 个 template 稳定。
```

---

## 12. P6：Existing-Action Minimal Controller 冻结实验

### 前置条件

P6 只有在以下至少一个条件满足时运行：

```text
P2 证明 Core77 本身 LDO 稳定，并找到安全 expansion；
或 P3 找到 dataset-blind stable transport；
或 P4 找到 horizon-coupled transfer rule；
或 P5 找到 legal stable mechanism。
```

### Controller 形式

Controller 尽量简单：

```text
Accept(a)=1
当且仅当：
  value rank in top region；
  longrisk veto pass；
  bad/null veto pass；
  memory/offdiag veto pass；
  template / group support pass；
  cost pass。
```

不允许：

```text
dataset-specific threshold；
outcome-derived field；
future branch outcome；
validation/test metric；
old table label。
```

### 记录指标

```text
accepted_count
coverage
GradeAB precision
V_integrated LCB
h20/h80/h240 V LCB
longrisk/bad/null/memory/offdiag UCB
LDO/LSO/LTO drop
max dataset share
max template share
max family share
feature compute q90
payload apply q90
```

### Pass 标准

$$
N_{accept}\ge87
$$

$$
Coverage\ge0.03
$$

$$
Precision_{GradeAB}\ge0.75
$$

$$
LCB(V)>0
$$

$$
UCB(LongRisk)\le0.05
$$

$$
UCB(Bad)\le0.05
$$

$$
UCB(Null)\le0.15
$$

$$
LDO/LSO/LTO\_drop\le0.10
$$

---

## 13. P7：Selected Runtime Boundary

### 前置条件

只有 P6 pass 才运行。

### 目标

测 selected controller 的真实 online runtime。

### 记录指标

```text
selected_controller_id
feature_compute_ms_q50/q90/q99
certificate_compute_ms_q50/q90/q99
payload_apply_ms_q50/q90/q99
base_step_ms_q50/q90/q99
step_ratio_q90
memory_ratio
kernel_launch_count
sync_count
active_step_count
zero_candidate_step_count
```

### Pass 标准

$$
StepRatio_{q90}\le1.50
$$

$$
MemoryRatio\le1.05
$$

并且：

```text
selected controller 真实运行；
没有 offline materializer 混入 timed path；
没有 oracle mask；
没有 fake/proxy rows。
```

---

## 14. P8：Generated Route Stop / Reopen Rule

### 默认状态

```text
generated_route_status = stopped_no_new_objective
APGU/APGV/APGW/APGX_run = 0
```

### 允许重开的唯一条件

以下任一条件成立：

```text
P4 strong pass，证明 horizon-coupled transfer 可作为新目标；
P5 pass，发现旧 rank 的可生成机制；
P6 pass 且 controller 指出可生成方向的结构约束。
```

### 如果重开，只允许 direct solved sandbox

不允许盲目 APG family：

```text
不运行 APG 小变体；
不运行 blind residual blend；
只运行由明确目标导出的 direct solved update。
```

形式为：

$$
\Delta^* = \arg\max_{\Delta\in\mathcal{A}} Score_{new}(\Delta)
$$

其中 $Score_{new}$ 必须来自 P4/P5 的通过结果，而不是人工拼装。

---

## 15. P9：Paired Replay Boundary

### 前置条件

P6 controller pass 且 P7 runtime pass。

### 目标

证明 RealFunctional 不只是局部好，而是真正打过对照。

### 对照分支

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

### 记录指标

```text
paired action count
branch completion
horizon completion
RealFunctional vs each control V20/V80/V240
beats AdamWParallel
beats bestLR
beats NoOp
beats Random
shuffled payload fail
bad/null/longrisk
ECE/NLL/CEp99/margin_p10
```

### Pass 标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
ShuffledPayload 不通过；
bad/null/longrisk 不超阈值；
LDO/LSO/LTO 稳定。
```

---

## 16. P10：Short/Full Training Boundary

### 前置条件

P9 paired replay pass。

### 目标

才允许进入 short/full training。

### 记录指标

```text
train acc
val acc
test acc
CE
NLL
ECE
CEp99
margin_p10
time_to_target
steps_to_target
sample efficiency
robust noise corruption
old-family retained acc
continual forgetting
runtime step ratio
memory ratio
```

### 对照

```text
LQ base AdamW only
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
NoOp functional
Random functional
Shuffled functional payload
```

### 注意

这一步不能用来调前面 controller。它只能做最终验证。

---

## 17. 预期 route decision

### Route A：ExistingActionControllerPass

条件：P6/P7 pass。

下一步：paired replay。

### Route B：ExistingActionLDOBlocked

条件：existing-action precision/value/risk 过，但 LDO/LSO/LTO 不过。

下一步：停止 threshold tuning，进入 dataset-shift mechanism analysis。

### Route C：ExactTransferObjectiveMismatch

条件：P1/P4 显示 transfer 与好动作系统性失配。

下一步：transfer 只保留为 diagnostic，不进入 controller，不打开 direct update。

### Route D：HorizonTransferRescuesTransfer

条件：P4 strong pass。

下一步：允许 direct solved update sandbox。

### Route E：GeneratedRouteStopped

条件：P4/P5/P6 都没有新生成目标。

下一步：generated route 继续停止。

---

## 18. 本轮最重要的停止规则

以下情况必须停止小修：

```text
1. exact/windowed transfer 仍选不出 positive-value actions；
2. old rank 仍只能在 pooled 上好，LDO/LSO/LTO 不过；
3. Core77 expansion 仍只靠旧 rank 且 LDO 不过；
4. controller 只能靠 dataset-specific rule 才能过；
5. generated route 没有新数学目标；
6. selected runtime 没有 controller 就不跑；
7. paired replay 不在 controller + runtime pass 前打开。
```

---

## 19. 本轮最终判断标准

v9.7.4 不以“跑了多少表”为成功。最低有效推进是下面之一：

```text
1. 证明 Core77 本身稳定，只是 expansion10 失败；
2. 证明 one-step/windowed transfer 与好动作目标失配，并给出明确原因；
3. 找到 dataset-blind 的 expansion 规则，使 77 -> 87 且 LDO 过；
4. 找到 horizon-coupled transfer rule，能替代 current transfer；
5. 证明 existing-action route 暂时不可部署，generated route 继续停止；
6. 选出 minimal controller 并打开 selected runtime。
```

强成功是：

```text
existing-action controller pass；
selected runtime pass；
paired replay ready。
```

如果只得到：

```text
又一个 TopK 高但 LDO 不过；
又一个 AUC 高但 accepted region 不过；
又一个 generated primitive 工程过但 outcome 崩；
又一个 transfer score 低 LDO 但 value 负；
```

则不算实质成功。

---

## 20. 总结

v9.7.4 的核心不是继续调分数，而是解释失配：

```text
旧 rank 为什么好？
transfer 为什么杀掉旧 rank 的好动作？
Core77 为什么缺 10 个？
补的 10 个为什么跨数据集不稳？
windowed transfer 为什么仍然失败？
```

如果这些问题不解决，继续进入 controller / runtime / paired replay 都会是假成功。

本轮最重要的科学判断是：

$$
\boxed{
\text{现在项目不缺好动作样本，缺的是能解释并稳定选择这些好动作的简单原则。}
}
$$
