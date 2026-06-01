# DG-KAN v13.10：Cover-Forming Substrate Architecture Reset + Boundary-Conditioned Functional Update 完整计划

> 版本：v13.10 execution plan  
> 生成时间：2026-05-28  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；不使用 label-informed initialization；functional direction 不使用 validation / test / future / query batch；CEp99 / NLL / ECE / LineC hard target 只能作为审计与坏化约束，不能生成方向。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找一个只在某个 diagnostic score 上好看的 update。项目目标是：

$$
\boxed{
\text{构建一个 label-free strict FC-PureKAN substrate/base，}
\text{并通过 loss-interface-generic functional update}
\text{获得比普通 backprop / AdamW 更好的训练几何和模型。}
}
$$

最终成功必须同时满足：

```text
1. 表达力不打折，最好强于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何更好：真实信号进入 signal channel，噪声不进入 signal channel，真实信号不困在 reservoir；
5. functional update 的收益必须击败 AdamWParallel / RandomMatchedNorm / NoOp / SNR-only / MLP analog controls；
6. 不靠 label-informed initialization；
7. 不靠 CE-specific trick；
8. 不靠 dataset-specific tuning。
```

更简洁地说：

$$
\boxed{
\text{PureKAN substrate + boundary-conditioned functional update}
>
\text{same substrate + ordinary AdamW / backprop controls}.
}
$$

---

## 0.2 当前最新状态

v13.09 的最终状态不是 S3/S4/S5 success，而是：

```text
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
official_success_reached = 0
promotion_allowed = 0
kan_real_short_run_open_allowed = 0
```

已确认的正进展：

```text
1. Rational substrate 仍是唯一稳定 substrate family；
2. SNR signal transfer 没有完全丢失；
3. K0 retention / cosine 通过；
4. K8-K12 / K13 等 signal-to-cover 方法真实执行；
5. required artifacts 缺失为 0；
6. forbidden information violation 为 0；
7. v13.09 runner 不再是第一关 fail-fast。
```

已确认的失败：

```text
1. cover formation 没有成立；
2. cover_purity_median 远低于 gate；
3. KAN S3 只有 4/7 task family，没有达到 5/7；
4. S4 为 0；
5. K13 cover-debt guard focused repair 没有打开 X5/X6/X7；
6. Non-RAT exact substrate repair 仍为 0；
7. MLP control 不可写成 KAN promotion；
8. 当前 v13.09 functional-update runner 内继续加 K-token 已经没有价值。
```

所以当前核心 blocker 是：

$$
\boxed{
\text{coherent population-risk signal 存在并能部分转入 Rational basis/group，}
\text{但 Rational cover 没有形成稳定 specialization。}
}
$$

换句话说：

```text
不是没有 signal；
不是 signal 完全不能 lift 到 basis；
不是参数写不回；
不是 cover guard 没接上；
而是 basis cover architecture 本身不能稳定承载 signal。
```

---

# 1. 各条线进展百分比

下面百分比是研究完成度估计，不是 official artifact 字段。它综合考虑 gate、artifact 完整度、机制清晰度和离最终 claim 的距离。

| 线 | v13.8 后估计 | v13.09 后估计 | 变化 | 解释 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 99% | 0 | 工程闭包强，artifact 与 forbidden audit 干净 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强，但 label-informed init 禁用，不能 official |
| Label-free FHQ / A-DYN monitor | 8%-10% | 8% | -1 | 非主线，仅保留低预算 monitor |
| Line C：几何审计 | 88% | 88% | 0 | 审计可用，但不作为 direction source |
| Rational S1 efficient substrate | 85% | 85% | 0 | 唯一稳定 substrate family |
| Rational S1C channel controllability | 60% | 60% | 0 | 可执行 basis-channel movement，但 value 未成立 |
| PopRisk-SNR implementation | 86% | 88% | +2 | K8-K13 / transfer / cover telemetry 更完整 |
| K0 SNR signal transfer audit | 75% | 82% | +7 | retention / cosine 通过，说明 lift 不是主丢失点 |
| Signal-to-cover mechanism | 0%-5% | 12%-15% | +10 | K8-K13 有局部 source，但 cover formation fail |
| Cover formation / specialization | 新增 | 5%-10% | +5 | 已能测 cover purity/churn，但远未形成 |
| Generic MLP optimizer claim | 18%-22% | 15%-20% | -3 | MLP 不可作为 KAN promotion，generic claim 仍不稳 |
| KAN basis-cover functional official | 0%-5% | 0%-5% | 0 | S3/S4/S5 未达成 |
| Non-RAT substrate / S1C | 8%-15% | 5%-10% | -5 | exact substrate repair 仍未开，不能 functional proof |
| Classic no-BSpline portfolio 总体 | 58%-62% | 55%-60% | -3 | Rational 稳定，其它 basis 仍未成为 substrate |
| Substrate/base architecture reset | 新增 | 0% | 0 | 下一轮主线，尚未执行 |
| 整体 next-gen MLP claim | 38%-44% | 35%-42% | -3 | 继续下调；v13.09 证明 functional 小修路线到边界 |

结论：

$$
\boxed{
\text{v13.09 是定位进展，不是能力进展。}
}
$$

它清楚地证明：当前问题不是 SNR 信号丢失，而是 **signal 无法形成 stable basis cover**。

---

# 2. v13.09 独立分析

## 2.1 v13.09 做对了什么

v13.09 的问题不是执行太浅。它已经做了：

```text
1. K8 GradientClusterCover；
2. K9 CoverSplitMerge；
3. K10 ParamSNRThenCover；
4. K11 ReadoutBasisDecoupledSNR；
5. K12 Variance / LowVariance cover growth；
6. Case B/C focused repair；
7. K13 cover-debt guard repair；
8. Non-RAT exact substrate repair；
9. Rational mapped substrate direct signal-to-cover micro probes；
10. stop-contract 复核。
```

这些都没有打开 gate。因此不能再说：

```text
也许只是 K8-K12 没接 cover guard；
也许只是 X5/X6/X7 没 focused；
也许只是 Non-RAT 没补；
也许只是 Rational alternate substrate 没接入。
```

这些解释已经被当前 artifacts 排除。

---

## 2.2 v13.09 最重要的数据

official_v139：

```text
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7663904428482056
snr_transfer_median_cos_group_vs_param = 0.6665693521499634
cover_purity_median = 0.030393941327929497
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 4 / 7
kan_s4_task_pass_count = 0 / 7
nonrat_substrate_health_pass_count = 0
```

focused Case B/C repair：

```text
snr_transfer_gate_pass = 1
cover_purity_median = 0.03219109773635864
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

K13 cover-debt guard：

```text
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7194734811782837
snr_transfer_median_cos_group_vs_param = 0.6489424705505371
cover_purity_median = 0.03219109773635864
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

这些数字说明：

```text
1. signal retention 不是 0；
2. group cosine 也不是 0；
3. cover purity 极低；
4. K13 没修复 cover formation；
5. S4 完全没打开；
6. Non-RAT 没有可用 substrate；
7. 继续加 K-token 不会改变本质。
```

---

## 2.3 对“有没有进展”的判断

有进展，但不是模型能力进展。

v13.09 的科学价值是：

$$
\boxed{
\text{把 blocker 从“是否有 PopRisk signal / 是否能 lift 到 basis”}
\text{推进到“basis cover architecture 是否能形成 stable specialization”。}
}
$$

这比 v13.8 更清楚，但不更接近 success。因为最终 claim 需要的是：

```text
KAN S3 >= 5/7；
KAN S4 有 stable pass；
real short-run open；
S5 official。
```

当前全都没有。

---

# 3. 文献启发与当前路线重估

## 3.1 Generalization paper 的启发

Generalization paper 给出的核心分解是：

```text
signal channel:
  train motion 能 transfer 到 test；

reservoir:
  test-invisible；

failure mode 1:
  真实信号困在 reservoir；

failure mode 2:
  噪声进入 signal channel。
```

它还给了一个训练时可见的 population-risk signal：

$$
\bar g_B=\frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B=\frac{1}{b}\sum_i(g_i-\bar g_B)(g_i-\bar g_B)^T,
$$

$$
A_B=\bar g_B\bar g_B^T-\frac{1}{b-1}\Sigma_B.
$$

对角形式对应：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

这告诉我们：

```text
1. PopRisk-SNR 是合理 signal detector；
2. 它是 loss-interface generic，不是 CE-specific；
3. 它本身只告诉我们“哪里有 coherent signal”；
4. 它不能保证 KAN basis cover 会自动形成 stable specialization。
```

v13.09 正好验证了第 4 点。

---

## 3.2 Deep Manifold 的启发

Deep Manifold 的核心启发是：

```text
1. 网络坐标系是动态变化的；
2. node covers 是局部 piecewise-smooth manifold 单元；
3. boundary conditions 是迭代方向来源；
4. fixed-point regions 是训练中逐步构造出来的；
5. plasticity 先升后降，后期几何变硬。
```

这对 DG-KAN 的含义是：

$$
\boxed{
\text{basis 不是静态函数库，而是训练中形成的 cover / coordinate chart。}
}
$$

所以 functional update 不能只是：

```text
SNR mask；
cover guard；
一次性 signal-to-cover transform。
```

它应该配合一个 **能形成 cover 的 substrate architecture**。

v13.09 的失败说明：当前 Rational substrate 虽然可以算、可以动、可以接收 signal，但它没有足够的结构去形成 stable cover。现在需要重置 substrate/base architecture，而不是继续 functional-token 搜索。

---

# 4. 当前本质 blocker

当前 blocker 可以写成一句话：

$$
\boxed{
\text{coherent training signal 已经存在，}
\text{但当前 KAN substrate 缺少将 signal 分配到稳定 cover 的结构。}
}
$$

更具体：

```text
1. Rational substrate:
   可用，但 cover purity 极低；
   strong source rows 存在，但 family coverage 不足；
   cover specialization 没形成。

2. Non-RAT:
   还没有 exact/no-materialize + workspace + task-health substrate；
   不能进入 functional proof。

3. MLP:
   只能作为 generic optimizer control；
   不能替代 KAN-specific claim。

4. Functional update:
   当前 K8-K13 是 token-level cover manipulation；
   它没有改变 substrate 的 cover formation capacity。
```

因此继续：

```text
K14/K15；
G-token；
cover threshold；
EMA；
role norm；
freeze cluster；
weak cover；
oracle DeltaZ；
response dictionary；
```

都是小修。

---

# 5. v13.10 核心假设

v13.10 不再问：

```text
现有 Rational substrate 上，哪个 functional token 能把 S3 从 4/7 推到 5/7？
```

而是问：

```text
怎样的 label-free KAN substrate architecture 能在训练过程中形成 stable signal cover？
```

核心假设：

## H1：当前 Rational substrate 缺少 cover specialization inductive bias

Rational 现在有 denominator、slope、curvature、group diversity telemetry，但 group 之间没有足够强的分工压力。Population-risk signal 能到 group，但 group 没形成稳定任务 cover。

成立表现：

```text
signal_retention_group 高；
cover_purity 低；
cover_churn 低但 specialization 低；
source positive rows 分散；
family coverage 不足。
```

这正是 v13.09 观察到的状态。

## H2：需要 architecture-level cover formation，而不是 update-level cover repair

如果 substrate 没有 cover formation capacity，functional update 只能在坏坐标中小修。下一步需要新的 substrate：

```text
multi-cover Rational groups；
gradient-cluster initialized cover;
cover-memory EMA;
signal/reservoir split groups;
readout-basis decoupled groups;
cover competition / load balance;
delayed consolidation;
```

这些不是 loss modification，而是 KAN basis architecture / optimizer schedule。

## H3：basis-cover boundary schedule 必须成为训练过程

functional update 不应该是 one-shot。它应该是持续训练过程中的 boundary condition：

$$
\Delta \theta_t
=
-\eta_t
P_{\text{cover},t}
P_{\text{SNR},t}
M_{\text{basis},t}^{-1}
g_t.
$$

但 $P_{\text{cover},t}$ 不能只是 “guard”。它必须配合 architecture 让 cover 形成：

```text
Phase 1: plasticity-open；
Phase 2: cover specialization；
Phase 3: fixed-point consolidation。
```

## H4：Non-RAT 需要先做 substrate architecture，不应直接 functional

Chebyshev / Fourier / RBF / Wavelet 现在不是 functional failure，而是 substrate failure。它们必须先形成 exact/no-materialize + workspace + task-health vertical slice。

---

# 6. v13.10 实验总览

v13.10 分成七条线。

```text
Line R:
  Code / provenance / implementation readback。

Line A:
  Rational cover-forming substrate architecture reset。

Line K:
  Signal-to-cover training with cover-forming substrate。

Line N:
  Non-RAT substrate architecture vertical slices。

Line G:
  MLP generic PopRisk optimizer closure monitor。

Line C:
  Manifold-channel geometry audit。

Line Z:
  Finalizer / route / no-go boundary。
```

预算建议：

```text
Line A + K Rational cover-forming substrate: 50%
Line N Non-RAT substrate vertical slices: 20%
Line G MLP control monitor: 10%
Line C audit: 10%
Line R/Z infrastructure: 10%
```

---

# 7. Line R：代码与实现审计

## 7.1 目标

防止 Codex 把 functional-token 小修包装成 architecture reset。必须审查：

```text
1. 是否真的新增 substrate architecture；
2. 是否只是 K8-K13 alias；
3. 是否使用 label-informed init；
4. 是否使用 CEp99/NLL/ECE/LineC 生成方向；
5. 是否使用 validation/test/future/query；
6. 是否把 Non-RAT scout/proxy 写成 exact no-materialize success。
```

## 7.2 必须输出

```text
v1310_code_review_manifest.csv
v1310_architecture_diff_manifest.csv
v1310_forbidden_information_audit.csv
v1310_substrate_architecture_readback.md
v1310_route_decision.json
```

## 7.3 Gate

```text
uses_y_for_stats = 0
label_informed_init = 0
ce_tail_direction = 0
linec_direction = 0
validation_test_future_direction = 0
dataset_name_branch = 0
is_new_substrate_architecture = 1
is_k_token_only_extension = 0
```

如果 `is_k_token_only_extension = 1`，route 直接：

```text
R0-TokenSearchNotArchitectureReset
```

---

# 8. Line A：Rational cover-forming substrate architecture reset

## 8.1 目标

构造能形成 stable cover 的 Rational substrate，而不是继续修 K8-K13。

## 8.2 候选

### A-RCF1：MultiBand Rational Cover

把 Rational groups 初始化成不同 curvature / slope / denominator band：

```text
low-slope stable groups
medium plastic groups
high-curvature exploratory groups
reservoir groups
```

要求 label-free：

```text
不使用标签；
不使用 dataset name；
只用 fixed seed + train-stream input scale。
```

记录：

```text
band_id
den_p01_by_band
r_prime_p99_by_band
r_double_prime_p99_by_band
group_diversity_by_band
cover_load_by_band
```

### A-RCF2：SNR-Cluster Cover Warmup

在前若干 train steps 内，基于 per-example gradient SNR 聚类形成 cover assignment。

注意：

```text
这不是 label-informed init；
这是 training boundary 使用 generic loss interface。
```

记录：

```text
cluster_count
cluster_stability
cluster_snr_mean
cluster_snr_var
cluster_to_group_assignment_entropy
```

### A-RCF3：Persistent Cover Memory

为每个 Rational group 维护 EMA：

```text
ema_group_snr
ema_group_load
ema_group_purity_proxy
ema_group_churn
ema_signal_mass
ema_noise_proxy
```

并用它决定 group plasticity。

### A-RCF4：Signal / Reservoir Split Groups

明确把 groups 分成：

```text
signal_candidate_groups
reservoir_candidate_groups
exploration_groups
```

不是用 label / LineC 定义，而是用 train-stream SNR / variance / activation stability 定义。

### A-RCF5：Readout-Basis Decoupled Cover

解决 readout_signal_mass 压过 basis cover 的问题。把 readout fast adaptation 和 basis slow cover formation 分开：

```text
readout:
  fast AdamW / SNR blend

basis:
  slower SNR + cover schedule

cross-coupling:
  trust-limited
```

### A-RCF6：Overcomplete Cover Bank + Sparse Activation Budget

用更多 groups，但每步只激活部分 cover：

```text
overcomplete_group_count = 2x / 4x
active_group_budget = fixed fraction
selection = SNR + load balance
```

目的是让 cover 有足够容量分化，但不让 runtime 爆。

## 8.3 记录指标

```text
candidate_id
family = Rational
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
mean_delta_vs_adamw
worst_delta_vs_adamw
AUC_time_ratio
CEp99_delta
NLL_delta
ECE_delta

snr_retention_group
cos_group_vs_param
cover_purity_mean
cover_purity_p10
cover_churn
cover_load_gini
cover_specialization_entropy
signal_to_cover_score
false_drop_fraction
false_keep_fraction
group_alive_fraction
group_dead_fraction
readout_signal_mass
basis_signal_mass
reservoir_group_load
exploration_group_load

LineC_CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_majority_pass
```

## 8.4 Substrate-cover gate

A candidate passes cover-substrate gate if:

$$
workspace\_raw\_ratio \le 1.10,
$$

$$
workspace\_incremental\_ratio \le 1.75,
$$

$$
step\_ratio \le 1.75,
$$

$$
mean\_delta \ge -0.05,
$$

$$
worst\_delta \ge -0.10,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
median(signal\_retention\_group)\ge0.70,
$$

$$
median(cos\_group\_vs\_param)\ge0.60,
$$

$$
median(cover\_purity)\ge0.15,
$$

$$
mean(cover\_churn)\le0.50.
$$

注意：这不是 final S5，只是允许进入 Line K。

---

# 9. Line K：Signal-to-cover training on cover-forming substrate

## 9.1 目标

检验新的 cover-forming substrate 是否能把 PopRisk signal 稳定变成 task-family coverage。

## 9.2 Methods

```text
K0-RAT-AdamW
K1-ParameterSNR
K2-ParameterSNR-MinCoverGuard
K3-RoleWiseSNRLift
K8-GradientClusterCover
K10-ParamSNRThenCover
K13-CoverDebtGuard
K14-RCF-SNRClusterTraining
K15-RCF-PersistentCoverMemory
K16-RCF-SignalReservoirSplit
K17-RCF-ReadoutBasisDecoupled
K18-RCF-OvercompleteSparseCover
```

关键：K14-K18 只能在 A-RCF substrate 上跑。

## 9.3 Synthetic setup

```text
tasks = X1..X7
seeds = 0,1,2
loss_interfaces = CE,Brier
train_steps = 200
batch_size = 32
```

## 9.4 S3 gate

Task family pass requires:

```text
within each task:
  >=2 seeds pass or >=2 loss interfaces pass
```

Overall S3:

```text
>=5/7 synthetic task families pass
```

Row-level pass:

$$
source\_vs\_adamw \ge 0.005,
$$

$$
AUCtime\_ratio \le 1.0,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02,
$$

$$
LineC\_majority\_pass = 1.
$$

## 9.5 S4 gate

S4 requires:

```text
>=5/7 synthetic task families pass
and LineC all/near-all stable:
  LineC pass count >= 4/5 in multisketch audit
and cover formation gate pass
```

## 9.6 Failure-driven fallback

If `snr_retention` passes but `cover_purity` fails:

```text
run A-RCF2/A-RCF3/A-RCF6;
do not tune K threshold.
```

If `cover_purity` passes but task source fails:

```text
run readout-basis decoupling;
check readout_signal_mass vs basis_signal_mass.
```

If task source passes but LineC fails:

```text
run signal/reservoir split groups;
increase reservoir group load;
audit NoiseSignalLeak.
```

If X5 tail fails:

```text
do not use CEp99 direction;
activate tail-proxy boundary only:
  logit entropy floor
  high-curvature rational group veto
  denominator slope guard
```

If X6/X7 seed instability persists:

```text
run persistent cover memory and group load smoothing.
```

---

# 10. Line N：Non-RAT substrate architecture vertical slices

## 10.1 目标

Non-RAT 不进入 functional proof，除非先成为 substrate.

## 10.2 Families and candidates

### Fourier

```text
N-FOU1 FrequencyBandCover-lowK
N-FOU2 FrequencyBandCover-withPhaseStability
N-FOU3 LowFreqSignalHighFreqReservoirSplit
```

### Chebyshev

```text
N-CHE1 DegreeEnergyCover-K3K5
N-CHE2 HighDegreeLateEnable
N-CHE3 DegreeReservoirSplit
```

### RBF / FastKAN

```text
N-RBF1 CenterOccupancyCover
N-RBF2 WidthConditionGuard
N-RBF3 OutOfGridReservoirBoundary
```

### Wavelet

```text
N-WAV1 ScaleSupportCover
N-WAV2 LocalTailCoverage
N-WAV3 SupportOverlapGuard
```

## 10.3 Gate

A Non-RAT candidate must pass:

$$
workspace\_raw\_ratio \le 1.10,
$$

$$
workspace\_incremental\_ratio \le 1.75,
$$

$$
step\_ratio \le 1.75,
$$

$$
mean\_delta \ge -0.05,
$$

$$
worst\_delta \ge -0.10,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.30.
$$

Only then can it enter functional Line K in later versions.

---

# 11. Line G：MLP generic optimizer control

## 11.1 Goal

MLP is not the main claim, but it remains a control. We need to know whether PopRisk-SNR is simply a generic optimizer.

## 11.2 Methods

```text
MLP-AdamW
MLP-best-SNRBlend
MLP-SNRBlend-with-trust
MLP-SNRBlend-with-active-fraction-schedule
```

No new G-token grid unless there is a new hypothesis.

## 11.3 Gate

MLP generic optimizer confirm requires:

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
each dataset >=6/10 seed pass
3/3 datasets pass
```

If MLP passes and KAN fails:

```text
Generic optimizer works; KAN-specific basis advantage not established.
```

If MLP fails and KAN passes:

```text
KAN explicit basis cover provides functional leverage.
```

---

# 12. Line C：Geometry audit only

Line C records:

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
CEp99
NLL
ECE
Brier
margin_p10
kernel_drift
```

Line C cannot generate direction.

If any candidate improves source but worsens LineC:

```text
diagnose signal/reservoir split;
do not promote.
```

---

# 13. Required artifacts

```text
v1310_code_review_manifest.csv
v1310_architecture_diff_manifest.csv
v1310_forbidden_information_audit.csv
v1310_substrate_architecture_readback.md

v1310_rational_cover_substrate.csv
v1310_rational_cover_telemetry.csv
v1310_signal_to_cover_training.csv
v1310_signal_to_cover_summary.csv
v1310_signal_to_cover_linec.csv

v1310_nonrat_substrate_vertical_slice.csv
v1310_mlp_control_monitor.csv
v1310_linec_audit.csv
v1310_failure_table.csv
v1310_route_decision.json
v1310_no_go_boundary.md
v1310_next_hypothesis_queue.md
```

---

# 14. Required visualizations

```text
fig_progress_by_line.svg
fig_signal_retention_vs_cover_purity.svg
fig_cover_purity_by_method.svg
fig_cover_churn_by_method.svg
fig_signal_to_cover_score_by_task.svg
fig_task_family_pass_heatmap.svg
fig_linec_vs_source_scatter.svg
fig_readout_vs_basis_signal_mass.svg
fig_nonrat_substrate_health_matrix.svg
fig_mlp_vs_kan_control.svg
fig_failure_taxonomy_sankey.svg
```

---

# 15. Route definitions

```text
R0-TokenSearchNotArchitectureReset:
  New runner only adds K-token / alias, no new substrate architecture.

R1-NoCoverSubstrate:
  No candidate passes substrate-cover gate.

R2-SignalRetentionPassCoverFormationFail:
  signal retention/cos pass, cover formation fails.

R3-CoverFormationPassTaskFamilyFail:
  cover formation passes, synthetic S3 < 5/7.

R4-TaskFamilyPassLineCFail:
  synthetic S3 passes, LineC / tail / calibration fails.

R5-NonRATStillNoSubstrate:
  Rational remains only substrate and Non-RAT vertical slices all fail.

R6-GenericMLPOnly:
  MLP generic optimizer passes, KAN cover does not.

S1-CoverSubstrate:
  at least one candidate passes substrate-cover gate.

S2-CoverTrainingSignal:
  cover substrate + >=3/7 task family pass.

S3-KANSynthetic:
  >=5/7 synthetic task family pass.

S4-KANLineCStable:
  S3 + LineC multisketch stable.

S5-OfficialRealShortRun:
  S4 + real 3x3 short-run + controls + tail/calibration pass.
```

---

# 16. Stop / continue policy

Codex must not stop after R2 unless it has executed:

```text
1. A-RCF1..A-RCF6 scout;
2. top-2 A-RCF hardening;
3. K14-K18 training;
4. Non-RAT at least Fourier + Chebyshev vertical slices;
5. MLP control monitor;
6. final no-go boundary;
7. next hypothesis queue.
```

If all are executed and no S2, final stop is allowed.

Codex must not execute:

```text
K19/K20 token extension;
O7/O8/O9 oracle fallback;
response dictionary fallback;
BN/BM metric fallback;
LineC direction fallback;
CE-tail direction fallback.
```

---

# 17. Final expected outcomes

## Best case

```text
A-RCF substrate passes cover gate;
K14-K18 reaches S3 >=5/7;
LineC stable;
real short-run opens.
```

## Useful negative

```text
A-RCF substrates also fail cover formation;
then current Rational substrate architecture is insufficient.
Next must be stronger substrate/base architecture, not functional update.
```

## Hard no-go

```text
Rational cover architecture reset fails;
Non-RAT substrate vertical slices fail;
MLP generic control also fails or remains irrelevant.
Conclusion:
  current no-BSpline FC-PureKAN basis set has no functional substrate under this protocol;
  must redesign primitive family or relax scope.
```

---

# 18. 一句话总结

v13.10 的核心不是继续找 functional token，而是：

$$
\boxed{
\text{从 signal-to-cover update 转向 cover-forming substrate architecture。}
}
$$

v13.09 已经证明：

```text
signal retention 成立；
cover formation 失败；
当前 K-token 小修无效。
```

所以 v13.10 必须回答：

$$
\boxed{
\text{Rational / Non-RAT basis 能不能在 label-free、loss-interface-generic 训练中形成 stable cover？}
}
$$

如果不能，functional update 主张必须收缩，项目应转回 substrate / primitive architecture design。
