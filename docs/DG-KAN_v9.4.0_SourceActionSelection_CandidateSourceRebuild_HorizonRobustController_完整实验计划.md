# DG-KAN v9.4.0 Source Action Selection / Candidate Source Rebuild / Horizon-Robust Controller 完整实验计划

> 本计划基于 v9.3.9 `Generated-AP Failure Decomposition / Value-Preserving Certificate Primitive` 的真实执行结果制定。  
> v9.4.0 不再继续 AP1-AP8 transform threshold 修补，也不再继续 certificate pass/fail 小修。  
> 本轮要回答更上游的问题：**为什么 AP generator 的 source actions 本身就是 value-poor / long-risk high？是否能构造一个 legal、dataset-agnostic、commit-time 可用的 source action selector 或重新生成一个 value-producing candidate source？**

---

# 0. 执行摘要

v9.3.9 的核心进展不是 system pass，而是把 v9.3.8 的 failure 向前追溯了一层。

v9.3.8 证明：

```text
AP1-AP4 已真实生成 payload / certificate / action apply / branch-horizon outcomes；
但 generated AP certificate-pass rows 没有形成 value-positive / horizon-safe frontier。
```

v9.3.9 进一步证明：

```text
同源 AP0 source panel 本身 weak CP precision 只有 0.0729167；
strong CP precision 只有 0.03125；
horizon-robust precision = 0.0；
long-risk rate = 0.307292；
V_ctrl LCB = -1.498012；
```

这说明：

$$
\boxed{
\text{当前 AP failure 的更早 blocker 是 source action selection / candidate source，而不是 AP transform 本身。}
}
$$

但这并不等于“AP0 全体没有好 action”。v9.3.5/v9.3.6 已经显示 full AP0 universe 中存在 weak control-positive oracle frontier。v9.3.9 的关键新事实是：**被拿来生成 AP1-AP8/AP5-AP8 的 source panel 很差**。因此 v9.4.0 的核心不是继续改 AP transform，而是重建 source action selection / candidate source。

本轮的总目标是：

$$
\boxed{
G_{source}(s_t,b_t,g_t,\Delta\theta_{AdamW})
\rightarrow
\{(\Delta\theta_i, Cert_i, Cost_i)\}_{i=1}^{K}
}
$$

其中 $G_{source}$ 必须在 commit 前生成或选择 source actions，并让这些 source actions 在 immediate horizon 上有真实 positive direction，在 longer horizons 上不过度 long-risk，同时保持 no teacher、no loss modification、no dataset-specific tuning、manual training contract。

---

# 1. v9.3.9 独立判断

## 1.1 这轮有进展，但不是 functional success

v9.3.9 的进展有三层。

第一，它补了 v9.3.8 缺失的同源 AP0 baseline。v9.3.8 只看到 generated AP rows 不好，容易误判为 AP1-AP4 transform 破坏了原本好的 source actions。v9.3.9 用同一 source panel 跑 AP0 baseline，发现 source AP0 自身就只有：

```text
source_AP0_weak_CP_precision = 0.07291666666666667
source_AP0_strong_CP_precision = 0.03125
source_AP0_horizon_robust_CP_precision = 0.0
source_AP0_long_risk_rate = 0.3072916666666667
source_AP0_V_ctrl_lcb = -1.4980123384538895
```

这把 root cause 从 “generated AP bad” 推到了 “source panel bad”。

第二，它量化了 AP transform damage，但没有让 damage 成为 primary blocker：

```text
paired_horizon_count = 768
Damage_median = 0.04009834537282586
source_positive_lost_after_generation_rate = 0.9107142857142857
generator_damage_pass = 0
```

`source_positive_lost_after_generation_rate = 0.9107` 很严重，说明 transform 不是完全无辜；但由于 source panel 自身 already bad，不能把 v9.3.8 failure 主要归因于 AP1-AP4 transform。

第三，它测试了 AP5-AP8 low-distortion / value-preserving attempt：

```text
generated_action_count = 256
certificate_pass_action = 64
action_apply_error_linf_max = 0.0
branch_horizon_rows = 6912
completion = 1.0
cert_pass_weak_CP_precision = 0.07291666666666667
cert_pass_strong_CP_precision = 0.026041666666666668
cert_pass_horizon_robust_CP_precision = 0.0
cert_pass_long_risk_rate = 0.296875
cert_pass_V_ctrl_lcb = -1.4386910355299591
```

低扰动生成和 action apply 物理闭合了，但 value 没有回来。这说明“尽量保留 source”不能解决问题，因为 source 本身就很弱。

## 1.2 最重要的机制发现

v9.3.9 的核心机制发现是：

$$
\boxed{
\text{AP primitive 失败不是先从 transform 开始，而是从 source selection 开始。}
}
$$

更具体：

```text
1. full AP0 universe 曾经存在 weak CP oracle frontier；
2. 但 v9.3.8/v9.3.9 用来生成 AP 的 64 个 source actions 不是这个 frontier；
3. 这个 source panel immediate direction already weak；
4. long horizon 又有极高 risk；
5. certificate 只能给 weak CP 一点 lift，不能控制 long-risk；
6. AP5-AP8 low-distortion generation 只能保留 bad source 的性质。
```

因此下一步的重点不是：

```text
AP3 threshold；
certificate threshold；
support_lcb threshold；
low-distortion epsilon；
再换一个 AP transform；
```

而是：

```text
source action 是怎么被选出来的？
full AP0 universe 中 oracle-good actions 为什么没进入 source panel？
legal commit-time observable 是否能选择好 source？
如果 legal selector 不行，能否生成 value-positive source action，而不是从 AP0 opaque pool 中猜？
```

## 1.3 当前最深 blocker

当前最深 blocker 不是 controller、runtime 或 dataset，而是：

$$
\boxed{
\text{source action selection / candidate source cannot produce enough immediate-positive, horizon-safe actions.}
}
$$

它有四个子问题：

```text
B1. Source panel representativeness failure：64 个 source actions 是否只是 bad sample / biased panel？
B2. Source selector legal observability failure：如果 full universe 中有 good actions，legal selector 是否能找到？
B3. Source generator failure：如果 legal selector 找不到，能否生成 by-construction immediate descent source action？
B4. Horizon fragility：即使 h20 有 positive direction，h80/h240 是否会变成 long-risk？
```

v9.4.0 必须把这四个问题拆开测，不能再把它们混在 “AP generated value fail” 里。

---

# 2. v9.4.0 总体目标

v9.4.0 的总体目标是：

$$
\boxed{
\text{重建 source action selection / candidate source，使 AP primitive 的输入本身具有 immediate-positive 与 horizon-safe 潜力。}
}
$$

具体目标包括：

```text
1. 判断 v9.3.9 的 64-action source panel 是 bad sample、bad selector，还是 candidate source 本身不足；
2. 在 full AP0 outcome universe 上建立 source selection oracle upper bound；
3. 构造 legal source selector，评估 commit-time features 是否能选择 source actions；
4. 如果 legal selector 仍失败，直接实现新的 value-producing source generators AP0b/AP0c/AP0d/AP0e/AP0f；
5. 先过 immediate h20 direction gate，再扩展到 h80/h240；
6. 只在 source survivor 上重新测试 AP transform damage；
7. 将 certificate 从 construction-valid 改为 effect-valid；
8. 若 source + certificate + controller 过线，再测 selected runtime；
9. 全程不按 dataset 调参，只做 dataset/family/horizon diagnostic。
```

v9.4.0 不以 full success 为最低目标。最低有效推进是：

```text
1. source panel failure 被定性为 selection bias / legal selector failure / candidate source absence 三者之一；
2. 至少一个 source selector 或 source generator 在 h20 immediate direction 上显著超过 v9.3.9 source panel；
3. 若 h20 pass，至少一个 survivor 进入 h80/h240 long-risk audit；
4. 若 source survivor 存在，重新评估 AP transform damage；
5. route 明确进入 source-selector-pass / source-generator-pass / source-oracle-absent / horizon-risk-fail / runtime-fail 之一。
```

---

# 3. 硬约束

本轮继续遵守：

```text
no teacher
no self-teacher
no distillation
no auxiliary loss
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official selector/controller 不使用 dataset_name 分支
不使用 validation/test metric at commit time
不使用 future outcome at commit time
不把 diagnostic oracle 写成 official controller
不把 runtime microbench 写成 selected-controller runtime
```

Functional update 仍是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW-equivalent}
+
\Delta\theta_{functional}.
$$

任务 loss 仍是标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

允许做：

```text
full AP0 outcome universe oracle analysis；
source panel representativeness diagnostic；
family / horizon / dataset diagnostic；
legal commit-time feature evaluation；
calibration-frozen source selector；
new source candidate generator；
matched-control branch-horizon materialization；
parallel smoke experiments；
offline materializer batching；
selected-controller online runtime measurement。
```

但不允许：

```text
按 dataset 调 source selector；
按 dataset 调 generator norm；
按 dataset 调 certificate threshold；
用 source outcome labels 训练 commit-time decision without cross-fitting；
把 oracle source selector 当 official selector；
只报告 AP transform，不报告 source AP0 same-panel baseline；
在 h20 immediate fail 时继续跑 expensive full horizons 并声称有潜力；
在 no selected controller 时声明 runtime pass。
```

---

# 4. 核心定义

## 4.1 Source action

source action 是 AP generator 的输入 action：

$$
a_s = (\Delta\theta_s, meta_s, payload\_hash_s).
$$

v9.3.9 说明当前 source actions 不够好。v9.4.0 将 source action 拆成两类：

```text
SelectedSourceAction:
  从 AP0 full universe 中选择出来的 source action。

GeneratedSourceAction:
  由新的 source generator 直接生成的 action，不依赖 AP0 opaque selector。
```

## 4.2 Weak / strong / horizon-robust CP

Weak CP：

$$
WeakCP(e,h)=1
\iff
RealFunctional(e,h)
>
\max_{b\in Controls}V_b(e,h)
\land
BadEvent(e,h)=0
\land
NullEvent(e,h)=0.
$$

Strong CP：

$$
StrongCP(e,h)=1
\iff
WeakCP(e,h)=1
\land
V_{ctrl}(e,h)\ge \tau_V
\land
Support(e,h)\ge\tau_S.
$$

Horizon-robust CP：

$$
HorizonRobustCP(e)=1
\iff
WeakCP(e,20)=1
\land WeakCP(e,80)=1
\land WeakCP(e,240)=1
\land LongRisk(e)=0.
$$

Control value：

$$
V_{ctrl}(e,h)
=
V_{RealFunctional}(e,h)
-
\max_{b\in Controls}V_b(e,h).
$$

Controls 至少包括：

```text
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

## 4.3 Immediate direction

v9.3.9 显示 h20 already weak，因此 v9.4.0 先定义 immediate direction gate：

$$
ImmediatePositive(e)=1
\iff
V_{ctrl}(e,20)>0
\land
BadEvent(e,20)=0
\land
NullEvent(e,20)=0.
$$

如果 immediate direction fail，则不应继续投入大规模 AP transform / long horizon / controller。

## 4.4 Source selector

source selector 是 commit-time legal function：

$$
Sel_{source}(e)=f(x_e),
$$

其中 $x_e$ 只能包含 commit 前可得量：

```text
state CE / NLL / margin / tail stats；
manual gradient and AdamW direction；
payload geometry available before outcome；
linearized CE/margin effect；
family/horizon support from calibration split；
source generation cost；
```

不得包含：

```text
WeakCP label；
StrongCP label；
V_ctrl realized；
heldout branch outcome；
dataset_name；
validation/test metric；
future step information。
```

---

# 5. 核心假设

## H0：v9.3.9 source panel 是 source selection failure，不是 full AP0 universe absence

H0 认为 full AP0 universe 有 good actions，但 v9.3.9 的 source panel 没选到它们。

H0 成立标准：

```text
full AP0 universe oracle source selector at K=64/256 can produce:
  weak_CP_precision >= 0.50
  V_ctrl_lcb > 0
  long_risk_rate <= 0.10

while v9.3.9 source panel:
  weak_CP_precision = 0.0729
  V_ctrl_lcb < 0
  long_risk_rate ≈ 0.3073
```

H0 失败标准：

```text
even oracle source selector cannot produce a good K=64/256 panel。
```

若 H0 失败，说明 AP0 candidate source 本身不足，不能继续 source selection；必须 redesign source generator。

## H1：current 64-action panel 存在 representativeness bias

H1 认为 v9.3.9 的 64 source actions 不是 full AP0 candidate universe 的代表样本，可能来自排序、早期 step、family concentration、certificate precondition 或 convenience slice。

H1 成立标准：

```text
panel PSI / KL / max stratum gap 超过阈值；
source panel family/horizon/step/score distribution 与 full universe 明显不同；
importance-weighted CP estimate 与 full universe 差距 > 0.03；
```

H1 失败标准：

```text
source panel 与 full universe 代表性良好，但仍 weak CP low。
```

若 H1 失败，问题不是 sampling，而是 legal source selection 或 source generator。

## H2：legal source selector 能否从 full AP0 universe 找到 good actions

H2 认为如果 good AP0 actions 在 full universe 中存在，那么 commit-time legal features 可能能选到一部分。

H2 成立标准：

```text
legal source selector top64:
  h20 weak_CP_precision >= 0.25
  all-horizon weak_CP_precision >= 0.25
  V_ctrl_lcb > 0
  long_risk_rate <= 0.15

legal source selector top256:
  weak_CP_precision >= 0.18
  V_ctrl_lcb > 0
  support balance pass
```

H2 失败标准：

```text
oracle source selector pass，但 all legal source selectors fail。
```

若 H2 失败，说明 AP0 good actions are legally opaque for source selection；必须生成 by-construction source actions。

## H3：source generator 必须先产生 immediate-positive action

H3 认为 v9.3.9 的 immediate direction already fail，下一轮 source generator 必须先在 h20 上证明方向正确。

H3 成立标准：

```text
new source generator h20:
  ImmediatePositive precision >= 0.25
  h20 V_ctrl_lcb > 0
  h20 bad_event_rate <= 0.05
  h20 null_rate <= 0.20
  h20 weak_CP precision 至少是 v9.3.9 source panel 的 4 倍
```

v9.3.9 source panel h20 weak CP precision = 0.02734375，因此 4 倍阈值为：

$$
4 \times 0.02734375 = 0.109375.
$$

本轮 stronger target 设为：

$$
P(WeakCP_{h20}) \ge 0.25.
$$

H3 失败标准：

```text
all source generators h20 V_ctrl_lcb <= 0 or h20 weak CP precision < 0.10。
```

若 H3 失败，不进入 h80/h240。

## H4：horizon-risk 不能靠 posthoc threshold 修复

H4 认为 long-risk 是 generator/certificate 的内生属性，不能等到 controller 末端才过滤。

H4 成立标准：

```text
survivor generators across h20/h80/h240:
  long_risk_h240 <= 0.15
  horizon_robust_action_rate >= 0.03
  V_ctrl_lcb_all_horizons > 0
```

H4 失败标准：

```text
h20 pass but h240 long_risk > 0.30 or horizon_robust rate = 0。
```

## H5：AP transform 只能在 good source panel 上评价

H5 认为在 source panel already bad 的情况下，评价 AP transform 没意义。必须先拿 source-survivor panel 再测 transform damage。

H5 成立标准：

```text
source survivor exists；
AP transform generated rows:
  source_positive_lost_rate <= 0.30
  Damage_median >= -0.01
  generated weak CP precision >= 0.85 * source weak CP precision
  generated long_risk <= source long_risk + 0.05
```

H5 失败标准：

```text
source survivor good，但 AP transform destroys >50% positives。
```

若 H5 失败，AP transform redesign 才是主线。

## H6：certificate 必须 effect-valid，不只是 construction-valid

H6 认为 certificate 必须预测 value/risk/horizon outcome，而不是只证明 payload/hash/schema 合法。

H6 成立标准：

```text
AUC_certificate_weak_CP >= 0.70
P(weak_CP | cert_pass) >= 0.50 for source-survivor smoke
P(long_risk | cert_pass) <= 0.50 * P(long_risk | cert_fail)
certificate ECE <= 0.08
monotone sign pass = 1
```

H6 失败标准：

```text
certificate weak lift exists but AUC < 0.65 or long-risk unchanged。
```

v9.3.9 中：

```text
AUC_certificate_weak_CP = 0.5964
Lift_weak = 2.2656
Lift_longrisk = 0.9786
```

因此 v9.3.9 的 certificate 不是 sufficient statistic。

## H7：runtime 只有在 selected source/controller 存在后才 official

H7 认为没有 selected controller 时，payload runtime 只能 microbench，不得 official。

H7 成立标准：

```text
selected source generator / selector exists；
selected certificate controller exists；
measured online path includes feature + source generation/selection + certificate + payload apply + base step；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05。
```

H7 失败标准：

```text
only microbench or no selected controller。
```

---

# 6. v9.4.0 实验阶段

---

## P0：Boundary reanalysis and full-vs-panel source diagnosis

### 目标

把 v9.3.9 的 source panel 与 v9.3.5/v9.3.6 full AP0 universe 放在同一个坐标系里。P0 不做新训练，只做数据分析。

### 实现

读取：

```text
v9350 full_control_outcome_table
v9360 CP label decomposition
v9380 generated AP source mapping
v9390 source AP0 same-panel baseline
```

建立统一表：

```text
source_panel_flag
full_universe_flag
source_selection_reason
source_action_id
candidate_id
event_id
family_id
horizon
step
seed
dataset
payload_hash
WeakCP_h20 / h80 / h240
StrongCP_h20 / h80 / h240
LongRisk_h240
V_ctrl_h20 / h80 / h240
```

### 必须记录

```text
full_action_count
source_panel_action_count
source_panel_fraction
full_weak_CP_action_rate
source_panel_weak_CP_action_rate
full_weak_CP_row_rate
source_panel_weak_CP_row_rate
full_strong_CP_action_rate
source_panel_strong_CP_action_rate
full_horizon_robust_action_rate
source_panel_horizon_robust_action_rate
full_long_risk_rate
source_panel_long_risk_rate
full_V_ctrl_mean
source_panel_V_ctrl_mean
full_V_ctrl_lcb
source_panel_V_ctrl_lcb
selection_bias_PSI
selection_bias_KL
max_family_gap
max_horizon_gap
max_step_bucket_gap
max_score_bucket_gap
importance_weighted_source_CP_estimate
route_flip_full_vs_source_panel
```

### 判断标准

P0 pass：

```text
all full/panel rows joined by action_id or payload_hash；
source panel metrics reproduced；
full vs panel gap quantified；
source panel bias classification assigned。
```

Source panel bias pass：

```text
selection_bias_PSI <= 0.10
and max_stratum_gap <= 0.15
and route_flip_full_vs_source_panel = 0
```

若 source panel bias fail，则 route candidate：

```text
R1-SourcePanelSamplingOrSelectionBias
```

### 可视化

```text
p0_full_vs_source_panel_cp_rates.svg
p0_source_panel_representativeness_heatmap.svg
p0_family_horizon_step_distribution_shift.svg
p0_full_vs_panel_Vctrl_distribution.svg
p0_route_flip_dashboard.svg
```

### Artifacts

```text
p0_full_vs_source_panel_diagnosis.csv
source_panel_join_trace_v9400.csv
source_panel_bias_trace_v9400.csv
```

---

## P1：Source selection provenance audit

### 目标

回答 v9.3.8/v9.3.9 的 64 个 source actions 到底是怎么选出来的。这个问题如果不清楚，后续任何 AP primitive 结论都可能被 source sampling 污染。

### 实现

对每个 source action 重建 provenance：

```text
是否来自 top-k？
是否来自 earliest steps？
是否来自某个 family/horizon bucket？
是否来自 certificate diagnostic pass？
是否来自 payload availability？
是否来自 smoke convenience slice？
是否来自 runtime feasible slice？
是否来自 high StableAccept score？
是否来自 AP0 feature/probe score？
```

### 必须记录

```text
source_action_id
candidate_id
event_id
selection_rule_id
selection_rank
selection_score
selection_stage
eligible_pool_size_at_selection
selected_count_at_stage
selection_probability_estimate
source_selection_reason_tag
source_selection_commit_time_features
source_selection_uses_outcome
source_selection_uses_dataset_name
source_selection_uses_future_step
payload_availability_flag
certificate_availability_flag
```

Summary：

```text
selection_rule_identified
unknown_selection_reason_count
uses_outcome_in_source_selection
uses_dataset_name_in_source_selection
source_panel_family_concentration
source_panel_step_concentration
source_panel_score_concentration
source_panel_selection_bias_pass
```

### 判断标准

P1 pass：

```text
unknown_selection_reason_count = 0
source_selection_uses_outcome = 0 for official source selector
source_selection_uses_dataset_name = 0
source_selection_uses_future_step = 0
selection_rule_identified = 1
```

若 source panel 是 convenience slice 或 biased subset，则 v9.3.8/v9.3.9 只能判定 “that source panel bad”，不能判定 candidate source 全体 bad。

### 可视化

```text
p1_source_selection_flow_sankey.svg
p1_selection_score_vs_CP_label.svg
p1_source_panel_step_family_horizon_concentration.svg
p1_eligible_pool_vs_selected_panel.svg
```

### Artifacts

```text
p1_source_selection_provenance_audit.csv
source_selection_trace_v9400.csv
source_selection_unknown_reason_table_v9400.csv
```

---

## P2：Oracle source selector upper-bound test

### 目标

判断 “如果允许使用 outcome oracle，full AP0 universe 是否能选出好的 source panel”。这是区分 source selection failure 与 candidate source absence 的关键。

### Oracle selectors

```text
OS0-current-v9390-source-panel
OS1-random-stratified-source-panel
OS2-weak-CP-oracle-source-selector
OS3-strong-CP-oracle-source-selector
OS4-horizon-robust-oracle-source-selector
OS5-value-risk-pareto-oracle-source-selector
OS6-support-balanced-oracle-source-selector
```

Oracle score：

$$
S_{oracle}(e)
=
V_{ctrl}(e,20)
+\alpha V_{ctrl}(e,80)
+\beta V_{ctrl}(e,240)
-\lambda_L LongRisk(e)
-\lambda_B BadEvent(e)
-\lambda_N NullEvent(e).
$$

默认：

```text
alpha = 0.5
beta = 0.25
lambda_L = 2.0
lambda_B = 2.0
lambda_N = 0.5
```

这些只用于 oracle diagnostic，不进入 official selector。

### Panel sizes

```text
K = 64, 128, 256, 512
```

### 必须记录

```text
oracle_selector_id
K
accepted_action_count
weak_CP_precision_h20
weak_CP_precision_h80
weak_CP_precision_h240
weak_CP_precision_all_rows
strong_CP_precision_all_rows
horizon_robust_action_rate
long_risk_rate
bad_event_rate
null_rate
V_ctrl_mean_h20
V_ctrl_lcb_h20
V_ctrl_mean_all
V_ctrl_lcb_all
support_balance_pass
accepted_family_count
accepted_horizon_count
max_family_share
max_step_bucket_share
```

### 判断标准

P2 oracle source pass：

```text
For K=64:
  weak_CP_precision_all_rows >= 0.50
  V_ctrl_lcb_all > 0
  long_risk_rate <= 0.10
  support_balance_pass = 1
```

P2 weak oracle pass：

```text
For K=256:
  weak_CP_precision_all_rows >= 0.25
  V_ctrl_lcb_all > 0
  long_risk_rate <= 0.20
```

P2 oracle fail：

```text
No oracle selector at K <= 512 can produce V_ctrl_lcb_all > 0 and long_risk_rate <= 0.20。
```

若 P2 oracle fail，route：

```text
R2-AP0CandidateSourceOracleAbsent
```

这时不能继续 legal selector；必须实现 new source generator。

### 可视化

```text
p2_oracle_source_frontier_by_K.svg
p2_oracle_value_risk_pareto.svg
p2_random_vs_oracle_panel_distribution.svg
p2_oracle_support_balance.svg
```

### Artifacts

```text
p2_oracle_source_selector_upper_bound.csv
oracle_source_selector_trace_v9400.csv
random_panel_bootstrap_trace_v9400.csv
```

---

## P3：Legal source selector capacity test

### 目标

如果 P2 证明 full universe 有可选好 source，则 P3 测 commit-time legal selector 能不能找到它们。

### Legal feature groups

#### LS-A：state-tail features

```text
StateNLL
StateCEp99
StateMarginP10
WrongConfidenceP90
TailHardFraction
```

#### LS-B：gradient/action alignment features

```text
cos_delta_negative_grad
cos_delta_adamw
functional_norm_over_adamw_norm
gradient_conflict_rate
rolewise_conflict_score
```

#### LS-C：linearized effect features

$$
\widehat{\Delta CE}_{lin}(e)=\nabla_\theta CE(\theta_t)^T\Delta\theta_e.
$$

$$
\widehat{\Delta Margin}_{tail}(e)=J_{margin,tail}\Delta\theta_e.
$$

Record：

```text
linearized_CE_delta
linearized_tail_margin_delta
linearized_wrong_conf_delta
linearized_NLL_delta
```

#### LS-D：AdamW residual advantage features

$$
\Delta\theta_{res}(e)
=
\Delta\theta_e
-
Proj_{\Delta\theta_{AdamW}}(\Delta\theta_e).
$$

Record：

```text
residual_norm
residual_descent_score
residual_tail_margin_score
residual_adamw_conflict_score
```

#### LS-E：support / reliability features

```text
family_support_count
family_CP_lcb
horizon_CP_lcb
feature_neighborhood_support
empirical_bayes_value_lcb
empirical_bayes_longrisk_ucb
```

#### LS-F：cost features

```text
source_select_time_ms
feature_compute_time_ms
payload_load_time_ms
payload_apply_estimate_ms
```

### Legal selector families

```text
LSS0-current-source-selection-reference
LSS1-single-feature-topK
LSS2-monotone-linear-source-score
LSS3-value-risk-null-source-gate
LSS4-support-balanced-topK
LSS5-two-stage-cheap-plus-exact-linearization
LSS6-legal-beam-selector-per-active-step
```

Monotone source score：

$$
S_{source}(e)
=
-a_1\widehat{\Delta CE}_{lin}(e)
+a_2\widehat{\Delta Margin}_{tail}(e)
-a_3 Conflict(e)
+a_4 SupportLCB(e)
-a_5 LongRiskUCB(e)
-a_6 Cost(e),
$$

with $a_i\ge 0$.

### Splits

```text
seed-fold cross-fitting；
leave-dataset-out diagnostic；
leave-family-out diagnostic；
leave-horizon-bucket-out diagnostic；
```

No threshold may be tuned per dataset.

### 必须记录

```text
selector_id
feature_set
feature_count
monotonic_constraints
K
calibration_fold
heldout_fold
uses_dataset_name
uses_outcome_at_commit
uses_future_step
accepted_action_count
weak_CP_precision_h20
weak_CP_precision_h80
weak_CP_precision_h240
weak_CP_precision_all
strong_CP_precision_all
horizon_robust_action_rate
long_risk_rate
bad_event_rate
null_rate
V_ctrl_lcb_h20
V_ctrl_lcb_all
feature_auc_weak_CP
feature_auc_longrisk
topK_CP_precision
calibration_to_heldout_drift
leave_dataset_drop
leave_family_drop
feature_compute_time_ms_q90
```

### 判断标准

P3 legal source selector pass：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
feature_compute_time_ms_q90 <= 0.20
For K=64:
  weak_CP_precision_h20 >= 0.25
  weak_CP_precision_all >= 0.25
  V_ctrl_lcb_all > 0
  long_risk_rate <= 0.15
  support_balance_pass = 1
For K=256:
  weak_CP_precision_all >= 0.18
  V_ctrl_lcb_all > 0
```

P3 weak pass：

```text
For K=64:
  weak_CP_precision_h20 >= 0.15
  V_ctrl_lcb_h20 > 0
but all-horizon V_ctrl_lcb_all <= 0 or long_risk > 0.15。
```

P3 fail：

```text
oracle source pass from P2, but no legal selector beats v9.3.9 source panel by at least 2x on h20 weak CP。
```

若 P3 fail，route：

```text
R3-LegalSourceSelectorOpaque
```

Then go to P4 new source generator.

### 可视化

```text
p3_legal_selector_topK_frontier.svg
p3_feature_auc_and_topK_precision.svg
p3_selector_calibration_heldout_drift.svg
p3_source_score_vs_oracle_score.svg
p3_leaveout_drop_heatmap.svg
p3_cost_vs_signal_pareto.svg
```

### Artifacts

```text
p3_legal_source_selector_capacity.csv
legal_source_feature_trace_v9400.csv
source_selector_frontier_trace_v9400.csv
source_selector_leaveout_trace_v9400.csv
```

---

## P4：New value-producing source generator implementation

### 目标

如果 legal selector 不能从 AP0 opaque pool 中稳定找到 good source，则直接生成 by-construction source actions。P4 只测 generator implementation and legality，不写 value success。

### Source generators

#### AP0b：LastLayerLinearizedDescentSource

生成 last-layer / edge-head 层的 functional delta，使一阶 CE 下降：

$$
\Delta\theta_{AP0b}
=
-\eta_s P_{KAN}\nabla_\theta CE_{tail},
$$

其中 $P_{KAN}$ 是合法 KAN edge-basis 参数子空间投影，不引入 non-KAN trainable structure。

Certificate：

```text
linearized_CE_delta < 0
norm <= norm_cap
cos_delta_adamw not too negative
support_count >= threshold
```

#### AP0c：TailMarginPositiveSource

目标不是平均 CE，而是 tail margin：

$$
\Delta\theta_{AP0c}
=
\arg\max_{\Delta\theta\in\mathcal{S}_{KAN}}
J_{margin,tail}\Delta\theta
-
\lambda\|\Delta\theta\|^2.
$$

Certificate：

```text
linearized_tail_margin_delta > 0
wrong_conf_delta <= 0
CE_tail_delta <= small positive tolerance
```

#### AP0d：AdamWResidualOrthogonalSource

生成 AdamW 没覆盖的 residual direction：

$$
\Delta\theta_{AP0d}
=
\Delta\theta_{candidate}
-
Proj_{\Delta\theta_{AdamW}}(\Delta\theta_{candidate}).
$$

Certificate：

```text
residual_descent_score > 0
cos_residual_adamw ≈ 0
conflict_score <= threshold
```

#### AP0e：TinyStepHorizonGuardedSource

小范数 source，优先降低 long-risk：

$$
\Delta\theta_{AP0e}=\gamma\Delta\theta_{descent},\quad \gamma\in\{0.05,0.10,0.20,0.40\}.
$$

Certificate：

```text
predicted h20 descent positive
norm small
predicted h240 risk under cap
```

#### AP0f：PerStepParetoBeamSource

每个 active step 生成 beam：

```text
beam items = {
  CE descent source,
  tail margin source,
  AdamW residual source,
  tiny horizon-guarded source,
  support-balanced source
}
```

然后用 legal score 选 top $m$，不按 dataset。

### 必须记录

```text
primitive_id
generator_id
generated_source_action_count
source_action_id
candidate_id
event_id
payload_hash
certificate_hash
certificate_schema_version
generation_time_ms
generation_cost_q90
payload_apply_error_linf
payload_apply_error_relative
payload_apply_cosine
norm
lin_CE_delta
lin_margin_delta
lin_wrong_conf_delta
cos_delta_adamw
cos_delta_negative_grad
support_lcb
longrisk_ucb
uses_dataset_name
uses_outcome_at_commit
uses_future_step
```

### 判断标准

P4 implementation pass：

```text
generated_source_action_count >= 256 per source generator smoke
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-7
action_apply_cosine_min >= 0.999999
uses_dataset_name = 0
uses_outcome_at_commit = 0
generation_cost_q90 <= 0.50 ms diagnostic budget
```

P4 fail：

```text
payload/certificate missing；
action apply not replayable；
source generator violates legal contract；
```

### 可视化

```text
p4_source_generator_legality_dashboard.svg
p4_payload_norm_distribution_by_generator.svg
p4_linearized_descent_by_generator.svg
p4_action_apply_error_hist.svg
p4_generation_cost_pareto.svg
```

### Artifacts

```text
p4_value_producing_source_generator.csv
source_generator_payload_trace_v9400.csv
source_generator_certificate_trace_v9400.csv
action_apply_replay_trace_v9400.csv
```

---

## P5：Immediate-direction outcome smoke at h20

### 目标

先测 h20 immediate direction。如果 h20 都没有 positive value，长 horizon 和 AP transform 都不应继续。

### Materialization

For each source generator / selector survivor：

```text
horizon = 20
branches = RealSource, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload, CertificatePassNoPayload
source actions = 256 per primitive if available
```

### 必须记录

```text
source_candidate_id
generator_id
selector_id
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
V_ctrl
WeakCP_h20
StrongCP_h20
BadEvent_h20
NullEvent_h20
ImmediatePositive
longrisk_proxy_h20
outcome_runtime_ms
```

### 判断标准

P5 immediate pass：

```text
For at least one selector/generator:
  weak_CP_precision_h20 >= 0.25
  ImmediatePositive precision >= 0.25
  V_ctrl_lcb_h20 > 0
  bad_event_h20 <= 0.05
  null_rate_h20 <= 0.20
  weak_CP_precision_h20 >= 4 * v9390_source_h20_weak_CP_precision
```

Given v9.3.9 source h20 weak CP precision:

$$
v9390\_source\_h20 = 0.02734375,
$$

thus minimum improvement gate:

$$
4 \times 0.02734375 = 0.109375.
$$

P5 weak pass：

```text
weak_CP_precision_h20 >= 0.15
and V_ctrl_lcb_h20 > 0
but bad_event_h20 or null_rate_h20 slightly over gate。
```

P5 fail：

```text
all candidates have V_ctrl_lcb_h20 <= 0 or weak_CP_precision_h20 < 0.10。
```

If P5 fail, route：

```text
R4-ImmediateDirectionFail
```

No h80/h240 scale-up.

### 可视化

```text
p5_h20_immediate_direction_frontier.svg
p5_h20_Vctrl_distribution_by_generator.svg
p5_h20_weakCP_bad_null_bar.svg
p5_h20_real_vs_controls_scatter.svg
p5_h20_certificate_score_vs_outcome.svg
```

### Artifacts

```text
p5_h20_immediate_direction_smoke.csv
h20_branch_outcome_trace_v9400.csv
immediate_direction_frontier_trace_v9400.csv
```

---

## P6：Horizon extension and long-risk audit

### 目标

只对 P5 survivor 跑 h80/h240，判断 immediate-positive source 是否 horizon-fragile。

### Materialization

```text
horizons = 20, 80, 240
branches = RealSource, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload, CertificatePassNoPayload
source actions = all P5 survivors, up to 512 per primitive
```

### 必须记录

```text
source_candidate_id
generator_id
selector_id
horizon
WeakCP
StrongCP
HorizonRobustCP
LongRisk
BadEvent
NullEvent
V_ctrl
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
horizon_value_slope
horizon_risk_slope
h20_to_h240_damage
```

### 判断标准

P6 horizon pass：

```text
For at least one source candidate family:
  weak_CP_precision_all >= 0.20
  V_ctrl_lcb_all > 0
  long_risk_rate_h240 <= 0.15
  horizon_robust_action_rate >= 0.03
  h20_to_h240_damage median >= -0.05
```

P6 weak pass：

```text
h20/h80 pass but h240 long_risk <= 0.30；
needs horizon guard redesign。
```

P6 fail：

```text
h20 pass but h240 long_risk > 0.30 or horizon_robust_action_rate = 0。
```

If P6 fail, route：

```text
R5-HorizonRiskSourceFail
```

### 可视化

```text
p6_horizon_value_curve.svg
p6_horizon_risk_curve.svg
p6_horizon_robust_action_rate.svg
p6_h20_to_h240_damage_hist.svg
p6_longrisk_by_generator.svg
```

### Artifacts

```text
p6_horizon_extension_longrisk_audit.csv
horizon_value_curve_trace_v9400.csv
longrisk_attribution_trace_v9400.csv
```

---

## P7：Source-to-generated AP transform damage on good source panels

### 目标

只有 P6 source survivor 存在时，才重新评价 AP transform。v9.3.9 已经说明 bad source panel 上评价 transform 不够决定性。

### Transforms

```text
T0-identity-source-no-transform
T1-AP5-low-distortion
T2-AP6-value-preserving
T3-AP7-horizon-guarded
T4-AP8-tail-safe
T5-new-AP9-source-aligned-certificate-transform
```

### 必须记录

```text
source_action_id
generated_action_id
source_generator_id
transform_id
horizon
source_WeakCP
source_StrongCP
source_LongRisk
source_V_ctrl
generated_WeakCP
generated_StrongCP
generated_LongRisk
generated_V_ctrl
Damage_V_ctrl = generated_V_ctrl - source_V_ctrl
source_positive_lost
generated_new_positive
payload_geometry_distance
certificate_distance
norm_ratio
cos_source_generated
```

### 判断标准

P7 transform pass：

```text
source_positive_lost_rate <= 0.30
Damage_median >= -0.01
generated_weak_CP_precision >= 0.85 * source_weak_CP_precision
generated_long_risk_rate <= source_long_risk_rate + 0.05
cos_source_generated >= 0.95 for low-distortion variants
```

P7 fail：

```text
source survivor good, but transform loses >50% source positives or raises long-risk >0.10。
```

If P7 fail, route：

```text
R6-APTransformDamagePrimary
```

### 可视化

```text
p7_source_to_generated_damage_matrix.svg
p7_positive_lost_by_transform.svg
p7_source_generated_cosine_vs_damage.svg
p7_transform_value_risk_pareto.svg
```

### Artifacts

```text
p7_source_to_generated_transform_damage.csv
source_generated_damage_trace_v9400.csv
transform_geometry_trace_v9400.csv
```

---

## P8：Effect-valid certificate redesign

### 目标

把 certificate 从 construction-valid 变成 effect-valid。证书必须能预测 WeakCP / StrongCP / LongRisk，而不是只绑定 payload hash。

### Certificate components

```text
C1-linearized_CE_descent_certificate
C2-tail_margin_gain_certificate
C3-adamw_conflict_certificate
C4-horizon_guard_certificate
C5-support_reliability_certificate
C6-cost_certificate
C7-transform_preservation_certificate
```

Certificate score：

$$
CertScore(e)
=
-b_1\widehat{\Delta CE}_{lin}
+b_2\widehat{\Delta Margin}_{tail}
-b_3 Conflict
-b_4 LongRiskUCB
+b_5 SupportLCB
-b_6 Cost
-b_7 TransformDamageUCB.
$$

with $b_i\ge 0$.

### 必须记录

```text
certificate_id
certificate_component_set
cert_pass_count
cert_fail_count
P_weak_CP_given_cert_pass
P_weak_CP_given_cert_fail
P_strong_CP_given_cert_pass
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_weak
Lift_strong
Lift_longrisk
AUC_certificate_weak_CP
AUC_certificate_strong_CP
AUC_certificate_longrisk
ECE_certificate_weak
ECE_certificate_longrisk
monotone_sign_pass
feature_cost_ms_q90
```

### 判断标准

P8 certificate pass：

```text
AUC_certificate_weak_CP >= 0.70
P_weak_CP_given_cert_pass >= 0.50
P_longrisk_given_cert_pass <= 0.50 * P_longrisk_given_cert_fail
ECE_certificate_weak <= 0.08
monotone_sign_pass = 1
feature_cost_ms_q90 <= 0.20
```

P8 weak pass：

```text
AUC_certificate_weak_CP >= 0.65
but longrisk reduction insufficient。
```

P8 fail：

```text
AUC_certificate_weak_CP < 0.65 or Lift_longrisk near 1.0。
```

If P8 fail, route：

```text
R7-CertificateEffectInsufficient
```

### 可视化

```text
p8_certificate_lift_bar.svg
p8_certificate_roc_pr_curves.svg
p8_certificate_calibration_curve.svg
p8_cert_pass_fail_value_risk_distribution.svg
p8_certificate_component_ablation.svg
```

### Artifacts

```text
p8_effect_valid_certificate_redesign.csv
certificate_component_trace_v9400.csv
certificate_ablation_trace_v9400.csv
```

---

## P9：Minimal source-certificate controller

### 目标

若 P5/P6/P8 产生 source + certificate survivor，则构造最小 official candidate controller。P9 不允许 opaque large feature pile。

### Controller form

$$
Accept(e)=1
\iff
LCB(V_{src}(e))>0
\land
UCB(Bad(e))\le \tau_b
\land
UCB(Null(e))\le \tau_n
\land
UCB(LongRisk(e))\le \tau_l
\land
LCB(Support(e))\ge \tau_s
\land
C(e)\le C_{max}.
$$

Score：

$$
S(e)
=
LCB(V_{src}(e))
-\lambda_b UCB(Bad(e))
-\lambda_n UCB(Null(e))
-\lambda_l UCB(LongRisk(e))
+\lambda_s LCB(Support(e))
-\lambda_c C(e).
$$

### Minimality constraints

```text
feature groups <= 5
monotone coefficients only
single-feature baselines reported
ablation required
dataset_name_used = 0
thresholds frozen on calibration split
```

### Splits

```text
seed-fold split
leave-dataset-out diagnostic
leave-family-out diagnostic
leave-horizon-bucket-out diagnostic
```

### 必须记录

```text
controller_id
source_selector_id
source_generator_id
certificate_id
feature_groups
feature_count
thresholds
calibration_fold
heldout_fold
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
weak_CP_precision_cal
weak_CP_precision_heldout
strong_CP_precision_heldout
bad_event_heldout
null_event_heldout
long_risk_heldout
V_ctrl_lcb_heldout
precision_lcb
bad_event_ucb
longrisk_ucb
support_balance_pass
accepted_family_count
accepted_horizon_count
max_family_share
max_horizon_share
calibration_to_heldout_drift
uses_dataset_name
uses_outcome_at_commit
uses_future_step
```

### 判断标准

P9 source-controller pass：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
coverage_heldout in [0.03, 0.15]
weak_CP_precision_heldout >= 0.75
bad_event_heldout <= 0.05
null_event_heldout <= 0.15
long_risk_heldout <= 0.10
V_ctrl_lcb_heldout > 0
precision_lcb >= 0.70
bad_event_ucb <= 0.07
support_balance_pass = 1
```

P9 weak pass：

```text
coverage_heldout >= 0.03
weak_CP_precision_heldout >= 0.60
V_ctrl_lcb_heldout > 0
but longrisk or LCB/UCB not official。
```

P9 fail：

```text
coverage collapse；
heldout precision < 0.60；
long-risk remains high；
```

### 可视化

```text
p9_controller_frontier_precision_coverage_longrisk.svg
p9_calibration_heldout_drift.svg
p9_controller_ablation.svg
p9_support_balance_sunburst.svg
p9_selected_source_distribution.svg
```

### Artifacts

```text
p9_minimal_source_certificate_controller.csv
controller_frontier_trace_v9400.csv
controller_ablation_trace_v9400.csv
```

---

## P10：Selected source/controller online runtime

### 目标

只有 P9 有 selected controller 后才测 official runtime。P10 必须包含 source selection/generation、certificate computation、payload apply、base step。

### Runtime modes

```text
RT0-reference-no-source-controller
RT1-selected-source-selector-online
RT2-selected-source-generator-online
RT3-selected-certificate-controller-online
RT4-fused-source-cert-payload-online
RT5-offline-materializer-only-diagnostic
```

Official 只看 RT1-RT4 online path，不看 RT5。

### 必须记录

```text
runtime_candidate_id
controller_id
source_generator_id
certificate_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
source_generation_time_ms_q90
source_selection_time_ms_q90
certificate_compute_time_ms_q90
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
allocation_count_per_active_step
payload_apply_error_linf_max
accept_disagreement_count
audit_outside_timed_path
```

### 判断标准

P10 runtime pass：

```text
runtime_mode = online_selected_controller
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
payload_apply_error_linf_max <= 1e-7
audit_outside_timed_path = 1
```

P10 fail：

```text
no selected controller；
step_ratio_q90 > 1.50；
payload apply dominates and cannot be fused；
```

### 可视化

```text
p10_online_runtime_waterfall.svg
p10_step_ratio_distribution.svg
p10_runtime_component_pareto.svg
p10_payload_apply_error_hist.svg
p10_active_step_launch_sync_hist.svg
```

### Artifacts

```text
p10_selected_source_controller_online_runtime.csv
runtime_component_trace_v9400.csv
online_runtime_trace_v9400.csv
```

---

## P11：System integration gate

### 目标

组合 P9 decision 和 P10 runtime，判定 v9.4.0 是否产生 system-legal source/controller candidate。

### 必须记录

```text
system_candidate_id
source_selector_id
source_generator_id
certificate_id
controller_id
runtime_candidate_id
official_eligible
system_legal_controller_pass
materialized_system_path
selected_controller_used
selected_payload_apply_used
source_generation_used
certificate_compute_used
secondary_control_outcome_ready
weak_CP_precision_heldout
coverage_heldout
bad_event_heldout
null_event_heldout
long_risk_heldout
V_ctrl_lcb_heldout
step_ratio_q90
memory_ratio
uses_dataset_name
uses_outcome_at_commit
uses_future_step
diagnostic_promoted_to_official
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

P11 system pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
selected_controller_used = 1
selected_payload_apply_used = 1
secondary_control_outcome_ready = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Decision gates：

$$
Coverage\in[0.03,0.15],
$$

$$
WeakCPPrecision\ge0.75,
$$

$$
BadEvent\le0.05,
$$

$$
NullEvent\le0.15,
$$

$$
LongRisk\le0.10,
$$

$$
LCB(V_{ctrl})>0.
$$

Runtime gates：

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

### 可视化

```text
p11_system_gate_dashboard.svg
p11_quality_cost_pareto.svg
p11_system_failure_reason_matrix.svg
```

### Artifacts

```text
p11_system_integration_gate_v9400.csv
system_controller_trace_v9400.csv
route_decision.json
aggregate_decision.json
```

---

## P12：Leave-out and paired replay boundary

### 目标

只有 P11 pass 后打开 official LDO/LSO 和 paired replay。若 P11 未过，只允许 diagnostic scout，并必须隔离。

### LDO / LSO

```text
leave-dataset-out：train/calibrate on two datasets, evaluate third；
leave-family-out：hold out high-volume families；
leave-horizon-bucket-out：hold out horizon/step buckets；
```

### Paired replay branches

```text
RealFunctionalSource
AdamWParallel
bestLR
NoOp
Random
ShuffledSourcePayload
ShuffledCertificate
ShuffledSourceSelectorScore
ShuffledFamilySupport
CertificatePassNoPayload
```

### 必须记录

```text
split_type
heldout_entity
controller_id
source_generator_id
certificate_id
branch
horizon
weak_CP_precision
strong_CP_precision
bad_event
null_event
long_risk
V_ctrl
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
beats_adamwparallel
beats_bestlr
task_safe
step_ratio_q90
memory_ratio
diagnostic_used_for_controller
```

### 判断标准

Official paired replay pass：

```text
P11 system pass = 1
RealFunctionalSource beats AdamWParallel in >= 60% macro slices
RealFunctionalSource beats bestLR in >= 60% macro slices
Task safety: Acc_real >= Acc_adamw - 0.005 in every official slice
Shuffled controls fail to match RealFunctionalSource
```

Diagnostic isolation：

```text
diagnostic_used_for_controller = 0
```

### 可视化

```text
p12_leaveout_matrix.svg
p12_paired_replay_macro_beat_rate.svg
p12_shuffle_control_matrix.svg
p12_task_safety_by_slice.svg
```

### Artifacts

```text
p12_leaveout_and_paired_replay_boundary.csv
paired_replay_trace_v9400.csv
leaveout_trace_v9400.csv
```

---

## P13：Short/full/sample-efficiency/continual robustness boundary

### 目标

只有 P12 official paired replay pass 后打开。v9.4.0 可以预先准备 runner，但不得提前声明。

### 必须记录

```text
dataset
seed
controller_id
source_generator_id
steps
branch
final_acc
best_val_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
functional_event_count
coverage
bad_event
null_event
long_risk
step_ratio_q90
memory_ratio
continual_retained_accuracy
forgetting
backward_transfer
forward_transfer
```

### 判断标准

Short/full functional pass：

```text
Task safety: Acc_functional >= Acc_adamw - 0.005
At least one robust advantage:
  ValLossAUC_step better than AdamWParallel；
  ValLossAUC_time better than AdamWParallel；
  ECE lower；
  NLL lower；
  CEp99 lower；
  margin_p10 higher；
  curvature <= 0.90 * AdamW；
  time_to_target lower；
```

Continual pass：

```text
retained_accuracy >= AdamW retained_accuracy - 0.005
forgetting <= AdamW forgetting
backward_transfer >= AdamW backward_transfer
```

### 可视化

```text
p13_val_loss_auc_step_time.svg
p13_time_to_target.svg
p13_calibration_robustness.svg
p13_continual_forgetting_transfer.svg
p13_fullrun_quality_cost_pareto.svg
```

### Artifacts

```text
p13_short_full_sampleeff_continual_robustness.csv
short_full_trace_v9400.csv
continual_trace_v9400.csv
```

---

# 7. 并行执行计划

为了加快实验，v9.4.0 采用并行批次，但所有 official gates 必须按依赖顺序判断。

## Batch A：无需新训练的数据诊断

可立即并行运行：

```text
P0 full-vs-panel source diagnosis；
P1 source selection provenance audit；
P2 oracle source selector upper-bound；
```

预计产出：

```text
source panel 是 bad sample / bad selector / source absence 的初步分类。
```

## Batch B：legal source selector capacity

依赖 P0/P2 的 unified table，但不依赖新 generator。

```text
P3 legal source feature computation；
P3 topK source selector frontier；
P3 leave-out diagnostic；
```

若 P3 pass，可直接进入 P5/P6，不必等 P4 全部 primitive。

## Batch C：new source generator implementation

与 Batch B 并行。

```text
AP0b LastLayerLinearizedDescentSource；
AP0c TailMarginPositiveSource；
AP0d AdamWResidualOrthogonalSource；
AP0e TinyStepHorizonGuardedSource；
AP0f PerStepParetoBeamSource；
```

先跑 implementation + h20 smoke，不跑 full horizon。

## Batch D：h20 outcome materializer

一旦 Batch B 或 C 有 candidates，立即跑：

```text
P5 h20 immediate-direction outcome smoke；
```

P5 fail 的 candidate 不进入 h80/h240。

## Batch E：runtime microbench preparation

可并行准备，但不得 official：

```text
source generation cost microbench；
certificate compute cost microbench；
payload apply cost microbench；
fusion feasibility microbench；
```

只有 P9 selected controller 后，P10 才 official。

---

# 8. Route decision taxonomy

v9.4.0 的 route 必须精确，不再写模糊的 “value fail”。

```text
R0-BoundaryOrJoinFail
  P0/P1 无法复现或 source mapping 不可靠。

R1-SourcePanelSamplingOrSelectionBias
  full AP0 universe 有好 actions，但 v9.3.9 source panel 不代表 full universe。

R2-AP0CandidateSourceOracleAbsent
  即使用 oracle 也无法从 AP0 universe 选出 good source panel。

R3-LegalSourceSelectorOpaque
  oracle source selector pass，但 legal commit-time selector 找不到 good source。

R4-ImmediateDirectionFail
  new selector/generator h20 immediate direction 不过。

R5-HorizonRiskSourceFail
  h20 pass，但 h80/h240 long-risk / horizon robustness 不过。

R6-APTransformDamagePrimary
  source survivor good，但 AP transform 大量破坏 positive source。

R7-CertificateEffectInsufficient
  source/generated survivor 存在，但 certificate 不具备 effect-valid lift / risk control。

R8-ControllerDecisionFail
  source + certificate 有 signal，但 cross-fitted heldout controller 不过。

R9-SelectedRuntimeFail
  controller pass，但 selected online runtime 不过。

R10-SystemLegalSourceControllerPass
  source/controller/runtime 全部过 P11。

R11-PairedReplayFail
  system pass，但 official paired replay 打不过 controls。

R12-ShortFullRobustnessFail
  paired replay pass，但 short/full/continual 不过。
```

---

# 9. 需要落盘的 artifacts

```text
run_manifest.json
route_decision.json
aggregate_decision.json
contract_audit_v9400.csv
provenance_audit_v9400.csv
failure_table_v9400.csv
artifact_hashes.csv

p0_full_vs_source_panel_diagnosis.csv
source_panel_join_trace_v9400.csv
source_panel_bias_trace_v9400.csv

p1_source_selection_provenance_audit.csv
source_selection_trace_v9400.csv
source_selection_unknown_reason_table_v9400.csv

p2_oracle_source_selector_upper_bound.csv
oracle_source_selector_trace_v9400.csv
random_panel_bootstrap_trace_v9400.csv

p3_legal_source_selector_capacity.csv
legal_source_feature_trace_v9400.csv
source_selector_frontier_trace_v9400.csv
source_selector_leaveout_trace_v9400.csv

p4_value_producing_source_generator.csv
source_generator_payload_trace_v9400.csv
source_generator_certificate_trace_v9400.csv
action_apply_replay_trace_v9400.csv

p5_h20_immediate_direction_smoke.csv
h20_branch_outcome_trace_v9400.csv
immediate_direction_frontier_trace_v9400.csv

p6_horizon_extension_longrisk_audit.csv
horizon_value_curve_trace_v9400.csv
longrisk_attribution_trace_v9400.csv

p7_source_to_generated_transform_damage.csv
source_generated_damage_trace_v9400.csv
transform_geometry_trace_v9400.csv

p8_effect_valid_certificate_redesign.csv
certificate_component_trace_v9400.csv
certificate_ablation_trace_v9400.csv

p9_minimal_source_certificate_controller.csv
controller_frontier_trace_v9400.csv
controller_ablation_trace_v9400.csv

p10_selected_source_controller_online_runtime.csv
runtime_component_trace_v9400.csv
online_runtime_trace_v9400.csv

p11_system_integration_gate_v9400.csv
system_controller_trace_v9400.csv

p12_leaveout_and_paired_replay_boundary.csv
paired_replay_trace_v9400.csv
leaveout_trace_v9400.csv

p13_short_full_sampleeff_continual_robustness.csv
short_full_trace_v9400.csv
continual_trace_v9400.csv

figures/
```

---

# 10. 必须可视化清单

```text
p0_full_vs_source_panel_cp_rates.svg
p0_source_panel_representativeness_heatmap.svg
p0_full_vs_panel_Vctrl_distribution.svg

p1_source_selection_flow_sankey.svg
p1_selection_score_vs_CP_label.svg

p2_oracle_source_frontier_by_K.svg
p2_oracle_value_risk_pareto.svg
p2_random_vs_oracle_panel_distribution.svg

p3_legal_selector_topK_frontier.svg
p3_feature_auc_and_topK_precision.svg
p3_cost_vs_signal_pareto.svg

p4_source_generator_legality_dashboard.svg
p4_linearized_descent_by_generator.svg
p4_generation_cost_pareto.svg

p5_h20_immediate_direction_frontier.svg
p5_h20_Vctrl_distribution_by_generator.svg
p5_h20_real_vs_controls_scatter.svg

p6_horizon_value_curve.svg
p6_horizon_risk_curve.svg
p6_longrisk_by_generator.svg

p7_source_to_generated_damage_matrix.svg
p7_positive_lost_by_transform.svg

p8_certificate_lift_bar.svg
p8_certificate_calibration_curve.svg
p8_certificate_component_ablation.svg

p9_controller_frontier_precision_coverage_longrisk.svg
p9_controller_ablation.svg

p10_online_runtime_waterfall.svg
p10_step_ratio_distribution.svg

p11_system_gate_dashboard.svg
p11_quality_cost_pareto.svg

p12_paired_replay_macro_beat_rate.svg
p12_shuffle_control_matrix.svg

p13_val_loss_auc_step_time.svg
p13_continual_forgetting_transfer.svg
```

---

# 11. No-fake / legality audit

每个阶段必须记录：

```text
fake_data_used
proxy_row_used
cpu_offload_used
uses_loss_backward
uses_teacher
uses_loss_modification
uses_dataset_name_for_selector
uses_dataset_name_for_controller
uses_validation_or_test
uses_future_outcome_for_features
uses_outcome_at_commit
source_measured_gap_used
formula_proxy_used
diagnostic_promoted_to_official
```

Pass：

```text
all above = 0
manual_forward = 1
manual_backward = 1
manual_adamw_update = 1
train_stream_probe = 1
```

---

# 12. v9.4.0 的最终成功/失败解释

## v9.4.0 成功但不等于终极成功

如果 v9.4.0 过 P11，只能说明：

```text
存在一个 legal source selector/generator + certificate + controller + runtime path；
它能在 local branch-horizon outcome 上形成 control-positive、horizon-aware、system-legal frontier。
```

还不能说明 full functional success。必须继续：

```text
LDO/LSO；
official paired replay；
short/full run；
strong baseline；
sample efficiency；
continual / anti-forgetting；
external reproducibility。
```

## v9.4.0 失败时的解释

如果 P2 oracle fail：

```text
AP0 candidate source 本身没有足够 source frontier；重建 candidate source，不再从 AP0 full universe 选择。
```

如果 P2 oracle pass 但 P3 legal fail：

```text
AP0 good actions 存在但 legally opaque；必须生成 by-construction source actions。
```

如果 P5 fail：

```text
新的 source generator 没有 immediate positive direction；不要跑 long horizon 或 AP transform。
```

如果 P6 fail：

```text
immediate direction 有，但 horizon fragile；需要 horizon-aware source generation，而不是 posthoc threshold。
```

如果 P7 fail：

```text
source selection/generation 找到了好 source，但 AP transform 破坏它；AP transform redesign 成为主线。
```

如果 P8 fail：

```text
source/generation 有信号，但 certificate 不足；certificate redesign 成为主线。
```

如果 P9 fail：

```text
source/certificate 有局部信号，但 heldout controller frontier 不稳定；不能 official。
```

如果 P10 fail：

```text
decision 有，但 selected online runtime 不过；进入 runtime-native implementation。
```

---

# 13. 最终一句话

v9.4.0 的核心不是“再修 AP3/AP7”，而是：

$$
\boxed{
\text{先让 source actions 本身变成 value-producing objects，再谈 AP transform、certificate、controller 和 runtime。}
}
$$

如果 source action 已经 value-poor、immediate direction weak、long-risk high，那么所有后续 transform 都只是在坏输入上做精致加工。v9.4.0 必须把实验推进到 source selection / source generation 这一层，否则会从 `AP feature patching` 继续变成 `source certificate patching`，仍然不是第一性原理上的突破。
