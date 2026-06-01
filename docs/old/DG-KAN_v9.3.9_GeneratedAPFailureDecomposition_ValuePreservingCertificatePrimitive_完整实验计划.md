# DG-KAN v9.3.9 Generated-AP Failure Decomposition / Value-Preserving Certificate Primitive Redesign 完整实验计划

> 本计划基于 v9.3.8 `Real Certificate-Producing Action Primitive Materialization / Horizon-Robust System Closure` 的真实执行结果制定。  
> v9.3.9 不再做 AP0 threshold/probe patch，也不把 AP1-AP4 的 certificate threshold 继续小修小补。  
> 本轮核心目标是回答一个更根本的问题：
>
> $$
> \boxed{
> \text{为什么 AP1-AP4 真实生成的 certificate-producing action 没有继承 AP0 oracle-good frontier？}
> }
> $$
>
> 如果原因是当前 AP1-AP4 generator 破坏了 AP0 的有效 action geometry，则 v9.3.9 要转向 **value-preserving / low-distortion / horizon-aware certificate primitive**。  
> 如果原因是 source AP0 子集本身就没有 value，则 v9.3.9 要重建 source-action selection 和 primitive sampling。  
> 如果原因是 certificate 本身不是 effect sufficient statistic，则 v9.3.9 要重写 certificate 的语义，使它从“construction certificate”变成“effect certificate”。

---

# 0. 执行摘要

v9.3.8 的路线是：

```text
route = R5-APGeneratedActionsValueFail
base_candidate = LQ-t2-h256
success_v9380_strict_purekan_functional = False
success_v9380_full_functional = False
success_v9380_external_ready = False
```

v9.3.8 有真实进展：

```text
AP1-AP4 generated action count: 0 -> 256
primitive_materialized_count = 4
每个 primitive 生成 64 个 action
durable_payload_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
action_apply_error_relative_max = 0.0
certificate_schema_contract_pass = 1
ap_smoke_outcome_pass = 1
branch_horizon_row_count_actual = 6912
```

这说明 v9.3.7 的主 blocker “AP1-AP4 没有真实 generator” 已经被推进。v9.3.8 不是 audit-only，也不是 schema-only；它第一次让 AP1/AP2/AP3/AP4 真实生成 payload、真实写出 certificate tensor、真实 action replay、真实跑 branch-horizon outcomes。

但是 v9.3.8 没有 functional success。certificate-pass generated AP rows 的核心结果是：

```text
certificate_pass_outcome_count = 249
weak_CP_precision_certificate_pass = 0.10040160642570281
strong_CP_precision_certificate_pass = 0.028112449799196786
horizon_robust_CP_precision_certificate_pass = 0.0
long_risk_rate_certificate_pass = 0.3092369477911647
V_ctrl_lcb_certificate_pass = -1.409779974372909
certificate_calibration_pass = 0
certificate_controller_pass = 0
selected_payload_runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
```

换成计数更直观：

$$
N_{cert\_outcome}=249.
$$

$$
N_{weak\_CP}\approx249\times0.1004016064=25.
$$

$$
N_{strong\_CP}\approx249\times0.0281124498=7.
$$

$$
N_{long\_risk}\approx249\times0.3092369478=77.
$$

如果按 weak precision $0.75$ 作为可部署安全区域的粗下限，在 $249$ 个 cert-pass outcome rows 上至少需要：

$$
N_{weak\_CP,target}=\lceil0.75\times249\rceil=187.
$$

当前只有约 $25$，缺口约 $162$。如果 long-risk rate 要压到 $0.05$，最多允许：

$$
N_{long\_risk,target}=\lfloor0.05\times249\rfloor=12.
$$

当前约 $77$，多出约 $65$。这不是 certificate threshold 微调能补上的差距。

per-primitive 结果也说明 AP3 只是“相对最好”，不是可用：

| primitive | cert-pass outcomes | weak CP precision | strong CP precision | horizon robust precision | long-risk rate | V_ctrl LCB | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP1 | 84 | 0.059524 | 0.011905 | 0.0 | 0.333333 | -1.299056 | 0 |
| AP2 | 57 | 0.087719 | 0.035088 | 0.0 | 0.333333 | -1.762332 | 0 |
| AP3 | 57 | 0.192982 | 0.052632 | 0.0 | 0.228070 | -1.580291 | 0 |
| AP4 | 51 | 0.078431 | 0.019608 | 0.0 | 0.333333 | -1.744541 | 0 |

AP3 的 weak CP 最高，但：

$$
0.192982 \ll 0.75,
$$

$$
V_{ctrl,LCB}=-1.580291<0,
$$

$$
HorizonRobustPrecision=0.
$$

因此 v9.3.9 的核心不是“选 AP3 再调一调”，而是弄清楚：

```text
1. source AP0 action 本身是否有 value？
2. AP1-AP4 generator 是否把 source action 变坏？
3. certificate pass 是否完全没有 value lift？
4. failure 是否集中在 long horizon？
5. action magnitude / projection / tail operation / residual operation 哪一步造成 damage？
6. 是否存在 value-preserving 低扰动 generator？
7. 如果低扰动 generator 也失败，是否应放弃 AP1-AP4 方向，转向 AP0 selection + certificate-only controller 或新的 candidate source？
```

v9.3.9 的最低有效推进目标不是 system pass，而是把 `APGeneratedActionsValueFail` 拆成可操作的 root cause：

```text
R1-source_AP0_bad
R2-generator_transformation_damage
R3-certificate_not_effect_sufficient
R4-horizon_long_risk_primary
R5-sample_panel_bias
R6-low_distortion_generator_pass
R7-generated_AP_frontier_absent
R8-certificate_controller_ready
R9-system_ready_for_LDO_LSO
```

---

# 1. 独立判断：v9.3.8 到底说明了什么

## 1.1 有进展，而且是关键工程进展

v9.3.8 不是失败倒退。它首次把 AP1-AP4 从 “schema / plan / placeholder” 推进为真实生成：

```text
generated_action_count_total = 256
primitive_materialized_count = 4
certificate rows = 256
branch-horizon rows = 6912
```

这解决了 v9.3.7 的根本 blocker：

```text
v9.3.7 generated_action_count_total = 0
v9.3.8 generated_action_count_total = 256
```

这意味着现在可以讨论 AP primitive 本身的科学问题了。此前只能说“没有实现”；现在可以说“实现后 outcome 不好”。这是科学进展。

## 1.2 但它不是 controller 进展

v9.3.8 没有选中 certificate controller：

```text
certificate_calibration_pass = 0
certificate_controller_pass = 0
selected_payload_runtime_pass = 0
system_legal_controller_pass = 0
```

P7 selected runtime 正确 not-run。这个 gate 是对的，因为如果 certificate rows 没有 value-positive / horizon-safe frontier，那么 runtime 再快也没有意义。

## 1.3 也不是 functional success

functional success 至少需要：

```text
1. generated action 有 control-positive frontier；
2. certificate 能在 commit 前识别该 frontier；
3. selected controller 过 decision gate；
4. selected payload runtime 过 system gate；
5. LDO / LSO 过；
6. paired replay 打过 AdamWParallel / bestLR / shuffled controls；
7. short/full/sample-efficiency/continual/robustness 有真实优势。
```

v9.3.8 目前只完成了第 0 步：

```text
真实 AP action generator / payload / certificate / smoke outcome 链路可运行。
```

第 1 步就失败了：

$$
V_{ctrl,LCB}<0,
$$

$$
HorizonRobustPrecision=0.
$$

## 1.4 最本质的问题：certificate 还不是 effect certificate

当前 certificate contract 证明的是：

```text
payload 已生成；
payload hash 可绑定；
certificate tensor 已落盘；
certificate fields complete；
legality violation = 0；
action apply replay error = 0。
```

但它没有证明：

```text
RealAP beats AdamWParallel；
RealAP beats bestLR；
RealAP 不造成 long horizon risk；
certificate pass rows 比 certificate fail rows 有更高 CP density；
certificate score 与 V_ctrl 单调一致；
certificate 是 value / risk / horizon 的 sufficient statistic。
```

所以 v9.3.8 的核心发现是：

$$
\boxed{
\text{当前 AP1-AP4 的 certificate 是 construction-valid，但不是 effect-valid。}
}
$$

这比“阈值不合适”更深。阈值修的是 decision boundary；现在的问题是 certificate/pass 本身没有把 action-effect 说清楚。

## 1.5 不能误读成“certificate primitive 方向已死”

v9.3.8 只测试了：

```text
source AP0 actions = 64
primitive = 4
generated AP actions = 256
smoke branch-horizon rows = 6912
```

这足以拒绝当前 AP1/AP2/AP3/AP4 实现的 immediate promotion，但不足以证明所有 certificate-producing action primitive 不可能成功。

更严谨的判断是：

$$
\boxed{
\text{当前 AP1-AP4 generator 实现失败；certificate-producing primitive 思路尚未被全局证伪。}
}
$$

v9.3.9 必须做 root-cause decomposition，而不是直接再换一组 primitive 名字。

---

# 2. v9.3.9 总体目标

v9.3.9 的总体目标是：

$$
\boxed{
\text{把 generated AP failure 分解为 source、transformation、certificate、horizon、runtime 五类原因，并实现 value-preserving certificate primitive v2。}
}
$$

更具体地说，本轮必须回答：

```text
Q1. 用来生成 AP1-AP4 的 64 个 source AP0 actions 本身在同一 smoke protocol 下是否有 positive control value？
Q2. AP1-AP4 是继承了 source AP0 的坏，还是把 source AP0 的好 action 变坏了？
Q3. AP1-AP4 的 certificate pass 是否相对于 certificate fail 有任何 CP lift？
Q4. failure 是 short-horizon 就坏，还是 20-step 有用但 80/240-step long-risk？
Q5. action norm、direction cosine、AdamW conflict、tail projection、low-rank projection、horizon memory 哪个生成算子造成最大 damage？
Q6. 是否存在低扰动 / value-preserving / horizon-aware 的 AP5-AP8 generator，让 generated AP 至少恢复到 AP0 source 的 control-positive density？
Q7. 如果 AP5-AP8 仍失败，是否应停止 generated-AP primitive，转向 AP0 selected action + certificate-only controller 或重建 candidate source？
```

v9.3.9 不追求直接 external-ready。它的强目标是形成一个可进入 v9.4 的 primitive：

```text
1. generated AP oracle weak frontier exists；
2. certificate-pass rows have positive V_ctrl LCB；
3. long-risk controlled；
4. horizon robust density nonzero and above minimum；
5. certificate has measurable lift over fail rows；
6. source-to-generated transformation damage is bounded；
7. selected primitive can be runtime-measured without diagnostic promotion。
```

---

# 3. v9.3.9 明确不做什么

本轮不做：

```text
1. 不回到 AP0 threshold/probe patch；
2. 不继续只调 AP1-AP4 certificate thresholds；
3. 不把 certificate legality pass 写成 value pass；
4. 不把 64-action smoke pass 或 fail 直接写成 full universe final conclusion；
5. 不按 MNIST / Fashion-MNIST / KMNIST 分别调 primitive 或 threshold；
6. 不用 dataset_name 作为 controller feature 或 branch；
7. 不用 validation/test at commit time；
8. 不用 future outcome at commit time；
9. 不用 teacher / distillation / auxiliary loss / CE loss modification；
10. 不把 generated AP smoke outcome 结果用于修改同一轮 threshold 后再 official；
11. 不在 certificate controller 未过时跑 official paired replay；
12. 不把 selected runtime estimate 写成 measured runtime。
```

允许做：

```text
1. dataset/family/horizon diagnostics；
2. source AP0 vs generated AP paired diagnostic；
3. oracle-stratified diagnostic panel，用于理解原因，不进入 controller；
4. generic scale / norm / projection / horizon ablation，不按 dataset 调参；
5. calibration-only threshold freeze；
6. cross-fitted certificate calibration；
7. parallel smoke panels；
8. offline materializer batching；
9. selected-controller runtime 只有在 controller pass 后再 measured。
```

---

# 4. 核心对象与公式

## 4.1 source action 与 generated action

对每个 source AP0 action $a_i$，AP primitive $k$ 生成：

$$
G_k(s_t,b_t,g_t,\Delta\theta_{AP0,i})
\rightarrow
(\Delta\theta_{APk,i}, Cert_{APk,i}, Hash_{APk,i}).
$$

其中：

```text
s_t: model state
b_t: current batch state
g_t: manual gradient / AdamW-equivalent gradient
Δθ_AP0,i: source AP0 payload
Δθ_APk,i: generated APk payload
Cert_APk,i: commit-time certificate tensor
Hash_APk,i: payload/certificate hash binding
```

## 4.2 source-preservation metrics

方向保持：

$$
CosSrc_k(i)=
\frac{\langle \Delta\theta_{APk,i},\Delta\theta_{AP0,i}\rangle}
{\|\Delta\theta_{APk,i}\|\|\Delta\theta_{AP0,i}\|+\epsilon}.
$$

范数比例：

$$
NormRatio_k(i)=
\frac{\|\Delta\theta_{APk,i}\|}{\|\Delta\theta_{AP0,i}\|+\epsilon}.
$$

AdamW conflict：

$$
ConflictAdamW_k(i)=
-\cos(\Delta\theta_{APk,i},\Delta\theta_{AdamW,t}).
$$

source damage：

$$
Damage_k(i,h)=V_{ctrl}(APk,i,h)-V_{ctrl}(AP0,i,h).
$$

如果 $Damage_k<0$ 大量出现，说明 generator 在破坏 source action。

## 4.3 control-positive value

对 branch $b$ 和 horizon $h$，定义：

$$
V_{ctrl}(e,h)=
V_{RealAP}(e,h)-
\max_{c\in Controls}V_c(e,h).
$$

Controls 包括：

```text
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledAPPayload
CertificatePassNoPayload
```

weak CP：

$$
WeakCP(e,h)=1
\iff
V_{ctrl}(e,h)>0
\land Bad(e,h)=0
\land Null(e,h)=0.
$$

strong CP：

$$
StrongCP(e,h)=1
\iff
WeakCP(e,h)=1
\land V_{ctrl}(e,h)\ge\delta_{strong}
\land SafetyMargin(e,h)\ge m_{strong}.
$$

horizon robust CP：

$$
HorizonRobustCP(e)=1
\iff
\prod_{h\in\{20,80,240\}}WeakCP(e,h)=1.
$$

long-risk：

$$
LongRisk(e)=1
\iff
Bad(e,240)=1
\lor
V_{ctrl}(e,240)<-\delta_{risk}.
$$

## 4.4 certificate lift

Certificate pass 的 lift 不能只看 pass count，要看它相对 fail rows 的 CP lift：

$$
Lift_{weak}=
\frac{P(WeakCP=1\mid CertPass=1)}
{P(WeakCP=1\mid CertPass=0)+\epsilon}.
$$

$$
Lift_{longrisk}=
\frac{P(LongRisk=1\mid CertPass=1)}
{P(LongRisk=1\mid CertPass=0)+\epsilon}.
$$

好的 certificate 应满足：

$$
Lift_{weak}>1,
$$

$$
Lift_{longrisk}<1.
$$

当前 v9.3.8 只显示 cert-pass rows 的 CP 很低；v9.3.9 必须补上 pass-vs-fail lift，否则不知道 certificate 是否完全无效还是 threshold 太松。

---

# 5. 核心假设

## H0：v9.3.8 是真实 AP generated action failure，不是 materialization bug

H0 认为：v9.3.8 的 payload/certificate/outcome 链路是可信的，failure 来自 generated action 本身，而不是 artifact 或 label pipeline。

H0 成立标准：

```text
generated_action_count_total = 256
payload/certificate hash missing = 0
action apply replay error max = 0
branch_horizon_completion_rate = 1.0
outcome source = same_run_generated_AP_smoke
no fake/proxy/offload = 1
```

H0 失败标准：

```text
payload hash mismatch;
certificate hash mismatch;
action apply replay error > tolerance;
branch identity mismatch;
horizon hash mismatch;
control branch missing;
label exclusivity violation;
NaN/Inf metric violation。
```

如果 H0 失败，不能做 primitive redesign，必须先修 materializer。

## H1：source AP0 子集可能不是 value-positive source

H1 认为：v9.3.8 只用了 64 个 source AP0 actions，如果这些 source action 本身在 same smoke protocol 下 CP density 很低，则 AP1-AP4 failure 不能完全归因于 generator。

H1 成立标准：

```text
source_AP0_same_panel_weak_CP_precision < 0.20
source_AP0_V_ctrl_lcb <= 0
source_AP0_horizon_robust_precision = 0
```

H1 失败标准：

```text
source_AP0_same_panel_weak_CP_precision >= 0.35
source_AP0_V_ctrl_lcb > 0
source_AP0_long_risk_rate <= 0.15
但 AP1-AP4 generated rows 大幅更差。
```

若 H1 失败，说明 generator transformation 是主要 damage source。

## H2：AP1-AP4 generator 破坏了 AP0 的 hidden value geometry

H2 认为：AP0 oracle-good frontier 存在，但 AP1-AP4 对 source payload 的投影、残差、tail 操作、horizon memory 操作改变了有效方向，使 action 从 positive control 变成 value-negative。

H2 成立标准：

```text
source_AP0 positive rows 在 generated AP 中 V_ctrl 显著下降；
median Damage_k < 0；
P(Damage_k < -δ) >= 0.50；
CosSrc 或 NormRatio 与 Damage 显著相关；
某些 generator operation group 的 Damage 显著高于其他 group。
```

H2 失败标准：

```text
AP1-AP4 和 source AP0 value 接近；failure 主要来自 source selection 或 label noise。
```

## H3：当前 certificate 是 construction certificate，不是 effect sufficient statistic

H3 认为：certificate pass 没有显著提高 CP precision，也没有降低 long-risk。

H3 成立标准：

```text
Lift_weak <= 1.25
Lift_longrisk >= 0.90
certificate score AUC_CP < 0.65
certificate score AUC_longrisk < 0.65
monotone sign audit fail
```

H3 失败标准：

```text
certificate pass/fail 有明显 lift，但 threshold 或 support 太弱；
则可以进入 calibration repair。
```

## H4：horizon fragility 是 generated AP 的主失败模式

H4 认为：AP1-AP4 可能在 short horizon 有少量收益，但在 80/240 horizon 变成 long-risk，因此 horizon robust precision 为 0。

H4 成立标准：

```text
weak CP at h=20 > weak CP at h=80 > weak CP at h=240；
long_risk concentrated at h=240；
short_only_generated_count high；
first_risk_horizon distribution peaks at 80/240。
```

H4 失败标准：

```text
action 在 h=20 已经失败；
不是 horizon drift，而是 immediate direction/value 错误。
```

## H5：存在 value-preserving low-distortion generated AP regime

H5 认为：当前 AP1-AP4 可能动作太大、投影太激进或 residual 方向错。通过统一的、非 dataset-specific 的 scale/projection/norm/horizon 约束，可以找到低扰动生成区间。

H5 成立标准：

```text
在 generic α/rank/tail-clamp grid 中，至少一个 APk-v2 满足：
  generated_oracle_weak_frontier_pass = 1
  V_ctrl_lcb > 0
  long_risk_rate <= 0.15 in smoke
  certificate lift weak > 1.5
  certificate lift longrisk < 0.7
```

H5 强成立标准：

```text
weak_CP_precision_cert_pass >= 0.35 in smoke
strong_CP_precision_cert_pass >= 0.10
horizon_robust_CP_precision > 0
V_ctrl_lcb > 0
```

## H6：如果 low-distortion regime 也失败，AP generated primitive should pivot

H6 认为：如果 source AP0 有 value，但所有 generated AP variants 都不能保持 value，那么当前 “generate new AP payload from AP0” 的路线应停止，转向：

```text
1. AP0 action selection + legal certificate-only controller；
2. new candidate generator upstream；
3. local exact response micro-action；
4. direct value-producing primitive from branch-delta rather than AP0 transform。
```

H6 成立标准：

```text
source_AP0 value positive；
all AP1-AP8 generated variants have V_ctrl_lcb <= 0 or long_risk high；
transformation damage median < 0 for all variants。
```

---

# 6. v9.3.9 并行执行结构

为加快实验，v9.3.9 分为四个并行批次，但 route promotion 仍按 gate 顺序执行。

## Batch A：root-cause autopsy

```text
P0 boundary reanalysis
P1 source AP0 paired baseline
P2 transformation damage matrix
P3 certificate sufficiency / lift audit
P4 horizon failure dissection
```

目标：判断 failure 是 source、generator、certificate、horizon 哪类。

## Batch B：generator v2 smoke

```text
P5 generic low-distortion sweep
P6 AP5-AP8 value-preserving primitive implementation
P7 generated AP v2 smoke outcome materialization
P8 generated AP frontier and certificate calibration
```

目标：不按 dataset 调参，只测试 generic primitive changes 是否能恢复 value。

## Batch C：runtime and materializer support

```text
P9 materializer scale and payload runtime diagnostics
```

目标：确保如果 P8 有 survivor，可以快速进入 selected runtime；若 P8 没 survivor，runtime 保持 diagnostic not official。

## Batch D：gated downstream

```text
P10 system gate
P11 LDO/LSO
P12 diagnostic paired replay
P13 short/full/sample-efficiency/continual robustness
```

只有 P8/P10 pass 后打开 official downstream。

---

# 7. 实验阶段

---

## P0：v9.3.8 boundary independent reanalysis

### 目标

独立复算 v9.3.8 的 gate gap，不只读 route 名称。把 aggregate precision 转为真实 count，把 per-primitive shortfall 算清楚。

### 对应假设

H0：v9.3.8 failure 是真实 AP generated action failure，不是 materialization bug。

### 必须记录

```text
source_run_id
source_artifact_hash
route
base_candidate
generated_action_count_total
primitive_materialized_count
actions_per_primitive
source_ap0_action_count
certificate_rows
certificate_pass_count
certificate_pass_outcome_count
branch_horizon_row_count_expected
branch_horizon_row_count_actual
branch_completion_rate
horizon_completion_rate
weak_CP_precision_certificate_pass
strong_CP_precision_certificate_pass
horizon_robust_CP_precision_certificate_pass
long_risk_rate_certificate_pass
V_ctrl_lcb_certificate_pass
weak_CP_count_certificate_pass
strong_CP_count_certificate_pass
long_risk_count_certificate_pass
weak_CP_shortfall_to_0p35
weak_CP_shortfall_to_0p75
long_risk_excess_over_0p15
long_risk_excess_over_0p05
per_primitive_weak_CP_count
per_primitive_strong_CP_count
per_primitive_long_risk_count
per_primitive_V_ctrl_lcb
```

### 判断标准

P0 pass：

```text
所有 v9.3.8 summary metrics 可复算；
counts 与 precision/rate 一致；
branch/horizon completion = 1；
H0 materializer confidence = 1。
```

P0 fail：

```text
指标不可复算；
branch/horizon rows 不一致；
label exclusivity 或 hash audit 有 violation。
```

### 可视化

```text
p0_v9380_gate_gap_counts.svg
p0_per_primitive_cp_longrisk_bar.svg
p0_certificate_pass_value_distribution.svg
p0_route_ladder_v9370_to_v9380.svg
```

### Artifact

```text
p0_v9380_boundary_reanalysis.csv
v9380_gate_gap_count_table.csv
figures/p0_*.svg
```

---

## P1：source AP0 same-panel paired baseline

### 目标

对 v9.3.8 用来生成 AP1-AP4 的 64 个 source AP0 actions，跑完全相同的 branch/horizon smoke protocol。必须知道 source AP0 action 本身到底好不好。

### 对应假设

H1：source AP0 子集可能不是 value-positive source。

### 实现

新增 runner：

```text
experiments/run_v9390_source_ap0_same_panel_baseline.py
```

对每个 source action $i$ 记录：

```text
AP0 source payload outcome under the same horizons = 20,80,240
same branch set as generated AP smoke
same controls
same label definitions
same V_ctrl definition
same hash / replay audit
```

Branches：

```text
RealAP0
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledAP0Payload
CertificatePassNoPayload
```

### 必须记录

```text
source_action_id
source_payload_hash
source_event_id
source_family_id
source_horizon_origin
source_dataset
source_seed
source_step
source_AP0_action_apply_error_linf
source_AP0_branch_horizon_completion
horizon
branch
V_ctrl
weak_CP
strong_CP
bad_event
null_event
long_risk
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
```

Aggregate：

```text
source_AP0_weak_CP_precision
source_AP0_strong_CP_precision
source_AP0_horizon_robust_precision
source_AP0_long_risk_rate
source_AP0_V_ctrl_mean
source_AP0_V_ctrl_lcb
source_AP0_short_only_count
source_AP0_long_risk_count
```

### 判断标准

P1-source-good：

```text
source_AP0_weak_CP_precision >= 0.35
source_AP0_V_ctrl_lcb > 0
source_AP0_long_risk_rate <= 0.15
```

P1-source-bad：

```text
source_AP0_weak_CP_precision < 0.20
or source_AP0_V_ctrl_lcb <= 0
or source_AP0_long_risk_rate > 0.30
```

如果 P1-source-bad 成立，v9.3.9 不能只怪 AP1-AP4 generator；需要重建 source panel。

### 可视化

```text
p1_source_ap0_value_distribution.svg
p1_source_ap0_horizon_curve.svg
p1_source_ap0_control_comparison.svg
p1_source_ap0_vs_generated_aggregate.svg
```

### Artifact

```text
p1_source_ap0_same_panel_baseline.csv
source_ap0_branch_horizon_trace_v9390.csv
source_ap0_baseline_summary_v9390.csv
```

---

## P2：source-to-generated transformation damage matrix

### 目标

对同一个 source action，比较 AP0 payload 和 AP1/AP2/AP3/AP4 generated payload 的方向、范数、tail behavior、control value。判断 generator 哪一步造成 value damage。

### 对应假设

H2：AP1-AP4 generator 破坏了 AP0 的 hidden value geometry。

### 实现

对每个 $(source\_action_i, primitive_k)$ 计算：

```text
payload cosine with source AP0
payload norm ratio
payload linf ratio
last-layer norm ratio
rolewise norm ratios
tail-mask overlap
low-rank projection residual
AdamW cosine
negative-gradient cosine
linearized CE delta
linearized margin delta
source AP0 V_ctrl at each horizon
generated APk V_ctrl at each horizon
Damage_k(i,h)
```

### 必须记录

```text
source_action_id
generated_action_id
primitive_id
source_payload_hash
generated_payload_hash
CosSrc
NormRatio
LinfRatio
LastLayerNormRatio
RolewiseNormRatio
TailMaskOverlap
LowRankResidualNorm
CosAdamW
CosNegGrad
LinearizedCEDeltaSource
LinearizedCEDeltaGenerated
LinearizedMarginDeltaSource
LinearizedMarginDeltaGenerated
V_ctrl_source_h20
V_ctrl_source_h80
V_ctrl_source_h240
V_ctrl_generated_h20
V_ctrl_generated_h80
V_ctrl_generated_h240
Damage_h20
Damage_h80
Damage_h240
Damage_mean
Damage_min
long_risk_generated
long_risk_source
weak_CP_source
weak_CP_generated
```

### 判断标准

P2-generator-damage：

```text
median Damage_mean < 0
P(Damage_mean < -δ_damage) >= 0.50
source-good rows lose CP after generation >= 0.50
```

P2-operation-attribution pass：

```text
at least one operation feature explains damage with AUC_damage >= 0.70
or Spearman |rho| >= 0.30
```

### 可视化

```text
p2_damage_heatmap_source_by_primitive.svg
p2_damage_vs_cos_src.svg
p2_damage_vs_norm_ratio.svg
p2_damage_vs_adamw_conflict.svg
p2_source_generated_horizon_lines.svg
p2_operation_damage_attribution_bar.svg
```

### Artifact

```text
p2_transformation_damage_matrix.csv
source_generated_payload_geometry_trace_v9390.csv
damage_attribution_summary_v9390.csv
```

---

## P3：certificate sufficiency and lift audit

### 目标

判断 certificate pass 是否真的筛出了更好的 action。v9.3.8 只给出了 cert-pass rows 的低 precision；v9.3.9 必须比较 cert-pass 与 cert-fail。

### 对应假设

H3：当前 certificate 是 construction certificate，不是 effect sufficient statistic。

### 实现

对所有 generated AP action rows 记录每个 certificate primitive component：

```text
certificate_value_lcb
certificate_bad_ucb
certificate_null_ucb
certificate_support_lcb
certificate_horizon_score
certificate_tail_safety
certificate_adamw_conflict
certificate_norm_bound
certificate_reliability
certificate_pass_bit
```

分析：

```text
pass vs fail weak CP precision
pass vs fail strong CP precision
pass vs fail long-risk rate
certificate score AUC for weak CP
certificate score AUC for long-risk
certificate component ablation
monotone sign consistency
calibration curve
```

### 必须记录

```text
generated_action_id
primitive_id
certificate_pass
certificate_component_id
certificate_component_value
weak_CP
strong_CP
horizon_robust_CP
long_risk
V_ctrl
horizon
certificate_score
certificate_score_rank
certificate_component_ablation_score
```

Aggregate：

```text
P_weak_CP_given_cert_pass
P_weak_CP_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_weak
Lift_strong
Lift_longrisk
AUC_certificate_weak_CP
AUC_certificate_strong_CP
AUC_certificate_longrisk
ECE_certificate_CP
monotone_sign_pass
```

### 判断标准

P3 certificate useful diagnostic：

```text
Lift_weak >= 1.50
Lift_longrisk <= 0.70
AUC_certificate_weak_CP >= 0.65
AUC_certificate_longrisk >= 0.65
monotone_sign_pass = 1
```

P3 certificate insufficient：

```text
Lift_weak <= 1.25
or Lift_longrisk >= 0.90
or AUC_certificate_weak_CP < 0.60
or monotone_sign_pass = 0
```

### 可视化

```text
p3_cert_pass_fail_cp_lift.svg
p3_certificate_component_auc.svg
p3_certificate_score_calibration.svg
p3_certificate_score_vs_Vctrl.svg
p3_certificate_monotonicity_audit.svg
```

### Artifact

```text
p3_certificate_sufficiency_lift_audit.csv
certificate_component_trace_v9390.csv
certificate_ablation_summary_v9390.csv
```

---

## P4：horizon failure dissection

### 目标

确定 generated AP failure 是 immediate failure 还是 long-horizon failure。horizon robust precision 为 0，必须知道风险在哪个 horizon 出现。

### 对应假设

H4：horizon fragility 是 generated AP 的主失败模式。

### 必须记录

```text
generated_action_id
source_action_id
primitive_id
horizon
V_ctrl
weak_CP
strong_CP
bad_event
null_event
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
first_CP_horizon
first_bad_horizon
first_long_risk_horizon
short_only_CP
long_risk
horizon_robust_CP
V_ctrl_slope_20_to_80
V_ctrl_slope_80_to_240
```

### 判断标准

P4 long-horizon-risk primary：

```text
weak_CP_precision_h20 >= weak_CP_precision_h80 >= weak_CP_precision_h240
long_risk_rate_h240 >= 0.25
short_only_CP_count >= 0.25 * weak_CP_h20_count
```

P4 immediate-direction-fail：

```text
weak_CP_precision_h20 < 0.15
V_ctrl_lcb_h20 <= 0
```

### 可视化

```text
p4_horizon_cp_precision_curve.svg
p4_horizon_vctrl_curve_by_primitive.svg
p4_first_bad_horizon_hist.svg
p4_short_only_vs_long_risk_matrix.svg
p4_horizon_damage_source_to_generated.svg
```

### Artifact

```text
p4_horizon_failure_dissection.csv
horizon_value_curve_trace_v9390.csv
```

---

## P5：generic low-distortion generator sweep

### 目标

不按 dataset 调参，测试 AP1-AP4 是否存在低扰动 / 缩放 / 投影 / tail clamp regime 可以恢复 value。

### 对应假设

H5：存在 value-preserving low-distortion generated AP regime。

### Candidate sweep

所有 primitive 使用同一套 generic grid：

```text
scale α ∈ {0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00}
source_mix β ∈ {0.00, 0.25, 0.50, 0.75}
adamw_residual_mix γ ∈ {0.00, 0.10, 0.25}
tail_clamp τ ∈ {none, p90, p95, p99}
last_layer_only ∈ {0,1}
low_rank_rank r ∈ {4,8,16,32}
horizon_guard ∈ {none, 20-only, 20+80, 20+80+240}
```

生成形式示例：

$$
\Delta\theta_{APk,v2}
=\alpha\cdot P_{safe}(\Delta\theta_{APk})
+\beta\cdot \Delta\theta_{AP0}
+\gamma\cdot P_{orth}(\Delta\theta_{AdamW}).
$$

其中 $P_{safe}$ 是统一的、非 dataset-specific 的投影或裁剪算子。

### 必须记录

```text
sweep_id
primitive_id
source_action_id
alpha
beta
gamma
tail_clamp
last_layer_only
low_rank_rank
horizon_guard
generated_action_id
payload_hash
certificate_hash
action_apply_error_linf
CosSrc
NormRatio
CosAdamW
LinearizedCEDelta
LinearizedMarginDelta
certificate_pass
V_ctrl_h20
V_ctrl_h80
V_ctrl_h240
weak_CP_h20
weak_CP_h80
weak_CP_h240
strong_CP_h20
strong_CP_h80
strong_CP_h240
long_risk
horizon_robust_CP
```

### 判断标准

P5 smoke survivor：

```text
V_ctrl_lcb > 0
weak_CP_precision_cert_pass >= 0.35
strong_CP_precision_cert_pass >= 0.10
long_risk_rate_cert_pass <= 0.15
horizon_robust_CP_precision_cert_pass > 0
Lift_weak >= 1.50
Lift_longrisk <= 0.70
```

P5 fail：

```text
所有 generic sweep candidates:
  V_ctrl_lcb <= 0
  or weak_CP_precision_cert_pass < 0.20
  or long_risk_rate_cert_pass > 0.25
```

### 可视化

```text
p5_scale_value_risk_frontier.svg
p5_normratio_vctrl_frontier.svg
p5_alpha_beta_heatmap.svg
p5_tail_clamp_longrisk.svg
p5_horizon_guard_effect.svg
p5_lowdistortion_pareto.svg
```

### Artifact

```text
p5_generic_low_distortion_sweep.csv
low_distortion_payload_trace_v9390.csv
low_distortion_outcome_summary_v9390.csv
```

---

## P6：AP5-AP8 value-preserving certificate primitive implementation

### 目标

基于 P1-P5 的 root cause，真实实现 AP5-AP8，不只是 diagnostic certificate。每个 primitive 必须生成 payload、certificate tensor、hash，并可 action replay。

### 对应假设

H5/H6：要么存在 value-preserving AP generator，要么 generated AP 路线应 pivot。

### New primitives

#### AP5：Low-Distortion Source-Preserving Certificate

目标：尽量保留 source AP0 的有效方向，只做安全裁剪和低扰动缩放。

$$
\Delta\theta_{AP5}
=\alpha^* P_{clip}(\Delta\theta_{AP0}).
$$

Certificate：

```text
CosSrc >= τ_cos
NormRatio <= τ_norm
LinearizedCEDelta <= 0
TailDamageEstimate <= τ_tail
HorizonRiskUCB <= τ_h
```

#### AP6：AdamW-Compatible Residual Certificate

目标：避免与 AdamW 主下降方向冲突。

$$
\Delta\theta_{AP6}
=\alpha^*\left(
\Delta\theta_{AP0}-Proj_{conflict}(\Delta\theta_{AP0},\Delta\theta_{AdamW})
\right).
$$

Certificate：

```text
CosAdamW >= τ_adamw
ConflictAdamW <= τ_conflict
LinearizedCEDelta <= 0
NormRatio bounded
```

#### AP7：Horizon-Guarded Conservative Certificate

目标：优先降低 long-risk，而不是最大化 short horizon gain。

$$
AcceptCert_{AP7}=1
\iff
\widehat V_{20}>0
\land
\widehat V_{80}>0
\land
\widehat{Risk}_{240}\le\tau_{240}.
$$

Generated payload 使用最小安全 scale：

$$
\alpha^*=\arg\min_{\alpha\in A}\alpha
\quad
\text{s.t.}
\quad
\widehat V_{20}(\alpha)>0.
$$

#### AP8：Control-Residual Value Certificate

目标：显式打过 AdamWParallel / bestLR 的局部控制方向，而不是只追 CE descent。

$$
\widehat V_{ctrl}
=\widehat V_{AP}-\max(\widehat V_{AdamWParallel},\widehat V_{bestLR},\widehat V_{NoOp}).
$$

Certificate：

```text
V_ctrl_LCB_pred > 0
Bad_UCB_pred <= τ_b
Null_UCB_pred <= τ_n
Support_LCB >= τ_s
Cost <= Cmax
```

### 必须记录

```text
primitive_id
primitive_version
generated_action_count
source_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
certificate_schema_contract_pass
certificate_pass_count
certificate_components
uses_dataset_name
uses_outcome_at_commit
uses_future_outcome
uses_validation_test
```

### 判断标准

P6 implementation pass：

```text
generated_action_count > 0 for AP5/AP6/AP7/AP8
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= tolerance
certificate_schema_contract_pass = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

### 可视化

```text
p6_generated_count_by_primitive.svg
p6_certificate_pass_rate_by_primitive.svg
p6_payload_geometry_by_primitive.svg
p6_legality_contract_dashboard.svg
```

### Artifact

```text
p6_real_ap5_ap8_generator.csv
ap5_ap8_payload_trace_v9390.csv
ap5_ap8_certificate_trace_v9390.csv
action_apply_replay_trace_v9390.csv
```

---

## P7：generated AP v2 smoke outcome materialization

### 目标

对 AP5-AP8 generated payload 跑 same-run branch/horizon outcomes。不得使用 AP0 old outcome 替代。

### Materializer

```text
materializer_id = APSMOKE2-GeneratedAPV2PayloadBranchHorizonSmoke
```

Branches：

```text
RealAP
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledAPPayload
CertificatePassNoPayload
SourceAP0Payload
```

Horizon：

```text
20, 80, 240
```

### 必须记录

```text
generated_action_id
source_action_id
primitive_id
primitive_version
certificate_pass
branch
horizon
V_ctrl
weak_CP
strong_CP
horizon_robust_CP
long_risk
bad_event
null_event
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
source_AP0_V_ctrl
Damage_vs_source
```

### 判断标准

P7 smoke outcome pass：

```text
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
control_branch_missing_count = 0
label_exclusivity_violation_count = 0
metric_nan_count = 0
metric_inf_count = 0
```

P7 value smoke pass：

```text
at least one primitive_version has:
  V_ctrl_lcb > 0
  weak_CP_precision_cert_pass >= 0.35
  strong_CP_precision_cert_pass >= 0.10
  long_risk_rate_cert_pass <= 0.15
  horizon_robust_CP_precision_cert_pass > 0
```

### 可视化

```text
p7_ap5_ap8_value_risk_frontier.svg
p7_primitive_horizon_cp_matrix.svg
p7_source_vs_generated_damage_v2.svg
p7_branch_control_win_rate.svg
```

### Artifact

```text
p7_generated_ap_v2_smoke_outcome.csv
ap_v2_smoke_outcome_trace_v9390.csv
branch_horizon_completion_trace_v9390.csv
```

---

## P8：generated AP frontier and certificate calibration

### 目标

如果 P7 有 value smoke survivor，判断 generated AP v2 是否形成可部署 frontier，并进行 cross-fitted certificate calibration。P8 是是否进入 system/runtime 的关键 gate。

### 对应假设

H5：value-preserving generated AP regime 存在。

### Controller candidates

#### C0：certificate-pass only negative control

```text
Accept(e)=CertPass(e)
```

#### C1：certificate score monotone gate

$$
S_{cert}(e)
=LCB(\widehat V_{ctrl}(e))
-UCB(\widehat B(e))
-UCB(\widehat N(e))
+LCB(\widehat S(e))
-C(e).
$$

$$
Accept(e)=1\iff S_{cert}(e)\ge\tau.
$$

#### C2：horizon-safe certificate gate

$$
Accept(e)=1
\iff
LCB(\widehat V_{20})>0
\land LCB(\widehat V_{80})>0
\land UCB(\widehat R_{240})\le\tau_{240}.
$$

#### C3：source-preserving + value gate

$$
Accept(e)=1
\iff
CosSrc(e)\ge\tau_{cos}
\land NormRatio(e)\le\tau_{norm}
\land LCB(\widehat V_{ctrl})>0
\land UCB(LongRisk)\le\tau_{risk}.
$$

### Splits

```text
seed folds
leave-dataset-out diagnostic
leave-family-out diagnostic
leave-horizon-origin-out diagnostic
```

No dataset-specific branch is allowed.

### 必须记录

```text
controller_id
primitive_id
primitive_version
feature_set
thresholds
calibration_fold
heldout_fold
accepted_count_cal
accepted_count_heldout
weak_CP_precision_cal
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_CP_precision_heldout
long_risk_rate_heldout
V_ctrl_mean_heldout
V_ctrl_lcb_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
certificate_lift_weak
certificate_lift_longrisk
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
dataset_name_used
uses_outcome_at_commit
```

### 判断标准

P8 generated AP frontier pass：

```text
coverage_heldout in [0.03,0.15]
weak_CP_precision_heldout >= 0.75
long_risk_rate_heldout <= 0.05
V_ctrl_lcb_heldout > 0
horizon_robust_CP_precision_heldout >= 0.03
accepted_family_count >= 16 in smoke, >=32 in full
accepted_signal_strata_count >= 5
dataset_name_used = 0
uses_outcome_at_commit = 0
```

P8 weak survivor for scale-up：

```text
weak_CP_precision_heldout >= 0.50
long_risk_rate_heldout <= 0.10
V_ctrl_lcb_heldout > 0
coverage_heldout >= 0.02
```

If P8 fails for all AP5-AP8, route goes to:

```text
R7-GeneratedAPFrontierAbsent
```

and the next pivot is AP0 certificate-only selection or candidate source redesign.

### 可视化

```text
p8_certificate_controller_frontier.svg
p8_calibration_to_heldout_drift.svg
p8_horizon_robust_frontier.svg
p8_source_preservation_vs_value.svg
p8_support_balance_dashboard.svg
```

### Artifact

```text
p8_generated_ap_frontier_certificate_calibration.csv
certificate_controller_trace_v9390.csv
frontier_support_trace_v9390.csv
```

---

## P9：materializer scale and payload runtime diagnostics

### 目标

如果 P8 有 survivor，提前准备 selected runtime；如果 P8 没 survivor，runtime 只保留 diagnostic，不得 official。

### 必须记录

```text
runtime_candidate_id
primitive_id
primitive_version
selected_controller_used
payload_apply_used
payload_preloaded
feature_compute_time_ms_q90
certificate_score_time_ms_q90
accept_decision_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
action_apply_error_linf_max
payload_apply_cosine_min
materializer_rows_per_sec
branch_runtime_ms_q90
```

### 判断标准

P9 diagnostic runtime promising：

```text
payload_apply_time_ms_q90 <= 0.10
step_ratio_q90 <= 1.50 under selected-controller replay
memory_ratio <= 1.05
action_apply_error_linf_max <= tolerance
```

P9 official runtime pass requires P8 controller pass first：

```text
selected_controller_used = 1
selected_payload_apply_used = 1
diagnostic_derived_from_measured_components = 0
```

### 可视化

```text
p9_payload_runtime_waterfall.svg
p9_step_ratio_distribution.svg
p9_runtime_vs_accept_count.svg
p9_materializer_throughput.svg
```

### Artifact

```text
p9_payload_runtime_diagnostic_v9390.csv
payload_runtime_component_trace_v9390.csv
materializer_throughput_trace_v9390.csv
```

---

## P10：system gate for generated AP v2

### 目标

只有 P8 certificate controller pass + P9 selected runtime measured 后，才组合成 system candidate。

### 必须记录

```text
system_candidate_id
primitive_id
primitive_version
controller_id
runtime_candidate_id
generated_action_count
accepted_count
coverage
weak_CP_precision
strong_CP_precision
horizon_robust_CP_precision
long_risk_rate
V_ctrl_lcb
bad_event_rate
null_rate
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
step_ratio_q90
memory_ratio
payload_binding_pass
action_apply_pass
certificate_schema_contract_pass
materialized_system_path
selected_controller_used
selected_payload_runtime_measured
diagnostic_derived_from_measured_components
dataset_name_used
uses_outcome_at_commit
uses_future_outcome
uses_validation_test
official_eligible
system_legal_controller_pass
```

### 判断标准

P10 system pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
coverage in [0.03,0.15]
weak_CP_precision >= 0.75
long_risk_rate <= 0.05
V_ctrl_lcb > 0
horizon_robust_CP_precision >= 0.03
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_binding_pass = 1
action_apply_pass = 1
certificate_schema_contract_pass = 1
materialized_system_path = 1
diagnostic_derived_from_measured_components = 0
dataset_name_used = 0
uses_outcome_at_commit = 0
```

### 可视化

```text
p10_system_gate_dashboard.svg
p10_quality_cost_frontier.svg
p10_controller_runtime_pareto.svg
```

### Artifact

```text
p10_system_gate_generated_ap_v2.csv
system_controller_trace_v9390.csv
```

---

## P11：leave-dataset-out / leave-stratum-out

### 目标

只有 P10 pass 后 official 打开。证明 selected generated AP primitive 不依赖 dataset tuning。

### 设置

Leave-dataset-out：

```text
calibrate MNIST + Fashion-MNIST, evaluate KMNIST
calibrate MNIST + KMNIST, evaluate Fashion-MNIST
calibrate Fashion-MNIST + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate all but one signal stratum
evaluate held-out stratum
```

Leave-family-out diagnostic：

```text
calibrate all but one high-volume event family
evaluate held-out family
```

### 必须记录

```text
split_type
heldout_entity
primitive_id
controller_id
accepted_count
coverage
weak_CP_precision
strong_CP_precision
horizon_robust_CP_precision
long_risk_rate
V_ctrl_lcb
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### 判断标准

P11 LDO pass：

```text
at least 2/3 held-out datasets pass weak_CP_precision >= 0.75 and long_risk <= 0.05
coverage > 0 for all held-out datasets
no held-out dataset has long_risk_rate > 0.10
dataset_name_used = 0
```

P11 LSO pass：

```text
>=70% held-out strata pass task-safety / CP weak gate
macro V_ctrl_lcb > 0
```

### 可视化

```text
p11_leave_dataset_out_matrix.svg
p11_leave_stratum_out_matrix.svg
p11_leave_family_out_diagnostic.svg
```

### Artifact

```text
p11_leave_dataset_stratum_out_v9390.csv
leaveout_trace_v9390.csv
```

---

## P12：diagnostic paired replay scout

### 目标

只有 P10 pass 后 official；如果 P10 weak survivor 但未 official，可以跑 diagnostic scout，但结果不能反向调 P8/P10 threshold。

### Branches

```text
RealGeneratedAP
SourceAP0
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledGeneratedAPPayload
ShuffledCertificateScore
CertificatePassNoPayload
FunctionalChannelShuffled
TailMaskShuffled
RoleScoreShuffled
DatasetRouteShuffled
EventRouteShuffled
```

### 必须记录

```text
primitive_id
controller_id
dataset
seed
horizon
branch
accepted_count
coverage
V_ctrl
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
beats_adamwparallel
beats_bestlr
beats_source_AP0
beats_noop
beats_random
shuffle_control_pass
step_ratio_q90
memory_ratio
status = diagnostic_not_official or official
```

### 判断标准

Promising diagnostic：

```text
RealGeneratedAP beats AdamWParallel in >= 50% macro slices
RealGeneratedAP beats bestLR in >= 50% macro slices
RealGeneratedAP beats SourceAP0 in at least one meaningful macro mechanism metric
shuffled controls do not match RealGeneratedAP
```

Official paired replay pass requires P10/P11 pass first。

### 可视化

```text
p12_paired_replay_branch_pareto.svg
p12_macro_beat_rate.svg
p12_shuffle_control_matrix.svg
p12_source_ap0_vs_generated_ap.svg
```

### Artifact

```text
p12_diagnostic_paired_replay_scout_v9390.csv
paired_replay_branch_trace_v9390.csv
```

---

## P13：short/full/sample-efficiency/continual robustness boundary

### 目标

只有 P10/P11/P12 official pass 后打开。v9.3.9 通常不应到这里；但 plan 要提前定义，不允许后续临时改 gate。

### 必须记录

```text
dataset
seed
primitive_id
controller_id
steps
train_loss
val_loss
test_loss
final_acc
best_val_acc
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
bad_event_rate
long_risk_rate
forgetting
retained_accuracy
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_drift
step_ratio_q90
memory_ratio
strong_baseline_beaten
```

### 判断标准

Full functional pass：

```text
task safety: Acc_functional >= Acc_AdamW - 0.005
control advantage: beats AdamWParallel / bestLR macro >= 0.60
at least one mechanism improvement: ECE/NLL/CEp99/margin/curvature/sample-efficiency
continual forgetting not worse than AdamW by more than tolerance
strong baseline not explaining gain
```

### 可视化

```text
p13_sample_efficiency_curves.svg
p13_valloss_auc_time.svg
p13_continual_forgetting_matrix.svg
p13_robustness_strong_baseline_dashboard.svg
```

### Artifact

```text
p13_short_full_sampleeff_continual_robustness_v9390.csv
short_full_trace_v9390.csv
continual_trace_v9390.csv
```

---

# 8. 关键 route decision

```text
R0-BoundaryUnstable:
  v9.3.8 metrics cannot be reproduced or materialization audit fails.

R1-SourceAP0PanelBad:
  source AP0 same-panel baseline is value-negative or long-risk high.

R2-GeneratorTransformationDamage:
  source AP0 is value-positive but AP1-AP4 generated actions lose value.

R3-CertificateNotEffectSufficient:
  certificate pass/fail has no CP lift or no long-risk reduction.

R4-HorizonLongRiskPrimary:
  generated AP has short-horizon signal but long horizon risk dominates.

R5-ImmediateDirectionFail:
  generated AP is value-negative already at horizon 20.

R6-LowDistortionGeneratedAPSurvivor:
  generic low-distortion sweep finds a value-positive generated AP regime.

R7-GeneratedAPFrontierAbsent:
  AP5-AP8 generated AP actions have no oracle or certificate-pass frontier.

R8-CertificateControllerPass:
  generated AP certificate controller passes decision/value/horizon gates.

R9-SelectedPayloadRuntimePass:
  selected generated AP payload runtime passes system envelope.

R10-SystemLegalGeneratedAPControllerPass:
  P10 system gate passes.

R11-LeaveDatasetStratumOutPass:
  selected generated AP controller generalizes.

R12-PairedReplayPass:
  official paired replay beats controls.

R13-FullFunctionalPass:
  short/full/sample-efficiency/continual robustness pass.

R14-PivotToAP0CertificateOnlyController:
  AP generated route fails, but source AP0 value-positive and selectable.

R15-PivotToNewCandidateSource:
  source AP0 and generated AP both lack reliable value-positive frontier.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
source_route_v9380
source_AP0_same_panel_pass
generator_damage_pass
certificate_sufficiency_pass
horizon_failure_mode
low_distortion_sweep_pass
ap5_ap8_generation_pass
ap_v2_smoke_outcome_pass
ap_v2_frontier_pass
certificate_controller_pass
selected_payload_runtime_pass
system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_full_continual_pass
primary_blocker
next_required_implementation
fake_data_used
proxy_row_used
cpu_offload_used
dataset_name_used
uses_outcome_at_commit
uses_future_outcome
uses_validation_test
diagnostic_promoted_to_official
```

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9390.csv
provenance_audit_v9390.csv
route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv

p0_v9380_boundary_reanalysis.csv
v9380_gate_gap_count_table.csv

p1_source_ap0_same_panel_baseline.csv
source_ap0_branch_horizon_trace_v9390.csv
source_ap0_baseline_summary_v9390.csv

p2_transformation_damage_matrix.csv
source_generated_payload_geometry_trace_v9390.csv
damage_attribution_summary_v9390.csv

p3_certificate_sufficiency_lift_audit.csv
certificate_component_trace_v9390.csv
certificate_ablation_summary_v9390.csv

p4_horizon_failure_dissection.csv
horizon_value_curve_trace_v9390.csv

p5_generic_low_distortion_sweep.csv
low_distortion_payload_trace_v9390.csv
low_distortion_outcome_summary_v9390.csv

p6_real_ap5_ap8_generator.csv
ap5_ap8_payload_trace_v9390.csv
ap5_ap8_certificate_trace_v9390.csv
action_apply_replay_trace_v9390.csv

p7_generated_ap_v2_smoke_outcome.csv
ap_v2_smoke_outcome_trace_v9390.csv
branch_horizon_completion_trace_v9390.csv

p8_generated_ap_frontier_certificate_calibration.csv
certificate_controller_trace_v9390.csv
frontier_support_trace_v9390.csv

p9_payload_runtime_diagnostic_v9390.csv
payload_runtime_component_trace_v9390.csv
materializer_throughput_trace_v9390.csv

p10_system_gate_generated_ap_v2.csv
system_controller_trace_v9390.csv

p11_leave_dataset_stratum_out_v9390.csv
leaveout_trace_v9390.csv

p12_diagnostic_paired_replay_scout_v9390.csv
paired_replay_branch_trace_v9390.csv

p13_short_full_sampleeff_continual_robustness_v9390.csv
short_full_trace_v9390.csv
continual_trace_v9390.csv

figures/
```

---

# 10. Failure taxonomy

```text
F0_v9380_boundary_unstable
F1_materializer_hash_or_replay_failure
F2_source_AP0_panel_bad
F3_source_selection_bias
F4_generator_transformation_damage
F5_payload_norm_overshoot
F6_adamw_conflict_damage
F7_tail_projection_damage
F8_low_rank_projection_damage
F9_horizon_memory_damage
F10_certificate_no_CP_lift
F11_certificate_longrisk_not_reduced
F12_certificate_monotonicity_fail
F13_immediate_direction_fail
F14_long_horizon_risk_fail
F15_low_distortion_sweep_no_survivor
F16_AP5_AP8_generator_missing
F17_AP5_AP8_action_apply_fail
F18_AP_v2_smoke_outcome_incomplete
F19_generated_AP_frontier_absent
F20_certificate_controller_precision_fail
F21_certificate_controller_coverage_fail
F22_certificate_controller_longrisk_fail
F23_certificate_controller_Vctrl_fail
F24_support_balance_fail
F25_selected_payload_runtime_fail
F26_system_gate_fail
F27_leave_dataset_out_fail
F28_leave_stratum_out_fail
F29_paired_replay_control_equivalent
F30_shuffle_control_pass
F31_short_run_task_drop
F32_full_run_no_macro_gain
F33_sample_efficiency_fail
F34_continual_forgetting_fail
F35_strong_baseline_explains_gain
F36_dataset_tuning_detected
F37_future_outcome_feature_detected
F38_diagnostic_promoted_to_official
F39_fake_or_proxy_violation
F40_external_not_ready
```

---

# 11. 本轮成功与失败的解释规则

## 11.1 什么算进展

以下都算进展：

```text
1. source AP0 same-panel baseline 被测清楚；
2. transformation damage 被定位到具体 operation；
3. certificate pass/fail lift 被量化；
4. horizon risk 的 first-risk horizon 被定位；
5. low-distortion sweep 找到 value-preserving regime；
6. AP5-AP8 真实生成 payload/certificate/action apply；
7. AP v2 smoke outcomes 完整 materialize；
8. route 明确告诉我们 generated AP 应继续、低扰动重写，还是 pivot。
```

## 11.2 什么不算成功

以下不算成功：

```text
1. 只生成更多 AP action，但不跑 outcome；
2. 只让 certificate fields complete，但没有 CP lift；
3. 只让 AP3 相对最好，但 V_ctrl LCB 仍为负；
4. 只让 h=20 有收益，但 h=240 long-risk 高；
5. 只在某个 dataset 上 threshold 过线；
6. 只用 oracle labels 做 controller；
7. 只跑 diagnostic paired replay 后回头调 P8；
8. selected runtime 没 controller pass 就写成 official。
```

## 11.3 什么时候停止 generated AP route

如果满足以下组合，v9.3.9 应停止 AP1-AP8 generated route：

```text
source_AP0_same_panel_pass = 1
AND all AP1-AP8 variants have V_ctrl_lcb <= 0
AND median Damage_mean < 0 for all variants
AND certificate_lift_weak <= 1.25
AND long_risk_rate >= 0.25 for all variants
```

此时 route：

```text
R14-PivotToAP0CertificateOnlyController
```

或如果 source AP0 也 bad：

```text
R15-PivotToNewCandidateSource
```

---

# 12. 最终判断标准

v9.3.9 的强 pass：

```text
P1 source baseline complete
P2 generator damage quantified
P3 certificate lift quantified
P4 horizon failure quantified
P6 AP5-AP8 generated payload/certificate/action apply pass
P7 AP v2 smoke outcome complete
P8 generated AP certificate controller pass
P10 system legal controller pass
```

v9.3.9 的最低有效 pass：

```text
P1-P4 root-cause autopsy complete
并且 route 明确落在：
  source bad
  generator damage
  certificate insufficient
  horizon long-risk
  low-distortion survivor
  generated route pivot
之一。
```

v9.3.9 的失败：

```text
只重复 v9.3.8 的 AP1-AP4 generation；
只调 certificate threshold；
没有 source AP0 paired baseline；
没有 transformation damage matrix；
没有 pass/fail certificate lift；
没有 horizon failure dissection；
没有明确 route。
```

最终目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是继续逼近 DG-KAN 的第一性目标：

$$
\boxed{
\text{Clean PureKAN manual training system 中是否存在 task-safe、control-positive、horizon-robust、system-cheap 的 functional update rule？}
}
$$
