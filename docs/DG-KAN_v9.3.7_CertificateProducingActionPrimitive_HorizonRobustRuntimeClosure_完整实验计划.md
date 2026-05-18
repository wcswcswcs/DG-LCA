# DG-KAN v9.3.7 Certificate-Producing Action Primitive / Horizon-Robust Control-Positive Frontier / Selected Runtime Closure 完整实验计划

> 本计划基于 v9.3.6 `Legal Action-Effect Identifiability / Certificate Primitive / PayloadApply Runtime Closure` 的真实执行结果制定。  
> v9.3.7 不再继续对 AP0 opaque actions 做 feature search、threshold repair 或 probe patch。  
> 本轮目标是实现并验证 **certificate-producing action primitive**：action 在生成时必须携带 commit 前可审计的 value / risk / null / support / horizon / cost certificate，而不是先生成 opaque action，再事后猜它是否 control-positive。

---

# 0. 执行摘要

v9.3.6 的 terminal route 是：

```text
route = R12-CertificatePrimitiveFail
base_candidate = LQ-t2-h256
success_v9360_strict_purekan_functional = False
success_v9360_full_functional = False
success_v9360_external_ready = False
```

这不是普通失败。它把问题推进到了一个更深的边界：

```text
AP0 当前 action primitive 不是没有 oracle-good action；
AP0 的问题是 good action 在 legal commit-time sigma-algebra 下不可识别；
同时 AP0 的 weak CP frontier 不 horizon-robust，short-only / long-risk action 很多。
```

v9.3.6 的关键事实：

```text
weak_CP_row_count = 893
weak_CP_coverage = 0.0984347442680776
weak_CP_coverage_lcb = 0.09247342078415305
weak_CP_V_ctrl_lcb = 0.2833107634852314

strong_CP_row_count = 353
strong_CP_coverage = 0.03891093474426808

horizon_robust_CP_action_count = 13
horizon_robust_CP_coverage = 0.004298941798941799

short_only_CP_action_count = 201
long_risk_action_count = 294

best_legal_capacity_feature = F4-StateNLL
best_legal_capacity_auc_CP = 0.6151712728688206
best_top273_CP_precision = 0.12454212454212454

best_aef_feature = AEF6-ProbeReliability
best_aef_auc_CP = 0.5030687962043473
best_aef_top273_CP_precision = 0.28205128205128205
best_aef_cost_q90 = 2.4969042278826237 ms

best_microprobe_auc_CP = 0.5030687962043473
best_microprobe_cost_q90 = 2.4969042278826237 ms

certificate_primitive_smoke_pass = 0
system_legal_controller_pass = 0

best_runtime_candidate_id = RT3-LastLayerOnlyDiagnostic
payload_apply_time_ms_q90 = 0.0245068222284317 ms
step_ratio_q90 = 1.0305048474583738
payload_apply_runtime_pass = 0
```

我对 v9.3.6 的独立判断是：

$$
\boxed{
\text{AP0 is oracle-good but legally opaque and horizon-fragile.}
}
$$

因此 v9.3.7 的第一性问题不是：

```text
能不能再找一个 feature 看见 AP0？
能不能再调一个 threshold？
能不能让 microprobe 再便宜一点？
```

而是：

$$
\boxed{
\text{能否生成一种 action，使其 value / risk / horizon robustness 在生成时就有 legal certificate？}
}
$$

本轮强目标不是直接 external-ready，而是证明以下命题之一：

```text
命题 A：存在 AP1/AP2/AP3 certificate-producing primitive，能产生足够 dense、horizon-safer、legal-identifiable 的 control-positive actions。  
命题 B：在当前 LQ-t2-h256 / FC-PureKAN base 下，即使重构 primitive，certificate-producing action density 仍不足，需要更深层 primitive/base redesign。  
```

---

# 1. 当前实验进展的独立判断

## 1.1 有进展，但不是 controller 进展

v9.3.6 的真实进展不是 controller pass。P5 没有运行 threshold search，这是正确的，因为 P2/P3/P4 的 legal signal 都失败。如果在 legal observability capacity、AEF、microprobe 均失败时继续跑 controller search，那只是把 StableAccept patching 换成 AP0 feature patching。

本轮真正完成的是三件事：

```text
1. 把 full control-positive oracle 进一步分解成 weak / strong / horizon-robust / short-only / long-risk；
2. 证明 AP0 的 legal marginal features、AEF、cheap microprobe 都无法识别 CP frontier；
3. 证明 payload apply 在某些受限 runtime primitive 上可以非常快，但这还不是 official selected-controller path。
```

## 1.2 AP0 的 weak frontier 足够，但 robust frontier 很薄

weak CP coverage 是 `0.0984`，远高于 `0.03` 最低 coverage。strong CP coverage 是 `0.0389`，也刚好高于 `0.03`。这说明不应该说 AP0 完全没价值。

但是 horizon-robust CP coverage 只有 `0.0043`，而且 short-only action 有 `201`，long-risk action 有 `294`。这说明如果只用 weak CP 当目标，controller 可能学到的是短 horizon 局部收益，而不是稳定 functional advantage。

因此 v9.3.7 不能只优化：

$$
P(WeakCP=1).
$$

它必须同时优化：

$$
P(StrongCP=1),
$$

$$
P(HorizonRobustCP=1),
$$

并压低：

$$
P(ShortOnly=1), \quad P(LongRisk=1).
$$

## 1.3 legal observability 失败不是小幅不足，而是量级不足

best static / marginal legal feature 是 `F4-StateNLL`，AUC_CP = `0.6152`。更关键的是 top273 CP precision 只有 `0.1245`。

如果以 heldout denominator `9072` 计算，coverage 下限 `0.03` 对应 accepted count 约为：

$$
N_{accept,min}=\lceil 0.03\times 9072\rceil=273.
$$

`top273 CP precision = 0.1245` 意味着 top273 中大约只有：

$$
273\times0.1245\approx34
$$

个 CP row。这离 official controller 所需的 safe/useful density不是差一点，而是方向错了。

AEF / microprobe 的 top273 precision 稍高，为 `0.2821`，对应约：

$$
273\times0.2821\approx77
$$

个 CP row。但 AEF AUC 只有 `0.5031`，且 cost q90 = `2.4969 ms`，既不稳定，也太贵。

所以 v9.3.7 不应该再问：

```text
哪个 AP0 事后 feature 更好？
```

而应该问：

```text
怎样让 action 生成过程直接产生可验证的 certificate？
```

## 1.4 runtime 的好消息只能作为设计线索，不能转正

v9.3.5 的真实 payload-apply diagnostic step ratio 是 `2.0687`，payload apply q90 是 `0.762 ms`。v9.3.6 的 `RT3-LastLayerOnlyDiagnostic` 把 payload apply q90 降到 `0.0245 ms`，step ratio q90 到 `1.0305`。

这说明受限 action payload 形态可能让 runtime 进入 envelope。但这还不是 official，因为：

```text
没有 selected legal controller；
runtime path 是 microbench；
RT3 是 LastLayerOnlyDiagnostic，不一定覆盖真实 selected primitive；
payload apply pass 不能先于 decision / certificate pass。
```

不过它给了 v9.3.7 一个重要方向：certificate-producing primitive 应尽量采用 runtime-native payload 形态，例如 last-edge / block-sparse / low-rank / fused-apply，而不是生成任意全量 AP0 payload。

---

# 2. 是否在正确道路上

高层路线仍然正确。项目没有为了 MNIST / Fashion-MNIST / KMNIST 打榜而按 dataset 调参，也没有把 full oracle label、AEF/probe diagnostic 或 runtime microbench 写成 official system pass。v9.3.6 的 gate 很严格，严格是对的。

但具体路线必须再次 pivot：

```text
v9.2.80-v9.2.82:
  StableAccept final-controller route 结束。

v9.3.0-v9.3.5:
  action/control outcome materialization 与 full CP oracle 证明完成。

v9.3.6:
  AP0 legal identifiability failure 证明完成。

v9.3.7:
  必须实现 AP1/AP2/AP3 certificate-producing action primitive。
```

如果 v9.3.7 继续做下面这些，就会偏离正确道路：

```text
继续筛 AP0 static features；
继续微调 F4-StateNLL / StateMargin / PayloadNorm；
继续扩展 PR5-ProbeReliability 的低层视角；
继续拿 weak CP oracle 当 robust functional advantage；
继续把 LastLayerOnly microbench 当 system runtime pass；
继续在没有 certificate primitive 的情况下搜索 controller threshold。
```

正确道路是：

$$
\boxed{
\text{Action generator must produce the evidence required by the controller.}
}
$$

---

# 3. 离终极目标还有多远

当前已经完成或基本完成：

```text
LQ-t2-h256 base anchor；
manual forward / backward / AdamW update contract；
primary labels；
durable payload；
full control outcome universe；
weak control-positive oracle frontier；
AP0 non-identifiability diagnosis；
no fake / no proxy / no dataset-tuning discipline；
runtime microbench 的一个可行工程线索。
```

当前未完成：

```text
certificate-producing action primitive；
horizon-robust CP density；
legal certificate -> CP 的 stable mapping；
selected controller；
selected-controller payload runtime；
LDO / LSO；
official paired replay；
short/full run；
sample efficiency；
continual / anti-forgetting；
strong baseline exclusion；
external-ready package。
```

距离 system-legal local controller 还差三道硬门：

```text
1. AP1/AP2/AP3 primitive 必须产生足够 dense 的 certificate-pass actions；
2. certificate-pass actions 必须在 heldout / LDO / LSO 上 control-positive、horizon-safer；
3. selected primitive + controller + payload apply 必须 measured step_ratio_q90 <= 1.50。
```

距离 strict PureKAN functional causal evidence 还要再加：

```text
official paired replay beats AdamWParallel / bestLR；
shuffled controls fail；
short-run task-safe mechanism gain；
strong LR grid / QuadraticFeatureMLP / shuffled payload 不能解释。
```

所以当前不能说快成功了。更准确地说：

$$
\boxed{
\text{项目已经走到“必须发明可证明 action primitive”的阶段。}
}
$$

---

# 4. v9.3.7 总体目标

v9.3.7 的总体目标是：

$$
\boxed{
\text{实现并验证 certificate-producing functional action primitive，使 action 在生成时自带 legal value/risk/horizon/cost certificate。}
}
$$

AP0 的旧形式是：

$$
\Delta\theta_{AP0}=Generator(x_t,\theta_t),
$$

然后事后用 feature 判断：

$$
Score(x_t,\Delta\theta_{AP0})\rightarrow Accept.
$$

v9.3.7 目标形式是：

$$
(\Delta\theta_{APk}, Cert_{APk})=Generator_{cert}(x_t,\theta_t,g_t,State_t),
$$

其中：

$$
Cert(e)=
\{C_V(e), C_B(e), C_N(e), C_S(e), C_H(e), C_C(e)\}.
$$

Accept rule 不再依赖 opaque feature search，而是：

$$
Accept(e)=1
\iff
C_V(e)=1
\land C_B(e)=1
\land C_N(e)=1
\land C_S(e)=1
\land C_H(e)=1
\land C_C(e)=1.
$$

其中：

```text
C_V: value certificate，证明 action 有正收益倾向；
C_B: bad-tail certificate，证明 bad risk 被控制；
C_N: non-null certificate，证明 action 不是 no-op；
C_S: support certificate，证明当前 certificate family 有统计支持；
C_H: horizon robustness certificate，证明不是 short-only / long-risk；
C_C: cost certificate，证明 online runtime 可进入 envelope。
```

v9.3.7 的最低有效推进：

```text
1. 实现至少 3 个 AP certificate primitive；
2. 每个 primitive 真实生成 action + certificate + payload；
3. 在 stratified smoke panel 上测 weak/strong/horizon-robust CP density；
4. 至少一个 primitive 的 certificate-pass mask 明显优于 AP0 static/probe；
5. 至少一个 primitive 的 selected runtime measured，而不是 microbench-only；
6. 如果全部失败，能定位是 value density 不足、horizon robustness 不足、certificate calibration 不足，还是 runtime payload 形态不合格。
```

v9.3.7 strong pass：

```text
certificate_action_primitive_pass = 1
horizon_robust_or_strong_CP_frontier_pass = 1
certificate_controller_pass = 1
selected_payload_runtime_pass = 1
system_legal_controller_pass = 1
```

---

# 5. 本轮明确不做什么

v9.3.7 不做：

```text
1. 不继续 AP0 feature search；
2. 不继续把 F4-StateNLL / PayloadNorm / StateMargin 当主路线；
3. 不继续调 PR5-ProbeReliability threshold；
4. 不把 AEF/probe diagnostic 写成 official controller；
5. 不用 weak CP oracle mask 选 action；
6. 不用 future outcome label at commit time；
7. 不按 dataset 调 threshold、primitive、runtime path；
8. 不用 validation/test metric 进入 action generation；
9. 不把 LastLayerOnlyDiagnostic microbench 写成 selected-controller runtime；
10. 不在 system controller pass 前打开 official paired replay；
11. 不把 short-only weak CP 当 robust functional advantage；
12. 不放宽 step_ratio_q90 <= 1.50 或 bad/null gates。
```

允许做：

```text
1. dataset-level diagnostics，但不能 dataset-specific dispatch；
2. AP primitive family diagnostics；
3. calibration split 上冻结 certificate thresholds；
4. leave-dataset-out / leave-stratum-out stress；
5. parallel smoke panels；
6. diagnostic paired replay scout，但不能影响 official threshold；
7. runtime-native payload constraints；
8. certificate generator 的 ablation；
9. AP0 negative control。
```

---

# 6. 核心假设

## H1：AP0 不是 value-absent，而是 certificate-absent

H1 认为 AP0 的 weak/strong CP oracle frontier 存在，但 AP0 没有 legal certificate，因此无法部署。

H1 成立标准：

```text
AP0 weak_CP_coverage >= 0.03；
AP0 strong_CP_coverage >= 0.03 或接近 gate；
AP0 legal_capacity_auc_CP < 0.68；
AP0 AEF/probe_auc_CP <= 0.55；
AP0 certificate_primitive_smoke_pass = 0。
```

H1 失败标准：

```text
发现 AP0 其实有 legal feature/controller pass，只是 v9.3.6 搜索不完整。
```

若 H1 失败，可以回 AP0 controller；但基于 v9.3.6 当前数据，H1 大概率成立。

## H2：horizon robustness 是 AP0 的独立 blocker

H2 认为 weak CP 不能代表 robust functional advantage。v9.3.7 必须把 horizon robust / long-risk 写成主指标。

H2 成立标准：

```text
AP0 weak_CP_coverage >> horizon_robust_CP_coverage；
short_only_CP_action_count 与 long_risk_action_count 非平凡；
AP1/AP2/AP3 中如果只优化 weak CP，也会复现 long-risk。
```

H2 失败标准：

```text
AP1/AP2/AP3 只要 weak CP pass，就自动 horizon robust pass。
```

## H3：certificate-producing primitive 可以把 observability 从 posthoc feature 变成 generation-time property

H3 是 v9.3.7 的核心假设。

H3 成立标准：

```text
至少一个 APk 的 certificate-pass actions 满足：
  accepted_count >= 273 或 coverage >= 0.03；
  weak_CP_precision >= 0.75；
  bad_event_rate <= 0.05；
  null_rate <= 0.15；
  strong_CP_coverage >= 0.03 或 horizon_robust_CP_coverage 明显高于 AP0；
  no dataset-specific branch。
```

H3 强成立标准：

```text
horizon_robust_CP_coverage >= 0.03；
long_risk_rate <= 0.05；
short_only_rate <= 0.10；
certificate calibration ECE <= 0.05；
LDO/LSO diagnostic 不崩。
```

H3 失败标准：

```text
AP1/AP2/AP3 全部不能产生 coverage >=0.03 的 certificate-pass action；
或 certificate-pass actions 仍 weak-only / long-risk；
或 certificate fields 与 realized control outcomes 无稳定关系。
```

## H4：runtime-native payload shape 必须进入 primitive 设计，而不是最后补 runtime

H4 认为 AP primitive 如果生成任意 full payload，runtime 可能再次失败。AP primitive 应内生约束 payload shape。

H4 成立标准：

```text
APk selected payload apply q90 <= 0.10 ms；
selected online step_ratio_q90 <= 1.50；
selected memory_ratio <= 1.05；
payload_apply_error_linf_max <= tolerance；
cosine_logged_applied_min >= 0.99999。
```

H4 失败标准：

```text
APk decision pass，但 selected payload apply/runtime 超 envelope。
```

## H5：诊断不同 dataset 是必要的，但不能按 dataset 调参

H5 认为 dataset diagnostics 用于发现 failure mode，不用于 controller dispatch。

H5 成立标准：

```text
route_decision.json 中 dataset_name_used_for_controller = 0；
primitive thresholds shared across datasets；
LDO heldout dataset 不出现 coverage=0；
per-dataset diagnostics 只进入 autopsy，不进入 accept rule。
```

H5 失败标准：

```text
任何 APk / controller / runtime path 使用 dataset_name branch 或 dataset-specific threshold。
```

---

# 7. Certificate 定义

## 7.1 统一 certificate schema

每个 generated action 必须同时落盘：

```text
action_id
primitive_id
certificate_schema_version
candidate_id
event_id
step
seed
dataset_for_diagnostic_only
family_id
bucket_id
horizon_proxy_id
payload_hash
payload_shape_id
payload_apply_mode
```

Certificate fields：

```text
C_V_pass
C_B_pass
C_N_pass
C_S_pass
C_H_pass
C_C_pass
certificate_pass

V_hat
V_lcb
Bad_hat
Bad_ucb
Null_hat
Null_ucb
Support_eff_n
Support_lcb
HorizonRisk_hat
HorizonRisk_ucb
Cost_hat_ms
Cost_ucb_ms
```

Legality audit fields：

```text
uses_dataset_name_for_controller
uses_validation_or_test
uses_future_outcome_for_features
uses_outcome_at_commit
uses_source_measured_gap
uses_formula_proxy_for_official
uses_teacher
uses_loss_modification
uses_loss_backward
```

Pass condition：

```text
uses_dataset_name_for_controller = 0
uses_validation_or_test = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_source_measured_gap = 0
uses_formula_proxy_for_official = 0
uses_teacher = 0
uses_loss_modification = 0
uses_loss_backward = 0
```

## 7.2 Value certificate

Value certificate 用 commit 前可得的一阶或局部可审计量估计 action 是否有正收益。

基础形式：

$$
\widehat{\Delta CE}_{lin}(e)=g_t^T\Delta\theta_{func}(e).
$$

Value pass：

$$
C_V(e)=1
\iff
\widehat{\Delta CE}_{lin}(e) \le -\epsilon_{CE}
\land
\widehat{\Delta Margin}_{p10}(e) \ge \epsilon_M.
$$

记录：

```text
grad_dot_delta
linearized_CE_delta
linearized_margin_p10_delta
predicted_value_score
value_margin
value_lcb
```

## 7.3 Bad-tail certificate

Bad-tail certificate 不是均值 CE 下降，而是 hard-tail 不受损。

定义 hard-tail set：

```text
TailSet = samples with CE in top q or margin in bottom q on current train batch / legal tail memory。
```

Bad-tail linearized effect：

$$
\widehat{\Delta CE}_{tail}(e)=g_{tail}^T\Delta\theta_{func}(e).
$$

Pass：

$$
C_B(e)=1
\iff
\widehat{\Delta CE}_{tail,p90}(e) \le \tau_{tail}
\land
\widehat{\Delta Margin}_{tail,p10}(e) \ge -\tau_M
\land
Conflict_{tail}(e)\le\tau_{conflict}.
$$

记录：

```text
tail_CE_current_p90
tail_margin_current_p10
tail_grad_dot_delta
tail_linearized_CE_delta
tail_linearized_margin_delta
tail_conflict_score
bad_risk_hat
bad_risk_ucb
```

## 7.4 Non-null certificate

Non-null certificate 防止低 risk 但无用。

Pass：

$$
C_N(e)=1
\iff
|\widehat{\Delta CE}_{lin}(e)|\ge\epsilon_{nonzero}
\lor
|\widehat{\Delta Margin}_{p10}(e)|\ge\epsilon_{nonzero,M}.
$$

并加入 payload lower bound：

$$
\|\Delta\theta_{func}\|_2 \ge r_{min}\|\Delta\theta_{AdamW}\|_2.
$$

记录：

```text
payload_norm
payload_linf
payload_norm_over_adamw
predicted_nonnull_score
null_risk_hat
null_risk_ucb
```

## 7.5 Support certificate

Support 不应再只是 family count。它必须是 certificate family 的 empirical reliability。

定义 certificate bucket：

$$
Bucket(e)=hash(primitive\_id, certificate\_signs, payload\_shape, horizon\_proxy, family\_coarse).
$$

Support pass：

$$
C_S(e)=1
\iff
n_{eff}(Bucket(e))\ge n_{min}
\land
LCB(CP\mid Bucket(e))\ge \tau_{CP}
\land
UCB(LongRisk\mid Bucket(e))\le\tau_{LR}.
$$

记录：

```text
certificate_bucket_id
support_count
support_eff_n
bucket_CP_mean
bucket_CP_lcb
bucket_longrisk_mean
bucket_longrisk_ucb
bucket_shortonly_mean
bucket_shortonly_ucb
```

## 7.6 Horizon robustness certificate

Horizon certificate 不能用未来 outcome，但可以使用 commit 前 multi-scale proxies 与 calibration reliability。

Proxy features：

```text
short_proxy_value
mid_proxy_value
long_proxy_risk
tail_memory_conflict
adamw_conflict_drift
margin_tail_sensitivity
```

Pass：

$$
C_H(e)=1
\iff
HorizonRiskUCB(e)\le \tau_H
\land
ShortOnlyUCB(e)\le \tau_{SO}
\land
LongRiskUCB(e)\le \tau_{LR}.
$$

记录：

```text
horizon_proxy_short_score
horizon_proxy_mid_score
horizon_proxy_long_risk
shortonly_risk_ucb
longrisk_risk_ucb
horizon_robust_score
horizon_cert_pass
```

## 7.7 Cost certificate

Cost certificate 必须与 payload shape 绑定。

Pass：

$$
C_C(e)=1
\iff
CostUCB_{apply}(e)\le\tau_{apply}
\land
CostUCB_{feature}(e)\le\tau_{feature}
\land
PayloadShape(e)\in AllowedShapes.
$$

AllowedShapes：

```text
last_edge_only
last_layer_only_diagnostic_for_smoke
block_sparse_edge
low_rank_edge_update
rolewise_sparse_edge_update
```

记录：

```text
payload_shape_id
payload_nonzero_count
payload_block_count
payload_apply_mode
estimated_apply_time_ms
estimated_feature_time_ms
measured_apply_time_ms_q90
measured_step_ratio_q90
memory_ratio
```

---

# 8. AP primitive candidates

## AP0：Current opaque reference

AP0 作为 negative control。它不应作为主线继续 patch。

记录：

```text
AP0 weak_CP_coverage
AP0 strong_CP_coverage
AP0 horizon_robust_CP_coverage
AP0 legal_capacity_auc
AP0 microprobe_auc
AP0 runtime_reference
```

## AP1：LastEdge Linearized Tail-Safe Certificate Primitive

目标：利用 v9.3.6 RT3 的 runtime 线索，限制 update 到最后 edge / output edge block，使 payload apply 快，并让 value/tail certificate 可计算。

生成形式：

$$
\Delta\theta_{AP1}=Proj_{LastEdge}(u),
$$

其中 $u$ 来自当前 batch CE gradient 与 margin tail gradient 的 constrained combination：

$$
u = -\alpha g_{CE} + \beta g_{margin,tail} - \gamma g_{badtail}.
$$

约束：

$$
g_{CE}^T\Delta\theta \le -\epsilon_{CE},
$$

$$
g_{tail}^T\Delta\theta \le \tau_{tail},
$$

$$
\cos(\Delta\theta,\Delta\theta_{AdamW})\ge\rho_{min},
$$

$$
\|\Delta\theta\|_2\le r_{max}\|\Delta\theta_{AdamW}\|_2.
$$

AP1 适合先验证 runtime 与 certificate 是否能同时成立。

## AP2：AdamW-Residual Orthogonal Benefit Certificate Primitive

目标：functional update 不能只是 AdamW 的缩放版，也不能和 AdamW 冲突。AP2 生成 AdamW residual direction：

$$
\Delta\theta_{res}=\Delta\theta_{func}-Proj_{AdamW}(\Delta\theta_{func}).
$$

要求 residual 有额外 margin/tail value：

$$
\widehat{\Delta Margin}_{res,p10}>\epsilon_{res},
$$

且不损害 CE：

$$
\widehat{\Delta CE}_{res}\le\tau_{res}.
$$

AP2 用来测试 functional update 是否有非 AdamW 的可证明增益。

## AP3：Horizon-Robust Tail-Memory Certificate Primitive

目标：直接针对 AP0 的 short-only / long-risk 问题。AP3 生成 action 前必须通过 tail-memory 与 horizon proxy。

定义 legal tail memory：

```text
最近 K 个 train-stream hard-tail batch statistics；
不含 validation/test；
不含 future outcome；
只保存 aggregate gradient/logit/margin statistics。
```

Pass：

$$
C_H(e)=1
\iff
HorizonRiskUCB_{tailmemory}(e)\le\tau_H
\land
LongRiskUCB(e)\le\tau_{LR}.
$$

AP3 的目标不是最大 weak CP，而是减少 long-risk 与提高 horizon robust density。

## AP4：Low-Rank Edge Certificate Primitive

目标：避免 AP1 过于 last-layer，保留 KAN edge-function 特性，同时保持 runtime-native payload。

形式：

$$
\Delta W \approx a b^T,
$$

或 block-low-rank edge update：

$$
\Delta\theta_{edge}^{(block)}=U_r V_r^T.
$$

约束：

```text
rank <= r_max
block_count <= b_max
payload_apply_time_q90 <= threshold
certificate fields complete
```

AP4 用来测试是否可以在更 KAN-native 的 edge basis 上获得 certificate-pass frontier。

## AP5：NoOp / AdamWParallel / ShuffledPayload negative controls

必须保留 controls：

```text
AP5-NoOp
AP5-AdamWParallelOnly
AP5-RandomPayloadSameNorm
AP5-ShuffledCertificate
AP5-ShuffledPayloadWithinFamily
AP5-InvertedTailCertificate
```

这些 controls 必须失败，否则 certificate 不是 functional mechanism。

---

# 9. v9.3.7 实验阶段

---

## P0：v9.3.6 boundary independent reanalysis

### 目标

复现 v9.3.6 结果，重新计算 AP0 的 frontier / identifiability / horizon fragility，不直接接受 route 文本。

### 假设

H1/H2：AP0 有 weak/strong oracle density，但 legal identifiability 和 horizon robustness 不足。

### 必须记录

```text
source_run_id
source_artifact_hash
route
candidate_count
action_count
event_count
weak_CP_row_count
weak_CP_action_count
weak_CP_coverage
weak_CP_coverage_lcb
weak_CP_V_ctrl_lcb
strong_CP_row_count
strong_CP_action_count
strong_CP_coverage
horizon_robust_CP_action_count
horizon_robust_CP_coverage
short_only_CP_action_count
long_risk_action_count
best_legal_capacity_feature
best_legal_capacity_auc_CP
best_top273_CP_precision
best_aef_feature
best_aef_auc_CP
best_aef_top273_CP_precision
best_aef_cost_q90
best_microprobe_auc_CP
best_microprobe_cost_q90
certificate_primitive_smoke_pass
payload_apply_time_ms_q90_microbench
step_ratio_q90_microbench
system_legal_controller_pass
```

### 判断标准

P0 pass：

```text
v9.3.6 route reproduced；
AP0 weak frontier exists；
AP0 horizon-robust coverage < 0.03；
legal observability capacity pass = 0；
AEF / microprobe pass = 0；
certificate primitive missing = 1。
```

### 可视化

```text
p0_ap0_cp_decomposition_bar.svg
p0_weak_strong_horizon_coverage.svg
p0_shortonly_longrisk_counts.svg
p0_ap0_observability_auc_table.svg
p0_runtime_microbench_vs_official_boundary.svg
```

---

## P1：AP0 impossibility and threshold-search ban audit

### 目标

证明 v9.3.7 不应继续 AP0 threshold search。P1 要给出定量停止证据。

### 假设

AP0 当前 feature sigma-algebra 不足以形成 deployable controller。

### 必须记录

```text
feature_id
feature_group
AUC_CP
AUC_StrongCP
AUC_HorizonRobustCP
AUC_LongRisk
AUC_ShortOnly
PR_AUC_CP
Top273_CP_precision
Top273_StrongCP_precision
Top273_HorizonRobustCP_precision
Top273_LongRisk_rate
cost_q90_ms
feature_missing_rate
leave_dataset_auc_drop
leave_stratum_auc_drop
```

### 判断标准

P1 AP0 continuation allowed only if：

```text
Top273_CP_precision >= 0.60
and AUC_CP >= 0.70
and cost_q90 <= 0.20 ms
and leave_dataset_auc_drop <= 0.07
```

P1 AP0 stop condition：

```text
no AP0 feature satisfies continuation gate。
```

若 P1 stop condition 成立，所有 AP0 controller threshold search 必须标记为：

```text
not_run_ap0_observability_blocked
```

### 可视化

```text
p1_ap0_feature_capacity_pareto.svg
p1_top273_precision_by_feature.svg
p1_cp_vs_longrisk_feature_phase.svg
p1_feature_cost_vs_auc.svg
```

---

## P2：certificate schema implementation and legality contract

### 目标

实现统一 certificate schema，保证所有 AP1/AP2/AP3/AP4 action 都能落盘 certificate fields 与 legality audit。

### 假设

没有 certificate schema，就无法区分 primitive success、controller success 和 runtime success。

### 必须记录

```text
certificate_schema_version
primitive_id
action_id
certificate_fields_complete
C_V_pass
C_B_pass
C_N_pass
C_S_pass
C_H_pass
C_C_pass
certificate_pass
legality_audit_pass
cost_fields_complete
payload_fields_complete
```

### 判断标准

P2 pass：

```text
certificate_fields_complete = 1 for all AP-generated actions；
legality_audit_pass = 1；
payload_hash_missing_count = 0；
certificate_schema_version fixed before outcome evaluation；
no outcome labels used at commit time。
```

### 可视化

```text
p2_certificate_schema_completeness.svg
p2_certificate_pass_components.svg
p2_legality_audit_dashboard.svg
```

---

## P3：parallel AP primitive smoke generation

### 目标

并行实现 AP1/AP2/AP3/AP4，并在同一 frozen train-stream 上生成 actions、certificates、payloads。P3 不评价 full outcome，只检查 action generation 是否真实、合法、便宜。

### 执行设置

```text
primitive_ids = AP1, AP2, AP3, AP4, AP5-controls
seeds = 0,1,2 for smoke
sample_actions_per_primitive = 512 minimum
stratified by dataset / seed / family / horizon proxy / bucket
```

### 必须记录

```text
primitive_id
generated_action_count
certificate_pass_count
certificate_pass_rate
payload_hash_missing_count
payload_shape_id_distribution
payload_nonzero_count_mean
payload_nonzero_count_q90
payload_norm_mean
payload_norm_q90
payload_norm_over_adamw_mean
certificate_generation_time_ms_q90
payload_materialize_time_ms_q90
payload_apply_time_ms_q90_smoke
memory_ratio_smoke
legality_audit_pass
```

### 判断标准

P3 primitive generation pass：

```text
generated_action_count >= 512 per primitive；
certificate_fields_complete = 1；
certificate_pass_count > 0；
legality_audit_pass = 1；
payload_apply_error_linf_max <= tolerance；
payload_apply_cosine_min >= 0.99999。
```

P3 runtime smoke pass：

```text
payload_apply_time_ms_q90 <= 0.10 ms for AP1/AP4；
certificate_generation_time_ms_q90 <= 0.20 ms；
memory_ratio <= 1.05。
```

### 可视化

```text
p3_generated_actions_by_primitive.svg
p3_certificate_pass_rate.svg
p3_payload_shape_distribution.svg
p3_payload_cost_pareto.svg
```

---

## P4：control outcome materialization for AP smoke panel

### 目标

对 AP1/AP2/AP3/AP4 smoke panel 执行 matched-control outcome materialization，评估 certificate-pass actions 的 weak/strong/horizon-robust CP density。

### Branches

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayloadSameNorm
ShuffledCertificate
```

### Horizons

```text
horizon = 20, 80, 240
optional = 640 for top APs
```

### 必须记录

```text
primitive_id
action_id
certificate_pass
branch
horizon
safe_good_label
bad_event_label
null_event_label
weak_CP_label
strong_CP_label
horizon_robust_CP_label
short_only_label
long_risk_label
V_ctrl
V_ctrl_lcb_component
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

### 判断标准

P4 smoke outcome pass：

```text
branch_completion_rate = 1.0；
horizon_completion_rate = 1.0；
missing_secondary_delta_count = 0；
quality_audit_pass = 1；
no fake/proxy rows。
```

Primitive weak pass：

```text
certificate_pass_accepted_count >= 100 in smoke panel；
weak_CP_precision_certificate_pass >= 0.60；
bad_event_rate_certificate_pass <= 0.10；
null_rate_certificate_pass <= 0.20。
```

Primitive strong pass for promotion：

```text
weak_CP_precision_certificate_pass >= 0.75；
strong_CP_precision_certificate_pass >= 0.60；
long_risk_rate_certificate_pass <= 0.10；
short_only_rate_certificate_pass <= 0.15；
horizon_robust_CP_precision_certificate_pass improves over AP0 by >= 3x。
```

### 可视化

```text
p4_certificate_pass_outcome_composition.svg
p4_weak_strong_horizon_by_primitive.svg
p4_shortonly_longrisk_by_primitive.svg
p4_vctrl_distribution_by_primitive.svg
p4_branch_outcome_matrix.svg
```

---

## P5：certificate calibration and sufficient-statistic audit

### 目标

检查 certificate 是否真的成为 sufficient statistic，而不是另一个 opaque feature。

### 必须记录

```text
primitive_id
certificate_component
predicted_value_bin
realized_CP_rate
realized_StrongCP_rate
realized_HorizonRobustCP_rate
realized_LongRisk_rate
calibration_ECE_CP
calibration_ECE_LongRisk
Brier_CP
Brier_LongRisk
certificate_ablation_id
ablation_CP_precision
ablation_longrisk_rate
monotone_sign_consistency
leave_dataset_drop
leave_stratum_drop
```

### 判断标准

P5 certificate pass：

```text
calibration_ECE_CP <= 0.08 in smoke；
calibration_ECE_LongRisk <= 0.08；
certificate_pass improves CP precision over AP0 best feature by >= 2x；
certificate ablation shows C_V, C_B, C_H each necessary；
monotone_sign_consistency = 1；
no dataset-specific branch。
```

P5 strong pass：

```text
calibration_ECE_CP <= 0.05；
leave_dataset_drop <= 0.07；
leave_stratum_drop <= 0.10；
certificate-pass Top273_CP_precision >= 0.75。
```

### 可视化

```text
p5_certificate_calibration_curve_CP.svg
p5_certificate_calibration_curve_LongRisk.svg
p5_certificate_ablation_waterfall.svg
p5_leaveout_certificate_drop_heatmap.svg
p5_certificate_component_correlation.svg
```

---

## P6：full AP frontier completion for top primitives

### 目标

对 P4/P5 survivor primitives 扩展到 full action universe 或 official-minimum large panel，判断 AP1/AP2/AP3/AP4 是否真正有 enough density。

### 执行条件

进入 P6 的 primitive 必须满足：

```text
P3 generation pass；
P4 primitive weak pass；
P5 certificate pass。
```

### Materialization scale

```text
Tier A: official-minimum panel >= 30% action universe；
Tier B: full generated AP action universe for top-2 primitives；
Tier C: full branch-horizon rows for selected primitive。
```

### 必须记录

```text
primitive_id
action_count_generated
action_count_certificate_pass
row_count_expected
row_count_actual
rows_per_sec
branch_completion_rate
horizon_completion_rate
weak_CP_count
weak_CP_coverage
weak_CP_precision
strong_CP_count
strong_CP_coverage
strong_CP_precision
horizon_robust_CP_count
horizon_robust_CP_coverage
horizon_robust_CP_precision
short_only_count
short_only_rate
long_risk_count
long_risk_rate
V_ctrl_mean
V_ctrl_lcb
support_balance_pass
```

### 判断标准

P6 weak frontier pass：

```text
certificate_pass_coverage >= 0.03；
weak_CP_precision >= 0.75；
bad_event_rate <= 0.05；
null_rate <= 0.15；
V_ctrl_lcb > 0；
support_balance_pass = 1。
```

P6 horizon robust pass：

```text
horizon_robust_CP_coverage >= 0.03
or
horizon_robust_CP_coverage >= 3 * AP0_horizon_robust_CP_coverage and strong_CP_coverage >= 0.03。
```

P6 fail routes：

```text
R6a-ValueDensityFail: weak_CP_coverage < 0.03。
R6b-HorizonRobustFail: weak pass but horizon robust remains too sparse。
R6c-CertificateCalibrationFail: certificate pass but realized precision/long-risk fails。
R6d-SupportCollapse: only one family/stratum carries pass。
```

### 可视化

```text
p6_full_frontier_by_primitive.svg
p6_coverage_precision_longrisk_pareto.svg
p6_horizon_robust_density.svg
p6_support_balance_by_primitive.svg
p6_value_density_phase_diagram.svg
```

---

## P7：certificate controller without threshold patching

### 目标

建立最小 controller。Controller 不做复杂 feature search，只使用 certificate pass 与少量 frozen global thresholds。

### Controller candidates

### C0：AP0 negative control

```text
Accept(e)=AP0_best_previous_score(e) >= threshold.
```

Expected fail。

### C1：CertificatePassOnly

$$
Accept(e)=1 \iff certificate\_pass(e)=1.
$$

### C2：CertificatePass + LongRisk hard gate

$$
Accept(e)=1
\iff
certificate\_pass(e)=1
\land LongRiskUCB(e)\le\tau_{LR}.
$$

### C3：CertificatePass + cost hard gate

$$
Accept(e)=1
\iff
certificate\_pass(e)=1
\land CostUCB(e)\le\tau_C.
$$

### C4：Minimal monotone certificate score

$$
S_{cert}(e)=
V_{LCB}(e)
-Bad_{UCB}(e)
-Null_{UCB}(e)
-LongRisk_{UCB}(e)
+Support_{LCB}(e)
-Cost_{UCB}(e).
$$

Accept：

$$
Accept(e)=1 \iff S_{cert}(e)\ge\tau
\land C_B(e)=1
\land C_H(e)=1
\land C_C(e)=1.
$$

Feature group count limit：

```text
certificate groups <= 6；
no opaque learned score with >6 groups；
all signs monotone；
thresholds frozen on calibration fold。
```

### 必须记录

```text
controller_id
primitive_id
thresholds
calibration_fold
heldout_fold
split_type
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
weak_CP_precision_cal
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_CP_precision_heldout
bad_event_heldout
null_rate_heldout
long_risk_rate_heldout
short_only_rate_heldout
V_ctrl_mean_heldout
V_ctrl_lcb_heldout
precision_lcb
bad_event_ucb
long_risk_ucb
support_balance_pass
dataset_name_used
```

### 判断标准

P7 decision pass：

```text
coverage_heldout in [0.03, 0.15]
weak_CP_precision_heldout >= 0.75
bad_event_heldout <= 0.05
null_rate_heldout <= 0.15
long_risk_rate_heldout <= 0.10
V_ctrl_lcb_heldout > 0
precision_lcb >= 0.75
bad_event_ucb <= 0.05
support_balance_pass = 1
dataset_name_used = 0
```

P7 horizon-robust pass：

```text
horizon_robust_CP_coverage >= 0.03
or
strong_CP_coverage >= 0.03 and long_risk_rate <= 0.05。
```

### 可视化

```text
p7_controller_frontier.svg
p7_certificate_controller_gate_dashboard.svg
p7_calibration_to_heldout_drift.svg
p7_longrisk_shortonly_by_controller.svg
p7_support_balance_sunburst.svg
```

---

## P8：selected primitive online runtime closure

### 目标

对 P7 selected primitive/controller 测量完整 online runtime。不能只测 payload apply microbench。

### Runtime path

必须包括：

```text
base manual forward/backward/update；
certificate generation；
certificate score / accept；
payload materialization；
payload apply；
no-event preservation；
audit outside timed path；
```

### Runtime candidates

```text
RT0-AP0Reference
RT1-AP1LastEdgeFusedApply
RT2-AP2ResidualSparseApply
RT3-AP3TailMemoryCertificateRuntime
RT4-AP4LowRankEdgeFusedApply
RT5-SelectedHybridRuntime
```

### 必须记录

```text
runtime_candidate_id
primitive_id
controller_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
accepted_action_count
candidate_count_in_step
certificate_generation_time_ms_q90
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_materialize_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
active_step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
allocation_count_per_active_step
payload_apply_error_linf_max
payload_apply_cosine_min
accept_disagreement_count
no_event_preservation_pass
```

### 判断标准

P8 selected runtime pass：

```text
runtime_mode = online_sequential_official；
zero_candidate_controller_kernel_count = 0；
zero_candidate_controller_sync_count = 0；
controller_launches_per_active_step_q90 <= 2；
controller_syncs_per_active_step_q90 <= 1；
allocation_count_per_active_step <= 0.05；
payload_apply_error_linf_max <= tolerance；
payload_apply_cosine_min >= 0.99999；
no_event_preservation_pass = 1；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05。
```

Strong runtime pass：

```text
step_ratio_q90 <= 1.20
or
payload_apply_time_ms_q90 <= 0.05 ms with full selected controller。
```

### 可视化

```text
p8_runtime_waterfall_selected.svg
p8_step_ratio_distribution.svg
p8_payload_apply_time_by_shape.svg
p8_certificate_generation_cost.svg
p8_zero_event_semantics_audit.svg
```

---

## P9：system integration gate

### 目标

将 selected AP primitive、certificate controller、selected runtime 合并成 official system candidate。

### 必须记录

```text
system_candidate_id
primitive_id
controller_id
runtime_candidate_id
certificate_schema_version
candidate_count
action_count
accepted_count
coverage_heldout
weak_CP_precision_heldout
strong_CP_precision_heldout
horizon_robust_CP_coverage
bad_event_heldout
null_rate_heldout
long_risk_rate_heldout
short_only_rate_heldout
V_ctrl_lcb_heldout
precision_lcb
bad_event_ucb
support_balance_pass
step_ratio_q90
memory_ratio
payload_binding_pass
certificate_legality_pass
materialized_system_path
diagnostic_derived_from_measured_components
projection_used
source_measured_gap_used
formula_proxy_used
dataset_name_used
official_eligible
system_legal_controller_pass
```

### 判断标准

P9 system pass：

```text
official_eligible = 1；
system_legal_controller_pass = 1；
certificate_legality_pass = 1；
payload_binding_pass = 1；
materialized_system_path = 1；
diagnostic_derived_from_measured_components = 0；
projection_used = 0；
source_measured_gap_used = 0；
formula_proxy_used = 0；
dataset_name_used = 0；
P7 decision pass = 1；
P8 runtime pass = 1。
```

Decision gates：

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
WeakCPPrecision_{heldout}\ge0.75,
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
NullRate_{heldout}\le0.15,
$$

$$
LongRiskRate_{heldout}\le0.10,
$$

$$
V_{ctrl,LCB}>0.
$$

System gates：

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

### 可视化

```text
p9_system_gate_dashboard.svg
p9_quality_cost_frontier.svg
p9_system_failure_matrix.svg
p9_certificate_to_runtime_trace.svg
```

---

## P10：leave-dataset-out / leave-stratum-out official diagnostic

### 目标

验证 selected certificate primitive/controller 是否 dataset-agnostic，而不是 pooled calibration artifact。

### Split

Leave-dataset-out：

```text
calibrate MNIST + Fashion-MNIST, evaluate KMNIST
calibrate MNIST + KMNIST, evaluate Fashion-MNIST
calibrate Fashion-MNIST + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
hold out signal strata / family / horizon proxy bins
```

### 必须记录

```text
split_type
heldout_entity
primitive_id
controller_id
runtime_candidate_id
accepted_count
coverage
weak_CP_precision
strong_CP_precision
horizon_robust_CP_coverage
bad_event_rate
null_rate
long_risk_rate
V_ctrl_lcb
step_ratio_q90
memory_ratio
dataset_name_used
support_balance_pass
```

### 判断标准

LDO pass：

```text
2/3 heldout datasets pass decision gates；
coverage > 0 for all heldout datasets；
no heldout dataset has bad_event_rate > 0.10；
dataset_name_used = 0。
```

LSO pass：

```text
>=70% heldout strata pass task-safe / weak CP diagnostic；
macro bad_event_rate <= 0.05；
macro long_risk_rate <= 0.10。
```

### 可视化

```text
p10_leave_dataset_matrix.svg
p10_leave_stratum_matrix.svg
p10_dataset_diagnostic_no_tuning_audit.svg
p10_horizon_robust_leaveout.svg
```

---

## P11：diagnostic paired replay scout

### 目标

加快机制判断。P11 可以与 P8/P10 并行准备，但结果不能影响 P7/P9 official threshold。

### 设置

```text
primitive/controller = top selected APk from P7
controls = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload, ShuffledCertificate
horizons = 20, 80, 240, optional 640
seeds = 0,1,2
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
weak_CP_precision
strong_CP_precision
horizon_robust_CP
long_risk_rate
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
status = diagnostic_not_official
diagnostic_used_for_controller = 0
```

### 判断标准

P11 promising：

```text
RealFunctional beats AdamWParallel in >=50% macro slices；
RealFunctional beats bestLR in >=50% macro slices；
shuffled payload/certificate controls do not match RealFunctional；
task_safe holds。
```

P11 fail 不判死刑，除非 P9/P10 official system pass 后 P12 仍 fail。

### 可视化

```text
p11_diagnostic_paired_replay_branch_matrix.svg
p11_shuffle_control_matrix.svg
p11_macro_beat_rate.svg
p11_horizon_effect_curve.svg
```

---

## P12：official paired replay

### 前置条件

```text
P9 system pass = 1
P10 LDO/LSO pass = 1
secondary/control outcome ready = 1
selected runtime pass = 1
```

### 目标

证明 selected certificate primitive 的 RealFunctional branch 在 strong controls 下有局部因果优势。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches =
  RealFunctional
  AdamWOnly
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledPayloadSameNorm
  ShuffledCertificate
  ShuffledPrimitiveId
  ShuffledTailMemory
  ShuffledHorizonCertificate
  ShuffledSupportBucket
  FunctionalChannelShuffled
  TailMaskShuffled
  EventRouteShuffled
  InvertedCertificate
```

### 必须记录

```text
primitive_id
controller_id
dataset
seed
horizon
signal_stratum
event_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
task_safe
accepted_count
coverage
bad_event_rate
null_rate
long_risk_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

$$
BeatRate_{macro,Real\ vs\ AdamWParallel}\ge0.60.
$$

$$
BeatRate_{macro,Real\ vs\ bestLR}\ge0.60.
$$

Task safety：

$$
Acc_{Real,slice}\ge Acc_{AdamW,slice}-0.005.
$$

Shuffle controls must fail：

```text
ShuffledPayloadSameNorm fail；
ShuffledCertificate fail；
ShuffledTailMemory fail；
ShuffledHorizonCertificate fail；
InvertedCertificate fail。
```

### 可视化

```text
p12_official_paired_replay_pareto.svg
p12_macro_beat_rate.svg
p12_shuffle_control_failure.svg
p12_signal_stratum_win_matrix.svg
```

---

## P13：short-run / full-run / sample-efficiency / continual robustness

### 前置条件

```text
P12 paired replay pass = 1
```

### 目标

验证 certificate-producing primitive 的 local causal advantage 能否进入连续训练，且不是 LR grid / QuadraticFeatureMLP / shuffled controller 解释。

### 设置

```text
short-run steps = 50, 240, 640
full-run seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
controls = AdamWOnly, AdamWParallel, bestLR, StrongLRGrid, QuadraticFeatureMLP, NoOp, Random, ShuffledCertificate, ShuffledPayload
continual protocol = dataset/task sequence diagnostic, no dataset-specific tuning
```

### 必须记录

```text
dataset
seed
primitive_id
controller_id
steps
final_acc
best_val_acc
test_acc
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
bad_event_rate
null_rate
long_risk_rate
step_ratio_q90
memory_ratio
continual_retained_accuracy
forgetting
backward_transfer
forward_transfer
strong_baseline_beaten
```

### 判断标准

Short/full functional pass：

$$
Acc_{functional}\ge Acc_{AdamW}-0.005.
$$

At least one meaningful advantage：

$$
Acc_{functional}>Acc_{AdamWParallel},
$$

or

$$
ECE_{functional}<ECE_{AdamW},
$$

or

$$
NLL_{functional}<NLL_{AdamW},
$$

or

$$
ValLossAUC_{functional}<ValLossAUC_{AdamW},
$$

or

$$
Forgetting_{functional}<Forgetting_{AdamW}.
$$

Strong baseline pass：

```text
not explained by StrongLRGrid；
not explained by QuadraticFeatureMLP；
not explained by shuffled certificate/payload/controller。
```

### 可视化

```text
p13_short_run_learning_curves.svg
p13_valloss_auc_step_time.svg
p13_time_to_target.svg
p13_full_run_metric_dashboard.svg
p13_continual_forgetting.svg
p13_strong_baseline_comparison.svg
```

---

# 10. 并行执行计划

v9.3.7 必须并行推进，避免单 runner 串行等待。

## Batch A：boundary + schema + AP generation

并行：

```text
A1: P0 v9.3.6 boundary reanalysis
A2: P1 AP0 impossibility audit
A3: P2 certificate schema implementation
A4: P3 AP1/AP2/AP3/AP4 generation smoke
A5: RT smoke for AP payload shapes
```

产物：

```text
p0_v9360_boundary_reanalysis.csv
p1_ap0_impossibility_audit.csv
p2_certificate_schema_contract.csv
p3_ap_primitive_generation_smoke.csv
p3_payload_runtime_smoke.csv
```

## Batch B：AP outcome panel + certificate calibration

P3 smoke pass 后并行：

```text
B1: P4 AP1 control outcome panel
B2: P4 AP2 control outcome panel
B3: P4 AP3 control outcome panel
B4: P4 AP4 control outcome panel
B5: P5 certificate calibration / ablation
```

## Batch C：frontier completion + selected runtime

P4/P5 survivors 后并行：

```text
C1: P6 full AP frontier for top-2 primitives
C2: P7 minimal certificate controller
C3: P8 selected-runtime measurement for top-2 primitives
C4: P11 diagnostic paired replay scout preparation
```

## Batch D：official system

只有 P7/P8 pass 后：

```text
D1: P9 system integration
D2: P10 LDO / LSO
D3: P11 diagnostic paired replay scout if not already run
```

## Batch E：causal / long-run

只有 P9/P10 pass 后：

```text
E1: P12 official paired replay
E2: P13 short-run
E3: P13 full-run / robustness / continual
```

---

# 11. Required artifacts

```text
run_manifest.json
contract_audit_v9370.csv
provenance_audit_v9370.csv
artifact_hashes.csv
route_decision.json
aggregate_decision.json
failure_table.csv

p0_v9360_boundary_reanalysis.csv
p1_ap0_impossibility_audit.csv
p2_certificate_schema_contract.csv
p3_ap_primitive_generation_smoke.csv
p3_payload_runtime_smoke.csv
p4_ap_control_outcome_panel.csv
p4_branch_horizon_completion_trace.csv
p5_certificate_calibration_audit.csv
p5_certificate_ablation_trace.csv
p6_full_ap_frontier_completion.csv
p6_ap_frontier_trace.csv
p7_certificate_controller.csv
p7_controller_frontier_trace.csv
p8_selected_primitive_online_runtime.csv
p8_runtime_component_trace.csv
p9_system_legal_controller_v9370.csv
p9_system_controller_trace.csv
p10_leave_dataset_stratum_out.csv
p10_leaveout_trace.csv
p11_diagnostic_paired_replay_scout.csv
p11_diagnostic_replay_trace.csv
p12_official_paired_replay.csv
p12_official_replay_trace.csv
p13_short_full_sampleeff_continual_robustness.csv
p13_longrun_trace.csv

figures/
```

---

# 12. Failure taxonomy

```text
F1_boundary_reanalysis_fail
F2_AP0_continuation_gate_unexpected_pass
F3_certificate_schema_incomplete
F4_certificate_legality_violation
F5_AP_generation_fail
F6_payload_hash_missing
F7_payload_apply_error
F8_payload_shape_runtime_fail
F9_certificate_generation_too_expensive
F10_AP_value_density_fail
F11_AP_bad_event_fail
F12_AP_null_rate_fail
F13_AP_longrisk_fail
F14_AP_shortonly_fail
F15_horizon_robust_density_fail
F16_certificate_calibration_fail
F17_certificate_ablation_fail
F18_support_collapse
F19_dataset_tuning_detected
F20_controller_threshold_patch_detected
F21_controller_coverage_fail
F22_controller_precision_fail
F23_controller_bad_event_fail
F24_controller_longrisk_fail
F25_selected_runtime_fail
F26_selected_payload_apply_too_slow
F27_no_event_preservation_fail
F28_memory_ratio_fail
F29_system_integration_fail
F30_LDO_fail
F31_LSO_fail
F32_paired_replay_control_equivalent
F33_shuffle_control_pass
F34_short_run_task_drop
F35_full_run_no_advantage
F36_strong_baseline_explains_gain
F37_continual_forgetting_fail
F38_external_not_ready
F39_fake_or_proxy_violation
F40_artifact_missing
```

---

# 13. Route decision

```text
R1-BoundaryReanalyzed:
  v9.3.6 metrics reproduced.

R2-AP0Stopped:
  AP0 feature/probe continuation gate fails; no AP0 controller search.

R3-CertificateSchemaPass:
  certificate schema and legality audit pass.

R4-APPrimitiveGenerationPass:
  AP1/AP2/AP3/AP4 generate legal actions with payloads and certificates.

R5-APSmokeOutcomePass:
  AP smoke panels complete with matched controls and horizons.

R6-CertificateCalibrationPass:
  certificate fields are calibrated and ablations show necessity.

R7-APWeakFrontierPass:
  at least one AP has certificate-pass weak CP frontier.

R8-APHorizonRobustFrontierPass:
  at least one AP improves horizon robust / long-risk metrics enough.

R9-CertificateControllerPass:
  minimal certificate controller passes heldout decision gates.

R10-SelectedRuntimePass:
  selected primitive/controller runtime reaches step_ratio_q90 <= 1.50.

R11-SystemLegalControllerPass:
  selected AP + controller + runtime pass all system gates.

R12-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R13-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R14-PairedReplayPass:
  RealFunctional beats AdamWParallel / bestLR in official paired replay.

R15-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R16-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R17-CertificatePrimitiveFail:
  AP1/AP2/AP3/AP4 cannot produce sufficient certificate-pass frontier.

R18-HorizonRobustPrimitiveFail:
  weak CP pass but horizon robustness / long-risk fails.

R19-RuntimeNativePrimitiveFail:
  certificate primitive decision pass but selected payload runtime fails.

R20-ComputePassButCausalFail:
  system legal but paired replay/control advantage fails.

R21-ExternalReady:
  strict PureKAN functional route passes final external-ready gates.
```

`route_decision.json` must record：

```text
route
base_candidate
source_route_v9360
candidate_count
action_count
event_count

ap0_weak_CP_coverage
ap0_strong_CP_coverage
ap0_horizon_robust_CP_coverage
ap0_best_legal_auc_CP
ap0_best_aef_auc_CP
ap0_continuation_allowed

certificate_schema_pass
certificate_legality_pass
primitive_ids_tested
primitive_survivor_count
selected_primitive_id
selected_controller_id
selected_runtime_id

generated_action_count
certificate_pass_count
certificate_pass_rate
weak_CP_precision_heldout
weak_CP_coverage_heldout
strong_CP_precision_heldout
strong_CP_coverage_heldout
horizon_robust_CP_coverage
short_only_rate
long_risk_rate
bad_event_rate
null_rate
V_ctrl_lcb
support_balance_pass

certificate_ECE_CP
certificate_ECE_LongRisk
certificate_ablation_pass
leave_dataset_drop
leave_stratum_drop

payload_shape_id
payload_apply_time_ms_q90
certificate_generation_time_ms_q90
score_accept_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
no_event_preservation_pass

system_legal_controller_pass
official_eligible
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
robustness_pass
continual_pass
external_ready

uses_dataset_name_for_controller
uses_validation_or_test
uses_future_outcome_for_features
uses_outcome_at_commit
uses_teacher
uses_loss_modification
uses_loss_backward
fake_data_used
proxy_row_used
diagnostic_promoted_to_official

primary_blocker
next_required_implementation
success_v9370_strict_purekan_functional
success_v9370_full_functional
success_v9370_external_ready
```

---

# 14. 停止条件

## Minimum diagnostic success

```text
P0 boundary reproduced；
P1 AP0 continuation banned or justified；
P2 certificate schema implemented；
P3 at least 3 AP primitives generated；
P4 AP smoke outcome panel materialized；
P5 certificate calibration measured；
P8 selected runtime measured for at least one AP；
no fake/proxy/offload/loss/teacher violation。
```

## Certificate primitive success

```text
Minimum diagnostic success
+
at least one AP has:
  certificate_pass_count sufficient for coverage >=0.03 projection；
  weak_CP_precision >=0.75；
  bad_event_rate <=0.05；
  null_rate <=0.15；
  long_risk_rate <=0.10；
  V_ctrl_lcb >0；
  support_balance_pass =1。
```

## Horizon robust success

```text
Certificate primitive success
+
horizon_robust_CP_coverage >=0.03
or
strong_CP_coverage >=0.03 and long_risk_rate <=0.05。
```

## Runtime success

```text
Certificate primitive success
+
selected runtime measured
+
step_ratio_q90 <=1.50
+
memory_ratio <=1.05
+
payload_apply_error_linf_max <= tolerance
+
no_event_preservation_pass =1。
```

## System success

```text
Certificate primitive success
+
Runtime success
+
P7 controller pass
+
P9 official_eligible =1
+
system_legal_controller_pass =1。
```

## Local functional success

```text
System success
+
LDO pass
+
LSO pass
+
official paired replay beats AdamWParallel / bestLR。
```

## Full functional success

```text
Local functional success
+
short-run task-safe mechanism gain
+
full-run / robustness / strong-baseline pass
+
sample-efficiency or continual/anti-forgetting advantage。
```

---

# 15. 最终判断

v9.3.7 的核心不是“再修一个 controller”，而是把 action primitive 从：

```text
opaque action + posthoc guess
```

重构为：

```text
certificate-producing action + minimal certificate controller + runtime-native payload。
```

如果 AP1/AP2/AP3/AP4 中至少一个 primitive 通过 certificate frontier 与 selected runtime，则项目会第一次拥有真正接近 system-legal functional controller 的候选。

如果所有 certificate-producing primitive 都失败，则结论也很有价值：问题不再是 controller observability，而是当前 LQ-t2-h256 / AP family 无法产生足够 dense、horizon-safe、certificate-able 的 functional actions。那时下一步应进入更深层的 base/primitive co-design，而不是继续 AP0 或 AP1 threshold patching。

本轮的科学分叉是：

$$
\boxed{
\text{Do not find better labels for opaque actions. Generate actions that carry their own legal evidence.}
}
$$
