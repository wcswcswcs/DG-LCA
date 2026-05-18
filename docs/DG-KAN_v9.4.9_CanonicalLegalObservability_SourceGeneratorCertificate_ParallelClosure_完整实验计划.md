# DG-KAN v9.4.9 Canonical Legal Observability / Source Generator / Certificate Parallel Closure 完整实验计划

> 本计划基于更新后的 v9.4.8 recovery 结果制定。v9.4.8 已经不再停在 `canonical_outcome_table_incomplete`，而是停在：
>
> ```text
> route = R4-CanonicalSourceFrontierExistsLegalOpaque
> primary_blocker = canonical_legal_observability_failed
> base_candidate = LQ-t2-h256
> success_v9480_strict_purekan_functional = False
> success_v9480_full_functional = False
> success_v9480_external_ready = False
> ```
>
> 本轮 v9.4.9 的目标不是继续调 `PayloadNorm / PayloadLinf / Step / CandidateId / NegLongRiskPayloadNorm` 这类静态特征阈值，也不是在 MNIST / Fashion-MNIST / KMNIST 上打榜。目标是回答一个更本质的问题：
>
> $$
> \boxed{\text{canonical AP0 robust source frontier 已经存在，但它是否能被 legal commit-time 信号识别、生成、证书化，并进入 system-legal runtime？}}
> $$
>
> 公式说明：本文所有公式均使用 `$...$` 或 `$$...$$`，Typora 友好；不使用方括号 display math。

---

# 0. 执行摘要

v9.4.8 recovery 的关键变化是：canonical truth base 终于闭合。第一次 full rebuild 因 `169` 个 action 的 CUDA OOM 停在 `48726 / 51768` rows；recovery run 复用已完成 `2707` 个 action / `48726` rows，对缺失 `169` actions 逐 action 清缓存重跑，最终补齐 `3042` rows，得到完整 canonical table：

```text
action_count_expected / completed = 2876 / 2876
row_count_expected / actual = 51768 / 51768
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
quality_audit_pass = 1
old_table_quarantine_enforced = 1
canonical_replay_preflight_pass = 1
```

这说明 v9.4.8 已经把 v9.4.7 之后最大的 truth-base blocker 推掉。旧 v9.3.5 outcome table 与 canonical runner 的 drift 仍然非常大：

```text
old_new_label_match_rate = 0.5562123319425127
old_new_V_ctrl_abs_diff_max = 10.772544227307662
D8_old_runner_materializer_bug_count = 8628
old_table_official_quarantine_remains = 1
```

因此，v9.4.9 必须完全基于 canonical_v9480 truth base，不能回读旧 v9.3.5 表做 official controller / generator / certificate。

v9.4.8 也给出强 oracle 信号：

```text
canonical weak CP oracle:
  accepted_count = 1279
  coverage = 0.14823829392675011
  precision_control_positive = 1.0
  bad_event_rate = 0.0
  null_rate = 0.0
  V_ctrl_lcb = 0.33908824425448797

canonical horizon-robust oracle:
  horizon_robust_action_count = 120
  horizon_robust_coverage = 0.04172461752433936

best source frontier diagnostic:
  best_panel_id = SRC-ORC-YRobust-K16
  h20_weak_CP = 1.0
  h20_strong_CP = 1.0
  h20_V_ctrl_lcb = 0.8638057411703248
  h80_weak_CP = 0.1875
  h80_V_ctrl_lcb = -1.53319783648607
  h240_weak_CP = 0.75
  h240_V_ctrl_lcb = 0.07955017988543481
  h240_long_risk = 0.0
  Y_robust_count = 16
  Y_robust_rate = 1.0
```

但当前 legal commit-time observability 失败得很明显：

```text
feature_count = 6
best_feature_id = NegLongRiskPayloadNorm
best_AUC_Yrobust = 0.5236967827769714
best_TopK64_Yrobust_precision = 0.046875
best_TopK64_h240_longrisk = 0.546875
legal_observability_pass = 0
```

这不是“表没闭合”了，也不是“AP0 没有好 action”。当前真正的问题是：

$$
\boxed{\text{canonical AP0 有 robust source frontier，但现有 legal static features 几乎看不见它。}}
$$

v9.4.9 因此不应继续做小修小补。它应该并行推进三件事：

```text
1. source frontier anatomy：弄清 YRobust action 到底有什么机制结构；
2. legal effect observability：构造真正的 action × state × gradient × horizon 交互特征；
3. generator/certificate reset：如果 AP0 robust source 是 outcome-only pattern，就直接生成带 effect certificate 的 source action，而不是事后猜 AP0。
```

---

# 1. 独立数据判断

## 1.1 v9.4.8 的进展不是 acc 进展，而是 truth-base + oracle-frontier 进展

v9.4.8 recovery 后，canonical outcome universe 完整闭合。这个结果比上一版 v9.4.8 incomplete 有本质区别：

```text
上一版：
  route = R0-CanonicalOutcomeTableIncomplete
  canonical_rows = 48726 / 51768
  quality_audit_pass = 0
  primary_blocker = canonical_outcome_table_incomplete

更新版：
  route = R4-CanonicalSourceFrontierExistsLegalOpaque
  canonical_rows = 51768 / 51768
  quality_audit_pass = 1
  primary_blocker = canonical_legal_observability_failed
```

这意味着项目已经从“测量系统还没闭合”推进到“真实科学问题重新暴露”。

我不把这轮看成没进度。它清掉了一个非常大的不确定性：旧 v9.3.5 表不可信，而新 canonical 表可信。更重要的是，新 canonical 表证明 AP0 action population 中有 control-positive frontier 和 horizon-robust frontier。现在的问题不是“有没有好 action”，而是“为什么 legal commit-time 选不中好 action”。

## 1.2 current AP0 frontier 的密度足够，不能再说 frontier absent

canonical weak CP oracle：

$$
Coverage_{weakCP}=0.14823829392675011
$$

这接近常见 coverage 上限 `0.15`，并且 precision / bad / null 都是理想状态：

$$
Precision_{CP}=1.0,
\quad
BadEvent=0,
\quad
Null=0,
\quad
LCB(V_{ctrl})=0.33908824425448797.
$$

horizon-robust action coverage：

$$
Coverage_{YRobust}=\frac{120}{2876}=0.04172461752433936.
$$

这已经超过 `0.03` 这类最低 density gate。所以 v9.4.9 不能再写 `source frontier absent`。更准确的判断是：

$$
\boxed{\text{canonical AP0 source/action frontier 存在，而且 density 不低；但它是 oracle-visible，不是 legal-visible。}}
$$

## 1.3 当前 legal feature 的失败不是“差一点”

当前 best feature 是 `NegLongRiskPayloadNorm`：

```text
AUC_Yrobust = 0.5236967827769714
TopK64_Yrobust_precision = 0.046875
TopK64_h240_longrisk = 0.546875
```

base YRobust rate 是：

$$
BaseRate(YRobust)=\frac{120}{2876}=0.0417246.
$$

TopK64 precision `0.046875` 等于 3 / 64，只比 base rate 略高：

$$
Lift_{TopK64}=\frac{0.046875}{0.0417246}\approx1.12.
$$

这几乎没有 selection value。更糟的是 TopK64 的 h240 long-risk 是 `0.546875`，也就是 35 / 64 个 action 长 horizon 风险高。说明 current legal feature 不但看不见 robust source，也没有可靠排除 long-risk。

所以不能继续围绕这 6 个静态特征调阈值。v9.4.9 的 feature 思路必须从：

```text
payload norm / payload linf / step / candidate id / family count
```

转为：

```text
action-effect fingerprint
state × gradient × payload interaction
linearized CE/margin effect
AdamW conflict and complementarity
horizon risk certificate
support-calibrated certificate
```

## 1.4 source frontier summary 有一个需要严肃复核的口径问题

P5 最佳 panel `SRC-ORC-YRobust-K16` 同时记录：

```text
Y_robust_count = 16
Y_robust_rate = 1.0
h20_weak_CP = 1.0
h20_V_ctrl_lcb = 0.8638057411703248
h80_weak_CP = 0.1875
h80_V_ctrl_lcb = -1.53319783648607
h240_weak_CP = 0.75
h240_V_ctrl_lcb = 0.07955017988543481
h240_long_risk = 0.0
```

如果 `YRobust` 的定义是 “all horizons robust control-positive”，那 h80 weak CP = `0.1875` 和 h80 V_ctrl LCB = `-1.533` 看起来不一致。可能的解释有三种：

```text
1. YRobust 是 action-level label，h80 weak_CP 是 row-level / branch-level summary，两者分母不同；
2. YRobust 定义不是 all-horizon weak CP，而是某种 composite support / no-long-risk / selected-horizon criterion；
3. p5 summary 的字段 join 或 aggregation 仍有口径 bug。
```

v9.4.9 必须先做 `YRobustDefinitionAudit`。这不是小细节：如果 label 口径不清，后续 certificate 会学错目标。

## 1.5 有在数据集上训练，但只是 Base-Acc Sentinel

v9.4.8 的 Base-Acc Sentinel 复用 v9.4.7 sentinel：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP_mean_test_acc = 0.09088541666666672
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
base_acc_used_for_controller = 0
```

也就是说，固定 sentinel 设置下：

$$
Acc_{LQ}-Acc_{MatchedMLP}=+0.0908854.
$$

$$
Acc_{LQ}-Acc_{StrongLRGridMLP}=+0.0252604.
$$

这说明 LQ-t2-h256 base 没有 catastrophic fail，而且在这个固定健康检查里比 MatchedMLP 和 StrongLRGridMLP 都高。但这不是 official functional success，因为：

```text
1. Base-Acc Sentinel 没有用于 controller；
2. selected functional controller 没有打开；
3. paired replay 没有打开；
4. short/full functional training 没有打开；
5. acc 绝对值不是打榜级别；
6. 我们目标不是在这些数据集上调榜。
```

因此正确表述是：

$$
\boxed{\text{base 是活的；functional system 还没被证明。}}
$$

---

# 2. 当前卡点

## 2.1 第一卡点：legal observability

当前 primary blocker 是：

```text
canonical_legal_observability_failed
```

这不是 feature threshold 问题，而是 observability object 错了。现有 static features 看的是边际属性：payload norm、payload linf、step、family count、candidate id、NegLongRiskPayloadNorm。它们没有直接表达：

```text
这个 payload 对当前 state 的 CE / margin 有什么线性化影响；
这个 payload 与 AdamW update 是互补还是冲突；
这个 payload 是否会在 hard-tail 样本上破坏 margin；
这个 payload 的作用是否能跨 h20 / h80 / h240 保持；
这个判断在 calibration stratum 中是否有足够支持。
```

因此下一步应构造 action-effect legal features，而不是堆更多 static payload features。

## 2.2 第二卡点：YRobust label definition / aggregation 需要审计

P5 的 `Y_robust_rate = 1.0` 与 h80 weak CP / V_ctrl 字段存在口径张力。v9.4.9 不能跳过这个审计。必须明确：

$$
YRobust(a)=1
$$

到底表示：

```text
A. action 在 h20/h80/h240 三个 horizon 都 control-positive；
B. action 没有 h240 long-risk，且至少一个 horizon value-positive；
C. action 的 source-panel oracle criterion 过某个 composite score；
D. row-level labels 聚合后的 action-level support condition。
```

v9.4.9 必须输出：

```text
YRobust_definition_version
YRobust_action_denominator
YRobust_horizon_required_count
YRobust_branch_required_count
YRobust_control_positive_required
YRobust_longrisk_required
YRobust_support_rule
YRobust_manual_recompute_match_rate
```

如果定义审计不过，不能训练 certificate。

## 2.3 第三卡点：generator / certificate 还没有在 canonical truth 上重开

v9.4.7 修好了 no-transform replay semantics；v9.4.8 重建了 full canonical outcome universe。此前 v9.4.4-v9.4.6 关于 generator destructive 的部分结论，曾受到旧表 / replay semantics 影响。现在需要在 canonical truth 上重跑：

```text
1. no-transform preservation sanity；
2. existing AP0b-AP0q generator preservation；
3. direct objective-solved source generators；
4. effect-valid certificate。
```

但重跑不能只是“看看 AP0d/AP0l 还能不能过”。应该围绕更高层假设：

$$
\boxed{\text{能否生成一个 action，使它的 certificate 本身就是 value / risk / support 的 legal statistic？}}
$$

## 2.4 第四卡点：controller/runtime/downstream 仍没有打开

v9.4.8 的 P7-P10/P13-P14 全部 gate-blocked：

```text
no generator preservation
no effect-valid certificate
no source controller
no selected runtime
no paired replay
no short/full validation
```

所以现在还不能谈 official functional training、LDO/LSO、paired replay 或 full external advantage。v9.4.9 只能在 upstream pass 后有条件打开这些阶段。

---

# 3. 是否还在正确道路上

高层方向仍然是正确的，因为项目没有把以下内容误写成 success：

```text
1. 没有把旧 v9.3.5 table 继续当 official truth；
2. 没有把 oracle diagnostic frontier 当 legal controller；
3. 没有把 Base-Acc Sentinel 当 functional success；
4. 没有按 dataset 调阈值；
5. 没有在 legal observability fail 后强行打开 generator/controller/runtime/downstream。
```

但是执行策略必须提速。v9.4.1 到 v9.4.8 的推进很扎实，但串行感强：

```text
v9.4.1: source generator materialized，但 source outcome rows = 0；
v9.4.2: source outcome materializer 修好，但 source panel / generator value-poor；
v9.4.3: multi-panel source frontier 未找到 survivor；
v9.4.4: 旧表上找到 oracle survivor，但 generator destructive；
v9.4.5: no-transform 也失败，怀疑 identity；
v9.4.6: identity/payload 过，定位 runner drift；
v9.4.7: runner 语义修好，但旧表 survivor 不复现；
v9.4.8: canonical full table 修好，frontier 存在，但 legal observability 失败。
```

这不是原地踏步，但确实慢。v9.4.9 必须改成并行验证，而不是一轮只暴露一个 blocker。具体策略：

```text
A. P0/P1 做 label/definition audit；
B. P2/P3 并行做 oracle anatomy 和 legal feature expansion；
C. P4/P5 并行做 existing generator revalidation 和 new objective generator；
D. P6 并行做 certificate calibration；
E. P8 提前做 runtime cost preflight，不等 controller 过后才发现成本问题；
F. Base-Acc Sentinel 继续隔离跑，但不进入 controller。
```

---

# 4. v9.4.9 总体目标

v9.4.9 的总体目标是：

$$
\boxed{\text{把 canonical oracle-visible robust source frontier 转化为 legal-visible 或 generator-produced frontier。}}
$$

这可以拆成三个互斥结论之一：

```text
Route A: Legal observable pass
  canonical robust AP0 source frontier 能被 legal commit-time action-effect features 识别。

Route B: Generator/certificate pass
  现有 AP0 source 不可见，但新的 source generator 能产生 certificate-valid robust actions。

Route C: Primitive redesign required
  既不能 legal 识别 AP0 robust source，也不能生成 effect-valid robust source；当前 source/action primitive 需要更大重构。
```

强目标：

$$
SystemLegalControllerPass = 1
$$

$$
LegalObservabilityPass = 1
$$

$$
CertificateEffectValidPass = 1
$$

$$
SelectedRuntimeStepRatio_{q90} \le 1.50
$$

最低有效推进目标：

```text
1. YRobust definition audit pass；
2. canonical robust source anatomy 完成；
3. 至少 4 组 action-effect legal feature 被 materialize 并测成本；
4. 至少一个 legal feature group 或 certificate 达到 weak observability gate；
5. existing generator preservation 在 canonical truth 上被重新验证；
6. direct objective-solved generator 至少完成 smoke + h20/h80/h240 outcomes；
7. selected runtime preflight 成本测量完成；
8. 明确 route 到 LegalObservable / GeneratorCertificate / PrimitiveRedesign。
```

---

# 5. 硬约束

v9.4.9 继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload for official rows
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name branch
Base-Acc Sentinel 不进入 selector/generator/certificate/controller
旧 v9.3.5 outcome table 不进入 official gate
oracle labels 不进入 commit-time features
validation/test metric 不进入 controller
future outcome 不进入 feature
```

functional update 仍然是 update rule：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{functional}.
$$

任务损失仍是标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

允许：

```text
1. 使用 canonical_v9480 outcome labels 做 calibration / heldout / diagnostic；
2. 使用 oracle labels 做 upper-bound anatomy，但不能 official selector；
3. 使用 train-batch logits / gradients / margins / hard-tail statistics 做 legal feature；
4. 使用 manual JVP / linearized CE/margin estimates，只要 commit-time 可得且成本记录；
5. 使用 cross-fitted certificate，在 calibration fold 学参数，在 heldout fold 固定评估；
6. 做 dataset diagnostics，但不能 dataset-specific tuning。
```

---

# 6. 核心假设

## H1：canonical AP0 frontier 真实存在

H1 认为 v9.4.8 recovery 后的 canonical AP0 table 证明 frontier 存在，不应再纠缠 table incompleteness。

H1 成立标准：

```text
canonical_full_control_outcome_ready = 1
quality_audit_pass = 1
canonical_control_positive_oracle_pass = 1
canonical_horizon_robust_oracle_pass = 1
old_table_quarantine_enforced = 1
```

H1 失败标准：

```text
manual recompute 发现 canonical table row mismatch / label exclusivity / hash / branch-horizon completion bug。
```

若 H1 失败，v9.4.9 必须停止 generator/certificate，回到 truth-base repair。

## H2：YRobust label definition 可以被一致复算

H2 认为 `YRobust` 是一个可复算、可审计的 action-level label，而不是 artifact join 口径造成的幻觉。

H2 成立标准：

```text
YRobust_manual_recompute_match_rate = 1.0
YRobust_action_count = 120
YRobust_coverage = 0.04172461752433936
YRobust_label_exclusivity_pass = 1
YRobust_horizon_definition_consistency_pass = 1
```

H2 失败标准：

```text
P5 source frontier 的 YRobust 与 h20/h80/h240 row-level labels 无法解释，或 manual recompute mismatch > 0。
```

## H3：现有 six-feature legal observability 不足，但更强的 legal action-effect features 可能有效

H3 认为 current six static features 失败，不等于所有 legal observability 都失败。它只是说明边际静态特征不够。

H3 成立标准：

至少一个新 feature group 达到 weak gate：

```text
AUC_YRobust >= 0.70
TopK64_YRobust_precision >= 0.20
TopK64_h240_longrisk <= 0.20
feature_cost_q90_ms <= 0.25
leave_dataset_auc_drop <= 0.08
leave_stratum_auc_drop <= 0.10
```

strong gate：

```text
AUC_YRobust >= 0.80
TopK64_YRobust_precision >= 0.35
TopK16_YRobust_precision >= 0.50
TopK64_h240_longrisk <= 0.10
feature_cost_q90_ms <= 0.15
```

H3 失败标准：

```text
所有 legal feature groups AUC_YRobust < 0.65 或 TopK64 precision < 0.12。
```

## H4：oracle source frontier 的 anatomy 能解释 legal miss reason

H4 认为 K16/K64 robust source 不是纯随机 outcome-only pattern；应该能从 state、gradient、payload、support、horizon structure 中看到至少部分机制。

H4 成立标准：

```text
oracle_action_anatomy_coverage >= 0.80
at_least_one_legal_feature_distribution_shift_pass = 1
miss_reason_taxonomy_complete = 1
```

H4 失败标准：

```text
YRobust actions 与 non-YRobust actions 在所有 legal anatomy dimensions 上都不可区分。
```

若 H4 失败，应减少 AP0 posthoc selector 研究，转向 certificate-producing source generator。

## H5：existing generator 在 canonical truth 上必须重新验证

H5 认为此前 generator destructive 结论受到旧表与 runner semantics 影响，因此必须用 canonical_v9480 truth base 重跑，而不是沿用旧结论。

H5 成立标准：

```text
no_transform_canonical_preservation_pass = 1
existing_generator_preservation_measured = 1
source_positive_lost_rate_canonical measured
Damage_median_canonical measured
```

H5 失败标准：

```text
no-transform 在 canonical runner 下仍不等价。
```

如果 no-transform 失败，立即停止 generator/certificate，回到 runner semantics。

## H6：direct objective-solved generator 必须优化 robust source objective，而不是弱 CP

H6 认为 weak CP 和 robust source objective 不等价。v9.4.9 generator 目标应是：

$$
\max_{\Delta\theta} \; \widehat V_{h20}(\Delta\theta)+\widehat V_{h80}(\Delta\theta)+\widehat V_{h240}(\Delta\theta)-\lambda \widehat R_{h240}(\Delta\theta)-\mu C(\Delta\theta)
$$

subject to：

$$
\|\Delta\theta\| \le r,
$$

$$
\cos(\Delta\theta,\Delta\theta_{AdamW}) \ge -\rho,
$$

$$
\widehat{\Delta Margin}_{tail,h240} \ge -m.
$$

H6 成立标准：

```text
at least one generator:
  h20_V_ctrl_lcb > 0
  h80_V_ctrl_lcb > -epsilon or weak_CP_h80 >= 0.50
  h240_longrisk <= 0.10
  YRobust_precision_topK >= 0.20
```

H6 strong pass：

```text
YRobust_precision_topK64 >= 0.35
h240_longrisk <= 0.05
support_balance_pass = 1
```

## H7：effect-valid certificate 是 necessary gate

H7 认为没有 effect-valid certificate，就不能 official select。certificate 必须预测 robust value 和 long-risk，不只是 schema 完整。

H7 weak pass：

```text
AUC_YRobust >= 0.70
AUC_LongRisk <= 0.30 when lower score means safer, or >=0.70 after sign normalization
P(YRobust | CertPass) / P(YRobust) >= 3.0
P(LongRisk | CertPass) <= 0.20
monotone_sign_pass = 1
```

H7 strong pass：

```text
AUC_YRobust >= 0.80
P(YRobust | CertPass) >= 0.35
P(LongRisk | CertPass) <= 0.10
calibration_ECE <= 0.05
leaveout_drop <= 0.10
```

## H8：runtime 不应等 controller 选中后才测

H8 认为 v9.4.9 应提前并行做 feature/certificate/payload runtime preflight。否则可能 controller 终于过了，却发现成本不合格。

H8 pass：

```text
feature_compute_q90_ms recorded for every candidate feature group
certificate_compute_q90_ms recorded
payload_apply_q90_ms recorded
selected_path_step_ratio_estimate measured as diagnostic
no_event_preservation_pass = 1
```

Official runtime pass 仍必须等 selected legal controller 后：

$$
StepRatio_{q90}\le1.50,
\quad
MemoryRatio\le1.05.
$$

---

# 7. 数据合同

## 7.1 canonical outcome row contract

每个 row 必须包含：

```text
action_id
candidate_id
event_id
branch_id
horizon
outcome_table_version = canonical_v9480
runner_semantics_version = canonical_branch_name_invariant_v9470_or_later
state_before_hash
optimizer_state_hash
rng_state_hash
batch_sequence_hash
payload_hash
branch_config_hash
horizon_config_hash
label_config_hash
CE_delta
NLL_delta
ECE_delta
margin_delta
V_ctrl
weak_CP
strong_CP
long_risk
YRobust_action_label
row_quality_pass
```

Pass：

```text
row_count_expected = 51768
row_count_actual = 51768
missing_branch_count = 0
missing_horizon_count = 0
missing_secondary_delta_count = 0
metric_nan_count = 0
metric_inf_count = 0
label_exclusivity_violation_count = 0
duplicate_row_id_count = 0
quality_audit_pass = 1
```

## 7.2 legal feature contract

每个 legal feature 必须记录：

```text
feature_id
feature_group
feature_version
candidate_id
action_id
event_id
commit_time_available
uses_dataset_name
uses_validation_or_test
uses_future_outcome
uses_outcome_at_commit
uses_old_table
uses_oracle_label_at_commit
feature_compute_time_ms
feature_memory_delta_mb
feature_value
feature_missing
```

Legal pass：

```text
commit_time_available = 1
uses_dataset_name = 0
uses_validation_or_test = 0
uses_future_outcome = 0
uses_outcome_at_commit = 0
uses_old_table = 0
uses_oracle_label_at_commit = 0
feature_missing_rate <= 0.01
```

## 7.3 generator contract

每个 generated action 必须记录：

```text
generated_action_id
source_action_id
primitive_id
primitive_version
generator_family
payload_hash
certificate_hash
state_before_hash
commit_time_available
uses_dataset_name
uses_validation_or_test
uses_future_outcome
uses_outcome_at_commit
payload_tensor_written
certificate_tensor_written
action_apply_error_linf
action_apply_error_relative
action_apply_cosine
no_transform_equivalence_flag
```

Pass：

```text
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-7
action_apply_cosine_min >= 0.999999
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

## 7.4 certificate contract

Certificate 必须拆成可解释组成：

```text
cert_value_h20_lcb
cert_value_h80_lcb
cert_value_h240_lcb
cert_longrisk_h240_ucb
cert_margin_tail_guard
cert_adamw_conflict_guard
cert_support_lcb
cert_payload_norm_guard
cert_cost_estimate
cert_score_total
cert_pass
```

公式：

$$
CertScore(a)=w_1LCB(\widehat V_{h20})+w_2LCB(\widehat V_{h80})+w_3LCB(\widehat V_{h240})-w_4UCB(\widehat R_{h240})+w_5LCB(Support)-w_6Cost.
$$

约束：

```text
w_i >= 0
feature_count <= 8
thresholds frozen on calibration fold
no dataset-specific threshold
```

## 7.5 runtime contract

每个 timed step 记录：

```text
step_id
candidate_count_in_step
active_step_flag
zero_candidate_step_flag
feature_compute_time_ms
certificate_compute_time_ms
score_accept_time_ms
payload_lookup_time_ms
payload_apply_time_ms
base_train_step_time_ms
total_step_time_ms
step_ratio
memory_ratio
controller_kernel_launch_count
controller_sync_count
audit_outside_timed_path
no_event_preservation_pass
base_adamw_equivalence_on_zero_candidate_steps
event_scheduler_batch_order_preserved
```

Official runtime pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_controller_launch_count = 0
no_event_preservation_pass = 1
audit_outside_timed_path = 1
```

---

# 8. v9.4.9 实验阶段

---

## P0：v9.4.8 recovery boundary reproduction

### 目标

确认 v9.4.8 recovery 后的真实 boundary 是 `R4-CanonicalSourceFrontierExistsLegalOpaque`，不是上一版 `R0-CanonicalOutcomeTableIncomplete`。

### 假设

H1：canonical full outcome table 已闭合，legal observability 是当前 primary blocker。

### 执行

读取：

```text
results/real_rerun_20260506/v9480_canonical_outcome_universe_rebuild_source_frontier_revalidation_recovery_20260514T160000Z/
```

不得使用 first run incomplete artifact 作为 official，只可用于 P0 recovery comparison。

### 必须记录

```text
source_artifact_id
route_v9480
canonical_full_control_outcome_ready
canonical_row_count_expected
canonical_row_count_actual
quality_audit_pass
old_table_quarantine_enforced
canonical_replay_preflight_pass
canonical_control_positive_oracle_pass
canonical_horizon_robust_oracle_pass
canonical_source_frontier_pass
legal_observability_pass
system_legal_controller_pass
primary_blocker
```

### 判断标准

P0 pass：

```text
route_v9480 = R4-CanonicalSourceFrontierExistsLegalOpaque
canonical_row_count_expected = 51768
canonical_row_count_actual = 51768
quality_audit_pass = 1
old_table_quarantine_enforced = 1
legal_observability_pass = 0
primary_blocker = canonical_legal_observability_failed
```

### 可视化

```text
p0_v9480_first_vs_recovery_rows.svg
p0_recovery_boundary_ladder.svg
p0_truth_base_status_matrix.svg
```

---

## P1：YRobust definition and source frontier consistency audit

### 目标

解决 `SRC-ORC-YRobust-K16` 的 label/summary 口径问题，确认 YRobust 是可复算的 action-level label。

### 假设

H2：YRobust 可以从 canonical outcome table 通过明确规则复算。

### 执行

对所有 `2876` actions 重新计算：

```text
weak_CP_h20
weak_CP_h80
weak_CP_h240
strong_CP_h20
strong_CP_h80
strong_CP_h240
V_ctrl_lcb_h20
V_ctrl_lcb_h80
V_ctrl_lcb_h240
longrisk_h240
YRobust_v1
YRobust_v2
YRobust_v3
```

三个候选定义：

```text
YRobust_v1:
  all horizons weak_CP = 1 and h240 longrisk = 0

YRobust_v2:
  h20 V_ctrl LCB > 0, h240 longrisk = 0, support_balance = 1

YRobust_v3:
  composite robust score top action-level set with h20/h240 positive and h80 not catastrophic
```

### 必须记录

```text
action_id
YRobust_original
YRobust_recomputed_v1
YRobust_recomputed_v2
YRobust_recomputed_v3
h20_weak_CP
h80_weak_CP
h240_weak_CP
h20_V_ctrl_lcb
h80_V_ctrl_lcb
h240_V_ctrl_lcb
h240_longrisk
manual_recompute_match
mismatch_reason
```

### 判断标准

P1 pass：

```text
YRobust_manual_recompute_match_rate = 1.0
YRobust_definition_consistency_pass = 1
mismatch_count = 0
```

若 mismatch 存在：

```text
YRobust_mismatch_count > 0
=> generator/certificate/controller 不得打开
=> route = R1-YRobustDefinitionMismatch
```

### 可视化

```text
p1_yrobust_definition_confusion_matrix.svg
p1_horizon_label_consistency_heatmap.svg
p1_source_panel_yrobust_waterfall.svg
p1_vctrl_by_horizon_for_yrobust.svg
```

---

## P2：canonical robust source anatomy

### 目标

分析 120 个 YRobust actions 和 K16/K64 oracle source panels 到底在什么维度上与非 robust actions 不同。不是为了用 oracle official，而是为了找机制。

### 假设

H4：oracle survivor 不是完全 outcome-only pattern；至少部分差异能被 legal anatomy 解释。

### 执行

比较以下 action groups：

```text
G0: all canonical actions, N = 2876
G1: YRobust actions, N = 120
G2: SRC-ORC-YRobust-K16, N = 16
G3: top64 oracle robust panel, N = 64
G4: high weak CP but long-risk actions
G5: high h20 value but h80/h240 fail actions
G6: random matched controls by family/step/payload norm
```

### 必须记录

Legal anatomy dimensions：

```text
payload_norm
payload_linf
payload_role_entropy
payload_tail_selectivity
step_bucket
family_id
candidate_origin_tag
stable_score_q
score_margin
state_NLL
state_CEp99
state_margin_p10
hard_tail_fraction
functional_norm_over_adamw_norm
cos_functional_adamw
cos_functional_negative_grad
rolewise_gradient_alignment
rolewise_conflict_score
linearized_CE_delta_trainbatch
linearized_margin_delta_hardtail
linearized_tail_risk_score
support_count
support_lcb
family_support_count
family_reliability_lcb
```

### 判断标准

P2 pass：

```text
anatomy_rows_complete = 1
at least one legal dimension has |effect_size| >= 0.50 or AUC >= 0.65
miss_reason_taxonomy_complete = 1
```

若所有 legal dimensions AUC < 0.60：

```text
oracle_outcome_only_pattern_likely = 1
```

### 可视化

```text
p2_yrobust_vs_nonrobust_feature_effect_size.svg
p2_oracle_k16_feature_radar.svg
p2_family_step_distribution_yrobust.svg
p2_payload_geometry_yrobust_scatter.svg
p2_vctrl_horizon_curve_yrobust_vs_controls.svg
p2_miss_reason_sankey.svg
```

---

## P3：legal action-effect feature factory v2

### 目标

构造真正 legal 的 action-effect features，不再围绕静态 payload norm 小修。

### 假设

H3：当前 six-feature fail 不代表 legal observability fail；更强的 action-effect features 可能识别 YRobust。

### Feature groups

#### G1：Gradient-action interaction

记录：

```text
g_dot_delta
g_dot_delta_rolewise
cos_delta_grad
cos_delta_adamw
cos_delta_adamw_rolewise
norm_delta_over_adamw
conflict_fraction_rolewise
```

核心公式：

$$
DescentProxy(a)=-g_t^T\Delta\theta_a.
$$

#### G2：Linearized train-batch CE / margin effect

使用当前 train batch / hard-tail subset，不使用 future outcome：

$$
\widehat{\Delta CE}(a) \approx \nabla_\theta CE_t^T\Delta\theta_a.
$$

$$
\widehat{\Delta Margin}_{tail}(a) \approx J_{margin,tail}\Delta\theta_a.
$$

记录：

```text
linearized_CE_delta_mean
linearized_CE_delta_p90
linearized_margin_delta_p10
linearized_hardtail_margin_delta_p10
linearized_wrong_conf_delta_p90
```

#### G3：AdamW complementarity / conflict

记录：

```text
adamw_parallel_alignment
adamw_residual_norm
functional_complementarity_score
adamw_conflict_guard
orthogonal_benefit_score
```

公式：

$$
Complement(a)=\|Proj_{\Delta\theta_{AdamW}^{\perp}}(\Delta\theta_a)\|.
$$

#### G4：Horizon-risk proxy from stability / tail state

记录：

```text
state_tail_CE_p99
state_margin_p10
state_NLL
margin_volatility_proxy
gradient_noise_tail_norm
payload_tail_overlap
horizon_risk_proxy
```

#### G5：Support-calibrated reliability

只用 calibration fold 统计：

```text
support_count
support_lcb
family_reliability_lcb
stratum_reliability_lcb
empirical_bayes_yrobust_lcb
empirical_bayes_longrisk_ucb
```

#### G6：Cheap synthetic micro-response

允许使用当前 train batch 做低成本 micro-response，但必须记录成本，不得用 validation/test/future labels：

```text
micro_response_CE_delta_current_batch
micro_response_margin_delta_hardtail
micro_response_entropy_delta
micro_response_cost_ms
```

### 必须记录

```text
feature_id
feature_group
action_id
feature_value
feature_missing
feature_compute_q90_ms
memory_delta_mb
commit_time_available
uses_dataset_name
uses_future_outcome
uses_outcome_at_commit
uses_old_table
AUC_YRobust
AUC_LongRisk
TopK16_YRobust_precision
TopK64_YRobust_precision
TopK64_longrisk
leave_dataset_auc_drop
leave_stratum_auc_drop
```

### 判断标准

P3 weak pass：

```text
AUC_YRobust >= 0.70
TopK64_YRobust_precision >= 0.20
TopK64_h240_longrisk <= 0.20
feature_compute_q90_ms <= 0.25
```

P3 strong pass：

```text
AUC_YRobust >= 0.80
TopK16_YRobust_precision >= 0.50
TopK64_YRobust_precision >= 0.35
TopK64_h240_longrisk <= 0.10
feature_compute_q90_ms <= 0.15
```

### 可视化

```text
p3_feature_auc_bar_yrobust_longrisk.svg
p3_topk_precision_curve.svg
p3_pr_curve_yrobust.svg
p3_feature_cost_vs_auc.svg
p3_leaveout_auc_drop.svg
p3_feature_ablation_matrix.svg
```

---

## P4：canonical existing generator preservation revalidation

### 目标

在 canonical_v9480 truth base 下重新验证 AP0b-AP0q / AP0r-AP0w / AP0l-AP0q 的 generator preservation。之前旧表和 replay semantics 影响过结论，现在必须重测。

### 假设

H5：no-transform sanity pass 后，才能判断 generator 是否 destructive。

### 输入 panels

```text
S0: SRC-ORC-YRobust-K16 diagnostic only
S1: Top64 canonical YRobust oracle diagnostic
S2: Top64 legal feature P3 best if P3 weak pass
S3: random matched control panel
S4: high weak CP but long-risk panel
S5: representative stratified panel
```

### Generator families

```text
G0-NoTransformCanonicalReplay
G1-AP0b_to_AP0f_existing
G2-AP0g_to_AP0q_direct_existing
G3-AP0r_to_AP0w_objective_existing
```

### 必须记录

```text
source_panel_id
generator_id
source_action_count
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
no_transform_equivalence_pass
action_apply_error_linf_max
source_YRobust_count
generated_YRobust_count
source_positive_lost_rate
source_negative_fixed_rate
Damage_mean
Damage_median
h20_weak_CP
h80_weak_CP
h240_weak_CP
h20_V_ctrl_lcb
h80_V_ctrl_lcb
h240_V_ctrl_lcb
h240_longrisk
generator_preserve_pass
generator_improve_pass
```

### 判断标准

P4 sanity pass：

```text
G0-NoTransformCanonicalReplay:
  metric_abs_diff_max = 0
  label_match_rate = 1.0
  horizon_state_hash_match_rate = 1.0
```

P4 preservation pass：

```text
source_positive_lost_rate <= 0.20
Damage_median >= -0.05
h240_longrisk <= 0.10
```

P4 improve pass：

```text
generated_YRobust_rate > source_YRobust_rate
h20_V_ctrl_lcb > 0
h240_longrisk <= 0.10
```

### 可视化

```text
p4_source_to_generated_damage_heatmap.svg
p4_no_transform_equivalence_grid.svg
p4_generator_preservation_by_panel.svg
p4_vctrl_horizon_before_after.svg
p4_longrisk_before_after.svg
```

---

## P5：direct robust source generator v2

### 目标

如果 legal selector 看不见 AP0 robust source，直接生成带 certificate 的 robust source action。不能再生成“合法但无效”的 payload。

### 假设

H6：objective-solved generator 可以产生 h20 value-positive 且 h240 safe 的 action。

### Generator candidates

#### VG1：Constrained Linearized Source Solver

优化：

$$
\max_{\Delta\theta} -g_t^T\Delta\theta - \lambda R_{tail}(\Delta\theta) - \mu\|\Delta\theta\|^2.
$$

约束：

$$
\|\Delta\theta\|\le r,
\quad
\cos(\Delta\theta,\Delta\theta_{AdamW})\ge -\rho.
$$

#### VG2：Horizon-Guarded Residual Blend

构造：

$$
\Delta\theta = \alpha\Delta\theta_{functional} + (1-\alpha)Proj_{safe}(\Delta\theta_{AdamW}).
$$

#### VG3：Tail-Margin Conservative Source

目标：

$$
\widehat{\Delta Margin}_{tail,p10} \ge 0
$$

且最大化 CE descent proxy。

#### VG4：Role-Sparse Edge Update

只在 KAN edge / last-edge / low-rank role 上生成 sparse payload，避免全局扰动。

#### VG5：Support-Memory Source

只在 calibration support 高的 family/stratum 内激活，source certificate 内置 support LCB。

#### VG6：YRobust-Anatomy-Informed Generator

使用 P2 anatomy 得到的 legal mechanism，不使用 oracle label at commit；例如若 YRobust 与某类 gradient-action alignment / hard-tail margin proxy 强相关，则直接优化该 proxy。

### 必须记录

```text
generator_id
generator_version
action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
certificate_fields_complete
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_outcome
branch_horizon_rows_actual
h20_weak_CP
h80_weak_CP
h240_weak_CP
h20_V_ctrl_lcb
h80_V_ctrl_lcb
h240_V_ctrl_lcb
h240_longrisk
YRobust_precision
certificate_pass_rate
generator_cost_ms
```

### 判断标准

P5 weak pass：

```text
h20_V_ctrl_lcb > 0
h240_longrisk <= 0.15
YRobust_precision >= 0.20
support_balance_pass = 1
```

P5 strong pass：

```text
h20_V_ctrl_lcb > 0
h80_V_ctrl_lcb > -0.05
h240_V_ctrl_lcb > 0
h240_longrisk <= 0.05
YRobust_precision >= 0.35
```

### 可视化

```text
p5_generator_frontier_scatter_vctrl_longrisk.svg
p5_generator_yrobust_precision_by_primitive.svg
p5_generator_cost_vs_value.svg
p5_horizon_value_curves_by_generator.svg
p5_certificate_pass_vs_outcome.svg
```

---

## P6：effect-valid certificate v2

### 目标

训练 / 校准一个 legal certificate，使它能预测 YRobust 和 long-risk，而不是只通过 schema audit。

### 假设

H7：effect-valid certificate 是 selected controller 的必要条件。

### Certificate candidates

```text
CERT9-ActionEffectLinearizedCertificate
CERT10-AdamWComplementarityCertificate
CERT11-HorizonTailGuardCertificate
CERT12-SupportCalibratedYRobustCertificate
CERT13-MinimalMonotoneCompositeCertificate
```

Composite formula：

$$
CertScore(a)=w_1LCB(\widehat V_{h20})+w_2LCB(\widehat V_{h80})+w_3LCB(\widehat V_{h240})-w_4UCB(\widehat R_{h240})+w_5LCB(Support)-w_6Cost.
$$

### Cross-fitting protocol

```text
folds = seed-fold × dataset-fold × stratum-fold
thresholds frozen on calibration fold
report macro average
no dataset-specific threshold
```

### 必须记录

```text
certificate_id
feature_groups_used
feature_count
w_i
thresholds
AUC_YRobust
AUC_LongRisk
PR_AUC_YRobust
ECE_YRobust
TopK16_YRobust_precision
TopK64_YRobust_precision
TopK64_longrisk
P_YRobust_given_cert_pass
P_LongRisk_given_cert_pass
leave_dataset_auc_drop
leave_stratum_auc_drop
certificate_compute_q90_ms
certificate_memory_delta_mb
```

### 判断标准

P6 weak pass：

```text
AUC_YRobust >= 0.70
TopK64_YRobust_precision >= 0.20
P_LongRisk_given_cert_pass <= 0.20
ECE_YRobust <= 0.08
```

P6 strong pass：

```text
AUC_YRobust >= 0.80
TopK16_YRobust_precision >= 0.50
TopK64_YRobust_precision >= 0.35
P_LongRisk_given_cert_pass <= 0.10
ECE_YRobust <= 0.05
```

### 可视化

```text
p6_certificate_roc_pr.svg
p6_certificate_calibration_curve.svg
p6_certificate_topk_yrobust_longrisk.svg
p6_certificate_component_ablation.svg
p6_leaveout_certificate_stability.svg
```

---

## P7：minimal source certificate controller

### 目标

用 P3/P5/P6 的 best legal features / generated actions / certificates，形成一个 cross-fitted minimal controller。不是 feature factory opaque score。

### Controller form

$$
Accept(a)=1
\iff
LCB(\widehat V_{h20})>0
\land
LCB(\widehat V_{h240})>0
\land
UCB(\widehat R_{h240})\le\tau_r
\land
LCB(Support)\ge\tau_s
\land
Cost(a)\le C_{max}.
$$

若 h80 label definition 被 P1 证明是必要，则加入：

$$
LCB(\widehat V_{h80})> -\epsilon.
$$

### 必须记录

```text
controller_id
controller_version
feature_count
certificate_id
generator_id
calibration_split
heldout_split
accepted_count_cal
accepted_count_heldout
coverage_heldout
YRobust_precision_heldout
h240_longrisk_heldout
V_ctrl_lcb_heldout
support_balance_pass
family_balance_pass
precision_lcb
longrisk_ucb
thresholds_frozen
uses_dataset_name
uses_outcome_at_commit
```

### 判断标准

P7 weak pass：

```text
coverage_heldout >= 0.03
YRobust_precision_heldout >= 0.20
h240_longrisk_heldout <= 0.15
V_ctrl_lcb_heldout > 0
support_balance_pass = 1
```

P7 system-candidate pass：

```text
coverage_heldout in [0.03, 0.15]
YRobust_precision_heldout >= 0.35
h240_longrisk_heldout <= 0.10
V_ctrl_lcb_heldout > 0
precision_lcb >= required_lcb
no dataset-specific behavior
```

### 可视化

```text
p7_controller_frontier_curve.svg
p7_coverage_vs_yrobust_precision.svg
p7_coverage_vs_longrisk.svg
p7_family_support_balance.svg
p7_calibration_to_heldout_drift.svg
```

---

## P8：selected runtime preflight and official runtime

### 目标

提前测 feature/certificate/payload 成本；若 P7 有 selected controller，则跑 official selected runtime。

### 假设

H8：runtime 成本不能等 controller 过后才测。

### 必须记录

```text
runtime_candidate_id
controller_id
feature_compute_q90_ms
certificate_compute_q90_ms
score_accept_q90_ms
payload_lookup_q90_ms
payload_apply_q90_ms
base_train_step_q90_ms
total_step_q90_ms
step_ratio_q90
memory_ratio
zero_candidate_controller_launch_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
no_event_preservation_pass
audit_outside_timed_path
```

### 判断标准

Runtime preflight pass：

```text
feature_compute_q90_ms <= 0.25
certificate_compute_q90_ms <= 0.25
payload_apply_q90_ms <= 0.05
```

Official runtime pass：

```text
selected_controller_exists = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_controller_launch_count = 0
no_event_preservation_pass = 1
```

### 可视化

```text
p8_runtime_waterfall.svg
p8_step_ratio_distribution.svg
p8_feature_cost_vs_signal.svg
p8_payload_apply_cost_by_payload_type.svg
p8_zero_event_semantics_audit.svg
```

---

## P9：leave-dataset-out / leave-stratum-out

### 目标

验证 controller/generator/certificate 不是 dataset-specific 或 stratum-specific artifact。

### 必须记录

```text
split_id
leaveout_type
left_out_dataset
left_out_stratum
controller_id
certificate_id
generator_id
accepted_count
coverage
YRobust_precision
h240_longrisk
V_ctrl_lcb
AUC_YRobust
support_balance_pass
thresholds_frozen
```

### 判断标准

P9 pass：

```text
leave_dataset_out_macro_pass = 1
leave_stratum_out_macro_pass = 1
AUC_drop <= 0.10
coverage_drop <= 0.03 absolute
longrisk_increase <= 0.05 absolute
```

### 可视化

```text
p9_leaveout_metric_grid.svg
p9_dataset_generalization_boxplot.svg
p9_stratum_generalization_heatmap.svg
```

---

## P10：official paired replay

### 目标

只有 system candidate pass 后才打开 official paired replay。验证 RealFunctional 是否真实打过 controls。

### Branches

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledCertificate
```

### Horizons

```text
20, 80, 240
optional 640
```

### 必须记录

```text
action_id
branch
horizon
CE_delta
NLL_delta
ECE_delta
margin_delta
acc_delta
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_fail
paired_replay_quality_pass
```

### 判断标准

P10 pass：

```text
RealFunctional beats AdamWParallel rate >= 0.60
RealFunctional beats bestLR rate >= 0.55
RealFunctional beats NoOp rate >= 0.70
shuffled payload does not pass
V_ctrl_lcb > 0
h240 longrisk <= 0.10
```

### 可视化

```text
p10_branch_value_curve.svg
p10_real_vs_control_pairplot.svg
p10_shuffle_control_negative_test.svg
p10_horizon_value_stability.svg
```

---

## P11：short/full training and MLP comparison boundary

### 目标

只有 P7/P8/P9/P10 过线后，才进入 short/full functional training。继续 Base-Acc Sentinel，但它不能进入 controller。

### 训练对象

```text
LQ-t2-h256 base
LQ-t2-h256 + selected functional controller
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
NoOp functional ablation
ShuffledFunctionalPayload ablation
```

### 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

这些数据集只作为 standardized diagnostic，不做 dataset-specific tuning。

### 必须记录

```text
train_acc_curve
val_acc_curve
test_acc
train_loss_curve
val_loss_curve
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
ECE
NLL
CEp99
margin_p10
calibration_curve
robustness_curve
sample_efficiency_curve
forgetting_score
backward_transfer
forward_transfer
runtime_step_ratio
memory_ratio
```

### 判断标准

Base sentinel pass：

```text
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

Functional short-run pass：

```text
Functional LQ improves over LQ base on at least one non-accuracy metric
No degradation beyond tolerance on test acc
beats MatchedMLP and StrongLRGridMLP on pre-registered aggregate metric
```

Full success requires：

```text
leave-dataset-out pass
paired replay pass
short/full pass
runtime pass
strong baseline pass
```

### 可视化

```text
p11_acc_curves.svg
p11_val_loss_auc.svg
p11_time_to_target.svg
p11_ece_nll_bar.svg
p11_sample_efficiency_curve.svg
p11_continual_forgetting_curve.svg
```

---

# 9. 并行执行策略

v9.4.9 不能再串行一轮只暴露一个 blocker。建议并行成 5 个 worker 组。

## Worker A：truth / label audit

执行：

```text
P0 boundary reproduction
P1 YRobust definition audit
old table quarantine audit
```

必须最先完成。若 P1 fail，其他结果都降级为 diagnostic。

## Worker B：oracle anatomy / legal features

执行：

```text
P2 source anatomy
P3 legal feature factory v2
feature cost trace
leaveout feature trace
```

可与 Worker C 并行，但不得在 P1 fail 时 official。

## Worker C：generator/certificate smoke

执行：

```text
P4 canonical existing generator revalidation
P5 direct robust source generator v2
P6 effect-valid certificate v2
```

先做 1-action / 16-action / 64-action preflight，再扩展。

## Worker D：runtime

执行：

```text
P8 runtime preflight
feature cost q90
certificate cost q90
payload apply q90
no-event preservation
```

不等 P7 过线；提前发现成本瓶颈。

## Worker E：Base-Acc Sentinel

执行：

```text
P11 Base-Acc Sentinel continuation
optional strong baseline replication
```

与 controller 隔离，禁止进入 feature/threshold/generator selection。

---

# 10. Route decision

## R0：truth base failure

条件：

```text
canonical_full_control_outcome_ready = 0
or quality_audit_pass = 0
or YRobust_definition_consistency_pass = 0
```

行动：停止 generator/controller，修 canonical materializer / label definition。

## R1：frontier exists but legal opaque

条件：

```text
canonical_horizon_robust_oracle_pass = 1
legal_feature_pass = 0
generator_pass = 0
certificate_pass = 0
```

行动：primitive redesign；不要继续 static feature patch。

## R2：legal observability pass, controller pending

条件：

```text
legal_feature_pass = 1
certificate_pass = 0 or controller_pass = 0
```

行动：focus controller/certificate calibration，不新增 generator。

## R3：generator pass, certificate pending

条件：

```text
generator_pass = 1
certificate_pass = 0
```

行动：focus effect-valid certificate。

## R4：certificate controller pass, runtime pending

条件：

```text
controller_pass = 1
runtime_pass = 0
```

行动：selected runtime optimization only。

## R5：system pass, downstream pending

条件：

```text
system_legal_controller_pass = 1
paired_replay_pass = 0
```

行动：open LDO/LSO and official paired replay。

## R6：strict PureKAN functional local success

条件：

```text
system_legal_controller_pass = 1
leaveout_pass = 1
paired_replay_pass = 1
short_run_pass = 1
```

行动：move to full-run / robustness / external-ready.

---

# 11. 明确停止的小修小补

v9.4.9 禁止把以下事情作为主线：

```text
1. 再调 PayloadNorm / PayloadLinf / NegLongRiskPayloadNorm threshold；
2. 再把 CandidateId / Step 排序当成 legal selector；
3. 再用旧 v9.3.5 table 的 oracle survivor；
4. 再用 Base-Acc Sentinel 证明 functional success；
5. 再把 oracle source panel 直接写成 official controller；
6. 再在 YRobust definition 不清楚时训练 certificate；
7. 再新增 AP0x 小变体但不测 canonical action-effect；
8. 再跑完整大 runner 后才发现 no-transform / label / cost preflight fail。
```

---

# 12. 预期结果解释

## 如果 P3 legal action-effect features 过线

说明 AP0 robust source 不是完全 outcome-only。下一步应优先做 minimal legal controller，而不是继续 generator。

## 如果 P3 fail 但 P5/P6 generator + certificate 过线

说明 AP0 opaque action 不容易选择，但可以通过 certificate-producing source generator 直接产生可部署 action。下一步转向 generated source system path。

## 如果 P3/P5/P6 全部失败

说明当前 AP0/AP0x source primitive 仍不够。此时应该进入更大一轮 primitive redesign，例如：

```text
1. 更强的 edge-local source generator；
2. 更明确的 KAN basis role certificate；
3. 更长 horizon-aware update primitive；
4. 更严格的 support-memory functional update；
5. 重新审视 functional event source 本身，而不是 controller。
```

## 如果 P7/P8 过但 P10 paired replay fail

说明 local robust source selection 可能只是 label-level artifact，不产生 real causal advantage。此时要回到 paired replay branch objective，而不是继续优化 controller threshold。

---

# 13. 交付 artifacts

v9.4.9 必须落盘：

```text
p0_v9480_recovery_boundary_reproduction.csv
p1_yrobust_definition_consistency_audit.csv
p2_canonical_robust_source_anatomy.csv
p3_legal_action_effect_feature_factory_v2.csv
p4_canonical_existing_generator_preservation.csv
p5_direct_robust_source_generator_v2.csv
p6_effect_valid_certificate_v2.csv
p7_minimal_source_certificate_controller.csv
p8_selected_runtime_preflight_and_official.csv
p9_leaveout_boundary.csv
p10_official_paired_replay_boundary.csv
p11_short_full_training_boundary.csv
no_fake_audit_v9490.csv
contract_audit_v9490.csv
route_decision_v9490.json
run_manifest_v9490.json
hash_manifest_v9490.json
```

必须可视化：

```text
p0_recovery_boundary_ladder.svg
p1_yrobust_definition_confusion_matrix.svg
p2_yrobust_vs_nonrobust_feature_effect_size.svg
p3_feature_auc_cost_frontier.svg
p3_topk_yrobust_precision_curve.svg
p4_generator_damage_heatmap.svg
p5_generator_value_risk_frontier.svg
p6_certificate_calibration_curve.svg
p7_controller_coverage_precision_risk_frontier.svg
p8_runtime_waterfall.svg
p9_leaveout_metric_grid.svg
p10_paired_replay_branch_value_curve.svg
p11_base_acc_and_functional_training_curves.svg
```

---

# 14. 最终判断

v9.4.8 recovery 后，项目的状态已经不是“truth base 不完整”，而是：

$$
\boxed{\text{canonical AP0 有 robust source frontier，但当前 legal static features 几乎完全看不见它。}}
$$

这是一种实质进展，因为它把问题从 materializer / old-table / runner semantics 重新推进到真实 functional-controller science 问题。但它也说明不能继续小修。v9.4.9 必须围绕以下本质问题设计：

$$
\boxed{\text{robust source 的证据是在 commit-time 可见，还是只能通过 outcome oracle 事后看见？}}
$$

如果可见，做 legal controller。如果不可见，就不要继续 AP0 posthoc selector，转向 certificate-producing source generator。如果 generator/certificate 也失败，再承认当前 source primitive 需要更大重构。
