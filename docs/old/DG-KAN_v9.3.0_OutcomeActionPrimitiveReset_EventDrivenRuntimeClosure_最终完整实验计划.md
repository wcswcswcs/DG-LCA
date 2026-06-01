# DG-KAN v9.3.0 Outcome-Action Primitive Reset 与 Event-Driven Runtime Closure 最终完整实验计划

> 本文件是 v9.3.0 的最终修订版。它吸收了对上一版计划的审稿意见，但不机械照搬；核心原则是：**不再抢救 StableAccept，不把 OGP feature factory 变成新的补丁堆，而是回到 functional update primitive / candidate action / outcome-grounded controller / measured runtime 的第一性问题。**
>
> 本轮不是为了在 MNIST / Fashion-MNIST / KMNIST 上调榜，也不是为了找一个刚好能过当前 split 的阈值。数据集只允许作为诊断维度、leave-out 维度和 stress 维度；controller、primitive、runtime path 都不能按 dataset 分支。
>
> v9.3.0 的中心命题是：
>
> $$
> \boxed{
> \text{是否存在一个 dataset-agnostic、outcome-grounded、system-legal、runtime-measured 的 functional update action primitive？}
> }
> $$

---

# 0. 审稿意见判断与本版修改摘要

上一版 v9.3.0 的方向是正确的：它已经把路线从 `StableAccept repair` 转到 `Outcome-Grounded Primitive Reset + Event-Driven Runtime Closure`。这比继续修 score、rank、tie policy、risk token、bad-level filter 更接近第一性原理。

但是上一版仍有一个危险：如果只把 StableAccept 换成 OGP，然后堆很多 feature group，v9.3.0 会变成新的 patching 路线。也就是说，项目会从：

```text
StableAccept patching
```

换皮成：

```text
OGP feature patching
```

这不是我们要的。真正要问的是：

$$
\boxed{
\text{当前 functional update primitive 是否产生了足够好的 candidate action？}
}
$$

因此本版计划做了以下硬修改。

## 0.1 完全采纳的意见

```text
1. Oracle frontier 分 weak / strong 两级。
   不用过强 oracle gate 过早判死 candidate population。

2. 增加 P3.5 Candidate Primitive Insufficiency Autopsy。
   如果 oracle fail，必须知道是 candidate 太少、太坏、payload 方向错、horizon 错、support 崩、还是 generator drift。

3. OGP controller 增加 minimality / sufficient-statistic gate。
   不能用 20 个 feature 拼 opaque score 后声称 pass。

4. P4 feature factory 增加 feature cost contract。
   feature 不能只看 AUC，必须记录 commit-time compute cost、memory、online materialization feasibility。

5. P2 outcome materializer 增加 sample coverage / cost budget / branch completeness。
   不能再出现“补了一些 secondary deltas，但 official sample 不够”的状态。

6. P7 runtime 增加 empty-event semantics contract。
   跳过 zero-candidate controller kernel 不能改变 base AdamW 语义。

7. P9 diagnostic paired replay 加隔离条款。
   diagnostic 结果不能反向影响 P5 threshold、feature selection 或 official route。

8. P12 增加收敛速度、sample efficiency、continual / anti-forgetting 指标。
   终极目标不是只看局部 paired replay，而是要证明 Beyond-MLP 的机制优势。
```

## 0.2 需要调整后采纳的意见

`avg_candidates_per_kernel >= 8` 不能作为 online sequential official runtime 的硬指标。v9.2.81 / v9.2.82 的数据表明：

$$
\frac{candidate\_count}{active\_step\_count}=\frac{2876}{1412}=2.0368.
$$

因此在线训练模式下，即使每个 active step 只 launch 一个 controller kernel，平均每 active step 也只有约 2 个 candidates。强行要求 online `avg_candidates/kernel >= 8` 会把 runtime target 设计成不符合训练时序的目标。

本版将 runtime 拆成两类：

```text
online_sequential_official_runtime:
  目标是 zero empty launch、no empty sync、active step 1-2 个 fused launches、step_ratio_q90 <= 1.50。

offline_replay_materializer_runtime:
  允许跨 step grouping，目标是 avg_candidates/kernel >= 8，用于 labels、audit、paired replay materializer。
```

## 0.3 本版新增的核心 pivot

上一版标题是 `Outcome-Grounded Primitive Reset`。本版更明确地把 primitive 定义为 **action primitive**：

```text
candidate event 不是一个 score row；
candidate event 是一个可执行的 functional update action。
```

因此 v9.3.0 不只要问 controller 能不能选对，还要问 action 本身是否足够好：

$$
Action(e)=\Delta\theta_{func}(e).
$$

真正的评价对象是：

$$
\left(x_t, \theta_t, g_t, \Delta\theta_{AdamW,t}, \Delta\theta_{func}(e), h\right)
\rightarrow
\left(V(e), B(e), N(e), S(e), C(e)\right).
$$

其中：

```text
V(e): value / useful improvement
B(e): bad-event risk
N(e): null-event / no-op risk
S(e): support / reliability
C(e): online runtime cost
```

---

# 1. 当前独立数据判断

## 1.1 v9.2.80：从不能评估推进到真实失败

v9.2.80 已经解决 full train-stream materialization blocker：

```text
event_count = 24192
candidate_count = 2876
stable_candidate_rows = 2876
outcome_labels_present = 1
missing_primary_label_count = 0
label_join_mode = direct_same_run_event_id
native_bucket_kernel_used_in_p6 = 1
new_full_system_step_ratio_measured = 1
```

这说明项目已经从“缺字段不能 official evaluate”进入“字段齐了之后真实 gate failure”。

但 decision 结果很差：

```text
accepted_count_heldout = 468
precision_heldout = 0.5128205128205128
coverage_heldout = 0.051587301587301584
bad_event_heldout = 0.3034188034188034
null_rate_heldout = 0.11752136752136752
precision_lcb = 0.47816314913056096
bad_event_ucb = 0.33529565729782573
step_ratio_q90 = 2.213009156635521
```

按 accepted count 反推：

```text
safe_good_count ≈ 240
bad_event_count ≈ 142
null_event_count ≈ 55-57
```

official gate 要求：

$$
Precision_{heldout}\ge0.75,
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
Coverage_{heldout}\in[0.03,0.15].
$$

由 coverage 反推 heldout denominator：

$$
N_{heldout}=\frac{468}{0.0515873016}=9072.
$$

coverage 下限对应：

$$
N_{accept,min}=\lceil0.03\times9072\rceil=273.
$$

若 accepted count 为 273，则必须满足：

$$
SafeGood\ge\lceil0.75\times273\rceil=205,
$$

$$
BadEvent\le\lfloor0.05\times273\rfloor=13,
$$

$$
NullEvent\le\lfloor0.15\times273\rfloor=40.
$$

v9.2.80 的 StableAccept accepted region 要过 gate，不是少删几个坏样本，而是必须在保住 coverage 的同时，把 bad rows 从约 142 降到约 13 量级。这是新的 frontier 问题，不是 quantized/rank 小误差。

## 1.2 v9.2.81：contract 修了，但 decision region 仍不安全

v9.2.81 用 QR5 fallback 关闭了 score/rank mismatch：

```text
score_quantized_disagreement: 18 -> 0
rank_disagreement: 2 -> 0
accept_disagreement: 0 -> 0
```

但 best repair `DR2-BadLevelTailRiskFilter` 仍是：

```text
accepted_count_heldout = 273
precision_heldout = 0.63003663003663
coverage_heldout = 0.03009259259259259
bad_event_heldout = 0.18315018315018314
null_rate_heldout = 0.19413919413919414
precision_lcb = 0.5713310098976009
bad_event_ucb = 0.23332190828443805
```

在 accepted count 273 时，gate 要求：

```text
safe_good_count >= 205
bad_event_count <= 13
null_event_count <= 40
```

DR2 粗略对应：

```text
safe_good_count ≈ 172
bad_event_count ≈ 50
null_event_count ≈ 53
```

因此 DR2 在 coverage 下限处仍然：

```text
safe_good 少约 33 个；
bad_event 多约 37 个；
null_event 多约 13 个。
```

这不是“阈值差一点”。继续加严格 filter 会掉 coverage；放宽 filter 会增加 bad-event。说明 StableAccept accepted region 本身不是 safe-good sufficient statistic。

## 1.3 v9.2.82：risk/support 有弱信号，但没有形成可迁移 frontier

v9.2.82 最好的 risk feature 是 `RSF6-RiskResidualScore`：

```text
bad-event AUC = 0.7509612892076385
safe-good AUC = 0.6931195175438596
```

这说明 risk signal 不是完全没有，但 feature ceiling 不高。更重要的是，best selected repair：

```text
repair = DR7-_risk_residual_score-thr-0.715503-n1.0-s0.05
accepted_count_cal = 230
precision_cal = 0.7630434782608696
coverage_cal = 0.03042328042328042
bad_event_cal = 0.10652173913043478
null_rate_cal = 0.13260869565217392
accepted_count_heldout = 0
coverage_heldout = 0.0
```

两个结论：

```text
1. calibration bad-event 仍然大于 0.05；
2. frozen heldout coverage 直接变成 0。
```

这不是一个可部署 frontier，而是 calibration split 上的局部 support。v9.3.0 不能继续沿着 `RSF6 + threshold` 细调。

## 1.4 candidate lifecycle 是 contract blocker，但不是唯一 scientific blocker

v9.2.81 / v9.2.82 都显示 candidate lifecycle 未闭合：

```text
old_candidate_count = 2493
new_candidate_count = 2876
shared_candidate_count = 2123
old_only_candidate_count = 370
new_only_candidate_count = 753
candidate_jaccard = 0.6540357362908195
candidate_id_mismatch_count = 2120
payload_hash_mismatch_count = 468
```

这会污染跨版本对比，必须修。但 shared rows 自身 bad-event 也高：

```text
new_only_bad_event_rate ≈ 0.278884
shared_bad_event_rate ≈ 0.277909
new_only_precision ≈ 0.398406
shared_precision ≈ 0.544984
```

因此不能把 failure 全归咎于 candidate drift。v9.3.0 要 freeze candidate lifecycle，但即便 freeze 成功，也必须重新判断 action primitive / controller 是否有真实 frontier。

## 1.5 runtime 的本质是 sparse event stream

当前 runtime 数据：

```text
step_count = 8064
active_step_count = 1412
zero_candidate_step_count = 6652
candidate_count = 2876
kernel_count_after = 72576
sync_count_after = 8064
avg_candidates_per_kernel_after = 0.03962742504409171
empty_step_kernel_fraction = 0.8249007936507936
step_ratio_q90 = 2.213009156635521
```

关键比率：

$$
\frac{zero\_candidate\_step\_count}{step\_count}=\frac{6652}{8064}=0.8249.
$$

$$
\frac{candidate\_count}{active\_step\_count}=\frac{2876}{1412}=2.0368.
$$

因此在线 runtime 要从 dense batch-major 思维改成 sparse event-driven 思维：

```text
zero-candidate step 不 launch controller kernel；
zero-candidate step 不 controller sync；
active step 只做 1-2 个 fused controller/payload launches；
audit 移出 timed path；
offline materializer 才允许跨 step batch-major grouping。
```

---

# 2. 是否仍在正确道路上

## 2.1 高层路线仍正确

DG-KAN 的高层目标不是小数据集打榜，而是：

$$
\boxed{
\text{Clean FullEdge PureKAN + manual training + kernel-native efficiency + task-safe functional update + outcome-grounded controller}
}
$$

当前已经拥有的资产包括：

```text
LQ-t2-h256 base anchor；
manual forward / backward / AdamW update；
full-row materializer；
same-run primary outcome labels；
true branch-delta / payload binding；
PF5 candidate prefilter；
StableAccept local numerical contract；
risk/support feature factory；
runtime attribution trace；
no fake / no proxy / no teacher / no loss-modification audit。
```

这些资产应该保留。

## 2.2 具体路线必须 pivot

如果下一步仍叫：

```text
StableAccept repair
```

就是错误道路。v9.2.80-v9.2.82 已经显示：

```text
StableAccept 可以稳定地产生 accept bit；
StableAccept 不能稳定地产生 safe-good accepted region；
risk/support patch 有弱信号，但 heldout frontier 不成立。
```

v9.3.0 的正确路线是：

```text
StableAccept -> feature / prefilter / negative control
Outcome-Action Primitive -> primary experimental object
Minimal OGP Controller -> final accept rule candidate
Measured Event-Driven Runtime -> official system path
```

## 2.3 离终极目标还有多远

当前离终极目标还有多个 gate：

```text
已基本完成：
  base anchor
  manual training contract
  primary outcome materialization
  payload binding
  StableAccept numerical local contract

未完成：
  canonical candidate/action lifecycle
  secondary outcome deltas
  oracle frontier classification
  candidate primitive insufficiency diagnosis
  legal OGP observability
  minimal cross-fitted controller
  measured event-driven runtime
  empty-event semantics proof
  system-legal controller
  leave-dataset-out / leave-stratum-out
  official paired replay
  short/full/continual/robustness
```

v9.3.0 不应承诺 full functional success。v9.3.0 的真正目标是：

$$
\boxed{
\text{判断当前 action primitive 是否有可部署 frontier；若有，闭合 minimal OGP controller 与 measured online runtime；若没有，明确进入 primitive redesign。}
}
$$

---

# 3. v9.3.0 总体目标

v9.3.0 的总体目标是：

$$
\boxed{
\text{把 functional event 从 score-row 问题重置为 action-primitive 问题，并建立可证伪的 outcome-grounded system path。}
}
$$

新的 accept rule 不是：

```text
StableAccept(e) + 若干 risk patch
```

而是：

$$
Accept(e)=1
\iff
LCB(V(e))>0
\land
UCB(B(e))\le\tau_b
\land
UCB(N(e))\le\tau_n
\land
LCB(S(e))\ge\tau_s
\land
C(e)\le C_{max}.
$$

其中：

```text
V(e): action value / useful improvement
B(e): bad-event downside risk
N(e): null / no-op risk
S(e): support / reliability
C(e): online runtime cost
```

StableAccept 只允许作为：

```text
stable_score_q feature；
stable_rank feature；
score_margin feature；
prefilter candidate；
negative control / ablation baseline。
```

v9.3.0 的 strong target：

$$
OfficialEligible=1,
$$

$$
SystemLegalControllerPass=1,
$$

$$
Precision_{heldout}\ge0.75,
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
StepRatio_{q90}\le1.50.
$$

v9.3.0 的最低有效推进：

```text
1. canonical candidate/action lifecycle freeze 完成；
2. secondary outcomes 对 official sample 完成；
3. oracle weak/strong frontier 被量化；
4. oracle fail 时 candidate primitive insufficiency 被归因；
5. legal OGP features 的 predictive signal、cost、minimality 被量化；
6. 至少一个 minimal OGP controller 完成 cross-fitting / LDO / LSO diagnostic；
7. empty-step-free online event-driven runtime 被 measured；
8. diagnostic causal scout 与 official controller search 完全隔离；
9. route 明确落在 OGP pass / primitive fail / feature fail / controller fail / runtime fail / causal fail 中之一。
```

---

# 4. v9.3.0 明确不做什么

本轮不做：

```text
1. 不继续把 StableAccept repair 作为主线；
2. 不按 dataset 调 threshold、risk filter、kernel route、fallback rule；
3. 不用 validation/test metric at commit time；
4. 不用 teacher、self-teacher、distillation、auxiliary loss；
5. 不引入 label smoothing、focal loss、margin loss、calibration loss；
6. 不把 functional update 改成 loss trick；
7. 不用 old artifact join 作为 official label；
8. 不把 diagnostic estimate 写成 measured runtime；
9. 不在 secondary outcomes 未 ready 时 official 打开 paired replay；
10. 不把 local native q90 reduction 当 full-system step ratio；
11. 不为了保 coverage 放宽 bad-event gate；
12. 不为了保 precision 接受 heldout accepted count = 0；
13. 不用 20 个 feature 拼 opaque score 后宣称 OGP success；
14. 不让 diagnostic paired replay 影响 P5 threshold / feature selection。
```

允许做：

```text
1. dataset-level diagnostics；
2. leave-dataset-out / leave-stratum-out diagnostics；
3. calibration split 上冻结 threshold；
4. cross-fitted minimal risk/value/null models；
5. oracle upper-bound analysis；
6. candidate primitive insufficiency autopsy；
7. legal pre-commit feature analysis；
8. parallel diagnostic paired replay，但必须隔离；
9. measured event-driven online runtime；
10. offline/replay batching for materializer / audit。
```

---

# 5. 核心假设

## H0：StableAccept final-controller 路线已经耗尽

H0 认为 StableAccept 不能通过 dataset-agnostic patch 达到 official gate。它可以保留为 feature / prefilter / baseline，但不能继续作为 final controller。

H0 成立标准：

```text
在 canonical candidate/action set 上：
  StableAccept-only fail；
  StableAccept + all registered minimal patches fail；
  failure 不是 score_q/rank/accept disagreement；
  failure 主要体现在 precision/bad_event/heldout coverage。
```

具体 gate：

$$
Precision_{heldout}<0.75
\quad\text{or}\quad
BadEventRate_{heldout}>0.05
\quad\text{or}\quad
Coverage_{heldout}\notin[0.03,0.15].
$$

H0 失败标准：

```text
一个 pre-registered StableAccept patch 在 canonical set 上过 decision gate，且 LDO/LSO 不崩。
```

若 H0 失败，可以恢复 StableAccept 作为 candidate controller，但仍必须完成 runtime、paired replay、short/full/robustness。

## H1：candidate/action lifecycle drift 是 contract 问题，但不是全部 scientific failure

H1 认为 candidate/action drift 必须修，因为它污染所有跨版本统计；但 shared rows 自身 bad-event 也高，因此修完 drift 不等于 StableAccept 成功。

H1 成立标准：

```text
candidate/action canonicalization 后：
  candidate_id_mismatch_count = 0
  action_id_mismatch_count = 0
  payload_hash_mismatch_count = 0
  candidate_count_change_explained = 1

但 StableAccept-only 仍：
  precision_heldout < 0.75 或 bad_event_heldout > 0.05。
```

H1 失败标准：

```text
canonicalization 后 StableAccept-only 或 minimal risk gate 直接过 decision gate。
```

## H2：secondary outcomes 是 value/risk/null/action autopsy 的必要条件

H2 认为 primary labels 可以评估 gate，但不足以设计 action primitive。必须补齐 secondary outcomes 才能区分：

```text
有 value 但 tail risk 高；
低 bad 但 null；
短 horizon 有用但长 horizon 伤害；
对 CE 有用但对 calibration/curvature 有害；
functional action 和 AdamW/bestLR 等价。
```

H2 成立标准：

```text
official_sample_coverage >= 0.30 of frozen candidates
or family × horizon × bucket 每格 n >= 20；
all accepted rows + matched rejected rows secondary deltas complete；
matched_control_count_per_event >= 4；
missing_secondary_delta_count_official_sample = 0。
```

H2 失败标准：

```text
secondary deltas 仍缺失，导致 P3/P4/P5/P11 无法判断 action value 或 paired replay readiness。
```

## H3：oracle frontier 必须分 weak / strong 两级

H3 认为 current candidate population 必须先通过 oracle feasibility。如果 oracle weak 都 fail，说明问题在 action primitive / candidate generator，不在 controller。

Oracle 使用 outcome label，不能部署；它只是 upper bound。

Weak oracle feasible：

$$
Coverage\ge0.03,
$$

$$
Precision\ge0.75,
$$

$$
BadEventRate\le0.05,
$$

$$
NullRate\le0.15.
$$

Strong oracle feasible：

$$
Coverage\ge0.03,
$$

$$
Precision\ge0.90,
$$

$$
BadEventRate\le0.02,
$$

$$
NullRate\le0.10.
$$

H3 route logic：

```text
weak fail:
  candidate/action primitive insufficient，停止 controller tuning。

weak pass but strong fail:
  candidate population 有潜力但 margin 薄，可以继续 OGP，但必须标记为 fragile frontier。

strong pass:
  candidate population 有足够安全 frontier，可以正式寻找 legal OGP controller。
```

## H4：oracle fail 必须触发 candidate primitive insufficiency autopsy

H4 认为 oracle fail 不能只写一句 `primitive insufficient`，必须知道为什么。

H4 成立标准：

```text
oracle miss rows / bad rows / null rows / value-negative rows 被归因到：
  candidate too sparse；
  candidate too bad；
  payload direction wrong；
  payload norm wrong；
  horizon mismatch；
  support collapse；
  candidate generator drift；
  action conflicts with AdamW；
  linearization unreliable；
  null-risk high。

attribution_fraction >= 0.90。
```

H4 失败标准：

```text
oracle fail 后无法判断该重建 candidate generator、payload primitive、horizon target、support system 还是 value observable。
```

## H5：legal OGP features 必须满足 minimality，不允许 opaque feature pile

H5 认为 OGP 应该是 primitive，不是大杂烩 score。

OGP controller 限制：

```text
最多 5 个 primitive feature groups；
必须有 single-feature baseline；
必须有 leave-one-feature-group ablation；
必须有 monotone sign consistency audit；
必须有 calibration-to-heldout drift audit；
必须报告 feature cost；
禁止 opaque 20-feature model 直接 official。
```

可接受的 minimal score 形式：

$$
S(e)=
 a_1LCB(V(e))
-a_2UCB(B(e))
-a_3UCB(N(e))
+a_4LCB(S(e))
-a_5C(e),
$$

其中：

$$
a_i\ge0.
$$

H5 成立标准：

```text
feature_group_count <= 5；
all signs match monotone prior；
no dataset branch；
feature ablation 不显示单个 leak-like feature；
heldout drift within threshold。
```

## H6：legal OGP feature 必须同时有 signal 和 online cost feasibility

H6 认为 feature 不只看 AUC，还必须看 commit-time feasibility。

H6 成立标准：

```text
AUC_bad_event >= 0.80 或 AUC_safe_good >= 0.78；
或 PR_AUC_bad_event >= 2.0 * bad_event_base_rate；
risk calibration ECE <= 0.05；
leave_dataset_auc_drop <= 0.07；
leave_stratum_auc_drop <= 0.10；
feature_compute_time_ms_q90 within budget；
feature_memory_ratio <= 1.05；
materialized_online_path = 1。
```

H6 失败标准：

```text
feature 有 AUC 但 cost 太高、需要 future info、需要 outcome at commit、或 leave-out 崩。
```

## H7：cross-fitting 是 controller 必要条件

H7 认为 single split calibration 已经导致 v9.2.82 的 heldout coverage collapse。v9.3.0 必须 cross-fit。

H7 成立标准：

```text
K-fold seed split；
leave-dataset-out；
leave-stratum-out；
thresholds frozen per calibration fold；
final metrics macro averaged；
no dataset-specific branch；
no heldout accepted count = 0 in main folds。
```

## H8：online runtime 是 event-driven sparse scheduling 问题

H8 认为 official online runtime 应以 sparse event stream 为核心，而不是追不现实的 online avg candidates/kernel >= 8。

H8 成立标准：

```text
runtime_mode = online_sequential_official；
empty_step_controller_kernel_count = 0；
empty_step_controller_sync_count = 0；
controller_launches_per_active_step_q90 <= 2；
controller_syncs_per_active_step_q90 <= 1；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05。
```

## H9：empty-event semantics 必须保持训练语义

H9 防止 runtime skip 改变 base optimizer 或 batch order。

H9 成立标准：

```text
zero-candidate step 不 launch controller kernel；
zero-candidate step 不 controller sync；
base AdamW update 与 no-controller reference bit/metric equivalent；
event scheduler does not reorder training batches；
optimizer state unchanged except allowed functional action on accepted events；
audit outside timed path；
no-event preservation pass = 1。
```

## H10：diagnostic paired replay 必须与 controller selection 隔离

H10 防止提前看 downstream 后调 controller。

H10 成立标准：

```text
diagnostic_paired_replay_status = diagnostic_not_official；
diagnostic_downstream_used_for_controller = 0；
diagnostic_downstream_used_for_threshold = 0；
diagnostic_downstream_used_for_feature_selection = 0；
route_decision.json 明确记录隔离字段。
```

## H11：system pass 后 paired replay failure 才能说明 functional value 不足

H11 防止混淆 system legality 与 functional causal advantage。

```text
P8/P10 未过时，paired replay fail 不能解释为 functional route fail；
P8/P10 过后，P11 fail 才说明 legal controller 下 functional action control-equivalent。
```

---

# 6. 数据合同

## 6.1 candidate identity contract

每个 candidate 必须有 canonical identity：

$$
candidate\_id = hash(event\_id, schema\_version, family, horizon, bucket, payload\_hash, primitive\_id).
$$

必须记录：

```text
event_id
global_row_id
candidate_id
candidate_schema_version
primitive_id
payload_hash
payload_hash_source
candidate_origin_tag
old_new_status
dataset
seed
step
batch_id
sample_group_id
family_id
bucket_id
horizon
carrier_id
stable_accept_contract_id
stable_score_q
stable_rank
score_margin
```

Pass：

```text
duplicate_event_id_count = 0
duplicate_candidate_id_count = 0
event_id_mismatch_count = 0
candidate_id_mismatch_count = 0
payload_hash_mismatch_count = 0
candidate_count_change_explained = 1
candidate_set_frozen_before_decision_search = 1
```

## 6.2 action primitive contract

v9.3.0 新增 action identity。每个 candidate 必须绑定一个可执行 functional action：

$$
action(e)=\Delta\theta_{func}(e).
$$

Action ID：

$$
action\_id = hash(candidate\_id, primitive\_id, payload\_hash, direction\_family, norm\_bucket, horizon).
$$

必须记录：

```text
action_id
candidate_id
primitive_id
functional_update_direction_family
payload_norm
payload_norm_bucket
payload_role_entropy
payload_tail_selectivity
true_delta_norm
true_delta_tail_norm
cos_action_adamw
cos_action_negative_grad
linearized_CE_delta
linearized_margin_delta
action_apply_contract_id
action_apply_error_max
action_noop_flag
action_nan_inf_flag
```

Pass：

```text
duplicate_action_id_count = 0
action_payload_missing_count = 0
action_apply_error_max <= tolerance
action_nan_inf_count = 0
action_noop_unexplained_count = 0
action_id_mismatch_count = 0
```

## 6.3 outcome contract

Primary labels：

```text
safe_good_label
bad_event_label
null_event_label
task_safe_label
useful_label
```

Secondary deltas：

```text
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
```

Branch outcomes：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledOGPScore
ShuffledActionDirection
ShuffledActionPayloadNorm
ShuffledSupportStat
```

Coverage contract：

```text
official_sample_coverage >= 0.30 of frozen candidates
or every family × horizon × bucket has n >= 20；
all accepted rows materialized；
matched rejected rows materialized；
matched_control_count_per_event >= 4；
missing_secondary_delta_count_official_sample = 0。
```

Cost contract：

```text
outcome_materializer_wallclock_sec
outcome_materializer_step_ratio
branch_materialization_cost_ms
horizon_materialization_cost_ms
matched_control_cost_ms
```

## 6.4 OGP feature contract

Feature groups allowed：

```text
Group A: StableAccept legacy features
Group B: logit / CE / margin state features
Group C: AdamW/action conflict and alignment features
Group D: branch-delta / payload geometry features
Group E: support / reliability / empirical-Bayes features
Group F: null-risk features
Group G: cost features
```

Feature legality：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
uses_validation_or_test = 0
source_measured_gap_used = 0
formula_proxy_used_for_official = 0
materialized_online_path = 1
```

Feature cost：

```text
feature_compute_time_ms_mean
feature_compute_time_ms_q90
feature_memory_mb
feature_memory_ratio
commit_time_available
requires_extra_forward
requires_extra_backward
requires_cpu_offload
```

## 6.5 minimal OGP controller contract

每个 controller 必须记录：

```text
controller_id
feature_group_count
feature_groups_used
model_class
monotonic_constraints
opaque_model_flag
single_feature_baseline_pass
leave_one_group_ablation_pass
monotone_sign_consistency_pass
calibration_to_heldout_drift
thresholds_frozen
```

Official controller constraints：

```text
feature_group_count <= 5
opaque_model_flag = 0
monotonic_constraints_valid = 1
single_feature_baseline_reported = 1
ablation_reported = 1
dataset_name_used = 0
diagnostic_downstream_used_for_controller = 0
```

## 6.6 runtime contract

每 step 记录：

```text
step_id
runtime_mode
candidate_count_in_step
active_step_flag
zero_candidate_step_flag
empty_event_semantics_pass
no_event_preservation_pass
base_adamw_equivalence_on_zero_candidate_steps
event_scheduler_does_not_reorder_batches
controller_kernel_launch_count
controller_sync_count
allocation_count
controller_launches_per_active_step
candidate_pack_time_ms
feature_compute_time_ms
risk_value_score_time_ms
accept_decision_time_ms
payload_materialize_time_ms
payload_apply_time_ms
audit_time_ms_outside_timed
base_train_step_time_ms
controller_extra_time_ms
total_step_time_ms
mlp_step_time_ms
step_ratio
peak_memory_mb
memory_ratio
```

必须区分：

```text
online_sequential_official_runtime
offline_replay_materializer_runtime
```

official system gate 只看 online sequential runtime。

---

# 7. 实验阶段

---

## P0：boundary independent reanalysis

### 目标

不信任 route 名称，只从 v9.2.80-v9.2.82 landed metrics 重新计算当前失败程度、gate gap、oracle feasibility 前置条件和 runtime lower bound。

### 假设

H0/P0：当前 failure 不是 quantized/rank mismatch，而是 decision frontier、action primitive 和 sparse runtime failure。

### 必须记录

```text
source_run_id
source_artifact_hash
candidate_count
event_count
heldout_denominator
accepted_count
safe_good_count
bad_event_count
null_event_count
precision
coverage
bad_event_rate
null_rate
precision_lcb
bad_event_ucb
min_accepted_for_coverage
safe_needed_at_min_coverage
bad_allowed_at_min_coverage
null_allowed_at_min_coverage
stable_accept_gap_to_gate
DR2_gap_to_gate
DR7_gap_to_gate
step_count
active_step_count
zero_candidate_step_count
candidate_count_per_active_step
kernel_count
sync_count
kernel_reduction_needed_for_one_launch_per_active_step
kernel_reduction_needed_for_avg8_offline
step_ratio_q90
step_ratio_reduction_needed_to_1p50
```

### 判断标准

P0 pass：

```text
boundary metrics reproduced；
gate gap computed；
quantized/rank mismatch not primary after QR5；
StableAccept patch gap classified；
runtime sparse geometry computed。
```

### 可视化

```text
p0_decision_gate_gap_bar.svg
p0_accepted_population_composition.svg
p0_stable_vs_DR2_vs_DR7_gate_table.svg
p0_runtime_sparse_stream_geometry.svg
p0_kernel_reduction_lower_bound.svg
```

---

## P1：canonical candidate/action lifecycle freeze

### 目标

把 candidate/action lifecycle 从跨版本 drift 状态变成 v9.3.0 frozen official population。所有 decision search、oracle frontier、feature analysis、runtime measurement 必须在同一个 frozen set 上执行。

### 假设

H1：修 candidate/action lifecycle 是必要 contract step，但不是 StableAccept decision success 的充分条件。

### 实现

新增：

```text
experiments/run_v9300_candidate_action_lifecycle_freeze.py
```

核心逻辑：

```text
1. 从 same-run train stream 生成 candidate rows；
2. candidate_id 使用 canonical hash；
3. action_id 使用 candidate_id + primitive_id + payload_hash + direction_family；
4. payload_hash 使用 content-addressed true payload hash；
5. candidate_schema_version 固定为 v9300；
6. action_schema_version 固定为 v9300；
7. downstream 只读 frozen_candidate_action_table_v9300.csv；
8. old v9272/v9280/v9281/v9282 只作为 diagnostic comparison。
```

### 必须记录

```text
candidate_id
action_id
event_id
candidate_schema_version
action_schema_version
payload_hash
primitive_id
family_id
bucket_id
horizon
step
dataset
seed
candidate_origin_tag
functional_update_direction_family
payload_norm
payload_role_entropy
true_delta_norm
stable_score_q
stable_rank
stable_accept_bit
payload_tensor_hash
branch_logits_hash
true_delta_logits_hash
functional_update_payload_hash
```

### 判断标准

P1 pass：

```text
duplicate_candidate_id_count = 0
duplicate_action_id_count = 0
duplicate_event_id_count = 0
payload_hash_missing_count = 0
payload_hash_mismatch_count = 0
action_payload_missing_count = 0
action_apply_error_max <= tolerance
candidate_set_frozen_before_decision_search = 1
candidate_schema_version = v9300 for all official rows
action_schema_version = v9300 for all official rows
```

### 可视化

```text
p1_candidate_lifecycle_sankey.svg
p1_action_lifecycle_sankey.svg
p1_candidate_set_old_new_overlap.svg
p1_payload_hash_mismatch_matrix.svg
p1_candidate_action_count_by_family_horizon.svg
```

---

## P2：secondary outcome and matched-control materializer with coverage/cost budget

### 目标

补齐 value/risk/null/action autopsy 所需 secondary outcomes。P2 不做 controller；它只产生可训练、可审计、可估计成本的 outcome table。

### 假设

H2：primary labels 足够评估 gate，但不足以设计新的 action primitive。必须补齐 secondary deltas、matched controls、coverage 和 materializer cost。

### 实现

新增：

```text
experiments/run_v9300_secondary_outcome_control_materializer.py
```

Materialization 范围：

```text
Tier 1 official-minimum:
  all accepted rows under StableAccept/legacy controllers；
  all OGP selected rows；
  matched rejected rows by family/horizon/score bucket/seed/action family。

Tier 2 stratified-candidate sample:
  family × horizon × bucket × action_family 每格 n >= 20。

Tier 3 full-candidate optional:
  若成本允许，全部 frozen candidates/actions。
```

Outcome horizons：

```text
horizon = 20, 80, 240
optional horizon = 640
```

Branches：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
ShuffledBranchDelta
ShuffledActionDirection
ShuffledActionPayloadNorm
ShuffledOGPScore
ShuffledSupportStat
```

### 必须记录

```text
candidate_id
action_id
event_id
branch
horizon
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
local_lipschitz_delta
basis_usage_entropy_delta
functional_channel_entropy_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
task_safe_label
useful_label
bad_event_label
null_event_label
safe_good_label
outcome_source
outcome_runtime_ms
matched_control_count
sample_coverage
family_horizon_bucket_n
```

Cost fields：

```text
outcome_materializer_wallclock_sec
outcome_materializer_step_ratio
branch_materialization_cost_ms
horizon_materialization_cost_ms
matched_control_cost_ms
```

### 判断标准

P2 primary pass：

```text
missing_primary_label_count = 0
ambiguous_label_count = 0
label_exclusivity_violation_count = 0
```

P2 downstream-ready pass：

```text
missing_secondary_delta_count_official_sample = 0
matched_control_branch_present = 1
matched_control_count_per_event >= 4
horizon_consistency_audit_pass = 1
```

P2 coverage pass：

```text
official_sample_coverage >= 0.30
or every family × horizon × bucket × action_family has n >= 20
```

P2 cost pass：

```text
outcome_materializer_wallclock_sec recorded
outcome_materializer_step_ratio recorded
cost not used as online runtime pass
```

### 可视化

```text
p2_secondary_delta_missing_before_after.svg
p2_metric_delta_distribution_by_branch.svg
p2_horizon_consistency_matrix.svg
p2_real_vs_control_delta_scatter.svg
p2_safe_bad_null_value_phase_diagram.svg
p2_materializer_coverage_heatmap.svg
p2_materializer_cost_waterfall.svg
```

---

## P3：oracle frontier weak/strong and impossibility test

### 目标

判断当前 frozen candidate/action population 是否有可用 frontier。如果 weak oracle 都不能过，继续做 legal controller 没意义，必须回到 candidate/action primitive。

### 假设

H3：存在 oracle frontier 是 legal controller 成功的必要条件。

### Oracle candidates

```text
OR0-StableAcceptCurrent
OR1-SafeGoodOracle
OR2-ValueMaxOracle
OR3-BadMinOracle
OR4-ValueRiskParetoOracle
OR5-HorizonRobustOracle
OR6-SupportBalancedOracle
OR7-ActionPrimitiveOracle
```

Oracle score 示例：

$$
S_{oracle}(e)=V_{real}(e)-\lambda_bB(e)-\lambda_nN(e)-\lambda_hHorizonInstability(e).
$$

其中：

$$
V_{real}(e)=
-w_1CEp99_{\Delta}(e)
+w_2MarginP10_{\Delta}(e)
-w_3ECE_{\Delta}(e)
-w_4NLL_{\Delta}(e)
-w_5Curvature_{\Delta}(e).
$$

### 必须记录

```text
oracle_candidate_id
oracle_level
coverage_target
accepted_count
precision
bad_event_rate
null_rate
value_mean
CEp99_delta_mean
margin_p10_delta_mean
ECE_delta_mean
NLL_delta_mean
curvature_delta_mean
support_balance_pass
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
max_action_family_share
oracle_uses_outcome_label = 1
```

### 判断标准

Weak oracle pass：

```text
coverage >= 0.03
precision >= 0.75
bad_event_rate <= 0.05
null_rate <= 0.15
support_balance_pass = 1
```

Strong oracle pass：

```text
coverage >= 0.03
precision >= 0.90
bad_event_rate <= 0.02
null_rate <= 0.10
support_balance_pass = 1
```

P3 route：

```text
weak fail:
  route = R30-OracleWeakFrontierAbsent
  go to P3.5 and P13; stop controller search.

weak pass but strong fail:
  route = R31-OracleWeakOnlyFrontier
  continue OGP with fragile-frontier warning.

strong pass:
  route = R32-OracleStrongFrontierPresent
  continue OGP with legal observability search.
```

### 可视化

```text
p3_oracle_precision_bad_coverage_frontier.svg
p3_oracle_value_risk_pareto.svg
p3_oracle_support_balance.svg
p3_oracle_action_family_balance.svg
p3_oracle_vs_stableaccept_overlap.svg
p3_weak_strong_oracle_dashboard.svg
```

---

## P3.5：candidate/action primitive insufficiency autopsy

### 目标

如果 P3 oracle weak fail，必须知道当前 primitive 为什么不够。P3.5 是防止“controller fail 后盲目设计新 feature”的关键阶段。

### 假设

H4：oracle fail 的根因可以归因到 candidate/action primitive 的若干机制。

### Autopsy dimensions

```text
A1-candidate-too-sparse:
  总 candidate/action 数不足，coverage 无法达到 0.03。

A2-candidate-too-bad:
  candidate 足够多，但 bad-event base rate 太高。

A3-action-direction-wrong:
  action 与 AdamW / negative grad 冲突，linearized CE delta 为正。

A4-action-norm-wrong:
  payload norm 太大导致 bad tail，或太小导致 null。

A5-action-role-collapse:
  payload_role_entropy 太低，集中在脆弱 channel。

A6-horizon-mismatch:
  horizon 20 有效但 80/240 伤害，或反过来。

A7-support-collapse:
  safe rows 只集中于极少 family/stratum/action_family。

A8-generator-drift:
  candidate_origin_tag / schema drift 改变 population。

A9-observability-gap:
  oracle frontier 存在，但 legal features 与 oracle overlap 很低。

A10-null-dominant:
  bad 不高，但 useful value 太少，null rate 高。
```

### 必须记录

```text
candidate_id
action_id
candidate_source
candidate_origin_tag
functional_update_direction_family
payload_norm
payload_role_entropy
true_delta_norm
cos_action_adamw
cos_action_negative_grad
linearized_CE_delta
linearized_margin_delta
real_value_distribution
bad_event_distribution
null_event_distribution
candidate_rejection_reason
oracle_miss_reason
autopsy_mode
autopsy_submode
```

Aggregate metrics：

```text
autopsy_attribution_fraction
candidate_density_by_step
candidate_density_by_family
safe_good_density_by_action_family
bad_event_rate_by_action_family
null_rate_by_payload_norm_bucket
value_by_horizon
support_entropy
oracle_miss_count_by_mode
```

### 判断标准

P3.5 pass：

```text
if P3 weak fail:
  oracle miss attribution_fraction >= 0.90
  primary primitive blocker identified
  next primitive redesign target specified

if P3 weak pass:
  P3.5 diagnostic still reports fragile modes but does not block P4
```

### 可视化

```text
p35_primitive_insufficiency_sankey.svg
p35_action_value_bad_null_by_family.svg
p35_payload_norm_vs_value_risk.svg
p35_action_alignment_vs_outcome.svg
p35_horizon_mismatch_heatmap.svg
p35_support_collapse_by_action_family.svg
```

---

## P4：legal OGP observability and minimal feature factory

### 目标

构造 legal pre-commit features，判断它们是否能逼近 oracle frontier。P4 不搜索 final thresholds，只评估 feature validity、minimality、cost 和 leave-out stability。

### 假设

H5/H6：存在少量 legal primitive features 可以估计 value、bad risk、null risk、support 和 cost；如果需要大量 opaque features 才有信号，则 OGP 不合格。

### OGP primitive feature groups

#### OGP-A：AdamW-aligned action value primitive

记录：

```text
cos_action_negative_grad
cos_action_adamw_delta
functional_norm_over_adamw_norm
projected_CE_descent_estimate
rolewise_conflict_score
```

线性化：

$$
\widehat{\Delta CE}_{lin}(e)=\nabla_{\theta}CE(\theta_t)^T\Delta\theta_{func}(e).
$$

Good sign：

$$
\widehat{\Delta CE}_{lin}(e)<0.
$$

#### OGP-B：tail-risk primitive

记录：

```text
CEp99_current
margin_p10_current
wrong_conf_p90
hard_tail_fraction
branch_delta_tail_norm
branch_delta_tail_alignment
```

Score：

$$
TailRisk(e)=a_1CEp99+a_2WrongConfP90-a_3MarginP10+a_4BranchDeltaTailConflict.
$$

#### OGP-C：linearization reliability primitive

记录：

```text
linearized_CE_delta
linearized_margin_delta
audit_realized_delta_subset
linearization_abs_error
linearization_error_ucb
```

Reliability：

$$
Reliability(e)=1-UCB(|\Delta_{realized}-\Delta_{linearized}|).
$$

#### OGP-D：support empirical-Bayes primitive

记录：

```text
neighbor_count
family_count
horizon_count
feature_bin_count
empirical_bayes_bad_mean
empirical_bayes_bad_ucb
empirical_bayes_value_lcb
support_effective_sample_size
```

Bad UCB：

$$
BadUCB_{EB}(e)=
\hat p_{bad}(e)
+z_{\alpha}\sqrt{\frac{\hat p_{bad}(e)(1-\hat p_{bad}(e))}{n_{eff}(e)+\epsilon}}
+\lambda_{shift}ShiftRisk(e).
$$

#### OGP-E：null-risk primitive

记录：

```text
payload_norm
branch_delta_norm
predicted_margin_gain
predicted_CE_gain
functional_channel_entropy
null_ucb
```

Null UCB：

$$
NullUCB(e)=P(|V(e)|<\epsilon \mid x_e).
$$

#### OGP-F：cost-aware primitive

记录：

```text
candidate_count_in_step
kernel_shape_id
payload_apply_cost_estimate
feature_compute_cost_estimate
active_step_flag
```

Cost gate：

$$
C(e)\le C_{max}.
$$

### 必须记录

```text
feature_id
feature_group
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_test
AUC_bad_event
AUC_safe_good
AUC_null_event
PR_AUC_bad_event
PR_AUC_safe_good
Brier_bad
ECE_bad
calibration_slope
leave_dataset_auc_drop
leave_stratum_auc_drop
feature_missing_rate
feature_compute_time_ms_mean
feature_compute_time_ms_q90
feature_memory_ratio
materialized_online_path
commit_time_available
requires_extra_forward
requires_extra_backward
```

Minimality fields：

```text
single_feature_baseline_auc
single_feature_baseline_pr_auc
feature_group_ablation_delta
monotone_sign_expected
monotone_sign_observed
monotone_sign_consistency_pass
opaque_model_flag
```

### 判断标准

P4 feature pass：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
feature_missing_rate <= 0.01
materialized_online_path = 1
feature_compute_time_ms_q90 within budget
feature_memory_ratio <= 1.05
```

Signal pass：

```text
AUC_bad_event >= 0.80
or AUC_safe_good >= 0.78
or PR_AUC_bad_event >= 2.0 * bad_event_base_rate
risk calibration ECE <= 0.05
leave_dataset_auc_drop <= 0.07
leave_stratum_auc_drop <= 0.10
```

Minimality pass：

```text
feature_group_count_candidate <= 5
single_feature_baseline_reported = 1
leave_one_group_ablation_reported = 1
monotone_sign_consistency_pass = 1
opaque_model_flag = 0
```

P4 weak pass：

```text
AUC_bad_event >= 0.75 but LDO/LSO unstable，或 cost 不可 online。
```

Weak pass 不能进入 official controller，只能进入 feature/primitive redesign。

### 可视化

```text
p4_feature_auc_bar.svg
p4_feature_pr_auc_lift.svg
p4_risk_calibration_curve.svg
p4_feature_oracle_overlap.svg
p4_ogp_feature_pareto_runtime_vs_auc.svg
p4_leaveout_auc_drop_heatmap.svg
p4_value_risk_feature_phase_diagram.svg
p4_feature_ablation_waterfall.svg
p4_monotone_sign_consistency.svg
```

---

## P5：cross-fitted minimal OGP value-risk-null controller

### 目标

建立不使用 dataset branch 的 final controller candidate。所有阈值必须在 calibration folds 上冻结，再在 heldout folds、LDO、LSO 上评估。P5 的 controller 必须 minimal，不允许 opaque feature pile。

### 假设

H7：cross-fitted minimal OGP controller 能避免 v9.2.82 的 calibration pass / heldout zero coverage 崩溃。

### Candidate controllers

#### C0：StableAccept legacy negative control

```text
Accept(e)=StableAccept(e).
```

#### C1：StableAccept + one OGP risk primitive

```text
Accept(e)=StableAccept(e) and UCB(B(e)) <= tau_b.
```

#### C2：minimal OGP value-risk-null gate

$$
Accept(e)=1
\iff
LCB(V(e))>0
\land
UCB(B(e))\le\tau_b
\land
UCB(N(e))\le\tau_n
\land
LCB(S(e))\ge\tau_s.
$$

#### C3：minimal OGP monotone score + hard safety gates

$$
S_{OGP}(e)=
LCB(V(e))
-\lambda_bUCB(B(e))
-\lambda_nUCB(N(e))
+\lambda_sLCB(S(e))
-\lambda_cC(e).
$$

Accept：

$$
Accept(e)=1
\iff
S_{OGP}(e)\ge\tau
\land UCB(B(e))\le\tau_b
\land UCB(N(e))\le\tau_n
\land LCB(S(e))\ge\tau_s.
$$

#### C4：two-stage cheap OGP + exact pre-commit safety audit

```text
Stage 1: cheap OGP prefilter。
Stage 2: exact pre-commit safety confirm for borderline / high-value events。
```

Stage 2 只能使用 commit 前可得 exact computation，不得用 outcome labels。

### Splits

```text
Seed folds:
  rotate train/calibration/heldout seeds；
  main report macro averaged。

Leave-dataset-out:
  train/calibrate on two datasets, evaluate third。

Leave-stratum-out:
  train/calibrate on all but one signal stratum, evaluate held-out stratum。

Leave-action-family-out diagnostic:
  train/calibrate on all but one action family, evaluate held-out action family。
```

### 必须记录

```text
controller_id
feature_set
feature_group_count
model_class
monotonic_constraints
opaque_model_flag
thresholds
calibration_fold
heldout_fold
split_type
heldout_entity
dataset_name_used
diagnostic_downstream_used_for_controller
accepted_count_cal
accepted_count_heldout
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
value_mean_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
value_mean_heldout
precision_lcb
bad_event_ucb
null_event_ucb
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
max_action_family_share
oracle_overlap
stableaccept_overlap
single_feature_baseline_reported
leave_one_group_ablation_reported
monotone_sign_consistency_pass
calibration_to_heldout_drift
```

### 判断标准

Decision gate：

$$
Precision_{heldout}\ge0.75,
$$

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
NullRate_{heldout}\le0.15,
$$

$$
Precision_{LCB}\ge0.75,
$$

$$
BadEventRate_{UCB}\le0.05.
$$

Support balance：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
accepted_action_family_count >= 3
max_family_share <= 0.50
max_stratum_share <= 0.60
max_action_family_share <= 0.70
```

Minimality gate：

```text
feature_group_count <= 5
opaque_model_flag = 0
single_feature_baseline_reported = 1
leave_one_group_ablation_reported = 1
monotone_sign_consistency_pass = 1
diagnostic_downstream_used_for_controller = 0
```

Cross-fit stability：

```text
coverage_heldout > 0 in every main fold；
bad_event gate pass in at least 2/3 seed folds；
macro decision score pass；
no dataset-specific branch。
```

### 可视化

```text
p5_controller_frontier_precision_bad_coverage.svg
p5_calibration_to_heldout_drift.svg
p5_crossfit_fold_matrix.svg
p5_ogp_vs_stableaccept_overlap.svg
p5_support_balance_sunburst.svg
p5_null_bad_decoupling.svg
p5_value_risk_null_3d_frontier.svg
p5_minimality_ablation_matrix.svg
```

---

## P6：decision failure autopsy v3

### 目标

如果 P5 未过，必须知道是 oracle frontier 不够、legal observability 不够、support collapse、controller form 不对、feature cost gate 过滤过多，还是 label/horizon conflict。不能只写 `no_dataset_agnostic_decision_repair`。

### Failure modes

```text
DF1-oracle_weak_frontier_absent
DF2-oracle_strong_frontier_absent
DF3-legal_feature_oracle_gap
DF4-risk_underestimation
DF5-null_bad_conflict
DF6-value_positive_but_bad_tail
DF7-support_collapse
DF8-heldout_coverage_zero
DF9-family_horizon_action_concentration
DF10-candidate_action_lifecycle_regression
DF11-label_horizon_mismatch
DF12-runtime_cost_gate_filters_too_much
DF13-minimality_gate_fail
DF14-feature_cost_infeasible
DF15-diagnostic_leak_detected
```

### 必须记录

```text
failed_controller_id
bad_accepted_count
null_accepted_count
missed_safe_good_count
coverage_lost_count
failure_mode
failure_submode
feature_values
oracle_label
legal_score
support_stats
family_id
horizon
bucket_id
action_family
payload_norm_bucket
seed
```

### 判断标准

P6 pass：

```text
bad accepted attribution fraction >= 0.95
missed safe-good attribution fraction >= 0.90
coverage collapse attribution fraction >= 0.90
minimality/cost/observability failure separated
```

### 可视化

```text
p6_decision_failure_sankey.svg
p6_missed_safe_good_vs_bad_accepted.svg
p6_support_collapse_by_family_horizon_action.svg
p6_legal_score_vs_oracle_score.svg
p6_feature_cost_failure_matrix.svg
```

---

## P7：online event-driven runtime with empty-event semantics

### 目标

把 P6 path 从 per-step fixed launch/sync 改成 event-driven sparse scheduler。P7 必须是 measured runtime，不允许 diagnostic estimate。

### 假设

H8/H9：当前 runtime 主要浪费在 zero-candidate fixed launch/sync；online official runtime 应以 active-step event-driven scheduling 为目标，同时保持 empty-event semantics。

### Runtime candidates

#### RT0：v9.2.82 reference

```text
fixed per-step launch
kernel_count = 72576
sync_count = 8064
step_ratio_q90 = 2.213009
```

#### RT1：empty-step skip measured

```text
zero-candidate step 不 launch controller kernel；
zero-candidate step 不 controller sync；
只保留 base training step。
```

#### RT2：one active-step fused controller kernel

```text
每个 active step 将 feature、score、risk/value lookup、accept decision 合入一个 kernel。
```

#### RT3：two-kernel active-step path

```text
kernel 1: feature/score/accept；
kernel 2: payload materialize/apply。
```

#### RT4：persistent workspace event scheduler

```text
preallocate candidate buffers；
preallocate feature buffers；
preallocate accept mask；
no per-step allocation。
```

#### RT5：offline replay batch-major materializer

```text
允许跨 step grouping；
用于 secondary outcomes / audit / paired replay；
不作为 online official runtime。
```

#### RT6：hybrid official

```text
online RT2/RT3 + offline RT5 for audit outside timed path。
```

### 必须记录

```text
runtime_candidate_id
runtime_mode
step_count
active_step_count
zero_candidate_step_count
candidate_count
empty_step_controller_kernel_count
empty_step_controller_sync_count
empty_event_semantics_pass
no_event_preservation_pass
base_adamw_equivalence_on_zero_candidate_steps
event_scheduler_does_not_reorder_batches
optimizer_state_equivalence_on_empty_steps
audit_outside_timed_path
controller_kernel_launch_count
controller_sync_count
controller_launches_per_active_step_mean
controller_launches_per_active_step_q90
controller_syncs_per_active_step_mean
controller_syncs_per_active_step_q90
allocation_count
allocation_count_per_active_step
candidate_pack_time_ms_q90
feature_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
audit_outside_timed_ms
total_step_time_ms_q50
total_step_time_ms_q90
active_step_time_ms_q90
empty_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
active_step_ratio_q90
memory_ratio
accept_disagreement_count
score_error_max
payload_apply_error_max
```

### 判断标准

P7 online runtime pass：

```text
runtime_mode = online_sequential_official
empty_step_controller_kernel_count = 0
empty_step_controller_sync_count = 0
empty_event_semantics_pass = 1
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
event_scheduler_does_not_reorder_batches = 1
audit_outside_timed_path = 1
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
allocation_count_per_active_step <= 0.05
accept_disagreement_count = 0
payload_apply_error_max <= tolerance
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

P7 diagnostic pass：

```text
step_ratio_q90 <= 2.00
and launch/sync reduction >= 80%
```

Offline materializer pass：

```text
runtime_mode = offline_replay_materializer
avg_candidates_per_kernel >= 8
batch_major_grouping_measured = 1
not_used_as_online_official_runtime = 1
```

### 可视化

```text
p7_runtime_mode_comparison.svg
p7_empty_vs_active_step_timing.svg
p7_kernel_sync_reduction.svg
p7_launches_per_active_step_hist.svg
p7_runtime_waterfall.svg
p7_step_ratio_q50_q90.svg
p7_memory_ratio.svg
p7_empty_event_semantics_audit.svg
```

---

## P8：controller-runtime integration

### 目标

组合 P5 的 best minimal OGP controller 与 P7 的 measured online runtime。只要 decision、minimality、diagnostic isolation 或 runtime 任一失败，P8 不得 official pass。

### 必须记录

```text
system_candidate_id
controller_id
runtime_candidate_id
feature_set
feature_group_count
thresholds
candidate_count
action_count
event_count
accepted_count
candidate_rate
action_rate
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
accepted_family_count
accepted_signal_strata_count
accepted_action_family_count
max_family_share
max_stratum_share
max_action_family_share
step_ratio_q90
active_step_ratio_q90
memory_ratio
controller_launches_per_active_step_q90
empty_step_controller_kernel_count
empty_event_semantics_pass
no_event_preservation_pass
accept_disagreement_count
payload_binding_pass
action_binding_pass
secondary_outcome_ready
materialized_system_path
diagnostic_derived_from_measured_components
projection_used
source_measured_gap_used
formula_proxy_used
dataset_name_used
diagnostic_downstream_used_for_controller
official_eligible
system_legal_controller_pass
```

### 判断标准

P8 system pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
payload_binding_pass = 1
action_binding_pass = 1
secondary_outcome_ready = 1
materialized_system_path = 1
diagnostic_derived_from_measured_components = 0
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
dataset_name_used = 0
diagnostic_downstream_used_for_controller = 0
minimality_gate_pass = 1
empty_event_semantics_pass = 1
```

Decision gates：

$$
Precision_{heldout}\ge0.75,
$$

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
BadEventRate_{heldout}\le0.05,
$$

$$
NullRate_{heldout}\le0.15,
$$

$$
Precision_{LCB}\ge0.75,
$$

$$
BadEventRate_{UCB}\le0.05.
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
p8_system_quality_cost_frontier.svg
p8_official_gate_dashboard.svg
p8_controller_runtime_pareto.svg
p8_failure_reason_matrix.svg
p8_minimality_runtime_contract_dashboard.svg
```

---

## P9：parallel diagnostic causal scouts with isolation

### 目标

加快实验，不等 full official path 才开始看 causal signal。但 P9 diagnostic 不得写成 official paired replay，也不得影响 controller threshold / feature selection。

### 设计

P9 与 P5/P7 并行准备，一旦 P5 出现 top-2 controller candidates，就跑小规模 paired replay scout：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
horizons = 20,80,240
controllers = top2 OGP + StableAccept negative control
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, shuffled controls
```

### 隔离合同

```text
diagnostic_paired_replay_status = diagnostic_not_official
diagnostic_downstream_used_for_controller = 0
diagnostic_downstream_used_for_threshold = 0
diagnostic_downstream_used_for_feature_selection = 0
diagnostic_downstream_used_for_runtime_choice = 0
```

### 必须记录

```text
controller_id
dataset
seed
horizon
branch
accepted_count
coverage
bad_event_rate
null_rate
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
diagnostic_downstream_used_for_controller
```

### 判断标准

P9 scout promising：

```text
RealFunctional beats AdamWParallel in >= 50% macro slices；
RealFunctional beats bestLR in >= 50% macro slices；
task_safe holds: Acc_real >= Acc_adamw - 0.005；
shuffled controls do not match RealFunctional。
```

P9 fail 不代表 full route fail，除非 P8/P10 system pass 后 official P11 仍 fail。

### 可视化

```text
p9_diagnostic_paired_replay_macro_beat.svg
p9_branch_delta_pareto.svg
p9_shuffle_control_matrix.svg
p9_task_safety_by_slice.svg
p9_diagnostic_isolation_audit.svg
```

---

## P10：leave-dataset-out / leave-stratum-out official

### 目标

只有 P8 pass 后 official 打开。证明 controller 不是 pooled calibration artifact，也不是 dataset-specific route。

### 设置

Leave-dataset-out：

```text
calibrate MNIST + Fashion, evaluate KMNIST
calibrate MNIST + KMNIST, evaluate Fashion
calibrate Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate all but one signal stratum
evaluate held-out stratum
```

Leave-action-family-out diagnostic：

```text
calibrate all but one high-volume action family
evaluate held-out action family
```

### 必须记录

```text
split_type
heldout_entity
controller_id
runtime_candidate_id
candidate_rate
action_rate
precision
coverage
bad_event_rate
null_rate
precision_lcb
bad_event_ucb
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
support_balance_pass
minimality_gate_pass
```

### 判断标准

LDO decision pass：

```text
at least 2/3 held-out datasets pass decision gates；
no held-out dataset has bad_event_rate > 0.10；
coverage > 0 for all held-out datasets；
dataset_name_used = 0。
```

LSO pass：

```text
>=70% held-out strata task-safe；
macro bad_event_rate <= 0.05；
macro precision >= 0.75。
```

### 可视化

```text
p10_leave_dataset_out_matrix.svg
p10_leave_stratum_out_matrix.svg
p10_leave_action_family_out_diagnostic.svg
p10_dataset_diagnostic_no_tuning_audit.svg
```

---

## P11：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。只有 P8 system pass 与 P10 leave-out pass 后打开。

### 前置条件

```text
P8 system pass = 1
P10 LDO/LSO pass = 1
secondary outcome ready = 1
runtime official pass = 1
diagnostic isolation pass = 1
```

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
  ShuffledTrueBranchDelta
  ShuffledPF5Prefilter
  ShuffledCandidatePayload
  ShuffledActionDirection
  ShuffledActionPayloadNorm
  ShuffledStableAcceptScore
  ShuffledStableAcceptRank
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledOGPSupportScore
  ShuffledOutcomeLabel
  ShuffledNativeRuntimePath
  ShuffledFunctionalUpdatePayload
  ShuffledFrozenBridge
  FunctionalChannelShuffled
  TailMaskShuffled
  RoleScoreShuffled
  DatasetRouteShuffled
  EventRouteShuffled
  InvertedRoleMask
```

### 必须记录

```text
controller_id
dataset
seed
horizon
signal_stratum
event_family
action_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
candidate_rate
action_rate
coverage
bad_event_rate
null_rate
step_q90
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
Acc_{each\ slice,Real}\ge Acc_{AdamW}-0.005.
$$

Shuffle controls：

```text
所有 shuffled control 不得达到 RealFunctional 的 macro beat pattern。
```

### 可视化

```text
p11_official_paired_replay_pareto.svg
p11_macro_beat_rate.svg
p11_signal_stratum_win_matrix.svg
p11_action_family_win_matrix.svg
p11_shuffle_control_matrix.svg
p11_system_gate_distribution.svg
```

---

## P12：short-run / full-run / continual / robustness

### 目标

只有 P11 pass 后打开。验证 local causal advantage 能否进入连续训练，并排除 LR grid、QuadraticFeatureMLP、shuffled-controller、sample-efficiency 和 anti-forgetting explanations。

### 设置

```text
short-run steps = 50, 240, 640
full-run seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  ShuffledTrueBranchDelta
  ShuffledOGPValueScore
  ShuffledOGPRiskScore
  ShuffledOGPSupportScore
  ShuffledFunctionalPayload
  ShuffledActionDirection
  ShuffledRuntimePath
```

Continual / anti-forgetting diagnostic：

```text
sequence A: MNIST -> Fashion -> KMNIST
sequence B: Fashion -> KMNIST -> MNIST
sequence C: KMNIST -> MNIST -> Fashion
```

### 必须记录

```text
dataset
seed
candidate
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
functional_event_count
candidate_rate
action_rate
coverage
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
strong_baseline_beaten
robustness_pass
base_checkpoint_hash
```

Sample efficiency / convergence：

```text
ValLossAUC_step
ValLossAUC_time
TrainLossAUC_step
TrainLossAUC_time
time_to_target_acc
steps_to_target_acc
time_to_target_loss
steps_to_target_loss
sample_efficiency_gain
```

Continual metrics：

```text
retained_accuracy_old_tasks
forgetting_rate
backward_transfer
forward_transfer
old_task_CEp99_drift
old_task_margin_p10_drift
old_task_ECE_drift
```

### 判断标准

Full functional pass：

$$
Acc_{functional}\ge Acc_{AdamW}-0.005
$$

且至少一个成立：

$$
Acc_{functional}>Acc_{AdamWParallel},
$$

$$
ECE_{functional}<ECE_{AdamW},
$$

$$
NLL_{functional}<NLL_{AdamW},
$$

$$
Curvature_{functional}\le0.90Curvature_{AdamW}.
$$

Sample efficiency pass：

```text
ValLossAUC_time improves over AdamWParallel or bestLR；
time_to_target_acc <= 0.90 * baseline；
steps_to_target_acc <= 0.90 * baseline。
```

Anti-forgetting pass：

```text
retained_accuracy_old_tasks >= baseline - 0.005；
forgetting_rate <= baseline forgetting_rate；
old_task_CEp99_drift <= baseline drift。
```

Strong baseline pass：

```text
Functional gain not explained by QuadraticFeatureMLP；
Functional gain not explained by LR grid；
Functional gain not explained by shuffled OGP/controller/runtime/payload/action。
```

### 可视化

```text
p12_short_run_learning_curves.svg
p12_full_run_macro_results.svg
p12_valloss_auc_step_time.svg
p12_time_to_target.svg
p12_continual_forgetting_matrix.svg
p12_backward_forward_transfer.svg
p12_strong_baseline_comparison.svg
p12_robustness_dashboard.svg
```

---

## P13：candidate/action primitive reset branch

### 目标

如果 P3 weak oracle fail 或 P3.5 显示 current action primitive 不足，P13 打开。P13 不是 controller tuning，而是重新设计 candidate/action primitive。

### 前置条件

```text
P3 weak oracle fail
or P3.5 primary blocker in:
  candidate-too-sparse
  candidate-too-bad
  action-direction-wrong
  action-norm-wrong
  horizon-mismatch
  null-dominant
```

### Primitive candidates

#### PA0：current action primitive reference

用于复现 v9.2.82 current action population。

#### PA1：AdamW-aligned action primitive

只生成与 AdamW / negative gradient 不强冲突的 actions：

$$
cos(\Delta\theta_{func}, \Delta\theta_{AdamW})\ge\tau_{align}.
$$

#### PA2：tail-safe action primitive

限制 action 对 hard-tail samples 的负影响：

$$
TailRisk_{pred}(e)\le\tau_{tail}.
$$

#### PA3：null-avoidant action primitive

避免 payload 太小或 branch_delta 太弱：

$$
LCB(|V(e)|)>\epsilon_V.
$$

#### PA4：horizon-robust action primitive

只保留在多个 horizon 上不冲突的 action：

$$
sign(V_{20})=sign(V_{80})=sign(V_{240})
$$

或至少：

$$
UCB(B_{240})\le\tau_b.
$$

#### PA5：support-balanced action primitive

生成时强制覆盖多个 family/action_family，避免 support collapse。

#### PA6：payload-norm controlled primitive

对 payload norm / role entropy 加约束，减少 bad tail 和 null：

$$
Norm_{min}\le||\Delta\theta_{func}||\le Norm_{max}.
$$

### 必须记录

```text
primitive_candidate_id
candidate_count
action_count
candidate_density_by_step
action_density_by_step
safe_good_base_rate
bad_event_base_rate
null_event_base_rate
oracle_weak_pass
oracle_strong_pass
action_alignment_distribution
payload_norm_distribution
payload_role_entropy_distribution
horizon_consistency
support_balance
runtime_cost_estimate
```

### 判断标准

P13 primitive candidate pass：

```text
oracle weak pass = 1；
bad_event_base_rate improves over PA0；
null_event_base_rate improves or value_mean improves；
support_balance_pass = 1；
online cost estimate within envelope candidate；
no dataset-specific generation branch。
```

P13 strong pass：

```text
oracle strong pass = 1；
legal OGP P4 signal improves over current RSF6-level signal；
P5 minimal controller has non-zero heldout coverage in all main folds。
```

### 可视化

```text
p13_primitive_candidate_frontier.svg
p13_action_density_by_step.svg
p13_action_alignment_distribution.svg
p13_payload_norm_value_risk.svg
p13_oracle_frontier_by_primitive.svg
p13_support_balance_by_primitive.svg
```

---

# 8. 并行执行计划

v9.3.0 必须并行验证，避免每轮只试一个 runner。

## Batch A：contract / materializer / runtime smoke

可以并行执行：

```text
A1: P0 boundary reanalysis
A2: P1 candidate/action lifecycle freeze
A3: P2 secondary outcome materializer smoke
A4: P7 runtime scheduler smoke with dummy accept mask
```

产物：

```text
p0_boundary_reanalysis.csv
frozen_candidate_action_table_v9300.csv
secondary_outcome_materializer_smoke.csv
runtime_event_scheduler_smoke.csv
```

## Batch B：oracle / primitive autopsy / feature / runtime

P1 frozen table ready 后并行：

```text
B1: P3 oracle weak/strong frontier
B2: P3.5 candidate/action primitive insufficiency autopsy
B3: P4 OGP feature factory
B4: P7 measured online runtime with frozen candidate stream
B5: offline replay batch-major materializer
```

## Batch C：controller search 与 diagnostic causal scout

P4 feature candidates ready 后并行：

```text
C1: P5 cross-fitted minimal OGP controller
C2: P6 decision failure autopsy for all non-pass controllers
C3: P9 diagnostic paired replay for top-2 candidates with isolation
C4: runtime integration microbenchmark for top-2 candidates
```

## Batch D：official system

只有 P5 decision pass 与 P7 runtime pass 后：

```text
D1: P8 system controller
D2: P10 LDO/LSO
D3: P11 official paired replay
```

## Batch E：longer runs

只有 P11 pass 后：

```text
E1: P12 short-run
E2: P12 full-run
E3: P12 continual / anti-forgetting
E4: robustness / strong baseline
```

## Batch F：primitive reset branch

如果 P3 weak oracle fail：

```text
F1: P3.5 primitive autopsy completed
F2: P13 primitive candidate generation
F3: P13 oracle frontier rerun
F4: return to P4/P5 only if primitive weak oracle pass
```

---

# 9. Required artifacts

```text
run_manifest.json
contract_audit_v9300.csv
p0_boundary_reanalysis.csv
p1_candidate_action_lifecycle_freeze.csv
frozen_candidate_action_table_v9300.csv
candidate_action_lifecycle_trace_v9300.csv
p2_secondary_outcome_control_materializer.csv
secondary_outcome_trace_v9300.csv
matched_control_outcome_trace_v9300.csv
p3_oracle_frontier_weak_strong.csv
oracle_frontier_trace_v9300.csv
p35_candidate_action_primitive_autopsy.csv
primitive_autopsy_trace_v9300.csv
p4_ogp_feature_factory.csv
ogp_feature_trace_v9300.csv
feature_cost_trace_v9300.csv
minimality_audit_trace_v9300.csv
p5_crossfitted_minimal_ogp_controller.csv
controller_frontier_trace_v9300.csv
p6_decision_failure_autopsy_v3.csv
decision_failure_trace_v9300.csv
p7_online_event_driven_runtime.csv
runtime_event_scheduler_trace_v9300.csv
runtime_empty_event_semantics_trace_v9300.csv
runtime_component_trace_v9300.csv
p8_system_legal_controller_v9300.csv
system_controller_trace_v9300.csv
p9_diagnostic_paired_replay_scout.csv
diagnostic_isolation_audit_v9300.csv
diagnostic_paired_replay_trace_v9300.csv
p10_leave_dataset_stratum_out.csv
leaveout_trace_v9300.csv
p11_official_paired_replay.csv
official_paired_replay_trace_v9300.csv
p12_short_full_continual_robustness.csv
short_full_continual_trace_v9300.csv
p13_candidate_action_primitive_reset.csv
primitive_reset_trace_v9300.csv
route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

---

# 10. Failure taxonomy

```text
F1_contract_violation
F2_dataset_tuning_detected
F3_teacher_or_loss_modification_detected
F4_candidate_lifecycle_freeze_fail
F5_action_lifecycle_freeze_fail
F6_payload_hash_mismatch
F7_candidate_set_unstable
F8_action_set_unstable
F9_primary_outcome_missing
F10_secondary_outcome_missing
F11_outcome_sample_coverage_insufficient
F12_matched_control_missing
F13_oracle_weak_frontier_absent
F14_oracle_strong_frontier_absent
F15_candidate_action_primitive_insufficient
F16_primitive_autopsy_incomplete
F17_legal_feature_oracle_gap
F18_ogp_feature_unpredictive
F19_ogp_feature_leaveout_unstable
F20_ogp_feature_cost_infeasible
F21_minimality_gate_fail
F22_no_dataset_agnostic_ogp_decision
F23_decision_precision_fail
F24_decision_coverage_fail
F25_decision_bad_event_fail
F26_decision_null_rate_fail
F27_decision_lcb_ucb_fail
F28_support_balance_fail
F29_stableaccept_patch_exhausted
F30_empty_step_runtime_fail
F31_empty_event_semantics_fail
F32_no_event_preservation_fail
F33_active_step_launch_count_fail
F34_runtime_still_fragmented
F35_step_ratio_fail
F36_memory_ratio_fail
F37_runtime_measured_path_missing
F38_diagnostic_promoted_to_official
F39_diagnostic_downstream_leak
F40_source_gap_or_formula_proxy_used
F41_projection_used_for_official
F42_leave_dataset_out_fail
F43_leave_stratum_out_fail
F44_paired_replay_control_equivalent
F45_shuffle_control_pass
F46_functional_lr_equivalent
F47_short_run_task_drop
F48_full_run_no_macro_or_hard_stratum_gain
F49_sample_efficiency_fail
F50_continual_forgetting_fail
F51_strong_baseline_explains_gain
F52_robustness_fail
F53_external_not_ready
F54_fake_or_proxy_violation
F55_artifact_missing
```

---

# 11. Route decision

```text
R1-BoundaryReanalyzed:
  v9.2.80-v9.2.82 metrics reproduced and gate gaps computed.

R2-CandidateActionLifecycleFrozen:
  canonical candidate/action set and payload hashes are stable.

R3-SecondaryOutcomeReady:
  secondary deltas, matched controls, coverage and cost materialized for official sample.

R4-OracleWeakFrontierPass:
  current candidate/action population has deployability-level oracle frontier.

R5-OracleStrongFrontierPass:
  current candidate/action population has high-margin oracle frontier.

R6-OracleWeakFrontierFail:
  even oracle cannot reach weak frontier at required coverage.

R7-PrimitiveAutopsyPass:
  oracle fail or fragile frontier is attributed to concrete action primitive failure modes.

R8-OGPFeaturePass:
  legal OGP features show stable value/risk/null predictivity with cost feasibility.

R9-OGPFeatureFail:
  legal features cannot improve over weak RSF-level signal or cost is infeasible.

R10-MinimalOGPDecisionPass:
  cross-fitted dataset-agnostic minimal OGP controller passes decision gates.

R11-StableAcceptPatchExhausted:
  StableAccept and all registered patches fail on canonical candidate/action set.

R12-EventDrivenRuntimePass:
  online event-driven runtime reaches step_ratio_q90 <= 1.50 and empty-event semantics pass.

R13-EventDrivenRuntimeFail:
  runtime remains above envelope after measured event-driven implementation.

R14-SystemLegalControllerPass:
  decision + runtime + contracts + minimality + diagnostic isolation pass.

R15-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R16-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R17-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R18-PairedReplayFail:
  system is legal but functional action is control-equivalent.

R19-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R20-FullFunctionalPass:
  full run task / geometry / system / control / robustness gates pass.

R21-ContinualFunctionalPass:
  retained accuracy / forgetting / transfer gates pass.

R22-CandidateActionPrimitiveInsufficient:
  oracle frontier absent or primitive autopsy shows action generation insufficient.

R23-PrimitiveResetPass:
  redesigned primitive restores oracle weak/strong frontier.

R24-ExternalReady:
  strict PureKAN functional route passes final external-ready gates.
```

`route_decision.json` 必须记录：

```text
route
base_candidate
candidate_lifecycle_pass
action_lifecycle_pass
candidate_count
action_count
event_count
candidate_set_frozen
action_set_frozen
payload_hash_mismatch_count
action_payload_missing_count
secondary_outcome_ready
missing_secondary_delta_count
official_sample_coverage
matched_control_count_per_event
oracle_weak_frontier_pass
oracle_strong_frontier_pass
oracle_precision
oracle_coverage
oracle_bad_event
oracle_null_rate
primitive_autopsy_pass
primary_primitive_blocker
ogp_feature_pass
best_feature_group
best_feature_auc_bad
best_feature_auc_safe
feature_leaveout_drop
feature_cost_q90_ms
feature_memory_ratio
minimality_gate_pass
feature_group_count
monotone_sign_consistency_pass
ogp_decision_pass
controller_id
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
support_balance_pass
stableaccept_patch_exhausted
runtime_candidate_id
runtime_mode
empty_step_controller_kernel_count
empty_event_semantics_pass
no_event_preservation_pass
controller_launches_per_active_step_q90
step_ratio_q90
memory_ratio
event_driven_runtime_pass
diagnostic_downstream_used_for_controller
system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
sample_efficiency_pass
continual_pass
robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9300_strict_purekan_functional
success_v9300_full_functional
success_v9300_external_ready
fake_data_used
proxy_row_used
cpu_offload_used
uses_dataset_name_for_controller
uses_loss_backward
uses_teacher
uses_loss_modification
```

---

# 12. 停止条件

## Minimum diagnostic success

```text
P0 boundary reanalysis pass；
P1 candidate/action lifecycle freeze measured；
P2 secondary outcome materializer measured with coverage/cost；
P3 oracle weak/strong frontier measured；
P3.5 primitive autopsy measured；
P4 OGP feature factory measured with minimality/cost；
P5 OGP controller measured；
P7 event-driven runtime measured with empty-event semantics；
no fake/proxy/offload/teacher/loss violation。
```

## Pivot to candidate/action primitive reset

如果：

```text
P3 weak oracle frontier fail
```

则停止 controller work，进入：

```text
P3.5 primitive autopsy；
P13 candidate/action primitive reset；
new value-producing event source redesign。
```

## Pivot to feature/observability redesign

如果：

```text
P3 weak oracle pass
but P4 OGP feature fail
```

说明 candidate/action population 有潜力，但 legal pre-commit observability 不够。下一步修 feature/observable primitive，不调 threshold。

## Pivot to controller redesign

如果：

```text
P4 feature pass
but P5 decision fail
```

说明 feature 有信息，但 calibration/support/minimal decision rule 不稳定。下一步修 cross-fitting、support、monotone decision rule。

## Pivot to runtime redesign

如果：

```text
P5 decision pass
but P7/P8 runtime fail
```

停止 controller tuning，专注 measured event-driven runtime。

## Open official causal validation

只有当：

```text
P8 system pass
and P10 LDO/LSO pass
```

才 official 打开 P11 paired replay。

## Full functional success

只有当：

```text
P11 paired replay pass
and P12 short/full/continual/robustness pass
```

才能声明 full functional success。

---

# 13. 最终建议

v9.3.0 的一句话策略是：

$$
\boxed{
\text{停止 StableAccept 小修小补；先判断当前 action primitive 是否有 oracle frontier，再用 minimal legal OGP controller 和 measured event-driven runtime 闭合 system path。}
}
$$

具体执行原则：

```text
1. StableAccept 只保留为 feature / prefilter / baseline。
2. 先 freeze candidate/action lifecycle，否则所有跨轮比较都不干净。
3. 补 secondary outcomes，并要求 official sample coverage / matched controls / cost 记录。
4. oracle frontier 分 weak / strong；weak fail 就停止 controller tuning。
5. oracle fail 后必须做 primitive insufficiency autopsy，不能只写 primitive insufficient。
6. OGP features 必须 minimal、legal、cost-feasible、leave-out stable。
7. OGP controller feature groups <= 5，必须 ablation 和 monotone sign audit。
8. runtime 改成 sparse event-driven official mode，并证明 empty-event semantics。
9. diagnostic paired replay 可以加快实验，但不得影响 controller / threshold / feature selection。
10. dataset 只能用于 diagnostics / leave-out，不能用于 controller branch。
11. 如果 OGP 也失败，要诚实 pivot 到 candidate/action primitive redesign，而不是继续调阈值。
```

v9.3.0 成功不一定意味着项目终极成功；但它必须让项目摆脱 `StableAccept patching` 和潜在的 `OGP feature patching`，进入真正可证伪的 action primitive / controller / runtime / causal validation 闭环。
