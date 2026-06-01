# DG-KAN v14.13 Transfer Mechanism Clarification + Functional Continue-Open + All-Basis Parallel 完整计划

生成时间：2026-05-30（Asia/Singapore）

> 本计划基于 v14.12.1 真实结果复盘。核心修正：**继续推进 functional update**，但不回到 v9/v12 之前的“寻找好动作 / action search / controller search / 局部 positive 后扩 token”错误。Promotion 必须 fail-closed；exploration 必须 continue-open。停止条件不能太严苛，除非出现信息泄漏、artifact 缺失、action-search 违规或 fake/proxy 违规，否则 Codex 必须继续执行预注册 fallback ladder。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base/substrate 上，通过 functional update 获得比 ordinary AdamW/backprop 更好的训练几何、收敛与模型。}
}
$$

这里的“更好”不是 MNIST / Fashion-MNIST / KMNIST 打榜，而是机制性 claim：

```text
1. 表达力不打折，不能靠弱模型换稳定；
2. forward / backward / step / memory 与 MLP 可比；
3. 普通 AdamW/backprop 的训练轨迹被 functional update 改善；
4. 改善必须打过 NoOp / Random / AdamWParallel / Generic-FMS / MLP analog controls；
5. 几何改善不是 audit 分数好看，而是 source、AUCtime、tail、calibration、LineC/Signal-Reservoir 不同时坏化；
6. functional direction 必须 loss-interface-generic，不 hardcode CE，不用 validation/test/future/query，不用 LineC/CEp99/NLL/ECE/AUCtime 作为方向源；
7. 不允许 dataset-name branch、seed-specific scale、label-informed initialization、teacher/distillation/loss modification/sampler/class weight。
```

最终要证明的是：

$$
\boxed{
\text{PureKAN substrate + functional update}
>
\text{same substrate + ordinary AdamW / strong controls}
}
$$

而不是：

```text
1. 某个 synthetic task family 过了；
2. 某个 real-lite row 局部 positive；
3. 某个 diagnostic oracle positive；
4. 某个 train-stream proxy AUC 局部好；
5. MLP generic optimizer positive；
6. substrate eligibility 过。
```

---

## 0.2 v14.12.1 当前真实状态

v14.12.1 已经执行 Line R / E2 / V2 / F-Diag / D / M / C / Z。用户继续要求后，补齐了 actual bounded micro-horizon h=1/2/4 train-stream probe，并修正 F-Diag D4 语义，开启 actual_micro_horizon_gating。它不是 fail-fast。

最新合法结论是：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
manifest_missing_sum = 0
```

关键事实：

```text
line_v2_best_auc = 0.6565
line_v2_exploration_gate_pass = 1
line_v2_micro_horizon_probe_rows = 630
real_lite_dataset_seed_pass_count = 1/9
line_d_best_non_dche_dataset_seed_pass_count = 2/9
D-CHE no-regression: no substrate regression detected
D-FOU v14.12 replay dataset-seed pass count = 2
D-RBF v14.12 replay dataset-seed pass count = 1
D-WAV not opened / weak monitor
```

解释：

```text
1. v14.12.1 证明 train-stream observability 不是完全为 0：line_v2_best_auc=0.6565 是弱正信号。
2. 但这个弱可观测信号没有转化成 effective FMS real-lite：F-Diag 只有 1/9。
3. D-CHE substrate 没有回退，但 D-CHE FMS real transfer 仍不成立。
4. D-FOU/RBF/WAV 在 v14.12.1 exact replay 中没有复现到 >=6/9。
5. 当前不能 promotion，也不能在 v14.12.1 内继续新增 method/action/controller/reset route。
```

因此 v14.13 的核心不是“停止 functional update”，也不是“再找好动作”，而是：

$$
\boxed{
\text{澄清 train-stream transfer observability 到 FMS real-transfer 之间的机制断点。}
}
$$

---

# 1. 各线当前进展百分比

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | **99%** | artifact、manifest、forbidden/no-action-search audit 基本闭合 |
| Historical FHQ / B320-current | **85% frozen** | 历史强，但 label-informed init 已禁用，不能 official |
| Line C 几何审计 | **88%** | 审计稳定；只能做 gate/audit，不能做 direction source |
| PopRisk / FMS infrastructure | **94%** | persistent state、per-example gradient、streaming memory、projection telemetry 成熟 |
| Generic MLP-FMS / MLP controls | **35%-40%** | 有 control 价值，但不能写成 KAN-specific success |
| Rational substrate | **85%** | 稳定，但 reset/optimizer-state route 已被 generic confound 打回 |
| Rational-FMS causal specificity | **10%-15%** | v14.9 后降级为 monitor/no-regression |
| D-CHE substrate | **85%** | 当前最强 Non-RAT substrate，9/9 substrate eligibility |
| D-CHE-FMS synthetic | **55%-60%** | best synthetic 5/7，S3 已打开 |
| D-CHE-FMS real transfer | **10%-15%** | best real 2/9；v14.12.1 real-lite 1/9，S4/S5 未打开 |
| Synthetic-to-real predictivity | **5%-10%** | 旧 synthetic gate AUC/Spearman 不足；v14.12.1 V2 有弱 signal |
| Train-stream transfer observability | **15%-20%** | line_v2_best_auc=0.6565，弱正信号；但无法转化为 real-lite success |
| Transfer mechanism realization | **0%-5%** | 当前最大 blocker：proxy 到 FMS effect 的链路断裂 |
| D-FOU substrate | **35%-45%** | v14.9 历史 6/9 未在 v14.12.1 复现；当前 replay 2/9 |
| D-RBF / FastKAN substrate | **30%-40%** | v14.9 历史 6/9 未在 v14.12.1 复现；当前 replay 1/9，task-health 不稳 |
| D-WAV substrate | **15%-25%** | 低预算保留；目前没有官方 FMS eligibility |
| Non-RAT official FMS proof | **0%-5%** | 只有 D-CHE eligible；D-CHE real-transfer 未成，FOU/RBF/WAV 不能 proof |
| Official S5 functional success | **0%** | 尚未达成 |
| 整体 next-gen MLP claim | **38%-46%** | 有 substrate + synthetic signal + weak observability，但 real-transfer mechanism 未闭合 |

---

# 2. 对 v14.12.1 的独立判断

## 2.1 有进展吗？

有，但不是模型能力进展，而是 **transfer-observability 诊断进展**。

v14.11 证明旧 synthetic gate 不可靠：synthetic source 与 real source 的 Spearman 接近 0，预测 real pass 的 AUC 约 0.544；旧 U_train / split-batch proxy 也很弱。v14.12.1 继续开放 exploration 后，Line V2 把 train-stream proxy 做到 `best_auc=0.6565`，这说明：

$$
\boxed{
\text{合法 train-stream signal 不是完全不可见。}
}
$$

但是 v14.12.1 的 F-Diag 仍只有 `1/9`，说明：

$$
\boxed{
\text{可观测性弱正信号没有被当前 FMS definitions 转化成 real-lite value。}
}
$$

这轮真实进展是把问题从：

```text
有没有 train-stream observability？
```

推进到：

```text
为什么 weak-positive observability 不能驱动 FMS real transfer？
```

---

## 2.2 为什么还很慢？

因为最终目标仍然没有推进到 S4/S5：

```text
real-lite = 1/9
S4 = 0
S5 = 0
D-FOU/RBF/WAV replay <6/9
promotion = 0
```

而且 v14.12.1 已经执行过合法 continuation：

```text
1. Line E2 synthetic-real predictivity rebuild；
2. Line V2 train-stream proxy expansion；
3. actual micro-horizon h=1/2/4 probe；
4. F-Diag D-CHE real-lite diagnostic；
5. Line D all-basis reconciliation + hardening；
6. MLP/generic controls；
7. LineC / failure taxonomy；
8. route / no-go / next queue。
```

所以当前慢不是因为 Codex 停得太早，而是因为：

$$
\boxed{
\text{当前 FMS mechanism 无法利用弱 train-stream transfer signal。}
}
$$

---

## 2.3 当前真正 blocker

当前 blocker 不是：

```text
1. D-CHE substrate 不够；
2. FMS infrastructure 不成熟；
3. LineC 主导失败；
4. Codex 没继续；
5. 缺一个 F-CHE8 token；
6. 需要 action/controller/reset。
```

真正 blocker 是：

$$
\boxed{
\text{Transfer observability 与 functional realization 之间存在断点。}
}
$$

更具体：

```text
1. v14.12.1 的 train-stream proxy AUC=0.6565，说明 observability 有弱正信号；
2. F-Diag 1/9，说明当前 D-CHE FMS definitions 不能把该信号变成 real-lite gain；
3. D-CHE no-regression，说明 substrate 没坏；
4. D-FOU/RBF/WAV replay 不到 6/9，说明没有替代 carrier；
5. 若继续 F-CHE token 或 action/controller，会重蹈 v9/v12 action-search 错误。
```

这要求 v14.13 不直接做“更多 FMS methods”，而是先做 **mechanism chain decomposition**：

$$
\text{train-stream proxy}
\rightarrow
\text{FMS event prediction}
\rightarrow
\text{micro-horizon train effect}
\rightarrow
\text{real-lite outcome}
$$

看断点到底在 proxy、projection、degree constraint、event amplitude、timing、还是 D-CHE 参数化本身。

---

# 3. 从文献和历史错误吸收的原则

## 3.1 Generalization paper 的启发

我们继续采用 population-risk / drift-diffusion 观点。per-example gradient 的 batch mean / covariance 给出 population-safe signal：

$$
\bar g_B = \frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B = \frac{1}{b}\sum_i (g_i-\bar g_B)(g_i-\bar g_B)^T,
$$

$$
A_B = \bar g_B\bar g_B^T - \frac{1}{b-1}\Sigma_B.
$$

对 diagonal 情况：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}.
$$

这说明 functional update 的合法信息接口应该是：

```text
current train-stream loss-interface cotangent；
per-example gradient drift/diffusion；
train-stream split agreement；
FMS state 与 optimizer state；
basis telemetry as safety / constraint。
```

不应该是：

```text
LineC hard target；
CEp99/NLL/ECE/AUCtime audit target；
validation/test/future；
dataset/seed branch；
oracle DeltaZ。
```

---

## 3.2 Deep Manifold 的启发

Deep Manifold 的核心启发是：网络训练是 boundary-conditioned iteration，node covers / coordinates 随训练移动，fixed-point regions 在训练中构造出来。对我们而言：

```text
D-CHE degree groups / Rational groups / Fourier bands / RBF centers / Wavelet supports
不是静态标签；
它们是训练中形成的 movable cover / coordinate chart。
```

因此 functional update 不应该是一脚踢参数，而应该是训练过程边界。v14.12.1 的失败说明：我们已有一些合法边界信号，但还没有把它转成 stable real-transfer path。

---

## 3.3 v9/v12 老错误必须写成硬约束

过去错误：

```text
1. 看到局部 positive row 就扩 action；
2. oracle frontier positive 就训练 controller；
3. actuatability pass 当 causality；
4. harmless-null / control-equivalent 当 success；
5. risk/value/support/control 混成一个分数；
6. dataset/seed-specific 修补；
7. runtime/materialization 未闭合就 promotion；
8. P3/synthetic positive 直接推进 P4/S5。
```

v14.13 硬约束：

```text
1. 不新增 action bank；
2. 不启动 controller；
3. 不新增 F-CHE8/F-CHE9 token；
4. 不按 dataset/seed 写 branch；
5. 不用 audit metrics 生成方向；
6. 不把 real-lite 写成 promotion；
7. 不把 weak proxy AUC 写成 success；
8. 不把 D-CHE substrate eligibility 写成 FMS success。
```

---

# 4. v14.13 总体设计

v14.13 不是继续 v14.12.1 的同类小修。它的目标是：

$$
\boxed{
\text{定位 transfer-observability 到 FMS-realization 的断点，}
\text{并在不 action-search 的前提下继续 functional exploration。}
}
$$

分线如下：

```text
Line R: provenance / anti-action-search audit。
Line E3: transfer-observability robust rebuild。
Line V3: proxy-to-effect mechanism chain decomposition。
Line F3: D-CHE FMS mechanism clarification, bounded real-lite。
Line D: all-basis cross-version reconciliation and substrate hardening。
Line M: MLP/generic controls。
Line C: manifold-channel/tail/calibration audit。
Line Z: route / no-go / next hypothesis。
```

---

# 5. Line R：实现与审计硬门

## 5.1 目标

确保 v14.13 不重蹈 action-search / audit-direction / dataset-branch / proxy-success 错误。

## 5.2 必须检查

```text
no_action_search_violation_count
forbidden_information_violation_count
required_artifact_missing_count
uses_validation_test_future_query_for_direction
uses_linec_cep99_nll_ece_auctime_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
uses_label_informed_initialization
uses_teacher_distillation_loss_mod_sampler_class_weight
readout_feature_proxy_only
feature_table_proxy_only
```

## 5.3 Hard stop 条件

只有以下情况允许 hard stop：

```text
1. forbidden_information_violation_count > 0；
2. no_action_search_violation_count > 0；
3. required_artifact_missing_count > 0；
4. direction 使用 validation/test/future/query；
5. direction 使用 LineC/CEp99/NLL/ECE/AUCtime audit target；
6. dataset-name branch / seed-specific scale；
7. fake/proxy/CPU offload。
```

其他 gate fail 只能触发 fallback，不允许 final stop。

---

# 6. Line E3：Transfer-observability robust rebuild

## 6.1 目标

v14.12.1 的 `line_v2_best_auc=0.6565` 说明 train-stream proxy 有弱正信号。Line E3 要判断这个信号是否稳健、是否跨 task/dataset/seed、是否只是局部 artifact。

## 6.2 数据来源

只能读取已存在或本轮合法产生的：

```text
current train batch / train splits B1,B2
current logits
current per-example train loss
current margins / entropy / logit RMS
per-example gradient SNR
FMS state
D-CHE degree telemetry
projection telemetry
AdamW state norm / update norm / update cosine
micro-horizon B1/B2 train loss only
synthetic outcome for diagnostic labels
real-lite outcome for diagnostic labels
```

不能用于生成方向的字段：

```text
validation/test/future/query
LineC
CEp99
NLL
ECE
AUCtime
real dataset/seed branch
```

## 6.3 特征族

```text
E3-A Split-window consistency:
  agreement between B1 and B2 proxy signs.

E3-B Micro-horizon integral:
  train-stream loss integral over h=1/2/4, but only on B1/B2.

E3-C Drift-diffusion group utility:
  group-level U_c = mu_c^T M_c^{-1} mu_c - tr(M_c^{-1} Sigma_c)/(b-1).

E3-D Recovery-lag proxy:
  short-term loss spike and recovery on train split only.

E3-E State disagreement:
  FMS direction vs AdamW direction cosine, update norm, moment mismatch.

E3-F Degree stability:
  degree energy entropy, high-degree fraction, degree drift after projected update.

E3-G Projection value retention:
  cos_projected_vs_generic, value_retention, rejection_fraction.
```

## 6.4 评估协议

不能只报告全局 AUC。必须做 leaveout：

```text
leave-task-family-out
leave-dataset-out
leave-seed-out
leave-loss-interface-out
```

记录：

```text
auc_predict_synthetic_pass
auc_predict_real_lite_pass
spearman_proxy_to_source
precision_at_topk
recall_at_topk
calibration_brier
leaveout_auc_min
leaveout_auc_mean
false_positive_rate_on_controls
```

## 6.5 Gate

Exploration gate：

$$
AUC_{mean} \ge 0.60
$$

and:

$$
AUC_{leaveout,min} \ge 0.55.
$$

Promotion-enabling gate：

$$
AUC_{mean} \ge 0.70,
$$

$$
AUC_{leaveout,min} \ge 0.65,
$$

$$
Spearman \ge 0.30.
$$

如果 E3 exploration gate fail，不允许 promotion，但不能停止；必须进入 Line V3 的断点分解。

---

# 7. Line V3：Proxy-to-effect mechanism chain decomposition

## 7.1 目标

v14.12.1 的问题不是没有 weak proxy，而是 weak proxy 没有变成 real-lite success。Line V3 要拆清楚：

$$
\text{proxy score}
\rightarrow
\text{candidate FMS event}
\rightarrow
\text{actual parameter/projection effect}
\rightarrow
\text{micro-horizon train effect}
\rightarrow
\text{real-lite outcome}
$$

到底哪一步断了。

## 7.2 必须记录

对每个 D-CHE FMS diagnostic row：

```text
proxy_score_before_event
generic_fms_norm
degree_projection_rejection_fraction
value_retention_after_degree_projection
cos_projected_vs_generic
actual_update_norm
actual_degree_energy_delta
actual_high_degree_fraction_delta
B1_loss_delta_h1/h2/h4
B2_loss_delta_h1/h2/h4
B2_q95_loss_delta
B2_margin_p10_delta
B2_logit_rms_delta
B2_entropy_delta
micro_horizon_integral
recovery_lag
matched_random_same_norm_effect
NoOp_matched_overhead_effect
real_lite_pass
```

## 7.3 断点分类

```text
V3-B1 ProxyWeak:
  proxy cannot predict event effect or real-lite.

V3-B2 ProjectionKillsValue:
  proxy predicts generic value, but degree projection rejection too high or retention too low.

V3-B3 EventTooWeak:
  update has correct sign but actual norm/effect too small.

V3-B4 EventTooVolatile:
  B1 improves but B2 worsens, split agreement fails.

V3-B5 MicroHorizonGoodRealBad:
  train-stream micro-horizon improves but real-lite fails.

V3-B6 ControlEquivalent:
  random / NoOp / MLP generic control explains effect.
```

## 7.4 Gate

Line V3 不要求 success。它要求输出断点结论。通过标准是：

```text
breakpoint_coverage = 1
ambiguous_rows_fraction <= 0.20
controls_explained_rows recorded
```

如果 ambiguous > 0.20，Codex 必须继续补充 diagnostics，而不是进入 FMS-D training。

---

# 8. Line F3：D-CHE FMS mechanism clarification, bounded real-lite

## 8.1 目标

继续推进 functional update，但不能走 F-CHE8/F-CHE9 token search。Line F3 只允许预注册的机制定义，并且必须以 Line V3 的断点为前置解释。

## 8.2 允许的机制定义

```text
FMS-M1 GenericValueOnlyDegreeSafety:
  generic FMS value path unchanged;
  degree telemetry only as safety/trust region;
  tests whether basis constraints are killing value.

FMS-M2 DegreeContinuousBoundary:
  persistent state controls degree plasticity continuously;
  no discrete action choice;
  no dataset/seed branch.

FMS-M3 MicroHorizonGatedBoundary:
  uses B1/B2 micro-horizon loss-integral proxy;
  if no train-stream advantage over NoOp, suppress FMS amplitude globally.

FMS-M4 RecoveryLagSuppressedBoundary:
  if train-stream recovery-lag proxy is high, reduce FMS amplitude;
  not based on AUCtime audit.

FMS-M5 ProjectionRetentionFloor:
  commit only if value_retention_after_degree_projection >= pre-registered floor;
  floor is global and not dataset/seed specific.
```

这些是 mechanism definitions，不是 action bank。它们不允许被继续扩成 M6/M7/M8，除非 v14.13 的 Line Z 明确判定旧 mechanism family no-go 且下一版重写计划。

## 8.3 Real-lite 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
D-CHE candidate = no-regression confirmed D-CHE substrate
methods = FMS-M1..M5 + controls
controls = AdamW, NoOpMatchedOverhead, RandomMatchedNorm, GenericMLPFMS, SameActiveFractionControl
train_steps = bounded real-lite budget
no promotion from real-lite
```

## 8.4 Exploration gate

Weak exploration：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
LineC_fail_count = 0 or reduced vs v14.12.1
no forbidden violation
```

Meaningful exploration：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.002
at least two datasets have >=1 seed pass
controls cannot explain majority of pass rows
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUC/tail/LineC non-harm in passed rows
controls fail
step_time_ratio <= 1.50 exploratory
```

Official S5 仍然严格：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
promotion_allowed = 1 only after audit/code review
```

## 8.5 Fallback ladder

如果 F3 <3/9：

```text
1. Do not add methods.
2. Use Line V3 failure taxonomy.
3. If ProjectionKillsValue dominates, report D-CHE degree constraint blocker.
4. If MicroHorizonGoodRealBad dominates, report train-stream proxy insufficient for real transfer.
5. If ControlEquivalent dominates, report FMS value no-go.
6. Continue Line D all-basis; do not stop whole experiment.
```

如果 F3 3/9 到 5/9：

```text
1. Record S4-lite exploration positive.
2. Do not promotion.
3. Run pre-registered repeat seeds/loss-interface confirmation if budget allows.
4. No dataset-specific patch.
```

如果 F3 >=6/9：

```text
1. Open S4 exploration;
2. Run stronger controls;
3. Prepare v14.14 official confirmation plan;
4. Do not claim S5 in v14.13 unless full S5 gate is reached.
```

---

# 9. Line D：All-basis parallel substrate reconciliation

## 9.1 目标

继续所有 active no-BSpline basis，不让 functional 主线吞掉 basis 修复。v14.12.1 显示 D-FOU replay=2/9、D-RBF replay=1/9、D-WAV weak，和 v14.9 历史 6/9 存在差异。必须解释差异。

## 9.2 Line D0：cross-version reconciliation

比较：

```text
v14.9 D-FOU / D-RBF 6/9 candidate IDs
v14.12.1 D-FOU / D-RBF replay IDs
v14.3 manual exact kernel path
v12.35 substrate-health rows
v14.11 D-FOU27..30 / D-RBF26..29 / D-WAV25..27
```

判断差异来自：

```text
candidate mismatch
gate mismatch
run budget mismatch
linec seed mismatch
finalizer mismatch
true non-reproducibility
```

## 9.3 D-FOU

候选方向：

```text
D-FOU-LFIdentityResidual
D-FOU-BandwiseSNRWarmup
D-FOU-PhaseStableBandMix
D-FOU-NoMaterializeLifetimeV2
D-FOU-HighFrequencyQuarantine
```

目标：从 replay 2/9 推回 >=6/9 exploration，再评估是否能向 9/9 substrate 发展。

## 9.4 D-RBF / FastKAN

候选方向：

```text
D-RBF-ActiveCenterOccupancy
D-RBF-WidthConditionGuard
D-RBF-CompactBumpNoDenseMaterialization
D-RBF-IdentityResidual
D-RBF-GaussianLocalK4TaskHealth
```

重点不是 workspace，而是 task-health / source / AUCtime 不崩。

## 9.5 D-WAV

低预算保留：

```text
D-WAV-TriangularSupport
D-WAV-ScaleOccupancy
D-WAV-LocalSupportOverlapDamping
D-WAV-LocalTailCoverageAudit
```

## 9.6 D-CHE

只做 no-regression monitor：

```text
D-CHE no regression rows
D-CHE step/memory
D-CHE source/AUC/tail/LineC under AdamW
D-CHE degree telemetry
```

不再把 D-CHE substrate eligibility 写成 FMS success。

## 9.7 Line D gate

Exploration substrate：

```text
dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
LineC non-catastrophic
mean source/task non-catastrophic
```

Official substrate：

```text
dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
mean_delta_vs_MLP >= -0.005
AUCtime_ratio <= 1.0
LineC_pass = 1
```

未过 substrate gate，不允许 official FMS proof。

---

# 10. Line M：MLP / generic controls

## 10.1 目标

判断任何 FMS signal 是否只是 generic optimizer / MLP analog，而不是 KAN-specific。

## 10.2 必须记录

```text
MLP-AdamW
MLP-GenericFMS
MLP-MicroHorizonFMS
MLP-RecoveryLagFMS
MLP-NoOpMatchedOverhead
MLP-RandomMatchedNorm
```

## 10.3 判断

如果 MLP/generic controls 解释 D-CHE gains：

```text
route = R5-GenericFMSConfound
KAN-specific claim not allowed
```

如果 D-CHE 明显超过 MLP/generic controls：

```text
FMS-KAN-specific exploration allowed
```

---

# 11. Line C：Manifold-channel / tail / calibration audit

Line C 仍然重要，但只做审计。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC pass
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
AUCtime_ratio
source_vs_best_control
```

不能用这些字段生成 direction。它们只用于：

```text
1. pass/fail gate；
2. failure taxonomy；
3. next hypothesis explanation。
```

---

# 12. 必须记录的 CSV/JSON artifacts

v14.13 必须生成：

```text
v1413_route_decision.json
v1413_progress_table.csv
v1413_line_r_audit.csv
v1413_forbidden_information_audit.csv
v1413_no_action_search_audit.csv
v1413_e3_transfer_observability_features.csv
v1413_e3_transfer_observability_summary.csv
v1413_e3_leaveout_predictivity.csv
v1413_v3_proxy_to_effect_chain.csv
v1413_v3_breakpoint_summary.csv
v1413_f3_dche_fms_real_lite.csv
v1413_f3_dche_fms_controls.csv
v1413_f3_failure_taxonomy.csv
v1413_d0_cross_version_reconciliation.csv
v1413_d_fou_hardening.csv
v1413_d_rbf_hardening.csv
v1413_d_wav_monitor.csv
v1413_dche_no_regression.csv
v1413_mlp_controls.csv
v1413_linec_tail_audit.csv
v1413_required_artifact_manifest.csv
v1413_code_review_packet.zip
v1413_no_go_boundary.md
v1413_next_hypothesis_queue.md
```

---

# 13. 必须可视化

```text
fig_e3_proxy_auc_by_leaveout.svg
fig_e3_proxy_calibration_curve.svg
fig_v3_proxy_to_effect_waterfall.svg
fig_v3_breakpoint_taxonomy_bar.svg
fig_f3_real_lite_pass_matrix.svg
fig_f3_source_auc_tail_linec_heatmap.svg
fig_allbasis_cross_version_replay.svg
fig_d_fou_rbf_wav_substrate_matrix.svg
fig_dche_degree_telemetry_no_regression.svg
fig_mlp_vs_dche_fms_controls.svg
fig_route_dashboard.svg
```

---

# 14. Route definitions

```text
S0-ExecutionCompleteNoPromotion:
  runner/artifacts complete but no scientific gate.

S3-DCHESyntheticFMSPass:
  historical D-CHE synthetic 5/7 remains valid, but no real-transfer promotion.

S4-lite-RealLiteExplorationPositive:
  F3 real_lite_pass_count >= 4/9, controls not explaining majority, no promotion.

S4-RealTransferExplorationPositive:
  F3 real_lite_pass_count >= 6/9, controls fail, no promotion.

S5-OfficialFunctionalSuccess:
  full real 9/9 official gate + controls + provenance/code review pass.

R1-TransferObservabilityWeak:
  E3/V3 cannot reach exploration observability, but artifacts complete.

R2-ProjectionOrRealizationBreak:
  proxy positive but projection/event realization breaks.

R3-DCHERelTransferFail:
  D-CHE FMS mechanisms fail real-lite <4/9 after allowed ladder.

R4-AllBasisSubstrateBlocked:
  D-CHE no S4/S5 and no other basis reaches >=6/9 substrate exploration.

R5-GenericFMSConfound:
  MLP/generic controls explain observed gains.

R6-NoGoCurrentFMSDefinitions:
  FMS-M1..M5 all fail and V3 explains blocker as current-definition no-go.

R0-HardStopViolation:
  forbidden/action/artifact/fake/proxy violation.
```

---

# 15. Stop / continue contract

## 15.1 Promotion fail-closed

Promotion 只能在 S5：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
provenance pass
code review pass
promotion_allowed = 1
```

## 15.2 Exploration continue-open

以下情况不能导致 hard stop：

```text
E3 AUC < 0.70；
V3 找到断点但未解决；
F3 real-lite <4/9；
D-FOU/RBF/WAV replay <6/9；
D-CHE FMS real transfer fail；
MLP/generic controls positive；
LineC/tail/AUC fail。
```

必须继续到预注册 fallback：

```text
E3 fail -> V3 breakpoint decomposition；
V3 ambiguous -> add diagnostics, not new methods；
F3 fail -> Line D/M/C/Z still execute；
Line D fail -> cross-version reconciliation and failure taxonomy；
Generic confound -> record R5, do not promotion；
```

## 15.3 Hard stop only for violations

```text
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
required_artifact_missing_count > 0
uses audit metric as direction
uses validation/test/future/query as direction
dataset-name branch / seed-specific scale
fake/proxy/cpu offload
```

---

# 16. Codex 执行顺序

Codex 必须按顺序执行：

```text
1. Line R audit。
2. Line E3 transfer-observability robust rebuild。
3. Line V3 proxy-to-effect mechanism chain decomposition。
4. Line F3 D-CHE bounded real-lite FMS clarification。
5. Line D all-basis cross-version reconciliation + hardening。
6. Line M MLP/generic controls。
7. Line C audit。
8. Line Z route/no-go/next queue。
```

Codex 不许做：

```text
1. 新增 F-CHE8/F-CHE9；
2. 新增 action token；
3. 启动 controller；
4. 使用 action bank；
5. 根据 real dataset/seed fail pattern 写 branch；
6. 用 LineC/CEp99/NLL/ECE/AUCtime 生成 direction；
7. 把 real-lite 写成 promotion；
8. 把 weak proxy AUC 写成 success；
9. 把 synthetic S3 写成 S4/S5；
10. 把 D-CHE substrate eligibility 写成 FMS success；
11. 把 MLP generic positive 写成 KAN-specific success；
12. 降低 official gate。
```

---

# 17. 最终判断

v14.12.1 的真实结论是：

$$
\boxed{
\text{functional update 不能停，但旧 synthetic gate / U_train proxy / current FMS definitions 不能直接带来 real-transfer。}
}
$$

最重要的事实不是 `real-lite=1/9` 本身，而是：

```text
line_v2_best_auc=0.6565 说明 legal train-stream observability 有弱信号；
但 F-Diag 1/9 说明当前 mechanism 不能利用这个信号；
D-CHE substrate no-regression；
D-FOU/RBF/WAV 没有替代 substrate；
继续新增 action/token/controller 会重蹈 v9/v12 旧错。
```

所以 v14.13 的核心是：

$$
\boxed{
\text{promotion fail-closed，exploration continue-open；}
\text{先定位 proxy-to-effect 断点，再决定是否重写 FMS definition 或回 substrate。}
}
$$

