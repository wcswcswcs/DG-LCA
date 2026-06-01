# DG-KAN v14.12.1 Functional-Update Continue-Open / Transfer Observability / All-Basis Parallel 完整计划

生成时间：2026-05-30（Asia/Singapore）

> 本计划是对 v14.11 结果与 v14.12 计划边界的修正版。核心修正：**functional update 继续推进，但不再以旧式 action/token/controller 搜索推进；promotion gate 继续 fail-closed，exploration gate 必须 continue-open。**
>
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。
>
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed init；functional direction 不使用 validation/test/future/query batch；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate，不能生成 direction；不新增 action token；不启动 controller；不使用 action bank；不把 diagnostic / real-lite / synthetic S3 / substrate eligibility 写成 promotion。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个局部 positive 的 update。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN / PureKAN-like efficient substrate 上，通过 functional update 获得比 ordinary backprop / AdamW 更好的训练几何与模型。}
}
$$

最终 claim 必须同时满足：

```text
1. 表达力不打折，至少不弱于同规模 MLP / same substrate AdamW；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹不比 MLP / same substrate AdamW 慢；
4. functional update 的收益必须击败 NoOp / RandomMatched / AdamWParallel / generic optimizer controls；
5. 几何更健康：source、AUCtime、LineC、tail、calibration 同位；
6. functional update 不能依赖 validation/test/future/query，也不能用 audit metric 生成方向。
```

真正要证明的是：

$$
\boxed{
\text{PureKAN substrate + functional update}
>
\text{same substrate + ordinary AdamW / backprop controls}
}
$$

而不是只证明：

```text
1. 某个 synthetic task pass；
2. 某个 real-lite row positive；
3. 某个 substrate 过 gate；
4. MLP generic optimizer 有正信号；
5. action/oracle/controller 有局部上界。
```

## 0.2 当前已知事实

当前最新落盘结果仍是 v14.11 复核状态，而不是 v14.12 execution success。v14.11 的合法结论是：

```text
route = R1-SyntheticGateNotPredictiveForRealTransfer
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5/7
real_dataset_seed_pass_count = 2/9
line_e_pass = 0
line_v_pass = 0
line_fche_executed = 0
line_d_v1411_route = R8-NonRATSubstrateStillMissing
line_d_v1411_best_family_dataset_seed_pass_count = 0/9
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
```

v14.11 已证明：

```text
1. D-CHE substrate 9/9 是真实进展；
2. D-CHE-FMS synthetic 5/7 是真实进展；
3. 但 synthetic S3 不能可靠预测 real transfer；
4. 当前 U_train / split-batch train-stream proxy 不能可靠预测 synthetic 或 real pass；
5. 新增 D-FOU/RBF/WAV substrate-only hardening 仍为 0/9；
6. S4/S5 仍未打开；
7. 不能继续在 v14.11 内新增 F-CHE8/F-CHE9 / action / controller / reset route。
```

## 0.3 当前各线完成度

| 线 | 完成度 | 当前判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | artifact 与 forbidden audit 基本闭合，不是当前 blocker |
| Historical FHQ / B320-current | 85% frozen | 历史强，但 label-informed init 已禁用，不作为 official base claim |
| Line C 几何审计 | 88% | 审计稳定，但只能做 gate / audit，不能生成方向 |
| PopRisk / FMS infrastructure | 94% | persistent state、per-example gradient、streaming memory、projection telemetry 成熟 |
| Generic MLP-FMS / MLP control | 35%-40% | 只能作为 generic/control 线，不能写成 KAN-specific success |
| Rational substrate | 85% | 稳定，但 reset / optimizer-state route 被 generic confound 打回 |
| Rational-FMS causal specificity | 10%-15% | v14.9 后降级为 no-regression monitor |
| D-CHE substrate | 85% | 当前最强 Non-RAT substrate，9/9 eligibility |
| D-CHE-FMS synthetic | 55%-60% | best synthetic 5/7，S3 已打开 |
| D-CHE-FMS real transfer | 10%-15% | best real 2/9，S4/S5 未打开 |
| Synthetic-to-real predictivity | 5% | 已诊断，当前 AUC/Spearman 不足 |
| Train-stream transfer observability | 5% | U_train 与 split-batch proxy 均失败，需要重建 |
| D-FOU substrate | 45%-55% | v14.9 有 6/9 历史信号，v14.11 新候选 0/9，需复核 |
| D-RBF / FastKAN substrate | 40%-50% | v14.9 有 6/9 历史信号，v14.11 新候选 0/9，task-health 不稳 |
| D-WAV substrate | 15%-25% | v14.11 新候选仍 0/9，低预算保留 |
| Non-RAT official FMS proof | 0%-5% | D-CHE eligible，但 real transfer 未成；FOU/RBF/WAV 不能 proof |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 38%-46% | 有 substrate 和 synthetic functional signal，但 real-transfer value 未闭合 |

解释：整体完成度不能因为 D-CHE synthetic 5/7 而大幅上调。v14.11 已证明当前 synthetic gate 不预测 real transfer，所以当前真正进展是问题定位，而不是能力闭合。

---

# 1. v14.11 独立分析

## 1.1 有进展吗？

有，但不是能力进展，而是 **gate validity 诊断进展**。

v14.11 做对了三件事：

```text
1. 没有把 D-CHE synthetic S3 写成 real success；
2. 没有新增 F-CHE8/F-CHE9 / action token / controller；
3. 明确测了 synthetic-to-real predictivity 与 train-stream value observability。
```

关键数据：

```text
Line E:
  rows = 3780
  positive_real_pass_rows = 168
  spearman_synthetic_source_real_source = -0.00516569843263
  auc_predict_synthetic_features_to_real_pass = 0.543990270527
  predictivity_gate_pass = 0

Line V:
  synthetic U_train AUC = 0.362215061463
  real U_train AUC = 0.473837209302
  U_DCHE_minus_U_MLP = -0.0517916755588
  split_batch_probe_auc_U_train_to_synthetic_pass = 0.394617496121
  split_batch_control_explains_fms_count = 3
  Line V pass = 0

Line D:
  candidate_rows = 99
  best_family_dataset_seed_pass_count = 0/9
  exploration_open_family_count = 0
  official_fms_eligible_family_count = 0
```

因此当前结论是：

$$
\boxed{
\text{当前 synthetic gate 和当前 train-stream proxy 都不能指导 real transfer。}
}
$$

## 1.2 为什么仍感觉慢？

因为 final target 没动：

```text
S4/S5 = 0
real transfer = 2/9
Line E/V = 0
Non-RAT new hardening = 0/9
promotion = 0
```

但慢的本质已经不是 Codex fail-fast，而是 measurement interface 不对：

```text
1. synthetic task-family pass 不能预测 real transfer；
2. train-stream U_train proxy 不能预测 synthetic/real pass；
3. split-batch B1/B2 one-step counterfactual 也不能预测；
4. D-FOU/RBF/WAV 新候选没有给出 alternative substrate。
```

## 1.3 当前真正 blocker

当前 blocker 不是：

```text
D-CHE substrate 不够；
FMS 没有 synthetic signal；
LineC 主导失败；
代码没跑；
real gate 太严；
缺一个 F-CHE8 token。
```

当前 blocker 是：

$$
\boxed{
\text{我们不知道什么 train-stream-only quantity 能预测或驱动 real-transfer value。}
}
$$

更具体：

```text
1. Synthetic S3 是局部 surrogate，不是 real-transfer surrogate；
2. U_train 是弱 proxy，甚至低于 MLP/generic control；
3. 当前 D-CHE FMS value 可能是 synthetic-specific；
4. real transfer 的 source / AUCtime / tail 组合在 train-stream 上不可见；
5. 如果继续小修 F-CHE，会重蹈 v9/v12 “寻找好动作”错误。
```

---

# 2. 本轮核心制度修正

## 2.1 Promotion fail-closed

任何 official promotion 仍必须严格满足：

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
no-action-search audit pass
forbidden audit pass
code review pass
controls cannot explain gain
promotion_allowed = 1
```

## 2.2 Exploration continue-open

这些情况 **不能** 作为全局 hard stop：

```text
Line E2 predictivity < 0.70；
Line V2 proxy AUC < 0.70；
D-CHE real-lite < 6/9；
D-FOU/RBF/WAV 新候选 0/9；
MLP/generic control 有正信号但不能解释 KAN-specific。
```

它们只能决定：

```text
不能 promotion；
不能降低 gate；
必须进入预注册 fallback ladder；
必须输出 failure taxonomy。
```

## 2.3 Hard stop 只给真正违规

只有以下情况才 hard stop：

```text
1. no_action_search_violation_count > 0；
2. forbidden_information_violation_count > 0；
3. required_artifact_missing_count > 0；
4. direction 使用 validation/test/future/query；
5. direction 使用 LineC/CEp99/NLL/ECE/AUCtime/Brier audit target；
6. dataset-name branch；
7. seed-specific scale；
8. synthetic/real-lite/diagnostic 被写成 promotion。
```

---

# 3. 文档启发与新假设

## 3.1 Generalization theory 启发

`A Theory of Generalization in Deep Learning` 的启发是：有效训练信号来自 per-example gradient 的 drift-vs-diffusion，而不是手造几何目标。核心形式：

$$
A_B = \bar g_B\bar g_B^T - \frac{1}{b-1}\Sigma_B.
$$

对角形式近似：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}.
$$

这说明 PopRisk/FMS 是合理的 signal detector，但 v14.11 说明：**signal detector 还不是 real-transfer value observer**。

## 3.2 Deep Manifold 启发

`Deep Manifold` 的启发是：模型训练是 boundary-conditioned iteration，node covers / coordinates 在训练中移动，fixed-point regions 是训练过程里形成的。这说明 single-step synthetic family 可能不能代表 real training path。

因此，D-CHE-FMS 需要：

$$
\boxed{
\text{real-transfer boundary observability}
}
$$

而不是只靠 synthetic pass。

## 3.3 v14.12.1 总体假设

### H1：synthetic gate 无效，但可能被 transfer-calibrated synthetic battery 替换

当前 X1-X7 5/7 不预测 real。可能原因：

```text
1. X1-X7 不覆盖真实三数据集的 trajectory modes；
2. synthetic pass 主要测 final/source，而 real gate 同时看 AUCtime / tail / calibration；
3. D-CHE degree dynamics 在 synthetic 和 real 中分布不同；
4. synthetic seed/loss coverage 不等价于 real dataset/seed coverage。
```

### H2：train-stream value 不是不存在，而是当前 U_train feature family 太弱

当前 U_train 只是一类简单 proxy。下一步测试更结构化的 train-stream observability：

```text
split-window consistency；
loss-integral proxy；
gradient drift accumulation；
degree-energy movement；
post-FMS recovery lag proxy；
micro-horizon B1/B2 stability；
AdamW/FMS disagreement；
train-tail q95/q99 proxy；
margin p10 / entropy / logit RMS collapse proxy。
```

### H3：D-CHE-FMS 需要 exploration-open real-lite，而不是 promotion-closed synthetic gate

如果 synthetic gate 不预测 real，继续只跑 synthetic 没有意义。v14.12.1 允许 **diagnostic real-lite exploration**，但必须明确：

```text
1. real-lite 不用于 promotion；
2. real-lite 不生成下一步 direction；
3. real-lite 只用于验证预注册 train-stream observability 是否有 value；
4. promotion gate 仍要求 official 3x3 / 9/9。
```

---

# 4. 关键原则：不重蹈 v9 / v12 的错误

## 4.1 禁止 action search

禁止：

```text
F-CHE8 / F-CHE9 token extension；
K-RT / K-AUC / reset route；
action bank expansion；
controller；
基于局部 positive row 继续构造新 action；
根据 Fashion-MNIST seed2 或 KMNIST seed2 写 branch。
```

## 4.2 不把 diagnostic 当 promotion

以下都不能 promotion：

```text
synthetic S3；
real-lite positive；
oracle / action-bank reference；
U_train predictor AUC；
LineC / tail / AUC audit improvement；
Non-RAT substrate eligibility；
MLP generic FMS positive。
```

## 4.3 继续 functional update，但必须换接口

functional update 不停。停止的是：

```text
旧 F-CHE token 搜索；
action/controller/reset route；
audit target 反推方向；
synthetic positive -> real success 的错误转写。
```

继续的是：

```text
transfer observability；
D-CHE FMS definition reset；
bounded real-lite diagnostic；
all-basis substrate repair；
MLP/generic controls；
LineC/tail/calibration audit。
```

---

# 5. 实验设计

## Line R：Provenance / no-action-search / code audit

### 目标

确认本轮没有越界：

```text
1. 没有新增 F-CHE8/F-CHE9；
2. 没有 action token / action bank expansion；
3. 没有 controller；
4. 没有 dataset-name branch / seed-specific scale；
5. 没有 validation/test/future/query direction；
6. 没有用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 做 direction；
7. 没有把 real-lite / diagnostic 写成 promotion。
```

### 必须输出

```text
v1412_1_provenance_audit.csv
v1412_1_no_action_search_audit.csv
v1412_1_forbidden_information_audit.csv
v1412_1_method_surface_manifest.csv
v1412_1_code_review_packet.zip
```

### Hard stop

```text
任何 forbidden / action-search / missing artifact 违规，route = R0-ContractViolation。
```

---

## Line E2：Synthetic-to-real transfer observability rebuild

### 目标

不再验证旧 synthetic gate 是否成立；而是寻找哪些 synthetic / telemetry / train-stream features 能预测 real transfer。

### 数据来源

汇总并对齐：

```text
v14.3 Rational synthetic/real；
v14.4 Rational real short-run；
v14.5 oracle / missing-state audit-only reference；
v14.9 D-CHE substrate；
v14.10 D-CHE synthetic/real；
v14.11 Line E/V；
v14.12.1 新 real-lite diagnostic。
```

### 特征组

必须记录：

```text
synthetic_source_vs_control
synthetic_AUCtime_proxy
synthetic_CEp99_delta
synthetic_NLL_delta
synthetic_ECE_delta
synthetic_LineC_pass
synthetic_failure_pattern
D-CHE degree_energy_delta
high_degree_fraction_delta
degree_entropy_delta
value_retention_after_degree_projection
cos_projected_vs_generic
projection_rejection_fraction
update_norm_fraction
degree_low_high_energy_ratio
train_loss_integral_proxy
recovery_lag_proxy
```

### 分析

对每个 feature family 计算：

```text
Spearman(feature, real_source)
AUC(feature -> real_pass)
leave-dataset-out AUC
leave-seed-out AUC
calibration curve
false-positive / false-negative table
```

### Gate

Exploration gate：

$$
AUC_{predict} \ge 0.60.
$$

Promotion-enabling gate：

$$
AUC_{predict} \ge 0.70,
$$

$$
Spearman \ge 0.30.
$$

### Failure handling

若 AUC < 0.60：

```text
不能停止全局实验；
必须继续 Line V2；
route 标记 R1-SyntheticGateWeakButExplorationContinues。
```

---

## Line V2：Train-stream transfer observability expansion

### 目标

重建合法 train-stream-only real-transfer value interface。

### 允许输入

只允许：

```text
current train batch / train split B1/B2；
current train logits；
current train loss per example；
current train margins；
current train entropy；
current train logit RMS；
per-example gradient SNR；
FMS state；
D-CHE degree telemetry；
AdamW state norm / update norm / update cosine；
split-half agreement；
micro-horizon train loss on B1/B2。
```

禁止：

```text
validation / test / future outcome / query batch；
LineC hard target；
CEp99 / NLL / ECE / AUCtime audit target；
dataset-name branch；
seed-specific scale。
```

### Proxy family

#### V2-A：split-window agreement value

$$
U_{split} = Agreement(\Delta\theta_{B1},\Delta\theta_{B2}) - \lambda_r RMSDrift - \lambda_e EntropyCollapse.
$$

#### V2-B：micro-horizon loss-integral value

在 train split B1 上做 bounded micro-horizon probe，不提交权重；在 B2 上计算：

$$
U_{integral} = -\sum_{s=1}^{h} L_{B2}(s) - \lambda_q q95Loss_{B2}(h) + \lambda_m margin_{p10,B2}(h).
$$

#### V2-C：drift-diffusion group utility

对 group $r$：

$$
U_r = \mu_r^T M_r^{-1}\mu_r - \frac{1}{b-1}\operatorname{tr}(M_r^{-1}\Sigma_r).
$$

#### V2-D：recovery-lag proxy

FMS update 后若 train loss 短期 spike 并恢复慢，则 AUCtime 易坏。定义：

$$
Lag = \sum_{s=1}^{h}\max(0,L_{B2}(s)-L_{B2}(0)).
$$

proxy：

$$
U_{lag} = -Lag - \lambda_r RMSDrift - \lambda_h EntropyCollapse.
$$

### Gate

Exploration gate：

$$
AUC(U_{train} \rightarrow synthetic\_pass) \ge 0.60
$$

或：

$$
AUC(U_{train} \rightarrow real\_pass) \ge 0.60.
$$

若二者都低于 0.60，不停止全局实验；进入 Line F-Diag，route 标记：

```text
R2-TrainStreamValueWeakButDiagnosticRealLiteAllowed
```

Promotion-enabling gate：

$$
AUC \ge 0.70
$$

且 controls 不能解释。

---

## Line F-Diag：D-CHE FMS real-transfer diagnostic exploration

### 目标

在不新增 action token 的前提下，验证预注册 FMS definitions 是否有 real-transfer potential。

Line F-Diag 是 exploration，不是 promotion。

### 允许 methods

只允许这些预注册 method families：

```text
FMS-D1: generic value + degree constraint as safety only
FMS-D2: train-stream transfer utility gated FMS
FMS-D3: continuous low-amplitude degree-phase FMS
FMS-D4: micro-horizon loss-integral gated FMS
FMS-D5: recovery-lag suppressed FMS
```

说明：D4/D5 不是 F-CHE8/F-CHE9 action token。它们是 v14.12.1 预注册 transfer-observability FMS definitions，方向仍来自 train-stream generic loss interface / FMS state；degree telemetry 只作 safety / trust region。

### 禁止

```text
1. 不允许 method 根据 dataset/seed 分支；
2. 不允许读取 LineC / CEp99 / NLL / ECE / AUCtime 做 trigger；
3. 不允许使用 real failure pattern 设计 direction；
4. 不允许启动 controller；
5. 不允许 action bank。
```

### Real-lite protocol

为了避免 v14.11 过早停止，允许 real-lite diagnostic：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_steps = fixed short budget
methods = FMS-D1..D5 + controls
promotion_allowed = 0
```

real-lite 只用于估计：

```text
1. 是否有 method family 达到 >=4/9 或 >=5/9；
2. 是否明显超过 D-CHE AdamW / matched controls；
3. 是否主要失败在 source / AUCtime / tail / LineC / overhead；
4. train-stream proxy 是否能解释 pass/fail。
```

### Exploration success

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean > 0
LineC_fail_count = 0 or reduced
no forbidden violation
```

### S4 exploration

```text
real_lite_pass_count >= 6/9
controls cannot explain
step_time_ratio <= 1.50 exploratory
```

### S5 official

仍然要求：

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
promotion_allowed = 1 only after code/provenance review
```

---

## Line D：All-basis substrate parallel

### 目标

不能因为 D-CHE 主线而停止其他 basis。v14.11 的 Line D 新候选为 0/9，但 v14.9 曾经有 D-FOU / D-RBF 6/9 的历史信号。v14.12.1 要先做 cross-version reconciliation，再决定是否继续 hardening。

### D0：Cross-version substrate reconciliation

比较：

```text
v14.9 D-FOU / D-RBF 6/9 candidates；
v14.11 D-FOU27..30 / D-RBF26..29 0/9 candidates；
v14.3 manual exact kernel path；
v12.35 substrate-health rows。
```

判断差异来自：

```text
1. candidate mismatch；
2. gate mismatch；
3. run budget mismatch；
4. true non-reproducibility；
5. artifact replay / finalizer mismatch。
```

### D1：D-FOU hardening

只允许围绕已有 6/9 lineage：

```text
low-frequency identity residual；
bandwise SNR warmup；
phase-stable band mix；
no-materialize lifetime；
high-frequency quarantine。
```

不允许直接高频扩张。

### D2：D-RBF / FastKAN hardening

只允许：

```text
active center occupancy；
width condition；
compact bump no dense materialization；
identity residual；
Gaussian local K4 task-health。
```

不允许 dense RBF basis materialization。

### D3：D-WAV monitor

Wavelet 当前弱，保留低预算：

```text
triangular support；
scale occupancy；
local support overlap damping；
local-tail coverage audit。
```

### D4：D-CHE no-regression

D-CHE 作为主 substrate，监控：

```text
substrate 9/9 是否保持；
step/memory 是否保持；
degree telemetry 是否健康。
```

### Gate

Substrate exploration：

```text
family_dataset_seed_pass_count >= 6/9
```

Official FMS eligibility：

```text
family_dataset_seed_pass_count = 9/9
```

未过前不得进入 official FMS proof。

---

## Line M：MLP / generic FMS control

### 目标

防止把 generic optimizer 或 FMS signal 写成 KAN-specific。

必须保留：

```text
MLP-AdamW
MLP-GenericFMS
MLP-TransferUtilityFMS
MLP-RandomMatchedNorm
MLP-NoOpMatchedOverhead
```

如果 MLP 也达到与 D-CHE 类似收益，则结论应写成：

```text
generic FMS signal exists;
KAN / D-CHE-specific advantage not established.
```

---

## Line C：Geometry / tail / transfer audit

### 目标

Line C 继续作为 audit，不作为 direction source。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC pass
CEp99
NLL
ECE
Brier
AUCtime
margin_p10
logit_rms
entropy
source_vs_best_control
```

### 新增 transfer audit

```text
real_transfer_failure_mode = source / AUCtime / CEp99 / NLL / ECE / LineC / overhead
synthetic_real_feature_alignment
train_stream_proxy_alignment
D-CHE_degree_energy_alignment
MLP_explains_DCHE_flag
```

---

# 6. 必须记录的 artifacts

```text
v1412_1_route_decision.json
v1412_1_progress_table.csv
v1412_1_provenance_audit.csv
v1412_1_no_action_search_audit.csv
v1412_1_forbidden_information_audit.csv
v1412_1_method_surface_manifest.csv
v1412_1_synthetic_real_predictivity.csv
v1412_1_train_stream_proxy_expansion.csv
v1412_1_real_lite_dche_fms_diagnostic.csv
v1412_1_real_lite_controls.csv
v1412_1_allbasis_reconciliation.csv
v1412_1_d_fou_hardening.csv
v1412_1_d_rbf_hardening.csv
v1412_1_d_wav_monitor.csv
v1412_1_dche_no_regression.csv
v1412_1_mlp_controls.csv
v1412_1_linec_tail_audit.csv
v1412_1_failure_taxonomy.csv
v1412_1_required_manifest.csv
v1412_1_code_review_packet.zip
```

---

# 7. 必须记录的指标

## 7.1 Transfer observability 指标

```text
feature_name
feature_family
spearman_to_real_source
spearman_to_real_pass
auc_to_real_pass
auc_to_synthetic_pass
leave_dataset_out_auc
leave_seed_out_auc
false_positive_rate
false_negative_rate
calibration_slope
calibration_intercept
```

## 7.2 Train-stream proxy 指标

```text
split_agreement
micro_horizon_loss_integral
recovery_lag
q95_train_loss_delta
q99_train_loss_delta
margin_p10_delta
entropy_delta
logit_rms_delta
gradient_snr_mean
gradient_snr_p90
fms_adamw_update_cosine
adamw_state_disagreement
degree_energy_delta
high_degree_fraction_delta
degree_entropy_delta
```

## 7.3 FMS real-lite 指标

```text
method
dataset
seed
source_vs_adamw
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
control_equivalent
forbidden_violation
no_action_search_violation
```

## 7.4 All-basis 指标

```text
family
candidate
family_dataset_seed_pass_count
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
mean_delta_vs_MLP
worst_delta_vs_MLP
AUCtime_ratio
LineC_pass_rate
basis_telemetry_health
failure_reason
```

---

# 8. 必须可视化

```text
fig_synthetic_real_predictivity_auc.svg
fig_synthetic_real_spearman_matrix.svg
fig_train_stream_proxy_auc_matrix.svg
fig_micro_horizon_loss_integral_vs_real_pass.svg
fig_recovery_lag_vs_AUCtime.svg
fig_degree_energy_vs_real_transfer.svg
fig_real_lite_method_heatmap_3x3.svg
fig_real_lite_failure_taxonomy.svg
fig_allbasis_cross_version_reconciliation.svg
fig_allbasis_family_status_matrix.svg
fig_mlp_vs_dche_fms_comparison.svg
```

---

# 9. Route definitions

## S1-TransferObservabilityExplorationOpened

```text
Line E2 或 V2 任一 proxy AUC >= 0.60，
且 real-lite diagnostic 有 >=4/9，
但未达到 official S5。
```

## S2-DCHEDiagnosticRealTransferPositive

```text
D-CHE F-Diag real-lite >= 6/9，
controls 不能解释，
但 official gate 未开。
```

## S3-DCHEOfficialRealTransferCandidate

```text
D-CHE real 3x3 >= 8/9，
但仍缺某个 tail/AUC/LineC/overhead 子门。
```

## S5-OfficialFunctionalSuccess

```text
D-CHE or other basis + FMS 达到 official 9/9，
所有 controls / provenance / code review 通过。
```

## R1-SyntheticGateNotPredictiveButExplorationComplete

```text
Line E2/V2/F-Diag 已执行，
但无 proxy / no real-lite signal。
```

## R2-TrainStreamTransferValueUnobservable

```text
所有合法 train-stream proxy 都低于 AUC 0.60，
controls 解释 FMS，
且 real-lite 无改善。
```

## R3-DCHEFMSRealTransferNoGoCurrentDefinition

```text
D-CHE substrate 与 synthetic S3 保留，
但所有预注册 FMS definitions 均 real transfer <4/9。
```

## R4-AllBasisSubstrateBlocked

```text
D-CHE 之外没有任何 family 达到 >=6/9，
且 D-CHE FMS real transfer 也不成立。
```

---

# 10. Codex 执行要求

Codex 必须按顺序执行：

```text
1. Line R audit；
2. Line E2 synthetic-real predictivity rebuild；
3. Line V2 train-stream proxy expansion；
4. Line F-Diag D-CHE real-lite diagnostic；
5. Line D all-basis cross-version reconciliation + hardening；
6. Line M MLP controls；
7. Line C audit；
8. Line Z route / no-go / next queue。
```

Codex 不许做：

```text
1. 新增 F-CHE8/F-CHE9；
2. 新增 action token；
3. 启动 controller；
4. 使用 action bank；
5. 根据 real dataset/seed fail pattern 写 branch；
6. 用 audit metric 生成 direction；
7. 把 real-lite 写成 promotion；
8. 把 synthetic S3 写成 S4/S5；
9. 把 D-CHE substrate eligibility 写成 FMS success；
10. 把 MLP generic positive 写成 KAN-specific success。
```

---

# 11. Failure handling：Codex 先尝试什么

## Case A：Line E2 AUC < 0.60

先尝试：

```text
1. 加入 degree telemetry interaction terms；
2. 加入 recovery-lag / loss-integral features；
3. 分 synthetic task family 做 leave-family-out；
4. 检查 Rational reference 与 D-CHE split 是否分布不一致；
5. 输出 R1，但继续 Line V2。
```

不允许：

```text
1. 降低 AUC gate；
2. 用 real fail labels 训练 predictor；
3. 用 validation/test/future feature。
```

## Case B：Line V2 proxy < 0.60

先尝试：

```text
1. micro-horizon h=1/2/4 的 loss-integral proxy；
2. split B1/B2 agreement；
3. recovery-lag proxy；
4. AdamW/FMS state disagreement；
5. degree-energy stability proxy。
```

如果仍低于 0.60，不停止；进入 F-Diag，route 标记 R2 candidate。

## Case C：F-Diag real-lite < 4/9

先尝试：

```text
1. 检查 controls 是否解释；
2. 检查是否 source fail 主导；
3. 检查是否 AUCtime fail 主导；
4. 检查是否 tail/calibration fail 主导；
5. 输出 failure taxonomy。
```

不允许新增 method token。

## Case D：D-FOU/RBF/WAV cross-version reconciliation 不一致

先尝试：

```text
1. replay v14.9 candidate exact IDs；
2. 对齐 gate；
3. 对齐 run budget；
4. 检查 finalizer mismatch；
5. 再判定 true non-reproducibility。
```

## Case E：D-CHE substrate regression

先尝试：

```text
1. replay D-CHE17 / D-CHE24 lineage；
2. 检查 candidate mismatch；
3. 检查 degree telemetry；
4. 检查 step/memory regression；
5. 不允许通过 lowering gate 修复。
```

---

# 12. 最终判断标准

v14.12.1 不是为了马上 promotion，而是为了回答：

$$
\boxed{
\text{D-CHE synthetic S3 失败 transfer，}
\text{是因为 synthetic gate 错，train-stream value 不可观测，}
\text{还是 D-CHE-FMS definition 本身不适合 real？}
}
$$

如果 v14.12.1 能回答这个问题，就是实质进展。不能把它写成“又没成功”。但如果 v14.12.1 继续停在：

```text
Line E/V fail 后直接停止，
或者继续 F-CHE token search，
或者继续 action/controller，
```

那就是失败。

---

# 13. 一句话总结

v14.11 证明：当前 synthetic gate 与当前 train-stream proxy 都不能预测 real transfer。v14.12.1 不能继续小修 F-CHE，也不能过早停止。正确路线是：

$$
\boxed{
\text{promotion 继续 fail-closed；exploration 必须 continue-open。}
}
$$

在严格禁止 action search / audit-direction / dataset branch 的前提下，重建 transfer observability，并用 diagnostic real-lite 与 all-basis parallel 继续推动问题闭合。
