# DG-KAN v12.17.2：B320 Locked + Loss-Agnostic Target Visibility / Functional Geometry Rebuild + Implementation Readback Contract 独立分析与下一步计划

> 版本：v12.17.2 execution plan  
> 基于：v12.16 `B320Locked ExplicitSignalReservoirFunctional` 真实结果复盘；v12.17 计划；本版在 Codex implementation readback / code rationale 审计硬门基础上，新增 Critical Code Review Surface / 核心代码审查面硬门  
> 核心原则：B320 base 不再小修；Functional update 必须 loss-agnostic；CE / ECE / CEp99 / Brier 只能作为审计与坏化约束，不能作为 functional direction 的设计目标。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；official KAN path 不依赖 PyTorch `loss.backward()` graph；functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome 或 dataset 名称。

---

# 0. 一句话结论

v12.16 不是没有进展。它把 functional update 的失败原因从“actuator 可能动不了”推进到更准确的判断：

$$
\boxed{
\text{B320 base 已经站住；actuator 也不是完全无效；真正 blocker 是 loss-agnostic target visibility 不足。}
}
$$

更直白地说：

```text
我们现在有一个很强的 PureKAN base；
也能构造会移动 output / sketch / projector 的 perturbation；
但当前 loss-agnostic features 看不到哪些 movement 会稳定降低 NoiseSignalLeak 和 RealSignalReservoirRatio；
因此不能把 movement 变成 functional update 成功。
```

所以 v12.17 不应该继续：

```text
1. 小修 B320 architecture；
2. 增加 sketch_dim / rank / batch_size 后重跑同一 P1；
3. 继续 CouplingR2-only promotion；
4. 继续 BM2 / BM3 / F14 / F15 小网格；
5. 把 CE-specific projector 包装成 functional success；
6. 用单一 dataset/seed/window 的 partial positive 进入 P4。
```

v12.17 应该转向：

$$
\boxed{
\text{先证明 loss-agnostic observables 能否看见 release basis，}
\text{再把 fused primitive actuator 当 executor，}
\text{最后才进入 P4 short-run。}
}
$$

---

# 1. 当前实验结果独立分析

## 1.1 B320 base 已经不是 blocker

v12.16 的 P0 复用 locked B320 anchor，不再小修 base。关键指标仍然是：

```text
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta_vs_mlp = 0.027669270833333332
worst_delta_vs_mlp = -0.001953125
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
LineC_nontearing_pass = 1
```

这说明当前失败不能再解释为：

```text
B320 不够快；
B320 表达力不够；
B320 AUC 轨迹不健康；
B320 几何已撕裂。
```

B320 现在是 functional update 的实验地基。后续只允许做 no-regression monitor，不允许继续靠 base 小修制造“进展”。

## 1.2 P1 失败说明：当前 Line C v2 sketch 看不到 release basis

v12.16 P1 做了四类 loss-agnostic sketch / target reconstruction：

```text
multi_window_gradient
role_conditioned
stable_unstable_output
persistent_projector
```

official 设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
windows = 3,5,10
functional_batch_size = 32
sketch_dim = 12
output_subspace_rank = 3
rows = 864
```

结果很清楚：

```text
best noise abs spearman = 0.04916432196353432
best reservoir abs spearman = 0.15995361757658177
negative_release_precision = 0.0
negative_release_recall = 0.0
p1_pass = 0
```

更关键的是，P1 不是没有 release event：

```text
Noise hard support = 80 / 864
Reservoir hard support = 76 / 864
```

也就是说，数据里确实存在 hard release 样本，但当前 loss-agnostic sketch 无法预测它们。

这点非常重要。它否定的是：

```text
“只要把 CouplingR2 / projector / gradient sketch 再调一调，就能看到 release。”
```

实际情况是：当前 feature family 与 hard release basis 错位。

## 1.3 P1 repair 也失败，说明不是简单维度/样本量问题

repair 设置把 batch、sketch、rank 都加大：

```text
functional_batch_size = 64
sketch_dim = 24
output_subspace_rank = 5
rows = 864
```

但结果仍然失败：

```text
p1_best_noise_abs_spearman = 0.07642684028921456
p1_best_reservoir_abs_spearman = 0.17545880235411232
negative_release_precision = 0.0
negative_release_recall = 0.0
p1_pass = 0
```

这说明当前 blocker 不是：

```text
sketch_dim 太小；
rank cutoff 太窄；
batch_size 太小；
window ensemble 不够。
```

而是：

$$
\boxed{
\text{target construction 本身没有显式捕捉 release basis。}
}
$$

## 1.4 P2 失败说明：explicit target calibration 没成立

P2 直接做 explicit target calibration。official 结果：

```text
p2_pass = 0
p2_strong_pass = 0
p2_noise_sign_accuracy = 0.5252525252525253
p2_reservoir_sign_accuracy = 0.3838383838383838
p2_noise_spearman = 0.06204081632653061
p2_reservoir_spearman = 0.06430426716141002
p2_near_joint_release_rows = 0
```

这说明目标符号都没有校准好。noise sign accuracy 约等于随机，reservoir sign accuracy 甚至低于随机。此时继续构造 B16 candidate 会变成 blind search。

## 1.5 P3 失败说明：不是 rank/condition 数值表面问题，而是 safe actuator combo 不成立

P3 的 actuator response matrix rank / condition 数值并不差，condition 最小约 `5.96`。但是 safe actuator combo gate 仍然是：

```text
safe_actuator_combo_gate = 0 / 9
```

也就是说，没有一个 dataset-seed 行同时满足：

```text
sketch_delta_fro >= 0.01
projector_angle >= 1 degree
logit_drift <= 0.05
```

同时 P3 response matrix 中：

```text
official noise hard support = 0 / 99
official reservoir hard support = 0 / 99
repair noise hard support = 0 / 99
repair reservoir hard support = 0 / 99
```

这说明 current actuator directions 能改变某些 sketch / projector 数值，但没有进入 hard release 区域。v12.15 continuation 曾经说明 fused primitive actuator 可以产生更大的 projector movement；v12.16 进一步说明：movement 本身不等于 release。

## 1.6 P4 正确关闭

v12.16 的 P4 / P4 short-run 正确 gated not-run：

```text
candidate_id = B16-NOT-RUN
fail_reason = P1_estimator_gate_failed;P2_explicit_target_calibration_failed;P3_actuator_response_gate_failed
P4_NOT_OPENED = no_v1216_functional_P3_survivor
```

这是正确纪律。没有 P3 survivor 就跑 P4，会把未过前置因果验证的 movement 写成在线收益，是过去多轮实验最容易跑偏的地方。

## 1.7 Classic No-BSpline 支线不是本轮主 blocker

v12.16 只输出了 Line D family status，没有重新跑新的 family targeted repair。原因合理：

```text
1. v12.15 Line D focused repair 已经补跑，0 FamilyPass，且 coupling_collapse。
2. v12.16 P1/P2/P3 functional gates 已失败。
3. 当前 runner 没有实现新的 family-specific candidate IDs。
4. 不允许把旧 focused candidates 重新包装进 B16/P3。
```

所以 Line D 没有新 pass 不是本轮 functional result 的数据缺口。它仍然是并行 portfolio，但不应该抢 v12.17 的主预算。

---

# 2. 为什么感觉慢：这不是实验量问题，而是问题层级变了

项目早期慢，是因为不知道 base 是否可行；现在慢，是因为要证明一个更难的机制命题：

$$
\boxed{
\text{存在 loss-agnostic functional update，}
\text{在合格 B320 base 上产生独立、可预测、control-resistant 的几何收益。}
}
$$

现在不能靠以下任何单点推进：

```text
final accuracy 好；
CouplingR2 提高；
projector angle 变大；
一个 dataset/seed/window pass；
CE-specific projector 有正效果；
random/control gap 很小但为正。
```

因为 functional update 的主张必须同时成立：

```text
1. loss-agnostic；
2. task non-harm；
3. noise leak 下降；
4. real signal trapped in reservoir 下降；
5. train-probe coupling 不坏，最好改善；
6. tail / ECE / margin 不坏；
7. beat AdamWParallel / RandomMatchedNorm / SNR-only / NoOp controls；
8. 不按 dataset 调参；
9. 不使用 CE vector / label / validation / future outcome 生成方向。
```

这就是为什么它看起来慢。它不是缺少实验，而是在排除一类又一类“看起来像成功但不是机制”的假阳性。

---

# 3. 当前真正的本质问题

v12.16 后，functional blocker 应该被重新定义为：

$$
\boxed{
\text{label-defined release audit 与 loss-agnostic observable 之间存在 observability gap。}
}
$$

`NoiseSignalLeak` 和 `RealSignalReservoirRatio` 是非常有价值的审计指标，但它们本质上依赖 real/noise residual 的定义。functional update 又被硬约束为 loss-agnostic，不允许使用 label / CE / permuted CE / validation 等信息生成方向。于是出现一个核心科学问题：

$$
\boxed{
\text{loss-agnostic geometry observables 是否足够预测或驱动 label-audited release？}
}
$$

如果答案是 yes，functional update 有独立机制空间。  
如果答案是 no，那么 functional update 不能被设计成“直接释放 real-signal reservoir / 降低 noise leak”，只能降级为更弱的 **label-free geometry maintenance**，例如稳定 projector、降低 tail drift、减少 logit tearing，然后由 task optimizer 负责 label signal。

这正是 v12.17 要回答的问题。

---

# 4. v12.17 总目标

v12.17 的整体目标不是“再找一个 functional candidate”，而是：

$$
\boxed{
\text{在 B320 locked base 上，判定 loss-agnostic observables 是否能够支撑 functional update 的 value source。}
}
$$

具体分成三个层级。

## 4.1 Minimum progress

证明下面任一项：

```text
1. 找到一组 loss-agnostic observables，能在 heldout dataset/seed/window 上预测 hard release rows；
2. 或证明当前 observable family 不可见，从而明确 functional direct-release route 不成立；
3. 或找到 label-free geometry maintenance，它不直接优化 release，但能稳定改善 tail / projector / coupling 且不被 controls 解释。
```

## 4.2 Functional diagnostic success

在 clone / one-step / five-step 层面，找到 functional candidate 满足：

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01,
$$

$$
control\_gap \ge 0.005.
$$

并且：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
```

## 4.3 Official re-entry condition

只有当同一 candidate 在 3x3 条件下通过，或至少 `8/9` pass 且 bootstrap CI 支持，才打开 P4 short-run。

P4 仍必须满足：

$$
AUCtime_{func} \le AUCtime_{B320},
$$

$$
CEp99_{func} \le CEp99_{B320}+0.05,
$$

$$
ECE_{func} \le ECE_{B320}+0.02,
$$

$$
T_{amortized,func}/T_{B320}\le 1.05,
$$

并且 beat controls。

---

# 5. 总体实验结构

v12.17 分成六条线，但可以并行执行。

```text
Line A:
  B320 Anchor Monitor。
  只做 no-regression，不再小修 base。

Line C:
  Line C hard audit calibration。
  确保 NoiseSignalLeak / RealSignalReservoirRatio / CouplingR2 的 null 与 variance 稳定。

Line T:
  Loss-Agnostic Target Visibility Atlas。
  新增主线。判定 label-free observables 是否能看见 hard release rows。

Line I:
  Fused Primitive Actuator Response Dictionary。
  把 actuator 当 executor，测它对 label-free proxy 和 hard audit 的响应。

Line B:
  Functional Candidate Construction。
  只用 loss-agnostic target / response dictionary 生成 candidate。

Line D:
  Classic No-BSpline Portfolio Monitor。
  Rational / Chebyshev / Wavelet / RBF / Fourier 只在有新 hypothesis 时推进；B-spline frozen。
```

## 5.1 Codex Implementation Readback / Code Rationale Contract（新增硬门）

本轮不能完全信任 Codex 对代码的隐式理解。过去多轮实验已经多次出现：runner 字段看似正确，但实现路径、旧代码假设、hook/timing 路径、metric 定义、candidate id 语义或 gate 触发条件被误读。因此 v12.17.2 新增一个和指标同等重要的硬门：**任何重要实现和任何重要修改，都必须在实验结果复盘文件中写清楚实现思路。**

这个要求不是额外文档，而是结果复盘的一部分。也就是说，Codex 生成的 `docs/DG-KAN_v12.17*_执行复盘.md` 或 `docs/DG-KAN_v12.17*_结果复盘.md` 必须包含完整的 **Implementation Readback** 章节；缺失、空泛或只写“实现完成”的复盘不能被接受，实验 route 必须降级为：

```text
R0-ImplementationReadbackIncomplete
```

该硬门优先级高于所有 metric pass。即使 P2/P3/P4 指标过了，只要 implementation readback 不完整，也不能 promotion。

### 5.1.1 为什么这是硬门

v12.17 的核心问题是 loss-agnostic target visibility。这里的每个实现细节都会改变结论：

```text
1. Line C 的 projector / reservoir / signal sketch 怎么构造；
2. hard release label 只用于 audit，是否意外进入 direction generation；
3. actuator response dictionary 是否真的调用 fused primitive path；
4. NoOp / Random / AdamWParallel controls 是否共享相同测量窗口；
5. P3 candidate 是否只是改变 logits，而没有改变 primitive-level sketch；
6. P4 是否真的 gated behind P1/P2/P3；
7. timing 是否混入 diagnostic hook；
8. classic family frozen / active status 是否被旧 artifact 覆盖。
```

这些不能只从 CSV 字段猜。Codex 必须把现存代码和新代码的实现逻辑写出来，让人能检查它到底理解了什么、改了什么、没有改什么。

### 5.1.2 结果复盘文件必须新增的章节

每次 v12.17.1 运行后，结果复盘必须包含以下章节，标题必须原样出现，便于 grep 和自动审计：

```text
## Implementation Readback / Code Rationale
### Existing Code Path Understanding
### New Code Implementation Rationale
### Mathematical Objects and Tensor Shapes
### Gradient / Backward / Update Semantics
### Loss-Agnostic and No-Dataset-Branch Audit
### Control and Gate Semantics
### Timing / Memory Measurement Semantics
### Known Ambiguities and Risks
### Files Changed and Diff Intent Table
### Reproduction Commands and Artifact Hashes
Critical Code Review Surface
CR0 Entrypoint / Runner / Route Decision
CR1 B320 Candidate Registry / Model Construction
CR2 FHQ Fused Forward-Backward-Update Kernel
CR3 Manual Optimizer / No-Autograd Semantics
CR4 Line C Coupling / Signal-Reservoir Metrics
CR5 Hard Release Label Audit-Only Isolation
CR6 Loss-Agnostic Visibility Features
CR7 Fused Primitive Actuator Response Dictionary
CR8 Controls and Matched Measurement Windows
CR9 Functional Candidate Construction / Solver
CR10 P4 Short-Run Integration
CR11 Timing / Memory Profiler Semantics
CR12 Provenance / Hash / Artifact Reuse
CR13 Classic No-BSpline Status Monitor
CR14 Dataset-Agnostic / Split Discipline
```

这些章节不能只写概括句，必须有文件路径、函数/类名、数据流、公式或 shape、风险和 gate 关系。

### 5.1.3 Existing Code Path Understanding 必须写什么

Codex 必须在复盘中解释本轮依赖的现存代码，而不是只解释新写的代码。至少包括：

```text
entrypoint script：
  experiments/run_v1217_*.py 或实际 runner 名称；
  每个 CLI flag 的意义；
  哪些 flag 会打开 P1/P2/P3/P4；
  哪些 flag 只是 audit，不允许 promotion。

B320 anchor path：
  B320 candidate id；
  model class / primitive class；
  fused kernel / manual update 路径；
  B320 monitor 从哪个 artifact 读取；
  是否重新训练、复用 checkpoint，还是只做 no-regression。

Line C path：
  CouplingR2 怎么算；
  signal / reservoir projector 怎么构造；
  NoiseSignalLeak 和 RealSignalReservoirRatio 怎么审计；
  hard-release label 如何生成；
  audit-only label 是否被隔离。

Functional candidate path：
  loss-agnostic features 从哪里来；
  candidate direction 如何生成；
  actuator 如何 apply；
  controls 如何构造；
  P3/P4 gate 如何决定。

Classic family path：
  Rational / Chebyshev / Wavelet / RBF / Fourier 当前读取哪些代表 artifact；
  B-spline 是否仍 frozen；
  Line D 是否只 status monitor，还是实际新跑 repair。
```

如果 Codex 没有实际阅读或无法确定某个路径，必须写：

```text
unknown_or_not_inspected = 1
reason = ...
impact = ...
recommended_manual_review = ...
```

不允许把不确定性写成确定结论。

### 5.1.4 New Code Implementation Rationale 必须写什么

对每个新增或修改的文件，结果复盘必须给出如下表格：

```text
file_path
change_type = new / modified / deleted / config-only
main_symbols = class/function/kernel names
why_changed
old_behavior
new_behavior
related_hypothesis
related_gate
loss_agnostic_safety
expected_metric_effect
possible_failure_mode
fallback_if_failed
```

特别是涉及以下内容时必须展开说明：

```text
1. 新的 Line C feature family；
2. 新的 target visibility estimator；
3. 新的 actuator response dictionary；
4. 新的 functional candidate constructor；
5. 新的 fused / Triton / CUDA kernel；
6. 新的 control implementation；
7. 新的 route decision / gate logic；
8. 新的 artifact parser 或 hash/provenance audit。
```

示例格式：

```text
file_path = dgkan/functional/linec_visibility.py
main_symbols = build_loss_agnostic_features, fit_visibility_model
why_changed = add heldout visibility atlas for hard release rows
old_behavior = no visibility model; P1 used fixed sketch families only
new_behavior = builds label-free feature matrix X and audit-only release labels y_release; y_release never enters candidate direction
related_hypothesis = H-T: release basis may be visible from label-free geometry features
related_gate = P2 visibility gate
loss_agnostic_safety = labels used only in audit target; direction_generation_reads_label = 0
expected_metric_effect = heldout AUC / precision / recall for hard release rows
possible_failure_mode = feature leakage, split leakage, label entering candidate construction
fallback_if_failed = freeze P2, route R2-ReleaseUnobservableUnderLossAgnosticFeatures
```

### 5.1.5 Mathematical Objects and Tensor Shapes 必须写什么

凡是涉及 projector、sketch、cotangent、response matrix、actuator 或 fused kernel，必须写出数学对象和 shape。最低要求：

```text
B = update batch size
Q = probe batch size
C = number of classes
S = sketch dimension
R = output subspace rank
P = parameter / actuator dimension or compressed basis dimension
```

例如：

$$
\Delta U_B\in\mathbb{R}^{B\times C},
$$

$$
\Delta U_Q\in\mathbb{R}^{Q\times C},
$$

$$
\Phi_B\in\mathbb{R}^{B\times S},
$$

$$
\hat K_{BB}=\Phi_B\Phi_B^T\in\mathbb{R}^{B\times B}.
$$

对 response dictionary，必须说明：

$$
R_{mj}=\frac{G_m(\theta+\epsilon a_j)-G_m(\theta)}{\epsilon},
$$

其中 $G_m$ 是哪一个 audit metric，$a_j$ 是哪一个 actuator basis。必须写清楚 $R$ 的 shape：

$$
R\in\mathbb{R}^{M\times J}.
$$

如果实现用近似、压缩、低秩或 stop-gradient，必须明确写出近似在哪里。

### 5.1.6 Gradient / Backward / Update Semantics 必须写什么

v12.17.1 仍然要求 official KAN path 不依赖 PyTorch `loss.backward()` graph。复盘必须写清楚：

```text
uses_loss_backward
uses_torch_autograd_graph
uses_torch_autograd_grad
manual_forward_available
manual_backward_available
manual_update_available
fused_kernel_used
saved_tensor_total_bytes
manual_cache_bytes
```

如果某条路径只是 diagnostic，允许使用 autograd reference，但必须标记：

```text
path_status = diagnostic_only
promotion_allowed = 0
```

对每个 new update / actuator，必须解释：

```text
它改的是哪些参数 role；
是否改变 B320 task optimizer；
是否改变 AdamW state；
是否需要 optimizer-state transport；
是否改变 logits only，还是改变 primitive-level sketch/projector；
是否会被 AdamWParallel control 解释。
```

### 5.1.7 Loss-Agnostic and No-Dataset-Branch Audit 必须写什么

复盘文件必须有一张 loss-agnostic audit 表。每个 candidate / feature / actuator 必须记录：

```text
candidate_id
feature_family
actuator_family
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
permuted_label_used_for_direction
validation_used_for_commit
test_used_for_commit
future_outcome_used_for_commit
dataset_name_used_for_commit
label_used_for_audit_only
ce_used_for_audit_only
violation_reason
promotion_allowed
```

硬规则：

```text
如果 label / CE 进入 direction generation：promotion_allowed = 0。
如果 dataset_name 进入 commit/controller：promotion_allowed = 0。
如果 validation/test/future outcome 进入 commit：promotion_allowed = 0。
如果 label 只用于 hard-release audit：允许，但必须 label_used_for_audit_only = 1，并证明 candidate constructor 无法访问该列。
```

### 5.1.8 Control and Gate Semantics 必须写什么

复盘必须解释每个 control 的实现，而不是只报 control 分数：

```text
NoOp：是否完全不改变参数；是否仍计算同样 metrics。
RandomMatchedNorm：norm 匹配哪个 candidate；在哪个参数子空间采样；随机种子如何固定。
AdamWParallel：是否使用同一 batch / same norm / same roles；是否包括 AdamW state。
SNR-only：是否 loss-agnostic；是否使用 labels；是否只作 audit。
ShuffledPayload：shuffle 的对象是什么；是否保持 norm / role / timing。
```

每个 gate 必须写：

```text
gate_name
input_artifacts
exact_thresholds
pass_count
fail_count
not_run_reason
which previous gate blocked it
```

如果 P4 没跑，必须明确写：

```text
P4_NOT_OPENED = <exact upstream fail reason>
```

### 5.1.9 Timing / Memory Measurement Semantics 必须写什么

复盘必须解释 timing 是 architecture timing 还是 diagnostic-hook timing。必须记录：

```text
architecture_step_time_ms
diagnostic_hook_time_ms
online_metric_time_ms
offline_analysis_time_ms
compile_warmup_excluded
profile_warmup_steps
profile_measure_steps
synchronization_points
```

硬规则：

```text
diagnostic_hook_time 不得混入 architecture_step_time。
offline Line C analysis 不得混入 B320 / functional amortized step unless explicitly online。
compile warmup 必须从 steady timing 中排除。
```

### 5.1.10 Known Ambiguities and Risks 必须写什么

Codex 必须主动列出不确定性。每个复盘至少包含：

```text
risk_id
risk_description
affected_files
affected_metrics
risk_level = low / medium / high
how_checked
remaining_uncertainty
recommended_next_check
```

如果没有风险，也必须写：

```text
No known ambiguity after inspection.
```

但不能空着。

### 5.1.11 Files Changed and Diff Intent Table 必须写什么

所有改动必须列入 diff intent table：

```text
file_path
lines_or_symbols_touched
intent
hypothesis
expected_artifacts
rollback_plan
```

特别是 runner / kernels / model primitives / metric code / route decision code / artifact parser，必须列。

### 5.1.12 自动判定 readback 是否合格

新增 artifact：

```text
v1217_implementation_readback_audit.csv
v1217_code_path_map.json
v1217_diff_intent_table.csv
v1217_critical_code_review_manifest.csv
v1217_core_symbol_map.json
v1217_code_semantics_trace.csv
v1217_manual_review_packet.md
v1217_review_blocker_table.csv
```

`v1217_implementation_readback_audit.csv` 至少包含：

```text
required_section
present
nonempty
has_file_paths
has_symbols
has_tensor_shapes
has_gate_link
has_loss_agnostic_statement
has_risk_statement
pass
fail_reason
```

通过标准：

```text
所有 required_section present = 1
所有 required_section nonempty = 1
Implementation sections 中至少 5 个真实 file_path
所有 new/modified file 都出现在 diff intent table
所有 functional candidate 都有 loss-agnostic statement
所有 kernel / projector / response matrix 相关改动都有 tensor shape 或数学对象说明
```

失败时 route 必须是：

```text
R0-ImplementationReadbackIncomplete
```

并且：

```text
p2_visibility_pass = 0
p3_response_pass = 0
p4_open = 0
promotion_allowed = 0
```

除非用户人工确认补充复盘。




## 5.2 Critical Code Review Surface / 核心代码审查面（v12.17.2 新增硬门）

v12.17.1 已要求 Codex 在结果复盘中写 Implementation Readback，但这还不够。Codex 可能写出“看起来合理”的实现说明，却没有真正审查最关键的代码路径。v12.17.2 新增更具体的要求：**Codex 必须明确指出每个关键实现位于哪些文件、哪些 class/function/kernel、哪些 line range，并把这些实现组织成一个可人工审查的 Code Review Packet。**

本节的目的不是让 Codex 多写说明，而是解决一个具体风险：过去很多核心结论来自 runner 字段和 artifact 汇总，但实际核心实现没有被人逐段审查。v12.17 现在的问题是 loss-agnostic target visibility；这里任何一个实现细节错了，都会直接改变科学结论。因此，重要代码必须被定位、解释、审查。

### 5.2.1 新增 hard gate

新增 gate：

```text
critical_code_review_surface_pass = 1
```

它要求：

```text
1. Codex 必须列出 CR0-CR14 的实际代码文件、核心 symbol、line range、代码角色。
2. 每个 critical review item 都必须有 existing-code readback 和 new-code diff intent。
3. 每个 item 必须说明是否需要人工审查。
4. 所有 CR0-CR11 必须默认 requires_manual_review = 1。
5. 如果 Codex 找不到对应代码，必须写 unknown_or_not_inspected = 1，不能猜。
6. 任何 CR0-CR11 为 unknown_or_not_inspected = 1 时，P2/P3/P4 不允许 promotion。
7. 如果关键代码只在 runner wrapper 或 postprocess 中，而没有指向实际 model/kernel/metric implementation，视为 CodeReviewSurfaceIncomplete。
```

新增 route：

```text
R0-CodeReviewSurfaceIncomplete:
  Codex 没有指出关键代码路径、核心 symbol、line range，或者只解释 runner/postprocess，没有解释实际 model/kernel/metric/control implementation。

R0-CriticalCodeUninspected:
  CR0-CR11 任一关键项 unknown_or_not_inspected = 1；允许继续 smoke/diagnostic，但不允许 promotion。

R0-CodeSemanticsMismatch:
  Codex readback 与代码语义、artifact 字段或 gate 行为冲突；本轮结果只能作为 untrusted diagnostic。

R0-WrapperOnlyNoCoreEvidence:
  runner 只读取旧 artifact 或做 postprocess，未真实调用本轮声明的 core code path；不能把 route 写成新机制成功。
```

Promotion 允许条件变为：

$$
implementation\_readback\_pass=1,
$$

$$
critical\_code\_review\_surface\_pass=1,
$$

$$
loss\_agnostic\_contract=1.
$$

如果任一不满足，即使 metric 过了，也只能写：

```text
metric_positive_but_code_untrusted
promotion_allowed = 0
```

### 5.2.2 Codex 必须生成的审查 artifact

新增 artifact：

```text
v1217_critical_code_review_manifest.csv
v1217_core_symbol_map.json
v1217_code_semantics_trace.csv
v1217_manual_review_packet.md
v1217_review_blocker_table.csv
```

其中 `v1217_critical_code_review_manifest.csv` 至少包含：

```text
review_id
component
priority = P0/P1/P2
expected_file_hint
actual_file_path
actual_line_start
actual_line_end
main_symbols
is_existing_code
is_new_or_modified_code
called_by
calls_into
artifact_fields_written
mathematical_object
expected_tensor_shapes
uses_label
uses_ce_vector
uses_validation_or_test_for_commit
uses_dataset_name_branch
uses_loss_backward
uses_autograd_graph
uses_fused_kernel
uses_manual_update
is_wrapper_only
unknown_or_not_inspected
codex_summary
codex_confidence = high/medium/low
requires_manual_review
manual_review_status = pending/pass/fail/not_required
blocker_if_wrong
recommended_manual_check
```

`v1217_core_symbol_map.json` 必须给出真实调用链，而不是只列文件名。格式示例：

```json
{
  "CR4_LineC_Projector": {
    "entrypoint": "experiments/run_v1217_...py::run_p1_linec_calibration",
    "feature_builder": "dgkan/functional/linec_visibility.py::build_loss_agnostic_features",
    "projector_builder": "dgkan/functional/linec_metrics.py::build_signal_reservoir_projectors",
    "metric_writer": "experiments/run_v1217_...py::write_linec_hard_support_csv",
    "artifacts": ["v1217_linec_hard_support.csv", "v1217_linec_null_distribution.csv"]
  }
}
```

`v1217_manual_review_packet.md` 必须按 CR0-CR14 分节，每节包含：

```text
1. Why this code matters
2. Actual files and symbols
3. Minimal code excerpt or line range
4. Data flow
5. Tensor shapes / formulas
6. Gate / artifact fields controlled by this code
7. Loss-agnostic risks
8. What human reviewer should verify
9. What would invalidate the experiment
```

### 5.2.3 我需要重点审查的代码细节

下面这些是必须由 Codex 定位并解释、且默认需要人工审查的代码细节。Codex 可以指出实际文件与我给出的 expected hints 不一致，但不能省略定位。

#### CR0：Entrypoint / runner / route decision

**为什么重要：** 如果 runner 只是 wrapper 或 postprocess，就不能把旧 artifact 派生结果写成新机制成功。v12.17 的 P1/P2/P3/P4 gate 必须由真实代码触发，而不是手填 route。

Codex 必须指出：

```text
expected_file_hint:
  experiments/run_v1217_*.py
  或实际执行入口

必须定位的 symbol:
  main / run / parse_args
  run_p0_anchor_monitor
  run_p1_linec_calibration
  run_p2_target_visibility
  run_p3_actuator_response
  run_p4_short_run
  decide_route / write_route_decision
```

我需要审查：

```text
1. 哪些 CLI flag 打开 P1/P2/P3/P4。
2. P4 是否真的 gated behind P1/P2/P3。
3. route_decision 是否由真实 artifact 计算。
4. 是否有旧 artifact 被直接复用为新结果。
5. 如果是 postprocess wrapper，是否明确标注 wrapper_only = 1。
```

失败时 route：

```text
R0-WrapperOnlyNoCoreEvidence
```

#### CR1：B320 candidate registry / model construction / strict PureKAN invariant

**为什么重要：** B320 是 anchor。若 candidate id 映射错、模型结构不对、non-KAN 参数混入、或 fused path 没启用，后续 functional 结果都不可信。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/models/fc_purekan_primitives.py
  experiments/run_v1283_b109_classic_family_functional_geometry.py
  experiments/run_v1210_b320_functional_classic_nobspline.py
  experiments/run_v1217_*.py

必须定位的 symbol:
  B320 candidate registration
  SimpleFastTaskGeometry / B320 primitive class
  build_candidate / instantiate_model / candidate_id registry
  strict PureKAN parameter manifest
```

我需要审查：

```text
1. B320 id 是否唯一且未被 B321/B314/B109 覆盖。
2. B320 具体 forward 组件是否是 expected FHQ / SimpleFastTaskGeometry path。
3. nonKAN_param_count 是否真实由代码枚举，不是硬编码。
4. edge / branch / projection / gain 参数是否都被 manifest 覆盖。
5. manual update 是否覆盖所有 trainable KAN 参数。
```

#### CR2：FHQ / B320 fused forward + backward + update kernel

**为什么重要：** B320 的效率和 strict path 来自 fused hinge/quadratic workspace。如果 kernel fallback 到 eager/autograd，step/memory 和 gradient 语义都不可信。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/kernels/fused_hinge_quadratic.py
  dgkan/models/fc_purekan_primitives.py

必须定位的 symbol:
  fused forward kernel
  fused delta/readout/projection gradient kernel
  fused or manual AdamW update path
  gradient correctness audit function
```

我需要审查：

```text
1. forward/backward/update 是否真的走 fused/Triton/manual path。
2. 是否调用 torch loss.backward() 或 autograd graph。
3. projection gradient、direct readout、quad readout、classbranch/fixedgain 的 shape 是否一致。
4. mixed precision / dtype 是否影响 correctness。
5. update kernel 是否覆盖 optimizer state，而不是只更新一部分参数。
6. timing profile 是否单独测 architecture path，未混入 diagnostic hook。
```

必须写出 shape，例如：

$$
x\in\mathbb{R}^{B\times D},\quad h\in\mathbb{R}^{B\times H},\quad logits\in\mathbb{R}^{B\times C}.
$$

#### CR3：Manual optimizer / AdamW / no-autograd training semantics

**为什么重要：** official KAN path 不允许依赖 PyTorch `loss.backward()` graph；functional 也不能被 optimizer wrapper 偷换。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/optim/*.py
  experiments/run_v1217_*.py
  dgkan/kernels/fused_hinge_quadratic.py

必须定位的 symbol:
  manual CE delta or task optimizer delta
  manual AdamW state update
  optimizer state allocation
  update coverage audit
```

我需要审查：

```text
1. task optimizer 使用 CE 训练 base 可以存在，但 functional direction 不能使用 CE vector。
2. manual update 是否与 reported step time 一致。
3. controls 是否也使用同样 optimizer/update path。
4. optimizer state 是否在 GPU 上，是否有 CPU offload。
```

#### CR4：Line C CouplingR2 / signal-reservoir projector / hard audit metric

**为什么重要：** 这是 v12.17 的科学核心。若 projector 或 residual 定义错，functional 失败/成功结论都无效。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/functional/linec_metrics.py
  dgkan/functional/manifold_channel.py
  experiments/run_v1217_*.py
  或实际实现位置

必须定位的 symbol:
  compute_coupling_r2
  build_train_probe_displacement
  build_gradient_or_jacobian_sketch
  build_signal_reservoir_projectors
  compute_noise_signal_leak
  compute_real_signal_reservoir_ratio
```

我需要审查：

```text
1. Train batch B 和 probe batch Q 是否分离。
2. CouplingR2 的 ridge fit 是否只用于 audit，不用于 candidate construction unless explicitly loss-agnostic。
3. Signal/reservoir projector 的 eigen cutoff / rank 是否确定且可复现。
4. Projector 是否在 update 前后正确重算。
5. NoiseSignalLeak / RealSignalReservoirRatio 是否只作为 audit hard gate，不进入 loss-agnostic direction generation。
6. bootstrap / threshold sensitivity 是否与 hard support 计算一致。
```

必须写出：

$$
\Delta U_B\in\mathbb{R}^{B\times C},\quad
\Delta U_Q\in\mathbb{R}^{Q\times C},
$$

$$
\Phi_B\in\mathbb{R}^{B\times S},\quad
\hat K_{BB}=\Phi_B\Phi_B^T\in\mathbb{R}^{B\times B}.
$$

#### CR5：Hard release label / audit-only isolation

**为什么重要：** functional update 必须 loss-agnostic。hard release label 可以用于审计 visibility，但不能进入 direction generation 或 commit decision。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/functional/linec_visibility.py
  experiments/run_v1217_*.py

必须定位的 symbol:
  build_release_labels
  write_release_labels_audit_only
  split_visibility_train_holdout
  loss_agnostic_contract_check
```

我需要审查：

```text
1. release label 是否从 label/CE residual 派生。
2. 它是否只写入 audit artifact。
3. candidate direction constructor 是否无法读取该 label。
4. visibility model 如果用 release label 训练，是否仅用于判断 observable 可见性，不用于 P3/P4 commit。
5. 是否有 split leakage 或 dataset-name leakage。
```

如果 hard release label 被用于生成 functional direction，则本轮 route 必须是：

```text
R0-LossAgnosticContractViolation
```

#### CR6：Loss-agnostic target visibility features

**为什么重要：** v12.16 的失败是 observability gap。v12.17 必须证明 label-free features 是否能看见 release basis。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/functional/linec_visibility.py
  dgkan/functional/loss_agnostic_features.py
  experiments/run_v1217_*.py

必须定位的 symbol:
  build_loss_agnostic_features
  fit_visibility_model
  evaluate_visibility_leaveout
  feature_ablation
```

我需要审查：

```text
1. feature 输入是否只来自 logits、unlabeled output movement、gradient sketch、projector geometry、role stats。
2. 是否包含 label、CE vector、permuted label、validation/test/future outcome。
3. leaveout 是否按 dataset/seed/window 正确分层。
4. AUC/precision/recall 是否对 hard release rows 计算。
5. feature ablation 是否能指出真正来源，而非黑箱拼接。
```

#### CR7：Fused primitive actuator response dictionary

**为什么重要：** v12.15/v12.16 已说明 movement 不等于 release。必须确认 actuator response 真的作用在 primitive-level sketch，而不是只改 logits。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/functional/actuator_response.py
  dgkan/kernels/fused_hinge_quadratic.py
  experiments/run_v1217_*.py

必须定位的 symbol:
  build_actuator_basis
  apply_actuator_delta
  rollback_actuator_delta
  measure_actuator_response
  build_response_matrix
  solve_safe_combo
```

我需要审查：

```text
1. actuator 是否调用 B320 fused primitive path。
2. apply/rollback 是否精确，不污染 base checkpoint。
3. response matrix 是否在重新 forward / recompute projector 后计算。
4. safe combo gate 是否包含 sketch_delta_fro、projector_angle、logit_drift。
5. response matrix 是否只是 logit movement proxy。
```

必须写出：

$$
R_{mj}=\frac{G_m(\theta+\epsilon a_j)-G_m(\theta)}{\epsilon},
$$

$$
R\in\mathbb{R}^{M\times J}.
$$

#### CR8：Control implementations and matched measurement windows

**为什么重要：** functional 只有打过 controls 才有意义。NoOp / RandomMatchedNorm / AdamWParallel / SNR-only 必须在相同 checkpoint、batch、window、event budget 上测量。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/functional/controls.py
  experiments/run_v1217_*.py

必须定位的 symbol:
  build_noop_control
  build_random_matched_norm_control
  build_adamw_parallel_control
  build_snr_only_control
  evaluate_controls_same_window
```

我需要审查：

```text
1. controls 是否共享同一个 B320 checkpoint / batch / probe / window。
2. norm matching 是否按 same role / same parameter subset。
3. AdamWParallel 是否是同 step task update direction，而不是弱控制。
4. RandomMatchedNorm 是否有 seed/provenance。
5. controls 是否有相同 timing/accounting。
```

#### CR9：Functional candidate construction / constrained solver

**为什么重要：** 这是最容易偷偷用 CE、release labels 或 posthoc score 的地方。

Codex 必须指出：

```text
expected_file_hint:
  dgkan/functional/candidates.py
  dgkan/functional/functional_solver.py
  experiments/run_v1217_*.py

必须定位的 symbol:
  construct_functional_candidates
  solve_constrained_response
  apply_candidate_direction
  candidate_selection_rule
```

我需要审查：

```text
1. candidate 是否只使用 loss-agnostic target / response dictionary。
2. 是否使用 hard release label 参与 solve。
3. objective 中每一项来自哪里。
4. constraints 是否包括 logit drift、tail risk、holdout/task-safety。
5. candidate selection rule 是否预注册，不是按结果挑选。
6. 单一 dataset/seed/window positive 是否被禁止 promotion。
```

#### CR10：P4 short-run integration / event schedule / overhead accounting

**为什么重要：** P3 通过不等于 P4 成功。P4 需要真实在线训练，不允许用诊断 hook 或 validation commit 偷换。

Codex 必须指出：

```text
expected_file_hint:
  experiments/run_v1217_*.py
  dgkan/functional/p4_short_run.py

必须定位的 symbol:
  run_p4_short
  functional_event_schedule
  apply_online_functional_event
  write_p4_event_log
  write_p4_efficiency
```

我需要审查：

```text
1. P4 是否真的 only after P3 survivor。
2. event schedule 是否 dataset-agnostic。
3. commit decision 是否不使用 validation/test/future outcome。
4. functional overhead 是否计入 amortized time。
5. P4 controls 是否同步运行同等 event schedule。
6. P4 是否记录 rejected events / accepted events / no-op equivalent events。
```

#### CR11：Timing / memory profiler and diagnostic hook separation

**为什么重要：** 早期多轮实验出现过 diagnostic hook 混入 architecture timing。v12.17 必须避免把 offline Line C 分析混进 step ratio。

Codex 必须指出：

```text
expected_file_hint:
  experiments/run_v1217_*.py
  dgkan/profiling/*.py
  或实际 profiler 文件

必须定位的 symbol:
  profile_architecture_step
  profile_online_functional_overhead
  reset_peak_memory_after_warmup
  synchronize_timing_scope
```

我需要审查：

```text
1. compile warmup 是否排除。
2. diagnostic_hook_time 是否单独记录。
3. offline analysis 是否未混入 architecture_step_time。
4. synchronization points 是否明确。
5. timing baseline 是否和 controls 同口径。
```

#### CR12：Provenance / artifact reuse / hash / no fake-proxy-cpu audit

**为什么重要：** v12.17 会复用 B320 locked artifact。复用可以，但必须可追溯，不能把旧结果冒充新实验。

Codex 必须指出：

```text
expected_file_hint:
  experiments/run_v1217_*.py
  dgkan/utils/provenance.py
  docs/result recap generator

必须定位的 symbol:
  write_hash_manifest
  verify_source_artifact_hash
  no_fake_proxy_cpu_audit
  artifact_reuse_policy
```

我需要审查：

```text
1. source artifact 的 hash 是否记录。
2. reused vs newly computed artifact 是否分开。
3. fake_data_used / proxy_row_used / cpu_offload_used 是否真实检查。
4. old artifact fields 是否被覆盖成 current route。
```

#### CR13：Classic No-BSpline status monitor

**为什么重要：** B-spline 已 frozen，Line D 不应抢 functional 主预算，也不能被旧 artifact 误覆盖。

Codex 必须指出：

```text
expected_file_hint:
  experiments/run_v1217_*.py
  classic family status parser

必须定位的 symbol:
  read_classic_family_status
  update_family_status
  freeze_bspline_policy
```

我需要审查：

```text
1. B-spline 是否 active_followup = 0。
2. Rational / Chebyshev / Wavelet / RBF / Fourier 是否只在新 hypothesis 下推进。
3. FamilyPass / TaskBlocked / ExpressionBlocked 是否来自对应 artifact，不是硬编码。
```

#### CR14：Dataset-agnostic discipline / split discipline

**为什么重要：** 可以诊断不同数据集暴露的问题，但不能按数据集调参或 route。

Codex 必须指出：

```text
expected_file_hint:
  experiments/run_v1217_*.py
  config / candidate selection / route decision files

必须定位的 symbol:
  dataset loop
  seed loop
  split construction
  candidate selection rule
  route decision
```

我需要审查：

```text
1. dataset name 是否只用于 loop/logging，不用于 candidate decision。
2. train/probe/holdout/test 分离是否正确。
3. visibility leaveout 是否不会把同一 split 泄漏到评估。
4. P3 promotion 是否必须跨 dataset/seed/window。
```

### 5.2.4 Codex 必须在结果复盘中新增的章节

在 v12.17.2 之后，结果复盘除了 v12.17.1 的 Implementation Readback 章节，还必须新增：

```text
## Critical Code Review Surface
### CR0 Entrypoint / Runner / Route Decision
### CR1 B320 Candidate Registry / Model Construction
### CR2 FHQ Fused Forward-Backward-Update Kernel
### CR3 Manual Optimizer / No-Autograd Semantics
### CR4 Line C Coupling / Signal-Reservoir Metrics
### CR5 Hard Release Label Audit-Only Isolation
### CR6 Loss-Agnostic Visibility Features
### CR7 Fused Primitive Actuator Response Dictionary
### CR8 Controls and Matched Measurement Windows
### CR9 Functional Candidate Construction / Solver
### CR10 P4 Short-Run Integration
### CR11 Timing / Memory Profiler Semantics
### CR12 Provenance / Hash / Artifact Reuse
### CR13 Classic No-BSpline Status Monitor
### CR14 Dataset-Agnostic / Split Discipline
```

每个 CR 小节都必须包含：

```text
actual_file_path
actual_line_range
main_symbols
called_by
calls_into
data_flow_summary
tensor_shapes_or_formula
artifact_fields_controlled
loss_agnostic_risk
review_status
remaining_uncertainty
```

### 5.2.5 人工审查策略

Codex 完成 readback 后，不要求一次性人工审查所有代码。为了加快实验，可以分层：

```text
Manual Review Blocker Set A，必须先审，未审不准 P2/P3/P4 promotion：
  CR0, CR1, CR2, CR4, CR5, CR6, CR7, CR8, CR9, CR10, CR11, CR12。

Manual Review Set B，可在 diagnostic 运行后审，但未审不准 official success：
  CR3, CR13, CR14。
```

允许并行：

```text
1. Codex 可以先跑 P0/P1 smoke diagnostic。
2. 但 P2 visibility promotion、P3 actuator survivor、P4 short-run opening 之前，Set A 必须至少完成 code review packet，并且 no unknown_or_not_inspected。
3. 若 Set A 有 high-risk unresolved item，route = R0-CriticalCodeUninspected。
```

### 5.2.6 如果 Codex 找不到代码怎么办

Codex 不允许编造文件名或把“应该在哪里”写成“已经在哪里”。如果找不到，必须写：

```text
unknown_or_not_inspected = 1
expected_file_hint = ...
actual_file_path = NOT_FOUND
reason = ...
impact = ...
recommended_next_action = locate implementation or implement explicitly
```

如果某关键逻辑目前只在 runner 内部 inline 实现，也必须标注：

```text
implementation_style = inline_in_runner
refactor_recommended = 1 if reused by multiple stages
```

如果某关键逻辑目前不存在，但 artifact 字段已经输出，则必须标注为严重问题：

```text
artifact_field_without_core_implementation = 1
route = R0-CodeSemanticsMismatch
```

### 5.2.7 不满足时 Codex 先尝试

```text
如果 CR0 runner/route 不清楚：
  先生成 call graph 和 CLI flag table；
  标注 wrapper-only / true-runner；
  不进入 P2 promotion。

如果 CR1/CR2 B320 path 不清楚：
  先定位 candidate registry、model class、fused kernel、manual update；
  跑 py_compile + gradient correctness smoke；
  不进入 functional candidate。

如果 CR4/CR5/CR6 Line C 与 release label 隔离不清楚：
  先输出 data-flow graph；
  增加 runtime assertion，禁止 release labels 被 candidate constructor 读取；
  不进入 P3。

如果 CR7 actuator path 不清楚：
  先做 apply/rollback identity test、projector recompute test、primitive-sketch movement test；
  不允许只用 logit drift 作为 actuator response。

如果 CR8 controls 不清楚：
  先生成 same-window control audit；
  确认 checkpoint/batch/window/event budget 一致；
  不允许 functional beat weak controls。

如果 CR11 timing 不清楚：
  分离 architecture_step_time、online_functional_overhead、offline_analysis_time；
  重新 profile；
  不允许使用混合 timing。
```


并行策略：

```text
Wave 0:
  同时跑 Line A / Line C null calibration / Line T feature extraction。
  这些互不依赖。

Wave 1:
  若 Line T 有可见性信号，立即并行跑 Line I response dictionary 和 Line B candidate construction。
  若 Line T 无信号，Line B 不进入 P3，只做 observable repair。

Wave 2:
  只对 P3 survivor 跑 P4 short-run。
  Line D 低优先级，不阻塞 Line B。
```

---

# 6. P0：B320 Anchor Monitor 与 Loss-Agnostic Contract

## 6.1 目标

确认 v12.17 没有让 B320 base 回归，也没有让 functional direction 混入 CE / label / dataset branch。

## 6.2 必跑

```text
B320 locked anchor exact monitor
B320 + NoOp
B320 + RandomMatchedNorm audit
B320 + AdamWParallel audit
```

## 6.3 记录 artifact

```text
v1217_anchor_monitor.csv
v1217_loss_agnostic_contract.csv
v1217_provenance_audit.csv
v1217_baseline_controls.csv
v1217_implementation_readback_audit.csv
v1217_code_path_map.json
v1217_diff_intent_table.csv
v1217_critical_code_review_manifest.csv
v1217_core_symbol_map.json
v1217_code_semantics_trace.csv
v1217_manual_review_packet.md
v1217_review_blocker_table.csv
```

## 6.4 必须记录字段

```text
candidate_id
run_id
dataset
seed
window
step_ratio_q90
memory_ratio_q90
mean_delta_vs_mlp
worst_delta_vs_mlp
AUC_step_ratio
AUC_time_ratio
ECE_delta
LineC_nontearing_pass
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
permuted_label_used_for_direction
validation_used_for_commit
test_used_for_commit
future_outcome_used_for_commit
dataset_name_used_for_commit
fake_data_used
proxy_row_used
cpu_offload_used
implementation_readback_pass
implementation_readback_fail_reason
existing_code_path_understanding_present
new_code_rationale_present
diff_intent_table_complete
code_path_map_complete
critical_code_review_surface_pass
critical_code_review_fail_reason
core_symbol_map_complete
manual_review_packet_complete
review_blocker_count
unknown_or_not_inspected_count
wrapper_only_core_evidence_flag
```

## 6.5 通过标准

$$
step\_ratio_{q90}\le 0.50,
$$

$$
memory\_ratio_{q90}\le 0.20,
$$

$$
mean\_delta\ge 0,
$$

$$
worst\_delta\ge -0.003,
$$

$$
AUC\_step\le 1.00,
$$

$$
AUC\_time\le 1.00,
$$

$$
LineC\_nontearing=1.
$$

Loss-agnostic contract：

```text
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
test_used_for_commit = 0
future_outcome_used_for_commit = 0
dataset_name_used_for_commit = 0
```

Implementation readback contract：

```text
implementation_readback_pass = 1
existing_code_path_understanding_present = 1
new_code_rationale_present = 1
diff_intent_table_complete = 1
code_path_map_complete = 1
all_new_or_modified_files_documented = 1
all_important_existing_paths_documented = 1
```

如果 implementation readback 不过，本轮不能进入 P2/P3/P4 promotion，即使 metric 过了也只能作为 untrusted diagnostic。

## 6.6 不满足时 Codex 先尝试

```text
如果 B320 anchor monitor 失败：
  先检查 artifact/hash/protocol 是否错配；
  再复现 v12.14/v12.15 anchor；
  不允许通过修改 B320 architecture 让本轮继续。

如果 loss-agnostic contract 失败：
  立即停止该 candidate；
  输出 violation table；
  不允许把该行降级后继续 promotion。

如果 implementation readback contract 失败：
  不允许继续 promotion；
  Codex 必须先补写结果复盘中的 Implementation Readback / Code Rationale 章节；
  必须补齐 existing code path、new code rationale、tensor shape、gate semantics、loss-agnostic audit、diff intent table；
  补齐前 route = R0-ImplementationReadbackIncomplete。
```

## 6.7 可视化

```text
fig_A_anchor_scorecard.svg
fig_A_anchor_auc_matrix.svg
fig_A_loss_agnostic_contract_heatmap.svg
fig_A_implementation_readback_audit.svg
fig_A_control_null_distribution.svg
```

---

# 7. P1：Line C Hard Audit Calibration

## 7.1 目标

在做 target visibility 之前，先确认 hard audit 的方差、null distribution 与阈值可靠。v12.13 已经看到 NoOp / RandomMatchedNorm 没有 false positive，v12.17 要在 B320 locked anchor 上固定这件事，避免后续把测量噪声当机制。

## 7.2 运行设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
windows = 3,5,10
functional_batch_size = 32,64
sketch_dim = 12,24
output_subspace_rank = 3,5
controls = NoOp, RandomMatchedNorm, AdamWParallel, SNR-only
```

## 7.3 记录 artifact

```text
v1217_linec_null_distribution.csv
v1217_linec_variance.csv
v1217_linec_threshold_sensitivity.csv
v1217_linec_hard_support.csv
```

## 7.4 指标

```text
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
Brier_delta
margin_p10_delta
logit_max_abs_drift
projector_angle
sketch_delta_fro
null_false_positive_rate
random_false_positive_rate
control_hard_release_rate
threshold_sensitivity
bootstrap_ci_low
bootstrap_ci_high
```

## 7.5 通过标准

```text
NoOp hard-pass rows = 0
RandomMatchedNorm hard-pass rows = 0
control false positive rate <= 0.01
Noise hard support >= 5% rows or explicitly mark sparse-support mode
Reservoir hard support >= 5% rows or explicitly mark sparse-support mode
bootstrap std for deltas <= 0.003 preferred
```

## 7.6 不满足时 Codex 先尝试

```text
如果 NoOp/Random 出现 false positives：
  扩大 batch / bootstrap；
  提高 hard gate margin；
  检查 projector recomputation determinism；
  不进入 P2。

如果 hard support 太低：
  增加 candidate perturbation dictionary；
  增加 windows；
  但不改变 release threshold 以制造 support。
```

## 7.7 可视化

```text
fig_C_null_noise_reservoir_hist.svg
fig_C_threshold_sensitivity.svg
fig_C_bootstrap_ci_by_dataset_seed.svg
fig_C_hard_support_heatmap.svg
```

---

# 8. P2：Loss-Agnostic Target Visibility Atlas

## 8.1 目标

v12.16 的核心失败是 target visibility 不足。P2 不生成 functional update，而是回答：

$$
\boxed{
\text{有没有一组 commit-time loss-agnostic features 能预测 hard release rows？}
}
$$

这一步可以使用 hard release label 做离线评价，但不能把 hard release label 或 CE/label 作为 official direction 的输入。它的作用是判断 loss-agnostic observability 是否存在。

## 8.2 Feature families

v12.16 的四类 feature 不够。v12.17 加入六组新的 loss-agnostic feature family。

### F1：output spectral features

只使用 logits / output displacement，不使用 label。

```text
logit_cov_eigenvalues
logit_cov_top_eigen_share
logit_entropy_mean
logit_entropy_p95
pred_margin_mean
pred_margin_p10
class_mean_free_logit_norm
stable_output_subspace_overlap
unstable_output_subspace_overlap
```

### F2：train-probe coupling dynamics

```text
CouplingR2
CouplingCorr
coupling_residual_norm
coupling_split_stability
KernelDrift
projector_angle_t_to_tplus
persistent_projector_overlap
```

### F3：gradient-sketch spectrum

不使用 CE gradient。使用 label-free cotangent probes：

```text
random_rademacher_cotangent
orthogonal_cotangent
class_mean_free_cotangent
logit_whitened_cotangent
augmentation_cotangent
```

记录：

```text
sketch_eigen_top_share
sketch_effective_rank
reservoir_fraction_proxy
dissipation_condition_proxy
sketch_delta_fro
signal_mass_topk
```

### F4：primitive / role response features

```text
direct_role_norm
quad_role_norm
branch_role_norm
direct_quad_ratio
branch_over_total
proj_grad_norm
readout_grad_norm
actuator_response_rank
actuator_condition
role_projector_angle
```

### F5：augmentation consistency features

不使用 label。对同一 batch 做轻微输入扰动：

```text
augment_logit_drift_mean
augment_logit_drift_p95
augment_coupling_stability
augment_projector_stability
augment_entropy_change
```

### F6：time persistence features

跨 window 统计：

```text
feature_persistence_3_5
feature_persistence_5_10
release_proxy_persistence
projector_persistence
sketch_spectrum_persistence
```

## 8.3 Label-free feature matrix

令每个 candidate row 的 loss-agnostic feature 为：

$$
z_i \in \mathbb{R}^m.
$$

Hard release audit label 只用于离线评价：

$$
y_i^{noise}=\mathbb{1}[\Delta NoiseSignalLeak_i \le -0.01],
$$

$$
y_i^{res}=\mathbb{1}[\Delta RealSignalReservoirRatio_i \le -0.01],
$$

$$
y_i^{joint}=\mathbb{1}[y_i^{noise}=1 \land y_i^{res}=1].
$$

训练一个非常简单、可解释的 offline scorer：

$$
s_i = w^\top z_i,
$$

其中 $w$ 只用于 diagnosis，不直接作为 online black-box controller。正式 candidate generation 只能使用 pre-registered feature thresholds / linear score，不允许训练复杂 selector。

## 8.4 数据切分

```text
split_mode = leave-dataset-out, leave-seed-out, leave-window-out
calibration: two datasets / seeds subset
heldout: left-out dataset / seed / window
```

不允许 dataset-specific threshold。所有 threshold 必须在 calibration split 冻结，再到 heldout 评价。

## 8.5 记录 artifact

```text
v1217_target_visibility_features.csv
v1217_release_labels_audit_only.csv
v1217_visibility_scores.csv
v1217_visibility_leaveout.csv
v1217_feature_ablation.csv
```

## 8.6 必须记录字段

```text
row_id
dataset
seed
window
candidate_probe_id
feature_family
feature_name
feature_value
hard_noise_release
hard_reservoir_release
hard_joint_release
score_calibrated
score_heldout
AUC_noise
AUC_reservoir
AUC_joint
precision_at_k_noise
precision_at_k_reservoir
precision_at_k_joint
recall_at_k_noise
recall_at_k_reservoir
recall_at_k_joint
dataset_name_used_for_commit
label_used_for_feature
ce_vector_used_for_feature
```

## 8.7 P2 判断标准

Exploratory visibility pass：

$$
AUC_{joint,heldout}\ge 0.60,
$$

或：

$$
AUC_{noise,heldout}\ge 0.65
\quad \text{and}\quad
AUC_{reservoir,heldout}\ge 0.65.
$$

Hard visibility pass：

$$
Precision_{joint,heldout}\ge 0.25,
$$

$$
Recall_{joint,heldout}\ge 0.20.
$$

Strong visibility pass：

$$
AUC_{joint,heldout}\ge 0.70,
$$

$$
Precision_{joint,heldout}\ge 0.40,
$$

$$
Recall_{joint,heldout}\ge 0.30.
$$

注意：P2 pass 不是 functional success，只是说明有资格构造 P3 candidates。

## 8.8 不满足条件时 Codex 先尝试

```text
如果所有 feature family AUC < 0.55：
  不要继续调 functional lambda；
  加入 augmentation consistency 和 primitive role response features；
  如果仍失败，route = R2-ReleaseUnobservableUnderLossAgnosticFeatures。

如果 coupling features 有 AUC 但 noise/reservoir 无 AUC：
  禁止 CouplingR2-only promotion；
  将 coupling 仅作为 non-collapse / nontearing 约束。

如果 leave-dataset-out 失败但 pooled pass：
  不允许 dataset-specific threshold；
  输出 dataset-shift autopsy；
  尝试 feature normalization by train-stream stats，而不是 dataset name branch。

如果 release support 太薄：
  增加 actuator response dictionary；
  不降低 release threshold。
```

## 8.9 可视化

```text
fig_T_visibility_auc_by_feature_family.svg
fig_T_precision_recall_joint_release.svg
fig_T_leave_dataset_out_matrix.svg
fig_T_feature_ablation_waterfall.svg
fig_T_release_support_umap.svg
fig_T_coupling_vs_noise_reservoir_scatter.svg
```

---

# 9. P3：Fused Primitive Actuator Response Dictionary

## 9.1 目标

v12.16 P3 说明 actuator response matrix rank / condition 不差，但 safe combo 不成立。P3 要把 actuator 从“value source”降级为“executor”，并建立 response dictionary：

$$
a_j \rightarrow
(\Delta z_{LA},\Delta G_{audit})
$$

其中 $a_j$ 是候选 actuator basis direction，$\Delta z_{LA}$ 是 loss-agnostic proxy 变化，$\Delta G_{audit}$ 是只用于审计的 Line C hard metric 变化。

## 9.2 Actuator basis

```text
A0 NoOp
A1 RandomMatchedNorm
A2 AdamWParallel audit only
A3 SNR-only audit only
A4 direct-role primitive spectral shaping
A5 quad-role primitive spectral shaping
A6 branch-role primitive spectral shaping
A7 direct+quad orthogonal residual
A8 branch-orthogonal residual
A9 low-rank mixed primitive actuator
A10 augmentation-consistency actuator
A11 projector-persistence actuator
A12 tail-safe damping actuator
```

注意：

```text
A2/A3 可作为 controls / audit；
official loss-agnostic candidate 不能直接复制 CE/AdamW direction 作为 value claim。
```

## 9.3 Response measurement

对每个 actuator basis，测小步响应：

$$
R_{j,k}=\frac{m_k(\theta+\epsilon a_j)-m_k(\theta)}{\epsilon}.
$$

其中 $m_k$ 包括：

```text
loss-agnostic proxy metrics；
Line C audit metrics；
task/tail safety audit；
efficiency cost。
```

## 9.4 记录 artifact

```text
v1217_actuator_response_dictionary.csv
v1217_actuator_rank_condition.csv
v1217_actuator_safe_combo.csv
v1217_response_to_visibility_score.csv
```

## 9.5 关键字段

```text
actuator_id
role
direction_source
loss_agnostic_direction
functional_norm_ratio
sketch_delta_fro
projector_angle_degree
logit_max_abs_drift
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
Brier_delta
margin_p10_delta
response_rank
response_condition
safe_combo_gate
control_gap_vs_best
```

## 9.6 P3 通过标准

Safe movement gate：

$$
sketch\_delta\_fro\ge 0.01,
$$

$$
projector\_angle\ge 1^\circ,
$$

$$
logit\_max\_abs\_drift\le 0.05.
$$

Audit exploratory release gate：

$$
\Delta NoiseSignalLeak\le -0.005,
$$

$$
\Delta RealSignalReservoirRatio\le -0.005.
$$

Official hard release gate 保持：

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio\le -0.01.
$$

P3 official pass 要求：

```text
>= 6/9 rows pass safe movement gate；
>= 3/9 rows pass exploratory release gate；
heldout score from P2 predicts pass rows better than random；
NoOp/Random controls do not pass。
```

## 9.7 不满足条件时 Codex 先尝试

```text
如果 response_rank 低：
  增加 primitive roles，不增加 CE/label basis；
  加入 augmentation consistency actuator；
  检查 fused primitive kernel 是否实际更新对应 role。

如果 safe movement 过但 release 不过：
  不调大 lambda；
  回到 P2 feature/target visibility，说明 target 不可见或 actuator basis 与 target错位。

如果 logit drift 超标：
  加 tail-safe projection；
  降低 norm budget；
  不使用 CEp99 作为 direction，只作为约束。

如果 control_gap 低：
  去除 AdamWParallel 分量；
  做 AdamW-orthogonal residual；
  但仍不能用 CE direction。
```

## 9.8 可视化

```text
fig_I_response_matrix_heatmap.svg
fig_I_actuator_rank_condition.svg
fig_I_safe_combo_by_role.svg
fig_I_response_vector_pareto.svg
fig_I_release_vs_logit_drift.svg
```

---

# 10. P4：Loss-Agnostic Functional Candidate Construction

## 10.1 目标

只有 P2/P3 都给出可见性和 executor 条件后，才构造 B17 candidate。P4 不能再直接最大化 hard release audit，因为 hard release audit 是 label-defined；P4 只能优化 loss-agnostic proxy，并用 hard release audit 判断是否有效。

## 10.2 Candidate objective

令 $a$ 是低秩 actuator 组合系数。定义 loss-agnostic objective：

$$
J_{LA}(a)
=
-\alpha \Delta ProxyCoupling(a)
+
\beta \Delta ProxyTopEigenShare(a)
+
\gamma \Delta ProxyProjectorInstability(a)
+
\eta \Delta ProxyAugmentDrift(a)
+
\lambda \|a\|^2.
$$

同时满足约束：

$$
LogitDrift(a)\le 0.05,
$$

$$
TailProxy(a)\le \tau_{tail},
$$

$$
FunctionalNormRatio(a)\le \tau_{norm}.
$$

其中所有 proxy 都必须 label-free / CE-free。

审计时再记录：

$$
\Delta CouplingR^2,
$$

$$
\Delta NoiseSignalLeak,
$$

$$
\Delta RealSignalReservoirRatio.
$$

## 10.3 Candidate families

```text
B17a-SpectralFlatteningFunctional
B17b-ProjectorPersistenceFunctional
B17c-AugmentationConsistencyFunctional
B17d-PrimitiveRoleBalancedFunctional
B17e-AdamWOrthogonalLabelFreeFunctional
B17f-CompositeLossAgnosticFunctional
```

解释：

```text
SpectralFlattening:
  降低 sketch top eigen share，增加 effective rank，不使用 label。

ProjectorPersistence:
  降低 adjacent windows projector angle，使几何运动更平稳。

AugmentationConsistency:
  降低小扰动下 logit/projector drift。

PrimitiveRoleBalanced:
  防止 branch-only / coupling-only coordinate move。

AdamWOrthogonalLabelFree:
  只用于避免与 CE/AdamW controls 重合；方向本身仍由 label-free proxy 生成。

Composite:
  由 P2 visibility 选择权重，权重在 calibration split 冻结，heldout 评价。
```

## 10.4 记录 artifact

```text
v1217_functional_candidates.csv
v1217_functional_p3_audit.csv
v1217_functional_controls.csv
v1217_functional_candidate_selection_rule.json
```

## 10.5 必须记录字段

```text
candidate_id
candidate_family
proxy_objective_value
proxy_coupling_delta
proxy_top_eigen_share_delta
proxy_projector_instability_delta
proxy_augment_drift_delta
functional_norm_ratio
logit_max_abs_drift
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
Brier_delta
margin_p10_delta
holdout_CE_loss_ratio_audit
holdout_Brier_ratio_audit
control_gap_vs_NoOp
control_gap_vs_Random
control_gap_vs_AdamWParallel
control_gap_vs_SNR
official_p3_pass
fail_reason
loss_agnostic_direction
```

## 10.6 P4/P3 candidate gate

Row-level gate：

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01,
$$

$$
control\_gap \ge 0.005,
$$

$$
logit\_max\_abs\_drift \le 0.05.
$$

Task/tail audit non-harm：

$$
holdout\_CE\_loss\_ratio \le 1.002,
$$

$$
holdout\_Brier\_ratio \le 1.002,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
ECE\_delta \le 0.02.
$$

Promotion gate：

```text
3x3 all-row pass；
or >=8/9 pass + bootstrap CI lower >= 0；
failure 不集中在单一 dataset；
NoOp / Random / AdamWParallel / SNR-only 不通过同一 gate。
```

## 10.7 不满足条件时 Codex 先尝试

```text
如果 proxy improves but hard audit 不改善：
  route = ProxyAuditMismatch；
  不调 lambda；
  回到 P2 visibility atlas 加 feature family。

如果 coupling improves but noise/reservoir 不改善：
  加强 spectral/top-eigen/noise-null proxy；
  不允许 CouplingR2-only promotion。

如果 noise improves but reservoir 不改善：
  加 projector persistence / low-eigen accessibility proxy；
  不用 label residual release target。

如果 logit drift 超标：
  加 norm trust region；
  尝试 tail-safe projection；
  不把 CEp99 当 objective。

如果 control_gap 不够：
  计算 functional 与 AdamWParallel / SNR-only 的 cosine；
  做 label-free orthogonalization；
  不将 AdamW direction 当 functional success。
```

## 10.8 可视化

```text
fig_B_proxy_vs_hard_audit_scatter.svg
fig_B_functional_pareto_frontier.svg
fig_B_control_gap_by_candidate.svg
fig_B_noise_reservoir_delta_matrix.svg
fig_B_logit_drift_vs_release.svg
fig_B_candidate_fail_reason_heatmap.svg
```

---

# 11. P5：P4 Short-Run Official Re-entry

## 11.1 目标

只有 P4/P3 promotion gate 过，才允许 short-run。v12.16 没有 P3 survivor，所以 P4 正确关闭。v12.17 仍保持这个纪律。

## 11.2 方法

```text
base = B320 locked
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 3 or 5
functional_event_interval = 24,48
functional_candidate = P4 survivor only
controls = NoOpMatchedOverhead, RandomMatchedNorm, AdamWParallel, SNR-only
```

Functional event 使用 low-frequency maintenance，不替代 task optimizer。

## 11.3 必须记录

```text
v1217_p4_short_run.csv
v1217_p4_event_log.csv
v1217_p4_controls.csv
v1217_p4_linec_trajectory.csv
v1217_p4_efficiency.csv
```

字段：

```text
dataset
seed
step
event_id
event_accepted
candidate_id
functional_norm_ratio
amortized_overhead
train_loss
val_loss
AUC_step_ratio
AUC_time_ratio
val_acc_delta_vs_B320
ECE_delta_vs_B320
CEp99_delta_vs_B320
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
control_gap
fail_reason
```

## 11.4 通过标准

$$
AUCtime_{func}\le AUCtime_{B320},
$$

$$
AUCstep_{func}\le AUCstep_{B320},
$$

$$
Acc_{func}\ge Acc_{B320}-0.003,
$$

$$
ECE_{func}\le ECE_{B320}+0.02,
$$

$$
CEp99_{func}\le CEp99_{B320}+0.05,
$$

$$
T_{amortized,func}/T_{B320}\le 1.05,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{B320}-0.01,
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{B320}-0.01.
$$

并且 beat all controls：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallel
SNR-only
```

## 11.5 不满足条件时 Codex 先尝试

```text
如果 P4 one-step pass 但 short-run fail：
  做 P3-to-P4 autopsy；
  检查 effect persistence；
  检查 repeated event 是否导致 projector overcorrection；
  尝试 lower event frequency；
  不做 lambda 小网格优先。

如果 overhead fail：
  移除 runtime backtracking；
  使用 precomputed response dictionary；
  只保留低频 event。

如果 AUC fail：
  判断 AUC-step 与 AUC-time 是否同时坏；
  如果 AUC-step 坏，机制坏；
  如果只 AUC-time 坏，kernel / overhead 坏。

如果 control gap fail：
  停止 official claim；
  只保留 diagnostic。
```

## 11.6 可视化

```text
fig_P4_val_loss_vs_step.svg
fig_P4_val_loss_vs_time.svg
fig_P4_event_accept_timeline.svg
fig_P4_linec_trajectory.svg
fig_P4_control_comparison.svg
fig_P4_overhead_breakdown.svg
```

---

# 12. Line D：Classic No-BSpline Portfolio Monitor

## 12.1 目标

Line D 不替代 B320 functional 主线。B-spline 继续冻结。其余 family 只在有新 hypothesis 时推进，避免重复旧 focused candidates。

当前 status：

```text
Rational: TaskBlocked / GeometryBlocked
Chebyshev: TaskBlocked
Wavelet: TaskBlocked
RBF: ExpressionBlocked
Fourier: ExpressionBlocked
BSpline: RejectedForThisVersion / Frozen
```

## 12.2 v12.17 只允许的新增方向

```text
Rational:
  只允许新 loss-agnostic geometry hypothesis，如 denominator-safe + LineC coupling repair；
  不再重复 B7lu/B7lv 等旧 focused candidates。

Chebyshev:
  只允许修 task trajectory / AUC；
  不再只提高 degree K。

Wavelet:
  只允许 local support / scale diversity 的 task-stable repair；
  不做 Morlet heavy path。

RBF/FastKAN:
  只允许修 A4 expression while preserving existing L3 efficiency；
  不回到 dense RBF。

Fourier:
  只允许 low-frequency + expression capacity repair；
  不做 high-frequency noise-heavy path。
```

## 12.3 记录 artifact

```text
v1217_classic_family_status.csv
v1217_classic_family_new_hypothesis.csv
v1217_classic_family_linec.csv
```

## 12.4 判断标准

每个 family 输出：

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
RejectedForThisVersion
Frozen
```

不得输出“未跑所以失败”。

## 12.5 不满足条件时 Codex 先尝试

```text
Rational A5 fail:
  先看 Line C coupling/reservoir，不先调 LR；
  若 coupling_collapse，尝试 group diversity / denominator-safe geometry；
  若 ECE/AUC fail，记录 calibration，不做 CE-specific loss。

Chebyshev task fail:
  降低 high-degree energy；
  尝试 degree damping；
  不按 dataset 调 degree。

Wavelet task fail:
  检查 scale_dead_fraction 和 local_tail_coverage；
  不直接加重 wavelet family。

RBF/Fourier expression fail:
  增加 compact capacity only if L3 still pass；
  不回 dense materialization。
```

---

# 13. 总 artifact contract

v12.17 必须落盘以下文件。

```text
v1217_route_decision.json
v1217_anchor_monitor.csv
v1217_loss_agnostic_contract.csv
v1217_provenance_audit.csv
v1217_implementation_readback_audit.csv
v1217_code_path_map.json
v1217_diff_intent_table.csv
v1217_critical_code_review_manifest.csv
v1217_core_symbol_map.json
v1217_code_semantics_trace.csv
v1217_manual_review_packet.md
v1217_review_blocker_table.csv

v1217_linec_null_distribution.csv
v1217_linec_variance.csv
v1217_linec_threshold_sensitivity.csv
v1217_linec_hard_support.csv

v1217_target_visibility_features.csv
v1217_release_labels_audit_only.csv
v1217_visibility_scores.csv
v1217_visibility_leaveout.csv
v1217_feature_ablation.csv

v1217_actuator_response_dictionary.csv
v1217_actuator_rank_condition.csv
v1217_actuator_safe_combo.csv
v1217_response_to_visibility_score.csv

v1217_functional_candidates.csv
v1217_functional_p3_audit.csv
v1217_functional_controls.csv
v1217_functional_candidate_selection_rule.json

v1217_p4_short_run.csv
v1217_p4_event_log.csv
v1217_p4_controls.csv
v1217_p4_linec_trajectory.csv
v1217_p4_efficiency.csv

v1217_classic_family_status.csv
v1217_failure_table.csv
v1217_hash_manifest.json

# required sections inside docs/DG-KAN_v12.17*_执行复盘.md or 结果复盘.md
Implementation Readback / Code Rationale
Existing Code Path Understanding
New Code Implementation Rationale
Mathematical Objects and Tensor Shapes
Gradient / Backward / Update Semantics
Loss-Agnostic and No-Dataset-Branch Audit
Control and Gate Semantics
Timing / Memory Measurement Semantics
Known Ambiguities and Risks
Files Changed and Diff Intent Table
Reproduction Commands and Artifact Hashes
Critical Code Review Surface
CR0 Entrypoint / Runner / Route Decision
CR1 B320 Candidate Registry / Model Construction
CR2 FHQ Fused Forward-Backward-Update Kernel
CR3 Manual Optimizer / No-Autograd Semantics
CR4 Line C Coupling / Signal-Reservoir Metrics
CR5 Hard Release Label Audit-Only Isolation
CR6 Loss-Agnostic Visibility Features
CR7 Fused Primitive Actuator Response Dictionary
CR8 Controls and Matched Measurement Windows
CR9 Functional Candidate Construction / Solver
CR10 P4 Short-Run Integration
CR11 Timing / Memory Profiler Semantics
CR12 Provenance / Hash / Artifact Reuse
CR13 Classic No-BSpline Status Monitor
CR14 Dataset-Agnostic / Split Discipline
```

---

# 14. 总判断标准

## 14.1 Route definitions

```text
R0-CodeReviewSurfaceIncomplete:
  result recap missing actual file paths / symbols / line ranges for critical code review surface; no promotion allowed。

R0-CriticalCodeUninspected:
  CR0-CR11 任一关键 code path unknown_or_not_inspected = 1; diagnostics allowed but promotion blocked。

R0-CodeSemanticsMismatch:
  Codex readback contradicts actual code semantics, artifact fields, route gate, or loss-agnostic contract。

R0-WrapperOnlyNoCoreEvidence:
  runner/postprocess exists but no evidence of real core model/kernel/metric/control implementation path。

R0-ImplementationReadbackIncomplete:
  result recap missing implementation readback / code rationale / diff intent / code path map; no promotion allowed。

R0-AnchorRegression:
  B320 anchor monitor failed。

R1-LineCMetricUnstable:
  NoOp / Random false positive or bootstrap variance too high。

R2-ReleaseUnobservableUnderLossAgnosticFeatures:
  hard release support exists, but all loss-agnostic feature families fail visibility gate。

R3-ProxyAuditMismatch:
  loss-agnostic proxy improves, but hard release audit does not improve。

R4-ActuatorExecutorMismatch:
  target visible, but actuator cannot move safe projector/sketch without logit drift。

R5-P3SurvivorFoundP4Opened:
  P3 candidate passes and P4 short-run begins。

R6-FunctionalShortRunFailed:
  P3 survivor exists but P4 cannot beat controls / task non-harm。

R7-OfficialFunctionalSuccess:
  P4 passes task, Line C, controls, and efficiency gates。

R8-ClassicFamilyOnlyProgress:
  no functional progress, but Line D family status improved。
```

## 14.2 Official functional success

v12.17 official functional success 只能是：

$$
P3_{pass}=1,
$$

$$
P4_{pass}=1,
$$

$$
control\_gap\ge 0.005,
$$

$$
loss\_agnostic\_contract=1,
$$

$$
implementation\_readback\_pass=1,

$$
critical\_code\_review\_surface\_pass=1.
$$
$$

并且不能依赖：

```text
CE-specific projector；
label / permuted label；
validation/test/future outcome；
dataset-name branch；
single-row partial positive；
CouplingR2-only improvement。
```

---

# 15. 为什么这个计划是必要的

v12.16 的失败不是普通失败。它告诉我们：

```text
1. base 够强；
2. hard release event 存在；
3. 现有 loss-agnostic sketch 看不到这些 event；
4. 加大 batch / sketch / rank 无效；
5. actuator rank/condition 表面不差，但 safe release combo 不成立；
6. 因此前置因果 gate 必须关闭 P4。
```

这意味着再继续调 functional lambda、window、cotangent K、branch damping 都是在小修。真正的问题是：

$$
\boxed{
\text{我们是否能从不含 label / CE 的几何观测中，看见并控制 label-audited signal/reservoir/noise release？}
}
$$

v12.17 的计划就是直接回答这个问题。它既不会放弃 functional update，也不会把 CE-specific trick 包装成成功；它会先判定 loss-agnostic observability，再进入 actuator response 和 short-run。

---

# 16. 最终执行优先级

第一优先级：

```text
P0/P1/P2：
  B320 no-regression；
  implementation readback / code rationale audit；
  Line C null calibration；
  target visibility atlas。
```

第二优先级：

```text
P3/P4：
  只有 P2 visibility 有信号才构造 functional candidate。
```

第三优先级：

```text
P5：
  只有 P3 survivor 才跑 short-run。
```

第四优先级：

```text
Line D：
  只做新 hypothesis，不重复旧 focused repair。
```

一句话总结：

$$
\boxed{
\text{v12.17 要从“构造 update”转向“证明 loss-agnostic value 可见”。}
}
$$

如果 value 不可见，functional update 的目标必须降级为 label-free geometry maintenance；如果 value 可见，才有资格重开 P4 并冲击 official functional success。
