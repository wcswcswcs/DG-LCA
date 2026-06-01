# DG-KAN v12.25：S4-to-S5 Precommit Functional Bridge、Tail-Safe Control-Residual Update、Label-Free Anchor 与 Classic No-BSpline 并行计划

> 版本：v12.25 execution plan  
> 生成目的：基于 v12.24 `S4-to-S5 FunctionalBridge / LabelFreeClassic` 的实际结果，重新设计下一轮实验。  
> 核心修正：Codex 不得在第一个、第二个 gate fail 后用 no-go 停止；必须继续执行预注册的机制级 fallback，直到达到 official success、明确机制级 no-go，或硬预算耗尽。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；official KAN path 不依赖 PyTorch `loss.backward()` graph；functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome 或 dataset 名称。CE / NLL / CEp99 / ECE / Brier 只能作为审计和坏化约束，不能作为 functional direction 的构造目标。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是单纯做一个 KAN，也不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是验证下面这个科学命题：

$$
\boxed{
\text{strict FC-PureKAN base} + \text{loss-agnostic functional update}
>
\text{strict FC-PureKAN base} + \text{ordinary backprop / AdamW controls}
}
$$

这里的“更好”必须同时满足：

```text
1. 表达力不打折，不能靠变弱模型换平滑。
2. forward / backward / step / memory 进入 MLP-like efficiency envelope。
3. 收敛轨迹不比 MLP 慢，AUC-step / AUC-time 不坏。
4. 几何更健康：train-probe coupling、signal/reservoir、noise leakage、tail stability 不坏，最好改善。
5. functional update 的收益必须 loss-agnostic、precommit-safe、control-resistant。
6. 可以诊断不同数据集 failure slice，但不能针对数据集调参或分支。
```

最终可写成论文主张的成功形式是：

$$
Acc_{KAN+FU} \ge Acc_{KAN} - \epsilon,
$$

$$
AUCtime_{KAN+FU} \le AUCtime_{KAN},
$$

$$
GeometryRisk_{KAN+FU} < GeometryRisk_{KAN},
$$

并且：

$$
KAN+FU > \max(\text{NoOp}, \text{RandomMatchedNorm}, \text{AdamWParallel}, \text{SNR-only}, \text{matched train-stream control}).
$$

其中 $FU$ 表示 functional update event。

## 0.2 当前进展地图

截至 v12.24，当前状态可以概括为：

```text
B320-current / FHQ anchor:
  很强，但仍带 label-informed trainprobe initialization 的 claim 限制。

Label-free B320-like base:
  task 接近，但 LineC non-tearing 仍未恢复。

Functional update:
  已经从完全 no-go 推进到 S4：P3 或 audit-P4 局部信号存在；但 S5 official functional success 仍为 0。

Precommit / train-stream functional bridge:
  已经出现 source-vs-NoOp、甚至 source-vs-control 的局部 accuracy 信号；但无法同时通过 matched-control margin、CEp99 tail safety、LineC multi-sketch 和 train-shuffle robustness。

Classic no-BSpline family:
  Rational/Fourier 已做 formal/depth-2 hardening；Rational 接近 task gate但 memory / CouplingR2 / reservoir 不闭合，Fourier 暂时不接近 exploration。
```

## 0.3 v12.24 的最重要事实

v12.24 最终仍是 no-go：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
fallback_all_executed = 1
required_artifact_missing_count = 0
```

这不是成功，但也不是无效失败。它说明：

```text
1. 计划要求的 depth-2 fallback 已经执行，artifact/code review 缺失不再是主要 blocker。
2. precommit/train-stream source 可以产生真实局部 task signal。
3. 这些 signal 还不能同时满足 control margin、tail safety、LineC multi-sketch、train-shuffle robustness。
```

因此 v12.25 的目标不是降低 gate，而是把这些 partial signals 机制化：

$$
\boxed{
\text{把 train-stream / precommit 的局部 task signal 转成 tail-safe、control-resistant、multi-sketch robust 的 S5 候选。}
}
$$

---

# 1. v12.24 结果独立分析

## 1.1 有进展吗？

有。v12.24 的价值不在于成功，而在于它把 functional update 的边界推得更清楚。

过去我们只有：

```text
query-reference I24 可以产生 audit-only P4 gain；
但 provenance 不干净，不能 official。
```

v12.24 之后，我们知道：

```text
1. train-stream / precommit bridge 不是完全没有信号；
2. I30 / NG4 能出现 source 同时赢 NoOp 和 matched control 的近似点；
3. NG9 / NG17 / NG18 说明 direct / direct_gain / quad_direct role policy 可以带来真实 task accuracy margin；
4. 但这些点会在 CEp99、LineC CouplingR2、多 sketch、train-shuffle 或 strict task margin 上失败。
```

这意味着 functional update 不是完全 dead end；它现在卡在多目标闭合，而不是“没有任何作用”。

## 1.2 为什么仍然感觉慢？

因为 v12.24 的进展主要是 **机制边界进展**，不是 **official capability 进展**。每轮都看似做了很多 fallback，但最终还是：

```text
S5 official functional success = no
P4 pass = no
promotion_allowed = 0
```

这个感觉是合理的。当前最难的部分不是 base 或效率，而是证明：

```text
functional update 的收益不是 query-reference 偶然性；
不是 NoOp positive；
不是 matched control 也能做到；
不是 CEp99 tail 风险换来的；
不是某个 train-shuffle seed 偶然；
不是单个 LineC sketch 偶然。
```

这正是从 S4 到 S5 的难点。

## 1.3 v12.24 发现的核心问题

### 1.3.1 Line A：label-free repair 仍未恢复 LineC

A51 仍是最强 label-free task candidate，mean delta 为正、worst delta 小，但 LineC pass rate 为 0。A66 / A68 这种 LineC-aware multi-sketch frame 没有修复 LineC，还明显伤 task/AUC；A67 / A69 明显破坏任务和效率。

结论：

$$
\boxed{
\text{label-free task 可以接近，但 label-free signal/reservoir geometry 仍没有恢复。}
}
$$

下一步不能再继续给 A51 加普通 frame token，而要专门设计 **LineC-residual repair**：只修 A51 相对 B320-current 的 LineC residual，不替换整个 projector。

### 1.3.2 Line I/B：train-stream bridge 有局部 signal，但输在 control / tail / robustness

v12.24 里 I28/I29/I31 没能复制 I24 query-reference 的 S5 级 task+LineC+robustness。I30/T1B-guided direct compensation 可以局部赢 NoOp，但输 matched control。I32 policy-aware probe没有产生 strict pass。

后续 NG4 是关键：precommit train-stream I30 找到一个近似点：

```text
source_vs_noop = +0.01953125
source_vs_control = +0.0078125
LineC pass = 3/5
```

但失败在：

```text
source_CEp99 = 10.7112
noop_CEp99 = 9.3257
strict CEp99 requires source_CEp99 <= noop_CEp99 + 0.05
```

这说明它不是纯噪声；它真的能产生 task/control-positive signal。但该 signal 伴随 hard-tail 风险。

### 1.3.3 Tail repair 出现明确 tradeoff

NG5 提高 weight decay 没能修复 CEp99。NG6 / NG7 label smoothing 可以明显降低 CEp99，但 source-control advantage 消失。

这说明：

$$
\boxed{
\text{普通 smoothing / weight decay 不是解。它们只是把 functional signal 一起抹掉。}
}
$$

v12.25 不能继续把 “CEp99 高” 简化成 “加 smoothing”。必须找到 functional event 内部的 tail-safe role / scale / clipping 机制，并且该机制不能针对 CE 设计。

### 1.3.4 NG17/NG18 给出新的重要边界

expanded I32 policy probe 中，direct_gain / direct_only 在多个 train-shuffle seed 上都能出现 source-vs-NoOp 和 source-vs-control 的正向 accuracy margin，例如：

```text
NG17/NG18 best_source_vs_best_control = +0.01171875
```

但它们 LineC 0/5，CouplingR2 低于 NoOp，NLL / CEp99 略差。结论是：

$$
\boxed{
\text{direct-only / direct-gain 可以提供 task accuracy signal，}
\text{但它损害了 train-probe geometry 与 tail stability。}
}
$$

这提示下一步不能只追 task gain。需要把 direct/gain source 与 quad/reservoir-preserving source 做组合，而不是二选一。

### 1.3.5 T1B / response-distilled 仍没有 positive rows

v12.24 中 T1B microprobe、response-distilled features 扩到更多行后仍没有正例，AUC 不定义。这说明：

```text
T2/T3 completed-response teacher 确实说明“事后看得到信号”；
但当前还没有把它蒸馏成 train-stream / precommit / deployable source。
```

所以后续不能把 T2/T3 当成 functional source。它们只能作为 upper-bound teacher。

### 1.3.6 Line D：Rational 比 Fourier 更值得保留，但不是 S4c

Rational/Fourier formal hardening没有打开 S4c。初始 formal hardening中 Rational mean delta vs MLP 约 -0.189，Fourier 约 -0.231；后续 depth-2 Rational B7em/B7en 等有更多 LineC pass 行和接近 task gate的行，但 memory ratio 约 2.10，MNIST CouplingR2 仍不足。FourierB4p/q/v/w 未接近 exploration。

因此：

```text
Rational：保留为独立 classic-family hardening 主线，重点修 memory / CouplingR2 / cross-dataset。
Fourier：暂时降为低预算 monitor，除非有新 expression/efficiency hypothesis。
B-spline：继续 frozen。
```

---

# 2. 当前是否在正确道路上？

方向是对的，但执行协议仍要升级。

正确的地方：

```text
1. 没有把 query-reference I24 写成 official。
2. 没有降低 P3/P4/S5 gate。
3. 没有把 NoOp-positive 写成 success。
4. 开始要求 train-shuffle、multi-sketch、matched control。
5. classic family 没有被完全放弃，Rational/Fourier 已做 hardening。
6. code packet / fallback manifest / required artifact manifest 比以前完整。
```

不够的地方：

```text
1. Codex 仍然可以在 depth-2 no-go 后停止，而不是继续到机制级 saturation。
2. 当前 fallback 多是 sign/scale/policy/smoothing 级别，仍偏局部。
3. 没有系统化地把 direct task signal 与 quad geometry preservation 组合起来。
4. tail-safe observable 仍只是在排序层面，不进入 update construction。
5. P3/P4 对齐仍不足：P3 pass、P4 task gain、LineC pass、control gap 常常不在同一个 candidate 上。
```

所以 v12.25 必须明确：

$$
\boxed{
\text{S5 不是通过继续扩大 I30/I32 grid 达成，而是通过设计新的 composite functional event 达成。}
}
$$

---

# 3. 离目标还差多远？

| 模块 | 当前完成度 | 判断 |
|---|---:|---|
| B320-current diagnostic anchor | 85% | 强，但 label-informed trainprobe claim 限制仍在 |
| Label-free B320-like base | 45% | task 接近，LineC 仍 0 |
| LineC audit | 80% | 审计可靠性较高，multi-sketch必要 |
| Precommit/train-stream source | 40% | 已有 source-vs-control 近似点，但 tail/control/robustness不闭合 |
| Actuator / role policy | 55% | direct/gain有task signal，quad有geometry潜力，但组合未闭合 |
| Functional P3 | 60% | 已经不是 0；S4 opened 过，但不等于 S5 |
| Functional P4 official | 20% | audit-only/near-pass存在，official为0 |
| Classic no-BSpline family | 40% | Rational有保留价值，Fourier低优先 |
| Overall next-gen MLP claim | 60% | base强，functional仍是核心缺口 |

最重要的差距不是效率，而是：

$$
\boxed{
\text{precommit-safe functional source 的 task gain、control gap、LineC、tail safety、robustness 尚未同位。}
}
$$

---

# 4. v12.25 总体实验目标

v12.25 的总目标是从 v12.24 的 depth-2 no-go 推进到机制级决策：

```text
要么找到 S5-ready functional bridge；
要么证明当前 train-stream compensation family 和 role-policy family 需要结构性替换。
```

具体目标：

```text
G1. 设计 composite functional event：direct/gain 提供 task signal，quad/reservoir component 保持 LineC 和 tail stability。
G2. 构建 loss-agnostic tail-safe precommit observable，不读 label/CE/LineC target。
G3. 将 T2/T3 response-level teacher 蒸馏成真正 precommit micro-probe，而不是 post-hoc scorer。
G4. 对 NG4 / NG9 / NG17 / NG18 近似点做机制化组合，而不是继续 sign/scale 小网格。
G5. Rational line 只保留 B7lp/B7lz/B7em/B7en 相关 memory/Coupling 修复，不继续大网格 Fourier。
G6. Codex 必须执行 depth-3 fallback queue；没有执行完不能 final stop。
```

---

# 5. v12.25 核心假设

## H1：task-positive direct/gain source 与 geometry-positive quad source 可以互补

v12.24 的 direct_gain / direct_only 能产生 task accuracy margin，但 LineC/CouplingR2/tail 不过；quad_direct / I30 能产生 near LineC majority 和 task-control near-positive，但 CEp99 不过。

假设：

$$
\boxed{
\Delta\theta_{func}
=
\lambda_d \Delta\theta_{direct/gain}
+
\lambda_q \Delta\theta_{quad/reservoir}
}
$$

可以让 task signal 与 geometry preservation 同时成立。关键不是继续单独扩大 $\lambda_d$ 或 $\lambda_q$，而是求一个 constrained mixture。

## H2：CEp99 blocker 不是 CE-specific 问题，而是 unlabeled logit-tail drift 可以提前发现

CEp99 不能作为 direction source，但可以作为 audit。我们需要用 label-free proxy 预测它：

```text
logit_abs_p99_delta
max_logit_p99_delta
entropy_p01_delta
marginless_confidence_p99_delta
probe_logit_drift_p99
classwise_gain_spread
role_energy_tail_shift
```

假设这些 precommit observable 能过滤掉 NG4/NG9 的 tail-bad variants，同时不抹掉 source-control advantage。

## H3：matched controls 跟上 source，是因为 source direction 没有做 control-response residualization

当前 contrast selector能改变 selection，但 matched control仍吃掉 task advantage。下一步不只比较 source/control score，而是在 train-stream response space 中显式做 residualization：

$$
R_s = response(\Delta\theta_s),
$$

$$
R_c = response(\Delta\theta_c),
$$

$$
R_{res}=R_s - \Pi_{\mathcal{C}}R_s,
$$

其中 $\mathcal{C}$ 是 matched controls response subspace。只有 $R_{res}$ 保持 task/geometry proxy signal 时，才允许进入 P4。

## H4：T2/T3 upper-bound signal 需要转成 precommit micro-run，而不是 feature distillation

response-distilled T1B 直接建 feature 没有 positive rows。可能原因是 feature 太静态。下一步应该允许一个 **loss-agnostic clone micro-run**：在 commit 前对 source/control 做 1-2 个无标签 micro-step，只读取 logits/geometry proxy，不读 label/CE/validation/test。

这不是 future outcome，也不是 validation；它是 train-stream precommit geometry probe。

## H5：Rational 的问题不是是否有 task signal，而是 memory/Coupling 与 task不能同过

Rational B7lz / B7lp / B7em / B7en 的信息说明 family 不该放弃，但不能再大网格。需要 focused repair：降低 memory ratio、保留 L3/A4、提升 MNIST CouplingR2，同时不伤 Fashion task。

---

# 6. v12.25 执行规则：Fail-Closed, Explore-Open, Depth-3 Mandatory

## 6.1 Stop rule

Codex 不允许因为某个 gate fail 直接 final stop。`final_stop_allowed=1` 只能由以下情况触发：

```text
1. official S5 success reached；
2. user_stop_flag = 1；
3. hard_compute_budget_exhausted = 1 且 depth-3 fallback 全部执行；
4. mechanism-level no-go 成立，且 no-go 的所有必要反事实实验都已执行。
```

明确禁止：

```text
route = no-go
next_hypothesis_written = 1
=> final_stop_allowed = 1
```

## 6.2 Depth-3 fallback obligation

每条主线如果失败，必须至少执行三层 fallback：

```text
Depth 1: 当前 family 内参数/role/policy 修复。
Depth 2: source/control/tail/LineC constraint 组合修复。
Depth 3: 机制替换或 no-go 反事实验证。
```

如果某条线只完成 Depth 1 或 Depth 2，route 必须是：

```text
R0-ExploreDepthIncomplete
```

## 6.3 Promotion 与 exploration 分离

```text
promotion_allowed = 1:
  只有所有 official gates 通过。

exploration_continue = 1:
  只要出现以下任一信号，必须继续下一层 fallback：
  - source_vs_control_acc_delta > 0
  - source_vs_noop_acc_delta > 0.015
  - LineC pass >= 3/5
  - T2/T3 upper-bound AUC > 0.7
  - direct/gain policy 在 >=2 train-shuffle seeds 有正 margin
  - Rational family 有 task near-pass 且 efficiency/Coupling单项接近
```

---

# 7. 实验设计总览

v12.25 分七条线并行执行。

```text
Line R: Code/provenance/implementation review。
Line A: Label-free anchor monitor + A51 residual LineC repair。
Line F: S4-to-S5 composite functional bridge。
Line T: Precommit tail-safe / control-residual value source。
Line P: Policy-aware P3/P4 alignment。
Line D: Rational-focused classic no-BSpline hardening。
Line Z: Finalizer, no-go boundary, next-generation queue execution。
```

---

# 8. Line R：代码、provenance 与可部署性审查

## 8.1 目标

确保 v12.25 的结论不会被代码路径、artifact 游离、query/validation 使用、CE/label leakage、或 incomplete code packet 污染。

## 8.2 必须审查的代码路径

Codex 复盘必须指出这些文件、核心 symbol、line range、调用链：

```text
R1 B320 / A51 / Line A model construction:
  run_v1218_b320_label_free_ablation.py
  fc_purekan_primitives.py

R2 train-stream bridge source/control:
  run_v1224_train_stream_functional_bridge.py
  run_v1224_i30_i32_policy_bridge.py

R3 P4 verifier / trajectory gate:
  run_v1223_p4_trajectory_gate_verifier.py
  run_v1223_p4_compensation_modes.py
  run_v1223_p4_official_row_scan.py

R4 LineC / multi-sketch metrics:
  run_v1252_efficiency_functional_manifold.py
  run_v120_good_geometry_battery.py

R5 Classic line hardening:
  run_v1224_classic_hardening.py
  run_v1223_line_d_hardening.py

R6 Finalizer:
  run_v1224_finalize_functional_bridge.py
  new run_v1225_finalize.py
```

## 8.3 新增 artifacts

```text
v1225_code_semantics_manifest.csv
v1225_transitive_code_packet_manifest.csv
v1225_feature_provenance.csv
v1225_direction_source_audit.csv
v1225_control_scope_audit.csv
v1225_dataset_branch_audit.csv
```

## 8.4 Gate

必须满足：

```text
transitive_code_packet_missing_count = 0
unknown_core_symbol_count = 0
uses_query_batch_for_promotion = 0
uses_validation_or_test_for_commit = 0
uses_label_or_ce_for_direction = 0
dataset_name_branch_count = 0
matched_control_scope_pass = 1
```

否则 route：

```text
R0-CodeOrProvenanceIncomplete
```

---

# 9. Line A：Label-free anchor monitor 与 A51 residual LineC repair

## 9.1 目标

B320-current 仍是 diagnostic anchor，但 label-informed。Label-free A51/A1 已经 task 接近，但 LineC 全 0。Line A 下一步不再替换整个 projector，而是尝试 **A51 residual LineC repair**。

## 9.2 假设

$$
\boxed{
\text{A51 的 task 坐标已经接近；LineC 失败来自少数 role/branch 的 signal-reservoir residual。}
}
$$

因此应当做低秩 residual repair，而不是重新设计整个 label-free base。

## 9.3 候选

```text
A70-A51ResidualLineCFrame-rank8
A71-A51ResidualLineCFrame-rank16
A72-A51ControlResidualFrame-rank8
A73-A51ControlResidualFrame-rank16
A74-A51DirectQuadBalance-noY
A75-A51RoleEnergyTailClamp-noY
A76-A51LineCEMAAdapt-noY
```

所有候选必须：

```text
uses_y_for_stats = 0
uses_label_for_init = 0
uses_ce_vector = 0
uses_dataset_name_branch = 0
```

## 9.4 实验设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
batch_size = 128
LineC batch = 64
sketch_dim = 24
multi_sketch_seeds = 5
```

## 9.5 记录指标

CSV：`v1225_label_free_residual_linec.csv`

```text
candidate_id
dataset
seed
uses_y_for_stats
mean_delta_vs_A0
worst_delta_vs_A0
mean_delta_vs_MLP
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
ECE_delta
NLL_delta
CEp99_delta
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_seed_pass_count
LineC_all_pass
LineC_majority_pass
role_energy_tail_shift
branch_gain_spread
logit_tail_proxy_p99
```

## 9.6 Gate

Exploration gate：

```text
mean_delta_vs_A0 >= -0.005
worst_delta_vs_A0 >= -0.025
AUC_time_ratio_vs_mlp <= 1.10
LineC_majority_pass_count >= 5/9 rows
```

Official label-free base gate：

```text
mean_delta_vs_A0 >= -0.003
worst_delta_vs_A0 >= -0.010
AUC_time_ratio_vs_mlp <= 1.00
LineC_all_pass = 1 for >=8/9 rows
```

## 9.7 Failure -> Codex 自动尝试

```text
如果 task 接近但 LineC=0:
  自动运行 A70/A72 rank 扩展和 role-energy tail clamp。

如果 LineC 提升但 AUC_time > 1.10:
  自动运行 EMAAdapt warmup delay 和 residual strength halving。

如果 CEp99 / logit_tail_proxy 爆:
  自动运行 RoleEnergyTailClamp；不得使用 CEp99 作为方向，只能用 unlabeled logit-tail proxy。

如果所有 A70-A76 均 fail:
  输出 A51-no-go boundary，并继续 Line F/T/P，不得 final stop。
```

## 9.8 可视化

```text
fig_a51_task_linec_pareto.svg
fig_a51_linec_seed_pass_heatmap.svg
fig_a51_tail_proxy_vs_CEp99_audit.svg
fig_a51_role_energy_shift.svg
```

---

# 10. Line F：Composite Functional Bridge，从 I30/NG4/NG9/NG17 的 partial signal 组合出 S5 候选

## 10.1 目标

当前最有希望的 functional 信号不是单一 source，而是：

```text
I30 / NG4: acc + control near-positive + LineC majority，但 CEp99 fail。
NG9: quad_direct 改善 source-vs-NoOp，但 control margin / tail fail。
NG17/NG18: direct_gain/direct_only 有 task-control margin，但 CouplingR2 / LineC / tail fail。
```

Line F 要设计 composite event，让它们互补。

## 10.2 候选 family

### F25-C1：Direct-Gain + Quad-Reservoir Mixture

$$
\Delta\theta
=
\lambda_d\Delta\theta_{direct\_gain}
+
\lambda_q\Delta\theta_{quad\_reservoir}.
$$

Grid：

```text
lambda_d in {0.25, 0.50, 0.75, 1.00}
lambda_q in {0.25, 0.50, 0.75, 1.00}
normalize_response = true
```

### F25-C2：Control-Residualized Source

在 train-stream logit response space 中投影掉 matched controls：

$$
R_{res} = R_s - C(C^TC + \rho I)^{-1}C^TR_s.
$$

然后把 actuator 组合解为：

$$
\min_a \|R(a) - R_{res}\|^2 + \rho\|a\|^2.
$$

### F25-C3：Tail-Budgeted Composite Event

用 label-free tail proxy 做 hard constraint：

$$
LogitTailProxy_{p99}(source) \le LogitTailProxy_{p99}(noop) + \tau.
$$

其中 `LogitTailProxy` 由 train-stream unlabeled logits 构造：

```text
max_logit_abs_p99
confidence_p99
entropy_p01
logit_drift_p99
role_energy_tail_shift
class_gain_spread
```

### F25-C4：Policy-aware quad-direct schedule

不是固定 `quad_only / quad_direct / direct_only`，而是在 event 后做 2-step loss-agnostic policy probe：

```text
probe_policy in {quad_only, quad_direct, direct_gain, direct_branch_gain, direct_only}
probe uses train-stream unlabeled logit movement only
no label / CE / validation / test
```

## 10.3 实验设置

```text
source neighborhoods:
  NG4-like I30 train-stream source
  NG9 quad_direct neighborhood
  NG17/NG18 direct_gain/direct_only neighborhood

datasets:
  MNIST, Fashion-MNIST, KMNIST

seeds:
  0,1,2

train_shuffle_seeds:
  12240400, 12241400, 12242400

multi_sketch_seeds:
  5 per row

short-run:
  epochs = 8, 12
  lr = 0.0015, 0.002
```

## 10.4 记录指标

CSV：`v1225_composite_functional_bridge.csv`

```text
candidate_id
component_ids
lambda_direct
lambda_quad
control_residualized
role_policy
precommit_score
precommit_tail_proxy
uses_query_batch
uses_train_batch
uses_label_for_direction
uses_ce_for_direction
source_acc
noop_acc
best_control_acc
source_vs_noop
source_vs_best_control
source_NLL
noop_NLL
source_CEp99
noop_CEp99
source_ECE
noop_ECE
source_CouplingR2
noop_CouplingR2
source_NoiseSignalLeak
source_RealSignalReservoirRatio
LineC_seed_pass_count
strict_majority_pass
strict_all_pass
train_shuffle_robust_majority_pass
train_shuffle_robust_all_pass
amortized_overhead_ratio
promotion_allowed
```

## 10.5 Gate

Exploration S4a gate：

```text
source_vs_best_control >= 0.00390625
source_vs_noop >= 0.015625
LineC_seed_pass_count >= 3/5
source_CEp99 <= noop_CEp99 + 0.50
train_shuffle_positive_count >= 2/3
```

Official S5 gate：

```text
source_vs_best_control >= 0.0078125
source_vs_noop >= 0.015625
source_NLL <= noop_NLL
source_CEp99 <= noop_CEp99 + 0.05
source_ECE <= noop_ECE + 0.02
LineC_seed_pass_count = 5/5
train_shuffle_robust_majority_pass = 1
combined_bridge_any_train_shuffle_robust_all_pass = 1
amortized_overhead_ratio <= 1.05
```

## 10.6 Failure -> Codex 自动尝试

```text
如果 source_vs_control > 0 但 CEp99 fail:
  不允许普通 label smoothing 作为唯一修复。
  自动进入 F25-C3 tail-budgeted composite event。

如果 source_vs_noop > 0 但 source_vs_control <= 0:
  自动进入 F25-C2 control-residualized source。

如果 LineC 0/5 但 task margin 正:
  自动混入 quad-reservoir component，运行 F25-C1。

如果 LineC majority 但 all-pass fail:
  自动执行 multi-sketch response variance decomposition，区分 metric noise vs real instability。

如果 train-shuffle robust fail:
  自动执行 train-stream reference bootstrap/median-of-means，不得直接 stop。

如果 F25-C1-C4 全 fail:
  输出 mechanism no-go，并继续 Line T / P，不得 final stop，除非 depth-3全执行。
```

## 10.7 可视化

```text
fig_composite_task_vs_tail.svg
fig_composite_source_control_margin.svg
fig_linec_sketch_pass_heatmap.svg
fig_train_shuffle_robustness.svg
fig_direct_quad_lambda_pareto.svg
fig_tail_proxy_vs_CEp99.svg
fig_control_residual_response_norm.svg
```

---

# 11. Line T：Precommit value source v2，重新设计可部署 tail-safe / control-resistant observable

## 11.1 目标

T1B / response-distilled features 到目前没有 positive rows。v12.25 不再简单做静态 response distillation，而是做 **precommit micro-probe**：允许在 commit 前对 source/control 在 train-stream 上做极短的无标签几何 probe，只读 logits/role energy/LineC proxy，不读 label/CE。

## 11.2 Feature tiers

```text
T1-static:
  commit 前静态 logits/activation/role energy。

T1B-optimizer-observable:
  base optimizer update as opaque vector；不读 CE/label 内容。

T1C-microprobe:
  1-2 step clone micro-probe，只读 unlabeled logits and role geometry。

T2-response-teacher:
  completed response；只用于 upper-bound / distillation target，不 promotion。

T3-audit:
  CE / label / LineC hard target；只审计，不 direction。
```

## 11.3 新增 target

定义 label-free precommit score：

$$
G_{pre}
=
\alpha G_{control\_residual}
+
\beta G_{linec\_proxy}
-
\gamma G_{tail\_proxy}
-
\eta G_{instability}.
$$

其中：

```text
G_control_residual:
  source response minus matched control response norm / direction.

G_linec_proxy:
  unlabeled train-probe coupling proxy, random-cotangent sketch stability, role-energy balance.

G_tail_proxy:
  max_logit_abs_p99, entropy_p01 drop, confidence_p99, logit_drift_p99.

G_instability:
  response variance across micro-batches / sketch seeds / train-shuffle seeds.
```

## 11.4 实验设置

```text
input rows:
  all v12.24 bridge candidates + NG4/NG8/NG9/NG10/NG17/NG18 neighborhoods

train splits:
  source/control/noop all share same train microprobe batches

leaveout:
  leave-dataset-out
  leave-seed-out
  leave-train-shuffle-out
```

## 11.5 记录指标

CSV：`v1225_precommit_value_source.csv`

```text
row_id
candidate_id
feature_tier
precommit_available
uses_label
uses_ce
uses_linec_target
uses_query_batch
uses_validation_test
G_control_residual
G_linec_proxy
G_tail_proxy
G_instability
G_pre
actual_source_vs_control
actual_CEp99_delta_vs_noop
actual_LineC_seed_pass_count
actual_train_shuffle_pass
AUC_source_control
AUC_tail_safe
AUC_linec_majority
precision_at_k_s5_proxy
recall_at_k_s5_proxy
leave_dataset_auc_min
leave_seed_auc_min
leave_shuffle_auc_min
```

## 11.6 Gate

Exploration visibility gate：

```text
AUC_source_control_min >= 0.60
AUC_tail_safe_min >= 0.60
AUC_linec_majority_min >= 0.60
precision_at_k_s5_proxy >= 0.20
```

Official visibility gate：

```text
AUC_source_control_min >= 0.70
AUC_tail_safe_min >= 0.70
AUC_linec_majority_min >= 0.70
precision_at_k_s5_proxy >= 0.30
leave_dataset_auc_min >= 0.60
leave_seed_auc_min >= 0.60
leave_shuffle_auc_min >= 0.60
```

## 11.7 Failure -> Codex 自动尝试

```text
如果 T1-static fail 但 T1C-microprobe pass:
  允许 T1C 作为 precommit source，但必须记录 overhead；不能降级为 T2。

如果 G_tail_proxy 与 CEp99 相关性低:
  自动增加 logit_drift_p99 / entropy_p01 / role_energy_tail_shift ablation。

如果 control_residual AUC 高但 tail AUC 低:
  与 Line F tail-budgeted composite event 联动。

如果所有 T1/T1B/T1C fail:
  形成 value-source no-go，要求 Line F 使用 brute-force exploratory 但 promotion=0。
```

## 11.8 可视化

```text
fig_precommit_score_roc.svg
fig_tail_proxy_corr.svg
fig_control_residual_auc.svg
fig_leaveout_visibility_heatmap.svg
fig_microprobe_overhead_vs_auc.svg
```

---

# 12. Line P：Policy-aware P3/P4 alignment

## 12.1 目标

v12.23/v12.24 的核心错配是：P3 pass 点、P4 task gain 点、LineC pass 点经常不重合。Line P 要把 P3 改成 policy-aware：P3 不再只评估 event 本身，而是评估：

$$
\text{event} + \text{planned post-event policy}.
$$

## 12.2 Policy set

```text
quad_only
quad_direct
direct_gain
direct_branch_gain
direct_only
freeze_quad
role_mixture_auto
```

## 12.3 P3 metric

P3 不使用 label / CE 方向，但可以用 audit labels评价 gate。

P3 precommit response包括：

```text
unlabeled coupling proxy
logit tail proxy
control residual response
role energy transport
multi-sketch stability
micro-run response variance
```

## 12.4 记录指标

CSV：`v1225_policy_aware_p3.csv`

```text
candidate_id
policy_id
precommit_score
control_residual_score
tail_proxy_score
linec_proxy_score
multi_sketch_stability
P3_pass
P4_source_vs_noop
P4_source_vs_control
P4_CEp99_delta
P4_LineC_seed_pass_count
P3_to_P4_rank_alignment
```

## 12.5 Gate

```text
P3_to_P4_rank_alignment >= 0.30 exploratory
P3_to_P4_rank_alignment >= 0.50 official
```

如果 rank alignment 仍低于 0.30，则说明 P3 不可用，不能继续扩 P3 grid。

## 12.6 Failure -> Codex 自动尝试

```text
如果 P3 alignment 低:
  自动输出 P3-invalid boundary，并让 Line F 直接用 micro-P4 selection。

如果某 policy task positive 但 LineC fail:
  自动转入 role_mixture_auto，不再单 policy 扫。

如果 all policies source_vs_control <= 0:
  自动执行 control-residualized source construction。
```

---

# 13. Line D：Rational-focused classic no-BSpline hardening

## 13.1 目标

Classic line 不再大范围扩 Fourier。v12.24 显示 Rational 比 Fourier 更接近可解释的 family signal，但仍 memory/Coupling/task 不闭合。

## 13.2 Candidate focus

```text
RationalB7lp
RationalB7lz
RationalB7em
RationalB7en
RationalB7me
RationalB7ma
```

新增 focused repairs：

```text
D25-RationalMemoryCut-readscaleShared
D26-RationalCouplingLift-noWhiten
D27-RationalPairNormStopGrad-light
D28-RationalTaskTrajectoryWarmNoExtraMem
D29-RationalB7lpLowMemCouplingMix
```

Fourier 只运行一个 low-budget monitor：

```text
FourierB4v / B4w no new grid unless D25-D29 blocked early.
```

## 13.3 实验设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
```

## 13.4 记录指标

CSV：`v1225_classic_rational_hardening.csv`

```text
family
candidate_id
dataset
seed
step_ratio
memory_ratio
A4_pass
A5_task_pass
mean_delta_vs_mlp
worst_delta_vs_mlp
AUC_time_ratio
ECE_delta
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
LineC_pass
exploration_gate_pass
family_status
```

## 13.5 Gate

Exploration S4c gate：

```text
step_ratio <= 1.25
memory_ratio <= 1.25
A4_pass = 1
mean_delta_vs_mlp >= -0.03
CouplingR2 >= MLP_CouplingR2 - 0.05
RealSignalReservoirRatio <= MLP_ReservoirRatio + 0.10
```

Official FamilyPass：

```text
step_ratio <= 1.10
memory_ratio <= 1.00
A4_pass = 1
A5_task_pass = 1
LineC_pass = 1
```

## 13.6 Failure -> Codex 自动尝试

```text
如果 memory_ratio > 1.25:
  自动运行 shared readscale / reduced pair feature；不继续加 width。

如果 CouplingR2 低:
  自动运行 no-whiten coupling lift / pairNorm stopgrad。

如果 task near但 AUC fail:
  自动运行 warm schedule，不改 loss。

如果 Rational D25-D29 均 fail:
  输出 Rational mechanism boundary，保留 monitor，不抢 functional预算。
```

## 13.7 可视化

```text
fig_rational_memory_task_pareto.svg
fig_rational_coupling_vs_task.svg
fig_rational_reservoir_vs_memory.svg
fig_classic_family_status.svg
```

---

# 14. Line Z：Finalizer 与 no-go / success route

## 14.1 必须输出 artifact

```text
v1225_route_decision.json
v1225_required_artifact_manifest.csv
v1225_fallback_execution_manifest.csv
v1225_core_code_review_manifest.csv
v1225_no_go_boundary.md
v1225_next_generation_queue.csv
v1225_code_review_packet.zip
```

## 14.2 Route 定义

```text
S5-OfficialFunctionalSuccess:
  Line F official S5 gate pass + Line R pass。

S4a-PrecommitBridgeExplorationOpened:
  Line F exploration gate pass but official fail。

S4b-PolicyAwareP3Opened:
  Line P P3-to-P4 alignment pass。

S4c-RationalClassicExplorationOpened:
  Line D exploration pass。

R4-S4toS5NoGoAfterDepth3Fallbacks:
  All depth-3 fallbacks executed, no exploration route opened。

R0-ExploreDepthIncomplete:
  A failure happened but required next fallback was not executed。

R0-CodeOrProvenanceIncomplete:
  Code/provenance audit failed。
```

## 14.3 No-go 不能只是文字

如果最终 no-go，必须写清楚：

```text
1. 哪个 mechanism family 被排除；
2. 哪个假设仍未验证；
3. 哪个 candidate 有 partial signal；
4. 为什么不能 promotion；
5. 下一轮是否还值得继续；
6. 若继续，必须先做什么，而不是重新扫小网格。
```

---

# 15. 必须生成的图

```text
fig_v1225_progress_dashboard.svg
fig_s4_to_s5_blocker_matrix.svg
fig_ng4_ng9_ng17_signal_tradeoff.svg
fig_tail_safety_tradeoff.svg
fig_source_control_margin_vs_CEp99.svg
fig_linec_multisketch_stability.svg
fig_train_shuffle_robustness.svg
fig_precommit_visibility_roc.svg
fig_direct_quad_composite_pareto.svg
fig_policy_p3_p4_alignment.svg
fig_rational_classic_pareto.svg
fig_route_decision_tree.svg
```

---

# 16. Codex 最短执行摘要

Codex 下一轮不要再做：

```text
1. 只扩大 I30/I32 sign/scale 网格。
2. 只调 smoothing/weight decay 来压 CEp99。
3. 把 T2/T3 response teacher 当作 deployable source。
4. 单个 train-shuffle seed near-pass 后停止。
5. Line D 继续大规模 Fourier 网格。
6. 写 no-go 后停止但不执行 depth-3 fallback。
```

Codex 必须做：

```text
1. 执行 Line R code/provenance审计。
2. 执行 Line A A70-A76 residual LineC repair。
3. 执行 Line F F25-C1/C2/C3/C4 composite functional bridge。
4. 执行 Line T T1C precommit microprobe + control residual visibility。
5. 执行 Line P policy-aware P3/P4 alignment。
6. 执行 Line D Rational-focused D25-D29。
7. 如果任何 line fail，继续其 depth-3 fallback；否则 route = R0-ExploreDepthIncomplete。
8. 最终输出完整 code packet、route、figures、failure table。
```

---

# 17. 最终判断

v12.24 的真正进展是：

$$
\boxed{
\text{functional update 已经出现 precommit/train-stream 的局部 task signal，}
\text{但这个 signal 与 control gap、tail safety、LineC、robustness尚未同位。}
}
$$

v12.25 的核心不是继续找“更大一点”的 I30/I32，而是构造一个 **composite functional event**：

```text
direct/gain 提供 task signal；
quad/reservoir 保持 LineC；
control-residualization 防 matched controls 追平；
loss-agnostic tail proxy 防 CEp99 爆；
policy-aware P3 让 P3 和 P4 对齐。
```

如果这条路线仍然失败，并且 depth-3 fallback 全部完成，那么才可以得出更强 no-go：当前 train-stream compensation / direct-logit functional bridge family 不足以完成 S4-to-S5，需要重新定义 functional update primitive。

