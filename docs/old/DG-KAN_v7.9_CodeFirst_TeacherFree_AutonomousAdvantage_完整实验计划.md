# DG-KAN v7.9 Code-First Teacher-Free Autonomous Advantage：代码契约硬化、自主表达力闭合与系统级验证完整实验计划

> 本计划基于最新上传的 `dgkan_core(8).py`、`run_gafu_v78_real.py` 以及 v7.8 Teacher-Free Official 实验复盘重新制定。  
> 这次重新思考的核心不是继续沿用上一版实验计划，而是从代码层重新判断：**v7.8 runner 本质上是复用 v7.6 runner 后做 teacher-free route postprocess，它并没有真正实现 teacher-free gap attribution、self-bootstrap、phase-mapped profiler 或 dense-preserving fused/streaming package。**  
> 因此 v7.9 的第一目标不是再增加一个表格或再跑一个小 probe，而是把 official teacher-free 实验路径从“postprocess route”升级为“真实 candidate registry + contract validator + no-external training branch + profiler instrumentation + full-grid gate”的 code-first runner。

---

## 0. 从代码重新得到的核心判断

### 0.1 `run_gafu_v78_real.py` 是 postprocess wrapper，不是完整机制 runner

`run_gafu_v78_real.py` 的顶层说明已经写得很清楚：v7.8 复用 v7.6 的真实训练/评估实现，只改变 route decision：C3-teacher-distilled candidates 是 diagnostic only，official success 只看 no-external-teacher PureKAN candidates，目前是 M13。

代码中：

```python
def run(args):
    v76.PLAN_PATH = PLAN_PATH
    v76.SCRIPT_PATH = SCRIPT_PATH
    v76.run(args)
    out_dir = Path(args.out_dir)
    _write_v78_postprocess(out_dir, args)
```

这意味着 v7.8 主要做的是：

```text
1. 调用 v76.run(args) 产生已有 artifacts；
2. 从已有 artifacts 中读 task / significance / grad / efficiency / AUC；
3. 重新标记 B2/M12 为 external teacher；
4. 在 postprocess 中选择 M13 作为 official teacher-free candidate。
```

这本身是正确的纠偏，但它不是新的机制实验。它没有真正实现：

```text
self-distill
EMA teacher
delayed snapshot teacher
teacher-free gap attribution
validation split audit
phase-mapped profiler
dense-preserving fused package
S1 memory repair package
```

因此，v7.8 没有达成目标不只是模型还差一点，也暴露出实验代码还停留在 “route audit wrapper” 阶段。

---

### 0.2 v7.8 的 teacher 标记是 hardcoded，未来容易出错

v7.8 的 teacher 标记来自：

```python
def _external_teacher_used(cid: str) -> int:
    return int(cid in {"B2", "M12"})
```

这对当前四个候选 `B0,B2,M12,M13` 足够，但它不是可扩展的 official gate。只要以后新增候选，比如：

```text
TF-self-distill
M13-C3-offline
M13-EMA
M13-snapshot
M13-teacherfree-repair
```

如果没有严格 candidate metadata，postprocess 可能错误地把带 teacher 的候选当作 official，或者把 no-external self-bootstrap 候选误判为 external teacher。

所以 v7.9 必须把 teacher contract 从：

```text
candidate_id hardcode
```

改成：

```text
CandidateSpec.external_teacher_used
CandidateSpec.self_teacher_used
CandidateSpec.teacher_source
CandidateSpec.teacher_logits_used
CandidateSpec.official_eligible
```

并且每个 artifact row 都必须落盘这些字段，route decision 不能再靠 candidate id 猜测。

---

### 0.3 v7.8 的 StrictPass postprocess 过弱

`run_gafu_v78_real.py` 里的 `_candidate_row` 对 strict pass 的判断主要是：

```python
strict_pass = int(
    str(task.get("head_is_kan")) in {"1", "1.0"}
    and str(task.get("non_kan_trainable_param_count")) in {"0", "0.0", ""}
) if cid not in {"B0", "B2"} else 0
```

这没有直接检查：

```text
manual_forward_available
manual_backward_available
manual_update_available
uses_loss_backward
uses_torch_autograd_graph
```

虽然 v76/v7.8 的复盘中记录了 StrictPass/GradPass，但从代码看，v7.8 postprocess 本身并没有完整执行 strict contract。下一版必须把 strict pass 改成：

$$
\text{StrictPass}
=
[\text{nonKAN}=0]
\land
[\text{head\_is\_kan}=1]
\land
[\text{manual\_forward}=1]
\land
[\text{manual\_backward}=1]
\land
[\text{manual\_update}=1]
\land
[\text{uses\_loss\_backward}=0].
$$

否则 official route 的可信度会被实现细节削弱。

---

### 0.4 `dgkan_core(8).py` 说明当前系统具备多个潜在低 live-set primitive，但 v7.8 official runner 没系统使用它们

`dgkan_core(8).py` 里已经有不少结构基础：

```text
PureKANClassifier:
  input_kan -> PureResidualKANBlock stack -> output_kan

SparseInterpKANDense:
  sparse two-bin interpolation，不 materialize [batch, in_dim, basis]

SparseSplineKANDense:
  smoothstep sparse interpolation

DWM2LiteDense:
  y = W(x + s r(x))，单 GEMM + channel residual

DWM2Dense:
  pre_mix -> depthwise AB function -> post_mix

GEMMNativeDepthwiseMixDense:
  GEMM-native depthwise AB-RBF transform + mixing

CPABRBFDense:
  low-rank CP AB-RBF edge
```

这说明代码层不是没有可探索的 teacher-free official primitive。问题是 v7.8 runner 只复用 v76 的候选组合，postprocess 后只选择 M13，并没有把这些 primitive 系统纳入 teacher-free official 实验。

v7.9 的核心应该是把这些已有结构纳入 **metadata-driven candidate registry**，而不是继续让 runner 隐式复用旧候选。

---

### 0.5 当前实验数据的结论没有变：M13 是 near-pass，不是 success

v7.8 复盘显示，official candidate `M13` 是 teacher-free，StrictPass、GradPass、FullGridS2 在主 run 中成立，但 TeacherFreeMacroSignificantPass 未达成：

```text
M13 val gap vs MLP = +0.01784
CI95 low = +0.01224
Holm p = 7.97e-06
test gap = +0.02298
ECE delta = -0.00342
NLL delta = -0.09742
memory max = 1.0381
step max = 1.4857
S2 shapes = 9/9
```

official gate 是：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

所以 M13 离 official gate 还差：

$$
0.0200-0.01784=0.00216.
$$

这个差距很小，但不能四舍五入，也不能用 `M12+C3` 代替。`M12+C3` 的 gap 是 `+0.02057`，但 external teacher used = true，只能作为 diagnostic assisted success。

---

## 1. 目标是否达成

### 1.1 v7.8 Teacher-Free Official 目标没有达成

当前 official route 是：

```text
best official candidate = M13
external_teacher_used = false
teacher_candidate = none
strict_pass = true
grad_pass = true
teacher_free_macro_significant_pass = false
fullgrid_s2_pass = true
fullgrid_s1_pass = false
time_auc_pass = false
success_v78_minimum = false
success_v78_formal = false
```

因此：

$$
\boxed{
\text{v7.8 teacher-free official target 没有达成。}
}
$$

### 1.2 但 teacher-free 方向不是失败路线

M13 仍然有稳定正信号：

$$
\Delta Acc_{\text{macro,val}}=+0.01784,
$$

$$
CI_{95\%,macro}^{low}=+0.01224,
$$

$$
p_{\text{Holm}}=7.97\times10^{-6},
$$

$$
\Delta Acc_{\text{macro,test}}=+0.02298.
$$

并且：

$$
\Delta ECE=-0.00342,
$$

$$
\Delta NLL=-0.09742.
$$

这说明：

$$
\boxed{
\text{teacher-free PureKAN-NG 已经稳定优于 MLP，但优势 margin 不够厚。}
}
$$

当前不是要否定 M13，而是要解释并闭合：

$$
\boxed{
+0.01784 \rightarrow +0.02000
}
$$

这 $0.00216$ 的差距。

---

## 2. 基于代码和数据的实验进展

### 2.1 表达力主线已经从 dense oracle 走到 teacher-free near-pass

早期 dense D3 证明 KAN family 有超过 MLP 的表达力，但 dense D3 不是 efficient official candidate。后续 C3/M12 证明 strict/cached/packed 路线可以保留表达力。v7.8 的 M13 进一步证明：即使不使用 C3 external teacher，PureKAN-NG 也能达到 `+0.01784` 的稳定 macro gap。

这不是小信号。它说明：

```text
1. KAN 表达力不是空的；
2. teacher-free route 不是完全依赖 C3；
3. M13 已经接近 official gate；
4. 现在剩余的是 autonomous margin，而不是从零开始找方向。
```

### 2.2 代码契约从 “模型信号” 进入 “实验系统可信性” 阶段

v7.8 的 code wrapper 说明我们现在遇到的不是单纯模型问题：

```text
- candidate teacher metadata 还不是数据驱动；
- strict pass 还不是完整 contract gate；
- self-distill / EMA 等 no-external teacher branch 没实现；
- repeated validation / gap attribution 没实现；
- phase-mapped profiler 没实现；
- fused/streaming package 没实现。
```

因此，v7.9 不能只说 “再试一个 candidate”。必须先把实验系统升级：

$$
\boxed{
\text{CandidateSpec}
+
\text{ContractValidator}
+
\text{TeacherFreeRunner}
+
\text{GapAttribution}
+
\text{ProfilerInstrumentation}
}
$$

### 2.3 当前最强辅助结果仍是 M12+C3，但它不能作为 official conclusion

M12+C3 的确达到 `+0.02057`，比 M13 高：

$$
0.02057-0.01784=0.00273.
$$

但从代码和 route 逻辑看，M12 被 hardcoded 为 `external_teacher_used=1`，因此它只能说明：

```text
C3 teacher 可以给 M13/M12 family 提供 mild boost。
```

不能说明：

```text
teacher-free PureKAN-NG official success。
```

这正是用户指出的路线问题。v7.9 必须把 C3 teacher 从 success path 移到 diagnostic path。

---

## 3. 问题所在

### 3.1 问题一：official route 依赖 postprocess，不是完整实验路径

v7.8 先跑 v76，再 postprocess route。这样可以审计已有结果，但不能回答新的机制问题。比如：

```text
M13 为什么差 +0.00216？
self-bootstrap 能不能补？
validation split 是否低估？
feature/rank/margin 是否缺？
profiler bottleneck 在哪里？
```

这些都不能靠 postprocess 回答。必须新增真实 measured stages。

### 3.2 问题二：teacher-free 差距很小，但不能用小修补糊过去

M13 已经接近，容易诱惑我们做：

```text
再加一点 budget
再调一点 lr
再调一点 basis
再调一点 loss
```

但 v7.8 repair 已经显示：

```text
long-budget 480 只提升约 +0.00059，还破坏 S2；
M4/M9 没超过 M13；
hidden80 接近但效率明显变差；
basis12/16 到 +0.01927，但没过 +0.0200，ECE 也不如 baseline。
```

所以这不是普通小调问题。核心要问：

$$
\boxed{
\text{M13 缺的是 evaluation robustness、self-regularization、representation rank，还是 architecture source？}
}
$$

### 3.3 问题三：M13 test gap 高于 validation gap，不能忽略

M13 的 validation gap：

$$
+0.01784.
$$

M13 的 test gap：

$$
+0.02298.
$$

这提示 validation split 可能低估了 teacher-free candidate，但不能用 test 调参。因此 v7.9 必须做 repeated validation / K-fold holdout audit，预注册 split，不碰 test。

### 3.4 问题四：core 里有低 live-set primitive，但 official runner 没系统化使用

`SparseInterpKANDense` 明确是 sparse two-bin interpolation，不 materialize `[batch, in_dim, basis]`。`DWM2LiteDense` 是 single-GEMM residual path。`PureKANClassifier` 支持通过 `dense_cls` 注入不同 edge。v7.9 应该利用这个结构做 teacher-free architecture-level repair，而不是只在 M13 basis/hidden 上扫。

### 3.5 问题五：系统目标还没轮到 final claim

M13 还没过 teacher-free macro gate，所以 S1/TimeAUC/Profiler 不能作为 official formal success。但它们仍是 diagnostic。当前顺序应该是：

```text
先 official teacher-free macro pass；
再 official S1 / TimeAUC / Profiler。
```

否则会出现：

```text
system path 做得再漂亮，但 official model 没过 macro。
```

---

## 4. 是否在正确道路上

是，但必须做一次代码层纠偏。

正确道路不是：

```text
继续使用 M12+C3 当 final；
继续做 loss/classwise/sampler 小修；
继续用 postprocess 包装旧 runner；
继续 hardcode candidate id 判断 teacher；
继续把 strict pass 简化成 head_is_kan + nonKAN=0。
```

正确道路是：

```text
1. code-first official teacher-free runner；
2. metadata-driven candidate registry；
3. strict contract validator；
4. repeated validation audit；
5. M13 vs M12+C3 gap attribution；
6. no-external self-bootstrap；
7. representation repair；
8. teacher-free full-grid S2/S1/TimeAUC verification。
```

这条路线直指本质：

$$
\boxed{
\text{PureKAN-NG 自身是否能稳定超过 MLP-AdamW。}
}
$$

---

## 5. 离目标还差多远

### 5.1 离 teacher-free minimum success

只差 macro gate：

```text
StrictPass = pass
GradPass = pass
FullGridS2Pass = pass in main run
TeacherFreeMacroSignificantPass = fail
```

数值差距：

$$
0.0200-0.01784=0.00216.
$$

### 5.2 离 teacher-free formal success

除了 macro gate，还差：

```text
FullGridS1Pass
TimeAUCPass
ProfilerPass
```

M13 当前：

```text
S1 shapes = 2/9
ValLossAUC_time ratio vs B0 = 1.3922
```

但 M13 的 step AUC 是好的：

$$
\frac{ValLossAUC_{\text{step,M13}}}{ValLossAUC_{\text{step,B0}}}=0.9370.
$$

这说明 TimeAUC 问题更像 runtime / loop / profiler path，而不是学习曲线本身。

### 5.3 离最终系统目标

最终系统目标还差四层：

```text
1. teacher-free +0.0200 macro gate；
2. full-grid S1；
3. wall-clock TimeAUC；
4. scaling / robustness / geometry Pareto。
```

当前第一优先级是第 1 层。第 2-4 层必须在 teacher-free macro pass 后进入 official route。

---

## 6. v7.9 总体目标

v7.9 的总体目标是：

$$
\boxed{
\text{把 teacher-free PureKAN-NG 从 stable near-pass 推到 official autonomous success。}
}
$$

这要求同时完成两件事：

第一，代码层要把 official 实验路径做实：

$$
\boxed{
\text{CandidateSpec}
+
\text{ContractValidator}
+
\text{TeacherFreeRunner}
+
\text{NoExternalCandidateArtifacts}
}
$$

第二，机制层要解释并闭合 M13 的 $0.00216$ gap：

$$
\boxed{
\text{validation robustness}
+
\text{gap attribution}
+
\text{self-bootstrap}
+
\text{representation repair}
}
$$

---

## 7. v7.9 代码层必须先修的内容

### 7.1 CandidateSpec metadata registry

新增 `CandidateSpec`，每个候选必须显式声明：

```text
candidate_id
candidate_name
model_family
dense_cls
hidden_dim
basis_count
depth
init_policy
optimizer_policy
external_teacher_used
self_teacher_used
teacher_source
teacher_logits_used
teacher_forward_used
official_eligible
expected_manual_forward
expected_manual_backward
expected_manual_update
expected_nonkan_count
```

判断标准：

```text
candidate_id 不再用于推断 external_teacher_used；
route 只能使用 CandidateSpec.official_eligible；
artifact rows 必须携带 CandidateSpec 字段；
candidate_registry.csv 必须是 route 的唯一元数据来源。
```

### 7.2 ContractValidator

新增强 contract validator，真正执行：

$$
\text{StrictPass}
=
[\text{nonKAN}=0]
\land
[\text{head\_is\_kan}=1]
\land
[\text{manual\_forward}=1]
\land
[\text{manual\_backward}=1]
\land
[\text{manual\_update}=1]
\land
[\text{uses\_loss\_backward}=0].
$$

必须记录：

```text
nonKAN_param_count
KAN_edge_param_count
manual_forward_available
manual_backward_available
manual_update_available
uses_loss_backward
uses_torch_autograd_graph
head_is_kan
input_kan_is_kan
output_kan_is_kan
edge_param_coverage
```

如果某字段不可测，不能默认 pass，必须写：

```text
metric_unavailable
```

并使 StrictPass = 0。

### 7.3 TeacherLeakAudit

新增 teacher leakage audit：

```text
external_teacher_used
teacher_logits_used
teacher_forward_used
teacher_checkpoint_path
teacher_precompute_path
self_teacher_used
self_teacher_update_rule
teacher_artifact_hash
```

Official candidate 必须满足：

```text
external_teacher_used = 0
teacher_logits_used = 0
teacher_forward_used = 0
```

self-bootstrap candidate 可以满足：

```text
external_teacher_used = 0
self_teacher_used = 1
```

但要单独标记为 no-external self-training recipe。

### 7.4 True teacher-free runner stages

v7.9 不允许只复用 v76 后 postprocess。需要新增真实 stages：

```text
P1 validation split audit
P2 teacher-gap attribution
P3 teacher-free repair
P4 self-bootstrap
P5 representation repair
P6 official 10-seed confirmation
P7 teacher-free time accounting
P8 phase-mapped profiler
```

每个 stage 必须产出 measured CSV，而不是从旧 artifacts 派生一个 pass。

---

## 8. 核心假设

### H1：M13 的 autonomous gap 是可闭合的，而不是结构上限

H1 假设：M13 已经拥有主要结构优势，剩余 $0.00216$ 来自 validation robustness、训练稳定性、margin 或表示初始化。

H1 成立标准：

某个 no-external candidate 满足：

$$
Acc_{\text{candidate}}-Acc_{\text{M13}}\geq0.0022.
$$

并且：

$$
\Delta Acc_{\text{macro,val,candidate}}\geq0.0200.
$$

同时：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

若所有 no-external candidates 增益都小于：

$$
0.001,
$$

则 H1 不成立，teacher-free route 需要结构重设计。

---

### H2：当前 validation split 可能低估 M13

H2 基于事实：

$$
\Delta Acc_{\text{val,M13}}=+0.01784,
$$

$$
\Delta Acc_{\text{test,M13}}=+0.02298.
$$

H2 假设：当前 validation hard gate 可能低估了 teacher-free candidate，但不能用 test 调参。

H2 成立标准：

使用预注册 repeated validation shards：

$$
\Delta Acc_{\text{macro,val-split-mean,M13}}\geq0.0200.
$$

且：

$$
CI_{95\%,split}^{low}>0.
$$

如果 repeated validation mean 仍低于：

$$
0.0180,
$$

则不是 split issue，而是真实 macro gap。

---

### H3：C3 teacher 的 mild boost 可以由 no-external self-bootstrap 复现

C3 teacher boost：

$$
\Delta_{\text{teacher}}
=
Acc_{\text{M12+C3}}-Acc_{\text{M13}}
\approx0.00273.
$$

H3 假设这主要是 soft-target / calibration / margin smoothing，可由 no-external self-bootstrap 复现。

候选包括：

```text
EMA self-teacher
delayed self-distill
snapshot self-distill
SWA logit smoothing
dual-view consistency
stopgrad previous epoch teacher
```

H3 成立标准：

$$
Acc_{\text{self-bootstrap}}-Acc_{\text{M13}}\geq0.0020.
$$

并且：

```text
external_teacher_used = 0
```

且时间开销可接受：

$$
ValLossAUC_{\text{time,self}}\leq1.10\cdot ValLossAUC_{\text{time,M13}}.
$$

---

### H4：如果 self-bootstrap 不够，M13 缺的是 representation / margin，不是 optimizer

H4 假设：若 self-bootstrap 增益不足，则 M13 与 M12+C3 的差距来自 representation。

需要比较：

```text
feature CKA
logit KL
margin p10 / p50
feature effective rank
hard sample overlap
dataset-wise gap contribution
NLL decomposition
```

H4 成立标准之一：

$$
CKA(M13,M12)<0.90,
$$

or:

$$
rank_{\text{M13}}<0.95\cdot rank_{\text{M12}},
$$

or:

$$
margin_{p10,M13}<0.95\cdot margin_{p10,M12}.
$$

若 H4 成立，应进入 representation repair，而不是继续 schedule 小修。

---

### H5：已有 core primitive 应用于 teacher-free architecture repair，而不是继续 hidden/basis 粗扫

v7.8 已经说明 hidden80、basis12、basis16 都是 near-miss，但没有解决本质。H5 假设应改用结构性 primitive：

```text
SparseInterpKANDense
SparseSplineKANDense
DWM2LiteDense
GEMMNativeDepthwiseMixDense
prototype KAN head
local cross-feature bridge lite
feature-rank-preserving init
```

H5 成立标准：

某个 teacher-free structural candidate 满足：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

并且：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

如果 structural repair 提升 task 但破坏 S2，它只能算 diagnostic。

---

### H6：TimeAUC fail 主要来自 runtime path，而不是 learning dynamics

M13 已经满足：

$$
ValLossAUC_{\text{step,M13}}/ValLossAUC_{\text{step,B0}}=0.9370.
$$

但：

$$
ValLossAUC_{\text{time,M13}}/ValLossAUC_{\text{time,B0}}=1.3922.
$$

H6 成立标准：

若：

$$
ValLossAUC_{\text{step,candidate}}\leq ValLossAUC_{\text{step,B0}},
$$

但：

$$
ValLossAUC_{\text{time,candidate}}>ValLossAUC_{\text{time,B0}},
$$

则优先做 profiler/time accounting，而不是 optimizer dynamics。

---

### H7：S1 repair 必须建立在 teacher-free macro pass 之后

H7 是 route 顺序假设。S1 / TimeAUC / Profiler 很重要，但它们不能替代 official macro gate。

H7 成立标准：

只有在：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200
$$

且：

```text
external_teacher_used = 0
```

之后，S1/TimeAUC 才进入 official formal route。

在此之前，S1/TimeAUC 只能作为 diagnostic。

---

## 9. Candidate 设计

### 9.1 Baselines

```text
B0-MLP-AdamW
B2-MLP-C3-distill-diagnostic
M12-C3-distill-diagnostic
M13-teacher-free-official-baseline
```

### 9.2 Official teacher-free candidates

```text
TF0-M13-baseline
TF1-M13-repeated-validation-repro
TF2-M13-M5init-stabilized-no-teacher
TF3-M13-margin-stabilized-CE
TF4-M13-NLL-balanced-CE
TF5-M13-calibration-aware-CE
TF6-M13-warmup-stability-no-external
TF7-M13-SWA-final-averaging
TF8-M13-EMA-weights-no-logit-teacher
TF9-M13-bootstrap-consistency-no-external
```

### 9.3 No-external self-bootstrap candidates

```text
SB0-M13-EMA-self-teacher
SB1-M13-delayed-self-distill-40steps
SB2-M13-snapshot-self-distill-bestval
SB3-M13-dual-view-consistency
SB4-M13-SWA-logit-smoothing
SB5-M13-stopgrad-previous-epoch-teacher
SB6-M13-self-distill-temperature2
SB7-M13-self-distill-temperature4
```

### 9.4 Representation repair candidates

```text
RR0-M13-current
RR1-M13-sparseinterp-edge
RR2-M13-sparsespline-edge
RR3-M13-DWM2Lite-rbf-residual
RR4-M13-DWM2Lite-lut-residual
RR5-M13-prototype-KAN-head-lite
RR6-M13-margin-head-calibrated-KAN
RR7-M13-late-poly2-residual-lite
RR8-M13-local-cross-feature-bridge-lite
RR9-M13-feature-rank-preserving-init
```

### 9.5 System repair candidates, gated after macro pass

```text
SYS0-best-teacher-free-current
SYS1-root-input-lifetime-trim
SYS2-hidden-y-lifetime-trim
SYS3-streaming-root-input-cache
SYS4-streaming-hidden-y-cache
SYS5-fused-head-forward
SYS6-fused-linear-silu-backward
SYS7-phase-mapped-fused-forward
SYS8-dense-preserving-S1-combo
SYS9-time-accounting-clean-loop
```

---

## 10. 实验阶段总览

v7.9 分为十七个阶段：

```text
P0: code contract hardening and reproduction
P1: candidate metadata registry audit
P2: teacher-free baseline 10-seed reproduction
P3: repeated validation / split robustness audit
P4: M13 vs M12+C3 gap attribution
P5: no-external self-bootstrap branch
P6: targeted teacher-free objective / initialization repair
P7: representation repair branch using core primitives
P8: official teacher-free 10-seed confirmation
P9: teacher-free time accounting
P10: phase-mapped kernel profiler
P11: S1 memory attribution on teacher-free path
P12: dense-preserving fused / streaming package
P13: full-grid S1/S2 profiler
P14: conditional scaling and robustness
P15: geometry Pareto diagnostic
P16: route decision
P17: artifact and failure audit
```

P0-P1 是代码契约硬化。  
P2-P4 是理解差距。  
P5-P7 是 no-external repair。  
P8 是 official 结论阶段。  
P9-P13 是系统 formal route。  
P14-P15 是扩展验证。  
P16-P17 是决策与审计。

---

## 11. P0：code contract hardening and reproduction

### 11.1 目的

P0 先修代码契约，不寻找新 model result。v7.9 必须确保 route decision 的 strict / teacher-free / official eligibility 不再依赖硬编码 candidate id 或 postprocess 猜测。

### 11.2 必须实现

```text
CandidateSpec dataclass
candidate_registry.csv
contract_validator.csv
teacher_leak_audit.csv
strict_contract_enforcer
official_eligibility_filter
```

### 11.3 必须记录字段

```text
candidate_id
candidate_name
model_family
dense_cls
hidden_dim
basis_count
depth
init_policy
optimizer_policy
external_teacher_used
self_teacher_used
teacher_source
teacher_logits_used
teacher_forward_used
teacher_checkpoint_path
official_eligible
expected_manual_forward
expected_manual_backward
expected_manual_update
actual_manual_forward
actual_manual_backward
actual_manual_update
uses_loss_backward
uses_torch_autograd_graph
nonKAN_param_count
KAN_edge_param_count
head_is_kan
input_kan_is_kan
output_kan_is_kan
edge_param_coverage
strict_pass
```

### 11.4 判断标准

Contract pass：

```text
candidate_registry exists
all measured rows join candidate_registry by candidate_id
no unknown candidate_id
external_teacher_used is metadata-driven, not hardcoded
official candidates have external_teacher_used = 0
strict_pass uses full contract, not only head_is_kan + nonKAN
```

StrictPass：

$$
\text{StrictPass}
=
[\text{nonKAN}=0]
\land
[\text{head\_is\_kan}=1]
\land
[\text{manual\_forward}=1]
\land
[\text{manual\_backward}=1]
\land
[\text{manual\_update}=1]
\land
[\text{uses\_loss\_backward}=0].
$$

### 11.5 可视化

```text
p0_contract_heatmap.svg
p0_teacher_leak_matrix.svg
p0_candidate_registry_graph.svg
```

---

## 12. P1：candidate metadata registry audit

### 12.1 目的

确保所有 candidates 的 teacher status、official eligibility、model construction 都是 explicit metadata，不再靠 id 猜。

### 12.2 必跑对象

```text
B0
B2
M12
M13
all TF*
all SB*
all RR*
```

### 12.3 必须记录字段

```text
candidate_id
official_eligible
external_teacher_used
self_teacher_used
teacher_source
teacher_mode
teacher_logits_used
teacher_forward_used
can_enter_official_route
can_enter_diagnostic_route
reason_if_not_official
```

### 12.4 判断标准

Metadata pass：

```text
all official candidates external_teacher_used = 0
all external-teacher candidates official_eligible = 0
self-distill candidates external_teacher_used = 0 and self_teacher_used = 1
route decision rejects any row with missing metadata
```

### 12.5 可视化

```text
p1_official_eligibility_table.md
p1_teacher_status_bar.svg
```

---

## 13. P2：teacher-free baseline 10-seed reproduction

### 13.1 目的

确认 M13 / M12-no-distill 的真实 10-seed gap，使用修正后的 strict contract 和 metadata route。

### 13.2 必跑对象

```text
B0-MLP-AdamW
M13-teacher-free
M12-C3 diagnostic
B2 diagnostic
```

### 13.3 必须记录字段

```text
candidate
external_teacher_used
dataset
seed
val_acc
test_acc
val_loss
test_loss
ECE
NLL
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
seed_win_rate
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
```

### 13.4 判断标准

M13 reproduction：

$$
|\Delta Acc_{\text{M13}}-0.01784|\leq0.004.
$$

If:

$$
\Delta Acc_{\text{M13}}\geq0.0200,
$$

then proceed to P8 official confirmation immediately.

If:

$$
0.017\leq\Delta Acc_{\text{M13}}<0.020,
$$

then proceed to P3/P4.

### 13.5 可视化

```text
p2_m13_seedwise_gap.svg
p2_m13_bootstrap_ci.svg
p2_m13_vs_m12_diagnostic.svg
```

---

## 14. P3：repeated validation / split robustness audit

### 14.1 目的

判断 M13 的 validation gap 是否被 single validation split 低估。因为 M13 test gap 高于 validation gap，但 test 不能用于调参，必须用预注册 validation shards。

### 14.2 设置

```text
test set untouched
validation shards = 5 or 10
same train size budget
balanced shard construction
pre-registered shard seeds
```

### 14.3 必须记录字段

```text
candidate
split_id
seed
val_acc
val_loss
val_gap_vs_MLP
ECE
NLL
class_distribution
sample_count
macro_gap_split_mean
macro_gap_split_std
CI95_low_split
Holm_p_split
```

### 14.4 判断标准

Validation split effect：

$$
\Delta Acc_{\text{macro,val-split-mean,M13}}\geq0.0200.
$$

and:

$$
CI_{95\%,split}^{low}>0.
$$

No split effect：

$$
\Delta Acc_{\text{macro,val-split-mean,M13}}<0.0180.
$$

### 14.5 可视化

```text
p3_val_gap_by_split.svg
p3_split_distribution_ci.svg
p3_val_vs_test_gap_comparison.svg
p3_class_distribution_by_split.svg
```

---

## 15. P4：M13 vs M12+C3 gap attribution

### 15.1 目的

解释 external teacher 提供的 $+0.00273$ boost 来源。P4 不修模型，只做 attribution。

### 15.2 必跑对象

```text
M13-teacher-free
M12-C3 diagnostic
B0
B2 diagnostic
```

### 15.3 必须记录字段

```text
dataset
seed
candidate
val_acc
val_loss
ECE
NLL
margin_mean
margin_p10
margin_p50
confidence_mean
wrong_confidence_mean
logit_norm_mean
feature_effective_rank
feature_CKA_M13_M12
logit_KL_M12_M13
error_overlap_rate
hard_sample_gain
early_loss_slope
mid_loss_slope
late_loss_slope
```

### 15.4 判断标准

Calibration gap：

```text
error_overlap_rate >= 0.80
feature_CKA >= 0.95
NLL gap explains most delta
```

Representation gap：

$$
feature\_CKA<0.90
$$

or:

$$
rank_{M13}<0.95rank_{M12}.
$$

Margin gap：

$$
margin_{p10,M13}<0.95margin_{p10,M12}.
$$

Optimization timing gap：

```text
early_loss_slope_M13 worse than M12
mid/late slope similar
```

### 15.5 可视化

```text
p4_feature_cka_heatmap.svg
p4_logit_kl_distribution.svg
p4_margin_distribution.svg
p4_error_overlap_venn.svg
p4_loss_slope_comparison.svg
p4_teacher_gap_decomposition_dashboard.svg
```

---

## 16. P5：no-external self-bootstrap branch

### 16.1 目的

测试不依赖外部 C3 的 self-bootstrap 是否可以复现 teacher 的 mild boost。这是 v7.9 最重要的 repair branch。

### 16.2 Candidates

```text
SB0-M13-EMA-self-teacher
SB1-M13-delayed-self-distill-40steps
SB2-M13-snapshot-self-distill-bestval
SB3-M13-dual-view-consistency
SB4-M13-SWA-logit-smoothing
SB5-M13-stopgrad-previous-epoch-teacher
SB6-M13-self-distill-temperature2
SB7-M13-self-distill-temperature4
```

### 16.3 必须记录字段

```text
candidate
self_teacher_type
external_teacher_used
self_teacher_used
teacher_update_rule
temperature
val_acc
test_acc
val_gap_vs_MLP
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
grad_relerr_max
self_teacher_time_overhead
macro_delta_vs_M13
```

### 16.4 判断标准

Self-bootstrap useful：

$$
Acc_{\text{self-bootstrap}}-Acc_{\text{M13}}\geq0.0020.
$$

Official no-external pass：

```text
external_teacher_used = 0
```

and:

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

Time constraint：

$$
ValLossAUC_{\text{time,self}}\leq1.10\cdot ValLossAUC_{\text{time,M13}}.
$$

### 16.5 可视化

```text
p5_self_bootstrap_gain.svg
p5_self_bootstrap_time_overhead.svg
p5_self_bootstrap_vs_c3_teacher.svg
p5_self_bootstrap_learning_curve.svg
```

---

## 17. P6：targeted teacher-free objective / initialization repair

### 17.1 目的

只根据 P4 attribution 做有目标的 teacher-free repair，不做大 sweep。

### 17.2 Candidates

```text
TF2-M13-M5init-stabilized-no-teacher
TF3-M13-margin-stabilized-CE
TF4-M13-NLL-balanced-CE
TF5-M13-calibration-aware-CE
TF6-M13-warmup-stability-no-external
TF7-M13-SWA-final-averaging
TF8-M13-EMA-weights-no-logit-teacher
```

### 17.3 必须记录字段

```text
candidate
repair_type
external_teacher_used
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
ValLossAUC_step
ValLossAUC_time
grad_relerr_max
memory_ratio_max
step_ratio_max
macro_delta_vs_M13
```

### 17.4 判断标准

Repair useful：

$$
Acc_{\text{repair}}-Acc_{\text{M13}}\geq0.0022.
$$

Official macro pass：

$$
\Delta Acc_{\text{macro,val,repair}}\geq0.0200.
$$

Efficiency preservation：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

Grad preservation：

$$
grad\_relerr_{max}\leq10^{-4}.
$$

### 17.5 可视化

```text
p6_repair_gain_bar.svg
p6_task_efficiency_pareto.svg
p6_repair_auc_step_time.svg
p6_repair_nll_ece_bar.svg
```

---

## 18. P7：representation repair branch using core primitives

### 18.1 目的

如果 P4 指向 representation/rank/margin gap，则使用 core 中已有 low-live-set primitive 做 teacher-free architecture repair。

### 18.2 Candidates

```text
RR0-M13-current
RR1-M13-sparseinterp-edge
RR2-M13-sparsespline-edge
RR3-M13-DWM2Lite-rbf-residual
RR4-M13-DWM2Lite-lut-residual
RR5-M13-prototype-KAN-head-lite
RR6-M13-margin-head-calibrated-KAN
RR7-M13-late-poly2-residual-lite
RR8-M13-local-cross-feature-bridge-lite
RR9-M13-feature-rank-preserving-init
```

### 18.3 必须记录字段

```text
candidate
structural_change
dense_cls
external_teacher_used
nonKAN_param_count
edge_param_delta
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
feature_rank
margin_p10
grad_relerr_max
memory_ratio_max
step_ratio_max
S2_shape_count
```

### 18.4 判断标准

Representation repair useful：

$$
Acc_{\text{RR}}-Acc_{\text{M13}}\geq0.0022.
$$

Feature repair：

$$
rank_{\text{RR}}\geq1.05rank_{\text{M13}}
$$

or:

$$
margin_{p10,RR}\geq1.05margin_{p10,M13}.
$$

Official pass：

$$
\Delta Acc_{\text{macro,val,RR}}\geq0.0200.
$$

and:

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

### 18.5 可视化

```text
p7_representation_repair_gain.svg
p7_feature_rank_vs_acc.svg
p7_margin_vs_acc.svg
p7_structural_overhead_pareto.svg
```

---

## 19. P8：official teacher-free 10-seed confirmation

### 19.1 目的

对 P5/P6/P7 产生的 best no-external candidate 做 official 10-seed confirmation。

### 19.2 必跑对象

```text
B0-MLP-AdamW
M13-baseline
best-P5-self-bootstrap
best-P6-objective-repair
best-P7-representation-repair
M12-C3 diagnostic only
B2 diagnostic only
```

### 19.3 必须记录字段

```text
candidate
external_teacher_used
self_teacher_used
dataset
seed
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
```

### 19.4 判断标准

Teacher-free official pass：

```text
external_teacher_used = 0
```

and:

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

FullGridS2Pass：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

### 19.5 可视化

```text
p8_teacher_free_bootstrap_ci.svg
p8_seedwise_gap_boxplot.svg
p8_official_candidate_scorecard.svg
p8_task_efficiency_pareto.svg
```

---

## 20. P9：teacher-free time accounting

### 20.1 目的

如果 P8 过 macro，则重新评估 TimeAUC。M13 已有 step AUC 优势但 time AUC 失败，因此 runtime/loop/profiler 是 formal blocker。

### 20.2 必跑对象

```text
B0
M13-baseline
best-teacher-free-candidate
best-self-bootstrap-candidate
```

### 20.3 必须记录字段

```text
candidate
external_teacher_used
self_teacher_used
wall_clock_time_total
train_step_time
forward_time
backward_time
update_time
self_teacher_time
validation_time
logging_time
cuda_sync_time
data_loading_time
unknown_time_fraction
ValLossAUC_step
ValLossAUC_time_train_only
ValLossAUC_time_total
time_to_target_acc
```

### 20.4 判断标准

TimeAccountingPass：

```text
unknown_time_fraction <= 0.10
validation/logging/sync separated
self-teacher overhead separated if any
```

Teacher-free TimeAUCPass：

$$
ValLossAUC_{\text{time,total,TF}}\leq ValLossAUC_{\text{time,B0}}.
$$

If step AUC pass but time AUC fail, proceed to P10/P12 profiler and fused path.

### 20.5 可视化

```text
p9_teacher_free_time_breakdown.svg
p9_total_vs_train_only_auc.svg
p9_time_to_target.svg
p9_step_auc_vs_time_auc.svg
```

---

## 21. P10：phase-mapped kernel profiler

### 21.1 目的

建立 teacher-free path 的真实 kernel attribution，避免 proxy。必须先 profiler pass，再做 fused package claim。

### 21.2 必跑对象

```text
PROF0-B0
PROF1-M13-baseline
PROF2-best-teacher-free
PROF3-best-self-bootstrap
PROF4-M12-C3 diagnostic
```

### 21.3 必须记录字段

```text
candidate
kernel_count_total
mapped_kernel_time_fraction
unknown_kernel_time_fraction
kernel_count_forward
kernel_count_backward
kernel_count_update
kernel_count_validation
top_kernel_name_1
top_kernel_phase_1
top_kernel_time_1
small_kernel_count_under_10us
layout_conversion_count
cuda_memcpy_time
cuda_sync_time
```

### 21.4 判断标准

ProfilerPass：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

Kernel fragmentation：

$$
N_{\text{small kernels under 10us}}\geq0.30N_{\text{kernel total}}.
$$

### 21.5 可视化

```text
p10_kernel_timeline.svg
p10_kernel_count_by_phase.svg
p10_top_kernel_time_bar.svg
p10_mapped_vs_unknown_kernel_time.svg
```

---

## 22. P11：S1 memory attribution on teacher-free path

### 22.1 目的

S1 repair 必须在 teacher-free final candidate 上做，不只在 M12+C3 上做。

### 22.2 必跑对象

```text
B0
M13-baseline
best-teacher-free
best-self-bootstrap
```

### 22.3 必须记录字段

```text
candidate
batch_size
memory_ratio
step_ratio
root_input_cache_MB
hidden_y_cache_MB
manual_cache_MB
optimizer_state_MB
head_temp_MB
allocator_padding_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

### 22.4 判断标准

S1 attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
```

Actionable source：

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.02.
$$

### 22.5 可视化

```text
p11_s1_memory_waterfall.svg
p11_bs512_memory_source.svg
p11_memory_ratio_by_batch.svg
```

---

## 23. P12：dense-preserving fused / streaming package

### 23.1 目的

修复 teacher-free candidate 的 S1/TimeAUC/forward bottleneck。只允许 dense-preserving repair，不允许破坏 macro signal。

### 23.2 Candidates

```text
SYS0-best-teacher-free-current
SYS1-root-input-lifetime-trim
SYS2-hidden-y-lifetime-trim
SYS3-streaming-root-input-cache
SYS4-streaming-hidden-y-cache
SYS5-fused-head-forward
SYS6-fused-linear-silu-backward
SYS7-phase-mapped-fused-forward
SYS8-dense-preserving-S1-combo
SYS9-time-accounting-clean-loop
```

### 23.3 必须记录字段

```text
candidate
components
external_teacher_used
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S1_shape_count
S2_shape_count
kernel_count_total
mapped_kernel_time_fraction
materialized_tensor_count
macro_delta_vs_parent
```

### 23.4 判断标准

Task preservation：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

and:

$$
\Delta Acc_{\text{new}}\geq \Delta Acc_{\text{parent}}-0.003.
$$

FullGridS1：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

TimeAUCPass：

$$
ValLossAUC_{\text{time,new}}\leq ValLossAUC_{\text{time,B0}}.
$$

### 23.5 可视化

```text
p12_package_task_efficiency_pareto.svg
p12_s1_s2_shape_heatmap.svg
p12_time_auc_repair_bar.svg
p12_kernel_mapping_after_repair.svg
```

---

## 24. P13：full-grid S1/S2 profiler

### 24.1 目的

确认 final teacher-free candidate 仍然 full-grid S2/S1，不允许只看 mean。

### 24.2 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch:
  128
  256
  512
```

### 24.3 必须记录字段

```text
candidate
dataset
batch
memory_ratio
step_ratio
forward_ratio
backward_ratio
S2_pass
S1_pass
top_memory_source
kernel_count_total
```

### 24.4 判断标准

FullGridS2Pass：

```text
S2 shapes = 9/9
```

FullGridS1Pass：

```text
S1 shapes = 9/9
```

### 24.5 可视化

```text
p13_memory_heatmap.svg
p13_step_heatmap.svg
p13_s1_s2_boundary.svg
```

---

## 25. P14：conditional scaling and robustness

### 25.1 打开条件

Only open if:

```text
TeacherFreeMacroSignificantPass = true
FullGridS2Pass = true
GradPass = true
```

### 25.2 设置

```text
train_size:
  256
  512
  1024
  1536
  4096

noise:
  label_noise = 0.05, 0.10, 0.20
  input_noise = 0.05, 0.10

hidden_dim:
  48,64,96,128
```

### 25.3 必须记录字段

```text
candidate
train_size
noise_type
hidden_dim
val_acc
test_acc
ECE
NLL
accuracy_drop
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
```

### 25.4 判断标准

Sample efficiency pass：

$$
AUC_{\text{data,KAN}}>AUC_{\text{data,MLP}}.
$$

Robustness pass：

$$
AccDrop_{\text{KAN}}<AccDrop_{\text{MLP}}
$$

for at least two settings.

### 25.5 可视化

```text
p14_accuracy_vs_train_size.svg
p14_sample_efficiency_auc.svg
p14_noise_robustness.svg
```

---

## 26. P15：geometry Pareto diagnostic

### 26.1 打开条件

Only after teacher-free MacroSignificant + S2.

### 26.2 必须记录字段

```text
lambda_geo
geometry_loss_type
val_acc
test_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
geometry_metric_value
memory_ratio
step_ratio
```

### 26.3 判断标准

Useful geometry：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005.
$$

and one of:

$$
ECE_{\lambda}<ECE_{\lambda=0},
$$

$$
NLL_{\lambda}<NLL_{\lambda=0},
$$

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

### 26.4 可视化

```text
p15_geometry_pareto_frontier.svg
p15_accuracy_vs_geometry.svg
p15_ece_nll_vs_geometry.svg
```

---

## 27. P16：route decision

### 27.1 Survivor types

```text
S0:
  TeacherFreeMacroSignificant + FullGridS1 + TimeAUCPass + ProfilerPass

S1:
  TeacherFreeMacroSignificant + FullGridS2 + TimeAUCPass

S2:
  TeacherFreeMacroSignificant + FullGridS2 but TimeAUC fail

S3:
  Teacher-free near-pass only, external teacher pass

S4:
  External teacher only success

S5:
  Self-bootstrap no-external success

S6:
  Teacher-free macro pass but S2 fail

S7:
  GradFail

S8:
  Profiler/time incomplete

S9:
  Code contract incomplete
```

### 27.2 Route cases

```text
R1-TeacherFreeFullSystemAdvantage:
  S0. Official PureKAN-NG success.

R2-TeacherFreeS2QualityAdvantage:
  S2. Teacher-free quality/S2 success, system time still open.

R3-SelfBootstrapOfficialNoExternalTeacher:
  Self-bootstrap passes; report as no-external-teacher route.

R4-TeacherAssistedOnly:
  Only C3 teacher passes; official standalone goal not met.

R5-TeacherFreeNearPass:
  Teacher-free remains around +0.018 to +0.020; continue representation/initialization.

R6-KernelizationNeeded:
  Teacher-free macro pass but S1/time fail due to kernel/memory.

R7-NoReproduction:
  Teacher-free signal does not reproduce.

R8-CodeContractFail:
  Candidate metadata or strict contract incomplete; no scientific claim allowed.
```

### 27.3 必须记录字段

```text
candidate
route
external_teacher_used
self_teacher_used
strict_pass
grad_pass
teacher_free_macro_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
profiler_pass
code_contract_pass
val_gap_vs_MLP
test_gap_vs_MLP
memory_ratio_max
step_ratio_max
primary_blocker
next_required_implementation
```

### 27.4 可视化

```text
p16_route_decision_tree.svg
p16_final_scorecard.svg
p16_system_pareto.svg
```

---

## 28. P17：artifact and failure audit

### 28.1 Required artifacts

```text
run_manifest.json
candidate_registry.csv
contract_validator.csv
teacher_leak_audit.csv
provenance_audit.csv
gate_config.json
p0_code_contract.csv
p1_candidate_metadata_audit.csv
p2_teacher_free_baseline.csv
p3_validation_split_audit.csv
p4_teacher_gap_attribution.csv
p5_self_bootstrap.csv
p6_teacher_free_repair.csv
p7_representation_repair.csv
p8_teacher_free_10seed.csv
p9_time_accounting.csv
p10_phase_mapped_profiler.csv
p11_s1_memory_attribution.csv
p12_fused_streaming_package.csv
p13_fullgrid_profiler.csv
p14_scaling_robustness.csv
p15_geometry_pareto.csv
p16_route_decision.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 28.2 Failure taxonomy

```text
F1_code_contract_fail
F2_candidate_metadata_missing
F3_teacher_leak_detected
F4_teacher_free_macro_fail
F5_validation_split_underpowered
F6_self_bootstrap_fail
F7_representation_gap
F8_external_teacher_only_success
F9_s2_fail
F10_s1_fail
F11_time_auc_fail
F12_grad_fail
F13_profiler_incomplete
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 29. 第一轮推荐执行顺序

### Step 1：P0/P1 code contract hardening

先修 runner。没有 metadata-driven teacher status 和完整 StrictPass，后续结果都会被质疑。

### Step 2：P2 M13 reproduction

用新 runner 复现 M13，确认 code hardening 没改变数字。

### Step 3：P3 validation split audit

先判断 `+0.01784` 是否被 validation split 低估。因为 test gap `+0.02298` 不能直接用于成功，但它强烈提示需要 split audit。

### Step 4：P4 gap attribution

拆 C3 teacher 的 `+0.00273` mild boost 到底来自 calibration、margin、feature rank、hard samples 还是 early dynamics。

### Step 5：P5 self-bootstrap

这是最值得优先实现的 no-external repair，因为它最接近 C3 teacher 的作用机制，但不使用外部 teacher。

### Step 6：P7 representation repair

如果 P4 指向 representation gap，再用 core 中的 sparse/DWM2Lite/GEMM-native primitive 做结构修复，不再粗扫 hidden/basis。

### Step 7：P8 official 10-seed

只有 no-external candidate 过线，才进入 official claim。

### Step 8：P9-P13 system path

teacher-free macro pass 后，才把 TimeAUC、Profiler、S1 作为 official formal route。

---

## 30. 停止条件

### 30.1 成功停止

出现以下任一情况立即复盘：

```text
TeacherFreeMacroSignificant + FullGridS2 + GradPass + CodeContractPass
TeacherFreeMacroSignificant + FullGridS1 + GradPass + CodeContractPass
TeacherFreeMacroSignificant + FullGridS1 + TimeAUCPass + ProfilerPass + CodeContractPass
```

### 30.2 失败停止

出现以下任一情况停止对应路线：

```text
1. code contract cannot verify manual_forward/manual_backward/manual_update；
2. candidate metadata missing for any route candidate；
3. teacher-free 10-seed macro gap < +0.015；
4. repeated validation mean gap < +0.018；
5. all self-bootstrap repairs improve < +0.001；
6. representation repairs improve < +0.001；
7. any official candidate uses external teacher；
8. S1 repair makes teacher-free macro gap < +0.0200；
9. GradPass fails and cannot be fixed by numerical equivalent rewrite；
10. no-fake/no-proxy audit fail。
```

---

## 31. 成功与失败解释规则

### Case A：Teacher-free candidate crosses +0.020 and S2

这是 v7.9 最重要的成功。可以写：

```text
PureKAN-NG teacher-free official quality advantage is established.
```

### Case B：Self-bootstrap crosses +0.020

可以写：

```text
No-external-teacher training recipe succeeds.
```

但必须明确它是 self-bootstrap recipe，不是 pure CE-only。

### Case C：Repeated validation passes but original validation fails

说明原始 validation hard gate 可能太敏感；需要预注册更稳健 validation protocol。不能用 test 替代 validation。

### Case D：Only C3 teacher crosses +0.020

不能写 official success。只能写：

```text
PureKAN-NG is currently teacher-assisted only.
```

### Case E：Teacher-free passes macro but TimeAUC fails

说明 quality advantage 成立，但 system wall-clock 未闭合。继续 profiler/kernel path。

### Case F：Teacher-free passes macro but S1 fails

说明 architecture advantage 成立，S1 memory/kernel 仍需 repair。

### Case G：Teacher-free remains around +0.018 to +0.0195

说明路线非常接近但还没成立。下一步应该转向 representation-level design，而不是继续 schedule 小修。

### Case H：Code contract fails

不允许做科学 claim。必须先修 runner/metadata/contract。

---

## 32. 最终建议

v7.9 的一句话策略是：

$$
\boxed{
\text{先把代码契约变成可信 official runner，再闭合 M13 的 teacher-free autonomous margin。}
}
$$

当前应这样定位：

```text
M13:
  official teacher-free target，稳定 positive，但 near-pass

M12+C3:
  diagnostic assisted success，不是 official

B2:
  MLP teacher diagnostic/fairness control

self-bootstrap:
  下一步最值得实现的 no-external-teacher route

code contract:
  v7.9 第一优先级，不再靠 candidate id hardcode

phase-mapped profiler / S1:
  teacher-free macro pass 后进入 official formal success path
```

如果 v7.9 能让 M13-derived no-external candidate 从：

$$
+0.01784
$$

提升到：

$$
\geq +0.0200
$$

并保持：

$$
r_{\text{mem,max}}\leq1.05,\quad r_{\text{step,max}}\leq1.50,
$$

同时通过完整 code contract，那么项目才能真正回到最初目标：

$$
\boxed{
\text{PureKAN-NG 自身系统性优于 MLP-AdamW。}
}
