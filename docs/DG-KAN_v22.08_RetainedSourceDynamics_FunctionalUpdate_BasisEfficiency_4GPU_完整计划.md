# DG-KAN v22.08：Retained-Source Dynamics Functional Update + Basis Efficiency Closure + 4GPU 完整计划

> 版本：v22.08 execution plan  
> 目标读者：不假设读者熟悉全部历史实验；本文开头先说明项目目标、当前状态和为什么 v22.08 必须换 Functional Update 思路。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no seed-specific rule；no fake/proxy/CPU offload；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit、debt readback 和 gate，不能生成方向。

---

## 0. 项目总目标与 v22.07 后的真实状态

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{最终得到比普通 backprop / AdamW 更好的模型。}
}
$$

这里的“更好”不是某个局部 horizon 正值，也不是某个 optimizer 名字更先进，而是同时满足：

```text
1. strict PureKAN base 不依赖 non-KAN 参数；
2. forward / backward / update / memory 与 same-param MLP 接近；
3. functional update 的收益超过 AdamW / SGD / NoOp / random / stable-random / same-overhead controls；
4. source 能从 early chain 留到 h3200/h4800，不能只是 late rebound；
5. tail / LineC / calibration / AUCtime debt 不爆；
6. 如果 MLP 也成功，要写成 generic training-dynamics insight，不能写成 KAN-specific；
7. 只有 KAN 在同机制下超过 MLP，才讨论 KAN-specific functional advantage。
```

v22.07 后，当前真实状态如下：

```text
代码：
  required source path 基本自包含，compile/import/LineC/source-chain/terminal-retention/metric-solver tests 通过。
  但 C2 repair 存在一个重要实现语义问题：target rescale / block-role repair 没有重新求解 solver，而是复用了旧 job_order 的 C2 solver diagnostics。

效率：
  D-CHE / D-FOU 继续是 OfficialEfficientCarrier。
  D-RAT / D-RBF 已经有 near-E1 / limited smoke 线索，但 robust multibatch officialization 未闭合。

Functional update：
  原始 C0/C1/C2/C3 全部没过。
  repair 后 C0 repair pass=1，C1 repair pass rows=49，C2 repair pass rows=16，但 C3 repair smoke pass rows=0。
  source-theory repair 有 moderate observer signal，AUC 约 0.795，但没有 official early chain。
  observer2-observer7 六类 train-only source observer 全部没能打开 C1/h1600/h3200 retention。
  final route 是 local source-observer no-go boundary，不是 universal no-go。
```

v22.08 的核心判断是：

$$
\boxed{
\text{当前 blocker 不是“metric 无用”，也不是“solver 完全做不到”，}
\text{而是 train-only source observability 仍没有找到第一性定义。}
}
$$

因此 v22.08 不再继续扩同族 observer/solver 名字池，而是把 Functional Update 改成 **Retained-Source Dynamics**：先定义什么 train-only 信号有资格成为 retained source，再用函数空间 metric solver 写入并在后续训练动力学中保留。

---

## 1. v22.07 独立问题诊断

### 1.1 代码层问题：C2 repair 不能复用旧 solver diagnostics

v22.07 的 `C2_target_rescale_solver_audit` 里，`target_repair_variant` 包含：

```text
all_x1 / all_x4 / all_x16 / all_x64
hidden_only_x*
readout_only_x*
```

但当前实现中，C2 repair rows 用 `job_order` 去读取已经存在的 `v22_07_c2_metric_solver_matrix.csv`：

$$
\text{C2 repair}(target\_scale, block\_role)
\leftarrow
\text{old C2 diagnostics}(job\_order)
$$

这意味着：

```text
1. target scale 没有重新求解 projection；
2. hidden_only / readout_only block role 没有重新做 block-restricted solver；
3. projection_residual_Gf 和 ActuationR2 在 summary 中对不同 target variants 近似常数；
4. C2 repair pass rows 只能看作“C1 target repair + old C2 readback”，不能当成真正 solver repair。
```

v22.08 必须修正：每个 C2 repair variant 都要重新构造 target、重新运行 solver、重新计算 projection residual、ActuationR2、JVP/VJP/CG count。否则不能进入 C3。

### 1.2 Functional 层问题：local source-observer no-go 不是 universal no-go

v22.07 的 final route 是：

```text
C3-SourceObserverLocalNoGoBoundary
```

这句话的含义必须严格解释：

```text
它只说明：v22.07 已尝试的 train-only observer families 没有打开 source formation。
它不说明：所有 possible source observability theory 都失败。
```

因此下一步不能继续在同族 observer 上调 scale / cap / gate，而要换问题：

$$
\boxed{
\text{source observer 不能只是预测历史 source label，}
\text{它必须是一个可提交前使用的 retained-source certificate。}
}
$$

### 1.3 效率层问题：D-CHE/D-FOU 已可用，D-RAT/D-RBF 仍需 robust closure

D-CHE / D-FOU 现在不再是 functional failure 的主要解释。它们应该继续作为主 KAN carrier。

D-RAT / D-RBF 不能继续躺在 blocked 表里，但也不能直接进入 full FU。它们必须先完成 multibatch robust officialization：

```text
batch 128 / 256 / 512 / 1024；
forward / backward / update / memory；
component telemetry complete；
production fused kernel complete；
functional runner kernel match；
no hidden fallback path。
```

---

## 2. 哪些旧思路可以重开，哪些必须停止

### 2.1 可以重开的旧方向

这些方向过去失败，不一定是思想失败，可能是旧实现、旧 metric、旧 kernel、旧 route 使其失败。

#### A. Function metric / basis natural update

v15.02.1 的 function metric 失败，v13.2 / v13.3 的 basis-natural / task-family robust metric 失败，但当时主要还是 proxy / old-kernel / old-retention gate。现在 D-CHE/D-FOU 已有高效 carrier，metric 可以作为真实 solver 的几何，而不是 flat-gradient filter。

重开方式：

```text
不重开 old G-token / BN-token。
只重开：JVP/VJP-based function-space solver + basis-channel metric pullback。
```

#### B. Operator-level basis-channel functional

v13.4 的 operator-level basis-channel思路比后来的许多 Fxx 工程变体更接近 functional update 本质。它可以重开，但必须加入：

```text
metric-defined target；
projection residual；
actuation fidelity；
source formation；
terminal retention；
controls。
```

#### C. PopRisk / SNR

v13.6-v13.8 的 PopRisk/SNR 不能按旧方式作为 one-shot parameter gate 或 cover objective 重开。它应该作为 drift-diffusion source observer 的一部分：

$$
SNR = \frac{\|\mathbb{E}[g_i]\|^2}{\operatorname{Var}(g_i)+\epsilon}
$$

但它必须预测 **early retained source chain**，不能只预测 train loss 或 late positive。

#### D. Graph-free analytic adjoint

v6.x 的 graph-free analytic adjoint 当时卡在 primitive/kernel efficiency。但现在 D-CHE/D-FOU 已高效，D-RAT/D-RBF 也有 micro-kernel 线索。可以在 v22.08 中重新要求 metric solver 的 JVP/VJP 与 KAN kernel path对齐，避免 solver 成为新的 autograd-heavy bottleneck。

### 2.2 必须停止的方向

以下方向不再进入主预算：

```text
1. action bank / controller / reset route；
2. cover objective 小修；
3. M31/M32 同族小修；
4. G/N/Q token 扩展；
5. 只调 target scale / cap / floor / lookahead / hold；
6. high ActuationR2 target 继续放大；
7. 同族 observer2-observer7 继续微调。
```

原因：这些方向已经多次显示 control-equivalent、not reproducible、late rebound only、或 local source-observer no-go。

---

## 3. 当前研究进展给 Functional Update 的真正启发

v22.08 不机械换 optimizer，而从已有训练理论中抽取算法原则。

### 3.1 从 Muon 曲率视角得到的原则

Muon 曲率研究提示：两个 update 的一阶收益可以相近，但长期效果差异可能来自二阶曲率惩罚，尤其是 Normalized Directional Sharpness。对我们而言，source 能到 h3200 但 h4800 erosion，可能不是 source amplitude 不够，而是 source direction 的 NDS 太高。

因此 v22.08 记录：

$$
I^{(1)} = -\langle g, u\rangle
$$

$$
I^{(2)} = \frac{1}{2}u^T H u
$$

$$
NDS(u)=\frac{u^T H u}{\|u\|^2+\epsilon}
$$

并做 block-level NDS：

```text
hidden block NDS；
readout block NDS；
D-CHE degree-readout NDS；
D-FOU band-readout NDS；
D-RAT num/den NDS；
D-RBF local-center/readout NDS。
```

### 3.2 从 Nora 得到的原则

Nora 的启发不是“换成 Nora optimizer”，而是：matrix block 的 row norm 与 angular velocity 稳定性很重要。对于 MLP hidden/readout、D-CHE degree-readout、D-FOU band-readout，source update 应避免过强 radial jitter。

记录：

$$
radial(u,W)=\frac{\langle u,W\rangle}{\|W\|^2+\epsilon}W
$$

$$
angular(u,W)=u-radial(u,W)
$$

以及：

```text
row_norm_drift；
row_angular_velocity；
source_projection_radial；
source_projection_angular；
terminal_retention_delta。
```

### 3.3 从 SOAP / block-coordinate optimizer 得到的原则

SOAP 的启发是：adaptive moment 不必在原始 flat parameter coordinate 中运行，可以在 slowly-changing coordinate basis 中运行。v22.08 应把 source-state 按 block 组织：

```text
MLP hidden/readout block；
D-CHE degree-readout block；
D-FOU band-readout block；
D-RAT numerator/denominator block；
D-RBF local-center/readout block。
```

### 3.4 从 Schedule-Free / AdEMAMix 得到的原则

单一 EMA 不能同时保留近期梯度和旧梯度。v22.08 的 source state 必须拆成：

```text
short source state：负责 h100/h400/h800 source formation；
long source state：负责 h1600/h3200/h4800 source retention；
fast iterate：负责探索；
slow iterate：负责保存 source。
```

### 3.5 从 generalization signal/reservoir 理论得到的原则

训练可动方向不等于泛化方向。真正 source 应进入 signal channel，而不是 reservoir。v22.08 的 observer 不再只看 B2 transfer gain，而要估计：

```text
coherent drift；
idiosyncratic diffusion；
signal-channel energy；
reservoir leakage；
noise-in-signal risk。
```

### 3.6 从 Deep Manifold 得到的原则

Functional update 不是孤立一步，而是训练系统的 boundary condition。一个好的 FU 应改变后续 fixed-point convergence path，而不是在一步上制造高 ActuationR2。

因此 v22.08 必须记录 micro-trajectory：

```text
同一 state 下多条短训练路径；
不同 minibatch order；
不同 optimizer micro-dynamics；
source 是否在这些路径中一致出现。
```

---

## 4. v22.08 总体假设

### H1：v22.07 的失败来自 source observability，而不是 metric solver 理论失败

如果 train-only observer 无法在提交前识别 retained source，任何 solver 都只是在把错误 target 执行得更精确。

判定标准：

```text
C0 observer 在 heldout candidate families 上不能预测 early-chain source；
即使 C1/C2 局部 pass，C3 source formation 仍失败。
```

### H2：source observer 必须预测 micro-trajectory invariance，而不是单步 B2 gain

真正 retained source 应该在多个合法 train-stream micro-trajectory 中保持方向一致。

判定标准：

```text
同一 candidate 在 B1/B2/B3 split、batch-order perturbation、SGD/AdamW/Momentum micro-dynamics 下，
source direction sign / transfer sign / low-NDS score 保持一致。
```

### H3：metric solver 必须重新求解每个 target / block / scale，不得复用旧 solver diagnostics

判定标准：

```text
每个 target_repair_variant 都有独立 solver row；
projection_residual_Gf、ActuationR2、JVP/VJP/CG iterations 随 target/block 变化；
不存在 target variants 共用同一 old job_order projection readback。
```

### H4：source 必须写入 optimizer-state / slow-state，不能只做 direct parameter commit

如果 FU 只是一小步 parameter perturbation，后续训练很容易 washout。v22.08 要测试 source 是否应写入 slow momentum / block-state。

### H5：KAN source-channel mismatch 只有在 MLP source chain 打开后才有意义

若 MLP 都没有 early-to-h3200 source chain，KAN mapping 应 fail-closed。若 MLP 开了，再问 D-CHE/D-FOU 是否能承载同一 source channel。

---

## 5. 实验总结构

v22.08 分为五条线：

```text
Line A: Code / metric / solver correctness hard gate。
Line B: Basis efficiency closure：D-CHE/D-FOU reconfirm + D-RAT/D-RBF robust multibatch。
Line C: Retained-source observer theory：new observability principle。
Line D: Metric-as-dynamics solver：observer -> target -> solver -> source-state integration。
Line E: KAN source-channel mapping：只在 MLP source formation 通过后进入。
```

四卡并行：

```text
GPU0:
  Line A + Line C observer no-commit + C0/C1 heldout challenge。

GPU1:
  Line D MLP metric-as-dynamics solver + source-state integration。

GPU2:
  D-CHE / D-FOU efficiency reconfirm + KAN mapping smoke。

GPU3:
  D-RAT / D-RBF multibatch officialization + limited smoke。
```

必须生成：

```text
v22_08_runnable_queue.csv
v22_08_gpu_assignment_manifest.csv
v22_08_gpu_utilization_timeline.csv
v22_08_idle_violation.csv
v22_08_queue_drain_report.csv
```

硬规则：

```text
runnable queue 非空 && 任一 GPU idle > 10 min
=> execution_contract_violation = 1
=> final route 不能写 completed-no-go。
```

---

## 6. Line A：代码 / 指标 / solver 正确性硬门

### 6.1 目标

防止再次出现：

```text
1. code packet 在原始 repo pass，但解压后缺依赖；
2. target repair 没重新求解 solver；
3. mechanism 名字不同但 update / function displacement 相同；
4. kernel official status 和 profiler status 不一致；
5. C3 使用 summary 代替真实 fresh run。
```

### 6.2 必跑检查

在最终 zip 解压目录执行：

```text
compileall；
required import closure；
self_contained_import_check；
LineC fast/channel golden；
source-chain / terminal-retention tests；
metric solver tests；
semantic alias audit；
C2 repair recompute tests；
kernel gradcheck；
profiler phase tests；
forbidden direction audit。
```

### 6.3 新增 C2 repair correctness tests

构造两个不同 target scale：

$$
\Delta f_1 = \Delta f
$$

$$
\Delta f_2 = 64\Delta f
$$

要求：

```text
projection_residual_Gf 不得完全相同；
update_norm 应随 target scale 变化，除非 trust region 明确饱和；
ActuationR2 必须独立重算；
JVP/VJP count 必须记录；
block_role=hidden_only/readout_only 时必须执行 block-restricted solver 或 fail-closed，不能读旧 all-block projection。
```

### 6.4 通过标准

```text
required_source_files = 100%
self_contained_import_check = 1
LineC golden pass = 1
source-chain tests pass = 1
C2 repair recompute tests pass = 1
semantic_alias_undeclared = 0
kernel status consistency = 1
forbidden_direction_violation = 0
```

不通过时，Codex 必须先修代码，不能跑 scientific route。

---

## 7. Line B：Basis efficiency closure

### 7.1 D-CHE / D-FOU reconfirm

D-CHE / D-FOU 只做 reconfirm，不继续大规模效率搜索。

记录：

```text
forward_ratio_vs_mlp；
backward_ratio_vs_mlp；
step_ratio_vs_mlp；
memory_ratio_vs_mlp；
functional_direction_ms；
metric_solver_ms；
LineC_audit_ms；
horizon_readback_ms；
same_kernel_functional_runner_proof；
fallback_kernel_used；
official_fused_kernel_complete。
```

成功标准：

```text
D-CHE:
  >=2 variants × >=3 batch sizes S1 pass；
  forward <=1.25；
  step <=1.25；
  memory <=1.05；
  fallback_kernel_used = 0。

D-FOU:
  FOU-R3 >=3 batch sizes S1 pass；
  forward <=1.25；
  step <=1.25；
  memory <=1.05；
  fallback_kernel_used = 0。
```

### 7.2 D-RAT / D-RBF robust multibatch officialization

D-RAT / D-RBF 不能只看单 batch 或 micro-near-E1。

测试：

```text
batch = 128,256,512,1024
D-RAT variants:
  rational-k4 trainpath；
  numerator/denominator fused；
  reciprocal approximation；
  telemetry-free train path。

D-RBF variants:
  local-K sparse trainpath；
  active-center threshold；
  exp approximation；
  no-dense materialization；
  local backward fused。
```

记录：

```text
forward_ratio；
backward_ratio；
step_ratio；
memory_ratio；
component telemetry completeness；
manual grad relerr/cos；
functional runner kernel match；
official fused complete；
no materialize complete。
```

Near-E1 标准：

```text
forward <= 3.0
step <= 2.0
memory <= 1.2
gradcheck_pass = 1
```

Robust official 标准：

```text
>=3/4 batch sizes:
  forward <= 1.25
  step <= 1.25
  memory <= 1.05
component_telemetry_complete = 1
functional_runner_kernel_match = 1
```

若 D-RAT/D-RBF 不能 robust official，则只允许 limited smoke，不进入 full functional matrix。

---

## 8. Line C：Retained-source observer theory

### 8.1 为什么这是 v22.08 的核心

v22.07 已经尝试了多个 observer families，但 source observer local no-go 成立。下一步不能再继续同族微调，而要重新定义 observer 的第一性标准。

一个合法 source observer 必须满足：

```text
1. 只使用 train-stream precommit 信息；
2. 能预测 early chain source，不只是 h3200 late positive；
3. 能区分 random/sign/corrupt/control target；
4. 在 heldout dataset/seed/method family 上仍有效；
5. 选择出的 candidate 在 fresh C3 中能产生 source formation。
```

### 8.2 Observer families to test

#### C-O1：Drift-Diffusion Retention Observer

对每个候选 target，构造多 split per-example output displacement：

$$
D = \|\mathbb{E}[\Delta f_i]\|^2
$$

$$
V = \mathbb{E}\|\Delta f_i - \mathbb{E}[\Delta f_i]\|^2
$$

$$
R_{drift} = \frac{D}{V+\epsilon}
$$

记录：

```text
coherent_drift_norm；
diffusion_variance；
drift_diffusion_ratio；
classwise_drift_consistency；
split_drift_consistency；
random_target_gap；
corrupt_target_gap。
```

#### C-O2：Micro-Trajectory Invariance Observer

从同一 state 出发，做多个合法 train-stream micro-trajectory：

```text
B1->B2；
B2->B1；
SGD micro；
AdamW micro；
Momentum micro；
random batch order。
```

不使用 validation/test/future。只在当前 train stream 内评估。

记录：

```text
micro_source_sign_agreement；
micro_transfer_gain_mean；
micro_transfer_gain_std；
micro_optimizer_invariance；
source_direction_cosine_across_paths；
loss_drop_stability；
control_gap_stability。
```

#### C-O3：Low-NDS / Low-Curvature Observer

记录：

```text
first_order_gain；
second_order_penalty；
NDS；
within_block_NDS；
cross_block_NDS；
NDS_vs_random_gap；
NDS_vs_gradient_gap。
```

Observer 不直接选择最大 first-order gain，而选择：

$$
Q_{NDS} = I^{(1)} - \lambda I^{(2)}
$$

或：

$$
Q_{NDS} = \frac{I^{(1)}}{NDS+\epsilon}
$$

#### C-O4：Information-Volume / No-Fold Observer

记录：

```text
source_path_intrinsic_dim；
info_volume_logdet；
neighbor_preservation；
fold_proxy；
Jacobian_condition_proxy；
local_stretch_p95。
```

Observer 不接受：

```text
InfoVol collapse；
neighbor tearing；
fold proxy high；
local stretch p95 爆。
```

#### C-O5：Control-Nullspace Observer

构造 control span：

```text
AdamW step；
SGD step；
Momentum step；
RandomMatched；
StableRandom；
NoOp overhead。
```

候选 target 必须在 control-null residual 中仍有 signal：

$$
\Delta f_{res} = \Delta f - P_{control}\Delta f
$$

记录：

```text
control_projection_fraction；
control_null_residual_norm；
control_null_B2_transfer_gain；
control_null_NDS；
control_null_drift_diffusion_ratio。
```

### 8.3 Observer pass 标准

No-commit observer pass 不等于 success。它只是允许进入 C1/C2/C3。

必须满足：

```text
AUC_predict_official_early_chain >= 0.75
precision_at_top20 >= 0.50
control_equivalent_fraction <= 0.10
heldout_dataset_seed_precision_at_top20 >= 0.40
random/sign/corrupt gap positive
```

如果所有 observer families 都失败，则写：

```text
RetainedSourceObserverLocalNoGo_v22.08
```

并停止同族 observer 小修。

---

## 9. Line D：Metric-as-dynamics solver

### 9.1 总体原则

只有 C-O observer 过 gate，才构造 target 和 solver。v22.08 不允许 high ActuationR2 直接进入 horizon run。

Pipeline：

```text
C0 observer no-commit
-> C1 target contrast
-> C2 solver readback
-> C3 source formation
-> C4 terminal preservation
-> C5 KAN mapping
```

### 9.2 Function-space solver levels

#### S1：Readout exact solve

当前已有。继续作为 baseline。

#### S2：Hidden/readout joint block CG

新要求：hidden block 不再使用 residual heuristic。必须显式求解：

$$
\min_{u_h,u_r}
\|J_hu_h + J_ru_r - \Delta f_{target}\|_{G_f}^2 + \rho_h\|u_h\|^2 + \rho_r\|u_r\|^2
$$

记录：

```text
projection_residual_Gf；
ActuationR2；
B2_transfer_gain；
JVP_count；
VJP_count；
CG_iterations；
solver_rank；
condition_estimate；
solve_time_ms。
```

#### S3：Matrix-block source-state solver

对 matrix block 记录 row norm / angular velocity，并做 row-orthogonal component：

$$
u_{tan}=u-\frac{\langle u,W\rangle}{\|W\|^2+\epsilon}W
$$

记录：

```text
row_norm_drift；
angular_velocity；
radial_source_fraction；
tangential_source_fraction；
NDS_radial；
NDS_tangential。
```

#### S4：Optimizer-state integrated solver

不只改参数，还写入 optimizer source state：

```text
source_short_momentum；
source_long_momentum；
source_preconditioner_diag；
source_block_coordinate_state。
```

记录：

```text
source_state_norm；
source_state_half_life；
source_state_alignment_h800/h1600/h3200；
AdamW_state_conflict；
slow_state_retention_gain。
```

#### S5：KAN basis-channel solver

仅在 MLP C3 pass 后进入。

D-CHE：

```text
low-degree bank；
degree-readout block；
readout-only；
basis-estimate/readout-commit。
```

D-FOU：

```text
low-frequency bank；
band-readout block；
readout-only；
basis-estimate/readout-commit。
```

D-RAT / D-RBF：只在 robust official efficiency 过后进入 limited mapping。

### 9.3 Solver success gates

C2 solver gate：

```text
projection_residual_Gf <= 0.35
ActuationR2 >= 0.60
B2_transfer_gain >= 0.005
B3_safety_gain >= -0.005
solve_time_ms <= 1.25 * make_update_wall_ms
```

C3 source formation gate：

```text
source_h100 >= 0.005
source_h400 >= 0.005
source_h800 >= 0.005
source_h1600 >= 0.005
source_h3200 >= 0.005
row_h3200_positive_count >= 6/9
matched controls fail
LineC / tail / ECE / Brier debt not exploded
```

C4 terminal retention gate：

```text
source_h4800 >= 0.005
R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
stable-random control fails
AUCtime <= 1.05
```

---

## 10. Line E：KAN source-channel mapping

KAN mapping only enters if MLP C3 source formation passes.

### 10.1 MLP source decomposition

Record：

```text
hidden_source_energy；
readout_source_energy；
source_subspace_rank；
source_singular_values；
hidden/readout function displacement cosine；
source reconstruction error；
NDS hidden/readout；
InfoVol hidden/readout。
```

### 10.2 D-CHE mapping

Compare：

```text
D-CHE low-degree bank commit；
D-CHE degree-readout commit；
D-CHE readout-only commit；
D-CHE basis-estimate/readout-commit；
D-CHE high-degree control。
```

### 10.3 D-FOU mapping

Compare：

```text
D-FOU low-frequency bank commit；
D-FOU band-readout commit；
D-FOU readout-only commit；
D-FOU basis-estimate/readout-commit；
D-FOU high-frequency control。
```

### 10.4 KAN mapping success gate

```text
KAN source_h100/h400/h800 >= 0.005
KAN source_h1600/h3200 >= 0.005
KAN R4800_over_3200 >= 0.50
KAN_specific_delta_vs_MLP_same_metric >= 0.005
controls fail
basis high-frequency / high-degree leakage not dominant
```

If KAN fails while MLP passes：

```text
Route = KANSourceChannelMismatchConfirmed
Action = stop direct KAN-FU variants; redesign source-channel carrier.
```

---

## 11. Required metrics

### 11.1 Source and horizon metrics

```text
source_h100/h400/h800/h1600/h2400/h3200/h4000/h4800/h6400
R1600_over_800
R3200_over_1600
R4800_over_3200
row_positive_count per horizon
matched_control_best_source
control_equivalent_fraction
stable_random_gap
random_sign_corrupt_gap
```

### 11.2 Observer metrics

```text
AUC_predict_h800_positive
AUC_predict_h3200_positive
AUC_predict_official_early_chain
precision_at_top20
recall_at_top20
heldout_precision_at_top20
Spearman_score_vs_source_h800/h3200
coherent_drift_norm
diffusion_variance
drift_diffusion_ratio
micro_trajectory_agreement
control_null_residual_fraction
```

### 11.3 Metric / geometry metrics

```text
metric_energy_L2
metric_energy_Fisher
metric_energy_Sobolev
metric_energy_RKHS
NDS
within_block_NDS
cross_block_NDS
first_order_gain
second_order_penalty
info_volume_logdet
intrinsic_dim
neighbor_preservation
fold_proxy
local_stretch_p95
```

### 11.4 Solver metrics

```text
projection_residual_Gf
ActuationR2
ActuationCosine
B2_transfer_gain
B3_safety_gain
JVP_count
VJP_count
CG_iterations
solver_rank
condition_estimate
solve_time_ms
make_update_wall_ms
function_displacement_cosine
update_tensor_cosine
```

### 11.5 Optimizer-state metrics

```text
optimizer_cumulative_projection_on_source
AdamW_m_cos_source
AdamW_v_weighted_cos_source
SGD_momentum_cos_source
source_short_state_norm
source_long_state_norm
source_state_alignment
source_state_half_life
row_norm_drift
angular_velocity
radial_fraction
tangential_fraction
```

### 11.6 Debt metrics

```text
CEp99_debt_peak/final/recovery
LineC_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
NLL_delta
AUCtime_ratio
```

### 11.7 Efficiency metrics

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
metric_solver_ms
functional_commit_ms
LineC_audit_ms
horizon_readback_ms
forward_peak_memory
backward_peak_memory
basis_activation_bytes
functional_state_bytes
same-param MLP ratios
```

---

## 12. Required visualizations

```text
1. Observer AUC / precision dashboard。
2. Drift-diffusion scatter: drift_diffusion_ratio vs h3200 source。
3. Micro-trajectory invariance heatmap。
4. NDS vs source retention scatter。
5. First-order gain vs second-order penalty Pareto。
6. Metric energy waterfall: L2/Fisher/Sobolev/RKHS/LowNDS。
7. Solver projection residual vs ActuationR2 scatter。
8. Source horizon curves h100-h6400。
9. Terminal retention ratio histogram。
10. Optimizer cumulative projection on source curve。
11. MLP hidden/readout source decomposition bar。
12. KAN low-degree/low-frequency source bank heatmap。
13. D-CHE/D-FOU/D-RAT/D-RBF efficiency Pareto。
14. 4GPU utilization timeline。
```

---

## 13. Failure decision rules and Codex fallback actions

### Case A：Code gate fails

Action：

```text
修 code packet / import closure / C2 repair recompute / source-chain tests。
不跑 scientific route。
```

### Case B：D-RAT / D-RBF robust officialization fails

Action：

```text
D-RAT:
  输出 numerator / denominator / reciprocal / telemetry component waterfall；
  只继续最慢 component 的 kernel repair；
  不进入 full FU。

D-RBF:
  输出 active-center fraction / local-K / dense materialization / exp eval waterfall；
  若 local support 不降低 runtime，则限制为 low-budget smoke。
```

### Case C：All C-O observers fail

Action：

```text
停止同族 observer variants；
写 RetainedSourceObserverLocalNoGo_v22.08；
提出新的 source observability principle；
不允许继续 scale/cap/floor 小修。
```

### Case D：Observer pass but C3 fails

解释：observer 是 retrospective 或 target construction 不对。

Action：

```text
做 fresh selected rerun；
增加 micro-trajectory invariance；
如果 still fail，标记 observer not causal。
```

### Case E：C2 solver fails

Action：

```text
修 true block solver；
降低 rank / batch；
增加 CG / damping；
输出 projection residual heatmap；
不进入 C3。
```

### Case F：C3 passes but C4 fails

Action：

```text
进入 terminal preservation；
记录 optimizer cumulative source projection；
测试 source-state integration / slow-state / low-NDS preservation；
不再扩大 h800 source amplitude。
```

### Case G：MLP passes but KAN fails

Action：

```text
确认 KANSourceChannelMismatch；
比较 readout-only vs low-degree/low-frequency bank；
如果 high-degree/high-frequency leakage dominant，重设 KAN carrier；
不继续直接复制 MLP update。
```

---

## 14. Minimum success definitions

### Minimum Success A：Code and artifact integrity

```text
S0.15 pass = 1
self_contained_import_check = 1
C2 repair recompute test pass = 1
queue violation = 0
```

### Minimum Success B：Efficiency progress

```text
D-CHE / D-FOU reconfirm pass；
D-RAT or D-RBF robust officialization improves from v22.07;
component telemetry complete。
```

### Minimum Success C：Functional diagnostic progress

At least one of：

```text
1. A new observer family passes C0 heldout gate and selected fresh C3 is attempted；
2. C2 true block solver passes after recompute and produces non-proxy ActuationR2；
3. MLP C3 source formation passes h100-h3200；
4. terminal preservation reaches h4800 gate。
```

### Promotion candidate

```text
MLP or KAN:
  source_h100/h400/h800/h1600/h3200 >= 0.005
  row_h3200_positive_count >= 6/9
  source_h4800 >= 0.005
  R4800_over_3200 >= 0.50
  row_h4800_positive_count >= 7/9
  matched controls fail
  stable-random fails
  debt not exploded
```

### KAN-specific promotion candidate

```text
KAN all above;
KAN_specific_delta_vs_MLP_same_metric >= 0.005;
step/memory pass;
basis high-frequency/high-degree leakage controlled。
```

---

## 15. Final expected route outputs

v22.08 finalizer must output one of：

```text
S3-MLPSourceFormationOpened
S4-MLPTerminalRetentionOpened
S4-KANSourceChannelOpened
R-ObserverLocalNoGoBoundary
R-SolverProjectionBlocked
R-KANSourceChannelMismatchConfirmed
R-EfficiencyCarrierBlocked-DRAT-DRBF
R-CodeMetricMechanismInvalid
```

No vague route allowed. The final route must state which condition failed and what Codex should try next.

---

## 16. One-sentence summary

v22.08 的核心不是继续找更多 functional update 名字，而是：

$$
\boxed{
\text{先找到一个合法 train-only retained-source certificate，}
\text{再用真正的 metric solver 写入训练动力学，}
\text{最后才做 KAN source-channel mapping。}
}
$$

如果没有 source observer，metric solver 只会把错误 target 执行得更精确；如果没有 source-state integration，source 仍会被后续训练冲淡；如果没有 KAN source-channel mapping，KAN 仍只会出现 late rebound 而非 retained source。

