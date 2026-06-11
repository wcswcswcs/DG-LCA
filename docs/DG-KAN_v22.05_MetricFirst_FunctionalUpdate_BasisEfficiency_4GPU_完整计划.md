# DG-KAN v22.05：Metric-First Functional Update + Terminal Retention + Basis Efficiency 4GPU 完整计划

> 版本：v22.05  
> 生成时间：2026-06-06 Asia/Singapore  
> 适用背景：基于 v22.04 结果、v22.03/v22.02 历史对照、当前 optimizer / geometry / manifold 研究启发  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no validation / test / future / query 生成 functional direction；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能做 audit / gate / debt readback；不重启 action bank / controller / reset route。

---

# 0. 项目总目标与 v22.04 后的真实状态

DG-KAN 项目的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN / efficient KAN carrier 上，}
\text{通过 functional update 改善训练动力学，}
\text{最终得到比 ordinary backprop / AdamW 更好的模型。}
}
$$

这里的“更好”不是单个 source row 变正，也不是某个 horizon 局部好，而是同时满足：

```text
1. 代码和指标实现正确，不能被错误 route / 错误 metric / 缺文件误导。
2. KAN carrier 的 forward / backward / update / memory 接近 same-param MLP。
3. Functional update 的 source 能从 early horizon 连续保留到 terminal horizon。
4. Functional update 的收益不能被 AdamW / SGD / NoOp / random / stable-random / matched-overhead controls 解释。
5. Tail / LineC / calibration / AUCtime debt 不能在 terminal phase 爆发。
6. 如果只在 MLP 上成功，写成 generic training-dynamics insight；只有 KAN 在同机制下更强，才写 KAN-specific。
```

v22.04 的真实状态是：

```text
代码：
  required source files 29/29；compileall pass；required import closure pass。
  但机制合同表存在字段语义错位；D1 系列多个机制在源码层实际共享同一 momentum-like update，只是 source label 不同。

效率：
  D-CHE / D-FOU full-loop official closure 继续站住。
  D-RAT / D-RBF 从 blocked 表推进到 micro-near-E1，但 official fused missing。

Functional：
  D1 source-preserving family 有接近 h4800 gate 的 MLP rows。
  最好现象：某些 row h4800 / h3200 超过 0.50，但 early h100 为负；另一些 row early positive，但 h4800 / h3200 低于 0.50。
  这说明 source 形成与 terminal preservation 仍未同时闭合。
  KAN source mapping 仍为 KANSourceChannelMismatch。
```

所以 v22.05 不再继续“加更多 Fxx 或 D1x 名字”。v22.05 的核心转向是：

$$
\boxed{
\text{Metric-first functional update：先定义函数空间几何，}
\text{再生成和保护 update。}
}
$$

---

# 1. v22.04 独立审计得到的关键问题

## 1.1 代码审计问题

v22.04 的代码包比前几轮强，但仍有三个需要在 v22.05 硬修的问题。

### 问题 A：import closure 的 self-contained 字段没有作为 hard gate

`v22_04_import_closure.csv` 中 `pass=1`，但 `self_contained=0`。当前 route 只看 import return code，没有把“最终 zip 解压目录是否自包含”作为硬门。v22.05 必须要求：

```text
unzip final code packet 到临时目录；
在临时目录执行 compileall / import closure；
self_contained_import_check 必须等于 1；
否则 CodeRoute 不能写 pass。
```

### 问题 B：mechanism_contracts 字段语义错位

`v22_04_mechanism_contracts.csv` 中 `uses_optimizer_primary` 列实际填入了类似 `early100_h800_slow_ema_terminal_info_volume_guard` 的 source pattern 字符串，而不是 bool。源码里 specs tuple 的第二项是 source pattern，却被写进 `uses_optimizer_primary`。

v22.05 必须把 contract schema 改为：

```text
mechanism
mechanism_family
source_pattern
uses_optimizer_primary: 0/1
uses_slow_state: 0/1
uses_matrix_block: 0/1
uses_function_space_metric: 0/1
uses_sobolev_metric: 0/1
uses_rkhs_metric: 0/1
uses_fisher_metric: 0/1
uses_basis_channel_metric: 0/1
implementation_is_prototype: 0/1
semantic_contract_declared: 0/1
semantic_noncollapse_group
```

并且新增硬门：

```text
bool columns 若不是 0/1 => S0 fail。
source_pattern 不能写入 bool column。
mechanism family 中若多个机制最终使用完全相同 update tensor，必须标 semantic_alias_group。
```

### 问题 C：D1 / D1a / D1b 等 terminal-preservation 机制实际严重 alias

源码中 v22.04 多个 terminal mechanisms 最终返回：

```python
UpdateTensor(_momentum_like_precondition(g), "step", "subtract", "slow_state", source, mechanism, ...)
```

它们的差异主要是 `source` 字符串不同，而不是 update 计算不同。这意味着 v22.04 的“多机制”有相当一部分是名义多样，实际同义。

v22.05 必须做 semantic non-collapse audit：

```text
1. 对每个 mechanism 在同一 model/batch/seed 上生成 update tensor。
2. 计算 pairwise cosine、L2 distance、support overlap、role-wise energy。
3. 若 cosine > 0.999 且 norm ratio 在 [0.99,1.01]，标记 semantic alias。
4. 同一 alias group 只允许一个代表进入 expensive h4800/h6400。
```

---

# 2. v22.04 后的问题本质：从工程搜索转向函数空间几何

v22.04 前我们主要在做：

```text
找 train-stream source；
提交 update；
继续训练；
读 h100/h800/h3200/h4800；
看 source 是否留存。
```

这套流程能观测现象，但不够解释和控制现象。现在最核心的问题是：

$$
\boxed{
\text{一个 update 在什么函数空间 metric 下是低能量、低曲率、平滑、可留存的？}
}
$$

因此 v22.05 的算法中心从 `source-preserving trick` 改成 `metric-defined functional update`。

对模型函数 $f_\theta(x)$，参数更新 $u$ 诱导输出变化：

$$
\Delta f(x) \approx J_\theta(x)u.
$$

如果定义函数空间 metric $G_f$，则函数变化的能量是：

$$
\|\Delta f\|_{G_f}^2 = \langle \Delta f, G_f \Delta f\rangle.
$$

对应的参数空间 pullback metric 是：

$$
M_\theta = J_\theta^\top G_f J_\theta.
$$

functional update 不应该只是 heuristic gradient wrapper，而应该求：

$$
u^\star = \arg\min_u
\|J_\theta u - \Delta f_{target}\|_{G_f}^2
+ \rho \|u\|_P^2.
$$

或者在没有显式 target 时，用 metric-preconditioned loss cotangent：

$$
u^\star = -\eta (J_\theta^\top G_f J_\theta + \rho P)^{-1}g.
$$

---

# 3. 需要重新考虑的旧路线

## 3.1 可以重开，但必须换语义的方向

### 3.1.1 v15.02.1 function metric

过去 function-metric / proximal diagnostic 失败，不能直接说明 function metric 无效。它当时的问题是：

```text
1. 作为一个 update token 使用，而不是完整 metric-first solver。
2. 当时 h4800 terminal retention 还不是核心 gate。
3. D-CHE / D-FOU efficiency 还没站住。
4. LineC / debt / source-chain 指标还不够严格。
```

v22.05 允许重开，但必须改成：

```text
metric-defined function-space projection + terminal source preservation。
```

### 3.1.2 v13.2 / v13.3 basis natural update

过去 basis-natural / low-rank tangent natural update 没打开，但这些思路和 v22.05 的 $J^\top G_f J$ 很接近。可以重开：

```text
basis natural metric；
low-rank tangent metric；
block group natural metric；
D-CHE degree metric；
D-FOU frequency metric。
```

但不能重开旧错误：

```text
直接把 basis-natural local positive 写成 promotion。
```

### 3.1.3 v13.4 operator-level basis-channel functional

operator-level basis-channel target 本来就更接近真正 functional update。v22.05 应该把它和 Sobolev / RKHS / Fisher metric 合并：

```text
先在函数空间定义 target；
用 metric 判断 target 是否平滑 / 可留存；
再投影回 basis / readout 参数；
最后验证 h4800 retention。
```

### 3.1.4 PopRisk / SNR 与 split-consensus

PopRisk/SNR 和 split-consensus 不能继续作为 one-shot mask 或 selector。它们应该成为 metric 中的权重或 source-channel estimator：

```text
PopRisk/SNR -> diag metric weights；
split-consensus -> RKHS / graph edge weights；
source-state -> terminal preservation direction。
```

## 3.2 不能重启的方向

以下方向不进入主预算：

```text
action bank / controller；
reset route；
cover objective 小修；
M31/M32 同族修补；
G/N/Q token 扩展；
单纯再调 terminal floor / hold / lookahead / scale。
```

---

# 4. 当前研究进展给 v22.05 的具体启发

## 4.1 Muon curvature / NDS

Muon curvature 视角提示：一阶收益相近时，长期稳定性可能由二阶曲率惩罚决定。v22.05 不直接“换 Muon”，而是记录每个 source candidate 的：

$$
I^{(1)} = -\langle g,u\rangle,
$$

$$
I^{(2)} = \frac{1}{2}u^\top H u,
$$

$$
NDS(u)=\frac{u^\top H u}{\|u\|^2+\epsilon}.
$$

如果 h4800 erosion 与高 NDS 相关，下一步应该做 low-NDS metric projection，而不是扩大 source amplitude。

## 4.2 Nora row-orthogonal matrix update

Nora 的启发是：矩阵参数的 row norm 与 angular velocity 需要稳定。v22.05 把它转成：

```text
MLP hidden/readout matrix source preservation；
D-CHE degree-readout matrix source preservation；
D-FOU band-readout matrix source preservation。
```

但 v22.04 已经说明简单 row-orthogonal source preservation 并没有直接过 h4800。因此 v22.05 不再只做 Nora-like update，而是把 row-orthogonal 当成 metric / preservation constraint 的一个 component。

## 4.3 Schedule-free / AdEMAMix

Schedule-free 的 fast / slow iterate 与 averaging 启发我们：source 可能要写入 slow state，而不是 fast iterate。AdEMAMix 启发我们：单一 EMA 不足以同时保留近期和旧梯度。因此 v22.05 使用：

```text
short source memory: 负责 h800/h1600；
long source memory: 负责 h3200/h4800；
metric-projected late update: 避免 long source 被 fast optimizer 擦掉。
```

## 4.4 SOAP / matrix basis

SOAP 的启发不是“换成 SOAP optimizer”，而是：在 slowly changing coordinate basis 中运行 adaptive moments。对我们来说对应：

```text
D-CHE degree/readout eigenbasis；
D-FOU band/readout eigenbasis；
MLP hidden/readout matrix SVD basis；
D-RAT numerator-denominator tangent basis；
D-RBF local-center kernel basis。
```

## 4.5 Inference manifold / information volume

压缩到低维并不等于健康。v22.05 需要记录 source path 的：

```text
intrinsic dimension；
information volume；
subspace condition；
fold proxy；
neighborhood preservation。
```

如果 h4800 erosion 伴随 information volume collapse，则 update 不是平滑 signal-channel deformation，而是被压成退化方向。

---

# 5. v22.05 总体实验结构

v22.05 分成四个强制部分：

```text
Part A: S0.12 代码 / 指标 / 机制语义硬门。
Part B: Basis efficiency officialization and repair。
Part C: Metric-first functional update。
Part D: 4GPU dynamic queue and reporting contract。
```

v22.05 的总体目标不是直接 S5，而是至少完成以下三个推进：

```text
1. 证明 v22.04 代码/机制语义不再误导实验。
2. 把 D-CHE / D-FOU 作为 stable efficient carriers 固化，并推进 D-RAT / D-RBF official fused wiring。
3. 裁决 metric-first FU 是否能把 MLP h4800 retention ratio 推过 0.50，并开始解决 KAN source-channel mismatch。
```

---

# 6. Part A：S0.12 代码 / 指标 / 机制语义硬门

## 6.1 目标

确保 v22.05 不再因为 code packet、route、contract、mechanism alias 或 profiler 口径错误继续原地打转。

## 6.2 必须打包的 code packet

Codex 必须输出：

```text
v22_05_code_review_packet.zip
```

包内必须有：

```text
00_README.md
02_SOURCE_TREE/dgkan/**/*.py
02_SOURCE_TREE/experiments/**/*.py
02_SOURCE_TREE/tests/**/*.py
03_IMPORT_CLOSURE/
04_LINEC_CORRECTNESS/
05_SOURCE_CHAIN_RETENTION_DEBT/
06_MECHANISM_SEMANTIC_CONTRACTS/
07_FUNCTION_SPACE_METRICS/
08_BASIS_KERNELS_AND_PROFILERS/
09_GPU_QUEUE_CONTRACT/
10_RESULTS_POINTERS/
packet_manifest.csv
packet_sha256_manifest.csv
```

尤其必须包含：

```text
dgkan/fu/core.py
dgkan/fu/source_chain.py
dgkan/fu/terminal_retention.py
dgkan/fu/terminal_erosion.py
dgkan/fu/function_space_metrics.py
dgkan/fu/sobolev_metric.py
dgkan/fu/rkhs_metric.py
dgkan/fu/fisher_metric.py
dgkan/fu/metric_projection.py
dgkan/fu/mechanisms.py
dgkan/metrics/linec.py
dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py
dgkan/profiling/efficiency_v20.py
dgkan/profiling/efficiency_v21.py
dgkan/profiling/efficiency_v22.py
dgkan/profiling/efficiency_v22_03.py
dgkan/profiling/efficiency_v22_04.py
dgkan/profiling/efficiency_v22_05.py
experiments/run_v22_05_*.py
```

## 6.3 必须执行的代码检查

在最终 zip 解压目录中执行，不允许在原始 repo 里替代：

```bash
python -m compileall -q dgkan experiments tests
python experiments/run_v22_05_s012_truth_gate.py --mode all --source-root ./02_SOURCE_TREE --self-contained-import-check 1
```

必须输出：

```text
v22_05_required_source_files.csv
v22_05_compileall.csv
v22_05_import_closure.csv
v22_05_clean_unzip_self_test.csv
v22_05_code_truth_gate.csv
v22_05_code_route_decision.json
```

## 6.4 通过标准

```text
required_source_files = 100%
compileall pass = 1
import_closure pass = 1
self_contained_import_check = 1
linec_fast/channel golden pass = 1
source_chain tests pass = 1
terminal_retention tests pass = 1
debt accounting tests pass = 1
mechanism contract schema pass = 1
semantic non-collapse audit pass = 1
profiler phase tests pass = 1
kernel gradcheck pass = 1
```

如果任一失败，Codex 必须先按以下方向修：

```text
缺文件：补 source tree，而不是把 required list 改短。
import 失败：补 shim 或真实依赖，但必须在 zip 解压目录验证。
mechanism contract 错：修 schema，不允许将 string 塞入 bool column。
semantic alias 过多：合并 alias group，只保留 representative，不能继续冒充多机制。
LineC 或 source-chain fail：先修 metric，不允许继续跑 scientific rows。
```

---

# 7. Part B：基函数效率路线

## 7.1 总目标

v22.05 的效率目标是：

```text
D-CHE / D-FOU：确认 stable efficient carrier status，并绑定到 metric-first FU runner。
D-RAT / D-RBF：从 micro-near-E1 推进到 official fused runner 或给出明确 blocker。
LQ / D-WAV：低预算 smoke，不抢主 GPU。
```

## 7.2 D-CHE / D-FOU full-loop reconfirm

### 假设

$$
H_B1:
\text{D-CHE / D-FOU 已经不是 functional failure 的效率 blocker。}
$$

### 实验

对以下 variants 执行 full-loop timing：

```text
D-CHE:
  CHE21-R2-low-degree-k3-official
  CHE21-R4-k5-gradbuf-triton-ablation

D-FOU:
  FOU21-R3-tablelookup-bandreadout-official
```

每个 variant 跑：

```text
batch = 8, 32, 128, 256, 512
same-param MLP reference
metric-first FU runner path
ordinary training path
LineC/horizon readback separated path
```

### 记录指标

```text
forward_only_ms
basis_eval_ms
readout_contraction_ms
backward_grad_ms
optimizer_update_ms_AdamW
optimizer_update_ms_SGD
optimizer_update_ms_manualFU
functional_metric_build_ms
functional_projection_solve_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
training_step_ms
full_loop_step_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
functional_overhead_ratio
same_kernel_functional_runner_proof
fallback_kernel_used
official_fused_kernel_complete
no_materialize_complete
gradcheck_pass
```

### 成功标准

```text
D-CHE:
  >= 2 variants × >= 3 batch sizes S1 pass。

D-FOU:
  >= 1 variant × >= 3 batch sizes S1 pass。

所有 pass row:
  forward_ratio <= 1.25
  step_ratio <= 1.25
  memory_ratio <= 1.05
  same_kernel_functional_runner_proof = 1
  fallback_kernel_used = 0
```

若失败，Codex 先尝试：

```text
D-CHE batch512 forward blocked：检查 tile size / degree bank cache / gradbuf fallback。
D-FOU batch512 forward blocked：优先修 FOU-R3 tablelookup-bandreadout，不扩大 FOU-R4。
functional overhead > 1.10：拆 metric build / projection solve / commit 三段，先降低 metric build。
```

## 7.3 D-RAT active repair

### 假设

$$
H_B2:
\text{D-RAT 当前失败来自 rational forward / reciprocal / telemetry path，而不是 basis family 本身必然低效。}
$$

### 实验

D-RAT 只做 kernel active repair，不进入 full FU matrix。variants：

```text
RAT-R1-Horner-numden-fused
RAT-R2-branchless-denominator-safe
RAT-R3-reciprocal-approx-trainpath
RAT-R4-telemetry-free-trainpath
RAT-R5-numden-fused-backward
RAT-R6-low-degree-rational-readout-only
RAT-R7-production-fused-candidate
```

### 记录指标

```text
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
rational_forward_ms
rational_backward_ms
derivative_telemetry_ms
safety_guard_ms
forward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
den_min
den_p01
den_condition
r_prime_p99
r_double_prime_p99
gradcheck_pass
official_fused_kernel_complete
telemetry_free_trainpath
```

### 成功标准

Near-E1：

```text
forward_ratio <= 3.0
step_ratio <= 2.0
memory_ratio <= 1.2
gradcheck_pass = 1
```

E1：

```text
forward_ratio <= 1.75
step_ratio <= 1.50
memory_ratio <= 1.10
official_fused_kernel_complete = 1
```

若未达 near-E1，Codex 先尝试：

```text
若 reciprocal_ms dominant：切 approximate reciprocal + Newton refinement。
若 denominator_eval_ms dominant：Horner fused + branchless safety。
若 telemetry_ms dominant：train path 去 telemetry，audit path 单独读。
若 memory ok 但 forward high：不要继续 functional smoke，继续 kernel。
```

## 7.4 D-RBF active repair

### 假设

$$
H_B3:
\text{D-RBF 当前失败来自 dense materialization / active-center density，而不是 local basis 思想无效。}
$$

### 实验 variants

```text
RBF-R1-compact-local-k4-no-dense
RBF-R2-active-center-mask-fused
RBF-R3-local-gather-contraction
RBF-R4-exp-approx-trainpath
RBF-R5-width-conditioned-local-k
RBF-R6-production-sparse-local-candidate
```

### 记录指标

```text
active_center_fraction
mean_local_k
max_local_k
dense_basis_materialized
basis_activation_bytes
exp_eval_ms
local_gather_ms
local_contraction_ms
local_backward_ms
forward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
official_fused_kernel_complete
```

### 成功标准

Near-E1：

```text
forward_ratio <= 3.0
step_ratio <= 2.0
memory_ratio <= 1.2
active_center_fraction <= 0.50
mean_local_k <= 4.0
dense_basis_materialized = 0
gradcheck_pass = 1
```

如果 active center fraction 仍大于 0.75，Codex 先尝试：

```text
width annealing；
compact support hard mask；
center occupancy regularization audit-only；
no-dense local gather rewrite；
暂不进入 full FU matrix。
```

---

# 8. Part C：Metric-First Functional Update

## 8.1 总目标

v22.05 functional 线的目标是：

$$
\boxed{
\text{把 MLP 的 h4800 retention ratio 从约 0.43-0.48 提高到 >= 0.50，}
\text{并找到 KAN source-channel mapping 的可行或不可行证据。}
}
$$

Functional update 不再以 mechanism name 为中心，而以：

```text
Metric × Target × Carrier × Commit × Preservation
```

为中心。

## 8.2 Metric ladder

v22.05 必须实现以下函数空间 metrics。

### G0：Output L2 metric

$$
\|\Delta f\|_{L^2}^2 = \frac{1}{n}\sum_i \|\Delta f_i\|^2.
$$

用途：baseline。

### G1：Diagonal Fisher metric

对分类概率 $p_i$：

$$
\|\Delta f\|_{FisherDiag}^2
= \sum_{i,c} p_{i,c}(1-p_{i,c})\Delta f_{i,c}^2.
$$

用途：避免 logit scale 误导。

### G2：PopRisk / SNR diagonal metric

$$
w_k = \frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}.
$$

用途：把 coherent train-stream signal 权重大。

### G3：Sobolev-H1 metric

$$
\|\Delta f\|_{H^1}^2
= \mathbb{E}\|\Delta f(x)\|^2
+\lambda\mathbb{E}\|\nabla_x \Delta f(x)\|_F^2.
$$

实际近似：

```text
input augmentation neighborhood finite difference；
hidden representation neighborhood finite difference；
KAN basis-channel finite difference。
```

### G4：RKHS / graph Laplacian metric

$$
\|\Delta f\|_G^2
= \sum_{i,j}K_{ij}\|\Delta f_i-\Delta f_j\|^2.
$$

近似：

```text
KNN graph in hidden representation；
Nyström low-rank K；
RBF kernel on normalized hidden states。
```

### G5：Fisher + RKHS hybrid

$$
G_f = G_{FisherDiag}+\lambda G_{Graph}.
$$

### G6：Low-NDS metric

记录但不一定直接求逆：

$$
NDS(u)=\frac{u^\top H u}{\|u\|^2+\epsilon}.
$$

若 candidate high-NDS，则 projection 或 damping。

### G7：KAN basis metric

D-CHE degree metric：

$$
\|\Delta f\|_{CHE}^2=\sum_k (1+\alpha k^2+\beta k^4)\|\Delta f_k\|^2.
$$

D-FOU frequency metric：

$$
\|\Delta f\|_{FOU}^2=\sum_\omega (1+\alpha \omega^2)\|\Delta f_\omega\|^2.
$$

D-RAT tangent metric：

$$
M_{RAT}=J_{num,den}^\top G_f J_{num,den}+\rho I.
$$

D-RBF locality metric：

$$
\sum_{i,j}K_{ij}^{local}\|\Delta f_i-\Delta f_j\|^2.
$$

---

# 9. Metric-first FU experiments

## 9.1 H-C1：Metric projection improves terminal retention on MLP

### Hypothesis

$$
H_{C1}:\quad
\text{h4800 erosion is caused by high-energy / high-curvature / non-smooth source directions,}
\text{and metric projection improves }R_{4800/3200}.
$$

### Experiment

Use top MLP candidates from v22.02-v22.04:

```text
F118 / F121 / F122 / F154 / D1a-SPP-lambda025 / D1b-roworth-hidden-readout
```

For each candidate, run metric projection variants:

```text
G0-L2
G1-DiagFisher
G2-PopRiskDiag
G3-SobolevH1-hidden
G4-RKHS-KNN
G5-Fisher-RKHS
G6-LowNDS
G8-MetricEnsemble top2 average
```

### Metrics to record

```text
source_h100/h400/h800/h1600/h2400/h3200/h4000/h4800/h6400
R1600_over_800
R3200_over_1600
R4800_over_3200
row_h4800_positive_count
metric_energy_L2
metric_energy_Fisher
metric_energy_Sobolev
metric_energy_RKHS
NDS
first_order_gain
second_order_penalty
source_hidden_fraction
source_readout_fraction
info_volume
fold_proxy
neighbor_preservation
LineC_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
control_equivalent_fraction
```

### Success criteria

Exploration S2:

```text
R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
source_h4800 >= 0.005
controls fail
```

Productive S3:

```text
R4800_over_3200 >= 0.60
row_h4800_positive_count >= 7/9
LineC/ECE/Brier debt recovery >= 0.60
AUCtime_ratio <= 1.05
independent rerun pass
```

If not satisfied, Codex must compute failure taxonomy:

```text
HighNDSNoRetention
SobolevHighEnergyNoRetention
RKHSNeighborhoodTear
FisherOverconfidentDrift
DebtErosion
OptimizerErosion
ControlEquivalent
MetricNoEffect
```

## 9.2 H-C2：Metric-preserving late update prevents source erosion

### Hypothesis

$$
H_{C2}:\quad
\text{After h3200, ordinary optimizer updates contain a component that destroys source under }G_f.
$$

### Experiment

At h3200, capture source direction $s_{3200}$ in function space. For each subsequent optimizer update $u_t$, compute:

$$
\Delta f_t=J_\theta u_t.
$$

If:

$$
\langle \Delta f_t, s_{3200}\rangle_{G_f}<0,
$$

project away the destructive component:

$$
\Delta f_t' = \Delta f_t -
\lambda
\frac{\langle \Delta f_t, s_{3200}\rangle_{G_f}}
{\|s_{3200}\|_{G_f}^2+\epsilon}s_{3200}.
$$

Compare:

```text
No projection
L2 projection
Fisher projection
Sobolev projection
RKHS projection
Low-NDS projection
```

### Metrics

```text
optimizer_cumulative_projection_on_source_Gf
negative_projection_fraction
source_preservation_projection_norm
source_h3200/h4000/h4800
R4800_over_3200
task_loss_delta
LineC/ECE/Brier debt
AUCtime_ratio
```

### Success

```text
R4800_over_3200 increases by >= 0.05 vs no-projection baseline
and R4800_over_3200 >= 0.50
and AUCtime_ratio <= 1.05
and debt not worse than baseline
```

If failure:

```text
If projection improves R but hurts AUCtime => projection too expensive or overconstraining; reduce lambda.
If projection no effect and negative_projection_fraction low => erosion not optimizer washout; inspect debt / target metric.
If projection worsens source => source direction not stable; recompute source state at h4000.
```

## 9.3 H-C3：Metric target beats high ActuationR2 target

### Hypothesis

$$
H_{C3}:\quad
\text{High ActuationR2 is insufficient; target must be low-energy under a geometry metric to retain.}
$$

### Experiment

Compare targets:

```text
T0 loss-cotangent target
T1 split-consensus target
T2 PopRisk-SNR target
T3 Sobolev-low-energy target
T4 RKHS-smooth target
T5 Fisher-RKHS hybrid target
T6 Diffeomorphic no-fold target
T7 Random matched ActuationR2 target
T8 Sign-flipped target
```

For each target, record both local actuation and long-horizon retention.

### Metrics

```text
ActuationR2
B1_gain
B2_transfer_gain
B3_safety_gain
metric_energy_L2/Fisher/Sobolev/RKHS
neighbor_preservation
fold_proxy
source_h100/h800/h3200/h4800
R4800_over_3200
controls
```

### Success

```text
Target has ActuationR2 >= 0.80
and source_h4800 >= 0.005
and R4800_over_3200 >= 0.50
and random/signflip controls fail
```

If high ActuationR2 but no retention, mark:

```text
TargetObservableNoRetention
```

and do not continue this target family.

## 9.4 H-C4：KAN source-channel mapping under metric

### Hypothesis

$$
H_{C4}:\quad
\text{KAN fails because MLP source is written into wrong KAN channel; metric-based low-degree / low-frequency mapping can recover early chain.}
$$

### Experiment

First decompose MLP source:

```text
hidden matrix energy
readout matrix energy
rank / singular values
source subspace basis
source metric energy
```

Then map to KAN:

```text
D-CHE low-degree bank commit
D-CHE readout-only commit
D-CHE basis-estimate/readout-commit
D-FOU low-frequency bank commit
D-FOU band-readout commit
D-FOU basis-estimate/readout-commit
D-RAT numerator/denominator tangent smoke if near-E1 officialized
D-RBF local RKHS source smoke if near-E1 officialized
```

### Metrics

```text
KAN source_h100/h400/h800/h1600/h3200/h4800
R4800_over_3200
low_degree_energy_fraction
low_frequency_energy_fraction
high_degree_leakage
high_frequency_leakage
basis_vs_readout_source_fraction
KAN_specific_delta_vs_MLP_same_metric
controls
```

### Success

Exploration:

```text
KAN h100/h400/h800 >= 0.005
KAN h3200 >= 0.005
controls fail
```

Productive:

```text
KAN R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
KAN_specific_delta_vs_MLP_same_metric >= 0.005
```

If failure:

```text
If readout-only works but basis commit fails => basis channel writeback is blocker.
If low-degree/low-frequency works but full basis fails => high-order reservoir leakage.
If none works => keep KAN functional route closed; continue MLP metric-first FU first.
```

---

# 10. Visualizations required

v22.05 must generate:

```text
1. code_truth_dashboard.svg
2. mechanism_semantic_alias_heatmap.svg
3. efficiency_forward_backward_step_memory_dashboard.svg
4. D-RAT_component_waterfall.svg
5. D-RBF_component_waterfall.svg
6. terminal_source_trajectory_h100_to_h6400.svg
7. R4800_over_3200_by_metric.svg
8. metric_energy_vs_retention_scatter.svg
9. NDS_vs_terminal_erosion_scatter.svg
10. Sobolev_energy_vs_h4800_scatter.svg
11. RKHS_neighbor_preservation_vs_h4800.svg
12. optimizer_projection_on_source_trace.svg
13. debt_transition_h3200_to_h4800.svg
14. MLP_source_decomposition_hidden_readout.svg
15. KAN_source_channel_mapping_heatmap.svg
16. controls_attribution_heatmap.svg
17. GPU_utilization_timeline.svg
```

---

# 11. 4GPU dynamic queue

v22.05 必须充分使用四张 GPU。

## GPU assignment

```text
GPU0:
  S0.12 truth gate；
  MLP metric-first FU H-C1/H-C2；
  source erosion autopsy。

GPU1:
  D-CHE / D-FOU full-loop reconfirm；
  D-CHE / D-FOU metric source-channel mapping。

GPU2:
  D-RAT active repair；
  D-RBF active repair；
  D-RAT/RBF near-E1 limited smoke if gate opens。

GPU3:
  function-space target reset H-C3；
  controls / independent reruns / figures / packet generation。
```

## Required queue artifacts

```text
v22_05_runnable_queue.csv
v22_05_gpu_assignment_manifest.csv
v22_05_gpu_utilization_timeline.csv
v22_05_idle_violation.csv
v22_05_deferred_items.csv
v22_05_queue_drain_report.csv
```

`gpu_utilization_timeline.csv` 必须不是单行 snapshot。必须包含：

```text
timestamp
gpu_id
job_id
queue_nonempty
busy
memory_used_mb
utilization_percent
```

如果 runnable queue 非空且任一 GPU idle 超过 10 分钟：

```text
execution_contract_violation = 1
final route 不能写 completed-no-go
```

---

# 12. Final route rules

## S0 code route

```text
S0 pass only if:
  source packet self-contained；
  schema valid；
  semantic non-collapse passed；
  metric tests passed。
```

## Efficiency route

```text
E1-D-CHE-DFOU-OfficialEfficient:
  D-CHE and D-FOU full-loop pass。

E2-DRAT-DRBF-NearE1:
  D-RAT or D-RBF reaches near-E1 but official fused missing。

E3-DRAT-DRBF-OfficialRepair:
  D-RAT or D-RBF official fused complete and limited runner pass。
```

## Functional route

```text
F0-MetricNoEffect:
  no metric improves R4800_over_3200 by >=0.03。

F1-MetricRetentionImproved:
  some metric improves R4800_over_3200 by >=0.03 but <0.50。

F2-MLPProductiveTerminalSource:
  MLP R4800_over_3200 >= 0.50 and controls fail。

F3-KANSourceChannelOpened:
  KAN h100/h400/h800/h3200 source chain opens。

F4-KANProductiveTerminalSource:
  KAN R4800_over_3200 >= 0.50 and KAN-specific delta positive。
```

No promotion unless S3/S5 gates pass.

---

# 13. What Codex must do when conditions fail

## If code packet fails

```text
Do not run science rows.
Fix packet closure and rerun S0.12.
```

## If mechanisms collapse into aliases

```text
Keep one representative per alias group.
Do not run all aliases to h4800.
Implement true metric variants before continuing.
```

## If D-RAT/D-RBF near-E1 fails

```text
Do not run full FU on them.
Write component waterfall and continue micro-kernel repair.
```

## If no metric improves MLP retention

```text
Run terminal erosion autopsy:
  optimizer projection;
  NDS;
  Sobolev/RKHS energy;
  debt transition;
  hidden/readout source decay.
Then propose one new metric, not one new source trick.
```

## If MLP metric works but KAN fails

```text
Do not claim KAN-specific.
Run source-channel mapping:
  low-degree;
  low-frequency;
  readout-only;
  basis-estimate/readout-commit.
```

## If KAN low-degree/low-frequency works

```text
Extend to 3x3 and h6400.
Add D-CHE/D-FOU same-metric MLP controls.
```

---

# 14. v22.05 expected minimum deliverable

v22.05 must not end with only `route no-go`.

It must deliver at least one of:

```text
1. A true metric that increases MLP R4800_over_3200 by >= 0.03.
2. A productive MLP h4800 candidate with R4800_over_3200 >= 0.50.
3. A KAN source-channel early chain candidate on D-CHE or D-FOU.
4. D-RAT or D-RBF near-E1 to official fused transition.
```

If none occurs, final report must explicitly answer:

```text
Which metric families were tried?
Which metric energy correlated with terminal erosion?
Is h4800 erosion caused by optimizer projection, debt, high NDS, Sobolev/RKHS tearing, or source-channel decay?
Why did KAN source-channel mapping fail?
What exact kernel blocker remains for D-RAT/D-RBF?
```

---

# 15. 总结

v22.05 的核心不是再扩大 terminal-preservation variants，而是建立真正的 functional geometry：

$$
\boxed{
\text{先定义函数空间 metric，}
\text{再做最小能量/低曲率/平滑/可留存的 functional update。}
}
$$

如果 v22.05 还不能打开 h4800 gate，也必须至少把失败从“source erosion”进一步定位为：

```text
high NDS;
Sobolev/RKHS tearing;
optimizer destructive projection;
debt erosion;
source information volume collapse;
KAN source-channel mismatch;
control-equivalent terminal drift。
```

只有这样，下一步才不是继续原地打转。
