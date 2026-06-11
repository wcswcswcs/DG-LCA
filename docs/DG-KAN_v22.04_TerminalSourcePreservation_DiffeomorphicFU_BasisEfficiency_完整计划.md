# DG-KAN v22.04：Terminal Source Preservation + Diffeomorphic Source-Channel FU + D-CHE/D-FOU Full-Loop Closure + D-RAT/D-RBF Officialization 完整实验计划

> 版本：v22.04  
> 目标：停止扩大 Fxx 变体，集中解决 v22.03 暴露出的两个硬 blocker：  
> **Functional update：source 已能在 MLP 上形成 h100→h3200/h4800 正向链，但 h4800 retention ratio 不够；KAN 仍没有 source-channel writer。**  
> **Basis efficiency：D-CHE/D-FOU 已接近 full-loop official carrier；D-RAT/D-RBF 已有 micro-near-E1 线索，但 official fused production path 未闭合。**  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 约束：strict FC-PureKAN；不使用 teacher、distillation、loss modification、sampler/class weight、dataset-name branch、seed-specific scale、validation/test/future/query 生成 direction；LineC / CEp99 / NLL / ECE / Brier / AUC 只做 audit / gate / debt readback，不能生成 update direction。

---

## 0. 项目总目标与 v22.03 后的真实状态

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{获得比普通 backprop/AdamW 更好的模型。}
}
$$

这里“更好”不能只看某个 source row，也不能只看某个 horizon 的短期正值。最终必须同时满足：

```text
1. base carrier 的 forward / backward / update / memory 接近 same-param MLP；
2. functional update 的收益不能被 AdamW、SGD、random matched、NoOp overhead、stable-random trajectory controls 解释；
3. source 必须从早期形成，并在 h3200/h4800 后仍保留；
4. tail / LineC / calibration / AUC debt 不能在 terminal phase 爆发；
5. 如果 MLP 成功而 KAN 不成功，只能写成 generic training-dynamics insight，不能写成 KAN-specific；
6. 只有 KAN 在同机制下超过 MLP / controls，才允许讨论 KAN-specific source-channel advantage。
```

v22.03 的真实状态是：

```text
Code:
  v22.03 required path 基本闭合，但全历史 repo 仍不闭合；机制合同表字段语义需要修正。

Efficiency:
  D-CHE / D-FOU 已经是主力 high-efficiency carrier。
  D-CHE full-loop official closure = 1。
  D-FOU full-loop official closure = 1。
  D-RAT 从 forward 约 10x 推到 micro best forward 约 1.05，但 official fused missing。
  D-RBF 从 forward 约 9x 推到 micro best forward 约 1.88，但 official fused missing。

Functional:
  MLP source 已经不是局部 positive，而是能形成 h100→h3200 的 continuous source chain；
  很多 h4800 仍为正，但 h4800 / h3200 retention ratio 不够；
  v22.03 best h4800 = 0.1503，best retention ratio = 0.4324，仍低于 productive gate；
  KAN route = KANSourceBankMismatch；
  promotion_allowed = 0。
```

因此 v22.04 的核心问题不是“有没有 source”，而是：

$$
\boxed{
\text{source preservation after h3200}
}
$$

以及：

$$
\boxed{
\text{KAN carrier 如何承载 MLP 中出现的 source channel}
}
$$

这版计划必须避免两种错误：

```text
错误 1：
  继续添加 Fxx 变体，只追更大的 h800/h3200 source。
  这会继续得到 h3200 很好、h4800 不够。

错误 2：
  只推进 D-CHE/D-FOU 效率，忽略 functional。
  这会得到高效 carrier，但仍没有 source-channel writer。
```

---

## 1. v22.04 的总假设

v22.04 不再问“哪个 update token 更好”，而是围绕下列假设设计实验。

### H1：h4800 失败主要不是 source amplitude 不足，而是 terminal erosion

定义：

$$
R_{4800/3200}
=
\frac{\max(0, Source_{h4800})}
{\max(\epsilon, Source_{h3200})}.
$$

v22.03 中许多候选的 $Source_{h4800}$ 仍为正，但 $R_{4800/3200}$ 不够。因此新的目标不是让 $Source_{h3200}$ 更大，而是让 $R_{4800/3200}$ 提高。

判断标准：

```text
如果 h4800 / h3200 提升，而 h3200 没明显下降，
说明 source preservation 方向有效。

如果 h800/h3200 更大但 h4800/h3200 更低，
说明只是放大短期 source，不能算推进。
```

---

### H2：terminal erosion 可能由四类机制导致

v22.04 要把 terminal erosion 拆成四类，而不是继续用一个 route 名称概括。

```text
OptimizerErosion:
  h3200 后普通 optimizer 的累计更新方向破坏 source。

CurvatureErosion:
  source direction 进入高曲率 / 高 NDS 区域，短期收益被二阶项吞掉。

DebtErosion:
  source 以 tail / LineC / calibration / AUC debt 为代价，terminal phase 债务爆发。

InformationVolumeCollapse:
  source trajectory 被压缩成低维但退化的方向，局部仍有 source，但无法保留。
```

对应必须记录：

$$
P_{\text{opt}}
=
\left\langle
\sum_{t=h3200}^{h4800} \Delta \theta_t,
u_{\text{source},h3200}
\right\rangle.
$$

$$
I^{(1)}
=
-\langle g, u\rangle,
\qquad
I^{(2)}
=
\frac{1}{2} u^\top H u.
$$

$$
\mathrm{NDS}(u)
=
\frac{u^\top H u}{\|u\|^2 + \epsilon}.
$$

其中 $u$ 是候选 source update 或 source-preserving direction。

---

### H3：source preservation 需要低曲率 / 低 NDS / matrix-block 约束，而不是单步 source 最大化

Muon curvature 研究的启发是：一阶收益相近时，后期稳定性可能由二阶曲率惩罚和 Normalized Directional Sharpness 决定。v22.04 不是“换成 Muon”，而是把这个 insight 转成：

```text
每个 FU candidate 必须记录：
  first-order gain；
  second-order penalty；
  update norm；
  NDS；
  within-block / cross-block NDS；
  h4800 retention。
```

判断标准：

```text
若 high h3200 candidate 的 NDS 高且 h4800 retention 低，
说明 terminal erosion 与 curvature/sharpness 有关。

若 low-NDS repair 提高 h4800/h3200，
说明 source-preserving low-curvature route 有价值。
```

---

### H4：row-orthogonal / angular source preservation 可能比 radial update 更稳定

Nora 的启发不是简单替换 optimizer，而是：

```text
矩阵参数的有效运动往往是角向运动；
径向 momentum / radial jitter 可能扰乱有效学习率和 source 保存。
```

v22.04 将在三类 matrix block 上测试 row-orthogonal source preservation：

```text
MLP hidden/readout matrices；
D-CHE degree-readout / low-degree source bank；
D-FOU band-readout / low-frequency source bank。
```

核心操作：

$$
u_{\perp,i}
=
u_i
-
\frac{\langle u_i, w_i\rangle}{\|w_i\|^2+\epsilon}w_i.
$$

记录：

```text
row_radial_fraction
row_orthogonal_fraction
angular_velocity
weight_norm_drift
source_retention_h4800
```

---

### H5：低维 manifold compression 不够，source path 还必须保留非退化 information volume

Constrained inference manifold 研究的启发是：低维流形压缩本身不保证可靠推理，还要保持非退化 information volume。v22.04 不把“source path 低维”当成功，而要同时看：

```text
intrinsic_dim(source_path)
information_volume(source_path)
condition_number(source_subspace)
h4800 retention
```

判断标准：

```text
如果 source path intrinsic_dim 降低但 information_volume collapse，
说明 source 被压成退化方向，不算好几何。

如果 intrinsic_dim 适度降低、volume 不退化、h4800 retention 提高，
说明 source 进入了稳定 signal channel。
```

---

### H6：微分同胚直觉应该转成可测约束：平滑、不撕裂、不折叠、可逆近似

你给的微分同胚材料强调“平滑变形，不撕裂不折叠”。v22.04 将 functional target 约束成：

```text
nearby samples after update remain nearby；
local neighbor order 不大面积翻转；
Jacobian condition 不恶化；
source update 不把 class boundary 折叠到异常区域；
noise/reservoir 不被注入 signal channel。
```

记录：

```text
neighbor_order_preservation
local_jacobian_condition_delta
source_path_fold_rate
class_boundary_flip_rate
actuation_R2
B2_transfer_gain
h4800_retention
```

ActuationR2 只做 actuator fidelity；它不能单独表示 success。

---

## 2. 哪些旧方向可以重开，哪些必须停止

### 2.1 可重开，但必须换语义

这些方向过去失败过，但失败可能来自代码口径、目标定义、one-shot 使用方式或旧 profiler，不应直接判死。

#### PopRisk / SNR

过去 PopRisk/SNR 作为 one-shot parameter SNR 或 cover objective 没打开，但 v22.04 可以重开为 **terminal-retention estimator**，不再直接生成 direction。

合法用途：

```text
predict h4800/h3200 retention；
predict NDS / debt risk；
predict signal-channel vs reservoir；
作为 precommit gate，不直接提交 update。
```

不合法用途：

```text
把 PopRisk/SNR top-k 参数直接当 FU direction；
用未来 h4800 outcome 训练 selector 后回填。
```

---

#### Split-consensus

过去 split-consensus 很容易变成局部 source 或 delayed rebound。v22.04 重开时只作为：

```text
cross-split source stability estimator；
terminal erosion risk estimator；
D-CHE / D-FOU source-bank mapping estimator。
```

不再作为“直接提交的 source vector”。

---

#### Function-space actuation

过去已经证明 high ActuationR2 不等于 retained source。v22.04 允许重开，但必须加上：

```text
diffeomorphic constraints；
low-NDS constraints；
information-volume constraints；
terminal retention gate。
```

---

#### Graph-free analytic adjoint / operator-level basis-channel

旧版 graph-free / operator-level basis-channel 很多失败来自 efficiency 或 actuation 不完整。现在 D-CHE/D-FOU 效率已经站住，可以重开 **basis-channel source mapping**：

```text
basis estimates source；
readout commits source；
low-degree / low-frequency bank carries stable source。
```

---

#### MLP functional update

MLP 不是 control-only，而是 functional mechanism discovery bench。v22.04 必须继续 MLP terminal-retention repair，因为 MLP 是目前唯一形成 h3200/h4800-positive source chain 的 carrier。

---

### 2.2 不再作为主预算方向

以下方向不应继续消耗主预算：

```text
action bank / controller / reset route；
cover objective 小修；
M31/M32 同族小修；
G-token / N-token / Q-token 扩展；
dataset-specific branch；
seed-specific scale；
只追 h800/h3200 source amplitude 的 terminal floor / lookahead / hold 小变体；
只看 ActuationR2 的 function-space target。
```

如果 Codex 想重启这些，必须先写 `revival_justification.md`，说明旧 no-go 是由哪一个已修复的代码 bug 导致；否则 route 直接判为 `ForbiddenOldRouteRestart`。

---

## 3. v22.04 总体实验结构

v22.04 分成四条并行主线。

```text
Line A: Code / Metric / Packaging Truth Gate
Line B: Basis Efficiency Officialization and Active Repair
Line C: Functional Terminal-Erosion Autopsy
Line D: Source-Preserving Functional Update and KAN Source Mapping
```

四条线必须同时推进，不允许只跑效率或只跑 functional。

---

# Line A：代码、指标、打包闭包硬门

## A0. 目标

确保本轮不再出现“运行环境通过，但用户收到的 code packet 不能独立 import”的问题。最终 zip 必须在解压目录中自检，而不是在原始 repo 里自检。

## A1. 必须打包文件

Codex 必须生成：

```text
v22_04_code_review_packet.zip
v22_04_results_bundle.zip
```

`v22_04_code_review_packet.zip` 必须包含：

```text
00_README.md
01_ENVIRONMENT/
02_SOURCE_TREE/
03_IMPORT_CLOSURE/
04_LINEC_CORRECTNESS/
05_SOURCE_CHAIN_AND_RETENTION/
06_DEBT_ACCOUNTING/
07_UPDATE_SEMANTICS/
08_FUNCTIONAL_MECHANISMS/
09_TERMINAL_EROSION/
10_BASIS_EFFICIENCY_KERNELS/
11_PROFILER_CORRECTNESS/
12_GPU_QUEUE/
13_REPRO_COMMANDS/
packet_manifest.csv
packet_sha256_manifest.csv
```

其中 `02_SOURCE_TREE` 至少包含：

```text
dgkan/fu/core.py
dgkan/fu/source_chain.py
dgkan/fu/terminal_retention.py
dgkan/fu/debt_accounting.py
dgkan/fu/terminal_erosion.py
dgkan/fu/source_preservation.py
dgkan/fu/diffeomorphic_target.py
dgkan/fu/matrix_block.py
dgkan/fu/poprisk_snr.py
dgkan/fu/mechanisms.py

dgkan/metrics/linec.py

dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py
dgkan/kernels/v17_basis.py

dgkan/profiling/efficiency_v20.py
dgkan/profiling/efficiency_v21.py
dgkan/profiling/efficiency_v22.py
dgkan/profiling/efficiency_v22_03.py
dgkan/profiling/efficiency_v22_04.py

experiments/run_v22_04_common.py
experiments/run_v22_04_s011_truth_gate.py
experiments/run_v22_04_efficiency_officialization.py
experiments/run_v22_04_drat_drbf_officialization.py
experiments/run_v22_04_terminal_erosion_autopsy.py
experiments/run_v22_04_source_preserving_fu.py
experiments/run_v22_04_kan_source_mapping.py
experiments/run_v22_04_merge_finalize.py
```

## A2. 解压后自检

Codex 必须在一个临时目录里执行：

```bash
unzip v22_04_code_review_packet.zip -d /tmp/v2204_packet_check
cd /tmp/v2204_packet_check/02_SOURCE_TREE
python -m compileall -q dgkan experiments tests
python experiments/run_v22_04_s011_truth_gate.py --mode import_closure
python experiments/run_v22_04_s011_truth_gate.py --mode linec_golden
python experiments/run_v22_04_s011_truth_gate.py --mode source_chain_tests
python experiments/run_v22_04_s011_truth_gate.py --mode terminal_retention_tests
python experiments/run_v22_04_s011_truth_gate.py --mode mechanism_contracts
python experiments/run_v22_04_s011_truth_gate.py --mode kernel_gradcheck
python experiments/run_v22_04_s011_truth_gate.py --mode profiler_phase_tests
```

这些命令必须基于 zip 解压内容运行，不能基于原 repo 运行。

## A3. A 线记录指标

必须输出：

```text
v22_04_required_source_files.csv
v22_04_compileall.csv
v22_04_import_closure.csv
v22_04_linec_fast_golden.csv
v22_04_linec_channel_golden.csv
v22_04_source_chain_unit_tests.csv
v22_04_terminal_retention_unit_tests.csv
v22_04_mechanism_contracts.csv
v22_04_profiler_phase_tests.csv
v22_04_kernel_gradcheck.csv
v22_04_code_route_decision.json
```

## A4. A 线成功标准

```text
required_source_files_missing = 0
compileall_error_count = 0
required_import_error_count = 0
LineC fast golden = 9/9
LineC channel golden >= 8/8
source_chain tests pass
terminal_retention tests pass
mechanism contract schema valid
profiler phase tests pass
kernel gradcheck pass or explicit deferred with blocker
```

如果 A 线失败，Codex 必须先尝试修复：

```text
missing source file -> add to packet and manifest
import error -> add dependency or compatibility shim
LineC fail -> fix MeasurementInvalid / audit-only semantics
source-chain fail -> fix source_eps and control-equivalent handling
profiler fail -> separate forward/backward/update/FU/audit phases
```

A 线不过，不允许写 functional no-go 或 efficiency no-go。

---

# Line B：基函数效率路线

## B0. 总目标

v22.04 的效率线目标不是继续 census，而是把 carriers 分成明确状态：

```text
OfficialEfficientCarrier
FullLoopEfficientCarrier
MicroNearE1Carrier
NearE1Carrier
ForwardKernelBlocked
MaterializationBlocked
OfficialFusedMissing
RejectedForFunctionalThisVersion
```

当前预期状态：

```text
D-CHE:
  进入 full-loop official closure reconfirm。

D-FOU:
  进入 full-loop official closure reconfirm。

D-RAT:
  从 micro-near-E1 推到 production near-E1 / official fused candidate。

D-RBF:
  从 micro-near-E1 推到 no-dense sparse local production candidate。

LQ / D-WAV:
  low-budget smoke，不占主 GPU。
```

---

## B1. D-CHE / D-FOU full-loop reconfirm

### 假设

D-CHE / D-FOU 的效率已经不是主要 blocker，但必须确认：

```text
profiler path；
functional runner path；
horizon readback path；
LineC audit path；
```

使用的是同一个 official kernel，不存在 fallback 到旧 torch path。

### 实验

对以下 variants 跑 batch 8/32/128/256/512：

```text
D-CHE:
  CHE21-R2-low-degree-k3-official
  CHE21-R4-k5-gradbuf-triton-ablation

D-FOU:
  FOU21-R3-tablelookup-bandreadout-official
  FOU21-R4-k4-triton-no-materialize-official
```

必须同时在两种 runner 中测：

```text
efficiency profiler runner
functional source runner
```

### 记录指标

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
step_total_ms

forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp

official_fused_kernel_complete
no_materialize_complete
same_kernel_functional_runner_proof
fallback_kernel_used
kernel_sha256
gradcheck_relerr
gradcheck_cos
batch_size
```

### 成功标准

D-CHE：

```text
>= 2 variants x >= 3 batch sizes S1 pass
forward_ratio <= 1.25
step_ratio <= 1.25
memory_ratio <= 1.05
same_kernel_functional_runner_proof = 1
fallback_kernel_used = 0
```

D-FOU：

```text
>= 1 variant x >= 3 batch sizes S1 pass
forward_ratio <= 1.25
step_ratio <= 1.25
memory_ratio <= 1.05
same_kernel_functional_runner_proof = 1
fallback_kernel_used = 0
```

如果 D-CHE / D-FOU fail，Codex 先尝试：

```text
if batch512 fails only:
  record batch512-specific blocker; do not invalidate batch128/256 carrier.

if profiler pass but functional runner fail:
  inspect kernel dispatch path and fallback flags.

if LineC/horizon audit pollutes timing:
  separate audit timing and rerun training-only timing.

if gradcheck fails:
  demote to FullLoopEfficientButKernelUnverified.
```

---

## B2. D-RAT active repair: rational forward officialization

### 假设

D-RAT 的 primary blocker 不是 memory 或 gradient correctness，而是 rational forward path：

```text
numerator eval；
denominator eval；
reciprocal/division；
denominator safety；
derivative telemetry；
materialization / audit pollution。
```

### 实验 variants

```text
RAT22.04-R1-horner-branchless-den
RAT22.04-R2-reciprocal-approx-trainpath
RAT22.04-R3-telemetry-free-forward
RAT22.04-R4-numden-fused-backward
RAT22.04-R5-safe-den-clamp-no-branch
RAT22.04-R6-vectorized-group-eval
```

每个 variant 跑 batch 8/32/128/256。

### 记录指标

```text
num_eval_ms
den_eval_ms
reciprocal_ms
safety_ms
telemetry_ms
readout_contract_ms
backward_num_ms
backward_den_ms
materialized_bytes
forward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
den_p01
den_min
r_prime_p99
r_double_prime_p99
NaN_count
Inf_count
official_fused_kernel_complete
functional_runner_kernel_match
```

### near-E1 成功标准

```text
forward_ratio <= 3.0
step_ratio <= 2.0
memory_ratio <= 1.20
gradcheck_pass = 1
denominator_safety_pass = 1
```

### E1 / S1 目标

```text
E1:
  forward_ratio <= 1.75
  step_ratio <= 1.50
  memory_ratio <= 1.10

S1:
  forward_ratio <= 1.25
  step_ratio <= 1.25
  memory_ratio <= 1.05
  official_fused_kernel_complete = 1
```

如果不满足 near-E1，Codex 先尝试：

```text
if num_eval dominates:
  use Horner fused numerator.

if den_eval dominates:
  fuse denominator and numerator in same pass.

if reciprocal dominates:
  compare torch reciprocal vs approximate reciprocal with safety audit.

if telemetry dominates:
  move derivative telemetry out of training path.

if materialization dominates:
  no-materialize group-eval path.

if safety branch dominates:
  branchless denominator guard.
```

D-RAT 在 near-E1 前不能进入 full functional matrix；只能跑 one-dataset one-seed limited smoke。

---

## B3. D-RBF active repair: sparse local no-dense officialization

### 假设

D-RBF 过去的 local support 没有转成 runtime sparsity。真正 blocker 是 dense materialization / active center fraction / exp eval。

### 实验 variants

```text
RBF22.04-R1-compact-local-k4-no-dense
RBF22.04-R2-active-center-mask-topk
RBF22.04-R3-exp-approx-trainpath
RBF22.04-R4-local-gather-fused-backward
RBF22.04-R5-width-conditioned-sparse
RBF22.04-R6-hybrid-local-linear-tail
```

### 记录指标

```text
active_center_fraction
mean_local_k
p95_local_k
dense_basis_materialized
materialized_bytes
gather_ms
exp_ms
local_contract_ms
backward_center_ms
backward_width_ms
forward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
width_p01
width_p99
center_grad_snr
official_fused_kernel_complete
functional_runner_kernel_match
```

### near-E1 成功标准

```text
forward_ratio <= 3.0
step_ratio <= 2.0
memory_ratio <= 1.20
dense_basis_materialized = 0
gradcheck_pass = 1
```

如果不满足，Codex 先尝试：

```text
if active_center_fraction > 0.50:
  tighten compact support and width schedule.

if exp_ms dominates:
  use exp approximation or piecewise quadratic local kernel.

if gather_ms dominates:
  sort by active center bucket and fuse local gather.

if materialized_bytes > target:
  remove dense [B,H,K] basis tensor completely.

if backward_width_ms dominates:
  freeze width in smoke or use width telemetry only.
```

D-RBF 在 near-E1 前不进入 full FU；near-E1 后只跑 limited smoke。

---

# Line C：functional terminal-erosion autopsy

## C0. 目标

Line C 不生成新 update。它只解剖 v22.03 最接近成功的 MLP candidates，回答：

$$
\boxed{
\text{h3200 到 h4800 的 source 为什么被侵蚀？}
}
$$

候选池：

```text
F118 / F121 / F122 / F123 / F120 / F119 / F106 / F115
F154 / F156 / F160 / F162
F148 / F150 / F157 / F159
```

每个 candidate 跑 9-row grouped readback，并至少对 top-6 做 detailed autopsy。

## C1. 必须记录 source trajectory

```text
source_h100
source_h400
source_h800
source_h1600
source_h2400
source_h3200
source_h4000
source_h4800

retention_h1600_over_h800
retention_h3200_over_h1600
retention_h4000_over_h3200
retention_h4800_over_h3200
retention_h4800_over_h4000

source_derivative_h3200_to_h4000
source_derivative_h4000_to_h4800
```

## C2. 必须记录 optimizer erosion

```text
optimizer_cumulative_projection_h3200_h4800
AdamW_momentum_cosine_to_source
SGD_momentum_cosine_to_source
Nora_row_orthogonal_projection_gain
radial_update_fraction
angular_update_fraction
source_direction_norm
update_norm_h3200_h4800
```

## C3. 必须记录 curvature / NDS

```text
first_order_gain_estimate
second_order_penalty_estimate
update_norm_squared
NDS_total
NDS_hidden
NDS_readout
NDS_source_block
NDS_non_source_block
curvature_group_balance
within_layer_NDS
cross_layer_NDS
```

`NDS` 的近似可用 Hutchinson / finite difference：

$$
u^\top H u
\approx
\frac{L(\theta+\epsilon u)-2L(\theta)+L(\theta-\epsilon u)}{\epsilon^2}.
$$

## C4. 必须记录 debt erosion

```text
tail_debt_peak
tail_debt_h3200
tail_debt_h4800
LineC_debt_peak
LineC_debt_h3200
LineC_debt_h4800
ECE_debt_h3200
ECE_debt_h4800
Brier_debt_h3200
Brier_debt_h4800
AUCtime_ratio_h4800
```

## C5. 必须记录 information volume / diffeomorphic geometry

```text
source_path_intrinsic_dim
source_path_info_volume
source_subspace_condition
neighbor_order_preservation
local_jacobian_condition_delta
fold_rate
class_boundary_flip_rate
signal_channel_energy
reservoir_energy
noise_leakage_proxy
```

## C6. 判断标准

C 线不追 promotion，但必须输出 terminal erosion taxonomy：

```text
OptimizerErosion
CurvatureErosion
DebtErosion
InfoVolumeCollapse
DatasetLocalizedErosion
ControlEquivalentTerminalDrift
UnknownTerminalErosion
```

分类规则：

```text
if optimizer_cumulative_projection < -threshold:
  OptimizerErosion

if NDS high and second_order_penalty explains source drop:
  CurvatureErosion

if tail/LineC/ECE/Brier debt increases sharply h3200->h4800:
  DebtErosion

if info_volume collapses while source_path_dim shrinks:
  InfoVolumeCollapse

if failure concentrated in one dataset/seed:
  DatasetLocalizedErosion

if stable-random shows same h4800 erosion pattern:
  ControlEquivalentTerminalDrift
```

如果 C 线无法分类超过 70% terminal erosion groups，Codex 必须：

```text
add finer horizon h3600/h4000/h4400/h4800；
add per-dataset/seed autopsy；
add readout/hidden decomposition；
add controls autopsy。
```

---

# Line D：source-preserving functional update

## D0. 总目标

Line D 才提交新 functional update。目标是：

$$
\boxed{
\text{在保持 h3200 source 的同时，提高 h4800 / h3200 retention。}
}
$$

因此 D 线评价不看 h800 source 最大值，而看：

```text
h4800_retention_ratio；
h4800_positive_count；
debt stability；
controls failure；
independent rerun。
```

## D1. MLP source-preserving repair

### Mechanism D1a：SourcePreservingLateProjection

在 h3200 后约束后续 update：

$$
\Delta \theta'_t
=
\Delta \theta_t
-
\lambda
\min(0, \langle \Delta \theta_t, u_s\rangle)
\frac{u_s}{\|u_s\|^2+\epsilon}.
$$

其中 $u_s$ 是 h3200 source direction。

实验：

```text
D1a-MLP-SPP-lambda025
D1a-MLP-SPP-lambda050
D1a-MLP-SPP-lambda100
D1a-MLP-SPP-readout-only
D1a-MLP-SPP-hidden-only
```

记录：

```text
projection_correction_norm
source_direction_cos_after_projection
h4800_retention_ratio
tail/LineC/ECE/Brier debt
step_time_overhead
```

---

### Mechanism D1b：NoraStyleRowOrthogonalSourceUpdate

对 matrix rows 做 row-orthogonal source-preserving update：

$$
u_{\perp,i}
=
u_i
-
\frac{\langle u_i, w_i\rangle}{\|w_i\|^2+\epsilon}w_i.
$$

实验：

```text
D1b-MLP-roworth-hidden
D1b-MLP-roworth-readout
D1b-MLP-roworth-hidden-readout
D1b-MLP-roworth-lateonly
```

记录：

```text
row_radial_fraction_before
row_radial_fraction_after
weight_norm_drift
angular_velocity
h4800_retention_ratio
```

---

### Mechanism D1c：LowNDSMatrixBlockFU

选择 source direction 中 NDS 最低的 matrix block：

```text
hidden block；
readout block；
low-rank source subspace；
source-plus-control-orthogonal block。
```

实验：

```text
D1c-MLP-lowNDS-hidden
D1c-MLP-lowNDS-readout
D1c-MLP-lowNDS-top2blocks
D1c-MLP-lowNDS-with-source-preservation
```

成功条件：

```text
NDS lower than baseline by >= 20%
h3200 source not reduced by more than 20%
h4800/h3200 >= 0.50
```

---

### Mechanism D1d：DualMemorySourceState

维护 short 和 long source state：

$$
s_t^{short}
=
\beta_s s_{t-1}^{short}
+
(1-\beta_s)u_t,
$$

$$
s_t^{long}
=
\beta_l s_{t-1}^{long}
+
(1-\beta_l)u_t.
$$

提交方向：

$$
u_t^{commit}
=
\alpha_s s_t^{short}
+
\alpha_l s_t^{long}.
$$

实验：

```text
D1d-MLP-dualmem-0p9-0p99
D1d-MLP-dualmem-0p8-0p995
D1d-MLP-dualmem-longonly-terminal
D1d-MLP-dualmem-short-form-long-preserve
```

判断：

```text
if short improves h800/h3200 but long improves h4800:
  dual-memory is useful.

if long reduces h3200 too much:
  tune alpha_l or use terminal-only long memory.
```

---

### Mechanism D1e：DebtAwareTerminalFU

h3200 后 priority 改为 debt reduction，而不是 source amplification：

$$
u_t^{terminal}
=
u_t^{source}
-
\gamma \Pi_{\text{debt}}(u_t).
$$

注意：`debt` 只能作为 audit/readback，不能直接读取 validation/test/future；这里的 debt proxy 必须来自 train-stream split / train-stream calibration proxy，不使用 heldout labels 生成方向。

实验：

```text
D1e-MLP-train-tail-debt-aware
D1e-MLP-split-LineC-debt-aware
D1e-MLP-train-calibration-proxy-aware
D1e-MLP-combined-debt-aware
```

---

### Mechanism D1f：InfoVolumePreservingFU

限制 source path 不退化成低维 collapse：

```text
maximize source retention;
preserve source_path_info_volume;
avoid source_subspace_condition explosion.
```

实验：

```text
D1f-MLP-info-volume-floor
D1f-MLP-info-volume-plus-lowNDS
D1f-MLP-info-volume-plus-roworth
```

---

## D2. KAN source-channel mapping

### 目标

KAN 不能直接复制 MLP update。要先确定 MLP source 在哪里，然后映射到 KAN 的稳定 source channel。

### D2a：MLP source decomposition

记录：

```text
hidden_source_fraction
readout_source_fraction
low_rank_rank90
singular_value_spectrum
source_alignment_with_gradient
source_alignment_with_momentum
source_alignment_with_lowNDS_blocks
```

如果 MLP source 主要在 readout，KAN 应优先做 readout-only commit；如果主要在 hidden matrix低秩方向，KAN 应找 low-degree / low-frequency source bank。

---

### D2b：D-CHE low-degree source bank

实验：

```text
D2b-CHE-lowdegree-bank-k3
D2b-CHE-lowdegree-readout-only
D2b-CHE-basis-estimate-readout-commit
D2b-CHE-lowdegree-roworth
D2b-CHE-lowdegree-lowNDS
```

记录：

```text
degree_energy_low/mid/high
source_projection_low_degree
source_projection_high_degree
readout_commit_fraction
basis_commit_fraction
h100/h400/h800/h1600/h3200/h4800 source
```

成功标准：

```text
h100/h400/h800 >= 0.005；
h1600/h3200 continuous；
h4800/h3200 >= 0.50；
controls fail；
not late rebound。
```

---

### D2c：D-FOU low-frequency source bank

实验：

```text
D2c-FOU-lowfreq-bank-k2
D2c-FOU-lowfreq-readout-only
D2c-FOU-band-estimate-readout-commit
D2c-FOU-lowfreq-roworth
D2c-FOU-lowfreq-lowNDS
```

记录：

```text
band_energy_low/mid/high
phase_drift
high_freq_ratio
source_projection_low_freq
source_projection_high_freq
horizon source chain
```

成功标准同 D-CHE。

---

### D2d：Diffeomorphic source target

对 D-CHE / D-FOU 设计 output-space displacement target，但必须满足：

```text
ActuationR2 high；
neighbor order preserved；
no fold；
info volume preserved；
NDS not high；
terminal retention improved。
```

实验：

```text
D2d-CHE-diffeo-loss-consensus
D2d-CHE-diffeo-lowdegree
D2d-FOU-diffeo-lowfreq
D2d-FOU-diffeo-readout
```

如果 ActuationR2 高但 h4800 retention 仍失败，则输出：

```text
TargetActuatesButDoesNotRetain
```

不允许再把 actuation success 写成 functional success。

---

## D3. Retention estimator / precommit selector

v22.03 的问题之一是很多 selector 是 retrospective。v22.04 要求所有 selector 使用提交前可见信息。

合法 precommit features：

```text
train_loss_h100/h400；
train-stream split gain；
cross-split source consistency；
NDS estimate；
source_path early info volume；
row radial fraction；
optimizer conflict proxy；
train-stream tail proxy；
LineC-fast train-split proxy；
```

非法 features：

```text
h800/h1600/h3200/h4800 outcome；
validation/test/future；
audit metric direct target；
dataset-name branch。
```

输出：

```text
v22_04_precommit_selector_matrix.csv
```

记录：

```text
feature_name
uses_future
uses_validation
uses_test
uses_audit_target
AUC_predict_h4800_retention
precision_at_k
recall_at_k
selected_group_h4800_retention
independent_rerun_pass
```

成功标准：

```text
precommit_selector_pass = 1
uses_future/validation/test = 0
selected independent rerun h4800/h3200 >= 0.50
matched controls fail
```

如果没有 h4800 positive labels enough，Codex 必须先 run source-preserving mechanisms to generate labels, not train selector from zero positives.

---

## 4. Controls

每个 functional candidate 必须至少比较：

```text
AdamW baseline
SGD baseline
Momentum baseline
NoOpMatchedOverhead
RandomMatchedNorm
StableRandomTrajectory
SameActuationR2RandomTarget
SameNDSRandomDirection
SameUpdateNormRandomDirection
SourcePreservingControlWithoutFU
RecoveryOnly
MLP same mechanism
```

特别新增：

```text
SameNDSRandomDirection:
  防止 low-NDS 本身解释全部收益。

SameActuationR2RandomTarget:
  防止 function-space actuator 本身解释收益。

SourcePreservingControlWithoutFU:
  防止只是后期保护 optimizer 就能解释收益。
```

---

## 5. 成功标准

### S0.11：代码与指标成功

```text
code packet self-contained
required import error = 0
LineC / source-chain / terminal-retention / debt tests pass
mechanism contract schema pass
kernel gradcheck pass or explicit deferred
4GPU queue artifacts complete
```

### E1：basis efficiency exploration

```text
forward_ratio <= 1.75
step_ratio <= 1.50
memory_ratio <= 1.10
gradcheck_pass = 1
```

### S1：basis efficiency official

```text
forward_ratio <= 1.25
step_ratio <= 1.25
memory_ratio <= 1.05
official_fused_kernel_complete = 1
same_kernel_functional_runner_proof = 1
fallback_kernel_used = 0
```

### F2：functional source chain

```text
h100 >= 0.005
h400 >= 0.005
h800 >= 0.005
h1600 >= 0.005
h3200 >= 0.005
controls fail
```

### F3：productive terminal retention

```text
h4800 > 0
h4800 / h3200 >= 0.50
row_positive_count_h4800 >= 7/9
tail_debt_recovery >= 0.50
LineC_debt_recovery >= 0.50
ECE/Brier not worse than best control
AUCtime_ratio <= 1.05
matched controls fail
independent rerun pass
```

### F4：KAN source-channel success

```text
D-CHE or D-FOU satisfies F3
KAN_specific_delta_vs_MLP_same_mechanism > 0
same carrier efficiency S1 pass
same source-channel controls fail
```

### S5：official success

```text
9/9 dataset-seed pass
source_vs_best_control >= 0.005
h4800/h3200 >= 0.50
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
Brier_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
promotion_allowed = 1
```

---

## 6. 可视化要求

必须生成以下图，不允许只给 CSV。

### Functional

```text
source_trajectory_h100_to_h4800.svg
retention_ratio_h4800_over_h3200.svg
terminal_erosion_taxonomy_bar.svg
optimizer_projection_vs_retention_scatter.svg
NDS_vs_h4800_retention_scatter.svg
debt_transition_h3200_h4800.svg
info_volume_vs_retention.svg
MLP_source_decomposition.svg
KAN_lowdegree_lowfreq_mapping.svg
controls_comparison_h4800.svg
```

### Efficiency

```text
basis_efficiency_forward_step_memory_pareto.svg
D_CHE_D_FOU_full_loop_timing.svg
D_RAT_component_waterfall.svg
D_RBF_component_waterfall.svg
kernel_gradcheck_summary.svg
profiler_vs_functional_runner_timing.svg
```

### Code / Queue

```text
code_packet_completeness_dashboard.svg
mechanism_contract_schema_dashboard.svg
gpu_queue_timeline.svg
gpu_busy_duration_by_device.svg
```

---

## 7. 4GPU 动态执行安排

### GPU0

```text
S0.11 truth gate
MLP terminal erosion autopsy
MLP source-preserving mechanisms D1a-D1f
```

### GPU1

```text
D-CHE / D-FOU full-loop efficiency reconfirm
D-CHE / D-FOU source-channel mapping D2b-D2c
```

### GPU2

```text
D-RAT active repair
D-RBF active repair
near-E1 limited smoke only if gate opens
```

### GPU3

```text
controls matrix
precommit selector
figures
merge/finalize
independent reruns for top candidates
```

必须输出：

```text
v22_04_runnable_queue.csv
v22_04_gpu_assignment_manifest.csv
v22_04_gpu_utilization_timeline.csv
v22_04_idle_violation.csv
v22_04_queue_drain_report.csv
```

4GPU 硬规则：

```text
if runnable_queue_nonempty and any_gpu_idle_over_10min:
  execution_contract_violation = 1
  final route cannot be completed-no-go
```

---

## 8. Codex fallback ladder

### 如果 source-preserving repair 提高 h3200 但不提高 h4800/h3200

Codex 先尝试：

```text
reduce source amplification strength；
increase preservation weight；
terminal-only preservation；
readout-only preservation；
low-NDS block-only preservation；
add debt-aware terminal phase。
```

### 如果 h4800/h3200 提高但 h3200 大幅下降

Codex 先尝试：

```text
two-stage schedule:
  source formation until h1600/h3200；
  source preservation after h3200。

increase short-memory coefficient；
reduce long-memory coefficient before h3200；
freeze preservation until h2400。
```

### 如果 D-CHE/D-FOU source mapping仍失败

Codex 先尝试：

```text
readout-only commit；
basis-estimate/readout-commit；
low-degree/low-frequency-only commit；
remove high-degree/high-frequency writeback；
map MLP source subspace into KAN readout first。
```

### 如果 D-RAT/D-RBF near-E1 不开

Codex 先尝试：

```text
D-RAT:
  remove telemetry from train path；
  branchless denominator；
  fused numerator-denominator；
  approximate reciprocal；
  freeze denominator for smoke.

D-RBF:
  reduce active center fraction；
  compact support hard cutoff；
  top-k local gather；
  remove dense basis materialization；
  approximate exp / piecewise quadratic.
```

### 如果 code packet 不自包含

Codex 必须：

```text
run unzip-based self-check；
add missing files；
update packet_manifest；
rerun import_closure from extracted directory；
do not proceed to scientific route.
```

---

## 9. 最终 route taxonomy

v22.04 final route 必须从以下选择，不允许写 vague no-go。

```text
R0-CodePacketInvalid
R1-CodePassEfficiencyPassFunctionalNotRun
R2-EfficiencyRegression
R3-DRATDRBFRepairBlocked
R4-MLPSourcePreservedKANMismatch
R5-TerminalErosionOptimizerDriven
R6-TerminalErosionCurvatureDriven
R7-TerminalErosionDebtDriven
R8-TerminalErosionInfoVolumeCollapse
R9-SourcePreservingFUWeakPositive
R10-ProductiveMLPGenericSource
R11-KANSourceChannelPositive
R12-OfficialS5Success
R13-NoActionableSourceTheoryAfterCompleteAutopsy
```

每个 route 必须附：

```text
evidence_table
blocking_metric
next_codex_action
deferred_items
```

---

## 10. v22.04 的最低可接受交付

如果没有 S5，也必须至少交付：

```text
1. 自包含 code packet；
2. terminal erosion taxonomy 覆盖 >= 70% groups；
3. D-CHE / D-FOU full-loop efficiency reconfirm；
4. D-RAT / D-RBF officialization progress or precise blocker；
5. 至少 4 个 source-preserving mechanisms 的 h100-h4800 grouped results；
6. MLP source decomposition；
7. D-CHE / D-FOU source-channel mapping results；
8. complete controls matrix；
9. complete visual dashboard。
```

如果这些都完成，即使没有 promotion，也不是原地打转；因为它将明确告诉我们 source 被什么侵蚀，以及 KAN 为什么没有 source-channel writer。

---

## 11. 简明执行摘要

v22.04 不再做三件事：

```text
不再追更大的 h800/h3200 source；
不再把 ActuationR2 当作 functional success；
不再让 D-RAT/D-RBF 躺在 blocked 表里。
```

v22.04 必须做三件事：

```text
1. 找出 h3200 -> h4800 erosion 的因果类型。
2. 用 source-preserving / low-NDS / row-orthogonal / dual-memory / info-volume 方法提升 h4800/h3200。
3. 把 D-CHE/D-FOU 作为 high-efficiency KAN source carriers，同时把 D-RAT/D-RBF 推到 production near-E1 或给出明确 kernel blocker。
```

最终目标：

$$
\boxed{
\text{让 functional update 从“能制造 source”推进到“能保存 source”，}
\text{并让 KAN 从“高效 carrier”推进到“能承载 source channel”。}
}
$$
