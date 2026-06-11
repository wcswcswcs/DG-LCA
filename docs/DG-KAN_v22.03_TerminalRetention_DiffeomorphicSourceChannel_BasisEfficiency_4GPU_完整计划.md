# DG-KAN v22.03：Terminal Retention / Diffeomorphic Source-Channel FU / Basis Efficiency Full-Loop 4GPU 完整计划

> 版本：v22.03 execution plan  
> 目标：停止继续堆 Fxx 小变体，正面解决 v22.02 暴露的两个硬问题：  
> 1. Functional update 已经能让 MLP source 连续到 h3200，甚至 h4800 仍有正值，但 h4800 相对 h3200 衰减过快；KAN 仍没有 early continuous retained source。  
> 2. D-CHE / D-FOU efficiency 已经进入 MLP-like envelope，但必须继续做 full-loop official closure；D-RAT / D-RBF 不能继续停在 blocked 表里，必须 active repair。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为 direction source。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是只找到一个局部 positive source。项目目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{最终得到比普通 MLP / PureKAN AdamW 更好的训练系统。}
}
$$

这里的“更好”必须同时满足：

```text
1. base / substrate 效率接近同参数量 MLP；
2. functional update 的收益超过 AdamW / SGD / Momentum / random / matched overhead / stable-random controls；
3. source 不是 single-row、不是 late rebound、不是 retrospective selector artifact；
4. source 至少能从 h100/h400/h800 形成 early chain，并连续保留到 h3200 / h4800；
5. tail / LineC / ECE / Brier / NLL / AUCtime debt 没有后期爆发；
6. MLP positive 只能写成 generic functional dynamics，不能写成 KAN-specific；
7. KAN-specific claim 只有在 D-CHE / D-FOU / D-RAT / D-RBF 等 carrier 打过 same-mechanism MLP control 后才允许；
8. 所有结果必须来自自包含 code packet，可独立 import / compile / rerun。
```

v22.02 的真实状态可以压缩成三句话：

```text
代码：
  v22.02 required path 基本自包含，source-chain >=0 bug 已修；
  但全历史 repo 仍未完全闭合，kernel gradcheck 还应独立于 efficiency runner。

效率：
  D-CHE / D-FOU 已经进入 MLP-like envelope；
  D-RAT / D-RBF 主动 repair 后仍被 forward / materialization 卡住。

functional：
  MLP source 已能连续到 h3200，很多 h4800 仍为正；
  但 h4800/h3200 retention ratio 不够，productive_h4800=0；
  KAN 仍没有 early continuous retained source。
```

因此 v22.03 的目标不是“再找更大的 h800 source”，而是：

$$
\boxed{
\text{把 MLP 的 h3200 source 保护到 h4800，}
\text{并找到 KAN 能承载 source 的低阶 / 低频 / readout source channel。}
}
$$

---

# 1. v22.03 的核心判断：现在卡在 terminal erosion，而不是 source absence

v22.02 的关键变化是：过去很多候选到 h4800 直接为负；现在很多 MLP 候选 h4800 仍然为正，只是相对 h3200 衰减太多。例如最接近的候选形态大致是：

```text
h100  > 0
h400  > 0
h800  > 0
h1600 > 0
h3200 > 0
h4800 > 0, but h4800 / h3200 < 0.50
```

所以失败不再应被笼统称为 terminal collapse。更准确地说：

$$
\boxed{
\text{当前 failure 是 terminal erosion：source 到 h4800 仍存在，但被后期训练侵蚀过多。}
}
$$

这会改变实验方向。继续扩大 h800/h3200 source 并不一定有用；真正要测的是 h3200 到 h4800 之间 source 为什么被侵蚀。

我们定义：

$$
R_{4800/3200}
=
\frac{\max(0, Source_{h4800})}{\max(\epsilon, Source_{h3200})}.
$$

当前 gate 要求：

$$
R_{4800/3200} \ge 0.50.
$$

v22.03 将直接围绕这个 ratio 做 terminal-retention repair，而不是再跑更多 source-amplitude 小修。

---

# 2. 外部研究进展给我们的启发

本节不是把外部 optimizer 或几何文章机械搬进来，而是抽取能转化成实验的原则。

## 2.1 Muon / curvature 研究：不要只看一阶 source，要看二阶 curvature penalty / NDS

`Why Muon Outperforms Adam: A Curvature Perspective` 的核心启发是：Muon 和 Adam 的一阶 gains 可以相近，但 Muon 的 realized loss decrease 更好，是因为它的二阶 curvature penalty 更小；该 penalty 可分解成 update norm 与 Normalized Directional Sharpness，Muon 的优势主要来自更低 NDS，而不是更小 update norm。

对应到 DG-KAN：我们不能只看 functional update 的一阶 source，也要看 source direction 是否在高曲率方向上付出过大二阶代价。v22.03 要记录：

$$
I^{(1)}(u)=\langle g,u\rangle,
$$

$$
I^{(2)}(u)=\frac{1}{2}\langle u,H[u]\rangle,
$$

$$
NDS(u)=\frac{\langle u,H[u]\rangle}{\|u\|^2+\epsilon}.
$$

如果一个 FU candidate 在 h800/h3200 source 很强，但 h3200 到 h4800 erosion 严重，并且它的 late-phase NDS 高，那么下一步应从 source-amplitude update 转为 low-NDS / matrix-block / row-normalized source-preserving update。

## 2.2 Nora：source preservation 不应只靠 scale，要稳定 row norm 与 angular velocity

Nora 的启发不是“直接换 optimizer”，而是它强调 matrix optimizer 的三个要求：efficiency、stability、speed。它通过 row-wise momentum projection 到权重正交补空间来稳定 weight norm 和 angular velocity，并保持 $O(mn)$ 复杂度。

对应到 DG-KAN：v22.03 不把 Nora 当成普通 optimizer sweep，而把它变成一个 terminal-preservation diagnostic：

```text
source-preserving late phase:
  update 不应在 h3200 后大幅破坏 source-carrying rows 的 norm / angle。
```

对 MLP hidden/readout matrix、D-CHE degree-readout matrix、D-FOU band-readout matrix，记录：

$$
\cos(u_t, W_t)=\frac{\langle u_t,W_t\rangle}{\|u_t\|\|W_t\|+\epsilon},
$$

并测试 row-orthogonalized source-preserving update 是否提高 $R_{4800/3200}$。

## 2.3 Constrained inference manifolds：低维压缩不够，必须保留 non-degenerate information volume

`Reasoning emerges from constrained inference manifolds...` 的启发是：内部动态会自组织成低维流形，但低维本身不保证可靠；健康动态需要表达能力、低维组织和压缩子空间中的非退化信息体积同时存在。

对应到 DG-KAN：我们不应该把 “source trajectory 变低维 / ActuationR2 高 / LineC pass” 单独当 success。v22.03 要为 source chain 加两个指标：

```text
ID_source_path:
  source trajectory 的 intrinsic dimension。

InfoVol_source_path:
  source trajectory 在压缩子空间里的 information volume。
```

成功候选不是 source trajectory 最低维，而是：

$$
\text{ID 适度降低，同时 InfoVol 不塌缩。}
$$

如果 h4800 erosion 伴随 InfoVol collapse，说明 source 被压成退化方向；如果 ID 过高且 source 不稳，说明 source 仍在 diffuse trajectory。

## 2.4 微分同胚视角：functional update 应像平滑可逆变形，而不是一次撕裂式 perturbation

用户给的“微分同胚”材料强调：深度表示变换可以理解为空间的平滑、可逆、不撕裂的变形。CNN / Transformer / flow matching / diffusion 都可以从连续变形、全局交互或可逆 ODE 的角度理解。

对应到 DG-KAN：v22.03 将 function-space target 从 “局部 ActuationR2 高” 改为 “diffeomorphic source transport”。也就是说，target 不仅要能被 actuation，还要满足：

```text
small smooth displacement；
no fold / no tearing；
Jacobian condition 不爆；
local neighbor order 不大幅破坏；
train split B1 -> B2 transfer 为正；
source 到 h4800 不被 terminal erosion。
```

对于 output logits / hidden representation / KAN low-degree bank，记录：

$$
J_T = \frac{\partial T(z)}{\partial z},
$$

$$
\kappa(J_T)=\frac{\sigma_{max}(J_T)}{\sigma_{min}(J_T)+\epsilon},
$$

以及 fold / neighbor-tearing 指标。v22.03 的 H-FU3 将比较：

```text
loss-cotangent target；
cross-split consensus target；
low-NDS diffeomorphic target；
random matched ActuationR2 target；
sign-flipped target；
non-diffeomorphic matched target。
```

## 2.5 Signal channel / reservoir 理论：不要把 h4800 positive 和 retained source 混淆

泛化理论提醒我们：训练运动会被分解到 signal channel 和 reservoir；干净 signal 应该进入 signal channel，噪声最好留在 reservoir。v22.02 已经说明很多 h4800 positive 不等于 productive h4800，因为 retention ratio 和 controls 仍不过。

v22.03 的结论标准将从：

```text
h4800 > 0
```

升级为：

```text
h4800 / h3200 >= 0.50；
source 在 signal-channel projection 中保留；
stable-random / matched target controls 不解释；
debt 不爆。
```

---

# 3. 哪些旧路线可以重开，哪些必须停止

## 3.1 可以重开的方向

### 3.1.1 PopRisk / SNR：重开为 signal-channel estimator，不再作为 one-shot parameter selector

v13.6-v13.8 的 PopRisk/SNR 线有价值，因为它把 source 从 future/oracle 转向 train-stream per-example gradient statistics。但之前的失败说明：one-shot SNR / parameter-level SNR / cover objective 不能直接 promotion。

v22.03 允许重开，但必须改语义：

```text
允许：
  作为 terminal retention predictor；
  作为 signal-channel noise estimator；
  作为 source-state diagnostic。

不允许：
  直接作为 parameter mask；
  直接生成 update；
  用 h4800 outcome 训练 selector。
```

### 3.1.2 Split-consensus：重开为 source target estimator，不再直接提交 update

v15.04-v15.7 说明 split-consensus 有 signal，但 source/hazard 重叠和 implementation 修复后仍不能 promotion。v22.03 重开它作为 train-only target estimator：

```text
B1/B2/B3 cross-split source consistency；
source target 在 output-space / low-degree / low-frequency bank 中的稳定性；
不直接把 split-consensus vector 当最终 update。
```

### 3.1.3 Function-space actuation：重开，但目标从 ActuationR2 改为 retention-aware diffeomorphic target

v20-v22 已证明 high ActuationR2 不等于 source success。重开条件是：ActuationR2 只是必要条件，target 必须同时满足 smoothness / no-tearing / InfoVol / terminal-retention。

### 3.1.4 MLP-FU：继续作为主实验台，不是普通 control

MLP 是目前唯一能形成 strong h3200 source chain 的 carrier。v22.03 先在 MLP 上解决 terminal retention，再映射到 KAN。不能因为它不是 KAN-specific 就丢掉。

### 3.1.5 Graph-free analytic adjoint：可以在 D-RAT / D-RBF kernel repair 中重开

v6.1 已证明 manual gradient correctness 不是最大问题，但当时 efficiency 失败。现在 D-CHE/DFOU 已有 official-like efficient path，D-RAT/D-RBF 仍卡 forward；graph-free / analytic adjoint 对 D-RAT denominator 和 D-RBF local K 仍有推进希望。

## 3.2 不应重启的方向

```text
action bank / controller；
reset route；
cover objective 小修；
M31/M32 同族修补；
G/N/Q-token 扩展；
dataset-name branch；
seed-specific scale；
ActuationR2-only promotion；
late rebound 放大；
terminal floor/lookahead/hold 同族小修。
```

这些方向不是永远没有数学价值，而是在本项目当前约束下已经消耗过大量预算并暴露出控制等价、不可复现、或 oracle upper-bound 不足的问题。

---

# 4. v22.03 总体实验目标

v22.03 必须同时推进三条线，不能顾此失彼。

```text
Line A: Code / Metric / Artifact Truth Gate
  目标：最终 zip 解压后自包含，route / source-chain / debt / kernel status 不再自说自话。

Line B: Basis Efficiency Full-Loop + D-RAT / D-RBF Active Repair
  目标：D-CHE / D-FOU 从 full-loop official closure 变成 functional runner proof；D-RAT / D-RBF 至少推进到 near-E1 或明确 kernel rejection。

Line C: Functional Update Terminal Retention
  目标：把 MLP h3200 source 的 terminal erosion 分型，并至少打开一个 productive h4800 candidate；同时构造 KAN low-degree / low-frequency source-channel writer。
```

---

# 5. Line A：代码审计 / 指标真值 / 打包硬门

## 5.1 目标

v22.02 的 required path 基本闭合，但历史 repo 仍有旧 import 问题，kernel gradcheck 仍依赖 efficiency runner。v22.03 要把 code audit 做成真正自包含门：

$$
\boxed{\text{final zip 解压后，直接在临时目录中 compile/import/test 全通过。}}
$$

## 5.2 Codex 必须打包的文件

输出：

```text
v22_03_code_review_packet.zip
```

结构：

```text
00_README.md
01_ENVIRONMENT/
  python_version.txt
  torch_version.txt
  cuda_version.txt
  gpu_info.txt
  pip_freeze.txt
02_SOURCE_TREE/
  dgkan/**/*.py
  experiments/**/*.py
  tests/**/*.py
03_IMPORT_CLOSURE/
  v22_03_compileall.log
  v22_03_required_import_closure.csv
  v22_03_full_repo_import_closure.csv
  v22_03_archive_legacy_allowlist.csv
04_METRIC_TESTS/
  linec_fast_golden.csv
  linec_channel_golden.csv
  source_chain_unit_tests.csv
  retention_formula_tests.csv
  debt_accounting_tests.csv
  route_aggregation_tests.csv
05_FUNCTIONAL_MECHANISM_TESTS/
  mechanism_contracts.csv
  update_semantics_tests.csv
  forbidden_direction_audit.csv
  control_equivalence_tests.csv
06_KERNEL_TESTS/
  che_fused_gradcheck.csv
  fou_fused_gradcheck.csv
  rat_kernel_gradcheck.csv
  rbf_kernel_gradcheck.csv
  no_materialize_audit.csv
07_PROFILER_TESTS/
  profiler_phase_isolation_tests.csv
  full_loop_timing_consistency.csv
08_RESULTS/
  raw_matrices/
  route_decision.json
  failure_taxonomy.csv
09_GPU_QUEUE/
  runnable_queue.csv
  gpu_assignment_manifest.csv
  gpu_utilization_timeline.csv
  idle_violation.csv
  queue_drain_report.csv
packet_manifest.csv
packet_sha256_manifest.csv
```

必须包含这些关键源码：

```text
dgkan/fu/core.py
dgkan/fu/source_chain.py
dgkan/fu/debt_accounting.py
dgkan/fu/mechanisms.py
dgkan/fu/source_channel.py
dgkan/fu/source_state.py
dgkan/fu/function_space_actuation.py
dgkan/fu/matrix_block.py
dgkan/fu/poprisk_snr.py
dgkan/fu/diffeomorphic_target.py

dgkan/metrics/linec.py

dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py
dgkan/kernels/v17_basis.py

dgkan/profiling/efficiency_v17.py
dgkan/profiling/efficiency_v20.py
dgkan/profiling/efficiency_v21.py
dgkan/profiling/efficiency_v22.py
dgkan/profiling/efficiency_v22_03.py

experiments/run_v17_common.py
experiments/run_v21_efficiency_officialization.py
experiments/run_v21_01_common.py
experiments/run_v21_01_source_retention.py
experiments/run_v22_03_code_truth_gate.py
experiments/run_v22_03_efficiency_full_loop.py
experiments/run_v22_03_terminal_erosion_autopsy.py
experiments/run_v22_03_source_preservation.py
experiments/run_v22_03_kan_source_channel_writer.py
experiments/run_v22_03_drat_drbf_repair.py
experiments/run_v22_03_finalize.py
```

## 5.3 自包含审计命令

Codex 必须在最终 zip 解压目录执行，而不是在原始 repo 执行：

```bash
python -m compileall -q dgkan experiments tests
python experiments/run_v22_03_code_truth_gate.py --mode required-import-closure
python experiments/run_v22_03_code_truth_gate.py --mode metric-tests
python experiments/run_v22_03_code_truth_gate.py --mode kernel-tests
python experiments/run_v22_03_code_truth_gate.py --mode profiler-tests
```

## 5.4 S0.10 成功标准

```text
compileall_ok = 1
required_import_error_count = 0
required_source_missing_count = 0
linec_fast_golden_pass = 1
linec_channel_golden_pass = 1
source_chain_tests_pass = 1
debt_tests_pass = 1
route_aggregation_tests_pass = 1
forbidden_direction_violation_count = 0
kernel_status_consistency_pass = 1
profiler_phase_isolation_pass = 1
```

如果 S0.10 不满足，Codex 不许写 functional no-go 或 efficiency official closure。必须先尝试：

```text
1. 补齐缺失源码；
2. 修 import closure；
3. 如果旧 legacy 脚本 import 不过，加入 archive allowlist，但 required path 必须 0 error；
4. 重新在解压目录跑 truth gate；
5. 重新生成 packet_manifest 和 sha256。 
```

---

# 6. Line B：基函数效率路线

## 6.1 Line B 总目标

v22.03 的效率目标分两层：

```text
B1: D-CHE / D-FOU full-loop official closure 与 functional runner integration。
B2: D-RAT / D-RBF active repair，至少推进到 near-E1 或明确拒绝原因。
```

D-CHE / D-FOU 已经接近可用，不应再只反复证明 profiler 快；必须证明它们在 functional training loop 里同样快。D-RAT / D-RBF 则不能继续只写 ForwardBlocked，必须拆解 forward kernel。

---

## 6.2 B1：D-CHE / D-FOU full-loop official closure

### 6.2.1 假设

$$
\boxed{
\text{D-CHE / D-FOU 已经不是效率 blocker；}
\text{如果 full functional runner 使用同一 kernel，它们应保持 MLP-like envelope。}
}
$$

### 6.2.2 实验对象

D-CHE：

```text
CHE22.03-R2-low-degree-k3-official
CHE22.03-R4-k5-gradbuf-triton
```

D-FOU：

```text
FOU22.03-R3-tablelookup-bandreadout-official
FOU22.03-R4-k4-no-materialize-official
```

MLP 同参数量 baseline：

```text
MLP-same-param-batch8/32/128/256/512
```

### 6.2.3 记录指标

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
LineC_audit_ms
horizon_readback_ms
training_step_ms
functional_step_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
functional_step_ratio_vs_mlp
memory_ratio_vs_mlp
forward_peak_memory_mb
backward_peak_memory_mb
basis_activation_bytes
functional_state_bytes
kernel_name
kernel_sha256
fallback_kernel_used
official_fused_kernel_complete
no_materialize_complete
full_loop_timing_pass
```

### 6.2.4 成功标准

Exploration E1：

```text
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
kernel_gradcheck_pass = 1
fallback_kernel_used = 0
```

Official-like S1：

```text
forward_ratio_vs_mlp <= 1.15
functional_step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.05
official_fused_kernel_complete = 1
no_materialize_complete = 1
full_loop_timing_pass = 1
```

Family closure：

```text
D-CHE: >= 2 variants x >= 3 batch sizes S1 pass
D-FOU: >= 1 variant x >= 3 batch sizes S1 pass
functional runner uses same kernel = 1
```

### 6.2.5 如果不满足，Codex 先尝试

```text
1. 检查 profiler path 与 functional runner path 是否不同；
2. 如果 fallback_kernel_used=1，强制 fail，修 kernel dispatch；
3. 如果 forward ratio fail，输出 component waterfall：basis eval / contraction / readout / audit；
4. 如果 functional_direction_ms 太高，拆 source-state update 与 projection；
5. 如果 batch512 fail，先做 batch128/256 official closure，不让 batch512 阻断所有效率路线；
6. 如果 D-FOU R4 失败但 R3 通过，先 officialize R3，不继续平均撒网。
```

---

## 6.3 B2：D-RAT active repair

### 6.3.1 假设

D-RAT 当前的失败不是 memory，也不是 gradient correctness，而是 rational forward path 太慢。需要验证：

$$
\boxed{
\text{如果把 numerator / denominator / reciprocal / telemetry 分离，}
\text{D-RAT 是否能至少进入 near-E1。}
}
$$

### 6.3.2 实验对象

```text
RAT22.03-R0-current-reference
RAT22.03-R1-Horner-numden-fused
RAT22.03-R2-branchless-denominator-safe
RAT22.03-R3-reciprocal-approx-trainpath
RAT22.03-R4-telemetry-free-trainpath
RAT22.03-R5-numden-fused-backward
RAT22.03-R6-low-degree-rational-readout-only
```

### 6.3.3 必须拆的 phase

```text
numerator_eval_ms
denominator_eval_ms
reciprocal_or_division_ms
denominator_safety_ms
derivative_telemetry_ms
train_forward_without_telemetry_ms
audit_telemetry_ms
numden_backward_ms
readout_contraction_ms
```

### 6.3.4 记录指标

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
denominator_min
denominator_p01
denominator_condition
r_prime_p99
r_double_prime_p99
telemetry_overhead_ratio
gradcheck_pass
finite_rate
```

### 6.3.5 成功标准

Near-E1：

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.20
gradcheck_pass = 1
denominator_p01 > safety_eps
finite_rate = 1.0
```

E1：

```text
forward_ratio_vs_mlp <= 1.75
step_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.10
```

D-RAT 只有达到 Near-E1 才能跑 limited functional smoke；未达 Near-E1 不允许进入 full FU matrix。

### 6.3.6 如果不满足，Codex 先尝试

```text
1. 如果 reciprocal dominates：改 reciprocal approximation / clamp / safe rsqrt path；
2. 如果 telemetry dominates：把 denominator/derivative telemetry 从 train path 移到 audit path；
3. 如果 numerator/denominator eval dominates：Horner + vectorized fused eval；
4. 如果 denominator safety fails：降级为 D-RAT SafetyBlocked，不跑 FU；
5. 如果 forward 仍 > 5x：D-RAT 本轮只保留 kernel research，不进入 functional smoke。
```

---

## 6.4 B3：D-RBF active repair

### 6.4.1 假设

D-RBF 的理论优势是 local support，但当前实现没有转成 runtime gain。需要验证：

$$
\boxed{
\text{D-RBF 是否可以通过 true sparse local-K / no dense materialization 进入 near-E1。}
}
$$

### 6.4.2 实验对象

```text
RBF22.03-R0-current-reference
RBF22.03-R1-compact-local-k4-no-dense
RBF22.03-R2-active-center-mask-fused
RBF22.03-R3-local-gather-contraction
RBF22.03-R4-exp-approx-trainpath
RBF22.03-R5-width-conditioned-local-k
RBF22.03-R6-sparse-local-backward
```

### 6.4.3 必须记录

```text
dense_basis_materialized
basis_materialized_bytes
active_center_fraction
mean_local_k
p95_local_k
center_occupancy_entropy
width_condition_p01
width_condition_p99
exp_eval_ms
local_gather_ms
local_backward_ms
readout_contraction_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
```

### 6.4.4 成功标准

Near-E1：

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.20
gradcheck_pass = 1
dense_basis_materialized = 0 or basis_materialized_bytes reduced >= 80%
mean_local_k <= 4.5
```

E1：

```text
forward_ratio_vs_mlp <= 1.75
step_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.10
```

### 6.4.5 如果不满足，Codex 先尝试

```text
1. 如果 active_center_fraction > 0.80：local support 实际太密，先修 width / center schedule；
2. 如果 basis_materialized_bytes 高：禁止 dense basis fallback；
3. 如果 exp_eval dominates：测试 exp approximation / table lookup / bounded support polynomial；
4. 如果 local gather dominates：改 tile-major / batch-major layout；
5. 如果 still forward > 5x：D-RBF 本轮不做 FU，只输出 KernelBlocked certificate。
```

---

# 7. Line C：Functional Update terminal-retention 主线

## 7.1 Line C 总目标

v22.03 的 functional 目标是：

$$
\boxed{
\text{把 MLP h3200 source 的 retained fraction 提高到 h4800 gate，}
\text{并诊断 KAN 为什么不能形成 early source chain。}
}
$$

成功不能再只看 h800 / h3200 source。v22.03 必须记录完整 source trajectory：

```text
h100, h400, h800, h1600, h2400, h3200, h3600, h4000, h4400, h4800, h5600, h6400
```

---

## 7.2 C0：Terminal erosion autopsy

### 7.2.1 假设

v22.02 的 top MLP candidates 已经有 h4800 正值但 retention ratio 不够。h3200->h4800 的 erosion 可能来自：

```text
OptimizerErosion；
DebtErosion；
DatasetLocalizedErosion；
ReadoutSourceDecay；
HiddenSourceDecay；
ControlEquivalentTerminalDrift；
InfoVolumeCollapse；
HighNDSCurvaturePenalty；
TargetMisalignment。
```

### 7.2.2 实验对象

从 v22.02 old-shape / source-chain 表中选择：

```text
Top MLP near-h4800 candidates:
  F118, F121, F122, F123, F120, F119, F106, F115, F53, F70, F74

Controls:
  AdamW
  NoOpMatchedOverhead
  RandomMatchedNorm
  StableRandomMatched
  SameActuationR2RandomTarget
  SameSourceNormRandomDirection
```

### 7.2.3 记录指标

Source：

```text
source_h100
source_h400
source_h800
source_h1600
source_h2400
source_h3200
source_h3600
source_h4000
source_h4400
source_h4800
source_h5600
source_h6400
R_4800_3200
R_5600_4800
R_6400_4800
source_derivative_h3200_to_h3600
source_derivative_h3600_to_h4000
source_derivative_h4000_to_h4400
source_derivative_h4400_to_h4800
```

Optimizer projection：

$$
P_{erase}(a,b)=
\left\langle
\sum_{t=a}^{b} u_t,
\hat u_{source,h3200}
\right\rangle.
$$

记录：

```text
optimizer_projection_h3200_h4800
adamw_m_cos_source
adamw_v_precond_cos_source
momentum_cos_source
source_preserving_projection_violation_rate
```

Curvature / NDS：

```text
first_order_gain_I1
second_order_penalty_I2
normalized_directional_sharpness_NDS
NDS_h3200
NDS_h4800
NDS_delta_terminal
```

Debt：

```text
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
ECE_debt_peak
ECE_debt_final
Brier_debt_peak
Brier_debt_final
AUCtime_ratio
```

Representation / source location：

```text
hidden_source_fraction
readout_source_fraction
source_rank
source_singular_value_entropy
source_ID
source_information_volume
InfoVol_h3200
InfoVol_h4800
InfoVol_ratio_4800_3200
```

### 7.2.4 成功标准

C0 不要求直接 promotion。它要求每个 h3200 candidate 归入一个非 Unknown 的 failure class：

```text
terminal_erosion_class != Unknown for >= 80% top candidates
```

至少要给出：

```text
primary_erosion_mechanism ∈ {
  OptimizerErosion,
  DebtErosion,
  DatasetLocalizedErosion,
  ReadoutSourceDecay,
  HiddenSourceDecay,
  ControlEquivalentTerminalDrift,
  InfoVolumeCollapse,
  HighNDSCurvaturePenalty,
  TargetMisalignment
}
```

### 7.2.5 如果不满足，Codex 先尝试

```text
1. 增加 h3600/h4000/h4400 密集 horizon；
2. 对每个 candidate 输出 per-dataset/seed source trajectories；
3. 如果 optimizer projection unavailable，加入 checkpoint delta replay；
4. 如果 NDS 无法算全 Hessian，使用 Hessian-vector product / finite difference approximation；
5. 如果 InfoVol 不稳定，先用 source subspace singular values 与 covariance determinant proxy。
```

---

## 7.3 C1：Source-preserving terminal update

### 7.3.1 假设

如果 C0 显示 h3200->h4800 erosion 来自 optimizer 对 source 方向的负投影或高 NDS，那么在 h3200 后应该切换为 source-preserving update，而不是继续 source-amplifying FU。

### 7.3.2 Mechanisms

```text
C1-M1 SourcePreservingLateProjection
  h3200 后禁止 optimizer update 在 source direction 上强负投影。

C1-M2 NoraStyleRowOrthogonalSourceUpdate
  对 matrix rows 做 row-wise radial component control，减少 source-carrying row 的 norm/angle jitter。

C1-M3 LowNDSMatrixBlockFU
  在 hidden/readout matrix block 中选择低 NDS source-preserving direction。

C1-M4 DualMemorySourceState
  short EMA 负责 h800/h1600 source；long EMA 负责 h3200/h4800 retention。

C1-M5 DebtAwareTerminalFU
  如果 h3200->h4800 erosion 与 tail/ECE/Brier debt 同步，terminal phase 优先偿还 debt。

C1-M6 InfoVolumePreservingFU
  source update 不能让 source information volume 在 h4800 collapse。
```

### 7.3.3 记录指标

除 C0 所有 source/debt 指标外，必须记录：

```text
source_projection_clipped_fraction
source_negative_projection_before
source_negative_projection_after
row_radial_jitter
row_angular_velocity
NDS_before
NDS_after
short_memory_source_cos
long_memory_source_cos
InfoVol_before
InfoVol_after
terminal_update_overhead_ms
```

### 7.3.4 成功标准

Exploration S2：

```text
h100/h400/h800 >= 0.005
h1600/h3200 continuous source positive
R_4800_3200 >= 0.40
row_h4800_positive_count >= 6/9
matched controls fail
```

Productive S3：

```text
R_4800_3200 >= 0.50
row_h4800_positive_count >= 7/9
mean_source_h4800 >= 0.005
stable-random control fails
NoOpMatchedOverhead fails
RandomMatchedNorm fails
LineC/tail/ECE/Brier debt not worse than controls
AUCtime_ratio <= 1.05
independent rerun pass
```

### 7.3.5 如果不满足，Codex 先尝试

```text
1. 如果 h4800 remains positive but ratio < 0.50：提高 preservation strength，不提高 h800 source amplitude；
2. 如果 h3200 source drops：terminal update 太早或太强，改为 h3600 attach；
3. 如果 debt worsens：切换到 DebtAwareTerminalFU；
4. 如果 row jitter high：启用 Nora-style row orthogonal update；
5. 如果 NDS high：使用 LowNDSMatrixBlockFU；
6. 如果 controls also pass：标记 ControlEquivalentTerminalDrift，停止同族修补。
```

---

## 7.4 C2：Retention-aware function-space target reset

### 7.4.1 假设

v20-v22 的 high ActuationR2 失败说明 actuator 不是主瓶颈，target 定义才是瓶颈。v22.03 要验证：diffeomorphic / no-tearing / info-volume-preserving target 是否比 loss-cotangent target 更能产生 retained source。

### 7.4.2 Target families

```text
T1 LossCotangentTarget
T2 CrossSplitConsensusTarget
T3 LowNDSDiffeomorphicTarget
T4 InfoVolumePreservingTarget
T5 SignalReservoirSeparatingTarget
T6 LowRankReadoutTransportTarget
T7 RandomMatchedActuationR2Target
T8 SignFlippedTarget
T9 NonDiffeomorphicMatchedNormTarget
```

### 7.4.3 记录指标

Local actuation：

```text
ActuationR2
projection_residual_norm
B1_gain
B2_transfer_gain
B3_safety_gain
random_B2_gain
signflip_B2_gain
corrupt_B2_gain
```

Diffeomorphic constraints：

```text
jacobian_condition_mean
jacobian_condition_p95
fold_rate
neighbor_order_flip_rate
local_distance_distortion
smoothness_norm
curvature_norm
```

Retention：

```text
source_h100..h6400
R_4800_3200
tail/LineC/ECE/Brier/AUC debt
InfoVol_ratio_4800_3200
```

### 7.4.4 成功标准

Target observability gate：

```text
ActuationR2 >= 0.50
B2_transfer_gain > random_B2_gain + 0.005
fold_rate <= 0.02
neighbor_order_flip_rate <= matched_random
InfoVol not collapsed
```

Target retention gate：

```text
R_4800_3200 >= 0.50
mean_source_h4800 >= 0.005
matched target controls fail
```

### 7.4.5 如果不满足，Codex 先尝试

```text
1. 如果 ActuationR2 high but retention fail：target observable no retention，停止 ActuationR2-only variants；
2. 如果 fold_rate high：reduce target norm / increase smoothness / use low-rank transport；
3. 如果 B2 transfer not better than random：target not source-like，remove from FU matrix；
4. 如果 InfoVol collapse：add InfoVolumePreserving constraint；
5. 如果 T3/T4 outperform loss target locally but not terminally：send to C1 source-preservation repair。
```

---

## 7.5 C3：KAN source-channel writer

### 7.5.1 假设

MLP 能形成 source chain，KAN 不能，说明 KAN 的 source writeback channel 错。D-CHE / D-FOU 效率已基本站住，所以现在必须问：

$$
\boxed{\text{MLP source 对应的 signal channel，能否映射到 D-CHE low-degree / D-FOU low-frequency / readout bank？}}
$$

### 7.5.2 实验对象

MLP source decomposition：

```text
MLP-F118/F121/F122/F123 source candidates
hidden matrix source
readout matrix source
source rank and singular vectors
```

D-CHE writer：

```text
CHE-SW1-readout-only-commit
CHE-SW2-low-degree-bank-commit
CHE-SW3-basis-estimate-readout-commit
CHE-SW4-low-degree+readout-block-matrix
CHE-SW5-low-NDS-degree-readout
```

D-FOU writer：

```text
FOU-SW1-readout-only-commit
FOU-SW2-low-frequency-bank-commit
FOU-SW3-band-estimate-readout-commit
FOU-SW4-lowfreq+readout-block-matrix
FOU-SW5-low-NDS-band-readout
```

### 7.5.3 记录指标

```text
MLP_hidden_source_fraction
MLP_readout_source_fraction
source_rank
source_singular_entropy
source_projection_to_CHE_lowdegree
source_projection_to_FOU_lowfreq
KAN_readout_source_fraction
KAN_basis_source_fraction
high_degree_leakage
high_frequency_leakage
lowbank_source_retention
source_h100..h6400
R_4800_3200
KAN_specific_delta_vs_MLP_same_mechanism
```

### 7.5.4 成功标准

Exploration K-S2：

```text
KAN h100/h400/h800 >= 0.005
KAN h1600/h3200 continuous positive
matched controls fail
high_degree/high_frequency leakage <= threshold
```

Productive K-S3：

```text
KAN R_4800_3200 >= 0.50
KAN mean_source_h4800 >= 0.005
KAN row_h4800_positive_count >= 6/9
KAN_specific_delta_vs_MLP_same_mechanism > 0
same-kernel efficiency gate pass
```

### 7.5.5 如果不满足，Codex 先尝试

```text
1. 如果 readout-only works better than basis commit：keep basis-estimate/readout-commit, stop basis+readout mixed commit；
2. 如果 high-degree leakage high：hard low-degree mask / low-degree energy penalty as audit-only gate；
3. 如果 low-frequency source fails but readout works：DFOU basis source channel mismatch；
4. 如果 KAN still no early chain while MLP works：stop KAN source writer variants and return to MLP terminal retention first；
5. 如果 KAN late rebound only：do not promote; classify DelayedMigrationNotEarlySource。
```

---

## 7.6 C4：D-RAT / D-RBF limited functional smoke

D-RAT / D-RBF 只有达到 Near-E1 后才跑 limited smoke。

### 7.6.1 D-RAT smoke

```text
RAT-FU1 readout-only rational source
RAT-FU2 denominator-safe low-degree source
RAT-FU3 numerator-only source writer
```

记录：

```text
source_h100/h400/h800/h1600
rational denominator safety
denominator telemetry overhead
source vs D-CHE/DFOU/MLP same mechanism
```

Gate：

```text
Near-E1 pass first;
h800 source >= 0.005;
denominator safety pass;
controls fail.
```

### 7.6.2 D-RBF smoke

```text
RBF-FU1 readout-only local-support source
RBF-FU2 active-center low-K source
RBF-FU3 compact-local source writer
```

记录：

```text
active_center_fraction
mean_local_k
source_h100/h400/h800/h1600
materialization bytes
```

Gate：

```text
Near-E1 pass first;
h800 source >= 0.005;
dense_basis_materialized = 0 or reduced >=80%;
controls fail.
```

---

# 8. Full carrier × mechanism matrix

v22.03 的 matrix 不再平均铺开，而是按 readiness 分层。

## 8.1 Tier 1 full functional carriers

```text
MLP:
  primary source-retention lab。

D-CHE:
  efficient KAN carrier, low-degree source writer。

D-FOU:
  efficient KAN carrier, low-frequency source writer。
```

## 8.2 Tier 2 active repair + smoke carriers

```text
D-RAT:
  active kernel repair，Near-E1 后 limited smoke。

D-RBF:
  active kernel repair，Near-E1 后 limited smoke。
```

## 8.3 Tier 3 monitor

```text
LQ:
  low-budget forward repair monitor。

D-WAV:
  sparse support smoke only。
```

## 8.4 Mechanism matrix

```text
M0 Controls:
  AdamW, NoOpMatchedOverhead, RandomMatchedNorm, StableRandom, SameActuationR2RandomTarget。

M1 TerminalErosionAutopsy:
  no new update, only diagnosis。

M2 SourcePreservingLateProjection。

M3 NoraStyleRowOrthogonalSourceUpdate。

M4 LowNDSMatrixBlockFU。

M5 DualMemorySourceState。

M6 DebtAwareTerminalFU。

M7 InfoVolumePreservingFU。

M8 RetentionAwareDiffeomorphicTarget。

M9 KANLowDegreeLowFrequencySourceWriter。

M10 DRAT/DRBFLimitedSmoke。
```

---

# 9. 4GPU 动态队列

v22.03 必须真正利用四张 GPU，而不是静态分配后空等。

## 9.1 GPU assignment

```text
GPU0:
  S0.10 code truth gate；
  D-CHE full-loop efficiency；
  D-CHE source-channel writer；
  fallback: terminal erosion autopsy plots。

GPU1:
  MLP terminal erosion autopsy；
  MLP source-preserving repair；
  MLP diffeomorphic target；
  fallback: controls / independent rerun。

GPU2:
  D-FOU full-loop efficiency；
  D-FOU source-channel writer；
  D-RAT active repair if queue available。

GPU3:
  D-RAT active repair；
  D-RBF active repair；
  limited smoke for D-RAT/D-RBF when Near-E1 opens；
  fallback: figures / packet generation。
```

## 9.2 Required queue artifacts

```text
v22_03_runnable_queue.csv
v22_03_gpu_assignment_manifest.csv
v22_03_gpu_utilization_timeline.csv
v22_03_idle_violation.csv
v22_03_deferred_items.csv
v22_03_queue_drain_report.csv
```

每个 job 必须记录：

```text
job_id
line
carrier
mechanism
priority
gpu_id
start_time
end_time
wall_clock_sec
status
artifact_written
fallback_triggered
```

## 9.3 4GPU 成功标准

```text
queue_drained = 1
missing_tasks = 0
execution_contract_violation = 0
runnable_queue_nonempty_idle_minutes_per_gpu <= 10
per_gpu_busy_time_recorded = 1
```

如果某 GPU idle > 10 min 且 runnable queue 非空，Codex 必须先尝试：

```text
1. 启动 fallback jobs；
2. 把 independent rerun / controls / figure generation 分配到 idle GPU；
3. 如果无法执行，写明 dependency blocker；
4. 不允许写 completed-no-go。
```

---

# 10. 成功标准与 route 规则

## 10.1 S0.10 Code / Metric Truth Gate

```text
required_import_error_count = 0
required_source_missing_count = 0
metric_tests_pass = 1
kernel_status_consistency_pass = 1
source_chain_controls_not_early = 1
```

S0.10 不过时，所有 scientific route invalid。

## 10.2 Efficiency success

E1 Exploration:

```text
forward_ratio <= 1.25
step_ratio <= 1.25
memory_ratio <= 1.10
gradcheck_pass = 1
```

S1 Official-like:

```text
forward_ratio <= 1.15
functional_step_ratio <= 1.25
memory_ratio <= 1.05
official_fused_kernel_complete = 1
no_materialize_complete = 1
functional_runner_same_kernel = 1
```

Near-E1 for D-RAT/D-RBF:

```text
forward_ratio <= 3.0
step_ratio <= 2.0
memory_ratio <= 1.2
gradcheck_pass = 1
```

## 10.3 Functional success

S2 Early-to-h3200 source:

```text
h100/h400/h800 >= 0.005
h1600/h3200 positive
R_3200_800 >= 0.40
matched controls fail
```

S3 Terminal-retained source:

```text
mean_source_h4800 >= 0.005
R_4800_3200 >= 0.50
row_h4800_positive_count >= 7/9
stable-random control fail
NoOpMatchedOverhead fail
RandomMatchedNorm fail
tail/LineC/ECE/Brier debt not worse than controls
AUCtime_ratio <= 1.05
independent rerun pass
```

K-S3 KAN-specific:

```text
KAN S3 pass
KAN_specific_delta_vs_MLP_same_mechanism > 0
D-CHE or D-FOU efficiency S1 pass
same-kernel functional runner timing pass
```

S5 official:

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
R_4800_3200 >= 0.50
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

# 11. 必须生成的可视化

## 11.1 Functional figures

```text
source_trajectory_h100_to_h6400.svg
R4800_over_R3200_bar.svg
terminal_erosion_waterfall.svg
optimizer_projection_vs_source_loss.svg
NDS_vs_terminal_retention.svg
InfoVol_ID_source_path.svg
debt_transition_h3200_h4800.svg
MLP_source_decomposition_hidden_readout.svg
KAN_lowbank_source_projection.svg
controls_comparison_terminal.svg
```

## 11.2 Efficiency figures

```text
DCHE_DFOU_full_loop_efficiency.svg
DRAT_phase_waterfall.svg
DRBF_phase_waterfall.svg
basis_efficiency_pareto.svg
profiler_vs_functional_runner_timing.svg
kernel_status_consistency_dashboard.svg
```

## 11.3 Code / execution figures

```text
import_closure_dashboard.svg
metric_truth_gate_dashboard.svg
gpu_utilization_timeline.svg
queue_drain_gantt.svg
```

---

# 12. Failure taxonomy

v22.03 final route 不能写 vague no-go。必须落入以下之一：

```text
R0-CodeOrMetricInvalid
R1-EfficiencyOfficializationIncomplete
R2-DRATDRBFKernelBlocked
R3-MLPTerminalErosionExplained
R4-MLPTerminalErosionUnknown
R5-MLPS3RetainedGenericSource
R6-KANSourceChannelMismatch
R7-KANSourceWriterS3Pass
R8-ControlEquivalentTerminalDrift
R9-DebtErosionDominates
R10-OptimizerErosionDominates
R11-InfoVolumeCollapseDominates
R12-TargetMisalignmentDominates
R13-S5OfficialSuccess
```

每个 route 必须附：

```text
primary_evidence_artifact
control_status
next_action_queue
promotion_allowed
```

---

# 13. 预期最有价值的结果

v22.03 的最低有价值结果不是 S5，而是以下至少一个：

```text
1. 明确证明 terminal erosion 的主因是 OptimizerErosion / DebtErosion / InfoVolumeCollapse / TargetMisalignment 中某一类；
2. 至少一个 MLP candidate 达到 S3 terminal-retained generic source；
3. D-CHE 或 D-FOU 在 low-degree/low-frequency source writer 中打开 early chain；
4. D-CHE / D-FOU full functional runner efficiency 与 profiler 一致；
5. D-RAT / D-RBF 至少一个达到 Near-E1；
6. 若全部失败，必须给出非模糊 route，而不是 “more Fxx variants needed”。
```

---

# 14. 最终一句话

v22.03 的核心不是继续问：

```text
哪个 functional update 在 h800/h3200 更强？
```

而是问：

$$
\boxed{
\text{哪个 train-only source 能写入长期 signal channel，}
\text{并在 h3200 后不被 optimizer、debt、曲率或 target mismatch 侵蚀？}
}
$$

同时，效率路线不能只庆祝 D-CHE / D-FOU 已经快，也不能继续忽略 D-RAT / D-RBF。v22.03 要把 D-CHE / D-FOU 变成 functional runner 中的真实 efficient carrier，并把 D-RAT / D-RBF 主动推进到 near-E1 或给出清晰 kernel rejection。

