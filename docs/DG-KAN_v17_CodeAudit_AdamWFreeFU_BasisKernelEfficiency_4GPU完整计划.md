# DG-KAN v17：代码审计闭环 + AdamW-free Functional Update + Basis Kernel Efficiency + 4GPU 动态并行完整计划

> 版本：v17 execution plan  
> 生成时间：2026-06-02  
> 基于：`DG-KAN_v16.4.1_code_audit_bundle.zip` 本地解压静态审查、v16.4.1/v16.5 结果复盘、v12.5.1 好几何定义、v13.6-v13.8 PopRisk/SNR、v14-v16 functional no-go 证据、以及最近关于 optimizer dynamics / signal channel / boundary-conditioned training 的理论反思。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 本计划的核心修正：不再默认 `AdamW + FU` 是 functional update 的标准形态；不再让 LineC / tail / AUC / calibration 的实现错误或静态 gate 误导项目；不再只做 basis efficiency census，而是把 basis kernel efficiency 作为与 functional update 同等重要的突破主线。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed initialization；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为 direction source；禁止 action bank / controller / reset route 作为推进方式。

---

# 0. 项目总目标、当前进展与 v17 为什么必须大改

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN 或同等严格 carrier 上，通过 functional update 改善长期训练动力学，}
\text{最终得到比普通 AdamW/backprop 更好的模型。}
}
$$

这个目标包含两个必须同时成立的条件。

第一，functional update 必须有独立长期价值。它不能只是某个 horizon 上 source 暂时为正，也不能只是 MLP/generic optimizer controls 同样能解释的普通优化器收益。一个可推进的 functional update 必须满足：

```text
source 能在 h800/h1600 或更长 horizon 留存；
tail / LineC / calibration / AUC debt 能被偿还；
NoOp / Random / AdamW-extra-steps / Recovery-only / MLP analog controls 不能解释；
如果 MLP-FU 也成功，只能写成 generic training-dynamics insight；
只有 KAN carrier 明显强于 MLP analog，才允许讨论 KAN-specific functional advantage。
```

第二，basis / carrier 必须接近同参数量 MLP 的效率 envelope。每个 active basis 都必须知道自己相对 same-param MLP 的真实 phase-level 成本：

```text
forward 是否可承受；
backward 是否可承受；
parameter update 是否可承受；
functional direction / projection / commit 是否可承受；
LineC / tail / horizon audit 是否污染 training timing；
backward peak memory / optimizer state / basis activation / functional state 是否可承受。
```

v16.4.1 的真实结果说明，执行覆盖更完整，但 scientific capability 没有推进：efficiency truth table 已经有 `rows=114, measured=114`，但 functional smoke `source>=0.005 rows=0`；全局 best h800 source 来自 MLP，不是 KAN；h1600 后 best source 归零；KAN-specific advantage rows 为 0。也就是说，v16.4.1 的问题不是“没跑”，而是当前 FU family 没有形成长期 source retention，basis efficiency 也还没有形成完整可用 envelope。

v17 不能继续写成“再补一个 method”。v17 要做的是三件大事：

```text
1. 先审代码与指标是否正确，尤其 LineC、update sign、AdamW coupling、efficiency profiler。
2. 把 functional update 从 AdamW residual correction 改成 AdamW-free / optimizer-pluggable / role-partition / slow-state / matrix-block 的机制矩阵。
3. 把 basis efficiency 从 census 改成 family-specific kernel repair + correctness + phase-level profiler。
```

一句话：

$$
\boxed{
\text{v17 的目标不是继续解释为什么失败，}
\text{而是清理错误指标/错误实现，然后一次性裁决 AdamW-free FU 与 basis kernel efficiency。}
}
$$

---

# 1. 本地代码审计结论：当前代码包不能直接被认为可信

我已经本地解压 `DG-KAN_v16.4.1_code_audit_bundle.zip` 到：

```text
/mnt/data/dgkan_code_audit_v17
```

并执行了两类基础检查。

## 1.1 语法编译通过，但 import closure 失败

执行：

```bash
python - <<'PY'
import compileall
ok = compileall.compile_dir('.', quiet=1, maxlevels=10)
print('compileall_ok', ok)
PY
```

结果：

```text
compileall_ok True
```

但逐模块 import 检查失败：

```text
modules = 85
import errors = 17
```

典型错误包括：

```text
experiments.run_v1410_nonrat_fms_transfer_fms_definition_reset
  ModuleNotFoundError: No module named 'experiments.run_v133_task_family_robust_basis_natural'

experiments.run_v144_real_transfer_fms_all_basis_substrate
  ModuleNotFoundError: No module named 'experiments.run_v1231_basis_kernel_workspace'

experiments.run_v1641_functional_update_efficiency_breakthrough_4gpu
  ModuleNotFoundError: No module named 'experiments.run_v133_task_family_robust_basis_natural'
```

这意味着：

$$
\boxed{
\text{当前上传代码包不是自包含的；}
\text{v14-v16 的核心 runner 无法从这个 bundle 直接 import。}
}
$$

这不是小问题。只要 bundle 不能自包含，就不能证明 LineC、functional update、efficiency profiler 的实际执行语义可靠。v17 必须把 import closure 作为 S0 硬门：任何 runner 不能 import，就不能进入实验。

## 1.2 LineC 实现本身缺失，调用方还会吞掉异常

代码包中核心 `linec_metrics` 实现不在当前 bundle 内；多个 runner 从缺失文件导入它。更严重的是，调用方捕获任何异常后，把 LineC 结果写成 NaN 或 failed row，而不是 measurement invalid。

例如 `experiments/run_v144_real_transfer_fms_all_basis_substrate.py` 中，LineC 调用逻辑大致是：

```python
try:
    lm = linec_metrics(...)
    status = "executed"
except Exception as exc:
    lm = {"CouplingR2": nan, "NoiseSignalLeak": nan, "RealSignalReservoirRatio": nan}
    status = "blocked"
...
passed = int(fnum(CouplingR2, -999) >= 0.15 and ...)
...
linec_coupling_r2 = mean(fnum(row["CouplingR2"], 0.0))
```

`experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py` 中也有同类逻辑，并且在非 real LineC 场景里会使用 `linec_proxy(...)`。

这会产生一个非常危险的结果：

```text
LineC 实现坏了 / import 失败 / 参数错了；
代码不会把它判成 measurement invalid；
而是把该 row 记成 LineC fail / 几何失败；
最终路线可能错误地认为 functional update 撕裂几何。
```

这正是用户说的“原地打转”：不是程序死循环，而是错误指标或错误 gate 让我们不断拒绝可能有价值的方向，或者继续推进错误方向。

v17 必须修复：

$$
\boxed{
\text{LineC exception = MeasurementInvalid，不能计入 LineC fail。}
}
$$

## 1.3 Functional update 多处被 AdamW 包裹，当前结论只证明 AdamW+FU 组合失败

之前我默认 `AdamW + FU` 是标准形态，这是错误假设。代码审计也支持这个判断。

在 `experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py` 中，训练 loop 先创建 AdamW：

```python
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

然后 FMS/FU 常常被写成 pseudo-gradient：

```python
assign_flat_grad(specs, projected_flat)
opt.step()
```

也就是说，functional direction 并不是直接作为参数位移提交，而是交给 AdamW 的 moment、second moment、weight decay 和 update rule 再加工。这会带来三种混淆：

```text
1. AdamW 可能把 FU source 洗掉；
2. FU positive 可能只是 AdamW optimizer-state dynamics；
3. FU residual 很容易被 AdamW / Cautious / MGUP / matched controls 解释。
```

在 `experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py` 的 low-budget smoke 里，逻辑也不是独立 FU 训练，而是：

```text
第一步如果是 FU，则 apply one flat update；
之后继续 AdamW recovery。
```

对于 non-D-CHE carrier，`functional_projection` 还常常退化成 identity；这意味着所谓 FU smoke 很可能只是一个 one-step gradient pulse + AdamW recovery，而不是真正 carrier-specific FU。

v17 必须把 AdamW 从“默认主干”降级为一种 control/recovery 条件，而不是 functional update 的默认实现。

## 1.4 update sign / update kind 没有统一，可能造成方向语义混乱

不同 runner 中有两种语义：

```text
add_flat_update(..., alpha=+lr)
apply_flat_update(..., alpha=-lr)
```

有些函数返回的是 step-like direction，例如 `-m/sqrt(v)`；有些函数返回的是 gradient-like direction；有些函数又叫 update，但实际符号要靠调用方决定。这会导致：

```text
同一个 method name 在不同 runner 中可能方向相反；
matched controls 的符号可能不一致；
loss_after_small_step sanity 不可解释。
```

v17 必须引入统一的 `UpdateTensor` 数据结构，明确：

```text
kind = gradient / step / cotangent / metric_state
sign_rule = subtract / add / solve
space = parameter / function / basis_channel / matrix_block
source = train_stream / optimizer_state / split_gradient / random_control
```

并在每个 update 进入训练前执行 small-step sanity check。

## 1.5 多机制矩阵存在“名义多方案、实际同语义”的风险

v16.2/v16.3 里有 M1a..M8f 的 method surface，但代码上很多方法最终调用相同的 `choose_updates(...)`，核心只在 `pulse_kind`、recovery override、metric_choice 上变化。很多 method 名字并不保证实际 update vector 有实质差异。

这会导致：

```text
表面跑了 40 个 methods；
实际只跑了少数几类 update；
失败后说 full matrix no-go，但其实很多机制没有真正被裁决。
```

v17 必须新增 semantic non-collapse audit：

$$
\cos(u_i, u_j),\quad
\frac{\|u_i-u_j\|}{\|u_i\|+\epsilon},\quad
\text{active mask Jaccard},\quad
\text{rolewise energy distance}
$$

如果两个不同 method 的 update cosine $>0.995$ 且 active mask / norm / role energy 都几乎一样，则它们是 alias，不能算独立方案。

## 1.6 efficiency profiler 仍然存在污染风险

v16.3/v16.4.1 的 profiler 比以前强，但代码审计发现仍有风险：

```text
1. 部分 timing phase 会 mutate 同一个 active model；phase 顺序可能污染后续计时。
2. optimizer update timing 使用持久 AdamW 的改进是对的，但仍需要区分 persistent optimizer 与 fresh optimizer。
3. step_total_ms 可能混入 LineC / tail / horizon readback，而 MLP 不一定有等价 audit cost。
4. activation_saved_bytes 是由 peak memory - param - grad - optimizer_state 估算，不是真正 saved activation。
5. same-param MLP 是 hidden width 近似匹配，不是严格参数量匹配，必须输出 param_delta 并作为 gate。
```

v17 必须用 isolated cloned model / isolated optimizer state 做每个 phase 的 profiler，且 training step timing 和 audit readback timing 必须完全分开。

---

# 2. v17 的总设计：先修代码语义，再跑机制矩阵

v17 不允许直接跑新的大矩阵。必须先通过 S0 code-correctness gate。

v17 的执行分成三层：

```text
Layer 0: Code correctness and metric validity
  import closure、LineC golden tests、update sign、AdamW coupling、efficiency profiler correctness、kernel gradcheck。

Layer 1: Mechanism implementation refactor
  AdamW-free FU、optimizer-pluggable FU、role-partition optimizer、slow-state / matrix-block / function-space update。

Layer 2: Full experiment matrix
  full carrier matrix × full mechanism matrix × mandatory efficiency census × 4GPU dynamic queue。
```

如果 Layer 0 不过，不能写 scientific no-go，只能写 implementation no-go。

---

# 3. v17 代码重构计划

## 3.1 新增统一模块

v17 要把现在散落在多个 historical runner 中的逻辑抽出来，不能继续让 v17 runner import v13/v14/v15/v16 runner。

建议新增：

```text
dgkan/metrics/linec.py
  统一实现 LineC / train-probe coupling / signal-reservoir / noise leakage。

dgkan/fu/core.py
  UpdateTensor、ParamRoleSpec、FunctionalCarrierSpec、UpdateValidator。

dgkan/fu/loss_interface.py
  CE/Brier/通用 loss-interface output cotangent；只产生 train-stream cotangent，不含 audit direction。

dgkan/fu/mechanisms.py
  AdamW-primary FU, FU-primary, FU-only, SGD/Momentum+FU, schedule-free/slow-state FU, matrix-block FU, function-space proximal FU, PopRisk/SNR FU。

dgkan/fu/optimizers.py
  AdamW, SGD, momentum, schedule-free, lookahead, role-partition optimizer；都使用同一 UpdateTensor 接口。

dgkan/fu/controls.py
  NoOp、RandomMatchedNorm、AdamWParallelDirection、same-active、same-projection-retention、same-overhead、recovery-only。

dgkan/profiling/efficiency_v17.py
  phase-level timing / memory profiler，使用 cloned state，分离 training cost 和 audit cost。

dgkan/kernels/v17_che.py
  D-CHE recurrence / fused degree eval / analytic backward hooks。

dgkan/kernels/v17_fou.py
  Fourier recurrence / sincos precompute / fused band eval。

dgkan/kernels/v17_rbf.py
  active-center local K / no-dense materialization。

dgkan/kernels/v17_lq.py
  LQ frame / projection update / fixed-frame late attach。

dgkan/kernels/v17_rational.py
  branchless rational eval / denominator telemetry / analytic derivative readback。

dgkan/kernels/v17_wavelet.py
  sparse support index / sparse backward。
```

对应新 runner：

```text
experiments/run_v170_code_audit_and_semantic_tests.py
experiments/run_v171_adamw_free_fu_mechanism_matrix.py
experiments/run_v172_basis_kernel_efficiency_breakthrough.py
experiments/run_v173_full_carrier_mechanism_4gpu.py
experiments/run_v17_finalize.py
```

## 3.2 禁止 v17 直接依赖缺失旧 runner

v17 runner 不能从这些旧文件直接 import 核心逻辑：

```text
experiments.run_v133_task_family_robust_basis_natural
experiments.run_v1231_basis_kernel_workspace
run_v120_good_geometry_battery
```

旧 artifact 可以读，但核心实现必须搬入 `dgkan/*` reusable modules。否则 import closure 永远不可信。

---

# 4. S0：代码与指标正确性硬门

S0 是 v17 第一阶段，必须在任何训练矩阵前完成。

## 4.1 Import closure

执行：

```bash
python -m compileall dgkan experiments
python experiments/run_v170_code_audit_and_semantic_tests.py --check import_closure
```

必须满足：

```text
compileall_ok = 1
all_v17_imports_ok = 1
legacy_missing_import_count = 0 for v17 runners
```

如果旧 runner 缺 import，只能写 legacy_non_self_contained，不影响 v17 运行；但 v17 不能依赖它。

## 4.2 LineC golden tests

LineC 需要四类 golden tests。

### Golden 1：NoOp null

同一个 model、同一批数据，NoOp update 后：

```text
CouplingR2_delta ≈ 0
NoiseSignalLeak_delta ≈ 0
RealSignalReservoirRatio_delta ≈ 0
```

### Golden 2：RandomMatchedNorm null

随机同范数扰动必须不能系统性通过 LineC：

```text
random_false_positive_rate <= 0.05
```

### Golden 3：Synthetic known-transfer update

构造一个能让 B1 和 B2 同向移动的 synthetic linear model update，LineC 应该显示：

```text
CouplingR2 high
NoiseSignalLeak low
ReservoirRatio low
```

### Golden 4：Synthetic noise-leak update

构造一个只改善 B1、破坏 B2 的 update，LineC 应该显示：

```text
CouplingR2 low
NoiseSignalLeak high
```

如果 LineC exception 发生：

```text
linec_measurement_valid = 0
route = R0-LineCMeasurementInvalid
```

不能把它写成 LineC fail。

## 4.3 Update sign / semantics tests

每个 mechanism 必须通过：

```text
small_step_loss_sanity
opposite_sign_loss_sanity
matched_control_norm_sanity
rolewise_energy_sanity
```

定义：

$$
L_+(\epsilon)=L(\theta+\epsilon u),
$$

$$
L_-(\epsilon)=L(\theta-\epsilon u).
$$

如果 method 声称 $u$ 是 descent step，则应满足：

```text
L_plus <= L_minus or marked as perturbation_not_descent
```

若不满足但 method 是 productive perturbation，也必须显式标记：

```text
update_kind = trajectory_boundary
not one_step_descent
```

## 4.4 AdamW overwrite diagnostic

v17 必须记录 AdamW 是否洗掉 FU source。

给定 FU step $u_{FU}$，后续 optimizer 累积位移：

$$
\Delta\theta_{opt}(t,H)=\sum_{\tau=t}^{t+H}\Delta\theta_{opt,\tau}.
$$

记录：

$$
\cos\left(\Delta\theta_{opt}(t,H), u_{FU}\right)
$$

以及：

```text
optimizer_overwrite_projection = <Delta_opt, u_FU> / (||u_FU||^2 + eps)
source_retention_after_optimizer
orthogonal_drift_norm
```

如果 AdamW overwrite projection 长期为负，说明 AdamW 可能正在抵消 FU。

## 4.5 Efficiency profiler correctness tests

Profiler 必须满足：

```text
1. each phase uses cloned model state or restores exact snapshot;
2. forward timing excludes backward/update/audit;
3. backward timing includes real live gradients;
4. optimizer update uses persistent optimizer when mode says persistent;
5. fresh optimizer construction, if measured, has its own phase name;
6. audit readback separated from training step;
7. peak memory measured for each phase independently;
8. same-param MLP param delta <= 1% for official efficiency claim, <=5% for exploration;
9. blocked rows have exact blocker class.
```

## 4.6 Kernel correctness tests

每个 basis kernel 必须做：

```text
forward equivalence vs reference
input gradient equivalence
parameter gradient equivalence
finite difference spot check
no NaN/Inf
workspace no-dense-materialization audit
```

梯度误差：

$$
\operatorname{relerr}(g_m,g_r)=\frac{\|g_m-g_r\|_2}{\|g_r\|_2+\epsilon}.
$$

探索门：

```text
relerr <= 1e-4 or cosine >= 0.999
```

official kernel gate：

```text
relerr <= 1e-5 and cosine >= 0.9999
```

---

# 5. Functional update 新定义：AdamW 不是默认主干

v17 的最大 functional 改动是：

$$
\boxed{
\text{functional update 必须结合 backprop / loss-interface signal，}
\text{但不必结合 AdamW。}
}
$$

反传 / loss-interface 提供任务压力，例如：

$$
g_t = \nabla_\theta L(\theta_t),
$$

或输出空间 cotangent：

$$
\delta_t = \nabla_f L(f_{\theta_t}).
$$

但 optimizer 是如何把这个信号变成参数更新的系统。AdamW 只是其中一种选择。v17 必须测试 AdamW-free 形态。

---

# 6. v17 Functional Mechanism Matrix

## 6.1 Carrier matrix

v17 active carriers：

```text
C0-MLP:
  active mechanism discovery line, not only control.

C1-D-CHE:
  strongest current KAN carrier.

C2-LQ:
  reanchor + late attach + FU smoke.

C3-Rational:
  monitor + denominator-safe FU smoke; no reset/controller/action.

C4-D-FOU:
  substrate + kernel repair + minimum FU smoke.

C5-D-RBF/FastKAN:
  active-center / compact support + minimum FU smoke.

C6-D-WAV:
  sparse support + low-budget FU smoke.
```

每个 carrier 至少有：

```text
baseline optimizer rows;
AdamW-primary FU rows;
AdamW-free FU rows;
controls;
efficiency census;
LineC/tail/AUC debt readback.
```

## 6.2 Mechanism matrix

### M0：Controls

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWExtraStepsMatchedTime
SGDExtraStepsMatchedTime
RecoveryOnly
DecayOnly
SameActiveFractionRandom
SameProjectionRetentionRandom
MLP analog / KAN analog
```

### M1：AdamW-primary + FU residual baseline

这保留旧路线作为 baseline，不再作为默认主线：

$$
\Delta\theta = \Delta\theta_{AdamW} + \eta_{FU} u_{FU}.
$$

如果 M1 成功但 M2/M3 不成功，也要检查是否其实是 AdamW dynamics。

### M2：SGD/Momentum-primary + FU

测试 AdamW 是否在洗掉 FU source：

$$
\Delta\theta = \Delta\theta_{SGD/Momentum} + \eta_{FU} u_{FU}.
$$

比较：

```text
SGD+FU
Momentum+FU
NesterovMomentum+FU
AdamW+FU
```

### M3：FU-primary

FU 是主更新，optimizer 只提供 loss-interface signal，不提供 AdamW moment dynamics：

$$
\Delta\theta = \operatorname{FU}(\delta_t, J_\theta, \text{carrier state}).
$$

这里 backprop 仍然可以用来取得 cotangent / gradient，但不能调用 AdamW step。

### M4：FU-only smoke

只用 FU 更新关键 carrier 参数，非必要参数冻结或 slow update：

```text
KAN basis params: FU
readout/mixing params: frozen or SGD small lr
scale/decay params: decoupled shrink only
```

这条线回答：FU 本身是否能支撑训练的一部分。

### M5：Alternating FU / gradient training

不是每步混合，而是：

```text
k steps gradient optimizer
1 step FU boundary pulse
k steps recovery
```

recovery 比较：

```text
AdamW recovery
SGD recovery
Momentum recovery
Schedule-free recovery
Lookahead recovery
No optimizer recovery
```

### M6：Slow-state Functional Update

解决 h800 source 有、h1600 source 消失问题。

维护 slow source state：

$$
s_{t+1}=\beta s_t+(1-\beta)P_{signal}(u_t).
$$

提交：

$$
\theta_{t+1}=\theta_t+u_{base,t}+\eta_s P_{safe}(s_t).
$$

这里 $P_{signal}$ 只来自 train-stream split-consensus / PopRisk / SNR，不来自 validation/test/LineC/tail/AUC。

### M7：Matrix-block Functional Update

不再逐参数写 FU，而是在 matrix/block 空间保留 source：

```text
MLP hidden weight matrix
D-CHE degree-readout matrix
D-FOU band-readout matrix
D-RBF center-readout matrix
LQ projection frame
Rational numerator/denominator block
```

测试 block update：

```text
block-gradient
block-SNR
orthogonalized block step
SOAP/Muon-like diagnostic control
random matrix same spectrum control
```

### M8：PopRisk / SNR signal-channel FU

从 per-example gradient 的 coherent drift / noisy diffusion 构造 signal state：

$$
\mu_r=\frac{1}{b}\sum_i g_{i,r},
$$

$$
\sigma_r^2=\frac{1}{b}\sum_i (g_{i,r}-\mu_r)^2,
$$

$$
SNR_r=\frac{\mu_r^2}{\sigma_r^2/(b-1)+\epsilon}.
$$

用 $SNR_r$ 作为 role / basis / matrix-block preconditioner，不是 action token。

### M9：Function-space / operator FU

先求函数空间小位移，再映射回参数：

$$
\alpha^*=\arg\min_\alpha L_{B1}(f_\theta + J_{B1}U\alpha)+\lambda L_{B2}(f_\theta + J_{B2}U\alpha)+\rho\|\alpha\|^2.
$$

固定 alpha grid，不做 adaptive search。

### M10：Role-partition optimizer

不同参数角色使用不同更新规则：

```text
basis / degree / frequency / center params: FU or signal-state update
readout / mixing matrix: SGD/Momentum/Muon-like block update
scale / denominator safety params: decoupled shrink or slow EMA
non-essential params: frozen / slow update
```

这条线直接回应“hybrid 架构当初才需要 AdamW”的历史问题。

---

# 7. Basis Kernel Efficiency 主线

v17 对每个 basis 不再只做 census。每个 family 必须有：

```text
1. reference path;
2. repaired kernel path;
3. correctness test;
4. phase-level profiler;
5. task smoke;
6. FU smoke;
7. same-param MLP comparison.
```

## 7.1 D-CHE

目标：把 Chebyshev carrier 从 high-level recurrence / audit-heavy path 改成可训练 kernel path。

必做：

```text
CHE-K1 recurrence vectorization;
CHE-K2 fused degree eval + readout mixing;
CHE-K3 analytic backward for degree coefficients;
CHE-K4 split training cost vs LineC/horizon readback cost;
CHE-K5 low-degree active bank + high-degree late enable;
CHE-K6 no-materialize degree energy readback.
```

记录：

```text
che_degree_eval_ms
che_readout_mix_ms
che_backward_coeff_ms
che_backward_input_ms
che_audit_readback_ms
dense_degree_tensor_materialized
```

## 7.2 D-FOU

当前线索显示 D-FOU 主要是 forward bottleneck。v17 必须测试：

```text
FOU-K1 sin/cos precompute;
FOU-K2 recurrence-based sin/cos;
FOU-K3 fused band eval;
FOU-K4 low-frequency-only active bank;
FOU-K5 table lookup diagnostic;
FOU-K6 analytic backward for band coefficients.
```

如果 forward ratio 仍然 >1.75，必须输出 component waterfall。

## 7.3 LQ

LQ 当前 near/reanchor 有线索，但 forward/update 慢。v17 必须测试：

```text
LQ-K1 fixed-frame forward cache;
LQ-K2 fused projection update;
LQ-K3 persistent optimizer vs manual update;
LQ-K4 late attach without full reanchor;
LQ-K5 projection-frame matrix-block FU.
```

## 7.4 Rational

Rational 不重启 reset route，但要修 kernel/telemetry：

```text
RAT-K1 branchless denominator eval;
RAT-K2 denominator/derivative fused telemetry;
RAT-K3 numerator/denominator block update;
RAT-K4 analytic backward correctness;
RAT-K5 denominator safety hard readback.
```

## 7.5 D-RBF / FastKAN

核心是 active-center local K，不允许 dense materialization。

```text
RBF-K1 active-center occupancy;
RBF-K2 compact support K4;
RBF-K3 gaussian approximation / lookup;
RBF-K4 no-dense backward scatter;
RBF-K5 width condition guard.
```

## 7.6 D-WAV

```text
WAV-K1 triangular support index-only;
WAV-K2 sparse support backward;
WAV-K3 overlap damping;
WAV-K4 scale occupancy;
WAV-K5 local-tail coverage audit.
```

---

# 8. Mandatory Efficiency Census v17

每个 carrier / basis / mechanism / batch 必须记录：

```text
forward_only_ms
loss_delta_ms
backward_grad_ms
optimizer_update_ms
manual_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
kernel_eval_ms
basis_eval_ms
readout_mix_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_training_only_ms
step_with_audit_ms
```

memory：

```text
forward_peak_memory
backward_peak_memory
update_peak_memory
optimizer_state_memory
basis_activation_bytes
functional_state_bytes
audit_state_bytes
workspace_temp_bytes
actual_saved_tensor_bytes
manual_cache_bytes
```

ratios vs same-param MLP：

```text
forward_ratio
backward_ratio
update_ratio
training_step_ratio
memory_ratio
functional_overhead_ratio
audit_overhead_ratio
```

v17 hard gate：

```text
same-param MLP missing => route R0-EfficiencyReferenceMissing
phase-level timing missing => route R0-EfficiencyProfilerInvalid
audit cost not separated => route R0-AuditCostPolluted
basis row blocked without exact blocker => route R0-ProfilerInvalid
```

---

# 9. 4GPU 动态队列执行

v17 必须利用四张 GPU，不允许静态分配后空等。

## 9.1 GPU 初始分工

```text
GPU0:
  Code S0 tests + D-CHE FU matrix + D-CHE kernel repair.

GPU1:
  MLP FU matrix + MLP AdamW-free / slow-state / matrix-block tests.

GPU2:
  LQ + Rational reanchor/smoke + LQ/RAT kernel repair.

GPU3:
  D-FOU / D-RBF / D-WAV kernel repair + substrate/FU smoke.
```

## 9.2 动态补位规则

每张 GPU 有三层 queue：

```text
primary_queue
fallback_queue
fill_queue
```

如果 primary 完成或 blocked：

```text
1. 优先补同 carrier 的 missing mechanism；
2. 然后补 Line M controls；
3. 然后补 efficiency missing rows；
4. 然后补 h1600 extension；
5. 最后补 figure / audit generation。
```

必须生成：

```text
v17_runnable_queue.csv
v17_gpu_assignment_manifest.csv
v17_gpu_utilization_dashboard.csv
v17_idle_violation.csv
v17_deferred_items.csv
v17_queue_drain_report.csv
```

若 runnable_queue 非空且任一 GPU idle > 10 min：

```text
execution_contract_violation = 1
final completed-no-go invalid
```

---

# 10. 实验成功标准

## S0：代码/指标可信

必须满足：

```text
import closure pass;
LineC golden tests pass;
update sign tests pass;
AdamW overwrite diagnostic implemented;
efficiency profiler correctness tests pass;
kernel gradcheck pass for active kernel paths;
semantic non-collapse audit pass.
```

## S1：执行覆盖

必须完成：

```text
full carrier matrix;
full mechanism matrix;
mandatory efficiency census;
4GPU queue drain;
controls;
required / forbidden / no-action audits.
```

## S2：weak productive dynamics

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
tail_recovery_rate_h800 >= 0.40 or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

## S3：productive retained dynamics

```text
source_vs_best_control_h1600 >= 0.005
source_retention_h1600 >= 0.50
tail_recovery_rate_h1600 >= 0.60
LineC_recovery_rate_h1600 >= 0.60
AUCtime_ratio_h1600 <= 1.05
random pulse + same recovery fails
recovery-only fails
NoOp overhead fails
```

## S4：real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## S5：official success，不降低

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
promotion_allowed = 1
```

Efficiency official gate：

```text
forward_ratio <= 1.25
backward_ratio <= 1.40
update_ratio <= 1.40
training_step_ratio <= 1.25
backward_memory_ratio <= 1.25
```

Exploration gate：

```text
forward_ratio <= 1.75
backward_ratio <= 1.75
update_ratio <= 1.75
training_step_ratio <= 1.75
backward_memory_ratio <= 1.50
```

---

# 11. Stop / continue 规则

## 11.1 不能 hard stop 的情况

```text
D-CHE fail;
MLP fail;
LQ reanchor fail;
Rational smoke fail;
D-FOU/RBF/WAV <6/9;
AdamW-free FU fail;
FU-primary fail;
SGD/Momentum+FU fail;
slow-state FU fail;
matrix-block FU fail;
LineC immediate fail;
tail immediate fail;
h800 fail;
h1600 fail;
overhead high;
MLP/generic controls positive;
random pulse explains result;
efficiency repair fails;
```

这些必须进入 failure taxonomy / fallback ladder / exhaustion certificate / next hypothesis queue。

## 11.2 真正 hard stop

只有这些可以 hard stop：

```text
required artifact missing;
import closure fail for v17 runner;
LineC measurement invalid and no fix attempted;
forbidden information violation;
no-action-search violation;
direction uses validation/test/future/query;
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier;
dataset-name branch;
seed-specific scale;
action token / controller / action bank / reset route;
fake / proxy / CPU offload;
GPU idle while runnable queue non-empty and no hard reason;
```

---

# 12. 必须记录的指标

## 12.1 Functional dynamics

```text
source_vs_best_control_h1/h20/h100/h800/h1600
source_retention_h1/h20/h100/h800/h1600
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
calibration_debt_peak
calibration_recovery_rate
AUCtime_ratio
AUCdebt
optimizer_overwrite_projection
cos_FU_AdamW
cos_FU_SGD
cos_FU_Momentum
cos_FU_slow_state
cos_FU_matrix_block
semantic_update_alias_cluster
```

## 12.2 LineC

```text
linec_measurement_valid
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
linec_exception_type
linec_exception_message
linec_null_false_positive_rate
linec_golden_transfer_pass
linec_golden_noise_fail_pass
```

## 12.3 Efficiency

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
manual_update_ms
functional_direction_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
training_step_ms
step_with_audit_ms
forward_peak_memory
backward_peak_memory
update_peak_memory
optimizer_state_memory
basis_activation_bytes
functional_state_bytes
actual_saved_tensor_bytes
same_param_mlp_param_delta
same_param_mlp_forward_ratio
same_param_mlp_backward_ratio
same_param_mlp_update_ratio
same_param_mlp_step_ratio
same_param_mlp_memory_ratio
```

## 12.4 Kernel / basis

```text
kernel_forward_relerr
kernel_grad_relerr
kernel_grad_cosine
dense_basis_materialized
basis_eval_ms
basis_backward_ms
readout_mix_ms
degree_energy
band_energy
center_occupancy
width_condition
support_overlap
denominator_p01
derivative_p99
```

## 12.5 4GPU execution

```text
gpu_id
job_id
carrier
mechanism
planned_rows
executed_rows
start_time
end_time
runtime_sec
idle_time_sec
queue_depth
fallback_taken
deferred_reason
```

---

# 13. 必须生成的可视化

```text
fig_v17_code_audit_import_closure.svg
fig_v17_linec_golden_test_dashboard.svg
fig_v17_update_sign_sanity.svg
fig_v17_adamw_overwrite_cosine_curve.svg
fig_v17_carrier_mechanism_source_heatmap.svg
fig_v17_source_retention_horizon_curves.svg
fig_v17_tail_linec_debt_recovery_curves.svg
fig_v17_kan_vs_mlp_attribution_matrix.svg
fig_v17_semantic_noncollapse_cluster.svg
fig_v17_efficiency_forward_ratio_by_basis.svg
fig_v17_efficiency_backward_ratio_by_basis.svg
fig_v17_efficiency_update_ratio_by_basis.svg
fig_v17_efficiency_memory_ratio_by_basis.svg
fig_v17_phase_time_stacked_bar_by_basis.svg
fig_v17_memory_phase_stacked_bar_by_basis.svg
fig_v17_basis_kernel_component_waterfall.svg
fig_v17_gpu_utilization_timeline.svg
fig_v17_queue_depth_over_time.svg
fig_v17_job_completion_gantt.svg
fig_v17_route_taxonomy_heatmap.svg
```

---

# 14. Codex fallback ladder

## 14.1 如果 S0 import closure fail

Codex 必须：

```text
1. 生成 missing_imports.csv；
2. 把缺失 helper 搬入 dgkan/* reusable modules；
3. 不允许 v17 runner import legacy missing runner；
4. 重跑 compileall + import closure；
5. 若仍 fail，route = R0-CodeBundleNotSelfContained。
```

## 14.2 如果 LineC golden tests fail

Codex 必须：

```text
1. 定位是 implementation bug、statistical threshold bug、batch split bug、sketch bug 还是 gate bug；
2. 生成 linec_debug_packet；
3. 不允许把 LineC fail 当 geometry fail；
4. 修复后重跑 golden + null tests；
5. 若无法修复，所有 LineC-dependent scientific route invalid。
```

## 14.3 如果 update sign test fail

Codex 必须：

```text
1. 标记该 mechanism 返回 gradient-like 还是 step-like；
2. 修正 UpdateTensor kind/sign_rule；
3. 重跑 small-step sanity；
4. 若该 method 是 perturbation boundary，必须显式标记 not_descent，不得混入 descent gate。
```

## 14.4 如果 AdamW-free FU 全 fail

Codex 不能直接 no-go，必须继续：

```text
1. FU-primary low LR smoke；
2. SGD/Momentum+FU；
3. schedule-free/slow-state FU；
4. role-partition optimizer；
5. matrix-block FU；
6. AdamW overwrite diagnostic；
7. 写 CurrentFURequiresAdamW / AdamWWashesFU / FUIntrinsicNoGo 三分类。
```

## 14.5 如果 efficiency repair fail

Codex 必须按 family 输出 blocker：

```text
D-CHE: recurrence / readout / backward / audit pollution
D-FOU: sincos / recurrence / band eval / backward
LQ: frame forward / projection update / optimizer update
RBF: active center / gaussian eval / dense materialization / scatter backward
WAV: support indexing / sparse backward / overlap
RAT: denominator eval / derivative telemetry / branchless update
```

不能只写：

```text
efficiency fail
```

## 14.6 如果 GPU idle

Codex 必须：

```text
1. 检查 runnable_queue；
2. 若非空，自动调度 fallback/fill job；
3. 若无法调度，写 hard_blocker；
4. 没有 hard_blocker 的 idle >10min => route invalid。
```

---

# 15. v17 最终 route 逻辑

```text
R0-CodeOrMetricInvalid:
  S0 code/LineC/update/profiler/kernel gate fail。

R1-EfficiencyEnvelopeUnknown:
  mandatory efficiency census incomplete。

R2-AdamWWashesFU:
  FU source appears, but AdamW overwrite diagnostic shows source systematically erased; AdamW-free variants not yet pass。

R3-FUIntrinsicNoSource:
  AdamW-free / FU-primary / slow-state / matrix-block / PopRisk all source absent。

R4-SourceNotRetained:
  source appears at h800 but not h1600。

R5-DebtRecoveryFail:
  source retained but tail/LineC/calibration/AUC debt not recovered。

R6-ControlEquivalentOrGenericOnly:
  controls / MLP explain positive。

R7-BasisKernelEfficiencyBlocked:
  functional signal candidate exists but basis efficiency envelope fails。

R8-CarrierSpecificPartialPositive:
  some carrier/mechanism has weak source but lacks S4。

S4-RealTransferExplorationPositive:
  real-lite >=6/9, controls fail, no S5。

S5-OfficialFunctionalSuccess:
  9/9 strict pass, efficiency pass, controls fail, promotion_allowed=1。
```

---

# 16. 最终判断

v17 的核心不是“继续 v16.5”。v17 是一次必要的大改。

当前代码审计已经显示：

```text
1. 代码包不自包含，核心 runner import 缺失；
2. LineC 实现缺失且异常会被当成几何失败；
3. FU 多处被 AdamW 包裹，不能证明 AdamW-free FU no-go；
4. update sign / update kind 混乱；
5. 多机制矩阵有 alias / semantic collapse 风险；
6. efficiency profiler 有 mutation / audit pollution / memory estimate 风险；
7. basis repair 仍过多停留在 census，不是真 kernel repair。
```

因此 v17 的第一原则是：

$$
\boxed{
\text{先保证指标和实现不再误导实验，}
\text{再裁决 functional update 与 basis efficiency。}
}
$$

第二原则是：

$$
\boxed{
\text{functional update + backprop 必要；}
\text{functional update + AdamW 不必要。}
}
$$

第三原则是：

$$
\boxed{
\text{basis efficiency 必须从 truth table 进入 family-specific kernel repair，}
\text{否则 KAN 不能成为 next-gen MLP candidate。}
}
$$

v17 只有在 S0 代码/指标可信、S1 全矩阵覆盖、S2/S3 动力学 gate、以及 basis efficiency gate 中至少部分打开后，才算真正推进。否则只能写 implementation-correct no-go，而不能继续用错误指标或错误实现打转。
