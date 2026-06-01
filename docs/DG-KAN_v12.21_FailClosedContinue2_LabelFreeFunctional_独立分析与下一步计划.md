# DG-KAN v12.21：Fail-Closed Continue 2.0、Label-Free Signal Frame 与 Loss-Agnostic Functional Update 重建计划

> 基于：v12.20 `Fail-Closed Continue` 实验结果复盘。  
> 目标：不降低 promotion gate，不按数据集调参，不针对 CE 设计 functional direction；但也不能让 Codex 在第一个 gate fail 后停止。下一轮必须采用 **promotion fail-closed, diagnostics continue-open, hypothesis generation continues until budget exhausted** 的执行方式。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome 或 dataset 名称。CE/ECE/CEp99/Brier 只允许作为审计与坏化约束。

---

# 0. 一句话结论

v12.20 不是没有进展。它修正了 v12.19 的 fail-fast 问题：Line A / C / T / I / B / D continuation 都执行了，`continuation_missing_count = 0`。但它仍然不是能力突破，也不是 functional update 成功。它真正证明的是：

$$
\boxed{
\text{当前 label-free signal frame 没有替代 B320 的 supervised trainprobe signal；}
\text{当前 legal T1/T2 features 看不到 hard/soft release；}
\text{当前 inherited actuator dictionary 没有足够 safe movement/release capacity。}
}
$$

最终 route 是：

```text
R2-LabelFreeSignalFrameMissing
```

这不是一个“模型失败”的普通结论，而是一个 **机制缺口**：

```text
B320-current 很强，但依赖 supervised trainprobe initialization；
label-free B320-like attempts 没有同时通过 task + Line C；
functional update 没有合法 value source；
actuator 没有 safe/release movement；
P4 正确关闭。
```

下一轮不能继续给 projector token 做小修，也不能继续只做 visibility scorer 小修。v12.21 要解决的是：

$$
\boxed{
\text{怎样构造一个 label-free、precommit、loss-agnostic、control-resistant 的 value source，}
\text{并让它能驱动真实 primitive actuator？}
}
$$

---

# 1. v12.20 结果的独立分析

## 1.1 有进展，但不是 functional success

v12.20 的正进展在于实验纪律和失败定位：

```text
line_r_pass = 1
linea_rows = 270
linec_target_reset_rows = 216
T1A visibility completed
T2 upper-bound diagnostic completed
actuator_capacity_rows = 198
continuation_missing_count = 0
required_artifact_missing_count = 0
```

这说明 Codex 这次没有像 v12.19 那样在第一个 gate fail 后直接停止。所有预注册 continuation artifact 都生成了。

但是 promotion 全部关闭：

```text
label_free_official_pass_count = 0
c_t1_value_source_pass = 0
T1A_visibility_pass = 0
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 0
p4_open = 0
```

所以它不是 functional update 成功，也不是 label-free base 成功。它是一个更强的 no-go 证据。

## 1.2 Line A：label-free base 不是“差一点”

v12.20 的 Line A 不是没跑，270 行 continuation 已经覆盖 3 个数据集、3 个 seed、3/8 epoch、Line C batch/sketch 配置。结果最好的 official label-free 候选是：

```text
A1-noYForStats
mean_delta_vs_A0 = -0.0030381944444444445
worst_delta_vs_A0 = -0.03125
AUC_time_ratio_vs_mlp = 1.083605297058177
LineC all pass = 0
official = 0
```

它的 mean delta 接近，但最差 row、AUC-time、Line C 全不过。也就是说，A1 不是可以靠轻微阈值放宽得到的可靠 base。

新增 label-free frame attempts 更差：

```text
A23-MultiFrameBank-labelFree mean_delta_vs_A0 = -0.7388237847222222
A24-MultiFrameBankDirect125-labelFree mean_delta_vs_A0 = -0.7388237847222222
A27-MultiFrameLowFreqBias-labelFree mean_delta_vs_A0 = -0.7388237847222222
A25-SelfConditionResidualP-labelFree mean_delta_vs_A0 = -0.029513888888888888
A26-CovAdaptP-labelFree mean_delta_vs_A0 = -0.030164930555555556
```

A29 small-label oracle 接近 mean gate：

```text
mean_delta_vs_A0 = -0.004123263888888889
AUC_time_ratio_vs_mlp = 0.9535006537974816
```

但它使用少量真实 label，不能 promotion，并且 Line C 也没有过。这件事的含义是：supervised signal 对 B320-current 很关键，但简单 small-label oracle 也不能自动修复几何。

独立判断：

$$
\boxed{
\text{B320-current 的强性能不能再被当作纯 label-free architecture claim；}
\text{label-free signal frame 需要重新设计，而不是继续叠 projector token。}
}
$$

## 1.3 Line C：当前 C-T1 target reset 没有成为 value source

v12.20 尝试把 functional target 从 CE-residual hard release 转成 deployable geometric proxy，但相关性没有成立：

```text
c_t1_spearman_noise = 0.03645397607489652
c_t1_spearman_reservoir = 0.1970492654002053
c_t1_value_source_pass = 0
```

计划要求的是对 audit badness 的负相关，例如：

$$
\rho(G_{LA},\Delta NoiseSignalLeak)\le -0.30,
$$

$$
\rho(G_{LA},\Delta RealSignalReservoirRatio)\le -0.30.
$$

实际结果不但没有负相关，reservoir 还是弱正相关。说明当前 `G_LA` 不是 functional value source。

独立判断：

```text
Line C as audit：仍然有价值。
Line C as deployable value source：当前失败。
```

## 1.4 Line T：hard release 和 soft release 都不可见

T1A-only 结果：

```text
rows = 1728
feature_columns = 31
AUC_joint_min = 0.0
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
soft_target_r2_noise_min = -11956.931957256009
soft_target_r2_reservoir_min = -401245.5453505141
visibility_pass = 0
```

T2 upper-bound diagnostic 也没有救回：

```text
AUC_joint_min = 0.03773584905660377
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
soft_target_r2_noise_min = -13474.094195762538
soft_target_r2_reservoir_min = -261669.11702240884
```

这里有一个很值得警惕的信号：T2 是 richer diagnostic / upper-bound 级别，却出现了极低 AUC 和强负 R2。这不应该被简单解释成“彻底不可能”。下一轮必须做 **sign / label convention / join / rank direction sanity audit**。如果目标符号反了，AUC 可能会接近 $1 - 0.0377$；如果 join 仍然错，所有 R2 会被摧毁；如果都没错，才说明 target 对当前 feature family 真不可见。

因此 v12.21 不能只说“visibility fail，所以停”。它必须继续查：

```text
1. top-k 与 bottom-k 是否互换；
2. hard release label sign 是否与 delta convention 一致；
3. row join 是否在所有 source 上稳定；
4. soft target 是否被极端 scale / NaN / degenerate variance 污染；
5. T2 upper-bound feature 是否真的包含 clone-probe response，还是 inherited artifact 中缺失了关键列。
```

## 1.5 Line I：actuator 不是当前可用 executor

v12.20 继承 actuator dictionary 并做 exploratory capacity：

```text
actuator_capacity_rows = 198
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 0
```

示例：

```text
A1-DirectRole:
  sketch_delta_fro = 0.0010618959786370397
  projector_angle_deg = 0.08156468719244003
  logit_max_abs_drift = 0.007739067077636719
  NoiseSignalLeak_delta_audit = 0.00010123848915100098
  RealSignalReservoirRatio_delta_audit = 2.1517276763916016e-05

A4-ProjectionP:
  sketch_delta_fro = 0.0046552433632314205
  projector_angle_deg = 0.1453699767589569
  logit_max_abs_drift = 0.04452776908874512
  NoiseSignalLeak_delta_audit = -0.0001470521092414856
  RealSignalReservoirRatio_delta_audit = -0.001662135124206543
```

这些 movement 都离 gate 很远。当前 actuator 不是“差一点”，而是 **movement amplitude 和 release effect 都不足**。下一轮若继续用 inherited actuator dictionary，只会重复失败。

## 1.6 Line D：classic family 没有新进展

v12.20 的 Line D 是 monitor-only：

```text
classic_status_rows = 12
classic_new_hypothesis_count = 0
line_d_role = monitor_only
BSpline frozen
```

这不是失败，但也不是进展。下一轮如果 GPU/人力允许，经典支线必须只跑有新假设的 family repair；不能只是重跑旧 candidates。

---

# 2. 当前真正卡在哪里

我认为当前项目不再卡在“PureKAN 能不能高效训练”。B320/FHQ 主线已经证明效率和任务能力可以很强。当前卡在两个更深的问题：

## 2.1 Label-free signal frame 缺失

B320-current 的强结果来自 supervised trainprobe signal。去掉 `y_for_stats` 后，A1-noYForStats 虽然 mean 还接近，但 worst、AUC-time、Line C 不过。新 label-free attempts 也没有替代 supervised signal。

本质问题是：

$$
\boxed{
\text{当前 label-free geometry frame 不能把任务相关方向稳定放进 B320 的可训练坐标系统。}
}
$$

这不是靠再加一个 PCA / SRHT / augmentation-stable token 就能解决的。需要重新设计 **训练轨迹可见的 label-free signal frame**。

## 2.2 Functional update 没有合法 value source

Functional update 的合法方向必须：

```text
precommit；
loss-agnostic；
不读 label / CE / validation / future；
不按 dataset branch；
能预测或驱动 Line C audit 改善；
能打过 controls。
```

当前 T1/T2/C-T1 都失败，所以 functional update 没有 value source。此时构造 actuator 或 P4 short-run 会变成 blind search。

## 2.3 当前实验设置仍有一个设计问题：continuation 太“封闭”

v12.20 相比 v12.19 已经不再 fail-fast；但它仍然只完成了预注册 continuation，然后停在 `Success D`。这在科学审计上是正确的，但在研究推进上还不够。

下一版必须规定：

```text
Promotion fail -> 不 promotion；
Continuation fail -> 继续进入 next-hypothesis generator；
Generator fail -> 输出 explicit no-go theorem / reset target；
不能只以 Success D 结束，除非同时完成 no-go boundary 和下一轮 candidate generator。
```

---

# 3. 是否在正确道路上

方向是对的，但推进方式要改。

正确的是：

```text
1. 没有放宽 promotion gate；
2. 没有把 label-free fail 包装成 B320 成功；
3. 没有把 T2 diagnostic 当作 official；
4. 没有在 P3/P4 前置失败时强开 P4；
5. 没有按数据集调 controller；
6. 完成了 continuation queue。
```

需要改的是：

```text
1. v12.21 不能继续 projector token patch；
2. visibility fail 后必须做 sign/join/soft-target degeneracy sanity；
3. T1B 不能继续 inherited-unavailable，必须原生记录 optimizer-update observables；
4. actuator 不能只继承旧 dictionary，必须设计 stronger primitive-level actuator；
5. label-free base 与 functional value source 必须并行推进，不能互相卡死。
```

---

# 4. 离目标还差多远

| 模块 | 完成度估计 | 当前判断 |
|---|---:|---|
| B320-current diagnostic anchor | 80%-85% | 强，但 label-informed，不是 external-ready label-free claim |
| Label-free B320-like base | 35%-40% | A1 mean 接近，但 worst/AUC/LineC 不过；新 frame attempts 明显失败 |
| Line C audit | 80%-85% | 作为审计基本可用，但需要继续 target convention sanity |
| Deployable loss-agnostic value source | 5%-10% | C-T1、T1A、T2 全失败 |
| T1B native optimizer-observable features | 0%-5% | 当前 inherited artifacts 不包含，必须新增 runner logging |
| Actuator executor | 10%-15% | inherited dictionary movement 太弱，无 safe/release rows |
| Functional official update | 0% | P3/P4 正确关闭 |
| Classic no-BSpline portfolio | 40%-50% | 多数效率已开过，但本轮 monitor-only，无新 family hypothesis |

整体项目已经有强 base，但离最终目标还差一个核心闭环：

$$
\boxed{
\text{B320-like base} + \text{loss-agnostic functional update}
>
\text{B320-like base} + \text{ordinary backprop / controls}.
}
$$

这个闭环目前还没有开始 P4，因为 value source 和 actuator 都没过。

---

# 5. v12.21 总体目标

v12.21 不以“再找到一个 functional candidate”为目标，而以回答三个更本质的问题为目标：

## Q1. 是否存在 label-free signal frame 能替代 supervised trainprobe？

目标不是让 mean accuracy 接近，而是同时满足：

$$
\Delta Acc_{mean}\ge -0.005,
$$

$$
\Delta Acc_{worst}\ge -0.015,
$$

$$
AUCtime\le 1.0,
$$

$$
LineC_{nontearing}=1.
$$

## Q2. 是否存在 precommit / loss-agnostic observable 能预测或驱动 audit release？

目标不是提高 AUC 单项，而是：

$$
AUC_{joint,min}\ge 0.65,
$$

$$
Precision@K_{joint,min}>0,
$$

$$
Recall@K_{joint,min}>0,
$$

并且 leave-dataset / leave-seed / leave-window 不集中。

## Q3. 是否存在足够强且安全的 primitive-level actuator？

目标不是 logit drift，而是同时满足：

$$
sketch\_delta\_fro\ge 0.01,
$$

$$
projector\_angle\ge 1^\circ,
$$

$$
logit\_max\_abs\_drift\le 0.05,
$$

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio\le -0.01.
$$

---

# 6. v12.21 执行总原则：Fail-Closed Continue 2.0

## 6.1 三层 gate

v12.21 必须把 gate 分成三层：

```text
Promotion gate:
  决定是否可声称 success / 是否进入 P4 official。

Continuation gate:
  决定是否继续跑 fallback diagnostics。
  Promotion fail 不等于 continuation stop。

Branch-change gate:
  如果同一假设 family 连续失败，必须切换假设层级，而不是继续小修。
```

## 6.2 禁止单点终止

任何阶段失败都必须执行对应 fallback block，除非：

```text
1. 预算耗尽；
2. 代码 hard error 且无法恢复；
3. 结果已达到预注册 no-go boundary；
4. 缺少必要 artifact，route 写 R0。
```

例如：

```text
T1A fail:
  不能直接停。
  必须继续 T1B native logging、sign/join sanity、soft-target degeneracy audit、T2 upper-bound invert-check。

Line A fail:
  不能直接停。
  必须继续 label-free upper-bound decomposition、unlabeled frame alignment audit、small-label oracle interpretation。

Line I fail:
  不能直接停。
  必须继续 stronger actuator smoke and signed/budgeted response audit。
```

## 6.3 Route 降级规则

如果 Codex 只执行 promotion gate，未执行 continuation fallback，则 route 必须是：

```text
R0-FailFastIncomplete
```

如果 Codex 只执行 old hypotheses，没有进入 next-hypothesis generator，则 route 必须是：

```text
R1-ContinuationNoNewHypothesis
```

如果 Codex 完成所有 continuation 和 generator，但仍无合法 path，则：

```text
R2-LabelFreeSignalFrameMissing
R3-LossAgnosticValueSourceMissing
R4-ActuatorCapacityMissing
R5-TargetResetNoGo
```

---

# 7. Line R：代码与 artifact 语义审查

## 7.1 目标

确认 v12.21 不再被 artifact join、feature provenance、sign convention、label leakage 或 unavailable features 误导。

## 7.2 必须审查的代码路径

Codex 必须在复盘中写出以下文件、class/function、line range、tensor shape、输入/输出字段：

```text
CR0: run_v1221 entrypoint and route decision
CR1: B320-current and label-free model construction
CR2: y_for_stats / trainprobe / trainprobeP usage
CR3: label-free projector tokens and frame bank construction
CR4: Line C hard audit metric computation
CR5: T1/T2 feature construction and tiering
CR6: hard release label construction
CR7: row join key and row uniqueness
CR8: sign convention of all delta fields
CR9: soft-target regression target normalization
CR10: T1B optimizer-update native logging
CR11: actuator dictionary construction
CR12: controls and matched windows
CR13: P3/P4 gate code
CR14: no dataset-name branch audit
CR15: classic family monitor / new hypothesis handling
```

## 7.3 必须落盘

```text
v1221_code_review_manifest.csv
v1221_core_symbol_map.json
v1221_feature_provenance.csv
v1221_delta_sign_convention_audit.csv
v1221_join_key_uniqueness.csv
v1221_implementation_readback.md
```

## 7.4 Gate

如果 CR0-CR14 任一项缺失：

```text
route = R0-CodeReviewIncomplete
promotion_allowed = 0
```

---

# 8. Line A：Label-Free Signal Frame 重新设计

## 8.1 核心假设

当前失败不是因为 label-free frame 数量不够，而是因为 frame 与训练轨迹中的 signal/reservoir geometry 不对齐。

$$
H_A:
\text{存在不读 label/CE 的 signal frame，能接近 supervised trainprobe 的 task trajectory，且不破坏 Line C。}
$$

## 8.2 候选 family

### A30：Optimizer-Observable Signal Frame

利用 base optimizer 在训练早期产生的 update covariance，但不读取 CE vector 或 label。允许读取 opaque update vector：

```text
u_t = task_optimizer_update(theta_t)
```

记录的是 optimizer 已经产生的参数运动，不读取 per-example label residual。

构造：

$$
C_u = \sum_t u_tu_t^T.
$$

从 $C_u$ 提取低秩 frame 初始化 `quad_proj` / role frame。

### A31：Augmentation-Consistency Tangent Frame

对同一未标注样本的两种弱增强，计算 output / hidden stability方向：

$$
\Delta h = h(x+\epsilon_1)-h(x+\epsilon_2).
$$

目标不是分类，而是保留稳定方向、抑制不稳定方向。

### A32：Persistent Drift Frame

在前若干训练窗口记录 hidden / logit drift 的持久方向：

$$
D = \sum_t \Delta z_t\Delta z_t^T.
$$

选取跨 window 稳定的方向，而不是单 window PCA。

### A33：Role-Balanced Primitive Energy Frame

按 B320 primitive roles：direct / quad / branch / hinge / absdiag，构造 role energy balance。目标是避免 label-free candidate 过度依赖某个 role 导致 Line C fail。

### A34：Hybrid A1 + A30

以 A1-noYForStats 为底，只加入轻量 optimizer-observable frame，避免 A23/A24/A27 那种 full multiframe collapse。

### A35：Hybrid A1 + A31/A32

同样以 A1 为底，只加入小幅 augmentation / persistent drift residual，不重写整个 P frame。

## 8.3 运行设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = 3 and 8
train_size = 1024
val_size = 512
test_size = 512
batch_size = 128
LineC batch = 64
LineC sketch_dim = 24
```

如果有 candidate 同时满足 task + Line C，则扩展：

```text
seeds = 0..9
train_size = 2048 optional
```

## 8.4 必须记录字段

```text
v1221_label_free_signal_frame.csv

candidate_id
frame_family
uses_label
uses_ce_vector
uses_validation_or_test
uses_dataset_name
uses_optimizer_update_opaque
uses_clone_probe_response
mean_delta_vs_A0
worst_delta_vs_A0
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
ECE_delta_vs_A0
CEp99_delta_vs_A0
LineC_nontearing_all_pass
CouplingR2_delta_vs_A0
NoiseSignalLeak_delta_vs_A0
RealSignalReservoirRatio_delta_vs_A0
frame_rank
frame_condition
frame_energy_by_role
P_energy_on_frame
quad_feature_std_mean
branch_scale_norm
failure_reason
```

## 8.5 判断标准

Official label-free candidate 必须满足：

$$
mean\_delta\_vs\_A0\ge -0.005,
$$

$$
worst\_delta\_vs\_A0\ge -0.015,
$$

$$
AUCtime\_ratio\_vs\_MLP\le 1.0,
$$

$$
LineC\_nontearing\_all\_pass=1.
$$

## 8.6 不满足条件时 Codex 必须继续尝试

```text
如果 mean 接近但 worst fail：
  做 classwise / margin-p10 / CEp99 failure decomposition；
  尝试 role-balanced small residual frame，不做 dataset-specific threshold。

如果 AUC-time fail：
  同时看 AUC-step；
  AUC-step fail -> primitive trajectory fail；
  AUC-time-only fail -> profiler/overhead fail。

如果 LineC fail：
  输出 reservoir/noise decomposition；
  不继续 direct scalar patch；
  尝试 A30/A31/A32 中未跑的 frame family。

如果 A23-style collapse：
  禁止继续 full multiframe overwrite；
  只允许 A1 residual injection。
```

## 8.7 可视化

```text
fig_v1221_label_free_task_linec_pareto.svg
fig_v1221_frame_energy_by_role.svg
fig_v1221_auc_step_time_by_candidate.svg
fig_v1221_linec_noise_reservoir_by_candidate.svg
fig_v1221_worst_row_failure_heatmap.svg
```

---

# 9. Line C：Audit Target 与 Deployable Geometry Target 重建

## 9.1 核心假设

当前 hard release target 依赖 label/CE audit，不一定能直接被 T1 features 预测。v12.21 要构造新的 loss-agnostic deployable geometry target $G_{LA}$，并只用 audit target 验证 non-harm。

$$
H_C:
\exists G_{LA}\text{ label-free, precommit},
\quad G_{LA}\text{ 与 audit non-harm / release 有稳定统计关系。}
$$

## 9.2 候选 $G_{LA}$

### C1：Output covariance stability

$$
G_{cov}=\|\operatorname{Cov}(f_t(B))-\operatorname{Cov}(f_{t-1}(B))\|_F.
$$

### C2：Random-cotangent gradient sketch isotropy

用 random cotangent $v_k$ 构造：

$$
g_k = J^T v_k.
$$

看 sketch eigen spectrum 是否过度塌缩：

$$
G_{iso}=\frac{\lambda_1}{\sum_i\lambda_i+\epsilon}.
$$

### C3：Train-probe displacement stability

不使用 label，只看：

$$
\Delta U_B=f_{t+\Delta}(B)-f_t(B),
$$

$$
\Delta U_Q=f_{t+\Delta}(Q)-f_t(Q).
$$

记录 coupling residual，但不能直接用 audit label。

### C4：Primitive role energy drift

$$
G_{role}=\sum_r |E_r(t+\Delta)-E_r(t)|.
$$

### C5：Optimizer update spectrum

需要 v12.21 原生记录 opaque optimizer update：

$$
G_{opt}=\operatorname{spectrum}(\Delta\theta_{AdamW}).
$$

这仍是 loss-agnostic for functional source，因为不读取 CE vector / label；它只把 base optimizer update 当 opaque observable。

## 9.3 必须记录字段

```text
v1221_linec_deployable_targets.csv

source_run
dataset
seed
window
method
G_cov
G_iso
G_coupling_stability
G_role
G_opt_spectrum
G_composite
Delta_NoiseSignalLeak_audit
Delta_RealSignalReservoirRatio_audit
Delta_CouplingR2_audit
CEp99_delta_audit
ECE_delta_audit
uses_label_for_feature
uses_ce_for_feature
feature_tier
spearman_noise
spearman_reservoir
spearman_tail
```

## 9.4 判断标准

Deployable target 不需要直接等于 audit target，但必须满足：

$$
|\rho(G_{LA}, \Delta NoiseSignalLeak)| \ge 0.20
$$

或

$$
|\rho(G_{LA}, \Delta RealSignalReservoirRatio)| \ge 0.20
$$

作为 exploratory；official target 要求方向正确且：

$$
\rho(G_{LA}, \Delta NoiseSignalLeak)\le -0.30,
$$

$$
\rho(G_{LA}, \Delta RealSignalReservoirRatio)\le -0.30.
$$

如果 sign 不稳定，则不能作为 direction source，只能 diagnostic。

## 9.5 必须继续执行的 sanity block

无论 C1-C5 是否过，都必须执行：

```text
sign inversion audit；
top-k vs bottom-k audit；
row-join stability audit；
soft-target variance audit；
NaN / Inf / constant-target audit；
source-run leaveout audit。
```

## 9.6 可视化

```text
fig_v1221_GLA_vs_noise_audit.svg
fig_v1221_GLA_vs_reservoir_audit.svg
fig_v1221_target_sign_sanity.svg
fig_v1221_soft_target_residuals.svg
fig_v1221_geometry_target_family_ablation.svg
```

---

# 10. Line T：Visibility v5，必须原生记录 T1B

## 10.1 目标

v12.20 的 T1B 是 unavailable，不能再继续 inherited artifact scoring。v12.21 必须在 training runner 中原生记录 T1B opaque optimizer-update features。

## 10.2 Feature tiers

```text
T1A:
  pure precommit state features；
  logits/hidden/covariance/random-cotangent sketch；
  no CE/label。

T1B:
  optimizer-observable opaque update features；
  update vector norm/spectrum/cosine/role energy；
  no per-example CE vector/label。

T2:
  clone-probe diagnostic；
  不 promotion。

T3:
  audit-only CE/label target；
  不 direction source。
```

## 10.3 必须记录字段

```text
v1221_visibility_features.csv
v1221_visibility_scores.csv
v1221_visibility_leaveout.csv
v1221_visibility_sanity.csv

feature_tier
feature_family
feature_name
source_column
uses_label
uses_ce_vector
uses_validation_or_test
uses_future_outcome
uses_dataset_name
precommit_available
native_logged
clone_probe_only
score_direction
auc_noise
auc_reservoir
auc_joint
precision_at_k_joint
recall_at_k_joint
bottom_precision_at_k_joint
bottom_recall_at_k_joint
soft_target_r2_noise
soft_target_r2_reservoir
visibility_pass
failure_reason
```

## 10.4 Gate

T1 exploratory：

$$
AUC_{joint,min}\ge 0.60,
$$

$$
Precision@K_{joint,min}>0,
$$

$$
Recall@K_{joint,min}>0.
$$

T1 official：

$$
AUC_{joint,min}\ge 0.65,
$$

$$
Precision@K_{joint,min}\ge 0.10,
$$

$$
Recall@K_{joint,min}\ge 0.10.
$$

Leaveout 不能集中在单一 dataset/seed/window。

## 10.5 不满足条件时 Codex 必须继续尝试

```text
如果 T1A fail：
  继续 T1B native logging，不停止。

如果 T1B unavailable：
  route = R0-T1BNativeLoggingMissing。

如果 AUC < 0.5：
  执行 sign inversion / bottom-k audit；
  不允许只报告 fail。

如果 precision/recall = 0 但 AUC > 0.65：
  调查 class imbalance / support thin；
  扩 source/windows/seeds；
  不降低 release threshold。

如果 soft R2 强负：
  做 target normalization / row join / variance audit；
  若仍强负，标记 target unobservable。
```

## 10.6 可视化

```text
fig_v1221_visibility_auc_by_feature_family.svg
fig_v1221_precision_recall_top_bottom.svg
fig_v1221_leaveout_visibility_heatmap.svg
fig_v1221_T1A_vs_T1B_comparison.svg
fig_v1221_support_distribution.svg
```

---

# 11. Line I：Primitive-Level Actuator v2

## 11.1 目标

继承 v12.20 的 actuator dictionary 不够。v12.21 要测试更强、更贴近 B320 primitive 的 actuator basis，但仍只作为 diagnostic，除非 T/C visibility 过。

## 11.2 新 actuator families

```text
I1 direct-role scale / shift actuator
I2 quad-proj local rotation actuator
I3 hinge amplitude / hinge threshold actuator
I4 absdiag energy actuator
I5 branch gain redistribution actuator
I6 role-balanced combined actuator
I7 optimizer-update-aligned actuator
I8 random matched role-energy actuator control
I9 shuffled actuator basis control
```

## 11.3 必须记录字段

```text
v1221_actuator_response_dictionary.csv
v1221_actuator_controls.csv
v1221_actuator_safety.csv

actuator_id
role
basis_type
norm_budget
signed_direction
uses_label
uses_ce_vector
matched_control_id
sketch_delta_fro
projector_angle_deg
logit_max_abs_drift
CouplingR2_delta
NoiseSignalLeak_delta_audit
RealSignalReservoirRatio_delta_audit
CEp99_delta
ECE_delta
safe_movement_pass
release_audit_pass
control_gap
failure_reason
```

## 11.4 Gate

Exploratory safe movement：

$$
sketch\_delta\_fro\ge 0.01,
$$

$$
projector\_angle\ge 1^\circ,
$$

$$
logit\_max\_abs\_drift\le 0.05.
$$

Release gate：

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio\le -0.01.
$$

Control-resistant：

$$
control\_gap\ge 0.005.
$$

## 11.5 不满足条件时 Codex 必须继续尝试

```text
如果 movement 太弱：
  增加 norm_budget bracket 0.5x/1x/2x/4x，直到 logit safety 边界；
  不直接进入 P4。

如果 movement 强但 release 不变：
  说明 actuator 不是 release executor；
  转向 target/source，而不是继续加幅度。

如果 release 好但 control gap fail：
  增加强 controls；不 promotion。

如果 logit drift fail：
  加 role-balanced projection或降低频率；不调 dataset。
```

## 11.6 可视化

```text
fig_v1221_actuator_movement_vs_release.svg
fig_v1221_actuator_safety_pareto.svg
fig_v1221_actuator_control_gap.svg
fig_v1221_role_response_heatmap.svg
```

---

# 12. Line B：Functional Candidate Construction，仅在前置 gate 后打开

## 12.1 目标

Functional update 必须是 low-frequency geometry maintenance，不替代 AdamW/task optimizer。

形式：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{task}+\lambda_t\Delta\theta_{func}.
$$

其中：

```text
Delta theta task:
  ordinary B320 task optimizer update。

Delta theta func:
  loss-agnostic, precommit, low-frequency geometry maintenance。
```

## 12.2 候选

只有 Line C/T/I 至少一个合法路径通过，才构造：

```text
B1-G_LA_geometric_stabilizer
B2-T1B_update_spectrum_preconditioner
B3-role-balanced actuator maintenance
B4-noise-leak-vetoed functional event
B5-reservoir-release low-frequency event
B6-hybrid G_LA + actuator response solver
```

## 12.3 Controls

必须包含：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
ShuffledPayload/Event
MatchedRoleEnergyRandomActuator
MLPAnalogGeometryMaintenance
```

## 12.4 P3 gate

$$
\Delta CouplingR^2\ge 0.02,
$$

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio\le -0.01,
$$

$$
control\_gap\ge 0.005,
$$

$$
logit\_max\_abs\_drift\le 0.05.
$$

## 12.5 P4 short-run gate

P4 运行设置：

```text
base = B320-current diagnostic anchor and best label-free candidate if available
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 3 or 5
functional_event_interval = 24,48
```

P4 通过标准：

$$
AUCtime_{func}\le AUCtime_{base},
$$

$$
AUCstep_{func}\le AUCstep_{base},
$$

$$
Acc_{func}\ge Acc_{base}-0.003,
$$

$$
ECE_{func}\le ECE_{base}+0.02,
$$

$$
CEp99_{func}\le CEp99_{base}+0.05,
$$

$$
T_{amortized,func}/T_{base}\le 1.05.
$$

并且 Line C 改善和 controls 全部满足。

## 12.6 不满足条件时 Codex 必须继续尝试

```text
如果 P3 pass 但 P4 fail：
  做 P3-to-P4 persistence autopsy；
  检查 effect half-life；
  降低 event frequency；
  不做 lambda 小网格优先。

如果 P4 AUC-step 坏：
  机制坏，回到 Line C/T/I。

如果只 AUC-time 坏：
  检查 overhead / implementation。

如果 control gap fail：
  停止 official claim，只保留 diagnostic。
```

---

# 13. Line D：Classic No-BSpline Portfolio 并行支线

## 13.1 当前状态

```text
BSpline: frozen
Rational: TaskBlocked / GeometryBlocked
Chebyshev: TaskBlocked
Wavelet: TaskBlocked
RBF/FastKAN: ExpressionBlocked
Fourier: ExpressionBlocked
```

## 13.2 v12.21 只允许的新 hypothesis

```text
Rational:
  denominator-safe + LineC coupling repair；
  group diversity / rational tangent metric；
  不继续旧 gain/cap 小修。

Chebyshev:
  degree-energy damping + task trajectory repair；
  不单纯提高 degree。

Wavelet:
  local support / scale diversity task-stable repair；
  不做 Morlet heavy path。

RBF/FastKAN:
  compact capacity expression repair；
  不回 dense RBF。

Fourier:
  low-frequency expression repair；
  不做 high-frequency noise-heavy path。
```

## 13.3 执行规则

Line D 不得抢占 Line A/C/T/I 主预算。只有在以下条件满足时并行运行：

```text
1. 有空闲 GPU；
2. 新 hypothesis 明确；
3. 不是旧 candidate 重跑；
4. 不影响 v12.21 functional main artifacts。
```

## 13.4 必须记录

```text
v1221_classic_family_status.csv
v1221_classic_family_new_hypothesis.csv
v1221_classic_family_linec.csv
v1221_classic_family_failure_table.csv
```

每个 family 输出：

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
Frozen
RejectedForThisVersion
```

不得输出“未跑所以失败”。

---

# 14. 总 artifact contract

v12.21 必须落盘：

```text
v1221_route_decision.json
v1221_continuation_manifest.csv
v1221_provenance_audit.csv
v1221_code_review_manifest.csv
v1221_implementation_readback.md

v1221_label_free_signal_frame.csv
v1221_label_free_linec.csv
v1221_label_free_failure_decomposition.csv

v1221_linec_deployable_targets.csv
v1221_target_sign_sanity.csv
v1221_soft_target_regression.csv
v1221_target_family_ablation.csv

v1221_visibility_features.csv
v1221_visibility_scores.csv
v1221_visibility_leaveout.csv
v1221_visibility_sanity.csv
v1221_t1b_optimizer_update_features.csv

v1221_actuator_response_dictionary.csv
v1221_actuator_controls.csv
v1221_actuator_safety.csv

v1221_functional_p3.csv
v1221_functional_controls.csv
v1221_functional_p4_short.csv
v1221_p4_event_log.csv
v1221_p4_linec_trajectory.csv

v1221_classic_family_status.csv
v1221_classic_family_new_hypothesis.csv
v1221_classic_family_linec.csv

v1221_no_go_boundary.md
v1221_next_hypothesis_generator.md
```

如果缺少 `v1221_no_go_boundary.md` 或 `v1221_next_hypothesis_generator.md`，即使 continuation 完成，也不能写 final_stop_allowed。

---

# 15. 必须生成的可视化

```text
fig_v1221_linea_task_linec_pareto.svg
fig_v1221_label_free_frame_role_energy.svg
fig_v1221_auc_step_time_by_label_free_candidate.svg

fig_v1221_GLA_correlation_matrix.svg
fig_v1221_target_sign_inversion_audit.svg
fig_v1221_soft_target_regression_residuals.svg

fig_v1221_visibility_auc_precision_recall.svg
fig_v1221_T1A_T1B_T2_comparison.svg
fig_v1221_leaveout_visibility_heatmap.svg
fig_v1221_support_distribution.svg

fig_v1221_actuator_movement_release_pareto.svg
fig_v1221_actuator_control_gap.svg
fig_v1221_role_response_heatmap.svg

fig_v1221_functional_p3_control_comparison.svg
fig_v1221_p4_val_loss_step_time.svg
fig_v1221_p4_linec_trajectory.svg

fig_v1221_classic_family_status.svg
```

---

# 16. 最终 route 判定

## Success A：Label-free base success

```text
label_free_official_pass_count >= 1
```

且通过 task + AUC + Line C。

## Success B：Deployable value source success

```text
C-T1 / G_LA target pass
or T1B visibility pass
```

且不使用 label/CE/validation/future。

## Success C：Actuator executor success

```text
actuator_safe_movement_rows > 0
actuator_release_audit_rows > 0
control_gap >= 0.005
```

## Success D：Functional P4 success

```text
P4_open = 1
P4_pass = 1
beats all controls = 1
```

## Success E：Strong no-go with next hypothesis generated

允许 no-go，但必须同时满足：

```text
continuation_missing_count = 0
no_go_boundary_written = 1
next_hypothesis_generator_written = 1
at least one new next-round hypothesis defined per failed line
```

否则不能 final stop。

---

# 17. v12.21 的优先执行顺序

并行执行，但优先级如下：

```text
Priority 1:
  Line R code/feature/sign/join audit。

Priority 2:
  Line T native T1B logging + sign/soft target sanity。

Priority 3:
  Line A label-free signal frame A30-A35。

Priority 4:
  Line C deployable target G_LA reset。

Priority 5:
  Line I stronger primitive actuator v2。

Priority 6:
  Line B functional candidate only if C/T/I gate opens。

Priority 7:
  Line D classic no-BSpline with new hypotheses only。
```

---

# 18. 这份计划的 justification

v12.20 的数据说明：

```text
1. Codex 已经执行 continuation，不是完全 fail-fast；
2. 但 continuation 没有产生新合法路径；
3. label-free signal frame 是当前 base claim 的核心缺口；
4. loss-agnostic value source 是 functional update 的核心缺口；
5. actuator executor 目前太弱；
6. classic family 本轮没有新进展，不应误写成失败或成功。
```

所以 v12.21 的核心不是“再跑更多同类候选”，而是：

$$
\boxed{
\text{让实验从 fail-closed continuation 升级为 fail-closed continue-and-generate。}
}
$$

也就是：

```text
不放宽成功标准；
不让 Codex 停在第一个失败；
不靠 CE-specific 或 dataset-specific trick；
必须继续生成下一层机制假设；
直到得到合法 success 或可审计 no-go boundary。
```
