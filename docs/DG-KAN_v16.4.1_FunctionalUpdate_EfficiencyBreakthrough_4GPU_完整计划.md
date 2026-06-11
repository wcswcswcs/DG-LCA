# DG-KAN v16.4.1：Functional Update 机制矩阵 + Basis Efficiency Breakthrough + 4GPU 动态并行完整计划

> 版本：v16.4.1 revised execution plan  
> 生成时间：2026-06-02  
> 核心修正：不再只“测效率真值表”，而是把 basis efficiency 作为必须推进的工程主线；不再只在 D-CHE/MLP 上跑 full functional matrix，其它 basis 也必须至少跑 low-budget functional smoke；不再让 4GPU 静态分工后空转，必须动态 queue 化。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed initialization；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为方向源；不新增 action bank / controller / reset route。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{最终得到比 ordinary AdamW/backprop 更好的模型。}
}
$$

这里的“更好”不是局部 source、不是某个 dataset/seed 正、也不是某个 diagnostic 指标好看，而是同时满足：

```text
1. base / substrate 效率接近同参数量 MLP；
2. functional update 的收益超过 AdamW / Random / NoOp / matched controls；
3. 如果 MLP-FU 也成功，要写成 generic training-dynamics insight，不能写成 KAN-specific；
4. 只有 KAN carrier 明显强于 MLP，才允许讨论 KAN-specific functional advantage；
5. 短期 tail / LineC / calibration / AUC debt 可以出现，但必须长期偿还；
6. functional positive 必须在 h800 / h1600 或 real-lite 上保留 source，并且 debt 恢复；
7. 最终 S5 不降低：9/9 real dataset-seed strict pass。
```

v16.3 的真实状态是：执行面比 v16.2 强，Line P efficiency census 开始补齐 same-param MLP 对照；D-CHE / MLP functional matrix 也继续执行。但科学结果仍是 no-go：

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
KAN_specific_advantage_rows = 0
```

当前不允许再说“只是缺一个小修”。真正问题是双重的：

$$
\boxed{
\text{functional update 没有形成 source-retaining debt recovery；}
\text{basis efficiency 也没有形成完整可用 envelope。}
}
$$

---

# 1. 我对过去几版计划的反思

## 1.1 我之前的问题

我之前的计划有三个明显缺陷。

第一，**把 functional update 的失败主要归结为某个机制没跑够**，于是不断写新机制名，但没有强制把每个 carrier 上的多套机制同时跑完。这导致每轮只得到一个局部 no-go，信息量太低。

第二，**把 basis efficiency 当成附属 census**。v16.3 开始测 forward / backward / update / memory，但它仍然更像读数，不是“修复任务”。如果 D-FOU forward 10x、LQ update 2.4x、D-CHE forward/step blocked，只记录原因还不够，必须进入 per-family kernel/update repair。

第三，**4GPU 没有变成真正动态调度**。有 manifest 不等于充分利用。必须有 runnable queue、fallback queue、idle violation、queue drain report，否则仍然会出现“计划并行、实际串行”。

## 1.2 这版 v16.4.1 的核心修正

v16.4.1 不再是：

```text
D-CHE / MLP full matrix + 其它 basis substrate-only + efficiency census。
```

而是：

```text
1. D-CHE / MLP：full multi-scheme functional matrix；
2. LQ / Rational / D-FOU / D-RBF / D-WAV：minimum functional smoke + substrate repair 同步跑；
3. Efficiency census：final route 的 hard prerequisite；
4. Basis efficiency repair：每个 basis 必须给出可执行的工程修复尝试，不只是读数；
5. 4GPU：dynamic runnable queue，不允许空卡等主线。
```

一句话：

$$
\boxed{
\text{functional update 与 basis efficiency 必须同时突破，}
\text{不能再用一个 no-go 掩盖另一个没解决。}
}
$$

---

# 2. 当前各线进展百分比

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / no-action audit | 99% | required / forbidden / no-action 审计成熟，不是 blocker |
| 4GPU 执行框架 | 85% | 有 manifest，但缺动态 queue / idle violation / drain report |
| Efficiency census | 30% | 有 phase timing rows，但 blocked basis 缺完整 phase 数值；还不是 hard prerequisite |
| D-CHE substrate | 85% | 当前最强 KAN carrier，但 functional + efficiency 都没闭合 |
| D-CHE functional dynamics | 18%-25% | 有局部 source，h800/h1600 不能保 source + debt recovery |
| MLP functional dynamics | 30%-40% | 当前最强 h800 source 来自 MLP，必须作为 active line |
| LQ reanchor | 25%-35% | near 9/9，但 reanchor gate 未开；不能 functional promotion |
| Rational monitor | 70%-80% substrate / 10% FU | 只做 monitor；不重启 reset/controller/action |
| D-FOU substrate/efficiency | 30%-40% | best non-D-CHE 线索；step/memory 局部可承受，但 forward 极慢 |
| D-RBF / FastKAN | 20%-30% | task-health 和效率都不稳；需要 active-center/width/local-K repair |
| D-WAV | 15%-20% | 弱线索；需要 sparse support/path repair，不宜大预算 |
| Dynamic geometry / debt recovery | 25% | 概念已修正，但没有 productive recovery candidate |
| Multi-scheme functional matrix | 30% | D-CHE/MLP 较完整；其它 carrier 不足 |
| Basis efficiency repair | 10%-15% | 仍以 census 为主，缺真实 per-family repair |
| Official S5 | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 23%-31% | 执行成熟，但科学 claim 仍弱 |

---

# 3. 最新实验结果的独立分析

## 3.1 Functional update 线

v16.3 最强 h800 source 来自 MLP，不是 KAN。该 row 是：

```text
B-M4a-MLP-AdamSubspaceProximal-alpha000
source_vs_best_control_h800 = 0.14581246508492363
source_retention_h800 = 0.8296399083402421
tail_recovery_rate_h800 = 0.01966376412104744
LineC_recovery_rate_h800 = 0.0
AUCtime_ratio_h800 = 1.0373429473020996
```

它说明 MLP functional 有 source 和 retention，但没有 tail/LineC recovery，因此不是 productive dynamics。它也说明：**MLP-FU 不能被降级成 control，必须是 active mechanism discovery line。**

D-CHE 最强 h800 source 仍没有打开 S2/S3。v16.2 / v16.3 的共性是：D-CHE 局部 source 有时不弱，但无法同时保 source、还 debt、打过 controls。h1600 也没有自然翻盘。

关键判断：

$$
\boxed{
\text{当前 functional update 的问题不是完全没有 source，}
\text{而是 source 不能成为长期 productive dynamics。}
}
$$

## 3.2 KAN-specific 判断

v16.3 的 Line M attribution 显示 KAN-specific advantage rows = 0。这说明当前不能把 D-CHE 或其它 KAN 的局部 source 写成 KAN-specific success。

因此 v16.4.1 必须把结论分成三类：

```text
Generic-FU success:
  MLP 和 KAN 都成功，或 MLP 更强。

KAN-carrier success:
  KAN 明显强于 MLP，且 controls 解释不了。

No-go:
  MLP/KAN 都不能形成 productive debt recovery。
```

## 3.3 LQ / Rational / all-basis

LQ 现在不是死线。它有 near=9/9 的 reanchor 线索，但 reanchor gate 未开，所以 functional promotion 不能打开。v16.4.1 要求 LQ 即使 reanchor 未完全打开，也必须执行 promotion-disabled late-attach smoke，用来判断是否存在 functional signal。

Rational 只做 monitor，不重启 reset/controller/action route。Rational 的价值是 sanity 和对照，不是当前主 functional carrier。

D-FOU / D-RBF / D-WAV 不能继续只做 substrate-only。D-FOU 已经多次成为 best non-D-CHE 线索，且 v16.3 efficiency census 显示它的 step/memory 可能并非最大 blocker；它必须进入 minimum functional smoke 和 forward-kernel repair。

---

# 4. 基函数效率现状与真正 blocker

## 4.1 已有的效率真值片段

当前已知的 phase-level 片段说明：

| carrier/path | batch | forward ratio | backward ratio | update ratio | memory ratio | step ratio | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP functional top2 | 128 | 0.935 | 0.623 | 0.829 | 0.997 | 0.930 | official efficiency pass |
| MLP functional top2 | 32 | 1.058 | 0.603 | 1.515 | 0.997 | 0.940 | update 慢，但 step 可承受 |
| MLP reference | 8/32/128 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | same-param baseline |
| D-FOU current | 32 | 10.207 | 1.890 | 1.276 | 1.001 | 1.291 | memory/step 可承受，但 forward 极慢 |
| LQ current/reanchored | 8 | 6.643 | 1.538 | 2.398 | 0.988 | 1.304 | memory/step 可承受，但 forward/update 慢 |

这张表的意义很大：

```text
1. MLP path 是效率健康的 active FU carrier。
2. D-FOU 的主要 blocker 可能是 forward basis eval，而不是 memory。
3. LQ 的主要 blocker 是 forward + optimizer update。
4. D-CHE blocked 不能只归咎 scientific no-go，也要拆 audit/readback/functional carrier runtime。
5. D-RBF/D-WAV 还缺完整 phase census，不能继续模糊判断。
```

## 4.2 basis efficiency 的本质问题

目前 basis efficiency 不是“都慢”这么简单，而是不同 family 的瓶颈不同：

```text
D-CHE:
  可能慢在 degree recurrence / functional carrier / horizon LineC readback / split-gradient 反向。

D-FOU:
  forward 10x，说明 sin/cos / band eval / phase mixing / high-frequency path 是主 blocker。

LQ:
  forward 6.6x + update 2.4x，说明 LQ frame/readout/update path 没有 fused 或 foreach 化。

D-RBF:
  likely active-center / width / gaussian eval / dense materialization / backward scatter blocker。

D-WAV:
  local support 本应 sparse，但当前 forward/backward/step 均不稳定，说明 support indexing / overlap / materialization 仍错。

Rational:
  monitor-only；若要恢复效率线，需要 denominator/derivative fused telemetry 和 branchless rational update，但不能重启 reset route。
```

---

# 5. v16.4.1 的总设计：两个硬主线同步推进

v16.4.1 不再把 functional update 和 basis efficiency 分开排队。两条硬主线必须同时推进。

## 主线 A：Functional Update 多方案矩阵

目标：判断 functional update 到底是 generic training dynamics、KAN carrier-specific，还是当前定义 no-go。

核心矩阵：

$$
\boxed{
\text{Carrier} \times \text{Mechanism} \times \text{Horizon} \times \text{Controls}
}
$$

## 主线 B：Basis Efficiency Breakthrough

目标：补齐每个 basis vs same-param MLP 的 forward / backward / update / memory 真值表，并对每个 family 做至少一个真实效率修复 attempt。

核心矩阵：

$$
\boxed{
\text{Basis family} \times \text{Phase-level profiler} \times \text{Family-specific repair} \times \text{Smoke validation}
}
$$

---

# 6. Carrier × mechanism functional matrix

## 6.1 Carrier 维度

v16.4.1 的 carrier 不分“高贵主线”和“低级附属”。每个都必须至少有 functional smoke。

| carrier | 必跑内容 |
|---|---|
| D-CHE | full M1-M8 functional matrix + full efficiency repair |
| MLP | full M1-M8 functional matrix；active mechanism discovery，不只是 control |
| LQ | R0-R10 reanchor + M1/M2/M4/M7 minimum smoke；即使 reanchor fail 也跑 promotion-disabled late attach |
| Rational | M1/M2/M3 minimum smoke + denominator/slope safety audit；不重启 reset/controller/action |
| D-FOU | substrate repair + M1/M2/M3/M4 minimum smoke |
| D-RBF/FastKAN | substrate repair + M1/M3/M4 minimum smoke |
| D-WAV | substrate repair + M1/M3 minimum smoke |

## 6.2 Mechanism 维度

每个 active carrier 至少按能力运行以下机制族。D-CHE/MLP 全量跑；LQ/Rational/D-FOU/RBF/WAV 低预算 smoke 跑子集。

### M1：Productive pulse + recovery

问题：短期坏 update 是否能通过后续训练动力学偿还。

记录：

```text
source_vs_best_control_h1/h20/h100/h800/h1600
source_retention_h800/h1600
tail_debt_peak / tail_debt_final
LineC_debt_peak / LineC_debt_final
calibration_debt_peak / final
AUCtime_ratio_h800/h1600
recovery_update_norm
recovery_to_pulse_norm_ratio
```

### M2：Split-consensus signal metric

问题：跨 train split 一致的 signal subspace 是否比普通 local update 更能 transfer。

记录：

```text
split_count
signal_to_noise_ratio
projection_retention
negative_eigen_fraction
effective_rank
source_vs_control
random_subspace_control_gap
```

### M3：PopRisk / drift-diffusion SNR

问题：coherent population signal 是否比局部 geometry proxy 更可靠。

公式：

$$
SNR_k = \frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}
$$

记录：

```text
snr_mean
snr_p90
snr_active_fraction
snr_update_norm
cos_snr_adam
source_retention
debt_recovery
```

### M4：Function-space proximal

问题：是否要先在 function/output space 求小位移，再映射回参数。

固定：

```text
alpha_grid = 0, 0.025, 0.05, 0.10, 0.20
```

不允许 adaptive search / controller。

### M5：Optimizer-aware aligned FU

问题：FU 是否因为和 gradient / momentum / second moment 冲突而失败。

记录：

```text
cos_fu_neg_grad
sign_agreement_fraction
anti_alignment_fraction
adam_v_scaled_norm
cautious_keep_fraction
mgup_weight_mean
curvature_clip_fraction
```

### M6：Carrier reparameterized FU

问题：source/hazard 是否因为 carrier 坐标错而绑死。

候选不是 action token，而是 carrier coordinate readback / projection。

### M7：Late attach / snapshot FU

问题：functional update 是否应该在某个训练阶段 attach，而不是全程使用。

阶段：

```text
early checkpoint
mid checkpoint
late checkpoint
```

### M8：Recovery-only deconfound

问题：收益是否来自 recovery 机制本身，而不是 functional update。

必须比较：

```text
FU + recovery
recovery-only
random pulse + same recovery
NoOp + same recovery
AdamW extra steps matched time
```

---

# 7. Basis efficiency breakthrough plan

## 7.1 Line P0：完整 efficiency census hard gate

v16.4.1 里，efficiency census 不再是“有就好”的工程证据，而是 final route 前置条件。

必须生成：

```text
v1641_efficiency_truth_table.csv
v1641_efficiency_phase_breakdown.csv
v1641_memory_phase_breakdown.csv
v1641_basis_efficiency_blocker_taxonomy.csv
v1641_same_param_mlp_manifest.csv
```

每个 basis / batch / method 必须记录：

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_total_ms
forward_peak_memory
backward_peak_memory
optimizer_state_memory
basis_activation_bytes
functional_state_bytes
audit_state_bytes
same_param_mlp_forward_ratio
same_param_mlp_backward_ratio
same_param_mlp_update_ratio
same_param_mlp_step_ratio
same_param_mlp_memory_ratio
```

如果以下任一缺失，则 final route invalid：

```text
1. same-param MLP reference missing；
2. any active basis missing forward/backward/update/memory phase；
3. audit cost not separated from training cost；
4. blocked rows lack exact blocker reason；
5. GPU utilization missing。
```

## 7.2 D-CHE efficiency repair

当前判断：D-CHE 是 strong carrier，但 functional path 被 forward/step 和 audit/readback 污染。

必做修复：

```text
CHE-E1: separate training cost from LineC/horizon readback cost。
CHE-E2: degree recurrence cache / vectorized recurrence。
CHE-E3: low-degree active bank only smoke。
CHE-E4: high-degree late-enable with no extra audit in timing path。
CHE-E5: fused degree-role projection update。
CHE-E6: no-materialize degree energy readback。
```

必须判断：

```text
D-CHE slow because basis eval?
D-CHE slow because split-gradient functional path?
D-CHE slow because LineC readback?
D-CHE slow because optimizer update?
```

## 7.3 D-FOU efficiency repair

当前判断：D-FOU batch32 step/memory 局部可承受，但 forward 约 10x，是明显 forward bottleneck。

必做修复：

```text
FOU-E1: low-frequency-only forward recurrence。
FOU-E2: sin/cos precompute within fixed batch shape。
FOU-E3: fused band eval for K small。
FOU-E4: phase-stable band mix without high-frequency materialization。
FOU-E5: high-frequency quarantine smoke。
FOU-E6: compare torch trig vs recurrence vs table lookup。
```

成功判断：

```text
forward_ratio <= 2.0 exploration
step_ratio <= 1.5 exploration
memory_ratio <= 1.25 exploration
```

如果 forward 仍 >5x，D-FOU 不进入 full FU，只保留 low-budget smoke。

## 7.4 LQ efficiency repair

当前判断：LQ 不是显存爆，主要是 forward 6.6x 与 update 2.4x。

必做修复：

```text
LQ-E1: optimizer update phase decomposition。
LQ-E2: persistent AdamW / foreach update for frame/readout params。
LQ-E3: fused LQ projection update。
LQ-E4: fixed-frame late attach smoke，避免每步高成本 frame update。
LQ-E5: output-scale reanchor with exact same-param MLP reference。
```

判断：

```text
如果 update_ratio > 1.75，则先修 update path；
如果 forward_ratio > 3.0，则先修 frame/projection forward；
如果 reanchor gate 未开，但 near >= 8/9，仍跑 promotion-disabled late attach smoke。
```

## 7.5 D-RBF / FastKAN efficiency repair

当前判断：RBF/FastKAN workspace 有线索，但 task-health 和 efficiency 都不稳。

必做修复：

```text
RBF-E1: active-center occupancy audit。
RBF-E2: width condition guard。
RBF-E3: local K4 no-dense materialization。
RBF-E4: gaussian eval approximation / table lookup smoke。
RBF-E5: center/update separation。
```

## 7.6 D-WAV efficiency repair

当前判断：理论上 local support 应该 sparse，但当前路径没有体现稀疏优势。

必做修复：

```text
WAV-E1: triangular support index-only forward。
WAV-E2: support overlap damping readback。
WAV-E3: local-tail coverage audit only, not direction。
WAV-E4: sparse support backward without dense basis materialization。
```

## 7.7 Rational efficiency monitor

Rational 不重启 reset route，但必须做 efficiency monitor：

```text
RAT-E1: denominator/derivative telemetry cost。
RAT-E2: branchless denominator safety readback。
RAT-E3: rational function eval vs MLP timing。
RAT-E4: optimizer-state cost / update cost。
```

---

# 8. 4GPU dynamic scheduling

服务器有 4 张 GPU。v16.4.1 必须充分利用，不允许静态分配后空转。

## 8.1 必须生成调度文件

```text
v1641_runnable_queue.csv
v1641_gpu_assignment_manifest.csv
v1641_gpu_utilization_dashboard.csv
v1641_idle_violation.csv
v1641_deferred_items.csv
v1641_queue_drain_report.csv
```

## 8.2 GPU 初始分工

```text
GPU0:
  D-CHE full functional matrix + D-CHE efficiency repair。

GPU1:
  MLP full functional matrix + MLP active FU + same-param baseline。

GPU2:
  LQ reanchor/smoke + Rational monitor/smoke + LQ/Rational efficiency repair。

GPU3:
  D-FOU / D-RBF / D-WAV substrate repair + minimum functional smoke + all-basis efficiency repair。
```

## 8.3 动态补位规则

如果某 GPU 当前 primary queue 完成，则按顺序补：

```text
1. unmatched controls for positive-looking rows；
2. missing efficiency phase rows；
3. low-budget functional smoke for non-D-CHE basis；
4. h800/h1600 extension for top source-retaining candidates；
5. audit figures / route readback jobs。
```

硬规则：

```text
如果 runnable_queue 非空，但任一 GPU idle > 10 min，
则写入 v1641_idle_violation.csv，final route 不允许写 completed-no-go。
```

---

# 9. 成功标准

## 9.1 S1：执行覆盖成功

必须同时满足：

```text
Line P efficiency truth table complete；
D-CHE full matrix complete；
MLP full matrix complete；
LQ reanchor + smoke complete；
Rational monitor + smoke complete；
D-FOU/RBF/WAV substrate + smoke complete；
matched controls complete；
4GPU queue drain report complete；
required / forbidden / no-action audit pass。
```

## 9.2 S2：weak productive dynamics

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
tail_recovery_rate_h800 >= 0.40 or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

## 9.3 S3：productive debt recovery

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.50
tail_recovery_rate_h800 >= 0.60
LineC_recovery_rate_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
random pulse + same recovery fails
recovery-only fails
NoOp overhead fails
```

## 9.4 S4：real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## 9.5 S5：official success，不降低

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
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

---

# 10. Failure taxonomy

任何 no-go 必须写明属于哪一类，不允许只写 “all failed”。

```text
F1-SourceAbsent:
  source_vs_best_control <= 0。

F2-SourceNotRetained:
  source h800 有，但 h1600 消失。

F3-DebtNotRecovered:
  tail / LineC / calibration / AUC debt 不恢复。

F4-ControlEquivalent:
  random / NoOp / AdamW / MLP / recovery-only explains result。

F5-KANSpecificAbsent:
  MLP >= KAN 或 KAN-specific advantage rows = 0。

F6-CarrierNotReanchored:
  LQ / D-FOU / RBF / WAV substrate/reanchor 不足。

F7-EfficiencyForwardBlocked:
  forward_ratio > gate。

F8-EfficiencyBackwardBlocked:
  backward_ratio > gate。

F9-EfficiencyUpdateBlocked:
  update_ratio > gate。

F10-MemoryBlocked:
  memory_ratio > gate。

F11-AuditCostPolluted:
  LineC / horizon readback cost 混入 training phase timing。

F12-QueueContractFail:
  4GPU idle while runnable queue nonempty。
```

---

# 11. Stop / continue 规则

## 11.1 不能 hard stop 的情况

```text
D-CHE fail；
MLP fail；
LQ reanchor fail；
Rational smoke fail；
D-FOU/RBF/WAV <6/9；
任一 mechanism fail；
h800 fail；
h1600 fail；
MLP/generic controls positive；
LineC/tail/AUC immediate fail；
overhead high；
random pulse explains result；
efficiency blocked but smoke not executed。
```

这些只能进入 failure taxonomy / fallback ladder / exhaustion certificate / next hypothesis queue。

## 11.2 真正 hard stop

只有以下情况可以 hard stop：

```text
required artifact missing；
forbidden information violation；
no-action-search violation；
direction uses validation/test/future/query；
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier；
dataset-name branch；
seed-specific scale；
action token / controller / action bank / reset route；
fake / proxy / CPU offload。
```

---

# 12. 必须可视化

## 12.1 Functional dynamics

```text
source_vs_control_curve_h1_h20_h100_h800_h1600.svg
source_retention_curve.svg
tail_debt_recovery_curve.svg
linec_debt_recovery_curve.svg
auc_debt_curve.svg
carrier_mechanism_heatmap.svg
kan_vs_mlp_attribution_matrix.svg
matched_control_explainability_heatmap.svg
```

## 12.2 Efficiency

```text
efficiency_forward_ratio_by_basis.svg
efficiency_backward_ratio_by_basis.svg
efficiency_update_ratio_by_basis.svg
efficiency_memory_ratio_by_basis.svg
phase_time_stacked_bar_by_basis.svg
memory_phase_stacked_bar_by_basis.svg
basis_efficiency_pareto_step_vs_memory.svg
functional_overhead_vs_source_scatter.svg
```

## 12.3 4GPU execution

```text
gpu_utilization_timeline.svg
queue_depth_over_time.svg
idle_violation_timeline.svg
job_completion_gantt.svg
```

---

# 13. Codex 必须写入的实现复盘

每个 line 都必须在复盘里写实现思路，避免核心代码不审查：

```text
1. 本轮复用哪些旧 runner / helper；
2. 新增哪些 files；
3. 每个 functional method 的实际 update 公式在哪里；
4. 每个 efficiency profiler 的 timing phase 如何计时；
5. same-param MLP 如何匹配参数量；
6. direction provenance 如何保证不使用 forbidden source；
7. 哪些 rows 是 training，哪些 rows 是 audit/readback；
8. 哪些 jobs deferred，为什么 deferred；
9. 哪些 GPU idle，为什么 idle；
10. 哪些 result 可以支持 route，哪些只支持 engineering insight。
```

---

# 14. 最终判断逻辑

v16.4.1 的 route 必须按以下优先级：

```text
R0-ArtifactOrForbiddenViolation
  artifact / provenance / no-action audit fail。

R1-EfficiencyCensusIncomplete
  efficiency truth table 不完整。

R2-QueueContractFail
  4GPU runnable queue 未充分执行。

R3-FunctionalSourceAbsent
  all carrier all mechanism source absent。

R4-FunctionalSourceNotRetained
  h800 source 有，但 h1600 不保留。

R5-DebtRecoveryFail
  source 有，但 tail/LineC/calibration/AUC debt 不恢复。

R6-ControlEquivalentOrGenericOnly
  controls / MLP 解释 positive。

R7-BasisEfficiencyBlocked
  promising carrier 被 forward/backward/update/memory 阻断。

R8-CarrierMatrixPartialPositive
  有 weak signal，但不足 S4/S5。

S4-RealTransferExplorationPositive
  real-lite >=6/9，但 official S5 未达。

S5-OfficialFunctionalSuccess
  9/9 strict pass。
```

---

# 15. 一句话总结

v16.4.1 的本质不是“再写一个 functional update 方法”，而是：

$$
\boxed{
\text{用 4GPU 动态队列同时裁决 functional 机制与 basis efficiency，}
\text{并且不允许任何一条线用不完整证据草草 no-go。}
}
$$

如果 v16.4.1 仍然没有 S2/S3，但 efficiency truth table 完整、4GPU queue drain 完整、每个 basis 至少 smoke 完整，那么我们才有资格说：当前 train-stream FU family 需要理论级重构，而不是继续在实现层面猜。

