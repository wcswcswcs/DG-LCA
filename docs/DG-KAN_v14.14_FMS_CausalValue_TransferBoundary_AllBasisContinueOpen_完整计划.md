# DG-KAN v14.14：FMS Causal Value 重建、Transfer Boundary 训练与 All-Basis Continue-Open 完整计划

> 版本：v14.14 execution plan  
> 生成时间：2026-05-30 Asia/Singapore  
> 基于：v14.13 `TransferMechanismClarification / FunctionalContinueOpen / AllBasisParallel` 真实结果复盘；v14.12.1 / v14.11 / v14.10 / v14.9 历史边界；v9 系列 action-search 教训  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 总原则：functional update 继续作为突破口；promotion fail-closed；exploration continue-open；不新增 action bank / controller / reset route；不使用 validation/test/future/query；不使用 LineC / CEp99 / NLL / ECE / AUCtime 作为方向源；不按 dataset/seed 调参；不把 weak proxy、real-lite、substrate eligibility、control-equivalent row 写成 success。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个局部好看的 update rule。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN / no active B-spline base 上，}
\text{通过 functional update 得到比 ordinary backprop / AdamW 更好的训练过程与模型。}
}
$$

最终 claim 必须同时满足：

```text
1. 表达力不打折，最好强于同规模 MLP；
2. forward/backward/step/memory 与 MLP 可比；
3. 收敛轨迹更健康，AUC-step / AUC-time 不输；
4. 几何更好：train-probe coupling、signal/reservoir/noise、tail/calibration 不坏；
5. functional update 的收益必须独立于 AdamW / Random / NoOp / SNR-only / generic optimizer-state controls；
6. functional direction 必须 loss-interface-generic，不能针对 CE / CEp99 / ECE / LineC / AUCtime 设计。
```

最终要证明的是：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{same PureKAN base + ordinary AdamW / backprop controls}
}
$$

而不是只证明：

```text
PureKAN base 单独不错；
某个 synthetic S3 过；
某个 real-lite row positive；
某个 geometry diagnostic 上升；
某个 control-equivalent update 不坏。
```

## 0.2 当前最新状态：v14.13 后

v14.13 的真实结果可以压缩成一句话：

$$
\boxed{
\text{E3 找到了弱可预测 transfer signal，}
\text{但 V3 证明当前 FMS effect 主要是 ControlEquivalent；}
\text{F3 real-lite 只有 1/9；All-basis 仍 blocked。}
}
$$

关键结果：

```text
Line E3:
  best_feature = degree_projection_rejection_fraction
  best_auc_mean = 0.75
  best_leaveout_auc_min = 0.7586206896551724
  exploration_gate_pass = 1
  promotion_enabling_gate_pass = 0

Line V3:
  breakpoint_coverage = 1
  ambiguous_rows_fraction = 0.0
  dominant_breakpoint = V3-B6-ControlEquivalent

Line F3:
  f3_real_lite_pass_count = 1/9
  f3_mean_source_vs_best_control = -0.032626233498255414
  f3_source_fail_count = 40
  f3_auctime_fail_count = 41
  f3_tail_fail_count = 47
  f3_linec_fail_count = 18

Line D:
  best_non_dche_family = D-FOU
  best_non_dche_dataset_seed_pass_count = 2/9
  line_d_official_fms_eligible_family_count = 0

Final:
  route = R4-AllBasisSubstrateBlocked
  minimum_success = S3-DCHESyntheticFMSPass
  official_s5_reached = 0
  promotion_allowed = 0
  required_artifact_missing_count = 0
  forbidden_information_violation_count = 0
  no_action_search_violation_count = 0
```

v14.13 的进展不是能力成功，而是定位：

```text
1. train-stream / synthetic telemetry 里确实存在 transfer observability 线索；
2. 但当前 FMS-M1..M5 没有把该线索转化成 causal FMS value；
3. 主要断点不是 ambiguous、不是 artifact 缺失，而是 ControlEquivalent；
4. D-CHE 仍是唯一 functional-eligible carrier；D-FOU/RBF/WAV 未达到 >=6/9 exploration substrate gate；
5. 继续在 v14.13 内扩 FMS-M6/M7、F-CHE8/F-CHE9、action/controller/reset，都会重蹈 v9/v12 的 action-search 错误。
```

## 0.3 各线进展百分比

| 线 | 当前完成度 | 解释 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | required manifest、forbidden audit、no-action-search audit clean |
| Historical FHQ / B320-current | 85% frozen | 历史强 anchor；label-informed init 禁用，不能 official |
| Line C 几何审计 | 88% | 可稳定审计，不能做方向源 |
| PopRisk / FMS infrastructure | 94% | per-example gradient、persistent state、streaming memory、projection telemetry 成熟 |
| D-CHE substrate | 85% | 9/9 substrate eligibility，当前最强 Non-RAT carrier |
| D-CHE-FMS synthetic | 55%-60% | best synthetic 5/7，S3 历史成立 |
| Transfer observability | 25%-30% | E3 AUC 0.75 / leaveout 0.7586 是进展，但 promotion gate 未开 |
| FMS causal realization | 0%-5% | V3 主断点 ControlEquivalent；F3 real-lite 1/9 |
| D-CHE-FMS real transfer | 10%-15% | best real 2/9；v14.13 real-lite 1/9 |
| Rational substrate | 85% | 稳定，但 reset / optimizer-state route 已被 generic confound 打回 |
| Rational-FMS causal specificity | 10%-15% | v14.9 后降级为 monitor / no-regression |
| D-FOU substrate | 35%-45% | v14.9 历史 6/9，v14.13 best 2/9；需 cross-version reconciliation |
| D-RBF / FastKAN substrate | 30%-40% | 历史信号有，但当前 replay 不稳；task-health 不闭合 |
| D-WAV substrate | 15%-25% | 弱线索，低预算保留 |
| Non-RAT official FMS proof | 0%-5% | 只有 D-CHE eligible；其它 basis 不可 proof |
| Generic MLP-FMS / controls | 35%-40% | control 价值高；不能写成 KAN-specific success |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 38%-46% | 有 substrate + synthetic signal + weak observability，但 causal FMS value 未成立 |

---

# 1. v14.13 独立分析：到底有没有进展？

## 1.1 有进展：E3 证明 transfer observability 不是完全没有

v14.11 之前，我们只能说 synthetic gate 不可靠；v14.12.1 说明 train-stream proxy 有弱信号；v14.13 则进一步把 E3 提到：

$$
AUC_{mean}=0.75,
$$

$$
AUC_{leaveout,min}=0.7586.
$$

这说明 `degree_projection_rejection_fraction` 这个 feature 对 real-lite / transfer outcome 有较强诊断相关性。它不是 promotion-enabling，因为还没有证明因果性、还没有证明 FMS 可利用，也没有通过 full S5 gate；但它已经足以告诉我们：

$$
\boxed{
\text{transfer observability 不是完全不可见。}
}
$$

## 1.2 但不是能力进展：V3 说明当前 FMS 被 controls 解释

v14.13 的 V3 结果更关键：

```text
dominant_breakpoint = V3-B6-ControlEquivalent
ambiguous_rows_fraction = 0.0
```

这说明当前 FMS-M1..M5 的效果主要不是“FMS direction 独立有价值”，而是与 matched controls 等价。换句话说：

$$
\boxed{
\text{当前 FMS 机制没有证明 causal value，}
\text{只是某些 proxy 能看见一部分状态。}
}
$$

这正好呼应 v9 系列旧错：过去我们经常看到 oracle / actuatability / local positive / harmless-null，然后继续扩 action。现在必须避免这种错误。

## 1.3 F3 real-lite 失败说明“proxy-to-effect”没有闭合

F3 的结果是：

```text
real-lite pass = 1/9
mean source vs best control = -0.032626
source fail = 40
AUCtime fail = 41
tail fail = 47
LineC fail = 18
```

这不是“小阈值没调好”。如果一个 FMS 定义真正抓住了 transfer value，至少应该在 real-lite 上超过 4/9 exploration gate。现在 1/9 且 mean source 为负，说明当前 FMS-M1..M5 没有将 E3/V3 的诊断信息变成训练收益。

## 1.4 Line D 仍 blocked，不能把 all-basis 写成进展

Line D 的 best non-D-CHE 是 D-FOU，但只有 2/9。D-RBF、D-WAV 也未达到 >=6/9 exploration substrate gate。因此：

```text
D-CHE 仍是唯一 official functional carrier；
D-FOU/RBF/WAV 只能继续 substrate repair；
不允许进入 official FMS proof。
```

---

# 2. 当前真正卡在哪里？

当前 blocker 不是：

```text
D-CHE substrate 缺失；
FMS infrastructure 缺失；
LineC 主导失败；
artifact / manifest 缺失；
Codex 没继续；
差一个 F-CHE8 token；
差一个 controller；
差一个 reset trick。
```

真正 blocker 是：

$$
\boxed{
\text{FMS observability 与 FMS causal value 断裂。}
}
$$

具体说：

```text
1. E3 能看见某些 transfer-related state；
2. V3 说明当前 FMS effect 多数被 controls 解释；
3. F3 说明当前 FMS definitions 不能产生 real-lite gain；
4. Line D 没有替代 carrier；
5. 因此不能继续 method/action/token 搜索；
6. 必须先证明 FMS 的 causal value，而不只是 observability。
```

这是一类比“找好动作”更深的问题：

$$
\text{Can observe} \not\Rightarrow \text{Can act}.
$$

---

# 3. v9 / v12 旧错必须写进硬约束

v14.14 必须显式继承旧教训：

```text
1. actuatability pass 不等于 causality；
2. oracle frontier 不等于 legal precommit observability；
3. harmless-null / control-equivalent 不等于 useful functional；
4. dataset/seed-specific repair 是歧路；
5. action bank 只能做 upper-bound diagnostic，不能做搜索空间；
6. local positive row 不能驱动新 method；
7. 如果 matched controls 能解释，必须 fail-closed；
8. 如果 current method family 的 legal fallback 已覆盖，必须进入下一版机制重写，而不是在同一 runner 里临时补 token。
```

因此 v14.14 的首要制度是：

$$
\boxed{
\text{不能为了继续 functional update 而回到寻找好动作。}
}
$$

---

# 4. v14.14 总体目标

v14.14 的目标不是继续 FMS-M6/M7，也不是把 F3 real-lite 1/9 强行修到 4/9。它要回答三个根本问题：

## Q1：E3 feature 是真实 observability，还是 artifact？

`degree_projection_rejection_fraction` 的 AUC 很高，但它可能只是反映：

```text
1. projection 拒绝了坏 update；
2. FMS update 本身没有 value；
3. high rejection state 本来就更接近 NoOp / AdamW control；
4. 控制组同样能解释。
```

所以必须做 E3 deconfounding。

## Q2：当前 FMS direction 是否有独立 causal value？

需要比较：

$$
\Delta_{FMS-specific}
=
(FMS - matched\ control)
$$

在相同 active fraction、same norm、same degree rejection、same projection rejection 的条件下是否仍然大于 0。

## Q3：如果 current FMS direction 没有 causal value，functional update 的下一种形式是什么？

不是新增 action，而是重新定义 functional update 的作用形式：

```text
1. value source 仍来自 generic train-stream population-risk / FMS；
2. basis telemetry 只做 safety / trust region；
3. transfer-observable feature 只做 gating / abstention / boundary，不直接生成新方向；
4. 允许 continuous boundary FMS，不允许 event-token search。
```

---

# 5. 实验路线总览

v14.14 分为 7 条线：

```text
Line R: 代码 / provenance / no-action-search audit。
Line E4: E3 observability robustness and deconfounding。
Line Q: FMS causal value vs matched controls。
Line B: Boundary-only / value-preserving FMS mechanism reset。
Line F: Bounded D-CHE functional real-lite exploration。
Line D: All-basis substrate parallel and cross-version reconciliation。
Line Z: route / no-go / next hypothesis。
```

---

# 6. Line R：审计与硬约束

## 6.1 目标

确保 v14.14 不重蹈 v9/v12 旧错。

## 6.2 必须记录

```text
required_artifact_missing_count
forbidden_information_violation_count
no_action_search_violation_count
direction_uses_validation_test_future_query
direction_uses_linec_cep99_nll_ece_auctime
dataset_name_branch_used
seed_specific_scale_used
new_action_token_count
new_controller_executed
reset_route_used
fche_token_extension_count
fms_method_extension_count
```

## 6.3 硬停条件

只有以下情况 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction_uses_validation_test_future_query = 1
direction_uses_linec_cep99_nll_ece_auctime = 1
dataset_name_branch_used = 1
seed_specific_scale_used = 1
new_action_token_count > 0
new_controller_executed = 1
reset_route_used = 1
```

注意：

```text
F3 fail、Line D fail、E4 AUC below promotion gate、Q fail 都不是 hard stop；
它们触发预注册 fallback 或 no-go route。
```

---

# 7. Line E4：Observability robustness and deconfounding

## 7.1 目标

判断 E3 的 best feature `degree_projection_rejection_fraction` 是否真的有 transfer observability，而不是 artifact。

## 7.2 输入

读取：

```text
v14.10 D-CHE synthetic/real artifacts
v14.11 Line E/V artifacts
v14.12.1 Line E2/V2/F-Diag artifacts
v14.13 E3/V3/F3 artifacts
v14.9 D-CHE substrate and Line D artifacts
```

## 7.3 必须构造的 feature set

```text
E4-A: degree_projection_rejection_fraction
E4-B: value_retention_after_degree_projection
E4-C: cos_projected_vs_generic
E4-D: projection_rejection_fraction
E4-E: high_degree_fraction_delta
E4-F: degree_entropy_delta
E4-G: generic_fms_norm
E4-H: actual_update_norm
E4-I: split_agreement
E4-J: micro_horizon_loss_integral
E4-K: recovery_lag
E4-L: AdamW/FMS update cosine
E4-M: matched_random_same_norm response
E4-N: NoOp matched overhead response
```

## 7.4 必须做的 deconfounding

对每个 feature，必须报告：

```text
raw_auc_predict_real_lite_pass
leave_task_family_out_auc
leave_dataset_out_auc
leave_seed_out_auc
leave_loss_interface_out_auc
spearman_proxy_to_source
precision_at_top10pct
false_positive_rate_on_controls
control_matched_auc
incremental_auc_over_controls
```

## 7.5 Gate

Exploration gate：

$$
AUC_{mean} \ge 0.60
$$

and

$$
AUC_{leaveout,min} \ge 0.55.
$$

Promotion-enabling observability gate：

$$
AUC_{mean} \ge 0.70,
$$

$$
AUC_{leaveout,min} \ge 0.65,
$$

$$
Spearman \ge 0.30,
$$

$$
IncrementalAUC_{over\ controls} \ge 0.05.
$$

如果 raw AUC 高但 incremental AUC 不高，route：

```text
R2-ObservabilityExplainedByControls
```

---

# 8. Line Q：FMS causal value vs matched controls

## 8.1 目标

回答：当前 FMS event 是否有独立 causal value？

不是问：

```text
某个 row 是否 source positive；
某个 proxy 是否 AUC 高；
某个 diagnostic 是否看起来好。
```

而是问：

$$
\boxed{
\text{在严格 matched controls 下，FMS 是否仍然有 source / AUC / tail / LineC advantage？}
}
$$

## 8.2 必须比较的 groups

```text
G0: D-CHE AdamW baseline
G1: D-CHE FMS-M1..M5 existing methods only
G2: RandomMatchedNorm
G3: SameActiveFractionControl
G4: SameDegreeProjectionRejectionControl
G5: SameValueRetentionRandomDirection
G6: AdamWParallelDirectionControl
G7: NoOpMatchedOverhead
G8: GenericOptimizerStateControl
```

注意：G4/G5 是 **controls**，不是新 FMS method。

## 8.3 核心指标

```text
fms_specific_source_delta
fms_specific_auc_delta
fms_specific_tail_delta
fms_specific_linec_delta
control_equivalent_fraction
harmless_null_fraction
bad_event_fraction
source_vs_best_control
AUCtime_ratio_vs_best_control
CEp99_delta_vs_best_control
NLL_delta_vs_best_control
ECE_delta_vs_best_control
LineC_pass_gap_vs_control
```

定义：

$$
\Delta_{FMS-specific}
=
(FMS - BestMatchedControl).
$$

## 8.4 Gate

Causal exploration gate：

```text
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.60
bad_event_fraction <= 0.25
```

Strong causal gate：

```text
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.40
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass_rate >= 0.80
```

如果 V3-B6 ControlEquivalent 仍 dominates，route：

```text
R3-FMSDirectionControlEquivalent
```

并进入 Line B，不允许继续 FMS-M token 扩展。

---

# 9. Line B：Boundary-only / value-preserving FMS mechanism reset

## 9.1 触发条件

Line B 只在以下条件触发：

```text
Line E4 显示 observability 有信号；
但 Line Q 显示 current FMS direction control-equivalent；
且 no-action-search audit clean。
```

## 9.2 核心假设

如果 FMS direction 没有独立 causal value，那么 functional update 可能不是“新方向”，而是训练边界：

$$
\boxed{
\text{FMS should modulate when/how much to trust generic update,}
\text{not invent a separate direction.}
}
$$

## 9.3 只允许三类机制级定义

这些不是 action token，也不是 FMS-M6/M7。它们是机制假设：

### B1：Abstention Boundary FMS

如果 E4 proxy 显示 current update likely control-equivalent 或 harmful，则执行 NoOp / lower-plasticity，而不是强行提交 FMS。

方向仍然来自 generic train-stream gradient / FMS state，不从 audit metric 来。

### B2：Constraint-only Boundary FMS

FMS 不生成 value direction，只做 safety projection / trust region：

$$
\Delta\theta
=
\Pi_{D-CHE-safe}(\Delta\theta_{generic}).
$$

如果它优于 FMS direction，说明过去 basis telemetry 作为 value source 是错误的。

### B3：Continuous Low-Amplitude Boundary FMS

不做 event-level pulse，而是持续低幅调节 degree roles：

$$
\theta_{t+1}
=
\theta_t-\\eta_t T(c_t)g_t,
$$

其中 $T(c_t)$ 只来自 train-stream FMS state 和 D-CHE degree safety telemetry。

## 9.4 禁止事项

Line B 不允许：

```text
B4/B5/B6 临时新增；
按 dataset/seed 分支；
使用 CEp99/NLL/ECE/LineC/AUCtime 方向；
action bank / controller；
reset route；
strength/lambda/interval 小网格；
```

## 9.5 Gate

Boundary exploration gate：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction decreased by >= 20% relative to F3
```

Meaningful exploration gate：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.002
LineC_fail_count not increased
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
controls cannot explain
```

S5 remains strict 9/9。

---

# 10. Line F：Bounded D-CHE real-lite continuation

## 10.1 目标

继续 functional update，但只在机制级范围内，不做 token search。

## 10.2 Allowed definitions

```text
F-B1: Abstention Boundary FMS
F-B2: Constraint-only Boundary FMS
F-B3: Continuous Low-Amplitude Boundary FMS
```

不允许：

```text
F-CHE8/F-CHE9
FMS-M6/M7/M8
K-token
controller
action bank
reset route
```

## 10.3 记录字段

```text
method
control_family
dataset
seed
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
control_equivalent
bad_event
harmless_null
proxy_score
proxy_stratum
value_retention_after_projection
projection_rejection_fraction
cos_projected_vs_generic
active_fraction
step_time_ratio
memory_ratio
```

## 10.4 Real-lite gate

Exploration only：

```text
>=3/9: weak signal
>=4/9: meaningful real-lite
>=6/9: S4 exploration
```

Promotion：

```text
9/9 strict full real gate only。
```

---

# 11. Line D：All-basis parallel substrate continuation

## 11.1 当前状态

```text
D-CHE: 9/9 substrate, current FMS carrier.
D-FOU: v14.9 historical 6/9; v14.13 best 2/9.
D-RBF: v14.9 historical 6/9; recent replay weak.
D-WAV: weak, low-budget monitor.
Rational: stable but FMS causal reset route blocked.
B-spline: frozen.
```

## 11.2 目标

不能因为 D-CHE real-transfer fail 就停止 all-basis。也不能因为 all-basis fail 就停止 functional。两条线并行。

## 11.3 D-FOU / D-RBF cross-version reconciliation

必须比较：

```text
v14.9 6/9 candidates
v14.11 / v14.12.1 / v14.13 replay candidates
candidate id mismatch
gate mismatch
budget mismatch
LineC seed mismatch
finalizer route mismatch
true non-reproducibility
```

输出：

```text
reconciliation_status in {
  CandidateMismatch,
  GateMismatch,
  BudgetMismatch,
  ArtifactReplayMismatch,
  TrueNonReproducible,
  NeedsFreshHardening
}
```

## 11.4 Allowed substrate families

### D-FOU

```text
low-frequency identity residual
bandwise SNR warmup
phase-stable band mix
no-materialize lifetime
high-frequency quarantine
```

### D-RBF / FastKAN

```text
active center occupancy
width condition
compact bump no dense materialization
identity residual
Gaussian local K4 task-health
```

### D-WAV

```text
triangular support
scale occupancy
local support overlap damping
local-tail coverage audit
```

### D-CHE

```text
no-regression monitor
```

## 11.5 Gate

Substrate exploration：

```text
family_dataset_seed_pass_count >= 6/9
```

Official FMS eligibility：

```text
family_dataset_seed_pass_count = 9/9
```

No basis below 6/9 may enter official FMS proof.

---

# 12. Line C：Geometry audit only

Line C must record：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
Brier
margin_p10
source_vs_control
AUCtime_ratio
LineC_fail_reason
```

But Line C must not generate directions.

Allowed：

```text
audit
gate
failure taxonomy
visualization
```

Forbidden：

```text
LineC hard target direction
CE-tail direction
AUCtime direction
calibration-target direction
```

---

# 13. 必须生成的 artifact

## 13.1 CSV / JSON

```text
v1414_route_decision.json
v1414_required_artifact_manifest.csv
v1414_forbidden_information_audit.csv
v1414_no_action_search_audit.csv
v1414_e4_observability_deconfound.csv
v1414_e4_leaveout_summary.csv
v1414_q_fms_vs_matched_controls.csv
v1414_q_control_equivalence_summary.csv
v1414_b_boundary_fms_real_lite.csv
v1414_b_boundary_controls.csv
v1414_d_cross_version_reconciliation.csv
v1414_d_all_basis_substrate_hardening.csv
v1414_linec_tail_audit.csv
v1414_failure_taxonomy.csv
v1414_no_go_boundary.md
v1414_next_hypothesis_queue.md
v1414_code_review_packet.zip
```

## 13.2 Figures

```text
fig_v1414_gate_ladder.svg
fig_v1414_e4_roc_pr.svg
fig_v1414_leaveout_auc_heatmap.svg
fig_v1414_proxy_control_deconfound.svg
fig_v1414_v3_breakpoint_sankey.svg
fig_v1414_fms_vs_controls_effect.svg
fig_v1414_control_equivalence_by_stratum.svg
fig_v1414_proxy_to_effect_chain_waterfall.svg
fig_v1414_all_basis_substrate_matrix.svg
fig_v1414_failure_taxonomy_heatmap.svg
```

---

# 14. Stop / continue contract

## 14.1 Promotion fail-closed

Promotion only if：

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
provenance pass
code review pass
promotion_allowed = 1
```

## 14.2 Exploration continue-open

The following must not hard stop：

```text
E4 AUC < 0.70
E4 incremental AUC over controls < 0.05
Q FMS causal value fail
B boundary FMS real-lite <4/9
Line D non-D-CHE <6/9
LineC/tail/AUC fail
MLP/generic controls positive
```

These cases route to no-go/fallback, not hard stop.

## 14.3 Hard stop only for violations

Hard stop only if：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
validation/test/future/query used for direction
audit metric used for direction
dataset/seed branch used
action token/controller/reset route added
fake/proxy/CPU offload introduced
```

---

# 15. Codex 执行顺序

Codex must execute in order：

```text
1. Line R audit。
2. Line E4 observability robustness / deconfound。
3. Line Q FMS causal value vs matched controls。
4. If Q shows ControlEquivalent, execute Line B boundary-only / value-preserving mechanism reset。
5. Line F bounded D-CHE real-lite continuation。
6. Line D all-basis cross-version reconciliation + substrate hardening。
7. Line C audit。
8. Line Z route / no-go / next hypothesis。
```

Codex must not：

```text
1. Add F-CHE8/F-CHE9。
2. Add FMS-M6/M7/M8。
3. Add action token。
4. Start controller。
5. Use action bank。
6. Use reset route。
7. Use real fail pattern as direction。
8. Use LineC/CEp99/NLL/ECE/AUCtime as direction。
9. Write real-lite as promotion。
10. Write synthetic S3 as S4/S5。
11. Write D-CHE substrate eligibility as FMS success。
12. Write MLP/generic positive as KAN-specific success。
```

---

# 16. Route definitions

```text
S0-ExecutionCompleteNoPromotion:
  runner complete, no missing artifacts, no promotion.

S3-DCHESyntheticFMSPass:
  historical D-CHE synthetic 5/7 retained, but no real success.

S3b-ObservabilityExplorationPositive:
  E4 exploration AUC >=0.60 and leaveout min >=0.55.

R2-ObservabilityExplainedByControls:
  E4 raw AUC high but incremental AUC over controls <0.05.

R3-FMSDirectionControlEquivalent:
  Q shows ControlEquivalent dominates; current FMS direction has no causal value.

S4-lite-BoundaryExplorationPositive:
  Boundary FMS real-lite >=4/9, no promotion.

S4-ExplorationPositive:
  real-lite or bounded real >=6/9, controls not explanatory, no promotion.

R4-AllBasisSubstrateBlocked:
  no non-D-CHE basis >=6/9.

R5-CurrentFMSValueNoGo:
  E4 observability exists but Q/B/F all fail; current D-CHE FMS definition closed.

S5-OfficialFunctionalSuccess:
  full 9/9 strict real gate, controls fail, audit/code pass, promotion_allowed=1.
```

---

# 17. 本轮最重要的判断

v14.13 的结果不能被解释成“functional update 应该停止”。相反，它告诉我们：

$$
\boxed{
\text{functional update 的 observability 线索存在，}
\text{但当前 FMS direction 没有 causal value。}
}
$$

因此 v14.14 的突破口是：

```text
1. 证明 E3 observability 是否 independent of controls；
2. 证明 current FMS direction 是否有 causal value；
3. 如果没有，把 functional update 从 direction generator 改为 training boundary；
4. 同时继续 all-basis substrate repair；
5. promotion 严格关闭，exploration 保持开放。
```

一句话：

$$
\boxed{
\text{不要再寻找“好动作”；}
\text{要证明 functional update 的 causal value，}
\text{或者把它重定义成合法的训练边界。}
}
$$
