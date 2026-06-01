# DG-KAN v12.21 FailClosedContinue2 LabelFreeFunctional 实验结果复盘

本复盘对应计划文档 `docs/DG-KAN_v12.21_FailClosedContinue2_LabelFreeFunctional_独立分析与下一步计划.md`。结论只引用本轮落盘 artifact 中的实际数据；没有编造或补填缺失指标。

## 1. 本轮目标回读

v12.21 的核心不是继续给旧 projector/scorer 打补丁，而是把 v12.20 的 fail-closed continuation 升级成 continue-and-generate：

1. 找 label-free signal frame，验证能否替代 supervised trainprobe。
2. 重建 loss-agnostic / deployable 的 G_LA value source。
3. 原生记录 T1B optimizer-update features，并做 visibility v5。
4. 扩展 primitive-level actuator v2，并检查 safe movement / audit release / control gap。
5. 只有 C/T/I 至少一个合法路径打开，才进入 Line B functional P3/P4。
6. 即使没有 promotion，也必须完成 no-go boundary 和下一轮 hypothesis generator。

## 2. 本轮代码修改与审计结论

### 2.1 修改文件

- `experiments/run_v1218_b320_label_free_ablation.py`
  - 新增 A30-A35。
  - 新增 T1B optimizer-update native logging。
  - 新增 output covariance drift logging。
  - 修复 T1B logging 污染 step timing 的问题。
  - 修复 NaN summary 被当作 0 的问题。

- `dgkan/models/fc_purekan_primitives.py`
  - 新增 `augtangentp` 和 `rolebalancedp` projector token。
  - 注意：该文件本轮前已有未提交改动，本轮未回滚无关内容。

- `experiments/run_v1221_failclosed_continue2_label_free_functional.py`
  - 新增 v12.21 artifact/route/zip runner。
  - 修复 provenance audit 写入时机。

### 2.2 审计面结果

来自 `v1221_route_decision.json`：

```text
line_r_rows=16
line_r_required_missing=0
line_r_pass=1
required_artifact_missing_count=0
provenance_missing_count=0
join_duplicate_key_count=0
continuation_missing_count=0
final_stop_allowed=1
```

解释：CR0-CR14 没有缺失；CR15 也落盘。required artifact、provenance、join key、continuation 都没有缺口。

## 3. 最终 route

最终 route：

```text
route=R2-LabelFreeSignalFrameMissing
minimum_success=Success E
fail_reason=no legal promotion path passed; continuation complete and next hypotheses generated
```

这不是 functional 成功。它表示本轮完成了 fail-closed continuation 和 next hypothesis generation，但没有找到可 promotion 的 label-free base / value source / actuator / P4。

## 4. Line A：Label-Free Signal Frame A30-A35

运行规模：

```text
datasets=MNIST,Fashion-MNIST,KMNIST
seeds=0,1,2
epochs=3,8
train_size=1024
val/test=512
batch_size=128
LineC batch=64
sketch_dim=24
linea_rows=180
```

关键结果来自 `v1221_label_free_signal_frame.csv`：

| candidate | mean_delta_vs_A0 | worst_delta_vs_A0 | AUC_time_ratio_vs_mlp | LineC_all_pass | official_pass |
|---|---:|---:|---:|---:|---:|
| A1-noYForStats | -0.007161458333333333 | -0.037109375 | 1.1323156778204422 | 0 | 0 |
| A30-OptimizerObservableFrame | -0.020073784722222224 | -0.052734375 | 1.18804570478349 | 0 | 0 |
| A31-AugConsistencyTangentFrame | -0.014322916666666666 | -0.0546875 | 1.159040707110978 | 0 | 0 |
| A32-PersistentDriftFrame | -0.019097222222222224 | -0.046875 | 1.1837772594334148 | 0 | 0 |
| A33-RoleBalancedPrimitiveEnergyFrame | -0.7384982638888888 | -0.826171875 | blank due real NaN loss | 0 | 0 |
| A34-HybridA1OptimizerFrame | -0.020073784722222224 | -0.052734375 | 1.1918080493597845 | 0 | 0 |
| A35-HybridA1AugDriftFrame | -0.028211805555555556 | -0.068359375 | 1.2565560954725747 | 0 | 0 |

Gate 对照：

```text
mean_delta_vs_A0 >= -0.005
worst_delta_vs_A0 >= -0.015
AUCtime_ratio_vs_MLP <= 1.0
LineC_nontearing_all_pass = 1
```

结论：

- `label_free_official_pass_count=0`。
- 最好的 label-free 候选仍是 A1-noYForStats，但它 mean / worst / AUC-time / LineC 均未同时达标。
- A31 在 e8 Fashion-MNIST/KMNIST 单项上较接近，但跨全 official aggregation 不达标。
- A33 产生真实 NaN loss，已作为失败保留；修复点是防止 NaN 被 summary 写成 0，不是修掉失败。

## 5. Line C：G_LA Deployable Target

来自 `v1221_linec_deployable_targets.csv` 和 `v1221_route_decision.json`：

```text
linec_deployable_rows=126
g_la_spearman_noise=0.24805891375384642
g_la_spearman_reservoir=-0.08387894261778107
g_la_exploratory=1
g_la_official_pass=0
```

解释：

- `|rho(G_LA, Delta NoiseSignalLeak)| >= 0.20`，所以 exploratory diagnostic 有信号。
- 但 noise 方向是正相关，不满足 official 要求 `rho <= -0.30`。
- reservoir 相关只有 `-0.08387894261778107`，也远低于 official 要求。
- 因此 G_LA 不能作为 v12.21 的 functional direction source，只能作为 diagnostic。

Sanity block 已执行：

- sign inversion audit：`v1221_target_sign_sanity.csv`
- top-k/bottom-k audit：同表中 top/bottom precision/recall 字段
- row join stability：`v1221_join_key_uniqueness.csv`
- soft target regression：`v1221_soft_target_regression.csv`
- NaN/Inf/constant audit：soft target regression 表内记录

## 6. Line T：Visibility v5 和 T1B 原生日志

来自 `v1221_visibility_scores.csv`：

| tier | auc_joint | precision@k_joint | recall@k_joint | pass | failure |
|---|---:|---:|---:|---:|---|
| T1A | 0.636734693877551 | 0.1 | 0.1 | 0 | AUC_joint_gate_failed |
| T1B | 0.45714285714285713 | 0.0 | 0.0 | 0 | AUC_joint_gate_failed;precision_recall_gate_failed |

额外事实：

```text
T1B_native_logged_rows=108
T1B_native_logging_available=1
```

结论：

- v12.20 的 T1B unavailable 缺口本轮已补上：T1B 是原生训练 runner 记录，不再是 inherited artifact。
- T1A 达到 exploratory 附近：AUC 0.6367、precision/recall 0.1/0.1，但 official gate 要 `AUC_joint >= 0.65`，因此不能 promotion。
- T1B 原生可用但不可见：AUC 低于 0.5，precision/recall 为 0。按计划执行了 sign inversion/bottom-k sanity，不允许 promotion。

## 7. Line I：Primitive-Level Actuator v2

运行范围：

```text
datasets=MNIST,Fashion-MNIST,KMNIST
seed=0
families=I1-I9
budgets=0.0025,0.005,0.01,0.02
signed_direction=-1,+1
actuator_rows=216
```

来自 `v1221_actuator_safety.csv`：

```text
actuator_safe_movement_rows=0
actuator_release_audit_rows=3
actuator_control_resistant_rows=0
actuator_executor_success=0
```

三个 release-only 行：

| dataset | actuator | budget | sign | safe_movement | control_gap | logit_max_abs_drift |
|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | I2-quad-proj-local-rotation | 0.01 | -1 | 0 | -0.07951141148805618 | 0.9599435329437256 |
| Fashion-MNIST | I6-role-balanced-combined | 0.02 | -1 | 0 | -0.06166898459196091 | 0.9673179984092712 |
| Fashion-MNIST | I6-role-balanced-combined | 0.02 | +1 | 0 | 0.02369210124015808 | 0.8194356560707092 |

解释：

- release audit 有 3 行改善，但全部 logit drift 远超安全阈值 `<=0.05`，因此 safe movement=0。
- 没有一行同时满足 safe movement、release audit、control gap。
- 结论是 actuator 仍不是合法 executor；不能打开 functional P3/P4。

## 8. Line B：Functional Candidate / P3 / P4

来自 `v1221_functional_p3.csv` 与 route：

```text
functional_p3_rows=6
p4_open=0
p4_pass=0
```

原因：

- Line A official pass=0。
- G_LA official pass=0。
- T1B visibility pass=0。
- actuator executor success=0。

因此 Line B 按计划 fail-closed，没有执行 P4 short-run，也没有 functional success claim。

## 9. Line D：Classic No-BSpline 并行支线

落盘：

- `v1221_classic_family_status.csv`
- `v1221_classic_family_new_hypothesis.csv`
- `v1221_classic_family_linec.csv`
- `v1221_classic_family_failure_table.csv`

结果：

```text
classic_status_rows=5
classic_new_hypothesis_count=5
```

本轮没有把 classic family 记成“未跑所以失败”。状态是 `RejectedForThisVersion`，并为下一轮生成了非旧重跑 hypothesis：

- Rational：denominator-safe + LineC coupling repair + group diversity tangent metric。
- Chebyshev：degree-energy damping + task trajectory repair。
- Wavelet：local support / scale diversity task-stable repair。
- RBF/FastKAN：compact capacity expression repair，不回 dense RBF。
- Fourier：low-frequency expression repair，不做 high-frequency noise-heavy path。

## 10. 关键 blocker 与修复审计

### Blocker 1：T1B instrumentation 污染 AUC-time

现象：

- 初版 T1B native logging 把 snapshot/statistics 放在 task timing window 内，导致 B320 AUC-time ratio 被放大到 10x 以上。

修复：

- 在 `experiments/run_v1218_b320_label_free_ablation.py` 中把 `_trainable_param_snapshot()` 放到计时前，`optimizer_update_observable_stats()` 放到计时后。
- 重新运行全部 Line A official jobs。

结果：

- A1 AUC-time 从污染版约 10.75 修正为最终 `1.1323156778204422`。
- 仍不达标，所以没有改变 promotion 结论。

### Blocker 2：NaN 被 summary 当作 0

现象：

- A33 产生真实 `NLL=nan`。
- 旧 summary 用 `safe_float(... default=0)`，导致 `AUC_time_ratio_vs_mlp=0.0`，这是错误解释。

修复：

- 新增 `finite_float_or_none()`。
- 对 AUC ratio、NLL delta、ECE delta 等 summary 指标，仅在 numerator/denominator 都有限时写值。
- 重新运行全部 Line A official jobs。

结果：

- A33 的 AUC-time 现在为空，按 fail 处理。
- A33 的真实失败保留，不被伪装成 fast candidate。

### Blocker 3：Provenance audit 自身写入顺序

现象：

- 初版汇总里 `provenance_missing_count=2`，因为 route/provenance 自身在 finalization 前还未存在。

修复：

- 在 `run_v1221...py` 中将 `v1221_route_decision.json` 和 `v1221_provenance_audit.csv` 作为 finalization intended presence 审计。
- 重新运行汇总。

结果：

- 最终 `provenance_missing_count=0`。

## 11. 最终 artifact 与 zip

最终 zip：

```text
results/v12_21_failclosed_continue2_label_free_functional/official_continuation/v1221_code_review_packet.zip
sha256=e556a7ffed314752aa51390a7986f2425be1f3f412ffbf9facc53bf273420762
entries=67
```

最终完整性：

```text
required_artifact_missing_count=0
continuation_missing_count=0
provenance_missing_count=0
join_duplicate_key_count=0
line_r_required_missing=0
```

## 12. 总结

v12.21 完成了计划要求的 continuation 和 hypothesis generation，但没有打开 functional 路径：

- Label-free base：失败，官方通过数 0。
- G_LA value source：只有 exploratory signal，方向不满足 official。
- T1B：已原生记录，但 visibility 不达标。
- Actuator：有 release-only 行，但全部不安全；executor 失败。
- P4：未打开。
- No-go boundary 与 next hypothesis generator 已写入，满足 Success E。
- Completion-audit 补充后，route JSON 已显式写入 `final_stop_allowed=1`，并记录 `next_hypothesis_failed_lines_covered=1`、`next_hypothesis_lines=Line A,Line C,Line T,Line I,Line B,Line D`。

下一轮应优先处理两个机制缺口：

1. Label-free frame 不要再做 whole-projector replacement，改为 per-role low-rank residual injection。
2. G_LA 必须在训练时原生日志 exact output covariance / random-cotangent isotropy，并用 source-run leaveout 验证 sign-stable 后再作为 direction source。
