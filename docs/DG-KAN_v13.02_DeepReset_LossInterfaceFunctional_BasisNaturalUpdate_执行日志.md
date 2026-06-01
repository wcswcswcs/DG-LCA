# DG-KAN v13.2 DeepReset LossInterfaceFunctional BasisNaturalUpdate 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志只记录实际执行过的命令与产物路径；不把 smoke/diagnostic/fallback row 写成 promotion。

## 1. 计划文件

```text
docs/DG-KAN_v13.2_DeepReset_LossInterfaceFunctional_BasisNaturalUpdate.md
```

核心执行目标：

```text
1. 从 v12.35 substrate artifact 重新筛 S1/S2。
2. 在真实 basis 参数上执行 loss-interface generic cotangent + basis-natural update。
3. 写出 code surface / basis param manifest / writeback trace / loss-interface audit / forbidden-information audit。
4. 跑 synthetic X1..X7、P3 controls、MLP analog、required manifest、figures、route。
5. blocker 后按计划尝试 metric/norm repair、candidate breadth、safety projection。
```

## 2. 新增 runner

新增文件：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
```

主要实现：

```text
1. v132_substrate_map.csv：读取 v12.35 basis substrate health，按 v13.2 S1/S2 gate 重算。
2. v132_basis_natural_p3.csv：真实模型参数上的 BN1/BN2/BN3/BN4/BN5 update 与 controls。
3. v132_functional_writeback_trace.csv：记录参数写回前后 sha256。
4. v132_loss_interface_audit.csv：记录 CE/Brier loss-interface 是否作为 generic output cotangent 使用。
5. v132_forbidden_information_audit.csv：记录 validation/test/CE-tail/dataset-name branch 等 forbidden source。
6. v132_mlp_analog_control.csv：MLP analog，不允许 KAN-specific claim。
7. v132_route_decision.json / required manifest / code review packet。
```

## 3. 语法检查

```bash
conda run -n kan python -m py_compile experiments/run_v132_loss_interface_basis_natural_update.py
```

结果：

```text
pass
```

## 4. smoke 1：发现 synthetic dim blocker

初始 smoke：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/smoke_v132 \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --synthetic-train-size 32 \
  --synthetic-val-size 24 \
  --checkpoint-steps 5 \
  --future-steps 5 \
  --loss-interfaces CE \
  --update-candidates BN1-NaturalDiag \
  --max-s1-per-family 1 \
  --batch-size 16 \
  --device cuda:0
```

失败：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x36 and 136x3)
```

原因：

```text
v12.35 S1 D-RAT substrate 包含 paircrossR136；synthetic_dim=8 只产生 36 个 pair features。
```

修复：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  synthetic_dim 默认改为 16，使 16*17/2 = 136，与 paircrossR136 匹配。
```

## 5. smoke 2：runner 可执行

重新执行同一 smoke 命令。

结果：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
required_artifact_missing_count = 0
substrate_s1_count = 11
basis_natural_p3_rows = 1
basis_natural_p3_pass_count = 0
synthetic_rows = 1
full_basis_param_update_rows = 1
```

解释：

```text
smoke 只证明链路可运行与 writeback trace 可生成，不允许 promotion。
```

## 6. 初始 official

执行：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN1-NaturalDiag,BN2-SNROnly,BN3-DampedNatural \
  --max-s1-per-family 1 \
  --batch-size 32
```

结果：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
basis_natural_p3_rows = 42
basis_natural_p3_pass_count = 0
synthetic_rows = 7
synthetic_task_success_count = 0
full_basis_param_update_rows = 42
required_artifact_missing_count = 0
```

## 7. fallback 1：metric/norm + future optimizer sensitivity

触发原因：

```text
P3=0；failure table 建议 inspect SNR/metric/safety，并尝试 stronger basis metric or lower update norm。
```

执行：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_metric_norm_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN1-NaturalDiag,BN2-SNROnly,BN3-DampedNatural \
  --max-s1-per-family 1 \
  --batch-size 32 \
  --functional-eta 0.005 \
  --metric-rho 0.01 \
  --max-update-norm-ratio 0.005 \
  --future-lr 0.003
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_pass_count = 0
synthetic_task_success_count = 0
```

## 8. fallback 2：S1 candidate breadth

触发原因：

```text
metric/norm fallback 仍 P3=0；检查是否只是 D-RAT26 单一 substrate 问题。
```

执行：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_candidate_breadth_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN2-SNROnly \
  --max-s1-per-family 3 \
  --batch-size 32
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_rows = 42
basis_natural_p3_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
```

## 9. fallback 3：label-free logit-energy safety projection

触发原因：

```text
candidate breadth 中部分 rows 已有 source_vs_best_control > 0.005 与 CouplingR2 gain，
但 NoiseSignalLeak / RealSignalReservoirRatio 变坏，P3 被审计约束拒绝。
```

代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  新增 BN4-LogitEnergyOrthogonalNatural。
  新增 BN5-LogitEnergyOrthogonalSNR。
  在 train-stream logits 上计算 logit energy gradient。
  若 functional direction 一阶增加 logit energy，则投掉该分量。
```

合法性：

```text
1. 不改变 P3 gate。
2. 不使用 validation/test/future outcome。
3. 不使用 CEp99/NLL/ECE/tail 作为方向源。
4. safety projection 只读 train-stream current logits 和 basis params。
```

执行：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_energy_projection_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN4-LogitEnergyOrthogonalNatural,BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32
```

结果：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
required_artifact_missing_count = 0
```

## 10. final official rerun

将已通过 blocker 修复的 BN4/BN5 配置作为最终 official 重跑：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN4-LogitEnergyOrthogonalNatural,BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32
```

最终结果：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 84
provenance_violation_count = 0
```

## 11. final artifact 目录

```text
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/
```

主要文件：

```text
v132_route_decision.json
v132_required_artifact_manifest.csv
v132_code_surface.csv
v132_basis_param_manifest.csv
v132_functional_writeback_trace.csv
v132_loss_interface_audit.csv
v132_forbidden_information_audit.csv
v132_substrate_map.csv
v132_gradient_snr_histogram.csv
v132_basis_metric_condition.csv
v132_safety_projection.csv
v132_basis_natural_p3.csv
v132_synthetic_mechanism_proof.csv
v132_real_short_run_repair.csv
v132_mlp_analog_control.csv
v132_linec_audit.csv
v132_failure_table.csv
v132_next_hypothesis_queue.md
v132_code_review_packet.zip
```

Required manifest：

```text
rows = 30
missing required rows = 0
```

## 12. 用户再次追问后的 Case C fallback：future optimizer sensitivity

触发原因：

```text
当前 final official 为 R3-P3OnlySyntheticFailed；
计划 Case C 要求在 P3 pass 但 synthetic fail 时检查 future probe 是否被 task optimizer 覆盖。
```

执行：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_energy_projection_low_future_lr_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN4-LogitEnergyOrthogonalNatural,BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32 \
  --future-lr 0.003 \
  --future-weight-decay 0.001
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_pass_count = 0
synthetic_task_success_count = 0
required_artifact_missing_count = 0
```

结论：

```text
低 future LR 没有扩大 synthetic coverage，反而消除了原有 X7 P3 pass。
```

## 13. 用户再次追问后的 Case C fallback：optimizer-state transport

触发原因：

```text
计划 Case C 还要求检查 optimizer-state transport；
此前 runner 只写回参数，没有给后续 AdamW warm-start optimizer state。
```

新增代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  train_model 新增 optimizer_state_update / optimizer_state_scale。
  run_future_case 新增 optimizer_state_transport / optimizer_state_scale。
  CLI 新增 --optimizer-state-transport 与 --optimizer-state-scale。
  writeback trace 新增 optimizer_state_transport / optimizer_state_scale。
```

实现说明：

```text
1. 不改变 loss，不修改 sampler/class weight/dataset branch。
2. 不使用 validation/test/future outcome 选择方向。
3. 只把 functional update 派生的方向作为 AdamW exp_avg warm-start。
4. exp_avg_sq 使用 unit second moment，使它是谨慎 warm-start，不是额外大步参数跳跃。
```

Smoke 命令：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/smoke_state_transport_v132 \
  --device cuda:0 \
  --synthetic-tasks X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 10 \
  --future-steps 20 \
  --loss-interfaces Brier \
  --update-candidates BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 1 \
  --batch-size 32 \
  --optimizer-state-transport \
  --optimizer-state-scale 0.20
```

初次 smoke blocker：

```text
RuntimeError: The size of tensor a (3) must match the size of tensor b (6)
```

修复：

```text
参数 membership 判断从 `p in params` 改为 `id(p) in param_ids`，避免 Tensor equality broadcast。
```

修复后 smoke：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_rows = 1
basis_natural_p3_pass_count = 0
full_basis_param_update_rows = 1
required_artifact_missing_count = 0
```

正式 state transport fallback 1：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_optimizer_state_transport_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32 \
  --optimizer-state-transport \
  --optimizer-state-scale 0.20
```

结果：

```text
route = R3-P3OnlySyntheticFailed
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
writeback_trace_rows = 42
optimizer_state_transport_rows = 42
```

正式 state transport fallback 2：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_optimizer_state_transport_scale1_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32 \
  --optimizer-state-transport \
  --optimizer-state-scale 1.0
```

结果：

```text
route = R3-P3OnlySyntheticFailed
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
writeback_trace_rows = 42
optimizer_state_transport_rows = 42
```

结论：

```text
optimizer-state transport 没有把 synthetic coverage 从 2/21 推进到 5/7。
```

## 14. 用户再次追问后：BN6 consensus clipped metric repair

用户再次要求未达成则继续。本次按计划 Case B/Case C 的“family-specific metric repair / update norm robustness”继续尝试，不改 gate、不用 validation/test/future outcome 选方向。

代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  新增 BN6-ConsensusClippedEnergyNatural。
  BN6 使用 train-stream per-example gradient 的 sign consensus mask。
  BN6 使用 sqrt(var)+rho diagonal metric，并对 raw natural direction 做 75% quantile clip。
  BN6 继承 logit-energy orthogonal safety projection。
  metric_repair_attempted 将 BN6 标记为 1。
```

审计说明：

```text
1. BN6 仍使用 loss-interface-generic output cotangent。
2. 不使用 CEp99/NLL/ECE/LineC 作为方向源。
3. 不使用 validation/test/future/query batch 或 dataset-name branch。
4. 仍写回真实 basis 参数，不是 readout-feature proxy。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v132_loss_interface_basis_natural_update.py
```

focused fallback 命令：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_consensus_metric_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN6-ConsensusClippedEnergyNatural \
  --max-s1-per-family 3 \
  --batch-size 32
```

结果：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
basis_natural_p3_rows = 42
basis_natural_p3_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
full_basis_param_update_rows = 42
required_artifact_missing_count = 0
provenance_violation_count = 0
```

最接近但不能 pass 的 rows：

| candidate | task | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | fail |
|---|---|---|---:|---:|---:|---:|---:|---|
| D-RAT27 | X7 | Brier | 0.010846398201265095 | 1.4404830932617188 | 0.002047896385192871 | 0.031162261962890625 | -2.790827751159668 | noise;reservoir |
| D-RAT26 | X7 | Brier | 0.0073690773573587 | 0.8988113403320312 | -0.0005670785903930664 | 0.061533451080322266 | -2.483287811279297 | reservoir |
| D-RAT27 | X4 | CE | 0.0028140443622677314 | 2.827362060546875 | 0.03775596618652344 | -0.13692188262939453 | -0.16125726699829102 | source;noise |

结论：

```text
BN6 consensus clipped metric 没有改善 official BN5 的结果。
它保持了部分 X7 source/CEp99 优势，但 NoiseSignalLeak 或 Reservoir 坏化导致 P3=0。
因此不替换 official_v132；official 仍以 BN4/BN5 的 R3-P3OnlySyntheticFailed 为最终可信结果。
```

## 15. 用户再次追问后：future-case update hyperparameter propagation blocker 修复

用户再次要求未达成则继续。本次重新审计 runner 时发现一个实现 blocker：

```text
run_future_case() 在 copied model 上重算 functional update 时，
硬编码 eta=0.02, rho=1e-3, max_norm_ratio=0.02。
```

影响：

```text
此前 metric/norm fallback 的 CLI 参数会进入初始 NaturalUpdate audit，
但不会真实进入 future probe 的 copied-model writeback。
因此“update norm / clipping fallback”需要修复后重跑才可信。
```

代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  NaturalUpdate dataclass 新增 eta / rho / max_norm_ratio。
  natural_update() 返回这些超参数。
  run_future_case() 重算 copied-model update 时使用 update.eta / update.rho / update.max_norm_ratio。
  SNROnlyControl 也使用 matched eta/rho/max_norm_ratio。
```

审计说明：

```text
1. 这是执行语义修复，不改变 gate。
2. 不使用 forbidden information。
3. 修复后 fallback CLI 参数才真实作用于 future probe writeback。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v132_loss_interface_basis_natural_update.py
```

修复后 metric/norm fallback 重跑：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_metric_norm_repaired_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN4-LogitEnergyOrthogonalNatural,BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32 \
  --functional-eta 0.005 \
  --metric-rho 0.01 \
  --max-update-norm-ratio 0.005
```

结果：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
full_basis_param_update_rows = 84
required_artifact_missing_count = 0
provenance_violation_count = 0
```

pass rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| D-RAT26 | X7 | BN5 | Brier | 0.007548595096722038 | 1.9736995697021484 | -0.0045899152755737305 | -0.13720464706420898 | -2.2259368896484375 |
| D-RAT27 | X7 | BN5 | Brier | 0.008000795437745678 | 4.89793586730957 | -0.0030603408813476562 | -0.41165685653686523 | -2.750546455383301 |

修复后 official 重跑：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN4-LogitEnergyOrthogonalNatural,BN5-LogitEnergyOrthogonalSNR \
  --max-s1-per-family 3 \
  --batch-size 32
```

official 结果：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
full_basis_param_update_rows = 84
required_artifact_missing_count = 0
provenance_violation_count = 0
```

official pass rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| D-RAT26 | X7 | BN5 | Brier | 0.011487269300395009 | 1.9736881256103516 | -0.004589974880218506 | -0.13720369338989258 | -2.2259368896484375 |
| D-RAT27 | X7 | BN5 | Brier | 0.009255758784707857 | 4.897928237915039 | -0.003060281276702881 | -0.41165590286254883 | -2.7505455017089844 |

结论：

```text
hyperparameter propagation blocker 已修复。
修复后重跑证明：metric/norm fallback 仍没有把 coverage 从 2/21 推进到 5/7。
最终 official route 仍为 R3-P3OnlySyntheticFailed。
```

## 16. 用户再次追问后：BN7 output-Jacobian basis metric fallback

用户再次要求未达成则继续。本次不继续调 BN4/BN5/BN6 局部尺度，而是补计划第 7 节公式中的真实 basis tangent metric：

```text
M_basis = E[J_theta(x)^T J_theta(x)] + rho I
```

代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  新增 BN7-JacobianDiagEnergyNatural。
  对 train-stream 的每个样本/输出维度计算真实参数 output-Jacobian diagonal。
  BN7 direction 使用 loss-interface cotangent gradient mu，但 metric 使用 label-free output-Jacobian diagonal。
  BN7 继续使用 SNR gate 与 logit-energy orthogonal safety projection。
  metric_repair_attempted 覆盖 BN7。
```

审计说明：

```text
1. J_theta metric 只来自 train-stream input 与 model output Jacobian，不读取 validation/test/future/query batch。
2. loss label 只通过 generic cotangent 进入 mu，不进入 metric。
3. CEp99/NLL/ECE/LineC 仍只作 audit/gate。
4. 仍写回真实 basis 参数。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v132_loss_interface_basis_natural_update.py
```

focused fallback 命令：

```bash
conda run -n kan python experiments/run_v132_loss_interface_basis_natural_update.py \
  --out-dir results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_jacobian_diag_metric_v132 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --checkpoint-steps 40 \
  --future-steps 50,200 \
  --loss-interfaces CE,Brier \
  --update-candidates BN7-JacobianDiagEnergyNatural \
  --max-s1-per-family 3 \
  --batch-size 32
```

结果：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
basis_natural_p3_rows = 42
basis_natural_p3_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
full_basis_param_update_rows = 42
required_artifact_missing_count = 0
provenance_violation_count = 0
metric_condition_min = 1.3862051963806152
metric_condition_max = 3.5969626903533936
```

最接近但不能 pass 的 rows：

| candidate | task | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | fail |
|---|---|---|---:|---:|---:|---:|---:|---|
| D-RAT26 | X7 | Brier | 0.0018138602472285992 | 4.893610000610352 | 0.028290987014770508 | -0.36788415908813477 | -2.9503769874572754 | source;noise |
| D-RAT28 | X6 | Brier | 0.0006062911758760239 | -152.47772979736328 | 0.03697788715362549 | 5.895464897155762 | 1.275080680847168 | source;coupling;noise;reservoir;cep99 |
| D-RAT28 | X2 | Brier | 0.00014374580432294226 | -3.7359256744384766 | -0.043040692806243896 | 0.5299921035766602 | 0.745448112487793 | source;coupling;reservoir;cep99 |

结论：

```text
BN7 更接近计划中的 label-free basis tangent metric，但没有打开 P3。
它把 metric condition 控制在较小范围，却显著降低 source_vs_best，说明当前 blocker 不是单纯 metric conditioning。
official_v132 不替换，最终可信 route 仍为 R3-P3OnlySyntheticFailed。
```

## 17. 用户再次追问后的 stop-contract 复核

用户再次要求未达成则继续。本次重新读取 v13.2 计划 stop rule 与 `v132_next_hypothesis_queue.md`。

复核命令：

```bash
sed -n '760,920p' docs/DG-KAN_v13.2_DeepReset_LossInterfaceFunctional_BasisNaturalUpdate.md
cat results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_next_hypothesis_queue.md
python - <<'PY'
import json,csv
from pathlib import Path
base=Path('results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132')
d=json.loads((base/'v132_route_decision.json').read_text())
for k in ['route','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_all_executed','required_artifact_missing_count','substrate_s1_count','substrate_s2_healthy_base_count','basis_natural_p3_pass_count','synthetic_task_success_count','synthetic_5of7_pass','real_short_run_pass_count','mlp_analog_pass_count','full_basis_param_update_rows','provenance_violation_count']:
    print(k, d.get(k))
rows=list(csv.DictReader((base/'v132_required_artifact_manifest.csv').open()))
print(len(rows), sum(int(r.get('missing','0')) for r in rows))
PY
```

当前 final route：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 84
provenance_violation_count = 0
manifest_rows = 30
missing_rows = 0
```

计划 stop rule 对照：

```text
1. 所有 S1 substrate family 都已完成 Line B/X/M 判定：当前 S1 只有 D-RAT，已完成。
2. hard_compute_budget_exhausted=1。
3. 每个 failure case 都有 no-go / next hypothesis。
4. 不属于不允许停止项：不是 readout-feature proxy，不是 response dictionary，synthetic 使用真实参数，P3 fail / synthetic fail 后已有 SNR/metric/safety/optimizer-state/norm/future-probe fallback。
```

`v132_next_hypothesis_queue.md` 对照：

```text
1. inspect per-example gradient SNR and basis metric conditioning: 已完成。
2. optimizer-state transport and tighter norm clipping: 已完成。
3. no S1 for non-RAT -> return to kernel/lifetime/task-health substrate design: 属于新的 substrate design 工作，不是当前 v13.2 functional runner 内可安全补完的局部修复。
```

结论：

```text
v13.2 仍没有达成 S5。
当前合法 route 仍为 R3-P3OnlySyntheticFailed。
本次不新增训练实验、不新增 CSV 指标、不修改 gate。
原因：计划内 functional fallback 已执行完；继续在 BN4/BN5/BN6/BN7 局部 family 内扩变体会变成低价值搜索。
我目前不确定如何在现有 v13.2 代码路径内安全实现新的 Non-RAT substrate design 或非对角 task-family metric，并保证不把 proxy 写成 success。
```
