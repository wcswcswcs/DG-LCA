# DG-KAN v12.16 B320Locked ExplicitSignalReservoirFunctional 执行复盘

生成时间：`2026-05-24T02:42:34Z`

对应计划：

```text
docs/DG-KAN_v12.16_B320Locked_ExplicitSignalReservoirFunctional_实验结果分析与下一步计划.md
```

本轮严格保持 v12.16，不回退到 v12.15，也不把 P1/P2/P3 前置 gate 失败后的 B16/P4 写成成功。

## 1. 执行入口

新增 runner：

```text
experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
```

该 runner 实现：

```text
P0 B320 anchor monitor；
P1 Line C v2 sketch / target reconstruction；
P2 explicit noise/reservoir target calibration；
P3 fused primitive actuator response matrix；
P4 B16 functional candidates gated not-run；
P4 short-run gated not-open；
Line D status carry-forward；
loss_agnostic_audit / provenance_audit / failure_table / hash_manifest / route_decision。
```

新增结果目录：

```text
results/v12_16_b320locked_explicit_signal_reservoir_functional/smoke_mnist_seed0
results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10
results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

## 2. 编译检查

```bash
conda run -n kan python -m py_compile experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
```

结果：通过。

## 3. Smoke Run

目的：只检查接口、artifact contract 和 gate 路由，不作为 official 结论。

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py --probe-device cuda:0 --probe-datasets MNIST --probe-seeds 0 --probe-splits 1 --functional-batch-size 16 --windows 3,5 --actuator-windows 5 --probe-train-size 128 --probe-val-size 64 --probe-test-size 64 --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/smoke_mnist_seed0 --fresh
```

smoke route：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p2_pass = 0
p3_response_pass = 0
p4_open = 0
```

smoke 没有暴露代码 blocker。

## 4. Official 3x3 Run

official 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 32 --windows 3,5,10 --actuator-windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10 --fresh
```

official 输出目录：

```text
results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10
```

official route：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p2_pass = 0
p3_response_pass = 0
p4_open = 0
p3_survivor_count = 0
```

official artifact：

```text
v1216_route_decision.json
v1216_anchor_monitor.csv
v1216_linec_v2_sketch_targets.csv
v1216_explicit_target_calibration.csv
v1216_actuator_response_matrix.csv
v1216_functional_p3_candidates.csv
v1216_functional_p3_summary.csv
v1216_p4_short_run.csv
v1216_failure_table.csv
v1216_loss_agnostic_audit.csv
v1216_classic_family_status.csv
v1216_provenance_audit.csv
v1216_hash_manifest.json
fig_A_b320_anchor_monitor.svg
fig_C_sketch_target_estimator_oos.svg
fig_C_projector_stability_heatmap.svg
fig_C_noise_reservoir_null_distribution.svg
fig_C_role_conditioned_visibility.svg
fig_T_explicit_target_pred_actual.svg
fig_I_actuator_response_matrix.svg
fig_I_projector_angle_vs_release.svg
fig_B_p3_pareto_noise_reservoir.svg
fig_B_control_gap_by_candidate.svg
fig_B_fail_reason_waterfall.svg
fig_D_classic_family_status.svg
```

说明：本轮 SVG 是基于 route metrics 的审计占位图，用于满足 artifact contract 和快速复现检查；没有伪造额外曲线。

## 5. P1 修复尝试

official P1 失败后，按计划的 P1 修复方向继续尝试：

```text
1. 增加 sketch_dim；
2. 增加 / 保持 window ensemble = 3,5,10；
3. 保留 role-conditioned sketch；
4. 保留 persistent projector / stable-unstable output sketch；
5. 改变 output subspace rank cutoff；
6. 增大 functional batch，检查 batch 方差。
```

代码修改：

```text
experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
新增 --output-subspace-rank 参数；
output_subspaces 从固定 rank=3 改成 rank_limit 控制。
```

修复 run 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 64 --windows 3,5,10 --actuator-windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --sketch-dim 24 --output-subspace-rank 5 --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10 --fresh
```

修复 run 输出目录：

```text
results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

修复 run route：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p2_pass = 0
p3_response_pass = 0
p4_open = 0
p3_survivor_count = 0
```

## 6. P4/P4 Short-Run Gate

official 和 repair run 均没有打开 P4：

```text
v1216_functional_p3_summary.csv:
  method = P4_NOT_RUN
  candidate_id = B16-NOT-RUN
  p3_pass_rows = 0
  strong_promotion = 0
  weak_promotion = 0
  fail_reason = P1_estimator_gate_failed;P2_explicit_target_calibration_failed;P3_actuator_response_gate_failed

v1216_p4_short_run.csv:
  method = P4_NOT_OPENED
  reason = no_v1216_functional_P3_survivor
```

这符合 v12.16 gate：没有 P3 survivor 不允许跑 P4 short-run。

## 7. 修改审计

本轮新增/修改：

```text
新增 experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
新增 docs/DG-KAN_v12.16_B320Locked_ExplicitSignalReservoirFunctional_执行复盘.md
新增 docs/DG-KAN_v12.16_B320Locked_ExplicitSignalReservoirFunctional_实验结果复盘.md
```

runner 设计审计：

```text
1. functional direction generation 不使用 CE vector / label-loss VJP / validation / dataset-name branch。
2. CEp99、Brier、holdout loss 只作为 audit / no-harm 字段。
3. P1/P2/P3 gate 失败时，P4 candidates 写 gated-not-run，不盲跑 B16。
4. P4 short-run 只有 P3 survivor 才允许打开，本轮保持 not_run。
5. B320 anchor 只复用 locked artifact 做 no-regression monitor，不小修 B320。
6. Line D 只 carry forward v12.15 status，不把 0 FamilyPass 的 focused repairs 接入 functional。
```

## 8. Official Hash

official hash 摘要：

| artifact | sha256 |
|---|---|
| `v1216_route_decision.json` | `3cc5ae164df94a1c0a2086f638793f243920f49573b99fe097414122951a323d` |
| `v1216_anchor_monitor.csv` | `285df493438810ee512b59ea9b464cb06a7f15ce26ea1e50f8ae8457cd0200df` |
| `v1216_linec_v2_sketch_targets.csv` | `952022da645ae6f287e53dee2d29cf75b801952d2e16b7b9eeef774a4675ffc4` |
| `v1216_explicit_target_calibration.csv` | `25cb6ce9f512a5c465732e872f90e836268c5a141fd5eef2114266c0982fb5a9` |
| `v1216_actuator_response_matrix.csv` | `320b94bee95d030abb6ca01eecf842103feba7cf2dea0709c8181f8b267c9bb6` |
| `v1216_functional_p3_candidates.csv` | `21e76bc08bba5072778f2f659e04bb3bd76c5536259bd5ed9fe9673c2bc693fb` |
| `v1216_functional_p3_summary.csv` | `ea1c8fe22a5e71a987a7d2102ccb1bc0a814ffb46442c324a4043b7c669a3340` |
| `v1216_p4_short_run.csv` | `a3e5c007edf07d5479dad6a4d65690cd01854eec90cbb0a5daba93e3a8273810` |
| `v1216_failure_table.csv` | `bf2d3e3d72b6cea233afa96f6488f27979b4587cf829a3b76d9ac2dd39cf9e36` |
| `v1216_loss_agnostic_audit.csv` | `b8aa6ea104d020e3c422e6cdd6e01dccdc7f4545f2c6bcbe929662e3bf17e45a` |
| `v1216_classic_family_status.csv` | `7d4df77d9756cfb9172e636af520bf1220e4e2e57ddeec63a377a36a3217f154` |
| `v1216_provenance_audit.csv` | `1413c6124b8e5355a4c38d7febcf55a7487ca26a7f73f5a0f02d277b8337f9b9` |

完整 hash 见：

```text
results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10/v1216_hash_manifest.json
results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10/v1216_hash_manifest.json
```
