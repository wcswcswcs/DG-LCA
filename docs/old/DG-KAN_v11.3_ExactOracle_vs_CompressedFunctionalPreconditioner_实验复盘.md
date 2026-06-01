# DG-KAN v11.3 Exact Oracle vs Compressed Functional Preconditioner 实验复盘

> 本复盘记录 `DG-KAN_v11.3_ExactOracle_vs_CompressedFunctionalPreconditioner_完整计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v11.2 真实 artifact、真实 MNIST/Fashion-MNIST/KMNIST 小样本诊断与计划内 synthetic/tabular/sequence 诊断；没有 fake data、proxy rows、占位数据或 CPU offload。旧 generated/action-selector/FPO/FPAU/GoodCone/LFRO 路线继续关闭。

## 0. 最新结论

```text
route = CaseA-ExactOracleFailsStopFunctionSpaceNaturalUpdate
primary_blocker = exact_oracle_or_compression_not_validated
secondary_blocker = stop_current_function_space_update_mainline
system_legal_controller_pass = 0
generated_route_status = stopped_function_space_mainline
```

最终 artifact：`results/real_rerun_20260506/v1130_exact_oracle_vs_compressed_functional_preconditioner_full_20260519T030000Z`

核心结论：

1. P0 v11.2 boundary lock pass = `1`；source route = `R6-MLPFunctionSpaceUpdateStillFails`；P4 source task pass = `1` / `8`。
2. P1 cost q90 = `123.71353525668383` ms；amortized q90 = `155.64006332351633`；compression focus = `cached_jacobian_or_layerwise_approximation`。
3. P2 exact oracle discovery/strong = `0` / `0`；task pass = `2` / `8`；positive signal tasks = `6`。
4. P3 signal pass = `0`；real/random consistency = `0.3314160031529711` / `0.35634383895815464`；noise split improvement = `0.5416666666666666`。
5. P4 compression/production = `0` / `0`；P5/P6/P7/P8 = `not_run` / `not_run` / `not_run` / `not_run`。
6. No-fake rows checked = `99`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1130_exact_oracle_vs_compressed_functional_preconditioner.py
```

```bash
python experiments/run_v1130_exact_oracle_vs_compressed_functional_preconditioner.py --out-dir results/real_rerun_20260506/v1130_exact_oracle_vs_compressed_functional_preconditioner_full_20260519T030000Z --fresh --device auto --data-root data --seed 2413 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1130",
  "status": "summary",
  "route": "CaseA-ExactOracleFailsStopFunctionSpaceNaturalUpdate",
  "primary_blocker": "exact_oracle_or_compression_not_validated",
  "secondary_blocker": "stop_current_function_space_update_mainline",
  "route_explanation": "CaseA-ExactOracleFailsStopFunctionSpaceNaturalUpdate: exact_oracle_or_compression_not_validated; stop_current_function_space_update_mainline.",
  "source_route_v1120": "R6-MLPFunctionSpaceUpdateStillFails",
  "P0_boundary_pass": 1,
  "P1_compression_focus": "cached_jacobian_or_layerwise_approximation",
  "P2_discovery_pass": 0,
  "P2_strong_pass": 0,
  "P2_task_pass_count": 2,
  "P2_task_count": 8,
  "P3_signal_pass": 0,
  "P4_compression_pass": 0,
  "P4_production_candidate_pass": 0,
  "P5_coupling_pass": 0,
  "P6_weak_pass": 0,
  "P7_independent_value_pass": 0,
  "P8_graph_free_migration_pass": 0,
  "generated_route_status": "stopped_function_space_mainline",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. 实现思路与伪代码

### 3.1 P0 边界冻结

```text
load v11.2 route / P1-P8 / no-fake
assert route == R6-MLPFunctionSpaceUpdateStillFails
assert P2 best == PSFP-C-gradient-replacement and P4 task pass == 1/8
assert P6/P7/P8 not_run and fake/proxy/cpu == 0
keep old generated/action selector routes disabled
```

### 3.2 P1 exact oracle cost decomposition

```text
for method in PSFP-C-ref4/ref8/ref16/memory/rank8:
  build real reference batch
  time forward, VJP loop, kernel build, solve, JVP, constraint assembly
  compute total cost and amortized ratio against AdamW step
  record whether one-step heldout/noise/memory/hardtail improves
```

P1 不是成功 gate，而是定位压缩方向。本轮平均主要成本占比：

```text
solve_fraction = 0.027127032106102955
jvp_vjp_fraction = 0.8634278154356722
kernel_fraction = 0.021551380432792415
compression_focus = cached_jacobian_or_layerwise_approximation
```

### 3.3 P2 exact oracle unlimited-budget short training

```text
methods = AdamW, same-norm random, v11.1 additive FSNU,
          PSFP-C exact replacement, PSFP-H population-risk adapter,
          PSFP-C memory constrained, PSFP-C split consistency

for task in eight MLP diagnostics:
  run small short-training curve
  compare each exact candidate against AdamW and random control
  require val AUC gain, real-noise gap, memory/hardtail noharm,
          and random-control specificity
```

P2 task summary：

| task | best method | real improved | noise improved | task pass |
|---|---|---:|---:|---:|
| `mnist` | `v1110-additive-FSNU` | `1` | `1` | `0` |
| `fashion-mnist` | `v1110-additive-FSNU` | `1` | `0` | `1` |
| `kmnist` | `v1110-additive-FSNU` | `1` | `0` | `1` |
| `mnist_label_noise_035` | `v1110-additive-FSNU` | `1` | `1` | `0` |
| `synthetic_interaction` | `PSFP-C-memory-constrained` | `0` | `0` | `0` |
| `synthetic_tabular` | `PSFP-C-memory-constrained` | `1` | `1` | `0` |
| `synthetic_input_noise` | `PSFP-C-split-consistency` | `1` | `0` | `0` |
| `toy_sequence_diagnostic` | `PSFP-H-exact-population-risk-adapter` | `0` | `0` | `0` |

### 3.4 P3 signal/noise decomposition

```text
split batch into sub-batches
for label_mode in real/shuffled/random:
  solve exact direction per split
  compute split consistency, cosine between splits,
          heldout improvement, noise improvement,
          reservoir energy ratio
apply repair logic: more splits, smaller trust radius, low-rank shrinkage
```

### 3.5 P4-P8 hard gate

```text
open compression only if P2 exact oracle discovery or P3 signal pass
open coupling only if compression pass
open decisive short training only if P2/P4/P5 pass
open KAN migration only if MLP strong and independent value pass
```

本轮没有把 high-cost exact oracle、compressed diagnostic 或 split signal diagnostic 写成 official controller。

## 4. A/B/C/D 摘要

```text
P1 compression focus = cached_jacobian_or_layerwise_approximation
P2 discovery/strong = 0 / 0
P3 signal pass = 0
P4 compression/production = 0 / 0
P6 weak/strong = 0 / 0
```

## 5. No-Fake / Hash

```text
rows_checked = 99
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `contract_audit_v1130.csv` | `0ad3c55069bcf7f2718298faaa665a237540e328aec49bc99da5290f98e9eb31` |
| `fig_p1_cost_waterfall_v1130.svg` | `675d6432ee5a6ecf3305051a7cf37b47e2323907c6e4061ea666d0f55b9ceec0` |
| `fig_p2_exact_oracle_training_v1130.svg` | `487c1fdfd6152217b0bdd42dfc3ee478a14777ac193cb7abc9fc1ef6eed2ddc5` |
| `fig_p3_split_consistency_v1130.svg` | `847166c931f64dce36faffbff8b1f08f9b4cbafd09c71e7a149664eeba8d1c03` |
| `fig_p4_compression_matrix_v1130.svg` | `c0b62b8e3cb3800d6e3daccea1f1cd4d37463fad7dd006e0bb01c191e92f4dc6` |
| `fig_p9_route_matrix_v1130.svg` | `ce6e958806744941ca0922fed2ef37360c275b9c6c4aebeadb9dce493b52b08b` |
| `functional_space_natural_step` | `70e8bf027c0042ab4418f59d2bc2d1179695ee37614fa683c98cf126546c0e34` |
| `no_fake_audit_v1130.csv` | `d7a3ed3fd1226aa3982e95f72367972321dd082f5454ffef8083bb9f4eab4cbe` |
| `p0_v1120_boundary_lock_v1130.csv` | `d9bcc4a9ac1f6c14e290d05fd8a25ec9f1c5ae623dbf39342a483c27388246e6` |
| `p1_exact_oracle_cost_decomposition_v1130.csv` | `5bd1b5a8001c05379277745086e42fc422e87e40dfa711932169cda166df46b6` |
| `p2_exact_oracle_unlimited_short_training_v1130.csv` | `b4ee3fbbe7adca529822266e7f2623a9b5feb6bff99f59e27247546aad579590` |
| `p3_exact_signal_noise_decomposition_v1130.csv` | `714551c61836f64971a0464079c57740b0d154ab2b8cf7ca0d3b73a9f83ae4cb` |
| `p4_compressed_functional_preconditioner_v1130.csv` | `7cbfcc853230ed64a15cd47c620d2a14d3cb1a4900c457a9f03887aa07927b1d` |
| `p5_adamw_coupling_compatibility_v1130.csv` | `408c143d2eb675ae486995aed3ac3a359bc3166e71ce37c8e088139a7a6ee753` |
| `p6_decisive_short_training_v1130.csv` | `d13bde7be9ccd8efdf75f85bbc879cfef2cfa941949a479037c72ce3de196d3f` |
| `p7_existing_geometry_optimizer_comparison_v1130.csv` | `793c6e211ba2f844a210fa3721b4a72126bf844f1df54b3745effe90d6edaefc` |
| `p8_kan_graph_free_migration_gate_v1130.csv` | `4a553125acbb577aac345c3da2031fb267fba047a24102b5ac1ce15575923172` |
| `plan` | `874c9d8b5596b0d8c3d889751194c602df9f60e7a7b2f4525981cf7158964e2f` |
| `route_decision_v1130.json` | `87190cb3c961a5cb3a845d33a8224818fa58e57b9e8d930e3e4f7339b17dcb6d` |
| `run_manifest_v1130.json` | `340977bea30ec732d1f48e5ac2fb41e597ebe595a9ee1d4054a0a95587fe6bf7` |
| `runner` | `8415ba2d1d893eb284e34b36e91e9b1fb93884f3d2867b2f91c24a4a0db9de79` |

## 6. 最终分析结论

```text
1. v11.3 按计划先裁决 high-cost exact oracle，而不是继续小修 cheap PSFP-C/rank/beta/periodic。
2. P1 真实拆解 exact oracle 成本，记录 forward/JVP/VJP/kernel/solve/constraint/overhead，不只写总耗时。
3. P2 在 8 个 MLP 诊断任务上跑 exact oracle unlimited-budget short training，并按 real-noise gap、memory/hardtail、random control 判定。
4. P3 已执行 signal/noise split decomposition 和计划内 repair；若 exact oracle/P3 不证明价值，P4-P8 合法关闭。
5. 本轮不使用 future outcome label、generated action label 或 dataset-specific threshold，也不把 discovery diagnostic 写成 controller。
```

最终一句话：v11.3 真实执行后停在 `CaseA-ExactOracleFailsStopFunctionSpaceNaturalUpdate`：CaseA-ExactOracleFailsStopFunctionSpaceNaturalUpdate: exact_oracle_or_compression_not_validated; stop_current_function_space_update_mainline.
