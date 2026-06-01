# DG-KAN v11.1 Function-Space Natural Update 短程失败诊断与重建 实验复盘

> 本复盘记录 `DG-KAN_v11.1_FunctionSpaceNaturalUpdate_短程失败诊断与重建实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v11.0 真实 artifact、真实 MNIST/Fashion-MNIST/KMNIST 小样本诊断、计划内 synthetic/tabular 诊断与 autograd discovery reference；没有 fake data、proxy rows、占位数据或 CPU offload。旧 action-selection / generated-action / FPO/FPAU/GoodCone/LFRO 路线继续关闭。

## 0. 最新结论

```text
route = CaseA-MLPShortTrainingMechanizedVariantsFail
primary_blocker = mechanized_function_space_updates_not_training_stable
secondary_blocker = short_training_smoke_failed_after_P1_P4_repairs
system_legal_controller_pass = 0
generated_route_status = stopped_function_space_natural_update_theory_rebuild
```

最终 artifact：`results/real_rerun_20260506/v1110_function_space_short_failure_rebuild_full_20260519T010000Z`

核心结论：

1. P0 v11.0 boundary lock pass = `1`；source route = `RouteC-MLPFunctionalUpdateFails`；P5 task pass = `0` / `6`。
2. P1 stability weak/strong = `1` / `0`；best = `ref8_tr0.03_lam0.1_rank16_p5`；heldout/random/memory/hardtail = `1.0` / `1.0` / `1.0` / `1.0`。
3. P2 signal weak/strong = `0` / `0`；best = `best_p1`；real/random signal = `1.2308283529085389` / `0.3243995049592254`；noise improved = `0.8333333333333334`。
4. P3 AdamW interaction weak/strong = `1` / `0`；best mode = `periodic10`；best pass count = `2`。
5. P4 constrained solver weak/strong = `0` / `0`；best solver = `unconstrained`；constraint/heldout/cost = `1.0` / `0.8333333333333334` / `15.471136236223156`。
6. P5 mechanized smoke weak/strong = `0` / `0`；task pass = `0` / `6`；P6/P7/P8 = `not_run` / `not_run` / `not_run`。
7. No-fake rows checked = `250`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1110_function_space_short_failure_rebuild.py
```

```bash
python experiments/run_v1110_function_space_short_failure_rebuild.py --out-dir results/real_rerun_20260506/v1110_function_space_short_failure_rebuild_full_20260519T010000Z --fresh --device auto --data-root data --seed 2211 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1110",
  "status": "summary",
  "route": "CaseA-MLPShortTrainingMechanizedVariantsFail",
  "primary_blocker": "mechanized_function_space_updates_not_training_stable",
  "secondary_blocker": "short_training_smoke_failed_after_P1_P4_repairs",
  "route_explanation": "CaseA-MLPShortTrainingMechanizedVariantsFail: mechanized_function_space_updates_not_training_stable; short_training_smoke_failed_after_P1_P4_repairs.",
  "source_route_v1100": "RouteC-MLPFunctionalUpdateFails",
  "P0_boundary_pass": 1,
  "P1_weak_pass": 1,
  "P1_strong_pass": 0,
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P3_weak_pass": 1,
  "P3_strong_pass": 0,
  "P4_weak_pass": 0,
  "P4_strong_pass": 0,
  "P5_weak_pass": 0,
  "P5_strong_pass": 0,
  "P6_weak_pass": 0,
  "P6_strong_pass": 0,
  "P7_weak_pass": 0,
  "P7_strong_pass": 0,
  "P8_controller_pass": 0,
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "generated_route_status": "stopped_function_space_natural_update_theory_rebuild",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. 实现思路与伪代码

### 3.1 P0 边界复现

```text
load v11.0 route_decision / P1-P8 summaries
assert source_route_v1100 == RouteC-MLPFunctionalUpdateFails
assert source v11.0 P1_correctness, P2_weak, P4_strong are true
assert P5_weak is false
keep old action/generated routes disabled
```

### 3.2 P1 稳定性矩阵

为控制成本，本轮使用 fractional factorial matrix，覆盖计划要求的所有 reference/trust/damping/rank/period 取值，而不是全笛卡尔积。

```text
for config in matrix(reference_size, trust_radius, damping_lambda, kernel_rank, update_period):
  for task in six MLP diagnostics:
    solve FSNU delta
    apply delta to clone
    record predicted/actual cosine, train/heldout/memory/hardtail deltas
    compare against AdamW and same-norm random
select best config by weak_count, heldout, random-control separation, cost
```

### 3.3 P2 signal/noise repair

```text
variants = best_p1, split_consistency_rank1, strong_damping_rank4
for variant:
  split train batch into 4 parts
  solve one FSNU direction per split
  project all directions on heldout probe
  compute real signal ratio and matched random ratio
  run random-label and partial-noise controls
  record memory/hardtail harm
select best variant
```

### 3.4 P3 AdamW 交互诊断

```text
methods = AdamW, functional_only, additive_every_step,
          preconditioned_gradient, periodic5, periodic10,
          moment_reset, moment_transport, same_norm_random

for task in first four diagnostics:
  run short curve
  record val AUC, ECE, memory/hardtail, loss spikes,
         cosine with AdamW and optimizer moment norm changes
choose best interaction mode
```

### 3.5 P4 约束求解器重建

```text
solvers = unconstrained, memory_reference, hardtail_reference,
          memory_hardtail_reference, cep99_focused_reference,
          lowrank_periodic_tiny_constraint

for solver:
  put memory/hardtail samples into reference kernel instead of post-hoc veto
  if solver cost is high, try low-rank / tiny memory-hardtail / period10 repair
  record constraint violations, conflict angles, solver cost
select best constrained solver
```

### 3.6 P5 机制化短程重跑

```text
candidates = AdamW,
             Best-P1-damped-periodic,
             Best-P2-signal-filtered,
             Best-P3-interaction-mode,
             Best-P4-constrained-solve,
             same-norm random function-space perturbation

for each of six MLP diagnostic tasks:
  run short training only with these mechanism-backed variants
  require val_loss_auc_time, memory/hardtail, ECE,
          and random-control specificity to pass
```

## 4. A/B/C/D 摘要

```text
P1 weak/strong = 1 / 0
P2 weak/strong = 0 / 0
P3 weak/strong = 1 / 0
P4 weak/strong = 0 / 0
P5 weak/strong = 0 / 0
```

## 5. No-Fake / Hash

```text
rows_checked = 250
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `contract_audit_v1110.csv` | `75f77ed413486c753683ff8a01d43c4384a5ad4d0761ab4b79ef7ebb1ebeea67` |
| `fig_p1_stability_matrix_v1110.svg` | `eb7008afe15c041df526ce2b14dddffab0ab8afcb8bea456e200d43d584a636f` |
| `fig_p2_signal_noise_v1110.svg` | `b9d4040e84ba2790129f4bf7de3b5dd95cda2bc5896b5c4f1471e18f5fb37847` |
| `fig_p3_adamw_interaction_v1110.svg` | `3b6019f5fc4b6d3d2f74b7f639a53b73188fe0af115648af31ff61e27059f7ed` |
| `fig_p4_constraint_solver_v1110.svg` | `973b49dcab02d0cfa919f0a6e9bfce01c1a7a6cb5081c576bde5ef346c3010b2` |
| `fig_p5_mechanized_smoke_v1110.svg` | `960bb39708f62a4adf4985cdac7496a5a1e7a7c6a6294567f025bc55ce4d9460` |
| `fig_v1110_route_matrix.svg` | `b6289427bd5d5eaab6a0ca580e9b64e547f883963f09db34ccb3ddf653f80d79` |
| `functional_space_natural_step` | `70e8bf027c0042ab4418f59d2bc2d1179695ee37614fa683c98cf126546c0e34` |
| `no_fake_audit_v1110.csv` | `2c30e1ab8e907540abb63c7bf53e44167a2bd0c4f19bc53276ca03be1d8dfbd3` |
| `p0_v1100_boundary_lock_v1110.csv` | `9409c99a46243d9fd7a62eb08e97c468a86cb0319bd278471bf18fca16f09922` |
| `p1_stability_matrix_v1110.csv` | `4c89282ad08a82fde34cbddfa7320a4abdc04f5e0d78bc10f7753d58f127a8e3` |
| `p2_signal_noise_reservoir_v1110.csv` | `a0392c2b2a083ee78a987a280e4d57904e37363896889dbb2a317aeed99f1707` |
| `p3_adamw_interaction_v1110.csv` | `e01b34a37083a6e8529e75b5e0ee3e3fef20b6ee05709c520d3106fd7a887803` |
| `p4_constrained_solver_rebuild_v1110.csv` | `b7543a148fcde59ea1f1124769390f32d3c1de725ca1775323db9df2fc5a1bb7` |
| `p5_mechanized_short_smoke_v1110.csv` | `275508c0035efe5aaca42ed4c575ac47332c84525aaabf4b08f4fc4c08a97f80` |
| `p6_edge_function_migration_v1110.csv` | `172fe5d984cae8bb714eed664c6701711b86e547693819504afab88e5e061a34` |
| `p7_future_path_verification_v1110.csv` | `b7af4892146027e6a90d760a4e553eb8245d8ce491f960fc6dc89e746a550173` |
| `p8_controller_runtime_boundary_v1110.csv` | `ed3832c4e1a327dd36cc031dc830818c0359096db3bcf2a27fdb601125aadffe` |
| `plan` | `851b7e4209674eb4c270aaf7f72464e9ebc84e83c728446c4f161d323c7e6904` |
| `route_decision_v1110.json` | `af34afc79530d9c4be72fc1227c038e7dc397a1ccf1d6e5c5bbca4924178e835` |
| `run_manifest_v1110.json` | `0310052f8a7e33de998771e98575519a3a692da08d19eb654b9d6c198accf5e4` |
| `runner` | `8502386e3aa2e436316c85b58c47005595858a95e23233db520057dd1dcf136f` |

## 6. 最终分析结论

```text
1. v11.1 没有回到旧 action/generated/FastSlow selector 路线，而是复现 v11.0 的短程失败边界后继续做训练动力学诊断。
2. P1/P2/P3/P4 都按计划执行了 blocker repair：trust/damping/rank/period、split consistency、AdamW interaction、memory/hardtail in-kernel constraints，以及 P4 cost blocker 的 low-rank/periodic/tiny repair。
3. P2 仍未过 weak，因为 random/noise label benefit 没有被可靠压低；P4 仍未过 weak，主要因为 constrained solver 的 amortized cost 仍高。
4. P5 只使用 P1-P4 给出的机制化变体，不盲目扩大预算；P5 失败后 P6/P7/P8 合法保持 not_run。
5. 最终 route 由真实 P5 mechanized smoke 决定，不使用 fake/proxy/placeholder，也不把 discovery diagnostic 写成 official controller。
```

最终一句话：v11.1 真实执行后停在 `CaseA-MLPShortTrainingMechanizedVariantsFail`：CaseA-MLPShortTrainingMechanizedVariantsFail: mechanized_function_space_updates_not_training_stable; short_training_smoke_failed_after_P1_P4_repairs.
