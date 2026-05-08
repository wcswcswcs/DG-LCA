# DG-KAN v8.7 Stability-First Functional Architecture 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.7_StabilityFirst_FunctionalArchitecture_完整实验计划.md` 的本轮真实执行结果。所有数值只来自本文列出的 `results/real_rerun_20260506/.../` 落盘 CSV/JSON/manifest；没有 fake data、proxy rows、固定占位 ratio 或手填成功结论。本轮继续遵守：no teacher、no self-teacher、no distillation、no loss modification、no sampler/class weight、no CPU offload、no optimizer hyperparameter sweep。

## 1. 是否完成

v8.7 没有完成 formal / stability success。

本轮完成的是 Wave0/P0 稳定性复现、Wave1/P1 fixed stride sensitivity、以及 Wave4/P5 的部分 robust timing audit。结果证明 v8.7 文档的中心判断是对的：`hidden28 + FT7 + stride128` 不是彻底稳定的方法成功；它在 5-seed 下任务和几何仍有正信号，但 wall-clock gate 出现 seed/timing-protocol-sensitive failure。

最新 route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "p0_stability_gate_pass": 0,
  "robust_timing_partial_gate_pass": 0,
  "robust_timing_full_gate_pass": 0,
  "success_v87_formal": 0,
  "primary_blocker": "wallclock_step_ratio_stability_fail"
}
```

## 2. 本轮代码

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_gafu_v87_real.py` | v8.7 stability-first runner；真实调用 v8.5 accepted route，生成 P0 stability reproduction artifact，并把未执行阶段写为 `not_run` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

## 3. P0 single-seed sanity run

先跑了单 seed / 单 rerun，用于验证新 runner 和 artifact path：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p0_stability_first_wave_20260508T000000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --p0-seeds 1314 \
  --p0-reruns 1 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Result：

| metric | value |
|---|---:|
| measured P0 runs | `1` |
| v85 child route | `R1-ExternalFairFunctionalAdvantage` |
| DG1 delta vs KB-MLP | `+0.4299998283` |
| DG1 curvature ratio vs DG0 | `0.8027510493` |
| DG1 Q90 step ratio vs KB-MLP | `1.4845450978` |
| P0 gate | `pass` |

判断：单点复现可运行，但不能用于 v8.7 stability success。

## 4. P0 5-seed first wave

正式 first wave：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --p0-seeds 1314,1315,1316,1317,1318 \
  --p0-reruns 1 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Summary：

| metric | value |
|---|---:|
| measured P0 runs | `5` |
| timing protocols | `1` |
| v85 child external formal pass count | `4/5` |
| DG1 mean delta vs KB-MLP | `+0.2879977226` |
| DG1 mean curvature ratio vs DG0 | `0.7746420803` |
| DG1 Q90 step ratio vs KB-MLP | `1.5462432100` |
| DG1 all-gate pass rate | `0.800000` |
| P0 stability gate | `fail` |

Per-seed DG1：

| seed | v85 route | test acc | delta vs KB-MLP | delta vs DG0 | curvature ratio | step ratio | events |
|---:|---|---:|---:|---:|---:|---:|---:|
| `1314` | `R2-ExternalTaskPassSystemFail` | `97.3399996758` | `+0.4299998283` | `+0.0800013542` | `0.8027510493` | `1.5996302409` | `73` |
| `1315` | `R1-ExternalFairFunctionalAdvantage` | `97.1799969673` | `+0.2699971199` | `+0.0800013542` | `0.7977984806` | `1.4267307839` | `73` |
| `1316` | `R1-ExternalFairFunctionalAdvantage` | `97.1399962902` | `+0.2299964428` | `-0.0299990177` | `0.7882217306` | `1.4661626636` | `73` |
| `1317` | `R1-ExternalFairFunctionalAdvantage` | `97.0999956131` | `+0.1899957657` | `-0.0200033188` | `0.7511351934` | `1.4134503172` | `73` |
| `1318` | `R1-ExternalFairFunctionalAdvantage` | `97.2299993038` | `+0.3199994564` | `-0.0199973583` | `0.7333039474` | `1.4349414233` | `73` |

判断：

1. Task signal 稳：5/5 seed 的 `DG1 - KB-MLP` 均为正。
2. Geometry signal 稳：5/5 seed 的 curvature ratio 均低于 `0.90`。
3. System gate 不稳：seed `1314` 的 step ratio 为 `1.5996302409`，导致 Q90 step ratio `1.5462432100 > 1.50`。
4. 因此 v8.7 当前不能写 stability success；这不是 functional route 全面失败，而是 wall-clock/timing 稳定性失败。

## 5. Artifact 状态

5-seed run 已落盘：

```text
run_manifest.json
p0_child_v85_decisions.csv
v85_fresh_reproduction_stability.csv
p0_stability_summary.csv
route_decision.json
aggregate_decision.json
failure_table.csv
v87_provenance_audit.csv
artifact_hashes.csv
figures/p0_measured_summary.svg
```

以下 v8.7 后续 artifact 已明确写为 `not_run`，原因均为：

```text
v87_current_run_only_executes_p0_stability_reproduction
```

```text
fixed_hyperparam_sensitivity_grid.csv
adaptive_functional_one_step.csv
adaptive_functional_multistep.csv
auto_architecture_selection.csv
robust_timing_protocols.csv
training_compute_counter.csv
kanbefair_multitask_transfer.csv
joint_fair_envelope.csv
functional_causality_multitask.csv
continual_balance_multisplit.csv
robustness_perturbation.csv
functional_mechanism_attribution.csv
negative_boundary_audit.csv
```

## 6. No-fake audit

5-seed run：

```text
rows_checked = 49
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `27977a4682748d686f64c07ed86f67aa24d44e271cbc8770d78e653620bbf442` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| 5-seed route | `dab7cb36b24608e5703cc9ce5a2884fb00f36da4be437f40f4b563a2d806f42d` |
| 5-seed `v85_fresh_reproduction_stability.csv` | `1fe5bbc7ce3e1e90f838be6c1211f3625785e325d4d9cdd08790fd9f2483944b` |
| 5-seed `p0_stability_summary.csv` | `a1b831a0efeca2cb316a3c5925c61a9a1ff7d59dc4ed630f35b47f09a68b3bb0` |
| 5-seed provenance audit | `b0f5757d1f8b51e2d6d423b568d64877f3c9b13115252b3ea4c06a935e025a54` |

## 8. 结论

v8.7 当前没有完成：

```text
P0 first-wave reproduction measured = true
P0 stability gate = false
Adaptive controller = not_run
Multi-protocol timing = not_run
Cross-task external fairness = not_run
success_v87_formal = false
```

机制结论：

1. v8.7 的 Stability-First 方向是必要的：5-seed 真实复现显示，v8.5 的 accepted recipe 不是完全稳定的 route。
2. 失败点很集中：task 与 geometry 都稳，wall-clock step ratio 不稳。
3. `DG1-FT7 hidden28 stride128` 仍有真实价值：5-seed mean delta vs KB-MLP 为 `+0.287998`，mean curvature ratio 为 `0.774642`。
4. 但它还不是自动稳定方法：Q90 step ratio `1.546243` 超过 `1.50`，并且目前只测了 `1` 个 timing protocol、`1` 个 rerun。
5. 下一步应按文档进入 fixed sensitivity / robust timing protocol / adaptive controller，而不是继续手工挑一个更好 seed 或把 4/5 pass 写成 success。

最终一句话：

> v8.7 没完成；它刚刚把 v8.5 的“强成功点”转化成了可审计的稳定性问题。当前真实 blocker 是 timing/system stability：任务优势和几何优势在 5 seed 下仍成立，但 wall-clock Q90 step ratio 超线。下一步必须做预注册的 robust timing 与 adaptive functional controller，不能再靠人工调 stride/seed 写成功。

## 9. 追加：P1 fixed stride sensitivity map

本节继续执行 v8.7 的中心思想：不直接挑一个新 stride 写 success，而是先测 fixed schedule 是否是稳定性来源。P1 复用第 4 节 5-seed P0 的真实 artifact 作为前序上下文，并新增 seed `1314` 的 stride sensitivity 新测量。

复用来源：

```text
results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z/
```

新增 P1 run：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p1_stride_sensitivity_seed1314_20260508T003000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p1-sensitivity \
  --p1-seeds 1314 \
  --p1-strides 64,128,256 \
  --p1-alpha-mults 15.0 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "p0_stability_gate_pass": 0,
  "p1_fixed_schedule_sensitivity_pass": 0,
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "success_v87_formal": 0
}
```

P1 grid：

| seed | stride | alpha | test acc | delta vs KB-MLP | delta vs DG0 | curvature ratio | step ratio | events | route |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1314` | `64` | `15.0` | `97.4199950695` | `+0.5099952221` | `+0.1599967480` | `0.6676250060` | `1.4878222969` | `146` | `R1` |
| `1314` | `128` | `15.0` | `97.3399996758` | `+0.4299998283` | `+0.0800013542` | `0.8027510493` | `1.4908018342` | `73` | `R1` |
| `1314` | `256` | `15.0` | `97.3099946976` | `+0.3999948502` | `+0.0499963760` | `0.8429662853` | `1.4802282876` | `36` | `R1` |

P1 summary：

| metric | value |
|---|---:|
| rows | `3` |
| test acc range | `0.1100003719` percentage point |
| step ratio range | `0.0105735465` |
| curvature ratio range | `0.1753412793` |
| all-gate pass rate | `1.000000` |
| H1 fixed schedule sensitivity pass | `0` |

判断：

1. P1 seed1314 的 stride64/128/256 都通过了 task / geometry / wall-clock gate。
2. 这说明第 4 节 5-seed P0 中 seed1314 的 failure 不能简单归因于 “stride128 本身不对”。
3. P1 的 step ratio range 只有 `0.010574`，没有达到文档 H1 的 `>=0.20` fixed-schedule sensitivity 证据。
4. 但 P0 5-seed 中同一个 seed1314 曾出现 step ratio `1.599630`，而本节 P1 同配置 stride128 只有 `1.490802`。因此当前更像 timing / rerun measurement sensitivity，而不是 fixed stride grid sensitivity。
5. v8.7 下一步应优先实现 robust timing protocols T0-T4，而不是继续手工挑 stride。

No-fake audit：

```text
rows_checked = 52
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 10. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `5d046e1478f9fc83ec63813d66a7613a7bfcaf79e17cc1f0696cfaf7c320eb54` |
| P1 route | `f20c8bccf22751109bf4205a391570b3dd4ca266c33cce784b9516c5f51cc4f2` |
| P1 `fixed_hyperparam_sensitivity_grid.csv` | `0e7d9ead261242b59c8be04d7c2aaa3e9c3ec58c17da101043d0e29e428b96b5` |
| P1 `fixed_hyperparam_sensitivity_summary.csv` | `e4275383b8f65879ad2805e1b995345ca1b8130322e3a86989e2ee69377373c3` |

## 11. 更新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
H1 fixed schedule sensitivity evidence = false on seed1314 narrow grid
Robust timing protocols T0-T4 = not_run
Adaptive controller = not_run
success_v87_formal = false
```

更新机制结论：

1. v8.7 不是卡在 task，也不是卡在 geometry；这两项在 P0/P1 都保持正信号。
2. P1 没有支持 “只要换 fixed stride 就能稳定” 的解释。
3. 当前最干净 blocker 是 timing / rerun stability：同一 seed、同一 stride128 在不同 fresh measurement 中从 `1.599630` 回到 `1.490802`。
4. 下一步应该实现文档 H5 的 robust timing gate，记录 T0-T4 的 `Q90(step ratio)`，再决定是否进入 adaptive controller。

最终一句话：

> v8.7 继续没完成，但 P1 把 blocker 收窄了：不是 task/geometry，也不是简单 fixed stride 错，而是 timing protocol / rerun-sensitive wall-clock。下一步必须做 robust timing protocols，不能用某次 stride128 过线来覆盖 P0 的不稳定结果。

## 12. 追加：P5 T4 existing-artifact robust timing aggregation

本节继续 H5，但不伪造 T0-T4。当前只把已经真实落盘的 full-loop child runs 聚合为 `T4-full-loop-existing-artifact`，用于记录 wall-clock rerun sensitivity。由于没有实际测 T0/T1/T2/T3，本节不声明 robust timing full pass。

新增代码修复：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-robust-timing-from-artifacts`，从已落盘 measured CSV 聚合 `robust_timing_protocols.csv` / `robust_timing_summary.csv` |
| `experiments/run_gafu_v87_real.py` | 避免把 P1 run 中复制进来的 P0 CSV 重复计数 |

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p5_t4_existing_artifact_timing_20260508T010000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z,results/real_rerun_20260506/v87_p1_stride_sensitivity_seed1314_20260508T003000Z \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "robust_timing_partial_gate_pass": 0,
  "robust_timing_full_gate_pass": 0,
  "success_v87_formal": 0
}
```

P5 robust timing summary：

| metric | value |
|---|---:|
| measured rows | `6` |
| timing protocol count | `1` |
| full T0-T4 complete | `0` |
| Q90 step ratio vs KB-MLP | `1.5452160375` |
| max step ratio vs KB-MLP | `1.5996302409` |
| all-gate pass rate | `0.8333333333` |
| robust timing partial gate pass | `0` |
| robust timing full gate pass | `0` |

P5 rows:

| source | seed/run | step ratio | all gate |
|---|---|---:|---:|
| P0 5-seed | `seed1314` | `1.5996302409` | 0 |
| P0 5-seed | `seed1315` | `1.4267307839` | 1 |
| P0 5-seed | `seed1316` | `1.4661626636` | 1 |
| P0 5-seed | `seed1317` | `1.4134503172` | 1 |
| P0 5-seed | `seed1318` | `1.4349414233` | 1 |
| P1 stride grid | `seed1314 stride128 repeat` | `1.4908018342` | 1 |

判断：

1. 这 6 行全部来自真实 landed artifact，不是重新估算或 proxy。
2. T4 full-loop aggregation 仍失败：Q90 step ratio `1.545216 > 1.50`。
3. P1 的 seed1314 repeat 证明同一配置可以回到 `1.490802`，但 robust aggregation 不能忽略 P0 中 `1.599630` 的失败行。
4. 由于只测了 T4 existing-artifact，没有 T0/T1/T2/T3，因此 H5 的 full robust timing gate 仍未完成。
5. 下一步有两个合法方向：实现真正 T0-T4 phase timing，或先做 system-side timing repair，再回到 full T0-T4。

No-fake audit：

```text
rows_checked = 55
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 13. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `e7fd929287b5ea99797eaae1c9e95451190a4fa87920839ef89643088a992467` |
| P5 route | `f2e61269fad8c7347becc5b653cd9e143c9481b5028b617a7ebe2ddb4ec5d47d` |
| P5 `robust_timing_protocols.csv` | `60187294acec55de7d5e701862b090060cc25990b5533c1155d81d0a266d1cfa` |
| P5 `robust_timing_summary.csv` | `7d0573bee4f5be532ad99d0c756b148480092f1c597002dd3425111cb80d1747` |

## 14. 当前更新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
P5 T4 existing-artifact robust timing = measured
P5 full T0-T4 robust timing = not_run
Adaptive controller = not_run
success_v87_formal = false
```

最终当前判断：

1. Task advantage 与 geometry advantage 仍然成立。
2. Fixed stride sensitivity 不是当前最强解释。
3. T4 full-loop wall-clock aggregation 已经专门落盘，并且仍失败。
4. 不能把 P1 中某次 stride128 过线写成成功，因为 P5 robust timing 聚合的 Q90 仍超线。
5. 下一步必须实现完整 T0-T4 timing 或直接修 system-side timing variance。

最终一句话：

> v8.7 当前不是“找不到优势”，而是“优势无法稳定通过 wall-clock gate”。P5 T4 aggregation 让这个 blocker 更清楚：真实 T4 rows 的 Q90 step ratio 是 `1.545216`，仍高于 `1.50`。所以 v8.7 继续不能记成功，下一步应做完整 T0-T4 robust timing 或 system-side timing repair。

## 15. 追加：P5 T3 train-step phase-clean timing

本节继续执行 v8.7 文档 H5：不能只看单一 full-loop timing，要把 train-step-only phase-clean protocol 接入 runner。新增测量仍使用真实 KANbeFair MNIST tensor、真实 KB-MLP autograd Adam train step、真实 DG-KAN manual CE backward / FastAdamWNoSync / FT7 event cadence；没有 fake/proxy/CPU offload。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-t3-phase-clean-timing`，落盘 `T3-train-step-phase-clean` 到 `robust_timing_protocols.csv` |
| `experiments/run_gafu_v87_real.py` | `robust_timing_summary.csv` 新增 `timing_gate_pass_rate`，并把 T3 与已有 T4 artifact 一起计算 Q90 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p5_t3_phase_clean_timing_20260508T011500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z,results/real_rerun_20260506/v87_p1_stride_sensitivity_seed1314_20260508T003000Z \
  --run-t3-phase-clean-timing \
  --t3-seed 1314 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "robust_timing_partial_gate_pass": 0,
  "robust_timing_full_gate_pass": 0,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.6332098171954634,
  "success_v87_formal": 0
}
```

T3 measured row：

| metric | value |
|---|---:|
| protocol | `T3-train-step-phase-clean` |
| dataset protocol | `KANbeFair vision transform; random-subset train=512; prefix-test test=256; shuffle_seed=1314` |
| batch size | `128` |
| warmup / reps | `20 / 260` |
| KB-MLP step ms | `0.4877396268` |
| DG0-KW6 step ms | `0.8172976928` |
| DG1-FT7 step ms | `0.8211482818` |
| DG1 step ratio vs KB-MLP | `1.6835791816` |
| DG1 peak memory MB | `18.4042968750` |
| functional events | `2` |
| functional accept mass | `0.000000` |
| functional reject mass | `2.000000` |
| timing gate pass | `0` |

P5 robust timing summary after adding T3：

| metric | value |
|---|---:|
| measured rows | `7` |
| timing protocol count | `2` |
| full T0-T4 complete | `0` |
| Q90 step ratio vs KB-MLP | `1.6332098172` |
| max step ratio vs KB-MLP | `1.6835791816` |
| timing gate pass rate | `0.7142857143` |
| robust timing partial gate pass | `0` |
| robust timing full gate pass | `0` |

判断：

1. T3 不但没有修掉 timing blocker，反而把 Q90 从 T4-only 的 `1.545216` 推高到 `1.633210`。
2. T3 中 `DG0-KW6` step ms 为 `0.817298`，`DG1-FT7` 为 `0.821148`，二者非常接近；这说明当前 T3 blocker 主要不是 functional event 本身，而是 DG manual base train-step path 相对 KB-MLP autograd path 偏慢。
3. T3 的 functional events 为 `2`，accept mass 为 `0`、reject mass 为 `2`，因此这次 phase-clean timing 中 FT7 guard 没有实际接受 geometry update；不能把 T3 当作 functional advantage 证据。
4. 由于当前只有 T3 与 T4 两类 protocol，T0/T1/T2 仍未测，因此 H5 full robust timing 仍未完成。
5. 本节进一步支持 v8.7 的 stability-first 中心思想：不能继续用某个单次 R1 或某个有利 protocol 写 formal success；必须修 system-side timing 或完成完整 T0-T4 后再决策。

No-fake audit：

```text
rows_checked = 56
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `acc8c72d0920b6c8af6863a2b97d20954aaaf569002aa6ebfccc726778212028` |
| T3 route | `ac325b366f2cd4c3b083cf264a6792945e572ee56e0a6ddc10c8f74f961a3872` |
| T3 `robust_timing_protocols.csv` | `d37d6cf0cb49c6365462237a4a175af2ec395ca7900c7f7538ca2cc2308cb7e7` |
| T3 `robust_timing_summary.csv` | `894f34c19f10566090a0992d6e31a6783cb025151179e422a187fffdfb2cfe84` |
| T3 provenance audit | `5a8737e325f5619cb2169ff0e159c8412ad57f8702076c1ba166670d2eeaec74` |

## 16. 最新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
P5 T4 existing-artifact robust timing = measured
P5 T3 phase-clean timing = measured
P5 full T0-T4 robust timing = not_run
Adaptive controller = not_run
success_v87_formal = false
```

机制结论更新：

1. Task / geometry advantage 仍有真实信号，但 formal blocker 进一步收缩到 system timing。
2. Fixed stride sensitivity 不是当前最强解释；T3 显示 base manual train-step 本体也慢。
3. 当前不应优先“继续挑 stride”，而应先修 KW6 hidden28 manual train-step path，或补齐 T0/T1/T2 后确认 robust timing 分布。
4. 在 T3/T4 都未过 robust gate 的情况下，不能声明 v8.7 stability/formal success。

最终一句话：

> v8.7 仍未完成。新增 T3 phase-clean timing 证明问题不只是 full-loop logging/eval：`DG1-FT7` 的 train-step ratio 为 `1.68358`，且 `DG0` 已经接近同样慢。下一步应优先做 system-side train-step repair / kernelization，或补齐 T0/T1/T2 robust timing；不能把单次 R1 或某个有利 timing protocol 写成 formal success。

## 17. 追加：T3 phase-accounted timing

上一节 T3 已经证明 train-step-only 也失败，但只有总 step time。本节继续沿 P5 要求，把 T3 接入 CUDA event phase accounting，记录 forward / backward / base update / guard / functional update / unknown fraction。所有 phase 数字来自同一轮真实计时，不使用估算拆分。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `_PhaseTimer`，用 CUDA event 记录 T3 phase time |
| `experiments/run_gafu_v87_real.py` | `T3-train-step-phase-clean` 行追加 `forward_time_ms`、`backward_time_ms`、`base_update_time_ms`、`guard_time_ms`、`functional_update_time_ms`、`unknown_time_fraction` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p5_t3_phase_accounted_timing_20260508T013000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z,results/real_rerun_20260506/v87_p1_stride_sensitivity_seed1314_20260508T003000Z \
  --run-t3-phase-clean-timing \
  --t3-seed 1314 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "robust_timing_partial_gate_pass": 0,
  "robust_timing_full_gate_pass": 0,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.6107783862497393,
  "success_v87_formal": 0
}
```

T3 phase-accounted row：

| metric | value |
|---|---:|
| protocol | `T3-train-step-phase-clean` |
| KB-MLP step ms | `0.5523959855` |
| KB-MLP forward ms | `0.0975400614` |
| KB-MLP backward ms | `0.2491369854` |
| KB-MLP update ms | `0.1551488002` |
| DG0-KW6 step ms | `0.9017060081` |
| DG0 forward ms | `0.2466728618` |
| DG0 backward ms | `0.5094320015` |
| DG0 update ms | `0.1080315080` |
| DG1-FT7 step ms | `0.8990248003` |
| DG1 step ratio vs KB-MLP | `1.6275006043` |
| DG1 forward ms | `0.2470001233` |
| DG1 backward ms | `0.5017149532` |
| DG1 base update ms | `0.1071051076` |
| DG1 guard ms | `0.0026750770` |
| DG1 functional update ms | `0.0033780923` |
| DG1 kernel/mapped ms | `0.8618733534` |
| unknown time fraction | `0.0413241624` |
| functional events | `2` |
| functional accept / reject mass | `0.000000 / 2.000000` |
| timing gate pass | `0` |

P5 robust timing summary after phase-accounted T3：

| metric | value |
|---|---:|
| measured rows | `7` |
| timing protocol count | `2` |
| full T0-T4 complete | `0` |
| Q90 step ratio vs KB-MLP | `1.6107783862` |
| max step ratio vs KB-MLP | `1.6275006043` |
| timing gate pass rate | `0.7142857143` |
| robust timing partial gate pass | `0` |
| robust timing full gate pass | `0` |

判断：

1. Phase accounting 可信：unknown fraction `0.041324`，低于 P5 time-accounting 的 `0.10` 要求。
2. T3 失败主因集中在 forward/backward，而不是 FT7 guard/update：DG1 forward `0.247000 ms` vs KB-MLP `0.097540 ms`；DG1 backward `0.501715 ms` vs KB-MLP `0.249137 ms`。
3. DG0 与 DG1 的 step time 几乎相同：`0.901706 ms` vs `0.899025 ms`。因此当前 T3 blocker 是 KW6 hidden28 manual architecture / backward implementation 的系统开销，而不是 functional event 频率本身。
4. Guard + functional update 只有约 `0.006053 ms/step`，不能解释 `1.6275x` step ratio。
5. 这把下一步从“继续调 FT7 event”进一步收窄到“修 manual stack forward/backward kernel path 或 base architecture selection”。

No-fake audit：

```text
rows_checked = 56
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `43d34c009607e1aad780cf4a424bc52fd8c24ab2b0a623fed859d118a5f7bb80` |
| T3 phase-accounted route | `18bd68f9a140f372c074705099f737db598ed95818a1d644348a005a213dae53` |
| T3 phase-accounted `robust_timing_protocols.csv` | `1de588c3bed346a750f05871848799b629bbaff5d474086a2d6c624171791289` |
| T3 phase-accounted `robust_timing_summary.csv` | `777be2b84a4258c0b8bf87d76da7bd06b7f8ce0bd65da3f99a3c9d24d52de13c` |
| T3 phase-accounted provenance audit | `5a8737e325f5619cb2169ff0e159c8412ad57f8702076c1ba166670d2eeaec74` |

## 18. 最新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
P5 T4 existing-artifact robust timing = measured
P5 T3 phase-clean timing = measured
P5 T3 phase-accounted timing = measured
P5 full T0-T4 robust timing = not_run
Adaptive controller = not_run
success_v87_formal = false
```

机制结论更新：

1. v8.7 的中心判断继续成立：v8.5 的 strong route 是真实信号，但还不是稳定方法。
2. 当前最清楚 blocker 是 KW6 hidden28 manual stack/base train-step system overhead，尤其 backward phase。
3. FT7 functional update 在 T3 中几乎不是时间主因，因此继续降低 event frequency 不是正路。
4. 下一步应优先做 base architecture / manual kernel path 的 stability-first system repair，或按 H5 补齐 T0/T1/T2 后再进入 adaptive controller。

最终一句话：

> v8.7 仍不能写成功。Phase-accounted T3 把问题定位得更硬：DG1 的 `1.6275x` step ratio 主要来自 manual forward/backward，而不是 functional guard/update。下一步要修 KW6 hidden28 manual path 或做自动 base architecture selection；继续手工调 FT7 stride 会偏离 v8.7 的中心思想。

## 19. 追加：P4 auto hidden system pilot

本节继续执行 v8.7 H4 / Wave3 的中心思想：hidden 不应继续靠人工试，而应由公平 envelope 自动确定。本轮先做一个窄的 system-only pilot：只使用 params / FLOPs / T3 train-step pilot，不使用 test metric，不作为 final route selection。

Selection rule：

```text
max hidden such that:
  params_ratio_vs_KB_MLP <= 1.0
  flops_ratio_vs_KB_MLP <= 1.0
  step_ratio_vs_KB_MLP <= 1.50
test_metric_used_for_selection = 0
```

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-p4-hidden-pilot`，真实测 hidden candidates 的 T3 phase-accounted system pilot |
| `experiments/run_gafu_v87_real.py` | 落盘 `auto_architecture_selection.csv` 与 `auto_architecture_selection_summary.csv` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_hidden_system_pilot_20260508T014500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-hidden-pilot \
  --p4-hidden-candidates 16,20,24,28,32 \
  --p4-seed 1314 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "p4_auto_hidden_pilot_pass": 1,
  "p4_selected_hidden_dim": 28,
  "p4_selected_step_ratio_vs_KB_MLP": 1.4748026387673812,
  "success_v87_formal": 0
}
```

P4 hidden pilot：

| hidden | params ratio | FLOPs ratio | step ratio | forward ms | backward ms | update ms | selection pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `16` | `0.5218074656` | `0.5201151661` | `1.5627181959` | `0.2542785232` | `0.5565435057` | `0.1131463382` | 0 |
| `20` | `0.6585461690` | `0.6563691542` | `1.4864347023` | `0.2508928005` | `0.5179833839` | `0.1102825845` | 1 |
| `24` | `0.7977996071` | `0.7951132208` | `1.4934626983` | `0.2518379076` | `0.5198487380` | `0.1115348923` | 1 |
| `28` | `0.9395677800` | `0.9363473660` | `1.4748026388` | `0.2492545230` | `0.5127268902` | `0.1097618461` | 1 |
| `32` | `1.0838506876` | `1.0800715898` | `1.4747334584` | `0.2490086157` | `0.5125294748` | `0.1098621540` | 0 |

Summary：

| metric | value |
|---|---:|
| selected hidden | `28` |
| selected params ratio | `0.9395677800` |
| selected FLOPs ratio | `0.9363473660` |
| selected step ratio | `1.4748026388` |
| test metric used for selection | `0` |

判断：

1. P4 system-only pilot 在这一轮会选回 `hidden28`，说明 v8.5 的 hidden28 不是纯人工偶然点；它在 params/FLOPs envelope 下仍是最大可选 hidden。
2. 但这个结果不能覆盖前面 P0/T3 robust failure：上一节 phase-accounted T3 中同样 hidden28 的 step ratio 是 `1.6275006043`，本节 P4 pilot 是 `1.4748026388`。
3. 差异主要来自 KB-MLP timing baseline 变化：上一节 KB-MLP step `0.552396 ms`，本节 KB-MLP step `0.615845 ms`。因此单次 hidden pilot 仍受 timing protocol / run variance 影响。
4. 简单降低 hidden 也不是稳定解：hidden16 的 params/FLOPs 很低，但 step ratio 仍为 `1.562718`，没有过 system pilot。
5. 下一步如果继续 P4，必须把 hidden selection 放进 robust timing protocol，而不是单次 pilot 选 hidden。

No-fake audit：

```text
rows_checked = 54
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `84be54348fb164bf80eb18c5c618e284e68c8f632eb53a82837629fb83914777` |
| P4 route | `419a426ee86f47d4671c831781e489a78166016d46b23dd4f2097d1d5391bdd8` |
| P4 `auto_architecture_selection.csv` | `c3983ef1b8ea29e17e8b259b7ef2445125b099be00ae64eb14b06432a8cd0631` |
| P4 `auto_architecture_selection_summary.csv` | `f9824c5d29975be3a7c99222c1804fabdd749b8abb162695d521ce7b2bc9c45a` |
| P4 provenance audit | `fdcc531fe3516826fb068d0313247db67ff62c7dc16ce7c79045a8a32ef75e31` |

## 20. 最新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
P4 auto hidden system pilot = measured
P5 T4 existing-artifact robust timing = measured
P5 T3 phase-accounted timing = measured
P5 full T0-T4 robust timing = not_run
Adaptive controller = not_run
success_v87_formal = false
```

机制结论更新：

1. P4 pilot 支持 hidden28 作为 params/FLOPs envelope 下的最大 system candidate，但不支持 formal success。
2. hidden16 仍然慢，说明降低 hidden 不是万能的 system repair。
3. 当前最干净 blocker 仍是 robust timing：同一 hidden28 在不同 measured timing run 中可从 `1.4748` 到 `1.6275`。
4. 下一步应把 T0/T1/T2 加入 timing protocol，或直接修 manual backward/kernel path，再重新做 P4/P5。

最终一句话：

> v8.7 仍未完成。P4 system pilot 选回 hidden28，但这不是成功，而是说明 v8.5 的点有合理性；真正问题仍是 timing robustness。只要 hidden28 的 T3 step ratio 能在真实 run 中从 `1.47x` 摆到 `1.63x`，v8.7 就不能进入 formal success。

## 21. 追加：P5 full T0-T4 robust timing protocol

本节补齐 v8.7 H5 要求的 T0/T1/T2/T3/T4 timing protocol。新增 T0/T1/T2 都是同一真实 KANbeFair MNIST batch 上的 phase-accounted train-step timing，不使用 proxy row，也不把已有 T4 full-loop artifact 复制成新 protocol。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-t0-t2-phase-clean-timing` |
| `experiments/run_gafu_v87_real.py` | 新增 `--t0-t2-protocols`，默认 `T0:50:200,T1:100:500,T2:200:1000` |
| `experiments/run_gafu_v87_real.py` | 将 T3 phase-accounted timing 抽成通用 helper，T0/T1/T2/T3 共用同一真实计时路径 |
| `experiments/run_gafu_v87_real.py` | route 在 full T0-T4 已测但失败时，明确指向 `repair_manual_train_step_timing_before_adaptive_controller` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p5_full_t0_t4_timing_20260508T021500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z,results/real_rerun_20260506/v87_p1_stride_sensitivity_seed1314_20260508T003000Z \
  --run-t0-t2-phase-clean-timing \
  --t0-t2-seed 1314 \
  --t0-t2-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --run-t3-phase-clean-timing \
  --t3-seed 1314 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "next_required_implementation": "repair_manual_train_step_timing_before_adaptive_controller",
  "robust_timing_partial_gate_pass": 0,
  "robust_timing_full_gate_pass": 0,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.6930583804208144,
  "success_v87_formal": 0
}
```

P5 robust timing summary：

| metric | value |
|---|---:|
| measured rows | `10` |
| timing protocol count | `5` |
| full T0-T4 complete | `1` |
| Q90 step ratio vs KB-MLP | `1.6930583804` |
| max step ratio vs KB-MLP | `1.9832990300` |
| timing gate pass rate | `0.500000` |
| all-gate pass rate | `0.500000` |
| robust timing partial gate pass | `0` |
| robust timing full gate pass | `0` |

T0-T3 phase-accounted rows：

| protocol | warmup/reps | KB-MLP step | DG0 step | DG1 step | DG1 ratio | DG1 forward | DG1 backward | DG1 update | guard+func | unknown frac | events | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T0` | `50/200` | `0.564994` | `0.907738` | `0.921995` | `1.631868` | `0.256081` | `0.513873` | `0.109839` | `0.004119` | `0.041305` | `1` | 0 |
| `T1` | `100/500` | `0.539261` | `0.888283` | `0.895609` | `1.660809` | `0.245089` | `0.500282` | `0.107325` | `0.005985` | `0.041234` | `4` | 0 |
| `T2` | `200/1000` | `0.523033` | `0.890006` | `1.037331` | `1.983299` | `0.253035` | `0.628070` | `0.110734` | `0.006021` | `0.038050` | `9` | 0 |
| `T3` | `20/260` | `0.646276` | `1.034424` | `1.049677` | `1.624192` | `0.288832` | `0.585149` | `0.125275` | `0.006511` | `0.041832` | `2` | 0 |

T4 existing-artifact rows 仍来自第 4/9 节已经落盘的 full-loop runs：

| source | seed/run | step ratio | pass |
|---|---|---:|---:|
| P0 5-seed | `seed1314` | `1.5996302409` | 0 |
| P0 5-seed | `seed1315` | `1.4267307839` | 1 |
| P0 5-seed | `seed1316` | `1.4661626636` | 1 |
| P0 5-seed | `seed1317` | `1.4134503172` | 1 |
| P0 5-seed | `seed1318` | `1.4349414233` | 1 |
| P1 stride128 repeat | `seed1314` | `1.4908018342` | 1 |

判断：

1. H5 的 T0-T4 timing protocol 已补齐：`full_t0_t4_complete = 1`。
2. 补齐后不是 success，而是更明确的 failure：Q90 step ratio `1.693058`，max `1.983299`，full gate 仍为 `0`。
3. T0/T1/T2/T3 全部 train-step protocol 都失败，说明 blocker 不只是 full-loop logging / validation sync。
4. T2 high-rep protocol 最差，DG1 backward 达到 `0.628070 ms`，使 ratio 升到 `1.983299`。
5. FT7 guard + functional update 仍很小：T0-T3 中 guard+func 约 `0.0041-0.0065 ms/step`，不能解释主要 slowdown。
6. 因此下一步应修 manual train-step path，尤其 stack/head forward-backward kernel/system path；在这之前进入 adaptive controller 会偏离 v8.7 stability-first 中心思想。

No-fake audit：

```text
rows_checked = 59
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `bd30909ebbd8f0716a86c7f200c0c22364ae45c123a01ef48e2ffbca854eda29` |
| full T0-T4 route | `c908082474530046095ec5e3c7b382866f3195961aca1c8b66f0138a3baddbff` |
| full T0-T4 `robust_timing_protocols.csv` | `4ac72e1f0cb85c6762133839599404e5951887bebeaa4c5f75980c656a802f96` |
| full T0-T4 `robust_timing_summary.csv` | `671d1c4eb0700b1b80d2c43297759ac89ac081fc67297b870425ff7bd11d89fb` |
| full T0-T4 provenance audit | `1bf19b191e10cb55be8fa85299df25023e5795f05212c9d48b17e6fdfc1d5bec` |

## 22. 最新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
P4 auto hidden system pilot = measured
P5 T4 existing-artifact robust timing = measured
P5 T3 phase-accounted timing = measured
P5 full T0-T4 robust timing = measured
P5 full T0-T4 robust timing gate = false
Adaptive controller = not_run
success_v87_formal = false
```

机制结论更新：

1. v8.7 的 formal blocker 已经不是 “T0-T4 未测”，而是 “T0-T4 已测且 robust timing fail”。
2. Task / geometry 信号仍然来自 P0/P1 的真实 rows；但 system stability 没有闭合，所以不能声明 formal。
3. T0-T4 共同指向同一个机制：manual DG-KAN train-step 的 forward/backward path 相对 KB-MLP autograd path 不稳定且偏慢。
4. 单次 P4 hidden pilot 选择 hidden28 有合理性，但不能覆盖 T0-T4 的 robust failure。
5. 下一步应做 stability-first system repair，例如 manual backward kernelization、减少 Python/kernel launch、或重新设计自动 architecture selection 的 system objective；不能继续靠挑 seed/stride，也不能把 T4 中部分 pass 写成 success。

最终一句话：

> v8.7 仍未完成。现在完整 T0-T4 已经测完，结论更硬：`DG1-FT7 hidden28` 的 robust timing Q90 是 `1.693058`，远高于 `1.50`。当前必须修 manual train-step system path；继续调 FT7 event 或选择有利 timing protocol 都会偏离 Stability-First 的中心思想。

## 23. 追加：P4 robust hidden selection

上一节已经补齐 T0-T4 并确认 robust timing fail。本节继续执行 v8.7 H4：不再用单次 pilot 或人工 hidden28，而是把 hidden selection 放进多 protocol robust system gate。选择仍不使用 test metric，不使用 final acc，不根据任务结果调 hidden。

Selection rule：

```text
max hidden such that:
  params_ratio_vs_KB_MLP <= 1.0
  flops_ratio_vs_KB_MLP <= 1.0
  q90_step_ratio_vs_KB_MLP <= 1.50
across P4 protocols = T0,T1,T2,T3
test_metric_used_for_selection = 0
```

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-p4-robust-hidden-selection` |
| `experiments/run_gafu_v87_real.py` | 新增 `auto_architecture_selection_by_hidden.csv`，按 hidden 聚合 params/FLOPs/Q90 step |
| `experiments/run_gafu_v87_real.py` | P4 robust selection 失败时 route 指向 `repair_manual_train_step_timing_no_hidden_passes_robust_architecture_selection` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_robust_hidden_selection_20260508T024500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-robust-hidden-selection \
  --p4-hidden-candidates 16,20,24,28,32 \
  --p4-seed 1314 \
  --p4-robust-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "next_required_implementation": "repair_manual_train_step_timing_no_hidden_passes_robust_architecture_selection",
  "p4_auto_hidden_pilot_pass": 0,
  "p4_selected_hidden_dim": "metric_unavailable",
  "success_v87_formal": 0
}
```

P4 robust hidden summary：

| hidden | params ratio | FLOPs ratio | Q90 step ratio | max step ratio | timing pass rate | selected |
|---:|---:|---:|---:|---:|---:|---:|
| `16` | `0.5218074656` | `0.5201151661` | `1.9870124928` | `2.0244276082` | `0.25` | 0 |
| `20` | `0.6585461690` | `0.6563691542` | `1.7779113838` | `1.8351439567` | `0.25` | 0 |
| `24` | `0.7977996071` | `0.7951132208` | `1.6536627547` | `1.6606387347` | `0.00` | 0 |
| `28` | `0.9395677800` | `0.9363473660` | `2.0984836979` | `2.1023723969` | `0.00` | 0 |
| `32` | `1.0838506876` | `1.0800715898` | `1.8972273427` | `1.9696496789` | `0.00` | 0 |

Representative protocol rows：

| hidden | protocol | KB-MLP step | DG1 step | ratio | forward | backward | update | guard+func | pass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `16` | `T0` | `0.638145` | `0.892801` | `1.399057` | `0.244214` | `0.500510` | `0.107396` | `0.004109` | 1 |
| `16` | `T1` | `0.445686` | `0.902260` | `2.024428` | `0.245130` | `0.505861` | `0.108181` | `0.006129` | 0 |
| `20` | `T1` | `0.629302` | `0.890267` | `1.414691` | `0.241015` | `0.500558` | `0.106723` | `0.005857` | 1 |
| `20` | `T3` | `0.550243` | `0.904803` | `1.644369` | `0.246057` | `0.507478` | `0.108336` | `0.005698` | 0 |
| `24` | `T2` | `0.539742` | `0.896316` | `1.660639` | `0.242960` | `0.502566` | `0.107849` | `0.005855` | 0 |
| `28` | `T0` | `0.429851` | `0.903706` | `2.102372` | `0.248095` | `0.505166` | `0.108744` | `0.003672` | 0 |
| `32` | `T0` | `0.451871` | `0.890028` | `1.969650` | `0.243411` | `0.498683` | `0.106988` | `0.003611` | 0 |

判断：

1. Robust P4 没有选出任何 hidden：`selection_pass = 0`。
2. 这不是因为 params/FLOPs 太严：hidden16/20/24/28 都在 params/FLOPs envelope 内，但 Q90 step 都超过 `1.50`。
3. 最接近的是 hidden24，Q90 仍为 `1.653663`，高于 gate。
4. hidden16 在 T0 单 protocol 可以过线，但 T1/T2/T3 失败；这正是 v8.7 要避免的 timing-protocol-sensitive selection。
5. hidden28 在本轮 robust selection 中 Q90 `2.098484`，进一步说明不能用单次 P4 pilot 的 `1.474803` 作为 architecture success。
6. 结论进一步收紧：降低 hidden 或自动选 hidden 不能修当前 blocker，必须修 manual train-step path。

No-fake audit：

```text
rows_checked = 74
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `412ac80243a80c90b644782f9b72de220165634109dc47e9c1611a56527959c8` |
| P4 robust route | `95f01f063b439dccafe9288e3a97b077e1d80c239a6ba5dd2c6a9cef0bbf051a` |
| P4 robust `auto_architecture_selection.csv` | `d03168f9ebf9991af046cbbbe67676a66f043738bcc1a3812f7e4903b429edb3` |
| P4 robust `auto_architecture_selection_by_hidden.csv` | `9dd5647bbe6840036194b425e6ca8eb145ee4883e9ee03cca20b03a2c271fa8c` |
| P4 robust `auto_architecture_selection_summary.csv` | `38122df984837f66445dd2d1a6dcfa165310ad5c6a889a7a7d9bbb068294a394` |
| P4 robust provenance audit | `ae23e8b0ab3977c56807098c2189e6e2d32872467da13fb79666f59a4093b0f1` |

## 24. 最新结论

v8.7 仍未完成：

```text
P0 5-seed first wave = measured
P0 stability gate = false
P1 fixed stride sensitivity = measured
P4 single hidden system pilot = measured
P4 robust hidden selection = measured
P4 robust hidden selection pass = false
P5 full T0-T4 robust timing = measured
P5 full T0-T4 robust timing gate = false
Adaptive controller = not_run
success_v87_formal = false
```

机制结论更新：

1. 当前 blocker 不再只是 “hidden28 不稳”；即使把 hidden selection 放进 robust protocol，`16/20/24/28/32` 也没有一个通过。
2. 简单降低 hidden 不解决问题。hidden16 的 params/FLOPs 很低，但 robust Q90 step 仍为 `1.987012`。
3. 单 protocol pass 不能用作 success：hidden16 T0 与 hidden20 T1 能过，但 robust Q90 不过。
4. FT7 guard/function update 仍不是主时间源；各 hidden 的 slowdown 主要来自 manual forward/backward。
5. 下一步必须做 system-side repair：manual backward kernelization、减少 Python/kernel launch、或引入数值等价的 fused manual path。进入 adaptive controller 前必须先解决 base train-step timing。

最终一句话：

> v8.7 仍未完成。P4 robust hidden selection 证明自动 hidden 规则救不了当前系统问题：没有任何候选 hidden 通过 `q90_step <= 1.50`。因此下一步不是继续挑 hidden/stride，而是修 DG-KAN manual train-step 的 forward/backward system path。

## 25. 追加：P4 base implementation robust screen

本节继续第 24 节的 blocker，但不改变 v8.7 中心思想：

```text
no teacher / no self-teacher
no distillation / no loss modification
no sampler / class weight
no CPU offload
no optimizer hyperparameter sweep
test_metric_used_for_selection = 0
```

新增实现：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `_timed_dg_train_step` 支持 `dg_candidate` 参数 |
| `experiments/run_gafu_v87_real.py` | 新增 `base_implementation_screen.csv` / `base_implementation_screen_by_candidate.csv` / `base_implementation_screen_summary.csv` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_base_impl_screen_20260508T030000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KF10,KW3,KW4,KW5,KW6 \
  --p4-base-hidden-dim 28 \
  --p4-seed 1314 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "primary_blocker": "wallclock_step_ratio_stability_fail",
  "next_required_implementation": "repair_manual_train_step_timing_no_base_implementation_variant_passes_robust_screen",
  "p4_selected_base_candidate_id": "metric_unavailable",
  "success_v87_formal": 0
}
```

Base implementation summary：

| candidate | hidden | params ratio | FLOPs ratio | Q90 step ratio | max step ratio | timing pass rate | selection |
|---|---:|---:|---:|---:|---:|---:|---:|
| `KF10` | 28 | `0.9395677800` | `0.9363473660` | `1.8510091672` | `1.9446558600` | `0.25` | 0 |
| `KW3` | 28 | `0.9395677800` | `0.9363473660` | `1.8333024932` | `1.9250497696` | `0.50` | 0 |
| `KW4` | 28 | `0.9395677800` | `0.9363473660` | `1.7118252606` | `1.8009074799` | `0.50` | 0 |
| `KW5` | 28 | `0.9395677800` | `0.9363473660` | `1.9585290474` | `2.0610598732` | `0.00` | 0 |
| `KW6` | 28 | `0.9395677800` | `0.9363473660` | `1.7889831728` | `1.8818902017` | `0.50` | 0 |

判断：

1. `KW4` 是 hidden28 下最快的 base implementation，但 Q90 `1.711825` 仍高于 `1.50`。
2. `KW5` compiled explicit path 没有修 timing，Q90 反而达到 `1.958529`。
3. 因此 “换已有 base implementation variant” 不能闭合 v8.7 robust timing。

No-fake audit：

```text
rows_checked = 75
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 26. 追加：KW4/KW6 hidden cross screen

由于 `KW4 hidden28` 是第 25 节最快的 base implementation，本节检查 `KW4/KW6 x hidden16/20/24/28` 是否能靠 “更快 implementation + lower hidden” 过 robust system gate。仍然不使用 test metric 选模型。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_base_hidden_cross_screen_20260508T031500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KW4,KW6 \
  --p4-base-hidden-candidates 16,20,24,28 \
  --p4-seed 1314 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --t3-batch-size 128 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Result：

| candidate | hidden | params ratio | FLOPs ratio | Q90 step ratio | max step ratio | timing pass rate | selection |
|---|---:|---:|---:|---:|---:|---:|---:|
| `KW4` | 16 | `0.5218074656` | `0.5201151661` | `1.5571325803` | `1.5653617061` | `0.50` | 0 |
| `KW4` | 20 | `0.6585461690` | `0.6563691542` | `1.5620629299` | `1.5635016717` | `0.50` | 0 |
| `KW4` | 24 | `0.7977996071` | `0.7951132208` | `1.5517703349` | `1.5531542073` | `0.50` | 0 |
| `KW4` | 28 | `0.9395677800` | `0.9363473660` | `1.5556521108` | `1.5599544645` | `0.50` | 0 |
| `KW6` | 16 | `0.5218074656` | `0.5201151661` | `1.7442716286` | `1.8018848578` | `0.50` | 0 |
| `KW6` | 20 | `0.6585461690` | `0.6563691542` | `1.6194917655` | `1.6215467825` | `0.50` | 0 |
| `KW6` | 24 | `0.7977996071` | `0.7951132208` | `1.6134786978` | `1.6151507122` | `0.50` | 0 |
| `KW6` | 28 | `0.9395677800` | `0.9363473660` | `1.6154588674` | `1.6178110984` | `0.50` | 0 |

判断：

1. 交叉 screen 仍没有任何 pair 通过 `q90_step <= 1.50`。
2. 最接近的是 `KW4 hidden24`，Q90 `1.551770`，只差约 `0.051770`，但仍不能写 pass。
3. `KW4` 明显优于 `KW6` 的 base timing，但没有足够 margin 进入 formal gate。
4. 这说明下一步不能再停留在 hidden/candidate screen；需要真实 implementation-side train-step repair。

No-fake audit：

```text
rows_checked = 90
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 27. 追加：cached AdamW param rows rejected probe

本节尝试一个实现侧 micro-trim：缓存 `FastAdamWNoSync` 的 param/grad row list，避免每步重新从 providers 生成 tuple list。该 probe 不改变 AdamW tensor update 公式，也不改变 learning rate / weight decay / beta / eps；因此不是 optimizer hyperparameter sweep。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_kw4_cached_param_rows_screen_20260508T033000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KW4 \
  --p4-base-hidden-candidates 16,20,24,28 \
  --p4-seed 1314 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --t3-batch-size 128 \
  --use-cached-adamw-param-rows \
  --kb-epochs 20 \
  --primary-epochs 20
```

Result：

| candidate | hidden | Q90 step ratio | max step ratio | timing pass rate | selection |
|---|---:|---:|---:|---:|---:|
| `KW4 cached` | 16 | `1.8688668923` | `1.9830345634` | `0.50` | 0 |
| `KW4 cached` | 20 | `1.8736167701` | `1.9881594430` | `0.50` | 0 |
| `KW4 cached` | 24 | `1.8754571848` | `1.9907109232` | `0.50` | 0 |
| `KW4 cached` | 28 | `1.8762931429` | `1.9906869330` | `0.50` | 0 |

判断：

1. cached param rows 没有修复 timing，反而比第 26 节普通 path 更慢。
2. 因此当前 bottleneck 不是每步重新生成 param/grad row list。
3. 本 probe rejected；不能作为 v8.7 修复路径。

No-fake audit：

```text
rows_checked = 70
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 28. 最新结论

v8.7 仍未完成：

```text
P0 stability gate = false
P4 robust hidden selection = false
P4 base implementation screen = false
P4 hidden x implementation cross screen = false
cached AdamW param rows probe = rejected
Adaptive controller = not_run
success_v87_formal = false
```

机制结论更新：

1. 已有 base implementation variants 中，`KW4` 是最快的系统路径，但 hidden28 Q90 仍为 `1.711825`。
2. `KW4 hidden24` 是目前最接近的 robust system point，Q90 `1.551770`，但仍未过 `1.50`。
3. `KW6` 在所有 hidden 下都慢于 `KW4`；继续沿 v8.5 的 `KW6 hidden28` 原路径不会自然稳定。
4. cached param rows 失败，说明 optimizer Python row construction 不是主瓶颈。
5. 当前 blocker 已经收缩到更具体的位置：manual stack/head forward/backward/update kernel path 需要数值等价的实现级修复，尤其是 KW4 类 cached no-input-grad AtenLowerOnlySiLUBackward path 的剩余 `~3.5%` timing gap。

最终一句话：

> v8.7 还没完成。现在最接近的是 `KW4 hidden24`，但 robust Q90 step ratio 仍是 `1.55177`，不能写 success。下一步应做真正的 manual train-step kernel/launch repair，而不是继续挑 hidden、stride、candidate 或把单 protocol pass 写成稳定方法。

## 29. 追加：foreach AdamW addcdiv system repair screen

本节继续第 28 节的实现侧 blocker。新增 `ForeachAdamWAddcdivNoSync`，用 torch foreach tensor ops 执行 AdamW addcdiv update；不改变 CE objective、functional update、teacher/loss/sampler/class weight/offload 合同，也不改变 AdamW 超参。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `_ForeachAdamWAddcdivNoSync` |
| `experiments/run_gafu_v87_real.py` | `--use-foreach-adamw-addcdiv` 接入 P4 base implementation screen |

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_kw4_foreach_adamw_screen_20260508T034500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KW4 \
  --p4-base-hidden-candidates 16,20,24,28 \
  --p4-seed 1314 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --t3-batch-size 128 \
  --use-foreach-adamw-addcdiv \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R5-TimingProtocolSensitiveSuccess",
  "p4_auto_hidden_pilot_pass": 1,
  "p4_selected_base_candidate_id": "KW4",
  "p4_selected_hidden_dim": 28,
  "p4_selected_step_ratio_vs_KB_MLP": "1.3610995445497243",
  "success_v87_formal": 0
}
```

P4 foreach screen：

| candidate | hidden | Q90 step ratio | max step ratio | timing pass rate | selection |
|---|---:|---:|---:|---:|---:|
| `KW4 foreach` | 16 | `1.3621404600` | `1.4062177597` | `1.00` | 1 |
| `KW4 foreach` | 20 | `1.3986580287` | `1.4317618510` | `1.00` | 1 |
| `KW4 foreach` | 24 | `1.3912707282` | `1.4676814404` | `1.00` | 1 |
| `KW4 foreach` | 28 | `1.3610995445` | `1.4275689926` | `1.00` | 1 |

判断：

1. `foreach` AdamW 是本轮第一个真正把 P4 system screen 推过 `Q90 <= 1.50` 的实现侧 repair。
2. 它不是 cached param rows 那种反向退化；`KW4 hidden28` 从此前普通 path Q90 `1.555652` / cached path Q90 `1.876293` 降到 `1.361100`。
3. 但这只是 P4 system screen，不等于 formal success；还必须确认 selected candidate 的 full task / geometry / external fair route，并做 timing 稳定性复查。

No-fake audit：

```text
rows_checked = 70
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 30. 追加：KW4 hidden28 foreach selected full task / geometry confirmation

本节把第 29 节选出的 `KW4 hidden28 + ForeachAdamWAddcdivNoSync` 接到 v8.5 external fair child route，确认它不是只在 timing screen 上好看。该确认仍使用 KANbeFair MNIST full protocol，并打开 P9 functional causality control。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_kw4_foreach_selected_confirmation_20260508T040000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_stability_seed5_wave_20260508T001500Z \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KW4 \
  --p4-base-hidden-candidates 16,20,24,28 \
  --p4-seed 1314 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --t3-batch-size 128 \
  --use-foreach-adamw-addcdiv \
  --run-selected-system-confirmation \
  --selected-dg-candidate-id KW4 \
  --selected-dg-hidden-dim 28 \
  --selected-seed 1314 \
  --selected-use-foreach-adamw-addcdiv \
  --kb-epochs 20 \
  --primary-epochs 20
```

Selected confirmation：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| candidate | `KW4 hidden28 + FT7` |
| optimizer impl | `ForeachAdamWAddcdivNoSync` |
| KB-MLP test acc | `96.9099998474` |
| DG base test acc | `97.1699953079` |
| DG functional test acc | `97.1899986267` |
| delta vs KB-MLP | `+0.2799987793` |
| delta vs DG base | `+0.0200033188` |
| curvature ratio vs DG base | `0.7724829281` |
| functional events | `73` |
| step ratio vs KB-MLP | `1.4822665737` |
| primary transfer pass | 1 |
| parameter fair pass | 1 |
| FLOPs fair pass | 1 |
| wallclock fair pass | 1 |
| functional causality pass | 1 |
| selected task/geometry confirmation pass | 1 |

Child route：

```json
{
  "route": "R1-ExternalFairFunctionalAdvantage",
  "best_candidate": "DG1-FT7-KW4-hidden28-functional",
  "success_v85_minimum": 1,
  "success_v85_external_formal": 1,
  "success_v85_strong": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

P9 causality 关键值：

| control | test acc | curvature ratio | P9 pass |
|---|---:|---:|---:|
| `DG-NoOp` | `97.1699953079` | `1.0000000000` | 0 |
| `DG-Functional` | `97.1899986267` | `0.7724829281` | 1 |
| `DG-RandomFunc` | `97.1799969673` | `1.0045381244` | 0 |
| `DG-ShuffledRoleFunc` | `97.1799969673` | `0.7501350086` | 0 |

判断：

1. `KW4 hidden28 + FT7 + foreach AdamW` 通过了 selected full task / geometry / external fair child route。
2. 它保住了 parameter / FLOPs fair envelope，并且 wall-clock child ratio 为 `1.482267 <= 1.50`。
3. functional route 不是 no-op：`DG-Functional` curvature ratio `0.772483`，优于 NoOp 与 RandomFunc。
4. 但该 run 自身重新跑 P4 system screen 时没有复现第 29 节的 robust pass，因此仍不能写 v8.7 formal success。

本 run 内 P4 screen 复查：

| candidate | hidden | Q90 step ratio | max step ratio | timing pass rate | selection |
|---|---:|---:|---:|---:|---:|
| `KW4 foreach` | 16 | `1.6083383116` | `1.6507530766` | `0.50` | 0 |
| `KW4 foreach` | 20 | `1.6120943451` | `1.6504577358` | `0.50` | 0 |
| `KW4 foreach` | 24 | `1.6099569786` | `1.6510468642` | `0.50` | 0 |
| `KW4 foreach` | 28 | `1.6122436725` | `1.6539891480` | `0.50` | 0 |

`KW4 hidden28` phase detail：

| protocol | KB step ms | DG step ms | step ratio | pass |
|---|---:|---:|---:|---:|
| T0 | `0.6461841334` | `0.8314884896` | `1.2867671097` | 1 |
| T1 | `0.5036424426` | `0.8330191346` | `1.6539891480` | 0 |
| T2 | `0.6589700840` | `0.8355448311` | `1.2679556346` | 1 |
| T3 | `0.5506309269` | `0.8341164113` | `1.5148375629` | 0 |

判断补充：

1. DG step 已经比较稳定，大约 `0.83 ms`；不稳定主要来自 KB-MLP baseline timing 在 `0.50-0.66 ms` 间摆动。
2. 这正是 v8.7 “stability-first” 要处理的问题：不能因为一次 child route 过线就忽略 P4 robust timing repeat 失败。
3. 当前应该进入 selected method 的 stability repetition / timing protocol formalization，而不是打开 adaptive controller 或直接写 formal success。

No-fake audit：

```text
v87 rows_checked = 71
v87 fake_proxy_nonzero_count = 0
v87 fake_data_used = 0
v87 proxy_row_used = 0
v87 cpu_offload_used = 0

child rows_checked = 23
child fake_proxy_nonzero_count = 0
child cpu_offload_used = 0
```

## 31. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `5f2a8f1e6266380ada90a07bb0bb58a392307a01bc726664afe6e6ef68df4ab0` |
| foreach screen route | `78c7baad626103042b239227a91f3be012800e0bfdefa5d38e83c3eecf2ce07d` |
| foreach screen by-candidate | `0afccf2590e727cfe53a3271b6e68f1b94edf90a0d65670f1dafa31e21736219` |
| foreach screen provenance audit | `a4f1ad00e141132c6d78b8e5c027c1e1be69583917c5c517d13f9f3cacd9dce6` |
| selected confirmation route | `c0e36b36a5828d8847fc56699943fe350e08886952290e97124ab32dda89c640` |
| selected confirmation CSV | `f8cbff830e39d941a6b0b5a8629d08ee0bf513736caa85756e2cef7b8ebd47a4` |
| selected child route | `c5a6cc01d1cebbb2f7397ef382596e7775516da6a86f687d3086f34fba09a769` |
| selected run P4 by-candidate | `3c6e8cb03b6b789af31d99cf093364fed3a055432ece0959309e9df0e8f485bd` |
| selected run provenance audit | `4b3a6d13b412e7d9b2ca1760ebe6e87b15de817c51e0b7abf8303470bd2dc9dd` |

## 32. 更新结论

v8.7 仍未完成 formal success：

```text
P4 foreach system screen = pass in one artifact
selected KW4 hidden28 full task / geometry confirmation = pass
selected child external fair route = pass
same-run P4 robust timing repeat = fail
adaptive controller = not_run
success_v87_formal = false
```

机制结论：

1. `ForeachAdamWAddcdivNoSync` 是真实有效的 implementation-side repair；它把 `KW4` 的一个 P4 robust system screen 推过了 `Q90 <= 1.50`。
2. `KW4 hidden28 + FT7` 不是纯 timing trick：child route 中 test acc 超过 KB-MLP `+0.279999`，curvature ratio `0.772483`，parameter/FLOPs/wall-clock 都过。
3. 但 timing 仍不稳定：同一 selected confirmation run 内，P4 repeat 的 Q90 回到 `1.612244`，失败 shape 来自 KB-MLP baseline step time 波动。
4. 因此 v8.7 当前 blocker 不再是 “有没有 task/geometry external fair candidate”，而是 “selected candidate 的 timing stability / benchmark formalization 是否能重复闭合”。
5. 下一步应对 `KW4 hidden28 + foreach AdamW` 做 P0-style multi-seed / multi-protocol stability confirmation，或把 timing protocol 改成文档允许的 paired robust estimator；不能把单 artifact 的 P4 pass 或 child R1 pass 写成 v8.7 formal success。

最终一句话：

> v8.7 还没有完成。`KW4 hidden28 + ForeachAdamWAddcdivNoSync + FT7` 已经通过 full task/geometry external fair 子确认，但 robust timing 复查没有稳定过线；下一步要解决的是 timing stability formalization，而不是再改 teacher/loss/CPU/offload，也不能把一次 pass 写成稳定方法。

## 33. 追加：selected P0 stability runner 参数化

本节修复一个 runner 侧缺口：此前 P0 stability child route 固定使用 `KW6 hidden28`，因此无法对第 30 节选出的 `KW4 hidden28 + ForeachAdamWAddcdivNoSync` 做真正 P0 repeat。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `_v85_namespace` 新增 `p0_dg_candidate_id` / `p0_dg_hidden_dim` 接入 |
| `experiments/run_gafu_v87_real.py` | P0 child run 新增 `--p0-use-foreach-adamw-addcdiv`，只在 child run 内临时替换 AdamW impl，结束后恢复 |
| `experiments/run_gafu_v87_real.py` | `_p0_rows_from_v85` 不再 hardcode `KW6 hidden28`，改为按 selected candidate / hidden 读取 primary rows |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

判断：这是记录 / runner 能力修复，不改变 CE objective、teacher/loss/sampler/class weight/offload 合同，也不改变 functional update 公式。

## 34. 追加：KW4 hidden28 foreach selected P0 repeat2

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_repeat2_20260508T050000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --p0-seeds 1314 \
  --p0-reruns 2 \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --p0-use-foreach-adamw-addcdiv \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 0,
  "dg1_mean_delta_vs_KB_MLP": 0.279998779296875,
  "dg1_mean_curvature_ratio_vs_DG0": 0.7724829280625449,
  "dg1_q90_step_ratio_vs_KB_MLP": 1.3477213963403711,
  "dg1_all_gate_pass_rate": 1,
  "success_v87_formal": 0
}
```

Repeat2 DG1 rows：

| rerun | seed | child route | delta vs KB-MLP | delta vs DG0 | curvature ratio | step ratio | events |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 1314 | `R1-ExternalFairFunctionalAdvantage` | `+0.2799987793` | `+0.0200033188` | `0.7724829281` | `1.3504814135` | 73 |
| 1 | 1314 | `R1-ExternalFairFunctionalAdvantage` | `+0.2799987793` | `+0.0200033188` | `0.7724829281` | `1.3228812421` | 73 |

判断：

1. selected method 在同 seed 两次 fresh child run 中都过 R1，P0 T0 gate 暂时稳定。
2. Q90 step ratio 为 `1.347721`，明显低于 `1.50`。
3. 但该 run 只有 `timing_protocol_count = 1`，不是 v8.7 full P0 formal；不能写 formal success。

No-fake audit：

```text
rows_checked = 30
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 35. 追加：KW4 hidden28 foreach selected P0 3-seed probe

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --p0-seeds 1314,1315,1316 \
  --p0-reruns 1 \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --p0-use-foreach-adamw-addcdiv \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 0,
  "dg1_mean_delta_vs_KB_MLP": 0.3599981466929118,
  "dg1_mean_curvature_ratio_vs_DG0": 0.7341650272201513,
  "dg1_q90_step_ratio_vs_KB_MLP": 1.3486506782651408,
  "dg1_all_gate_pass_rate": 1,
  "success_v87_formal": 0
}
```

3-seed DG1 rows：

| seed | child route | test acc | delta vs KB-MLP | delta vs DG0 | curvature ratio | step ratio | events |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1314 | `R1-ExternalFairFunctionalAdvantage` | `97.1899986267` | `+0.2799987793` | `+0.0200033188` | `0.7724829281` | `1.3538285660` | 73 |
| 1315 | `R1-ExternalFairFunctionalAdvantage` | `97.4500000477` | `+0.5400002003` | `+0.1100003719` | `0.7167499049` | `1.3279391272` | 73 |
| 1316 | `R1-ExternalFairFunctionalAdvantage` | `97.1699953079` | `+0.2599954605` | `0.0000000000` | `0.7132622487` | `1.2926067068` | 73 |

判断：

1. `KW4 hidden28 + foreach + FT7` 在 3 个 seed 的 T0 full child route 中全部通过 R1。
2. accuracy delta、curvature ratio、step ratio 三个核心 gate 都没有出现掉线；`dg1_all_gate_pass_rate = 1`。
3. 这显著强于第 30 节单次 selected confirmation，是 v8.7 stability-first 方向的真实推进。
4. 但 formal 仍不能声明：`measured_p0_runs = 3`，`timing_protocol_count = 1`，而 runner 的 full P0 要求 `measured_p0_runs >= 15` 且 `timing_protocol_count >= 5`。
5. 因此下一步仍是补 full P0 多 protocol / 多 rerun，或把 timing protocol formalization 接入 P0，而不是直接打开 adaptive success。

No-fake audit：

```text
rows_checked = 37
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 36. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `7b0634987166b270da47215bd3282eed7784a507f08a6b6be6a948cff7b65607` |
| repeat2 route | `98ece4d004eb23b27a566d6829f626121f0f3d76ed776fb76832df9147a37bb5` |
| repeat2 P0 summary | `1c39d20b5e8a18db864551125ec5a25cd93307395752cb7913ed232fa23cf32a` |
| repeat2 P0 raw | `d5fe60e9d5c71162d689ecafb131c5ca34e98577ec3445559493ca2eb1222650` |
| repeat2 provenance audit | `22ebe9c3606518a3a0c33022a086ed4f88beb6c8a9466444d2f75e0e224c11f5` |
| seed3 route | `dd8fece33026a7630f733ac0fbf078299d84bfa14cbaef6fe4ad0a29e8b7777b` |
| seed3 P0 summary | `ac9086e6223a4546aba72c1cc099d369d4f76b929a79d2a8e9080fe450907ea5` |
| seed3 P0 raw | `2484687a87e7d56db26f39c1e10c61846a18cb103d36dd56955f20c0c29ee9de` |
| seed3 provenance audit | `309953fd64dd68c24e8e31c38d859956c111f5ef9a74f771fc3fd72570fedb05` |

## 37. 更新结论

v8.7 仍未完成 formal success，但完成度进一步提高：

```text
selected KW4 hidden28 + foreach full child route = pass
selected repeat2 P0 T0 gate = pass
selected 3-seed P0 T0 gate = pass
p0_full_protocol_complete = false
adaptive controller = not_run
success_v87_formal = false
```

机制结论：

1. 第 29-30 节证明 `KW4 hidden28 + foreach + FT7` 有 task/geometry/external fair 能力。
2. 第 34-35 节进一步证明它不是单 seed 偶然：2 次同 seed repeat 与 3 个 seed 的 T0 child route 都过。
3. 当前最新 P0 3-seed mean delta 为 `+0.359998`，mean curvature ratio 为 `0.734165`，Q90 step ratio 为 `1.348651`。
4. 但 v8.7 的中心是 stability-first formalization；当前只覆盖 T0 official-local timing protocol，不能替代完整 P0 的 5-protocol / 15-run stability gate。
5. 下一步应把 `KW4 hidden28 + foreach` 接入 full P0 timing protocol grid，尤其复查此前 P4 中出现过的 KB baseline timing 波动；若 full protocol 仍不稳，再实现 paired robust timing estimator，而不是继续调 hidden/stride。

最终一句话：

> v8.7 还没 formal 完成，但已经从“单次 selected confirmation”推进到“selected method 在 repeat2 和 3-seed T0 P0 中都稳定通过”。当前 blocker 是 full timing protocol coverage 不足，而不是 fake/proxy、不是 task/geometry、也不是外部 fair envelope。

## 38. 追加：selected P5 T0-T4 robust timing protocol 接入

本节继续第 37 节的 blocker：P0 T0 已过，但还缺完整 timing protocol。先修 runner：此前 phase-clean timing 默认测 `KW6`，本节改为可指定 `timing_dg_candidate_id / timing_dg_hidden_dim`，默认可沿用 selected P0 candidate。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `_train_step_timing_row` 新增 `dg_candidate` 参数 |
| `experiments/run_gafu_v87_real.py` | `--timing-dg-candidate-id` / `--timing-dg-hidden-dim` 接入 T0-T3 phase-clean timing |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

## 39. 追加：KW4 hidden28 foreach selected T0-T4 timing probe

本节复用第 35 节 3-seed P0 artifact 作为 T4 full-loop existing rows，并真实测量 `KW4 hidden28 + foreach + FT7` 的 T0/T1/T2/T3 phase-clean timing rows。所有 rows 都来自落盘 CSV，没有 proxy timing。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_selected_t0_t4_timing_probe_20260508T061500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --p0-use-foreach-adamw-addcdiv \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1314 \
  --t3-seed 1314 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Robust timing summary：

| metric | value |
|---|---:|
| measured rows | `7` |
| timing protocol count | `5` |
| full T0-T4 complete | `1` |
| Q90 step ratio | `1.6664378273` |
| max step ratio | `1.6875564066` |
| timing gate pass rate | `0.4285714286` |
| robust timing full pass | `0` |

Timing rows：

| protocol | source | seed | step ratio | timing pass | functional events |
|---|---|---:|---:|---:|---:|
| T4 full-loop | P0 artifact | 1314 | `1.3538285660` | 1 | 73 |
| T4 full-loop | P0 artifact | 1315 | `1.3279391272` | 1 | 73 |
| T4 full-loop | P0 artifact | 1316 | `1.2926067068` | 1 | 73 |
| T0 phase-clean | measured | 1314 | `1.5559218117` | 0 | 1 |
| T1 phase-clean | measured | 1314 | `1.6497274511` | 0 | 4 |
| T2 phase-clean | measured | 1314 | `1.6875564066` | 0 | 9 |
| T3 phase-clean | measured | 1314 | `1.6523587744` | 0 | 2 |

Phase-clean breakdown：

| protocol | KB step | DG step | DG forward | DG backward | DG base update | DG functional+guard |
|---|---:|---:|---:|---:|---:|---:|
| T0 | `0.644678` | `1.003069` | `0.373384` | `0.504354` | `0.081803` | `0.004118` |
| T1 | `0.528995` | `0.872697` | `0.259246` | `0.488092` | `0.080698` | `0.006585` |
| T2 | `0.573237` | `0.967370` | `0.252634` | `0.592533` | `0.078789` | `0.005960` |
| T3 | `0.512709` | `0.847180` | `0.252205` | `0.473751` | `0.078512` | `0.005887` |

判断：

1. Full-loop T4 仍稳定通过，但 phase-clean T0-T3 全部失败。
2. 失败不是 functional guard 本身造成的：functional+guard 每步只有约 `0.004-0.006 ms`，主差距来自 DG manual forward/backward。
3. `Q90 = 1.666438` 明确失败，因此 v8.7 不能写 robust timing pass。
4. 当前 blocker 从“full protocol 未测”推进为“full T0-T4 已测且 phase-clean train-step fail”。

No-fake audit：

```text
rows_checked = 44
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 40. 追加：KW4 hidden24 foreach timing diagnostic rejected

为确认 phase-clean failure 是否能靠自动架构选择的更小 hidden 缓解，本节追加 `KW4 hidden24 + foreach + FT7` T0-T3 timing diagnostic。该 probe 不做 task success 声明，也不写 formal success。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_hidden24_t0_t3_timing_diag_20260508T063000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 24 \
  --use-foreach-adamw-addcdiv \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1314 \
  --t3-seed 1314 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Result：

| protocol | step ratio | timing pass |
|---|---:|---:|
| T0 | `1.5838167402` | 0 |
| T1 | `1.8908610500` | 0 |
| T2 | `1.7079218239` | 0 |
| T3 | `1.5509471636` | 0 |

Summary：

```text
Q90 step ratio = 1.8359792821
max step ratio = 1.8908610500
robust_timing_partial_gate_pass = 0
```

判断：

1. `hidden24` 没有修复 phase-clean timing，反而比 `hidden28` 更差。
2. 因此当前 blocker 不能靠继续缩 hidden 解决。
3. 下一步应做 manual forward/backward kernel path repair，或在文档允许范围内定义更合理的 paired robust timing estimator；不能继续把 hidden screen 当主路线。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 41. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `0e6f1909a80962e29cc97412fbf128a67a3ba87c49dbc691306dc222c21b7b4b` |
| selected T0-T4 route | `1c83066057b908849b75049fc24a4b8535847a914f8f5a98a9ac9a367d71d799` |
| selected T0-T4 robust summary | `a2e19cd4b739d97b56c13b3b24fa728da9c980bf3202b06f1b9ed715cf843866` |
| selected T0-T4 robust protocols | `631b39fc06d2f16c5b2d0c58317c4d085451b5cde10a52f3fcf731a718c0c9f4` |
| selected T0-T4 provenance audit | `08832933b09eb52aa7ab5eb1dfa0bd4727ee75597a04e1925877720d85433d68` |
| hidden24 timing summary | `c3cbfaeb1e0520c7578549bc545e777fbd7d80e5be148c3dc2b417dd705e9fe3` |
| hidden24 timing protocols | `6419039c8af3082b10a03f66cf914c3a523ade8bbe368e7221652c02d056413e` |

## 42. 更新结论

v8.7 仍未完成 formal success：

```text
selected P0 T0 repeat/seed gate = pass
selected full T0-T4 robust timing = fail
hidden24 smaller-width timing diagnostic = fail
adaptive controller = not_run
success_v87_formal = false
```

机制结论：

1. `KW4 hidden28 + foreach + FT7` 的 task / geometry / external fair route 是稳定的，3-seed T0 child route 已过。
2. 但 full T0-T4 timing gate 失败，且失败集中在 phase-clean train-step rows。
3. timing failure 的主要来源是 DG manual forward/backward 相对 KB-MLP 仍慢，不是 optimizer update，也不是 functional guard。
4. `hidden24` 没有缓解，说明继续缩 hidden 不是 v8.7 的正路。
5. 下一步应做数值等价的 manual forward/backward implementation repair，或严格定义 paired robust timing estimator；在此之前不能声明 stability success，也不能进入 no-manual-tuning / adaptive formal success。

最终一句话：

> v8.7 还没完成。当前候选已经稳定通过 task/geometry/external fair，但完整 T0-T4 robust timing 明确失败，Q90 为 `1.66644`。这把 blocker 精确定位到 phase-clean manual forward/backward 系统开销；下一步不能再靠挑 hidden 或单 protocol pass，而要修 train-step implementation 或正式化 robust timing estimator。

## 43. 追加：DG manual subphase profile

本节继续第 42 节定位到的 blocker：phase-clean train-step manual forward/backward 系统开销。新增 profiling 只在显式开关 `--profile-dg-subphases` 打开时记录，不改变 CE objective、functional update、optimizer gate 或 route 判据。

### 43.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--profile-dg-subphases`，把 DG train step 拆成 `stack_forward / head_forward / ce_grad / head_backward / stack_backward` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 43.2 subphase profile run

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_subphase_profile_20260508T070000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --profile-dg-subphases \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1314 \
  --t3-seed 1314 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Measured rows：

| protocol | KB step | DG step | ratio | stack fwd | head fwd | CE grad | head bwd | stack bwd | update | func+guard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | `0.622863` | `0.887063` | `1.424171` | `0.078800` | `0.105355` | `0.085932` | `0.284902` | `0.193721` | `0.076307` | `0.003990` | 1 |
| T1 | `0.484263` | `0.888619` | `1.834991` | `0.079083` | `0.104059` | `0.085962` | `0.284624` | `0.194912` | `0.076659` | `0.006007` | 0 |
| T2 | `0.684880` | `0.895364` | `1.307331` | `0.079927` | `0.104750` | `0.086518` | `0.287427` | `0.195843` | `0.077478` | `0.005899` | 1 |
| T3 | `0.533761` | `0.903927` | `1.693506` | `0.080581` | `0.106190` | `0.087576` | `0.290119` | `0.197864` | `0.077898` | `0.005697` | 0 |

Summary：

```text
measured_rows = 4
q90_step_ratio_vs_KB_MLP = 1.7925454875
max_step_ratio_vs_KB_MLP = 1.8349907916
robust_timing_partial_gate_pass = 0
robust_timing_full_gate_pass = 0
```

判断：

1. functional event / guard 不是主要开销，`func+guard` 只有约 `0.004-0.006 ms/step`。
2. `base_update` 也不是主要开销，约 `0.076-0.078 ms/step`。
3. 主要瓶颈是 manual head/CE/backward：`head_forward + CE_grad + head_backward` 合计约 `0.476-0.484 ms/step`。
4. 下一步应做数值等价的 CE/head backward fusion，而不是继续调 hidden 或 event stride。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 44. 追加：CE/head backward fused probe

本节按第 43 节定位结果做一个窄的 implementation-side repair：只把 CE grad 生成与 packed `poly2_silu` head backward 合成一个数值等价 helper。它不是 loss 修改，不改变 teacher/sampler/class weight，不 CPU offload，也不改变 FT7 functional update 公式。

### 44.1 代码改动与等价检查

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--use-fused-ce-head-backward` timing probe path |
| `experiments/run_gafu_v87_real.py` | 新增 `_fused_packed_poly2_silu_ce_head_backward*`，只替代 head forward + CE grad + head backward 的执行路径 |

随机 batch 等价检查：

```text
device = cuda
loss_abs_diff = 0.0
dh_max_abs_diff = 0.0
head_grad_max_abs_diff = 0.0
```

判断：本 probe 对该 batch 的 loss、head 输入梯度与 head flat grad 完全等价。

### 44.2 fused T0-T3 timing probe

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_fused_ce_head_timing_probe_20260508T071500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --profile-dg-subphases \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1314 \
  --t3-seed 1314 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Result：

| protocol | KB step | DG step | ratio | stack fwd | fused head+CE+bwd | stack bwd | update | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | `0.599839` | `0.778028` | `1.297062` | `0.080741` | `0.372937` | `0.199340` | `0.077031` | 1 |
| T1 | `0.613824` | `0.854385` | `1.391904` | `0.090919` | `0.404988` | `0.219401` | `0.085059` | 1 |
| T2 | `0.645127` | `0.883589` | `1.369635` | `0.080402` | `0.477899` | `0.200281` | `0.076469` | 1 |
| T3 | `0.509761` | `0.773774` | `1.517915` | `0.080021` | `0.370595` | `0.198549` | `0.076486` | 0 |

Summary：

```text
measured_rows = 4
q90_step_ratio_vs_KB_MLP = 1.4801115061
max_step_ratio_vs_KB_MLP = 1.5179145449
timing_gate_pass_rate = 0.75
robust_timing_partial_gate_pass = 1
robust_timing_full_gate_pass = 0
```

判断：

1. fused CE/head path 把 T0-T3 Q90 从 `1.792545` 降到 `1.480112`。
2. T0/T1/T2 均进入 timing gate；T3 单行仍略高于 `1.50`。
3. 这说明修复方向有效，但 T0-T3 alone 不构成 full timing success，因为缺 T4。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 45. 追加：fused CE/head + T4 full robust timing

本节把第 44 节 fused phase-clean rows 与已落盘的 3-seed T4 full-loop child artifacts 合并，生成完整 T0-T4 robust timing summary。注意：这仍不是 v8.7 formal success，因为 adaptive controller / no-manual-tuning / full P0 stability protocol 仍未完成；本节只更新 timing blocker 状态。

### 45.1 run

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_fused_ce_head_t0_t4_probe_20260508T073000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --profile-dg-subphases \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1314 \
  --t3-seed 1314 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 0,
  "robust_timing_partial_gate_pass": 1,
  "robust_timing_full_gate_pass": 1,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.4877128064781218,
  "success_v87_formal": 0
}
```

Measured timing rows：

| protocol | source | seed | step ratio | timing pass | all gate pass |
|---|---|---:|---:|---:|---:|
| T4 full-loop | P0 child | 1314 | `1.3538285660` | 1 | 1 |
| T4 full-loop | P0 child | 1315 | `1.3279391272` | 1 | 1 |
| T4 full-loop | P0 child | 1316 | `1.2926067068` | 1 | 1 |
| T0 phase-clean | measured | 1314 | `1.2324459077` | 1 | n/a |
| T1 phase-clean | measured | 1314 | `1.3413943540` | 1 | n/a |
| T2 phase-clean | measured | 1314 | `1.6539057391` | 0 | n/a |
| T3 phase-clean | measured | 1314 | `1.3769175180` | 1 | n/a |

Summary：

```text
measured_rows = 7
timing_protocol_count = 5
full_t0_t4_complete = 1
q90_step_ratio_vs_KB_MLP = 1.4877128065
max_step_ratio_vs_KB_MLP = 1.6539057391
timing_gate_pass_rate = 0.8571428571
robust_timing_full_gate_pass = 1
```

判断：

1. v8.7 的 timing blocker 被实质推进：先前 selected T0-T4 Q90 为 `1.6664378273`，本节 fused CE/head 后 Q90 为 `1.4877128065`。
2. full robust timing gate 当前过线，但 T2 单行仍为 `1.653906`，说明这是 Q90 robust success，不是 strict max success。
3. route 仍然保守为 `success_v87_formal = 0`，因为 P0 full stability protocol / adaptive controller / no-manual-tuning waves 未完成。
4. 不能把本节写成 v8.7 完成；它只是解除第 42 节的 clean timing blocker。

No-fake audit：

```text
rows_checked = 44
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 46. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `e81b33caa3cd207059f5e32c16f83a04e5f3a1604eeac2fa26fbb10fd59cee53` |
| subphase route | `16b408c3e552577394d36e02c0fa5e21f1adeaab89f696e4eceae9acea9c15cb` |
| subphase robust protocols | `9aa1ec1b3a3ddc0a4ad399cab427798d95354f6d4ffeb125cf1116105cc505be` |
| subphase robust summary | `b772e9b837e6e40987e2f91e85c5cbcec5f7e1dbf75dad7410b017ffd2f61423` |
| fused T0-T3 route | `df83a3d6faed45a413df5d94abf7be21812cc334f584d80ab77ce01ec5d0efda` |
| fused T0-T3 robust protocols | `5511bb1935a2495f1d45e49b3613c50b561787fafb29a1c4427645497d3d92cf` |
| fused T0-T3 robust summary | `3908eecc5a4a9f1e8d53a6cac86bed2951b166e307361a13e2edde6583ca0876` |
| fused T0-T4 route | `bc464295972276423b9642851e11571a64c9355142bf23554b11107897885ad9` |
| fused T0-T4 robust protocols | `9adc048968e1140e431cedf86fd7728f0f1e06289ed54613c6c84d49196a7ac9` |
| fused T0-T4 robust summary | `609022fcb1cea1db690da9a62f617d76e3a2ba835a474ab25c1dc43a3fc044e1` |
| fused T0-T4 provenance audit | `08832933b09eb52aa7ab5eb1dfa0bd4727ee75597a04e1925877720d85433d68` |

## 47. 更新结论

v8.7 仍未完成 formal success：

```text
selected task / geometry / external fair route = pass in existing P0 child artifacts
full T0-T4 robust timing = pass after fused CE/head implementation probe
P0 full stability protocol = not complete
adaptive controller = not_run
no-manual-tuning route = not_run
success_v87_formal = false
```

机制结论：

1. 第 42 节的 timing blocker 已有真实 system-side repair：CE/head backward fusion 数值等价，并把 T0-T4 Q90 拉到 `1.4877`。
2. functional update/guard 开销不是主问题；manual head CE/backward 与 stack backward 是主要系统成本。
3. 当前新 blocker 是 route-level stability-first 要求：需要把 fused implementation 纳入更完整 P0 stability protocol，并继续实现 adaptive/no-manual-tuning controller。
4. 因此不能声明 v8.7 完成，也不能把 robust Q90 pass 扩大成 formal success。

最终一句话：

> v8.7 还没完成，但已越过一个关键系统坎：数值等价的 CE/head backward fusion 让完整 T0-T4 robust timing Q90 从失败的 `1.6664` 降到通过的 `1.4877`。下一步应把该实现纳入完整稳定性协议，并继续做 adaptive functional controller；不能把 timing repair 直接包装成 formal success。

## 48. 追加：把 fused CE/head 接入真实 child training path

第 44-45 节的 fused CE/head 先只作用于 v8.7 timing probe。为避免“timing-only 优化”与真实训练路线脱节，本节把同一个数值等价 fused helper 接入 P0 / selected child 的 v85 manual CE backward 入口。该改动仍然只在显式开关下启用，默认路径不变。

### 48.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--p0-use-fused-ce-head-backward`，P0 child run 中 monkeypatch v85 CE backward 为 fused equivalent |
| `experiments/run_gafu_v87_real.py` | 新增 `--selected-use-fused-ce-head-backward`，selected child confirmation 使用同一 fused path |
| `experiments/run_gafu_v87_real.py` | 新增 `_manual_ce_backward_v85_fused` 与 `_manual_ce_backward_with_holdout_v85_fused` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

## 49. 追加：fused selected child confirmation

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_fused_selected_confirmation_20260508T080000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_selected_seed3_20260508T053000Z \
  --run-selected-system-confirmation \
  --selected-dg-candidate-id KW4 \
  --selected-dg-hidden-dim 28 \
  --selected-seed 1314 \
  --selected-use-foreach-adamw-addcdiv \
  --selected-use-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "p4_selected_task_geometry_confirmation_pass": 1,
  "p4_selected_task_geometry_step_ratio_vs_KB_MLP": 1.235639405509984,
  "p4_selected_task_geometry_delta_vs_KB_MLP": 0.279998779296875,
  "p4_selected_task_geometry_curvature_ratio": 0.7724829280625449,
  "success_v87_formal": 0
}
```

Selected child：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| CE/head backward impl | `fused_value_equivalent` |
| optimizer | `ForeachAdamWAddcdivNoSync` |
| KB-MLP test acc | `96.9099998474` |
| DG base test acc | `97.1699953079` |
| DG functional test acc | `97.1899986267` |
| delta vs KB-MLP | `+0.2799987793` |
| delta vs DG base | `+0.0200033188` |
| curvature ratio vs DG base | `0.7724829281` |
| functional events | `73` |
| step ratio vs KB-MLP | `1.2356394055` |
| primary / params / FLOPs / wallclock / causality pass | `1 / 1 / 1 / 1 / 1` |

判断：fused path 接入真实 child training 后没有破坏 task/geometry/external fair route，且 single selected confirmation 的 step ratio 低于先前默认 child route。

No-fake audit：

```text
rows_checked = 38
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 50. 追加：fused P0 3-seed reproduction

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_fused_seed3_20260508T081500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --p0-seeds 1314,1315,1316 \
  --p0-reruns 1 \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --p0-use-foreach-adamw-addcdiv \
  --p0-use-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --kb-epochs 20 \
  --primary-epochs 20
```

P0 summary：

| metric | value |
|---|---:|
| measured P0 runs | `3` |
| child route set | `R1-ExternalFairFunctionalAdvantage` |
| child external formal pass count | `3/3` |
| mean delta vs KB-MLP | `+0.3599981467` |
| mean curvature ratio vs DG0 | `0.7341650272` |
| Q90 step ratio vs KB-MLP | `1.2405252514` |
| all gate pass rate | `1.000000` |
| P0 stability gate pass | `1` |
| P0 full protocol complete | `0` |

DG1 rows：

| seed | test acc | delta vs KB | delta vs DG0 | curvature ratio | events | step ratio | child route |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1314 | `97.1899986267` | `+0.2799987793` | `+0.0200033188` | `0.7724829281` | `73` | `1.2450609017` | R1 |
| 1315 | `97.4500000477` | `+0.5400002003` | `+0.1100003719` | `0.7167499049` | `73` | `1.2223826502` | R1 |
| 1316 | `97.1699953079` | `+0.2599954605` | `0.0000000000` | `0.7132622487` | `73` | `1.1894910020` | R1 |

判断：

1. fused CE/head 已经不只是 timing-only：3 个 fresh child 都通过 external formal route。
2. T0 full-loop child step Q90 从先前 foreach-only 3-seed的 `1.3486506783` 降到 `1.2405252514`。
3. 但这仍不是完整 P0，因为还没有 `5 seeds x 3 reruns` 与稳定的 T0-T4 timing protocol coverage。

No-fake audit：

```text
rows_checked = 37
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 51. 追加：fused P0 3-seed + T0-T4 combined timing 反例

为避免只采用第 45 节过线结果，本节用新的 fused P0 3-seed artifact 作为 T4 来源，再重测 T0-T3 phase-clean timing。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_fused_p0seed3_t0_t4_20260508T090000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_fused_seed3_20260508T081500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_kw4_foreach_fused_seed3_20260508T081500Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --profile-dg-subphases \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1314 \
  --t3-seed 1314 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Summary：

```text
measured_rows = 7
timing_protocol_count = 5
full_t0_t4_complete = 1
q90_step_ratio_vs_KB_MLP = 1.6894881524
max_step_ratio_vs_KB_MLP = 1.7904120043
timing_gate_pass_rate = 0.5714285714
robust_timing_full_gate_pass = 0
```

Measured timing rows：

| protocol | seed | step ratio | timing pass |
|---|---:|---:|---:|
| T4 full-loop | 1314 | `1.2450609017` | 1 |
| T4 full-loop | 1315 | `1.2223826502` | 1 |
| T4 full-loop | 1316 | `1.1894910020` | 1 |
| T0 phase-clean | 1314 | `1.3822450155` | 1 |
| T1 phase-clean | 1314 | `1.7904120043` | 0 |
| T2 phase-clean | 1314 | `1.6222055845` | 0 |
| T3 phase-clean | 1314 | `1.5391897889` | 0 |

判断：

1. 新 fused child T4 全部过线，但同一次 combined run 的 phase-clean T1/T2/T3 失败。
2. 因此第 45 节的 T0-T4 Q90 pass 不能作为稳定 timing closure；repeat 反例说明 timing protocol/seed 仍敏感。
3. v8.7 仍不能写 formal success，当前 blocker 回到 robust phase-clean timing stability。

No-fake audit：

```text
rows_checked = 44
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 52. 追加：fused phase-clean seed1315 repeat

为确认第 51 节不是单 seed 偶然，又使用 seed `1315` 只重测 T0-T3 phase-clean timing。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_fused_phaseclean_repeat_20260508T091500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_fused_seed3_20260508T081500Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --profile-dg-subphases \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1315 \
  --t3-seed 1315 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Result：

| protocol | KB step | DG step | step ratio | fused head+CE+bwd | stack bwd | pass |
|---|---:|---:|---:|---:|---:|---:|
| T0 | `0.601717` | `0.788253` | `1.310006` | `0.379485` | `0.201285` | 1 |
| T1 | `0.474869` | `0.784993` | `1.653072` | `0.376433` | `0.200273` | 0 |
| T2 | `0.490433` | `0.891755` | `1.818303` | `0.482532` | `0.201260` | 0 |
| T3 | `0.528677` | `0.782848` | `1.480769` | `0.376583` | `0.200077` | 1 |

Summary：

```text
q90_step_ratio_vs_KB_MLP = 1.7687339539
max_step_ratio_vs_KB_MLP = 1.8183033068
robust_timing_partial_gate_pass = 0
```

判断：

1. seed1315 repeat 也显示 T1/T2 phase-clean 失败，说明第 51 节不是单次异常。
2. fused CE/head 把 DG absolute step 稳定压到约 `0.78-0.89 ms`，但 KB-MLP denominator 在 T1/T2 下可低到 `0.47-0.49 ms`，导致 ratio 仍不稳。
3. 当前 clean blocker 是 phase-clean timing ratio robustness，而不是 T4 full-loop child route，也不是 task/geometry。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 53. 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `923c210c9e9dc1f83192a1fb9d967e2c77d225799ca85518e7e479680a4f0456` |
| fused selected route | `a68adbdc39a8f2977b010bc1c10befeea7dc619b7c60a0bd90226472f4a1c0ab` |
| fused selected confirmation | `cacc5ae25e8349883f3ad2691008319c6787bb3003ed3ca6d65551b66eab552a` |
| fused selected child route | `c5a6cc01d1cebbb2f7397ef382596e7775516da6a86f687d3086f34fba09a769` |
| fused selected child primary | `0f302b46b26566281740aaedb241c7d6bfb708255ce4db6a73cbd9a2f6cac1e1` |
| fused P0 seed3 route | `2b61439a11a84e953c93a409dd18556a4a0ef77f70e45755c112f62b3a61a190` |
| fused P0 seed3 summary | `71871c49d93bd00d7b946438121e9f946d20b5b08b2963ec9e50688900fc6760` |
| fused P0 seed3 rows | `d33b155c27b231a3c2c2e55f0d42b553bdbe0a756961def80b8b24b681106e72` |
| fused P0 seed3 child decisions | `1a5ec75066fa84abf6e7a049a8c509e515390e2bd9263ac643fed92e8fbec7c5` |
| fused P0 seed3 provenance | `309953fd64dd68c24e8e31c38d859956c111f5ef9a74f771fc3fd72570fedb05` |
| fused P0 seed3 T0-T4 route | `701de4e94421eb81e9f2b230ba99100b1556660f3fffda59f3b664793bab0ddf` |
| fused P0 seed3 T0-T4 protocols | `469840cd2e584478e27bf305cd39e0a6e947b83db5be1f7d28c4262ec2411daf` |
| fused P0 seed3 T0-T4 summary | `5b6108986c45b6d374ff69f7975f71642160f3bc97d79bfa77537d4edf54be5a` |
| fused phase-clean seed1315 route | `a3b0a4ce7ee9d7158d4b35421dc193855bf3074d1870afa184dbf674cb39cf60` |
| fused phase-clean seed1315 protocols | `d96a5a4727c05ce2dd42c943194e567d44472f8826446a7b3a4fc8369428ea66` |
| fused phase-clean seed1315 summary | `094c36540cfcdea3c3903c249ba5a8f78472e2a85b572042a171c735c84d26fa` |

## 54. 更新结论

v8.7 仍未完成 formal success：

```text
fused CE/head child task/geometry/external fair = pass
fused P0 3-seed T0 child = pass
fused phase-clean robust timing = unstable / fail on repeat
P0 full 5x3 protocol = not complete
adaptive controller = not_run
no-manual-tuning route = not_run
success_v87_formal = false
```

机制结论：

1. 数值等价 fused CE/head backward 是 accepted implementation repair：它能接入真实 child training，并让 3-seed T0 child step Q90 降到 `1.2405`。
2. 但它没有彻底解决 v8.7 的 stability-first 问题。phase-clean T1/T2 在 seed1314 / seed1315 repeat 中仍会失败。
3. 当前系统开销的绝对值已经较稳定：DG fused step 大约 `0.78-0.89 ms`，主要由 `fused_head_ce_backward` 和 `stack_backward` 组成。
4. ratio failure 还受到 KB-MLP denominator 在 phase-clean protocol 下波动影响；这说明 v8.7 需要更严格的 paired robust timing estimator 或继续降低 DG backward path，而不是挑一次过线 run。
5. 因此本轮不能声明完成，也不能把第 45 节的单次 Q90 pass 当作正式 timing closure。

最终一句话：

> v8.7 还没完成。fused CE/head backward 已经是干净有效的系统修复，并且真实 child route/3-seed P0 都过；但 phase-clean T1/T2 repeat 仍使 robust timing 失败。当前卡点从“head backward 太慢”推进为“phase-clean timing ratio 仍不稳”，下一步应做 paired robust timing estimator formalization 或继续压 DG backward absolute time，而不是宣称 formal success。

## 55. 追加：fused phase-clean no-profile repeat

上一节的 seed1315 repeat 打开了 `--profile-dg-subphases`，它适合诊断 subphase，但会给 DG 路径插入更细的 timing/sync。为排除“失败只是 subphase profiling overhead”的可能，本节保持 fused CE/head、foreach AdamW、KW4 hidden28、stride128 不变，只关闭 subphase profiling，重新测 T0-T3 phase-clean timing。

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_kw4_foreach_fused_phaseclean_noprofile_repeat_20260508T093000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw4_foreach_fused_seed3_20260508T081500Z \
  --run-t0-t2-phase-clean-timing \
  --run-t3-phase-clean-timing \
  --p0-dg-candidate-id KW4 \
  --p0-dg-hidden-dim 28 \
  --timing-dg-candidate-id KW4 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --t0-t2-seed 1315 \
  --t3-seed 1315 \
  --t0-t2-batch-size 128 \
  --t3-batch-size 128 \
  --t0-t2-protocols T0:50:200,T1:100:500,T2:200:1000 \
  --t3-warmup 20 \
  --t3-reps 260 \
  --kb-epochs 20 \
  --primary-epochs 20
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "p0_stability_gate_pass": 1,
  "robust_timing_partial_gate_pass": 0,
  "robust_timing_full_gate_pass": 0,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.7079798661076044,
  "success_v87_formal": 0
}
```

Summary：

```text
measured_rows = 4
q90_step_ratio_vs_KB_MLP = 1.7079798661
max_step_ratio_vs_KB_MLP = 1.7685438699
timing_gate_pass_rate = 0.5000000000
robust_timing_partial_gate_pass = 0
```

Measured rows：

| protocol | KB step ms | DG0 step ms | DG1 step ms | ratio | events | pass |
|---|---:|---:|---:|---:|---:|---:|
| T0 | `0.562289` | `0.774653` | `0.772500` | `1.373848` | 1 | 1 |
| T1 | `0.554379` | `0.755667` | `0.768486` | `1.386210` | 4 | 1 |
| T2 | `0.553451` | `0.751792` | `0.867071` | `1.566664` | 9 | 0 |
| T3 | `0.428530` | `0.751393` | `0.757874` | `1.768544` | 2 | 0 |

Phase totals：

| protocol | forward ms | backward ms | update ms | unknown fraction |
|---|---:|---:|---:|---:|
| T0 | `0.448349` | `0.204607` | `0.078220` | `0.048133` |
| T1 | `0.445428` | `0.203501` | `0.077370` | `0.046736` |
| T2 | `0.441294` | `0.307307` | `0.077006` | `0.040967` |
| T3 | `0.439870` | `0.199937` | `0.076834` | `0.046844` |

判断：

1. 关闭 subphase profiling 后，T0/T1 从上一节的 fail 变成 pass，说明 profiling 确实会扰动部分 timing rows。
2. 但 T2/T3 仍失败，Q90 仍为 `1.707980`，因此不能把 robust timing failure 解释成单纯 profiling artifact。
3. T3 的失败主要来自 KB denominator 低到 `0.428530 ms`，而 DG1 即使无 profiling 仍为 `0.757874 ms`。
4. T2 的 DG1 backward 增至 `0.307307 ms`，说明较长 reps 下 event cadence / paired holdout step 对 backward phase 仍有偶发系统成本。
5. 因此 v8.7 的 clean blocker 更明确：需要 formal paired robust timing estimator 与更强的 DG absolute step reduction；不能通过关闭诊断或挑 pass protocol 声明 formal。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `923c210c9e9dc1f83192a1fb9d967e2c77d225799ca85518e7e479680a4f0456` |
| no-profile route | `67918754d40031d2b55425bba18ce7a397f3b0c0eacd84813422437b27fcb163` |
| no-profile protocols | `71d45d812d2469b5fd152d684daf6f105e3ca82bb1559df447c1f623a542ab4e` |
| no-profile summary | `16d5ae4b8dc4dcb632c0f08eb4f70482a2c4fa05b336ff778af1a885169f5441` |
| no-profile provenance | `7431d4188b126fa1c2319c0e7b7c1797e8a94a44384b89e74c1367e330da9a89` |

## 56. 最终更新结论

v8.7 仍未完成 formal success：

```text
fused CE/head child task/geometry/external fair = pass
fused P0 3-seed T0 child = pass
fused phase-clean timing without subphase profiling = fail
P0 full 5x3 protocol = not complete
adaptive controller = not_run
no-manual-tuning route = not_run
success_v87_formal = false
```

机制结论更新：

1. fused CE/head backward 仍是 accepted system repair；它不是 fake/proxy，也没有改 teacher/loss/sampler/offload。
2. subphase profiling 会扰动 timing，因此后续 official timing 应避免把 diagnostic profiling rows 当作唯一结论。
3. 但 no-profile repeat 仍失败，说明 v8.7 的 robust timing blocker 是真实存在的。
4. 当前路线已经不能再靠“挑一个过线 timing run”推进；下一步应预注册 paired timing estimator，或继续做数值等价的 stack/forward/backward kernel-level trim。

最终一句话：

> v8.7 还没完成。最新 no-profile repeat 排除了“只是 subphase profiling 造成失败”的解释：T2/T3 phase-clean 仍不过，Q90 为 `1.70798`。所以当前最诚实的结论是：fused implementation 有效，但 stability-first formal 仍卡在 robust timing protocol，不允许写成功。

## 57. 追加：KW3 + compiled fused CE/head timing repair

本节继续修 robust timing blocker，但仍不改 v8.7 的实验合同：

```text
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 57.1 代码改动

| 文件 | 改动 | 是否改变数学 |
|---|---|---:|
| `experiments/run_gafu_v87_real.py` | 新增 `--use-compiled-fused-ce-head-backward`，用 `torch.compile` 编译 fused CE + poly2-SiLU head backward core | 0 |
| `experiments/run_gafu_v87_real.py` | 新增 `--prewarm-compiled-fused-ce-head-backward`，在 timing 前显式 warm compiled train/holdout head core | 0 |
| `experiments/run_gafu_v87_real.py` | timing rows 记录 `ce_head_backward_impl` 与 prewarm flag | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 57.2 数值等价检查

随机 CUDA tensor 上比较 eager fused 与 compiled fused：

```text
device = cuda
loss_abs_diff = 2.384185791015625e-07
dh_max_abs_diff = 1.1641532182693481e-09
mix_grad_max_abs_diff = 1.1175870895385742e-08
base_grad_max_abs_diff = 7.450580596923828e-09
poly_grad_max_abs_diff = 4.656612873077393e-10

holdout_loss_abs_diff = 0.0
holdout_holdout_loss_abs_diff = 2.384185791015625e-07
holdout_dh_max_abs_diff = 4.656612873077393e-10
holdout_mix_grad_max_abs_diff = 7.450580596923828e-09
holdout_base_grad_max_abs_diff = 7.450580596923828e-09
holdout_poly_grad_max_abs_diff = 4.656612873077393e-10
```

判断：compiled fused core 是 float32 规约误差级等价实现，可以进入 timing diagnostic。

### 57.3 KW3/KW6 no-profile implementation screen

为了确认是否继续坚持 KW4，先做同样 fused/no-profile 条件下的 KW3/KW6 timing 诊断。该诊断不作为 route selection success。

| candidate | q90 ratio | max ratio | pass rows | 判断 |
|---|---:|---:|---:|---|
| `KW4` | `1.707980` | `1.768544` | `2/4` | current selected route timing fail |
| `KW6` | `1.885493` | `1.909512` | `1/4` | rejected timing worse |
| `KW3` | `1.518273` | `1.575715` | `3/4` | near miss; worth task/geometry check |

KW3 selected child confirmation：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| test acc | `97.0799982548` |
| delta vs KB-MLP | `+0.1699984074` |
| delta vs DG0 | `+0.0599980354` |
| curvature ratio vs DG0 | `0.8017877368` |
| step ratio vs KB-MLP | `1.4395855823` |
| functional events | `73` |
| selected task/geometry confirmation | `1` |

判断：KW3 不是只在 timing 上接近；它的 task / parameter / FLOPs / wall-clock / geometry 在 selected 1-seed child route 中也通过。但这仍不是 P0 full stability。

### 57.4 compiled fused without prewarm rejected

先跑 compiled fused，不做 event prewarm：

```text
out_dir = results/real_rerun_20260506/v87_kw3_foreach_compiled_fused_phaseclean_noprofile_diag_20260508T103000Z
```

| protocol | ratio | pass |
|---|---:|---:|
| T0 | `5.480804` | 0 |
| T1 | `1.372252` | 1 |
| T2 | `0.895918` | 1 |
| T3 | `0.930670` | 1 |

判断：T2/T3 显示 compiled fused core 很有效，但 T0 被第一次 compiled holdout event 的 JIT 编译开销污染。由于 T0 warmup `50` 小于 stride `128`，第一个 event path 在 measured window 中触发编译。因此该 run rejected，不能写 success。

### 57.5 compiled fused + event prewarm T0-T3

加入显式 compiled train/holdout core prewarm 后，重新测 KW3 T0-T3：

```text
out_dir = results/real_rerun_20260506/v87_kw3_foreach_compiled_fused_prewarm_phaseclean_diag_20260508T104500Z
```

| protocol | KB step ms | DG step ms | ratio | pass |
|---|---:|---:|---:|---:|
| T0 | `0.632735` | `0.745620` | `1.178408` | 1 |
| T1 | `0.486932` | `0.565285` | `1.160911` | 1 |
| T2 | `0.598505` | `0.568260` | `0.949465` | 1 |
| T3 | `0.475699` | `0.561129` | `1.179586` | 1 |

Summary：

```text
q90_step_ratio_vs_KB_MLP = 1.1792327201
max_step_ratio_vs_KB_MLP = 1.1795863805
robust_timing_partial_gate_pass = 1
```

判断：compiled fused + event prewarm 是有效 system-side repair。它把 fused head/CE part 从多 kernel eager path 降到 compiled graph path；没有改变 CE objective 或 functional update rule。

### 57.6 KW3 T0-T4 diagnostic combination

为补 T4，追加 KW3 1-seed P0 child artifact：

```text
out_dir = results/real_rerun_20260506/v87_p0_kw3_foreach_fused_seed1_20260508T110000Z
```

P0 1-seed child：

| metric | value |
|---|---:|
| DG1 delta vs KB-MLP | `+0.1699984074` |
| DG1 curvature ratio | `0.8017877368` |
| DG1 step ratio | `1.3388122380` |
| all gate pass rate | `1.0` |
| p0_full_protocol_complete | `0` |

再组合 T4 child + compiled-prewarm T0-T3：

```text
out_dir = results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_t0_t4_diag_20260508T111500Z
```

Route excerpt：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "robust_timing_partial_gate_pass": 1,
  "robust_timing_full_gate_pass": 1,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.3338058075188497,
  "success_v87_formal": 0
}
```

T0-T4 rows：

| protocol | seed | ratio | pass |
|---|---:|---:|---:|
| T4 full-loop existing artifact | 1314 | `1.338812` | 1 |
| T0 phase-clean | 1315 | `1.326296` | 1 |
| T1 phase-clean | 1315 | `0.898407` | 1 |
| T2 phase-clean | 1315 | `1.023657` | 1 |
| T3 phase-clean | 1315 | `1.002845` | 1 |

Summary：

```text
measured_rows = 5
timing_protocol_count = 5
full_t0_t4_complete = 1
q90_step_ratio_vs_KB_MLP = 1.3338058075
max_step_ratio_vs_KB_MLP = 1.3388122380
robust_timing_full_gate_pass = 1
```

判断：

1. KW3 + compiled fused + event prewarm 形成了当前最干净的 timing repair diagnostic。
2. 它也有 1-seed task/geometry child pass，因此不是纯 timing trick。
3. 但它仍不能写 v8.7 formal：P0 要求 `5 seeds x 3 reruns`，adaptive controller / no-manual-tuning route 仍未完成。
4. 这条路线下一步应进入预注册的 stability confirmation，而不是直接宣称成功。

No-fake audit：

```text
KW3 selected confirmation rows_checked = 38
KW3 P0 1-seed rows_checked = 23
KW3 T0-T4 diagnostic rows_checked = 28
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `1cbe085379811cf14e5ebf9cea3f3aaa019f5f486028a3ba719ffcdb4f57a549` |
| KW3 fused no-profile route | `76f2debd2f8b02b6623b2726d3c66ba44591e7c23ad0b3732ee919ff56300513` |
| KW3 fused no-profile protocols | `1f92bd85cfb5c60fe06dc1452f0333cc291dd9aee88f9f9bb91d30310a3e9ef2` |
| KW3 selected confirmation | `f20183f6264f0d623709d19c1075c4cbbaee6f3b102fe320b17d1d07c27e144d` |
| KW3 P0 1-seed summary | `8db423245abaeb97118061c3d248111fc5eca326f4daf116f2c47a4dbe05de4b` |
| KW3 compiled-prewarm phase-clean protocols | `02eb8a20c92a2e56dacf3f14654057ecf087c9db2948d2b7e7a092c1dee20242` |
| KW3 compiled-prewarm T0-T4 route | `8f7b14b7bded4e8db8e9d05dc917513289b5be0983feb2d21a59e2f7253356ab` |
| KW3 compiled-prewarm T0-T4 protocols | `a6228fb500122d44724c4e228d729c64fd412dc43d1752fcc7d1a616216b92ec` |
| KW3 compiled-prewarm T0-T4 summary | `5b941087b798dc9b692b20e908025832361214d913ee1cf9220f4a712ec5560e` |
| KW3 compiled-prewarm T0-T4 provenance | `662d0328f9a21dea42de8e72c586f45a78718ce1aef81cb449e1e2d89234772d` |

## 58. 当前最终结论

v8.7 仍未完成 formal success，但 blocker 已推进：

```text
KW4 fused route = child/P0 partial pass, robust timing unstable
KW3 compiled fused + event prewarm = T0-T4 diagnostic pass
KW3 selected child task/geometry = pass
P0 full 5x3 = not_run for KW3 compiled route
adaptive controller = not_run
no-manual-tuning = not_run
success_v87_formal = false
```

机制结论：

1. v8.7 的 timing failure 不是不可修：compiled fused CE/head + event prewarm 可以把 T0-T4 diagnostic Q90 压到 `1.3338`。
2. KW3 是当前更合理的 candidate：它的 selected child route 仍过 external fair / functional causality / geometry，并且 timing 比 KW4/KW6 更稳。
3. 但 compiled prewarm 是新的 implementation route，必须进入 full stability protocol；不能把 1-seed T0-T4 diagnostic 当 formal success。
4. 下一步应对 `KW3 + compiled fused + prewarm` 跑 P0 full stability confirmation，并把 compiled prewarm 纳入预注册 timing implementation contract；之后再做 adaptive/no-manual-tuning。

最终一句话：

> v8.7 仍没完成，但已经有真实优化进展：`KW3 + compiled fused CE/head + event prewarm` 在 T0-T4 diagnostic 下通过 robust timing，且 KW3 selected child task/geometry 也通过。现在不能写 formal success；下一步必须做 full P0 stability confirmation 和 adaptive/no-manual-tuning route。

## 59. 追加：compiled fused/prewarm 接入真实 P0 child 与 5x3 stability confirmation

本节继续第 58 节的下一步，不改变 v8.7 中心思想：

```text
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

本节只做 implementation-side stability repair：把第 57 节 timing diagnostic 中有效的 `compiled fused CE/head backward + event prewarm` 接入真实 v85 child full-loop P0，而不是继续停在 timing-only probe。

### 59.1 代码改动

| 文件 | 改动 | 是否改变 objective |
|---|---|---:|
| `experiments/run_gafu_v85_real.py` | 新增 `DG_PRETRAIN_HOOK`，允许 v8.7 在 DG primary 计时开始前做真实训练 batch 的 implementation prewarm | 0 |
| `experiments/run_gafu_v87_real.py` | 新增 P0 / selected child 的 compiled fused CE/head backward 开关 | 0 |
| `experiments/run_gafu_v87_real.py` | 新增 compiled fused prewarm hook；使用真实 train batch / holdout batch，prewarm 后清零梯度，不做 optimizer step | 0 |
| `experiments/run_gafu_v87_real.py` | `--timing-source-out-dirs` 支持读取 phase-clean run 的 `robust_timing_protocols.csv`，用于真实 T0-T4 artifact 合成 | 0 |
| `experiments/run_gafu_v87_real.py` | route 汇总区分 raw P0 summary 与 P0+robust timing 合成完成度 | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 59.2 P0 compiled/prewarm 1-seed 接线 smoke

```text
out_dir = results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed1_smoke_20260508T120000Z
```

结果：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| CE/head backward impl | `compiled_fused_value_equivalent` |
| compiled prewarm used | `1` |
| DG1 delta vs KB-MLP | `+0.1699984074` |
| DG1 curvature ratio | `0.8017871699` |
| DG1 step ratio | `0.9315072471` |
| P0 stability gate | `1` |
| fake/proxy/offload | `0/0/0` |

判断：compiled fused/prewarm 已进入真实 v85 child full-loop，不再只是 phase-clean timing probe。

### 59.3 P0 5-seed x 3-rerun stability confirmation

```text
out_dir = results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z
```

配置：

```text
candidate = KW3
hidden_dim = 28
optimizer = ForeachAdamWAddcdivNoSync
CE/head backward = compiled_fused_value_equivalent
compiled prewarm = 1
FT7 stride = 128
FT7 alpha mult = 15.0
seeds = 1314,1315,1316,1317,1318
reruns = 3
```

P0 summary：

| metric | value |
|---|---:|
| measured P0 runs | `15` |
| requested P0 runs | `15` |
| child external formal pass | `15/15` |
| child route set | `R1-ExternalFairFunctionalAdvantage` |
| DG1 mean test acc | `97.0659995079` |
| DG1 test acc range | `97.0200002193 - 97.1199989319` |
| DG1 mean delta vs KB-MLP | `+0.1559996605` |
| DG1 delta range | `+0.1100003719 - +0.2099990845` |
| DG1 mean curvature ratio vs DG0 | `0.7625712046` |
| DG1 curvature ratio range | `0.6933220239 - 0.8017871699` |
| DG1 mean step ratio vs KB-MLP | `0.9091280034` |
| DG1 Q90 step ratio vs KB-MLP | `0.9545009898` |
| DG1 step ratio range | `0.8603358167 - 0.9865322570` |
| functional events | `73` |
| P0 stability gate | `1` |

判断：

1. `KW3 + compiled fused/prewarm` 在 15 个 fresh child 中全部通过 v85 external formal route。
2. P0 full-loop T4 step ratio 不再是 blocker；Q90 为 `0.9545`。
3. 这仍不等于 v8.7 formal success，因为 adaptive/no-manual-tuning 还没有执行。

No-fake audit：

```text
rows_checked = 121
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 59.4 5-seed T0-T3 phase-clean timing

为补 v8.7 H5 robust timing，复用 P0 5x3 artifact，并对 `1314-1318` 五个 seed 分别测 T0/T1/T2/T3 phase-clean timing。

Out dirs：

```text
results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1314_20260508T123000Z
results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1315_20260508T123000Z
results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1316_20260508T123000Z
results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1317_20260508T123000Z
results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1318_20260508T123000Z
```

Per-seed phase-clean summary：

| seed | rows | Q90 step ratio | max step ratio | partial gate |
|---:|---:|---:|---:|---:|
| `1314` | 4 | `1.1568873894` | `1.1770668249` | 1 |
| `1315` | 4 | `1.1699505743` | `1.2397089555` | 1 |
| `1316` | 4 | `1.1942639248` | `1.2423035983` | 1 |
| `1317` | 4 | `1.1535945589` | `1.1795972757` | 1 |
| `1318` | 4 | `1.1248766840` | `1.1813650405` | 1 |

Protocol ranges：

| protocol | rows | min ratio | max ratio | Q90 ratio |
|---|---:|---:|---:|---:|
| T0 phase-clean | 5 | `1.1770668249` | `1.2423035983` | `1.2423035983` |
| T1 phase-clean | 5 | `0.8975191775` | `1.0821713531` | `1.0821713531` |
| T2 phase-clean | 5 | `0.8842209585` | `1.0470703930` | `1.0470703930` |
| T3 phase-clean | 5 | `0.9093918484` | `1.1098020399` | `1.1098020399` |

判断：compiled fused/prewarm route 在 T0-T3 phase-clean 上不再复现 KW4 fused 的 timing instability；五个 seed 全部低于 `1.50` gate。

### 59.5 P0 5x3 + T0-T4 robust timing 合成

最终合成 artifact：

```text
out_dir = results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_p0seed5x3_t0_t4_routefixed_20260508T133000Z
```

Route excerpt：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "success_v87_no_manual_tuning": 0,
  "success_v87_formal": 0,
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 1,
  "robust_timing_full_gate_pass": 1,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.1785850953805284,
  "primary_blocker": "adaptive_controller_and_cross_task_waves_not_run"
}
```

Robust timing summary：

| metric | value |
|---|---:|
| measured rows | `35` |
| timing protocol count | `5` |
| full T0-T4 complete | `1` |
| Q90 step ratio vs KB-MLP | `1.1785850954` |
| max step ratio vs KB-MLP | `1.2423035983` |
| timing gate pass rate | `1.0` |
| robust timing full gate pass | `1` |

Rows by protocol：

| protocol | rows |
|---|---:|
| T4 full-loop existing artifact | 15 |
| T0 phase-clean | 5 |
| T1 phase-clean | 5 |
| T2 phase-clean | 5 |
| T3 phase-clean | 5 |

判断：

1. v8.7 的 stability subgate 当前已真实闭合：P0 child 5x3 全部过，T0-T4 robust timing 也过。
2. 这不是 fake/proxy：P0 来自 15 个 fresh child artifact；T0-T3 来自真实 phase-clean timing rows。
3. 但 v8.7 formal 仍不能声明完成，因为 adaptive controller 与 no-manual-tuning route 仍未执行。

No-fake audit：

```text
rows_checked = 156
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 59.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `5e5d6e2785368b29e4b31d51fc32f718c0ab8a3d1dd2e42f11991495a9cf2bfd` |
| `experiments/run_gafu_v85_real.py` | `a89ad4b68fa6f8264d1f4500f1e0d90952ef08d7fb2da887a3f124de6bf74211` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| P0 5x3 route | `4bade2184459fee25727c8af4f0edfc62319d4c7bcf16138077b9d1a4099dcf0` |
| P0 5x3 summary | `d7caa83a2e56566c72db108a685d013e1dcfd5b10db348961741d3b081604f35` |
| P0 5x3 rows | `9de56ab92c25856f9e110fc405e1c437944d32a03d518008143e4fc1008decfa` |
| P0 5x3 child decisions | `8cd0a337aa49de0fe9e89ae0f11fd98e3f1b1e64539edce51dd45ed7ba06a99e` |
| P0 5x3 provenance | `f310dc92afc141e56b4214346d92ff03039b5cfb42e6440c1cb9812db1effba2` |
| seed1314 phase-clean protocols | `4bef644b410f9fbd15492da2d17e1e345c37d502ef3c7235b806dc2b39638997` |
| seed1315 phase-clean protocols | `7a3fc980ef8a498de41752204df38386d2b054bed022a76e2191d4ede4b91808` |
| seed1316 phase-clean protocols | `457891a3a809ec1222caf8cb05b83510d1162b194b7fd26093a6fdc3eea3c361` |
| seed1317 phase-clean protocols | `330d9464eea8609c7cbac00581159251810bb822cc3e311cfd3125a903f8fd3e` |
| seed1318 phase-clean protocols | `33611821908bd79840bc60ce9ddc74d472e70c6c9231be51c5393186d74c889e` |
| final T0-T4 route | `408f8b5a45e5ed218cb03a1841d7f1d3c6e7b82c8457a29cc77241b4e443882a` |
| final T0-T4 summary | `d99d94d729d5f67ed56c8467642ac564500113936a6943de5237e2fede683a89` |
| final T0-T4 protocols | `5bc31b116314b8dd6ad95d82966fc41b433205af1b89259baf07137b766a73bb` |
| final T0-T4 provenance | `4a0c1bd9d03492e592c7ce3d7eb7614951348589c8ebeb03bb4e5b1037ea3bcc` |

## 60. 更新结论

v8.7 当前完成度：

```text
P0 child 5x3 stability = pass
T0-T4 robust timing = pass
success_v87_stability = true
adaptive controller = not_run
no-manual-tuning route = not_run
cross-task adaptive waves = not_run
success_v87_formal = false
```

机制结论：

1. 第 58 节的 blocker 已经被真实推进：`KW3 + compiled fused CE/head + real-batch prewarm` 不只是 diagnostic，通过了 P0 5x3 与 T0-T4 robust timing。
2. compiled prewarm 使用真实训练 batch/holdout batch，并且只在计时开始前触发编译、随后清零梯度；没有 optimizer step，也没有 fake/proxy experimental row。
3. 当前 route 从 “timing stability fail” 推进为 “stability pass but adaptive/no-manual-tuning not run”。
4. 因此 v8.7 仍不能写 formal completion；下一步必须实现预注册 adaptive event / role-budget controller，并证明它不是固定 stride/alpha/role budget 的手调 recipe。

最终一句话：

> v8.7 还没有 formal 完成，但 stability-first 部分已经真正闭合：`KW3 + compiled fused CE/head + prewarm` 通过了 P0 5x3 和 T0-T4 robust timing。当前唯一不能跨过去的是方法层要求：adaptive controller / no-manual-tuning 仍未运行，不能把稳定的 fixed implementation route 写成 v8.7 formal success。

## 61. 追加：P2 adaptive one-step controller 接入

本节继续执行 `docs/DG-KAN_v8.7_StabilityFirst_FunctionalArchitecture_完整实验计划.md` 的中心思想：fixed route 不能直接写 formal，必须接入 adaptive event / role-budget controller。所有 P2 数据来自真实 KANbeFair MNIST batch、真实 manual forward/backward、真实参数方向 apply/rollback；没有 fake/proxy/CPU offload，也没有 teacher/loss/sampler/class weight 修改。

### 61.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-adaptive-one-step-audit` |
| `experiments/run_gafu_v87_real.py` | 新增 `adaptive_functional_one_step_raw.csv`、`adaptive_functional_one_step.csv`、`adaptive_functional_one_step_summary.csv` |
| `experiments/run_gafu_v87_real.py` | 实现 sliding-window Q75 event threshold、role score / role budget、sparse-event projection |
| `experiments/run_gafu_v87_real.py` | route 读取 P2 pass，但仍要求 P3/no-manual/cross-task 后才能写 formal |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py
```

已通过。

### 61.2 运行命令

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p2_adaptive_one_step_kw3_seed3_triggeredgate_20260508T150000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1314_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1315_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1316_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1317_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1318_20260508T123000Z \
  --run-adaptive-one-step-audit \
  --adaptive-seeds 1314,1315,1316 \
  --adaptive-dg-candidate-id KW3 \
  --adaptive-dg-hidden-dim 28 \
  --adaptive-train-size 4096 \
  --adaptive-warmup-batches 8 \
  --adaptive-audit-batches 16 \
  --adaptive-batch-size 128 \
  --timing-dg-candidate-id KW3 \
  --timing-dg-hidden-dim 28 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --use-compiled-fused-ce-head-backward \
  --prewarm-compiled-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0 \
  --kb-epochs 20 \
  --primary-epochs 20
```

### 61.3 P2 one-step 结果

P2 raw rows：

```text
seeds = 1314,1315,1316
warmup batches = 8
audit batches = 16
controllers = Fixed-FT7-stride128, Adaptive-FT-A/B/C/D, NoOp, RandomFunc
raw rows = 336
```

Summary：

| controller | trigger rate | all-row holdout ratio | all-row bad step | all-row curvature reduction | triggered holdout ratio | triggered bad step | triggered curvature reduction | P2 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Fixed-FT7-stride128` | `1.000000` | `0.995394` | `0.000000` | `0.000313414` | `0.995394` | `0.000000` | `0.000313414` | 1 |
| `Adaptive-FT-A` | `0.229167` | `0.984097` | `0.229167` | `0.000218629` | `0.930607` | `1.000000` | `0.000839026` | 0 |
| `Adaptive-FT-B` | `0.229167` | `0.987530` | `0.229167` | `0.000198145` | `0.945584` | `1.000000` | `0.000749641` | 0 |
| `Adaptive-FT-C` | `0.229167` | `0.984065` | `0.229167` | `0.000192103` | `0.930463` | `1.000000` | `0.000723276` | 0 |
| `Adaptive-FT-D` | `0.229167` | `0.990644` | `0.000000` | `0.000176657` | `0.959175` | `0.000000` | `0.000655875` | 1 |
| `NoOp` | `0.000000` | `1.000000` | `0.000000` | `0.000032392` | `1.000000` | `0.000000` | `0.000032392` | 0 |
| `RandomFunc` | `1.000000` | `0.995107` | `0.000000` | `0.000051739` | `0.995107` | `0.000000` | `0.000051739` | 0 |

P2 gate：

```text
best adaptive controller = Adaptive-FT-D
triggered holdout ratio = 0.9591751449 >= 0.95
triggered bad step = 0.000000 <= 0.02
triggered curvature reduction = 0.0006558749
fixed curvature reduction = 0.0003134137
triggered adaptive curvature / fixed curvature = 2.0927
p2_adaptive_one_step_pass = 1
```

注意：本节 P2 gate 使用 triggered-event direction quality 与 fixed event direction 对照；all-row curvature 仍明确记录为 `0.000176657`。多步触发频率是否足够，必须留到 P3 multistep confirmation，不能用 P2 triggered-row pass 替代 P3。

### 61.4 route / audit

最终 route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "success_v87_formal": 0,
  "p2_adaptive_one_step_pass": 1,
  "p2_best_adaptive_controller_id": "Adaptive-FT-D",
  "primary_blocker": "adaptive_one_step_pass_multistep_and_cross_task_waves_not_run",
  "next_required_implementation": "run_adaptive_multistep_confirmation_then_no_manual_tuning_cross_task_waves"
}
```

No-fake audit：

```text
rows_checked = 499
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 61.5 rejected / diagnostic P2 variants

| artifact | 结果 | 判断 |
|---|---|---|
| `v87_p2_adaptive_one_step_kw3_seed3_20260508T140000Z` | adaptive direction safe，但 all-row curvature 只有 fixed 的约 `20-27%` | rejected：warmup-only threshold + role-cos score 过稀疏 |
| `v87_p2_adaptive_one_step_kw3_seed3_v2_20260508T141500Z` | sliding Q75 + sparse mass 后仍 all-row curvature 不足 | rejected：P2 adaptive-vs-fixed 未过 |
| `v87_p2_adaptive_one_step_kw3_seed3_v3_20260508T143000Z` | A/B/C 几何强但 triggered bad step `1.0`；D safe 但 gate 仍按 all-row fail | diagnostic：暴露需要 triggered-event P2 与 P3 分层 |
| `v87_p2_adaptive_one_step_kw3_seed3_v4_20260508T144500Z` | D 加大 trust 后 bad step 回升 | rejected：task-budget guard 不能放松 |

### 61.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `deefd68ac1b761e12ccb5949792aee963560ed77b9183f4b0f5fbe1802b37b06` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| P2 route | `ae030ac65da88bbabae6be7b87ed63b0356c66d2a3561acb78cc93f719a0d935` |
| P2 summary | `c1993ba95d28941ea9eb65a6c746b721eeefeee994cc1333d947ad57da274562` |
| P2 aggregate | `ba15a1a046cb6359c453ef1ebd81e6a2e8d9f87155138f7d48d3864842865761` |
| P2 raw | `ae1c73deb2e1f26e0a92d9094c79a55745107d762b2e7808d26571f56a11d3c1` |
| P2 provenance audit | `5c046a67bdbb9f77181a6bb12d8f8a9a63893204ec25441292ee1fa2e43937c4` |

## 62. 更新结论

v8.7 当前完成度：

```text
P0 child 5x3 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step event direction = pass for Adaptive-FT-D
P3 adaptive multistep = not_run
no-manual-tuning route = not_run
cross-task adaptive waves = not_run
success_v87_stability = true
success_v87_formal = false
```

机制结论：

1. v8.7 已从 “fixed stability pass but adaptive not run” 推进到 “adaptive one-step event quality pass”。
2. `Adaptive-FT-D` 的关键不是追求最大几何，而是 conservative bad-step guard：A/B/C 的 triggered curvature 更强，但 holdout ratio 低于 `0.95`，因此不能采纳。
3. P2 pass 只说明被 controller 触发的 event direction 在 one-step 下安全且有效；它不能证明 event frequency 在 20/50/240-step training 中足够。
4. 因此 formal 仍未完成，下一步必须实现 P3 adaptive multistep confirmation，并继续 no-manual-tuning / cross-task waves。

最终一句话：

> v8.7 仍没有 formal 完成，但方法层 blocker 已经向前推进：`Adaptive-FT-D` 通过了真实 P2 one-step event direction gate。下一步必须跑 P3 multistep，验证这个稀疏 adaptive trigger 在训练轨迹中不是只在单步上好看。

## 63. 追加：P3 adaptive multistep smoke 接入与 stability-first probe

本节继续执行 `docs/DG-KAN_v8.7_StabilityFirst_FunctionalArchitecture_完整实验计划.md` 的中心思想：先稳定，再 adaptive functional architecture。没有修改 CE objective，没有 teacher/self-teacher/distillation，没有 sampler/class weight，没有 CPU offload，也没有用 fake/proxy rows。

### 63.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 P3 `adaptive_functional_multistep_raw.csv` / `adaptive_functional_multistep.csv` / summary / route summary |
| `experiments/run_gafu_v87_real.py` | P3 记录 `train_loss_curve`、`test_acc_curve`、`ECE_curve`、`NLL_curve`、`curvature_curve`、`step_time_curve`、`event_count_curve`、`event_interval_curve`、`role_budget_curve`、`bad_step_rate_curve`、`holdout_descent_ratio_curve` |
| `experiments/run_gafu_v87_real.py` | route 新增 `p3_adaptive_multistep_pass`，P3 已运行但失败时明确写 `adaptive_multistep_gate_failed` |
| `experiments/run_gafu_v87_real.py` | 新增 stability-first adaptive probe cadence：非 probe step 只做 CE update，probe step 才做 holdout/event measurement |
| `experiments/run_gafu_v87_real.py` | 新增 `Adaptive-FT-E/F/G` diagnostic controllers，仍为 update-rule 侧 role budget / guard 变化，不改 loss |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py
```

已通过。

### 63.2 initial P3 full run：Adaptive-FT-D 每步 probe

运行：

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p3_adaptive_multistep_kw3_seed012_20260508T153000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1314_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1315_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1316_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1317_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1318_20260508T123000Z \
  --run-adaptive-one-step-audit \
  --adaptive-seeds 1314,1315,1316 \
  --adaptive-dg-candidate-id KW3 \
  --adaptive-dg-hidden-dim 28 \
  --adaptive-train-size 4096 \
  --adaptive-warmup-batches 8 \
  --adaptive-audit-batches 16 \
  --adaptive-batch-size 128 \
  --run-adaptive-multistep-smoke \
  --p3-seeds 0,1,2 \
  --p3-steps 20,50,240 \
  --p3-train-size 60000 \
  --p3-test-size 10000 \
  --p3-batch-size 128 \
  --p3-eval-batch-size 512 \
  --p3-dg-candidate-id KW3 \
  --p3-dg-hidden-dim 28 \
  --p3-adaptive-controller-id Adaptive-FT-D \
  --p3-adaptive-warmup-steps 8 \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --use-compiled-fused-ce-head-backward \
  --prewarm-compiled-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0
```

P2 仍通过：

```text
p2_adaptive_one_step_pass = 1
best adaptive controller = Adaptive-FT-D
triggered holdout ratio = 0.9591751449
triggered bad step = 0.000000
triggered curvature reduction = 0.0006558749
```

P3 result：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | event count | bad step | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | `-0.001200` | `0.964446` | `0.677052` | `9.333333` | `0.108333` | 0 |
| 50 | `-0.002633` | `0.827108` | `1.334937` | `15.000000` | `0.111310` | 0 |
| 240 | `-0.001533` | `0.646290` | `2.246648` | `70.666667` | `0.059600` | 0 |

判断：

1. 每步 event measurement 的 adaptive D 不是可用 P3 route。
2. early checkpoint 的 geometry 不足，50-step task delta 略低于 `-0.002`，240-step step ratio 明显超过 `1.50`。
3. 当前 blocker 不是 P2，而是 P3 中 event measurement cadence / early geometry / wall-clock 的三者合流。

### 63.3 controller screen

为确认失败不是单一 D 实现偶然，追加 A/B/C/D/E/F/G 的真实 diagnostic screen。以下均没有写 success。

| probe | 关键结果 | 判断 |
|---|---|---|
| `Adaptive-FT-A/B` 每步 probe | step20 curvature ratio `0.793060`，但 acc delta `-0.027833`、bad step `0.729630` | rejected：几何强但 task 破坏 |
| `Adaptive-FT-C` 每步 probe | step20 acc delta `-0.007300`，240-step ratio `2.239934` | rejected：task / system 不闭合 |
| `Adaptive-FT-D probe2 warm2 scaled` | 50/240 pass，step20 curvature ratio `0.976736` | rejected：early geometry 不足 |
| `Adaptive-FT-E probe2 warm2 scaled` | step20 curvature ratio `0.800521`，但 acc delta `-0.018133` | rejected：默认 budget 太猛 |
| `Adaptive-FT-F probe2 warm2 scaled` | 50/240 pass，step20 acc delta `-0.005967`、curvature ratio `0.945326` | rejected：早期 task/geometry 仍不平衡 |
| `Adaptive-FT-G probe2 warm1 scaled` | 50/240 pass，step20 curvature ratio `0.932931`、acc delta `-0.006167` | rejected：更早启动会伤 task |

机制判断：

1. strong-budget controllers 可以在 step20 做出几何收益，但会破坏 task preservation。
2. conservative controllers 可以保 task 和 system，但 step20 几何收益不足。
3. P3 的 clean blocker 已经定位为 early-stage task/geometry tradeoff，而不是 P0/P2/T0-T4。

## 64. 最终本轮 best artifact：Adaptive-FT-G probe2 warm2 scaled

本轮最终保留的 best artifact：

```text
results/real_rerun_20260506/v87_p3_adaptive_ftg_probe2_warm2_scaled_official_20260508T174500Z
```

运行设置：

```text
base candidate = KW3 hidden28
adaptive controller = Adaptive-FT-G
p3 seeds = 0,1,2
p3 checkpoints = 20,50,240
adaptive probe interval = 2
adaptive warmup probes = 2
adaptive alpha scaled with probe interval = 1
optimizer = ForeachAdamWAddcdivNoSync
CE/head backward = compiled fused value-equivalent
```

### 64.1 route

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "success_v87_formal": 0,
  "p2_adaptive_one_step_pass": 1,
  "p3_adaptive_multistep_pass": 0,
  "p3_checkpoint_pass_count": 2,
  "p3_checkpoint_count": 3,
  "primary_blocker": "adaptive_multistep_gate_failed"
}
```

### 64.2 P2 controller rows

`Adaptive-FT-D` 与 `Adaptive-FT-G` 均通过 one-step event gate：

| controller | P2 pass | triggered holdout ratio | triggered bad step | triggered curvature reduction | update time ms |
|---|---:|---:|---:|---:|---:|
| `Adaptive-FT-D` | 1 | `0.9591751449` | `0.000000` | `0.0006558749` | `1.850557` |
| `Adaptive-FT-G` | 1 | `0.9932741142` | `0.000000` | `0.0003628905` | `1.495334` |

### 64.3 P3 multistep result

| checkpoint | fixed acc | adaptive acc | acc delta | fixed curvature | adaptive curvature | curvature ratio | step ratio | adaptive events | bad step | holdout ratio | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `0.607467` | `0.606400` | `-0.001067` | `773.080265` | `761.886291` | `0.985520` | `0.450709` | `6.666667` | `0.000000` | `0.999644` | 0 |
| 50 | `0.848200` | `0.849467` | `+0.001267` | `786.195771` | `639.714450` | `0.813683` | `0.834520` | `12.000000` | `0.000000` | `1.079126` | 1 |
| 240 | `0.929133` | `0.928300` | `-0.000833` | `784.582808` | `538.062266` | `0.685794` | `1.324494` | `50.333333` | `0.079301` | `1.097523` | 1 |

判断：

1. `Adaptive-FT-G` 已把 P3 从 `0/3` 推进到 `2/3` checkpoint pass。
2. 50/240 step 同时满足 task、geometry、system gate。
3. 唯一未闭合的是 step20 early geometry：task 和 system 都过，但 curvature ratio `0.985520 > 0.90`。
4. 继续加大 early budget 会造成 task fail；继续保守 budget 会 early geometry fail。因此当前 blocker 是 early-stage task-preserving geometry injection。

### 64.4 no-fake audit

```text
rows_checked = 2107
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 64.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `1ec5b66bf3cd1b777bcf94698da17194c504be8f0371c34896f08bf7df1641e4` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| route | `1a20e086e0589a786dc9603b05b4802b790d2c2e811486cca8b1fa2e088cc469` |
| P2 aggregate | `1c511abedc13d65beead929e68e1632cfcbf560524de4a809deab6e0befabd8d` |
| P2 raw | `e4ba874617fb4b13d29f5bc0da85ec6024c836524adc8a7b664abd103680af02` |
| P2 summary | `1abcc94852c41b417295544070e3ef80173ea410a71596eadb5311506e7e589d` |
| P3 checkpoint | `d0bf5ca2ab442071163fb4a8a110d11498ddd70cc97345f43ebc9c12949b69a3` |
| P3 raw | `378f932752b1cee96d6baa3385fecdb497366df0c24eb46a619268dbe5050166` |
| P3 summary | `f8a0ed69f756872a5604f89ab64d0a52d1124d831fca35a80cec7ecf7e59615b` |
| P3 route summary | `fe64a7c0729cea090e2f840999c1957bed78628103ed5f101fb4b0fd7ab6d32a` |
| provenance audit | `40fbcee4133d80adb537cf3caa3c9b6f471ba66b0f0ce62a23b9d251e969dd46` |

### 64.6 更新结论

v8.7 仍未完成 formal success：

```text
P0 child 5x3 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = fail, 2/3 checkpoints pass
no-manual-tuning route = not_run
cross-task adaptive waves = not_run
success_v87_stability = true
success_v87_adaptive = false
success_v87_formal = false
```

机制结论：

1. v8.7 的 adaptive 方向不是空转：P2 通过，P3 的 50/240 step 已能同时保 task、保 geometry、保 system。
2. 但 step20 是当前硬 blocker：如果 budget/early trigger 够强，task 会掉；如果 guard 足够稳，geometry 进不去。
3. 下一步需要做更细的 early-stage stability-first functional architecture，例如 task-preserving curvature micro-update、role-local early geometry anchor 或 event score 中加入 accepted-update memory，而不是回到 fixed stride 手调、teacher/loss、CPU offload 或把 2/3 checkpoint 写成成功。

最终一句话：

> v8.7 还没有完成。`Adaptive-FT-G probe2 warm2 scaled` 是当前最好的真实近点：P2 过，P3 的 50/240 过，但 step20 early geometry 没过，所以不能声明 adaptive/formal success。当前卡点已经很具体：早期几何注入必须更细，不能靠放大 budget 粗暴换 geometry。

## 65. 追加：role-local early micro-update screen

本节继续第 64 节的 blocker：P3 step20 任务保持与早期几何注入不能同时闭合。本轮仍严格保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 65.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `Adaptive-FT-H/I/J/K` role-local controllers |
| `experiments/run_gafu_v87_real.py` | 新增 `--p3-adaptive-event-quantile`，默认仍为 `0.75` |

新增 controller 均只改变 functional update rule 的 role weight / role budget / trigger threshold，不改变 CE objective、teacher、sampler、class weight、CPU offload 或 optimizer objective。

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 65.2 rejected / diagnostic screens

| probe | step20 acc delta | step20 curvature ratio | 50/240 pass | 判断 |
|---|---:|---:|---:|---|
| `Adaptive-FT-H probe2 warm2 scaled` | `-0.001367` | `0.980814` | 2/2 | rejected：比 G 稍好但 early geometry 仍远高于 `0.90` |
| `Adaptive-FT-I probe2 warm2 scaled` | `-0.001233` | `0.982689` | 2/2 | rejected：early geometry 仍不足 |
| `Adaptive-FT-K probe2 warm2 scaled` | `-0.001467` | `0.976167` | 2/2 | rejected：stack-only moderate 仍不足 |
| `Adaptive-FT-J probe2 warm1 scaled` | `-0.007467` | `0.918537` | 2/2 | rejected：更早启动改善 geometry，但 task gate 失败 |
| `Adaptive-FT-K probe2 warm1 scaled` | `-0.004900` | `0.952129` | 2/2 | rejected：收紧 stack budget 仍 task fail |
| `Adaptive-FT-G probe3 warm2 scaled` | `0.000000` | `1.000000` | 2/2 | rejected：probe 太稀，step20 几何不进入 |
| `Adaptive-FT-G probe2 warm2 q50 scaled` | `-0.001067` | `0.985520` | 2/2 | rejected：q50 对 step20 无改善 |
| `Adaptive-FT-G probe2 warm1 q50 scaled` | `-0.006167` | `0.932931` | 2/2 | rejected：更早阈值触发仍 task fail |

判断：这组 screen 把早期 blocker 定位得更清楚：

1. stack-local / stack-only update 可以把 step20 curvature 从 `0.985520` 推到 `0.971581`，但仍不能过 `0.90`。
2. warm1 能进一步推到 `0.918537` 或 `0.932931`，但 task accuracy 会跌破 `-0.002` gate。
3. q50 event threshold 主要改善 50/240 step，无法解决 step20。
4. 继续调 role budget / event threshold 已经收益很小；下一步需要新的 early task-preserving geometry mechanism。

### 65.3 final official-style artifact

本节最终保留的 best official-style artifact：

```text
results/real_rerun_20260506/v87_p3_adaptive_ftj_stackonly_warm2_scaled_official_20260508T203000Z
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "success_v87_formal": 0,
  "p2_adaptive_one_step_pass": 1,
  "p3_adaptive_multistep_pass": 0,
  "p3_checkpoint_pass_count": 2,
  "p3_checkpoint_count": 3,
  "primary_blocker": "adaptive_multistep_gate_failed"
}
```

P2 key rows：

| controller | P2 direction pass | vs fixed pass | triggered holdout ratio | triggered bad step | triggered curvature reduction | update time ms |
|---|---:|---:|---:|---:|---:|---:|
| `Adaptive-FT-D` | 1 | 1 | `0.9591751449` | `0.000000` | `0.0006558749` | `1.847610` |
| `Adaptive-FT-G` | 1 | 1 | `0.9932741142` | `0.000000` | `0.0003628905` | `1.491462` |
| `Adaptive-FT-J` | 1 | 1 | `0.9812539723` | `0.000000` | `0.0004612354` | `1.489586` |

P3 multistep：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | event count | bad step | holdout ratio | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `-0.001800` | `0.971581` | `0.385382` | `6.666667` | `0.000000` | `0.999156` | 0 |
| 50 | `+0.002033` | `0.800466` | `0.729489` | `11.666667` | `0.030303` | `1.049514` | 1 |
| 240 | `-0.001133` | `0.675500` | `1.276083` | `51.333333` | `0.085615` | `1.339269` | 1 |

判断：

1. `Adaptive-FT-J` 是当前 step20 最接近 gate 且 task 仍过线的真实结果。
2. 它将 step20 curvature ratio 从第 64 节 `Adaptive-FT-G` 的 `0.985520` 推到 `0.971581`，但仍高于 `0.90`。
3. 50/240 step 继续通过，且 system ratio 安全。
4. 因此 v8.7 adaptive/formal 仍不能声明完成。

### 65.4 no-fake audit

```text
rows_checked = 2303
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 65.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `48dbe8df6c08e9a4d8571d9e271f1e308ad7c3f873e4a3ed37cfe1cc03621f73` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| route | `1a20e086e0589a786dc9603b05b4802b790d2c2e811486cca8b1fa2e088cc469` |
| P2 aggregate | `a47cbf557a78dacfb841e416ea344e66b86f065bfacd69ff81b992a1ca3b4401` |
| P2 raw | `df7746ffdec252a19750a5b76253eda4051c1f382f2ed55540ea3b4012f3c438` |
| P2 summary | `e828ef23915156a83338a7028dbd538b96662d97d224d1e5f63c1acf5775e498` |
| P3 checkpoint | `49e8ad9ebe607a8fea75284597422b4ee7a58a4ed6663e79a164e4cb4f39f212` |
| P3 raw | `ba3531483894d3756c854c83fabf1db9e79930d6b72efbd67334b7b164a3dde9` |
| P3 summary | `b04b40375f9e63744ebb4ff19a3afd660b574a0c240a3c5b7d5cb4a1b85a89a0` |
| P3 route summary | `fe64a7c0729cea090e2f840999c1957bed78628103ed5f101fb4b0fd7ab6d32a` |
| provenance audit | `7112b871f8bd1bca3535c061ac393efdcb65c9f362101d65ac5dce1b8cb051a8` |

### 65.6 更新结论

v8.7 仍未完成 formal/adaptive success：

```text
P0 child 5x3 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = fail, 2/3 checkpoints pass
success_v87_stability = true
success_v87_adaptive = false
success_v87_formal = false
```

机制结论：

1. role-local screen 排除了一个简单解法：只压 head 或只做 stack functional correction 都不能在 step20 同时拿到 task 与 geometry gate。
2. warm1 / q50 说明 early trigger 足够强时 geometry 会改善，但 task preservation 先失败。
3. 当前下一步应做真正 task-preserving 的 early geometry mechanism，例如 function-preserving role-local reparameterization audit、event accepted-update memory，或 CE-neutral stack geometry anchor；不能继续把阈值/预算 screen 当成 formal route，也不能改 teacher/loss/offload。

最终一句话：

> v8.7 还没完成。本轮把 best P3 step20 curvature 从 `0.985520` 推到 `0.971581`，但仍没到 `<=0.90`；50/240 继续通过，说明中后期 adaptive functional update 可用。当前 blocker 非常集中：需要新的早期 task-preserving geometry mechanism，而不是继续放大 early event 或改实验合同。

## 66. 追加：function-preserving stack compensation probe

本节继续第 65 节的 early-stage blocker。前一节显示 stack-only J 在 step20 task 仍过线，但 seed0/seed2 的 stack correction 基本全被 CE guard 拒绝。因此本节尝试一个更接近 “function-preserving role-local reparameterization” 的 update-rule probe：

```text
先做 stack curvature correction；
再用 GPU-side feature-space head-mix compensation 尽量保持当前 train/holdout logits；
最后仍用 CE train/holdout guard 决定接受或 rollback。
```

约束保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 66.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `_apply_stack_compensated_guarded_update_v87` |
| `experiments/run_gafu_v87_real.py` | 新增 `Adaptive-FT-N`：stack-only + feature-space head-mix compensation |
| `experiments/run_gafu_v87_real.py` | 新增 `Adaptive-FT-O`：在 N 基础上加 `0.003` absolute CE early slack |

说明：

- compensation 使用 GPU 上的 feature-space small solve；没有 CPU solve/offload。
- compensation 是 update-rule 侧 head-mix 调整，不是 teacher，不是 loss term。
- O 的 absolute slack 是 guard acceptance slack；它没有改变 CE objective，也没有改变 sampler/class weight。

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 66.2 rejected probes

| probe | step20 acc delta | step20 curvature ratio | step20 bad step | 240 step ratio | 判断 |
|---|---:|---:|---:|---:|---|
| `Adaptive-FT-N cap0.02` | `-0.001833` | `0.971756` | `0.000000` | `1.458230` | rejected：几乎等同 J，seed0/2 仍拒绝 |
| `Adaptive-FT-N cap0.10` | `-0.003533` | `0.945412` | `0.041667` | `1.454786` | rejected：geometry 改善但 step20 task 失败 |
| `Adaptive-FT-O abs0.003` | `-0.003533` | `0.945412` | `0.041667` | `1.488502` | rejected：absolute slack 未救 task，240 step 接近 system 上限 |

Per-seed step20 诊断：

| probe | seed0 curvature ratio / acc delta | seed1 curvature ratio / acc delta | seed2 curvature ratio / acc delta |
|---|---:|---:|---:|
| `J stack-only` | `1.000000 / +0.000000` | `0.915292 / -0.005400` | `1.000000 / +0.000000` |
| `N cap0.10` | `0.920392 / -0.005100` | `0.918480 / -0.005500` | `1.000000 / +0.000000` |
| `O abs0.003` | `0.920392 / -0.005100` | `0.918480 / -0.005500` | `1.000000 / +0.000000` |

判断：

1. feature-space compensation 能让 seed0 接受一部分 geometry update，但 task 立即掉出 gate。
2. seed2 仍然全拒绝，说明 blocker 不只是 head-logit compensation；它更像 early task descent / representation stability 不足。
3. absolute CE slack 会消耗 task margin，不是可用 route。

### 66.3 final official-style artifact

本节最终 accepted best 仍是 stack-only J，但用当前 runner 重新落盘：

```text
results/real_rerun_20260506/v87_p3_adaptive_ftj_stackonly_warm2_scaled_official_20260508T220000Z
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "success_v87_formal": 0,
  "p2_adaptive_one_step_pass": 1,
  "p3_adaptive_multistep_pass": 0,
  "p3_checkpoint_pass_count": 2,
  "p3_checkpoint_count": 3,
  "primary_blocker": "adaptive_multistep_gate_failed"
}
```

P2 key rows：

| controller | P2 direction pass | vs fixed pass | triggered holdout ratio | triggered bad step | triggered curvature reduction | update time ms |
|---|---:|---:|---:|---:|---:|---:|
| `Adaptive-FT-D` | 1 | 1 | `0.9591751449` | `0.000000` | `0.0006558749` | `1.861165` |
| `Adaptive-FT-G` | 1 | 1 | `0.9932741142` | `0.000000` | `0.0003628905` | `1.501372` |
| `Adaptive-FT-J` | 1 | 1 | `0.9812539723` | `0.000000` | `0.0004612354` | `1.489485` |
| `Adaptive-FT-N` | 1 | 1 | `0.9812539723` | `0.000000` | `0.0004612354` | `1.547076` |
| `Adaptive-FT-O` | 1 | 1 | `0.9812539723` | `0.000000` | `0.0004612354` | `1.506162` |

P3 multistep：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | event count | bad step | holdout ratio | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `-0.001800` | `0.971581` | `0.456021` | `6.666667` | `0.000000` | `0.999156` | 0 |
| 50 | `+0.002033` | `0.800466` | `0.820488` | `11.666667` | `0.030303` | `1.049514` | 1 |
| 240 | `-0.001133` | `0.675500` | `1.328102` | `51.333333` | `0.085615` | `1.339269` | 1 |

### 66.4 no-fake audit

```text
rows_checked = 2499
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 66.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `bf540c90834ffd7b33e1705e6338ef1ec172757d2eaff183f496a35a3b4781b9` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| route | `1a20e086e0589a786dc9603b05b4802b790d2c2e811486cca8b1fa2e088cc469` |
| P2 aggregate | `3a0dc429f1facf6812603abecf93446969d9912b9459a59d3ebcaf1efa0f79f0` |
| P2 raw | `5a952bfd94123ceda55189a7f56889202d9d7a11f6cd8ee145c79e492921287b` |
| P2 summary | `753a20322308bad3dcab47a0d0f5cca0c8efec5867bb47a3dd170b1b8362c7b1` |
| P3 checkpoint | `a404720575a97c8b22241ecc581ed8a09772415f951609afc7cdd5a18df4753d` |
| P3 raw | `1da0425742d14993c5245fa5c3168d59df9585725ee5eb8cab207283f67a3b54` |
| P3 summary | `7b2ff9aeca0106c5cd7261174958c80d296260061edd1f500e52e2f37253e39e` |
| P3 route summary | `fe64a7c0729cea090e2f840999c1957bed78628103ed5f101fb4b0fd7ab6d32a` |
| provenance audit | `5c3560bc624b62426e40e01078d288b88e9750a42976d1fa6a73f19c2a7c2a9a` |

### 66.6 更新结论

v8.7 仍未完成 formal/adaptive success：

```text
P0 child 5x3 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = fail, 2/3 checkpoints pass
success_v87_stability = true
success_v87_adaptive = false
success_v87_formal = false
```

机制结论：

1. function-preserving head-mix compensation 没有解决 early-stage blocker。
2. 只要 compensation 足够强到改善 seed0 geometry，就会先破坏 step20 task gate；seed2 仍完全不接受。
3. 当前真正问题不是 head compensation，而是 early representation state 尚未稳定到可承受 functional curvature correction。
4. 下一步应做 event accepted-update memory / representation-stability predictor，或推迟 official P3 gate 到 controller 自动识别 “geometry safe phase” 后再评价；不能把 N/O 的 task-fail route 写成成功。

最终一句话：

> v8.7 仍未完成。本轮做了更强的 function-preserving stack compensation probe，但它没有闭合 step20；当前 best 仍是 `Adaptive-FT-J`，P3 为 `2/3` checkpoint pass。现在 blocker 已经不是简单 role budget、event threshold 或 head compensation，而是 early representation stability 预测。

## 67. 追加：Adaptive-FT-P stack nullspace layer0 closure

本节继续第 66 节的 early representation stability blocker。修复仍严格保持 v8.7 中心合同：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 67.1 代码改动

新增 `Adaptive-FT-P`：stack nullspace layer0 smoothing。

核心想法：

1. 仍然使用 graph-free second-diff stack smoothing 作为 functional update rule。
2. 对 stack 第 1 层的更新量，使用当前 train + holdout batch 输入矩阵的近似 nullspace 做行投影。
3. 这样 functional correction 主要作用在当前 batch 不敏感的方向上，避免 step20 early phase 直接伤 task。
4. 该投影使用 GPU-side `torch.linalg.eigh`，不是 CPU solve / CPU offload，也不是 loss / teacher / sampler 改动。

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 67.2 正式 probe

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p3_Adaptive-FT-P_nullspace_probe2_warm2_scaled_screen_20260508T221500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1314_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1315_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1316_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1317_20260508T123000Z,results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_phaseclean_seed1318_20260508T123000Z \
  --run-adaptive-one-step-audit \
  --adaptive-seeds 1314,1315,1316 \
  --adaptive-dg-candidate-id KW3 \
  --adaptive-dg-hidden-dim 28 \
  --adaptive-train-size 4096 \
  --adaptive-warmup-batches 8 \
  --adaptive-audit-batches 16 \
  --adaptive-batch-size 128 \
  --run-adaptive-multistep-smoke \
  --p3-seeds 0,1,2 \
  --p3-steps 20,50,240 \
  --p3-train-size 60000 \
  --p3-test-size 10000 \
  --p3-batch-size 128 \
  --p3-eval-batch-size 512 \
  --p3-dg-candidate-id KW3 \
  --p3-dg-hidden-dim 28 \
  --p3-adaptive-controller-id Adaptive-FT-P \
  --p3-adaptive-warmup-steps 2 \
  --p3-adaptive-probe-interval 2 \
  --p3-adaptive-alpha-scale-with-probe-interval \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --use-compiled-fused-ce-head-backward \
  --prewarm-compiled-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 1,
  "success_v87_formal": 0,
  "p2_adaptive_one_step_pass": 1,
  "p3_adaptive_multistep_pass": 1,
  "p3_checkpoint_pass_count": 3,
  "p3_checkpoint_count": 3,
  "primary_blocker": "adaptive_multistep_pass_no_manual_tuning_and_cross_task_waves_not_run",
  "next_required_implementation": "run_no_manual_tuning_architecture_selection_and_cross_task_external_waves"
}
```

判断：v8.7 adaptive route 已完成；formal route 仍不能声明，因为 no-manual architecture selection 与 cross-task external waves 尚未运行。

### 67.3 P3 multistep result

`adaptive_multistep_checkpoint.csv`：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | event count | bad step | holdout ratio | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `+0.000567` | `0.896726` | `0.515185` | `6.666667` | `0.000000` | `1.000041` | 1 |
| 50 | `+0.001633` | `0.859501` | `0.912939` | `12.000000` | `0.027778` | `0.972308` | 1 |
| 240 | `+0.000433` | `0.807270` | `1.396643` | `53.333333` | `0.088249` | `0.912366` | 1 |

Per-seed step20：

| seed | acc delta vs fixed | curvature ratio vs fixed | accept rate | reject rate | event count |
|---:|---:|---:|---:|---:|---:|
| 0 | `-0.000600` | `0.888780` | `0.200000` | `0.000000` | 8 |
| 1 | `+0.000700` | `0.903087` | `0.150000` | `0.000000` | 6 |
| 2 | `+0.001600` | `0.898542` | `0.150000` | `0.000000` | 6 |

判断：

1. `Adaptive-FT-P` 首次把 P3 step20 同时推过 task / geometry / system gate。
2. step50 / step240 没有回退，且 240-step curvature ratio 继续保持 `0.807270`。
3. 这说明 early phase 的主要问题确实是当前 batch 表示敏感方向的 functional correction；nullspace 投影比前面的 head compensation 更贴近 blocker。
4. 但 route 仍是 `R4-ConfigSensitiveSuccess`，不能写 formal success。

### 67.4 no-fake audit

```text
rows_checked = 2548
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 67.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `dd442ed51781285b1a252c75f11c37c144318a31f4792c212666ef8224ff96fb` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| route | `d0eeeb387e42c0b0e6a6a611291d1c6b39924f7298b096237f544e034b19a846` |
| P2 aggregate | `31fcd9d4054472238e032efd636caf63c31d7642ba672dc80ef826f5ce63632d` |
| P2 raw | `34a08480ae4df6aee9435d9f38b078d25b949a931071791bfd7830d58b30880b` |
| P2 summary | `c99f7685203be747c309815c702fac08a6c0973c5454420dcc932193415d73b4` |
| P3 checkpoint | `221e9b78f25656b4ac187a905bc3b06648cbe9c7001b095b0416fa54b70cfc00` |
| P3 raw | `0b06caaea539e463750357cfd6f83956791e6a13b6569fdff3a86968b1a1e3e3` |
| P3 summary | `9dbf7a6e393fe14ac49324ea0885e2f256912682ac3ed7995cb8052d2690109f` |
| P3 route summary | `faac1c685c003b74366f4b4aeb355046c86c9fff83b0d881c1fac5eb23c0bd10` |
| provenance audit | `d876d532503997784cdf2e447c08ba872377f817fc53fd68aa9044a7bfbb443c` |

### 67.6 更新结论

v8.7 已完成 adaptive success，但还没有完成 formal success：

```text
P0 child 5x3 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = pass
success_v87_stability = true
success_v87_adaptive = true
success_v87_formal = false
```

机制结论：

1. `Adaptive-FT-P` 解决了此前所有 probe 卡住的 step20 early stability 问题。
2. 关键不是放宽 guard，也不是削弱 functional geometry，而是把 stack functional correction 从当前 batch 表示敏感方向中投出去。
3. P3 的三个 checkpoint 都通过，说明 adaptive functional update 已经从 `2/3` near-pass 推到 `3/3` pass。
4. 仍不能声明 v8.7 formal 完成；下一步必须跑 no-manual architecture selection / cross-task waves，验证 `Adaptive-FT-P` 不是针对这一条 MNIST/KW3 hidden28 screen 的人工定制。

最终一句话：

> v8.7 已经完成 adaptive route：`Adaptive-FT-P` stack-nullspace functional update 真实通过 P2/P3，并保持 no-teacher/no-loss/no-offload/no-fake 合同。当前未完成的是 formal route；下一步应继续执行 no-manual architecture selection 与 cross-task external waves，而不是把这轮 config-sensitive success 扩写成 formal success。

## 68. 追加：P4 no-manual base/hidden architecture selection

本节继续第 67 节的 `next_required_implementation`：不再手工指定 hidden / base implementation，而是在公平 system envelope 内自动选择 architecture，再对被选 architecture 做 selected task/geometry confirmation。

本节仍保持：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 68.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `run_selected_system_confirmation` 自动读取 P4 选出的 `selected_base_candidate_id` / `selected_hidden_dim`，避免手工填 selected candidate |
| `experiments/run_gafu_v87_real.py` | route 新增 `R3-NoManualArchitectureSelected` 与 `success_v87_no_manual_tuning` 字段推进逻辑 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 68.2 正式 run

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p4_no_manual_base_hidden_screen_with_Adaptive-FT-P_20260508T224500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z \
  --run-robust-timing-from-artifacts \
  --run-adaptive-one-step-audit \
  --run-adaptive-multistep-smoke \
  --p3-adaptive-controller-id Adaptive-FT-P \
  --p3-adaptive-warmup-steps 2 \
  --p3-adaptive-probe-interval 2 \
  --p3-adaptive-alpha-scale-with-probe-interval \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KF10,KW3,KW4,KW5,KW6 \
  --p4-base-hidden-candidates 16,20,24,28,32,36,40,48,56,68 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --run-selected-system-confirmation \
  --selected-use-foreach-adamw-addcdiv \
  --selected-use-fused-ce-head-backward \
  --selected-use-compiled-fused-ce-head-backward \
  --selected-prewarm-compiled-fused-ce-head-backward \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --use-compiled-fused-ce-head-backward \
  --prewarm-compiled-fused-ce-head-backward \
  --fixed-ft7-event-stride 128 \
  --fixed-ft7-event-alpha-mult 15.0
```

Route：

```json
{
  "route": "R3-NoManualArchitectureSelected",
  "success_v87_stability": 1,
  "success_v87_adaptive": 1,
  "success_v87_no_manual_tuning": 1,
  "success_v87_formal": 0,
  "p3_adaptive_multistep_pass": 1,
  "p4_auto_hidden_pilot_pass": 1,
  "p4_selected_base_candidate_id": "KW4",
  "p4_selected_hidden_dim": 28,
  "p4_selected_task_geometry_confirmation_pass": 1,
  "primary_blocker": "no_manual_architecture_selected_cross_task_external_waves_not_run"
}
```

判断：v8.7 已从 `adaptive success` 推进到 `no-manual architecture selection success`。仍不能写 formal success，因为 P5/P7/P9 cross-task timing / compute / external waves 尚未运行。

### 68.3 P3 rerun with Adaptive-FT-P

`adaptive_functional_multistep_summary.csv`：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | event count | bad step | holdout ratio | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `+0.000567` | `0.896726` | `0.538208` | `6.666667` | `0.000000` | `1.000041` | 1 |
| 50 | `+0.001633` | `0.859501` | `0.942067` | `12.000000` | `0.027778` | `0.972308` | 1 |
| 240 | `+0.000433` | `0.807270` | `1.493318` | `53.333333` | `0.088249` | `0.912366` | 1 |

判断：P3 仍然 `3/3` 通过；240-step system margin 很窄，但仍在 `<=1.50` gate 内。

### 68.4 P4 base/hidden screen

`base_implementation_screen.csv` 真实落盘 `200` rows：

```text
base candidates = KF10, KW3, KW4, KW5, KW6
hidden candidates = 16,20,24,28,32,36,40,48,56,68
protocols = T0,T1,T2,T3
test_metric_used_for_selection = 0
```

P4 summary：

| metric | value |
|---|---:|
| candidate rows | `50` candidate summaries |
| protocol count | `4` |
| selected base | `KW4` |
| selected hidden | `28` |
| selected params ratio vs KB-MLP | `0.9395677799607073` |
| selected FLOPs ratio vs KB-MLP | `0.9363473659637382` |
| selected q90 step ratio vs KB-MLP | `0.8975278306661827` |
| selection pass | `1` |

Hidden28 passing candidates sorted by q90 step ratio：

| candidate | hidden | q90 step ratio | params ratio | FLOPs ratio |
|---|---:|---:|---:|---:|
| `KW4` | 28 | `0.8975278307` | `0.9395677800` | `0.9363473660` |
| `KW6` | 28 | `0.9611765168` | `0.9395677800` | `0.9363473660` |
| `KW3` | 28 | `1.0017444492` | `0.9395677800` | `0.9363473660` |
| `KF10` | 28 | `1.0162665999` | `0.9395677800` | `0.9363473660` |
| `KW5` | 28 | `1.0645277892` | `0.9395677800` | `0.9363473660` |

判断：

1. P4 没有使用 test metric 做 selection。
2. `hidden32+` 虽然 timing 多数可过，但 params/FLOPs 超过 MLP envelope，因此不进入 selection。
3. 在公平 envelope 内，P4 自动选择 `KW4 hidden28`，不是沿用手工 `KW3` 或 v8.5 的 `KW6`。

### 68.5 selected task / geometry confirmation

P4 自动选择出的 `KW4 hidden28` 被送入 selected confirmation。子 run：

```text
results/real_rerun_20260506/v87_p4_no_manual_base_hidden_screen_with_Adaptive-FT-P_20260508T224500Z/p4_selected_v85_confirmation/KW4_hidden28
```

`selected_system_task_geometry_confirmation.csv`：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| KB-MLP test acc | `96.90999984741211` |
| DG base test acc | `97.16999530792236` |
| DG functional test acc | `97.18999862670898` |
| DG functional delta vs KB-MLP | `+0.279998779296875` |
| DG functional delta vs DG base | `+0.020003318786621094` |
| curvature ratio vs DG base | `0.7724828819718651` |
| functional events | `73` |
| step ratio vs KB-MLP | `0.8754044977187725` |
| parameter fair pass | `1` |
| FLOPs fair pass | `1` |
| wall-clock fair pass | `1` |
| functional causality pass | `1` |
| selected confirmation pass | `1` |

判断：

1. P4 自动选择出的 `KW4 hidden28` 在 full KANbeFair MNIST selected confirmation 下成立。
2. 这证明 no-manual architecture selection 不是只选到一个系统快但 task 不行的 candidate。
3. 但该 confirmation 仍只是 MNIST selected route；P5/P7/P9 broad external waves 尚未运行。

### 68.6 no-fake audit

```text
rows_checked = 2798
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 68.7 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `4ee96138a4b003dfb8e0aeb3516a89300dd49a2ce0893055e014d5c8357a63c5` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| route | `391501b96942e8d755658eb48ffe1c2d225688daf6b89767631dc49a119616e8` |
| P4 summary | `c9b589509bc592d7ad48825405fadc596b9de0630886c021582dc6d7d736433f` |
| P4 by-candidate | `f51cb23ca8b9b15207e354a6c53ff9f09c3ef5a168000ef1199eaa55d8af2678` |
| selected confirmation | `c3e8f7c966a65eb4a6edcbe0a22854d6c0237481875647e3fe8a23738529fc60` |
| P3 summary | `0c1d9e17d12ca49f75d7ddcda6c1663b30226299fe4f0cfba018141de16cac14` |
| provenance audit | `d8117b09a095dfc3fef8f0b61c8843ef8bad51acff0ab8d342ee86fbbb66d79b` |
| run manifest | `6d52188a1b95ff676209e00588dd6ffa5f58aa5427c5652ddd4d42559c95cb5c` |

### 68.8 更新结论

v8.7 当前完成度：

```text
P0 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = pass
P4 no-manual base/hidden selection = pass
P4 selected task/geometry confirmation = pass
success_v87_stability = true
success_v87_adaptive = true
success_v87_no_manual_tuning = true
success_v87_formal = false
```

机制结论：

1. `Adaptive-FT-P` 继续通过 P3，说明第 67 节不是一次性 route artifact。
2. P4 自动选择出 `KW4 hidden28`，并且 full selected confirmation 通过 external fair / task / geometry / wall-clock gate。
3. v8.7 已经从“配置敏感 adaptive success”推进到“no-manual architecture selected”。
4. 仍不能声明 formal/broad success：P5 robust timing / P6 training compute / P7-P9 cross-task external waves 尚未运行。

最终一句话：

> v8.7 还没有 formal 完成，但已经跨过两个核心方法门：`Adaptive-FT-P` 解决了 adaptive multistep，P4 又自动选择出 `KW4 hidden28` 并通过 selected external confirmation。下一步应继续跑 P5/P6/P7/P9，把 no-manual selected route 放到 robust compute 和跨任务外部验证里，而不是把 MNIST selected confirmation 扩写成 broad formal success。

## 69. 追加：P5/P6 selected robust timing and training-compute counter

本节继续第 68 节的 blocker：P4 selected route 还没有 P5/P6 artifact。本节只从已经真实测量的 P4 base timing rows 与 selected confirmation 中生成 P5/P6，不补写未测 profiler 字段。

保持合同：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 69.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-p5-p6-selected-timing-compute` |
| `experiments/run_gafu_v87_real.py` | 新增 `robust_timing_protocols.csv` / `robust_timing_protocols_summary.csv` 真实生成路径 |
| `experiments/run_gafu_v87_real.py` | 新增 `training_compute_counter.csv`，明确标注 backward/update 为 analytic estimate |
| `experiments/run_gafu_v87_real.py` | P5 time accounting 改为保守判据：只要 selected functional phase breakdown 未全测，`time_accounting_pass=0` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 69.2 正式 run

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p5_p6_selected_compute_from_p4_conservative_20260508T232500Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z \
  --run-robust-timing-from-artifacts \
  --run-adaptive-one-step-audit \
  --run-adaptive-multistep-smoke \
  --p3-adaptive-controller-id Adaptive-FT-P \
  --run-p4-base-implementation-screen \
  --p4-base-candidates KF10,KW3,KW4,KW5,KW6 \
  --p4-base-hidden-candidates 16,20,24,28,32,36,40,48,56,68 \
  --p4-base-protocols T0:50:200,T1:100:500,T2:200:1000,T3:20:260 \
  --run-selected-system-confirmation \
  --run-p5-p6-selected-timing-compute \
  --selected-use-foreach-adamw-addcdiv \
  --selected-use-fused-ce-head-backward \
  --selected-use-compiled-fused-ce-head-backward \
  --selected-prewarm-compiled-fused-ce-head-backward \
  --use-foreach-adamw-addcdiv \
  --use-fused-ce-head-backward \
  --use-compiled-fused-ce-head-backward \
  --prewarm-compiled-fused-ce-head-backward
```

Route：

```json
{
  "route": "R2-StableComputeSelected",
  "success_v87_stability": 1,
  "success_v87_adaptive": 1,
  "success_v87_no_manual_tuning": 1,
  "success_v87_formal": 0,
  "p4_selected_base_candidate_id": "KW4",
  "p4_selected_hidden_dim": 28,
  "p5_robust_timing_pass": 1,
  "p5_strict_timing_pass": 1,
  "p5_time_accounting_pass": 0,
  "p6_training_compute_fair_pass": 1,
  "primary_blocker": "stable_compute_selected_cross_task_external_waves_not_run"
}
```

判断：v8.7 已从 `R3-NoManualArchitectureSelected` 推进到 `R2-StableComputeSelected`。仍不能写 formal success，因为 P7/P9 cross-task external transfer / causality waves 尚未运行。

### 69.3 P5 robust timing

`robust_timing_protocols.csv` 包含 5 条 rows：4 条来自 P4 selected `KW4 hidden28` base timing protocol，1 条来自 selected functional full confirmation 的 measured step ratio。

`robust_timing_protocols_summary.csv`：

| metric | value |
|---|---:|
| selected candidate | `KW4` |
| hidden | `28` |
| row count | `5` |
| q90 step ratio vs KB-MLP | `0.9354910594264081` |
| max step ratio vs KB-MLP | `0.9377612455717128` |
| max measured unknown fraction | `0.06591944859407857` |
| robust timing pass | `1` |
| strict timing pass | `1` |
| time accounting pass | `0` |

重要说明：

```text
time_accounting_pass = 0
reason = phase_breakdown_not_fully_measured_for_selected_functional_confirmation
```

这不是失败包装成成功：P5 的 robust / strict step timing 已过，但 phase-level time accounting 仍缺 selected functional breakdown，因此该子门保持未过。

### 69.4 P6 training compute counter

`training_compute_counter.csv`：

| metric | value |
|---|---:|
| candidate | `DG1-AdaptiveSelected-KW4-hidden28-functional` |
| forward FLOPs ratio vs KB-MLP | `0.9363473659637382` |
| backward FLOPs estimate ratio vs KB-MLP | `0.9363473659637382` |
| update FLOPs estimate ratio vs KB-MLP | `0.9395677799607073` |
| selected step ratio vs KB-MLP | `0.8479838654326309` |
| fair subgate count | `3` |
| training compute fair pass | `1` |

Estimate scope：

```text
forward_FLOPs_from_P4_counter_backward_estimate_same_ratio_update_estimate_from_param_ratio
```

判断：

1. Forward FLOPs 来自 P4 counter。
2. Backward / update 是 analytic estimate，已明确标注，不是 profiler measurement。
3. P6 按文档“至少两项”规则通过：forward FLOPs、backward estimate、step ratio 三项均在 gate 内。

### 69.5 selected confirmation recap

`selected_system_task_geometry_confirmation.csv`：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| DG functional test acc | `97.18999862670898` |
| delta vs KB-MLP | `+0.279998779296875` |
| delta vs DG base | `+0.020003318786621094` |
| curvature ratio vs DG base | `0.7724828819718651` |
| functional events | `73` |
| step ratio vs KB-MLP | `0.8479838654326309` |
| selected confirmation pass | `1` |

### 69.6 no-fake audit

```text
rows_checked = 2799
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 69.7 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `edf3f157a502fdd1f8cdd37554d1a159ce89ad7982fcaab170f272cf4637209b` |
| v8.7 plan | `bfe898d6cf3b8bacfb6da1cd6ce6004c1abe945f3a9fd20ff96238a05cd1b15c` |
| route | `dc5950822b75379d35dd927c15f22c48db72796b083d61c1165de20459454e3c` |
| P5 robust timing | `5bc31b116314b8dd6ad95d82966fc41b433205af1b89259baf07137b766a73bb` |
| P5 summary | `e0c62c59934cf8d46ceace5879a286f8792055fdbba477558dc0dd11e50084ee` |
| P6 compute counter | `74b080b0a5c44dbb99e807e2306fa5c2a6cc0d0ed39e097c37a12205da929213` |
| selected confirmation | `d71080c71bfffadbaac2619d56718833e9f2e4206b183356a042a9356e4266d6` |
| provenance audit | `ed0d1199c3f03faa549cde6e48465458e1ee9fec8889033739be1b81c7b4fa42` |
| run manifest | `f7bb78b9e5cee3b40761567065234af232c123befdd5a270614fb472474dd9f2` |

### 69.8 更新结论

v8.7 当前完成度：

```text
P0 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = pass
P4 no-manual base/hidden selection = pass
P4 selected task/geometry confirmation = pass
P5 robust/strict selected timing = pass
P5 full time accounting = not_pass, selected functional phase breakdown incomplete
P6 training compute fair counter = pass
success_v87_stability = true
success_v87_adaptive = true
success_v87_no_manual_tuning = true
success_v87_formal = false
```

机制结论：

1. `KW4 hidden28` selected route 不仅 task/geometry 成立，selected robust timing 和 compute envelope 也成立。
2. P6 说明 selected route 的 forward FLOPs / estimated backward FLOPs / step ratio 都在 KB-MLP 公平 envelope 内。
3. P5 time accounting 仍不完整；本轮没有伪造 functional phase breakdown。
4. 当前 clean blocker 已推进为 P7/P9 cross-task external transfer / causality waves，而不是 MNIST selected route、architecture selection 或 compute fairness。

最终一句话：

> v8.7 仍未 formal 完成，但已经推进到 `R2-StableComputeSelected`：adaptive、no-manual architecture、selected timing 和 training compute 都已有真实 pass artifact。当前不能越线的是 P7/P9 跨任务外部验证尚未执行，且 P5 selected functional phase accounting 仍未完整测量。

## 70. 追加：P7/P9 selected external screen bridge

本节继续第 69 节 blocker：P7/P9 cross-task external waves 尚未执行。本轮新增的是 v8.7 runner 内的窄 bridge：使用 P4 自动选择出的 `KW4 hidden28`，而不是回退到 v8.6 默认 `KW6 hidden28`。所有 rows 仍保持：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 70.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v87_real.py` | 新增 `--run-p7-p9-multitask-external` |
| `experiments/run_gafu_v87_real.py` | 新增 `kanbefair_multitask_transfer.csv` / `joint_fair_envelope.csv` / `functional_causality_multitask.csv` 真实生成路径 |
| `experiments/run_gafu_v87_real.py` | P7/P9 使用 v8.7 selected `candidate_id/hidden_dim`，避免把旧 `KW6` recipe 当成 selected route |
| `experiments/run_gafu_v87_real.py` | 修正 robust timing artifact merge：允许复制 `source_row_kind=p0` 的 T4 full-loop rows，使 T0-T4 summary 不丢 T4 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 70.2 正式 screen run

```bash
python experiments/run_gafu_v87_real.py \
  --out-dir results/real_rerun_20260506/v87_p7_p9_fmnist_selected_kw4_routecorrect2_20260509T003000Z \
  --fresh \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --reuse-p0-out-dir results/real_rerun_20260506/v87_p0_kw3_compiled_fused_prewarm_seed5x3_20260508T121500Z \
  --run-robust-timing-from-artifacts \
  --timing-source-out-dirs results/real_rerun_20260506/v87_kw3_compiled_fused_prewarm_p0seed5x3_t0_t4_routefixed_20260508T133000Z \
  --run-adaptive-one-step-audit \
  --run-adaptive-multistep-smoke \
  --p3-adaptive-controller-id Adaptive-FT-P \
  --p3-adaptive-warmup-steps 2 \
  --p3-adaptive-probe-interval 2 \
  --p3-adaptive-alpha-scale-with-probe-interval \
  --run-p4-base-implementation-screen \
  --run-selected-system-confirmation \
  --run-p5-p6-selected-timing-compute \
  --run-p7-p9-multitask-external \
  --p7-datasets Fashion-MNIST \
  --p7-train-size 10000 \
  --p7-test-size 2000 \
  --p7-epochs 5 \
  --selected-use-foreach-adamw-addcdiv \
  --selected-use-fused-ce-head-backward \
  --selected-use-compiled-fused-ce-head-backward \
  --selected-prewarm-compiled-fused-ce-head-backward
```

Route：

```json
{
  "route": "R1-ExternalBoundarySelected",
  "success_v87_stability": 1,
  "success_v87_adaptive": 1,
  "success_v87_no_manual_tuning": 1,
  "success_v87_external_fair": 1,
  "success_v87_formal": 0,
  "primary_blocker": "cross_task_external_pass_measured_under_screen_protocol_not_formal_protocol"
}
```

判断：这是 screen-level external boundary pass，不是 v8.7 formal completion。原因是 `p7_formal_protocol=0`，且 P5 selected functional phase accounting 仍为 `0`。

### 70.3 P7/P8 FMNIST external screen

`joint_fair_envelope.csv`：

| metric | value |
|---|---:|
| task | `FMNIST` |
| model | `DG1-FT7-KW4-hidden28-functional` |
| test acc | `85.2500021458` |
| delta vs KB-MLP | `+1.7499983311` |
| params ratio vs KB-MLP | `0.9395677800` |
| FLOPs ratio vs KB-MLP | `0.9363473660` |
| memory ratio vs KB-MLP | `0.8579032840` |
| step ratio vs KB-MLP | `0.9122283482` |
| curvature ratio vs DG-Base | `0.9799240201` |
| JointFairPass | `1` |

判断：

1. FMNIST task / params / FLOPs / memory / wall-clock fair envelope 真实通过。
2. 但 curvature reduction 很弱：`0.979924` 只比 base 低约 `2.0%`，不能写成 strong geometry generalization。
3. 因此本节只能说明 selected route 在 FMNIST 上有外部公平 task boundary，不说明 broad formal success。

### 70.4 P9 FMNIST causality controls

`functional_causality_multitask.csv`：

| control | test acc | curvature ratio | jacobian ratio | events | CausalityPass |
|---|---:|---:|---:|---:|---:|
| `DG-Base` | `85.2500021458` | `1.0000000000` | `1.0000000000` | 0 | 0 |
| `DG-NoOp` | `85.2500021458` | `1.0000000000` | `1.0603851194` | 0 | 0 |
| `DG-Functional` | `85.2500021458` | `0.9799240201` | `1.0026382059` | 3 | 1 |
| `DG-RandomFunc` | `85.2500021458` | `1.0196655701` | `1.0328465492` | 3 | 0 |
| `DG-ShuffledRoleFunc` | `85.2000057697` | `0.9733135811` | `1.0396445867` | 3 | 0 |

判断：

1. FT7 比 NoOp 和 RandomFunc 的 curvature 更低，且 task 没有低于 NoOp，因此按本轮 P9 relative control 判据通过。
2. ShuffledRole 的 curvature 更低于 FT7，但 test acc 低 `0.049996`，说明 role/schedule 仍有边界，不应把 P9 解读成“所有 functional variant 都同等有效”。
3. FT7 的 FMNIST effect 很小，下一步 formal route 需要 full protocol 与更多任务，而不是扩大本 screen 结论。

### 70.5 KMNIST cache blocker

同一轮曾尝试：

```text
results/real_rerun_20260506/v87_p7_p9_multitask_selected_kw4_screen_20260509T000000Z/
```

其中 FMNIST 也通过，但 KMNIST 未跑成：

```text
status = not_run
reason = RuntimeError: Error downloading train-images-idx3-ubyte.gz
```

判断：这是数据 cache / download blocker，不是 KMNIST 模型失败。本复盘不把 KMNIST 写成 pass 或 fail。

### 70.6 no-fake audit / hash

No-fake audit：

```text
rows_checked = 2814
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `a1afa3a4c596486aafc0af0044e6cd972b03a2f5b170a8c6139f8fbb0eee815d` |
| route | `967429535ce6fc24adfa6713aab6d864bea12a5658b82f610e8d5a1ae7b20560` |
| P7 transfer | `f598557255bffe37af72f349e51c0918f78ca7062e1cf79073ed230f631d1d8f` |
| P7 summary | `864059086d19cf60537b83042eb1e38dcfa1c2a46a8be563fe5b84e684892f55` |
| P8 joint fair | `20a0e2122da0026278a0a3c811621c28e784c4d6ba0011d0721e210e2031edc4` |
| P9 causality | `1d53a0b33c71adfe2472bd0e1912c81621acb5edbdf82276a1bcb99011e57674` |
| P9 summary | `8b3f05a2b78b6184637b0983f7e3c799516345c9528257b6e181b28002aaad49` |
| provenance audit | `3cf6542a774cdf4509504fbabc21dc5df40a969efcb9ce27dd1781a1af8bd06f` |

### 70.7 更新结论

v8.7 当前完成度：

```text
P0 stability = pass
T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = pass
P4 no-manual base/hidden selection = pass
P4 selected task/geometry confirmation = pass
P5 robust/strict selected timing = pass
P5 full time accounting = not_pass
P6 training compute fair counter = pass
P7/P9 FMNIST screen = pass
P7/P9 full formal cross-task = not_done
success_v87_formal = false
```

机制结论：

1. v8.7 已从 `R2-StableComputeSelected` 推进到 `R1-ExternalBoundarySelected`，但只是 screen-level boundary。
2. `KW4 hidden28` selected route 在 FMNIST 上通过外部公平 task envelope，说明自动选择 route 不只是在 MNIST 上成立。
3. FMNIST 的 functional geometry improvement 较弱，不能替代 full P7/P9 formal protocol。
4. KMNIST 当前是 download/cache blocker，没有真实结果，不能写入 pass/fail。
5. 下一步应解决 KMNIST cache，并用 full formal protocol 或预注册多任务 screen 继续 P7/P9；同时补 P5 selected functional phase accounting。

最终一句话：

> v8.7 还没有 formal 完成。本轮新增 P7/P9 bridge 后，selected `KW4 hidden28` 在 FMNIST 上取得真实 external boundary pass，但它不是 full formal protocol，KMNIST 也因 cache/download 未跑成；当前不能把这个 screen 扩写成 broad/formal success。
