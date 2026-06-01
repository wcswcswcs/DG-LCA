# DG-KAN v9.1 BasisJustification CleanFullEdge Redesign 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.1_BasisJustification_CleanFullEdge_Redesign_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论或把未实现阶段写成通过。

## 0. 最新结论

> 本文件按执行顺序追加。前面第 1-13 节保留 first-wave 结果；最新结论以第 18 节为准。

截至第 18 节，v9.1 继续执行到一个新的可审计 terminal route：

```text
route = R8-BasisJustificationIncomplete
success_v91_basis_justification = false
success_v91_full_edge_external_fair = false
success_v91_compute_repair = false
success_v91_symbolic = false
success_v91_robustness = false
```

最新 artifact：

```text
results/real_rerun_20260506/v91_basis_justification_implclosure_20260509T021500Z/
```

最新结论：

1. 已补齐 planned basis family 的 dense manual full-edge forward/backward，并且 BAS0-BAS9 的 P3 gradcheck 全部通过。
2. P4 512-sample overfit 中，BAS0、BAS6、BAS7、BAS9 通过；BAS1/BAS2/BAS3/BAS4/BAS5/BAS8 未过。
3. P1 raw-input conditioning 仍然 0/10 basis 通过；所有 basis 都卡在 dominant component 或 condition number。
4. 因此没有任何 basis 同时通过 conditioning + fit + gradcheck + overfit，`justified_basis_count_wave0_to_wave2 = 0`。
5. v9.1 仍不能进入 FullEdge external success；下一步应该先修 conditioning / feature-source / fan-in normalization，再做 shared/low-rank compute repair。

first-wave artifact：

```text
results/real_rerun_20260506/v91_basis_justification_firstwave_20260509T010000Z/
```

first-wave 结论：

1. v9.1 已真实执行 P0-P4：basis registry、raw-input conditioning、one-layer fit、manual gradcheck、512-sample overfit。
2. P5-P8 对 v9.0 已真实测过的 FullEdge external / causality / timing / compute rows 做可追溯重整；每行记录 `source_artifact`，没有伪装成新训练。
3. 当前 planned basis set 没有完整 full-edge manual backward / overfit / full-task implementation，尤其 normalized orthogonal、RBF8、cubic B-spline、rational Pade 仍只是 diagnostic。
4. 已实现的 DenseFullEdge basis 中，`BAS0-MonomialCompactPoly3` 与 `BAS6-PiecewiseLinear4` 能在 MNIST 512-sample overfit，但 P1 raw-input conditioning 均未过，且 v9.0 external rows 已证明外部公平仍失败。
5. 因此 v9.1 不能进入 basis-justified FullEdge success；下一步必须先实现 normalized orthogonal / shared-basis / low-rank full-edge manual paths，再重跑 Wave2-Wave8。

## 1. 本轮代码

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v91_basis_justification.py` | v9.1 basis justification runner；生成 P0-P11 required artifacts、route、failure table、no-fake audit |

复用：

| 文件 | 作用 |
|---|---|
| `dgkan/models/manual_full_edge.py` | v9.0 manual FullEdge basis implementation |
| `dgkan/training/manual_full_edge.py` | v9.0 manual CE training / AdamW / functional smoothing helper |

代码检查：

```text
python -m py_compile experiments/run_v91_basis_justification.py
```

已通过。

## 2. 运行命令

```bash
python experiments/run_v91_basis_justification.py \
  --out-dir results/real_rerun_20260506/v91_basis_justification_firstwave_20260509T010000Z \
  --fresh \
  --device auto \
  --data-root data \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seed 1314 \
  --batch-size 128 \
  --p1-train-size 512 \
  --p1-max-values 20000 \
  --p1-seeds 0,1,2 \
  --p2-grid-size 2048 \
  --p4-steps 300 \
  --v90-artifact-dir results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z
```

## 3. Route

`route_decision.json`：

```json
{
  "route": "R8-BasisJustificationIncomplete",
  "success_v91_basis_justification": 0,
  "success_v91_full_edge_external_fair": 0,
  "success_v91_compute_repair": 0,
  "success_v91_symbolic": 0,
  "success_v91_robustness": 0,
  "basis_registry_pass": 1,
  "planned_full_basis_implementation_complete": 0,
  "promotion_pass_count": 0,
  "external_fair_pass_count": 0,
  "primary_blocker": "planned_basis_families_not_fully_implemented_for_gradcheck_overfit_and_full_task",
  "next_required_implementation": "implement_normalized_orthogonal_and_shared_lowrank_full_edge_manual_paths_then_rerun_wave2_to_wave8"
}
```

判断：本轮没有选择任何未通过 Wave0-Wave2 的 basis 进入成功声明。由于 planned basis family 的 full-edge manual path 不完整，路线按计划停在 `R8-BasisJustificationIncomplete`。

## 4. P0 basis registry audit

`basis_registry_audit.csv` 记录了计划中的 10 个 basis：

| basis | name | status |
|---|---|---|
| BAS0 | MonomialCompactPoly3 | implemented as `compact_poly_silu3` |
| BAS1 | NormalizedLegendre3 | diagnostic only，full-edge backward 未实现 |
| BAS2 | NormalizedChebyshev3 | diagnostic only，full-edge backward 未实现 |
| BAS3 | NormalizedHermite3 | diagnostic only，full-edge backward 未实现 |
| BAS4 | SharedRBF4 | current dense `rbf4` implemented，但不是 shared-basis repaired |
| BAS5 | SharedRBF8 | diagnostic only，full-edge backward 未实现 |
| BAS6 | PiecewiseLinear4 | implemented as triangular hat `spline4` |
| BAS7 | CubicBSpline4 | diagnostic only，full-edge backward 未实现 |
| BAS8 | CubicBSpline8 | diagnostic only，full-edge backward 未实现 |
| BAS9 | RationalPade2 | diagnostic only，需要 stability audit |

P0 registry 本身通过：

```text
basis_registry_pass = 1
```

但这只说明公式 / params / FLOPs / geometry compatibility 被记录，不代表 full-edge official implementation 完整。

## 5. P1 basis activation conditioning

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
input_source = raw_flattened_input
train_size = 512
seeds = 0,1,2
max sampled values = 20000
```

结果：

| basis | pass rows | max condition | max dead frac | max dominant frac |
|---|---:|---:|---:|---:|
| BAS0 | `0/9` | `44553.128906` | `0.000000` | `0.991656` |
| BAS1 | `0/9` | `1890.537476` | `0.000000` | `0.936183` |
| BAS2 | `0/9` | `2615.537842` | `0.000000` | `0.962539` |
| BAS3 | `0/9` | `61770.867188` | `0.000000` | `0.995236` |
| BAS4 | `0/9` | `1744.549683` | `0.000000` | `0.875923` |
| BAS5 | `0/9` | `560233840640.000000` | `0.000000` | `0.870722` |
| BAS6 | `0/9` | `12394724.000000` | `0.250000` | `0.804349` |
| BAS7 | `0/9` | `1280.144043` | `0.000000` | `0.824626` |
| BAS8 | `0/9` | `2193285.250000` | `0.000000` | `0.811982` |
| BAS9 | `0/9` | `11161.389649` | `0.000000` | `0.993303` |

判断：

1. 按计划阈值 `condition <= 1e4`、`dead <= 0.30`、`dominant <= 0.70`，所有 basis 在 raw flattened input 上都未通过 P1。
2. 大多数 failure 不是 dead basis，而是 dominant component 过强，说明 raw pixel distribution 下 basis channel covariance 很不均衡。
3. `transitional_hidden_features` 本轮没有实现 extractor，已写为 `not_run`，没有用 proxy feature 替代。

## 6. P2 one-layer fit diagnostics

设置：

```text
grid = [-2,2]
grid_size = 2048
targets = identity, silu, quadratic, piecewise_ramp, local_bump, sinusoidal, symbolic_polynomial
fit method = torch.linalg.lstsq
```

Simple target pass：

| basis | identity/quadratic/silu pass | local bump MSE |
|---|---:|---:|
| BAS0 | `3/3` | `0.046292592` |
| BAS1 | `3/3` | `0.047517717` |
| BAS2 | `3/3` | `0.047517724` |
| BAS3 | `3/3` | `0.047517721` |
| BAS4 | `3/3` | `0.015103368` |
| BAS5 | `3/3` | `0.000730471` |
| BAS6 | `1/3` | `0.010368602` |
| BAS7 | `3/3` | `0.011270877` |
| BAS8 | `3/3` | `0.002370813` |
| BAS9 | `3/3` | `0.030017031` |

判断：

1. RBF / cubic B-spline diagnostics 在 local bump 上明显优于 compact poly，支持计划里“local basis 适合局部函数”的 justification。
2. BAS6 piecewise linear 对 simple smooth targets 只过 `1/3`，但 local bump MSE 优于 BAS0。
3. `random_projection_target_from_transitional_hidden` 未运行，因为 transitional hidden extractor 本轮未实现；没有使用伪 feature。

## 7. P3 manual gradcheck

通过：

| basis | edge kind | GradRelErrMax | GradCosMin | GradPass |
|---|---|---:|---:|---:|
| BAS0 | `compact_poly_silu3` | `2.463477e-07` | `1.000000` | 1 |
| BAS4 | `rbf4` dense | `0.000000` | `0.999999881` | 1 |
| BAS6 | `spline4` triangular hat | `0.000000` | `1.000000` | 1 |

未运行：

```text
BAS1/BAS2/BAS3/BAS5/BAS7/BAS8/BAS9
reason = full-edge manual backward not implemented / stability audit required
```

判断：当前可进入 small-overfit 的只有 BAS0、BAS4、BAS6；normalized orthogonal / RBF8 / cubic B-spline / rational basis 仍不具备 official full-edge training 合同。

## 8. P4 small overfit smoke

设置：

```text
dataset = MNIST
train subset = 512
steps = 300
functional_update = none
```

结果：

| basis | dims | train acc | train loss | OverfitPass |
|---|---|---:|---:|---:|
| BAS0 MonomialCompactPoly3 | `784x11x11x10` | `1.000000` | `0.000774818` | 1 |
| BAS4 SharedRBF4 dense diagnostic | `784x8x10` | `0.9765625` | `0.791095674` | 0 |
| BAS6 PiecewiseLinear4 | `784x8x10` | `0.994140625` | `0.599354088` | 1 |

判断：

1. BAS0 和 BAS6 具备小样本 overfit 能力。
2. BAS4 当前 dense RBF4 未达到 `TrainAcc >= 0.98`，不能按计划晋级 official full task。
3. 但 BAS0/BAS6 的 P1 conditioning 仍未过，因此不能把 P4 overfit pass 单独当成 basis justification pass。

## 9. P5-P8 external / causality / compute

本轮没有重新训练 FullEdge external；P5-P8 对 v9.0 已测 artifact 做重整，并在 CSV 中记录：

```text
source_artifact = results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z
```

重整 rows：

| artifact | rows |
|---|---:|
| `basis_parameterization_matrix.csv` | `18` |
| `functional_basis_causality.csv` | `12` |
| `materialization_free_compute_audit.csv` | `3` |
| `external_fair_validation.csv` | `3` |

Summary：

```text
promotion_pass_count = 0
external_fair_pass_count = 0
functional_useful_pass_count > 0
compute_repair_pass_count = 0
```

关键复用结论仍与 v9.0 一致：

| task | best FullEdge functional | acc | delta vs KB-MLP | forward FLOPs ratio | backward estimate ratio | memory issue | external fair |
|---|---|---:|---:|---:|---:|---|---:|
| MNIST | `CompactPolySilu3-h11x2+Functional` | `94.949996` | `-2.039999` | `1.550366` | `3.100731` | yes | 0 |
| Fashion-MNIST | `CompactPolySilu3-h11+Functional` | `85.670000` | `-1.309997` | `1.529181` | `3.058361` | yes | 0 |
| KMNIST | `CompactPolySilu3-h11x2+Functional` | `78.679997` | `-4.510003` | `1.550366` | `3.100731` | yes | 0 |

判断：

1. Functional causality 有真实几何信号，但 external fair 仍未过。
2. 当前 dense FullEdge 的主要系统问题仍是 compute / memory envelope，尤其 backward estimate ratio 约 `3.06-3.10`。
3. v9.1 的 shared-basis / low-rank compute repair 尚未实现，所以不能声称 H4 已解决。

## 10. P9/P10/P11 boundary

P9 symbolic validation：

```text
status = not_run
reason = symbolic task runner not implemented in v9.1 first execution
```

P10 robustness validation：

```text
status = not_run
reason = robustness perturbation runner not implemented in v9.1 first execution
```

P11 boundary audit：

| basis | boundary labels |
|---|---|
| BAS0 | `basis_not_conditioned` |
| BAS1 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |
| BAS2 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |
| BAS3 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |
| BAS4 | `basis_not_conditioned,basis_small_overfit_fail_or_not_run` |
| BAS5 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |
| BAS6 | `basis_not_conditioned` |
| BAS7 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |
| BAS8 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |
| BAS9 | `basis_not_conditioned,basis_implementation_incomplete,basis_gradcheck_missing_or_fail,basis_small_overfit_fail_or_not_run` |

Global boundary：

```text
R8-BasisJustificationIncomplete
primary_blocker = planned_basis_families_not_fully_implemented_for_gradcheck_overfit_and_full_task
```

## 11. No-fake audit

```text
rows_checked = 248
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P0-P4 是本轮新测。
2. P5-P8 是对 v9.0 已测 rows 的 source-artifact reuse，不是 fake/proxy。
3. P9/P10 未运行，明确写 `not_run`。

## 12. Hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v91_basis_justification.py` | `86fd0e2936f9f0c9033f1b7a0afe677cb2b52b9d24fa60732c1c0b02b173b468` |
| v9.1 plan | `58c3d9cf3b3d30cc702784d4957bfb00afa43da6520f01145ce829f10ad28ecd` |
| `basis_registry_audit.csv` | `dd15b12dea5180f21505544fe6b8e92217caa2d8ee31f1cec8d282b6e8355b4b` |
| `basis_activation_conditioning.csv` | `62aedf42942d9d0a06f2f1bd4ef63eaf588ceb46f7f12e54d7bedc0db514df76` |
| `basis_fit_diagnostics.csv` | `39b74c00373b9e8d7abfb08d4b146c09dfa3287af2607d626656df5b3319eafe` |
| `basis_gradcheck.csv` | `99803fb4c4cc33f7537933475e8b504cfee099f4efe70616fa6a41a056158b59` |
| `basis_small_overfit.csv` | `d057502fcc5b6492e55cf62a9ded359ca44b5fdfdac6878d32c0ffedfac8b723` |
| `basis_parameterization_matrix.csv` | `1b1e21d919148dc1919030824ae71286b18d4e881301b44c6f8d8a9c09f3f05d` |
| `functional_basis_causality.csv` | `86e7783d5162c1cd3c80fc1bf2a6cfca43372666d9203d209566a984446d9d3a` |
| `materialization_free_compute_audit.csv` | `e59f2d0bbb6d958f365e31451fc0c1c42cf661fce379df5b423ea500c4125e6f` |
| `external_fair_validation.csv` | `a591632545036e745112f6a6c147fcdf13292ba760a3d1f068f53c87bbe99f55` |
| `basis_boundary_audit.csv` | `34e4cbbfe31be3912c4b46f156490afa1a67057110c4e3da9a669d195b3b5f40` |
| route | `d6f356aff1a50d498360b6c48f8c468454ac19fe9da29cac5e5098700f1a46fd` |
| provenance audit | `c6b11ed7384a207577d3cb4c4125c764a24d77d0bfd7c1ee1f59f28bc63432c8` |

## 13. 最终分析结论

v9.1 本轮没有证明 FullEdge PureKAN 成功，也没有证明 FullEdge 形式失败。它证明的是一个更窄但更关键的事实：

```text
当前 FullEdge basis selection protocol 还不完整；
现有 dense basis 中有 overfit 能力和 functional geometry 信号；
但 raw-input conditioning、planned basis implementation、compute repair 和 external fair 都没有闭合。
```

机制判断：

1. `BAS0` / `BAS6` 能 overfit 512 samples，说明 full-edge manual training 不是完全不可达。
2. P1 全部 basis 在 raw flattened input 上 conditioning fail，说明必须做 basis normalization / feature-space input source / fan-in normalization，不能直接把 raw pixel distribution 喂给 dense edge basis 后声明 justified。
3. RBF / B-spline 在 one-layer local bump target 上明显更合适，但它们的 full-edge manual / shared-basis compute path 尚未实现。
4. v9.0 external 失败不能被解释为 “FullEdge 永远不行”；但 v9.1 也不能把未实现的 normalized/shared/lowrank basis 当作成功路线。

下一步必须做：

```text
1. 实现 BAS1/BAS2/BAS3 的 full-edge manual backward + gradcheck；
2. 实现 BAS5/BAS7/BAS8 的 shared-basis 或 low-rank materialization-free path；
3. 补 transitional hidden feature extractor，重跑 P1；
4. 重跑 P3/P4，只有通过 conditioning + gradcheck + overfit 的 basis 才进入 P5；
5. 再做 P6-P8 external fair validation。
```

最终一句话：

> v9.1 第一轮按计划执行后，诚实停在 `R8-BasisJustificationIncomplete`：现在不是继续盲试 basis 的阶段，而是先补齐 normalized orthogonal / shared low-rank full-edge 实现和 conditioning 闭环；否则任何 FullEdge external success 或 failure 结论都仍然缺 basis justification。

## 14. 追加：planned basis manual implementation closure

根据第 13 节的 blocker，本轮继续补齐 planned basis 的 dense manual FullEdge 路径，目标是排除 “manual backward 未实现导致 R8” 这个解释。所有新增实现仍遵守：

```text
loss_type = CE
label_smoothing = 0
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake/proxy = 0
```

代码改动：

| 文件 | 改动 |
|---|---|
| `dgkan/models/manual_full_edge.py` | 新增 `legendre3`、`chebyshev3`、`hermite3`、`rbf8`、`cubic_bspline4`、`cubic_bspline8`、`rational_pade2` 的 manual forward/backward |
| `experiments/run_v91_basis_justification.py` | P0 registry 改为记录这些 basis 的 dense implementation；P3 autograd-reference gradcheck 覆盖 BAS0-BAS9；route 改为区分 implementation incomplete 与 Wave0-2 未闭合 |

代码检查：

```text
python -m py_compile dgkan/models/manual_full_edge.py experiments/run_v91_basis_justification.py
```

已通过。

正式 rerun：

```bash
python experiments/run_v91_basis_justification.py \
  --out-dir results/real_rerun_20260506/v91_basis_justification_implclosure_20260509T021500Z \
  --fresh \
  --device auto \
  --data-root data \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seed 1314 \
  --batch-size 128 \
  --p1-train-size 512 \
  --p1-max-values 20000 \
  --p1-seeds 0,1,2 \
  --p2-grid-size 2048 \
  --p4-steps 300 \
  --v90-artifact-dir results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z
```

## 15. Updated route

`route_decision.json`：

```json
{
  "route": "R8-BasisJustificationIncomplete",
  "basis_registry_pass": 1,
  "planned_full_basis_implementation_complete": 1,
  "justified_basis_count_wave0_to_wave2": 0,
  "justified_basis_ids_wave0_to_wave2": "",
  "promotion_pass_count": 0,
  "external_fair_pass_count": 0,
  "success_v91_basis_justification": 0,
  "success_v91_full_edge_external_fair": 0,
  "success_v91_compute_repair": 0,
  "primary_blocker": "no_basis_passed_conditioning_fit_gradcheck_and_small_overfit",
  "next_required_implementation": "fix_basis_conditioning_with_fan_norm_or_feature_source_then_implement_shared_lowrank_compute_repair_and_rerun_wave3_to_wave8"
}
```

判断：

1. 本轮已经排除了 “BAS1/BAS2/BAS3/BAS5/BAS7/BAS8/BAS9 没有 manual backward” 这个 first-wave blocker。
2. 新 blocker 更硬：没有任何 basis 同时通过 P1 conditioning、P2 fit、P3 gradcheck、P4 small overfit。
3. 因此 v9.1 继续停在 R8，但不是同一个原因；现在是 basis justification 的数值闭环失败。

## 16. P1 / P3 / P4 updated results

P1 raw-input conditioning：

| basis | pass rows | max condition | max dominant frac | max dead frac |
|---|---:|---:|---:|---:|
| BAS0 | `0/9` | `44553.128906` | `0.991656` | `0.000000` |
| BAS1 | `0/9` | `1890.537476` | `0.936183` | `0.000000` |
| BAS2 | `0/9` | `2615.537842` | `0.962539` | `0.000000` |
| BAS3 | `0/9` | `61770.867188` | `0.995236` | `0.000000` |
| BAS4 | `0/9` | `1744.549683` | `0.875923` | `0.000000` |
| BAS5 | `0/9` | `560233840640.000000` | `0.870722` | `0.000000` |
| BAS6 | `0/9` | `12394724.000000` | `0.804349` | `0.250000` |
| BAS7 | `0/9` | `1280.144043` | `0.824626` | `0.000000` |
| BAS8 | `0/9` | `2193285.250000` | `0.811982` | `0.000000` |
| BAS9 | `0/9` | `11161.389649` | `0.993303` | `0.000000` |

P3 gradcheck：

| basis | GradRelErrMax | GradCosMin | GradPass |
|---|---:|---:|---:|
| BAS0 | `2.463477e-07` | `1.000000` | 1 |
| BAS1 | `0.000000` | `1.000000` | 1 |
| BAS2 | `0.000000` | `1.000000` | 1 |
| BAS3 | `0.000000` | `0.999999940` | 1 |
| BAS4 | `0.000000` | `0.999999881` | 1 |
| BAS5 | `0.000000` | `1.000000` | 1 |
| BAS6 | `0.000000` | `1.000000` | 1 |
| BAS7 | `0.000000` | `1.000000` | 1 |
| BAS8 | `0.000000` | `0.999999940` | 1 |
| BAS9 | `1.563150e-07` | `1.000000` | 1 |

P4 small overfit：

| basis | dims | train acc | train loss | OverfitPass |
|---|---|---:|---:|---:|
| BAS0 | `784x11x11x10` | `1.000000` | `0.000774818` | 1 |
| BAS1 | `784x11x11x10` | `0.380859` | `1.782633` | 0 |
| BAS2 | `784x11x11x10` | `0.480469` | `1.658861` | 0 |
| BAS3 | `784x11x11x10` | `0.609375` | `1.256531` | 0 |
| BAS4 | `784x8x10` | `0.976563` | `0.791096` | 0 |
| BAS5 | `784x6x10` | `0.296875` | `1.844241` | 0 |
| BAS6 | `784x8x10` | `0.994141` | `0.599354` | 1 |
| BAS7 | `784x8x10` | `0.980469` | `0.906086` | 1 |
| BAS8 | `784x6x10` | `0.972656` | `1.047446` | 0 |
| BAS9 | `784x11x11x10` | `0.998047` | `0.039985` | 1 |

判断：

1. BAS0/BAS6/BAS7/BAS9 能 overfit 512 samples，说明这些 dense basis 的训练不是完全不可达。
2. Normalized Legendre/Chebyshev/Hermite 的 dense stopgrad-normalized implementation 虽然 gradcheck 过，但 300-step 小样本 overfit 明显不足。
3. RBF4/RBF8 在本配置下没有过 `0.98` overfit gate。
4. 最关键的是：所有 basis 的 P1 conditioning 仍为 `0/9` pass，因此没有 basis 能被称为 justified official candidate。

## 17. P5-P8 reuse status and no-fake audit

P5-P8 本轮仍复用 v9.0 已测 FullEdge external rows，并保留 source tracking：

```text
source_artifact = results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z
```

rows：

| artifact | rows |
|---|---:|
| `basis_parameterization_matrix.csv` | `18` |
| `functional_basis_causality.csv` | `12` |
| `materialization_free_compute_audit.csv` | `3` |
| `external_fair_validation.csv` | `3` |

Summary：

```text
promotion_pass_count = 0
external_fair_pass_count = 0
compute_repair_pass_count = 0
```

No-fake audit：

```text
rows_checked = 248
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P0-P4 是本轮新测。
2. P5-P8 是对 v9.0 真实 rows 的可追溯重整，不是 fake/proxy。
3. P9/P10 仍是 `not_run`，没有补写 symbolic 或 robustness success。

## 18. Updated hash / final conclusion

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_v91_basis_justification.py` | `ef44a46db6c7252b00d20c63165784a2c8bd50bb733e9d5ce38a5ee041c2603c` |
| `dgkan/models/manual_full_edge.py` | `da66d01488b7d658ac6e671df39ff9ee139e54aadc41e69842ee7e852e2a28db` |
| v9.1 plan | `58c3d9cf3b3d30cc702784d4957bfb00afa43da6520f01145ce829f10ad28ecd` |
| `basis_registry_audit.csv` | `c1f1ff5b3f445a13ebb7440191b187f085b7f9c088284a15a258774a643d388b` |
| `basis_activation_conditioning.csv` | `62aedf42942d9d0a06f2f1bd4ef63eaf588ceb46f7f12e54d7bedc0db514df76` |
| `basis_gradcheck.csv` | `7ed79580bd2bbdb56521a60186660dff1eead401e9601e258f5b7373583629b6` |
| `basis_small_overfit.csv` | `98e9cd9878bc8dbe46eae076be825c4b7835d2b5012b60530af34388d28054a5` |
| route | `72e903f97e059b6e8bf11e5835e3ad5e4170c985a1de0667a2fc4ca268df71ec` |
| provenance audit | `c6b11ed7384a207577d3cb4c4125c764a24d77d0bfd7c1ee1f59f28bc63432c8` |

最终结论：

```text
planned full-edge basis manual implementation = closed for BAS0-BAS9 dense paths
P3 gradcheck = 10/10 pass
P4 overfit = 4/10 pass
P1 conditioning = 0/10 pass
justified_basis_count_wave0_to_wave2 = 0
external_fair_pass_count = 0
route = R8-BasisJustificationIncomplete
```

机制判断：

1. v9.1 第二轮已经不是 “basis 没实现所以不能判断”；现在 manual gradcheck 全部过，问题转移到 basis conditioning 与 small-overfit/compute/external fair。
2. Raw flattened pixel 分布对所有 basis 都产生强 dominant component；normalized orthogonal basis 降低了部分 condition number，但仍没有解决 dominant fraction。
3. BAS0/BAS6/BAS7/BAS9 的 overfit 能力说明 FullEdge 不是训练完全不可达；但 conditioning 未过，不能把它们升为 justified basis。
4. 下一步不能再只补 basis 公式，必须改 input source / fan-in normalization / feature-space basis evaluation，并实现 shared-basis 或 low-rank compute repair。

最终一句话：

> v9.1 继续执行后仍停在 `R8-BasisJustificationIncomplete`，但 blocker 已推进：planned dense manual basis 已补齐且 gradcheck 全过，真正卡点变成 raw-input conditioning 和 compute/external fair。当前仍不能声明任何 FullEdge basis justified 或 external success。
