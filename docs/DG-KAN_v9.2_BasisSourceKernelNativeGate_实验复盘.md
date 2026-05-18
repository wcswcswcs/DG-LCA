# DG-KAN v9.2 BasisSource KernelNativeGate 实验复盘

> 本复盘记录 `docs/DG-KAN_总计划与v9.2下一步完整实验计划_补充kernel_native_gate.md` 中 **Part II v9.2 下一步计划** 的本轮真实执行结果。本文不执行总计划的全部远期路线；只执行 v9.2 first-wave gate。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把未打开的 P5/P6/P7 写成通过。

## 0. 最新结论

本文件按执行顺序追加。前面第 1-12 节保留 first-wave P4 失败；第 13-16 节保留 P4 修复成功；最新结论以第 20 节为准。

截至第 20 节，v9.2 后续部分继续执行到 P5 terminal route：

```text
route = R4-AdamWTrainabilityFail
best_candidate = S3-B2-P4-F0
success_v92_basis_source = true
success_v92_full_edge_trainability = false
success_v92_functional_advantage = false
success_v92_external_fair = false
```

核心结论：

1. v9.1 的 raw-input conditioning blocker 被推进：`S3-EdgeAffineNorm + B2-Chebyshev3Residual` 在 MNIST / Fashion-MNIST / KMNIST、seed 0/1/2 上 `9/9` 通过 P1 conditioning。
2. first-wave 的 `P1 IdentityPlusLowRankResidual` P4 失败；后续新增 `P3 SharedBasisLowRank` 和 `P4 ActiveKSharedResidual`，并用 compiled + foreach update 真实重跑 P3/P4。
3. P4 已被修复：`S3-B2-P4-F0` active-k shared residual 在 hidden4096/rank1 和 hidden4096/rank4 下均真实通过 kernel-native gate。
4. 继续打开 P5 后，AdamW-only trainability 没有通过：三任务三 seed 的 9 行均低于同参数 MLP-match 的 `-0.01` tolerance。
5. 因此 P6 functional 与 P7 external fair 按计划不打开，均以 `not_run` 落盘；不能用 functional update 去救 AdamW-only 不成立的 candidate。

最新 artifact：

```text
results/real_rerun_20260506/v92_p5_adamw_trainability_activeP4_T2_h4096r4_3task3seed_e20_smoke_20260509T124500Z/
```

P4 pass artifact：

```text
results/real_rerun_20260506/v92_basis_source_kernel_gate_h4096r1_activeP4_compiled_foreach_20260509T110000Z/
```

first-wave artifact：

```text
results/real_rerun_20260506/v92_basis_source_kernel_gate_first_20260509T073000Z/
```

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | v9.2 Part II runner；生成 P0-P8 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v92_basis_source_kernel_gate.py
```

已通过。

正式运行：

```bash
python experiments/run_v92_basis_source_kernel_gate.py \
  --out-dir results/real_rerun_20260506/v92_basis_source_kernel_gate_first_20260509T073000Z \
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
  --p3-steps 600 \
  --p3-hidden-dim 64 \
  --p3-rank 16 \
  --p4-warmup 20 \
  --p4-reps 120
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R6-ComputeFail-KernelizationRequired",
  "best_candidate": "S3-B2-P1-F0",
  "best_source": "S3",
  "best_basis": "B2",
  "best_parameterization": "P1",
  "best_functional_mode": "F0-NoFunctional",
  "contract_pass": 1,
  "p1_source_basis_pass_count": 1,
  "p2_source_basis_survivor_count": 1,
  "p3_overfit_pass_count": 1,
  "p4_kernel_native_pass_count": 0,
  "primary_blocker": "P4_kernel_native_forward_backward_memory_or_flops_gate_failed",
  "next_required_implementation": "implement_torch_compile_or_triton_kernel_native_forward_backward_then_rerun_P3_P4",
  "success_v92_basis_source": 1,
  "success_v92_full_edge_trainability": 0,
  "success_v92_functional_advantage": 0,
  "success_v92_external_fair": 0
}
```

判断：v9.2 minimum 的前半段成立，即找到了一个 basis-source-parameterization 组合通过 conditioning / fit / gradcheck / overfit；但 minimum success 需要 microbench near-pass，本轮 P4 没过，因此不能写 v9.2 success。

## 3. P0 contract and registry

Artifact rows：

| artifact | rows |
|---|---:|
| `candidate_registry_v92.csv` | `150` |
| `contract_audit_v92.csv` | `150` |

结果：

```text
contract pass = 150/150
official P1 candidates materializes_dense_edge_tensor = 0
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
uses_loss_backward = 0
```

判断：本轮 v9.2 没有混入旧 transitional route，也没有把 dense materialization negative control 记成 official eligible。

## 4. P1 basis-source conditioning

P1 matrix：

```text
Sources = S0,S1,S2,S3,S4
Basis = B0,B1,B2,B3,B4,B5
Datasets = MNIST,Fashion-MNIST,KMNIST
Seeds = 0,1,2
rows = 270
```

完整通过的 source-basis 只有一个：

| source | basis | pass rows | max condition | max dominant frac | max dead frac |
|---|---|---:|---:|---:|---:|
| `S3-EdgeAffineNorm` | `B2-Chebyshev3Residual` | `9/9` | `81.249916` | `0.662755` | `0.000000` |

`S3-B2` per task：

| dataset | seeds | condition | dominant frac | dead frac | pass |
|---|---:|---:|---:|---:|---:|
| MNIST | `0,1,2` | `33.078354` | `0.616606` | `0.000000` | 1 |
| Fashion-MNIST | `0,1,2` | `9.133300` | `0.662755` | `0.000000` | 1 |
| KMNIST | `0,1,2` | `81.249916` | `0.634632` | `0.000000` | 1 |

部分通过但未晋级：

| source | basis | pass rows | max dominant frac | 失败原因 |
|---|---|---:|---:|---|
| `S3` | `B1` | `6/9` | `0.768510` | dominant 超过 `0.70` |
| `S3` | `B3` | `6/9` | `0.780696` | dominant 超过 `0.70` |
| `S4` | `B1` | `6/9` | `0.715021` | dominant 超过 `0.70` |
| `S0` | `B3` | `3/9` | `0.817142` | dominant 超过 `0.70` |
| `S1` | `B1` | `3/9` | `0.722524` | dominant 超过 `0.70` |
| `S2` | `B1` | `3/9` | `0.722524` | dominant 超过 `0.70` |

判断：

1. H1 得到支持：v9.1 的 `0/10 raw-input conditioning fail` 不是所有 basis family 都无效，而是 raw source 不合适。
2. `S3 EdgeAffineNorm` 明显改善 conditioning；Chebyshev3 residual 是本轮唯一跨三任务三 seed 稳定通过的 basis。
3. 其他接近项大多卡在 `dominant_basis_fraction > 0.70`，说明 source normalization 仍是主 blocker。

## 5. P2 one-layer fit diagnostics

只对 P1 survivor `S3-B2` 进入 P2。

| target | fit R2 | fit MSE | fit pass |
|---|---:|---:|---:|
| identity | `1.000000` | `2.343749e-16` | 1 |
| silu | `0.9999779463` | `1.946622e-06` | 1 |
| quadratic | `1.000000` | `7.438096e-16` | 1 |
| piecewise ramp | `0.9956105351` | `0.000759669` | 1 |
| local bump | `0.8645794988` | `0.016826110` | 1 |
| low-frequency sine | `0.9911794066` | `0.004408150` | 1 |
| symbolic polynomial | `1.000000` | `2.777086e-15` | 1 |
| transitional hidden projection | `not_run` |  |  |

说明：

```text
transitional_hidden_projection status = not_run
reason = transitional_hidden_extractor_not_implemented_in_v92_first_wave
```

判断：`S3-B2` 对 basic analytic targets 的近似能力足够进入 P3；没有用 proxy hidden feature 替代未实现 extractor。

## 6. P3 hybrid residual gradcheck and overfit

Candidate：

```text
S3-B2-P1-F0
source = EdgeAffineNorm
basis = Chebyshev3Residual
parameterization = IdentityPlusLowRankResidual
functional = off
hidden_dim = 64
rank = 16
```

Gradcheck：

| metric | value |
|---|---:|
| `GradRelErrMax` | `5.929376e-10` |
| `GradCosMin` | `1.0000001192` |
| `OutputAbsDiffMax` | `0.000000` |
| `DxAbsDiffMax` | `2.980232e-08` |
| `ParamGradAbsDiffMax` | `0.000000` |
| `GradPass` | 1 |
| `uses_loss_backward` | 0 |

512-sample overfit：

| metric | value |
|---|---:|
| `TrainAcc512` | `1.000000` |
| `TrainLossFinal` | `8.756091e-05` |
| `OverfitPass` | 1 |
| params | `65664` |

判断：

1. P3 排除了 “P1 low-rank residual manual backward 不对” 这个解释。
2. `S3-B2-P1-F0` 能真实 overfit 512 MNIST samples，说明它不是只在 conditioning/fit 上好看。
3. 但 P3 不是 official success；必须过 P4 kernel-native gate 才能进入 P5。

## 7. P4 kernel-native feasibility gate

同参数 MLP-match：

```text
matched_mlp_id = ManualMLP-hidden83
params_kan = 65664
params_mlp_match = 65902
params_ratio_vs_mlp_match = 0.996389
```

P4 result：

| metric | value | gate |
|---|---:|---:|
| forward ratio vs MLP-match | `14.357235` | fail |
| backward ratio vs MLP-match | `3.388516` | fail |
| step ratio vs MLP-match | `4.834536` | fail |
| memory ratio vs MLP-match | `1.821198` | fail |
| forward FLOPs ratio | `1.612576` | fail |
| backward FLOPs ratio | `3.225152` | fail |
| materializes dense edge tensor | `0` | pass |
| kernel native pass | `0` | fail |

Raw timing:

| metric | KAN | MLP-match |
|---|---:|---:|
| forward ms | `0.446991` | `0.031133` |
| backward ms | `0.450359` | `0.132908` |
| step ms | `1.233888` | `0.255224` |
| peak memory MB | `40.607910` | `22.297363` |

判断：

1. P4 是本轮 terminal blocker。虽然 `S3-B2-P1-F0` 不 materialize dense edge tensor，但当前 PyTorch/einsum materialization-free path 仍远慢于同参数 MLP-match。
2. forward ratio `14.36x` 明显不只是 AdamW/update 问题，而是 basis eval + residual projection path 需要 fused/kernel-native 实现。
3. backward FLOPs ratio `3.225x` 也超过 `1.50`，说明即使 step time 通过，compute accounting 也会失败。
4. 因此按计划写入 `KernelizationRequired`，P5/P6/P7 不打开。

## 8. Downstream stages

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | status | reason |
|---|---|---|
| `adamw_trainability_task.csv` | `not_run` | `P4_has_no_kernel_native_survivor_so_P5_P6_P7_not_opened` |
| `adamw_trainability_trace.csv` | `not_run` | `P4_has_no_kernel_native_survivor_so_P5_P6_P7_not_opened` |
| `functional_causality_v92.csv` | `not_run` | `P4_has_no_kernel_native_survivor_so_P5_P6_P7_not_opened` |
| `functional_event_trace.csv` | `not_run` | `P4_has_no_kernel_native_survivor_so_P5_P6_P7_not_opened` |
| `kanbefair_external_validation.csv` | `not_run` | `P4_has_no_kernel_native_survivor_so_P5_P6_P7_not_opened` |

判断：没有把未打开的 downstream waves 写成成功，也没有继承 v8.7/v9.0/v9.1 外部结果。

## 9. Failure table and boundary

`failure_table.csv`：

| candidate | failure code | forward ratio | backward ratio | step ratio | memory ratio | FLOPs ratio |
|---|---|---:|---:|---:|---:|---:|
| `S3-B2-P1-F0` | `F7b_kernelization_required` | `14.357235` | `3.388516` | `4.834536` | `1.821198` | `1.612576 / 3.225152` |

`boundary_audit.csv`：

```text
candidate_id = S3-B2-P1-F0
boundary_labels = kernel_native_gate_not_passed
route = R6-ComputeFail-KernelizationRequired
```

## 10. No-fake audit

```text
rows_checked = 590
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

说明：

1. P0-P4 是本轮真实测量 / 真实计算。
2. P5-P7 是 `not_run` artifact，不是 fake/proxy。
3. `kernel_count_total` / phase subcomponent timing 未测，CSV 中写为 `not_measured`，未伪造数值。

## 11. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | `c1fcadd5895f1538526a5b90d23081e433ffd4379908d54ac0a131684a21ec95` |
| v9.2 plan | `76d06f47999a3ebf12c75f3866650cf6b1c8418b228688d4332e0754ecc8277f` |
| route | `44875f1e8e507a5bface94b93123f279c6e549d1f6298f789da0c0b6faa48686` |
| `basis_source_conditioning.csv` | `70e4760d523ef95289a19e9dd803f9700c8dcbd1bcd468f71dba1153e529dbea` |
| `hybrid_residual_overfit.csv` | `d83cd4bac77c4272d1f07510a67f39cee3361ef2e1e902b028360d355f3ed272` |
| `kernel_native_feasibility_vs_mlp.csv` | `a6f1ea7167364fc9eaee06afe71c6df6375a50d1c409981cfc7119debebc9851` |
| provenance audit | `7f6e922bc16844de3e26be1215b42464648688d74d4edf0cb29dbfcf906574a7` |

## 12. 最终分析结论

v9.2 first-wave 得到的真实结论不是 FullEdge external success，而是：

```text
Basis-source blocker partially solved:
  S3 EdgeAffineNorm + B2 Chebyshev3Residual passed P1/P2/P3.

Kernel-native blocker now becomes primary:
  current P1 low-rank residual PyTorch path failed P4 by a large margin.
```

机制判断：

1. `EdgeAffineNorm` 是有效方向：它把 Chebyshev3 residual 的 dominant component 压到 `0.662755` 以下，并且三任务三 seed 都过。
2. `Chebyshev3Residual` 在 analytic fit 和 512 overfit 上成立，说明 basis 本身不是空的。
3. 当前 low-rank residual 公式虽然没有 dense edge tensor，但实现仍由多个 PyTorch/einsum 小算子组成，forward/backward launch 与中间 cache 成本远高于 MLP。
4. v9.2 下一步不应直接 full task，也不应打开 functional update；应该先做 kernel-native repair：fused basis eval、rank residual accumulation、output projection、manual backward 与 AdamW update fusion。

最终一句话：

> v9.2 已把 v9.1 的 raw-input conditioning blocker 推进到一个明确 survivor：`S3-B2-P1-F0`。但它在 P4 同参数 MLP-match kernel-native gate 上失败，route 按计划停在 `R6-ComputeFail-KernelizationRequired`。当前不能声明 FullEdge trainability、functional advantage 或 external fair success；下一步必须先做 fused/kernel-native forward-backward 后重跑 P3/P4。

## 13. 追加：P4 failure repair implementation

根据第 12 节的 P4 blocker，本轮没有打开 P5/P6/P7，也没有修改 CE / teacher / sampler / class weight / CPU offload 合同；只修 P3/P4 的 materialization-free 参数化与测时路径。

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | 将 `P3-SharedBasisLowRank` 补为 implemented materialization-free candidate |
| `experiments/run_v92_basis_source_kernel_gate.py` | 将 `P4-ActiveKSharedResidual` 补为 implemented active-k materialization-free candidate |
| `experiments/run_v92_basis_source_kernel_gate.py` | P4 Chebyshev path 新增 compiled direct / active direct forward-backward correctness check |
| `experiments/run_v92_basis_source_kernel_gate.py` | P4 step timing 使用 foreach AdamW-equivalent update，KAN 与 MLP-match 同时使用 |
| `experiments/run_v92_basis_source_kernel_gate.py` | CUDA run 设置 `torch.set_float32_matmul_precision("high")` 并写入 manifest |

代码检查：

```text
python -m py_compile experiments/run_v92_basis_source_kernel_gate.py
```

已通过。

失败修复过程没有删除 first-wave 失败结论。中间真实 probes 显示：

| candidate | hidden/rank | P4 forward | P4 backward | P4 step | 判定 |
|---|---:|---:|---:|---:|---|
| `S3-B2-P1-F0` packed/compiled/foreach | `1024/1` | `1.260461` | `1.554352` | `1.176748` | near fail |
| `S3-B2-P3-F0` direct shared basis | `1024/1` | `1.750986` | `1.788697` | `1.272373` | fail |
| `S3-B2-P3-F0` direct shared basis | `4096/1` | `1.403498` | `1.597797` | `1.201594` | fail |
| `S3-B2-P4-F0` active-k shared residual | `1024/1` | `1.304007` | `1.683286` | `1.179543` | fail |
| `S3-B2-P4-F0` active-k shared residual | `4096/1` | `0.953444` | `1.154653` | `1.060843` | pass |

判断：

1. P1 factorized residual 的主要问题是 backward / update 拆解和 fixed basis overhead。
2. P3 direct shared-basis 修掉了部分 step 开销，但 full three-channel basis 仍然 forward/backward 超线。
3. P4 active-k shared residual 才真正闭合 P4：减少 active basis channel 后，hidden4096 下 fixed overhead 被摊薄，forward/backward/step/memory/FLOPs 全部进入 MLP-match envelope。
4. 这不是改 gate，也不是把失败 run 挑掉；first-wave P1 failure 仍保留为真实边界，最新 route 改变来自新增 official P4 candidate 的真实 pass artifact。

### 13.1 P4 是怎么解决的

first-wave 的失败 candidate 是：

```text
S3-B2-P1-F0 = EdgeAffineNorm source + Chebyshev3Residual basis + IdentityPlusLowRankResidual
```

它虽然没有 materialize dense edge tensor，但实现上仍有两个成本：

1. Full Chebyshev residual 使用 3 个 basis channels，forward 里要构造 `basis_flat = B x (in*K)`，再做 residual projection。
2. P1 因子化参数是 `U/A/V`，backward 里先得到 `dM`，再拆成 `dU` 和 `dA`；这让 backward/update 比同参数 MLP 多出固定开销。

修复分三步做，每一步都真实重跑 P3/P4：

| step | 做了什么 | 结果 |
|---|---|---|
| packed P1 | 把 residual path 从 `B x in x rank` einsum 中间量改成 packed `basis_flat @ M` | memory / FLOPs 改善，但 P4 仍 fail |
| compiled + foreach | 给 Chebyshev path 加 `torch.compile` forward/backward，并让 KAN 与 MLP-match 都用 foreach AdamW-equivalent update | step gate 基本修掉，但 forward/backward 仍 fail |
| P4 active-k | 新增 `ActiveKSharedResidual`：只保留 Chebyshev 的一个 active residual channel，用直接 `M/V` shared-basis 参数化，减少 basis channel 和 `U/A` 梯度拆解 | hidden4096/rank1 下 P4 全 gate pass |

最终通过的 candidate 是：

```text
S3-B2-P4-F0
source = EdgeAffineNorm
basis = Chebyshev3Residual
parameterization = ActiveKSharedResidual
active basis channel = Chebyshev T3 channel
hidden = 4096
rank = 1
```

它的关键变化不是换 loss、不是 teacher、不是调 sampler，也不是放宽 P4 gate，而是把 residual 参数化从：

```text
full 3-channel Chebyshev basis + U/A/V factorized residual
```

改为：

```text
single active Chebyshev residual channel + direct shared M/V residual
```

这样减少了 forward 的 basis channel 数，也减少了 backward 中 `dM -> dU/dA` 的拆解。再用 compiled forward/backward 和 foreach update，把固定小算子 overhead 压到同参数 MLP-match 的 envelope 内。

本次仍保持以下合同不变：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake/proxy = 0
P4 thresholds unchanged
```

并且 compiled path 做了 correctness check：

```text
compiled output max abs diff = 0.0000019073
compiled grad max abs diff = 0.0000000171
```

因此这次 P4 pass 是新的真实实现通过 gate，不是把旧失败结果包装成成功。

## 14. 追加：P4 repaired run

正式 rerun：

```bash
python experiments/run_v92_basis_source_kernel_gate.py \
  --out-dir results/real_rerun_20260506/v92_basis_source_kernel_gate_h4096r1_activeP4_compiled_foreach_20260509T110000Z \
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
  --p3-hidden-dim 4096 \
  --p3-rank 1 \
  --p3-parameterizations P4 \
  --p3-steps 600 \
  --p3-train-size 1024 \
  --p3-overfit-size 512 \
  --p4-batch-size 128 \
  --p4-warmup 20 \
  --p4-reps 120
```

Route：

```json
{
  "route": "R2-HybridResidualFullEdgeNearPass",
  "best_candidate": "S3-B2-P4-F0",
  "best_source": "S3",
  "best_basis": "B2",
  "best_parameterization": "P4",
  "p1_source_basis_pass_count": 1,
  "p2_source_basis_survivor_count": 1,
  "p3_overfit_pass_count": 1,
  "p4_kernel_native_pass_count": 1,
  "primary_blocker": "P4_passed_but_P5_P7_not_executed_in_first_wave_runner"
}
```

判断：P4 已经解决，但当前 runner 仍只执行到 P4；P5/P6/P7 未打开，因此不能声明 trainability / functional / external fair success。

## 15. 追加：P3/P4 repaired result

P3 gradcheck：

| metric | value |
|---|---:|
| candidate | `S3-B2-P4-F0` |
| parameterization | `P4-ActiveKSharedResidual` |
| GradRelErrMax | `9.8794827874e-10` |
| GradCosMin | `1.000000` |
| DxAbsDiffMax | `4.7683715820e-07` |
| ParamGradAbsDiffMax | `0.000000` |
| GradPass | `1` |
| uses_loss_backward | `0` |

P3 overfit：

| metric | value |
|---|---:|
| hidden / rank | `4096 / 1` |
| params | `3261210` |
| TrainAcc512 | `1.000000` |
| TrainLossFinal | `0.0000463071` |
| OverfitPass | `1` |

P4 kernel-native gate：

| metric | value | gate |
|---|---:|---|
| matched MLP | `CompiledManualMLP-hidden4107` | - |
| params ratio vs MLP-match | `1.000077` | pass |
| materializes dense edge tensor | `0` | pass |
| forward ratio vs MLP-match | `0.953444` | pass |
| backward ratio vs MLP-match | `1.154653` | pass |
| step ratio vs MLP-match | `1.060843` | pass |
| memory ratio vs MLP-match | `1.000016` | pass |
| forward FLOPs ratio | `1.003070` | pass |
| backward FLOPs ratio | `1.001574` | pass |
| compiled output max abs diff | `0.0000019073` | pass |
| compiled grad max abs diff | `0.0000000171` | pass |
| kernel native pass | `1` | pass |

Raw timing：

| metric | KAN | MLP-match |
|---|---:|---:|
| forward ms | `0.081586` | `0.085569` |
| backward ms | `0.095799` | `0.082967` |
| step ms | `0.714801` | `0.673804` |
| peak memory MB | `186.446777` | `186.443848` |

判断：

1. P4 active-k candidate 的 forward 已经低于同参数 MLP-match，backward/step 也在 gate 内。
2. FLOPs 与 memory 都接近 1，说明这次不是靠超额 compute / memory 换 timing。
3. `kernelization_required_candidates.csv` 变成 `status=not_required`，reason 为 `all_P4_candidates_passed_kernel_native_gate`。
4. 但这只是 P4 资格门打开；P5-P7 仍未真实执行，不能写 external fair 成功。

## 16. No-fake audit / hash / 更新结论

No-fake audit：

```text
rows_checked = 590
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | `d54f51412194e2ce4ace1c78ea748ced6442ba37bad183e507514c5a33d43144` |
| v9.2 plan | `76d06f47999a3ebf12c75f3866650cf6b1c8418b228688d4332e0754ecc8277f` |
| route | `1712c999d4198c521f2f628a2d9923cf31ad9a340a4238e0b0bdfd030267f9b4` |
| `hybrid_residual_gradcheck.csv` | `4fd89864aa6ea883681973ee35031b7cca7a1234c8c5202b501a49756ce29e50` |
| `hybrid_residual_overfit.csv` | `231ca70d82caa89b8034e72754a31bb3b26c7147baef137af652e96195cb08a9` |
| `kernel_native_feasibility_vs_mlp.csv` | `a49b3786422a68c46042821283d03bdb32de9db284c4b6eaa16d31852519f688` |
| provenance audit | `7f6e922bc16844de3e26be1215b42464648688d74d4edf0cb29dbfcf906574a7` |

更新结论：

```text
P1 source-basis conditioning = pass, 1 survivor
P2 one-layer fit = pass for S3-B2
P3 active-k hybrid residual gradcheck = pass
P3 active-k 512 overfit = pass
P4 kernel-native feasibility = pass
P5 AdamW full-task trainability = not_run
P6 functional advantage = not_run
P7 external fair = not_run
route = R2-HybridResidualFullEdgeNearPass
success_v92_basis_source = true
success_v92_full_edge_trainability = false
success_v92_functional_advantage = false
success_v92_external_fair = false
```

机制结论：

1. v9.2 的 P4 失败已经被解决：`S3-B2-P4-F0` 是第一个真实通过 conditioning / fit / gradcheck / overfit / kernel-native gate 的 clean FullEdge hybrid residual candidate。
2. P4 修复的关键不是放宽阈值，而是参数化变化：从 P1 full Chebyshev low-rank residual 转为 P4 active-k shared residual，减少 basis channel 与 residual backward overhead。
3. 当前仍不能声明 v9.2 完成，因为 full-task AdamW、functional update 和 external fair validation 尚未执行。
4. 下一步应继续实现并运行 P5/P6/P7；如果这些失败，不能把 P4 pass 外推成最终成功。

最终一句话：

> P4 已解决：`S3-B2-P4-F0` 在 hidden4096/rank1 下真实通过 kernel-native feasibility gate，route 从 `R6-ComputeFail-KernelizationRequired` 推进到 `R2-HybridResidualFullEdgeNearPass`。但 v9.2 还没完成，P5/P6/P7 仍是未运行边界。

## 17. 追加：继续打开 P5 AdamW-only trainability

根据第 16 节结论，本轮继续执行 v9.2 后续部分。计划要求 P5 先证明 AdamW-only 能训练；P5 不过时 P6/P7 不得打开。本节仍保持：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake/proxy = 0
```

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | 新增 `--run-p5-trainability`，只对 P4 survivor 打开 P5 |
| `experiments/run_v92_basis_source_kernel_gate.py` | 新增 AdamW-only KAN vs same-param MLP-match trainability rows |
| `experiments/run_v92_basis_source_kernel_gate.py` | 新增 P5 train trace、failure table、boundary route 更新 |
| `experiments/run_v92_basis_source_kernel_gate.py` | 新增 `--p4-active-cheb-index`，用于真实比较 active Chebyshev T2/T3 channel |

代码检查：

```text
python -m py_compile experiments/run_v92_basis_source_kernel_gate.py
```

已通过。

P5 修复尝试：

| candidate | active channel | hidden/rank | epochs | P4 | P5 min pass rows | 结论 |
|---|---:|---:|---:|---|---:|---|
| `S3-B2-P4-F0` | T3 / index2 | `4096/1` | 5 | pass | `0/9` | fail |
| `S3-B2-P4-F0` | T3 / index2 | `4096/4` | 5 | pass | `0/9` | fail |
| `S3-B2-P4-F0` | T2 / index1 | `4096/4` | 5 | pass | `0/9` | fail |
| `S3-B2-P4-F0` | T2 / index1 | `4096/4` | 20 | pass | `0/9` | fail |

判断：

1. Rank4 与 T2 active channel 都没有破坏 P4；说明 P4 的 compute repair 不是只在单一 rank1/T3 点上成立。
2. 但这些 P4 survivor 都没有通过 P5 AdamW-only trainability。
3. 20 epoch 下 train acc 上升但 test loss / ECE 恶化，说明不是简单训练步数不足；当前 active-k residual 表达/优化仍不能追上同参数 MLP-match。

## 18. P5 final run

最终 P5 run：

```bash
python experiments/run_v92_basis_source_kernel_gate.py \
  --out-dir results/real_rerun_20260506/v92_p5_adamw_trainability_activeP4_T2_h4096r4_3task3seed_e20_smoke_20260509T124500Z \
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
  --p3-hidden-dim 4096 \
  --p3-rank 4 \
  --p3-parameterizations P4 \
  --p4-active-cheb-index 1 \
  --p3-steps 600 \
  --p3-train-size 1024 \
  --p3-overfit-size 512 \
  --p4-batch-size 128 \
  --p4-warmup 20 \
  --p4-reps 120 \
  --run-p5-trainability \
  --p5-datasets MNIST,Fashion-MNIST,KMNIST \
  --p5-seeds 0,1,2 \
  --p5-train-size 9984 \
  --p5-test-size 2000 \
  --p5-epochs 20 \
  --p5-lr 0.002
```

Route：

```json
{
  "route": "R4-AdamWTrainabilityFail",
  "best_candidate": "S3-B2-P4-F0",
  "best_source": "S3",
  "best_basis": "B2",
  "best_parameterization": "P4",
  "p1_source_basis_pass_count": 1,
  "p2_source_basis_survivor_count": 1,
  "p3_overfit_pass_count": 1,
  "p4_kernel_native_pass_count": 1,
  "p5_trainability_row_count": 9,
  "p5_min_trainability_pass_count": 0,
  "primary_blocker": "P5_adamw_only_trainability_failed"
}
```

P4 recap for this final run：

| metric | value | gate |
|---|---:|---|
| active Chebyshev channel | `T2 / index1` | - |
| hidden / rank | `4096 / 4` | - |
| forward ratio | `1.082807` | pass |
| backward ratio | `1.299241` | pass |
| step ratio | `1.057769` | pass |
| memory ratio | `1.000010` | pass |
| forward FLOPs ratio | `1.003776` | pass |
| backward FLOPs ratio | `1.001920` | pass |
| kernel native pass | `1` | pass |

说明：P5 failure 不是 P4 compute failure；compute gate 在最终 P5 run 里仍然通过。

## 19. P5 AdamW-only trainability result

P5 protocol：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 9984
test_size = 2000
epochs = 20
batch_size = 128
functional_update = off
baseline = same-param MLP-match AdamW
```

P5 rows：

| dataset | seed | KAN acc | MLP-match acc | delta | min pass |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `0.878500` | `0.955000` | `-0.076500` | 0 |
| MNIST | 1 | `0.859000` | `0.947500` | `-0.088500` | 0 |
| MNIST | 2 | `0.884500` | `0.955500` | `-0.071000` | 0 |
| Fashion-MNIST | 0 | `0.770500` | `0.872000` | `-0.101500` | 0 |
| Fashion-MNIST | 1 | `0.781500` | `0.860500` | `-0.079000` | 0 |
| Fashion-MNIST | 2 | `0.800000` | `0.867000` | `-0.067000` | 0 |
| KMNIST | 0 | `0.694500` | `0.849000` | `-0.154500` | 0 |
| KMNIST | 1 | `0.693000` | `0.826000` | `-0.133000` | 0 |
| KMNIST | 2 | `0.713000` | `0.845500` | `-0.132500` | 0 |

Final train trace snapshot：

| dataset | seed | KAN train acc head2048 | KAN train loss | MLP train acc head2048 | MLP train loss |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `0.934570` | `2.690409` | `0.997559` | `0.010813` |
| MNIST | 1 | `0.892578` | `2.753054` | `0.999023` | `0.012628` |
| MNIST | 2 | `0.928223` | `2.393146` | `0.998535` | `0.017649` |
| Fashion-MNIST | 0 | `0.806152` | `1.808071` | `0.957031` | `0.130335` |
| Fashion-MNIST | 1 | `0.823242` | `1.625704` | `0.961914` | `0.123919` |
| Fashion-MNIST | 2 | `0.865723` | `1.665445` | `0.969727` | `0.133456` |
| KMNIST | 0 | `0.907227` | `2.972878` | `1.000000` | `0.000159` |
| KMNIST | 1 | `0.882812` | `2.461814` | `0.993164` | `0.043904` |
| KMNIST | 2 | `0.919434` | `2.492811` | `1.000000` | `0.000052` |

判断：

1. P5 的 minimum trainability 是 `Acc_KAN >= Acc_MLP - 0.01`；最终 9/9 rows 都未通过。
2. 20 epoch 下 KAN 训练准确率有提升，但 loss 仍远高于 MLP-match；这说明当前 active-k residual 不是单纯训练轮数不足。
3. KMNIST 训练 head acc 可到 `0.88-0.92`，但 test acc 只有 `0.69-0.71`，泛化和校准都明显弱于 MLP-match。
4. 因此按计划不能打开 P6 functional update；functional update 不能用来掩盖 AdamW-only trainability failure。

P6/P7 artifact：

| artifact | status | reason |
|---|---|---|
| `functional_causality_v92.csv` | `not_run` | `P5_adamw_trainability_failed_so_P6_P7_not_opened` |
| `functional_event_trace.csv` | `not_run` | `P5_adamw_trainability_failed_so_P6_P7_not_opened` |
| `kanbefair_external_validation.csv` | `not_run` | `P5_adamw_trainability_failed_so_P6_P7_not_opened` |

## 20. No-fake audit / hash / final conclusion

No-fake audit：

```text
rows_checked = 965
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_v92_basis_source_kernel_gate.py` | `376cf0870d66c33df2d1a9e82e7a914063c8898a3483a9c7cc192f770e5c46d2` |
| v9.2 plan | `76d06f47999a3ebf12c75f3866650cf6b1c8418b228688d4332e0754ecc8277f` |
| route | `53b3d46b20194779ca58912b170b6c62e408b1b837d2902c1bd09c035429dd7e` |
| `adamw_trainability_task.csv` | `b9830af8ac8ce7ec244e71238f1d170a13386c2ba707544b8d932db285a7f878` |
| `adamw_trainability_trace.csv` | `fae60a5c15b4adad416ee43ad48967756035964c3ab929d0a22a845414240ae2` |
| `kernel_native_feasibility_vs_mlp.csv` | `ad79c5b5f599bc6f2bb60d248233c8eb51e702675382ec730a825a54e127458c` |
| provenance audit | `67edaf69b5fa09aecb93c2687501ceae898439bf7deb9ae21b8a770b245ea3a4` |

最终结论：

```text
P1 source-basis conditioning = pass
P2 one-layer fit = pass
P3 gradcheck / overfit = pass
P4 kernel-native gate = pass
P5 AdamW-only trainability = fail
P6 functional update = not_run
P7 external fair = not_run
route = R4-AdamWTrainabilityFail
success_v92_basis_source = true
success_v92_full_edge_trainability = false
success_v92_functional_advantage = false
success_v92_external_fair = false
```

机制结论：

1. v9.2 已经推进过 v9.1 的 conditioning blocker，也解决了 P4 kernel-native blocker。
2. 新 terminal blocker 是 P5 AdamW-only trainability：active-k shared residual 在 real task smoke 上不能追平同参数 MLP-match。
3. T2 active channel 比 T3 有改善趋势，但 20 epoch 三任务三 seed 仍是 0/9 pass。
4. 下一步不能直接做 P6/P7，也不能用 functional update 救当前 candidate；应先修 active-k 容量或 optimizer/trainability protocol，例如多 active channels 的真正 fused path、nonlinear base activation、残差 rank/capacity 与正则策略。

最终一句话：

> v9.2 后续部分已经继续执行到 P5，并诚实停在 `R4-AdamWTrainabilityFail`：P4 过了，但 AdamW-only full-task smoke 没过，所以 P6 functional 和 P7 external fair 不能打开。
