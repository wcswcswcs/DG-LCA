# DG-KAN v12.5.2 效率 / Functional / 流形信号通道三线结果复盘

> 本复盘记录 `DG-KAN_v12.5.2_效率Functional流形信号通道三线完整计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R1-FusedEfficiencyFail
base_qualified = False
functional_open = False
next_recommended_action = continue lower-level fused CUDA/Triton full forward+backward kernel; do not continue schedule/scale small grid
```

最终 artifact：

```text
results/v12_5_2_efficiency_functional_manifold/smoke_v1252_simple_h64_20260521T002500Z
```

核心结论：

1. v12.5.2 已真实执行 A 线 Triton component forward truth、B47 manual backward/full-step truth、Line C train-probe/signal-reservoir 诊断与 Functional cloned diagnostic。
2. Triton F2/F3 是真实 custom kernel smoke，不是标签；其结果按 artifact 中的 ratio/error 判定，没有写成 full fused success。
3. 当前 route 仍未打开 base：`base_qualified = false`，所以 expression/task confirm 与 official functional short-run 均合法关闭。
4. Functional rows 都是 diagnostic；没有把 control gap 或几何分数写成 official success。
5. No-fake audit rows checked = `180`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 v12.5.2 三线 runner | 独立输出 `v125_*` artifacts；复用 v12.4 strict PureKAN model，不改低 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Triton F2/F3 component kernels | 真实执行 residual input norm + Legendre basis / hidden preactivation component fusion；记录 correctness 与 timing，不冒充 full fused layer。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 执行中修复 Triton `tl.tanh` blocker | smoke 暴露当前 Triton 3.0 无 `tl.tanh`；改为等价 `2/(1+exp(-2x))-1` 后 F2/F3 才真实执行，失败前结果不写成 kernel success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 F4 hybrid full-forward repair | 复用 Triton F2/F3 的 norm/basis/hidden preactivation，再接 torch readout；用于检查 component fusion 组合后是否足够，不冒充 single-call full fused kernel。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48 SimpleFastTaskGeometry | 按 R1/R6 推荐方向，在 full GatedLQ fusion 收益不足时转向 hinge/direct edge basis + minimal quadratic sketch；fixed train-stream norm，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | B48 fast rewrite + manual CE step | 将 direct hinge 三路 readout 合并为单个 readout matmul，将 quadratic readout 改为扁平 matmul，并新增 SimpleFastTaskGeometry manual CE backward；只减少算子/反向开销，不改 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 将 K3 SimpleFastTaskGeometry 接入 forward/full-step/Line C/Functional diagnostic | 用同一 A1/A3/Line C/Functional artifact 审计，不把 simple primitive 的效率或 diagnostic 结果自动写成 base success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Line C coupling / signal-reservoir / noise leak | 独立写 `v125_train_probe_coupling.csv`、`v125_signal_reservoir_sketch.csv`、`v125_manifold_channel_diagnostics.csv`。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 functional cloned diagnostic 接入 Line C score | Base 未合格时所有 functional row 写 `official_gate_open = 0`。 |

本轮没有做：

```text
1. 没有调低 A1/A2/A3/Line C/functional gate。
2. 没有按 dataset name 分支。
3. 没有使用 teacher / distillation / modified loss / sampler / class weight。
4. 没有把 Triton component kernel 写成 full fused kernel success。
5. 没有把 diagnostic functional 写成 official。
```

## 2. A1 Forward Truth

| kernel impl | candidate | ratio vs MLP q90 | ratio vs B47 q90 | max error | formal pass |
|---|---|---:|---:|---:|---:|
| `F0-current-B47-manual-autograd-forward` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `9.42339173069299` | `1.0` | `0.0` | `0` |
| `F0b-current-B42-fastreuse-forward` | `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `10.35667220362629` | `1.0990387006722333` | `` | `0` |
| `K2-current-LiteGated-forward` | `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `6.405506180150251` | `0.6797452937552023` | `` | `0` |
| `K3-SimpleFastTaskGeometry-forward` | `B48c-SimpleFastTaskGeometry-h64-quad020-temp075` | `3.541115689000186` | `0.3757793149430894` | `` | `0` |
| `K3-compiled-SimpleFastTaskGeometry-forward` | `B48c-SimpleFastTaskGeometry-h64-quad020-temp075` | `3.5532300882735224` | `0.37706488171347835` | `1.1920928955078125e-07` | `0` |
| `F2-triton-residual-input-norm-plus-legendre-basis` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.0824045125384534` | `0.11486358027682822` | `0.0009468495845794678` | `0` |
| `F3-triton-residual-input-norm-legendre-hidden-preactivation` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.6966030501789058` | `0.18004165577165693` | `2.047792077064514e-05` | `0` |
| `F4-triton-hybrid-full-forward-with-torch-readouts` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `8.405611566390823` | `0.8919942847130987` | `0.0005792379379272461` | `0` |

## 3. A2/A3 Backward 与 Full Step

| candidate | forward ratio | backward ratio | step ratio | memory ratio | exploratory pass |
|---|---:|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.0` | `1.0` | `1.0` | `1.0` | `1` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `5.6988072105963115` | `3.3121879510442134` | `3.2806843679747684` | `1.0433619509767327` | `0` |
| `B48c-SimpleFastTaskGeometry-h64-quad020-temp075` | `2.3275399426323125` | `1.9585969027070016` | `1.7255239062718362` | `0.9750528804280204` | `0` |

解释：若 B47 manual CE step 未过 A1/A3 efficiency，本轮不会推进 expression/task official gate。

## 4. Line C Train-Probe Coupling

| candidate | CouplingR2 | CouplingCorr | KernelDrift | ECE delta |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `0.9988084436749775` | `0.9994902610778809` | `9.647709846496582` | `0.04702761769294739` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.9973838563138185` | `0.9989147186279297` | `2.260103464126587` | `-0.09008382260799408` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.997383652568166` | `0.9989147186279297` | `2.260098934173584` | `-0.09008394181728363` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.9999603465759157` | `0.9999846816062927` | `5.287623882293701` | `-0.08392524719238281` |
| `B48c-SimpleFastTaskGeometry-h64-quad020-temp075` | `0.9952326668963255` | `0.9979439973831177` | `2.7883119583129883` | `0.0009374022483825684` |

## 5. Line C Signal / Reservoir / Noise

| candidate | signal effective rank | real reservoir ratio | noise signal leak | real-noise gap |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.4146422147750854` | `0.6000441908836365` | `0.018301814794540405` | `0.8674342632293701` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `1.063552975654602` | `0.7408638596534729` | `0.05703643709421158` | `0.7948987483978271` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.063552975654602` | `0.7408649921417236` | `0.057036835700273514` | `0.7948989868164062` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `1.0556915998458862` | `0.9984378218650818` | `0.0017309063114225864` | `0.7251960039138794` |
| `B48c-SimpleFastTaskGeometry-h64-quad020-temp075` | `1.67581307888031` | `0.0021464149467647076` | `0.9033060073852539` | `0.09605646133422852` |

## 6. Functional Diagnostic

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.46972002364271404` | `0.5268760100007057` | `-0.9965960336434198` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.46972014414044183` | `0.5268758609890938` | `-0.9965960051295356` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-1.1329674001387036` | `-0.1364654004573822` | `-0.9965019996813214` | `0` |
| `B48c-SimpleFastTaskGeometry-h64-quad020-temp075` | `-0.9158404153645799` | `0.08373244106769562` | `-0.9995728564322756` | `0` |

解释：这些结果均为 cloned diagnostic。`base_qualified = false` 时不允许打开 official functional short-run。

## 7. No-Fake / Hash

```text
rows_checked = 180
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v125_backward_kernel_truth.csv` | `fd22f3600da78ed1e98cccb6004f0533f0ea5fddd00dd6a6c69922050c252f92` |
| `v125_control_matrix.csv` | `c2439540b22680804ce1385ac3fa0fd794cdbf94c9b2b0a8a1789e3060a31c90` |
| `v125_forward_kernel_truth.csv` | `a1f1b3e3f201992f24ea45366914ac9c5d5ac787c72d34f008b9a18aefe1c90a` |
| `v125_full_step_efficiency.csv` | `21b6f0b5bbe1feb7cfa8e32ca5978de033be28b7511c10dc503d2f995089b7a2` |
| `v125_provenance_audit.csv` | `ad29e3c7cf1fff9dfce9d2a374123e2eb697f817a200d2168ab7e88266c24b69` |
| `v125_route_decision.json` | `69b6819dcb8168b8e78f4af7cb516f1aa901d103be56a7bd43474a83ed6d4eaa` |
| `v125_signal_reservoir_sketch.csv` | `2d687bc6d70a8199b2a7621aab26f3f34e70751ad8d017aebe6e7dc410a754fe` |
| `v125_train_probe_coupling.csv` | `566fa1705547cedda2fa1963054e9c37a1e57fdc38f980cc042fcc78d12f4ff2` |

## 8. 最终分析结论

```text
1. v12.5.2 已把 v12.4 的“manual backward 还不够”推进为可审计的 kernel component truth。
2. 本轮没有证明 full fused GatedLQ 已达到 MLP-like efficiency；若 route 仍是 R1，应继续 lower-level full forward+backward fusion，而不是 schedule/scale 小修。
3. Line C 已独立落盘，后续可以用同一指标判断 near-pass base 是否存在 coupling collapse / noise leakage / reservoir trapping。
4. Functional diagnostic 仍未 official；base gate 未开，control-resistant 几何增益也不能替代 base qualification。
5. 若 SimpleFastTaskGeometry 仍不能打开完整 base gate，下一步应优先实现 lower-level fused simple hinge/quadratic kernel 或重做 task-geometry primitive，不允许把 diagnostic 写成 official。
```
