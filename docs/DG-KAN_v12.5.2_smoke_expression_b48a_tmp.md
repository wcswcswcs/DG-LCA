# DG-KAN v12.5.2 效率 / Functional / 流形信号通道三线结果复盘

> 本复盘记录 `DG-KAN_v12.5.2_效率Functional流形信号通道三线完整计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = R3-ExpressionPassTaskFail
base_qualified = False
functional_open = False
next_recommended_action = run/repair A5 task triage after expression pass; no functional official route before base qualification
```

最终 artifact：

```text
results/v12_5_2_efficiency_functional_manifold/smoke_v1252_expression_b48a_20260521T005000Z
```

核心结论：

1. v12.5.2 已真实执行 A 线 Triton component forward truth、B47 manual backward/full-step truth、Line C train-probe/signal-reservoir 诊断与 Functional cloned diagnostic。
2. Triton F2/F3 是真实 custom kernel smoke，不是标签；其结果按 artifact 中的 ratio/error 判定，没有写成 full fused success。
3. 当前 route 按 A1/A4/A5 顺序判定；A4 expression 已在 A1 full-step official pass 后真实运行，A5 task 只有 A4 通过才允许打开。
4. Functional rows 都是 diagnostic；没有把 control gap 或几何分数写成 official success。
5. No-fake audit rows checked = `310`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 v12.5.2 三线 runner | 独立输出 `v125_*` artifacts；复用 v12.4 strict PureKAN model，不改低 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Triton F2/F3 component kernels | 真实执行 residual input norm + Legendre basis / hidden preactivation component fusion；记录 correctness 与 timing，不冒充 full fused layer。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 执行中修复 Triton `tl.tanh` blocker | smoke 暴露当前 Triton 3.0 无 `tl.tanh`；改为等价 `2/(1+exp(-2x))-1` 后 F2/F3 才真实执行，失败前结果不写成 kernel success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 F4 hybrid full-forward repair | 复用 Triton F2/F3 的 norm/basis/hidden preactivation，再接 torch readout；用于检查 component fusion 组合后是否足够，不冒充 single-call full fused kernel。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48 SimpleFastTaskGeometry | 按 R1/R6 推荐方向，在 full GatedLQ fusion 收益不足时转向 hinge/direct edge basis + minimal quadratic sketch；fixed train-stream norm，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | B48 fast rewrite + manual CE step | 将 direct hinge 三路 readout 合并为单个 readout matmul，将 quadratic readout 改为扁平 matmul，并新增 SimpleFastTaskGeometry manual CE backward；只减少算子/反向开销，不改 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 将 K3 SimpleFastTaskGeometry 接入 forward/full-step/A4 expression/Line C/Functional diagnostic | 用同一 A1/A3/A4/Line C/Functional artifact 审计，不把 simple primitive 的效率或 diagnostic 结果自动写成 base success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 修复 full-step peak memory warmup accounting | formal B48 compiled step 暴露 compile/warmup allocation 污染 memory ratio；改为 warmup 结束后 reset peak memory，再计 steady measurement，不改变 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 A4 expression qualification | 复用 v12.4 expression battery 的真实训练/冻结读出逻辑，按 v12.5.2 stricter gate 计算 B1 key deltas、frozen R2 与 dead basis；未过 A4 不打开 A5。 |
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
| `F0-current-B47-manual-autograd-forward` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `12.306635985987437` | `1.0` | `0.0` | `0` |
| `F0b-current-B42-fastreuse-forward` | `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `13.449960198908773` | `1.0929030658112537` | `` | `0` |
| `K2-current-LiteGated-forward` | `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `8.291576026529764` | `0.6737483773771081` | `` | `0` |
| `K3-SimpleFastTaskGeometry-forward` | `B48a-SimpleFastTaskGeometry-h128-temp075` | `4.630542745082873` | `0.3762638913148398` | `` | `0` |
| `K3-compiled-SimpleFastTaskGeometry-forward` | `B48a-SimpleFastTaskGeometry-h128-temp075` | `2.280860348122581` | `0.1853358099418566` | `2.384185791015625e-07` | `0` |
| `F2-triton-residual-input-norm-plus-legendre-basis` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.6633955648371055` | `0.13516249011761447` | `0.0010108202695846558` | `0` |
| `F3-triton-residual-input-norm-legendre-hidden-preactivation` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `3.3790191850927993` | `0.2745688739750012` | `1.9073486328125e-05` | `0` |
| `F4-triton-hybrid-full-forward-with-torch-readouts` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `12.011937104607581` | `0.9760536606660498` | `0.0004998147487640381` | `0` |

## 3. A2/A3 Backward 与 Full Step

| candidate | forward ratio | backward ratio | step ratio | memory ratio | exploratory pass |
|---|---:|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.0` | `1.0` | `1.0` | `1.0` | `1` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `6.176329422402234` | `3.602765256609352` | `3.389020859841563` | `1.2055553649246817` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `2.2206045376709675` | `1.7664825682587113` | `1.5802970733781776` | `1.108568095254435` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `1.4221978627675025` | `0.926247058459632` | `1.083322904639457` | `0.4033043955666884` | `1` |

解释：若 B47 manual CE step 未过 A1/A3 efficiency，本轮不会推进 expression/task official gate。

## 4. A4 Expression Qualification

| candidate | trainable key delta OK | frozen key R2 OK | dead basis max | A4 pass |
|---|---:|---:|---:|---:|
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0` | `1` | `0.0` | `1` |

解释：A4 pass 规则是 E1/E2/E6/E8 的 B1 delta 全部 `>= -0.01`，或这些 key targets 的 frozen R2 全部 `>= 0.89` 且 dead basis `<= 0.30`。未满足时 A5 task 继续合法关闭。

## 5. Line C Train-Probe Coupling

| candidate | CouplingR2 | CouplingCorr | KernelDrift | ECE delta |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `0.6095117627820872` | `0.7807225584983826` | `8.486675262451172` | `0.19238337874412537` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.5280548317475608` | `0.7267348766326904` | `1.8674850463867188` | `-0.04222309589385986` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.5280555377020303` | `0.7267352342605591` | `1.8674851655960083` | `-0.04222309589385986` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.5700975801347126` | `0.7550583481788635` | `4.409830570220947` | `0.1844300925731659` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0.5666470197964535` | `0.7527995109558105` | `2.58101224899292` | `0.13492795825004578` |

## 6. Line C Signal / Reservoir / Noise

| candidate | signal effective rank | real reservoir ratio | noise signal leak | real-noise gap |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.2451450824737549` | `0.5623162388801575` | `0.2198314517736435` | `0.5934246778488159` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `1.1804330348968506` | `0.7277566194534302` | `0.02517196163535118` | `0.5053439140319824` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.1804330348968506` | `0.7277566194534302` | `0.025171438232064247` | `0.5053439140319824` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `1.1914806365966797` | `0.970586359500885` | `0.004154191818088293` | `0.4547455310821533` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `1.3830387592315674` | `0.8798787593841553` | `0.10408840328454971` | `0.46566927433013916` |

## 7. Functional Diagnostic

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.35582386718049475` | `0.6412949003279209` | `-0.9971187675084157` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.3558235876047673` | `0.6412948593497276` | `-0.9971184469544949` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-1.270303084081693` | `-0.27226561307907104` | `-0.9980374710026219` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `-0.6695289966294542` | `0.32724229991436005` | `-0.9967712965438142` | `0` |

解释：这些结果均为 cloned diagnostic。`base_qualified = false` 时不允许打开 official functional short-run。

## 8. No-Fake / Hash

```text
rows_checked = 310
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v125_backward_kernel_truth.csv` | `a0ff5bd8b8856dd009ac202d3431f4c1b2f4a6c6ad53c8646c65df75c7faece7` |
| `v125_control_matrix.csv` | `beffed12a120f70ab72c1fc5b290b083dd225fd2387520b3ca15cc78bf957c80` |
| `v125_expression_battery.csv` | `1235e9bc0a0255157475a04f0dc900a1f8a19bad5b4a5d603f7dae507cf33c7c` |
| `v125_expression_summary.csv` | `db37407d8f432e91aa2d0fe64f2a6e75906033dd9ba0ad9ce3188f22ce5375db` |
| `v125_forward_kernel_truth.csv` | `cf15edde69c8f5d317336c08d427da8901e26f1efd0e436b737f77b1dc75c6cd` |
| `v125_frozen_readout.csv` | `3298edc1e9fbaae073f7db0006834af1b5ad18f37bea20f3363c6a7cbeba3753` |
| `v125_full_step_efficiency.csv` | `1d09cc14b9dcd033430ed387698c74ae29f863e698047fb053011027bb604ecd` |
| `v125_matrix_span.csv` | `952d0423931ee41d5f8a44730efef35bc78456c7784f1358693c189348e83190` |
| `v125_provenance_audit.csv` | `f4b122524d1494c4af29b88f95094dab042bbc35a3bf785ea5a7d2a909cb6b86` |
| `v125_route_decision.json` | `ab2790b0f22ebcb20d56f3e15f9a92d5751a8d0184d57675336e52fcd733741b` |
| `v125_signal_reservoir_sketch.csv` | `83874e6eb41f51525451199edf569c11d81cb0b7921a08472b7edb8e12d35377` |
| `v125_train_probe_coupling.csv` | `29c7defefdb7ab6975035d8a81e453150c68feee4af80288e867e473c97075ec` |

## 9. 最终分析结论

```text
1. v12.5.2 已把 v12.4 的“manual backward 还不够”推进为可审计的 kernel component truth。
2. 若 B48 compiled simple primitive 已达到 A1/A3 efficiency，必须继续看 A4 expression；不能把 efficiency 当成 base success。
3. Line C 已独立落盘，后续可以用同一指标判断 near-pass base 是否存在 coupling collapse / noise leakage / reservoir trapping。
4. Functional diagnostic 仍未 official；base gate 未开，control-resistant 几何增益也不能替代 base qualification。
5. 若 SimpleFastTaskGeometry 不能通过 A4，应恢复更强的 branch-specific Legendre 表达深度或设计新的低成本表达 primitive；若 A4 通过再打开 A5 task。
```
