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
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b50a_temp100_20260520T101000Z
```

核心结论：

1. v12.5.2 已真实执行 A 线 Triton component forward truth、B47 manual backward/full-step truth、Line C train-probe/signal-reservoir 诊断与 Functional cloned diagnostic。
2. Triton F2/F3 是真实 custom kernel smoke，不是标签；其结果按 artifact 中的 ratio/error 判定，没有写成 full fused success。
3. 当前 route 按 A1/A4/A5 顺序判定；A4 expression 已在 A1 full-step official pass 后真实运行，A5 task 只有 A4 通过才允许打开。
4. Functional rows 都是 diagnostic；没有把 control gap 或几何分数写成 official success。
5. No-fake audit rows checked = `476`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 v12.5.2 三线 runner | 独立输出 `v125_*` artifacts；复用 v12.4 strict PureKAN model，不改低 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Triton F2/F3 component kernels | 真实执行 residual input norm + Legendre basis / hidden preactivation component fusion；记录 correctness 与 timing，不冒充 full fused layer。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 执行中修复 Triton `tl.tanh` blocker | smoke 暴露当前 Triton 3.0 无 `tl.tanh`；改为等价 `2/(1+exp(-2x))-1` 后 F2/F3 才真实执行，失败前结果不写成 kernel success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 F4 hybrid full-forward repair | 复用 Triton F2/F3 的 norm/basis/hidden preactivation，再接 torch readout；用于检查 component fusion 组合后是否足够，不冒充 single-call full fused kernel。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48 SimpleFastTaskGeometry | 按 R1/R6 推荐方向，在 full GatedLQ fusion 收益不足时转向 hinge/direct edge basis + minimal quadratic sketch；fixed train-stream norm，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | B48 fast rewrite + manual CE step | 将 direct hinge 三路 readout 合并为单个 readout matmul，将 quadratic readout 改为扁平 matmul，并新增 SimpleFastTaskGeometry manual CE backward；只减少算子/反向开销，不改 gate。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48d low-temperature repair | 保持 B48a h128 / quad030 结构，只把全局 logit gain 初始化为 `0.50`，用于修 A5 AUC/near-pass；不改 loss、不做 post-hoc calibration。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48e mid-temperature repair | 在 B48a `0.75` 与 B48d `0.50` 之间取全局 logit gain `0.65`，用于验证 AUC/accuracy trade-off；仍不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B49a diagonal quadratic direct channel | 在 B48 的 direct edge basis 中加入 fixed-centered `z^2` diagonal quadratic channel，并把 sketch hidden 降到 h96 控制效率；用于修 Fashion NLL/AUC，不按 dataset 分支。 |
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
| `F0-current-B47-manual-autograd-forward` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `19.47929433389001` | `1.0` | `0.0` | `0` |
| `F0b-current-B42-fastreuse-forward` | `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `14.400696475784521` | `0.739282246520103` | `` | `0` |
| `K2-current-LiteGated-forward` | `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `13.035207862529191` | `0.6691827557557144` | `` | `0` |
| `K3-SimpleFastTaskGeometry-forward` | `B50a-SimpleFastTaskGeometry-h128-temp100` | `5.099857344201693` | `0.26180914240455716` | `` | `0` |
| `K3-compiled-SimpleFastTaskGeometry-forward` | `B50a-SimpleFastTaskGeometry-h128-temp100` | `3.4055591181402627` | `0.17482969658789346` | `2.7567148208618164e-07` | `0` |
| `F2-triton-residual-input-norm-plus-legendre-basis` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.8623011477619158` | `0.09560413821161326` | `0.0010108202695846558` | `0` |
| `F3-triton-residual-input-norm-legendre-hidden-preactivation` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `3.3840801615851253` | `0.17372704080442555` | `1.9073486328125e-05` | `0` |
| `F4-triton-hybrid-full-forward-with-torch-readouts` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `12.615290575458403` | `0.6476256459408984` | `0.0004998147487640381` | `0` |

## 3. A2/A3 Backward 与 Full Step

| candidate | forward ratio | backward ratio | step ratio | memory ratio | exploratory pass |
|---|---:|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.0` | `1.0` | `1.0` | `1.0` | `1` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `4.636459146646677` | `2.384936603948625` | `2.4698265517296227` | `1.2055060034305318` | `0` |
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `2.1658354676490874` | `1.3834068536597184` | `1.425829141406614` | `1.108542024013722` | `0` |
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `1.297947838692764` | `1.0970505577310157` | `1.1263100225094536` | `0.40344768439108064` | `1` |

解释：若 B47 manual CE step 未过 A1/A3 efficiency，本轮不会推进 expression/task official gate。

## 4. A4 Expression Qualification

| candidate | trainable key delta OK | frozen key R2 OK | dead basis max | A4 pass |
|---|---:|---:|---:|---:|
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `1` | `0` | `0.0` | `1` |

解释：A4 pass 规则是 E1/E2/E6/E8 的 B1 delta 全部 `>= -0.01`，或这些 key targets 的 frozen R2 全部 `>= 0.89` 且 dead basis `<= 0.30`。未满足时 A5 task 继续合法关闭。

## 5. A5 Task Triage

| candidate | mean delta | worst delta | near pass | ECE ok | AUC-time ok | A5 pass |
|---|---:|---:|---:|---:|---:|---:|
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `0.0` | `-0.009765625` | `0.6666666666666666` | `1` | `0` | `0` |

解释：A5 只有 A4 通过后才运行；判定仍要求 mean delta、worst delta、near pass、ECE 与 AUC-time 同时满足，不能用单个 accuracy mean 替代。

## 6. Line C Train-Probe Coupling

| candidate | CouplingR2 | CouplingCorr | KernelDrift | ECE delta |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `0.21124812652485114` | `0.459617555141449` | `6.952402114868164` | `0.20366013050079346` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.27582591215472696` | `0.5251991748809814` | `1.2660874128341675` | `0.022825509309768677` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.2758251436014789` | `0.5251985788345337` | `1.2660882472991943` | `0.02282547950744629` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.21062014288593756` | `0.45893394947052` | `4.434932708740234` | `0.16107259690761566` |
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `0.22210064942587027` | `0.47127580642700195` | `1.5819576978683472` | `0.17265227437019348` |

## 7. Line C Signal / Reservoir / Noise

| candidate | signal effective rank | real reservoir ratio | noise signal leak | real-noise gap |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `3.4227702617645264` | `0.08012042939662933` | `0.4324778914451599` | `0.8470981121063232` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `1.8779172897338867` | `0.8620330691337585` | `0.13666313886642456` | `0.505378007888794` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.8779184818267822` | `0.8620335459709167` | `0.13666288554668427` | `0.505378246307373` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `1.0604147911071777` | `0.7806601524353027` | `0.10371564328670502` | `0.7313432693481445` |
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `1.2546095848083496` | `0.86297208070755` | `0.0011304743820801377` | `0.6918878555297852` |

## 8. Functional Diagnostic

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.4049402009186045` | `0.592840388417244` | `-0.9977805893358485` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.40494005259166777` | `0.5928403288125992` | `-0.997780381404267` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-0.7320558927514829` | `0.26641715224832296` | `-0.9984730449998058` | `0` |
| `B50a-SimpleFastTaskGeometry-h128-temp100` | `-0.7043352526053694` | `0.28868721425533295` | `-0.9930224668607024` | `0` |

解释：这些结果均为 cloned diagnostic。`base_qualified = false` 时不允许打开 official functional short-run。

## 9. No-Fake / Hash

```text
rows_checked = 476
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v125_backward_kernel_truth.csv` | `cd634469ff1b467acfe018f56a637eb10f87f8d5ad5de04b1af2362093f80948` |
| `v125_control_matrix.csv` | `2cab31ff12c2fa3f71a61a956daea594b9096fb38171c538482129ffdb6db5a2` |
| `v125_expression_battery.csv` | `43f399376795671e0de817c02bf9a7acfcb6d8874f475273158003f1078c2355` |
| `v125_expression_summary.csv` | `b0ad04832e85802436a5cae59b6fa76a316f7f55eee271d0f5cefc76a1576bc4` |
| `v125_forward_kernel_truth.csv` | `5301a7ba068d79cbc2e8a5dd94a91d5813c5eee1e9273611028698b1b89a0b61` |
| `v125_frozen_readout.csv` | `c25819ecfe0e8c371d462d4df57466914f60c1314abe4dc86e51cbe9f25415fe` |
| `v125_full_step_efficiency.csv` | `0a3cb2ce7aba8dec42bb6ecef2cfc7d316b570215233c854b9bddb83d5c8ee12` |
| `v125_matrix_span.csv` | `2ffa8fccc5331b9f1b6fa3c923b6ab2ea43b8086ddb081f1a802d0d2cb7a7a06` |
| `v125_provenance_audit.csv` | `5cb6d008b23774c1d42154014128dff56cc0c27627250218c1963824ebbf8163` |
| `v125_route_decision.json` | `bd76e4c20baf1564b41ac6afeaeef88986d3e9a3471bf6ead26090e3328f15e5` |
| `v125_signal_reservoir_sketch.csv` | `89908eefbe6fd63d6639515f29e94eb0051157bacffe02c8ff277fc5e74b828f` |
| `v125_task_failure_table.csv` | `bf3f0d1e9e439e1346e3f27326f75c17d886c05064b015468cf2c3eec4c4761f` |
| `v125_task_triage.csv` | `1acbbc40d0489370d7d086b13d0e12a153539a820d9b8ee858b2d7ca3cba714b` |
| `v125_train_probe_coupling.csv` | `b39f88dd168781d36c8fdef59b96454389e7cc3fadd0eca38cb959dd61f9716e` |

## 10. 最终分析结论

```text
1. v12.5.2 已把 v12.4 的“manual backward 还不够”推进为可审计的 kernel component truth。
2. 若 B48 compiled simple primitive 已达到 A1/A3 efficiency，必须继续看 A4 expression；不能把 efficiency 当成 base success。
3. Line C 已独立落盘，后续可以用同一指标判断 near-pass base 是否存在 coupling collapse / noise leakage / reservoir trapping。
4. Functional diagnostic 仍未 official；base gate 未开，control-resistant 几何增益也不能替代 base qualification。
5. 若 SimpleFastTaskGeometry 不能通过 A4，应恢复更强的 branch-specific Legendre 表达深度或设计新的低成本表达 primitive；若 A4 通过再打开 A5 task。
```
