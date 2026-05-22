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
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_foreachadamw_20260520T112000Z
```

核心结论：

1. v12.5.2 已真实执行 A 线 Triton component forward truth、B47 manual backward/full-step truth、Line C train-probe/signal-reservoir 诊断与 Functional cloned diagnostic。
2. Triton F2/F3 是真实 custom kernel smoke，不是标签；其结果按 artifact 中的 ratio/error 判定，没有写成 full fused success。
3. 当前 route 按 A1/A4/A5 顺序判定；A4 expression 已在 A1 full-step official pass 后真实运行，A5 task 只有 A4 通过才允许打开。
4. Functional rows 都是 diagnostic；没有把 control gap 或几何分数写成 official success。
5. No-fake audit rows checked = `191`，fake/proxy/cpu = `0` / `0` / `0`。

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
| `F0-current-B47-manual-autograd-forward` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `12.87337511666382` | `1.0` | `0.0` | `0` |
| `F0b-current-B42-fastreuse-forward` | `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `17.398756506002687` | `1.3515302978688961` | `` | `0` |
| `K2-current-LiteGated-forward` | `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `8.747843983597196` | `0.6795299526597055` | `` | `0` |
| `K3-SimpleFastTaskGeometry-forward` | `B48a-SimpleFastTaskGeometry-h128-temp075` | `5.103354224998577` | `0.3964270580752819` | `` | `0` |
| `K3-compiled-SimpleFastTaskGeometry-forward` | `B48a-SimpleFastTaskGeometry-h128-temp075` | `1.7945888689345595` | `0.13940313652567854` | `2.384185791015625e-07` | `0` |
| `F2-triton-residual-input-norm-plus-legendre-basis` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.3003379584823438` | `0.10100987089229875` | `0.0010108202695846558` | `0` |
| `F3-triton-residual-input-norm-legendre-hidden-preactivation` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `3.171075308328891` | `0.2463281990613419` | `1.9073486328125e-05` | `0` |
| `F4-triton-hybrid-full-forward-with-torch-readouts` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `11.484979910533298` | `0.8921498679601648` | `0.0004998147487640381` | `0` |

## 3. A2/A3 Backward 与 Full Step

| candidate | forward ratio | backward ratio | step ratio | memory ratio | exploratory pass |
|---|---:|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.0` | `1.0` | `1.0` | `1.0` | `1` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `5.794286012728808` | `2.084124479800807` | `2.7636091771412556` | `1.2055060034305318` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `2.2013408501761025` | `1.0777066695416464` | `1.3211055463753936` | `1.108542024013722` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `3.576311058409517` | `1.2183702779284364` | `2.214660506282393` | `0.40344768439108064` | `0` |

解释：若 B47 manual CE step 未过 A1/A3 efficiency，本轮不会推进 expression/task official gate。

## 4. A4 Expression Qualification

| candidate | trainable key delta OK | frozen key R2 OK | dead basis max | A4 pass |
|---|---:|---:|---:|---:|
| `not_run` |  |  |  | `0` |

解释：A4 pass 规则是 E1/E2/E6/E8 的 B1 delta 全部 `>= -0.01`，或这些 key targets 的 frozen R2 全部 `>= 0.89` 且 dead basis `<= 0.30`。未满足时 A5 task 继续合法关闭。

## 5. A5 Task Triage

| candidate | mean delta | worst delta | near pass | ECE ok | AUC-time ok | A5 pass |
|---|---:|---:|---:|---:|---:|---:|
| `not_run` |  |  |  |  |  | `0` |

解释：A5 只有 A4 通过后才运行；判定仍要求 mean delta、worst delta、near pass、ECE 与 AUC-time 同时满足，不能用单个 accuracy mean 替代。

## 6. Line C Train-Probe Coupling

| candidate | CouplingR2 | CouplingCorr | KernelDrift | ECE delta |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `0.21124812652485114` | `0.459617555141449` | `6.952402114868164` | `0.20366013050079346` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.27582591215472696` | `0.5251991748809814` | `1.2660874128341675` | `0.022825509309768677` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.2758251436014789` | `0.5251985788345337` | `1.2660882472991943` | `0.02282547950744629` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.21062014288593756` | `0.45893394947052` | `4.434932708740234` | `0.16107259690761566` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0.2157366284691863` | `0.46447503566741943` | `1.6143748760223389` | `0.2021782398223877` |

## 7. Line C Signal / Reservoir / Noise

| candidate | signal effective rank | real reservoir ratio | noise signal leak | real-noise gap |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `3.4227702617645264` | `0.08012042939662933` | `0.4324778914451599` | `0.8470981121063232` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `1.8779172897338867` | `0.8620330691337585` | `0.13666313886642456` | `0.505378007888794` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.8779184818267822` | `0.8620335459709167` | `0.13666288554668427` | `0.505378246307373` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `1.0604147911071777` | `0.7806601524353027` | `0.10371564328670502` | `0.7313432693481445` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `1.2135820388793945` | `0.870010256767273` | `0.0013872733106836677` | `0.5224477052688599` |

## 8. Functional Diagnostic

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.4049402009186045` | `0.592840388417244` | `-0.9977805893358485` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.40494005259166777` | `0.5928403288125992` | `-0.997780381404267` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-0.7320558927514829` | `0.26641715224832296` | `-0.9984730449998058` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `-0.7033287750843986` | `0.2942679077386856` | `-0.9975966828230842` | `0` |

解释：这些结果均为 cloned diagnostic。`base_qualified = false` 时不允许打开 official functional short-run。

## 9. No-Fake / Hash

```text
rows_checked = 191
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v125_backward_kernel_truth.csv` | `17869801936a1d45bb5c33c56199c2ca29b411e33b51c6f560954f4bac3799c1` |
| `v125_control_matrix.csv` | `912133906c65b1a6b9a0753ee49027ff1cc07a4999a9ec8f44c8ca7107968bf2` |
| `v125_expression_battery.csv` | `624b8d70a18b052a0458f67615feb19db169118230b0141950061381a4462288` |
| `v125_expression_summary.csv` | `aa6718594024be48e6ded5f7729f79014cdabcd8c6a1cc263220c55eda6a1e98` |
| `v125_forward_kernel_truth.csv` | `fbf1220cf7a38491c00da98544d4d8e25bccf0658189232ed13ffbe94c8c9fdb` |
| `v125_frozen_readout.csv` | `d4ff597928129c35f3e617abc80134f79708064b380363545e54efeca44bc76d` |
| `v125_full_step_efficiency.csv` | `c92c02e826ea387484da1083ea31f7f876b41bf0860a68d2dd463d9b93c62b19` |
| `v125_matrix_span.csv` | `dc87f117a7905b7b7e8b23980dcce63495aa673bf8908f2703c112588834ca70` |
| `v125_provenance_audit.csv` | `856eca58bd48823e1faf778a5f0e7d4e6871a2bd1e59f9d37e56007e5aca5439` |
| `v125_route_decision.json` | `37cfaf6809255058bac685c5b055027b40acb1af4cb744ca57c594b75f4614e8` |
| `v125_signal_reservoir_sketch.csv` | `8603bfc68d07c1057567f00c09d84a3ba829b0e2176b5c8d9d89dbc2c11b5fd1` |
| `v125_task_failure_table.csv` | `ba20cb90d8b8adf06054cf1beb5acfa443f8b83f4dea5b280922e410030734b3` |
| `v125_task_triage.csv` | `8adde185f2b2ad458c147b22bb34ed9d9f6562d95373b933af401f65ef9a1ee6` |
| `v125_train_probe_coupling.csv` | `36b8c881952b9e2cf8290b89c2e17ca1898155434328a08bc66812f64b10c90d` |

## 10. 最终分析结论

```text
1. v12.5.2 已把 v12.4 的“manual backward 还不够”推进为可审计的 kernel component truth。
2. 若 B48 compiled simple primitive 已达到 A1/A3 efficiency，必须继续看 A4 expression；不能把 efficiency 当成 base success。
3. Line C 已独立落盘，后续可以用同一指标判断 near-pass base 是否存在 coupling collapse / noise leakage / reservoir trapping。
4. Functional diagnostic 仍未 official；base gate 未开，control-resistant 几何增益也不能替代 base qualification。
5. 若 SimpleFastTaskGeometry 不能通过 A4，应恢复更强的 branch-specific Legendre 表达深度或设计新的低成本表达 primitive；若 A4 通过再打开 A5 task。
```
