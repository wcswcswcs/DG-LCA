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
results/v12_5_2_efficiency_functional_manifold/smoke_v1252
```

核心结论：

1. v12.5.2 已真实执行 A 线 Triton component forward truth、B47 manual backward/full-step truth、Line C train-probe/signal-reservoir 诊断与 Functional cloned diagnostic。
2. Triton F2/F3 是真实 custom kernel smoke，不是标签；其结果按 artifact 中的 ratio/error 判定，没有写成 full fused success。
3. 当前 route 仍未打开 base：`base_qualified = false`，所以 expression/task confirm 与 official functional short-run 均合法关闭。
4. Functional rows 都是 diagnostic；没有把 control gap 或几何分数写成 official success。
5. No-fake audit rows checked = `139`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 v12.5.2 三线 runner | 独立输出 `v125_*` artifacts；复用 v12.4 strict PureKAN model，不改低 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Triton F2/F3 component kernels | 真实执行 residual input norm + Legendre basis / hidden preactivation component fusion；记录 correctness 与 timing，不冒充 full fused layer。 |
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
| `F0-current-B47-manual-autograd-forward` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `10.06159865310989` | `1.0` | `0.0` | `0` |
| `F0b-current-B42-fastreuse-forward` | `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `12.584045823022953` | `1.2507004360716985` | `` | `0` |
| `K2-current-LiteGated-forward` | `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `6.659207244127827` | `0.6618438554065725` | `` | `0` |
| `F2/F3-triton-failed` | `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `inf` | `inf` | `failed:AttributeError:module 'triton.language' has no attribute 'tanh'` | `0` |

## 3. A2/A3 Backward 与 Full Step

| candidate | forward ratio | backward ratio | step ratio | memory ratio | exploratory pass |
|---|---:|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.0` | `1.0` | `1.0` | `1.0` | `1` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `5.250949501552541` | `2.2119630706140123` | `2.5981198742700546` | `1.044479897894065` | `0` |

解释：若 B47 manual CE step 未过 A1/A3 efficiency，本轮不会推进 expression/task official gate。

## 4. Line C Train-Probe Coupling

| candidate | CouplingR2 | CouplingCorr | KernelDrift | ECE delta |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `0.9988084436749775` | `0.9994902610778809` | `9.647709846496582` | `0.04702761769294739` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.9973838563138185` | `0.9989147186279297` | `2.260103464126587` | `-0.09008382260799408` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.997383652568166` | `0.9989147186279297` | `2.260098934173584` | `-0.09008394181728363` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.9999603465759157` | `0.9999846816062927` | `5.287623882293701` | `-0.08392524719238281` |
| `B1r-ReLU-KAN-stream-K2-repair` | `0.10796460328723656` | `0.749566912651062` | `0.03609845042228699` | `0.0044088587164878845` |

## 5. Line C Signal / Reservoir / Noise

| candidate | signal effective rank | real reservoir ratio | noise signal leak | real-noise gap |
|---|---:|---:|---:|---:|
| `MLP-same-param-AdamW` | `1.4146422147750854` | `0.6000441908836365` | `0.018301814794540405` | `0.8674342632293701` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `1.063552975654602` | `0.7408638596534729` | `0.05703643709421158` | `0.7948987483978271` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `1.063552975654602` | `0.7408649921417236` | `0.057036835700273514` | `0.7948989868164062` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `1.0556915998458862` | `0.9984378218650818` | `0.0017309063114225864` | `0.7251960039138794` |
| `B1r-ReLU-KAN-stream-K2-repair` | `1.4755059480667114` | `0.737666666507721` | `0.018307365477085114` | `0.004191398620605469` |

## 6. Functional Diagnostic

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.46972002364271404` | `0.5268760100007057` | `-0.9965960336434198` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.46972014414044183` | `0.5268758609890938` | `-0.9965960051295356` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-1.1329674001387036` | `-0.1364654004573822` | `-0.9965019996813214` | `0` |

解释：这些结果均为 cloned diagnostic。`base_qualified = false` 时不允许打开 official functional short-run。

## 7. No-Fake / Hash

```text
rows_checked = 139
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v125_backward_kernel_truth.csv` | `a58b218faf830f8b8c4e27bb7c017d04c3615682416ff27e1cf329d2dafca980` |
| `v125_control_matrix.csv` | `382c7c5a42c337374c4a20e8adca5b1b7ed8add7693816f072eada6c515df6a0` |
| `v125_forward_kernel_truth.csv` | `ba99df9868e9fbd817de0585887e96834e022cececf8c3d9ecf1c38add7c32ff` |
| `v125_full_step_efficiency.csv` | `524653522ec4e308384e1990114f7238ff5b4c3557d83a53969e7518193d3d82` |
| `v125_provenance_audit.csv` | `6c8c0aa710438fba879fcd0ee4f775a29868d958c534eca3ca5bc640ce19822c` |
| `v125_route_decision.json` | `cde52e5c97fc094cf0cb64669dfdbf01018fe7b260beaf569cfa680b3b68f055` |
| `v125_signal_reservoir_sketch.csv` | `242f49f488a8965a13162c1c88d340641baee370afe2ae8aa1fa370666778bfb` |
| `v125_train_probe_coupling.csv` | `4ec92366bdfde693a6be72b89e291d859a3a251a059bcb7c165e59284d5a4cb3` |

## 8. 最终分析结论

```text
1. v12.5.2 已把 v12.4 的“manual backward 还不够”推进为可审计的 kernel component truth。
2. 本轮没有证明 full fused GatedLQ 已达到 MLP-like efficiency；若 route 仍是 R1，应继续 lower-level full forward+backward fusion，而不是 schedule/scale 小修。
3. Line C 已独立落盘，后续可以用同一指标判断 near-pass base 是否存在 coupling collapse / noise leakage / reservoir trapping。
4. Functional diagnostic 仍未 official；base gate 未开，control-resistant 几何增益也不能替代 base qualification。
5. 下一步应优先实现 single-call full-layer fused forward+backward 或转 SimpleFastTaskGeometry，而不是继续 Python-level backward。
```
