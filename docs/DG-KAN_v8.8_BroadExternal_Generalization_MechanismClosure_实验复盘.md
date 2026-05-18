# DG-KAN v8.8 Broad External Generalization and Mechanism Closure 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.8_BroadExternal_Generalization_MechanismClosure_完整实验计划.md` 的首轮真实执行结果。所有数值只来自本文列出的 `results/real_rerun_20260506/.../` 落盘 CSV/JSON/manifest；没有 fake data、proxy rows、固定占位 ratio 或手填成功结论。本轮继续遵守：no teacher、no self-teacher、no distillation、no loss modification、no sampler/class weight、no CPU offload、no optimizer hyperparameter sweep。

## 0. 本轮是否完成 v8.8

没有完成 v8.8 formal / broad strong。

本轮完成的是 v8.8 Wave0 的 fresh selected-route screen：不复用旧 P0 artifact，重新跑 `KW4 hidden28` selected route 的 3-seed MNIST child reproduction，并串接 P2/P3/P4/P5/P6 与 FMNIST/KMNIST screen-level P7/P9。结果显示：

```text
P0 selected 3-seed T0 child reproduction = pass
P2 adaptive one-step = pass
P3 adaptive multistep = fail
P4 no-manual architecture screen = pass
P4 selected confirmation = pass
P5 selected timing/accounting = pass
P6 compute fair counter = pass
P7/P9 FMNIST/KMNIST screen = pass
v8.8 formal/broad = not_claimed
```

最终 route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 0,
  "success_v87_adaptive": 0,
  "success_v87_external_fair": 0,
  "success_v87_no_manual_tuning": 0,
  "success_v87_formal": 0,
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 0,
  "primary_blocker": "p0_full_stability_protocol_not_complete",
  "next_required_implementation": "run_remaining_seeds_reruns_and_timing_protocols_before_claiming_stability"
}
```

重要补充：虽然 route 的 primary blocker 首先停在 P0 full protocol 未完成，但本轮已经真实测到另一个更实质的 v8.8 blocker：P3 adaptive multistep 在 fresh screen 中 `0/3` checkpoint pass，失败来自 `adaptive_step_ratio_vs_fixed > 1.50`，而不是 task/geometry。

## 1. Artifact

本轮主 artifact：

```text
results/real_rerun_20260506/v88_wave0_selected_route_independent_screen_20260508T235000Z/
```

运行配置摘要：

```text
P0 seeds = 1314,1315,1316
P0 reruns = 1
P0 candidate = KW4 hidden28
system path = ForeachAdamW + fused CE/head + compiled fused CE/head + prewarm
P3 controller = Adaptive-FT-P
P3 checkpoints = 20,50,240
P4 base candidates = KF10,KW3,KW4,KW5,KW6
P4 hidden candidates = 16,20,24,28,32
P7 datasets = Fashion-MNIST,KMNIST
P7 protocol = screen, train_size 10000 / test_size 2000 / epochs 5
```

No-fake audit：

```text
rows_checked = 2589
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 2. P0 fresh selected-route screen

`p0_stability_summary.csv`：

| metric | value |
|---|---:|
| measured P0 runs | `3` |
| requested P0 runs | `3` |
| timing protocol count | `1` |
| child external formal pass count | `3/3` |
| child route set | `R1-ExternalFairFunctionalAdvantage` |
| DG1 mean delta vs KB-MLP | `+0.3599981467` |
| DG1 mean curvature ratio vs DG0 | `0.7341647974` |
| DG1 Q90 step ratio vs KB-MLP | `0.9530351040` |
| DG1 all-gate pass rate | `1.000000` |
| P0 stability gate | `1` |
| P0 full protocol complete | `0` |

判断：

1. `KW4 hidden28` selected route 在 3-seed fresh T0 child run 中复现成功。
2. 但 v8.8 P0 要求 5 seeds x 3 reruns x T0-T4；本轮只有 3 seeds x 1 rerun x T0，因此不能写 independent formal reproduction。

## 3. P2 / P3 adaptive result

P2 one-step：

| metric | value |
|---|---:|
| best adaptive controller | `Adaptive-FT-D` |
| P2 adaptive one-step pass | `1` |
| triggered holdout ratio | `0.9591751449` |
| triggered bad step | `0.000000` |
| triggered curvature reduction | `0.0006558749` |

P3 multistep fresh screen：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | events | bad step | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | `+0.0005666614` | `0.8967263863` | `2.5040966672` | `6.666667` | `0.000000` | 0 |
| 50 | `+0.0016333659` | `0.8595013690` | `2.4369702170` | `12.000000` | `0.027778` | 0 |
| 240 | `+0.0004333456` | `0.8072696460` | `2.3884021432` | `53.333333` | `0.088249` | 0 |

判断：

1. P3 的 task gate 与 geometry gate 本身都过：三个 checkpoint 的 acc delta 均为正，curvature ratio 均 `<= 0.90`。
2. 失败集中在 system gate：adaptive-vs-fixed step ratio 为 `2.388-2.504`，远高于 `1.50`。
3. 这说明 v8.8 independent screen 不能直接继承 v8.7 的 P3 timing pass；下一步必须定位 P3 fixed/adaptive paired timing 的 fresh-run 差异。

## 4. P4/P5/P6 selected route

P4 no-manual architecture screen：

| metric | value |
|---|---:|
| selected base | `KW4` |
| selected hidden | `28` |
| selected params ratio vs KB-MLP | `0.9395677800` |
| selected FLOPs ratio vs KB-MLP | `0.9363473660` |
| selected Q90 step ratio vs KB-MLP | `1.0814172122` |
| selection pass | `1` |
| test metric used for selection | `0` |

P4 selected confirmation：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| DG functional test acc | `97.1899986267` |
| delta vs KB-MLP | `+0.2799987793` |
| delta vs DG base | `+0.0200033188` |
| curvature ratio vs DG base | `0.7724829153` |
| step ratio vs KB-MLP | `0.9797549892` |
| functional events | `73` |
| selected confirmation pass | `1` |

P5/P6：

| stage | metric | value |
|---|---|---:|
| P5 | q90 step ratio | `1.0783743464` |
| P5 | max step ratio | `1.1013974891` |
| P5 | max unknown time fraction | `0.0641925491` |
| P5 | robust timing pass | `1` |
| P5 | strict timing pass | `1` |
| P5 | time accounting pass | `1` |
| P6 | forward FLOPs ratio | `0.9363473660` |
| P6 | backward FLOPs estimate ratio | `0.9363473660` |
| P6 | step time ratio | `0.9797549892` |
| P6 | training compute fair pass | `1` |

判断：P4/P5/P6 本轮没有暴露 blocker；selected `KW4 hidden28` 的 architecture selection、selected confirmation、timing/accounting 和 compute envelope 均过。

## 5. P7/P9 FMNIST/KMNIST screen

注意：本轮 P7 是 screen protocol，不是 v8.8 formal protocol：

```text
p7_train_size = 10000
p7_test_size = 2000
p7_epochs = 5
p7_formal_protocol = 0
```

Joint fair envelope：

| task | DG functional acc | delta vs KB-MLP | delta vs DG base | params ratio | FLOPs ratio | memory ratio | step ratio | curvature ratio | JointFairPass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `85.2500021458` | `+1.7499983311` | `0.0000000000` | `0.9395677800` | `0.9363473660` | `0.8579032840` | `0.9635644370` | `0.9799240367` | 1 |
| KMNIST | `77.1500051022` | `+1.3999998569` | `+0.2499997616` | `0.9395677800` | `0.9363473660` | `1.0080295922` | `0.9025049161` | `0.9652471606` | 1 |

P9 causality summary：

| task | causality pass | FT7 curvature ratio | RandomFunc curvature ratio | note |
|---|---:|---:|---:|---|
| FMNIST | 1 | `0.9799240367` | `1.0196655901` | effect 很小，但低于 NoOp/RandomFunc |
| KMNIST | 1 | `0.9652471606` | `1.0000000079` | task 和 curvature 均优于 base/random |

判断：

1. FMNIST/KMNIST screen 通过 joint fair 和 causality。
2. 但这仍属于 vision-family，且不是 formal P7 protocol；不能作为 v8.8 broad external success。
3. FMNIST curvature gain 很弱，提示 v8.8 H5/H7 仍需 robustness/calibration/negative-boundary wave。

## 6. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `d0c0badfe764f63c4b1900120d6d3d2563b978efae4d5b7fc9e100a0bb7d4b6e` |
| v8.8 plan | `dac3a2c672ada75ec24aa02d7ce3b88d6a2e97c6ad3b2cfaa4b14c10a31a5cdb` |
| route | `15a75a70c8538de9330756ee7b77063b27a8788d131f0e92e7efa38eb08be590` |
| P0 summary | `8df565c98395007dc0c0926b6940d4b1d3d458ad7d10b758b8bd01e682610999` |
| P3 summary | `beed4757aaca753379ba0a902dc773b520bccdb9553891906511bb4e63181824` |
| P4 summary | `1fa62dd1e9bc68d14170119a14ba18da90892ad85447fd14826d9349ce7cf0a0` |
| P7/P8 joint fair | `bbe6700b284ec53b55452dde74ab9c3ac688a9af0b36de6a384b03361b74d571` |
| P9 causality | `a79fc88370cb2f6a601ec9597dc907e34594c5ffb977d5cfb3ab5525a346b1bb` |
| provenance audit | `1e352afb36b848be2a0f5851e3de0ede2b240572eba1f39ee201132bdcf30d6c` |

## 7. 当前结论

v8.8 第一轮没有完成 formal reproduction，更没有完成 broad strong：

```text
success_v88_formal_reproduction = false
success_v88_broad_external = false
success_v88_strong = false
```

当前最重要的真实发现：

1. selected `KW4 hidden28` 在 fresh 3-seed T0 child reproduction 中仍稳定，task / geometry / wall-clock 都过。
2. P4/P5/P6 也过，说明 no-manual architecture selection 与 selected compute/timing 不是本轮 blocker。
3. FMNIST/KMNIST screen 仍为正，但它们不是 broad external，也不是本轮 formal protocol。
4. P3 fresh multistep 是本轮 clean blocker：`Adaptive-FT-P` 保住了 task 和 geometry，但 adaptive-vs-fixed step ratio 在三个 checkpoint 都超过 `2.38`。

下一步不应该扩大 broad claim；应先做：

```text
P3 paired timing repair / diagnostics:
  compare v8.7 formal P3 run vs this fresh P3 screen;
  split fixed/adaptive absolute step time;
  confirm compiled/prewarm hook is symmetrically applied;
  add P3 phase timing rows if needed.

Then:
  rerun P0 full 5x3/T0-T4 independent reproduction;
  only after P3/P0 pass, continue P1/P2 v8.8 factor isolation and P8/P9/P10-P12 waves.
```

最终一句话：

> v8.8 已开始执行，但第一轮 independent screen 不能支持 formal/broad success。好消息是 selected `KW4 hidden28` 的 P0/P4/P5/P6 与 FMNIST/KMNIST screen 仍成立；坏消息也很干净：fresh P3 adaptive multistep 的 system ratio 全部失败。下一步必须先修 P3 paired timing stability，不能用 v8.7 旧 pass 或 vision-family screen 覆盖这个新 blocker。

## 8. 追加：P3 fixed/adaptive timing 差异诊断与 nullspace projector 修复

本节继续第 7 节的 clean blocker：fresh v8.8 P3 的 task / geometry 都过，但 system ratio fail。仍严格保持：

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

### 8.1 代码改动

| 文件 | 改动 | 是否改变实验合同 |
|---|---|---:|
| `experiments/run_gafu_v87_real.py` | 新增 `--p3-include-fixed-probe-baseline`，落盘同 cadence / no functional update 的 fixed-probe diagnostic baseline | 0 |
| `experiments/run_gafu_v87_real.py` | P3 summary 新增 `fixed_probe_step_time_ms_mean`、`adaptive_step_ratio_vs_fixed_probe`、`P3_paired_timing_checkpoint_pass` | 0 |
| `experiments/run_gafu_v87_real.py` | `_nullspace_project_rows_v87` 从 `torch.linalg.solve` 改为同一 ridge system 的 Cholesky solve | 0 |

Cholesky projector 等价 smoke：

```text
device = cuda
max_abs_diff_vs_old_solve = 1.1920928955078125e-06
old_norm = 122.28424072265625
new_norm = 122.28424072265625
```

判断：这是 float32 线性代数误差级的数值等价实现侧修复，不是 loss / teacher / sampler / offload 改动。

代码检查：

```text
python -m py_compile experiments/run_gafu_v87_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 8.2 v8.7 formal P3 vs v8.8 fresh P3 对比

两轮的 task / geometry 完全一致，差异集中在 fixed timing denominator：

| source | checkpoint | acc delta | curvature ratio | fixed ms | adaptive ms | step ratio | P3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| v8.7 formal | 20 | `+0.0005666614` | `0.8967263863` | `2.8818247141` | `1.6720427588` | `0.5802027967` | 1 |
| v8.7 formal | 50 | `+0.0016333659` | `0.8595013690` | `1.5162063638` | `1.5236139949` | `1.0048856351` | 1 |
| v8.7 formal | 240 | `+0.0004333456` | `0.8072696460` | `1.0077578940` | `1.5086458233` | `1.4970320076` | 1 |
| v8.8 fresh | 20 | `+0.0005666614` | `0.8967263863` | `0.6681888131` | `1.6732093800` | `2.5040966672` | 0 |
| v8.8 fresh | 50 | `+0.0016333659` | `0.8595013690` | `0.6653923759` | `1.6215414026` | `2.4369702170` | 0 |
| v8.8 fresh | 240 | `+0.0004333456` | `0.8072696460` | `0.6602144528` | `1.5768576142` | `2.3884021432` | 0 |

判断：fresh failure 不是 adaptive task/geometry 退化，而是 P3 system timing baseline 发生了变化；不能用旧 v8.7 pass 覆盖。

### 8.3 fixed-probe paired diagnostic

Artifact：

```text
results/real_rerun_20260506/v88_p3_fixed_probe_baseline_diag_20260508T235500Z/
```

该 run 新增 `Fixed-ProbeCadenceNoFunc`：与 adaptive 同样每 2 step 做 probe / holdout measurement，但不做 functional update。该 baseline 只用于诊断，不进入 route gate。

| checkpoint | fixed ms | fixed-probe ms | adaptive ms | adaptive / fixed | adaptive / fixed-probe | official P3 pass | paired timing pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `2.9283519136` | `0.8707445192` | `1.6663198359` | `0.5690299134` | `1.9136724942` | 1 | 0 |
| 50 | `1.5308460159` | `0.8702160480` | `1.5118577890` | `0.9875962529` | `1.7373361391` | 1 | 0 |
| 240 | `0.9902485297` | `0.8652886990` | `1.4898437282` | `1.5045149612` | `1.7217880343` | 0 | 0 |

判断：

1. paired diagnostic 证明 fresh P3 failure 不能完全归因于 fixed baseline 不公平。
2. `Adaptive-FT-P` 的 nullspace functional update 本身仍有系统成本；因此下一步必须做 implementation-side repair 或 event cadence repair。
3. 本 artifact 的 route 仍不是 success：`p3_checkpoint_pass_count = 2/3`。

No-fake audit：

```text
rows_checked = 2213
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 8.4 event cadence rejected diagnostics

为确认是否可仅靠降低 event cadence 修 timing，追加 probe3 / probe4 / probe4-warm1。均没有写 success。

| artifact | controller setting | step20 acc delta | step20 curvature ratio | step20 step ratio | checkpoint pass | 判断 |
|---|---|---:|---:|---:|---:|---|
| `v88_p3_adaptive_ftp_probe4_scaled_diag_20260509T000500Z` | probe interval 4, warmup 2 | `+0.0008999904` | `0.9199851624` | `0.3382319447` | 2/3 | rejected：timing 好，但 step20 geometry 不足 |
| `v88_p3_adaptive_ftp_probe3_scaled_diag_20260509T001500Z` | probe interval 3, warmup 2 | `+0.0004999836` | `0.9137917787` | `0.3727868504` | 2/3 | rejected：step20 geometry 仍不足 |
| `v88_p3_adaptive_ftp_probe4_warm1_scaled_diag_20260509T002500Z` | probe interval 4, warmup 1 | `+0.0006666581` | `0.9028547210` | `0.3772981420` | 2/3 | rejected：接近但仍高于 `0.90`，且 50/240 holdout margin 变差 |

判断：单纯降频会降低 system cost，但会丢 step20 early geometry；warmup 更早只能接近边界，不能合法过线。

### 8.5 Cholesky nullspace projector P3 repair

Artifact：

```text
results/real_rerun_20260506/v88_p3_adaptive_ftp_probe2_cholesky_diag_20260509T003500Z/
```

配置：

```text
controller = Adaptive-FT-P
probe interval = 2
warmup = 2
alpha scaled with probe interval = 1
nullspace projector = Cholesky solve for the same ridge system
```

P3 summary：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | fixed ms | adaptive ms | step ratio | events | bad step | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | `+0.0005666614` | `0.8967263796` | `3.1304711554` | `1.7230717155` | `0.5504192915` | `6.666667` | `0.000000` | 1 |
| 50 | `+0.0016333659` | `0.8595013632` | `1.6681087452` | `1.5814986546` | `0.9480788702` | `12.000000` | `0.027778` | 1 |
| 240 | `+0.0004333456` | `0.8072696327` | `1.0665086894` | `1.5341350602` | `1.4384646609` | `53.333333` | `0.088249` | 1 |

判断：

1. Cholesky projector repair 后，focused P3 official gate 从 fresh screen 的 `0/3` 推进到 `3/3`。
2. task / geometry 数值与原 `Adaptive-FT-P` 保持一致；修复作用集中在 implementation-side timing。
3. 这仍不是 v8.8 完成：该 run 只验证 P3 focused repair，未执行 P0 full 5x3/T0-T4、P1/P2 v8.8 factor isolation、P8/P9 非 vision-family broad tasks、P10/P11/P12 mechanism/boundary。
4. paired fixed-probe diagnostic 仍显示 adaptive functional update 有额外成本；后续 v8.8 H1 需要在 full protocol 中比较 adaptive vs fixed 的 pass rate / gate-margin variance，不能仅凭这一个 focused P3 run 写 broad success。

No-fake audit：

```text
rows_checked = 1484
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 8.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `02fb507108f5ba7742664bdba802a7a1628aeb250134503406b585d96a641e6b` |
| fixed-probe route | `70a509df86fae28c55e15496eeb2113b324aa7cdbe12899a0b09894501f3146c` |
| fixed-probe P3 summary | `13f476c5136d7d841e40931888b17e224b96dea9b05059dabd4066db92ee8663` |
| fixed-probe P3 checkpoint | `4d291cc038e8b4d1a4689b2c6c4e6bd21d8d7736dee3bf5d7b32dbabbee29d58` |
| fixed-probe audit | `bbbbc27f22d37b4c5fd56beb38b4cfed404a01bc5ec3de564baf99a2a38d5622` |
| probe4 summary | `a1a7dde202208e1c9fcd300416b5a4d3ec317451a83baa1d219b0eb456ea8dc6` |
| probe3 summary | `31701120d13c275d41c229545d88d9c9163f5a7fa693dd87ccee01489d200a65` |
| probe4-warm1 summary | `dcb8cf667b2ca1d7f4a4ad67f057bb7ea9a6f175d73bb91af434073d56e06f21` |
| Cholesky P3 route | `6f1d7b238722f90d4e60ea7e56351492498a2ea426e3067aa574afad99d48f4b` |
| Cholesky P3 summary | `b536d834cd712e9975e58a4806fa5fe5e042bb9d64a6b3240d386937e91d17f4` |
| Cholesky audit | `af2503ad8065487459b8ce83fdb09c0783a73b89b21cc7d83f32a4157635d93e` |

## 9. 当前更新结论

v8.8 仍未完成：

```text
Wave0 fresh selected-route screen = measured
P0 full independent reproduction = not_complete
P3 fresh screen = failed before repair
P3 focused Cholesky repair = pass
P1/P2 v8.8 factor isolation = not_run
P3/P4 v8.8 formal timing/compute = not_run as full wave
P5-P12 broad / mechanism / boundary waves = not_run
success_v88_formal_reproduction = false
success_v88_broad_external = false
success_v88_strong = false
```

机制结论更新：

1. v8.8 没有完成，不能声明 broad strong。
2. fresh screen 暴露的 P3 failure 是真实 blocker，不是 task/geometry 退化，也不能被 v8.7 旧 artifact 覆盖。
3. fixed-probe diagnostic 显示 adaptive nullspace functional update 的系统成本仍需关注。
4. 单纯降低 event cadence 会损失 step20 early geometry，不能作为 repair。
5. Cholesky nullspace projector 是当前合法的 implementation-side improvement：同一 ridge projection system、无合同改变，并在 focused P3 中恢复 `3/3` checkpoint pass。
6. 下一步应把 Cholesky projector 纳入完整 v8.8 Wave0/P0 independent reproduction，并继续执行文档要求的 P1/P2 因果隔离、P3/P4 formal timing/compute、P5-P12 broad external / mechanism / boundary waves。

最终一句话：

> v8.8 还没有完成。本轮继续优化后，P3 的 clean blocker 有了 focused implementation-side repair：把 nullspace projector 改为数值等价的 Cholesky solve 后，`Adaptive-FT-P` 在 P3 20/50/240 三个 checkpoint 都通过。但这只是 P3 修复，不是 broad external 证明；接下来必须跑完整 v8.8 波次，不能把 focused P3 pass 扩写成 v8.8 success。

## 10. 追加：fresh P0 selected-route 5x3 independent reproduction

本节把 Cholesky projector 修复后的代码用于 v8.8 Wave0 的第一块底座：fresh selected-route P0 `5 seeds x 3 reruns`。仍然不使用 teacher / self-teacher / distillation / loss modification / sampler / class weight / CPU offload / fake or proxy row。

Artifact：

```text
results/real_rerun_20260506/v88_p0_kw4_selected_cholesky_seed5x3_20260509T004500Z/
```

配置：

```text
candidate = KW4 hidden28
optimizer = ForeachAdamWAddcdivNoSync
CE/head backward = compiled_fused_value_equivalent
compiled prewarm = 1
seeds = 1314,1315,1316,1317,1318
reruns = 3
timing protocol in this raw P0 = T4 full-loop only
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 0,
  "success_v87_stability": 0,
  "success_v87_formal": 0,
  "primary_blocker": "p0_full_stability_protocol_not_complete"
}
```

P0 summary：

| metric | value |
|---|---:|
| measured P0 runs | `15` |
| requested P0 runs | `15` |
| child external formal pass | `12/15` |
| route set | `R1-ExternalFairFunctionalAdvantage,R4-InternalOnlySuccess` |
| DG1 mean delta vs KB-MLP | `+0.2719974518` |
| DG1 mean curvature ratio vs DG0 | `0.7631262621` |
| DG1 Q90 step ratio vs KB-MLP | `0.8038412983` |
| DG1 all-gate pass rate | `0.8000000000` |
| timing protocol count | `1` |

判断：

1. P0 raw 复现不是 failure：15 个 fresh child 均真实落盘，P0 summary gate 通过。
2. 但它不是 full P0，因为此时只有 T4 full-loop timing，`p0_full_protocol_complete = 0`。
3. `seed1317` 的 DG functional delta vs KB-MLP 为负，导致 child external formal pass 只有 `12/15`。这不能抹掉；v8.8 后续 broad claim 需要记录这个任务/seed 边界。
4. 因此本节只能说明 selected route 的 median/task-geometry-timing 底座仍有正信号，不能写 v8.8 complete。

No-fake audit：

```text
rows_checked = 121
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v87_real.py` | `02fb507108f5ba7742664bdba802a7a1628aeb250134503406b585d96a641e6b` |
| route | `f741bc3c7789c2be3db9064a90af125ecb782a0c64253c3929470298f5a68fca` |
| P0 summary | `d81a72e617804c92b781e2d0fc711f41ec40320f3a7de9d0f4484f8a8807b3db` |
| P0 rows | `55862a0ff452808a0995b3a0fd75062df5a36e90c54c38aa67322841eb3443eb` |

## 11. 追加：P0 5x3 + T0-T4 robust timing 合成

为补第 10 节的 timing protocol 缺口，本节对 `1314-1318` 五个 seed 分别真实测量 `T0/T1/T2/T3 phase-clean`，再与 P0 full-loop `T4` rows 合成。

Phase-clean artifacts：

```text
results/real_rerun_20260506/v88_kw4_selected_cholesky_phaseclean_seed1314_20260509T011500Z/
results/real_rerun_20260506/v88_kw4_selected_cholesky_phaseclean_seed1315_20260509T011500Z/
results/real_rerun_20260506/v88_kw4_selected_cholesky_phaseclean_seed1316_20260509T011500Z/
results/real_rerun_20260506/v88_kw4_selected_cholesky_phaseclean_seed1317_20260509T011500Z/
results/real_rerun_20260506/v88_kw4_selected_cholesky_phaseclean_seed1318_20260509T011500Z/
```

Combined artifact：

```text
results/real_rerun_20260506/v88_kw4_selected_cholesky_p0seed5x3_t0_t4_20260509T013000Z/
```

Route excerpt：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "success_v87_formal": 0,
  "p0_stability_gate_pass": 1,
  "p0_full_protocol_complete": 1,
  "robust_timing_full_gate_pass": 1,
  "robust_timing_q90_step_ratio_vs_KB_MLP": 1.08785480477969,
  "primary_blocker": "adaptive_controller_and_cross_task_waves_not_run"
}
```

Robust timing summary：

| metric | value |
|---|---:|
| measured rows | `35` |
| timing protocol count | `5` |
| full T0-T4 complete | `1` |
| Q90 step ratio vs KB-MLP | `1.0878548048` |
| max step ratio vs KB-MLP | `1.2920914471` |
| timing gate pass rate | `1.0000000000` |
| robust timing full gate pass | `1` |

Per-protocol ratio range：

| protocol | rows | min ratio | max ratio | Q90 ratio |
|---|---:|---:|---:|---:|
| T0 phase-clean | `5` | `1.0492004936` | `1.2920914471` | `1.2920914471` |
| T1 phase-clean | `5` | `0.7684094263` | `0.9234704248` | `0.9234704248` |
| T2 phase-clean | `5` | `0.8115304216` | `0.9132767091` | `0.9132767091` |
| T3 phase-clean | `5` | `0.8521796342` | `1.0465390828` | `1.0465390828` |
| T4 full-loop | `15` | `0.6961913694` | `0.8159960809` | `0.8053490315` |

判断：

1. v8.8 Wave0 的 selected-route stability/timing 子门已真实推进：P0 5x3 + T0-T4 timing 合成通过。
2. 这仍不是 v8.8 formal/broad：adaptive controller、no-manual architecture、cross-task external waves 仍未在这条链路中运行。

No-fake audit：

```text
rows_checked = 156
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| route | `95387740c2e23d24a5744fbf143eeada6b8e3b4956c16047f7e02afd63bc371d` |
| robust timing summary | `d99d94d729d5f67ed56c8467642ac564500113936a6943de5237e2fede683a89` |
| robust timing protocols | `5bc31b116314b8dd6ad95d82966fc41b433205af1b89259baf07137b766a73bb` |

## 12. 追加：P2/P3 接回 fresh P0/T0-T4 链路

第 8 节的 focused P3 Cholesky repair 不能直接写入 full route。本节先把原 `Adaptive-FT-P probe2 warm2` 接到 fresh P0/T0-T4 链路，再做 cadence repair。

### 12.1 probe2 warm2 combined route：失败

Artifact：

```text
results/real_rerun_20260506/v88_wave0_p0_timing_p2_p3_cholesky_20260509T014500Z/
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 0,
  "p2_adaptive_one_step_pass": 1,
  "p3_adaptive_multistep_pass": 0,
  "p3_checkpoint_pass_count": 2,
  "p3_checkpoint_count": 3,
  "primary_blocker": "adaptive_multistep_gate_failed"
}
```

P3 summary：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | P3 pass |
|---:|---:|---:|---:|---:|
| 20 | `+0.0005666614` | `0.8967263796` | `0.6732098980` | 1 |
| 50 | `+0.0016333659` | `0.8595013632` | `1.1009011382` | 1 |
| 240 | `+0.0004333456` | `0.8072696327` | `1.5063067416` | 0 |

判断：fresh P0/T0-T4 链路中，原 probe2/warm2 仍在 240-step system gate 上略超 `1.50`。不能四舍五入写 pass。

### 12.2 probe3 warm1 cadence diagnostic：通过

Artifact：

```text
results/real_rerun_20260506/v88_p3_ftp_probe3_warm1_cholesky_diag_20260509T020000Z/
```

P3 summary：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | events | bad step | P3 pass |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | `+0.0005000035` | `0.8969066176` | `0.4161613119` | `4.333333` | `0.000000` | 1 |
| 50 | `+0.0015666684` | `0.8536683524` | `0.7690705982` | `8.666667` | `0.041667` | 1 |
| 240 | `+0.0006000002` | `0.8049026044` | `1.2110080158` | `41.666667` | `0.144789` | 1 |

判断：`probe3 warm1` 降低 event measurement cadence，同时不丢 step20 geometry，是当前比 `probe2 warm2` 更稳的 P3 route。

### 12.3 probe3 warm1 combined route：adaptive pass

Artifact：

```text
results/real_rerun_20260506/v88_wave0_p0_timing_p2_p3_probe3_warm1_20260509T021500Z/
```

Route：

```json
{
  "route": "R4-ConfigSensitiveSuccess",
  "success_v87_stability": 1,
  "success_v87_adaptive": 1,
  "success_v87_no_manual_tuning": 0,
  "success_v87_formal": 0,
  "p2_adaptive_one_step_pass": 1,
  "p3_adaptive_multistep_pass": 1,
  "p3_checkpoint_pass_count": 3,
  "p3_checkpoint_count": 3,
  "primary_blocker": "adaptive_multistep_pass_no_manual_tuning_and_cross_task_waves_not_run"
}
```

P3 combined summary：

| checkpoint | acc delta vs fixed | curvature ratio vs fixed | step ratio vs fixed | P3 pass |
|---:|---:|---:|---:|---:|
| 20 | `+0.0005000035` | `0.8969066176` | `0.4019191767` | 1 |
| 50 | `+0.0015666684` | `0.8536683524` | `0.7498075423` | 1 |
| 240 | `+0.0006000002` | `0.8049026044` | `1.2006713931` | 1 |

No-fake audit：

```text
rows_checked = 2548
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| probe2/warm2 combined route | `c7b354f8da5e58d0bae88b313698a15b3c2db9dede55839b15015aee2a2686bc` |
| probe3/warm1 diagnostic route | `847e732635471530d867f8d597f84e1053902e3892667df0b27a252543ed2b16` |
| probe3/warm1 diagnostic P3 summary | `bed36c2ec353b553b669966e28167cef3a8b5ba1538db840c01f0474fa36a296` |
| probe3/warm1 combined route | `35e9fb93ea995ec449752ffa80d70ce7e142b12e8424208e5e00eb874aac59e1` |
| probe3/warm1 combined P3 summary | `1e5d5f668c390d72ebac31d9a79eaf2472d9a5e128b946810cbcd5389053ab4d` |

## 13. 追加：P4/P5/P6 no-manual selected stable-compute route

本节把第 12 节通过的 `probe3 warm1` adaptive route 接到 P4 no-manual architecture selection、selected task/geometry confirmation、P5 robust timing 和 P6 compute counter。

Artifact：

```text
results/real_rerun_20260506/v88_wave0_p0_timing_p2_p3_p4_p6_probe3_warm1_20260509T023000Z/
```

Route：

```json
{
  "route": "R2-StableComputeSelected",
  "success_v87_stability": 1,
  "success_v87_adaptive": 1,
  "success_v87_no_manual_tuning": 1,
  "success_v87_external_fair": 0,
  "success_v87_formal": 0,
  "p4_selected_base_candidate_id": "KW4",
  "p4_selected_hidden_dim": 28,
  "p5_robust_timing_pass": 1,
  "p5_strict_timing_pass": 1,
  "p5_time_accounting_pass": 1,
  "p6_training_compute_fair_pass": 1,
  "primary_blocker": "stable_compute_selected_cross_task_external_waves_not_run"
}
```

P4 no-manual selection：

| metric | value |
|---|---:|
| candidate count | `50` |
| protocol count | `4` |
| selected base | `KW4` |
| selected hidden | `28` |
| params ratio vs KB-MLP | `0.9395677800` |
| FLOPs ratio vs KB-MLP | `0.9363473660` |
| selected Q90 step ratio | `0.9053793795` |
| test metric used for selection | `0` |

Selected task / geometry confirmation：

| metric | value |
|---|---:|
| child route | `R1-ExternalFairFunctionalAdvantage` |
| DG functional acc | `97.1899986267` |
| delta vs KB-MLP | `+0.2799987793` |
| delta vs DG base | `+0.0200033188` |
| curvature ratio vs DG base | `0.7724828820` |
| step ratio vs KB-MLP | `0.7976326676` |
| selected confirmation pass | `1` |

P5/P6：

| stage | metric | value |
|---|---|---:|
| P5 | q90 step ratio | `0.9292634635` |
| P5 | max step ratio | `0.9426309210` |
| P5 | max unknown fraction | `0.0659449717` |
| P5 | time accounting pass | `1` |
| P6 | forward FLOPs ratio | `0.9363473660` |
| P6 | backward FLOPs estimate ratio | `0.9363473660` |
| P6 | update FLOPs estimate ratio | `0.9395677800` |
| P6 | training compute fair pass | `1` |

No-fake audit：

```text
rows_checked = 2799
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| route | `19100284337286df6dc742cc06bfe1c69aab888c8d13080cb852e0e547ef91af` |
| P3 summary | `7b31cf81eeb12b535c45ee24517d7dee18a053efaa00e28a96b21a95cceaa466` |
| selected confirmation | `c3e8f7c966a65eb4a6edcbe0a22854d6c0237481875647e3fe8a23738529fc60` |
| P4 screen | `b6701316f371cf9ecd11c619225957a035e47b547b11fc37c7475435a6b1f558` |

## 14. 当前更新结论

v8.8 仍未完成：

```text
Wave0 fresh P0 5x3 = measured
Wave0 T0-T4 robust timing = pass
P2 adaptive one-step = pass
P3 adaptive multistep = pass with probe3 warm1 cadence
P4 no-manual base/hidden selection = pass
P5/P6 selected timing/compute = pass
P7/P9 cross-task external = not_run in current chain
non-vision broad task families = not_implemented / not_run
P8-P12 robustness / continual / mechanism / boundary waves = not_run
success_v88_formal_reproduction = false
success_v88_broad_external = false
success_v88_strong = false
```

当前机制结论：

1. v8.8 已经不再卡在 selected-route stability：fresh P0 5x3 + T0-T4 timing 已过。
2. 原 `Adaptive-FT-P probe2 warm2` 在 fresh 链路里仍有 240-step timing 边界；`probe3 warm1` 是当前更稳的 adaptive cadence，3/3 checkpoint 通过。
3. no-manual selected route 也重新闭合：P4 自动选出 `KW4 hidden28`，P5/P6 均过。
4. 但 v8.8 的中心目标是 broad external generalization and mechanism closure。当前还没有执行 P7/P9 cross-task external，更没有两个以上非 vision task families，也没有 P8-P12 robustness / continual / attribution / boundary waves。
5. 因此现在最诚实的 route 是 `R2-StableComputeSelected`，不是 `R1-BroadExternalFairFunctionalAdvantage`。

最终一句话：

> v8.8 还没有完成，但本轮继续优化已把 fresh selected route 推进到 stable-compute selected：P0 5x3、T0-T4、P2/P3、P4/P5/P6 都有真实 pass artifact。下一步必须执行 P7/P9 cross-task external，并实现/运行非 vision-family broad tasks 与 P8-P12 mechanism/boundary waves；不能把当前 MNIST-side stable-compute success 扩写成 broad strong。
