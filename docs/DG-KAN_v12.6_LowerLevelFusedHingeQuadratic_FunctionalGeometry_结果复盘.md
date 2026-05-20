# DG-KAN v12.6 Lower-Level Fused Hinge/Quadratic + Functional Geometry 结果复盘

> 本复盘记录 `DG-KAN_v12.6_LowerLevelFusedHingeQuadratic_FunctionalGeometry_完整计划.md` 的真实执行与连续修复结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位成功或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R2-AUCStepAndTaskFail
latest_targeted_route = R0-FusedBackwardFullStepNotOfficial
base_qualified = False
functional_open = False
best_current_candidate = B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075, protocolfix final075
best_current_artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b109_h160_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_protocolfix_20260520T210000Z
latest_targeted_candidate = B130b-SimpleFastTaskGeometry-h160-learnableP-cubicdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075
latest_targeted_artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b130_h160_cubicdiag025_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_20260520T223000Z
next_recommended_action = stop LR/gain/logitnorm/scalar/pooling/direct-tail/energy-tail/signed-polynomial micro-repairs; design a genuinely new single-kernel-friendly trajectory primitive or lower the actual fused backward/update cost without lowering gates
```

目标达成状态：

```text
Lower-level FHQ full-step = achieved for B109/B110/B111/B115-B119/B124/B125/B126/B129 exact or fixed-P paths
A4 expression = achieved for B109/B110/B119/B124/B125/B126/B129, plus prior eligible candidates
A5 task/base = not achieved
Functional official = not opened
Line C = diagnostic only
no fake/proxy/cpu = pass in all listed artifacts
```

核心结论：

1. v12.6 不是完成态。Lower-level FHQ 真实打开了多条 A1/A4 路径，但 A5 AUC-step/time 仍是 hard blocker。
2. `B109 protocolfix final075` 仍是当前 best：F3 step `0.9659355274876386`、memory `0.5966666666666667`、A4 pass、task mean `0.016927083333333332`、worst `0.00390625`、near `1.0`、ECE ok；但 Fashion-MNIST seed 1/2 strict steady AUC-step/time fail。
3. `B110 quad020`、`B124/B125 fixedbranch+fixedgain`、`B126 logitnorm150` 都不能修复 Fashion late NLL/AUC；B126 虽然压低 CEp99，但 ECE/AUC 明显恶化。
4. `B119-B122` 排除了 normabs+meanstat、signed mean only、groupabs4 当前实现：要么 task/AUC 不优于 B109，要么 full-step 未开。
5. `B127 absquad` 与 `B128 absmixsq` correctness/smoke 通过，但 direct tail arithmetic 增加后 full-step 回到 R0，A4/A5 合法关闭。
6. `B129 sqdiag025` 重新打开 A1/A4：F3 step `1.0632296234535383`、memory `0.5966666666666667`、A4 pass；但 task mean `0.008680555555555556`、worst `-0.0078125`、near `0.7777777777777778`，Fashion AUC-step/time 仍失败，不优于 B109。
7. `B130 cubicdiag025` signed polynomial tail smoke/correctness 通过，但正式 run 停在 R0：F3 step `1.2481101999833282`、F1 step `1.1436047527375663` 均未过 official full-step，A4/A5 合法关闭。
8. Functional official 仍关闭；所有 Functional/Line C/control rows 只能 diagnostic。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 FHQ Triton kernel module | 将 v12.6 forward/delta/direct-grad/quad-grad/proj-grad 下沉到正式 kernel path；runner 只负责 protocol/gate/artifact。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 支持 class branch、classwise gain、absdiag、normabs、meanstat、groupabs4、logitnorm | 均为数据集无关 primitive；没有按 dataset/label 分支，没有改 loss。 |
| `experiments/run_v126_lowerlevel_fhq_functional_geometry.py` | 新增 v12.6 runner、AUC attribution、protocol repair | A5 只有 A4 打开后才运行；AUC 同时要求 AUC-step 与 AUC-time，不允许 timing-only pass。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B124/B125/B126 | fixedbranch/fixedgain/logitnorm 只作为 AUC 诊断，不改 gate/loss。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B127/B128/B129/B130 | 按推荐方向继续设计 single-kernel-friendly task trajectory primitive；只改数据集无关 direct tail，不改 loss/gate。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 ABS_QUAD、ABS_MIX_SQ、CUBIC_DIAG FHQ support | 对 direct abs/square/mixed/cubic tail 下沉到 FHQ forward/direct-grad；失败只写 R0/R2，不冒充 success。 |

本轮没有做：

```text
1. 没有调低 A1/A4/A5/Line C/Functional gate。
2. 没有按 MNIST/Fashion-MNIST/KMNIST 名称分支。
3. 没有 teacher / distillation / modified loss / sampler / class weight。
4. 没有把 diagnostic functional 写成 official success。
5. 没有把 R0/R1/R2 exploratory rows 当成 base success。
```

编译与 smoke 验证：

```bash
python -m py_compile dgkan/models/fc_purekan_primitives.py dgkan/kernels/fused_hinge_quadratic.py experiments/run_v126_lowerlevel_fhq_functional_geometry.py
```

新增 B130 smoke：

```text
B130b direct_readout = (64, 10), supported_simple = true, forward max_abs = 4.76837158203125e-07
B130a direct_readout = (64, 10), supported_simple = true, forward max_abs = 3.5762786865234375e-07
```

## 2. 执行链

| run | artifact | route | 结论 |
|---|---|---|---|
| B109 h160 quad030 protocolfix final075 | `v126_b109_h160_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_protocolfix_20260520T210000Z` | `R2` | 当前 best；accuracy/worst/near/ECE 全过，只剩 strict steady AUC-step/time fail。 |
| B110 h160 quad020 protocolfix | `v126_b110_h160_absdiag050_classbranch_classgain_identitytailquad020_hingeamp025_f3_workspace_final075_protocolfix_20260520T212000Z` | `R2` | Full-step 很快，A4 过；Fashion seed1 AUC 仍高。 |
| B119 h128 normabs+meanstat | `v126_b119_h128_absdiag050_normabs050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_20260520T201000Z` | `R2` | A1/A4 打开，task/AUC 失败，不优于 B109。 |
| B122 groupabs4 gradopt | `v126_b122_h128_absdiag050_groupabs4_050_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_gradopt_20260520T205000Z` | `R0` | 专用 groupabs grad kernel correctness pass，但仍未过 full-step。 |
| B124 fixedbranch/fixedgain temp075 | `v126_b124_h160_absdiag050_fixedbranch_fixedgain_identitytailquad020_hingeamp025_f3_workspace_final075_20260520T213000Z` | `R2` | 固定 branch/gain 后 A4 过，但 AUC 未修。 |
| B125 fixedbranch/fixedgain temp050 | `v126_b125_h160_absdiag050_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp050_f2_final075_20260520T214000Z` | `R2` | 低 gain 降 CEp99 但伤 worst/near，Fashion AUC 仍 fail。 |
| B126 logitnorm150 | `v126_b126_h160_absdiag050_logitnorm150_fixedbranch_fixedgain_identitytailquad020_hingeamp025_f2_final075_20260520T215000Z` | `R2` | FHQ logitnorm gradient correctness pass，F2 过 efficiency；ECE/AUC 失败。 |
| B127 absquad050025 | `v126_b127_h160_absquad050025_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_20260520T220000Z` | `R0` | abs+square dual direct tail correctness pass，但 F3 step `1.980083680412736`，full-step 未开。 |
| B128 absmixsq025 | `v126_b128_h160_absmixsq025_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_20260520T221000Z` | `R0` | mixed abs/square 单 tail correctness pass，但 F3 step `1.3378124510102267`，full-step 未开。 |
| B129 sqdiag025 | `v126_b129_h160_sqdiag025_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_20260520T222000Z` | `R2` | F3 full-step 与 A4 打开，但 A5 near/AUC 失败。 |
| B130 cubicdiag025 | `v126_b130_h160_cubicdiag025_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_20260520T223000Z` | `R0` | signed cubic tail 接近但未过 full-step official，A4/A5 合法关闭。 |

## 3. Lower-Level Full-Step

| candidate | best official impl | exact F3 step / memory / pass | fixed-P F1 step / memory / pass | 判断 |
|---|---|---:|---:|---|
| B109 h160 quad030 protocolfix | F3 | `0.9659355274876386 / 0.5966666666666667 / 1` | `0.7448759253070407 / 0.13 / 1` | 当前 best。 |
| B110 h160 quad020 protocolfix | F3 | `0.7342712275546092 / 0.5966666666666667 / 1` | `0.7015653037735786 / 0.13 / 1` | A1 打开；task 不优于 B109。 |
| B119 h128 normabs+meanstat | F3 | `1.0574153123276442 / 0.5014285714285714 / 1` | `1.0309401129395066 / 0.1280952380952381 / 1` | A1 打开；task 不够。 |
| B122 h128 groupabs4 gradopt | none | `2.134409127761204 / 0.5014285714285714 / 0` | `1.9669711162922519 / 0.1280952380952381 / 0` | R0。 |
| B124 fixedbranch/fixedgain temp075 | F2/F1 only | `1.4433949663893555 / 0.5957142857142858 / 0` | `0.9569840822556168 / 0.12904761904761905 / 1` | F2 pass，F3 fail。 |
| B125 fixedbranch/fixedgain temp050 | F3 | `0.8494167185915186 / 0.5957142857142858 / 1` | `1.0241731374811742 / 0.12904761904761905 / 1` | A1 打开；task 不够。 |
| B126 logitnorm150 | F2/F1 only | `1.5670950355630384 / 0.5957142857142858 / 0` | `1.019029634791647 / 0.12904761904761905 / 1` | F2 pass，F3 fail；logitnorm correctness pass。 |
| B127 absquad050025 | none | `1.980083680412736 / 0.6261904761904762 / 0` | `1.5321638803678295 / 0.1595238095238095 / 0` | dual direct tail arithmetic 太贵。 |
| B128 absmixsq025 | none | `1.3378124510102267 / 0.5966666666666667 / 0` | `1.762266117059046 / 0.13 / 0` | mixed tail 仍超 full-step gate。 |
| B129 sqdiag025 | F3 | `1.0632296234535383 / 0.5966666666666667 / 1` | `1.011923118517769 / 0.13 / 1` | A1 打开；task/AUC 不够。 |
| B130 cubicdiag025 | none | `1.2481101999833282 / 0.5966666666666667 / 0` | `1.1436047527375663 / 0.13 / 0` | signed cubic tail 未过 official full-step。 |

## 4. A4 Expression

| candidate | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | frozen key R2 | A4 pass |
|---|---:|---:|---:|---:|---|---:|
| B109 h160 quad030 | recorded pass | recorded pass | recorded pass | recorded pass | recorded pass | `1` |
| B110 h160 quad020 protocolfix | recorded pass | recorded pass | recorded pass | recorded pass | recorded pass | `1` |
| B119 h128 normabs+meanstat | `-0.0091081261634826` | `-0.0006972551345825` | `0.0039785504341125` | `-0.0005893111228942` | E1/E6/E8 `0.9655158519744872 / 0.9844241142272948 / 0.9559640884399414` | `1` |
| B126 logitnorm150 | E1 delta `-8.595628798007965` | recorded | recorded | recorded | E1 frozen `0.9800488352775574` | `1` |
| B129 sqdiag025 | `-0.00956791639328003` | `0.00020140409469604492` | `0.0037595629692077637` | `0.0018149018287658691` | E1/E6/E8 `0.9741652607917786 / 0.9666521549224854 / 0.9995290637016296` | `1` |
| B127/B128/B130 | not_run | not_run | not_run | not_run | not_run | `0` |

解释：B129 证明 sqdiag energy tail 可保住 A4；B130 未过 official full-step，所以 A4 合法关闭。

## 5. A5 Task / AUC

| run | mean delta | worst delta | near pass | ECE ok | AUC-step ok | AUC-time ok | A5 pass | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B109 h160 quad030 protocolfix | `0.016927083333333332` | `0.00390625` | `1.0` | `1` | `0` | `0` | `0` | 当前 best；只剩 AUC hard gate。 |
| B110 h160 quad020 protocolfix | `0.014973958333333334` | `-0.001953125` | `1.0` | `1` | `0` | `0` | `0` | Fashion seed1 AUC 仍高。 |
| B119 h128 normabs+meanstat | `0.0008680555555555` | `-0.009765625` | `0.8888888888888888` | `1` | `0` | `0` | `0` | AUC 与 mean 都不够。 |
| B124 fixedbranch/fixedgain temp075 | `0.014756944444444444` | `-0.00390625` | `1.0` | `1` | `0` | `0` | `0` | 固定 branch/gain 不修 AUC。 |
| B125 fixedbranch/fixedgain temp050 | `0.009114583333333334` | `-0.015625` | `0.8888888888888888` | `1` | `0` | `0` | `0` | CEp99 降低但 task/near 变差。 |
| B126 logitnorm150 | `0.011284722222222222` | `-0.005859375` | `0.8888888888888888` | `0` | `0` | `0` | `0` | logitnorm 破坏 ECE/AUC。 |
| B129 sqdiag025 | `0.008680555555555556` | `-0.0078125` | `0.7777777777777778` | `1` | `0` | `0` | `0` | A1/A4 打开，但 Fashion AUC 与 near 不够。 |
| B130 cubicdiag025 | not_run | not_run | not_run | not_run | not_run | not_run | `0` | A4 未开，A5 合法关闭。 |

B109 protocolfix best AUC blockers:

| dataset | seed | AUC-step ratio | AUC-time ratio |
|---|---:|---:|---:|
| Fashion-MNIST | 0 | `1.0400800475649032` | `1.0400800475649032` |
| Fashion-MNIST | 1 | `1.2992527306295352` | `1.2992527306295352` |
| Fashion-MNIST | 2 | `1.2565948266110967` | `1.2565948266110967` |

B129 sqdiag025 per-row:

| dataset | seed | val acc delta | AUC-step ratio | AUC-time ratio | ECE delta |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `0.033203125` | `0.730567885467009` | `0.730567885467009` | `-0.019318275153636932` |
| MNIST | 1 | `0.013671875` | `0.9367402444118349` | `0.9367402444118349` | `-0.003755733370780945` |
| MNIST | 2 | `0.001953125` | `0.9146630271806361` | `0.9146630271806361` | `-0.005947045981884003` |
| Fashion-MNIST | 0 | `0.00390625` | `1.1682432188964107` | `1.1682432188964107` | `-0.005720645189285278` |
| Fashion-MNIST | 1 | `-0.005859375` | `1.216143071215535` | `1.216143071215535` | `0.0005218088626861572` |
| Fashion-MNIST | 2 | `-0.0078125` | `1.1999663195537906` | `1.1999663195537906` | `-0.007063955068588257` |
| KMNIST | 0 | `0.00390625` | `0.9948090271328497` | `0.9948090271328497` | `-0.017073914408683777` |
| KMNIST | 1 | `0.025390625` | `0.8880654527755425` | `0.8880654527755425` | `-0.021690890192985535` |
| KMNIST | 2 | `0.009765625` | `0.9873245825771858` | `0.9873245825771858` | `-0.018838301301002502` |

解释：B129 改善 MNIST/KMNIST 的 AUC，但 Fashion-MNIST seed 0/1/2 全部超过 `1.05`，并且 near 降到 `0.7777777777777778`。它不能替代 B109。

## 6. Line C / Functional

Line C 与 Functional 继续只作为 diagnostic。Base 未过 A5，`functional_open = false`。

最新 B130 artifact 的 route 为 R0；A4/A5 未开，Functional official 也未开。此前 B109/B129 的 diagnostic/control rows 也不能越过 base gate 写成 official functional success。

## 7. No-Fake / Hash

No-fake summary：

| artifact | rows checked | fake/proxy/cpu |
|---|---:|---|
| B119 | `466` | `0 / 0 / 0` |
| B122 gradopt | `117` | `0 / 0 / 0` |
| B126 logitnorm150 | recorded pass | `0 / 0 / 0` |
| B127 absquad050025 | `117` | `0 / 0 / 0` |
| B128 absmixsq025 | `117` | `0 / 0 / 0` |
| B129 sqdiag025 | `320` | `0 / 0 / 0` |
| B130 cubicdiag025 | `117` | `0 / 0 / 0` |

Selected SHA256 prefixes:

| run | full-step hash | task/triage hash | route hash |
|---|---|---|---|
| B109 protocolfix | `9999efd39fa0` | `e4957ecd598b` | `f69626883a77` |
| B110 protocolfix | `f69da16137af` | `ef08265eb967` | `f1567e1d32bf` |
| B119 | `23584cb9b5c2` | `7ac91217bd82` | `4e32ade2d2b2` |
| B124 fixedbranch/fixedgain | `21e2415f67d9` | `4f47109c163a` | `4e32ade2d2b2` |
| B125 fixedbranch/fixedgain temp050 | `5531ca251392` | `ba6e25939997` | `f1567e1d32bf` |
| B126 logitnorm150 | `6e46b1a7a92f` | `536694f0db36` | `4e32ade2d2b2` |
| B127 absquad050025 | `267d04d827af` | `0de181caf7ab` | `3c8f8d9bc8d3` |
| B128 absmixsq025 | `6250403d76ec` | `0de181caf7ab` | `3c8f8d9bc8d3` |
| B129 sqdiag025 | `5ef83ec4bbcc` | `320b5a37d5b6` | `bbdb7909c484` |
| B130 cubicdiag025 | `f766a285389a` | `0de181caf7ab` | `3c8f8d9bc8d3` |

## 8. 最终分析结论

```text
1. v12.6 已完成 lower-level FHQ 主线的多轮真实推进，但 base 仍未完成。
2. B109 protocolfix final075 是当前 best：A1/A4 通过，accuracy/worst/near/ECE 全过，只剩 strict steady AUC-step/time。
3. B110 quad020、B124/B125 fixedbranch/fixedgain、B126 logitnorm150 都未修复 Fashion late NLL/AUC。
4. B119-B122 排除了 normabs/meanstat/groupabs4 当前小网格。
5. B127/B128 排除了 direct abs+square tail 与 mixed abs-square tail：correctness pass，但 full-step 回到 R0。
6. B129 sqdiag025 证明 square-energy tail 可重新打开 A1/A4，但 task/near/AUC 不优于 B109，所以不能作为 base。
7. B130 cubicdiag025 排除了 signed cubic direct tail：F3/F1 接近但仍未过 official full-step，A4/A5 合法关闭。
8. Functional official 仍关闭；base 未过 A5 前所有 functional/control rows 都只能 diagnostic。
9. 下一步不应继续 LR/gain/logitnorm/标量、pooled feature、direct-tail arithmetic、energy-tail 或 signed-polynomial tail 小修。更合理方向是：
   a. 设计更强但单-kernel友好的 trajectory primitive，目标是降低 Fashion steady NLL AUC；
   b. 继续沿 B109 h160 的 exact FHQ path 做真正 fused trajectory/backward/update 优化，但不能把 timing pass 误写成 AUC-step pass；
   c. 若继续增加 direct/task geometry，必须先证明不伤 A1/A4，且不能只改善 MNIST/KMNIST 而牺牲 Fashion steady NLL。
```
