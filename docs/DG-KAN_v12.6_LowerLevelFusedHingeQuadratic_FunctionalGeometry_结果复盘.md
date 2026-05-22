# DG-KAN v12.6 Lower-Level Fused Hinge/Quadratic + Functional Geometry 结果复盘

> 本复盘记录 `DG-KAN_v12.6_LowerLevelFusedHingeQuadratic_FunctionalGeometry_完整计划.md` 的真实执行与连续修复结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位成功或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R2-AUCStepAndTaskFail
latest_targeted_route = R2-AUCStepAndTaskFail
base_qualified = False
functional_open = False
best_current_candidate = B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075, protocolfix final075
best_current_artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b109_h160_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_f3_workspace_final075_protocolfix_20260520T210000Z
latest_targeted_candidate = B135b-SimpleFastTaskGeometry-h160-learnableP-localdensepairtraj-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075
latest_targeted_artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b135_h160_localdensepairtraj_absdiag050_fixedbranch_fixedgain_quad020_final075_20260520T233000Z
next_recommended_action = stop dense random pair trajectory; keep the proven F4 fixed-P workspace cost reduction, but return to B109/B124 task-stable trajectory or design a sparse/local task-stable primitive with real backward/update cost reduction
```

目标达成状态：

```text
Lower-level FHQ full-step = achieved for B109/B110/B111/B115-B119/B124/B125/B126/B129, plus B131/B133/B135 fixed-P F4 workspace paths
A4 expression = achieved for B109/B110/B119/B124/B125/B126/B129/B133/B135, plus prior eligible candidates
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
8. `B131 pairtraj` 首次验证了真正降低 fixed-P fused backward/update cost 的路径：F4 fixed-P workspace step `0.685358064760707`、memory `0.13`、backward ratio `0.4557250926585637`、update ratio `0.7320725649235357`；但 pairtraj A4 失败，不能进入 A5。
9. `B133 localdensepairtraj` 保住 A1/A4 且 F4 fixed-P workspace step `0.7942844867303934`、memory `0.13`，但 task 明显坍塌：B133b mean `-0.10026041666666667`、worst `-0.333984375`、near `0.3333333333333333`、ECE fail。
10. `B135 localdensepairtraj + fixedbranch/fixedgain + quad020` 仍未救回 task：B135b mean `-0.08550347222222222`、worst `-0.26171875`、near `0.3333333333333333`、ECE fail；说明 dense pair trajectory 是 task-hostile，不应继续小修。
11. Functional official 仍关闭；所有 Functional/Line C/control rows 只能 diagnostic。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 FHQ Triton kernel module | 将 v12.6 forward/delta/direct-grad/quad-grad/proj-grad 下沉到正式 kernel path；runner 只负责 protocol/gate/artifact。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 支持 class branch、classwise gain、absdiag、normabs、meanstat、groupabs4、logitnorm | 均为数据集无关 primitive；没有按 dataset/label 分支，没有改 loss。 |
| `experiments/run_v126_lowerlevel_fhq_functional_geometry.py` | 新增 v12.6 runner、AUC attribution、protocol repair | A5 只有 A4 打开后才运行；AUC 同时要求 AUC-step 与 AUC-time，不允许 timing-only pass。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B124/B125/B126 | fixedbranch/fixedgain/logitnorm 只作为 AUC 诊断，不改 gate/loss。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B127/B128/B129/B130 | 按推荐方向继续设计 single-kernel-friendly task trajectory primitive；只改数据集无关 direct tail，不改 loss/gate。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 ABS_QUAD、ABS_MIX_SQ、CUBIC_DIAG FHQ support | 对 direct abs/square/mixed/cubic tail 下沉到 FHQ forward/direct-grad；失败只写 R0/R2，不冒充 success。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 pairtraj / localdensepairtraj 初始化与 B131-B135 | 只改变数据集无关的固定/可学习投影轨迹；用于检验 single-kernel-friendly pair trajectory 是否能降 AUC 与 backward/update cost。 |
| `experiments/run_v126_lowerlevel_fhq_functional_geometry.py` | 新增 F4 fixed-P workspace full-step 与 task path | 对 fixed-P candidate 复用 workspace、跳过 `proj_grad`，真实测量 lower backward/update cost；不把 fixed-P timing pass 写成 base success。 |

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

新增 B131 smoke：

```text
artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b131_pairtraj_smoke_20260520T230000Z
F1/F4 fixed-P path correctness and full-step smoke pass on tiny setting
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
| B131 pairtraj | `v126_b131_h160_pairtraj_absdiag050_f1_f4_final075_20260520T231000Z` | `R4` | F4 fixed-P workspace step `0.685358064760707`，backward/update cost 明显下降；但 A4 expression 失败。 |
| B133 localdensepairtraj | `v126_b133_h160_localdensepairtraj_absdiag050_f1_f4_final075_20260520T232000Z` | `R2` | F4 fixed-P workspace 过 A1 且 A4 过，但 A5 task 坍塌，ECE/AUC 失败。 |
| B135 localdensepairtraj fixedbranch/fixedgain | `v126_b135_h160_localdensepairtraj_absdiag050_fixedbranch_fixedgain_quad020_final075_20260520T233000Z` | `R2` | 用 B124 稳定 branch/gain 仍未救回 task；dense pair trajectory 方向被排除。 |

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
| B131 pairtraj | F4 fixed-P workspace | `0.974534355477866 / 0.5966666666666667 / 1` | `0.9621187912614572 / 0.13 / 1` | F4 fixed-P workspace step `0.685358064760707`，backward `0.4557250926585637`，update `0.7320725649235357`；但 A4 fail。 |
| B133 localdensepairtraj | F4 fixed-P workspace | `0.8480923833682528 / 0.5966666666666667 / 1` | `1.0182833047747049 / 0.13 / 1` | F4 fixed-P workspace step `0.7942844867303934`，backward `0.5679411197294583`，update `0.8469889615497344`；A4 pass，A5 fail。 |
| B135 localdensepairtraj fixedbranch/fixedgain | F4 fixed-P workspace | `1.1812666505763019 / 0.5957142857142858 / 0` | `0.9418252966770336 / 0.12904761904761905 / 1` | F4 fixed-P workspace step `0.9283274262455713`，memory `0.12904761904761905`；A4 pass，A5 fail。 |

## 4. A4 Expression

| candidate | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | frozen key R2 | A4 pass |
|---|---:|---:|---:|---:|---|---:|
| B109 h160 quad030 | recorded pass | recorded pass | recorded pass | recorded pass | recorded pass | `1` |
| B110 h160 quad020 protocolfix | recorded pass | recorded pass | recorded pass | recorded pass | recorded pass | `1` |
| B119 h128 normabs+meanstat | `-0.0091081261634826` | `-0.0006972551345825` | `0.0039785504341125` | `-0.0005893111228942` | E1/E6/E8 `0.9655158519744872 / 0.9844241142272948 / 0.9559640884399414` | `1` |
| B126 logitnorm150 | E1 delta `-8.595628798007965` | recorded | recorded | recorded | E1 frozen `0.9800488352775574` | `1` |
| B129 sqdiag025 | `-0.00956791639328003` | `0.00020140409469604492` | `0.0037595629692077637` | `0.0018149018287658691` | E1/E6/E8 `0.9741652607917786 / 0.9666521549224854 / 0.9995290637016296` | `1` |
| B131b pairtraj learnableP | `-0.011929810047149658` | recorded fail | recorded fail | recorded fail | E1/E6/E8 `-0.12065815925598145 / -0.008309125900268555 / 0.09326905012130737` | `0` |
| B131a pairtraj fixedP | `-1.078303873538971` | recorded fail | `-0.9032763242721558` | `-0.8477475643157959` | poor frozen key R2 | `0` |
| B133b localdensepairtraj learnableP | `-0.011291027069091797` | recorded pass | recorded pass | recorded pass | E1/E6/E8 `0.9800491333007812 / 0.9992880821228027 / 0.9989565014839172` | `1` |
| B133a localdensepairtraj fixedP | `-0.012483060359954834` | recorded pass | `-0.007696032524108887` | `-0.0048629045486450195` | E1/E6/E8 `0.9800477623939514 / 0.9992897510528564 / 0.9989572167396545` | `1` |
| B135b localdensepairtraj fixedbranch learnableP | `-0.011833131313323975` | recorded pass | `0.0031403303146362305` | `0.0022618770599365234` | E1/E6/E8 `0.9800503849983215 / 0.9992890357971191 / 0.9989565014839172` | `1` |
| B135a localdensepairtraj fixedbranch fixedP | `-0.018288731575012207` | recorded pass | `-0.02287536859512329` | `-0.015675604343414307` | E1/E6/E8 `0.9800494909286499 / 0.9992887377738953 / 0.9989572167396545` | `1` |
| B127/B128/B130 | not_run | not_run | not_run | not_run | not_run | `0` |

解释：B131 证明 sparse pair trajectory 虽然非常便宜，但表达不足；B133/B135 证明 local+dense pair trajectory 可通过 frozen expression gate，但这不等于 task/base success。B130 未过 official full-step，所以 A4 合法关闭。

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
| B131 pairtraj | not_run | not_run | not_run | not_run | not_run | not_run | `0` | A4 未开，A5 合法关闭。 |
| B133b localdensepairtraj learnableP | `-0.10026041666666667` | `-0.333984375` | `0.3333333333333333` | `0` | `0` | `0` | `0` | A1/A4 打开但 task 坍塌。 |
| B133a localdensepairtraj fixedP | `-0.109375` | `-0.154296875` | `0.0` | `0` | `0` | `0` | `0` | 固定 P 更差。 |
| B135b localdensepairtraj fixedbranch learnableP | `-0.08550347222222222` | `-0.26171875` | `0.3333333333333333` | `0` | `0` | `0` | `0` | B124 stable branch/gain 仍未救回 task。 |
| B135a localdensepairtraj fixedbranch fixedP | `-0.12109375` | `-0.16015625` | `0.0` | `0` | `0` | `0` | `0` | 固定 P task-hostile。 |

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

解释：B129 改善 MNIST/KMNIST 的 AUC，但 Fashion-MNIST seed 0/1/2 全部超过 `1.05`，并且 near 降到 `0.7777777777777778`。B133/B135 则说明 dense pair trajectory 即使能过 A4，也会明显伤害真实 task trajectory；它不能替代 B109。

## 6. Line C / Functional

Line C 与 Functional 继续只作为 diagnostic。Base 未过 A5，`functional_open = false`。

最新 B135 artifact 的 route 为 R2；A4 打开但 A5 未过，Functional official 仍未开。此前 B109/B129/B133 的 diagnostic/control rows 也不能越过 base gate 写成 official functional success。

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
| B131 pairtraj | `209` | `0 / 0 / 0` |
| B133 localdensepairtraj | `469` | `0 / 0 / 0` |
| B135 localdensepairtraj fixedbranch/fixedgain | `462` | `0 / 0 / 0` |

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
| B131 pairtraj | `c1610d7b9768` | `0de181caf7ab` | `536a63c309f8` |
| B133 localdensepairtraj | `5f65fc2d06c2` | `b05582fea4ab` | `84a5df32b3d9` |
| B135 localdensepairtraj fixedbranch/fixedgain | `55835607c435` | `49a4fbea8183` | `f1567e1d32bf` |

## 8. 最终分析结论

```text
1. v12.6 已完成 lower-level FHQ 主线的多轮真实推进，但 base 仍未完成。
2. B109 protocolfix final075 是当前 best：A1/A4 通过，accuracy/worst/near/ECE 全过，只剩 strict steady AUC-step/time。
3. B110 quad020、B124/B125 fixedbranch/fixedgain、B126 logitnorm150 都未修复 Fashion late NLL/AUC。
4. B119-B122 排除了 normabs/meanstat/groupabs4 当前小网格。
5. B127/B128 排除了 direct abs+square tail 与 mixed abs-square tail：correctness pass，但 full-step 回到 R0。
6. B129 sqdiag025 证明 square-energy tail 可重新打开 A1/A4，但 task/near/AUC 不优于 B109，所以不能作为 base。
7. B130 cubicdiag025 排除了 signed cubic direct tail：F3/F1 接近但仍未过 official full-step，A4/A5 合法关闭。
8. B131 pairtraj 证明 F4 fixed-P workspace 能实质降低 full-step/backward/update cost：step 到 0.685、backward 到 0.456、update 到 0.732；但 A4 失败，不能作为 base 候选。
9. B133 localdensepairtraj 证明 local+dense pair trajectory 能过 A1/A4，但 task 明显坍塌；B135 加 fixedbranch/fixedgain 与 quad020 仍未修复，说明 dense random pair trajectory 方向应停止。
10. Functional official 仍关闭；base 未过 A5 前所有 functional/control rows 都只能 diagnostic。
11. 下一步不应继续 LR/gain/logitnorm/标量、pooled feature、direct-tail arithmetic、energy-tail、signed-polynomial tail 或 dense pair trajectory 小修。更合理方向是：
   a. 保留 F4 fixed-P workspace 这种真实 cost reduction 路径，但把 trajectory 回到 B109/B124 这类 task-stable 结构；
   b. 设计更稀疏、更局部、可单-kernel化且 task-stable 的 trajectory primitive，先过 A1/A4 再进 A5；
   c. 若继续 B109 h160，重点应是真正降低 learnable-P backward/update cost，例如减少 `proj_grad` / 参数更新负担，而不是 optimizer wrapper 或 post-hoc calibration。
```
