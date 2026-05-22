# DG-KAN v12.9 B314 / Functional / Classic Family 执行复盘

> 执行日期：2026-05-22 UTC  
> 主 artifact：`results/v12_9_b314_functional_classic_family/v129_b314_reentry_20260522T131120Z`  
> 复用 runner：`experiments/run_v1283_b109_classic_family_functional_geometry.py`  
> 说明：本文件只记录本轮实际落盘结果，不继承旧报告数值，不把 diagnostic 写成 official success。

## 0. 执行与修复审计

本轮先做了 `py_compile`：

```text
python -m py_compile dgkan/models/fc_purekan_primitives.py dgkan/kernels/fused_hinge_quadratic.py experiments/run_v1283_b109_classic_family_functional_geometry.py
```

结果：通过。

第一次运行时我传入了 `--fixedp-candidate-id none`。复用的 v12.6 full-step profiler 要求 fixed-P 对照候选存在，因此在正式 profile 前触发 `AssertionError`。这不是候选实验结果，没有 task/profile 数据可用。

按 v12.9 文档的 “candidate id 是否完整” 修复方向，我没有改代码，也没有降 gate，而是把 fixed-P 对照改回真实候选：

```text
B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075
```

随后重跑成功。最终成功运行的核心设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
task_compile_warmup_steps = 12
exact_candidate = B314b
repair/control candidates = B315b, B109b
classic family selected candidates = B6r, B7kc, B2r, B3x, B4w, B5h
```

Provenance audit：`v1283_provenance_audit.csv` 共 34 行，未发现 fake/proxy/CPU offload；`v1283_route_decision.json` 中 `no_fake_provenance_pass = 1`。

## 1. Route 结论

Runner route：

```text
route = R4-B109NearOrPassClassicFamiliesKernelBlocked
base_qualified = 1
functional_open = 1
official_functional_success = 0
classic_family_pass_count = 0
```

我的 v12.9 判断更严格：

```text
B314 复现了 v12.8.3 base anchor：
  F3 efficiency pass、memory pass、10-seed A5 old gate pass、Line C nontearing pass。

但在 v12.9 文档写死的 AUC_step/time <= 1.00 strict gate 下：
  B314 有 2/30 rows 轻微超线。

因此不能写成 v12.9 hardening fully pass；
只能写成 old-gate reconfirm pass / v12.9 strict near-pass。
Functional short-run official 不应打开。
```

## 2. B314 / B315 / B109 关键数据

### 2.1 Full-step profile

| candidate | impl | step_ratio_q90 | memory_ratio_q90 | backward_ratio_q90 | update_ratio_q90 | official_efficiency_pass |
|---|---|---:|---:|---:|---:|---:|
| B314b | F2 learnableP | 0.7702714447449561 | 0.5961904761904762 | 0.515322574514827 | 0.6927603929253389 | 1 |
| B314b | F3 workspace learnableP | 0.7102266481026035 | 0.1295238095238095 | 0.5144623251311912 | 0.5656778787086075 | 1 |
| B315b | F2 learnableP | 0.7381905470930119 | 0.5961904761904762 | 0.4845515640772603 | 0.6983038711689874 | 1 |
| B315b | F3 workspace learnableP | 0.6832314410226663 | 0.1295238095238095 | 0.4908932819031727 | 0.533875707603002 | 1 |
| B109b | F3 workspace learnableP | 0.8190932773282539 | 0.5966666666666667 | not summarized here | not summarized here | 1 |

B314/B315 均满足 v12.9 efficiency/memory 要求：`step_ratio_q90 <= 0.85` 且 `memory_ratio_q90 <= 0.30`。

### 2.2 10-seed task / AUC

| candidate | mean_delta | worst_delta | near_pass_rate | ECE delta max | AUC max | old A5 pass | v12.9 strict note |
|---|---:|---:|---:|---:|---:|---:|---|
| B314b | 0.023567708333333333 | 0.00390625 | 1.0 | 0.0040039122104644775 | 1.0012168367024126 | 1 | 2 rows exceed 1.00 AUC |
| B315b | 0.0234375 | 0.001953125 | 1.0 | 0.005544498562812805 | 1.0027713714412094 | 1 | 1 row exceeds 1.00 AUC |
| B109b | 0.009440104166666666 | -0.01953125 | 0.9 | 0.02165275812149048 | 1.2511148988096217 | 0 | fails worst/ECE/AUC |

B314 v12.9 strict failures:

```text
Fashion-MNIST seed 7:
  acc_delta = 0.00390625
  ECE_delta = -0.022079825401306152
  AUC_step = AUC_time = 1.000032641333901

Fashion-MNIST seed 8:
  acc_delta = 0.04296875
  ECE_delta = -0.025340557098388672
  AUC_step = AUC_time = 1.0012168367024126
```

解释：这两个 row 的 accuracy/ECE 并不坏，失败只来自 v12.9 strict `AUC <= 1.00`。按文档不能放宽 gate，所以结论是 strict near-pass，不是 hardening pass。

## 3. Line C / Functional

Line C summary：

```text
B314 CouplingR2 = 0.23202819810379627
MLP CouplingR2 = 0.21133869467893396
B314 NoiseSignalLeak = 0.035404518246650696
B314 RealSignalReservoirRatio = 0.35730239748954773
LineC nontearing pass = 1
```

Functional official re-entry：

```text
functional_reentry_measured = 1
functional_beats_controls = 0
functional_control_gap_vs_best_control = -0.9950351626588338
official_functional_success = 0
official_functional_reason = official_gate_open_but_short_run_not_requested
```

Corrected delta-score repair：

```text
best_functional_update = F7-TaskPlusSNRGeometryResidual-w025
best_control_delta_score = 0.14781692151691028
best_functional_delta_score = 0.14502893528315774
delta_score_gap_vs_best_control = -0.0027879862337525374
delta_score_beats_controls = 0
```

结论：本轮 corrected metric 下 F7 没有复现 v12.8.3 那个小正 gap，反而低于 best control。不能进入 B2 short-run，更不能声明 functional official success。

## 4. Classic Family 状态

本轮每个 family 只选一个当前代表候选，不声称完成全候选网格。

| family | candidate | status | key evidence | blocker |
|---|---|---|---|---|
| BSpline | B6r | KernelBlocked | L3 manual step 1.50460224850239, memory 1.4452380952380952 | 缺 family-specific fused backward/update kernel |
| RBF | B2r | KernelBlocked | L3 manual step 1.4674100186743748, memory 1.6147619047619048 | 缺 family-specific fused backward/update kernel |
| Wavelet | B5h | KernelBlocked | L3 manual step 2.740506078472147, memory 3.6223809523809525 | 缺 family-specific fused backward/update kernel |
| Chebyshev | B3x | KernelBlocked | fused L3 measured, grad audit passed, step 1.2723012888284448 | fused L3 efficiency gate failed |
| Fourier | B4w | ExpressionBlocked | fused L3 step 1.1282646511896621, memory 0.12428571428571429 | A4 expression gate failed |
| Rational | B7kc | TaskBlocked | fused L3 step 1.247670074730072, A4 pass | A5 failed |

Rational B7kc task summary：

```text
mean_delta = -0.014453125
worst_delta = -0.076171875
near_pass_rate = 0.6
ece_ok = 0
auc_step_ok = 0
auc_time_ok = 0
```

这支持 v12.9 文档的判断：Rational 当前不是数学梯度 correctness blocker，而是 task geometry / calibration / AUC trajectory blocker。

## 5. Artifacts / Hash

主要文件：

```text
v1283_route_decision.json
v1283_b109_fullstep_profile.csv
v1283_b109_auc_attribution.csv
v1283_b109_task_autopsy_final.csv
v1283_b109_linec_functional_reentry_summary.csv
v1283_b109_functional_delta_score_repair_summary.csv
v1283_family_status.json
v1283_family_efficiency.csv
v1283_family_expression.csv
v1283_family_task_triage.csv
v1283_family_failure_table.csv
v1283_provenance_audit.csv
v1283_hash_manifest.json
```

Selected sha256 prefixes from `v1283_hash_manifest.json`：

```text
v1283_b109_auc_attribution.csv = 4446de88805f
v1283_b109_fullstep_profile.csv = 443992713cad
v1283_b109_functional_delta_score_repair.csv = 78e305d99033
v1283_b109_linec_functional_reentry_summary.csv = a701e7fd204f
v1283_family_efficiency.csv = 92d532fe8077
v1283_family_failure_table.csv = f82da981b76e
v1283_family_status.json = 45e1451e9def
```

## 6. 最终结论

本轮真实结论：

```text
Case 2 / Case 5 混合状态：
  B314 是当前最强 efficient PureKAN base anchor，并复现 old-gate success；
  但 v12.9 strict hardening 因 Fashion-MNIST seed 7/8 的 AUC 轻微超 1.00 未完全闭合；
  functional official 仍失败；
  classic family 没有 FamilyPass，但六个 family 都有明确 blocker/status。
```

下一步不应宣布 functional success，也不应继续调温度/epoch/CE calibration。按本轮实际 blocker，优先级是：

```text
1. B314 strict AUC hardening：专查 Fashion-MNIST seed 7/8 AUC trajectory 和 timing accounting，不改 dataset-specific 参数。
2. Functional：F7 corrected score 未 beat controls，停止 B2 short-run；回到 AdamW-orthogonal / task-nonharm geometry residual 方向。
3. Rational：做 Line C task-blocked autopsy，解释 A5/ECE/AUC 失败，不做 CE tune。
4. Classic family：BSpline/RBF/Wavelet 先补 family-specific fused L3；Chebyshev 优化 fused tiling/reduction；Fourier 修 A4 expression；Rational 修 task geometry。
```

## 7. 继续推进记录：B314 strict AUC hardening

用户要求继续按推荐思路推进后，本轮只做 label-free / loss-agnostic 的 base hardening 尝试；没有改 loss、sampler、class weight、teacher/distillation、dataset-name branch，也没有使用 fake/proxy/CPU offload 结果。

### 7.1 代码修改审计

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
```

新增候选：

| candidate | 修改意图 | 审计说明 |
|---|---|---|
| B316b | B314 + `directRamp106` | 在 B314 `directRamp105` 与 B315 `directRamp110` 之间做 label-free ramp bracket；不改 task/loss/data。 |
| B317b | B314 + `directRamp108` | 同上，用更接近 B315 的 direct branch ramp 检查 Fashion seed 7/8 AUC 方向。 |
| B318b | B314 + `logitcap500` | 尝试 bounded output tail，目标是降低尾部 CE/AUC；不做 post-hoc calibration，不按 dataset 分支。 |
| B319b | B314 + `logitcap400` | 更强 bounded output tail bracket，用于验证 softcap 是否能修 AUC。 |

`python -m py_compile dgkan/models/fc_purekan_primitives.py` 已通过。

### 7.2 B316/B317 direct ramp bracket smoke

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b316_b317_fashion78_smoke_20260522T131812Z
```

协议：Fashion-MNIST seed 7/8，train/val/test = 1024/512/512，epochs = 3，batch = 128；只作为 blocker smoke，不声明 official pass。

F3 efficiency：

| candidate | F3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---:|---:|---:|
| B316 | 0.7295267615526222 | 0.1295238095238095 | 1 |
| B317 | 0.6569855382967327 | 0.1295238095238095 | 1 |
| B314 | 0.7186073680972275 | 0.1295238095238095 | 1 |

AUC / task rows：

| candidate | seed | AUC_step_ratio | acc_delta | ECE_delta |
|---|---:|---:|---:|---:|
| B316 | 7 | 0.9993137750875852 | 0.00390625 | -0.02207997441291809 |
| B316 | 8 | 1.0016297990942213 | 0.044921875 | -0.024547144770622253 |
| B317 | 7 | 0.997537660766848 | 0.001953125 | -0.022307157516479492 |
| B317 | 8 | 1.0022944296255611 | 0.041015625 | -0.024098113179206848 |
| B314 | 7 | 1.0000325940276198 | 0.00390625 | -0.022079825401306152 |
| B314 | 8 | 1.0012197641742047 | 0.04296875 | -0.025340557098388672 |
| B315 | 7 | 0.996037578595838 | 0.0 | -0.022420763969421387 |
| B315 | 8 | 1.0027713714412094 | 0.0390625 | -0.023655176162719727 |

结论：B316/B317 能修 seed 7，但 seed 8 更差；B315 seed 7 最好但 seed 8 也更差。direct-ramp midpoint 不是 full fix。

### 7.3 B318/B319 logitcap smoke

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b318_b319_fashion78_smoke_20260522T131942Z
```

F3 efficiency：

| candidate | F3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---:|---:|---:|
| B318 | 1.1424416365963221 | 0.1295238095238095 | 0 |
| B319 | 1.1860096297509386 | 0.1295238095238095 | 0 |
| B314 | 0.7324157143207228 | 0.1295238095238095 | 1 |

Task / AUC：

| candidate | mean_delta | worst_delta | near_pass_rate | max AUC_step_ratio | ECE status |
|---|---:|---:|---:|---:|---|
| B318 | -0.0322265625 | -0.052734375 | 0.0 | 1.2201825908912134 | fail |
| B319 | -0.056640625 | -0.064453125 | 0.0 | 1.3939088054113844 | fail |

Seed-level rows：

| candidate | seed | AUC_step_ratio | ECE_delta |
|---|---:|---:|---:|
| B318 | 7 | 1.2201825908912134 | 0.05810362100601196 |
| B318 | 8 | 1.1619419318027504 | 0.04122397303581238 |
| B319 | 7 | 1.3939088054113844 | 0.08002322912216187 |
| B319 | 8 | 1.3129280837279598 | 0.06298908591270447 |

结论：logitcap 路线同时破坏 efficiency、accuracy、ECE 和 AUC，不可作为 B314 strict hardening 修复。

### 7.4 既有 B309/B310 direct ramp 对照

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b309_b310_fashion78_smoke_20260522T132052Z
```

F3 efficiency：

| candidate | F3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---:|---:|---:|
| B309 | 0.7512686221334154 | 0.1295238095238095 | 1 |
| B310 | 0.7501662582932273 | 0.1295238095238095 | 1 |
| B314 | 0.6791057117744373 | 0.1295238095238095 | 1 |

AUC / task rows：

| candidate | seed | AUC_step_ratio | acc_delta | ECE_delta |
|---|---:|---:|---:|---:|
| B309 | 7 | 1.004562690803971 | 0.001953125 | -0.02224414050579071 |
| B309 | 8 | 0.9996659376947122 | 0.041015625 | -0.02676968276500702 |
| B310 | 7 | 0.9927731613680636 | 0.005859375 | -0.021925076842308044 |
| B310 | 8 | 1.0046944843220664 | 0.037109375 | -0.022504955530166626 |
| B314 | 7 | 1.0000318844334046 | 0.00390625 | -0.022079825401306152 |
| B314 | 8 | 1.0012170255715604 | 0.04296875 | -0.025340557098388672 |

结论：directRamp100/105/106/108/110/115 呈现 seed tradeoff，没有一个同时闭合 Fashion seed 7/8 strict AUC。B314 仍是最平衡 anchor，但 strict gate 未过。

### 7.5 5-epoch tail diagnostic

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b314_fashion78_epoch5_diag_20260522T132500Z
```

协议：Fashion-MNIST seed 7/8，train/val/test = 1024/512/512，epochs = 5，batch = 128；这是诊断，不替代 v12.9 A1 exact 3-epoch gate。

关键 rows：

| candidate | seed | AUC_step_ratio | acc_delta | ECE_delta | CEp99_delta | margin_p10_delta |
|---|---:|---:|---:|---:|---:|---:|
| B314 | 7 | 1.0004573256617753 | 0.009765625 | -0.020505651831626892 | 3.5457472801208496 | 0.12595601379871368 |
| B315 | 7 | 0.9990357239255758 | 0.009765625 | -0.019788801670074463 | 3.4444804191589355 | 0.1071697324514389 |
| B309 | 7 | 1.0017035978862465 | 0.009765625 | -0.02141439914703369 | 3.7048211097717285 | 0.11271654069423676 |
| B310 | 7 | 0.9970073424336847 | 0.01171875 | -0.01924397051334381 | 3.395397663116455 | 0.10578660666942596 |
| B314 | 8 | 1.0212592917128587 | 0.0546875 | -0.03165195882320404 | 3.9576339721679688 | 0.06676743924617767 |
| B315 | 8 | 1.0231748568193428 | 0.056640625 | -0.03127419948577881 | 3.8758068084716797 | 0.04307188093662262 |
| B309 | 8 | 1.0187854814313988 | 0.052734375 | -0.03227469325065613 | 4.051787376403809 | 0.07554037868976593 |
| B310 | 8 | 1.024816283879693 | 0.052734375 | -0.032193005084991455 | 3.8019189834594727 | 0.041214823722839355 |

5-epoch final val-loss / AUC 对照：

| candidate | seed | val_acc | test_acc | ECE | NLL | val_loss_auc_step |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 7 | 0.80859375 | 0.814453125 | 0.21777966618537903 | 0.592120349407196 | 0.6167685389518738 |
| B314 | 7 | 0.818359375 | 0.810546875 | 0.19727401435375214 | 0.618205189704895 | 0.6170506030321121 |
| B315 | 7 | 0.818359375 | 0.810546875 | 0.19799086451530457 | 0.6159512996673584 | 0.6161738038063049 |
| MLP | 8 | 0.779296875 | 0.7890625 | 0.23083791136741638 | 0.6477123498916626 | 0.6352634131908417 |
| B314 | 8 | 0.833984375 | 0.814453125 | 0.19918595254421234 | 0.6653310656547546 | 0.6487686634063721 |
| B315 | 8 | 0.8359375 | 0.814453125 | 0.19956371188163757 | 0.6647807955741882 | 0.6499855518341064 |

结论：5 epoch 下 seed 8 的 AUC 超标扩大到 1.0188-1.0248 区间，说明问题不是 3-epoch 偶发尾点；它更像 B314/B315/B309/B310 这一组 direct-branch ramp 共同的 tail CE trajectory blocker。由于 acc_delta 和 ECE_delta 仍为正向，不能用 loss/calibration/post-hoc 方式修；下一步需要新的 geometry primitive 或 tail-loss attribution，而不是继续扫 directRamp。

### 7.6 Seed 8 tail attribution from trace

来自同一 5-epoch artifact 的 `v1283_b109_auc_autopsy_trace.csv`。这里只抽 Fashion-MNIST seed 8 的 MLP / B314 / B315 trace，不做额外推断数据。

| candidate | epoch | val_loss | val_acc | ECE | CEp99 | margin_p10 | direct_scale | logit_gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 1 | 1.0043243169784546 | 0.67578125 | 0.3157753348350525 | 7.581507205963135 | 0.15632270276546478 |  |  |
| MLP | 2 | 0.6810089349746704 | 0.75390625 | 0.25439170002937317 | 5.611819744110107 | 0.12045329809188843 |  |  |
| MLP | 3 | 0.6077512502670288 | 0.751953125 | 0.22921395301818848 | 4.426494598388672 | 0.18053556978702545 |  |  |
| MLP | 4 | 0.6045811176300049 | 0.79296875 | 0.23369470238685608 | 5.528866291046143 | 0.15583640336990356 |  |  |
| MLP | 5 | 0.6477123498916626 | 0.779296875 | 0.23083791136741638 | 6.1494340896606445 | 0.20111174881458282 |  |  |
| B314 | 1 | 0.6780935525894165 | 0.7734375 | 0.2934289872646332 | 4.4805498123168945 | 0.095347099006176 | 1.0500000715255737 | 0.75 |
| B314 | 2 | 0.6399568915367126 | 0.787109375 | 0.2492515742778778 | 6.196670055389404 | 0.153411403298378 | 1.125 | 0.7875000238418579 |
| B314 | 3 | 0.6152130961418152 | 0.837890625 | 0.2289518564939499 | 7.501077651977539 | 0.1855127066373825 | 1.2000000476837158 | 0.824999988079071 |
| B314 | 4 | 0.6746830940246582 | 0.8203125 | 0.21867981553077698 | 9.037041664123535 | 0.21199166774749756 | 1.2750000953674316 | 0.8624999523162842 |
| B314 | 5 | 0.6652215719223022 | 0.833984375 | 0.19917108118534088 | 10.101637840270996 | 0.265616238117218 | 1.350000023841858 | 0.8999999761581421 |
| B315 | 1 | 0.678244948387146 | 0.77734375 | 0.29288363456726074 | 4.559464454650879 | 0.0972166433930397 | 1.100000023841858 | 0.75 |
| B315 | 2 | 0.6417393684387207 | 0.78515625 | 0.25010916590690613 | 6.19091796875 | 0.15742482244968414 | 1.162500023841858 | 0.7875000238418579 |
| B315 | 3 | 0.618050217628479 | 0.8359375 | 0.22859501838684082 | 7.561938285827637 | 0.18976064026355743 | 1.225000023841858 | 0.824999988079071 |
| B315 | 4 | 0.6754863858222961 | 0.822265625 | 0.21925392746925354 | 8.986384391784668 | 0.21320827305316925 | 1.287500023841858 | 0.8624999523162842 |
| B315 | 5 | 0.6646662354469299 | 0.8359375 | 0.19954673945903778 | 10.019450187683105 | 0.24222588539123535 | 1.350000023841858 | 0.8999999761581421 |

审计结论：B314/B315 相比 MLP 的 val_acc 与 ECE 更好，但 epoch 4/5 的 CEp99 明显升高；direct_scale/logit_gain 同步上升时，tail CE 恶化没有被 ECE 改善抵消。这支持“tail CE trajectory blocker”判断，而不是 timing accounting 或 candidate-id 误配。

## 8. 更新后的执行结论

当前仍未完成 v12.9 全计划：

```text
B314 old-gate = pass
B314 v12.9 strict AUC = fail / near-pass
Functional official success = fail
Classic family pass count = 0
```

已尝试且拒绝的修复方向：

```text
1. directRamp midpoint bracket: rejected, seed 7/8 tradeoff persists.
2. logitcap softcap: rejected, efficiency/task/ECE/AUC 同时恶化.
3. longer epoch diagnostic: rejected as fix, seed 8 AUC blocker expands rather than disappears.
```

因此不能宣布完成，也不能把 functional short-run 作为 official success 继续推进。下一步若继续修，应先设计不依赖 dataset/loss/post-hoc calibration 的新 tail-trajectory geometry primitive，或者做更细的 CEp99 / margin / direct-branch scale attribution；当前 simple ramp 和 softcap 两条修复线已经被实测排除。

## 9. 第二轮继续推进：global gain tail-CE repair

继续推进时，上一轮 blocker 已经定位为：

```text
B314/B315/B309/B310 的 acc_delta 与 ECE_delta 多数为正向，
但 Fashion-MNIST seed 7/8 的 tail CE / AUC trajectory 轻微超 strict 1.00。
```

因此本轮没有继续扫 directRamp，也没有再做 logitcap；改为验证一个更窄的 loss-agnostic 假设：

```text
tail CEp99 上升来自 B314 temp090 / gainramp075 的最终 global logit gain 过高；
若只降低 global gain target，而不改 loss / sampler / class weight / dataset branch / post-hoc calibration，
可能压低 AUC tail，同时保留 task geometry。
```

### 9.1 代码修改审计

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
```

修改内容：

| item | 修改 | 审计说明 |
|---|---|---|
| parser | 新增 `temp080` / `temp085` 解析 | 只影响 `SimpleFastTaskGeometry` 的 `logit_gain` 初始化；不改 loss、数据、采样或 backward 数学。 |
| B320b | B314 + `temp075` | 保留 B314 的 `directRamp105`、`signalBroad035`、`signalBlock015`、`quadReadInit125`、FHQ fused update；把 final global gain target 从 0.90 降到 0.75。 |
| B321b | B314 + `temp085` | 同上，把 final global gain target 从 0.90 降到 0.85，作为 B320/B314 之间的中间档。 |

`python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1283_b109_classic_family_functional_geometry.py` 已通过。

### 9.2 Fashion-MNIST seed 7/8 smoke

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b320_b321_gain_tail_smoke_20260522T134000Z
```

协议：Fashion-MNIST seed 7/8，train/val/test = 1024/512/512，epochs = 3，batch = 128。

F3 efficiency：

| candidate | F3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---:|---:|---:|
| B320 | 0.7033807124668994 | 0.1295238095238095 | 1 |
| B321 | 0.5221647254465875 | 0.1295238095238095 | 1 |
| B314 | 0.6637630528815911 | 0.1295238095238095 | 1 |
| B315 | 0.6872759175762451 | 0.1295238095238095 | 1 |

AUC / task rows：

| candidate | seed | AUC_step_ratio | acc_delta | ECE_delta | CEp99_delta | margin_p10_delta |
|---|---:|---:|---:|---:|---:|---:|
| B320 | 7 | 0.9809972088348075 | 0.001953125 | -0.014784306287765503 | 2.3867201805114746 | 0.018503427505493164 |
| B320 | 8 | 0.9804236183891313 | 0.037109375 | -0.01679527759552002 | 2.18984317779541 | 0.04905448853969574 |
| B321 | 7 | 0.9925994527041737 | 0.00390625 | -0.019969984889030457 | 2.8784146308898926 | 0.03653833270072937 |
| B321 | 8 | 0.9932991115028676 | 0.041015625 | -0.022248923778533936 | 2.704822063446045 | 0.05577971041202545 |
| B314 | 7 | 1.0000314586768755 | 0.00390625 | -0.022080078721046448 | 3.1145901679992676 | 0.03009529411792755 |
| B314 | 8 | 1.0012196225223438 | 0.04296875 | -0.025340378284454346 | 2.96165132522583 | 0.06124810874462128 |
| B315 | 7 | 0.9960392343156735 | 0.001953125 | -0.022349387407302856 | 2.9964027404785156 | 0.02648034691810608 |
| B315 | 8 | 1.0027715603103573 | 0.0390625 | -0.023621320724487305 | 2.863276958465576 | 0.058415547013282776 |

结论：B320/B321 在原始 failing slice 上同时闭合 strict AUC，且没有引入 fake/proxy/CPU/offload；global gain repair 是有效方向。

### 9.3 Full runner with B320 exact

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b320_full_a1_20260522T134200Z
```

该 run 包含 family / Line C / functional diagnostic 慢段。关键 route：

```text
route = R5-B109BasePassLineCGeometryCollapse
base_qualified = 1
classic_family_pass_count = 0
official_functional_success = 0
B109_LineC_nontearing_pass = 0
B109_functional_delta_score_beats_controls = 0
B109_functional_delta_score_gap_vs_best_control = -0.004680030608518404
```

Full runner 中 B320/B321 的 6-row default task summary：

| candidate | n | mean_delta | worst_delta | near_rate | max_ECE_delta | max_AUC_step |
|---|---:|---:|---:|---:|---:|---:|
| B320 | 6 | 0.03515625 | 0.0 | 1.0 | 0.02822813391685486 | 0.9785668598488355 |
| B321 | 6 | 0.03515625 | 0.0 | 1.0 | 0.017188549041748047 | 0.9853979977840139 |
| B314 | 6 | 0.03515625 | 0.0 | 1.0 | 0.012908518314361572 | 0.9916376721382214 |
| B315 | 6 | 0.034505208333333336 | 0.0 | 1.0 | 0.013318568468093872 | 0.9912936319590471 |

Line C 明细显示 full runner 的 selected candidate 为 B321，且该 run 的 MLP CouplingR2 较高：

```text
MLP CouplingR2 = 0.3356805192996951
B320 CouplingR2 = 0.2578653727254685
B321 CouplingR2 = 0.25972957551091025
B321 NoiseSignalLeak = 0.04829196631908417
```

结论：B320 exact full runner 证明了 AUC repair 方向，但不能宣布 functional official，因为该 run 的 Line C route 是 `LineCGeometryCollapse`，functional corrected score 也没有 beat controls。

### 9.4 A1 exact 10-seed protocol with B321 exact

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b321_a1_10seed_20260522T135000Z
```

协议严格对应文档 A1：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
task_compile_warmup_steps = 12
```

Provenance audit：

```text
fake_data_used_sum = 0
proxy_row_used_sum = 0
cpu_offload_used_sum = 0
dataset_name_branch_used = 0
teacher/distillation/loss/sampler/class_weight = 0
```

10-seed A1 derived strict summary：

| candidate | rows | mean_delta | worst_delta | near_rate >= -0.003 | max_ECE_delta | max_AUC_step | strict_fail_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| B320 | 30 | 0.022526041666666666 | 0.0 | 1.0 | 0.01961517333984375 | 0.9809978238164607 | 0 |
| B321 | 30 | 0.022916666666666665 | 0.00390625 | 1.0 | 0.008459791541099548 | 0.9932989226337199 | 0 |
| B314 | 30 | 0.023567708333333333 | 0.00390625 | 1.0 | 0.004003927111625671 | 1.001220519650796 | 2 |
| B315 | 30 | 0.0234375 | 0.001953125 | 1.0 | 0.005544215440750122 | 1.0027718436140791 | 1 |

Efficiency：

| candidate | impl | step_ratio_q90 | memory_ratio_q90 | backward_ratio_q90 | update_ratio_q90 | official_efficiency_pass |
|---|---|---:|---:|---:|---:|---:|
| B320 | F3 | 0.9597426809570575 | 0.1295238095238095 | 1.189502923475249 | 0.5358158059798208 | 1 |
| B321 | F3 | 1.0238514363294797 | 0.1295238095238095 | 1.2286885769470595 | 0.5640795610224558 | 1 |
| B314 | F3 | 0.7984719769352102 | 0.1295238095238095 | 0.6228414158034491 | 0.5807141321576678 | 1 |
| B315 | F3 | 0.7409724918840128 | 0.1295238095238095 | 0.5661023117032848 | 0.5669227676702185 | 1 |

审计判断：

```text
B320：A1 strict task/ECE/AUC 全部 0 fail，且 F3 step_ratio_q90 < 1.0；可作为 strict-A1 efficient anchor 候选。
B321：A1 strict task/ECE/AUC 全部 0 fail，ECE margin 更好，但本 run F3 step_ratio_q90 = 1.0238514363294797，超过文档 “at least <= 1.0” 的严格解释；只能作为 task-balanced repair candidate，不能比 B320 更适合作 strict efficiency anchor。
B314/B315：仍复现原 strict AUC failure slice。
```

该 10-seed run 的 route 仍为：

```text
route = R4-B109NearOrPassClassicFamiliesKernelBlocked
base_qualified = 1
official_functional_success = 0
classic_family_pass_count = 0
```

注意：该 run 的 Line C / functional summary 仍记录 B314 为 `candidate_id`，因此不能把它当成 B320/B321 的 official functional re-entry 证据。B320/B321 目前只证明了 A1 strict base hardening；functional official 仍未完成。

### 9.5 B320-only Line C reconfirm

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b320_linec_only_20260522T140000Z
```

目的：full B320 runner 的 Line C 选择了 B321 作为 selected candidate，因此单独强制 B320 exact，确认 B320 自身是否 Line C nontearing。

协议：MNIST/Fashion-MNIST/KMNIST seed 0/1，train/val/test = 1024/512/512，epochs = 3，batch = 128，family skipped，repair candidates empty。

Route：

```text
route = R4-B109NearOrPassClassicFamiliesKernelBlocked
B109_A5_best_candidate = B320b
B109_FHQ_efficiency_anchor_candidate = B320b
B109_LineC_CouplingR2 = 0.23002676858920135
B109_LineC_NoiseSignalLeak = 0.04764750599861145
B109_LineC_nontearing_pass = 1
B109_functional_delta_score_beats_controls = 0
B109_functional_delta_score_gap_vs_best_control = -0.003365868631783675
official_functional_success = 0
classic_family_pass_count = 0
```

B320-only 6-row task summary：

| candidate | rows | mean_delta | worst_delta | max_ECE_delta | max_AUC_step |
|---|---:|---:|---:|---:|---:|
| B320 | 6 | 0.018229166666666668 | 0.0 | 0.014345675706863403 | 0.9345003554635004 |
| B109a | 6 | -0.2060546875 | -0.28125 | 0.3950099050998688 | 3.8177034896603437 |

Line C 对照：

```text
MLP CouplingR2 = 0.21133869467893396
B320 CouplingR2 = 0.23002676858920135
B320 NoiseSignalLeak = 0.04764750599861145
```

结论：B320 自身可以同时满足当前小确认的 strict task/AUC/ECE 与 Line C nontearing；full B320 runner 的 LineCGeometryCollapse 是 selected B321 / run-slice 相关 blocker，不能简单归因到 B320。功能更新仍失败，不能打开 official functional success。

## 10. 当前最新结论

截至本复盘最新一轮，真实状态更新为：

```text
Line A / A1 strict base hardening:
  B314 原候选 strict AUC 仍 fail；
  B320/B321 global gain repair 闭合 10-seed strict AUC/ECE/task；
  B320 是当前更符合 strict efficiency 且 B320-only Line C reconfirm pass 的 anchor candidate；
  B321 是 task/ECE margin 更平衡但 F3 step_ratio_q90 略超 1.0 的 repair candidate。

Line C:
  B314 原 Line C 曾 pass；
  B320 exact full runner 中 selected B321 出现 LineCGeometryCollapse；
  B320-only reconfirm pass；
  B321 / selected-candidate Line C 仍需独立澄清。

Functional:
  official_functional_success = 0；
  full B320 runner 中 corrected delta-score gap = -0.004680030608518404；
  B320-only reconfirm 中 corrected delta-score gap = -0.003365868631783675；
  10-seed B321 run 的 functional summary 不是 B321 official short-run 证据。

Classic family:
  pass count = 0；
  仍是 KernelBlocked / ExpressionBlocked / Rational L3 efficiency blocker。
```

所以，本计划仍未完成。已经完成的是一个实质性子目标：找到 B320/B321 这条 non-fake、loss-agnostic 的 strict A1 AUC 修复线，并且 B320-only Line C 小确认通过。下一步不能马上宣称 functional success；应把 B320 作为新的 strict-A1 anchor，继续修 functional corrected-score 不能 beat controls 的问题，同时 classic family 仍需按各自 blocker 推进。

## 11. Functional B1 继续推进记录：residual bracket / branch damping

### 11.1 代码修改审计

本轮在 `experiments/run_v1283_b109_classic_family_functional_geometry.py` 中继续推进文档推荐的 functional 修复方向，修改范围仅限 functional delta-score repair 的候选方向：

```text
新增 F7 / F7n：Task +/- SNRGeometryResidual，权重 0.005/0.010/0.015/0.025/0.050 或负向 bracket
新增 F8 / F8n：Task +/- OrthogonalGeometryResidual，权重 0.010/0.025
保留 F9：GeometryResidualOnlyDeltaScore
新增 F10：TaskPlusBranchScaleDamping，权重 0.005/0.010
新增 F10d/F10q：direct/quad branch scale damping
```

新增 `_branch_scale_damping_delta(...)` 只在已有 `branch_scale` 参数方向上构造 loss-agnostic damping delta；未改 loss、sampler、class weight、teacher、dataset branch，也未引入 fake/proxy/CPU offload。

### 11.2 B320 residual sign/weight bracket

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b320_functional_bracket_20260522T141000Z
```

Summary：

```text
base_qualified = 1
best_functional_update = F8n-TaskMinusOrthogonalGeometryResidual-w010
delta_score_beats_controls = 0
delta_score_gap_vs_best_control = -0.00019627973401759036
official_functional_success = 0
```

该 run 中 B320 的 best control 仍是 `C0/C3 TaskOnly/AdamWParallelDirection`，delta_score = `0.11649032363935918`；best functional 为 `F8n-TaskMinusOrthogonalGeometryResidual-w010`，delta_score = `0.1162940439053416`。相比 B320-only reconfirm 的 gap `-0.003365868631783675` 明显缩小，但仍未 beat strong control。

结论：residual 的 sign/weight bracket 有效缩小 functional gap，但不能宣称 B1 通过，也不能进入 B2 扩种子。

### 11.3 B320 branch-scale damping bracket

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b320_functional_branchdamp_20260522T141800Z
```

Summary：

```text
base_qualified = 1
best_functional_update = F10-TaskPlusBranchScaleDamping-w005
delta_score_beats_controls = 0
delta_score_gap_vs_best_control = -0.00014220178469248612
official_functional_success = 0
```

B320 ranked rows：

| update_type | delta_score |
|---|---:|
| C0-TaskOnlyAdamW | 0.11649032363935918 |
| C3-AdamWParallelDirection | 0.11649032363935918 |
| F10-TaskPlusBranchScaleDamping-w005 | 0.1163481218546667 |
| F8n-TaskMinusOrthogonalGeometryResidual-w010 | 0.1162940439053416 |
| F7n-TaskMinusSNRGeometryResidual-w010 | 0.11629374479444543 |
| F7-TaskPlusSNRGeometryResidual-w005 | 0.11625061999137576 |

结论：branch-scale damping 是当前最接近 strong control 的 functional 方向，但 gap 仍为负，距离计划中 B1 的 `gap >= +0.005` 或 CI lower `>= 0` 标准都不满足。该结果必须记录为失败，不得写成 near-success pass。

### 11.4 Functional 当前判断

Functional 线截至本节仍未完成：

```text
official_functional_success = 0
best observed B320 functional gap = -0.00014220178469248612
best observed functional update = F10-TaskPlusBranchScaleDamping-w005
```

Residual / orthogonal residual / branch damping 都没有超过 AdamWParallel strong control。下一步如果继续 functional，应当考虑新的 label-free structural direction 或重新定义 corrected-score 的合法几何贡献度量；继续小幅扫当前 residual/damping 权重，按现有证据收益很低。

## 12. Classic family 当前复查记录

本轮没有对 classic family kernel 做新的代码修改；这里只记录 B320 full runner 中已经产生的真实 family blocker 数据，避免把 family 状态和 B320 base repair 混在一起。

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b320_full_a1_20260522T134200Z
```

L3 efficiency 代表行：

| family | candidate | impl | step_ratio_q90 | memory_ratio_q90 | fused_backward_used | official_efficiency_pass |
|---|---|---|---:|---:|---:|---:|
| Chebyshev | B3e-ChebyKAN-K3-tritonL3-matmulTile | L3-analytic-manual-ce-torch-reduction | 1.128219018456115 | 1.0019047619047619 | 1 | 1 |
| Chebyshev | B3m-ChebyKAN-K3-h112-tritonL3-gradbuf | L3-analytic-manual-ce-torch-reduction | 1.1606733118718464 | 0.9928571428571429 | 1 | 1 |
| Fourier | B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile | L3-analytic-manual-ce-torch-reduction | 1.2249822836353161 | 0.9990476190476191 | 1 | 1 |
| Rational | B7io-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-manualAdamW-L3 | L3-analytic-manual-ce-torch-reduction | 1.3146997137559184 | 0.7714285714285715 | 1 | 0 |

Expression summary 代表 blocker：

| candidate | family | expression status |
|---|---|---|
| B3e-ChebyKAN-K3-tritonL3-matmulTile | Chebyshev | `A4_expression_gate_fail_after_l3_efficiency` |
| B3m-ChebyKAN-K3-h112-tritonL3-gradbuf | Chebyshev | `A4_expression_gate_fail_after_l3_efficiency` |
| B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile | Fourier | `A4_expression_gate_fail_after_l3_efficiency` |
| B7io-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-manualAdamW-L3 | Rational | efficiency gate fail，A4 legally closed |

Classic family 当前结论：

```text
classic_family_pass_count = 0
Chebyshev/Fourier: 已有 L3 fused efficiency 候选，但 A4 expression gate fail
Rational: L3 memory 好，但 step_ratio_q90 = 1.3146997137559184，official_efficiency_pass = 0
BSpline/RBF/Wavelet: 仍未取得 official L2/L3 full-step gate，因此 A4 legally closed
```

因此 classic family 线也仍未完成。下一步更合理的推进点不是重复 B320 base run，而是：

```text
1. Chebyshev/Fourier：修 expression capacity，同时保持已有 L3 fused path，不得牺牲效率 gate。
2. Rational：先修 L3 step efficiency，再打开 A4 expression。
3. BSpline/RBF/Wavelet：先补合法 L2/L3 full-step gate，再谈 expression。
```

## 13. Classic Chebyshev paircrossR32 efficiency-floor 修复尝试

### 13.1 代码修改审计

本轮继续按 classic family 的推荐方向推进 Chebyshev：B3o `h112-paircrossR32` 已有真实 `cheby_k3_triton_l3_paircross_gradbuf` fused L3 kernel，但 full runner 中 step ratio `1.27630184593882` 超过 official gate `<= 1.25`，导致 A4 不能打开。为避免改 gate 或编造 pass，我只新增两个同构低 hidden 候选：

```text
B3y-ChebyKAN-K3-h96-paircrossR32-tritonL3-gradbuf
B3z-ChebyKAN-K3-h88-paircrossR32-tritonL3-gradbuf
```

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1283_b109_classic_family_functional_geometry.py
```

修改内容仅为 `PrimitiveSpec` 和 `_family_plan()` entry，复用既有 `cheby_k3_paircrossR32_triton_l3_gradbuf` kernel path；没有改 loss/data/sampler/class weight/teacher/dataset branch，也没有改 efficiency 或 expression gate。

### 13.2 Focused smoke run

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b3yz_cheby_paircross_floor_20260522T150500Z
```

协议说明：这是 focused smoke，不是 full 10-seed family official portfolio run。只测 `MNIST seed 0`，train/val/test = `512/256/256`，family candidate 限定为 B3y/B3z，`compile_family_forward=0`，用于确认低 hidden paircrossR32 是否能打开 fused L3 efficiency 与 A4 expression。

Efficiency：

| candidate | impl | step_ratio_q90 | memory_ratio_q90 | official_fused_l3_kernel | official_efficiency_pass |
|---|---|---:|---:|---:|---:|
| B3y h96 paircrossR32 | L3-analytic-manual-ce-torch-reduction | 1.1686225692756207 | 0.8523809523809524 | 1 | 1 |
| B3z h88 paircrossR32 | L3-analytic-manual-ce-torch-reduction | 1.1017130406316982 | 0.7814285714285715 | 1 | 1 |

Expression summary：

| candidate | A4_expression_pass | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | failure |
|---|---:|---:|---:|---:|---:|---|
| B3y h96 paircrossR32 | 0 | -0.0619695782661438 | -0.009791791439056396 | -0.08376055955886841 | -0.08720695972442627 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3z h88 paircrossR32 | 0 | -0.06279575824737549 | -0.011111021041870117 | -0.08307385444641113 | -0.09445077180862427 | `A4_expression_gate_fail_after_l3_efficiency` |

Family status in this focused run：

```text
Chebyshev status = ExpressionBlocked
Chebyshev L3_official_efficiency_candidate_exists = true
Chebyshev best_L3_manual_step_ratio = 1.1017130406316982
classic_family_pass_count = 0
```

Route side note：该 focused run 的 B320 route 为 `R5-B109BasePassLineCGeometryCollapse`，`B109_LineC_nontearing_pass = 0`，functional gap `-0.00011216843293071488`。由于本 run 只用 MNIST seed 0 且目的是 classic smoke，不能替代前面的 B320-only Line C reconfirm，也不能写成 functional success。

### 13.3 Classic 当前判断更新

B3y/B3z 修复了 B3o 的 immediate efficiency blocker：Chebyshev paircrossR32 现在可以在 focused smoke 中同时满足 fused L3 kernel 与 efficiency gate，并合法打开 A4。真实新 blocker 是 expression 本身，尤其 E1/E6/E8 关键项仍明显低于 MLP 或 frozen-readout要求。

因此 classic family 仍未完成：

```text
classic_family_pass_count = 0
Chebyshev paircrossR32 efficiency floor = repaired in focused smoke
Chebyshev paircrossR32 A4 expression = fail
```

下一步不应继续只降 hidden；B3z 已经更快但 expression 更弱。更合理的方向是增加表达结构而非只调宽度，例如在保持 `cheby_k3_triton_l3_paircross_gradbuf` efficiency floor 的前提下，尝试更直接覆盖 E1/E6/E8 的结构项，或者先把 A4 expression 的失败分解为 pairwise / rotated-pairwise / random-quadratic 三类缺口。

## 14. Classic Chebyshev input-cross expression 修复尝试

### 14.1 代码修改审计

本轮按第 13 节结论继续推进：不再只降 hidden，而是在已修复 L3 efficiency floor 的 `paircrossR32` Chebyshev 路线上加入轻量 input-cross 表达项，观察能否改善 E1/E6/E8 expression blocker，同时保持 official L3 efficiency gate。

新增候选：

```text
B3aa-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P8-tritonL3-gradbuf
B3ab-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P16-tritonL3-gradbuf
B3ac-ChebyKAN-K3-h80-paircrossR32-inputcrossL4P16-tritonL3-gradbuf
B3ad-ChebyKAN-K3-h72-paircrossR32-inputcrossL4P32-tritonL3-gradbuf
B3ae-ChebyKAN-K3-h64-paircrossR32-inputcrossL4P48-tritonL3-gradbuf
B3af-ChebyKAN-K3-h72-paircrossR32-inputcrossL4P24-tritonL3-gradbuf
B3ag-ChebyKAN-K3-h68-paircrossR32-inputcrossL4P24-tritonL3-gradbuf
```

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1283_b109_classic_family_functional_geometry.py
```

修改内容仅为 `PrimitiveSpec` 与 `_family_plan()` entry，复用既有 `cheby_k3_paircrossR32_inputcross_localr4_projr*_triton_l3_gradbuf` kernel path；没有改 loss/data/sampler/class weight/teacher/dataset branch，也没有改 efficiency 或 expression gate。

### 14.2 Focused smoke run 协议与 artifact

本节所有结果均为 focused smoke，不是 full 10-seed family official portfolio run。协议为 `MNIST seed 0`，train/val/test = `512/256/256`，family candidate 限定为对应 Chebyshev 新候选，`compile_family_forward=0`。用途是判断 input-cross rank/hidden bracket 是否值得进入更大 official run。

Artifacts：

```text
results/v12_9_b314_functional_classic_family/v129_b3aa_ab_ac_cheby_inputcross_20260522T151900Z
results/v12_9_b314_functional_classic_family/v129_b3ad_ae_cheby_inputcross_rank_20260522T152600Z
results/v12_9_b314_functional_classic_family/v129_b3af_ag_cheby_inputcross_p24_20260522T153100Z
```

### 14.3 Efficiency 结果

| candidate | input-cross rank | hidden | impl | step_ratio_q90 | memory_ratio_q90 | official_fused_l3_kernel | official_efficiency_pass |
|---|---:|---:|---|---:|---:|---:|---:|
| B3aa | P8 | 88 | L3-analytic-manual-ce-torch-reduction | 1.2098414206497146 | 0.7928571428571428 | 1 | 1 |
| B3ab | P16 | 88 | L3-analytic-manual-ce-torch-reduction | 1.177860108901974 | 0.7923809523809524 | 1 | 1 |
| B3ac | P16 | 80 | L3-analytic-manual-ce-torch-reduction | 1.057740728128979 | 0.7885714285714286 | 1 | 1 |
| B3ad | P32 | 72 | L3-analytic-manual-ce-torch-reduction | 1.2671274405092494 | 0.7842857142857143 | 1 | 0 |
| B3ae | P48 | 64 | L3-analytic-manual-ce-torch-reduction | 1.3935882428333404 | 0.7795238095238095 | 1 | 0 |
| B3af | P24 | 72 | L3-analytic-manual-ce-torch-reduction | 1.2281573722281418 | 0.7842857142857143 | 1 | 1 |
| B3ag | P24 | 68 | L3-analytic-manual-ce-torch-reduction | 1.2268657172801187 | 0.7823809523809524 | 1 | 1 |

P8/P16/P24 能保住 L3 efficiency gate；P32/P48 即使用更低 hidden，也超过 `step_ratio_q90 <= 1.25`，因此 A4 expression legally closed，不能写成 expression pass/fail。

### 14.4 Expression 结果

| candidate | A4_expression_pass | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | failure |
|---|---:|---:|---:|---:|---:|---|
| B3aa P8 h88 | 0 | -0.050881922245025635 | -0.016064703464508057 | -0.08715766668319702 | -0.09538042545318604 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3ab P16 h88 | 0 | -0.0640605092048645 | -0.010161757469177246 | -0.08088749647140503 | -0.09559965133666992 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3ac P16 h80 | 0 | -0.05571103096008301 | -0.009836554527282715 | -0.08431625366210938 | -0.0922812819480896 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3ad P32 h72 | not run | n/a | n/a | n/a | n/a | L3 efficiency gate fail; A4 legally closed |
| B3ae P48 h64 | not run | n/a | n/a | n/a | n/a | L3 efficiency gate fail; A4 legally closed |
| B3af P24 h72 | 0 | -0.06791174411773682 | -0.011188387870788574 | -0.08964020013809204 | -0.08842742443084717 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3ag P24 h68 | 0 | -0.060373544692993164 | -0.015921950340270996 | -0.08197569847106934 | -0.097958505153656 | `A4_expression_gate_fail_after_l3_efficiency` |

Family status：

```text
B3aa/B3ab/B3ac run: Chebyshev status = ExpressionBlocked; best_L3_manual_step_ratio = 1.057740728128979
B3ad/B3ae run: Chebyshev status = KernelBlocked; best_L3_manual_step_ratio = 1.2671274405092494
B3af/B3ag run: Chebyshev status = ExpressionBlocked; best_L3_manual_step_ratio = 1.2268657172801187
classic_family_pass_count = 0
```

Route side note：这些 focused smoke run 仍只用于 classic candidate bracket。B320 route 在此类小样本 MNIST seed-only run 中可能显示 `LineCGeometryCollapse`，不能替代前面的 B320-only Line C reconfirm，也不能写成 functional official success。

### 14.5 当前判断更新

Input-cross rank/hidden bracket 没有解决 Chebyshev A4 expression blocker：

```text
P8/P16/P24: L3 efficiency pass, but A4 expression fail
P32/P48: L3 efficiency fail, A4 legally closed
classic_family_pass_count = 0
```

最好的 efficiency 仍是 B3ac 的 `step_ratio_q90 = 1.057740728128979`，但 expression fail；B3aa 的 E1 delta 在本 bracket 内相对最好，为 `-0.050881922245025635`，仍离 gate 很远；E6/E8 没有出现可继续放大的正向趋势。因此不应继续做单纯 input-cross rank sweep。Chebyshev 线下一步若继续，需要新增不同结构的表达项，例如 rotated/random quadratic channel 或改动更深的 basis mixing，而不是在当前 input-cross local-rank 参数上继续加 rank。

截至本节，v12.9 仍未完成：

```text
Functional official success = fail
Classic family pass count = 0
Chebyshev immediate efficiency floor = repaired in focused smoke
Chebyshev expression blocker = unresolved
```

## 15. Classic Chebyshev random-quadratic / rotated-local 结构尝试

### 15.1 代码修改审计

第 14 节的 input-cross product bracket 没有解决 E1/E6/E8。按当时结论，本轮继续尝试两个更直接的结构项：

```text
1. projection-square random quadratic：固定随机投影 `p`，特征从 `(z @ p_left) * (z @ p_right)` 改为 `(z @ p)^2`。
2. local rotated sum/diff square：在 local input pair 上追加 `0.5 * (z_l + z_r)^2` 和 `0.5 * (z_l - z_r)^2`。
```

新增候选：

```text
B3ah-ChebyKAN-K3-h80-paircrossR32-inputsqL4P16-tritonL3-gradbuf
B3ai-ChebyKAN-K3-h72-paircrossR32-inputsqL4P24-tritonL3-gradbuf
B3aj-ChebyKAN-K3-h72-paircrossR32-inputsqL4P8-tritonL3-gradbuf
B3ak-ChebyKAN-K3-h64-paircrossR32-inputsqL4P8-tritonL3-gradbuf
B3al-ChebyKAN-K3-h72-paircrossR32-inputrot2L4P8-tritonL3-gradbuf
B3am-ChebyKAN-K3-h64-paircrossR32-inputrot2L4P8-tritonL3-gradbuf
```

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1283_b109_classic_family_functional_geometry.py
```

代码层面只新增 fixed feature construction 与 candidate entry：`projsq` 让 input projection 的 right buffer 复用 left buffer；`localrot2` 追加 local pair sum/diff square。没有改 loss/data/sampler/class weight/teacher/dataset branch，也没有改 efficiency 或 expression gate。

### 15.2 Focused smoke run 协议与 artifact

本节仍是 focused smoke，不是 full 10-seed family official portfolio run。协议为 `MNIST seed 0`，train/val/test = `512/256/256`，`compile_family_forward=0`，仅用于候选结构筛选。

Artifacts：

```text
results/v12_9_b314_functional_classic_family/v129_b3ah_ai_cheby_inputsq_20260522T160000Z
results/v12_9_b314_functional_classic_family/v129_b3aj_ak_cheby_inputsq_p8_20260522T161000Z
results/v12_9_b314_functional_classic_family/v129_b3al_am_cheby_inputrot2_p8_20260522T162000Z
```

### 15.3 Efficiency 结果

| candidate | structure | hidden | step_ratio_q90 | memory_ratio_q90 | official_fused_l3_kernel | official_efficiency_pass |
|---|---|---:|---:|---:|---:|---:|
| B3ah | projection-square P16 | 80 | 1.5007880731083834 | 0.7885714285714286 | 1 | 0 |
| B3ai | projection-square P24 | 72 | 1.6870619891306067 | 0.7842857142857143 | 1 | 0 |
| B3aj | projection-square P8 | 72 | 1.1936898876492112 | 0.7852380952380953 | 1 | 1 |
| B3ak | projection-square P8 | 64 | 1.1540697329668508 | 0.7814285714285715 | 1 | 1 |
| B3al | localrot2 + P8 | 72 | 1.6762745345466654 | 0.7847619047619048 | 1 | 0 |
| B3am | localrot2 + P8 | 64 | 1.484863496235652 | 0.780952380952381 | 1 | 0 |

B3ah/B3ai 与 B3al/B3am 均因 `step_ratio_q90 > 1.25` 被 efficiency gate 关掉，A4 expression legally closed。只有 B3aj/B3ak 打开了 A4。

### 15.4 Expression 结果

| candidate | A4_expression_pass | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | failure |
|---|---:|---:|---:|---:|---:|---|
| B3aj projection-square P8 h72 | 0 | -0.056920528411865234 | -0.010918378829956055 | -0.08645617961883545 | -0.09720116853713989 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3ak projection-square P8 h64 | 0 | -0.05479186773300171 | -0.009630203247070312 | -0.08225303888320923 | -0.10192501544952393 | `A4_expression_gate_fail_after_l3_efficiency` |
| B3ah projection-square P16 h80 | not run | n/a | n/a | n/a | n/a | L3 efficiency gate fail; A4 legally closed |
| B3ai projection-square P24 h72 | not run | n/a | n/a | n/a | n/a | L3 efficiency gate fail; A4 legally closed |
| B3al localrot2 P8 h72 | not run | n/a | n/a | n/a | n/a | L3 efficiency gate fail; A4 legally closed |
| B3am localrot2 P8 h64 | not run | n/a | n/a | n/a | n/a | L3 efficiency gate fail; A4 legally closed |

Family status：

```text
B3ah/B3ai run: Chebyshev status = KernelBlocked; best_L3_manual_step_ratio = 1.5007880731083834
B3aj/B3ak run: Chebyshev status = ExpressionBlocked; best_L3_manual_step_ratio = 1.1540697329668508
B3al/B3am run: Chebyshev status = KernelBlocked; best_L3_manual_step_ratio = 1.484863496235652
classic_family_pass_count = 0
```

### 15.5 当前判断更新

projection-square P8 可以保住 efficiency，但没有改善 A4：B3ak 的 E1/E6 与 B3aa/B3ab/B3ac/B3af/B3ag 同量级，E8 反而更差到 `-0.10192501544952393`。projection-square P16/P24 与 localrot2 P8 都被 efficiency gate 关掉。

因此 Chebyshev 线目前可以收口为：

```text
paircrossR32: efficiency repaired, expression fail
input-cross product P8/P16/P24: efficiency pass, expression fail
projection-square P8: efficiency pass, expression fail
projection-square P16/P24: efficiency fail
localrot2 P8: efficiency fail
classic_family_pass_count = 0
```

继续在 Chebyshev 上追加小型 fixed sidecar feature 已经没有明显正向趋势。下一步若继续 classic family，应转向 Rational 的 L3 step efficiency repair，或为 Chebyshev/Fourier 做更底层的 fused feature/readout kernel，而不是继续在当前 GEMM sidecar 上叠特征。

## 16. Rational RAT-B4 / signal-channel 继续推进：L3 修复有效，但仍 A5 TaskBlocked

### 16.1 本轮实际代码修改

修改文件：

```text
dgkan/optim/manual_adamw.py
dgkan/models/fc_purekan_primitives.py
experiments/run_v1283_b109_classic_family_functional_geometry.py
```

修改内容：

```text
1. manual_adamw.py:
   AdamW manual update 从显式构造 update tensor 改为 decoupled weight decay + denom.add_(eps) + param.addcdiv_。
   目的：减少一次 update 临时张量分配，符合计划中的 RAT-B4 reduce temporary allocation。

2. fc_purekan_primitives.py / family plan:
   新增 B7kd/B7ke：B7kc 的 linear residual gain 192.0 -> 144.0 / 96.0 bracket。
   新增 B7kf：B7jt/B7jx bridge，R132 + hiddenTanhResidual005 + crossBatchSGCap125。
   新增 B7kg/B7kh：B7kf lower-cap bracket，cap 1.00 / 0.75。
```

没有改 loss、data、sampler、class weight、teacher、distillation、dataset-name branch，也没有放宽 gate。所有结果均来自下列 artifact 的 CSV/JSON。

### 16.2 B7kd/B7ke signal-channel bracket：未修复，降 linear gain 反而伤 task

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kd_ke_rational_gain_bracket_20260522T163000Z
```

在未加 AdamW 原地更新前，B7kc/B7kd/B7ke 全部因 L3 full-step efficiency fail 关闭 A4/A5：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---:|---:|---:|---|
| B7kc | 1.4697366172185744 | 0.770952380952381 | 0 | L3_fused_kernel_efficiency_fail |
| B7kd gain144 | 1.6818197392890226 | 0.770952380952381 | 0 | L3_fused_kernel_efficiency_fail |
| B7ke gain096 | 1.3765416411893485 | 0.770952380952381 | 0 | L3_fused_kernel_efficiency_fail |

这说明单纯降 linear residual gain 不能作为修复结论；该轮 A4/A5 legally closed。

### 16.3 RAT-B4 AdamW 原地更新：L3/A4 重新打开，但 A5 仍失败

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kd_ke_rational_inplace_adamw_20260522T164500Z
```

RAT-B4 修改后，B7kc/B7kd/B7ke 均打开 L3/A4：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | A4 opened |
|---|---:|---:|---:|---:|
| B7kc | 1.1634757587703326 | 0.770952380952381 | 1 | yes |
| B7kd gain144 | 1.1273307463075994 | 0.770952380952381 | 1 | yes |
| B7ke gain096 | 1.1895768476266755 | 0.770952380952381 | 1 | yes |

A5 task summary：

| candidate | A5_task_pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| B7kc | 0 | -0.005859375 | -0.0234375 | 0.5 | 1 | 0 | 0 |
| B7kd gain144 | 0 | -0.015625 | -0.0234375 | 0.25 | 0 | 0 | 0 |
| B7ke gain096 | 0 | -0.03125 | -0.05078125 | 0.0 | 0 | 0 | 0 |

结论：RAT-B4 的临时分配修复是有效的 L3 修复；但降 linear residual gain 不是 task 修复，B7kc 仍是这三者里最稳的 Rational anchor。

### 16.4 Signal-channel bracket：B7jt/B7jx/B7ka/B7kb

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7jt_jx_ka_kb_rational_signal_bracket_20260522T170500Z
```

Efficiency：

| candidate | structure | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---|---:|---:|---:|
| B7kc | R136 hiddenRatResidual005 cap none | 0.7853607993546255 | 0.770952380952381 | 1 |
| B7jt | R132 hiddenTanhResidual005 cap150 | 0.8805819009442447 | 0.7804761904761904 | 1 |
| B7jx | R132 no-hidden cap125 | 0.8406831047706085 | 0.7652380952380953 | 1 |
| B7ka | R128 hiddenSignSqResidual010 cap150 | 0.8970051908576979 | 0.7866666666666666 | 1 |
| B7kb | R120 hiddenSignSqResidual010 cap150 | 0.8985901441198079 | 0.7866666666666666 | 1 |

Failure/task summary：

| candidate | status | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---|---:|---:|---:|---:|---:|---:|
| B7jt | A5 fail | -0.005859375 | -0.0234375 | 0.5 | 1 | 0 | 0 |
| B7jx | A5 fail | -0.005859375 | -0.015625 | 0.5 | 0 | 0 | 0 |
| B7kc | A5 fail | -0.0078125 | -0.03515625 | 0.5 | 0 | 0 | 0 |
| B7ka | A4 fail | n/a | n/a | n/a | n/a | n/a | n/a |
| B7kb | A4 fail | n/a | n/a | n/a | n/a | n/a | n/a |

结论：sign-preserving quadratic residual 的 R128/R120 bracket 虽过 L3，但 A4 expression fail。B7jt 保住 ECE；B7jx 改善 worst_delta 但 ECE fail。合理下一步是桥接 B7jt/B7jx。

### 16.5 B7kf bridge：准确率/ECE 子门通过，但 AUC 仍失败

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kf_rational_cap_bridge_20260522T172500Z
```

Efficiency：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---:|---:|---:|
| B7jt cap150 residual005 | 0.6999306626892924 | 0.7804761904761904 | 1 |
| B7jv cap125 residual010 | 0.7205565482031593 | 0.7804761904761904 | 1 |
| B7jw cap150 no-hidden | 0.6906227361169354 | 0.7652380952380953 | 1 |
| B7jx cap125 no-hidden | 0.6963183847919985 | 0.7652380952380953 | 1 |
| B7kf cap125 residual005 | 1.2397868814747453 | 0.7804761904761904 | 1 |

A5 summary：

| candidate | A5_task_pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|
| B7jt | 0 | 0.01171875 | -0.0078125 | 1.0 | 1 | 0 | 0 |
| B7jv | 0 | 0.001953125 | -0.01171875 | 0.5 | 0 | 0 | 0 |
| B7jw | 0 | 0.001953125 | -0.01171875 | 0.75 | 0 | 0 | 0 |
| B7jx | 0 | 0.00390625 | -0.0234375 | 0.5 | 0 | 0 | 0 |
| B7kf | 0 | 0.0087890625 | -0.00390625 | 1.0 | 1 | 0 | 0 |

B7kf 的 per-task triage：

| dataset | seed | val_acc_delta_vs_mlp | ECE_delta_vs_mlp | AUC_step_ratio | AUC_time_ratio | CEp99 |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | 0.015625 | -0.004295766353607178 | 0.9959142626255663 | 0.9959142626255663 | 5.601113319396973 |
| MNIST | 1 | 0.0234375 | -0.00439026951789856 | 0.9544139591336583 | 0.9544139591336583 | 5.841071605682373 |
| KMNIST | 0 | 0.0 | 0.011241704225540161 | 1.1109845841448576 | 1.1109845841448576 | 7.507471561431885 |
| KMNIST | 1 | -0.00390625 | 0.01281699538230896 | 1.1000423001465953 | 1.1000423001465953 | 7.595818519592285 |

结论：B7kf 是目前最接近 Rational A5 的候选之一。它已经满足 task mean/worst/near/ECE 子门，但 KMNIST loss-AUC 仍为 `1.10-1.11`，超过 `<=1.05` gate。因此不能宣布 Rational family success。

### 16.6 B7kg/B7kh lower-cap：未能打开 L3，不能用来声明 AUC 修复

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kg_kh_rational_lower_cap_20260522T174500Z
```

| candidate | cap | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---:|---:|---:|---:|---|
| B7kf | 1.25 | 1.7852340695669986 | 0.7804761904761904 | 0 | L3_fused_kernel_efficiency_fail |
| B7kg | 1.00 | 1.7278975357268398 | 0.7804761904761904 | 0 | L3_fused_kernel_efficiency_fail |
| B7kh | 0.75 | 1.7556897540060892 | 0.7804761904761904 | 0 | L3_fused_kernel_efficiency_fail |

该轮显示 B7kf/B7kg/B7kh 的 L3 timing 对协议/测量非常敏感；由于 official L3 gate 未打开，本轮 A4/A5 legally closed，不能把 lower-cap 当作 AUC 修复证据。

### 16.7 Rational 当前状态

Rational 不是完成状态。当前真实状态为：

```text
RAT-B4 temporary allocation repair: effective for L3 in focused runs
B7kf: A4 pass and task mean/worst/near/ECE subgates pass in v129_b7kf_rational_cap_bridge_20260522T172500Z
B7kf remaining blocker: KMNIST AUC_step/AUC_time = 1.1109845841448576 / 1.1000423001465953
B7kg/B7kh lower-cap: L3 fail in focused run; A4/A5 closed
Rational family status: TaskBlocked when L3/A4 opens; not family success
classic_family_pass_count = 0
```

下一步如果继续 Rational，不应改 CE/loss/calibration post-hoc。更合理方向是：

```text
1. 先稳定 B7kf 这类 R132 cap/residual path 的 L3 timing，避免同一 candidate 在不同 bracket 中 L3 pass/fail 波动。
2. 在 L3 稳定后，针对 KMNIST loss-AUC 做 label-free logit trajectory / pair-cap schedule / signal reservoir 诊断。
3. 若继续降 cap，必须先保证 L3 gate；L3 不过不得打开 A4/A5 或宣称 task 修复。
```

### 16.8 B7ki/B7kj no-hidden lower-cap：未打开 L3

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7ki = R132 no-hidden + crossBatchSGCap100
  B7kj = R132 no-hidden + crossBatchSGCap075

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7ki/B7kj，对应 D2-KI / D2-KJ
```

目的：沿 B7jx no-hidden cap125 的方向继续降低 stop-gradient batch cap，测试是否能用更弱 pair-logit cap 修复 NLL/ECE/AUC，同时避免 hidden-residual VJP 成本。不改 loss、data、sampler、class weight、teacher、distillation、dataset branch 或 gate。

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7ki_kj_rational_nohidden_lower_cap_20260522T181000Z
```

| candidate | cap | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---:|---:|---:|---:|---:|---|
| B7ki | 1.00 | 0.7107239216566086 | 1.5485035823402566 | 0.7652380952380953 | 0 | L3_fused_kernel_efficiency_fail |
| B7kj | 0.75 | 0.6948071531951427 | 1.513824613150067 | 0.7652380952380953 | 0 | L3_fused_kernel_efficiency_fail |
| B7jw | 1.50 | 0.8109139278531075 | 1.7667945090735913 | 0.7652380952380953 | 0 | L3_fused_kernel_efficiency_fail |
| B7jx | 1.25 | 0.7033379748463631 | 1.5324113069205956 | 0.7652380952380953 | 0 | L3_fused_kernel_efficiency_fail |

结论：no-hidden lower-cap 不是可用修复。虽然绝对 step 在 `0.69-0.71ms`，但该轮 MLP baseline 更快，导致 strict step ratio 超过 `1.25`；A4/A5 legally closed。不能用 B7ki/B7kj 宣称 Rational task 修复。

### 16.9 B7fd/B7fe/B7ga/B7ge low-rank recheck：L3/表达存在明确 tradeoff

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7fd_fe_ga_ge_rational_lowrank_recheck_20260522T182500Z
```

| candidate | structure | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | downstream status |
|---|---|---:|---:|---:|---:|---|
| B7fd | paircross R96 | 1.226296741515398 | 2.080521885101597 | 0.7561904761904762 | 0 | L3 fail |
| B7fe | paircross R80 | 0.6147813983261585 | 1.0430315195898892 | 0.7566666666666667 | 1 | A4_expression_gate_fail_after_l3_efficiency |
| B7ga | balancedPairSumSq R64 | 0.6051409058272839 | 1.0266755635247358 | 0.7576190476190476 | 1 | A4_expression_gate_fail_after_l3_efficiency |
| B7ge | diagPairCross R80 | 0.6147682666778564 | 1.0430092405763722 | 0.7566666666666667 | 1 | A4_expression_gate_fail_after_l3_efficiency |

Status JSON 记录：

```text
Rational status = ExpressionBlocked
best_L3_manual_step_ratio = 1.0266755635247358
A4_expression_pass_candidates = []
```

结论：低秩路线能稳定打开 L3，但 R80/R64 级别表达不足；R132 cap/residual 路线可打开 A4 并接近 A5，但 L3 timing 和 KMNIST AUC 仍不稳。下一步更合理的是在 R120/R128 中间档测试 B7kf 形态的 `hiddenTanhResidual005 + crossBatchSGCap125`，而不是继续压到 R80/R64 或修改训练目标。

### 16.10 B7kk/B7kl middle-rank：接近但仍未过 L3

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7kk = R128 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kl = R120 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7kk/B7kl，对应 D2-KK / D2-KL
  modification audit 增加 B7kd-B7kl 与 manual AdamW in-place update 审计条目
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kk_kl_rational_midrank_20260522T190000Z
```

| candidate | rank | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---:|---:|---:|---:|---:|---|
| B7kf | R132 | 0.7537900470197201 | 1.3264660639757866 | 0.7804761904761904 | 0 | L3_fused_kernel_efficiency_fail |
| B7kk | R128 | 0.7349021732807159 | 1.293228528358932 | 0.780952380952381 | 0 | L3_fused_kernel_efficiency_fail |
| B7kl | R120 | 0.7434863597154617 | 1.3083343685561477 | 0.780952380952381 | 0 | L3_fused_kernel_efficiency_fail |

Status JSON：

```text
Rational status = KernelBlocked
best_L3_manual_step_ratio = 1.293228528358932
L3_official_efficiency_candidate_exists = false
```

A4/A5：

```text
B7kf/B7kk/B7kl: official L2/L3 full-step gate not available; A4 legally closed
B7kf/B7kk/B7kl: A4 legally closed; no official task triage
```

结论：R128 是该 bracket 中最接近 L3 gate 的点，但仍超过 `1.25`；R120 没有进一步降速，说明 hidden-residual VJP/forward 固定成本和 timing 噪声已占主要部分。不能宣布 Rational 修复。下一步继续沿同结构测试 R112/R96，看是否能以更低 rank 打开 L3；若打开 L3，再由 A4 判断表达是否已经被压坏。

### 16.11 B7km/B7kn lower-middle rank：打开 L3，但 A4 表达失败

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7km = R112 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kn = R96 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7km/B7kn，对应 D2-KM / D2-KN
  modification audit 扩展到 B7km/B7kn
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7km_kn_rational_lowermid_20260522T191500Z
```

Efficiency：

| candidate | rank | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---:|---:|---:|---:|---:|
| B7km | R112 | 0.7584019564092159 | 1.2443526615750249 | 0.7814285714285715 | 1 |
| B7kn | R96 | 0.7568242028355598 | 1.2417639527220257 | 0.7819047619047619 | 1 |

Expression summary：

| candidate | A4_expression_pass | E1 B1 delta | E1 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B7km | 0 | -0.7293840050697327 | 0.9444594383239746 | -0.5323372483253479 | 0.5704424381256104 | -0.48428481817245483 | 0.6098576784133911 | A4_expression_gate_fail_after_l3_efficiency |
| B7kn | 0 | -0.7131876349449158 | 0.9443696737289429 | -0.7243209481239319 | 0.3888845443725586 | -0.6291971802711487 | 0.47633498907089233 | A4_expression_gate_fail_after_l3_efficiency |

Task triage：

```text
B7km/B7kn: A4 expression gate failed; no official task triage
```

Status JSON：

```text
Rational status = ExpressionBlocked
best_L3_manual_step_ratio = 1.2417639527220257
L3_official_efficiency_candidate_exists = true
A4_expression_pass_candidates = []
```

结论：R112/R96 终于打开 Rational strict L3，但 rank 降低后 rotated/random quadratic 表达明显不足，不能进入 A5。此结果把 blocker 从 B7kk/B7kl 的 `KernelBlocked` 推进到 `ExpressionBlocked`，但仍不是 family success。下一步不应继续单纯降 rank；更合理方向是保持 R112/R96 的 L3 成本，同时调整 pair feature 覆盖方式以修复 E6/E8。

### 16.12 B7ko/B7kp/B7kq R112 pair coverage：L3 通过，A4 仍失败

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7ko = hybridPairCrossR112 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kp = balancedPairCrossR112 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kq = diagPairCrossR112 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7ko/B7kp/B7kq，对应 D2-KO / D2-KP / D2-KQ
  modification audit 扩展到 B7ko/B7kp/B7kq
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7ko_kp_kq_rational_paircoverage_20260522T193000Z
```

Efficiency：

| candidate | coverage | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---|---:|---:|---:|---:|
| B7ko | hybrid R112 | 0.7437638007104397 | 1.210651748524831 | 0.7814285714285715 | 1 |
| B7kp | balanced R112 | 0.7333733141422272 | 1.1937387706146199 | 0.7814285714285715 | 1 |
| B7kq | diag-rich R112 | 0.7644649595022202 | 1.2443477875950575 | 0.7814285714285715 | 1 |

Expression summary：

| candidate | A4_expression_pass | E1 B1 delta | E1 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B7ko | 0 | -0.520936131477356 | 0.944758415222168 | -0.7177954912185669 | 0.3947852849960327 | -0.6890914440155029 | 0.44365930557250977 | A4_expression_gate_fail_after_l3_efficiency |
| B7kp | 0 | -0.87193363904953 | 0.746212899684906 | -0.4224943518638611 | 0.6470457315444946 | -0.5244187116622925 | 0.526295006275177 | A4_expression_gate_fail_after_l3_efficiency |
| B7kq | 0 | -0.7080847024917603 | 0.945151686668396 | -0.4862184524536133 | 0.6249673366546631 | -0.3551555871963501 | 0.7327958345413208 | A4_expression_gate_fail_after_l3_efficiency |

Task triage：

```text
B7ko/B7kp/B7kq: A4 expression gate failed; no official task triage
```

Status JSON：

```text
Rational status = ExpressionBlocked
best_L3_manual_step_ratio = 1.1937387706146199
L3_official_efficiency_candidate_exists = true
A4_expression_pass_candidates = []
```

结论：pair coverage 改动是有效诊断但不是完成。B7kp 改善 E6，B7kq 改善 E8，但三者都未同时满足 A4 gate。下一步应尝试 R120/R128 的 hybrid/diag/balanced 覆盖中间档；如果 L3 仍过，才可能重新打开 A4/A5。

### 16.13 B7kr-B7kv R120/R128 pair coverage：仍然 A4 阻塞

本轮新增实现：

```text
fc_purekan_primitives.py:
  parser 扩展 hybrid/balanced/diag pair coverage 到 R120/R128
  B7kr = hybridPairCrossR128 + hiddenTanhResidual005 + crossBatchSGCap125
  B7ks = balancedPairCrossR128 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kt = diagPairCrossR128 + hiddenTanhResidual005 + crossBatchSGCap125
  B7ku = hybridPairCrossR120 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kv = diagPairCrossR120 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7kr-B7kv
  modification audit 扩展到 B7kr-B7kv
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kr_kv_rational_paircoverage_rank_20260522T194500Z
```

Efficiency：

| candidate | coverage | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---|---:|---:|---:|---:|
| B7kr | hybrid R128 | 0.7360649295151234 | 1.2165184621828058 | 0.780952380952381 | 1 |
| B7ks | balanced R128 | 0.7318378426134586 | 1.2095322181017112 | 0.780952380952381 | 1 |
| B7kt | diag R128 | 0.7724525406956673 | 1.276657451313861 | 0.780952380952381 | 0 |
| B7ku | hybrid R120 | 0.7334698922932148 | 1.2122295597179336 | 0.780952380952381 | 1 |
| B7kv | diag R120 | 0.7282314822077751 | 1.2035718689003314 | 0.780952380952381 | 1 |

Expression summary：

| candidate | A4_expression_pass | E1 B1 delta | E1 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B7kr | 0 | -0.5195634365081787 | 0.9446915984153748 | -0.7065818905830383 | 0.2891489863395691 | -0.6345597505569458 | 0.4302733540534973 | A4_expression_gate_fail_after_l3_efficiency |
| B7ks | 0 | -0.8688762784004211 | 0.742963969707489 | -0.30678486824035645 | 0.803571879863739 | -0.298319935798645 | 0.7926041483879089 | A4_expression_gate_fail_after_l3_efficiency |
| B7ku | 0 | -0.5039899349212646 | 0.9446340203285217 | -0.7196851372718811 | 0.31738394498825073 | -0.6419963240623474 | 0.4299589991569519 | A4_expression_gate_fail_after_l3_efficiency |
| B7kv | 0 | -0.7683863639831543 | 0.9449958205223083 | -0.4334074854850769 | 0.6566030383110046 | -0.2965828776359558 | 0.8435357809066772 | A4_expression_gate_fail_after_l3_efficiency |

B7kt 没打开 A4，因为 L3 失败。

Task triage：

```text
B7kt: A4 legally closed; no official task triage
B7kr/B7ks/B7ku/B7kv: A4 expression gate failed; no official task triage
```

Status JSON：

```text
Rational status = ExpressionBlocked
best_L3_manual_step_ratio = 1.2035718689003314
A4_expression_pass_candidates = []
```

结论：提高 rank 并更换 pair coverage 仍没打开 A4。B7ks 的 E6/E8 明显接近，但 E1 frozen R2 降到 `0.742963969707489`；B7kv 的 E8 frozen R2 达到 `0.8435357809066772`，但 E6/E1 仍不足。下一步如果继续，应尝试 balanced-dominant mixed coverage，用少量 standard pair 保住 E1，同时让大部分 rank 覆盖 balanced/diag 型 E6/E8。

### 16.14 B7kw/B7kx balanced-dominant mix：L3 失败，A4 合法关闭

本轮新增实现：

```text
fc_purekan_primitives.py:
  新增 balanced_blend_pair_index 模式：
    primary standard pair rank = round(0.25 * R)
    secondary balanced pair rank = remaining
  parser 增加 balblendpaircrossr120 / balblendpaircrossr128
  B7kw = balBlendPairCrossR128 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kx = balBlendPairCrossR120 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7kw/B7kx
  modification audit 扩展到 B7kw/B7kx
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kw_kx_rational_balblend_20260522T200000Z
```

Efficiency：

| candidate | coverage | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---|---:|---:|---:|---:|---|
| B7kw | balBlend R128 | 0.7796904072165489 | 1.3972226654214068 | 0.780952380952381 | 0 | L3_fused_kernel_efficiency_fail |
| B7kx | balBlend R120 | 0.7779400795698166 | 1.3940860390921708 | 0.780952380952381 | 0 | L3_fused_kernel_efficiency_fail |

A4/A5：

```text
B7kw/B7kx: official L2/L3 full-step gate not available; A4 legally closed
```

Status JSON：

```text
Rational status = KernelBlocked
best_L3_manual_step_ratio = 1.3940860390921708
L3_official_efficiency_candidate_exists = false
```

结论：balanced-dominant mixed coverage 没有成为修复路线；它在 R120/R128 都没有打开 L3，因此不能评估 A4/A5。到目前为止，Rational 的可审计状态是：

```text
R132/B7kf: 可打开 A4 并接近 A5，但 KMNIST AUC 失败，且 L3 timing 不稳定
R112/R96: L3 可打开，但 A4 表达失败
R112/R120/R128 pair coverage variants: L3 多数可打开，但 A4 仍失败；balanced-dominant mix 反而 L3 失败
R80/R64 low-rank: L3 稳定，但 A4 表达失败
```

因此 Rational 仍未完成。继续推进需要新的表达-效率设计，而不是继续微调已有 rank/cap；比较合理但尚未验证的方向是设计更低成本的 rotated/random quadratic feature path 或进一步优化 block-readout/hidden-residual VJP kernel，而不是改 loss/data/gate。

### 16.15 B7ky/B7kz/B7la R32 projection quadratic sidecar：L3 通过，A4 仍失败

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7ky = paircrossR80 + projBilinR32 + hiddenTanhResidual005 + crossBatchSGCap125
  B7kz = paircrossR80 + projSqR32 + hiddenTanhResidual005 + crossBatchSGCap125
  B7la = diagPairCrossR80 + projBilinR32 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7ky/B7kz/B7la
  modification audit 扩展到 B7ky/B7kz/B7la
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7ky_kz_la_rational_projquad_20260522T203000Z
```

Efficiency：

| candidate | sidecar | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass |
|---|---|---:|---:|---:|---:|
| B7ky | pair R80 + projBilin R32 | 0.8589792065322399 | 1.1599694891124175 | 0.8066666666666666 | 1 |
| B7kz | pair R80 + projSq R32 | 0.8364256471395493 | 1.12951305830774 | 0.8219047619047619 | 1 |
| B7la | diag R80 + projBilin R32 | 0.8446225896477699 | 1.1405822473419307 | 0.8066666666666666 | 1 |

Expression summary：

| candidate | A4_expression_pass | E1 B1 delta | E1 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B7ky | 0 | -0.5261868238449097 | 0.9450126886367798 | -0.7159373760223389 | 0.41630029678344727 | -0.5648972988128662 | 0.5014710426330566 | A4_expression_gate_fail_after_l3_efficiency |
| B7kz | 0 | -0.532431423664093 | 0.9450594782829285 | -0.6650040149688721 | 0.41205674409866333 | -0.5839048624038696 | 0.5010650157928467 | A4_expression_gate_fail_after_l3_efficiency |
| B7la | 0 | -0.5189192295074463 | 0.946117639541626 | -0.5515845417976379 | 0.5718390941619873 | -0.6093506813049316 | 0.5162208080291748 | A4_expression_gate_fail_after_l3_efficiency |

Gradcheck / audit：

```text
B7ky/B7kz/B7la L3 analytic manual CE vs autograd:
  official_analytic_gradcheck_pass = 1
  grad_cos_min = 0.9999999403953552 / 1.0 / 0.9999999403953552
  grad_relerr_max = 2.7583837436395697e-07 / 2.9616438723678584e-07 / 2.740164575243398e-07

provenance:
  fake_data_used_sum = 0
  proxy_row_used_sum = 0
  cpu_offload_used_sum = 0
```

Task triage：

```text
B7ky/B7kz/B7la: A4 expression gate failed; no official task triage
```

结论：固定 DCT projection quadratic sidecar 的 R32 版本能保住 L3，并且 B7kz 给出本轮 projection 方向最好 L3 step ratio `1.12951305830774`；但 E6/E8 frozen R2 仍远低于 `0.89`，A4 仍失败。R32 投影读出太弱，不能补足 Rational 低 rank 的 rotated/random quadratic 表达缺口。

### 16.16 B7lb/B7lc/B7ld R64 projection quadratic sidecar：L3 失败，A4 合法关闭

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7lb = paircrossR80 + projBilinR64 + hiddenTanhResidual005 + crossBatchSGCap125
  B7lc = paircrossR80 + projSqR64 + hiddenTanhResidual005 + crossBatchSGCap125
  B7ld = diagPairCrossR80 + projBilinR64 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lb/B7lc/B7ld
  modification audit 扩展到 B7lb/B7lc/B7ld
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lb_lc_ld_rational_projquad_r64_20260522T204500Z
```

Efficiency：

| candidate | sidecar | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---|---:|---:|---:|---:|---|
| B7lb | pair R80 + projBilin R64 | 0.8566365577280521 | 1.438080502858708 | 0.8514285714285714 | 0 | L3_fused_kernel_efficiency_fail |
| B7lc | pair R80 + projSq R64 | 0.8488385006785393 | 1.4249894974586743 | 0.8819047619047619 | 0 | L3_fused_kernel_efficiency_fail |
| B7ld | diag R80 + projBilin R64 | 0.8692636154592037 | 1.459278203759777 | 0.8514285714285714 | 0 | L3_fused_kernel_efficiency_fail |

Gradcheck：

```text
B7lb/B7lc/B7ld L3 analytic manual CE vs autograd:
  official_analytic_gradcheck_pass = 1
  grad_relerr_max = 2.7583837436395697e-07 / 2.9616438723678584e-07 / 2.740164575243398e-07
```

A4/A5：

```text
B7lb/B7lc/B7ld: official L2/L3 full-step gate not available; A4 legally closed
```

Status JSON：

```text
Rational status = KernelBlocked
best_L3_manual_step_ratio = 1.4249894974586743
L3_official_efficiency_candidate_exists = false
```

结论：R64 projection sidecar 梯度正确，但成本超出 L3 gate。它不能作为当前版本的表达修复路线。

### 16.17 B7le/B7lf/B7lg R48 projection bridge：仍然 L3 失败

本轮新增实现：

```text
fc_purekan_primitives.py:
  parser 增加 projBilinR48
  B7le = paircrossR80 + projBilinR48 + hiddenTanhResidual005 + crossBatchSGCap125
  B7lf = paircrossR80 + projSqR48 + hiddenTanhResidual005 + crossBatchSGCap125
  B7lg = diagPairCrossR80 + projBilinR48 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7le/B7lf/B7lg
  modification audit 扩展到 B7le/B7lf/B7lg
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7le_lf_lg_rational_projquad_r48_20260522T205500Z
```

Efficiency：

| candidate | sidecar | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---|---:|---:|---:|---:|---|
| B7le | pair R80 + projBilin R48 | 0.8562086150050163 | 1.536795194438199 | 0.829047619047619 | 0 | L3_fused_kernel_efficiency_fail |
| B7lf | pair R80 + projSq R48 | 1.04993162676692 | 1.8845055401534712 | 0.8519047619047619 | 0 | L3_fused_kernel_efficiency_fail |
| B7lg | diag R80 + projBilin R48 | 0.8374218828976154 | 1.5030751884538136 | 0.829047619047619 | 0 | L3_fused_kernel_efficiency_fail |

Gradcheck：

```text
B7le/B7lf/B7lg L3 analytic manual CE vs autograd:
  official_analytic_gradcheck_pass = 1
  grad_relerr_max = 2.7583837436395697e-07 / 2.9616438723678584e-07 / 2.740164575243398e-07
```

结论：R48 并没有提供 R32/R64 之间的可用窗口。projection quadratic sidecar 当前实现的结论是：

```text
R32: L3 pass, A4 fail
R48/R64: L3 fail, A4 legally closed
```

因此“固定投影二次读出叠加 R80 pair path”不是当前可行修复方向，除非先优化 projection feature/readout 的 fused kernel 成本。

### 16.18 B7lh/B7li/B7lj low-cost pair-feature fallback：仍然 A4 阻塞

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7lh = pairSumSqCrossR80 + hiddenTanhResidual005 + crossBatchSGCap125
  B7li = pairDiffSqCrossR80 + hiddenTanhResidual005 + crossBatchSGCap125
  B7lj = blendPairCrossR80 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lh/B7li/B7lj
  modification audit 扩展到 B7lh/B7li/B7lj
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lh_li_lj_rational_pairfeature_fallback_20260522T211000Z
```

Efficiency：

| candidate | feature mode | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---|---:|---:|---:|---:|---|
| B7lh | pair sum-square R80 | 1.252115610986948 | 2.012032396075213 | 0.7823809523809524 | 0 | L3_fused_kernel_efficiency_fail |
| B7li | pair diff-square R80 | 0.7575568743050098 | 1.2173228730609975 | 0.7823809523809524 | 1 | A4_expression_gate_fail_after_l3_efficiency |
| B7lj | 75/25 standard-balanced R80 | 0.7654679007828236 | 1.2300351509208858 | 0.7823809523809524 | 1 | A4_expression_gate_fail_after_l3_efficiency |

Expression summary：

| candidate | A4_expression_pass | E1 B1 delta | E1 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B7li | 0 | -0.6405795812606812 | 0.9447864890098572 | -0.8304551839828491 | 0.33285242319107056 | -0.7194696068763733 | 0.4296143651008606 | A4_expression_gate_fail_after_l3_efficiency |
| B7lj | 0 | -0.6508746147155762 | 0.9446768164634705 | -0.7652820944786072 | 0.3146774172782898 | -0.6940403580665588 | 0.4425596594810486 | A4_expression_gate_fail_after_l3_efficiency |

B7lh 没有 expression summary，因为 L3 失败，A4 合法关闭。

Gradcheck / audit：

```text
B7lh/B7li/B7lj L3 analytic manual CE vs autograd:
  official_analytic_gradcheck_pass = 1
  grad_relerr_max = 2.7583837436395697e-07 / 2.9616438723678584e-07 / 2.740164575243398e-07

provenance:
  fake_data_used_sum = 0
  proxy_row_used_sum = 0
  cpu_offload_used_sum = 0
```

Task triage：

```text
B7lh: A4 legally closed; no official task triage
B7li/B7lj: A4 expression gate failed; no official task triage
```

Status JSON：

```text
Rational status = ExpressionBlocked
best_L3_manual_step_ratio = 1.2173228730609975
A4_expression_pass_candidates = []
```

结论：低成本 pair-feature fallback 也没有打开 A4。B7li/B7lj 能保住 L3，但 E6/E8 frozen R2 仍只有 `0.33285242319107056/0.4296143651008606` 和 `0.3146774172782898/0.4425596594810486`，比 B7ks/B7kv 等更高 rank coverage 还弱；B7lh 直接 L3 失败。因此截至本节，Rational 仍未完成：

```text
Functional official success = 0
Classic family pass count = 0
Rational family success = 0
Rational current blocker = ExpressionBlocked
```

下一步不应继续扩展 R80 低成本 feature 枚举，也不应改 CE/loss/data/gates。可审计的下一修复方向只剩两类：

```text
1. kernel-side：优化 projection quadratic sidecar 的 fused feature/readout/VJP 成本，使 R48/R64 能合法进入 A4；
2. expression-side：设计新的低成本 rotated/random quadratic coverage，不再依赖当前 R80 pair subset 或 projection GEMM sidecar。
```

### 16.19 B7lk/B7ll/B7lm random-projection inputcross：梯度正确，但 L3 仍失败

本轮新增实现：

```text
fc_purekan_primitives.py:
  B7lk = inputcrossR64 + hiddenTanhResidual005 + crossBatchSGCap125
  B7ll = inputcrossR96 + hiddenTanhResidual005 + crossBatchSGCap125
  B7lm = inputcrossR112 + hiddenTanhResidual005 + crossBatchSGCap125

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lk/B7ll/B7lm
  modification audit 扩展到 B7lk/B7ll/B7lm
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lk_ll_lm_rational_inputcross_random_20260522T212500Z
```

Efficiency：

| candidate | route | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---|---:|---:|---:|---:|---|
| B7lk | random-projection inputcross R64 | 0.728351715952158 | 1.7222606874140864 | 0.8195238095238095 | 0 | L3_fused_kernel_efficiency_fail |
| B7ll | random-projection inputcross R96 | 0.7308217696845531 | 1.7281013772153637 | 0.8638095238095238 | 0 | L3_fused_kernel_efficiency_fail |
| B7lm | random-projection inputcross R112 | 0.7289335131645203 | 1.7236364052780704 | 0.8861904761904762 | 0 | L3_fused_kernel_efficiency_fail |

Gradcheck：

```text
B7lk/B7ll/B7lm L3 analytic manual CE vs autograd:
  official_analytic_gradcheck_pass = 1
  grad_relerr_max = 2.7583837436395697e-07 / 2.9616438723678584e-07 / 2.740164575243398e-07
```

Status JSON：

```text
Rational status = KernelBlocked
best_L3_manual_step_ratio = 1.7222606874140864
L3_official_efficiency_candidate_exists = false
```

结论：random-projection inputcross 的数学梯度没有暴露问题，但当前实现的 full-step L3 成本明显超过 gate。由于 L3 未开，A4/A5 合法关闭。本结果支持下一步继续做 RAT-B kernel/timing repair 或降低投影路径成本，而不是直接进入 task triage。

### 16.20 B7ln/B7lo low-rank inputcross：低秩能降成本，但仍未过 L3

根据 16.19 的 L3 close fail，继续按 RAT-B4 方向尝试低秩 random-projection inputcross，而不是增加表达量。

本轮新增/修改：

```text
fc_purekan_primitives.py:
  parser 增加 inputcrossR32 / inputcrossR48
  新增 B7ln / B7lo PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7ln/B7lo
  modification audit 扩展到 B7ln/B7lo
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7ln_lo_rational_inputcross_lowrank_20260522T214000Z
```

Efficiency：

| candidate | route | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | failure |
|---|---|---:|---:|---:|---:|---|
| B7ln | random-projection inputcross R32 | 0.7268406450748444 | 1.580100640309493 | 0.7995238095238095 | 0 | L3_fused_kernel_efficiency_fail |
| B7lo | random-projection inputcross R48 | 0.7347248494625092 | 1.5972403482849562 | 0.8066666666666666 | 0 | L3_fused_kernel_efficiency_fail |

Gradcheck：

```text
B7ln/B7lo L3 analytic manual CE vs autograd:
  official_analytic_gradcheck_pass = 1
  grad_relerr_max = 2.7583837436395697e-07 / 2.9616438723678584e-07
```

Expression：

```text
B7ln/B7lo:
  status = not_run
  reason = official L2/L3 full-step gate not available; A4 legally closed for classic family
```

Status JSON：

```text
Rational status = KernelBlocked
best_L3_manual_step_ratio = 1.580100640309493
L3_official_efficiency_candidate_exists = false
```

结论：R32/R48 相比 R64/R96/R112 确实降低了 step ratio，但仍远高于 `1.25` L3 gate。因此 random-projection inputcross 当前不是可用 family-success 路线；继续降 rank 预计主要降低表达能力，且从 R64 到 R32 的降幅仍不足以打开 L3。这个方向暂时关闭，除非实现真正 fused projection/readout/VJP。

### 16.21 B7kc Rational task-geometry autopsy：确认 TaskBlocked，并新增 denominator/derivative 诊断

按计划 10.3，Rational 在已有 L3/A4-capable 候选上不继续 CE-tune，而进入 task-geometry autopsy。本轮补充了一个只记录诊断、不改变 gate 的输出：

```text
experiments/run_v1283_b109_classic_family_functional_geometry.py:
  新增 v1283_family_basis_diagnostics.csv

fc_purekan_primitives.py:
  Rational basis_diagnostics 增加 den_min / den_p01 / den_condition
  增加 r_prime_p95 / r_double_prime_p95
  增加 group_function_diversity / group_dead_fraction
```

审计修复说明：

```text
第一次 B7kc autopsy 运行触发 NameError: z is not defined。
原因：新增 Rational diagnostic 中使用 z 但未在该 basis_diagnostics 作用域内定义。
修复：在 Rational basis_diagnostics 内显式计算 z = self._norm_input(x)。
随后用同一 artifact 路径 --fresh 重跑，避免半成品数据混入结论。
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kc_rational_task_geometry_autopsy_20260522T220500Z
```

Efficiency / gradcheck：

| candidate | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|---:|
| B7kc | 0.6561575457453728 | 1.0335377933718919 | 0.770952380952381 | 1 | 1.5542421749614732e-07 |

Expression summary：

| A4_expression_pass | E1 B1 delta | E1 frozen R2 | E2 B1 delta | E2 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.7208373546600342 | 0.9435589909553528 | -0.0537952184677124 | 0.9799932241439819 | -0.19320064783096313 | 0.9381912350654602 | -0.1697298288345337 | 0.9330353140830994 |

Task summary：

| A5_task_pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.001953125 | -0.0234375 | 0.75 | 0 | 0 | 0 |

Basis / rational safety diagnostic：

| den_min | den_p01 | den_condition | r_prime_p95 | r_double_prime_p95 | group_function_diversity | group_dead_fraction | basis_effective_rank | basis_condition_proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000005841255188 | 1.0000253915786743 | 1.2638226747512817 | 0.99552321434021 | 0.9919778108596802 | 0.5759848645683765 | 0.0 | 25.52700424194336 | 13525395.0 |

Provenance：

```text
v1283_family_basis_diagnostics.csv:
  fake_data_used_sum = 0
  proxy_row_used_sum = 0
  cpu_offload_used_sum = 0

v1283_family_task_triage.csv:
  fake_data_used_sum = 0
  proxy_row_used_sum = 0
  cpu_offload_used_sum = 0
```

Status JSON：

```text
Rational status = TaskBlocked
best_L3_manual_step_ratio = 1.0335377933718919
A4_expression_pass_candidates = [B7kc]
A5_task_pass_candidates = []
blocker = family-specific fused L3 efficiency and A4 expression passed; A5 fused task triage failed
```

结论：B7kc 确认不是 denominator unsafe。`den_p01 = 1.0000253915786743` 明显高于计划中的 `0.50` safety gate，`r_prime_p95` 和 `r_double_prime_p95` 也未显示爆炸；真正 blocker 仍在 task geometry / calibration / AUC trajectory。与此同时 `basis_condition_proxy = 13525395.0` 很高，说明 frozen/readout feature geometry 可能病态；后续 Rational 不应继续 CE-specific calibration，而应围绕 signal-channel geometry、feature conditioning、Line C / NoiseSignalLeak / RealSignalReservoirRatio 做 autopsy 和结构修复。

截至本节：

```text
Functional official success = 0
Classic family pass count = 0
Rational family success = 0
Rational best current status = TaskBlocked, not KernelBlocked
```

### 16.22 B7en pairNorm autopsy：feature normalization 改善部分表达，但仍 A5 TaskBlocked

基于 16.21 的 `basis_condition_proxy` 很高，继续测试一个已经存在于 runner/primitive 的结构性对照，而不是新增 CE-tuning：

```text
B7en = paircrossR136 + readblocktriton + pairNorm + crossZero + b32 + freezeBackbone + hiddenBias
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7en_rational_pairnorm_autopsy_20260522T222000Z
```

Efficiency / gradcheck：

| candidate | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|---:|
| B7en | 0.751862209290266 | 1.0376183715590022 | 0.7547619047619047 | 1 | 1.7611560565455875e-07 |

Expression summary：

| A4_expression_pass | E1 B1 delta | E1 frozen R2 | E2 B1 delta | E2 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.44707655906677246 | 0.9435766339302063 | -0.031053602695465088 | 0.9801877737045288 | -0.07333129644393921 | 0.9364123344421387 | -0.07964271306991577 | 0.931158721446991 |

Task summary：

| A5_task_pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | -0.02734375 | 0.75 | 0 | 0 | 0 |

Basis / rational safety diagnostic：

| den_min | den_p01 | den_condition | r_prime_p95 | r_double_prime_p95 | group_function_diversity | group_dead_fraction | basis_effective_rank | basis_condition_proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000005841255188 | 1.0000253915786743 | 1.2638226747512817 | 0.99552321434021 | 0.9919778108596802 | 0.5759848645683765 | 0.0 | 25.527292251586914 | 13305585.0 |

Status JSON：

```text
Rational status = TaskBlocked
best_L3_manual_step_ratio = 1.0376183715590022
A4_expression_pass_candidates = [B7en]
A5_task_pass_candidates = []
```

结论：pairNorm 是有价值的表达修复信号，E1/E6/E8 B1 delta 从 B7kc 的 `-0.7208373546600342/-0.19320064783096313/-0.1697298288345337` 改善到 `-0.44707655906677246/-0.07333129644393921/-0.07964271306991577`，且 L3 仍过。但 A5 没有改善为 success：`mean_delta = 0.0`、`worst_delta = -0.02734375`、`near_pass_rate = 0.75`、ECE/AUC 全部未过。因此 Rational blocker 进一步收窄为：

```text
不是 denominator unsafe；
不是 L3/A4 缺口；
不是单纯 pair feature normalization；
仍是 task trajectory / signal-channel / AUC-ECE geometry blocked。
```

### 16.23 B7lp pairNorm + hiddenRatResidual005：组合修复仍 A5 fail

根据 16.21/16.22，继续测试一个结构性组合候选：

```text
B7lp = B7en pairNorm + B7kc hiddenRatResidual005 + hiddenResVJPTriton + manualAdamW
```

本轮新增/修改：

```text
fc_purekan_primitives.py:
  新增 B7lp PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lp
  modification audit 扩展到 B7lp
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lp_rational_pairnorm_hiddenrat_20260522T223500Z
```

Efficiency / gradcheck：

| candidate | abs_step_q90_ms | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|---:|
| B7lp | 0.7086933590471745 | 0.7347600905789334 | 0.770952380952381 | 1 | 1.866684300466659e-07 |

Expression summary：

| A4_expression_pass | E1 B1 delta | E1 frozen R2 | E2 B1 delta | E2 frozen R2 | E6 B1 delta | E6 frozen R2 | E8 B1 delta | E8 frozen R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.43790775537490845 | 0.9434381723403931 | -0.027489900588989258 | 0.9806293845176697 | -0.07602697610855103 | 0.9358267784118652 | -0.08070600032806396 | 0.9303527474403381 |

Task summary：

| A5_task_pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | -0.02734375 | 0.75 | 0 | 0 | 0 |

Basis / rational safety diagnostic：

| den_min | den_p01 | den_condition | r_prime_p95 | r_double_prime_p95 | group_function_diversity | group_dead_fraction | basis_effective_rank | basis_condition_proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000005841255188 | 1.0000253915786743 | 1.2638226747512817 | 0.99552321434021 | 0.9919778108596802 | 0.5759848645683765 | 0.0 | 25.52739715576172 | 13583584.0 |

Status JSON：

```text
Rational status = TaskBlocked
best_L3_manual_step_ratio = 0.7347600905789334
A4_expression_pass_candidates = [B7lp]
A5_task_pass_candidates = []
```

结论：B7lp 是本轮最强的 Rational L3 result 之一，`step_ratio_q90 = 0.7347600905789334`，并且 A4 继续通过；但 A5 完全没有打开，task summary 与 B7en 同型：`mean_delta = 0.0`、`worst_delta = -0.02734375`、`near_pass_rate = 0.75`、ECE/AUC 全部失败。因此这一轮可以排除：

```text
pairNorm alone;
pairNorm + hiddenRatResidual005 damping;
random-projection inputcross low-rank;
projection quadratic sidecar;
R80 pair-feature fallback.
```

Rational 下一步不应继续小幅组合已有 pair/readout/cap/residual token。更合理的下一步是按计划进入 Line C / Manifold-channel autopsy，直接测 `NoiseSignalLeak`、`RealSignalReservoirRatio`、signal-channel alignment，然后再设计结构，而不是继续用 A5 task loss 盲筛。

### 16.24 Family Line C autopsy 接入：B7kc/B7en/B7lp 均显示 coupling collapse

本轮新增/修改：

```text
experiments/run_v1283_b109_classic_family_functional_geometry.py:
  新增 --family-linec-ids
  新增 run_family_linec_autopsy()
  新增 v1283_family_linec_coupling.csv
  新增 v1283_family_linec_signal_reservoir.csv
  新增 v1283_family_linec_noise_leak.csv
  新增 v1283_family_linec_diagnostics.csv
  新增 v1283_family_linec_summary.csv
  modification audit 记录 family Line C autopsy wrapper
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_family_linec_rational_b7kc_b7en_b7lp_20260522T225500Z
```

关键结果：

| candidate | L3 official_efficiency_pass | L3 step_ratio_q90 | A4 pass | A5 pass | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | LineC interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B7kc | 0 | 1.250309871294078 | 0 in this focused run | not_run | 0.23913910138598593 | -0.09654141791370918 | 0.2573724687099457 | 0.21777014434337616 | coupling_collapse |
| B7en | 0 | 1.7180338337496952 | 0 in this focused run | not_run | 0.24227255550604154 | -0.09340796379365357 | 0.2581622898578644 | 0.21490536630153656 | coupling_collapse |
| B7lp | 1 | 1.2466548833795918 | 1 | 0 | 0.24114959500970579 | -0.09453092428998933 | 0.25764310359954834 | 0.2142498642206192 | coupling_collapse |

MLP Line C reference in this run：

```text
MLP CouplingR2 = 0.3356805192996951
MLP NoiseSignalLeak = 0.4323427677154541
MLP RealSignalReservoirRatio = 0.08000694215297699
```

Basis/rational safety 仍正常：

```text
den_min = 1.000005841255188
den_p01 = 1.0000253915786743
den_condition = 1.2638226747512817
r_prime_p95 = 0.99552321434021
r_double_prime_p95 = 0.9919778108596802
group_dead_fraction = 0.0
```

结论：Family Line C 已成功接入并落盘，没有出现 fake/proxy/cpu offload。B7lp 在该 run 仍能 L3/A4，但 Line C 明确显示 `CouplingR2` 比 MLP 低约 `0.0945`，且 `RealSignalReservoirRatio` 高于 MLP 约 `0.1342`。这支持计划中的 H4：Rational 不是 denominator unsafe，也不是单纯 L3/A4 问题，而是 task geometry / signal-channel blocked。

### 16.25 B7lq/B7lr：pairNorm + stop-gradient batch cap 未修复 Line C

根据 16.24 的 Line C 结果，尝试两个非 CE-specific 的 pair-signal geometry repair：

```text
B7lq = B7lp + crossBatchSGCap125
B7lr = B7en + crossBatchSGCap125, no hidden residual
```

本轮新增/修改：

```text
dgkan/models/fc_purekan_primitives.py:
  新增 B7lq / B7lr PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lq / B7lr
  modification audit 扩展到 B7lq / B7lr
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lq_lr_rational_linec_repair_20260522T231500Z
```

Efficiency / gradcheck：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|
| B7lq | 1.0573983072188982 | 0.7804761904761904 | 1 | 2.1193838506405882e-07 |
| B7lr | 1.0114783733781545 | 0.7652380952380953 | 1 | 1.8992328421063576e-07 |

Expression / task：

| candidate | A4 pass | A5 pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B7lq | 1 | 0 | -0.0078125 | -0.03515625 | 0.5 | 1 | 0 | 0 |
| B7lr | 1 | 0 | -0.0048828125 | -0.01953125 | 0.75 | 1 | 0 | 0 |

Line C：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | LineC interpretation |
|---|---:|---:|---:|---:|---|
| B7lq | 0.24113954903334045 | -0.09454097026635466 | 0.2574821412563324 | 0.21440568566322327 | coupling_collapse |
| B7lr | 0.24226277150629372 | -0.0934177477934014 | 0.2580016255378723 | 0.2150624394416809 | coupling_collapse |

Status JSON：

```text
Rational status = TaskBlocked
best_L3_manual_step_ratio = 1.0114783733781545
A4_expression_pass_candidates = [B7lq, B7lr]
A5_task_pass_candidates = []
no_fake_provenance_pass = 1
```

结论：stop-gradient batch-centered pair-logit cap 保住了 L3/A4，且 B7lr 在 ECE subgate 上为 1；但它没有修复 Line C。CouplingR2 仍约 `0.241-0.242`，与 B7lp 同型，A5 仍由 `worst_delta / near_pass / AUC` 阻塞。因此可以排除：

```text
pairNorm + batch-centered cap;
pairNorm + batch-centered cap + hiddenRatResidual005;
hidden residual 是 coupling collapse 唯一来源。
```

### 16.26 B7ls/B7lt：放开 w1 backbone 仍未修复 coupling，并显著放大 reservoir

B7lq/B7lr 仍 coupling collapse 后，按“不要继续调 cap，而要修 signal-channel geometry”的方向，测试只冻结 rational coefficients、放开 `w1` backbone：

```text
B7ls = B7lr, freezeRational instead of freezeBackbone
B7lt = B7lq, freezeRational instead of freezeBackbone
```

本轮新增/修改：

```text
dgkan/models/fc_purekan_primitives.py:
  新增 B7ls / B7lt PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7ls / B7lt
  modification audit 扩展到 B7ls / B7lt
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7ls_lt_rational_trainw1_linec_20260522T233000Z
```

Efficiency / gradcheck：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|
| B7ls | 0.8957816696092159 | 0.6719047619047619 | 1 | 1.8362217701906047e-07 |
| B7lt | 0.948394450419736 | 0.6871428571428572 | 1 | 1.803635711894458e-07 |

Expression / task：

| candidate | A4 pass | A5 pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B7ls | 1 | 0 | -0.0078125 | -0.03515625 | 0.5 | 1 | 0 | 0 |
| B7lt | 1 | 0 | -0.0048828125 | -0.01953125 | 0.75 | 1 | 0 | 0 |

Line C：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | LineC interpretation |
|---|---:|---:|---:|---:|---|
| B7ls | 0.24222736944499512 | -0.09345314985469999 | 0.30508798360824585 | 0.8271082639694214 | coupling_collapse |
| B7lt | 0.24110396412257928 | -0.09457655517711583 | 0.30599039793014526 | 0.8281915187835693 | coupling_collapse |

Status JSON：

```text
Rational status = TaskBlocked
best_L3_manual_step_ratio = 0.8957816696092159
A4_expression_pass_candidates = [B7ls, B7lt]
A5_task_pass_candidates = []
no_fake_provenance_pass = 1
```

结论：放开 `w1` backbone 并没有提高 CouplingR2，反而把 `RealSignalReservoirRatio` 从 B7lq/B7lr 的约 `0.214-0.215` 提高到约 `0.827-0.828`。这说明当前 trainable-w1 repair 是负结果：它让真实 task signal 更强地困在 reservoir。Rational 继续保持：

```text
Family success = 0
Rational status = TaskBlocked
denominator/derivative safety = normal
L3/A4 = 可打开
A5 = fail
Line C = coupling_collapse + reservoir trapping
```

下一步不应继续在 B7lp/lq/lr/ls/lt 周围叠 cap 或 residual。更合理的方向是重新设计 signal-channel primitive，使 one-window update 的 probe drift 可由 train drift 预测，同时显式压低 `RealSignalReservoirRatio`。候选方向应偏向：

```text
1. non-readout-only signal projection with reservoir penalty diagnostic, not CE loss;
2. lower-rank / shared pair signal channel that减少 per-sample reservoir trapping;
3. Line C first-pass smoke before A5 long triage;
4. 如果继续 trainable backbone，必须加入结构性 reservoir control，而不是简单 unfreeze w1。
```

### 16.27 B7lu/B7lv：shared pair-signal 通道是负结果，B7lv L3 过但 A4 表达失败，Line C 未改善

按 16.26 的下一步方向，我新增了低秩 shared pair-signal readout：

```text
B7lu = B7en pairNorm + crossSignalR4
B7lv = B7en pairNorm + crossSignalR8
```

修改内容：

```text
dgkan/models/fc_purekan_primitives.py:
  新增 crosssignalrK parser
  新增 cross_signal_readout / cross_signal_class 参数
  forward/manual_ce_forward_cache 使用 cross -> K shared signal -> centered class map
  manual_logits_backward_from_cache 增加 generic dL/dlogits VJP
  frozen_readout_features 使用 shared signal features，而不是 full pair-class readout
  新增 B7lu / B7lv PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lu / B7lv
  official L3 manual variant 集合加入 rational_flashkat_grouped_paircross_signal_gemm*
  modification audit 增加 B7lu / B7lv 说明
```

代码验证：

```text
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1283_b109_classic_family_functional_geometry.py dgkan/optim/manual_adamw.py
```

已通过。额外用 CPU 小样本实例化 B7lu，`manual_gradient_audit` 得到：

```text
manual_forward_available = 1
manual_backward_available = 1
grad_relerr_max = 1.4301085116130707e-07
grad_cos_min = 0.9999999403953552
output_max_abs_error = 0.0
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lu_lv_rational_crosssignal_20260522T160531Z
```

Efficiency / gradcheck：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|
| B7lu | 1.607531 | 0.921905 | 0 | 1.615907e-07 |
| B7lv | 0.8695541096998669 | 0.92 | 1 | 1.283831e-07 |

Expression / task：

| candidate | A4 pass | status |
|---|---:|---|
| B7lu | not run | L3 fused kernel efficiency fail |
| B7lv | 0 | A4_expression_gate_fail_after_l3_efficiency |

B7lv A4 summary：

| metric | value |
|---|---:|
| E1 B1 delta vs MLP | -1.083744 |
| E1 frozen R2 | 0.082173 |
| E2 B1 delta vs MLP | -0.085158 |
| E2 frozen R2 | 0.922334 |
| E6 B1 delta vs MLP | -1.066752 |
| E6 frozen R2 | -0.000145 |
| E8 B1 delta vs MLP | -1.066346 |
| E8 frozen R2 | 0.12657 |
| dead_basis_max | 0.0 |

Line C：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | LineC interpretation |
|---|---:|---:|---:|---:|---|
| B7lu | 0.240296 | -0.095385 | 0.256825 | 0.217661 | coupling_collapse |
| B7lv | 0.240265 | -0.095415 | 0.256798 | 0.217762 | coupling_collapse |

Basis / rational safety：

| candidate | den_min | den_p01 | den_condition | r_prime_p95 | r_double_prime_p95 | basis_condition_proxy |
|---|---:|---:|---:|---:|---:|---:|
| B7lu | 1.000006 | 1.000025 | 1.263823 | 0.995523 | 0.991978 | 13417949.0 |
| B7lv | 1.000006 | 1.000025 | 1.263823 | 0.995523 | 0.991978 | 13332934.0 |

结论：shared pair-signal K=4/K=8 是负结果。B7lv 虽然打开了 L3，但 A4 在 pairwise / rotated / random quadratic 上严重失败；B7lu 连 L3 都没过。更重要的是，Line C 的 `CouplingR2` 仍约 `0.240`，比 B7kc/B7en/B7lp/B7lq/B7lr/B7ls/B7lt 没有实质改善，`RealSignalReservoirRatio` 仍约 `0.218`。因此不能声称 shared low-rank pair-signal 修复了 reservoir trapping，也不能进入 A5 official claim。

### 16.28 B7kd/B7ke Line C first-pass：降 linear residual gain 也未修复 coupling，且 task 更差

为了避免只根据 B7lu/B7lv 低秩 pair channel 做结论，我补跑了已有的低 linear-residual gain bracket 的 Line C first-pass：

```text
B7kd = B7kc, linearresGain19200 -> linearresGain14400
B7ke = B7kc, linearresGain19200 -> linearresGain09600
```

本轮没有新增代码，只使用 16.2 已有 candidates，在当前 Line C wrapper 下重测。

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7kd_ke_rational_linec_gainbracket_20260522T160758Z
```

Efficiency / gradcheck：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|
| B7kd | 1.142603 | 0.770952 | 1 | 1.233153e-07 |
| B7ke | 1.126554 | 0.770952 | 1 | 1.312538e-07 |

Expression / task：

| candidate | A4 pass | A5 pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B7kd | 1 | 0 | -0.017578 | -0.039062 | 0.25 | 0 | 0 | 0 |
| B7ke | 1 | 0 | -0.035156 | -0.0625 | 0.0 | 0 | 0 | 0 |

Line C：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | LineC interpretation |
|---|---:|---:|---:|---:|---|
| B7kd | 0.245601 | -0.090080 | 0.300016 | 0.248189 | coupling_collapse |
| B7ke | 0.242259 | -0.093422 | 0.354657 | 0.299772 | coupling_collapse |

Basis / rational safety：

| candidate | den_min | den_p01 | den_condition | r_prime_p95 | r_double_prime_p95 | basis_condition_proxy |
|---|---:|---:|---:|---:|---:|---:|
| B7kd | 1.000006 | 1.000025 | 1.263823 | 0.995523 | 0.991978 | 14271269.0 |
| B7ke | 1.000006 | 1.000025 | 1.263823 | 0.995523 | 0.991978 | 13759350.0 |

结论：降 linear residual gain 不是修复。B7kd/B7ke 能 L3/A4，但 A5 明显比 B7kc/B7lp/B7lr 更差；Line C 仍是 `coupling_collapse`，且 `RealSignalReservoirRatio` 从 B7kc/B7en/B7lp/B7lq/B7lr 的约 `0.214-0.218` 上升到 `0.248/0.300`。这进一步排除“简单降低线性残差幅度即可降低 reservoir”的解释。

当前 Rational 总结：

```text
Family success = 0
Rational status = TaskBlocked 或 ExpressionBlocked，取决于本轮 candidate 集
denominator/derivative safety = normal
shared pair-signal = 未改善 Line C，且损伤 A4
lower linear residual gain = 未改善 Line C，且损伤 A5
trainable w1 = 未改善 Line C，且显著提高 reservoir
```

下一步若继续 Rational，不应再做单轴幅度/rank 调整。需要新的结构性假设，例如对 pair/readout signal 做 train/probe alignment 约束的可诊断结构，或者重新回到 B7kc/B7en/B7lp 级别的 A4/L3 anchor，设计能在不牺牲 A4 的情况下显式改变 one-window drift geometry 的 primitive。

### 16.29 B7lw/B7lx：PCA whitening 修复方向失败，R128 能过 L3/A4 但 Line C 更坏

根据 16.28 的结论，我实现了无标签 pair-feature PCA conditioning，目标是处理 pairNorm 无法消除的高 feature condition proxy：

```text
B7lw = B7en + crossPCAWhiteR96
B7lx = B7en + crossPCAWhiteR128
```

修改内容：

```text
dgkan/models/fc_purekan_primitives.py:
  新增 crosspcawhiterK / crosspcarK parser
  用 x_for_stats 的无标签 pair features 计算 PCA basis
  crossPCAWhite 使用 inverse singular value whitening
  forward/manual_ce_forward_cache/manual_logits_backward_from_cache 接入 cross_pca_readout
  修复小样本 SVD 有效秩小于 requested PCA rank 时的 shape mismatch
  新增 B7lw / B7lx / B7ly / B7lz PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py:
  family plan 增加 B7lw / B7lx / B7ly / B7lz
  official L3 manual variant 集合加入 rational_flashkat_grouped_paircross_pca_gemm*
  modification audit 增加 PCA whitening 与 non-whitening 说明
```

代码验证：

```text
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1283_b109_classic_family_functional_geometry.py
```

已通过。CPU 小样本审计中，B7lw/B7lx 的 manual gradient audit 均通过；shape mismatch 修复后：

```text
B7lw grad_relerr_max = 1.8518264255362737e-07
B7lx grad_relerr_max = 2.1189231347307214e-07
output_max_abs_error = 0.0
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7lw_lx_rational_pca_whiten_20260522T161504Z
```

Efficiency / gradcheck：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|
| B7lw | 1.052640 | 0.948571 | 1 | 1.192976e-07 |
| B7lx | 1.0079492483263017 | 0.993333 | 1 | 1.284405e-07 |

Expression / task：

| candidate | A4 pass | A5 pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B7lw | 0 | not run | | | | | | |
| B7lx | 1 | 0 | -0.006836 | -0.023438 | 0.5 | 0 | 0 | 0 |

B7lx A4 summary：

| metric | value |
|---|---:|
| E1 B1 delta vs MLP | -0.108269 |
| E1 frozen R2 | 0.940526 |
| E2 B1 delta vs MLP | -0.022801 |
| E6 B1 delta vs MLP | -0.118157 |
| E6 frozen R2 | 0.931126 |
| E8 B1 delta vs MLP | -0.177456 |
| E8 frozen R2 | 0.910101 |

Line C：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | candidate_CEp99 | LineC interpretation |
|---|---:|---:|---:|---:|---:|---|
| B7lw | 0.130944 | -0.204736 | 0.256695 | 0.218970 | 286.371429 | coupling_collapse |
| B7lx | 0.130221 | -0.205459 | 0.256848 | 0.218822 | 248.525604 | coupling_collapse |

结论：PCA whitening 是明确负结果。B7lx 能通过 L3/A4，但 Line C 的 `CouplingR2` 从此前 Rational 常见的约 `0.240-0.245` 进一步跌到约 `0.130`，且 `candidate_CEp99` 爆到 `248-286`。这说明 inverse singular value whitening 在放大低能 pair directions，造成严重 logit 几何失真。该结果不能作为修复，只能作为“不要 whiten pair spectrum”的反证。

### 16.30 B7ly/B7lz：PCA rotation 去掉 logit 爆炸，但仍未修复 Line C

为了隔离 B7lw/B7lx 的失败是否来自 whitening 放大，我保留同一个无标签 PCA basis，但移除 inverse singular value scale：

```text
B7ly = B7lw without whitening, crossPCAR96
B7lz = B7lx without whitening, crossPCAR128
```

CPU 小样本 manual gradient audit：

```text
B7ly grad_relerr_max = 1.4226291966679128e-07
B7lz grad_relerr_max = 1.2417774541972904e-07
output_max_abs_error = 0.0
```

Artifact：

```text
results/v12_9_b314_functional_classic_family/v129_b7ly_lz_rational_pca_rotate_20260522T161738Z
```

Efficiency / gradcheck：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | official_efficiency_pass | grad_relerr_max |
|---|---:|---:|---:|---:|
| B7ly | 0.786595 | 0.948571 | 1 | 1.192976e-07 |
| B7lz | 0.7787012727569469 | 0.993333 | 1 | 1.257999e-07 |

Expression / task：

| candidate | A4 pass | A5 pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B7ly | 0 | not run | | | | | | |
| B7lz | 1 | 0 | 0.000977 | -0.023438 | 0.75 | 0 | 0 | 0 |

B7lz A4 summary：

| metric | value |
|---|---:|
| E1 B1 delta vs MLP | -0.115604 |
| E1 frozen R2 | 0.939705 |
| E2 B1 delta vs MLP | -0.021789 |
| E6 B1 delta vs MLP | -0.118629 |
| E6 frozen R2 | 0.930828 |
| E8 B1 delta vs MLP | -0.177389 |
| E8 frozen R2 | 0.911377 |

Line C：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | candidate_CEp99 | LineC interpretation |
|---|---:|---:|---:|---:|---:|---|
| B7ly | 0.239636 | -0.096045 | 0.257508 | 0.219989 | 3.644662 | coupling_collapse |
| B7lz | 0.239816 | -0.095865 | 0.257416 | 0.219492 | 3.644571 | coupling_collapse |

结论：去掉 whitening 后 logit 爆炸消失，B7lz 比 B7lx 的 A5 summary 明显更健康（`mean_delta = 0.000977`, `near_pass_rate = 0.75`），但它仍未通过 A5，且 Line C 回到此前 Rational 常见的 coupling collapse：`CouplingR2 ≈ 0.240`，`RealSignalReservoirRatio ≈ 0.219`。所以 PCA rotation/truncation 不是 Line C 修复，只是一个可 L3/A4、但仍 TaskBlocked 的变体。

当前补充结论：

```text
PCA whitening = 负结果，放大低能方向，Line C 更坏
PCA rotation/truncation = 去掉 logit 爆炸，但 Line C 与 B7en/B7lp 同型 collapse
Rational family success = 0
classic_family_pass_count = 0
```

到这里，已尝试并排除的 Rational repair 包括：pairNorm、hidden residual、batch-centered cap、trainable w1、shared low-rank pair signal、linear residual gain 降幅、PCA whitening、PCA rotation。下一步若继续推进，需要一个新的结构性机制，而不是再围绕 pair readout 做 rank/gain/normalization bracket。

### 16.31 B7ma-B7mh：output-scale geometry 可改善 Line C，但不是当前 Rational family success

基于 B7lz 的结果，我补了一个新的输出几何方向，而不是继续只调 pair readout 的 rank/gain：

```text
B7ma = B7lz + samplewise stop-gradient logit RMS normalization
B7mb = B7lp + samplewise stop-gradient logit RMS normalization
B7mc = B7ma 的 batch-level RMS 版本
B7md = B7mb 的 batch-level RMS 版本
B7me/B7mf = B7md 的 batch RMS scale 1.50 / 2.00
B7mg/B7mh = B7md 的 batch RMS residual mix 0.25 / 0.50
```

修改内容：

```text
dgkan/models/fc_purekan_primitives.py
  新增 logitRMSNormSG / logitBatchRMSNormSG / logitBatchRMSMixSG parser
  forward/manual_ce_forward_cache 末端加入 detached RMS factor
  manual_logits_backward_from_cache 对 generic dL/dlogits 乘同一个 detached factor
  新增 B7ma-B7mh PrimitiveSpec

experiments/run_v1283_b109_classic_family_functional_geometry.py
  family plan 增加 B7ma-B7mh
  official L3 manual variant 集合增加对应 logit RMS suffix
  modification audit 记录 output-scale geometry repair，不改 CE/loss/data/sampler/class weight/teacher/dataset branch/gates
```

Correctness audit：

```text
py_compile 通过
B7ma grad_relerr_max = 2.1441020692236634e-07, output_max_abs_error = 0.0
B7mb grad_relerr_max = 2.376377352675263e-07, output_max_abs_error = 0.0
B7mc grad_relerr_max = 1.6094172394787165e-07, output_max_abs_error = 0.0
B7md grad_relerr_max = 1.6387987500365853e-07, output_max_abs_error = 0.0
B7me grad_relerr_max = 1.6455975071494322e-07, output_max_abs_error = 0.0
B7mf grad_relerr_max = 1.9631939096598217e-07, output_max_abs_error = 0.0
B7mg grad_relerr_max = 1.5579006173993548e-07, output_max_abs_error = 0.0
B7mh grad_relerr_max = 1.5065621994381218e-07, output_max_abs_error = 0.0
```

Artifacts：

```text
results/v12_9_b314_functional_classic_family/v129_b7ma_mb_rational_logitrms_20260522T162705Z
results/v12_9_b314_functional_classic_family/v129_b7mc_md_rational_logitbatchrms_20260522T163008Z
results/v12_9_b314_functional_classic_family/v129_b7me_mf_rational_logitbatchrms_scale_20260522T163347Z
results/v12_9_b314_functional_classic_family/v129_b7mg_mh_rational_logitbatchrms_mix_20260522T163726Z
```

Efficiency / task：

| candidate | L3 step_ratio_q90 | memory_ratio_q90 | L3 pass | A4 pass | A5 pass | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B7ma | 1.3381785592422517 | 0.9933333333333333 | 0 | not run | not run | | | | | | |
| B7mb | 1.2853982988484955 | 0.770952380952381 | 0 | not run | not run | | | | | | |
| B7mc | 1.0463852496016943 | 0.9933333333333333 | 1 | 1 | 0 | -0.07682291666666667 | -0.125 | 0.0 | 0 | 0 | 0 |
| B7md | 1.024077662193385 | 0.770952380952381 | 1 | 1 | 0 | -0.07356770833333333 | -0.09375 | 0.0 | 0 | 0 | 0 |
| B7me | 1.1916723575773893 | 0.770952380952381 | 1 | 1 | 0 | -0.06901041666666667 | -0.109375 | 0.0 | 0 | 0 | 0 |
| B7mf | 1.200194593432953 | 0.770952380952381 | 1 | 1 | 0 | -0.064453125 | -0.08203125 | 0.0 | 0 | 0 | 0 |
| B7mg | 1.2815851986408002 | 0.770952380952381 | 0 | not run | not run | | | | | | |
| B7mh | 1.2828951554159596 | 0.770952380952381 | 0 | not run | not run | | | | | | |

Line C first-pass autopsy：

| candidate | CouplingR2 | delta vs MLP | NoiseSignalLeak | RealSignalReservoirRatio | candidate_CEp99 | interpretation |
|---|---:|---:|---:|---:|---:|---|
| MLP | 0.3356805192996951 | 0.0 | 0.4323427677154541 | 0.08000694215297699 | | reference |
| B7ma | 0.29729283213111246 | -0.03838768716858265 | 0.08944284170866013 | 0.13996051251888275 | 4.206838130950928 | better than B7lz, but L3 closed |
| B7mb | 0.30036688098371433 | -0.03531363831598078 | 0.09083676338195801 | 0.1422976851463318 | 4.206360340118408 | better than B7lp, but L3 closed |
| B7mc | 0.3075903065943283 | -0.02809021270536682 | 0.11378300935029984 | 0.32411879301071167 | 4.271028995513916 | L3/A4 open, A5 task collapse |
| B7md | 0.30423514595657497 | -0.03144537334312014 | 0.11395398527383804 | 0.32556167244911194 | 4.281782150268555 | L3/A4 open, A5 task collapse |
| B7me | 0.3113662178976532 | -0.0243143014020419 | 0.07250837236642838 | 0.2987658381462097 | 5.850865364074707 | best CouplingR2 among this bracket, A5 still bad |
| B7mf | 0.3038084795984016 | -0.03187203970129351 | 0.025636106729507446 | 0.9019861221313477 | 7.436211585998535 | task/Line C reservoir worsens |
| B7mg | 0.27182621511257743 | -0.06385430418711768 | 0.2577161192893982 | 0.24316151440143585 | 3.7187211513519287 | L3 closed; weak Line C |
| B7mh | 0.30690323936160946 | -0.02877727993808565 | 0.26056787371635437 | 0.2355165332555771 | 3.782304048538208 | L3 closed; no task evidence |

结论：output-scale geometry 是一个真实信号，不是无效噪声。Samplewise/batch RMS 能把 Rational Line C `CouplingR2` 从 B7lz 的约 `0.240` 提到 `0.297-0.311`，其中 B7me 的 `CouplingR2 = 0.3113662178976532` 是本组最好值。但它没有完成 family success：samplewise B7ma/B7mb 因 L3 step gate 关闭；batch B7mc/B7md/B7me/B7mf 虽打开 L3/A4，却使 A5 task 明显失败；residual mix B7mg/B7mh 又因 L3 关闭，且没有给出更好的 Line C/task 组合。

因此不能声称 Rational 已修复。当前最准确的状态是：

```text
official_functional_success = 0
classic_family_pass_count = 0
Rational status = TaskBlocked / KernelBlocked by variant
best new Line C signal = B7me CouplingR2 0.3113662178976532, still below MLP 0.3356805192996951 and A5 fail
no fake data / no proxy row / no CPU offload / no CE loss or data protocol modification was used
```

下一步不应再继续沿同一个 output RMS normalization 做细碎 scale/mix bracket；这个方向已经证明“能改善 Line C，但会破坏 task 或 efficiency”。若继续 Rational，应换结构性机制，例如让 task margin 与 Line C coupling 分离的双通道 readout/orthogonalized residual，而不是继续对最终 logits 做全局重标定。
