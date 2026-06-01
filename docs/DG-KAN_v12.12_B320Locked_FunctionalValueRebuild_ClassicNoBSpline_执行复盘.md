# DG-KAN v12.12 B320Locked FunctionalValueRebuild ClassicNoBSpline 执行复盘

生成时间：`2026-05-23T13:37:01Z`

## 1. 执行入口

```text
script = experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py
command = experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py --out-dir results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm3_branch_rebalance_3x3_20260523T2135 --report-path docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md --run-bm3-probe 1 --probe-device cuda:1 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-train-size 1024 --probe-val-size 512 --probe-test-size 512 --functional-batch-size 16 --bm3-strengths 0.005,0.01,0.02
plan_doc = docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_独立分析与下一步计划.md
out_dir = results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm3_branch_rebalance_3x3_20260523T2135
v1211_functional_dir = auto
```

本轮先执行 v12.12 Batch 1：`A0 B320 exact hardening summary rebuild`、`B0 P3-to-P4 autopsy table`、`C0 Line C value rule implementation`、`D0 family status freeze/rewrite`。这些结果全部来自已落盘真实 artifacts；没有 fake/proxy/CPU offload；没有 teacher、distillation、loss modification、sampler/class weight 或 dataset-name branch。

## 2. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py` | 新增 v12.12 Batch 1 wrapper | 读取真实 v12.10/v12.11 artifacts，生成 v1212 anchor hardening、P3/P4 autopsy、Line C Pareto value、family status、provenance、hash 和复盘文件；不运行 fake 数据，不把派生 artifact 冒充新训练。 |
| `docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md` | 新增 v12.12 复盘文件 | 记录计划理解、执行命令、source artifacts、关键指标、route 和后续修复方向，便于后人复现。 |

## 3. B320 Anchor

```text
B320_anchor_locked = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = 0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
fail_reason = 
```

## 4. Functional Value Autopsy

```text
v1212_p3_p4_autopsy_rows = 180
old_score_spearman_to_p4_gain = 0.06405518470831038
new_linec_score_spearman_to_p4_gain = 0.18625523432899613
old_score_pearson_to_p4_gain = -0.007065898088334216
new_linec_score_pearson_to_p4_gain = 0.012352855744075323
p4_strict_pass_rows = 0
linec_pareto_pass_rows = 0
do_not_promote_functional = 1
```

解释：`p3_linec_pareto_score` 是硬规则通过项比例，用于诊断相关性；promotion 仍只允许 hard Pareto pass，不允许训练黑箱 selector。

## 5. Line C Value Rule

```text
linec_rows = 182
pareto_pass_rows = 0
coupling_open_rows = 92
noise_release_rows = 0
reservoir_release_rows = 3
no_op_rows = 102
fail_reason_counts = {"CouplingR2_delta<0.02": 90, "NoiseSignalLeak_delta>-0.01": 182, "RealSignalReservoirRatio_delta>-0.01": 179, "no_op_geometry_delta<0.03": 102, "CEp99_delta>0.05": 1}
```

## 6. Classic Family Status

| family | status | best_task_candidate | blocker |
|---|---|---|---|
| `Rational` | `TaskBlocked` | `B7kd-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3` | `official L3 and A4 pass exist, but A5 task gate failed` |
| `Chebyshev` | `TaskBlocked` | `B3ao-ChebyKAN-K3-h112-inputcrossL4P128-linearraw010-tritonL3-gradbuf` | `official L3 and A4 pass exist, but A5 task gate failed` |
| `Wavelet` | `TaskBlocked` | `B5n-HatWaveletKAN-K4-inputcrossL4P128-linearraw010-tritonL3` | `official L3 and A4 pass exist, but A5 task gate failed` |
| `RBF` | `ExpressionBlocked` | `` | `official L3 pass exists, but A4 expression gate failed` |
| `Fourier` | `ExpressionBlocked` | `` | `official L3 pass exists, but A4 expression gate failed` |
| `BSpline` | `RejectedForThisVersion` | `` | `B-spline frozen by v12.12 plan` |

## 7. Batch 2 Mechanism Probes

### 7.1 B-M1 SNR / Signal-Channel Probe

执行目录：

```text
results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm1_snr_probe_3x3_routefix_20260523T2128
```

执行命令：

```text
conda run -n kan python experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py \
  --out-dir results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm1_snr_probe_3x3_routefix_20260523T2128 \
  --report-path docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md \
  --run-bm1-probe 1 --probe-device cuda:0 \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 \
  --probe-train-size 1024 --probe-val-size 512 --probe-test-size 512 \
  --functional-batch-size 16
```

摘要：

```text
bm1_rows = 54
bm1_pareto_pass_rows = 3
bm1_expected_rows_per_candidate = 9
bm1_full_3x3_pareto_candidate_count = 0
bm1_partial_pareto_candidates = BM1a-HardSNRGate@window5:1/9,BM1b-SoftSNRGate@window5:1/9,BM1d-SignalWeightedAdamW@window5:1/9
bm1_noise_release_rows = 3
bm1_reservoir_release_rows = 6
bm1_best_candidate = BM1b-SoftSNRGate
bm1_best_delta_score = 0.4889880365043845
bm1_fail_reason_counts = {'NoiseSignalLeak_delta>-0.01': 51, 'RealSignalReservoirRatio_delta>-0.01': 48, 'control_gap<0.005': 24}
```

如果 `bm1_enabled=0`，表示本次只执行 Batch 1。若开启 B-M1，它只使用真实 B320 batch 逐样本梯度估计 SNR gate，并以同一 Line C Pareto gate 判定；只有同一候选在全部 3×3 row 通过时才允许进入 P4，不把单个 dataset/seed 的 partial positive 写成 official survivor。

B-M1 partial pass 原始 row：

```text
BM1a-HardSNRGate / Fashion-MNIST seed=0 / window=5:
  SNR_positive_fraction=0.46328813347497344
  gradient_norm_removed=0.13898775266723495
  cos_to_adamw=0.8610121853501534
  holdout_loss_ratio=0.9979586015995325
  CouplingR2_delta=0.1599646811226645
  NoiseSignalLeak_delta=-0.012052297592163086
  RealSignalReservoirRatio_delta=-0.01344752311706543
  CEp99_delta=0.0004115104675292969
  control_gap_vs_best=0.08197024996271729

BM1b-SoftSNRGate / Fashion-MNIST seed=0 / window=5:
  SNR_positive_fraction=0.46328813347497344
  gradient_norm_removed=0.12725208236534113
  cos_to_adamw=0.9238812689972392
  holdout_loss_ratio=0.9980022825896946
  CouplingR2_delta=0.16610733182736603
  NoiseSignalLeak_delta=-0.011934816837310791
  RealSignalReservoirRatio_delta=-0.014312267303466797
  CEp99_delta=-0.00013017654418945312
  control_gap_vs_best=0.0889776448538202

BM1d-SignalWeightedAdamW / Fashion-MNIST seed=0 / window=5:
  SNR_positive_fraction=0.46328813347497344
  gradient_norm_removed=0.13900957006319026
  cos_to_adamw=0.8610047879141783
  holdout_loss_ratio=0.9979591175167392
  CouplingR2_delta=0.15996041926977123
  NoiseSignalLeak_delta=-0.01205354928970337
  RealSignalReservoirRatio_delta=-0.013447880744934082
  CEp99_delta=0.0004119873046875
  control_gap_vs_best=0.08196634573769268
```

分析：B-M1 证明 SNR gating 在一个 Fashion-MNIST seed/window 上可以同时移动 coupling/noise/reservoir，但 `1/9` 覆盖不足，且 51/54 row 仍 fail `NoiseSignalLeak_delta>-0.01`。因此它是 partial positive evidence，不是 B2/P4 survivor。

### 7.2 B-M3a Branch Rebalance + Logit Matching Probe

执行目录：

```text
results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm3_branch_rebalance_3x3_20260523T2135
```

执行命令：

```text
conda run -n kan python experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py \
  --out-dir results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm3_branch_rebalance_3x3_20260523T2135 \
  --report-path docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md \
  --run-bm3-probe 1 --probe-device cuda:1 \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 \
  --probe-train-size 1024 --probe-val-size 512 --probe-test-size 512 \
  --functional-batch-size 16 --bm3-strengths 0.005,0.01,0.02
```

修改审计：

```text
file = experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py
change = added B-M3a branch_scale/bias probe-only maintenance
reason = test function-preserving coordinate maintenance without touching quad_proj, optimizer state, loss, data, sampler, or dataset branch
```

摘要：

```text
bm3_rows = 27
bm3_pareto_pass_rows = 0
bm3_logit_safe_rows = 8
bm3_noise_release_rows = 0
bm3_reservoir_release_rows = 0
bm3_best_candidate = BM3a-BranchRebalanceLogitMatch-s0.02
bm3_best_delta_score = 0.7711390346392496
bm3_fail_reason_counts = {'NoiseSignalLeak_delta>-0.01': 27, 'RealSignalReservoirRatio_delta>-0.01': 27, 'logit_max_abs_drift>0.05': 19, 'holdout_loss_ratio>1.01': 6}
```

最佳但失败的代表 row：

```text
BM3a-BranchRebalanceLogitMatch-s0.02 / KMNIST seed=2:
  logit_match_mse=0.002086275490000844
  logit_max_abs_drift=0.13549089431762695
  KL_before_after=0.0002034392673522234
  holdout_loss_ratio=1.001701392730818
  CouplingR2_delta=0.7698809892042978
  NoiseSignalLeak_delta=0.0024562478065490723
  RealSignalReservoirRatio_delta=-0.005616992712020874
  CEp99_delta=-0.057579994201660156
  fail_reason=logit_max_abs_drift>0.05;NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01
```

分析：B-M3a 非常容易打开 CouplingR2，但没有释放 NoiseSignalLeak / Reservoir；较强 strength 还违反 `logit_max_abs_drift<=0.05`。因此 branch-scale rebalancing 是 coupling-only coordinate move，不是 functional value success。下一步若继续 B-M3，应增加显式 logit least-squares correction 或进入 B-M4 moment/state transport；不能把 coupling-only high score 写成成功。

## 8. Route

```text
route = R2-B320AnchorLockedFunctionalValueStillInvalid
official_functional_success = 0
classic_family_pass_count = 0
next_recommended_action = do not promote functional; run B-M1/B-M2/B-M3/B-M4 mechanism probes with Line C Pareto
```

本轮 Batch 1 的结论是：B320 anchor 可以继续作为 v12.12 functional anchor；旧 functional value 仍不能 promotion；Line C Pareto value 已落成可审计 CSV，但当前没有 functional 候选通过 hard Pareto。下一步应进入 Batch 2，只测试 B-M1/B-M2/B-M3/B-M4 这四类机制，不再扩大旧 F14/F15 小网格。

## 9. 复现命令

```text
conda run -n kan python experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py \
  --out-dir results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm3_branch_rebalance_3x3_20260523T2135 \
  --report-path docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md
```

本轮完整命令：

```text
experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py --out-dir results/v12_12_b320locked_functional_value_rebuild_classic_nobspline/v1212_bm3_branch_rebalance_3x3_20260523T2135 --report-path docs/DG-KAN_v12.12_B320Locked_FunctionalValueRebuild_ClassicNoBSpline_执行复盘.md --run-bm3-probe 1 --probe-device cuda:1 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-train-size 1024 --probe-val-size 512 --probe-test-size 512 --functional-batch-size 16 --bm3-strengths 0.005,0.01,0.02
```

若迁移机器后 source artifacts 缺失，必须先复现 v12.10 linec64 full/P4 与 v12.11 constrained mechanism probe；不要用空表或手填数据替代。

## 10. Hash

| artifact | sha256 |
|---|---|
| `v1212_b320_anchor_hardening.csv` | `c2d7b6532734df63b14603b631626d37c5fce0fc3b8e6b814be167b4a1ad8379` |
| `v1212_bm3_branch_rebalance_probe.csv` | `87078eee8ee8032d0756cddc0895e3ae9938ebdb168f7476b39397a6a94272f0` |
| `v1212_bm3_branch_rebalance_probe_summary.csv` | `535fd9733939011728b617135ec419bf30caf36bf13ff3dc76a3ad0acf6422a1` |
| `v1212_family_efficiency.csv` | `2467a033ebfc2559c57f70592e82e701b048c0288395b5b996b1bee0469a7852` |
| `v1212_family_expression.csv` | `e1200e42ce05b7a4f4e7f4a7b2f812bd002fce63052ab0f27d2b4dab36078de8` |
| `v1212_family_failure_table.csv` | `35f90d8fc445929e2412dc7ba7dfaaf897a95678cbceab37eba66af3ad5f4605` |
| `v1212_family_linec.csv` | `ac3d78c16f13e1ea4cc838d14555cf11996478098e4cd457e0eae7102158afb3` |
| `v1212_family_manifest.csv` | `b117801b9231d86c1e8e2a8f22c6c086826ca610da9e03c5889aa419c7cdcd55` |
| `v1212_family_status.csv` | `b117801b9231d86c1e8e2a8f22c6c086826ca610da9e03c5889aa419c7cdcd55` |
| `v1212_family_status.json` | `9585e751b3bdaa09f00b029506dc717c7cbec5634be25e7cbc9f88437e3201d5` |
| `v1212_family_task_triage.csv` | `e7d2f274379b287486d835529d1462b6cf710064583f803316c09ec592bbc5b1` |
| `v1212_linec_value_summary.csv` | `059f150ec4eebbcbeb30455b167873c0f916af60f7c10b89552b2630d8e24e98` |
| `v1212_linec_window_diagnostics.csv` | `c98eb9644d53d9fe59dd0fd15c7d425794aa340b473dd831d69c4c693b705b3e` |
| `v1212_p3_p4_autopsy.csv` | `7814e5e863b1e3a771c19817806970b99417f2416263976ac25f54d1013de5eb` |
| `v1212_p3_p4_autopsy_summary.csv` | `9664a6e16dd90b006be94104f04355814ece5dc5556e5ff523b48aa5bc47f2e1` |
| `v1212_provenance_audit.csv` | `b417d3f450ca9aba1e18f455e6026a94e5be9c0bd96c360ec24b89150b13f750` |
| `v1212_route_decision.json` | `59783eed584fbae5af474bef21489dc5c3c71237602a9ef16055a0834ca7ab6f` |
| `v1212_run_manifest.json` | `5db9bfe4548236e313dd2480c17deac7f881237e8480b47a098ccce214f9f169` |
