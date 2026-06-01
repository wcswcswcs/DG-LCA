# DG-KAN v12.11 B320 FunctionalMechanism ClassicNoBSpline 执行复盘

生成时间：`2026-05-23T07:11:16Z`

## 1. 执行入口

```text
script = experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py
command = experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_b320_mechanism_probe_constrained_3x3_20260523T1604 --report-path docs/DG-KAN_v12.11_B320_FunctionalMechanism_ClassicNoBSpline_执行复盘.md --run-p3v2-mechanism-probe 1 --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2
plan_doc = docs/DG-KAN_v12.11_B320_FunctionalMechanism_ClassicNoBSpline_独立分析与下一步计划.md
anchor_dir = /home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_taskminus_20260523T1405
b320_source_dir = /home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_full_20260523T1352
raw_dir = /home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_full_20260523T1352/raw_v1283_reuse
p4_dirs = /home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_taskminus_20260523T1405,/home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_quad_20260523T1405,/home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_negative_20260523T1405,/home/chengshun.wang/DG-LCA/results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_direct_20260523T1405
out_dir = results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_b320_mechanism_probe_constrained_3x3_20260523T1604
```

说明：本轮 v12.11 执行器复用已经真实跑完的 v12.10 `linec64` raw/P4 artifacts，生成 v12.11 contract 和 bridge autopsy；如果启用 `--run-p3v2-mechanism-probe 1`，会额外运行真实 B320 P3v2 机制探针。没有新增 fake/proxy 数据；没有把派生 artifact 冒充新训练。

## 2. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 新增/扩展 v12.11 wrapper | 读取真实 v12.10/v1283 artifacts，生成 v1211 anchor/LineC/P3v2/P4/bridge/family/provenance/hash/复盘文件；可选运行 F-M1/F-M2/F-M3/F-M5 P3v2 机制探针；不改模型、loss、sampler、class weight 或 raw runner。 |

## 3. B320 Anchor Lock

```text
anchor_status = B320Locked
B320_anchor_strict_pass = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = 0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
CouplingR2 = 0.2381078772319669
NoiseSignalLeak = 0.015539980493485928
LineC_nontearing_pass = 1
```

## 4. Functional P3v2 / P4 Bridge

```text
P3v2_candidate_count = 182
P3v2_pareto_pass_count = 0
P3v2_legacy_artifact_rows = 29
P3v2_mechanism_probe_rows = 153
P3_to_P4_bridge_rows = 180
P3_score_P4_gain_corr = -0.007065898088334216
P3_score_P4_strict_AUC = 
B320_bestFunctional_P4_pass = 0 / 36
```

P4 fail reason 计数：

```text
{
  "CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01": 1,
  "AUC_time_delta>0;CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01": 2,
  "AUC_time_delta>0;CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;amortized_overhead>1.05;control_gap_vs_best<=0": 12,
  "AUC_time_delta>0;CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;control_gap_vs_best<=0": 15,
  "CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;control_gap_vs_best<=0": 3,
  "AUC_time_delta>0;CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;amortized_overhead>1.05": 2,
  "AUC_time_delta>0;CEp99_delta>0.05;NoiseSignalLeak_delta>-0.01;amortized_overhead>1.05;control_gap_vs_best<=0": 1
}
```

## 5. Route

```text
route = R2-P3ScoreNotPredictive
official_functional_success = 0
classic_family_pass_count_excluding_bspline = 0
next_recommended_action = discard current P3 score as promotion score; rebuild functional value from Line C Pareto
```

结论：B320 anchor 已锁定；当前旧 P3 score 不能作为 promotion score。按 v12.11 计划，下一步不是继续 F14/F15 lambda 小网格，而是用 Line C Pareto 重建 functional value，并优先尝试 signal-reservoir / noise-projected / tail-safe 机制。

## 6. 复现说明

以后迁移项目时，优先执行：

```text
conda activate kan
python experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/<new_run> \
  --run-p3v2-mechanism-probe 1 \
  --anchor-dir results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_taskminus_20260523T1405 \
  --b320-source-dir results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_full_20260523T1352 \
  --raw-dir results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_full_20260523T1352/raw_v1283_reuse \
  --p4-dirs results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_taskminus_20260523T1405,results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_quad_20260523T1405,results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_negative_20260523T1405,results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_direct_20260523T1405
```

若 source artifacts 不存在，必须先按 v12.10 linec64 full + p3fix P4 bracket 复现 raw/P4，再跑本脚本；不要用空表或手填数据替代。

## 7. Hash

| artifact | sha256 |
|---|---|
| `fig_A_b320_anchor_scorecard.svg` | `2823229f8cf9c2f1290f956b31a1088c701276e4ea1b986fcb4473974c0f6c5b` |
| `fig_A_b320_auc_seed_matrix.svg` | `7d83484c9d67422aff380381f74b643e2ed8b0d094d92fc5a0dfbecf04483138` |
| `fig_B_event_accept_reject_timeline.svg` | `ac842f4c18a57bf02cb48ba7b50e8e90e19201086bf0ab9033679b97235a701c` |
| `fig_B_functional_pareto.svg` | `5309ffd59ce55b1321f7f3cc397fb1799bff8435bb10f2d3c7c25a5ff884a139` |
| `fig_B_noise_leak_vs_coupling.svg` | `dffb4907c5aaf9769086b4687ab6f2269e739c00daff613c09ee88b782a7f0fd` |
| `fig_B_p3_to_p4_prediction.svg` | `e7756a4c415286809c33c7f58ab4303d492300be8c621298eca814539cacac37` |
| `fig_B_p4_fail_reason_heatmap.svg` | `b2c7745f0a1f0890e3b70b96f7124d082f08051aad98f40069dc82752109dee5` |
| `fig_C_noise_leak_vs_tail.svg` | `75d83757359e5f028e390511ed260361f66f422cdd3a3787064783a69fc80d4d` |
| `fig_C_reservoir_vs_auc_time.svg` | `eda30dcecb44a1e6a686f3adf291a5021cd75c3205b7a930b6cd4a933d703872` |
| `fig_C_signal_reservoir_spectrum.svg` | `2cd7afafd0e7fe890af447c3b6c428ec41c1b26a074fb2eb9cef951846acfaeb` |
| `fig_D_family_bottleneck_waterfall.svg` | `358375b7039e69083fbff03921851280b871a4a508886832e30ab32550b90d69` |
| `fig_D_family_efficiency_expression_pareto.svg` | `87827689a6ead546cf8ce9d0811750f1ffaf4fce0fd440f2830db9277f1bd3ee` |
| `fig_D_family_linec_radar.svg` | `9a9b4fbaa6009b5db1607c913605f54d72128f397cf78e53d53cd35fdc0cdde1` |
| `fig_D_family_status_matrix.svg` | `8a1c29bfd246524d609401d997ea7270a717dc20c3617531612791aef999c9f7` |
| `fig_v1211_gate_ladder.svg` | `0bc784e4e7880dc5d1feb157338a4c7d99ad7dbfcaef41872c9595465fc70131` |
| `v1211_b320_anchor_lock.csv` | `7617fbf838af20794e754c178aeee15ff5c34acde9c9c318b9f97de039cf7f44` |
| `v1211_b320_efficiency_profile.csv` | `d4d944e745b3a9ead4a9894dcde2b6f370df40f9920194efee242863bf0c7ffb` |
| `v1211_family_efficiency.csv` | `46889f3da285c1a3dd47aaadcd83bb7a370d81540bc06427d4f26f903de0458c` |
| `v1211_family_expression.csv` | `4dd74daa05b400705acf62196f81fc6bb02f5d5bf1aa7e411a1395751df3b524` |
| `v1211_family_failure_table.csv` | `31bd4d35508ad365a04d9299bc3bf43d4d8b4643828e1d073dec9d32e6d12a1b` |
| `v1211_family_gradcheck.csv` | `96324035f43e4df4e93b3219bf52096fc10fd83a80a34aa0ed165c27354ec9bf` |
| `v1211_family_linec.csv` | `507b404deb5b07f7917d313ae47a378844935db54113fdf6e4cebdf5b0f51144` |
| `v1211_family_manifest.csv` | `951e1a0b45d76e7b884ee20858e3ee31fbd94f0dbd65d55465483ab388b436e2` |
| `v1211_family_status.json` | `631f54f10ada592d95b077e9895cb4a3c548bfc1c28b18ab9ef90cf9570a9ff3` |
| `v1211_family_task_triage.csv` | `879ced9b93a35bcc4342836178deb58fc4a40847bdd10be783c1c698d5fac1a1` |
| `v1211_functional_failure_table.csv` | `b8338cde16bd5f311dda2e7dc5933746a03eb497af8f99ab6e8a540d5708ef44` |
| `v1211_functional_p3v2_candidate.csv` | `5e47dfb52a087af199bbdbc8d839bba94f7c5aa07d32a4a5eb270786a7576bb2` |
| `v1211_functional_p4_short_run.csv` | `af72175fe3aece83577dbc95b020ead9c55be220b863aa351949232372b069f6` |
| `v1211_linec_diagnostics.csv` | `4bfece312dff6e56512aaee600406d02a0bfb2c26e7b3562489dc1ddb4da9ac6` |
| `v1211_p3_p4_bridge_autopsy.csv` | `34868f2816b59c79d78bac65f786f4c777ac3750bd57d0a325ac51fbad2f4281` |
| `v1211_provenance_audit.csv` | `7fa177a0b5a45a26901ad450c31e8733819f116231c8f493c153214701ec1a07` |
| `v1211_route_decision.json` | `4b6dd2f2fa15c5e75b0644a32bf4d22428063e93a492e10c45bb2c6dfc390a72` |
| `v1211_run_manifest.json` | `4163204a2ea63bc84ab9b8e2ca06c344dbd6d75ce84d05d39ca8936a09eae425` |

## 8. 2026-05-23 继续推进：真实 P3v2 机制探针

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 增加 `--run-p3v2-mechanism-probe` | 在复用 v12.10 artifact 的基础上，额外真实构造 B320、加载数据、计算 task VJP delta、basis/SNR/orthogonal/branch/noise/tail/projector functional delta，并用 Line C/P3v2 Pareto gate 评价 F-M1/F-M2/F-M3/F-M4/F-M5。没有改 loss、sampler、class weight、teacher/distillation 或 dataset-name branch。 |
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 增加 F-M4 branch null-space 探针 | 用局部 logits 线性化在 direct/quad branch-scale 二维子空间求最小 logit-drift eigen direction，测试 function-preserving branch reparameterization；负向候选只用于排除 eigenvector 符号任意性，不做 lambda 小网格。 |
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 增加 VJP noise/tail projection 探针 | 新增 `FM2-TaskNoiseOrthogonalProjection`、`FM2-RealMinusNoiseVJPProjection`、`FM3-TaskNoiseTailOrthogonalProjection`，用 shuffled-label noise VJP 与 tail-risk VJP 做方向投影，验证是否比 residual 族更接近 v12.11 的 signal/noise/reservoir 机制。 |
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 增加 projector-level release 探针 | 新增 `FM1-ProjectorReservoirRelease`、`FM2-ProjectorNoiseLeakRelease`、`FM1M2-ProjectorJointRelease` 和 sign-check，先由 Line C gradient sketch 构造 `P_sig/P_res`，再对 `RealSignalReservoirRatio` / `NoiseSignalLeak` 的可微 proxy 求 VJP。 |
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 增加固定 5-event projector sequence | 新增 `FM1M2-ProjectorJointRelease5Event`，每个 event 重新计算当前模型上的 projector-level joint release，用于验证是否只是单次 step budget 不足；固定 5-event 对齐 v12.11 one/five-step bridge，不做数据集特化或 lambda 小网格。 |
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 增加 constrained low-rank projector solve | 新增 `FM1M2-ConstrainedLowRankRelease`，在 reservoir/noise/tail 三个 projector proxy 梯度组成的低秩子空间内解固定目标的最小二乘方向，目标是同时压低 reservoir/noise 并限制 tail。 |
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 修正 route 判定优先级 | 若未来出现 P3v2 Pareto-pass 机制，应转入 P4 re-entry，而不是继续被旧 P3 score bridge 相关性锁死在 R2；本轮因 P3v2 pass=0，route 仍为 R2。 |

### 执行命令

最终记录采用 constrained low-rank projector solve 后的 3×3 run：

```text
conda run -n kan python experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_b320_mechanism_probe_constrained_3x3_20260523T1604 \
  --report-path docs/DG-KAN_v12.11_B320_FunctionalMechanism_ClassicNoBSpline_执行复盘.md \
  --run-p3v2-mechanism-probe 1 \
  --probe-device cuda:0 \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 0,1,2
```

### 真实结果

```text
total_v1211_functional_p3v2_rows = 182
legacy_artifact_rows = 29
new_mechanism_probe_rows = 153
probe_datasets = MNIST,Fashion-MNIST,KMNIST
probe_seeds = 0,1,2
probe_pareto_pass = 0 / 153
constrained_lowrank_rows = 9
constrained_lowrank_pareto_pass = 0 / 9
route = R2-P3ScoreNotPredictive
```

失败原因计数：

```text
64 x CouplingR2_delta<0.02;NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01;control_gap<0.005
59 x NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01;control_gap<0.005
30 x NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01
```

全部 153 条新机制探针都没有 P3v2 Pareto pass，因此按 v12.11 计划不进入 P4。

### Constrained Low-Rank 结果

```text
Constrained CouplingR2_delta: min=0.021930382861586217, max=0.14142073459834548, mean=0.06423320274330117
Constrained NoiseSignalLeak_delta: min=-0.0014591366052627563, max=0.003987159579992294, mean=0.00036814436316490173
Constrained RealSignalReservoirRatio_delta: min=-0.0016444921493530273, max=0.0028873980045318604, mean=5.267063776652018e-05
Constrained control_gap_vs_adamwparallel: min=-0.07315530945929849, max=0.05196148170531778, mean=-0.00663400972007375
Constrained holdout_loss_ratio: min=0.999380644197238, max=1.000609400883091, mean=1.000170792595637
Constrained CEp99_delta: min=-0.0021600723266601562, max=0.0024127960205078125, mean=0.000324143303765191
Constrained orthogonal_fraction: min=0.9910159989606783, max=0.9999618905006875, mean=0.9980869594061842
Constrained rows with CouplingR2_delta >= 0.02 = 9 / 9
Constrained rows with NoiseSignalLeak_delta <= -0.01 = 0 / 9
Constrained rows with RealSignalReservoirRatio_delta <= -0.01 = 0 / 9
```

最好的 constrained rows：

```text
Fashion-MNIST seed2:
  delta_score=0.1430652267476985
  CouplingR2_delta=0.14142073459834548
  NoiseSignalLeak_delta=-0.0003233104944229126
  RealSignalReservoirRatio_delta=-0.0016444921493530273
  CEp99_delta=4.4345855712890625e-05
  control_gap_vs_adamwparallel=0.03954934525978426
  fail=NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01

MNIST seed2:
  delta_score=0.07220169590311898
  CouplingR2_delta=0.07271137522059334
  NoiseSignalLeak_delta=-0.0002067089080810547
  RealSignalReservoirRatio_delta=0.0004717111587524414
  CEp99_delta=-0.0002067089080810547
  control_gap_vs_adamwparallel=0.05196148170531778
  fail=NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01
```

### 分析结论

constrained low-rank solve 没有解决 blocker。它证明三件事：

```text
1. CouplingR2 gate 已经不难打开，9/9 constrained rows 都 >= 0.02。
2. Tail/holdout 基本安全，CEp99 和 holdout_loss_ratio 没有明显爆炸。
3. 即使用 fixed target 在 reservoir/noise/tail proxy 梯度子空间里求解，NoiseSignalLeak 与 RealSignalReservoirRatio 的真实下降仍只有 1e-3 量级，不到 -0.01 gate。
```

因此 functional 主线当前的失败已不是旧 P3 score、不是 residual 方向、不是 sign、不是 step budget、不是低秩组合、也不是 tail 爆炸。剩余 blocker 是更根本的：

```text
Line C proxy-gradient 可达的局部方向，不能把真实 P3v2 gate 中的 NoiseSignalLeak / RealSignalReservoirRatio 降到阈值。
```

这时继续在 functional 线上盲目增加候选会变成无意义的小变体。按 v12.11 计划，合理分流是：

```text
functional_status = B320BaseOnlyFunctionalStillBlocked
next = classic family expression/task repair, while keeping projector-level results as diagnostic evidence
```

本轮没有虚构数据，没有使用 dataset-name branch，没有改 loss/sampler/class weight/teacher/distillation。

## 9. 2026-05-23 继续推进：Line D Rational Focused Family Repair

### 当前 family 选择依据

检查 v12.11 artifact：

```text
Rational: TaskBlocked
  L3 official efficiency exists = 1
  A4 expression pass candidates = B7lp, B7me
  A5 task pass candidates = []

Chebyshev / Fourier / RBF: ExpressionBlocked
Wavelet: KernelBlocked
BSpline: frozen / rejected for this version
```

因此 Line D 下一步优先 Rational，而不是先扩 Cheby/Fourier/RBF 表达式或 Wavelet kernel。Rational 已经最接近 FamilyPass：缺的是 A5 fused task triage。

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 在 CUDA device 解析后增加 `torch.cuda.set_device(device)` | focused run 使用 `--device cuda:1` 时，Triton kernel 仍可能沿用默认 CUDA device，导致 `Pointer argument cannot be accessed from Triton (cpu tensor?)`。该修复只保证多 GPU Triton launch 与 tensor device 一致，不改变模型、loss、数据、采样、gate 或候选。 |

首次 focused run：

```text
out_dir = results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rational_b7kc_b7kd_b7ke_3x3_20260523T1615
result = failed before task metrics
error = ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```

修复后重跑命令：

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rational_b7kc_b7kd_b7ke_3x3_20260523T1619 \
  --device cuda:1 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --family-candidate-ids B7kc...,B7kd...,B7ke... \
  --family-promotion-ids B7kc...,B7kd...,B7ke... \
  --family-linec-ids B7kc...,B7kd...,B7ke...
```

完整 candidate ids：

```text
B7kc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3
B7kd-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3
B7ke-RationalKAT-flashgroup-G16-h32-linearresGain09600-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Rational status = TaskBlocked
Rational A4_expression_pass_candidates = B7kc, B7kd, B7ke
Rational A5_task_pass_candidates = []
no_fake_provenance_pass = 1
```

效率结果：

```text
B7kc L3 step_ratio_q90=0.6180388297429606, memory_ratio_q90=0.770952380952381, official_efficiency_pass=1
B7kd L3 step_ratio_q90=0.7639622532525742, memory_ratio_q90=0.770952380952381, official_efficiency_pass=1
B7ke L3 step_ratio_q90=0.6217247935659349, memory_ratio_q90=0.770952380952381, official_efficiency_pass=1
best_L3_manual_step_ratio = 0.6180388297429606
```

A5 task summary：

```text
B7kc:
  mean_delta=-0.0030381944444444445
  worst_delta=-0.0625
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=0
  A5_task_pass=0

B7kd:
  mean_delta=-0.00021701388888888888
  worst_delta=-0.060546875
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=1
  A5_task_pass=0

B7ke:
  mean_delta=-0.0026041666666666665
  worst_delta=-0.041015625
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=1
  A5_task_pass=0
```

Family Line C autopsy：

```text
B7kc:
  CouplingR2_delta_vs_mlp=0.036106056773343886
  NoiseSignalLeak_delta_vs_mlp=-0.4931740164756775
  RealSignalReservoirRatio_delta_vs_mlp=0.015183523297309875
  interpretation=linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory

B7kd:
  CouplingR2_delta_vs_mlp=0.03140870001736695
  NoiseSignalLeak_delta_vs_mlp=-0.44938310980796814
  RealSignalReservoirRatio_delta_vs_mlp=0.04214000701904297
  interpretation=real_signal_reservoir_high

B7ke:
  CouplingR2_delta_vs_mlp=0.024754827396564982
  NoiseSignalLeak_delta_vs_mlp=-0.3310309052467346
  RealSignalReservoirRatio_delta_vs_mlp=0.11193053424358368
  interpretation=real_signal_reservoir_high
```

### 分析结论

这轮不是 FamilyPass。Rational 的状态仍是：

```text
Rational = L3 passed + A4 passed + A5 failed
```

但 blocker 进一步收窄：

```text
1. B7kc/B7kd/B7ke 的 L3 efficiency 都明确通过，Rational 当前不是 kernel efficiency blocker。
2. 三个候选都进入 A4 pass 集合，Rational 当前也不是 expression blocker。
3. A5 失败来自任务轨迹：mean_delta 仍略负，worst_delta 到 -0.04/-0.06，near_pass_rate 只有 2/3。
4. B7kc 的 Line C 不明显坏，说明它的几何不是主要失败点；更像 calibration / optimizer trajectory blocker。
5. 降低 linear residual gain 到 144/96 没有打开 A5，且 Line C reservoir 变高；因此继续只降 gain 不是合理方向。
```

下一步按 v12.11 Line D 推荐继续 Rational A5 repair，但应从 `B7kc` 分支做 calibration/trajectory 机制，而不是继续改 kernel 或表达式。可验证候选是当前 specs 中已有的 `B7kf/B7kg/B7kh` 低 cap/tanh residual 族或 `B7lq/B7lr` pair-signal geometry 隔离族；仍必须保持 no CE-specific tuning、no dataset branch、no sampler/class-weight change。

## 10. 2026-05-23 继续推进：Rational B7kf/B7kg/B7kh Calibration Follow-up

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rational_b7kf_b7kg_b7kh_3x3_20260523T1626 \
  --device cuda:2 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --family-candidate-ids B7kf...,B7kg...,B7kh... \
  --family-promotion-ids B7kf...,B7kg...,B7kh... \
  --family-linec-ids B7kf...,B7kg...,B7kh...
```

完整 candidate ids：

```text
B7kf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3
B7kg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3
B7kh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap075-freezeBackbone-hiddenBias-manualAdamW-L3
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Rational status = TaskBlocked
Rational A4_expression_pass_candidates = B7kf, B7kg, B7kh
Rational A5_task_pass_candidates = []
```

效率结果：

```text
B7kf L3 step_ratio_q90=0.8316219266149164, memory_ratio_q90=0.7804761904761904, official_efficiency_pass=1
B7kg L3 step_ratio_q90=0.767125189096943,  memory_ratio_q90=0.7804761904761904, official_efficiency_pass=1
B7kh L3 step_ratio_q90=0.7464732463446686, memory_ratio_q90=0.7804761904761904, official_efficiency_pass=1
best_L3_manual_step_ratio = 0.7464732463446686
```

A5 task summary：

```text
B7kf:
  mean_delta=-0.0028211805555555555
  worst_delta=-0.0625
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=0
  A5_task_pass=0

B7kg:
  mean_delta=-0.00390625
  worst_delta=-0.068359375
  near_pass_rate=0.5555555555555556
  auc_step_ok=0
  auc_time_ok=0
  A5_task_pass=0

B7kh:
  mean_delta=-0.005425347222222222
  worst_delta=-0.0546875
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=0
  A5_task_pass=0
```

Family Line C autopsy：

```text
B7kf:
  CouplingR2_delta_vs_mlp=0.03624594867832098
  NoiseSignalLeak_delta_vs_mlp=-0.4940708428621292
  RealSignalReservoirRatio_delta_vs_mlp=0.01487797498703003
  interpretation=linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory

B7kg:
  CouplingR2_delta_vs_mlp=0.03624594867832098
  NoiseSignalLeak_delta_vs_mlp=-0.4940708428621292
  RealSignalReservoirRatio_delta_vs_mlp=0.01487797498703003
  interpretation=linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory

B7kh:
  CouplingR2_delta_vs_mlp=0.03624594867832098
  NoiseSignalLeak_delta_vs_mlp=-0.494070902466774
  RealSignalReservoirRatio_delta_vs_mlp=0.01487797498703003
  interpretation=linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory
```

### 分析结论

低 cap/tanh residual follow-up 没有打开 Rational A5。它进一步排除：

```text
1. Rational 不是 L3 efficiency blocker：B7kf/B7kg/B7kh 都过。
2. Rational 不是 A4 expression blocker：三者都进入 A4 pass 集。
3. Line C 也不是主要坏点：三者 autopsy 均为 not_obviously_bad。
4. A5 仍被 task trajectory gate 卡住，尤其 auc_step_ok=0、near_pass_rate 只有 0.56-0.67。
5. 低 cap 越强并未改善 A5，B7kg/B7kh 反而 mean/near 更差；继续降低 cap 不是推荐方向。
```

因此 Rational 当前最精确 blocker：

```text
Rational_TaskBlocked_by_A5_AUC_Trajectory
```

下一步不应继续只改 cap/gain。若继续 Line D，应转向不增加 direct-gradient 成本的 optimizer/trajectory primitive，或评估 `B7lq/B7lr` 是否能隔离 pair-signal geometry；如果它们也失败，Rational family pass 在当前 v12.11 scope 内应保持 blocked，并转到 Chebyshev/Fourier/RBF 的 A4 expression repair。

## 11. 2026-05-23 继续推进：Rational B7lq/B7lr Pair-Signal Isolation

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rational_b7lq_b7lr_3x3_20260523T1632 \
  --device cuda:3 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --family-candidate-ids B7lq...,B7lr... \
  --family-promotion-ids B7lq...,B7lr... \
  --family-linec-ids B7lq...,B7lr...
```

完整 candidate ids：

```text
B7lq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3
B7lr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Rational status = TaskBlocked
Rational A4_expression_pass_candidates = B7lq, B7lr
Rational A5_task_pass_candidates = []
```

效率结果：

```text
B7lq L3 step_ratio_q90=0.8440095401834405, memory_ratio_q90=0.7804761904761904, official_efficiency_pass=1
B7lr L3 step_ratio_q90=0.8222350080317898, memory_ratio_q90=0.7652380952380953, official_efficiency_pass=1
best_L3_manual_step_ratio = 0.8222350080317898
```

A5 task summary：

```text
B7lq:
  mean_delta=-0.006076388888888889
  worst_delta=-0.060546875
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=0
  A5_task_pass=0

B7lr:
  mean_delta=-0.006944444444444444
  worst_delta=-0.05078125
  near_pass_rate=0.6666666666666666
  auc_step_ok=0
  auc_time_ok=0
  A5_task_pass=0
```

Family Line C autopsy：

```text
B7lq:
  CouplingR2_delta_vs_mlp=0.03805183618755004
  NoiseSignalLeak_delta_vs_mlp=-0.4908222258090973
  RealSignalReservoirRatio_delta_vs_mlp=0.011962890625
  interpretation=linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory

B7lr:
  CouplingR2_delta_vs_mlp=0.03805081450293257
  NoiseSignalLeak_delta_vs_mlp=-0.4906003922224045
  RealSignalReservoirRatio_delta_vs_mlp=0.012379869818687439
  interpretation=linec_not_obviously_bad_task_blocker_likely_calibration_or_optimizer_trajectory
```

### 收敛结论

Rational 的 v12.11 Line D focused repair 已排除以下方向：

```text
B7kc/B7kd/B7ke: bounded rational hidden residual + linear gain bracket
B7kf/B7kg/B7kh: tanh residual + stop-gradient cap bracket
B7lq/B7lr: pairNorm + pair-signal hidden-residual isolation
```

所有这些候选均：

```text
L3 efficiency pass = 1
A4 expression pass = 1
A5 task pass = 0
Line C autopsy = not obviously bad 或 reservoir high 但不是 primary gate opener
```

因此 Rational 的当前结论应写为：

```text
Rational_TaskBlocked_by_A5_AUC_Trajectory
FamilyPass = 0
```

不应把任何 focused run 冒充 full family pass；也不应继续降低 cap/gain 或做 dataset-specific trajectory 修补。下一步若继续 v12.11 classic family，应转到 Chebyshev/Fourier/RBF 的 A4 expression repair 或重新定义 Rational 的 optimizer/trajectory primitive，但这已经超出当前这些候选的证据范围。

## 12. 2026-05-23 继续推进：Chebyshev / Fourier A4 Focused Screening

### 执行命令

Chebyshev：

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_cheby_a4_inputcross_3x3_20260523T1718 \
  --device cuda:0 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B3p,B3q,B3r,B3s,B3v,B3w \
  --family-promotion-ids B3p,B3q,B3r,B3s,B3v,B3w \
  --family-linec-ids B3p,B3q,B3r,B3s,B3v,B3w
```

Fourier：

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_fourier_a4_k3k4_3x3_20260523T1718 \
  --device cuda:1 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B4h,B4i,B4j,B4k,B4l,B4m \
  --family-promotion-ids B4h,B4i,B4j,B4k,B4l,B4m \
  --family-linec-ids B4h,B4i,B4j,B4k,B4l,B4m
```

完整 candidate ids 见对应 run manifest；这里用短名只为复盘可读性。

### Chebyshev 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Chebyshev status = TaskBlocked
Chebyshev A4_expression_pass_candidates = B3s, B3v, B3w
Chebyshev A5_task_pass_candidates = []
best_L3_manual_step_ratio = 0.48436917049860834
```

效率结果：

```text
B3p L3 step_ratio_q90=0.585071670762617,  memory_ratio_q90=1.0885714285714285, official_efficiency_pass=0
B3q L3 step_ratio_q90=0.6527612546309104, memory_ratio_q90=0.9942857142857143, official_efficiency_pass=1
B3r L3 step_ratio_q90=0.5489303759803218, memory_ratio_q90=0.9957142857142857, official_efficiency_pass=1
B3s L3 step_ratio_q90=0.4948526037211923, memory_ratio_q90=0.9980952380952381, official_efficiency_pass=1
B3v L3 step_ratio_q90=0.48436917049860834, memory_ratio_q90=1.0276190476190477, official_efficiency_pass=1
B3w L3 step_ratio_q90=0.5422642816542512, memory_ratio_q90=1.0276190476190477, official_efficiency_pass=1
```

A4 expression summary：

```text
B3q A4=0: E1=-0.06924891471862793, E2=-0.019244074821472168, E6=-0.10132235288619995, E8=-0.11566013097763062
B3r A4=0: E1=-0.06976127624511719, E2=-0.02671760320663452, E6=-0.09360247850418091, E8=-0.1041225790977478
B3s A4=1: frozen_key_r2_ge_089=1, dead_basis_max=0.0, E1=-0.07277673482894897, E2=-0.018294334411621094, E6=-0.08836597204208374, E8=-0.09368455410003662
B3v A4=1: frozen_key_r2_ge_089=1, dead_basis_max=0.0, E1=-0.06978631019592285, E2=-0.018048465251922607, E6=-0.08662915229797363, E8=-0.09668225049972534
B3w A4=1: frozen_key_r2_ge_089=1, dead_basis_max=0.0, E1=-0.065035879611969, E2=-0.021075129508972168, E6=-0.08615607023239136, E8=-0.09819453954696655
```

A5 task summary：

```text
B3s A5=0: mean_delta=-0.4906684027777778, worst_delta=-0.568359375, near_pass_rate=0.0
B3v A5=0: mean_delta=-0.2847222222222222, worst_delta=-0.375, near_pass_rate=0.0
B3w A5=0: mean_delta=-0.2764756944444444, worst_delta=-0.373046875, near_pass_rate=0.0
```

解释：Chebyshev 从 `ExpressionBlocked` 推进到了 `TaskBlocked`，这是实质进展；但 task 崩塌很严重，不能写 FamilyPass。B3v/B3w 的 linear residual 能缓解崩塌，但远未接近 A5 gate。

### Fourier 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Fourier status = ExpressionBlocked
Fourier A4_expression_pass_candidates = []
best_L3_manual_step_ratio = 0.4156278432666729
```

效率结果：

```text
B4h L3 step_ratio_q90=0.4844417238386426, memory_ratio_q90=1.0409523809523809, official_efficiency_pass=1
B4i L3 step_ratio_q90=0.45367213159383,   memory_ratio_q90=0.8509523809523809, official_efficiency_pass=1
B4j L3 step_ratio_q90=0.4156278432666729, memory_ratio_q90=0.709047619047619,  official_efficiency_pass=1
B4k L3 step_ratio_q90=0.5107340173995142, memory_ratio_q90=0.5671428571428572, official_efficiency_pass=1
B4l L3 step_ratio_q90=0.6542136728477673, memory_ratio_q90=0.7561904761904762, official_efficiency_pass=1
B4m L3 step_ratio_q90=0.5736431943439012, memory_ratio_q90=0.5671428571428572, official_efficiency_pass=1
```

A4 expression summary：

```text
B4h A4=0: E1=-0.13705766201019287, E2=-0.012044429779052734, E6=-0.11068427562713623, E8=-0.18082863092422485
B4i A4=0: E1=-0.1853874921798706,  E2=-0.01865828037261963,  E6=-0.17481482028961182, E8=-0.2149871587753296
B4j A4=0: E1=-0.20633631944656372, E2=-0.020805180072784424, E6=-0.1658303141593933,  E8=-0.24199461936950684
B4k A4=0: E1=-0.21750831604003906, E2=-0.013938605785369873, E6=-0.16906559467315674, E8=-0.25398099422454834
B4l A4=0: E1=-0.22409266233444214, E2=-0.018288016319274902, E6=-0.38009536266326904, E8=-0.3977762460708618
B4m A4=0: E1=-0.1838397979736328,  E2=-0.019759833812713623, E6=-0.37485307455062866, E8=-0.44506365060806274
```

解释：Fourier 的 kernel/efficiency 已经很好，但 A4 仍明显失败。K4 variants 对 E6/E8 更差；继续单纯提高 Fourier order 不是当前推荐方向。

### 结论

```text
Chebyshev: ExpressionBlocked -> TaskBlocked, but A5 collapse is severe.
Fourier: remains ExpressionBlocked.
FamilyPass = 0
```

下一步合理继续点是 Chebyshev task repair，而不是 Fourier K-order 增强；因此已启动 B3t/B3u/B3x follow-up，继续检查 rank96/112 和 raw linear residual 是否能保留 A4 同时改善 A5。

## 13. 2026-05-23 继续推进：Chebyshev B3t/B3u/B3x Task Follow-up

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_cheby_task_b3t_b3u_b3x_3x3_20260523T1739 \
  --device cuda:2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B3t,B3u,B3x \
  --family-promotion-ids B3t,B3u,B3x \
  --family-linec-ids B3t,B3u,B3x
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Chebyshev status = TaskBlocked
Chebyshev A4_expression_pass_candidates = B3x
Chebyshev A5_task_pass_candidates = []
best_L3_manual_step_ratio = 0.6223447293636483
```

效率结果：

```text
B3t L3 step_ratio_q90=0.642352162673516,  memory_ratio_q90=1.09, official_efficiency_pass=0
B3u L3 step_ratio_q90=0.6223447293636483, memory_ratio_q90=0.9976190476190476, official_efficiency_pass=1
B3x L3 step_ratio_q90=0.6891098587472593, memory_ratio_q90=1.0276190476190477, official_efficiency_pass=1
```

A4 / A5：

```text
B3u A4=0:
  E1=-0.07669097185134888
  E2=-0.01957416534423828
  E6=-0.0932232141494751
  E8=-0.09301954507827759

B3x A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.065632164478302
  E2=-0.014262080192565918
  E6=-0.08103132247924805
  E8=-0.0999915599822998
  A5=0
  mean_delta=-0.064453125
  worst_delta=-0.09765625
  near_pass_rate=0.0
  auc_step_ok=0
  auc_time_ok=0
```

Family Line C autopsy：

```text
B3t:
  CouplingR2_delta_vs_mlp=-0.27099423154460234
  NoiseSignalLeak_delta_vs_mlp=-0.6892019920051098
  RealSignalReservoirRatio_delta_vs_mlp=0.18474049866199493
  interpretation=coupling_collapse

B3u:
  CouplingR2_delta_vs_mlp=-0.2698427096863478
  NoiseSignalLeak_delta_vs_mlp=-0.6449524536728859
  RealSignalReservoirRatio_delta_vs_mlp=0.28089894354343414
  interpretation=coupling_collapse

B3x:
  CouplingR2_delta_vs_mlp=0.015446717337771587
  NoiseSignalLeak_delta_vs_mlp=-0.6217205300927162
  RealSignalReservoirRatio_delta_vs_mlp=0.6040502041578293
  interpretation=real_signal_reservoir_high
```

### 分析结论

B3x 是目前 Chebyshev 最接近的 task repair：

```text
B3s mean_delta=-0.4906684027777778
B3v mean_delta=-0.2847222222222222
B3w mean_delta=-0.2764756944444444
B3x mean_delta=-0.064453125
```

但 B3x 仍远未通过 A5，且 Line C 显示 `RealSignalReservoirRatio_delta_vs_mlp=0.6040502041578293`，说明 raw linear residual 修复了部分任务轨迹，却把真实信号更多推入 reservoir。继续方向不应是扩大 raw branch，而是降低 B3x raw-linear 强度并验证是否能在保留 A4 的同时降低 reservoir 与 A5 损害。因此新增 B3an/B3ao 低强度 bracket，并单独跑 3×3 follow-up。

## 14. 2026-05-23 继续推进：Chebyshev B3an/B3ao Raw-Linear Strength Bracket

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B3an` / `B3ao` Chebyshev candidate specs | 两者沿用 B3x 的 K3 h112 inputcross rank128 + Triton gradbuf 路径，只把 raw-linear residual strength 固定为 0.25 / 0.10，用于验证 B3x 的 task 改善是否可在更低 reservoir 冲击下保留。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B3an` / `B3ao` 加入 `_family_plan()` | 让 focused run 能按原有 L3/A4/A5 gate 调度新候选。未改变 loss、sampler、class weight、teacher/distillation、dataset branch 或 gate。 |

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_cheby_rawlinear_b3an_b3ao_3x3_20260523T1748 \
  --device cuda:3 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B3an,B3ao \
  --family-promotion-ids B3an,B3ao \
  --family-linec-ids B3an,B3ao
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Chebyshev status = TaskBlocked
Chebyshev A4_expression_pass_candidates = B3ao
Chebyshev A5_task_pass_candidates = []
best_L3_manual_step_ratio = 0.6178294012355954
```

效率结果：

```text
B3an L3 step_ratio_q90=0.6178294012355954, memory_ratio_q90=1.120952380952381, official_efficiency_pass=0
B3ao L3 step_ratio_q90=0.6368389385354771, memory_ratio_q90=1.0276190476190477, official_efficiency_pass=1
```

A4 / A5：

```text
B3ao A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.06631195545196533
  E2=-0.021659135818481445
  E6=-0.08856058120727539
  E8=-0.09788310527801514
  A5=0
  mean_delta=-0.05555555555555555
  worst_delta=-0.09375
  near_pass_rate=0.0
  auc_step_ok=0
  auc_time_ok=0
```

Line C autopsy：

```text
B3an:
  CouplingR2_delta_vs_mlp=0.022565827163883445
  NoiseSignalLeak_delta_vs_mlp=-0.5757562518119812
  RealSignalReservoirRatio_delta_vs_mlp=0.57270847260952
  interpretation=real_signal_reservoir_high

B3ao:
  CouplingR2_delta_vs_mlp=0.03186174315116741
  NoiseSignalLeak_delta_vs_mlp=-0.416765421628952
  RealSignalReservoirRatio_delta_vs_mlp=0.572786495089531
  interpretation=real_signal_reservoir_high
```

### 分析结论

B3ao 比 B3x 略好，但仍不是成功：

```text
B3x  mean_delta=-0.064453125, worst_delta=-0.09765625, A5=0
B3ao mean_delta=-0.05555555555555555, worst_delta=-0.09375, A5=0
```

降低 raw-linear strength 没有解决核心 blocker：

```text
1. A5 task gate 仍失败，near_pass_rate 仍为 0。
2. RealSignalReservoirRatio_delta_vs_mlp 仍约 0.573，说明 reservoir high 没被修掉。
3. B3an 因 memory_ratio_q90=1.120952380952381 未过 official efficiency。
```

当前 Chebyshev 结论：

```text
Chebyshev_TaskBlocked_after_A4_open
best_current_candidate = B3ao
FamilyPass = 0
```

下一步如果继续 Chebyshev，需要新的 task-geometry primitive，而不是继续缩放 raw-linear residual。否则应转向 RBF/FastKAN 的 A4 expression repair 或更系统地重审 classic A5 task trajectory gate。

## 15. 2026-05-23 继续推进：RBF/FastKAN B2t/B2u/B2v Input-Cross A4 Repair

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 将已有 fixed input-cross sidecar 泛化给 RBF K4，并新增 `B2t` / `B2u` / `B2v` | RBF 原始 `B2s` L3 过但 E1/E6/E8 二次表达明显不足；本修改只增加固定 input-cross features + trainable readout，`B2v` 再加小 linear residual bracket。不改 loss、gate、sampler、class weight、teacher/distillation 或 dataset branch。 |
| `dgkan/models/fc_purekan_primitives.py` | 为 RBF input-cross manual CE path 增加显式 sidecar readout 梯度 | `manual_ce_forward_cache` 在 fused RBF logits 上叠加 input-cross logits；`manual_ce_backward_from_cache` 用同一个 `dL/dlogits` 更新 sidecar readout，保持对上游 loss/VJP 的结构可分离，不使用 CE-only 代理结论。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B2t` / `B2u` / `B2v` 加入 `_family_plan()`，并把 RBF input-cross variants 登记为 official fused L3 候选 | 让 focused run 继续使用原有 L3/A4 gate；只扩展候选注册，不改通过阈值。 |

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rbf_inputcross_b2t_b2u_b2v_3x3_20260523T1536 \
  --device cuda:0 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B2t,B2u,B2v \
  --family-promotion-ids B2t,B2u,B2v \
  --family-linec-ids B2t,B2u,B2v
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
RBF status = ExpressionBlocked
RBF A4_expression_pass_candidates = []
RBF A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B2t manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.5968650065750081
  memory_ratio_q90=1.0319047619047619
  official_efficiency_pass=1
  grad_relerr_max=2.0054692129178875e-07
  grad_cos_min=0.9999998807907104

B2u manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.6282700891672117
  memory_ratio_q90=1.01
  official_efficiency_pass=1
  grad_relerr_max=2.0604470307716838e-07
  grad_cos_min=0.9999999403953552

B2v manual_variant=rbf_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.604771393390114
  memory_ratio_q90=1.0395238095238095
  official_efficiency_pass=1
  grad_relerr_max=1.8457001260685502e-07
  grad_cos_min=0.9999998807907104
```

A4 expression：

```text
B2t A4=0:
  E1=-0.12794095277786255
  E2=-0.06740248203277588
  E6=-0.15417754650115967
  E8=-0.15362095832824707

B2u A4=0:
  E1=-0.10551226139068604
  E2=-0.06067854166030884
  E6=-0.14146482944488525
  E8=-0.14956265687942505

B2v A4=0:
  E1=-0.11363452672958374
  E2=-0.06286627054214478
  E6=-0.14546090364456177
  E8=-0.1506425142288208
```

失败表：

```text
B2t failure_code=A4_expression_gate_fail_after_l3_efficiency
B2u failure_code=A4_expression_gate_fail_after_l3_efficiency
B2v failure_code=A4_expression_gate_fail_after_l3_efficiency
action_recommended=repair family expression capacity without losing fused L3 efficiency; do not run A5 official for this family yet
```

### 分析结论

这轮 RBF 修复是有效但不充分：

```text
B2s baseline:
  E1=-0.17420965433120728
  E6=-0.2124583125114441
  E8=-0.23524999618530273

B2u best in this run:
  E1=-0.10551226139068604
  E6=-0.14146482944488525
  E8=-0.14956265687942505
```

`B2u` 证明 fixed input-cross/localrot2 确实改善了 RBF 的二次表达缺口，但离 A4 gate 仍很远；`B2v` 的 small linear residual 没有解决关键二次项，说明 blocker 不是线性 residual 缺失。当前 RBF 结论仍是：

```text
RBF_ExpressionBlocked_after_inputcross_repair
best_current_candidate = B2u
FamilyPass = 0
```

下一步不应推进 A5，也不应改数据或 loss；应在同一 RBF K4 fused path 上提高 fixed quadratic sidecar 的表达秩，优先验证 `projr256` 与 `projsq`/rotated-local pair 组合是否能继续缩小 E1/E6/E8 缺口，同时监控 L3 memory 是否仍 <=1.05。

## 16. 2026-05-23 继续推进：RBF/FastKAN B2w/B2x/B2y Rank256 Quadratic Sidecar

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B2w` / `B2x` / `B2y` RBF candidate specs | 沿用 B2u 的 RBF K4 fused path，把 fixed input-cross projection rank 从 128 提到 256；分别测试 localrot2、projected-square (`projsq`)、local rank 8。目标是直接修 E1/E6/E8，不改变 loss、gate、数据或优化目标。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B2w` / `B2x` / `B2y` 加入 `_family_plan()` | 让 focused run 按原有 L3/A4 gate 调度新候选；未新增任何 dataset-specific branch。 |

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rbf_rank256_b2w_b2x_b2y_3x3_20260523T1558 \
  --device cuda:1 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B2w,B2x,B2y \
  --family-promotion-ids B2w,B2x,B2y \
  --family-linec-ids B2w,B2x,B2y
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
RBF status = ExpressionBlocked
RBF A4_expression_pass_candidates = []
RBF A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B2w manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.5965240440699272
  memory_ratio_q90=1.0276190476190477
  official_efficiency_pass=1
  grad_relerr_max=1.8191994399785472e-07
  grad_cos_min=0.9999998807907104

B2x manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.5827233925743104
  memory_ratio_q90=1.0052380952380953
  official_efficiency_pass=1
  grad_relerr_max=2.2205558991572616e-07
  grad_cos_min=0.9999998807907104

B2y manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.5575649744009631
  memory_ratio_q90=1.006190476190476
  official_efficiency_pass=1
  grad_relerr_max=1.8235229504171002e-07
  grad_cos_min=0.9999999403953552
```

A4 expression：

```text
B2w A4=0:
  E1=-0.0685015320777893
  E2=-0.07272273302078247
  E6=-0.1966685652732849
  E8=-0.2467435598373413

B2x A4=0:
  E1=-0.1094176173210144
  E2=-0.07533818483352661
  E6=-0.19252163171768188
  E8=-0.22682052850723267

B2y A4=0:
  E1=-0.07912731170654297
  E2=-0.082774817943573
  E6=-0.19854480028152466
  E8=-0.2218306064605713
```

失败表：

```text
B2w failure_code=A4_expression_gate_fail_after_l3_efficiency
B2x failure_code=A4_expression_gate_fail_after_l3_efficiency
B2y failure_code=A4_expression_gate_fail_after_l3_efficiency
action_recommended=repair family expression capacity without losing fused L3 efficiency; do not run A5 official for this family yet
```

### 分析结论

rank256 没有解决 RBF A4，且暴露了新问题：

```text
B2u rank128/localrot2:
  E1=-0.10551226139068604
  E6=-0.14146482944488525
  E8=-0.14956265687942505

B2w rank256/localrot2:
  E1=-0.0685015320777893
  E6=-0.1966685652732849
  E8=-0.2467435598373413
```

也就是说，更高 rank 对 E1 有帮助，但 E6/E8 更差；`projsq` 也没有打开 random quadratic form。这个结果说明 blocker 不是“projection rank 不够”这么简单，而可能是 fixed random quadratic sidecar 的特征条件数/方向覆盖不稳定，或 RBF base 与 sidecar readout 在短程 expression fit 中互相干扰。

当前 RBF 结论更新为：

```text
RBF_ExpressionBlocked_after_rank256_sidecar
best_current_candidate = B2u for E6/E8 balance, B2w only improves E1
FamilyPass = 0
```

下一步应尝试正交化 fixed projection directions，验证是否能在不增加 rank、不改数据/loss 的前提下改善 E6/E8；如果正交化仍失败，应把 RBF 标为 `ExpressionBlocked`，转向其他 family 或回到 Chebyshev task-geometry primitive。

## 17. 2026-05-23 继续推进：RBF/FastKAN B2z/B2aa/B2ab Orthogonal Projection Repair

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 为 input-cross fixed projections 增加 `orthoproj` 选项 | 用 QR 正交化固定 projection directions，目标是降低 random projection sidecar 的病态相关；不改变 loss、gate、数据、teacher/distillation、sampler 或 dataset branch。 |
| `dgkan/models/fc_purekan_primitives.py` | 修复首版 `orthoproj` 在 expression 低维输入下的 shape bug | 首跑失败：`torch.linalg.qr(..., mode="reduced")` 在 input_dim < proj_rank 时返回列数小于 proj_rank，导致 `mat1 and mat2 shapes cannot be multiplied (128x20 and 132x1)`。修复为按块生成正交投影并拼接，保持请求的 projection feature 数。失败未计为实验成功，重新落盘到新目录。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B2z` / `B2aa` / `B2ab` RBF candidate specs | 分别测试 orthogonal rank128 localrot2、orthogonal rank256 localrot2、orthogonal projected-square rank128。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B2z` / `B2aa` / `B2ab` 加入 `_family_plan()` | 继续使用原有 L3/A4 gate 调度。 |

### 执行命令

首跑失败命令：

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rbf_orthoproj_b2z_b2aa_b2ab_3x3_20260523T1606 \
  --device cuda:2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B2z,B2aa,B2ab \
  --family-promotion-ids B2z,B2aa,B2ab \
  --family-linec-ids B2z,B2aa,B2ab
```

修复后重跑命令：

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_rbf_orthoproj_b2z_b2aa_b2ab_3x3_20260523T1611 \
  --device cuda:2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B2z,B2aa,B2ab \
  --family-promotion-ids B2z,B2aa,B2ab \
  --family-linec-ids B2z,B2aa,B2ab
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
RBF status = ExpressionBlocked
RBF A4_expression_pass_candidates = []
RBF A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B2z manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.6534921749468378
  memory_ratio_q90=1.0319047619047619
  official_efficiency_pass=1
  grad_relerr_max=1.8698725057220145e-07
  grad_cos_min=1.0

B2aa manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.6397248769608872
  memory_ratio_q90=1.0057142857142858
  official_efficiency_pass=1
  grad_relerr_max=2.0473602546644543e-07
  grad_cos_min=0.9999999403953552

B2ab manual_variant=rbf_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.5802115885764754
  memory_ratio_q90=1.01
  official_efficiency_pass=1
  grad_relerr_max=1.9080863467024756e-07
  grad_cos_min=0.9999998807907104
```

A4 expression：

```text
B2z A4=0:
  E1=-0.0990491509437561
  E2=-0.07071727514266968
  E6=-0.1632487177848816
  E8=-0.17441481351852417

B2aa A4=0:
  E1=-0.07173413038253784
  E2=-0.07158952951431274
  E6=-0.20155340433120728
  E8=-0.2418469786643982

B2ab A4=0:
  E1=-0.12800800800323486
  E2=-0.06563550233840942
  E6=-0.15369552373886108
  E8=-0.16399019956588745
```

失败表：

```text
B2z failure_code=A4_expression_gate_fail_after_l3_efficiency
B2aa failure_code=A4_expression_gate_fail_after_l3_efficiency
B2ab failure_code=A4_expression_gate_fail_after_l3_efficiency
action_recommended=repair family expression capacity without losing fused L3 efficiency; do not run A5 official for this family yet
```

### 分析结论

orthoproj 没有打开 RBF A4：

```text
B2u non-ortho rank128/localrot2:
  E1=-0.10551226139068604
  E6=-0.14146482944488525
  E8=-0.14956265687942505

B2z ortho rank128/localrot2:
  E1=-0.0990491509437561
  E6=-0.1632487177848816
  E8=-0.17441481351852417

B2ab ortho rank128/projsq:
  E1=-0.12800800800323486
  E6=-0.15369552373886108
  E8=-0.16399019956588745
```

正交化略改善 E1 或 projected-square 的部分稳定性，但没有解决 E6/E8；rank256 正交版本仍明显伤 E6/E8。因此，RBF/FastKAN 在当前 fixed-center K4 + fused RBF + input quadratic sidecar 路线下仍是 `ExpressionBlocked`，不能推进 A5，不能写成 FamilyPass。

当前 RBF 结论：

```text
RBF_ExpressionBlocked_after_inputcross_rank256_orthoproj
best_current_candidate = B2u for balanced E1/E6/E8
FamilyPass = 0
```

下一步不应继续堆 RBF fixed sidecar rank；更合理的方向是转向其他 family，或回到 Chebyshev 已 A4-open 的 `TaskBlocked`，设计 task-geometry primitive，而不是继续在 RBF 上重复容量扩张。

## 18. 2026-05-23 继续推进：Wavelet B5h Baseline Focused Run

### 修改审计

本节没有新增代码修改；只运行已有候选 `B5h-HatWaveletKAN-local-K4`，补齐 v12.11 里 Wavelet 的 L3/A4 baseline。

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_wavelet_b5h_3x3_20260523T1618 \
  --device cuda:3 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B5h \
  --family-promotion-ids B5h \
  --family-linec-ids B5h
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Wavelet status = ExpressionBlocked
Wavelet A4_expression_pass_candidates = []
Wavelet A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B5h manual_variant=hat_wavelet_k4_triton_l3_matmul
  step_ratio_q90=0.4472403196933529
  memory_ratio_q90=1.0266666666666666
  official_efficiency_pass=1
  grad_relerr_max=3.693167514029483e-07
  grad_cos_min=1.0
```

A4 expression：

```text
B5h A4=0:
  E1=-1.0245640277862549
  E2=-0.06694477796554565
  E6=-0.9987626075744629
  E8=-0.9816438555717468
```

失败表：

```text
B5h failure_code=A4_expression_gate_fail_after_l3_efficiency
action_recommended=repair family expression capacity without losing fused L3 efficiency; do not run A5 official for this family yet
```

### 分析结论

Wavelet 的 kernel/efficiency 已经不是 blocker，`B5h` 的 L3 非常快且 gradcheck 通过；真正 blocker 是表达结构，尤其 E1/E6/E8 二次项几乎完全失败：

```text
B5h:
  E1=-1.0245640277862549
  E6=-0.9987626075744629
  E8=-0.9816438555717468
```

这说明 local hat-wavelet fixed-center basis 本身对全局 pairwise / rotated quadratic 表达不足。下一步如果继续 Wavelet，应借鉴 RBF 的 input-cross repair，但预期必须先打开 A4；在 A4 失败前不能跑 A5，也不能写成 FamilyPass。

## 19. 2026-05-23 继续推进：Wavelet B5i/B5j/B5k Input-Cross A4 Repair

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 将 fixed input-cross sidecar 泛化给 `hat_wavelet` K4，并新增 `B5i` / `B5j` / `B5k` | 针对 B5h 的 E1/E6/E8 二次表达崩塌；只增加固定 input-cross features + trainable readout，`B5k` 再加小 linear residual bracket。未改变 loss、gate、sampler、class weight、teacher/distillation 或 dataset branch。 |
| `dgkan/models/fc_purekan_primitives.py` | 为 `hat_wavelet` input-cross manual CE path 增加显式 sidecar readout 梯度 | `manual_ce_forward_cache` 在 fused HatWavelet logits 上叠加 sidecar；`manual_ce_backward_from_cache` 用同一 `dL/dlogits` 更新 sidecar readout。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B5i` / `B5j` / `B5k` 加入 `_family_plan()`，并登记 HatWavelet input-cross variants 为 official fused L3 候选 | 继续使用原有 L3/A4/A5 gate；只扩展候选注册。 |

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_wavelet_inputcross_b5i_b5j_b5k_3x3_20260523T1624 \
  --device cuda:0 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B5i,B5j,B5k \
  --family-promotion-ids B5i,B5j,B5k \
  --family-linec-ids B5i,B5j,B5k
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Wavelet status = TaskBlocked
Wavelet A4_expression_pass_candidates = B5i,B5j,B5k
Wavelet A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B5i manual_variant=hat_wavelet_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.5734124486032839
  memory_ratio_q90=1.0319047619047619
  official_efficiency_pass=1
  grad_relerr_max=3.60195059556645e-07
  grad_cos_min=1.0

B5j manual_variant=hat_wavelet_k4_triton_l3_inputcross_gemm
  step_ratio_q90=0.7583891291159233
  memory_ratio_q90=1.01
  official_efficiency_pass=1
  grad_relerr_max=3.421532994707377e-07
  grad_cos_min=0.9999998807907104

B5k manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6297978164362019
  memory_ratio_q90=1.0395238095238095
  official_efficiency_pass=1
  grad_relerr_max=3.481761154944252e-07
  grad_cos_min=0.9999998807907104
```

A4 expression：

```text
B5i A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.2339571714401245
  E2=-0.02527838945388794
  E6=-0.2500254511833191
  E8=-0.275424063205719

B5j A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.15091335773468018
  E2=-0.01973944902420044
  E6=-0.28795069456100464
  E8=-0.22473055124282837

B5k A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.2485828399658203
  E2=-0.0189058780670166
  E6=-0.3873266577720642
  E8=-0.35094213485717773
```

A5 task：

```text
B5i A5=0:
  mean_delta=-0.5518663194444444
  worst_delta=-0.6640625
  near_pass_rate=0.0

B5j A5=0:
  mean_delta=-0.5405815972222222
  worst_delta=-0.6484375
  near_pass_rate=0.0

B5k A5=0:
  mean_delta=-0.23155381944444445
  worst_delta=-0.314453125
  near_pass_rate=0.0
```

失败表：

```text
B5i failure_code=A5_family_task_gate_fail
B5j failure_code=A5_family_task_gate_fail
B5k failure_code=A5_family_task_gate_fail
action_recommended=repair task trajectory without lowering gates, changing loss, using teacher/distillation, sampler/class weights, or dataset-name branches
```

### 分析结论

Wavelet 取得实质推进：

```text
B5h baseline:
  A4=0
  E1=-1.0245640277862549
  E6=-0.9987626075744629
  E8=-0.9816438555717468

B5i/B5j/B5k:
  A4=1
```

但 Wavelet 现在和 Chebyshev/Rational 一样转为 `TaskBlocked`，不是 FamilyPass。`B5k` 的 small linear residual 显著改善任务崩塌：

```text
B5i mean_delta=-0.5518663194444444
B5j mean_delta=-0.5405815972222222
B5k mean_delta=-0.23155381944444445
```

下一步应沿 `B5k` 做 task repair bracket，测试 raw-linear residual strength 是否能继续改善 A5；但必须继续监控 official efficiency 与 A4，且如果 Line C/Reservoir 暴露同类问题，不能把 task 改善写成成功。

## 20. 2026-05-23 继续推进：Wavelet B5l/B5m/B5n Raw-Linear Strength Bracket

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B5l` / `B5m` / `B5n` HatWavelet raw-linear residual candidates | 沿用 B5k 的 K4 inputcross rank128 + HatWavelet L3 path，只把 linear residual 改成 raw scale 0.50 / 0.25 / 0.10，用于验证 B5k 的 task 改善是否可继续增强。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B5l` / `B5m` / `B5n` 加入 `_family_plan()` | 继续使用原有 L3/A4/A5 gate；不改变 loss、sampler、class weight、teacher/distillation 或 dataset branch。 |

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_wavelet_rawlinear_b5l_b5m_b5n_3x3_20260523T1633 \
  --device cuda:1 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B5l,B5m,B5n \
  --family-promotion-ids B5l,B5m,B5n \
  --family-linec-ids B5l,B5m,B5n
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Wavelet status = TaskBlocked
Wavelet A4_expression_pass_candidates = B5m,B5n
Wavelet A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B5l manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6368477999094533
  memory_ratio_q90=1.0614285714285714
  official_efficiency_pass=0
  grad_relerr_max=3.5966365885542473e-07
  grad_cos_min=0.9999999403953552

B5m manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6346602332236295
  memory_ratio_q90=1.0395238095238095
  official_efficiency_pass=1
  grad_relerr_max=3.3424080925215094e-07
  grad_cos_min=0.9999999403953552

B5n manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6389508101593233
  memory_ratio_q90=1.0395238095238095
  official_efficiency_pass=1
  grad_relerr_max=3.489445248305856e-07
  grad_cos_min=1.0
```

A4 expression：

```text
B5m A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.3060670495033264
  E2=-0.029755711555480957
  E6=-0.3736472725868225
  E8=-0.24180954694747925

B5n A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.2454884648323059
  E2=-0.023116588592529297
  E6=-0.3631210923194885
  E8=-0.2962718605995178
```

A5 task：

```text
B5m A5=0:
  mean_delta=-0.05946180555555555
  worst_delta=-0.076171875
  near_pass_rate=0.0

B5n A5=0:
  mean_delta=-0.05555555555555555
  worst_delta=-0.076171875
  near_pass_rate=0.0
```

失败表：

```text
B5l failure_code=L3_fused_kernel_efficiency_fail
B5m failure_code=A5_family_task_gate_fail
B5n failure_code=A5_family_task_gate_fail
```

### 分析结论

Wavelet raw-linear residual 明显改善 task trajectory，但仍不是成功：

```text
B5k mean_delta=-0.23155381944444445, worst_delta=-0.314453125
B5m mean_delta=-0.05946180555555555, worst_delta=-0.076171875
B5n mean_delta=-0.05555555555555555, worst_delta=-0.076171875
```

当前 Wavelet 已从 `ExpressionBlocked` 进入 `TaskBlocked`，且 `B5n` 是目前最接近的 Wavelet candidate。但 `near_pass_rate=0.0`，worst delta 仍远低于 `-0.010`，不能写成 FamilyPass。`B5l` 说明更强 raw residual 会先触发 memory gate，因此下一步应测试更低 raw strength bracket，而不是继续增大 residual。

## 21. 2026-05-23 继续推进：Wavelet B5o/B5p/B5q Lower Raw-Linear Strength Bracket

### 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 将 `linearresNNN` scale 解析改为通用三位数比例 | 保留旧 `010/025/050` 语义，同时允许 `005/002/001` 等低强度 bracket；这是结构强度扫描，不使用数据集分支或修改 loss。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B5o` / `B5p` / `B5q` HatWavelet lower raw-linear candidates | 沿用 B5n 的 K4 inputcross rank128 + raw-linear path，把 raw residual scale 降到 0.05 / 0.02 / 0.01。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B5o` / `B5p` / `B5q` 加入 `_family_plan()` | 继续使用原有 L3/A4/A5 gate。 |

### 执行命令

```text
conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py \
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_lineD_wavelet_lowraw_b5o_b5p_b5q_3x3_20260523T1641 \
  --device cuda:2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --family-candidate-ids B5o,B5p,B5q \
  --family-promotion-ids B5o,B5p,B5q \
  --family-linec-ids B5o,B5p,B5q
```

### 真实结果

```text
route = R2-B109AUCTrajectoryStillBlocked
classic_family_pass_count = 0
Wavelet status = TaskBlocked
Wavelet A4_expression_pass_candidates = B5p,B5q
Wavelet A5_task_pass_candidates = []
```

效率与 gradcheck：

```text
B5o manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6186735981234691
  memory_ratio_q90=1.0614285714285714
  official_efficiency_pass=0
  grad_relerr_max=3.4572232721075125e-07
  grad_cos_min=1.0

B5p manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6049155090197379
  memory_ratio_q90=1.0395238095238095
  official_efficiency_pass=1
  grad_relerr_max=3.4043171126540983e-07
  grad_cos_min=0.9999999403953552

B5q manual_variant=hat_wavelet_k4_triton_l3_inputcross_linearres_gemm
  step_ratio_q90=0.6367842856350436
  memory_ratio_q90=1.0395238095238095
  official_efficiency_pass=1
  grad_relerr_max=3.46848537446931e-07
  grad_cos_min=1.0
```

A4 expression：

```text
B5p A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.25651609897613525
  E2=-0.015132904052734375
  E6=-0.3686752915382385
  E8=-0.37907224893569946

B5q A4=1:
  frozen_key_r2_ge_089=1
  dead_basis_max=0.0
  E1=-0.24100583791732788
  E2=-0.01899045705795288
  E6=-0.24853569269180298
  E8=-0.3615977168083191
```

A5 task：

```text
B5p A5=0:
  mean_delta=-0.05577256944444445
  worst_delta=-0.076171875
  near_pass_rate=0.0

B5q A5=0:
  mean_delta=-0.05685763888888889
  worst_delta=-0.080078125
  near_pass_rate=0.0
```

失败表：

```text
B5o failure_code=L3_fused_kernel_efficiency_fail
B5p failure_code=A5_family_task_gate_fail
B5q failure_code=A5_family_task_gate_fail
```

### 分析结论

更低 raw-linear strength 没有突破 A5：

```text
B5n scale=0.10 mean_delta=-0.05555555555555555, worst_delta=-0.076171875
B5p scale=0.02 mean_delta=-0.05577256944444445, worst_delta=-0.076171875
B5q scale=0.01 mean_delta=-0.05685763888888889, worst_delta=-0.080078125
```

因此 Wavelet 的 simple raw-linear residual scaling 已经到平台期，不能靠继续缩放 residual 得到 FamilyPass。当前 Wavelet 结论：

```text
Wavelet_TaskBlocked_after_A4_open
best_current_candidate = B5n
FamilyPass = 0
```

下一步如果继续 Wavelet，需要新的 task-geometry primitive 或 Line C/autopsy 定位；否则应回到 Chebyshev/Wavelet 共同的 `TaskBlocked` 现象，设计不依赖 CE 的任务轨迹修复机制。
