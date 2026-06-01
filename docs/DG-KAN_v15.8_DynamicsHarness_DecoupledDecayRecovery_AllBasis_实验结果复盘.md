# DG-KAN v15.8 DynamicsHarness DecoupledDecayRecovery AllBasis 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 dynamics harness diagnostic、decay-only rows、substrate-only rows 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.8 的目标是检验 G7R 的 local bad event 是否为可偿还 training debt，并测试 decoupled decay/recovery dynamics 是否能保留 source 同时偿还 tail/LineC debt。

## 2. 本轮代码修改

新增：

```text
experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py
experiments/run_v158_h800_extension.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.8 substrate-only candidates:
  D-FOU77..81, D-RBF75..79, D-WAV65..68。
```

过程说明：

```text
1. Line T 执行 T0..T5 与 TCTRL controls，记录 h=1/5/20/50/100 recovery rows。
2. Line W 执行 W0..W6 与 decay-only/random/noop/AdamW controls，记录 decoupled decay norm decomposition。
3. Line H 对 top-2 T/W candidates 执行 long-horizon consolidation。
4. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/failure taxonomy，不进入方向。
5. 所有训练/finalizer 命令均使用 --device cuda:0；cpu_offload_used=0。
6. 自查发现 v15.8 finalizer 初版查找旧 Line D 文件名，未读到实际 v149_line_d_substrate_repair_results.csv，
   已修正为兼容实际文件名并按 mean_delta_vs_MLP 写入 v158_line_d_allbasis_results.csv；该修复只影响 artifact 汇总/route，不改训练指标。
7. 用户追问后，按计划 top-2 T/W candidates 执行 H=800 长程确认；该 extension 不使用 audit metric 生成方向，不新增 controller/action/reset。
8. 反方复核发现 Line Z 缺少独立 no-go boundary 与 next hypothesis queue artifact，已补齐 required manifest / contract / docs；该修复只补闭环审计，不改训练指标。
```

<!-- V15.8_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line T/W/H/B/M/D/route 数据由 runner 从 artifact 写入。

v15.8 的判断重心已经从“单步是否安全”转到“bad debt 是否能偿还”。本轮结果没有支持 productive plasticity：Line T 最好的 T5 late pulse 只有 `source_vs_best_control_mean = 0.003614438904656304`，低于计划里需要的实质 source；同时 `bad_event_fraction = 1.0`、`real_lite_pass_count = 0 / 9`。也就是说，pulse timing 能略微改变表面 gain，但没有把 tail/LineC debt 变成可偿还的训练债。

Line W 更像是反证：最好的 W 仍是 `W0-G7R-PulseDefaultAdamWDecay`，新增的 W1..W5 decoupled decay 没有超过默认 decay/recovery；W0 的 source 为负，`source_retention_after_decay` 约 0.236，说明 decay/recovery 在偿还一部分 tail debt 的同时也把可用 source 吃掉了。Line H long-horizon consolidation 也没有打开 gate，best H 仍来自 T5，但没有形成 real-lite transfer。

Line B 的 `B9-source_retention_horizon` 通过了 readback proxy gate，这说明训练轨迹里确实有可观测的 recovery/debt 信号；但计划明确规定 Line B 只能 readback，不能作为 v15.8 的方向源。所以它只能作为下一版理论设计的线索，不能在本版升级为 controller 或 reset route。

Line M 有 5/9 KAN-specific delta 为正，generic/MLP dynamics 没有完全解释所有局部 gain；但 gain 太小且 official T/W/H gates 全部失败。Line D all-basis substrate 仍是 0/9，说明继续在当前 all-basis substrate 上堆 recovery harness 没有合法 promotion 分支。

自查修复方面，首次 finalizer 因 Line D 文件名对接旧路径，没有读到实际 `v149_line_d_substrate_repair_results.csv`，导致 manifest 缺 `v158_line_d_allbasis_results.csv` 并错误 route=R0。已修复为读取实际 Line D artifact，并用 `--reuse-if-present 1` 只重算 manifest/route/docs；该修复不改变任何 T/W/H/B/M/D 训练指标。修复后最终 route 是 `R6-AllBasisCarrierBlocked`。

用户再次追问后，我按计划 Line H 的 `train_steps = 400, 800` 思路补跑了 H=800 长程确认。这个追加分支没有改变方向源，只拿 official T/W 的 top-2：T5 late pulse 与 W0 default decay。结果仍然没有翻盘：`line_h800_gate_pass = 0`，`line_h800_real_lite_pass_count = 0`，`line_h800_source_vs_best_control_mean = -0.004856328169504802`，`line_h800_bad_event_fraction = 1.0`。这进一步支持当前判断：更长恢复 horizon 没有把 G7R bad debt 转成 productive plasticity。
<!-- V15.8_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

| contract item | status | details |
|---|---:|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line T pulse recovery dynamics | 1 | T0..T5 plus controls and h=1/5/20/50/100 |
| Line W decoupled decay recovery | 1 | W0..W6 plus decay/random/noop controls |
| Line H long-horizon consolidation | 1 | top-2 T/W candidates at long horizon |
| Line B proxy audit | 1 | B1..B10 leaveout readback only |
| Line D all-basis + monitors | 1 | v15.8 substrate candidates + no-regression |
| Line M generic controls | 1 | MLP/generic controls |
| Line C recovery audit | 1 | debt/recovery/geometry horizon audit |
| Line Z no-go and next hypothesis queue | 1 | route/no-go/failure/exhaustion/next queue present |
| Required figures | 1 | figures=12 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

深度覆盖审计：

| audit item | status | details |
|---|---:|---|
| Line T exact surface | 1 | T0..T5/controls x 3x3 |
| Line W exact surface | 1 | W0..W6 plus controls x 3x3 |
| Line H top-2 surface | 1 | top-2 x 3x3 |
| Direction provenance train-stream-only | 1 | direction_rows=936 |
| Line Z exact closure | 1 | no-go + next hypothesis + exhaustion present |
| No remaining legal v15.8 continuation | 1 | if no S5, continuation requires next theory/substrate plan; no G9/G10/action/controller/reset |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 27
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_provenance_rows = 936
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. Line S split-consensus stability

```text
line_s_rows = 108
line_s_k_batch_sensitivity_rows = 648
line_s_gate_pass = 1
line_s_best_snr = 1.9560231276069324
```

## 4. Line T pulse + recovery dynamics

```text
candidate_count = 5
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.003614438904656304
bad_event_fraction = 1.0
best_method = T5-G7R-PulseLateOnly-then-AdamWRecovery
line_t_gate_pass = 0
tail_debt_recovery_rate = 0.12783177237430454
LineC_debt_recovery_rate = 0.2222222222222222
```

| method | rows | pass | source | bad event | tail recovery | LineC recovery | AUC ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1-G7R-PulseOnce-then-AdamWRecovery | 9 | 0/9 | -0.009109576543172201 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 0.998865251351694 |
| T2-G7R-PulseEvery50-then-AdamWRecovery | 9 | 0/9 | -0.009109576543172201 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 0.998865251351694 |
| T3-G7R-PulseEarlyOnly-then-AdamWRecovery | 9 | 0/9 | -0.002955668502383762 | 0.8888888888888888 | 0.3019419036887385 | 0.2222222222222222 | 0.9945282222990319 |
| T4-G7R-PulseMidOnly-then-AdamWRecovery | 9 | 0/9 | 0.0029962129063076442 | 1.0 | 0.19439953957008232 | 0.2222222222222222 | 0.988677533377537 |
| T5-G7R-PulseLateOnly-then-AdamWRecovery | 9 | 0/9 | 0.003614438904656304 | 1.0 | 0.12783177237430454 | 0.2222222222222222 | 0.9880031883905329 |

## 5. Line W decoupled decay recovery

```text
candidate_count = 7
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.009109576543172201
bad_event_fraction = 1.0
best_method = W0-G7R-PulseDefaultAdamWDecay
line_w_gate_pass = 0
tail_debt_recovery_rate = 0.411967175694845
LineC_debt_recovery_rate = 0.3333333333333333
```

| method | rows | pass | source | bad event | tail recovery | LineC recovery | decay norm | source retention after decay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W0-G7R-PulseDefaultAdamWDecay | 9 | 0/9 | -0.009109576543172201 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 0.04183147061202261 | 0.23644847257269752 |
| W1-G7R-GlobalDecoupledWeightDecayRecovery | 9 | 0/9 | -0.009276005956861708 | 1.0 | 0.41235602177646047 | 0.3333333333333333 | 0.125474257601632 | 0.2363529172208574 |
| W2-G7R-DegreeWiseDecoupledDecay | 9 | 0/9 | -0.009192817740970187 | 1.0 | 0.41216133576937236 | 0.3333333333333333 | 0.08365622493955824 | 0.2364006969663832 |
| W3-G7R-HighDegreeExtraDecay | 9 | 0/9 | -0.009234388669331869 | 1.0 | 0.4122587021118807 | 0.3333333333333333 | 0.10456607697738542 | 0.2363767996430397 |
| W4-G7R-ReadoutBasisDecoupledDecay | 9 | 0/9 | -0.009435898727840848 | 1.0 | 0.4126790299213987 | 0.3333333333333333 | 0.11063951171106762 | 0.2362479509578811 |
| W5-G7R-SignalReservoirDualDecay | 9 | 0/9 | -0.009343134032355415 | 1.0 | 0.41248499204052425 | 0.3333333333333333 | 0.10415061397684945 | 0.23630617807308832 |
| W6-RationalNoRegressionDecayMonitor | 9 | 0/9 | -0.009109576543172201 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 0.04183147061202261 | 0.23644847257269752 |

## 6. Line H / B / P

```text
line_h_rows = 18
line_h_best_method = H-T5-G7R-PulseLateOnly-then-AdamWRecovery
line_h_gate_pass = 0
line_b_best_proxy = B9-source_retention_horizon
line_b_best_auc_bad_event = 0.912621359223301
line_b_best_leaveout_min_auc = 0.75
line_b_gate_pass = 1
Line P failure classes = P1-SourceNotRetained=26, P2-UnrecoveredDamage=1
```

Line B proxy summary：

| proxy | auc bad | control FPR | precision top20 |
|---|---:|---:|---:|
| B1-split_loss_disagreement | 0.6650485436893204 | 0.037037037037037035 | 1.0 |
| B2-recovery_lag | 0.5 | 1.0 | 0.9761904761904762 |
| B3-logit_rms_drift | 0.8883495145631068 | 0.5061728395061729 | 1.0 |
| B4-entropy_collapse | 0.6796116504854369 | 0.32098765432098764 | 1.0 |
| B5-margin_p10_drift | 0.7961165048543689 | 0.5061728395061729 | 1.0 |
| B6-update_cosine_to_adam | 0.8543689320388349 | 0.4444444444444444 | 1.0 |
| B7-loss_q95_over_median | 0.6019417475728155 | 0.06172839506172839 | 1.0 |
| B8-projection_retention_drift | 0.7233009708737864 | 0.2222222222222222 | 1.0 |
| B9-source_retention_horizon | 0.912621359223301 | 0.4444444444444444 | 1.0 |
| B10-hazard_proxy_horizon | 0.6553398058252428 | 0.4444444444444444 | 1.0 |

## 6.1 用户追问后的 H=800 长程确认

```text
extension_runner = experiments/run_v158_h800_extension.py
line_h800_rows = 18
line_h800_horizon_rows = 90
line_h800_best_method = H-T5-G7R-PulseLateOnly-then-AdamWRecovery
line_h800_gate_pass = 0
line_h800_real_lite_pass_count = 0
line_h800_source_vs_best_control_mean = -0.004856328169504802
line_h800_bad_event_fraction = 1.0
line_h800_tail_debt_recovery_rate = 0.14936363382158868
line_h800_LineC_debt_recovery_rate = 0.3333333333333333
promotion_allowed = 0
uses_audit_metric_for_direction = 0
controller_executed = 0
action_bank_used_as_search_space = 0
reset_route_used = 0
cpu_offload_used = 0
```

H800 method summary：

| method | rows | pass | source | control equiv | bad event | tail recovery | LineC recovery | AUC ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H-T5-G7R-PulseLateOnly-then-AdamWRecovery | 9 | 0/9 | -0.004856328169504802 | 1.0 | 1.0 | 0.14936363382158868 | 0.3333333333333333 | 1.002506574598114 |
| H-W0-G7R-PulseDefaultAdamWDecay | 9 | 0/9 | -0.012699074215359159 | 1.0 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 1.007825607505755 |

## 7. Line M / D 结果

```text
line_m_rows = 63
generic_dynamics_explains = 0
kan_specific_pass_count = 5
line_d_rows = 126
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
```

Line M KAN-specific delta：

| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta | generic explains | KAN-specific pass |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | T3-G7R-PulseEarlyOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.006274819374084473 | 0.03253436088562012 | -0.026259541511535645 | 1 | 0 |
| Fashion-MNIST | 1 | T5-G7R-PulseLateOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.004622519016265869 | 0.01734316349029541 | -0.012720644474029541 | 1 | 0 |
| Fashion-MNIST | 2 | T5-G7R-PulseLateOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.0048888325691223145 | -0.0926826000213623 | 0.09757143259048462 | 0 | 1 |
| KMNIST | 0 | T5-G7R-PulseLateOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.003481745719909668 | -0.06536293029785156 | 0.06884467601776123 | 0 | 1 |
| KMNIST | 1 | T4-G7R-PulseMidOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.002689957618713379 | 0.03107166290283203 | -0.028381705284118652 | 1 | 0 |
| KMNIST | 2 | T5-G7R-PulseLateOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.0023877620697021484 | 0.04077506065368652 | -0.038387298583984375 | 1 | 0 |
| MNIST | 0 | T5-G7R-PulseLateOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.004271745681762695 | -0.013692975044250488 | 0.017964720726013184 | 0 | 1 |
| MNIST | 1 | T4-G7R-PulseMidOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.0037621259689331055 | -0.020950376987457275 | 0.02471250295639038 | 0 | 1 |
| MNIST | 2 | T5-G7R-PulseLateOnly-then-AdamWRecovery | MLP-G7AnalogPulsePlusDecay | 0.0038823485374450684 | -0.05492055416107178 | 0.058802902698516846 | 0 | 1 |

Line D summary：

| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |
|---|---:|---:|---|---:|---:|
| D-FOU | 45 | 0/9 | D-FOU78-BandwiseConsensusMetricV3 | -0.109375 | 0 |
| D-RBF | 45 | 0/9 | D-RBF78-GaussianLocalK4TaskHealthV3 | -0.328125 | 0 |
| D-WAV | 36 | 0/9 | D-WAV67-SupportOverlapDampingV4 | -0.0234375 | 0 |

## 8. Line Z no-go / next hypothesis queue

```text
no_go_boundary_rows = 8
next_hypothesis_queue_rows = 3
line_z_used_for_direction = 0
promotion_allowed = 0
```

No-go boundary：

| boundary | status | evidence |
|---|---:|---|
| LineT-ProductivePlasticityNoGo | 1 | best=T5-G7R-PulseLateOnly-then-AdamWRecovery;source=0.003614438904656304;bad=1.0;tail_recovery=0.12783177237430454;linec_recovery=0.2222222222222222 |
| LineW-DecoupledDecayRecoveryNoGo | 1 | best=W0-G7R-PulseDefaultAdamWDecay;source=-0.009109576543172201;bad=1.0;tail_recovery=0.411967175694845;linec_recovery=0.3333333333333333 |
| LineH-LongHorizon400NoGo | 1 | best=H-T5-G7R-PulseLateOnly-then-AdamWRecovery;source=-0.003244764275021023;bad=1.0 |
| LineB-ReadbackOnlyNotDirection | 1 | best_proxy=B9-source_retention_horizon;gate=1;readback_only=1 |
| LineM-GenericControlsNotPromotion | 1 | generic_explains=0;kan_specific_pass_count=5;official_gates_failed=1 |
| LineD-AllBasisCarrierBlocked | 1 | best_family=D-FOU;pass_count=0/9 |
| NoForbiddenContinuation | 1 | no G9/G10/action bank/controller/reset/audit-directed branch allowed in v15.8 |
| LineH800-LongHorizonExtensionNoGo | 1 | best=H-T5-G7R-PulseLateOnly-then-AdamWRecovery;source=-0.004856328169504802;bad=1.0;tail_recovery=0.14936363382158868;linec_recovery=0.3333333333333333 |

Next hypothesis queue：

| priority | hypothesis | allowed next step |
|---:|---|---|
| 1 | TheoryLevelRecoveryMonitorFromB9 | Use only in a new pre-registered theory-level plan; not as v15.8 controller/reset. |
| 2 | SourceRetainingRecoveryArchitecture | Design substrate/base architecture where source retention is separated from recovery damping. |
| 3 | AllBasisCarrierSubstrateRepair | Continue substrate-only work before official FU proof; no promotion from substrate-only rows. |

## 9. 最终 route / 覆盖复核

```text
route = R6-AllBasisCarrierBlocked
minimum_success = S1-SourceSignalObservable
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
contract_unclosed_rows = 0
deep_coverage_unclosed_rows = 0
```

## 10. 科学结论

```text
1. v15.8 已执行 Line R/S/T/W/H/B/C/P/D/M/Z，并生成 required artifacts。
2. G7R pulse/recovery 与 decoupled decay 只使用 train-stream gradient/optimizer state；没有用 LineC/tail/AUC/calibration 反推方向。
3. 当前 route = R6-AllBasisCarrierBlocked，promotion_allowed = 0。
4. MLP/generic controls、decay-only rows 与 substrate-only rows 不写成 KAN-specific promotion。
5. 若 S5 未达成，v15.8 内不允许新增 G9/G10/action bank/controller/reset route。
6. 用户追问后补跑 H=800 长程确认，仍未打开 productive plasticity / long-horizon gate。
7. Line Z no-go boundary 与 next hypothesis queue 已补齐；继续需要下一版 theory-level/substrate-level 计划，不能在 v15.8 内 audit-directed 续跑。
```
