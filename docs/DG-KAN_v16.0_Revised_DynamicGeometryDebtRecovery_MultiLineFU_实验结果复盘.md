# DG-KAN v16.0 Revised DynamicGeometryDebtRecovery MultiLineFU 实验结果复盘

生成时间：2026-06-01（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把短程 source、MLP generic positive、LQ/Rational monitor、recovery-only 或 substrate-only rows 写成 KAN-specific promotion。

## 1. 计划理解

v16.0 的目标是把 functional update 从 single-step safety 改成 dynamic debt recovery：允许短期 bad debt，但必须在 h800/h1600 保留 source、偿还 tail/LineC/calibration/AUC debt，并打过 matched controls。

## 2. 本轮代码修改

新增：

```text
experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v16.0 substrate-only candidates: D-FOU87..91, D-RBF85..89, D-WAV73..76。
```

过程说明：

```text
1. Line A/B 全 surface 执行到 H=800；不因 h400 fail 停止。
2. Line A/B top-2 source-retaining candidates 执行 H=1600 extension。
3. Line G/DGS、LineC/tail/AUC/calibration 只作为 readback/audit/gate，不生成方向。
4. Line B MLP 是 active dynamics line，不写成 KAN-specific promotion。
5. Line C 先执行 C0..C6 LQ reanchor；未开 gate 时 C7..C10 functional fail-closed deferred。
6. Line D Rational monitor 不启动 reset/controller/action route。
7. Line E recovery matrix 从 A/B 实际 recovery rows readback，不作为方向源。
8. Line F all-basis substrate-only rows 只作为 carrier/substrate gate，不进入 official FU proof。
9. 四卡分片执行 A/B/C/D/F；cpu_offload_used=0。
```

<!-- V16.0_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line G/A/B/C/D/E/F/M/Z 数据仍由 runner 从 artifact 写入。

v16.0 的关键制度修复是：h400 失败不再自动阻断 h800/h1600 诊断。A/B full surface 已直接跑到 h800，再对 top-2 source-retaining candidates 执行 h1600。本轮是否 promotion 只看 S5/long-horizon gates，不把短程 source 或 generic MLP positive 写成 KAN-specific promotion。这个制度修复是必要的；它排除了“是不是因为 h400 太早挡住长期诊断”的反方质疑。

结论上，v16.0 没有证明 productive debt recovery。最接近希望的 D-CHE row 是 `A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery`：`source_vs_best_control_h800 = 0.049836417039235435`，`LineC_recovery_rate_h800 = 0.5555555555555556`，看起来比 v15.8/v15.9 的纯短程判断更有信号。但它同时有三个致命缺口：`source_retention_h800 = 0.24507864895794126` 低于计划阈值 0.50，`tail_recovery_rate_h800 = 0.023526686156257203` 远低于 0.60，`AUCtime_ratio_h800 = 1.1013260447040822` 高于 1.05。也就是说，A8 不是“好更新被坏 gate 误杀”，而是只恢复了部分 LineC，source 没保住、tail debt 没偿还、训练路径还变慢。

MLP active line 给出了另一个重要线索，但不是 KAN promotion。`B7-MLP-PulseOnce-then-LRCooldownRecovery` 的 `source_vs_best_control_h800 = 0.06957992580201891`，`source_retention_h800 = 0.648903876543045`，`AUCtime_ratio_h800 = 0.9137877291838247`，在 source/retention/AUC 上比 D-CHE A8 更像有效训练动力学；`B10-MLP-PulseOnce-then-SWAConsolidation` 也有 `source_vs_best_control_h800 = 0.0515155726008945` 与 `source_retention_h800 = 0.6673421528604295`。但 MLP line 的 `LineC_recovery_rate_h800 = 0.0`，tail recovery 也只有约 `0.001`，所以它不是 productive debt recovery。更关键的是，Line M 的 13 个 KAN-vs-MLP attribution rows 中 `KAN_specific_advantage = 0 / 13`，最好的 D-CHE A8 对 best MLP B7 的 `delta_KAN_specific = -0.019743508762783475`。这说明本轮最强 source 是 generic MLP dynamics，而不是 KAN-specific functional advantage。

h1600 进一步关闭了“再等更久会翻盘”的解释。A-line top-2 到 h1600 后，A8 的 mean source 约 `-0.0007289316919114855`，bad event fraction 仍为 `0.8888888888888888`；A6 的 mean source 约 `-0.07380426592297024`，bad event fraction 为 `1.0`。B-line top-2 到 h1600 后，B7 mean source 约 `-0.01218777232699924`，B10 mean source 约 `-0.03671520948410034`，bad event fraction 均为 `1.0`。因此，h800 的正 source 没有随更长 recovery horizon 巩固，反而退回到非 promotion 状态。

Line G 的 dynamic geometry 也没有给出正向解释：`line_g_best_dgs_h800 = -0.25`。这意味着当前 DGS/readback 没有捕捉到一种“虽有局部 debt 但长期几何改善”的模式。结合 failure taxonomy，失败不是某个单项偶然超阈值：A/B/A1600/B1600 中 dominant classes 是 `source,tail_debt,linec_debt,control_equivalent` 与 `source,linec_debt,control_equivalent`，说明 source retention、tail/LineC debt 与 control-equivalence 是耦合失败，而不是单独一个指标过严。

Line C 的 LQ reanchor 仍然值得保留为下一版线索，但 v16.0 内不能放行 functional。C0/C4 这类 historical/current reference 有 near-pass count 到 `6 / 9`，说明 LQ base 不是完全无信号；可是 reanchor gate 仍为 0，C7..C10 functional rows 全部按 `R-C-LQReanchorStillBlocked` fail-closed deferred。这个结果支持“LQ 需要 base/reanchor repair”，不支持在 v16.0 里直接开 LQ functional update。

Line D Rational 的作用仍然只是 monitor/sanity replay。D0/Rational AdamW monitor 没有打开 gate，且计划明确禁止从 Rational monitor 重启 reset/controller/action route。Line F all-basis substrate 也没有提供逃生口：D-FOU/D-RBF/D-WAV 的 pass count 都是 `0 / 9`，best family 仍是 D-FOU，但 `max_mean_delta_vs_MLP = -0.109375`，不能进入 official FU proof。

因此，本轮最有价值的 insight 不是“完全没有信号”，而是信号分裂得很清楚：D-CHE 有一点 LineC recovery 但 source retention/tail/AUC 不成立；MLP 有 source-retention/AUC 但 tail/LineC debt 不偿还且是 generic；LQ 有 near-pass base 但 reanchor 未闭合；all-basis substrate 没给出 carrier escape。v16.0 把“短程 bad event 是否可偿还”这个问题问到底，答案是否定的。当前 route `R16-CurrentFunctionalDynamicsFamilyNoGo` 是一个真正的 no-go boundary，而不是工程漏跑。

下一版不能简单沿着 v16.0 继续加 token/action/controller/reset；那会变成 audit-directed search。合理的下一步应当是新的预注册计划：要么重新定义 recovery/DGS，使它能解释并预测 B7 这类 source-retention dynamics 为什么不偿还 LineC/tail；要么先做 LQ/all-basis substrate repair，让 carrier 本身具备可偿还 debt 的动力学结构，再回到 official FU proof。
<!-- V16.0_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

| contract item | status | details |
| --- | --- | --- |
| Line R provenance/no-action audit | 1 | audit files present |
| Line G dynamic geometry | 1 | horizon debt/DGS readback present |
| Line A D-CHE h800 + h1600 | 1 | A0..A13 + ACTRL0..6, H=1..800 plus top2 H=1600 |
| Line B MLP active h800 + h1600 | 1 | B0..B10 + BCTRL0..5, H=1..800 plus top2 H=1600 |
| Line C LQ reanchor | 1 | C0..C6 and functional fail-closed table |
| Line D Rational monitor | 1 | D0..D3 + random control, no reset |
| Line E recovery matrix | 1 | A/B recovery-family readback matrix |
| Line F all-basis substrate | 1 | D-FOU/RBF/WAV substrate rows |
| Line M controls/attribution | 1 | KAN-vs-MLP attribution rows |
| Line Z closure | 1 | no-go + next queue + exhaustion |
| Required figures | 1 | figures=23 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

深度覆盖审计：

| audit item | status | details |
| --- | --- | --- |
| Line A exact surface | 1 | A methods x dataset/seed |
| Line A h800 coverage | 1 | H=800 rows present for all A methods |
| Line B exact surface | 1 | B methods x dataset/seed |
| Line B h800 coverage | 1 | H=800 rows present for all B methods |
| Line C reanchor coverage | 1 | C0..C6 x dataset/seed |
| Line D rational coverage | 1 | D0..D3 + control |
| Direction provenance train-stream-only | 1 | direction_rows=2034 |
| Budget exhaustion certificate | 1 | mandatory/fallback/deferred fields present |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 45
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_provenance_rows = 2034
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. Line G dynamic geometry / debt accounting

```text
line_g_rows = 2646
line_g_h800_rows = 378
line_g_best_dgs_h800 = -0.25
```

## 4. Line A / B h800 dynamics summary

```text
line_a_best_method = A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery
line_a_gate_pass = 0
line_a_source_vs_best_control_h800 = 0.049836417039235435
line_a_source_retention_h800 = 0.24507864895794126
line_a_tail_recovery_h800 = 0.023526686156257203
line_a_LineC_recovery_h800 = 0.5555555555555556
line_b_best_method = B7-MLP-PulseOnce-then-LRCooldownRecovery
line_b_gate_pass = 0
line_b_source_vs_best_control_h800 = 0.06957992580201891
line_b_source_retention_h800 = 0.648903876543045
line_b_tail_recovery_h800 = 0.0015119816696717206
line_b_LineC_recovery_h800 = 0.0
```

Line A method summary：

| method | rows | h800 source | retention | tail | LineC | AUC | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1-D-CHE-G7R-PulseOnce-then-AdamWRecovery | 9 | -0.010300404495663114 | 0.2371344417333603 | 0.007829453917055187 | 0.5555555555555556 | 1.0065316825449426 | 0 |
| A2-D-CHE-G7R-PulseEvery50-then-AdamWRecovery | 9 | -0.020112611629344797 | 0.2417039300004641 | 0.008725026689559968 | 0.5555555555555556 | 0.9452940747343817 | 0 |
| A3-D-CHE-G7R-PulseEarlyOnly-then-AdamWRecovery | 9 | -0.010124127070109049 | 0.24594695700539482 | 0.008567604273544856 | 0.3333333333333333 | 0.9557463600185775 | 0 |
| A4-D-CHE-G7R-PulseMidOnly-then-AdamWRecovery | 9 | -0.012912147574954562 | 0.24522851821449068 | 0.004439075599292098 | 0.2222222222222222 | 0.916066753215952 | 0 |
| A5-D-CHE-G7R-PulseLateOnly-then-AdamWRecovery | 9 | -0.018711262279086642 | 0.24395226190487543 | 0.008104788208351572 | 0.2222222222222222 | 0.8968995221429553 | 0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | 9 | -0.00908840364880032 | 0.2576528737942378 | 0.01683490080595655 | 0.4444444444444444 | 1.001526389527734 | 0 |
| A7-D-CHE-G7R-PulseOnce-then-SecondMomentAdaptRecovery | 9 | -0.6426673928896586 | 0.33778829872608185 | 0.00255685268760842 | 0.5555555555555556 | 1.2157535484185629 | 0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | 9 | 0.049836417039235435 | 0.24507864895794126 | 0.023526686156257203 | 0.5555555555555556 | 1.1013260447040822 | 0 |
| A9-D-CHE-G7R-PulseOnce-then-EMALookaheadRecovery | 9 | -0.010991255442301432 | 0.28029681576622856 | 0.0184512263613759 | 0.5555555555555556 | 1.0010220353776573 | 0 |
| A10-D-CHE-G7R-PulseOnce-then-DecoupledDecayRecovery | 9 | -0.009617699517144097 | 0.23505707167916828 | 0.007904391864198944 | 0.5555555555555556 | 1.006395872644388 | 0 |
| A11-D-CHE-G7R-PulseOnce-then-RoleWiseDecayRecovery | 9 | -0.009309775299496122 | 0.2343860318263372 | 0.00789737673394631 | 0.5555555555555556 | 1.006331113851274 | 0 |
| A12-D-CHE-G7R-PulseOnce-then-HighDegreeDecayRecovery | 9 | -0.009787480036417643 | 0.2355287704202864 | 0.007886836896612806 | 0.5555555555555556 | 1.0064296007766809 | 0 |
| A13-D-CHE-G7R-PulseOnce-then-SWAConsolidation | 9 | -0.011189864741431342 | 0.3069472528166241 | 0.019110936598916745 | 0.5555555555555556 | 1.0004256126786664 | 0 |

Line B method summary：

| method | rows | h800 source | retention | tail | LineC | AUC | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1-MLP-SplitConsensusHiddenMetricPulseOnce-then-AdamWRecovery | 9 | -0.1225720312860277 | 0.6137402455012003 | 0.00016144292736958618 | 0.0 | 1.0563262374625286 | 0 |
| B2-MLP-SplitConsensusHiddenMetricPulseEvery50-then-AdamWRecovery | 9 | -0.1321605951697738 | 0.6141988933086395 | 0.0003037664242788988 | 0.0 | 1.0734262068879277 | 0 |
| B3-MLP-FMS-AmortizedPulseOnce-then-AdamWRecovery | 9 | -0.5457111994425455 | 0.6005850964122348 | 0.015997400261506964 | 0.0 | 1.3500246591917753 | 0 |
| B4-MLP-PopRiskSNRPulseOnce-then-AdamWRecovery | 9 | -0.018140958415137395 | 0.6338544819090102 | 0.0008476138864217327 | 0.0 | 0.9648883564058178 | 0 |
| B5-MLP-PulseOnce-then-MomentumDampedRecovery | 9 | -0.006364716423882378 | 0.6506042745378282 | 0.0016246734719708963 | 0.0 | 0.9554960764835891 | 0 |
| B6-MLP-PulseOnce-then-SecondMomentAdaptRecovery | 9 | -2.1086046828163996 | 0.606909821430842 | 0.2560923780524897 | 0.0 | 1.74361102906608 | 0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | 9 | 0.06957992580201891 | 0.648903876543045 | 0.0015119816696717206 | 0.0 | 0.9137877291838247 | 0 |
| B8-MLP-PulseOnce-then-EMALookaheadRecovery | 9 | 0.03347541888554891 | 0.6588976118299696 | 0.0013769148577819689 | 0.0 | 0.926575136344607 | 0 |
| B9-MLP-PulseOnce-then-DecoupledDecayRecovery | 9 | -0.11170809798770481 | 0.5843808584743075 | 0.00012291560671945769 | 0.0 | 1.0523608597616871 | 0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | 9 | 0.0515155726008945 | 0.6673421528604295 | 0.0010801768480451432 | 0.0 | 0.915720188864548 | 0 |

Line A h1600 top-2：

| method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | MNIST | 0 | 0.0 | 1 | 0.8202739115814114 | 1.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | MNIST | 0 | -0.039568185806274414 | 1 | 0.6862210728326177 | 1.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | MNIST | 1 | 0.0 | 1 | 1.0 | 1.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | MNIST | 1 | -0.032341718673706055 | 1 | 0.6129434184318153 | 0.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | MNIST | 2 | -0.006560385227203369 | 1 | 0.1179676466812171 | 1.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | MNIST | 2 | 0.0 | 1 | 0.4801247356776426 | 1.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | Fashion-MNIST | 0 | 0.0 | 1 | 1.0 | 0.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | Fashion-MNIST | 0 | -0.0762944221496582 | 1 | 0.9680219237918923 | 0.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | Fashion-MNIST | 1 | 0.0 | 0 | 1.0 | 1.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | Fashion-MNIST | 1 | -0.10154056549072266 | 1 | 0.021338483875953707 | 1.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | Fashion-MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | Fashion-MNIST | 2 | -0.08722150325775146 | 1 | 0.128298240108393 | 0.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | KMNIST | 0 | 0.0 | 1 | 0.025479373762969708 | 0.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | KMNIST | 0 | -0.06328946352005005 | 1 | 0.12774591017909315 | 0.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | KMNIST | 1 | -0.15889573097229004 | 1 | 0.005644742580019627 | 0.0 |
| A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery | KMNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery | KMNIST | 2 | -0.1050868034362793 | 1 | 0.08821400171761189 | 1.0 |

Line B h1600 top-2：

| method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | MNIST | 0 | 0.0 | 1 | 0.012290713539210478 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | MNIST | 0 | -0.0057326555252075195 | 1 | 0.005731418358892217 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | MNIST | 1 | 0.0 | 1 | 0.0009827716929828505 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | MNIST | 1 | -0.010550916194915771 | 1 | 0.013078762961154414 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | MNIST | 2 | 0.0 | 1 | 0.013716407810840167 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | MNIST | 2 | -0.10081648826599121 | 1 | 0.03140470328171725 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | Fashion-MNIST | 0 | 0.0 | 1 | 0.0634814269749412 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | Fashion-MNIST | 0 | -0.0547182559967041 | 1 | 0.3490088629245103 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | Fashion-MNIST | 1 | -0.010969400405883789 | 1 | 0.0 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | Fashion-MNIST | 1 | 0.0 | 1 | 0.03439475831675646 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | Fashion-MNIST | 2 | -0.027425050735473633 | 1 | 0.0 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | Fashion-MNIST | 2 | 0.0 | 1 | 0.0018823886851037073 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | KMNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | KMNIST | 0 | -0.059560418128967285 | 1 | 0.0 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | KMNIST | 1 | -0.09905815124511719 | 1 | 0.0 | 0.0 |
| B7-MLP-PulseOnce-then-LRCooldownRecovery | KMNIST | 2 | -0.07129549980163574 | 1 | 0.0 | 0.0 |
| B10-MLP-PulseOnce-then-SWAConsolidation | KMNIST | 2 | 0.0 | 1 | 0.004423604869948636 | 0.0 |

## 5. Line C / D / E / F / M results

```text
line_c_rows = 63
line_c_reanchor_gate_pass = 0
line_c_best_method = C6-LQ-LineCNoRegressionCheck
line_c_best_macro_delta_vs_MLP = 0.03125
line_d_rat_rows = 45
line_d_rat_best_method = D0-RAT-AdamWMonitor
line_d_rat_gate_pass = 0
line_e_rows = 342
line_f_rows = 126
line_f_best_family = D-FOU
line_f_best_dataset_seed_pass_count = 0 / 9
line_m_rows = 13
generic_functional_dynamics_rows = 13
kan_specific_advantage_rows = 0
```

Line C reanchor summary：

| method | rows? | seed | delta | near | LineC | gate |
| --- | --- | --- | --- | --- | --- | --- |
| C0-HistoricalLQReferenceReplay | MNIST | 0 | 0.015625 | 1 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 0 | -0.015625 | 0 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 0 | -0.0234375 | 0 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 0 | -0.0625 | 0 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 0 | 0.015625 | 1 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 0 | 0.015625 | 1 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 0 | -0.015625 | 0 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | MNIST | 1 | 0.015625 | 1 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 1 | -0.0234375 | 0 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 1 | -0.03125 | 0 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 1 | 0.0078125 | 1 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 1 | -0.0078125 | 0 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 1 | 0.0078125 | 1 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 1 | -0.0625 | 0 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | MNIST | 2 | 0.015625 | 1 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 2 | 0.015625 | 1 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 2 | 0.015625 | 1 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 2 | 0.0234375 | 1 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 2 | 0.015625 | 1 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 2 | 0.0 | 1 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 2 | 0.015625 | 1 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 0 | 0.015625 | 1 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 0 | -0.015625 | 0 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 0 | 0.0 | 1 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 0 | 0.0234375 | 1 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 0 | 0.0 | 1 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 0 | -0.0078125 | 0 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 0 | 0.015625 | 1 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 1 | 0.0234375 | 1 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 1 | 0.0 | 1 | 1 | 0 |

Line D Rational summary：

| method | pass | source | AUC |
| --- | --- | --- | --- |
| D0-RAT-AdamWMonitor | 0 | -0.0024232864379882812 |  |
| D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery | 0 | -0.04729383521609836 |  |
| D2-RAT-G7AnalogPulseOnce-then-DecayRecovery | 0 | -0.07412189245223999 |  |
| D3-RAT-NoRegressionCheck | 0 | -0.0024232864379882812 |  |
| DCTRL-RandomPulseSameRecovery | 0 | -0.03902600871192084 |  |

Line F all-basis family summary：

| family | rows | pass | best candidate | mean delta | gate |
| --- | --- | --- | --- | --- | --- |
| D-FOU | 45 | 0 | D-FOU88-BandwiseSNRWarmupV6 | -0.33315972222222223 | 0 |
| D-RBF | 45 | 0 | D-RBF88-GaussianLocalK4TaskHealthV6 | -0.4791666666666667 | 0 |
| D-WAV | 36 | 0 | D-WAV75-SupportOverlapDampingV6 | -0.3565538194444444 | 0 |

## 6. Final route / no-go

```text
route = R16-CurrentFunctionalDynamicsFamilyNoGo
minimum_success = S1-MultiLineCoverageCompleted
official_s5_reached = 0
promotion_allowed = 0
line_a_gate_pass = 0
line_b_gate_pass = 0
line_c_reanchor_gate_pass = 0
line_d_rational_gate_pass = 0
line_f_allbasis_gate_pass = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

No-go boundary：

| boundary | status | evidence |
| --- | --- | --- |
| LineA-DCHENoGo | 1 | best=A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery;source_h800=0.049836417039235435;tail_h800=0.023526686156257203;linec_h800=0.5555555555555556 |
| LineB-MLPNoGo | 1 | best=B7-MLP-PulseOnce-then-LRCooldownRecovery;source_h800=0.06957992580201891;tail_h800=0.0015119816696717206;linec_h800=0.0 |
| LineC-LQReanchor | 1 | best=C6-LQ-LineCNoRegressionCheck;delta=0.03125;near=6/9;route=R-C-LQReanchorStillBlocked |
| LineD-RationalMonitorOnly | 1 | best=D0-RAT-AdamWMonitor;pass=0;no reset/controller |
| LineF-AllBasisSubstrate | 1 | best_family=D-FOU;pass=0/9 |
| R16-CurrentFunctionalDynamicsFamilyNoGo | 1 | A=0;B=0;C=0;D=0;F=0 |
| NoForbiddenContinuation | 1 | no G9/G10/action bank/controller/reset/audit-directed branch |

Next hypothesis queue：

| priority | hypothesis | allowed next step |
| --- | --- | --- |
| 1 | DynamicRecoveryDefinitionRevision | Only in a new pre-registered plan; h800/h1600 artifacts may inform theory but not act as controller. |
| 2 | MLPGenericDynamicsFollowup | Study MLP positive source as generic dynamics, not KAN-specific promotion. |
| 3 | LQReanchorOrAllBasisSubstrateRepair | Repair carrier/substrate before official FU proof. |

## 7. 科学结论

```text
1. v16.0 已执行 Line R/G/A/B/C/D/E/F/M/Z，并生成 required artifacts。
2. A/B 均完成 H=800 全 surface；h400 未作为 stopping rule；top-2 已执行 H=1600 extension。
3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/debt readback，没有反推方向。
4. 当前 route = R16-CurrentFunctionalDynamicsFamilyNoGo，promotion_allowed = 0。
5. MLP generic positive、LQ/Rational monitor、recovery-only 与 substrate-only rows 不写成 KAN-specific promotion。
```

## 11. Finalizer coverage repair

首次 finalizer 后 route=R0，因为 manifest 缺 v160_execution_contract_coverage_audit.csv 与 v160_deep_coverage_audit.csv。已从实际 artifact 生成两份 coverage audit，并补入 runner 的 build_audits 写入逻辑；重跑 finalizer 后 required_artifact_missing_count=0，route=R16-CurrentFunctionalDynamicsFamilyNoGo。该修复只影响覆盖审计/manifest/route/docs，不改变训练指标。

## 12. 用户再次追问后的计划闭环复核

计划复核结论：v16.0 已经按 promotion fail-closed / exploration dynamics-open 执行到 h800/h1600；A/B/C/D/E/F gates 全部为 0，promotion_allowed=0。完整计划明确要求在这种情况下写 R16-CurrentFunctionalDynamicsFamilyNoGo，并停止当前 train-stream functional-update family，不继续新增 token/action/controller/reset。本节只补闭环复核，不新增训练、不改变指标。
