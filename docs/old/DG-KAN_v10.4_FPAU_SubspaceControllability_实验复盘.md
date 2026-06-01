# DG-KAN v10.4 FPAU Subspace Controllability 实验复盘

> 本复盘记录 `DG-KAN_v10.4_FPAU_SubspaceControllability_结果解读与完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.3/v10.1 真实 artifact、CUDA payload sketch/toy mechanism 与本轮真实 FPAU14 generated branch-horizon rows；没有 fake data、proxy rows、占位数据或 CPU offload。本轮不新增 C-line natural density panel，重点是 value-direction subspace、adjoint collapse、memory/offdiag feasibility 与 generated controllability。

## 0. 最新结论

```text
route = CaseB-GoodSubspaceAbsentOrTooDiffuse
primary_blocker = good_value_subspace_not_stable
secondary_blocker = current_action_representation_insufficient
system_legal_controller_pass = 0
generated_route_status = stopped_good_subspace_absent_or_diffuse
```

最终 artifact：`results/real_rerun_20260506/v1040_fpau_subspace_controllability_full_20260518T210000Z`

核心结论：

1. P0 v10.3 boundary lock pass = `1`；source route = `R5-SubspaceMissingValueDirection`；v10.3 P3 mechanism = `1`；v10.3 P4 generated discovery = `0`。
2. P1 known-action rows = `15080`；Fast/Slow/Risky/SafeLow/Bad = `4` / `22` / `321` / `26` / `14707`；matched slowburn pairs = `44`；best AUC = `0.8920118343195266`；P1 discovery pass = `0`。
3. P2 GoodSubspace rank = `16`，explained variance = `0.9998584670459338`，good coverage = `1.0`，good-bad separation = `0.021201327577448192`，generated projection = `0.988584558169047`；P2 pass = `0`。
4. P3 collapse confirmed count = `0`，future adjoint signal count = `0`，max partial corr V240|V1 = `0.01635610475924454`，max CCI = `0.13271891503173477`。
5. P4 joint retention = `0.1986895203590393`，projected coverage = `0.1986895203590393`，P4 pass = `1`，repair = `soft_trust_region_alpha50_projection_attempted`。
6. P5 toy beats current-gradient = `7` / `7`；slow/memory/offdiag toy pass = `1` / `1` / `1`；P5 pass = `1`。
7. P6 Stage4/8/16/64 opened = `6` / `0` / `0` / `0`；generated discovery pass = `0`；best = `FPAU14B-GoodSubspaceMemorySoftConstraint` `FPAU14B-GoodSubspaceMemorySoftConstraint_stage4`。
8. P7 TopK64 precision/SlowBurn recall/risky rate/V240 = `0.03125` / `0.09090909090909091` / `0.0` / `-0.9347178489800314`；P7 discovery pass = `0`。
9. P8/P9/P10 = `not_run` / `not_run` / `not_run`；No-fake audit rows checked = `1038`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v1040_fpau_subspace_controllability.py` | v10.4 runner；执行 P0-P10、subspace sketch、adjoint collapse audit、toy upgrade、FPAU14 generated staircase、manifest/recap。 |
| `experiments/natural_ap0_extension_materializer.py` | 复用真实 payload materializer 与 branch-horizon replay。 |
| `docs/DG-KAN_v10.4_FPAU_SubspaceControllability_结果解读与完整实验计划.md` | 本轮计划与 gate 定义。 |

```text
python -m py_compile experiments/run_v1040_fpau_subspace_controllability.py
```

```bash
python experiments/run_v1040_fpau_subspace_controllability.py --out-dir results/real_rerun_20260506/v1040_fpau_subspace_controllability_full_20260518T210000Z --fresh --device auto --data-root data --seed 1616 --execution-profile full-gated
```

## 2. 执行边界

本轮执行的是 v10.4 FPAU subspace controllability 诊断，不是 natural density C-line panel。P0 只锁定 v10.3 terminal boundary；P1-P7 按计划检查 known-action contrast、value-direction subspace、adjoint collapse、约束可行性、toy upgrade、generated staircase 与 path-type FPO rebuild。

Contract audit：

```text
P0_pass = 1
v1040_plan_has_no_C_line_panel_gate = 1
no_v1040_natural_panel_artifact = 1
P6_no_stage8_without_stage4_gate = 1
P6_no_stage16_without_stage8_gate = 1
P6_no_stage64_without_stage16_gate = 1
controller_not_run_without_hard_gate = 1
```

## 3. Route

```json
{
  "stage": "ROUTE_DECISION_V1040",
  "status": "summary",
  "route": "CaseB-GoodSubspaceAbsentOrTooDiffuse",
  "primary_blocker": "good_value_subspace_not_stable",
  "secondary_blocker": "current_action_representation_insufficient",
  "route_explanation": "Known Fast/Slow actions do not form a stable enough value-direction subspace under current sketch and constraints.",
  "source_route_v1030": "R5-SubspaceMissingValueDirection",
  "P0_boundary_pass": 1,
  "P1_discovery_pass": 0,
  "P2_pass": 0,
  "P2_fail": 1,
  "P2_good_subspace_rank": 16,
  "P2_good_subspace_explained_variance": 0.9998584670459338,
  "P2_generated_projection_to_good": 0.988584558169047,
  "P3_collapse_confirmed": 0,
  "P3_future_adjoint_signal_exists": 0,
  "P4_pass": 1,
  "P4_fail": 0,
  "P4_joint_retention": 0.1986895203590393,
  "P5_pass": 1,
  "P6_generated_discovery_pass": 0,
  "P6_stage64_candidate_pass": 0,
  "P6_stage64_opened_count": 0,
  "P7_discovery_pass": 0,
  "P8_controller_pass": 0,
  "P9_runtime_pass": 0,
  "P10_paired_replay_pass": 0,
  "generated_route_status": "stopped_good_subspace_absent_or_diffuse",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 4. P0 Boundary Lock

| item | value |
|---|---:|
| source route | `R5-SubspaceMissingValueDirection` |
| source P3 mechanism pass | `1` |
| source P4 generated discovery pass | `0` |
| source P6/P7/P8 status | `not_run` / `not_run` / `not_run` |
| source fake/proxy/cpu | `0` / `0` / `0` |
| P0 boundary pass | `1` |

## 5. P1 Path-Type Mechanism Contrast

P1 只在 v10.1/v10.3 交集中的真实 known actions 上做 matched contrast；如果 strict matching 不足，按计划放松 action/payload norm 并合并 Fast+Slow first pass，仍只作为 discovery diagnostic。

| metric | value |
|---|---:|
| known action rows | `15080` |
| FastGood | `4` |
| SlowBurnGood | `22` |
| RiskyHighAUV | `321` |
| SafeLowValue | `26` |
| BadPath | `14707` |
| matched slowburn pairs | `44` |
| FastSlow coverage | `1.0` |
| strong effect families | `2` `exact_transfer,optimizer_state_alignment` |
| best matched AUC | `0.8920118343195266` |
| dataset/template max share | `0.6818181818181818` / `0.4090909090909091` |
| P1 discovery pass | `0` |
| repair attempted | `0` `` |

Top contrast rows：

| contrast | feature | family | pairs | effect | AUC | delta |
|---|---|---|---:|---:|---:|---:|
| `FastGood_vs_BadPath` | `exact_transfer` | `exact_transfer` | `4` | `1.2489350200053388` | `0.9375` | `1.3175850333063863` |
| `FastSlow_vs_BadPath_relaxed` | `exact_transfer` | `exact_transfer` | `26` | `1.2909305844595036` | `0.8920118343195266` | `0.7827621472623343` |
| `SlowBurn_vs_BadPath` | `exact_transfer` | `exact_transfer` | `22` | `1.4724207140385894` | `0.890495867768595` | `0.6855216225270521` |
| `FastGood_vs_BadPath` | `current_response` | `current_response` | `4` | `1.5644952731801078` | `0.875` | `0.1399714641738683` |
| `SafeLow_vs_SlowBurn` | `exact_transfer` | `exact_transfer` | `20` | `0.3382351220612748` | `0.7224999999999999` | `-0.07658454929478467` |
| `SlowBurn_vs_RiskyHighAUV` | `effective_derivative` | `optimizer_state_alignment` | `22` | `0.6826317004739758` | `0.6880165289256198` | `-0.017372305826707346` |
| `SlowBurn_vs_RiskyHighAUV` | `exact_transfer` | `exact_transfer` | `22` | `0.7248109101284813` | `0.6694214876033058` | `-0.24263990297086735` |
| `FastSlow_vs_BadPath_relaxed` | `current_response` | `current_response` | `26` | `0.4900568250488505` | `0.6568047337278107` | `0.05429498632796681` |
| `SlowBurn_vs_BadPath` | `current_response` | `current_response` | `22` | `0.35759257383696336` | `0.6301652892561983` | `0.038717444901439274` |
| `SafeLow_vs_SlowBurn` | `effective_derivative` | `optimizer_state_alignment` | `20` | `0.48960904877589234` | `0.63` | `0.013109536409378064` |
| `FastGood_vs_BadPath` | `action_adamw_cosine` | `adamw_alignment` | `4` | `0.5520588448546966` | `0.625` | `0.002860721212587446` |
| `FastGood_vs_BadPath` | `payload_cosine_to_adamw` | `adamw_alignment` | `4` | `0.5520588448546966` | `0.625` | `0.002860721212587446` |

## 6. P2 Known Good Value-Direction Basis Discovery

P2 使用真实 payload 的 CUDA blockwise sketch 构造 Good/Risky/Bad/SafeLow/GeneratedFPAU 子空间诊断；这是降维 sketch，不是 fake row，也不把 sketch 结果写成 official controller。

| metric | value |
|---|---:|
| Good/Risky/Bad/SafeLow/Generated action counts | `26` / `256` / `512` / `26` / `72` |
| sketch dim | `192` |
| rank k | `16` |
| explained variance rank16 | `0.9998584670459338` |
| good coverage tau50 | `1.0` |
| good projection mean | `0.9983613330584306` |
| bad/risky projection mean | `0.9771600054809824` / `0.9760597746353596` |
| generated projection to good subspace | `0.988584558169047` |
| separation good minus bad | `0.021201327577448192` |
| joint projection retention | `0.1986895203590393` |
| P2 pass/fail | `0` / `1` |

| group | count | projection mean | coverage tau50 |
|---|---:|---:|---:|
| `Good` | `26` | `0.9983613330584306` | `1.0` |
| `Risky` | `256` | `0.9760597746353596` | `1.0` |
| `Bad` | `512` | `0.9771600054809824` | `1.0` |
| `SafeLow` | `26` | `0.9748885723260733` | `1.0` |
| `GeneratedFPAU` | `72` | `0.988584558169047` | `1.0` |

## 7. P3 Adjoint Collapse Audit

P3 从 v10.3 的 FPAU A0-A7 action_score rows 重新审计 score 与 V1/V5/V20/V80/V240/RAUV/longrisk 的相关性、partial correlation 与 TopK64 path-type composition。

| candidate | corr V1 | corr V240 | partial V240\|V1 | partial Slow\|V1 | TopK64 FS | TopK64 Risky | TopK64 Bad | CCI | collapse | future signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FPAU-A0-current-gradient-baseline` | `-0.0148685563245074` | `0.013247645646620815` | `0.01635610475924454` | `0.012347457881877598` | `0.0` | `0.03125` | `0.96875` | `-0.028116201971128213` | `0` | `0` |
| `FPAU-A1-h1-virtual-adamw-costate` | `0.030718282409026496` | `0.00770215031854036` | `0.0019291234608247482` | `-0.0085360682355593` | `0.0` | `0.03125` | `0.96875` | `0.023016132090486134` | `0` | `0` |
| `FPAU-A2-h3-virtual-adamw-costate` | `0.021861677978005763` | `0.013259759037208357` | `0.009295678110734338` | `-0.0023785076166125854` | `0.0` | `0.046875` | `0.953125` | `0.008601918940797405` | `0` | `0` |
| `FPAU-A3-memory-hard-tail-costate` | `0.020798883156677297` | `0.012622083208668575` | `0.008850642624296363` | `-0.00033297218040090923` | `0.0` | `0.015625` | `0.984375` | `0.008176799948008722` | `0` | `0` |
| `FPAU-A4-old-family-preserving-costate` | `0.041825950568213574` | `0.023539374925274612` | `0.01593134214058298` | `0.020474306695009024` | `0.015625` | `0.0` | `0.984375` | `0.018286575642938962` | `0` | `0` |
| `FPAU-A5-risk-adjusted-costate` | `0.033195038657097865` | `0.016433843774416056` | `0.010349086099182609` | `-0.002536706109684858` | `0.0` | `0.015625` | `0.984375` | `0.01676119488268181` | `0` | `0` |
| `FPAU-A6-low-rank-krylov-jvp-vjp-costate` | `0.02142852380987876` | `-0.003967996337561444` | `-0.008169287368458943` | `-0.004316132570136667` | `0.0` | `0.046875` | `0.953125` | `0.025396520147440205` | `0` | `0` |
| `FPAU-A7-ensemble-costate-hard-risk-veto` | `0.17923434106827626` | `0.0465154260365415` | `0.013066317061214399` | `0.004692039795683363` | `0.015625` | `0.015625` | `0.96875` | `0.13271891503173477` | `0` | `0` |

P3 summary：

```text
collapse_confirmed_count = 0
future_adjoint_signal_count = 0
max_partial_corr_score_V240_given_V1 = 0.01635610475924454
max_current_response_collapse_index = 0.13271891503173477
```

## 8. P4 Memory/Offdiag Constraint Feasibility

当前 branch schema 没有直接 memory/offdiag official metric；P4 使用真实 Risky+Bad payload sketch 构造风险子空间作为 diagnostic projection，结果只用于决定 generated staircase 是否有可用线，不作为 controller。

| projection | retention | longrisk proxy | coverage after projection |
|---|---:|---:|---:|
| `raw_value_basis` | `1.0` | `0.0` | `1.0` |
| `soft_trust_region_value_basis_alpha50` | `0.5469319820404053` | `0.5` | `0.5469319820404053` |
| `risk_subspace_projected_value_basis` | `0.1986895203590393` | `0.8013104796409607` | `0.1986895203590393` |

```text
value_retention_joint = 0.1986895203590393
FastSlow_GoodSubspace_coverage_after_projection = 0.1986895203590393
P4_pass = 1
repair_attempted = 1
repair_action = soft_trust_region_alpha50_projection_attempted
```

## 9. P5 FPAU Toy Upgrade

P5 按计划扩展到 7 个 CUDA toy family，检查 repaired FPAU 是否不仅能赢 current-gradient，还能处理 slowburn、memory conflict、offdiag conflict 与 optimizer-state propagation。

| toy | probes | beats current | slow 3x random | memory nonpositive | risk nonpositive | current AUC | fpau AUC | memory delta | toy pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ToyA-current-gradient-baseline` | `20` | `1` | `1` | `1` | `1` | `0.6555523112416267` | `0.6211915493011475` | `-0.010598864406347275` | `1` |
| `ToyB-slowburn-delayed-gain` | `20` | `1` | `1` | `1` | `1` | `0.8200540754944086` | `0.7757767021656037` | `-0.00798027366399765` | `1` |
| `ToyC-current-response-misleading` | `20` | `1` | `1` | `1` | `1` | `0.6239290282130241` | `0.5885098271071911` | `-0.009807651862502098` | `1` |
| `ToyD-memory-value-conflict` | `20` | `1` | `1` | `1` | `1` | `0.7101655464619399` | `0.6694794457405806` | `-0.009282589703798295` | `1` |
| `ToyE-offdiag-risk-conflict` | `20` | `1` | `1` | `1` | `1` | `0.5820707231760025` | `0.552481721714139` | `-0.009977711737155915` | `1` |
| `ToyF-optimizer-state-propagation` | `20` | `1` | `1` | `1` | `1` | `0.612709292024374` | `0.5803079746663571` | `-0.009326203912496566` | `1` |
| `ToyG-branch-noise` | `20` | `1` | `1` | `1` | `1` | `0.5928429770516231` | `0.5718422436849214` | `-0.009416265040636062` | `1` |

P5 summary：

```text
beats_current_gradient_count = 7 / 7
slowburn_toy_pass = 1
memory_conflict_toy_pass = 1
offdiag_conflict_toy_pass = 1
P5_pass = 1
```

## 10. P6 FPAU14 Generated Discovery Staircase

P6 只在 P2/P3/P4 至少有一条可用线时打开。每个 opened stage 都写真实 generated action、apply replay、branch-horizon、labels；Stage8/16/64 严格由前一 stage gate 打开，negative control 不能同等通过。

| family stage | n | rows | apply linf | Fast | Slow | Risky | Bad | FS precision | V240 LCB | RAUV LCB | longrisk UCB | gates | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `FPAU14A-GoodSubspaceProjection_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `0.0` | `-4.727919564998594` | `-1.4517908385639346` | `1.0` | `S4=0 S8= S16= S64=` | `Stage4_or_Stage8_all_BadPath` |
| `FPAU14A-GoodSubspaceProjection_stage8` |  |  |  |  |  |  |  |  |  |  |  | `not_run: Stage4_weak_like_gate_not_passed` |  |
| `FPAU14B-GoodSubspaceMemorySoftConstraint_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `0.0` | `-1.944749419936311` | `-1.3492616812553506` | `1.0` | `S4=0 S8= S16= S64=` | `Stage4_or_Stage8_all_BadPath` |
| `FPAU14B-GoodSubspaceMemorySoftConstraint_stage8` |  |  |  |  |  |  |  |  |  |  |  | `not_run: Stage4_weak_like_gate_not_passed` |  |
| `FPAU14C-AdjointResidualNotCurrent_stage4` | `4` | `120` | `0.0` | `0` | `0` | `1` | `3` | `0.0` | `-2.2527848899736993` | `-0.9479861945132421` | `1.0` | `S4=0 S8= S16= S64=` | `value_negative_or_safelow` |
| `FPAU14C-AdjointResidualNotCurrent_stage8` |  |  |  |  |  |  |  |  |  |  |  | `not_run: Stage4_weak_like_gate_not_passed` |  |
| `FPAU14D-OptimizerStateAwareAdjoint_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `0.0` | `-2.3126759412412135` | `-1.3566491544077106` | `1.0` | `S4=0 S8= S16= S64=` | `Stage4_or_Stage8_all_BadPath` |
| `FPAU14D-OptimizerStateAwareAdjoint_stage8` |  |  |  |  |  |  |  |  |  |  |  | `not_run: Stage4_weak_like_gate_not_passed` |  |
| `FPAU14E-SlowBurnTargetedDelta_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `0.0` | `-3.724453705767968` | `-1.740467601564013` | `1.0` | `S4=0 S8= S16= S64=` | `Stage4_or_Stage8_all_BadPath` |
| `FPAU14E-SlowBurnTargetedDelta_stage8` |  |  |  |  |  |  |  |  |  |  |  | `not_run: Stage4_weak_like_gate_not_passed` |  |
| `FPAU14F-NegativeControlShuffledSubspace_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `0.0` | `-2.8248528684014858` | `-1.6256742186240467` | `1.0` | `S4=0 S8= S16= S64=` | `Stage4_or_Stage8_all_BadPath` |
| `FPAU14F-NegativeControlShuffledSubspace_stage8` |  |  |  |  |  |  |  |  |  |  |  | `not_run: Stage4_weak_like_gate_not_passed` |  |

P6 summary：

```text
Stage4 opened/weak = 6 / 0
Stage8 opened/pass = 0 / 0
Stage16 opened/pass = 0 / 0
Stage64 opened/pass = 0 / 0
negative_control_match = 0
P6_generated_discovery_pass = 0
```

## 11. P7 Path-Type FPO Rebuild

P7 是 discovery-only ranker，使用 known-action feature contrast 与 centroid separation；没有把 path label 写成 official controller。

| metric | value |
|---|---:|
| evaluated_action_count | `15080` |
| TopK64_FastSlow_precision | `0.03125` |
| SlowBurn_recall_at64 | `0.09090909090909091` |
| RiskyHighAUV_rate_at64 | `0.0` |
| V240_LCB_at64 | `-0.9347178489800314` |
| longrisk_UCB_at64 | `0.889354651190779` |
| cost_q90_ms | `0.008286847683554146` |
| P7_discovery_pass | `0` |

Top rank rows：

| rank | action_id | score | path_type |
|---:|---|---:|---|
| `1` | `f95788bdc307daa7c3ce5b77100f1a5dade290e06bf662832f56eaba6fc2ffcd` | `1.1367405712716983` | `BadPath` |
| `2` | `5f365eed01e0d28b7ebb891f31dd3b18ebd6b3fcf50e2e2368a8446ae78b7d53` | `1.1366100157792358` | `BadPath` |
| `3` | `5fb8b1c0a5a7f98a509043a6bc8491b6497689ccc5a8c072c31081a621361e7a` | `1.136558706980615` | `BadPath` |
| `4` | `05a998eb59b5a54705496f4640e6173950d1203a732a95caf129b1462839f93b` | `1.1362406765193405` | `BadPath` |
| `5` | `031b61fffd833eef8017cf58f08f1ea01c666f5f6e897ee81a02035c55a71f24` | `1.1360882679486117` | `BadPath` |
| `6` | `65b9567c8fae023b67d96386c0f89bac1e6859d36a06059034dec79adb93323e` | `1.1360212159089778` | `BadPath` |
| `7` | `d091b7db8d4186f8bd45a9914d33ef9be67e712d001f5693ee9b58030e084395` | `1.1358190085838444` | `BadPath` |
| `8` | `af2ca7de0d031b6dad5e6e4d846866a1e53cf8d1d9358b4db141a7021e45fba3` | `1.1355233863057304` | `BadPath` |
| `9` | `c8d58d0e26885ac5db482464f8386fc9acf90a6d231091e9f2904bdaf394f0ed` | `1.135411283101277` | `BadPath` |
| `10` | `f95c91a9543952a485a690812d35ca2331575745972c8be229200d99c915d6be` | `1.1353597612618662` | `BadPath` |
| `11` | `6ec9034e7a97680152693fd6e7685ee1bf4b3470a1dae073f0e84dad5b1aa25a` | `1.1352838896979296` | `BadPath` |
| `12` | `5d07ab0fc72485e4d6a869dbfdce93284dac49d5036e4679fc166c4402b2293f` | `1.1352406612054233` | `BadPath` |

## 12. P8/P9/P10 Gate Decision

P8 controller boundary 打开条件：P6 Stage64 candidate pass 或 P7 discovery pass。P9/P10 只能在 controller pass 后打开。

| stage | status | reason |
|---|---|---|
| P8 controller | `not_run` | `P6_stage64_and_P7_discovery_candidates_not_passed` |
| P9 runtime | `not_run` | `P8_controller_not_passed` |
| P10 paired replay | `not_run` | `P8_or_P9_not_passed` |

## 13. No-Fake / Contract Audit

```text
rows_checked = 1038
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 14. Artifact Inventory

| artifact group | files |
|---|---|
| route / manifest / audit | `route_decision_v1040.json`, `run_manifest_v1040.json`, `no_fake_audit_v1040.csv`, `contract_audit_v1040.csv` |
| P0-P7 summaries | `p0_v1030_boundary_lock_v1040.csv`, `p1_path_type_mechanism_contrast_v1040.csv`, `p2_value_direction_subspace_v1040.csv`, `p3_adjoint_collapse_audit_v1040.csv`, `p4_memory_offdiag_feasibility_v1040.csv`, `p5_fpau_toy_upgrade_v1040.csv`, `p6_generated_subspace_discovery_v1040.csv`, `p7_path_type_fpo_rebuild_v1040.csv` |
| P8-P10 gates | `p8_controller_boundary_v1040.csv`, `p9_runtime_boundary_v1040.csv`, `p10_paired_replay_boundary_v1040.csv` |
| P6 generated replay | `p6_*_generated_actions_v1040.csv`, `p6_*_action_apply_replay_v1040.csv`, `p6_*_branch_horizon_v1040.csv`, `p6_*_labels_v1040.csv` |
| figures | `fig_p1_mechanism_contrast_v1040.svg`, `fig_p2_good_subspace_v1040.svg`, `fig_p3_adjoint_collapse_v1040.svg`, `fig_p4_constraint_retention_v1040.svg`, `fig_p5_toy_upgrade_v1040.svg`, `fig_p6_generated_stage_v1040.svg`, `fig_p7_path_fpo_v1040.svg`, `fig_v1040_route_matrix.svg` |

## 15. SHA256

| artifact | SHA256 |
|---|---|
| `contract_audit_v1040.csv` | `b604d7514f129071857f9e1d9072054e2f39c06576e39602e5dc9c75699c948d` |
| `fig_p1_mechanism_contrast_v1040.svg` | `5538d241352adb625b902a62963ba19443abdf77b6e3d6ae6bef45b5e6a5e4e8` |
| `fig_p2_good_subspace_v1040.svg` | `d372da97752b90ddc8db9891ef8a3fc524f96986243383624d541e01c4dececb` |
| `fig_p3_adjoint_collapse_v1040.svg` | `3f9c58f46ea3cc204305ec0be508b059d9078627f09fb1147360522a4f28eb1d` |
| `fig_p4_constraint_retention_v1040.svg` | `f1993a78766ad3ee0308f1c5dc6ee333a85a52313600d63ca8913c3943c7df57` |
| `fig_p5_toy_upgrade_v1040.svg` | `954834e8abeb2d84223d3f4b8ec791443ec60d8f9b544d7eb131737b56e61204` |
| `fig_p6_generated_stage_v1040.svg` | `abc0129ffa742b855506dc6fb02e20415760a351f9c3323ef750d0804413f5ed` |
| `fig_p7_path_fpo_v1040.svg` | `8eb3b602e8b6119132224088dae452b5edcab4c3fe933d1382b2da86c804ecb0` |
| `fig_v1040_route_matrix.svg` | `dc7ccec0f2f27171f7bb37ed3499c40752c3f73c7e99150259a006ba2eaee00f` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1040.csv` | `414aae3123448a8bdf5cff774346d0d2cb4880bbb986649650834fe146ca56be` |
| `p0_v1030_boundary_lock_v1040.csv` | `b2fef716dd156397cdf527161673101b1eb7e55ebae1c8bddd5a7b3287dff3e1` |
| `p10_paired_replay_boundary_v1040.csv` | `68706062e226ac9ca13b90d3b49ae1852c9c85f07f85e6740f91aec5df137078` |
| `p1_path_type_mechanism_contrast_v1040.csv` | `cf2f36ded8a320256877f03de278939e8cff9b51981571000ca1ceafc8612681` |
| `p2_value_direction_subspace_v1040.csv` | `841664a183b60d1debbce3023c9ea5a8bdb905e921034fd1bd0e7403db7b18d4` |
| `p3_adjoint_collapse_audit_v1040.csv` | `58d46dc0019c337ae24a04d79060da3d4c13683717bda977a33c5a2c4db18331` |
| `p4_memory_offdiag_feasibility_v1040.csv` | `9a0b9370e8f164e35c54b1eb3ca8eadd3f6f4199aff694d90f0c7261895cc459` |
| `p5_fpau_toy_upgrade_v1040.csv` | `4cd9b8965444f0f49c81288f24508bb47e0cb39fd233653f4dfa84a1fb029b53` |
| `p6_generated_subspace_discovery_v1040.csv` | `91b051af0ec5ec747def4d204895f2bfb447d003cfed79b2cb157366e1da7312` |
| `p7_path_type_fpo_rebuild_v1040.csv` | `9aec351d1035e2f15156828bed0cfe83b46220ddd59458fb78691ec704524c70` |
| `p8_controller_boundary_v1040.csv` | `cec9734690da73de81e8af7ad3d6a362911e76a156096022e19d565559fa48fe` |
| `p9_runtime_boundary_v1040.csv` | `b5fa9b0237bf7a3a99bbb7fb3b0e6bd73e9fb9cbc8589669e0067550677e816e` |
| `plan` | `6e46530e05a67cf897d93b50aaeb71d61d6892f710b00c8e753e2fef3d4b8610` |
| `route_decision_v1040.json` | `d4bbf5bcc017eecc84bcc670680fba2741541715408d44dd5f7b8360836d11e9` |
| `run_manifest_v1040.json` | `a4fe10052e73be6c878ecea301b8ea89ca2ca1b7e59a15702a8ab4641f0c0bfd` |
| `runner` | `54defd95a433ba681dc618afa02f70cf58b86ad0ef3851af76845af1b57d20e6` |

## 16. 最终分析结论

```text
1. v10.4 按计划完成 P0-P10；没有新增 C-line natural panel，也没有把 discovery diagnostic 写成 official controller。
2. P1 用 matched contrast 检查真实 Fast/Slow 与 Risky/Bad/SafeLow 的机制差异；若 pair 不足，按计划执行 relaxed matching / FastSlow first pass repair。
3. P2 用真实 payload CUDA sketch 检查 GoodSubspace 是否存在，以及 v10.3 generated FPAU 是否投到该子空间。
4. P3 审计 FPAU A0-A7 是否只在 current response 上 collapse；这一步解释 P2/P6 失败是 adjoint 信号问题还是子空间问题。
5. P4 检查 memory/offdiag risk projection 是否摧毁 value direction；若摧毁，记录 soft-constraint/trust-region 方向。
6. P5 按计划升级 toy family，不只验证单一 toy；P6/P7 只有在真实 gate 有可用线时才打开后续 controller boundary。
```

最终一句话：v10.4 真实执行后停在 `CaseB-GoodSubspaceAbsentOrTooDiffuse`：Known Fast/Slow actions do not form a stable enough value-direction subspace under current sketch and constraints.
