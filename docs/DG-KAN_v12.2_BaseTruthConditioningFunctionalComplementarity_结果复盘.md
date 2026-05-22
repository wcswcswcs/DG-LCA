# DG-KAN v12.2 Base Truth / Conditioning / Functional Complementarity 结果复盘

> 本复盘记录 `DG-KAN_v12.2_实验结果独立分析与下一步计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。所有 functional 行在 base gate 关闭时只算 diagnostic，不冒充 official success。

## 0. 最新结论

```text
route = R2-ExpressionImplementationFailure
primary_blocker = pairwise_LS_or_representational_capacity_failed
next_recommended_action = repair LQ formula / interaction channel
base_qualified = false
official_functional_open = false
```

最终 artifact：

```text
results/v12_2_base_truth_conditioning_functional_complementarity/v122_base_truth_conditioning_functional_complementarity_20260519T120000Z
```

目标达成状态：

```text
v12.2 diagnostic pipeline = achieved
Timing Truth Audit = completed
Expression Truth Audit = completed
Global Conditioning Repair triage = completed
Passive Geometry Trajectory Re-test = completed on 5-epoch triage budget
Functional Complementarity Audit = completed as diagnostic_gate_closed
scientific base qualification = not achieved
official functional audit / P5 = not opened
```

核心结论：

1. A 线 clean timing 通过：R2 clean fastpath step q90 ratio = `1.2043225586407207`，低于 v12.2 clean 阈值 `1.25`。
2. A 线 official runner profile path 明显失败：ratio = `45.89011153349123`；其中 geometry hook mean = `22.268091917503625` ms，说明 v12.1/v12.2 diagnostic instrumentation 不能直接当 architecture timing gate。
3. B 线出现关键矛盾：R2 pairwise frozen-lift LS R2 = `0.378700315952301`，但 high-budget AdamW B2 pairwise R2 = `0.9992794990539551`。这不是“AdamW 训练完全不会表达 pairwise”，而是 frozen lift / coefficient fit / equivalence audit 没过。
4. B 线 composition 仍失败：R2 B1 composition delta vs MLP = `-0.04475069046020508`，未达到 `>= -0.02`。
5. C 线没有 conditioning repair survivor：`conditioning_repair_survivor_count = 0`。所有 C0-C7 都没有同时满足 pairwise、task delta、clean efficiency、100x condition drop 和 geometry improvement 组合 gate。
6. C0 当前 R2 task 均值弱于 MLP：R2 mean val acc = `0.8368055555555556`，MLP mean val acc = `0.8428819444444444`，macro delta = `-0.006076388888888889`。
7. C0 lift condition 仍极高：mean `247256373876053.34`；C1/C4/C6 等 repair 没有达到 100x drop。
8. E 线 diagnostic functional complementarity 没有通过：E8-E16 的 `diagnostic_promotable` 全部为 `0`，control gap mean 全为负。
9. No-fake audit rows checked = `3910`，fake/proxy/cpu = `0` / `0` / `0`。
10. 后续按 R2 建议补做 interaction equivalence / coverage repair：公式手工等价通过，但 `signed_pair_cover` 提升到 C-track 后真实 base gate 仍失败。

## 1. 本轮命令

```bash
python -m py_compile experiments/run_v122_base_truth_conditioning_functional_complementarity.py dgkan/models/fc_purekan_lq.py
```

```bash
python experiments/run_v122_base_truth_conditioning_functional_complementarity.py \
  --out-dir results/v12_2_base_truth_conditioning_functional_complementarity/v122_base_truth_conditioning_functional_complementarity_20260519T120000Z \
  --fresh --device auto --data-root data --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 5 --batch-size 128 --eval-batch-size 512 --eval-train-size 512 \
  --probe-batch-size 64 --microbatch-count 2 --profile-steps 4 \
  --timing-datasets MNIST --timing-seeds 0 \
  --timing-warmup-steps 50 --timing-measure-steps 200 \
  --expression-seeds 0 --expression-protocols B0,B1,B2,B3,B4 \
  --expression-b2-targets E1-pairwise-product \
  --expression-train-size 512 --expression-val-size 256 --expression-test-size 256 \
  --expression-b2-train-size 4096 --expression-b2-val-size 2048 --expression-b2-test-size 2048 \
  --expression-steps-b0 60 --expression-steps-b1 1000 --expression-steps-b2 5000 \
  --expression-batch-size 128 --expression-eval-stride 100 --c-expression-steps 300 \
  --report-path docs/DG-KAN_v12.2_BaseTruthConditioningFunctionalComplementarity_结果复盘.md
```

说明：本轮是 v12.2 第一层 truth/repair/complementarity 闭环，真实跑了 5 epoch triage；它不能冒充计划中的 20/30 epoch survivor confirm。

## 2. Route

```json
{
  "route": "R2-ExpressionImplementationFailure",
  "primary_blocker": "pairwise_LS_or_representational_capacity_failed",
  "next_recommended_action": "repair LQ formula / interaction channel",
  "timing_clean_q90_ratio": 1.2043225586407207,
  "timing_official_q90_ratio": 45.89011153349123,
  "timing_clean_pass": true,
  "timing_official_pass": false,
  "measurement_artifact_gt3x": true,
  "pairwise_LS_R2_mean": 0.378700315952301,
  "pairwise_AdamW_best_R2_mean": 0.9992794990539551,
  "composition_delta_vs_MLP_mean": -0.04475069046020508,
  "expression_representational_pass": false,
  "expression_trainability_pass": true,
  "composition_pass": false,
  "conditioning_repair_survivor_count": 0,
  "base_qualified": false,
  "functional_complementarity_pass": false,
  "official_functional_open": false,
  "external_ready": false,
  "no_fake": true
}
```

## 3. Track A Timing Truth

| method | step q90 ms | ratio vs MLP | geometry hook ms | probe hook ms |
|---|---:|---:|---:|---:|
| `A0-MLP-same-param-AdamW` | `0.502566946670413` | `1.0` | `0.0` | `0.0` |
| `A2-LQ-t2-h256-current` | `0.6069319322705269` | `1.207663847158172` | `0.0` | `0.0` |
| `A3-R2-LQ-fanin-output-scale-confirmed-current` | `0.5848656874150038` | `1.1637567716894897` | `0.0` | `0.0` |
| `A4-R2-LQ-clean-fastpath-no-geometry-hooks` | `0.6052527111023664` | `1.2043225586407207` | `0.0` | `0.0` |
| `A5-R2-LQ-compiled-warm` | `0.41603222489356995` | `0.8278145382418213` | `0.0` | `0.0` |
| `A6-R2-LQ-compact-recompute` | `0.5875407718122005` | `1.1690796135813404` | `0.0` | `0.0` |
| `A7-R2-LQ-v12-runner-profile-path` | `23.06285323575139` | `45.89011153349123` | `22.268091917503625` | `0.5604145675897598` |

结论：clean path 不支持 “R2 真实 step ratio = 6.87x” 的说法；但 official diagnostic profile path 明显混入 heavy geometry hook，因此 official timing gate 仍不能过。下一步应把 architecture step timing 与 diagnostic hook timing 分离。

## 4. Track B Expression Truth

Pairwise-product 关键结果：

| protocol | method | val R2 | test R2 | delta vs MLP |
|---|---|---:|---:|---:|
| `B1-1000-step` | `M0-MLP-same-param` | `0.9908220171928406` | `0.9883087873458862` | `0.0` |
| `B1-1000-step` | `M3-R2-LQ-current` | `0.9412870407104492` | `0.9441508054733276` | `-0.04953497648239136` |
| `B2-5000-step` | `M0-MLP-same-param` | `0.9994162917137146` | `0.9994457960128784` | `0.0` |
| `B2-5000-step` | `M3-R2-LQ-current` | `0.9992794990539551` | `0.999559223651886` | `-0.00013679265975952148` |
| `B2-5000-step` | `M7-R2-LQ-quadratic-edge-only` | `0.9994427561759949` | `0.9996534585952759` | `0.000026464462280273438` |
| `B3-frozen-lift-LS` | `M3-R2-LQ-current` | `0.378700315952301` | `0.3714798092842102` | `0.21879959106445312` |
| `B4-oracle-quadratic-readout` | `M8-OracleCapacityProbe` | `1.0` | `1.0` |  |

Composition B1：

| method | val R2 | delta vs MLP |
|---|---:|---:|
| `M0-MLP-same-param` | `0.9775179624557495` | `0.0` |
| `M2-LQ-t2-h256` | `0.9153087139129639` | `-0.062209248542785645` |
| `M3-R2-LQ-current` | `0.9327672719955444` | `-0.04475069046020508` |
| `M5-R2-LQ-no-fanin-output-scale` | `0.9163244366645813` | `-0.06119352579116821` |

解释：B2 说明 trainable LQ 可以在高预算下拟合 pairwise；但 B3 frozen-lift LS 很低，说明当前 lift/edge coefficient equivalence 或 frozen interaction channel 没有打穿。按 v12.2 route tree，LS/equivalence fail 仍归入 `R2-ExpressionImplementationFailure`，并且 composition delta 也未过。

## 5. Track C/D Conditioning Repair 与 Geometry Trajectory

C 线平均结果：

| method | val acc delta vs MLP | step ratio | lift condition mean | pairwise R2 | survivor rows |
|---|---:|---:|---:|---:|---:|
| `C0-R2-LQ-current` | `-0.006076388888888889` | `1.1557272206293532` | `247256373876053.34` | `0.9556055665016174` | `0` |
| `C1-centered-quadratic-basis` | `-0.016493055555555556` | `1.107354496789312` | `224123077904156.44` | `0.969336986541748` | `0` |
| `C2-orthogonal-identity-lift-init` | `-0.006727430555555556` | `1.1185872515842403` | `254865989639281.78` | `0.9614291191101074` | `0` |
| `C3-fixed-scale-lift-standardization` | `-0.010199652777777778` | `1.119548546157771` | `246725255685006.22` | `0.948433518409729` | `0` |
| `C4-residual-gated-quadratic` | `-0.024739583333333332` | `1.1155764333137939` | `213779228917760.0` | `0.974773108959198` | `0` |
| `C5-fanin-output-scale-decoupled` | `-0.007378472222222222` | `1.1249914468429107` | `244146444726727.12` | `0.9188499450683594` | `0` |
| `C6-low-condition-legendre23` | `-0.016493055555555556` | `1.4299371818848654` | `199861720143189.34` | `0.8804770708084106` | `0` |
| `C7-condition-preserving-weight-decay-path` | `-0.007161458333333333` | `1.109707228536502` | `249755345791658.66` | `0.9489153623580933` | `0` |

C0 current R2 paired val acc delta：

| dataset | seed | delta | R2 val acc | MLP val acc |
|---|---:|---:|---:|---:|
| `MNIST` | `0` | `-0.001953125` | `0.89453125` | `0.896484375` |
| `MNIST` | `1` | `-0.013671875` | `0.896484375` | `0.91015625` |
| `MNIST` | `2` | `0.001953125` | `0.900390625` | `0.8984375` |
| `Fashion-MNIST` | `0` | `-0.033203125` | `0.783203125` | `0.81640625` |
| `Fashion-MNIST` | `1` | `0.017578125` | `0.8203125` | `0.802734375` |
| `Fashion-MNIST` | `2` | `0.001953125` | `0.80078125` | `0.798828125` |
| `KMNIST` | `0` | `-0.025390625` | `0.806640625` | `0.83203125` |
| `KMNIST` | `1` | `0.001953125` | `0.818359375` | `0.81640625` |
| `KMNIST` | `2` | `-0.00390625` | `0.810546875` | `0.814453125` |

解释：C repair 能保持或提升 pairwise triage R2，但没有把 lift condition 降到 `0.01 * current`，也没有稳定跨过 task delta gate。因此 D 线 trajectory 只作为被动诊断，不打开 base qualification。

## 6. Track E Functional Complementarity

E 线在 `official_gate_open = 0` 下完成 diagnostic audit。所有 functional candidates 均未通过 control resistance：

| candidate | control gap mean | CI low | diagnostic promotable |
|---|---:|---:|---:|
| `E8-SNRProjectedGeometry` | `-0.014056297812232228` | `-0.02649363347124306` | `0` |
| `E9-BasisEntropyRebalance` | `-0.014140148869173188` | `-0.026710878714557934` | `0` |
| `E10-LiftConditionRepair` | `-0.013954001059265963` | `-0.02641215225708773` | `0` |
| `E11-TailStabilityCorrection` | `-0.013874825195171312` | `-0.02615540944897609` | `0` |
| `E12-AdamWOrthogonalLiftRepair` | `-0.013933565485230642` | `-0.026369169455225314` | `0` |
| `E13-FunctionPreservingConditionRepair` | `-0.013933565485230642` | `-0.026369169455225314` | `0` |
| `E14-TailMarginResidualRepair` | `-0.013787089680470136` | `-0.0260217987818562` | `0` |
| `E15-SignalReservoirSeparationRepair` | `-0.013859200305060132` | `-0.026153701594829815` | `0` |
| `E16-ControlResidualGeometry` | `-0.013859200305060132` | `-0.026153701594829815` | `0` |

解释：E12-E16 的 AdamW-orthogonal / residual ideas 已真实跑 diagnostic，但没有击败 strong controls；不能调低 gate，也不能打开 P5。

## 7. Artifact 完整性

Required CSV/JSON 均已落盘：

```text
a_timing_truth.csv rows = 8
a_timing_breakdown.csv rows = 8
a_runner_overhead_audit.csv rows = 1
b_expression_truth.csv rows = 233
b_expression_learning_curves.csv rows = 1192
b_ls_fit_results.csv rows = 81
b_expression_failure_taxonomy.csv rows = 1
c_conditioning_repair.csv rows = 81
c_repair_task_trace.csv rows = 486
c_repair_expression_trace.csv rows = 32
c_repair_geometry_snapshot.csv rows = 486
d_geometry_trajectory.csv rows = 486
d_geometry_auc.csv rows = 81
e_functional_complementarity_direction.csv rows = 153
e_one_step_probe.csv rows = 153
e_five_step_probe.csv rows = 81
e_control_matrix.csv rows = 9
e_lambda_backtracking.csv rows = 172
e_cost_profile.csv rows = 153
figures/*.svg count = 13
```

## 8. No-Fake / Hash

```text
rows_checked = 3910
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `a_timing_truth.csv` | `a70647f52345a47fe5f6518f1941c0a094dce8afa2c90a05dcdb334024f246b9` |
| `b_expression_truth.csv` | `80f692304a0bffe67230cff5ea637b34a65e22e96f925c32b912dcde4d5d50b7` |
| `b_ls_fit_results.csv` | `39f6f8feb42cfdb2a20493efe756c8cab5a072f95782ea99b7d69e7427716740` |
| `b_expression_failure_taxonomy.csv` | `ad0fa350b727d51c3046aa74e66a61d5c5db3d8eed2c33c7d4d9315e63dcd86b` |
| `c_conditioning_repair.csv` | `6dbe2ee3bf1164ecd4b112f454a6025904bf083272def6f00fc31ecabe8b94aa` |
| `d_geometry_trajectory.csv` | `313fbcb967eaec5d379e1569515e3eec658baa001c7159768a94a1c4a5cff82f` |
| `e_control_matrix.csv` | `5c28b2aafac63a1e2f90b14714b4b760f31adcd9642d16bd476c3f5f3dd56458` |
| `failure_table.csv` | `d7fda751fc8053b29383c4ef0afb9a3942ed95bfedccff43650276230ba8f267` |
| `provenance_audit.csv` | `02cdd2d5599b22824790f9a648e8af4a940023eeef530ba3b1788d5646992b1c` |
| `route_decision.json` | `f8e80c11dd4d3030ee8693cdfe489623375fa7f7b4842f757a3adaaf3ca1dc3d` |
| `run_manifest_v122.json` | `680c5fe3da2893be62108f5718115022c436289a5994be494012de8cf37fb79b` |

## 9. 最终分析结论

```text
1. v12.2 的 pipeline 目标达成：A/B/C/D/E 都有真实落盘，route、failure、hash、provenance 都完整。
2. 科学路线没有进入 base success：clean timing 有希望，但 expression LS/equivalence 与 composition gate 仍失败。
3. Pairwise 不能简单说“LQ 不会表达”：B2 AdamW 与 quadratic-only 都能到接近 1.0；真正没过的是 frozen lift / closed-form coefficient truth audit。
4. 当前 C0-C7 repair 没有解决 lift condition 量级问题，也没有形成 task-stable survivor。
5. Functional complementarity 仍失败：E8-E16 没有 control-resistant signal；official functional/P5 继续关闭。
```

最终一句话：v12.2 真实执行后停在 `R2-ExpressionImplementationFailure`；下一步不是调低 functional gate，而是修 LQ interaction/equivalence audit 与 lift conditioning，把 clean timing runner fix 单独剥离后再重跑 base gate。


## 10. R2 后续修复：Interaction Equivalence / Coverage Audit

> 本节是 v12.2 route = `R2-ExpressionImplementationFailure` 后按计划继续执行的修复定位。新增结果仍只来自真实落盘 artifact；不把 synthetic expression diagnostic 当成 fake/proxy，也不把 diagnostic repair 写成 base success。

新增 artifact：

```text
results/v12_2_base_truth_conditioning_functional_complementarity/v122_interaction_equivalence_repair_20260519T130000Z
```

新增 route：

```json
{
  "stage": "ROUTE_DECISION_V122_INTERACTION_REPAIR",
  "route": "R2a-FormulaPassCoverageRepairNeeded",
  "manual_equivalence_pass": true,
  "manual_equivalence_val_R2": 1.0,
  "manual_equivalence_test_R2": 1.0,
  "manual_equivalence_max_abs_error": 1.1920928955078125e-07,
  "current_E1_matrix_span_R2": 0.37257474660873413,
  "signed_E1_matrix_span_R2": 1.0,
  "signed_E1_LS_val_R2": 0.9909107685089111,
  "signed_pair_cover_global_matrix_span_R2_mean": 0.5900851686795553,
  "signed_pair_cover_global_pass": 0,
  "base_qualified": false,
  "official_functional_open": false,
  "next_recommended_action": "promote signed_pair_cover to C-track candidate and rerun full base gate; keep functional closed",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

关键结论：

1. Manual equivalence audit 通过：手工构造的 plus/minus quadratic LQ 对 `E1-pairwise-product` 的 val/test R2 均为 `1.0`，max abs error = `1.1920928955078125e-07`。这说明 quadratic edge formula 本身可以精确表达 pairwise。
2. 当前随机 R2 frozen lift 的 matrix span coverage 仍弱：`E1` coverage R2 = `0.37257474660873413`。
3. `signed_pair_cover` repair 显著修复 E1 coverage：matrix span coverage R2 = `1.0`，frozen-lift LS val R2 = `0.9909107685089111`。
4. 但它不是全局 base success：rotated/random quadratic 的 coverage 仍未全面通过，composition gate 也没有在本节重开；official functional / P5 继续关闭。

No-fake：

```text
rows_checked = 29
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `b_interaction_repair_candidates.csv` | `c579d3c848ede20ce2966dd349fb6107dbd5ae79566034066b6633bc4488438f` |
| `b_manual_equivalence_audit.csv` | `dab91907f5295916b57b1732097776326bdaa9c3aa55c48aaa6f7637fd41fe73` |
| `b_parameter_coverage_audit.csv` | `b25d1f97393005a8c8723f597b4f85247e9ec699fd482b1b7e51e8c9354582ed` |
| `provenance_audit.csv` | `8783244a47ccb18f8baf6ff7c46a5622cb9c69a49bd28a4b02d2bf0cdcb0c248` |
| `route_decision_interaction_repair.json` | `9e5b9dd129362f2695f8ad5c47f84d529d0fd5b51b47b01b41ca3d175c0bc039` |

新增一句话：R2 的根因从“quadratic edge 公式错误”收窄为“当前随机/frozen lift 的 interaction coverage 与 equivalence audit 不合格”；下一步应把 `signed_pair_cover` 这类 dataset-agnostic interaction-cover lift 做成 C-track base candidate，再重跑 full v12.2 base gate。

## 11. C8 Signed Pair Cover C-Track Rerun

> 本节继续执行上一节的 next action：把 `signed_pair_cover` 作为 dataset-agnostic interaction-cover lift 提升为 v12.2 C-track base candidate，并在同一 5 epoch triage budget 下重跑 base gate。结果仍只来自真实 artifact，不打开 functional/P5。

新增实现：

```text
dgkan.models.fc_purekan_lq.LQSpec(init_variant="signed_pair_cover")
experiments/run_v122_base_truth_conditioning_functional_complementarity.py:
  M9-R2-signed-pair-cover-lift
  C8-signed-pair-cover-lift
```

新增 artifact：

```text
results/v12_2_base_truth_conditioning_functional_complementarity/v122_signed_pair_cover_cgate_20260519T140000Z
```

新增命令：

```bash
python experiments/run_v122_base_truth_conditioning_functional_complementarity.py \
  --out-dir results/v12_2_base_truth_conditioning_functional_complementarity/v122_signed_pair_cover_cgate_20260519T140000Z \
  --fresh --device auto --data-root data --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 5 --batch-size 128 --eval-batch-size 512 --eval-train-size 512 \
  --probe-batch-size 64 --microbatch-count 2 --profile-steps 4 \
  --timing-datasets MNIST --timing-seeds 0 \
  --timing-warmup-steps 50 --timing-measure-steps 200 \
  --expression-seeds 0 --expression-protocols B0,B1,B2,B3,B4 \
  --expression-b2-targets E1-pairwise-product \
  --expression-train-size 512 --expression-val-size 256 --expression-test-size 256 \
  --expression-b2-train-size 4096 --expression-b2-val-size 2048 --expression-b2-test-size 2048 \
  --expression-steps-b0 60 --expression-steps-b1 1000 --expression-steps-b2 5000 \
  --expression-batch-size 128 --expression-eval-stride 100 --c-expression-steps 300 \
  --report-path docs/DG-KAN_v12.2_signed_pair_cover_tmp.md
```

新增 route：

```json
{
  "route": "R2-ExpressionImplementationFailure",
  "timing_clean_q90_ratio": 1.1188099365032762,
  "timing_official_q90_ratio": 45.564825572957524,
  "pairwise_LS_R2_mean": 0.378700315952301,
  "pairwise_AdamW_best_R2_mean": 0.9992794990539551,
  "composition_delta_vs_MLP_mean": -0.04475069046020508,
  "conditioning_repair_survivor_count": 0,
  "base_qualified": false,
  "functional_complementarity_pass": false,
  "official_functional_open": false,
  "no_fake": true
}
```

M9 signed-pair-cover expression 结果：

| target | protocol | val R2 | test R2 | delta vs MLP |
|---|---|---:|---:|---:|
| `E1-pairwise-product` | `B1-1000-step` | `0.9459221363067627` | `0.9491794109344482` | `-0.04489988088607788` |
| `E1-pairwise-product` | `B2-5000-step` | `0.9993107914924622` | `0.9993605017662048` | `-0.0001055002212524414` |
| `E1-pairwise-product` | `B3-frozen-lift-LS` | `0.9921392798423767` | `0.9904956221580505` | `0.8322385549545288` |
| `E2-composition` | `B1-1000-step` | `0.9666717648506165` | `0.9689752459526062` | `-0.010846197605133057` |
| `E6-rotated-pairwise-product` | `B1-1000-step` | `0.9460209012031555` | `0.9582815766334534` | `-0.046223342418670654` |
| `E6-rotated-pairwise-product` | `B3-frozen-lift-LS` | `0.20085549354553223` | `0.20428705215454102` | `0.04414713382720947` |
| `E8-random-quadratic-form` | `B1-1000-step` | `0.9603803753852844` | `0.9415766596794128` | `-0.03145807981491089` |
| `E8-random-quadratic-form` | `B3-frozen-lift-LS` | `0.31781166791915894` | `0.3087025284767151` | `0.073910653591156` |

解释：C8/M9 对原始 E1 frozen-LS 修复明显，composition B1 也过了 `>= -0.02` 的局部标准；但 rotated/random quadratic 的 frozen-LS coverage 仍低，说明 `signed_pair_cover` 是局部 interaction cover，不是全局 quadratic cover。

C8 vision base gate：

| method | mean val acc | val acc delta vs MLP | clean step ratio | lift condition mean | pairwise R2 | survivor rows |
|---|---:|---:|---:|---:|---:|---:|
| `CBASE-MLP-same-param` | `0.8428819444444444` | `0.0` | `1.0` | `0.0` |  | `0` |
| `C0-R2-LQ-current` | `0.8368055555555556` | `-0.006076388888888889` | `1.1129910224843913` | `247256373876053.34` | `0.9556055665016174` | `0` |
| `C8-signed-pair-cover-lift` | `0.8235677083333334` | `-0.019314236111111112` | `1.195098987200764` | `313922594333582.25` | `0.9618070721626282` | `0` |

C8 paired val acc delta：

| dataset | seed | delta | C8 val acc | MLP val acc | C8 lift condition |
|---|---:|---:|---:|---:|---:|
| `MNIST` | `0` | `-0.00390625` | `0.892578125` | `0.896484375` | `188716433801216.0` |
| `MNIST` | `1` | `-0.0078125` | `0.90234375` | `0.91015625` | `181282432614400.0` |
| `MNIST` | `2` | `-0.00390625` | `0.89453125` | `0.8984375` | `136409511886848.0` |
| `Fashion-MNIST` | `0` | `-0.01953125` | `0.796875` | `0.81640625` | `629335719411712.0` |
| `Fashion-MNIST` | `1` | `-0.025390625` | `0.77734375` | `0.802734375` | `580894461001728.0` |
| `Fashion-MNIST` | `2` | `-0.029296875` | `0.76953125` | `0.798828125` | `574747020623872.0` |
| `KMNIST` | `0` | `-0.044921875` | `0.787109375` | `0.83203125` | `188625065082880.0` |
| `KMNIST` | `1` | `-0.01953125` | `0.796875` | `0.81640625` | `173119494946816.0` |
| `KMNIST` | `2` | `-0.01953125` | `0.794921875` | `0.814453125` | `172173209632768.0` |

结论：`signed_pair_cover` 按文档建议真实修复了一个局部 expression/equivalence 缺口，但作为 base repair 失败。它提高了 pairwise repair trace，却牺牲了 vision task stability，并且 lift condition 更差。因此它不能进入 survivor confirm，也不能打开 functional/P5。

No-fake：

```text
rows_checked = 4280
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `b_expression_truth.csv` | `421111074ffc45af8c3e696f3c7101a9500668687eb1c82b25b133ffe1e409f7` |
| `c_conditioning_repair.csv` | `7dd5b41eb2778e0f4fac726af9971379707156e1d9e8ed1cb86369e7df69d055` |
| `e_control_matrix.csv` | `5c28b2aafac63a1e2f90b14714b4b760f31adcd9642d16bd476c3f5f3dd56458` |
| `failure_table.csv` | `d7fda751fc8053b29383c4ef0afb9a3942ed95bfedccff43650276230ba8f267` |
| `provenance_audit.csv` | `800cb36085ee2a82c3114f377483935ddf8fec2a12b5583deceeecb03800cfcc` |
| `route_decision.json` | `ec0d8c80365f74ed9cfcb4c00c7491737f5750555c5e08395e597c47c829f345` |
| `run_manifest_v122.json` | `dff9c9bae0b02846f3bca5fec2d5ad9420c5b1adf82ef180cc8842a73d5001aa` |

追加最终判断：v12.2 按文档建议的修复已经继续执行；`signed_pair_cover` 证明“局部 interaction coverage 可以被结构性初始化修复”，但同时证明“仅修局部 pair cover 不足以成为合格 PureKAN base”。下一步应设计更全局的 low-condition quadratic cover，而不是继续推进 functional route。

## 12. C9/C10 PCA-Whiten Lift Repair 与 Clean Timing 剥离

> 本节回应上一节的追加判断：继续设计更全局的 low-condition lift repair，并把 clean timing 与 diagnostic hook timing 分开看。结果仍只来自真实 artifact，不改变 functional gate，不打开 P5。

### 12.1 修改审计

本轮代码修改：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_lq.py` | 新增 `init_variant="data_pca_whiten_lift"` | 统一使用 train-stream `x_for_stats` 做 PCA/whitening 初始化，不按 dataset name/seed 分支；目标是降低 `h=x@A` 的 lift covariance condition。 |
| `experiments/run_v122_base_truth_conditioning_functional_complementarity.py` | 新增 `M10-R2-data-pca-whiten-lift` expression diagnostic | 只用于 B-track expression truth，不改 gate 阈值。 |
| `experiments/run_v122_base_truth_conditioning_functional_complementarity.py` | 新增 `C9-data-pca-whiten-lift` | hidden=256，同当前 R2 主 candidate 尺度一致，用于检查 full-size PCA-whiten 是否能成为 base repair。 |
| `experiments/run_v122_base_truth_conditioning_functional_complementarity.py` | 新增 `C10-data-pca-whiten-h64` | rank-limited diagnostic candidate，用于检查“少而稳”的 whiten lift 能否降低 condition；因参数量不等价，不能直接包装成 official same-param success。 |

本轮没有修改：

```text
1. 没有调低 P4 / functional control gate。
2. 没有按 dataset name 选择参数。
3. 没有把 clean timing pass 当成 base pass。
4. 没有打开 P5。
5. 没有把 C10 的低参数 diagnostic 当成同参 MLP 胜利。
```

编译验证：

```bash
python -m py_compile experiments/run_v122_base_truth_conditioning_functional_complementarity.py experiments/run_v122_interaction_equivalence_repair.py dgkan/models/fc_purekan_lq.py
```

新增 artifact：

```text
results/v12_2_base_truth_conditioning_functional_complementarity/v122_pca_whiten_cgate_20260519T150000Z
```

### 12.2 新增命令

```bash
python experiments/run_v122_base_truth_conditioning_functional_complementarity.py \
  --out-dir results/v12_2_base_truth_conditioning_functional_complementarity/v122_pca_whiten_cgate_20260519T150000Z \
  --fresh --device auto --data-root data --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 5 --batch-size 128 --eval-batch-size 512 --eval-train-size 512 \
  --probe-batch-size 64 --microbatch-count 2 --profile-steps 4 \
  --timing-datasets MNIST --timing-seeds 0 \
  --timing-warmup-steps 50 --timing-measure-steps 200 \
  --expression-seeds 0 --expression-protocols B0,B1,B2,B3,B4 \
  --expression-b2-targets E1-pairwise-product \
  --expression-train-size 512 --expression-val-size 256 --expression-test-size 256 \
  --expression-b2-train-size 4096 --expression-b2-val-size 2048 --expression-b2-test-size 2048 \
  --expression-steps-b0 60 --expression-steps-b1 1000 --expression-steps-b2 5000 \
  --expression-batch-size 128 --expression-eval-stride 100 --c-expression-steps 300 \
  --report-path docs/DG-KAN_v12.2_pca_whiten_tmp.md
```

### 12.3 Route

```json
{
  "route": "R2-ExpressionImplementationFailure",
  "timing_clean_q90_ratio": 1.0305889611622527,
  "timing_official_q90_ratio": 48.729744994122896,
  "pairwise_LS_R2_mean": 0.378700315952301,
  "pairwise_AdamW_best_R2_mean": 0.9992794990539551,
  "composition_delta_vs_MLP_mean": -0.04475069046020508,
  "conditioning_repair_survivor_count": 0,
  "base_qualified": false,
  "functional_complementarity_pass": false,
  "official_functional_open": false,
  "no_fake": true
}
```

### 12.4 Clean Timing 剥离结果

| method | step q90 ms | ratio vs MLP | geometry hook ms | probe hook ms |
|---|---:|---:|---:|---:|
| `A0-MLP-same-param-AdamW` | `0.5503514781594274` | `1.0` | `0.0` | `0.0` |
| `A4-R2-LQ-clean-fastpath-no-geometry-hooks` | `0.5671861581504345` | `1.0305889611622527` | `0.0` | `0.0` |
| `A7-R2-LQ-v12-runner-profile-path` | `26.818487187847495` | `48.729744994122896` | `23.224593438208103` | `0.5764240259304643` |

解释：clean timing runner 剥离后 R2 step ratio 在本轮为 `1.03x`，说明 architecture step timing 不是当前首要失败点；official diagnostic path 仍因 geometry hook 极重而失败。因此 timing fix 的方向是把 diagnostic hook accounting 从 official architecture step timing 中剥离，但这不等于 base qualified，因为 expression / task / condition gate 仍失败。

### 12.5 M10 Expression Diagnostic

| target | protocol | val R2 | test R2 | delta vs MLP |
|---|---|---:|---:|---:|
| `E1-pairwise-product` | `B1-1000-step` | `0.9531545639038086` | `0.9531338214874268` | `-0.03766745328903198` |
| `E1-pairwise-product` | `B2-5000-step` | `0.9995875954627991` | `0.9996036291122437` | `0.00017130374908447266` |
| `E1-pairwise-product` | `B3-frozen-lift-LS` | `0.4892768859863281` | `0.48961275815963745` | `0.3293761610984802` |
| `E2-composition` | `B1-1000-step` | `0.9553978443145752` | `0.9512563347816467` | `-0.022120118141174316` |
| `E2-composition` | `B3-frozen-lift-LS` | `0.8818498253822327` | `0.8826888799667358` | `0.12695515155792236` |
| `E6-rotated-pairwise-product` | `B1-1000-step` | `0.9204347729682922` | `0.9153234958648682` | `-0.07180947065353394` |
| `E8-random-quadratic-form` | `B1-1000-step` | `0.9362291097640991` | `0.9519991278648376` | `-0.05560934543609619` |

解释：PCA-whiten lift 对 B3 frozen-LS 有改善，但 E1 LS 仍只有 `0.4892768859863281`，远低于 `signed_pair_cover` 的 E1 LS `0.9921392798423767`；它也没有让 B1 composition delta 过线。

### 12.6 C9/C10 Vision Base Gate

| method | mean val acc | val acc delta vs MLP | clean step ratio | lift condition mean | pairwise R2 | survivor rows |
|---|---:|---:|---:|---:|---:|---:|
| `CBASE-MLP-same-param` | `0.8428819444444444` | `0.0` | `1.0` | `0.0` |  | `0` |
| `C0-R2-LQ-current` | `0.8368055555555556` | `-0.006076388888888889` | `1.1471906169052457` | `247256373876053.34` | `0.9556055665016174` | `0` |
| `C8-signed-pair-cover-lift` | `0.8235677083333334` | `-0.019314236111111112` | `1.1007391894741276` | `313922594333582.25` | `0.9618070721626282` | `0` |
| `C9-data-pca-whiten-lift` | `0.8170572916666666` | `-0.025824652777777776` | `1.0834722397234993` | `232051913756216.88` | `0.8967891931533813` | `0` |
| `C10-data-pca-whiten-h64` | `0.8246527777777778` | `-0.018229166666666668` | `1.035256268610569` | `45553827380833.78` | `0.8939852714538574` | `0` |

C9/C10 paired rows：

| method | dataset | seed | delta | val acc | MLP val acc | lift condition | step ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `C9` | `MNIST` | `0` | `-0.021484375` | `0.875` | `0.896484375` | `152996079992832.0` | `1.3354356524075082` |
| `C10` | `MNIST` | `0` | `-0.017578125` | `0.87890625` | `0.896484375` | `428583104.0` | `1.1086724966536794` |
| `C9` | `MNIST` | `1` | `-0.02734375` | `0.8828125` | `0.91015625` | `143880649441280.0` | `1.0951969902789123` |
| `C10` | `MNIST` | `1` | `-0.025390625` | `0.884765625` | `0.91015625` | `420843872.0` | `1.0786625809295822` |
| `C9` | `Fashion-MNIST` | `0` | `-0.033203125` | `0.783203125` | `0.81640625` | `450938984726528.0` | `1.0935517320029058` |
| `C10` | `Fashion-MNIST` | `0` | `-0.013671875` | `0.802734375` | `0.81640625` | `239493376.0` | `1.0776446753824245` |
| `C9` | `KMNIST` | `0` | `-0.044921875` | `0.787109375` | `0.83203125` | `138067310542848.0` | `0.9036259924173867` |
| `C10` | `KMNIST` | `0` | `-0.041015625` | `0.791015625` | `0.83203125` | `52383337938944.0` | `0.8983933102024826` |

解释：C10 在部分 MNIST/Fashion split 上把 condition 从 `1e14` 降到 `1e8` 量级，说明 rank-limited PCA-whiten 确实能修一部分 conditioning；但它的 mean task delta 仍是 `-0.018229166666666668`，KMNIST condition 仍高，pairwise R2 低于 C0/C8，且参数量不等价。因此 C10 是有价值的 diagnostic，不是 base survivor。

No-fake：

```text
rows_checked = 4834
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `a_timing_truth.csv` | `5611bbb5452dafaee085c3fc6077318383f15c2057050a9f8ef728e8bd33ce4b` |
| `b_expression_truth.csv` | `6375e26910f31df4c5c3f87fbdf49bfa7c7bddaacc0c715fda36e6ca95348a35` |
| `c_conditioning_repair.csv` | `cae0f38016dc62806c4b1c0df0834487b2e68f38e58051a3cf910d6e77ad22fe` |
| `e_control_matrix.csv` | `5c28b2aafac63a1e2f90b14714b4b760f31adcd9642d16bd476c3f5f3dd56458` |
| `failure_table.csv` | `d7fda751fc8053b29383c4ef0afb9a3942ed95bfedccff43650276230ba8f267` |
| `provenance_audit.csv` | `c42323a159b9c43d16ce03d3d2fbc95e0bb2c00f84ff9d022c95c113d78fd14c` |
| `route_decision.json` | `1cb277aeafa6689583bcaaef6d1140280a36de6d14e8b54564ac4626f768491e` |
| `run_manifest_v122.json` | `3e5784b58db0bafe99c39268210a27a349dc33168eb823f66ac22973b4d420cc` |

追加最终判断：本轮确实做了“clean timing runner fix 剥离”和“lift conditioning repair”两件下一步工作。clean timing 已经不再是主 blocker；conditioning repair 有局部进展但没有形成合格 base。当前更精确的下一步是设计同时满足三件事的 lift：全局 quadratic coverage、低 condition、同参 vision task stability。
