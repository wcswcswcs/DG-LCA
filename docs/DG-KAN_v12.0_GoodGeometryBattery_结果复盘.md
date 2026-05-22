# DG-KAN v12.0 Good Geometry Battery 结果复盘

> 本复盘记录 `DG-KAN_v12.1_Codex下一步执行计划_GoodGeometryBattery.md` 的修复后真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。P4 diagnostic audit 已执行，但 `official_gate_open = 0`，所以不能写成 functional success，也不能打开 P5 short-run。

## 0. 最新结论

```text
route = R2-BaseNotQualified
base_candidate = R2-LQ-fanin-output-scale-confirmed
primary_blocker = repaired_lq_base_not_qualified
next_recommended_action = global repaired LQ base repair
generated_route_status = functional_not_opened
```

最终 artifact：

```text
results/v12_0_good_geometry_battery/v121_good_geometry_battery_repaired_diagnostic_20260519T084335Z
```

目标达成状态：

```text
v12.1 pipeline target = mostly achieved after repair
scientific base qualification = not achieved
official functional audit = not opened
diagnostic P4 controls/five-step = completed with official_gate_open=0
P5 short-run = not_run
```

核心结论：

1. P0 implementation contract pass = `true`。
2. P1 repaired LQ base pass = `false`；runner gate 的 near pass rate = `0.7777777777777778`，未达到 `0.80`。
3. P1 R2 repaired LQ mean val acc = `0.8435329861111112`；same-param MLP mean val acc = `0.8415798611111112`；macro delta = `0.001953125`，均值不差，但稳定性未过。
4. P1 efficiency 失败：failure table 的 R2 step q90 ratio vs same-param MLP = `6.872180773276933`，阈值 `1.10`；compact memory ratio q90 = `1.100739543844385`，阈值 `1.00`。
5. P1 expression battery 失败：pairwise product R2 = `0.4191594322522481`，阈值 `0.85`；composition R2 delta vs MLP = `-0.05445587635040283`，阈值 `>= -0.05`。
6. P2 passive Good Geometry Battery 完整落盘：snapshot rows = `540`，checkpoint trace rows = `180`，correlation rows = `84`。
7. P3 GeometryCertificateV0 ready = `true` / pass = `false`；pareto pass = `true`，hard gate pass = `false`。失败 hard gates 是 `val_loss_auc_time`、`step_time_ratio`、`CEp99`、`margin_p10`。
8. P4 修复后不再整体 `not_run`：one-step audit rows = `126`，lambda rows = `277`，control matrix rows = `4`，five-step diagnostic rows = `60`；所有 P4 measured rows 都是 `status = measured_diagnostic_gate_closed` / `official_gate_open = 0`。
9. P4 diagnostic 中 functional candidates one-step 多数 task-safe，但全部被 strong controls 解释：`D8/D9/D10/D11` 的 best geometry score 都低于 control best `0.3162052612271161`，所以 official P4 pass = `false`。
10. P5 = `not_run`，reason = `P4_no_control_resistant_functional_candidate`。
11. No-fake audit rows checked = `1823`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```bash
python -m py_compile experiments/run_v120_good_geometry_battery.py experiments/analyze_v120_good_geometry_battery.py
```

```bash
python experiments/run_v120_good_geometry_battery.py \
  --out-dir results/v12_0_good_geometry_battery/smoke_v121_repair \
  --device auto --data-root data --datasets MNIST --seeds 0 \
  --train-size 64 --val-size 32 --test-size 32 --epochs 1 \
  --batch-size 32 --eval-batch-size 64 --probe-batch-size 32 \
  --kernel-sketch-directions 1 --expression-train-size 64 \
  --expression-val-size 32 --expression-steps 5 \
  --expression-batch-size 32 --profile-steps 1 \
  --fresh --no-download
```

```bash
python experiments/run_v120_good_geometry_battery.py \
  --out-dir results/v12_0_good_geometry_battery/v121_good_geometry_battery_repaired_diagnostic_20260519T084335Z \
  --device auto --data-root data --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 5 --batch-size 128 --eval-batch-size 512 \
  --probe-batch-size 128 --kernel-sketch-directions 2 \
  --expression-train-size 512 --expression-val-size 256 \
  --expression-steps 60 --expression-batch-size 128 \
  --profile-steps 4 --fresh --no-download
```

```bash
python experiments/analyze_v120_good_geometry_battery.py \
  --result-dir results/v12_0_good_geometry_battery/v121_good_geometry_battery_repaired_diagnostic_20260519T084335Z
```

说明：本轮为 v12.1 首轮真实闭环 run，使用真实 MNIST/Fashion-MNIST/KMNIST 小样本与 5 epoch budget；不能冒充计划中的 20/30 epoch final confirmation。

## 2. Route

```json
{
  "route": "R2-BaseNotQualified",
  "base_candidate": "R2-LQ-fanin-output-scale-confirmed",
  "p0_contract_pass": true,
  "p1_base_pass": false,
  "p2_geometry_battery_complete": true,
  "p3_certificate_ready": true,
  "p3_certificate_pass": false,
  "p4_functional_audit_pass": false,
  "p5_short_run_opened": false,
  "strict_functional_success": false,
  "external_ready": false,
  "primary_blocker": "repaired_lq_base_not_qualified",
  "next_recommended_action": "global repaired LQ base repair",
  "no_fake": true
}
```

## 3. P1 Base Qualification

| method | mean val acc | mean val loss | mean ECE | mean CEp99 |
|---|---:|---:|---:|---:|
| `B0-MLP-same-param-AdamW` | `0.8415798611111112` | `0.5172725286748674` | `0.044311523230539426` | `5.466893990834554` |
| `B1-MLP-same-step-or-same-FLOPs-AdamW` | `0.8348524305555556` | `0.5353745917479197` | `0.05071184577213393` | `6.373299280802409` |
| `B2-QuadraticFeatureMLP-diagnostic-AdamW` | `0.7619357638888888` | `0.7699054943190681` | `0.09892289630240864` | `4.562030262417263` |
| `B3-LQ-t2-h256-AdamW` | `0.8402777777777778` | `0.5280404024653964` | `0.04061164789729648` | `5.794698397318522` |
| `B4-R2-LQ-fanin-output-scale-confirmed-AdamW` | `0.8435329861111112` | `0.5126347442468008` | `0.04066781885921955` | `5.327720112270779` |

R2 repaired LQ vs same-param MLP paired val acc delta：

| dataset | seed | delta | R2 val acc | MLP val acc |
|---|---:|---:|---:|---:|
| `MNIST` | `0` | `0.009765625` | `0.91015625` | `0.900390625` |
| `MNIST` | `1` | `-0.001953125` | `0.908203125` | `0.91015625` |
| `MNIST` | `2` | `-0.013671875` | `0.904296875` | `0.91796875` |
| `Fashion-MNIST` | `0` | `0.0078125` | `0.8046875` | `0.796875` |
| `Fashion-MNIST` | `1` | `0.01953125` | `0.810546875` | `0.791015625` |
| `Fashion-MNIST` | `2` | `0.015625` | `0.8125` | `0.796875` |
| `KMNIST` | `0` | `0.00390625` | `0.818359375` | `0.814453125` |
| `KMNIST` | `1` | `-0.009765625` | `0.810546875` | `0.8203125` |
| `KMNIST` | `2` | `-0.013671875` | `0.8125` | `0.826171875` |

P1 gate failure table 记录的失败项：

```text
near_pass_rate = 0.7777777777777778 < 0.80
step_q90 = 6.872180773276933 > 1.10
compact_memory_ratio_q90 = 1.100739543844385 > 1.00
pairwise_product_R2 = 0.4191594322522481 < 0.85
composition_R2_delta_vs_MLP = -0.05445587635040283 < -0.05
```

Expression battery 中 R2 repaired LQ 的 target mean val R2：

| target | R2 mean val R2 | MLP mean val R2 | delta |
|---|---:|---:|---:|
| `E0-additive` | `0.9409700433413187` | `0.9866019686063131` | `-0.04563192526499438` |
| `E1-pairwise-product` | `0.4191594322522481` | `0.7380732695261637` | `-0.3189138372739156` |
| `E2-composition` | `0.8255044023195902` | `0.879960278669993` | `-0.05445587635040283` |
| `E3-local-XOR` | `0.28973392645517987` | `0.327415406703949` | `-0.037681480248769106` |
| `E4-high-frequency` | `-0.2573012908299764` | `-0.09876302878061931` | `-0.15853826204935706` |
| `E5-noise-stress` | `0.8196922341982523` | `0.8564085165659586` | `-0.0367162823677063` |

## 4. P2 Good Geometry Battery

P2 已完整执行，但只作为 passive measurement，不作为 functional success。

Final checkpoint val split，R2 repaired LQ vs same-param MLP：

| metric | MLP mean | R2 repaired LQ mean |
|---|---:|---:|
| `effective_rank_hidden` | `57.59919653998481` | `71.534361521403` |
| `basis_usage_entropy` | `0.0` | `0.7119486596849229` |
| `lift_condition_proxy` | `0.0` | `119167112154680.89` |
| `curvature_debt` | `0.0` | `0.032116507490475975` |
| `perturb_logit_drift_p95` | `0.013898245265914334` | `0.016620220409499273` |
| `signal_consistency_real` | `0.4542573061254289` | `0.3809775412082672` |
| `noise_leak` | `0.8620934189645737` | `0.49669690779898684` |
| `CE_p99` | `6.327774789598253` | `5.863484435611301` |
| `ECE` | `0.09202864269415538` | `0.08345588917533557` |

P2 相关性只说明 measured association，不作因果结论。较高绝对 Pearson 的非自相关项包括：

```text
real_noise_consistency_gap -> NLL: pearson = 0.8139623055326558
signal_consistency_real -> NLL: pearson = 0.7200370228862184
effective_rank_hidden -> noise_leak: pearson = -0.7192218778880195
ECE -> margin_p10: pearson = -0.7167439101304862
perturb_logit_drift_p95 -> CEp99: pearson = 0.678361568500262
local_jacobian_norm_p95 -> CEp99: pearson = 0.6733329622835523
```

## 5. P3 GeometryCertificateV0

```text
certificate_pass = false
hard_gate_pass = false
pareto_pass = true
failure_reasons = val_loss_auc_time; step_time_ratio; CEp99; margin_p10
```

解释：R2 repaired LQ 在部分 geometry/tail/calibration 指标上有被动 Pareto 信号，但 hard gate 中收敛 AUC、效率、CEp99 与 margin tail 未达标，所以不能进入 functional official audit。这里的 P3 gate 使用 certificate aggregate 口径；P2 表格是 final-checkpoint val split 口径，二者不能混成同一个结论。

## 6. P4 Diagnostic Audit 修复结果

修复点：

```text
旧行为：P1/P3 gate failed -> P4 全部 not_run
新行为：P1/P3 gate failed -> P4 diagnostic audit 继续执行，所有 row 写 official_gate_open = 0
```

P4 落盘结果：

| artifact | measured rows |
|---|---:|
| `p4_functional_direction_audit.csv` | `126` |
| `p4_one_step_probe.csv` | `126` |
| `p4_lambda_backtracking.csv` | `277` |
| `p4_control_matrix.csv` | `4` |
| `p4_five_step_probe.csv` | `60` |

P4 candidates covered：

```text
D0 TaskOnlyAdamW
D1 NoOpMatchedOverhead
D2 RandomMatchedNorm
D3 AdamWParallelDirection
D4 ShuffledPayload
D5 ShuffledEvent
D6 SNROnlyGate
D7 GeometryOnlyNoSNR
D8 SNRProjectedGeometry
D9 BasisEntropyRebalance
D10 LiftConditionRepair
D11 TailStabilityCorrection
D12 MLPAnalogGeometryMaintenance
D13 QuadraticFeatureMLPAnalogMaintenance
```

P4 control matrix：

| candidate | accepted rows | best geometry score | control best | beats strong controls |
|---|---:|---:|---:|---:|
| `D8-SNRProjectedGeometry` | `9` | `0.3039031238761103` | `0.3162052612271161` | `0` |
| `D9-BasisEntropyRebalance` | `9` | `0.3031435286577053` | `0.3162052612271161` | `0` |
| `D10-LiftConditionRepair` | `9` | `0.3037566523442021` | `0.3162052612271161` | `0` |
| `D11-TailStabilityCorrection` | `9` | `0.3033008925700202` | `0.3162052612271161` | `0` |

解释：diagnostic one-step/five-step 已经跑通，但 functional candidates 没有击败 strong controls；并且 official gate 本来就关闭，因此 P4 不能算 pass。

## 7. P5

```text
P5 status = not_run
P5 reason = P4_no_control_resistant_functional_candidate
```

本轮没有 short-run functional promotion。

## 8. No-Fake / Hash

```text
rows_checked = 1823
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `p0_contract.csv` | `b86c7ec6370a6d2544a82b1ddabd32cbc1cee913c7f24233eeddc97bdf474736` |
| `p1_base_qualification.csv` | `aebede0a5c83024befda98817c5201184de80d0bc74f4667251a2a75cceb6668` |
| `p1_base_task_trace.csv` | `46dd9c6663df564ffe0ba0265a8b1f7a2b9ba243dd469f2aa4ab95b08ed58077` |
| `p1_efficiency_profile.csv` | `52c213137b39727bd3d6c65a14bac7f097defee4cb7525f2251986c07c9690f9` |
| `p1_expression_battery.csv` | `d65de1f287ad9e907bd028a3837f5d574184cd6a045401e931429e00e0a4e8b5` |
| `p2_passive_geometry_snapshot.csv` | `4dfffa64dde3ce25ea3b3d0a7205c04cb0d46da1792ade7635e2f8337d7686a3` |
| `p2_geometry_checkpoint_trace.csv` | `516521ba1d9b18f2fcb0e28ac85eb1a0d3e344e7a974d0233cf8a6b9f99c44fc` |
| `p2_geometry_correlation.csv` | `62772c989a9b9859333ab81950cf22ac3fe4ffe7a3e6c59921c5b912f193e4f1` |
| `p3_geometry_certificate.csv` | `e8688235349602f497c85f1583c86ac4dd90f9803e34ab1567ec8df3b6c5f4fc` |
| `p3_geometry_certificate.json` | `ad55952ea923379b83be69a4d4e0bcd793b020d87ec854165037fd41a12074bd` |
| `p4_functional_direction_audit.csv` | `af898001d1c281e3b0ae62245f516e2aa0e7bfd85e096af7bc3b9f9aac1f8d96` |
| `p4_one_step_probe.csv` | `f39a33c8b1b152e570d51462012e4aef762fb70470cc9b8c5f3cc78f923dd033` |
| `p4_five_step_probe.csv` | `af586d8dbb4c77c6a82e8f47a2d8af1286f25f3344a321a100733f9cb6c49f00` |
| `p4_control_matrix.csv` | `08ccb0fef07b8849cb3940e4e3fce610a884bc8d2fae32ab0149548a70f6d133` |
| `p5_short_run_plan_or_notrun.csv` | `b39d5c44dfef2c1b935e8693f431155b067f9721e8e00ad506fb1cd75bb2c038` |
| `failure_table.csv` | `bea47baa29747ae43ddc4ac75b6834c8a9305aac2cecc011f59bac3c21be241f` |
| `provenance_audit.csv` | `99ffa6aad25ba5d941d3430bc0df9cd00d821940229e6c78822b1d8265bf1d29` |
| `route_decision.json` | `a8ced6dc07d169c67653ccbd4dffd46468c33da09adf1dabb22c31a8e91f1bee` |
| `run_manifest_v120.json` | `9d18cdbaa352c949a014ba3d9739ad0e5146dc7ee66d1542eb045b7ddc9d299b` |

## 9. 最终分析结论

```text
1. 原首轮结果没有完全达成 v12.1 目标，因为 P4 在 P1/P3 gate 失败时被整体 not_run。
2. 修复后，P4 diagnostic controls 与 five-step cloned rollout 已真实落盘，同时 official gate 仍正确关闭。
3. 工程 pipeline 目标基本达成：metric 清楚、gate 清楚、controls 清楚、not_run 清楚、failure 清楚。
4. 科学结论仍是 R2-BaseNotQualified：当前 repaired LQ 不满足 base gate，不允许进入 official functional audit 或 P5 short-run。
5. P4 diagnostic 说明当前 D8/D9/D10/D11 functional geometry directions 没有击败 strong controls，下一步不能调低 gate，应做 direction repair 或 base repair。
```

最终一句话：修复后 v12.1 的 pipeline 闭环已经更完整，但科学路线仍停在 `R2-BaseNotQualified`；下一步优先做 global repaired LQ base repair 与 control-resistant direction repair，而不是打开 functional short-run。
