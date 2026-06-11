# DG-KAN v17.1 三段式实验结果复盘

生成时间：2026-06-02 19:45:59 +08
复盘补充时间：2026-06-02 20:04:46 +08（基于既有 artifacts 整理；未重跑实验）

## Route

- route: `R8-CarrierSpecificPartialPositive`
- route_detail: fresh h100/h800/h1600 screen found partial positive source, but S4 real-transfer and S5 official gates were not executed/passed; v17.1 h3200 measured_group_rows=105, positive_mean_group_rows=1, retained_candidate_count=0, late_rebound_count=1
- CodeRoute: S0_1-CodeCorrectnessPassed
- EfficiencyRoute: R-EfficiencyForwardBlocked-D-CHE-D-FOU-D-RAT-D-RBF-D-WAV-LQ
- FunctionalRoute: R-SourceSingleRowOnlyOrWashedOut
- S0 pass: 1
- measured mechanism jobs: 945/945
- best h100 source vs best control: 0.11567223072052002
- best h800 source vs best control: 0.6183267831802368
- best h1600 source vs best control: 0.3805924654006958
- h800 valid summary rows: 105
- h1600 valid summary rows: 105
- measured horizon rows: 945
- h3200 measured group rows: 105
- h3200 positive mean group rows: 1
- h3200 retained candidate count: 0
- h3200 late rebound count: 1
- 注：route 中 best source 字段是单个 fresh row 的最大值；h800/h1600/h3200 表格是 carrier x mechanism 分组后的 9-row 均值排行。当前 retention ratio 仅在前一 horizon mean source 为正时定义，避免把 late rebound 写成 retention。

## 计划覆盖边界

- 已完成：S0.1 import/LineC/update/retention/debt/route/kernel correctness gate、full carrier x mechanism h100 screen、full 945-row horizon extension、mandatory exploration efficiency census、required/forbidden/no-action audits、v17.1 code review packet、4GPU shard execution记录。
- 已修复/补齐：v17.1 runner compatibility shims、dynamic geometry debt helper、`dgkan.diagnostics.linec` 稳定入口、FU update semantics shims、efficiency/profiling/kernel gradcheck shims、`loss_interface`、manual optimizer、matched controls；M2 SGD/Momentum branch 使用 `.step_gradient()`；basis repair 从 alias reference path 改为 family-specific torch repair path。
- 未完成/不可冒充：official fused CUDA/C++ basis kernels、S4 real-transfer exploration、S5 official success gate、真正动态 work-stealing queue。本次 4GPU 为 shard queue drain，并记录 idle/dashboard；不能写成 promotion-ready official result。

## 为什么上一版复盘偏短

上一版复盘文件走的是 `experiments/run_v17_common.py:write_docs` 的摘要模板，只截取了 route、top rows 和少量 insight；完整证据被放在 `results/.../official_v171` 下的 CSV/JSON/SVG 与 `v17_1_code_review_packet.zip` 中。这种写法对复现实验不友好，也不满足 v17.1 计划里“更细证据链”的要求。

本次补充不重跑实验，只把已经真实生成的 artifact 内容整理进正文。所有数字来自以下文件：

- `v17_s0_route_decision.json`
- `v17_1_route_decision.json`
- `v17_required_artifact_manifest.csv`
- `v17_1_efficiency_truth_table.csv`
- `v17_1_efficiency_blocker_table.csv`
- `v17_1_source_retention_matrix.csv`
- `v17_1_control_attribution.csv`
- `v17_1_kan_vs_mlp_attribution.csv`
- `v17_1_adamw_overwrite_diagnostics.csv`
- `v17_gpu_runtime_snapshots.csv`
- `v17_horizon_extension_h0/h1/h2/h3.csv`
- `v17_1_code_review_packet/packet_manifest.csv`

## Part A：代码审计

- code_audit_status: S0_1-CodeCorrectnessPassed
- required artifact missing count: 0
- LineC exception policy: MeasurementInvalid，不进入 geometry fail 均值。
- route aggregation: v17.1 额外输出 9-row mean source/retention matrix；single-row max 只作诊断字段。
- debt accounting: 公式单元测试通过；真实 tail/LineC/calibration/AUC debt 未完整测量的字段保持空值/measurement_status，不编造 recovery。

### A1-A10 审计明细

| item | value | evidence |
|---|---:|---|
| s0_pass | 1 | `v17_s0_route_decision.json` |
| compileall_ok | 1 | `v17_compileall.log` |
| all_v17_imports_ok | 1 | `v17_import_closure.csv` |
| import_error_count | 0 | `v17_import_errors.csv` |
| linec_golden_pass_count / total | 9 / 9 | `linec_golden_results.csv` |
| update_sign_pass_count / total | 15 / 15 | `v17_update_sign_sanity.csv` |
| source_retention_formula_pass | 1 | `source_retention_formula_tests.csv` |
| debt_accounting_formula_pass | 1 | `debt_accounting_formula_tests.csv` |
| route_aggregation_unit_pass | 1 | `route_aggregation_unit_tests.csv` |
| kernel_exploration_pass_count / total | 6 / 6 | `v17_kernel_correctness.csv` |
| candidate_semantic_alias_pairs | 0 | `v17_mechanism_noncollapse_summary.csv` |
| control_surface_rows | 7 | `v17_control_surface.csv` |
| required artifact missing count | 0 | `v17_required_artifact_manifest.csv` |
| packet manifest rows | 666 | `v17_1_code_review_packet/packet_manifest.csv` |
| packet sha256 rows | 666 | `v17_1_code_review_packet/packet_sha256_manifest.csv` |

### LineC Golden 证据

| fixture | pass | key evidence |
|---|---:|---|
| G1-NoOpNull | 1 | CouplingR2=0.0 |
| G2-RandomMatchedNormNull | 1 | null false positive rate=0.0 |
| G3-SyntheticKnownTransfer | 1 | CouplingR2=0.9908877528715863 |
| G4-SyntheticNoiseLeak | 1 | NoiseSignalLeak=0.2750449703115212 |
| G5-ReservoirOnlyPerturbation | 1 | CouplingR2=0.03895507872995552 |
| G6-BatchPermutationMismatchDrop | 1 | CouplingR2=0.002051890482749133, coupling_drop=0.9888358623888371 |
| G7-ExceptionPathMeasurementInvalid | 1 | route=R0-LineCMeasurementInvalid |
| G8-ScaleInvariance | 1 | scale_coupling_delta=3.5599477499204113e-07 |
| G9-ControlEquivalence | 1 | control_equivalent=1 |

### Route / Retention 公式修复

`source_retention_h3200_over_h1600` 现在只在 h1600 9-row mean source 为正时定义。否则即使 h3200 转正，也记录为 late rebound，不记录为 retained source。对应单元测试：

| test | expected | actual | pass |
|---|---:|---:|---:|
| retention_h800_over_h100 | 0.5 | 0.5 | 1 |
| retention_negative_source_clamped | 0.0 | 0.0 | 1 |
| retention_previous_nonpositive_is_undefined | blank | blank | 1 |

## Part B：基函数效率

- efficiency_status: R-EfficiencyForwardBlocked-D-CHE-D-FOU-D-RAT-D-RBF-D-WAV-LQ
- v17.1 输出 `v17_1_efficiency_truth_table.csv`、`v17_1_efficiency_blocker_table.csv`、`v17_1_family_repair_summary.csv`。
- official_fused_kernel_complete=0 的 family 不能写 official efficiency success。

### B1-B10 效率结论

本轮效率结论不是“KAN basis 全面不可行”，而是更具体的：

- MLP 3/3 exploration pass；非 MLP basis 0/18 exploration pass。
- 全部非 MLP family 都被 forward ratio 卡住：D-CHE、D-FOU、D-RAT、D-RBF、D-WAV、LQ。
- 11/21 rows 同时 forward+backward blocked，7/21 rows forward-only blocked，3/21 rows pass（全部是 MLP）。
- memory ratio 基本接近 1，当前主 blocker 不是显存，而是 forward/backward compute。
- repair 当前只是 torch family-specific repair，不是 official fused CUDA/C++ repair；`official_fused_kernel_complete=0`。

| blocker_class | row_count |
|---|---:|
| pass | 3 |
| ForwardBlocked | 7 |
| ForwardBlocked;BackwardBlocked | 11 |

### Efficiency Full Blocker Table

| carrier | batch | forward_ratio | backward_ratio | update_ratio | step_ratio | memory_ratio | blocker |
|---|---:|---:|---:|---:|---:|---:|---|
| D-CHE | 8 | 5.498238624213571 | 1.3490782168268838 | 1.4256663146836337 | 1.482397926191536 | 1.001449318228874 | ForwardBlocked |
| D-CHE | 32 | 4.211986400948355 | 1.518609377289657 | 1.0371971050035989 | 1.622724939753544 | 1.002246327550025 | ForwardBlocked |
| D-CHE | 128 | 4.882039777842973 | 1.582252302847153 | 0.9872087711283691 | 1.691044505844007 | 1.008511637462945 | ForwardBlocked |
| D-FOU | 8 | 5.932088872889892 | 1.4889144352205408 | 0.848076038632531 | 1.64371330220023 | 1.001449318228874 | ForwardBlocked |
| D-FOU | 32 | 4.8870176859929835 | 0.8684197970587696 | 0.9551850573369247 | 0.9875144341172103 | 1.0037537315638578 | ForwardBlocked |
| D-FOU | 128 | 4.260899407628147 | 3.2675951350675616 | 1.0329342385294922 | 3.245196502352426 | 1.0160253588095447 | ForwardBlocked;BackwardBlocked |
| D-RAT | 8 | 7.834065060854572 | 1.8076768956937441 | 0.9474507641356577 | 1.966362204273977 | 1.001449318228874 | ForwardBlocked;BackwardBlocked |
| D-RAT | 32 | 6.811426802080754 | 2.6634808705126236 | 1.1567526254069709 | 2.84455805025335 | 1.0032808205006947 | ForwardBlocked;BackwardBlocked |
| D-RAT | 128 | 6.684747063750775 | 1.824657067946317 | 1.1855274426669768 | 2.0007698878148603 | 1.0141469284728948 | ForwardBlocked;BackwardBlocked |
| D-RBF | 8 | 9.907866090763035 | 2.697841841888371 | 0.9102826342835816 | 3.011891124367022 | 1.0021000325357154 | ForwardBlocked;BackwardBlocked |
| D-RBF | 32 | 9.463999044061492 | 1.7409201864891366 | 1.0662021832998845 | 2.026030622954478 | 1.0022167706085774 | ForwardBlocked |
| D-RBF | 128 | 10.054311196663459 | 2.0553615600597874 | 1.077595527684329 | 2.3506572858822232 | 1.0087464412550262 | ForwardBlocked;BackwardBlocked |
| D-WAV | 8 | 18.784673985373512 | 2.1264015604612663 | 0.9796982788312979 | 2.6186042626963575 | 1.0021000325357154 | ForwardBlocked;BackwardBlocked |
| D-WAV | 32 | 16.4797721943787 | 2.448244434809765 | 1.1145899288952934 | 3.0028684573548303 | 1.0036355037980669 | ForwardBlocked;BackwardBlocked |
| D-WAV | 128 | 13.75407706812164 | 2.2876605721218994 | 1.0596307427498788 | 2.7363500592756385 | 1.0150861436412197 | ForwardBlocked;BackwardBlocked |
| LQ | 8 | 7.149200970940427 | 1.8122184539523307 | 1.18490436623761 | 1.9845848501009635 | 1.001449318228874 | ForwardBlocked;BackwardBlocked |
| LQ | 32 | 5.4218012787902365 | 1.7148125898277302 | 0.8707756705114068 | 1.8469311757428544 | 1.002246327550025 | ForwardBlocked |
| LQ | 128 | 5.530923259807159 | 2.7024099348028567 | 0.9606721647483114 | 2.831717155850073 | 1.008511637462945 | ForwardBlocked;BackwardBlocked |
| MLP | 8 | 1.317389060887513 | 0.8303365803166505 | 1.3009021246127077 | 0.8550685738365374 | 1.0 | pass |
| MLP | 32 | 1.2064267589044029 | 0.7724544758991213 | 1.38031178992306 | 0.801947750955202 | 1.0 | pass |
| MLP | 128 | 1.2092246141256167 | 0.7587830150387813 | 1.475987108494386 | 0.788851369108673 | 1.0 | pass |

### Family Repair Summary

| family | gradcheck_pass | dense_materialized | official_fused_kernel_complete | blocked_rows | repair_summary |
|---|---:|---:|---:|---:|---|
| D-CHE | 1 | 1 | 0 | 3 | torch_family_specific_repair_not_official_fused_cuda |
| D-FOU | 1 | 1 | 0 | 3 | torch_family_specific_repair_not_official_fused_cuda |
| LQ | 1 | 1 | 0 | 3 | torch_family_specific_repair_not_official_fused_cuda |
| D-RAT | 1 | 1 | 0 | 3 | torch_family_specific_repair_not_official_fused_cuda |
| D-RBF | 1 | 0 | 0 | 3 | torch_family_specific_repair_not_official_fused_cuda |
| D-WAV | 1 | 0 | 0 | 3 | torch_family_specific_repair_not_official_fused_cuda |

## Part C：Functional Update

- functional_status: R-SourceSingleRowOnlyOrWashedOut
- v17.1 输出 `v17_1_functional_source_matrix.csv`、`v17_1_source_retention_matrix.csv`、`v17_1_control_attribution.csv`、`v17_1_kan_vs_mlp_attribution.csv`。
- S4/S5 未执行/未通过前，promotion_allowed 保持 0。

### C1-C8 Functional 结论

本轮 functional update 有局部 source，但没有达到 v17.1 计划要求的“长期留存 + controls + KAN-specific + debt recovery”闭合。

- h100 grouped mean top：D-RBF/M1 为 0.0027029779222276476，未达强 source。
- h800 grouped mean top：MLP/M2 为 0.1751598914464315，但这是 MLP，不是 KAN-specific。
- h1600 grouped mean top：D-CHE/M1 为 0.0032123857074313695，低于 0.005 grouped positive threshold。
- h3200 grouped mean top：D-CHE/M2 为 0.09768134355545044，但 h1600 mean=-0.4415374795595805，因此是 late rebound，不是 retention。
- S2 candidate count=0，S3 candidate count=0。
- h3200 positive mean group rows=1，retained candidate count=0，late rebound count=1。
- control attribution 中 103/105 rows 被 control 解释；剩余 2 个不构成 promotion。
- KAN_specific_advantage 计数为 20/105，但这只是相对同机制 MLP h1600 表现差产生的 delta，不等于 KAN-specific functional success。

### Source / Retention Evidence Chain

| horizon | top carrier | mechanism | 9-row mean source | pass_count | interpretation |
|---|---|---|---:|---:|---|
| h100 | D-RBF | M1-AdamWPrimaryFUResidual | 0.0027029779222276476 | 6 | weak grouped source，不足以 promotion |
| h800 | MLP | M2-SGDMomentumPrimaryFU | 0.1751598914464315 | 7 | positive，但来自 MLP，先按 generic dynamics insight |
| h1600 | D-CHE | M1-AdamWPrimaryFUResidual | 0.0032123857074313695 | 7 | 接近但低于 0.005 grouped threshold |
| h3200 | D-CHE | M2-SGDMomentumPrimaryFU | 0.09768134355545044 | 5 | h1600 为负，归类 late rebound，不是 retention |

### h3200 Positive Row 细节

| carrier | mechanism | h100_mean | h800_mean | h1600_mean | h3200_mean | h3200_pass_count | retention_h3200_over_h1600 |
|---|---|---:|---:|---:|---:|---:|---:|
| D-CHE | M2-SGDMomentumPrimaryFU | -0.5592780113220215 | -0.8153564466370476 | -0.4415374795595805 | 0.09768134355545044 | 5 | blank |

解释：这个 row 值得进入下一轮 hypothesis queue，但不能写成 retained source，因为 h1600 grouped mean source 不是正数。v17.1 的 retention 修复正是为了防止把这种 late rebound 夸成“长期留存”。

### Control Attribution

| metric | value |
|---|---:|
| total grouped rows | 105 |
| control_explains_positive rows | 103 |
| not explained by control rows | 2 |

| carrier | mechanism | h800_mean | best_control_h800_mean | control_explains_positive |
|---|---|---:|---:|---:|
| MLP | M2-SGDMomentumPrimaryFU | 0.1751598914464315 | -0.10028591420915392 | 0 |
| D-CHE | M1-AdamWPrimaryFUResidual | -0.002686844931708442 | -0.012133015526665581 | 0 |

解释：MLP/M2 是 generic training dynamics 线索；D-CHE/M1 是相对 control 更好但绝对 source 仍为负，不能作为 scientific success。

### KAN vs MLP Attribution

| metric | value |
|---|---:|
| grouped rows | 105 |
| KAN_specific_advantage rows | 20 |

最高 delta rows 主要来自 MLP same-mechanism h1600 很差，而不是 KAN 机制本身稳定推进：

| carrier | mechanism | h1600_mean | mlp_same_mechanism_h1600 | delta | advantage |
|---|---|---:|---:|---:|---:|
| D-CHE | M1-AdamWPrimaryFUResidual | 0.0032123857074313695 | -1.163190523783366 | 1.1664029094907973 | 1 |
| D-RBF | M1-AdamWPrimaryFUResidual | -0.00665075249142117 | -1.163190523783366 | 1.1565397712919447 | 1 |
| LQ | M1-AdamWPrimaryFUResidual | -0.012305027908749051 | -1.163190523783366 | 1.1508854958746169 | 1 |
| D-WAV | M1-AdamWPrimaryFUResidual | -0.017111447122361925 | -1.163190523783366 | 1.146079076661004 | 1 |
| D-FOU | M1-AdamWPrimaryFUResidual | -0.03403410646650526 | -1.163190523783366 | 1.1291564173168607 | 1 |

解释：这些 rows 是 attribution 线索，不是 KAN-specific proof。真正 proof 需要 KAN row 本身 source 为正、跨 horizon retention 为正、controls 无法解释，并通过 S4/S5。

### AdamW Overwrite Diagnostics

| metric | n | min | max | mean |
|---|---:|---:|---:|---:|
| cos_FU_AdamW | 2850 | -0.07528498023748398 | 0.8370694518089294 | 0.3939429601662605 |
| optimizer_overwrite_projection | 2850 | -248.11611938476562 | 24562.177734375 | 1994.4000750265936 |
| source_retention_after_optimizer | 2850 | 0.0 | 24562.177734375 | 1996.214073293136 |

这些诊断说明 AdamW/optimizer writeback 与 FU direction 存在强耦合风险，因此 v17.1 不能把 AdamW-primary row 直接解释成纯 FU 机制成功。

### Debt Recovery 状态

`v17_1_debt_recovery_matrix.csv` 只真实记录了 CEp99 readback。tail debt、LineC debt、calibration debt、AUCtime ratio 字段保持 blank，并标注 `CEp99_readback_only_debt_metrics_not_fully_measured`。因此本轮不能写成 debt recovery success，也不能用空字段推断 no-go。

## 关键实验数据

| carrier | mechanism | dataset | seed | source_h100 | final_val_loss | LineC valid | CouplingR2 |
|---|---|---|---:|---:|---:|---:|---:|
| MLP | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 0 | 0.11567223072052002 | 1.1136894226074219 | 1 | 0.09645487360629446 |
| MLP | M1-AdamWPrimaryFUResidual | MNIST | 0 | 0.06858468055725098 | 1.3802064657211304 | 1 | 0.08413871879275892 |
| D-RAT | M1-AdamWPrimaryFUResidual | KMNIST | 2 | 0.04729962348937988 | 1.975417137145996 | 1 | 0.09557361833714939 |
| D-CHE | M1-AdamWPrimaryFUResidual | MNIST | 2 | 0.041661977767944336 | 1.7515848875045776 | 1 | 0.0 |
| D-FOU | M1-AdamWPrimaryFUResidual | MNIST | 2 | 0.03851354122161865 | 1.7809066772460938 | 1 | 0.0 |
| D-RBF | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 2 | 0.03650557994842529 | 1.9294363260269165 | 1 | 0.006222984902326312 |
| LQ | M1-AdamWPrimaryFUResidual | MNIST | 2 | 0.03245246410369873 | 1.9229087829589844 | 1 | 0.0 |
| D-RAT | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 1 | 0.028737664222717285 | 1.6809548139572144 | 1 | 0.0 |
| LQ | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 1 | 0.026569724082946777 | 1.5633560419082642 | 1 | 0.0 |
| D-RBF | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 1 | 0.02381765842437744 | 1.7538894414901733 | 1 | 0.0002541850130547557 |
| D-CHE | M1-AdamWPrimaryFUResidual | MNIST | 0 | 0.023011207580566406 | 1.8814640045166016 | 1 | 0.12088389164179958 |
| D-RBF | M1-AdamWPrimaryFUResidual | KMNIST | 2 | 0.01926732063293457 | 2.0813286304473877 | 1 | 0.005831970768654993 |

## h800 / h1600 / h3200 Fresh Extension

| horizon | carrier | mechanism | valid_rows | source_vs_best_control | retention_ratio | route |
|---|---|---|---:|---:|---:|---|
| h800 | MLP | M2-SGDMomentumPrimaryFU | 9 | 0.1751598914464315 |  | measured |
| h800 | D-CHE | M1-AdamWPrimaryFUResidual | 9 | -0.002686844931708442 | 0.0 | measured |
| h800 | D-RBF | CTRL-RecoveryOnly | 9 | -0.007917450533972846 |  | measured |
| h800 | D-WAV | CTRL-AdamW | 9 | -0.009836514790852865 |  | measured |
| h800 | D-CHE | CTRL-AdamW | 9 | -0.012133015526665581 |  | measured |
| h800 | LQ | CTRL-RecoveryOnly | 9 | -0.013121240668826632 |  | measured |
| h800 | D-WAV | CTRL-RecoveryOnly | 9 | -0.0134863191180759 |  | measured |
| h800 | D-RBF | CTRL-AdamW | 9 | -0.013663775391048856 |  | measured |
| h800 | D-FOU | CTRL-RecoveryOnly | 9 | -0.014129738012949625 |  | measured |
| h800 | D-FOU | M1-AdamWPrimaryFUResidual | 9 | -0.01462852292590671 |  | measured |
| h1600 | D-CHE | M1-AdamWPrimaryFUResidual | 9 | 0.0032123857074313695 |  | measured |
| h1600 | D-RBF | M1-AdamWPrimaryFUResidual | 9 | -0.00665075249142117 |  | measured |
| h1600 | D-WAV | CTRL-RecoveryOnly | 9 | -0.010287218623691134 |  | measured |
| h1600 | LQ | CTRL-RecoveryOnly | 9 | -0.011402944723765055 |  | measured |
| h1600 | LQ | M1-AdamWPrimaryFUResidual | 9 | -0.012305027908749051 |  | measured |
| h1600 | D-WAV | CTRL-AdamW | 9 | -0.013830310768551297 |  | measured |
| h1600 | MLP | CTRL-SGD | 9 | -0.0138778289159139 |  | measured |
| h1600 | D-FOU | CTRL-RecoveryOnly | 9 | -0.01632106304168701 |  | measured |
| h1600 | D-RBF | CTRL-AdamW | 9 | -0.016780740684933133 |  | measured |
| h1600 | D-WAV | M1-AdamWPrimaryFUResidual | 9 | -0.017111447122361925 |  | measured |
| h3200 | D-CHE | M2-SGDMomentumPrimaryFU | 9 | 0.09768134355545044 |  | measured |
| h3200 | MLP | CTRL-SGD | 9 | 0.0 |  | measured |
| h3200 | MLP | M5-AlternatingFUGradient | 9 | -0.009376459651523165 |  | measured |
| h3200 | D-RBF | M1-AdamWPrimaryFUResidual | 9 | -0.023841241995493572 |  | measured |
| h3200 | D-RBF | CTRL-AdamW | 9 | -0.02936577796936035 |  | measured |
| h3200 | D-WAV | M1-AdamWPrimaryFUResidual | 9 | -0.03233219517601861 |  | measured |
| h3200 | D-WAV | CTRL-RecoveryOnly | 9 | -0.032775786187913686 |  | measured |
| h3200 | D-RBF | CTRL-RecoveryOnly | 9 | -0.03377440240648058 |  | measured |
| h3200 | D-FOU | CTRL-RecoveryOnly | 9 | -0.0341747866736518 |  | measured |
| h3200 | D-RAT | CTRL-RecoveryOnly | 9 | -0.03553057379192776 |  | measured |

## Efficiency 证据

| carrier | batch | forward_ratio | backward_ratio | update_ratio | step_ratio | memory_ratio | exploration |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 128 | 1.2092246141256167 | 0.7587830150387813 | 1.475987108494386 | 0.788851369108673 | 1.0 | 1 |
| MLP | 32 | 1.2064267589044029 | 0.7724544758991213 | 1.38031178992306 | 0.801947750955202 | 1.0 | 1 |
| MLP | 8 | 1.317389060887513 | 0.8303365803166505 | 1.3009021246127077 | 0.8550685738365374 | 1.0 | 1 |
| D-FOU | 32 | 4.8870176859929835 | 0.8684197970587696 | 0.9551850573369247 | 0.9875144341172103 | 1.0037537315638578 | 0 |
| D-CHE | 8 | 5.498238624213571 | 1.3490782168268838 | 1.4256663146836337 | 1.482397926191536 | 1.001449318228874 | 0 |
| D-CHE | 32 | 4.211986400948355 | 1.518609377289657 | 1.0371971050035989 | 1.622724939753544 | 1.002246327550025 | 0 |
| D-FOU | 8 | 5.932088872889892 | 1.4889144352205408 | 0.848076038632531 | 1.64371330220023 | 1.001449318228874 | 0 |
| D-CHE | 128 | 4.882039777842973 | 1.582252302847153 | 0.9872087711283691 | 1.691044505844007 | 1.008511637462945 | 0 |
| LQ | 32 | 5.4218012787902365 | 1.7148125898277302 | 0.8707756705114068 | 1.8469311757428544 | 1.002246327550025 | 0 |
| D-RAT | 8 | 7.834065060854572 | 1.8076768956937441 | 0.9474507641356577 | 1.966362204273977 | 1.001449318228874 | 0 |
| LQ | 8 | 7.149200970940427 | 1.8122184539523307 | 1.18490436623761 | 1.9845848501009635 | 1.001449318228874 | 0 |
| D-RAT | 128 | 6.684747063750775 | 1.824657067946317 | 1.1855274426669768 | 2.0007698878148603 | 1.0141469284728948 | 0 |

## Kernel 证据

| family | forward_relerr | grad_relerr | grad_cosine | dense_materialized | exploration_pass |
|---|---:|---:|---:|---:|---:|
| D-CHE | 0.0 | 0.0 | 0.9999999999999999 | 1 | 1 |
| D-FOU | 0.0 | 1.0432822587251365e-16 | 0.9999999999999998 | 1 | 1 |
| LQ | 0.0 | 0.0 | 1.0 | 1 | 1 |
| D-RAT | 0.0 | 0.0 | 1.0 | 1 | 1 |
| D-RBF | 0.0 | 0.0 | 0.9999999999999999 | 0 | 1 |
| D-WAV | 0.0 | 8.432204087278424e-17 | 1.0 | 0 | 1 |

## 4GPU / Runtime 证据

本轮不是只生成表格，实际跑了 4GPU shard。`v17_gpu_runtime_snapshots.csv` 记录 536 行 nvidia-smi 快照，最大显存占用 693 MB，最大 GPU util 21%。由于模型/批量较小，GPU utilization 不高，但四张卡在 mechanism 和 horizon 阶段均有非零显存/利用率记录。

| artifact | rows | status |
|---|---:|---|
| v17_functional_mechanism_matrix.csv | 945 | measured 945/945 |
| v17_horizon_extension.csv | 945 | measured 945/945 |
| raw_h3200_matrix.csv | 945 | measured 945/945 |
| v17_efficiency_truth_table.csv | 21 | measured 21/21 |
| v17_kernel_correctness.csv | 6 | family rows 6/6 |
| v17_required_artifact_manifest.csv | 182 | missing 0 |
| v17_1_code_review_packet/packet_manifest.csv | 666 | present |
| v17_1_code_review_packet/packet_sha256_manifest.csv | 666 | present |

### Horizon Shard Runtime

| shard | measured_rows | runtime_sum_sec | status |
|---|---:|---:|---|
| h0 | 237 | 3220.40 | measured |
| h1 | 236 | 3245.64 | measured |
| h2 | 236 | 3313.05 | measured |
| h3 | 236 | 3169.34 | measured |

### 关键文件索引

| purpose | file |
|---|---|
| S0 route | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_s0_route_decision.json` |
| v17.1 route | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_1_route_decision.json` |
| full mechanism matrix | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_functional_mechanism_matrix.csv` |
| h3200 raw matrix | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/raw_h3200_matrix.csv` |
| source retention matrix | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_1_source_retention_matrix.csv` |
| efficiency blockers | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_1_efficiency_blocker_table.csv` |
| control attribution | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_1_control_attribution.csv` |
| KAN vs MLP attribution | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_1_kan_vs_mlp_attribution.csv` |
| GPU snapshots | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_gpu_runtime_snapshots.csv` |
| code packet | `results/v17_1_three_stage_codeaudit_basis_efficiency_functional_update_4gpu/official_v171/v17_1_code_review_packet.zip` |

## 修改记录（便于审计）

- 新增 `dgkan/metrics/linec.py`：LineC measurement invalid 独立 route，不把异常当 geometry fail。
- 新增 `dgkan/fu/core.py` 与 `dgkan/fu/mechanisms.py`：统一 UpdateTensor kind/sign/space/source，并实现 AdamW-free、slow-state、matrix-block、PopRisk/SNR、role-partition smoke。
- 新增 `dgkan/fu/loss_interface.py`：显式记录 CE/Brier train-stream cotangent readback，禁止 audit metrics 进入 direction source。
- 新增 `dgkan/fu/optimizers.py` 与 `dgkan/fu/controls.py`：manual SGD/Momentum optimizer 与 matched NoOp/Random controls；修复 M2 branch 的 `.step_gradient()` 调用。
- 新增 `dgkan/profiling/efficiency_v17.py`：每个 phase 使用 cloned state，training cost 与 audit cost 分离。
- 新增/修改 `dgkan/kernels/v17_*`：family kernel correctness/readback 入口；corrective pass 将 repaired path 改为 torch family-specific repair，但 `official_fused_kernel_complete=0`。
- 新增 `experiments/run_v17_common.py` 及 v170/v171/v172/v173/finalize 包装入口。
- 修复 v17.1 required artifact 合同：v17.1 code packet 使用 `v17_1_code_review_packet/...`，不再把 legacy `v17_code_review_packet/...` 目录 manifest 误判为缺失。
- 修复 source retention ratio：只有前一 horizon 的 9-row mean source 为正时才定义 retention ratio；h1600 非正而 h3200 转正记录为 late rebound，不写成 retained source。

## 分析与 insight

- S0 的价值在于先确认 import closure、LineC golden、update sign、kernel gradcheck 是否可信；若这里失败，不能写 scientific no-go。
- corrective pass 的机制矩阵与 h800/h1600/h3200 extension 都是 fresh real-dataset rows；这解决了之前 low-budget/top-only 违背计划覆盖的问题。
- h3200 出现 late rebound 时不能自动解释成 retention；本轮 retention ratio 修复后，长期推进要看 positive source 是否跨 horizon 连续留存，而不是只看终点反弹。
- AdamW overwrite diagnostic 提供了 FU 是否被 AdamW 抵消的证据链：`cos_FU_AdamW` 与 `optimizer_overwrite_projection` 为后续 AdamWWashesFU/IntrinsicNoSource 分类服务。
- Efficiency rows 分离了 `step_training_only_ms` 和 `linec_audit_ms`，避免把 audit readback 误计入 carrier 训练效率。
- 由于 S4/S5 未执行且 official fused kernel 未完成，route 最高只能是 partial/exploration 级别；任何 promotion_allowed=1 都必须等 real-transfer official rows 和 official efficiency/kernel gates 真实通过。
