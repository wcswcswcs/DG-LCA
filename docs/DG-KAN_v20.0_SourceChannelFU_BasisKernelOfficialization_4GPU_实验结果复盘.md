# DG-KAN v20.0 Source-Channel FU + Basis Kernel Officialization 实验结果复盘

生成时间：2026-06-03 12:40:01 +0800

## Route

- route: `R-A1-MLPGenericSourceNotReproducible`
- route_detail: S0_4_pass=1, D-CHE_S1_rows=4, D-FOU_S1_rows=1, MLP_M16_retained_h3200=0, MLP_M16_retained_h4800=0, KAN_retained_h3200=0, KAN_retained_h4800=0, high_actuation_rows=15, high_actuation_source_rows=0, debt_complete=1, missing_named_v19_sources=0
- CodeRoute: S0_4-CodeMetricMechanismPassed
- EfficiencyRoute: D-CHE_S1_rows=4;D-FOU_S1_rows=1
- FunctionalRoute: MLP_M16_h3200=0;KAN_h3200=0;KAN_h4800=0
- promotion_allowed: 0

## 关键结果

- S0.4 preflight pass: 1
- D-CHE S1 official-ish pass rows: 4
- D-FOU S1 official-ish pass rows: 1
- MLP M16 retained h3200/h4800: 0 / 0
- KAN retained h3200/h4800 candidate count: 0 / 0
- H4 high ActuationR2 rows / source-success rows: 15 / 0
- PopRisk source Spearman h1600/h3200: 

## Efficiency Evidence

| carrier | rows | officialish_pass_rows | best_forward_ratio | best_step_ratio |
|---|---:|---:|---:|---:|
| D-CHE | 12 | 4 | 1.0286997569856908 | 0.4577710676965765 |
| D-FOU | 8 | 1 | 1.241019463791623 | 0.5641365804414814 |
| D-RAT | 4 | 0 | 10.713551755041692 | 1.612053500876678 |
| D-RBF | 4 | 0 | 7.104393271272252 | 1.9615454732881377 |
| D-WAV | 4 | 0 | 11.942763589735762 | 2.518826126928077 |
| LQ | 4 | 0 | 6.869820872991035 | 1.2387477364660944 |

## MLP M16 Anatomy

| carrier | v20_id | h800 | h1600 | h3200 | h4800 | retained_h3200 | retained_h4800 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | CTRL-NoOpMatchedOverhead | -0.4744021495183309 | -0.7011289331648085 | -0.9004695812861124 | -0.8962909777959188 | 0 | 0 |
| MLP | CTRL-SGD | -0.051471299595303006 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| MLP | F1-MLP-M16-replay-exact | -0.02359339925977919 | 0.04039005438486735 | 0.021710203753577337 | -0.0386962890625 | 0 | 0 |
| MLP | F1a-remove-two-phase-warmup | 0.04735626114739312 | 0.05379288726382785 | 0.018807450930277508 | 0.013418740696377225 | 0 | 0 |
| MLP | F1b-remove-alternation | 0.1595442427529229 | -0.3347306383980645 | -0.6717666520012749 | -0.6747626331117418 | 0 | 0 |
| MLP | F1c-remove-LineC-filtered-source-estimator | -0.014416231049431695 | 0.025422957208421495 | -0.002265976534949409 | -0.02669964896308051 | 0 | 0 |
| MLP | F1d-random-same-norm-source-state | -0.48348740736643475 | -0.7099475065867106 | -0.9092100593778822 | -0.9048535426457723 | 0 | 0 |
| MLP | F1f-AdamW-primary-only | -0.4469364881515503 | -0.913026836183336 | -1.3936757511562772 | -1.5831840170754328 | 0 | 0 |
| MLP | F1j-matrix-block-hidden-writer | -0.4238426817788018 | -0.6283436086442735 | -0.787494937578837 | -0.7475992176267836 | 0 | 0 |
| MLP | F1k-output-readout-only-writer | -0.021943264537387423 | 0.038377722104390465 | 0.03123637702729967 | -0.003611319594913059 | 0 | 0 |

## KAN Source Writer Evidence

| carrier | variant | v20_id | h800 | h1600 | h3200 | h4800 | retained_h3200 |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R2-low-degree-k3-triton | CTRL-AdamW | 0.0 | 0.0 | -0.0003495481279161241 | -0.03352801005045573 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | CTRL-NoOpMatchedOverhead | -1.0617612335417006 | -0.9257967405849032 | -0.741528782579634 | -0.6387312942081027 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | CTRL-RandomMatchedNorm | -1.0629068348142836 | -0.9269749522209167 | -0.7426818543010287 | -0.6399048699273003 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | CTRL-SGD | -1.067404203944736 | -0.9480908380614387 | -0.791150364610884 | -0.7036332819196913 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | F3-PopRiskSlowState | -1.062144849035475 | -0.926807873778873 | -0.7443553937806023 | -0.644436862733629 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | F4-ExactReadoutActuation | -0.1259565618303087 | -0.042613612280951604 | 0.018079029189215765 | 0.05056570635901557 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | F5-MatrixBlockWriter | -1.0628902779685125 | -0.9267929858631558 | -0.7423784799045987 | -0.6396140787336562 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | F6-ReadoutCarrierWriter | -1.064917418691847 | -0.9460045430395339 | -0.7894899381531609 | -0.7003216213650174 | 0 |
| D-CHE | CHE-R2-low-degree-k3-triton | F6-TwoPhaseSourceWriter | -1.0748628113004897 | -0.9533248676194085 | -0.7924864623281691 | -0.7023135556115044 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | 0.0 | 0.0 | 0.0 | -0.008493158552381728 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | -0.9139841530058119 | -0.8232200543085734 | -0.6865101125505235 | -0.5889912446339926 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | -0.9132055309083726 | -0.8224401076634725 | -0.6857240200042725 | -0.5882134967380099 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | -0.9099630382325914 | -0.8176702790790134 | -0.6782084835900201 | -0.5774973763359917 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-PopRiskSlowState | -0.9155943261252509 | -0.8247762653562758 | -0.6879516442616781 | -0.5903783374362521 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F4-ExactReadoutActuation | -0.12618370850880942 | -0.06450030538770887 | 0.014635324478149414 | 0.07889314492543538 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F5-MatrixBlockWriter | -0.9149884780248007 | -0.8241657283571031 | -0.6873509089152018 | -0.5897314018673367 | 0 |

## H4 / Function-Space Actuation

| carrier | v20_id | ActuationR2 max | source_h1600 | route_precedence |
|---|---|---:|---:|---|
| D-CHE |  | 0.09360003471374512 |  | actuation_insufficient_or_not_measured |
| D-CHE | M21-exact-readout-fu0p0001 | 0.8924183249473572 |  | high_actuation_but_source_not_retained |
| D-CHE | M22-exact-readout-highcap-alt50-fu0p0001 | 0.9393141269683838 |  | high_actuation_but_source_not_retained |
| D-CHE | M23-exact-readout-ultracap-alt50-fu0p0001 | 0.983686089515686 | -0.042613612280951604 | high_actuation_but_source_not_retained |
| D-CHE | M23-exact-readout-ultracap-fu0p0001 | 0.9292091131210327 |  | high_actuation_but_source_not_retained |
| D-CHE | M25-adamw-ultracap-alt100-fu0p0001 | 0.9999836683273315 |  | high_actuation_but_source_not_retained |
| D-CHE | M26-gated-adamw-ultracap-alt100-fu0p0001 | 0.9999852180480957 |  | high_actuation_but_source_not_retained |
| D-CHE | M28-adamw-multibatch-gated-alt100-fu0p00005 | 0.9999886751174927 |  | high_actuation_but_source_not_retained |
| D-CHE | M28-adamw-multibatch-gated-alt200-fu0p0001 | 0.9999926686286926 |  | high_actuation_but_source_not_retained |
| D-FOU | M21-exact-readout-fu0p0001 | 0.9659602046012878 |  | high_actuation_but_source_not_retained |
| D-FOU | M22-exact-readout-highcap-alt50-fu0p0001 | 0.9796541929244995 |  | high_actuation_but_source_not_retained |
| D-FOU | M23-exact-readout-ultracap-alt50-fu0p0001 | 0.9839740991592407 |  | high_actuation_but_source_not_retained |
| D-FOU | M23-exact-readout-ultracap-fu0p0001 | 0.9810327887535095 |  | high_actuation_but_source_not_retained |
| D-FOU | M25-adamw-ultracap-alt100-fu0p0001 | 0.9999100565910339 |  | high_actuation_but_source_not_retained |
| D-FOU | M26-gated-adamw-ultracap-alt100-fu0p0001 | 0.9999153017997742 |  | high_actuation_but_source_not_retained |
| D-FOU | M23-exact-readout-ultracap-alt50-fu0p0001 | 0.9907996654510498 | -0.06450030538770887 | high_actuation_but_source_not_retained |

## Failure Taxonomy

| failure_class | severity | evidence |
|---|---|---|
| KANSourceWriterNoRetainedCandidate | functional | v20_kan_source_writer_matrix.csv |

## 修改记录

- 新增 v20 source-channel/function-space/profiling helpers 与计划要求的 kernel status shim。
- 新增 v20 S0.4、basis officialization、MLP anatomy、KAN source writer、function-space actuation、merge finalize runners。
- v20 finalizer 会把缺失 standalone H10-H13 source wrappers 记录为 audit blocker；没有把 v19 partial positive 写成 v20 success。

## 分析 / Insight / 结论

- v20 的核心判定仍以 9-row grouped source 和 matched controls 为准；single-seed smoke 或 late positive 不触发 promotion。
- high ActuationR2 只证明局部 output displacement 可执行，不等于 source retention；必须同时看 h800/h1600/h3200/h4800 source chain。
- S1 efficiency official-ish 比 v19 E2 更严格：forward/step/memory/audit/gradcheck/no-materialize/full-loop/official fused 同时满足才算。
- `promotion_allowed` 只有 S5 全部真实满足才可为 1；本轮若 route 仍为 0，说明证据链没有达成，而不是预算性放弃。

## 2026-06-03 12:45 追加审计补充

### 本轮是否达成 v20 目标

没有达成。最终 route 仍为 `R-A1-MLPGenericSourceNotReproducible`，`promotion_allowed=0`。

这不是因为 S0.4 或 basis efficiency 卡住：S0.4 全部通过，D-CHE 有 4 条 S1 official-ish pass rows，D-FOU 有 1 条 S1 official-ish pass row。失败点在 functional：MLP M16 fresh replay 与 x3 independent offsets 都没有 retained source，KAN source writers 也没有 h3200/h4800 retained candidate。

### 实际修复 / blocker 处理

- 修复 `dgkan/profiling/efficiency_v20.py`：S1 gate 现在区分 `audit_overhead_ratio_raw` 与 `audit_overhead_ratio_training_loop`。当 `audit_cost_separated=1` 时，LineC/horizon audit readback 不再被错误计入 training-loop overhead；raw audit cost 仍保留在表里。
- 修复 `experiments/run_v20_basis_kernel_officialization.py`：merge-only 阶段会用当前 gate 重新计算 `v20_officialish_gate`，避免 shard 旧 gate 污染最终 truth table。
- 修复 `experiments/run_v20_kan_source_channel_writer.py`：添加 repo-root `sys.path` guard；初次直接运行 wrapper 出现 `ModuleNotFoundError: experiments`，修复后 py_compile 通过并完成 162-row KAN writer full run。
- 运行中发现 GPU monitor 脚本一次死锁：四个 efficiency shards 已于 11:48 写完 b0-b3 表，但 bash `wait` 同时等 monitor，monitor 等 stop file。已手动写入 stop file 释放 monitor，并完成 merge-only；该事件没有改变 profiling row 数据。
- KAN writer 中 `M13-LowRankMatrixBlockFU` 触发一次 PyTorch SVD fallback warning；run 没有失败，相关 rows 已真实落表。

### MLP M16 independent offsets

计划要求 replay 失败时做 x3 independent offsets。本轮补跑 offsets 100000/200000/300000，结果如下：

| init_seed_offset | h800 | h1600 | h3200 | h4800 | retained_h3200 | retained_h4800 |
|---:|---:|---:|---:|---:|---:|---:|
| 100000 | -0.051207251018948026 | 0.06688324610392253 | 0.06185562743080987 | 0.07427934143278334 | 0 | 0 |
| 200000 | -0.10146198007795545 | -0.06255396207173665 | -0.014500154389275445 | 0.024966716766357422 | 0 | 0 |
| 300000 | -0.08991732862260607 | 0.00588577323489719 | -0.07053809033499824 | -0.07195277346505059 | 0 | 0 |

解释：offset100000 的 h1600/h3200/h4800 是正的，但 h800 为负；按 retention 定义，前一 horizon/早期 source 非正不能写 retained source。三个 offsets 的 retained_h3200/retained_h4800 全为 0，因此 v19 的 MLP M16 retained source 在 v20 fresh replay 下不能作为稳定机制锚点。

### KAN writer 细证据链

KAN full 9-row writer 覆盖 D-CHE/D-FOU，包含 PopRisk slow-state、ExactReadout actuation、MatrixBlock writer、Readout carrier writer、TwoPhase source writer 和 controls。无 retained candidate。

最接近的线索是 ExactReadout late positive：

| carrier | variant | writer | h800 | h1600 | h3200 | h4800 | retained |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R2-low-degree-k3-triton | F4-ExactReadoutActuation | -0.1259565618303087 | -0.042613612280951604 | 0.018079029189215765 | 0.05056570635901557 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F4-ExactReadoutActuation | -0.12618370850880942 | -0.06450030538770887 | 0.014635324478149414 | 0.07889314492543538 | 0 |

解释：h3200/h4800 为正但 h800/h1600 为负，只能写 late positive / delayed recovery 线索，不能写 source retention，更不能进入 S4/S5。

### Efficiency 修复后的边界

- D-CHE S1 pass rows: 4/12，best forward ratio 1.0286997569856908，best step ratio 0.4577710676965765。
- D-FOU S1 pass rows: 1/8，best forward ratio 1.241019463791623，best step ratio 0.5641365804414814。
- LQ/RAT/RBF/WAV 仍没有 S1 rows；它们在 v20 中只保留 targeted repair/smoke 证据，不能当作 official proof carrier。

### 最终 insight

v20 把 v19 的两个 blocker 拆开了：basis kernel officialization 在 D-CHE/D-FOU 上实质推进到了 S1；functional source-channel 没有推进到 retained source。更关键的是，v19 依赖的 MLP M16 anchor 在 v20 fresh replay 和 x3 offsets 下没有复现。因此当前不能继续说“MLP retained source 已稳，只差 KAN carrier”；更诚实的结论是：**D-CHE/D-FOU efficiency 变得可用，但 functional source writer 的可复现机制锚点丢失，KAN ExactReadout 只有 late-positive 线索，没有 retained source。**

## 2026-06-03 13:20 Case A continuation / F3-F5 repair 追加复盘

### 是否达成 v20 目标

没有达成。

- route: `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource`
- promotion_allowed: 0
- Case A full rows / grouped / traces: 99 / 11 / 990
- Case A full retained candidates: 0
- exact-K32 PopRisk smoke positive h800+h1600 specs: 0
- M36 momentum-warm readout smoke positive h800+h1600 specs: 0
- PopRisk score vs source Spearman h1600/h3200: -0.22445302445302445 / -0.2030888030888031

解释：v20 初轮已经落到 Case A：MLP M16 不可复现。按计划，这时不能继续把 M16 迁移到 KAN，而要在 MLP 上继续 PopRisk/SNR slow-state 与 matrix/block source discovery。本轮正是对 Case A 的继续执行。

### 本轮真实修复

- 修复 `experiments/run_v17_common.py`：`M11/M12/M14` 等 slow-state writer 之前在默认训练分支没有把 `slow_state` 传回 `make_update()`，导致 M14 名义上是 slow-state，实际每步重置。本轮改为真实跨 step 累积。
- 修改 `dgkan/fu/mechanisms.py`：为 PopRisk/SNR 增加 diagnostics，包括 `PopRisk_SNR_score`、micro examples、exact variance 标记、SNR mean/max、slow-state norm、block projection 标记。
- 新增 `M33-PopRiskMatrixBlockSlowFU`：PopRisk/SNR slow-state 后投到 matrix block。
- 新增 `M34-ExactPopRiskSlowFU` 与 `M35-ExactPopRiskMatrixBlockSlowFU`：按计划 F3 blocker 修复，把 micro-split 提到 K=32，并使用 exact/unbiased variance。
- 新增 `M36-MomentumWarmReadoutBlockFU`：针对 F5b “h1600 positive 但 h800/h3200 断链”的线索，先用 M2-style momentum warmup，再切到 readout-block pulse。
- 修改 `experiments/run_v20_common.py`：新增上述 Case A F3/F5 repair specs，全部可通过 `--spec-ids` 精确复现。

### Case A full 9-row evidence

| v20_id | rows | h800 | h1600 | h3200 | h4800 | retained |
|---|---:|---:|---:|---:|---:|---:|
| F3a-MLP-PopRiskSlowState-fu0p0005 | 9 | -0.4426127274831136 | -0.5602769586775038 | -0.5740269290076362 | -0.3881864680184258 | 0 |
| F3a2-MLP-PopRiskSlowState-fu0p0001 | 9 | -0.4997145864698622 | -0.6859739886389838 | -0.8477442529466417 | -0.8180199331707425 | 0 |
| F3b-MLP-PopRiskMatrixBlock-fu0p0005 | 9 | -0.4151933193206787 | -0.5541439056396484 | -0.6289181576834785 | -0.5148687097761366 | 0 |
| F3b2-MLP-PopRiskMatrixBlock-fu0p0001 | 9 | -0.517257399029202 | -0.7103677325778537 | -0.8838469717237685 | -0.8646737602021959 | 0 |
| F5a-MLP-LowRankMatrixBlock-fu0p0005 | 9 | -0.4782227675120036 | -0.6150543159908719 | -0.6895583338207669 | -0.5939959420098199 | 0 |
| F5a2-MLP-ExplicitMatrixBlock-fu0p0005 | 9 | -0.4878402551015218 | -0.6502929528554281 | -0.7678877777523465 | -0.6961241695615981 | 0 |
| F5b-MLP-OutputReadoutBlock-fu0p0001 | 9 | -0.0421242978837755 | 0.038103964593675405 | -0.010300629668765597 | -0.048463530010647245 | 0 |

Controls in the same Case A matrix:

| v20_id | rows | h800 | h1600 | h3200 | h4800 |
|---|---:|---:|---:|---:|---:|
| CTRL-SGD | 9 | -0.06980632411109076 | -0.0028549830118815103 | 0.0 | 0.0 |
| F1f-AdamW-primary-only | 9 | -0.5742698245578342 | -1.0368160274293687 | -1.5292708343929715 | -1.7407479683558147 |
| F1d-random-same-norm-source-state | 9 | -0.5372852484385172 | -0.747939215766059 | -0.9537383715311686 | -0.9646786981158786 |
| CTRL-NoOpMatchedOverhead | 9 | -0.5412375397152371 | -0.7514509095085992 | -0.9572984112633599 | -0.9689438740412394 |

关键判定：

- F3 PopRisk/SNR slow-state 全部为负；`PopRisk_SNR_score` 对 h1600/h3200 source 的 Spearman 也是负数，因此不能说 PopRisk score 正在预测 retention。
- F5 matrix/block writer 全部没有 retained candidate。
- F5b output-readout block 是最接近的线索，h1600=0.038103964593675405，但 h800/h3200/h4800 均为负；按 retention 定义不能升级。

### Repair smoke evidence

Plan F3 写明：如果 PopRisk score noisy，要增加 micro-split K、使用 exact variance，并切到 block-level score。本轮已做 exact-K32 与 exact-K32 block smoke：

| repair | spec | h800 | h1600 | decision |
|---|---|---:|---:|---|
| exact-K32 | F3r1-MLP-ExactPopRiskK32-fu0p0005 | -0.5322269201278687 | -0.7761497497558594 | negative smoke |
| exact-K32 block | F3r3-MLP-ExactPopRiskK32Block-fu0p0005 | -0.3986562490463257 | -0.6715526580810547 | negative smoke |

针对 F5b 的 h1600-only positive，本轮新增 M36：momentum warmup 后 readout-block pulse：

| repair | spec | h800 | h1600 | decision |
|---|---|---:|---:|---|
| M36 | F5r1-MLP-MomentumWarmReadoutBlock-alt50-fu0p0001 | -0.12453138828277588 | -0.05021870136260986 | negative smoke |
| M36 | F5r2-MLP-MomentumWarmReadoutBlock-alt100-fu0p00005 | -0.023813247680664062 | -0.07754659652709961 | negative smoke |

这些 smoke 都是 D=MNIST seed0、steps=1600 的早期 diagnostic。因为 h800/h1600 已经为负，所以没有升级 full 9-row；这符合 evidence-first，不把负 smoke 烧成更大的 full run。

### 4GPU / runtime evidence

- Case A full run GPU snapshot rows: 800
- max memory used MB: 759.0
- max GPU util percent: 35.0
- shard rows: 25 / 25 / 25 / 24
- no blocked rows: 0/99 blocked

### Artifacts

- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_mlp_source_discovery_summary.csv`
- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_mlp_source_discovery_matrix.csv`
- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_mlp_source_discovery_traces.csv`
- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_exactk32_smoke_summary.csv`
- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_m36_smoke_summary.csv`
- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_continuation_decision.csv`
- `results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20/v20_case_a_continuation_packet.zip`

### 追加结论 / insight

v20 现在比 12:45 的结论更窄：不只是 M16 anchor 丢失，连按计划继续的 MLP PopRisk/SNR slow-state、PopRisk+matrix-block、matrix-block writer、output-readout writer 都没有形成 retained source。F5b 曾给出 h1600 正值，但无法保住 h800/h3200/h4800；M36 的 momentum-warm readout 修复也没有 smoke signal。

因此当前不能诚实地继续沿同一 PopRisk/matrix/readout family 做参数 sweep。可信边界是：

- `promotion_allowed=0`
- `case_a_continue_same_family_recommended=0`
- `new_source_theory_required=1`

这不是证明所有 source writer 不可能；它只说明 v20 计划 Case A 下已经实现并运行的 F3/F5 同族方向没有可升级信号。下一步若继续，需要先提出新的 train-only source observability / toy correctness 理论，而不是继续给 M14/M33/M34/M35/M36 调 scale 或 alt period。

## 2026-06-03 13:55 F7 Split-Fisher source-observability 追加复盘

### 是否达成 v20 目标

没有达成。

- route: `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7SplitFisherNoRetained`
- promotion_allowed: 0
- F7 initial smoke rows: 7
- M39 final transition full rows / grouped / traces: 63 / 7 / 630
- M39 transition full retained candidates: 0
- required artifact missing count: 0

本轮是在 Case A 的 F3/F5 同族方向失败后，按 `new_source_theory_required=1` 继续提出并实现的新 source-observability 机制，不是继续调 M14/M33/M36。

### 新理论 / 修改记录

新增机制：

- `M37-SplitFisherAgreementSlowFU`
  - 用 train split A/B 的梯度符号一致性作为 source-observability。
  - 用 diagonal Fisher/RMS 归一化，避免高幅度单 split 主导。
  - 用 corrupted-label train batch 做坏方向投影和 gate，不使用 validation/test/future/query。
- `M38-AdamWSplitFisherAgreementResidualFU`
  - 同一个 Split-Fisher source estimator，作为 AdamW residual 审计 AdamW coupling。
- `M39-MomentumWarmSplitFisherFU`
  - 先做 M2-style momentum warmup，再切到 Split-Fisher source-state，专门修复 M37 的 h800 negative / h1600 positive phase mismatch。

实际修复：

- `experiments/run_v17_common.py`
  - 新增 M37/M38/M39 train loop branch。
  - trace 增加 `source_state_balance_mean`、`source_state_current_cos`、`source_state_whitened_norm`。
  - 修复 M39 warmup 语义：warmup 期间每 step 做 M2-style update，而不是只在 `alt_period` step 做。
  - 修复 momentum warmup primary optimizer：M2/M16/M36/M39 使用 `sgd-momentum` primary。
- `dgkan/fu/mechanisms.py`
  - 新增 M37/M38/M39 和 semantic contract rows。
- `experiments/run_v20_common.py`
  - 新增 F7 specs。

### F7 initial smoke

| spec | h800 | h1600 | decision |
|---|---:|---:|---|
| F7a M37 alt50 fu1e-4 | -0.15173625946044922 | 0.06143307685852051 | h1600-only, no full |
| F7b M37 alt100 fu5e-5 | -0.0352635383605957 | 0.03369641304016113 | h1600-only, no full |
| F7c M38 AdamW residual | -0.21722519397735596 | -0.6275719404220581 | negative |

解释：M37 不是空信号，gate 有接受且 h1600 为正；但 h800 为负，按 retention 定义不能升级。M38 说明 AdamW coupling 没有自动修复这个新 source estimator。

### M39 semantic blocker and repair chain

M37 的形态是 delayed source，所以做了 M39 momentum warmup 修复。过程中发现两个实现语义 blocker：

| phase | h800 | h1600 | audit decision |
|---|---:|---:|---|
| sparse-warmup bug smoke | 0.012569785118103027 | 0.1938631534576416 | positive but invalid for final decision |
| fixed every-step warmup smoke | -0.1283029317855835 | -0.12300193309783936 | negative |
| momentum-primary fixed smoke | 0.19289636611938477 | -0.39523887634277344 | h800 positive, h1600 washout |

解释：

- sparse-warmup bug smoke 曾出现 h800/h1600 双正，但后来确认 M39 warmup 被 `alt_period` 包住，只每 50 step 做一次 M2 residual；这不是计划中的 M2-style warmup，因此该正例只保留为 bug audit，不作为成功证据。
- 修成 every-step warmup 后，smoke 变负。
- 再修成 `sgd-momentum` primary 后，h800 转正但 h1600 washout，说明 M2-style early source 仍会被后续阶段洗掉。

### Transition smoke and full escalation

为了定位 transition mismatch，本轮只做了 warmup length smoke：

| spec | h800 | h1600 | full decision |
|---|---:|---:|---|
| F7r1 warm400 | 0.21023499965667725 | -0.19479095935821533 | no full |
| F7r2 warm800 | 0.4864776134490967 | 0.11225533485412598 | upgrade |
| F7r3 warm1200 | 0.5047791004180908 | 0.21117687225341797 | upgrade |

因为 warm800/warm1200 在 MNIST seed0 上 h800/h1600 双正，所以升级到 3 datasets x 3 seeds full 9-row，并加入 `F1b-remove-alternation` 判断它是否只是 M2 warmup。

### Full 9-row result

| spec | rows | h800 | h1600 | h3200 | h4800 | retained |
|---|---:|---:|---:|---:|---:|---:|
| F1b M2 momentum only | 9 | 0.2717749807569716 | -0.129564192559984 | -0.43085387017991805 | -0.43126800325181747 | 0 |
| F7r2 M39 warm800 | 9 | 0.22176614072587755 | -0.2540358834796482 | -0.5713619920942519 | -0.5791374842325846 | 0 |
| F7r3 M39 warm1200 | 9 | 0.325644400384691 | -0.09721063905292088 | -0.40291719966464573 | -0.40762672159406876 | 0 |

Controls:

| control | rows | h800 | h1600 | h3200 | h4800 |
|---|---:|---:|---:|---:|---:|
| CTRL-SGD | 9 | -0.08819409211476643 | -0.0030743281046549478 | 0.0 | 0.0 |
| CTRL-AdamW | 9 | -0.4792142974005805 | -0.9451240963406033 | -1.4260414573881361 | -1.6287080579333835 |
| CTRL-RandomMatchedNorm | 9 | -0.5056368245018853 | -0.7155049112108018 | -0.9167894787258573 | -0.9192315075132582 |
| CTRL-NoOpMatchedOverhead | 9 | -0.49252767033047146 | -0.702603260676066 | -0.9041100343068441 | -0.9073196649551392 |

关键 row-level 证据：

- F7r3 在 MNIST seed0 是 retained-like：h800=0.5743124485015869, h1600=0.21117687225341797, h3200=0.0924227237701416, h4800=0.06605172157287598。
- F7r3 在 KMNIST seed0 也为正：h800=0.48663628101348877, h1600=0.19899308681488037, h3200=0.15762150287628174, h4800=0.15449464321136475。
- 但 Fashion-MNIST seed2 强负：h800=-0.39583420753479004, h1600=-1.4409964084625244, h3200=-1.853568434715271, h4800=-1.7556462287902832。
- grouped mean 因 dataset/seed heterogeneity 和 post-warmup washout 转负，因此不能写 retained source。

### GPU / runtime evidence

- F7 initial smoke GPU snapshots: 16 rows, max_mem=693 MB, max_util=9%。
- M39 transition full GPU snapshots: 92 rows, max_mem=691 MB, max_util=10%。
- full run blocked rows: 0/63。

### Artifacts

- `v20_case_a_f7_fisher_smoke_summary.csv`
- `v20_case_a_f7_fisher_smoke_matrix.csv`
- `v20_case_a_f7_fisher_smoke_traces.csv`
- `v20_case_a_f7_m39_transition_smoke_summary.csv`
- `v20_case_a_f7_m39_transition_full_summary.csv`
- `v20_case_a_f7_m39_transition_full_matrix.csv`
- `v20_case_a_f7_m39_transition_full_traces.csv`
- `v20_case_a_f7_split_fisher_decision.csv`
- `v20_case_a_f7_split_fisher_rollup.csv`
- `v20_case_a_f7_split_fisher_decision.md`

### 分析 / Insight / 结论

- F7 比 F3/F5 有更明确的机制读数：train split Fisher agreement 能在部分 rows 上形成局部正 source，M39 能恢复 strong h800。
- 但它仍没有解决 retained channel：full grouped h1600/h3200/h4800 均为负。
- M39 的 full result 说明“先用 M2 找 h800，再用 Fisher 保留”这个因果假设没有成立；Fisher stage 多数情况下没有把 early momentum source 转成 long-horizon retained source。
- 这也再次说明 single-seed MNIST smoke positive 不能 promotion；F7r3 的 MNIST/KMNIST seed0 很好看，但 Fashion-MNIST seed2 足以把 grouped evidence 打穿。
- 当前不能把 F7 写成 breakthrough，也不能把 `promotion_allowed` 改成 1。

当前可信边界：

- `promotion_allowed=0`
- `case_a_continue_same_family_recommended=0`
- `new_source_theory_required=1`

在 v20 Case A 已经覆盖 M16 replay、PopRisk/SNR slow-state、matrix/block writer、readout block、Split-Fisher source-state 和 momentum-warm Split-Fisher 之后，继续在同一 MLP source writer family 内调 warmup/alt/threshold 的审计价值已经很低。下一步若继续，需要新的 toy-correctness/source-observability 假设，或先做 dataset/seed heterogeneity 的机制级解释，而不是继续同族 sweep。

### 最终验证 / 审计边界

- py_compile 通过：`dgkan/fu/mechanisms.py`、`experiments/run_v17_common.py`、`experiments/run_v20_common.py`、`experiments/run_v20_mlp_retained_source_anatomy.py`。
- `v20_required_artifact_manifest.csv` rows=64，required missing count=0。
- `v20_case_a_f7_split_fisher_packet.zip` 已刷新，包含 source files、v20 plan、执行日志、复盘日志、`official_v20/v20_route_decision.json`、manifest 与 F7 key artifacts。
- route 仍为 `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7SplitFisherNoRetained`。
- `promotion_allowed=0`，`official_success_reached=0`，`f7_m39_transition_full_retained_candidates=0`。
- 结束进程检查未发现持续运行的 v20 training 或 GPU monitor；`ps` 输出只包含瞬时 `nvidia-smi` 查询和检查命令自身。

最终结论保持不变：v20 目标没有达成。F7 是一次真实的新 source-observability 尝试，并且修复了两个实现语义 blocker；但 full grouped 证据仍没有 retained source。当前不能写 breakthrough / promotion-ready。继续推进需要新的 toy-correctness/source-observability 理论或 dataset/seed heterogeneity 的机制级解释，再落到新机制；继续同族 warmup/alt/threshold sweep 不具备足够审计价值。

## 2026-06-03 F8-F10 Momentum Washout Repair 追加复盘

### 为什么继续

F7 之后我没有把“同族 sweep 价值低”当成完成，而是先做了更窄的失败解剖。F7/M39 的 row-level 证据显示：部分 MNIST/KMNIST seed 能形成 h800 甚至 long horizon 正 source，但 grouped mean 在 h1600 后 washout。于是本轮只针对一个可检验问题继续：early momentum source 到底是被 post-warmup 训练反向抹掉，还是只是优势会被 controls 追上。

### 新增机制与修改

- `M40-MomentumWarmAntiWashoutFU`：warmup 后移除当前梯度中与 warmup source 反向的投影。
- `M41-MomentumWarmSourceAnchorFU`：warmup 后继续 SGD primary，并按 train-only gate 追加 source-anchor residual。
- `M42-MomentumWarmHoldFU`：warmup 后完全 hold，用来判断后续训练是否造成 washout。
- `M43-MomentumCycleHoldFU`：固定周期 momentum refresh + hold，用来测试 source 优势能否周期续命。
- 修改文件：`dgkan/fu/mechanisms.py`、`experiments/run_v17_common.py`、`experiments/run_v20_common.py`。
- 方向来源：全部仍为 train stream；没有读取 validation/test/future/query，也没有 dataset-name branch 或 seed-specific scale。

### Smoke 到 Full

| probe | smoke h800 | smoke h1600 | decision |
|---|---:|---:|---|
| F8a M40 warm800 | 0.3923906087875366 | 0.12926089763641357 | upgraded |
| F8b M40 warm1200 | 0.4177408218383789 | 0.1239250898361206 | upgraded |
| F8c M41 warm800 | 0.2523636817932129 | -0.11107504367828369 | no full |
| F8d M41 warm1200 | 0.36762428283691406 | 0.03485369682312012 | upgraded |
| F9a M42 hold warm800 | 0.522982120513916 | 0.23192834854125977 | upgraded |
| F9b M42 hold warm1200 | 0.3658924102783203 | 0.0013573169708251953 | no full |
| F10a M43 cycle-hold warm800 | 0.5634734630584717 | 0.26964306831359863 | upgraded |

Smoke positive 只触发 full rerun；没有被写成成功。

### Full 9-row Results

| phase | spec | rows | h800 | h1600 | h3200 | h4800 | retained |
|---|---|---:|---:|---:|---:|---:|---:|
| F8 antiwashout | F8a M40 warm800 | 9 | 0.2275738517443339 | -0.09565561347537571 | -0.3983711534076267 | -0.4689307345284356 | 0 |
| F8 antiwashout | F8b M40 warm1200 | 9 | 0.33461103174421525 | -0.043901529577043324 | -0.30157148838043213 | -0.34824442863464355 | 0 |
| F8 source anchor | F8d M41 warm1200 | 9 | 0.24928910202450222 | -0.19618584050072563 | -0.4315970738728841 | -0.39400559001498753 | 0 |
| F9 hold | F9a M42 warm800 | 9 | 0.2418862051433987 | 0.02575813399420844 | -0.1102049085828993 | -0.07841628127627903 | 0 |
| F10 cycle hold | F10a M43 warm800 | 9 | 0.2562202347649468 | 0.040092163615756564 | -0.31356167793273926 | -0.358368886841668 | 0 |

关键 row-level 证据（F9a，best h3200）：

| dataset | seed | h800 | h1600 | h3200 | h4800 |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | 0.12889719009399414 | -0.2554903030395508 | -0.24195456504821777 | -0.06497037410736084 |
| Fashion-MNIST | 1 | 0.4129772186279297 | 0.09603440761566162 | -0.029742717742919922 | 0.0008085966110229492 |
| Fashion-MNIST | 2 | -0.2491605281829834 | -0.5212523937225342 | -0.5033484697341919 | -0.29630446434020996 |
| KMNIST | 0 | 0.23569321632385254 | -0.06264472007751465 | -0.20002269744873047 | -0.13778114318847656 |
| KMNIST | 1 | 0.6059690713882446 | 0.5965802669525146 | 0.3149811029434204 | 0.30014538764953613 |
| KMNIST | 2 | 0.5336732864379883 | 0.23627877235412598 | 0.03171181678771973 | 0.003129124641418457 |
| MNIST | 0 | 0.4737701416015625 | 0.11319124698638916 | -0.014809012413024902 | 0.016460657119750977 |
| MNIST | 1 | -0.13933706283569336 | -0.28810083866119385 | -0.47079241275787354 | -0.5476449728012085 |
| MNIST | 2 | 0.17449331283569336 | 0.317226767539978 | 0.12213277816772461 | 0.020410656929016113 |

### Diagnostics

| phase | spec | diagnostic rows | gate accepts | signal gain mean | corrupt gain mean | current cos mean |
|---|---|---:|---:|---:|---:|---:|
| F8 antiwashout | F8b M40 warm1200 | 81 | 80 | -2.959157206482309e-06 | -0.0003897239350610309 | 0.7078561928169227 |
| F8 source anchor | F8d M41 warm1200 | 81 | 50 | 5.4924309248841986e-08 | -1.2377690937783984e-05 | 0.7065534416908099 |
| F9 hold | F9a M42 warm800 | 81 | 45 | 0.0 | 0.0 | 0.7154438878053132 |
| F10 cycle hold | F10a M43 warm800 | 81 | 54 | 0.0 | 0.0 | 0.7767732102239941 |

解释：

- F8 anti-washout 的 `source_state_antiwashout_removed_norm` 在 smoke/full 中基本没有形成有效修复；current cosine 多数为正，说明 washout 不是简单的“当前梯度与 source 反向”。
- M41 source-anchor gate 有 50/81 accepts，但 grouped h1600/h3200 更差，说明继续追加 anchor residual 不是充分解。
- F9 hold 是最有信息量的结果：h800/h1600 grouped mean 为正，但 h3200/h4800 转负。这说明停止 post-warmup 写入可以保住一部分 h1600 source，但 controls 会在更长 horizon 追上。
- F10 周期 refresh 反而比 F9 hold 的 h3200 更差，说明 naive periodic refresh 会重新引入 washout，而不是续命 source。

### GPU / Artifact Evidence

| run | GPU rows | max mem MB | max util % |
|---|---:|---:|---:|
| F8 antiwashout smoke | 20 | 693.0 | 14.0 |
| F8 source-anchor smoke | 20 | 693.0 | 14.0 |
| F8 source-anchor full | 80 | 693.0 | 15.0 |
| F9 hold smoke | 16 | 693.0 | 12.0 |
| F9 hold full | 80 | 693.0 | 18.0 |
| F10 cycle-hold smoke | 16 | 693.0 | 11.0 |
| F10 cycle-hold full | 76 | 693.0 | 11.0 |

新增 official artifacts：

- `v20_case_a_f8_f10_washout_repair_rollup.csv`
- `v20_case_a_f8_f10_washout_repair_diagnostics.csv`
- `v20_case_a_f8_f10_washout_repair_decision.csv`
- `v20_case_a_f8_f10_washout_repair_decision.md`
- `v20_case_a_f8_f10_washout_repair_packet.zip`
- F8/F9/F10 smoke/full summary、matrix、traces、GPU snapshot CSV 已复制到 `official_v20/` 根目录。

### F8-F10 结论

v20 目标仍未达成：

- route: `R-A1-MLPGenericSourceNotReproducible-CaseAContinuationNoRetainedSource-F7F10NoRetained`
- `promotion_allowed=0`
- `official_success_reached=0`
- `f8_f10_washout_repair_full_rows_total=45`
- `f8_f10_retained_h3200_candidates=0`
- `required_artifact_missing_count=0`

本轮把 blocker 进一步缩小：MLP 上可以恢复 h800，F9/F10 甚至能恢复 grouped h1600；但 h3200/h4800 仍无法 retained。现在不再只是 M16 偶然不可复现，也不是简单 anti-washout、source-anchor、hold 或周期 refresh 可以解决。继续同一 family 的 warmup/hold/anchor 参数扫描不具备足够审计价值；下一步需要新的 source-observability/toy-correctness 理论，或者先做 dataset/seed heterogeneity 的机制级解释后再落新机制。不能写 breakthrough，也不能把 `promotion_allowed` 改为 1。

### F8-F10 最终验证

- py_compile 通过：`dgkan/fu/mechanisms.py`、`experiments/run_v17_common.py`、`experiments/run_v20_common.py`、`experiments/run_v20_mlp_retained_source_anatomy.py`。
- `v20_case_a_f8_f10_washout_repair_rollup.csv` rows=5。
- `v20_case_a_f8_f10_washout_repair_diagnostics.csv` rows=4。
- `v20_case_a_f8_f10_washout_repair_decision.csv` rows=1。
- `v20_required_artifact_manifest.csv` rows=100，required missing=0。
- `v20_case_a_f8_f10_washout_repair_packet.zip` 已刷新，精确 size 由最终验证命令读取。
- `v20_results_bundle.zip` 已刷新，精确 size 由最终验证命令读取。
- 结束进程检查未发现持续运行的 v20 training 或 GPU monitor；`ps` 仅匹配验证命令自身。
