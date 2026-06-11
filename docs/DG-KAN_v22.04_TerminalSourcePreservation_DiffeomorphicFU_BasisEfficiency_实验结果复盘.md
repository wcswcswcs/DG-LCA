# DG-KAN v22.04 TerminalSourcePreservation DiffeomorphicFU BasisEfficiency 实验结果复盘

生成时间：2026-06-06 04:07:57 +0800

## Route

- route: `R3-DRATDRBFRepairBlocked`
- promotion_allowed: 0
- blocking_metric: `official_fused_missing;R4800_over_3200_below_0.50`
- next_codex_action: finish official fused D-RAT/D-RBF wiring while retiring exhausted terminal-preservation variants
- D-CHE/D-FOU S1 pass: 1 / 1
- D-RAT/D-RBF official fused blocked: 1
- D1 source-preserving groups / h4800 groups: 11 / 0
- best D1 candidate: `MLP-D1b-roworth-hidden-readout`
- best h4800 / retention ratio: 0.15684298011991712 / 0.4822038309091096
- terminal erosion dominant class: `OptimizerErosion`
- KAN mapping decision: `KANSourceChannelMismatch`

## Part 1 Code Audit

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 29/29 |  |
| compileall | 1 | py_compile | 0 |  |
| import_closure | 1 | import | 0 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 14/14 |  |
| mechanism_contracts | 1 | v22.04 mechanisms | 8/8 |  |
| kernel_gradcheck | 1 | kernels | 4 |  |
| profiler_phase_tests | 1 | tests | 3/3 |  |

## Part 2 Basis Efficiency / D-RAT / D-RBF

| carrier | S1_pass_rows | full_loop_official_closure | v22_04_same_kernel_runner_proof | v22_04_S1_pass | v22_04_decision | v22_04_blocker |
| --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 6 | 1 | 1 | 1 | OfficialEfficientCarrier |  |
| D-FOU | 3 | 1 | 1 | 1 | OfficialEfficientCarrier |  |

| carrier | profile_rows | micro_near_E1_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 14 | 10 | 1.1890159794580142 | 1.2043197684973785 | 1.03125 | MicroNearE1OfficialFusedBlocked | official_fused_missing |
| D-RBF | 14 | 2 | 1.7026325693797912 | 1.2708443620609988 | 1.0312490463256836 | MicroNearE1OfficialFusedBlocked | official_fused_missing |

## Part 3 Terminal Erosion Taxonomy

| terminal_erosion_class | groups | fraction |
| --- | --- | --- |
| DatasetLocalizedErosion | 1 | 0.017857142857142856 |
| InfoVolumeCollapse | 1 | 0.017857142857142856 |
| OptimizerErosion | 54 | 0.9642857142857143 |

| v22_id | h3200 | h4000 | h4800 | h4800_retention_ratio | terminal_erosion_class | terminal_erosion_evidence |
| --- | --- | --- | --- | --- | --- | --- |
| MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong | 0.32769686645931667 | 0.21169447236590916 | 0.13043878144688076 | 0.39804708191518406 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F146-early100-h800-source-slow-ema-terminal-source-preserve-gentle | 0.3173616627852122 | 0.20137612356079948 | 0.12014251285129124 | 0.3785665596685593 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | 0.33942452404234147 | 0.22343753443823922 | 0.14219152596261767 | 0.41891942358555095 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | 0.34045343928866917 | 0.2244438264105055 | 0.14314018024338615 | 0.4204398126876261 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F149-early100-h800-source-slow-ema-info-volume-diffeomorphic-target | 0.322532531287935 | 0.2065784732500712 | 0.1257212890519036 | 0.3897941350284081 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F150-early100-h800-source-slow-ema-low-rank-readout-transport | 0.33627867698669434 | 0.22029011779361302 | 0.13901315132776895 | 0.4133867558104772 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong | 0.3304118911425273 | 0.2144118779235416 | 0.13317129347059461 | 0.4030463098955162 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve | 0.3314318060874939 | 0.21546246939235264 | 0.13418350948227775 | 0.4048600858991034 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source | 0.3311949107382033 | 0.2151894768079122 | 0.13389651311768425 | 0.4042831238536882 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve | 0.3476023210419549 | 0.23160096009572348 | 0.15029900603824192 | 0.4323878091139131 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block | 0.32650257481469047 | 0.21051009164916146 | 0.12921792268753052 | 0.39576387034886124 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve | 0.3345329297913445 | 0.2185691793759664 | 0.13737184471554226 | 0.410637735427792 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor | 0.3424224787288242 | 0.2264139817820655 | 0.1451091104083591 | 0.42377215113636235 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator | 0.32183868686358136 | 0.20581589142481485 | 0.12448746297094557 | 0.38680080441576126 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport | 0.3335407144493527 | 0.21756445036994088 | 0.13635704583591884 | 0.40881679485826106 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F160-early100-h800-source-slow-ema-terminal-source-floor | 0.32949671480390763 | 0.21352261635992262 | 0.13233332501517403 | 0.401622593093012 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor | 0.31664108567767674 | 0.20062582360373604 | 0.1193248364660475 | 0.3768457154278255 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor | 0.3463519877857632 | 0.23032606972588432 | 0.14896754423777261 | 0.43010448760558817 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor | 0.3309069673220317 | 0.2149040632777744 | 0.1336647007200453 | 0.40393438011223753 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor | 0.32138868504100376 | 0.2053836915228102 | 0.12409891353713141 | 0.3861334244586068 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor | 0.33771517541673446 | 0.2217154105504354 | 0.14042934444215563 | 0.4158218364598758 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal | 0.32838569084803265 | 0.2123797701464759 | 0.1311053368780348 | 0.39924192963300137 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard | 0.31706199381086564 | 0.20103775130377877 | 0.11969963047239515 | 0.377527527136534 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |
| MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector | 0.33563917213016087 | 0.2196390794383155 | 0.13835234443346658 | 0.41220559434526777 | OptimizerErosion | monotone h3200->h4000->h4800 retained-source decay with positive h4800 rows |

_仅显示前 24 / 56 rows；完整 CSV 见 artifact。_

## Part 4 D1 Source-Preserving FU

| plan_line | v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | productive_h4800_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CTRL-AdamW | 9 | 0.0 | -0.8908253378338284 | -1.0994470450613234 | -1.272285474671258 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-NoOpMatchedOverhead | 9 | -1.1510869529512193 | -1.3135089211993747 | -1.401682482825385 | -1.468141343858507 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-RandomMatchedNorm | 9 | -1.1446411742104425 | -1.3085067669550579 | -1.396600776248508 | -1.4632093376583524 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-SGD | 9 | -0.6600571738349067 | -0.030274444156222872 | -0.009571996000077989 | 0.0 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| D1a-source-preserving-late-projection | MLP-D1a-SPP-lambda025 | 9 | 0.4732886188560062 | 0.31092920568254256 | 0.22278403573566014 | 0.15635869238111708 | 0.5028755405523361 | 9 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| D1a-source-preserving-late-projection | MLP-D1a-SPP-readout-only | 9 | 0.4447054929203457 | 0.2823181682162815 | 0.19416109720865884 | 0.127714474995931 | 0.452377811186739 | 7 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| D1b-Nora-row-orthogonal | MLP-D1b-roworth-hidden-readout | 9 | 0.48907800846629673 | 0.32672355241245693 | 0.23857712083392674 | 0.17214437325795492 | 0.5268808201517082 | 9 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| D1c-low-NDS-matrix-block | MLP-D1c-lowNDS-with-source-preservation | 9 | 0.4500727852185567 | 0.2876451810201009 | 0.1994700034459432 | 0.13302822576628792 | 0.4624733336206727 | 7 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| D1d-dual-memory-source-state | MLP-D1d-dualmem-longonly-terminal | 9 | 0.4676592813597785 | 0.3052816523445977 | 0.21712914440366957 | 0.15075196160210502 | 0.493812715059398 | 7 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| D1e-debt-aware-terminal-FU | MLP-D1e-combined-debt-aware | 9 | 0.46125078863567776 | 0.2989179955588447 | 0.21078213387065464 | 0.14443055788675943 | 0.4831778615962483 | 8 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| D1f-info-volume-preserving-FU | MLP-D1f-info-volume-plus-roworth | 9 | 0.4295783109135098 | 0.2672090530395508 | 0.17904667059580484 | 0.11260380347569783 | 0.4214071424407572 | 7 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-AdamW | 9 | -0.009399996863471137 | -0.9199768039915297 | -1.1415200961960688 | -1.3200003173616197 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-NoOpMatchedOverhead | 9 | -1.10538829697503 | -1.2978806893030803 | -1.3954215778244867 | -1.4658147361543443 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-RandomMatchedNorm | 9 | -1.1489323774973552 | -1.3411992258495755 | -1.4388693107499018 | -1.5093958510292902 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-SGD | 9 | -0.6346658865610758 | -0.0053694115744696725 | 0.0 | 0.0 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| D1a-source-preserving-late-projection | MLP-D1a-SPP-lambda050 | 9 | 0.5053239994578891 | 0.3129788902070787 | 0.21548091702991062 | 0.14515682061513266 | 0.46379108993290696 | 8 | 0 | TerminalErosion;PostH4000Erosion |
| D1a-source-preserving-late-projection | MLP-D1a-SPP-lambda100 | 9 | 0.45661331547631157 | 0.2641431755489773 | 0.16660667790306938 | 0.09616786572668287 | 0.3640747694003984 | 7 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-AdamW | 9 | -0.0001971986558702257 | -0.979147818353441 | -1.2034455471568637 | -1.386982838312785 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-NoOpMatchedOverhead | 9 | -1.0872336559825473 | -1.298830469449361 | -1.394963026046753 | -1.4673935572306316 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-RandomMatchedNorm | 9 | -1.0901426540480719 | -1.3025746213065252 | -1.398519171608819 | -1.470809433195326 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-SGD | 9 | -0.6196606622801887 | -0.011503855387369791 | 0.0 | 0.0 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| D1a-source-preserving-late-projection | MLP-D1a-SPP-lambda025 | 9 | 0.535184770822525 | 0.3235802716679043 | 0.2274504072136349 | 0.15502088268597922 | 0.4790801425776651 | 8 | 0 | TerminalErosion;PostH4000Erosion |
|  | CTRL-AdamW | 9 | -0.0001971986558702257 | -0.979147818353441 | -1.2034455471568637 | -1.386982838312785 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-NoOpMatchedOverhead | 9 | -1.0872336559825473 | -1.298830469449361 | -1.394963026046753 | -1.4673935572306316 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-RandomMatchedNorm | 9 | -1.0901426540480719 | -1.3025746213065252 | -1.398519171608819 | -1.470809433195326 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
|  | CTRL-SGD | 9 | -0.6196606622801887 | -0.011503855387369791 | 0.0 | 0.0 |  | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| D1b-Nora-row-orthogonal | MLP-D1b-roworth-hidden-readout | 9 | 0.5367387433846792 | 0.3252628246943156 | 0.22917521662182277 | 0.15684298011991712 | 0.4822038309091096 | 8 | 0 | TerminalErosion;PostH4000Erosion |

## Part 5 KAN Source Mapping

| carrier | variant | v22_id | h800 | h3200 | h4800 | h4800_retention_ratio | v22_04_source_mapping_decision | fresh_source_retention_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-AdamW | 0.0 | 0.0 | 0.0 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-NoOpMatchedOverhead | -1.6334137121836345 | -1.5455449488427904 | -1.4494818581475153 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-RandomMatchedNorm | -1.6256221400366888 | -1.5377527674039204 | -1.4416825506422255 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-SGD | -1.6285207801394992 | -1.53294352028105 | -1.4355009661780462 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW1-basis-estimate-readout-commit | -1.623139566845364 | -1.5316170387797885 | -1.434997320175171 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW10-h800-dual-memory-source-bank | -0.18122186263402304 | -0.09334645668665568 | 0.0027211440934075248 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW2-lowdegree-lowfreq-source-bank | -0.18363622162077162 | -0.0202968782848782 | 0.004610690805647109 |  | KANSourceChannelMismatch | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW9-h800-source-slow-ema-bank | -0.17689558532502916 | -0.08902051051457723 | 0.007046798865000407 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | 0.0 | 0.0 | 0.0 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | -1.5411422385109796 | -1.4102725783983867 | -1.3356308473481073 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | -1.5430146985583835 | -1.4121408263842266 | -1.3374980092048645 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | -1.5400735669665866 | -1.400064554479387 | -1.3185323410563998 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW1-basis-estimate-readout-commit | -1.5382218228446112 | -1.3981423444218106 | -1.3163674473762512 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW10-h800-dual-memory-source-bank | -0.09634670284059313 | 0.03452601035435995 | 0.10916988054911296 | 3.1619604880100773 | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW2-lowdegree-lowfreq-source-bank | -0.16873362329271105 | -0.07953935199313694 | -0.09196770191192627 |  | KANSourceChannelMismatch | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW9-h800-source-slow-ema-bank | -0.10394989119635688 | 0.02692327234480116 | 0.1015673279762268 | 3.7724733708248244 | KANSourceChannelMismatch | 1 |

## Part 6 Precommit Selector

| feature_name | uses_future | uses_validation | uses_test | uses_audit_target | AUC_predict_h4800_retention | precision_at_k | precommit_selector_pass | selector_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_loss_h100_h400 | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| train_stream_split_gain | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| cross_split_source_consistency | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| NDS_estimate | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| source_path_early_info_volume | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| row_radial_fraction | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| optimizer_conflict_proxy | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| train_stream_tail_proxy | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |
| LineC_fast_train_split_proxy | 0 | 0 | 0 | 0 |  | 0.0 | 0 | InsufficientPositiveProductiveLabels |

## 修改记录

- 新增 v22.04 runner：common、S0.11 truth gate、efficiency readback、D-RAT/D-RBF officialization wrapper、terminal erosion autopsy、D1 source-preserving FU aggregation、KAN source mapping、merge/finalize。
- 新增 `dgkan/fu/terminal_erosion.py`、`dgkan/fu/source_preservation.py`、`dgkan/profiling/efficiency_v22_04.py` 与 `efficiency_v22_03.py` 兼容层；这些 helper 只做审计分类/ratio 判定，不生成实验数值。
- 扩展 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`，注册计划命名 D1a-D1f source-preserving FU 候选，便于 fresh old-shape source-retention runner 直接执行。
- D1 混跑出现 ratio>0.50 但 early/h3200 chain 不成立的 retrospective-looking near-miss 后，追加 D1a/D1b 单候选复跑；单候选恢复 early+h3200 chain 但仍低于 R4800/3200 gate。
- 按 blocker 补测 D1a SPP lambda050/lambda100 强度分支；前两次 full 暴露 runner allowlist wiring 缺口并保留为 pre-wiring artifacts，修复 `experiments/run_v17_common.py` 外层机制集合、h800 slow-EMA active-update 集合、source-name/role 和 telemetry 集合后，用 `wired3` fresh full 作为正式科学判定。
- 执行 fresh D2 KAN source mapping：D-CHE/D-FOU x KSW1/KSW2/KSW9/KSW10 + matched controls，old-shape 6400 step，不把 efficiency closure 伪称为 KAN source-channel writer 成功。

## 分析 / Insight / 结论

- v22.04 的 functional 判据仍保持 productive h4800 gate：mean h4800 >= 0.005、R4800/3200 >= 0.50、row-positive >= 7/9；本复盘没有把 h4800 positive 或 ActuationR2 写成 promotion。
- 合法 early+h3200 链中的 D1 best candidate 为 `MLP-D1b-roworth-hidden-readout`，h4800_retention_ratio=0.4822038309091096；若 productive_h4800_group=0，则说明 source-preserving FU 仍只达到 weak-positive terminal source，而未过 retained-ratio gate。
- Terminal erosion taxonomy 覆盖率由 `v22_04_terminal_erosion_route.json` 决定；dominant class=OptimizerErosion，后续路线必须围绕该 class 的 train-only retained-target/source-observability 机制，而不是继续扩大 h800/h3200 source。
- D-CHE/D-FOU 的 efficiency 只能证明 official efficient carrier；KAN mapping matrix 仍需证明 source-channel writer 能承载 retained source，二者不能相互替代。
- D-RAT/D-RBF active micro-kernel benchmark 若仍显示 `official_fused_missing`，则只能作为 officialization progress/blocker，不允许进入 functional promotion。
- Precommit selector 在没有足够 productive h4800 正样本和 independent rerun 前保持失败/未训练状态，避免 retrospective selector 泄漏。

## Artifact Index

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/figures/late_rebound_vs_retention_scatter.svg | 1 | 329 | 36efb2b3d9cd658ad15d6152a9e3bff78e9562815b4c4f975d776e224c98c751 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/figures/source_trajectory_by_class.svg | 1 | 329 | e8c58c7cd36f70147002fc89de22de9836d7713ac30286e8f5c43ccd080889e8 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_command_journal.csv | 1 | 1703 | 5f25b36768e56ad6bcf5f924aa98d4b8b555760e759a0da8c0b20487b042bef6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_controls_attribution_matrix.csv | 1 | 1799 | eca5488a01c10bd9ba8d15657fb385ade7679658d9607abbc863ef0aca49a48d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_functional_raw_horizon_matrix.csv | 1 | 99499 | 2b3a6f2d028dd06850254a1824e70a38317bd06b6ce030fe1e74df5d9d07497f |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_mlp_source_retention_summary.csv | 1 | 5118 | df77096cddc95dcaadd2f3431a422cf382936ca8486f5f0ff3f99a7b7a474929 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_dynamics_classification.csv | 1 | 76 | 67bbafaf7416586ab19da8a81b62621bb09953b0c0a9c8957d67e920b9bf2a88 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_dynamics_matrix.csv | 1 | 5118 | df77096cddc95dcaadd2f3431a422cf382936ca8486f5f0ff3f99a7b7a474929 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_matrix.csv | 1 | 464633 | b1a28e13c39b3692e3a665e3b36419a943606f111ffe9b390d07eccf8b1b4be3 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_matrix_v22_04_d1_source_preserving_fu_oldshape_full_fc0.csv | 1 | 27369 | a7375ffb8c8f21a1316e4c1ce9cbe5a0f286e72d781c156bfa67b34d9c281049 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_matrix_v22_04_d1_source_preserving_fu_oldshape_full_fc1.csv | 1 | 27330 | faa7498a6f8deb1bc23d38b0b6c5aa9afd2e375650d9890119c408407e57b747 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_matrix_v22_04_d1_source_preserving_fu_oldshape_full_fc2.csv | 1 | 27076 | 795ce1f0b560c7aa06848b15890875f01a35a58d951e850d32b41d00d1d738a3 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_matrix_v22_04_d1_source_preserving_fu_oldshape_full_fc3.csv | 1 | 26253 | b4c9955e0632cbd65bfc807ab513f7e744647ed9eb91548c06b56bace9fe98bc |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_raw_traces.csv | 1 | 972665 | 618db181c2514c919480b9cb6700404754b1b0200042a4bbc8d78c6a04e26ab9 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_summary.csv | 1 | 5118 | df77096cddc95dcaadd2f3431a422cf382936ca8486f5f0ff3f99a7b7a474929 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_traces_v22_04_d1_source_preserving_fu_oldshape_full_fc0.csv | 1 | 248223 | a569ab0ca7b0773c47e866681ab3c5579523be72e903947a09522efd4da05a08 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_traces_v22_04_d1_source_preserving_fu_oldshape_full_fc1.csv | 1 | 248903 | d9fd73534154c0d8c8e4f77f2c8a804bab967ccb75ea51df51f153c89bd34f11 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_traces_v22_04_d1_source_preserving_fu_oldshape_full_fc2.csv | 1 | 245546 | c71a4217490cf9cdd3f2ecfd19cffb69a7f797a6859973ec6100dab646cc9130 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1_source_preserving_fu_oldshape_full/v21_01_source_retention_traces_v22_04_d1_source_preserving_fu_oldshape_full_fc3.csv | 1 | 238339 | ce65cc194f742d8fdc942a321fc74d2ca8ddf85a23c3b4c553e3dae7bd60f3ab |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/figures/late_rebound_vs_retention_scatter.svg | 1 | 321 | cf2349ec7482851eadb1dd23c6d17740eb800e0a18478a9eb2fa7ef5814e80ac |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/figures/source_trajectory_by_class.svg | 1 | 330 | 8f6ef2e220d1418e1ede7331101f5def315dab2aeadad04a0941550bda3b334a |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_command_journal.csv | 1 | 1702 | dec72b18412c6397b3e983b05a463c469f3798c934a84619b174b07ac02184b8 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_controls_attribution_matrix.csv | 1 | 1808 | 5aa157ac714c57d1d37b7485c3891986b7ad7d7abc3e0be3453edcec19f37b60 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_functional_raw_horizon_matrix.csv | 1 | 41587 | ac99c0adbd98bc109654cf37c12ceaa08deb4772de9bd49ae7cda8c2489e2807 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_mlp_source_retention_summary.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_dynamics_classification.csv | 1 | 76 | 53878945c34ec0a12b315d3c3fa94bb83f1548727e29ea38c80f9cdfe0b6d8cf |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_dynamics_matrix.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_matrix.csv | 1 | 232340 | 96fd0e558960ca3599089058e57916dcd5b7495152c0be664fe160bb23b4100c |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_oldshape_full_fc0.csv | 1 | 13569 | d69a52e660627b2c1a643bd33cd2202bfa41b5ea470fd4ea707babe2e0e16ad8 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_oldshape_full_fc1.csv | 1 | 12213 | abf409ec1f500ca0febf71a6ce8325999e2a6a27f6b9e9af65ec56a80003f331 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_oldshape_full_fc2.csv | 1 | 12934 | 2e45c76f8af294749c88204b4fac3cbaa8f84398c185b56997650e88068f452d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_oldshape_full_fc3.csv | 1 | 11400 | 88eef25a1867c952a515a7d7dfddd5880ecba80ab282832a9db065f0024e098d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_raw_traces.csv | 1 | 438474 | 774b148b6b83b7bf1fd5b7c04de21a998f6f0ba0ed2f9eac933012a60c054115 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_summary.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_oldshape_full_fc0.csv | 1 | 123905 | a6df5f640ec8fd24d45dc43e6294ed729beb0f0b090d27b058df1aaf4b7e154d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_oldshape_full_fc1.csv | 1 | 108718 | 9e9ae7e4e8021895418f05124c04a059c8cd7b4152cbbc19bcc874f975fad2af |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_oldshape_full_fc2.csv | 1 | 114783 | 270be21a507134e9431a27dbbef21bc69e10291b294b0be110d93422da9cd345 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_oldshape_full_fc3.csv | 1 | 99414 | 38b12e793f69d53490db76a13aafed669538edd08a342c29d3f3db53f97893b2 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/figures/late_rebound_vs_retention_scatter.svg | 1 | 321 | cf2349ec7482851eadb1dd23c6d17740eb800e0a18478a9eb2fa7ef5814e80ac |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/figures/source_trajectory_by_class.svg | 1 | 330 | 8f6ef2e220d1418e1ede7331101f5def315dab2aeadad04a0941550bda3b334a |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_command_journal.csv | 1 | 1702 | c503cb7d86df3059e3951dd4c366123db99b05f03f99e9d76a6cbc2de20c72be |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_controls_attribution_matrix.csv | 1 | 1808 | 5aa157ac714c57d1d37b7485c3891986b7ad7d7abc3e0be3453edcec19f37b60 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_functional_raw_horizon_matrix.csv | 1 | 41972 | 36277da1e4905876b33329f5bae4f46a43ee951da4edcf511a289bc3bfbef643 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_mlp_source_retention_summary.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_dynamics_classification.csv | 1 | 76 | 53878945c34ec0a12b315d3c3fa94bb83f1548727e29ea38c80f9cdfe0b6d8cf |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_dynamics_matrix.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_matrix.csv | 1 | 232725 | 7b430bc06da7c31672e76c0383621346f482e938e742ead809efdb8992301f43 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc0.csv | 1 | 13668 | c87fca82683a7c501cf02dbea5741ae79db60ea04c9d6f67b397b181cfffb061 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc1.csv | 1 | 12312 | c09c925cf07855dd823114b5eab19ae870462724114d8223b44412f22a1a8cdd |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc2.csv | 1 | 13027 | faa2292e50d3157afaa34e983cbc32d6d116a6f198207f171e987b0fb2ebabc6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc3.csv | 1 | 11494 | d7f99c0da51fe961bc4546641da4890269276607c550d50b0e9bcfe1e74023d1 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_raw_traces.csv | 1 | 443010 | dc6ad0c9f5b96474b5775652c4226544764ae33927501d226716d5bb2fee0212 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_summary.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc0.csv | 1 | 125081 | 6f1eb303e3bcd2692d26b57b4438e2084fb5693379b9bb13ea9e70cf4682bd14 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc1.csv | 1 | 109894 | d4c665c04cc0c63067831d4057b8d157202466d1fc882b08c25d2db1633ca547 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc2.csv | 1 | 115875 | a79ad4d43016f1b3b620193ab078ba4bc3f2f61f2b8606a1fce518538a8c47ac |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired2_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired2_oldshape_full_fc3.csv | 1 | 100506 | e4656f13e7d35f9ca2bf2272654c0822bfce2db797fdacafc120f66d55c16ac0 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/figures/late_rebound_vs_retention_scatter.svg | 1 | 327 | 01d9add24d910c0a82405ade1153634a0672fc1e7116782820d01480a309006c |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/figures/source_trajectory_by_class.svg | 1 | 327 | 267dc7fda05b204f4810775caae1f54cef0c4222cfb962e6e6deab5f96951e1b |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_command_journal.csv | 1 | 1702 | e0b216618736d455f9948b8e58e756a26dc00c6998ad816844cc1467f656128d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_controls_attribution_matrix.csv | 1 | 1808 | 5aa157ac714c57d1d37b7485c3891986b7ad7d7abc3e0be3453edcec19f37b60 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_functional_raw_horizon_matrix.csv | 1 | 47889 | 007c6d901303cc3f169d89468dafe17a23412c28f7069de6c68447158c4a87ca |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_mlp_source_retention_summary.csv | 1 | 2775 | f3b7860508e7319e44171fac3c25317327ac2fd5fd6e12df37e8e6f8e335408d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_dynamics_classification.csv | 1 | 76 | 53878945c34ec0a12b315d3c3fa94bb83f1548727e29ea38c80f9cdfe0b6d8cf |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_dynamics_matrix.csv | 1 | 2775 | f3b7860508e7319e44171fac3c25317327ac2fd5fd6e12df37e8e6f8e335408d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_matrix.csv | 1 | 239922 | 8e1a323733f72b2a4752c48d935d4855d5faffe74405fe5be3e3a98552782d99 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc0.csv | 1 | 15349 | 6c0456f45cfc69a14d8d32349ff9b641e18e5f240f9f2256715048cdfbecf6bb |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc1.csv | 1 | 13957 | 4557922e9cdaa0a9c908c53db624e36e5d98be7a23e3d61a069c125cccb2fd5d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc2.csv | 1 | 14205 | ea27f27872e5c5840f2ac920c489082f9f837f0228c256b9d270159faf30564d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc3.csv | 1 | 12907 | aa7983ce743c5004f0e602340c140c2f80d31a570833d4604b455a5c440fe746 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_raw_traces.csv | 1 | 478971 | 0ba74eb7394e4b1d380ef0b141c3887c16855ea2c4af460fc483773aacd5d209 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_summary.csv | 1 | 2775 | f3b7860508e7319e44171fac3c25317327ac2fd5fd6e12df37e8e6f8e335408d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc0.csv | 1 | 135080 | cbc153b365a8e3e294b8d71e71e979aa85a96835b7b8446ddf37e589aa6265ca |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc1.csv | 1 | 119758 | 007c923bf0b6a35216ccda9c2fc0996d787c0f04e8d0b69d8aad024f3eb29853 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc2.csv | 1 | 123000 | e61caab6c494809f0495259c526e6ebdd480fd6a3662f2d70bd2e4ff8b0c01cc |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired3_oldshape_full_fc3.csv | 1 | 109479 | f9762eaea15e85bf06bd25c6ac0d114cee5d276eb8f943848b453d73814ca682 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/figures/late_rebound_vs_retention_scatter.svg | 1 | 292 | 284ce4047aa8e8c1dab10224ae1a84519dbf8729262f1bb86271d03cdbb5e9e7 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/figures/source_trajectory_by_class.svg | 1 | 293 | 7a2e8b8b8ffa37260061cad6eb60b6bda65dd16de4926546377d03b00978a4f6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_command_journal.csv | 1 | 575 | 9f83e3c9e259f17289f79d4e9b92b2c46da4181215a87a3c08bd00132c157946 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_controls_attribution_matrix.csv | 1 | 852 | 974ff6d0172aa1ddd186112a1abd347b0a5edaa5784b525837cd7b68cb32465f |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_functional_raw_horizon_matrix.csv | 1 | 4504 | 4a33a0b4d3348216f3962844ca55b19ac70c9c4477c0003181dc014d5375525b |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_mlp_source_retention_summary.csv | 1 | 1148 | cbf2d3bbf5ff090952bc7c20e48895a2b64666879f99a87f925528bad4dfa27b |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_dynamics_classification.csv | 1 | 74 | 4efb85713c2a9c6873784a079e56323a0b09ad8c695457e28d32396153245b0d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_dynamics_matrix.csv | 1 | 1148 | cbf2d3bbf5ff090952bc7c20e48895a2b64666879f99a87f925528bad4dfa27b |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_retention_matrix.csv | 1 | 10560 | c248b3d476b40296d196ee7bb36cdc656a2d194686538ce6218180251fe44214 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired3_smoke_fc0.csv | 1 | 4504 | 4a33a0b4d3348216f3962844ca55b19ac70c9c4477c0003181dc014d5375525b |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_retention_raw_traces.csv | 1 | 11658 | 87c0d7e9a26da300a43ce67140bfcd73e485b16dbf086307b317cb221af19dc6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_retention_summary.csv | 1 | 1148 | cbf2d3bbf5ff090952bc7c20e48895a2b64666879f99a87f925528bad4dfa27b |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired3_smoke/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired3_smoke_fc0.csv | 1 | 11658 | 87c0d7e9a26da300a43ce67140bfcd73e485b16dbf086307b317cb221af19dc6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/figures/late_rebound_vs_retention_scatter.svg | 1 | 321 | cf2349ec7482851eadb1dd23c6d17740eb800e0a18478a9eb2fa7ef5814e80ac |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/figures/source_trajectory_by_class.svg | 1 | 330 | 8f6ef2e220d1418e1ede7331101f5def315dab2aeadad04a0941550bda3b334a |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_command_journal.csv | 1 | 1702 | 4cdbe94c66771f9b0a12b2509811be0cf8cc0173f313f6d620df95b38f7d3154 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_controls_attribution_matrix.csv | 1 | 1808 | 5aa157ac714c57d1d37b7485c3891986b7ad7d7abc3e0be3453edcec19f37b60 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_functional_raw_horizon_matrix.csv | 1 | 41912 | f9d6a612562339a881a1bf2c9714ac40fc7839faabdfcf669516df38d91d5923 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_mlp_source_retention_summary.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_dynamics_classification.csv | 1 | 76 | 53878945c34ec0a12b315d3c3fa94bb83f1548727e29ea38c80f9cdfe0b6d8cf |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_dynamics_matrix.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_matrix.csv | 1 | 232665 | cf76048d01c8e77ca94355be954b7c6f7221adf93c56d5ce1a5153f8e503dc74 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired_oldshape_full_fc0.csv | 1 | 13654 | ccaa30f0160778081409606b9d0d373f779b84692e0dd431199e775a73b01588 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired_oldshape_full_fc1.csv | 1 | 12297 | 0bac72a57491cf3c7f86c4dd31b8ca84af015497eec0ac9021c1870528d09199 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired_oldshape_full_fc2.csv | 1 | 13010 | 6a8297f456ba6aca2e0a3920246b9326c6845a9cefaf63fef5f877fed30e6060 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_matrix_v22_04_d1a_lambda_strength_wired_oldshape_full_fc3.csv | 1 | 11480 | f3a96d343e77b0240cd6c5cf92525cdd2c4ad95c05fd839fbec20fc8ff9e4dc8 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_raw_traces.csv | 1 | 442362 | 4bbc54f1da8a65ad3096947c212b1355c33a6c9878dd69cc6645c3633f37417d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_summary.csv | 1 | 2621 | aac10e3eb0e859538c17ee69bdd60f59a00d96eb076aeacfabda6adec1dd6b04 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired_oldshape_full_fc0.csv | 1 | 124913 | d5c4de863fb348dc26c8254be1b411c1bfcbc499ebdcad456c0e0d707594efee |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired_oldshape_full_fc1.csv | 1 | 109726 | fe01fa099aa6968b01d3063747340c002e27d03a62b6b26b5603b053f498fecc |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired_oldshape_full_fc2.csv | 1 | 115719 | ec9158e6fe4bee5ec7c1bdf5c8faeeb8199ea30a2bd83145eef2475f8dfb66e9 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_oldshape_full/v21_01_source_retention_traces_v22_04_d1a_lambda_strength_wired_oldshape_full_fc3.csv | 1 | 100350 | 5b961d8eb5d901827aad488a9fb34f311015199ce9930d8871206a51e506664d |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_smoke/figures/late_rebound_vs_retention_scatter.svg | 1 | 292 | 284ce4047aa8e8c1dab10224ae1a84519dbf8729262f1bb86271d03cdbb5e9e7 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_smoke/figures/source_trajectory_by_class.svg | 1 | 293 | 7a2e8b8b8ffa37260061cad6eb60b6bda65dd16de4926546377d03b00978a4f6 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_smoke/v21_01_command_journal.csv | 1 | 575 | 0f776087cf6bf73137cdb8d48e7c986080eefb1dbf58379e89212850bb01f229 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_smoke/v21_01_controls_attribution_matrix.csv | 1 | 852 | 974ff6d0172aa1ddd186112a1abd347b0a5edaa5784b525837cd7b68cb32465f |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_smoke/v21_01_functional_raw_horizon_matrix.csv | 1 | 4445 | 84e52ece4f721405197efc665ecefffc97ee3808ee0a6990046308a67ebb0b66 |
| results/v22_04_terminal_source_preservation_diffeomorphic_fu_basis_efficiency/official_v22_04/continuation_d1a_lambda_strength_wired_smoke/v21_01_kan_source_retention_summary.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |

_仅显示前 120 / 892 rows；完整 CSV 见 artifact。_
