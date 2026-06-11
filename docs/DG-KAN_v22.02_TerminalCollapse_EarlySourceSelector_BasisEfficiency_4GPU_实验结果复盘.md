# DG-KAN v22.02 TerminalCollapse EarlySourceSelector BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-05 09:29:45 +0800

## Route

- route: `R4-TerminalCollapseNoH4800`
- S0.9 pass: 1
- D-CHE/D-FOU full-loop official closure: 1 / 1
- D-RAT/D-RBF: ForwardKernelBlocked / MaterializationBlocked
- functional early / h3200 / h4800: 27 / 27 / 0
- terminal collapse groups: 12
- SelectorRoute: NoH4800LabelNoSelectorCanPass
- TargetRoute: TargetObservableNoRetention
- KANRoute: KANSourceBankMismatch
- promotion_allowed: 0
- required artifact missing count: 0

## Part 1 Code Audit

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| import_closure | 1 | clean compileall/import/source/manifest | 1 |  |
| metrics | 1 | linec/source/debt/selector/update/efficiency tests | 47/47 |  |
| mechanisms | 1 | v22 terminal/source mechanism contracts | 18/18 |  |
| kernels | 1 | kernel status consistency no false official claim | 1 |  |

## Part 2 Basis Efficiency

| carrier | rows | E1_pass_rows | S1_pass_rows | S1_pass_batch_sizes | S1_pass_variants | best_forward_ratio | best_step_ratio | full_loop_official_closure | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 12 | 9 | 6 | 3 | 2 | 0.9305108277155166 | 0.46682782835473463 | 1 | OfficialEfficientCarrier |
| D-FOU | 12 | 8 | 3 | 3 | 1 | 1.0087315984225318 | 0.5089279836996342 | 1 | OfficialEfficientCarrier |

| carrier | profile_rows | near_E1_rows | best_forward_ratio | best_step_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 2 | 0 | 9.945103242391575 | 1.1491903929619172 | ForwardKernelBlocked | forward_ratio;official_fused_missing;materialization;functional_runner_kernel_mismatch;forward_ratio;step_ratio;official_fused_missing;materialization;functional_runner_kernel_mismatch |
| D-RBF | 2 | 0 | 8.816428428771149 | 1.6501664590060272 | MaterializationBlocked | forward_ratio;step_ratio;official_fused_missing;materialization;functional_runner_kernel_mismatch |

## Part 3 Functional Update

| carrier | variant | v22_id | rows | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | R0-current | CTRL-AdamW | 9 | 0.0 | 0.0 | -0.0035047531127929688 | -0.3762444787555271 | -0.6632115377320184 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP | R0-current | CTRL-NoOpMatchedOverhead | 9 | -1.537603822019365 | -1.3963266611099243 | -1.1151884065734015 | -1.1113447348276775 | -1.1939316656854417 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP | R0-current | CTRL-RandomMatchedNorm | 9 | -1.5729280511538188 | -1.4318200614717271 | -1.151018136077457 | -1.1466877725389268 | -1.2288817365964253 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP | R0-current | CTRL-SGD | 9 | -1.4659851325882807 | -1.1357120540406969 | -0.6329910821384854 | -0.2397618293762207 | -0.07248895698123509 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP | R0-current | MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target | 9 | 0.03676839007271661 | 0.2345359656545851 | 0.505232486459944 | 0.5088075134489272 | 0.4255666799015469 | 0.3312460548347897 | 0.21527639362547132 | 0.13427998622258505 | 1 | 1 | 0 | 0 |  |
| MLP | R0-current | MLP-F125-early100-h800-source-slow-ema-source-bank-target | 9 | 0.027575214703877766 | 0.2277486821015676 | 0.49210549063152736 | 0.495961993932724 | 0.4133879608578152 | 0.3192800283432007 | 0.2035752170615726 | 0.12227069007025824 | 1 | 1 | 0 | 0 |  |
| MLP | R0-current | MLP-F126-early100-h800-source-slow-ema-dual-target-guard | 9 | 0.029030117723676894 | 0.23846699794133505 | 0.5110694964726766 | 0.5146651996506585 | 0.4325850009918213 | 0.33855552474657696 | 0.2227248748143514 | 0.14141722520192465 | 1 | 1 | 0 | 0 |  |

## Artifact Index

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| /home/chengshun.wang/DG-LCA/v22_02_f106_f144_oldshape_full_audit.csv | 1 | 17460 | 05554a059bfee61088c63ed65ae5aeaa523f11fd166d9233007a3a6a53c46751 |
| /home/chengshun.wang/DG-LCA/v22_02_f106_f144_oldshape_full_route.csv | 1 | 251 | dd2aae5fa56876d445aca4de68a8b071f1c791a0f6f8b273c563fcdba63f3dcb |
| /home/chengshun.wang/DG-LCA/v22_02_source_retention_matrix.csv | 1 | 300153 | 3c44ad4854a12c9b2d5f292976357e9de1c5ac98c3ff166749ed499cc4326309 |
| /home/chengshun.wang/DG-LCA/v22_02_precommit_selector_matrix.csv | 1 | 960 | 2839fe7e4cad56cf9da7251df007f7db9c02457f9ee4854bdf4d6688c3aec1c1 |
| /home/chengshun.wang/DG-LCA/v22_02_source_channel_target_matrix.csv | 1 | 11905 | 941d32452b5b615d3cb42b909cbc8435b0d5c5d033a834379f314cfcd0ec5da6 |
| /home/chengshun.wang/DG-LCA/v22_02_kan_source_bank_matrix.csv | 1 | 10129 | 85fb8db69c371b2fdccf839ad12fdf0f4f3ad40f339d8de643bd6c4bc76365d3 |
| /home/chengshun.wang/DG-LCA/continuation_f142_f144_topk_support_oldshape_full/v22_02_source_chain_summary.csv | 1 | 12654 | 95a7bb0c1388d8eec89b7985cc0edf86c09b81948e2135ea698b9245bfa3642c |
| /home/chengshun.wang/DG-LCA/continuation_f142_f144_topk_support_oldshape_full/v21_01_source_retention_raw_traces.csv | 1 | 540335 | 4ee9e54bef3da3397e08b923e56f84f8563eaeb502bb933582ab8710e264510f |

## Fresh Rerun Provenance

- `fresh_terminal_pool_oldshape`: 81-row mixed-candidate smoke/localization pool. It was superseded for scientific causal claims because mixed candidates change job-index/train-seed assignment.
- `fresh_terminal_single_oldshape`: 225-row single-candidate old-shape localization baseline, with one candidate plus matched controls per run-label and h4000 recorded.
- Baseline single-candidate candidates: `MLP-F53`, `MLP-F77`, `MLP-F78`, `MLP-F86`, `MLP-F87`; controls: `CTRL-SGD`, `CTRL-AdamW`, `CTRL-RandomMatchedNorm`, `CTRL-NoOpMatchedOverhead`.

## Continuation F88-F144 Source-State / Terminal Guard / Signal Target / Source Shape / Weak-Row Rescue / Top-K Support

- This section is rendered only from fresh continuation audit CSVs; no values are hand-promoted.
- Best near-miss in this family remains `MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard`, h4800_retention_ratio=0.4905266741203586, below the v22.02 productive gate.
- F120/F121/F122 terminal projected optimizer, projected blend, and antiwashout did not improve over F118; F123 h4000 checkpoint reentry also remained below the productive h4800 gate.
- F124/F125/F126 tested the plan-recommended train-only signal-channel/reservoir/source-bank target reset. The exact MLP readout target path was enabled, but the target gates did not produce retained h4800 source and all three worsened relative to F118.
- F127/F128/F129 test a source-state adaptive terminal family: train-split/corrupt-label gated adaptive raw, sparse source-axis, and ratio-preserving source guard. They are evaluated with the same v22.02 source-chain gate.
- F130/F131/F132 test the next plan-recommended retained-target reset: source-projected B3-null consensus, noise-orthogonal B3-null consensus, and easy-margin B3-null consensus. They are train-only terminal targets wrapped around the F118 early100 h800 source slow-EMA path.
- F133/F134/F135 test a terminal reject-rescue family: source-axis micro rescue, slow-EMA micro rescue, and hold-then-source rescue. The rescue is triggered only from train-stream terminal reject diagnostics and is still evaluated by the unchanged productive h4800 gate.
- F136/F137/F138 test a train-only source-shape preservation family: pure midlate source-shape preserve, preserve plus raw guard, and preserve plus terminal clamp. They are evaluated as a falsification of source-shape drift rather than as promotion candidates.
- F139/F140/F141 test terminal reject weak-row rescue: raw rescue, raw+source hybrid rescue, and late-only raw rescue. They are triggered by train-only source-state reject diagnostics and keep the unchanged productive h4800 gate.
- F142/F143/F144 test the FU-H6 recommended top-k source-support variant: sparse support guard, top-k raw hybrid guard, and debt-capped top-k guard. They protect only train-stream source-state support rather than the full source vector and keep the unchanged productive h4800 gate.
| source_chain_rows | source_chain_groups | candidate_groups | candidate_early_chain | candidate_continuous_h3200 | candidate_h4800 | promotion_allowed | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 972 | 108 | 36 | 27 | 27 | 0 | 0 | R4-TerminalCollapseNoH4800 |

### Continuation Ledger

| continuation | source_dir | candidates | early | continuous_h3200 | h4800 | best_v22_id | best_h800 | best_h3200 | best_h4000 | best_h4800 | best_h4800_retention_ratio | row_h4800_positive_max | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F88-F94 source-state oldshape full | continuation_f88_f94_source_state_oldshape_full | 4 | 0 | 0 | 0 | MLP-F88-h800-source-slow-ema-retention | 0.2556371953752305 | 0.0943417416678534 | -0.0068413813908894 | -0.0752062731319003 | 0.0 | 5 | NoH4800Candidate |
| F106-F107 early100 source-state combined full | continuation_f106_f107_early_source_state_oldshape_full | 2 | 0 | 0 | 0 | MLP-F106-early100-h800-source-slow-ema-retention | 0.5031206475363837 | 0.3107297685411241 | 0.2132231593132019 | 0.1428640021218193 | 0.4597692805313301 | 8 | NoH4800Candidate |
| F106 single early100 slow-EMA full | continuation_f106_single_early_source_state_oldshape_full | 1 | 1 | 1 | 0 | MLP-F106-early100-h800-source-slow-ema-retention | 0.5325850115882026 | 0.3210880822605557 | 0.2249905467033386 | 0.1525942418310377 | 0.4752410639371254 | 8 | NoH4800Candidate |
| F115 single early100 slow-EMA strong full | continuation_f115_single_early100_slowema_strong_oldshape_full | 1 | 1 | 1 | 0 | MLP-F115-early100-h800-source-slow-ema-strong-retention | 0.5324831472502815 | 0.3210328287548489 | 0.2249501678678724 | 0.1525664197074042 | 0.4752361940650902 | 8 | NoH4800Candidate |
| F117 single terminal source guard full | continuation_f117_single_terminal_guard_oldshape_full | 1 | 1 | 1 | 0 | MLP-F117-early100-h800-source-slow-ema-terminal-guard | 0.5298810799916586 | 0.3183568881617652 | 0.2222588625219133 | 0.1499197847313351 | 0.4709173581793432 | 8 | NoH4800Candidate |
| F118 single terminal raw guard full | continuation_f118_single_terminal_raw_guard_oldshape_full | 1 | 1 | 1 | 0 | MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | 0.5423558387491438 | 0.3307922316922081 | 0.2346738775571187 | 0.1622624132368299 | 0.4905266741203586 | 8 | NoH4800Candidate |
| F119 single terminal raw guard strong full | continuation_f119_single_terminal_raw_guard_strong_oldshape_full | 1 | 1 | 1 | 0 | MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong | 0.5348958770434061 | 0.3233591185675727 | 0.2272418571843041 | 0.1548460755083296 | 0.4788671993982172 | 8 | NoH4800Candidate |
| F120 terminal projected optimizer full | continuation_f120_terminal_projected_optimizer_oldshape_full | 1 | 1 | 1 | 0 | MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer | 0.5351371202203963 | 0.323529389169481 | 0.2273986405796474 | 0.1549697683917151 | 0.4789974993911114 | 8 | NoH4800Candidate |
| F121 terminal projected blend full | continuation_f121_terminal_projected_blend_oldshape_full | 1 | 1 | 1 | 0 | MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend | 0.5383261011706458 | 0.3266988396644592 | 0.2305566204918755 | 0.1580851872762044 | 0.4838865893694881 | 8 | NoH4800Candidate |
| F122 terminal antiwashout full | continuation_f122_terminal_antiwashout_oldshape_full | 1 | 1 | 1 | 0 | MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout | 0.5379865500662062 | 0.3264219197962019 | 0.2302946978145175 | 0.1578810877270168 | 0.4836718313083516 | 8 | NoH4800Candidate |
| F123 h4000 checkpoint reentry full | continuation_f123_h4000_reentry_oldshape_full | 1 | 1 | 1 | 0 | MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry | 0.5356751812828912 | 0.3240864939159817 | 0.2279582950803968 | 0.1555491255389319 | 0.4799617647110513 | 8 | NoH4800Candidate |
| F124-F126 signal/reservoir target reset full | continuation_f124_f126_signal_target_oldshape_full | 3 | 3 | 3 | 0 | MLP-F126-early100-h800-source-slow-ema-dual-target-guard | 0.5110694964726766 | 0.3385555247465769 | 0.2227248748143514 | 0.1414172252019246 | 0.4177076280405744 | 9 | NoH4800Candidate |
| F127-F129 source-state adaptive terminal full | continuation_f127_f129_source_state_adaptive_terminal_oldshape_full | 3 | 3 | 3 | 0 | MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw | 0.5026405254999796 | 0.3297484748893314 | 0.2137181427743699 | 0.1324055559105343 | 0.4015350062042641 | 9 | NoH4800Candidate |
| F130-F132 terminal target reset oldshape full | continuation_f130_f132_terminal_target_reset_oldshape_full | 3 | 3 | 3 | 0 | MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target | 0.5126018656624688 | 0.3397622671392228 | 0.2237474487887488 | 0.142462584707472 | 0.4193007831828947 | 9 | NoH4800Candidate |
| F133-F135 terminal reject rescue oldshape full | continuation_f133_f135_terminal_reject_rescue_oldshape_full | 3 | 3 | 3 | 0 | MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue | 0.5126342442300584 | 0.3398546510272556 | 0.2238691714074876 | 0.1426512897014618 | 0.4197420552294324 | 9 | NoH4800Candidate |
| F136-F138 source-shape preserve oldshape full | continuation_f136_f138_shape_preserve_oldshape_full | 3 | 0 | 0 | 0 | MLP-F136-early100-h800-source-slow-ema-shape-preserve | -1.082649701171451 | -1.1960659821828206 | -1.2939708299107022 | -1.357645571231842 |  | 0 | NoH4800Candidate |
| F139-F141 terminal reject weak-row rescue oldshape full | continuation_f139_f141_terminal_reject_weakrow_oldshape_full | 3 | 3 | 3 | 0 | MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue | 0.5133888986375597 | 0.3405913147661421 | 0.2246015701029035 | 0.1433822446399265 | 0.4209803316281745 | 9 | NoH4800Candidate |
| F142-F144 top-k source-support oldshape full | continuation_f142_f144_topk_support_oldshape_full | 3 | 3 | 3 | 0 | MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | 0.5111051168706682 | 0.338267895910475 | 0.2222603228357103 | 0.1409425006972419 | 0.4166594063497629 | 9 | NoH4800Candidate |

### Best Candidate Ranking

| continuation | v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F118 single terminal raw guard full | MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | 0.0492262641588846 | 0.2508025036917792 | 0.5423558387491438 | 0.5321504010094537 | 0.4195122321446736 | 0.3307922316922081 | 0.2346738775571187 | 0.1622624132368299 | 0.4905266741203586 | 1 | 1 | 0 | 8 |  |
| F121 terminal projected blend full | MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend | 0.0498369468583001 | 0.2403660284148322 | 0.5383261011706458 | 0.5280945863988664 | 0.415436887078815 | 0.3266988396644592 | 0.2305566204918755 | 0.1580851872762044 | 0.4838865893694881 | 1 | 1 | 0 | 8 |  |
| F122 terminal antiwashout full | MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout | 0.0435280601183573 | 0.2443078756332397 | 0.5379865500662062 | 0.527775956524743 | 0.4151379267374674 | 0.3264219197962019 | 0.2302946978145175 | 0.1578810877270168 | 0.4836718313083516 | 1 | 1 | 0 | 8 |  |
| F123 h4000 checkpoint reentry full | MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry | 0.0596391028828091 | 0.2510800361633301 | 0.5356751812828912 | 0.5254587067498101 | 0.4128122263484531 | 0.3240864939159817 | 0.2279582950803968 | 0.1555491255389319 | 0.4799617647110513 | 1 | 1 | 0 | 8 |  |
| F120 terminal projected optimizer full | MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer | 0.0533016721407572 | 0.2423147625393337 | 0.5351371202203963 | 0.5249133507410685 | 0.412259621752633 | 0.323529389169481 | 0.2273986405796474 | 0.1549697683917151 | 0.4789974993911114 | 1 | 1 | 0 | 8 |  |
| F119 single terminal raw guard strong full | MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong | 0.0345257321993509 | 0.2347747186819712 | 0.5348958770434061 | 0.5246972176763747 | 0.4120653072992961 | 0.3233591185675727 | 0.2272418571843041 | 0.1548460755083296 | 0.4788671993982172 | 1 | 1 | 0 | 8 |  |
| F106 single early100 slow-EMA full | MLP-F106-early100-h800-source-slow-ema-retention | 0.0407662987709045 | 0.2366060548358493 | 0.5325850115882026 | 0.5223965711063809 | 0.409780740737915 | 0.3210880822605557 | 0.2249905467033386 | 0.1525942418310377 | 0.4752410639371254 | 1 | 1 | 0 | 8 |  |
| F115 single early100 slow-EMA strong full | MLP-F115-early100-h800-source-slow-ema-strong-retention | 0.0407662987709045 | 0.2366316980785793 | 0.5324831472502815 | 0.5223103364308676 | 0.4097101357248094 | 0.3210328287548489 | 0.2249501678678724 | 0.1525664197074042 | 0.4752361940650902 | 1 | 1 | 0 | 8 |  |
| F117 single terminal source guard full | MLP-F117-early100-h800-source-slow-ema-terminal-guard | 0.0609749621815151 | 0.2458254165119595 | 0.5298810799916586 | 0.5196812848250071 | 0.4070614907476637 | 0.3183568881617652 | 0.2222588625219133 | 0.1499197847313351 | 0.4709173581793432 | 1 | 1 | 0 | 8 |  |
| F106-F107 early100 source-state combined full | MLP-F106-early100-h800-source-slow-ema-retention | 0.0039654705259535 | 0.2313903371493021 | 0.5031206475363837 | 0.5223599672317505 | 0.4309421115451389 | 0.3107297685411241 | 0.2132231593132019 | 0.1428640021218193 | 0.4597692805313301 | 0 | 0 | 0 | 8 | EarlySourceChainMissing;ContinuousH3200Missing |
| F139-F141 terminal reject weak-row rescue oldshape full | MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue | 0.0333744022581312 | 0.2395138409402635 | 0.5133888986375597 | 0.5172538956006368 | 0.4346884091695149 | 0.3405913147661421 | 0.2246015701029035 | 0.1433822446399265 | 0.4209803316281745 | 1 | 1 | 0 | 9 |  |
| F133-F135 terminal reject rescue oldshape full | MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue | 0.037391013569302 | 0.2404746148321363 | 0.5126342442300584 | 0.5165025856759813 | 0.43394402662913 | 0.3398546510272556 | 0.2238691714074876 | 0.1426512897014618 | 0.4197420552294324 | 1 | 1 | 0 | 7 |  |
| F130-F132 terminal target reset oldshape full | MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target | 0.0258794360690646 | 0.2334463198979695 | 0.5126018656624688 | 0.5164534615145789 | 0.4338747163613637 | 0.3397622671392228 | 0.2237474487887488 | 0.142462584707472 | 0.4193007831828947 | 1 | 1 | 0 | 8 |  |
| F139-F141 terminal reject weak-row rescue oldshape full | MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue | 0.0334433913230896 | 0.2413532733917236 | 0.5124931467903985 | 0.5163459678490957 | 0.4337684412797292 | 0.3396590550740559 | 0.2236685984664493 | 0.1424066258801354 | 0.4192634459549045 | 1 | 1 | 0 | 8 |  |
| F124-F126 signal/reservoir target reset full | MLP-F126-early100-h800-source-slow-ema-dual-target-guard | 0.0290301177236768 | 0.238466997941335 | 0.5110694964726766 | 0.5146651996506585 | 0.4325850009918213 | 0.3385555247465769 | 0.2227248748143514 | 0.1414172252019246 | 0.4177076280405744 | 1 | 1 | 0 | 9 |  |
| F142-F144 top-k source-support oldshape full | MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | 0.0339639981587727 | 0.2391603489716847 | 0.5111051168706682 | 0.5149583948983086 | 0.4323798153135512 | 0.338267895910475 | 0.2222603228357103 | 0.1409425006972419 | 0.4166594063497629 | 1 | 1 | 0 | 9 |  |
| F142-F144 top-k source-support oldshape full | MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | 0.0316448145442538 | 0.2345340914196438 | 0.5056587556997935 | 0.5095196564992269 | 0.4269552793767717 | 0.3328632546795739 | 0.2168703840838538 | 0.1355907983250088 | 0.407346850151824 | 1 | 1 | 0 | 8 |  |
| F130-F132 terminal target reset oldshape full | MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target | 0.0293125576443142 | 0.2294596963458591 | 0.5048207574420505 | 0.5086872180302938 | 0.4261287086539798 | 0.332035524977578 | 0.216049227449629 | 0.1347834401660495 | 0.405930781578722 | 1 | 1 | 0 | 9 |  |
| F124-F126 signal/reservoir target reset full | MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target | 0.0367683900727166 | 0.2345359656545851 | 0.505232486459944 | 0.5088075134489272 | 0.4255666799015469 | 0.3312460548347897 | 0.2152763936254713 | 0.134279986222585 | 0.4053783713425892 | 1 | 1 | 0 | 8 |  |
| F133-F135 terminal reject rescue oldshape full | MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue | 0.0230470233493381 | 0.2293337119950188 | 0.5044616527027554 | 0.5083251926634047 | 0.4257562922106849 | 0.3316524492369758 | 0.2156531843874189 | 0.1343820591767629 | 0.4051894068200981 | 1 | 1 | 0 | 8 |  |
| F127-F129 source-state adaptive terminal full | MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw | 0.0244074794981214 | 0.2260927557945251 | 0.5026405254999796 | 0.5064758592181735 | 0.4238789081573486 | 0.3297484748893314 | 0.2137181427743699 | 0.1324055559105343 | 0.4015350062042641 | 1 | 1 | 0 | 8 |  |
| F127-F129 source-state adaptive terminal full | MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve | 0.0341867208480834 | 0.2309097879462772 | 0.5018337799443139 | 0.5056931144661374 | 0.4231271511978573 | 0.3290293349160088 | 0.2130403055085076 | 0.1317636966705322 | 0.400461851537242 | 1 | 1 | 0 | 9 |  |
| F142-F144 top-k source-support oldshape full | MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | 0.0209643377198113 | 0.2270598146650526 | 0.4965431590874989 | 0.5003942714797126 | 0.4178156620926327 | 0.3237038155396779 | 0.2077026830779181 | 0.1263947089513143 | 0.3904640689532481 | 1 | 1 | 0 | 9 |  |
| F139-F141 terminal reject weak-row rescue oldshape full | MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue | 0.0232686665323045 | 0.2262647019492255 | 0.4956471456421746 | 0.4995193547672695 | 0.4169597261481815 | 0.3228688836097717 | 0.2068851954407162 | 0.1256653997633192 | 0.3892149604456832 | 1 | 1 | 0 | 9 |  |
| F130-F132 terminal target reset oldshape full | MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target | 0.0197222961319817 | 0.2256713642014397 | 0.4940762917200724 | 0.4979171024428473 | 0.4153325623936123 | 0.321227232615153 | 0.2052310076024797 | 0.1241881317562527 | 0.3866052412344397 | 1 | 1 | 0 | 9 |  |
| F133-F135 terminal reject rescue oldshape full | MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue | 0.0313817395104302 | 0.2263318300247192 | 0.4936133391327328 | 0.4974823461638556 | 0.4149239924218919 | 0.320832351843516 | 0.2048467000325521 | 0.1235881447792053 | 0.3852109803424209 | 1 | 1 | 0 | 9 |  |
| F124-F126 signal/reservoir target reset full | MLP-F125-early100-h800-source-slow-ema-source-bank-target | 0.0275752147038777 | 0.2277486821015676 | 0.4921054906315273 | 0.495961993932724 | 0.4133879608578152 | 0.3192800283432007 | 0.2035752170615726 | 0.1222706900702582 | 0.3829575269857686 | 1 | 1 | 0 | 9 |  |
| F106-F107 early100 source-state combined full | MLP-F107-early100-h800-readout-channel-retention | -0.0115754935476515 | 0.2040987511475881 | 0.4616460667716132 | 0.4808787902196248 | 0.3894584245151943 | 0.2692307763629489 | 0.1717093918058607 | 0.1013376547230614 | 0.3763969932859702 | 0 | 0 | 0 | 7 | EarlySourceChainMissing;ContinuousH3200Missing |
| F127-F129 source-state adaptive terminal full | MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source | 0.0130445692274305 | 0.2175094352828132 | 0.4871440662278069 | 0.4909894598854912 | 0.4084062874317169 | 0.3142895963456895 | 0.1982795099417368 | 0.1169712808397081 | 0.3721767509957616 | 1 | 1 | 0 | 9 |  |
| F88-F94 source-state oldshape full | MLP-F88-h800-source-slow-ema-retention | 0.0458766222000122 | -0.0006026228268941 | 0.2556371953752305 | 0.3166095813115437 | 0.223662429385715 | 0.0943417416678534 | -0.0068413813908894 | -0.0752062731319003 | 0.0 | 0 | 0 | 0 | 5 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F94 source-state oldshape full | MLP-F89-h800-readout-channel-retention | 0.0487066970931159 | -0.0152870151731703 | 0.238328390651279 | 0.2993032071325514 | 0.2063628567589653 | 0.0770309103859795 | -0.0241510007116529 | -0.092516541481018 | 0.0 | 0 | 0 | 0 | 3 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F94 source-state oldshape full | MLP-F93-h800-readout-channel-alt25-strong-retention | 0.0371184349060058 | 0.0008842547734578 | 0.2498369945420159 | 0.310828771856096 | 0.2179053823153177 | 0.0885904828707377 | -0.0125796794891357 | -0.0809141331248813 | 0.0 | 0 | 0 | 0 | 3 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F94 source-state oldshape full | MLP-F94-h1600-source-checkpoint-reentry | 0.005660679605272 | -0.0425370335578918 | 0.209342082341512 | 0.270304434829288 | 0.1773516800668504 | 0.048010508219401 | -0.0531877014372083 | -0.1215635736783345 | 0.0 | 0 | 0 | 0 | 4 | EarlySourceChainMissing;ContinuousH3200Missing |
| F136-F138 source-shape preserve oldshape full | MLP-F136-early100-h800-source-slow-ema-shape-preserve | -1.5243507424990337 | -1.3746363719304402 | -1.082649701171451 | -1.0579475561777751 | -1.120796846018897 | -1.1960659821828206 | -1.2939708299107022 | -1.357645571231842 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F136-F138 source-shape preserve oldshape full | MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard | -1.5489292873276606 | -1.3992769055896337 | -1.107371773984697 | -1.0831479496426053 | -1.1465456287066145 | -1.2223913669586182 | -1.3209153877364264 | -1.385283715195126 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F136-F138 source-shape preserve oldshape full | MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp | -1.5143434140417311 | -1.3658073080910578 | -1.0752617981698778 | -1.0532494915856256 | -1.1183765000767178 | -1.1955820719401042 | -1.2951786054505243 | -1.3604410820537145 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |

### Old-shape full audit

| continuation | source_dir | v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F88-F94 source-state oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full | MLP-F88-h800-source-slow-ema-retention | 0.0458766222000122 | -0.0006026228268941 | 0.2556371953752305 | 0.3166095813115437 | 0.223662429385715 | 0.0943417416678534 | -0.0068413813908894 | -0.0752062731319003 | 0.0 | 0 | 0 | 0 | 5 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F94 source-state oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full | MLP-F89-h800-readout-channel-retention | 0.0487066970931159 | -0.0152870151731703 | 0.238328390651279 | 0.2993032071325514 | 0.2063628567589653 | 0.0770309103859795 | -0.0241510007116529 | -0.092516541481018 | 0.0 | 0 | 0 | 0 | 3 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F94 source-state oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full | MLP-F93-h800-readout-channel-alt25-strong-retention | 0.0371184349060058 | 0.0008842547734578 | 0.2498369945420159 | 0.310828771856096 | 0.2179053823153177 | 0.0885904828707377 | -0.0125796794891357 | -0.0809141331248813 | 0.0 | 0 | 0 | 0 | 3 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F94 source-state oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f88_f94_source_state_oldshape_full | MLP-F94-h1600-source-checkpoint-reentry | 0.005660679605272 | -0.0425370335578918 | 0.209342082341512 | 0.270304434829288 | 0.1773516800668504 | 0.048010508219401 | -0.0531877014372083 | -0.1215635736783345 | 0.0 | 0 | 0 | 0 | 4 | EarlySourceChainMissing;ContinuousH3200Missing |
| F106-F107 early100 source-state combined full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_f107_early_source_state_oldshape_full | MLP-F106-early100-h800-source-slow-ema-retention | 0.0039654705259535 | 0.2313903371493021 | 0.5031206475363837 | 0.5223599672317505 | 0.4309421115451389 | 0.3107297685411241 | 0.2132231593132019 | 0.1428640021218193 | 0.4597692805313301 | 0 | 0 | 0 | 8 | EarlySourceChainMissing;ContinuousH3200Missing |
| F106-F107 early100 source-state combined full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_f107_early_source_state_oldshape_full | MLP-F107-early100-h800-readout-channel-retention | -0.0115754935476515 | 0.2040987511475881 | 0.4616460667716132 | 0.4808787902196248 | 0.3894584245151943 | 0.2692307763629489 | 0.1717093918058607 | 0.1013376547230614 | 0.3763969932859702 | 0 | 0 | 0 | 7 | EarlySourceChainMissing;ContinuousH3200Missing |
| F106 single early100 slow-EMA full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f106_single_early_source_state_oldshape_full | MLP-F106-early100-h800-source-slow-ema-retention | 0.0407662987709045 | 0.2366060548358493 | 0.5325850115882026 | 0.5223965711063809 | 0.409780740737915 | 0.3210880822605557 | 0.2249905467033386 | 0.1525942418310377 | 0.4752410639371254 | 1 | 1 | 0 | 8 |  |
| F115 single early100 slow-EMA strong full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f115_single_early100_slowema_strong_oldshape_full | MLP-F115-early100-h800-source-slow-ema-strong-retention | 0.0407662987709045 | 0.2366316980785793 | 0.5324831472502815 | 0.5223103364308676 | 0.4097101357248094 | 0.3210328287548489 | 0.2249501678678724 | 0.1525664197074042 | 0.4752361940650902 | 1 | 1 | 0 | 8 |  |
| F117 single terminal source guard full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f117_single_terminal_guard_oldshape_full | MLP-F117-early100-h800-source-slow-ema-terminal-guard | 0.0609749621815151 | 0.2458254165119595 | 0.5298810799916586 | 0.5196812848250071 | 0.4070614907476637 | 0.3183568881617652 | 0.2222588625219133 | 0.1499197847313351 | 0.4709173581793432 | 1 | 1 | 0 | 8 |  |
| F118 single terminal raw guard full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f118_single_terminal_raw_guard_oldshape_full | MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | 0.0492262641588846 | 0.2508025036917792 | 0.5423558387491438 | 0.5321504010094537 | 0.4195122321446736 | 0.3307922316922081 | 0.2346738775571187 | 0.1622624132368299 | 0.4905266741203586 | 1 | 1 | 0 | 8 |  |
| F119 single terminal raw guard strong full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f119_single_terminal_raw_guard_strong_oldshape_full | MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong | 0.0345257321993509 | 0.2347747186819712 | 0.5348958770434061 | 0.5246972176763747 | 0.4120653072992961 | 0.3233591185675727 | 0.2272418571843041 | 0.1548460755083296 | 0.4788671993982172 | 1 | 1 | 0 | 8 |  |
| F120 terminal projected optimizer full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f120_terminal_projected_optimizer_oldshape_full | MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer | 0.0533016721407572 | 0.2423147625393337 | 0.5351371202203963 | 0.5249133507410685 | 0.412259621752633 | 0.323529389169481 | 0.2273986405796474 | 0.1549697683917151 | 0.4789974993911114 | 1 | 1 | 0 | 8 |  |
| F121 terminal projected blend full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f121_terminal_projected_blend_oldshape_full | MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend | 0.0498369468583001 | 0.2403660284148322 | 0.5383261011706458 | 0.5280945863988664 | 0.415436887078815 | 0.3266988396644592 | 0.2305566204918755 | 0.1580851872762044 | 0.4838865893694881 | 1 | 1 | 0 | 8 |  |
| F122 terminal antiwashout full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f122_terminal_antiwashout_oldshape_full | MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout | 0.0435280601183573 | 0.2443078756332397 | 0.5379865500662062 | 0.527775956524743 | 0.4151379267374674 | 0.3264219197962019 | 0.2302946978145175 | 0.1578810877270168 | 0.4836718313083516 | 1 | 1 | 0 | 8 |  |
| F123 h4000 checkpoint reentry full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f123_h4000_reentry_oldshape_full | MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry | 0.0596391028828091 | 0.2510800361633301 | 0.5356751812828912 | 0.5254587067498101 | 0.4128122263484531 | 0.3240864939159817 | 0.2279582950803968 | 0.1555491255389319 | 0.4799617647110513 | 1 | 1 | 0 | 8 |  |
| F124-F126 signal/reservoir target reset full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full | MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target | 0.0367683900727166 | 0.2345359656545851 | 0.505232486459944 | 0.5088075134489272 | 0.4255666799015469 | 0.3312460548347897 | 0.2152763936254713 | 0.134279986222585 | 0.4053783713425892 | 1 | 1 | 0 | 8 |  |
| F124-F126 signal/reservoir target reset full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full | MLP-F125-early100-h800-source-slow-ema-source-bank-target | 0.0275752147038777 | 0.2277486821015676 | 0.4921054906315273 | 0.495961993932724 | 0.4133879608578152 | 0.3192800283432007 | 0.2035752170615726 | 0.1222706900702582 | 0.3829575269857686 | 1 | 1 | 0 | 9 |  |
| F124-F126 signal/reservoir target reset full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f124_f126_signal_target_oldshape_full | MLP-F126-early100-h800-source-slow-ema-dual-target-guard | 0.0290301177236768 | 0.238466997941335 | 0.5110694964726766 | 0.5146651996506585 | 0.4325850009918213 | 0.3385555247465769 | 0.2227248748143514 | 0.1414172252019246 | 0.4177076280405744 | 1 | 1 | 0 | 9 |  |
| F127-F129 source-state adaptive terminal full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full | MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw | 0.0244074794981214 | 0.2260927557945251 | 0.5026405254999796 | 0.5064758592181735 | 0.4238789081573486 | 0.3297484748893314 | 0.2137181427743699 | 0.1324055559105343 | 0.4015350062042641 | 1 | 1 | 0 | 8 |  |
| F127-F129 source-state adaptive terminal full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full | MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source | 0.0130445692274305 | 0.2175094352828132 | 0.4871440662278069 | 0.4909894598854912 | 0.4084062874317169 | 0.3142895963456895 | 0.1982795099417368 | 0.1169712808397081 | 0.3721767509957616 | 1 | 1 | 0 | 9 |  |
| F127-F129 source-state adaptive terminal full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f127_f129_source_state_adaptive_terminal_oldshape_full | MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve | 0.0341867208480834 | 0.2309097879462772 | 0.5018337799443139 | 0.5056931144661374 | 0.4231271511978573 | 0.3290293349160088 | 0.2130403055085076 | 0.1317636966705322 | 0.400461851537242 | 1 | 1 | 0 | 9 |  |
| F130-F132 terminal target reset oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full | MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target | 0.0258794360690646 | 0.2334463198979695 | 0.5126018656624688 | 0.5164534615145789 | 0.4338747163613637 | 0.3397622671392228 | 0.2237474487887488 | 0.142462584707472 | 0.4193007831828947 | 1 | 1 | 0 | 8 |  |
| F130-F132 terminal target reset oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full | MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target | 0.0197222961319817 | 0.2256713642014397 | 0.4940762917200724 | 0.4979171024428473 | 0.4153325623936123 | 0.321227232615153 | 0.2052310076024797 | 0.1241881317562527 | 0.3866052412344397 | 1 | 1 | 0 | 9 |  |
| F130-F132 terminal target reset oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f130_f132_terminal_target_reset_oldshape_full | MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target | 0.0293125576443142 | 0.2294596963458591 | 0.5048207574420505 | 0.5086872180302938 | 0.4261287086539798 | 0.332035524977578 | 0.216049227449629 | 0.1347834401660495 | 0.405930781578722 | 1 | 1 | 0 | 9 |  |
| F133-F135 terminal reject rescue oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full | MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue | 0.037391013569302 | 0.2404746148321363 | 0.5126342442300584 | 0.5165025856759813 | 0.43394402662913 | 0.3398546510272556 | 0.2238691714074876 | 0.1426512897014618 | 0.4197420552294324 | 1 | 1 | 0 | 7 |  |
| F133-F135 terminal reject rescue oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full | MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue | 0.0313817395104302 | 0.2263318300247192 | 0.4936133391327328 | 0.4974823461638556 | 0.4149239924218919 | 0.320832351843516 | 0.2048467000325521 | 0.1235881447792053 | 0.3852109803424209 | 1 | 1 | 0 | 9 |  |
| F133-F135 terminal reject rescue oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f133_f135_terminal_reject_rescue_oldshape_full | MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue | 0.0230470233493381 | 0.2293337119950188 | 0.5044616527027554 | 0.5083251926634047 | 0.4257562922106849 | 0.3316524492369758 | 0.2156531843874189 | 0.1343820591767629 | 0.4051894068200981 | 1 | 1 | 0 | 8 |  |
| F136-F138 source-shape preserve oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f136_f138_shape_preserve_oldshape_full | MLP-F136-early100-h800-source-slow-ema-shape-preserve | -1.5243507424990337 | -1.3746363719304402 | -1.082649701171451 | -1.0579475561777751 | -1.120796846018897 | -1.1960659821828206 | -1.2939708299107022 | -1.357645571231842 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F136-F138 source-shape preserve oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f136_f138_shape_preserve_oldshape_full | MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard | -1.5489292873276606 | -1.3992769055896337 | -1.107371773984697 | -1.0831479496426053 | -1.1465456287066145 | -1.2223913669586182 | -1.3209153877364264 | -1.385283715195126 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F136-F138 source-shape preserve oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f136_f138_shape_preserve_oldshape_full | MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp | -1.5143434140417311 | -1.3658073080910578 | -1.0752617981698778 | -1.0532494915856256 | -1.1183765000767178 | -1.1955820719401042 | -1.2951786054505243 | -1.3604410820537145 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F139-F141 terminal reject weak-row rescue oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f139_f141_terminal_reject_weakrow_oldshape_full | MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue | 0.0334433913230896 | 0.2413532733917236 | 0.5124931467903985 | 0.5163459678490957 | 0.4337684412797292 | 0.3396590550740559 | 0.2236685984664493 | 0.1424066258801354 | 0.4192634459549045 | 1 | 1 | 0 | 8 |  |
| F139-F141 terminal reject weak-row rescue oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f139_f141_terminal_reject_weakrow_oldshape_full | MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue | 0.0232686665323045 | 0.2262647019492255 | 0.4956471456421746 | 0.4995193547672695 | 0.4169597261481815 | 0.3228688836097717 | 0.2068851954407162 | 0.1256653997633192 | 0.3892149604456832 | 1 | 1 | 0 | 9 |  |
| F139-F141 terminal reject weak-row rescue oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f139_f141_terminal_reject_weakrow_oldshape_full | MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue | 0.0333744022581312 | 0.2395138409402635 | 0.5133888986375597 | 0.5172538956006368 | 0.4346884091695149 | 0.3405913147661421 | 0.2246015701029035 | 0.1433822446399265 | 0.4209803316281745 | 1 | 1 | 0 | 9 |  |
| F142-F144 top-k source-support oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full | MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | 0.0316448145442538 | 0.2345340914196438 | 0.5056587556997935 | 0.5095196564992269 | 0.4269552793767717 | 0.3328632546795739 | 0.2168703840838538 | 0.1355907983250088 | 0.407346850151824 | 1 | 1 | 0 | 8 |  |
| F142-F144 top-k source-support oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full | MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | 0.0209643377198113 | 0.2270598146650526 | 0.4965431590874989 | 0.5003942714797126 | 0.4178156620926327 | 0.3237038155396779 | 0.2077026830779181 | 0.1263947089513143 | 0.3904640689532481 | 1 | 1 | 0 | 9 |  |
| F142-F144 top-k source-support oldshape full | results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/continuation_f142_f144_topk_support_oldshape_full | MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | 0.0339639981587727 | 0.2391603489716847 | 0.5111051168706682 | 0.5149583948983086 | 0.4323798153135512 | 0.338267895910475 | 0.2222603228357103 | 0.1409425006972419 | 0.4166594063497629 | 1 | 1 | 0 | 9 |  |

### Per-Continuation Detail

### F88-F94 source-state oldshape full

- source_dir: `continuation_f88_f94_source_state_oldshape_full`
- candidate rows: 4
- early / continuous_h3200 / h4800: 0 / 0 / 0
- best: `MLP-F88-h800-source-slow-ema-retention` h4800=-0.0752062731319003, h4800_retention_ratio=0.0

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F88-h800-source-slow-ema-retention | 0.0458766222000122 | -0.0006026228268941 | 0.2556371953752305 | 0.3166095813115437 | 0.223662429385715 | 0.0943417416678534 | -0.0068413813908894 | -0.0752062731319003 | 0.0 | 0 | 0 | 0 | 5 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F89-h800-readout-channel-retention | 0.0487066970931159 | -0.0152870151731703 | 0.238328390651279 | 0.2993032071325514 | 0.2063628567589653 | 0.0770309103859795 | -0.0241510007116529 | -0.092516541481018 | 0.0 | 0 | 0 | 0 | 3 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F93-h800-readout-channel-alt25-strong-retention | 0.0371184349060058 | 0.0008842547734578 | 0.2498369945420159 | 0.310828771856096 | 0.2179053823153177 | 0.0885904828707377 | -0.0125796794891357 | -0.0809141331248813 | 0.0 | 0 | 0 | 0 | 3 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F94-h1600-source-checkpoint-reentry | 0.005660679605272 | -0.0425370335578918 | 0.209342082341512 | 0.270304434829288 | 0.1773516800668504 | 0.048010508219401 | -0.0531877014372083 | -0.1215635736783345 | 0.0 | 0 | 0 | 0 | 4 | EarlySourceChainMissing;ContinuousH3200Missing |

### F106-F107 early100 source-state combined full

- source_dir: `continuation_f106_f107_early_source_state_oldshape_full`
- candidate rows: 2
- early / continuous_h3200 / h4800: 0 / 0 / 0
- best: `MLP-F106-early100-h800-source-slow-ema-retention` h4800=0.1428640021218193, h4800_retention_ratio=0.4597692805313301

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F106-early100-h800-source-slow-ema-retention | 0.0039654705259535 | 0.2313903371493021 | 0.5031206475363837 | 0.5223599672317505 | 0.4309421115451389 | 0.3107297685411241 | 0.2132231593132019 | 0.1428640021218193 | 0.4597692805313301 | 0 | 0 | 0 | 8 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F107-early100-h800-readout-channel-retention | -0.0115754935476515 | 0.2040987511475881 | 0.4616460667716132 | 0.4808787902196248 | 0.3894584245151943 | 0.2692307763629489 | 0.1717093918058607 | 0.1013376547230614 | 0.3763969932859702 | 0 | 0 | 0 | 7 | EarlySourceChainMissing;ContinuousH3200Missing |

### F106 single early100 slow-EMA full

- source_dir: `continuation_f106_single_early_source_state_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F106-early100-h800-source-slow-ema-retention` h4800=0.1525942418310377, h4800_retention_ratio=0.4752410639371254

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F106-early100-h800-source-slow-ema-retention | 0.0407662987709045 | 0.2366060548358493 | 0.5325850115882026 | 0.5223965711063809 | 0.409780740737915 | 0.3210880822605557 | 0.2249905467033386 | 0.1525942418310377 | 0.4752410639371254 | 1 | 1 | 0 | 8 |  |

### F115 single early100 slow-EMA strong full

- source_dir: `continuation_f115_single_early100_slowema_strong_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F115-early100-h800-source-slow-ema-strong-retention` h4800=0.1525664197074042, h4800_retention_ratio=0.4752361940650902

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F115-early100-h800-source-slow-ema-strong-retention | 0.0407662987709045 | 0.2366316980785793 | 0.5324831472502815 | 0.5223103364308676 | 0.4097101357248094 | 0.3210328287548489 | 0.2249501678678724 | 0.1525664197074042 | 0.4752361940650902 | 1 | 1 | 0 | 8 |  |

### F117 single terminal source guard full

- source_dir: `continuation_f117_single_terminal_guard_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F117-early100-h800-source-slow-ema-terminal-guard` h4800=0.1499197847313351, h4800_retention_ratio=0.4709173581793432

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F117-early100-h800-source-slow-ema-terminal-guard | 0.0609749621815151 | 0.2458254165119595 | 0.5298810799916586 | 0.5196812848250071 | 0.4070614907476637 | 0.3183568881617652 | 0.2222588625219133 | 0.1499197847313351 | 0.4709173581793432 | 1 | 1 | 0 | 8 |  |

### F118 single terminal raw guard full

- source_dir: `continuation_f118_single_terminal_raw_guard_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard` h4800=0.1622624132368299, h4800_retention_ratio=0.4905266741203586

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | 0.0492262641588846 | 0.2508025036917792 | 0.5423558387491438 | 0.5321504010094537 | 0.4195122321446736 | 0.3307922316922081 | 0.2346738775571187 | 0.1622624132368299 | 0.4905266741203586 | 1 | 1 | 0 | 8 |  |

### F119 single terminal raw guard strong full

- source_dir: `continuation_f119_single_terminal_raw_guard_strong_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong` h4800=0.1548460755083296, h4800_retention_ratio=0.4788671993982172

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong | 0.0345257321993509 | 0.2347747186819712 | 0.5348958770434061 | 0.5246972176763747 | 0.4120653072992961 | 0.3233591185675727 | 0.2272418571843041 | 0.1548460755083296 | 0.4788671993982172 | 1 | 1 | 0 | 8 |  |

### F120 terminal projected optimizer full

- source_dir: `continuation_f120_terminal_projected_optimizer_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer` h4800=0.1549697683917151, h4800_retention_ratio=0.4789974993911114

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer | 0.0533016721407572 | 0.2423147625393337 | 0.5351371202203963 | 0.5249133507410685 | 0.412259621752633 | 0.323529389169481 | 0.2273986405796474 | 0.1549697683917151 | 0.4789974993911114 | 1 | 1 | 0 | 8 |  |

### F121 terminal projected blend full

- source_dir: `continuation_f121_terminal_projected_blend_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend` h4800=0.1580851872762044, h4800_retention_ratio=0.4838865893694881

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend | 0.0498369468583001 | 0.2403660284148322 | 0.5383261011706458 | 0.5280945863988664 | 0.415436887078815 | 0.3266988396644592 | 0.2305566204918755 | 0.1580851872762044 | 0.4838865893694881 | 1 | 1 | 0 | 8 |  |

### F122 terminal antiwashout full

- source_dir: `continuation_f122_terminal_antiwashout_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout` h4800=0.1578810877270168, h4800_retention_ratio=0.4836718313083516

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout | 0.0435280601183573 | 0.2443078756332397 | 0.5379865500662062 | 0.527775956524743 | 0.4151379267374674 | 0.3264219197962019 | 0.2302946978145175 | 0.1578810877270168 | 0.4836718313083516 | 1 | 1 | 0 | 8 |  |

### F123 h4000 checkpoint reentry full

- source_dir: `continuation_f123_h4000_reentry_oldshape_full`
- candidate rows: 1
- early / continuous_h3200 / h4800: 1 / 1 / 0
- best: `MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry` h4800=0.1555491255389319, h4800_retention_ratio=0.4799617647110513

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry | 0.0596391028828091 | 0.2510800361633301 | 0.5356751812828912 | 0.5254587067498101 | 0.4128122263484531 | 0.3240864939159817 | 0.2279582950803968 | 0.1555491255389319 | 0.4799617647110513 | 1 | 1 | 0 | 8 |  |

### F124-F126 signal/reservoir target reset full

- source_dir: `continuation_f124_f126_signal_target_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 3 / 3 / 0
- best: `MLP-F126-early100-h800-source-slow-ema-dual-target-guard` h4800=0.1414172252019246, h4800_retention_ratio=0.4177076280405744

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target | 0.0367683900727166 | 0.2345359656545851 | 0.505232486459944 | 0.5088075134489272 | 0.4255666799015469 | 0.3312460548347897 | 0.2152763936254713 | 0.134279986222585 | 0.4053783713425892 | 1 | 1 | 0 | 8 |  |
| MLP-F125-early100-h800-source-slow-ema-source-bank-target | 0.0275752147038777 | 0.2277486821015676 | 0.4921054906315273 | 0.495961993932724 | 0.4133879608578152 | 0.3192800283432007 | 0.2035752170615726 | 0.1222706900702582 | 0.3829575269857686 | 1 | 1 | 0 | 9 |  |
| MLP-F126-early100-h800-source-slow-ema-dual-target-guard | 0.0290301177236768 | 0.238466997941335 | 0.5110694964726766 | 0.5146651996506585 | 0.4325850009918213 | 0.3385555247465769 | 0.2227248748143514 | 0.1414172252019246 | 0.4177076280405744 | 1 | 1 | 0 | 9 |  |

### F127-F129 source-state adaptive terminal full

- source_dir: `continuation_f127_f129_source_state_adaptive_terminal_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 3 / 3 / 0
- best: `MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw` h4800=0.1324055559105343, h4800_retention_ratio=0.4015350062042641

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw | 0.0244074794981214 | 0.2260927557945251 | 0.5026405254999796 | 0.5064758592181735 | 0.4238789081573486 | 0.3297484748893314 | 0.2137181427743699 | 0.1324055559105343 | 0.4015350062042641 | 1 | 1 | 0 | 8 |  |
| MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source | 0.0130445692274305 | 0.2175094352828132 | 0.4871440662278069 | 0.4909894598854912 | 0.4084062874317169 | 0.3142895963456895 | 0.1982795099417368 | 0.1169712808397081 | 0.3721767509957616 | 1 | 1 | 0 | 9 |  |
| MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve | 0.0341867208480834 | 0.2309097879462772 | 0.5018337799443139 | 0.5056931144661374 | 0.4231271511978573 | 0.3290293349160088 | 0.2130403055085076 | 0.1317636966705322 | 0.400461851537242 | 1 | 1 | 0 | 9 |  |

### F130-F132 terminal target reset oldshape full

- source_dir: `continuation_f130_f132_terminal_target_reset_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 3 / 3 / 0
- best: `MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target` h4800=0.142462584707472, h4800_retention_ratio=0.4193007831828947

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target | 0.0258794360690646 | 0.2334463198979695 | 0.5126018656624688 | 0.5164534615145789 | 0.4338747163613637 | 0.3397622671392228 | 0.2237474487887488 | 0.142462584707472 | 0.4193007831828947 | 1 | 1 | 0 | 8 |  |
| MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target | 0.0197222961319817 | 0.2256713642014397 | 0.4940762917200724 | 0.4979171024428473 | 0.4153325623936123 | 0.321227232615153 | 0.2052310076024797 | 0.1241881317562527 | 0.3866052412344397 | 1 | 1 | 0 | 9 |  |
| MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target | 0.0293125576443142 | 0.2294596963458591 | 0.5048207574420505 | 0.5086872180302938 | 0.4261287086539798 | 0.332035524977578 | 0.216049227449629 | 0.1347834401660495 | 0.405930781578722 | 1 | 1 | 0 | 9 |  |

### F133-F135 terminal reject rescue oldshape full

- source_dir: `continuation_f133_f135_terminal_reject_rescue_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 3 / 3 / 0
- best: `MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue` h4800=0.1426512897014618, h4800_retention_ratio=0.4197420552294324

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue | 0.037391013569302 | 0.2404746148321363 | 0.5126342442300584 | 0.5165025856759813 | 0.43394402662913 | 0.3398546510272556 | 0.2238691714074876 | 0.1426512897014618 | 0.4197420552294324 | 1 | 1 | 0 | 7 |  |
| MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue | 0.0313817395104302 | 0.2263318300247192 | 0.4936133391327328 | 0.4974823461638556 | 0.4149239924218919 | 0.320832351843516 | 0.2048467000325521 | 0.1235881447792053 | 0.3852109803424209 | 1 | 1 | 0 | 9 |  |
| MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue | 0.0230470233493381 | 0.2293337119950188 | 0.5044616527027554 | 0.5083251926634047 | 0.4257562922106849 | 0.3316524492369758 | 0.2156531843874189 | 0.1343820591767629 | 0.4051894068200981 | 1 | 1 | 0 | 8 |  |

### F136-F138 source-shape preserve oldshape full

- source_dir: `continuation_f136_f138_shape_preserve_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 0 / 0 / 0
- best: `MLP-F136-early100-h800-source-slow-ema-shape-preserve` h4800=-1.357645571231842, h4800_retention_ratio=

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F136-early100-h800-source-slow-ema-shape-preserve | -1.5243507424990337 | -1.3746363719304402 | -1.082649701171451 | -1.0579475561777751 | -1.120796846018897 | -1.1960659821828206 | -1.2939708299107022 | -1.357645571231842 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard | -1.5489292873276606 | -1.3992769055896337 | -1.107371773984697 | -1.0831479496426053 | -1.1465456287066145 | -1.2223913669586182 | -1.3209153877364264 | -1.385283715195126 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp | -1.5143434140417311 | -1.3658073080910578 | -1.0752617981698778 | -1.0532494915856256 | -1.1183765000767178 | -1.1955820719401042 | -1.2951786054505243 | -1.3604410820537145 |  | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |

### F139-F141 terminal reject weak-row rescue oldshape full

- source_dir: `continuation_f139_f141_terminal_reject_weakrow_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 3 / 3 / 0
- best: `MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue` h4800=0.1433822446399265, h4800_retention_ratio=0.4209803316281745

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue | 0.0334433913230896 | 0.2413532733917236 | 0.5124931467903985 | 0.5163459678490957 | 0.4337684412797292 | 0.3396590550740559 | 0.2236685984664493 | 0.1424066258801354 | 0.4192634459549045 | 1 | 1 | 0 | 8 |  |
| MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue | 0.0232686665323045 | 0.2262647019492255 | 0.4956471456421746 | 0.4995193547672695 | 0.4169597261481815 | 0.3228688836097717 | 0.2068851954407162 | 0.1256653997633192 | 0.3892149604456832 | 1 | 1 | 0 | 9 |  |
| MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue | 0.0333744022581312 | 0.2395138409402635 | 0.5133888986375597 | 0.5172538956006368 | 0.4346884091695149 | 0.3405913147661421 | 0.2246015701029035 | 0.1433822446399265 | 0.4209803316281745 | 1 | 1 | 0 | 9 |  |

### F142-F144 top-k source-support oldshape full

- source_dir: `continuation_f142_f144_topk_support_oldshape_full`
- candidate rows: 3
- early / continuous_h3200 / h4800: 3 / 3 / 0
- best: `MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap` h4800=0.1409425006972419, h4800_retention_ratio=0.4166594063497629

| v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | early_source_chain_group | continuous_h3200_group | productive_h4800_group | row_h4800_positive_count | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | 0.0316448145442538 | 0.2345340914196438 | 0.5056587556997935 | 0.5095196564992269 | 0.4269552793767717 | 0.3328632546795739 | 0.2168703840838538 | 0.1355907983250088 | 0.407346850151824 | 1 | 1 | 0 | 8 |  |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | 0.0209643377198113 | 0.2270598146650526 | 0.4965431590874989 | 0.5003942714797126 | 0.4178156620926327 | 0.3237038155396779 | 0.2077026830779181 | 0.1263947089513143 | 0.3904640689532481 | 1 | 1 | 0 | 9 |  |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | 0.0339639981587727 | 0.2391603489716847 | 0.5111051168706682 | 0.5149583948983086 | 0.4323798153135512 | 0.338267895910475 | 0.2222603228357103 | 0.1409425006972419 | 0.4166594063497629 | 1 | 1 | 0 | 9 |  |

### F118 vs F142-F144 Dataset Localization

- This localization reads row-level fresh matrices from each continuation source_dir. F118 is included as the best near-miss baseline; F142-F144 are the latest top-k support variants.
| v22_id | dataset | rows | mean_h800 | mean_h1600 | mean_h2400 | mean_h3200 | mean_h4000 | mean_h4800 | h4800_positive_rows | early_rows | continuous_rows | h4800_rows | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | MNIST | 3 | 0.25092048446337384 | 0.39193562666575116 | 0.47797588507334393 | 0.48204462726910907 | 0.3622950315475464 | 0.2551374038060506 | 3 | 0 | 0 | 0 |  |
| MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | Fashion-MNIST | 3 | 0.5315764745076498 | 0.38961128393809 | 0.20424002408981323 | 0.11501852671305339 | 0.06446935733159383 | 0.03780712683995565 | 2 | 0 | 0 | 0 |  |
| MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | KMNIST | 3 | 0.8445705572764078 | 0.8149042924245199 | 0.5763207872708639 | 0.3953135410944621 | 0.277257243792216 | 0.19384270906448364 | 3 | 0 | 0 | 0 |  |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | MNIST | 3 | 0.2816585997740428 | 0.408263365427653 | 0.4872342844804128 | 0.49118786056836444 | 0.329880028963089 | 0.20791242520014444 | 3 | 2 | 2 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | MNIST | 3 | 0.2388604978720347 | 0.36545733610788983 | 0.4444211224714915 | 0.448368380467097 | 0.2870521346728007 | 0.1650643746058146 | 3 | 2 | 2 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | MNIST | 3 | 0.25925059119860333 | 0.38585154215494794 | 0.4648187756538391 | 0.46876704692840576 | 0.30745359261830646 | 0.18546704451243082 | 3 | 1 | 1 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | Fashion-MNIST | 3 | 0.5495149691899618 | 0.43420926729838055 | 0.26266805330912274 | 0.16617616017659506 | 0.1074996789296468 | 0.06792024771372478 | 3 | 2 | 0 | 0 | ContinuousH3200Missing;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | Fashion-MNIST | 3 | 0.5436116456985474 | 0.4282846252123515 | 0.256715993086497 | 0.1601790189743042 | 0.10148876905441284 | 0.06186666091283163 | 3 | 2 | 0 | 0 | ContinuousH3200Missing;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | KMNIST | 3 | 0.7399832010269165 | 0.7402462164560953 | 0.5850982268651327 | 0.3953356941541036 | 0.2673116723696391 | 0.18496674299240112 | 3 | 3 | 1 | 0 | ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | KMNIST | 3 | 0.685802698135376 | 0.6860863367716471 | 0.5309635003407797 | 0.3412257432937622 | 0.21323144435882568 | 0.13093972206115723 | 2 | 2 | 1 | 0 | ContinuousH3200Missing;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | Fashion-MNIST | 3 | 0.5340815583864847 | 0.41877742608388263 | 0.24722244342168173 | 0.15070094664891562 | 0.09201570351918538 | 0.052393714586893715 | 3 | 1 | 0 | 0 | ContinuousH3200Missing;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | KMNIST | 3 | 0.7071573336919149 | 0.7074408531188965 | 0.5523098707199097 | 0.36256404717763263 | 0.23456714550654092 | 0.15225309133529663 | 3 | 2 | 1 | 0 | ContinuousH3200Missing;EarlySourceChainMissing;ContinuousH3200Missing |

### F118 vs F142-F144 Row Localization

| v22_id | dataset | seed | source_h100 | source_h400 | source_h800 | source_h1600 | source_h2400 | source_h3200 | source_h4000 | source_h4800 | v22_02_early_source_chain | v22_02_continuous_h3200_chain | v22_02_productive_h4800_chain | v22_02_source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | MNIST | 0 | 0.13568055629730225 | 0.12485814094543457 | 0.28776419162750244 | 0.4286342263221741 | 0.5173377990722656 | 0.49888044595718384 | 0.3418847918510437 | 0.2342965006828308 |  |  |  |  |
|  | Fashion-MNIST | 1 | 0.03064894676208496 | 0.16486036777496338 | 0.5022470951080322 | 0.4155887961387634 | 0.23842346668243408 | 0.1625729203224182 | 0.10630577802658081 | 0.07896673679351807 |  |  |  |  |
|  | KMNIST | 2 | 0.06310051679611206 | 0.43888020515441895 | 0.8360615968704224 | 0.7916139364242554 | 0.49507415294647217 | 0.325972318649292 | 0.23383861780166626 | 0.17221462726593018 |  |  |  |  |
|  | MNIST | 1 | -0.015602350234985352 | 0.10264790058135986 | 0.19124212861061096 | 0.3195544481277466 | 0.3942229747772217 | 0.4488776624202728 | 0.37580054998397827 | 0.2401534914970398 |  |  |  |  |
|  | Fashion-MNIST | 2 | 0.006162405014038086 | 0.2885943651199341 | 0.7933565378189087 | 0.38613778352737427 | 0.18756014108657837 | 0.0683891773223877 | 0.00429689884185791 | -0.02453935146331787 |  |  |  |  |
|  | MNIST | 2 | -0.06441891193389893 | 0.08303266763687134 | 0.27375513315200806 | 0.42761820554733276 | 0.5223668813705444 | 0.4983757734298706 | 0.3691997528076172 | 0.29096221923828125 |  |  |  |  |
|  | KMNIST | 0 | 0.22008025646209717 | 0.6975750923156738 | 1.148403525352478 | 0.8822020292282104 | 0.6565818190574646 | 0.49858129024505615 | 0.3793349862098694 | 0.28489696979522705 |  |  |  |  |
|  | Fashion-MNIST | 0 | -0.022017240524291992 | 0.07436293363571167 | 0.2991257905960083 | 0.3671072721481323 | 0.18673646450042725 | 0.11409348249435425 | 0.08280539512634277 | 0.05899399518966675 |  |  |  |  |
|  | KMNIST | 1 | 0.0894021987915039 | 0.2824108600616455 | 0.5492465496063232 | 0.7708969116210938 | 0.5773063898086548 | 0.3613870143890381 | 0.2185981273651123 | 0.1244165301322937 |  |  |  |  |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | MNIST | 0 | 0.06109994649887085 | 0.025780141353607178 | 0.1939866542816162 | 0.30798351764678955 | 0.3798971176147461 | 0.4306755065917969 | 0.29368966817855835 | 0.20475131273269653 | 1 | 1 | 0 |  |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | MNIST | 1 | 0.06761348247528076 | 0.1137804388999939 | 0.27119120955467224 | 0.395537793636322 | 0.47120049595832825 | 0.47435900568962097 | 0.32150810956954956 | 0.22037869691848755 | 1 | 1 | 0 |  |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | MNIST | 2 | -0.05622220039367676 | 0.172784686088562 | 0.35286223888397217 | 0.494317889213562 | 0.5836327075958252 | 0.5415397882461548 | 0.34742772579193115 | 0.17154359817504883 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | Fashion-MNIST | 1 | 0.042949557304382324 | 0.23396283388137817 | 0.6576339602470398 | 0.3247182369232178 | 0.20141685009002686 | 0.1456194519996643 | 0.11399829387664795 | 0.08955466747283936 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | Fashion-MNIST | 2 | 0.10393399000167847 | 0.24257707595825195 | 0.6617316603660583 | 0.6115275025367737 | 0.39656442403793335 | 0.2604294419288635 | 0.1726779341697693 | 0.11358147859573364 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | KMNIST | 0 | 0.052173733711242676 | 0.5622465014457703 | 0.8436357975006104 | 0.7585673332214355 | 0.5223738551139832 | 0.33688998222351074 | 0.21495068073272705 | 0.1324670910835266 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | KMNIST | 2 | 0.08187687397003174 | 0.5109328031539917 | 0.8656752109527588 | 0.7672970294952393 | 0.5270068645477295 | 0.3543992042541504 | 0.25108373165130615 | 0.1930890679359436 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | MNIST | 0 | 0.017094790935516357 | 0.01327979564666748 | 0.178693950176239 | 0.2926739454269409 | 0.3645772337913513 | 0.41534483432769775 | 0.2783472537994385 | 0.18936294317245483 | 1 | 1 | 0 |  |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | MNIST | 1 | 0.05722320079803467 | 0.13434532284736633 | 0.2880200445652008 | 0.4123653769493103 | 0.48802649974823 | 0.49118226766586304 | 0.3383321762084961 | 0.23719841241836548 | 1 | 1 | 0 |  |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | Fashion-MNIST | 0 | -0.03659069538116455 | 0.08531349897384644 | 0.36204373836517334 | 0.39922845363616943 | 0.2228473424911499 | 0.1252463459968567 | 0.06857126951217651 | 0.03327423334121704 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | Fashion-MNIST | 1 | 0.06556791067123413 | 0.2307584285736084 | 0.6114017367362976 | 0.2785242795944214 | 0.15525054931640625 | 0.09948104619979858 | 0.06791967153549194 | 0.04351168870925903 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | Fashion-MNIST | 2 | -0.015633046627044678 | 0.16118019819259644 | 0.587897777557373 | 0.5376211404800415 | 0.3225785493850708 | 0.18640244007110596 | 0.09858369827270508 | 0.039425790309906006 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | KMNIST | 1 | 0.07413601875305176 | 0.2625161409378052 | 0.5261919498443604 | 0.7104251384735107 | 0.7214834690093994 | 0.5103062391281128 | 0.35150575637817383 | 0.24497872591018677 | 1 | 1 | 0 |  |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | KMNIST | 2 | 0.06912887096405029 | 0.5108405947685242 | 0.8184528350830078 | 0.7201032638549805 | 0.47983431816101074 | 0.3072474002838135 | 0.20396959781646729 | 0.14604222774505615 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | MNIST | 0 | -0.025265634059906006 | -0.030920743942260742 | 0.13686949014663696 | 0.25087136030197144 | 0.32279711961746216 | 0.37357908487319946 | 0.2366008758544922 | 0.14765912294387817 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | MNIST | 2 | -0.07772809267044067 | 0.21688735485076904 | 0.3816802501678467 | 0.5231368541717529 | 0.612454891204834 | 0.5703650116920471 | 0.3762562870979309 | 0.2003825306892395 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | Fashion-MNIST | 0 | -0.02437061071395874 | 0.07996046543121338 | 0.35770153999328613 | 0.3948020935058594 | 0.21833300590515137 | 0.12062656879425049 | 0.0638687014579773 | 0.028506815433502197 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | Fashion-MNIST | 1 | 0.04938244819641113 | 0.24914908409118652 | 0.6523198485374451 | 0.3195817470550537 | 0.19642847776412964 | 0.14074653387069702 | 0.10926741361618042 | 0.08493602275848389 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | KMNIST | 0 | -0.04525274038314819 | 0.42741096019744873 | 0.6655409336090088 | 0.5805368423461914 | 0.34440016746520996 | 0.15897178649902344 | 0.03710484504699707 | -0.04524862766265869 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | KMNIST | 1 | 0.07555949687957764 | 0.2547679543495178 | 0.48746830224990845 | 0.6717063188552856 | 0.6827549934387207 | 0.47156405448913574 | 0.31274640560150146 | 0.20619255304336548 | 1 | 1 | 0 |  |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | KMNIST | 2 | 0.1661745309829712 | 0.572526216506958 | 0.9187467098236084 | 0.8203577995300293 | 0.5800464749336243 | 0.40741831064224243 | 0.3040829300880432 | 0.24605190753936768 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | MNIST | 1 | 0.056389033794403076 | 0.11644864082336426 | 0.2693088948726654 | 0.3936697244644165 | 0.4693508446216583 | 0.4725230634212494 | 0.3196941316127777 | 0.21860343217849731 | 1 | 1 | 0 |  |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | MNIST | 2 | -0.1316523551940918 | 0.10709655284881592 | 0.26669633388519287 | 0.4081602692604065 | 0.4974856376647949 | 0.45540130138397217 | 0.26130104064941406 | 0.08545148372650146 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | Fashion-MNIST | 0 | 0.002837061882019043 | 0.10486501455307007 | 0.36202704906463623 | 0.39912939071655273 | 0.22266030311584473 | 0.12495386600494385 | 0.06819599866867065 | 0.03281933069229126 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard | Fashion-MNIST | 2 | 0.12792342901229858 | 0.23155444860458374 | 0.6288672089576721 | 0.5786811113357544 | 0.3637399673461914 | 0.22766268253326416 | 0.13992947340011597 | 0.08093184232711792 | 1 | 0 | 0 | ContinuousH3200Missing |
| MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard | KMNIST | 0 | -0.05419653654098511 | 0.4904770255088806 | 0.8155508637428284 | 0.7305129766464233 | 0.49434030055999756 | 0.30888068675994873 | 0.18698543310165405 | 0.10452449321746826 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap | KMNIST | 1 | 0.07500588893890381 | 0.22626686096191406 | 0.45756709575653076 | 0.6418135166168213 | 0.6528743505477905 | 0.4416987895965576 | 0.282901406288147 | 0.17638123035430908 | 1 | 1 | 0 |  |

### Smoke audit

| continuation | v22_id | h100 | h400 | h800 | h1600 | early_source_chain_group | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F88-F95 source-state h1600 smoke | MLP-F88-h800-source-slow-ema-retention | 0.0610187848409016 | 0.0275488297144571 | 0.2628185351689656 | 0.2779937585194905 | 1 | ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F89-h800-readout-channel-retention | 0.0256959597269694 | 0.0111789902051289 | 0.2400519649187723 | 0.2552424470583598 | 1 | ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F90-h800-dual-memory-source-retention | -0.0260337591171264 | -0.0128986636797587 | 0.2325759331385294 | 0.2477245330810547 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F91-h800-readout-channel-alt25-retention | 0.0437739094098409 | -0.0018481016159057 | 0.2282069524129232 | 0.2433095971743265 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F92-h800-readout-channel-strong-retention | -0.0065382321675618 | -0.0330585241317749 | 0.186824361483256 | 0.2019850413004557 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F93-h800-readout-channel-alt25-strong-retention | 0.0575431187947591 | 0.0252008438110351 | 0.258682628472646 | 0.2738717397054036 | 1 | ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F94-h1600-source-checkpoint-reentry | 0.0203903714815775 | 0.0271094640096028 | 0.2518690228462219 | 0.2670310338338216 | 1 | ContinuousH3200Missing |
| F88-F95 source-state h1600 smoke | MLP-F95-h2400-source-checkpoint-reentry | -0.0312713185946146 | 0.002075970172882 | 0.2327083746592203 | 0.2478442986806233 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F106-F109 early100 h800 smoke rerun | MLP-F106-early100-h800-source-slow-ema-retention | 0.0313844879468282 | 0.2615151802698771 | 0.5144874850908915 |  | 1 | ContinuousH3200Missing |
| F106-F109 early100 h800 smoke rerun | MLP-F107-early100-h800-readout-channel-retention | 0.0163868467013041 | 0.2160167495409647 | 0.4646130800247192 |  | 1 | ContinuousH3200Missing |
| F106-F109 early100 h800 smoke rerun | MLP-F108-early100-h800-readout-channel-alt25-strong-retention | 0.001573900381724 | 0.2143134276072184 | 0.477721889813741 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F106-F109 early100 h800 smoke rerun | MLP-F109-early100-h1600-source-checkpoint-reentry | -0.0031490325927734 | 0.2397437493006388 | 0.4987160563468933 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F110-F113 early1/50 h800 smoke | MLP-F110-early1-h800-source-slow-ema-retention | -0.9741917451222738 | 0.0279238820075988 | 0.4871559937795003 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F110-F113 early1/50 h800 smoke | MLP-F111-early1-h800-readout-channel-retention | -0.9939737319946288 | 0.0446561574935913 | 0.4727965792020162 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F110-F113 early1/50 h800 smoke | MLP-F112-early50-h800-source-slow-ema-retention | -0.2603133519490559 | 0.1706223090489705 | 0.4913331270217895 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F110-F113 early1/50 h800 smoke | MLP-F113-early50-h800-readout-channel-retention | -0.2688126564025879 | 0.1641320586204528 | 0.4988489747047424 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F114-F116 strength h800 smoke | MLP-F114-early100-h800-source-slow-ema-alt25-retention | -0.0035733580589294 | 0.1040950814882914 | 0.3450854619344075 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F114-F116 strength h800 smoke | MLP-F115-early100-h800-source-slow-ema-strong-retention | 0.01603764295578 | 0.1284376581509908 | 0.3593316475550334 |  | 1 | ContinuousH3200Missing |
| F114-F116 strength h800 smoke | MLP-F116-early100-h800-source-slow-ema-alt25-strong-retention | -0.0326912999153137 | 0.1023780306180318 | 0.3748794198036194 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F117 terminal source guard h800 smoke | MLP-F117-early100-h800-source-slow-ema-terminal-guard | 0.0445970098177591 | 0.1470733284950256 | 0.4158816337585449 |  | 1 | ContinuousH3200Missing |
| F118 terminal raw guard h800 smoke | MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | 0.0309682289759318 | 0.142364501953125 | 0.4111365079879761 |  | 1 | ContinuousH3200Missing |
| F119 terminal raw guard strong h800 smoke | MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong | 0.0300716360410054 | 0.1457124153772989 | 0.4130157430966695 |  | 1 | ContinuousH3200Missing |
| F120-F122 optimizer projection h800 smoke | MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer | 2.7318795522054035e-05 | 0.1836413939793904 | 0.4765799045562744 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F120-F122 optimizer projection h800 smoke | MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend | 0.0042237242062886 | 0.1880044142405192 | 0.502418577671051 |  | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F120-F122 optimizer projection h800 smoke | MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout | 0.010641594727834 | 0.2012949188550313 | 0.5231472849845886 |  | 1 | ContinuousH3200Missing |
| F123 h4000 reentry h800 smoke | MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry | 0.0705233017603556 | 0.1162826021512349 | 0.3631906708081563 |  | 1 | ContinuousH3200Missing |
| F124-F126 signal/reservoir target smoke2 h1600 | MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target | 0.0228763421376546 | 0.19784543911616 | 0.4832680424054463 | 0.3676867882410685 | 1 | ContinuousH3200Missing |
| F124-F126 signal/reservoir target smoke2 h1600 | MLP-F125-early100-h800-source-slow-ema-source-bank-target | -0.0135401288668314 | 0.1882272164026896 | 0.4986053705215454 | 0.3829055825869242 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| F124-F126 signal/reservoir target smoke2 h1600 | MLP-F126-early100-h800-source-slow-ema-dual-target-guard | -0.0010849038759867 | 0.2048581838607788 | 0.5178552468617758 | 0.4021964073181152 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |


## Terminal Collapse Autopsy


_无落盘 rows。_

## Precommit Selector

| selector_feature_name | selector_train_only | selector_uses_future | selector_AUC_retained_h4800_on_shadow_pool | selector_precision_at_k | fresh_selected_h4800_pass_rows | selector_decision |
| --- | --- | --- | --- | --- | --- | --- |
| train_loss_h100 | 1 | 0 |  | 0.0 | 0 | NoPrecommitPass |
| train_loss_h400 | 1 | 0 |  | 0.0 | 0 | NoPrecommitPass |
| B2_transfer_gain_h400 | 1 | 0 |  |  | 0 | NoPrecommitPass |
| B3_safety_gain_h400 | 1 | 0 |  |  | 0 | NoPrecommitPass |
| ActuationR2_h400 | 1 | 0 |  |  | 0 | NoPrecommitPass |
| source_state_current_cos_h400 | 1 | 0 |  |  | 0 | NoPrecommitPass |
| LineC_fast_loss_h400 | 1 | 0 |  |  | 0 | NoPrecommitPass |

## Source Target Reset

| carrier | v22_id | target_family | ActuationR2 | B2_transfer_gain | random_target_B2_gain | h4800_retention_rate | target_reset_decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | F10-T7-b1-cross-split-consensus-transfer | T2-cross-split-consensus | 0.2750802238782247 | 0.04400197002622816 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F25-loss-warm-to-b1-consensus-migration | T2-cross-split-consensus | 0.5971961087650723 | 0.1237022876739502 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F3-T1-loss-cotangent-target | T1-loss-cotangent | 0.5611197749773661 | 0.13524330655733743 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F3-T5-random-matched-target | T9-random-matched | 0.43602176507314044 | -0.008591161833869087 | -0.008591161833869087 | 0.0 | TargetControlEquivalentOrWeak |
| D-CHE | F30-gain-gated-loss-warm-b1-consensus | T2-cross-split-consensus | 0.5935484303368462 | 0.11416584915584987 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F33-loss-warm-to-b1-consensus-b3-null | T2-cross-split-consensus | 0.5801834662755331 | 0.12973969512515598 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F35-loss-warm-to-view-consistent-loss | T-other | 0.5958307186762491 | 0.13939236932330662 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.5779258476363288 | 0.13759788539674547 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.6032821271154616 | 0.12396695547633702 | -0.008591161833869087 | 0.0 | TargetObservableNoRetention |
| D-FOU | F10-T7-b1-cross-split-consensus-transfer | T2-cross-split-consensus | 0.24739316436979505 | 0.02276786168416341 | -0.012681000762515597 | 0.0 | TargetControlEquivalentOrWeak |
| D-FOU | F25-loss-warm-to-b1-consensus-migration | T2-cross-split-consensus | 0.612061427699195 | 0.14594736033015782 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |
| D-FOU | F3-T1-loss-cotangent-target | T1-loss-cotangent | 0.632168538040585 | 0.1273758245839013 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |
| D-FOU | F3-T5-random-matched-target | T9-random-matched | 0.5297892093658447 | -0.012681000762515597 | -0.012681000762515597 | 0.0 | TargetControlEquivalentOrWeak |
| D-FOU | F30-gain-gated-loss-warm-b1-consensus | T2-cross-split-consensus | 0.6486111217074924 | 0.1289143827226427 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |
| D-FOU | F33-loss-warm-to-b1-consensus-b3-null | T2-cross-split-consensus | 0.6600163645214505 | 0.12876971728271908 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |
| D-FOU | F35-loss-warm-to-view-consistent-loss | T-other | 0.628508190313975 | 0.13369801971647474 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |
| D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.6469230850537618 | 0.14389371540811327 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |
| D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.6506590909428067 | 0.11701262990633647 | -0.012681000762515597 | 0.0 | TargetObservableNoRetention |

## KAN Source Writer

| carrier | v22_id | writer_family | h800 | h3200 | h4800 | h4800_positive_rows | source_bank_fraction | source_bank_retention_h4800_over_h3200 | KAN_FU_S3_v22_02 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | F36-lowbank-loss-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -1.6258274449242487 | -1.515865898794598 | -1.412504815393024 | 0 |  |  | 0 |
| D-CHE | F38-gain-gated-lowbank-loss-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -1.6219656997256808 | -1.5150579777028825 | -1.4138417343298595 | 0 |  |  | 0 |
| D-CHE | F68-adamw-boundary-to-gated-lowbank-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -0.0005275342199537489 | -0.0021484394868214927 | -0.007300949758953518 | 3 |  |  | 0 |
| D-CHE | F69-adamw-boundary-lowbank-anchor-antiwashout | D-CHE-F5b-low-degree-low-frequency-bank | 0.0014568898412916395 | -0.013673616780175103 | -0.015228016508950127 | 4 |  |  | 0 |
| D-CHE | KSW1-basis-estimate-readout-commit | D-CHE-F5a-readout-commit | -1.6231121751997206 | -1.5118636190891266 | -1.410119225581487 | 0 |  |  | 0 |
| D-CHE | KSW2-lowdegree-lowfreq-source-bank | D-CHE-F5b-low-degree-low-frequency-bank | -0.18236266242133248 | -0.005438006586498684 | 0.027257995473013982 | 4 |  |  | 0 |
| D-FOU | F36-lowbank-loss-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | -1.0049330062336392 | 0.049605203999413386 | 0.15526423851648966 | 9 |  |  | 0 |
| D-FOU | F38-gain-gated-lowbank-loss-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | -1.2923956910769145 | -0.06800315777460735 | 0.13637220197253758 | 8 |  |  | 0 |
| D-FOU | F68-adamw-boundary-to-gated-lowbank-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | 0.011607607205708822 | -0.004651433891720242 | -0.005101071463690864 | 2 |  |  | 0 |
| D-FOU | F69-adamw-boundary-lowbank-anchor-antiwashout | D-FOU-F5b-low-degree-low-frequency-bank | 0.0022645857599046496 | -0.01170006725523207 | -0.01502860254711575 | 3 |  |  | 0 |
| D-FOU | KSW1-basis-estimate-readout-commit | D-FOU-F5a-readout-commit | -1.536535296175215 | -1.4180500507354736 | -1.345011830329895 | 0 |  |  | 0 |
| D-FOU | KSW2-lowdegree-lowfreq-source-bank | D-FOU-F5b-low-degree-low-frequency-bank | -0.1767431596914927 | -0.10215700334972805 | -0.13887516657511392 | 1 |  |  | 0 |

## Failure Taxonomy

| class | count | action |
| --- | --- | --- |
| E2-DRATForwardEvalBlocked | 1 | continue Horner/reciprocal/telemetry-free repair |
| E3-DRBFDenseMaterializationBlocked | 1 | continue local-K/no-dense/active-center repair |
| F1-H3200ChainButH4800Collapse | 12 | terminal autopsy plus retained-target reset |
| F3-NoLegalPrecommitSelector | 1 | stop threshold tuning; redesign train-only target features |
| F4-TargetObservableNoRetention | 1 | target theory reset rather than actuator tuning |
| F5-KANSourceBankMismatch | 1 | MLP source decomposition and KAN bank mapping |

## 修改记录

- 修改 `experiments/run_v17_common.py` 与 `experiments/run_v21_01_common.py`：新增 h4000 horizon 落盘，满足 v22.02 terminal-collapse localization 要求。
- 新增 `dgkan/fu/terminal_collapse.py`、`precommit_selector.py`、`kan_source_bank.py`、`dgkan/metrics/calibration.py`：提供 v22.02 source-chain、selector legality、KAN source-bank 与 calibration unit tests。
- 新增 `dgkan/profiling/efficiency_v22_02.py` 与 kernel alias：重判 full-loop official / near-E1，不伪称 D-RAT/D-RBF official fused。
- 新增 v22.02 runner/finalizer：S0.9 clean packet gate、efficiency readback、D-RAT/D-RBF active repair readback、fresh terminal autopsy、legal precommit selector、target reset、KAN source writer、packet/bundle finalization。
- 扩展 MLP readout function-space target path：`MLPBaseline.frozen_readout_features()` 允许 exact readout actuation 对 MLP 2D readout 执行 train-only signal/reservoir target reset。
- 新增 `MLP-F124/F125/F126`：signal-reservoir target、source-bank target、dual target guard，作为 F118 之后的 target-theory reset continuation。
- 新增 `MLP-F127/F128/F129`：source-state adaptive raw、sparse source-axis、ratio-preserving terminal guard，用 train split A/B 与 corrupt-label gain 做无未来泄漏的 terminal candidate gate。
- 新增 `MLP-F130/F131/F132`：source-projected / noise-orthogonal / easy-margin B3-null consensus terminal target reset；主路径保持 F118 early100 h800 source slow-EMA，只在 terminal target gate 提交 train-only readout target。
- 新增 `MLP-F133/F134/F135`：terminal reject source-axis / slow-EMA / hold-source rescue；只在 train-stream selector 接受但 source-state gate 拒绝、且 density/balance/corrupt-gain 满足条件时提交微量 source rescue。
- 新增 `MLP-F136/F137/F138`：source-shape preservation、source-shape preservation + raw guard、source-shape preservation + terminal clamp；用于检验 h3200->h4800 衰减是否来自 midlate source-shape drift，而不改 productive h4800 判据。
- 新增 `MLP-F139/F140/F141`：terminal reject weak-row raw rescue、raw+source hybrid rescue、late-only raw rescue；只由 train-only source-state reject 触发，用于检验 F118 弱行是否可被 terminal micro-rescue 修复。
- 新增 `MLP-F142/F143/F144`：terminal top-k source-support guard、top-k raw hybrid guard、top-k debt-capped guard；按 FU-H6 建议只保护 train-stream source-state 的 top-k support subspace，用于检验 full source-vector projection 是否过强。

## 分析 / Insight / 结论

- v22.02 的 code gate 只有在 clean-unzip import closure 也通过时才可作为科学 no-go 的前提；最终 route 读取 clean packet 输出和 required artifact manifest。
- D-CHE/D-FOU 的 efficiency 证据来自 full-loop timing artifact，并按 v22.02 更严格 S1 规则重判；它不能单独支撑 functional promotion。
- D-RAT/D-RBF 已有 active repair component breakdown，但 near-E1 未达成时，functional smoke 继续 deferred。
- single-candidate old-shape fresh rerun 中 5 个 MLP 候选均形成 early source chain，但 h3200 retention ratio 未达 v22.02 continuous gate；h4000 记录显示 source 从 h3200 后继续衰减。
- Functional route 只看 grouped continuous h3200 与 grouped h4800 productive retention；没有 h4800 candidate 时，selector/independent confirmation 不允许升级为 promotion。
- 若 target reset 显示 high ActuationR2 但 h4800 retention 为 0，则 blocker 是 retained target/source-observability theory，而不是 actuator 本身。
- F106/F115/F117/F118/F119 已能形成 grouped early source chain 和 continuous h3200，但 F118 的 h4800_retention_ratio=0.4905266741203586 仍低于 0.50，不能 promotion；F119 stronger raw guard 下降到 0.4788671993982172。
- F120/F121/F122 terminal projected optimizer/source-blend/antiwashout 分别为 h4800_retention_ratio=0.47899749939111147 / 0.48388658936948814 / 0.4836718313083516；F123 h4000 checkpoint reentry 为 0.4799617647110513。它们都没有超过 F118，也没有形成 productive_h4800_group。
- F124/F125/F126 train-only signal-channel/reservoir target reset 分别为 h4800_retention_ratio=0.405378 / 0.382958 / 0.417708；它们没有超过 F118，也没有形成 productive_h4800_group。
- F127/F128/F129 source-state adaptive terminal guard 分别为 h4800_retention_ratio=0.401535 / 0.372177 / 0.400462；它们没有超过 F118，也没有形成 productive_h4800_group。
- F130/F131/F132 terminal target reset 把 h4800 从负值 terminal collapse 改成 positive but nonproductive retention：h4800_retention_ratio=0.419301 / 0.386605 / 0.405931，row_h4800_positive_count=8/9 / 9/9 / 9/9；但 productive_h4800_group 仍为 0。
- F133/F134/F135 terminal reject rescue 只触发少量 train-only micro-rescue，h4800_retention_ratio=0.419742 / 0.385211 / 0.405189；仍低于 F118，productive_h4800_group 仍为 0。
- F136/F137/F138 source-shape preserve family 未形成 early source chain：h800=-1.082650 / -1.107372 / -1.075262，h4800=-1.357646 / -1.385284 / -1.360441，early_source_chain_group=0，continuous_h3200_group=0。
- F139/F140/F141 terminal reject weak-row rescue 保留 early+h3200 chain，但 h4800_retention_ratio=0.419263 / 0.389215 / 0.420980，全部低于 F118 的 0.490527，productive_h4800_group 仍为 0。
- F142/F143/F144 top-k source-support family 保留 early+h3200 chain，但 h4800_retention_ratio=0.407347 / 0.390464 / 0.416659，全部低于 F118 的 0.490527，productive_h4800_group 仍为 0。
- 因此当前 evidence 不再支持继续 terminal projector、raw guard、h4000 reentry、signal/reservoir target、source-state adaptive、B3-null consensus target、terminal reject-rescue、weak-row rescue、source-shape preserve 或 top-k support 小修；需要新的 train-only retained-target/source-observability 理论。

- v22_02_code_review_packet.zip: size=4615862 sha256=7deffe269a27110257b422f89ace518514df29e186f02f3af0898434b031b39f
- v22_02_results_bundle.zip: size=14601440 sha256=15b1391552122cb407c308ae226bdcbacddf03236ac97f11d847a31f3d39b682
