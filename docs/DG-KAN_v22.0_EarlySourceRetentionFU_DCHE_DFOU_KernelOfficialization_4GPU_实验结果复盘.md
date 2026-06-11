# DG-KAN v22.0 EarlySourceRetentionFU DCHE/DFOU KernelOfficialization 4GPU 实验结果复盘

生成时间：2026-06-04 17:42:35 +0800

## Route

- route: `R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71F72F73F74SourceTheoryResetFailed`
- route_detail: S0_7_pass=1, D-CHE_E1_rows=6, D-FOU_E1_rows=3, D-CHE_S1_rows=6, D-FOU_S1_rows=3, candidate_early_chain=29, candidate_continuous=12, candidate_h3200=12, candidate_h4800=0, independent=0
- CodeRoute: S0.7-CodeMetricMechanismPassed
- EfficiencyRoute: D-CHE_E1=6;D-FOU_E1=3;D-CHE_S1=6;D-FOU_S1=3
- FunctionalRoute: early=29;continuous=12;h3200=12;h4800=0;independent=0
- promotion_allowed: 0

## 是否达成 v22 目标

没有达成。

- S0.7 preflight pass: 1
- D-CHE / D-FOU E1 rows: D-CHE_E1=6;D-FOU_E1=3;D-CHE_S1=6;D-FOU_S1=3
- source-chain rows/groups: 2169 / 82
- late rebound groups: 3
- source observability decision: TrainOnlySelectorCandidateNeedsHeldOut
- passing train-only predictors: 2
- passing legal C4 predictors: 2
- F46 selector decision: F46TrainLossSelectorHoldsOutAsRepairHypothesis
- F47 online repair decision: F47EarlyChainNoContinuousRetentionRatio
- F48 hold-fallback repair decision: F48EarlyChainNoContinuousRetentionRatio
- F49 tiny-fallback repair decision: F49ContinuousH3200ButH4800Failed
- F50 boosted tiny-fallback repair decision: F50EarlyChainNoContinuousRetentionRatio
- F51 late-hold recovery decision: F51ContinuousH3200ButH4800Failed
- F52 late lookahead-floor decision: F52ContinuousH3200ButH4800Failed
- F53 terminal lookahead-floor decision: F53ContinuousH3200ButH4800Failed
- F54 early-terminal lookahead-floor decision: F54ContinuousH3200ButH4800Failed
- F55 split-Fisher source reset decision: F55PhaseResetNoEarlyChain
- F56 momentum-warm split-Fisher source reset decision: F56PhaseResetNoEarlyChain
- F57 anti-washout source reset decision: F57EarlyChainNoContinuousRetentionH3200
- F58 source-anchor reset decision: F58PhaseResetNoEarlyChain
- F59 param-EMA reentry reset decision: F59EarlyChainNoContinuousRetentionRatio
- F60 readout-channel reset decision: F60EarlyChainNoContinuousRetentionH3200
- F61 hidden-matrix channel reset decision: F61EarlyChainNoContinuousRetentionRatio
- F62 terminal projected lookahead-floor decision: F62EarlyChainNoContinuousRetentionRatio
- F63 terminal consensus lookahead-floor decision: F63ContinuousH3200ButH4800Failed
- F64 terminal selector lookahead-floor decision: F64ContinuousH3200ButH4800Failed
- F65 AdamW split-Fisher residual decision: F65EarlyChainNoContinuousRetentionH3200
- F66 KAN readout-only commit decision: F66PhaseResetNoEarlyChain
- F67 KAN low-bank B3-null decision: F67PhaseResetNoEarlyChain
- F68 AdamW-boundary low-bank B3-null decision: F68PhaseResetNoEarlyChain
- F69 AdamW low-bank anchor anti-washout decision: F69EarlyChainNoContinuousRetentionH3200
- F70 terminal positive lookahead-floor decision: F70ContinuousH3200ButH4800Failed
- F71 terminal h3200 checkpoint reentry decision: F71ContinuousH3200ButH4800Failed
- F72 terminal hard-split source decision: F72ContinuousH3200ButH4800Failed
- F73 terminal AdamW-lookahead decision: F73ContinuousH3200ButH4800Failed
- F74 terminal optimizer-selector decision: F74ContinuousH3200ButH4800Failed
- required artifact missing count: 0

## S0.7 Truth Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| import_closure | 1 | compileall/import/source self-contained | 1 |  |
| source_chain | 1 | early/continuous/late formulas | 4/4 |  |
| linec | 1 | linec_fast/channel golden | 9/9;8/8 |  |
| update_semantics | 1 | finite nonzero update semantics | 15/15 |  |
| mechanism_contracts | 1 | planned mechanism contracts | 32/32 |  |
| efficiency_profiler | 1 | v22 E1/S1 gate smoke | 1 |  |

## Efficiency Evidence

| carrier | rows | E1_pass_rows | S1_pass_rows | E2_pass_rows | best_forward_ratio | best_step_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 12 | 6 | 6 | 3 | 0.9305108277155166 | 0.46682782835473463 |
| D-FOU | 12 | 3 | 3 | 3 | 1.0087315984225318 | 0.5089279836996342 |

## Source-Chain Evidence

| carrier | variant | v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h3200_group | productive_h4800_group | late_rebound_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | R0-current | MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | 0.048824826876322426 | 0.06716635492112902 | 0.37619424528545803 | 0.36653946505652535 | 0.16515672206878662 | -0.003963543309105767 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 9 | 0.033847285641564265 | 0.06886153750949436 | 0.374403801229265 | 0.36406707101398045 | 0.1626881096098158 | -0.005899740589989556 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F74-trainloss-terminal-optimizer-selector | 9 | 0.03304861651526557 | 0.06570586893293592 | 0.3710273702939351 | 0.3603704770406087 | 0.15898103184170193 | -0.011182712184058296 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | 9 | 0.05093061261706882 | 0.05545302232106527 | 0.36371467510859173 | 0.353307565053304 | 0.1519218815697564 | -0.016645153363545735 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry | 9 | 0.04618631468878852 | 0.08161209689246283 | 0.3614519172244602 | 0.2964131236076355 | 0.01199809710184733 | -0.1983597609731886 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F72-trainloss-terminal-hard-split-source | 9 | 0.04502135515213013 | 0.05726057953304715 | 0.3613293833202786 | 0.35068535804748535 | 0.1492934226989746 | -0.0192489226659139 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 9 | 0.04779497782389323 | 0.08172988891601562 | 0.3612903422779507 | 0.31988142596350777 | 0.11850794818666247 | -0.0509403215514289 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F73-trainloss-terminal-adamw-lookahead | 9 | 0.03758410612742106 | 0.047415720091925725 | 0.3590083122253418 | 0.3483833935525682 | 0.14700835280948216 | -0.02212838331858317 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source | 9 | 0.04634739955266317 | 0.07411068677902222 | 0.3579259382353889 | 0.3451966577106052 | 0.14383376969231498 | -0.024836275312635634 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F51-trainloss-late-hold-recovery-source | 9 | 0.038394649823506675 | 0.04882542954550849 | 0.35773270659976536 | 0.3475710948308309 | 0.14619363678826225 | -0.02271825737423367 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 9 | 0.055467969841427274 | 0.07662753926383124 | 0.35400940974553424 | 0.31965380907058716 | 0.11868382162517971 | -0.14789369371202257 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F54-trainloss-early-terminal-lookahead-floor-source | 9 | 0.041801889737447105 | 0.039883712927500405 | 0.3537594742245144 | 0.34387920962439644 | 0.14249980449676514 | -0.02585990561379327 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F47-trainloss-gated-dual-timescale-tiny-late-source | 9 | 0.04881612459818522 | 0.06548382176293267 | 0.35250912772284615 | 0.29007894463009304 | 0.010728200276692709 | -0.20400049289067587 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F52-trainloss-late-lookahead-floor-source | 9 | 0.0458407998085022 | 0.041891256968180336 | 0.35123587979210746 | 0.34043533934487236 | 0.13905479510625204 | -0.029730810059441462 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F64-trainloss-terminal-selector-lookahead-floor-source | 9 | 0.049836946858300105 | 0.06163607041041056 | 0.3511849774254693 | 0.33645888831880355 | 0.1350817879041036 | -0.03357837597529093 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F60-adamw-boundary-dual-timescale-readout-channel | 9 | 0.04834309551450941 | 0.07113626268174914 | 0.3485870957374573 | 0.2784839868545532 | -0.0033842987484402126 | -0.21282377507951525 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 9 | 0.03919836547639635 | 0.061279495557149254 | 0.34796935982174343 | 0.3149078090985616 | 0.1136280828052097 | -0.05511781242158678 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F63-trainloss-terminal-consensus-lookahead-floor-source | 9 | 0.03839343123965793 | 0.06572912136713664 | 0.34744171963797676 | 0.3360384636455112 | 0.13465211788813272 | -0.03395819001727634 | 1 | 1 | 1 | 0 | 0 | pass |
| MLP | R0-current | MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel | 9 | 0.0434488852818807 | 0.05788376596238878 | 0.34573060274124146 | 0.28615807824664646 | 0.011278331279754639 | -0.20057602061165702 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F40-adamw-boundary-to-momentum-source | 18 | 0.04405393203099569 | 0.05879485607147217 | 0.3393445677227444 | 0.2780708008342319 | 0.0010147227181328668 | -0.20703593227598402 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 9 | 0.04412646426094903 | 0.05342723925908407 | 0.3387697670194838 | 0.2748594085375468 | 0.0008061064614189996 | -0.21680072943369547 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F41-adamw-boundary-to-dual-timescale-source | 9 | 0.03346565696928236 | 0.060353484418657094 | 0.3372511797481113 | 0.3062930968072679 | 0.10489116774664985 | -0.06369584136539036 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source | 9 | 0.036270201206207275 | 0.058142774634891085 | 0.3347284330262078 | 0.32330578565597534 | 0.1219278375307719 | -0.04832035303115845 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source | 9 | 0.04760628938674927 | 0.05699806743197971 | 0.3310249249140422 | 0.3187437587314182 | 0.11736520131429036 | -0.05160053571065267 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F62-trainloss-terminal-projected-lookahead-floor-source | 9 | 0.03944614860746595 | 0.047740320364634194 | 0.3297286894586351 | 0.3189403149816725 | 0.11756877766715156 | -0.050098571512434215 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F57-adamw-boundary-dual-timescale-antiwashout-source | 9 | 0.01391528050104777 | 0.04101914829678006 | 0.29924317863252425 | 0.2694499691327413 | -0.012103550963931613 | -0.2369269993570116 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F65-adamw-split-fisher-residual-source | 9 | 0.04382287131415473 | 0.06983120573891534 | 0.0827075441678365 | -0.25741936763127643 | -0.779679642783271 | -1.1511565181944106 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F68-adamw-boundary-to-gated-lowbank-b3-null | 9 | 0.008326477474636503 | 0.010237760014004178 | 0.011607607205708822 | 0.0015850961208343506 | -0.004651433891720242 | -0.005101071463690864 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F69-adamw-boundary-lowbank-anchor-antiwashout | 9 | 0.007455190022786458 | 0.00037747621536254883 | 0.0014568898412916395 | -0.0052944521109263105 | -0.013673616780175103 | -0.015228016508950127 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-AdamW | 54 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | 54 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0 | 0 | 0 | 0 | ContinuousRetentionMissing |
| MLP | R0-current | MLP-F1-M2-strong-source | 9 | -1.003287394841512 | -0.03091594245698717 | 0.45738417241308427 | 0.5034836464458041 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | MLP-F11-dual-timescale-retention-warm1200 | 9 | -1.0199487739139133 | -0.0166345304912991 | 0.4564314021004571 | 0.5333339505725436 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | MLP-F23-source-vs-sgd-lookahead-gate | 9 | -0.972764770189921 | -0.08067854907777575 | 0.4232677287525601 | 0.46392786502838135 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | MLP-F58-adamw-boundary-dual-timescale-source-anchor | 9 | -0.02303592363993327 | -0.023477594057718914 | 0.23931911256578234 | 0.20521441433164808 | -0.08447160323460896 | -0.29526517788569134 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | MLP-F56-momentum-warm-split-fisher-source | 9 | -0.361447188589308 | 0.38012144300672746 | 0.1787186066309611 | -0.2631121476491292 | -0.563570585515764 | -0.5396582616700066 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F69-adamw-boundary-lowbank-anchor-antiwashout | 9 | 0.0104879273308648 | -0.004357702202267117 | 0.0022645857599046496 | -0.007774823241763645 | -0.01170006725523207 | -0.01502860254711575 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F68-adamw-boundary-to-gated-lowbank-b3-null | 9 | 0.007974889543321397 | -0.0018029676543341742 | -0.0005275342199537489 | 0.00357857346534729 | -0.0021484394868214927 | -0.007300949758953518 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | CTRL-AdamW | 270 | 0.0 | -0.0047626676382841885 | -0.014133324004985667 | -0.4073572172058953 | -0.9892447933791175 | -1.3887088693796643 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | MLP-F55-split-fisher-source-observability-reset | 9 | -0.8327359490924411 | -0.2598619858423869 | -0.10597315099504259 | -0.0066691239674886065 | -0.02089711692598131 | -0.04255129893620809 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F8-KSW2-earlyboost-alt25 | 9 | 0.013114677535163032 | -0.08508913384543525 | -0.1133241057395935 | -0.11987284488148159 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F25-loss-warm-to-b1-consensus-migration | 9 | -0.5565746360354953 | -0.361707435713874 | -0.15561231639650133 | -0.024669541252983943 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F39-loss-warm-to-gated-lowbank-loss-b3-null | 9 | -0.5683881706661649 | -0.36914222770267063 | -0.1582929425769382 | -0.026102377308739558 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F33-loss-warm-to-b1-consensus-b3-null | 9 | -0.572408119837443 | -0.3851510551240709 | -0.16058513191011217 | -0.031747274928622775 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F8-KSW2-earlyboost-alt25 | 9 | -0.2652822732925415 | -0.16785234212875366 | -0.16301078928841484 | -0.18363457918167114 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T1-loss-cotangent-target | 9 | -0.5758589373694526 | -0.3785022762086656 | -0.16423779726028442 | -0.07906320359971789 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F30-gain-gated-loss-warm-b1-consensus | 9 | -0.5704515245225694 | -0.3807315296596951 | -0.16896929343541464 | -0.04007487164603339 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F33-loss-warm-to-b1-consensus-b3-null | 9 | -0.3018566370010376 | -0.31244563394122654 | -0.16903516981336805 | -0.05690753128793505 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F37-loss-warm-to-lowbank-loss-b3-null | 9 | -0.5649446646372477 | -0.38460779190063477 | -0.1727074384689331 | -0.026365160942077637 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F35-loss-warm-to-view-consistent-loss | 9 | -0.5819652345445421 | -0.3911263942718506 | -0.1751935150888231 | -0.046722100840674505 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW2-lowdegree-lowfreq-source-bank | 18 | -0.5546180937025282 | -0.3652562002340953 | -0.1767431596914927 | -0.09392255875799391 | -0.10215700334972805 | -0.13887516657511392 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F39-loss-warm-to-gated-lowbank-loss-b3-null | 9 | -0.32272589206695557 | -0.3479761481285095 | -0.17721559604008993 | -0.05555439326498243 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW2-lowdegree-lowfreq-source-bank | 18 | -0.3112146854400635 | -0.34383275773790145 | -0.18236266242133248 | -0.07237915032439762 | -0.005438006586498684 | 0.027257995473013982 | 0 | 0 | 0 | 0 | 1 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-CHE | CHE-R4-k3-gradbuf-triton | F37-loss-warm-to-lowbank-loss-b3-null | 9 | -0.3142985635333591 | -0.34857600927352905 | -0.18425064616733128 | -0.052726265456941396 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T1-loss-cotangent-target | 9 | -0.335018965933058 | -0.355637874868181 | -0.18799111578199598 | -0.057160367568333946 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F30-gain-gated-loss-warm-b1-consensus | 9 | -0.3151562346352471 | -0.367200579908159 | -0.19381692674424914 | -0.06285006139013502 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F35-loss-warm-to-view-consistent-loss | 9 | -0.32470248805152047 | -0.37968626949522233 | -0.19568810198042128 | -0.05814902981122335 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F25-loss-warm-to-b1-consensus-migration | 9 | -0.3284727864795261 | -0.3621560268931919 | -0.2032026383611891 | -0.06934237149026659 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F6-KSW2-density-smallstep-alt50 | 9 | -0.6515551937950982 | -0.8101509809494019 | -0.3558090594079759 | -0.08882333172692193 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F10-T7-b1-cross-split-consensus-transfer | 9 | -0.6235531700981988 | -0.761064714855618 | -0.3950337635146247 | -0.09203914139005873 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-smallstep-alt50 | 9 | -0.4074300395117866 | -0.7993055979410807 | -0.3987191518147786 | -0.1072934369246165 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F10-T7-b1-cross-split-consensus-transfer | 9 | -0.35107966264088947 | -0.7804214225875007 | -0.49972931543986004 | -0.16040034095446268 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T5-random-matched-target | 9 | -0.7079783810509576 | -1.1000941461986966 | -0.580721398194631 | -0.14567995071411133 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | CTRL-SGD | 270 | -1.4507989554493517 | -1.1199293083614774 | -0.6033495331252062 | -0.23496702600408484 | -0.010873916130217294 | 0.0 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | MLP-F2-M15-weak-stable | 9 | -1.4449443022410076 | -1.171965135468377 | -0.6692412826750014 | -0.2377561397022671 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T5-random-matched-target | 9 | -0.4879032373428345 | -1.275543404950036 | -0.8925885491900973 | -0.3363882137669457 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F36-lowbank-loss-b3-null | 9 | -0.6718335416581895 | -1.3025149703025818 | -1.0049330062336392 | -0.3352172374725342 | 0.049605203999413386 | 0.15526423851648966 | 0 | 0 | 0 | 0 | 1 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| MLP | R0-current | CTRL-NoOpMatchedOverhead | 270 | -1.497299783980405 | -1.35879672854035 | -1.0707866149920005 | -1.0840966825132017 | -1.285798700792449 | -1.4480590193517624 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | R0-current | CTRL-RandomMatchedNorm | 270 | -1.502781777690958 | -1.3643856697612338 | -1.0763936407036252 | -1.0898048612806532 | -1.290993305189269 | -1.4529662215047412 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F38-gain-gated-lowbank-loss-b3-null | 9 | -0.6818294260236952 | -1.3823313117027283 | -1.2923956910769145 | -0.7170495722028944 | -0.06800315777460735 | 0.13637220197253758 | 0 | 0 | 0 | 0 | 1 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW1-basis-estimate-readout-commit | 9 | -0.6899531417422824 | -1.4369259344206915 | -1.536535296175215 | -1.5015311241149902 | -1.4180500507354736 | -1.345011830329895 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-TCTRL-stable-random-target | 9 | -0.7062974241044786 | -1.4442483981450398 | -1.5403718882136874 | -1.5024392472373114 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | 54 | -0.6976539470531322 | -1.4411829445097182 | -1.5415957150635895 | -1.506245055684337 | -1.416989588075214 | -1.3422040541966755 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | 54 | -0.6965172070044058 | -1.4412056649172749 | -1.5431656042734783 | -1.5108974896095417 | -1.4270823101202648 | -1.3593043751186795 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | 54 | -0.6973756949106852 | -1.4420647223790486 | -1.5440239773856268 | -1.5117560128370922 | -1.428962293598387 | -1.361184557278951 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F38-gain-gated-lowbank-loss-b3-null | 9 | -0.4849250581529405 | -1.4243185387717352 | -1.6219656997256808 | -1.6307863593101501 | -1.5150579777028825 | -1.4138417343298595 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW1-basis-estimate-readout-commit | 9 | -0.48900026745266384 | -1.427678492334154 | -1.6231121751997206 | -1.6301616761419508 | -1.5118636190891266 | -1.410119225581487 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-TCTRL-stable-random-target | 9 | -0.48025378915998673 | -1.4265883564949036 | -1.6247111161549885 | -1.622945682870017 | nan | nan | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F36-lowbank-loss-b3-null | 9 | -0.48682689666748047 | -1.4254724317126803 | -1.6258274449242487 | -1.6340082155333624 | -1.515865898794598 | -1.412504815393024 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-SGD | 54 | -0.48823900355233085 | -1.4294601380825043 | -1.6273371720755543 | -1.632532403424934 | -1.5211187253395717 | -1.4211396392848756 | 0 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

_仅显示前 80 / 82 rows；完整 CSV 见 artifact。_

## C4 Source Observability Audit

- decision: `TrainOnlySelectorCandidateNeedsHeldOut`
- audit rows / candidate rows / candidate early rows: 2169 / 657 / 138
- passing train-only predictors: 2

| predictor | kind | finite_rows | positive_rows | continuous_positive_rows | AUC_early_chain | AUC_continuous_retention | dataset_min_AUC | carrier_min_AUC | Spearman_h800 | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_loss_h100 | train_only | 657 | 138 | 90 | 0.8351763424646058 | 0.95480109739369 | 0.7546182266009852 | 0.5828446976042888 | 0.8112847722701615 | 0 |
| train_loss_h400 | train_only | 657 | 138 | 90 | 0.8796598810421379 | 0.9409073094258279 | 0.7604679802955665 | 0.7118864131345284 | 0.8289038465859395 | 1 |
| train_loss_h800 | train_only | 657 | 138 | 90 | 0.8911368015414258 | 0.9434548304918675 | 0.7749384236453202 | 0.7252889931311778 | 0.8081660880956663 | 1 |
| train_loss_drop_h100_h800 | train_only | 657 | 138 | 90 | 0.363519588953115 | 0.21192435822065453 | 0.26086229946524064 | 0.473159509202454 | -0.10045996286103 | 0 |
| CEp99_h800 | audit_readback | 657 | 138 | 90 | 0.8758900896372623 | 0.7458847736625515 | 0.6796457219251337 | 0.5897239263803681 | 0.8342031213013084 | 0 |
| Brier_h800 | audit_readback | 657 | 138 | 90 | 0.6773617045042026 | 0.9610523221634333 | 0.4291871921182266 | 0.4659490701960127 | 0.5725868050352597 | 0 |
| LineC_channel_loss_h800 | audit_readback | 657 | 138 | 90 | 0.5169012314651923 | 0.6731040564373898 | 0.3669950738916256 | 0.4523809523809524 | -0.024246211201850696 | 0 |
| ActuationR2_h800 | function_readback | 324 | 138 | 90 | 0.21841417368573918 | 0.24161425576519915 | 0.20226308345120225 | 0.2012987012987013 | 0.21755869718175463 | 0 |
| B2_transfer_gain_h800 | function_readback | 324 | 138 | 90 | 0.17920418239907057 | 0.19916142557651992 | 0.15625 | 0.1444805194805195 | 0.1905614379580188 | 0 |
| source_channel_projection_h800 | train_stream_diagnostic | 324 | 138 | 90 | 0.08742375835027592 | 0.08438155136268344 | 0.07211538461538461 | 0.1090146750524109 | 0.12098817930043765 | 0 |
| source_state_gate_accept_h800 | train_stream_diagnostic | 459 | 138 | 90 | 0.5078897467154273 | 0.5266937669376693 | 0.4861828512396694 | 0.25 | 0.20760440622605172 | 0 |
| generalization_gate_accept_h800 | train_stream_diagnostic | 441 | 138 | 90 | 0.5071852584714321 | 0.5279563608026495 | 0.4841880341880342 | 0.25 | 0.2028265140847957 | 0 |
| C4_E1_split_fisher_snr_h800 | train_stream_diagnostic | 0 | 138 | 90 |  |  |  |  |  | 0 |
| C4_E6_split_fisher_coherence_h800 | train_stream_diagnostic | 0 | 138 | 90 |  |  |  |  |  | 0 |
| C4_E7_noise_reservoir_separation_h800 | train_stream_diagnostic | 0 | 138 | 90 |  |  |  |  |  | 0 |
| source_h400 | source_readback_forbidden | 2169 | 481 | 90 | 0.964513479027697 | 0.8193121693121693 | 0.9449523101697015 | 0.9530774496852487 | 0.9339522043425939 | 0 |

## F46 Train-Loss Selector Held-Out Validation

- decision: `F46TrainLossSelectorHoldsOutAsRepairHypothesis`
- fit / holdout rows: 45 / 18
- best predictor: train_loss_h400
- threshold raw / score: 0.020872684195637703 / -0.020872684195637703
- holdout selected rows / continuous rows: 8 / 6
- holdout selected h100/h400/h800/h1600/h3200/h4800: 0.02573563903570175 / 0.026670433580875397 / 0.23333857953548431 / 0.3243083208799362 / 0.2754524126648903 / 0.06581832468509674
- holdout group early / continuous / h4800: 1 / 1 / 1
- next repair recommended: 1

| predictor | fit_rows | fit_early_rows | fit_continuous_rows | threshold_raw | tp | fp | fn | tn | precision | recall | f1 | youden |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_loss_h100 | 45 | 22 | 13 | 0.4754267930984497 | 12 | 3 | 1 | 29 | 0.8 | 0.9230769230769231 | 0.8571428571428571 | 0.8293269230769231 |
| train_loss_h400 | 45 | 22 | 13 | 0.020872684195637703 | 12 | 8 | 1 | 24 | 0.6 | 0.9230769230769231 | 0.7272727272727274 | 0.6730769230769231 |
| train_loss_h800 | 45 | 22 | 13 | 0.016799673438072205 | 12 | 8 | 1 | 24 | 0.6 | 0.9230769230769231 | 0.7272727272727274 | 0.6730769230769231 |

| split | predictor | rows | continuous_rows_all | selected_rows | selected_continuous_rows | selected_h800 | selected_h1600 | selected_h3200 | selected_h4800 | selected_group_continuous | selected_group_h4800 | selected_group_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fit_f40_f43 | train_loss_h100 | 45 | 13 | 15 | 12 | 0.134438423315684 | 0.27504361867904664 | 0.36873523394266766 | 0.14794831971327463 | 1 | 1 | EarlySourceChainMissing |
| holdout_f44_f45 | train_loss_h100 | 18 | 6 | 6 | 6 | 0.13508603970209757 | 0.2749844392140706 | 0.3650663395722707 | 0.13764872153600058 | 1 | 1 | EarlySourceChainMissing |
| all_f40_f45 | train_loss_h100 | 63 | 19 | 21 | 18 | 0.13462345656894503 | 0.2750267102604821 | 0.3675122691525353 | 0.14451512032084995 | 1 | 1 | EarlySourceChainMissing |
| fit_f40_f43 | train_loss_h400 | 45 | 13 | 20 | 12 | 0.2247294992208481 | 0.31347682178020475 | 0.26442690938711166 | 0.06123514845967293 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h400 | 18 | 6 | 8 | 6 | 0.23333857953548431 | 0.3243083208799362 | 0.2754524126648903 | 0.06581832468509674 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | 63 | 19 | 28 | 18 | 0.22718923645360128 | 0.31657153580869946 | 0.2681020771463712 | 0.06276287386814754 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h800 | 45 | 13 | 20 | 12 | 0.2247294992208481 | 0.31347682178020475 | 0.26442690938711166 | 0.06123514845967293 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h800 | 18 | 6 | 8 | 6 | 0.23333857953548431 | 0.3243083208799362 | 0.2754524126648903 | 0.06581832468509674 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h800 | 63 | 19 | 28 | 18 | 0.22718923645360128 | 0.31657153580869946 | 0.2681020771463712 | 0.06276287386814754 | 1 | 1 | pass |

| split | predictor | carrier | v22_id | rows | continuous_rows | h800 | h1600 | h3200 | h4800 | continuous_retention_group | productive_h4800_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fit_f40_f43 | train_loss_h100 | MLP | MLP-F40-adamw-boundary-to-momentum-source | 6 | 3 | 0.1276391545931498 | 0.269844392935435 | 0.36424771944681805 | 0.14691887299219766 | 1 | 1 | EarlySourceChainMissing |
| fit_f40_f43 | train_loss_h100 | MLP | MLP-F41-adamw-boundary-to-dual-timescale-source | 3 | 3 | 0.14879363775253296 | 0.287678857644399 | 0.3777689536412557 | 0.15085750818252563 | 1 | 1 | EarlySourceChainMissing |
| fit_f40_f43 | train_loss_h100 | MLP | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 3 | 3 | 0.1296302080154419 | 0.2700020670890808 | 0.3650001684824626 | 0.14825389782587686 | 1 | 1 | EarlySourceChainMissing |
| fit_f40_f43 | train_loss_h100 | MLP | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 3 | 3 | 0.1384899616241455 | 0.27784838279088336 | 0.3679240942001343 | 0.14576299985249838 | 1 | 1 | EarlySourceChainMissing |
| holdout_f44_f45 | train_loss_h100 | MLP | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 3 | 3 | 0.15671980381011963 | 0.29398735364278156 | 0.384072204430898 | 0.1566838026046753 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h100 | MLP | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 3 | 3 | 0.11345227559407552 | 0.2559815247853597 | 0.3460604747136434 | 0.11861364046732585 | 1 | 1 | EarlySourceChainMissing |
| all_f40_f45 | train_loss_h100 | MLP | MLP-F40-adamw-boundary-to-momentum-source | 6 | 3 | 0.1276391545931498 | 0.269844392935435 | 0.36424771944681805 | 0.14691887299219766 | 1 | 1 | EarlySourceChainMissing |
| all_f40_f45 | train_loss_h100 | MLP | MLP-F41-adamw-boundary-to-dual-timescale-source | 3 | 3 | 0.14879363775253296 | 0.287678857644399 | 0.3777689536412557 | 0.15085750818252563 | 1 | 1 | EarlySourceChainMissing |
| all_f40_f45 | train_loss_h100 | MLP | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 3 | 3 | 0.1296302080154419 | 0.2700020670890808 | 0.3650001684824626 | 0.14825389782587686 | 1 | 1 | EarlySourceChainMissing |
| all_f40_f45 | train_loss_h100 | MLP | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 3 | 3 | 0.1384899616241455 | 0.27784838279088336 | 0.3679240942001343 | 0.14576299985249838 | 1 | 1 | EarlySourceChainMissing |
| all_f40_f45 | train_loss_h100 | MLP | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 3 | 3 | 0.15671980381011963 | 0.29398735364278156 | 0.384072204430898 | 0.1566838026046753 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h100 | MLP | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 3 | 3 | 0.11345227559407552 | 0.2559815247853597 | 0.3460604747136434 | 0.11861364046732585 | 1 | 1 | EarlySourceChainMissing |
| fit_f40_f43 | train_loss_h400 | MLP | MLP-F40-adamw-boundary-to-momentum-source | 8 | 3 | 0.22889168560504913 | 0.31728510558605194 | 0.26920461654663086 | 0.07073266804218292 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h400 | MLP | MLP-F41-adamw-boundary-to-dual-timescale-source | 4 | 3 | 0.2325826734304428 | 0.3227091282606125 | 0.27386224269866943 | 0.06525267660617828 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h400 | MLP | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 4 | 3 | 0.20861941576004028 | 0.29682759940624237 | 0.25021979212760925 | 0.052545562386512756 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h400 | MLP | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 4 | 3 | 0.22466203570365906 | 0.3132771700620651 | 0.2644209861755371 | 0.05640968680381775 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h400 | MLP | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 4 | 3 | 0.2483004927635193 | 0.33764931559562683 | 0.2887967675924301 | 0.07899987697601318 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h400 | MLP | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 4 | 3 | 0.21837666630744934 | 0.3109673261642456 | 0.26210805773735046 | 0.0526367723941803 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | MLP | MLP-F40-adamw-boundary-to-momentum-source | 8 | 3 | 0.22889168560504913 | 0.31728510558605194 | 0.26920461654663086 | 0.07073266804218292 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | MLP | MLP-F41-adamw-boundary-to-dual-timescale-source | 4 | 3 | 0.2325826734304428 | 0.3227091282606125 | 0.27386224269866943 | 0.06525267660617828 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | MLP | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 4 | 3 | 0.20861941576004028 | 0.29682759940624237 | 0.25021979212760925 | 0.052545562386512756 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | MLP | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 4 | 3 | 0.22466203570365906 | 0.3132771700620651 | 0.2644209861755371 | 0.05640968680381775 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | MLP | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 4 | 3 | 0.2483004927635193 | 0.33764931559562683 | 0.2887967675924301 | 0.07899987697601318 | 1 | 1 | pass |
| all_f40_f45 | train_loss_h400 | MLP | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 4 | 3 | 0.21837666630744934 | 0.3109673261642456 | 0.26210805773735046 | 0.0526367723941803 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h800 | MLP | MLP-F40-adamw-boundary-to-momentum-source | 8 | 3 | 0.22889168560504913 | 0.31728510558605194 | 0.26920461654663086 | 0.07073266804218292 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h800 | MLP | MLP-F41-adamw-boundary-to-dual-timescale-source | 4 | 3 | 0.2325826734304428 | 0.3227091282606125 | 0.27386224269866943 | 0.06525267660617828 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h800 | MLP | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 4 | 3 | 0.20861941576004028 | 0.29682759940624237 | 0.25021979212760925 | 0.052545562386512756 | 1 | 1 | pass |
| fit_f40_f43 | train_loss_h800 | MLP | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 4 | 3 | 0.22466203570365906 | 0.3132771700620651 | 0.2644209861755371 | 0.05640968680381775 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h800 | MLP | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 4 | 3 | 0.2483004927635193 | 0.33764931559562683 | 0.2887967675924301 | 0.07899987697601318 | 1 | 1 | pass |
| holdout_f44_f45 | train_loss_h800 | MLP | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 4 | 3 | 0.21837666630744934 | 0.3109673261642456 | 0.26210805773735046 | 0.0526367723941803 | 1 | 1 | pass |

_仅显示前 30 / 36 rows；完整 CSV 见 artifact。_

## F40 AdamW Boundary Phase Reset

- decision: `F40EarlyChainNoContinuousRetentionH3200`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600: 0.04405393203099569 / 0.05879485607147217 / 0.3393445677227444 / 0.2780708008342319
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F40-adamw-boundary-to-momentum-source | 9 | 0 | 0.04405393203099569 | 0.05879485607147217 | 0.3393445677227444 | 0.2780708008342319 | 0.0010147227181328668 | -0.20703593227598402 | 0.819434955745114 | 0.003649152356481254 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F41 Dual-Timescale Phase Reset

- decision: `F41EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.03346565696928236 / 0.060353484418657094 / 0.3372511797481113 / 0.3062930968072679 / 0.10489116774664985 / -0.06369584136539036
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.9082046711772347 / 0.3424535807042744 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F41-adamw-boundary-to-dual-timescale-source | 9 | 0 | 0.03346565696928236 | 0.060353484418657094 | 0.3372511797481113 | 0.3062930968072679 | 0.10489116774664985 | -0.06369584136539036 | 0.9082046711772347 | 0.3424535807042744 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F42 Dual-Timescale With SGD Floor

- decision: `F42EarlyChainNoContinuousRetentionH3200`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.04412646426094903 / 0.05342723925908407 / 0.3387697670194838 / 0.2748594085375468 / 0.0008061064614189996 / -0.21680072943369547
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.8113457436174897 / 0.0029327955906915315 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 9 | 0 | 0.04412646426094903 | 0.05342723925908407 | 0.3387697670194838 | 0.2748594085375468 | 0.0008061064614189996 | -0.21680072943369547 | 0.8113457436174897 | 0.0029327955906915315 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F43 Late SGD Floor After H3200

- decision: `F43EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.055467969841427274 / 0.07662753926383124 / 0.35400940974553424 / 0.31965380907058716 / 0.11868382162517971 / -0.14789369371202257
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.902952860208879 / 0.37128861992998025 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 9 | 0 | 0.055467969841427274 | 0.07662753926383124 | 0.35400940974553424 | 0.31965380907058716 | 0.11868382162517971 | -0.14789369371202257 | 0.902952860208879 | 0.37128861992998025 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F44 Tiny Late SGD Floor

- decision: `F44EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.04779497782389323 / 0.08172988891601562 / 0.3612903422779507 / 0.31988142596350777 / 0.11850794818666247 / -0.0509403215514289
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.8853860414497714 / 0.3704746151787566 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 9 | 0 | 0.04779497782389323 | 0.08172988891601562 | 0.3612903422779507 | 0.31988142596350777 | 0.11850794818666247 | -0.0509403215514289 | 0.8853860414497714 | 0.3704746151787566 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F45 Reinforced Tiny Late Floor

- decision: `F45EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.03919836547639635 / 0.061279495557149254 / 0.34796935982174343 / 0.3149078090985616 / 0.1136280828052097 / -0.05511781242158678
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.9049871783535237 / 0.36082967624866286 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 9 | 0 | 0.03919836547639635 | 0.061279495557149254 | 0.34796935982174343 | 0.3149078090985616 | 0.1136280828052097 | -0.05511781242158678 | 0.9049871783535237 | 0.36082967624866286 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F47 Train-Loss Gated Late Retention Repair

- decision: `F47EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.04881612459818522 / 0.06548382176293267 / 0.35250912772284615 / 0.29007894463009304 / 0.010728200276692709 / -0.20400049289067587
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.8228976835407289 / 0.0369837262417417 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F47-trainloss-gated-dual-timescale-tiny-late-source | 9 | 0 | 0.04881612459818522 | 0.06548382176293267 | 0.35250912772284615 | 0.29007894463009304 | 0.010728200276692709 | -0.20400049289067587 | 0.8228976835407289 | 0.0369837262417417 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F48 Train-Loss Gate With Hold Fallback

- decision: `F48EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.04760628938674927 / 0.05699806743197971 / 0.3310249249140422 / 0.3187437587314182 / 0.11736520131429036 / -0.05160053571065267
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.9628995726354653 / 0.3682117628950512 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source | 9 | 0 | 0.04760628938674927 | 0.05699806743197971 | 0.3310249249140422 | 0.3187437587314182 | 0.11736520131429036 | -0.05160053571065267 | 0.9628995726354653 | 0.3682117628950512 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F49 Train-Loss Gate With Tiny Late Fallback

- decision: `F49ContinuousH3200ButH4800Failed`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.04634739955266317 / 0.07411068677902222 / 0.3579259382353889 / 0.3451966577106052 / 0.14383376969231498 / -0.024836275312635634
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.9644359931343889 / 0.4166719650365146 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 1 / 1 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source | 9 | 0 | 0.04634739955266317 | 0.07411068677902222 | 0.3579259382353889 | 0.3451966577106052 | 0.14383376969231498 | -0.024836275312635634 | 0.9644359931343889 | 0.4166719650365146 | 0.0 | 1 | 1 | pass |

## F50 Boosted Tiny Late Fallback

- decision: `F50EarlyChainNoContinuousRetentionRatio`
- rows / groups: 45 / 5
- candidate h100/h400/h800/h1600/h3200/h4800: 0.036270201206207275 / 0.058142774634891085 / 0.3347284330262078 / 0.32330578565597534 / 0.1219278375307719 / -0.04832035303115845
- retention ratios h1600/h800, h3200/h1600, h4800/h3200: 0.965874881715417 / 0.3771285357092663 / 0.0
- candidate early/continuous/h3200/h4800: 1 / 0 / 0 / 0

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source | 9 | 0 | 0.036270201206207275 | 0.058142774634891085 | 0.3347284330262078 | 0.32330578565597534 | 0.1219278375307719 | -0.04832035303115845 | 0.965874881715417 | 0.3771285357092663 | 0.0 | 1 | 0 | ContinuousRetentionMissing |

## F51-F54 Late-Phase Source Preservation / Terminal Recovery

- F51 decision: `F51ContinuousH3200ButH4800Failed`; h3200/h4800: 0.14619363678826225 / -0.02271825737423367
- F52 decision: `F52ContinuousH3200ButH4800Failed`; h3200/h4800: 0.13905479510625204 / -0.029730810059441462
- F53 decision: `F53ContinuousH3200ButH4800Failed`; h3200/h4800: 0.16515672206878662 / -0.003963543309105767
- F54 decision: `F54ContinuousH3200ButH4800Failed`; h3200/h4800: 0.14249980449676514 / -0.02585990561379327

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F51-trainloss-late-hold-recovery-source | 9 | 0 | 0.038394649823506675 | 0.04882542954550849 | 0.35773270659976536 | 0.3475710948308309 | 0.14619363678826225 | -0.02271825737423367 | 0.971594401122782 | 0.4206150596597146 | 0.0 | 1 | 1 | pass |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F52-trainloss-late-lookahead-floor-source | 9 | 0 | 0.0458407998085022 | 0.041891256968180336 | 0.35123587979210746 | 0.34043533934487236 | 0.13905479510625204 | -0.029730810059441462 | 0.9692498942487657 | 0.4084616931187185 | 0.0 | 1 | 1 | pass |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | 0 | 0.048824826876322426 | 0.06716635492112902 | 0.37619424528545803 | 0.36653946505652535 | 0.16515672206878662 | -0.003963543309105767 | 0.9743356514621679 | 0.4505837373973283 | 0.0 | 1 | 1 | pass |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F54-trainloss-early-terminal-lookahead-floor-source | 9 | 0 | 0.041801889737447105 | 0.039883712927500405 | 0.3537594742245144 | 0.34387920962439644 | 0.14249980449676514 | -0.02585990561379327 | 0.9720706714024359 | 0.41438912417069695 | 0.0 | 1 | 1 | pass |

## F55-F56 C4 Split-Fisher Source Observability Reset

- F55 decision: `F55PhaseResetNoEarlyChain`; h800/h1600/h3200/h4800: -0.10597315099504259 / -0.0066691239674886065 / -0.02089711692598131 / -0.04255129893620809
- F56 decision: `F56PhaseResetNoEarlyChain`; h800/h1600/h3200/h4800: 0.1787186066309611 / -0.2631121476491292 / -0.563570585515764 / -0.5396582616700066

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | -0.14288002914852566 | -0.4092753595776028 | -0.8489522139231364 | -1.3210341334342957 | -1.5022942291365728 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.8772423399819268 | -0.47509414619869655 | -0.5230130354563395 | -0.7265468041102091 | -0.9348907272020975 | -0.9276053176985847 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -0.8826241625679864 | -0.48051899009280735 | -0.5285689036051432 | -0.7313359445995755 | -0.9396731654802958 | -0.9326130880249871 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -0.8311372995376587 | -0.24080322848425972 | -0.07520310084025066 | -0.0017648802863226996 | 0.0 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F55-split-fisher-source-observability-reset | 9 | 0 | -0.8327359490924411 | -0.2598619858423869 | -0.10597315099504259 | -0.0066691239674886065 | -0.02089711692598131 | -0.04255129893620809 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F56-momentum-warm-split-fisher-source | 9 | 0 | -0.361447188589308 | 0.38012144300672746 | 0.1787186066309611 | -0.2631121476491292 | -0.563570585515764 | -0.5396582616700066 | 0.0 |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | -0.14288002914852566 | -0.4092753595776028 | -0.8489522139231364 | -1.3210341334342957 | -1.5022942291365728 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.8772423399819268 | -0.47509414619869655 | -0.5230130354563395 | -0.7265468041102091 | -0.9348907272020975 | -0.9276053176985847 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -0.8826241625679864 | -0.48051899009280735 | -0.5285689036051432 | -0.7313359445995755 | -0.9396731654802958 | -0.9326130880249871 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -0.8311372995376587 | -0.24080322848425972 | -0.07520310084025066 | -0.0017648802863226996 | 0.0 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F55-split-fisher-source-observability-reset | 9 | 0 | -0.8327359490924411 | -0.2598619858423869 | -0.10597315099504259 | -0.0066691239674886065 | -0.02089711692598131 | -0.04255129893620809 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F56-momentum-warm-split-fisher-source | 9 | 0 | -0.361447188589308 | 0.38012144300672746 | 0.1787186066309611 | -0.2631121476491292 | -0.563570585515764 | -0.5396582616700066 | 0.0 |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

## F57-F65 Source-Theory Reset

- F57 decision: `F57EarlyChainNoContinuousRetentionH3200`; h800/h1600/h3200/h4800: 0.29924317863252425 / 0.2694499691327413 / -0.012103550963931613 / -0.2369269993570116
- F58 decision: `F58PhaseResetNoEarlyChain`; h800/h1600/h3200/h4800: 0.23931911256578234 / 0.20521441433164808 / -0.08447160323460896 / -0.29526517788569134
- F59 decision: `F59EarlyChainNoContinuousRetentionRatio`; h800/h1600/h3200/h4800: 0.3614519172244602 / 0.2964131236076355 / 0.01199809710184733 / -0.1983597609731886
- F60 decision: `F60EarlyChainNoContinuousRetentionH3200`; h800/h1600/h3200/h4800: 0.3485870957374573 / 0.2784839868545532 / -0.0033842987484402126 / -0.21282377507951525
- F61 decision: `F61EarlyChainNoContinuousRetentionRatio`; h800/h1600/h3200/h4800: 0.34573060274124146 / 0.28615807824664646 / 0.011278331279754639 / -0.20057602061165702
- F62 decision: `F62EarlyChainNoContinuousRetentionRatio`; h800/h1600/h3200/h4800: 0.3297286894586351 / 0.3189403149816725 / 0.11756877766715156 / -0.050098571512434215
- F63 decision: `F63ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.34744171963797676 / 0.3360384636455112 / 0.13465211788813272 / -0.03395819001727634
- F64 decision: `F64ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.3511849774254693 / 0.33645888831880355 / 0.1350817879041036 / -0.03357837597529093
- F65 decision: `F65EarlyChainNoContinuousRetentionH3200`; h800/h1600/h3200/h4800: 0.0827075441678365 / -0.25741936763127643 / -0.779679642783271 / -1.1511565181944106

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.009399996863471137 | -0.34789032406277126 | -0.9199768039915297 | -1.3200003173616197 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.5458155539300706 | -1.3870148261388142 | -1.10538829697503 | -1.0861819055345323 | -1.2978806893030803 | -1.4658147361543443 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.5889208912849426 | -1.4302488035625882 | -1.1489323774973552 | -1.1294043064117432 | -1.3411992258495755 | -1.5093958510292902 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4990846912066143 | -1.1522363556755915 | -0.6346658865610758 | -0.2143194013171726 | -0.0053694115744696725 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F57-adamw-boundary-dual-timescale-antiwashout-source | 9 | 0 | 0.01391528050104777 | 0.04101914829678006 | 0.29924317863252425 | 0.2694499691327413 | -0.012103550963931613 | -0.2369269993570116 | 0.9004381331733896 | 0.0 |  | 1 | 0 | ContinuousRetentionMissing |
| MLP | MLP-F58-adamw-boundary-dual-timescale-source-anchor | 9 | 0 | -0.02303592363993327 | -0.023477594057718914 | 0.23931911256578234 | 0.20521441433164808 | -0.08447160323460896 | -0.29526517788569134 | 0.8574927933314987 | 0.0 |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.009399996863471137 | -0.34789032406277126 | -0.9199768039915297 | -1.3200003173616197 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.5458155539300706 | -1.3870148261388142 | -1.10538829697503 | -1.0861819055345323 | -1.2978806893030803 | -1.4658147361543443 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.5889208912849426 | -1.4302488035625882 | -1.1489323774973552 | -1.1294043064117432 | -1.3411992258495755 | -1.5093958510292902 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4990846912066143 | -1.1522363556755915 | -0.6346658865610758 | -0.2143194013171726 | -0.0053694115744696725 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F57-adamw-boundary-dual-timescale-antiwashout-source | 9 | 0 | 0.01391528050104777 | 0.04101914829678006 | 0.29924317863252425 | 0.2694499691327413 | -0.012103550963931613 | -0.2369269993570116 | 0.9004381331733896 | 0.0 |  | 1 | 0 | ContinuousRetentionMissing |
| MLP | MLP-F58-adamw-boundary-dual-timescale-source-anchor | 9 | 0 | -0.02303592363993327 | -0.023477594057718914 | 0.23931911256578234 | 0.20521441433164808 | -0.08447160323460896 | -0.29526517788569134 | 0.8574927933314987 | 0.0 |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry | 9 | 0 | 0.04618631468878852 | 0.08161209689246283 | 0.3614519172244602 | 0.2964131236076355 | 0.01199809710184733 | -0.1983597609731886 | 0.8200623913790562 | 0.040477617710777576 | 0.0 | 1 | 0 | ContinuousRetentionMissing |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

_仅显示前 20 / 47 rows；完整 CSV 见 artifact。_

## F66 KAN C3 Readout-Only Source Writer

- F66 decision: `F66PhaseResetNoEarlyChain`; h800/h1600/h3200/h4800: -1.6231121751997206 / -1.6301616761419508 / -1.5118636190891266 / -1.410119225581487

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.4968926641676161 | -1.43982019689348 | -1.6382287873162165 | -1.6473306483692594 | -1.5299246211846669 | -1.428863482342826 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -0.4900570710500081 | -1.4329893721474543 | -1.631406095292833 | -1.6405073205629985 | -1.5231055584218767 | -1.4220541682508256 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-SGD | 9 | 0 | -0.4857100115882026 | -1.4255716933144464 | -1.6215447849697537 | -1.629043545987871 | -1.511746476093928 | -1.410317748785019 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | KSW1-basis-estimate-readout-commit | 9 | 0 | -0.48900026745266384 | -1.427678492334154 | -1.6231121751997206 | -1.6301616761419508 | -1.5118636190891266 | -1.410119225581487 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | KSW2-lowdegree-lowfreq-source-bank | 9 | 0 | -0.3032250139448378 | -0.3417712450027466 | -0.18008546034495035 | -0.07460847828123304 | -0.005438006586498684 | 0.027257995473013982 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-FOU | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.6905753082699246 | -1.4386717677116394 | -1.5397865838474698 | -1.5077248414357503 | -1.4303334554036458 | -1.364314039548238 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.6935511430104574 | -1.441648370689816 | -1.5427624980608623 | -1.510699643029107 | -1.4333072768317328 | -1.3672899802525837 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-SGD | 9 | 0 | -0.6922047932942709 | -1.439205374982622 | -1.5388435059123569 | -1.5038206312391493 | -1.4204048315684001 | -1.347576101620992 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | KSW1-basis-estimate-readout-commit | 9 | 0 | -0.6899531417422824 | -1.4369259344206915 | -1.536535296175215 | -1.5015311241149902 | -1.4180500507354736 | -1.345011830329895 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | KSW2-lowdegree-lowfreq-source-bank | 9 | 0 | -0.557450983259413 | -0.3531768520673116 | -0.15396682421366373 | -0.06793245342042711 | -0.10215700334972805 | -0.13887516657511392 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

## F67 KAN C3 Low-Bank B3-Null Source Writer

- F67 decision: `F67PhaseResetNoEarlyChain`; h800/h1600/h3200/h4800: -1.6219656997256808 / -1.6307863593101501 / -1.5150579777028825 / -1.4138417343298595

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.4968926641676161 | -1.43982019689348 | -1.6382287873162165 | -1.6473306483692594 | -1.5299246211846669 | -1.428863482342826 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -0.4900570710500081 | -1.4329893721474543 | -1.631406095292833 | -1.6405073205629985 | -1.5231055584218767 | -1.4220541682508256 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-SGD | 9 | 0 | -0.4857100115882026 | -1.4255716933144464 | -1.6215447849697537 | -1.629043545987871 | -1.511746476093928 | -1.410317748785019 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | F36-lowbank-loss-b3-null | 9 | 0 | -0.48682689666748047 | -1.4254724317126803 | -1.6258274449242487 | -1.6340082155333624 | -1.515865898794598 | -1.412504815393024 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | F38-gain-gated-lowbank-loss-b3-null | 9 | 0 | -0.4849250581529405 | -1.4243185387717352 | -1.6219656997256808 | -1.6307863593101501 | -1.5150579777028825 | -1.4138417343298595 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.6905753082699246 | -1.4386717677116394 | -1.5397865838474698 | -1.5077248414357503 | -1.4303334554036458 | -1.364314039548238 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.6935511430104574 | -1.441648370689816 | -1.5427624980608623 | -1.510699643029107 | -1.4333072768317328 | -1.3672899802525837 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-SGD | 9 | 0 | -0.6922047932942709 | -1.439205374982622 | -1.5388435059123569 | -1.5038206312391493 | -1.4204048315684001 | -1.347576101620992 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | F36-lowbank-loss-b3-null | 9 | 0 | -0.6718335416581895 | -1.3025149703025818 | -1.0049330062336392 | -0.3352172374725342 | 0.049605203999413386 | 0.15526423851648966 |  |  | 3.129998992007487 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-FOU | F38-gain-gated-lowbank-loss-b3-null | 9 | 0 | -0.6818294260236952 | -1.3823313117027283 | -1.2923956910769145 | -0.7170495722028944 | -0.06800315777460735 | 0.13637220197253758 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |

## F68 AdamW-Boundary Low-Bank B3-Null Source Writer

- F68 decision: `F68PhaseResetNoEarlyChain`; h800/h1600/h3200/h4800: -0.0005275342199537489 / 0.00357857346534729 / -0.0021484394868214927 / -0.007300949758953518

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.4876655870013767 | -1.43657210138109 | -1.6380775703324213 | -1.6471367677052815 | -1.540601564778222 | -1.4429288903872173 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -0.4890313280953301 | -1.4379423989189997 | -1.6394559741020203 | -1.6485039922926161 | -1.5419820613331265 | -1.4443014131651983 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-SGD | 9 | 0 | -0.49327494038475883 | -1.4370821581946478 | -1.6341689891285367 | -1.6391930050320096 | -1.5304909745852153 | -1.4319615297847323 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | F68-adamw-boundary-to-gated-lowbank-b3-null | 9 | 0 | 0.007974889543321397 | -0.0018029676543341742 | -0.0005275342199537489 | 0.00357857346534729 | -0.0021484394868214927 | -0.007300949758953518 |  | 0.0 |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.6984561416837904 | -1.4388298988342285 | -1.538353595468733 | -1.5050356056955125 | -1.4238311648368835 | -1.3542947106891208 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.699242975976732 | -1.4396168655819364 | -1.5391405357254877 | -1.5058234466446772 | -1.424617310365041 | -1.3550791343053181 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-SGD | 9 | 0 | -0.7002888123194376 | -1.4395086500379775 | -1.5374954740206401 | -1.5011191566785176 | -1.413574344582028 | -1.3368320067723591 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | F68-adamw-boundary-to-gated-lowbank-b3-null | 9 | 0 | 0.008326477474636503 | 0.010237760014004178 | 0.011607607205708822 | 0.0015850961208343506 | -0.004651433891720242 | -0.005101071463690864 | 0.1365566643274053 | 0.0 |  | 1 | 0 | ContinuousRetentionMissing |

## F69 AdamW Low-Bank Anchor Anti-Washout

- F69 decision: `F69EarlyChainNoContinuousRetentionH3200`; h800/h1600/h3200/h4800: 0.0014568898412916395 / -0.0052944521109263105 / -0.013673616780175103 / -0.015228016508950127

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.4876655870013767 | -1.43657210138109 | -1.6380775703324213 | -1.6471367677052815 | -1.540601564778222 | -1.4429288903872173 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -0.4890313280953301 | -1.4379423989189997 | -1.6394559741020203 | -1.6485039922926161 | -1.5419820613331265 | -1.4443014131651983 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CTRL-SGD | 9 | 0 | -0.49327494038475883 | -1.4370821581946478 | -1.6341689891285367 | -1.6391930050320096 | -1.5304909745852153 | -1.4319615297847323 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | F69-adamw-boundary-lowbank-anchor-antiwashout | 9 | 0 | 0.007455190022786458 | 0.00037747621536254883 | 0.0014568898412916395 | -0.0052944521109263105 | -0.013673616780175103 | -0.015228016508950127 | 0.0 |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-FOU | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  | 1 | 0 | ContinuousRetentionMissing |
| D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.6984561416837904 | -1.4388298988342285 | -1.538353595468733 | -1.5050356056955125 | -1.4238311648368835 | -1.3542947106891208 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.699242975976732 | -1.4396168655819364 | -1.5391405357254877 | -1.5058234466446772 | -1.424617310365041 | -1.3550791343053181 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | CTRL-SGD | 9 | 0 | -0.7002888123194376 | -1.4395086500379775 | -1.5374954740206401 | -1.5011191566785176 | -1.413574344582028 | -1.3368320067723591 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | F69-adamw-boundary-lowbank-anchor-antiwashout | 9 | 0 | 0.0104879273308648 | -0.004357702202267117 | 0.0022645857599046496 | -0.007774823241763645 | -0.01170006725523207 | -0.01502860254711575 | 0.0 |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

## F70 Terminal Positive Lookahead Floor

- F70 decision: `F70ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.374403801229265 / 0.36406707101398045 / 0.1626881096098158 / -0.005899740589989556

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 9 | 0 | 0.033847285641564265 | 0.06886153750949436 | 0.374403801229265 | 0.36406707101398045 | 0.1626881096098158 | -0.005899740589989556 | 0.9723914923370266 | 0.4468630166323624 | 0.0 | 1 | 1 | pass |

## F71 Terminal H3200 Checkpoint Reentry

- F71 decision: `F71ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.36371467510859173 / 0.353307565053304 / 0.1519218815697564 / -0.016645153363545735

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | 9 | 0 | 0.05093061261706882 | 0.05545302232106527 | 0.36371467510859173 | 0.353307565053304 | 0.1519218815697564 | -0.016645153363545735 | 0.9713866094289967 | 0.4299989487823045 | 0.0 | 1 | 1 | pass |

## F72 Terminal Hard-Split Source

- F72 decision: `F72ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.3613293833202786 / 0.35068535804748535 / 0.1492934226989746 / -0.0192489226659139

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F72-trainloss-terminal-hard-split-source | 9 | 0 | 0.04502135515213013 | 0.05726057953304715 | 0.3613293833202786 | 0.35068535804748535 | 0.1492934226989746 | -0.0192489226659139 | 0.9705420434535807 | 0.42571900785991523 | 0.0 | 1 | 1 | pass |

## F73 Terminal AdamW Lookahead

- F73 decision: `F73ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.3590083122253418 / 0.3483833935525682 / 0.14700835280948216 / -0.02212838331858317

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F73-trainloss-terminal-adamw-lookahead | 9 | 0 | 0.03758410612742106 | 0.047415720091925725 | 0.3590083122253418 | 0.3483833935525682 | 0.14700835280948216 | -0.02212838331858317 | 0.9704048115016776 | 0.42197290551192645 | 0.0 | 1 | 1 | pass |

## F74 Terminal Optimizer Selector

- F74 decision: `F74ContinuousH3200ButH4800Failed`; h800/h1600/h3200/h4800: 0.3710273702939351 / 0.3603704770406087 / 0.15898103184170193 / -0.011182712184058296

| carrier | v22_id | rows | blocked_rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | h1600_retention_ratio | h3200_retention_ratio | h4800_retention_ratio | early_source_chain_group | continuous_retention_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | CTRL-AdamW | 9 | 0 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-RandomMatchedNorm | 9 | 0 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | CTRL-SGD | 9 | 0 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 |  |  |  | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP | MLP-F74-trainloss-terminal-optimizer-selector | 9 | 0 | 0.03304861651526557 | 0.06570586893293592 | 0.3710273702939351 | 0.3603704770406087 | 0.15898103184170193 | -0.011182712184058296 | 0.9712773393378397 | 0.44115997832915427 | 0.0 | 1 | 1 | pass |

## F53/F70/F71/F72/F73/F74 Terminal Recovery Row-Level Localization

| v22_id | dataset | seed | source_h800 | source_h1600 | source_h3200 | source_h4800 | early_source_chain | continuous_retention_chain | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F53-trainloss-terminal-lookahead-floor-source | Fashion-MNIST | 0 | 0.22544801235198975 | 0.29342931509017944 | 0.040415942668914795 | -0.015426993370056152 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | Fashion-MNIST | 1 | 0.32990264892578125 | 0.2431182861328125 | -0.010123670101165771 | -0.0944567322731018 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | Fashion-MNIST | 2 | 0.6944085359573364 | 0.28719139099121094 | -0.030561089515686035 | -0.12596487998962402 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | KMNIST | 0 | 0.813734769821167 | 0.5475614070892334 | 0.16402530670166016 | -0.0505518913269043 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | KMNIST | 1 | 0.4000767469406128 | 0.6217105388641357 | 0.212191641330719 | -0.02529221773147583 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | KMNIST | 2 | 0.5355304479598999 | 0.49152421951293945 | 0.02587294578552246 | -0.12781846523284912 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | MNIST | 0 | 0.17646336555480957 | 0.3192850351333618 | 0.3895149230957031 | 0.12493276596069336 | 1 | 1 | pass |
| MLP-F53-trainloss-terminal-lookahead-floor-source | MNIST | 1 | 0.020914971828460693 | 0.15060502290725708 | 0.2799164652824402 | 0.07123345136642456 | 0 | 1 | EarlySourceChainMissing |
| MLP-F53-trainloss-terminal-lookahead-floor-source | MNIST | 2 | 0.18926870822906494 | 0.34442996978759766 | 0.4151580333709717 | 0.2076730728149414 | 0 | 1 | EarlySourceChainMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | Fashion-MNIST | 0 | 0.23452860116958618 | 0.3025099039077759 | 0.04948955774307251 | -0.005643010139465332 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | Fashion-MNIST | 1 | 0.31575489044189453 | 0.22897052764892578 | -0.02427440881729126 | -0.10851192474365234 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | Fashion-MNIST | 2 | 0.7141324281692505 | 0.306915283203125 | -0.010809898376464844 | -0.10377371311187744 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | KMNIST | 0 | 0.7940394878387451 | 0.5278661251068115 | 0.1443321704864502 | -0.06921029090881348 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | KMNIST | 1 | 0.3779338598251343 | 0.5995676517486572 | 0.19005882740020752 | -0.046901047229766846 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | KMNIST | 2 | 0.5473171472549438 | 0.5013833045959473 | 0.0357288122177124 | -0.1179666519165039 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | MNIST | 0 | 0.16551613807678223 | 0.3044423460960388 | 0.37467265129089355 | 0.1100953221321106 | 1 | 1 | pass |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | MNIST | 1 | 0.06461840867996216 | 0.1952981948852539 | 0.3246103525161743 | 0.11592137813568115 | 0 | 1 | EarlySourceChainMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | MNIST | 2 | 0.15579324960708618 | 0.3096503019332886 | 0.3803849220275879 | 0.1728922724723816 | 0 | 1 | EarlySourceChainMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | Fashion-MNIST | 0 | 0.21529215574264526 | 0.28327345848083496 | 0.030268549919128418 | -0.02487313747406006 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | Fashion-MNIST | 1 | 0.2775481939315796 | 0.19076383113861084 | -0.06250542402267456 | -0.14657670259475708 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | Fashion-MNIST | 2 | 0.6203817129135132 | 0.2131645679473877 | -0.1045830249786377 | -0.19740843772888184 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | KMNIST | 0 | 0.7877135276794434 | 0.5215401649475098 | 0.13799548149108887 | -0.07562506198883057 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | KMNIST | 1 | 0.3907405138015747 | 0.6123743057250977 | 0.20286321640014648 | -0.034165918827056885 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | KMNIST | 2 | 0.5075109004974365 | 0.4625732898712158 | -0.003081679344177246 | -0.15676796436309814 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | MNIST | 0 | 0.20090025663375854 | 0.3400067687034607 | 0.41023164987564087 | 0.14566850662231445 | 1 | 1 | pass |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | MNIST | 1 | 0.08514535427093506 | 0.21393036842346191 | 0.34324145317077637 | 0.13456249237060547 | 0 | 1 | EarlySourceChainMissing |
| MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | MNIST | 2 | 0.1881994605064392 | 0.342141330242157 | 0.4128667116165161 | 0.20537984371185303 | 0 | 1 | EarlySourceChainMissing |
| MLP-F72-trainloss-terminal-hard-split-source | Fashion-MNIST | 0 | 0.20976656675338745 | 0.27774786949157715 | 0.024717211723327637 | -0.029725730419158936 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F72-trainloss-terminal-hard-split-source | Fashion-MNIST | 1 | 0.31013089418411255 | 0.2233465313911438 | -0.029932081699371338 | -0.11393308639526367 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F72-trainloss-terminal-hard-split-source | Fashion-MNIST | 2 | 0.6572788953781128 | 0.2500617504119873 | -0.06773507595062256 | -0.16123396158218384 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F72-trainloss-terminal-hard-split-source | KMNIST | 0 | 0.8251702785491943 | 0.5589969158172607 | 0.17547130584716797 | -0.038147926330566406 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F72-trainloss-terminal-hard-split-source | KMNIST | 1 | 0.3676410913467407 | 0.5892748832702637 | 0.17976748943328857 | -0.05706912279129028 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F72-trainloss-terminal-hard-split-source | KMNIST | 2 | 0.4997260570526123 | 0.45455384254455566 | -0.011099696159362793 | -0.1647547483444214 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F72-trainloss-terminal-hard-split-source | MNIST | 0 | 0.1869562864303589 | 0.3251163363456726 | 0.39534103870391846 | 0.1307390332221985 | 1 | 1 | pass |
| MLP-F72-trainloss-terminal-hard-split-source | MNIST | 1 | 0.011406123638153076 | 0.1408945918083191 | 0.2702099680900574 | 0.061444878578186035 | 0 | 1 | EarlySourceChainMissing |
| MLP-F72-trainloss-terminal-hard-split-source | MNIST | 2 | 0.1838882565498352 | 0.33617550134658813 | 0.40690064430236816 | 0.1994403600692749 | 0 | 1 | EarlySourceChainMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | Fashion-MNIST | 0 | 0.23981237411499023 | 0.30779367685317993 | 0.054783761501312256 | 0.0015067458152770996 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | Fashion-MNIST | 1 | 0.31521397829055786 | 0.2284296154975891 | -0.024804651737213135 | -0.10761207342147827 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | Fashion-MNIST | 2 | 0.6390295028686523 | 0.23181235790252686 | -0.08591294288635254 | -0.18086040019989014 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | KMNIST | 0 | 0.8238548040390015 | 0.5576814413070679 | 0.1741575002670288 | -0.041439056396484375 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | KMNIST | 1 | 0.3524138927459717 | 0.5740476846694946 | 0.16456282138824463 | -0.0743173360824585 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | KMNIST | 2 | 0.4693293571472168 | 0.4227316379547119 | -0.0429304838180542 | -0.19811761379241943 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | MNIST | 0 | 0.17882263660430908 | 0.317737877368927 | 0.3879668712615967 | 0.1230270266532898 | 1 | 1 | pass |
| MLP-F73-trainloss-terminal-adamw-lookahead | MNIST | 1 | 0.04030406475067139 | 0.16846585273742676 | 0.2977737784385681 | 0.08854734897613525 | 0 | 1 | EarlySourceChainMissing |
| MLP-F73-trainloss-terminal-adamw-lookahead | MNIST | 2 | 0.17229419946670532 | 0.32675039768218994 | 0.39747852087020874 | 0.19010990858078003 | 0 | 1 | EarlySourceChainMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | Fashion-MNIST | 0 | 0.24170172214508057 | 0.30968302488327026 | 0.056662023067474365 | -0.0028145313262939453 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | Fashion-MNIST | 1 | 0.35229939222335815 | 0.2655150294303894 | 0.01220715045928955 | -0.07156103849411011 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | Fashion-MNIST | 2 | 0.6490557193756104 | 0.24183857440948486 | -0.0758967399597168 | -0.1702643632888794 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | KMNIST | 0 | 0.7583363056182861 | 0.49216294288635254 | 0.10863590240478516 | -0.10857844352722168 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | KMNIST | 1 | 0.39828550815582275 | 0.6199193000793457 | 0.21040070056915283 | -0.028932273387908936 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | KMNIST | 2 | 0.5313410758972168 | 0.4837934970855713 | 0.018138408660888672 | -0.13636672496795654 | 1 | 0 | ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | MNIST | 0 | 0.21101367473602295 | 0.3496575951576233 | 0.4198802709579468 | 0.1545063853263855 | 1 | 1 | pass |
| MLP-F74-trainloss-terminal-optimizer-selector | MNIST | 1 | 0.07975620031356812 | 0.20860886573791504 | 0.3379213809967041 | 0.12850910425186157 | 0 | 1 | EarlySourceChainMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | MNIST | 2 | 0.11745673418045044 | 0.2721554636955261 | 0.3428801894187927 | 0.13485747575759888 | 0 | 1 | EarlySourceChainMissing |

## 修改记录

- 新增 `dgkan/fu/source_chain.py`：实现 v22 early-source / continuous-retention / late-rebound 公式和单元测试。
- 新增 `dgkan/profiling/efficiency_v22.py`：实现 v22 E1/S1/E2 efficiency gates。
- 新增 v22 runner：S0.7 truth gate、efficiency wrapper、source-chain dynamics wrapper、MLP/KAN/target wrapper、finalizer。
- 新增 `experiments/run_v22_source_observability_audit.py`：离线审计 train-only / audit-readback / forbidden source-readback predictors，结果只作为 C4 no-go 或 held-out selector 候选，不直接 promotion。
- 新增 `M87-AdamWBoundaryThenMomentumSourceFU` 与 `MLP-F40-adamw-boundary-to-momentum-source`：前 400 step 使用 AdamW boundary 对齐 h100/h400，之后切换到 M2 momentum source，用来检验 early-phase mismatch 是否可修复。
- 新增 `M88-AdamWBoundaryThenDualTimescaleSourceFU` 与 `MLP-F41-adamw-boundary-to-dual-timescale-source`：在 F40 打开 early chain 后加入 short/long slow-state retention gate，检验 h3200/h4800 washout 是否可修复。
- 新增 `M89-AdamWBoundaryDualTimescaleSGDFloorFU` 与 `MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source`：在 F41 short/long source-state gate 后保留 SGD floor，检验 h4800 断链是否来自后段优化落后 controls。
- 新增 `M90-AdamWBoundaryDualTimescaleLateSGDFloorFU` 与 `MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source`：保持 F41 到 h3200，再开启 SGD floor，检验 F41 h4800 失败是否来自 h3200 之后缺少 optimizer progress。
- 新增 `M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU` 与 `MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source`：h3200 后只使用 0.1x train-gradient floor，检验 full SGD floor 过强导致的 source 破坏。
- 新增 `M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU` 与 `MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source`：在 F44 的 tiny floor 上把 gated slow-source residual 提到 0.5x FU，检验 h3200 ratio 是否只是 source residual 强度不足。
- 新增 `experiments/run_v22_f46_train_loss_selector_holdout.py`：用 F40-F43 phase-reset rows 拟合 train-loss selector 阈值，并在 F44/F45 held-out rows 上验证 selected subgroup 是否形成 continuous/h4800 source chain；该审计不直接 promotion，只生成下一轮 repair hypothesis。
- 新增 `M93-TrainLossGatedDualTimescaleTinyLateFU` 与 `MLP-F47-trainloss-gated-dual-timescale-tiny-late-source`：把 F46 学到的 train-loss threshold 变成在线 gate；h800 后 gate 不通过时回退 SGD，gate 通过时才继续 F44-style tiny late retention。
- 新增 `M94-TrainLossGatedDualTimescaleHoldFallbackFU` 与 `MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source`：针对 F47 的 reject-row fallback washout，h800 后 train-loss gate 不通过时不再执行 SGD fallback，而是 hold 参数；gate 通过时保留 tiny late floor 与 slow-source residual。
- 新增 `M95-TrainLossGatedDualTimescaleTinyLateFallbackFU` 与 `MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source`：针对 F48 的长期 hold 后 h4800 仍输 controls，reject rows 在 h3200 前 hold，h3200 后只执行 0.1x tiny train-gradient floor，不提交 slow-source residual。
- 新增 `M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU` 与 `MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source`：针对 F49 h4800 仍略负，reject rows 在 h3200 后使用 0.2x tiny train-gradient floor；selected rows 的 source path 不变。
- 新增 `M97-TrainLossLateHoldRecoveryFU` / `MLP-F51-trainloss-late-hold-recovery-source`：h3200 后默认 hold，仅当 train loss 相对 h800 gate 明显恶化时提交 0.05x micro floor，检验 h4800 失败是否来自后段扰动。
- 新增 `M98-TrainLossLateLookaheadFloorFU` / `MLP-F52-trainloss-late-lookahead-floor-source`：h3200 后用 train split A/B 与 corrupt split 做 micro-floor precommit gate，只提交 train-only 即时有益且不明显帮助 corrupt 的恢复步。
- 新增 `M99-TrainLossTerminalLookaheadFloorFU` / `MLP-F53-trainloss-terminal-lookahead-floor-source`：h3200-h4000 保持 source，最后 800 step 才允许 train-split lookahead micro-floor，分离 source 保存与末端 SGD catch-up。
- 新增 `M100-TrainLossEarlyTerminalLookaheadFloorFU` / `MLP-F54-trainloss-early-terminal-lookahead-floor-source`：把 terminal catch-up 提前到 step 3600，检验 F53 near-miss 是否只是 terminal phase 太短。
- 新增 v22 F55/F56 official specs：复用既有 `M37-SplitFisherAgreementSlowFU` 与 `M39-MomentumWarmSplitFisherFU`，把 C4 的 split-gradient / Fisher-normalized / corrupt-filter source observability reset 纳入 fresh h4800 matrix。
- 新增 `M101-AdamWBoundaryDualTimescaleAntiWashoutFU` / `MLP-F57-adamw-boundary-dual-timescale-antiwashout-source`：复用 F40/F41 early source 起点，h1200 后用 train-only source-axis anti-washout projection 继续优化，检验长期 optimizer 是否反向冲掉 source。
- 新增 `M102-AdamWBoundaryDualTimescaleSourceAnchorFU` / `MLP-F58-adamw-boundary-dual-timescale-source-anchor`：复用 F40/F41 early source 起点，h1200 后保持 optimizer progress 并在 train-only/corrupt gate 通过时提交 source-anchor residual，检验 retained target anchor 是否优于 terminal floor。
- 新增 `M103-AdamWBoundaryDualTimescaleParamEMAReentryFU` / `MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry`：复用 F40/F41 early source 起点，source warm 阶段维护参数 EMA，h3200 后只在 train-only/corrupt gate 通过时向 EMA 小幅回投，检验 schedule-free/fast-slow 权重轨迹是否能保留 terminal source。
- 新增 `M104-AdamWBoundaryDualTimescaleReadoutChannelFU` / `MLP-F60-adamw-boundary-dual-timescale-readout-channel`：复用 F40/F41 early source 起点，h1200 后只把 short/long agreed source 投影到 readout/classifier channel，并通过 train A/B + corrupt gate 提交，检验 retained target 是否应由低容量 readout channel 承载。
- 新增 `M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU` / `MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel`：复用 F40/F41 early source 起点，h1200 后只把 short/long agreed source 投影到 hidden matrix block，并通过 train A/B + corrupt gate 提交，检验 retained target 是否应由隐藏矩阵通道承载。
- 新增 `M106-TrainLossTerminalProjectedLookaheadFloorFU` / `MLP-F62-trainloss-terminal-projected-lookahead-floor-source`：复用 F53 的 terminal lookahead micro-floor，但在提交 terminal gradient 前投掉与 accumulated source-axis 反向的分量，检验 cumulative optimizer projection 是否能保留 h3200 source 并完成 h4800 recovery。
- 新增 `M107-TrainLossTerminalConsensusLookaheadFloorFU` / `MLP-F63-trainloss-terminal-consensus-lookahead-floor-source`：复用 F53 的 terminal recovery 时机，但 terminal floor 改为 train split A/B 共识梯度，并投掉 corrupt-label 方向后用 train/corrupt lookahead gate 提交，检验 C4 E2/E7 noise-reservoir separation 是否能把 F53 near-miss 闭合。
- 新增 `M108-TrainLossTerminalSelectorLookaheadFloorFU` / `MLP-F64-trainloss-terminal-selector-lookahead-floor-source`：保持 F53 的 terminal recovery 时机与尺度不变，用 train split A/B + corrupt-label lookahead 在 raw floor、source-projected floor、cross-split consensus floor 三个合法方向中逐步选择，检验 C4 train-only terminal direction selector 能否闭合 h4800。
- 新增 `MLP-F65-adamw-split-fisher-residual-source`：复用既有 `M38-AdamWSplitFisherAgreementResidualFU`，把 C4 split-Fisher estimator 与 AdamW base/residual 直接组合，检验 F55/F56 未覆盖的 AdamW+Fisher source-observability reset 是否能形成 h4800 retained chain。
- 新增 F66 KAN C3 fresh h4800 复验入口：把既有 `KSW1-basis-estimate-readout-commit` 纳入 v22 KAN scope/default 与 S0.7 合同，并与 `KSW2-lowdegree-lowfreq-source-bank`、controls 同 run 验证 readout-only commit 是否优于 full low-bank writer。
- 新增 F67 KAN C3 fresh h4800 复验入口：用既有 `F36-lowbank-loss-b3-null` / `F38-gain-gated-lowbank-loss-b3-null` 与 controls 检验低阶/低频 source bank 加 B3 reservoir-null 是否能把 KSW2 的 D-CHE late rebound 转成 early/continuous retained source。
- 新增 `M109-AdamWBoundaryToGainGatedLowBankB3NullFU` / `F68-adamw-boundary-to-gated-lowbank-b3-null`：前 400 step 使用 AdamW boundary，之后只在 train split B2 gain 明显优于 B3 safety gain 时提交低阶/低频 B3-null source-channel residual，检验 F67 的 late rebound 是否能被 boundary-conditioned writer 转成 early chain。
- 新增 `M110-AdamWBoundaryLowBankAnchorAntiWashoutFU` / `F69-adamw-boundary-lowbank-anchor-antiwashout`：在 F68 的 accepted low-bank source-state 上加入 AdamW gradient anti-washout projection，检验 D-FOU h800 early chain 是否只是被后段 optimizer 反向冲掉。
- 新增 `M111-TrainLossTerminalPositiveLookaheadFloorFU` / `MLP-F70-trainloss-terminal-positive-lookahead-floor-source`：复用 F53 的 source-preserving terminal recovery，但 terminal floor gate 必须满足 train split A/B 正 gain，并把 corrupt gain allowance 从 0.50 收紧到 0.25，检验 F53 h4800 near-miss 是否来自 terminal gate 过宽。
- 新增 `M112-TrainLossTerminalCheckpointReentryFU` / `MLP-F71-trainloss-terminal-h3200-checkpoint-reentry`：在 h3200 保存 source-bearing 参数检查点，terminal phase 先用 train split A/B + corrupt gate 尝试向 h3200 checkpoint 小幅回灌，若不通过再回退 F53 raw terminal lookahead floor，检验 h4800 断链是否只是 h3200 retained target 被终端漂移冲掉。
- 新增 `M113-TrainLossTerminalHardSplitSourceFU` / `MLP-F72-trainloss-terminal-hard-split-source`：基于 F53/F70/F71 row-level localization，terminal phase 不再使用全样本平均 floor，而只从 train split A/B 高 CE hard examples 构造 split-consensus source，并用 corrupt-label hard split gate 过滤，检验 Fashion-MNIST/KMNIST terminal 失败是否来自 hard-example source channel 被平均 floor 稀释。
- 新增 `M114-TrainLossTerminalAdamWLookaheadFU` / `MLP-F73-trainloss-terminal-adamw-lookahead`：基于 F53 近失败，terminal phase 不再提交 SGD-style raw floor，而用 train split A/B + corrupt gate 预览缩放 AdamW step，检验 Fashion-MNIST/KMNIST h4800 失败是否来自 terminal catch-up optimizer carrier 错配。
- 新增 `M115-TrainLossTerminalOptimizerSelectorFU` / `MLP-F74-trainloss-terminal-optimizer-selector`：在 terminal phase 用 train split A/B + corrupt gate 在 F53 raw floor 与 F73 AdamW-lookahead carrier 间逐步选择，检验 AdamW carrier 是否只需被 train-only selector 局部启用。
- 扩展 `experiments/run_v22_source_observability_audit.py`：新增 C4 E1 split-Fisher SNR、E6 split-Fisher coherence、E7 noise/reservoir separation 三个 train-stream diagnostic composite predictors。
- 修复 source/control attribution key：`experiments/run_v21_01_source_retention.py` 的 matched control key 加入 `run_label`，避免 F40/F41/F42 等 fresh run 在 cumulative merge 时被其他 run 的同 seed/offset controls 污染。
- 扩展 `experiments/run_v22_f40_phase_reset_summary.py`：可按 `run_prefix` / `candidate_prefix` / `artifact_prefix` 写出 F40 或 F41 独立 phase-reset matrix / summary / decision。
- 训练 loop 复用 v21.01 已审计 source-retention runner；v22 wrapper 在独立 official dir 中重算 source-chain 判定，不复用 v21.01 旧结论。

## 分析 / Insight / 结论

- v22 的 promotion gate 仍以 grouped source-chain 为准；single-row positive、h1600-only signal、late rebound 或 smoke positive 都不能 promotion。
- 若 source-chain evidence 中 `early_source_chain_group=0`，说明当前尝试连 h100/h400/h800 的连续早期 source 链都没有打开；按计划不能升级 full 或 independent confirmation。
- C4/F46 显示 train-only loss 指标能解释部分 row-level retained source，但这不是 promotion：必须通过真实在线机制把 selected subgroup 的 row-level 信号转成 grouped continuous/h4800 evidence，并排除独立 offset/control contamination。
- F40 若仍不能形成 early chain，说明问题不是简单的 AdamW early boundary/h100-h400 相位错配；若形成 early chain，则只能升级 full h3200/h4800 验证，不能直接 promotion。
- F41 若在 F40 early chain 基础上仍不能通过 h3200/h4800，说明 dual-timescale slow-state gate 没能解决长期 retention washout，不能把 early-chain smoke 写成 retained source。
- F42 若保留 SGD floor 后仍不能通过 h3200/h4800，说明后段优化停滞不是唯一 blocker；如果它保住 h4800，则还必须做 independent offsets 和 controls attribution，不能直接 promotion。
- F43 若 h3200 前保留 F41、h3200 后才加 SGD floor 仍失败，则 blocker 更像 source/optimizer 两相动态不可简单拼接，而不是单纯 floor 时机问题。
- F44 若 tiny late floor 仍失败，则可以关闭简单 floor-scale 修复；继续推进需要新的 train-only source observability，而不是继续调 floor 强度。
- F45 若 reinforced source residual 仍失败，则关闭 phase-reset + dual-timescale + late-floor 这条修复线，避免继续做无审计价值的 scale sweep。
- F46 若 held-out selected subgroup 有 continuous/h4800，说明下一步可以尝试 train-loss gated late-retention repair；如果只停留在 post-hoc selection，仍不能写 breakthrough。
- F47 若不能把离线 selected subgroup 变成全 grouped continuous/h4800，说明 train-loss selector 主要是 failure taxonomy，而不是足够强的在线 source-retention mechanism。
- F48 若 hold fallback 仍不能形成 grouped continuous/h4800，说明 F47 失败不只是 reject rows 被 SGD fallback 冲刷；train-loss selector 仍无法覆盖全 dataset/seed grouped source-retention dynamics。
- F49 若 tiny late fallback 仍不能形成 grouped continuous/h4800，说明 rejected rows 需要的不只是 h3200 后少量 optimizer progress；当前 train-loss gate 不能把 high-loss rows 纳入 retained source chain。
- F50 是本轮对 rejected-row late floor 的最后定向强度验证；若 h4800 仍不过线，应关闭 train-loss gate + late-floor 修复线，避免继续做无审计价值的 scale sweep。
- F51 说明纯 late hold 略优于 F49 但仍 h4800 负；F52 说明全 late-phase train split lookahead micro-floor 反而退化。
- F53 是当前最强 late-phase repair，h4800=-0.003963543309105767，仍未达到 `>=0.005`；F54 提前 terminal catch-up 退化，说明不是简单把 terminal window 加长即可闭合。
- F55/F56 是按 C4 进行的 source observability reset：若仍不能超过 F53 并形成 h4800 retained chain，说明 split-Fisher / noise-reservoir separation 也不足以把 early source 保存到 terminal horizon。
- F57/F58 是 F55/F56 后的 source-theory reset，不属于 terminal boundary 小扫；若仍不能形成 h4800 retained chain，说明当前 train-only source-axis anti-washout / anchor residual 仍不足以把 F40/F41 的 early source 变成 terminal retained source。
- F59 是 schedule-free/fast-slow iterate 假设的参数轨迹回投测试；若仍不能形成 h4800 retained chain，说明仅把 early source 轨迹保存为参数 EMA 也不足以抵抗 terminal washout。
- F60 是 representation/channel reset：若 readout-only source channel 仍不能形成 h4800 retained chain，说明当前 MLP retained target 不只是全参数 carrier 错，而是 source-target definition 本身仍不够。
- F61 是 hidden-matrix channel reset：若 hidden matrix block 仍不能形成 h4800 retained chain，说明 matrix-block carrier 也不足以承接当前 source target，后续需要重新定义 train-only source observability/retained target，而不是继续小扫通道强度。
- F62 是 source-preserving terminal recovery：若 terminal gradient 投影仍不能形成 h4800 retained chain，说明 F53 near-miss 不是简单的 terminal optimizer/source-axis 冲突，不能继续靠 terminal floor 小修写 breakthrough。
- F63 是 C4 terminal consensus recovery：若 split-consensus/noise-reservoir terminal floor 仍不能形成 h4800 retained chain，说明当前 train-only cross-split estimator 也不足以把 h3200 retained source 转成 h4800 source。
- F64 是 C4 terminal selector recovery：若 train-only lookahead 在 raw/projected/consensus 三个合法方向之间选择仍不能形成 h4800 retained chain，说明当前 terminal 方向选择器不足以修复 source-retention 断链，后续应转向新的 retained target/source observability theory，而不是继续围绕 terminal floor 做局部小扫。
- F65 是非 terminal 的 C4 AdamW split-Fisher residual reset：若它仍不能形成 h4800 retained chain，说明把 split-Fisher source estimator 直接接到 AdamW base/residual 也不足以修复 source observability/retention 断链。
- F66 是 C3 representation/channel reset：若 KAN readout-only commit 仍不能形成 early/continuous/h4800 retained source，说明当前 KAN low-bank/readout channel 也没有承载 v22 retained source，不应把 KAN writer smoke 或 efficiency pass 写成 functional breakthrough。
- F67 是 C3 reservoir-null reset：若低阶/低频 source bank + B3 null 仍不能形成 early/continuous/h4800 retained source，说明 KSW2 的 D-CHE h4800 positive 仍只是 delayed migration/late rebound，而不是 source-channel writer breakthrough。
- F68 是 C3 boundary-conditioned source-channel writer reset：若 AdamW boundary + gain-gated low-bank B3-null 仍不能形成 early/continuous/h4800 retained source，说明当前 low-bank source-channel target 与 optimizer boundary 的组合也不能把 late migration 转成 retained source。
- F69 是 F68 后的 source-state anti-washout 验证：若 accepted low-bank source-state + AdamW anti-source projection 仍不能闭合 h1600/h3200/h4800，说明当前 blocker 不只是后段 AdamW 梯度反向冲刷。
- F70 是 F53 near-miss 后的 terminal gate 收紧验证：若 positive-only terminal lookahead 仍不能让 h4800 达到 `>=0.005`，说明 F53 失败不是单纯由 terminal gate 允许微负 train-split gain 导致。
- F71 是 h3200 retained target 参数检查点回灌验证：若 checkpoint reentry 仍不能让 h4800 达到 `>=0.005`，说明 blocker 不只是 h3200 后参数漂移遗失 source-bearing target；当前需要重新定义 train-only source observability 或 retained target，而不是继续围绕 terminal 回灌做局部小扫。
- F72 是困难样本 source theory reset：若 hard-split terminal source 仍不能让 h4800 达到 `>=0.005`，说明 Fashion-MNIST/KMNIST 的 terminal blocker 不只是全样本 floor 稀释 hard-example source channel；下一步需要更上游的 train-only source observability theory，而不是继续在 terminal phase 调方向或尺度。
- F73 是 terminal optimizer-carrier reset：若 AdamW-lookahead terminal catch-up 仍不能让 h4800 达到 `>=0.005`，说明 F53 近失败不是简单因为 SGD-style terminal floor 输给 AdamW/SGD controls；当前 h4800 blocker 更可能在 source observability/retained target definition，而不是 terminal optimizer carrier。
- F74 是 terminal carrier selector reset：若 raw-vs-AdamW train-only selector 仍不能让 h4800 达到 `>=0.005`，说明即使允许合法 online carrier selection，当前 terminal phase 也无法把 h3200 retained source 变成 h4800 retained source；应停止 terminal selector/carrier 局部修复线。
- 因此当前不能把 F53 near-miss 或 F55/F56 的局部 positive 写成 breakthrough；下一步若继续，不应继续小扫 terminal boundary，而应转向新的 source observability / retained target theory。
- 若 efficiency S1 有 rows 但 functional source-chain 未闭合，主 blocker 是 functional source retention，而不是 kernel officialization。
- 当前 `promotion_allowed=0` 表示真实证据链未闭合，不是预算性放弃；后续只能从新的 train-only early-source observability 或 target-to-retention theory 继续。

- v22_code_review_packet.zip: size=9008158 sha256=b46ba6ef0aa814674550fd3f0dabd6bf789f6b51acd537240c2926b75a40aeeb

- v22_results_bundle.zip: size=8792993 sha256=4e6c4117a5309817390a9289191d55d2a9ae4afe4dde90cf7d25a4339672d3fd
