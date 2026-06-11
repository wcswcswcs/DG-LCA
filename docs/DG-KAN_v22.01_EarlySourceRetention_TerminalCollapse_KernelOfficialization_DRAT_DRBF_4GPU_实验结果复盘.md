# DG-KAN v22.01 EarlySourceRetention TerminalCollapse KernelOfficialization DRAT/DRBF 4GPU 实验结果复盘

生成时间：2026-06-04 19:27:05 +0800

## Route

- route: `R4-TerminalCollapseNoH4800-SourceTargetTheoryInsufficient`
- S0.8 pass: 1
- D-CHE/D-FOU S1 rows: 6 / 3
- functional early / h3200 / h4800: 28 / 12 / 0
- terminal collapse groups: 12
- MLPRoute: MLPStillTerminalCollapse
- SelectorRoute: RetrospectiveOnlyNoPrecommitSelector
- TargetRoute: TargetObservabilityNoRetention
- KANRoute: KANWriterNoEarlyContinuousSource
- D-RAT/D-RBF: RejectedForThisVersion / RejectedForThisVersion
- promotion_allowed: 0
- required artifact missing count: 0

## Part A Code Audit

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| import_closure | 1 | compileall/import/source self-contained | 1 |  |
| source_chain | 1 | source/retention/route formulas | 7/7 |  |
| linec | 1 | linec fast/channel/audit-only | 9/9;8/8 |  |
| debt | 1 | peak/final/recovery/AUC formulas | 5/5 |  |
| mechanisms | 1 | planned terminal/source mechanisms | 7/7 |  |
| kernel_status | 1 | truth/status no false official claim | 1 |  |

## Part B Basis Efficiency

| carrier | rows | E1_pass_rows | S1_pass_rows | E1_pass_batch_sizes | best_forward_ratio | best_step_ratio | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 12 | 6 | 6 | 3 | 0.9305108277155166 | 0.46682782835473463 | S1OfficialLike |
| D-FOU | 12 | 3 | 3 | 3 | 1.0087315984225318 | 0.5089279836996342 | S1OfficialLike |

| carrier | profile_rows | near_E1_rows | best_forward_ratio | best_step_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 2 | 0 | 9.945103242391575 | 1.1491903929619172 | RejectedForThisVersion | forward_ratio;official_fused_missing;materialization;forward_ratio;step_ratio;official_fused_missing;materialization |
| D-RBF | 2 | 0 | 8.816428428771149 | 1.6501664590060272 | RejectedForThisVersion | forward_ratio;step_ratio;official_fused_missing;materialization |

## Part C Functional Update

| carrier | variant | v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-AdamW | 54 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-NoOpMatchedOverhead | 54 | -0.4933430420027839 | -1.4384159478876326 | -1.639491468667984 | -1.647347499926885 | -1.5352630929814444 | -1.4358961863650217 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-RandomMatchedNorm | 54 | -0.48931482323893793 | -1.4343930184841156 | -1.6354732325783483 | -1.6433252416275166 | -1.5325438098775015 | -1.433177790708012 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-SGD | 54 | -0.48823900355233085 | -1.4294601380825043 | -1.6273371720755543 | -1.632532403424934 | -1.5211187253395717 | -1.4211396392848756 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F10-T7-b1-cross-split-consensus-transfer | 9 | -0.35107966264088947 | -0.7804214225875007 | -0.49972931543986004 | -0.16040034095446268 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F25-loss-warm-to-b1-consensus-migration | 9 | -0.3284727864795261 | -0.3621560268931919 | -0.2032026383611891 | -0.06934237149026659 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T1-loss-cotangent-target | 9 | -0.335018965933058 | -0.355637874868181 | -0.18799111578199598 | -0.057160367568333946 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T5-random-matched-target | 9 | -0.4879032373428345 | -1.275543404950036 | -0.8925885491900973 | -0.3363882137669457 | nan | nan | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F30-gain-gated-loss-warm-b1-consensus | 9 | -0.3151562346352471 | -0.367200579908159 | -0.19381692674424914 | -0.06285006139013502 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F33-loss-warm-to-b1-consensus-b3-null | 9 | -0.3018566370010376 | -0.31244563394122654 | -0.16903516981336805 | -0.05690753128793505 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F35-loss-warm-to-view-consistent-loss | 9 | -0.32470248805152047 | -0.37968626949522233 | -0.19568810198042128 | -0.05814902981122335 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F36-lowbank-loss-b3-null | 9 | -0.48682689666748047 | -1.4254724317126803 | -1.6258274449242487 | -1.6340082155333624 | -1.515865898794598 | -1.412504815393024 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F37-loss-warm-to-lowbank-loss-b3-null | 9 | -0.3142985635333591 | -0.34857600927352905 | -0.18425064616733128 | -0.052726265456941396 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F38-gain-gated-lowbank-loss-b3-null | 9 | -0.4849250581529405 | -1.4243185387717352 | -1.6219656997256808 | -1.6307863593101501 | -1.5150579777028825 | -1.4138417343298595 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F39-loss-warm-to-gated-lowbank-loss-b3-null | 9 | -0.32272589206695557 | -0.3479761481285095 | -0.17721559604008993 | -0.05555439326498243 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-smallstep-alt50 | 9 | -0.4074300395117866 | -0.7993055979410807 | -0.3987191518147786 | -0.1072934369246165 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F68-adamw-boundary-to-gated-lowbank-b3-null | 9 | 0.007974889543321397 | -0.0018029676543341742 | -0.0005275342199537489 | 0.00357857346534729 | -0.0021484394868214927 | -0.007300949758953518 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F69-adamw-boundary-lowbank-anchor-antiwashout | 9 | 0.007455190022786458 | 0.00037747621536254883 | 0.0014568898412916395 | -0.0052944521109263105 | -0.013673616780175103 | -0.015228016508950127 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F8-KSW2-earlyboost-alt25 | 9 | 0.013114677535163032 | -0.08508913384543525 | -0.1133241057395935 | -0.11987284488148159 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-TCTRL-stable-random-target | 9 | -0.48025378915998673 | -1.4265883564949036 | -1.6247111161549885 | -1.622945682870017 | nan | nan | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW1-basis-estimate-readout-commit | 9 | -0.48900026745266384 | -1.427678492334154 | -1.6231121751997206 | -1.6301616761419508 | -1.5118636190891266 | -1.410119225581487 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW2-lowdegree-lowfreq-source-bank | 18 | -0.3112146854400635 | -0.34383275773790145 | -0.18236266242133248 | -0.07237915032439762 | -0.005438006586498684 | 0.027257995473013982 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | 54 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | 54 | -0.6965172070044058 | -1.4412056649172749 | -1.5431656042734783 | -1.5108974896095417 | -1.4270823101202648 | -1.3593043751186795 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | 54 | -0.6973756949106852 | -1.4420647223790486 | -1.5440239773856268 | -1.5117560128370922 | -1.428962293598387 | -1.361184557278951 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | 54 | -0.6976539470531322 | -1.4411829445097182 | -1.5415957150635895 | -1.506245055684337 | -1.416989588075214 | -1.3422040541966755 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F10-T7-b1-cross-split-consensus-transfer | 9 | -0.6235531700981988 | -0.761064714855618 | -0.3950337635146247 | -0.09203914139005873 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F25-loss-warm-to-b1-consensus-migration | 9 | -0.5565746360354953 | -0.361707435713874 | -0.15561231639650133 | -0.024669541252983943 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T1-loss-cotangent-target | 9 | -0.5758589373694526 | -0.3785022762086656 | -0.16423779726028442 | -0.07906320359971789 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T5-random-matched-target | 9 | -0.7079783810509576 | -1.1000941461986966 | -0.580721398194631 | -0.14567995071411133 | nan | nan | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F30-gain-gated-loss-warm-b1-consensus | 9 | -0.5704515245225694 | -0.3807315296596951 | -0.16896929343541464 | -0.04007487164603339 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F33-loss-warm-to-b1-consensus-b3-null | 9 | -0.572408119837443 | -0.3851510551240709 | -0.16058513191011217 | -0.031747274928622775 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F35-loss-warm-to-view-consistent-loss | 9 | -0.5819652345445421 | -0.3911263942718506 | -0.1751935150888231 | -0.046722100840674505 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F36-lowbank-loss-b3-null | 9 | -0.6718335416581895 | -1.3025149703025818 | -1.0049330062336392 | -0.3352172374725342 | 0.049605203999413386 | 0.15526423851648966 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-FOU | FOU-R4-k4-triton-no-materialize | F37-loss-warm-to-lowbank-loss-b3-null | 9 | -0.5649446646372477 | -0.38460779190063477 | -0.1727074384689331 | -0.026365160942077637 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F38-gain-gated-lowbank-loss-b3-null | 9 | -0.6818294260236952 | -1.3823313117027283 | -1.2923956910769145 | -0.7170495722028944 | -0.06800315777460735 | 0.13637220197253758 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;LateReboundNotRetained |
| D-FOU | FOU-R4-k4-triton-no-materialize | F39-loss-warm-to-gated-lowbank-loss-b3-null | 9 | -0.5683881706661649 | -0.36914222770267063 | -0.1582929425769382 | -0.026102377308739558 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F6-KSW2-density-smallstep-alt50 | 9 | -0.6515551937950982 | -0.8101509809494019 | -0.3558090594079759 | -0.08882333172692193 | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F68-adamw-boundary-to-gated-lowbank-b3-null | 9 | 0.008326477474636503 | 0.010237760014004178 | 0.011607607205708822 | 0.0015850961208343506 | -0.004651433891720242 | -0.005101071463690864 | 1 | 0 | 0 | 0 | ContinuousRetentionMissing |
| D-FOU | FOU-R4-k4-triton-no-materialize | F69-adamw-boundary-lowbank-anchor-antiwashout | 9 | 0.0104879273308648 | -0.004357702202267117 | 0.0022645857599046496 | -0.007774823241763645 | -0.01170006725523207 | -0.01502860254711575 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

_仅显示前 40 / 82 rows；完整 CSV 见 artifact。_

## Terminal Collapse Autopsy

| class | groups | mean_h3200 | mean_h4800 |
| --- | --- | --- | --- |
| DatasetHeterogeneity | 11 | 0.14841196934382123 | -0.02044672677011201 |
| UnknownTerminalCollapse | 1 | 0.14383376969231498 | -0.024836275312635634 |

| carrier | v22_id | h3200 | h4800 | source_derivative_h3200_to_h4800 | retention_h4800_over_h3200 | stable_random_h4800_positive_fraction | dataset_seed_heterogeneity_score | terminal_collapse_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source | 0.14383376969231498 | -0.024836275312635634 | -0.16867004500495061 | 0.0 | 0.0 | 0.75 | UnknownTerminalCollapse |
| MLP | MLP-F51-trainloss-late-hold-recovery-source | 0.14619363678826225 | -0.02271825737423367 | -0.1689118941624959 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F52-trainloss-late-lookahead-floor-source | 0.13905479510625204 | -0.029730810059441462 | -0.16878560516569352 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F53-trainloss-terminal-lookahead-floor-source | 0.16515672206878662 | -0.003963543309105767 | -0.16912026537789238 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F54-trainloss-early-terminal-lookahead-floor-source | 0.14249980449676514 | -0.02585990561379327 | -0.1683597101105584 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F63-trainloss-terminal-consensus-lookahead-floor-source | 0.13465211788813272 | -0.03395819001727634 | -0.16861030790540907 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F64-trainloss-terminal-selector-lookahead-floor-source | 0.1350817879041036 | -0.03357837597529093 | -0.16866016387939453 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 0.1626881096098158 | -0.005899740589989556 | -0.16858785019980535 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F71-trainloss-terminal-h3200-checkpoint-reentry | 0.1519218815697564 | -0.016645153363545735 | -0.16856703493330213 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F72-trainloss-terminal-hard-split-source | 0.1492934226989746 | -0.0192489226659139 | -0.16854234536488852 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F73-trainloss-terminal-adamw-lookahead | 0.14700835280948216 | -0.02212838331858317 | -0.16913673612806532 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |
| MLP | MLP-F74-trainloss-terminal-optimizer-selector | 0.15898103184170193 | -0.011182712184058296 | -0.17016374402576023 | 0.0 | 0.0 | 1.0 | DatasetHeterogeneity |

## MLP Source Lab / Selector

| f2_family | v22_id | h800 | h3200 | h4800 | h4800_positive_rows | retention_h4800_over_h3200 | MLP_FU_S3 | MLP_FU_S4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2-M10-control | CTRL-AdamW | -0.014133324004985667 | -0.9892447933791175 | -1.3887088693796643 | 0 | 0.0 | 0 | 0 |
| F2-M10-control | CTRL-NoOpMatchedOverhead | -1.0707866149920005 | -1.285798700792449 | -1.4480590193517624 | 0 | 0.0 | 0 | 0 |
| F2-M10-control | CTRL-RandomMatchedNorm | -1.0763936407036252 | -1.290993305189269 | -1.4529662215047412 | 0 | 0.0 | 0 | 0 |
| F2-M10-control | CTRL-SGD | -0.6033495331252062 | -0.010873916130217294 | 0.0 | 0 | 0.0 | 0 | 0 |
| F2-M1-strong-source-replay | MLP-F1-M2-strong-source | 0.45738417241308427 | nan | nan | 0 | 0.0 | 0 | 0 |
| F2-M1-strong-source-replay | MLP-F11-dual-timescale-retention-warm1200 | 0.4564314021004571 | nan | nan | 0 | 0.0 | 0 | 0 |
| F2-M2-weak-stable-replay | MLP-F2-M15-weak-stable | -0.6692412826750014 | nan | nan | 0 | 0.0 | 0 | 0 |
| F2-M2-weak-stable-replay | MLP-F23-source-vs-sgd-lookahead-gate | 0.4232677287525601 | nan | nan | 0 | 0.0 | 0 | 0 |
| F2-M4-slow-anchor-boundary-family | MLP-F40-adamw-boundary-to-momentum-source | 0.3393445677227444 | 0.0010147227181328668 | -0.20703593227598402 | 3 | -204.03202626323278 | 0 | 0 |
| F2-M4-slow-anchor-boundary-family | MLP-F41-adamw-boundary-to-dual-timescale-source | 0.3372511797481113 | 0.10489116774664985 | -0.06369584136539036 | 3 | -0.6072564805383699 | 0 | 0 |
| F2-other | MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source | 0.3387697670194838 | 0.0008061064614189996 | -0.21680072943369547 | 3 | -268.94801097619086 | 0 | 0 |
| F2-other | MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source | 0.35400940974553424 | 0.11868382162517971 | -0.14789369371202257 | 4 | -1.2461150280371975 | 0 | 0 |
| F2-M4-slow-anchor-boundary-family | MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source | 0.3612903422779507 | 0.11850794818666247 | -0.0509403215514289 | 3 | -0.42984729995656107 | 0 | 0 |
| F2-M4-slow-anchor-boundary-family | MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source | 0.34796935982174343 | 0.1136280828052097 | -0.05511781242158678 | 3 | -0.48507209715114286 | 0 | 0 |
| F2-other | MLP-F47-trainloss-gated-dual-timescale-tiny-late-source | 0.35250912772284615 | 0.010728200276692709 | -0.20400049289067587 | 3 | -19.015350909683537 | 0 | 0 |
| F2-other | MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source | 0.3310249249140422 | 0.11736520131429036 | -0.05160053571065267 | 3 | -0.43965788098017605 | 0 | 0 |
| F2-other | MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source | 0.3579259382353889 | 0.14383376969231498 | -0.024836275312635634 | 4 | -0.1726734644149609 | 0 | 0 |
| F2-other | MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source | 0.3347284330262078 | 0.1219278375307719 | -0.04832035303115845 | 4 | -0.3963028788972285 | 0 | 0 |
| F2-other | MLP-F51-trainloss-late-hold-recovery-source | 0.35773270659976536 | 0.14619363678826225 | -0.02271825737423367 | 3 | -0.1553984008697819 | 0 | 0 |
| F2-other | MLP-F52-trainloss-late-lookahead-floor-source | 0.35123587979210746 | 0.13905479510625204 | -0.029730810059441462 | 3 | -0.21380643534603816 | 0 | 0 |
| F2-M3-terminal-family-replay | MLP-F53-trainloss-terminal-lookahead-floor-source | 0.37619424528545803 | 0.16515672206878662 | -0.003963543309105767 | 3 | -0.023998679917218137 | 0 | 0 |
| F2-other | MLP-F54-trainloss-early-terminal-lookahead-floor-source | 0.3537594742245144 | 0.14249980449676514 | -0.02585990561379327 | 3 | -0.18147327082389303 | 0 | 0 |
| F2-other | MLP-F55-split-fisher-source-observability-reset | -0.10597315099504259 | -0.02089711692598131 | -0.04255129893620809 | 4 | 0.0 | 0 | 0 |
| F2-other | MLP-F56-momentum-warm-split-fisher-source | 0.1787186066309611 | -0.563570585515764 | -0.5396582616700066 | 0 | 0.0 | 0 | 0 |
| F2-M5-fast-slow-memory-source-writer | MLP-F57-adamw-boundary-dual-timescale-antiwashout-source | 0.29924317863252425 | -0.012103550963931613 | -0.2369269993570116 | 4 | 0.0 | 0 | 0 |
| F2-M5-fast-slow-memory-source-writer | MLP-F58-adamw-boundary-dual-timescale-source-anchor | 0.23931911256578234 | -0.08447160323460896 | -0.29526517788569134 | 3 | 0.0 | 0 | 0 |
| F2-M5-fast-slow-memory-source-writer | MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry | 0.3614519172244602 | 0.01199809710184733 | -0.1983597609731886 | 3 | -16.53260173587422 | 0 | 0 |
| F2-M7-matrix-readout-channel-writer | MLP-F60-adamw-boundary-dual-timescale-readout-channel | 0.3485870957374573 | -0.0033842987484402126 | -0.21282377507951525 | 3 | 0.0 | 0 | 0 |
| F2-M7-matrix-readout-channel-writer | MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel | 0.34573060274124146 | 0.011278331279754639 | -0.20057602061165702 | 3 | -17.784193036757525 | 0 | 0 |
| F2-other | MLP-F62-trainloss-terminal-projected-lookahead-floor-source | 0.3297286894586351 | 0.11756877766715156 | -0.050098571512434215 | 3 | -0.42612139469773225 | 0 | 0 |

_仅显示前 30 / 38 rows；完整 CSV 见 artifact。_

| predictor | kind | AUC_continuous_retention | pass | precommit_selector_pass | retrospective_only_flag | selector_decision |
| --- | --- | --- | --- | --- | --- | --- |
| train_loss_h100 | train_only | 0.95480109739369 | 0 | 0 | 0 | NoPrecommitPass |
| train_loss_h400 | train_only | 0.9409073094258279 | 1 | 0 | 1 | RetrospectiveOnly |
| train_loss_h800 | train_only | 0.9434548304918675 | 1 | 0 | 1 | RetrospectiveOnly |
| train_loss_drop_h100_h800 | train_only | 0.21192435822065453 | 0 | 0 | 0 | NoPrecommitPass |
| CEp99_h800 | audit_readback | 0.7458847736625515 | 0 | 0 | 0 | NoPrecommitPass |
| Brier_h800 | audit_readback | 0.9610523221634333 | 0 | 0 | 0 | NoPrecommitPass |
| LineC_channel_loss_h800 | audit_readback | 0.6731040564373898 | 0 | 0 | 0 | NoPrecommitPass |
| ActuationR2_h800 | function_readback | 0.24161425576519915 | 0 | 0 | 0 | NoPrecommitPass |
| B2_transfer_gain_h800 | function_readback | 0.19916142557651992 | 0 | 0 | 0 | NoPrecommitPass |
| source_channel_projection_h800 | train_stream_diagnostic | 0.08438155136268344 | 0 | 0 | 0 | NoPrecommitPass |
| source_state_gate_accept_h800 | train_stream_diagnostic | 0.5266937669376693 | 0 | 0 | 0 | NoPrecommitPass |
| generalization_gate_accept_h800 | train_stream_diagnostic | 0.5279563608026495 | 0 | 0 | 0 | NoPrecommitPass |
| C4_E1_split_fisher_snr_h800 | train_stream_diagnostic |  | 0 | 0 | 0 | NoPrecommitPass |
| C4_E6_split_fisher_coherence_h800 | train_stream_diagnostic |  | 0 | 0 | 0 | NoPrecommitPass |
| C4_E7_noise_reservoir_separation_h800 | train_stream_diagnostic |  | 0 | 0 | 0 | NoPrecommitPass |
| source_h400 | source_readback_forbidden | 0.8193121693121693 | 0 | 0 | 0 | NoPrecommitPass |

## Function-Space Target Reset

| carrier | v22_id | target_family | ActuationR2 | B2_transfer_gain | random_target_B2_gain | early_source_chain_rate | h3200_retention_rate | h4800_retention_rate | Target_S2 | Target_S3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | F10-T7-b1-cross-split-consensus-transfer | T2-cross-split-consensus | 0.2750802238782247 | 0.04400197002622816 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F25-loss-warm-to-b1-consensus-migration | T2-cross-split-consensus | 0.5971961087650723 | 0.1237022876739502 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F3-T1-loss-cotangent-target | T1-loss-cotangent | 0.5611197749773661 | 0.13524330655733743 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F3-T5-random-matched-target | T9-random-matched | 0.43602176507314044 | -0.008591161833869087 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F30-gain-gated-loss-warm-b1-consensus | T2-cross-split-consensus | 0.5935484303368462 | 0.11416584915584987 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F33-loss-warm-to-b1-consensus-b3-null | T2-cross-split-consensus | 0.5801834662755331 | 0.12973969512515598 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F35-loss-warm-to-view-consistent-loss | T-other | 0.5958307186762491 | 0.13939236932330662 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.5779258476363288 | 0.13759788539674547 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.6032821271154616 | 0.12396695547633702 | -0.008591161833869087 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F10-T7-b1-cross-split-consensus-transfer | T2-cross-split-consensus | 0.24739316436979505 | 0.02276786168416341 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F25-loss-warm-to-b1-consensus-migration | T2-cross-split-consensus | 0.612061427699195 | 0.14594736033015782 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F3-T1-loss-cotangent-target | T1-loss-cotangent | 0.632168538040585 | 0.1273758245839013 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F3-T5-random-matched-target | T9-random-matched | 0.5297892093658447 | -0.012681000762515597 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F30-gain-gated-loss-warm-b1-consensus | T2-cross-split-consensus | 0.6486111217074924 | 0.1289143827226427 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F33-loss-warm-to-b1-consensus-b3-null | T2-cross-split-consensus | 0.6600163645214505 | 0.12876971728271908 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F35-loss-warm-to-view-consistent-loss | T-other | 0.628508190313975 | 0.13369801971647474 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.6469230850537618 | 0.14389371540811327 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | T3-low-degree-low-frequency | 0.6506590909428067 | 0.11701262990633647 | -0.012681000762515597 | 0.0 | 0.0 | 0.0 | 0 | 0 |

## KAN Source Writer

| carrier | v22_id | writer_family | h800 | h3200 | h4800 | h3200_positive_rows | h4800_positive_rows | KAN_FU_S2 | KAN_FU_S3 | same_run_efficiency_S1_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | F36-lowbank-loss-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -1.6258274449242487 | -1.515865898794598 | -1.412504815393024 | 0 | 0 | 0 | 0 | 6 |
| D-CHE | F38-gain-gated-lowbank-loss-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -1.6219656997256808 | -1.5150579777028825 | -1.4138417343298595 | 0 | 0 | 0 | 0 | 6 |
| D-CHE | F68-adamw-boundary-to-gated-lowbank-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -0.0005275342199537489 | -0.0021484394868214927 | -0.007300949758953518 | 3 | 3 | 0 | 0 | 6 |
| D-CHE | F69-adamw-boundary-lowbank-anchor-antiwashout | D-CHE-F5b-low-degree-low-frequency-bank | 0.0014568898412916395 | -0.013673616780175103 | -0.015228016508950127 | 3 | 4 | 0 | 0 | 6 |
| D-CHE | KSW1-basis-estimate-readout-commit | D-CHE-F5a-readout-commit | -1.6231121751997206 | -1.5118636190891266 | -1.410119225581487 | 0 | 0 | 0 | 0 | 6 |
| D-CHE | KSW2-lowdegree-lowfreq-source-bank | D-CHE-F5b-low-degree-low-frequency-bank | -0.18236266242133248 | -0.005438006586498684 | 0.027257995473013982 | 4 | 4 | 0 | 0 | 6 |
| D-FOU | F36-lowbank-loss-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | -1.0049330062336392 | 0.049605203999413386 | 0.15526423851648966 | 6 | 9 | 0 | 0 | 3 |
| D-FOU | F38-gain-gated-lowbank-loss-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | -1.2923956910769145 | -0.06800315777460735 | 0.13637220197253758 | 4 | 8 | 0 | 0 | 3 |
| D-FOU | F68-adamw-boundary-to-gated-lowbank-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | 0.011607607205708822 | -0.004651433891720242 | -0.005101071463690864 | 2 | 2 | 0 | 0 | 3 |
| D-FOU | F69-adamw-boundary-lowbank-anchor-antiwashout | D-FOU-F5b-low-degree-low-frequency-bank | 0.0022645857599046496 | -0.01170006725523207 | -0.01502860254711575 | 3 | 3 | 0 | 0 | 3 |
| D-FOU | KSW1-basis-estimate-readout-commit | D-FOU-F5a-readout-commit | -1.536535296175215 | -1.4180500507354736 | -1.345011830329895 | 0 | 0 | 0 | 0 | 3 |
| D-FOU | KSW2-lowdegree-lowfreq-source-bank | D-FOU-F5b-low-degree-low-frequency-bank | -0.1767431596914927 | -0.10215700334972805 | -0.13887516657511392 | 0 | 1 | 0 | 0 | 3 |

## 修改记录

- 修正 `dgkan/fu/source_chain.py`：early-source 改为 `>=0.005`，加入 control-equivalent 排除、h4800 continuous 和 terminal-collapse 标记。
- 新增 `dgkan/fu/debt_accounting.py`：提供 audit-only debt peak/final/recovery 与 AUCtime ratio 公式测试。
- 新增 v22.01 runner：S0.8 truth gate、efficiency recheck、D-RAT/D-RBF active repair、terminal-collapse autopsy、MLP source lab、function-space target reset、KAN writer audit、finalizer。
- v22.01 functional 数据来自 v22 raw source-retention matrix 重新聚合；没有复用旧 route 结论。

## 分析 / Insight / 结论

- 修正 source-chain rule 后，zero/control rows 不再能打开 early chain；h3200 candidate 必须同时满足 h100/h400/h800 >= 0.005。
- 当前主 blocker 仍是 terminal collapse：有 h3200 continuous groups，但 h4800 retained group 仍未闭合，因此不能 promotion。
- F46/C4 train-loss 类指标在 v22 中仍只能作为 retrospective taxonomy；本轮没有 precommit independent selector pass，不能用作方向成功证据。
- D-CHE/D-FOU efficiency 是真实进展，但 functional source retention 未闭合，所以 blocker 不是 kernel officialization 单项。
- D-RAT/D-RBF 已执行 active repair/profiling；若 Near-E1 未达成，limited functional smoke 按 gate deferred，不写 carrier source breakthrough。
- KAN low-degree/low-frequency/readout writer 仍未形成 KAN-FU-S3；若 MLP 也没有 MLP-FU-S3，则当前 train-stream retained target/source observability theory 仍不足。

- v22_01_code_review_packet.zip: size=1410867 sha256=9858a75f489d328b503f78d9869054f4738edf56793ec6b69fa92ffff17b6481
- v22_01_results_bundle.zip: size=1311801 sha256=7490322340810e96b561c55dd0b087b854c1b223daf3a9dce1c4e1a7bed61df0

## v22.01 continuation smoke: F75-F77 syntax/run check

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_smoke`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 0 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: not_measured_steps120

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F75-dataset-invariant-poprisk-slow-source | 1 | -0.7583121061325073 | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F76-dataset-invariant-readout-consensus-source | 1 | -0.6789335012435913 | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F77-source-conserving-optimizer-only | 1 | 0.31180906295776367 | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 1 | 0.0 | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 1 | -0.571987509727478 | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增 `M116-DatasetInvariantPopRiskSlowFU` / `MLP-F75-dataset-invariant-poprisk-slow-source`：用 train split exact PopRisk/SNR 的 sign-consensus 和慢状态作为 DatasetHeterogeneity 分支的 train-only invariant selector。
- 新增 `M117-DatasetInvariantReadoutConsensusFU` / `MLP-F76-dataset-invariant-readout-consensus-source`：用 train split readout-gradient consensus，并投掉 corrupt-label 正投影，检验低容量 readout carrier 是否能承接跨数据集不变量。
- 新增 `M118-SourceConservingOptimizerOnlyFU` / `MLP-F77-source-conserving-optimizer-only`：沿用 AdamW boundary + dual-timescale source warmup，但 h3200 后只允许 source-preserving projected optimizer；gate 不通过时 hold，不再 SGD fallback，用于验证 OptimizerWashout。
- 扩展 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`，使 F75-F77 走既有 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。
- 新增 `experiments/run_v22_01_continue_terminal_repair.py`：只读取落盘 fresh matrix，生成 continuation source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F75-F77 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F75-F77 terminal-collapse repair

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 0 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: -0.460429588953654

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F75-dataset-invariant-poprisk-slow-source | 9 | -0.9853300584687127 | -0.5260503954357572 | -0.542392823431227 | -0.7592335343360901 | -0.8859658042589823 | -0.8279068271319071 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F76-dataset-invariant-readout-consensus-source | 9 | -1.0016246967845492 | -0.5457395050260756 | -0.5670836104287041 | -0.7941816449165344 | -0.9391998516188728 | -0.8970164524184333 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F77-source-conserving-optimizer-only | 9 | -0.019946429464552138 | -0.05650415023167928 | -0.08056249221165974 | -0.31666482819451225 | -0.4793124066458808 | -0.460429588953654 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | -0.2571325699488322 | -0.5585073365105523 | -1.0504576166470845 | -1.5069670610957675 | -1.6925712757640414 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -0.9593977464569939 | -0.5067642397350736 | -0.533097677760654 | -0.7714051206906637 | -0.9402147730191549 | -0.9211042258474562 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.0167191624641418 | -0.5643292400572035 | -0.5913352833853828 | -0.8288919197188483 | -0.9977009097735087 | -0.9799526598718431 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -0.9261961049503751 | -0.2854050662782457 | -0.1031907664404975 | -0.021287375026279025 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增 `M116-DatasetInvariantPopRiskSlowFU` / `MLP-F75-dataset-invariant-poprisk-slow-source`：用 train split exact PopRisk/SNR 的 sign-consensus 和慢状态作为 DatasetHeterogeneity 分支的 train-only invariant selector。
- 新增 `M117-DatasetInvariantReadoutConsensusFU` / `MLP-F76-dataset-invariant-readout-consensus-source`：用 train split readout-gradient consensus，并投掉 corrupt-label 正投影，检验低容量 readout carrier 是否能承接跨数据集不变量。
- 新增 `M118-SourceConservingOptimizerOnlyFU` / `MLP-F77-source-conserving-optimizer-only`：沿用 AdamW boundary + dual-timescale source warmup，但 h3200 后只允许 source-preserving projected optimizer；gate 不通过时 hold，不再 SGD fallback，用于验证 OptimizerWashout。
- 扩展 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`，使 F75-F77 走既有 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。
- 新增 `experiments/run_v22_01_continue_terminal_repair.py`：只读取落盘 fresh matrix，生成 continuation source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F75-F77 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation final artifact update

- post-continuation S0.8 truth gate: `v22_01_code_truth_gate.csv`, all checks pass.
- v22_01_code_review_packet.zip: size=1469041 sha256=91bda5284e73c2271b5a04bd9e9802b4d9f349935c0f175aef8a020f58ce747c
- v22_01_results_bundle.zip: size=1620345 sha256=60f282c9bc59409f6e0fb7212a0047d5a5850257801517b053e906955c6eacd1

## v22.01 continuation: F78-F80 retained-target/source-observability repair

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f80_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 0 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: -0.514527353975508

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F78-terminal-source-conserving-route | 9 | -0.15653717517852783 | -0.268445266617669 | -0.29320891698201496 | -0.5302442775832282 | -0.6990611884329054 | -0.6799682180086771 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F79-trainloss-risk-profile-route | 9 | -0.09669452905654907 | -0.20271107885572645 | -0.2283464272816976 | -0.46492934889263576 | -0.6337442066934373 | -0.6146387987666659 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F80-debt-aware-source-gate | 9 | -0.03888762659496731 | -0.10432642036014134 | -0.1284964680671692 | -0.36482216252221 | -0.5336362057261996 | -0.514527353975508 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | -0.2571325699488322 | -0.5585073365105523 | -1.0504576166470845 | -1.5069670610957675 | -1.6925712757640414 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -0.9593977464569939 | -0.5067642397350736 | -0.533097677760654 | -0.7714051206906637 | -0.9402147730191549 | -0.9211042258474562 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.0167191624641418 | -0.5643292400572035 | -0.5913352833853828 | -0.8288919197188483 | -0.9977009097735087 | -0.9799526598718431 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -0.9261961049503751 | -0.2854050662782457 | -0.1031907664404975 | -0.021287375026279025 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增 `M119-TerminalSourceConservingRouteFU` / `MLP-F78-terminal-source-conserving-route`：复用 F53 系列能到 h3200 的 AdamW-boundary + dual-timescale source path，但 terminal phase 只在 source-axis / source-projected / split-consensus 三个 source-conserving 候选之间做 train-only precommit；全部 reject 时 hold。
- 新增 `M120-TrainLossRiskProfileRouteFU` / `MLP-F79-trainloss-risk-profile-route`：在 F78 route 上加入 train split A/B loss risk-profile；split risk 失衡时只允许 source-axis 保守候选，检验 DatasetHeterogeneity 是否需要 train-only invariant route 而不是 terminal floor。
- 新增 `M121-DebtAwareSourceGateFU` / `MLP-F80-debt-aware-source-gate`：在 F78 route 上加入 train-only debt gate；只有当前 train loss debt 存在时才允许 projected/consensus catch-up，否则只保留 source-axis 候选，用于检验 DebtCollapse 假设。
- 扩展 `experiments/run_v17_common.py` 的 terminal branch、`dgkan/fu/mechanisms.py` 的 manifest/contract、`experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py` 的 MLP scope，使 F78-F80 走 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F78_F79_F80 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F81-F83 ungated h3200 warm retained-target repair

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f81_f83_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 0 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: -0.45469733079274494

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F81-ungated-warm-terminal-source-route | 9 | -0.08341315057542589 | -0.2931891679763794 | -0.31715720229678684 | -0.5503895746337043 | -0.7101349896854825 | -0.6910313301616244 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F82-ungated-warm-risk-profile-route | 9 | -0.143451193968455 | -0.2320158084233602 | -0.2589491738213433 | -0.49609459108776516 | -0.6592469414075216 | -0.6401442885398865 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F83-ungated-warm-debt-raw-bailout | 9 | -0.016139129797617596 | -0.04862670765982734 | -0.07314776049719916 | -0.30995086828867596 | -0.47380512952804565 | -0.45469733079274494 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | -0.2571325699488322 | -0.5585073365105523 | -1.0504576166470845 | -1.5069670610957675 | -1.6925712757640414 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -0.9593977464569939 | -0.5067642397350736 | -0.533097677760654 | -0.7714051206906637 | -0.9402147730191549 | -0.9211042258474562 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.0167191624641418 | -0.5643292400572035 | -0.5913352833853828 | -0.8288919197188483 | -0.9977009097735087 | -0.9799526598718431 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -0.9261961049503751 | -0.2854050662782457 | -0.1031907664404975 | -0.021287375026279025 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增 `M122-UngatedWarmTerminalSourceRouteFU` / `MLP-F81-ungated-warm-terminal-source-route`：先关闭早期 train-loss selector，强制复用 AdamW-boundary + fast source warm 到 h3200；terminal phase 再只允许 source-axis/source-projected/split-consensus route 或 hold，用于分离 early selector failure 与 terminal transition failure。
- 新增 `M123-UngatedWarmRiskProfileRouteFU` / `MLP-F82-ungated-warm-risk-profile-route`：在 F81 的 h3200 warm 基础上，把 DatasetHeterogeneity fallback 落成 train split A/B risk-profile；risk 失衡时只允许 source-axis 保守候选。
- 新增 `M124-UngatedWarmDebtRawBailoutFU` / `MLP-F83-ungated-warm-debt-raw-bailout`：在 F81 的 h3200 warm 基础上，只在 train-only loss debt 存在时允许 positive-only raw terminal bailout，否则走 source-conserving route/hold，用于检验 DebtCollapse 是否需要有限 raw catch-up。
- 扩展 `experiments/run_v17_common.py` 的 terminal branch、`dgkan/fu/mechanisms.py` 的 manifest/contract、`experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py` 的 MLP scope，使 F81-F83 走 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F81_F82_F83 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F84-F86 terminal-collapse hold without future leakage

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 0 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: -0.5175521373748779

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F84-h2400-terminal-hold-source | 9 | -0.13206974665323892 | -0.3062124517228868 | -0.3290703429116143 | -0.5654446615113152 | -0.7342582609918382 | -0.7151477138201395 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F85-h2800-terminal-hold-source | 9 | -0.13897497786415947 | -0.2519453896416558 | -0.276842647128635 | -0.5129691825972663 | -0.6817829542689853 | -0.6626724070972867 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F86-h2400-debt-bailout-source | 9 | -0.020544138219621446 | -0.10826621452967326 | -0.1305447949303521 | -0.3678503566318088 | -0.5366626845465766 | -0.5175521373748779 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | -0.2571325699488322 | -0.5585073365105523 | -1.0504576166470845 | -1.5069670610957675 | -1.6925712757640414 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -0.9593977464569939 | -0.5067642397350736 | -0.533097677760654 | -0.7714051206906637 | -0.9402147730191549 | -0.9211042258474562 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.0167191624641418 | -0.5643292400572035 | -0.5913352833853828 | -0.8288919197188483 | -0.9977009097735087 | -0.9799526598718431 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -0.9261961049503751 | -0.2854050662782457 | -0.1031907664404975 | -0.021287375026279025 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增 `M125-TrainLossH2400CheckpointHoldFU` / `MLP-F84-h2400-terminal-hold-source`：复用 F53-style train-loss gated dual-timescale source path 到 h2400，之后不再提交 source/optimizer 更新，检验高 h2400 source 是否可通过 terminal hold 跨过 h4800。
- 新增 `M126-TrainLossH2800CheckpointHoldFU` / `MLP-F85-h2800-terminal-hold-source`：把 hold 起点后移到 h2800，检验 h2400-h3200 的 source decay 是否来自过早冻结还是 terminal drift。
- 新增 `M127-TrainLossH2400DebtBailoutFU` / `MLP-F86-h2400-debt-bailout-source`：h2400 后默认 hold，只在 train-only loss debt 明显存在时允许 0.03x positive-only lookahead micro bailout，检验 DebtCollapse 是否需要极小无泄漏 catch-up。
- 扩展 `experiments/run_v17_common.py` 的 terminal branch、`dgkan/fu/mechanisms.py` 的 manifest/contract、`experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py` 的 MLP scope，使 F84-F86 走 fresh source-retention runner 和 v22.01 source-chain reaggregation 口径。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F84_F85_F86 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F53/F70/F74 fresh terminal-family replay

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_f70_f74_replay_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 0 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: -0.5206785599390665

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | -0.12279798587163289 | -0.3048023515277439 | -0.32988613181644016 | -0.5647294984923469 | -0.7335471245977614 | -0.7144517832332187 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 9 | -0.08592069811291164 | -0.20483995808495414 | -0.2299711439344618 | -0.4660869240760803 | -0.634902748796675 | -0.6157922016249763 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F74-trainloss-terminal-optimizer-selector | 9 | -0.0261256562338935 | -0.10795525709788005 | -0.13297305504480997 | -0.36953753895229763 | -0.5383505622545878 | -0.5206785599390665 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | -0.2571325699488322 | -0.5585073365105523 | -1.0504576166470845 | -1.5069670610957675 | -1.6925712757640414 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -0.9593977464569939 | -0.5067642397350736 | -0.533097677760654 | -0.7714051206906637 | -0.9402147730191549 | -0.9211042258474562 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.0167191624641418 | -0.5643292400572035 | -0.5913352833853828 | -0.8288919197188483 | -0.9977009097735087 | -0.9799526598718431 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -0.9261961049503751 | -0.2854050662782457 | -0.1031907664404975 | -0.021287375026279025 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 未新增机制；在当前代码与当前 matched-control attribution 口径下 fresh rerun `MLP-F53` / `MLP-F70` / `MLP-F74`，检验 v22.01 baseline terminal-source family 是否可复现。
- replay 只作为校准与故障定位证据；若 fresh replay 不能复现 h3200 continuous chain，后续不能继续基于旧 raw matrix 的 F53 near-miss 设计 confirmation。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F53_F70_F74_REPLAY 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 F53/F70/F74 fresh replay delta audit

- audit artifact: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_f53_f70_f74_fresh_replay_delta.csv`
- conclusion: current-code fresh replay did not reproduce the old reaggregated F53/F70/F74 h3200 terminal-collapse candidates; therefore subsequent repairs cannot use the old F53 near-miss as causal promotion evidence without first resolving attribution/provenance drift.

| v22_id | old_h100 | fresh_h100 | delta_h100 | old_h800 | fresh_h800 | delta_h800 | old_h3200 | fresh_h3200 | delta_h3200 | old_h4800 | fresh_h4800 | delta_h4800 | old_early | fresh_early | old_continuous | fresh_continuous | old_h4800_group | fresh_h4800_group | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F53-trainloss-terminal-lookahead-floor-source | 0.048824826876322426 | -0.12279798587163289 | -0.17162281274795532 | 0.37619424528545803 | -0.32988613181644016 | -0.7060803771018982 | 0.16515672206878662 | -0.7335471245977614 | -0.898703846666548 | -0.003963543309105767 | -0.7144517832332187 | -0.7104882399241129 | 1 | 0 | 1 | 0 | 0 | 0 | fresh replay did not reproduce old terminal-collapse candidate under current code/control attribution |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 0.033847285641564265 | -0.08592069811291164 | -0.1197679837544759 | 0.374403801229265 | -0.2299711439344618 | -0.6043749451637268 | 0.1626881096098158 | -0.634902748796675 | -0.7975908584064908 | -0.005899740589989556 | -0.6157922016249763 | -0.6098924610349867 | 1 | 0 | 1 | 0 | 0 | 0 | fresh replay did not reproduce old terminal-collapse candidate under current code/control attribution |
| MLP-F74-trainloss-terminal-optimizer-selector | 0.03304861651526557 | -0.0261256562338935 | -0.05917427274915907 | 0.3710273702939351 | -0.13297305504480997 | -0.5040004253387451 | 0.15898103184170193 | -0.5383505622545878 | -0.6973315940962896 | -0.011182712184058296 | -0.5206785599390665 | -0.5094958477550082 | 1 | 0 | 1 | 0 | 0 | 0 | fresh replay did not reproduce old terminal-collapse candidate under current code/control attribution |

## v22.01 final artifact update after fresh replay delta

- v22_01_code_review_packet.zip: size=1672586 sha256=97c913f9962c21e568277a5794a3f5acbefcbf933bfa26eac5566b72ef493606
- v22_01_results_bundle.zip: size=2762057 sha256=f31bca1722cd7bdb6c735a3f8ceced37a7a1235ba05c9071398c64c25075c1ea

## v22.01 F53/F70/F74 replay provenance audit

- artifact: `v22_01_f53_f70_f74_job_index_provenance_audit.csv`
- artifact: `v22_01_f53_f70_f74_run_shape_summary.csv`
- artifact: `v22_01_f53_f70_f74_candidate_seed_delta.csv`
- conclusion: old v22 terminal-family runs and the v22.01 combined fresh replay do not have the same run shape/job-index assignment; since `train_seed = 210100 + job_index + init_seed_offset`, the combined replay changes model initialization seeds and is not a same-seed reproduction test.

| source | run_label | v21_id | rows | job_index_minmax | train_seed_minmax | mean_h4800 |
| --- | --- | --- | --- | --- | --- | --- |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-AdamW | 9 | 2..42 | 210102..210142 | -1.386982838312785 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-NoOpMatchedOverhead | 9 | 4..44 | 210104..210144 | -1.4673935572306316 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-RandomMatchedNorm | 9 | 3..43 | 210103..210143 | -1.470809433195326 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-SGD | 9 | 1..41 | 210101..210141 | 0.0 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | 0..40 | 210100..210140 | -0.003963543309105767 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-AdamW | 9 | 2..42 | 210102..210142 | -1.386982838312785 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-NoOpMatchedOverhead | 9 | 4..44 | 210104..210144 | -1.4673935572306316 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-RandomMatchedNorm | 9 | 3..43 | 210103..210143 | -1.470809433195326 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-SGD | 9 | 1..41 | 210101..210141 | 0.0 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 9 | 0..40 | 210100..210140 | -0.005899740589989556 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-AdamW | 9 | 2..42 | 210102..210142 | -1.386982838312785 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-NoOpMatchedOverhead | 9 | 4..44 | 210104..210144 | -1.4673935572306316 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-RandomMatchedNorm | 9 | 3..43 | 210103..210143 | -1.470809433195326 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-SGD | 9 | 1..41 | 210101..210141 | 0.0 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | MLP-F74-trainloss-terminal-optimizer-selector | 9 | 0..40 | 210100..210140 | -0.011182712184058296 |


_无落盘 rows。_

## v22.01 F53/F70/F74 replay provenance audit corrected

- artifact: `v22_01_f53_f70_f74_job_index_provenance_audit_corrected.csv`
- artifact: `v22_01_f53_f70_f74_run_shape_summary_corrected.csv`
- artifact: `v22_01_f53_f70_f74_candidate_seed_delta_corrected.csv`
- correction: previous provenance pass filtered the fresh run with the wrong run_label; this corrected pass uses `v2201_cont_f53_f70_f74_replay_full` and supersedes that table for seed/run-shape conclusions.
- conclusion: paired candidate rows=27, same train-seed pairs=1; old v22 single-candidate terminal runs use candidate job_index ranges `0..40`, while the combined v22.01 replay interleaves F53/F70/F74 and changes F70/F74 candidate job_index/train_seed assignments. A single-candidate fresh replay is therefore required before treating old-vs-fresh deltas as mechanism failure.

| source | run_label | v21_id | rows | job_index_minmax | train_seed_minmax | mean_h4800 |
| --- | --- | --- | --- | --- | --- | --- |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-AdamW | 9 | 2..42 | 210102..210142 | -1.386982838312785 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-NoOpMatchedOverhead | 9 | 4..44 | 210104..210144 | -1.4673935572306316 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-RandomMatchedNorm | 9 | 3..43 | 210103..210143 | -1.470809433195326 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | CTRL-SGD | 9 | 1..41 | 210101..210141 | 0.0 |
| old_v22 | v2200_f53_terminal_lookahead_floor_full_h4800 | MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | 0..40 | 210100..210140 | -0.003963543309105767 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-AdamW | 9 | 2..42 | 210102..210142 | -1.386982838312785 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-NoOpMatchedOverhead | 9 | 4..44 | 210104..210144 | -1.4673935572306316 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-RandomMatchedNorm | 9 | 3..43 | 210103..210143 | -1.470809433195326 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | CTRL-SGD | 9 | 1..41 | 210101..210141 | 0.0 |
| old_v22 | v2200_f70_terminal_positive_lookahead_floor_full_h4800 | MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 9 | 0..40 | 210100..210140 | -0.005899740589989556 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-AdamW | 9 | 2..42 | 210102..210142 | -1.386982838312785 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-NoOpMatchedOverhead | 9 | 4..44 | 210104..210144 | -1.4673935572306316 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-RandomMatchedNorm | 9 | 3..43 | 210103..210143 | -1.470809433195326 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | CTRL-SGD | 9 | 1..41 | 210101..210141 | 0.0 |
| old_v22 | v2200_f74_terminal_optimizer_selector_full_h4800 | MLP-F74-trainloss-terminal-optimizer-selector | 9 | 0..40 | 210100..210140 | -0.011182712184058296 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | CTRL-AdamW | 9 | 4..60 | 210104..210160 | -1.6925712757640414 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | CTRL-NoOpMatchedOverhead | 9 | 6..62 | 210106..210162 | -0.9211042258474562 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | CTRL-RandomMatchedNorm | 9 | 5..61 | 210105..210161 | -0.9799526598718431 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | CTRL-SGD | 9 | 3..59 | 210103..210159 | 0.0 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | 0..56 | 210100..210156 | -0.7144517832332187 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | MLP-F70-trainloss-terminal-positive-lookahead-floor-source | 9 | 1..57 | 210101..210157 | -0.6157922016249763 |
| v22_01_fresh_replay | v2201_cont_f53_f70_f74_replay_full | MLP-F74-trainloss-terminal-optimizer-selector | 9 | 2..58 | 210102..210158 | -0.5206785599390665 |

| v21_id | dataset | seed | old_job_index | fresh_job_index | old_train_seed | fresh_train_seed | old_h3200 | fresh_h3200 | old_h4800 | fresh_h4800 | old_early_cont | fresh_early_cont | same_train_seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F53-trainloss-terminal-lookahead-floor-source | Fashion-MNIST | 0 | 15 | 21 | 210115 | 210121 | 0.040415942668914795 | -0.9557486772537231 | -0.015426993370056152 | -0.8794459104537964 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | Fashion-MNIST | 1 | 20 | 28 | 210120 | 210128 | -0.010123670101165771 | -0.5210501551628113 | -0.0944567322731018 | -0.5575787425041199 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | Fashion-MNIST | 2 | 25 | 35 | 210125 | 210135 | -0.030561089515686035 | -2.617416024208069 | -0.12596487998962402 | -2.3779698610305786 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | KMNIST | 0 | 30 | 42 | 210130 | 210142 | 0.16402530670166016 | -0.5523775815963745 | -0.0505518913269043 | -0.5281580686569214 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | KMNIST | 1 | 35 | 49 | 210135 | 210149 | 0.212191641330719 | -0.526878833770752 | -0.02529221773147583 | -0.5698630809783936 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | KMNIST | 2 | 40 | 56 | 210140 | 210156 | 0.02587294578552246 | -0.08477377891540527 | -0.12781846523284912 | -0.15253007411956787 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | MNIST | 0 | 0 | 0 | 210100 | 210100 | 0.3895149230957031 | -0.7474198341369629 | 0.12493276596069336 | -0.7605167627334595 | / | / | 1 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | MNIST | 1 | 5 | 7 | 210105 | 210107 | 0.2799164652824402 | -0.2806292772293091 | 0.07123345136642456 | -0.272269606590271 | / | / | 0 |
| MLP-F53-trainloss-terminal-lookahead-floor-source | MNIST | 2 | 10 | 14 | 210110 | 210114 | 0.4151580333709717 | -0.3156299591064453 | 0.2076730728149414 | -0.33173394203186035 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | Fashion-MNIST | 0 | 15 | 22 | 210115 | 210122 | 0.04948955774307251 | -0.719591498374939 | -0.005643010139465332 | -0.6433080434799194 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | Fashion-MNIST | 1 | 20 | 29 | 210120 | 210129 | -0.02427440881729126 | -0.6020650267601013 | -0.10851192474365234 | -0.6385788321495056 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | Fashion-MNIST | 2 | 25 | 36 | 210125 | 210136 | -0.010809898376464844 | -2.002860903739929 | -0.10377371311187744 | -1.7633222341537476 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | KMNIST | 0 | 30 | 43 | 210130 | 210143 | 0.1443321704864502 | -0.6076105833053589 | -0.06921029090881348 | -0.5833848714828491 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | KMNIST | 1 | 35 | 50 | 210135 | 210150 | 0.19005882740020752 | -0.8910610675811768 | -0.046901047229766846 | -0.9340337514877319 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | KMNIST | 2 | 40 | 57 | 210140 | 210157 | 0.0357288122177124 | 0.3937326669692993 | -0.1179666519165039 | 0.3259845972061157 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | MNIST | 0 | 0 | 1 | 210100 | 210101 | 0.37467265129089355 | -0.3195391893386841 | 0.1100953221321106 | -0.3326253890991211 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | MNIST | 1 | 5 | 8 | 210105 | 210108 | 0.3246103525161743 | -0.6302307844161987 | 0.11592137813568115 | -0.6218615770339966 | / | / | 0 |
| MLP-F70-trainloss-terminal-positive-lookahead-floor-source | MNIST | 2 | 10 | 15 | 210110 | 210115 | 0.3803849220275879 | -0.33489835262298584 | 0.1728922724723816 | -0.35099971294403076 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | Fashion-MNIST | 0 | 15 | 23 | 210115 | 210123 | 0.056662023067474365 | -0.8404457569122314 | -0.0028145313262939453 | -0.7640907764434814 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | Fashion-MNIST | 1 | 20 | 30 | 210120 | 210130 | 0.01220715045928955 | -0.16840726137161255 | -0.07156103849411011 | -0.2060145139694214 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | Fashion-MNIST | 2 | 25 | 37 | 210125 | 210137 | -0.0758967399597168 | -2.0358461141586304 | -0.1702643632888794 | -1.8019260168075562 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | KMNIST | 0 | 30 | 44 | 210130 | 210144 | 0.10863590240478516 | -1.007090449333191 | -0.10857844352722168 | -0.9842602014541626 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | KMNIST | 1 | 35 | 51 | 210135 | 210151 | 0.21040070056915283 | -0.5280481576919556 | -0.028932273387908936 | -0.5723124742507935 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | KMNIST | 2 | 40 | 58 | 210140 | 210158 | 0.018138408660888672 | 0.05979633331298828 | -0.13636672496795654 | -0.008685588836669922 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | MNIST | 0 | 0 | 2 | 210100 | 210102 | 0.4198802709579468 | -0.4002925157546997 | 0.1545063853263855 | -0.4149763584136963 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | MNIST | 1 | 5 | 9 | 210105 | 210109 | 0.3379213809967041 | -0.08146798610687256 | 0.12850910425186157 | -0.07393884658813477 | / | / | 0 |
| MLP-F74-trainloss-terminal-optimizer-selector | MNIST | 2 | 10 | 16 | 210110 | 210116 | 0.3428801894187927 | 0.15664684772491455 | 0.13485747575759888 | 0.1400977373123169 | / | / | 0 |

## v22.01 F53 MNIST seed0 raw val/control delta

- artifact: `v22_01_f53_mnist0_raw_val_control_delta.csv`
- purpose: same candidate train_seed case check; separates candidate raw val_loss drift from matched-control attribution drift.

| source | v21_id | job_index | train_seed | val_loss_h800 | val_loss_h3200 | val_loss_h4800 | source_h800 | source_h3200 | source_h4800 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_v22 | MLP-F53-trainloss-terminal-lookahead-floor-source | 0 | 210100 | 0.6567789316177368 | 0.6548259258270264 | 0.654845654964447 | 0.17646336555480957 | 0.3895149230957031 | 0.12493276596069336 |
| old_v22 | CTRL-SGD | 1 | 210101 | 1.9671927690505981 | 1.0443408489227295 | 0.7797784209251404 | -1.1339504718780518 | 0.0 | 0.0 |
| old_v22 | CTRL-AdamW | 2 | 210102 | 0.8332422971725464 | 1.123717188835144 | 1.224871277809143 | 0.0 | -0.07937633991241455 | -0.4450928568840027 |
| old_v22 | CTRL-RandomMatchedNorm | 3 | 210103 | 2.421020984649658 | 2.423219680786133 | 2.4233474731445312 | -1.5877786874771118 | -1.3788788318634033 | -1.6435690522193909 |
| old_v22 | CTRL-NoOpMatchedOverhead | 4 | 210104 | 2.310645580291748 | 2.310645580291748 | 2.310645580291748 | -1.4774032831192017 | -1.2663047313690186 | -1.5308671593666077 |
| fresh_combined | MLP-F53-trainloss-terminal-lookahead-floor-source | 0 | 210100 | 2.011204719543457 | 2.0074899196624756 | 2.007500648498535 | -0.26783883571624756 | -0.7474198341369629 | -0.7605167627334595 |
| fresh_combined | CTRL-AdamW | 4 | 210104 | 1.7433658838272095 | 2.140031576156616 | 2.3061492443084717 | 0.0 | -0.8799614906311035 | -1.059165358543396 |
| fresh_combined | CTRL-RandomMatchedNorm | 5 | 210105 | 2.2620553970336914 | 2.26442813873291 | 2.263904571533203 | -0.5186895132064819 | -1.0043580532073975 | -1.0169206857681274 |
| fresh_combined | CTRL-NoOpMatchedOverhead | 6 | 210106 | 2.3515377044677734 | 2.3515377044677734 | 2.3515377044677734 | -0.608171820640564 | -1.0914676189422607 | -1.1045538187026978 |
| fresh_combined | CTRL-SGD | 3 | 210103 | 1.9310888051986694 | 1.2600700855255127 | 1.2469838857650757 | -0.18772292137145996 | 0.0 | 0.0 |

## v22.01 F53 replay parameter-drift audit

- artifact: `v22_01_f53_replay_parameter_drift_audit.csv`
- conclusion: the previous v22.01 F53/F70/F74 combined replay was not equivalent to old v22 F53 because it used default `train_size=64`, `val_size=48`, `batch_size=32`, `steps=6400`, while the old v22 F53 near-miss run used `train_size=512`, `val_size=256`, `batch_size=64`, `steps=4800`, single candidate plus controls. A matched old-shape replay is required before interpreting F53 as code drift or source-theory failure.

| case | run_label | spec_ids | train_size | val_size | batch_size | steps | shard_count | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_v22_f53 | v2200_f53_terminal_lookahead_floor_full_h4800 | MLP-F53,CTRL-SGD,CTRL-AdamW,CTRL-RandomMatchedNorm,CTRL-NoOpMatchedOverhead | 512 | 256 | 64 | 4800 | 1 | results/v22_0.../official_v22/v22_command_journal.csv |
| v22_01_combined_replay | v2201_cont_f53_f70_f74_replay_full | MLP-F53,MLP-F70,MLP-F74,controls | 64 | 48 | 32 | 6400 | 4 | v22.01 execution log / runner defaults |

## v22.01 continuation: F53 old-shape single-candidate replay

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f53_oldshape_replay_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 1 / 1 / 0
- terminal collapse groups: 1
- best candidate h4800: -0.003963543309105767

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F53-trainloss-terminal-lookahead-floor-source | 9 | 0.048824826876322426 | 0.06716635492112902 | 0.37619424528545803 | 0.36653946505652535 | 0.16515672206878662 | -0.003963543309105767 | 1 | 1 | 0 | 1 | TerminalCollapse |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增/扩展 continuation `F53_OLDSHAPE_REPLAY` 的机制与 runner scope；只读取 fresh matrix 聚合，不生成或篡改实验数据。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F53_OLDSHAPE_REPLAY 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮仍出现 h3200 continuous 但 h4800 失败，说明 DatasetHeterogeneity 的 train-only invariant selector 和 source-conserving optimizer 仍未解决 terminal collapse；下一步应进一步重定义 retained target，而不是继续 terminal floor/carrier 小扫。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F84-F86 old-shape terminal-hold replay

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f84_f86_oldshape_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 3 / 3 / 0
- terminal collapse groups: 3
- best candidate h4800: -0.029913862546284992

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F84-h2400-terminal-hold-source | 9 | 0.030987527635362413 | 0.017280764049953885 | 0.2978205018573337 | 0.30185233222113717 | 0.12514504459169176 | -0.07218090693155925 | 1 | 1 | 0 | 1 | TerminalCollapse |
| MLP-F85-h2800-terminal-hold-source | 9 | 0.020413219928741455 | 0.033282425668504506 | 0.3175995813475715 | 0.3212255769305759 | 0.14451784557766384 | -0.05280810594558716 | 1 | 1 | 0 | 1 | TerminalCollapse |
| MLP-F86-h2400-debt-bailout-source | 9 | 0.03131790293587579 | 0.05182778835296631 | 0.3400600751241048 | 0.34411412477493286 | 0.16740855905744764 | -0.029913862546284992 | 1 | 1 | 0 | 1 | TerminalCollapse |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | 0.0 | -0.0035047531127929688 | -0.3762444787555271 | -0.8993779950671725 | -1.3196280201276143 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -1.537603822019365 | -1.3963266611099243 | -1.1151884065734015 | -1.1113447348276775 | -1.2880509429507785 | -1.4853768944740295 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.5729280511538188 | -1.4318200614717271 | -1.151018136077457 | -1.1466877725389268 | -1.322744369506836 | -1.5203083157539368 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -1.4659851325882807 | -1.1357120540406969 | -0.6329910821384854 | -0.2397618293762207 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增/扩展 continuation `F84_F86_OLDSHAPE` 的机制与 runner scope；只读取 fresh matrix 聚合，不生成或篡改实验数据。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F84_F86_OLDSHAPE 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮仍出现 h3200 continuous 但 h4800 失败，说明 DatasetHeterogeneity 的 train-only invariant selector 和 source-conserving optimizer 仍未解决 terminal collapse；下一步应进一步重定义 retained target，而不是继续 terminal floor/carrier 小扫。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F78-F83 old-shape terminal-route replay

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f78_f83_oldshape_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 1 / 1 / 0
- terminal collapse groups: 1
- best candidate h4800: -0.027273886733584933

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F78-terminal-source-conserving-route | 9 | 0.011324564615885416 | 0.03816947672102186 | 0.3170126610332065 | 0.36793909470240277 | 0.16839882400300768 | -0.029487133026123047 | 1 | 1 | 0 | 1 | TerminalCollapse |
| MLP-F79-trainloss-risk-profile-route | 9 | -0.024459607071346708 | 0.024459169970618352 | 0.30371585819456315 | 0.35450133350160384 | 0.15494787693023682 | -0.0429157018661499 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F80-debt-aware-source-gate | 9 | -0.007588095135158963 | 0.012517597940233018 | 0.2957891556951735 | 0.346835454305013 | 0.1472885873582628 | -0.05063644382688734 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F81-ungated-warm-terminal-source-route | 9 | -0.004243870576222737 | -0.046675556235843234 | 0.23759949207305908 | 0.28216638829973006 | 0.0685323675473531 | -0.12950033611721462 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F82-ungated-warm-risk-profile-route | 9 | -0.0020010670026143393 | 0.06080322133170234 | 0.33958592017491657 | 0.3838532699478997 | 0.17072099447250366 | -0.027273886733584933 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F83-ungated-warm-debt-raw-bailout | 9 | -0.017507447136773005 | -0.06508193413416545 | 0.21814411878585815 | 0.26715819040934247 | 0.05523624685075548 | -0.1427413754993015 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | 0.0 | 0.0 | -0.3027854892942641 | -0.8484888407919142 | -1.269555754131741 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -1.5433161391152277 | -1.4113258586989508 | -1.1377007563908894 | -1.0868528021706476 | -1.2863919801182218 | -1.4843365881178114 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.546318358845181 | -1.414257029692332 | -1.1406490405400593 | -1.089727070596483 | -1.2892885539266798 | -1.4870325459374323 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -1.5047311120563083 | -1.1852239569028218 | -0.6734018060896132 | -0.23076176643371582 | -9.980466630723741e-05 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增/扩展 continuation `F78_F83_OLDSHAPE` 的机制与 runner scope；只读取 fresh matrix 聚合，不生成或篡改实验数据。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F78_F83_OLDSHAPE 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮仍出现 h3200 continuous 但 h4800 失败，说明 DatasetHeterogeneity 的 train-only invariant selector 和 source-conserving optimizer 仍未解决 terminal collapse；下一步应进一步重定义 retained target，而不是继续 terminal floor/carrier 小扫。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 continuation: F75-F77 old-shape invariant/optimizer replay

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f75_f77_oldshape_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 1 / 1 / 0
- terminal collapse groups: 1
- best candidate h4800: -0.0313612355126275

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F75-dataset-invariant-poprisk-slow-source | 9 | -1.525176677438948 | -1.377108719613817 | -1.085836695300208 | -1.0599160724216037 | -1.1919295522901747 | -1.34698881705602 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F76-dataset-invariant-readout-consensus-source | 9 | -1.5505791438950434 | -1.4055386781692505 | -1.1189089020093281 | -1.1033790641360812 | -1.2565706041124132 | -1.4315452112091913 | 0 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| MLP-F77-source-conserving-optimizer-only | 9 | 0.03997104035483466 | 0.07954811387591892 | 0.3590804139773051 | 0.3558522131707933 | 0.1660852829615275 | -0.0313612355126275 | 1 | 1 | 0 | 1 | TerminalCollapse |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | 0.0 | -0.0035047531127929688 | -0.3762444787555271 | -0.8993779950671725 | -1.3196280201276143 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -1.537603822019365 | -1.3963266611099243 | -1.1151884065734015 | -1.1113447348276775 | -1.2880509429507785 | -1.4853768944740295 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.5729280511538188 | -1.4318200614717271 | -1.151018136077457 | -1.1466877725389268 | -1.322744369506836 | -1.5203083157539368 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -1.4659851325882807 | -1.1357120540406969 | -0.6329910821384854 | -0.2397618293762207 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增/扩展 continuation `F75_F77_OLDSHAPE` 的机制与 runner scope；只读取 fresh matrix 聚合，不生成或篡改实验数据。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F75_F77_OLDSHAPE 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮仍出现 h3200 continuous 但 h4800 失败，说明 DatasetHeterogeneity 的 train-only invariant selector 和 source-conserving optimizer 仍未解决 terminal collapse；下一步应进一步重定义 retained target，而不是继续 terminal floor/carrier 小扫。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 old-shape continuation aggregate audit
- route audit: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_oldshape_continuation_route_audit.csv`
- candidate summary: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_oldshape_continuation_candidate_summary.csv`
- dataset localization: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_oldshape_continuation_dataset_localization.csv`
- continuation route rows: 4
- candidate summary rows: 13
- grouped h4800 passing candidate rows: 0

### Route audit

| continuation | decision | early | continuous_h3200 | h4800 | terminal_collapse | best_h4800 |
| --- | --- | --- | --- | --- | --- | --- |
| F53 old-shape replay | ContinuationNoH4800SourceChain | 1 | 1 | 0 | 1 | -0.003963543309105767 |
| F75-F77 old-shape invariant/optimizer replay | ContinuationNoH4800SourceChain | 1 | 1 | 0 | 1 | -0.0313612355126275 |
| F78-F83 old-shape terminal-route replay | ContinuationNoH4800SourceChain | 1 | 1 | 0 | 1 | -0.027273886733584933 |
| F84-F86 old-shape terminal-hold replay | ContinuationNoH4800SourceChain | 3 | 3 | 0 | 3 | -0.029913862546284992 |

### Best candidate h4800 rows

| continuation | v22_id | h800 | h1600 | h3200 | h4800 | early | continuous | h4800_group | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F53 old-shape replay | MLP-F53-trainloss-terminal-lookahead-floor-source | 0.37619424528545803 | 0.36653946505652535 | 0.16515672206878662 | -0.003963543309105767 | 1 | 1 | 0 | TerminalCollapse |
| F78-F83 old-shape terminal-route replay | MLP-F82-ungated-warm-risk-profile-route | 0.33958592017491657 | 0.3838532699478997 | 0.17072099447250366 | -0.027273886733584933 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| F78-F83 old-shape terminal-route replay | MLP-F78-terminal-source-conserving-route | 0.3170126610332065 | 0.36793909470240277 | 0.16839882400300768 | -0.029487133026123047 | 1 | 1 | 0 | TerminalCollapse |
| F84-F86 old-shape terminal-hold replay | MLP-F86-h2400-debt-bailout-source | 0.3400600751241048 | 0.34411412477493286 | 0.16740855905744764 | -0.029913862546284992 | 1 | 1 | 0 | TerminalCollapse |
| F75-F77 old-shape invariant/optimizer replay | MLP-F77-source-conserving-optimizer-only | 0.3590804139773051 | 0.3558522131707933 | 0.1660852829615275 | -0.0313612355126275 | 1 | 1 | 0 | TerminalCollapse |
| F78-F83 old-shape terminal-route replay | MLP-F79-trainloss-risk-profile-route | 0.30371585819456315 | 0.35450133350160384 | 0.15494787693023682 | -0.0429157018661499 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| F78-F83 old-shape terminal-route replay | MLP-F80-debt-aware-source-gate | 0.2957891556951735 | 0.346835454305013 | 0.1472885873582628 | -0.05063644382688734 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| F84-F86 old-shape terminal-hold replay | MLP-F85-h2800-terminal-hold-source | 0.3175995813475715 | 0.3212255769305759 | 0.14451784557766384 | -0.05280810594558716 | 1 | 1 | 0 | TerminalCollapse |

### F53 row localization by dataset

| dataset | rows | mean_h800 | mean_h1600 | mean_h3200 | mean_h4800 | h4800_positive_rows | early_rows | continuous_rows | continuous_h4800_rows | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fashion-MNIST | 3 | 0.41658639907836914 | 0.27457966407140094 | -8.960564931233724e-05 | -0.078616201877594 | 0 | 1 | 0 | 0 | ContinuousRetentionMissing;EarlySourceChainMissing;ContinuousRetentionMissing |
| KMNIST | 3 | 0.5831139882405599 | 0.5535987218221029 | 0.1340299646059672 | -0.06788752476374309 | 0 | 3 | 0 | 0 | ContinuousRetentionMissing |
| MNIST | 3 | 0.12888234853744507 | 0.2714400092760722 | 0.361529807249705 | 0.13461309671401978 | 3 | 1 | 1 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;pass |

### Aggregate insight

- F53/F75-F86 old-shape continuations produced zero grouped h4800 source-chain candidates; v22.01 remains not achieved and `promotion_allowed=0`.
- The best reproduced mechanism is still F53 at h4800=-0.003963543309105767; every plan-continuation old-shape repair after F53 is worse on grouped h4800.
- F53 row localization shows MNIST carries positive h4800 rows, while Fashion-MNIST/KMNIST remain terminal-collapse blockers; the failed invariant/readout/optimizer continuations did not convert that row-level heterogeneity into grouped retained source.
- Current evidence supports stopping terminal floor/carrier/hold micro-sweeps and moving only with a genuinely new train-only retained-target/source-observability theory.

## v22.01 continuation: F87 old-shape terminal raw-then-source-guard repair

- source_dir: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/continuation_f87_oldshape_full`
- decision: `ContinuationNoH4800SourceChain`
- promotion_allowed: 0
- candidate groups early / continuous_h3200 / h4800: 1 / 0 / 0
- terminal collapse groups: 0
- best candidate h4800: -0.03485855129030016

### Continuation candidate evidence

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F87-terminal-raw-then-source-guard | 9 | 0.04120239946577284 | 0.04146580563651191 | 0.34517982933256364 | 0.3349376916885376 | 0.13356016079584757 | -0.03485855129030016 | 1 | 0 | 0 | 0 | ContinuousRetentionMissing |

### Continuation controls

| v22_id | rows | h100 | h400 | h800 | h1600 | h3200 | h4800 | early_source_chain_group | continuous_retention_group | productive_h4800_group | terminal_collapse_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | 0.0 | 0.0 | -0.0001971986558702257 | -0.397111005253262 | -0.979147818353441 | -1.386982838312785 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-NoOpMatchedOverhead | 9 | -1.517841445075141 | -1.3883475462595622 | -1.0872336559825473 | -1.0974532564481099 | -1.298830469449361 | -1.4673935572306316 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-RandomMatchedNorm | 9 | -1.520640320248074 | -1.3912535243564181 | -1.0901426540480719 | -1.1005120873451233 | -1.3025746213065252 | -1.470809433195326 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |
| CTRL-SGD | 9 | -1.4710029760996501 | -1.1490896940231323 | -0.6196606622801887 | -0.24530141221152413 | -0.011503855387369791 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousRetentionMissing |

### 本轮修改记录

- 新增/扩展 continuation `F87_OLDSHAPE` 的机制与 runner scope；只读取 fresh matrix 聚合，不生成或篡改实验数据。
- 扩展 `experiments/run_v22_01_continue_terminal_repair.py`：支持按 continuation tag / ids 读取落盘 fresh matrix，生成 source-chain summary/route/row localization，并追加本节复盘；不生成或篡改实验数据。

### 分析 / Insight / 结论

- F87_OLDSHAPE 没有形成 grouped h4800 retained source chain；因此 v22.01 目标仍未达成，不能 promotion。
- 若本轮连 h3200 continuous 都没有，说明当前 invariant/source-conserving 改法未复现 v22 中 F53/F70 的 h3200 source 保存能力；后续优先分析 row localization，而不是升级 confirmation。
- 本节所有数值来自 `v21_01_source_retention_matrix.csv` 的 fresh continuation run，并用 `dgkan.fu.source_chain` 的 v22.01 source epsilon/control-equivalent 规则重新聚合。

## v22.01 old-shape continuation aggregate audit
- route audit: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_oldshape_continuation_route_audit.csv`
- candidate summary: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_oldshape_continuation_candidate_summary.csv`
- dataset localization: `results/v22_01_early_source_retention_terminal_collapse_kernel_officialization_drat_drbf_4gpu/official_v22_01/v22_01_oldshape_continuation_dataset_localization.csv`
- continuation route rows: 5
- candidate summary rows: 14
- grouped h4800 passing candidate rows: 0

### Route audit

| continuation | decision | early | continuous_h3200 | h4800 | terminal_collapse | best_h4800 |
| --- | --- | --- | --- | --- | --- | --- |
| F53 old-shape replay | ContinuationNoH4800SourceChain | 1 | 1 | 0 | 1 | -0.003963543309105767 |
| F75-F77 old-shape invariant/optimizer replay | ContinuationNoH4800SourceChain | 1 | 1 | 0 | 1 | -0.0313612355126275 |
| F78-F83 old-shape terminal-route replay | ContinuationNoH4800SourceChain | 1 | 1 | 0 | 1 | -0.027273886733584933 |
| F84-F86 old-shape terminal-hold replay | ContinuationNoH4800SourceChain | 3 | 3 | 0 | 3 | -0.029913862546284992 |
| F87 old-shape terminal raw-then-source-guard repair | ContinuationNoH4800SourceChain | 1 | 0 | 0 | 0 | -0.03485855129030016 |

### Best candidate h4800 rows

| continuation | v22_id | h800 | h1600 | h3200 | h4800 | early | continuous | h4800_group | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F53 old-shape replay | MLP-F53-trainloss-terminal-lookahead-floor-source | 0.37619424528545803 | 0.36653946505652535 | 0.16515672206878662 | -0.003963543309105767 | 1 | 1 | 0 | TerminalCollapse |
| F78-F83 old-shape terminal-route replay | MLP-F82-ungated-warm-risk-profile-route | 0.33958592017491657 | 0.3838532699478997 | 0.17072099447250366 | -0.027273886733584933 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| F78-F83 old-shape terminal-route replay | MLP-F78-terminal-source-conserving-route | 0.3170126610332065 | 0.36793909470240277 | 0.16839882400300768 | -0.029487133026123047 | 1 | 1 | 0 | TerminalCollapse |
| F84-F86 old-shape terminal-hold replay | MLP-F86-h2400-debt-bailout-source | 0.3400600751241048 | 0.34411412477493286 | 0.16740855905744764 | -0.029913862546284992 | 1 | 1 | 0 | TerminalCollapse |
| F75-F77 old-shape invariant/optimizer replay | MLP-F77-source-conserving-optimizer-only | 0.3590804139773051 | 0.3558522131707933 | 0.1660852829615275 | -0.0313612355126275 | 1 | 1 | 0 | TerminalCollapse |
| F87 old-shape terminal raw-then-source-guard repair | MLP-F87-terminal-raw-then-source-guard | 0.34517982933256364 | 0.3349376916885376 | 0.13356016079584757 | -0.03485855129030016 | 1 | 0 | 0 | ContinuousRetentionMissing |
| F78-F83 old-shape terminal-route replay | MLP-F79-trainloss-risk-profile-route | 0.30371585819456315 | 0.35450133350160384 | 0.15494787693023682 | -0.0429157018661499 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |
| F78-F83 old-shape terminal-route replay | MLP-F80-debt-aware-source-gate | 0.2957891556951735 | 0.346835454305013 | 0.1472885873582628 | -0.05063644382688734 | 0 | 0 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing |

### F53 row localization by dataset

| dataset | rows | mean_h800 | mean_h1600 | mean_h3200 | mean_h4800 | h4800_positive_rows | early_rows | continuous_rows | continuous_h4800_rows | blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fashion-MNIST | 3 | 0.41658639907836914 | 0.27457966407140094 | -8.960564931233724e-05 | -0.078616201877594 | 0 | 1 | 0 | 0 | ContinuousRetentionMissing;EarlySourceChainMissing;ContinuousRetentionMissing |
| KMNIST | 3 | 0.5831139882405599 | 0.5535987218221029 | 0.1340299646059672 | -0.06788752476374309 | 0 | 3 | 0 | 0 | ContinuousRetentionMissing |
| MNIST | 3 | 0.12888234853744507 | 0.2714400092760722 | 0.361529807249705 | 0.13461309671401978 | 3 | 1 | 1 | 0 | EarlySourceChainMissing;ContinuousRetentionMissing;pass |

### Aggregate insight

- F53/F75-F86 old-shape continuations produced zero grouped h4800 source-chain candidates; v22.01 remains not achieved and `promotion_allowed=0`.
- The best reproduced mechanism is still F53 at h4800=-0.003963543309105767; every plan-continuation old-shape repair after F53 is worse on grouped h4800.
- F53 row localization shows MNIST carries positive h4800 rows, while Fashion-MNIST/KMNIST remain terminal-collapse blockers; the failed invariant/readout/optimizer continuations did not convert that row-level heterogeneity into grouped retained source.
- Current evidence supports stopping terminal floor/carrier/hold micro-sweeps and moving only with a genuinely new train-only retained-target/source-observability theory.

## v22.01 post-F87 closure note

- F87 old-shape terminal raw-then-source-guard also failed: route audit keeps `candidate_h4800=0`, `promotion_allowed=0`, and best F87 h4800=-0.03485855129030016.
- Across F53 and F75-F87 old-shape continuations, grouped h4800 passing candidates remain 0. The best observed grouped h4800 remains the reproduced F53 near-miss at -0.003963543309105767.
- Because F77/F78/F84-F86/F87 all worsened h4800 relative to F53, current evidence no longer supports another terminal floor/carrier/hold/guard tweak. The honest blocker is: train-only retained-target/source-observability theory is still insufficient for Fashion-MNIST/KMNIST terminal retention.
- This is not a promotion and not a budget stop; it is a no-go for the current v22.01 repair family under the old-shape comparable run parameters.

## v22.01 artifact update after old-shape continuations

- v22_01_code_review_packet.zip: size=1929391 sha256=e14906673b699317b2c41f2b791172b385d4b04ca9b05986986df954eb9983c1
- v22_01_results_bundle.zip: size=4016948 sha256=f5122e2fb830c0f04f0a08e77dd1ee38c8deb6e819378c08d6803b83c5e440b9
