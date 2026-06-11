# DG-KAN v21.0 Source Retention + Kernel Officialization 实验结果复盘

生成时间：2026-06-03 22:27:24 +0800

## Route

- route: `R3-WeakMLPSourceOnly-H3200Washout`
- route_detail: S0_5_pass=1, D-CHE_E1_rows=5, D-FOU_E1_rows=4, D-CHE_S1_rows=3, D-FOU_S1_rows=4, MLP_weak_h1600=1, MLP_productive_h3200=0, KAN_weak_h1600=0, KAN_productive_h3200=0, KAN_h4800=0, high_actuation_rows=2, high_actuation_source_rows=0, target_observable_no_go_rows=0, retention_estimator_invalid=0, debt_complete=1
- CodeRoute: S0_5-CodeMetricMechanismPassed
- EfficiencyRoute: D-CHE_E1=5;D-FOU_E1=4;D-CHE_S1=3;D-FOU_S1=4
- FunctionalRoute: MLP_weak=1;MLP_h3200=0;KAN_weak=0;KAN_h3200=0;KAN_h4800=0
- promotion_allowed: 0

## 关键结果

- S0.5 preflight pass: 1
- D-CHE/D-FOU E1 exploration rows: 5 / 4
- D-CHE/D-FOU S1 official-like rows: 3 / 4
- MLP weak h1600 / productive h3200 candidates: 1 / 0
- KAN weak h1600 / productive h3200 / h4800 candidates: 0 / 0 / 0
- function-space high ActuationR2 rows / source-success rows: 2 / 0
- debt complete: 1

## Efficiency Evidence

| carrier | rows | E1 exploration | S1 official-like | best forward | best step |
|---|---:|---:|---:|---:|---:|
| D-CHE | 12 | 5 | 3 | 0.8396598403474669 | 0.43267989422299125 |
| D-FOU | 12 | 4 | 4 | 0.9039954151783288 | 0.5005505084509732 |
| D-RAT | 4 | 0 | 0 | 8.189205551205196 | 1.7339874451875055 |
| D-RBF | 4 | 0 | 0 | 8.477386493572494 | 1.8770586183656492 |
| D-WAV | 4 | 0 | 0 | 11.605863117463699 | 2.327962941503731 |
| LQ | 4 | 0 | 0 | 6.670286000210682 | 0.9390656882827326 |

## MLP Source Dynamics

| v21_id | rows | h800 | h1600 | h3200 | h4800 | weak_h1600 | productive_h3200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTRL-AdamW | 9 | -0.4654239945941501 | -0.9000058571497599 | -1.3858900798691645 | -1.5615805321269565 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | 9 | -0.5467097759246826 | -0.7240315543280708 | -0.9123760792944167 | -0.8772483600510491 | 0 | 0 |
| CTRL-RandomMatchedNorm | 9 | -0.6011857721540663 | -0.7784774833255343 | -0.9679356084929572 | -0.9322867327266269 | 0 | 0 |
| CTRL-SGD | 9 | -0.13343090481228298 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| MLP-F1-M2-strong-source | 9 | 0.1373782687717014 | -0.2717866500218709 | -0.5421817236476474 | -0.4903893801901076 | 0 | 0 |
| MLP-F11-dual-timescale-retention-warm1200 | 9 | 0.20106730196211073 | -0.11220832665761311 | -0.3005838990211487 | -0.26549150546391803 | 0 | 0 |
| MLP-F2-M15-weak-stable | 9 | -0.1437634759479099 | -0.007729490598042806 | 0.005254639519585503 | 0.030545175075531006 | 0 | 0 |
| MLP-F3-F9-hold | 9 | 0.08885670370525783 | -0.0884650746981303 | -0.27680959966447616 | -0.24168188042110866 | 0 | 0 |
| MLP-F4-F10-cycle-hold | 9 | 0.31621605157852173 | 0.1388942731751336 | -0.24906698862711588 | -0.28254638777838814 | 1 | 0 |
| MLP-F5-M2-plus-M15-anchor | 9 | 0.11730257007810804 | -0.30320071511798435 | -0.6175312068727281 | -0.5917447805404663 | 0 | 0 |
| MLP-F6-M2-source-with-slow-anchor | 9 | 0.2727375162972344 | -0.12892581356896293 | -0.4151756829685635 | -0.3809780346022712 | 0 | 0 |
| MLP-F7-M2-source-with-matrix-block-retention | 9 | 0.27446021636327106 | -0.15635029474894205 | -0.45658133427302044 | -0.4289841453234355 | 0 | 0 |
| MLP-F9-cycle1200-to-M15-anchor | 9 | 0.19538146919674343 | -0.11649780803256565 | -0.30486123429404366 | -0.2697662313779195 | 0 | 0 |

## KAN Source Writer

| carrier | variant | v21_id | h800 | h1600 | h3200 | h4800 | productive_h3200 |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-AdamW | -0.06594128873613146 | -0.07928047047721015 | -0.10328533914354113 | -0.14634346961975098 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-NoOpMatchedOverhead | -1.084653902053833 | -0.9567378242810567 | -0.7935974213812086 | -0.6974363671408759 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-RandomMatchedNorm | -1.0903623474968804 | -0.9624610092904833 | -0.7993394414583842 | -0.7031937413745456 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-SGD | -1.0972653865814208 | -0.9867439839575026 | -0.8509721000989278 | -0.7693887525134616 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T1-loss-cotangent-target | -0.17014653152889675 | -0.056964748435550265 | 0.026712576548258465 | 0.06482330958048503 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T5-random-matched-target | -0.6287222041024102 | -0.11030812395943536 | 0.1290053261650933 | 0.19790532191594443 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T6-sign-flipped-target | -1.1066131856706407 | -1.0003897547721863 | -0.8729673160447015 | -0.7874626186158922 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T7-corrupted-label-target | -0.7092151112026639 | -0.3172903127140469 | -0.09495444430245294 | -0.033858795960744224 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-alt100 | -0.22847294145160252 | 0.013041185008154975 | 0.1299898624420166 | 0.17205823792351616 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-smallstep-alt100 | -0.4973144001430935 | 0.022982332441541884 | 0.24416455957624647 | 0.2911813192897373 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-smallstep-alt50 | -0.18547199169794717 | 0.0823837055100335 | 0.17639156182607016 | 0.2091420425309075 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F7-KSW2-warm400-smallstep-alt50 | -0.5733198059929742 | 0.048869676060146756 | 0.18189224269655016 | 0.2072928614086575 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F7-KSW2-warm800-smallstep-alt50 | -1.1010113557179768 | -0.06217914157443576 | 0.19809132152133518 | 0.22228250900904337 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F8-KSW2-earlyboost-alt25 | -0.1999136341942681 | -0.15231910016801622 | -0.10880810022354126 | -0.08013690842522515 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F8-KSW2-earlyboost-highstep-alt50 | -0.24004947476916844 | -0.24222366677390206 | -0.19286837180455527 | -0.1641250318951077 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW1-basis-estimate-readout-commit | -1.0948187245262995 | -0.9849815832244025 | -0.8496781918737624 | -0.7680488559934828 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW2-lowdegree-lowfreq-source-bank | -0.13043776485655043 | -0.044620898034837514 | 0.03229019376966688 | 0.06687237156762017 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW3-dualbank-source-reservoir | -1.165903992123074 | -1.0253980623351202 | -0.7390022608968947 | -0.45373720592922634 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | -0.03743810455004374 | -0.04624353183640374 | -0.06479539142714606 | -0.09208467271592882 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | -0.9539676109949747 | -0.8816002408663431 | -0.777377188205719 | -0.7031143373913236 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | -0.9526841772927178 | -0.8803169661098056 | -0.7761004434691535 | -0.701839976840549 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | -0.9527630276150174 | -0.8788762821091546 | -0.7720220817459954 | -0.694565667046441 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T1-loss-cotangent-target | -0.16823722256554496 | -0.13407624430126613 | -0.09074081314934625 | -0.053048173586527504 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T5-random-matched-target | -0.23672142955991957 | -0.011898650063408745 | 0.11602528889973958 | 0.16437860329945883 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T6-sign-flipped-target | -0.9558619525697496 | -0.8888364831606547 | -0.7925142778290643 | -0.7174178096983168 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T7-corrupted-label-target | -0.4212048053741455 | -0.16376509269078574 | -0.016067193614112005 | 0.030873219172159832 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW1-basis-estimate-readout-commit | -0.9514029688305325 | -0.8779414958424039 | -0.7718896137343513 | -0.6954575777053833 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW2-lowdegree-lowfreq-source-bank | -0.19072292910681832 | -0.1549445324473911 | -0.11295051707161798 | -0.0759023560418023 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW3-dualbank-source-reservoir | -0.9350418647130331 | -0.7827491561571757 | -0.4449089831776089 | -0.16835025946299234 | 0 |

## Function-Space Target Reset

| carrier | v21_id | ActuationR2 max | source_h1600 | source_h3200 | decision |
|---|---|---:|---:|---:|---|
| D-CHE | KSW2-lowdegree-lowfreq-source-bank | 0.9460849761962891 | -0.010244336393144395 | 0.06749698850843641 | high_actuation_but_source_not_retained |
| D-FOU | KSW2-lowdegree-lowfreq-source-bank | 0.9952160120010376 | -0.10503021213743421 | -0.05527836084365845 | high_actuation_but_source_not_retained |

## Function-Space Target Contrast

| carrier | variant | loss B2 gain | random B2 gain | sign-flip B2 gain | corrupt B2 gain | no_go |
|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R4-k3-gradbuf-triton | 0.5593363046646118 | -0.016290982564290363 | -0.6067532168494331 | -0.08347996075948079 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | 0.7169851064682007 | -0.014322678248087565 | -0.787827279832628 | -0.13488774829440647 | 0 |

## Function-Space Target Source Writer

| carrier | variant | v21_id | h800 | h1600 | h3200 | h4800 | productive_h3200 |
|---|---|---|---:|---:|---:|---:|---:|
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T1-loss-cotangent-target | -0.17014653152889675 | -0.056964748435550265 | 0.026712576548258465 | 0.06482330958048503 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T5-random-matched-target | -0.6287222041024102 | -0.11030812395943536 | 0.1290053261650933 | 0.19790532191594443 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T6-sign-flipped-target | -1.1066131856706407 | -1.0003897547721863 | -0.8729673160447015 | -0.7874626186158922 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T7-corrupted-label-target | -0.7092151112026639 | -0.3172903127140469 | -0.09495444430245294 | -0.033858795960744224 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T1-loss-cotangent-target | -0.16823722256554496 | -0.13407624430126613 | -0.09074081314934625 | -0.053048173586527504 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T5-random-matched-target | -0.23672142955991957 | -0.011898650063408745 | 0.11602528889973958 | 0.16437860329945883 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T6-sign-flipped-target | -0.9558619525697496 | -0.8888364831606547 | -0.7925142778290643 | -0.7174178096983168 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T7-corrupted-label-target | -0.4212048053741455 | -0.16376509269078574 | -0.016067193614112005 | 0.030873219172159832 | 0 |

## Source-Retention Estimator Audit

| predictor | finite rows | Spearman h3200 | AUC retained/washout | LDO min | LSO min | inc AUC | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0_early_source_readback | 558 | 0.6899782275271426 | 0.5464216634429401 | 0.9447513812154696 | 0.9372253609541745 | 0.19157055630936226 | 1 |
| P1_parameter_snr_baseline | 0 |  |  |  |  |  | 0 |
| P2_output_transfer_b2_gain | 144 | 0.33092195161160676 |  | 0.4444444444444444 | 0.5079365079365079 | -0.22593627896063084 | 0 |
| P3_split_consensus_signal_minus_corrupt | 45 | 0.12015810276679842 | 0.3333333333333333 | 0.42857142857142855 | 0.42857142857142855 | -0.1654821132433073 | 0 |
| P4_linec_channel_quality | 558 | 0.056808425736902556 | 0.4796905222437137 | 0.5115740740740741 | 0.4558011049723757 | -0.20658073270013566 | 0 |
| P5_matrix_block_norm | 0 |  |  |  |  |  | 0 |
| P6_slow_state_stability | 81 | 0.3512646793134598 | 0.6481481481481481 | 0.5769230769230769 | 0.62 | -0.04026165468951537 | 0 |
| P7_random_or_control_indicator | 558 | 0.38547191872601844 | 0.5 | 0.7651933701657458 | 0.7651933701657458 | 0.0 | 0 |
| P8_train_source_readback | 558 | 0.5061692141984856 | 0.3114119922630561 | 0.6648148148148149 | 0.7001404494382022 | -0.03409090909090906 | 0 |
| P9_train_loss_drop | 558 | -0.5698950128623719 | 0.24129593810444874 | 0.2935185185185185 | 0.2991573033707865 | -0.39077340569877883 | 0 |
| P10_train_source_x_linec_quality | 558 | 0.027886637949740008 | 0.4825918762088975 | 0.4218181818181818 | 0.43628374136848713 | -0.2950305291723202 | 0 |

F5 decision: OnlyEarlySourceReadbackPredictsRetention_NoTrainOnlyDirectionSelector; rows=558; passing_predictors=1; passing_train_only_predictors=0; direction_selector_allowed=0.

### F5 Early-Source Stratification

| group | carrier | dataset | v21_id | rows | h800+ | h3200 retained | washout | late rebound | h800 mean | h3200 mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| carrier | D-CHE |  |  | 306 | 7 | 7 | 0 | 45 | -0.7271331100682028 | -0.4257562515003229 |
| carrier | D-FOU |  |  | 135 | 4 | 4 | 0 | 13 | -0.64352660090835 | -0.4595757546248259 |
| carrier | MLP |  |  | 117 | 58 | 11 | 47 | 4 | -0.02208567888308794 | -0.49413375365428436 |
| carrier_dataset | D-CHE | Fashion-MNIST |  | 102 | 4 | 4 | 0 | 8 | -0.8366040365368712 | -0.51314396016738 |
| carrier_dataset | D-CHE | KMNIST |  | 102 | 1 | 1 | 0 | 20 | -0.6379862404337117 | -0.35542349488127467 |
| carrier_dataset | D-CHE | MNIST |  | 102 | 2 | 2 | 0 | 17 | -0.7068090532340255 | -0.4087012994523142 |
| carrier_dataset | D-FOU | Fashion-MNIST |  | 45 | 1 | 1 | 0 | 5 | -0.7508738623725043 | -0.5078905344009399 |
| carrier_dataset | D-FOU | KMNIST |  | 45 | 2 | 2 | 0 | 7 | -0.4436364968617757 | -0.24445658259921604 |
| carrier_dataset | D-FOU | MNIST |  | 45 | 1 | 1 | 0 | 1 | -0.7360694434907701 | -0.6263801468743219 |
| carrier_dataset | MLP | Fashion-MNIST |  | 39 | 13 | 1 | 12 | 2 | -0.22411985122240508 | -0.8424308162469131 |
| carrier_dataset | MLP | KMNIST |  | 39 | 21 | 2 | 19 | 1 | 0.04787350006592579 | -0.36418883617107684 |
| carrier_dataset | MLP | MNIST |  | 39 | 24 | 8 | 16 | 1 | 0.10998931450721545 | -0.27578160854486317 |
| carrier_v21_id | D-CHE |  | CTRL-AdamW | 45 | 0 | 0 | 0 | 0 | -0.06594128873613146 | -0.10328533914354113 |
| carrier_v21_id | D-CHE |  | CTRL-NoOpMatchedOverhead | 45 | 0 | 0 | 0 | 0 | -1.084653902053833 | -0.7935974213812086 |
| carrier_v21_id | D-CHE |  | CTRL-RandomMatchedNorm | 45 | 0 | 0 | 0 | 0 | -1.0903623474968804 | -0.7993394414583842 |
| carrier_v21_id | D-CHE |  | CTRL-SGD | 45 | 0 | 0 | 0 | 0 | -1.0972653865814208 | -0.8509721000989278 |
| carrier_v21_id | D-CHE |  | F3-T1-loss-cotangent-target | 9 | 2 | 2 | 0 | 1 | -0.17014653152889675 | 0.026712576548258465 |
| carrier_v21_id | D-CHE |  | F3-T5-random-matched-target | 9 | 0 | 0 | 0 | 5 | -0.6287222041024102 | 0.1290053261650933 |
| carrier_v21_id | D-CHE |  | F3-T6-sign-flipped-target | 9 | 0 | 0 | 0 | 0 | -1.1066131856706407 | -0.8729673160447015 |
| carrier_v21_id | D-CHE |  | F3-T7-corrupted-label-target | 9 | 0 | 0 | 0 | 1 | -0.7092151112026639 | -0.09495444430245294 |
| carrier_v21_id | D-CHE |  | F6-KSW2-density-alt100 | 9 | 1 | 1 | 0 | 4 | -0.22847294145160252 | 0.1299898624420166 |
| carrier_v21_id | D-CHE |  | F6-KSW2-density-smallstep-alt100 | 9 | 0 | 0 | 0 | 8 | -0.4973144001430935 | 0.24416455957624647 |
| carrier_v21_id | D-CHE |  | F6-KSW2-density-smallstep-alt50 | 9 | 1 | 1 | 0 | 6 | -0.18547199169794717 | 0.17639156182607016 |
| carrier_v21_id | D-CHE |  | F7-KSW2-warm400-smallstep-alt50 | 9 | 0 | 0 | 0 | 8 | -0.5733198059929742 | 0.18189224269655016 |
| carrier_v21_id | D-CHE |  | F7-KSW2-warm800-smallstep-alt50 | 9 | 0 | 0 | 0 | 7 | -1.1010113557179768 | 0.19809132152133518 |
| carrier_v21_id | D-CHE |  | F8-KSW2-earlyboost-alt25 | 9 | 1 | 1 | 0 | 0 | -0.1999136341942681 | -0.10880810022354126 |
| carrier_v21_id | D-CHE |  | F8-KSW2-earlyboost-highstep-alt50 | 9 | 0 | 0 | 0 | 1 | -0.24004947476916844 | -0.19286837180455527 |
| carrier_v21_id | D-CHE |  | KSW1-basis-estimate-readout-commit | 9 | 0 | 0 | 0 | 0 | -1.0948187245262995 | -0.8496781918737624 |
| carrier_v21_id | D-CHE |  | KSW2-lowdegree-lowfreq-source-bank | 9 | 2 | 2 | 0 | 3 | -0.13043776485655043 | 0.03229019376966688 |
| carrier_v21_id | D-CHE |  | KSW3-dualbank-source-reservoir | 9 | 0 | 0 | 0 | 1 | -1.165903992123074 | -0.7390022608968947 |
| carrier_v21_id | D-FOU |  | CTRL-AdamW | 18 | 0 | 0 | 0 | 0 | -0.03743810455004374 | -0.06479539142714606 |
| carrier_v21_id | D-FOU |  | CTRL-NoOpMatchedOverhead | 18 | 0 | 0 | 0 | 0 | -0.9539676109949747 | -0.777377188205719 |
| carrier_v21_id | D-FOU |  | CTRL-RandomMatchedNorm | 18 | 0 | 0 | 0 | 0 | -0.9526841772927178 | -0.7761004434691535 |
| carrier_v21_id | D-FOU |  | CTRL-SGD | 18 | 0 | 0 | 0 | 0 | -0.9527630276150174 | -0.7720220817459954 |
| carrier_v21_id | D-FOU |  | F3-T1-loss-cotangent-target | 9 | 1 | 1 | 0 | 2 | -0.16823722256554496 | -0.09074081314934625 |
| carrier_v21_id | D-FOU |  | F3-T5-random-matched-target | 9 | 1 | 1 | 0 | 5 | -0.23672142955991957 | 0.11602528889973958 |
| carrier_v21_id | D-FOU |  | F3-T6-sign-flipped-target | 9 | 0 | 0 | 0 | 0 | -0.9558619525697496 | -0.7925142778290643 |
| carrier_v21_id | D-FOU |  | F3-T7-corrupted-label-target | 9 | 1 | 1 | 0 | 3 | -0.4212048053741455 | -0.016067193614112005 |
| carrier_v21_id | D-FOU |  | KSW1-basis-estimate-readout-commit | 9 | 0 | 0 | 0 | 0 | -0.9514029688305325 | -0.7718896137343513 |
| carrier_v21_id | D-FOU |  | KSW2-lowdegree-lowfreq-source-bank | 9 | 1 | 1 | 0 | 1 | -0.19072292910681832 | -0.11295051707161798 |

## Optimizer / Bridge Diagnostics

| diagnostic | v21_id | rows | key mean | decision |
|---|---|---:|---:|---|
| AdamW overwrite | MLP-F1-M2-strong-source | 9 | 1213.9837646484375 | NoAdamWOverwriteBlockerAtFirstStep |
| AdamW overwrite | MLP-F4-F10-cycle-hold | 9 | 1249.192159016927 | NoAdamWOverwriteBlockerAtFirstStep |
| AdamW overwrite | MLP-F5-M2-plus-M15-anchor | 9 | 1304.0552775065105 | NoAdamWOverwriteBlockerAtFirstStep |
| AdamW overwrite | MLP-F6-M2-source-with-slow-anchor | 9 | 1268.8939412434895 | NoAdamWOverwriteBlockerAtFirstStep |
| AdamW overwrite | MLP-F7-M2-source-with-matrix-block-retention | 9 | 1292.2105577256943 | NoAdamWOverwriteBlockerAtFirstStep |
| AdamW overwrite | MLP-F8-cycle-hold-to-M15-anchor | 9 | 1269.0923665364583 | NoAdamWOverwriteBlockerAtFirstStep |
| AdamW overwrite | MLP-F9-cycle1200-to-M15-anchor | 9 | 1182.1486952039932 | NoAdamWOverwriteBlockerAtFirstStep |
| bridge full | MLP-F11-dual-timescale-retention-warm1200 | 9 | -0.3005838990211487 | retained_h3200=0 |
| bridge full | MLP-F9-cycle1200-to-M15-anchor | 9 | -0.30486123429404366 | retained_h3200=0 |

## Failure Taxonomy

| failure_class | severity | evidence |
|---|---|---|
| MLPSourceH3200WashoutOrNoRetention | functional | v21_mlp_source_dynamics.csv |
| KANSourceWriterNoProductiveRetainedCandidate | functional | v21_kan_source_writer_matrix.csv |

## 修改记录

- 新增 v21 runner 体系：S0.5 truth gate、S1 efficiency officialization、MLP source dynamics、KAN source-channel writer、function-space target reset 和 finalizer。
- 新增 v21 source-state/kernel/profiling shim：`source_state.py`、`poprisk_source.py`、`efficiency_v21.py`、`che_official.py`、`fou_official.py`。
- 新增 M44/M45/M46：分别尝试 M2+LineC slow anchor、M2 slow anchor、M2+matrix-block retention；方向只来自 train stream，trace 记录 source-state gate/gain/cos 诊断。
- 新增 M47：先用 momentum cycle-hold 保早期 source，再在 h1600 后用 M15-style train-split LineC anchor 写入 slow state；LineC 只做 gate/readback，不作为方向源。
- 新增 M48：按计划 3.2 的 AdEMAMix/long-memory 启发维护 short/long 双时间尺度 source state，只提交 short/long 符号一致的 train-stream source 分量。
- 新增 `experiments/run_v21_optimizer_overwrite_diagnostic.py`：按 v21 spec 输出 FU-vs-AdamW first-step cosine/projection，闭合计划 13.3 的 AdamW overwrite 诊断。
- 新增 `experiments/run_v21_function_space_target_contrast.py`：按计划 13.5 对 high-ActuationR2/source-fail 路线做 random/sign-flip/corrupt/B1-only/lower-rank target contrast。
- 新增 M49/M50/M51/M52/M53 与 `experiments/run_v21_function_space_target_source.py`：把 F3 loss-cotangent target 和 matched random/sign-flip/corrupt/low-rank controls 接入 long-horizon source writer。
- 新增 `experiments/run_v21_source_retention_estimator_audit.py`：按计划 F5 离线审计 PopRisk/SNR、output transfer、split-consensus、LineC、matrix-block、slow-state 等 predictor 是否能预测 h3200/h4800 retention。
- Finalizer 只从落盘 CSV/trace 汇总 route；缺失或失败证据写 blocker，不翻 promotion。

## 分析 / Insight / 结论

- v21 把 v20 的 h1600 可保留但 h3200/h4800 washout 作为核心 blocker，而不是继续只看单 seed smoke。
- S2 weak FU 只说明机制在 h1600 还有相对 controls 的 source；S3/S5 仍要求 h3200/h4800、controls、debt 和 efficiency 同时过。
- Function-space high ActuationR2 仍只是局部 displacement 可实现，不能替代 source retention。
- M47/M48 的 smoke 在 MNIST seed0 上出现 h800/h1600/h3200 连续正值，但 full 9-row grouped mean 在 h1600/h3200 转负；这说明 single-seed bridge signal 不能跨 dataset/seed 稳定留存。
- v21 AdamW overwrite first-step 诊断没有显示负投影阻碍；当前 washout 更像 source target / dataset-seed heterogeneity 问题，而不是 AdamW 直接反向覆盖。
- Function-space target contrast 如果显示 loss target 不能稳定强过 random/sign-flip/corrupt target，应按 TargetObservableNoGo 处理，不继续 actuation solver 小修。
- 如果 loss target contrast 可观测但 target source writer 仍不 retained，则 blocker 从 actuation/observable 推进为 target-to-retention dynamics，而不是继续只调 solver cap。
- F5 source-retention estimator audit 若无法通过跨 dataset/seed AUC 与 Spearman 门，只能作为 failure taxonomy，不能作为下一轮 direction selector。
- `promotion_allowed=1` 只有 KAN retained source、efficiency officialization、debt recovery、controls attribution 都真实通过才允许。


## 2026-06-03 22:35 F5-F8 Source-Retention / KSW2 Repair 追加复盘

### 是否达成 v21 目标

没有达成。

- route: `R3-WeakMLPSourceOnly-H3200Washout`
- promotion_allowed: 0
- F5 estimator rows: 558
- F5 passing predictors / train-only passing predictors: 1 / 0
- F6/F7/F8 KSW2 repair full rows: 63 / 54 / 54
- F8 blocked rows: 0/54
- KAN productive h3200 candidates: 0
- required artifact missing count: 0

### 本轮修改记录

- 新增 `experiments/run_v21_source_retention_estimator_audit.py`：离线读取既有 matrix/trace，审计 h800 早期 readback、PopRisk/SNR、B2 transfer、split-consensus、LineC、slow-state stability、train-source 等 predictor 是否能跨 dataset/seed 预测 h3200/h4800 retention。
- 更新 `experiments/run_v21_common.py`：把 F5 audit 脚本加入 packet source tree；新增 F6 KSW2 density、F7 KSW2 warmup、F8 KSW2 early-boost specs。
- 更新 `experiments/run_v21_s05_truth_gate.py`：把 F5 audit runner 纳入 S0.5 import/compile 检查。
- 更新 `experiments/run_v21_finalize.py`：把 F5 rows/summary/decision/stratification 加入 required artifact 与复盘输出；route 增加 `source_retention_estimator_invalid` 字段。

### F5 Source-Retention Estimator Audit

| item | value |
|---|---:|
| estimator rows | 558 |
| retained h1600 labels | 49 |
| retained h3200 labels | 22 |
| retained h4800 labels | 22 |
| washout h3200 labels | 47 |
| late rebound h3200 labels | 62 |
| passing predictors | 1 |
| passing train-only predictors | 0 |

| predictor | finite rows | Spearman h3200 | AUC retained h3200 | LDO min | LSO min | incremental AUC | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 early source readback | 558 | 0.6899782275271426 | 0.9602272727272727 | 0.9447513812154696 | 0.9372253609541745 | 0.19157055630936226 | 1 |
| P2 output transfer B2 gain | 144 | 0.33092195161160676 | 0.5427204374572796 | 0.4444444444444444 | 0.5079365079365079 | -0.22593627896063084 | 0 |
| P6 slow-state stability | 81 | 0.3512646793134598 | 0.7283950617283951 | 0.5769230769230769 | 0.62 | -0.04026165468951537 | 0 |
| P8 train source readback | 558 | 0.5061692141984856 | 0.7345658073270014 | 0.6648148148148149 | 0.7001404494382022 | -0.03409090909090906 | 0 |
| P10 train source x LineC quality | 558 | 0.027886637949740008 | 0.47362618724559025 | 0.4218181818181818 | 0.43628374136848713 | -0.2950305291723202 | 0 |

F5 的判定是 `OnlyEarlySourceReadbackPredictsRetention_NoTrainOnlyDirectionSelector`。含义是：如果 h800 已经真实测到 source 为正，它确实很强地预测 h3200/h4800 retention；但这不是可提前使用的 train-only direction selector。P8 train-source readback 有相关性，但 incremental AUC 为负，不能替代 validation/control source readback。这个结果不能 promotion，也不能作为新的方向选择器直接进入 S4/S5。

### F5 Stratification Insight

| group | rows | h800 positive | retained h3200 | washout h3200 | late rebound h3200 |
|---|---:|---:|---:|---:|---:|
| D-CHE | 306 | 7 | 7 | 0 | 45 |
| D-FOU | 135 | 4 | 4 | 0 | 13 |
| MLP | 117 | 58 | 11 | 47 | 4 |

关键 insight：KAN carrier 上一旦 h800 positive，当前数据里几乎都会留到 h3200；真正的问题是 h800 positive density 太低。MLP 则相反，h800 positive 很多，但 47 rows 到 h3200 washout。这就是为什么本轮不再继续 MLP warmup/hold sweep，而转向 KSW2 的 h800 early-source-density 修复。

### F6-F8 KSW2 Repair Evidence

| v21_id | rows | h800 | h800 pass | h1600 | h1600 pass | h3200 | h3200 pass | h4800 | h4800 pass | retained | late |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KSW2-lowdegree-lowfreq-source-bank | 9 | -0.13043776485655043 | 2 | -0.044620898034837514 | 4 | 0.03229019376966688 | 5 | 0.06687237156762017 | 5 | 0 | 1 |
| F6-KSW2-density-smallstep-alt50 | 9 | -0.18547199169794717 | 1 | 0.0823837055100335 | 5 | 0.17639156182607016 | 7 | 0.2091420425309075 | 7 | 0 | 1 |
| F6-KSW2-density-alt100 | 9 | -0.22847294145160252 | 1 | 0.013041185008154975 | 4 | 0.1299898624420166 | 5 | 0.17205823792351616 | 5 | 0 | 1 |
| F6-KSW2-density-smallstep-alt100 | 9 | -0.4973144001430935 | 0 | 0.022982332441541884 | 4 | 0.24416455957624647 | 8 | 0.2911813192897373 | 8 | 0 | 1 |
| F7-KSW2-warm400-smallstep-alt50 | 9 | -0.5733198059929742 | 0 | 0.048869676060146756 | 4 | 0.18189224269655016 | 8 | 0.2072928614086575 | 8 | 0 | 1 |
| F7-KSW2-warm800-smallstep-alt50 | 9 | -1.1010113557179768 | 0 | -0.06217914157443576 | 3 | 0.19809132152133518 | 7 | 0.22228250900904337 | 7 | 0 | 1 |
| F8-KSW2-earlyboost-alt25 | 9 | -0.1999136341942681 | 1 | -0.15231910016801622 | 1 | -0.10880810022354126 | 1 | -0.08013690842522515 | 2 | 0 | 0 |
| F8-KSW2-earlyboost-highstep-alt50 | 9 | -0.24004947476916844 | 0 | -0.24222366677390206 | 1 | -0.19286837180455527 | 1 | -0.1641250318951077 | 1 | 0 | 0 |

解释：

- F6 证明 KSW2 的温和密度修复能显著增强 h1600/h3200/h4800 late-positive，但 h800 grouped mean 仍为负，因此不能写 retained source。
- F7 warmup 没有修复 early phase；h800 反而更负，同时 late-positive 仍存在。
- F8 early-boost 是对 F5 insight 的直接修复尝试：更频繁 commit 或更大步长，目标是把 h800 提前打正。结果失败，且 h3200/h4800 late-positive 也被破坏。
- 因此 KSW2 当前 blocker 不是简单 “commit 太少 / 步子太小 / warmup 不够”；更像是 early target 与 late target 存在相位冲突。温和写入能产生 delayed recovery，硬推早期会损坏后续 source。

### 当前边界

v21 仍未达成目标。basis/kernel officialization 不是当前主 blocker；functional source retention 才是。F5 给出一个强但不可行动的事实：真实 h800 source readback 能预测 retention；但没有 train-only predictor 能稳定提前选择这个方向。F6-F8 进一步说明，KSW2 可以制造 late-positive source，但不能形成 h800/h1600/h3200 连续 retained chain。

当前不能写 breakthrough，不能把 `promotion_allowed` 改为 1。继续推进需要新的 train-only early-source observability 或 target-to-retention writer 理论；继续同族 KSW2 alt/scale/warmup sweep 已经没有足够审计价值。
