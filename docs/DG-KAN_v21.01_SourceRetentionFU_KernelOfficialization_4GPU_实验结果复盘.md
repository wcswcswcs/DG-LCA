# DG-KAN v21.01 SourceRetentionFU KernelOfficialization 4GPU 实验结果复盘

生成时间：2026-06-04 02:51:25 +0800

## Route

- route: `R2-LateReboundNoContinuousRetention`
- route_detail: S0_6_pass=1, D-CHE_E1_rows=8, D-FOU_E1_rows=3, D-CHE_S1_rows=6, D-FOU_S1_rows=2, MLP_h3200=0, MLP_h4800=0, KAN_h3200=0, KAN_h4800=0, independent_confirmation_pass=0, late_rebound_groups=26
- CodeRoute: S0_6-CodeMetricMechanismPassed
- EfficiencyRoute: D-CHE_E1=8;D-FOU_E1=3;D-CHE_S1=6;D-FOU_S1=2
- FunctionalRoute: MLP_h3200=0;KAN_h3200=0;KAN_h4800=0;independent=0;late_rebound=26
- promotion_allowed: 0

## 关键结果

- S0.6 preflight pass: 1
- D-CHE / D-FOU E1 rows: 8 / 3
- D-CHE / D-FOU S1 rows: 6 / 2
- source grouped rows: 61
- efficiency rows: 33
- efficiency fallback rows: 15
- independent confirmation: independent_confirmation_failed
- required artifact missing count: 0

## S0.6 Truth Gate

| check | pass | value | blocker |
|---|---:|---|---|
| import_closure | 1 | 1 |  |
| linec | 1 | 9/9;8/8 |  |
| retention_debt_route | 1 | 1 |  |
| update_semantics | 1 | 1 |  |
| mechanism_contracts | 1 | 5/5 |  |
| efficiency_profiler | 1 | 1 |  |
| kernel_gradcheck | 1 | exploration=6/6;consistency=2/2 |  |

## Efficiency Evidence

| carrier | rows | official-like pass rows | best forward | best step |
|---|---:|---:|---:|---:|
| D-CHE | 9 | 0 | 1.4555440083962712 | 0.48730425845164144 |
| D-FOU | 9 | 0 | 1.399751037547353 | 0.6023496667060054 |

### Efficiency Fallback Repeat/Warmup Evidence

| carrier | rows | E1 pass | S1 pass | best forward | best step |
|---|---:|---:|---:|---:|---:|
| D-CHE | 12 | 8 | 6 | 0.9440247951264752 | 0.4269233776176479 |
| D-FOU | 3 | 3 | 2 | 1.111236890399642 | 0.4717201382550595 |

## Source-Retention Evidence

| carrier | variant | v21_id | rows | h800 | h1600 | h3200 | h4800 | h6400 | h3200 cand | h4800 cand | late rebound |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-AdamW | 18 | 0.0 | 0.0 | -0.010691775215996636 | -0.0454862912495931 | -0.08985416094462077 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-NoOpMatchedOverhead | 18 | -1.0427165428797405 | -0.9020136263635423 | -0.7229514122009277 | -0.6138217581642998 | -0.5328324834505717 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-RandomMatchedNorm | 18 | -1.0398643149269953 | -0.8991770545641581 | -0.7200969854990641 | -0.610995888710022 | -0.5299443602561951 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-SGD | 18 | -1.053671611679925 | -0.9286401867866516 | -0.7739667097727457 | -0.678559316529168 | -0.6042143702507019 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F10-T10-lowdegree-b1-consensus-transfer | 9 | -0.8274781968858507 | -0.5980270107587179 | -0.37350776460435653 | -0.3276219103071425 | -0.32277829779518974 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F10-T7-b1-cross-split-consensus-transfer | 9 | -0.39051520824432373 | 0.017322414451175265 | 0.2920369572109646 | 0.3785059452056885 | 0.433913442823622 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F10-T8-stable-b1-consensus-transfer | 9 | -1.0926813417010837 | -0.9770745237668356 | -0.8087313969930013 | -0.6992867125405205 | -0.6217128899362352 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F10-T9-b1-weak-stable-transfer | 9 | -1.1268161535263062 | -0.9898061553637186 | -0.8251791530185275 | -0.728652834892273 | -0.6144335468610128 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T1-loss-cotangent-target | 9 | -0.13140477074517143 | -0.035111831294165716 | 0.02942140234841241 | 0.07144372330771552 | 0.10998483498891194 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T1c-lowrank-loss-target | 9 | -1.094551099671258 | -0.9866169095039368 | -0.7967271010080973 | -0.7116134299172295 | -0.6226385103331672 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T5-random-matched-target | 9 | -0.48462027973598903 | -0.052856246630350746 | 0.2002672619289822 | 0.28580395380655926 | 0.34949660301208496 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T6-sign-flipped-target | 9 | -1.0494346221288045 | -0.9344958530531989 | -0.7889611456129286 | -0.6809942324956259 | -0.5811227361361185 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F3-T7-corrupted-label-target | 9 | -0.7132357226477729 | -0.27627329693900216 | 0.004983127117156982 | 0.077364186445872 | 0.11761703093846639 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-smallstep-alt100 | 9 | -0.45339955223931205 | 0.05144613981246948 | 0.290288335747189 | 0.36509129736158585 | 0.4081125060717265 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F6-KSW2-density-smallstep-alt50 | 9 | -0.1390277279747857 | 0.11703603797488743 | 0.2371760606765747 | 0.27402565876642865 | 0.3094544874297248 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F7-KSW2-warm400-smallstep-alt50 | 9 | -0.5142850743399726 | 0.0940022071202596 | 0.2275436520576477 | 0.2567328347100152 | 0.2859741846720378 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F8-KSW2-earlyboost-alt25 | 9 | -0.07972156339221531 | -0.04466014438205295 | 0.02271670765346951 | 0.05959012111028036 | 0.09423143333858913 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-T2-cross-split-consensus-target | 9 | -0.2430162231127421 | 0.07391336229112414 | 0.2416585154003567 | 0.29902200566397774 | 0.3369799852371216 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-T3-lowdegree-readout-target | 9 | -0.8301213317447238 | -0.7218688660197787 | -0.7310739225811429 | -0.8993670013215807 | -1.1286873883671231 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-T4-weak-stable-target | 9 | -1.1047808196809557 | -1.0039309461911519 | -0.8232811821831597 | -0.7168515655729506 | -0.6281381646792094 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-T5-b1-readout-transfer-target | 9 | -0.21748598416646323 | 0.052585972679985896 | 0.18800072537528145 | 0.24562016460630628 | 0.2925923665364583 | 0 | 0 | 1 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-T6-reservoir-excluding-consensus-target | 9 | -1.0933928622139826 | -0.9871740672323439 | -0.8311774465772841 | -0.7023883793089125 | -0.6345442599720426 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | F9-TCTRL-stable-random-target | 9 | -1.0669430494308472 | -0.9384166730774773 | -0.7906905280219184 | -0.6965491904152764 | -0.6270589762263827 | 0 | 0 | 0 |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW2-lowdegree-lowfreq-source-bank | 9 | -0.08301209741168553 | 0.015771216816372342 | 0.09415087434980604 | 0.13373857736587524 | 0.16506418916914198 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | 45 | 0.0 | 0.0 | -0.012087551752726237 | -0.03301738103230794 | -0.06400837898254394 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | 45 | -0.8818269663386875 | -0.7933920767572191 | -0.6746153712272644 | -0.5900603161917792 | -0.5243152247534858 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | 45 | -0.8788756145371331 | -0.790440316994985 | -0.6716669175359938 | -0.5871121750937568 | -0.5213639047410753 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | 45 | -0.8767326871554056 | -0.7869928850067986 | -0.6658408575587802 | -0.5783109956317478 | -0.5075859493679471 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F10-T10-lowdegree-b1-consensus-transfer | 9 | -0.7151699198616875 | -0.5494156744745042 | -0.4494073920779758 | -0.46017000410291886 | -0.5282894770304362 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F10-T7-b1-cross-split-consensus-transfer | 36 | -0.18991980453332266 | 0.0861756983730528 | 0.25859954290919834 | 0.3307586477862464 | 0.3805100487338172 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F10-T8-stable-b1-consensus-transfer | 9 | -0.8924983739852905 | -0.7914425598250495 | -0.32308777173360187 | 0.22099792295032078 | 0.4364536272154914 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F10-T9-b1-weak-stable-transfer | 9 | -0.9322298765182495 | -0.7159982721010844 | 0.014278014500935873 | 0.3472797671953837 | 0.45017073551813763 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T1-loss-cotangent-target | 9 | -0.01555772622426351 | 0.04722543557484945 | 0.10538820425669353 | 0.16130071216159397 | 0.2098553048239814 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T1c-lowrank-loss-target | 9 | -0.9088649087482028 | -0.7255818380249871 | 0.04462651411692301 | 0.32716427909003365 | 0.3677068286471897 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T5-random-matched-target | 9 | -0.24248802661895752 | 0.011266264650556777 | 0.15649689568413627 | 0.22090653578440347 | 0.28695056835810345 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T6-sign-flipped-target | 9 | -0.8662228186925253 | -0.7819264531135559 | -0.6712187661064996 | -0.5851422548294067 | -0.509845084614224 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F3-T7-corrupted-label-target | 9 | -0.4071096049414741 | -0.1144491367869907 | 0.07931431134541829 | 0.15014776918623182 | 0.2003944052590264 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F6-KSW2-density-smallstep-alt100 | 9 | -0.2220599518881904 | 0.10519052214092678 | 0.2422975500424703 | 0.3061026400989956 | 0.3575943112373352 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F6-KSW2-density-smallstep-alt50 | 9 | -0.055008888244628906 | 0.08625256352954441 | 0.16650489966074625 | 0.22061736053890652 | 0.27035339673360187 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F7-KSW2-warm400-smallstep-alt50 | 9 | -0.30730051464504665 | 0.12329170438978407 | 0.20506715774536133 | 0.2526482873492771 | 0.2992824051115248 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F8-KSW2-earlyboost-alt25 | 9 | -0.05788575278388129 | -0.026340186595916748 | 0.02359928025139703 | 0.07558147112528484 | 0.12368635336558025 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-T2-cross-split-consensus-target | 36 | -0.13877991338570914 | 0.051229746805297 | 0.17458238701025644 | 0.23599712053934732 | 0.27936848170227474 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-T3-lowdegree-readout-target | 9 | -0.5427244769202338 | -0.4203954339027405 | -0.4367693265279134 | -0.5640884902742174 | -0.7519177728229098 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-T4-weak-stable-target | 9 | -0.8370161188973321 | -0.21025078164206612 | 0.21198384629355538 | 0.27786217133204144 | 0.3219326933224996 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-T5-b1-readout-transfer-target | 36 | -0.06525200936529371 | 0.10889734327793121 | 0.20619777176115248 | 0.2617678675386641 | 0.30250321494208443 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-T6-reservoir-excluding-consensus-target | 9 | -0.8800850311915079 | -0.5580020546913147 | 0.19581478834152222 | 0.3143537640571594 | 0.4034358991516961 | 0 | 0 | 1 |
| D-FOU | FOU-R4-k4-triton-no-materialize | F9-TCTRL-stable-random-target | 36 | -0.9038917107714547 | -0.8260871238178678 | -0.6458035194211535 | -0.3279457456535763 | 0.04870697193675571 | 0 | 0 | 0 |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW2-lowdegree-lowfreq-source-bank | 36 | -0.050664595431751676 | 0.018502303295665316 | 0.08222057090865241 | 0.12972956399122873 | 0.16854933069811928 | 0 | 0 | 1 |
| MLP | R0-current | CTRL-AdamW | 9 | -0.5008625189463297 | -0.9800520208146837 | -1.391445451312595 | -1.5842475891113281 | -1.707501212755839 | 0 | 0 | 0 |
| MLP | R0-current | CTRL-NoOpMatchedOverhead | 9 | -0.5517058107588027 | -0.767681532435947 | -0.8915511237250434 | -0.8834781646728516 | -0.8295371929804484 | 0 | 0 | 0 |
| MLP | R0-current | CTRL-RandomMatchedNorm | 9 | -0.6187129020690918 | -0.8350612454944186 | -0.9592831399705675 | -0.95108429590861 | -0.8971814182069566 | 0 | 0 | 0 |
| MLP | R0-current | CTRL-SGD | 9 | -0.11904197269015843 | -0.051366209983825684 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F1-M2-strong-source | 9 | 0.2695307003127204 | -0.20655422740512425 | -0.4670611619949341 | -0.46855762269761825 | -0.3937758472230699 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F11-dual-timescale-retention-warm1200 | 9 | 0.1538872652583652 | -0.21655022435718113 | -0.34044403500027126 | -0.33240082528856063 | -0.2784902850786845 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F12-schedule-free-source-iterate | 9 | -0.5015321572621664 | -0.6932035949495103 | -0.7719775835673014 | -0.7212546136644151 | -0.6252798636754354 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F2-M15-weak-stable | 9 | -0.10026762220594618 | -0.029487589995066326 | 0.05191346009572347 | 0.0273274646864997 | 0.004945013258192275 | 0 | 0 | 1 |
| MLP | R0-current | MLP-F4-F10-cycle-hold | 9 | 0.005662838617960612 | -0.21031288305918375 | -0.5422436396280924 | -0.5993529690636529 | -0.5664696163601346 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F5-M2-plus-M15-anchor | 9 | 0.23094167974260119 | -0.25116262833277386 | -0.5198599166340299 | -0.532263974348704 | -0.4659545487827725 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F6-M2-source-with-slow-anchor | 9 | 0.12119702498118083 | -0.3652806546952989 | -0.5896648698382907 | -0.5796372890472412 | -0.4988101190990872 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F7-M2-source-with-matrix-block-retention | 9 | 0.1579514741897583 | -0.31929952568478054 | -0.5523486004935371 | -0.5457255707846748 | -0.4675045808156331 | 0 | 0 | 0 |
| MLP | R0-current | MLP-F9-cycle1200-to-M15-anchor | 9 | 0.2330689165327284 | -0.13539531495836046 | -0.2592810392379761 | -0.25122471650441486 | -0.1972733736038208 | 0 | 0 | 0 |

### D-FOU KSW2 Independent Confirmation

| run_label | offset | rows | h800 | h1600 | h3200 | h4800 | h6400 | h3200 cand | h6400 cand | late rebound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_kan_h6400 | 0 | 9 | 0.007741524113549126 | 0.06555078426996867 | 0.12966605689790514 | 0.1802871955765618 | 0.23139607244067723 | 1 | 1 | 0 |
| v2101_dfou_ksw2_independent_100000 | 100000 | 9 | -0.1071485612127516 | -0.050160970952775746 | 0.0011636151207817926 | 0.03729675875769721 | 0.06336870458390978 | 0 | 0 | 0 |
| v2101_dfou_ksw2_independent_200000 | 200000 | 9 | -0.029194533824920654 | 0.05271098348829481 | 0.12039375967449611 | 0.17117702960968018 | 0.2009810076819526 | 0 | 0 | 1 |
| v2101_dfou_ksw2_independent_300000 | 300000 | 9 | -0.07405681080288357 | 0.00590841637717353 | 0.07765885194142659 | 0.13015727202097574 | 0.1784515380859375 | 0 | 0 | 1 |

| decision | offset groups | reproduced h3200 | reproduced h6400 | pass |
|---|---:|---:|---:|---:|
| independent_confirmation_failed | 3 | 0 | 0 | 0 |

## F0 Source Dynamics Classification

| rows | classified_rows | classified_fraction | ambiguous_fraction |
|---:|---:|---:|---:|
| 864 | 864 | 1.0 | 0.0 |

## 修改记录

- 新增 v21.01 专用 common / S0.6 truth gate / source-retention runner / finalizer。
- `train_one()` horizon readback 增加 h6400，用于计划 13.6 的 late rebound 判定；v21.0 既有 artifact 未被覆盖。
- v21.01 source-retention runner 记录 h100/h400/h800/h1600/h2400/h3200/h4800/h6400、source derivative、retention ratio、washout/late-rebound/weak-stable/retained flags。
- 修复 v21.01 source matching：best matched control key 加入 `init_seed_offset`，避免 independent rerun 与 offset0 controls 混比。
- 新增 `experiments/run_v21_01_independent_confirmation.py`，把 offset0 与 independent offsets 分开汇总，输出 independent confirmation decision。
- S0.6 期间修复 tests repo-root import guard；修复 update semantics 测试使用 `UpdateTensor.tensor` API。
- Efficiency full-loop 初跑 D-CHE/D-FOU 均被 forward ratio 卡住；按计划追加 profiler repeat/warmup fallback 后 D-FOU 出现 S1 rows，D-CHE repeat2 在 128/256 batch 上也出现 S1 rows。
- KAN h6400 full matrix 出现 D-FOU KSW2 offset0 productive candidate；按计划追加 x3 independent offsets 后未复现，candidate 降级为 late rebound。
- S0.6 写入 import closure、LineC、retention/debt/route、update semantics、mechanism contract、profiler、kernel gradcheck 与 official fused status consistency artifacts。

## 分析 / Insight / 结论

- 本轮仍以 grouped source retention 为准；single row 或 late rebound 不触发 promotion。
- h6400 readback 用来区分 delayed migration、stochastic rebound 和真实 retained chain；前一 horizon 非正时，后续转正只写 late rebound。
- MLP 侧补齐 M2/M15/M47/M48/M12 后仍没有 productive h3200/h6400；M15 只形成 late rebound。
- D-FOU KSW2 的 offset0 结果一度满足 h800-h6400 连续正链，但 independent offsets 的 h800 grouped mean 全部转负，说明不是稳健 source-retention 机制。
- Function-space target source writer 产生较多 h3200/h6400 late rebound；random/corrupt target 也能出现后期正值，因此 target observable 仍不能作为 retained source 证据。
- `promotion_allowed=1` 只有 S0.6、D-CHE/D-FOU efficiency officialization、KAN h4800 retained source 与 controls attribution 都通过时才允许。
- 若 route 仍非 promotion，表示真实证据链未闭合，不是预算性放弃。

## 2026-06-04 02:52 F9-F12 continuation / late rebound independent classification

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention`
- promotion_allowed: 0
- source-retention matrix rows: 864
- source grouped rows: 61
- D-CHE / D-FOU S1 rows: 6 / 2
- F12 independent rows: 108
- F12 productive continuous groups: 0
- F12 late rebound target groups: 3
- F12 control h6400 late positive: 1
- required artifact missing count: 0

F12 是按计划 13.6 做的 independent late-rebound classification，不是继续调 scale/alt/rank。对象是 F9/F10 最强的 D-FOU target-family late rebound：

- `F9-T2-cross-split-consensus-target`
- `F9-T5-b1-readout-transfer-target`
- `F10-T7-b1-cross-split-consensus-transfer`
- matched control: `F9-TCTRL-stable-random-target`

三个 independent offsets 为 100000 / 200000 / 300000。

### 本轮新增/刷新 artifacts

- `v21_01_f12_late_rebound_independent_per_offset.csv`
- `v21_01_f12_late_rebound_independent_aggregate.csv`
- `v21_01_f12_late_rebound_independent_retained_row_examples.csv`
- `v21_01_f12_late_rebound_independent_decision.csv`
- 重新 merge `v21_01_source_retention_matrix.csv`，rows 从 756 增至 864。
- 重新 finalize route / manifest / packet / bundle。

### F12 independent aggregate

| v21_id | rows | h800 | h1600 | h2400 | h3200 | h4800 | h6400 | h800+ | h3200+ | h6400+ | retained rows | late rebound rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F10-T7-b1-cross-split-consensus-transfer | 27 | -0.186708 | 0.084997 | 0.197816 | 0.255302 | 0.325484 | 0.372676 | 5 | 25 | 26 | 5 | 6 |
| F9-T2-cross-split-consensus-target | 27 | -0.148453 | 0.036063 | 0.116023 | 0.157154 | 0.218211 | 0.257385 | 5 | 24 | 26 | 5 | 12 |
| F9-T5-b1-readout-transfer-target | 27 | -0.070701 | 0.097568 | 0.160779 | 0.194012 | 0.248332 | 0.285102 | 10 | 22 | 26 | 10 | 10 |
| F9-TCTRL-stable-random-target | 27 | -0.908067 | -0.826578 | -0.742400 | -0.655028 | -0.333128 | 0.038372 | 0 | 1 | 14 | 0 | 14 |

关键判定：

- 三个 target specs 都能独立复现 delayed migration：h1600/h3200/h4800/h6400 grouped mean 为正。
- 三个 target specs 的 independent h800 grouped mean 全部为负，所以没有一个是 continuous retained chain。
- F9-T5 最接近 early closure，h800 mean = -0.070701，且 10/27 rows h800 positive；但 grouped h800 仍未过线。
- stable-random control 在 h6400 也出现 late positive，h6400+ = 14/27，说明“后期正值”存在 control-equivalent / trajectory drift 成分，不能当 source-retention 证据。

### F12 per-offset evidence

| offset | v21_id | h800 | h1600 | h3200 | h4800 | h6400 | retained rows | late rebound rows |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100000 | F10-T7-b1-cross-split-consensus-transfer | -0.206022 | 0.067692 | 0.221363 | 0.287575 | 0.331103 | 2 | 3 |
| 200000 | F10-T7-b1-cross-split-consensus-transfer | -0.131768 | 0.142951 | 0.301167 | 0.370965 | 0.410782 | 2 | 1 |
| 300000 | F10-T7-b1-cross-split-consensus-transfer | -0.222335 | 0.044349 | 0.243375 | 0.317912 | 0.376142 | 1 | 2 |
| 100000 | F9-T2-cross-split-consensus-target | -0.143734 | 0.056695 | 0.148829 | 0.198325 | 0.228762 | 3 | 3 |
| 200000 | F9-T2-cross-split-consensus-target | -0.152629 | 0.017576 | 0.161647 | 0.228540 | 0.265127 | 1 | 5 |
| 300000 | F9-T2-cross-split-consensus-target | -0.148997 | 0.033918 | 0.160987 | 0.227768 | 0.278265 | 1 | 4 |
| 100000 | F9-T5-b1-readout-transfer-target | -0.075641 | 0.090766 | 0.181802 | 0.229131 | 0.259435 | 3 | 2 |
| 200000 | F9-T5-b1-readout-transfer-target | -0.041570 | 0.117158 | 0.205876 | 0.262720 | 0.292570 | 3 | 4 |
| 300000 | F9-T5-b1-readout-transfer-target | -0.094893 | 0.084780 | 0.194356 | 0.253144 | 0.303300 | 4 | 4 |
| 100000 | F9-TCTRL-stable-random-target | -0.917029 | -0.848268 | -0.660921 | -0.348280 | -0.041644 | 0 | 4 |
| 200000 | F9-TCTRL-stable-random-target | -0.896948 | -0.797326 | -0.628234 | -0.256379 | 0.120785 | 0 | 5 |
| 300000 | F9-TCTRL-stable-random-target | -0.910224 | -0.834140 | -0.675929 | -0.394725 | 0.035975 | 0 | 5 |

解释：

- target specs 在每个 independent offset 都呈现相同形态：early h800 negative，late horizons positive。
- 这更支持 delayed migration，而不是 stochastic single-offset rebound。
- 但 stable-random control 也在 h6400 有 late positive，说明 late horizon drift 不能独立说明 target 正确。
- 因此 F12 关闭了“F9/F10 late rebound 可能独立复现后变成 retained chain”的解释。

### 修改记录

- 本轮没有新增机制代码；使用既有 M54/M57/M60/M59 specs 做独立复现。
- 新增 F12 independent rerun artifacts 与 decision artifact。
- 执行日志补充 F12 exact `--spec-ids`、offset、shard、merge 命令，便于后续复现。
- 重新执行 `experiments/run_v21_01_finalize.py`，route 仍为 `R2-LateReboundNoContinuousRetention`。

### 最终 insight

- v21.01 的 functional blocker 已经从“是否有 late positive”推进到“early h800 source density 不能跨 dataset/seed/grouped mean 过线”。
- target-family 可以稳定制造 delayed migration，但这不是 retained source，因为 h800 grouped mean 没有正。
- stable-random control 的 h6400 late positive 提醒我们：后期 source 可能包含 trajectory drift / control-equivalent 成分，不能用 late horizon 单独 promotion。
- 当前继续同族 target scale、alt-period、rank、B1/B2 组合扫描的审计价值很低。
- 如果继续 v21.01 之外的新一轮，需要新的 train-only early-source selector 或 target-to-retention theory；否则只会把 delayed migration 做得更强，而不是形成连续 retained chain。

## 2026-06-04 F13 Late-Rebound Source-Target Anatomy

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F13DelayedMigrationControlDrift`
- promotion_allowed: 0
- F13 anatomy rows: 108
- delayed migration groups: 3
- continuous retention groups: 0
- stable-random control h6400 positive rows: 14
- train-only early-source predictor pass: 0
- actionable same-family repair: 0

F13 没有新增训练 row。它读取 F12 independent rerun 的 108 rows 和 matched stable-random control，按计划 13.6 把 late rebound 进一步拆成 delayed migration / control-equivalent drift / continuous retention。

### F13 Group Anatomy

| v21_id | classification | rows | h800 | h1600 | h3200 | h4800 | h6400 | matched control h6400+ | target late with control h6400+ | target-control h6400 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F9-T2-cross-split-consensus-target | delayed_migration_with_control_late_drift | 27 | -0.14845311420935173 | 0.03606309493382772 | 0.15715440335097136 | 0.21821109453837076 | 0.25738463799158734 | 14 | 14 | 0.21901251872380575 |
| F9-T5-b1-readout-transfer-target | delayed_migration_with_control_late_drift | 27 | -0.07070130772060818 | 0.09756777463135896 | 0.1940115248715436 | 0.24833166599273682 | 0.2851016565605446 | 14 | 14 | 0.24672953729276303 |
| F10-T7-b1-cross-split-consensus-transfer | delayed_migration_with_control_late_drift | 27 | -0.1867082052760654 | 0.08499738463649044 | 0.2553016918676871 | 0.3254839777946472 | 0.37267552260999326 | 14 | 14 | 0.3343034033422117 |
| F9-TCTRL-stable-random-target | control_equivalent_late_drift | 27 | -0.9080670299353423 | -0.8265783499788355 | -0.6550279347984879 | -0.33312831984625924 | 0.03837211926778158 |  |  |  |

### F13 Stratification Snapshot

| group | key | rows | h800+ | h3200+ | h6400+ | late rebound rows | h800 mean | h3200 mean | h6400 mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v21_id | F10-T7-b1-cross-split-consensus-transfer | 27 | 5 | 25 | 26 | 6 | -0.1867082052760654 | 0.2553016918676871 | 0.37267552260999326 |
| v21_id | F9-T2-cross-split-consensus-target | 27 | 6 | 24 | 26 | 12 | -0.14845311420935173 | 0.15715440335097136 | 0.25738463799158734 |
| v21_id | F9-T5-b1-readout-transfer-target | 27 | 10 | 22 | 26 | 10 | -0.07070130772060818 | 0.1940115248715436 | 0.2851016565605446 |
| dataset | Fashion-MNIST | 27 | 8 | 23 | 27 | 14 | -0.1486548008742156 | 0.18169523389251144 | 0.2616783203902068 |
| dataset | KMNIST | 27 | 12 | 25 | 27 | 4 | -0.02043073265640824 | 0.30231858182836463 | 0.38066296224240903 |
| dataset | MNIST | 27 | 1 | 23 | 24 | 10 | -0.23677709367540148 | 0.12245380436932599 | 0.2728205345295094 |
| offset | 100000 | 27 | 8 | 23 | 24 | 8 | -0.1417987788165057 | 0.18399815206174497 | 0.27309972047805786 |
| offset | 200000 | 27 | 6 | 25 | 27 | 10 | -0.10865557193756104 | 0.22289675253408928 | 0.322826553274084 |
| offset | 300000 | 27 | 7 | 23 | 27 | 10 | -0.15540827645195854 | 0.1995727154943678 | 0.31923554340998334 |
| dataset_offset | Fashion-MNIST/100000 | 9 | 3 | 8 | 9 | 5 | -0.15290623241000706 | 0.19422883457607693 | 0.25770288705825806 |
| dataset_offset | Fashion-MNIST/200000 | 9 | 1 | 7 | 9 | 6 | -0.13764034377204049 | 0.1543068223529392 | 0.23688295152452257 |
| dataset_offset | Fashion-MNIST/300000 | 9 | 4 | 8 | 9 | 3 | -0.15541782644059923 | 0.1965500447485182 | 0.2904491225878398 |
| dataset_offset | KMNIST/100000 | 9 | 5 | 9 | 9 | 2 | 0.022115707397460938 | 0.31997643576727974 | 0.38889486259884304 |
| dataset_offset | KMNIST/200000 | 9 | 5 | 9 | 9 | 0 | -0.007686813672383626 | 0.34545475906795925 | 0.4260674715042114 |
| dataset_offset | KMNIST/300000 | 9 | 2 | 7 | 9 | 2 | -0.07572109169430202 | 0.24152455064985487 | 0.32702655262417263 |
| dataset_offset | MNIST/100000 | 9 | 0 | 6 | 6 | 1 | -0.294605811436971 | 0.03778918584187826 | 0.1727014117770725 |
| dataset_offset | MNIST/200000 | 9 | 0 | 9 | 9 | 4 | -0.180639558368259 | 0.16892867618136936 | 0.30552923679351807 |
| dataset_offset | MNIST/300000 | 9 | 1 | 8 | 9 | 5 | -0.2350859112209744 | 0.16064355108473036 | 0.34023095501793754 |

### F13 Predictor Scan

| predictor | finite rows | Spearman h800 | Spearman h3200 | AUC h800+ | dataset min AUC | offset min AUC | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2_transfer_gain_h800 | 81 | -0.2938572719060524 | -0.10914634146341463 | 0.34444444444444444 | 0.11538461538461539 | 0.2714285714285714 | 0 |
| ActuationR2_h800 | 81 | -0.20300361336946704 | -0.15350045167118337 | 0.38333333333333336 | 0.19230769230769232 | 0.23015873015873015 | 0 |
| source_state_current_cos_h800 | 0 |  |  |  |  |  | 0 |
| source_state_gate_accept_h800 | 0 |  |  |  |  |  | 0 |
| generalization_gate_accept_h800 | 0 |  |  |  |  |  | 0 |
| LineC_channel_loss_h800 | 81 | -0.17709815450517588 | -0.3299167233120615 | 0.4103174603174603 | 0.038461538461538464 | 0.35714285714285715 | 0 |
| train_loss_h800 | 81 | -0.3062556458897922 | -0.001851851851851852 | 0.2968253968253968 | 0.0 | 0.2 | 0 |
| train_loss_drop_h100_h800 | 81 | 0.17809394760614272 | -0.08084914182475159 | 0.6198412698412699 | 0.46710526315789475 | 0.5634920634920635 | 0 |

### 修改记录

- 新增 `experiments/run_v21_01_f13_late_rebound_anatomy.py`。
- 新增 F13 official artifacts：`v21_01_f13_late_rebound_anatomy_summary.csv`、`v21_01_f13_late_rebound_row_vs_control.csv`、`v21_01_f13_late_rebound_stratification.csv`、`v21_01_f13_late_rebound_predictor_scan.csv`、`v21_01_f13_late_rebound_decision.csv`。
- 更新 `v21_01_route_decision.json`，只追加 F13 audit 字段；`promotion_allowed` 保持 0。
- 刷新 `v21_01_code_review_packet.zip` 与 `v21_01_results_bundle.zip`；精确 byte size 以最终 file stat 为准，避免把 zip 自身大小写进归档后造成自引用变动。

### F13 结论 / Insight

- F9/F10 target family 的 late positive 不是单个 offset 偶然反弹：三个 independent offsets 都呈现 h800 negative、h1600 以后转正的 delayed migration。
- 但它也不是 retained source：三个 target group 的 h800 grouped mean 仍为负，continuous retention groups = 0。
- stable-random control 在 h6400 出现 14 个 positive rows，说明 late horizon 正值里有 control-equivalent drift 成分，不能单独当成 source retention。
- target rows 相对 stable-random control 的 late horizon 确实更强，但 F13 predictor scan 没找到稳定可用的 train-only early-source selector。因此继续同族 target scale / alt / rank / B1-B2 组合扫描，预计只会增强 delayed migration，而不是闭合 h800->h6400 连续 retained chain。
- 当前诚实边界：`promotion_allowed=0`，`new_source_target_theory_required=1`，同族 repair 暂无可审计的下一步。

### F13 最终验证

- `v21_01_source_retention_matrix.csv` rows = 864。
- `v21_01_source_retention_summary.csv` rows = 61。
- F12 aggregate / F13 summary / F13 row-vs-control / F13 predictor / F13 decision rows = 4 / 4 / 81 / 8 / 1。
- `v21_01_required_artifact_manifest.csv` rows = 26，required missing = 0。
- `v21_01_code_review_packet.zip` 已刷新，packet 内包含 F12 aggregate 与 F13 decision。
- `v21_01_results_bundle.zip` 已刷新。
- 结束进程检查未发现持续运行的 v21.01 training、efficiency runner 或 GPU monitor；`ps` 仅匹配验证命令自身。

## 2026-06-04 F14 Early-Source Selector / Near-Closure Audit

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F13DelayedMigrationControlDrift`
- promotion_allowed: 0
- train-only selector pass: 0
- near-early-closure candidates: 1
- F15 independent confirmation recommended: 1
- best candidate: `F3-T1-loss-cotangent-target`

F14 没有新增训练 row。它读取已有 v21.01 source-retention matrix / summary，专门检查是否存在可提前使用的 train-only h800 selector，并排序所有 D-FOU target/KSW2 late-positive 候选。

### F14 Candidate Ranking

| rank | v21_id | rows | h800 gap | h800 | h1600 | h3200 | h6400 | h800+ rows | late chain | F15 recommended |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | F3-T1-loss-cotangent-target | 9 | 0.01555772622426351 | -0.01555772622426351 | 0.04722543557484945 | 0.10538820425669353 | 0.2098553048239814 | 4 | 1 | 1 |
| 2 | KSW2-lowdegree-lowfreq-source-bank | 36 | 0.050664595431751676 | -0.050664595431751676 | 0.018502303295665316 | 0.08222057090865241 | 0.16854933069811928 | 11 | 1 | 0 |
| 3 | F6-KSW2-density-smallstep-alt50 | 9 | 0.055008888244628906 | -0.055008888244628906 | 0.08625256352954441 | 0.16650489966074625 | 0.27035339673360187 | 3 | 1 | 0 |
| 4 | F9-T5-b1-readout-transfer-target | 36 | 0.06525200936529371 | -0.06525200936529371 | 0.10889734327793121 | 0.20619777176115248 | 0.30250321494208443 | 14 | 1 | 0 |
| 5 | F9-T2-cross-split-consensus-target | 36 | 0.13877991338570914 | -0.13877991338570914 | 0.051229746805297 | 0.17458238701025644 | 0.27936848170227474 | 8 | 1 | 0 |
| 6 | F10-T7-b1-cross-split-consensus-transfer | 36 | 0.18991980453332266 | -0.18991980453332266 | 0.0861756983730528 | 0.25859954290919834 | 0.3805100487338172 | 6 | 1 | 0 |
| 7 | F6-KSW2-density-smallstep-alt100 | 9 | 0.2220599518881904 | -0.2220599518881904 | 0.10519052214092678 | 0.2422975500424703 | 0.3575943112373352 | 1 | 1 | 0 |
| 8 | F7-KSW2-warm400-smallstep-alt50 | 9 | 0.30730051464504665 | -0.30730051464504665 | 0.12329170438978407 | 0.20506715774536133 | 0.2992824051115248 | 1 | 1 | 0 |
| 9 | F8-KSW2-earlyboost-alt25 | 9 | 0.05788575278388129 | -0.05788575278388129 | -0.026340186595916748 | 0.02359928025139703 | 0.12368635336558025 | 3 | 0 | 0 |
| 10 | F9-T3-lowdegree-readout-target | 9 | 0.5427244769202338 | -0.5427244769202338 | -0.4203954339027405 | -0.4367693265279134 | -0.7519177728229098 | 0 | 0 | 0 |
| 11 | F10-T10-lowdegree-b1-consensus-transfer | 9 | 0.7151699198616875 | -0.7151699198616875 | -0.5494156744745042 | -0.4494073920779758 | -0.5282894770304362 | 0 | 0 | 0 |
| 12 | F9-T4-weak-stable-target | 9 | 0.8370161188973321 | -0.8370161188973321 | -0.21025078164206612 | 0.21198384629355538 | 0.3219326933224996 | 0 | 0 | 0 |

### F14 Train-Only Predictor Scan

| predictor | kind | finite rows | Spearman h800 | AUC h800+ | dataset min AUC | offset min AUC | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| train_loss_h100 | train_only | 252 | -0.6357434931832991 | 0.2980769230769231 | 0.16296296296296298 | 0.20347222222222222 | 0 |
| train_loss_h400 | train_only | 252 | -0.6746647889900964 | 0.27192307692307693 | 0.14074074074074075 | 0.19930555555555557 | 0 |
| train_loss_h800 | train_only | 252 | -0.7226070309389208 | 0.2414423076923077 | 0.09925925925925926 | 0.16701388888888888 | 0 |
| train_loss_drop_h100_h800 | train_only | 252 | 0.6788805628230538 | 0.7449038461538462 | 0.7096296296296296 | 0.5843621399176955 | 0 |
| CEp99_h800 | audit_metric | 252 | 0.6739974039471701 | 0.7414423076923077 | 0.7128526645768025 | 0.5020576131687243 | 0 |
| ECE_h800 | audit_metric | 252 | 0.384249563012771 | 0.5949038461538462 | 0.6 | 0.43209876543209874 | 0 |
| Brier_h800 | audit_metric | 252 | -0.5325177738134607 | 0.3697115384615385 | 0.13925925925925925 | 0.23159722222222223 | 0 |
| LineC_channel_loss_h800 | audit_metric | 252 | -0.13174109709797094 | 0.37456730769230767 | 0.22074074074074074 | 0.30246913580246915 | 0 |
| B2_transfer_gain_h800 | function_readback | 252 | -0.1780328338443703 | 0.4051923076923077 | 0.373469387755102 | 0.2716049382716049 | 0 |
| ActuationR2_h800 | function_readback | 252 | 0.5385309880373106 | 0.6579807692307692 | 0.5163265306122449 | 0.3333333333333333 | 0 |
| source_h400 | source_readback_forbidden | 252 | 0.9056699983427855 | 0.9263461538461538 | 0.9040816326530612 | 0.8847736625514403 | 0 |

### F14 结论 / 下一步

- F14 没有找到合法 train-only early-source selector：所有 train-only predictors 的 pass 都为 0。
- 但 D-FOU `F3-T1-loss-cotangent-target` 是明确的 near-early-closure candidate：h800 gap 最小，且 h1600/h3200/h6400 均为正。
- 这不能 promotion，也不能写 retained source；它只允许一个计划内 F15 independent confirmation：用 x3 init offsets + matched random/sign-flip/corrupt/stable-random controls 检查它是否只是 offset0 近似闭合。

## 2026-06-04 F15 Loss-Target Near-Closure Independent Confirmation

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F15LossTargetNoEarlyClosure`
- promotion_allowed: 0
- F15 rows: 162
- blocked rows: 0
- candidate: `F3-T1-loss-cotangent-target`
- candidate h800 mean: -0.08055646772737857
- candidate continuous group: 0
- matched random h6400 mean: 0.3037824586585716
- stable-random h6400 positive rows: 15

F15 是 F14 推荐的唯一 near-early-closure 候选验证：`F3-T1-loss-cotangent-target` 在 offset0 h800 gap 只有 0.0156，因此按计划 13.6 做 x3 independent offsets，并加入 matched random / sign-flip / corrupted-label / stable-random controls。

### F15 Aggregate

| v21_id | rows | h800 | h1600 | h2400 | h3200 | h4800 | h6400 | h800+ rows | retained rows | late rebound rows | continuous group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F3-T1-loss-cotangent-target | 27 | -0.08055646772737857 | -0.015824158986409504 | 0.022960119777255587 | 0.047377071998737474 | 0.09254664182662964 | 0.1281579821198075 | 7 | 7 | 12 | 0 |
| F3-T1c-lowrank-loss-target | 27 | -0.9362201580294857 | -0.7830616301960416 | -0.4063717215149491 | 0.032271025357423 | 0.3211403422885471 | 0.34869962047647546 | 0 | 0 | 26 | 0 |
| F3-T5-random-matched-target | 27 | -0.17654183396586665 | 0.07028456970497414 | 0.15729349630850334 | 0.1939143935839335 | 0.25667446189456516 | 0.3037824586585716 | 6 | 6 | 10 | 0 |
| F3-T6-sign-flipped-target | 27 | -0.8915346971264592 | -0.8061893935556765 | -0.7373089348828351 | -0.68953318728341 | -0.6010457895420216 | -0.5323047196423566 | 0 | 0 | 2 | 0 |
| F3-T7-corrupted-label-target | 27 | -0.5084020848627444 | -0.18191019473252473 | -0.04421748055352105 | 0.007341638759330467 | 0.08403587341308594 | 0.12813695271809897 | 2 | 2 | 11 | 0 |
| F9-TCTRL-stable-random-target | 27 | -0.9037925821763498 | -0.8292763917534439 | -0.7589026910287363 | -0.6634987084953873 | -0.3675036871874774 | 0.027585373984442815 | 0 | 0 | 14 | 0 |

### F15 Per-Offset

| offset | v21_id | rows | h800 | h1600 | h3200 | h6400 | retained rows | late rebound rows |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100000 | F3-T1-loss-cotangent-target | 9 | -0.1591210232840644 | -0.11890086200502184 | -0.07414650917053223 | -0.012657437059614394 | 1 | 4 |
| 100000 | F3-T1c-lowrank-loss-target | 9 | -0.9429599245389303 | -0.7965508500734965 | -0.020007663302951388 | 0.2866210208998786 | 0 | 8 |
| 100000 | F3-T5-random-matched-target | 9 | -0.21482052405675253 | 0.0002044240633646647 | 0.12352223528756036 | 0.2167593174510532 | 2 | 5 |
| 100000 | F3-T6-sign-flipped-target | 9 | -0.9064521855778165 | -0.8295862740940518 | -0.721458011203342 | -0.5804702705807157 | 0 | 1 |
| 100000 | F3-T7-corrupted-label-target | 9 | -0.601659185356564 | -0.23525361882315743 | -0.02748299969567193 | 0.07695894771152073 | 1 | 4 |
| 100000 | F9-TCTRL-stable-random-target | 9 | -0.9229085246721903 | -0.8674018979072571 | -0.7298364109463162 | -0.05117343531714545 | 0 | 4 |
| 200000 | F3-T1-loss-cotangent-target | 9 | -0.029992957909901936 | 0.049778680006663 | 0.1251522633764479 | 0.21050584978527492 | 4 | 4 |
| 200000 | F3-T1c-lowrank-loss-target | 9 | -0.9232323235935636 | -0.753449903594123 | 0.02698550621668498 | 0.38251574171913993 | 0 | 9 |
| 200000 | F3-T5-random-matched-target | 9 | -0.17825724681218466 | 0.0952747729089525 | 0.22574908865822685 | 0.34202998214297825 | 2 | 3 |
| 200000 | F3-T6-sign-flipped-target | 9 | -0.8752119077576531 | -0.7880727317598131 | -0.6726759870847067 | -0.5110730197694566 | 0 | 1 |
| 200000 | F3-T7-corrupted-label-target | 9 | -0.5113873680432638 | -0.16981473233964708 | 0.035717533694373235 | 0.15971529483795166 | 0 | 4 |
| 200000 | F9-TCTRL-stable-random-target | 9 | -0.8835233118798997 | -0.7973431878619723 | -0.6199299030833774 | 0.04597836070590549 | 0 | 4 |
| 300000 | F3-T1-loss-cotangent-target | 9 | -0.05255542198816935 | 0.021649705039130315 | 0.09112546179029676 | 0.18662553363376194 | 2 | 4 |
| 300000 | F3-T1c-lowrank-loss-target | 9 | -0.9424682259559631 | -0.799184136920505 | 0.08983523315853542 | 0.3769620988104079 | 0 | 9 |
| 300000 | F3-T5-random-matched-target | 9 | -0.1365477310286628 | 0.11537451214260525 | 0.23247185680601332 | 0.35255807638168335 | 2 | 2 |
| 300000 | F3-T6-sign-flipped-target | 9 | -0.892939998043908 | -0.8009091748131646 | -0.6744655635621812 | -0.5053708685768975 | 0 | 0 |
| 300000 | F3-T7-corrupted-label-target | 9 | -0.41215970118840534 | -0.14066223303476968 | 0.013790382279290093 | 0.14773661560482448 | 1 | 3 |
| 300000 | F9-TCTRL-stable-random-target | 9 | -0.9049459099769592 | -0.8230840894911025 | -0.6407298114564683 | 0.08795119656456842 | 0 | 6 |

### 修改记录

- 新增 `experiments/run_v21_01_f15_loss_target_independent_summary.py`。
- 新增 F15 official artifacts：`v21_01_f15_loss_target_independent_aggregate.csv`、`v21_01_f15_loss_target_independent_per_offset.csv`、`v21_01_f15_loss_target_independent_row_examples.csv`、`v21_01_f15_loss_target_independent_decision.csv`。
- F15 训练命令已写入执行日志；本节只汇总真实落盘 rows。
- 更新 `v21_01_route_decision.json`，`promotion_allowed` 仍为 0。

### F15 结论 / Insight

- F14 的 near-closure 候选没有独立转正：`F3-T1-loss-cotangent-target` independent h800 mean = -0.08055646772737857，h1600 mean = -0.015824158986409504，不是连续 retained chain。
- 它仍然有 delayed migration：h2400/h3200/h4800/h6400 grouped mean 转正，但前两个 horizon 没闭合。
- matched random target 也出现 h1600-h6400 正值，且 retained rows 与 candidate 接近，说明 loss target 的 late positive 不是足够强的 attribution。
- stable-random control 仍有 h6400 positive rows，继续支持 F13 的 control-equivalent late drift 判定。
- 因此 F14 提出的唯一可行动 near-closure 线索已被 F15 关闭。当前 v21.01 不能继续同族 target/source repair；需要新的 source-target theory，而不是继续调现有 target 族。

### F15 最终验证 / 审计边界

- `v21_01_source_retention_matrix.csv` rows=1026。
- F15 independent rows=162，blocked rows=0。
- `v21_01_required_artifact_manifest.csv` rows=33，required missing count=0。
- route 已更新为 `R2-LateReboundNoContinuousRetention-F15LossTargetNoEarlyClosure`。
- `promotion_allowed=0`，`new_source_target_theory_required=1`。
- 结束进程检查没有发现持续运行的 v21.01 training、GPU monitor 或 `nvidia-smi` 采样进程。
- `v21_01_code_review_packet.zip` 和 `v21_01_results_bundle.zip` 在本节写入后再次刷新，确保最终包包含最新执行日志和复盘。

最终边界：v21.01 没有达成目标。F14/F15 已按计划把最接近 h800 闭合的 D-FOU loss-cotangent target 做了 x3 independent confirmation；结果没有连续 h800/h1600/h3200/h4800/h6400 retained chain，且 matched random/stable-random controls 仍支持 late drift / delayed migration 解释。当前不能写 breakthrough，也不能把 `promotion_allowed` 改为 1。

## 2026-06-04 F16 Source-Target Theory Closure Audit

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F16SourceTargetTheoryInsufficient`
- promotion_allowed: 0
- F16 target-like rows: 657
- continuous rows: 78
- continuous groups: 0
- delayed migration rows: 199
- legal continuous predictor pass: 1
- current target/source theory closed: 0

F16 没有新增 FU token，也没有新增训练 row。它按计划 13.5 在 MLP/KAN 均未 retained、F15 已关闭 near-closure 后，回到 target/source theory：检查现有 target/source diagnostics 是否能合法区分 continuous retained source、delayed migration 和 control-equivalent late drift。

### F16 Target/Source Class Summary

| carrier | v21_id | rows | control | h800 | h3200 | h6400 | continuous rows | delayed rows | continuous group | delayed group |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-FOU | F9-T5-b1-readout-transfer-target | 36 | 0 | -0.06525200936529371 | 0.20619777176115248 | 0.30250321494208443 | 14 | 15 | 0 | 1 |
| D-FOU | KSW2-lowdegree-lowfreq-source-bank | 36 | 0 | -0.050664595431751676 | 0.08222057090865241 | 0.16854933069811928 | 11 | 12 | 0 | 1 |
| D-FOU | F3-T1-loss-cotangent-target | 36 | 0 | -0.0643067823515998 | 0.06187985506322649 | 0.14858231279585096 | 11 | 8 | 0 | 1 |
| D-FOU | F9-T2-cross-split-consensus-target | 36 | 0 | -0.13877991338570914 | 0.17458238701025644 | 0.27936848170227474 | 9 | 20 | 0 | 1 |
| D-FOU | F3-T5-random-matched-target | 36 | 1 | -0.19302838212913936 | 0.1845600191089842 | 0.29957448608345455 | 8 | 15 | 0 | 1 |
| D-FOU | F10-T7-b1-cross-split-consensus-transfer | 36 | 0 | -0.18991980453332266 | 0.25859954290919834 | 0.3805100487338172 | 6 | 25 | 0 | 1 |
| D-FOU | F3-T7-corrupted-label-target | 36 | 1 | -0.4830789648824268 | 0.025334806905852422 | 0.14620131585333082 | 3 | 10 | 0 | 0 |
| D-FOU | F6-KSW2-density-smallstep-alt50 | 9 | 0 | -0.055008888244628906 | 0.16650489966074625 | 0.27035339673360187 | 3 | 3 | 0 | 1 |
| D-FOU | F8-KSW2-earlyboost-alt25 | 9 | 0 | -0.05788575278388129 | 0.02359928025139703 | 0.12368635336558025 | 3 | 1 | 0 | 1 |
| D-CHE | F6-KSW2-density-smallstep-alt50 | 9 | 0 | -0.1390277279747857 | 0.2371760606765747 | 0.3094544874297248 | 2 | 5 | 0 | 1 |
| D-CHE | F6-KSW2-density-smallstep-alt100 | 9 | 0 | -0.45339955223931205 | 0.290288335747189 | 0.4081125060717265 | 1 | 7 | 0 | 1 |
| D-FOU | F6-KSW2-density-smallstep-alt100 | 9 | 0 | -0.2220599518881904 | 0.2422975500424703 | 0.3575943112373352 | 1 | 7 | 0 | 1 |
| D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.39051520824432373 | 0.2920369572109646 | 0.433913442823622 | 1 | 6 | 0 | 1 |
| D-CHE | F9-T5-b1-readout-transfer-target | 9 | 0 | -0.21748598416646323 | 0.18800072537528145 | 0.2925923665364583 | 1 | 6 | 0 | 1 |
| D-CHE | F3-T5-random-matched-target | 9 | 1 | -0.48462027973598903 | 0.2002672619289822 | 0.34949660301208496 | 1 | 5 | 0 | 1 |
| D-CHE | KSW2-lowdegree-lowfreq-source-bank | 9 | 0 | -0.08301209741168553 | 0.09415087434980604 | 0.16506418916914198 | 1 | 5 | 0 | 1 |

### F16 F15 Candidate vs Controls

| control | paired rows | h800 adv | h800 wins | h1600 adv | h1600 wins | h3200 adv | h3200 wins | h6400 adv | h6400 wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F3-T5-random-matched-target | 27 | 0.09598536623848809 | 20 | -0.08610872869138364 | 8 | -0.14653732158519603 | 6 | -0.17562447653876412 | 5 |
| F3-T6-sign-flipped-target | 27 | 0.8109782293990806 | 27 | 0.790365234569267 | 27 | 0.7369102592821475 | 26 | 0.660462701762164 | 26 |
| F3-T7-corrupted-label-target | 27 | 0.4278456171353658 | 26 | 0.16608603574611522 | 21 | 0.04003543323940701 | 16 | 2.1029401708532264e-05 | 15 |
| F9-TCTRL-stable-random-target | 27 | 0.8232361144489713 | 27 | 0.8134522327670345 | 27 | 0.7108757804941248 | 27 | 0.10057260813536467 | 20 |

### F16 Observability Predictor Scan

| predictor | kind | finite rows | AUC continuous | AUC late | dataset min AUC | offset min AUC | Spearman h6400 | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train_loss_drop_h100_h800 | train_only | 657 | 0.8298126743722599 | 0.7136226986460688 | 0.7952577319587629 | 0.7512605042016807 | 0.3782393241846217 | 1 |
| train_loss_h800 | train_only | 657 | 0.17782649129799388 | 0.29595576133944834 | 0.07604895104895106 | 0.16052747252747251 | -0.34654628191123016 | 0 |
| LineC_channel_loss_h800 | audit_readback | 657 | 0.43773526416013464 | 0.49399837616027736 | 0.28780594405594406 | 0.3630952380952381 | -0.07175281074594761 | 0 |
| linec_loss_drop_h100_h800 | audit_readback | 657 | 0.5348412382091139 | 0.5040541133615677 | 0.4187062937062937 | 0.5055384615384615 | 0.04874037825967732 | 0 |
| B2_transfer_gain_h800 | function_readback | 657 | 0.5106726894291661 | 0.6431941366219746 | 0.48560667204735003 | 0.5109243697478991 | 0.17554210260081501 | 0 |
| b2_gain_drop_h100_h800 | function_readback | 639 | 0.7114202523455192 | 0.6021488889937255 | 0.6659574468085107 | 0.6655462184873949 | 0.1582693467612634 | 0 |
| ActuationR2_h800 | function_readback | 657 | 0.7528895974491829 | 0.7173750850321476 | 0.6797938144329897 | 0.6781512605042017 | 0.3490635961957246 | 0 |
| CEp99_h800 | audit_readback | 657 | 0.8080465878393339 | 0.6592460117179786 | 0.7644230769230769 | 0.7470588235294118 | 0.2873246343370495 | 0 |
| Brier_h800 | audit_readback | 657 | 0.26287586909348565 | 0.27515305786574795 | 0.08916083916083917 | 0.22804395604395605 | -0.3012654295305284 | 0 |
| source_state_gate_accept_h800 | train_stream_diagnostic | 0 |  |  |  |  |  | 0 |
| source_state_current_cos_h800 | train_stream_diagnostic | 0 |  |  |  |  |  | 0 |
| generalization_gate_accept_h800 | train_stream_diagnostic | 0 |  |  |  |  |  | 0 |
| projection_residual_norm | target_diagnostic | 657 | 0.26389442451618617 | 0.293443198525378 | 0.14772727272727273 | 0.2119047619047619 | -0.43428503689576403 | 0 |
| function_displacement_norm | target_diagnostic | 657 | 0.34566671095168505 | 0.4316341532992473 | 0.24431818181818182 | 0.3083076923076923 | -0.08572694733297066 | 0 |
| operator_parameter_norm | target_diagnostic | 657 | 0.3702448961516319 | 0.4996488995194312 | 0.32954545454545453 | 0.3100840336134454 | 0.15993010793739773 | 0 |
| source_h400 | source_readback_forbidden | 657 | 0.9312696514769053 | 0.6473085953786399 | 0.8954639175257731 | 0.916043956043956 | 0.4424214235289356 | 0 |

### 修改记录

- 新增 `experiments/run_v21_01_f16_source_target_theory_audit.py`。
- 新增 F16 official artifacts：`v21_01_f16_source_target_theory_summary.csv`、`v21_01_f16_pairwise_target_control_advantage.csv`、`v21_01_f16_observability_predictor_scan.csv`、`v21_01_f16_route_closure_decision.csv`。
- 更新 `v21_01_route_decision.json`：`promotion_allowed` 仍为 0，`new_source_target_theory_required=1`。

### F16 结论 / Insight

- 当前 target/source theory 仍没有打开 S2/S3 route：continuous group 数为 0。
- F16 发现的 legal continuous predictor pass 数为 1。这不是 promotion；它只允许一个 held-out selector 验证，检查该 train-only 信号是否能在独立 rows 上形成连续 retained group，并排除 matched-random contamination。
- `F3-T1-loss-cotangent-target` 在 F15 中只在 h800 早期强过 random control；到 h1600/h3200/h6400 反而输给 matched random target，说明当前 loss-cotangent target 不是可靠的 retained-source target。
- 若 predictor 只预测 late migration 而不能预测 continuous retention，它不能作为 direction selector；F16 没有发现跨 dataset/offset 稳定的合法 continuous selector。
- source readback 类 predictor 仍可作为 forbidden reference，但不能用于生成方向或 promotion。
- 因此 v21.01 在当前证据边界下从“继续调 target 参数”推进到“必须验证 train-only selector 是否可行动”。如果 held-out selector 不能形成连续 retained group，则当前 source-target observable 理论仍不足。

## 2026-06-04 F17 Train-Loss Selector Held-Out Validation

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F17TrainLossSelectorNotActionable`
- promotion_allowed: 0
- selector: `train_loss_drop_h100_h800`
- threshold learned on non-F15 rows: 1.68036288022995
- holdout selected rows: 40
- holdout selected h800 mean: -0.08922969549894333
- holdout selected h3200 mean: 0.10135614424943924
- holdout selected group continuous: 0
- selector actionable: 0

F17 验证 F16 发现的唯一 legal predictor：`train_loss_drop_h100_h800`。阈值只在非 F15 rows 上选择，然后在 F15 independent rows 上验证，避免把同表相关性直接写成可行动 selector。

### F17 Threshold Fit

| selector | train rows | threshold | tp | fp | fn | tn | precision | recall | f1 | youden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_loss_drop_h100_h800 | 495 | 1.68036288022995 | 39 | 76 | 24 | 356 | 0.3391304347826087 | 0.6190476190476191 | 0.4382022471910112 | 0.443121693121693 |

### F17 Hold-Out Validation

| split | rows | continuous all | selected rows | selected controls | selected continuous | selected continuous fraction | selected h800 | selected h1600 | selected h3200 | selected h6400 | selected group continuous |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fit_non_f15 | 495 | 63 | 115 | 3 | 39 | 0.3391304347826087 | -0.06978786302649456 | 0.06257589537164439 | 0.15242347976435786 | 0.250881288362586 | 0 |
| holdout_f15 | 162 | 15 | 40 | 13 | 11 | 0.275 | -0.08922969549894333 | 0.022504188120365143 | 0.10135614424943924 | 0.18228867799043655 | 0 |
| all | 657 | 78 | 155 | 16 | 50 | 0.3225806451612903 | -0.07480511011615876 | 0.05223480962937878 | 0.13924481253470145 | 0.23317996955687 | 0 |

### F17 Selected Group Evidence

| split | carrier | v21_id | rows | control | continuous rows | score mean | h800 | h3200 | h6400 | group continuous |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit_non_f15 | D-FOU | KSW2-lowdegree-lowfreq-source-bank | 34 | 0 | 11 | 1.8045074069762932 | -0.04822462095933802 | 0.0884980717126061 | 0.17571969242656932 | 0 |
| fit_non_f15 | D-FOU | F9-T2-cross-split-consensus-target | 25 | 0 | 8 | 1.764448925256729 | -0.13203233003616333 | 0.17481001138687133 | 0.27219802618026734 | 0 |
| fit_non_f15 | D-FOU | F9-T5-b1-readout-transfer-target | 23 | 0 | 10 | 1.7508120484974072 | -0.04392377449118573 | 0.22686045843621958 | 0.3171667897182962 | 0 |
| fit_non_f15 | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | 4 | 1.8107737112376425 | -0.01555772622426351 | 0.10538820425669353 | 0.2098553048239814 | 0 |
| fit_non_f15 | D-FOU | F6-KSW2-density-smallstep-alt50 | 9 | 0 | 3 | 1.8334608342912462 | -0.055008888244628906 | 0.16650489966074625 | 0.27035339673360187 | 0 |
| fit_non_f15 | D-CHE | KSW2-lowdegree-lowfreq-source-bank | 6 | 0 | 1 | 1.7468948513269424 | -0.08616578578948975 | 0.12784514824549356 | 0.27793190876642865 | 0 |
| fit_non_f15 | D-CHE | F3-T1-loss-cotangent-target | 4 | 0 | 0 | 1.730243280529976 | -0.1587112545967102 | 0.07666411995887756 | 0.22914323210716248 | 0 |
| fit_non_f15 | D-FOU | F3-T5-random-matched-target | 3 | 1 | 1 | 1.7976078589757283 | -0.10575604438781738 | 0.13682218392690024 | 0.23493963479995728 | 0 |
| fit_non_f15 | D-CHE | F6-KSW2-density-smallstep-alt50 | 1 | 0 | 1 | 1.695587396621704 | 0.13337552547454834 | 0.7016005516052246 | 0.7706557512283325 | 1 |
| fit_non_f15 | D-CHE | F9-T5-b1-readout-transfer-target | 1 | 0 | 0 | 1.7015155851840973 | -0.1040802001953125 | 0.2988924980163574 | 0.3955744504928589 | 0 |
| holdout_f15 | D-FOU | F3-T1-loss-cotangent-target | 27 | 0 | 7 | 1.815257972865193 | -0.08055646772737857 | 0.047377071998737474 | 0.1281579821198075 | 0 |
| holdout_f15 | D-FOU | F3-T5-random-matched-target | 13 | 1 | 4 | 1.789384241287525 | -0.1072433224091163 | 0.21346652507781982 | 0.29471396941405076 | 0 |

### 修改记录

- 新增 `experiments/run_v21_01_f17_train_loss_selector_holdout.py`。
- 新增 F17 official artifacts：`v21_01_f17_train_loss_selector_threshold.csv`、`v21_01_f17_train_loss_selector_holdout.csv`、`v21_01_f17_train_loss_selector_selected_groups.csv`、`v21_01_f17_train_loss_selector_decision.csv`。
- 更新 `v21_01_route_decision.json`，`promotion_allowed` 仍为 0。

### F17 结论 / Insight

- F16 的 train-only predictor 不是可直接行动的 selector：F15 holdout selected group 的 h800 mean = -0.08922969549894333，仍为负，continuous group=0。
- 该 selector 同时选中 matched random target rows；在 F15 selected groups 中，random control 也保留 late-positive source，因此不能作为 target attribution。
- 它可以作为 failure taxonomy 的解释变量：train loss drop 能预测一部分 row-level continuous，但不能把 grouped source chain 闭合，也不能排除 control contamination。
- 当前不能把 `train_loss_drop_h100_h800` 写成 direction selector，更不能把它接入 promotion。继续推进需要新的 toy-correctness/source-target 假设，而不是把这个 post-h800 train metric 当作突破。

### F17 最终验证 / 审计边界

- `v21_01_source_retention_matrix.csv` rows=1026。
- `v21_01_required_artifact_manifest.csv` rows=41，required missing count=0。
- F16 decision rows=1，F17 decision rows=1。
- route 已更新为 `R2-LateReboundNoContinuousRetention-F17TrainLossSelectorNotActionable`。
- `promotion_allowed=0`，`f16_legal_predictor_pass=1`，`f17_selector_actionable=0`，`new_source_target_theory_required=1`。
- 结束进程检查没有发现持续运行的 v21.01 training、GPU monitor 或 `nvidia-smi` 采样进程。
- `v21_01_code_review_packet.zip` 和 `v21_01_results_bundle.zip` 在本节写入后再次刷新，确保最终包包含最新执行日志和复盘。

最终边界：v21.01 仍没有达成目标。F16 找到了一个 train-only 解释信号，但 F17 held-out 验证显示它不能形成可行动 selector：选中 rows 的 h800 grouped mean 仍为负，且 matched random control 仍混入 late-positive。当前不能继续把同族 target/FU 小修写成进展；需要新的 source-target toy-correctness 假设之后才能开启下一轮机制实现。

## 2026-06-04 F18 Contrastive Loss-Random Orthogonal Target

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F18ContrastiveTargetSmokeNoEarlySignal`
- promotion_allowed: 0
- F18 rows / blocked rows: 90 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- control early-chain groups: 0
- full escalation recommended: 0

F18 是 F17 后的新 source-target toy-correctness 尝试：把 loss-cotangent target 中按 row 可由 matched random target 解释的分量投影掉，检验 late rebound 是否来自 random-control 等价方向。它不是调 actuation rank、alt period 或 FU lr。

### F18 Group Evidence

| run_label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h6400 | early chain | productive h3200 | late rows |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-AdamW | 9 | 0 | -0.019990212387508817 | -0.029322776529524062 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9282578229904175 | -0.8520674705505371 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9262769884533353 | -0.8500887817806668 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.9207724067899916 | -0.8435791333516439 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T11-contrastive-loss-random-orthogonal | 9 | 0 | -0.1183696190516154 | -0.07488199075063069 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T12-b1-contrastive-loss-random-orthogonal | 9 | 0 | -0.12344996134440105 | 0.07461712095472547 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2527148061328464 | -0.0570918255382114 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T6-sign-flipped-target | 9 | 0 | -0.9338163137435913 | -0.8614021142323812 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T7-corrupted-label-target | 9 | 0 | -0.49592339992523193 | -0.20482831531100804 |  |  | 0 | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F9-TCTRL-stable-random-target | 9 | 0 | -0.9318630562888252 | -0.8837844530741373 |  |  | 0 | 0 | 0 |

### F18 Row Examples

| run_label | carrier | v21_id | dataset | seed | offset | h800 | h1600 | h3200 | h6400 | retained | late |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T11-contrastive-loss-random-orthogonal | MNIST | 0 | 0 | 0.04325675964355469 | 0.09499537944793701 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T6-sign-flipped-target | MNIST | 0 | 0 | -0.9913449287414551 | -0.9697138071060181 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 0 | 0 | -0.9889907836914062 | -0.9639567136764526 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 1 | 0 | -0.455005407333374 | -0.25089704990386963 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-SGD | MNIST | 1 | 0 | -0.8548134565353394 | -0.8008289337158203 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T11-contrastive-loss-random-orthogonal | MNIST | 2 | 0 | -0.14355039596557617 | -0.11387109756469727 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T6-sign-flipped-target | MNIST | 2 | 0 | -1.0821428298950195 | -1.0443769693374634 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 2 | 0 | -1.0800020694732666 | -1.0419865846633911 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T5-random-matched-target | Fashion-MNIST | 0 | 0 | -0.2529538869857788 | -0.030997872352600098 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 0 | 0 | -1.3939650058746338 | -1.3386546969413757 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T11-contrastive-loss-random-orthogonal | Fashion-MNIST | 1 | 0 | -0.2913907766342163 | -0.2846899628639221 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T6-sign-flipped-target | Fashion-MNIST | 1 | 0 | -1.4483168125152588 | -1.4119346737861633 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | 0 | -1.4399449825286865 | -1.4019303917884827 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T5-random-matched-target | Fashion-MNIST | 2 | 0 | -0.03349876403808594 | 0.14506042003631592 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 2 | 0 | -0.6784794330596924 | -0.5039129257202148 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T11-contrastive-loss-random-orthogonal | KMNIST | 0 | 0 | -0.017802119255065918 | 0.13025391101837158 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T6-sign-flipped-target | KMNIST | 0 | 0 | -0.40747570991516113 | -0.24316024780273438 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 0 | 0 | -0.39584803581237793 | -0.22707176208496094 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T5-random-matched-target | KMNIST | 1 | 0 | -0.17576074600219727 | -0.04084062576293945 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-SGD | KMNIST | 1 | 0 | -0.6432894468307495 | -0.5716111660003662 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T11-contrastive-loss-random-orthogonal | KMNIST | 2 | 0 | -0.11730289459228516 | -0.05659377574920654 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F3-T6-sign-flipped-target | KMNIST | 2 | 0 | -0.8433526754379272 | -0.786517858505249 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 2 | 0 | -0.8547724485397339 | -0.7958335876464844 |  |  | 0 | 0 |
| v2101_f18_contrastive_target_smoke | D-FOU | F18-T12-b1-contrastive-loss-random-orthogonal | MNIST | 0 | 0 | -0.16226720809936523 | -0.014464139938354492 |  |  | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `loss_orthogonal_random` target kind，并新增 `M64-ContrastiveLossRandomOrthogonalTargetFU` 与 `M65-B1ContrastiveLossRandomOrthogonalTransferFU`。
- 修改 `experiments/run_v17_common.py`：把 M64/M65 加入 function-space target 训练分支。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F18 specs 并加入 v21.01 target scope 白名单。
- 新增 `experiments/run_v21_01_f18_contrastive_target_summary.py`，只从落盘 matrix 汇总 F18 结果并更新 route/manifest/复盘。

### F18 结论 / Insight

- F18 不是 breakthrough：没有形成 candidate productive h3200 group。
- 如果 smoke 阶段 `full escalation recommended=0`，说明新 target 连 h800/h1600 grouped early-chain 都没有稳定打开，不能升级 full。
- 如果 smoke 阶段出现 early-chain 但 full 后 productive h3200 仍为 0，则该 target 仍只是局部/短期信号，不能替代 retained source。
- 当前仍不能把 late rebound、single-row positive 或 high ActuationR2 写成 promotion；`promotion_allowed` 保持 0。

## 2026-06-04 F19 Top-Wrong Margin Target

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F19MarginTargetSmokeNoEarlySignal`
- promotion_allowed: 0
- F19 rows / blocked rows: 90 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- control early-chain groups: 0
- full escalation recommended: 0

F19 是 F18 失败后的新 target 语义：不再推动整行 loss-cotangent，而只推动正确类与当前最强错误类之间的 margin，测试更稀疏、更分类相关的 target 是否能减少随机等价 late drift。

### F19 Group Evidence

| run_label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h6400 | early chain | productive h3200 | late rows |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f19_margin_target_smoke | D-FOU | CTRL-AdamW | 9 | 0 | -0.019990212387508817 | -0.029322776529524062 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9282578229904175 | -0.8520674705505371 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9262769884533353 | -0.8500887817806668 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.9207724067899916 | -0.8435791333516439 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T13-topwrong-margin-target | 9 | 0 | -0.1761088106367323 | -0.12356552812788221 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T14-b1-topwrong-margin-transfer | 9 | 0 | -0.18595778942108154 | 0.06658314996295506 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2527148061328464 | -0.0570918255382114 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T6-sign-flipped-target | 9 | 0 | -0.9338163137435913 | -0.8614021142323812 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T7-corrupted-label-target | 9 | 0 | -0.49592339992523193 | -0.20482831531100804 |  |  | 0 | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F9-TCTRL-stable-random-target | 9 | 0 | -0.9318630562888252 | -0.8837844530741373 |  |  | 0 | 0 | 0 |

### F19 Row Examples

| run_label | carrier | v21_id | dataset | seed | offset | h800 | h1600 | h3200 | h6400 | retained | late |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f19_margin_target_smoke | D-FOU | F19-T13-topwrong-margin-target | MNIST | 0 | 0 | 0.002470850944519043 | 0.12155020236968994 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T6-sign-flipped-target | MNIST | 0 | 0 | -0.9913449287414551 | -0.9697138071060181 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 0 | 0 | -0.9889907836914062 | -0.9639567136764526 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 1 | 0 | -0.455005407333374 | -0.25089704990386963 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-SGD | MNIST | 1 | 0 | -0.8548134565353394 | -0.8008289337158203 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T13-topwrong-margin-target | MNIST | 2 | 0 | -0.10312163829803467 | -0.07588815689086914 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T6-sign-flipped-target | MNIST | 2 | 0 | -1.0821428298950195 | -1.0443769693374634 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 2 | 0 | -1.0800020694732666 | -1.0419865846633911 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T5-random-matched-target | Fashion-MNIST | 0 | 0 | -0.2529538869857788 | -0.030997872352600098 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 0 | 0 | -1.3939650058746338 | -1.3386546969413757 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T13-topwrong-margin-target | Fashion-MNIST | 1 | 0 | -0.4399256706237793 | -0.3726453185081482 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T6-sign-flipped-target | Fashion-MNIST | 1 | 0 | -1.4483168125152588 | -1.4119346737861633 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | 0 | -1.4399449825286865 | -1.4019303917884827 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T5-random-matched-target | Fashion-MNIST | 2 | 0 | -0.03349876403808594 | 0.14506042003631592 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 2 | 0 | -0.6784794330596924 | -0.5039129257202148 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T13-topwrong-margin-target | KMNIST | 0 | 0 | 0.018166065216064453 | 0.12258410453796387 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T6-sign-flipped-target | KMNIST | 0 | 0 | -0.40747570991516113 | -0.24316024780273438 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 0 | 0 | -0.39584803581237793 | -0.22707176208496094 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T5-random-matched-target | KMNIST | 1 | 0 | -0.17576074600219727 | -0.04084062576293945 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-SGD | KMNIST | 1 | 0 | -0.6432894468307495 | -0.5716111660003662 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T13-topwrong-margin-target | KMNIST | 2 | 0 | -0.17556726932525635 | -0.0762944221496582 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F3-T6-sign-flipped-target | KMNIST | 2 | 0 | -0.8433526754379272 | -0.786517858505249 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 2 | 0 | -0.8547724485397339 | -0.7958335876464844 |  |  | 0 | 0 |
| v2101_f19_margin_target_smoke | D-FOU | F19-T14-b1-topwrong-margin-transfer | MNIST | 0 | 0 | -0.13477277755737305 | 0.06778609752655029 |  |  | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `top_wrong_margin` target kind，并新增 `M66-TopWrongMarginTargetFU` 与 `M67-B1TopWrongMarginTransferFU`。
- 修改 `experiments/run_v17_common.py`：把 M66/M67 加入 function-space target 训练分支。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F19 specs 并加入 v21.01 target scope 白名单。
- 新增 `experiments/run_v21_01_f19_margin_target_summary.py`，只从落盘 matrix 汇总 F19 结果并更新 route/manifest/复盘。

### F19 结论 / Insight

- F19 不是 breakthrough：没有形成 candidate productive h3200 group。
- 如果 `full escalation recommended=0`，说明 margin target 没能稳定打开 h800/h1600 grouped early-chain，不能升级 full。
- 如果 smoke 有 early-chain 但 full 后 productive h3200 仍为 0，则说明 margin target 仍不能解决 retained source dynamics。
- 当前仍不能把 single-row positive 或 late rebound 写成 promotion；`promotion_allowed` 保持 0。

## 2026-06-04 F20 Margin Split-Consensus Source-Channel Estimator

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F20MarginSourceSmokeNoEarlySignal`
- promotion_allowed: 0
- F20 rows / blocked rows: 90 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- control early-chain groups: 0
- full escalation recommended: 0

F20 是 F18/F19 target 语义失败后的 source-channel estimator redesign：不再直接拟合 function-space target，而是用 train split A/B 的 top-wrong margin 梯度一致性形成 slow source state，并用 corrupted-label train batch 做 gate。它不使用 validation/test/future/query 生成方向，也不是继续调 actuation rank、alt period 或 target threshold。

### F20 Group Evidence

| run_label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h6400 | early chain | productive h3200 | gate accepts | consensus density | signal gain | corrupt gain |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-AdamW | 9 | 0 | -0.041648553477393255 | -0.05499368243747287 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.948963893784417 | -0.8834070629543729 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9475154479344686 | -0.8819580872853597 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-SGD | 9 | 0 | -0.9496894280115763 | -0.8825243049197726 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_kan | D-FOU | KSW4-margin-split-consensus-source | 9 | 0 | -0.9456017679638333 | -0.8786419232686361 |  |  | 0 | 0 | 18 | 0.6916447745429145 | 3.8991371790568036e-07 | -7.996956507364909e-07 |
| v2101_f20_margin_source_smoke_mlp | MLP | CTRL-AdamW | 9 | 0 | -0.40112125873565674 | -0.8088014258278741 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_mlp | MLP | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.6411108573277792 | -0.8103717830446031 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_mlp | MLP | CTRL-RandomMatchedNorm | 9 | 0 | -0.6460719241036309 | -0.8154104948043823 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_mlp | MLP | CTRL-SGD | 9 | 0 | -0.2070104546017117 | -0.10651348696814643 |  |  | 0 | 0 | 0 |  |  |  |
| v2101_f20_margin_source_smoke_mlp | MLP | MLP-F20-margin-split-consensus-source | 9 | 0 | -0.1680133475197686 | -0.05853323141733805 |  |  | 0 | 0 | 18 | 0.5498866041501363 | 1.2094817874539229e-05 | -2.436257070965237e-05 |

### F20 Row Examples

| run_label | carrier | v21_id | dataset | seed | offset | h800 | h1600 | h3200 | h6400 | gate h800 | gate h1600 | cos h1600 | retained | late |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f20_margin_source_smoke_kan | D-FOU | KSW4-margin-split-consensus-source | MNIST | 0 | 0 | -0.9866158962249756 | -0.9602142572402954 |  |  | 1 | 1 | 0.3927437961101532 | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-NoOpMatchedOverhead | MNIST | 0 | 0 | -0.9864327907562256 | -0.9613910913467407 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-RandomMatchedNorm | MNIST | 1 | 0 | -0.8537555932998657 | -0.7955694198608398 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-AdamW | MNIST | 2 | 0 | 0.0 | 0.0 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-SGD | Fashion-MNIST | 0 | 0 | -1.3992583751678467 | -1.348380982875824 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | KSW4-margin-split-consensus-source | Fashion-MNIST | 1 | 0 | -1.4293606281280518 | -1.3884499669075012 |  |  | 1 | 1 | 0.29718753695487976 | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | 0 | -1.433408498764038 | -1.3953881859779358 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | 0 | -0.6925840377807617 | -0.5229442119598389 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-AdamW | KMNIST | 0 | 0 | -0.05977678298950195 | -0.09481978416442871 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-SGD | KMNIST | 1 | 0 | -0.7233554124832153 | -0.6739789247512817 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | KSW4-margin-split-consensus-source | KMNIST | 2 | 0 | -0.8865536451339722 | -0.858218789100647 |  |  | 1 | 1 | 0.538648247718811 | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-NoOpMatchedOverhead | KMNIST | 2 | 0 | -0.8953331708908081 | -0.8696945905685425 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-SGD | MNIST | 0 | 0 | -0.9870824813842773 | -0.9608522653579712 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | KSW4-margin-split-consensus-source | MNIST | 1 | 0 | -0.85645592212677 | -0.80216383934021 |  |  | 1 | 1 | 0.21392108500003815 | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-NoOpMatchedOverhead | MNIST | 1 | 0 | -0.8595198392868042 | -0.8013224601745605 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-RandomMatchedNorm | MNIST | 2 | 0 | -1.157815933227539 | -1.1551481485366821 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-AdamW | Fashion-MNIST | 0 | 0 | -0.011506795883178711 | 0.0 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-SGD | Fashion-MNIST | 1 | 0 | -1.43674635887146 | -1.3926756978034973 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | KSW4-margin-split-consensus-source | Fashion-MNIST | 2 | 0 | -0.6830475330352783 | -0.5054600238800049 |  |  | 1 | 1 | 0.5919672846794128 | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | 0 | -0.6917769908905029 | -0.5221450328826904 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 0 | 0 | -0.39130377769470215 | -0.22252726554870605 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-AdamW | KMNIST | 1 | 0 | 0.0 | 0.0 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-SGD | KMNIST | 2 | 0 | -0.906086802482605 | -0.8750079870223999 |  |  |  |  |  | 0 | 0 |
| v2101_f20_margin_source_smoke_kan | D-FOU | CTRL-AdamW | MNIST | 0 | 0 | -0.07227754592895508 | -0.1162717342376709 |  |  |  |  |  | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M68-MarginSplitConsensusSlowFU` semantic contract 与 source 描述。
- 修改 `experiments/run_v17_common.py`：新增 M68 训练分支，用 top-wrong margin split-consensus 梯度构造 slow source state，并记录 consensus density、signal/corrupt gain、corrupt cosine 与 gate accept。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `MLP-F20-margin-split-consensus-source` 与 `KSW4-margin-split-consensus-source` specs。
- 新增 `experiments/run_v21_01_f20_margin_source_summary.py`，只从落盘 matrix 汇总 F20 结果并更新 route/manifest/复盘。

### F20 结论 / Insight

- F20 不是 breakthrough：没有形成 candidate productive h3200 group。
- 如果 `full escalation recommended=0`，说明 source-channel estimator 在 smoke 阶段没有形成 h800/h1600 grouped early-chain，不能升级 full。
- 如果 smoke 有 early-chain 但 full 后 productive h3200 仍为 0，则说明 margin split-consensus 仍没有解决 source retention dynamics。
- 当前仍不能把 gate accepts、single-row positive 或 late rebound 写成 promotion；`promotion_allowed` 保持 0。

## 2026-06-04 F21 Optimizer-Dynamics Decoupling

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F21OptimizerDynamicsNoRetained`
- promotion_allowed: 0
- F21 rows / blocked rows: 81 / 0
- candidate productive h3200 groups: 0
- candidate productive h4800 groups: 0
- AdamW washout hypothesis supported: 0
- all optimizer conditions failed retained source: 1

F21 按计划 13.3 补做 optimizer-dynamics decoupling：把 AdamW-primary+FU residual、momentum-primary FU、SGD/LineC FU、dual-memory source state、schedule-free source iterate 与 matched controls 放在同一个 fresh h6400 matrix 中。它不是新 target，也不是调 scale；目的是回答 AdamW 是否是 h1600->h3200 washout 主因。

### F21 Group Evidence

| condition | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | productive h4800 | washout rows | late rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW control | CTRL-AdamW | 9 | 0 | -0.7136350340313382 | -1.118668966823154 | -1.5491891702016194 | -1.741767750846015 | -1.868686040242513 | 0 | 0 | 0 | 0 | 0 |
| NoOp matched overhead | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.7023795710669624 | -0.8570738633473715 | -0.9971845547358195 | -0.9825587802463107 | -0.9264096021652222 | 0 | 0 | 0 | 0 | 0 |
| Random matched norm | CTRL-RandomMatchedNorm | 9 | 0 | -0.7214462492201064 | -0.8751431571112739 | -1.0149736801783245 | -1.001124382019043 | -0.9441756539874606 | 0 | 0 | 0 | 0 | 0 |
| SGD control | CTRL-SGD | 9 | 0 | -0.21048650476667616 | -0.0699147383371989 | -0.037272625499301486 | -0.06951710912916395 | -0.08986344602372912 | 0 | 0 | 0 | 0 | 0 |
| O2 Momentum-primary + FU | MLP-F1-M2-strong-source | 9 | 0 | 0.037572900454203285 | -0.41555605994330513 | -0.6971066262986925 | -0.6865415838029649 | -0.6122481293148465 | 0 | 0 | 0 | 6 | 0 |
| O3 Dual-memory source state | MLP-F11-dual-timescale-retention-warm1200 | 9 | 0 | 0.01897267500559489 | -0.27998318937089706 | -0.4201384385426839 | -0.4055559105343289 | -0.34944400522443986 | 0 | 0 | 0 | 6 | 0 |
| O3 Schedule-free source iterate | MLP-F12-schedule-free-source-iterate | 9 | 0 | -0.6317545043097602 | -0.7635786533355713 | -0.8597420189115736 | -0.8019158707724677 | -0.7033373382356432 | 0 | 0 | 0 | 0 | 0 |
| O1 SGD/LineC-primary + FU | MLP-F2-M15-weak-stable | 9 | 0 | -0.2404869794845581 | -0.1740645037757026 | -0.14474705855051676 | -0.15023744106292725 | -0.18469402525160047 | 0 | 0 | 0 | 1 | 1 |
| O0 AdamW-primary + FU residual | MLP-F21-adamw-primary-fu-residual | 9 | 0 | -0.6605676809946696 | -1.084807448916965 | -1.5276222361458673 | -1.726245231098599 | -1.8584550354215834 | 0 | 0 | 0 | 2 | 0 |

### F21 Row Examples

| v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP-F21-adamw-primary-fu-residual | MNIST | 0 | -0.6461576223373413 | -0.9763311147689819 | -1.393978476524353 | -1.6490424871444702 | -1.8013828992843628 | 0 | 0 | 0 |
| MLP-F12-schedule-free-source-iterate | MNIST | 0 | -0.6939626932144165 | -0.8052557706832886 | -0.9643124341964722 | -1.0160046815872192 | -0.9820653200149536 | 0 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | MNIST | 0 | -0.758696436882019 | -0.8927608728408813 | -1.0931357145309448 | -1.1828914880752563 | -1.1853867769241333 | 0 | 0 | 0 |
| MLP-F11-dual-timescale-retention-warm1200 | MNIST | 1 | 0.38887929916381836 | 0.18560099601745605 | -0.047655701637268066 | -0.13140332698822021 | -0.11345255374908447 | 1 | 0 | 0 |
| CTRL-RandomMatchedNorm | MNIST | 1 | -0.43124985694885254 | -0.6237918138504028 | -0.8581250905990601 | -0.9405784606933594 | -0.920831561088562 | 0 | 0 | 0 |
| MLP-F2-M15-weak-stable | MNIST | 2 | -0.4509925842285156 | -0.028869986534118652 | -0.0025789737701416016 | 0.0016052722930908203 | -0.0008342266082763672 | 0 | 0 | 0 |
| CTRL-AdamW | MNIST | 2 | -0.31723642349243164 | -0.3239154815673828 | -0.8483721017837524 | -1.0039926767349243 | -1.0880036354064941 | 0 | 0 | 0 |
| MLP-F1-M2-strong-source | Fashion-MNIST | 0 | -0.22949588298797607 | -0.8623529672622681 | -1.0876377820968628 | -0.8838920593261719 | -0.6829012632369995 | 0 | 0 | 0 |
| CTRL-SGD | Fashion-MNIST | 0 | -0.14172732830047607 | -0.1684269905090332 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |
| MLP-F21-adamw-primary-fu-residual | Fashion-MNIST | 1 | -0.15034794807434082 | -0.667914867401123 | -1.0199662446975708 | -1.18571138381958 | -1.3269355297088623 | 0 | 0 | 0 |
| MLP-F12-schedule-free-source-iterate | Fashion-MNIST | 1 | -0.9199568033218384 | -1.2051554918289185 | -1.2757127285003662 | -1.2034940719604492 | -1.1210415363311768 | 0 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -1.1123117208480835 | -1.427756428718567 | -1.5600438117980957 | -1.5541133880615234 | -1.5437493324279785 | 0 | 0 | 0 |
| MLP-F11-dual-timescale-retention-warm1200 | Fashion-MNIST | 2 | -0.5370582342147827 | -1.4666997194290161 | -1.373548150062561 | -1.0708836317062378 | -0.8179426193237305 | 0 | 0 | 0 |
| CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | -0.7342954874038696 | -1.034563660621643 | -0.9396730661392212 | -0.6350904703140259 | -0.3818321228027344 | 0 | 0 | 0 |
| MLP-F2-M15-weak-stable | KMNIST | 0 | -0.11267709732055664 | 0.013243436813354492 | 0.048634886741638184 | -0.014153003692626953 | -0.051241397857666016 | 0 | 0 | 0 |
| CTRL-AdamW | KMNIST | 0 | -1.060314655303955 | -1.4972437620162964 | -2.023768663406372 | -2.31693172454834 | -2.5083950757980347 | 0 | 0 | 0 |
| MLP-F1-M2-strong-source | KMNIST | 1 | 0.6624417304992676 | 0.3189626932144165 | 0.0029371976852416992 | -0.048453569412231445 | -0.017284870147705078 | 1 | 0 | 0 |
| CTRL-SGD | KMNIST | 1 | -0.0951765775680542 | -0.01242828369140625 | 0.0 | 0.0 | -0.021003365516662598 | 0 | 0 | 0 |
| MLP-F21-adamw-primary-fu-residual | KMNIST | 2 | -0.5676507949829102 | -0.6800485849380493 | -0.8447655439376831 | -1.090301752090454 | -1.2512094974517822 | 0 | 0 | 0 |
| MLP-F12-schedule-free-source-iterate | KMNIST | 2 | -0.9384373426437378 | -0.8262344598770142 | -0.7176855802536011 | -0.7629809379577637 | -0.7459557056427002 | 0 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | KMNIST | 2 | -1.0716577768325806 | -0.9759539365768433 | -0.9003595113754272 | -0.9785075187683105 | -0.9940948486328125 | 0 | 0 | 0 |
| MLP-F1-M2-strong-source | MNIST | 0 | 0.13184332847595215 | -0.07693850994110107 | -0.31016218662261963 | -0.3781646490097046 | -0.35262858867645264 | 1 | 0 | 0 |
| CTRL-SGD | MNIST | 0 | -0.18591547012329102 | 0.0 | -0.01946854591369629 | -0.13695263862609863 | -0.16741466522216797 | 0 | 0 | 0 |
| MLP-F21-adamw-primary-fu-residual | MNIST | 1 | 0.0695270299911499 | -0.257607102394104 | -0.6377052068710327 | -0.8258285522460938 | -0.9079576730728149 | 1 | 0 | 0 |
| MLP-F12-schedule-free-source-iterate | MNIST | 1 | -0.21719622611999512 | -0.39816009998321533 | -0.6079531908035278 | -0.6688382625579834 | -0.6275376081466675 | 0 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | MNIST | 1 | -0.3097858428955078 | -0.5030626058578491 | -0.7363101243972778 | -0.8200483322143555 | -0.8020874261856079 | 0 | 0 | 0 |
| MLP-F11-dual-timescale-retention-warm1200 | MNIST | 2 | 0.02315366268157959 | 0.061678171157836914 | -0.31553876399993896 | -0.361527681350708 | -0.34833264350891113 | 1 | 0 | 0 |
| CTRL-RandomMatchedNorm | MNIST | 2 | -0.9468214511871338 | -0.8243364095687866 | -1.2020312547683716 | -1.2488340139389038 | -1.2351410388946533 | 0 | 0 | 0 |
| MLP-F2-M15-weak-stable | Fashion-MNIST | 0 | -0.3586149215698242 | -0.32794904708862305 | -0.255540132522583 | -0.20386672019958496 | -0.15315032005310059 | 0 | 0 | 0 |
| CTRL-AdamW | Fashion-MNIST | 0 | -0.6974531412124634 | -1.2458873987197876 | -1.588335394859314 | -1.589315414428711 | -1.6059643030166626 | 0 | 0 | 0 |
| MLP-F1-M2-strong-source | Fashion-MNIST | 1 | 0.21094799041748047 | -0.33028197288513184 | -0.6112122535705566 | -0.6365772485733032 | -0.6249918937683105 | 1 | 0 | 0 |
| CTRL-SGD | Fashion-MNIST | 1 | -0.2773411273956299 | -0.22114288806915283 | -0.28621017932891846 | -0.29438960552215576 | -0.31530535221099854 | 0 | 0 | 0 |
| MLP-F21-adamw-primary-fu-residual | Fashion-MNIST | 2 | -3.3690332174301147 | -4.489503264427185 | -5.302874684333801 | -5.585804343223572 | -5.871908903121948 | 0 | 0 | 0 |
| MLP-F12-schedule-free-source-iterate | Fashion-MNIST | 2 | -0.715359091758728 | -0.9920722246170044 | -0.8546181917190552 | -0.5087558031082153 | -0.2128458023071289 | 0 | 0 | 0 |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | -0.7091072797775269 | -1.0098694562911987 | -0.9164491891860962 | -0.6135109663009644 | -0.36033082008361816 | 0 | 0 | 0 |
| MLP-F11-dual-timescale-retention-warm1200 | KMNIST | 0 | 0.15700984001159668 | -0.03605306148529053 | -0.20039081573486328 | -0.2418992519378662 | -0.20875823497772217 | 1 | 0 | 0 |

### 修改记录

- 修改 `experiments/run_v21_common.py`：新增 `MLP-F21-adamw-primary-fu-residual`，机制为 `M1-AdamWPrimaryFUResidual`，用于 F2/O0 AdamW-primary+FU 对照。
- 修改 `experiments/run_v21_01_source_retention.py`：把 F21 spec 纳入 `mlp` 与 `optimizer` scope 白名单。
- 新增 `experiments/run_v21_01_f21_optimizer_dynamics_summary.py`，只从 fresh `v2101_f21_` rows 汇总 optimizer condition evidence、更新 route/manifest/复盘。

### F21 结论 / Insight

- F21 没有支持“只要去掉 AdamW 就能 retained source”：所有候选 optimizer condition 都没有形成 productive h3200/h4800 group。
- 如果 AdamW-primary 为负，而 SGD/Momentum/ScheduleFree 任一 condition 形成 retained source，才可支持 AdamW washout hypothesis；本轮没有出现这个证据。
- 因此当前 blocker 更像 source/target observability 本身不足，而不是单纯 AdamW 长期洗掉 FU source。
- 当前仍不能写 breakthrough 或 promotion；`promotion_allowed` 保持 0。

## 2026-06-04 F22 Rotated/Cautious Matrix Source-State

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F22RotatedCautiousSmokeNoEarlyChain`
- promotion_allowed: 0
- F22 rows / blocked rows: 54 / 0
- F22 smoke rows / full rows: 54 / 0
- F22 candidate early-chain groups: 0
- F22 candidate productive h3200 groups: 0
- full escalation recommended: 0

F22 是 F21 optimizer-dynamics decoupling 失败后的 F1-H/F1-I 实现：用每个矩阵参数的 low-rank SVD 分量维护 slow source state，并对与当前 low-rank 分量冲突的坐标做 cautious downweight，而不是 hard mask。方向仍只来自 train gradients；validation/source readback 只用于实验评估。

### F22 Group Evidence

| condition | run label | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate accepts h1600 | current cos h1600 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead | v2101_f22_rotated_cautious_smoke | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.7004084322187636 | -0.8656324677997165 |  |  |  | 0 | 0 | 0 |  |
| Random matched norm | v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | 9 | 0 | -0.7059643003675673 | -0.8704216082890829 |  |  |  | 0 | 0 | 0 |  |
| SGD control | v2101_f22_rotated_cautious_smoke | CTRL-SGD | 9 | 0 | -0.2061484389834934 | -0.06016712718539768 |  |  |  | 0 | 0 | 0 |  |
| M2 momentum early-source reference | v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | 9 | 0 | 0.013181037373012967 | -0.38670094807942706 |  |  |  | 0 | 0 | 0 |  |
| F22 rotated/cautious matrix slow state | v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | 9 | 0 | 0.12980402178234524 | -0.26329288217756486 |  |  |  | 0 | 0 | 9 | 0.4141755137178633 |
| M46 prior matrix-block retention reference | v2101_f22_rotated_cautious_smoke | MLP-F7-M2-source-with-matrix-block-retention | 9 | 0 | 0.009699808226691352 | -0.40365710523393417 |  |  |  | 0 | 0 | 9 | 0.41721051434675854 |

### F22 Row Examples

| run label | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | MNIST | 0 | 0.20881390571594238 | -0.06847226619720459 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | MNIST | 0 | -0.7156742811203003 | -0.8506065607070923 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | MNIST | 1 | 0.29398393630981445 | -0.07132172584533691 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | MNIST | 2 | 0.021789193153381348 | 0.06459236145019531 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | MNIST | 2 | -0.9239120483398438 | -0.8073824644088745 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | Fashion-MNIST | 0 | -0.07306098937988281 | -0.6153857707977295 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | Fashion-MNIST | 1 | 0.0711737871170044 | -0.20831537246704102 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.0091779232025146 | -1.2248514890670776 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | Fashion-MNIST | 2 | -0.26326489448547363 | -1.624883770942688 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | KMNIST | 0 | 0.0495455265045166 | -0.34698736667633057 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | KMNIST | 0 | -0.40017950534820557 | -0.7064211368560791 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | KMNIST | 1 | 0.4108271598815918 | -0.08529472351074219 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | KMNIST | 2 | -0.1154019832611084 | -0.16290318965911865 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | KMNIST | 2 | -0.9964514970779419 | -0.9000040292739868 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F7-M2-source-with-matrix-block-retention | MNIST | 0 | 0.11026453971862793 | -0.07056665420532227 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-NoOpMatchedOverhead | MNIST | 0 | -0.6826800107955933 | -0.8167444467544556 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-SGD | MNIST | 1 | -0.0077211856842041016 | -0.008277177810668945 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F7-M2-source-with-matrix-block-retention | MNIST | 2 | 0.22060155868530273 | 0.2344965934753418 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-NoOpMatchedOverhead | MNIST | 2 | -0.8536853790283203 | -0.7378588914871216 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-SGD | Fashion-MNIST | 0 | -0.11149537563323975 | -0.0782616138458252 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F7-M2-source-with-matrix-block-retention | Fashion-MNIST | 1 | 0.10986244678497314 | -0.28013134002685547 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -1.12729811668396 | -1.3431862592697144 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-SGD | Fashion-MNIST | 2 | -0.1223604679107666 | -0.07879757881164551 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F7-M2-source-with-matrix-block-retention | KMNIST | 0 | 0.012400150299072266 | -0.3746662139892578 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-NoOpMatchedOverhead | KMNIST | 0 | -0.4326823949813843 | -0.7390868663787842 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-SGD | KMNIST | 1 | -0.09028446674346924 | -0.0701366662979126 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F7-M2-source-with-matrix-block-retention | KMNIST | 2 | -0.5834985971450806 | -0.6903990507125854 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-NoOpMatchedOverhead | KMNIST | 2 | -0.9876559972763062 | -0.8919521570205688 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | MNIST | 0 | 0.15181052684783936 | 0.06515192985534668 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | MNIST | 1 | 0.32141220569610596 | 0.019496917724609375 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | MNIST | 1 | -0.31577038764953613 | -0.5061677694320679 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | MNIST | 2 | 0.06185352802276611 | 0.03600049018859863 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | Fashion-MNIST | 0 | -0.4339030981063843 | -0.9245270490646362 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | CTRL-RandomMatchedNorm | Fashion-MNIST | 0 | -0.911617636680603 | -1.2156740427017212 |  |  |  | 0 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F22-rotated-cautious-matrix-source | Fashion-MNIST | 1 | 0.2548688054084778 | -0.056622862815856934 |  |  |  | 1 | 0 | 0 |
| v2101_f22_rotated_cautious_smoke | MLP-F1-M2-strong-source | Fashion-MNIST | 2 | -0.6080292463302612 | -2.053423762321472 |  |  |  | 0 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M69-RotatedCautiousMatrixSlowFU`，first-step/audit update 使用 low-rank matrix SVD slow state，并声明 matrix-block semantic contract。
- 修改 `experiments/run_v17_common.py`：新增 M69 训练分支；先保留 momentum primary early source，再维护 low-rank matrix slow state，对冲突分量做 cautious downweight，并记录 gate/cos/density/gain 诊断。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `MLP-F22-rotated-cautious-matrix-source` spec 和 v21.01 白名单。
- 新增 `experiments/run_v21_01_f22_rotated_cautious_summary.py`，只从 fresh `v2101_f22_` rows 汇总 F22 evidence、更新 route/manifest/复盘。

### F22 结论 / Insight

- F22 不是 breakthrough：没有形成 productive h3200 group。
- 如果 `full escalation recommended=0`，说明 rotated/cautious matrix state 连 h800/h1600 grouped early-chain 都没有打开，不能升级 full。
- 如果 full rows > 0 但 productive h3200 仍为 0，说明 matrix-rotated cautious state 仍不能把 early source 转成 retained source。
- 当前仍不能写 promotion；`promotion_allowed` 保持 0。

## 2026-06-04 F23 Train-Only Source-vs-Gradient Lookahead Gate

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F23LookaheadGateSmokeNoEarlyChain`
- promotion_allowed: 0
- F23 rows / blocked rows: 54 / 0
- F23 smoke rows / full rows: 54 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- full escalation recommended: 0

F23 是 F22 后的 train-only selector 修复：candidate source residual 必须在 train split A/B lookahead 上至少不输同范数 gradient residual，并且不能更明显帮助 corrupted-label split，才允许提交。它不使用 validation/test/future/query 来构造方向。

### F23 Group Evidence

| condition | run label | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | lookahead advantage h1600 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead | v2101_f23_lookahead_gate_smoke | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.7004084322187636 | -0.8656324677997165 |  |  |  | 0 | 0 | 0 | 0 |  |
| Random matched norm | v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | 9 | 0 | -0.7059643003675673 | -0.8704216082890829 |  |  |  | 0 | 0 | 0 | 0 |  |
| SGD control | v2101_f23_lookahead_gate_smoke | CTRL-SGD | 9 | 0 | -0.2061484389834934 | -0.06016712718539768 |  |  |  | 0 | 0 | 0 | 0 |  |
| M2 momentum reference | v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | 9 | 0 | 0.013181037373012967 | -0.38670094807942706 |  |  |  | 0 | 0 | 0 | 0 |  |
| F22 rotated/cautious reference | v2101_f23_lookahead_gate_smoke | MLP-F22-rotated-cautious-matrix-source | 9 | 0 | 0.008817553520202637 | -0.3959830602010091 |  |  |  | 0 | 0 | 9 | 9 |  |
| F23 source-vs-gradient lookahead gate | v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | 9 | 0 | 0.12773324383629692 | -0.2534742487801446 |  |  |  | 0 | 0 | 9 | 9 |  |

### F23 Row Examples

| run label | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | MNIST | 0 | 0.20881390571594238 | -0.06847226619720459 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | MNIST | 0 | -0.7156742811203003 | -0.8506065607070923 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | MNIST | 1 | 0.3303591012954712 | -0.035840392112731934 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | MNIST | 2 | 0.021789193153381348 | 0.06459236145019531 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | MNIST | 2 | -0.9239120483398438 | -0.8073824644088745 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | Fashion-MNIST | 0 | -0.11932229995727539 | -0.6392436027526855 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | Fashion-MNIST | 1 | 0.0711737871170044 | -0.20831537246704102 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.0091779232025146 | -1.2248514890670776 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | Fashion-MNIST | 2 | -0.23580431938171387 | -1.536716103553772 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | KMNIST | 0 | 0.0495455265045166 | -0.34698736667633057 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | KMNIST | 0 | -0.40017950534820557 | -0.7064211368560791 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | KMNIST | 1 | 0.4063680171966553 | -0.061347365379333496 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | KMNIST | 2 | -0.1154019832611084 | -0.16290318965911865 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | KMNIST | 2 | -0.9964514970779419 | -0.9000040292739868 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F22-rotated-cautious-matrix-source | MNIST | 0 | 0.1314840316772461 | -0.07774102687835693 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-NoOpMatchedOverhead | MNIST | 0 | -0.6826800107955933 | -0.8167444467544556 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-SGD | MNIST | 1 | -0.0077211856842041016 | -0.008277177810668945 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F22-rotated-cautious-matrix-source | MNIST | 2 | 0.22601830959320068 | 0.2366790771484375 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-NoOpMatchedOverhead | MNIST | 2 | -0.8536853790283203 | -0.7378588914871216 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-SGD | Fashion-MNIST | 0 | -0.11149537563323975 | -0.0782616138458252 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F22-rotated-cautious-matrix-source | Fashion-MNIST | 1 | 0.16614627838134766 | -0.18120861053466797 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -1.12729811668396 | -1.3431862592697144 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-SGD | Fashion-MNIST | 2 | -0.1223604679107666 | -0.07879757881164551 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F22-rotated-cautious-matrix-source | KMNIST | 0 | -0.01898336410522461 | -0.4123833179473877 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-NoOpMatchedOverhead | KMNIST | 0 | -0.4326823949813843 | -0.7390868663787842 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-SGD | KMNIST | 1 | -0.09028446674346924 | -0.0701366662979126 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F22-rotated-cautious-matrix-source | KMNIST | 2 | -0.5680506229400635 | -0.6465731859207153 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-NoOpMatchedOverhead | KMNIST | 2 | -0.9876559972763062 | -0.8919521570205688 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | MNIST | 0 | 0.16342103481292725 | 0.08361124992370605 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | MNIST | 1 | 0.32141220569610596 | 0.019496917724609375 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | MNIST | 1 | -0.31577038764953613 | -0.5061677694320679 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | MNIST | 2 | 0.08516919612884521 | 0.05715799331665039 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | Fashion-MNIST | 0 | -0.4339030981063843 | -0.9245270490646362 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | CTRL-RandomMatchedNorm | Fashion-MNIST | 0 | -0.911617636680603 | -1.2156740427017212 |  |  |  | 0 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F23-source-vs-sgd-lookahead-gate | Fashion-MNIST | 1 | 0.24150323867797852 | -0.06727755069732666 |  |  |  | 1 | 0 | 0 |
| v2101_f23_lookahead_gate_smoke | MLP-F1-M2-strong-source | Fashion-MNIST | 2 | -0.6080292463302612 | -2.053423762321472 |  |  |  | 0 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M70-TrainLookaheadCautiousSourceFU` semantic contract 与 audit update。
- 修改 `experiments/run_v17_common.py`：新增 M70 训练分支，用 train split A/B 与同范数 gradient residual 做在线 lookahead 对照，并记录 lookahead advantage、gate accept 与 source-state diagnostics。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `MLP-F23-source-vs-sgd-lookahead-gate` spec 和 v21.01 白名单。
- 新增 `experiments/run_v21_01_f23_lookahead_gate_summary.py`，只从 fresh `v2101_f23_` rows 汇总并更新 route/manifest/复盘。

### F23 结论 / Insight

- F23 不是 breakthrough：没有形成 productive h3200 group。
- 如果 full escalation recommended=0，说明在线 train-only source-vs-gradient selector 没能打开 grouped h800/h1600 early-chain。
- 如果 full rows > 0 但 productive h3200 仍为 0，则说明即使 train-only lookahead 能保早期 source，也不能稳定转成 retained source。
- 当前仍不能写 promotion；`promotion_allowed` 保持 0。

## 2026-06-04 最终审计边界

### 是否达成 v21.01 目标

没有达成。

- final route: `R2-LateReboundNoContinuousRetention-F23LookaheadGateSmokeNoEarlyChain`
- promotion_allowed: 0
- S0.6 preflight pass: 1
- required artifact missing count: 0
- manifest rows: 59
- final packet: `v21_01_code_review_packet.zip` size 5.2M
- final bundle: `v21_01_results_bundle.zip` size 11M

### 本轮追加尝试链

- F21 optimizer-dynamics decoupling：rows=81，blocked=0，candidate productive h3200/h4800 groups=0/0，AdamW washout hypothesis supported=0。
- F22 rotated/cautious matrix source-state：smoke rows=54，blocked=0，candidate early-chain groups=0，full escalation recommended=0。
- F23 train-only source-vs-gradient lookahead gate：smoke rows=54，blocked=0，candidate early-chain groups=0，full escalation recommended=0。

### 最终 insight

F21 关闭了“只要换掉 AdamW / optimizer dynamics 就能 retained”的简单解释；F22 说明 SOAP-like low-rank rotated/cautious matrix state 也没能把 h800 source 保到 h1600；F23 进一步说明在线 train-only source-vs-gradient lookahead selector 也没有形成 grouped early-chain。当前不能继续把问题描述成单一 optimizer overwrite、matrix coordinate mismatch 或 gate 不接收。

可信边界是：v21.01 的 kernel officialization / artifact debt 不是主 blocker，functional source retention 仍失败；继续推进需要新的 source observability / target-to-retention 理论，而不是继续对同族 M2/M46/M69/M70 调 scale、alt period 或 warmup。结束进程检查未发现持续运行的 v21.01 training 或 GPU monitor；`ps` 仅匹配检查命令自身。

## 2026-06-04 F24 B1 Consensus Lookahead-Gated Transfer

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F24B1LookaheadSmokeNoEarlyChain`
- promotion_allowed: 0
- F24 rows / blocked rows: 108 / 0
- F24 smoke rows / full rows: 108 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- full escalation recommended: 0

F24 是 target-to-retention 修复，不是继续 MLP matrix/optimizer 同族 sweep。它以 F10-T7 的 B1 cross-split consensus transfer 为候选 target，但每次提交前用 train split A/B 对比同范数 SGD/gradient residual；只有候选 target 在 train lookahead 上不明显输给 gradient 且没有明显帮助 corrupted-label split 时才提交，否则退回普通 SGD。

### F24 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | lookahead advantage h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead | v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0445110930336847 | -0.9038081765174866 |  |  |  | 0 | 0 | 0 | 0 |  |
| Random matched norm | v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.040397736761305 | -0.8997116155094571 |  |  |  | 0 | 0 | 0 | 0 |  |
| SGD control | v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0572091341018677 | -0.9318582018216451 |  |  |  | 0 | 0 | 0 | 0 |  |
| B1 consensus transfer reference | v2101_f24_b1_lookahead_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.3883764213985867 | 0.016976488961113825 |  |  |  | 0 | 0 | 0 | 0 |  |
| F24 B1 consensus lookahead gate | v2101_f24_b1_lookahead_smoke | D-CHE | F24-B1-consensus-lookahead-gated-transfer | 9 | 0 | -1.0501737991968791 | -0.9244767361217074 |  |  |  | 0 | 0 | 9 | 9 |  |
| random matched target control | v2101_f24_b1_lookahead_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.5505752431021796 | -0.08207249641418457 |  |  |  | 0 | 0 | 0 | 0 |  |
| NoOp matched overhead | v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9466580152511597 | -0.8811011844211154 |  |  |  | 0 | 0 | 0 | 0 |  |
| Random matched norm | v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9497320519553291 | -0.8841751151614718 |  |  |  | 0 | 0 | 0 | 0 |  |
| SGD control | v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.9468624459372627 | -0.8803887102339003 |  |  |  | 0 | 0 | 0 | 0 |  |
| B1 consensus transfer reference | v2101_f24_b1_lookahead_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.2595802280637953 | 0.015987296899159748 |  |  |  | 0 | 0 | 0 | 0 |  |
| F24 B1 consensus lookahead gate | v2101_f24_b1_lookahead_smoke | D-FOU | F24-B1-consensus-lookahead-gated-transfer | 9 | 0 | -0.9509785307778252 | -0.8839651478661431 |  |  |  | 0 | 0 | 9 | 9 |  |
| random matched target control | v2101_f24_b1_lookahead_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2588990396923489 | -0.019432445367177326 |  |  |  | 0 | 0 | 0 | 0 |  |

### F24 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f24_b1_lookahead_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | MNIST | 0 | -0.42167162895202637 | -0.10428714752197266 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 0 | -1.0832973718643188 | -0.9866167306900024 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F3-T5-random-matched-target | MNIST | 1 | -0.622395396232605 | -0.18562710285186768 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | MNIST | 2 | -0.5300658941268921 | -0.18342792987823486 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 2 | -1.2678548097610474 | -1.2380868196487427 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F3-T5-random-matched-target | Fashion-MNIST | 0 | -0.8363925814628601 | -0.30534863471984863 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 1 | -0.540497362613678 | -0.2266862392425537 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.610187590122223 | -1.5851511359214783 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F3-T5-random-matched-target | Fashion-MNIST | 2 | 0.06839752197265625 | 0.6275893449783325 |  |  |  | 1 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 0 | -0.19430029392242432 | 0.3334618806838989 |  |  |  | 1 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-RandomMatchedNorm | KMNIST | 0 | -0.5809676647186279 | -0.38844478130340576 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F3-T5-random-matched-target | KMNIST | 1 | -0.5217810869216919 | 0.038330078125 |  |  |  | 1 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 2 | -0.3850637674331665 | -0.0010924339294433594 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-RandomMatchedNorm | KMNIST | 2 | -1.0353271961212158 | -0.9322339296340942 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 0 | -0.21436083316802979 | 0.11906719207763672 |  |  |  | 1 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | MNIST | 1 | -0.4118525981903076 | -0.0680842399597168 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 1 | -0.8615459203720093 | -0.8033525943756104 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 2 | -0.4258764982223511 | -0.2970101833343506 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 0 | -0.5166939496994019 | -0.1347978711128235 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 0 | -1.3814918994903564 | -1.3318080306053162 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F3-T5-random-matched-target | Fashion-MNIST | 1 | -0.3150707483291626 | -0.050995707511901855 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 2 | -0.05188131332397461 | 0.1765270233154297 |  |  |  | 1 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | -0.7043540477752686 | -0.5346968173980713 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F3-T5-random-matched-target | KMNIST | 0 | -0.04006040096282959 | 0.3849993944168091 |  |  |  | 1 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 1 | -0.2914249897003174 | -0.047669053077697754 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 1 | -0.7157844305038452 | -0.6634169816970825 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F3-T5-random-matched-target | KMNIST | 2 | -0.27132999897003174 | -0.07087993621826172 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F24-B1-consensus-lookahead-gated-transfer | MNIST | 0 | -1.0863174200057983 | -1.0091270208358765 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 0 | -1.0615226030349731 | -0.9648040533065796 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-SGD | MNIST | 1 | -0.8508697748184204 | -0.7449877262115479 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F24-B1-consensus-lookahead-gated-transfer | MNIST | 2 | -1.2503820657730103 | -1.2146469354629517 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 2 | -1.2270070314407349 | -1.1972447633743286 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-SGD | Fashion-MNIST | 0 | -1.3786210417747498 | -1.2738736867904663 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F24-B1-consensus-lookahead-gated-transfer | Fashion-MNIST | 1 | -1.6245096325874329 | -1.6090325713157654 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -1.6158869862556458 | -1.5907639861106873 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-SGD | Fashion-MNIST | 2 | -0.7225826978683472 | -0.33296430110931396 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F24-B1-consensus-lookahead-gated-transfer | KMNIST | 0 | -0.5948944091796875 | -0.42440474033355713 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 0 | -0.5796687602996826 | -0.3870645761489868 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-SGD | KMNIST | 1 | -0.9224551916122437 | -0.7991310358047485 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | F24-B1-consensus-lookahead-gated-transfer | KMNIST | 2 | -1.0497119426727295 | -0.9474302530288696 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 2 | -1.0764522552490234 | -0.9734023809432983 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-SGD | MNIST | 0 | -0.9902966022491455 | -0.9649990797042847 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F24-B1-consensus-lookahead-gated-transfer | MNIST | 1 | -0.8623934984207153 | -0.8077752590179443 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-NoOpMatchedOverhead | MNIST | 1 | -0.8531190156936646 | -0.7949216365814209 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-SGD | MNIST | 2 | -1.144469976425171 | -1.1390506029129028 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | F24-B1-consensus-lookahead-gated-transfer | Fashion-MNIST | 0 | -1.39552903175354 | -1.3429288268089294 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 0 | -1.4050548076629639 | -1.3554009795188904 |  |  |  | 0 | 0 | 0 |
| v2101_f24_b1_lookahead_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 1 | -1.433319330215454 | -1.3916135430335999 |  |  |  | 0 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M71-TrainLookaheadB1ConsensusTransferFU`，方向来自 train-stream B1 consensus transfer target，并写入 semantic contract。
- 修改 `experiments/run_v17_common.py`：新增 M71 训练分支，用 train split A/B 与同范数 gradient residual 做在线 lookahead 对照，并用 corrupted-label split 做坏方向过滤；记录 gate accept、target/gradient lookahead gain、current cosine 与 target diagnostics。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `F24-B1-consensus-lookahead-gated-transfer` spec 与 v21.01 target 白名单。
- 新增 `experiments/run_v21_01_f24_b1_lookahead_summary.py`，只从 fresh `v2101_f24_` rows 汇总并更新 route/manifest/复盘。

### F24 结论 / Insight

- F24 没有形成 promotion-ready retained source；`promotion_allowed` 保持 0。
- 如果 candidate early-chain groups 为 0，说明 train-only target-vs-gradient selector 没有把 B1 late-rebound target 提前变成连续 h800/h1600 source。
- 如果 candidate h3200 groups 为 0，即便局部 rows 可能出现 late-positive 或 retained-like 信号，也不能通过 grouped 证据写 breakthrough。
- 当前 blocker 继续定位在 train-only source/target observability 与 target-to-retention dynamics；不能用 high ActuationR2 或 single-row positive 替代 retained chain。

## 2026-06-04 F25 Loss-Warm To B1 Consensus Migration

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F25MigrationSmokeNoEarlyChain`
- promotion_allowed: 0
- F25 rows / blocked rows: 126 / 0
- F25 smoke rows / full rows: 126 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- full escalation recommended: 0

F25 是 delayed migration 测试：前 800 step 的 FU commit 用 loss-cotangent target 尝试 early closure，之后迁移到 B1 cross-split consensus transfer。非 FU commit step 使用 SGD；不使用 validation/test/future/query 生成方向。

### F25 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | migration phase h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead | v2101_f25_migration_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0424859921137493 | -0.9017830755975511 |  |  |  | 0 | 0 | 0 | 0 |  |
| Random matched norm | v2101_f25_migration_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.0524432791603937 | -0.9117626150449117 |  |  |  | 0 | 0 | 0 | 0 |  |
| SGD control | v2101_f25_migration_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0452582703696356 | -0.9237990180651346 |  |  |  | 0 | 0 | 0 | 0 |  |
| B1 consensus migration reference | v2101_f25_migration_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.343934350543552 | 0.06247780058119032 |  |  |  | 0 | 0 | 0 | 0 |  |
| F25 loss-warm to B1 consensus migration | v2101_f25_migration_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.08756848838594225 | 0.06413359112209743 |  |  |  | 0 | 0 | 9 | 9 |  |
| loss-cotangent warmup reference | v2101_f25_migration_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.10001919004652235 | -0.0019819670253329808 |  |  |  | 0 | 0 | 0 | 0 |  |
| random matched target control | v2101_f25_migration_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.4615841971503364 | 0.013317339950137667 |  |  |  | 0 | 0 | 0 | 0 |  |
| NoOp matched overhead | v2101_f25_migration_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9474037355846829 | -0.8818469047546387 |  |  |  | 0 | 0 | 0 | 0 |  |
| Random matched norm | v2101_f25_migration_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9516729116439819 | -0.886117140452067 |  |  |  | 0 | 0 | 0 | 0 |  |
| SGD control | v2101_f25_migration_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.9435887734095255 | -0.8768064181009928 |  |  |  | 0 | 0 | 0 | 0 |  |
| B1 consensus migration reference | v2101_f25_migration_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.2737775643666585 | -0.01371704207526313 |  |  |  | 0 | 0 | 0 | 0 |  |
| F25 loss-warm to B1 consensus migration | v2101_f25_migration_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.13780654801262748 | -0.06563691298166911 |  |  |  | 0 | 0 | 9 | 9 |  |
| loss-cotangent warmup reference | v2101_f25_migration_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.10813199149237739 | -0.08508133888244629 |  |  |  | 0 | 0 | 0 | 0 |  |
| random matched target control | v2101_f25_migration_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.3230452405081855 | -0.06945261028077868 |  |  |  | 0 | 0 | 0 | 0 |  |

### F25 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f25_migration_smoke | D-CHE | F3-T1-loss-cotangent-target | MNIST | 0 | -0.17388534545898438 | -0.021915078163146973 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-SGD | MNIST | 0 | -1.0906251668930054 | -1.0151838064193726 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | MNIST | 1 | -0.3660318851470947 | 0.04047811031341553 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 1 | -0.794141411781311 | -0.6533653736114502 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | MNIST | 2 | -0.1910388469696045 | -0.10599184036254883 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 2 | -1.2584210634231567 | -1.2286587953567505 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T5-random-matched-target | Fashion-MNIST | 0 | -0.5116680264472961 | -0.07374656200408936 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T1-loss-cotangent-target | Fashion-MNIST | 1 | -0.2022133469581604 | -0.21721643209457397 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-SGD | Fashion-MNIST | 1 | -1.6176932454109192 | -1.6032180190086365 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 2 | -0.11263442039489746 | 0.5723283290863037 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | -0.7489258050918579 | -0.3407024145126343 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | KMNIST | 0 | 0.12027108669281006 | 0.3322310447692871 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 0 | -0.6059637069702148 | -0.41335952281951904 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T5-random-matched-target | KMNIST | 1 | -0.4899942874908447 | -0.02943873405456543 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T1-loss-cotangent-target | KMNIST | 2 | -0.19014060497283936 | -0.09723365306854248 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-SGD | KMNIST | 2 | -1.0502369403839111 | -0.9484730958938599 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | MNIST | 0 | -0.3592745065689087 | -0.10673749446868896 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 0 | -0.9878089427947998 | -0.9627701044082642 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | MNIST | 1 | -0.06854510307312012 | -0.003514885902404785 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-NoOpMatchedOverhead | MNIST | 1 | -0.8498085737228394 | -0.7916111946105957 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 2 | -0.5836602449417114 | -0.3680307865142822 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F3-T1-loss-cotangent-target | Fashion-MNIST | 0 | -0.18507206439971924 | -0.12083464860916138 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 0 | -1.3866298198699951 | -1.3355757594108582 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 1 | -0.3463444709777832 | -0.11817663908004761 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.4417741298675537 | -1.4037685990333557 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | Fashion-MNIST | 2 | -0.11919879913330078 | -0.03685140609741211 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | -0.6960229873657227 | -0.5263910293579102 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F3-T5-random-matched-target | KMNIST | 0 | -0.13440942764282227 | 0.11997199058532715 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F3-T1-loss-cotangent-target | KMNIST | 1 | -0.18892204761505127 | -0.16885530948638916 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-SGD | KMNIST | 1 | -0.7210439443588257 | -0.6716409921646118 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 2 | -0.24172449111938477 | 0.01424252986907959 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 2 | -0.9001432657241821 | -0.8745254278182983 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | MNIST | 0 | -0.2938110828399658 | 0.039618730545043945 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 0 | -1.0615867376327515 | -0.9648886919021606 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | MNIST | 1 | -0.003956198692321777 | 0.1774376630783081 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 1 | -0.809522271156311 | -0.6687462329864502 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T5-random-matched-target | MNIST | 2 | -0.7400363683700562 | -0.22409391403198242 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T1-loss-cotangent-target | Fashion-MNIST | 0 | -0.1619088053703308 | -0.11199235916137695 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-SGD | Fashion-MNIST | 0 | -1.3674864172935486 | -1.2696551084518433 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 1 | -0.5644077658653259 | -0.2207401990890503 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.6109398007392883 | -1.5859219431877136 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | Fashion-MNIST | 2 | -0.09443986415863037 | 0.14204084873199463 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | -0.7239609956741333 | -0.3156510591506958 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T5-random-matched-target | KMNIST | 0 | -0.3772928714752197 | 0.19124436378479004 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F3-T1-loss-cotangent-target | KMNIST | 1 | -0.30204737186431885 | -0.27738630771636963 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-SGD | KMNIST | 1 | -0.8988839387893677 | -0.7851747274398804 |  |  |  | 0 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 2 | -0.29411792755126953 | 0.013410210609436035 |  |  |  | 1 | 0 | 0 |
| v2101_f25_migration_smoke | D-CHE | CTRL-RandomMatchedNorm | KMNIST | 2 | -1.1148579120635986 | -1.0117846727371216 |  |  |  | 0 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M72-LossWarmToB1ConsensusMigrationFU` 与 semantic contract。
- 修改 `experiments/run_v17_common.py`：新增 M72 两阶段训练分支，warmup 阶段使用 `M49-LossCotangentTargetFU`，迁移阶段使用 `M60-B1CrossSplitConsensusTransferFU`，并记录 migration phase/gate/current-cos 诊断。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `F25-loss-warm-to-b1-consensus-migration` spec 和 v21.01 target 白名单。
- 新增 `experiments/run_v21_01_f25_migration_summary.py`，从 fresh `v2101_f25_` rows 汇总并更新 route/manifest/复盘。

### F25 结论 / Insight

- F25 只有在 grouped h800/h1600 同时为正时才允许升级 full；否则 smoke 只作为 delayed migration blocker。
- 如果 F25 仍无 early-chain，说明 loss-target early closure 与 B1 delayed target 不能简单串联成 retained source。
- 如果 controls 或 random matched target 同样出现 positive，则必须按 control-equivalent 处理，不能 promotion。

## 2026-06-04 F24-F25 最终审计边界

### 是否达成 v21.01 目标

没有达成。

- final route: `R2-LateReboundNoContinuousRetention-F25MigrationSmokeNoEarlyChain`
- promotion_allowed: 0
- S0.6 import_closure / mechanism_contracts after F25: 1 / 1
- required artifact missing count: 0
- manifest rows: 65
- final packet: `v21_01_code_review_packet.zip` size 6.8M
- final bundle: `v21_01_results_bundle.zip` size 14M
- end process check: no persistent v21.01 training / GPU monitor process

### 本轮新增修复与审计说明

- 修复 M71 诊断覆盖 bug：gate accept 后 `update.diagnostics` 会被通用 post-loop 覆盖掉 `last_actuation` 中的 gate/lookahead diagnostics。本轮改为在 accept 时把完整 `last_actuation` 写回 `gated_update.diagnostics`，然后重跑 F24 smoke。
- 修复 M71 commit 语义：最初 accept 后执行 `SGD primary + target residual`，但 F10-T7 reference 是 target commit 替代该 FU step 的 SGD。该语义会让 F24 被 SGD 主步主导。本轮改为 accept 时 target-only commit，reject 时 SGD fallback，并再次重跑 F24 smoke。
- 新增 M72/F25 delayed migration writer：前 800 step 使用 loss-cotangent target 试图 early closure，之后迁移到 B1 consensus transfer；非 FU commit step 使用 SGD。

### 关键结果

| phase | rows | blocked | D-CHE h800 | D-CHE h1600 | D-FOU h800 | D-FOU h1600 | early-chain groups | full escalation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F24 target-vs-gradient lookahead gate | 108 | 0 | -1.0501737991968791 | -0.9244767361217074 | -0.9509785307778252 | -0.8839651478661431 | 0 | 0 |
| F25 loss-warm to B1 migration | 126 | 0 | -0.08756848838594225 | 0.06413359112209743 | -0.13780654801262748 | -0.06563691298166911 | 0 | 0 |

F25 对 D-CHE 有实际改善：相对 F10-T7 的 D-CHE h800=-0.343934350543552，F25 提到 h800=-0.08756848838594225，h1600=0.06413359112209743。但 h800 仍为负，D-FOU 也未打开，因此不能升级 full，也不能写 retained source。

### 最终 insight

F24 说明 train-only target-vs-gradient lookahead gate 没有把 B1 late-rebound target 提前变成 h800/h1600 连续 source；即便 gate 在 h800/h1600 audit 点有 accept，grouped source 仍接近 SGD/NoOp 负值区间。F25 说明“loss target early closure + B1 delayed migration”能把 D-CHE 推近 early-chain，但还不能跨过 h800 threshold，也不能迁移到 D-FOU。

因此当前 blocker 进一步收窄为：现有 legal target/source theory 可以产生局部或 h1600 signal，但不能稳定地产生 grouped h800->h1600 连续链；继续对 M71/M72 做 scale/alt/warmup 小修审计价值低。下一步若继续，需要新的 train-only early-source observability 或更强的 source-channel target theory，而不是把 single-row positive 或 h1600-only positive 写成 breakthrough。

## 2026-06-04 F26-F33 Easy/Blended/Gated/B3-Null B1 Consensus Target

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F26EasyConsensusSmokeNoEarlyChain`
- promotion_allowed: 0
- F26-F33 rows / blocked rows: 702 / 0
- smoke rows / full rows: 702 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- full escalation recommended: 0

F26-F33 是新的 source/target theory 尝试，不是 M71/M72 的 scale/alt/warmup 小修。F26/F27 只用 train batch 内 low-loss / already-stable examples 生成 B1 split-consensus target；F28/F29 保留 loss-cotangent early closure，同时叠加 easy B1 consensus；F30/F31 增加 train-split B2 transfer precommit gate，只有 B2 gain 明确压过 B3 safety/noise 时才提交迁移 update；F32/F33 把 B3/safety split 作为 readout operator 的零位移约束，测试 late-rebound target 是否能被 source-channel/reservoir exclusion 提前成 h800/h1600 连续链。

### F26-F33 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate h800 | gate h1600 | easy b1 h800 | easy b2 h800 | B3-null rows h800 | B3-null rows h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead | v2101_f26_easy_consensus_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0427165428797405 | -0.9020136263635423 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_easy_consensus_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.0398643149269953 | -0.8991770545641581 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_easy_consensus_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0393751329845853 | -0.9190202885203891 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_easy_consensus_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.3519224723180135 | 0.06765468253029717 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_easy_consensus_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.09201386239793566 | 0.039422571659088135 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| easy-example B1 consensus transfer | v2101_f26_easy_consensus_smoke | D-CHE | F26-easy-b1-consensus-transfer | 9 | 0 | -1.5135107967588637 | -1.6488897071944342 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to easy B1 consensus migration | v2101_f26_easy_consensus_smoke | D-CHE | F27-loss-warm-to-easy-b1-consensus-migration | 9 | 0 | -0.058507290151384145 | 0.11211446921030681 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_easy_consensus_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.13140477074517143 | -0.035111831294165716 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_easy_consensus_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.500873843828837 | -0.01942558421028985 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| NoOp matched overhead | v2101_f26_easy_consensus_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.953698992729187 | -0.8881421618991427 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_easy_consensus_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9493180513381958 | -0.8837602403428819 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_easy_consensus_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.946563548511929 | -0.8795875708262125 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_easy_consensus_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.2807768185933431 | -0.04995621575249566 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_easy_consensus_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.16001799371507433 | -0.06499796443515354 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| easy-example B1 consensus transfer | v2101_f26_easy_consensus_smoke | D-FOU | F26-easy-b1-consensus-transfer | 9 | 0 | -1.666903058687846 | -1.909982787238227 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to easy B1 consensus migration | v2101_f26_easy_consensus_smoke | D-FOU | F27-loss-warm-to-easy-b1-consensus-migration | 9 | 0 | -0.059531827767690025 | 0.01231600840886434 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_easy_consensus_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.1014464298884074 | -0.06274944543838501 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_easy_consensus_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2130549218919542 | 0.021477739016215008 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| NoOp matched overhead | v2101_f26_f28_blend_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0542309151755438 | -0.9135279986593459 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_f28_blend_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.0387049913406372 | -0.898009306854672 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_f28_blend_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0477516253789265 | -0.9241670303874545 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_f28_blend_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.367335041364034 | 0.02658810218175252 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_f28_blend_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.09142195516162449 | 0.07362303468916151 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-warm to easy B1 consensus migration | v2101_f26_f28_blend_smoke | D-CHE | F27-loss-warm-to-easy-b1-consensus-migration | 9 | 0 | -0.0733358727561103 | 0.046553201145595975 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss + easy B1 consensus blend | v2101_f26_f28_blend_smoke | D-CHE | F28-loss-easy-b1-consensus-blend | 9 | 0 | -1.0761215819252863 | -1.2504825790723164 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to loss + easy B1 consensus blend | v2101_f26_f28_blend_smoke | D-CHE | F29-loss-warm-to-loss-easy-b1-consensus-blend | 9 | 0 | -0.07673159572813246 | 0.06967877017127143 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_f28_blend_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.08120625548892552 | 0.042479621039496526 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_f28_blend_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.5517797470092773 | -0.014498160945044624 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| NoOp matched overhead | v2101_f26_f28_blend_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9510703749126859 | -0.8855135440826416 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_f28_blend_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.950757622718811 | -0.8852050569322374 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_f28_blend_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.94707723458608 | -0.8803360462188721 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_f28_blend_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.20035911930931938 | 0.05153660641776191 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_f28_blend_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.08923818005455865 | -0.0075536370277404785 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-warm to easy B1 consensus migration | v2101_f26_f28_blend_smoke | D-FOU | F27-loss-warm-to-easy-b1-consensus-migration | 9 | 0 | -0.13979125022888184 | -0.06717842155032688 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss + easy B1 consensus blend | v2101_f26_f28_blend_smoke | D-FOU | F28-loss-easy-b1-consensus-blend | 9 | 0 | -1.5722297297583685 | -1.973734630478753 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to loss + easy B1 consensus blend | v2101_f26_f28_blend_smoke | D-FOU | F29-loss-warm-to-loss-easy-b1-consensus-blend | 9 | 0 | -0.16202743848164877 | -0.09170716338687473 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_f28_blend_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.07866508430904812 | -0.038517514864603676 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_f28_blend_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2982826232910156 | -0.019733395841386583 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| NoOp matched overhead | v2101_f26_f30_gated_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0542309151755438 | -0.9135279986593459 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_f30_gated_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.0387049913406372 | -0.898009306854672 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_f30_gated_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0477516253789265 | -0.9241670303874545 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_f30_gated_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.367335041364034 | 0.02658810218175252 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_f30_gated_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.09142195516162449 | 0.07362303468916151 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-warm to loss + easy B1 consensus blend | v2101_f26_f30_gated_smoke | D-CHE | F29-loss-warm-to-loss-easy-b1-consensus-blend | 9 | 0 | -0.0848848025004069 | 0.05341894096798367 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_f30_gated_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.08120625548892552 | 0.042479621039496526 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_f30_gated_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.5517797470092773 | -0.014498160945044624 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| gain-gated loss-warm to B1 consensus | v2101_f26_f30_gated_smoke | D-CHE | F30-gain-gated-loss-warm-b1-consensus | 9 | 0 | -0.048347261216905385 | 0.11888429191377428 |  |  |  | 0 | 0 | 9 | 7 |  |  |  |  |
| gain-gated loss-warm to blended consensus | v2101_f26_f30_gated_smoke | D-CHE | F31-gain-gated-loss-warm-blend-consensus | 9 | 0 | -0.08630077706442939 | 0.06482019689348009 |  |  |  | 0 | 0 | 9 | 5 |  |  |  |  |
| NoOp matched overhead | v2101_f26_f30_gated_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9510703749126859 | -0.8855135440826416 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_f30_gated_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.950757622718811 | -0.8852050569322374 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_f30_gated_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.94707723458608 | -0.8803360462188721 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_f30_gated_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.20035911930931938 | 0.05153660641776191 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_f30_gated_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.08923818005455865 | -0.0075536370277404785 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-warm to loss + easy B1 consensus blend | v2101_f26_f30_gated_smoke | D-FOU | F29-loss-warm-to-loss-easy-b1-consensus-blend | 9 | 0 | -0.15922502676645914 | -0.09435535801781549 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_f30_gated_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.07866508430904812 | -0.038517514864603676 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_f30_gated_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2982826232910156 | -0.019733395841386583 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| gain-gated loss-warm to B1 consensus | v2101_f26_f30_gated_smoke | D-FOU | F30-gain-gated-loss-warm-b1-consensus | 9 | 0 | -0.14217416445414224 | -0.06320655345916748 |  |  |  | 0 | 0 | 9 | 5 |  |  |  |  |
| gain-gated loss-warm to blended consensus | v2101_f26_f30_gated_smoke | D-FOU | F31-gain-gated-loss-warm-blend-consensus | 9 | 0 | -0.1360050704744127 | -0.058523747656080455 |  |  |  | 0 | 0 | 9 | 3 |  |  |  |  |
| NoOp matched overhead | v2101_f26_f32_b3null_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0542309151755438 | -0.9135279986593459 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_f32_b3null_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.0387049913406372 | -0.898009306854672 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_f32_b3null_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0477516253789265 | -0.9241670303874545 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_f32_b3null_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.367335041364034 | 0.02658810218175252 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_f32_b3null_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.09142195516162449 | 0.07362303468916151 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_f32_b3null_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.08120625548892552 | 0.042479621039496526 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_f32_b3null_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.5517797470092773 | -0.014498160945044624 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| gain-gated loss-warm to B1 consensus | v2101_f26_f32_b3null_smoke | D-CHE | F30-gain-gated-loss-warm-b1-consensus | 9 | 0 | -0.07061402665244208 | 0.09000831842422485 |  |  |  | 0 | 0 | 9 | 4 |  |  |  |  |
| B1 consensus transfer with B3-null safety projection | v2101_f26_f32_b3null_smoke | D-CHE | F32-b1-consensus-b3-null-transfer | 9 | 0 | -0.9093748728434244 | -1.0746632748179965 |  |  |  | 0 | 0 | 0 | 0 |  |  | 11.0 | 11.0 |
| loss-warm to B1 consensus with B3-null migration | v2101_f26_f32_b3null_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | 9 | 0 | -0.08457400401433308 | 0.05967905124028524 |  |  |  | 0 | 0 | 9 | 9 |  |  |  | 11.0 |
| NoOp matched overhead | v2101_f26_f32_b3null_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9510703749126859 | -0.8855135440826416 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| Random matched norm | v2101_f26_f32_b3null_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.950757622718811 | -0.8852050569322374 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| SGD control | v2101_f26_f32_b3null_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.94707723458608 | -0.8803360462188721 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| raw B1 consensus reference | v2101_f26_f32_b3null_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | 9 | 0 | -0.20035911930931938 | 0.05153660641776191 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| loss-warm to raw B1 consensus reference | v2101_f26_f32_b3null_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.08923818005455865 | -0.0075536370277404785 |  |  |  | 0 | 0 | 9 | 9 |  |  |  |  |
| loss-cotangent warmup reference | v2101_f26_f32_b3null_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.07866508430904812 | -0.038517514864603676 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| random matched target control | v2101_f26_f32_b3null_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2982826232910156 | -0.019733395841386583 |  |  |  | 0 | 0 | 0 | 0 |  |  |  |  |
| gain-gated loss-warm to B1 consensus | v2101_f26_f32_b3null_smoke | D-FOU | F30-gain-gated-loss-warm-b1-consensus | 9 | 0 | -0.1413480970594618 | -0.05897029240926107 |  |  |  | 0 | 0 | 9 | 5 |  |  |  |  |
| B1 consensus transfer with B3-null safety projection | v2101_f26_f32_b3null_smoke | D-FOU | F32-b1-consensus-b3-null-transfer | 9 | 0 | -1.5086084206899006 | -1.898079925113254 |  |  |  | 0 | 0 | 0 | 0 |  |  | 11.0 | 11.0 |
| loss-warm to B1 consensus with B3-null migration | v2101_f26_f32_b3null_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | 9 | 0 | -0.20267429616716173 | -0.12699764304690891 |  |  |  | 0 | 0 | 9 | 9 |  |  |  | 11.0 |

### F26-F33 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T1-loss-cotangent-target | MNIST | 0 | -0.17388534545898438 | -0.021915078163146973 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F27-loss-warm-to-easy-b1-consensus-migration | MNIST | 0 | -0.08820927143096924 | 0.02431774139404297 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 0 | -1.0710850954055786 | -0.9743665456771851 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F26-easy-b1-consensus-transfer | MNIST | 1 | -1.0548394918441772 | -1.1613452434539795 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 1 | -0.800024151802063 | -0.6592352390289307 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | MNIST | 2 | -0.17750346660614014 | -0.07051706314086914 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-SGD | MNIST | 2 | -1.2419205904006958 | -1.2139922380447388 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | Fashion-MNIST | 0 | -0.5954847931861877 | -0.08709251880645752 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T5-random-matched-target | Fashion-MNIST | 0 | -0.600997269153595 | -0.0013606548309326172 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T1-loss-cotangent-target | Fashion-MNIST | 1 | -0.09338337182998657 | -0.06109493970870972 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F27-loss-warm-to-easy-b1-consensus-migration | Fashion-MNIST | 1 | -0.12537360191345215 | -0.059024930000305176 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -1.6190503239631653 | -1.5939273238182068 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F26-easy-b1-consensus-transfer | Fashion-MNIST | 2 | -2.005918860435486 | -2.027565360069275 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | -0.7161132097244263 | -0.3078709840774536 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | KMNIST | 0 | -0.022940635681152344 | 0.193925142288208 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-SGD | KMNIST | 0 | -0.5807831287384033 | -0.4176739454269409 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 1 | -0.5954278707504272 | -0.10506439208984375 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T5-random-matched-target | KMNIST | 1 | -0.4806886911392212 | 0.10109162330627441 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T1-loss-cotangent-target | KMNIST | 2 | -0.21414721012115479 | -0.09985136985778809 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F27-loss-warm-to-easy-b1-consensus-migration | KMNIST | 2 | -0.1270751953125 | 0.07376575469970703 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 2 | -1.0838901996612549 | -0.9808403253555298 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F26-easy-b1-consensus-transfer | MNIST | 0 | -2.2780330181121826 | -2.758487582206726 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 0 | -1.0020992755889893 | -0.9770580530166626 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | MNIST | 1 | -0.24894332885742188 | -0.15180552005767822 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-SGD | MNIST | 1 | -0.8575433492660522 | -0.8030798435211182 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | MNIST | 2 | -0.41779470443725586 | -0.2505286931991577 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 2 | -0.3295940160751343 | -0.17860734462738037 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F3-T1-loss-cotangent-target | Fashion-MNIST | 0 | -0.1187509298324585 | -0.039713501930236816 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F27-loss-warm-to-easy-b1-consensus-migration | Fashion-MNIST | 0 | -0.2741612195968628 | -0.1695120930671692 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 0 | -1.4112365245819092 | -1.3615826964378357 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F26-easy-b1-consensus-transfer | Fashion-MNIST | 1 | -1.4206616878509521 | -1.6630752682685852 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.440943717956543 | -1.4029343724250793 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | Fashion-MNIST | 2 | -0.11420118808746338 | -0.041620612144470215 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 2 | -0.6854417324066162 | -0.511577844619751 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F10-T7-b1-cross-split-consensus-transfer | KMNIST | 0 | -0.05991089344024658 | 0.2033522129058838 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F3-T5-random-matched-target | KMNIST | 0 | -0.2542397975921631 | 0.27763426303863525 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F3-T1-loss-cotangent-target | KMNIST | 1 | -0.30040276050567627 | -0.3174753189086914 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F27-loss-warm-to-easy-b1-consensus-migration | KMNIST | 1 | -0.15751564502716064 | -0.11876583099365234 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-NoOpMatchedOverhead | KMNIST | 1 | -0.7203992605209351 | -0.6680334806442261 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | F26-easy-b1-consensus-transfer | KMNIST | 2 | -1.0868538618087769 | -1.3178011178970337 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 2 | -0.8921345472335815 | -0.8665083646774292 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F10-T7-b1-cross-split-consensus-transfer | MNIST | 0 | -0.2938110828399658 | 0.039618730545043945 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T5-random-matched-target | MNIST | 0 | -0.6817605495452881 | -0.11877155303955078 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F3-T1-loss-cotangent-target | MNIST | 1 | -0.05551314353942871 | 0.08500385284423828 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F27-loss-warm-to-easy-b1-consensus-migration | MNIST | 1 | -0.027146577835083008 | 0.09292447566986084 |  |  |  | 1 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 1 | -0.7819260358810425 | -0.6411499977111816 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | F26-easy-b1-consensus-transfer | MNIST | 2 | -1.6303542852401733 | -1.7578550577163696 |  |  |  | 0 | 0 | 0 |
| v2101_f26_easy_consensus_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 2 | -1.2135649919509888 | -1.1837722063064575 |  |  |  | 0 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M73-EasyB1ConsensusTransferFU` 与 `M74-LossWarmToEasyB1ConsensusMigrationFU`，并新增 `easy_split_consensus` target kind。该 target 只使用当前 train batch 内低 loss 或已预测正确的样本，叠加 B1/B2 类方向一致 mask。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M75-LossEasyB1ConsensusBlendFU` 与 `M76-LossWarmToLossEasyB1ConsensusBlendFU`，并新增 `loss_easy_consensus_blend` target kind。该 target 保留 loss-cotangent 分量，同时加入 easy B1 consensus 分量。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M77-GainGatedLossWarmB1ConsensusMigrationFU` 与 `M78-GainGatedLossWarmBlendMigrationFU`，用于给迁移 target 提供 train-split B2 gain precommit 语义。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M79-B1ConsensusB3NullTransferFU` 与 `M80-LossWarmToB1ConsensusB3NullMigrationFU`，并新增 `b1_b3zero` / `b1b2_b3zero` fit scope。该 scope 用 B1 target 求解 readout operator，同时把 B3 split 作为零位移约束纳入同一个 least-squares system。
- 修改 `experiments/run_v17_common.py`：扩展 M72/M74 two-phase migration branch；M74 warmup 后迁移到 M73 easy-consensus target。
- 修改 `experiments/run_v17_common.py`：继续扩展 M76 two-phase migration branch；M76 warmup 后迁移到 M75 blended target。
- 修改 `experiments/run_v17_common.py`：新增 M77/M78 gate 分支；gate reject 时不提交 FU residual，回退到普通 SGD step，并把 gate_accept/gain 写入 trace。
- 修改 `experiments/run_v17_common.py`：扩展 M80 two-phase migration branch；M80 warmup 后迁移到 M79 B3-null B1 consensus target。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F26-F33 specs 和 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：把 F26 summary runner 与 M73-M80 semantic contracts 纳入 S0.6。
- 新增 `experiments/run_v21_01_f26_easy_consensus_summary.py`，从 fresh `v2101_f26_` rows 汇总并更新 route/manifest/复盘。

### F26-F33 结论 / Insight

- 如果 grouped h800/h1600 仍不能同时为正，说明“只保留 train low-loss/easy examples 的 B1 consensus target”仍不能形成 early source chain。
- 如果 F27 比 F25 更差，说明 F25 的 near-chain 改善不是因为 hard examples 过多，而可能是 loss target 本身提供了早期 closure。
- 如果 F29 比 F27 更好但仍不能 early-chain，说明保留 loss target 可以缓解硬切换问题，但 easy-consensus 分量仍不是足够的 retained source selector。
- 如果 F30/F31 gate accept 很少且 h800 仍负，说明“无门槛提交”不是主要 blocker；如果 gate accept 很多但仍负，说明 train-split B2 gain 本身不能预测 early source retention。
- 如果 F32/F33 仍不能 early-chain，说明 B3/safety null 约束也不能把 B1 late-rebound target 转成 early retained source，blocker 更接近 B1 target 本身的早期相位问题，而不是 reservoir 泄漏。
- smoke positive 只允许升级 full，不允许直接写 breakthrough；controls 或 random target 若同步 positive，必须写 control-equivalent。

## 2026-06-04 07:16 F32/F33 B3-Null Source-Channel Repair 最终边界

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F26EasyConsensusSmokeNoEarlyChain`
- promotion_allowed: 0
- F26-F33 fresh smoke rows / blocked rows: 702 / 0
- candidate groups / early-chain groups / productive h3200 groups: 22 / 0 / 0
- full escalation recommended: 0

### 本轮新增修改

- 修改 `dgkan/fu/mechanisms.py`：新增 `M79-B1ConsensusB3NullTransferFU` 与 `M80-LossWarmToB1ConsensusB3NullMigrationFU`。
- 修改 `dgkan/fu/mechanisms.py`：`_exact_readout_function_space_actuation_update()` 新增 `b1_b3zero` / `b1b2_b3zero` fit scope；B1 target 求解时把 B3/safety split 加为零位移约束，并记录 `target_b3_null_rows`。
- 修改 `experiments/run_v17_common.py`：M80 two-phase branch warmup 后迁移到 M79；trace 额外写出 `target_b3_null_rows` 以便审计。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F32/F33 specs 与 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：M79/M80 纳入 mechanism semantic contract。
- 修改 `experiments/run_v21_01_f26_easy_consensus_summary.py`：F26 summary 扩展为 F26-F33，并输出 B3-null h800/h1600 诊断列。

### F32/F33 关键证据

| carrier | spec | rows | h800 | h1600 | h800+ rows | h1600+ rows | early chain | B3-null h800 | B3-null h1600 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-CHE | F32-b1-consensus-b3-null-transfer | 9 | -0.9093748728434244 | -1.0746632748179965 | 0 | 0 | 0 | 11.0 | 11.0 |
| D-CHE | F33-loss-warm-to-b1-consensus-b3-null | 9 | -0.08457400401433308 | 0.05967905124028524 | 1 | 6 | 0 |  | 11.0 |
| D-FOU | F32-b1-consensus-b3-null-transfer | 9 | -1.5086084206899006 | -1.898079925113254 | 0 | 0 | 0 | 11.0 | 11.0 |
| D-FOU | F33-loss-warm-to-b1-consensus-b3-null | 9 | -0.20267429616716173 | -0.12699764304690891 | 1 | 3 | 0 |  | 11.0 |

参照：

| carrier | spec | rows | h800 | h1600 | h800+ rows | h1600+ rows | early chain |
|---|---|---:|---:|---:|---:|---:|---:|
| D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | -0.09142195516162449 | 0.07362303468916151 | 2 | 6 | 0 |
| D-CHE | F30-gain-gated-loss-warm-b1-consensus | 9 | -0.07061402665244208 | 0.09000831842422485 | 4 | 4 | 0 |
| D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | -0.08923818005455865 | -0.0075536370277404785 | 3 | 4 | 0 |
| D-FOU | F30-gain-gated-loss-warm-b1-consensus | 9 | -0.1413480970594618 | -0.05897029240926107 | 2 | 3 | 0 |

### 分析 / Insight / 结论

- F32 直接 B3-null transfer 在 D-CHE/D-FOU 上都显著变差；说明把 safety/reservoir split 零位移约束直接压进 B1 consensus operator，并不能自然产生 early source。
- F33 保留 loss warmup 后，D-CHE 仍是 h800 负、h1600 正的 delayed shape；D-FOU 仍 h800/h1600 都为负。它没有把 F25/F30 的 late-rebound 线索变成连续 retained chain。
- B3-null 约束确实执行过：F32 h800/h1600 的 B3-null rows 均为 11.0，F33 在 h1600 进入 B3-null phase 后 rows=11.0；不是因为机制没有触发。
- 因为 candidate early-chain groups=0，本轮按计划不升级 full。继续把 F32/F33 的 scale、alt、warmup 或 B3 权重小扫，审计价值低。
- 当前 blocker 进一步收窄为：B1 consensus / late-rebound target 的早期相位仍错，不只是 easy-example 选择、B2 gain gate 或 B3 reservoir 泄漏问题。下一步若继续，需要新的 train-only early-source observability 或新的 target/source theory。

### 最终验证

- py_compile 通过：`dgkan/fu/mechanisms.py`、`experiments/run_v17_common.py`、`experiments/run_v21_common.py`、`experiments/run_v21_01_source_retention.py`、`experiments/run_v21_01_s06_truth_gate.py`、`experiments/run_v21_01_f26_easy_consensus_summary.py`。
- S0.6 import_closure=1；mechanism_contracts=15/15。
- `v21_01_required_artifact_manifest.csv` rows=68，required missing=0。
- 结束 GPU 查询：4 张 GPU utilization/memory 均为 0 / 0。

## 2026-06-04 F34-F35 View-Consistent Loss Target

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F34ViewConsistencySmokeNoEarlyChain`
- promotion_allowed: 0
- F34/F35 rows / blocked rows: 180 / 0
- smoke rows / full rows: 180 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- full escalation recommended: 0

F34/F35 是在 F17 selector 不可行动、F26-F33 B1 consensus/B3-null 仍无 early chain 后的新 source-target theory。它不使用 h800 source readback，不读 validation/test/future/query；只用当前 train batch 的 loss-cotangent target 与小输入扰动视角下的 per-sample/class 方向一致性，测试“对局部扰动稳定的 loss target”能否比 raw loss/random/B1 consensus 更早形成 h800/h1600 连续链。

### F34/F35 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | view stable b1 | view stable b2 | view align b1 | view align b2 | consensus density | B3-null rows |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead control | v2101_f34_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0542309151755438 | -0.9135279986593459 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| random matched norm control | v2101_f34_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.0387049913406372 | -0.898009306854672 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| SGD control | v2101_f34_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0477516253789265 | -0.9241670303874545 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| loss-warm B1 consensus reference | v2101_f34_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.1436084508895874 | 0.01679274770948622 |  |  |  | 0 | 0 |  |  |  |  | 0.588888900147544 |  |
| loss-cotangent reference | v2101_f34_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.08120625548892552 | 0.042479621039496526 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| random matched target control | v2101_f34_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.5088950660493639 | -0.044063323073916964 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| loss-warm B1 consensus B3-null reference | v2101_f34_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | 9 | 0 | -0.061333013905419245 | 0.09088578489091662 |  |  |  | 0 | 0 |  |  |  |  | 0.4555555714501275 | 11.0 |
| view-consistent loss target with B3 null | v2101_f34_smoke | D-CHE | F34-view-consistent-loss-b3-null | 9 | 0 | -0.8262767659293281 | -0.9412166873613993 |  |  |  | 0 | 0 | 1.0 | 1.0 | 0.9394038849406772 | 0.9521390464570787 | 0.40000000844399136 | 11.0 |
| loss-warm to view-consistent target | v2101_f34_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | 9 | 0 | -0.06328849660025702 | 0.09396105342441136 |  |  |  | 0 | 0 | 1.0 | 1.0 | 0.9747731884320577 | 0.967930879857805 | 0.4333333381348186 | 11.0 |
| stable random target control | v2101_f34_smoke | D-CHE | F9-TCTRL-stable-random-target | 9 | 0 | -1.0708018806245592 | -0.9489729735586379 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| NoOp matched overhead control | v2101_f34_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9510703749126859 | -0.8855135440826416 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| random matched norm control | v2101_f34_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.950757622718811 | -0.8852050569322374 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| SGD control | v2101_f34_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.94707723458608 | -0.8803360462188721 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| loss-warm B1 consensus reference | v2101_f34_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.191809786690606 | -0.11896745363871257 |  |  |  | 0 | 0 |  |  |  |  | 0.622222234805425 |  |
| loss-cotangent reference | v2101_f34_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.07866508430904812 | -0.038517514864603676 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| random matched target control | v2101_f34_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2445882029003567 | -0.0455379221174452 |  |  |  | 0 | 0 |  |  |  |  |  |  |
| loss-warm B1 consensus B3-null reference | v2101_f34_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | 9 | 0 | -0.08361487918429905 | 0.01046140988667806 |  |  |  | 0 | 0 |  |  |  |  | 0.5333333445919884 | 11.0 |
| view-consistent loss target with B3 null | v2101_f34_smoke | D-FOU | F34-view-consistent-loss-b3-null | 9 | 0 | -1.405264311366611 | -1.7105072339375813 |  |  |  | 0 | 0 | 1.0 | 1.0 | 0.9500866929690043 | 0.9542344278759427 | 0.3333333391282294 | 11.0 |
| loss-warm to view-consistent target | v2101_f34_smoke | D-FOU | F35-loss-warm-to-view-consistent-loss | 9 | 0 | -0.13765691386328804 | -0.05332006348503961 |  |  |  | 0 | 0 | 1.0 | 1.0 | 0.9857441782951355 | 0.9890266060829163 | 0.5444444484180875 | 11.0 |
| stable random target control | v2101_f34_smoke | D-FOU | F9-TCTRL-stable-random-target | 9 | 0 | -0.9556660254796346 | -0.8855741288926866 |  |  |  | 0 | 0 |  |  |  |  |  |  |

### F34/F35 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f34_smoke | D-CHE | F3-T1-loss-cotangent-target | MNIST | 0 | -0.17388534545898438 | -0.021915078163146973 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | MNIST | 0 | -0.11329531669616699 | 0.07600104808807373 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 0 | -1.0711513757705688 | -0.9744185209274292 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | MNIST | 1 | 0.028129100799560547 | 0.17117798328399658 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F9-TCTRL-stable-random-target | MNIST | 1 | -0.8579078912734985 | -0.752873420715332 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F3-T1-loss-cotangent-target | MNIST | 2 | -0.08958554267883301 | 0.02758955955505371 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | MNIST | 2 | -0.09640955924987793 | -0.029676437377929688 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 2 | -1.2551089525222778 | -1.2252906560897827 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | Fashion-MNIST | 0 | -0.19019633531570435 | -0.0721123218536377 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F9-TCTRL-stable-random-target | Fashion-MNIST | 0 | -1.4219478964805603 | -1.3394964933395386 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F3-T1-loss-cotangent-target | Fashion-MNIST | 1 | -0.1547895073890686 | -0.044454216957092285 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | Fashion-MNIST | 1 | -0.1668083667755127 | -0.09826457500457764 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.6555317044258118 | -1.6305643916130066 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | Fashion-MNIST | 2 | 0.14688456058502197 | 0.4026355743408203 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F9-TCTRL-stable-random-target | Fashion-MNIST | 2 | -0.7725387811660767 | -0.3670731782913208 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F3-T1-loss-cotangent-target | KMNIST | 0 | 0.11937904357910156 | 0.2629399299621582 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | KMNIST | 0 | 0.1076655387878418 | 0.3173791170120239 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | CTRL-RandomMatchedNorm | KMNIST | 0 | -0.5962033271789551 | -0.4036945104598999 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | KMNIST | 1 | -0.03943932056427002 | 0.14467179775238037 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F9-TCTRL-stable-random-target | KMNIST | 1 | -0.9415744543075562 | -0.8312855958938599 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F3-T1-loss-cotangent-target | KMNIST | 2 | -0.21131837368011475 | -0.1494230031967163 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | KMNIST | 2 | -0.06348288059234619 | 0.13528132438659668 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | CTRL-RandomMatchedNorm | KMNIST | 2 | -1.0531611442565918 | -0.9500881433486938 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | MNIST | 0 | 0.06326782703399658 | 0.1674809455871582 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F9-TCTRL-stable-random-target | MNIST | 0 | -0.9849693775177002 | -0.9687122106552124 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F3-T1-loss-cotangent-target | MNIST | 1 | -0.09702801704406738 | -0.027045249938964844 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F35-loss-warm-to-view-consistent-loss | MNIST | 1 | -0.07878565788269043 | 0.04767417907714844 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 1 | -0.8604913949966431 | -0.8023040294647217 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | MNIST | 2 | -0.25074291229248047 | -0.1997671127319336 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F9-TCTRL-stable-random-target | MNIST | 2 | -1.1713616847991943 | -1.1524723768234253 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F3-T1-loss-cotangent-target | Fashion-MNIST | 0 | -0.21801698207855225 | -0.1935301423072815 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F35-loss-warm-to-view-consistent-loss | Fashion-MNIST | 0 | -0.3244055509567261 | -0.24588829278945923 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 0 | -1.3960349559783936 | -1.3463672995567322 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | Fashion-MNIST | 1 | -0.23798608779907227 | -0.1896631121635437 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F9-TCTRL-stable-random-target | Fashion-MNIST | 1 | -1.403867483139038 | -1.3125365376472473 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F3-T1-loss-cotangent-target | Fashion-MNIST | 2 | 0.02070748805999756 | 0.0033916234970092773 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F35-loss-warm-to-view-consistent-loss | Fashion-MNIST | 2 | 0.04277503490447998 | 0.1150045394897461 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | -0.6971573829650879 | -0.5275270938873291 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | KMNIST | 0 | 0.14841890335083008 | 0.3181658983230591 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F9-TCTRL-stable-random-target | KMNIST | 0 | -0.4216606616973877 | -0.2471330165863037 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F3-T1-loss-cotangent-target | KMNIST | 1 | -0.10489058494567871 | -0.09933149814605713 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F35-loss-warm-to-view-consistent-loss | KMNIST | 1 | -0.35720300674438477 | -0.2960643768310547 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 1 | -0.7326239347457886 | -0.6802486181259155 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | KMNIST | 2 | -0.05884838104248047 | 0.035767555236816406 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-FOU | F9-TCTRL-stable-random-target | KMNIST | 2 | -0.8925403356552124 | -0.8239139318466187 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | MNIST | 0 | -0.07812857627868652 | 0.049637556076049805 |  |  |  | 1 | 0 | 0 |
| v2101_f34_smoke | D-CHE | F3-T5-random-matched-target | MNIST | 0 | -0.6817605495452881 | -0.11877155303955078 |  |  |  | 0 | 0 | 0 |
| v2101_f34_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 0 | -1.0695632696151733 | -0.9728447198867798 |  |  |  | 0 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `view_consistent_loss` target kind。该 target 对 B1/B2 train split 分别加入固定 seed 小输入扰动，只有 clean/perturbed loss-cotangent 方向 per-sample cosine 达标、且 B1/B2/扰动视角 class-sign 一致的分量才进入 target。
- 修改 `dgkan/fu/mechanisms.py`：新增 `M81-ViewConsistentLossTargetFU`，使用 `view_consistent_loss` + `b1_b3zero` fit scope；B3 split 作为零位移安全约束，不作为方向源。
- 修改 `dgkan/fu/mechanisms.py` 与 `experiments/run_v17_common.py`：新增 `M82-LossWarmToViewConsistentLossMigrationFU`，warmup 阶段用 `M49` loss-cotangent，warmup 后迁移到 `M81`。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F34/F35 specs，并纳入 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：把 F34 summary runner、M81/M82 合约与 M81 update-semantics smoke 纳入 S0.6。
- 新增 `experiments/run_v21_01_f34_view_consistency_summary.py`，只读取 fresh `v2101_f34_` rows 汇总 route、manifest 和复盘。

### F34/F35 结论 / Insight

- 若 F34/F35 grouped h800/h1600 仍不能同时为正，说明“train-only 扰动视角稳定性”也没有把 late-rebound target 转成 early retained source。
- 若 F35 相对 F34 改善 h800 但 h1600/h3200 断链，说明 loss-cotangent early closure 仍不能迁移成 retained source，blocker 继续指向 target-to-retention dynamics。
- 若 random/stable-random controls 同步变好，必须按 control-equivalent 处理，不能写 breakthrough。
- 只有 fresh full grouped h800/h1600/h3200/h4800 与 controls attribution 同时通过，才允许进入 promotion；本轮 smoke 只负责决定是否升级 full。

## 2026-06-04 F36-F39 Low-Bank Source-Channel Writer

### 是否达成 v21.01 目标

没有达成。

- route: `R2-LateReboundNoContinuousRetention-F34ViewConsistencySmokeNoEarlyChain-F36LowBankSmokeNoEarlyChain`
- promotion_allowed: 0
- F36-F39 rows / blocked rows: 252 / 0
- smoke rows / full rows: 252 / 0
- candidate early-chain groups: 0
- candidate productive h3200 groups: 0
- full escalation recommended: 0

F36-F39 是在 F34/F35 view-consistency 仍没有 early-chain 后，对计划 7.6 “低阶/低频 source bank + high-degree/high-frequency reservoir” 的直接实现与 gate 修复。方向仍只来自 train stream；B3 split 只作为零位移安全约束，不作为方向源；h800 source readback 只用于事后审计，不用于选择方向。F38/F39 只使用 train split B2/B3 precommit gains 决定是否提交 low-bank update，不使用 validation/test/future/query。

### F36-F39 Group Evidence

| condition | run label | carrier | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | early chain | productive h3200 | source proj h800 | source proj h1600 | reservoir proj h800 | reservoir proj h1600 | source bank h800 | source bank h1600 | reservoir h800 | reservoir h1600 | source energy h800 | source energy h1600 | reservoir energy h800 | reservoir energy h1600 | entropy h800 | B3-null h800 | B3-null h1600 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoOp matched overhead control | v2101_f36_fix_smoke | D-CHE | CTRL-NoOpMatchedOverhead | 9 | 0 | -1.0479830503463745 | -0.9072801338301765 |  |  | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| random matched norm control | v2101_f36_fix_smoke | D-CHE | CTRL-RandomMatchedNorm | 9 | 0 | -1.036706460846795 | -0.8960260616408454 |  |  | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SGD control | v2101_f36_fix_smoke | D-CHE | CTRL-SGD | 9 | 0 | -1.0383023818333943 | -0.9175342520078024 |  |  | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| loss-warm B1 consensus reference | v2101_f36_fix_smoke | D-CHE | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.08679509825176662 | 0.05475292603174845 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 72.0 | 72.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |
| loss-cotangent reference | v2101_f36_fix_smoke | D-CHE | F3-T1-loss-cotangent-target | 9 | 0 | -0.1588323712348938 | -0.07088491651746961 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 72.0 | 72.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |
| random matched target control | v2101_f36_fix_smoke | D-CHE | F3-T5-random-matched-target | 9 | 0 | -0.5437098079257541 | -0.05988399849997626 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 72.0 | 72.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |
| loss-warm B1 consensus B3-null reference | v2101_f36_fix_smoke | D-CHE | F33-loss-warm-to-b1-consensus-b3-null | 9 | 0 | -0.11076767577065362 | 0.032885485225253634 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 72.0 | 72.0 | 0.0 | 0.0 |  |  |  |  |  | 11.0 | 11.0 |
| view-consistent loss B3-null reference | v2101_f36_fix_smoke | D-CHE | F34-view-consistent-loss-b3-null | 9 | 0 | -0.991401645872328 | -1.1422222918934293 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 72.0 | 72.0 | 0.0 | 0.0 |  |  |  |  |  | 11.0 | 11.0 |
| loss-warm view-consistent reference | v2101_f36_fix_smoke | D-CHE | F35-loss-warm-to-view-consistent-loss | 9 | 0 | -0.0675118366877238 | 0.08259686496522692 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 72.0 | 72.0 | 0.0 | 0.0 |  |  |  |  |  | 11.0 | 11.0 |
| low-bank loss target with B3 reservoir null | v2101_f36_fix_smoke | D-CHE | F36-lowbank-loss-b3-null | 9 | 0 | -1.0737184815936618 | -0.9665189650323656 |  |  | 0 | 0 | 0.48231419920921326 | 0.48249539401796127 | 0.5176857908566793 | 0.517504612604777 | 24.0 | 24.0 | 48.0 | 48.0 | 22.4499454498291 | 22.4499454498291 | 24.096499337090385 | 24.078978220621746 | 0.9400820069842868 | 11.0 | 11.0 |
| loss-warm to low-bank source-channel writer | v2101_f36_fix_smoke | D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | 9 | 0 | -0.12386010090510051 | 0.03957306676440769 |  |  | 0 | 0 | 1.0 | 0.4766092167960273 | 0.0 | 0.5233907699584961 | 72.0 | 24.0 | 0.0 | 48.0 | 22.4499454498291 | 22.4499454498291 | 24.653867721557617 | 24.653867721557617 | 0.9519095023473104 | 11.0 | 11.0 |
| gain-gated low-bank loss target | v2101_f36_fix_smoke | D-CHE | F38-gain-gated-lowbank-loss-b3-null | 9 | 0 | -1.0669111013412476 | -0.9491784903738234 |  |  | 0 | 0 | 0.4831782976786296 | 0.48321253061294556 | 0.5168216824531555 | 0.5167874561415778 | 24.0 | 24.0 | 48.0 | 48.0 | 22.4499454498291 | 22.4499454498291 | 24.013252046373154 | 24.009999593098957 | 0.9385165108574761 | 11.0 | 11.0 |
| loss-warm to gain-gated low-bank writer | v2101_f36_fix_smoke | D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | 9 | 0 | -0.08992742829852635 | 0.08402078019248115 |  |  | 0 | 0 | 1.0 | 0.47594929734865826 | 0.0 | 0.5240506993399726 | 72.0 | 24.0 | 0.0 | 48.0 | 22.4499454498291 | 22.4499454498291 | 24.71895917256673 | 24.71895917256673 | 0.9536222087012397 | 11.0 | 11.0 |
| stable random target control | v2101_f36_fix_smoke | D-CHE | F9-TCTRL-stable-random-target | 9 | 0 | -1.0655020740297105 | -0.9426843921343485 |  |  | 0 | 0 | 0.32194600833786857 | 0.32197634710205925 | 0.6780539883507622 | 0.6780236562093099 | 16.0 | 16.0 | 56.0 | 56.0 | 14.966629981994629 | 14.966629981994629 | 31.52165963914659 | 31.517203013102215 | 0.9390192164315118 |  |  |
| NoOp matched overhead control | v2101_f36_fix_smoke | D-FOU | CTRL-NoOpMatchedOverhead | 9 | 0 | -0.9499522712495592 | -0.884395440419515 |  |  | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| random matched norm control | v2101_f36_fix_smoke | D-FOU | CTRL-RandomMatchedNorm | 9 | 0 | -0.9465207391315036 | -0.8809675375620524 |  |  | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SGD control | v2101_f36_fix_smoke | D-FOU | CTRL-SGD | 9 | 0 | -0.9459336996078491 | -0.8791209856669108 |  |  | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| loss-warm B1 consensus reference | v2101_f36_fix_smoke | D-FOU | F25-loss-warm-to-b1-consensus-migration | 9 | 0 | -0.14735788769192165 | -0.06212415960099962 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 96.0 | 96.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |
| loss-cotangent reference | v2101_f36_fix_smoke | D-FOU | F3-T1-loss-cotangent-target | 9 | 0 | -0.125939495033688 | -0.09351510471767849 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 96.0 | 96.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |
| random matched target control | v2101_f36_fix_smoke | D-FOU | F3-T5-random-matched-target | 9 | 0 | -0.2576432095633613 | 0.0005036327573988172 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 96.0 | 96.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |
| loss-warm B1 consensus B3-null reference | v2101_f36_fix_smoke | D-FOU | F33-loss-warm-to-b1-consensus-b3-null | 9 | 0 | -0.15306585364871556 | -0.07181317276424831 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 96.0 | 96.0 | 0.0 | 0.0 |  |  |  |  |  | 11.0 | 11.0 |
| view-consistent loss B3-null reference | v2101_f36_fix_smoke | D-FOU | F34-view-consistent-loss-b3-null | 9 | 0 | -1.4270322190390692 | -1.7185308933258057 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 96.0 | 96.0 | 0.0 | 0.0 |  |  |  |  |  | 11.0 | 11.0 |
| loss-warm view-consistent reference | v2101_f36_fix_smoke | D-FOU | F35-loss-warm-to-view-consistent-loss | 9 | 0 | -0.20091076691945395 | -0.11380968491236369 |  |  | 0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 96.0 | 96.0 | 0.0 | 0.0 |  |  |  |  |  | 11.0 | 11.0 |
| low-bank loss target with B3 reservoir null | v2101_f36_fix_smoke | D-FOU | F36-lowbank-loss-b3-null | 9 | 0 | -0.5058081414964464 | -0.06202507019042969 |  |  | 0 | 0 | 0.036975773258341685 | 0.04327867180109024 | 0.9630242188771566 | 0.9567213124699063 | 24.0 | 24.0 | 72.0 | 72.0 | 0.6489823261896769 | 0.819296293788486 | 16.858379364013672 | 18.058501773410374 | 0.9099030627144707 | 11.0 | 11.0 |
| loss-warm to low-bank source-channel writer | v2101_f36_fix_smoke | D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | 9 | 0 | -0.1801101631588406 | -0.10215540726979573 |  |  | 0 | 0 | 1.0 | 0.03831401177578502 | 0.0 | 0.9616859952608744 | 96.0 | 24.0 | 0.0 | 72.0 | 0.6854256126615736 | 0.6854256126615736 | 17.169183095296223 | 17.169183095296223 | 0.9157457749048868 | 11.0 | 11.0 |
| gain-gated low-bank loss target | v2101_f36_fix_smoke | D-FOU | F38-gain-gated-lowbank-loss-b3-null | 9 | 0 | -0.6553081936306424 | -0.21389214197794595 |  |  | 0 | 0 | 0.03485476184222433 | 0.04393300414085388 | 0.9651452435387505 | 0.9560669991705153 | 24.0 | 24.0 | 72.0 | 72.0 | 0.5964930852254232 | 0.8393825888633728 | 16.455475701226128 | 18.19470723470052 | 0.9045398235321045 | 11.0 | 11.0 |
| loss-warm to gain-gated low-bank writer | v2101_f36_fix_smoke | D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | 9 | 0 | -0.12710568639967176 | -0.03381265534294976 |  |  | 0 | 0 | 1.0 | 0.037479144831498466 | 0.0 | 0.9625208444065518 | 96.0 | 24.0 | 0.0 | 72.0 | 0.6632831825150384 | 0.6632831825150384 | 16.994245953030056 | 16.994245953030056 | 0.9131585889392428 | 11.0 | 11.0 |
| stable random target control | v2101_f36_fix_smoke | D-FOU | F9-TCTRL-stable-random-target | 9 | 0 | -0.9699249135123359 | -0.9120136102040609 |  |  | 0 | 0 | 0.47000859181086224 | 0.4632128179073334 | 0.5299914081891378 | 0.5367871854040358 | 16.0 | 16.0 | 80.0 | 80.0 | 7.434975465138753 | 7.431736734178331 | 8.388677597045898 | 8.616915067036947 | 0.8855023317866855 |  |  |

### F36-F39 Row Examples

| run label | carrier | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | washout | late | retained |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2101_f36_fix_smoke | D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | MNIST | 0 | -0.12436985969543457 | 0.07151710987091064 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-RandomMatchedNorm | MNIST | 0 | -1.0889517068862915 | -0.9922345876693726 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | MNIST | 1 | -0.11584639549255371 | 0.062170982360839844 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F9-TCTRL-stable-random-target | MNIST | 1 | -0.8409558534622192 | -0.753511905670166 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | MNIST | 2 | -0.3008606433868408 | -0.1736356019973755 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | Fashion-MNIST | 0 | -0.14043539762496948 | -0.004243373870849609 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F9-TCTRL-stable-random-target | Fashion-MNIST | 0 | -1.376140058040619 | -1.28243887424469 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | Fashion-MNIST | 1 | -0.19452929496765137 | -0.1737334132194519 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -1.6039286255836487 | -1.57892245054245 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | Fashion-MNIST | 2 | 0.0731496810913086 | 0.46051716804504395 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | KMNIST | 0 | 0.12785911560058594 | 0.353005051612854 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-RandomMatchedNorm | KMNIST | 0 | -0.5631105899810791 | -0.3705407381057739 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F37-loss-warm-to-lowbank-loss-b3-null | KMNIST | 1 | -0.27260470390319824 | -0.0919959545135498 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F9-TCTRL-stable-random-target | KMNIST | 1 | -0.9409240484237671 | -0.8225597143173218 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F39-loss-warm-to-gated-lowbank-loss-b3-null | KMNIST | 2 | -0.14116299152374268 | 0.05540347099304199 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | MNIST | 0 | -0.017339706420898438 | 0.07576239109039307 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F9-TCTRL-stable-random-target | MNIST | 0 | -1.0277328491210938 | -0.9705623388290405 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | MNIST | 1 | -0.19175517559051514 | -0.0665140151977539 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-RandomMatchedNorm | MNIST | 1 | -0.8441089391708374 | -0.7859210968017578 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | MNIST | 2 | -0.33166635036468506 | -0.2840869426727295 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | Fashion-MNIST | 0 | -0.2661677598953247 | -0.18522661924362183 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-RandomMatchedNorm | Fashion-MNIST | 0 | -1.4054679870605469 | -1.355791985988617 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | Fashion-MNIST | 1 | -0.25672709941864014 | -0.18540877103805542 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F9-TCTRL-stable-random-target | Fashion-MNIST | 1 | -1.4854905605316162 | -1.390839397907257 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | Fashion-MNIST | 2 | 0.05561983585357666 | 0.13816726207733154 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | KMNIST | 0 | -0.151566743850708 | -0.004521369934082031 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F9-TCTRL-stable-random-target | KMNIST | 0 | -0.42118000984191895 | -0.3042116165161133 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F39-loss-warm-to-gated-lowbank-loss-b3-null | KMNIST | 1 | -0.47503721714019775 | -0.4420963525772095 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-RandomMatchedNorm | KMNIST | 1 | -0.7220946550369263 | -0.6697090864181519 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F37-loss-warm-to-lowbank-loss-b3-null | KMNIST | 2 | -0.06297802925109863 | 0.004587650299072266 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F36-lowbank-loss-b3-null | MNIST | 0 | -1.0971227884292603 | -1.0579952001571655 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F3-T5-random-matched-target | MNIST | 0 | -0.6049913167953491 | -0.16268110275268555 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-NoOpMatchedOverhead | MNIST | 0 | -1.102005124092102 | -1.0052865743637085 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F38-gain-gated-lowbank-loss-b3-null | MNIST | 1 | -0.8505822420120239 | -0.7848720550537109 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-SGD | MNIST | 1 | -0.7958782911300659 | -0.7006452083587646 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F36-lowbank-loss-b3-null | MNIST | 2 | -1.2717055082321167 | -1.2035285234451294 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F38-gain-gated-lowbank-loss-b3-null | Fashion-MNIST | 0 | -1.4116224646568298 | -1.2846287488937378 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-SGD | Fashion-MNIST | 0 | -1.3521649241447449 | -1.2571409940719604 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F36-lowbank-loss-b3-null | Fashion-MNIST | 1 | -1.6817978024482727 | -1.6713995337486267 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F3-T5-random-matched-target | Fashion-MNIST | 1 | -0.7284670472145081 | -0.3036828637123108 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -1.6425426602363586 | -1.6174196600914001 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F38-gain-gated-lowbank-loss-b3-null | Fashion-MNIST | 2 | -0.7435861825942993 | -0.35720765590667725 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F36-lowbank-loss-b3-null | KMNIST | 0 | -0.6187267303466797 | -0.4375230073928833 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F3-T5-random-matched-target | KMNIST | 0 | -0.29811692237854004 | 0.14450788497924805 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-NoOpMatchedOverhead | KMNIST | 0 | -0.5685405731201172 | -0.3759363889694214 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F38-gain-gated-lowbank-loss-b3-null | KMNIST | 1 | -0.9445310831069946 | -0.8289893865585327 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | CTRL-SGD | KMNIST | 1 | -0.9385603666305542 | -0.8088763952255249 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-CHE | F36-lowbank-loss-b3-null | KMNIST | 2 | -1.026305913925171 | -0.9515334367752075 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F38-gain-gated-lowbank-loss-b3-null | MNIST | 0 | -0.7418854236602783 | -0.2397632598876953 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-SGD | MNIST | 0 | -0.9855473041534424 | -0.9593929052352905 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F36-lowbank-loss-b3-null | MNIST | 1 | -0.6600147485733032 | -0.24835968017578125 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F3-T5-random-matched-target | MNIST | 1 | -0.30019259452819824 | -0.003291606903076172 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-NoOpMatchedOverhead | MNIST | 1 | -0.8503905534744263 | -0.7921931743621826 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F38-gain-gated-lowbank-loss-b3-null | MNIST | 2 | -0.806538462638855 | -0.3666374683380127 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F36-lowbank-loss-b3-null | Fashion-MNIST | 0 | -0.6067334413528442 | -0.1479927897453308 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F3-T5-random-matched-target | Fashion-MNIST | 0 | -0.19018793106079102 | 0.044439613819122314 |  |  | 1 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-NoOpMatchedOverhead | Fashion-MNIST | 0 | -1.407346248626709 | -1.3576924204826355 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F38-gain-gated-lowbank-loss-b3-null | Fashion-MNIST | 1 | -0.7207756042480469 | -0.3696097731590271 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | CTRL-SGD | Fashion-MNIST | 1 | -1.4488484859466553 | -1.4064550995826721 |  |  | 0 | 0 | 0 |
| v2101_f36_fix_smoke | D-FOU | F36-lowbank-loss-b3-null | Fashion-MNIST | 2 | -0.17277705669403076 | 0.23475253582000732 |  |  | 1 | 0 | 0 |

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `low_bank`/`reservoir_bank` readout feature selection，并记录 `source_channel_projection`、`reservoir_projection`、source/reservoir feature norm、degree/band entropy 等诊断。
- 新增 `M83-LowBankLossB3NullFU`：只允许 readout low-bank feature 承接 loss-cotangent target，同时使用 `b1_b3zero`，把 B3 作为 reservoir null 约束。
- 新增 `M84-LossWarmToLowBankLossB3NullMigrationFU`：前 800 step 使用 `M49` loss-cotangent warmup，之后迁移到 `M83`，用于测试 early closure 是否能桥接到 low-bank source channel。
- 修改 `experiments/run_v17_common.py`：把 low-bank source/reservoir 诊断写入 trace，并把 M83 纳入 direct target-mechanism branch；这修复了初版 F36 direct writer 没有真正接入训练 loop 的实现 blocker。
- 修改 `experiments/run_v17_common.py`：把 M84/M85/M86 纳入 migration/gain-gated branch；M85 direct gated warmup 强制为 0，避免被错误当作 800-step warmup writer。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F36/F37 specs 和 target scope 白名单。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 F38/F39 gain-gated low-bank specs 和 target scope 白名单。
- 修改 `experiments/run_v21_01_s06_truth_gate.py`：把 F36 summary runner、M83-M86 合约和 M83/M85 update-semantics smoke 纳入 S0.6。
- 新增 `experiments/run_v21_01_f36_lowbank_source_channel_summary.py`，只读取指定 `--run-prefix` 的 fresh rows 汇总 route、manifest 和复盘；本轮最终判定使用 `v2101_f36_fix_`，不把初版 M83 branch blocker run 混入最终统计。

### F36-F39 结论 / Insight

- 如果 F36-F39 source projection/gate 有效但 h800/h1600 仍不能同时为正，说明把 target 限制到低阶/低频 bank 并用 B2/B3 precommit gate，也没有解决 early-source observability。
- 如果 h1600/h3200 late-positive 改善但 h800 仍为负，则 blocker 仍是 early target phase mismatch，而不是 reservoir 泄漏。
- 如果 controls 同步变好，按 control-equivalent 处理，不能写 breakthrough。
- 只有 fresh full grouped h800/h1600/h3200/h4800 与 controls attribution 同时通过，才允许继续 promotion；本轮 smoke 只负责决定是否升级 full。
