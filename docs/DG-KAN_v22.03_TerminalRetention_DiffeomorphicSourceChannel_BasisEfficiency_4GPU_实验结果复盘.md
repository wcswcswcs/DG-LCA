# DG-KAN v22.03 TerminalRetention DiffeomorphicSourceChannel BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-06 00:25:58 +0800

## Route

- route: `R3-MLPTerminalErosionExplained`
- S0.10 pass: 1
- D-CHE/D-FOU full-loop official closure: 1 / 1
- D-RAT/D-RBF: MicroNearE1LimitedRunnerPassedOfficialFusedBlocked / MicroNearE1LimitedRunnerPassedOfficialFusedBlocked
- functional early / h3200 / h4800: 45 / 45 / 0
- terminal erosion groups: 45
- best source continuation: `F154-F156 terminal debt/low-NDS/dual-memory oldshape full`
- best source candidate: `MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve`
- best h4800 / retention ratio: 0.15029900603824192 / 0.4323878091139131
- terminal autopsy decision: TerminalErosionExplained
- KANRoute: KANSourceBankMismatch
- promotion_allowed: 0
- required artifact missing count: 0

## Part 1 Code Audit

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 16/16 |  |
| compileall | 1 | py_compile | 0 |  |
| import_closure | 1 | import | 0 |  |
| terminal_retention_tests | 1 | tests | 4/4 |  |
| diffeomorphic_target_tests | 1 | tests | 1/1 |  |
| mechanism_contracts | 1 | v22.03 mechanisms | 48/48 |  |

## Part 2 Basis Efficiency / D-RAT / D-RBF

| carrier | S1_pass_rows | full_loop_official_closure | v22_03_same_kernel_functional_runner_proof | v22_03_decision |
| --- | --- | --- | --- | --- |
| D-CHE | 6 | 1 | 1 | OfficialEfficientCarrier |
| D-FOU | 3 | 1 | 1 | OfficialEfficientCarrier |

| carrier | profile_rows | near_E1_rows | micro_near_E1_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | gradcheck_pass_rows | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 14 | 0 | 13 | 1.0529286035698215 | 1.2559891610405118 | 1.03125 | 14 | MicroNearE1LimitedRunnerPassedOfficialFusedBlocked | official_fused_missing |
| D-RBF | 14 | 0 | 2 | 1.8830231952045868 | 1.3595845386003278 | 1.0312498211860657 | 14 | MicroNearE1LimitedRunnerPassedOfficialFusedBlocked | official_fused_missing |

### D-RAT / D-RBF Limited Same-Kernel Runner Integration

| carrier | component_variant | device | runner_steps | finite_loss_rows | nonzero_grad_rows | same_shape_rows | loss_start | loss_end | loss_delta | median_step_ms | limited_runner_kernel_match | official_fused_kernel_complete | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | RAT22.03-R0-current-reference | cuda:0 | 24 | 24 | 24 | 24 | 2.386381149291992 | 2.1802539825439453 | -0.20612716674804688 | 1.1283839121460915 | 1 | 0 | official_fused_missing |
| D-RBF | RBF22.03-R4-exp-approx-trainpath | cuda:0 | 24 | 24 | 24 | 24 | 2.3612561225891113 | 2.1940298080444336 | -0.16722631454467773 | 1.1743963696062565 | 1 | 0 | official_fused_missing |

### D-RAT / D-RBF Existing-Carrier Limited Functional Smoke

- This smoke uses the existing `PrimitiveKAN` carrier path in `run_v21_01_source_retention.py`; it is not the active micro-kernel production fused path.
- Scope: MNIST seed0, h1600, KSW1/KSW2/KSW9 plus matched controls. It is a blocker readback, not promotion evidence.

| carrier | limited_smoke_scope | candidate_rows | candidate_h100_positive | candidate_h400_positive | candidate_h800_positive | candidate_h1600_positive | best_h800_v21_id | best_h800 | best_h1600_for_best_h800 | best_positive_early_v21_id | best_positive_early_h100 | best_positive_early_h400 | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | existing_primitivekan_carrier_path | 3 | 1 | 1 | 0 | 0 | KSW9-h800-source-slow-ema-bank | -0.19887077808380127 | -0.2576025724411011 | KSW9-h800-source-slow-ema-bank | 0.018291950225830078 | 0.04062372446060181 | ExistingCarrierSmokeNoH800Source | h800_source_missing;micro_kernel_path_not_used;official_fused_missing |
| D-RBF | existing_primitivekan_carrier_path | 3 | 1 | 0 | 0 | 0 | KSW2-lowdegree-lowfreq-source-bank | -0.09932762384414673 | -0.015008985996246338 | KSW9-h800-source-slow-ema-bank | 0.013185977935791016 | -0.05124133825302124 | ExistingCarrierSmokeNoH800Source | h800_source_missing;micro_kernel_path_not_used;official_fused_missing |

| carrier | v21_id | source_h100_mean | source_h400_mean | source_h800_mean | source_h1600_mean | source_h100_pass_count | source_h400_pass_count | source_h800_pass_count | source_h1600_pass_count | decision | blocker | uses_micro_kernel_path | official_fused_kernel_complete | promotion_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | CTRL-AdamW | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | CTRL-NoOpMatchedOverhead | -0.3673520088195801 | -1.3743072152137756 | -1.6139620542526245 | -1.672694444656372 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | CTRL-RandomMatchedNorm | -0.3674502372741699 | -1.3744054436683655 | -1.6140602827072144 | -1.672792673110962 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | CTRL-SGD | -0.3684689998626709 | -1.3753696084022522 | -1.6149526834487915 | -1.6735420227050781 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | KSW1-basis-estimate-readout-commit | -0.36789703369140625 | -1.3747935891151428 | -1.6143707036972046 | -1.6729552745819092 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RAT | KSW2-lowdegree-lowfreq-source-bank | -0.3602449893951416 | -1.2478657364845276 | -0.4999626874923706 | -0.05122256278991699 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RAT | KSW9-h800-source-slow-ema-bank | 0.018291950225830078 | 0.04062372446060181 | -0.19887077808380127 | -0.2576025724411011 | 1 | 1 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RBF | CTRL-AdamW | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | CTRL-NoOpMatchedOverhead | -0.25667333602905273 | -1.3836166858673096 | -1.6201412081718445 | -1.69853276014328 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | CTRL-RandomMatchedNorm | -0.2743353843688965 | -1.4012668132781982 | -1.6377736926078796 | -1.7161373496055603 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | CTRL-SGD | -0.25414490699768066 | -1.378448724746704 | -1.614050567150116 | -1.6921083331108093 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | KSW1-basis-estimate-readout-commit | -0.2616105079650879 | -1.3851425647735596 | -1.619764506816864 | -1.6974827647209167 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RBF | KSW2-lowdegree-lowfreq-source-bank | -0.10958385467529297 | -0.24163174629211426 | -0.09932762384414673 | -0.015008985996246338 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RBF | KSW9-h800-source-slow-ema-bank | 0.013185977935791016 | -0.05124133825302124 | -0.2749130129814148 | -0.3533021807670593 | 1 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |

### D-RAT / D-RBF C4 Named FU Limited Smoke

- This is the plan-named C4 smoke: RAT-FU1/FU2/FU3 and RBF-FU1/FU2/FU3 plus matched controls.
- The FU names are existing source-retention aliases; D-RBF uses the available no-dense Triton RBF path, while D-RAT remains blocked by the absence of a production official fused rational path.

| carrier | limited_smoke_scope | candidate_rows | candidate_h100_positive | candidate_h400_positive | candidate_h800_positive | candidate_h1600_positive | best_h800_v21_id | best_h800 | best_h1600_for_best_h800 | best_positive_early_v21_id | best_positive_early_h100 | best_positive_early_h400 | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | existing_primitivekan_carrier_path | 3 | 1 | 1 | 0 | 0 | RAT-FU3-numerator-only-source-writer | -0.33644765615463257 | -0.5026246309280396 | RAT-FU3-numerator-only-source-writer | 0.03333449363708496 | 0.028262019157409668 | ExistingCarrierSmokeNoH800Source | h800_source_missing;micro_kernel_path_not_used;official_fused_missing |
| D-RBF | existing_primitivekan_carrier_path | 3 | 1 | 0 | 0 | 0 | RBF-FU3-compact-local-source-writer | -0.2967539429664612 | -0.3605765700340271 | RBF-FU3-compact-local-source-writer | 0.013073921203613281 | -0.028554558753967285 | ExistingCarrierSmokeNoH800Source | h800_source_missing;micro_kernel_path_not_used;official_fused_missing |

| carrier | v21_id | source_h100_mean | source_h400_mean | source_h800_mean | source_h1600_mean | source_h100_pass_count | source_h400_pass_count | source_h800_pass_count | source_h1600_pass_count | decision | blocker | uses_micro_kernel_path | official_fused_kernel_complete | promotion_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | CTRL-AdamW | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | CTRL-NoOpMatchedOverhead | -0.2050330638885498 | -1.1161019802093506 | -1.4833338856697083 | -1.6495115756988525 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | CTRL-RandomMatchedNorm | -0.20456600189208984 | -1.1156349182128906 | -1.4828668236732483 | -1.6490445137023926 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | CTRL-SGD | -0.2044992446899414 | -1.1154975891113281 | -1.482636272907257 | -1.6486287117004395 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RAT | RAT-FU1-readout-only-rational-source | -0.20433568954467773 | -1.1153168678283691 | -1.4824326634407043 | -1.6483888626098633 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RAT | RAT-FU2-denominator-safe-low-degree-source | -0.19958949089050293 | -1.0917096138000488 | -1.3662553429603577 | -0.5469076633453369 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RAT | RAT-FU3-numerator-only-source-writer | 0.03333449363708496 | 0.028262019157409668 | -0.33644765615463257 | -0.5026246309280396 | 1 | 1 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RBF | CTRL-AdamW | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | CTRL-NoOpMatchedOverhead | -0.2551536560058594 | -1.3842896223068237 | -1.6738958358764648 | -1.7377207279205322 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | CTRL-RandomMatchedNorm | -0.22713947296142578 | -1.356268048286438 | -1.6458752155303955 | -1.7096760272979736 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | CTRL-SGD | -0.21671772003173828 | -1.3466647863388062 | -1.638942003250122 | -1.7081782817840576 | 0 | 0 | 0 | 0 | ControlRow | control | 0 | 0 | 0 |
| D-RBF | RBF-FU1-readout-only-local-support-source | -0.23404788970947266 | -1.3608626127243042 | -1.6497712135314941 | -1.7165915966033936 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RBF | RBF-FU2-active-center-low-k-source | -0.1580660343170166 | -0.6229804754257202 | -0.3010181188583374 | -0.09910190105438232 | 0 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |
| D-RBF | RBF-FU3-compact-local-source-writer | 0.013073921203613281 | -0.028554558753967285 | -0.2967539429664612 | -0.3605765700340271 | 1 | 0 | 0 | 0 | ExistingCarrierSmokeNoH800Source | h800_source_missing | 0 | 0 | 0 |

## Part 3 Terminal Erosion Autopsy

| terminal_erosion_class | groups | fraction |
| --- | --- | --- |
| MeasuredMonotoneTerminalErosion | 18 | 0.6666666666666666 |
| PostH3200PostH4000Erosion | 9 | 0.3333333333333333 |

| v22_id | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | terminal_erosion_class | v22_03_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F106-early100-h800-source-slow-ema-retention | 0.3210880822605557 | 0.2249905467033386 | 0.1525942418310377 | 0.4752410639371254 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F115-early100-h800-source-slow-ema-strong-retention | 0.3210328287548489 | 0.2249501678678724 | 0.1525664197074042 | 0.4752361940650902 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F117-early100-h800-source-slow-ema-terminal-guard | 0.3183568881617652 | 0.2222588625219133 | 0.1499197847313351 | 0.4709173581793432 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard | 0.3307922316922081 | 0.2346738775571187 | 0.1622624132368299 | 0.4905266741203586 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong | 0.3233591185675727 | 0.2272418571843041 | 0.1548460755083296 | 0.4788671993982172 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer | 0.323529389169481 | 0.2273986405796474 | 0.1549697683917151 | 0.4789974993911114 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend | 0.3266988396644592 | 0.2305566204918755 | 0.1580851872762044 | 0.4838865893694881 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout | 0.3264219197962019 | 0.2302946978145175 | 0.1578810877270168 | 0.4836718313083516 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry | 0.3240864939159817 | 0.2279582950803968 | 0.1555491255389319 | 0.4799617647110513 | 8 | PostH3200PostH4000Erosion | TerminalErosion;PostH4000Erosion |
| MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target | 0.3312460548347897 | 0.2152763936254713 | 0.134279986222585 | 0.4053783713425892 | 8 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F125-early100-h800-source-slow-ema-source-bank-target | 0.3192800283432007 | 0.2035752170615726 | 0.1222706900702582 | 0.3829575269857686 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F126-early100-h800-source-slow-ema-dual-target-guard | 0.3385555247465769 | 0.2227248748143514 | 0.1414172252019246 | 0.4177076280405744 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw | 0.3297484748893314 | 0.2137181427743699 | 0.1324055559105343 | 0.4015350062042641 | 8 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source | 0.3142895963456895 | 0.1982795099417368 | 0.1169712808397081 | 0.3721767509957616 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve | 0.3290293349160088 | 0.2130403055085076 | 0.1317636966705322 | 0.400461851537242 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target | 0.3397622671392228 | 0.2237474487887488 | 0.142462584707472 | 0.4193007831828947 | 8 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target | 0.321227232615153 | 0.2052310076024797 | 0.1241881317562527 | 0.3866052412344397 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target | 0.332035524977578 | 0.216049227449629 | 0.1347834401660495 | 0.405930781578722 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue | 0.3398546510272556 | 0.2238691714074876 | 0.1426512897014618 | 0.4197420552294324 | 7 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue | 0.320832351843516 | 0.2048467000325521 | 0.1235881447792053 | 0.3852109803424209 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue | 0.3316524492369758 | 0.2156531843874189 | 0.1343820591767629 | 0.4051894068200981 | 8 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue | 0.3396590550740559 | 0.2236685984664493 | 0.1424066258801354 | 0.4192634459549045 | 8 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue | 0.3228688836097717 | 0.2068851954407162 | 0.1256653997633192 | 0.3892149604456832 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue | 0.3405913147661421 | 0.2246015701029035 | 0.1433822446399265 | 0.4209803316281745 | 9 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |

_仅显示前 24 / 27 rows；完整 CSV 见 artifact。_

## Part 4 C1/C2/C1-Fallback Continuation Ledger

- All rows below are rendered from fresh old-shape continuation CSV/JSON artifacts.
- Productive h4800 gate is unchanged: mean h4800 >= 0.005, `R_4800/3200 >= 0.50`, and row-positive >= 7/9.

| continuation | candidate_groups | candidate_early_chain | candidate_continuous_h3200 | candidate_h4800 | terminal_erosion_groups | best_v22_id | best_h4800 | best_h4800_retention_ratio | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F145-F147 source-preservation oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | 0.14219152596261767 | 0.41891942358555095 | NoH4800Candidate |
| F148-F150 diffeomorphic target oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | 0.14314018024338615 | 0.4204398126876261 | NoH4800Candidate |
| F151-F153 terminal-preserve repair oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve | 0.13418350948227775 | 0.4048600858991034 | NoH4800Candidate |
| F154-F156 terminal debt/low-NDS/dual-memory oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve | 0.15029900603824192 | 0.4323878091139131 | NoH4800Candidate |
| F157-F159 signal-estimator terminal retention oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor | 0.1451091104083591 | 0.42377215113636235 | NoH4800Candidate |
| F160-F162 post-h4000 source-floor terminal retention oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor | 0.14896754423777261 | 0.43010448760558817 | NoH4800Candidate |
| F163-F165 raw-guard post-h4000 floor oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor | 0.14042934444215563 | 0.4158218364598758 | NoH4800Candidate |
| F166-F168 anti-erosion orthogonal/transport oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector | 0.13835234443346658 | 0.41220559434526777 | NoH4800Candidate |
| F169-F171 h3200-anchor transport oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport | 0.14096160067452324 | 0.41673011176809377 | NoH4800Candidate |
| F172-F174 source-preserving progress-carry oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry | 0.13602299822701347 | 0.4079885425908324 | NoH4800Candidate |
| F175-F177 raw-guard gentle progress oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress | 0.13225279251734415 | 0.4013371874988695 | NoH4800Candidate |
| F178-F180 control-relative terminal catch-up oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup | 0.13699083195792305 | 0.4097839293516009 | NoH4800Candidate |
| F181-F183 trajectory-adaptive terminal retention oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F181-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve | 0.1249746216668023 | 0.38785703646301367 | NoH4800Candidate |
| F184-F186 minimal terminal transport oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny | 0.11503769954045613 | 0.36831442816645493 | NoH4800Candidate |
| F187-F189 terminal accept-memory oldshape full | 3 | 3 | 3 | 0 | 3 | MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory | 0.12118528948889838 | 0.3805887480765032 | NoH4800Candidate |

### Best Candidate Ranking

| continuation | v22_id | h100 | h400 | h800 | h1600 | h2400 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F154-F156 terminal debt/low-NDS/dual-memory oldshape full | MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve | 0.03423149718178643 | 0.2391418284840054 | 0.5204394923316108 | 0.5242930716938443 | 0.44171450204319423 | 0.3476023210419549 | 0.23160096009572348 | 0.15029900603824192 | 0.4323878091139131 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F160-F162 post-h4000 source-floor terminal retention oldshape full | MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor | 0.03514507081773546 | 0.24017716116375393 | 0.5192443165514204 | 0.5230807893806033 | 0.4404814773135715 | 0.3463519877857632 | 0.23032606972588432 | 0.14896754423777261 | 0.43010448760558817 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F157-F159 signal-estimator terminal retention oldshape full | MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor | 0.028437336285909016 | 0.23713875479168361 | 0.5152796771791246 | 0.5191255509853363 | 0.43653973274760777 | 0.3424224787288242 | 0.2264139817820655 | 0.1451091104083591 | 0.42377215113636235 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F148-F150 diffeomorphic target oldshape full | MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | 0.023919410175747342 | 0.23278671834203932 | 0.5133035580317179 | 0.5171512762705485 | 0.4345679415596856 | 0.34045343928866917 | 0.2244438264105055 | 0.14314018024338615 | 0.4204398126876261 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F145-F147 source-preservation oldshape full | MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | 0.01605015993118286 | 0.23353919718000624 | 0.5122234655751122 | 0.5160844590928819 | 0.4335179593827989 | 0.33942452404234147 | 0.22343753443823922 | 0.14219152596261767 | 0.41891942358555095 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F169-F171 h3200-anchor transport oldshape full | MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport | 0.029290364848242864 | 0.23739412095811632 | 0.511077708668179 | 0.514933145708508 | 0.43236282136705184 | 0.3382563359207577 | 0.22225730948978 | 0.14096160067452324 | 0.41673011176809377 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F163-F165 raw-guard post-h4000 floor oldshape full | MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor | 0.02430157528983222 | 0.23183465666241115 | 0.5105579230520461 | 0.5144051909446716 | 0.4318276147047679 | 0.33771517541673446 | 0.2217154105504354 | 0.14042934444215563 | 0.4158218364598758 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F148-F150 diffeomorphic target oldshape full | MLP-F150-early100-h800-source-slow-ema-low-rank-readout-transport | 0.03562576903237237 | 0.2363457183043162 | 0.5090375675095452 | 0.5129128893216451 | 0.4303617411189609 | 0.33627867698669434 | 0.22029011779361302 | 0.13901315132776895 | 0.4133867558104772 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F166-F168 anti-erosion orthogonal/transport oldshape full | MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector | 0.034461776415506996 | 0.23405538333786857 | 0.5084546340836419 | 0.5123150481118096 | 0.4297398428122203 | 0.33563917213016087 | 0.2196390794383155 | 0.13835234443346658 | 0.41220559434526777 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F169-F171 h3200-anchor transport oldshape full | MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry | 0.023271964655982122 | 0.23289917574988472 | 0.5079161193635728 | 0.5117919213242001 | 0.4292367200056712 | 0.3351471159193251 | 0.21916387809647453 | 0.13790216710832384 | 0.411467563222352 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F154-F156 terminal debt/low-NDS/dual-memory oldshape full | MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve | 0.028927783171335857 | 0.23650172021653917 | 0.5072741078005897 | 0.5111578471130795 | 0.42861564291848075 | 0.3345329297913445 | 0.2185691793759664 | 0.13737184471554226 | 0.410637735427792 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F178-F180 control-relative terminal catch-up oldshape full | MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup | 0.024892005655500624 | 0.23378056950039333 | 0.5071572495831383 | 0.5110023551517062 | 0.42841746078597176 | 0.33430015709665084 | 0.21828498111830819 | 0.13699083195792305 | 0.4097839293516009 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F157-F159 signal-estimator terminal retention oldshape full | MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport | 0.03191579050487942 | 0.23535731103685167 | 0.5062927140129937 | 0.5101695987913344 | 0.4276178810331557 | 0.3335407144493527 | 0.21756445036994088 | 0.13635704583591884 | 0.40881679485826106 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F172-F174 source-preserving progress-carry oldshape full | MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry | 0.030281133121914335 | 0.2322329216533237 | 0.5062744286325243 | 0.510111411412557 | 0.42752213610543144 | 0.33339906401104397 | 0.21735644340515137 | 0.13602299822701347 | 0.4079885425908324 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F151-F153 terminal-preserve repair oldshape full | MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve | 0.024158146646287706 | 0.23245551188786825 | 0.504243983162774 | 0.508101761341095 | 0.4255329933431413 | 0.3314318060874939 | 0.21546246939235264 | 0.13418350948227775 | 0.4048600858991034 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F151-F153 terminal-preserve repair oldshape full | MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source | 0.0309207903014289 | 0.22927120327949524 | 0.5040274759133657 | 0.5078812274667952 | 0.4253038167953491 | 0.3311949107382033 | 0.2151894768079122 | 0.13389651311768425 | 0.4042831238536882 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F163-F165 raw-guard post-h4000 floor oldshape full | MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor | 0.021443731255001493 | 0.2292703456348843 | 0.5037513275941213 | 0.5076059003671011 | 0.4250189595752292 | 0.3309069673220317 | 0.2149040632777744 | 0.1336647007200453 | 0.40393438011223753 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F151-F153 terminal-preserve repair oldshape full | MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong | 0.016952700085110135 | 0.22711801528930664 | 0.5032383733325534 | 0.5070920586585999 | 0.42451685004764134 | 0.3304118911425273 | 0.2144118779235416 | 0.13317129347059461 | 0.4030463098955162 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F160-F162 post-h4000 source-floor terminal retention oldshape full | MLP-F160-early100-h800-source-slow-ema-terminal-source-floor | 0.025328218936920166 | 0.23075381914774576 | 0.5022448930475447 | 0.5061230659484863 | 0.42357585827509564 | 0.32949671480390763 | 0.21352261635992262 | 0.13233332501517403 | 0.401622593093012 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F175-F177 raw-guard gentle progress oldshape full | MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress | 0.02335088120566474 | 0.22701325019200644 | 0.5023462639914619 | 0.5062010354465909 | 0.4236292640368144 | 0.32953037155999076 | 0.21354025602340698 | 0.13225279251734415 | 0.4013371874988695 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F172-F174 source-preserving progress-carry oldshape full | MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend | 0.034256703323788114 | 0.2307098905245463 | 0.5021474328305986 | 0.5060018433464898 | 0.4234279990196228 | 0.32932109302944607 | 0.21333535843425327 | 0.13201194339328343 | 0.40086088072554665 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F178-F180 control-relative terminal catch-up oldshape full | MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup | 0.03051006131701999 | 0.2280600666999817 | 0.5017823113335503 | 0.50562541352378 | 0.42305686738755965 | 0.3289473189247979 | 0.2129563887914022 | 0.13166724310980904 | 0.4002684792816629 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F166-F168 anti-erosion orthogonal/transport oldshape full | MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal | 0.024917635652754042 | 0.2258118987083435 | 0.5012382169564565 | 0.5050929387410482 | 0.42250615027215743 | 0.32838569084803265 | 0.2123797701464759 | 0.1311053368780348 | 0.39924192963300137 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F145-F147 source-preservation oldshape full | MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong | 0.031599137518141 | 0.234635720650355 | 0.5005269183052911 | 0.5043803420331743 | 0.42180012663205463 | 0.32769686645931667 | 0.21169447236590916 | 0.13043878144688076 | 0.39804708191518406 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F175-F177 raw-guard gentle progress oldshape full | MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress | 0.022939364115397137 | 0.22418111893865797 | 0.5003823671076033 | 0.5042592883110046 | 0.4217046797275543 | 0.3276138967937893 | 0.21163182126151192 | 0.13036236498090956 | 0.39791463749464756 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F154-F156 terminal debt/low-NDS/dual-memory oldshape full | MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block | 0.02318859100341797 | 0.22953039407730103 | 0.499278810289171 | 0.5031533638636271 | 0.420597427421146 | 0.32650257481469047 | 0.21051009164916146 | 0.12921792268753052 | 0.39576387034886124 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F169-F171 h3200-anchor transport oldshape full | MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport | 0.02282049258550008 | 0.22802523771921793 | 0.49639925691816544 | 0.5002618100908067 | 0.41769155528810287 | 0.32358843088150024 | 0.2075935403505961 | 0.12629363934199014 | 0.3902909600258222 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F148-F150 diffeomorphic target oldshape full | MLP-F149-early100-h800-source-slow-ema-info-volume-diffeomorphic-target | 0.024481554826100666 | 0.2253530224164327 | 0.4952084223429362 | 0.49911774198214215 | 0.41659241914749146 | 0.322532531287935 | 0.2065784732500712 | 0.1257212890519036 | 0.3897941350284081 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F181-F183 trajectory-adaptive terminal retention oldshape full | MLP-F181-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve | 0.03923486338721381 | 0.20031070046954685 | 0.4950275421142578 | 0.49888328711191815 | 0.4163099726041158 | 0.3222182657983568 | 0.2062396009763082 | 0.1249746216668023 | 0.38785703646301367 | 8 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |
| F157-F159 signal-estimator terminal retention oldshape full | MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator | 0.023344748549991183 | 0.2290959490670098 | 0.49472278025415206 | 0.4985627267095778 | 0.4159648021062215 | 0.32183868686358136 | 0.20581589142481485 | 0.12448746297094557 | 0.38680080441576126 | 9 | 1 | 1 | 0 | 1 | TerminalErosion;PostH4000Erosion |

_仅显示前 30 / 45 rows；完整 CSV 见 artifact。_

## Part 5 C1 F145-F147 Source-Preservation Continuation

- continuation source: `continuation_f145_f147_source_preserve_oldshape_full`
- C1 repair family: source-preserve strong, source-preserve gentle, and information-volume guard around the F118-style early100 h800 slow-EMA path.
- best C1 candidate: `MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard` h4800=0.14219152596261767, h4800_retention_ratio=0.41891942358555095

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong | 9 | 0.5005269183052911 | 0.32769686645931667 | 0.21169447236590916 | 0.13043878144688076 | 0.39804708191518406 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F146-early100-h800-source-slow-ema-terminal-source-preserve-gentle | 9 | 0.4901217619578044 | 0.3173616627852122 | 0.20137612356079948 | 0.12014251285129124 | 0.3785665596685593 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |
| MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | 9 | 0.5122234655751122 | 0.33942452404234147 | 0.22343753443823922 | 0.14219152596261767 | 0.41891942358555095 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion |

## Part 6 C2 F148-F150 Diffeomorphic Target Continuation

- continuation source: `continuation_f148_f150_diffeomorphic_target_oldshape_full`
- C2 repair family: low-NDS weak-stable target, information-volume split-consensus target, and low-rank readout transport target.
- Diffeomorphic metrics are local train-stream logits displacement proxies recorded for audit; they are not promoted as exact Jacobian/Hessian proof.
- best C2 candidate: `MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target` h4800=0.14314018024338615, h4800_retention_ratio=0.4204398126876261

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | 9 | 0.5133035580317179 | 0.34045343928866917 | 0.2244438264105055 | 0.14314018024338615 | 0.4204398126876261 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8463396636849065 | 0.15366033631509368 | 0.7671288272791803 | 0.3092119401251828 | 1 |
| MLP-F149-early100-h800-source-slow-ema-info-volume-diffeomorphic-target | 9 | 0.4952084223429362 | 0.322532531287935 | 0.2065784732500712 | 0.1257212890519036 | 0.3897941350284081 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8502798041970534 | 0.14972019580294674 | 0.8157749423228219 | 0.1448161679285544 | 1 |
| MLP-F150-early100-h800-source-slow-ema-low-rank-readout-transport | 9 | 0.5090375675095452 | 0.33627867698669434 | 0.22029011779361302 | 0.13901315132776895 | 0.4133867558104772 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8455664938964076 | 0.1544335061035917 | 0.773198843663366 | 0.3973623915678925 | 1 |

### C2 Target-Gate Trace Evidence

| v22_id | target_attempt_rows | operator_accept_rows | target_families | operator_statuses | mean_ActuationR2 | max_B2_transfer_gain | max_B3_safety_gain | max_jacobian_condition_mean | max_fold_rate | max_neighbor_order_flip_rate | max_local_distance_distortion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | 54 | 0 | T3-low-NDS-weak-stable-readout | h800_source_slow_ema_low_nds_diffeomorphic_target:pre_terminal_target_gate;h800_source_slow_ema_low_nds_diffeomorphic_target:reject | 0.2642672724193997 | 0.023189842700958252 | 0.004485189914703369 | 1.0111160278320312 | 0.002322880318388343 | 0.02380952425301075 | 907.375732421875 |
| MLP-F149-early100-h800-source-slow-ema-info-volume-diffeomorphic-target | 54 | 1 | T4-info-volume-stable-consensus | h800_source_slow_ema_info_volume_diffeomorphic_target:info_volume_diffeomorphic;h800_source_slow_ema_info_volume_diffeomorphic_target:pre_terminal_target_gate;h800_source_slow_ema_info_volume_diffeomorphic_target:reject | 0.28955805743182145 | 0.06745278835296631 | 0.006702065467834473 | 1.0153052806854248 | 0.0011614401591941714 | 0.0476190485060215 | 1095.297607421875 |
| MLP-F150-early100-h800-source-slow-ema-low-rank-readout-transport | 54 | 0 | T6-low-rank-view-consistent-readout-transport | h800_source_slow_ema_low_rank_readout_transport:pre_terminal_target_gate;h800_source_slow_ema_low_rank_readout_transport:reject | 0.0479045068776166 | 0.004059106111526489 | 0.003501683473587036 | 1.013807773590088 | 0.002322880318388343 | 0.02380952425301075 | 0.0025348106864839792 |

## Part 7 C1-Fallback F151-F153 Terminal-Preserve Repair

- continuation source: `continuation_f151_f153_terminal_preserve_repair_oldshape_full`
- Trigger: C1/C2 left h4800 positive but retention ratio below 0.50, so the plan directs increasing source preservation strength, trying h3600 terminal attach, and using row-orthogonal source updates rather than more ActuationR2-only target variants.
- F151 increases terminal preservation strength without increasing h800 amplitude; F152 attaches source preservation at h3600; F153 tests a Nora-style row-orthogonal source update using train-stream source state only.
- best C1-fallback candidate: `MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve` h4800=0.13418350948227775, h4800_retention_ratio=0.4048600858991034

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong | 9 | 0.5032383733325534 | 0.3304118911425273 | 0.2144118779235416 | 0.13317129347059461 | 0.4030463098955162 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8447945527676057 | 0.1552054472323944 | 0.7544640863227873 | 0.04801387036288226 | 1 |
| MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve | 9 | 0.504243983162774 | 0.3314318060874939 | 0.21546246939235264 | 0.13418350948227775 | 0.4048600858991034 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8481416497400138 | 0.15185835025998598 | 0.786189027186713 | 0.043252018866715605 | 1 |
| MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source | 9 | 0.5040274759133657 | 0.3311949107382033 | 0.2151894768079122 | 0.13389651311768425 | 0.4042831238536882 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8455524832438835 | 0.15444751675611643 | 0.7652328618267142 | 0.04581132531166077 | 1 |

## Part 8 C1-Fallback F154-F156 Debt / Low-NDS / Dual-Memory Repair

- continuation source: `continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full`
- Trigger: F151-F153 worsened retention relative to the F118 near-miss, so this run tests the remaining C1 mechanisms rather than stronger preservation: debt-aware terminal preserve, low-NDS hidden/readout matrix-block source, and long-memory terminal source preservation.
- best C1-fallback-2 candidate: `MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve` h4800=0.15029900603824192, h4800_retention_ratio=0.4323878091139131

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve | 9 | 0.5204394923316108 | 0.3476023210419549 | 0.23160096009572348 | 0.15029900603824192 | 0.4323878091139131 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8456269247518395 | 0.1543730752481606 | 0.7417619276924253 | 0.046840552930478695 | 1 |
| MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block | 9 | 0.499278810289171 | 0.32650257481469047 | 0.21051009164916146 | 0.12921792268753052 | 0.39576387034886124 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8492051155739542 | 0.15079488442604605 | 0.7865889720917969 | 0.04815562234984504 | 1 |
| MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve | 9 | 0.5072741078005897 | 0.3345329297913445 | 0.2185691793759664 | 0.13737184471554226 | 0.410637735427792 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.844195488225146 | 0.1558045117748539 | 0.764509109541859 | 0.04628916581471761 | 1 |

## Part 9 F157-F159 Signal-Estimator Terminal Retention

- continuation source: `continuation_f157_f159_signal_estimator_oldshape_full`
- Trigger: F154-F156 still left terminal erosion below the 0.50 retention gate, so this run reopens PopRisk/SNR only as a train-stream signal estimator: terminal-retention predictor, split-consensus estimator, and signal/reservoir transport. It is not a direct parameter mask and does not use h4800 outcomes as a selector.
- best signal-estimator candidate: `MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor` h4800=0.1451091104083591, h4800_retention_ratio=0.42377215113636235

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor | 9 | 0.5152796771791246 | 0.3424224787288242 | 0.2264139817820655 | 0.1451091104083591 | 0.42377215113636235 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8453202607504738 | 0.15467973924952608 | 0.7629115297859381 | 0.04496950352633441 | 1 |
| MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator | 9 | 0.49472278025415206 | 0.32183868686358136 | 0.20581589142481485 | 0.12448746297094557 | 0.38680080441576126 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8493817123401023 | 0.15061828765989754 | 0.8000654086230368 | 0.04009130707493535 | 1 |
| MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport | 9 | 0.5062927140129937 | 0.3335407144493527 | 0.21756445036994088 | 0.13635704583591884 | 0.40881679485826106 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8461882957169701 | 0.1538117042830298 | 0.7343183284384195 | 0.050524256847522875 | 1 |

## Part 10 F160-F162 Post-H4000 Source-Floor Terminal Retention

- continuation source: `continuation_f160_f162_post_h4000_source_floor_oldshape_full`
- Trigger: F157-F159 still left h4000->h4800 erosion below the 0.50 retention gate. This continuation tests source-floor, h4000-anchor floor, and decay-aware source floor using only train-stream source/support/corrupt telemetry; it is not a pure freeze/hold candidate and does not use h4800 outcomes as selector input.
- best post-h4000 source-floor candidate: `MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor` h4800=0.14896754423777261, h4800_retention_ratio=0.43010448760558817

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F160-early100-h800-source-slow-ema-terminal-source-floor | 9 | 0.5022448930475447 | 0.32949671480390763 | 0.21352261635992262 | 0.13233332501517403 | 0.401622593093012 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8472568038374833 | 0.1527431961625167 | 0.7640379425376915 | 0.0466360229033011 | 1 |
| MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor | 9 | 0.4894979265001085 | 0.31664108567767674 | 0.20062582360373604 | 0.1193248364660475 | 0.3768457154278255 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8472802445378097 | 0.15271975546219033 | 0.7337725218902205 | 0.06322628259658813 | 1 |
| MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor | 9 | 0.5192443165514204 | 0.3463519877857632 | 0.23032606972588432 | 0.14896754423777261 | 0.43010448760558817 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8451352345104649 | 0.1548647654895346 | 0.7660406319179383 | 0.0498178987591355 | 1 |

## Part 11 F163-F165 Raw-Guard Post-H4000 Source-Floor Retention

- continuation source: `continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full`
- Trigger: F160-F162 did not improve over F154 and remained far below the v22.02 F118 near miss. This continuation keeps the F118-style terminal raw guard before h4000, then switches only the post-h4000 terminal phase to source-floor, h4000-anchor floor, or decay-aware floor. It does not freeze/hold training and does not use h4800 outcomes as selector input.
- best raw-guard post-h4000 floor candidate: `MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor` h4800=0.14042934444215563, h4800_retention_ratio=0.4158218364598758

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor | 9 | 0.5037513275941213 | 0.3309069673220317 | 0.2149040632777744 | 0.1336647007200453 | 0.40393438011223753 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8462234900697773 | 0.1537765099302227 | 0.7886848472136649 | 0.04740508838936135 | 1 |
| MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor | 9 | 0.4942131307390001 | 0.32138868504100376 | 0.2053836915228102 | 0.12409891353713141 | 0.3861334244586068 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8480293172586931 | 0.15197068274130662 | 0.7547815527820396 | 0.04461463513197722 | 1 |
| MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor | 9 | 0.5105579230520461 | 0.33771517541673446 | 0.2217154105504354 | 0.14042934444215563 | 0.4158218364598758 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.845348612967717 | 0.15465138703228332 | 0.7493082162883788 | 0.05488010357927393 | 1 |

## Part 12 F166-F168 Anti-Erosion Orthogonal / Reflection / Transport

- continuation source: `continuation_f166_f168_anti_erosion_oldshape_full`
- Trigger: F163-F165 still left post-h4000 erosion below the 0.50 retention gate. This continuation tests a different train-only hypothesis: late erosion is caused by train-stream gradient components pointing against the retained source channel, so the terminal route removes, reflects, or anchor-transports only the anti-source component after h4000. It does not use h4800 outcomes as selector input.
- best anti-erosion candidate: `MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector` h4800=0.13835234443346658, h4800_retention_ratio=0.41220559434526777

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal | 9 | 0.5012382169564565 | 0.32838569084803265 | 0.2123797701464759 | 0.1311053368780348 | 0.39924192963300137 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8458474243127015 | 0.15415257568729848 | 0.7891255871406042 | 0.04183958967526754 | 1 |
| MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard | 9 | 0.4899744722578261 | 0.31706199381086564 | 0.20103775130377877 | 0.11969963047239515 | 0.377527527136534 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8466339954863238 | 0.15336600451367643 | 0.7499912463374154 | 0.05804484972247371 | 1 |
| MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector | 9 | 0.5084546340836419 | 0.33563917213016087 | 0.2196390794383155 | 0.13835234443346658 | 0.41220559434526777 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8459017076157743 | 0.15409829238422576 | 0.7493516470949282 | 0.04976061759171663 | 1 |

## Part 13 F169-F171 H3200-Anchor Transport

- continuation source: `continuation_f169_f171_h3200_anchor_transport_oldshape_full`
- Trigger: F166-F168 showed the post-h4000 anti-source gradient component was mostly zero, so this continuation changes the hypothesis from anti-gradient removal to transporting the terminal route toward the train-stream h3200 source anchor. It still does not use h4800 outcomes as selector input.
- best h3200-anchor transport candidate: `MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport` h4800=0.14096160067452324, h4800_retention_ratio=0.41673011176809377

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport | 9 | 0.511077708668179 | 0.3382563359207577 | 0.22225730948978 | 0.14096160067452324 | 0.41673011176809377 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8457259236223104 | 0.15427407637769008 | 0.7685580068240747 | 0.05793385704358419 | 1 |
| MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport | 9 | 0.49639925691816544 | 0.32358843088150024 | 0.2075935403505961 | 0.12629363934199014 | 0.3902909600258222 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8477402066549931 | 0.15225979334500697 | 0.7505030853366973 | 0.05060151992020784 | 1 |
| MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry | 9 | 0.5079161193635728 | 0.3351471159193251 | 0.21916387809647453 | 0.13790216710832384 | 0.411467563222352 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8442074685513431 | 0.15579253144865657 | 0.7724989882239538 | 0.04596102679217303 | 1 |

## Part 14 F172-F174 Source-Preserving Progress Carry

- continuation source: `continuation_f172_f174_progress_carry_oldshape_full`
- Trigger: F154/F166/F169 decomposition showed the candidate terminal loss is nearly flat after h3200/h4000, while the matched `CTRL-SGD` best-control continues improving; the retained-source ratio therefore falls because the control baseline moves, not only because the source channel is actively erased. This continuation lets the terminal phase carry train-stream task progress while preserving the h3200/h4000 source anchor.
- F172 uses h3200 progress carry, F173 uses h4000 progress carry, and F174 blends source-projected progress with the h3200 source anchor. None use h4800 outcomes as selector input.
- best progress-carry candidate: `MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry` h4800=0.13602299822701347, h4800_retention_ratio=0.4079885425908324

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry | 9 | 0.5062744286325243 | 0.33339906401104397 | 0.21735644340515137 | 0.13602299822701347 | 0.4079885425908324 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8460533593967046 | 0.15394664060329535 | 0.7798860667062827 | 0.04319954139214975 | 1 |
| MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry | 9 | 0.4939001467492845 | 0.3211288054784139 | 0.20513932572470772 | 0.12389977110756768 | 0.3858257776750462 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8497433737110599 | 0.1502566262889399 | 0.7305065617701554 | 0.07023691027252763 | 1 |
| MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend | 9 | 0.5021474328305986 | 0.32932109302944607 | 0.21333535843425327 | 0.13201194339328343 | 0.40086088072554665 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8454315468522249 | 0.15456845314777515 | 0.7773578898567368 | 0.056218087673187256 | 1 |

## Part 15 F175-F177 Raw-Guard Gentle Progress

- continuation source: `continuation_f175_f177_gentle_progress_oldshape_full`
- Trigger: F172-F174 confirmed naive progress carry lowers source/validation while only slightly improving train loss. This continuation follows the plan fallback for h3200 source drop: attach later and weaker, keep raw-guard source geometry, and allow progress only through train-split lookahead-positive alt steps; non-alt terminal steps hold instead of running plain SGD.
- F175 attaches gentle progress at h3600, F176 at h4000, and F177 at h4400 with projected progress. None use h4800 outcomes as selector input.
- best gentle-progress candidate: `MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress` h4800=0.13225279251734415, h4800_retention_ratio=0.4013371874988695

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress | 9 | 0.5023462639914619 | 0.32953037155999076 | 0.21354025602340698 | 0.13225279251734415 | 0.4013371874988695 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8464149092036604 | 0.15358509079634006 | 0.7699975053851741 | 0.04786238736576504 | 1 |
| MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress | 9 | 0.49172060357199776 | 0.31885785195562577 | 0.20284747083981833 | 0.12152753935919867 | 0.3811339084605989 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8485383059081852 | 0.15146169409181454 | 0.7440777830877576 | 0.051928300548482825 | 1 |
| MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress | 9 | 0.5003823671076033 | 0.3276138967937893 | 0.21163182126151192 | 0.13036236498090956 | 0.39791463749464756 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8445587807931546 | 0.1554412192068456 | 0.7616906772541524 | 0.0479197104771932 | 1 |

## Part 16 F178-F180 Control-Relative Terminal Catch-Up

- continuation source: `continuation_f178_f180_control_relative_catchup_oldshape_full`
- Trigger: F172-F177 showed that source-preserving progress or late gentle progress still fails to catch the matched-control terminal improvement. This continuation tests a control-relative terminal objective using only train split A/B immediate loss gain and corrupt-label veto, not validation/test/future/h4800 outcome.
- F178 uses SGD catch-up, F179 uses AdamW catch-up, and F180 uses a source-balanced catch-up direction. All three keep the unchanged productive h4800 gate.
- best control-relative candidate: `MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup` h4800=0.13699083195792305, h4800_retention_ratio=0.4097839293516009

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup | 9 | 0.5071572495831383 | 0.33430015709665084 | 0.21828498111830819 | 0.13699083195792305 | 0.4097839293516009 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8456547330084093 | 0.15434526699159046 | 0.7732224860853152 | 0.049036623151214036 | 1 |
| MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup | 9 | 0.48276226388083565 | 0.30988823042975533 | 0.19372723499933878 | 0.11134498649173313 | 0.3593069228131674 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8482363886193425 | 0.15176361138065797 | 0.761987777770716 | 0.05068557129965888 | 1 |
| MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup | 9 | 0.5017823113335503 | 0.3289473189247979 | 0.2129563887914022 | 0.13166724310980904 | 0.4002684792816629 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8458978426932533 | 0.15410215730674678 | 0.7746061313348881 | 0.0473393268055386 | 1 |

## Part 17 F181-F183 Trajectory-Adaptive Terminal Retention

- continuation source: `continuation_f181_f183_trajectory_adaptive_oldshape_full`
- Trigger: F154/F178 localization shows mixed failure modes: some rows lose too much between h1600 and h3200, while others keep continuous h3200 but fail the h4800/h3200 ratio. This continuation uses only reached train-stream source anchors (h1600/h2400/h3200/h4000), split A/B one-step gain, and corrupt-label veto to choose a mid-erosion bridge or post-h4000 ratio repair.
- F181 uses trajectory-adaptive preserve, F182 emphasizes mid-erosion bridging before h3200, and F183 uses a two-phase mid bridge plus post-h4000 ratio repair. None read dataset name, validation/test/future, or h4800 outcome as selector input.
- best trajectory-adaptive candidate: `MLP-F181-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve` h4800=0.1249746216668023, h4800_retention_ratio=0.38785703646301367

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F181-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve | 9 | 0.4950275421142578 | 0.3222182657983568 | 0.2062396009763082 | 0.1249746216668023 | 0.38785703646301367 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8632455175733009 | 0.13675448242669938 | 0.7336622679927362 | 0.04466764794455634 | 1 |
| MLP-F182-early100-h800-source-slow-ema-terminal-mid-erosion-bridge | 9 | 0.4844011134571499 | 0.31163445446226334 | 0.1956859827041626 | 0.11442835463417901 | 0.36718775153289557 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8659719634416023 | 0.1340280365583975 | 0.7212270995909199 | 0.04691829946306017 | 1 |
| MLP-F183-early100-h800-source-slow-ema-terminal-two-phase-ratio-repair | 9 | 0.4851308796140883 | 0.3123616311285231 | 0.1963998344209459 | 0.11514037847518921 | 0.3686124254736459 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8649232552413719 | 0.13507674475862813 | 0.7179390559203329 | 0.04876340870504026 | 1 |

## Part 18 F184-F186 Minimal Terminal Transport / Hold

- continuation source: `continuation_f184_f186_minimal_transport_oldshape_full`
- Trigger: F181-F183 preserved early/h3200 but worsened post-h4000 retention relative to F154, suggesting terminal intervention was too disruptive rather than too weak. This continuation keeps the F154-style debt-aware preserve before h4000, then tests minimal train-stream post-h4000 transport.
- F184 clips only the train-stream anti-source component, F185 holds post-h4000 unless debt or anti-source evidence requires a tiny clip, and F186 applies a very small h3200/h4000 anchor flow. None use dataset name, validation/test/future, or h4800 outcome as selector input.
- best minimal-transport candidate: `MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny` h4800=0.11503769954045613, h4800_retention_ratio=0.36831442816645493

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip | 9 | 0.48026986916859943 | 0.3074439631568061 | 0.19144247637854683 | 0.11015128427081639 | 0.35828084942632543 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8639820990573401 | 0.13601790094266014 | 0.7353248708220786 | 0.044666814583319205 | 1 |
| MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold | 9 | 0.4808646109369066 | 0.30804969867070514 | 0.1920506093237135 | 0.110755721728007 | 0.359538484231407 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.865587103204565 | 0.13441289679543503 | 0.7292126385876208 | 0.04612600913754216 | 1 |
| MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny | 9 | 0.48516794045766193 | 0.3123355772760179 | 0.19633140166600546 | 0.11503769954045613 | 0.36831442816645493 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8631717072417273 | 0.13682829275827268 | 0.7204992304277305 | 0.04497633598468922 | 1 |

## Part 19 F187-F189 Terminal Accept-Memory

- continuation source: `continuation_f187_f189_accept_memory_oldshape_full`
- Trigger: F184-F186 fixed the runner wiring but the minimal post-h4000 transport worsened relative to F154, while F154 raw traces showed many successful train-stream terminal accepts before later noisy rejects. This continuation changes the retained-target theory: source observability may require persisting the last train-only accepted terminal transport instead of re-solving a noisy one-step target at every terminal horizon.
- F187 replays the last accepted terminal transport if the current train split A/B and corrupt-label gate still permits it, F188 adds a debt-aware transport candidate, and F189 stores a source-progress transport memory. None read dataset name, validation/test/future, or h4800 outcome as selector input.
- best accept-memory candidate: `MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory` h4800=0.12118528948889838, h4800_retention_ratio=0.3805887480765032

| v22_id | rows | h800 | h3200 | h4000 | h4800 | h4800_retention_ratio | row_h4800_positive_count | early_source_chain_group | continuous_h3200_group | productive_h4800_group | terminal_erosion_group | terminal_erosion_class | source_chain_blocker | source_hidden_fraction_mean | source_readout_fraction_mean | source_norm_over_whitened_norm_mean | source_projection_fold_proxy_mean | diffeomorphic_audit_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | 9 | -0.0035047531127929688 | -0.8993779950671725 | -1.1315686371591356 | -1.3196280201276143 |  | 0 | 0 | 0 | 0 | 0 | PostH3200PostH4000Erosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8413753468907773 | 0.1586246531092226 |  |  | 1 |
| CTRL-NoOpMatchedOverhead | 9 | -1.1151884065734015 | -1.2880509429507785 | -1.4040647149085999 | -1.4853768944740295 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |  |  |  |  | 1 |
| CTRL-RandomMatchedNorm | 9 | -1.151018136077457 | -1.322744369506836 | -1.4386262430085077 | -1.5203083157539368 |  | 0 | 0 | 0 | 0 | 0 | MeasuredMonotoneTerminalErosion | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.8939609212558122 | 0.10603907874418794 |  |  | 1 |
| CTRL-SGD | 9 | -0.6329910821384854 | 0.0 | 0.0 | 0.0 |  | 0 | 0 | 0 | 0 | 0 | MeasuredTerminalErosionUnresolved | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing | 0.7638181842520056 | 0.23618181574799452 |  |  | 1 |
| MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory | 9 | 0.4912160303857591 | 0.3184153238932292 | 0.20244564612706503 | 0.12118528948889838 | 0.3805887480765032 | 8 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8632661577424878 | 0.13673384225751212 | 0.7115915341655845 | 0.043775975704193115 | 1 |
| MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt | 9 | 0.4851671059926351 | 0.3123748103777568 | 0.19639780786302355 | 0.11512987481223212 | 0.3685632483394063 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8665083433082637 | 0.1334916566917361 | 0.7059520043455255 | 0.041568121424427736 | 1 |
| MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory | 9 | 0.4897601008415222 | 0.31693363189697266 | 0.20092393292321098 | 0.11964407232072619 | 0.37750513129392194 | 9 | 1 | 1 | 0 | 1 | MeasuredMonotoneTerminalErosion | TerminalErosion;PostH4000Erosion | 0.8640555273094254 | 0.1359444726905748 | 0.6984234513661863 | 0.042845455584702666 | 1 |

## Part 20 Dataset Localization

| v22_id | dataset | rows | mean_h800 | mean_h3200 | mean_h4000 | mean_h4800 | h4800_positive_rows | early_rows | continuous_rows | h4800_rows | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | Fashion-MNIST | 3 | -0.010514259338378906 | -1.710185964902242 | -1.9524550040562947 | -2.165576179822286 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | KMNIST | 3 | 0.0 | -0.9298243920008341 | -1.168380339940389 | -1.3493971427281697 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | MNIST | 3 | 0.0 | -0.05812362829844157 | -0.2738705674807231 | -0.44391073783238727 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 3 | -1.0693123737970989 | -1.4528566598892212 | -1.5115971167882283 | -1.5512685378392537 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | KMNIST | 3 | -0.7241322994232178 | -1.068701942761739 | -1.1966933806737263 | -1.2789833943049114 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | MNIST | 3 | -1.5521205464998882 | -1.3425942262013753 | -1.5039036472638447 | -1.6258787512779236 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | Fashion-MNIST | 3 | -1.034249742825826 | -1.4170781373977661 | -1.475982626279195 | -1.516519824663798 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | KMNIST | 3 | -0.8193213144938151 | -1.162659764289856 | -1.2894699176152546 | -1.3718656301498413 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | MNIST | 3 | -1.59948335091273 | -1.3884952068328857 | -1.550426185131073 | -1.672539492448171 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | Fashion-MNIST | 3 | -0.28509851296742755 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | KMNIST | 3 | -0.3919684886932373 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | MNIST | 3 | -1.2219062447547913 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong | Fashion-MNIST | 3 | 0.5561087330182394 | 0.1727412541707357 | 0.11406606435775757 | 0.07470029592514038 | 3 | 2 | 0 | 0 | ContinuousH3200Missing |
| MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong | KMNIST | 3 | 0.6669755379358927 | 0.32231974601745605 | 0.19429582357406616 | 0.11186595757802327 | 2 | 2 | 1 | 0 | ContinuousH3200Missing |
| MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong | MNIST | 3 | 0.27849648396174115 | 0.4880295991897583 | 0.32672152916590375 | 0.20475009083747864 | 3 | 2 | 2 | 0 | TerminalErosion;PostH4000Erosion |
| MLP-F146-early100-h800-source-slow-ema-terminal-source-preserve-gentle | Fashion-MNIST | 3 | 0.5511945684750875 | 0.1679924726486206 | 0.10935495297114055 | 0.06997174024581909 | 3 | 2 | 0 | 0 | ContinuousH3200Missing |
| MLP-F146-early100-h800-source-slow-ema-terminal-source-preserve-gentle | KMNIST | 3 | 0.6786378423372904 | 0.3340454697608948 | 0.20603811740875244 | 0.12370141347249348 | 3 | 2 | 1 | 0 | ContinuousH3200Missing |
| MLP-F146-early100-h800-source-slow-ema-terminal-source-preserve-gentle | MNIST | 3 | 0.24053287506103516 | 0.4500470459461212 | 0.2887353003025055 | 0.16675438483556113 | 3 | 2 | 2 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | Fashion-MNIST | 3 | 0.5337667465209961 | 0.15041484435399374 | 0.09176162878672282 | 0.05231614907582601 | 3 | 1 | 0 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | KMNIST | 3 | 0.7323490977287292 | 0.38776687781016034 | 0.259765883286794 | 0.17743519941965738 | 3 | 3 | 1 | 0 | TerminalErosion;PostH4000Erosion |
| MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard | MNIST | 3 | 0.2705545524756114 | 0.4800918499628703 | 0.31878509124120075 | 0.1968232293923696 | 3 | 1 | 1 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | Fashion-MNIST | 3 | -0.010514259338378906 | -1.710185964902242 | -1.9524550040562947 | -2.165576179822286 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | KMNIST | 3 | 0.0 | -0.9298243920008341 | -1.168380339940389 | -1.3493971427281697 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | MNIST | 3 | 0.0 | -0.05812362829844157 | -0.2738705674807231 | -0.44391073783238727 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 3 | -1.0693123737970989 | -1.4528566598892212 | -1.5115971167882283 | -1.5512685378392537 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | KMNIST | 3 | -0.7241322994232178 | -1.068701942761739 | -1.1966933806737263 | -1.2789833943049114 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | MNIST | 3 | -1.5521205464998882 | -1.3425942262013753 | -1.5039036472638447 | -1.6258787512779236 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | Fashion-MNIST | 3 | -1.034249742825826 | -1.4170781373977661 | -1.475982626279195 | -1.516519824663798 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | KMNIST | 3 | -0.8193213144938151 | -1.162659764289856 | -1.2894699176152546 | -1.3718656301498413 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | MNIST | 3 | -1.59948335091273 | -1.3884952068328857 | -1.550426185131073 | -1.672539492448171 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | Fashion-MNIST | 3 | -0.28509851296742755 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | KMNIST | 3 | -0.3919684886932373 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | MNIST | 3 | -1.2219062447547913 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | Fashion-MNIST | 3 | 0.5609042247136434 | 0.1773891051610311 | 0.1186624566713969 | 0.07901098330815633 | 3 | 2 | 0 | 0 | ContinuousH3200Missing |
| MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | KMNIST | 3 | 0.7037384708722433 | 0.35917965571085614 | 0.23118712504704794 | 0.14890193939208984 | 3 | 2 | 1 | 0 | TerminalErosion;PostH4000Erosion |
| MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target | MNIST | 3 | 0.27526797850926715 | 0.4847915569941203 | 0.3234818975130717 | 0.20150761802991232 | 3 | 1 | 1 | 0 | EarlySourceChainMissing;ContinuousH3200Missing |

_仅显示前 36 / 315 rows；完整 CSV 见 artifact。_

### Row Localization

| v22_id | dataset | seed | source_h800 | source_h3200 | source_h4000 | source_h4800 | v22_03_early_source_chain | v22_03_continuous_h3200_chain | v22_03_productive_h4800_chain | v22_03_terminal_erosion | v22_03_source_chain_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-AdamW | MNIST | 0 | 0.0 | -0.007803916931152344 | -0.19305634498596191 | -0.3298472762107849 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | Fashion-MNIST | 1 | -0.03154277801513672 | -1.950606346130371 | -2.180308520793915 | -2.434894323348999 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | KMNIST | 2 | 0.0 | -1.2602074146270752 | -1.5024124383926392 | -1.6839063167572021 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | Fashion-MNIST | 0 | 0.0 | -1.1179918050765991 | -1.3086252808570862 | -1.4437002539634705 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | KMNIST | 1 | 0.0 | -0.37587225437164307 | -0.6091593503952026 | -0.7797720432281494 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | MNIST | 2 | 0.0 | -0.1121058464050293 | -0.36713385581970215 | -0.5956544876098633 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | KMNIST | 0 | 0.0 | -1.1533935070037842 | -1.3935692310333252 | -1.5845130681991577 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | MNIST | 1 | 0.0 | -0.054461121559143066 | -0.2614215016365051 | -0.40623044967651367 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-AdamW | Fashion-MNIST | 2 | 0.0 | -2.061959743499756 | -2.3684312105178833 | -2.6181339621543884 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | MNIST | 2 | -1.4920803308486938 | -1.3033781051635742 | -1.4974815845489502 | -1.6733428239822388 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | KMNIST | 0 | -0.5388028621673584 | -1.045372724533081 | -1.1672396659851074 | -1.2496076822280884 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | MNIST | 1 | -1.5316681265830994 | -1.3284789323806763 | -1.4813163876533508 | -1.5824309587478638 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 2 | -1.0034855604171753 | -1.4049808979034424 | -1.4927996397018433 | -1.5519575476646423 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | MNIST | 0 | -1.632613182067871 | -1.3959256410598755 | -1.5329129695892334 | -1.6218624711036682 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 1 | -0.9768359661102295 | -1.488898515701294 | -1.5205432772636414 | -1.545022964477539 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | KMNIST | 2 | -0.5143189430236816 | -1.0255897045135498 | -1.1288996934890747 | -1.1868820190429688 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | Fashion-MNIST | 0 | -1.2276155948638916 | -1.4646905660629272 | -1.5214484333992004 | -1.5568251013755798 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-NoOpMatchedOverhead | KMNIST | 1 | -1.1192750930786133 | -1.1351433992385864 | -1.293940782546997 | -1.4004604816436768 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | MNIST | 1 | -1.7893840670585632 | -1.5827230215072632 | -1.73803049325943 | -1.8398717641830444 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | Fashion-MNIST | 2 | -0.9035712480545044 | -1.3079161643981934 | -1.3963288068771362 | -1.4564520716667175 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | MNIST | 0 | -1.5321521759033203 | -1.2955232858657837 | -1.433678150177002 | -1.5213847756385803 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | Fashion-MNIST | 1 | -0.817002534866333 | -1.3282618522644043 | -1.3605114817619324 | -1.3850362300872803 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | KMNIST | 2 | -0.6747612953186035 | -1.182074785232544 | -1.2826379537582397 | -1.3410465717315674 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | Fashion-MNIST | 0 | -1.3821754455566406 | -1.6150563955307007 | -1.6711075901985168 | -1.7080711722373962 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | KMNIST | 1 | -1.1916522979736328 | -1.2080684900283813 | -1.3670737743377686 | -1.4736433029174805 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | MNIST | 2 | -1.4769138097763062 | -1.2872393131256104 | -1.479569911956787 | -1.6563619375228882 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-RandomMatchedNorm | KMNIST | 0 | -0.591550350189209 | -1.0978360176086426 | -1.2186980247497559 | -1.300907015800476 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | Fashion-MNIST | 0 | -0.4896559715270996 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | KMNIST | 1 | -0.787503719329834 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | MNIST | 2 | -1.0834238529205322 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | KMNIST | 0 | -0.2065906524658203 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | MNIST | 1 | -1.2798187136650085 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | Fashion-MNIST | 2 | -0.3656395673751831 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | MNIST | 0 | -1.302476167678833 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | Fashion-MNIST | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |
| CTRL-SGD | KMNIST | 2 | -0.18181109428405762 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | ControlEquivalent;EarlySourceChainMissing;ContinuousH3200Missing |

_仅显示前 36 / 945 rows；完整 CSV 见 artifact。_

## Part 21 KAN Source-Channel Writer

| carrier | v22_id | writer_family | h800 | h3200 | h4800 | h4800_positive_rows | KAN_FU_S3_v22_02 | v22_03_source_channel_writer_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | F36-lowbank-loss-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -1.6258274449242487 | -1.515865898794598 | -1.412504815393024 | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | F38-gain-gated-lowbank-loss-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -1.6219656997256808 | -1.5150579777028825 | -1.4138417343298595 | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | F68-adamw-boundary-to-gated-lowbank-b3-null | D-CHE-F5b-low-degree-low-frequency-bank | -0.0005275342199537489 | -0.0021484394868214927 | -0.007300949758953518 | 3 | 0 | KANSourceChannelMismatch |
| D-CHE | F69-adamw-boundary-lowbank-anchor-antiwashout | D-CHE-F5b-low-degree-low-frequency-bank | 0.0014568898412916395 | -0.013673616780175103 | -0.015228016508950127 | 4 | 0 | KANSourceChannelMismatch |
| D-CHE | KSW1-basis-estimate-readout-commit | D-CHE-F5a-readout-commit | -1.6231121751997206 | -1.5118636190891266 | -1.410119225581487 | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | KSW2-lowdegree-lowfreq-source-bank | D-CHE-F5b-low-degree-low-frequency-bank | -0.18236266242133248 | -0.005438006586498684 | 0.027257995473013982 | 4 | 0 | KANSourceChannelMismatch |
| D-FOU | F36-lowbank-loss-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | -1.0049330062336392 | 0.049605203999413386 | 0.15526423851648966 | 9 | 0 | KANSourceChannelMismatch |
| D-FOU | F38-gain-gated-lowbank-loss-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | -1.2923956910769145 | -0.06800315777460735 | 0.13637220197253758 | 8 | 0 | KANSourceChannelMismatch |
| D-FOU | F68-adamw-boundary-to-gated-lowbank-b3-null | D-FOU-F5b-low-degree-low-frequency-bank | 0.011607607205708822 | -0.004651433891720242 | -0.005101071463690864 | 2 | 0 | KANSourceChannelMismatch |
| D-FOU | F69-adamw-boundary-lowbank-anchor-antiwashout | D-FOU-F5b-low-degree-low-frequency-bank | 0.0022645857599046496 | -0.01170006725523207 | -0.01502860254711575 | 3 | 0 | KANSourceChannelMismatch |
| D-FOU | KSW1-basis-estimate-readout-commit | D-FOU-F5a-readout-commit | -1.536535296175215 | -1.4180500507354736 | -1.345011830329895 | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | KSW2-lowdegree-lowfreq-source-bank | D-FOU-F5b-low-degree-low-frequency-bank | -0.1767431596914927 | -0.10215700334972805 | -0.13887516657511392 | 1 | 0 | KANSourceChannelMismatch |

## 修改记录

- 新增 `dgkan/fu/terminal_retention.py`：实现 v22.03 productive h4800 / terminal erosion gate 和 measured terminal-erosion taxonomy。
- 新增 `dgkan/fu/diffeomorphic_target.py`：提供 source-channel information-volume / fold proxy audit helper；本轮只作为审计 helper，不把它当成功证据。
- 扩展 `dgkan/fu/mechanisms.py`、`experiments/run_v21_common.py`、`experiments/run_v21_01_source_retention.py`、`experiments/run_v17_common.py`：新增 M170/F145 source-preserve-strong、M171/F146 source-preserve-gentle、M172/F147 info-volume guard，并接入 h4000 后 train-only source-preserving terminal route。
- 继续按计划 C2 扩展 M173/F148、M174/F149、M175/F150 以及 target-only M176/M177/M178；在 `experiments/run_v17_common.py` 中记录 ActuationR2、B2/B3、local distortion、fold/order-flip 等 train-stream target-gate telemetry。
- C2 仍未达成后，按计划 C1 fallback 新增 M179/F151 very-strong source preservation、M180/F152 h3600 terminal source preservation、M181/F153 Nora-style row-orthogonal source update；三者都保持 train-only gate，不使用 validation/test/future/dataset-name 分支。
- F151-F153 仍未达成后，继续按计划 C1 剩余机制新增 M182/F154 debt-aware terminal preserve、M183/F155 low-NDS matrix-block source、M184/F156 dual-memory terminal preserve；不使用 h4800 outcome 训练 selector。
- F154-F156 仍未达成后，按计划 3.1.1 允许重开 PopRisk/SNR 的受限语义，新增 M185/F157 terminal SNR predictor、M186/F158 split-consensus estimator、M187/F159 signal-reservoir transport；三者只把 train-stream signal/corrupt readback 作为 terminal gate/estimator，不作为直接参数 mask 或 h4800 outcome selector。
- F157-F159 仍未达成后，基于 post-h4000 erosion 证据新增 M188/F160 terminal source floor、M189/F161 h4000-anchor source floor、M190/F162 decay-aware source floor；三者不做 pure freeze/hold，不读取 validation/test/future/h4800 outcome，只用 train-stream source/support/corrupt telemetry gate。
- F160-F162 仍未改善后，回到 v22.02 最接近的 F118 raw-guard 证据链，新增 M191/F163 raw-guard source floor、M192/F164 raw-guard h4000-anchor floor、M193/F165 raw-guard decay-aware floor：h4000 前沿用 raw guard，h4000 后才切到 train-stream source floor。
- F163-F165 仍未达到 productive h4800 后，新增 M194/F166 anti-erosion orthogonal、M195/F167 source-reflection guard、M196/F168 h4000 transport corrector：只在 h4000 后对 train-stream anti-source gradient component 做移除/反射/anchor transport，不读取 h4800 outcome。
- F166-F168 仍未达到 productive h4800 且 anti-source component 基本为 0 后，新增 M197/F169 h3200-anchor transport、M198/F170 raw-guard+h3200-anchor transport、M199/F171 h3200-ratio reentry：用 train-stream h3200 source anchor 作为 post-h4000 transport target，不读取 h4800 outcome。
- F169-F171 仍未达到 productive h4800 后，对 F154/F166/F169 做 terminal source/control decomposition，发现 h3200/h4000 之后 candidate 自身 loss 近乎持平，而 `CTRL-SGD` best-control 继续改善；据此新增 M200/F172 h3200 progress carry、M201/F173 h4000 progress carry、M202/F174 h3200 source-progress blend：在保留 source anchor 的同时允许 train-stream task-progress update 继续推进。
- F172-F174 仍未达到 productive h4800，且 F172 只改善 train loss、不改善 validation/source 后，新增 M203/F175 raw-guard h3600 gentle progress、M204/F176 raw-guard h4000 gentle progress、M205/F177 raw-guard h4400 projected progress：按计划 fallback 将 terminal attach 变晚/变轻，非 alt terminal step 不再普通 SGD。
- F175-F177 仍未达到 productive h4800 后，新增 M206/F178 control-relative SGD catch-up、M207/F179 control-relative AdamW catch-up、M208/F180 source-balanced catch-up：只用 train split A/B one-step gain 与 corrupt-label veto 判断 terminal update 是否可提交，不读取 h4800 outcome。
- F178-F180 仍未达到 productive h4800 后，新增 M209/F181 trajectory-adaptive preserve、M210/F182 mid-erosion bridge、M211/F183 two-phase ratio repair：按已到达的 train-stream source anchor（h1600/h2400/h3200/h4000）选择 mid-erosion bridge 或 post-h4000 ratio repair，不读取 dataset-name、validation/test/future 或 h4800 outcome。
- F181-F183 full 显示 trajectory-adaptive terminal route 比 F154 更弱后，新增 M212/F184 anti-source clip、M213/F185 debt-aware hold、M214/F186 anchor-flow-tiny：h4000 前沿用 F154-style debt-aware preserve，h4000 后只做极小 train-stream anti-source clipping/anchor flow 或 hold，不读取 h4800 outcome。
- F184-F186 smoke 暴露两处 runner wiring 问题后，修复 M212/M213/M214 在 `experiments/run_v17_common.py` 的 source-retention 主 allowlist 与 early active-source mode 列表；两次 pre-wiring smoke 分别保留为 `continuation_f184_f186_minimal_transport_smoke_pre_front_wiring_fix` 与 `continuation_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix`，不作为科学成功证据。
- F184-F186 仍未达到 productive h4800 后，基于 F154/F181/F186 raw trace 诊断新增 M215/F187 accept-memory、M216/F188 accept-memory-debt、M217/F189 source-progress-memory：只持久化 train split A/B 与 corrupt-label gate 接受过的 terminal transport，h4000 后 replay 仍需当前 train-only gate 通过，不读取 dataset-name、validation/test/future 或 h4800 outcome。
- 新增 v22.03 runner：code truth gate、efficiency readback、D-RAT/D-RBF active micro-kernel repair、terminal erosion autopsy、source-preservation aggregation、KAN readback、finalizer。
- 将 `experiments/run_v22_03_drat_drbf_repair.py` 从 readback-only 扩展为 active micro-kernel benchmark：覆盖 D-RAT Horner/reciprocal/telemetry-free/low-degree rational 和 D-RBF local-k/no-dense/active-center/exp-approx/sparse-backward 方向；输出 component waterfall、gradcheck、finite-rate、memory/materialization 和 micro-near-E1 判定，但不把 micro-kernel 结果伪称为 official fused 或 functional runner 证据。
- 新增 `experiments/run_v22_03_drat_drbf_runner_integration.py`：对 D-RAT/D-RBF 的 micro-near-E1 变体执行 limited same-kernel runner integration，验证 CUDA train loop 中有限 loss、非零梯度、shape 一致、无 CPU offload 和同 callable hash；该证据只移除 functional_runner_kernel_mismatch，不把 `official_fused_kernel_complete=0` 伪称为 promotion。
- 新增 `experiments/run_v22_03_drat_drbf_limited_smoke_summary.py`：汇总 D-RAT/D-RBF existing-carrier limited functional smoke；该 smoke 只读 `run_v21_01_source_retention.py` 的现有 PrimitiveKAN carrier 路径，不把结果伪称为 active micro-kernel production runner 或 official fused kernel。
- 扩展 `experiments/run_v21_common.py` / `experiments/run_v17_common.py`：注册计划指定的 RAT-FU1/FU2/FU3 与 RBF-FU1/FU2/FU3 C4 smoke aliases，并将 RBF22.03 compact/active-center repair label 映射到已有 `rbf_k*_triton_l3_matmul` no-dense path；D-RAT 仍不伪称 production official fused rational path 已接入。
- 执行 F145-F147 smoke 后，运行 4GPU old-shape full：train_size=512、val_size=256、batch_size=64、steps=6400、MNIST/Fashion-MNIST/KMNIST x seed0/1/2、F145-F147 + matched controls。
- F145-F147 没有达到 productive h4800 后，继续执行 F148-F150 h1600/h4200 smoke 和 4GPU old-shape full；所有数值来自落盘 `v21_01_source_retention_matrix.csv` 与 raw trace，不手写提升。
- F148-F150 仍没有达到 productive h4800 后，继续执行 F151-F153 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F151-F153 仍没有达到 productive h4800 后，继续执行 F154-F156 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F154-F156 仍没有达到 productive h4800 后，继续执行 F157-F159 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F157-F159 仍没有达到 productive h4800 后，继续执行 F160-F162 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F160-F162 仍没有达到 productive h4800 后，继续执行 F163-F165 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F163-F165 仍没有达到 productive h4800 后，继续执行 F166-F168 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F166-F168 仍没有达到 productive h4800 后，继续执行 F169-F171 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F169-F171 仍没有达到 productive h4800 后，继续执行 F172-F174 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F172-F174 仍没有达到 productive h4800 后，继续执行 F175-F177 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F175-F177 仍没有达到 productive h4800 后，继续执行 F178-F180 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F178-F180 仍没有达到 productive h4800 后，继续执行 F181-F183 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。
- F181-F183 仍没有达到 productive h4800 且低于 F154 后，继续执行 F184-F186 smoke；前两次 smoke 暴露 wiring 缺口并已留档，修复后 smoke 恢复 early source chain，再执行 4GPU old-shape full；所有正式数值来自修复后的 fresh matrix 与 raw trace。
- F184-F186 仍没有达到 productive h4800 后，继续执行 F187-F189 accept-memory smoke 和 4GPU old-shape full；所有正式数值来自该 continuation 的 fresh matrix 与 raw trace。

## 分析 / Insight / 结论

- v22.03 重新表述了 blocker：F118 之后的最好状态不是 h4800 变负的 terminal collapse，而是 h4800 仍为正但 retained ratio 未达 0.50 的 terminal erosion。
- C0 autopsy 只基于 measured h3200/h4000/h4800 trajectory 与 row-positive 证据分类；非因果指标不会被写成机制证明。
- F145-F147 是按计划 C1 的 source-preserving terminal route 做的真实修复尝试：不使用 validation/test/future，不使用 dataset-name branch，不修改 loss/sampler/class weight。
- C1 best 为 `MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard`，h4800_retention_ratio=0.41891942358555095；它保留 early/h3200 chain，但仍是 terminal erosion。
- 因 C1 未达 productive h4800，继续按计划执行 C2 diffeomorphic/source-channel target reset；C2 target telemetry 显示 F148/F150 的 target gate 全部 reject，F149 仅 1/54 次 accept，且该 accept 没有转化为 productive h4800，所以 high local actuation/readout observability 不能写成 retention success。
- C2 best 为 `MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target`，h4800_retention_ratio=0.4204398126876261；它略高于 C1 best，但仍低于 0.50 gate，也低于 v22.02 F118 near-miss ratio=0.4905266741203586。
- C1 fallback best 为 `MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve`，h4800_retention_ratio=0.4048600858991034；它检验 preservation strength、h3600 attach 与 row-orthogonal source update 是否能修复 C1/C2 的 terminal erosion。
- C1 fallback-2 best 为 `MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve`，h4800_retention_ratio=0.4323878091139131；它检验 debt-aware、low-NDS matrix-block 与 dual-memory source state 是否能修复 post-h4000 erosion。
- Signal-estimator best 为 `MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor`，h4800_retention_ratio=0.42377215113636235；它检验 PopRisk/SNR 是否能作为 train-only terminal-retention estimator 改善 source transport，而不是恢复旧的 one-shot parameter selector。
- Post-h4000 source-floor best 为 `MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor`，h4800_retention_ratio=0.43010448760558817；它检验 terminal erosion 是否能被低幅度 source floor / h4000 anchor floor 修复，同时避免 pure freeze/hold 造成的假保留。
- Raw-guard post-h4000 floor best 为 `MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor`，h4800_retention_ratio=0.4158218364598758；它检验 F118-style raw guard 近失误点是否能通过 h4000 后 source-floor repair 跨过 productive gate。
- Anti-erosion best 为 `MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector`，h4800_retention_ratio=0.41220559434526777；它检验后 h4000 anti-source gradient 是否是主要 erosion source，而不是继续调 source floor 强度。
- H3200-anchor transport best 为 `MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport`，h4800_retention_ratio=0.41673011176809377；它检验 h3200 时刻的 train-stream source geometry 是否比 h4000 anchor 更适合作为 terminal transport target。
- Progress-carry best 为 `MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry`，h4800_retention_ratio=0.4079885425908324；它检验 source-retention route 是否需要同时携带 train-stream task progress，避免 candidate flat 而 matched control 继续改善导致 retained-ratio 被动下滑。
- Gentle-progress best 为 `MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress`，h4800_retention_ratio=0.4013371874988695；它检验 F172 的失败是否来自 terminal progress 过早/过强，以及非 alt SGD 是否会破坏 retained source。
- Control-relative catch-up best 为 `MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup`，h4800_retention_ratio=0.4097839293516009；它检验 terminal objective 是否应显式追赶 matched-control 的 train-stream task progress，同时用 corrupt-label veto 避免伪进展。
- Trajectory-adaptive best 为 `MLP-F181-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve`，h4800_retention_ratio=0.38785703646301367；它检验同一 continuation 内的 row-state 异质性是否需要 train-only horizon-anchor 分流，而不是单一后段 source/SGD 强度。
- Minimal-transport best 为 `MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny`，h4800_retention_ratio=0.36831442816645493；它检验 F181 失败是否来自 terminal route 过强/扰动过多，改为 h4000 后 minimal clip/hold/tiny anchor flow。
- Accept-memory best 为 `MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory`，h4800_retention_ratio=0.3805887480765032；它检验 F154 raw trace 暗示的 last-good accepted terminal transport 是否比每个 terminal horizon 重新求 noisy one-step target 更稳定。
- F151-F189 已覆盖计划内的 C1/C2 fallback 与后续解释性修复：preservation strength、h3600 attach、row-orthogonal、debt-aware、low-NDS matrix-block、dual-memory、SNR estimator、source-floor、anti-erosion、h3200 transport、progress carry、gentle progress、control-relative catch-up、trajectory-adaptive horizon-anchor 分流、minimal anti-source clipping、post-h4000 hold/tiny anchor flow 与 train-only accepted terminal transport memory；若仍全部 productive_h4800_group=0，则继续同族小修缺少 plan-backed 因果依据，下一步需要新的 train-only retained-target/source-observability 定义，而不是再调 source amplitude、terminal hold 或 replay 强度。
- 本轮 overall best candidate 为 `MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve`，productive_h4800_group=0；因此不能 promotion，也不能启动 independent confirmation。
- D-RAT active repair 结果为 `MicroNearE1LimitedRunnerPassedOfficialFusedBlocked`：micro_near_E1_rows=13/14，best_forward_ratio=1.0529286035698215，best_step_ratio=1.2559891610405118；limited same-kernel runner 已通过，但 blocker 仍是 official_fused_missing。
- D-RBF active repair 结果为 `MicroNearE1LimitedRunnerPassedOfficialFusedBlocked`：micro_near_E1_rows=2/14，best_forward_ratio=1.8830231952045868，best_step_ratio=1.3595845386003278；active-center/local-k 方向降低了 materialization，limited same-kernel runner 已通过，但 blocker 仍是 official_fused_missing。
- D-RAT existing-carrier limited smoke 结果为 `ExistingCarrierSmokeNoH800Source`：best_h800_v21_id=`KSW9-h800-source-slow-ema-bank`，best_h800=-0.19887077808380127，best_h1600=-0.2576025724411011；best positive early row=`KSW9-h800-source-slow-ema-bank`，h100=0.018291950225830078，h400=0.04062372446060181。该 smoke 没有打开 h800 source，不能进入 functional promotion。
- D-RBF existing-carrier limited smoke 结果为 `ExistingCarrierSmokeNoH800Source`：best_h800_v21_id=`KSW2-lowdegree-lowfreq-source-bank`，best_h800=-0.09932762384414673，best_h1600=-0.015008985996246338；best positive early row=`KSW9-h800-source-slow-ema-bank`，h100=0.013185977935791016，h400=-0.05124133825302124。这说明 micro-near-E1 进展尚未迁移到现有 KAN source writer 路径。
- C4 named D-RAT smoke 结果为 `ExistingCarrierSmokeNoH800Source`：best_h800_v21_id=`RAT-FU3-numerator-only-source-writer`，best_h800=-0.33644765615463257，best_h1600=-0.5026246309280396；其 blocker 仍包含 official fused rational path 未接入 functional runner。
- C4 named D-RBF smoke 结果为 `ExistingCarrierSmokeNoH800Source`：best_h800_v21_id=`RBF-FU3-compact-local-source-writer`，best_h800=-0.2967539429664612，best_h1600=-0.3605765700340271；即使映射到已有 no-dense Triton RBF path，也没有达到 h800 source gate 时不能 promotion。
- D-CHE/D-FOU efficiency 继续作为 official efficient carrier 证据；D-RAT/D-RBF active micro-kernel repair 和 limited runner integration 有进展，但 near_E1_rows official 仍为 0 且 production official fused kernel 仍缺失，所以不能做 functional smoke promotion。
- KAN source-channel writer 本轮为 readback evidence；若 KAN_FU_S3 仍为 0，说明 MLP source-channel 形状尚未能可靠映射到 KAN writer。
- 证据链的核心失败模式是：h4800 大多为正、row-positive 已经足够，但 h4000->h4800 持续衰减让 `R_4800/3200` 停在 0.42 左右；这不是 threshold tuning 能解决的 selector 问题，而是 retained target/source observability 仍不足。

## Artifact Index

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_clean_unzip_self_test.csv | 1 | 27 | ccebde725152f9fa72b0b20a07ad30ad5f783e0a6657143b47f952465883718f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_clean_unzip_self_test.log | 1 | 111 | f316623e6c277b684f492c7adf19b53d45c6f41d7b66a8110af3a0c209da60fb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_code_review_packet.zip | 1 | 4766926 | 9e4cfdfb91f3f956c9cfb2ddd27180fc953e2451b0c613b4dcd8ac88d69b72af |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_code_truth_gate.csv | 1 | 256 | ce0c9ebc7215265946943e2ca7b393a701a4b881fdac056c94d29fef08572a88 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_command_journal.csv | 1 | 126680 | 974021d87f5e0c8e8934807ede8eb34082134df4c6b5f64fe57c2a6c37bd36bb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_compileall.log | 1 | 1209 | fdfbfdc7e766922645fa6e4761ab5172cf9575e805fc3b4ec67f71f5a1db958a |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_deferred_items.csv | 1 | 883 | 0a06eb9fcd6ed9416ea549c5fdccbfb3bcf223ed9e9e88b88d50734fa7976eed |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_diffeomorphic_target_unit_tests.csv | 1 | 119 | dd1085fabd64ea13f5f97f6aa12dd1d9ed4f7d90a49eab6e7d078979e3383dad |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_active_repair.csv | 1 | 10220 | 55751624680f550ef7f44f32f02c7480bee2b71f9765c167fdfe7e8c25e1c092 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_active_repair_summary.csv | 1 | 446 | e5b6b0b161b3f5542efc084ba680dfb2e7cf9855dd291a97a8866d6ea59996da |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_c4_smoke_route.csv | 1 | 1307 | b625f544741ec0652bd45e617fc62cf4dd4d50f2d4f83600f85ac8fbbefe412d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_c4_smoke_route.json | 1 | 652 | 21fcb6b0bce4a240eda94b1af77812fb83d7cb95c68ce07255d0b21fd32047d0 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_c4_smoke_summary.csv | 1 | 4445 | 48975cb2913c80f4773f64e425a088e607b2e44d57c5cf77b01fd658a0765470 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_component_waterfall.csv | 1 | 7116 | b5ed8b25b4c3b8215bba4213b28c9cc3a18aac882908b19dd5a5dae544a5e0a4 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_limited_smoke_route.csv | 1 | 1289 | 5834f6d06e97dfc9eefc9754389a05182c58cbd379e239659c67999f17f46d9c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_limited_smoke_route.json | 1 | 599 | b87093d933288c46a8fe96db7e0bc620b2631a3a8de89436af3b10ede1cbc8e7 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_limited_smoke_summary.csv | 1 | 4227 | 13cb66d7d1d097abb3db395e86dee5dfe4ea42d16a9b27d344daba5f5fa39cc2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_repair_decision.csv | 1 | 616 | 88d18cc07487d91db473ac1f7b4bcd8b42f7226857ecc8c86904aedc113d2ae2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_repair_decision.json | 1 | 1098 | 55b6ad9911f0985db9476fa803a586071ae44117f9f3afafd8f44d6079bd14d2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_runner_integration_decision.json | 1 | 1557 | 16f2782abc5b7cf27a01454d0ae50828313e79e40b56328f8c986a07a0153718 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_drbf_runner_integration_summary.csv | 1 | 822 | 58b9fd8fd27a4049651e842f6150f9a6bce495be5d169909030892559da560fb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drat_repair_table.csv | 1 | 5522 | c4e0cbace5bc68fadbfcc64d1b547495b454cf5f9b7b56fc63847b8cd1891cfe |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_drbf_repair_table.csv | 1 | 4579 | d08cd1e98b8acfc74a1a98575194b55b548cc51cf61dd429d814830515bd1ada |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_efficiency_full_loop_decision.json | 1 | 244 | 393c120bc1fff9d3c20caa337954f71d7b237b76ec9e4b7125c0f386594e7e3c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_efficiency_full_loop_summary.csv | 1 | 575 | 69494adb92c4264bb914b0d6578c81db2bfe3a5b457deb5aa7abbaea0846c281 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_efficiency_full_loop_table.csv | 1 | 7327 | 25d08cbe6815cf9d61805b354c5cb3a9c3b08e2cbc2d039526eb391623478225 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_efficiency_truth_table.csv | 1 | 36664 | 1fbdb6236ab06ae756be3d6090031f1557709921e2e1cd23f3562dc9dc487944 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f145_f147_source_preserve_audit.csv | 1 | 6521 | a3e280460ba14cae9d2ca10a48ae49c6a7bcc77638b886ab8f2277efcf5c7131 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f145_f147_source_preserve_dataset_localization.csv | 1 | 5848 | 3a3b09708bfdcad43ec01e9d24c2cbef9af0c5ae2384b1682d1f066a8a039138 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f145_f147_source_preserve_route.csv | 1 | 590 | 59aa688a91c9545c6e98826729606b40ca058384442225a9638c3600d0181228 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f145_f147_source_preserve_route.json | 1 | 683 | 816feec86ae3eadc70404a5f9fc8fc9d78e2acde3ebbe3021638185ba7ad9881 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f145_f147_source_preserve_row_localization.csv | 1 | 17233 | 0a20f6d099007da45d390b2bc6175940559863028a69ac23f8504c6d8538dfe8 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f148_f150_diffeomorphic_target_audit.csv | 1 | 6547 | ca3b8e9f1c190b2fc68ebd0d8991cf8b51907cb9df59f731e52dbd0946a42dc5 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f148_f150_diffeomorphic_target_dataset_localization.csv | 1 | 5888 | 37ffe32afc482f52385b354721ff85ea787b6608bfc7045a04c5f3f165189a72 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f148_f150_diffeomorphic_target_route.csv | 1 | 597 | fb1303536b725ef3971c6127f85f60250471c2e517ee5fb2c172efbba47857c8 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f148_f150_diffeomorphic_target_route.json | 1 | 690 | f3ca20951909fbe9a2136dd567c33e077495fc138c13e97322e672003a3b2bfa |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f148_f150_diffeomorphic_target_row_localization.csv | 1 | 17300 | 4da432987b1f5bb86533e878aad834c76ff3297c07c2a2cf6645335b6d03b99d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f148_f150_diffeomorphic_target_trace_summary.csv | 1 | 1473 | 4d91e3757aca5dd8f62060a88c2737189a28a6b7c79af465306800cb096d38ef |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f151_f153_terminal_preserve_repair_audit.csv | 1 | 6710 | 31bfd088efdc46c368a0a204aef2522f2a7fb7cd9827f1d05027c992d28e438b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f151_f153_terminal_preserve_repair_dataset_localization.csv | 1 | 5915 | 85d8fa30cbc304246b18f6ce8ccc0e044e4d9cfdca542c1faaa9aea4aa57bb80 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f151_f153_terminal_preserve_repair_route.csv | 1 | 607 | da383a2e16003954f3b1f17b3216f1e2677141c894ee3f929f42de3c8ca4158e |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f151_f153_terminal_preserve_repair_route.json | 1 | 700 | 35a303d3db490200c520405ad1dfbad286362862aa76955d071fb981693f31be |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f151_f153_terminal_preserve_repair_row_localization.csv | 1 | 17393 | 49277de79aab44c3d4174767962c3a5fcc207b6d9ce06adc63bae2542a771e44 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f154_f156_terminal_debt_lownds_dualmem_audit.csv | 1 | 6772 | 464b62feefbff90639284153224acf460d4e95c4499090e52b6eb8fee9d8d73c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f154_f156_terminal_debt_lownds_dualmem_dataset_localization.csv | 1 | 5895 | 477a151c07d565b9fbed6dbf4941f61931223d4ecd29dec10c2cf27b85ca68da |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f154_f156_terminal_debt_lownds_dualmem_route.csv | 1 | 618 | 2e5fa1acb7a198c03325c0731e6132ab39c737657e7fa5a6ab7da8f61e53ffb6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f154_f156_terminal_debt_lownds_dualmem_route.json | 1 | 711 | 0e68e768dc4da3508816adeef1810c0e34c2922775e02c4dd45471a264c3fc69 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f154_f156_terminal_debt_lownds_dualmem_row_localization.csv | 1 | 17173 | fad9f9804158e744970181ffb7326503b2a9b3cd9e64442c51a89cca55633eaf |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f157_f159_signal_estimator_audit.csv | 1 | 6744 | 87e24fbd7815a34a3585c96ebc07522ffefe38552dc896430d8bbf26ec1e2cbb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f157_f159_signal_estimator_dataset_localization.csv | 1 | 5834 | 958cc0e9985dc933d31eaa9e649de0b303748df4f2f867a4940d42447dc9972d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f157_f159_signal_estimator_route.csv | 1 | 602 | c34b0192c6b530037a98a2b99e243124cdfe19a5c389c7c0aa1366bc09ea47a6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f157_f159_signal_estimator_route.json | 1 | 695 | 62e6a75f49af0638d6d1dc59b34c96ddd27a45b46d4feb85742df9e1069e9b5b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f157_f159_signal_estimator_row_localization.csv | 1 | 17099 | 02ab7dc9c68bc70b036d4750665bfa23fa72964087b5dd9da4203c81e7c4da7b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f160_f162_post_h4000_source_floor_audit.csv | 1 | 6779 | 4d2a3b470b1b66a381e2f585a508fab3d78978033b8ea4c139e0dd9e4eb57c64 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f160_f162_post_h4000_source_floor_dataset_localization.csv | 1 | 5848 | b8027b8b1f56341f77ea476d34ff67d9e63f4b1ac118e64400b89d143fe8fc39 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f160_f162_post_h4000_source_floor_route.csv | 1 | 621 | b50365ea8e31b3b38567327e351b6d35400143f5aaa95a745dba0b4674016c38 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f160_f162_post_h4000_source_floor_route.json | 1 | 714 | 16de2820962420a7bdac46bce917b1382552ddff12a9413436c620efe0dffb15 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f160_f162_post_h4000_source_floor_row_localization.csv | 1 | 17091 | 3ffef2707a8e351551ad3ab90cff512d0d305a5882b9ae865188ab94de6be38d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f163_f165_raw_guard_post_h4000_floor_audit.csv | 1 | 6710 | f34af67e754905d255482d2f7070d1966f8da4bc7014ba6e268a82e6aa9e74ba |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f163_f165_raw_guard_post_h4000_floor_dataset_localization.csv | 1 | 5959 | f4664cc83dd3b0ff29734707163a990c6e80e55e0200da909471e11c2cd7caf2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f163_f165_raw_guard_post_h4000_floor_route.csv | 1 | 617 | 4c47096f606f05c6698e620d5ce1ce359bf586df1e1f1cd4608e45307bc6f547 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f163_f165_raw_guard_post_h4000_floor_route.json | 1 | 710 | 582625b3048ee4915632b4c8874b200794420476395f7b2ba13a335a77c609a6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f163_f165_raw_guard_post_h4000_floor_row_localization.csv | 1 | 17386 | 28f056e3c89887fbe3a780c5676c3a20b4159462a1d640cd03e951a0674f8667 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_audit.csv | 1 | 6670 | 8d78ae63638b969e04ddc47f01f77768b6ab8fd917d30ef88db7a02c06368e25 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_dataset_localization.csv | 1 | 5858 | b47e8fae77abfdb1014dc6075de44ea02fa68c49ed0fe853db6cabd7dbec859a |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_route.csv | 1 | 609 | 9cb40310bbeb94c08aaf6898bce36c298d17ea104e46b063b553b3f6de30498c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_route.json | 1 | 702 | f8a1eaf7d805ab2de1e5e38715e34daded56b0ced1daedc00a80429bd263f2e2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_row_localization.csv | 1 | 17370 | 4e3796da497bb8a1a3180b38d40d1f2b70c9e05d3223a8316adbf860fec4ea4f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_smoke_audit.csv | 1 | 5955 | bc232bf448de978021a328dfc7f2ed42978e2f0121baa400317ff2ee13884f9c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_smoke_dataset_localization.csv | 1 | 1854 | e7eddabe05aa16bef38c1dc051d051c0080b754850a5dab3989f34bd1b378304 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_smoke_route.csv | 1 | 534 | d71926bc0498d8d0e59d849d0f5959e1c3a097a2034b9a167c9e2462f3bb1247 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_smoke_route.json | 1 | 629 | de8a12f7c4139cfc4eb1b3d89ca179583fd6dc36062b136e0e22b5bd38156076 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f166_f168_anti_erosion_smoke_row_localization.csv | 1 | 1902 | 2fa5f15048c8b842398ee83ec3900b33fc9ac9e54a339660683f606250073d77 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_audit.csv | 1 | 6634 | 9f228e990b4efb32b7db19bb0f2830a880856d786e38dd0527450e2b7b674e2a |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_dataset_localization.csv | 1 | 5928 | 90dde908a9b3a01e90e12cf144317ec1d962aa558e3cf0cfce82fa36698b172f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_route.csv | 1 | 605 | a73462e81ff4229c4daa9f3e277f45b0dc75a2816f49a829b8aca86cdac36cef |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_route.json | 1 | 698 | 4a641c687d5c77522df63c9589c23e1a2404e0d10d621ef55e5d04effb489a55 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_row_localization.csv | 1 | 17369 | e2ff27cfe21db8330b7a6433e3093458a077201ccfbc727e5766b6d7e529de82 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_smoke_audit.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_smoke_dataset_localization.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_smoke_route.csv | 1 | 480 | 6efa4eb1320d8319739fda5e11b2a999692a9fa233d9e05edd43f5750243d6ef |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_smoke_route.json | 1 | 577 | 66de34bd3f57987bd4537637d1a1e98c61fdb8e4bfdc932dae7b02981150410e |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f169_f171_h3200_anchor_transport_smoke_row_localization.csv | 1 | 2 | 7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_audit.csv | 1 | 6669 | 7309eaea67044e738bbbdcc8122a3e619494eb7fe6eabcf1ed4c5606da9baab8 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_dataset_localization.csv | 1 | 5874 | d783bc60c81a995dc794c04a8033c341a7483266cfdfae26d6acbd59f5c30f8a |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_route.csv | 1 | 604 | d801144c2300acb408f890bb67d74e06e7f1fc7466b213a331a3d7356ad7ef16 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_route.json | 1 | 697 | 78e9e11f1c25a218654418f0c2376f31f8e6792d144586f26597b61de6dbd9ef |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_row_localization.csv | 1 | 17255 | 183a2c33f31831eecb730835f4268e37e4d9434f11c4dc3eb485f0f00f68ae23 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_smoke_audit.csv | 1 | 6318 | 76b92218cf45c26b589c8ee4635fc8b124315d65bfc55511b0cf90d62d56555b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_smoke_dataset_localization.csv | 1 | 1943 | fc1e4c474781d3adc10fe29e995a1a090dcb097ec8c57a687b0b5d5d4e124e5e |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_smoke_route.csv | 1 | 587 | 827ec32be6db74a88d4dffb4e9bd8df4fb06ffa22e495724752f52418db822a8 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_smoke_route.json | 1 | 680 | 3089826ffe57a85fd6da92dd714b567f864100ef4218c14bdf79db49109244ef |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f172_f174_progress_carry_smoke_row_localization.csv | 1 | 2012 | 6d0a0e554773017ad7f96e0fa43ef23d283869f6d9ce3124d81eea966f583906 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_audit.csv | 1 | 6661 | b6d1ab7ac1ddbe4d933cbec24821156304e241a4aaaf2ac1cd24551b6916fc6c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_dataset_localization.csv | 1 | 5995 | ab77a25aa1c53b732b010c99be5e4d76982ae8e55bc6ea1f2edf687a143efc3f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_route.csv | 1 | 609 | 0ca53b0b2964daf1a6b10eec81eda5da9a70c944d7f50578170f13abe6ab8582 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_route.json | 1 | 702 | e522df5487da9bae240c64feb1efefe559b33f976dac283ec58ceaae66834a7c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_row_localization.csv | 1 | 17574 | 9551a7a808d989df6157cb19996488f8e42d57b5c54a5bbc7973113c8e03c319 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_smoke_audit.csv | 1 | 6310 | b12dd1ffe11c3d0f5eccaf49983d1ff5f1364bfa01125b7ec0d957f5605377f2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_smoke_dataset_localization.csv | 1 | 1970 | c5d59f70e2fa8a71636b6eb8086ffdbcc26e3a7b211eaf8ad8f99a94633309a0 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_smoke_route.csv | 1 | 592 | 055b01d178baab6fc7186439e1655c1c906176079f26c01fea19401637176f2c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_smoke_route.json | 1 | 685 | c79d9519433465deaf9cf93631a12c0558fd99bbaa2e30dbbf2ee32814d49ed5 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f175_f177_gentle_progress_smoke_row_localization.csv | 1 | 2039 | 059a845dfde3052f285942d9c996b90262e631b1549bde2c48bd85126630fc5d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_audit.csv | 1 | 6831 | 96c9aa2da6bbd15af3d16cf9516825ba792c45be48fd3446fde6e8bf39d84a8d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_dataset_localization.csv | 1 | 6002 | 965fdc8b6885c308b74c017a94a031e155a173fdb7b53f32fedeef8b7571ca80 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_route.csv | 1 | 624 | 8954aa071eb6249ea774c436b2568e40883f4f4a637d3b6be4ebb8a0ccc53928 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_route.json | 1 | 717 | 6aabb7410e1651c8c9aea217d8e7b81cb21d26519a76fa1865e463de91379ace |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_row_localization.csv | 1 | 17576 | 487055a74ea499212cb212dd7c199a029ecbb940d44a198f635a38e387f9bda2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_audit.csv | 1 | 6481 | 2110cdff048719a5f8b31b2afca8b6e02ce54d29c97caa1ece8936645991e209 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_dataset_localization.csv | 1 | 1974 | b353de8b03901b7046a4dc725ada242def4b9cb271e9eb5852ada1ad433c9b47 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_pre_wiring_fix_audit.csv | 1 | 6044 | b280a9346fd1a073fbbe31b80c9d09650a062debb8301e9c9f932d7881e78526 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_pre_wiring_fix_dataset_localization.csv | 1 | 2026 | bd11172c6e75cd66843e71e8a74a5c257b6f41d159ab7b4a2acffb9485db8eda |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_pre_wiring_fix_route.csv | 1 | 588 | 603232a8e60293d9723ec4414678c9065321d24c256631686eddf814466f5915 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_pre_wiring_fix_route.json | 1 | 683 | 87dc736623962514b9f31c777ed7609b3ed31ab8a719e9902bce246a1153a5fb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_pre_wiring_fix_row_localization.csv | 1 | 2095 | 07011c24af7fdf02490cc2c758ef95f33c223f4cc6d10f42e675ce61fde4348f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_route.csv | 1 | 608 | 8671c1c670ee324d751fcc437bfd5e3d8eb9b023683f4eac1adf1762203237bb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_route.json | 1 | 701 | 5b2feb97a7917b11a10a2365283838bad80cdebcebfeeb0569063d48df554fc6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f178_f180_control_relative_catchup_smoke_row_localization.csv | 1 | 2043 | cad1ca2c229ea7aa1e9a0941a20e16c12ee17757979e37e3a31951a15abed46c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_audit.csv | 1 | 6747 | f57a8a65a16f7d01c8cbc9305516477faa716e8b32e8b165f4017b75cb0f3951 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_dataset_localization.csv | 1 | 5895 | 9912f07cebc79713bb00b8b13e48feaf5e602df6402ca1ca337ecbd63b587597 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_route.csv | 1 | 623 | 355445fd79bcd4cf09dfcc56bdafcc4abf9cd685d47b4bd07fb2fc9b075aa4fd |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_route.json | 1 | 716 | 5ebcf4bef394a5d6d9f6ef1d59f3835941671008dc3295852070dfecf56f2410 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_row_localization.csv | 1 | 17300 | 1cf8281f483f6576ef8852fac93179c84832d9181244f77f127b459b5082285e |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_smoke_audit.csv | 1 | 6393 | 6ae6b3e6ab564f5f2f430863ee5028f04bc0251d7df36edc761811bed01692a1 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_smoke_dataset_localization.csv | 1 | 1937 | b5611840f57301cc2bfbbb63d3303954bd5bd5c1cc9a1d702e7b112f2c3b98e6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_smoke_route.csv | 1 | 604 | 10fbbde21a4e245f30289a4263a6d3252bef60ae81d3aeec26eb8c1ca279c5fa |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_smoke_route.json | 1 | 697 | a0fc8dbe860dfaba0240ecb67e19e1ab68d3eb147a2e57767c36c0a4e48a4ffb |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f181_f183_trajectory_adaptive_smoke_row_localization.csv | 1 | 2006 | e595fc833ea5f393f13029742b68f3fe6525d16a98d4f936a32423011e1f86d1 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_audit.csv | 1 | 6637 | 1502152d8174d94b2c5c9546d19c7f15b8412fcd30a0991b3014dc7382bcadee |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_dataset_localization.csv | 1 | 5803 | 236e0550553ae5814efa71f2af5d52ee6dabf5396c292ad0f6feb356b6d9af57 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_route.csv | 1 | 598 | 7b9bdd3983357086111744bfd06b16f258d3ea4d36ca16369b6f5a9508caed9f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_route.json | 1 | 691 | 76a7e0cd6fd6f3345b0c4c36a9c107adbf184c8a584daacb5380a2c26ac84552 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_row_localization.csv | 1 | 17175 | dae52b8cb478d3d126c806dd51007a5206984c254430d2a04de5e3dfd2c67c28 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_audit.csv | 1 | 6289 | ce17d0621dfd2827db3a3485dc08bb7d53e89cc69cd58d48b79df31869d7ad1e |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_dataset_localization.csv | 1 | 1921 | 44ac2080cc766f504b21eb8ef5a730b0dd36bbb60459355b79c5dcfa01b21052 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_front_wiring_fix_audit.csv | 1 | 5851 | 7a0a54545f515baff897539dc0d8355894fe5dbc34ddee5d39bc6595e779ed35 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_front_wiring_fix_dataset_localization.csv | 1 | 1973 | 9581fb8f96d20a403fdba1528b709c3308aaed0d1eda2a50fb5b995bd439311d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_front_wiring_fix_route.csv | 1 | 562 | 4a6c8a16959fb6f4c25ec7fba9a17f82e1ae6921b29f5a06cf1361c9ca12140b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_front_wiring_fix_route.json | 1 | 657 | 6f0c9e351b60533bf776442a692094639cf8b0de4a2f78cb7a11ab34aa522181 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_front_wiring_fix_row_localization.csv | 1 | 2042 | e61842f8a8e8a22df968146324dfae2c24acd7267846459e7e4213f73d0029d8 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix_audit.csv | 1 | 5851 | 7a0a54545f515baff897539dc0d8355894fe5dbc34ddee5d39bc6595e779ed35 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix_dataset_localization.csv | 1 | 1973 | 9581fb8f96d20a403fdba1528b709c3308aaed0d1eda2a50fb5b995bd439311d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix_route.csv | 1 | 562 | 4a6c8a16959fb6f4c25ec7fba9a17f82e1ae6921b29f5a06cf1361c9ca12140b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix_route.json | 1 | 657 | 6f0c9e351b60533bf776442a692094639cf8b0de4a2f78cb7a11ab34aa522181 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix_row_localization.csv | 1 | 2042 | e61842f8a8e8a22df968146324dfae2c24acd7267846459e7e4213f73d0029d8 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_route.csv | 1 | 579 | c728b2bbafe7e5b6559443ae287ec703b2dbcf574edec81c81b36b6c236b0f38 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_route.json | 1 | 672 | bbb9c2fa3171c2a4019b5564bfe97e86688661426814c20e39112fcbe2f22679 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f184_f186_minimal_transport_smoke_row_localization.csv | 1 | 1990 | 79c8545c65b0f4758726d869b7502f93b1ffc6bb1c0cd3b3f5b89380075c2b09 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_audit.csv | 1 | 6610 | fcc8cb4cf39db86a03a41c179dc5d4535b9747911343b9bc3312fccbd491b61e |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_dataset_localization.csv | 1 | 5791 | 562061be0b9d75611af9c3dddc24971d690bcc82d01527f3cbbe7ff4690b4720 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_route.csv | 1 | 586 | 99d4509f720486908324eec75225964a3d4c5dc8194ebb8042e6f6443cb02eee |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_route.json | 1 | 679 | fc3f77ae5fe5422dd02371156c089ce7109d3ae4dc3112169e639a2616465ab1 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_row_localization.csv | 1 | 17153 | 797c01d8e64d027e1a385c7ed6bf1ccf4f77b09a2d90f9795a91e47d5f129bb9 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_smoke_audit.csv | 1 | 6253 | ecb2664b88dd15a50984cba49f197727a2a82479a76de7ff4ae81cbedb3637df |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_smoke_dataset_localization.csv | 1 | 1925 | d3aa367fadcddddc3556df485171d0203baa5573d60f3250c059364171cfa91f |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_smoke_route.csv | 1 | 569 | 98d734fc379e6871b9310d691e425b0bf630dce1c399a5e4d8c49ddb32c2cbcd |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_smoke_route.json | 1 | 662 | 28fc76b028ce0cc281b5723d5853225c92932e040942ad436822317a11742ce5 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_f187_f189_accept_memory_smoke_row_localization.csv | 1 | 1994 | 5da2c4abe066b1e90901ae8c543c32e7403d643ee0c0f82f7efbe70d34e1cb41 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_gpu_assignment_manifest.csv | 1 | 57071 | 4ae7d418ddfeb16d246ab27cf2af781e37df53b12258373631b9cdd85cf586a5 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_gpu_utilization_timeline.csv | 1 | 160 | 4a77bf5edf6046b2189e650191d5c46b8e37a831ed99d35fab65423158038761 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_idle_violation.csv | 1 | 75 | dfae76840d98fd32f5c94509cdc28d8b532f7bf4271de7a3d24ee97d544729f6 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_import_closure.log | 1 | 386 | 044743c50c8e858eee2c79b6eb34875f936b3875c5c7b491855189a424b3f729 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_kan_source_channel_writer_decision.json | 1 | 316 | 9e7f91d84ee8652ac594ebefaeffb5c05f0746b009d3cb0fff66049b78021710 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_kan_source_channel_writer_matrix.csv | 1 | 10467 | 362c0846b733fa7e68789ed1848e3047cdd3d4f56b11b85f4fe88736d1e474bc |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_mechanism_contracts.csv | 1 | 6912 | 36b2083c1ae4d98e4dd02a6d906a3d6d067a87fef8cd71adefb34ac012a51c9d |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_queue_drain_report.csv | 1 | 57 | cfd8d05c8a1f50bae522d21560aab47810edf3601707f0a2bd0915c057673bd2 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_required_artifact_manifest.csv | 1 | 39619 | 9c0cb95b224ebec012b2d966a2772eae336351920e7a787166fb035f8aa31ac7 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_results_bundle.zip | 1 | 25122913 | e0076611f8be0e7b6e9c29c651dc2beeb61f98a7a1bae07fc6fa0272827e3c79 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_route_decision.csv | 1 | 745 | 144ed091eabbe7df8e98f029d1721e5990ab2a3138b820b0cc4b0142fa411465 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_route_decision.json | 1 | 876 | 10e41656acce18a1707d2d7acfd24fd3ff908bf0761c01611223b8885f4bfc7b |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_runnable_queue.csv | 1 | 73985 | bd3d76172a6b457be9a545058f4373d2f75d56ce84f27c3d833442523b007ebd |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_terminal_erosion_autopsy.csv | 1 | 16663 | dd9a53b777ae2fc3486fc3e62c89426d081e565aeb071f6403911b221075e50c |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_terminal_erosion_class_summary.csv | 1 | 143 | 87c8115f3086a2b2b5ad0816731f58b87e9aa3ecffa2110c97e5125fad0d5311 |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_terminal_erosion_route.csv | 1 | 361 | df3f1295bda16a157585f476f5e51f60d953d4dbb39f7a4b317994575d0c56bc |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_terminal_erosion_route.json | 1 | 414 | aac861fdc72fca64e8c493fee65579e1ecf6da108a522c4c8199b0c25d7b04bc |
| results/v22_03_terminal_retention_diffeomorphic_source_channel_basis_efficiency_4gpu/official_v22_03/v22_03_terminal_retention_unit_tests.csv | 1 | 239 | bc1ecd8a19c948600db1a43f1275137910d253529347dbf99521b6db22eb1d84 |


- v22_03_code_review_packet.zip: size=4766926 sha256=9e4cfdfb91f3f956c9cfb2ddd27180fc953e2451b0c613b4dcd8ac88d69b72af
- v22_03_results_bundle.zip: size=25122913 sha256=e0076611f8be0e7b6e9c29c651dc2beeb61f98a7a1bae07fc6fa0272827e3c79
