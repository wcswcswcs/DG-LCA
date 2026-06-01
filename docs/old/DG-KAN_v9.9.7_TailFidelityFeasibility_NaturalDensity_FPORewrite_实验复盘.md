# DG-KAN v9.9.7 Tail Fidelity Feasibility / Natural Density / FPO Rewrite 实验复盘

> 本复盘记录 `DG-KAN_v9.9.7_结果解读_TailFidelity可行性_自然密度裁决_FPO重写完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 branch-horizon、density、controller、generated、runtime 均显式 `not_run`。

## 0. 最新结论

```text
route = CaseB-TailKeyV3FeasibleGeneratorFail
primary_blocker = tail_key_feasible_but_no_major_tail_generator_passed
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

最终 artifact：`results/real_rerun_20260506/v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite_full_20260517T200000Z`

核心结论：

1. P0 复现 v9.9.6 boundary：source route = `RouteD-TailFidelityStillFails`，G20-G29 official/weak pass count = `0` / `0`，fake/proxy/cpu = `0` / `0` / `0`。
2. P1 current tail key = `TK2-hierarchical-major-tail-key`；tail groups = `337`，support<3 fraction = `0.12166172106824925`。
3. P1 expected missing tail @1024/@5000 = `72.73671413360148` / `5.388789904783137`；official feasible @1024/@5000 = `0` / `0`。
4. P2 tail key v3 candidate/pass count = `8` / `1`；selected = `TK3H-major_template_step_coarse`，selected pass = `1`。
5. P3 G30-G39 candidate count = `10`，official/weak pass count = `0` / `0`，best = ``。
6. P3 best failed = `G33-major-first-tail-reservoir-refill-generator`，major/tail PSI = `0.5578023370995995` / `0.3067280643885967`，missing tail = `0`。
7. P4 largest completed panel = `0`，density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
8. P5 future path weak/strong = `0` / `0`。
9. P6 FPO v9 weak/strong = `0` / `0`；best = `FPO9C-signal-reservoir-drift-diffusion-v2`，precision = `0.011494252873563218`，V LCB = `-0.383912815103318`。
10. P7 controller = `not_run`；P8 generated sandbox allowed = `0`。
11. No-fake audit：rows checked = `10658`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.7 G30-G39 generator profiles 与 tail-key-v3 结构采样入口。 |
| `experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py` | v9.9.7 runner；执行 P0/P1 feasibility/P2 tail key v3/P3 repair/FPO v9，并按 gate 开 P4-P9。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py
```

```bash
python experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py --out-dir results/real_rerun_20260506/v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite_full_20260517T200000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9970",
  "status": "summary",
  "route": "CaseB-TailKeyV3FeasibleGeneratorFail",
  "primary_blocker": "tail_key_feasible_but_no_major_tail_generator_passed",
  "secondary_blocker": "density_panels_blocked",
  "route_explanation": "a feasible tail key was selected, but G30-G39 did not produce a weak or official major+tail fidelity pass, so branch-horizon and density panels stay blocked.",
  "source_route_v9960": "RouteD-TailFidelityStillFails",
  "P0_boundary_pass": 1,
  "P1_current_tail_key_official_feasible": 0,
  "P1_expected_missing_tail_at_1024": 72.73671413360148,
  "P1_expected_missing_tail_at_5000": 5.388789904783137,
  "P2_tail_key_v3_pass": 1,
  "selected_tail_key_id": "TK3H-major_template_step_coarse",
  "selected_tail_key_source": "tail_key_v3_passed",
  "P3_official_pass_count": 0,
  "P3_weak_pass_count": 0,
  "P3_weak_pass": 0,
  "best_generator_id": "",
  "best_failed_generator_id": "G33-major-first-tail-reservoir-refill-generator",
  "P4_largest_completed_panel_size": 0,
  "P4_density_sufficient": 0,
  "P4_density_insufficient": 0,
  "P4_density_inconclusive": 1,
  "P5_future_path_weak_pass": 0,
  "P5_future_path_strong_pass": 0,
  "P6_FPO_weak_pass": 0,
  "P6_FPO_strong_pass": 0,
  "P7_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_generator_fidelity_failed",
  "P9_runtime_pass": 0,
  "P9_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Tail Fidelity Feasibility

```text
tail_group_count = 337
support_lt_3_fraction = 0.12166172106824925
expected_missing_tail_at_1024 = 72.73671413360148
expected_missing_tail_at_5000 = 5.388789904783137
tail_coverage_possible_at_1024 = 0.8783382789317508
tail_coverage_possible_at_5000 = 1.0
P1_current_tail_key_official_feasible = 0
```

## 4. P2 Tail Key v3

| tail key | groups | support<3 | tail PSI/JS self-sample | Core77/OldOnly/SlowBurn coverage | pass |
|---|---:|---:|---|---|---:|
| `TK3A-major_recipe_precursor` | `1879` | `0.883448642895157` | `2.327431869265844`/`0.39431277882772864` | `0.2987012987012987`/`0.2`/`0.24074074074074073` | `0` |
| `TK3B-major_recipe_precursor_norm` | `1879` | `0.883448642895157` | `2.3274318692658365`/`0.3943127788277284` | `0.2987012987012987`/`0.2`/`0.24074074074074073` | `0` |
| `TK3C-major_recipe_memory_offdiag_proxy` | `1922` | `0.880853277835588` | `2.3691782165155724`/`0.3977060790961243` | `0.2727272727272727`/`0.2`/`0.24074074074074073` | `0` |
| `TK3D-major_template_step_norm` | `202` | `0.19801980198019803` | `0.05606245229076569`/`0.06065389246715566` | `0.961038961038961`/`0.9`/`0.9629629629629629` | `0` |
| `TK3E-major_recipe_action_shape` | `2090` | `0.9172248803827752` | `2.6611750803695617`/`0.4225487077724811` | `0.18181818181818182`/`0.2`/`0.16666666666666666` | `0` |
| `TK3F-major_recipe_precursor_merged_low_support` | `1946` | `0.8766700924974307` | `2.3749529500681845`/`0.39871252856411926` | `0.2857142857142857`/`0.2`/`0.2222222222222222` | `0` |
| `TK3G-major_template_step_norm_merged_support3` | `197` | `0.16243654822335024` | `0.04180711218366807`/`0.05318442885071887` | `0.974025974025974`/`0.9`/`0.9814814814814815` | `0` |
| `TK3H-major_template_step_coarse` | `89` | `0.0898876404494382` | `0.007805208227195761`/`0.02399732919258897` | `1.0`/`0.9`/`0.9814814814814815` | `1` |

## 5. P3 Generator Repair Matrix

| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |
|---|---|---|---:|---:|
| `G30-IPF-hierarchical-major-tail-generator` | `1.5489318428398824`/`0.3419141871065975`/`0.0244140625` | `0.8081457068114235`/`0.3010997589797565`/`1`/`0.9887640449438202` | `0` | `0` |
| `G31-mincostflow-major-tail-quota-generator` | `2.1234303091256765`/`0.36434526212687707`/`0.02734375` | `0.972656258995131`/`0.32196660718337683`/`3`/`0.9662921348314607` | `0` | `0` |
| `G32-tail-first-major-corrected-generator` | `2.002504569332323`/`0.36752792185435457`/`0.0263671875` | `1.017789440581231`/`0.33185429294654306`/`3`/`0.9662921348314607` | `0` | `0` |
| `G33-major-first-tail-reservoir-refill-generator` | `0.5578023370995995`/`0.25624626700452185`/`0.009765625` | `0.3067280643885967`/`0.19192799733168833`/`0`/`1.0` | `0` | `0` |
| `G34-stochastic-rounded-tail-quota-generator` | `4.223849992317016`/`0.44237124722244603`/`0.03125` | `4.482203428314805`/`0.436751363417191`/`41`/`0.5393258426966292` | `0` | `0` |
| `G35-precursor-bootstrap-tail-generator` | `4.659731535345041`/`0.46239525856064345`/`0.0341796875` | `4.982016053378079`/`0.46010046585795855`/`43`/`0.5168539325842697` | `0` | `0` |
| `G36-entropy-regularized-sampler` | `1.2675563816649873`/`0.32341710298947596`/`0.0244140625` | `0.6450715197863423`/`0.27380781788387815`/`0`/`1.0` | `0` | `0` |
| `G37-tail-coverage-negative-control` | `0.6724085074775694`/`0.2810045648065544`/`0.005859375` | `0.3528479489997663`/`0.20539714004376386`/`0`/`1.0` | `0` | `0` |
| `G38-tail-oversample-importance-weighted-diagnostic` | `4.239532751338325`/`0.44331422554126637`/`0.0322265625` | `4.490485533512091`/`0.43790752790284004`/`41`/`0.5393258426966292` | `0` | `0` |
| `G39-two-stage-feasible-tail-key-v3-generator` | `2.3528593030972837`/`0.3787797870750702`/`0.0283203125` | `1.5283505538373487`/`0.3515054686368132`/`7`/`0.9213483146067416` | `0` | `0` |

## 6. P4-P9 Boundary

```text
P4 density = 0 / 0 / 1
P5 future path = 0 / 0
P6 FPO = 0 / 0
P7 controller = not_run
P8 generated = not_run, allowed = 0
```

## 7. P6 FPO v9

| fpo | Fast/Slow/Path precision | V/V240 LCB | longrisk UCB | cost q90 ms | weak | strong |
|---|---|---|---:|---:|---:|---:|
| `FPO9A-tiny-virtual-adamw-sketch` | `0.0`/`0.0`/`0.0` | `-1.3481689497458516`/`-1.2562536657885377` | `1.0` | `0.0027548521757125854` | `0` | `0` |
| `FPO9B-jvp-gradient-alignment-sketch` | `0.0`/`0.0`/`0.0` | `-1.2005238086611552`/`-1.0767627474972639` | `0.997968144575522` | `0.0014719553291797638` | `0` | `0` |
| `FPO9C-signal-reservoir-drift-diffusion-v2` | `0.0`/`0.0`/`0.011494252873563218` | `-0.383912815103318`/`-0.23272393384564966` | `0.952667674022474` | `0.0012721866369247437` | `0` | `0` |
| `FPO9D-memory-offdiag-hard-gate-plus-value-sketch` | `0.0`/`0.0`/`0.0` | `-0.9982565611651513`/`-1.088668876115889` | `1.0` | `0.0017723068594932556` | `0` | `0` |
| `FPO9E-slowburn-detector` | `0.0`/`0.0`/`0.0` | `-1.0768743890687797`/`-1.112311419343473` | `1.0` | `0.0013923272490501404` | `0` | `0` |
| `FPO9F-negative-control-immediate-response-only` | `0.0`/`0.0`/`0.0` | `-1.4598077747549532`/`-1.3567645703748321` | `1.0` | `0.0006710179150104523` | `0` | `0` |

## 8. No-Fake / Contract / Failure

```text
rows_checked = 10658
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9970.csv` | `c85b9fba5d69d885cfd701f1ccb7d886e2fabaf42a832061a8bb8a935d4d43e3` |
| `failure_taxonomy_v9970.csv` | `3b80294f80e43c3aaac6733b86797ee6b5b864f3a88dd4a107540039bfe30f54` |
| `fig_p1_expected_missing_vs_panel_size_v9970.svg` | `25656f5fe71ad098f5e9cbfb0a59caebe54fb626d685b75d18856b404c1075bc` |
| `fig_p1_tail_feasibility_support_histogram_v9970.svg` | `0a856aa0d43a243cc9f99a0c9f0688ed42fd21f63cba88620dd684276bea069e` |
| `fig_p2_tail_key_candidate_matrix_v9970.svg` | `2cd77e51072b5f6ed82beab96723d079c8cdc6c492b0095cff376865efe63f68` |
| `fig_p3_generator_major_tail_pareto_v9970.svg` | `4ea987ddc16790a324311252f841ed45d7f5dffca431e0df9f3b348a05f8cdf7` |
| `fig_p3_missing_tail_by_generator_v9970.svg` | `6f1570d03760fa53712708c072e4604c89bfa24942dcd9e09c73e7b14b189d06` |
| `fig_p4_density_ci_by_panel_v9970.svg` | `829e23a3765d850e637e5c98a3c68eb7c1d58ad14151946d3cabaa930b504dec` |
| `fig_p5_future_path_types_v9970.svg` | `a4a7df379c6d5acc37139c5427447a5622bcd9702cf72645dc7e58488faa9233` |
| `fig_p6_fpo_precision_vs_cost_v9970.svg` | `7602b045d4f920ebeff9c3714b20fc6d911dc1daa0899a0b3a5616fe54a910f2` |
| `fig_v9970_stop_pivot_matrix.svg` | `1467325a1bf08b51f3938caf8441ccab73fc69022dea932ed4c55de94499af7f` |
| `no_fake_audit_v9970.csv` | `ffc6438ca4838baf89dd2f2dbfc9d1bb174f2442b104320127023c9d93d70315` |
| `p0_v9960_boundary_reproduction_v9970.csv` | `0746928e826f0abf00effdf2168deabbc35628cea79347ae612f57f6de754234` |
| `p1_tail_fidelity_feasibility_audit_v9970.csv` | `0544e9c0bc245a03349ef31636da5fdd349f8aa48dd9fc5dd0dd21d69d4fc20f` |
| `p2_tail_key_v3_audit_v9970.csv` | `d543cb511335a8810d3a854bf98e87b828961b18014e40eacef7fec5dc36be70` |
| `p3_G30-IPF-hierarchical-major-tail-generator_distribution_fidelity_v9970.csv` | `17283853ae0e7a7c7e74b0f1c01b402423a6eaaf87959ffc5085acf95bcb2fc5` |
| `p3_G30-IPF-hierarchical-major-tail-generator_distribution_only_actions_v9970.csv` | `7167ed358f21884086b712fa0ca6ee859891893ecc85576ae4b7047fb9d61b83` |
| `p3_G30-IPF-hierarchical-major-tail-generator_distribution_only_smoke_v9970.csv` | `9aff6ba85eba71dd1fed73f7261066e7349909c8b4680245a90b48a8187ce1e7` |
| `p3_G31-mincostflow-major-tail-quota-generator_distribution_fidelity_v9970.csv` | `4a3fdd77e69d1505cfd0e2cd33e31fd76b7058f0370f5bbdab38e60f3c93c7b8` |
| `p3_G31-mincostflow-major-tail-quota-generator_distribution_only_actions_v9970.csv` | `a6f8a8da6abc82f70396b1b165fdaf82cae0d0bc9639ca4c827071a820d3b27f` |
| `p3_G31-mincostflow-major-tail-quota-generator_distribution_only_smoke_v9970.csv` | `cab82d17131a7a2e1ec7bb1af01b415fa9971dd8ae1f1c53b74541222676da2e` |
| `p3_G32-tail-first-major-corrected-generator_distribution_fidelity_v9970.csv` | `1b0f5869a4426325910414966dd5b8dc24d9b1c547cf90172afb9c3296e59bd1` |
| `p3_G32-tail-first-major-corrected-generator_distribution_only_actions_v9970.csv` | `14b7e5e514aece0c9765c46c585677dece5d3c76cf618e8396f200128ae3ba4c` |
| `p3_G32-tail-first-major-corrected-generator_distribution_only_smoke_v9970.csv` | `8922aec0bcc37759d7bd72f2adda909b4e35400bd8b274a0b44a451c7e924578` |
| `p3_G33-major-first-tail-reservoir-refill-generator_distribution_fidelity_v9970.csv` | `d2eafce3a36432b39d7ef71f5e77762b0471d268b79671289e3039b5a9aa0a49` |
| `p3_G33-major-first-tail-reservoir-refill-generator_distribution_only_actions_v9970.csv` | `7de90fc71df3430114bc64c6a42e3c127b3161c573d052449ea2df5666185e71` |
| `p3_G33-major-first-tail-reservoir-refill-generator_distribution_only_smoke_v9970.csv` | `e7e6bc84ba241bbd40bda7a670b3256780c7e012a340586e151491299c263d50` |
| `p3_G34-stochastic-rounded-tail-quota-generator_distribution_fidelity_v9970.csv` | `87bbc3ac31613de69473e0faddc4bad78ec09e1ceb63c91423a4d088e738f611` |
| `p3_G34-stochastic-rounded-tail-quota-generator_distribution_only_actions_v9970.csv` | `4fccf58e78f689dea3d40ef0d5328a8abdd457b88484e2ff386accf7ea36d860` |
| `p3_G34-stochastic-rounded-tail-quota-generator_distribution_only_smoke_v9970.csv` | `b86d089ef6a87cc0cac46bdc3568b76910d47bf9d8fb7327470d51eb053d82fa` |
| `p3_G35-precursor-bootstrap-tail-generator_distribution_fidelity_v9970.csv` | `2d5204747bb90293fb65f8030c1ff2bcbb48f0556f757051df7bcf12eec1b687` |
| `p3_G35-precursor-bootstrap-tail-generator_distribution_only_actions_v9970.csv` | `91f388c94c778e0c37523ef17d4e07d01beaf4fe8f6b9bc1a9569d0060965c3e` |
| `p3_G35-precursor-bootstrap-tail-generator_distribution_only_smoke_v9970.csv` | `54a0298eba4232cf438a90d3d903f48cec99c9fdbf96305fc97dbed4787d94cb` |
| `p3_G36-entropy-regularized-sampler_distribution_fidelity_v9970.csv` | `86e63e8cca15a22a259ff70c8a709ae4200a3a0619788ace94545040190dfe08` |
| `p3_G36-entropy-regularized-sampler_distribution_only_actions_v9970.csv` | `810580639c7a13568c1f186d585569110cd504b33f2e281443ebb5aa6b6b717f` |
| `p3_G36-entropy-regularized-sampler_distribution_only_smoke_v9970.csv` | `3d46d5a303a59f68bdbd463a7125d891b0ceb9fd385722b8b53f2ece5a1e87da` |
| `p3_G37-tail-coverage-negative-control_distribution_fidelity_v9970.csv` | `a86851d207de3a58a4945f1649c69f5dcd4fb1f0e65365c9f23d22fb77af8b51` |
| `p3_G37-tail-coverage-negative-control_distribution_only_actions_v9970.csv` | `3958df9e1c8b3bfe82e56050248ddc376c6a5baa43214ccbb8adc3fd0d78ebe0` |
| `p3_G37-tail-coverage-negative-control_distribution_only_smoke_v9970.csv` | `c266887c8f64ce384d7d9ef387465909670b41f678a087cba486ea92f8882897` |
| `p3_G38-tail-oversample-importance-weighted-diagnostic_distribution_fidelity_v9970.csv` | `35d658a81bd4108545d9229789d14e6ead7e2f6e27912bb2148e2c7230d0ac33` |
| `p3_G38-tail-oversample-importance-weighted-diagnostic_distribution_only_actions_v9970.csv` | `9f8bafd980051d41b2b65d198d9457e2b4c934f70a7882d02790d61520754240` |
| `p3_G38-tail-oversample-importance-weighted-diagnostic_distribution_only_smoke_v9970.csv` | `1361b9b7d6c081371bc42cd0c8b78c5b2752f1328d5c6bc69b501429bc77ab46` |
| `p3_G39-two-stage-feasible-tail-key-v3-generator_distribution_fidelity_v9970.csv` | `a4cd03ef6039bbbc5dc0fb549129475701b3750a731819ddafe2be927dac7133` |
| `p3_G39-two-stage-feasible-tail-key-v3-generator_distribution_only_actions_v9970.csv` | `ae1759d452e6f4a0fa026e396cc86682a845417f90207bfc8229f20c1fbe0ee6` |
| `p3_G39-two-stage-feasible-tail-key-v3-generator_distribution_only_smoke_v9970.csv` | `ea8d3533a2bdefa5a55e40e8490d159345bae74108af849945b303246a2caa7b` |
| `p3_generator_repair_matrix_v9970.csv` | `58c69daf464ab1fa591648a236b0ab1ecf0eba47c931593fc330688fe15bcb35` |
| `p4_natural_density_group_rates_v9970.csv` | `7f84aa1939876801067620fe7d3adb975a619ea92095f742c98b47a7c11a4ddf` |
| `p4_sequential_natural_density_panel_v9970.csv` | `0c420a5df377a43dbb0a748a16ae3f9adfea8b9eb79f673173f2aba44aed0d80` |
| `p5_future_path_type_revalidation_v9970.csv` | `9f3255ce22f224a928233b2a0455d602337977cd8da4f83bb3a46a3db9e34482` |
| `p6_future_path_operator_sketch_v9_v9970.csv` | `cb359b14df70fcda64c0821efa70b4051178168efe888b196475c6ac600240a0` |
| `p7_existing_action_controller_gate_v9970.csv` | `b709af77ff9f0164c323c20f96b372441a2c48abfc6245d7418f2bb4b792bc1a` |
| `p8_generated_sandbox_gate_v9970.csv` | `40a76b642d13ae2478a3c0d6ccee1f2f57cfe670356dc9bd250bd2c22dba5451` |
| `p9_paired_replay_boundary_v9970.csv` | `8cd54a335adc141d0899c7ee93f51f27101d2caf83cc56ba39367e2505e3e345` |
| `p9_selected_runtime_boundary_v9970.csv` | `ba895de3594209bf61605e430f6af5d3f7b264d43f7247e915e40a31af330b1a` |
| `route_decision_v9970.json` | `4c271c1443e12ee7e822f221f1695f1f577d3fc67ea5c1daf5e259a9d71bb908` |
| `run_manifest_v9970.json` | `9d3965c727a76e994b2041b3ba950647817254b8eba97917f64068bf8f8cc051` |
| `plan` | `fbfd29e0fa6ef2fb892f4b400102b0a3b63c51a3b15cfc435dd311868abef358` |
| `runner` | `d9478cea2a366b51bff48fee3358982ed705aa03324505de24b7a3acb9d44720` |
| `materializer` | `092c458fcf1fa8d53e98cf4a8d1c85010d0595a8bf7b57a82032f2594eaab02f` |

## 10. 最终分析结论

```text
1. v9.9.7 先判断当前 tail fidelity gate 是否在有限 pilot/panel size 下可采样，而不是继续直接堆 generator。
2. 只有当前 tail key 可行或 tail key v3 通过后，G30-G39 才允许进入 distribution-only repair matrix。
3. 只有 G30-G39 出现 weak/official major+tail fidelity pass 后，branch-horizon 和 sequential density panel 才打开。
4. FPO v9 即使诊断运行，也必须在 faithful natural panel context 下才能写成 controller-ready。
5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。
```

最终一句话：v9.9.7 真实执行后停在 `CaseB-TailKeyV3FeasibleGeneratorFail`：a feasible tail key was selected, but G30-G39 did not produce a weak or official major+tail fidelity pass, so branch-horizon and density panels stay blocked.
