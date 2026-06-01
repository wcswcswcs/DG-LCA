# DG-KAN v9.9.6 Tail Fidelity Root Cause / Natural Density / FPO Rewrite 实验复盘

> 本复盘记录 `DG-KAN_v9.9.6_结果解读_TailFidelity根因_自然密度裁决_FPO重写完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 branch-horizon、density、controller、generated、runtime 均显式 `not_run`。

## 0. 最新结论

```text
route = RouteD-TailFidelityStillFails
primary_blocker = no_major_tail_fidelity_generator_passed_after_root_cause_repair
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

最终 artifact：`results/real_rerun_20260506/v9960_tail_fidelity_rootcause_natural_density_fpo_rewrite_full_20260517T190000Z`

核心结论：

1. P0 复现 v9.9.5 boundary：source route = `R2-TailKeyPassGeneratorFidelityFail`，official/weak pass count = `0` / `0`，fake/proxy/cpu = `0` / `0` / `0`。
2. P1 missing-tail root-cause attribution complete = `1`；assigned/unknown fraction = `1.0` / `0.0`；top reason = `MR8-tail-key-too-fine`。
3. P1 root-cause counts：MR1 quota rounded = `30`，MR5 cursor collapse = `12`，MR6 major-tail conflict = `55`，MR8 tail key too fine = `162`。
4. P2 G20-G29 candidate count = `10`，official/weak pass count = `0` / `0`，best = ``。
5. P2 best failed = `G26-canonical-tail-precursor-replay-generator`，major/tail PSI = `0.029000497474792362` / `0.24015656834967286`，missing tail = `24`。
6. P3 largest completed panel = `0`，density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
7. P4 future path weak/strong = `0` / `0`。
8. P5 FPO v8 weak/strong = `0` / `0`；best = `FPO8G-signal-channel-drift-diffusion-sketch`，precision = `0.011494252873563218`，V LCB = `-0.383912815103318`。
9. P6 controller = `not_run`；P7 generated sandbox allowed = `0`。
10. No-fake audit：rows checked = `12673`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.6 G20-G29 generator profiles，并修复新 payload norm bucket 对齐。 |
| `experiments/run_v9960_tail_fidelity_rootcause_natural_density_fpo_rewrite.py` | v9.9.6 runner；执行 P0/P1 root-cause/P2 repair/FPO v8，按 gate 开 P3-P8，并写 manifest/recap。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9960_tail_fidelity_rootcause_natural_density_fpo_rewrite.py
```

```bash
python experiments/run_v9960_tail_fidelity_rootcause_natural_density_fpo_rewrite.py --out-dir results/real_rerun_20260506/v9960_tail_fidelity_rootcause_natural_density_fpo_rewrite_full_20260517T190000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9960",
  "status": "summary",
  "route": "RouteD-TailFidelityStillFails",
  "primary_blocker": "no_major_tail_fidelity_generator_passed_after_root_cause_repair",
  "secondary_blocker": "density_panels_blocked",
  "route_explanation": "P1 attribution completed, but G20-G29 did not produce a weak or official major+tail fidelity pass, so branch-horizon and density panels stay blocked.",
  "source_route_v9950": "R2-TailKeyPassGeneratorFidelityFail",
  "P0_boundary_pass": 1,
  "P1_attribution_complete": 1,
  "P1_assigned_missing_reason_fraction": 1.0,
  "P1_unknown_missing_reason_fraction": 0.0,
  "P1_top_missing_reason": "MR8-tail-key-too-fine",
  "tail_key_version_selected": "TK2-hierarchical-major-tail-key",
  "P2_official_pass_count": 0,
  "P2_weak_pass_count": 0,
  "P2_weak_pass": 0,
  "best_generator_id": "",
  "best_failed_generator_id": "G26-canonical-tail-precursor-replay-generator",
  "P3_largest_completed_panel_size": 0,
  "P3_density_sufficient": 0,
  "P3_density_insufficient": 0,
  "P3_density_inconclusive": 1,
  "P4_future_path_weak_pass": 0,
  "P4_future_path_strong_pass": 0,
  "P5_FPO_weak_pass": 0,
  "P5_FPO_strong_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_generator_fidelity_failed",
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Missing Tail Root-Cause

```text
generator_count = 7
tail_group_count = 337
root_cause_row_count = 2359
missing_tail_row_count = 259
assigned_missing_reason_fraction = 1.0
unknown_missing_reason_fraction = 0.0
top_missing_reason = MR8-tail-key-too-fine
P1_attribution_complete = 1
```

| reason | count |
|---|---:|
| `MR1-quota-rounded-to-zero` | `30` |
| `MR2-no-precursor-available` | `0` |
| `MR4-dedup-or-collision-deleted-tail` | `0` |
| `MR5-cursor-collapse` | `12` |
| `MR6-major-tail-quota-conflict` | `55` |
| `MR8-tail-key-too-fine` | `162` |

## 4. P2 Generator Repair Matrix

| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |
|---|---|---|---:|---:|
| `G20-minquota-tail-refill-generator` | `0.02819187377657834`/`0.05861348330099616`/`0.0234375` | `0.33562679311847404`/`0.14321669295217954`/`26`/`0.9228486646884273` | `0` | `0` |
| `G21-mincost-flow-major-tail-generator` | `0.02819187377657834`/`0.05861348330099616`/`0.0234375` | `0.26906816524487975`/`0.13496319254709427`/`25`/`0.9258160237388724` | `0` | `0` |
| `G22-tail-first-major-residual-generator` | `0.12379445342757801`/`0.10786613293355547`/`0.0244140625` | `0.47067177922157716`/`0.18125047753915718`/`45`/`0.8664688427299704` | `0` | `0` |
| `G23-major-first-tail-residual-generator` | `0.13315762675905876`/`0.11218340677843491`/`0.021484375` | `0.6345231655890194`/`0.2042909167139296`/`58`/`0.827893175074184` | `0` | `0` |
| `G24-per-tail-independent-cursor-generator` | `0.029000497474792362`/`0.05946569902367241`/`0.0234375` | `0.26764102894391156`/`0.13515158980333616`/`25`/`0.9258160237388724` | `0` | `0` |
| `G25-tail-repair-after-filter-generator` | `0.02819187377657834`/`0.05861348330099616`/`0.0234375` | `0.2619103944511142`/`0.13341094038305146`/`25`/`0.9258160237388724` | `0` | `0` |
| `G26-canonical-tail-precursor-replay-generator` | `0.029000497474792362`/`0.05946569902367241`/`0.0234375` | `0.24015656834967286`/`0.12522362778561455`/`24`/`0.9287833827893175` | `0` | `0` |
| `G27-entropy-regularized-generator` | `0.02819187377657834`/`0.05861348330099616`/`0.0234375` | `0.2939298115358346`/`0.13737672264963904`/`26`/`0.9228486646884273` | `0` | `0` |
| `G28-importance-weighted-density-generator` | `0.13722312672176834`/`0.11842771514605493`/`0.0244140625` | `0.5704317393321812`/`0.19785816789051167`/`53`/`0.8427299703264095` | `0` | `0` |
| `G29-two-stage-mixture-generator` | `0.09009676937194736`/`0.0963950902711423`/`0.0224609375` | `0.4486738415393885`/`0.17592949423716475`/`40`/`0.8813056379821959` | `0` | `0` |

## 5. P3-P8 Boundary

```text
P3 density = 0 / 0 / 1
P4 future path = 0 / 0
P5 FPO = 0 / 0
P6 controller = not_run
P7 generated = not_run, allowed = 0
```

## 6. P5 FPO v8

| fpo | Fast/Slow/Path precision | V/V240 LCB | longrisk UCB | cost q90 ms | weak | strong |
|---|---|---|---:|---:|---:|---:|
| `FPO8A-slowburn-signal-reservoir-sketch` | `0.0`/`0.0`/`0.0` | `-1.12502193128993`/`-1.1157596802793937` | `1.0` | `0.0015720725059509277` | `0` | `0` |
| `FPO8B-tiny-virtual-adamw-path-3step` | `0.0`/`0.0`/`0.0` | `-1.379096154552172`/`-1.3453874672664443` | `1.0` | `0.001443084329366684` | `0` | `0` |
| `FPO8C-jvp-gradient-transport-risk-veto` | `0.0`/`0.0`/`0.0` | `-1.3139449768920153`/`-1.1219098217332373` | `0.997968144575522` | `0.0011515803635120392` | `0` | `0` |
| `FPO8D-memory-hardtail-delayed-gain-sketch` | `0.0`/`0.0`/`0.0` | `-0.8527094262954846`/`-0.6196319202639751` | `1.0` | `0.0013117678463459015` | `0` | `0` |
| `FPO8E-fastgood-slowburn-two-head-sketch` | `0.0`/`0.0`/`0.0` | `-0.9472660345967531`/`-0.9322829569731912` | `1.0` | `0.0014523975551128387` | `0` | `0` |
| `FPO8F-risky-high-AUV-veto-sketch` | `0.0`/`0.0`/`0.0` | `-1.3536995301971524`/`-1.5502016809977284` | `1.0` | `0.001271720975637436` | `0` | `0` |
| `FPO8G-signal-channel-drift-diffusion-sketch` | `0.0`/`0.0`/`0.011494252873563218` | `-0.383912815103318`/`-0.23272393384564966` | `0.952667674022474` | `0.0010421499609947205` | `0` | `0` |
| `FPO8H-lowrank-future-path-operator-sketch` | `0.0`/`0.0`/`0.0` | `-1.1523160253075244`/`-1.1865208962282279` | `1.0` | `0.0011222437024116516` | `0` | `0` |

## 7. No-Fake / Contract / Failure

```text
rows_checked = 12673
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9960.csv` | `2e51f88d1da0a002d79a3a5ae3803dd993015721a7f538704428d490cf2e8140` |
| `failure_taxonomy_v9960.csv` | `0589abfe69df07cb2f1030a570834e5bee0309f83f17d4e0a3eef5694832fc0b` |
| `fig_p1_missing_tail_root_cause_v9960.svg` | `96db7e5274e3ce83a3e560d2d4064245836f0f6a840336fbf49083f85dbc0641` |
| `fig_p2_major_tail_psi_scatter_v9960.svg` | `a56535f44024acad2ea35fd1bb31faf63215c0fb369c6c2d9ecaa300e6df8fac` |
| `fig_p2_missing_tail_heatmap_v9960.svg` | `34caafed94358255294ed0ed4c0467ea37b40062cdd0db81edc46be48ae098ea` |
| `fig_p3_sequential_density_ci_v9960.svg` | `db4f04fb094f4518719ff2cf0c5b7e7e5dd4ffc2d13de8be9b438dd45be2dbb0` |
| `fig_p4_future_path_types_v9960.svg` | `3eb52c9139d836164b2d5a494514d54c06685b8164a595a9808c1a0f3a67a6e2` |
| `fig_p5_fpo_confusion_v9960.svg` | `73d19ad582b70bfa6e8d22663059f76c1887c476ec4ead6ec4131d154a0c8df3` |
| `fig_v9960_stop_pivot_matrix.svg` | `9304596aea31b4d26868f8371bfb52d7fb2c3a51c37aeb525a62f3100cf398f7` |
| `no_fake_audit_v9960.csv` | `9da6636e000802e76ce1e967732b0f7a724744eed6d251d0345471fc815e2bf4` |
| `p0_v9950_boundary_reproduction_v9960.csv` | `1f4f13118145ac269ad92f223de70ef5d1e27d480430364a9fc4e430ca4d183a` |
| `p1_missing_tail_root_cause_audit_v9960.csv` | `94c2b8208aaee5c0a000086bd1af0720af96b766948fb14a1401f22553c18bc2` |
| `p2_G20-minquota-tail-refill-generator_distribution_fidelity_v9960.csv` | `9db73b3d0bf82a3224812824e344763727841a73e285dcdf1790fcb3635fdaf0` |
| `p2_G20-minquota-tail-refill-generator_distribution_only_actions_v9960.csv` | `de46f2f8d91f0eeebd4bf5462a421935ba3ae165caae720071d348947b1317a2` |
| `p2_G20-minquota-tail-refill-generator_distribution_only_smoke_v9960.csv` | `cec49153c567e313e9be76ecdc91080f687703c9293901c748653ce96f9091dd` |
| `p2_G21-mincost-flow-major-tail-generator_distribution_fidelity_v9960.csv` | `6de7a66d7239c38b27cfad867068fbe8f17940ba737dc9f28b278cb79faa9b02` |
| `p2_G21-mincost-flow-major-tail-generator_distribution_only_actions_v9960.csv` | `77d58829042eac580e1098e8d7331b2994920db8f87ff299728401c7849962b4` |
| `p2_G21-mincost-flow-major-tail-generator_distribution_only_smoke_v9960.csv` | `cded95fdf85eca246694b82eb56208338b5cb5582b010bc2d0ba06fc1342a29d` |
| `p2_G22-tail-first-major-residual-generator_distribution_fidelity_v9960.csv` | `812016fbe09a47704c4e58ba302d23334389a75fb3d664ec0872f588ecf51f29` |
| `p2_G22-tail-first-major-residual-generator_distribution_only_actions_v9960.csv` | `f3bfdd4f5dc4e753665057ec87a943d957063a3e62b0cf1d8e6a7c33a7788f9e` |
| `p2_G22-tail-first-major-residual-generator_distribution_only_smoke_v9960.csv` | `f30161282d8df42cb0fe3add5dc5df6eb14958c188527825cf43404c65bc04e1` |
| `p2_G23-major-first-tail-residual-generator_distribution_fidelity_v9960.csv` | `bf5e626702d3f92da109c5c6e87c290d771c34a0eb09f0eee95c79959231396e` |
| `p2_G23-major-first-tail-residual-generator_distribution_only_actions_v9960.csv` | `9f1cd8337cd502fadea59476ff9e7d1a8b77e390eba01d11759408e7c0f43990` |
| `p2_G23-major-first-tail-residual-generator_distribution_only_smoke_v9960.csv` | `5210b315fa0acd4f9963059f50d69fc48d3c9925937c9f12b56bf41bb3996fa0` |
| `p2_G24-per-tail-independent-cursor-generator_distribution_fidelity_v9960.csv` | `00db6fb83c6179f6a7e8966e4571e87b12d2589b5b7808fb38e188ac6467c2ec` |
| `p2_G24-per-tail-independent-cursor-generator_distribution_only_actions_v9960.csv` | `9f136b4ed01d10260895cf19fd7f4a4855302d7afbe4b10c007ea9e8e73c4270` |
| `p2_G24-per-tail-independent-cursor-generator_distribution_only_smoke_v9960.csv` | `90f78820877679ee6d7357abfab3d4264fa0fbe9990a0311c59f73a39565cdd0` |
| `p2_G25-tail-repair-after-filter-generator_distribution_fidelity_v9960.csv` | `882fffdf60e9f8cc5c23fd40a1fe65fbeee6325f292e79f77b23436e86eac80a` |
| `p2_G25-tail-repair-after-filter-generator_distribution_only_actions_v9960.csv` | `10d99f12ec0335943349a7a1df28b0774e11af9a158ec83a30262bc836c611d6` |
| `p2_G25-tail-repair-after-filter-generator_distribution_only_smoke_v9960.csv` | `15a1ce9725fece5dd402eeeaa27dcfc770ab6e1dd1c4f0ff8d8da3bcd3d5ad2a` |
| `p2_G26-canonical-tail-precursor-replay-generator_distribution_fidelity_v9960.csv` | `4ab68d16ad8048996965a084bece0c6af6a133733cf4d2f871ae5295e92ce299` |
| `p2_G26-canonical-tail-precursor-replay-generator_distribution_only_actions_v9960.csv` | `27a66ff9917a97007ff7c2c39352fccb44aa50f698bc25d725e39e2b9c592b2c` |
| `p2_G26-canonical-tail-precursor-replay-generator_distribution_only_smoke_v9960.csv` | `c50a7c6720b9688aea5f913eaf6633e7c9c36f166dcca13f5325c0f0a3d92475` |
| `p2_G27-entropy-regularized-generator_distribution_fidelity_v9960.csv` | `5969b940985d1a3be4f07d8b542288477f7f831dffecac42cb8dae47a0e47363` |
| `p2_G27-entropy-regularized-generator_distribution_only_actions_v9960.csv` | `8d0226015cbc6cdca13b1aea99c03fac0aed66f94ff1fab1eaf6bbdb8ef7c45e` |
| `p2_G27-entropy-regularized-generator_distribution_only_smoke_v9960.csv` | `aceacae0ad175dd1596abff474a542e706502d02d5c8baf2cb45edff21758b02` |
| `p2_G28-importance-weighted-density-generator_distribution_fidelity_v9960.csv` | `26aecbb40e95805ac779b79ce792933933460357419a3af63536ce6beac4e9a1` |
| `p2_G28-importance-weighted-density-generator_distribution_only_actions_v9960.csv` | `c186b586855b9f0041acefefcab68f16ee2363e8c162861e63ba830d9eba0a90` |
| `p2_G28-importance-weighted-density-generator_distribution_only_smoke_v9960.csv` | `1fa3b06397ea83ade831978aeee152530bef66cbef3270ca2e1dc7cc8e9dfcbe` |
| `p2_G29-two-stage-mixture-generator_distribution_fidelity_v9960.csv` | `e730fa330eee74c1b37c77c8583b29d5fff0684851747eb08e8a2a5354a3d432` |
| `p2_G29-two-stage-mixture-generator_distribution_only_actions_v9960.csv` | `04e2d0a6d6445747ec3babd93f1ac5db3db1b4160bc067b26ff33d5d6c96ae9a` |
| `p2_G29-two-stage-mixture-generator_distribution_only_smoke_v9960.csv` | `5aa5b4cefe67ce6817358473f8545643bc7d073f0b2f28311f89a03033d6bbe6` |
| `p2_generator_repair_matrix_v9960.csv` | `ed2655cbcc831cc99d130e7c004795964a87f7d8d4b501258442639898bc92bc` |
| `p3_natural_density_group_rates_v9960.csv` | `aa855cd769b9607ca7fdc215453b93a4b70e5e4457c1ccdcc8d1ffcd5288a55d` |
| `p3_sequential_natural_density_panel_v9960.csv` | `12b49b65f41c8c84cc68656e2ca0d3f8b3a4e3285bb5fc9c1b4276369478e3e4` |
| `p4_future_path_type_revalidation_v9960.csv` | `44863f42249aae6b36b1c13b736976cf62ec5b4b3194f4dd9f8673fa04fcad57` |
| `p5_future_path_operator_sketch_v8_v9960.csv` | `f5eb49f4defac936778c13dc31e9fd582bd5be20648bc2132872eb1bc8351b73` |
| `p6_existing_action_controller_gate_v9960.csv` | `2ae8ffe7dfe0627e8493b2dcca045c50caf1e7284ae8dd5bb22155768021102b` |
| `p7_generated_sandbox_gate_v9960.csv` | `047fde7479a7f44dacde4f9b6401272a7b3899e7f3df37b60de06b351b081fb0` |
| `p8_paired_replay_boundary_v9960.csv` | `d3fd90fb72ea71f8ff6506153043ab2545037629d11f940222f8271b50721be0` |
| `p8_selected_runtime_boundary_v9960.csv` | `0ab2e89327d12a1b7bc684101024f8432cb400129edb99d991b48da121e61b22` |
| `route_decision_v9960.json` | `2d4016a7e386c7820e2de48ae17d88c9b7cff1192785ac13182c369053c43993` |
| `run_manifest_v9960.json` | `f72b77141992bee9a6d033dc4ff20c9a3009fc485736370eca6e18835a1883dd` |
| `plan` | `bdcc1a0eaa32955a936e7a6cf16c1116bd5b13a17efa9eaba5468eeb24d0ef0d` |
| `runner` | `d02443ae757388b9a71e7ba97852b15c6978ac364362f8f9cb720ef4803c331c` |
| `materializer` | `f54d82649675d4eb767935d52a562c64cf627c66418ae45f1c38fbf6f415c878` |

## 9. 最终分析结论

```text
1. v9.9.6 先追溯 v9.9.5 G13-G19 的 missing-tail root cause，不把 tail key 已通过误写成 generator 已通过。
2. P1 attribution 达到合同后才尝试 G20-G29；若没有 weak/official fidelity pass，则不打开 branch-horizon 或 density panel。
3. G28 是 importance-weighted diagnostic，不允许作为 official density generator。
4. P5 FPO v8 即使诊断运行，也必须在 official fidelity context 下才能写成 controller-ready。
5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。
```

最终一句话：v9.9.6 真实执行后停在 `RouteD-TailFidelityStillFails`：P1 attribution completed, but G20-G29 did not produce a weak or official major+tail fidelity pass, so branch-horizon and density panels stay blocked.
