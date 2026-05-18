# DG-KAN v9.9.5 Tail Fidelity Repair / Natural Density / FPO 实验复盘

> 本复盘记录 `DG-KAN_v9.9.5_结果解读_TailFidelity修复_自然密度裁决_FPO计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 density/controller/generated/runtime 均显式 `not_run`。

## 0. 最新结论

```text
route = R2-TailKeyPassGeneratorFidelityFail
primary_blocker = no_major_tail_fidelity_generator_passed
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

最终 artifact：`results/real_rerun_20260506/v9950_tail_fidelity_repair_natural_density_fpo_full_20260517T180000Z`

核心结论：

1. P0 复现 v9.9.4 boundary：source route = `CaseC-GeneratorFidelityStillFails`，official pass count = `0`，fake/proxy/cpu = `0` / `0` / `0`。
2. P1 选中 tail key = `TK2-hierarchical-major-tail-key`，weak/strong = `1` / `1`，group count = `337`，support<3 fraction = `0.12166172106824925`。
3. P2 G13-G19 candidate count = `7`，official/weak pass count = `0` / `0`，best = ``。
4. P2 best failed = `G18-sequential-rejection-generator`，major/tail PSI = `0.03448359529300191` / `0.2473464173705414`，missing tail = `23`。
5. P3 largest completed panel = `0`，density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
6. P4 future path weak/strong = `0` / `0`。
7. P5 FPO v7 weak/strong = `0` / `0`；best = `FPO7D-signal-reservoir-snr-gate`，precision = `0.011494252873563218`。
8. P6 controller = `not_run`；P7 generated sandbox allowed = `0`。
9. No-fake audit：rows checked = `7233`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.5 G13-G19 generator profiles。 |
| `experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py` | v9.9.5 runner；执行 P0/P1/P2，按 gate 开 P3-P8，并写 manifest/recap。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py
```

```bash
python experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py --out-dir results/real_rerun_20260506/v9950_tail_fidelity_repair_natural_density_fpo_full_20260517T180000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
```

## 2. Route

```json
{
  "P0_boundary_pass": 1,
  "P1_strong_pass": 1,
  "P1_weak_pass": 1,
  "P2_official_pass_count": 0,
  "P2_weak_pass": 0,
  "P2_weak_pass_count": 0,
  "P3_density_inconclusive": 1,
  "P3_density_insufficient": 0,
  "P3_density_sufficient": 0,
  "P3_largest_completed_panel_size": 0,
  "P4_future_path_strong_pass": 0,
  "P4_future_path_weak_pass": 0,
  "P5_FPO_strong_pass": 0,
  "P5_FPO_weak_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "P8_paired_replay_pass": 0,
  "P8_runtime_pass": 0,
  "best_failed_generator_id": "G18-sequential-rejection-generator",
  "best_generator_id": "",
  "cpu_offload_used": 0,
  "fake_data_used": 0,
  "generated_route_status": "stopped_generator_fidelity_failed",
  "primary_blocker": "no_major_tail_fidelity_generator_passed",
  "proxy_row_used": 0,
  "route": "R2-TailKeyPassGeneratorFidelityFail",
  "route_explanation": "tail key was made sampleable, but G13-G19 did not produce a major+tail fidelity pass, so density/controller/generated stay blocked.",
  "secondary_blocker": "density_panels_blocked",
  "source_route_v9940": "CaseC-GeneratorFidelityStillFails",
  "stage": "ROUTE_DECISION_V9950",
  "status": "summary",
  "system_legal_controller_pass": 0,
  "tail_key_version_selected": "TK2-hierarchical-major-tail-key"
}
```

## 3. P1 Tail Key Audit v2

| tail key | groups | support<3 fraction | max share | entropy ratio | weak | strong |
|---|---:|---:|---:|---:|---:|---:|
| `TK0-original-v9940-tail-key` | `2191` | `0.9361022364217252` | `0.003129346314325452` | `0.9813860637002352` | `0` | `0` |
| `TK1-merged-support3-tail-key` | `342` | `0.11988304093567251` | `0.01773296244784423` | `0.9283016076368611` | `1` | `1` |
| `TK2-hierarchical-major-tail-key` | `337` | `0.12166172106824925` | `0.01773296244784423` | `0.9291338663309526` | `1` | `1` |
| `TK3-precursor-only-no-template-tail-key` | `2090` | `0.9320574162679426` | `0.00521557719054242` | `0.9730140496593591` | `0` | `0` |
| `TK4-memory-offdiag-hardtail-precursor-tail-key` | `154` | `0.23376623376623376` | `0.09075104311543811` | `0.8570441510069715` | `1` | `1` |

## 4. P2 Generator Repair Matrix v2

| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |
|---|---|---|---:|---:|
| `G13-support-aware-hierarchical-generator` | `0.04148246894173764`/`0.06734289331684044`/`0.0234375` | `0.2841625902566213`/`0.138215820239358`/`25`/`0.9258160237388724` | `0` | `0` |
| `G14-tail-balanced-with-major-projection-generator` | `0.042647943010342394`/`0.06868911616712785`/`0.0234375` | `0.27556803068204155`/`0.13611303330516267`/`25`/`0.9258160237388724` | `0` | `0` |
| `G15-min-divergence-transport-generator` | `0.04558356546306973`/`0.06774661569926874`/`0.0234375` | `0.2685438084964744`/`0.13310409908760135`/`25`/`0.9258160237388724` | `0` | `0` |
| `G16-canonical-tail-replay-plus-new-residual-generator` | `0.04514660372384343`/`0.0700614632913517`/`0.0234375` | `0.30889357896677916`/`0.14177122844931772`/`27`/`0.9198813056379822` | `0` | `0` |
| `G17-two-buffer-generator` | `0.20009714987705726`/`0.13021584693926866`/`0.0224609375` | `0.6488183575408967`/`0.2072616302752354`/`65`/`0.8071216617210683` | `0` | `0` |
| `G18-sequential-rejection-generator` | `0.03448359529300191`/`0.06492298915120376`/`0.0234375` | `0.2473464173705414`/`0.13073471127743141`/`23`/`0.9317507418397626` | `0` | `0` |
| `G19-stratified-random-baseline-v2` | `0.3834783838752178`/`0.1875250469363204`/`0.0224609375` | `0.8538533837465137`/`0.24658687539842714`/`69`/`0.7952522255192879` | `0` | `0` |

## 5. P3-P8 Boundary

```text
P3 density = 0 / 0 / 1
P4 future path = 0 / 0
P5 FPO = 0 / 0
P6 controller = not_run
P7 generated = not_run, allowed = 0
```

## 6. P5 FPO v7

| fpo | precision | V LCB | longrisk UCB | cost q90 ms | weak | strong |
|---|---:|---:|---:|---:|---:|---:|
| `FPO7A-tiny-virtual-adamw-1step` | `0.0` | `-0.5540980111257308` | `0.9680115964278527` | `0.00272504985332489` | `0` | `0` |
| `FPO7B-tiny-virtual-adamw-3step` | `0.0` | `-1.4089067124703243` | `1.0` | `0.0014719553291797638` | `0` | `0` |
| `FPO7C-jvp-vjp-gradient-transport` | `0.0` | `-1.284959410019149` | `1.0` | `0.0008717179298400879` | `0` | `0` |
| `FPO7D-signal-reservoir-snr-gate` | `0.011494252873563218` | `-0.2744455578755013` | `0.9446155167180331` | `0.0009918585419654846` | `0` | `0` |
| `FPO7E-hardtail-memory-delayed-gain` | `0.0` | `-1.325763579870307` | `1.0` | `0.0010919757187366486` | `0` | `0` |
| `FPO7F-risk-adjusted-path-type-classifier` | `0.0` | `-1.2934340200879795` | `1.0` | `0.0010919757187366486` | `0` | `0` |
| `FPO7G-slowburn-detector` | `0.0` | `-1.0187429657427693` | `1.0` | `0.0010509975254535675` | `0` | `0` |
| `FPO7H-fastgood-slowburn-two-head` | `0.0` | `-1.2181969684185558` | `1.0` | `0.0011920928955078125` | `0` | `0` |

## 7. No-Fake / Contract / Failure

```text
rows_checked = 7233
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9950.csv` | `585b4de9f495e31255c4925a013ba75a257d3b56f0f82afd65e4c290e81df719` |
| `failure_taxonomy_v9950.csv` | `bfd23d1284285e068afb15d9a3745bb8a0c9c203a9fd6cd78faecfb5d4e19dc9` |
| `fig_p1_tail_support_histogram_v9950.svg` | `c5f484d31dc5aaaa9646762ae651ffa33ead676ed0c8f19d3007ed4caa12563d` |
| `fig_p2_major_tail_psi_scatter_v9950.svg` | `8b8eeecd0d0e55aad5f06175a69f5b035936534b77e3f879984d35c2e48af559` |
| `fig_p2_missing_tail_heatmap_v9950.svg` | `d8bc9b8c2f4eeac512d24d6d3020172fc58841c25e9484700066e00e7f4737e9` |
| `fig_p3_sequential_density_ci_v9950.svg` | `db4f04fb094f4518719ff2cf0c5b7e7e5dd4ffc2d13de8be9b438dd45be2dbb0` |
| `fig_p4_future_path_types_v9950.svg` | `3eb52c9139d836164b2d5a494514d54c06685b8164a595a9808c1a0f3a67a6e2` |
| `fig_p5_fpo_confusion_v9950.svg` | `2f68b24d7e6365324dbea5e31dcc52f8f362c4e862fd542785a9cd95624fa816` |
| `fig_v9950_stop_pivot_matrix.svg` | `9304596aea31b4d26868f8371bfb52d7fb2c3a51c37aeb525a62f3100cf398f7` |
| `no_fake_audit_v9950.csv` | `83adde3a56668695463f68428e7093a5d4c70c09aa4471d6c5ac67d6b62cbd2b` |
| `p0_v9940_boundary_reproduction_v9950.csv` | `f151a2389ebb18423baa14d1d3d6d3632c4b6559a4a3a21963e013e7cbbc01c1` |
| `p1_tail_key_reference_audit_v9950.csv` | `a4766951a1f09253d63d20921146e97485a03221e8b029f6d4c44dbba2515543` |
| `p2_G13-support-aware-hierarchical-generator_distribution_fidelity_v9950.csv` | `9f68c92d2a07009ee6e7538b4408238d7e3a766b63606475e8650635a0b54f30` |
| `p2_G13-support-aware-hierarchical-generator_distribution_only_actions_v9950.csv` | `2ef85baa52a9b6e757d107024d86d1f9416f71d90c8a1791b848b5201423e46f` |
| `p2_G13-support-aware-hierarchical-generator_distribution_only_smoke_v9950.csv` | `1ccfb11551b16cb717e595b5aa28353aa2593153c7290db666eafc90038d101c` |
| `p2_G14-tail-balanced-with-major-projection-generator_distribution_fidelity_v9950.csv` | `302880ab6933c59ab48df5fc5f9d8d4fb79461c765fa0e55c248ff317c63e529` |
| `p2_G14-tail-balanced-with-major-projection-generator_distribution_only_actions_v9950.csv` | `8037c1d5837ed525d42ee7496413f37b901fa1a58afd9e4a086a7c44c80e5f65` |
| `p2_G14-tail-balanced-with-major-projection-generator_distribution_only_smoke_v9950.csv` | `7e5e6d44ccf7451cb0dc2005be7682ef9d5d135d12c3c728943eb9566e8d0d2d` |
| `p2_G15-min-divergence-transport-generator_distribution_fidelity_v9950.csv` | `3580b476549ce7b4b0cd84949e7899187311fe925d98f4c08b0f24e737c846e0` |
| `p2_G15-min-divergence-transport-generator_distribution_only_actions_v9950.csv` | `f357c37029c93e9f0bdc1521a1e13a31a20a580b5778d1ce66c5c1396c5c00cb` |
| `p2_G15-min-divergence-transport-generator_distribution_only_smoke_v9950.csv` | `adafac5918feda4a7cd56334342dbfa2b7858ed8a29abfdc7a2f8f2cc0f3ef3c` |
| `p2_G16-canonical-tail-replay-plus-new-residual-generator_distribution_fidelity_v9950.csv` | `ca48d412ea74ea3c5c6c48626e692d1d2f4de18d71f5ce390159c13659ed5b60` |
| `p2_G16-canonical-tail-replay-plus-new-residual-generator_distribution_only_actions_v9950.csv` | `17c83897bf34f8a6ca5f280fb58f0fe853b96c84add2300ca62d23f7c85f761a` |
| `p2_G16-canonical-tail-replay-plus-new-residual-generator_distribution_only_smoke_v9950.csv` | `c35c92231b8a3a7b8702db1c59a4fb81b4a6fa8f5ab645139b00ad069f7f2da1` |
| `p2_G17-two-buffer-generator_distribution_fidelity_v9950.csv` | `0bc20fdf320fdcf9b2c72a686e6bc8108fde18fd68e3eade7c122b2004b062a9` |
| `p2_G17-two-buffer-generator_distribution_only_actions_v9950.csv` | `6d9a92ad134d952942ec6c1f1f695409bff2340932a4f2dff6b3a3f7b929e1fe` |
| `p2_G17-two-buffer-generator_distribution_only_smoke_v9950.csv` | `e05f0537a916b7ae39a9e754ea64173b96c68ad5aff9eea9e6bb0d13b842706e` |
| `p2_G18-sequential-rejection-generator_distribution_fidelity_v9950.csv` | `59f3cb89ff9a840505b4835c2c332ccafe19895e6896ad3bd0368a917ac764c5` |
| `p2_G18-sequential-rejection-generator_distribution_only_actions_v9950.csv` | `9f25c6e7f043624c8451cca7aeaaf7f2b61c5135b011e05a4acb20d66c6a3be1` |
| `p2_G18-sequential-rejection-generator_distribution_only_smoke_v9950.csv` | `4a1369bd743b0926a59baf52a4f19f21f7fc374dc0d3fbe4f0a3827370d29d76` |
| `p2_G19-stratified-random-baseline-v2_distribution_fidelity_v9950.csv` | `f924085b67c704269780038fd15ca4f45e56ce6ed9ccd17fd312e723207496bb` |
| `p2_G19-stratified-random-baseline-v2_distribution_only_actions_v9950.csv` | `34215d2fc1c955c2dfd01fb92e7b455bc59dcced6fd953a1922d35100ccc484c` |
| `p2_G19-stratified-random-baseline-v2_distribution_only_smoke_v9950.csv` | `a66fe673da79bfdabe6961f7a7776d757e6fab59a91415c2d704598c1ff43063` |
| `p2_generator_repair_matrix_v9950.csv` | `21dbea15463441b546623ed29bd4045ee50f9924453ec1342bf620fd6f0e94dc` |
| `p3_natural_density_group_rates_v9950.csv` | `f871ceef9d099bc123724c0436e7c946c02ef7a14fc56236c59dc42ee3481336` |
| `p3_sequential_natural_density_panel_v9950.csv` | `8a30d299384efeb1b6b0412c9320be6063319b07325e9de551b5143c9ed8bca8` |
| `p4_future_path_type_revalidation_v9950.csv` | `3513b002044918d1b26a9b179ef2059902232cadf73c67a8ecc6fa6845f18139` |
| `p5_future_path_operator_sketch_v7_v9950.csv` | `d4f579fde3bfcc28618b99b4f36ca5ccc6e320bbdd9b3f8d3de0379ed1e476af` |
| `p6_existing_action_controller_gate_v9950.csv` | `c838fedc0eefa97256482795ee7190f10c87a0d42095f2ce2b142113b7b03db3` |
| `p7_generated_sandbox_gate_v9950.csv` | `afa4e2ddc24230f317fc324391ab0aaecb862cc9a652416a740acf3c5d9cbcd2` |
| `p8_paired_replay_boundary_v9950.csv` | `078f1f1c0c8cc7495cc99e99c7bc76b4bfe7cd44b9ff299cff1c8615779a0f43` |
| `p8_selected_runtime_boundary_v9950.csv` | `d85477e56293cecf42dee8d458139640bee4d09ef43f31e08444eacb859d05ea` |
| `route_decision_v9950.json` | `753490eb7f8287f773d2a59a3cef6f95f7161e611ae61c0e88443ae7b997cbde` |
| `run_manifest_v9950.json` | `07ab6040524267a50ce068af6043a46a6b2c646c344ee3da21779791304b979e` |
| `plan` | `adc8da60def348d0738864e968c6406a252d7e463f05d1a29b0c6f8dd23f078b` |
| `runner` | `cafd3388d07033708a3a6aafafb908353049c8e315bd5ca05c1e5a42b5751e82` |
| `materializer` | `5c3f7c6233e04ed968c14487f737073c23ddb42931f241a20f3d3f5669842e85` |

## 9. 最终分析结论

```text
1. v9.9.5 没有直接用 v9.9.4 的 tail key 继续加 generator，而是先审计 TK0-TK4。
2. 只有 P1 选出的 commit-time tail key 通过后，P2 G13-G19 才进入 distribution-only 1024。
3. 只有 P2 official/weak pass 才允许打开 P3 branch-horizon 和 sequential density panel。
4. P5 FPO v7 即使诊断运行，也必须在 official fidelity context 下才能写成 controller-ready。
5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。
```

最终一句话：v9.9.5 真实执行后停在 `R2-TailKeyPassGeneratorFidelityFail`：tail key was made sampleable, but G13-G19 did not produce a major+tail fidelity pass, so density/controller/generated stay blocked.
