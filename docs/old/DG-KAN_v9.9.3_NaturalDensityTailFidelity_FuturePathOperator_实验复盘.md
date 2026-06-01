# DG-KAN v9.9.3 Natural Density / Tail Fidelity / FuturePathOperator 实验复盘

> 本复盘记录 `DG-KAN_v9.9.3_结果解读_自然密度裁决_TailFidelity_FuturePathOperator_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 与本轮真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。tail fidelity 未通过或 density 已在合法 sequential gate 裁决时，后续 panel 均显式 `not_run`。

## 0. 最新结论

```text
route = R9-MaterializerOrFidelityEngineeringBlocked
primary_blocker = major_and_tail_fidelity_failed_after_repair_attempt
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_fidelity_failed
```

最终 artifact：`results/real_rerun_20260506/v9930_natural_density_tail_fidelity_futurepathoperator_full_20260517T040000Z`

核心结论：

1. P0 复现 v9.9.2 boundary：source route = `R2-DistributionFidelityPassDensityPending`，best generator = `G5-hybrid-quota-random-generator`，PSI/JS/max-share/entropy = `0.016435209021362127` / `0.045276968538283786` / `0.2626953125` / `0.9355692755194838`。
2. P1 selected generator = `G5-hybrid-quota-random-generator`，major/tail pass = `0` / `0`；tail PSI/JS/missing = `0.2983309215401979` / `0.1431685310441598` / `13`。
3. Tail repair attempt = `G6-tail-aware-quota-generator`，G5/G6 tail pass = `0` / `0`，reason = `G6_tail_repair_did_not_pass_full_major_tail_gate`。
4. P2 largest completed panel = `0`，density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
5. P2 CoreLike count/LCB/UCB = `0` / `0` / `0`；PathGood count/LCB/UCB = `0` / `0` / `0`；Core+Slow UCB = `0`。
6. P3 source adequacy decision = `not_run`，raw/IW density = `None` / `None`。
7. P4 future-path weak/strong = `0` / `0`。
8. P5 FPO v5 weak/strong = `0` / `0`；best = `None`，precision = `None`。
9. P6 controller = `not_run`；P7 generated sandbox allowed = `0`。
10. No-fake audit：rows checked = `67640`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加/保留 G5 与 G6 tail-aware profile；G6 不使用 outcome label，只按 canonical AP0 precursor 分组轮转。 |
| `experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py` | v9.9.3 runner；执行 P0/P1 tail fidelity、必要的 tail repair、P2 sequential density、P3 anatomy、P4/P5/P6-P8 gates。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py
```

```bash
python experiments/run_v9930_natural_density_tail_fidelity_futurepathoperator.py --out-dir results/real_rerun_20260506/v9930_natural_density_tail_fidelity_futurepathoperator_full_20260517T040000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9930",
  "status": "summary",
  "route": "R9-MaterializerOrFidelityEngineeringBlocked",
  "primary_blocker": "major_and_tail_fidelity_failed_after_repair_attempt",
  "secondary_blocker": "density_panels_blocked",
  "source_route_v9920": "R2-DistributionFidelityPassDensityPending",
  "selected_generator_id": "G5-hybrid-quota-random-generator",
  "P0_boundary_pass": 1,
  "P1_major_tail_pass": 0,
  "P1_major_fidelity_pass": 0,
  "P1_tail_fidelity_pass": 0,
  "P1_PSI_major_max": 0.016903746143043054,
  "P1_PSI_tail_max": 0.2983309215401979,
  "P1_missing_tail_group_count": 13,
  "P2_largest_completed_panel_size": 0,
  "P2_density_sufficient": 0,
  "P2_density_insufficient": 0,
  "P2_density_inconclusive": 1,
  "CoreLike_count": 0,
  "CoreLike_LCB": 0,
  "CoreLike_UCB": 0,
  "PathGood_count": 0,
  "PathGood_LCB": 0,
  "PathGood_UCB": 0,
  "P4_future_path_weak_pass": 0,
  "P4_future_path_strong_pass": 0,
  "P5_FPO_weak_pass": 0,
  "P5_FPO_strong_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_fidelity_failed",
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Tail Fidelity

```text
generator = G5-hybrid-quota-random-generator
major pass = 0
tail pass = 0
PSI_major_max = 0.016903746143043054
PSI_tail_max = 0.2983309215401979
JS_tail_max = 0.1431685310441598
missing_tail_group_count = 13
candidate_template_coverage = 1.0
Core77/OldOnly/RiskClean coverage = 1.0 / 1.0 / 1.0
```

## 4. P2 Density Decision

```text
largest_completed_panel_size = 0
CoreLike count/LCB/UCB = 0 / 0 / 0
PathGood count/LCB/UCB = 0 / 0 / 0
CoreLike+SlowBurn UCB = 0
density sufficient/insufficient/inconclusive = 0 / 0 / 1
reason = P1_major_tail_fidelity_failed_do_not_open_density_panels
```

## 5. Boundary

```text
P4 future path = 0 / 0
P5 FPO = 0 / 0
P6 controller = not_run, reason = P2_density_not_sufficient_and_P5_strong_not_passed
P7 generated = not_run, allowed = 0
P8 runtime/paired replay = not_run
```

## 6. No-Fake / Contract / Failure

```text
rows_checked = 67640
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9930.csv` | `febacd378b107560064275e489c45308b3f388a24a81d0be34f930f728cb528f` |
| `failure_taxonomy_v9930.csv` | `8d4b364586a9cfae138f2cdd6e41c9638f6465cca8b4f1d7f33b444898c89568` |
| `fig_p1_tail_fidelity_v9930.svg` | `565042ecf355b6da894e775c5cf743c7dde9559b81e92207323c70d2d8af9e76` |
| `fig_p2_density_ci_v9930.svg` | `4eb7cf77e3b694afbbf7ec6135ad4b9ba15f32b7e70a236bd443c690eedfc549` |
| `fig_p5_fpo_best_v9930.svg` | `3d5c8fbff8a540aa33f1ed1041c283af575f1c5a86d42f659113544049e29f5b` |
| `fig_v9930_gate_matrix.svg` | `6c45e8eab4595a2505e69b23dc7e865ac84eecd37fc62a57d0a72bdff5d9d0ad` |
| `materializer` | `9c41d2b18a0181cb0c31b0b8987c44b7b27890d56b47666bfb2244a143e60018` |
| `no_fake_audit_v9930.csv` | `23b1a3f254d47e1ebd4e166c659c888c1971c24838ba8f0f7431acc0ba6d185d` |
| `p0_v9920_boundary_reproduction.csv` | `889e165e1ffb61b25a42bfa8038c14d7b43f6d5c0c3e5742a67cb2494f822d6f` |
| `p1_g5_1024_action_apply_replay_v9930.csv` | `04627962a70ea69eb9226feb1ccd8b26368b689756df54ef993c0318351ff373` |
| `p1_g5_1024_action_labels_v9930.csv` | `3d5fef9ccaea8819ab38475f8d0d55325a89ebfaaa4e1b99db42a74cbbf0644f` |
| `p1_g5_1024_action_rows_v9930.csv` | `0811ab300930fd2ac168de67af996ac8582947119265be8754a4587c7853f13c` |
| `p1_g5_1024_branch_horizon_v9930.csv` | `cfb27afadce8d4533e94cee14ccef1bdb894e43f1633029e6eb56da8176092c1` |
| `p1_g5_1024_panel_smoke_v9930.csv` | `07d2a8e370bf9078f9c3f8260c55da370f9d9de0d154f95ad16476313bb9d987` |
| `p1_g6_tail_repair_1024_action_apply_replay_v9930.csv` | `d5ac9fac36e9e111ff25f65ce08e14543fed4fa662df61343072d99a094ab87a` |
| `p1_g6_tail_repair_1024_action_labels_v9930.csv` | `647959deec9d3614a4ef591d9b58870d3f1f7a6fade781daa1c7a78998027285` |
| `p1_g6_tail_repair_1024_action_rows_v9930.csv` | `f2cbdebeddffb5dfb9bd6400d56179f79d72c7ad45bdcf560eb9faddc37ebda5` |
| `p1_g6_tail_repair_1024_branch_horizon_v9930.csv` | `4da2ec98764bd61ebccd802f0bf06a1c1e9d371a8065fd33b3991d24bf7606fb` |
| `p1_g6_tail_repair_1024_panel_smoke_v9930.csv` | `82212730cf31f38ea5ef65106fa0a579c0f4be4c2365b8afd3b3d6614c67ddc1` |
| `p1_g6_tail_repair_fidelity_audit_v9930.csv` | `bb9e061c033b47354662e2312c32f51a48f53109a76fad78c0271901e1986b7e` |
| `p1_major_tail_fidelity_audit_v9930.csv` | `0041e45b521b8631cc512805b469ca243c0615935baa06ddc9dbcbbe6667a69b` |
| `p1_tail_repair_attempt_v9930.csv` | `22e7f180104e38cf72c201736c1e7238527d4023ac0a1f33adf421d1fd55087f` |
| `p2_natural_density_group_rates_v9930.csv` | `587fcd4a6da87ac1ffcee97503ee7e8e5c6fade97bd4a3865e9f24fdda2068d2` |
| `p2_sequential_natural_density_panel_v9930.csv` | `11c97aa3c26920db20d9b5e830cf142e0ca376807db33ea188dd1f898afe94d6` |
| `p3_density_anatomy_source_adequacy_v9930.csv` | `193e22a5769996af46406cc020d0a6138cb4af72751a1cf49e383776b74dafb6` |
| `p4_future_path_type_revalidation_v9930.csv` | `6e968968949956fbf606a98ea74d735862b5a54d9fe14d1f4005c2c01d0e3dc2` |
| `p5_future_path_operator_sketch_v5_v9930.csv` | `6186b7ce7d6907eee8460c89a58753b7ff3e942b594e7445e11c4f2ccac5fc08` |
| `p6_existing_action_controller_gate_v9930.csv` | `467b2e866365165e520d1de20ccdaa0bfd223ce503d7c330397eaad3b6d929bd` |
| `p7_generated_sandbox_gate_v9930.csv` | `fe250dbf5dd3bf02328ed93d4d053641ab2d91eff109121747a0c46afbb974de` |
| `p8_paired_replay_boundary_v9930.csv` | `175b0523b00513c786cfd850e4d505c55a4873d5d1d333a6bb6bfc86dc70718b` |
| `p8_selected_runtime_boundary_v9930.csv` | `28d308d79e5e2d59fcd844ee7de55576960fce97bfa6dd879d477b69e9fd8873` |
| `plan` | `7117032323915ce1e742e2445a708b5e60f1d2b799c21b92219c1d2fb81cd293` |
| `route_decision_v9930.json` | `9ea21c5c2b09c6a239ce5a208a0469b01ef35a4c21f8a461230d84056d8af237` |
| `run_manifest_v9930.json` | `501ec98113c3905ec62b1f3a365db2d28ed3cd18f239ea9e78d47cb9eb5365f2` |
| `runner` | `ec99e7c21a9cef0baa671739c699d878acab30f690d04f28f13a71c83370df07` |

## 8. 最终分析结论

```text
1. v9.9.3 不再只看 G5 major PSI，而是新增 P1 tail fidelity gate。
2. 只有 P1 major+tail 通过后，P2 才真实打开 sequential density panel；未打开的 panel 均显式 not_run。
3. density 裁决使用 CoreLike、PathGood、SlowBurnGood 的 Wilson CI，不把 1024 或 diagnostic panel 写成 full closure。
4. P5 FPO v5 只用 commit-time fields 打分，evaluation 才使用真实 replay labels。
5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 boundary。
```

最终一句话：

> v9.9.3 真实执行后停在 `R9-MaterializerOrFidelityEngineeringBlocked`：本轮完成 tail fidelity gate 与修复尝试；由于 P1 fidelity 未通过，sequential density panels 被正确 gate-block，没有 fake/proxy/cpu 数据，也没有把未过 gate 的 diagnostic 写成 official controller pass。
