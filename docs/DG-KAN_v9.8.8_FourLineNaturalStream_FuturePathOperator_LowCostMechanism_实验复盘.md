# DG-KAN v9.8.8 Four-Line Natural Stream / FuturePathOperator / Low-Cost Mechanism 实验复盘

> 本复盘记录 `DG-KAN_v9.8.8_四线并行_自然扩流_FuturePathOperator_低成本机制_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.7/v9.8.4/v9.8.2 已真实 materialized rows；没有 fake data、proxy rows，也没有把 generator missing、density diagnostic、future-path diagnostic 或 low-cost proxy diagnostic 写成 official controller pass。

## 0. 最新结论

```text
route = R7-EngineeringBlocked
primary_blocker = natural_extension_generator_missing
secondary_blocker = low_cost_proxy_failed
route_recommendation = engineering_only_materializer_task
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_density_evidence
```

最终 artifact：`results/real_rerun_20260506/v9880_four_line_natural_futurepathoperator_lowcost_full_20260516T230000Z`

核心结论：

1. P0 复现 v9.8.7 boundary：source route = `CaseE-MaterializerExtensionGeneratorMissing`，P1c/P2/P4strong/system = `1` / `1` / `0` / `0`。
2. P1 generator entrypoint search：natural_extension_action_generator_found = `0`，entrypoint_count = `0`。
3. P2 1/16/256 new-natural-action preflight = `not_run`，reason = `P1_natural_extension_action_generator_missing`。
4. P3 density 仍 inconclusive：PanelA CoreLike LCB/UCB = `0.021475240123524635` / `0.033333885489495195`，5000/10000/20000 panel not_run。
5. P4 future-path mechanism weak/strong = `1` / `0`；Core77 RAUV LCB = `2.8166481132099395`。
6. P5 low-cost proxy v2 weak/strong = `0` / `0`；best = `LC8_RAUV_surrogate_sketch`，precision/V/cost q90 = `0.3218390804597701` / `-0.2633631421716069` / `0.0010509975254535675`。
7. P6 controller = `not_run`；P7 generated sandbox allowed = `0`。
8. No-fake audit：rows checked = `38`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py` | v9.8.8 runner；复现 v9.8.7 boundary，定位 natural AP0 extension generator，按 gate 写 P2/P3/P6-P10，使用 landed rows 做 P4 future-path 机制和 P5 LC2-LC8 低成本 proxy v2。 |

```text
python -m py_compile experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py
```

```bash
python experiments/run_v9880_four_line_natural_futurepathoperator_lowcost.py --out-dir results/real_rerun_20260506/v9880_four_line_natural_futurepathoperator_lowcost_full_20260516T230000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9880",
  "status": "summary",
  "route": "R7-EngineeringBlocked",
  "primary_blocker": "natural_extension_generator_missing",
  "secondary_blocker": "low_cost_proxy_failed",
  "route_recommendation": "engineering_only_materializer_task",
  "source_route_v9870": "CaseE-MaterializerExtensionGeneratorMissing",
  "P0_boundary_pass": 1,
  "P1_generator_found": 0,
  "P2_preflight_pass": 0,
  "P3_density_sufficient": 0,
  "P3_density_insufficient": 0,
  "P3_density_inconclusive": 1,
  "P4_weak_pass": 1,
  "P4_strong_pass": 0,
  "P5_weak_pass": 0,
  "P5_strong_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_no_legal_mechanism_or_density_evidence",
  "P8_generated_weak_pass": 0,
  "P9_runtime_pass": 0,
  "P10_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P4 Future Path Mechanism

| group | actions | RAUV LCB | V1 LCB | V240 LCB | LongRisk UCB | LDO/LSO/LTO/LFO |
|---|---:|---:|---:|---:|---:|---|
| `Core77` | `77` | `2.8166481132099395` | `0.08815910625793319` | `4.750885696264312` | `0.03827572901862327` | `0.4415584415584416`/`0.038961038961038974`/`0.012987012987012991`/`0.18181818181818177` |
| `CoreExpansion10` | `10` | `0.6225266023303726` | `-0.04039991800257209` | `1.623630340725588` | `0.0` | `0.0`/`0.0`/`0.0`/`0.0` |
| `Core77+CoreExpansion10` | `87` | `2.710737520242875` | `0.11702769816449304` | `4.588476154696239` | `0.03389313880385762` | `0.39080459770114945`/`0.034482758620689724`/`0.011494252873563204`/`0.16091954022988508` |
| `Core77+RiskCleanButLowValueTop10` | `87` | `3.2442709696296745` | `0.16186559086365848` | `5.462997655266469` | `0.03389313880385762` | `0.39080459770114945`/`0.034482758620689724`/`0.011494252873563204`/`0.16091954022988508` |
| `OldOnly` | `10` | `1.0287178879851673` | `-0.174770564086397` | `2.0678046281088522` | `0.0` | `0.0`/`0.0`/`0.0`/`0.0` |
| `ExactOnly` | `68` | `1.5250480927868013` | `0.13699866747303657` | `2.922090918869295` | `0.0` | `0.0`/`0.0`/`0.0`/`0.0` |
| `RandomMatched` | `87` | `0.17625930908187082` | `0.09405836253137967` | `3.5922551893613908` | `0.8485412092119363` | `0.0`/`0.0`/`0.0`/`0.0` |

## 4. P5 Low-Cost Proxy v2

| proxy | legal | cost q90 | precision | V LCB | RAUV LCB | longrisk UCB | weak | strong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `LC2_memory_hardtail_directional_response` | `green` | `0.0008610077202320099` | `0.26436781609195403` | `-0.30666439916806476` | `2.2955369629874767` | `0.0` | `0` | `0` |
| `LC3_function_displacement_stability` | `green` | `0.0007213093340396881` | `0.12643678160919541` | `-0.4626285104268694` | `0.31911219265276325` | `0.4447068304756729` | `0` | `0` |
| `LC4_AdamW_compatibility_sketch` | `green` | `0.000801868736743927` | `0.16091954022988506` | `-0.8981221643414681` | `3.641682964698954` | `0.5761577373425045` | `0` | `0` |
| `LC5_cross_sample_sign_consistency` | `green` | `0.01865765079855919` | `0.10344827586206896` | `-0.6034471686218974` | `1.129897966691829` | `0.5053405200342863` | `0` | `0` |
| `LC6_mini_virtual_path_1step` | `green` | `0.0008209608495235443` | `0.16091954022988506` | `-1.0316022529274267` | `4.082519662120671` | `0.6108072140067218` | `0` | `0` |
| `LC7_mini_virtual_path_3step` | `green` | `0.0008209608495235443` | `0.13793103448275862` | `-0.9270119202024064` | `3.7452154518083507` | `0.5644965010207544` | `0` | `0` |
| `LC8_RAUV_surrogate_sketch` | `yellow:name_is_surrogate_no_future_label_used_in_score` | `0.0010509975254535675` | `0.3218390804597701` | `-0.2633631421716069` | `3.160170717869798` | `0.0` | `0` | `0` |

## 5. Boundary

```text
P2 natural extension preflight = not_run
P3 5000/10000/20000 density panels = not_run
P6 controller = not_run
P7 generated_sandbox_allowed = 0
P8/P9/P10 = not_run
```

## 6. No-Fake / Contract / Failure

```text
rows_checked = 38
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `base_acc_sentinel_v9880.csv` | `1bee6609cfb7f6f2c36dc4199c2f09747dd9a692281ff784b6ede02f540d85aa` |
| `contract_audit_v9880.csv` | `41ad065e10670bf0095d76b9b4b1c8195115b1ade654658d9badf716b06bf319` |
| `failure_taxonomy_v9880.csv` | `da61cb9056cd30328b8d2b102c12a2d5c75422356fa945ae0124ffc68abd4cad` |
| `field_legality_ledger_v9880.csv` | `11e6e4962922f6f6a1809d01e69fbaf5ff2ce9100e3641b2a6c8d4f34b0bc872` |
| `fig_AUV_longrisk_scatter_v9880.svg` | `15718033162415cf8c8683f0b43e9662414db47a1315356a6b62147a39eb3c50` |
| `fig_A_future_path_curves_v9880.svg` | `5e7575a92cc2246c803e249987c9b5cd32b70467a33d723a0b463f6fe2650da9` |
| `fig_B_cost_quality_pareto_v9880.svg` | `6418c3ea522fc6f976af8cc63d5a727acce84fca23da2bb987d2200ac8d27135` |
| `fig_C_preflight_ladder_v9880.svg` | `cfaf0d1f73347babf12dec2894793ecee1e3893d4898d01dcc081f9e174552e6` |
| `fig_density_curve_core_path_slow_v9880.svg` | `dddeecfa6ec58134672cb29f13c0e628c6d270329a873c58cf8868d2b1d6d2ef` |
| `fig_generated_gate_v9880.svg` | `e735cb7b0e1a996071dde0bbe8b6775c0cec5aa536d8d4048f9021e9ba8fb5ab` |
| `fig_no_fake_audit_v9880.svg` | `f8f8bf284f6228391526182bef4eac7dfc9d84705989f5070ef10c7f68ee3a4d` |
| `fig_proxy_cost_vs_budget_v9880.svg` | `671261a88a2c8b4ffdd5a320ac1cabd6096e28739c157800835f5e65e5c7173e` |
| `fig_proxy_precision_vs_V_v9880.svg` | `c57f21791420189ac798be7ce9186e7344b7d761a294ceae63660a8707fbac53` |
| `fig_route_waterfall_v9880.svg` | `536fa8a37db240e568f320aeb60076592752376f8f9cc93546533b1e7fe11520` |
| `no_fake_audit_v9880.csv` | `95974e6b97caaf265d6a00cb4c4dc61a5f8b284bdb7b6d6f3bb7e5edc85e3993` |
| `p0_boundary_reproduction_v9880.csv` | `db75c04e9de0fbeca5353951323145c7b48d16e9ba1792bddeeac4b14af46a9e` |
| `p10_paired_replay_short_full_boundary_v9880.csv` | `f28b61f10116ff2c4adb05211a6af8a4799c8af8b0207743b2f719e5552ec174` |
| `p1_natural_extension_generator_entrypoint_v9880.csv` | `ae7b381bd1b2f082e46cfba097af6376ee0a373d577f8ced0a348c7ff6866b0b` |
| `p2_natural_extension_preflight_v9880.csv` | `c733942b6174cd01d1757a1b56cca715ad7f0bf3db6db336231a00825985f161` |
| `p3_natural_density_panel_v9880.csv` | `a04cdc3bf94c4090b38cb4b6943e49f12b9982e0d143d2f0909686b45a4dbdc4` |
| `p4_future_path_type_mechanism_v9880.csv` | `0378c801051f188cc1efbe82b0f83a85c6af1e65603f75bc32738eba6e1eede6` |
| `p5_low_cost_future_operator_proxy_v2_v9880.csv` | `994ec481cbd70ac3f3cd744db9301afee64bf46763db8dc209c9f9277a471d12` |
| `p6_existing_action_controller_gate_v9880.csv` | `5ddd719952280d35cbacbc1ed7d09a558cb24bb8e24d9a5ab225835c701ce7f4` |
| `p7_generated_reopen_gate_v9880.csv` | `d5c4412890580bd8bb1f42f216dc4dc6810e1004156864d82d5a13b58a2e5a84` |
| `p8_generated_sandbox_boundary_v9880.csv` | `e99c260e3d1780c0558fd403df2ab1bb83985033fefabed1cef1e6d3421fd4ef` |
| `p9_runtime_boundary_v9880.csv` | `6bb8925bf4a3df632cd6cf2b30612584ccb3804d784b565f4df335a9ffe4e0bd` |
| `plan` | `3b09f136da86d1a409ffdc4fb0382c4f30b2bf505e8a264f14a6775f093eccda` |
| `route_decision_v9880.json` | `2b796f2f49abefe396fdfb139a5c71df30567842b4270830e2472c0edf4c4cdc` |
| `run_manifest_v9880.json` | `197198e5863109599c0d6652fa251f3513068245717a246ea67b5b0e1d518aae` |
| `runner` | `d38259e134dbc8fca4da67452c3620a927a80564fff5e0f29f3884ee7d846591` |

## 8. 最终分析结论

```text
1. v9.8.8 首先查找新自然 AP0 extension action generator，结果仍未落地。
2. 因 generator missing，P2 新动作 preflight 与 P3 5000/10000/20000 density panel 全部 not_run；不能作 density success/fail claim。
3. A/P4 future path weak signal 继续存在，但 strong 仍被 87-action leaveout/risk gate 阻断。
4. B/P5 LC2-LC8 v2 仍没有 legal low-cost controller-ready proxy。
5. P6-P10 全部 gate-blocked，strict PureKAN functional 仍未成功。
```

最终一句话：

> v9.8.8 真实执行后停在 `R7-EngineeringBlocked`：自然扩流的新 action generator 仍未落地，低成本 FuturePathOperator 代理也没有过 gate，因此不能进入 official controller 或 generated sandbox。
