# DG-KAN v9.9.8 Multi-Marginal Natural Sampling / FPO Generated Hard Gate 实验复盘

> 本复盘记录 `DG-KAN_v9.9.8_结果解读_多边际自然采样_FPO与生成路线硬门_完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest、真实 materialized natural AP0 extension rows 与真实 64-action generated sandbox branch-horizon rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 controller/runtime 均显式 `not_run`。

## 0. 最新结论

```text
route = CaseC-NaturalDensityInsufficientGeneratedSandboxFail
primary_blocker = natural_density_insufficient
secondary_blocker = generated_64_sandbox_failed
system_legal_controller_pass = 0
generated_route_status = generated_64_sandbox_failed
```

最终 artifact：`results/real_rerun_20260506/v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z`

核心结论：

1. P0 复现 v9.9.7 boundary：source route = `CaseB-TailKeyV3FeasibleGeneratorFail`，selected tail key = `TK3H-major_template_step_coarse`，G30-G39 official/weak = `0` / `0`。
2. P1 TK3H coarseness pass = `1`；tail groups = `89`，slow+risky mixed tails = `22`。
3. P2 G40-G48 candidate count = `9`，official/weak pass count = `4` / `4`，best = `G40-IPF-major-tail-raking-sampler`。
4. P4 largest completed panel = `5000`，density sufficient/insufficient/inconclusive = `0` / `1` / `0`；CoreLike+SlowBurn UCB = `0.007370975315718785`。
5. P6 FPO v10 weak/strong = `0` / `0`；best = `FPO10C-signal-reservoir-drift-diffusion-v2`，precision = `0.011494252873563218`，V LCB = `-0.7536804607446157`。
6. P8 generated sandbox executed/pass = `1` / `0`；generated actions = `64`，branch rows = `1920` / `1920`。
7. P8 Fast/Slow precision = `0.0` / `0.0`；V/V240 LCB = `-1.1452801937561405` / `-1.0240962102829358`；longrisk UCB = `0.9972365292495987`。
8. P8 new-positive/longrisk-created rate = `0.0` / `0.015625`；failure type = `low_value_or_high_risk_generated_sandbox`。
9. P7 controller = `not_run`；No-fake audit rows checked = `211098`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.8 G40-G48 多边际 sampler profiles，并对新 profile 做 reference payload norm 对齐。 |
| `experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py` | v9.9.8 主 runner；执行 P0/P1/P2/P4-P8 gate。 |
| `experiments/run_v9980_generated_sandbox_supplement.py` | P8 补充 runner；只在 gate 允许后执行 64-action generated sandbox。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py experiments/run_v9980_generated_sandbox_supplement.py
```

```bash
python experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py --out-dir results/real_rerun_20260506/v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
python experiments/run_v9980_generated_sandbox_supplement.py --out-dir results/real_rerun_20260506/v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z --device auto --data-root data --seed 1314 --generated-actions 64 --force
```

## 2. Route

```json
{
  "P0_boundary_pass": 1,
  "P1_slow_risky_mixed_tail_count": 22,
  "P1_tail_key_coarseness_pass": 1,
  "P1_within_tail_path_type_entropy_mean": 0.46620015919493263,
  "P2_official_pass_count": 4,
  "P2_weak_pass": 1,
  "P2_weak_pass_count": 4,
  "P4_density_inconclusive": 0,
  "P4_density_insufficient": 1,
  "P4_density_sufficient": 0,
  "P4_largest_completed_panel_size": 5000,
  "P5_future_path_strong_pass": 0,
  "P5_future_path_weak_pass": 1,
  "P6_FPO_strong_pass": 0,
  "P6_FPO_weak_pass": 0,
  "P7_controller_pass": 0,
  "P7_generated_sandbox_allowed": 1,
  "P9_paired_replay_pass": 0,
  "P9_runtime_pass": 0,
  "best_failed_generator_id": "",
  "best_generator_id": "G40-IPF-major-tail-raking-sampler",
  "cpu_offload_used": 0,
  "fake_data_used": 0,
  "generated_route_status": "generated_64_sandbox_failed",
  "primary_blocker": "natural_density_insufficient",
  "proxy_row_used": 0,
  "route": "CaseC-NaturalDensityInsufficientGeneratedSandboxFail",
  "route_explanation": "natural density is insufficient, but the required 64-action generated sandbox did not pass, so generated/controller/runtime stay blocked.",
  "secondary_blocker": "generated_64_sandbox_failed",
  "selected_tail_key_id": "TK3H-major_template_step_coarse",
  "selected_tail_key_source": "v9970_selected_TK3H_passed_coarseness_audit",
  "source_route_v9970": "CaseB-TailKeyV3FeasibleGeneratorFail",
  "stage": "ROUTE_DECISION_V9980",
  "status": "summary",
  "system_legal_controller_pass": 0,
  "P8_generated_sandbox_executed": 1,
  "P8_generated_sandbox_pass": 0,
  "P8_generated_sandbox_action_count": 64,
  "P8_FastGood_precision": 0.0,
  "P8_SlowBurnGood_precision": 0.0,
  "P8_V_LCB": -1.1452801937561405,
  "P8_V240_LCB": -1.0240962102829358,
  "P8_longrisk_UCB": 0.9972365292495987
}
```

## 3. P2 Multi-Marginal Sampler Matrix

| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |
|---|---|---|---:|---:|
| `G40-IPF-major-tail-raking-sampler` | `0.04923558761592247`/`0.05893867474545041`/`0.0224609375` | `0.00448427827051975`/`0.023444775598959872`/`0`/`1.0` | `1` | `1` |
| `G41-mincostflow-major-tail-sampler` | `0.04923558761592247`/`0.05893867474545041`/`0.0224609375` | `0.00448427827051975`/`0.023444775598959872`/`0`/`1.0` | `1` | `1` |
| `G42-entropy-regularized-multimarginal-sampler` | `0.05606245229076565`/`0.06065389246715565`/`0.0224609375` | `0.017523285482106757`/`0.033713048989058726`/`5`/`0.9438202247191011` | `0` | `0` |
| `G43-major-tail-residual-balancing-sampler` | `0.04923558761592247`/`0.05893867474545041`/`0.0224609375` | `0.00448427827051975`/`0.023444775598959872`/`0`/`1.0` | `1` | `1` |
| `G44-tail-conditional-major-refill-sampler` | `0.111494080386796`/`0.09153790103107015`/`0.0234375` | `0.0346590316948022`/`0.05469887870142179`/`2`/`0.9775280898876404` | `0` | `0` |
| `G45-major-conditional-tail-refill-sampler` | `0.09401858901651486`/`0.08854440770521256`/`0.021484375` | `0.032269446285904446`/`0.05680900599345736`/`1`/`0.9887640449438202` | `0` | `0` |
| `G46-mixture-IPF-with-tail-replay-sampler` | `0.09078492108754188`/`0.08506082936843241`/`0.0224609375` | `0.018837219999988716`/`0.04790872481901743`/`0`/`1.0` | `0` | `0` |
| `G47-canonical-precursor-tail-local-mutation-sampler` | `0.04923558761592247`/`0.05893867474545041`/`0.0224609375` | `0.00448427827051975`/`0.023444775598959872`/`0`/`1.0` | `1` | `1` |
| `G48-real-train-stream-harvest-sampler` | `0.05657785304684996`/`0.06118032625559148`/`0.0224609375` | `0.017523285482106757`/`0.033713048989058726`/`5`/`0.9438202247191011` | `0` | `0` |

## 4. P4 Density

```text
P3 branch-horizon 1024 = summary
largest_completed_panel_size = 5000
CoreLike count/LCB/UCB = 14 / 0.0016686615436143242 / 0.0046947693051959186
PathGood count/LCB/UCB = 23 / 0.0030672294768888477 / 0.006893437543426104
CoreLike+SlowBurn count/LCB/UCB = 25 / 0.0033890775204861016 / 0.007370975315718785
density sufficient/insufficient/inconclusive = 0 / 1 / 0
```

## 5. P6 FPO v10

| fpo | Fast/Slow/Path precision | V/V240 LCB | longrisk UCB | cost q90 ms | weak | strong |
|---|---|---|---:|---:|---:|---:|
| `FPO10A-tiny-virtual-adamw-sketch` | `0.0`/`0.0`/`0.0` | `-1.0380896805789999`/`-1.172540052443914` | `1.0` | `0.0010807998478412628` | `0` | `0` |
| `FPO10B-jvp-gradient-alignment-sketch` | `0.0`/`0.0`/`0.0` | `-1.2509709519604`/`-1.1362025191295593` | `1.0` | `0.0007110647857189178` | `0` | `0` |
| `FPO10C-signal-reservoir-drift-diffusion-v2` | `0.0`/`0.011494252873563218`/`0.011494252873563218` | `-0.7536804607446157`/`-0.4890679725642063` | `0.9752047278432054` | `0.0007506459951400757` | `0` | `0` |
| `FPO10D-memory-offdiag-hard-gate-plus-value-sketch` | `0.0`/`0.0`/`0.011494252873563218` | `-1.0101783410572922`/`-1.2688985078839776` | `0.997968144575522` | `0.0007310882210731506` | `0` | `0` |
| `FPO10E-slowburn-detector` | `0.0`/`0.0`/`0.011494252873563218` | `-0.8617875631725055`/`-1.172771766328015` | `0.9936730475286012` | `0.0006109476089477539` | `0` | `0` |
| `FPO10F-negative-control-immediate-response-only` | `0.0`/`0.0`/`0.0` | `-1.2393395810747627`/`-1.0515990069897905` | `0.997968144575522` | `0.00031013041734695435` | `0` | `0` |

## 6. P8 Generated 64-Action Sandbox

```text
generated_sandbox_allowed = 1
generated_sandbox_executed/pass = 1 / 0
generated_action_count = 64
branch_horizon rows = 1920 / 1920
FastGood/SlowBurn precision = 0.0 / 0.0
PathGoodOrSlowFast precision = 0.0
V/V240/RAUV LCB = -1.1452801937561405 / -1.0240962102829358 / -0.8050691868183811
longrisk/bad/null UCB = 0.9972365292495987 / 1.0 / 0.056626022971156334
new_positive_created_rate = 0.0
longrisk_created_rate = 0.015625
action_apply_linf_max = 0.0
failure_type = low_value_or_high_risk_generated_sandbox
```

判断：P8 只执行了 64-action sandbox，没有打开 512/1024 generated 大跑。memory/offdiag UCB 在本轮 natural branch schema 中不可直接观测，因此显式记录为 `not_available_in_natural_branch_schema`，没有用 proxy 数值填充。

## 7. No-Fake / Contract / Failure

```text
rows_checked = 211098
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9980.csv` | `f82a4727625bfa08f5d1a36d4388bfe355eab9729c64f87ac3f612af2f35e2f7` |
| `failure_taxonomy_v9980.csv` | `1f422acb8324f38c165311d1441ab7be1925156f8ac7e3304e950775614ae1d6` |
| `fig_p1_tail_good_density_heatmap.svg` | `56a0d657bcd5556eef8a96b27e3f0a1d4f2d96c578912df331f76fef19731644` |
| `fig_p1_tail_good_gini.svg` | `b122dcbc6099719c532a0e03e872ff866805c3a570a749becba291e8568f7bc8` |
| `fig_p1_tail_path_type_purity.svg` | `414f4f52338d26da27d02630e02e2a4946f5096aa84f64cfa4d1caed3f554fb9` |
| `fig_p1_within_tail_variance.svg` | `113e7167c4aba88837a43a2416807309024b899673e3a1fe4e8d95f45ad20242` |
| `fig_p2_major_tail_psi_pareto.svg` | `c3d86643f2a22c5e5febc2c4c29ca9ef59c9a3ae7dd2ecc8732a549447b5b289` |
| `fig_p2_major_vs_tail_heatmap.svg` | `5b5e723010515ea4a6bcf970f3eb25d1d834ad26a01d1e06194b595435af8218` |
| `fig_p2_missing_tail_by_generator.svg` | `5d2998a1503041ce2744193647fac062a6e0f7219e0a35d2d6cbb0f86a387f16` |
| `fig_p4_density_ci_by_panel.svg` | `831ecfafae03dc0bb3193afb44edf38b812d3a41b47af2667bee099a04037eb0` |
| `fig_p4_density_curve_core_path_slow.svg` | `4206d754b9185c28570423ce712e15485ef2f5f852b7cd343ec53e26b8c85c83` |
| `fig_p5_future_path_type_distribution.svg` | `4230549faa5325972e8dff6f270ca90d9211dd691d4299bfe84956e1c994c4c9` |
| `fig_p6_fpo_false_positive_negative_matrix.svg` | `9d34da26d90af17a41f8f9f2b1c8ce24f7d4e316e984fee31c842c49d6ab66b6` |
| `fig_p7_controller_gate_matrix.svg` | `9745305eb1aef92737254095624ef0b1ce864d903dda6e34451e11d267b92e53` |
| `fig_stop_pivot_matrix_v9980.svg` | `5cb4900c8bfed333d77987684da270ed108f7a2bc87a00569d0b93b6c9f0343b` |
| `no_fake_audit_v9980.csv` | `a158ba4cd11d7ace0e1524b113350c72ee4269fed9d804ec0e1a47e131bd7276` |
| `p0_boundary_v9980.csv` | `cd8314a1fd02cadb3b5863c3181abbf024612b4384967ab0304b79b4d5c3af68` |
| `p1_tail_key_coarseness_audit_v9980.csv` | `d688b260d2a7e169ee76dd93dc8092bcdbe33eeda0fcce45b7f27495e403cc18` |
| `p1_tail_path_type_purity_v9980.csv` | `138963cf8cce94b1d35ce728859eb6064b2462c918e972fd35fa215e954d3f01` |
| `p2_G40-IPF-major-tail-raking-sampler_distribution_fidelity_v9980.csv` | `37b3bf26e903db7539bf041fdb3d13a44ccf4887f1af37a07f5d61430e710292` |
| `p2_G40-IPF-major-tail-raking-sampler_distribution_only_actions_v9980.csv` | `f3c66dc3c314a59a07ab8d342f14a8aac0ece36b5082d2c4eb9e9c2f1146351b` |
| `p2_G40-IPF-major-tail-raking-sampler_distribution_only_smoke_v9980.csv` | `7160776eaf3f4ca083246821fcddb33aad7b0dfc18aa7952808f53714d4494df` |
| `p2_G41-mincostflow-major-tail-sampler_distribution_fidelity_v9980.csv` | `dbb8198fc0f74630a7cfc8062c821dc2bdb5b00d3d0d525b996948897a1fbec1` |
| `p2_G41-mincostflow-major-tail-sampler_distribution_only_actions_v9980.csv` | `ef42a07240a31a6e1854b2bcd78a555286b30a4274e02d0121e484c154be36c6` |
| `p2_G41-mincostflow-major-tail-sampler_distribution_only_smoke_v9980.csv` | `09b2ce2504a7b3b32d30198414036e10715201681855e729ad6e314588ed1e99` |
| `p2_G42-entropy-regularized-multimarginal-sampler_distribution_fidelity_v9980.csv` | `8fd06c41a03b65c68a8c8211b11a8a28c92fa64a54b0019222237f9bc6687f66` |
| `p2_G42-entropy-regularized-multimarginal-sampler_distribution_only_actions_v9980.csv` | `3835c3a044590413e2fe6bdbf0405221a4a736183c365eaf29a7439500fddfa8` |
| `p2_G42-entropy-regularized-multimarginal-sampler_distribution_only_smoke_v9980.csv` | `0f9963b7c4ce19f79545763e5e7315c66ced456d1b1250e93d261ef5bc37e193` |
| `p2_G43-major-tail-residual-balancing-sampler_distribution_fidelity_v9980.csv` | `c8db6ba2905e69f0a4abbfe559d42c59858dd07fd400b961fb6b10f510ad2acf` |
| `p2_G43-major-tail-residual-balancing-sampler_distribution_only_actions_v9980.csv` | `0d82283069d45b98220c287a59f37a4ca1f69192f3db8e701499b2be0327a217` |
| `p2_G43-major-tail-residual-balancing-sampler_distribution_only_smoke_v9980.csv` | `9f39d5e63299b7b23b72cc2a429650fac85fa6d9f9b4f3930a952bbbc9579ef9` |
| `p2_G44-tail-conditional-major-refill-sampler_distribution_fidelity_v9980.csv` | `0ba51ee61e0957ac34fa7f204d56dd926a4e025954ef88856ee7ef746c53bcee` |
| `p2_G44-tail-conditional-major-refill-sampler_distribution_only_actions_v9980.csv` | `6ccbd750d8573f3494f17e50378102287f02c47b80dba0307962b2cb0665f73d` |
| `p2_G44-tail-conditional-major-refill-sampler_distribution_only_smoke_v9980.csv` | `80d2a85cd75a2812ad7bb5da22abf3d7ce9209482fe44acfc027150f90f8607a` |
| `p2_G45-major-conditional-tail-refill-sampler_distribution_fidelity_v9980.csv` | `647bd4962bbf38feb2e33dbdb441d8061f035ff18d9a612f02c908748b66fc98` |
| `p2_G45-major-conditional-tail-refill-sampler_distribution_only_actions_v9980.csv` | `6317051620a731ca60d7ba57c1eba18b16791a5e134d9df34e69403cb64c9bdb` |
| `p2_G45-major-conditional-tail-refill-sampler_distribution_only_smoke_v9980.csv` | `d12a9f653ebaef14aa15c287a524c42bbc555aa9f75d94f59ea2aca11d99f7f4` |
| `p2_G46-mixture-IPF-with-tail-replay-sampler_distribution_fidelity_v9980.csv` | `9d80bd7314ba75e3799bae8588df985fb9c0d3c84412d44c3c660a491ad68439` |
| `p2_G46-mixture-IPF-with-tail-replay-sampler_distribution_only_actions_v9980.csv` | `70d1cb3c46b73e24fadbe3cf31d2039f511f6030e7fe1b973f5d5f3d05f73447` |
| `p2_G46-mixture-IPF-with-tail-replay-sampler_distribution_only_smoke_v9980.csv` | `4147dd25db141768c5e3e05d476a52588309ba396d95899c07ad157b59b3e94f` |
| `p2_G47-canonical-precursor-tail-local-mutation-sampler_distribution_fidelity_v9980.csv` | `55d1589c2da3789f7a4875fcc9201fbb25644360f7b34db4275ab2289bb07da3` |
| `p2_G47-canonical-precursor-tail-local-mutation-sampler_distribution_only_actions_v9980.csv` | `d00eb14739b80200e6bcd6ae8b089ef6a4bf318262f73983c3b8c6792ccefee9` |
| `p2_G47-canonical-precursor-tail-local-mutation-sampler_distribution_only_smoke_v9980.csv` | `be7859312234f2d3c369ff049e1c88e9d7fd3efc016c6ec8a923d3b064ba2834` |
| `p2_G48-real-train-stream-harvest-sampler_distribution_fidelity_v9980.csv` | `fb464a23d0744ba419386616f6aba496a406dba6059b1ce751999e005b5bceb4` |
| `p2_G48-real-train-stream-harvest-sampler_distribution_only_actions_v9980.csv` | `2d5b1de86249842675601f62e946aa2e8e5b6167386b5e530e456e362cbd71ff` |
| `p2_G48-real-train-stream-harvest-sampler_distribution_only_smoke_v9980.csv` | `f7b3237fea43dffaca6899618e71d28c364cc5f67d0aba592fea08e9acc54bbe` |
| `p2_generator_distribution_only_matrix_v9980.csv` | `13b70ce39e1fe5da4f0aac22142dbea4ca0c4119dbacd121665336a629843227` |
| `p2_generator_failure_taxonomy_v9980.csv` | `cfe9da7ac35c3db65acf1f132243206324276b6a37034f85c38851dd63e4e140` |
| `p3_branch_horizon_1024_pilot_v9980.csv` | `8000cd6ff7b9ad826f7e8b67ebeafca58d54184d33cd9d612c3f028dfce279ea` |
| `p4_G40-IPF-major-tail-raking-sampler_1024_action_apply_replay_v9980.csv` | `b11cfdf9bf62c3b2e07c90aaa2985cae73ac905bd5ee576b79fc07afc0108e9b` |
| `p4_G40-IPF-major-tail-raking-sampler_1024_action_labels_v9980.csv` | `8fa66ae5c975b214da2b627cca8de0b8f59aee47310ef97fee67ed11a1da344b` |
| `p4_G40-IPF-major-tail-raking-sampler_1024_action_rows_v9980.csv` | `86346553bb081daf8aa46f3b1f38c363616f185e34d2c051346086304d621bea` |
| `p4_G40-IPF-major-tail-raking-sampler_1024_branch_horizon_v9980.csv` | `b505a66a90ab59d4ec650dd22b2137bc3ea37d30c8249394f955e54693a0c5a0` |
| `p4_G40-IPF-major-tail-raking-sampler_1024_panel_smoke_v9980.csv` | `2902b5443809f35017be8f7c6e5a5aa399e4b0c41b76403e9d253a4c711a5bec` |
| `p4_G40-IPF-major-tail-raking-sampler_5000_action_apply_replay_v9980.csv` | `928b3f5c97e313a9e28f4fc29f8a8585518f8f3929b1b6e739618500e8a60819` |
| `p4_G40-IPF-major-tail-raking-sampler_5000_action_labels_v9980.csv` | `a2d202cce6291f854c6b6425528e5c830b18856f586e8fc84749e426aef84c50` |
| `p4_G40-IPF-major-tail-raking-sampler_5000_action_rows_v9980.csv` | `6ef6e2f7da3b0f3d994071178c98a4ed4d6116da318aa053adb648ecf6dbbed6` |
| `p4_G40-IPF-major-tail-raking-sampler_5000_branch_horizon_v9980.csv` | `5b7aef294b3480be669811fd8c1e833c6ef826f0c45d56bf94bab643978a4a93` |
| `p4_G40-IPF-major-tail-raking-sampler_5000_panel_smoke_v9980.csv` | `177fc628917f14d45a9025f188a5928f9642c0b51138c3fcc8966cc05e20118e` |
| `p4_natural_density_group_rates_v9980.csv` | `ce95a18f0e038a904ecc2c58639a81f5954ed37e25b27e540a0254a0171ed479` |
| `p4_sequential_natural_density_panel_v9980.csv` | `5e71a234935158b6bfb5f40b7bdb4b043ecf8d4acac0fd4c5091dc27219daad0` |
| `p5_future_path_type_revalidation_v9980.csv` | `40d7e93c813745f82ad1c1715f6da5d9f0d598ab0b34aea0732b612219cb2059` |
| `p6_fpo_v10_path_type_predictor_v9980.csv` | `de612a2939ef42aa4023c9a9a13e98d1b8345442f5363c496af3e143c42b0985` |
| `p7_existing_action_controller_gate_v9980.csv` | `5de67543fafadcc287b0a634e2f1691ec563b0ba21cf7b30b073398042a4bdc5` |
| `p8_generated_64_sandbox_action_apply_replay_v9980.csv` | `6d7b317279d4d6544d57e36840572a78e955978dc0c43bd4ef1ed065460d048f` |
| `p8_generated_64_sandbox_actions_v9980.csv` | `fa3e7fca938f104bba9bf92d1d947584d399fa218db3851a1ea16c6c0e052c50` |
| `p8_generated_64_sandbox_branch_horizon_v9980.csv` | `9d16308f4140ea8647b654d5c4ad33352d190908e6d54b69ac00cd6d4e7956cf` |
| `p8_generated_64_sandbox_labels_v9980.csv` | `eaa78357fc947ffb8907c1cda19183ab8a74379ca284ea9f74e05a04561b07a6` |
| `p8_generated_sandbox_gate_v9980.csv` | `c575c6b2aed2050d61daf94167c6cf58eecdcfa911325253c748e064e30e04cc` |
| `p9_paired_replay_boundary_v9980.csv` | `2825045c74a23f912996ae9644a6973d863d5bdd7f35792c5d66eff695aaf5ba` |
| `p9_selected_runtime_boundary_v9980.csv` | `497a78a1e71b40f91f54ba7a74ef715e18710fe83e909c880cdbc9230feb1972` |
| `route_decision_v9980.json` | `dab4f1f6452148c3361ca85cb6d9c0bbcbad31c56500aa5809eb9ac5f6e16aac` |
| `run_manifest_v9980.json` | `27b2ea4f6830aa83c1bc0651e3c4a3595451ee3194e80e127ef78431c43a9c3e` |
| `plan` | `72fcd75b01bf7063d0bded618f5224d7cfa57a421dc21cc62f21e7abec54d32e` |
| `runner` | `39512a9225e10cc7588b77bab9c0766e716adc2b7e7b76e813f929b8d0175b57` |
| `generated_sandbox_supplement` | `ba1099dce6ec00dbe85ac243650a0ceea969fb5c297b0ac43a2789f0496783d8` |
| `materializer` | `7567bf4f41c21f0d8641d827650c94a9dada1e5a28180218f648a9dea6d99816` |

## 9. 最终分析结论

```text
1. v9.9.8 没有把 TK3H 可采样误写成 density closure；P2 先做 G40-G48 多边际自然采样修复。
2. G40 通过 major+tail fidelity 后打开真实 1024/5000 branch-horizon panel；5000 panel 给出 natural density insufficient。
3. natural density insufficient 只允许打开 64-action generated sandbox；本轮已真实执行该 sandbox，没有扩大到 512/1024。
4. generated sandbox 未通过 value/risk gate，因此 controller/generated scale-up/runtime/paired replay 仍保持 blocked。
5. No-fake audit 要求 fake/proxy/cpu 全为 0；memory/offdiag 不可观测项没有编造成 proxy 数值。
```

最终一句话：v9.9.8 真实执行后停在 `CaseC-NaturalDensityInsufficientGeneratedSandboxFail`：natural density is insufficient, but the required 64-action generated sandbox did not pass, so generated/controller/runtime stay blocked.
