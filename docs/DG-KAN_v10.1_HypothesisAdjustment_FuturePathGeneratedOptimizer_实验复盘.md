# DG-KAN v10.1 Hypothesis Adjustment / FuturePath Generated Optimizer 实验复盘

> 本复盘记录 `DG-KAN_v10.1_结果解读_假设调整_FuturePathGeneratedOptimizer_完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest、真实 G40 natural AP0 panel rows 与真实 generated discovery branch-horizon rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 controller/runtime/paired replay 均显式 `not_run`。

## 0. 最新结论

```text
route = R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset
primary_blocker = natural_density_insufficient
secondary_blocker = FPO12_and_generated_discovery_failed
system_legal_controller_pass = 0
generated_route_status = stopped_future_path_generated_optimizer_reset
```

最终 artifact：`results/real_rerun_20260506/v1010_hypothesis_adjustment_futurepath_generated_optimizer_full_20260518T050000Z`

核心结论：

1. P0 复现 v10.0 boundary pass = `1`；source route = `CaseD-NaturalInsufficientBDDiscoveryFail`。
2. A 线 Fast/Slow/Risky/SafeLow/Bad = `8` / `26` / `521` / `34` / `19579`；discovery target pass = `0`。
3. B 线 FPO12 weak/discovery = `0` / `0`；best = `FPO12D-signal-reservoir-path-type-classifier`，precision = `0.011494252873563218`，V240 LCB = `-0.18263554403155388`。
4. C 线 completed panels = `2`，largest = `10000`，natural density sufficient/insufficient/inconclusive = `0` / `1` / `0`；C3 20000 按用户 cost cap 显式 `not_run`，且无 20000 artifact。
5. D 线 D0/D1/D2 = `1` / `1` / `0`；D2 weak = `0`，FastSlow precision = ``。
6. P7 controller = `not_run`；No-fake audit rows checked = `519271`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v1010_hypothesis_adjustment_futurepath_generated_optimizer.py` | v10.1 runner；执行 P0/A/B/C/D/P7-P9 gate，并写 manifest/recap。 |
| `experiments/natural_ap0_extension_materializer.py` | 复用真实 natural/generated payload materializer 与 branch-horizon replay。 |

```text
python -m py_compile experiments/run_v1010_hypothesis_adjustment_futurepath_generated_optimizer.py
```

```bash
python experiments/run_v1010_hypothesis_adjustment_futurepath_generated_optimizer.py --out-dir results/real_rerun_20260506/v1010_hypothesis_adjustment_futurepath_generated_optimizer_full_20260518T050000Z --fresh --device auto --data-root data --seed 1314 --repeat-seed 4041 --confirm-seed 5051 --execution-profile full-gated --chunk-actions 512
```

```bash
python experiments/run_v1010_hypothesis_adjustment_futurepath_generated_optimizer.py --out-dir results/real_rerun_20260506/v1010_hypothesis_adjustment_futurepath_generated_optimizer_full_20260518T050000Z --device auto --data-root data --seed 1314 --repeat-seed 4041 --confirm-seed 5051 --execution-profile full-gated --chunk-actions 512 --c-density-mode strict-new-seed
```

说明：第一段 strict run 完成 C1 后，按用户要求将 C3 20000 cost cap 固化为永不执行；随后用第二段命令复用已落盘 C1，并继续完成 C2/D/P7-P9/manifest/recap。最终 artifact 中 `c_natural_density_confirmation_v1010.csv` 的 C3 行为 `status=not_run`、`panel_size=20000`、`C3_user_cost_cap=1`；输出目录中没有任何 `*20000*` artifact。

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1010",
  "status": "summary",
  "route": "R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset",
  "primary_blocker": "natural_density_insufficient",
  "secondary_blocker": "FPO12_and_generated_discovery_failed",
  "route_explanation": "Natural density was confirmed insufficient, but FPO12 and generated discovery both failed, so controller/runtime stay blocked.",
  "source_route_v1000": "CaseD-NaturalInsufficientBDDiscoveryFail",
  "P0_boundary_pass": 1,
  "A_discovery_target_pass": 0,
  "B_FPO12_weak_pass": 0,
  "B_FPO12_discovery_pass": 0,
  "C_natural_density_sufficient": 0,
  "C_natural_density_insufficient": 1,
  "C_natural_density_inconclusive": 0,
  "C_largest_completed_panel_size": 10000,
  "C_density_mode": "strict-new-seed",
  "C_new_seed_rerun_performed": 1,
  "C_reuse_source_v1000": 0,
  "D_generated_discovery_pass": 0,
  "D2_weak_pass": 0,
  "D3_256_opened": 0,
  "P7_controller_pass": 0,
  "P8_runtime_pass": 0,
  "P9_paired_replay_pass": 0,
  "generated_route_status": "stopped_future_path_generated_optimizer_reset",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A/B/C/D 摘要

```text
A discovery target pass = 0
B FPO12 weak/discovery = 0 / 0
C density sufficient/insufficient/inconclusive = 0 / 1 / 0
D generated discovery pass = 0
```

## 4. No-Fake / Hash

```text
rows_checked = 519271
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `a_future_path_type_labels_v1010.csv` | `78250969e122a28bb507347f1fcaff85880336a511dba170d28fb46b21efe221` |
| `b_fpo12_path_type_predictor_v1010.csv` | `f45136edb34a6169cd053f82c508f2c47ffa78ff6996fcdb9926907621eb7ffa` |
| `c_natural_density_confirmation_v1010.csv` | `4f2d84a62cc9ae3b53be0f9ea3c07744e3b094268536ae01e02bd9f9b45aecbc` |
| `c_natural_density_group_rates_v1010.csv` | `a3ed16ba17d498ecdc5e11d2ea8ec82d88f3cff9dac7546d7dc3b5e001acf68e` |
| `contract_audit_v1010.csv` | `acb15df196fba2280346e016db33556cccaed1f67456de575db4c095f04450fe` |
| `d_generated_discovery_staircase_v1010.csv` | `d67eb4c27817e63e6241f74c38472242e11f7e89f312367d444e947f41e84e6f` |
| `failure_taxonomy_v1010.csv` | `011a9b66ca4be01015d3be55e1ea65c2343cfa34cee5b122a5f78f0121801770` |
| `fig_A_path_type_counts_v1010.svg` | `e237cf45ec252d8531ba6a03555e5c7893c25e6ca0bb03591fd83b1d3c6c2506` |
| `fig_B_fpo12_gate_v1010.svg` | `5eb6754bee5e11f93066541054bc9cfea30ed02a2cda4ec19a0b55505d4e10de` |
| `fig_C_density_repeat_curve_v1010.svg` | `66a477e38ff81c94f3e6e1958caa9b152f2aa324d580ddace4576be18e066d18` |
| `fig_D_generated_discovery_v1010.svg` | `d82e7e834abe2b07e3e636ebb7a1bf8a938f799be8c4121afb918d04a0d7bbbb` |
| `fig_v1010_route_matrix.svg` | `b4afefb326ccae9150cf48ec4920d4ec21537e56680b5e1b93f58e058fae2c03` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1010.csv` | `5de1f249b2795159b2905dce4052dd9f51e5043bd220061d6c0009a102a7f7d3` |
| `p0_v1000_boundary_reproduction_v1010.csv` | `3b8cf2352915a8fffb1e3c32d3933f2a6a6d56237a9c6e405a36b4b24113e25d` |
| `p7_controller_boundary_v1010.csv` | `22f1d2d69b50922dc7f04aa519599292729526f067b31305732f8b009462b395` |
| `p8_runtime_boundary_v1010.csv` | `521a4213148ed0a1b6fd0ac1950454b5ce770b3ab1a613f4b84a6e64d4330e25` |
| `p9_paired_replay_boundary_v1010.csv` | `77264e96507f487c18631eb4462a9307feaf5bace65064babb78e7419e6d0481` |
| `plan` | `5053e78a4cb828c1de57ca19e60c51b68eed9c979c285ee95ab2b59133ade5d1` |
| `route_decision_v1010.json` | `7b131f8656f579740f9322f94ac55f7aa5cce1faec5d509c1d64a43fdf8babee` |
| `run_manifest_v1010.json` | `68a86372b873cf4a66092cd784eb5624b7c89852f9fee2701088e6ea253e9c37` |
| `runner` | `11a52a46b23e2a8cef3ba555449b23ee2aa227470bd30a3a4b1f9967d9ab7e38` |

## 5. 最终分析结论

```text
1. v10.1 将 official gate、science discovery 和 engineering preflight 分开记录，没有把 discovery label 写成 official controller。
2. C 线新跑 5000 repeat 与 10000 confirmation；C3 20000 因用户 cost cap 无论是否 contradiction 均不执行，本轮实际落为 `not_run`。
3. B 线 FPO12 使用 commit-time features 做 path-type predictor；yellow calibrated sketch 只作为 discovery 诊断，不写成 official。
4. D 线按 family 执行 Stage8 -> Stage32 -> Stage64 discovery staircase；Stage64 未过 weak 时不打开 controller。
5. controller/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。
```

最终一句话：v10.1 真实执行后停在 `R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset`：Natural density was confirmed insufficient, but FPO12 and generated discovery both failed, so controller/runtime stay blocked.
