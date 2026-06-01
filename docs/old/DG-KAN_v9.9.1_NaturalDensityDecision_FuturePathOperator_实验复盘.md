# DG-KAN v9.9.1 Natural Density Decision / FuturePathOperator 实验复盘

> 本复盘记录 `DG-KAN_v9.9.1_NaturalDensityDecision_FuturePathOperator_四线并行完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 和真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。1024 pilot 不被写成 full density closure，5000/10000/20000 未打开时显式 `not_run`。

## 0. 最新结论

```text
route = R-P1-NaturalGeneratorDistributionMismatch
primary_blocker = natural_generator_distribution_fidelity_failed
secondary_blocker = full_density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_P1_distribution_fidelity_failed
```

最终 artifact：`results/real_rerun_20260506/v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z`

核心结论：

1. P0 复现 v9.9.0 boundary：P1a/P1b/P1c/P1d = `1` / `1` / `1` / `1`，system = `0`。
2. P2 真实 1024-action pilot：branch rows = `30720` / `30720`，completion = `1`，peak GPU MB = `5679.9873046875`。
3. P1 分布保真 weak/strong = `0` / `0`；PSI max = `13.018683555518681`，max group share = `1.0`，missing major groups = `17`。
4. P2 pilot CoreLike count/LCB/UCB = `2` / `0.0005357696110014586` / `0.007093421501714742`；PathGood count/LCB/UCB = `2` / `0.0005357696110014586` / `0.007093421501714742`。
5. P2 5000/10000/20000 panels = `not_run`，reason = `P1_distribution_fidelity_failed_do_not_open_full_density_panels`；density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
6. P3 future path weak/strong = `0` / `0`。
7. P4 FuturePathOperator sketch weak/strong = `0` / `0`；best = `FOS4_RiskAdjustedHardGate`，precision = `0.0`。
8. P5 controller = `not_run`；P6 generated sandbox allowed = `0`。
9. No-fake audit：rows checked = `33834`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9910_natural_density_decision_futurepathoperator.py` | v9.9.1 runner；执行 P0 boundary、1024 natural pilot、P1 distribution fidelity、P2 density decision、P3 path type、P4 low-cost sketch 与 P5-P8 boundary。 |
| `experiments/natural_ap0_extension_materializer.py` | v9.9.0 已落地的 natural AP0 extension materializer，本轮复用它生成真实 pilot action/replay rows。 |

```text
python -m py_compile experiments/run_v9910_natural_density_decision_futurepathoperator.py
```

```bash
python experiments/run_v9910_natural_density_decision_futurepathoperator.py --out-dir results/real_rerun_20260506/v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9910",
  "status": "summary",
  "route": "R-P1-NaturalGeneratorDistributionMismatch",
  "primary_blocker": "natural_generator_distribution_fidelity_failed",
  "secondary_blocker": "full_density_panels_blocked",
  "source_route_v9900": "R-C1B-NaturalGeneratorLandedSmokePassDensityPending",
  "P0_boundary_pass": 1,
  "P1_weak_pass": 0,
  "P1_strong_pass": 0,
  "P1_PSI_major_max": 13.018683555518681,
  "P1_max_group_share": 1.0,
  "P2_pilot_action_count": 1024,
  "P2_density_sufficient": 0,
  "P2_density_insufficient": 0,
  "P2_density_inconclusive": 1,
  "CoreLike_count": 2,
  "CoreLike_LCB": 0.0005357696110014586,
  "CoreLike_UCB": 0.007093421501714742,
  "PathGood_count": 2,
  "PathGood_LCB": 0.0005357696110014586,
  "PathGood_UCB": 0.007093421501714742,
  "P3_future_path_weak_pass": 0,
  "P3_future_path_strong_pass": 0,
  "P4_future_operator_sketch_weak_pass": 0,
  "P4_future_operator_sketch_strong_pass": 0,
  "P5_controller_pass": 0,
  "P6_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_P1_distribution_fidelity_failed",
  "P7_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Distribution Fidelity

```text
old_action_count = 2876
new_action_count = 1024
PSI_major_max = 13.018683555518681
max_group_share = 1.0
missing_major_group_count = 17
old action/payload collision = 0 / 0
P1 weak/strong = 0 / 0
```

判断：P1 使用 old canonical AP0 panel 与本轮 1024 pilot 的 dataset/template/step_bucket/payload 分布做 PSI 与 coverage 审计；如果 P1 weak 不过，full 5000/10000/20000 panel 不打开。

## 4. P2 Density Decision

```text
pilot actions = 1024
CoreLike count/LCB/UCB = 2 / 0.0005357696110014586 / 0.007093421501714742
PathGood count/LCB/UCB = 2 / 0.0005357696110014586 / 0.007093421501714742
RiskyHighAUV count = 40
BadPath count = 1022
density sufficient/insufficient/inconclusive = 0 / 0 / 1
```

判断：1024 pilot 是真实 replay panel，但不是 full density closure；在 P1 weak 失败时不允许继续宣称 full density sufficient/insufficient。

## 5. Boundary

```text
P3 future path = 0 / 0
P4 sketch = 0 / 0
P5 controller = not_run, reason = P2_density_not_sufficient_and_P4_sketch_not_passed
P6 generated = not_run, allowed = 0
P7 runtime = not_run
P8 paired replay = not_run
```

## 6. No-Fake / Contract / Failure

```text
rows_checked = 33834
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9910.csv` | `fa524b7c75e5ffdc4e3a3c2b70b564ed3b9756d216b5aba861a9c799a99d7027` |
| `failure_taxonomy_v9910.csv` | `2887f96fa50321b6d6d64dda48d23e587608b13086d406d2d7968004b5036577` |
| `fig_p1_distribution_psi.svg` | `ce254753bb5cd182d423df445d161bf1f27b66887e651dd3a1998ced9fa54f38` |
| `fig_p2_density_core_path.svg` | `6c7d35b81a044b6e61c5ff6ba99289000cf2728bc676efe7fe5c66ea56e27759` |
| `fig_p2_path_type_counts.svg` | `5994287117cb1ad0f4c93d69f745dd4b4c7bec421360a7db9ec8b68cac339f04` |
| `fig_p4_sketch_best.svg` | `c49b949179c566366e8e9c866618648d33eeafdec111ed575d751cc2a8972f17` |
| `fig_stop_pivot_matrix_v9910.svg` | `8e16a82bd46390f6c3cd1d914dbef928b25d61a737e9107cc598a60465bb09df` |
| `fig_v9910_gate_matrix.svg` | `01cf33a7eb8213706f9c788b99c1fa3e0ae326736fe4b9b29e97abfcd2cbc4bc` |
| `no_fake_audit_v9910.csv` | `c72a15782c940999ab178b459a984f71fd934128b29c0a2c4e5f13378952a03c` |
| `p0_boundary_reproduction_v9910.csv` | `e668744b26fc485c4a7f5ef806ccbc89d54260bd3f2c202d5975f257a43dfae0` |
| `p1_natural_generator_distribution_audit_v9910.csv` | `9c83ce3ebdc4d69e1af023d7a3894036049168f91bfc22fd5c4402b539f04413` |
| `p2_1024_action_apply_replay_v9910.csv` | `671bd1024596465533893f8b811c11a8935205d04aafefa86e60094fdd59e2b8` |
| `p2_1024_action_pilot_smoke_v9910.csv` | `c48cf24233a941ad4850456adaa25bbcc31027bfdc57c8b0d8423ba3fbdd4822` |
| `p2_1024_branch_horizon_v9910.csv` | `cd36fe37202dec213abb5d4167f671d9086c9576a63c0610bdcb7e15347fe606` |
| `p2_1024_natural_action_rows_v9910.csv` | `0e735813fc2f748aa31b9abe394c87c6d604a1449b77f03bc3c9bcd014c3ff2b` |
| `p2_density_group_rates_v9910.csv` | `ca19135b61e9c16ecba71a035188814843702431e94c5294199e1696e6a45be2` |
| `p2_natural_action_labels_v9910.csv` | `514f164ac136338c66cb14d8333d610e8d01994ec4a1972ad421b40511f8917e` |
| `p2_sequential_natural_density_decision_v9910.csv` | `06d08245fdb6639715928d566c2b6ab485d123713744c5c280f5facecc19c162` |
| `p3_future_path_type_decomposition_v9910.csv` | `62cfb44f057346dab7549c24f7a75e5f7624566abfc2a39baa541a1487bcf608` |
| `p4_future_path_operator_sketch_v4_v9910.csv` | `56eeaf4b030b7aa5179f00a9081d6ce489e13fe0e3fdf71ceedbafc7b69d25e4` |
| `p5_existing_action_controller_boundary_v9910.csv` | `4ecdcf3002b7daa19aef823f3d2824b4c628daf06936a346ceee6093e1114168` |
| `p6_generated_route_reopen_gate_v9910.csv` | `c8ebde3aa6c997882d97e9e9c49ff2f4ed5f7a2118d24c75d8220fca7ff44aac` |
| `p7_runtime_boundary_v9910.csv` | `8b46c16ee560e21760c6941196e244e55af297422f2b78d5a656891ade7c80a4` |
| `p8_paired_replay_boundary_v9910.csv` | `edda668c72601341f56df1cd40c2f547b17aa27f657fc49057aaad28011fac88` |
| `plan` | `c1fd3485beced9046cc0968dac8c223a5660220956eda46ab395e4c5cc3e86ce` |
| `route_decision_v9910.json` | `e310c19b588ebdaaebaa25db2c9959b87ae395b936191c8cd68229c385a38ee3` |
| `run_manifest_v9910.json` | `559ae1c5bf570c3465a52f5d1261c8272da4af499fa367ac0d33e23054509f84` |
| `runner` | `fe2e3cf17054ae88c71a0d39f2ed0915bfa60a86cfa5974a159479d60f2f5349` |

## 8. 最终分析结论

```text
1. v9.9.1 真实运行了 1024-action natural AP0 pilot，不是复用 v9.9.0 的 256 smoke。
2. P1 分布保真度是本轮关键 gate；它决定是否允许打开 5000/10000/20000 density panels。
3. P2 pilot density 只作为序贯裁决的第一步，不被写成 full density closure。
4. P4 low-cost FuturePathOperator sketch 只使用 commit-time fields 做排序；evaluation 使用真实 replay labels，但不把 diagnostic label 放入 controller。
5. controller/generated/runtime/paired replay 都保持 gate-blocked，直到 C 线或 B 线真的过 gate。
```

最终一句话：

> v9.9.1 真实执行后停在 `R-P1-NaturalGeneratorDistributionMismatch`：本轮把 v9.9.0 的工程 smoke 推进到 1024-action natural pilot 和分布/密度/低成本 sketch 裁决；但没有满足 official controller/runtime/generated gate，因此 strict PureKAN functional 仍未成功。
