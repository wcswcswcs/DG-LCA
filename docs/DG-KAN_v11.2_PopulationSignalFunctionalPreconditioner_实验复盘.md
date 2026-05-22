# DG-KAN v11.2 Population-Signal Functional Preconditioner 实验复盘

> 本复盘记录 `DG-KAN_v11.2_PopulationSignalFunctionalPreconditioner_完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v11.1 真实 artifact、真实 MNIST/Fashion-MNIST/KMNIST 小样本诊断与计划内 synthetic/tabular/sequence 诊断；没有 fake data、proxy rows、占位数据或 CPU offload。旧 action-selection / generated-action / FPO/FPAU/GoodCone/LFRO 路线继续关闭。

## 0. 最新结论

```text
route = R6-MLPFunctionSpaceUpdateStillFails
primary_blocker = psfp_candidate_too_expensive_for_discovery_gate
secondary_blocker = short_training_smoke_not_validated_after_cost_repairs
system_legal_controller_pass = 0
generated_route_status = stopped_expensive_function_space_preconditioner
```

最终 artifact：`results/real_rerun_20260506/v1120_population_signal_functional_preconditioner_full_20260519T020000Z`

核心结论：

1. P0 v11.1 boundary lock pass = `1`；source route = `CaseA-MLPShortTrainingMechanizedVariantsFail`；P5 source task pass = `0` / `6`。
2. P1 dominant failure = `F1-noise-reservoir-amplification`；real/noise improvement = `0.6666666666666666` / `0.8333333333333334`；cos(func, AdamW) median = `0.0`。
3. P2 weak/strong = `0` / `0`；best = `PSFP-C-gradient-replacement`；noise improved = `0.0`；real/random = `7.899966612920136`。
4. P3 weak/strong = `0` / `0`；passing/strong modes = `1` / `0`。
5. P4 weak/strong = `0` / `0`；task pass = `1` / `8`。
6. P5 independent/existing-dominates = `0` / `0`；P6/P7/P8 = `not_run` / `not_run` / `not_run`。
7. No-fake rows checked = `924`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1120_population_signal_functional_preconditioner.py
```

```bash
python experiments/run_v1120_population_signal_functional_preconditioner.py --out-dir results/real_rerun_20260506/v1120_population_signal_functional_preconditioner_full_20260519T020000Z --fresh --device auto --data-root data --seed 2312 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1120",
  "status": "summary",
  "route": "R6-MLPFunctionSpaceUpdateStillFails",
  "primary_blocker": "psfp_candidate_too_expensive_for_discovery_gate",
  "secondary_blocker": "short_training_smoke_not_validated_after_cost_repairs",
  "route_explanation": "R6-MLPFunctionSpaceUpdateStillFails: psfp_candidate_too_expensive_for_discovery_gate; short_training_smoke_not_validated_after_cost_repairs.",
  "source_route_v1110": "CaseA-MLPShortTrainingMechanizedVariantsFail",
  "P0_boundary_pass": 1,
  "P1_dominant_failure_class": "F1-noise-reservoir-amplification",
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P2_best_method": "PSFP-C-gradient-replacement",
  "P2_best_noise_improved_rate": 0.0,
  "P3_weak_pass": 0,
  "P3_strong_pass": 0,
  "P4_weak_pass": 0,
  "P4_strong_pass": 0,
  "P4_task_pass_count": 1,
  "P4_task_count": 8,
  "P5_independent_value_pass": 0,
  "P5_existing_optimizer_geometry_dominates": 0,
  "P6_graph_free_pass": 0,
  "P7_weak_pass": 0,
  "P8_controller_pass": 0,
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "generated_route_status": "stopped_expensive_function_space_preconditioner",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. 实现思路与伪代码

### 3.1 P0 边界冻结

```text
load v11.1 route_decision and P1-P8 summaries
assert source_route == CaseA-MLPShortTrainingMechanizedVariantsFail
assert P5 task pass == 0/6 and P6/P7/P8 are not_run
keep old action/generated/FPO/FPAU/GoodCone/LFRO routes disabled
```

### 3.2 P1 短程失败归因

```text
for task in v11.1 six MLP diagnostics:
  replay AdamW, AdamW+additive FSNU, AdamW+same-norm random
  record per-step loss, noise-label loss, memory/hard-tail drift,
         cos(function update, AdamW), AdamW moment changes, kernel condition
classify F1/F2/F3/F5/F6 from real-vs-noise, cosine, moment, memory/hard-tail evidence
```

### 3.3 P2 PSFP A-H 候选

```text
adam_delta = -lr * grad_theta(L_current)
func_delta = J^T (K + lambda I)^-1 P_sig r

PSFP-A: eff = adam_delta
PSFP-B: eff = adam_delta + func_delta
PSFP-C: eff = func_delta
PSFP-D: eff = (1-beta) * adam_delta + beta * func_delta
PSFP-E: eff = adam_delta + beta * (func_delta - Proj_adam(func_delta))
PSFP-F: use low-rank/split-consistency signal shrinkage
PSFP-G: add memory/hard-tail reference kernel
PSFP-H: leave-one/split-style population-risk diagnostic adapter

evaluate only by current visible train/reference/memory/hard-tail plus heldout diagnostics
do not use future outcome label or old generated label as training signal
```

P2 method summary：

| method | noise improved | real/random | heldout nonharm | amortized cost | weak |
|---|---:|---:|---:|---:|---:|
| `PSFP-A-adamw` | `0.5` | `16.049580991992308` | `0.8333333333333334` | `0.1377823919246977` | `0` |
| `PSFP-B-additive-negative-control` | `0.16666666666666666` | `11.723639144966768` | `1.0` | `12.982009547715718` | `0` |
| `PSFP-C-gradient-replacement` | `0.0` | `7.899966612920136` | `1.0` | `12.351404223431267` | `0` |
| `PSFP-D-interp-beta0.05` | `0.5` | `15.793918768256638` | `0.8333333333333334` | `12.21420241971047` | `0` |
| `PSFP-D-interp-beta0.1` | `0.16666666666666666` | `15.42365329828159` | `0.8333333333333334` | `13.557448210723283` | `0` |
| `PSFP-D-interp-beta0.25` | `0.16666666666666666` | `14.083411069476204` | `0.8333333333333334` | `13.428170333389462` | `0` |
| `PSFP-D-interp-beta0.5` | `0.16666666666666666` | `11.723639144966768` | `1.0` | `14.35218157832818` | `0` |
| `PSFP-E-orthogonal-residual` | `0.16666666666666666` | `15.843039847811037` | `0.8333333333333334` | `11.571588135345856` | `0` |
| `PSFP-F-signal-channel-shrink` | `0.5` | `16.04565264516909` | `0.8333333333333334` | `12.877316087646939` | `0` |
| `PSFP-G-memory-constrained-shrink` | `0.5` | `16.11420554273751` | `0.8333333333333334` | `31.194498011120952` | `0` |
| `PSFP-H-population-risk-adapter` | `0.5` | `16.069474256154525` | `0.8333333333333334` | `39.79786604068896` | `0` |

### 3.4 P3 AdamW coupling repair

```text
test C0-C8:
  baseline AdamW
  additive after step
  replace gradient before AdamW moment
  interpolate gradient
  residual correction
  periodic5 / periodic10
  warmup-enable
  late-plateau-enable

count coupling modes only if val AUC, bad-step, memory/hard-tail, and cost gates pass
```

### 3.5 P4/P5 decisive smoke

```text
run eight MLP diagnostic tasks:
  real MNIST/Fashion/KMNIST
  synthetic interaction, label noise, input noise, tabular, toy sequence

compare AdamW, v11.1 additive FSNU, random function update, and PSFP-D/E/F/G/H
then compare PSFP-best against SAM-like, diagonal-SNR, SOAP-like, Muon-like diagnostics
```

本轮没有把 P2/P3/P4 discovery diagnostic 写成 official controller；P8 只有 P4 weak、P5 independent value、P6 graph-free、P7 future-path 全部通过才打开。

## 4. A/B/C/D 摘要

```text
P1 dominant failure = F1-noise-reservoir-amplification
P2 weak/strong = 0 / 0
P3 weak/strong = 0 / 0
P4 weak/strong = 0 / 0
P5 independent value = 0
```

## 5. No-Fake / Hash

```text
rows_checked = 924
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `contract_audit_v1120.csv` | `05b3f8369b10018b827a09e4ff3f060a89d332ca61f8439f21f8b6062c79ad94` |
| `fig_p1_failure_attribution_v1120.svg` | `5bc300040ccc751535c07093d25785c13ccd65d1e81d6c4ce84133fb549ea19b` |
| `fig_p2_psfp_candidates_v1120.svg` | `78680c434da226365132797627a3ddd0bcff1ad850ea97f04d67445f188eb0f1` |
| `fig_p3_adamw_coupling_v1120.svg` | `0d3062f75d2404d6ec056f7c1544630e387bd2ddb8da93cd367c324d992eb7f3` |
| `fig_p4_short_smoke_v1120.svg` | `41458c7937119a7699cb27ce54efbe0a3105e54729313ab91f9bc0d677a71572` |
| `fig_p5_optimizer_comparison_v1120.svg` | `122d5cfa4ebd4041dad2f238767408c7935538937179223151efdf1fe96bced9` |
| `fig_v1120_route_matrix.svg` | `44c94023885156bef1dcd7c23541a43d610a7af1989aed8f38269835a7c789e4` |
| `functional_space_natural_step` | `70e8bf027c0042ab4418f59d2bc2d1179695ee37614fa683c98cf126546c0e34` |
| `no_fake_audit_v1120.csv` | `56f0d89d839f499bf973b8f364b2346f6bdbb729e6db122c6f7e0ae1011a4d34` |
| `p0_v1110_boundary_lock_v1120.csv` | `b78d83262193cf118a969fe83d60588a97b530bcb12c2e7e39c3512e57b66760` |
| `p1_short_failure_attribution_v1120.csv` | `f196ed97d5940d26c74c6813dd6a2db8e363246b85659cd03993d84acb3f2922` |
| `p2_psfp_candidates_v1120.csv` | `e5c5a58a0d9855bf7c60140fe34a353d02e517296c306164d5cd75c165d1f010` |
| `p3_adamw_coupling_v1120.csv` | `e0c424d01f64e1ea93b019198da88bc37f41921624024424d15be72e5da1e28d` |
| `p4_short_training_smoke_v1120.csv` | `4b7cf882408bebd8bf1f843072606b510d099cb7da9b7948e08bea7f2a472150` |
| `p5_optimizer_style_comparison_v1120.csv` | `ef1a4526e3ed943b94ed2ddb5ec0eca7226f3df72e05dcfb1f11f25c2374a8a6` |
| `p6_kan_graph_free_readiness_v1120.csv` | `064d077658b3eceecb4320aee9646fa52e249b5b5c0e21260f7a528cdf56dfd4` |
| `p7_future_path_verification_v1120.csv` | `1cff6902902a4f1bed4e12c3e27ce76ad0c556631387e9ceb4ed46b5c5bb3899` |
| `p8_controller_runtime_boundary_v1120.csv` | `528652ef0f354674a92c11984b86ca19c5adc3200f66f348441a17ce980bfb4e` |
| `plan` | `bf68e2db91016b7f161bd46c667028756f2a057b97757c318525d28fdbda1b8a` |
| `route_decision_v1120.json` | `075eb06f8041978e972cbbc2cc855529861f37b71084e075bc7e50028280eac1` |
| `run_manifest_v1120.json` | `34ff02c2c5bd9530c5e2cf6f0b6bd00b3941a2b8d0ca61b94066f999af2a2e72` |
| `runner` | `90cca11a734a2f4406cc6bd71995d3b8558e90ceb029422f710f33e18254d6c7` |

## 6. 最终分析结论

```text
1. v11.2 按计划把 function-space update 从 additive step 重构为 AdamW 梯度的 population-signal preconditioner。
2. P1 没有只复述 v11.1 P5=0/6，而是逐步重放短程失败并记录 real/noise、AdamW moment、memory/hard-tail 与随机同范数对照。
3. P2 已执行 A-H 候选与 noise/heldout/memory/cost blocker repair：降低 beta、低 rank、split/signal shrinkage、memory/hard-tail reference 和更高 ridge。
4. P3/P4 即使在上游 gate 不理想时也做了 discovery-only decisive smoke，用真实短程曲线裁决，不把失败前的 diagnostic 冒充 controller。
5. P6/P7/P8 仍由硬 gate 控制；未满足时显式 not_run。
```

最终一句话：v11.2 真实执行后停在 `R6-MLPFunctionSpaceUpdateStillFails`：R6-MLPFunctionSpaceUpdateStillFails: psfp_candidate_too_expensive_for_discovery_gate; short_training_smoke_not_validated_after_cost_repairs.
