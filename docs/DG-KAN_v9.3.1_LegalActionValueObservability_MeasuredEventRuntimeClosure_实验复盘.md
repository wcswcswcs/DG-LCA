# DG-KAN v9.3.1 Legal Action-Value Observability 与 Measured Event Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.1_LegalActionValueObservability_MeasuredEventRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把缺失的 action apply、secondary/control outcome 或 diagnostic runtime 写成 official pass。

## 0. 最新结论

```text
route = R1-BoundaryReanalyzed
base_candidate = LQ-t2-h256
success_v9310_strict_purekan_functional = False
success_v9310_full_functional = False
success_v9310_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9310_legal_action_value_observability_measured_event_runtime_closure_first_20260513T211849Z/
```

核心结论：

1. P0 复现 v9.3.0 boundary：candidate/action/event = `2876` / `2876` / `24192`，source route = `R9-OGPFeatureFail`。
2. P1 action lifecycle 未过：`action_apply_error_measured = 0`，`action_apply_error_missing_count = 2876`，没有可重放 tensor artifact。
3. P2 secondary/control outcome 未过：`secondary_outcome_ready = 0`，`missing_secondary_delta_count = 28760`，matched control per event = `0`。
4. Primary oracle frontier 仍存在：oracle precision = `1.0`，coverage = `0.05445326278659612`，bad/null = `0.0` / `0.0`。
5. Legal observability 只能作为 diagnostic：best observable = `OGP-C1-CandidateScore`，bad AUC = `0.6170115546218488`，safe AUC = `0.7569791472566192`，feature cost pass = `0`。
6. P5 action-conditioned observable factory 没有 survivor：现有 frozen action rows 缺 payload norm、true delta norm、action/AdamW cosine、linearized deltas 等字段。
7. P6/P8/P10 不能打开：`ogp_decision_pass = 0`，`system_legal_controller_pass = 0`。
8. P9 measured event-driven runtime 未 materialize：reference step ratio q90 = `2.213009156635521`，empty-step controller kernels = `59868`，runtime pass = `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py` | v9.3.1 runner；读取 v9.3.0/v9.2.80/v9.2.82 artifacts，审计 action apply、secondary/control outcome、legal observable gap 和 event-driven runtime materialization |

代码检查：

```text
python -m py_compile experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py
```

正式运行：

```bash
python experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py \
  --out-dir results/real_rerun_20260506/v9310_legal_action_value_observability_measured_event_runtime_closure_first_20260513T211849Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "action_apply_error_linf": "",
  "action_apply_error_measured": 0,
  "action_apply_error_missing_count": 2876,
  "action_apply_error_relative": "",
  "action_count": 2876,
  "action_lifecycle_pass": 0,
  "action_primitive_id": "AP0-current-action-reference",
  "bad_event_heldout": 0.0,
  "bad_event_ucb": 1.0,
  "base_candidate": "LQ-t2-h256",
  "best_observable_BadCountAt273": 49,
  "best_observable_NullCountAt273": 16,
  "best_observable_SafeGoodCountAt273": 210,
  "best_observable_auc_bad": 0.6170115546218488,
  "best_observable_auc_safe": 0.7569791472566192,
  "best_observable_ece_bad": "",
  "best_observable_id": "OGP-C1-CandidateScore",
  "candidate_count": 2876,
  "candidate_lifecycle_pass": 1,
  "continual_pass": 0,
  "control_oracle_pass": 0,
  "controller_id": "not_selected_materializer_blocked",
  "controller_launches_per_active_step_q90": "9.0",
  "controller_syncs_per_active_step_q90": "1.0",
  "coverage_heldout": 0.0,
  "cpu_offload_used": 0,
  "diagnostic_downstream_used_for_controller": 0,
  "empty_step_controller_kernel_count": "59868",
  "empty_step_controller_sync_count": "6652",
  "event_count": 24192,
  "event_driven_runtime_pass": 0,
  "external_ready": 0,
  "fake_data_used": 0,
  "feature_cost_pass": 0,
  "full_run_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "matched_control_count_per_event": 0,
  "memory_ratio": "1.0",
  "missing_secondary_delta_count": 28760,
  "next_required_implementation": "materialize_action_apply_error_and_secondary_control_outcomes",
  "no_event_preservation_pass": 0,
  "null_rate_heldout": 0.0,
  "observability_gap_pass": 1,
  "official_sample_coverage": 0.0,
  "ogp_decision_pass": 0,
  "oracle_bad_event": 0.0,
  "oracle_beats_adamwparallel_rate": "",
  "oracle_beats_bestlr_rate": "",
  "oracle_coverage": 0.05445326278659612,
  "oracle_null_rate": 0.0,
  "oracle_precision": 1.0,
  "oracle_value_lcb": "",
  "paired_replay_pass": 0,
  "precision_heldout": 0.0,
  "precision_lcb": 0.0,
  "primary_blocker": "action_apply_and_secondary_control_materialization_missing",
  "primary_oracle_pass": 1,
  "proxy_row_used": 0,
  "robustness_pass": 0,
  "route": "R1-BoundaryReanalyzed",
  "runtime_candidate_id": "RT0-v9300-reference-fixed-per-step",
  "runtime_mode": "online_sequential_official_reference",
  "secondary_outcome_ready": 0,
  "short_run_pass": 0,
  "source_route_v9300": "R9-OGPFeatureFail",
  "stableaccept_patch_exhausted": 1,
  "step_ratio_q90": "2.213009156635521",
  "strong_baseline_pass": 0,
  "success_v9310_external_ready": 0,
  "success_v9310_full_functional": 0,
  "success_v9310_strict_purekan_functional": 0,
  "support_balance_pass": 0,
  "system_legal_controller_pass": 0,
  "useful_oracle_pass": 0,
  "uses_dataset_name_for_controller": 0,
  "uses_loss_backward": 0,
  "uses_loss_modification": 0,
  "uses_teacher": 0,
  "value_mean_heldout": ""
}
```

判断：v9.3.1 按计划没有继续调 StableAccept/DR7。由于 P1 action apply error 与 P2 secondary/control outcomes 都未 materialize，controller 和 downstream 必须 gate-block。

## 3. P1/P2 materializer blocker

```text
action_apply_error_measured = 0
action_apply_error_missing_count = 2876
secondary_outcome_ready = 0
missing_secondary_delta_count = 28760
matched_control_count_per_event = 0
```

判断：当前 artifact 只有 action/candidate identity、payload/hash 和 primary labels；没有可审计的 action apply error 或 matched-control branch outcomes。不能进入 official action-value controller。

## 4. Observability Diagnostic

```text
primary_oracle_pass = 1
useful_oracle_pass = 0
control_oracle_pass = 0
observability_gap_pass = 1
best_observable_id = OGP-C1-CandidateScore
best_observable_BadCountAt273 = 49
best_observable_SafeGoodCountAt273 = 210
best_observable_NullCountAt273 = 16
```

判断：primary safe-good oracle 仍强，但这不是 deployable controller。现有 legal observables 不是 action-conditioned value/risk observables，且 feature cost 未测。

## 5. Runtime Boundary

```text
runtime_candidate_id = RT0-v9300-reference-fixed-per-step
runtime_mode = online_sequential_official_reference
empty_step_controller_kernel_count = 59868
controller_launches_per_active_step_q90 = 9.0
step_ratio_q90 = 2.213009156635521
event_driven_runtime_pass = 0
```

判断：本轮没有 measured zero-candidate skip 或 active-step fused runtime；diagnostic empty-step skip 继续保持 diagnostic，不能 official。

## 6. Downstream Boundary

P11-P14 均以 `not_run` 落盘，原因是 `P10_system_controller_not_official`。没有把 primary oracle、observability diagnostic 或 runtime estimate 写成 paired replay / short-run / full-run success。

## 7. No-fake Audit

```text
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
diagnostic_downstream_used_for_controller = 0
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| plan | `ca4d95898d4c06414457f1809e8b5388e79b1c90afe48fbed2948ebb664bea41` |
| runner | `bf7a218e18ee4a7c336f8cd704271bd8779352ca52dd3918b5a2e62ee351ae5a` |
| run manifest | `2c235e9046797f861303aef898da784975e924fe9eda73cb34ec6d9da1920197` |
| route | `1eebefce9356c30481b8568a6a248f5897241c5074cb34a546e0ce07ac20f3e7` |
| P0 boundary | `560c6fe1801da58e392cc46779f08ab1c3eec1a36d59853b524e112c564c7bb0` |
| P1 action lifecycle | `08584876901837266ae10ad62fcac3aef185033a5bb2afb575a9e158921697cc` |
| P2 secondary control | `7b545523b73ff11cd5f252361f78c0661be1d28b946244ce81b9a600cd7493cd` |
| P3 oracle | `e8232c22ebf7536813292cb21da3f32ee4e6e156a4ba28a6f50e7723d553fa55` |
| P4 observability | `e5a466afb25c5be6f32d6db9237be98061beb43ed276dee2b497dc2a5ae795d5` |
| P5 observable factory | `35ffd72af9b9fe38246b2c3b389ca3285d44a10851d7d806b0dfe0d099134276` |
| P6 controller | `b06f29d9d26c104b6c3519b35b890e6f561fdcf97eb27e584391347f7d0cd903` |
| P7 autopsy | `de303246bb06b14d7b7d54351a486e3e3174601c617a5b0abfc549d5ec86a454` |
| P8 primitive | `71697e0bbce6052fca9ba40bf39ecc8e9449e1a7ef97ff7ba655cb4162c3f262` |
| P9 runtime | `615be4833fa181f92aa51751d56c47bdbb81863174f12cf3d44d0d74931ef662` |
| P10 system | `159de4e1a122107d8072c832fa99d66295113c97a155c73297b6e4e508a2602a` |
| contract | `a806fb4d066a49d6ab6b32dad988308de1f04c65f36616d0e7ace9996242ed44` |
| provenance | `f9fb2ba459a6fbeaefb9d25c3be5295432ccc87b558d2d107bd6ca5c0c29099a` |


## 9. 最终分析结论

v9.3.1 的真实推进是：

```text
v9.3.0:
  candidate/action population 有 oracle frontier；
  但 legal OGP feature、secondary/control outcome 和 measured event runtime 未闭合。

v9.3.1:
  严格审计 action apply 与 secondary/control materializer；
  发现当前落盘 artifact 仍不能提供 action apply error、matched controls 或完整 secondary outcomes；
  因而按计划停止 controller/downstream promotion。
```

机制判断：

1. H1 仍成立但只能停在 primary oracle：oracle frontier 存在，但 useful/control oracle 无法在缺 secondary/control outcomes 时声明。
2. H2 未闭合：现有 feature 仍是 state/support diagnostic，不是 action-conditioned value observable。
3. H3 未闭合：secondary/control outcomes 缺失，action value 的完整定义不存在。
4. H4 未闭合：event-driven runtime 没有 measured materialization。
5. 当前下一步必须先实现真实 action apply error materializer 和 secondary/control outcome materializer，再回到 legal action-value observable/controller。

最终一句话：

> v9.3.1 真实执行后停在 `R1-BoundaryReanalyzed`：v9.3.0 的 oracle frontier 没回退，但 action apply error 与 secondary/control outcome 仍缺失，measured event-driven runtime 未实现；strict PureKAN functional 仍不能转正。
