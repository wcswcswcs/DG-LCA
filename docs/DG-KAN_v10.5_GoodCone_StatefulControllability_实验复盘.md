# DG-KAN v10.5 GoodCone / Stateful Controllability 实验复盘

> 本复盘记录 `DG-KAN_v10.5_GoodCone_StatefulControllability_结果解读与完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.4/v10.3/v10.1 真实 artifact、真实 payload CUDA sketch、真实 tiny virtual AdamW unroll diagnostic 与 P6 gate-closed not_run rows（本轮未生成 GoodCone branch-horizon）；没有 fake data、proxy rows、占位数据或 CPU offload。

## 0. 最新结论

```text
route = CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient
primary_blocker = goodcone_not_valid
secondary_blocker = mini_unroll_adjoint_no_future_signal
system_legal_controller_pass = 0
generated_route_status = stopped_representation_insufficient
```

最终 artifact：`results/real_rerun_20260506/v1050_goodcone_stateful_controllability_full_20260518T220000Z`

核心结论：

1. P0 v10.4 boundary lock pass = `1`；source route = `CaseB-GoodSubspaceAbsentOrTooDiffuse`；fake/proxy/cpu = `0` / `0` / `0`。
2. P1 state buckets = `720`，weak/strong = `0` / `0`；best bucket AUC = `1.0`。
3. P2 GoodCone weak/strong = `0` / `0`；best = `linear_cone`，TopK32 precision/V240/risk = `0.03125` / `-0.20918085482838006` / `0.9675990373736805`。
4. P3 soft Pareto weak/strong = `0` / `0`；best alpha/lambda = `0.0` / `0.25`，retention = `1.0`。
5. P4 basis coverage weak/strong = `0` / `0`；FastSlow/Bad coverage = `1.0` / `1.0`；risk-safe dim = `0`。
6. P5 mini-unroll weak/strong = `0` / `0`；best k = `1`，partial corr V240|V1 = `-0.0017652611090787737`，AUC SlowBurn vs Risky = `0.6931818181818181`。
7. P6 Stage4/8/32/64 opened = `0` / `0` / `0` / `0`；generated discovery pass = `0`；best = `` ``。
8. P7 weak/strong = `0` / `0`；TopK64 precision/SlowBurn recall/risky/V240 = `0.046875` / `0.13636363636363635` / `0.15625` / `-0.1964372838440122`。
9. P8/P9/P10 = `not_run` / `not_run` / `not_run`；No-fake rows checked = `411`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1050_goodcone_stateful_controllability.py
```

```bash
python experiments/run_v1050_goodcone_stateful_controllability.py --out-dir results/real_rerun_20260506/v1050_goodcone_stateful_controllability_full_20260518T220000Z --fresh --device auto --data-root data --seed 1717 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1050",
  "status": "summary",
  "route": "CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient",
  "primary_blocker": "goodcone_not_valid",
  "secondary_blocker": "mini_unroll_adjoint_no_future_signal",
  "route_explanation": "Neither GoodCone nor mini-unroll adjoint provides enough future-path signal under current representation.",
  "source_route_v1040": "CaseB-GoodSubspaceAbsentOrTooDiffuse",
  "P0_boundary_pass": 1,
  "P1_weak_pass": 0,
  "P1_strong_pass": 0,
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P3_weak_pass": 0,
  "P3_strong_pass": 0,
  "P4_weak_pass": 0,
  "P4_strong_pass": 0,
  "P5_weak_pass": 0,
  "P5_strong_pass": 0,
  "P6_generated_discovery_pass": 0,
  "P6_stage64_strong_discovery_pass": 0,
  "P7_weak_pass": 0,
  "P7_strong_pass": 0,
  "P8_controller_pass": 0,
  "P9_runtime_pass": 0,
  "P10_paired_replay_pass": 0,
  "generated_route_status": "stopped_representation_insufficient",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. Contract Audit

```text
P0_pass = 1
no_v1050_natural_panel_artifact = 1
P6_no_stage8_without_stage4_gate = 1
P6_no_stage32_without_stage8_gate = 1
P6_no_stage64_without_stage32_gate = 1
controller_not_run_without_hard_gate = 1
```

## 4. P1 State-Conditioned Contrast

| metric | value |
|---|---:|
| known_action_rows | `15080` |
| Fast/Slow/Risky/SafeLow/Bad | `4` / `22` / `321` / `26` / `14707` |
| state_bucket_count | `720` |
| evaluated_bucket_count | `23` |
| weak_bucket_count | `0` |
| best_bucket_auc | `1.0` |
| consistent_feature_family | `` |
| P1 weak/strong | `0` / `0` |

Top state buckets：

| bucket | FS | controls | best feature | AUC | effect | weak |
|---|---:|---:|---|---:|---:|---:|
| `KMNIST|114813|A3-LateAttachRoleWiseFT7EdgeCarrier|s192_223` | `1` | `7` | `current_response` | `1.0` | `0.0` | `0` |
| `KMNIST|137383|A1-RiskBoundedTailCarrier|s128_159` | `1` | `13` | `current_response` | `1.0` | `0.0` | `0` |
| `KMNIST|282974|A3-LateAttachRoleWiseFT7EdgeCarrier|s192_223` | `1` | `24` | `optimizer_state_alignment` | `1.0` | `0.0` | `0` |
| `KMNIST|335775|A3-LateAttachRoleWiseFT7EdgeCarrier|s192_223` | `1` | `60` | `exact_transfer` | `1.0` | `0.0` | `0` |
| `KMNIST|417981|A1-RiskBoundedTailCarrier|s032_063` | `1` | `5` | `current_response` | `1.0` | `0.0` | `0` |
| `KMNIST|671146|A2-LateAttachControlGapChannel|s256_287` | `1` | `7` | `exact_transfer` | `1.0` | `0.0` | `0` |
| `KMNIST|708410|A1-RiskBoundedTailCarrier|s192_223` | `1` | `15` | `exact_transfer` | `1.0` | `0.0` | `0` |
| `KMNIST|922485|A1-RiskBoundedTailCarrier|s160_191` | `1` | `6` | `exact_transfer` | `1.0` | `0.0` | `0` |
| `MNIST|282974|A3-LateAttachRoleWiseFT7EdgeCarrier|s160_191` | `1` | `54` | `current_response` | `1.0` | `0.0` | `0` |
| `MNIST|797103|A3-LateAttachRoleWiseFT7EdgeCarrier|s128_159` | `1` | `11` | `exact_transfer` | `1.0` | `0.0` | `0` |
| `MNIST|282974|A3-LateAttachRoleWiseFT7EdgeCarrier|s192_223` | `2` | `42` | `exact_transfer` | `0.9880952380952381` | `2.2796201327204866` | `0` |
| `MNIST|335775|A3-LateAttachRoleWiseFT7EdgeCarrier|s160_191` | `1` | `75` | `exact_transfer` | `0.9866666666666667` | `0.0` | `0` |

## 5. P2 GoodCone / Quadratic Metric

| metric | value |
|---|---:|
| evaluated_known_action_count | `1397` |
| Good/Bad/Risky/SafeLow/Generated | `26` / `1024` / `321` / `26` / `24` |
| basis_rank / explained_variance | `16` / `0.9998586957815142` |
| good coverage / good-bad separation | `1.0` / `0.022603988647460938` |
| generated_projection_norm | `0.9941820800304413` |
| best_basis_id | `linear_cone` |
| TopK32 precision / V240 / longrisk | `0.03125` / `-0.20918085482838006` / `0.9675990373736805` |
| P2 weak/strong | `0` / `0` |

| basis | TopK32 FS | TopK32 V240 | TopK32 longrisk | TopK64 FS | TopK64 V240 | weak | strong |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_cone` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0.046875` | `-0.1964372838440122` | `0` | `0` |
| `diag_lda` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0.046875` | `-0.1964372838440122` | `0` | `0` |
| `exact_transfer_signed_cone` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0.046875` | `-0.1964372838440122` | `0` | `0` |

## 6. P3 Soft Trust Pareto

| alpha | lambda | retention | TopK32 FS | V240 | longrisk | weak |
|---:|---:|---:|---:|---:|---:|---:|
| `0.0` | `0.0` | `1.0` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0` |
| `0.0` | `0.25` | `1.0` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.0` | `0.5` | `1.0` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.1` | `0.0` | `0.9044007658958435` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0` |
| `0.1` | `0.25` | `0.9044007658958435` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.1` | `0.5` | `0.9044007658958435` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.25` | `0.0` | `0.7619336247444153` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0` |
| `0.25` | `0.25` | `0.7619336247444153` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.25` | `0.5` | `0.7619336247444153` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.5` | `0.0` | `0.5287730097770691` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0` |
| `0.5` | `0.25` | `0.5287730097770691` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.5` | `0.5` | `0.5287730097770691` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.75` | `0.0` | `0.30849090218544006` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0` |
| `0.75` | `0.25` | `0.30849090218544006` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `0.75` | `0.5` | `0.30849090218544006` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `1.0` | `0.0` | `0.1503976434469223` | `0.03125` | `-0.20918085482838006` | `0.9675990373736805` | `0` |
| `1.0` | `0.25` | `0.1503976434469223` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |
| `1.0` | `0.5` | `0.1503976434469223` | `0.4375` | `0.07533975305675933` | `0.1071827150573635` | `0` |

## 7. P4 Basis Coverage / Controllability

| metric | value |
|---|---:|
| basis_dim | `16` |
| condition_number | `78064.046875` |
| FastSlow_projection_coverage | `1.0` |
| RiskyBad_projection_coverage | `1.0` |
| FastSlow_vs_Bad_cone_margin | `6.41821060408067e-05` |
| risk_safe_basis_subset_dim | `0` |
| P4 weak/strong | `0` / `0` |
| conclusion | `FastSlow_and_Bad_both_high_coverage_requires_GoodCone` |

## 8. P5 Mini-Unroll Adjoint Validation

| k | actions | cost ms | corr V1 | partial V240\|V1 | AUC Slow/Risky | TopK32 FS | V240 | longrisk | weak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `102` | `524.3669756980357` | `0.17415210894743097` | `-0.0017652611090787737` | `0.6931818181818181` | `0.09375` | `-0.7285388627170879` | `0.9675990373736805` | `0` |
| `3` | `102` | `1.9755176469391467` | `0.2731083106367367` | `-0.06672923406795012` | `0.7443181818181819` | `0.125` | `-0.784101436317192` | `0.9502994615552569` | `0` |
| `5` | `102` | `2.825038402578702` | `0.05744665544855965` | `-0.040894807396344546` | `0.6212121212121212` | `0.15625` | `-0.7441170409006583` | `0.8897630306846814` | `0` |
| `10` | `102` | `4.950485055280082` | `-0.015799450181617363` | `-0.08491419857800439` | `0.6022727272727273` | `0.125` | `-0.9012569381362547` | `0.9111057606383722` | `0` |

## 9. P6 GoodCone Generated Discovery

| stage | n | rows | apply | Fast | Slow | Risky | SafeLow | Bad | FS precision | V240 | longrisk | gates | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `P6_GOODCONE_GENERATED_DISCOVERY_V1050` |  |  |  |  |  |  |  |  |  |  |  | `not_run: P2_P3_P4_P5_no_usable_line` |  |

## 10. P7 FPO Path-Type Predictor

| metric | value |
|---|---:|
| evaluated_action_count | `1397` |
| TopK64_FastSlow_precision | `0.046875` |
| SlowBurn_recall | `0.13636363636363635` |
| RiskyHighAUV_rate | `0.15625` |
| V240_LCB | `-0.1964372838440122` |
| longrisk_UCB | `0.9459995524731696` |
| cost_ms_q90 | `0.00017113235639774893` |
| P7 weak/strong | `0` / `0` |

## 11. P8/P9/P10 Gate Decision

| stage | status | reason |
|---|---|---|
| P8 controller | `not_run` | `P6_stage64_and_P7_strong_gates_not_passed` |
| P9 runtime | `not_run` | `P8_controller_not_passed` |
| P10 paired replay | `not_run` | `P8_or_P9_not_passed` |

## 12. No-Fake / Hash

```text
rows_checked = 411
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `contract_audit_v1050.csv` | `013c0e1d18eb0fef98f1eb8b86fce1114e712fecdccb369dbdfa789a058eab4a` |
| `fig_p1_state_bucket_signal_v1050.svg` | `134998b09c9643c11dce80e885d27786830fa3720c9f98e6e783a92b14187c1e` |
| `fig_p2_goodcone_metric_v1050.svg` | `4649b4e73fdf79c8f153a7372d827033a9ae4a02b77cc8670a735b11e6645ff3` |
| `fig_p3_pareto_v1050.svg` | `97705d7b4034575c0df5703f842925834ef9203e43fbcc29b736ad82669b1810` |
| `fig_p4_basis_coverage_v1050.svg` | `2a2f3580d59965cd59551350f098c7c55afb08919f6b9b6e7d7f7f2f0bc2e6a6` |
| `fig_p5_unroll_signal_v1050.svg` | `97381aae1a6345ba181aa23455f14b195cf6990d73b5d5424272e214c9473342` |
| `fig_p6_generated_waterfall_v1050.svg` | `2f74be3f3e4ffe0b396411d0d84af7e19f100410713a573cdb14f02dcde816d5` |
| `fig_p7_fpo_path_type_v1050.svg` | `3df9d9c0022dbc6f0343c884149a9341ba960682a98711a71a78775240ec3e98` |
| `fig_v1050_route_matrix.svg` | `515322f4e0f651ebf8111336deff018ca9cc9d4d7270e747b1ec178efb673ff3` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1050.csv` | `29a06b624f3a02970776769ed6afe82f59a881ff6a546855587a6a6236b9f6ca` |
| `p0_v1040_boundary_lock_v1050.csv` | `fd0624165a6bbfcc58d7d1562ffb57e5b3be90e8b2bd8ef991cb69dfa974d629` |
| `p10_paired_replay_boundary_v1050.csv` | `593674ffe347b413f6af67f121db659ab4c9b68170023c899411e92b6bdf4d1e` |
| `p1_state_conditioned_path_contrast_v1050.csv` | `839d58152723bd16433d612bc26183b223115fdca39652fc71ae13ff3f13bae8` |
| `p2_goodcone_quadratic_metric_v1050.csv` | `dca29e3969544e126aaf595931166d6c3a4e33efd876b3c1e0fbdc111920d47b` |
| `p3_memory_offdiag_soft_trust_pareto_v1050.csv` | `e960d2c8173493666a55bd470b8bccecc6ccdddea738517cd4f8e3ea70974b52` |
| `p4_basis_coverage_controllability_audit_v1050.csv` | `993e0b0dd7749e8d14105a43810968ae2b39a2317cacffd06a8e46b3a4528d2f` |
| `p5_mini_unroll_adjoint_validation_v1050.csv` | `27399e40358df850e627c68b93d290f65cf9abb146fcaf73287d26333edb601c` |
| `p6_goodcone_generated_discovery_v1050.csv` | `77d025c61f08ccf61e94c157697a362ef53ce27ad7be87fcdc632603e07459c4` |
| `p7_fpo_path_type_predictor_rebuild_v1050.csv` | `e7bf551041fa72069921b83f598fa0a86de11806af0ff6f4dea890d284a0629e` |
| `p8_controller_boundary_v1050.csv` | `9fad40d41365b8a7c011d344cdb2a02e15a675a1ce9daf3ef063d03e54a8e6dc` |
| `p9_runtime_boundary_v1050.csv` | `5e70e91a1a891c38fca3dfa2465209d5939e7823fa99b91eba7334f29111f178` |
| `plan` | `d1698db1141d0a864ac9386534aa583c193077d041aaa1668bd8d81d5847d390` |
| `route_decision_v1050.json` | `bd21b1ab595d2a386963d5480e465eefb73a3a96abeb28b742e7486f7d46ec28` |
| `run_manifest_v1050.json` | `901d0955887c117edef922550d7f644886ca49e903a2bf18e76858031787141b` |
| `runner` | `5ce0eda19b9d50ddb1dad91e6079dc0949f338c3711f1d67d1b603be1c366007` |

## 13. 最终分析结论

```text
1. v10.5 按计划从固定 GoodSubspace 升级到 state-conditioned GoodCone / Pareto / basis / mini-unroll 诊断。
2. 所有 science diagnostic 都保持 discovery-only；P8/P9/P10 只在 P6 Stage64 strong 或 P7 strong 后打开。
3. memory/offdiag direct metric 当前 schema 不提供，因此本轮没有把相关 diagnostic 写成 official pass。
4. P6 generated 只在 P2/P3/P4/P5 至少一条可用线时打开，Stage8/32/64 严格由前一 stage gate 控制。
5. 最终 route 只来自真实 CSV/JSON/source artifact rows；P6 未开时不声称存在本轮 generated branch-horizon，不使用 fake/proxy/placeholder。
```

最终一句话：v10.5 真实执行后停在 `CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient`：Neither GoodCone nor mini-unroll adjoint provides enough future-path signal under current representation.
