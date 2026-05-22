# DG-KAN v10.7 Controllability Reset 实验复盘

> 本复盘记录 `DG-KAN_v10.7_ControllabilityReset_深度反思与完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.6/v10.3/v10.1 真实 artifact、本轮真实 controllability basis payload replay、真实 optimizer-state-induced payload replay 与真实 branch-horizon rows；没有 fake data、proxy rows、占位数据或 CPU offload。controller/runtime/paired replay 仅在硬 gate 通过后打开。

## 0. 最新结论

```text
route = CaseD-WeakParameterPocketsNotControllable
primary_blocker = weak_parameter_pockets_not_pareto_feasible
secondary_blocker = optimizer_state_and_architecture_bridge_failed
system_legal_controller_pass = 0
generated_route_status = stopped_weak_pockets_not_functional_controllability
```

最终 artifact：`results/real_rerun_20260506/v1070_controllability_reset_full_20260518T235000Z`

核心结论：

1. P0 v10.6 boundary lock pass = `1`；old LFRO/FPO/GoodCone route stopped = `1`。
2. P1 label stability pass = `0`；Fast/Slow/Risky/SafeLow/Bad = `4` / `22` / `321` / `26` / `14707`。
3. P2 basis actions = `480`，branch rows = `14400`；risk-safe positive = `4`，rate = `0.008333333333333333`；weak/strong = `1` / `0`。
4. P3 Pareto efficient = `4`；risk-safe positive Pareto = `1`；weak/strong = `0` / `0`。
5. P4 gated LFRO refit = `not_run`；reason = `P2_or_P3_controllable_response_gate_not_passed`。
6. P5 optimizer-state induced candidates = `7`；risk-safe positive = `0`；weak/strong = `0` / `0`。
7. P6 architecture bridge pass = `0`；blockwise limited-real rate = `0.020833333333333332`。
8. P7/P8/P9 = `summary` / `not_run` / `not_run`；No-fake rows checked = `31660`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1070_controllability_reset.py
```

```bash
python experiments/run_v1070_controllability_reset.py --out-dir results/real_rerun_20260506/v1070_controllability_reset_full_20260518T235000Z --fresh --device auto --data-root data --seed 1919 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1070",
  "status": "summary",
  "route": "CaseD-WeakParameterPocketsNotControllable",
  "primary_blocker": "weak_parameter_pockets_not_pareto_feasible",
  "secondary_blocker": "optimizer_state_and_architecture_bridge_failed",
  "route_explanation": "Parameter basis produced weak isolated risk-safe positives, but Pareto/LFRO/generated gates did not convert them into controllability.",
  "source_route_v1060": "CaseD-NoRiskSafeLocalFutureResponse",
  "P0_boundary_pass": "1",
  "P2_weak_pass": "1",
  "P2_strong_pass": "0",
  "P3_weak_pass": "0",
  "P3_strong_pass": "0",
  "P4_weak_pass": "0",
  "P4_strong_pass": "0",
  "P5_weak_pass": "0",
  "P5_strong_pass": "0",
  "P6_bridge_pass": "0",
  "P7_stage64_weak_pass": "0",
  "P8_controller_pass": "0",
  "P9_runtime_pass": "0",
  "P9_paired_replay_pass": "0",
  "generated_route_status": "stopped_weak_pockets_not_functional_controllability",
  "system_legal_controller_pass": "0",
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P2 Controllability Basis Expansion

| metric | value |
| --- | ---: |
| basis_action_count | 480 |
| state_count | 6 |
| basis_family_count | 10 |
| branch_horizon_expected_rows | 14400 |
| branch_horizon_actual_rows | 14400 |
| FastGood_count | 0 |
| SlowBurnGood_count | 2 |
| RiskyHighAUV_count | 19 |
| SafeLowValue_count | 2 |
| BadPath_count | 457 |
| risk_safe_positive_count | 4 |
| risk_safe_positive_rate | 0.008333333333333333 |
| risk_safe_positive_state_count | 4 |
| risk_safe_positive_basis_family_count | 4 |
| best_basis_family | old_family |
| best_path_type | SafeLowValue |
| best_V240 | 0.6702954513020813 |
| V240_LCB_all | -0.5806200300026599 |
| longrisk_UCB_all | 0.988645266229703 |
| materializer_contract_pass | 1 |
| P2_weak_pass | 1 |
| P2_strong_pass | 0 |

Top P2 candidates:

| basis_family | basis_id | path | V20 | V80 | V240 | RAUV | risk | positive |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old_family | B4-OldFamilyPreserving | SafeLowValue | -0.1986205445136875 | 1.3831156510859728 | 0.6702954513020813 | 0.4044809639453888 | 0 | 1 |
| blockwise_edge_basis | B7-BlockwiseEdgeBasis | SafeLowValue | 0.5858225969132036 | -0.16820808756165206 | 0.16291281813755631 | 0.11082880501635373 | 0 | 1 |
| signal_channel_snr | B5-SignalChannelSNR | SlowBurnGood | 0.13174660201184452 | 0.07357263774611056 | 0.0370302596129477 | 0.01355617018416524 | 0 | 1 |
| optimizer_current | B0-AdamWDirection | SlowBurnGood | 0.7174831537995487 | 0.2922741584479809 | 0.03467259695753455 | 0.1575877618044615 | 0 | 1 |
| old_family | B4-OldFamilyPreserving | BadPath | -1.7929561629425734 | -1.0548838556278497 | 1.3338774049188942 | -0.5809038114733994 | 1 | 0 |
| blockwise_edge_basis | B7-BlockwiseEdgeBasis | BadPath | -1.5909279233310372 | -1.8093959940597415 | 1.174712327774614 | -0.7462468550540506 | 1 | 0 |
| hard_tail | B3-HardTailCorrection | BadPath | -0.40292517840862274 | -0.3898475565947592 | 1.0004341742023826 | -0.30457437797449527 | 1 | 0 |
| old_family | B4-OldFamilyPreserving | RiskyHighAUV | -0.294318271568045 | -0.28399659460410476 | 0.9395502582192421 | 0.03289955658838153 | 1 | 0 |
| random_orthogonal | B6-LowRankRandomOrthogonal | BadPath | -1.164076100103557 | 0.007336524315178394 | 0.6336523883510381 | -0.3226780456956476 | 1 | 0 |
| signal_channel_snr | B5-SignalChannelSNR | RiskyHighAUV | 1.1390522113069892 | -0.619290514383465 | 0.605193458031863 | 0.14894356317818164 | 1 | 0 |
| curvature_trust_region | B9-CurvatureTrustScaled | BadPath | -1.4908923101611435 | 0.8177770962938666 | 0.5197944100946188 | -0.10009321360848844 | 1 | 0 |
| optimizer_state_alignment | B1-MomentumDirection | BadPath | -0.4140145273413509 | -1.360805788775906 | 0.49411530513316393 | -0.5527802580501884 | 1 | 0 |

## 4. P3/P4/P5/P6

| stage | key metrics |
| --- | --- |
| P3 Pareto | efficient=4, risk_safe_positive=4, weak=0, strong=0 |
| P4 LFRO refit | status=not_run, reason=P2_or_P3_controllable_response_gate_not_passed |
| P5 optimizer-state | positive=0, precision=0.0, V240=-1.6153376087599987, risk=1.0, weak=0 |
| P6 bridge | bridge=0, B7_rate=0.020833333333333332, conclusion=no_limited_real_architecture_bridge_signal |

Top P5 optimizer-state induced rows:

| optimizer_update | path | V20 | V80 | V240 | RAUV | risk | positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| O6-hard-tail-local-step-rescale | BadPath | 0.9157764127012342 | -0.24919088673777878 | -0.08343206299468875 | -0.007907898351550102 | 1 | 0 |
| O5-blockwise-orthogonalized-update | BadPath | -0.10604797350242734 | -0.23118678806349635 | -0.09287685854360461 | -0.1205964564345777 | 1 | 0 |
| O1-momentum-damp-amplify | BadPath | -0.830866419011727 | 0.19141009682789445 | -0.2854710458777845 | -0.22343740430660547 | 1 | 0 |
| O2-second-moment-trust-scaling | BadPath | -0.1815776366274804 | -0.5644864074420184 | -0.319953469093889 | -0.34811182590201495 | 1 | 0 |
| O7-old-family-preserving-state-correction | RiskyHighAUV | 1.0832389679271728 | 1.1572482259944081 | -0.6119994050823152 | 0.10522026177495718 | 1 | 0 |
| O4-signal-channel-snr-preconditioner | BadPath | -1.6634423034265637 | -3.0711380601860583 | -0.9263223193120211 | -1.6356029830873013 | 1 | 0 |
| O3-memory-safe-preconditioner | BadPath | -1.283527836902067 | -1.972424428910017 | -3.2207482662051916 | -1.6428989257663489 | 1 | 0 |

## 5. Gate Decision

| stage | status | reason/key |
| --- | --- | --- |
| P7 generated | summary | Stage4_reused_true_P2_branch_horizon |
| P8 controller | not_run | P4_P5_P7_strong_or_stage64_gates_not_passed |
| P9 runtime/paired | not_run | P8_controller_not_passed |

## 6. No-Fake / Contract Audit

```text
rows_checked = 31660
fake/proxy/cpu = 0 / 0 / 0
```

## 7. SHA256

| artifact | SHA256 |
| --- | --- |
| `contract_audit_v1070.csv` | `26655b00f1fddf2b21617cb2e437ece5bab179bda0a7718c34314b3013bc26d0` |
| `fig_p1_path_type_continuous_v1070.svg` | `f0e8f08bae3e54a3b71ca57f7ca8a1b3f9364b8281284c10cd575020e6abf6d2` |
| `fig_p2_basis_controllability_v1070.svg` | `58ea4b48a298f67f09458f8600a6944106b672116c91b4c3a9a02b90eda613ed` |
| `fig_p3_pareto_v1070.svg` | `4ef7782625de7c25115265a4c0f5d479b832b2444e4e4e5924c2a1d39149c22d` |
| `fig_p5_optimizer_state_v1070.svg` | `29be269c8042773e1d0d8e9f5538d80a5870388487cf6a1ccbb8fe227697b0bb` |
| `fig_p6_arch_bridge_v1070.svg` | `aded988cd892fbce943eebfb7b84703c53cfc75c4d6953bd96b1ec46cc89f7ea` |
| `fig_v1070_route_matrix.svg` | `730a5d53157cd0937596e0ced1ebc22caa6670411ca2b75db4120425753b20c6` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1070.csv` | `7bf258008d4d916a40ecf9eca3f776b98a2824df6ed740f96647f16cd46e4005` |
| `p0_v1060_boundary_lock_v1070.csv` | `645f1fb17690cc5609302d3a5275b8ebc06d009f2c794e82f485cc92166f6ae0` |
| `p1_path_type_continuous_targets_v1070.csv` | `588880ee78967627a446370beb328394fa94946d2366f5f224dee6525cc1be83` |
| `p2_controllability_basis_action_apply_replay_v1070.csv` | `d60f9e8600337aa323acc338ed24ca0c2c44462a34d6a76604144508155f68fd` |
| `p2_controllability_basis_actions_v1070.csv` | `ec45dbad9ce4712aa6999227a3c17a346915fa2ef6edf11f9879302e4024af66` |
| `p2_controllability_basis_branch_horizon_v1070.csv` | `af0704ea96cfc057e270d7b2d6e8a81df52d54d7e90223111c8cc699bd0fd3e7` |
| `p2_controllability_basis_expansion_v1070.csv` | `0359cd4cee4e7ce10f77cc358c6f764dfa767240c9a9725f276d9bd5274c91ad` |
| `p2_controllability_basis_labels_v1070.csv` | `69b3cf4c27bc7bcd53ea5d2e22db4d1402b6ac9716f0177e3f57050ed338952c` |
| `p3_risk_value_pareto_frontier_v1070.csv` | `e02c3849c598a8703ec826782eeaabdcdb5af4d0f146caef0c22c33324f35525` |
| `p4_gated_lfro_refit_v1070.csv` | `6c1609ad5558d2b4888da42aa3fb00d3850a814aa807edaf7522e3bf71410b93` |
| `p5_optimizer_state_action_apply_replay_v1070.csv` | `545f0fce77a5f4e44b90303ee5df1cd8c9e762bf80fd277235bd6e2167567055` |
| `p5_optimizer_state_branch_horizon_v1070.csv` | `fe0d93cfa1e6e7215a9d88a638cfe49735da00bc598320db6a79378e343c3afd` |
| `p5_optimizer_state_labels_v1070.csv` | `d3bd94e2735292cd640fd8eaf94b70eb8de3dc7482fe7bd594f182b7e4b4d170` |
| `p5_optimizer_state_update_actions_v1070.csv` | `ab682df4742c8bb0cdc818c1bea5fb72d1fa9ad4fe1cb45759e57fe1c964383b` |
| `p5_optimizer_state_update_discovery_v1070.csv` | `9ae688d42a87223e7f625ec1a523681682876baab4391f692c3a85d9bacb8f8e` |
| `p6_architecture_compositional_bridge_v1070.csv` | `bd855e53f86029942f710daa10ae7f0efbed1e26c97905fea8d64fe60f4b3322` |
| `p7_generated_discovery_staircase_v1070.csv` | `08dc486ea2794e7d7a5699dd9bd0bf97546f966fa3d4469ffc3f30cc7298627f` |
| `p8_controller_boundary_v1070.csv` | `3d66ae4e8485037a36ffad41b813480a8792b80bd897566e9d9d4ae0808c05bc` |
| `p9_runtime_paired_replay_boundary_v1070.csv` | `9a43981e42a62ece016cbcc6fb4abc3f09333dad19a51351db28b692df862582` |
| `plan` | `225f0bfd4d01cd71c7467591e891ffb5409115d831013d526c8866bef09e85bb` |
| `route_decision_v1070.json` | `c132f8e154eac0c2e7479e6e3b2f39e2665b45b7e62ff050919134b365caf757` |
| `run_manifest_v1070.json` | `750ea6583b82b4b41c3e08e4bc56a01fafcffe3ad47427389577632df4ef525f` |
| `runner` | `d0b4917340246b1ca6bf9cd1f74b3b59c5f6b3e60fd8bfcf90a8b9724ce2b964` |

## 8. 最终分析结论

```text
1. v10.7 按计划停止旧式 LFRO/FPO/GoodCone 小修，转为 controllability reset。
2. P2 扩展到 B0-B9 basis family，并覆盖 sign/norm/soft-trust 变体；所有响应来自真实 h1/h5/h20/h80/h240 branch-horizon。
3. P3 只判断 value-risk Pareto feasibility；P4 只有在 P2/P3 有可控 response 后才允许拟合 LFRO。
4. P5 尝试 optimizer-state-induced update，但未把 induced parameter delta 写成 direct optimizer-state official pass。
5. P6 synthetic diagnostic 只作为 architecture bridge 的辅助，是否 pass 取决于 limited-real blockwise signal。
6. P8/P9 未满足硬 gate 时保持 not_run，不降 official controller 门槛。
```

最终一句话：v10.7 真实执行后停在 `CaseD-WeakParameterPocketsNotControllable`：Parameter basis produced weak isolated risk-safe positives, but Pareto/LFRO/generated gates did not convert them into controllability.
