# DG-KAN v10.8 Decisive Controllability Exit 实验复盘

> 本复盘记录 `DG-KAN_v10.8_DecisiveControllabilityExit_完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.7/v10.3/v10.1 真实 artifact、本轮 P1 known-good replay 与 P5 limited-real architecture bridge 的真实 branch-horizon rows；没有 fake data、proxy rows、占位数据或 CPU offload。P2/P3/P4 因 P1 direct replay gate 未通过而显式 `not_run`，没有生成对应 replay rows。controller/runtime/paired replay 仅在硬 gate 通过后打开。

## 0. 最新结论

```text
route = CaseA-KnownGoodDirectReplayFail
primary_blocker = known_good_direct_replay_not_reproducible
secondary_blocker = state_or_replay_identity_mismatch_or_old_labels_not_replayable
system_legal_controller_pass = 0
generated_route_status = stopped_fix_state_replay_identity_before_generated
```

最终 artifact：`results/real_rerun_20260506/v1080_decisive_controllability_exit_full_20260518T235900Z`

核心结论：

1. P0 v10.7 boundary lock pass = `1`；source route = `CaseD-WeakParameterPocketsNotControllable`。
2. P1 known-good direct replay D0 FastSlow-like = `0` / `26`；D0 functional = `0`；D1 projection functional = `0`；P1 weak/strong = `0` / `0`。
3. P2 local cone = `not_run`；reason = `P1_D0_direct_known_good_replay_not_functional`；weak/strong = `0` / `0`。
4. P3 state transfer = `not_run`；reason = `P1_D0_direct_known_good_replay_not_functional`；weak/strong = `0` / `0`。
5. P4 optimizer-state controllability = `not_run`；reason = `P1_P3_no_optimizer_state_intervention_hint`；weak/strong = `0` / `0`。
6. P5 supported/unsupported bridge = `4` / `2`；best = `A5-memory-preserving-edge-residual-basis`，rate = `0.0`；weak/strong = `0` / `0`。
7. P7/P8 = `not_run` / `not_run`；No-fake rows checked = `13375`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1080_decisive_controllability_exit.py
```

```bash
python experiments/run_v1080_decisive_controllability_exit.py --out-dir results/real_rerun_20260506/v1080_decisive_controllability_exit_full_20260518T235900Z --fresh --device auto --data-root data --seed 2028 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1080",
  "status": "summary",
  "route": "CaseA-KnownGoodDirectReplayFail",
  "primary_blocker": "known_good_direct_replay_not_reproducible",
  "secondary_blocker": "state_or_replay_identity_mismatch_or_old_labels_not_replayable",
  "route_explanation": "Known Fast/Slow deltas did not reproduce enough Fast/Slow-like future paths under direct replay.",
  "source_route_v1070": "CaseD-WeakParameterPocketsNotControllable",
  "P0_boundary_pass": 1,
  "P1_D0_direct_functional_success": 0,
  "P1_D1_projection_functional_success": 0,
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
  "generated_route_status": "stopped_fix_state_replay_identity_before_generated",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Known-Good Replay

| metric | value |
| --- | ---: |
| known_FastSlow_action_count | 26 |
| control_Risky_count | 26 |
| control_Bad_count | 26 |
| D0_direct_count | 26 |
| D0_direct_FastSlow_like_count | 0 |
| D0_direct_longrisk_UCB | 1.0 |
| D0_direct_functional_success | 0 |
| D1_projection_count | 26 |
| D1_projection_FastSlow_like_count | 0 |
| D1_projection_median_r2 | 1.0 |
| D1_projection_longrisk_UCB | 1.0 |
| D1_projection_functional_success | 0 |
| D4_soft_trust_FastSlow_like_count | 0 |
| memory_offdiag_metric_available | 0 |
| P1_weak_pass | 0 |
| P1_strong_pass | 0 |
| branch_horizon_actual_rows | 5460 |
| materializer_contract_pass | 1 |

Top P1 replay rows:

| replay | original | path | V20 | V80 | V240 | risk | r2 | cos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0-direct-original | SlowBurnGood | BadPath | -3.070725847966969 | -0.5768484976142645 | -0.9177464612293988 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -0.8992049929220229 | -1.3295913587789983 | -0.5940293106250465 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -0.7388385508675128 | -0.8993123124819249 | -0.564662363845855 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -0.5759471093770117 | -0.43618093710392714 | -0.5573037606664002 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -3.448234514100477 | -0.1063756076619029 | -0.552719546481967 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -1.9049738282337785 | 0.5610976787284017 | -0.49424559134058654 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -0.312355921138078 | 0.2196238017641008 | -0.4031051564961672 | 1 | 1.0 | 1.0 |
| D0-direct-original | FastGood | BadPath | -1.2389735265169293 | -0.3211684091947973 | -0.28817055746912956 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -1.2172746004071087 | -0.811601951951161 | -0.2758359182626009 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -1.5838993617799133 | -2.1882969764992595 | -0.23046175902709365 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | RiskyHighAUV | 0.9354660797398537 | 0.4302793357055634 | -0.20850042207166553 | 1 | 1.0 | 1.0 |
| D0-direct-original | SlowBurnGood | BadPath | -0.8797346346545964 | -0.03223017370328307 | -0.19760166364721954 | 1 | 1.0 | 1.0 |

## 4. P1 Direct Replay Identity Audit

D0 direct replay 失败后，按计划追加检查 state / optimizer state / RNG batch sequence。结论是 payload 本身可重载，但 replay identity 不一致：

| metric | value |
| --- | ---: |
| D0 direct action count | 26 |
| source original h240 RealFunctional rows found | 26 |
| replay h240 RealFunctional rows found | 26 |
| payload hash match count | 26 |
| source vs replay payload hash match | 26 |
| source vs replay state_before_hash match | 0 |
| source vs replay optimizer_state_hash match | 0 |
| source vs replay batch_sequence_hash match | 0 |
| D0 replay FastSlow-like / Risky / Bad | 0 / 1 / 25 |
| metric NaN / Inf | 0 / 0 |

```text
identity_audit_conclusion = payload_reloads_match_but_state_optimizer_batch_identity_mismatch
recommended_fix = repair_state_optimizer_batch_sequence_identity_before_any_generated_or_basis_route
```

## 5. P2 Local Good-Cone

P2 按计划只围绕 P1 中 D0 direct replay 成功的 known-good anchor 打开。本轮 P1 的 26 个 known Fast/Slow direct replay 均未保持 Fast/Slow-like，因此 P2 合法关闭：

```text
status = not_run
reason = P1_D0_direct_known_good_replay_not_functional
P2_weak_pass = 0
P2_strong_pass = 0
```

本轮没有 `p2_local_good_cone_actions`、`p2_local_good_cone_branch_horizon` 或 P2 cone label rows。

## 6. P3/P4/P5

| stage | key metrics |
| --- | --- |
| P3 state phase | status=not_run, reason=P1_D0_direct_known_good_replay_not_functional, weak=0, strong=0 |
| P4 optimizer-state | status=not_run, reason=P1_P3_no_optimizer_state_intervention_hint, weak=0, strong=0 |
| P5 bridge | best=A5-memory-preserving-edge-residual-basis, rate=0.0, V240=0.0, weak=0, unsupported=2 |

P5 bridge summaries:

| bridge | rate | FS precision | V240+ | risk | weak | strong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A5-memory-preserving-edge-residual-basis | 0.0 | 0.0 | 0.0 | 0.9913880422874835 | 0 | 0 |
| A1-blockwise-limited-real-bridge | 0.0 | 0.0 | 0.0 | 1.0 | 0 | 0 |
| A0-current-base-basis | 0.0 | 0.0 | 0.0 | 0.9839312745678148 | 0 | 0 |
| A4-low-rank-dynamic-basis-expansion | 0.0 | 0.0 | 0.0 | 0.9913880422874835 | 0 | 0 |

## 7. Controller / Runtime Gate

| stage | status | reason |
| --- | --- | --- |
| P7 controller | not_run | P1_P2_P5_strong_gates_not_passed |
| P8 runtime/paired | not_run | P7_controller_not_passed |

## 8. No-Fake / Contract Audit

```text
rows_checked = 13375
fake/proxy/cpu = 0 / 0 / 0
```

## 9. SHA256

| artifact | SHA256 |
| --- | --- |
| `contract_audit_v1080.csv` | `ed3045710a5e8aafb4234f5a26cf622436a73a995b493611a4e7d1a16bc79eb2` |
| `fig_p1_known_good_replay_v1080.svg` | `4fd246e111712d4505cdab68811a63792a12768b4898daa0134b3d72c2620191` |
| `fig_p2_local_good_cone_v1080.svg` | `37338c69c982701f9ae4daf1c0b49164f327b4e1370cce6ad93e50d9e4b07698` |
| `fig_p3_state_transfer_v1080.svg` | `926462bb44c230a975b5f468ad5b9cf0c6c993ad7152f18f88923a0b33e29e7f` |
| `fig_p4_optimizer_state_v1080.svg` | `c5d7b31919009437a7fe380a216a3b9b91519c876f04ba8b9b1b47197b55f57e` |
| `fig_p5_architecture_bridge_v1080.svg` | `5fd3ed18e4b0d03240c320e86de71ae37198cf2d2b09682a0860f3fb16d06314` |
| `fig_v1080_route_matrix.svg` | `87a19615e76f184165af9a9d805ac076706c7b624dda486b04accb5c579fe270` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1080.csv` | `a7b49cd82fa4ca12d663e43994b6ed9036516907a57c7323ec89f4b3e6cca26f` |
| `p0_v1070_boundary_lock_v1080.csv` | `1f1a5786993d2b8d298a4e706258364f99174cfb3882f39696d00a573843cd4b` |
| `p1_direct_replay_identity_audit_v1080.csv` | `7a15011df5ea2f9178205b1362b47e8aa40b65018b3db7400cf584bf32eac2a4` |
| `p1_known_good_replay_action_apply_replay_v1080.csv` | `f41c0d9f37a5a24fa43230945c5c4f85a6a217c45c1364ad45f40945a892478e` |
| `p1_known_good_replay_actions_v1080.csv` | `205b7bef427d267a6c46be07606eead32e92cff64640ea376e31d604c94155c5` |
| `p1_known_good_replay_branch_horizon_v1080.csv` | `99d22742d27565c9db63d93434d676d0a3c0c7fbf59bc022d754b3b7ac01d7c8` |
| `p1_known_good_replay_labels_v1080.csv` | `75a7fb9577dce09214e66991a09f25061e89a76fb4f48dd3c954440096733989` |
| `p1_known_good_replay_v1080.csv` | `00aed660c1e5d39c329013a5e651682f05d7c8224341e97ae46bb926b5f266a8` |
| `p2_local_good_cone_v1080.csv` | `6008fa20e6c907ed548ef113cbf7f8c377cc40dd65c9027678fb6f4096ddc172` |
| `p3_state_phase_controllability_v1080.csv` | `6dffbadf8c909038d69cd1cefddefca1791bdd9b2289b34dab9b8c7a15809289` |
| `p4_optimizer_state_controllability_v1080.csv` | `0aa4007146b81ac3a2c6f27a3602f60874cc73398af56150de2c016cd35c8013` |
| `p5_architecture_bridge_action_apply_replay_v1080.csv` | `a302651af0a9feb93d3d7512aa399b8fea983c01f28b19666f60dd76d167740d` |
| `p5_architecture_bridge_actions_v1080.csv` | `f6bf811c7945a9d2c34013a2649a7f726af0d6cdc4ab2ff44ecc2da5f8335f2b` |
| `p5_architecture_bridge_branch_horizon_v1080.csv` | `17ab6f85d2cdb73cba06c6b71faf01a6a3e533512c5b68a7d642213dfb11179a` |
| `p5_architecture_bridge_labels_v1080.csv` | `4e7245237c42588c6390ee3055723e58cd51c54f038b771602b954bde57642ea` |
| `p5_architecture_representation_bridge_v1080.csv` | `f3f2cc4059c76adc2f64ef86184c5fcf406f5511344f508f90d398e12f18c426` |
| `p7_controller_boundary_v1080.csv` | `3512bf4002ae3c2b0edac4559f6672ce45e1e31b52a7891667aedb8058885551` |
| `p8_runtime_paired_replay_boundary_v1080.csv` | `4e833f3a72c25180831da61fb8a3aba830b0623b0a77ab4a69bdf76fa3b2bbd2` |
| `plan` | `e3d283f5aa09375b51c100ad880cd9e97dc1a21203bddfa142956b014e39799a` |
| `route_decision_v1080.json` | `99da9a155b048a16150d03fe62130d3c48519b499acdec7dfdabed6877311b39` |
| `run_manifest_v1080.json` | `b4656f1d248a35a526bdfe361686c6e9f8aa7ff6eace60c6e9ded3a70fd1f26d` |
| `runner` | `2b5db70a8558fd36c40bafb973c13a341a2d7d22ca25880bf1e0e80995682f27` |

## 10. 最终分析结论

```text
1. v10.8 按计划把 old FPO/FPAU/GoodCone/LFRO 小修停掉，先验证 known-good delta 是否能被真实 replay 复现。
2. P1 的 D0/D1/D4 全部来自真实 payload replay；D0 direct replay 已先失败，因此本轮首要 blocker 是 state/replay identity 或 old good label 不可复现，而不是继续调 threshold。
3. P2/P3/P4 都受 P1 direct gate 保护；D0 未过时不继续做 local cone、state transfer 或 optimizer-state intervention。
4. P4 未运行；当前 direct optimizer-state official pass 仍保持关闭。
5. P5 只运行 current materializer 能真实表达的 limited-real bridge；A2/A3 直接 architecture mutation 未支持时显式 not_run。
6. P7/P8 只有 P1/P2/P5 strong 后才打开；未满足时保持 not_run。
```

最终一句话：v10.8 真实执行后停在 `CaseA-KnownGoodDirectReplayFail`：Known Fast/Slow deltas did not reproduce enough Fast/Slow-like future paths under direct replay.
