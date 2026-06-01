# DG-KAN v10.3 Update Rule Theory Rebuild / FPAU 实验复盘

> 本复盘记录 `DG-KAN_v10.3_UpdateRuleTheoryRebuild_FPAU_完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.2/v10.1 真实 artifact、CUDA toy mechanism 与本轮真实 FPAU generated branch-horizon rows；没有 fake data、proxy rows、占位数据或 CPU offload。v10.3 按计划聚焦 FPAU update-rule repair，不把 C-line natural panel 作为本轮 gate。

## 0. 最新结论

```text
route = R5-SubspaceMissingValueDirection
primary_blocker = fpau_known_action_or_toy_signal_present
secondary_blocker = generated_subspace_failed
system_legal_controller_pass = 0
generated_route_status = stopped_fpau_subspace_missing_value_direction
```

最终 artifact：

```text
results/real_rerun_20260506/v1030_update_rule_theory_rebuild_fpau_repair_full_20260518T203000Z
```

一句话结论：

```text
FPAU 在 toy mechanism 上经 repair 后可解，但真实 generated branch-horizon 的 FPAU 子空间仍不能产生 Stage8 weak signal，因此没有资格打开 Stage32 / Stage64 / controller。
```

核心数据：

1. P0 v10.2 boundary lock pass = `1`；source route = `R4-BDDiscoveryFail_UpdateRuleTheoryRebuild`。
2. P1 Fast/Slow/Risky/SafeLow/Bad = `12` / `48` / `842` / `60` / `34286`；FastSlow = `60`，低于 P1 discovery 所需 `64`。
3. P2 FPAU weak/discovery/controller = `0` / `0` / `0`；best = `FPAU-A7-ensemble-costate-hard-risk-veto`；TopK64 FastSlow precision = `0.015625`；TopK87 V240 LCB = `-0.41331319354703405`。
4. P3 初始 FPAU toy 失败后执行 repair；repair 后 mechanism pass = `1`，beats current-gradient toys = `4 / 4`，slowburn 3x random toys = `4 / 4`，memory toy nonpositive = `1`.
5. P4 Stage4/8/32/64 opened = `6` / `6` / `0` / `0`；repair Stage8 opened/pass = `3` / `0`；generated discovery pass = `0`.
6. P5 assigned fraction = `1.0`，dominant failure = `F2-adjoint-current-response-collapse`；同时 P4 主要失败类为 `F4-subspace-missing-value-direction` 与 `F5-subspace-violates-memory-offdiag`。
7. P6/P7/P8 = `not_run` / `not_run` / `not_run`；No-fake audit rows checked = `158364`，fake/proxy/cpu = `0 / 0 / 0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v1030_update_rule_theory_rebuild_fpau.py` | v10.3 runner；执行 P0-P8 gate、P3 repair、P4 repair staircase、manifest/recap。 |
| `experiments/natural_ap0_extension_materializer.py` | 复用真实 payload materializer 与 branch-horizon replay。 |
| `docs/DG-KAN_v10.3_UpdateRuleTheoryRebuild_FPAU_完整实验计划.md` | 本轮执行计划与 gate 定义。 |

```text
python -m py_compile experiments/run_v1030_update_rule_theory_rebuild_fpau.py
```

```bash
python experiments/run_v1030_update_rule_theory_rebuild_fpau.py --out-dir results/real_rerun_20260506/v1030_update_rule_theory_rebuild_fpau_repair_full_20260518T203000Z --fresh --device auto --data-root data --seed 1515 --execution-profile full-gated
```

## 2. 执行边界

本轮执行的是 v10.3 FPAU update-rule rebuild，不是 natural density panel line。P0 只锁定 v10.2 terminal boundary；P3/P4 按计划尝试 FPAU objective/subspace repair。contract audit 中记录：

```text
v1030_plan_has_no_C_line_panel_gate = 1
no_v1030_natural_panel_artifact = 1
controller_not_run_without_hard_gate = 1
```

本轮没有新建 `c_G40*` 或 `*20000*` artifact。

## 3. Route

```json
{
  "stage": "ROUTE_DECISION_V1030",
  "status": "summary",
  "route": "R5-SubspaceMissingValueDirection",
  "primary_blocker": "fpau_known_action_or_toy_signal_present",
  "secondary_blocker": "generated_subspace_failed",
  "route_explanation": "FPAU had diagnostic signal, but generated subspace did not produce Stage64 discovery.",
  "source_route_v1020": "R4-BDDiscoveryFail_UpdateRuleTheoryRebuild",
  "P0_boundary_pass": 1,
  "P1_discovery_pass": 0,
  "P2_weak_pass": 0,
  "P2_discovery_pass": 0,
  "P2_controller_candidate_pass": 0,
  "P3_mechanism_pass": 1,
  "P4_generated_discovery_pass": 0,
  "P4_generated_weak_but_risky": 0,
  "P4_stage64_opened_count": 0,
  "P5_pass": 1,
  "P6_controller_pass": 0,
  "P7_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "generated_route_status": "stopped_fpau_subspace_missing_value_direction",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 4. P0 Boundary Lock

P0 读取 v10.2 真实 artifact：

| item | value |
|---|---:|
| source route | `R4-BDDiscoveryFail_UpdateRuleTheoryRebuild` |
| P2 FPO13 discovery pass in source | `0` |
| P4 GUP13 discovery pass in source | `0` |
| P6/P7/P8 source status | `not_run / not_run / not_run` |
| source fake/proxy/cpu | `0 / 0 / 0` |
| P0 boundary pass | `1` |

P0 通过后才进入 P1-P5。

## 5. P1 Path Type Solidification

P1 复用 v10.2 path-type mechanism row，不新增 natural panel。统计如下：

| path type | count |
|---|---:|
| FastGood | `12` |
| SlowBurnGood | `48` |
| RiskyHighAUV | `842` |
| SafeLowValue | `60` |
| BadPath | `34286` |
| FastSlow total | `60` |
| exclusivity violation | `0` |

P1 discovery gate：

```text
FastSlow >= 64
SlowBurnGood >= 20
RiskyHighAUV >= 50
BadPath >= 1000
exclusivity violation = 0
```

结果：

```text
P1_discovery_pass = 0
reason = FastSlow_count_below_64
```

## 6. P2 FPAU Known-Action Adjoint Scoring

P2 在 `15080` 个真实已知 action 上评估 FPAU A0-A7。所有候选均未过 weak/discovery/controller gate。原因不是成本，成本均在毫秒级以下；主要失败是 TopK FastSlow precision 太低且 V240 LCB 为负。

| candidate | TopK64 FS | TopK64 V240 | TopK64 longrisk | RiskReject | BadReject | weak | discovery | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `FPAU-A0-current-gradient-baseline` | `0.0` | `-1.0727252151909152` | `1.0` | `0.9937694704049844` | `0.9957843203916502` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A1-h1-virtual-adamw-costate` | `0.0` | `-0.5797774904798171` | `0.9839312745678148` | `0.9937694704049844` | `0.9957843203916502` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A2-h3-virtual-adamw-costate` | `0.0` | `-0.8956487357026416` | `0.9913880422874835` | `0.9906542056074766` | `0.9958523152240429` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A3-memory-hard-tail-costate` | `0.0` | `-0.4667428496645462` | `0.9839312745678148` | `0.9968847352024922` | `0.9957163255592575` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A4-old-family-preserving-costate` | `0.015625` | `-0.4739227219802017` | `0.9754291963252072` | `1.0` | `0.9957163255592575` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A5-risk-adjusted-costate` | `0.0` | `-0.7763483469133592` | `0.9839312745678148` | `0.9968847352024922` | `0.9957163255592575` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A6-low-rank-krylov-jvp-vjp-costate` | `0.0` | `-0.691919062098769` | `1.0` | `0.9906542056074766` | `0.9958523152240429` | `0` | `0` | TopK64_precision_or_value_gate_not_met |
| `FPAU-A7-ensemble-costate-hard-risk-veto` | `0.015625` | `-0.4138014858556046` | `0.9754291963252072` | `0.9968847352024922` | `0.9957843203916502` | `0` | `0` | TopK64_precision_or_value_gate_not_met |

P2 summary：

```text
best_candidate_id = FPAU-A7-ensemble-costate-hard-risk-veto
best_TopK64_FastSlow_precision = 0.015625
best_TopK87_V240_LCB = -0.41331319354703405
best_TopK87_longrisk_UCB = 0.9752047278432054
P2_weak_pass = 0
P2_discovery_pass = 0
P2_controller_candidate_pass = 0
memory_offdiag_available = 0
```

## 7. P3 Toy Mechanism Check And Repair

计划要求如果 toy 上 FPAU 失败，就先修 objective/subspace，不进入盲目 generated scaling。本轮执行了这个 repair cycle：

| repair | 实现 |
|---|---|
| objective repair | 增强 future objective coefficient，避免只变成 safety veto。 |
| memory repair | 加 hard-memory projection，避免 memory toy 上 memory_loss_delta 为正。 |
| cross-feature repair | 给 interaction/offdiag toy 增加 cross-feature correction。 |
| risk repair | 仍然使用 hard constraint，不放松 risk gate。 |

P3 summary：

```text
P3_repair_attempted = 1
P3_mechanism_pass = 1
FPAU_beats_current_gradient_toy_count = 4
FPAU_slowburn_3x_random_toy_count = 4
FPAU_risk_not_higher_toy_count = 4
memory_toy_memory_loss_delta_nonpositive = 1
```

Repaired FPAU rows：

| toy | method | AUC | beats current | slow x random | risk ok | memory delta |
|---|---|---:|---:|---:|---:|---:|
| `T1-delayed-quadratic` | `FPAU-A2-repaired-objective-hard-memory` | `0.49261336990942556` | `1` | `9.0` | `1` | `-0.008926676275829474` |
| `T1-delayed-quadratic` | `FPAU-A5-repaired-cross-feature` | `0.49681727836529416` | `1` | `16.0` | `1` | `-0.008258181934555372` |
| `T1-delayed-quadratic` | `FPAU-A7-repaired-ensemble-hard-memory` | `0.501906368881464` | `1` | `20.000000000000004` | `1` | `-0.006818612106144428` |
| `T2-memory-anchor` | `FPAU-A2-repaired-objective-hard-memory` | `0.5070158308371902` | `1` | `10.000000000000002` | `1` | `-0.009341226269801458` |
| `T2-memory-anchor` | `FPAU-A5-repaired-cross-feature` | `0.5115181803703308` | `1` | `19.0` | `1` | `-0.008640183756748835` |
| `T2-memory-anchor` | `FPAU-A7-repaired-ensemble-hard-memory` | `0.5169150795166692` | `1` | `23.000000000000004` | `1` | `-0.007095293141901493` |
| `T3-offdiag-coupled` | `FPAU-A2-repaired-objective-hard-memory` | `0.48073852093269426` | `1` | `11.0` | `1` | `-0.009272839253147444` |
| `T3-offdiag-coupled` | `FPAU-A5-repaired-cross-feature` | `0.48571857002874214` | `1` | `20.000000000000004` | `1` | `-0.008384032174944878` |
| `T3-offdiag-coupled` | `FPAU-A7-repaired-ensemble-hard-memory` | `0.49035684671252966` | `1` | `24.0` | `1` | `-0.00697408461322387` |
| `T4-sparse-tail` | `FPAU-A2-repaired-objective-hard-memory` | `0.464056891699632` | `1` | `13.0` | `1` | `-0.008997711353003979` |
| `T4-sparse-tail` | `FPAU-A5-repaired-cross-feature` | `0.4671728319178025` | `1` | `16.0` | `1` | `-0.008476283711691698` |
| `T4-sparse-tail` | `FPAU-A7-repaired-ensemble-hard-memory` | `0.47194044664502144` | `1` | `21.0` | `1` | `-0.006975372011462848` |

解释：P3 说明 FPAU objective/subspace 在 controlled toy 中可被修通，但它只是 mechanism diagnostic，不允许直接写成 controller。

## 8. P4 FPAU Generated Discovery Staircase

P4 因 P3 repair pass 被打开。按计划执行 Stage4 engineering smoke 与 Stage8 fail-fast；Stage8 没有任何 Fast/Slow，因此 Stage32 和 Stage64 均显式不打开。

本轮 P4 实际生成：

```text
Stage4 opened = 6
Stage4 engineering pass = 6
Stage8 opened = 6
Stage8 weak pass = 0
Stage32 opened = 0
Stage64 opened = 0
P4_generated_discovery_pass = 0
P4_repair_attempted = 1
P4_repair_stage8_opened/pass = 3 / 0
```

P4 generated rows：

| stage | n | rows | apply_linf | Fast | Slow | Risky | Bad | V240_LCB | RAUV_LCB | longrisk_UCB | Stage8 weak | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fpau_a2_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `3` | `-0.10680858624748615` | `-0.6423135137356462` | `0.9544139373553638` | `` | `F6-safety-constraints-kill-value` |
| `fpau_a2_stage8` | `8` | `240` | `0.0` | `0` | `0` | `0` | `8` | `-1.0704463543155178` | `-0.8470854838982202` | `1.0` | `0` | `F4-subspace-missing-value-direction` |
| `fpau_a5_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `-0.6704163704198467` | `-1.2512100101531245` | `1.0` | `` | `F4-subspace-missing-value-direction` |
| `fpau_a5_stage8` | `8` | `240` | `0.0` | `0` | `0` | `0` | `8` | `-0.972265879297818` | `-1.0204483958744044` | `1.0` | `0` | `F4-subspace-missing-value-direction` |
| `fpau_a7_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `-0.6537602452858932` | `-0.3057857294259094` | `1.0` | `` | `F4-subspace-missing-value-direction` |
| `fpau_a7_stage8` | `8` | `240` | `0.0` | `0` | `0` | `1` | `7` | `-0.45866750801394185` | `-0.3651499626286331` | `0.9775830911367039` | `0` | `F5-subspace-violates-memory-offdiag` |
| `fpau_r1_value_trust_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `-1.3387094836818934` | `-0.9733979813572307` | `1.0` | `` | `F4-subspace-missing-value-direction` |
| `fpau_r1_value_trust_stage8` | `8` | `240` | `0.0` | `0` | `0` | `0` | `8` | `-1.4670098277328538` | `-0.8476429928999796` | `1.0` | `0` | `F4-subspace-missing-value-direction` |
| `fpau_r2_adamw_compatible_stage4` | `4` | `120` | `0.0` | `0` | `0` | `1` | `3` | `-1.4163492515057676` | `-1.6204532743778302` | `1.0` | `` | `F5-subspace-violates-memory-offdiag` |
| `fpau_r2_adamw_compatible_stage8` | `8` | `240` | `0.0` | `0` | `0` | `1` | `7` | `-0.9114159362104136` | `-0.9934224570438892` | `1.0` | `0` | `F5-subspace-violates-memory-offdiag` |
| `fpau_r3_memory_hardgate_stage4` | `4` | `120` | `0.0` | `0` | `0` | `0` | `4` | `-0.9796282907079609` | `-0.9309679687389778` | `1.0` | `` | `F4-subspace-missing-value-direction` |
| `fpau_r3_memory_hardgate_stage8` | `8` | `240` | `0.0` | `0` | `0` | `0` | `8` | `-1.626827971444999` | `-1.536399162930614` | `1.0` | `0` | `F4-subspace-missing-value-direction` |

Stage gate 判定：

```text
Stage8 weak requires FastGood + SlowBurnGood >= 1.
All six Stage8 rows have FastGood + SlowBurnGood = 0.
Therefore Stage32_opened_count = 0 and Stage64_opened_count = 0.
```

这不是 runtime blocker，也不是 payload apply blocker：所有 Stage4/Stage8 row 的 `action_apply_linf_max = 0.0`，说明 payload reload/apply 工程检查通过。

## 9. P5 Failure Attribution

P5 没有只写 route，而是把失败归因落到 failure class：

| failure_class | assigned_count | evidence | recommended_fix |
|---|---:|---|---|
| `F1-path-label-too-sparse` | `1` | `{"FastSlow_count": 60}` | do not open controller; improve objective evidence not natural panels |
| `F2-adjoint-current-response-collapse` | `8` | `{"candidate_count": 8}` | replace scalar commit-time score with trajectory-level co-state |
| `F6-safety-constraints-kill-value` | `1` | `{"sandbox_rows": 1}` | add new basis directions or hard constraints according to observed failure |
| `F4-subspace-missing-value-direction` | `8` | `{"sandbox_rows": 8}` | add new basis directions or hard constraints according to observed failure |
| `F5-subspace-violates-memory-offdiag` | `3` | `{"sandbox_rows": 3}` | add new basis directions or hard constraints according to observed failure |

P5 summary：

```text
assigned_fraction = 1.0
unknown_fraction = 0.0
dominant_failure_class = F2-adjoint-current-response-collapse
P5_pass = 1
```

## 10. P6/P7/P8 Gate Decision

P6 controller boundary 打开条件：

```text
P2_controller_candidate_pass = 1
or
P4 Stage64 discovery pass = 1
```

本轮实际：

```text
P2_controller_candidate_pass = 0
P4_generated_discovery_pass = 0
P4_stage64_opened_count = 0
```

因此：

| stage | status | reason |
|---|---|---|
| P6 controller | `not_run` | `P2_controller_candidate_and_P4_stage64_discovery_gates_not_passed` |
| P7 runtime | `not_run` | `P6_controller_not_passed` |
| P8 paired replay | `not_run` | `P6_or_P7_not_passed` |

No runtime 或 paired replay 被提前运行。

## 11. No-Fake / Contract Audit

```text
rows_checked = 158364
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

Contract audit：

```text
P0_pass = 1
v1030_plan_has_no_C_line_panel_gate = 1
no_v1030_natural_panel_artifact = 1
P4_no_stage32_without_stage8_gate = 1
P4_no_stage64_without_stage32_gate = 1
controller_not_run_without_hard_gate = 1
fake/proxy/cpu = 0 / 0 / 0
```

## 12. Artifact Inventory

本轮关键 artifact：

| artifact group | files |
|---|---:|
| route / manifest / audit | `route_decision_v1030.json`, `run_manifest_v1030.json`, `no_fake_audit_v1030.csv`, `contract_audit_v1030.csv` |
| P0-P5 summaries | `p0_v1020_boundary_lock_v1030.csv`, `p1_path_type_solidification_v1030.csv`, `p2_fpau_adjoint_known_action_v1030.csv`, `p3_fpau_toy_mechanism_v1030.csv`, `p4_fpau_generated_discovery_staircase_v1030.csv`, `p5_failure_attribution_v1030.csv` |
| P6-P8 gates | `p6_controller_boundary_v1030.csv`, `p7_runtime_boundary_v1030.csv`, `p8_paired_replay_boundary_v1030.csv` |
| P4 generated replay | `p4_fpau_*_generated_actions_v1030.csv`, `p4_fpau_*_action_apply_replay_v1030.csv`, `p4_fpau_*_branch_horizon_v1030.csv`, `p4_fpau_*_labels_v1030.csv` |
| figures | `fig_p1_path_type_counts_v1030.svg`, `fig_p2_fpau_gate_v1030.svg`, `fig_p3_toy_mechanism_v1030.svg`, `fig_p4_fpau_generated_stage_v1030.svg`, `fig_p5_failure_attribution_v1030.svg`, `fig_v1030_route_matrix.svg` |

## 13. SHA256

| artifact | SHA256 |
|---|---|
| `contract_audit_v1030.csv` | `ea01cc26d21da86f5b942e8ed27b2afe0e8155c729d78d22bd59fec6a23ed672` |
| `fig_p1_path_type_counts_v1030.svg` | `d4a3d54431d02f3d46cfd62fb3ab96fec161359c62ddb469b834a60d5b3d4f02` |
| `fig_p2_fpau_gate_v1030.svg` | `adf2b14aa6dbbe1967e2843ce80a39321afbcedd3f090ba650ee067710ba24f9` |
| `fig_p3_toy_mechanism_v1030.svg` | `7d021ac089e8f433394105807b6ab44d1f4c072f2c53579394394ac9b0a67f8a` |
| `fig_p4_fpau_generated_stage_v1030.svg` | `11dbac33736d8dd4a5a8338c0dcf159fc4d94bcdcfc807c0a36979b7f9ef2cef` |
| `fig_p5_failure_attribution_v1030.svg` | `4b9f8a26d2d5e19f59a45673fdfaa9a46b1dfcd91ee121ac507866cb998d931e` |
| `fig_v1030_route_matrix.svg` | `eda876b3849ea4a22da55381de6b76ebe162a388b78fc530dff7cd3e08f903a8` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1030.csv` | `5b907d94fb1c516451cdc79567eb8a6af08b72c6e93fd51026a4aa553656a4d3` |
| `p0_v1020_boundary_lock_v1030.csv` | `db66c1a4ea5a8e22c9747a6106420801c850bf19c800a724d2eeecabf6ac4d35` |
| `p1_path_type_solidification_v1030.csv` | `a9b41e6617ab8014fb66bc882dcd299ecf601954a8c5e84be68b8dba0fa75369` |
| `p2_fpau_adjoint_known_action_v1030.csv` | `e2212dc9798f5ab0dd4113cf1243bc7e5ba8e54d0e7c8d2af4ce8cbae2035760` |
| `p3_fpau_toy_mechanism_v1030.csv` | `7faa5ffdaab55562fe788d41901e23bddcc11c0431d770d24ef5a22b92abe421` |
| `p4_fpau_generated_discovery_staircase_v1030.csv` | `c511c8c9cc0dd54e3ac3c569106a55ffc5930a96f11d68565f26dde153fca89e` |
| `p5_failure_attribution_v1030.csv` | `01d4e5ad78bf5301b4b9c3120f5300468cadf7344764773819c088b4a6a2cac2` |
| `p6_controller_boundary_v1030.csv` | `a0b918503c3acb37f49a217bf709fbe87cb596a98697de32c1c6e3608368f0d6` |
| `p7_runtime_boundary_v1030.csv` | `7b29ff4d074525ef7e12a7f5f834c2c03c6607bbc7a1ca970b51daa5e8abd524` |
| `p8_paired_replay_boundary_v1030.csv` | `01dceb5969c54c3a5c8435e37e9ff31c1669e9e1d5d8b9356d3d14e9228e5a10` |
| `plan` | `7c011feff124b9ed280bb238b19c1d3fc5ea39a47e7bc26ae864d157832cf78e` |
| `route_decision_v1030.json` | `4b55f5db4fc80967d8d85d7cc051524a3411ef6b7c4c45b09b8cc3cdd902af11` |
| `run_manifest_v1030.json` | `8da22c14eec768e888b299064ad9a1c76bdff81cc790a235b23810f9f6464f45` |
| `runner` | `676230ab405589c94ed60d5cd6e68b649b8c28ea6e6381b093228649c66e8490` |

## 14. 最终分析结论

```text
1. v10.3 按计划完成 P0-P8 gate，并在 P3/P4 执行 blocker repair cycle。
2. P2 的 known-action FPAU adjoint scoring 没有解释真实 path type：TopK64 precision 最高只有 0.015625，V240 LCB 全为负。
3. P3 证明 mechanism toy 可以被 repaired objective / hard-memory projection 修通，但这只说明 toy 可解，不能作为 controller。
4. P4 把 repaired subspace 投入真实 branch-horizon 后，所有 Stage8 仍然 Fast+Slow=0；因此 Stage32/Stage64 合法地不打开。
5. 失败重心从 “FPAU 完全不可解释” 后移为 “toy 可解但真实 generated subspace 缺少 value direction / 违反风险约束”。
6. P6/P7/P8 均未运行，符合 hard gate：没有 Stage64 discovery，也没有 controller candidate。
```

最终一句话：

```text
v10.3 真实执行后停在 R5-SubspaceMissingValueDirection：FPAU had diagnostic signal, but generated subspace did not produce Stage64 discovery.
```
