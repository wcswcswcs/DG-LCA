# DG-KAN v10.2 Update Rule Discovery Reset 实验复盘

> 本复盘记录 `DG-KAN_v10.1_结果解读与v10.2_完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.1 真实 artifact 与本轮真实 GUP13 generated branch-horizon rows；没有 fake data、proxy rows、占位数据或 CPU offload。C3 20000 继续按 cost cap 显式 `not_run`。

## 0. 最新结论

```text
route = R4-BDDiscoveryFail_UpdateRuleTheoryRebuild
primary_blocker = natural_density_insufficient
secondary_blocker = FPO13_and_GUP13_generated_discovery_failed
system_legal_controller_pass = 0
generated_route_status = stopped_update_rule_discovery_reset
```

最终 artifact：`results/real_rerun_20260506/v1020_update_rule_discovery_reset_full_20260518T190000Z`

核心结论：

1. P0 v10.1 boundary lock pass = `1`；source route = `R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset`；C3 status = `not_run`，C3 cost cap = `1`。
2. P1 Fast/Slow/Risky/SafeLow/Bad = `12` / `48` / `842` / `60` / `34286`；P1 discovery pass = `0`。
3. P2 FPO13 weak/discovery/controller = `0` / `0` / `0`；best = `FPO13D-memory-offdiag-hardgate-delayed-gain-proxy`，TopK64 precision = `0.015625`，V240 LCB = `-0.3264349469551365`。
4. P3 natural density close/retain/inconclusive = `1` / `0` / `0`；new panel run = `0`；C3 20000 run = `0`。
5. P4 GUP13 Stage8/32/64 opened = `8` / `0` / `0`；generated discovery pass = `0`；best = `gup13c` `gup13c_stage8`。
6. P5 assigned fraction = `1.0`，dominant failure = `F2-FPO-measures-current-response-not-future-path`。
7. P6/P7/P8 = `not_run` / `not_run` / `not_run`；No-fake audit rows checked = `37410`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1020_update_rule_discovery_reset.py
```

```bash
python experiments/run_v1020_update_rule_discovery_reset.py --out-dir results/real_rerun_20260506/v1020_update_rule_discovery_reset_full_20260518T190000Z --fresh --device auto --data-root data --seed 1414 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1020",
  "status": "summary",
  "route": "R4-BDDiscoveryFail_UpdateRuleTheoryRebuild",
  "primary_blocker": "natural_density_insufficient",
  "secondary_blocker": "FPO13_and_GUP13_generated_discovery_failed",
  "route_explanation": "Natural harvesting stays downgraded, and both FPO13 and GUP13 discovery failed; v10.3 should rebuild update-rule theory.",
  "source_route_v1010": "R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset",
  "P0_boundary_pass": 1,
  "P1_discovery_pass": 0,
  "P2_weak_pass": 0,
  "P2_discovery_pass": 0,
  "P2_controller_candidate_pass": 0,
  "P3_natural_density_close_mainline": 1,
  "P3_natural_density_retain_mainline": 0,
  "P3_C3_20000_run": 0,
  "P4_generated_discovery_pass": 0,
  "P4_stage64_opened_count": 0,
  "P5_pass": 1,
  "P6_controller_pass": 0,
  "P7_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "generated_route_status": "stopped_update_rule_discovery_reset",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A/B/C/D 摘要

```text
P1 discovery pass = 0
P2 FPO13 discovery pass = 0
P3 close natural mainline = 1
P4 generated discovery pass = 0
```

## 4. No-Fake / Hash

```text
rows_checked = 37410
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `contract_audit_v1020.csv` | `a3f163a4f14c4202336e2c30526b58c72b762405bc186f93843bdf1205b5b55c` |
| `fig_p1_path_type_counts_v1020.svg` | `d4a3d54431d02f3d46cfd62fb3ab96fec161359c62ddb469b834a60d5b3d4f02` |
| `fig_p2_fpo13_gate_v1020.svg` | `61e346a32fa148f43c86bb6369b23998ad55c05ee9c9ed41299cf4397b5c0dc1` |
| `fig_p3_density_ci_v1020.svg` | `15232f5f2bf4539f40a4e6535423f5d866491f64db859507413bda38cfc02917` |
| `fig_p4_generated_stage_v1020.svg` | `beb8d358152b15d5fcc53df8f4edb01e27b6b31f6547d45fbcccaf875c901ae1` |
| `fig_p5_failure_taxonomy_v1020.svg` | `4b9f8a26d2d5e19f59a45673fdfaa9a46b1dfcd91ee121ac507866cb998d931e` |
| `fig_v1020_route_matrix.svg` | `eda876b3849ea4a22da55381de6b76ebe162a388b78fc530dff7cd3e08f903a8` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1020.csv` | `3d884cfde140ffeb1b07590bc9eb7e380ab57a07c350cb9b81a65cb521730883` |
| `p0_v1010_boundary_lock_v1020.csv` | `42cdd6f1f82e1514d3898bd009b884446b800f044d293833a26373d674f26e87` |
| `p1_path_type_mechanism_v1020.csv` | `52cee5d524e197db9efe957ef86388ce2a31ec1d241c7c90e609bd99f0785211` |
| `p2_fpo13_path_type_predictor_v1020.csv` | `793de3d678bda4ddcc8aa2843e078a31956ef03a81e1353e6581ac403b906842` |
| `p3_natural_density_finite_confirmation_v1020.csv` | `b3e495d3fc11c577d6a7723402c35db4b8aeab6f10c55b75d88d01d66efd36c3` |
| `p4_gup13_generated_discovery_staircase_v1020.csv` | `735f64e577f784463baf397c27a53a2c140c85c9a00d980d85c1d02f65a3385e` |
| `p5_failure_taxonomy_v1020.csv` | `f9ecdd991ac12598ab86354ca55a0a29905cc6fe4c3b5d577c8b2f7766f4f1e2` |
| `p6_controller_boundary_v1020.csv` | `d26049c971f576c8cfc6a04982e7b659caad6863f5f021b1b3fe5398dec9b80f` |
| `p7_runtime_boundary_v1020.csv` | `28dd0d29a72e0652f9662b2fb7b1e4eb7158392be191399a24c215544498e433` |
| `p8_paired_replay_boundary_v1020.csv` | `4a5777c48f0bcb713d58f17a09fe1abf3d02bd65eae4bc8ec5b0418c0289929c` |
| `plan` | `1758986d3a9c54e7cad400a5fdbdc01b35a99340b152e1b792b237ec07ea9c63` |
| `route_decision_v1020.json` | `7dfc5b568a1763e313ceed2b80c636d4d0968bd3f7fce873d12ed38597e60cd7` |
| `run_manifest_v1020.json` | `ef9a65e356fee180a908f0e0b31d63b9bffd9f9f47c0a1f0a39befbc45fc153b` |
| `runner` | `5c7f7926cd82415f5ed46a0a535f5a1461260b3c64191cda371b77562b102ee1` |

## 5. 最终分析结论

```text
1. v10.2 复现并锁定 v10.1 的 terminal boundary，未把 discovery label 写成 official controller。
2. P3 未新增 natural panel，明确复用 v10.1 真实 5000/10000 作为有限确认；C3 20000 继续不运行。
3. FPO13 仍未达到 discovery/controller candidate gate，negative control 未被写成成功。
4. GUP13 generated discovery 只按 8 -> 32 -> 64 gate 打开；未满足 gate 的 stage 均显式 not_run。
5. P6/P7/P8 仍被硬门阻止，未运行 runtime 或 paired replay。
```

最终一句话：v10.2 真实执行后停在 `R4-BDDiscoveryFail_UpdateRuleTheoryRebuild`：Natural harvesting stays downgraded, and both FPO13 and GUP13 discovery failed; v10.3 should rebuild update-rule theory.
