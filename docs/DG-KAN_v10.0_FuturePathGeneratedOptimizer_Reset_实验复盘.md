# DG-KAN v10.0 FuturePath Generated Optimizer Reset 实验复盘

> 本复盘记录 `DG-KAN_v10.0_FuturePathGeneratedOptimizer_Reset_完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest、真实 G40 natural AP0 panel rows 与真实 generated discovery branch-horizon rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 controller/runtime/paired replay 均显式 `not_run`。

## 0. 最新结论

```text
route = CaseD-NaturalInsufficientBDDiscoveryFail
primary_blocker = natural_density_insufficient
secondary_blocker = FPO11_and_generated_discovery_failed
system_legal_controller_pass = 0
generated_route_status = stopped_future_path_generated_optimizer_reset
```

最终 artifact：`results/real_rerun_20260506/v1000_futurepath_generated_optimizer_reset_full_20260517T230000Z`

核心结论：

1. P0 复现 v9.9.8 boundary pass = `1`；source route = `CaseC-NaturalDensityInsufficientGeneratedSandboxFail`。
2. A 线 Fast/Slow/Risky/SafeLow/Bad = `3` / `11` / `121` / `9` / `4920`；discovery target pass = `0`。
3. B 线 FPO11 weak/discovery = `0` / `0`；best = `FPO11C-signal-reservoir-path-type-snr`，precision = `0.011494252873563218`，V240 LCB = `-0.6468261963476977`。
4. C 线 completed panels = `3`，largest = `10000`，natural density sufficient/insufficient/inconclusive = `0` / `1` / `0`。
5. D 线 D0/D1/D2 = `1` / `1` / `1`；D2 weak = `0`，FastSlow precision = `0.0`。
6. P7 controller = `not_run`；No-fake audit rows checked = `504261`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v1000_futurepath_generated_optimizer_reset.py` | v10.0 runner；执行 P0/A/B/C/D/P7-P9 gate，并写 manifest/recap。 |
| `experiments/natural_ap0_extension_materializer.py` | 复用真实 natural/generated payload materializer 与 branch-horizon replay。 |

```text
python -m py_compile experiments/run_v1000_futurepath_generated_optimizer_reset.py
```

```bash
python experiments/run_v1000_futurepath_generated_optimizer_reset.py --out-dir results/real_rerun_20260506/v1000_futurepath_generated_optimizer_reset_full_20260517T230000Z --fresh --device auto --data-root data --seed 1314 --repeat-seed 2027 --confirm-seed 3031 --execution-profile full-gated --chunk-actions 512
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1000",
  "status": "summary",
  "route": "CaseD-NaturalInsufficientBDDiscoveryFail",
  "primary_blocker": "natural_density_insufficient",
  "secondary_blocker": "FPO11_and_generated_discovery_failed",
  "route_explanation": "Natural density was confirmed insufficient, but FPO11 and generated discovery both failed, so controller/runtime stay blocked.",
  "source_route_v9980": "CaseC-NaturalDensityInsufficientGeneratedSandboxFail",
  "P0_boundary_pass": 1,
  "A_discovery_target_pass": 0,
  "B_FPO11_weak_pass": 0,
  "B_FPO11_discovery_pass": 0,
  "C_natural_density_sufficient": 0,
  "C_natural_density_insufficient": 1,
  "C_natural_density_inconclusive": 0,
  "C_largest_completed_panel_size": 10000,
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
B FPO11 weak/discovery = 0 / 0
C density sufficient/insufficient/inconclusive = 0 / 1 / 0
D generated discovery pass = 0
```

## 4. No-Fake / Hash

```text
rows_checked = 504261
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `a_future_path_type_labels_v1000.csv` | `1495c94c5aaffef465c3ac4cc6380d639e668ae1ef8807cad7457bf3fb69927e` |
| `b_fpo11_path_type_predictor_v1000.csv` | `9dfb8490228434b3424f145f25f2b8c23968f937161735e350744adb6e6e48a8` |
| `c_natural_density_confirmation_v1000.csv` | `393e178fcd9ae0fc81383acbb7c2fedb72f38ddec334616b2127bc6c528a287e` |
| `c_natural_density_group_rates_v1000.csv` | `bd60253f306d651f0a6e031416f9e64f0c5267a047f87fac592e7535e7617ee9` |
| `contract_audit_v1000.csv` | `e293acd698159368a18adad2aa4049fb58c7b4a4168fd47fe5f7b0ce3f266013` |
| `d_generated_discovery_staircase_v1000.csv` | `16ef297c27b9487711fa332e2949aa65f2bd9ca6341679eaacb632e8b3e2f8ea` |
| `failure_taxonomy_v1000.csv` | `a3afd73fd49cfde3828582edbc26a4ce9d59b1bf6d9db50edd1db8c5c110418a` |
| `fig_A_path_type_counts_v1000.svg` | `67e88de2c6ad0977592001900eba4ea2dbefdb673c6e2fcfd5b94965573cc1df` |
| `fig_B_fpo11_gate_v1000.svg` | `5b3e59aaeb9a85aa0704a681d12569212a92fc827c1731662adacd091b03fb43` |
| `fig_C_density_repeat_curve_v1000.svg` | `0e7769c494cec23bf3d3892e2b7032ecb6e6dee1ab4728da51f5563389397300` |
| `fig_D_generated_discovery_v1000.svg` | `d82e7e834abe2b07e3e636ebb7a1bf8a938f799be8c4121afb918d04a0d7bbbb` |
| `fig_v1000_route_matrix.svg` | `b4afefb326ccae9150cf48ec4920d4ec21537e56680b5e1b93f58e058fae2c03` |
| `materializer` | `7567bf4f41c21f0d8641d827650c94a9dada1e5a28180218f648a9dea6d99816` |
| `no_fake_audit_v1000.csv` | `1d9d801efaab5feec83437d4dc38b2c7fcde7f025df0b1bead21a2e840fb19e4` |
| `p0_v9980_boundary_reproduction_v1000.csv` | `4332b844a6a4806e5bb46249a19a36d0fcbcc51445cfb9106adeb8bb0b4a1df4` |
| `p7_controller_boundary_v1000.csv` | `c531e64643113dda5ff39c1c1d3e9d3b1915b570e7fd7ae7889cbe8c21e27d00` |
| `p8_runtime_boundary_v1000.csv` | `c38d30c927b7e52a5a6a3e56c96605e6373c30d4af111d69678d6fb08e2a0267` |
| `p9_paired_replay_boundary_v1000.csv` | `21fea82cbaf5a128d55df65b860b44fd72b728beca755105acc735c2cba9cca3` |
| `plan` | `607be2ff7e3aa183c807d939de43c3d969fa4f21a7e21b58caf72044477552f7` |
| `route_decision_v1000.json` | `dc2ba1f6d88343932161a832781d70c4b6c4abe5fad8ae57982ef537dbca5b71` |
| `run_manifest_v1000.json` | `b9d8101dd2ce89111e1bbed0fde00f837ab3889e18605f46ab4888f314bfc2d8` |
| `runner` | `660519c179ef8b9a76fcda17b301b7ed4c21ad64b238ef744f5b881fa2c5285e` |

## 5. 最终分析结论

```text
1. v10.0 将 official gate、science discovery 和 engineering preflight 分开记录，没有把 discovery label 写成 official controller。
2. C 线按计划复用 v9.9.8 真实 5000，并新增 seed repeat；只有满足两个 5000 条件才打开 10000 confirmation。
3. B 线 FPO11 使用 commit-time features 做 path-type predictor；yellow calibrated sketch 只作为 discovery 诊断，不写成 official。
4. D 线只执行 8 -> 32 -> 64 discovery staircase；D2 未过 weak 时不打开 256。
5. controller/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。
```

最终一句话：v10.0 真实执行后停在 `CaseD-NaturalInsufficientBDDiscoveryFail`：Natural density was confirmed insufficient, but FPO11 and generated discovery both failed, so controller/runtime stay blocked.
