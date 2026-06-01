# DG-KAN v9.8.1 Future-Trajectory Geometry-Adaptive Optimizer 实验复盘

> 本复盘记录 `DG-KAN_v9.8.1_FutureTrajectoryGeometryAdaptiveOptimizer_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 h1/h5 unavailable、natural stream extension blocker、future outcome diagnostic、geometry diagnostic、generated preflight blocker 或 not_run controller/runtime 写成 official system pass。

## 0. 最新结论

```text
route = R4-ArchitectureAgnosticGeometryFail
primary_blocker = geometry_score_low_precision_high_risk
secondary_blocker = natural_labeled_stream_extension_missing
system_legal_controller_pass = 0
generated_route_status = stopped_no_future_trajectory_mechanism_and_no_official_branch_horizon
```

最终 artifact：`results/real_rerun_20260506/v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z`

核心结论：

1. P0 复现 v9.8.0 boundary：source route = `RouteD-FutureTrajectoryUnsupported`，system pass = `0`，generated route = `stopped_no_official_generated_branch_horizon_pass`，field red count = `0`。
2. P1 natural stream 只完成现有 labeled AP0 panel：existing labeled actions = `2876`，completed panel = `1`，not-run panel = `3`。
3. P1 Panel A CoreLike count/rate/LCB/UCB = `77` / `0.026773296244784424` / `0.021475240123524635` / `0.033333885489495195`。
4. P1 extension blocker = `extension_panels_blocked_by_missing_labeled_materializer`；没有为 5000/10000/20000 panel 生成假 labeled rows。
5. P2 full path 未打开：requested horizons = `1,5,20,80,240`，existing-action materialized horizons = `20,80,240`，horizon unavailable count = `14`。
6. P2 observed h20/h80/h240 AUV LCB：Core77 = `2.7546595085012537`，OldOnly = `2.36612274912119`，ExactOnly = `1.4622620508289148`，RandomMatched = `1.4778647567964251`。
7. P2 V20 LCB：Core77 = `1.573549156031694`，OldOnly = `1.2768640608784525`，ExactOnly = `0.49058574242981423`。
8. P3 mechanism pass count = `0`；best mechanism = `M4-CleanCoreLikeLabel`，precision = `0.8850574712643678`，AUV LCB = `2.6816104392609805`。
9. P4 geometry score 继续失败：TopK87 precision = `0.06896551724137931`，V LCB = `-0.9348850937038008`，longrisk UCB = `0.8485412092119363`，LDO = `0.011494252873563218`。
10. P5 existing-action controller = `not_run`，reason = `P1_or_P2_or_P3_or_P4_not_official_controller_ready`。
11. P6 generated preflight v2 = `not_run`，reason = `P2_P3_future_trajectory_mechanism_not_established_generated_route_stopped`。
12. P8 raw/support gate resolution 没有允许修改 official gate：E_raw/E_no_backfill/E_density_required = `0` / `0` / `0`。
13. P9/P10/P11 均 gate-blocked：runtime `not_run`，paired replay `not_run`，short/full `not_run`。
14. No-fake audit：rows checked = `2944`，fake/proxy/cpu = `0` / `0` / `0`。
15. 当前 primary blocker：`geometry_score_low_precision_high_risk`；secondary blocker：`natural_labeled_stream_extension_missing`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py` | v9.8.1 runner；复现 v9.8.0 boundary，执行 natural stream extension audit、future trajectory full-path audit、mechanism decomposition、architecture-agnostic geometry v2、existing-action / generated / LDO gate boundary，并写 manifest / audits / failure taxonomy。 |

代码检查：

```text
python -m py_compile experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py
```

smoke 预检：

```bash
python experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py \
  --out-dir results/real_rerun_20260506/tmp_v9810_future_trajectory_geometry_smoke \
  --fresh --device auto --data-root data --seed 1314 \
  --execution-profile smoke --panel-targets 64,128 --generated-per-family 4
```

smoke 输出：

```json
{
  "out_dir": "results/real_rerun_20260506/tmp_v9810_future_trajectory_geometry_smoke",
  "p1_density_strong_pass": 0,
  "p2_future_trajectory_full_path_pass": 0,
  "p3_mechanism_pass": 0,
  "p4_geometry_pass": 0,
  "primary_blocker": "geometry_score_low_precision_high_risk",
  "route": "R4-ArchitectureAgnosticGeometryFail",
  "secondary_blocker": "none",
  "system_legal_controller_pass": 0
}
```

正式运行：

```bash
python experiments/run_v9810_future_trajectory_geometry_adaptive_optimizer.py --out-dir results/real_rerun_20260506/v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000 --generated-per-family 64
```

正式 full-gated 输出：

```json
{
  "out_dir": "results/real_rerun_20260506/v9810_future_trajectory_geometry_adaptive_optimizer_full_20260516T100000Z",
  "p1_density_strong_pass": 0,
  "p2_future_trajectory_full_path_pass": 0,
  "p3_mechanism_pass": 0,
  "p4_geometry_pass": 0,
  "primary_blocker": "geometry_score_low_precision_high_risk",
  "route": "R4-ArchitectureAgnosticGeometryFail",
  "secondary_blocker": "natural_labeled_stream_extension_missing",
  "system_legal_controller_pass": 0
}
```

说明：本轮不是只跑 smoke。最终结论只引用 full-gated 正式目录。GPU 显存为 `0` 是 gate 结果：P2 full future path、P3 mechanism、P4 geometry 都没有 official pass，因此 P6/P7 generated branch-horizon 按计划 `not_run`，没有打开 CUDA generated route；这不是 CPU offload，也不是跳过正式运行。

## 2. Route

`route_decision_v9810.json` 摘要：

```json
{
  "stage": "ROUTE_DECISION_V9810",
  "status": "summary",
  "route": "R4-ArchitectureAgnosticGeometryFail",
  "route_family": "geometry",
  "primary_blocker": "geometry_score_low_precision_high_risk",
  "secondary_blocker": "natural_labeled_stream_extension_missing",
  "source_route_v9800": "RouteD-FutureTrajectoryUnsupported",
  "P0_boundary_reproduced": 1,
  "P1_natural_stream_extension_completed": 0,
  "P1_density_strong_pass": 0,
  "P1_density_weak_pass": 0,
  "P2_future_trajectory_full_path_pass": 0,
  "P3_mechanism_pass": 0,
  "P4_architecture_agnostic_geometry_pass": 0,
  "P5_controller_pass": 0,
  "P6_generated_preflight_v2_pass": 0,
  "P7_generated_branch_horizon_pass": 0,
  "P8_gate_modification_allowed": 0,
  "P9_runtime_pass": 0,
  "P10_paired_replay_pass": 0,
  "P11_short_full_pass": 0,
  "system_legal_controller_pass": 0,
  "generated_route_status": "stopped_no_future_trajectory_mechanism_and_no_official_branch_horizon",
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

判断：本轮停在 route 层，不进入 controller/runtime/system。原因不是单个数值阈值没调好，而是完整 future path 的 h1/h5 rows、natural labeled stream extension、legal mechanism 和 generated branch-horizon evidence 都没有同时成立。

## 3. P0 Boundary

Artifacts：

```text
p0_boundary_reproduction_v9810.csv
p0_field_legality_audit_v9810.csv
```

Summary：

```text
source_route_v9800 = RouteD-FutureTrajectoryUnsupported
source_primary_blocker_v9800 = future_trajectory_theory_not_supported
system_legal_controller_pass_v9800 = 0
generated_route_status_v9800 = stopped_no_official_generated_branch_horizon_pass
field green/yellow/red = 14 / 4 / 0
boundary_reproduced = 1
```

判断：P0 pass。v9.8.1 没有跳过 v9.8.0 的 no-system / generated-stop / no-fake boundary。

## 4. P1 Natural AP0 Labeled Stream Extension

Artifacts：

```text
p1_natural_ap0_stream_extension_materializer_v9810.csv
p1_density_curve_v9810.csv
```

Panel summary：

| panel | target | status | action count | CoreLike count | CoreLike rate | LCB | UCB | reason |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Panel-target-2876` | `2876` | `panel_row` | `2876` | `77` | `0.026773296244784424` | `0.021475240123524635` | `0.033333885489495195` | `` |
| `Panel-target-5000` | `5000` | `not_run` | `2876` | `` | `` | `` | `` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9810` |
| `Panel-target-10000` | `10000` | `not_run` | `2876` | `` | `` | `` | `` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9810` |
| `Panel-target-20000` | `20000` | `not_run` | `2876` | `` | `` | `` | `` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9810` |

判断：P1 未完成 natural labeled extension。现有 2876 行能复核 density，但 5000/10000/20000 没有 landed labeled materializer，不能声称 density closure。

## 5. P2 Future Trajectory Full Path

Artifacts：

```text
p2_future_trajectory_full_path_v9810.csv
p2_group_trajectory_summary_v9810.csv
```

Group summary：

| group | materialized horizons | missing horizons | full path | observed AUV LCB | h20 LCB | h80 LCB | h240 LCB | h240 longrisk UCB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `Core77` | `20,80,240` | `1,5` | `0` | `2.7546595085012537` | `1.573549156031694` | `3.1200217020467886` | `4.750885696264312` | `0.0` |
| `OldOnly` | `20,80,240` | `1,5` | `0` | `2.36612274912119` | `1.2768640608784525` | `2.6511661678501364` | `4.171856278488452` | `0.0` |
| `ExactOnly` | `20,80,240` | `1,5` | `0` | `1.4622620508289148` | `0.49058574242981423` | `1.5634505908603344` | `2.901934793211203` | `0.0` |
| `RandomMatched` | `20,80,240` | `1,5` | `0` | `1.4778647567964251` | `0.41123129387330626` | `1.4509081003016129` | `3.1326150107920494` | `0.8684269299978762` |
| `E4Expansion10` | `20,80,240` | `1,5` | `0` | `2.7207160659209277` | `1.544542302743137` | `3.0933350204937695` | `4.688599946956708` | `0.0` |
| `V9800GeneratedPreflight` | `20` | `1,5,80,240` | `0` | `0.0397442975740636` | `0.1589771902962544` | `` | `` | `` |

判断：h20/h80/h240 endpoint 信号仍然真实存在，但 h1/h5 缺失，所以不能把 endpoint-only 结果升级成 full future path theory pass。

## 6. P3 Mechanism Decomposition

Artifact：

```text
p3_future_trajectory_mechanism_decomposition_v9810.csv
```

| mechanism | effect size | corr/AUC proxy | TopK87 precision | TopK87 AUV LCB | longrisk UCB | legal | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M1-OldRankScore` | `0.5269648622934627` | `0.05694202466901287` | `0.8735632183908046` | `2.7378917778324734` | `0.0` | `0` | `0` |
| `M2-ExactTransferScore` | `0.19476308538251055` | `-0.0010127200150605285` | `0.20689655172413793` | `2.0181662996752396` | `0.0` | `0` | `0` |
| `M3-ImmediateValue` | `0.7940361750431921` | `-0.006217546927890931` | `0.4942528735632184` | `3.3202712196354365` | `0.5053405200342863` | `0` | `0` |
| `M4-CleanCoreLikeLabel` | `0.4909205695817936` | `0.08822453128538012` | `0.8850574712643678` | `2.6816104392609805` | `0.1674432324061406` | `0` | `0` |
| `M5-v9800FunctionResponseDiagnostic` | `2.9632252555420653` | `0.8031619087650855` | `0.06896551724137931` | `7.443207887129655` | `0.8485412092119363` | `0` | `0` |
| `M6-MemorySafeInverse` | `-0.07297765809498276` | `-0.01915307430900944` | `0.10344827586206896` | `1.6920562547123403` | `0.0` | `1` | `0` |

判断：P3 没有找到 legal + leaveout-stable + full-path-supported 的机制。部分 diagnostic score 有 AUV 相关性，但不能进入 controller。

## 7. P4 Architecture-Agnostic Function Geometry v2

Artifacts：

```text
p4_architecture_agnostic_function_geometry_ledger_v9810.csv
p4_geometry_score_evaluation_v9810.csv
```

Summary：

```text
geometry_action_row_count = 2876
TopK87_precision = 0.06896551724137931
TopK87_V_LCB = -0.9348850937038008
TopK87_longrisk_UCB = 0.8485412092119363
TopK87_bad_UCB = 0.07282499698983116
TopK87_null_UCB = 0.03389313880385762
TopK87_LDO/LSO/LTO = 0.011494252873563218 / 0.011494252873563218 / 0.011494252873563218
P4_architecture_agnostic_geometry_pass = None
```

判断：P4 仍是清晰失败。score 能找到低 LDO 区域，但 TopK87 precision 很低、V 为负、longrisk 很高，因此不是 controller-ready geometry principle。

## 8. P5-P7 Controller / Generated Boundary

P5：

```text
status = not_run
reason = P1_or_P2_or_P3_or_P4_not_official_controller_ready
controller_pass = 0
```

P6：

```text
status = not_run
reason = P2_P3_future_trajectory_mechanism_not_established_generated_route_stopped
P6_generated_preflight_v2_pass = 0
```

P7：

```text
status = not_run
reason = P6_generated_preflight_v2_not_passed
```

判断：generated route 没有因为 v9.8.0 h20 正信号而盲目重开。P2/P3 没给出完整 mechanism evidence，因此 P6/P7 按 gate 停止。

## 9. P8 Raw LDO vs Support-Aware Gate Resolution

Artifact：

```text
p8_ldo_gate_resolution_v9810.csv
```

Summary：

```text
candidate_id = E4-DatasetBlindScoreNormalizedTop10
accepted_count = 87
precision = 0.8850574712643678
V_LCB = 0.14120469391749882
raw_LDO = 0.39080459770114945
no_backfill_LDO = 0.07373671654738667
density_fail_count = 3
mean_backfill_precision = 0.009523809523809523
E_raw/E_no_backfill/E_density_required = 0 / 0 / 0
official_gate_modification_discussion_allowed = 0
```

判断：P8 没有解除 raw/support gate conflict。即使 no-backfill 角度可以解释一部分 backfill 惩罚，density-required 和 natural stream density 仍不过，因此 raw official gate 不能修改。

## 10. P9-P11 Runtime / Replay / Short-Full

```text
P9 status = not_run, reason = P5_controller_and_P7_generated_not_passed
P10 status = not_run, reason = P9_runtime_not_passed
P11 status = not_run, reason = P10_paired_replay_not_passed
```

判断：没有 official controller 或 generated update，所以 runtime、paired replay、short/full 全部保持关闭。

## 11. No-Fake / Contract / Failure

No-fake audit：

```text
rows_checked = 2944
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake/no_proxy = 1 / 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
v9800_boundary_pass = 1
natural_stream_extension_completed = 0
future_path_full_pass = 0
mechanism/controller/runtime/system = 0/0/0/0
uses_future_outcome_for_official_controller = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure taxonomy：

```text
route = R4-ArchitectureAgnosticGeometryFail
F0_boundary_fail = 0
F1_natural_stream_extension_missing = 1
F2_h1_h5_future_path_missing = 1
F3_mechanism_absent = 1
F4_geometry_score_fail = 1
F5_controller_blocked = 1
F6_generated_route_stopped = 1
F7_runtime_replay_shortfull_blocked = 1
primary_blocker = geometry_score_low_precision_high_risk
secondary_blocker = natural_labeled_stream_extension_missing
```

## 12. Figures

本轮落盘图：

```text
fig_p1_corelike_density_curve_v9810.svg
fig_p2_auv_observed_by_group_v9810.svg
fig_p2_v20_by_group_v9810.svg
fig_p3_mechanism_precision_v9810.svg
fig_p4_geometry_topk_quality_v9800.svg
```

这些图只用于复核诊断，不构成 official pass。

## 13. Hash

| artifact | SHA256 |
|---|---|
| `base_acc_sentinel_v9810.csv` | `0ee2c7c39710bff75d1695c8e69c3eaade0f89fff3f51158a72791c09513fa7a` |
| `contract_audit_v9810.csv` | `bc061d375fb3f6e57fe29bdbae2ed54084990246fa4ec2f45bfeb7848440b2cd` |
| `failure_taxonomy_v9810.csv` | `4a222f40fe357b7253adaad917e0c14dbf2d6cba656ccdd13888dd233d5745ce` |
| `no_fake_audit_v9810.csv` | `64e860433be95e5aa1055fa65026f69ba5635935997ac9cb891e678a2c989e32` |
| `p0_boundary_reproduction_v9810.csv` | `1da594a7a60d56428875d22f5a425ee8e814a51a986cd04f7acf36298e47cae9` |
| `p0_field_legality_audit_v9810.csv` | `9791f3c46ff8c21b713d8343ff8280efb1295c49da4a928eacf01058fff03f3c` |
| `p10_paired_replay_boundary_v9810.csv` | `7149a5fc6c9dd6eea9595b1873b25e7a44b4839251e6e65ec1b0f6ca3fae0258` |
| `p11_short_full_boundary_v9810.csv` | `181c4d0432bf89efcaa8991d22f3c3d8a237bab144053f59a66e4435a17bd626` |
| `p1_density_curve_v9810.csv` | `b091c91f323750dc7dadb4961d01cffdf0f8ffdbf94d1fe9bbe5625b75dd6058` |
| `p1_natural_ap0_stream_extension_materializer_v9810.csv` | `d4676a10353fac98e067e5620cf0825a7df553ad74b2948108fb979cab530f3c` |
| `p2_future_trajectory_full_path_v9810.csv` | `a873aa5663a3fb2e5cb1a59230ed356a321a16e3563f31e5698ddc9b6f664830` |
| `p2_group_trajectory_summary_v9810.csv` | `5c365eef00c395e049de7f756877bb9c2479fd79d0c06a5c44c73708a0ab95b0` |
| `p3_future_trajectory_mechanism_decomposition_v9810.csv` | `01706e36e973cd2bd8b51e3f57e8f998668d2f019968423d4e3bab3e66489181` |
| `p4_architecture_agnostic_function_geometry_ledger_v9810.csv` | `66da3a9c67dd817ffd8b6369a2ac26bdf4b8d1163fd96caf0bb80015f2f432c9` |
| `p4_geometry_score_evaluation_v9810.csv` | `d11bcb8de9b07ed6eff5c06f0ce8727b71a55251e5522d4ac9b748ff67c64637` |
| `p5_existing_action_minimal_controller_boundary_v9810.csv` | `a0a392f80f4321a97f1faaa5a44e557e29aa12c6b05be66584b484d5e78befbb` |
| `p6_generated_update_preflight_v2_v9810.csv` | `24f1771a89b3a3eb7f498feaccc4a485dadb6a067fe8b8782cac3df9a10c0d0d` |
| `p7_generated_branch_horizon_smoke_v9810.csv` | `33af7de7dbbf2c8271d2680d5904ad24b60009b01fbfe7c168ccd256393e3111` |
| `p8_ldo_gate_resolution_v9810.csv` | `307a03b414b95191b66e5419d4c0fe77e3ac88e9a2c01ccceb57f4fbc136b3bb` |
| `p9_runtime_boundary_v9810.csv` | `e775648a14876110ba723a1a9ee624cb2082696fade9b372fecdc4f2aa6c4c4e` |
| `plan` | `206065633fa4646fb84835ef8b0151437aade1c4ed6eae124c5f7ec37bdf18e3` |
| `route_decision_v9810.json` | `d8605180e0621c33d7be062ab1ab4ff56347794b3ff76027d0eabab94da0fc66` |
| `run_manifest_v9810.json` | `30f383f86fa720c5b644203f529a284a085bce52aa3f7f1de1b691a532237c64` |
| `runner` | `0c086f475c7b59486f419c85d3c000e01417e965b2aa8045a212b0ced727217c` |

## 14. 最终分析结论

v9.8.1 的真实推进是：

```text
v9.8.0:
  future trajectory 有 h20/h80/h240 endpoint 信号；
  但 natural stream extension、完整 h1/h5 path、geometry score、generated branch-horizon 都没打开。

v9.8.1:
  复现 v9.8.0 boundary；
  明确确认仓库当前没有 landed h1/h5 full-path materializer；
  对 h20/h80/h240 做 observed AUV 诊断，但不写成 full-path pass；
  机制分解没有找到 legal + leaveout-stable 的 controller-ready feature；
  architecture-agnostic geometry v2 仍选择低质高风险区域；
  generated route 因缺少机制证据和 long-horizon official rows 继续停止；
  raw/support gate conflict 没有被解除。
```

机制判断：

1. H1 未能 official 成立：Core77 / OldOnly endpoint 很强，但 h1/h5 path 缺失，不能声称完整 future trajectory theory pass。
2. H2 未成立：当前 architecture-agnostic function geometry score 仍没有对准 value/risk，不能作为跨架构 optimizer principle。
3. H3 未打开：natural labeled stream extension 缺 materializer，density closure 仍未回答。
4. H4 未打开：generated route 没有 P6/P7 长程 branch-horizon evidence，不能凭 v9.8.0 h20 positive smoke 重开。
5. P8 gate conflict 仍存在：density-required 不成立，raw official gate 继续保留。
6. P9-P11 全部 gate-blocked：没有 controller/runtime/replay/short-full。

最终一句话：

> v9.8.1 真实执行后停在 `R4-ArchitectureAgnosticGeometryFail`：它把 v9.8.0 的 endpoint-only future trajectory 诊断拆成 full-path、mechanism、geometry、density 和 raw/support gate 五个边界；但 h1/h5 full-path materializer 与 natural labeled stream extension 都未落地，geometry score 仍失败，generated route 没有长程 official evidence，因此不能进入 official controller，strict PureKAN functional 仍未成功。
