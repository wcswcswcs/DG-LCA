# DG-KAN v9.9.4 Tail Fidelity / Natural Density / FuturePathOperator 实验复盘

> 本复盘记录 `DG-KAN_v9.9.4_TailFidelityNaturalDensity_FuturePathOperator_四线并行完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 density/controller/generated/runtime 均显式 `not_run`。

## 0. 最新结论

```text
route = CaseC-GeneratorFidelityStillFails
primary_blocker = no_official_major_tail_fidelity_generator_passed
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

最终 artifact：`results/real_rerun_20260506/v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z`

核心结论：

1. P0 复现 v9.9.3 boundary：source route = `R9-MaterializerOrFidelityEngineeringBlocked`，G5 major/tail = `0` / `0`，G6 tail = `0`。
2. P1 tail reference audit pass = `1`；tail groups = `2191`，support<3 = `2051`，missing-rate = `0.0`。
3. P2 candidate count = `8`，official pass count = `0`；route 中 `best_generator_id` 为空是因为只允许 official pass generator 进入 best/P3。
4. P2 最接近但仍失败的 official-eligible candidate = `G9-two-stage-major-then-tail-fill-generator`：major PSI/JS/max-share = `0.0771752380044815` / `0.09765778310376505` / `0.40234375`，tail PSI/JS/missing = `0.178287981037832` / `0.13478134811256073` / `1`。
5. P3 largest completed panel = `0`，density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
6. P3 CoreLike count/LCB/UCB = `0` / `0` / `0`；PathGood count/LCB/UCB = `0` / `0` / `0`。
7. P4 future path weak/strong = `0` / `0`。
8. P5 FPO v6 weak/strong = `0` / `0`；best = `None`，precision = `None`。
9. P6 controller = `not_run`；P7 generated sandbox allowed = `0`。
10. No-fake audit：rows checked = `2267`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加 G7-G12 generator repair profiles；G12 标记 `official_density_eligible=0`。 |
| `experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py` | v9.9.4 runner；执行 P0/P1/P2 repair matrix、按 gate 开 P3-P8，并写 manifest/recap。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py
```

```bash
python experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py --out-dir results/real_rerun_20260506/v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
```

执行中发现并修复的 blocker：

```text
blocker = runner_empty_best_guard_bug
symptom = P2 official_pass_count=0 后，空 best 被 str(None) 转成 "None"，误启动 p2_none replay branch
action = stop erroneous branch, remove partial p2_none payload dir, patch run_density_panels guard, add --resume-after-p2
resume_scope = consume already materialized P2 CSV rows only, then write P3-P8 boundary/route/manifest/recap
fake_data_used = 0
proxy_row_used = 0
```

恢复命令：

```bash
python experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py --out-dir results/real_rerun_20260506/v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z --resume-after-p2 --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9940",
  "status": "summary",
  "route": "CaseC-GeneratorFidelityStillFails",
  "primary_blocker": "no_official_major_tail_fidelity_generator_passed",
  "secondary_blocker": "density_panels_blocked",
  "source_route_v9930": "R9-MaterializerOrFidelityEngineeringBlocked",
  "P0_boundary_pass": 1,
  "P1_reference_audit_pass": 1,
  "P2_official_generator_pass": 0,
  "P2_official_generator_pass_count": "0",
  "best_generator_id": "",
  "best_major_PSI_max": "",
  "best_tail_PSI_max": "",
  "best_missing_tail_group_count": "",
  "P3_largest_completed_panel_size": 0,
  "P3_density_sufficient": 0,
  "P3_density_insufficient": 0,
  "P3_density_inconclusive": 1,
  "CoreLike_count": 0,
  "CoreLike_LCB": 0,
  "CoreLike_UCB": 0,
  "PathGood_count": 0,
  "PathGood_LCB": 0,
  "PathGood_UCB": 0,
  "P4_future_path_weak_pass": 0,
  "P4_future_path_strong_pass": 0,
  "P5_FPO_weak_pass": 0,
  "P5_FPO_strong_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_generator_fidelity_failed",
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Tail Reference Audit

```text
tail_group_count_total = 2191
tail_group_key_missing_rate = 0.0
tail_group_singleton_count = 1801
tail_group_support_lt3_count = 2051
reference_tail_group_has_future_label_flag = 0
generator_quota_uses_outcome_tail = 0
```

## 4. P2 Generator Repair Matrix

```text
candidate_count = 8
official_pass_count = 0
route_best_generator_id = empty_because_no_official_pass
```

| generator | eligible | rows | major PSI/JS/share/entropy | tail PSI/JS/missing/coverage | CoreLike/PathGood/SlowBurn | official |
|---|---:|---|---|---|---|---:|
| `G5-hybrid-quota-random-generator` | `1` | `30720/30720` | `0.010737906793276122`/`0.036615381012353704`/`0.478515625`/`0.9306829849470822` | `0.17162945301293958`/`0.12137450698650669`/`9`/`0.9466666666666667` | `2`/`4`/`1` | `0` |
| `G6-tail-aware-quota-generator` | `1` | `30720/30720` | `0.5361416360865698`/`0.25026875823479056`/`0.5341796875`/`0.8657518702179914` | `1.2636767528720672`/`0.2900303011542032`/`56`/`0.5333333333333333` | `2`/`3`/`1` | `0` |
| `G7-multi-axis-precursor-quota-generator` | `1` | `30720/30720` | `0.1416297641706466`/`0.13140519211924445`/`0.4423828125`/`0.9206981053193084` | `2.5543767352206785`/`0.3348734522203027`/`121`/`0.3695652173913043` | `0`/`2`/`0` | `0` |
| `G8-tail-reservoir-capped-generator` | `1` | `30720/30720` | `0.10771006955215136`/`0.1152483914490968`/`0.4423828125`/`0.945606841393811` | `0.8622362943245747`/`0.2390676096530837`/`37`/`0.782608695652174` | `1`/`6`/`1` | `0` |
| `G9-two-stage-major-then-tail-fill-generator` | `1` | `30720/30720` | `0.0771752380044815`/`0.09765778310376505`/`0.40234375`/`0.9709714540474768` | `0.178287981037832`/`0.13478134811256073`/`1`/`0.9866666666666667` | `2`/`3`/`2` | `0` |
| `G10-alias-proportional-tail-min-generator` | `1` | `30720/30720` | `0.011076956339668183`/`0.037191767024139157`/`0.4609375`/`0.9401977253101717` | `0.12226564439741208`/`0.10090000439426848`/`3`/`0.9764705882352941` | `2`/`4`/`1` | `0` |
| `G11-stratified-mixture-generator` | `1` | `30720/30720` | `0.027590682563384563`/`0.058656869051056715`/`0.4306640625`/`0.9469939350913917` | `0.20590269111116513`/`0.13590271358712186`/`7`/`0.9565217391304348` | `2`/`4`/`1` | `0` |
| `G12-diagnostic-tail-upper-bound-sampler` | `0` | `30720/30720` | `0.1662978953610479`/`0.1414904233228188`/`0.4296875`/`0.9280457822664199` | `1.6075279863854106`/`0.27603688643338237`/`82`/`0.5434782608695652` | `0`/`2`/`0` | `0` |

判断：G9 的 tail missing group 最少，但 major PSI/JS/max-share 与 tail PSI/JS 仍未达 official gate；G10/G11 的 major PSI/JS 较好，但 max-share 与 tail gate 仍失败。G12 是 diagnostic-only 且本身也未过 fidelity，因此不能打开 P3。

## 5. P3 Density Boundary

```text
largest_completed_panel_size = 0
CoreLike count/LCB/UCB = 0 / 0 / 0
PathGood count/LCB/UCB = 0 / 0 / 0
density sufficient/insufficient/inconclusive = 0 / 0 / 1
reason = P2_no_official_major_tail_fidelity_generator_passed
```

## 6. Boundary

```text
P4 future path = 0 / 0
P5 FPO = 0 / 0
P6 controller = not_run, reason = P3_density_not_sufficient_and_P5_P4_strong_not_passed
P7 generated = not_run, allowed = 0
P8 runtime/paired replay = not_run unless P6/P7 pass
```

## 7. No-Fake / Contract / Failure

```text
rows_checked = 2267
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `codex_retry_suggestions.csv` | `2c395dc269f96bc19c13ecb80de4b29830c83e818dadc8ec602227a9426ea13b` |
| `contract_audit_v9940.csv` | `a090322b31ddabd5a11723214f17aecfdedab96d3aab11ca26faa02489c50d4b` |
| `failure_taxonomy_v9940.csv` | `6d28c34158a01a80c40c0321f2c8f4bb2db31ff11341c32fe1370aa3854b33ea` |
| `fig_p2_generator_repair_matrix_v9940.svg` | `98889cea77448fb0ea522e6a86c21809a7fdbb6dd70944789832d83a92302ae9` |
| `fig_p3_density_ci_v9940.svg` | `4eb7cf77e3b694afbbf7ec6135ad4b9ba15f32b7e70a236bd443c690eedfc549` |
| `fig_p5_fpo_v6_v9940.svg` | `c3de33ac6b639bedede409587024355c8b39a8ece2ab10375fe439c849edfe30` |
| `fig_v9940_gate_matrix.svg` | `fabc091ed269bfe99af755637dd35a048b4dc551640c5b507a33597c728c8a37` |
| `materializer` | `16a7d2e37e6c7ae54474fc62909378751ae3315a660e8f6320f3a9b30f6c2616` |
| `no_fake_audit_v9940.csv` | `957d678267e18d8ba0bf920c7628340ee87e213dcddca3a1dbf47af3f6bc1f34` |
| `p0_v9930_boundary_reproduction.csv` | `d83294534ac07f66018fae70965e60e1b7e1a2041e3ef7cc61fdfd9aef0e72b2` |
| `p1_tail_group_reference_audit_v9940.csv` | `2c877b58b1de6596bd5a2e4c5654d07f8ac72f499cede56f1b36aee8b0e582f9` |
| `p2_generator_major_tail_fidelity_v9940.csv` | `e392f8f8f0c106061d2d95558c40fe7e208035242ffb31de30d584591cf64ece` |
| `p2_generator_repair_matrix_v9940.csv` | `e6e8ace86a3d3f9982775fcb27eeb380429cdfacd030b36ce5d3e4082e39de80` |
| `p2_missing_tail_group_debug_v9940.csv` | `4068dda1f174ce1e5465cfc0eacfa3c39cacadbf0ea8326a89d855e4b5a2bdf4` |
| `p3_natural_density_group_rates_v9940.csv` | `f0f1603eb681e6fd02dca20b2e33cbc9a58233f90c0c3cb7cd7a148366fe5eab` |
| `p3_sequential_natural_density_panel_v9940.csv` | `ff51b9b51b6941e420bc7161387fa94a6bb72484b3d8717fccc4e5178847c1aa` |
| `p4_future_path_type_decomposition_v9940.csv` | `903bbcbc3d260d79dfeeb03655abc658845c6177c3e2ef7fcaa8eb4432744914` |
| `p5_future_path_operator_sketch_v6_v9940.csv` | `ae3a3ac73c324ce44e147aa5afa7f71c12cb41f9c0f06e5521cbd136a39eead4` |
| `p6_existing_action_controller_gate_v9940.csv` | `7eeb2754f7f822d4db4e8c058d952148caf76e0b74e5adaa7e05e505e7fc19ad` |
| `p7_generated_sandbox_gate_v9940.csv` | `39e591c71934f8ac4d70857f05c5e1f91f48c3c78bc5c7e32798190d25721d62` |
| `p8_paired_replay_boundary_v9940.csv` | `9203bdd32841bdac868b28542cde070b5693fab8ebb8d9fbcdd9748475530d8f` |
| `p8_selected_runtime_boundary_v9940.csv` | `5769147965d027f0e2fec15036172d7db3278f327847d30737ee4f7219ef739a` |
| `plan` | `9202bf97921d0a4e9587f3c269c95cb3813ae66bd4a2662189e9646d88987d52` |
| `route_decision_v9940.json` | `141350b4bc52cc038aa21baedd3b5bcf0fab429d5d58d6254b45814e58a1eb69` |
| `run_manifest_v9940.json` | `ce23bb9d13dcde35c8f3293931c9662a34e73b5c0315ac9c71a633ae2025f9de` |
| `runner` | `ce4b233ab36c5d4e3107cfd2910368b5757a1166f62d009a9126a0156513f624` |
| `tail_group_lineage_debug.csv` | `b082c511b7a9c9979bb237311bb4af75354e5eebab8c542269ba15d9a6b101d2` |

## 9. 最终分析结论

```text
1. v9.9.4 没有沿用旧完成稿；所有结论来自本轮 v9940 artifacts。
2. P1 先审计 tail reference，把 outcome diagnostic tail 与 commit-time precursor tail 分离，support<3 tail group 只作为 diagnostic/merged-tail 处理。
3. P2 按计划运行 generator repair matrix，G12 只允许 diagnostic upper-bound，不能打开 official density。
4. 只有 official major+tail fidelity generator 通过时，P3 才打开真实 sequential density panel；否则 density/controller/generated/runtime 均 gate-block。
5. No-fake audit 继续要求 fake/proxy/cpu 全为 0。
```

最终一句话：

> v9.9.4 真实执行后停在 `CaseC-GeneratorFidelityStillFails`：本轮按 major+tail fidelity gate 重新审计并尝试 generator repair；未满足 gate 的阶段保持 not_run，没有把旧结论或 diagnostic panel 写成 official controller pass。

## 10. 补充计划执行结果

> 本节对应用户补充计划。它不复用旧完成稿结论；所有数字来自补充 artifact。C 线先做 distribution-only 1024，只有 major+tail 通过才打开 branch-horizon/5000。

```text
supplement_artifact = results/real_rerun_20260506/v9940_tail_fidelity_natural_density_futurepathoperator_supplement_20260517T170000Z
route = Supplement-C-NoMajorTailDistributionPass
primary_blocker = no_supplemental_generator_major_tail_distribution_pass
system_legal_controller_pass = 0
generated_route_status = stopped_supplement_gate_not_met
```

### 10.1 C 线 Tail Fidelity 修复

```text
candidate_count = 6
distribution_pass_count = 0
best_distribution_generator_id = 
best_major_PSI_max = 
best_tail_PSI_max = 
best_missing_tail_group_count = 
branch_1024_opened = 0
panel_5000_opened = 0
density_sufficient/insufficient/inconclusive = 0 / 0 / 1
```

| generator | spec | major PSI/JS/share | tail PSI/JS/missing/coverage | branch 1024 | panel 5000 | pass |
|---|---|---|---|---|---|---:|
| `G7-exact-tail-quota-stochastic-refill` | exact tail quota with stochastic refill | `2.5057293757818218`/`0.5025596296769392`/`0.6142578125` | `11.246144700810124`/`0.680365811572192`/`144`/`0.32941176470588235` | `not_run` | `not_run` | `0` |
| `G8-hierarchical-major-tail-quota` | hierarchical major->tail quota | `0.4582716016230103`/`0.2369566070641189`/`0.53515625` | `1.0692450512649707`/`0.3387872992771653`/`17`/`0.8470588235294118` | `not_run` | `not_run` | `0` |
| `G9-tail-oversampling-importance-weights` | tail oversampling + importance weights | `0.5682869262676787`/`0.2627556392769717`/`0.556640625` | `1.7658786606293881`/`0.4307377609598343`/`33`/`0.7529411764705882` | `not_run` | `not_run` | `0` |
| `G10-recipe-replay-canonical-tail-precursor` | recipe replay from canonical tail precursor | `0.29307364512132544`/`0.18208152183320736`/`0.5693359375` | `7.760083151652164`/`0.5079447598765361`/`11`/`0.8064516129032258` | `not_run` | `not_run` | `0` |
| `G11-g5-major-tail-replay-mixture` | mixture of G5 major-preserving and tail replay | `0.2813652987225589`/`0.18610559488785425`/`0.5263671875` | `0.7796697550768278`/`0.2626969782170868`/`13`/`0.8588235294117647` | `not_run` | `not_run` | `0` |
| `G12-rejection-resampled-tail-fidelity-generator` | rejection-resampled tail fidelity generator | `0.8382792967822849`/`0.3169222238822192`/`0.5546875` | `7.681114821644478`/`0.49966308255653735`/`10`/`0.8064516129032258` | `not_run` | `not_run` | `0` |

判断：补充 C 线已经按计划尝试 G7-G12，但没有任何候选同时满足 major+tail distribution gate。G12 missing tail group 最少，但 major/tail PSI 仍明显失败；G11 tail PSI 最低，但 missing tail group 仍为 `13` 且 major gate 未通过。因此不打开 branch-horizon 1024，也不打开 5000。

### 10.2 A 线 Future Path 类型

```text
landed_future_action_rows = 339
FastGood_count = 129
SlowBurnGood_count = 116
SlowBurnGood_relaxed_count = 148
RiskyHighAUV_count = 79
SafeLowValue_count = 46
BadPath_count = 81
A_line_generation_target_ready = 1
```

| path type | actions | V1/V20/V80/V240 LCB | RAUV LCB | longrisk UCB | memory/offdiag UCB | mechanism signal |
|---|---:|---|---:|---:|---|---:|
| `FastGood` | `129` | `0.4777126489001311`/`1.4345634933736964`/`3.107172511439381`/`5.003540349832131` | `2.920589958256682` | `0.028918651988533713` | `0.028918651988533713`/`0.028918651988533713` | `0` |
| `SlowBurnGood` | `116` | `-0.14603216164819185`/`0.7635074741407439`/`2.046132388363771`/`3.596780247094066` | `1.899958279797453` | `0.03205564678709229` | `0.03205564678709229`/`0.03205564678709229` | `1` |
| `SlowBurnGoodRelaxed` | `148` | `-0.0731482415696314`/`0.7653903144212331`/`2.002999484248088`/`3.5508001548314847` | `1.8828656254854272` | `0.02530004952529478` | `0.02530004952529478`/`0.02530004952529478` | `1` |
| `RiskyHighAUV` | `79` | `0.08465391237936143`/`0.6883042653458938`/`2.008811036102984`/`3.760436324471754` | `1.9770100975978409` | `0.9109181967901335` | `0.9726655381852706`/`0.9870021486979305` | `0` |
| `SafeLowValue` | `46` | `-0.0627935797887234`/`-0.14033430019419024`/`0.5741638415625261`/`1.6248768976122374` | `0.6358691447673925` | `0.07707617732978073` | `0.07707617732978073`/`0.07707617732978073` | `0` |
| `BadPath` | `81` | `0.09066497840786442`/`0.5998248986928776`/`1.9068392630416968`/`3.7074759738907037` | `1.9065685496461469` | `0.9131737718333655` | `0.9733478809104459`/`0.9873251155414513` | `0` |

判断：A 线能形成 SlowBurnGood 诊断信号，但它仍不是训练当下合法 FuturePathOperator 或 generated target；因此只记为 diagnostic，不作为 D 线 reopen 条件。

### 10.3 B 线 FuturePathOperator Sketch

```text
sketch_count = 3
evaluated_action_count = 8192
B_line_weak/strong = 0 / 0
best_sketch = Sketch3_signal_reservoir_split_proxy
best_precision/V/longrisk/cost_q90 = 0.011494252873563218 / -0.8045174417254286 / 0.9752047278432054 / 0.0010817311704158783
```

| sketch | TopK87 precision | V LCB | RAUV LCB | LongRisk UCB | Bad/Null UCB | cost q90 ms | strong |
|---|---:|---:|---:|---:|---|---:|---:|
| `Sketch1_tiny_virtual_AdamW` | `0.0` | `-1.0076302779404567` | `-0.7388197077576191` | `1.0` | `1.0`/`0.04228899535014795` | `0.0017918646335601807` | `0` |
| `Sketch2_JVP_VJP_path_sketch` | `0.0` | `-0.9102539927361704` | `-0.6300427674102851` | `1.0` | `1.0`/`0.04228899535014795` | `0.0014621764421463013` | `0` |
| `Sketch3_signal_reservoir_split_proxy` | `0.011494252873563218` | `-0.8045174417254286` | `-0.57363266947211` | `0.9752047278432054` | `0.997968144575522`/`0.04228899535014795` | `0.0010817311704158783` | `0` |

判断：B 线三个 sketch 都满足低成本，但 precision/value/risk gate 全失败；这不是 cost blocker，而是 selector 质量 blocker，不能放入 official controller。

### 10.4 D 线 Generated Gate

```text
generated_sandbox_allowed = 0
reason = A_line_path_type_diagnostic_not_legal_generation_target_mechanism
```

### 10.5 No-Fake / Hash

```text
rows_checked = 2455
fake/proxy/cpu = 0 / 0 / 0
```

| supplement artifact | SHA256 |
|---|---|
| `a_line_future_path_type_decomposition_v9940_supplement.csv` | `c569e2fbe6d97fa436530d0500bc89c2fd5ceea1839fc80418abb8c272c72ae8` |
| `b_line_future_path_operator_sketch_v9940_supplement.csv` | `e044d6b622def81ba00fdfb90ad3f19859c3f43259c947a8a86a9b7c675d4112` |
| `c_line_branch_and_5000_gate_v9940_supplement.csv` | `8e4f70f1f2cc94e712112e130b1a5728726ed0b7650a2a7f0efdbd9a13132456` |
| `c_line_generator_repair_matrix_v9940_supplement.csv` | `e76287a9b1a10012d91664f9907ccbea922d12fa453a266eec8034ff27929ae3` |
| `c_line_missing_tail_group_detail_v9940_supplement.csv` | `318e5663d7170a971f1bb4b6c53c33892eccbbd331ad1dcdf4c7a375483de951` |
| `c_line_tail_group_lineage_debug_v9940_supplement.csv` | `b082c511b7a9c9979bb237311bb4af75354e5eebab8c542269ba15d9a6b101d2` |
| `c_line_tail_group_reference_audit_v9940_supplement.csv` | `2c877b58b1de6596bd5a2e4c5654d07f8ac72f499cede56f1b36aee8b0e582f9` |
| `contract_audit_v9940_supplement.csv` | `8395f025149848744b344c765a18ca23ce8667dc0ced877ad994b7962eac39ee` |
| `d_line_generated_route_gate_v9940_supplement.csv` | `a304423ff301c9d332a21af53e79245b86784fa70e1df920fcce417de4636f1d` |
| `failure_taxonomy_v9940_supplement.csv` | `ed24ec3f4959e1892a56cb819978720ccafa367cfa0ec514926bd4929a8969c8` |
| `fig_supplement_a_path_types.svg` | `adbfb160ce096ba505787f3e7be81926f51d74e3a8bc0cdf7b6e47e0c08b9653` |
| `fig_supplement_b_sketch_gate.svg` | `305484816247a55c1ebf45421bd33b74d236578e1c90573a30b7c64bb1104d1d` |
| `fig_supplement_c_gate_matrix.svg` | `d8142b01c9ca0fd87e3c3c7c7c382f10750516bd433f6bf66302fdd552597bbe` |
| `materializer` | `1613fc8b802a72ffd8478ffdc81b0345fdb59a809091de9e4eee2e11700b7c5c` |
| `no_fake_audit_v9940_supplement.csv` | `4090a2dfc5a6fc6dbea962cdd0a1194a38a2e61f60b336642a2be149169e8476` |
| `p0_v9940_base_boundary_supplement.csv` | `358277f9133d8f7e58d19c1d6980b53912e22b481a9a603ad834c8247f9780f2` |
| `plan` | `9202bf97921d0a4e9587f3c269c95cb3813ae66bd4a2662189e9646d88987d52` |
| `route_decision_v9940_supplement.json` | `7fe7f288cd60587b9e002e6a94b7c9fbc8abbeb31d1b59846a27c4fe4e59c578` |
| `run_manifest_v9940_supplement.json` | `7d752973bfd5dc1b98e56088de55b6b607f2fd75ec0a04d6d17e48184ab53f7c` |
| `runner` | `898e408229631cd25e61ced7e58eeccbbf4070fc48f695bb35ebd50293a7d9b6` |

补充最终判断：

```text
1. C 线按补充计划先执行 distribution-only 1024；未过 major+tail 的 generator 没有打开 branch-horizon/5000。
2. A 线用已落地 future rows 拆出 Fast/SlowBurn/Risky/SafeLow/Bad 类型，SlowBurn 若不足只放宽 V1，不放宽 h240/risk。
3. B 线只用 commit-time fields 重新做三个低成本 sketch；evaluation 才读真实 replay label。
4. D 线继续严格 gate；C/B/A 条件不足时 generated sandbox 不打开。
```
