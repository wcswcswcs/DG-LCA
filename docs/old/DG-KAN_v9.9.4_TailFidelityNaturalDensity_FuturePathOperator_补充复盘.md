# DG-KAN v9.9.4 Tail Fidelity / Natural Density / FuturePathOperator 补充复盘

> 本补充复盘记录 v9.9.4 full run 出最终 route/manifest 之前，Codex 可并行完成的只读审计、fail-fast 判据和下一步执行建议。本文不是正式 route 裁决；正式结论仍以 `route_decision_v9940.json`、`run_manifest_v9940.json` 和主复盘为准。

## 0. 当前快照

```text
snapshot_utc = 2026-05-17T12:25:21Z
running_process = python experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py ...
artifact_dir = results/real_rerun_20260506/v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z
route_decision_v9940.json = not_materialized_yet
run_manifest_v9940.json = not_materialized_yet
snapshot_completed_scope = P0/P1/G5-1024 only
status_update_2026-05-17T13:09:00Z = same out-dir was refreshed by a new --fresh v9940 process; current official scope is back to P0/P1 while the pre-result diagnostic artifacts below preserve the earlier G5 snapshot
```

已落盘事实：

```text
P1_reference_audit_pass = 1
canonical_action_count = 2876
tail_group_count_total = 2191
major_group_count_total = 202
tail_group_key_missing_rate = 0.0
tail_group_singleton_count = 1801
tail_group_support_lt3_count = 2051
reference_tail_group_has_future_label_flag = 0
generator_quota_uses_outcome_tail = 0
```

G5 当前 1024 pilot：

```text
major_fidelity_pass = 0
tail_fidelity_pass = 0
PSI_major_max = 0.010737906793276122
JS_major_max = 0.03661538101235371
max_major_group_share = 0.478515625
entropy_major_min = 0.9306829849470822
PSI_tail_max = 0.1716294530129395
JS_tail_max = 0.12137450698650668
missing_tail_group_count = 9
```

G5 missing tail axis snapshot：

| axis | missing_group_count | coverage | worst_missing_group |
|---|---:|---:|---|
| recipe_template_pair | 4 | 0.9466666666666667 | `573281|A2-LateAttachControlGapChannel` |
| recipe_step_pair | 2 | 0.9565217391304348 | `324566|s160_191` |
| source_recipe_id | 3 | 0.9647058823529412 | `483271` |

No-fake 局部检查：

```text
G5 action_rows = 1024
official_density_eligible = 1
uses_future_outcome_sum = 0
uses_old_table_label_sum = 0
fake/proxy/cpu = 0 / 0 / 0
```

## 0.1 Codex 已补做的实际诊断

这次不是只把建议抄进文档。已在独立目录生成 pre-result 诊断 artifact，避免污染正在运行的 official v9940 artifact：

```text
diagnostic_dir = results/real_rerun_20260506/v9940_pre_result_codex_diagnostics_20260517T1225Z
```

| artifact | rows | SHA256 |
|---|---:|---|
| `g5_tail_fidelity_diagnostic_summary_v9940.csv` | 1 | `e282bb86330cf575366b5303c6e1c652717fecc2cf030c4c6aeab0f90f7041d0` |
| `g5_missing_tail_group_detail_v9940.csv` | 9 | `e5cbc28bd3a3ccc481e066363fa0a258c7f1ef425dc05e1a99e6be7a280ec603` |
| `g5_major_share_detail_v9940.csv` | 7 | `f21bdc00146a5a98faf4fa3619147bdca284f01a9bb712108610f9dad760d5ed` |
| `g5_exact_reference_tail_key_missing_top200_v9940.csv` | 158 | `d14bba814a462145e256f4d7634359dc8db532f2e26600e5c4fb1a5225491a83` |
| `g5_g6_tail_fidelity_diagnostic_comparison_v9940.csv` | 2 | `01ae81d95b37e467860e829c4d098fd2a6f06bc5dd40504c6afdb05f0c9ea815` |

诊断结论：

```text
tail_missing_group_rows = 9
tail_missing_quota_rounding_zero_rows = 0
tail_missing_expected_quota_floor_positive_rows = 9
cursor_min/max = 1200000 / 1201023
chunk_start_histogram = 0:512;512:512
max_major_share = 0.478515625
max_major_share_axis = template_id
exact_reference_tail_key_missing_rows_support_ge2_top200 = 158
```

这说明 G5 的 9 个 axis-level missing tail groups 不是“rare group 四舍五入到 0”。每个 missing group 在 1024 下的 expected quota floor 都为正，失败更像是 G5 hybrid sampler 没有 exact tail residual/refill；同时 major share gate 也不是小问题，`template_id` 与 `dataset_id` 都超过 0.35。

说明：`g5_g6_tail_fidelity_diagnostic_comparison_v9940.csv` 只保留 G5 pre-refresh 诊断和 G6 unavailable 状态；G6 artifact 未在 official `--fresh` 重跑刷新前稳定保存，需等待当前 run 重新落盘后再补 G6-G12 诊断。

G5 major share detail：

| axis | top_generator_group | top_count | top_share | share_gate_pass |
|---|---|---:|---:|---:|
| dataset_id | `Fashion-MNIST` | 435 | 0.4248046875 | 0 |
| template_id | `A3-LateAttachRoleWiseFT7EdgeCarrier` | 490 | 0.478515625 | 0 |
| step_bucket | `s000_031` | 206 | 0.201171875 | 1 |
| payload_norm_bucket | `payload_norm_bin_2` | 255 | 0.2490234375 | 1 |
| action_norm_bucket | `payload_norm_bin_2` | 255 | 0.2490234375 | 1 |
| family | `873234` | 179 | 0.1748046875 | 1 |
| stratum | `464058` | 20 | 0.01953125 | 1 |

G5 missing tail groups with quota/cursor/debug：

| axis | tail_group_key | canonical_count | expected_quota_floor_at_1024 | expected_quota_round_at_1024 | quota_rounding_to_zero | filter_reason |
|---|---|---:|---:|---:|---:|---|
| recipe_template_pair | `573281|A2-LateAttachControlGapChannel` | 8 | 2 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| recipe_template_pair | `648505|A3-LateAttachRoleWiseFT7EdgeCarrier` | 8 | 2 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| recipe_template_pair | `721168|A3-LateAttachRoleWiseFT7EdgeCarrier` | 8 | 2 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| recipe_template_pair | `961046|A3-LateAttachRoleWiseFT7EdgeCarrier` | 8 | 2 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| recipe_step_pair | `324566|s160_191` | 9 | 3 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| recipe_step_pair | `512132|s192_223` | 10 | 3 | 4 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| source_recipe_id | `483271` | 10 | 3 | 4 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| source_recipe_id | `608203` | 9 | 3 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |
| source_recipe_id | `648505` | 9 | 3 | 3 | 0 | `not_selected_by_reference_sampler_or_refill;no_post_generation_filter_observed` |

## 1. 立即结论

P1 tail reference 口径目前看是干净的：tail key 从 commit-time canonical AP0 precursor 字段构造，reference audit 没有 missing key，也没有 outcome-label quota。下一步不要优先怀疑 outcome leakage；优先怀疑 generator quota/cursor/refill 机制。

G5 不是“major 已过、只差 tail”的状态。虽然 major PSI/JS/entropy 过线，但 `max_major_group_share = 0.478515625` 超过 `0.35` gate，同时 tail PSI/JS/missing 也失败。补充诊断进一步确认：missing tail groups 不由 quota rounding to zero 导致，major share 主要由 dataset/template concentration 导致。因此后续修复不应只加 tail oversampling；更应该用 hierarchical major->tail quota 或 capped reservoir，先压 dataset/template share，再补 tail residual/refill。

当前 v9.9.4 runner 已包含 G7-G12，但 `run_generator_matrix()` 先调用 `v9930.run_panel_sharded()`，写出 action/apply/branch-horizon 后才做 `tail_fidelity_audit()`。这与“cheap distribution-only 1024 先 fail-fast，过 major+tail 后才跑 branch-horizon”的建议不一致。它不直接作假，但会浪费大量时间，并让失败 generator 也产生 branch-horizon rows。

## 2. C 线补充判据

当前可等待 G6-G12 完成，但正式结果出来后应按下列顺序读：

```text
1. p2_generator_repair_matrix_v9940.csv
2. p2_missing_tail_group_debug_v9940.csv
3. 各 p2_G*_1024_major_tail_fidelity_v9940.csv
4. 若且仅若 official_density_generator_pass=1，再读 p3 sequential density panel
```

G7-G12 与建议矩阵的当前代码映射：

| generator | 当前实现 | 与建议的差异 |
|---|---|---|
| G7 | multi-axis precursor quota | 接近 exact tail quota，但没有显式 stochastic refill / residual carry |
| G8 | tail-reservoir capped | 更像 rare-first reservoir，不是 hierarchical major->tail |
| G9 | two-stage major then tail fill | 最接近 hierarchical major->tail quota |
| G10 | alias proportional tail-min | 不是 recipe replay from canonical tail precursor |
| G11 | stratified mixture | 接近 G5 major-preserving + tail replay mixture |
| G12 | diagnostic tail upper-bound sampler | diagnostic only，`official_density_eligible=0`，不是 official rejection-resampled generator |

正式结果若仍失败，优先补一个 distribution-only preflight，而不是继续加 full branch-horizon 大跑：

```text
P2a_distribution_only_1024:
  materialize action_rows only
  audit major/tail fidelity
  write missing tail group debug with canonical_count/generator_count/quota/cursor/filter_reason

P2b_branch_horizon_1024:
  only open when major_fidelity_pass=1 and tail_fidelity_pass=1

P2c_5000:
  only open when P2b branch-horizon completion/pass is clean
```

fail-fast 解释：

```text
missing_tail_group_count > 0:
  first check quota rounding / residual carry
  then filter deletion
  then cursor cycle
  then generated metadata tail key mismatch

major pass but tail fail:
  add tail-level quota only

tail pass but major fail:
  use hierarchical quota / major cap

1024 pass but 5000 fail:
  inspect batch state reset, cursor periodicity, and seed cycle collapse
```

## 3. A 线补充判据

Future path 类型不能只看 AUV。正式 run 若产生 completed density labels，补充复盘要额外落以下分组：

```text
FastGood
SlowBurnGood
RiskyHighAUV
SafeLowValue
BadPath
```

每组至少比较：

```text
V1 / V5 / V20 / V80 / V240
RAUV
longrisk
bad / null
memory / offdiag
dataset / family / template
hard-tail change
```

重点判据：

```text
SlowBurnGood:
  h1 不强或为负
  h20/h80/h240 变好
  longrisk 低

RiskyHighAUV:
  AUV 高但 longrisk/memory/offdiag 不干净
  不能作为 generation target

SafeLowValue:
  low risk but low value
  只允许作为 h240+ 延迟收益 diagnostic，不写成 official pass
```

## 4. B 线补充判据

当前 v9.9.4 仍调用 `future_operator_sketch_v5()`，并把 stage 重命名为 v6；其中 FPO1/FPO2/FPO3/FPO4 是 commit-time scalar proxy，并不等同于真实 tiny virtual AdamW、JVP/VJP path sketch、signal-reservoir split proxy。正式结果若 P5 失败，不应继续调 LC8/FOS4 或只调阈值。

下一版应把 B 线拆成三类低成本 sketch：

```text
Sketch 1: tiny virtual AdamW
  1-3 个小步
  只在 train-memory / hard-tail stratified mini-buffer 上算
  输出 value proxy 和 risk proxy

Sketch 2: JVP/VJP path sketch
  估计 action 对下一步 AdamW gradient direction 的影响
  目标是预测后续是否更容易优化

Sketch 3: signal-reservoir split proxy
  用 per-example response mean/variance 和 old-family consistency
  估计更新是否进入 signal channel
```

判定不要只看 AUC：

```text
TopK87_precision >= 0.75
V_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
cost_q90_ms <= 1.5
final_runtime_step_ratio_q90 <= 1.50
```

失败解释：

```text
precision low but risk low:
  proxy 是 veto，不是 value selector；加 value direction sketch

value positive but longrisk high:
  加 memory/offdiag hard gate，不调 value threshold

cost high:
  减少 virtual samples
  用 stratified mini-buffer
  cache JVP/VJP
  禁止放入 official controller
```

## 5. D 线补充判据

Generated route 继续严格 gate。正式 v9.9.4 只有三种情况能重开：

```text
1. C 线证明 natural density insufficient
2. B 线找到可用 FuturePathOperator
3. A 线给出明确 path type 机制，可作为 generation target
```

即便重开，也只允许：

```text
generated_sandbox_action_count = 64
direct_512_or_1024_generated_run_allowed = 0
```

generated 失败时先分型，不调 blend ratio：

```text
value_negative:
  check memory/offdiag/hard-tail damage

high_longrisk:
  check RiskyHighAUV, do not continue because AUV is high

low_risk_low_value:
  classify SafeLowValue and check h240+ delayed gain
```

## 6. 等正式结果后的落盘清单

正式 v9.9.4 结束后，补充复盘应追加：

```text
1. route_decision_v9940.json 摘要
2. p2_generator_repair_matrix_v9940.csv 中 G5-G12 每个 generator 的 major/tail gate
3. p2_missing_tail_group_debug_v9940.csv 的 canonical_count / generator_count / quota / cursor / filter reason
4. 若 P3 打开，追加 A 线 path type 分布和 B 线 sketch gate
5. 若 D 线打开，确认 only 64-action sandbox，并记录 generated failure type
6. no_fake / proxy / cpu 三项仍必须全为 0
```

最终一句话：

> v9.9.4 正式结果出来前，不应空等；当前最有价值的准备是把 tail fidelity failure 从“看起来像 tail 问题”拆成 major-share、tail-missing、quota/cursor/refill 三类，并在结果落地后优先补 distribution-only preflight，避免再次让未过 fidelity 的 generator 消耗 branch-horizon 大跑。
