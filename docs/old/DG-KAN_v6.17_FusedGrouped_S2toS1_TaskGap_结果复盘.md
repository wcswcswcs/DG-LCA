# DG-KAN v6.17 FusedGrouped S2toS1 TaskGap 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v617_real_all_20260505T103233Z` 下的 final real-only run。运行启动于 2026-05-05 10:32:35 UTC，完成于 2026-05-05 10:33:45 UTC。所有结论来自该目录落盘的 CSV/JSON/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v617_real.py \
  --packages V6_17_ALL \
  --out-dir results/real_rerun_20260505/v617_real_all_20260505T103233Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v617-real-20260505 \
  --wandb-name-prefix v617-real
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v617_real_all_20260505T103233Z`
- 本地主日志：`results/real_rerun_20260505/v617_real_all_20260505T103233Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/9wkeqv85>
- W&B run name：`v617-real-v617_real_all_20260505T103233Z`

### 1.3 不用于结论的检查

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v617_smoke` | 极小规模 smoke，检查 runner/gate/CSV shape | 0 |
| `results/real_rerun_20260505/v617_real_all_20260505T102858Z` | 第一次 full run 缺少 P9/P10 gated artifacts，修正 runner 后重跑 | 0 |

### 1.4 本轮真实实现与未实现边界

v6.17 沿用 v6.16 已实现的真实 Triton fused grouped kernel；本轮新增的是效率 margin repair：

| 文件位置 | 实现 |
|---|---|
| `experiments/run_gafu_v617_real.py:64` | `V617RecomputePrimitiveStack`，forward 不保存 hidden cache，backward 从原始 input 重算 layer input |
| `experiments/run_gafu_v617_real.py:98` | `reset_v10_gt4_recompute_hidden_cache` policy translation |
| `experiments/run_gafu_v617_real.py:139` | P2 `E1-GT4-no-extra-output-cache` 真实 measured candidate |

关键 no-fake 修正：

- `E4-GT4-forward-fastpath` 没有真实新 fastpath，因此明确写为 `not_implemented`。
- P4/P8/P9/P10 因 gate 阻断时全部写 `not_run`，不写假 loss/acc。
- 所有 CSV 中 `fake_data_used=0` 且 `proxy_row_used/proxy_rows_used=0`。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V6_17_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；task gate 未打开，因此无 task multi-seed |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| task steps | 计划 `120`，但本轮无 S0/S1/S2，所以未进入 task |
| run duration | `70.85` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| W&B logging | 记录 P0-P6/P7-P10 gated summary；无每 20 step task loss 曲线 |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 8 |
| `p0_reproduction_check.csv` | 真实生成 | 1 |
| `p1_gt4_efficiency_margin_attribution.csv` | 真实生成 | 72 |
| `p1_component_kernel_audit.csv` | 真实生成 | 720 |
| `p2_efficiency_margin_repair.csv` | 真实生成 | 11 |
| `p2_efficiency_margin_repair_detail.csv` | 真实生成 | 216 |
| `p3_fused_grouped_candidate_selection.csv` | 真实生成 | 1 |
| `p4_diagnostic_task_gap_attribution.csv` | `not_run` | 1 |
| `p4_task_trace.csv` | `not_run` | 1 |
| `p5_expressivity_repair.csv` | gated / not_implemented rows | 8 |
| `p6_optimizer_regularization_diagnostic.csv` | not_implemented rows | 11 |
| `p6_optimizer_regularization_trace.csv` | `not_run` | 1 |
| `p7_one_step_probe.csv` | `not_run` | 1 |
| `p8_task_reentry.csv` | `not_run` | 1 |
| `p8_task_trace.csv` | `not_run` | 1 |
| `p9_optimizer_exploration.csv` | `not_run` | 1 |
| `p10_functional_correction_smoke.csv` | `not_run` | 1 |
| `failure_table.csv` | 真实生成 | 37 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |

关键 hash：

| 文件 | SHA256 |
|---|---|
| `run.log` | `daee6f7480015e65d2c362884857013d1eac279083d83ea1d019371f40705c63` |
| `p1_gt4_efficiency_margin_attribution.csv` | `5af3716dc5921f4a9ecb510fdf94c1bd84478d9cb3c433c1ffd27519081dc12d` |
| `p2_efficiency_margin_repair.csv` | `6bac3a944850b83f80445f7761d46946053051a2382007653b011c468457852a` |
| `p2_efficiency_margin_repair_detail.csv` | `cced8a33e569b0ebe24d1d5a9c8b54c14eab93f578d1ea909aa378fdbc1f4bda` |
| `p3_fused_grouped_candidate_selection.csv` | `1bb5306cf9491c2bdd76d283c7e2ce5bf36e3471f8fc26801487e7130eada436` |
| `p4_diagnostic_task_gap_attribution.csv` | `43562682922edde4ef62ac73f8d3050ea71261a22a1c0823933ad802e539160e` |
| `p8_task_reentry.csv` | `35f11011ae019e7574ca2142f0bdf30bf855f8490ef60198d5f19b34560c8477` |
| `failure_table.csv` | `d92b5ee388c7a8aad3c08f89fe3d1f1325bdd37db57cac8d9ffe44b2dfbd3a7d` |
| `route_decision.json` | `1fa3b895c8b3ed30a7733b3f78654f8a84bdf7b9ff4b1234cc2872abb29f1b72` |

## 4. P0 Contract / v6.16 Reproduction

P0 结论：

- v6.17 继续使用 v6.16 的真实 Triton fused grouped path，不新增 fake fused row。
- `GT4-recompute-hidden-cache-v617` 是真实 manual forward/backward/update code path。
- DWM2 patch 主线继续冻结。
- `forward-fastpath`、crossgroup-lite 等未实现项写为 `not_implemented`。

Reproduction / comparability：

| metric | v6.16 GT4 ref | v6.17 best | delta | pass |
|---|---:|---:|---:|---:|
| memory ratio mean | `1.0311948157` | `1.0234326653` | `-0.0077621504` | 1 |
| step ratio mean | `1.4425903271` | `1.5891392104` | `+0.1465488833` | 0 for S2/S1, comparable for rerun |

注意：v6.17 best 不是 v6.16 GT4 baseline，而是 E1 recompute candidate。它 memory 更低，但 step 明显变差。

## 5. P1 GT4 Efficiency Margin Attribution

P1 覆盖：

```text
3 datasets x 3 batch sizes x 2 depths x 4 variants = 72 rows
```

Measured summary：

| variant | memory ratio vs MLP | step ratio vs MLP | backward ratio vs MLP | grad relerr max |
|---|---:|---:|---:|---:|
| `A0-MLP-autograd-reference` | `1.0000 / 1.0000 / 1.0000` | `1.0000` | `1.0000` | `0.0` |
| `MLP-manual-linear-reference` | `1.0652 / 1.1085 / 1.1718` | `1.1471` | `0.6771` | `3.93e-08` |
| `E0-GT4-v616-baseline` | `0.9866 / 1.0312 / 1.1017` | `1.5073` | `0.7656` | `1.92e-08` |
| `E1-GT4-recompute-hidden-cache` | `0.9813 / 1.0234 / 1.0951` | `1.5760` | `0.9226` | `1.92e-08` |

P1 判断：

- E1 真实降低了 mean memory ratio，从 `1.0312` 到 `1.0234`。
- E1 也降低了 worst memory ratio，从 `1.1017` 到 `1.0951`。
- 但 E1 的 step ratio 从 `1.5073` 变差到 `1.5760`，backward ratio 从 `0.7656` 变差到 `0.9226`。
- 梯度正确性通过，不是 numerical failure。

## 6. P2 Efficiency Margin Repair

P2 覆盖：

```text
18 shapes x (MLP reference + 2 measured candidates + 9 not-implemented candidates) = 216 detail rows
```

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | grad relerr max | S0/S1/S2 |
|---|---:|---:|---:|---:|---:|---|
| `E0-GT4-v616-baseline` | `1.0312` | `1.5191` | `0.7648` | `1.2338` | `1.92e-08` | `0/0/0` |
| `E1-GT4-no-extra-output-cache` | `1.0234` | `1.5891` | `0.9187` | `1.1988` | `1.92e-08` | `0/0/0` |

未实现 candidates：

```text
E2-GT4-grad-buffer-reuse
E3-GT4-update-temp-reuse
E4-GT4-forward-fastpath
E5-GT4-launch-fused-step
E6-GT4-reserved-memory-trim
E7-GT4-memory-margin-combo
E8-GT4-step-margin-combo
E9-GT4-S1-combo
E10-GT4-S0-aggressive
```

P2 判断：

- E1 是唯一新增真实 margin repair。
- E1 memory improvement vs GT4 为 `0.7527%`，但 step improvement vs GT4 为 `-10.16%`。
- E0/E1 都没有达到 S2，因为 step ratio 均高于 `1.50`。
- 未实现项没有写假 ratio。

## 7. P3 Candidate Selection

P3 最终选择：

| field | value |
|---|---:|
| best candidate | `E1-GT4-no-extra-output-cache` |
| best family | `memory` |
| best memory ratio | `1.0234326653` |
| best step ratio | `1.5891392104` |
| best backward ratio | `0.9186536554` |
| best forward ratio | `1.1987692169` |
| memory improvement vs GT4 | `0.7527%` |
| step improvement vs GT4 | `-10.16%` |
| survivor type | `S3` |
| open one-step probe | 0 |
| open diagnostic task | 0 |
| open official task | 0 |

P3 判断：

- v6.17 没有把 v6.16 S2 推到 S1/S0。
- 更严格地说，v6.17 best candidate 从 S2 退到了 S3：memory 更好，但 step gate 失败。
- 因此 P7/P8 不允许启动，不能生成 task loss 曲线。

## 8. P4-P6 Task Gap / Repair 状态

由于 P3 没有 S0/S1/S2，后续 task-gap 相关阶段没有资格运行。

| stage | status | reason |
|---|---|---|
| P4 diagnostic task gap attribution | `not_run` | task not opened |
| P4 task trace | `not_run` | task not opened |
| P5 expressivity repair baseline | gated, `val_acc/test_acc=nan` | P8 没有 task result |
| P5 X1-X7 repairs | `not_implemented` | 没有真实实现，不写假结果 |
| P6 optimizer regularization diagnostic | `not_implemented` rows | P8 未打开，不能从 task 结果派生 |

重要说明：

> 本轮 W&B 没有每 20 step loss 曲线。这不是漏记，而是 gate 正确关闭：没有 S2 及以上 candidate，因此按 v6.17 计划不能进入 diagnostic task。

## 9. P7-P10 Gate 状态

| stage | status | reason |
|---|---|---|
| P7 one-step probe | `not_run` | No S0/S1/S2 candidate |
| P8 task re-entry | `not_run` | P7 gated |
| P8 task trace | `not_run` | P8 gated |
| P9 optimizer exploration | `not_run` | P8 gated |
| P10 functional correction smoke | `not_run` | P9/P8 gated |

这部分不能解读成 task/optimizer/functional 失败；准确说法是 v6.17 efficiency margin gate 没有打开，所以按计划没有进入训练。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 28 | P2/P5/P6 未实现项或 P4/P7 gated |
| `F1_memory_fail` | 3 | measured summary rows memory ratio 未达到对应 gate |
| `F2_step_time_fail` | 3 | measured summary rows step ratio 未达到对应 gate |
| `F13_official_task_gated` | 3 | P8/P9/P10 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R5-EfficiencyMarginFailed",
  "best_candidate": "E1-GT4-no-extra-output-cache",
  "best_family": "memory",
  "best_memory_ratio": 1.0234326652684669,
  "best_step_ratio": 1.5891392103788484,
  "best_backward_ratio": 0.9186536553516209,
  "best_forward_ratio": 1.1987692169384692,
  "memory_improvement_vs_GT4": 0.007527336540000005,
  "step_improvement_vs_GT4": -0.10158731870397072,
  "survivor_type": "S3",
  "one_step_probe_pass": 0,
  "official_task_opened": false,
  "diagnostic_task_opened": false,
  "best_val_acc": -1.0,
  "best_test_acc": -1.0,
  "task_gap_taxonomy": "not_available",
  "open_optimizer_exploration": false,
  "open_functional_correction": false,
  "primary_blocker": "efficiency_margin",
  "next_required_implementation": "kernel_margin_repair",
  "no_fake": true,
  "no_proxy": true
}
```

## 12. 结论

本轮 v6.17 支持以下真实结论：

1. v6.17 确实执行了真实 runner 和 W&B run，没有 fake/proxy 数据。
2. v6.16 的 fused grouped GT4 路线仍是真实 baseline；v6.17 新增的 E1 hidden-cache recompute 也是真实 code path。
3. E1 把 memory ratio mean 从 `1.0312` 降到 `1.0234`，但 step ratio 从约 `1.52` 变差到 `1.59`。
4. E1 梯度正确，不是数值错误导致失败。
5. v6.17 没有得到 S0/S1/S2 candidate，因此没有 one-step probe、diagnostic task、official task、optimizer 或 functional correction。
6. 由于没有 task run，本轮无法分析 task gap；`task_gap_taxonomy` 正确写为 `not_available`。
7. 最终 route 是 `R5-EfficiencyMarginFailed`。

最终一句话：

> v6.17 没有把 v6.16 的 fused grouped S2 推到 S1/S0；唯一真实新增的 memory-margin repair 让 memory 更好但 step 变差，导致 candidate 退为 S3。下一步不能开 task，必须先做真正的 kernel margin repair，尤其是减少 recompute 造成的 backward/step 开销。

## 13. 下一步建议

1. 不要把 E1 作为 task candidate；它是 memory diagnostic，不是 S2/S1 survivor。
2. 若继续 fused grouped，应优先实现真实 grad/update buffer reuse 或 launch-fused step，而不是 backward 重算 hidden。
3. 目标仍是同时满足 `memory <= 1.05` 和 `step <= 1.50` 恢复 S2，再进一步推到 `step <= 1.35` 和 `memory <= 1.00`。
4. P8/P9/P10 继续关闭，直到出现真实 S2/S1/S0 candidate。
5. 下一轮如果要分析 task gap，必须先重新达到 S2；否则 loss/acc 曲线不应生成。
