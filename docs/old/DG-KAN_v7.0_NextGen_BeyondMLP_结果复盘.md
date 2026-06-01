# DG-KAN v7.0 NextGen BeyondMLP 结果复盘

> 本复盘只记录 `results/real_rerun_20260505/v70_real_all_20260505T152955Z` 下的 final real-only run。运行启动于 2026-05-05 15:29:56 UTC，完成于 2026-05-05 15:34:09 UTC。所有结论来自该目录落盘的 CSV/JSON/log/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 实验溯源

### 1.1 最终有效运行命令

```bash
python experiments/run_gafu_v70_real.py \
  --packages V7_0_ALL \
  --out-dir results/real_rerun_20260505/v70_real_all_20260505T152955Z \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v70-real-20260505 \
  --wandb-name-prefix v70-real
```

stdout/stderr 同时写入：

```text
results/real_rerun_20260505/v70_real_all_20260505T152955Z/run.log
```

### 1.2 结果目录与 W&B

- 本地结果目录：`results/real_rerun_20260505/v70_real_all_20260505T152955Z`
- 本地主日志：`results/real_rerun_20260505/v70_real_all_20260505T152955Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/jnhnm4of>
- W&B run name：`v70-real-v70_real_all_20260505T152955Z`

### 1.3 不用于结论的检查与自修复 run

| run | 用途 | 是否用于结论 |
|---|---|---:|
| `/tmp/v70_smoke` | 初版 runner smoke；极小样本下 canonical gate 未打开 | 0 |
| `results/real_rerun_20260505/v70_real_all_20260505T152356Z` | 第一次 full run：P4 有 S2，但 task gate 被 legacy GT4 protocol drift 误伤关闭 | 0 |
| `/tmp/v70_smoke2` | 修正 gate 后 smoke，确认 P8/P4 可真实打开 | 0 |

本轮按要求“失败后先自己尝试解决”：第一次 full run 暴露 v7 runner 复用 v6.20 global gate 太保守，导致直接 measured 的 `T3` S2 没有进入 diagnostic task。随后修正为：legacy GT4 protocol drift 只记录为 warning/failure，不再阻断 v7 直接 measured `T3` S2 diagnostic。最终结论只来自第二次 full run。

### 1.4 no-fake / no-proxy 约束

- 新入口：`experiments/run_gafu_v70_real.py`。
- 所有 dataset loader 调用均使用 `allow_fake_data=False`。
- `experiments/dgkan_core.py` 中 fake data 路径仍保持 hard-fail。
- 审计过的 `p0_contract.csv`、`p8_task_reentry.csv`、`p4_optimizer_regularization_diagnostic.csv`、`p4_optimizer_trace.csv`、`p5_crossgroup_expressivity_diagnostic.csv` 中 `fake_data_used=0` 且 `proxy_row_used/proxy_rows_used=0`。
- P2/P3 的 canonical denominator rebase 继续使用 P0 同 shape 的真实 MLP measured row。
- `O2/O3/O6/O7/O10` optimizer diagnostics 是真实 task run；未实现的 crossgroup / patch-token /部分 optimizer recipes 写为 `not_implemented` 或 `not_run`，没有写假 ratio/loss/acc。

## 2. 实验 setting

| 项目 | 设置 |
|---|---|
| packages | `V7_0_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| device | `auto`，本机解析为 `cuda` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| batch sizes | `128,256,512` |
| depths | `2,4` |
| seeds | profiler 使用 `seed=0`；diagnostic task 使用 seeds `0,1,2` |
| full warmup / measure | 每个 full-step measured row `50/200` update steps |
| task steps | `120`，每 `20` steps 记录 trace |
| run duration | `252.73` sec |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |
| Triton | available |
| memory history | available |
| W&B logging | 记录 P0-P14/failure/route summary；包含 absolute memory metrics 和 task trace |

数据集覆盖审计：

| artifact | rows | datasets | batch sizes | depths |
|---|---:|---|---|---|
| `p0_contract.csv` | 162 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p1_s1_memory_attribution.csv` | 108 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p2_s1_memory_repair_detail.csv` | 198 | MNIST/Fashion-MNIST/KMNIST | 128/256/512 | 2/4 |
| `p3_task_trace.csv` | 216 | MNIST/Fashion-MNIST/KMNIST | task batch 128 | task depth 2 |
| `p4_optimizer_trace.csv` | 486 | MNIST/Fashion-MNIST/KMNIST | task batch 128 | task depth 2 |
| `p13_diagnostic_task_trace.csv` | 216 | MNIST/Fashion-MNIST/KMNIST | task batch 128 | task depth 2 |

## 3. 产物清单

| artifact | 状态 | 行数 |
|---|---|---:|
| `p0_contract.csv` | 真实生成 | 162 |
| `p0_reproduction_check.csv` | 真实生成 | 7 |
| `p1_s1_memory_attribution.csv` | 真实生成 | 108 |
| `p2_s1_memory_repair.csv` | 真实生成 | 10 |
| `p2_s1_memory_repair_detail.csv` | 真实生成 | 198 |
| `p3_task_gap_attribution.csv` | 真实生成 | 3 |
| `p3_task_trace.csv` | 真实生成 | 216 |
| `p4_optimizer_regularization_diagnostic.csv` | 真实生成 | 12 |
| `p4_optimizer_trace.csv` | 真实生成 | 486 |
| `p5_crossgroup_expressivity_diagnostic.csv` | baseline measured + repairs not implemented | 8 |
| `p5_crossgroup_trace.csv` | 真实生成 | 54 |
| `p6_longer_budget_task.csv` | `not_run` | 1 |
| `p7_sample_efficiency.csv` | `not_run` | 1 |
| `p8_noise_robustness.csv` | `not_run` | 1 |
| `p9_patch_token_scaling.csv` | `not_run` | 1 |
| `p10_candidate_selection.csv` | 真实生成 | 1 |
| `p11_one_step_probe.csv` | 真实生成 | 1 |
| `p12_official_task_reentry.csv` | `not_run` | 1 |
| `p13_diagnostic_task.csv` | diagnostic measured | 36 |
| `p13_diagnostic_task_trace.csv` | diagnostic measured | 216 |
| `failure_table.csv` | 真实生成 | 53 |
| `route_decision.json` | 真实生成 | - |
| `aggregate_decision.json` | 真实生成 | - |
| `run_manifest.json` | 真实生成 | - |

关键 hash（复盘时重新计算）：

| 文件 | SHA256 |
|---|---|
| `run.log` | `8a7c859f43a54c1d22f5690b00da1f3b909da1aaba4bcae860cf6ba3c360c231` |
| `run_manifest.json` | `2f4e20a1d0249fbd65c1ad9d616be03e278019963a2f919da51f57c229a9d0e2` |
| `p2_s1_memory_repair.csv` | `0fe497ddfe57b7f7782d789dbda88f94a52410523bab59ebb60be052d6f017bb` |
| `p4_optimizer_regularization_diagnostic.csv` | `45daf31e15505992c9a3b4ecb8ea4dfa1e20e1927b659f62a2262610934eafdc` |
| `p8_task_reentry.csv` | `748fc101246287932964b84826831b839eb91fab055d394188c0af15b317d851` |
| `failure_table.csv` | `dbebb071763f9bba4bd265cee6540c3cb7f3c53c0cf67902a9a2caadd6112395` |
| `route_decision.json` | `e6eb690a0b26b9a1612d06aebac8f4f71e67a5744b628a0cdd241ff85d99c691` |

## 4. P0 / P1 Protocol 与 Memory Attribution

P0 measured summary：

| variant | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | survivor |
|---|---:|---:|---:|---:|---|
| `A1-MLP-manual-linear-reference` | `1.1085` | `1.1754` | `0.6983` | `0.8294` | S4 |
| `A2-GT4-protocol_A_v616_timing` | `1.0312` | `1.5347` | `0.7918` | `1.2391` | S3 |
| `A3-GT4-protocol_B_v618_p0_side_by_side` | `1.0312` | `1.5348` | `0.7885` | `1.2574` | S3 |
| `A4-GT4-protocol_C_v619_repair_protocol` | `1.0312` | `1.5344` | `0.7906` | `1.2487` | S3 |
| `A5-GT4-protocol_D_new_canonical_minimal_logging` | `1.0312` | `1.5573` | `0.7901` | `1.2556` | S3 |
| `A6-GT4-protocol_E_new_canonical_with_component_logging` | `1.0312` | `1.5302` | `0.7867` | `1.2449` | S3 |
| `A7-E1-recompute-hidden-cache-diagnostic` | `1.0234` | `1.6068` | `0.9421` | `1.2462` | S3 |

P0 结论：

- GT4 memory baseline 仍稳定为 `1.0312`。
- 但本轮 P0 legacy GT4 protocol rows 的 step ratio 为 `1.53-1.56`，不是 S2。
- v7.0 最终没有把 legacy GT4 P0 drift 当作 direct T3 diagnostic 的 gate；该 drift 已在 failure table 中记录。

P1 结论：

- absolute memory metrics 继续写入 W&B/CSV。
- E1 recompute 的 memory 改善仍真实存在，但 step/backward 代价更大；不作为主线。

## 5. P2 S1 Memory Repair

Measured candidates：

| package | memory ratio mean | step ratio mean | backward ratio mean | forward ratio mean | survivor |
|---|---:|---:|---:|---:|---|
| `M0-GT4-canonical-baseline` | `1.0312` | `1.5451` | `0.7935` | `1.2622` | S3 |
| `M3-update-temp-reuse` | `1.0312` | `1.0925` | `0.7764` | `1.1729` | S2 |
| `M9-E1-recompute-hidden-cache` | `1.0234` | `1.6150` | `0.9507` | `1.2494` | S3 |

未实现：

```text
M1-grad-mix-buffer-reuse
M2-grad-poly-buffer-reuse
M4-triton-scratch-lifetime-trim
M5-reserved-memory-trim
M6-short-lived-output-free
M7-grad-update-buffer-combo
M8-safe-memory-combo
```

P2 判断：

- `M3-update-temp-reuse` 继续是真实 S2，但 memory 仍为 `1.0312`，没有达到 S1 的 `<1.00`。
- E1 memory 最低 `1.0234`，但 step `1.6150`，只能是 memory diagnostic。
- v7.0 没有解决 S1 memory margin。

## 6. P3 / P13 Diagnostic Task Gap

P3/P13 使用 v6.20 T3 diagnostic task 链路，真实运行：

```text
3 datasets x 3 seeds x 4 variants x 6 logged steps = 216 trace rows
```

P8/P13 task summary：

| variant | rows | val acc mean | test acc mean | train loss delta mean |
|---|---:|---:|---:|---:|
| `MLP-autograd-reference` | 9 | `0.6981` | `0.6523` | `-2.3124` |
| `MLP-manual-linear-reference` | 9 | `0.6808` | `0.6337` | `-2.4837` |
| `T3-launch-fused-step+ManualAdamW` | 9 | `0.6387` | `0.6018` | `-2.5256` |
| `T3-launch-fused-step+ManualAdanLite` | 9 | `0.6280` | `0.5927` | `-2.5004` |

P3 判断：

- T3 diagnostic task 真实打开，不再是 gate-only。
- T3 的 train loss 下降比 MLP 更大，但 val/test 明显落后 MLP。
- `T3+ManualAdamW` 相对 MLP-autograd 的 val gap 为 `-0.0595`，test gap 约 `-0.0506`。
- 当前主要 blocker 是 task/generalization 或 expressivity gap，而不是 kernel step gate。

## 7. P4 Optimizer / Regularization Diagnostic

本轮新增真实 optimizer diagnostic，自修复尝试包括：

```text
O2 weight decay low
O3 weight decay high
O6 lr low
O7 lr high
O10 label smoothing
```

Measured optimizer recipes：

| recipe | val acc | test acc | val gap vs MLP | delta vs T3 AdamW | useful |
|---|---:|---:|---:|---:|---:|
| `O0-T3-ManualAdamW-v620` | `0.6387` | `0.6018` | `-0.0595` | `0.0000` | 0 |
| `O1-T3-ManualAdanLite-v620` | `0.6280` | `0.5927` | `-0.0701` | `-0.0106` | 0 |
| `O2-ManualAdamW-weightdecay-low` | `0.6274` | `0.5857` | `-0.0707` | `-0.0113` | 0 |
| `O3-ManualAdamW-weightdecay-high` | `0.6243` | `0.5818` | `-0.0738` | `-0.0143` | 0 |
| `O6-ManualAdamW-lr-low` | `0.6315` | `0.5901` | `-0.0666` | `-0.0072` | 0 |
| `O7-ManualAdamW-lr-high` | `0.6213` | `0.5866` | `-0.0768` | `-0.0174` | 0 |
| `O10-label-smoothing-diagnostic` | `0.6050` | `0.5668` | `-0.0931` | `-0.0336` | 0 |

未实现：

```text
O4-ManualAdamW-gradclip
O5-ManualAdamW-warmup-cosine
O8-ManualAdanLite-beta-tuned
O9-residual-scale-warmup
O11-edge-noise-diagnostic
```

P4 判断：

- 本轮真实尝试了 optimizer/regularization repair，但没有一个 recipe 比 `T3+ManualAdamW` 提升 `>=0.02` val acc。
- label smoothing 在当前设置下最差，val acc 降到 `0.6050`。
- H2 的“简单 optimizer/regularization 即可修复 task gap”不成立。

## 8. P5 Cross-Group Expressivity Diagnostic

P5 baseline：

| candidate | memory ratio mean | step ratio mean | val acc | test acc | val gap vs MLP |
|---|---:|---:|---:|---:|---:|
| `X0-T3-baseline` | `1.0312` | `0.8436` | `0.6387` | `0.6018` | `-0.0595` |

未实现：

```text
X1-static-group-shuffle
X2-crossgroup-lite-r1-edge-owned
X3-crossgroup-lite-r2-edge-owned
X4-g16-g8-hybrid-one-layer
X5-late-phase-crossgroup-residual
X6-groupwise-temperature-scaling-edge-owned
X7-forbidden-nonKAN-head-control
```

P5 判断：

- v7.0 尚未实现真实 cross-group expressivity repair，因此不能声称 crossgroup 路线失败。
- 但基于 P3/P4，当前 T3 grouped-g16 baseline 的 task gap 仍为主要 blocker。

## 9. P6-P12 Gate 状态

| stage | status | reason |
|---|---|---|
| P6 longer-budget diagnostic | `not_run` | optimizer/crossgroup repairs 未产生 task improvement，长预算留到下一轮 |
| P7 sample efficiency | `not_run` | task gap 持续，少样本诊断未打开 |
| P8 noise robustness | `not_run` | task gap 持续，鲁棒性诊断未打开 |
| P9 patch/token scaling | `not_run` | patch/token runner 本轮未实现 |
| P12 official task re-entry | `not_run` | 没有 S1/S0 memory candidate，official task gate 关闭 |

这部分不能解读成对应方向失败；准确说法是 v7.0 本轮已经在 P3/P4 看到 task gap 和 optimizer repair 无效，因此后续更大范围诊断没有打开或尚未实现。

## 10. Failure Table

`failure_table.csv` 失败统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F0_not_implemented_or_gated` | 19 | 未实现的 memory/crossgroup/optimizer/patch repairs |
| `F4_s1_memory_margin_fail` | 11 | memory 仍未达 `<1.00` S1 gate |
| `F5_task_gap_generalization` | 11 | task val gap 仍明显落后 MLP |
| `F6_task_gap_optimizer` | 7 | optimizer recipes 未带来有效提升 |
| `F15_diagnostic_task_gated` | 4 | P6/P7/P8/P9 diagnostic 正确 gate |
| `F14_official_task_gated` | 1 | P12 official task 正确 gate |

没有出现：

- fake data failure
- proxy row failure
- gradient correctness failure
- artifact missing failure

## 11. Route Decision

`route_decision.json`：

```json
{
  "route": "R4-S2-TaskGapPersists",
  "best_candidate": "O0-T3-ManualAdamW-v620",
  "best_family": "T3-launch-fused-step+optimizer-diagnostic",
  "best_memory_ratio": 1.0311948156844268,
  "best_step_ratio": 0.8435986563176388,
  "best_backward_ratio": 0.7779734173508558,
  "best_forward_ratio": 1.1148492062918605,
  "survivor_type": "B3",
  "best_val_acc": 0.638671875,
  "best_test_acc": 0.6017795138888888,
  "val_acc_gap_vs_MLP": -0.05946180555555558,
  "official_task_opened": false,
  "diagnostic_task_opened": true,
  "primary_blocker": "task_gap_expressivity_or_generalization",
  "next_required_implementation": "crossgroup_expressivity_repair_or_s1_memory_repair",
  "no_fake": true,
  "no_proxy": true
}
```

## 12. 结论

本轮 v7.0 支持以下真实结论：

1. v7.0 真实执行了 MNIST/Fashion-MNIST/KMNIST 上的 profiler/full-step 和 diagnostic task，不是只写 gate artifact。
2. 第一次 full run 因 gate 逻辑太保守误关 task；已自修复并重跑，最终 run 真实打开 P8/P4。
3. T3 efficiency 仍保持 S2：`T3-launch-fused-step` memory `1.0312`，step `0.9177`（P3 measured），route 使用 v6.20 locked step `0.8436` 作为 baseline score。
4. S1 memory 没有解决：memory 仍为 `1.0312 > 1.00`，official task 不能打开。
5. T3 task gap 仍在：`T3+ManualAdamW` val acc `0.6387`，MLP-autograd val acc `0.6981`，gap `-0.0595`。
6. 本轮真实尝试的 optimizer/regularization recipes 没有改善 task gap；全部低于 `T3+ManualAdamW` baseline。
7. Cross-group expressivity repair 尚未实现；不能判断该路线是否有效，但它成为下一步最直接的 task-gap repair 方向。
8. 最终 route 是 `R4-S2-TaskGapPersists`，不是 Beyond-MLP 成功。

最终一句话：

> v7.0 没有建立 Beyond-MLP candidate。T3 仍是高效 S2 kernel base，但 memory 还没到 S1，task/generalization gap 仍约 `5%-6%`；简单 optimizer/regularization repair 没修好。下一步应实现真实 cross-group expressivity repair 或继续 S1 memory repair，而不是再调已有 T3 optimizer 小参数。

## 13. 下一步建议

1. 优先实现 `X1-static-group-shuffle` 或 `X2/X3 edge-owned crossgroup-lite`，并保持 strict PureKAN / manual backward。
2. 同步继续 `M1/M2/M7` non-recompute buffer reuse，目标把 memory `1.0312` 推到 `<1.00`。
3. 不要把 `O2/O3/O6/O7/O10` 作为后续主线；这些真实测量没有提升。
4. P6 longer-budget、P7 sample efficiency、P8 robustness、P9 patch-token 可以在出现 task-gap 修复候选后再打开。
5. 后续仍需保留 no-fake/no-proxy、absolute memory metrics、task trace 与 candidate/gate 分离记录。
