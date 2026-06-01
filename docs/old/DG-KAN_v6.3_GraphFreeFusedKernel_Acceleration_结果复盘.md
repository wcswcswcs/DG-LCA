# DG-KAN v6.3 Graph-Free Fused Kernel Acceleration 结果复盘

> 本文档只记录 2026-05-04 重新执行的真实训练与真实 profiler 结果。旧复盘中把固定代理行写成通过的内容已废弃；本轮不接受 `--allow-fake-data`、不接受固定 mock/proxy 结论，未真实实现或未真实训练的包统一标记为 `not_run`。

## 1. 实验溯源

### 1.1 运行命令

```bash
python experiments/run_gafu_v63.py \
  --packages V6_3_ALL \
  --out-dir results/real_rerun_20260504/v63_wandb_real_all_20260504T200259Z \
  --fresh \
  --no-download \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v63-real-20260504
```

### 1.2 结果目录与在线日志

- 本地结果目录：`results/real_rerun_20260504/v63_wandb_real_all_20260504T200259Z`
- 本地主日志：`results/real_rerun_20260504/v63_wandb_real_all_20260504T200259Z/run.log`
- W&B project：<https://wandb.ai/edward20121127/DG-KAN>
- W&B run：<https://wandb.ai/edward20121127/DG-KAN/runs/khtyg6m1>
- W&B run name：`v63-real-v63_wandb_real_all_20260504T200259Z`

### 1.3 防造假约束

本轮在代码层面增加了硬约束：

- `experiments/dgkan_core.py` 中的 fake vision bundle 被禁用。
- `allow_fake_data=True` 直接抛错。
- CLI 的 `--allow-fake-data` 被隐藏并改为拒绝执行。
- `experiments/run_gafu_v63.py` 默认启用 W&B，并在训练 trace 中按 step 记录 loss、accuracy、ECE、NLL、step time 等指标。
- P5/P6 中原先的固定代理行不再被当作实验结果，统一写为 `not_run`。

因此，本复盘中的训练指标来自实际训练循环与实际 W&B 同步，不包含伪造数据。

### 1.4 关键文件校验

| 文件 | SHA256 |
|---|---|
| `run.log` | `21e901c88007b888e6c43b718f0c9256a7457889c8762a67aa53e9a4602a2bbd` |
| `p3_training_trace.csv` | `3916604694a66fec429633f1857206860824a791aadb09e26ffe736694a9a90a` |
| `p4_acceleration_trace.csv` | `563e975f300a3d2fae6ba0fe42342cc75d9e95fc4f3b496828fcb9ef49ef3748` |

### 1.5 实验 setting

本轮命令没有显式覆盖 datasets/seeds/steps，因此使用 `experiments/run_gafu_v63.py` 的 v6.3 默认设置：

| 项目 | 设置 |
|---|---|
| packages | `V6_3_ALL` |
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| seeds | `0,1,2` |
| device | `auto`，本机解析为 `cuda` |
| download | `--no-download`，只使用本地已有真实数据 |
| train/val/test size | 每个 dataset/seed 为 `1536/512/512` |
| batch size | `128` |
| eval batch size | `512` |
| hidden dim | `64` |
| depth | `2` |
| basis count | `8` |
| MLP reference optimizer | `AdamW`, lr `1e-3` |
| manual optimizer lr | 默认 `2.5e-3`，`Adan/Nesterov` 类候选为 `5e-3` |
| weight decay | manual optimizer 默认 `1e-4` |
| P1 profiler shapes | batch `128/256/512` x depth `2/4`，hidden dim `64` |
| P1 profiler reps | warmup `12`，reps `18` |
| P2 profiler reps | warmup `6`，reps `9`，由 P2 内部取 P1 设置的一半 |
| W&B log cadence | 每 `20` step 记录一次 trace，最后 step 也记录 |

关于 epoch：

- 这个 runner 的训练循环按 fixed step 运行，不是按 `epochs` 参数运行。
- P3 和 P4 都是 `120` training steps。
- 每个训练子集 `1536` 个样本，batch size `128`，所以 `12` steps 约等于 1 个 epoch。
- 因此 `120` steps 等价于大约 `10` 个 epoch。

关于 multi-seed：

- P3 是多 seed：`3 datasets x 3 seeds x (1 MLP baseline + 4 optimizer candidates) = 45` 个最终训练结果，trace 为 `45 x 6 = 270` 行。
- P4 不是多 seed：虽然全局默认 seeds 是 `0,1,2`，但 P4 当前实现固定使用 `seed=0`，因此是 `3 datasets x 1 seed x (1 MLP baseline + 9 acceleration candidates) = 30` 个最终训练结果，trace 为 `30 x 6 = 180` 行。
- 所以本复盘中 P3 结论有 3-seed 支撑；P4 结论只有 seed 0 支撑，不能当作多 seed 稳定性结论。

## 2. 产物清单

本轮共生成以下核心 CSV：

| 文件 | 行数 | 说明 |
|---|---:|---|
| `p0_baseline_contract.csv` | 7 | v6.3 基线契约与代码防护检查 |
| `p1_component_kernel_cache_profiler.csv` | 42 | component/kernel/cache profiling |
| `p2_kernel_repair_package.csv` | 17 | kernel 修复候选包 |
| `p2_kernel_repair_selection.csv` | 1 | P2 修复包选择 |
| `p3_candidate_selection.csv` | 4 | P3 optimizer 候选选择 |
| `p3_minimal_task_recipe.csv` | 45 | P3 minimal task 训练摘要 |
| `p3_training_trace.csv` | 270 | P3 每 20 step/final 训练 trace |
| `p4_acceleration_package.csv` | 30 | P4 加速候选包 |
| `p4_acceleration_trace.csv` | 180 | P4 每 20 step/final 训练 trace |
| `p5_lightsmooth_compatibility.csv` | 1 | not_run |
| `p6_functional_no_autograd_smoke.csv` | 1 | not_run |
| `p7_joint_selection3.csv` | 1 | not_run |
| `p8_confirm5.csv` | 1 | not_run |
| `p9_confirm10.csv` | 1 | not_run |
| `failure_table.csv` | 46 | 失败/门禁原因汇总 |

W&B 中可直接监控的训练指标路径示例：

- `trace/latest/train_loss`
- `trace/latest/val_loss`
- `trace/latest/test_acc`
- `trace/P3/MNIST/seed0/DWM2-poly2-compiled/ManualAdam/train_loss`
- `trace/P3/MNIST/seed0/DWM2-poly2-compiled/ManualAdam/val_acc`
- `trace/P4/KMNIST/seed0/DWM2-poly2-compiled/A2-ManualAdanLite/train_loss`

## 3. 分包结果

## P0. 基线契约

P0 通过，7/7 项检查通过。

结论：v6.3 runner、核心构件和本轮 no-fake 防护具备基本可运行性。

## P1. Component/Kernel/Cache Profiler

P1 实际 profiling 生成 42 行记录，同时暴露多个性能瓶颈：

| 失败类型 | 次数 | 含义 |
|---|---:|---|
| `F1_kernel_launch_overhead` | 16 | kernel launch 开销仍然是主要瓶颈 |
| `F2_transform_cost` | 8 | transform 成本明显 |
| `F3_cache_memory` | 1 | cache memory 存在局部压力 |

分析：

- P1 没有证明 graph-free fused path 已经稳定加速。
- profiling 结果显示当前瓶颈并不只在单个 kernel，而是 kernel launch、transform、cache memory 三类问题同时存在。
- 后续应优先缩小到最小可复现实例，分别处理 launch fusion 与 transform reuse，而不是直接进入更大模型训练。

## P2. Kernel Repair Package

P2 选择结果：

| selected_id | step_ratio | bmem_ratio | 结论 |
|---|---:|---:|---|
| `DWM2-poly2-compiled` | 0.833302 | 1.144985 | 通过 P2 选择 |

分析：

- `DWM2-poly2-compiled` 在 step time 上相对基线有改善，`step_ratio < 1`。
- 但 `bmem_ratio = 1.144985`，说明显存/内存占用没有同步改善。
- 该候选可以进入 P3/P4 真实训练验证，但不能直接宣称整体方案完成加速。

## P3. Minimal Task Candidate Selection

P3 对 4 个优化器候选进行了真实训练验证。

| method | dataset_pass_count | mean_acc | time_win_count | p3_pass |
|---|---:|---:|---:|---:|
| `ManualAdanLite` | 3 | 0.815755 | 5 | 1 |
| `ManualAdam` | 3 | 0.813802 | 4 | 1 |
| `ManualAdamW` | 3 | 0.813802 | 4 | 1 |
| `ManualNesterovAdam` | 3 | 0.806424 | 0 | 0 |

聚合结果：

| method | mean_acc | mean_gap_vs_mlp | mean_auc_delta |
|---|---:|---:|---:|
| `ManualAdam` | 0.813802 | -0.021918 | 0.019648 |
| `ManualAdamW` | 0.813802 | -0.021918 | 0.006362 |
| `ManualAdanLite` | 0.815755 | -0.023872 | -0.020309 |
| `ManualNesterovAdam` | 0.806424 | -0.014540 | -0.028699 |

分析：

- `ManualAdanLite` 的 mean accuracy 最高，且 time win count 最好，是 P3 最优候选。
- `ManualAdam` 与 `ManualAdamW` 也通过 P3，但精度略低于 `ManualAdanLite`。
- `ManualNesterovAdam` 虽然 dataset pass count 达到 3，但 time win count 为 0，最终未通过 P3。
- `mean_gap_vs_mlp` 为负数，表示这些 DG-KAN 候选在该统计口径下高于 MLP 参考；但幅度不大，仍需更大样本和更多 seed 确认。

P3 结论：`DWM2-poly2-compiled + ManualAdanLite` 是本轮最强组合，可以作为后续真实实验主线。

## P4. Acceleration Package

P4 对 9 个加速候选包进行了真实训练验证，但没有得到完整 all-dataset winner。

| package | mean_acc | gap_vs_mlp | pass_count | 结论 |
|---|---:|---:|---:|---|
| `A0-ManualAdamW` | 0.807292 | -0.023438 | 2 | partial |
| `A1-ManualNesterovAdamW` | 0.789063 | -0.005208 | 1 | failed |
| `A2-ManualAdanLite` | 0.809896 | -0.026042 | 2 | partial |
| `A3-ManualWinLite` | 0.807943 | -0.024089 | 2 | partial |
| `A4-Lookahead-ManualAdamW` | 0.807943 | -0.024089 | 2 | partial |
| `A5-Restart-ManualAdamW` | 0.776042 | 0.007812 | 1 | failed |
| `A6-WarmupCosine-ManualAdamW` | 0.808594 | -0.024740 | 2 | partial |
| `A7 RoleWiseLR InputHeavy` | 0.565755 | 0.218099 | 0 | failed |
| `A8 RoleWiseLR OutputWarmup` | 0.472656 | 0.311198 | 0 | failed |

分析：

- P4 的最好候选仍是 `A2-ManualAdanLite`，mean_acc 最高，gap 也最好。
- 但它只在 2 个 dataset/seed 条件下通过，未达到完整通过标准。
- `A7` 和 `A8` 明显退化，应从后续候选集中移除。
- P4 failure table 中 `F6_acceleration_unstable` 出现 15 次，说明加速包还不稳定，不能写成已完成突破。

P4 结论：当前只有 partial winner，没有全量胜出包。下一步应围绕 `A2-ManualAdanLite` 做稳定性修复与更多 seed 复验。

## P5-P9. 门禁状态

| package | 状态 | 原因 |
|---|---|---|
| P5 LightSmooth compatibility | `not_run` | empirical LightSmooth audit 尚未实现，固定代理行被禁止 |
| P6 functional no-autograd smoke | `not_run` | empirical functional update smoke 尚未实现，固定代理行被禁止 |
| P7 joint selection3 | `not_run` | P4 未产生完整 winner，后续门禁不应继续宣称 |
| P8 confirm5 | `not_run` | P4 未产生完整 winner |
| P9 confirm10 | `not_run` | P4 未产生完整 winner |

这部分是本轮最重要的纠偏：P5/P6 不能再用固定数字或代理判断写成通过。它们必须等真实实现和真实训练/检查完成后才能进入结论。

## 4. 失败表分析

`failure_table.csv` 共 46 行，主要失败来源如下：

| 阶段 | failure_code | 次数 | 解释 |
|---|---|---:|---|
| P1 | `F1_kernel_launch_overhead` | 16 | kernel launch overhead 是最大问题 |
| P1 | `F2_transform_cost` | 8 | transform 开销仍需优化 |
| P1 | `F3_cache_memory` | 1 | cache memory 有一次明确失败 |
| P3 | `F4_task_underfit` | 1 | 一个 P3 候选存在任务欠拟合/未达标 |
| P4 | `F6_acceleration_unstable` | 15 | 加速包跨数据集不稳定 |
| P5-P9 | `F9_gated_not_run` | 5 | 后续包被正确门禁，没有继续伪造结论 |

核心判断：

- v6.3 当前不是“全链路成功”，而是“P2/P3 有可继续推进的真实候选，P4 仍不稳定，P5-P9 正确阻断”。
- 最值得继续投入的是 `DWM2-poly2-compiled + ManualAdanLite`。
- 最需要工程修复的是 P1 的 launch/transform/cache 问题与 P4 的跨数据集稳定性。

## 5. 结论

### 5.1 可以确认的正向结果

1. v6.3 能在 no-fake 约束下完成真实训练流程。
2. W&B 已同步真实 step-level trace，可监控每 20 step 的 `train_loss`、`val_loss`、`val_acc`、`test_acc` 等指标。
3. P2 选出了可用 kernel 修复候选：`DWM2-poly2-compiled`。
4. P3 选出了当前最优优化器候选：`ManualAdanLite`。
5. P4 中 `A2-ManualAdanLite` 是最值得继续修复的加速包。

### 5.2 不能宣称的内容

1. 不能宣称 P5 LightSmooth compatibility 已通过。
2. 不能宣称 P6 no-autograd functional smoke 已通过。
3. 不能宣称 P7/P8/P9 已完成确认。
4. 不能宣称 v6.3 已经得到稳定全数据集加速包。
5. 不能使用固定 rows、proxy rows、fake data 作为实验结论。

### 5.3 最终结论

本次真实重跑支持以下结论：

> DG-KAN v6.3 在 no-fake 模式下跑通了真实 P0-P4 实验，并通过 W&B 留下了 step-level 训练记录。当前最强路线是 `DWM2-poly2-compiled + ManualAdanLite`，但 P4 尚未形成稳定 all-dataset winner，P5-P9 必须保持 `not_run`，不能继续用代理数据包装成成功。

## 6. 下一步建议

1. 修复 P1 中的 kernel launch overhead 与 transform cost，再重跑 P1/P2。
2. 以 `DWM2-poly2-compiled + ManualAdanLite` 为唯一主线，增加 P4 的 seed 数与 dataset 覆盖。
3. 删除或隔离 `A7 RoleWiseLR InputHeavy` 与 `A8 RoleWiseLR OutputWarmup`，它们本轮明显退化。
4. 为 P5 LightSmooth compatibility 写真实 empirical audit，不能使用固定结论。
5. 为 P6 no-autograd smoke 写真实 functional update 检查，至少要验证参数更新、loss 曲线和 autograd bypass 行为。
6. 后续每次实验都必须默认启用 W&B，并保留本地 CSV、run.log、artifact hash 与 run URL。
