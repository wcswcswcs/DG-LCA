# DG-KAN v8.2 CE-Only KC6/KF4 Architecture-System Convergence 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.2_CEOnly_KC6_KF4_ArchitectureSystemConvergence_完整实验计划.md` 的本轮真实执行结果。所有数值只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填成功结论。本轮继续严格遵守：CE-only、no external teacher、no self-teacher、no loss modification、no sampler/class weight、no CPU offload。

## 1. 是否全部完成

没有全部完成。

已完成：

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 contract / registry | completed | 新增 `experiments/run_gafu_v82_real.py`，落盘 v8.2 registry / no-teacher-no-loss contract |
| P1 reproduction / co-selection | completed | KC6/KF4/KF5/KF6 与新增 KF7-KF10 同协议 5-seed full-grid |
| P2 trajectory divergence | completed | 从真实 `p9_task_trace.csv` 生成 `p2_trajectory_divergence.csv` |
| P3 factor isolation | completed | 分离 full-ATen、upper-only ATen、lower-only ATen、addcdiv 等因素 |
| P4/P5 GPU live-set / allocator sensitivity | completed | 复用真实 CUDA allocated peak，生成 v8.2 liveset/allocator artifact |
| P6/P7 repair probes | completed | 新增 KF7-KF10；追加 KF10 hidden60/58/56 memory bracket |
| P8 official co-selection | completed | `p8_official_coselection.csv` 已落盘 |

未完成或未打开：

| 阶段 | 状态 | 原因 |
|---|---|---|
| P9 time accounting | not_run | 没有单一 candidate 同时 macro + FullGridS2 |
| P10 phase-mapped profiler | not_run | minimum success 未达成 |
| P11 S1 memory package | not_run | minimum success 未达成 |
| 10-seed final confirmation | not_run | 5-seed 未出现可确认的 merged survivor |

## 2. 本轮代码

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v80_real.py` | 新增 `KF7-KF10` CE-only GPU-native mixed ATen SiLU backward candidates |
| `experiments/run_gafu_v81_real.py` | 将 `KF7-KF10` 纳入 v8.1/v8.2 contract 与 GPU live allocator 支持 |
| `experiments/run_gafu_v82_real.py` | 新增 v8.2 runner，复用真实 measured path 并写 v8.2 convergence artifacts |

新增候选：

| candidate | 改动 | update | CPU offload |
|---|---|---|---:|
| `KF7` | upper nonlinear layer only 使用 ATen fused SiLU backward | default AdamW | 0 |
| `KF8` | upper nonlinear layer only 使用 ATen fused SiLU backward | AdamW addcdiv | 0 |
| `KF9` | lower nonlinear layer only 使用 ATen fused SiLU backward | default AdamW | 0 |
| `KF10` | lower nonlinear layer only 使用 ATen fused SiLU backward | AdamW addcdiv | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

## 3. Smoke

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/tmp_v82_kf7_kf10_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,KF7,KF8,KF9,KF10 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

结果：`KF7-KF10` task / grad / bench path 可执行；`v82_provenance_audit.csv` rows checked `116`，fake/proxy nonzero `0`。Smoke 只验证代码路径，不作为正式成功结论。

## 4. 主 run：KC6/KF4/KF5/KF6/KF7-KF10 5-seed co-selection

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kc6_kf4_convergence_mixed_5seed_20260508T060000Z \
  --fresh \
  --device auto \
  --candidates B0,KC6,KF4,KF5,KF6,KF7,KF8,KF9,KF10 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route:

```json
{
  "route": "R4-QualitySystemSplitNotMerged",
  "best_official_candidate_id": "KC6",
  "quality_best_candidate_id": "KC6",
  "system_best_candidate_id": "KF4",
  "success_v82_minimum": 0
}
```

Official co-selection:

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | macro pass | S2 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KC6` | `+0.024349` | `+0.015234` | `3.25e-03` | `+0.023177` | 1 | `1.053081` | `1.513245` | `5/9` | 1 | 0 |
| `KF10` | `+0.020313` | `+0.013411` | `1.89e-03` | `+0.020703` | 1 | `1.053081` | `1.388334` | `6/9` | 1 | 0 |
| `KF9` | `+0.019661` | `+0.012760` | `3.37e-03` | `+0.021745` | 1 | `1.053081` | `1.420783` | `6/9` | 0 | 0 |
| `KF8` | `+0.019010` | `+0.010807` | `1.98e-02` | `+0.018880` | 1 | `1.053081` | `1.391516` | `6/9` | 0 | 0 |
| `KF7` | `+0.018620` | `+0.011198` | `9.98e-03` | `+0.024089` | 1 | `1.053081` | `1.415068` | `6/9` | 0 | 0 |
| `KF4` | `+0.018359` | `+0.012109` | `2.47e-03` | `+0.027734` | 1 | `1.046778` | `1.339375` | `9/9` | 0 | 1 |
| `KF5` | `+0.018229` | `+0.011328` | `9.32e-03` | `+0.023177` | 1 | `1.053081` | `1.423581` | `6/9` | 0 | 0 |
| `KF6` | `+0.018099` | `+0.010417` | `9.32e-03` | `+0.023438` | 1 | `1.053081` | `1.395562` | `6/9` | 0 | 0 |

判断：

- `KF10` 是本轮新增的最有价值中间点：它用 lower-only ATen + addcdiv 重新跨过 macro gate，并把 step 压回 S2 内。
- 但 `KF10` memory max 仍为 `1.053081`，和 `KC6` 同形，FullGridS2 仍失败。
- `KF4` 仍是系统 best：FullGridS2 `9/9`，但 macro 未过 gate。
- 因此 v8.2 主 run 仍是 quality/system split，没有 minimum success。

No-fake audit：

```text
rows_checked = 6250
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

## 5. Dataset-level 差异

Dataset gap:

| candidate | Fashion-MNIST | KMNIST | MNIST | macro |
|---|---:|---:|---:|---:|
| `KC6` | `+0.018750` | `+0.026953` | `+0.027344` | `+0.024349` |
| `KF10` | `+0.011719` | `+0.021875` | `+0.027344` | `+0.020313` |
| `KF9` | `+0.016797` | `+0.018750` | `+0.023438` | `+0.019661` |
| `KF4` | `+0.014844` | `+0.014844` | `+0.025391` | `+0.018359` |

判断：

- `KF10` 完全保住了 KC6 的 MNIST gap。
- `KF10` 相比 KC6 的主要损失仍来自 KMNIST 与 Fashion-MNIST，尤其 KMNIST 从 `+0.026953` 降到 `+0.021875`。
- lower-only ATen 明显优于 full ATen：`KF10` KMNIST gap 比 `KF4` 高 `+0.007031`。

## 6. Trajectory divergence

`p2_trajectory_divergence.csv` 基于真实 `p9_task_trace.csv`。在 step 240 的 mean val acc delta vs KC6：

| candidate | Fashion-MNIST | KMNIST | MNIST |
|---|---:|---:|---:|
| `KF10` | `-0.007031` | `-0.005078` | `0.000000` |
| `KF4` | `-0.003906` | `-0.012109` | `-0.001953` |
| `KF9` | `-0.001953` | `-0.008203` | `-0.003906` |

判断：

- `KF10` 对 KC6 的 late-step divergence 不再是全局 collapse；MNIST 已对齐。
- `KF4` 的最大 late-step loss 仍在 KMNIST。
- 当前差异更像 dataset-sensitive long-trajectory divergence，不是单步 gradient bug。

## 7. GPU live-set / allocator

MNIST bs512 allocator sensitivity：

| candidate | top peak MB | top phase | stack peak MB | update peak MB | delta vs KC6 |
|---|---:|---|---:|---:|---:|
| `KC6` | `21.1782` | update | `20.8853` | `21.1782` | `0.0000` |
| `KF10` | `20.9556` | update | `20.8853` | `20.9556` | `-0.2227 MB` |
| `KF4` | `20.9556` | update | `20.7603` | `20.9556` | `-0.2227 MB` |

判断：

- `KF10` 已经继承了 addcdiv 的 update peak reduction，但 stack backward peak 仍和 KC6 一样。
- `KF4` 的 S2 额外来自 full ATen fused lowering stack backward peak。
- 这解释了为什么 KF10 step 过线但 memory 仍卡在 `1.053081`。

## 8. KF10 hidden memory bracket

### 8.1 hidden60

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kf10_hidden60_memory_repair_5seed_20260508T064500Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,KF10 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Result:

| candidate | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | macro pass | S2 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KF10 hidden60` | `+0.023177` | `+0.015885` | `5.95e-05` | `+0.022786` | `1.051299` | `1.639600` | `6/9` | 1 | 0 |

判断：hidden60 保住 macro，但 memory 与 step 仍未闭合 S2。

### 8.2 hidden58 / hidden56

| run | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | macro pass | S2 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KF10 hidden58` | `+0.018750` | `+0.013021` | `9.08e-05` | `+0.023828` | `1.050443` | `1.372626` | `6/9` | 0 | 0 |
| `KF10 hidden56` | `+0.012891` | `+0.004297` | `4.24e-02` | `+0.017708` | `1.049662` | `1.360124` | `9/9` | 0 | 1 |

判断：

- hidden58 几乎碰到 memory gate，但 macro 已经掉出 `+0.0200`。
- hidden56 进入 FullGridS2，但 macro 大幅下降。
- 简单压 hidden 不是合流解。

No-fake audit:

| run | rows checked | fake/proxy nonzero | CPU offload |
|---|---:|---:|---:|
| `v82_kf10_hidden60_memory_repair_5seed_20260508T064500Z` | `853` | `0` | `0` |
| `v82_kf10_hidden58_memory_repair_5seed_20260508T070000Z` | `853` | `0` | `0` |
| `v82_kf10_hidden56_memory_repair_5seed_20260508T063000Z` | `853` | `0` | `0` |

## 9. Artifact 状态

主 run 落盘了 v8.2 要求的关键 artifact：

```text
candidate_registry_v82.csv
contract_no_teacher_no_loss.csv
p0_contract_audit.csv
p1_reproduction.csv
p2_trajectory_divergence.csv
p3_factor_isolation.csv
p4_gpu_liveset_attribution.csv
p5_allocator_sensitivity.csv
p6_quality_preserving_s2_repair.csv
p7_system_preserving_quality_recovery.csv
p8_official_coselection.csv
p9_time_accounting.csv
p10_phase_mapped_profiler.csv
p11_s1_memory_package.csv
route_decision.json
aggregate_decision.json
failure_table.csv
v82_manifest.json
v82_provenance_audit.csv
```

`p9_time_accounting.csv`、`p10_phase_mapped_profiler.csv`、`p11_s1_memory_package.csv` 均为 `not_run`，原因是没有单一 candidate 同时达到 macro + FullGridS2。

## 10. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v82_real.py` | `77684294140714dc601183852a74b952a0acdd4f82527d87f07e245e778075bb` |
| `experiments/run_gafu_v81_real.py` | `d87dcfeeb2ebd1e5446a78d1c8cd8c7880b78780fd46383791293acf142e817b` |
| `experiments/run_gafu_v80_real.py` | `595a564e31db3e743bf55687d4cc8dc9effb88292ee34f81a359b54097ddf436` |
| v8.2 plan | `6cc0baf34ff0b419bb10bdce73c5591fa24f102122d67df369bae89e112786b0` |
| main route | `7a9f33839e4c21be8ebb61cd39c8e6120a696d87ef238074bd0e213ad1f8ec23` |
| main provenance | `77537fcf0b9b4b327b782e058bb790fdc13bf277fa6b0b65fabd2a19ebc750e9` |
| main `p8_official_coselection.csv` | `9f438332c573b75744b1b2c31b5fde30ae30826970c405e599ec61d3b841b6f7` |
| main `p3_factor_isolation.csv` | `6b4c65669e8a0a7ec41fc87b11fcb4f1c52b8dddf4ec1cd32baede21d2bf03ca` |
| main `p2_trajectory_divergence.csv` | `5571bb6a6b470249da8e979d529ab517763f4c8636e1d64c15a062839fe79aed` |
| main `p5_allocator_sensitivity.csv` | `f5e6d340f4b4b8b69304f6ec208e3c60d48b06d859ffc1b0e7b7cba8ffb38557` |
| hidden60 route | `4503c14b59b15a90ba6e1cf453c2385433c52c6fa71afad06a426b753d28683f` |
| hidden58 route | `b97289579cd2af324db32b24824ee8205e3d81d749492d3c829b6cf6654cd816` |
| hidden56 route | `64a13ce72e2d57fb9670cc30d1284b272a249b35316fb048668d96de4bff4f4c` |

## 11. 最终结论

v8.2 本轮没有达成 minimum success：

```text
NoTeacherNoLossPass = true
StrictPass = true
GradPass = true
Best quality = KC6, macro pass, S2 fail
Best merged attempt = KF10, macro pass, S2 fail
Best system = KF4 or KF10 hidden56, S2 pass, macro fail
success_v82_minimum = false
```

机制结论：

1. `KF10` 是新增候选里最接近合流的路径：macro `+0.02031`，CI low `+0.01341`，test gap `+0.02070`，GradPass，step max `1.3883`。
2. `KF10` 没有解决 memory：memory max 仍是 `1.053081`，S2 `6/9`。
3. `KF10` 说明 lower-layer-only ATen fused backward 比 full ATen 更保质量；它把 KF4 的 KMNIST gap 从 `+0.01484` 拉回到 `+0.02188`，但仍低于 KC6 的 `+0.02695`。
4. allocator 显示 `KF10` 只降低 update peak，没有降低 stack backward peak；FullGridS2 还需要 stack-backward workspace/lifetime 修复。
5. hidden bracket 证明简单降 hidden 不是解法：hidden60 保质量但不保 S2，hidden56 保 S2但严重伤 quality。
6. 本轮没有使用 fake/proxy，没有 CPU offload，也没有 teacher/loss/sampler/class weight 修改。

最终一句话：

> v8.2 把 KC6/KF4 的分裂推进到了更清楚的中间点：`KF10` 同时保住 CE-only macro 和 step，但 memory 仍差 `0.003081`；hidden 压缩无法合流。下一步不应回到 teacher/loss/CPU，而应在 `KF10` 或 `KC6` 的表达力路径上做真正 stack-backward workspace/lifetime trim，让 stack peak 下降到 KF4 水平，同时避免 full ATen 导致的 KMNIST trajectory collapse。

## 12. 追加：GPU-side stack-backward workspace / lifetime trim

本节执行用户要求的 GPU-side stack-backward workspace/lifetime trim。约束保持不变：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
class_weight_used = 0
sampler_changed = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 12.1 代码改动

本轮新增 `KW1/KW2/KW3`。核心修复不是 CPU offload，也不是 loss/teacher trick，而是在当前 manual training path 中跳过未被使用的 stack input-gradient materialization。现有训练只消费 head/stack parameter gradients；`dL/dx0` 没有上游 trainable module 使用，因此该 tensor 是 stack-backward workspace/lifetime waste。本轮只跳过该 unused tensor，保留 parameter gradient 路径，并用现有 GradPass 检查参数梯度。

| candidate | 改动 | CPU offload | teacher/loss |
|---|---|---:|---|
| `KW1` | KC6 explicit derivative path + skip unused input-grad materialization | 0 | CE-only / no teacher |
| `KW2` | KW1 + AdamW `addcdiv_` update | 0 | CE-only / no teacher |
| `KW3` | KF10 lower-layer ATen path + skip unused input-grad materialization + AdamW `addcdiv_` update | 0 | CE-only / no teacher |

代码检查：

```text
python -m py_compile experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 12.2 Smoke

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/tmp_v82_kw_workspace_trim_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,KW1,KW2,KW3 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 只用于路径验证，不作为正式 success 结论：

| candidate | GradPass | memory ratio | step ratio | fake/proxy |
|---|---:|---:|---:|---:|
| `KW1` | 1 | `0.995003` | `1.124903` | 0 |
| `KW2` | 1 | `0.982883` | `8.575313` | 0 |
| `KW3` | 1 | `0.982883` | `1.065879` | 0 |

Smoke no-fake audit：

```text
rows_checked = 105
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 12.3 Full-grid 5-seed co-run

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_stack_backward_workspace_trim_5seed_20260508T073000Z \
  --fresh \
  --device auto \
  --candidates B0,KC6,KF4,KF10,KW1,KW2,KW3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R2-QualityPassSystemFail",
  "best_official_candidate_id": "KC6",
  "quality_best_candidate_id": "KC6",
  "system_best_candidate_id": "KW3",
  "success_v82_minimum": 0,
  "no_fake": true,
  "no_proxy": true,
  "cpu_offload_used": 0
}
```

Result：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KC6` | `+0.024349` | `+0.015234` | `2.24e-03` | `+0.023177` | 1 | `1.053081` | `1.631251` | `4/9` | quality best, S2 fail |
| `KF10` | `+0.020313` | `+0.013411` | `1.42e-03` | `+0.020703` | 1 | `1.053081` | `1.575467` | `5/9` | macro pass, S2 fail |
| `KF4` | `+0.018359` | `+0.012109` | `1.75e-03` | `+0.027734` | 1 | `1.046778` | `1.511643` | `8/9` | S2 near, macro fail |
| `KW1` | `+0.019141` | `+0.012240` | `1.42e-03` | `+0.017057` | 1 | `1.007386` | `1.622624` | `7/9` | memory repaired, macro fail |
| `KW2` | `+0.018099` | `+0.010674` | `7.53e-03` | `+0.026432` | 1 | `1.007386` | `1.582504` | `8/9` | memory repaired, macro fail |
| `KW3` | `+0.020573` | `+0.011589` | `9.38e-03` | `+0.020313` | 1 | `1.007386` | `1.564057` | `8/9` | macro pass + memory repaired, step fail |

Dataset-level gap：

| candidate | F-MNIST gap | KMNIST gap | MNIST gap | macro gap |
|---|---:|---:|---:|---:|
| `KC6` | `+0.018750` | `+0.026953` | `+0.027344` | `+0.024349` |
| `KF10` | `+0.011719` | `+0.021875` | `+0.027344` | `+0.020313` |
| `KW3` | `+0.007813` | `+0.028125` | `+0.025781` | `+0.020573` |

GPU live-set, MNIST bs512：

| candidate | top peak MB | top phase | stack peak MB | update peak MB | manual cache MB | CPU offload |
|---|---:|---|---:|---:|---:|---:|
| `KC6` | `21.1782` | update | `20.8853` | `21.1782` | `1.5313` | 0 |
| `KF10` | `20.9556` | update | `20.8853` | `20.9556` | `1.5313` | 0 |
| `KW3` | `19.8540` | head_backward | `19.8540` | `19.4243` | `1.5313` | 0 |

判断：

- `KW3` 是本轮最接近合流的候选：CE-only macro pass、GradPass、no-teacher/no-loss、memory max `1.007386`。
- stack workspace trim 真实降低了 bs512 peak：`KC6 21.1782 MB -> KW3 19.8540 MB`，约 `-1.3242 MB`。
- 但 `KW3` step max `1.564057 > 1.50`，FullGridS2 仍失败；主要 fail shape 是 Fashion-MNIST bs256。
- `KW3` 的 macro 构成与 KC6 不同：KMNIST 很强，但 F-MNIST 明显下降；不能把它写成 KC6 quality 完整保留。

No-fake audit：

```text
rows_checked = 4712
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 12.4 KW3-focused confirmation

为避免把单次 run 的一个 shape 当成结论，追加 KW3-only 高 warmup/reps confirmation：

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kw3_workspace_trim_confirm_5seed_20260508T080000Z \
  --fresh \
  --device auto \
  --candidates B0,KW3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Result：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW3` | `+0.020573` | `+0.011589` | `2.21e-03` | `+0.020313` | 1 | `1.007386` | `1.890488` | `8/9` |

Per-shape step:

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.982883` | `1.459810` | 1 |
| MNIST | 256 | `0.984568` | `1.330479` | 1 |
| MNIST | 512 | `1.007386` | `1.331403` | 1 |
| Fashion-MNIST | 128 | `0.982883` | `1.343963` | 1 |
| Fashion-MNIST | 256 | `0.984568` | `1.496859` | 1 |
| Fashion-MNIST | 512 | `1.007386` | `1.364901` | 1 |
| KMNIST | 128 | `0.982883` | `1.890488` | 0 |
| KMNIST | 256 | `0.984568` | `1.331920` | 1 |
| KMNIST | 512 | `1.007386` | `1.325974` | 1 |

判断：

- KW3 的 memory repair 复现稳定：memory max 仍为 `1.007386`。
- KW3 的 macro pass 也复现稳定：macro gap 仍为 `+0.020573`。
- 但 step instability 仍存在，且 fail shape 从主 run 的 Fashion-MNIST bs256 变成 confirmation 的 KMNIST bs128；因此不能记 FullGridS2Pass。

No-fake audit：

```text
rows_checked = 856
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 12.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `4ecaefd5a2ef8a9dd3f07f82058cb10d433c124a9517f31783ef839d6dea4896` |
| `experiments/run_gafu_v81_real.py` | `4304e7ef10199d1d5a691dfee173c0593e6d3d8572bea4263325db33c29070dc` |
| `experiments/run_gafu_v82_real.py` | `86fc0c6935377152fc976167f2e817b086fc5958b7c8315a3510a9e365abdd22` |
| workspace main route | `e29c0394cf3965fd45b4711f7f52c73c2d0f3d7ee81eb75ec8a5bc955ee15b0d` |
| workspace main `p8_official_coselection.csv` | `d188e5153fb6e9821cff6cbd25a86233c5a90d98f3b135be15a9773121399554` |
| workspace main `p4_gpu_liveset_attribution.csv` | `2b6d503e035c213e56314fc7723698e079036fb3b9bff84b1372b20e125affb2` |
| workspace main provenance | `7935bbfe3f80e0357778fe5edf3ad71d5b668df373e0ee71918913917de87695` |
| KW3 confirm route | `ba47e38d708e0c6fea084943259f8fb13e0a19632d2d0997ecc8ea6a8302fc2a` |
| KW3 confirm `p8_official_coselection.csv` | `88ffd66a48dc2045a15e4d9f9fd71a3227d09a8d837ddeb5700128584ac6abd5` |
| KW3 confirm provenance | `45e3eb71c1f03f2a2c138fd56eeed7f96886ce209025262e6256256937681f23` |

### 12.6 更新结论

本轮 GPU-side stack-backward workspace/lifetime trim 没有达成 v8.2 minimum success：

```text
KW3:
  NoTeacherNoLossPass = true
  StrictPass = true
  GradPass = true
  TeacherFreeCEMacroPass = true
  memory gate = repaired
  FullGridS2Pass = false
```

机制结论：

1. 跳过 unused input-grad materialization 是有效的 GPU-side workspace trim：bs512 peak 从 `21.1782 MB` 降到 `19.8540 MB`。
2. `KW3` 是当前第一个同时达到 CE-only macro pass 和 memory repair 的候选：macro `+0.020573`，memory max `1.007386`。
3. 但 `KW3` 仍不满足 FullGridS2，因为 step max 在两次 run 中分别为 `1.564057` 和 `1.890488`，均超过 `1.50`。
4. 这说明当前 blocker 已经从 bs512 memory 转移到 step stability / kernel timing；不能把 v8.2 改判为 success。
5. 全程没有 CPU offload，没有 teacher/self-teacher，没有 loss/sampler/class weight 修改，也没有 fake/proxy。

更新后的最终一句话：

> v8.2 的 stack-backward workspace trim 做成了真实进展：`KW3` 过 CE-only macro，并把 memory 从 KC6 的 `1.05308` 修到 `1.00739`，但 FullGridS2 仍被 step instability 卡住。下一步应在 `KW3` 上做 step-side kernel/timing repair，而不是再做 memory/offload。

## 13. 追加：KW3 step-side cached workspace repair

本节继续上一节的真实 blocker：`KW3` 已经闭合 macro 与 memory，但 FullGridS2 被 step instability 卡住。因此新增一个更窄的 step-side 候选 `KW4`：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
class_weight_used = 0
sampler_changed = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 13.1 代码改动

`KW4` 在 `KW3` 的基础上做短生命周期 GPU workspace：在 stack backward phase 内缓存本次 backward 需要的 `x_i / y_i`，减少 prefix recompute 的重复 matmul。该 workspace 只存在于 stack backward 内，不是 CPU offload，也不是把 forward cache 常驻到训练 trace。

| candidate | 改动 | 目的 |
|---|---|---|
| `KW3` | no-input-grad + lower-only ATen + addcdiv | macro + memory anchor |
| `KW4` | `KW3` + stack-backward cached GPU workspace | 降低 backward step ratio |

代码检查：

```text
python -m py_compile experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 13.2 Smoke

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/tmp_v82_kw4_step_repair_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,KW3,KW4 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke result：

| candidate | GradPass | max relerr | memory ratio | step ratio | fake/proxy |
|---|---:|---:|---:|---:|---:|
| `KW3` | 1 | `5.05e-05` | `0.982883` | `1.667586` | 0 |
| `KW4` | 1 | `5.05e-05` | `0.982883` | `1.550159` | 0 |

Smoke 只用于路径验证，不作为 success 结论。

### 13.3 Full-grid 5-seed run

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kw4_cached_workspace_step_repair_5seed_20260508T083000Z \
  --fresh \
  --device auto \
  --candidates B0,KW3,KW4,KF4,KF10 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R4-QualitySystemSplitNotMerged",
  "best_official_candidate_id": "KW3",
  "quality_best_candidate_id": "KW3",
  "system_best_candidate_id": "KW4",
  "success_v82_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

Result：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KW3` | `+0.020573` | `+0.011589` | `7.17e-03` | `+0.020313` | 1 | `1.007386` | `1.592782` | `8/9` | macro pass, step fail |
| `KW4` | `+0.016146` | `+0.008724` | `1.14e-02` | `+0.018359` | 1 | `1.013689` | `1.494909` | `9/9` | S2 pass, macro fail |
| `KF10` | `+0.020313` | `+0.013411` | `9.46e-04` | `+0.020703` | 1 | `1.053081` | `1.594905` | `5/9` | macro pass, S2 fail |
| `KF4` | `+0.018359` | `+0.012109` | `1.19e-03` | `+0.027734` | 1 | `1.046778` | `1.545018` | `8/9` | near S2, macro fail |

KW3 vs KW4 per-shape step：

| dataset | batch | KW3 step | KW4 step | KW4 S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `1.390738` | `1.299790` | 1 |
| MNIST | 256 | `1.592782` | `1.494909` | 1 |
| MNIST | 512 | `1.303252` | `1.217543` | 1 |
| Fashion-MNIST | 128 | `1.489716` | `1.384626` | 1 |
| Fashion-MNIST | 256 | `1.304678` | `1.238055` | 1 |
| Fashion-MNIST | 512 | `1.311450` | `1.242664` | 1 |
| KMNIST | 128 | `1.352544` | `1.244675` | 1 |
| KMNIST | 256 | `1.157693` | `1.111328` | 1 |
| KMNIST | 512 | `1.205369` | `1.122237` | 1 |

判断：

- `KW4` 的 cached workspace 确实是有效 step repair：每个 shape 的 step ratio 都低于 `KW3`，FullGridS2 达到 `9/9`。
- 但 `KW4` 没有保住 macro：macro gap 从 `KW3 +0.020573` 掉到 `+0.016146`。
- 因此本轮不是 success，而是又一次 clean split：`KW3 = quality pass / S2 fail`，`KW4 = S2 pass / quality fail`。

No-fake audit：

```text
rows_checked = 2091
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 13.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `b4bdfa1404bf4cbdec50ea88e3450e95a9bc58262134991b13e0f2d8e4f939d1` |
| `experiments/run_gafu_v81_real.py` | `00a4426b0e550a7e28bb934839002760aba625897f98874194f11943ddef635e` |
| `experiments/run_gafu_v82_real.py` | `a497589118a45291e47b70a94acecd25ecad7a4d5dfba5bb0544798f58b16eca` |
| KW4 route | `8f529f4ecb33ebbd4b3b8ed40ed051a39e6f30dafd09319f73cb8922d9f52a1e` |
| KW4 `p8_official_coselection.csv` | `f77e2dd619589d4a6211ecbd6f8174d629e18ba18398df444fc8aae7511e2c7c` |
| KW4 `p10_efficiency_profiler.csv` | `3ba504e42a87f22a3e8b3666f383f4a901cec27d24de8748c88ba7de9f8c6c79` |
| KW4 provenance | `6d29ef9fb8ee9a58c72a46bad3e6907c610686a7ea942b3807049e97f764a031` |

### 13.5 更新结论

v8.2 minimum success 仍未达成：

```text
KW3:
  TeacherFreeCEMacroPass = true
  FullGridS2Pass = false

KW4:
  TeacherFreeCEMacroPass = false
  FullGridS2Pass = true
```

机制结论：

1. `KW4` 证明 step-side cached workspace 能把 FullGridS2 修到 `9/9`。
2. 但这个 repair 不保质量，macro 退到 `+0.016146`，低于 hard gate。
3. `KW3 -> KW4` 的变化说明“缓存/reuse backward activations”会改变多步训练轨迹，不能当作纯系统等价替换。
4. 当前最干净的 blocker 是 quality-preserving step repair：需要保留 `KW3` 的 recompute/trajectory 语义，同时降低 step overhead，而不是缓存 `y_i` 改训练轨迹。

更新后的最终一句话：

> `KW4` 把系统侧 S2 修成了，但丢了 CE-only macro；`KW3` 保住 macro 和 memory，却丢 step。v8.2 仍不能记 success。下一步如果继续，应做 trajectory-preserving step repair，例如只优化 kernel launch/measurement path 或做数值等价 fused recompute，而不是缓存会改变训练轨迹的 backward activations。

## 14. 追加：trajectory-preserving step repair

本节回应继续要求：保留 `KW3` 的 recompute 轨迹语义，只修 kernel launch / timing overhead 或做数值等价 fused recompute。

这里的 artifact 不是口头汇总，指 runner 实际落盘的 CSV/JSON 文件。本节所有数值只来自这些文件：

```text
results/real_rerun_20260506/tmp_v82_kw5_compiled_explicit_smoke/
results/real_rerun_20260506/v82_kw5_trajectory_preserving_step_repair_5seed_20260507T075700Z/
results/real_rerun_20260506/tmp_v82_kw6_fast_mix_view_smoke/
results/real_rerun_20260506/v82_kw6_fast_mix_view_step_repair_5seed_20260507T080000Z/
results/real_rerun_20260506/v82_kw3_kw6_trajectory_preserving_confirm_10seed_20260507T081000Z/
```

关键落盘文件包括：

```text
route_decision.json
p8_official_coselection.csv
p10_efficiency_profiler.csv
p4_gpu_liveset_attribution.csv
v82_provenance_audit.csv
```

没有从日志肉眼估数，没有补字段，没有把 smoke 或 5-seed 诊断 run 当作 10-seed official success。

### 14.1 代码改动

新增两个 CE-only / no-teacher / no-loss / no-CPU-offload 候选：

| candidate | 设计 | 目的 | 结论 |
|---|---|---|---|
| `KW5` | packed-prefix + compiled explicit SiLU derivative + no input-grad + AdamW addcdiv | 数值等价 compiled explicit recompute | compiled path 可调用，但 full-grid 变慢/变重 |
| `KW6` | packed-prefix + fast mix-view + no input-grad + Aten-lowered SiLU backward + AdamW addcdiv | 保留 recompute 语义，只减少 repeated view / launch overhead | 每个 shape step 都低于 `KW3`，但 10-seed 仍未过 S2/macro |

代码检查：

```text
python -m py_compile experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 14.2 KW5 compiled explicit smoke

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/tmp_v82_kw5_compiled_explicit_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,KW3,KW5 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 结果：

| candidate | GradPass | memory ratio | step ratio | compiled explicit available | calls | fallback |
|---|---:|---:|---:|---:|---:|---:|
| `KW5` | 1 | `0.970551` | `0.101367` | 1 | 1 | 0 |

Smoke 只证明路径可运行，不能写 official success。

No-fake audit：

```text
rows_checked = 93
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 14.3 KW5 full-grid 5-seed

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kw5_trajectory_preserving_step_repair_5seed_20260507T075700Z \
  --fresh \
  --device auto \
  --candidates B0,KW3,KW4,KW5 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R2-QualityPassSystemFail",
  "best_official_candidate_id": "KW3",
  "success_v82_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

Result：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KW3` | `+0.020573` | `+0.011589` | `6.07e-03` | `+0.020313` | 1 | `1.007386` | `1.643414` | `8/9` | macro pass, step fail |
| `KW4` | `+0.016146` | `+0.008724` | `9.34e-03` | `+0.018359` | 1 | `1.013688` | `1.546338` | `8/9` | macro fail |
| `KW5` | `+0.020313` | `+0.013408` | `1.32e-03` | `+0.028516` | 1 | `1.076713` | `1.786177` | `6/9` | macro pass, system worse |

判断：

- `KW5` 的 compiled explicit path 是真实调用，不是 fake：`compiled_explicit_silu_backward_available=1`、measured calls 非零、fallback 为 `0`。
- 但 `KW5` 没有修 step，反而把 memory max 推到 `1.076713`、step max 推到 `1.786177`。
- 因此 compiled explicit recompute 不是当前可用 repair。

No-fake audit：

```text
rows_checked = 1681
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 14.4 KW6 fast mix-view 5-seed

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kw6_fast_mix_view_step_repair_5seed_20260507T080000Z \
  --fresh \
  --device auto \
  --candidates B0,KW3,KW6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R1-CEOnlyArchitectureSystemConverged",
  "best_official_candidate_id": "KW3",
  "quality_best_candidate_id": "KW3",
  "system_best_candidate_id": "KW6",
  "success_v82_minimum": 1,
  "success_v82_formal": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

5-seed Result：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KW3` | `+0.020573` | `+0.011589` | `4.41e-03` | `+0.020313` | 1 | `1.007386` | `1.366738` | `9/9` | 5-seed minimum pass |
| `KW6` | `+0.016927` | `+0.008594` | `9.27e-03` | `+0.022786` | 1 | `1.007386` | `1.331260` | `9/9` | S2 pass, macro fail |

KW6 对 KW3 的 step repair 是真实的：5-seed full-grid 中每个 shape 的 `KW6` step ratio 都低于 `KW3`，memory ratio 完全相同。MNIST bs512 live-set 也相同：

| candidate | top phase | top peak MB | stack peak MB | update peak MB | manual cache MB | CPU offload |
|---|---|---:|---:|---:|---:|---:|
| `KW3` | head_backward | `19.8540` | `19.8540` | `19.4243` | `1.53125` | 0 |
| `KW6` | head_backward | `19.8540` | `19.8540` | `19.4243` | `1.53125` | 0 |

但这只是 5-seed，必须进入 10-seed official confirmation。

No-fake audit：

```text
rows_checked = 1269
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 14.5 KW3/KW6 10-seed official confirmation

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v82_kw3_kw6_trajectory_preserving_confirm_10seed_20260507T081000Z \
  --fresh \
  --device auto \
  --candidates B0,KW3,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route：

```json
{
  "route": "R5-NoConvergenceNearPassOrFail",
  "best_official_candidate_id": "KW3",
  "quality_best_candidate_id": "KW3",
  "system_best_candidate_id": "KW6",
  "teacher_free_ce_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v82_minimum": 0,
  "success_v82_formal": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

10-seed Result：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | MacroPass | S2Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW3` | `+0.019076` | `+0.012824` | `1.50e-05` | `+0.019596` | `-0.002270` | `-0.094589` | 1 | `1.007386` | `1.651137` | `7/9` | 0 | 0 |
| `KW6` | `+0.017773` | `+0.011847` | `1.51e-05` | `+0.021549` | `-0.005621` | `-0.093395` | 1 | `1.007386` | `1.598352` | `7/9` | 0 | 0 |

10-seed per-shape step：

| dataset | batch | KW3 step | KW6 step | KW3 S2 | KW6 S2 |
|---|---:|---:|---:|---:|---:|
| MNIST | 128 | `1.266265` | `1.216358` | 1 | 1 |
| MNIST | 256 | `1.249707` | `1.233468` | 1 | 1 |
| MNIST | 512 | `1.651137` | `1.581728` | 0 | 0 |
| Fashion-MNIST | 128 | `1.649811` | `1.598352` | 0 | 0 |
| Fashion-MNIST | 256 | `1.165068` | `1.126067` | 1 | 1 |
| Fashion-MNIST | 512 | `1.291103` | `1.259316` | 1 | 1 |
| KMNIST | 128 | `1.386907` | `1.365133` | 1 | 1 |
| KMNIST | 256 | `1.368515` | `1.301827` | 1 | 1 |
| KMNIST | 512 | `1.296338` | `1.252903` | 1 | 1 |

10-seed 判断：

- `KW6` 保持了 trajectory-preserving step repair 的方向：9/9 shapes 中 step ratio 都低于 `KW3`。
- 但 `KW6` 没有把两个失败 shape 拉进 S2：MNIST bs512 仍为 `1.581728`，Fashion-MNIST bs128 仍为 `1.598352`。
- `KW3` 的 10-seed macro 从 5-seed `+0.020573` 回落到 `+0.019076`，低于 hard gate。
- 因此 10-seed official confirmation 不通过，不能写 v8.2 minimum success。

No-fake audit：

```text
rows_checked = 2350
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 14.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `d0ac1f3bbd33390e63bf15c66db92f656fc49ae92cd46a2c689980025b55fd3a` |
| `experiments/run_gafu_v81_real.py` | `d2907dc9e13857f1c2c1c8272932d0d2275cd35ab103441131fc246d6817a468` |
| `experiments/run_gafu_v82_real.py` | `aa6b85e761f5c0033d56199347794a560418e7583cd543bd2aaa45166bcab69c` |
| KW5 route | `8a157aff29d7edb8a4a65baa4ba7bd01a7f82b2d54042e05cfe35bd83ad5804f` |
| KW5 `p8_official_coselection.csv` | `225e183ba9cd3db3e0835f5fb24707d89fbb4be1e2a3cf6e00f36e80757f8e39` |
| KW5 `p10_efficiency_profiler.csv` | `c72759aac64afa64163423ce08dbff0302b2842ed024546fc34c1eed0eab00b2` |
| KW6 5-seed route | `f68d747952cb5bb84af9bba0ff34bbcfe4e07f48382bea35137daed724f86ada` |
| KW6 5-seed `p8_official_coselection.csv` | `3a0c65c7e8d364c65d4fd55a6acf9b0afb1e3e5987d14033e259f081ad2781fb` |
| KW6 5-seed `p10_efficiency_profiler.csv` | `d9f137605e9e87ed110b0fe9ed06cce3416473ac5af75191f96c0b010c1e8eaf` |
| 10-seed route | `120ed6bc15537f79c6c5dd64e6841487730b655f62212564fbebd9f2109dfa96` |
| 10-seed `p8_official_coselection.csv` | `9cada7a19ecfe19a071fb9b1fcca1e542c9cd9cbc3cfcb8f3430942fe5a1502b` |
| 10-seed `p10_efficiency_profiler.csv` | `94bfe1a0bcd1ebf61e880963a0ae358f6de6b7392d0d275d97107684d94abb1f` |
| 10-seed `p4_gpu_liveset_attribution.csv` | `2346a438bb30a928ee83fd21632868b01ea5d89586e7d83c04fcba614bf51151` |
| 10-seed `v82_provenance_audit.csv` | `004f4cf02d7a8b73fb5d74abe010f5fafc9b0ad78a9db9eed7a80563eede084b` |

### 14.7 更新结论

本轮没有达成 v8.2 minimum success：

```text
KW3 10-seed:
  TeacherFreeCEMacroPass = false
  FullGridS2Pass = false
  GradPass = true

KW6 10-seed:
  TeacherFreeCEMacroPass = false
  FullGridS2Pass = false
  GradPass = true
```

机制结论：

1. `KW5` 证明 compiled explicit recompute path 可真实调用，但它不是有效 repair：full-grid memory/step 均变差。
2. `KW6` 是更干净的 trajectory-preserving step repair：不缓存 backward activations，不改 teacher/loss，不 CPU offload，且每个 shape step 都低于 `KW3`。
3. 但 `KW6` 的 step 修复幅度不够：10-seed 仍有 2 个 shape 超过 S2 step gate。
4. 5-seed 中 `KW3` 曾出现 minimum pass，但 10-seed confirmation 回落，因此不允许写 official success。
5. 当前 blocker 从“是否能做 trajectory-preserving step repair”推进为“step repair 幅度不够，同时 10-seed macro 没有稳过 +0.0200”。

最终一句话：

> 本轮做了不改 teacher/loss、不 CPU offload 的 trajectory-preserving step repair。`KW6` 确实降低了 `KW3` 的 step overhead，但 10-seed 下 `KW3` macro 回落到 `+0.01908`，`KW6` macro 为 `+0.01777`，且二者都只到 S2 `7/9`。所以 v8.2 仍不能记 minimum success；下一步需要同时做 10-seed macro 稳定性与更强的数值等价 step repair，不能把 5-seed 短暂闭合写成成功。
