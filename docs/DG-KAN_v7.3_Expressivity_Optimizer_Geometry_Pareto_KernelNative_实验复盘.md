# DG-KAN v7.3 Expressivity Optimizer Geometry Pareto KernelNative 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.3_Expressivity_Optimizer_Geometry_Pareto_KernelNative_完整实验计划.md` 的真实执行结果。v7.3 的目标不是在某个数据集上刷榜，也不是继续调 classwise loss；本轮只围绕 expressivity source、optimizer reachability、Pareto frontier 与 kernel-native blocker 做结构性实验。所有结论均来自落盘 CSV/JSON/manifest；没有 fake data、proxy rows、固定占位 ratio 或手填 pass。

## 1. 执行范围

本轮新增 runner：

```text
experiments/run_gafu_v73_real.py
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v73_real.py
```

已通过。

真实 runs：

| run | 目录 | task rows | gradient rows | efficiency rows | 目的 |
|---|---|---:|---:|---:|---|
| structural probe | `results/real_rerun_20260506/v73_structural_probe_5seed_20260506T012355Z` | 315 | 114 | 189 | P1/P2/P3/P5/P8 主结构证据 |
| A2 cached follow-up | `results/real_rerun_20260506/v73_a2_cached_followup_5seed_20260506T013700Z` | 105 | 36 | 63 | 首次失败后，自修复 A2/A2C/A2R 结构路线 |
| hidden56 follow-up | `results/real_rerun_20260506/v73_a2_hidden56_followup_5seed_20260506T014055Z` | 75 | 24 | 45 | A2C 近成功后，压缩容量检查 S2 memory 边界 |

覆盖：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..4
bench batch sizes = 128,256,512
source commit in manifest = f76ff27ea03c6fbfc01cadf24ea281c32a72b18f
```

工作区是 dirty 状态，manifest 已记录 `git_status`。本复盘不把 dirty 状态里的未提交内容当作额外实验结论。

## 2. v7.3 gate 解释

v7.3 不再把 classwise non-collapse 当硬门。`classwise_drop_max` 只保留为诊断项。

本轮采用的核心成功条件是：

| gate | 条件 |
|---|---|
| macro task | mean val gap `>= +0.02` |
| statistical | CI95 low `> 0` 且 Holm p `< 0.05` |
| test | mean test gap `>= +0.015` |
| calibration | ECE delta `<= +0.005`，NLL delta `<= +0.01` |
| GradPass | full manual-vs-autograd gradient check pass |
| KernelNative | S2/S1 efficiency pass |

## 3. 主结构 probe 结果

命令：

```bash
python experiments/run_gafu_v73_real.py \
  --out-dir results/real_rerun_20260506/v73_structural_probe_5seed_20260506T012355Z \
  --fresh \
  --device auto \
  --candidates B0,B3,C3,A1,A2,A3,H1,H2,H4,C1,G3,G5,G6,L1,L2,L3,O10,O11,O12,NP5,NP6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

v7.3 route：

```json
{
  "route": "R3-MacroExpressivityPositiveButNotKernelNative",
  "best_candidate_id": "C3",
  "macro_pass_v73": 1,
  "grad_pass": 1,
  "s2_pass": 0,
  "best_macro_gap": 0.023046875,
  "best_ci95_low": 0.015885416666666666,
  "best_holm_p": 0.0011449932263149526,
  "best_test_gap": 0.024479166666666666,
  "best_memory_ratio": "1.2239384238385154",
  "best_step_ratio": "2.795111488039678"
}
```

Task summary：

| candidate | role | val gap vs MLP | test gap vs MLP | gradient | memory ratio | step ratio |
|---|---|---:|---:|---:|---:|---:|
| `C3` | cached dense strict reference | `+0.0230` | `+0.0245` | `6/6` | `1.2239` | `2.7951` |
| `A1` | no gate poly2 | `+0.0193` | `+0.0186` | `6/6` | `1.2210` | `2.7778` |
| `A2` | linear/no-poly2 dense stack | `+0.0186` | `+0.0219` | `5/6` | `1.0242` | `1.8029` |
| `A3` | no dense cross g16 | `-0.0628` | `-0.0603` | `6/6` | `1.2029` | `2.7503` |
| `G3` | grouped g2 | `+0.0096` | `+0.0115` | `6/6` | `1.2178` | `2.7433` |
| `G5` | grouped g8 | `-0.0422` | `-0.0346` | `6/6` | `1.2050` | `2.7394` |
| `L2` | low-rank r32 | `+0.0029` | `+0.0009` | `3/6` | `1.2064` | `2.1934` |
| `NP5` | sparse interp primitive | `+0.0199` | `+0.0229` | `6/6` | `1.2210` | `2.7290` |
| `NP6` | fast rational primitive | `+0.0184` | `+0.0203` | `6/6` | `1.2210` | `2.7209` |

P1 source-of-gain：

| ablation | val gap vs MLP | gap vs `C3` | 结论 |
|---|---:|---:|---|
| `A1` no gate poly2 | `+0.0193` | `-0.0038` | gate 有贡献，但不是主因 |
| `A2` no poly2 linear stack | `+0.0186` | `-0.0044` | poly2/gate 层有贡献，但不是主因 |
| `A3` no dense cross g16 | `-0.0628` | `-0.0858` | dense cross-channel mixing 是核心表达来源 |

P3 distillation reachability：

| student | supervised control | val gap vs MLP | improvement vs control | gap vs C3 | 判断 |
|---|---|---:|---:|---:|---|
| `O10` | `G3` | `+0.0182` | `+0.0086` | `-0.0048` | distill 有帮助，但未达到 `+0.02` |
| `O11` | `G5` | `-0.0376` | `+0.0046` | `-0.0607` | g8 约束仍明显损失表达力 |
| `O12` | `L2` | `+0.0057` | `+0.0029` | `-0.0173` | low-rank distill 不能追上 C3 |

注意：`O10/O11/O12` 的 distillation task rows 是真实测量，但本轮 distill gradient checker 未通过完整 gate，因此不能作为成功 candidate。

## 4. 第一次失败后的自修复：A2 cached/no-recompute

主结构 probe 的失败点是：`C3` task/grad 过，但效率远不进 S2；`A2` memory 很好，但 task 略低且 gradient 有一条 marginal fail。因此新增结构性 candidates：

| candidate | 修改 |
|---|---|
| `A1C` | `A1` 的 cached hidden-y 版本 |
| `A2C` | `A2` 的 cached hidden-y 版本 |
| `A2R` | `A2C` 加 rbf-poly-exp KAN head |

命令：

```bash
python experiments/run_gafu_v73_real.py \
  --out-dir results/real_rerun_20260506/v73_a2_cached_followup_5seed_20260506T013700Z \
  --fresh \
  --device auto \
  --candidates B0,C3,A1,A1C,A2,A2C,A2R \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

结果：

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory ratio | step ratio | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0230` | `+0.0159` | `3.48e-04` | `+0.0245` | `-0.0018` | `-0.0995` | `6/6` | `1.3069` | `2.6483` | `0/9` |
| `A1C` | `+0.0203` | `+0.0138` | `1.28e-03` | `+0.0255` | `+0.0005` | `-0.0962` | `5/6` | `1.2131` | `1.9094` | `0/9` |
| `A2` | `+0.0186` | `+0.0096` | `2.51e-02` | `+0.0219` | `+0.0015` | `-0.0945` | `5/6` | `1.0330` | `1.1966` | `5/9` |
| `A2C` | `+0.0223` | `+0.0158` | `8.09e-04` | `+0.0254` | `+0.0034` | `-0.0968` | `5/6` | `1.0368` | `1.2370` | `5/9` |
| `A2R` | `+0.0217` | `+0.0128` | `7.34e-03` | `+0.0279` | `-0.0034` | `-0.0955` | `6/6` | `1.1694` | `1.6724` | `0/9` |

判断：

- `A2C` 是最接近 v7.3 成功的新结构：macro/test/calibration 都过，平均 memory/step 也明显优于 `C3`。
- 但 `A2C` 不能记为成功：`GradPass=5/6`，Fashion-MNIST batch 8 的 max relerr `1.2289e-04 > 1e-4`；S2 也只有 `5/9` shapes，通过不了 full S2。
- `A2R` 修复了 gradient gate 并保持 task，但 memory `1.1694`、step `1.6724`，效率不进 S2。
- `A2` 平均 step/memory 很强，但 macro gap `+0.0186 < +0.02`，且同样 `5/6` gradient。

因此第一次自修复失败，但给出了新的 Pareto 边界：`A2C` 说明“linear dense stack + KAN head”可能是接近 kernel-native 的骨架，问题剩在 gradient numerical gate 与 batch-512 memory shape。

## 5. 第二次自修复：hidden56 capacity boundary

为检查 `A2C` 的 batch-512 memory failure 是否能通过轻度容量压缩闭合，继续跑 hidden-dim 56。该实验仍是结构性 capacity/memory probe，不是数据集调参。

命令：

```bash
python experiments/run_gafu_v73_real.py \
  --out-dir results/real_rerun_20260506/v73_a2_hidden56_followup_5seed_20260506T014055Z \
  --fresh \
  --device auto \
  --hidden-dim 56 \
  --candidates B0,C3,A2,A2C,A2R \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

结果：

| candidate | val gap | CI95 low | Holm p | test gap | GradPass | memory ratio | step ratio | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3` | `+0.0133` | `+0.0068` | `1.91e-02` | `+0.0201` | `4/6` | `1.3042` | `2.8992` | `0/9` |
| `A2` | `+0.0064` | `-0.0022` | `1.0` | `+0.0174` | `5/6` | `1.0279` | `1.5263` | `6/9` |
| `A2C` | `+0.0107` | `+0.0030` | `1.51e-01` | `+0.0173` | `5/6` | `1.0312` | `1.3092` | `6/9` |
| `A2R` | `+0.0176` | `+0.0094` | `1.24e-02` | `+0.0133` | `6/6` | `1.1422` | `1.5540` | `3/9` |

判断：

- hidden56 没有闭合目标。
- `A2C` 的 efficiency 更接近 S2，但 macro gap 降到 `+0.0107`，表达力丢失。
- `A2R` gradient 过，但 macro gap `+0.0176`、test gap `+0.0133` 都不满足 v7.3 task gate，且 efficiency 仍不过 S2。
- 容量压缩不能作为当前闭合路线。

## 6. No-fake / No-proxy 审计

| run | full audit rows checked | v73 audit rows checked | fake/proxy nonzero |
|---|---:|---:|---:|
| structural probe | 5277 | 212 | 0 |
| A2 cached follow-up | 1777 | 66 | 0 |
| hidden56 follow-up | 1277 | 44 | 0 |

## 7. 关键 Hash

| run | file | SHA256 |
|---|---|---|
| structural | `run_manifest.json` | `9ba9bd2b404f8920c04f0968c08e4082bc489e9f0b9e325f7a66b6529841a187` |
| structural | `v73_route_decision.json` | `b0ab2bce02db29b220f59ccc3ba311c43994e22ab8e3117e9d21ab7bf6786c90` |
| structural | `p9_task_summary.csv` | `b6bbe0c00f1bf5d820e5ad6bc7dc3a38613349392de1a23504387f23a8d9e2c2` |
| structural | `p1_significance_audit.csv` | `9fa717955af65917ec49e7dbc5b55dbb57c142fbabbeac17fc906d4a47457ee5` |
| structural | `p2_full_gradient_correctness.csv` | `4b1c86b30d72c8e567f9500c31bdc56a01507d722d226dcc5cff7936a7c5fdba` |
| structural | `p10_efficiency_summary.csv` | `2c5ae37adb148a6d37e562a14aa70fe64aeae23278e8c71719433dee3bd21e9f` |
| A2 cached | `run_manifest.json` | `8f92f4e81f9b3cbf13d56356d7de5f489870b0a42cda08ef93015e27a4f56535` |
| A2 cached | `v73_route_decision.json` | `50dd5b059e611314a86ecb20114d4bd32fb21de170738d33d74d35b709d5a85b` |
| A2 cached | `p9_task_summary.csv` | `11e747c715026ea83facd7742251c13d8a0966d7d49e3f7026798fd0b7c5bc27` |
| A2 cached | `p1_significance_audit.csv` | `7adb24daec121526a3a230613b39df752a911b2406a9dde3f77925afef41b5e6` |
| A2 cached | `p2_full_gradient_correctness.csv` | `3f9a17f8eab4571ac1b5c4b09eb7b55326c4306e45eb083e161e203fcca78e96` |
| A2 cached | `p10_efficiency_summary.csv` | `20c1024da4d9380b1ae43eb6b6823c1825903672ad20c950e19edc0912932814` |
| hidden56 | `run_manifest.json` | `c0832f438da77b463b11dfe74e0419eb4acba45db2c3601eddddea04ba4f7a7a` |
| hidden56 | `v73_route_decision.json` | `c4c733a3c7267aa06623c68a96b7fae76057990f4bb54da5917d3bbc8a63716c` |
| hidden56 | `p9_task_summary.csv` | `1304f2897e81a28cd58813a8c483b089a22a3abcfa5c55bd12de4b5502d904eb` |
| hidden56 | `p1_significance_audit.csv` | `70192ab959366e62511ead604b4fc4efe638fbf8943265f43f383b0636acf822` |
| hidden56 | `p2_full_gradient_correctness.csv` | `94537805d1ec92c8b3d9cdcf92698d9efa325cedb5a8f2a30a0b1225648c6917` |
| hidden56 | `p10_efficiency_summary.csv` | `bfc606bbf7bbaaea6ec1f384bbc7bed628503c17257ddd51504dc154b2e3b961` |

## 8. 最终结论

v7.3 到目前没有达成完整目标。

真实成立的结论：

1. `C3` 在 v7.3 口径下已经是 macro expressivity positive：val gap `+0.0230`，CI95 low `+0.0159`，Holm p `3.48e-04` 到 `1.14e-03` 量级，test gap `+0.0245`，GradPass `6/6`。
2. `C3` 失败点不是 classwise hard gate，而是 kernel-native efficiency：memory/step 仍远高于 S2。
3. P1 消融说明主要表达来源是 dense cross-channel mixing。去掉 gate 或 poly2 只损失约 `0.004`，去掉 dense cross 直接从 `+0.0230` 掉到 `-0.0628`。
4. Distillation 能把 `G3` 从 `+0.0096` 推到 `+0.0182`，但没有达到 `+0.02`，且 distill gradient checker 当前未过，不能算 optimizer reachability 成功。
5. `NP5/NP6` 新 primitive diagnostic 仍未超过 `+0.02`，也不进 S2。
6. 自修复后的 `A2C` 是最接近 Pareto 闭合的新结构：macro gap `+0.0223`，test gap `+0.0254`，ECE/NLL 过，平均 memory `1.0368`、step `1.2370`。但它只有 `GradPass=5/6`，且 S2 只有 `5/9` shapes，因此不能记为成功。
7. hidden56 capacity compression 不能解决问题：效率更近，但表达力明显下降。

最终一句话：

> v7.3 证明了目标不该继续是 classwise 调参或刷榜；真正的结构结论是 dense cross-channel expressivity 很关键，`A2C` 暴露了一个接近 S2 的线性 dense-stack + KAN-head Pareto 方向，但当前还差 full gradient gate 与 batch-512 S2 memory shape。下一步应做真正 materialization-free / fused linear-dense-KAN backward，并把 `A2C` 的 marginal gradient failure 作为实现级 correctness blocker 处理，而不是继续调 loss 或采样。
