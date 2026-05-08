# DG-KAN v7.1 Final DensePoly2Gate KernelNative PureKAN-NG 实验复盘

> 本复盘只记录 `docs/DG-KAN_v7.1_Final_DensePoly2Gate_KernelNative_PureKANNG_完整实验计划.md` 的真实执行结果。最终主结论来自 `results/real_rerun_20260505/v71_poly2gate_kernel_final_20260505T221338Z`。所有数字来自落盘 CSV/JSON/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 最终判断

| 目标 | 结论 | 证据 |
|---|---|---|
| v7.1 runner 是否跑通 | 跑通 | `experiments/run_gafu_v71_real.py` 已生成完整 task / trace / efficiency / route artifacts |
| strict KAN head 是否可替代 linear head | 基本成立 | `H1/H2/H3/H4/H5/H6` 全部通过 basic Beyond task gate |
| basic Beyond-MLP task gate | 达成 | 最佳 strict `H3` val gap vs MLP `+0.0189`，3/3 datasets within 1% MLP 且 3/3 `KAN >= MLP` |
| S2 efficiency gate | 未达成 | 最佳 strict `H3` memory ratio `1.3031`、step ratio `3.6324`；所有 candidate `survivor=FAIL` |
| v7.1 最低成功标准 | 未达成 | 缺 S2Pass；本轮也未做 full grad relerr/cos gate，只记录 one-step finite/pass |

最终路线：

```json
{
  "route": "R2-StrictTaskPassEfficiencyFail",
  "best_candidate_id": "H3",
  "best_candidate": "D3-dense-poly2-gate-d3-KAN-head-rbf-poly-exp",
  "primary_blocker": "efficiency_kernelization",
  "no_fake": true,
  "no_proxy": true
}
```

最终一句话：

> v7.1 成功把 dense D3 的 Beyond-MLP accuracy 从 linear head 迁移到 strict KAN head，但仍没有得到 kernel-native / S2 candidate。当前 blocker 已明确转为 materialization-free kernelization 和低内存 structured bridge。

## 2. 实验溯源

最终有效命令：

```bash
python experiments/run_gafu_v71_real.py \
  --out-dir results/real_rerun_20260505/v71_poly2gate_kernel_final_20260505T221338Z \
  --fresh \
  --device auto \
  --candidates B0,B3,H1,H2,H3,H4,H5,H6,K0,K0s,K0q,K0qs,K1,K2,K3,K4
```

关键设置：

| 项目 | 值 |
|---|---|
| datasets | `MNIST,Fashion-MNIST,KMNIST` |
| seeds | `0,1,2` |
| train/val/test size | `1536/512/512` |
| hidden dim | `64` |
| basis count | `8` |
| task steps | `240` |
| trace interval | `20` |
| bench batch sizes | `128,256,512` |
| bench warmup/reps | `5/30` |
| source commit | `f76ff27ea03c6fbfc01cadf24ea281c32a72b18f` |

不用于最终结论但用于自修复的 runs：

| run | 用途 | 结果 |
|---|---|---|
| `/tmp/v71_smoke`, `/tmp/v71_smoke2` | runner smoke | 跑通真实 loader / manual path / bench path |
| `v71_poly2gate_kernel_all_20260505T220412Z` | 第一轮 D3 strict heads + g8/g16 grouped | strict head 任务过，efficiency fail |
| `v71_poly2gate_kernel_repair1_20260505T220956Z` | 失败后自修复：depth-2 strict heads + g2/g4 grouped/shuffle | depth-2 和 g2 task 过，但 S2 仍 fail |
| `v71_poly2gate_kernel_final_20260505T221338Z` | ECE 修正后的 combined final run | 本复盘唯一主结论来源 |

过程中的真实自修复：

1. 第一轮发现 strict KAN head 可过 task，但 dense efficiency 失败。
2. 随后补 `H4/H5/H6` depth-2 strict heads，试图降低 dense cost。
3. 同时补 `K0/K0s/K0q/K0qs` g2/g4 grouped/shuffle，试图保留 task 且降低 live set。
4. 发现 ECE 计算调用传入概率而非 logits，修正 runner 并重跑 final combined run；最终 ECE 只引用修正后的 final run。
5. 修正 route 派生逻辑：strict-task 候选按 `basic_beyond_pass`、`val_gap`、`test_acc` 排序，route 从 CSV 重新生成。

## 3. 产物清单

| artifact | 行数 | 状态 |
|---|---:|---|
| `candidate_registry.csv` | 16 | 真实生成 |
| `p1_p2_p3_task.csv` | 144 | 真实生成 |
| `p1_p2_p3_task_trace.csv` | 1728 | 真实生成 |
| `p1_p2_p3_task_summary.csv` | 16 | 真实生成 |
| `p5_efficiency_detail.csv` | 144 | 真实生成 |
| `p5_efficiency_summary.csv` | 15 | 真实生成 |
| `failure_table.csv` | 22 | 真实生成 |
| `provenance_audit.csv` | 1 | 真实生成 |
| `route_decision.json` | - | 真实生成，route 已按 CSV 派生修正 |
| `run_manifest.json` | - | 真实生成 |

关键 SHA256：

| 文件 | SHA256 |
|---|---|
| `run_manifest.json` | `d71fdfab3713a465dea8c09d7ed5349933b04fc6b845d8c95886898cbeb7892b` |
| `candidate_registry.csv` | `57fb31ffa788505901406feee4a6501194ec219f8f612ce346667e567a45285e` |
| `p1_p2_p3_task.csv` | `0fb59ea09faf67106ac46975279054f0ee9cb8faa695290de269e69243cfbfc8` |
| `p1_p2_p3_task_summary.csv` | `a5bc3837f43c5c5b08e9e38d13c61c9c5d998a3eef7ac832c9f245598743c6d0` |
| `p5_efficiency_detail.csv` | `56dee9a0864e570cb5ceaea53cc1437ea631401e478fce756c335b4487070e7c` |
| `p5_efficiency_summary.csv` | `e97c3f6496e901e9f90e68855a89fe6a756191ad9ed5cbdb49d5776857102508` |
| `failure_table.csv` | `49a9a6603b9be9a0fc7fefe212b078af4015fef91d23692d17fdfe17c55c5bb6` |
| `route_decision.json` | `46eb5dc0c7f6cb3ebb950002218b3ddc4a55daa3b8533ccb91210b436ba34b5b` |
| `provenance_audit.csv` | `187c4c2f1ce19554274483f27881702c62af350f6829ad5a8e27cb7b6dc5e819` |

## 4. No-Fake / No-Proxy 审计

最终目录所有 CSV 合计审计：

```text
total_csv_rows = 2086
fake_proxy_nonzero = 0
```

`provenance_audit.csv`：

| rows_checked | fake_proxy_nonzero_count | no_fake | no_proxy |
|---:|---:|---|---|
| `2086` | `0` | `true` | `true` |

代码入口中所有 dataset loader 调用均使用 `allow_fake_data=False`。未实现内容没有写 fake ratio；未跑的完整 fused Triton / materialization-free kernel 只在结论中标为下一步。

## 5. Task Gate 结果

Summary：

| candidate | family | head | val acc mean | test acc mean | val gap vs MLP | within 1% | KAN >= MLP | basic pass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `B0` | baseline | MLP | `0.8440` | `0.7951` | `0.0000` | 3 | 3 | reference |
| `B3` | dense oracle | linear | `0.8635` | `0.8153` | `+0.0195` | 3 | 3 | 1 |
| `H1` | strict head | poly2-gate | `0.8602` | `0.8121` | `+0.0163` | 3 | 3 | 1 |
| `H2` | strict head | poly2-silu | `0.8624` | `0.8136` | `+0.0184` | 3 | 3 | 1 |
| `H3` | strict head | rbf-poly-exp | `0.8628` | `0.8181` | `+0.0189` | 3 | 3 | 1 |
| `H4` | depth repair | d2 poly2-gate | `0.8609` | `0.8021` | `+0.0169` | 3 | 3 | 1 |
| `H5` | depth repair | d2 poly2-silu | `0.8585` | `0.8095` | `+0.0145` | 3 | 3 | 1 |
| `H6` | depth repair | d2 rbf-poly-exp | `0.8609` | `0.8082` | `+0.0169` | 3 | 3 | 1 |
| `K0` | grouped g2 | poly2-gate | `0.8561` | `0.8084` | `+0.0122` | 3 | 2 | 1 |
| `K0s` | grouped g2 shuffle | poly2-gate | `0.8539` | `0.8047` | `+0.0100` | 3 | 2 | 1 |
| `K0q` | grouped g4 | poly2-gate | `0.8301` | `0.7854` | `-0.0139` | 2 | 1 | 0 |
| `K0qs` | grouped g4 shuffle | poly2-gate | `0.8451` | `0.7951` | `+0.0011` | 2 | 1 | 0 |
| `K1` | grouped g8 | poly2-gate | `0.8075` | `0.7563` | `-0.0365` | 1 | 0 | 0 |
| `K2` | grouped g16 | poly2-gate | `0.7784` | `0.7257` | `-0.0655` | 0 | 0 | 0 |
| `K3` | grouped g8 shuffle | poly2-gate | `0.8212` | `0.7791` | `-0.0228` | 1 | 1 | 0 |
| `K4` | grouped g16 shuffle | poly2-gate | `0.7791` | `0.7296` | `-0.0649` | 0 | 0 | 0 |

Dataset detail for best strict `H3`:

| dataset | MLP val | H3 val | gap | H3 test |
|---|---:|---:|---:|---:|
| MNIST | `0.8919` | `0.9121` | `+0.0202` | `0.9238` |
| Fashion-MNIST | `0.8066` | `0.8275` | `+0.0208` | `0.8105` |
| KMNIST | `0.8333` | `0.8490` | `+0.0156` | `0.7201` |

判断：

- H4 假设“strict KAN readout 可以替代 linear head”在 task accuracy 上成立：`H1/H2/H3` 相比 linear oracle `B3` 只低 `0.0007-0.0033` mean val acc。
- `H3` 是 task 最好 strict candidate：val gap `+0.0189`，test acc `0.8181`。
- `K0/K0s` 说明 g2 structured bridge 能保留一部分 task gain，并通过 basic gate；但 g4/g8/g16 不能稳定闭合 KMNIST。

## 6. One-Step / Gradient 状态

所有 manual candidates 的 one-step loss 均真实下降：

| family | candidates | one-step pass |
|---|---|---:|
| dense oracle / strict heads | `B3,H1,H2,H3,H4,H5,H6` | `63/63` |
| grouped / shuffle | `K0,K0s,K0q,K0qs,K1,K2,K3,K4` | `72/72` |

但本轮没有完成 plan 中的 full gradient correctness gate：

```text
grad_relerr_max = metric_unavailable
grad_cos_min = metric_unavailable
```

因此即使某些 candidate 通过 task，也不能宣称 `GradPass`，只能说 manual one-step finite/pass 已验证。完整 v7.1 成功仍要求后续补 autograd/finite-difference gradient relerr/cos。

## 7. Efficiency 结果

P5 使用同 run、同 shape MLP denominator，覆盖：

```text
3 datasets x 3 batch sizes x (MLP + 15 manual candidates) = 144 rows
```

关键 candidates：

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `B3` dense linear oracle | `1.3027` | `3.0661` | `2.8674` | `2.5955` | FAIL |
| `H3` best strict task | `1.3031` | `3.6324` | `4.0550` | `3.0451` | FAIL |
| `H4` best depth repair efficiency | `1.2956` | `2.6287` | `2.7074` | `2.0660` | FAIL |
| `K0` grouped g2 | `1.1489` | `5.7211` | `5.7039` | `4.4637` | FAIL |
| `K0q` grouped g4 | `1.0690` | `10.0062` | `9.0897` | `7.8575` | FAIL |
| `K1` grouped g8 | `1.0430` | `18.6210` | `15.8845` | `14.6987` | FAIL |
| `K2` grouped g16 | `1.0463` | `36.0729` | `29.4507` | `28.5946` | FAIL |

判断：

- Dense strict head task pass 不是效率 pass。`H3` 的 memory `1.3031 > 1.05`，step `3.6324 > 1.50`。
- Depth repair 有效但不足：`H4` step 从 `H1` 的 `3.4919` 降到 `2.6287`，仍未进 S2；memory 仍约 `1.30`。
- Grouped g8/g16 的 memory 接近 S2，但 Python grouped implementation step 极慢，不能作为 kernel-native 成功证据。
- g2/g4 grouped 是有用 diagnostic：g2 task pass，但 memory/step 仍失败；g4 memory 更低但 task gate 不稳。

## 8. Failure Table

最终 `failure_table.csv` 统计：

| failure_type | count | 解释 |
|---|---:|---|
| `F1_oracle_nonkan_head` | 1 | `B3` linear head 是 oracle，不允许 final route |
| `F2_task_gate_fail` | 6 | g4/g8/g16 structured candidates 未过 basic task gate |
| `F4_efficiency_s2_fail` | 15 | 所有 manual candidates 都未进 S2 |

没有出现：

- fake data failure
- proxy row failure
- artifact missing failure
- finite one-step failure

## 9. 结论

本轮 v7.1 支持以下真实结论：

1. `experiments/run_gafu_v71_real.py` 已建立 v7.1 real-only targeted runner，并成功跑完 final combined run。
2. strict KAN head repair 是成功方向：`H1/H2/H3` 都保留了 dense D3 的 Beyond-MLP accuracy，且不依赖 trainable linear head。
3. 最佳 strict task candidate 是 `H3-D3-dense-poly2-gate-d3-KAN-head-rbf-poly-exp`：val acc `0.8628`，test acc `0.8181`，val gap `+0.0189`。
4. depth-2 strict repair `H4/H5/H6` 也通过 task gate，说明 dense depth 可以压一点，但效率仍远不够。
5. structured g2 bridge `K0/K0s` 通过 basic task gate，但 step ratio 仍为 `5.72/5.98`，当前 Python grouped path 不是 kernel-native 证据。
6. 所有 candidates 都没有进入 S2；最终 blocker 是 `efficiency_kernelization`。
7. 本轮没有完成 full gradient relerr/cos gate，所以不能宣称 v7.1 最低成功标准。

最终一句话：

> v7.1 已经把 Beyond-MLP task signal 从 dense linear-head oracle 推进到 strict KAN-head candidate；但仍卡在 efficiency 和完整 gradient correctness。下一步不应再证明 strict head 能不能学，而应实现 materialization-free fused dense/poly2-gate 或真正 kernel-native grouped/low-rank bridge，把 `H3/H4/K0` 的 task signal 压进 S2/S1。

## 10. 下一步建议

1. 优先对 `H4` 做 materialization-free dense poly2-gate microkernel，因为它是当前 task pass 且 step 最低的 strict dense candidate。
2. 同步保留 `H3` 作为 task-quality teacher/reference，因为它的 val/test 最好。
3. 对 `K0/K0s` 做真正 fused grouped implementation；当前 Python group loop 不能代表 kernel-native 速度。
4. 补 P4 full gradient correctness：`grad_relerr_max`、`grad_cos_min`、rollback error，否则不能进入 final route。
5. P5 live-set attribution 需要进一步拆 `basis/gate/cache/workspace`，本轮只给出 aggregate peak/time，还没有达到 plan 的 top3 memory source 完整解释。
