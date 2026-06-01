# DG-KAN v7.1 Final DensePoly2Gate KernelNative PureKAN-NG 实验复盘

> 本文件按用户指定文件名记录 `docs/DG-KAN_v7.1_Final_DensePoly2Gate_KernelNative_PureKANNG_完整实验计划.md` 的执行结果。主结论来自 `results/real_rerun_20260505/v71_poly2gate_kernel_final_20260505T221338Z`。所有数字均来自落盘 CSV/JSON/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。

## 1. 计划目标是否达成

| 目标 | 结论 | 证据 |
|---|---|---|
| v7.1 real-only runner | 达成 | `experiments/run_gafu_v71_real.py` 跑通并落盘完整 artifacts |
| strict KAN head 替代 linear head | 达成 task 侧 | `H1/H2/H3/H4/H5/H6` 全部通过 basic Beyond task gate |
| basic Beyond-MLP task gate | 达成 | 最佳 strict `H3` val gap vs MLP `+0.0189`，3/3 datasets `KAN >= MLP` |
| S2 efficiency gate | 未达成 | 最佳 strict `H3` memory ratio `1.3031`、step ratio `3.6324` |
| v7.1 最低成功标准 | 未达成 | 缺 S2Pass；full grad relerr/cos gate 未完成 |

最终判断：

> v7.1 成功把 dense D3 的 Beyond-MLP task signal 迁移到 strict KAN head，但还没有得到 kernel-native / S2 candidate。计划的完整目标未达成，主要 blocker 是 `efficiency_kernelization`。

## 2. 最终运行

```bash
python experiments/run_gafu_v71_real.py \
  --out-dir results/real_rerun_20260505/v71_poly2gate_kernel_final_20260505T221338Z \
  --fresh \
  --device auto \
  --candidates B0,B3,H1,H2,H3,H4,H5,H6,K0,K0s,K0q,K0qs,K1,K2,K3,K4
```

覆盖：

```text
task rows = 144
task trace rows = 1728
efficiency rows = 144
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
task steps = 240
bench batch sizes = 128,256,512
```

## 3. 遇到的问题与自修复

| 问题 | 自修复尝试 | 结果 |
|---|---|---|
| 第一轮 strict KAN head task 过但 efficiency fail | 增加 `H4/H5/H6` depth-2 strict heads | step 降低，但仍未进 S2 |
| structured g8/g16 task gap 明显 | 增加 `K0/K0s/K0q/K0qs` g2/g4 grouped/shuffle | g2 通过 task gate，但 efficiency 仍 fail |
| ECE 指标传入概率而非 logits | 修正 runner 并重跑 final combined run | final run 的 ECE/NLL 可引用 |
| route best 选择未按 val gap 排序 | 修正 route 派生逻辑并从 CSV 重写 `route_decision.json` | final route 指向 `H3` |

## 4. Task 关键结果

| candidate | family | head | val acc | test acc | val gap vs MLP | basic pass |
|---|---|---|---:|---:|---:|---:|
| `B0` | baseline | MLP | `0.8440` | `0.7951` | `0.0000` | reference |
| `B3` | dense oracle | linear | `0.8635` | `0.8153` | `+0.0195` | 1 |
| `H1` | strict head | poly2-gate | `0.8602` | `0.8121` | `+0.0163` | 1 |
| `H2` | strict head | poly2-silu | `0.8624` | `0.8136` | `+0.0184` | 1 |
| `H3` | strict head | rbf-poly-exp | `0.8628` | `0.8181` | `+0.0189` | 1 |
| `H4` | depth repair | d2 poly2-gate | `0.8609` | `0.8021` | `+0.0169` | 1 |
| `K0` | grouped g2 | poly2-gate | `0.8561` | `0.8084` | `+0.0122` | 1 |
| `K0s` | grouped g2 shuffle | poly2-gate | `0.8539` | `0.8047` | `+0.0100` | 1 |

最佳 strict `H3` 分 dataset：

| dataset | MLP val | H3 val | gap | H3 test |
|---|---:|---:|---:|---:|
| MNIST | `0.8919` | `0.9121` | `+0.0202` | `0.9238` |
| Fashion-MNIST | `0.8066` | `0.8275` | `+0.0208` | `0.8105` |
| KMNIST | `0.8333` | `0.8490` | `+0.0156` | `0.7201` |

## 5. Efficiency 关键结果

| candidate | memory ratio | step ratio | forward ratio | backward ratio | survivor |
|---|---:|---:|---:|---:|---|
| `B3` dense linear oracle | `1.3027` | `3.0661` | `2.8674` | `2.5955` | FAIL |
| `H3` best strict task | `1.3031` | `3.6324` | `4.0550` | `3.0451` | FAIL |
| `H4` depth repair | `1.2956` | `2.6287` | `2.7074` | `2.0660` | FAIL |
| `K0` grouped g2 | `1.1489` | `5.7211` | `5.7039` | `4.4637` | FAIL |
| `K1` grouped g8 | `1.0430` | `18.6210` | `15.8845` | `14.6987` | FAIL |
| `K2` grouped g16 | `1.0463` | `36.0729` | `29.4507` | `28.5946` | FAIL |

判断：

- `H3` task 最好，但 memory/step 都远高于 S2 gate。
- `H4` depth-2 repair 降低 step，但仍未达到 `step_ratio <= 1.50`。
- g8/g16 memory 接近 S2，但 Python grouped implementation step 极慢，不能作为 kernel-native 成功证据。

## 6. Route Decision

```json
{
  "route": "R2-StrictTaskPassEfficiencyFail",
  "best_candidate_id": "H3",
  "best_candidate": "D3-dense-poly2-gate-d3-KAN-head-rbf-poly-exp",
  "best_val_acc": 0.8628472222222222,
  "best_test_acc": 0.8181423611111112,
  "best_val_gap_vs_MLP": 0.018880208333333332,
  "basic_beyond_pass": 1,
  "strict_head": 1,
  "s2_pass": 0,
  "best_memory_ratio": 1.3031381078118751,
  "best_step_ratio": 3.63238711255002,
  "primary_blocker": "efficiency_kernelization",
  "no_fake": true,
  "no_proxy": true
}
```

## 7. No-Fake / No-Proxy 审计

最终目录所有 CSV 合计：

```text
total_csv_rows = 2086
fake_proxy_nonzero = 0
```

关键 artifact hash：

| 文件 | SHA256 |
|---|---|
| `run_manifest.json` | `d71fdfab3713a465dea8c09d7ed5349933b04fc6b845d8c95886898cbeb7892b` |
| `p1_p2_p3_task_summary.csv` | `a5bc3837f43c5c5b08e9e38d13c61c9c5d998a3eef7ac832c9f245598743c6d0` |
| `p5_efficiency_summary.csv` | `e97c3f6496e901e9f90e68855a89fe6a756191ad9ed5cbdb49d5776857102508` |
| `route_decision.json` | `46eb5dc0c7f6cb3ebb950002218b3ddc4a55daa3b8533ccb91210b436ba34b5b` |
| `provenance_audit.csv` | `187c4c2f1ce19554274483f27881702c62af350f6829ad5a8e27cb7b6dc5e819` |

## 8. 结论

本轮支持的真实结论：

1. v7.1 final 计划已执行到 targeted P1/P2/P3/P5。
2. strict KAN head repair 成功保留 Beyond-MLP task signal。
3. `H3` 是当前最佳 strict task candidate。
4. `H4` 是当前较好的 dense depth/efficiency repair，但仍未进 S2。
5. `K0/K0s` 证明 g2 structured bridge 有 task potential，但当前实现不是高效 kernel-native path。
6. 完整目标未达成，下一步必须做 materialization-free fused kernel 或真正 kernel-native grouped/low-rank bridge，并补 full gradient correctness gate。
