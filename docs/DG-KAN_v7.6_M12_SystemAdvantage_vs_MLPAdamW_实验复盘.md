# DG-KAN v7.6 M12 SystemAdvantage vs MLP-AdamW 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.6_M12_SystemAdvantage_vs_MLPAdamW_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。本轮没有做 loss、sampler、classwise、focal、margin 或 class weight 调参，追加尝试只围绕 v7.6 P5/P6 的 S1/S2 memory/cache 结构修复。

## 1. 目标是否达成

最终 clean 10-seed confirmation run：

```bash
python experiments/run_gafu_v76_real.py \
  --out-dir results/real_rerun_20260506/v76_m12_clean_10seed_confirm_20260506T130000Z \
  --fresh \
  --device auto \
  --candidates B0,B2,M12 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

| 目标 | 结论 | 证据 |
|---|---|---|
| real-only / no-fake / no-proxy | 达成 | `v76_provenance_audit.csv`: `rows_checked=2453`, fake/proxy nonzero `0` |
| StrictPass | 达成 | `M12` strict KAN head，non-KAN trainable params `0` |
| GradPass | 达成 | `M12` gradient `6/6` pass，max relerr `8.07e-05` |
| 10-seed MacroSignificantPass | 达成 | val gap `+0.02057`，CI95 low `+0.01491`，Holm p `1.35e-06`，test gap `+0.01966` |
| Fairness vs MLP-distill | 达成 | `M12` vs `B2` val gap `+0.01517`，NLL delta `-0.02413` |
| FullGridS2Pass | 达成 | memory mean `1.0143`，max `1.0381`; step mean `1.1683`, max `1.2591`; S2 shapes `9/9` |
| FullGridS1Pass | 未达成 | S1 shapes `3/9`; bs256/512 memory ratio 仍 `>1.00` |
| TimeAUCPass | 未达成 | mean `ValLossAUC_time`: M12 `0.2941` > B0 `0.1582` |
| v7.6 minimum success | 达成 | route `S2-FairSystemMinimumSuccess` |
| v7.6 formal success | 未达成 | 缺 FullGridS1Pass 与 TimeAUCPass |

最终判断：

> v7.6 minimum success 达成：`M12` 在 10 seeds 下相对 MLP-AdamW 仍有稳定 macro advantage，并且在同 teacher/distill 条件的 MLP 对照下仍显著更好，同时恢复 FullGridS2。v7.6 formal system success 未达成，因为 S1 与 wall-clock AUC 仍未闭合。

## 2. Route Decision

`v76_route_decision.json`：

```json
{
  "route": "S2-FairSystemMinimumSuccess",
  "best_candidate_id": "M12",
  "strict_pass": 1,
  "grad_pass": 1,
  "macro_significant_pass_10seed": 1,
  "fullgrid_s2_pass": 1,
  "fullgrid_s1_pass": 0,
  "fairness_pass": 1,
  "time_auc_pass": 0,
  "success_v76_minimum": 1,
  "success_v76_formal": 0,
  "macro_gap": 0.020572916666666666,
  "ci95_low": 0.014908854166666667,
  "holm_p": 1.346826955306228e-06,
  "test_gap": 0.019661458333333333,
  "ECE_delta": -0.0041540805250406265,
  "NLL_delta": -0.10058752596378326,
  "seed_win_rate": 1.0,
  "m12_gap_vs_mlp_distill": 0.015169270833333378,
  "m12_nll_delta_vs_mlp_distill": -0.02412515679995214,
  "memory_ratio_mean": 1.0143359939345469,
  "memory_ratio_max": 1.0381213653964307,
  "step_ratio_mean": 1.1683173565223521,
  "step_ratio_max": 1.2591264865730807,
  "s2_shape_count": 9,
  "s1_shape_count": 3,
  "primary_blocker": "s1_or_time_auc_not_closed",
  "classwise_is_hard_gate": false,
  "no_fake": true,
  "no_proxy": true
}
```

## 3. Task / Fairness 结果

| candidate | role | val acc | test acc | val gap vs MLP | ECE | NLL |
|---|---|---:|---:|---:|---:|---:|
| `B0` | MLP-AdamW | `0.8449` | `0.7958` | `0.0000` | `0.0619` | `0.5745` |
| `B2` | MLP + C3 logit distill | `0.8503` | `0.8001` | `+0.0054` | `0.0450` | `0.4981` |
| `M12` | PureKAN-NG + C3 logit distill | `0.8655` | `0.8154` | `+0.0206` | `0.0578` | `0.4739` |

Macro significance:

| candidate | mean val gap | CI95 low | Holm p | mean test gap | ECE delta | NLL delta |
|---|---:|---:|---:|---:|---:|---:|
| `M12` | `+0.02057` | `+0.01491` | `1.35e-06` | `+0.01966` | `-0.00415` | `-0.10059` |
| `B2` | `+0.00540` | not macro pass | `0.152` in first full run | `+0.00436` | `-0.01695` | `-0.07646` |

Fairness 判断：

- `M12` 比同 teacher objective 的 `B2` 高 `+0.01517` macro val gap。
- `M12` 的 NLL 比 `B2` 低 `0.02413`。
- 因此 v7.6 的 “distillation-only” 质疑没有成立：C3 teacher 对 MLP 也有帮助，但 MLP-distill 没有追平 M12。

Teacher-free 对照来自第一轮 full run：

```text
run = results/real_rerun_20260506/v76_m12_system_fairness_10seed_20260506T120000Z
M13 val gap vs MLP = +0.01784
M12 - M13 = +0.00273
teacher_dependency_mild = 1
teacher_independent_pass = 1
```

判断：M12 的 C3 distill 有轻度增益，但 M12 architecture / M5 init 在无 teacher 时已经有 `+0.0178` 的独立 signal。

## 4. Gradient Correctness

Clean run:

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `M12` | `6/6` | `8.07e-05` | `0.999998` |

本轮新增 S1 repair 的 gradient 结果：

| candidate | 目的 | pass rows | max relerr | 判断 |
|---|---|---:|---:|---|
| `M14` | backward 重算 hidden-y，减少常驻 cache | `6/6` | `8.07e-05` | GradPass，但效率未闭合 |
| `M15` | hidden-y half cache | smoke `0/1` | `4.82e-02` | GradFail，停止 |
| `M16` | 只缓存第一层 y，第二层 y backward 重算 | `6/6` | `8.07e-05` | GradPass，但效率未闭合 |

`M15` 只做了 2-step smoke，不作为 task/efficiency 正式结论；它因为 GradPass 失败，没有进入 full probe。

## 5. Efficiency / S1 Margin

Clean run `M12` full-grid:

| dataset | batch | memory ratio | step ratio | S2 | S1 |
|---|---:|---:|---:|---:|---:|
| MNIST | 128 | `0.9962` | `1.2591` | 1 | 1 |
| MNIST | 256 | `1.0086` | `1.1610` | 1 | 0 |
| MNIST | 512 | `1.0381` | `1.1588` | 1 | 0 |
| Fashion-MNIST | 128 | `0.9962` | `1.1596` | 1 | 1 |
| Fashion-MNIST | 256 | `1.0086` | `1.1555` | 1 | 0 |
| Fashion-MNIST | 512 | `1.0381` | `1.1571` | 1 | 0 |
| KMNIST | 128 | `0.9962` | `1.1520` | 1 | 1 |
| KMNIST | 256 | `1.0086` | `1.1626` | 1 | 0 |
| KMNIST | 512 | `1.0381` | `1.1491` | 1 | 0 |

Summary:

| candidate | memory mean | memory max | step mean | step max | S2 shapes | S1 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M12` | `1.0143` | `1.0381` | `1.1683` | `1.2591` | `9/9` | `3/9` |

S1 blocker:

- step 已满足 S1 step gate in all shapes (`<=1.35`)。
- S1 fail 只来自 memory：bs256 memory `1.0086`，bs512 memory `1.0381`，均高于 `<1.00`。
- P5 attribution 中 top memory sources 为 `fused_stack_checkpoint_root_input`、`fused_stack_hidden_y_cache`、`optimizer_state`；`materialized_tensor_count=3`，`kernel_count_total=3`，unknown memory fraction `0.0`。

## 6. Time / Wall-Clock AUC

Mean across task rows:

| candidate | ValLossAUC_time | ValAccAUC_time | wall-clock sec | samples/sec |
|---|---:|---:|---:|---:|
| `B0` | `0.1582` | `0.2248` | `0.3049` | `1.2169e6` |
| `B2` | `0.2667` | `0.3884` | `0.4982` | `7.4105e5` |
| `M12` | `0.2941` | `0.4542` | `0.5822` | `6.3373e5` |

判断：

- `M12` final accuracy / NLL 更好，但 wall-clock AUC 没有优于 MLP。
- v7.6 因此只能称为 S2 + fairness minimum success，不能称为 full system wall-clock success。

## 7. 失败后自修复

第一轮 10-seed full run：

```bash
python experiments/run_gafu_v76_real.py \
  --out-dir results/real_rerun_20260506/v76_m12_system_fairness_10seed_20260506T120000Z \
  --fresh \
  --device auto \
  --candidates B0,B2,M12,M13 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

结果：

```text
MacroSignificantPass = 1
FairnessPass = 1
GradPass = 1
FullGridS2Pass = 0
memory mean = 1.0143
memory max = 1.0613
step mean = 1.5174
step max = 1.6441
S2 shapes = 4/9
```

失败后尝试：

| attempt | run | 结果 | 结论 |
|---|---|---|---|
| M12 isolated rerun | same out_dir reuse task rows | step 恢复到 mean `1.2735` / max `1.3100`，但 bs512 memory `1.0613`，S2 `6/9` | step 波动可修复，memory 仍不稳 |
| `M14` recompute hidden-y | `v76_m14_s1_cache_trim_5seed_20260506T123000Z` | task 与 M12 一致，GradPass `6/6`; memory mean `1.0150`; step mean `1.3998`; S2 `7/9` | memory 有改善，但 backward 重算导致部分 shape step fail |
| `M15` half hidden-y cache | `tmp_v76_m15_smoke` | grad relerr `4.82e-02` | 直接 GradFail，停止 |
| `M16` first-y cache + second-y recompute | `v76_m16_s1_first_y_cache_5seed_20260506T124500Z` | task 与 M12 一致，GradPass `6/6`; step mean `1.3340`; memory mean `1.0224`; S2 `6/9` | step 接近 S1，但 bs512 memory 反而更差 |
| clean B0/B2/M12 rerun | `v76_m12_clean_10seed_confirm_20260506T130000Z` | `FullGridS2Pass=1`，minimum success 达成 | 最终采用 clean run 作为 v7.6 route |

这些尝试均为结构/cache/runtime方向，不是 loss 或 classwise 调参。

## 8. No-Fake / No-Proxy 审计

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v76_m12_clean_10seed_confirm_20260506T130000Z` | `2453` | `0` |
| `v76_m12_system_fairness_10seed_20260506T120000Z` | `3252` | `0` |
| `v76_m14_s1_cache_trim_5seed_20260506T123000Z` | `1677` | `0` |
| `v76_m16_s1_first_y_cache_5seed_20260506T124500Z` | `1679` | `0` |
| `tmp_v76_m15_smoke` | `80` | `0` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py experiments/run_gafu_v75_real.py
```

已通过。

关键 hash（clean run）：

| file | SHA256 |
|---|---|
| `run_manifest.json` | `9b228065bd312b6c216590b0268b6731ba9bee98f5cc21df5c43eab1930fa9eb` |
| `v76_manifest.json` | `1f500276002990e831676a5957c4ebb9229b1bf2c0648ba18ca9085b368be216` |
| `v76_route_decision.json` | `6828c2401bdcd02e0d2e37dcce2385a8098a49d9228fc7b5bca282af663b94e0` |
| `p9_task_summary.csv` | `bb27b02f1170145254846e173ee6d166971a1fd27ad180247b62f6c69f020e1c` |
| `p1_significance_audit.csv` | `ee53590095cfd62b05ec5ff552e46c13b4f68b7656c1c626960cc2794d243a4c` |
| `p2_full_gradient_correctness.csv` | `a909838d5e66f1ee6a20e83bc0e629e5c6c3cf030758829c131cef8ac3169df0` |
| `p10_efficiency_summary.csv` | `f48df5bead733cb22702eb2127901955aa814929f66419921225684b62a95c22` |
| `p2_mlp_distill_fairness_v76.csv` | `777d129e9e668f2f969ee0da51bb398c2acaa4c52e453d6ad65b9fffe73a5049` |
| `p5_s1_margin_attribution_v76.csv` | `860c9a4541bfb83d3004942d12e03dadef6f63bb353386ced81133aa2a0d5c01` |
| `v76_provenance_audit.csv` | `f866333149b9e79b691fcd949cf0e181298b9339eed78b58fcef0b371ff233a7` |
| first full run `p3_teacher_dependency_v76.csv` | `b4bbbe492f62e482b1798e2359fa9831c36bcae9fa33d316c3ea27af89c3d626` |

## 9. 最终结论

v7.6 的最低系统优势目标已经达成：

```text
StrictPass = true
GradPass = true
MacroSignificantPass_10seed = true
FairnessPass_vs_MLP_distill = true
FullGridS2Pass = true
success_v76_minimum = true
```

但 v7.6 正式目标未达成：

```text
FullGridS1Pass = false
TimeAUCPass = false
success_v76_formal = false
```

机制结论：

1. `M12` 对 MLP-AdamW 的 +2% macro advantage 在 10 seeds 下仍稳定存在，不是 5-seed 偶然。
2. `B2` 证明 C3 teacher distillation 本身会提高 MLP，但提升不足以解释 M12；M12 仍比 MLP-distill 高 `+0.01517`，且 NLL 更低。
3. `M13` 说明 M12 architecture / M5 init 本身已经有 `+0.01784` signal；C3 distill 是 mild boost，不是唯一来源。
4. `M12` clean run 恢复 FullGridS2：memory/step 全 shape 过 S2。
5. S1 blocker 已收敛为 memory，而不是 step：bs256/512 memory 仍高于 `1.00`。
6. M14/M15/M16 的 S1 repair 没有产生可用新路线：要么 step fail，要么 grad fail，要么 bs512 memory 更差。
7. wall-clock AUC 仍是系统级短板：M12 final quality 更好，但训练时间与 ValLossAUC_time 不优于 MLP。

最终一句话：

> v7.6 达成了 `S2-FairSystemMinimumSuccess`：M12 的 PureKAN-NG 系统优势相对 MLP-AdamW 和 MLP-distill 都是真实、10-seed 稳定、gradient-correct 且 FullGridS2 的。但它还不是正式 full system success，因为 S1 memory 与 wall-clock AUC 没闭合。下一步应聚焦 S1 memory source 与 wall-clock forward path，而不是回到 loss 或 classwise 修补。
