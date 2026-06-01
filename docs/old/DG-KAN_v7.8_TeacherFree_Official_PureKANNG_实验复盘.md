# DG-KAN v7.8 Teacher-Free Official PureKAN-NG 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.8_TeacherFree_Official_PureKANNG_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。v7.8 的核心纠偏是：`M12 + C3 teacher` 只能作为 diagnostic assisted candidate，不能作为 official success；official route 只看 `external_teacher_used=0` 的 teacher-free PureKAN-NG。

## 1. 目标是否达成

主 run：

```bash
python experiments/run_gafu_v78_real.py \
  --out-dir results/real_rerun_20260506/v78_teacherfree_official_10seed_20260506T150000Z \
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

| 目标 | 结论 | 证据 |
|---|---|---|
| real-only / no-fake / no-proxy | 达成 | 主 run `v78_provenance_audit.csv`: `rows_checked=3270`, fake/proxy nonzero `0` |
| Official external teacher free | 达成 | official best 为 `M13`，`external_teacher_used=0`，`teacher_candidate=none` |
| StrictPass | 达成 | `M13` strict KAN head，non-KAN trainable params `0` |
| GradPass | 达成 | `M13` gradient `6/6` pass，max relerr `6.32e-05` |
| TeacherFreeMacroSignificantPass | 未达成 | `M13` val gap `+0.01784 < +0.0200` |
| FullGridS2Pass | 达成 in main run | `M13` memory max `1.0381`，step max `1.4857`，S2 shapes `9/9` |
| FullGridS1Pass | 未达成 | `M13` S1 shapes `2/9` |
| TimeAUCPass | 未达成 | `M13` ValLossAUC_time ratio vs B0 `1.3922` |
| v7.8 minimum success | 未达成 | 缺 TeacherFreeMacroSignificantPass |
| v7.8 formal success | 未达成 | 缺 macro、S1、TimeAUC |

最终判断：

> v7.8 没有达成 teacher-free official success。`M13` 真实复现了 teacher-free positive signal，但只到 `+0.01784`，没有跨过 `+0.0200` hard gate；`M12+C3` 仍能过 macro gate，但只能记为 `TeacherAssistedSuccessOnly`，不能作为 PureKAN-NG official success。

## 2. Route Decision

主 run `v78_route_decision.json`：

```json
{
  "route": "R1-TeacherAssistedSuccessOnly",
  "best_official_candidate_id": "M13",
  "external_teacher_used": 0,
  "self_teacher_used": 0,
  "teacher_candidate": "none",
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_significant_pass": 0,
  "fullgrid_s2_pass": 1,
  "fullgrid_s1_pass": 0,
  "time_auc_pass": 0,
  "success_v78_minimum": 0,
  "success_v78_formal": 0,
  "macro_gap": "0.017838541666666666",
  "ci95_low": "0.012239583333333333",
  "holm_p": "7.974602500622449e-06",
  "test_gap": "0.022981770833333335",
  "memory_ratio_max": 1.0381213653964307,
  "step_ratio_max": 1.4856588856004525,
  "diagnostic_m12_c3_macro_gap": "0.020572916666666666",
  "diagnostic_m12_minus_m13_gap": 0.0027343750000000298,
  "primary_blocker": "teacher_free_macro_gap_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

说明：

- `M13` 是 official teacher-free candidate。
- `M12` 仍是 diagnostic assisted candidate，不进入 official route。
- route 是 `TeacherAssistedSuccessOnly`，因为 assisted `M12` 过 macro gate，而 teacher-free `M13` 没过。

## 3. Task / Fairness 结果

主 run task summary：

| candidate | role | external teacher | val acc | test acc | val gap vs MLP | ECE | NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| `B0` | MLP-AdamW | 0 | `0.8449` | `0.7958` | `0.0000` | `0.0619` | `0.5745` |
| `B2` | MLP + C3 distill diagnostic | 1 | `0.8503` | `0.8001` | `+0.0054` | `0.0450` | `0.4981` |
| `M12` | PureKAN + C3 distill diagnostic | 1 | `0.8655` | `0.8154` | `+0.0206` | `0.0578` | `0.4739` |
| `M13` | PureKAN teacher-free official | 0 | `0.8628` | `0.8188` | `+0.0178` | `0.0585` | `0.4771` |

Macro significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | official? |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M12` | `+0.02057` | `+0.01491` | `2.02e-06` | `+0.01966` | `-0.00415` | `-0.10059` | no, C3 teacher |
| `M13` | `+0.01784` | `+0.01224` | `7.97e-06` | `+0.02298` | `-0.00342` | `-0.09742` | yes, but macro fail |
| `B2` | `+0.00540` | `+0.00098` | `0.1520` | `+0.00436` | `-0.01695` | `-0.07646` | no, MLP diagnostic |

判断：

- `M13` 的 teacher-free signal 真实存在：CI、Holm p、test gap、ECE/NLL 都支持 positive。
- 但 v7.8 hard gate 是 macro val gap `>= +0.0200`，`M13` 实测 `+0.01784`，不能四舍五入为成功。
- `M12 - M13 = +0.00273`，符合 v7.8 计划里 “teacher dependency mild” 的判断，但这 `+0.0027` 正是 official gate 差距。

## 4. Gradient Correctness

主 run：

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `M12` | `6/6` | `8.07e-05` | `0.999998` |
| `M13` | `6/6` | `6.32e-05` | `0.999999` |

Teacher-free repair probes：

| run | candidate | pass rows | max relerr | 判断 |
|---|---|---:|---:|---|
| longbudget480 | `M13` | `6/6` | `6.32e-05` | GradPass |
| family probe | `M4` | `6/6` | `8.40e-05` | GradPass |
| family probe | `M9` | `6/6` | `8.40e-05` | GradPass |
| family probe | `M13` | `6/6` | `6.32e-05` | GradPass |
| hidden80 5seed | `M13` | `3/3` | `8.10e-05` | GradPass |
| basis12 5seed | `M13` | `3/3` | `6.32e-05` | GradPass |
| basis16 5seed | `M13` | `3/3` | `6.32e-05` | GradPass |

没有手动改 pass。

## 5. Efficiency / TimeAUC

主 run `M13` full-grid summary：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | S1 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M13` | `1.0143` | `1.0381` | `1.3080` | `1.4857` | `9/9` | `2/9` |

TimeAUC：

| candidate | ValLossAUC_step ratio vs B0 | ValLossAUC_time ratio vs B0 | TimeAUCPass |
|---|---:|---:|---:|
| `M13` | `0.9370` | `1.3922` | 0 |
| `M12` | not official here | not official here | diagnostic only |

判断：

- `M13` 按 step 学得比 MLP 好，说明 teacher-free optimizer dynamics 不是完全失败。
- 但 wall-clock AUC 仍失败：time ratio `1.3922 > 1.0`。
- v7.8 minimum 已经因 macro fail 而失败；formal 还额外缺 S1 和 TimeAUC。

## 6. 失败后自修复

所有追加尝试都保持 `external_teacher_used=0`，没有使用 C3 logits、teacher forward、classwise/loss/sampler/focal/margin/class weight。

### 6.1 Long-budget 480

```bash
python experiments/run_gafu_v78_real.py \
  --out-dir results/real_rerun_20260506/v78_teacherfree_longbudget480_10seed_20260506T153000Z \
  --fresh \
  --device auto \
  --candidates B0,M13 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --task-steps 480 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 20 \
  --bootstrap-reps 10000
```

| candidate | val gap | CI95 low | Holm p | test gap | S2 shapes | TimeAUC ratio |
|---|---:|---:|---:|---:|---:|---:|
| `M13` | `+0.01842` | `+0.01478` | `1.10e-09` | `+0.02285` | `6/9` | `1.3321` |

判断：

- 长训练只把 gap 从 `+0.01784` 推到 `+0.01842`，增益约 `+0.00059`，远小于需要的约 `+0.00216`。
- 同时 memory max 到 `1.0657`，FullGridS2 失败。
- long-budget 不是 v7.8 解法。

### 6.2 Teacher-free family probe

```bash
python experiments/run_gafu_v78_real.py \
  --out-dir results/real_rerun_20260506/v78_teacherfree_family_probe_10seed_20260506T160000Z \
  --fresh \
  --device auto \
  --candidates B0,M4,M9,M13 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

| candidate | val gap | CI95 low | Holm p | test gap | GradPass | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M13` | `+0.01784` | `+0.01224` | `7.25e-06` | `+0.02298` | 1 | `6/9` |
| `M4` | `+0.01699` | `+0.01191` | `6.10e-06` | `+0.01973` | 1 | `5/9` |
| `M9` | `+0.01699` | `+0.01178` | `6.10e-06` | `+0.01973` | 1 | `6/9` |

判断：

- M4/M9 没有超过 M13。
- teacher-free family 中当前仍是 M13 最强，但还没过 hard gate。

### 6.3 Hidden80 capacity probe

5-seed probe，B0 同步使用 hidden80，对照仍公平：

| candidate | val gap | CI95 low | Holm p | test gap | GradPass | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M13 hidden80` | `+0.01914` | `+0.01107` | `0.00172` | `+0.02344` | 1 | `2/3` |

判断：

- hidden80 接近但没过 `+0.0200`。
- step max `2.7815`，效率明显变差。
- 不扩成 10-seed official run。

### 6.4 Basis12 / Basis16 probe

5-seed probe：

| candidate | val gap | CI95 low | Holm p | test gap | GradPass | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M13 basis12` | `+0.01927` | `+0.01250` | `0.00111` | `+0.02747` | 1 | `2/3` |
| `M13 basis16` | `+0.01927` | `+0.01250` | `0.00111` | `+0.02747` | 1 | `3/3` |

判断：

- basis12/16 都仍低于 `+0.0200`。
- ECE delta 为 `+0.0010`，没有像 M13 baseline 那样保持 ECE 更优。
- 这是 measured near-miss，不记成功。

## 7. 未执行 / 未实现项

| stage | 状态 | 说明 |
|---|---|---|
| P4 self-distill | not_run | 本轮先测 teacher-free baseline、longbudget、family、capacity/basis probe；未实现 EMA/self-teacher |
| P8 phase-mapped profiler | not_run | v7.8 official macro 未过，未继续 profiler success claim |
| P10 fused/streaming package | not_implemented | 没有 teacher-free official macro candidate，不写 fused pass |
| Scaling/robustness | not_run | strong success 前置条件未满足 |

这些不是缺失数据被补为成功；它们没有进入 route success。

## 8. No-Fake / No-Proxy 审计

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v78_teacherfree_official_10seed_20260506T150000Z` | `3270` | `0` |
| `v78_teacherfree_longbudget480_10seed_20260506T153000Z` | `1671` | `0` |
| `v78_teacherfree_family_probe_10seed_20260506T160000Z` | `3221` | `0` |
| `v78_teacherfree_hidden80_probe_5seed_20260506T163000Z` | `870` | `0` |
| `v78_teacherfree_basis12_probe_5seed_20260506T164500Z` | `870` | `0` |
| `v78_teacherfree_basis16_probe_5seed_20260506T170000Z` | `870` | `0` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

关键 hash（主 run）：

| file | SHA256 |
|---|---|
| `run_manifest.json` | `1ca34489f94a51aa6ca31c46a2dbdaac2dcf386e0b219218b3464db700b3320f` |
| `v78_manifest.json` | `811f4c3fcdcf1b715ccdcce5b3ce2695131421c8eddcc4b6ce35311295f1569e` |
| `v78_route_decision.json` | `c206b1c9fe82ebcdd854bc4b8dd990b430c9f3d38d923be845f82a927566d823` |
| `p9_task_summary.csv` | `59232334e1785a3cf4cadeaeb8038551d1cf755e0200020e28ecfe25758665bc` |
| `p1_significance_audit.csv` | `00612f8377cc3cdefe63d3bf586562e5e899c430c166085bc45caadc6bfeb548` |
| `p2_full_gradient_correctness.csv` | `5d6577113c8d9778168ea32706b7e237db798b11765e2331b15e2ef0783780f0` |
| `p10_efficiency_summary.csv` | `9f59ad20fba6ebc79461cd8bcec302aaf8e435e6d9f45d9f16b612e77f01e7d2` |
| `v78_provenance_audit.csv` | `83fc3380a64d4ef5bbc63f2887b4f5eb3a62635fa75289b84def844b2ca001fd` |
| `p0_contract_reproduction_v78.csv` | `550407f7239a501139879c8b8355f9b446040317093847bd58bd3f31132e3dcd` |
| `p5_external_teacher_diagnostic_v78.csv` | `e71ec6398c70055c87e478ef02ff2e1aaa8d8a3c597739fb83a1ac66c0d460d4` |

关键 hash（repair probes）：

| run | file | SHA256 |
|---|---|---|
| long480 | `v78_route_decision.json` | `8ec7b1e76f1272d0b47a6593d41ad9ac66bc4cb77fe9a0ac602296d860fae0f6` |
| family | `v78_route_decision.json` | `3ee1a49ec3ccda2fb5cfd1fb28f5de5bcca1a942592d669c08688b3b0c9322a8` |
| hidden80 | `v78_route_decision.json` | `867333a7cfa7ca6897479956b0febf50fa9e723e80ffc1e18bd9fb6d6b5d95a9` |
| basis12 | `v78_route_decision.json` | `acca6521180d6f707b4a2f62378dd2bade242ebb1697dd1303e7eaa9899e2bf6` |
| basis16 | `v78_route_decision.json` | `d712b2384d0d6e4f655a279497bd20d6ee30a166840f43b7a9d8945dc09d4d85` |

## 9. 最终结论

v7.8 没有达成 teacher-free official target：

```text
Official candidate = M13
external_teacher_used = false
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = false
FullGridS2Pass = true in main run
FullGridS1Pass = false
TimeAUCPass = false
success_v78_minimum = false
success_v78_formal = false
```

机制结论：

1. `M13` 证明 teacher-free PureKAN-NG 已有稳定 positive signal：macro gap `+0.01784`，CI95 low `+0.01224`，test gap `+0.02298`，ECE/NLL 均优于 MLP。
2. 但 v7.8 official hard gate 是 `+0.0200`，`M13` 仍差约 `0.00216`。
3. `M12+C3` 的 `+0.02057` 仍成立，但只能说明 C3 teacher 提供了约 `+0.00273` 的 mild boost；不能作为 official teacher-free success。
4. Long-budget 480、M4/M9 teacher-free family probe、hidden80、basis12/16 都没有闭合 teacher-free macro gate。
5. 多个 repair 同时暴露系统侧问题：TimeAUC 仍失败，部分 runs 的 S2/S1 不稳，说明即使 macro 过线，formal success 也仍需要 runtime/S1 继续修。
6. 本轮没有使用 fake/proxy，也没有把 teacher-assisted candidate 混同为 official candidate。

最终一句话：

> v7.8 证明了 PureKAN-NG teacher-free 路线“很接近但还没成立”：M13 不靠外部 teacher 已经稳定超过 MLP，但没有跨过 `+0.0200` official gate；C3 teacher 的小增益正好补上缺口，因此当前只能叫 teacher-assisted success，不能叫 teacher-free official success。下一步若继续，应实现真正 no-external-teacher self-distill/EMA 或表示层初始化修复，并同时保住 S2/TimeAUC，而不是回到 C3 teacher 或 loss/classwise 调参。
