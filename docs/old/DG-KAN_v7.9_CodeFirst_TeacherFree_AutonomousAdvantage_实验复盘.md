# DG-KAN v7.9 CodeFirst TeacherFree AutonomousAdvantage 实验复盘

> 本复盘记录 `docs/DG-KAN_v7.9_CodeFirst_TeacherFree_AutonomousAdvantage_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。v7.9 的重点是 code-first/metadata-first 地判定 teacher-free official candidate，避免把 `M12 + C3 teacher` 混入 official success。

## 1. 目标是否达成

主 run：

```bash
python experiments/run_gafu_v79_real.py \
  --out-dir results/real_rerun_20260506/v79_codefirst_teacherfree_baseline_10seed_20260506T180000Z \
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
| real-only / no-fake / no-proxy | 达成 | 主 run `v79_provenance_audit.csv`: `rows_checked=3255`, fake/proxy nonzero `0` |
| Code-first candidate metadata | 达成 | `candidate_registry.csv`, `p1_candidate_metadata_audit.csv`, `contract_validator.csv` 已落盘 |
| Teacher leak audit | 达成 | `teacher_leak_audit.csv`: official `M13 external_teacher_used=0`, `teacher_logits_used=0`, `teacher_forward_used=0` |
| StrictPass | 达成 | `M13` strict KAN head，non-KAN trainable params `0`，manual forward/backward/update `1/1/1` |
| GradPass | 达成 | `M13` gradient pass，max relerr `6.32e-05` |
| TeacherFreeMacroSignificantPass | 未达成 | `M13` macro val gap `+0.01784 < +0.0200` |
| FullGridS2Pass | 未达成 in main run | `M13` memory max `1.0381`，step max `1.6199`; step 超 S2 |
| TimeAUCPass | 未达成 | `M13` ValLossAUC_time ratio vs B0 `1.4474` |
| v7.9 minimum success | 未达成 | 缺 teacher-free macro gate，主 run 也缺 S2 |
| v7.9 formal success | 未达成 | 缺 macro、S2/S1稳定、TimeAUC、profiler/fused package |

最终判断：

> v7.9 没有达成 teacher-free official success。它修正了 v7.8 的代码层审计问题：official route 由 metadata/contract 驱动，`M12` 被明确排除为 external-teacher diagnostic。但 official `M13` 仍只有 `+0.01784`，结构修复 probe 最高也只到 `+0.019921875`，没有跨过 `+0.0200` hard gate。

## 2. Route Decision

主 run `v79_route_decision.json`：

```json
{
  "route": "R4-TeacherAssistedOnly",
  "best_official_candidate_id": "M13",
  "external_teacher_used": 0,
  "self_teacher_used": 0,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "fullgrid_s1_pass": 0,
  "time_auc_pass": 0,
  "profiler_pass": 0,
  "code_contract_pass": 1,
  "success_v79_minimum": 0,
  "success_v79_formal": 0,
  "macro_gap": "0.017838541666666666",
  "ci95_low": "0.012239583333333333",
  "holm_p": "7.974602500622449e-06",
  "test_gap": "0.022981770833333335",
  "memory_ratio_max": 1.0381213653964307,
  "step_ratio_max": 1.6199482365677846,
  "diagnostic_m12_c3_macro_gap": "0.020572916666666666",
  "diagnostic_m12_minus_best_gap": 0.0027343750000000298,
  "primary_blocker": "teacher_free_macro_gap_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

说明：

- `M13` 是 best official candidate，`external_teacher_used=0`。
- `M12` 仍然达到 macro gate，但 `external_teacher_used=1`，只进入 diagnostic，不进入 official route。
- 主 blocker 仍是 teacher-free macro gap 未闭合；系统侧 S2/TimeAUC 也没有闭合。

## 3. Code Contract / Teacher Leak

`p0_code_contract.csv` 关键行：

| candidate | official eligible | external teacher | strict pass | manual f/b/u | uses loss backward | non-KAN params |
|---|---:|---:|---:|---|---:|---:|
| `B0` | 0 | 0 | 0 | `0/0/0` | 1 | metric unavailable |
| `B2` | 0 | 1 | 0 | `0/0/0` | 1 | metric unavailable |
| `M12` | 0 | 1 | 1 | `1/1/1` | 0 | 0 |
| `M13` | 1 | 0 | 1 | `1/1/1` | 0 | 0 |

`teacher_leak_audit.csv` 判断：

| candidate | teacher source | teacher logits | teacher forward | teacher leak detected | official |
|---|---|---:|---:|---:|---:|
| `B0` | none | 0 | 0 | 0 | 0 |
| `B2` | C3 | 1 | 1 | 0 | 0 |
| `M12` | C3 | 1 | 1 | 0 | 0 |
| `M13` | none | 0 | 0 | 0 | 1 |

这轮的主要代码改动是新增 `experiments/run_gafu_v79_real.py`，将 candidate registry、teacher metadata、contract validator 和 route decision 显式落盘。没有把 id 硬编码为 official success。

## 4. Task / Fairness 结果

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
| `M13` | `+0.01784` | `+0.01224` | `7.97e-06` | `+0.02298` | `-0.00342` | `-0.09742` | yes, macro fail |
| `B2` | `+0.00540` | `+0.00098` | `0.1520` | `+0.00436` | `-0.01695` | `-0.07646` | no, MLP diagnostic |

判断：

- `M13` teacher-free signal 真实存在：CI95 low 大于 0、Holm p 显著、test gap 更高、NLL/ECE 不劣化。
- 但 v7.9 official gate 是 `mean val gap >= +0.0200`，`M13` 实测 `+0.01784`，不能记为 pass。
- `M12 - M13 = +0.00273`，正好说明 external C3 teacher 的小增益仍是官方 gate 差距来源。

## 5. Gradient / Efficiency / TimeAUC

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `M12` | pass | `8.07e-05` | `0.999998` |
| `M13` | pass | `6.32e-05` | `0.999999` |

主 run `M13` full-grid summary：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | S1 shapes |
|---|---:|---:|---:|---:|---:|---:|
| `M13` | `1.0143` | `1.0381` | `1.3853` | `1.6199` | `8/9` | `1/9` |

TimeAUC：

| candidate | ValLossAUC_step ratio vs B0 | ValLossAUC_time ratio vs B0 | TimeAUCPass |
|---|---:|---:|---:|
| `M13` | `0.9370` | `1.4474` | 0 |
| `M12` | `0.9320` | `1.8419` | diagnostic only |

判断：

- `M13` 仍按 step 学得比 MLP 好，但 wall-clock AUC 失败。
- 主 run S2 不稳定：memory 过 S2，但 step max 到 `1.6199`。
- 即使 macro 过线，v7.9 formal success 也还需要 S2/S1/TimeAUC 修复。

## 6. 失败后自修复 A：Teacher-Free Structural Screen

运行：

```bash
python experiments/run_gafu_v79_real.py \
  --out-dir results/real_rerun_20260506/v79_teacherfree_structural_screen_5seed_20260506T183000Z \
  --fresh \
  --device auto \
  --candidates B0,A2S,E1,Z1,U1,M1,M4,M7,M9,M13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果摘要：

| candidate | external teacher | val gap | test gap | GradPass | S2 small-grid | step max | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `A2S` | 0 | `+0.01992` | `+0.02331` | 1 | 0 | `1.7238` | macro near-miss, S2 fail |
| `M9` | 0 | `+0.01992` | `+0.02331` | 1 | 0 | `1.5358` | macro near-miss, S2 fail |
| `M13` | 0 | `+0.01927` | `+0.02747` | 1 | 1 | `1.4789` | co-selection best, macro fail |

判断：

- 结构候选把 teacher-free val gap 推到 `+0.019921875`，这是本轮最接近 macro gate 的结果。
- 但 `+0.019921875 < +0.0200`，且 `A2S/M9` small-grid step 没进 S2。
- 不能把 near-miss 记为成功。

No-fake/no-proxy：`v79_provenance_audit.csv` rows checked `4046`，fake/proxy nonzero `0`。

## 7. 失败后自修复 B：Basis16 Structural Probe

运行：

```bash
python experiments/run_gafu_v79_real.py \
  --out-dir results/real_rerun_20260506/v79_teacherfree_basis16_structural_screen_5seed_20260506T190000Z \
  --fresh \
  --device auto \
  --basis-count 16 \
  --candidates B0,A2S,M9,M13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果摘要：

| candidate | external teacher | val gap | CI95 low | Holm p | test gap | GradPass | S2 small-grid | S1 small-grid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A2S` | 0 | `+0.01992` | `+0.01250` | `0.00189` | `+0.02331` | 1 | 0 | 0 |
| `M9` | 0 | `+0.01992` | `+0.01250` | `0.00189` | `+0.02331` | 1 | 0 | 0 |
| `M13` | 0 | `+0.01927` | `+0.01250` | `0.00222` | `+0.02747` | 1 | 1 | 1 |

判断：

- basis16 没有提高 macro gap；`A2S/M9` 仍停在 `+0.019921875`。
- `M13` 在这个 small-grid 里过了 S2/S1，但 macro 仍是 `+0.01927`，不能作为 v7.9 success。
- 该 probe 支持“representation tweak 接近但仍不够”的结论。

No-fake/no-proxy：`v79_provenance_audit.csv` rows checked `1657`，fake/proxy nonzero `0`。

## 8. 未执行 / 未实现项

| stage | 状态 | 说明 |
|---|---|---|
| `SB0/SB1/SB2/SB3` self-bootstrap | `not_implemented` | 已进入 registry，但未写假 task/efficiency row |
| `RR1/RR2/RR3/RR5` representation repair | `not_implemented` | 仅 registry 记录，不 claim pass |
| `SYS1` system memory trim | `not_implemented` | 未写 S1 fake ratio |
| P9 time accounting | `not_run` | 没有 phase-mapped timing，不写 pass |
| P10 phase-mapped profiler | `not_run` | 没有 profiler phase mapping，不写 pass |
| P12 fused/streaming package | `not_implemented` | 没有 teacher-free macro pass candidate，不写 fused success |
| P14 scaling / P15 geometry | `not_run` | 前置 official macro gate 未满足 |

## 9. No-Fake / No-Proxy 审计与 Hash

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v79_codefirst_teacherfree_baseline_10seed_20260506T180000Z` | `3255` | `0` |
| `v79_teacherfree_structural_screen_5seed_20260506T183000Z` | `4046` | `0` |
| `v79_teacherfree_basis16_structural_screen_5seed_20260506T190000Z` | `1657` | `0` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

关键 hash：

| file | SHA256 |
|---|---|
| `experiments/run_gafu_v79_real.py` | `f32ada50d327441b332df896b59083d7ca7f367a46143f1ac205822fd46cfc13` |
| `docs/DG-KAN_v7.9_CodeFirst_TeacherFree_AutonomousAdvantage_完整实验计划.md` | `2107fdffa55bcb8207fb407b3093101c160f4c0d5c25e6802f37490d6f0eb616` |
| baseline `v79_route_decision.json` | `9ffacd987d41dfda8d041d0d9d615578748cbe9af7bdd7e05c5b7201cf8028db` |
| baseline `v79_provenance_audit.csv` | `f6a91312a6b07c267552b410447fb332cc2f63e55cdd9646669d07f2fe5f5707` |
| baseline `p9_task_summary.csv` | `59232334e1785a3cf4cadeaeb8038551d1cf755e0200020e28ecfe25758665bc` |
| baseline `p1_significance_audit.csv` | `00612f8377cc3cdefe63d3bf586562e5e899c430c166085bc45caadc6bfeb548` |
| baseline `p2_full_gradient_correctness.csv` | `5d6577113c8d9778168ea32706b7e237db798b11765e2331b15e2ef0783780f0` |
| structural screen `v79_route_decision.json` | `f93ac7b3336cb137031d9d9c9384202290510973665efc045291c2944c1c716e` |
| structural screen `v79_provenance_audit.csv` | `6d60e87538bbb127368755782bc329ac8448d144e8446101b6a956c28b3911c8` |
| basis16 screen `v79_route_decision.json` | `5c662a3c6e529649d9b6fd73d585935c3b1f880d4c0274adf49a72aa9e42d294` |
| basis16 screen `v79_provenance_audit.csv` | `474f02e36ccfc25f47796b04c62551f33cbb1daf1ab8ad03b6611c822acc0b7a` |

## 10. 最终结论

v7.9 没有达成 teacher-free official target：

```text
Official candidate = M13
external_teacher_used = false
CodeContractPass = true
TeacherLeakAuditPass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = false
FullGridS2Pass = false in main run
TimeAUCPass = false
success_v79_minimum = false
success_v79_formal = false
```

机制结论：

1. v7.9 解决了一个重要的工程审计问题：candidate metadata、teacher leak audit、contract validator 和 route decision 都显式落盘，official candidate 不再靠人工解释。
2. `M13` 仍是 10-seed official teacher-free best：macro gap `+0.01784`，CI95 low `+0.01224`，test gap `+0.02298`，NLL/ECE 均优于 MLP。
3. `M12` 的 `+0.02057` 仍然存在，但因为 `external_teacher_used=1`，只能是 diagnostic assisted result。
4. 失败后做了两个 no-external-teacher structural repair：`A2S/M9` 最高到 `+0.019921875`，非常接近但仍低于 hard gate，且 small-grid S2 不稳定。
5. basis16 没有带来进一步 macro 提升；`M13` small-grid 可过 S2/S1，但 macro 仍不足。
6. 当前 blocker 已收敛为：teacher-free representation/self-bootstrap 还差约 `8e-05` 到 `0.0022` 的 macro gap，同时 system runtime/S2/TimeAUC 仍不稳定。
7. 下一步不应回到 C3 teacher、loss/classwise/sampler，而应实现真正 no-external-teacher self-bootstrap/EMA 或新的 representation primitive，并同步保住 full-grid S2/TimeAUC。

最终一句话：

> v7.9 没有成功，但失败位置更干净了：代码层已经能严格区分 official teacher-free 和 assisted diagnostic；teacher-free PureKAN-NG 的实测最好结果已经逼近 `+0.02`，但仍没有跨线。当前不能宣称 official autonomous advantage，下一步必须做真实 self-bootstrap 或表示层修复，而不是借 C3 teacher 补差。
