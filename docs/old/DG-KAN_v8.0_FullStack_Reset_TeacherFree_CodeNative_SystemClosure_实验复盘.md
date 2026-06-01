# DG-KAN v8.0 FullStack Reset TeacherFree CodeNative SystemClosure 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填结论。未实现或未测的阶段只记录为 `not_implemented` / `not_run` / `metric_unavailable`。

## 1. 第一阶段目标状态（TF7/TF8）

第一阶段 10-seed confirmation run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_teacherfree_nearpass_10seed_20260506T203000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,A2S,M9,TF7 \
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
| real-only / no-fake / no-proxy | 达成 | `v80_provenance_audit.csv`: `rows_checked=3838`, fake/proxy nonzero `0` |
| CodeNativePass | 达成 | `candidate_registry_v2.csv`, `candidate_factory_audit.csv`, `contract_validator_v2.csv` 落盘；route `code_native_pass=1` |
| TeacherContractV2Pass | 达成 | official candidates `external_teacher_used=0`, `external_teacher_logits_used=0`, `external_teacher_forward_used=0` |
| StrictPass | 达成 | best official `TF7` strict KAN head，manual forward/backward/update `1/1/1`，non-KAN params `0` |
| GradPass | 达成 | `TF7` full gradient `6/6` pass，max relerr `6.32e-05` |
| TeacherFreeMacroSignificantPass | 未达成 | best official `TF7` macro val gap `+0.01862 < +0.0200` |
| FullGridS2Pass | 未达成 | `TF7` full-grid memory max `1.10035`，step max `1.65238` |
| FullGridS1Pass | 未达成 | `TF7` memory/step 均未满足 S1 |
| TimeAUCPass / ProfilerPass | 未达成 | 本轮未闭合 time/profiler；`profiler_pass=0` |
| v8.0 minimum success | 未达成 | 缺 teacher-free macro gate 与 FullGridS2 |
| v8.0 formal success | 未达成 | 缺 macro、S2/S1、TimeAUC、Profiler |

最终判断：

> v8.0 没有达成 teacher-free code-native system closure。代码层的 official/diagnostic 区分更干净，新增 `TF7/TF8` 均为 no-external-teacher real code path；但 10-seed 最佳 official `TF7` 只到 `+0.01862`，没有跨过 `+0.0200` hard gate，且 full-grid S2 也失败。

追加 P4/P6、RR3/RR8/RR9/RR10 code-native primitive repair、RR11/RR12 torch.compile helper probes、buffer-clone repair、RR13 Triton transform smoke 与 RR14 Triton-backward probe 后，最新最终判断见第 47-62 节；RR 系列 basis2 仍是当前最有信息量的新 primitive：teacher-free macro 与 GradPass 成立，但 S2 仍因 memory/step 失败；RR14 证明 Triton backward 可过 GradPass，但 step/memory 仍未进入 S2，因此 v8.0 minimum success 仍未达成。

## 2. 第一阶段 Route Decision

`v80_route_decision.json`：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "TF7",
  "external_teacher_used": 0,
  "self_teacher_used": 0,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "fullgrid_s1_pass": 0,
  "time_auc_pass": 0,
  "profiler_pass": 0,
  "success_v80_minimum": 0,
  "success_v80_formal": 0,
  "macro_gap": "0.018619791666666666",
  "ci95_low": "0.013997395833333334",
  "holm_p": "7.0821499988828e-07",
  "test_gap": "0.0228515625",
  "memory_ratio_max": 1.100352069330576,
  "step_ratio_max": 1.6523824692171676,
  "fullgrid_shape_complete": 1,
  "measured_s2_pass": 0,
  "primary_blocker": "teacher_free_macro_gap_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

说明：`v79_route` 字段是 legacy v79 postprocess 对新增 TF7/TF8 metadata 不完整的诊断，不作为 v8.0 route 判断；v8.0 route 以 `candidate_registry_v2.csv` / `teacher_contract_v2.csv` / `contract_validator_v2.csv` 为准。

## 3. 代码与候选实现

本轮新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_gafu_v80_real.py` | v8.0 wrapper；新增 CandidateSpecV2、candidate factory audit、teacher contract v2、artifact join validator、v8.0 route |

新增真实 teacher-free 候选：

| candidate | 实现 | external teacher | 说明 |
|---|---|---:|---|
| `TF7` | `M13` architecture + final SWA weight averaging | 0 | 只做权重平均，不使用 teacher logits |
| `TF8` | `M13` architecture + EMA weights | 0 | 只做 EMA 权重，不使用 teacher logits |

代码修复：

| 问题 | 修复 |
|---|---|
| screen 只测 batch 128 时 route 曾可能写成 full-grid pass | 新增 `fullgrid_shape_complete`；只有 3 datasets x batch 128/256/512 完整时才允许 FullGridS2/S1Pass |
| v8.0 route 未映射旧 significance 字段名 | 将 `bootstrap_ci95_low` / `holm_corrected_p` / `ECE_delta_macro` / `NLL_delta_macro` 映射到 route |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

## 4. 10-seed Task 结果

| candidate | official | val acc | test acc | val gap vs MLP | dataset gaps | ECE | NLL |
|---|---:|---:|---:|---:|---|---:|---:|
| `B0` | 0 | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | `0.0619` | `0.5745` |
| `TF7` | 1 | `0.8635` | `0.8186` | `+0.0186` | F-MNIST `+0.0115`, KMNIST `+0.0201`, MNIST `+0.0242` | `0.0569` | `0.4708` |
| `M13` | 1 | `0.8628` | `0.8188` | `+0.0178` | F-MNIST `+0.0092`, KMNIST `+0.0207`, MNIST `+0.0236` | `0.0585` | `0.4771` |
| `A2S` | 1 | `0.8619` | `0.8155` | `+0.0170` | F-MNIST `+0.0084`, KMNIST `+0.0199`, MNIST `+0.0227` | `0.0568` | `0.4822` |
| `M9` | 1 | `0.8619` | `0.8155` | `+0.0170` | F-MNIST `+0.0084`, KMNIST `+0.0199`, MNIST `+0.0227` | `0.0568` | `0.4822` |

Macro significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | MacroPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TF7` | `+0.01862` | `+0.013997` | `7.08e-07` | `+0.02285` | `-0.00499` | `-0.10368` | 0 |
| `M13` | `+0.01784` | `+0.01224` | `9.42e-06` | `+0.02298` | `-0.00342` | `-0.09742` | 0 |
| `A2S` | `+0.01699` | `+0.01198` | `7.63e-06` | `+0.01973` | `-0.00511` | `-0.09232` | 0 |
| `M9` | `+0.01699` | `+0.01178` | `7.63e-06` | `+0.01973` | `-0.00511` | `-0.09232` | 0 |

判断：

- `TF7` 是 10-seed 最佳 official teacher-free candidate，但仍差 `0.00138` 到 `+0.0200`。
- `TF7` 的 CI/Holm/test/NLL/ECE 均显示 positive signal 真实存在，但 hard gate 没过，不能记作成功。
- `A2S/M9` 在 5-seed screen 的 near-miss 没有在 10-seed confirmation 中保持，回落到 `+0.01699`。

## 5. Gradient Correctness

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `M13` | `6/6` | `6.32e-05` | `0.9999988` |
| `A2S` | `6/6` | `9.27e-05` | `0.9999961` |
| `M9` | `6/6` | `8.40e-05` | `0.9999961` |
| `TF7` | `6/6` | `6.32e-05` | `0.9999988` |

判断：所有 official teacher-free candidates 都通过 GradPass；没有手动改 pass。

## 6. Efficiency / System

Full-grid efficiency summary：

| candidate | memory mean | memory max | step mean | step max | FullGridS2 |
|---|---:|---:|---:|---:|---:|
| `TF7` | `1.0363` | `1.1004` | `1.3588` | `1.6524` | 0 |
| `M13` | `1.0363` | `1.1004` | `1.3758` | `1.6519` | 0 |
| `M9` | `1.0363` | `1.1004` | `1.3583` | `1.6512` | 0 |
| `A2S` | `1.0484` | `1.1163` | `1.5345` | `1.8690` | 0 |

判断：

- 本轮 full-grid S2 没有闭合；最佳 task candidate `TF7` 同时 memory max 和 step max 都超过 S2 gate。
- 这与 v7.9 / v7.8 的系统侧不稳定一致：teacher-free macro 即使未来过线，也仍需要 system closure。

## 7. 失败后自修复

### 7.1 TF7 / TF8 5-seed screen

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_teacherfree_tf7_tf8_screen_5seed_20260506T200000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,A2S,M9,TF7,TF8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | measured S2 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `A2S` | `+0.01992` | `+0.01250` | `0.00314` | `+0.02331` | 1 | near-miss，不是 pass |
| `M9` | `+0.01992` | `+0.01250` | `0.00314` | `+0.02331` | 1 | near-miss，不是 pass |
| `TF7` | `+0.01953` | `+0.01211` | `0.00473` | `+0.02760` | 1 | 没过 macro |
| `M13` | `+0.01927` | `+0.01250` | `0.00444` | `+0.02747` | 1 | 没过 macro |
| `TF8` | `+0.00404` | `-0.00469` | `1.0` | `+0.01263` | 0 | 退化 |

说明：该 screen 只测 batch 128，因此只可称为 measured/small-grid S2，不称为 FullGridS2。

### 7.2 10-seed confirmation

5-seed near-miss 后，扩成 10-seed full-grid confirmation。结果：

- `TF7` 成为最佳 official，但 macro 只有 `+0.01862`。
- `A2S/M9` 从 5-seed 的 `+0.01992` 回落到 10-seed `+0.01699`。
- full-grid S2 全部失败。

因此不能继续扩大为 success claim。

## 8. 未实现 / 未运行项

| stage | 状态 | 说明 |
|---|---|---|
| `SB0/SB1` self-bootstrap | implemented + measured | 已完成 smoke、5-seed screen、10-seed confirmation；无 external teacher |
| `SB2/SB3` self-bootstrap variants | implemented + measured | 已完成 smoke 与 5-seed screen；`SB2` 追加完成 10-seed full-grid confirmation；无 external teacher |
| `RR1/RR2/RR3/RR5` representation primitive | `not_implemented` | registry 记录，无 fake pass |
| `SYS1` system memory trim | implemented + measured | 已完成 smoke 与 5-seed full-grid screen；质量过线但 S2 仍失败；未写 fake S1 ratio |
| P8 phase-clean profiler | `not_run` | 本轮未做 phase-mapped timing |
| P10 true kernel profiler | `not_run` | 未写 ProfilerPass |
| P12 fused/streaming package | `not_implemented` | 没有 teacher-free macro + S2 candidate，不写 fused success |

## 9. No-Fake / No-Proxy 审计

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_smoke` | `95` | `0` |
| `v80_teacherfree_tf7_tf8_screen_5seed_20260506T200000Z` | `2341` | `0` |
| `v80_teacherfree_nearpass_10seed_20260506T203000Z` | `3838` | `0` |
| `tmp_v80_sb_smoke` | `100` | `0` |
| `v80_self_bootstrap_sb0_sb1_screen_5seed_20260506T213000Z` | `1963` | `0` |
| `v80_self_bootstrap_sb0_confirm_10seed_20260506T220000Z` | `3092` | `0` |
| `tmp_v80_sb23_smoke` | `116` | `0` |
| `tmp_v80_sys1_smoke` | `116` | `0` |
| `v80_self_bootstrap_sb2_sb3_screen_5seed_20260506T230000Z` | `2354` | `0` |
| `v80_self_bootstrap_sb2_confirm_10seed_20260506T233000Z` | `2362` | `0` |
| `v80_sys1_cache_trim_screen_5seed_20260506T235000Z` | `1229` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `cc6a2fe5a2dc1aa01a69f3ee4e0e8b093a21f64510698bd6f02a8c24a8f35968` |
| `docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md` | `d34543da27565ad559a6d3c5b1c5e4660d0eec70e389c11a943931a4876fcf8f` |
| `v80_route_decision.json` | `e1b2c912d6cdbbbd2ad239b65f3f4913d80ead193c6b27302b4bc61e376c7c70` |
| `v80_provenance_audit.csv` | `c258c810e2671f629ae3b498386ae975cddbcd5b44ce0e4328bee8fc3016ac1d` |
| `p9_task_summary.csv` | `a50b1bbc2d6dbef790f5a9b2006b45bb15113c55b510bf986e183e1b4db56c21` |
| `p1_significance_audit.csv` | `2b6022b2c21d145a9dcc97f0fcaa3027ec79187af44ae658d6f3717ab17d2431` |
| `p2_full_gradient_correctness.csv` | `f29e247010673b5c62408bb5cc27f529109756bf61ed3f145aa3b1f7348da5f8` |
| `p10_efficiency_summary.csv` | `8fa814c282fca396983896fba50e86abdb2a92d7cbcb39c15cdb93b534f11ac8` |
| `candidate_registry_v2.csv` | `da5570056c8591b2b64fb1b55ca46fe57dfe46cd811895e7b7eecd24fae86557` |
| `teacher_contract_v2.csv` | `52fb02961cb50505daf32eec1b77e3138448f11739a4e2d5a3b504f9a4954a5e` |

追加 self-bootstrap 10-seed confirmation 关键 hash：

| file | SHA256 |
|---|---|
| `v80_route_decision.json` | `c4d71f30d4fd05d9fa867b9410d11526cb6918101736603f17dae2328b4e5458` |
| `v80_provenance_audit.csv` | `de40f2b7810ab09cdc05c0d62bee6ec4db3f674c6183e82253f9105b3ba164f0` |
| `p9_task_summary.csv` | `cb8dbfa7cbbfb226450f938e2d22e4617ce02d753cbabb489647d257c94e86df` |
| `p1_significance_audit.csv` | `ed2e41f2c3c31633e35626d1803d3c43df256d48c6b0babaa551b6eab9fcab1f` |
| `p2_full_gradient_correctness.csv` | `218745294a66d3430b1ce2e7a61b5749a2f5702b631838289258a57a8464c8eb` |
| `p10_efficiency_summary.csv` | `a6c884df9795f66894d7db3f7103ab91a33f1bd8b8f6268e9ec5bf03002c6483` |
| `candidate_registry_v2.csv` | `01a6fbdc944b562bdd84c68c89de9e1c53a777c8c2efe872a6ef4d6a0e9b9ec1` |
| `teacher_contract_v2.csv` | `bfdf2cc1177bf1f4f8a999504d7b0461be233b5596d318d95c02d8cbfce74861` |

## 10. 追加执行 P4 Self-Bootstrap

TF7/TF8 后继续实现并执行 no-external-teacher self-bootstrap。该追加不是 C3 external teacher，也不是 loss/classwise/sampler/focal/margin/class weight 调参。

新增候选：

| candidate | self teacher | external teacher | 实现 |
|---|---:|---:|---|
| `SB0` | 1 | 0 | M13 + same-run EMA self-bootstrap，T=2，alpha=0.25 |
| `SB1` | 1 | 0 | M13 + delayed snapshot self-bootstrap，T=2，alpha=0.25 |

实现审计：

- `SB0/SB1` 的 teacher logits 来自 same-run self teacher，不读取 C3 logits，不执行 C3 forward。
- `teacher_contract_v2.csv` 中 `external_teacher_used=0`，`self_teacher_used=1`。
- gradient checker 使用 same-run self snapshot logits 作为 stop-gradient reference，验证 self-bootstrap objective 的手写 logits gradient。

### 10.1 5-seed screen

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_self_bootstrap_sb0_sb1_screen_5seed_20260506T213000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,TF7,SB0,SB1 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | measured S2 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SB0` | `+0.02018` | `+0.01380` | `0.00223` | `+0.02591` | `+0.09578` | `-0.00204` | 1 | macro 过线但只测 batch 128，不能记 full-grid success |
| `TF7` | `+0.01953` | `+0.01211` | `0.00403` | `+0.02760` | `+0.00021` | `-0.09711` | 1 | near-miss |
| `M13` | `+0.01927` | `+0.01250` | `0.00388` | `+0.02747` | `+0.00101` | `-0.09217` | 1 | near-miss |
| `SB1` | `+0.01771` | `+0.01107` | `0.00403` | `+0.02669` | `-0.00355` | `-0.09229` | 1 | macro fail |

判断：

- `SB0` 是第一个 no-external self-bootstrap 5-seed macro 过线信号。
- 但该 run 只测 batch 128，`fullgrid_shape_complete=0`，不能记为 v8.0 minimum success。
- `SB0` 的 ECE 明显变差，说明 self-bootstrap 的 objective 形态有校准风险。

### 10.2 10-seed full-grid confirmation

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_self_bootstrap_sb0_confirm_10seed_20260506T220000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,TF7,SB0 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "SB0",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.019596354166666666",
  "ci95_low": "0.014518229166666667",
  "holm_p": "6.255305466958242e-07",
  "test_gap": "0.0224609375",
  "memory_ratio_max": 1.065686781396952,
  "step_ratio_max": 1.380622896742841,
  "no_fake": true,
  "no_proxy": true
}
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | ECE | NLL |
|---|---:|---:|---:|---|---:|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | `0.0619` | `0.5745` |
| `SB0` | `0.8645` | `0.8182` | `+0.0196` | F-MNIST `+0.0111`, KMNIST `+0.0229`, MNIST `+0.0248` | `0.1530` | `0.5659` |
| `TF7` | `0.8635` | `0.8186` | `+0.0186` | F-MNIST `+0.0115`, KMNIST `+0.0201`, MNIST `+0.0242` | `0.0569` | `0.4708` |
| `M13` | `0.8628` | `0.8188` | `+0.0178` | F-MNIST `+0.0092`, KMNIST `+0.0207`, MNIST `+0.0236` | `0.0585` | `0.4771` |

Macro / calibration：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | MacroPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SB0` | `+0.01960` | `+0.01452` | `6.26e-07` | `+0.02246` | `+0.09108` | `-0.00860` | 0 |
| `TF7` | `+0.01862` | `+0.013997` | `5.31e-07` | `+0.02285` | `-0.00499` | `-0.10368` | 0 |
| `M13` | `+0.01784` | `+0.01224` | `6.86e-06` | `+0.02298` | `-0.00342` | `-0.09742` | 0 |

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min | external teacher | self teacher |
|---|---:|---:|---:|---:|---:|
| `SB0` | `6/6` | `8.90e-05` | `0.999999` | 0 | 1 |
| `TF7` | `6/6` | `6.32e-05` | `0.999999` | 0 | 0 |
| `M13` | `6/6` | `6.32e-05` | `0.999999` | 0 | 0 |

Efficiency：

| candidate | memory mean | memory max | step mean | step max | FullGridS2 |
|---|---:|---:|---:|---:|---:|
| `SB0` | `1.0247` | `1.0657` | `1.3275` | `1.3806` | 0 |
| `TF7` | `1.0247` | `1.0657` | `1.3329` | `1.3618` | 0 |
| `M13` | `1.0247` | `1.0657` | `1.3371` | `1.3864` | 0 |

判断：

- 10-seed confirmation 中 `SB0` 取代 `TF7` 成为当前最佳 official teacher-free candidate。
- `SB0` 的 macro gap 为 `+0.019596`，仍低于 `+0.0200` hard gate，不能记为 success。
- `SB0` 相比 `M13` 提升约 `+0.00176`，证明 self-bootstrap 有真实 task-side 作用，但还不够。
- `SB0` 的 ECE delta `+0.09108` 明显变差；即使 macro 接近，也暴露出校准风险。
- FullGridS2 失败来自 memory max `1.0657 > 1.05`，step 已低于 S2 gate。

## 11. 追加后最终结论

v8.0 完整实验计划没有全部完成，也没有达成 minimum / formal success：

```text
CodeNativePass = true
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = false
FullGridS2Pass = false
FullGridS1Pass = false
TimeAUCPass = false
ProfilerPass = false
success_v80_minimum = false
success_v80_formal = false
```

追加后的机制结论：

1. v8.0 进一步修正了工程审计层：candidate spec、teacher contract、contract validator、artifact join 和 route 都由代码落盘，不靠人工解释。
2. `SB0` 是追加后最强 official teacher-free candidate：10-seed macro gap `+0.019596`，CI95 low `+0.01452`，Holm p `6.26e-07`，test gap `+0.02246`，strict/GradPass 成立。
3. `SB0` 仍没有跨过 `+0.0200` teacher-free macro hard gate，差约 `0.000404`；不能四舍五入为 autonomous official advantage。
4. `SB0` 的 5-seed `+0.02018` 没有在 10-seed full-grid 中保持，因此不能把 screen 当最终成功。
5. `SB0` 明确证明 no-external self-bootstrap 有真实 task-side 增益：相对 `M13` 提升约 `+0.00176`，但同时带来 ECE 明显恶化。
6. full-grid S2 仍失败：`SB0` memory max `1.0657 > 1.05`；step 已满足 S2，但 memory 仍未 closure。
7. `SB2/SB3`、RR representation primitive、SYS memory trim、phase-mapped profiler、fused/streaming package 仍未实现或未运行；没有补 fake/proxy row。
8. 当前 blocker 收敛为：需要改进 no-external self-bootstrap 的校准/稳定性，或实现新的 representation primitive，同时修复 full-grid memory S2/TimeAUC；不能回到 C3 external teacher，也不能用 fake/proxy 补结果。

最终一句话：

> v8.0 还没有完成全部计划，也没有成功；但追加 P4 后失败位置更接近核心：`SB0` 不靠 external teacher 已把 teacher-free macro gap 推到 `+0.019596`，但 10-seed 仍差 `0.000404`，且 ECE 与 full-grid memory S2 失败。下一步应继续做真实 no-external self-bootstrap 稳定化或 representation primitive，并同步修 memory/TimeAUC；不能把 5-seed near-miss 或 external C3 teacher 当成功。

## 12. 追加执行 P4：SB2 / SB3 Self-Bootstrap

`SB0` 10-seed 仍未跨过 `+0.0200` 后，继续实现两个 no-external-teacher self-bootstrap 变体：

| candidate | self teacher | external teacher | 实现 |
|---|---:|---:|---|
| `SB2` | 1 | 0 | best-validation snapshot self-bootstrap，T=2，alpha=0.25 |
| `SB3` | 1 | 0 | dual-view consistency self-bootstrap，T=2，alpha=0.25 |

实现审计：

- `SB2/SB3` 不读取 C3 logits，不执行 C3 forward。
- `teacher_contract_v2.csv` 中 official `SB2/SB3` 的 `external_teacher_used=0`，`external_teacher_logits_used=0`，`external_teacher_forward_used=0`。
- gradient checker 使用 same-run self snapshot logits 作为 stop-gradient reference，验证 self-bootstrap objective 的手写 gradient。

### 12.1 5-seed screen

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_self_bootstrap_sb2_sb3_screen_5seed_20260506T230000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,TF7,SB0,SB2,SB3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | measured S2 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SB2` | `+0.02174` | `+0.01615` | `3.05e-04` | `+0.02552` | `-0.00183` | `-0.10012` | 1 | 5-seed macro 过线，小网格 S2 |
| `SB0` | `+0.02018` | `+0.01380` | `2.23e-03` | `+0.02591` | `+0.09578` | `-0.00204` | 1 | macro 过线但校准差 |
| `SB3` | `+0.01966` | measured positive | measured positive | measured positive | measured | measured | measured | near-miss |

判断：

- `SB2` 是首个同时显示 macro 过线、ECE/NLL 不劣化的 no-external self-bootstrap candidate。
- 但该 screen 只测 batch 128，`fullgrid_shape_complete=0`，不能记为 v8.0 minimum success。

### 12.2 10-seed full-grid confirmation

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_self_bootstrap_sb2_confirm_10seed_20260506T233000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,SB2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

随后同目录复用真实 task rows 重算 grad/efficiency：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_self_bootstrap_sb2_confirm_10seed_20260506T233000Z \
  --device auto \
  --candidates B0,M13,SB2 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "SB2",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.021028645833333335",
  "ci95_low": "0.0166015625",
  "holm_p": "7.176345590368165e-09",
  "test_gap": "0.0224609375",
  "memory_ratio_max": 1.100352069330576,
  "step_ratio_max": 1.3745004984976652,
  "no_fake": true,
  "no_proxy": true
}
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | ECE | NLL |
|---|---:|---:|---:|---|---:|---:|
| `B0` | `0.8449` | `0.7958` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | `0.0619` | `0.5745` |
| `M13` | `0.8628` | `0.8188` | `+0.0178` | F-MNIST `+0.0092`, KMNIST `+0.0207`, MNIST `+0.0236` | `0.0585` | `0.4771` |
| `SB2` | `0.8660` | `0.8182` | `+0.0210` | F-MNIST `+0.0166`, KMNIST `+0.0232`, MNIST `+0.0232` | `0.0565` | `0.4675` |

Macro / calibration：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | v8.0 macro pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SB2` | `+0.02103` | `+0.01660` | `7.18e-09` | `+0.02246` | `-0.00542` | `-0.10700` | 1 |
| `M13` | `+0.01784` | `+0.01224` | `5.07e-06` | `+0.02298` | `-0.00342` | `-0.09742` | 0 |

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min | external teacher | self teacher |
|---|---:|---:|---:|---:|---:|
| `SB2` | `6/6` | `8.90e-05` | `0.999999` | 0 | 1 |
| `M13` | `6/6` | `6.32e-05` | `0.999999` | 0 | 0 |

Efficiency：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | FullGridS2 |
|---|---:|---:|---:|---:|---:|---:|
| `SB2` | `1.0363` | `1.1004` | `1.3291` | `1.3745` | `6/9` | 0 |
| `M13` | `1.0363` | `1.1004` | `1.3243` | measured | `6/9` | 0 |

`SB2` per-shape S2：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.9933` | `1.3389` | 1 |
| MNIST | 256 | `1.0152` | `1.3373` | 1 |
| MNIST | 512 | `1.1004` | `1.3326` | 0 |
| Fashion-MNIST | 128 | `0.9933` | `1.3355` | 1 |
| Fashion-MNIST | 256 | `1.0152` | `1.3745` | 1 |
| Fashion-MNIST | 512 | `1.1004` | `1.3428` | 0 |
| KMNIST | 128 | `0.9933` | `1.2300` | 1 |
| KMNIST | 256 | `1.0152` | `1.3378` | 1 |
| KMNIST | 512 | `1.1004` | `1.3324` | 0 |

判断：

- `SB2` 真实达成 teacher-free macro quality gate：macro gap、CI、Holm、test gap、ECE/NLL 都过 v8.0 质量口径。
- `SB2` 不能记为 v8.0 minimum success，因为 FullGridS2Pass 仍为 0。
- S2 fail 已收敛为 bs512 memory：step 全 shape 均低于 `1.50`，但 bs512 memory `1.1004 > 1.05`。

## 13. 追加执行 P6：SYS1 Cache-Trim Repair

`SB2` 10-seed 质量过线但 S2 memory 失败后，继续实现系统侧 `SYS1`：

| candidate | objective | external teacher | system change |
|---|---|---:|---|
| `SYS1` | `SB2` best-val self-bootstrap | 0 | 使用 recompute/cache-trim fused stack，减少 hidden-y cache 常驻 |

### 13.1 Smoke

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/tmp_v80_sys1_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,SB2,SYS1 \
  --seeds 0 \
  --task-steps 4 \
  --trace-every 1 \
  --bench-batch-sizes 128 \
  --bench-warmup 1 \
  --bench-reps 3 \
  --grad-batch-sizes 8 \
  --bootstrap-reps 100
```

结果：

```text
rows_checked = 116
fake/proxy nonzero = 0
SYS1 grad pass = 1
SYS1 max relerr = 5.81e-05
SYS1 measured memory ratio bs128 = 0.9899
SYS1 measured step ratio bs128 = 1.4930
```

### 13.2 5-seed full-grid screen

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys1_cache_trim_screen_5seed_20260506T235000Z \
  --fresh \
  --device auto \
  --candidates B0,SB2,SYS1 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route:

```json
{
  "best_official_candidate_id": "SYS1",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "macro_gap": "0.022005208333333335",
  "ci95_low": "0.0171875",
  "holm_p": "4.124388769713541e-06",
  "test_gap": "0.022395833333333334",
  "memory_ratio_max": 1.0593840017726568,
  "step_ratio_max": 3.4451982716159333,
  "no_fake": true,
  "no_proxy": true
}
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap |
|---|---:|---:|---:|---:|---:|---:|
| `SB2` | `0.8660` | `0.8204` | `+0.02174` | measured positive | measured positive | `+0.02552` |
| `SYS1` | `0.8663` | `0.8173` | `+0.02201` | `+0.01719` | `4.12e-06` | `+0.02240` |

Gradient / efficiency：

| candidate | grad pass | max relerr | memory mean | memory max | step mean | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SYS1` | `3/3` | `8.90e-05` | `1.0204` | `1.0594` | `1.5328` | `3.4452` | partial fail |
| `SB2` | `3/3` | `8.90e-05` | `1.0247` | measured | `1.1925` | measured | measured |

判断：

- `SYS1` 质量侧没有退化，5-seed macro gap 甚至略高于 `SB2`。
- `SYS1` 没有修复系统 gate：bs512 memory 仍 `>1.05`，且 recompute path 引入明显 step 抖动，Fashion-MNIST bs256 step 达到 `3.4452`。
- 因为 full-grid S2 在 5-seed screen 已失败，未扩展为 10-seed official confirmation。

## 14. 当前最终结论（SB2 / SYS1 后）

v8.0 全部实验仍未完成，minimum / formal success 仍未达成：

```text
CodeNativePass = true
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = true for SB2
FullGridS2Pass = false
FullGridS1Pass = false
TimeAUCPass = false
ProfilerPass = false
success_v80_minimum = false
success_v80_formal = false
```

当前最强 official candidate：

```text
candidate = SB2
external_teacher_used = false
self_teacher_used = true
macro gap = +0.02103
CI95 low = +0.01660
Holm p = 7.18e-09
test gap = +0.02246
ECE delta = -0.00542
NLL delta = -0.10700
GradPass = true
FullGridS2Pass = false
```

机制结论：

1. v8.0 的 teacher-free quality gate 已被 `SB2` 真实跨过：这是 no-external-teacher、code-native、strict、gradient-correct 的 positive result。
2. 但 v8.0 minimum success 不是只看 quality；`SB2` full-grid S2 失败，因此不能记为 system closure。
3. `SB2` 相比 `M13` 的提升约 `+0.00319`，说明 best-val self-bootstrap 比 SB0 更有效，并且没有 SB0 的 ECE collapse。
4. S2 blocker 已经从 “macro 还差一点” 转移到 “bs512 memory live set 过大”：`SB2` step 全 shape 过 S2，但 bs512 memory ratio 固定在 `1.1004`。
5. `SYS1` 证明单纯 recompute/cache-trim 不是闭合解：它略降 memory mean，但 bs512 memory 仍不过，且 step 变得不稳定。
6. 未完成项仍包括 RR representation primitive、phase-mapped profiler、true fused/streaming package、TimeAUC closure、S1 closure；这些没有 fake/proxy row，也没有被写成成功。

最终一句话：

> v8.0 还没有完成全部计划，也没有达成 minimum system closure；但关键质量问题已有真实突破。`SB2` 不靠 external teacher 跨过了 `+0.02` teacher-free macro gate，并通过 strict/Grad/ECE/NLL 审计；失败点现在明确是 full-grid S2 memory，尤其 bs512 live set。下一步应集中做真实 memory live-set / fused streaming package，而不是继续借 C3 teacher 或补 fake/proxy。

追加关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `cc6a2fe5a2dc1aa01a69f3ee4e0e8b093a21f64510698bd6f02a8c24a8f35968` |
| `SB2 confirm v80_route_decision.json` | `f519929284100e0c087de1f3af3e888b00c1c5411c88f2ace27c46269264e53e` |
| `SB2 confirm v80_provenance_audit.csv` | `d774cbd99a1868bce1f4cd742200137a38c34cd6e114ac338e535ec46572e6c5` |
| `SB2 confirm p9_task_summary.csv` | `71131db3bf3e0c52877fd0e3c5a539cea6b6a9a7b8a99577e36f1a5329059f7c` |
| `SB2 confirm p1_significance_audit.csv` | `bf47f67e4a151a5eda0ad60ca7b16796e4a62500aca0f01d41133b769b1ca3a1` |
| `SB2 confirm p2_full_gradient_correctness.csv` | `ddbad2efa516b4f1064c6650c4042c60a32675e2171da50b5ed73b7b89f64b9c` |
| `SB2 confirm p10_efficiency_summary.csv` | `9e0f69bd360fe7873876018fae4c369180fdf52be94343224a8e7f4bee8ffe2f` |
| `SYS1 screen v80_route_decision.json` | `e5d3d7ec06fb3ce6392b31fecc7daf7ddaf5feebdd110c9f1604dc480450f6cb` |
| `SYS1 screen v80_provenance_audit.csv` | `07e5a5d037340ac79a0d857229deedd6a2f0a6e6ff39757d1ee26c1b46b3101e` |

## 15. 追加执行 P12：Hidden60 / Hidden48 System Compression Repair

`SB2` 与 `SYS1` 后，blocker 已收敛到 bs512 memory live set，因此继续做两轮 no-external-teacher system compression probe。两轮都不使用 C3 teacher，不做 loss / classwise / sampler 调参。

Hidden60 run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sb2_hidden60_system_screen_5seed_20260507T000500Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,M13,SB2 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Hidden48 run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sb2_hidden48_system_screen_5seed_20260507T001000Z \
  --fresh \
  --device auto \
  --hidden-dim 48 \
  --candidates B0,M13,SB2 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Hidden60 task / route：

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M13 hidden60` | `+0.02188` | `+0.01302` | `1.38e-03` | `+0.01992` | `-0.00286` | `-0.08853` | 1 | `6/9` |
| `SB2 hidden60` | `+0.02174` | `+0.01367` | `8.30e-04` | `+0.02005` | `-0.00259` | `-0.09986` | 1 | `6/9` |

Hidden60 efficiency：

| candidate | memory mean | memory max | step mean | step max | survivor |
|---|---:|---:|---:|---:|---|
| `M13 hidden60` | `1.0223` | route max `1.0634` | `1.1762` | route max `1.2677` | FAIL |
| `SB2 hidden60` | `1.0223` | measured bs512 fail | `1.3785` | measured step fail | FAIL |

判断：

- hidden60 可以保住 teacher-free macro gate，route best `M13 hidden60` 的 macro gap 为 `+0.02188`。
- 但 hidden60 仍没有修复 S2：bs512 memory ratio 仍到约 `1.0634`，`SB2 hidden60` 还有 step instability。
- 因此 hidden60 不能作为 v8.0 minimum success。

Hidden48 task / route：

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M13 hidden48` | `+0.01576` | `+0.01029` | `9.94e-04` | `+0.01589` | `+0.00107` | `-0.06337` | 1 | `6/9` |
| `SB2 hidden48` | `+0.01589` | `+0.00964` | `2.37e-03` | `+0.01563` | `+0.00164` | `-0.07618` | 0 | `6/9` |

Hidden48 efficiency：

| candidate | memory mean | memory max | step mean | step max | survivor |
|---|---:|---:|---:|---:|---|
| `M13 hidden48` | `1.0151` | route max `1.0560` | `1.0780` | measured S2 step ok | FAIL |
| `SB2 hidden48` | `1.0151` | route max `1.0560` | `1.1249` | route max `1.4501` | FAIL |

判断：

- hidden48 明显降低 step 和部分 memory pressure，但 bs512 memory 仍略高于 S2 gate，最大约 `1.0560`。
- hidden48 破坏 teacher-free macro：`SB2 hidden48` 只有 `+0.01589`。
- `SB2 hidden48` 还有 1 条 gradient relerr `1.0562e-04 > 1e-4`，所以 GradPass 不成立。
- 因此 hidden48 也不能作为 v8.0 解法。

追加 no-fake / no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v80_sb2_hidden60_system_screen_5seed_20260507T000500Z` | `1246` | `0` |
| `v80_sb2_hidden48_system_screen_5seed_20260507T001000Z` | `1247` | `0` |

追加 hash：

| artifact | SHA256 |
|---|---|
| `hidden60 v80_route_decision.json` | `dfa4db27822aac79105ec63ba7e5d31b217744b09af7e3aeca385872638a3fde` |
| `hidden60 v80_provenance_audit.csv` | `2092d04ab404ceb1f669aaf37b6e94c8a9b80f0f04c82d582659a6c382da76b4` |
| `hidden48 v80_route_decision.json` | `dd43541dcf0f6816e16dd036ae6232ac0df20650f93537189eba668ea6547d57` |
| `hidden48 v80_provenance_audit.csv` | `5f2b9578899b20c44724c4b84dd567256fe919954bad1b0d09d5a1b0a4cb4611` |

## 16. 当前最终结论（Hidden Compression 后）

v8.0 全部实验仍未完成，minimum / formal success 仍未达成。

当前最强质量侧结果仍是 `SB2` 10-seed confirmation：

```text
external_teacher_used = false
self_teacher_used = true
CodeNativePass = true
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = true
macro gap = +0.02103
CI95 low = +0.01660
Holm p = 7.18e-09
test gap = +0.02246
ECE delta = -0.00542
NLL delta = -0.10700
FullGridS2Pass = false
```

追加修复结论：

1. `SYS1` recompute/cache trim 没有闭合 S2：memory 仍 fail，step 还变得更不稳定。
2. hidden60 保住了 teacher-free macro，但 S2 仍失败；这说明单纯轻度 compression 不能解决 bs512 live-set。
3. hidden48 接近降低系统开销，但 macro/grad 被破坏；这说明继续压 hidden size 会损失有效表达力。
4. 当前路线已经从 v7.9 的 “teacher-free macro 未过” 推进到 “teacher-free macro 已过但 system S2 未闭合”。
5. 未完成项仍包括 true fused/streaming package、phase-mapped profiler、RR representation primitive、S1/TimeAUC closure；这些均没有被写成 fake/proxy success。

最终一句话：

> v8.0 仍没有完成全部计划，也没有达到 minimum system closure；但已取得关键质量突破。`SB2` 是真实 no-external-teacher、code-native、strict、gradient-correct 的 teacher-free macro pass candidate。失败点现在不是 C3 teacher 或 loss，而是 full-grid S2 memory，尤其 bs512 live-set；SYS1 与 hidden compression 都未闭合。下一步必须做真正 fused/streaming live-set package 或新的 code-native representation primitive，不能靠 fake/proxy 或四舍五入过线。

## 17. 继续追加执行：SYS1 hidden60 / hidden56 combo

在 hidden60 / hidden48 后，继续尝试把已实现的 `SYS1` recompute/cache-trim 系统路径与中等 hidden compression 组合。该尝试仍保持：

```text
external_teacher_used = false
self_teacher_used = true
no fake / no proxy
no C3 logits
no C3 forward
no loss / sampler / classwise / focal / margin / class weight tuning
```

### 17.1 SYS1 hidden60 combo

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys1_hidden60_combo_screen_5seed_20260507T003000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SB2,SYS1 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "SYS1",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.024739583333333332",
  "ci95_low": "0.015751953125",
  "holm_p": "0.0004480547178595149",
  "test_gap": "0.020833333333333332",
  "memory_ratio_max": 1.057424422050978,
  "step_ratio_max": 1.4844005687232527,
  "primary_blocker": "fullgrid_s2_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | v8.0 macro gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `0.8406` | `0.7954` | `0.0000` | reference | reference | `0.0000` | reference | reference | reference |
| `SB2` | `0.8624` | `0.8155` | `+0.0217` | `+0.0137` | `7.26e-04` | `+0.0201` | `-0.0026` | `-0.0999` | 1 |
| `SYS1` | `0.8654` | `0.8163` | `+0.0247` | `+0.0158` | `4.48e-04` | `+0.0208` | `-0.0028` | `-0.0931` | 1 |

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min | external teacher |
|---|---:|---:|---:|---:|
| `SB2` | `3/3` | `8.42e-05` | `0.999995` | 0 |
| `SYS1` | `3/3` | `8.42e-05` | `0.999995` | 0 |

Efficiency：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `SB2` | `1.0223` | `1.0634` | `1.2294` | `1.4259` | `6/9` | FAIL |
| `SYS1` | `1.0183` | `1.0574` | `1.2852` | `1.4844` | `6/9` | FAIL |

`SYS1 hidden60` per-shape S2：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.9870` | `0.9634` | 1 |
| MNIST | 256 | `1.0103` | `1.4116` | 1 |
| MNIST | 512 | `1.0574` | `1.2639` | 0 |
| Fashion-MNIST | 128 | `0.9870` | `1.4312` | 1 |
| Fashion-MNIST | 256 | `1.0103` | `1.2885` | 1 |
| Fashion-MNIST | 512 | `1.0574` | `1.4243` | 0 |
| KMNIST | 128 | `0.9870` | `1.4398` | 1 |
| KMNIST | 256 | `1.0103` | `0.8595` | 1 |
| KMNIST | 512 | `1.0574` | `1.4844` | 0 |

判断：

- `SYS1 hidden60` 是目前最强的 5-seed system-near candidate：macro `+0.0247`，GradPass，step 全 shape 低于 S2 step gate `1.50`。
- 但它仍没有 FullGridS2Pass，因为 bs512 memory ratio 固定在 `1.0574 > 1.05`。
- 这不是可以四舍五入的成功；v8.0 minimum success 仍为 `0`。
- `SYS1` 的 hidden-y cache 已为 `0`，bs512 仍 fail，说明剩余 blocker 不再只是 hidden-y cache。

### 17.2 SYS1 hidden56 combo

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys1_hidden56_combo_screen_5seed_20260507T010000Z \
  --fresh \
  --device auto \
  --hidden-dim 56 \
  --candidates B0,SB2,SYS1 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "SB2",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.019401041666666667",
  "ci95_low": "0.012369791666666666",
  "holm_p": "0.0011354324875725282",
  "test_gap": "0.019921875",
  "memory_ratio_max": 1.060862885039526,
  "step_ratio_max": 1.4544165405251754,
  "primary_blocker": "teacher_free_macro_gap_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | v8.0 macro gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | `0.8467` | `0.7966` | `0.0000` | reference | reference | `0.0000` | reference | reference | reference |
| `SB2` | `0.8661` | `0.8165` | `+0.0194` | `+0.0124` | `1.14e-03` | `+0.0199` | `+0.0049` | `-0.0890` | 0 |
| `SYS1` | `0.8608` | `0.8141` | `+0.0141` | `+0.0079` | `3.46e-03` | `+0.0174` | `+0.0027` | `-0.0740` | 0 |

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min | external teacher |
|---|---:|---:|---:|---:|
| `SB2` | `3/3` | `5.29e-05` | `0.999991` | 0 |
| `SYS1` | `3/3` | `5.29e-05` | `0.999991` | 0 |

Efficiency：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `SB2` | `1.0197` | `1.0609` | `1.2584` | `1.4544` | `6/9` | FAIL |
| `SYS1` | `1.0159` | `1.0553` | `1.3240` | `1.5237` | `5/9` | FAIL |

判断：

- hidden56 没有闭合 teacher-free macro gate：best `SB2` 只有 `+0.0194`。
- hidden56 也没有闭合 S2：bs512 memory 仍高于 `1.05`，`SYS1` 还有 KMNIST bs128 step `1.5237 > 1.50`。
- 该结果说明继续压 hidden size 会损失有效表达力，同时 memory 仍没有完全闭合。

### 17.3 追加审计与 hash

No-fake / no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v80_sys1_hidden60_combo_screen_5seed_20260507T003000Z` | `1229` | `0` |
| `v80_sys1_hidden56_combo_screen_5seed_20260507T010000Z` | `1230` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `hidden60 v80_route_decision.json` | `d68c0c8078ed04dd5310535a3c4f3b96b24c2716478ec6a39f737b89be4feb8c` |
| `hidden60 v80_provenance_audit.csv` | `07e5a5d037340ac79a0d857229deedd6a2f0a6e6ff39757d1ee26c1b46b3101e` |
| `hidden60 p9_task_summary.csv` | `933e0bc8c73665fc5b75daf409788db05838848327cca378b3fd2d5fec1894d9` |
| `hidden60 p10_efficiency_summary.csv` | `0664a31bfd6efa7c879bbc2384d44b32d34182df68a357c8dbe4be2031a75af1` |
| `hidden56 v80_route_decision.json` | `71014db7595a9af1d5c43c95c696109cb41fabd0c55c7825e37141aef13f89b5` |
| `hidden56 v80_provenance_audit.csv` | `9f2eb41e8d1cc25d3a288e0ff6b5860bdef7fdffb00213118151e5d510b428bb` |
| `hidden56 p9_task_summary.csv` | `ebdc7634b46dd33b7f4ae55904fcbc13fb633452ef361a3d04c6421cbd9b69ab` |
| `hidden56 p10_efficiency_summary.csv` | `a02a41cfae323b94397210015a6eb00cd90d8d4e74c2379132f096eee5c9955f` |

## 18. 当前最终结论（SYS1 combo 后）

v8.0 全部实验仍未完成，minimum / formal success 仍未达成。

当前最强 10-seed official-quality candidate 仍是 `SB2`：

```text
external_teacher_used = false
self_teacher_used = true
CodeNativePass = true
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = true
macro gap = +0.02103
CI95 low = +0.01660
Holm p = 7.18e-09
test gap = +0.02246
FullGridS2Pass = false
```

当前最强 5-seed system-near candidate 是 `SYS1 hidden60`：

```text
external_teacher_used = false
self_teacher_used = true
teacher_free_macro_pass = true
macro gap = +0.02474
CI95 low = +0.01575
Holm p = 4.48e-04
test gap = +0.02083
GradPass = true
step max = 1.4844
memory max = 1.0574
FullGridS2Pass = false
```

综合判断：

1. v8.0 已经真实完成 code-native teacher-free self-bootstrap 的关键质量突破：`SB2` 10-seed 过 teacher-free macro gate，且没有 external teacher。
2. v8.0 minimum success 仍失败，因为 `SB2` full-grid S2 不闭合，主要是 bs512 memory。
3. `SYS1 hidden60` 说明 cache trim + hidden60 可以把 step 拉进 S2，并显著增强 5-seed macro，但 bs512 memory 仍为 `1.0574`，离 S2 还差约 `0.0074`。
4. `SYS1 hidden56` 说明继续压 hidden size 会破坏 macro，并且仍不能稳定闭合 S2。
5. 已尝试的 system repair 没有使用 fake/proxy，也没有借 C3 teacher 或 loss/classwise 调参。
6. 未完成项仍包括 true fused/streaming live-set package、RR representation primitive、phase-mapped profiler、S1/TimeAUC closure；这些都不能写成成功。

最终一句话：

> v8.0 仍未完成全部计划，也没有达到 minimum system closure。最重要的正向结果是 `SB2` 已成为真实 no-external-teacher、code-native、strict、gradient-correct 的 10-seed teacher-free macro pass candidate；最新 `SYS1 hidden60` 进一步证明系统侧已接近 S2，但 bs512 memory 仍卡在 `1.0574 > 1.05`。下一步必须做真正 fused/streaming memory package 或新 code-native primitive，不能用 fake/proxy、C3 teacher 或四舍五入补差。

## 19. 追加自修复：Optimizer-State Memory Repair

`SYS1 hidden60` 的失败已经很窄：macro/grad/step 都过，但 bs512 memory `1.0574 > 1.05`。因此继续做 optimizer-state memory 方向的结构修复，而不是 loss、sampler、classwise 或 external teacher。

本轮新增候选：

| candidate | 修改 | 目的 |
|---|---|---|
| `SYS2` | `SYS1` + no-momentum SGD update | 去掉 AdamW first/second moment state，测试 optimizer state 是否足以闭合 S2 |
| `SYS3` | `SYS1` + one-state RMS adaptive update | 保留二阶缩放，去掉 first moment state，测试是否能同时保 task 与降 memory |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 19.1 SYS2 hidden60 SGD state trim

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys2_sgd_state_trim_hidden60_5seed_20260507T013000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS2 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | macro gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS1` | `0.8654` | `0.8163` | `+0.0247` | `+0.0158` | `3.36e-04` | `+0.0208` | `-0.0028` | `-0.0931` | 1 |
| `SYS2` | `0.6645` | `0.6172` | `-0.1762` | `-0.2091` | `3.52e-07` | `-0.1783` | `+0.1559` | `+0.5843` | 0 |

Gradient / efficiency：

| candidate | grad pass | max relerr | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS1` | 1 | `8.42e-05` | `1.0183` | `1.0574` | `1.3272` | `1.3888` | `6/9` | FAIL |
| `SYS2` | 1 | `8.42e-05` | `0.9952` | `1.0360` | `1.1206` | `1.2974` | `9/9` | S2 |

判断：

- `SYS2` 真实闭合了 FullGridS2：memory max `1.0360`，step max `1.2974`，S2 shapes `9/9`。
- 但 `SYS2` 的 teacher-free task 明显 collapse：macro gap `-0.1762`，ECE/NLL 也显著变差。
- 因此 optimizer-state trim 本身可以解决 memory/step，但无动量 SGD 破坏了当前 PureKAN-NG 的优化可达性，不能作为 v8.0 success。

No-fake/no-proxy：`rows_checked=1232`，fake/proxy nonzero `0`。

### 19.2 SYS3 hidden60 RMS one-state trim

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys3_rms_state_trim_hidden60_5seed_20260507T020000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | macro gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS1` | `0.8654` | `0.8163` | `+0.0247` | `+0.0158` | `4.48e-04` | `+0.0208` | `-0.0028` | `-0.0931` | 1 |
| `SYS3` | `0.8598` | `0.8141` | `+0.0191` | `+0.0112` | `1.46e-03` | `+0.0186` | `-0.0028` | `-0.0847` | 0 |

Gradient / efficiency：

| candidate | grad pass | max relerr | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS1` | 1 | `8.42e-05` | `1.0183` | `1.0574` | `1.2861` | `1.4176` | `6/9` | FAIL |
| `SYS3` | 1 | `8.42e-05` | `1.0064` | `1.0467` | `1.1979` | `1.3207` | `9/9` | S2 |

判断：

- `SYS3 hidden60` 同时保住了 GradPass 和 FullGridS2，说明 one-state RMS 是比 `SYS2` 更合理的 optimizer-state trim。
- 但 `SYS3 hidden60` macro gap 只有 `+0.0191 < +0.0200`，不能记为 TeacherFreeMacroSignificantPass。
- 该结果说明一阶动量/AdamW update 对当前 self-bootstrap task 仍有实际表达/优化收益；单纯去掉 optimizer state 会让 task 从 pass 退到 near-miss。

No-fake/no-proxy：`rows_checked=1235`，fake/proxy nonzero `0`。

### 19.3 SYS3 default64 RMS one-state trim

hidden60 的 SYS3 过 S2 但 macro 不过，因此继续测默认 hidden64，看能否恢复表达力同时保住 S2。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys3_rms_state_trim_default64_5seed_20260507T023000Z \
  --fresh \
  --device auto \
  --candidates B0,SYS1,SYS3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "SYS1",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.022005208333333335",
  "ci95_low": "0.0171875",
  "holm_p": "4.124388769713541e-06",
  "test_gap": "0.022395833333333334",
  "memory_ratio_max": 1.0593840017726568,
  "step_ratio_max": 1.4304404043179186,
  "primary_blocker": "fullgrid_s2_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | macro gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS1` | `0.8663` | `0.8173` | `+0.0220` | `+0.0172` | `4.12e-06` | `+0.0224` | `+0.0016` | `-0.0954` | 1 |
| `SYS3` | `0.8609` | `0.8191` | `+0.0167` | `+0.0103` | `2.26e-03` | `+0.0242` | `-0.0014` | `-0.0906` | 0 |

Gradient / efficiency：

| candidate | grad pass | max relerr | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS1` | 1 | `8.90e-05` | `1.0204` | `1.0594` | `1.3328` | `1.4304` | `6/9` | FAIL |
| `SYS3` | 1 | `8.90e-05` | `1.0073` | `1.0480` | `1.2313` | `1.2698` | `9/9` | S2 |

判断：

- 默认 hidden64 的 `SYS3` 仍真实闭合 S2：memory max `1.0480`，step max `1.2698`，S2 shapes `9/9`。
- 但 `SYS3` macro gap 降到 `+0.0167`，比 hidden60 SYS3 还低，不能进入 macro success route。
- `SYS1` 保住 macro `+0.0220`，但 memory max `1.0594` 仍超过 S2。
- 因此本轮没有找到 `macro pass + S2 pass` 的同一 candidate。

No-fake/no-proxy：`rows_checked=1235`，fake/proxy nonzero `0`。

### 19.4 SYS4 hidden60 AdamW half-state trim

`SYS3` 说明 one-state RMS 能过 S2 但损失 macro。为了更贴近 `SYS1` 的 AdamW 可达性，继续新增 `SYS4`：保留 AdamW two-state update 形态，但把常驻 optimizer state 存为 half precision。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys4_adamw_half_state_hidden60_5seed_20260507T030000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS4 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task / significance：

| candidate | val acc | test acc | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | macro gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS1` | `0.8654` | `0.8163` | `+0.0247` | `+0.0158` | `2.24e-04` | `+0.0208` | `-0.0028` | `-0.0931` | 1 |
| `SYS4` | `0.1010` | `0.0997` | `-0.7396` | `-0.7587` | `8.64e-19` | `-0.6957` | `-0.0602` | `metric_unavailable` | 0 |

Gradient / efficiency：

| candidate | grad pass | max relerr | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS1` | 1 | `8.42e-05` | `1.0183` | `1.0574` | `1.3388` | `1.4583` | `6/9` | FAIL |
| `SYS4` | 1 | `8.42e-05` | `1.0088` | `1.0468` | `1.4383` | `1.5730` | `7/9` | FAIL |

判断：

- `SYS4` 的 gradient checker 仍通过，但 task 退化到接近随机，说明 half-state AdamW 在当前实现下破坏了优化轨迹。
- `SYS4` memory max `1.0468` 进入 S2 memory，但 step max `1.5730` 超过 S2 step gate，且 task collapse，因此不能作为系统路线。
- 这个结果比 `SYS3` 更差；half precision optimizer state 不是可用修复。

No-fake/no-proxy：`rows_checked=1238`，fake/proxy nonzero `0`。

### 19.5 SYS5 hidden60 AdamW v-half-state trim

`SYS4` 半精度保存 `m/v` 直接 collapse 后，继续做更温和版本：只把 AdamW second moment `v` 设为 half precision，first moment `m` 仍保留 full precision。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys5_adamw_v_half_state_hidden60_5seed_20260507T033000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS5 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS5` | `-0.7475` | `-0.7702` | `1.05e-17` | `-0.6983` | 1 | `1.0521` | `1.6269` | `5/9` | task collapse + S2 fail |

判断：

- 只压缩 `v` state 到 half precision 仍然严重破坏 task，NLL 也爆炸到 `2.96e12` 量级。
- `SYS5` memory max `1.0521` 仍略高于 S2 memory gate，step max `1.6269` 也超 S2。
- 该路线停止；低精度 optimizer state 对当前 self-bootstrap 不可靠。

No-fake/no-proxy：`rows_checked=1241`，fake/proxy nonzero `0`。

### 19.6 SYS6 hidden60 AdamW CPU-state offload

`SYS5` 说明低精度 state 不可用。为了分离 memory 与 runtime，继续做诊断型 `SYS6`：保留 FP32 AdamW state，但将 `m/v` offload 到 CPU。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys6_adamw_cpu_state_hidden60_5seed_20260507T040000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS6` | `+0.0215` | `+0.0122` | `2.04e-03` | `+0.0203` | 1 | `1.0360` | `15.0571` | `0/9` | macro + memory pass, step fail |

判断：

- `SYS6` 保住了 teacher-free macro gate：val gap `+0.0215`，test gap `+0.0203`。
- `SYS6` 也真实闭合了 S2 memory：memory max `1.0360`。
- 但 CPU offload 带来极大 runtime 代价：step max `15.0571`，FullGridS2Pass 仍为 0。
- 这个结果说明 AdamW state 是可行动的 GPU memory source；但 CPU offload 不是系统解法，只是定位实验。

No-fake/no-proxy：`rows_checked=1244`，fake/proxy nonzero `0`。

### 19.7 SYS7 hidden60 AdamW v-CPU-state offload

`SYS6` 证明 full CPU offload 太慢。继续做更窄的 offload：只把 AdamW second moment `v` 放 CPU，first moment `m` 留 GPU。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys7_adamw_v_cpu_state_hidden60_5seed_20260507T043000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS7 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS7` | `+0.0221` | `+0.0155` | `3.77e-05` | `+0.0243` | `-0.0043` | `-0.0905` | 1 | `1.0467` | `16.0406` | `0/9` | macro + memory pass, step fail |

判断：

- `SYS7` 是目前最干净的定位结果之一：teacher-free macro pass、GradPass、S2 memory pass 同时成立。
- 但 `SYS7` step max `16.0406`，说明哪怕只 offload `v` state，per-step CPU/GPU transfer 仍完全不可接受。
- 因此下一步应做 GPU-native fused/streaming state/cache，而不是继续 CPU offload。

No-fake/no-proxy：`rows_checked=1247`，fake/proxy nonzero `0`。

### 19.8 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `75c136c3697faad93b1a8ab5a608f0d986f8ea261ebe4bea1fde36dccb3faed5` |
| `experiments/run_gafu_v72_real.py` | `f3e97fc0bc6b755512b12228a0ec200fec80ef69e5245930cb2354e408740284` |
| `SYS2 route` | `49299f95901c54df0d9d22d3fcb2d83bb8c3a35d44c8d64bb142143bb07ac7b2` |
| `SYS2 provenance` | `69932e8d3d83e2fe4ccace939227e2388148786dfd9107d7c097655274ec7e67` |
| `SYS2 p9_task_summary.csv` | `78a1d86972b9967efa49d8f6e55241cdf75928ca4dbbc236e0ae54c0db5f6181` |
| `SYS2 p10_efficiency_summary.csv` | `911a2c2e33871138dfe115f6e40de787e57fd0e95496728d00d5481090f44273` |
| `SYS3 hidden60 route` | `56e08cc3cefdccedd6449dba115f90e8bfa61bdfb2707c558e431741f0d99b19` |
| `SYS3 hidden60 provenance` | `83998d7fdb89c0f81ef45a7741fe61aa99f767f132cb012e51b83bde5618ea16` |
| `SYS3 hidden60 p9_task_summary.csv` | `2c83167ff65be415b03c5db8d0670c40a5ffb2726d540f3428ff2062de4b14a4` |
| `SYS3 hidden60 p10_efficiency_summary.csv` | `901930ee05765b5b11c7d68661412c9011dcc2eef1e2d833bed8fe576767127a` |
| `SYS3 default64 route` | `f1fc92f767411ad8bf676712cd6e1f1752d7c38bb014394a60d65374b03118ea` |
| `SYS3 default64 provenance` | `83998d7fdb89c0f81ef45a7741fe61aa99f767f132cb012e51b83bde5618ea16` |
| `SYS3 default64 p9_task_summary.csv` | `5fb0c74ec7b7c6ba574e64fe2346b2c8e629f939cfc6845d902075962ca8e82b` |
| `SYS3 default64 p10_efficiency_summary.csv` | `0e95556f59d3def5b0d35db794c1a99a5166ae0fa76f031f7e7d354635f290a8` |
| `SYS4 route` | `a00dd73b4a84d3d3cd0e525853b91ffa14a71427fae3b1f2e48a3809fbd04742` |
| `SYS4 provenance` | `f3880de279433299e921d8c2c1cf24cc8e18fb9486d48e8e026139848aa62cbf` |
| `SYS4 p9_task_summary.csv` | `abffba81d241c0acdf404374263a73e9afcdf54f9f8f7f6d73ef949baad53bc6` |
| `SYS4 p10_efficiency_summary.csv` | `171e7a191a24c145ea9fc407ac701963261840ee160782300fb942ff50b06d66` |
| `SYS5 route` | `ef43666060164c7be18916c99472d2f692bfc3a918d84d09aab89409be04f975` |
| `SYS5 provenance` | `1bf72a8255c2738d22cb6c10837412e8f9a6cbd6500215fa72e4968ffdf49acf` |
| `SYS5 p9_task_summary.csv` | `9eba0fd07f0226772e5a54da29fc615408553bfd70064fffd9a31ed81265320d` |
| `SYS5 p10_efficiency_summary.csv` | `488c35b2cc44bfffda3c8d29a7e0fd287e215905e6499aac3152b23cc26ba356` |
| `SYS6 route` | `6e9bc30e35e250c0ad833815e415dfce8a135934f14c641cad006edd74a497fd` |
| `SYS6 provenance` | `631777c717d8cf71045b337952a3cfe03b278598acf3a02ed3f75a16911f52a2` |
| `SYS6 p9_task_summary.csv` | `a8ce2e16a2cea539ffe96e9a1873b5b85cdc0579e66a1b69aa47e73ce94bfc7e` |
| `SYS6 p10_efficiency_summary.csv` | `2a912dcd54de4de6722f5eddfd9e774301d64d9f4312544193d2c4d7631a009b` |
| `SYS7 route` | `b2abd3e8a9077a5144ba51427ff894c9c12b804e21597047655e1f0c7dfde06f` |
| `SYS7 provenance` | `5f2b9578899b20c44724c4b84dd567256fe919954bad1b0d09d5a1b0a4cb4611` |
| `SYS7 p9_task_summary.csv` | `e4a3c306994591221e3b4ec8017a406b1ffd8660a95fa759431acebe2b340633` |
| `SYS7 p10_efficiency_summary.csv` | `f242cea0aada74e6babbec4b2b827c7129c4bbbc27c6ea7816870fafa9fee820` |

## 20. 更新后最终结论

截至 `SYS2/SYS3/SYS4/SYS5/SYS6/SYS7` optimizer-state memory repair，v8.0 全部实验仍未完成，minimum / formal success 仍未达成。

当前路线图更清楚：

| candidate | setting | teacher-free macro | FullGridS2 | 结论 |
|---|---|---:|---:|---|
| `SB2` | 10-seed default64 | pass, `+0.0210` | fail, memory max `1.1004` | 最强 10-seed teacher-free macro candidate |
| `SYS1` | 5-seed hidden60 | pass, `+0.0247` | fail, memory max `1.0574` | 最强 5-seed system-near candidate |
| `SYS2` | 5-seed hidden60 SGD | fail, `-0.1762` | pass, memory max `1.0360` | 证明 optimizer-state trim 可闭合 S2，但 task collapse |
| `SYS3` | 5-seed hidden60 RMS | fail, `+0.0191` | pass, memory max `1.0467` | S2 pass + near macro，但未过 gate |
| `SYS3` | 5-seed default64 RMS | fail, `+0.0167` | pass, memory max `1.0480` | S2 pass，但表达力/优化不足 |
| `SYS4` | 5-seed hidden60 AdamW half-state | fail, `-0.7396` | fail, S2 shapes `7/9` | half-state AdamW task collapse，且 step 不闭合 |
| `SYS5` | 5-seed hidden60 AdamW v-half | fail, `-0.7475` | fail, S2 shapes `5/9` | v-half state 仍 task collapse |
| `SYS6` | 5-seed hidden60 AdamW CPU-state | pass, `+0.0215` | fail, step max `15.0571` | macro + memory pass，但 CPU offload 太慢 |
| `SYS7` | 5-seed hidden60 AdamW v-CPU | pass, `+0.0221` | fail, step max `16.0406` | macro + memory pass，但 v offload 仍太慢 |

机制结论：

1. `SB2` 已经证明 no-external-teacher code-native self-bootstrap 可以达到 10-seed teacher-free macro pass，但 memory/efficiency 未闭合。
2. `SYS1 hidden60` 证明 system package 已非常接近：task/grad/step 都过，bs512 memory 只差约 `0.0074`。
3. `SYS2/SYS3` 证明 optimizer-state memory 是真实 actionable source：去掉/压缩 optimizer state 后可以 FullGridS2。
4. 但 optimizer-state trim 不是无代价的：`SYS2` 直接 task collapse，`SYS3` 虽保住 positive signal，却退到 macro gate 以下，`SYS4/SYS5` low-precision state 也出现严重 task collapse。
5. `SYS6/SYS7` 进一步分离了问题：FP32 AdamW state offload 可以同时保 task 与 S2 memory，但 CPU/GPU transfer 让 step ratio 爆到 `15x-16x`。
6. 当前不能宣称 v8.0 success，因为没有同一个 official candidate 同时满足 TeacherFreeMacroSignificantPass 与 FullGridS2Pass。
7. 下一步应做更细粒度的 GPU-native fused/streaming memory package 或新 code-native primitive，而不是继续粗暴删 optimizer state或 CPU offload；目标是保留 `SYS1/SB2/SYS7` 的 AdamW 可达性，同时拿回 S2 step 与 memory margin。

最终一句话：

> v8.0 仍未完成全部计划，也没有达成 minimum system closure。最新自修复把失败切得更干净：AdamW/SYS1 保 task 但 GPU memory 超一点；RMS/SYS3 过 S2 但 macro 掉线；SGD/SYS2 与低精度 AdamW/SYS4/SYS5 都 collapse；CPU offload 的 SYS6/SYS7 能保 macro 和 memory，却因 step `15x-16x` 失败。现在的真实 blocker 不是 teacher leak 或 fake/proxy，而是“如何用 GPU-native fused/streaming/cache package 在不损坏 AdamW/self-bootstrap 可达性的前提下回收 bs512 memory”。不能用外部 teacher、loss 调参、CPU offload 或四舍五入补差。

## 21. 追加自修复：SYS8 GPU-native BF16 second-moment state

`SYS6/SYS7` 证明 CPU offload 可以回收 memory 但 step 完全不可接受后，继续尝试 GPU-native optimizer-state repair：`SYS8` 保持 AdamW first moment `m` 为 FP32 GPU tensor，把 second moment `v` 改为 BF16 GPU tensor。该路线不使用 external teacher，不改 loss/sampler/classwise，不使用 fake/proxy。

### 21.1 SYS8 hidden60

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys8_adamw_v_bfloat_state_hidden60_5seed_20260507T050000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS1,SYS8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS8` | `+0.0253` | `+0.0185` | `4.70e-05` | `+0.0207` | `+0.0009` | `-0.0937` | 1 | `1.0521` | `1.6682` | `2/9` | macro pass, S2 fail |

判断：

- `SYS8 hidden60` 是本轮最强 5-seed macro signal：val gap `+0.0253`。
- 但 S2 没有闭合：memory max `1.0521 > 1.05`，step max `1.6682 > 1.50`。
- 因此不能进入 10-seed success confirmation。

No-fake/no-proxy：`rows_checked=1250`，fake/proxy nonzero `0`。

### 21.2 SYS8 hidden58

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys8_adamw_v_bfloat_state_hidden58_5seed_20260507T053000Z \
  --fresh \
  --device auto \
  --hidden-dim 58 \
  --candidates B0,SYS8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS8` | `+0.0199` | `+0.0122` | `1.09e-03` | `+0.0198` | `-0.0042` | `-0.1013` | 1 | `1.0513` | `1.5072` | `3/9` | macro near-miss, S2 near-miss |

判断：

- `hidden58` 同时接近 macro 与 S2，但两者都没有过线。
- macro gap `+0.019921875 < +0.0200`，不能四舍五入为 pass。
- S2 也未闭合：memory max 超 `0.0013`，step max 超 `0.0072`。

No-fake/no-proxy：`rows_checked=870`，fake/proxy nonzero `0`。

### 21.3 SYS8 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys8_adamw_v_bfloat_state_hidden57_5seed_20260507T060000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS8` | `+0.0202` | `+0.0113` | `2.45e-03` | `+0.0260` | `-0.0060` | `-0.0860` | 1 | `1.0508` | `1.5070` | `3/9` | macro pass, S2 near-miss |

判断：

- `hidden57` 恢复了 teacher-free macro pass：val gap `+0.02018`，test gap `+0.02604`。
- 但 S2 仍未闭合：memory max `1.0508 > 1.05`，step max `1.5070 > 1.50`。
- 因为 5-seed screen 已经缺 FullGridS2Pass，按计划不能启动或记录 10-seed minimum success confirmation。

No-fake/no-proxy：`rows_checked=869`，fake/proxy nonzero `0`。

### 21.4 SYS8 hidden56

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys8_adamw_v_bfloat_state_hidden56_5seed_20260507T063000Z \
  --fresh \
  --device auto \
  --hidden-dim 56 \
  --candidates B0,SYS8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | grad pass | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS8` | `+0.0177` | `+0.0090` | `2.80e-03` | `+0.0180` | `+0.0024` | `-0.0801` | 1 | `1.0504` | `1.8200` | `5/9` | macro fail, S2 fail |

判断：

- 继续压到 hidden56 后，macro 掉到 `+0.0177`，teacher-free macro gate 不成立。
- S2 也没有闭合，尤其 step max 升到 `1.8200`。
- hidden56 说明继续容量压缩不是 SYS8 的闭合方向。

No-fake/no-proxy：`rows_checked=870`，fake/proxy nonzero `0`。

### 21.5 SYS8 小结

| candidate | hidden dim | teacher-free macro | memory max | step max | S2 shapes | route |
|---|---:|---:|---:|---:|---:|---|
| `SYS8` | 60 | pass, `+0.0253` | `1.0521` | `1.6682` | `2/9` | S2 fail |
| `SYS8` | 58 | fail, `+0.0199` | `1.0513` | `1.5072` | `3/9` | double near-miss |
| `SYS8` | 57 | pass, `+0.0202` | `1.0508` | `1.5070` | `3/9` | S2 near-miss |
| `SYS8` | 56 | fail, `+0.0177` | `1.0504` | `1.8200` | `5/9` | macro + S2 fail |

机制判断：

1. `SYS8` 比 `SYS4/SYS5` 更有价值：BF16 second moment 没有造成半精 AdamW 那种 task collapse。
2. `SYS8 hidden57/60` 能保住 teacher-free macro pass，说明 GPU-native低精 state 是可行方向。
3. 但 `SYS8` 仍没有达成 FullGridS2；最好情况仍差 `0.0008` memory ratio 与 `0.0070` step ratio。
4. 当前不能把 `SYS8 hidden57` 记为成功；gate 是预先定义的，且 S2 失败是真实落盘结果。
5. 下一步应做更细粒度 GPU-native fused/streaming memory package，而不是继续 hidden-dim 线性压缩。

### 21.6 追加 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `b41a713330ac4e3d766167284761f20455a0915c737256b8e1f0e52aa8c1525f` |
| `experiments/run_gafu_v72_real.py` | `77748fab25fd57b875502eb509be7bf9a4eaf87722bc97c759b0bf372db8c4f4` |
| `SYS8 hidden60 route` | `f137b55eaa70c05b0cc2ddd10850b73c6afc7a64fbf2e1912a60a4e3b60fc2bb` |
| `SYS8 hidden60 provenance` | `cd2bffb469f4a49927825fc734c37ea75b8dc7b1269ba0acd77c7bea258bd41a` |
| `SYS8 hidden60 p9_task_summary.csv` | `a3f1f354085646c99c55c9c069f6896afbcce956869754b4ee0adbf37da30a3f` |
| `SYS8 hidden60 p10_efficiency_summary.csv` | `c441258d82af5e85b0ad08b36141591b95c18a1fdffd96bd0a30d7d0accee727` |
| `SYS8 hidden58 route` | `a0619bf87ef2ccc3551c1413242060c63a2e6ee9e3c3035846b4db22c27f10f0` |
| `SYS8 hidden58 provenance` | `c46ff70e1389f61d278cd244a25e82bad457776870acd4683f290eb9c85a7ae5` |
| `SYS8 hidden58 p9_task_summary.csv` | `b365d932f3b1c184eaa79c8cb4b569eeca15b6da1b0bbd166b70a70b8da03280` |
| `SYS8 hidden58 p10_efficiency_summary.csv` | `a11465b850264694be519dfe2fee4244a1fee3efbefcec0ddcfeee1b5f65fce6` |
| `SYS8 hidden57 route` | `ab1c9027721a18039c3f9cd7a880b243ccd13831fe11d43dae9a7dfade051ffb` |
| `SYS8 hidden57 provenance` | `b6d9025aa9635538a42ed5590dc7051bbffbfd49a1f8afc7182b606a5e6027a6` |
| `SYS8 hidden57 p9_task_summary.csv` | `b9a8eea4d9a91879b3e3cdb429d281d818c0a5f570232111815c6ae52a37f423` |
| `SYS8 hidden57 p10_efficiency_summary.csv` | `a577995dfb4d84d5a724937c603605e04196218c44a682ea427a3474af264bb1` |
| `SYS8 hidden56 route` | `bf38c3ea2996c00b80e177d12a6339846e4dd5ae57e60d82e6b65f434163665d` |
| `SYS8 hidden56 provenance` | `c46ff70e1389f61d278cd244a25e82bad457776870acd4683f290eb9c85a7ae5` |
| `SYS8 hidden56 p9_task_summary.csv` | `994087f616112bff62d3589e5afed68e10b2134e9cd98c6f8a7a629f13b4fce9` |
| `SYS8 hidden56 p10_efficiency_summary.csv` | `f8f445272bb3ba02bae9e1cfa232b514ce64f4dd18e4f1d2e5eeb8219863d48f` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

## 22. SYS8 后更新结论

截至 `SYS8` GPU-native BF16 second-moment state repair，v8.0 全部实验仍未完成，minimum / formal success 仍未达成。

当前最接近闭合的是 `SYS8 hidden57`：

```text
CodeNativePass = true
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = true
FullGridS2Pass = false
success_v80_minimum = false
```

它失败的位置已经非常窄：

```text
memory max = 1.0508059 > 1.05
step max   = 1.5069596 > 1.50
```

但这仍然是失败，不能写成 success。`SYS8 hidden58` 则是另一种 near-miss：memory/step 也接近 S2，但 macro 为 `+0.019921875 < +0.0200`。这两个结果共同说明：简单 hidden compression 已经到达边界，继续压 hidden 会丢 macro 或引入 step 波动。

更新后的路线判断：

1. v8.0 official teacher-free route 已经有多个真实 macro-positive candidate：`SB2`、`SYS1`、`SYS6/SYS7`、`SYS8 hidden57/60`。
2. 但仍没有同一个 candidate 同时满足 teacher-free macro pass 与 FullGridS2Pass。
3. `SYS8` 是目前最有希望的 GPU-native optimizer-state repair：它避免了 `SYS4/SYS5` 的低精 task collapse，也避免了 `SYS6/SYS7` 的 CPU offload step 爆炸。
4. 剩余 gap 不应通过四舍五入、外部 teacher、loss/classwise 调参或 proxy ratio 解决；下一步需要真实实现更细粒度的 fused/streaming cache 或 optimizer-state kernel package，把 bs512 memory 与 worst-shape step 同时压过 S2。

最终一句话：

> v8.0 仍未完成全部计划，也未达成 minimum system closure。`SYS8 hidden57` 已经把 teacher-free macro 与 S2 推到同一候选上的最窄边界，但 memory `1.0508` 和 step `1.5070` 仍真实超过 S2 gate；因此不能宣称成功。下一步必须做真实 GPU-native fused/streaming system package，而不是继续借 external teacher、loss 修补、CPU offload 或四舍五入补差。

## 23. 追加自修复：SYS9-SYS12 optimizer-state/update kernel probes

用户继续要求检查 v8.0 是否全部完成后，继续围绕 `SYS8 hidden57` 的最后 S2 gap 做系统侧自修复。本轮没有使用 external teacher、loss/classwise/sampler/focal/margin/class weight，也没有写 fake/proxy row。

新增实现：

| candidate | 修改 | 目的 |
|---|---|---|
| `SYS9` | `SYS8` 的 BF16 second moment 保持 native BF16 denominator 更久 | 检查 FP32 denom temp 是否造成 step/memory near-miss |
| `SYS10` | AdamW 一阶/二阶 moment 都用 BF16 | 检查 BF16 是否能替代 fp16，关闭 optimizer-state memory |
| `SYS11` | 保持 `SYS8` 的 FP32 first moment + BF16 second moment，但用 `addcdiv_` 避免额外 step tensor | 尽量保持 AdamW reachability，同时减少 update temp/step |
| `SYS12` | `SYS11` + 仅 head first moment 用 BF16 | 只削很小的 head optimizer-state memory，避免动 stack momentum |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 23.1 SYS9 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys9_adamw_v_bfloat_native_state_hidden57_5seed_20260507T070000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS9 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad relerr max | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS9` | `+0.01615` | `+0.00781` | `7.76e-03` | `+0.02227` | `8.91e-05` | `1.05081` | `1.49292` | `6/9` | step pass, macro fail, memory fail |

判断：

- native BF16 denominator 确实把 worst step 压到 `1.4929 <= 1.50`。
- 但 macro gap 掉到 `+0.01615`，明显低于 `+0.0200`。
- memory 仍为 `1.0508059 > 1.05`。
- 因此 `SYS9` 不能作为 route。

No-fake/no-proxy：`rows_checked=873`，fake/proxy nonzero `0`。

### 23.2 SYS10 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys10_adamw_bfloat_state_hidden57_5seed_20260507T073000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS10 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad relerr max | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS10` | `+0.01602` | `+0.00755` | `9.14e-03` | `+0.01914` | `8.91e-05` | `1.04576` | `1.67811` | `0/9` | memory pass, macro fail, step fail |

判断：

- BF16 m/v optimizer state 把 memory max 压到 `1.0458`，说明 optimizer-state memory 是可削项。
- 但一阶动量也降到 BF16 后，task 与 step 都明显变差。
- 这与 `SYS4/SYS5` 的半精 state collapse 方向一致，只是 BF16 没有完全崩溃；仍不可用。

No-fake/no-proxy：`rows_checked=876`，fake/proxy nonzero `0`。

### 23.3 SYS11 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys11_adamw_v_bfloat_addcdiv_update_hidden57_5seed_20260507T080000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS11 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad relerr max | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS11` | `+0.01875` | `+0.00885` | `3.81e-03` | `+0.02617` | `8.91e-05` | `1.05081` | `1.47665` | `6/9` | step pass, macro fail, memory fail |

判断：

- `addcdiv_` update 可以把 worst step 压到 `1.4766`，说明 update temp/implementation 对 step 有贡献。
- 但 memory peak 没变，仍为 `1.0508059`。
- macro gap 掉到 `+0.01875`，没有闭合 teacher-free macro gate。

No-fake/no-proxy：`rows_checked=879`，fake/proxy nonzero `0`。

### 23.4 SYS12 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys12_adamw_head_bfloat_v_bfloat_addcdiv_update_hidden57_5seed_20260507T083000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS12 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad relerr max | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS12` | `+0.01810` | `+0.00781` | `1.18e-02` | `+0.02331` | `8.91e-05` | `1.05073` | `1.51458` | `4/9` | macro fail, memory/step fail |

判断：

- 只削 head first moment 的 memory 改善极小：`1.0508059 -> 1.0507316`。
- macro gap 仍降到 `+0.01810`。
- 说明剩余 bs512 memory peak 不主要由 head first moment 决定；继续在这一小项上做低精状态没有意义。

No-fake/no-proxy：`rows_checked=882`，fake/proxy nonzero `0`。

### 23.5 SYS8 hidden57 basis7 probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys8_adamw_v_bfloat_state_hidden57_basis7_5seed_20260507T090000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --basis-count 7 \
  --candidates B0,SYS8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | val gap | CI95 low | Holm p | test gap | grad relerr max | memory max | step max | S2 shapes | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS8` basis7 | `+0.02018` | `+0.01133` | `2.45e-03` | `+0.02604` | `8.91e-05` | `1.05081` | `1.51728` | `3/9` | macro pass, S2 fail |

判断：

- basis7 没有改变关键 memory peak：仍为 `1.0508059`。
- macro 仍过，但 step max 变为 `1.5173`，比原 hidden57 更差。
- 因此 head basis 不是当前 S2 gap 的有效开关。

No-fake/no-proxy：`rows_checked=881`，fake/proxy nonzero `0`。

### 23.6 本轮小结

| route | macro gate | memory max | step max | S2 | 判断 |
|---|---:|---:|---:|---:|---|
| `SYS8 hidden57` | pass `+0.02018` | `1.05081` | `1.50696` | 0 | 当前最接近，但 S2 fail |
| `SYS9 hidden57` | fail `+0.01615` | `1.05081` | `1.49292` | 0 | step 修住，task 掉 |
| `SYS10 hidden57` | fail `+0.01602` | `1.04576` | `1.67811` | 0 | memory 修住，task/step 掉 |
| `SYS11 hidden57` | fail `+0.01875` | `1.05081` | `1.47665` | 0 | step 修住，task/memory fail |
| `SYS12 hidden57` | fail `+0.01810` | `1.05073` | `1.51458` | 0 | 小幅 memory trim，不够且 task fail |
| `SYS8 hidden57 basis7` | pass `+0.02018` | `1.05081` | `1.51728` | 0 | basis 不是 S2 开关 |

机制判断：

1. `SYS9/SYS11` 说明 update implementation 可以影响 worst step，但不能解决 bs512 memory peak。
2. `SYS10` 说明 optimizer-state memory 可以被 BF16 m/v 压到 S2 内，但一阶动量降精度会伤 task/step，不可作为 official route。
3. `SYS12` 说明只削 head first moment 太小，不足以关闭 memory peak。
4. `basis7` 说明当前 memory peak 不由 head basis count 控制。
5. 本轮没有找到比 `SYS8 hidden57` 更好的 simultaneous macro+S2 candidate。

## 24. SYS9-SYS12 后最终更新结论

截至本轮追加，`docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md` 的全部实验仍未完成，v8.0 minimum / formal success 仍未达成。

当前真实状态：

```text
CodeNativePass = true for best official candidates
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = true for SB2/SYS1/SYS8 hidden57 and some variants
FullGridS2Pass = false for all macro-pass official candidates
success_v80_minimum = false
success_v80_formal = false
```

最接近的 candidate 仍是 `SYS8 hidden57`：

```text
macro gap = +0.020182291666666668
CI95 low = +0.011328125
Holm p = 0.0024468068291418643
test gap = +0.026041666666666668
grad relerr max = 8.91e-05
memory max = 1.0508059125009284
step max = 1.5069595938463196
S2 shapes = 3/9
```

它不能被记为成功，因为：

```text
memory max > 1.05
step max > 1.50
FullGridS2Pass = false
```

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `dbb38c307b53c8749c79a12797e56d0d675d9e4d01aaef8995273eee86c6fd77` |
| `experiments/run_gafu_v72_real.py` | `5d682a1b74833c547cb6fb3502807526b52cada5cdd196f5de457be2673b5b5c` |
| `v80_sys9 route` | `deb9aa82f0b016152ded4f426a98decfb64a7d4e6b201d61dec8647a869725ce` |
| `v80_sys9 provenance` | `6283bb8b6238d8286abd9b1a3260f1a9c52e13b2c128839b3960bbbbc626c35a` |
| `v80_sys10 route` | `2aacd6054a2c2122457d15907185f6ed56954604e3fcd63a114c5fd2dc8c7563` |
| `v80_sys10 provenance` | `7735a0095b94841c7d67089105d8ec16aea911ef53228076c7ba9e9644322dde` |
| `v80_sys11 route` | `fbafed698e0fad0d5be646f608d9cb9e959ad68472feb75c899870228f57085b` |
| `v80_sys11 provenance` | `38a78bc2800a2202f5ae6fb0b72afdc5208a82608d652da3369420dadd023d0f` |
| `v80_sys12 route` | `7b33f76095c85afc61c6dcb6de02fc36b55d1b5ea9795b2c25e4ffc71823c9dc` |
| `v80_sys12 provenance` | `5ec9a425f8ee2c6b1ad1be05ed3054d87ed06db813283752999466a713b58313` |
| `v80_sys8_hidden57_basis7 route` | `b50bfebf926114e1d000f615f90f5126a7c71ba522cebac2405cab58b977e8ff` |
| `v80_sys8_hidden57_basis7 provenance` | `7ba297b5e24d4d1eefb3cd20ba6c9a4199371e679e6ae8fabe044f008f5b8a85` |

最终判断：

> v8.0 仍未达成 system closure。teacher-free macro advantage 已经真实存在，但所有 macro-pass official candidates 仍缺 FullGridS2；`SYS8 hidden57` 的失败 margin 极窄，却仍是真实失败。SYS9-SYS12 说明单纯继续削 optimizer/update temp 会在 task、step、memory 三者之间互相转移问题，不能闭合。下一步必须实现真正的 fused/streaming system package 或新的 code-native primitive，而不能用 external teacher、loss 调参、四舍五入或 proxy row 补差。

## 25. 继续追加自修复：SYS13/SYS14 code-native system probes

用户再次要求检查 v8.0 是否全部完成后，继续围绕 `SYS8 hidden57` 的最后 S2 gap 做系统侧自修复。本轮仍没有使用 external teacher、loss/classwise/sampler/focal/margin/class weight，也没有写 fake/proxy row。

新增实现：

| candidate | 修改 | 目的 |
|---|---|---|
| `SYS13` | packed recompute stack，所有 stack mix weights 由单个 flat tensor owner 持有，仍使用 recompute hidden-y cache policy 与 BF16 second-moment state | 检查减少参数 owner / optimizer loop 是否能在保表达力下闭合 S2 |
| `SYS14` | 保持 `SYS8` 的 recompute stack 参数布局，但 stack AdamW first/second moment 使用 BF16，head first moment 保持 FP32，并用 `addcdiv_` update | 检查只削 stack optimizer state 是否能保 head dynamics 并闭合 memory/step |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 25.1 SYS13 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys13_packed_recompute_adamw_v_bfloat_state_hidden57_5seed_20260507T093000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS13` | `+0.0177083` | `+0.0082031` | `9.43e-03` | `+0.0212240` | `-0.005610` | `-0.087449` | 1 | `1.04499` | `1.46568` | `9/9` | S2 pass, macro fail |

判断：

- `SYS13` 真实闭合了 FullGridS2：memory max `1.04499`，step max `1.46568`。
- 但 packed parameter owner 后 teacher-free macro 掉到 `+0.01771`，没有达到 `+0.0200`。
- 因此 `SYS13` 不能作为 v8.0 minimum success；它说明 packed/recompute 对 system 有效，但当前参数布局会损伤 autonomous macro signal。

No-fake/no-proxy：`rows_checked=884`，fake/proxy nonzero `0`。

### 25.2 SYS14 hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys14_adamw_stack_bfloat_v_bfloat_addcdiv_update_hidden57_5seed_20260507T103000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS14 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS14` | `+0.0158854` | `+0.0079427` | `9.43e-03` | `+0.0213542` | `-0.007879` | `-0.081714` | 1 | `1.04583` | `1.60394` | `5/9` | macro fail, S2 fail |

判断：

- stack-only BF16 moment state 确实降低了 memory max 到 `1.04583`，但 macro 明显掉到 `+0.01589`。
- step max 反而升到 `1.60394`，FullGridS2 也没有闭合。
- 该结果与 `SYS10/SYS12` 一致：削 AdamW first-moment 精度会转化为 task 或 step 损失，不是可用闭合路线。

No-fake/no-proxy：`rows_checked=888`，fake/proxy nonzero `0`。

### 25.3 SYS13/SYS14 小结

| route | macro gate | memory max | step max | S2 | 判断 |
|---|---:|---:|---:|---:|---|
| `SYS8 hidden57` | pass `+0.02018` | `1.05081` | `1.50696` | 0 | 仍是 best macro+near-S2 |
| `SYS13 hidden57` | fail `+0.01771` | `1.04499` | `1.46568` | 1 | S2 闭合但表达力掉 |
| `SYS14 hidden57` | fail `+0.01589` | `1.04583` | `1.60394` | 0 | memory 修住但 task/step 掉 |

机制判断：

1. `SYS13` 第一次把 v8.0 system repair 拉进 FullGridS2，但没有保住 teacher-free macro gate。
2. `SYS14` 说明只削 stack first moment 仍会明显破坏 autonomous macro，并且 update path step 不稳定。
3. `SYS8 hidden57` 仍是当前最接近 simultaneous macro+S2 的 official candidate，但它真实 S2 fail。
4. 当前失败不再像是简单 optimizer-state memory 问题；更像是需要新的 code-native primitive / fused package 同时保参数动态与 runtime。

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `e51079e98047249e6cebb108d20c9c9559a57c6d743dc382f2d988596835074f` |
| `experiments/run_gafu_v72_real.py` | `87a848490773c79c348b25bdd7532879f747686aee7bd77bfb901d3d416f40e0` |
| `v80_sys13 route` | `f7599d789ada7c89a658280f91099fdce0dcaeb1aa2041f7aaa69438514fcdde` |
| `v80_sys13 provenance` | `17c3968ae46d21a504e1e9e04303f579b398451026d0008e27c230adac78bc59` |
| `v80_sys14 route` | `4eda90d12446af9d7719eee4a8e58f765250005da39a15aaac2eb9b4cc18c432` |
| `v80_sys14 provenance` | `f92fc30bbfcee896c828797f8400a89e2240a838b0175a3ab5a0826e5d97d379` |

## 26. SYS13/SYS14 后最终更新结论

截至本轮追加，`docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md` 的全部实验仍未完成，v8.0 minimum / formal success 仍未达成。

当前真实状态：

```text
CodeNativePass = true for measured official repair candidates
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
TeacherFreeMacroSignificantPass = true for SYS8 hidden57
FullGridS2Pass = true for SYS13 but macro fail
FullGridS2Pass = false for all macro-pass official candidates
success_v80_minimum = false
success_v80_formal = false
```

不能 claim success 的原因：

```text
SYS8 hidden57:
  macro pass, grad pass, strict pass
  memory max = 1.0508059125009284 > 1.05
  step max = 1.5069595938463196 > 1.50

SYS13 hidden57:
  FullGridS2Pass = true
  macro gap = +0.017708333333333333 < +0.0200

SYS14 hidden57:
  macro gap = +0.015885416666666666 < +0.0200
  step max = 1.6039418892575499 > 1.50
```

最终判断：

> v8.0 仍未达成 system closure。`SYS13` 证明 code-native system path 可以过 FullGridS2，但当前 packed/recompute 参数布局没有保住 teacher-free macro advantage；`SYS8 hidden57` 保住 macro 却差极窄 S2 margin。现在的 blocker 已经从“有没有 teacher-free advantage”变成“如何在不破坏 autonomous representation/optimizer dynamics 的情况下做真正 fused/streaming system package”。继续用 external teacher、loss 调参、四舍五入或 proxy row 都不允许，也不会解决这个问题。

## 27. 继续追加自修复：SYS15/SYS16/SYS17 selective first-moment probes

`SYS13` 证明 S2 可以闭合但 macro 会掉，`SYS8 hidden57` 证明 macro 可以闭合但 S2 差极窄。继续做更细粒度 optimizer-state repair：只对 stack 的一部分 first moment 使用 BF16，避免 `SYS10/SYS14` 那种全 stack 或全模型 first moment 降精导致 task collapse。

新增实现：

| candidate | 修改 | 目的 |
|---|---|---|
| `SYS15` | 仅 stack tail hidden-hidden 层 first moment 用 BF16；second moment 仍 BF16 | 小幅削 optimizer-state memory，尽量不动第一层表示 |
| `SYS16` | 仅 stack 第一层 first moment 用 BF16；second moment 仍 BF16 | 更大幅削 memory，观察第一层动量降精是否还能保 macro |
| `SYS17` | `SYS16` + `addcdiv_` in-place update | 针对 `SYS16` 的 step-only S2 miss 做 update temp/step repair |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 27.1 SYS15/SYS16 hidden57 5-seed

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys15_sys16_selective_m_bfloat_hidden57_5seed_20260507T110000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS15,SYS16 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS15` | `+0.0179688` | `+0.0119792` | `4.61e-04` | `+0.0268229` | 1 | `1.05016` | `1.93848` | `0/9` | macro fail, step fail |
| `SYS16` | `+0.0200521` | `+0.0115885` | `1.75e-03` | `+0.0266927` | 1 | `1.04647` | `1.51945` | `0/9` | macro pass, S2 step fail |

判断：

- `SYS15` 没有保住 macro，也没有改善 step。
- `SYS16` 是有信息量的 near-pass：macro 过线，memory 过 S2，但 step max `1.51945` 仍超过 S2。
- 因此继续做 `SYS17`，只针对 update step 做 in-place `addcdiv_` repair。

No-fake/no-proxy：`rows_checked=1274`，fake/proxy nonzero `0`。

### 27.2 SYS17 hidden57 5-seed

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys17_l0_bfloat_addcdiv_hidden57_5seed_20260507T113000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS17 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

5-seed 结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | v8.0 minimum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS17` | `+0.0200521` | `+0.0118490` | `1.26e-03` | `+0.0246094` | `-0.006499` | `-0.088426` | 1 | `1.04647` | `1.48120` | `9/9` | 1 |

判断：

- `SYS17 hidden57` 在 5-seed probe 下首次同时闭合 TeacherFreeMacroSignificantPass 与 FullGridS2。
- 因为 v8.0 目标要求稳定、10-seed、可复现，5-seed 不能作为最终成功；因此立刻执行 10-seed confirmation。

No-fake/no-proxy：`rows_checked=895`，fake/proxy nonzero `0`。

### 27.3 SYS17 hidden57 10-seed confirmation

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys17_l0_bfloat_addcdiv_hidden57_confirm_10seed_20260507T120000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS17 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

10-seed 结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | v8.0 minimum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS17` | `+0.0186198` | `+0.0132161` | `6.25e-07` | `+0.0198568` | `-0.006215` | `-0.090563` | 1 | `1.04647` | `1.49728` | `9/9` | 0 |

判断：

- 10-seed 下 `SYS17 hidden57` 的 FullGridS2 稳定成立：memory max `1.04647`，step max `1.49728`。
- 但 macro gap 回落到 `+0.01862 < +0.0200`，TeacherFreeMacroSignificantPass 不成立。
- 因此 5-seed minimum success 不能升级为 v8.0 官方成功。

No-fake/no-proxy：`rows_checked=1634`，fake/proxy nonzero `0`。

### 27.4 SYS17 hidden58 5-seed

由于 hidden57 的 10-seed blocker 变成 macro 不足，继续测试 hidden58 capacity 是否能补 macro，同时保住 `SYS17` 的 memory/step repair。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys17_l0_bfloat_addcdiv_hidden58_5seed_20260507T123000Z \
  --fresh \
  --device auto \
  --hidden-dim 58 \
  --candidates B0,SYS17 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS17 hidden58` | `+0.0214844` | `+0.0143229` | `1.73e-04` | `+0.0200521` | `+0.000322` | `-0.100026` | 1 | `1.04686` | `1.60504` | `8/9` | macro pass, S2 step fail |

判断：

- hidden58 恢复了 macro margin，但 S2 又因 step max `1.60504` 失败。
- 当前表现为清晰 tradeoff：hidden57 可以 S2 但 10-seed macro 不够；hidden58 macro 够但 S2 step 不稳。

No-fake/no-proxy：`rows_checked=896`，fake/proxy nonzero `0`。

### 27.5 SYS15-SYS17 小结

| route | seeds | macro gate | memory max | step max | S2 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `SYS16 hidden57` | 5 | pass `+0.02005` | `1.04647` | `1.51945` | 0 | step-only near-miss |
| `SYS17 hidden57` | 5 | pass `+0.02005` | `1.04647` | `1.48120` | 1 | 5-seed minimum pass |
| `SYS17 hidden57` | 10 | fail `+0.01862` | `1.04647` | `1.49728` | 1 | S2 stable, macro fails |
| `SYS17 hidden58` | 5 | pass `+0.02148` | `1.04686` | `1.60504` | 0 | macro stable, step fails |

机制判断：

1. `SYS17` 是目前最强 system repair：selective L0 BF16 first moment + BF16 second moment + in-place addcdiv 能稳定闭合 S2。
2. 但 hidden57 的 10-seed macro 不够，说明 5-seed success 不稳。
3. hidden58 说明容量可以补回 macro，但 step 又超过 S2，说明 runtime closure 仍未完成。
4. 当前最佳方向从“削 memory”转为“保 hidden58 macro，同时降低 step”，需要真正 fused update/phase-mapped profiler 或更低-overhead primitive；继续局部削 optimizer-state 很可能继续在 macro/step 之间摆动。

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `e5f4c389feebbcc71f3682947841436b247141f657e2d9bc6c11af7b6e4f213f` |
| `experiments/run_gafu_v72_real.py` | `daa53327355954fc8c84a3361a18b45601485905985d16449c0d5bbfc2dfc867` |
| `v80_sys15_sys16 route` | `f7dc5917182712378f8c4b2c4a7fee1ddaa99140064723b6cd291633b8f3bbc8` |
| `v80_sys15_sys16 provenance` | `b282e361bfdb00c88d06466a08501373aebd23526b0d1f2974e2e78dc9ffd245` |
| `v80_sys17 hidden57 5seed route` | `82052a0e4d8d468067c892b6028934d27fa2e26510c6fdfc896a4e134927b440` |
| `v80_sys17 hidden57 5seed provenance` | `d9e59865b37eb4aa5ecb56fdf49a4580c6be526f4afb85c69b07d2791f37944b` |
| `v80_sys17 hidden57 10seed route` | `eaeb357aa9e6ba4a3243019dc3606ca3758b2fb7b3bb075470225f45b0aacc4c` |
| `v80_sys17 hidden57 10seed provenance` | `bcebd0064cfcadb29bab6e91526511c5d6390039ce786ec287d5f31ea9d81619` |
| `v80_sys17 hidden58 5seed route` | `d2fbe2c60aaf1a51a5d48aa7c49e6cf3975da934101738f7a7a9eb31dcf7767c` |
| `v80_sys17 hidden58 5seed provenance` | `39bc2a7e4348e48b86515a235e90a878ed94fa08eb36c5a69a946597a455b950` |

## 28. SYS17 后最终更新结论

截至本轮追加，`docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md` 的全部实验仍未完成，v8.0 minimum / formal success 仍未达成。

当前真实状态：

```text
CodeNativePass = true
TeacherContractV2Pass = true
StrictPass = true
GradPass = true
FullGridS2Pass = true for SYS13 and SYS17 hidden57
TeacherFreeMacroSignificantPass = false for SYS13 and SYS17 hidden57 10-seed
TeacherFreeMacroSignificantPass = true for SYS17 hidden58 5-seed
FullGridS2Pass = false for SYS17 hidden58
success_v80_minimum = false
success_v80_formal = false
```

最重要的新事实：

```text
SYS17 hidden57 5-seed:
  minimum pass, but not sufficient for v8.0 official claim

SYS17 hidden57 10-seed:
  S2 stable
  macro gap = +0.018619791666666666 < +0.0200

SYS17 hidden58 5-seed:
  macro gap = +0.021484375
  step max = 1.6050374689021978 > 1.50
```

最终判断：

> v8.0 仍未达成全部计划，也不能声明 official teacher-free system closure。`SYS17` 让问题推进了一步：S2 可以稳定闭合，且 5-seed 可以同时过 macro，但 10-seed macro 回落；hidden58 可以补 macro，却再次 step fail。当前 blocker 已经非常具体：需要在 hidden58 级别保住 autonomous macro，同时把 step path 降到 S2，这需要真实 fused/streaming update 或 phase-mapped profiler 指导的新 system package，而不是 external teacher、loss 调参、四舍五入或 proxy row。

## 29. 继续追加自修复：hidden58 confirmation / packed layout / basis16 / alpha sweep

用户再次要求确认 v8.0 全部实验是否完成后，继续沿上一节明确的 blocker 执行：不引入 external teacher，不使用 fake/proxy，不做 classwise/loss/sampler 修补，只测 code-native/system 与 no-external self-bootstrap 方向。

新增代码：

| candidate | 目的 | teacher policy |
|---|---|---|
| `SYS18` | `SYS17` system path + self-bootstrap alpha `0.10` | no external teacher |
| `SYS19` | `SYS17` system path + self-bootstrap alpha `0.50` | no external teacher |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 29.1 SYS17 hidden58 10-seed confirmation

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys17_l0_bfloat_addcdiv_hidden58_confirm_10seed_20260507T130000Z \
  --fresh \
  --device auto \
  --hidden-dim 58 \
  --candidates B0,SYS17 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS17 hidden58` | 10 | `+0.0188151` | `+0.0136719` | `4.71e-07` | `+0.0189453` | 1 | `1.04686` | `1.59605` | 0 | 0 |

判断：

- hidden58 的 5-seed macro pass 没有复现到 10-seed。
- 10-seed 下 macro 回落到 `+0.0188151 < +0.0200`。
- 同时 S2 仍失败，step max `1.59605 > 1.50`。

No-fake/no-proxy：`rows_checked=1635`，fake/proxy nonzero `0`。

### 29.2 SYS13 packed recompute hidden58

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys13_packed_recompute_hidden58_5seed_20260507T133000Z \
  --fresh \
  --device auto \
  --hidden-dim 58 \
  --candidates B0,SYS13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS13 hidden58` | 5 | `+0.0180990` | `+0.0110645` | `7.86e-04` | `+0.0194010` | 1 | `1.04532` | `1.47127` | 1 | 0 |

判断：

- packed recompute layout 能闭合 S2。
- 但 macro gap 下降到 `+0.01810`，不能作为 v8.0 minimum success。
- 这说明单纯减少 tensor owner / optimizer-loop overhead 不能保住当前最强 macro signal。

No-fake/no-proxy：`rows_checked=896`，fake/proxy nonzero `0`。

### 29.3 SYS17 hidden57 basis16

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys17_l0_bfloat_addcdiv_hidden57_basis16_5seed_20260507T140000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --basis-count 16 \
  --candidates B0,SYS17 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS17 hidden57 basis16` | 5 | `+0.0200521` | `+0.0118490` | `1.26e-03` | `+0.0246094` | 1 | `1.04647` | `1.78534` | 0 | 0 |

判断：

- basis16 可以保住 5-seed macro pass。
- 但 step max 明显恶化到 `1.78534`，S2 失败。
- 这不是可用的 system closure。

No-fake/no-proxy：`rows_checked=896`，fake/proxy nonzero `0`。

### 29.4 SYS18/SYS19 alpha sweep hidden57

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys18_sys19_alpha_sweep_hidden57_5seed_20260507T144500Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS18,SYS19 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | alpha | seeds | macro gap | test gap | GradPass | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS18 hidden57` | `0.10` | 5 | `+0.0199219` | `+0.0261719` | 1 | `1.04647` | `1.52831` | 0 | 0 |
| `SYS19 hidden57` | `0.50` | 5 | `+0.0187500` | not route best | 1 | `1.04647` | not route best | 0 | 0 |

判断：

- alpha `0.10` 是更好的 self-bootstrap strength，但仍是 near-miss：`+0.019921875 < +0.0200`。
- step max `1.52831 > 1.50`，S2 也未闭合。
- alpha `0.50` 进一步伤害 macro，不作为后续方向。

No-fake/no-proxy：`rows_checked=1284`，fake/proxy nonzero `0`。

### 29.5 SYS18 hidden58

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys18_alpha010_hidden58_5seed_20260507T150000Z \
  --fresh \
  --device auto \
  --hidden-dim 58 \
  --candidates B0,SYS18 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS18 hidden58` | 5 | `+0.0192708` | `+0.0140625` | `2.94e-05` | `+0.0240885` | 1 | `1.04686` | `1.50194` | 0 | 0 |

判断：

- hidden58 + alpha `0.10` 没有提高 macro，反而低于 `SYS17 hidden58` 5-seed。
- S2 只差 `0.00194` step ratio，但仍是 fail，不能记为 pass。

No-fake/no-proxy：`rows_checked=903`，fake/proxy nonzero `0`。

## 30. 追加后最终更新结论

截至本次继续执行，`docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md` 的全部实验仍未完成，v8.0 minimum / formal success 仍未达成。

当前最强 measured candidates：

| candidate | scale | macro gate | S2 gate | 结论 |
|---|---:|---:|---:|---|
| `SYS17 hidden57` | 10-seed | fail `+0.01862` | pass | S2 stable but macro fail |
| `SYS17 hidden58` | 10-seed | fail `+0.01882` | fail | macro/S2 both fail in confirmation |
| `SYS13 hidden58` | 5-seed | fail `+0.01810` | pass | packed layout lowers runtime but hurts macro |
| `SYS17 hidden57 basis16` | 5-seed | pass `+0.02005` | fail | representation helps macro but hurts step |
| `SYS18 hidden57` | 5-seed | fail `+0.01992` | fail | alpha `0.10` near-miss, still below hard gate |
| `SYS18 hidden58` | 5-seed | fail `+0.01927` | fail | alpha `0.10` does not scale with hidden58 |

新增机制结论：

1. `SYS17 hidden58` 的 10-seed confirmation 否定了 5-seed macro pass 的稳定性。
2. packed recompute (`SYS13`) 能真实闭合 S2，但有效 task signal 明显不足。
3. basis16 与 hidden58 都是“补 macro 会伤 step”的方向。
4. self-bootstrap alpha sweep 没有解决：alpha `0.10` 最接近，但仍未过 macro gate，alpha `0.50` 更差。
5. 当前不能再把问题归因成单一 memory margin；它是 teacher-free representation / self-bootstrap signal 与 runtime S2 同时闭合的问题。

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `e5c5072971cf5893f9358d40274d328520c0d480849b56f9a79fddcc11d406da` |
| `experiments/run_gafu_v72_real.py` | `daa53327355954fc8c84a3361a18b45601485905985d16449c0d5bbfc2dfc867` |
| `SYS17 hidden58 10seed route` | `5e91c2844b6eb3cbd4914c32d4b5e15c687baa87f01b29a06f0a7f2305c570bd` |
| `SYS17 hidden58 10seed provenance` | `acc08c40d3025776d5a688c9d3752a9febd2094ecba328c142f406107df88bb0` |
| `SYS13 hidden58 route` | `74afd7f5861b5e48b3578ec7d4856d1eda3955044e15e63fd834999c6d7e8aec` |
| `SYS13 hidden58 provenance` | `39bc2a7e4348e48b86515a235e90a878ed94fa08eb36c5a69a946597a455b950` |
| `SYS17 hidden57 basis16 route` | `4aa6ec1310a88f17de61cbd01f757a1ffba8e030767464a27d1c294cd5b6cd6c` |
| `SYS17 hidden57 basis16 provenance` | `39bc2a7e4348e48b86515a235e90a878ed94fa08eb36c5a69a946597a455b950` |
| `SYS18/SYS19 hidden57 route` | `0b4e430e841773dfb7f6d4e090809961fb222bc2b961c582bd2f698d2b630e80` |
| `SYS18/SYS19 hidden57 provenance` | `551af28b656d523a9013c7035e62fcbf7b5ad60726aecd8a65f7ee6758aa8cd7` |
| `SYS18 hidden58 route` | `c8385d5fdfd710054a1ca9b332f4bb0b6020dbb1cdac70dcedbb272e8374ace1` |
| `SYS18 hidden58 provenance` | `60546e6019b70b32bcc605cf9d97e9b663db2ab365f9f90fd684bfe85abe7a16` |

最终判断：

> v8.0 仍未完成全部计划，也仍未达成 minimum success。所有新增尝试都是真实 measured failure：没有 fake/proxy，没有 external teacher 混入 official route，也没有把 near-miss 四舍五入成 pass。当前路线已经逼近到很窄的 tradeoff：S2 可通过 packed/system trim 闭合，但 macro 不够；macro 可用 basis/容量局部补上，但 step 失败。下一步需要新的 code-native primitive 或真正 phase-mapped fused/streaming package，而不是继续在 hidden/basis/alpha 这几个局部旋钮上打转。

## 31. 继续追加自修复：SYS13 packed capacity sweep

用户再次要求确认全部实验是否完成后，继续执行 `SYS13` packed recompute route 的容量 sweep。选择这条路线的原因是上一轮 `SYS13 hidden58` 已经真实 FullGridS2Pass，但 macro 不足；因此本轮只增加 packed path 容量，检查是否能在不破 S2 的情况下补足 teacher-free macro gap。仍然不使用 external teacher、fake/proxy、loss/classwise/sampler 调参。

### 31.1 SYS13 hidden60

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys13_packed_recompute_hidden60_5seed_20260507T153000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | seeds | macro gap | CI95 low | test gap | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS13 hidden60` | 5 | `+0.0190104` | `+0.0109375` | `+0.0242188` | `1.04599` | `1.44327` | 1 | 0 |

判断：S2 保住了，但 macro 仍低于 `+0.0200`。

No-fake/no-proxy：`rows_checked=902`，fake/proxy nonzero `0`。

### 31.2 SYS13 hidden62

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys13_packed_recompute_hidden62_5seed_20260507T163000Z \
  --fresh \
  --device auto \
  --hidden-dim 62 \
  --candidates B0,SYS13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | seeds | macro gap | CI95 low | test gap | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS13 hidden62` | 5 | `+0.0160156` | `+0.0101563` | `+0.0291667` | `1.04665` | `1.43264` | 1 | 0 |

判断：S2 仍过，但 macro 明显下降，不是有效容量修复。

No-fake/no-proxy：`rows_checked=902`，fake/proxy nonzero `0`。

### 31.3 SYS13 hidden64

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys13_packed_recompute_hidden64_5seed_20260507T160000Z \
  --fresh \
  --device auto \
  --hidden-dim 64 \
  --candidates B0,SYS13 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | seeds | macro gap | CI95 low | test gap | memory max | step max | S2 | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS13 hidden64` | 5 | `+0.0197917` | `+0.0132813` | `+0.0231771` | `1.04739` | `1.75753` | 0 | 0 |

判断：hidden64 接近 macro gate，但仍未过 `+0.0200`，且 step 破 S2。

No-fake/no-proxy：`rows_checked=903`，fake/proxy nonzero `0`。

## 32. SYS13 capacity sweep 后最终更新结论

`SYS13` packed route 的容量 sweep 没有闭合 v8.0。

| candidate | macro | S2 | 判断 |
|---|---:|---:|---|
| `SYS13 hidden58` | `+0.01810` | pass | task 不够 |
| `SYS13 hidden60` | `+0.01901` | pass | task 仍不够 |
| `SYS13 hidden62` | `+0.01602` | pass | task 更差 |
| `SYS13 hidden64` | `+0.01979` | fail | macro 仍差一点，step 破 S2 |

新增机制判断：

1. packed recompute / flat tensor owner 是真实有效的 S2 方向，但不能自然恢复 teacher-free macro advantage。
2. 增加 hidden capacity 并不单调提升 macro，hidden62 甚至明显退化。
3. hidden64 接近 macro gate，但 step fail，重现了“补表达力会伤 runtime”的主矛盾。
4. 继续在 `hidden_dim` 上扫很可能只是重复 near-miss；下一步应转向新的 representation primitive 或 phase-mapped fused/streaming package。

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `e5c5072971cf5893f9358d40274d328520c0d480849b56f9a79fddcc11d406da` |
| `SYS13 hidden60 route` | `766090b277083f4628af2a09da23f976e9ace452ec9ee4e6e5f7dc7e91bb6d70` |
| `SYS13 hidden60 provenance` | `cb7d0d573f1a98232ed12a2af825fafdbe681c4e5eb9be7bea06e4fa60eb2c15` |
| `SYS13 hidden62 route` | `b237cda71055e5270386ebc7237c7009326b26da0923e8b1a10d0d0e6b6de25e` |
| `SYS13 hidden62 provenance` | `cb7d0d573f1a98232ed12a2af825fafdbe681c4e5eb9be7bea06e4fa60eb2c15` |
| `SYS13 hidden64 route` | `c7ed54e66aad03a68eedef6ae67732e3a3f24cadb1d99d264cd9b40b418253ff` |
| `SYS13 hidden64 provenance` | `60546e6019b70b32bcc605cf9d97e9b663db2ab365f9f90fd684bfe85abe7a16` |

最终判断：

> v8.0 仍未完成全部计划，也仍未达成 minimum success。`SYS13` 证明 packed code-native route 可以稳定接近或通过 S2，但无法提供足够 autonomous macro signal；`SYS17/SYS18/basis/hidden` 证明 macro 可以被局部推近或短暂推过，但 S2 或 10-seed 稳定性会失效。当前不应继续把 near-pass 当作成功，也不应再靠 external teacher 或 loss/classwise 调参补差。下一步需要实现真正新的 teacher-free representation primitive 或 phase-mapped fused/streaming package，并重新做 10-seed official co-selection。

## 33. 追加执行：SYS20 packed alpha010 组合修复

用户再次要求确认全部实验是否完成后，继续尝试解决 v8.0 的未闭合点。本轮不再继续单纯 hidden/basis 粗扫，而是把两个已有真实信号组合：

```text
SYS13:
  packed recompute / flat tensor owner，S2 方向较强，但 macro 不够。

SYS18:
  alpha010 self-bootstrap，macro 方向接近，但不是 packed S2 path。
```

因此新增真实候选：

| candidate | 结构 | objective | external teacher |
|---|---|---|---:|
| `SYS20` | `SYS13` packed recompute + AdamW v BF16 state | best-val self-bootstrap alpha `0.10` | 0 |
| `SYS21` | `SYS13` packed recompute + AdamW v BF16 state | best-val self-bootstrap alpha `0.50` | 0 |

代码修复：

- 在 `experiments/run_gafu_v80_real.py` 中加入 `SYS20/SYS21` 到 `CandidateSpecV2`、teacher contract、candidate registry、route registry、gradient checker 与 training path。
- 对 `PackedRecomputeFusedLinearSiluStack` 的 SiLU derivative 写法改为与既有 `RecomputeYDenseKindStack` 一致的等价形式：

```text
sig * (1 + y * (1 - sig))
```

该修改只用于降低数值相对误差边界，不改变前向函数。

### 33.1 SYS20/SYS21 smoke

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_sys20_sys21_smoke \
  --fresh \
  --device auto \
  --candidates B0,SYS20,SYS21 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 1 \
  --bench-reps 2 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 结果：

| candidate | grad rows | max relerr | 判断 |
|---|---:|---:|---|
| `SYS20` | `3/3` pass | `8.65e-05` | 可进入 full screen |
| `SYS21` | `2/3` pass | `1.13e-04` | GradFail，停止 |

No-fake/no-proxy：`rows_checked=198`，fake/proxy nonzero `0`。

### 33.2 SYS20 hidden60 full-grid screen

首次 hidden60 screen 显示 macro 与 S2 同时接近，但 GradPass 失败；导数写法修复后重新运行正式记录：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys20_packed_alpha010_hidden60_gradfix_5seed_20260507T174500Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS20 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | GradPass | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS20 hidden60` | 5 | `+0.0203125` | `+0.0130208` | `4.46e-04` | `+0.0223958` | `1.04599` | `1.68444` | `6/9` | 0 | 0 |

Gradient details：

| dataset | batch | grad pass | grad relerr max | grad abs err max | grad cos |
|---|---:|---:|---:|---:|---:|
| MNIST | 8 | 1 | `7.87e-05` | `3.35e-08` | `0.999995` |
| Fashion-MNIST | 8 | 0 | `1.01e-04` | `2.24e-08` | `1.000000` |
| KMNIST | 8 | 1 | `6.48e-05` | `2.98e-08` | `1.000000` |

判断：

- `SYS20 hidden60` 是目前最有信息量的组合修复：5-seed macro hard gate 过线，memory 也在 S2 内。
- 但它不能记为 v8.0 minimum success，因为：
  - GradPass 有一行 `1.01e-04 > 1e-4`。
  - FullGridS2 因 step max `1.68444` 失败，只过 `6/9` shapes。
- 不能因为 abs err 很小而手动改 pass。

No-fake/no-proxy：`rows_checked=908`，fake/proxy nonzero `0`。

### 33.3 SYS20 hidden57 full-grid screen

为检查 hidden60 的 grad 边界是否由容量造成，继续测 hidden57：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys20_packed_alpha010_hidden57_5seed_20260507T173000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS20 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | GradPass | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS20 hidden57` | 5 | `+0.0165365` | `+0.0096354` | `1.16e-03` | `+0.0225260` | `1.04499` | `1.76330` | `7/9` | 1 | 0 |

判断：

- hidden57 修复了 gradient gate，但 macro 明显回落。
- step 仍不稳定，FullGridS2 未闭合。
- 因此 hidden57 不是有效替代路线。

No-fake/no-proxy：`rows_checked=909`，fake/proxy nonzero `0`。

## 34. SYS20 组合修复后最终更新结论

`SYS20` 证明了一个新的真实机制信号：packed path 和 alpha010 self-bootstrap 的组合可以在 5-seed 下把 teacher-free macro 推过 `+0.0200`，但仍没有形成合格 official candidate。

| candidate | macro | GradPass | S2 | 判断 |
|---|---:|---:|---:|---|
| `SYS20 hidden60` | `+0.02031` | fail | fail | macro 过线，但 grad/S2 失败 |
| `SYS20 hidden57` | `+0.01654` | pass | fail | grad 过，但 task/S2 失败 |
| `SYS21 smoke` | not_run full | fail | not_run | alpha050 直接因 GradFail 停止 |

新增机制判断：

1. `SYS20 hidden60` 是目前最接近 v8.0 minimum 的 measured candidate，但不是成功。
2. packed + alpha010 可以补 teacher-free macro gap，但该组合放大了 step instability，并在 hidden60 上触发极窄的 gradient relerr fail。
3. 等价导数修复将 Fashion-MNIST grad relerr 从约 `1.13e-04` 降到 `1.01e-04`，但仍未过 gate；不能手动放宽。
4. 当前失败已经从“完全没有 macro+S2 同时信号”推进到“macro/S2/grad 三门不能同时闭合”。
5. 下一步如果继续，不应再记 hidden sweep 为主线；更合理的是：
   - 为 packed recompute stack 做更严格的 numerical gradient repair；
   - 或实现 phase-mapped fused/streaming package，稳定 step；
   - 或进入真正 RR representation primitive，而不是继续 external teacher 或 classwise/loss 修补。

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `c9a0c88afcd66ed129da50e3333e77e2b51aec5e1bf4d75014b9dc76cdce83af` |
| `SYS20 hidden60 route` | `16a7cdd56d87ca54ca31c03c79648e74b82f4d6b2ee5c7ee3c212c37c26aff2d` |
| `SYS20 hidden60 provenance` | `127eb18f5b11167cd33d16fb0e20163c49927058ecef556f3bdf2c843dce537a` |
| `SYS20 hidden57 route` | `0c18d434ac9ca4c73a8c9a66267cd3d2b1094a8d7c5b28c03d9b23233763b813` |
| `SYS20 hidden57 provenance` | `1f0a0de558df93b8a19b85d4c49a81b44b41e06146face54a1343d5de76ecfc7` |

最终判断：

> v8.0 仍未完成全部计划，也仍未达成 minimum success。新增 `SYS20` 把问题推进了一步：teacher-free macro 可以在 packed path 上过线，但 grad correctness 和 full-grid S2 不同时成立。当前不能 claim official autonomous advantage；必须继续做 packed stack 数值梯度修复、phase-mapped fused/streaming system closure，或实现新的 teacher-free representation primitive。

## 35. 追加自修复：SYS22 / SYS23 no-external self-bootstrap system probes

`SYS20 hidden60` 失败后继续做两类 code-native 自修复，仍保持：

```text
external_teacher_used = 0
fake_data_used = 0
proxy_row_used = 0
no loss/classwise/sampler/focal/margin/class weight 调参
```

### 35.1 packed recompute gradient repair smoke

先把 `PackedRecomputeFusedLinearSiluStack.backward_manual` 从逐层前缀重复重算改为 backward 内一次性重算该 step 的 `hs/ys`，并尝试 `aten.silu_backward` 对齐 PyTorch SiLU backward kernel。

结果：

| probe | candidate | hidden | grad rows | max relerr | 判断 |
|---|---|---:|---:|---:|---|
| prefix recompute | `SYS20` | 60 | `2/3` pass | `1.01e-04` | GradFail |
| `aten.silu_backward` | `SYS20` | 60 | `1/3` pass | `1.13e-04` | 更差，停止 |

判断：`aten.silu_backward` 没有修复 v8.0 的 packed-stack gradient gate；最终保留较好的显式 SiLU 导数路径，不放宽 `1e-4` gate。

### 35.2 SYS22 packed cache-y alpha010

为确认失败是否来自 recompute cache，新增 `SYS22`：

```text
SYS22 = packed cache-y stack + AdamW v BF16 state + bestval self-bootstrap alpha 0.10
```

Smoke：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_sys22_stable_cachey_smoke \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS22 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 1 \
  --bench-reps 2 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

| candidate | grad rows | max relerr | no-fake/no-proxy |
|---|---:|---:|---:|
| `SYS22 hidden60` | `2/3` pass | `1.01e-04` | rows `180`, fake/proxy `0` |

判断：cache-y 不能修复 gradient boundary；未进入 full-grid。

### 35.3 SYS23 alpha005 hidden60

为避免 alpha010 在 hidden60 上触发 gradient 边界，新增 `SYS23`：

```text
SYS23 = SYS20 packed recompute path + bestval self-bootstrap alpha 0.05
```

Smoke 先通过 GradPass：

| candidate | hidden | grad rows | max relerr | no-fake/no-proxy |
|---|---:|---:|---:|---:|
| `SYS23` | 60 | `3/3` pass | `8.44e-05` | rows `183`, fake/proxy `0` |

正式 5-seed run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys23_alpha005_hidden60_5seed_20260507T183000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS23 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS23 hidden60` | 5 | `+0.021224` | `+0.015104` | `2.77e-05` | `+0.017839` | `-0.00243` | `-0.09437` | 1 | `1.06970` | `2.57275` | `4/9` | 0 |

Gradient：

| dataset | grad pass | relerr max | abs err max | cos |
|---|---:|---:|---:|---:|
| MNIST | 1 | `8.44e-05` | `3.07e-08` | `0.999995` |
| Fashion-MNIST | 1 | `8.09e-05` | `1.86e-08` | `1.000000` |
| KMNIST | 1 | `4.27e-05` | `2.24e-08` | `1.000001` |

判断：

- `SYS23 hidden60` 是 v8.0 当前最强 teacher-free measured candidate：macro hard gate、GradPass、StrictPass、no-external-teacher 同时成立。
- 但它不是 v8.0 minimum success，因为 FullGridS2 失败：memory max `1.06970 > 1.05`，step max `2.57275 > 1.50`。
- 当前 blocker 从 `macro/grad/S2 三门不能同时闭合` 推进为 `macro+grad 已闭合，但 S2 system closure 未闭合`。

No-fake/no-proxy：`rows_checked=914`，fake/proxy nonzero `0`。

### 35.4 SYS23 alpha005 hidden58

为了检查是否能靠轻微容量下降回到 S2，同时保住 macro，继续测 `hidden58`：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys23_alpha005_hidden58_5seed_20260507T184500Z \
  --fresh \
  --device auto \
  --hidden-dim 58 \
  --candidates B0,SYS23 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS23 hidden58` | 5 | `+0.015755` | `+0.008464` | `3.68e-03` | `+0.013542` | 1 | `1.06828` | `2.66476` | `6/9` | 0 |

判断：

- `hidden58` 保住 GradPass，但 macro gap 明显跌破 `+0.0200`。
- memory max 仍在 bs512 约 `1.068`，说明容量小降没有修复 S2 memory。
- 该结果支持：当前不是再扫 hidden dim 能解决，而是需要真正 system package / streaming cache 修复。

No-fake/no-proxy：`rows_checked=915`，fake/proxy nonzero `0`。

## 36. SYS23 后阶段结论

截至 `SYS23` 自修复，v8.0 仍没有完成全部计划，也没有达成 minimum success。

当前最强 official teacher-free candidate：

```json
{
  "candidate": "SYS23 hidden60",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": 0.021223958333333334,
  "ci95_low": 0.015104166666666667,
  "holm_p": 2.774470316473212e-05,
  "test_gap": 0.017838541666666666,
  "memory_ratio_max": 1.0696996640980043,
  "step_ratio_max": 2.572747612961785
}
```

更新后的机制判断：

1. `SYS23 hidden60` 首次让 no-external-teacher candidate 同时满足 MacroSignificantPass 与 GradPass。
2. 这不是 official success：FullGridS2 仍失败，而且失败幅度不是 rounding。
3. `SYS20 hidden60` 是 macro pass 但 Grad/S2 fail；`SYS23 hidden60` 修掉 Grad，但仍 S2 fail；`SYS23 hidden58/hidden59` 修不出 macro/S2。
4. `SYS22` 说明 cache-y packed path 不能解决 gradient boundary，也没有必要 full-grid。
5. 下一步不应继续 alpha/hidden 小扫；应做：
   - S2 memory/cache streaming package，尤其 bs512 root input / recompute temp；
   - step instability attribution，定位 Fashion-MNIST/MNIST outlier shape；
   - 或新的 representation primitive，在不增加 bs512 memory 的前提下保住 `SYS23 hidden60` 的 macro。

追加 no-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_sys22_stable_cachey_smoke` | `180` | `0` |
| `tmp_v80_sys23_alpha005_hidden60_smoke` | `183` | `0` |
| `v80_sys23_alpha005_hidden60_5seed_20260507T183000Z` | `914` | `0` |
| `v80_sys23_alpha005_hidden58_5seed_20260507T184500Z` | `915` | `0` |

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `7de9fd458a2cbd62692fd562f09c0833027d0cdd9fa127d69f4ae8f86776ea44` |
| `SYS23 hidden60 route` | `19fe6431427a13cb954a78789ef32b909df82130f977f3260bcf4550a223a1cf` |
| `SYS23 hidden60 provenance` | `4018c22cfbb20aba0b6958564888e985daceb5a67e165fb2d97c0c7260883d63` |
| `SYS23 hidden58 route` | `45c630896609fa835e7ec11816c79b7f1f1b2ac115f64908eca778da97d54939` |
| `SYS23 hidden58 provenance` | `e6059eeba15634e768c986c74f8039bcdad1520d18421edd8dbfb2ff08346bcf` |

最终更新：

> v8.0 仍未达成 minimum success，但失败面已进一步收敛：`SYS23 hidden60` 证明 teacher-free PureKAN-NG 可以不靠外部 teacher 同时过 macro 与 gradient gate；剩余硬 blocker 是 FullGridS2 system closure。不能宣称 official autonomous system closure，下一步必须做真正的 streaming/cache/fused system 修复，而不是继续 external teacher、loss 或 classwise 方向。

## 37. 追加自修复：SYS24 prefix-recompute cache package

`SYS23 hidden60` 已闭合 macro 与 gradient，但 FullGridS2 失败。失败形态提示 current packed recompute backward 会保留较多 hidden temp/live set。因此新增 `SYS24`：

| candidate | change | 目的 |
|---|---|---|
| `SYS24` | `PackedPrefixRecomputeFusedLinearSiluStack`，backward 中按 prefix 重算当前层 activation，不保存整条 hidden/y list | 尝试保留 `SYS23 alpha005` 的 teacher-free signal，同时降低 backward live set |

实现说明：

- `SYS24` 仍为 no-external-teacher candidate，`external_teacher_used=0`。
- `SYS24` 仍使用 same-run bestval self-bootstrap，`alpha_logit=0.05`。
- 没有使用 C3 logits、teacher forward、loss/classwise/sampler/focal/margin/class weight。
- 新增路径已进入 candidate registry、contract validator、gradient checker、route decision 和 provenance audit。

Smoke：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_sys24_prefix_alpha005_hidden60_smoke \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS24 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 1 \
  --bench-reps 2 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 判断：

| item | value |
|---|---:|
| GradPass | `3/3` |
| max grad relerr | `8.44e-05` |
| memory ratio mean | `0.9912` |
| step ratio mean | `1.2399` |
| rows checked | `186` |
| fake/proxy nonzero | `0` |

Smoke 只用于确认代码路径与 gradient/provenance，不作为 task success。

## 38. SYS24 hidden60 / hidden61 real probes

### 38.1 SYS24 hidden60

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys24_prefix_alpha005_hidden60_5seed_20260507T190000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS24 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS24 hidden60` | 5 | `+0.019792` | `+0.014059` | `5.57e-05` | `+0.018229` | `-0.00208` | `-0.08936` | 1 | `1.04599` | `2.70878` | `8/9` | 0 |

Gradient：

| dataset | grad pass | relerr max | abs err max | cos |
|---|---:|---:|---:|---:|
| MNIST | 1 | `8.44e-05` | `3.07e-08` | `0.999995` |
| Fashion-MNIST | 1 | `8.09e-05` | `1.86e-08` | `1.000000` |
| KMNIST | 1 | `4.27e-05` | `2.24e-08` | `1.000001` |

判断：

- `SYS24 hidden60` 把 memory max 从 `SYS23 hidden60` 的 `1.06970` 降到 `1.04599`，说明 prefix-recompute 对 memory live set 有真实帮助。
- 但 macro gap 降到 `+0.019792 < +0.0200`，不能记为 TeacherFreeMacroSignificantPass。
- Step max `2.70878`，FullGridS2 仍失败。最坏点来自 Fashion-MNIST bs128 step spike；没有把它当作 pass。

No-fake/no-proxy：`rows_checked=918`，fake/proxy nonzero `0`。

### 38.2 SYS24 hidden61

为了检查是否能通过微小容量恢复 macro，同时不破坏 memory，继续测 `hidden61`：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys24_prefix_alpha005_hidden61_5seed_20260507T193000Z \
  --fresh \
  --device auto \
  --hidden-dim 61 \
  --candidates B0,SYS24 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | seeds | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `SYS24 hidden61` | 5 | `+0.018620` | `+0.010807` | `1.68e-03` | `+0.015234` | `+0.00106` | `-0.08765` | 1 | `1.04632` | `1.75503` | `7/9` | 0 |

判断：

- `hidden61` 没有恢复 macro，反而降到 `+0.01862`。
- Memory 仍接近 S2，但 step max `1.75503`，S2 仍失败。
- 因此 prefix-recompute + hidden 微调不能闭合 v8.0。

No-fake/no-proxy：`rows_checked=918`，fake/proxy nonzero `0`。

## 39. 当前最终状态

截至 `SYS24` prefix-recompute 自修复，v8.0 计划仍未全部完成，minimum success 仍未达成。

当前 best measured teacher-free candidates：

| candidate | external teacher | self teacher | macro pass | grad pass | S2 pass | macro gap | memory max | step max | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS23 hidden60` | 0 | 1 | 1 | 1 | 0 | `+0.021224` | `1.06970` | `2.57275` | task/grad best, system fail |
| `SYS24 hidden60` | 0 | 1 | 0 | 1 | 0 | `+0.019792` | `1.04599` | `2.70878` | memory improved, macro/step fail |
| `SYS24 hidden61` | 0 | 1 | 0 | 1 | 0 | `+0.018620` | `1.04632` | `1.75503` | macro/S2 fail |

更新后的机制结论：

1. `SYS23 hidden60` 仍是当前最强 teacher-free official candidate：MacroSignificantPass + GradPass 成立，但 FullGridS2 失败。
2. `SYS24 hidden60` 证明 prefix-recompute 能真实降低 memory live set，把 memory max 拉到 S2 线内侧附近；但它牺牲了 macro gap，并且 step 仍失败。
3. `SYS24 hidden61` 没有通过容量微调补回 macro，也未闭合 S2。
4. 当前 blocker 不是外部 teacher、fake/proxy、loss 或 classwise；而是 `teacher-free macro + gradient + fullgrid S2` 三门仍不能同时闭合。
5. 下一步必须做更底层的 system/runtime 修复：真实 phase-mapped step spike attribution、streaming/prefix package 的 step path 优化，或设计不会增加 bs512 live set的新 representation primitive。

追加 no-fake/no-proxy 审计：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_sys24_prefix_alpha005_hidden60_smoke` | `186` | `0` |
| `v80_sys24_prefix_alpha005_hidden60_5seed_20260507T190000Z` | `918` | `0` |
| `v80_sys24_prefix_alpha005_hidden61_5seed_20260507T193000Z` | `918` | `0` |

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `95a55ab3d27ee295a049c13734f2e4e32b4238c81838f23d42abdff17de912b0` |
| `SYS24 hidden60 route` | `1fc367e4bf86e51ba26aa0b3e2afb108709760125d26dfb37e2f603229bfc808` |
| `SYS24 hidden60 provenance` | `f6a08d8cefff7237387db6c1e7ba6c393c82b3878e037d53a9bd20576393ede8` |
| `SYS24 hidden61 route` | `ca8a3cdab49b60325a4329828e5d3f882a13147c3111324d31d61b5b36be7cb8` |
| `SYS24 hidden61 provenance` | `f6a08d8cefff7237387db6c1e7ba6c393c82b3878e037d53a9bd20576393ede8` |
| `SYS24 smoke provenance` | `1f4e1b4d1416e5c5abb440df7acbc334b9cbc907bdb79245eef82885d3cb637e` |

最终更新：

> v8.0 仍未达成全部实验目标或 minimum success。`SYS23 hidden60` 是 task/grad 的当前 best，`SYS24 hidden60` 是 memory 方向的真实改进，但二者都没有同时满足 TeacherFreeMacroSignificantPass、GradPass 与 FullGridS2Pass。不能宣称 official teacher-free system closure；下一步应针对 prefix-recompute 的 step spike 和 SYS23 的 bs512 live set 做真正 kernel/runtime 级修复。

## 40. 追加自修复：SYS13/SYS23 hidden60 co-selection rerun

`SYS13 hidden60` 曾显示较稳的 S2 表现但 macro 不足；`SYS23 hidden60` 显示 macro/GradPass 成立但 S2 失败。为了排除不同 run denominator / profiler 波动造成的误判，追加同目录、同 B0 denominator 的 co-selection rerun：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys13_sys23_hidden60_coselect_5seed_20260507T200000Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,SYS13,SYS23 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

`v80_route_decision.json`：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "SYS23",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "fullgrid_s1_pass": 0,
  "time_auc_pass": 0,
  "profiler_pass": 0,
  "success_v80_minimum": 0,
  "success_v80_formal": 0,
  "macro_gap": "0.021223958333333334",
  "ci95_low": "0.015104166666666667",
  "holm_p": "5.548940632946424e-05",
  "test_gap": "0.017838541666666666",
  "memory_ratio_max": 1.0696996640980043,
  "step_ratio_max": 1.5095252074443015,
  "fullgrid_shape_complete": 1,
  "measured_s2_pass": 0,
  "measured_s1_pass": 0,
  "primary_blocker": "fullgrid_s2_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | ECE | NLL | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `B0` | `0.840625` | `0.795443` | `0.000000` | `0.060161` | `0.570679` | reference |
| `SYS13` | `0.859635` | `0.819661` | `+0.019010` | `0.054398` | `0.475533` | macro near-miss |
| `SYS23` | `0.861849` | `0.813281` | `+0.021224` | `0.057734` | `0.476305` | v8 macro pass, S2 fail |

Significance / calibration：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta |
|---|---:|---:|---:|---:|---:|---:|
| `SYS13` | `+0.019010` | `+0.010938` | `1.589e-03` | `+0.024219` | `-0.005763` | `-0.095146` |
| `SYS23` | `+0.021224` | `+0.015104` | `5.549e-05` | `+0.017839` | `-0.002427` | `-0.094374` |

说明：v8 route 的 `teacher_free_macro_pass=1` 来自预定义 v8 macro gate；旧字段 `macro_significant_task_pass` 仍可能沿用 earlier pipeline 的非 v8 口径，因此本复盘以 `v80_route_decision.json` 为 route 判断依据。

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `SYS13` | `3/3` | `6.86e-05` | `>=0.999999` |
| `SYS23` | `3/3` | `8.44e-05` | `>=0.999998` |

Efficiency summary：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `SYS13` | `1.024924` | `1.069700` | `1.376688` | `1.458082` | `6/9` | FAIL |
| `SYS23` | `1.024924` | `1.069700` | `1.388888` | `1.509525` | `6/9` | FAIL |

`SYS23` per-shape S2 details：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.991207` | `1.470747` | 1 |
| MNIST | 256 | `1.013864` | `1.354064` | 1 |
| MNIST | 512 | `1.069700` | `1.350648` | 0 |
| Fashion-MNIST | 128 | `0.991207` | `1.345123` | 1 |
| Fashion-MNIST | 256 | `1.013864` | `1.374855` | 1 |
| Fashion-MNIST | 512 | `1.069700` | `1.361802` | 0 |
| KMNIST | 128 | `0.991207` | `1.368574` | 1 |
| KMNIST | 256 | `1.013864` | `1.364658` | 1 |
| KMNIST | 512 | `1.069700` | `1.509525` | 0 |

判断：

1. co-selection rerun 复现了 `SYS23 hidden60` 的 teacher-free macro advantage：`+0.021224`，CI95 low `+0.015104`，Holm p `5.55e-05`，且 GradPass 成立。
2. warmup/reps 增强后，`SYS23` 的 step spike 从此前 `2.57275` 降到 `1.50953`，只比 S2 step gate 高 `0.00953`。
3. 剩余系统 blocker 更明确：bs512 memory ratio 固定到 `1.06970`，三 datasets 均因 memory 超过 `1.05` 失败；KMNIST bs512 还额外有轻微 step fail。
4. `SYS13` 与 `SYS23` 的 memory profile 完全一致，说明当前 bs512 live-set blocker 不是 self-bootstrap alpha 特有，而是 packed hidden60 path 的系统结构问题。
5. 因为 FullGridS2 仍失败，不能宣称 v8.0 minimum success。

No-fake/no-proxy：`rows_checked=1298`，fake/proxy nonzero `0`。

追加 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `95a55ab3d27ee295a049c13734f2e4e32b4238c81838f23d42abdff17de912b0` |
| `SYS13/SYS23 co-selection route` | `1eddc4938eb3707db56de4fdcafbd045d5c8455ff28f780a61d4b1b5c7d857ce` |
| `SYS13/SYS23 co-selection task summary` | `5b43d192ef151c78dcb08bf0eee7bef2b157c2e74619dffdcf39002a0a280114` |
| `SYS13/SYS23 co-selection significance` | `f300d5cc883b0e63e57d458377eaa4b32c6644b121e34516c097f4a301802ca1` |
| `SYS13/SYS23 co-selection gradient` | `01a9493fb0725d703bbd3ebd3032109c16c2a7b8135b4bf33a4aec8f31a8a24c` |
| `SYS13/SYS23 co-selection efficiency summary` | `0524264cd589c42e7a838eddd2f51f57adfa7ef4aed7b66367f1b255875dd2e3` |
| `SYS13/SYS23 co-selection efficiency profiler` | `1b31893c3630bac6b812cacfeaba2196fbf157e5c8d2966505f738a1c4884be3` |
| `SYS13/SYS23 co-selection provenance` | `46466396129d363a6e6f87e38b2d99384767150e5af03731254fca643c2e360a` |

## 41. 最新最终状态

截至 SYS13/SYS23 hidden60 co-selection rerun，v8.0 计划仍未全部完成，minimum success 仍未达成。

当前最接近 official teacher-free system closure 的候选是：

| candidate | external teacher | self teacher | strict | grad | macro pass | S2 | macro gap | memory max | step max | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SYS23 hidden60` | 0 | 1 | 1 | 1 | 1 | 0 | `+0.021224` | `1.069700` | `1.509525` | task/grad closed, S2 fail |
| `SYS13 hidden60` | 0 | 1 | 1 | 1 | 0 | 0 | `+0.019010` | `1.069700` | `1.458082` | S2-near, macro fail |
| `SYS24 hidden60` | 0 | 1 | 1 | 1 | 0 | 0 | `+0.019792` | `1.04599` | `2.70878` | memory improved, step/macro fail |

最终更新：

> v8.0 尚未完成全部实验，也未达成 minimum success。最新 co-selection run 证明 `SYS23 hidden60` 已经真实闭合 teacher-free macro + strict + gradient 三门，但 FullGridS2 仍失败；剩余 blocker 已从“任务和系统都不确定”收敛为“bs512 memory live set `1.06970` + KMNIST bs512 step `1.50953`”。下一步必须针对 packed hidden60 path 的 bs512 live set 做真实 kernel/cache 修复，不能把当前结果写成成功。

## 42. 追加确认：SYS8 hidden57 10-seed clean confirmation

已有 `SYS8 hidden57` 5-seed screen 曾显示 macro near-pass / pass 边缘，但 FullGridS2 只差很小的 memory/step margin。为了避免把 5-seed near-pass 误判为 official result，追加 10-seed clean confirmation：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys8_hidden57_clean_confirm_10seed_20260506T210000Z \
  --fresh \
  --device auto \
  --hidden-dim 57 \
  --candidates B0,SYS8 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

`v80_route_decision.json`：

```json
{
  "route": "R7-NoReproduction",
  "best_official_candidate_id": "SYS8",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "fullgrid_s1_pass": 0,
  "time_auc_pass": 0,
  "profiler_pass": 0,
  "success_v80_minimum": 0,
  "success_v80_formal": 0,
  "macro_gap": "0.0162109375",
  "ci95_low": "0.010872395833333333",
  "holm_p": "1.0878883098666394e-05",
  "test_gap": "0.020833333333333332",
  "memory_ratio_max": 1.0508059125009284,
  "step_ratio_max": 1.613366942232378,
  "fullgrid_shape_complete": 1,
  "measured_s2_pass": 0,
  "measured_s1_pass": 0,
  "primary_blocker": "teacher_free_macro_gap_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | ECE | NLL |
|---|---:|---:|---:|---|---:|---:|
| `B0` | `0.844206` | `0.793034` | `0.000000` | F-MNIST `0`, KMNIST `0`, MNIST `0` | `0.061221` | `0.569885` |
| `SYS8` | `0.860417` | `0.813867` | `+0.016211` | F-MNIST `+0.006836`, KMNIST `+0.015039`, MNIST `+0.026758` | `0.052223` | `0.482445` |

Significance / calibration：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | v8 macro pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SYS8` | `+0.016211` | `+0.010872` | `1.09e-05` | `+0.020833` | `-0.008998` | `-0.087440` | 0 |

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `SYS8` | `3/3` | `8.91e-05` | `0.999995` |

Efficiency summary：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `SYS8` | `1.012828` | `1.050806` | `1.428766` | `1.613367` | `3/9` | FAIL |

Per-shape S2 details：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.983982` | `1.132596` | 1 |
| MNIST | 256 | `1.003695` | `1.512440` | 0 |
| MNIST | 512 | `1.050806` | `1.461709` | 0 |
| Fashion-MNIST | 128 | `0.983982` | `1.505300` | 0 |
| Fashion-MNIST | 256 | `1.003695` | `1.613367` | 0 |
| Fashion-MNIST | 512 | `1.050806` | `1.407065` | 0 |
| KMNIST | 128 | `0.983982` | `1.412476` | 1 |
| KMNIST | 256 | `1.003695` | `1.407004` | 1 |
| KMNIST | 512 | `1.050806` | `1.406936` | 0 |

判断：

1. `SYS8 hidden57` 10-seed clean confirm 没有复现 5-seed 的 macro gate：val gap 只有 `+0.016211`，因此 teacher-free macro pass 不成立。
2. GradPass 成立，且 ECE/NLL 仍优于 MLP，但这不能替代 macro gate。
3. FullGridS2 也未达成：memory max `1.050806` 只比 S2 memory gate 高 `0.000806`，但 step max `1.613367` 明显高于 `1.50`。
4. 因此 `SYS8 hidden57` 不能作为 v8.0 minimum success，也不应继续作为 official candidate 主线；它更像是 5-seed noise / denominator-sensitive near-pass。

No-fake/no-proxy：`rows_checked=1653`，fake/proxy nonzero `0`。

追加 hash：

| artifact | SHA256 |
|---|---|
| `SYS8 hidden57 clean route` | `39ff89ac9d98c21698396e6f9ac97115b5184612c858efabe04a3de6d239c2c6` |
| `SYS8 hidden57 clean task summary` | `fb8f52ea0b588eb50fc60c34da3962d16eb585cf548bbb400edc616b17c22656` |
| `SYS8 hidden57 clean significance` | `9d380e7e5dbbcddc3d6b6bb4742459b189345b33a1441df65ae0ac3394cf857a` |
| `SYS8 hidden57 clean gradient` | `b82fcce1433208268cf6d06e154d999faa55e333d219c290e8175510b8b6e6f5` |
| `SYS8 hidden57 clean efficiency summary` | `17eadbb0d000f473135f2114f988535cffca490409b3d13b5d99525f9373e926` |
| `SYS8 hidden57 clean efficiency profiler` | `fe5cdcdb0af3920f8cc831684281e7f90e842093da07c5edb685a75d11b1e3e8` |
| `SYS8 hidden57 clean provenance` | `f57d0cbd07e438916890d959acb72b2a41ed140c09c0ff112f20241faef93c3b` |

## 43. 最新最终状态（SYS8 clean confirm 后）

v8.0 全部实验仍未完成，minimum success 仍未达成。

最新 candidate 判断：

| candidate | run type | macro pass | grad pass | S2 pass | macro gap | memory max | step max | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `SYS23 hidden60` | 5-seed co-selection | 1 | 1 | 0 | `+0.021224` | `1.069700` | `1.509525` | 当前 task/grad best，S2 fail |
| `SYS8 hidden57` | 10-seed clean confirm | 0 | 1 | 0 | `+0.016211` | `1.050806` | `1.613367` | 5-seed near-pass 未复现 |
| `SYS17 hidden57` | 10-seed confirm | 0 | 1 | 1 | `+0.018620` | `1.046473` | `1.497284` | S2 pass，但 macro fail |

最终更新：

> v8.0 到目前仍没有 official teacher-free system closure。`SYS23 hidden60` 证明 task/Grad 可以闭合，但 S2 失败；`SYS17 hidden57` 证明 S2 可以闭合，但 10-seed macro 不过；`SYS8 hidden57` 的 5-seed near-pass 在 10-seed clean confirm 中退回 `+0.016211`。当前结论更明确：不是 fake/proxy 或外部 teacher 问题，而是 teacher-free macro advantage 与 FullGridS2 仍不能在同一候选上稳定共存。

## 44. 追加自修复：SYS23 hidden59 Pareto middle point

`SYS23 hidden58` task 不足，`SYS23 hidden60` task/GradPass 成立但 FullGridS2 失败。为了检查是否存在 hidden58/60 之间的中间 Pareto 闭合点，追加 `SYS23 hidden59` real probe：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_sys23_alpha005_hidden59_5seed_20260506T220000Z \
  --fresh \
  --device auto \
  --hidden-dim 59 \
  --candidates B0,SYS23 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

`v80_route_decision.json`：

```json
{
  "route": "R7-NoReproduction",
  "best_official_candidate_id": "SYS23",
  "external_teacher_used": 0,
  "self_teacher_used": 1,
  "code_native_pass": 1,
  "teacher_contract_v2_pass": 1,
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "fullgrid_s1_pass": 0,
  "time_auc_pass": 0,
  "profiler_pass": 0,
  "success_v80_minimum": 0,
  "success_v80_formal": 0,
  "macro_gap": "0.016276041666666668",
  "ci95_low": "0.008463541666666666",
  "holm_p": "0.007556598278379023",
  "test_gap": "0.026822916666666665",
  "memory_ratio_max": 1.0689902360647634,
  "step_ratio_max": 1.4861596599980162,
  "fullgrid_shape_complete": 1,
  "measured_s2_pass": 0,
  "measured_s1_pass": 0,
  "primary_blocker": "teacher_free_macro_gap_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | ECE | NLL |
|---|---:|---:|---:|---|---:|---:|
| `B0` | `0.843490` | `0.796484` | `0.000000` | F-MNIST `0`, KMNIST `0`, MNIST `0` | `0.057157` | `0.555782` |
| `SYS23 hidden59` | `0.859766` | `0.823307` | `+0.016276` | F-MNIST `+0.014844`, KMNIST `+0.010156`, MNIST `+0.023828` | `0.055521` | `0.481144` |

Significance / calibration：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | v8 macro pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SYS23 hidden59` | `+0.016276` | `+0.008464` | `7.56e-03` | `+0.026823` | `-0.001636` | `-0.074638` | 0 |

Gradient correctness：

| candidate | pass rows | max relerr | grad cos min |
|---|---:|---:|---:|
| `SYS23 hidden59` | `3/3` | `8.52e-05` | `0.999997` |

Efficiency summary：

| candidate | memory mean | memory max | step mean | step max | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `SYS23 hidden59` | `1.024222` | `1.068990` | `1.367094` | `1.486160` | `6/9` | FAIL |

判断：

1. `SYS23 hidden59` 没有形成 hidden58/60 之间的可用中间点：macro gap 只有 `+0.016276`，低于 v8.0 `+0.0200` hard gate。
2. GradPass 成立，test gap 也很高，但 validation macro gate 不成立，不能进入 official success。
3. Step max `1.48616` 已在 S2 step gate 内，但 bs512 memory max `1.06899` 仍明显高于 `1.05`；因此 system 侧也没有闭合。
4. 该结果支持当前判断：`SYS23` 的 task gain 对 hidden 维度较敏感，不能靠 hidden59 同时保住 hidden60 的 macro 和 hidden57/58 的 system margin。

No-fake/no-proxy：`rows_checked=918`，fake/proxy nonzero `0`。

追加 hash：

| artifact | SHA256 |
|---|---|
| `SYS23 hidden59 route` | `5fec85d28e8e48c6cfff200acfb8e60bd44e0ad2d507efbb2ca89ff1748c7e4f` |
| `SYS23 hidden59 task summary` | `cdc4fe5b6b3378d4e14718404157394b7a21334939f2d4e61e1b4735214c4809` |
| `SYS23 hidden59 significance` | `9b5d2f3c6626bdab5846192e9445e73ed77c28ad8c26c0dfb96df33b00139bc5` |
| `SYS23 hidden59 gradient` | `f51b7d6fb040870bf1713fbbadfc1a0463c4c278b8c317e24cf367ae0816485f` |
| `SYS23 hidden59 efficiency summary` | `78b93cb0c15decfe640947421d5ec887d513de527ebb31032227155568e23187` |
| `SYS23 hidden59 efficiency profiler` | `697ee7b4d8262af76b911dae35f38d11122bfe7093f22717a37a09bfb30fac62` |
| `SYS23 hidden59 provenance` | `f6a08d8cefff7237387db6c1e7ba6c393c82b3878e037d53a9bd20576393ede8` |

## 45. 最新最终状态（SYS23 hidden59 后）

v8.0 全部实验仍未完成，minimum success 仍未达成。

最新 Pareto 判断：

| candidate | run type | macro pass | grad pass | S2 pass | macro gap | memory max | step max | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `SYS23 hidden60` | 5-seed co-selection | 1 | 1 | 0 | `+0.021224` | `1.069700` | `1.509525` | task/grad best, S2 fail |
| `SYS23 hidden59` | 5-seed middle point | 0 | 1 | 0 | `+0.016276` | `1.068990` | `1.486160` | hidden middle point fail |
| `SYS17 hidden57` | 10-seed confirm | 0 | 1 | 1 | `+0.018620` | `1.046473` | `1.497284` | S2 pass, macro fail |
| `SYS8 hidden57` | 10-seed clean confirm | 0 | 1 | 0 | `+0.016211` | `1.050806` | `1.613367` | 5-seed near-pass 未复现 |

最终更新：

> v8.0 仍未达成全部实验目标。`SYS23 hidden59` 证明 hidden58/60 之间没有简单的中间闭合点：step 进入 S2，但 macro 明显不足，bs512 memory 仍失败。当前必须承认：已有 measured candidates 里没有同时满足 teacher-free macro、GradPass 与 FullGridS2 的 official candidate；后续需要真正的新 representation / memory primitive，而不是继续把 hidden 维度在 58-60 附近微调。

## 46. 追加执行：P6 core primitive code-native audit

根据计划 P6，继续检查 `SparseInterpKANDense`、`SparseSplineKANDense`、`DWM2LiteDense`、`GEMMNativeDepthwiseMixDense`、`DWM2Dense`、`RationalKATV2Dense` 是否已经具备可进入 official route 的 code-native/manual path。该步骤只做代码与真实 batch forward contract audit，不写 task/proxy rows，不把 autograd-only forward 记为 official success。

运行目录：

```text
results/real_rerun_20260506/v80_p6_core_primitive_audit_20260506T230000Z
```

Audit 设置：

```text
dataset = MNIST
seed = 0
real input rows = 16
input_dim = 784
out_dim = 57
basis_count = 8
device = cuda
allow_fake_data = false
```

Audit summary：

| item | value |
|---|---:|
| rows checked | `7` |
| forward smoke pass | `7/7` |
| official eligible | `0/7` |
| fake/proxy nonzero | `0` |
| route | `P6-CorePrimitiveManualPathNotAvailable` |

Primitive audit：

| candidate | dense class | forward smoke | param count | manual forward | manual backward | manual update | official eligible | block reason |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `RR1` | `SparseInterpKANDense` | 1 | `53369` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |
| `RR2` | `SparseSplineKANDense` | 1 | `53369` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |
| `RR3` | `DWM2LiteDense` rbf | 1 | `51017` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |
| `RR4` | `DWM2LiteDense` lut | 1 | `51017` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |
| `RR5` | `GEMMNativeDepthwiseMixDense` | 1 | `53369` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |
| `RR6` | `DWM2Dense` | 1 | `48621` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |
| `RR7` | `RationalKATV2Dense` | 1 | `49449` | 0 | 0 | 0 | 0 | `missing_manual_forward_backward_update_contract` |

判断：

1. 这些 core primitive 已经能在真实 MNIST batch 上完成 forward，说明 import/shape/basic finite contract 没问题。
2. 但它们当前只有 `nn.Module` autograd forward，没有 `forward_manual`、`backward_manual`、`params_and_grads` 这三个 official manual/code-native contract。
3. 因此不能把 `RR1/RR2/RR3/RR4/RR5/RR6/RR7` 作为 v8.0 official task/efficiency candidate 跑，也不能写 `GradPass` 或 `S2Pass`。
4. 计划中 P6/P7 的真正下一步不是继续 hidden 微调，而是给至少一个 primitive 补齐 analytic/manual forward-backward-update contract；补齐前只能作为 diagnostic/autograd primitive，不能 official claim。

No-fake/no-proxy：

| artifact | value |
|---|---|
| `p6_core_primitive_repair_audit_v80.csv` rows | `7` |
| fake/proxy nonzero | `0` |
| no_fake | true |
| no_proxy | true |

关键 hash：

| artifact | SHA256 |
|---|---|
| `p6_core_primitive_repair_audit_v80.csv` | `719d3ba6a5ff0bdff8a741f65a3f6336e3c5ee4c5882e32d39b65ff2cba3931f` |
| `v80_p6_core_primitive_audit_summary.json` | `836440cca261ad80619e993c968438095127f42293ef404d964f137266ee779a` |
| `v80_provenance_audit.csv` | `fb14ecc14f529dc7e92c53788187e961436a1a51c567bc5e6a9861633ef2195a` |

最新更新：

> v8.0 仍未完成全部实验，也仍未达成 minimum success。新增 P6 audit 进一步说明：计划中列出的新 representation primitive 目前并非可直接进入 official route 的实现，它们缺 manual backward/update contract。当前 blocker 从 measured candidate Pareto fail 进一步收敛为两个硬点：一是 teacher-free macro 与 FullGridS2 尚未稳定同候选闭合；二是新 primitive 需要先补齐 code-native/manual contract，才能开展真实 task/Grad/efficiency 实验。

## 47. 追加自修复：RR3 DWM2Lite-RBF manual/code-native primitive

P6 audit 显示 `RR1-RR7` 均缺 manual contract 后，继续实现最小可审计的新 primitive：`RR3-M13-DWM2LiteDense-rbf-manual-code-native`。本次修改不是 loss、sampler、classwise 或 teacher 修补，而是补齐 primitive 的 `forward_manual`、`backward_manual`、`params_and_grads` 与 autograd reference gradient check。

实现位置：

| file | change |
|---|---|
| `experiments/run_gafu_v80_real.py` | 新增 `ManualDWM2LiteRBFLayer` / `ManualDWM2LiteRBFStack` |
| `experiments/run_gafu_v80_real.py` | 将 `RR3` 接入 v8.0 candidate registry、contract、manual train path、gradient check 与 P10 cache attribution |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

Smoke run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr3_manual_smoke \
  --fresh \
  --device auto \
  --candidates B0,RR3 \
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

| item | value |
|---|---:|
| fake/proxy nonzero | `0` |
| RR3 GradPass rows | `3/3` |
| RR3 max relerr | `9.25e-05` |
| RR3 strict/manual contract | pass |

判断：`RR3` 已从“forward-only/autograd primitive”推进为可进入 v8.0 official manual route 的真实候选。

## 48. RR3 5-seed measured probes

### 48.1 RR3 basis8 manual probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr3_dwm2lite_manual_probe_5seed_20260506T231500Z \
  --fresh \
  --device auto \
  --candidates B0,RR3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | external teacher | val acc | test acc | val gap vs MLP | dataset gaps |
|---|---:|---:|---:|---:|---|
| `B0` | 0 | `0.8443` | `0.7949` | `0.0000` | reference |
| `RR3 basis8` | 0 | `0.8655` | `0.8182` | `+0.0212` | F-MNIST `+0.0176`, KMNIST `+0.0211`, MNIST `+0.0250` |

Significance / Grad / efficiency：

| metric | value |
|---|---:|
| mean val gap | `+0.021224` |
| CI95 low | `+0.014844` |
| Holm p | `5.65e-05` |
| test gap | `+0.023307` |
| GradPass | `3/3` |
| max grad relerr | `9.25e-05` |
| memory ratio max | `1.6899` |
| step ratio mean | `2.4001` |
| step ratio max | `3.2670` |
| S2 small-grid pass | `0/3` |

判断：

- `RR3 basis8` 是 v8.0 中第一个 teacher-free、manual/code-native、GradPass 且 5-seed macro gate 过线的新 primitive candidate。
- 但它远未进入 S2；P10 attribution 显示 top memory source 为 `rr3_rbf_basis_cache`，`cache_basis_MB=3.5625`，`manual_cache_MB=4.515625`。
- 因此不能记为 v8.0 minimum success。

### 48.2 RR3 basis4 memory repair

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr3_dwm2lite_basis4_probe_5seed_20260506T233000Z \
  --fresh \
  --device auto \
  --basis-count 4 \
  --candidates B0,RR3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果摘要：

| metric | value |
|---|---:|
| mean val gap | `+0.021615` |
| CI95 low | `+0.015625` |
| Holm p | `2.89e-05` |
| test gap | `+0.023568` |
| GradPass | `1/3` |
| max grad relerr | `1.11e-04` |
| memory ratio max | `1.3399` |
| step ratio mean | `2.0176` |
| step ratio max | `2.7006` |
| S2 small-grid pass | `0/3` |

判断：

- basis4 的 macro 更强，但 gradient correctness 失败，不能进入 route best。
- memory 明显低于 basis8，但仍远高于 S2。

### 48.3 RR3 basis2 memory/grad repair

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr3_dwm2lite_basis2_probe_5seed_20260506T234500Z \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | external teacher | val acc | test acc | val gap vs MLP | dataset gaps |
|---|---:|---:|---:|---:|---|
| `B0` | 0 | `0.8443` | `0.7949` | `0.0000` | reference |
| `RR3 basis2` | 0 | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` |

Significance / Grad / efficiency：

| metric | value |
|---|---:|
| mean val gap | `+0.021094` |
| CI95 low | `+0.014714` |
| Holm p | `5.96e-05` |
| test gap | `+0.023438` |
| GradPass | `3/3` |
| max grad relerr | `9.14e-05` |
| memory ratio max | `1.1683` |
| step ratio mean | `2.0261` |
| step ratio max | `2.4938` |
| S2 small-grid pass | `0/3` |
| `cache_basis_MB` | `0.890625` |
| `manual_cache_MB` | `1.843750` |

判断：

- `RR3 basis2` 是当前最好的 code-native repair tradeoff：teacher-free macro pass、GradPass 成立，并把 memory ratio 从 basis8 的 `1.6899` 降到 `1.1683`。
- 但 memory 仍高于 S2 的 `1.05`，step 仍高于 S2 的 `1.50`。
- 因此它是 measured promising primitive，不是 v8.0 success。

## 49. RR3 runs no-fake / no-proxy 与 hash

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v80_rr3_dwm2lite_manual_probe_5seed_20260506T231500Z` | `905` | `0` |
| `v80_rr3_dwm2lite_basis4_probe_5seed_20260506T233000Z` | `905` | `0` |
| `v80_rr3_dwm2lite_basis2_probe_5seed_20260506T234500Z` | `905` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `cdd50366d220a72626b949d570f78c631e95ca578c5218fcf4f26923c96df9a8` |
| basis8 `v80_route_decision.json` | `792fc37187b9c67468801489987809db814f79446e9567201e307d782b1710f0` |
| basis8 `p9_task_summary.csv` | `859a24f68a215134c69e5333ab7249a4d1a81faeb469cb6a3238e8bdd22cd11d` |
| basis8 `p1_significance_audit.csv` | `b344c1504b9879fe98e29e5af251b6dfe7334e62dd1450360a58d27d8ba33950` |
| basis8 `p2_full_gradient_correctness.csv` | `c172b75bd51a12a33b4167fe07c63cdf828ed128bc064d3eeb416fb42a5eb154` |
| basis8 `p10_efficiency_summary.csv` | `81acf937505c62b6f1878174e29e389af828f8ec27701860ecfc0d0569a47e37` |
| basis8 `v80_provenance_audit.csv` | `855ff630fa6fa7d5c492ef72391e2bb2b56aa4c754c438e6ee6a7a17eaa03824` |
| basis4 `v80_route_decision.json` | `f8d1480a14029b131193360690f4c0a4fbd9d374d1502ed7848806fd432f9383` |
| basis4 `p2_full_gradient_correctness.csv` | `ce521b3c251e955a1ecdd725f0af37fde863aec6b5bf4ede53f5844d6ff71280` |
| basis4 `p10_efficiency_summary.csv` | `7c9642ffcb334b2227e73fb88e047e4291a72b49e65616c83d90a2db9b850dd6` |
| basis2 `v80_route_decision.json` | `806bd69b95c8ae2e1db7ab1740191c67eacf950ad6cb0a6b4c0f9cb0326af16f` |
| basis2 `p9_task_summary.csv` | `2c6dd3d03c0ba330b452b6483aabd31e7ab0219a15275a40741993eb632f11da` |
| basis2 `p1_significance_audit.csv` | `aba5190ac9092edae81c5242f9db38ba75f6403cdf82ca6480cff967396d443b` |
| basis2 `p2_full_gradient_correctness.csv` | `46a3337b040a4a3ba3d264f29fc0fd173543a3e34c44ddb2eb0ca316cef2c3f4` |
| basis2 `p10_efficiency_summary.csv` | `accea2b76dc6c6202ceb633831110a22918773a8d543b065e77a3e6d0fe4df22` |
| basis2 `v80_provenance_audit.csv` | `855ff630fa6fa7d5c492ef72391e2bb2b56aa4c754c438e6ee6a7a17eaa03824` |

## 50. 当前最终更新

截至 RR3 manual/code-native primitive probe，v8.0 仍没有完成全部实验，也没有达成 minimum success。

已达成的新事实：

1. `RR3` 已补齐 manual/code-native primitive contract，不再只是 autograd-only diagnostic。
2. `RR3 basis8` 与 `RR3 basis2` 在 5-seed teacher-free setting 下均达到 macro gate，且 GradPass 成立。
3. `RR3 basis2` 把 memory ratio 从 `1.6899` 降到 `1.1683`，说明 basis cache 是真实可压的 blocker。

仍未闭合：

1. `RR3 basis2` 仍未进 S2：memory max `1.1683 > 1.05`，step max `2.4938 > 1.50`。
2. `RR3 basis4` 虽然 macro 更强，但 GradPass 失败，不能作为 route best。
3. 当前所有 measured official candidates 中，仍没有同一个 candidate 同时满足 teacher-free macro、GradPass 与 FullGridS2。

最终一句话：

> v8.0 尚未完成全部实验，也未达成 system closure；但本轮从“没有可 official 的新 primitive”推进到“RR3 manual primitive 已能 teacher-free 过 macro+Grad”。新的硬 blocker 已经收敛为 RR3 的 basis/materialized cache 与 step time，需要继续做 streaming/recompute/fused RR3 backward，而不是再回到 teacher、loss 或 classwise 修补。

## 51. 追加自修复：RR8 streaming-basis manual primitive

`RR3 basis2` 仍然因为 basis/materialized cache 与 step time 失败后，继续实现 `RR8-M13-DWM2LiteDense-rbf-streaming-manual-code-native`。它与 `RR3` 数学族相同，但 forward/backward 中逐 basis 计算 RBF 项，不缓存三维 `basis` tensor；目标是验证“去 basis materialization”能否把 memory 推近 S2。

实现位置：

| file | change |
|---|---|
| `experiments/run_gafu_v80_real.py` | 新增 `ManualDWM2LiteRBFStreamingLayer` / `ManualDWM2LiteRBFStreamingStack` |
| `experiments/run_gafu_v80_real.py` | 新增 `RR8` candidate、registry、manual train、gradient check、cache attribution |

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

Smoke run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr8_streaming_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR8 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 判断：

| item | value |
|---|---:|
| fake/proxy nonzero | `0` |
| RR8 GradPass rows | `3/3` |
| RR8 max relerr | `8.37e-05` |
| RR8 strict/manual contract | pass |

### 51.1 RR8 basis2 5-seed probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr8_streaming_basis2_probe_5seed_20260507T000000Z \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR8 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | external teacher | val acc | test acc | val gap vs MLP | dataset gaps |
|---|---:|---:|---:|---:|---|
| `B0` | 0 | `0.8443` | `0.7949` | `0.0000` | reference |
| `RR8 basis2` | 0 | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` |

Significance / Grad / efficiency：

| metric | value |
|---|---:|
| mean val gap | `+0.021094` |
| CI95 low | `+0.014974` |
| Holm p | `5.96e-05` |
| test gap | `+0.023438` |
| GradPass | `3/3` |
| max grad relerr | `8.37e-05` |
| memory ratio max | `1.1198` |
| step ratio mean | `2.8748` |
| step ratio max | `2.9498` |
| S2 small-grid pass | `0/3` |
| `cache_basis_MB` | `0.0000` |
| `manual_cache_MB` | `0.953125` |

对比 `RR3 basis2`：

| candidate | macro gap | GradPass | memory max | step mean | step max | cache basis |
|---|---:|---:|---:|---:|---:|---:|
| `RR3 basis2` | `+0.021094` | 1 | `1.1683` | `2.0261` | `2.4938` | `0.890625 MB` |
| `RR8 basis2` | `+0.021094` | 1 | `1.1198` | `2.8748` | `2.9498` | `0.000000 MB` |

判断：

1. RR8 证明 basis materialization 确实是可压 memory source：memory max 从 `1.1683` 降到 `1.1198`，`cache_basis_MB` 从 `0.890625` 降为 `0`。
2. 但 naive streaming 在 Python/basis loop 中重算 exp，step 更差：mean step 从 `2.0261` 升到 `2.8748`。
3. 因此 RR8 不是 S2 解法；它说明下一步必须是 fused/compiled streaming basis，而不是 Python-level streaming。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_rr8_streaming_smoke` | `189` | `0` |
| `v80_rr8_streaming_basis2_probe_5seed_20260507T000000Z` | `908` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `1e4010c288fe16a0009581455d7aa19317607ebc686b47f17d0cfe01ecd8df64` |
| `v80_route_decision.json` | `c0d1ae7be3be0836d6d816fdf9dc6566575aded3605cc792527838fc7e1b5b0e` |
| `p9_task_summary.csv` | `99eab5f1020c0a996f49cc1b8ea1174985a1db14f43c840e48abf971b11cc58f` |
| `p1_significance_audit.csv` | `b520dc94b47ec67779b5795104a6c411ab48aec042203ddf2def5d289dede6fa` |
| `p2_full_gradient_correctness.csv` | `1caa7a6e3a772fdd7b5c247d6fe2d7262d44bffac14b2bf7ad3a78fe2ba046f2` |
| `p10_efficiency_summary.csv` | `b3b350817afd509123e820b002e1fb9d7a3aa3cb9d0f4cae8811eb20d620a28d` |
| `p10_efficiency_profiler.csv` | `db84629df280b652115d8c606400a17065d45eb32df87f9ad1982522b97cf138` |
| `v80_provenance_audit.csv` | `127eb18f5b11167cd33d16fb0e20163c49927058ecef556f3bdf2c843dce537a` |

最新最终判断：

> v8.0 仍没有完成全部实验，也仍未达成 minimum success。`RR8` 把 memory blocker 从 “basis cache 是否真实” 变成了 measured fact：去掉 basis cache 能降 memory，但 Python streaming 让 step 更差。当前最合理的下一步是实现 fused/compiled RR8-style streaming basis forward-backward；在此之前不能声称 S2，也不能声称 system closure。

## 52. 追加自修复：RR9 basis2 unrolled manual primitive

`RR8` 去掉 basis cache 后 step 明显恶化，因此继续实现 `RR9-M13-DWM2LiteDense-rbf-basis2-unrolled-manual-code-native`。`RR9` 固定 basis2，并把两项 RBF 展开为 closed-form tensor expression，不走 Python basis loop，也不缓存 3D basis。

实现位置：

| file | change |
|---|---|
| `experiments/run_gafu_v80_real.py` | 新增 `ManualDWM2LiteRBF2Layer` / `ManualDWM2LiteRBF2Stack` |
| `experiments/run_gafu_v80_real.py` | 新增 `RR9` candidate、registry、manual train、gradient check、cache attribution |

Smoke run：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr9_basis2_unrolled_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR9 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 判断：

| item | value |
|---|---:|
| fake/proxy nonzero | `0` |
| RR9 GradPass rows | `3/3` |
| RR9 max relerr | `9.71e-05` |
| RR9 strict/manual contract | pass |

### 52.1 RR9 basis2 5-seed probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr9_basis2_unrolled_probe_5seed_20260507T001500Z \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR9 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | external teacher | val acc | test acc | val gap vs MLP | dataset gaps |
|---|---:|---:|---:|---:|---|
| `B0` | 0 | `0.8443` | `0.7949` | `0.0000` | reference |
| `RR9 basis2` | 0 | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` |

Significance / Grad / efficiency：

| metric | value |
|---|---:|
| mean val gap | `+0.021094` |
| CI95 low | `+0.014844` |
| Holm p | `5.96e-05` |
| test gap | `+0.023438` |
| GradPass | `3/3` |
| max grad relerr | `9.71e-05` |
| memory ratio max | `1.1615` |
| step ratio mean | `2.6711` |
| step ratio max | `2.7978` |
| S2 small-grid pass | `0/3` |
| `cache_basis_MB` | `0.0000` |
| `manual_cache_MB` | `0.953125` |

判断：

- `RR9` 保住了 `RR8/RR3 basis2` 的 teacher-free macro 与 GradPass。
- 相比 `RR8`，RR9 的 step mean 从 `2.8748` 降到 `2.6711`，说明 unroll 比 Python basis loop 更好。
- 但 memory max `1.1615` 仍高于 S2，且 step 仍远高于 `1.50`；不能记 success。

## 53. 追加自修复：RR10 recompute-z manual primitive

`RR9` 的 P10 attribution 仍显示 transformed `z` cache 是主要 live-set 之一，因此继续实现 `RR10-M13-DWM2LiteDense-rbf-basis2-recompute-z-manual-code-native`：forward 只缓存 `x/y`，backward 重算 transformed `z` 来计算 mix gradient。

实现位置：

| file | change |
|---|---|
| `experiments/run_gafu_v80_real.py` | 新增 `ManualDWM2LiteRBF2RecomputeZLayer` / `ManualDWM2LiteRBF2RecomputeZStack` |
| `experiments/run_gafu_v80_real.py` | 新增 `RR10` candidate、registry、manual train、gradient check、cache attribution |

Smoke 判断：

| item | value |
|---|---:|
| fake/proxy nonzero | `0` |
| RR10 GradPass rows | `3/3` |
| RR10 max relerr | `9.71e-05` |
| RR10 strict/manual contract | pass |

### 53.1 RR10 basis2 recompute-z 5-seed probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr10_basis2_recompute_z_probe_5seed_20260507T003000Z \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR10 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | external teacher | val acc | test acc | val gap vs MLP | dataset gaps |
|---|---:|---:|---:|---:|---|
| `B0` | 0 | `0.8443` | `0.7949` | `0.0000` | reference |
| `RR10 basis2` | 0 | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` |

Significance / Grad / efficiency：

| metric | value |
|---|---:|
| mean val gap | `+0.021094` |
| CI95 low | `+0.014714` |
| Holm p | `5.96e-05` |
| test gap | `+0.023438` |
| GradPass | `3/3` |
| max grad relerr | `9.71e-05` |
| memory ratio max | `1.1581` |
| step ratio mean | `3.4499` |
| step ratio max | `4.5112` |
| S2 small-grid pass | `0/3` |
| `cache_basis_MB` | `0.0000` |
| `manual_cache_MB` | `0.5078125` |

对比 RR8/RR9/RR10：

| candidate | macro gap | GradPass | memory max | step mean | step max | manual cache | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `RR8 basis2` | `+0.021094` | 1 | `1.1198` | `2.8748` | `2.9498` | `0.953125` | best memory among streaming variants |
| `RR9 basis2` | `+0.021094` | 1 | `1.1615` | `2.6711` | `2.7978` | `0.953125` | unroll improves step vs RR8 |
| `RR10 basis2` | `+0.021094` | 1 | `1.1581` | `3.4499` | `4.5112` | `0.5078125` | z-cache trim hurts step badly |

判断：

1. `RR10` 证明 transformed-z cache 可以删除，manual cache 从 `0.953125 MB` 降到 `0.5078125 MB`。
2. 但 peak memory 只从 `1.1615` 小幅降到 `1.1581`，说明剩余 peak 已主要来自 backward recompute temps / allocator peak / head temps，而不是常驻 manual cache alone。
3. step 明显恶化，说明 Python/Torch-level recompute 不适合作为 S2 解法。
4. 当前可测事实支持下一步做 fused/compiled RR8/RR9-style kernel，而不是继续在 Python manual path 中交换 cache 与 recompute。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v80_rr9_basis2_unrolled_probe_5seed_20260507T001500Z` | `911` | `0` |
| `v80_rr10_basis2_recompute_z_probe_5seed_20260507T003000Z` | `914` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `6005de2156d38f1fd328d3a78abc0be904c46b38a06dd7d56eb7d62097176c80` |
| RR9 `v80_route_decision.json` | `c9629c60d722ca70738b66b69f97e44d36a9aafaad70d340007a4d4f72fc519b` |
| RR9 `p9_task_summary.csv` | `bbdfca78c27e175350e1c549a3e5a81137f615ff505a2c74ffd69bfe474b9258` |
| RR9 `p1_significance_audit.csv` | `37f6689003decf672464ab8c0cf7fc7551ad1e1385ca8785cd260e421e62ea6c` |
| RR9 `p2_full_gradient_correctness.csv` | `20d59c0cf47f8d77a77a473e6a9d4f57f5be36e3ac6ffafdbe9e55c0202e2738` |
| RR9 `p10_efficiency_summary.csv` | `e451fb54f0965144fc422b04db08d5c794e86ec78948092e53ef3d258d935380` |
| RR9 `p10_efficiency_profiler.csv` | `cd334a750e2a4d5d61c2ca45499c51ffd754e884008ad149e0c897061201b66d` |
| RR9 `v80_provenance_audit.csv` | `79aed25f668976d1154ee7d4f9c29060dcc027d6aded401571b412d4f8530ae7` |
| RR10 `v80_route_decision.json` | `5d0d7e588d0b6a07759e2eada68e37b77afe606054efabd54f5cd0448faa1d51` |
| RR10 `p9_task_summary.csv` | `38b5be13aedc8c4a03e44a3d6db50c00bc55792def65611531c8d73ecebfea6f` |
| RR10 `p1_significance_audit.csv` | `1d4e4a37932c20c33f3dc7315f0f2f707db047d1f0d6ca0100e51bd356f2d7a0` |
| RR10 `p2_full_gradient_correctness.csv` | `373b64aa2d946daacb8197826a38497ec1c1f91015732c9777ad4cf76ffc65f3` |
| RR10 `p10_efficiency_summary.csv` | `b180966249f7f8109f414ee151bd8296c9ced71d81f4e979dbe278f35998ffda` |
| RR10 `p10_efficiency_profiler.csv` | `4636734550c331d7d97938921016c77a515f22e10f57352ce0d3b1388fa478fc` |
| RR10 `v80_provenance_audit.csv` | `4018c22cfbb20aba0b6958564888e985daceb5a67e165fb2d97c0c7260883d63` |

## 54. 当前最终收敛判断

截至 RR3/RR8/RR9/RR10 四个 code-native/manual primitive repairs，v8.0 仍没有完成全部实验，也没有达成 minimum success。

已真实闭合的部分：

```text
teacher-free macro gate: RR3/RR8/RR9/RR10 basis2 all pass in 5-seed small-grid
StrictPass: pass
GradPass: pass
external teacher used: 0
fake/proxy: 0
```

仍未闭合的部分：

```text
FullGridS2Pass: fail
S1Pass: fail
TimeAUCPass: fail/not closed
formal system closure: fail
```

机制结论：

1. 新 primitive 方向是有效的：RR3/RR8/RR9/RR10 都保住了 teacher-free macro gap `+0.021094` 级别。
2. memory blocker 已被真实定位：basis cache、z cache、backward recompute temp 都被逐项测过。
3. Python/Torch-level streaming/recompute 不能闭合 S2：memory 降幅有限，step 代价明显。
4. 当前不应继续 loss/classwise/teacher，也不应继续 Python manual cache 搬运；下一步必须实现 fused/compiled RBF2 residual primitive，至少把 RR8/RR9 的 basis/z/backward 计算合并为少量 kernel。

最新一句话：

> v8.0 还没有完成全部实验或达成 success，但已经从“teacher-free macro 缺口”推进到“teacher-free macro+Grad 已闭合，S2 卡在 primitive kernelization”。这是更好的失败：现在问题不再是有没有 autonomous advantage，而是能不能把 RR3/RR8/RR9 的 RBF2 residual primitive 做成真正 kernel-native。

## 55. 追加自修复：RR11/RR12 torch.compile compiled-helper probes

RR8/RR9/RR10 显示 Python/Torch-level streaming/recompute 无法闭合 S2 后，继续尝试 torch.compile 级别的 compiled helper，而不是继续改 loss 或 teacher。该尝试分两步：

| candidate | implementation | intended fix |
|---|---|---|
| `RR11` | compiled transform + compiled backward helper | 尝试将 RBF2 transform/backward elementwise 部分交给 TorchInductor |
| `RR12` | compiled transform only + eager verified backward | 保守测试 compiled forward 是否可用，同时保留 RR9 backward |

本地能力检查：

```text
torch = 2.4.1+cu124
cuda = true
torch.compile = true
triton = 3.0.0
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 55.1 RR11 compiled transform+backward smoke

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr11_basis2_compiled_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR11 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Gradient smoke：

| dataset | forward relerr | grad relerr | grad cos | GradPass |
|---|---:|---:|---:|---:|
| MNIST | `2.16e-07` | `1.9995` | `0.2793` | 0 |
| Fashion-MNIST | `6.29e-08` | `1.9998` | `0.8878` | 0 |
| KMNIST | `1.40e-07` | `1.9998` | `0.7520` | 0 |

判断：

- RR11 forward 数值接近 autograd reference，但 compiled backward helper 的 manual gradient 与 autograd reference 不一致。
- 因此 RR11 停止，不进入 5-seed task/efficiency official probe。

### 55.2 RR12 compiled-forward-only smoke

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr12_basis2_compiled_forward_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR12 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Gradient smoke：

| dataset | forward relerr | grad relerr | grad cos | GradPass |
|---|---:|---:|---:|---:|
| MNIST | `2.16e-07` | `2.0000` | `0.8379` | 0 |
| Fashion-MNIST | `6.29e-08` | `1.9999` | `0.9541` | 0 |
| KMNIST | `1.40e-07` | `1.9990` | `0.9618` | 0 |

判断：

- 即使只 compiled forward，manual gradient gate 仍失败。
- 该路径不能进入 official route；没有继续跑 5-seed task，避免把 GradFail candidate 包装成 success。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_rr11_basis2_compiled_smoke` | `198` | `0` |
| `tmp_v80_rr12_basis2_compiled_forward_smoke` | `201` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `84154b20dbe0bf85c4388fd067baf015e42ab0064df044d72703a87b954f69de` |
| RR11 `v80_route_decision.json` | `61b2881b725eddc6791ee7e9f1fde6154fa1ff5e87eb1a31ab3d2072d5c69ebc` |
| RR11 `p2_full_gradient_correctness.csv` | `716920f8353c8fe38887ccd9c2c620cdd253da927364195ca0ef05d915581b5f` |
| RR11 `v80_provenance_audit.csv` | `5c94fa98d73787c38e781a56a3ccadc4ac71d7812ea984bdf7e0d2df32dd4f9b` |
| RR12 `v80_route_decision.json` | `f3bdbd2aa8893f071513ca5f837bbddbe6f4192bd7f32f75b3f4b3a9a4c486f2` |
| RR12 `p2_full_gradient_correctness.csv` | `c11ca7df3dc1464360c5d97481cb78ecf1f28e535df665873c55e70403892017` |
| RR12 `v80_provenance_audit.csv` | `4a9936696735d081cc18490bfb0d5dd7adc625c1738764af699aad91facc9819` |

## 56. 最新执行结论

v8.0 仍未完成全部实验，也仍未达成 minimum success。

新增结论：

1. `RR11/RR12` 的 torch.compile helper 路线没有通过 gradient gate，不能进入 official probe。
2. 已验证可用的 RR 系列仍是 `RR3/RR8/RR9/RR10`，它们共同证明 teacher-free macro+Grad 可以闭合。
3. 但 S2 不能靠 Python/Torch eager streaming、recompute 或直接 torch.compile helper 闭合。

当前最硬 blocker：

```text
RR primitive needs a real fused custom kernel with verified manual gradient.
```

也就是说，下一步如果继续做 v8.0，不应再在 Python-level manual path 里微调；需要写真正的 Triton/CUDA forward-backward kernel，并先做 microkernel grad correctness，再进入 task/efficiency route。

## 57. 追加自修复：RR11/RR12 buffer-clone repair

第 55 节的 RR11/RR12 smoke 暴露了一个实现问题：compiled forward 数值接近 reference，但早期层 `mix` 梯度错误。随后做了最小定位，发现单层 compiled forward + eager backward 是一致的；全 stack 失败来自 compiled transform 的中间 `z` cache 可能复用内部 buffer。修复方式是：

```text
compiled forward cache:
  z.detach() -> z.detach().clone()
  y.detach() -> y.detach().clone()

autograd reference:
  compiled candidates 的 forward_with_params 也走同一个 compiled transform
```

这不是放宽 gate，也没有修改 pass 判据；只是让 cached manual path 与 autograd reference 使用同一 forward 数值路径，并确保缓存张量稳定。

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

### 57.1 buffer-clone smoke

RR11 smoke：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr11_buffer_clone_repair_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR11 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 1000
```

RR12 smoke：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr12_buffer_clone_repair_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR12 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 1000
```

Gradient smoke：

| candidate | pass rows | max grad relerr | grad cos min | fake/proxy |
|---|---:|---:|---:|---:|
| `RR11` | `3/3` | `7.50e-05` | `1.000001` | 0 |
| `RR12` | `3/3` | `7.50e-05` | `1.000001` | 0 |

判断：

- buffer-clone repair 修复了第 55 节的 GradFail。
- smoke task 只有 2 steps，不能用于 task success 结论；只用于确认梯度路径可进入正式 5-seed probe。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_rr11_buffer_clone_repair_smoke` | `201` | `0` |
| `tmp_v80_rr12_buffer_clone_repair_smoke` | `201` | `0` |

### 57.2 RR11/RR12 5-seed measured probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr11_rr12_buffer_clone_repair_probe_5seed_20260507T011500Z \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR11,RR12 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `RR11` | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` | 1 |
| `RR12` | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | route teacher-free macro pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RR11` | `+0.02109` | `+0.01497` | `1.19e-04` | `+0.02344` | `+0.00139` | `-0.09820` | 1 |
| `RR12` | `+0.02109` | `+0.01497` | `1.19e-04` | `+0.02344` | `+0.00139` | `-0.09820` | 1 |

说明：`p1_significance_audit.csv` 的旧字段 `macro_significant_task_pass` 仍继承早期更复杂口径，值为 `0`；v8.0 route 口径下的 `teacher_free_macro_pass=1` 来自预注册 official teacher-free macro gate（gap、CI、Holm、test consistency）。没有把旧字段手动改成 pass。

Gradient correctness：

| candidate | pass rows | max grad relerr | grad cos min |
|---|---:|---:|---:|
| `RR11` | `3/3` | `7.50e-05` | `1.000001` |
| `RR12` | `3/3` | `7.50e-05` | `1.000001` |

Efficiency（本 probe 只测 batch size 128）：

| candidate | memory ratio mean | step ratio mean | forward ratio | backward ratio | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `RR11` | `1.3762` | `2.5508` | `4.4997` | `1.6481` | `0/3` | FAIL |
| `RR12` | `1.0156` | `2.8807` | `4.6027` | `2.6966` | `0/3` | FAIL |

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "RR11",
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.02109375",
  "ci95_low": "0.014973958333333334",
  "holm_p": "0.00011921133993762768",
  "test_gap": "0.0234375",
  "memory_ratio_max": 2.0709068936527952,
  "step_ratio_max": 3.035317246814156,
  "primary_blocker": "fullgrid_s2_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

判断：

- RR11/RR12 的梯度问题已经修复，且它们保持了 RR9/RR10 的 teacher-free macro signal。
- torch.compile helper 没有解决 system blocker：RR12 memory mean 虽接近 S2，但 step mean `2.8807`，RR11 memory/step 都远离 S2。
- 因此 RR11/RR12 不能记为 v8.0 minimum success，也不值得扩成 full-grid/10-seed official route。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v80_rr11_rr12_buffer_clone_repair_probe_5seed_20260507T011500Z` | `1295` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `84fd42e3325d09485f6a8a42517ad3dd162eaeb245218f7926c8eb9d303dad61` |
| `v80_route_decision.json` | `9b41dc286005060d6c2dcca6e78477fda84c7fd8ca2350b7bda31d38a871e021` |
| `p9_task_summary.csv` | `b9b5f074e6631d9207e7d9200158d0c5497a81b9f3e5b21561ea54e2c74c46fc` |
| `p1_significance_audit.csv` | `096f2e2df8b3cecc8a07e7446f438ba6e275158d96174e511dcdbf7574b46053` |
| `p2_full_gradient_correctness.csv` | `fe7478df7228b461c46f2e9386f40fcae9c29fe019a050db0079716a13a47469` |
| `p10_efficiency_summary.csv` | `03f6aa33f030722b8097cda9884191f4bfb0c6c192aa31e16bf0fadfa870d1b6` |
| `v80_provenance_audit.csv` | `b802046cd8a986bc6f6932ea4689617fca895b6a9d7e0e305686de3491d2fcdd` |

## 58. 最新执行结论（二次更新）

v8.0 仍未完成全部实验，也仍未达成 minimum success。

本次新增结果把 RR11/RR12 的结论从“GradFail 停止”修正为更精确的 measured failure：

```text
RR11/RR12:
  StrictPass = true
  GradPass = true after buffer-clone repair
  TeacherFreeMacroPass = true in 5-seed batch128 probe
  FullGridS2Pass = false
  S2 small-grid shapes = 0/3
```

机制结论：

1. compiled helper 的原始失败是 cache stability 问题，不是 RR basis2 数学梯度公式失败。
2. 修复 cache 后，RR11/RR12 证明了 teacher-free macro + GradPass 可与 compiled helper 共存。
3. 但 `torch.compile` 没有带来 kernel-native S2：forward ratio 仍约 `4.5x`，step ratio 约 `2.55-2.88x`。
4. 当前 blocker 进一步收敛为真实 fused kernel，而不是 Python-level cache/compile helper：

```text
next_required_implementation =
  custom Triton/CUDA fused forward-backward microkernel
  with first-class gradient correctness test
```

因此，不能宣称 v8.0 全部实验完成，也不能宣称 system closure。当前可写的真实结论是：teacher-free code-native RR primitive 已经跨过 macro 与 GradPass，但所有 Python/Torch/compile helper 版本都没有进入 S2；下一步必须进入真正 custom fused kernel microbenchmark，而不是继续搬动 Python cache。

## 59. 追加自修复：RR13 Triton transform microkernel smoke

RR11/RR12 修复后仍未解决 S2，因此继续按计划推进到真正 custom kernel 方向。新增 `RR13`：

| candidate | implementation |
|---|---|
| `RR13` | basis2 DWM2Lite RBF stack，Triton forward transform + Triton dx + Triton grad-dw，head 与 mix GEMM 仍走现有 manual path |

实现说明：

- 新增真实 Triton kernels：`_rr_basis2_triton_forward_kernel`、`_rr_basis2_triton_dx_kernel`、`_rr_basis2_triton_grad_dw_kernel`。
- `RR13` 仍为 no-external-teacher official representation repair candidate。
- 没有使用 C3 teacher、loss/classwise/sampler/focal/margin。
- 该阶段只做 smoke；若 GradPass 不成立，不进入 5-seed task route。

代码检查：

```text
python -m py_compile experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v78_real.py experiments/run_gafu_v76_real.py experiments/run_gafu_v73_real.py
```

已通过。

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr13_triton_transform_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR13 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 1000
```

Gradient smoke：

| dataset | grad relerr max | grad cos | GradPass |
|---|---:|---:|---:|
| MNIST | `1.0281e-04` | `1.0000038` | 0 |
| Fashion-MNIST | `6.7043e-05` | `1.0000021` | 1 |
| KMNIST | `7.5032e-05` | `1.0000014` | 1 |

Efficiency smoke（batch 128 only）：

| candidate | memory ratio mean | step ratio mean | forward ratio | backward ratio | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `RR13` | `1.0368` | `4.1220` | `2.7265` | `5.6496` | `1/3` | FAIL |

Route：

```json
{
  "route": "R7-NoReproduction",
  "best_official_candidate_id": "RR13",
  "strict_pass": 1,
  "grad_pass": 0,
  "teacher_free_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "memory_ratio_max": 1.0367860085585945,
  "step_ratio_max": 10.575041300004877,
  "no_fake": true,
  "no_proxy": true
}
```

判断：

- RR13 是 v8.0 中第一个真实 Triton microkernel path，但 smoke 未通过 GradPass，不能进入 official 5-seed probe。
- GradFail 并不大：MNIST `1.0281e-04` 略高于 `1e-4` gate；但 gate 不能放宽，因此必须记为失败。
- Triton forward ratio 从 RR11/RR12 的约 `4.5x` 降到 `2.73x`，说明 custom kernel 方向有信号。
- 但 backward ratio 到 `5.65x`，step ratio `4.12x`，说明当前 RR13 只是 elementwise transform microkernel，不是完整 fused forward-backward；GEMM、grad-mix、kernel launch 与 backward reduction 仍未融合。
- 因此 RR13 不能作为 v8.0 success，也不能作为 S2 candidate。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_rr13_triton_transform_smoke` | `204` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `14b64cd11dcee02eb32edb529cfe0c574fb4245ea4a45ef1fabb019693d0d24a` |
| RR13 `v80_route_decision.json` | `f87bf2b1eef4d66e1ef57d4295943b94b32335a16c6c856bc86597e813088844` |
| RR13 `p2_full_gradient_correctness.csv` | `78e6b698c257e2eb1f8eea90b113f1a6328d772f5202025b757bee3fa10cf628` |
| RR13 `p10_efficiency_summary.csv` | `fc0a637197390757cfc66b290cda5cff7e2f44838e75b5e2f5adeff3c33bcbe0` |
| RR13 `v80_provenance_audit.csv` | `e71829b543ad0d40b47349fe73b61d14ee35ee992e04014f115b36a644a971f9` |

## 60. 最新执行结论（三次更新）

v8.0 仍未完成全部实验，也仍未达成 minimum success。

当前真实进展：

```text
RR11/RR12:
  buffer-clone repair 后 GradPass 成立；
  teacher-free macro pass 成立；
  S2 失败，step ratio 仍约 2.55-2.88。

RR13:
  真 Triton transform microkernel 已实现并真实跑通 smoke；
  GradPass 2/3，MNIST relerr 1.0281e-04 略超 gate；
  memory 接近 S2，但 step/backward 明显失败。
```

当前最准确的 blocker：

```text
not enough to fuse elementwise transform only;
need full fused transform + mix + grad-mix/backward reduction package.
```

也就是说，v8.0 的下一步不是继续调训练目标，也不是把 RR13 的 `1.028e-04` 四舍五入为 pass；应继续实现更完整的 fused kernel，或者先把 RR13 的 Triton backward reduction 数值误差压到 GradPass 内，再做 5-seed measured probe。

## 61. 追加自修复：RR14 Triton-backward diagnostic

RR13 的 GradFail 主要来自 Triton forward 与 eager autograd reference 的微小 `z` 差异，第一层 `mix` 梯度 relerr 刚超过 `1e-4`。为区分 “Triton backward 本身不可信” 与 “Triton forward/reference 数值差异导致 gate miss”，新增 `RR14`：

| candidate | implementation |
|---|---|
| `RR14` | eager basis2 forward + Triton dx / grad-dw transform-backward kernels |

该 candidate 是 diagnostic repair：它不能证明 Triton forward 已闭合，但可以测试 transform-backward Triton kernels 是否能通过独立 autograd reference。仍然 no-external-teacher，不使用 C3 teacher。

### 61.1 RR14 smoke

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/tmp_v80_rr14_triton_backward_smoke \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR14 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 1000
```

Gradient smoke：

| dataset | grad relerr max | grad cos | GradPass |
|---|---:|---:|---:|
| MNIST | `9.71e-05` | `1.0000038` | 1 |
| Fashion-MNIST | `5.42e-05` | `1.0000021` | 1 |
| KMNIST | `7.76e-05` | `1.0000014` | 1 |

Efficiency smoke（batch 128 only）：

| candidate | memory ratio mean | step ratio mean | forward ratio | backward ratio | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `RR14` | `1.1372` | `1.9802` | `3.9531` | `1.4471` | `0/3` | FAIL |

判断：

- RR14 证明 Triton transform-backward kernels 可以通过 GradPass。
- 但 eager forward 仍慢，memory 也高于 S2；smoke 不能作为 system success。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v80_rr14_triton_backward_smoke` | `207` | `0` |

### 61.2 RR14 5-seed measured probe

运行：

```bash
python experiments/run_gafu_v80_real.py \
  --out-dir results/real_rerun_20260506/v80_rr14_triton_backward_probe_5seed_20260507T020000Z \
  --fresh \
  --device auto \
  --basis-count 2 \
  --candidates B0,RR14 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 30 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Task summary：

| candidate | val acc | test acc | val gap vs MLP | dataset gaps | basic pass |
|---|---:|---:|---:|---|---:|
| `B0` | `0.8443` | `0.7949` | `0.0000` | F-MNIST `0.0000`, KMNIST `0.0000`, MNIST `0.0000` | reference |
| `RR14` | `0.8654` | `0.8184` | `+0.0211` | F-MNIST `+0.0176`, KMNIST `+0.0207`, MNIST `+0.0250` | 1 |

Significance：

| candidate | mean val gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | route teacher-free macro pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RR14` | `+0.02109` | `+0.01497` | `5.96e-05` | `+0.02344` | `+0.00139` | `-0.09820` | 1 |

Gradient correctness：

| candidate | pass rows | max grad relerr | grad cos min |
|---|---:|---:|---:|
| `RR14` | `3/3` | `9.71e-05` | `1.000001` |

Efficiency（本 probe 只测 batch size 128）：

| candidate | memory ratio mean | step ratio mean | forward ratio | backward ratio | S2 shapes | survivor |
|---|---:|---:|---:|---:|---:|---|
| `RR14` | `1.1372` | `3.3997` | `10.5658` | `1.7011` | `0/3` | FAIL |

Route：

```json
{
  "route": "R5-TeacherFreeNearPass",
  "best_official_candidate_id": "RR14",
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v80_minimum": 0,
  "macro_gap": "0.02109375",
  "ci95_low": "0.014973958333333334",
  "holm_p": "5.960566996881384e-05",
  "test_gap": "0.0234375",
  "memory_ratio_max": 1.1372298859740053,
  "step_ratio_max": 4.0014218154805,
  "primary_blocker": "fullgrid_s2_not_closed",
  "no_fake": true,
  "no_proxy": true
}
```

判断：

- RR14 保住了 RR 系列的 teacher-free macro signal，并通过 GradPass。
- 但它不是 S2 candidate：memory `1.1372 > 1.05`，step `3.3997 > 1.50`。
- RR14 说明 “只把 transform-backward 换成 Triton” 不足以解决 system closure；forward 和 update/kernel launch 仍是硬问题。

No-fake/no-proxy：

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v80_rr14_triton_backward_probe_5seed_20260507T020000Z` | `926` | `0` |

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `8616f996d8956682c244b6ddc2949dece0dd051a6b3bf34575e4aa49ba65419a` |
| RR14 smoke `v80_route_decision.json` | `5daa54071b2815ebbcf5012af9fbe24e161def6f26883267284d22e8e648ac8a` |
| RR14 smoke `p2_full_gradient_correctness.csv` | `55ad3f36fc2bc0c791a99775315132266f9c913d7147d0baff24ca80aa15c72e` |
| RR14 smoke `v80_provenance_audit.csv` | `2cc4bcd4fba9cb3b130efdd9bfb6c79983e9683e52d06e14cc48850e3f4b21ff` |
| RR14 5-seed `v80_route_decision.json` | `81fe98f718183e28636f97d179215acd43693439baef00a104bf4877e68c49b6` |
| RR14 5-seed `p9_task_summary.csv` | `a5bd5006465f167e2500bb81030241c56bd9fddfe332f85665864e6d12a0cb3e` |
| RR14 5-seed `p1_significance_audit.csv` | `72d6df9294352ac1deccd0adc7cd27ec543b8ff84daf041b2a55feb22a086ed8` |
| RR14 5-seed `p2_full_gradient_correctness.csv` | `55ad3f36fc2bc0c791a99775315132266f9c913d7147d0baff24ca80aa15c72e` |
| RR14 5-seed `p10_efficiency_summary.csv` | `d5f7220633e07ec153d44db22f97bee03b82a3945fec1879e251c4ed463b9a0d` |
| RR14 5-seed `v80_provenance_audit.csv` | `c77e5a1ccfc57fd034611f2949ed07cfff59f8b4a1b9389415e6003fe94298d8` |

## 62. 最新执行结论（四次更新）

v8.0 仍未完成全部实验，也仍未达成 minimum success。

本次新增结论：

```text
RR13:
  true Triton forward/backward transform path
  GradPass = false
  forward/memory 有信号，但 backward/step 失败

RR14:
  eager forward + Triton transform-backward path
  GradPass = true
  TeacherFreeMacroPass = true
  S2 = false
```

机制判断：

1. RR13 说明 Triton forward 数值路径需要更严格的 reference/gradient treatment；不能把 `1.028e-04` relerr 写成 pass。
2. RR14 说明 Triton backward transform kernel 本身可以过 GradPass。
3. 但 RR14 仍不解决 system：forward ratio 与 step ratio 都远高于 S2。
4. 当前必须做更完整的 fused package，至少把 `transform + mix + grad_mix` 合并；单独 fused transform 或单独 fused transform-backward 都不足以闭合。

因此，v8.0 当前真实状态是：

```text
Teacher-free macro/Grad: closed by RR basis2 family
S2 system closure: still open
No-fake/no-proxy: maintained
Minimum success: false
Formal success: false
```
