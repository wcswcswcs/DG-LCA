# DG-KAN v8.1 CE-Only Teacher-Free Architecture Matrix 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.1_CEOnly_TeacherFree_ArchitectureMatrix_完整实验计划.md` 的本轮真实执行结果。所有数值只来自落盘 CSV/JSON/manifest/log；没有使用 fake data、proxy rows、固定占位 ratio 或手填成功结论。v8.1 official route 只允许 CE-only、no external teacher、no self-teacher、no loss modification、strict PureKAN manual forward/backward/update。

## 1. 是否全部完成

没有全部完成。

本轮完成了：

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 CE-only contract runner | completed | 新增 `experiments/run_gafu_v81_real.py`，输出 CandidateSpecV3 / teacher contract / loss contract / strict contract |
| P1 baseline reproduction | completed | B0/M13/A2S/M9 10 seeds x 3 datasets |
| P3/P4 architecture matrix smoke | completed | B0/M13/A2S/M9/RR3/RR9/RR14 5 seeds, batch128 efficiency smoke |
| P6 official co-selection slice | completed | B0/M13/A2S/M9 10 seeds；RR14 单独 10-seed confirmation |
| failure self-repair | completed | RR14 hidden56 repair、RR14 default full-grid 5-seed、RR14 10-seed full-grid confirmation |

未完成或未实现：

| 阶段 | 状态 | 原因 |
|---|---|---|
| P2 validation robustness shards | not_run | validation shard runner 未实现，本轮不写 shard mean 或 split CI |
| P5 representation / hard-sample attribution | not_run | feature rank / margin / hard sample metrics 未实现 |
| P7 phase-clean profiler | not_run | train_step_only phase-clean profiler 未实现 |
| P8 time accounting | not_run | validation/logging/sync 分离未实现 |
| P9 phase-mapped kernel profiler | not_run | phase-to-kernel mapping 未实现 |
| P10 S1 system package | not_run | 没有 CE-only macro + S2 candidate 可进入 system repair |

## 2. 本轮代码

新增文件：

| 文件 | 作用 | SHA256 |
|---|---|---|
| `experiments/run_gafu_v80_real.py` | v8.0 measured path，本追加新增 `KC1-KC6` CE-only kernel-native candidates | `8187ffbc8eedd670624fdebc38187507350a08c634019c0a3001aaf267b6a117` |
| `experiments/run_gafu_v81_real.py` | v8.1 CE-only/no-teacher/no-loss contract wrapper，复用 v80 真实 measured path 并写 v8.1 artifacts | `3857dd439af155358bf9086d41e78071cd22d3cc4757e061d08835b4c69f3adc` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py
```

已通过。

说明：本轮没有把 v8.1 代码写入 `experiments/dgkan_core.py`。v8.1 的新增核心在 runner/contract 层，`dgkan_core.py` 当前工作区已有修改不是本轮 v8.1 的实现来源。

## 3. 主 run：P1/P6 10-seed CE-only co-selection

运行：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_archmatrix_p1_p6_10seed_20260507T210000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,A2S,M9 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route:

```json
{
  "route": "R7-NoCEOnlyArchitectureReproduction",
  "best_official_candidate_id": "M13",
  "teacher_free_ce_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v81_minimum": 0
}
```

Task / macro:

| candidate | val acc | test acc | macro val gap | CI95 low | Holm p | test gap | GradPass | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M13` | `0.862760` | `0.818750` | `+0.017839` | `+0.012240` | `7.25e-06` | `+0.022982` | 1 | `6/9` |
| `A2S` | `0.861914` | `0.815495` | `+0.016992` | `+0.011979` | `6.10e-06` | `+0.019727` | 1 | `2/9` |
| `M9` | `0.861914` | `0.815495` | `+0.016992` | `+0.011784` | `6.10e-06` | `+0.019727` | 1 | `6/9` |
| `B0` | `0.844922` | `0.795768` | `0.000000` | metric_unavailable | metric_unavailable | metric_unavailable | reference | reference |

Efficiency:

| candidate | memory max | step max | memory mean | step mean | FullGridS2 |
|---|---:|---:|---:|---:|---:|
| `M13` | `1.065687` | `1.486824` | `1.024738` | `1.357525` | 0 |
| `A2S` | `1.081641` | `1.685668` | `1.036812` | `1.528888` | 0 |
| `M9` | `1.065687` | `1.489537` | `1.024738` | `1.349936` | 0 |

判断：

- `M13` 复现 v7.8/v7.9 的 teacher-free positive signal，但没有过 `+0.0200` hard gate。
- `A2S/M9` 没有复现此前 5-seed structural screen 的 `+0.019921875`，10-seed 下回到 `+0.016992`。
- 主 run 没有 v8.1 minimum success。

No-fake audit：

```text
rows_checked = 152
fake_proxy_nonzero_count = 0
```

## 4. Architecture matrix smoke

运行：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_architecture_matrix_smoke_5seed_20260507T213000Z \
  --fresh \
  --device auto \
  --candidates B0,M13,A2S,M9,RR3,RR9,RR14 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128 \
  --bench-warmup 10 \
  --bench-reps 50 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route:

```json
{
  "route": "R4-CEOnlyTeacherFreeMacroButS2Fail",
  "best_official_candidate_id": "RR3",
  "teacher_free_ce_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v81_minimum": 0
}
```

Task:

| candidate | macro val gap | CI95 low | Holm p | test gap | GradPass | efficiency note |
|---|---:|---:|---:|---:|---:|---|
| `RR3` | `+0.021224` | `+0.014844` | `3.39e-04` | `+0.023307` | 1 | batch128 memory mean `1.689871`, step mean `2.543552` |
| `RR14` | `+0.021094` | `+0.014974` | `3.43e-04` | `+0.023438` | 1 | batch128 memory mean `1.137230`, step mean `2.038194` |
| `RR9` | `+0.021094` | `+0.014844` | `3.43e-04` | `+0.023438` | 1 | batch128 memory mean `1.161470`, step mean `2.474804` |
| `A2S` | `+0.019922` | `+0.012500` | `3.30e-03` | `+0.023307` | 1 | batch128 near-pass only |
| `M9` | `+0.019922` | `+0.012500` | `3.30e-03` | `+0.023307` | 1 | batch128 near-pass only |
| `M13` | `+0.019271` | `+0.012497` | `4.72e-03` | `+0.027474` | 1 | batch128 near-pass only |

判断：

- RR3/RR9/RR14 在 5-seed smoke 中证明 CE-only architecture primitive 可以跨过 macro gate。
- 但它们都没有 S2，且该 run 只测 batch128，不允许写 FullGridS2Pass。
- RR14 是系统上相对最接近的 RR candidate，因此进入后续 repair/confirmation。

No-fake audit：

```text
rows_checked = 172
fake_proxy_nonzero_count = 0
```

## 5. 失败后自修复

### 5.1 RR14 hidden56 S2 repair

运行：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_rr14_hidden56_s2_repair_5seed_20260507T220000Z \
  --fresh \
  --device auto \
  --hidden-dim 56 \
  --candidates B0,RR14 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Result:

| candidate | macro val gap | CI95 low | Holm p | test gap | memory max | step max | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RR14 hidden56` | `+0.013542` | `+0.007031` | `5.46e-03` | `+0.021745` | `1.611974` | `2.652256` | 0 |

判断：hidden56 同时损伤 macro 且没有解决 S2。停止该方向。

No-fake audit：

```text
rows_checked = 92
fake_proxy_nonzero_count = 0
```

### 5.2 RR14 default full-grid 5-seed

运行：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_rr14_default_fullgrid_5seed_20260507T223000Z \
  --fresh \
  --device auto \
  --candidates B0,RR14 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Result:

| candidate | macro val gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RR14` | `+0.021094` | `+0.014974` | `5.96e-05` | `+0.023438` | `1.614004` | `2.738844` | `0/9` |

判断：RR14 5-seed 可以过 CE-only macro gate，但系统效率远离 S2。

No-fake audit：

```text
rows_checked = 91
fake_proxy_nonzero_count = 0
```

### 5.3 RR14 10-seed full-grid confirmation

运行：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_rr14_quality_fullgrid_confirm_10seed_20260507T230000Z \
  --fresh \
  --device auto \
  --candidates B0,RR14 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route:

```json
{
  "route": "R7-NoCEOnlyArchitectureReproduction",
  "best_official_candidate_id": "RR14",
  "teacher_free_ce_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v81_minimum": 0
}
```

Result:

| candidate | val acc | test acc | macro val gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RR14` | `0.864323` | `0.815690` | `+0.019401` | `+0.015039` | `6.51e-09` | `+0.019922` | `6/6` | `1.614004` | `2.777986` | `0/9` |

Gradient max relerr:

```text
max grad_relerr = 9.710068843560293e-05
```

判断：

- RR14 的 5-seed macro pass 没有扩展为 10-seed official macro pass。
- RR14 仍然显著优于 MLP，但未过 `+0.0200` hard gate。
- S2 明显失败，不能作为 v8.1 minimum success。

No-fake audit：

```text
rows_checked = 95
fake_proxy_nonzero_count = 0
```

## 6. Artifact / Contract 状态

每个 v8.1 run 都落盘以下关键 artifact：

```text
candidate_registry_v3.csv
candidate_factory_audit_v81.csv
teacher_contract_no_teacher.csv
loss_contract_ce_only.csv
strict_purekan_contract.csv
contract_validator_v81.csv
artifact_join_coverage_v81.csv
p1_ce_only_reproduction.csv
p2_validation_robustness.csv
p3_architecture_implementation_matrix.csv
p4_architecture_task_matrix.csv
p5_representation_hardsample_attribution.csv
p6_official_coselection.csv
p7_phase_clean_fullgrid.csv
p8_time_accounting.csv
p9_phase_mapped_kernel_profiler.csv
p10_s1_memory_system_package.csv
v81_route_decision.json
v81_provenance_audit.csv
```

未实现阶段均写为 `not_run` 或 `not_implemented`，没有写入 fake pass。

## 7. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `8187ffbc8eedd670624fdebc38187507350a08c634019c0a3001aaf267b6a117` |
| `experiments/run_gafu_v81_real.py` | `3857dd439af155358bf9086d41e78071cd22d3cc4757e061d08835b4c69f3adc` |
| v8.1 plan | `d966723a2cbab5fc40fa431875767ead2318aa048ace3d65251c446c7f27ca5c` |
| main10 route | `de18eccb30a9cd1581f4a386674f88f48450f26e886e01b23f372425792292d8` |
| architecture smoke route | `e9b22da356b20471300e0f2b78f8d85a5e37bb8c70b49d8b8e98bcac5809659f` |
| RR14 fullgrid 5seed route | `17f5373ca8e41b10beb0f3fa2c2bda7791ed09e846c49c1525647f92e316e126` |
| RR14 fullgrid 10seed route | `0830e858ef951fd4e4926f2dcdd398e56eb28e7c7734a5fbd35cc9cae30246ae` |

## 8. 最终结论

v8.1 本轮未达成 minimum success：

```text
CodeNativePass = true
NoTeacherNoLossModificationPass = true
StrictPass = true
GradPass = true
TeacherFreeCEMacroPass = false for 10-seed official confirmation
FullGridS2Pass = false
success_v81_minimum = false
```

机制结论：

1. v8.1 成功建立了 CE-only/no-teacher/no-loss contract runner，所有 official candidates 均通过 no-teacher/no-loss 审计。
2. M13 10-seed 复现为 `+0.01784`，仍是稳定 near-pass，但不是 official success。
3. A2S/M9 在本轮 10-seed 中没有复现此前 5-seed `+0.01992`，说明 near-pass 不够稳。
4. RR3/RR9/RR14 在 5-seed architecture smoke 中能跨过 CE-only macro gate，证明结构 primitive 方向确实能提供 margin。
5. RR14 作为最接近可 kernelize 的候选，在 10-seed full-grid confirmation 中只到 `+0.01940`，且 S2 `0/9`。
6. hidden56 不是解法：它既没有修 S2，也明显伤 task。
7. 当前最准确 blocker 是：CE-only architecture margin 已经局部出现，但 10-seed 稳定性与 FullGridS2 没有同时闭合。

最终一句话：

> v8.1 没有证明 PureKAN-NG architecture-only official success；它证明了 CE-only/no-teacher 结构候选已经能在 5-seed 中跨过 +2% gate，但最强候选 RR14 在 10-seed 下回落到 +1.94%，并且 full-grid S2 明显失败。下一步应继续做 CE-only architecture primitive 的系统化实现，而不是回到 teacher/self-teacher/loss 修补。

## 9. 追加：CE-only kernel-native S2 repair

用户要求继续做真正的 CE-only architecture primitive / kernel-native S2 repair。本追加只做代码路径与系统结构修复：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
distill_loss_used = 0
special_loss_used = 0
fake/proxy = 0
```

代码改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v80_real.py` | 新增 `KC1-KC6` CE-only kernel-native candidates；复用真实 recompute / packed / prefix-recompute stack，实现训练、梯度检查、registry、route metadata |
| `experiments/run_gafu_v81_real.py` | 将 `KC1-KC6` 纳入 v8.1 official CE-only architecture/kernel-native contract |

说明：本次没有改 `experiments/dgkan_core.py`。原因是可测 kernel/cache primitive 已在 v76/v80 runner 中实现；本次修复是把这些 primitive 从 `SYS*` self-bootstrap 路线中剥离出来，注册为真正 CE-only/no-teacher 的 `KC*` official architecture path。

代码检查：

```text
python -m py_compile experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py
```

已通过。

### 9.1 新增候选

| candidate | 含义 | teacher | loss | 目的 |
|---|---|---:|---|---|
| `KC1` | recompute fused linear stack + packed head | 0 | CE | 去掉 hidden-y cache，保留默认 AdamW |
| `KC2` | KC1 + AdamW second moment BF16 path | 0 | CE | 进一步压 optimizer/live-set |
| `KC3` | packed recompute stack + AdamW second moment BF16 path | 0 | CE | flat owner + recompute |
| `KC4` | packed cache-y stable stack + AdamW second moment BF16 path | 0 | CE | 更快 cache-y path |
| `KC5` | packed prefix-recompute + AdamW second moment BF16 path | 0 | CE | 减少 backward hidden temps |
| `KC6` | packed prefix-recompute + default AdamW | 0 | CE | 保 task，同时继承 prefix-recompute S2 margin |

### 9.2 Smoke

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/tmp_v81_kc_s2_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,KC1,KC2,KC3,KC4,KC5 \
  --seeds 0 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 20 \
  --bootstrap-reps 200
```

结果：

```text
KC1-KC5 training path works
KC1-KC5 GradPass smoke = 1
KC1-KC5 bs128 S2 smoke = 1
v81_provenance_audit rows_checked = 130
fake_proxy_nonzero_count = 0
```

Smoke 只用于代码路径验证，不作为正式成功结论。

### 9.3 KC1-KC5 5-seed full-grid matrix

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_kernel_native_s2_repair_5seed_20260507T234500Z \
  --fresh \
  --device auto \
  --candidates B0,KC1,KC2,KC3,KC4,KC5 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

Route：

```text
route = R4-CEOnlyTeacherFreeMacroButS2Fail
best = KC1
CodeNativePass = 1
NoTeacherNoLossPass = 1
StrictPass = 1
GradPass = 1
TeacherFreeCEMacroPass = 1
FullGridS2Pass = 0
success_v81_minimum = 0
```

Summary：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KC1` | `+0.021484` | `+0.014323` | `1.24e-03` | `+0.023698` | 1 | `1.059384` | `1.365611` | `6/9` | macro pass, bs512 memory fail |
| `KC2` | `+0.019141` | `+0.011195` | `6.30e-03` | `+0.025521` | 1 | `1.053697` | `1.434706` | `6/9` | macro fail, bs512 memory fail |
| `KC3` | `+0.020833` | `+0.011719` | `9.00e-03` | `+0.020833` | 1 | `1.072605` | `1.300654` | `6/9` | macro pass, bs512 memory fail |
| `KC4` | `+0.018750` | `+0.013542` | `2.19e-04` | `+0.025260` | 1 | `1.053697` | `1.259031` | `6/9` | macro fail, bs512 memory fail |
| `KC5` | `+0.016536` | `+0.010286` | `6.30e-03` | `+0.018620` | 1 | `1.047394` | `1.375632` | `9/9` | S2 pass, macro fail |

判断：

- `KC1/KC3` 证明 CE-only kernel-native path 可以跨过 macro gate。
- `KC5` 证明 prefix-recompute 可以进入 FullGridS2。
- 但没有候选同时满足 macro gate 和 FullGridS2。
- S2 fail 全部集中在 bs512 memory，step 对 `KC1/KC3/KC4/KC5` 已不再是主要 blocker。

No-fake audit：

```text
rows_checked = 212
fake_proxy_nonzero_count = 0
```

### 9.4 Hidden60 S2 repair

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_kernel_native_hidden60_s2_repair_5seed_20260508T001500Z \
  --fresh \
  --device auto \
  --hidden-dim 60 \
  --candidates B0,KC1,KC2,KC3 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KC1 hidden60` | `+0.021745` | `+0.014193` | `9.81e-04` | `+0.023438` | 1 | `1.057424` | `1.408159` | `6/9` | macro pass, memory still fail |
| `KC2 hidden60` | `+0.019401` | `+0.012760` | `7.99e-04` | `+0.020833` | 1 | `1.052114` | `1.475221` | `6/9` | macro fail, memory still fail |
| `KC3 hidden60` | `+0.021484` | `+0.012240` | `2.93e-03` | `+0.023047` | 1 | `1.069700` | `1.354744` | `6/9` | macro pass, memory worse |

判断：hidden60 没有闭合 S2；memory gap 只略微收缩。

No-fake audit：

```text
rows_checked = 158
fake_proxy_nonzero_count = 0
```

### 9.5 Basis4 S2 repair

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_kernel_native_basis4_s2_repair_5seed_20260508T003000Z \
  --fresh \
  --device auto \
  --basis-count 4 \
  --candidates B0,KC1,KC2 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `KC1 basis4` | `+0.021484` | `+0.014323` | `5.23e-04` | `+0.023698` | 1 | `1.059384` | `2.079101` | `5/9` | memory 不变，step 变差 |
| `KC2 basis4` | `+0.019141` | `+0.011195` | `2.45e-03` | `+0.025521` | 1 | `1.053697` | `2.046668` | `2/9` | macro fail, step 变差 |

判断：head basis 不是 S2 memory source；basis4 反而引入 step instability。

No-fake audit：

```text
rows_checked = 131
fake_proxy_nonzero_count = 0
```

### 9.6 KC6 prefix-recompute default AdamW

`KC5` 的 FullGridS2 成立但 macro 失败，因此追加 `KC6`：同样 prefix-recompute stack，但不用 BF16 second-moment update，恢复默认 AdamW，以检查 task 下降是否来自 update path。

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_kc6_prefix_recompute_fullgrid_5seed_20260508T010000Z \
  --fresh \
  --device auto \
  --candidates B0,KC6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

结果：

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KC6` | `+0.024349` | `+0.015234` | `3.89e-04` | `+0.023177` | 1 | `1.053081` | `1.338148` | `6/9` |

Per-shape：

| batch | memory ratio | step ratio range | S2 |
|---:|---:|---:|---:|
| 128 | `0.995003` | `0.8696-1.1590` | `3/3` |
| 256 | `1.008596` | `1.1562-1.1673` | `3/3` |
| 512 | `1.053081` | `1.1468-1.3381` | `0/3` |

判断：

- `KC6` 是本轮最强 CE-only quality candidate：5-seed macro `+0.02435`。
- Step 已进入 S2/S1 step 区间，S2 只差 bs512 memory `+0.003081`。
- 这说明 `KC5` 的 macro 下降主要不是 prefix-recompute stack 本身，而更可能来自 BF16 update path。

No-fake audit：

```text
rows_checked = 106
fake_proxy_nonzero_count = 0
```

### 9.7 KC6 hidden60 / hidden58

KC6 默认只差 `0.003081` memory，因此继续做 very narrow hidden repair。

| run | candidate | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `v81_ceonly_kc6_hidden60_prefix_recompute_fullgrid_5seed_20260508T011500Z` | `KC6 hidden60` | `+0.023047` | `+0.015361` | `2.03e-04` | `+0.017839` | `1.051299` | `1.345855` | `6/9` | macro pass, memory still +0.001299 over S2 |
| `v81_ceonly_kc6_hidden58_prefix_recompute_fullgrid_5seed_20260508T013000Z` | `KC6 hidden58` | `+0.018750` | `+0.012109` | `3.85e-04` | `+0.020964` | `1.050443` | `1.328691` | `6/9` | memory still +0.000443 over S2, macro fails |

No-fake audit：

| run | rows_checked | fake/proxy nonzero |
|---|---:|---:|
| KC6 hidden60 | `106` | `0` |
| KC6 hidden58 | `107` | `0` |

### 9.8 追加后的判断

本追加没有达成 v8.1 minimum success：

```text
Best quality candidate = KC6
TeacherFreeCEMacroPass = true
GradPass = true
StrictPass = true
NoTeacherNoLossPass = true
FullGridS2Pass = false
success_v81_minimum = false
```

但 blocker 已经比 RR14 阶段清楚很多：

1. `KC1/KC3/KC6` 证明 CE-only/no-teacher kernel-native stack 可以真实跨过 macro gate。
2. `KC6` 将 step ratio 修到 S2 内，macro 甚至达到 `+0.02435`，但 bs512 memory 仍为 `1.05308`。
3. `KC6 hidden60` 保住 macro，memory 到 `1.05130`，仍未过 S2。
4. `KC6 hidden58` 几乎碰到 memory gate，但 macro 退回 `+0.01875`，说明继续压 hidden 会伤表达力。
5. `KC5` 说明 FullGridS2 可达，但 BF16 update path 会伤 macro。
6. basis4 说明 head basis 不是 S2 memory 主因；当前 memory source 主要仍是 root/checkpoint live-set 与 prefix backward temps。

更新后的最终一句话：

> v8.1 追加实现了真正 CE-only kernel-native repair path：`KC6` 已经把 architecture-only macro 推到 `+0.02435`，step 也闭合，但 bs512 memory 仍卡在 `1.05308`，差 `0.00308` 到 S2。v8.1 仍不能记 minimum success；下一步应做真正 root-input lifetime/streaming cache 或 allocator-level memory repair，而不是再压 hidden/basis 或回到 teacher/loss。

## 10. 追加：GPU-native allocator / lifetime repair

> 约束修正：不允许 CPU offload。曾临时落盘的 CPU-root / CPU optimizer-state diagnostic run 不进入 route、不作为 repair 结论；相关候选已从 `run_gafu_v80_real.py` / `run_gafu_v81_real.py` 的 v8.1 candidate registry 中移除。本节只记录 GPU-native、CE-only、no-teacher、no-loss-modification 候选。

### 10.1 代码改动

本轮只保留两个 GPU-native 修复候选：

| candidate | 改动 | CPU offload | teacher/loss |
|---|---|---:|---|
| `KC7` | packed-prefix-recompute + FP32 AdamW `addcdiv_` in-place update，避免额外 FP32 step tensor | 0 | CE-only / no teacher |
| `KC8` | `KC7` + head-cache 在 stack backward 前释放，测试 GPU lifetime overlap | 0 | CE-only / no teacher |

代码检查：

```text
python -m py_compile experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

关键 hash：

| file | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `b02fb6d9b651c81ac3498443d969679e603c80b0fe301974d2890d65d20c45a9` |
| `experiments/run_gafu_v81_real.py` | `ed51f2f24c22a1316a9bfc9d23415e651df5fec3c67da423a4c947c9942db10f` |
| `experiments/run_gafu_v72_real.py` | `cb7b79622d5a12ecece3a4ca68e66b369d916fdccf85ed9989f17f2bf711615d` |
| `experiments/run_gafu_v71_real.py` | `4311c0e075661eca6f967bad1da4e470648f5d55e276dcea9b9d4cb70ab6b0a8` |

### 10.2 Smoke：KC7 / KC8

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/tmp_v81_kc7_kc8_gpu_native_smoke \
  --fresh \
  --device auto \
  --candidates B0,KC7,KC8 \
  --datasets MNIST \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

| candidate | GradPass | memory ratio | step ratio | backward ratio | S2 smoke | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `KC7` | 1 | `0.9846` | `1.2853` | `1.3626` | 1 | 可进入 full-grid |
| `KC8` | 1 | `0.9840` | `10.0817` | `16.8720` | 0 | head-cache release 触发严重 step fail，停止 |

No-fake audit：

```text
rows_checked = 100
fake_proxy_nonzero_count = 0
```

### 10.3 KC7 full-grid 5-seed

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_gpu_allocator_addcdiv_repair_5seed_20260508T030000Z \
  --fresh \
  --device auto \
  --candidates B0,KC7 \
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
  "route": "R7-NoCEOnlyArchitectureReproduction",
  "best_official_candidate_id": "KC7",
  "teacher_free_ce_macro_pass": 0,
  "fullgrid_s2_pass": 0,
  "success_v81_minimum": 0,
  "macro_gap": "0.01796875",
  "ci95_low": "0.01171875",
  "holm_p": "0.0008059289572520373",
  "test_gap": "0.021614583333333333",
  "memory_ratio_max": 1.0530812221483614,
  "step_ratio_max": 1.363326880857917,
  "S2_shape_count": 6,
  "no_fake": true,
  "no_proxy": true
}
```

Task / gradient / efficiency：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory max | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KC7` | `+0.017969` | `+0.011719` | `8.06e-04` | `+0.021615` | `-0.001118` | `-0.097738` | `6/6` | `8.05e-05` | `1.053081` | `1.363327` | `6/9` |

Per-shape efficiency：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.9846` | `1.3566` | 1 |
| MNIST | 256 | `1.0086` | `1.3446` | 1 |
| MNIST | 512 | `1.0531` | `1.3557` | 0 |
| Fashion-MNIST | 128 | `0.9846` | `1.3609` | 1 |
| Fashion-MNIST | 256 | `1.0086` | `1.3497` | 1 |
| Fashion-MNIST | 512 | `1.0531` | `1.3454` | 0 |
| KMNIST | 128 | `0.9846` | `1.3633` | 1 |
| KMNIST | 256 | `1.0086` | `1.3581` | 1 |
| KMNIST | 512 | `1.0531` | `1.3570` | 0 |

No-fake audit：

```text
rows_checked = 111
fake_proxy_nonzero_count = 0
```

### 10.4 判断

本轮 GPU-native repair 没有达成 v8.1 minimum success：

```text
KC7:
  StrictPass = true
  GradPass = true
  NoTeacherNoLossPass = true
  TeacherFreeCEMacroPass = false
  FullGridS2Pass = false
```

机制结论：

1. `KC7` 的 in-place FP32 `addcdiv_` update 确实让 bs128 memory 进入 S1 memory 区间，step 也稳定在 S2 内。
2. 但 `KC7` 没有保住 `KC6` 的 macro advantage：本轮 macro gap 只有 `+0.01797`，低于 `+0.0200` hard gate。
3. `KC7` 没有修掉 bs512 memory：bs512 memory 仍是 `1.05308`，与 `KC6` 的 blocker 同形。
4. 因此 `KC6` 的 bs512 memory miss 不能归因于 FP32 AdamW step tensor；主要仍指向 root/checkpoint live-set、backward recompute temps 或 allocator baseline。
5. `KC8` 的 head-cache lifetime release 在 smoke 中 step ratio `10.08`，不能进入 full-grid；这个方向当前不可用。
6. CPU offload 不允许，且不作为本轮 repair 结论。

更新后的最终一句话：

> v8.1 的 GPU-native allocator repair 仍未闭合：`KC7` 不用 CPU、不改 teacher/loss，并通过 GradPass，但它既没有跨过 macro gate，也没有解决 bs512 memory `1.05308`。当前应停止 CPU/offload 方向；下一步若继续，应做真正 GPU-side live-set/allocator instrumentation 或 fused backward，先确认 bs512 peak 的真实 tensor source，再改 kernel。

## 11. 追加：GPU-side live-set / allocator instrumentation 与 fused backward

> 本节继续遵守 v8.1 约束：CE-only、teacher-free、no special loss、no sampler/class weight、no CPU offload。新增 artifact `p11_gpu_live_allocator_v81.csv` 直接记录 CUDA allocated/reserved peak、phase start allocation、manual cache、optimizer state 与 fused backward 调用数；未使用 fake/proxy。

### 11.1 代码改动

新增 GPU-only fused backward 候选：

| candidate | backward 改动 | update | CPU offload |
|---|---|---|---:|
| `KF1` | Triton in-place SiLU derivative backward kernel | FP32 AdamW | 0 |
| `KF2` | Triton in-place SiLU derivative backward kernel | FP32 AdamW `addcdiv_` | 0 |
| `KF3` | ATen fused `silu_backward` GPU op | FP32 AdamW | 0 |
| `KF4` | ATen fused `silu_backward` GPU op | FP32 AdamW `addcdiv_` | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

关键 hash：

| file | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `af1f5686e44e117b719755dfcc0e1a52fd93c4f7e2793bf2132f70b927bb5c6b` |
| `experiments/run_gafu_v81_real.py` | `61410914379e5c0ec9b80264bf4ff1a39a1ae457e008ad5e171d260e14b9b967` |

### 11.2 Triton fused backward：KF1 / KF2

Smoke run：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/tmp_v81_kf1_kf2_gpu_fused_backward_smoke \
  --fresh \
  --device auto \
  --candidates B0,KF1,KF2 \
  --datasets MNIST \
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

| candidate | GradPass | max relerr | memory ratio | step ratio | fused calls | fallback |
|---|---:|---:|---:|---:|---:|---:|
| `KF1` | 1 | `9.33e-05` | `0.9950` | `1.3438` | `14` | `0` |
| `KF2` | 1 | `9.33e-05` | `0.9829` | `1.3135` | `14` | `0` |

Full-grid run：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_gpu_liveset_fused_backward_5seed_20260508T040000Z \
  --fresh \
  --device auto \
  --candidates B0,KF1,KF2 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | macro gap | CI95 low | test gap | GradPass | max relerr | memory max | step max | S2 shapes | fused calls | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KF1` | `+0.019401` | `+0.011589` | `+0.024219` | `5/6` | `1.0136e-04` | `1.046778` | `1.776738` | `8/9` | `240/shape` | `0` |
| `KF2` | `+0.022786` | `+0.017188` | `+0.022917` | `5/6` | `1.0136e-04` | `1.046778` | `2.218698` | `7/9` | `240/shape` | `0` |

判断：

- Triton fused kernel 确实被调用，且没有 fallback。
- `KF2` macro 过线，但不能记成功：GradPass 只有 `5/6`，step max `2.2187`，FullGridS2 失败。
- `KF1` 接近 macro，但同样 GradPass 不完整，MNIST bs128 step `1.7767` 失败。
- 因此 Triton in-place SiLU backward 是真实 GPU fused path，但当前数值/step 不满足 official route。

### 11.3 ATen fused backward：KF3 / KF4

Smoke run：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/tmp_v81_kf3_kf4_aten_fused_backward_smoke \
  --fresh \
  --device auto \
  --candidates B0,KF3,KF4 \
  --datasets MNIST \
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

| candidate | GradPass | max relerr | memory ratio | step ratio | ATen fused calls |
|---|---:|---:|---:|---:|---:|
| `KF3` | 1 | `6.72e-05` | `0.9950` | `1.6431` | `14` |
| `KF4` | 1 | `6.72e-05` | `0.9829` | `1.5933` | `14` |

Full-grid run：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_gpu_aten_fused_backward_5seed_20260508T043000Z \
  --fresh \
  --device auto \
  --candidates B0,KF3,KF4 \
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
  "route": "R7-NoCEOnlyArchitectureReproduction",
  "best_official_candidate_id": "KF4",
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_ce_macro_pass": 0,
  "fullgrid_s2_pass": 1,
  "success_v81_minimum": 0,
  "macro_gap": "0.018359375",
  "ci95_low": "0.012109375",
  "holm_p": "0.0006369078466411447",
  "test_gap": "0.027734375",
  "memory_ratio_max": 1.0467784425240663,
  "step_ratio_max": 1.3119880822477525,
  "S2_shape_count": 9,
  "no_fake": true,
  "no_proxy": true
}
```

Task / gradient / efficiency：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory max | step max | S2 shapes | S1 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KF3` | `+0.017839` | `+0.008724` | `1.41e-02` | `+0.017969` | `+0.004777` | `-0.092685` | `6/6` | `8.05e-05` | `1.046778` | `1.339391` | `9/9` | `3/9` |
| `KF4` | `+0.018359` | `+0.012109` | `6.37e-04` | `+0.027734` | `-0.000434` | `-0.091186` | `6/6` | `8.05e-05` | `1.046778` | `1.311988` | `9/9` | `3/9` |

`KF4` per-shape efficiency：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `0.9829` | `1.0777` | 1 |
| MNIST | 256 | `1.0053` | `1.1183` | 1 |
| MNIST | 512 | `1.0468` | `1.1314` | 1 |
| Fashion-MNIST | 128 | `0.9829` | `1.2146` | 1 |
| Fashion-MNIST | 256 | `1.0053` | `1.1891` | 1 |
| Fashion-MNIST | 512 | `1.0468` | `1.1960` | 1 |
| KMNIST | 128 | `0.9829` | `1.1397` | 1 |
| KMNIST | 256 | `1.0053` | `1.3120` | 1 |
| KMNIST | 512 | `1.0468` | `1.2611` | 1 |

### 11.4 GPU live-set / allocator attribution

`p11_gpu_live_allocator_v81.csv` 使用 CUDA peak stats 分 phase 记录，不是 op-count proxy。最终 ATen full-grid run 中：

| candidate | batch | top peak phase | mean/top example | CPU offload |
|---|---:|---|---:|---:|
| `B0` | 128 | forward | `17.9434 MB` | 0 |
| `B0` | 256 | forward | `18.4258 MB` | 0 |
| `B0` | 512 | forward | `19.3906 MB` | 0 |
| `KF4` | 128 | update | `18.4390 MB` | 0 |
| `KF4` | 256 | update | `19.2778 MB` | 0 |
| `KF4` | 512 | update | `20.9556 MB` | 0 |

MNIST bs512 `KF4` phase peaks：

| phase/source | measured MB |
|---|---:|
| forward peak | `19.3145` |
| head backward peak | `19.8540` |
| stack backward peak | `20.7603` |
| update peak | `20.9556` |
| root input cache | `1.5313` |
| manual cache total | `1.5313` |
| hidden-y cache | `0.0000` |
| optimizer state | `0.4521` |

判断：

- `KF4` 已把 FullGridS2 闭合：memory max `1.04678 <= 1.05`，step max `1.31199 <= 1.50`。
- 但 `KF4` 不是 v8.1 minimum success，因为 macro gap 只有 `+0.01836`，没有过 `+0.0200` hard gate。
- GPU live-set 显示 `KF4` 的 peak 主要在 update phase，而不是 forward；bs512 stack backward 也接近 update peak。
- root input cache 已经是主要可见 cache source，hidden-y cache 为 `0`，说明 prefix/recompute 已经生效。
- optimizer state 只有约 `0.4521 MB`，单独小于 bs512 total gap；当前 S1 仍失败，不能通过 CPU offload 解决，也不允许走 CPU offload。

### 11.5 No-Fake / No-Proxy

| run | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `tmp_v81_kf1_kf2_gpu_fused_backward_smoke` | `107` | `0` |
| `v81_ceonly_gpu_liveset_fused_backward_5seed_20260508T040000Z` | `170` | `0` |
| `tmp_v81_kf3_kf4_aten_fused_backward_smoke` | `111` | `0` |
| `v81_ceonly_gpu_aten_fused_backward_5seed_20260508T043000Z` | `172` | `0` |

关键 hash（ATen full-grid run）：

| file | SHA256 |
|---|---|
| `v81_route_decision.json` | `2e74b599ffccadd2f4965bbc85c56eb16ceb6a17cb70cc29000e328a5cf09873` |
| `p11_gpu_live_allocator_v81.csv` | `27c84d9d0835c1a94b9eba46e15964181f49f70d1d061440443ab3b7cbc798fa` |
| `v81_provenance_audit.csv` | `5d4817be15eadcedcd481f728bc7ae4458899ed6a9c9338b5e05630d2afccc0b` |

### 11.6 更新结论

本轮完成了真实 GPU-side instrumentation 与 fused backward probe，但 v8.1 minimum success 仍未达成：

```text
KF4:
  StrictPass = true
  GradPass = true
  NoTeacherNoLossPass = true
  FullGridS2Pass = true
  TeacherFreeCEMacroPass = false
```

机制结论：

1. Triton in-place SiLU backward 是真实 fused GPU path，但 full-grid 里 GradPass 只到 `5/6`，不能 official。
2. ATen fused SiLU backward 数值更稳，`KF3/KF4` 均 GradPass `6/6`。
3. `KF4` 是当前最好的系统候选：FullGridS2 `9/9`，step max `1.3120`，memory max `1.0468`。
4. 但 `KF4` macro 只有 `+0.01836`，没有保住 `KC6` 的 `+0.02435` quality，因此不能记 minimum success。
5. GPU allocator 实测表明当前 peak 不靠 CPU/offload；`cpu_offload_used=0`。bs512 peak 在 update/stack-backward 附近，root input cache 与 optimizer state 是可见来源，但已不足以解释全部 formal S1 gap。

更新后的最终一句话：

> v8.1 现在已有一个真正 GPU-native、GradPass、FullGridS2 的 fused-backward 系统路径 `KF4`，但它牺牲了 CE-only macro margin；而之前最强 quality 路径 `KC6` 仍差 bs512 memory。下一步不应做 CPU offload，也不应改 loss/teacher，而应把 `KC6` 的表达力路径和 `KF4` 的 ATen fused/S2 路径合并或定位两者 macro 差异的具体数值/初始化来源。

## 12. KC6 / KF4 macro 差异定位与合并尝试

本节记录一次专门针对 `KC6` 与 `KF4` 的同协议 co-run。目的有两个：

1. 定位 `KC6` 的 macro advantage 与 `KF4` 的 S2/fused path 差异来源。
2. 尝试构造 `KF5/KF6`，把 `KC6` 的 explicit SiLU derivative 表达力路径与 fused/in-place backward、AdamW addcdiv update 合并。

约束：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
class_weight_used = 0
sampler_changed = 0
cpu_offload_used = 0
fake/proxy = 0
```

新增候选：

| candidate | 结构含义 | 目的 |
|---|---|---|
| `KF5` | packed-prefix + explicit in-place SiLU backward + FP32 AdamW | 保留 `KC6` explicit derivative 语义，同时减少 backward temp |
| `KF6` | packed-prefix + explicit in-place SiLU backward + AdamW addcdiv update | 在 `KF5` 基础上合并 `KF4` 的 update path |

主 run：

```bash
python experiments/run_gafu_v81_real.py \
  --out-dir results/real_rerun_20260506/v81_ceonly_kc6_kf4_merge_diagnosis_5seed_20260508T050000Z \
  --fresh \
  --device auto \
  --candidates B0,KC6,KF4,KF5,KF6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 20 \
  --bench-reps 100 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

### 12.1 Route

`v81_route_decision.json`：

```json
{
  "route": "R4-CEOnlyTeacherFreeMacroButS2Fail",
  "best_official_candidate_id": "KC6",
  "strict_pass": 1,
  "grad_pass": 1,
  "teacher_free_ce_macro_pass": 1,
  "fullgrid_s2_pass": 0,
  "success_v81_minimum": 0,
  "macro_gap": "0.024348958333333334",
  "ci95_low": "0.015234375",
  "holm_p": "0.0015692984264888779",
  "test_gap": "0.023177083333333334",
  "memory_ratio_max": 1.0530812221483614,
  "step_ratio_max": 1.5708835717265677,
  "S2_shape_count": 5,
  "no_fake": true,
  "no_proxy": true
}
```

判断：`KC6` 仍是 quality best，且过 CE-only macro gate；但 `KC6` 没过 FullGridS2，因此 v8.1 minimum success 仍未达成。

### 12.2 Macro / S2 co-selection

`p6_official_coselection.csv`：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | memory max | step max | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KC6` | `+0.024349` | `+0.015234` | `1.57e-03` | `+0.023177` | `+0.002713` | `-0.093125` | 1 | `1.053081` | `1.570884` | `5/9` |
| `KF4` | `+0.018359` | `+0.012109` | `1.27e-03` | `+0.027734` | `-0.000434` | `-0.091186` | 1 | `1.046778` | `1.449435` | `9/9` |
| `KF5` | `+0.018229` | `+0.011328` | `4.66e-03` | `+0.023177` | `-0.001984` | `-0.093340` | 1 | `1.053081` | `1.570727` | `5/9` |
| `KF6` | `+0.018099` | `+0.010417` | `4.66e-03` | `+0.023438` | `+0.001770` | `-0.099602` | 1 | `1.053081` | `1.518668` | `5/9` |

判断：

- `KF4` 成功保住 S2/fused path，但没有保住 `KC6` macro。
- `KF5/KF6` 没能完成合并：它们既没有恢复 `KC6` 的 macro，也没有恢复 `KF4` 的 FullGridS2。
- `KF6` 的 NLL 最低，但 accuracy macro 没有跟上，不能把 NLL 当作 official macro pass。

### 12.3 Dataset-level macro 来源

`p13_kc6_kf4_macro_source_v81.csv` 汇总自 `p1_significance_audit.csv`：

| candidate | F-MNIST gap | KMNIST gap | MNIST gap | macro gap |
|---|---:|---:|---:|---:|
| `KC6` | `+0.018750` | `+0.026953` | `+0.027344` | `+0.024349` |
| `KF4` | `+0.014844` | `+0.014844` | `+0.025391` | `+0.018359` |
| `KF5` | `+0.015234` | `+0.016797` | `+0.022656` | `+0.018229` |
| `KF6` | `+0.014844` | `+0.019141` | `+0.020313` | `+0.018099` |

相对 `KC6` 的主要损失：

| candidate | F-MNIST delta | KMNIST delta | MNIST delta |
|---|---:|---:|---:|
| `KF4` | `-0.003906` | `-0.012109` | `-0.001953` |
| `KF5` | `-0.003516` | `-0.010156` | `-0.004688` |
| `KF6` | `-0.003906` | `-0.007813` | `-0.007031` |

判断：`KF4/KF5/KF6` 与 `KC6` 的 macro 差异主要来自 KMNIST validation gap collapse；不是所有 dataset 均匀下降。

### 12.4 One-step backward delta

新增 `p12_kc6_kf4_backward_delta_v81.csv`，用同 init、同 dataset、同 batch=128 的单步 loss/gradient 比较 `KC6` vs `KF4/KF5/KF6`。

| comparison | dataset | loss delta | grad abs max | grad rel L2 |
|---|---|---:|---:|---:|
| `KC6` vs `KF4` | MNIST | `0.0` | `7.45e-09` | `7.31e-08` |
| `KC6` vs `KF4` | F-MNIST | `0.0` | `7.45e-09` | `6.08e-08` |
| `KC6` vs `KF4` | KMNIST | `0.0` | `3.73e-09` | `6.80e-08` |
| `KC6` vs `KF5` | all 3 | `0.0` | `0.0` | `0.0` |
| `KC6` vs `KF6` | all 3 | `0.0` | `0.0` | `0.0` |

判断：

- 单步 backward 没有暴露 gross gradient bug。
- `KF4` 与 `KC6` 的一阶梯度差异只有 `~1e-7` rel L2；`KF5/KF6` 单步梯度与 `KC6` 完全一致。
- 因此本轮不能把 macro 差异归因成“fused backward 数学错误”。更合理的当前判断是：训练轨迹对 kernel/update 数值路径、op 顺序、allocator/measurement path 有敏感性；差异在多步累积后主要反映到 KMNIST validation。

### 12.5 GPU live-set / S2 差异

`p11_gpu_live_allocator_v81.csv`，MNIST bs512：

| candidate | top peak MB | top phase | stack peak MB | update peak MB | manual cache MB | CPU offload |
|---|---:|---|---:|---:|---:|---:|
| `KC6` | `21.1782` | update | `20.8853` | `21.1782` | `1.5313` | 0 |
| `KF4` | `20.9556` | update | `20.7603` | `20.9556` | `1.5313` | 0 |
| `KF5` | `21.1782` | update | `20.8853` | `21.1782` | `1.5313` | 0 |
| `KF6` | `20.9556` | update | `20.8853` | `20.9556` | `1.5313` | 0 |

判断：

- `KF4` 的 S2 主要来自 ATen fused backward + addcdiv update 组合降低 stack/update peak。
- `KF5` 保留 explicit derivative 后，memory 回到 `KC6` 水平，说明 explicit in-place rewrite 没有真正减少 live-set。
- `KF6` 降低了 update peak，但 stack backward peak 仍是 `20.8853 MB`，因此 FullGridS2 仍失败。
- 全程 `cpu_offload_used=0`，没有 CPU optimizer state/offload 路径。

### 12.6 No-Fake / No-Proxy 与 hash

| artifact | rows checked | fake/proxy nonzero |
|---|---:|---:|
| `v81_provenance_audit.csv` | `248` | `0` |
| `p12_kc6_kf4_backward_delta_audit_v81.csv` | `9` | `0` |
| `p13_kc6_kf4_macro_source_audit_v81.csv` | `29` | `0` |

关键 hash：

| file | SHA256 |
|---|---|
| `experiments/run_gafu_v80_real.py` | `40b17de643315340e93568161bc83c9d7d1d4b4e2f310fbc853b699367e46a76` |
| `experiments/run_gafu_v81_real.py` | `0532921db73367f139bc9b0041de6be7578b7d7fadaeac43f7dad25ebc2f0091` |
| `v81_route_decision.json` | `2f2a89ffe7b80e840e402c0623a957984e20e1a2f179a64340c805b80cfe7dcf` |
| `p11_gpu_live_allocator_v81.csv` | `8d14b12c4dca60abe8628b24d8658b35a7e268b22995aac111444ad566836aea` |
| `p12_kc6_kf4_backward_delta_v81.csv` | `7f51cc042d3fd377b8e3a8b29906841f5810e6ce01cbf6ca967e9782dc673dd9` |
| `p12_kc6_kf4_backward_delta_audit_v81.csv` | `ad72a6c83a84f54ca9786a132b64f75524109d3bdffeb908b125a3edaafb7a80` |
| `p13_kc6_kf4_macro_source_v81.csv` | `896872fbc50e7666b1219050cda351d13412d9d018d61d70372e3f573fcd6a42` |
| `p13_kc6_kf4_macro_source_audit_v81.csv` | `3a360e80c535d75180f45b1880e3823051125a33cfb409c37214968e5950e9b1` |
| `v81_provenance_audit.csv` | `de19a85c60bfbec86c47634ea950da327e93a0412a085ee834d818e0764d0f3e` |

### 12.7 结论

本轮没有成功把 `KC6` 的表达力路径和 `KF4` 的 S2/fused path 合并：

```text
KC6:
  TeacherFreeCEMacroPass = true
  FullGridS2Pass = false

KF4:
  TeacherFreeCEMacroPass = false
  FullGridS2Pass = true

KF5/KF6:
  TeacherFreeCEMacroPass = false
  FullGridS2Pass = false
```

机制结论：

1. `KC6` 与 `KF4` 的差异不是单步 gradient correctness 问题；单步 loss 相同，gradient rel L2 约 `1e-7` 或 `0`。
2. macro 差异主要集中在 KMNIST：`KC6 +0.02695`，`KF4 +0.01484`。
3. `KF4` 的 S2 path 是真实的：FullGridS2 `9/9`，CPU offload `0`。
4. `KF5/KF6` 的合并尝试失败：显式 derivative/in-place rewrite 既没有保住 `KC6` macro，也没有保住 `KF4` S2。
5. 下一步如果继续，应做 late-step trajectory divergence / per-seed confusion / margin distribution，对比 `KC6` 与 `KF4` 在 KMNIST 上从 step 120 到 step 240 的样本级轨迹；系统侧则应优先在 `KC6` 原表达力路径上做 allocator/workspace-level trim，而不是继续替换 backward derivative 公式。
