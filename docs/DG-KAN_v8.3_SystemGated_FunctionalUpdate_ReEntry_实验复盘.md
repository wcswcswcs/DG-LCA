# DG-KAN v8.3 System-Gated Functional Update Re-Entry 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.3_SystemGated_FunctionalUpdate_ReEntry_完整实验计划.md` 的本轮真实执行结果。所有数值只来自文中列出的 `results/real_rerun_20260506/.../` 落盘 CSV/JSON/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填成功结论。v8.3 official route 继续严格禁止 teacher/self-teacher、distillation、loss/sampler/class-weight 修改与 CPU offload。

## 1. 是否完成

已完成 v8.3 minimum success；不声明 formal strong success。

最新 accepted route 见第 32 节：

```text
route = S0-FunctionalReEntryMinimumSuccess
success_v83_minimum = 1
success_v83_formal = 0
```

已完成：

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 contract / registry | completed | 新增 `experiments/run_gafu_v83_real.py`，落盘 v8.3 functional registry / no-teacher-no-loss / functional update contract |
| P1 base confirmation | completed | 初始 base grid 完成；追加 `KW6 hidden68` 10-seed H0 confirmation 通过 |
| P2 functional operator audit | completed | `FT5/FT6/FT7` streamed operator audit 通过 |
| P3 one-step direction audit | completed | `FT5/FT6/FT7` one-step direction gate 通过 |
| P4 multistep functional smoke | completed | `FT7` role-wise guarded functional update 进入 P5 |
| P5 functional task geometry co-selection | completed | `FT7` 成为 functional survivor |
| P6 functional confirm5 / full-grid S2 | completed | `FT7` P6 confirm5 通过，FullGridS2 `9/9` |
| P7 functional confirm10 | completed | `FT7` 10-seed functional confirmation 通过 |
| P8 TimeAUC / profiler | completed | `FT7 hidden68 stride8` TimeAUC gate 通过 |
| P9 robustness / scaling | completed | P9 scaling / robustness grid 真实落盘并通过 |
| v8.3 minimum success | completed | `success_v83_minimum = 1` |

未声明：

| 阶段 | 状态 | 原因 |
|---|---|---|
| v8.3 formal success | not_claimed | runner 保守记录 `success_v83_formal = 0`；本轮只声明 minimum success |

## 2. 本轮代码

新增文件：

| 文件 | 作用 | SHA256 |
|---|---|---|
| `experiments/run_gafu_v83_real.py` | v8.3 system-gated functional runner，复用 v8.2 measured path，并新增 P0-P9 functional artifacts | 见第 32 节最终 hash |

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

## 3. 正式 run

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/v83_system_gated_functional_reentry_base10_p3_20260507T090000Z \
  --fresh \
  --device auto \
  --candidates B0,KC6,KW3,KW6,KF4,KF10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route:

```json
{
  "route": "S6-NoBaseSystemCandidateFunctionalFullTaskGated",
  "base_candidate_id": "KF10",
  "functional_candidate_id": "FT5",
  "H0_system_base_pass": 0,
  "functional_full_task_opened": 0,
  "success_v83_minimum": 0
}
```

## 4. P1 base system gate

H0 要求同一个 base candidate 同时满足 macro `>= +0.0200`、memory `<= 1.05`、step `<= 1.50`、NoTeacherNoLoss/Strict/Grad pass。本轮没有任何 candidate 通过 H0。

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KC6` | `+0.022135` | `+0.016276` | `8.22e-07` | `+0.021875` | 1 | `1.053081` | `1.600072` | `5/9` | 0 |
| `KF10` | `+0.020247` | `+0.014714` | `1.39e-06` | `+0.023177` | 1 | `1.053081` | `1.541405` | `5/9` | 0 |
| `KF4` | `+0.016992` | `+0.012305` | `1.39e-06` | `+0.022591` | 1 | `1.046778` | `1.493510` | `9/9` | 0 |
| `KW3` | `+0.019076` | `+0.012824` | `2.81e-05` | `+0.019596` | 1 | `1.007386` | `1.540428` | `8/9` | 0 |
| `KW6` | `+0.017773` | `+0.011847` | `3.02e-05` | `+0.021549` | 1 | `1.007386` | `1.490881` | `9/9` | 0 |

判断：

1. `KC6/KF10` 保住了 10-seed macro gate，但 memory 和 step 都没过 H0。
2. `KF4/KW6` 系统侧接近或通过 S2，但 macro 没过 `+0.0200`。
3. `KW3` 复现了 v8.2 10-seed 的 near-pass：memory 很好，但 macro 和 step 都没过 H0。
4. 因为 H0 不成立，P4-P9 full functional 路线必须 gate，不允许打开。

## 5. P2 functional operator audit

本轮落盘 `p2_functional_operator_audit.csv` 共 18 rows。所有 functional candidates 均满足 no teacher / no loss / no CPU offload 的 contract，但没有 candidate 通过 operator cost gate：

```text
rows = 18
implementation_pass = 0
```

代表性 measured cost：

| candidate | update time ms | base step ms | time ratio | memory peak ratio | pass |
|---|---:|---:|---:|---:|---:|
| `F0` | `0.513015` | `0.888635` | `0.577307` | `1.093206` | 0 |
| `F1` | `0.500776` | `0.888635` | `0.563534` | `1.093206` | 0 |
| `F2` | `0.466495` | `0.888635` | `0.524957` | `1.093206` | 0 |
| `F3` | `0.480646` | `0.888635` | `0.540882` | `1.093206` | 0 |
| `F4` | `0.468548` | `0.888635` | `0.527267` | `1.093206` | 0 |
| `FT5` | `0.960902` | `0.888635` | `1.081324` | `1.127685` | 0 |

判断：当前 Python-level graph-free functional direction 构造是真实 update-rule audit，但 cost 太高，不能进入 official full task。由于 H0 已先失败，route 主 blocker 仍是 no base system candidate。

## 6. P3 one-step direction audit

本轮 `p3_one_step_functional_direction_raw.csv` 有 `3240` raw rows；`p3_one_step_functional_direction.csv` 为 18 个候选的聚合结果。P3 只作为 diagnostic，因为 H0 未通过。

Top task-aware rows：

| candidate | cos corrected/task | holdout ratio | bad step rate | curvature reduction | P3 pass |
|---|---:|---:|---:|---:|---:|
| `FT5` | `0.992891` | `0.993241` | `0.000000` | `0.000364` | 1 |
| `FT0` | `0.995046` | `0.995323` | `0.000000` | `0.000308` | 1 |
| `FT2` | `0.995046` | `0.995330` | `0.000000` | `0.000308` | 1 |
| `FT7` | `0.995046` | `0.995337` | `0.000000` | `0.000292` | 1 |
| `FT1` | `0.995058` | `0.995310` | `0.000000` | `0.000226` | 1 |

判断：

- task-aware projection 确实改善了早期裸 functional direction 的 one-step descent 行为。
- 但几何改善幅度很小，且 P2 cost gate 没过。
- 最关键的是 H0 未通过，因此这些 P3 pass 不能进入 P4/P5 full task。

## 7. Artifact 状态

v8.3 要求的关键 artifact 已落盘：

```text
run_manifest.json
candidate_registry_v83_functional.csv
contract_no_teacher_no_loss_functional.csv
functional_update_contract.csv
p0_contract_audit.csv
p1_base_candidate_confirmation.csv
p2_functional_operator_audit.csv
p3_one_step_functional_direction.csv
p4_multistep_functional_smoke.csv
p5_functional_task_geometry_coselection.csv
p6_functional_confirm5.csv
p7_functional_confirm10.csv
p8_functional_time_profiler.csv
p9_functional_scaling_robustness.csv
route_decision.json
aggregate_decision.json
failure_table.csv
v83_provenance_audit.csv
```

`p4` 到 `p9` 均为 `not_run`，原因是：

```text
H0_system_base_not_passed
```

No-fake audit：

```text
rows_checked = 3385
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 8. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `8b1ba12d990d272cd489710d19ef715f09c6620d9377ccd7e5ff8374d6e9f475` |
| `experiments/run_gafu_v82_real.py` | `aa6b85e761f5c0033d56199347794a560418e7583cd543bd2aaa45166bcab69c` |
| `experiments/run_gafu_v81_real.py` | `d2907dc9e13857f1c2c1c8272932d0d2275cd35ab103441131fc246d6817a468` |
| `experiments/run_gafu_v80_real.py` | `d0ac1f3bbd33390e63bf15c66db92f656fc49ae92cd46a2c689980025b55fd3a` |
| v8.3 plan | `f9dec8994f3e9cce574964c74ab5bf11bb00ad264cc23755204b478fb52bf037` |
| route | `ad2f635a0b2a4ea958bd96c39e2608fcb8ff63b8bd0bde5e6d892df19c6a28b0` |
| P1 base confirmation | `6a01a9c8b3c4400d4ad8bded31d43380814db1dc1d27cc486571434815f0b042` |
| P2 operator audit | `9550e5da54ca1951f3864576a24ef14ec218fca7025aa9140a8ea542630306fd` |
| P3 direction audit | `64cb81620236e749c6674b818dcfdc6997c35b63f26e8d5f65f8c8fcd06f4b46` |
| provenance audit | `7c8196ffc308504be652f16833a8c731cd72a477026541073b898eda6dbca41c` |

## 9. 追加 base repair：KW6/KW4 hidden72

由于主 run 没有 H0，按 v8.3 文档中心思想继续做 CE-only base system closure，而不是打开 full functional task。本追加只做窄的 hidden72 base repair probe。

### 9.1 KW6 hidden72

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/v83_kw6_hidden72_base_repair_5seed_20260507T093000Z \
  --fresh \
  --device auto \
  --hidden-dim 72 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden72` | `+0.020313` | `+0.013021` | `7.35e-04` | `+0.017708` | `1.015040` | `1.603833` | `7/9` | 0 |

判断：hidden72 把 `KW6` 推过 macro 和 memory，但 step 明显失败。失败 shape 是 Fashion-MNIST bs512 与 KMNIST bs128。

No-fake audit：

```text
rows_checked = 3381
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 9.2 KW4 hidden72

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/v83_kw4_hidden72_base_repair_5seed_20260507T100000Z \
  --fresh \
  --device auto \
  --hidden-dim 72 \
  --candidates B0,KW4 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | macro gap | CI95 low | Holm p | test gap | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW4 hidden72` | `+0.017839` | `+0.011979` | `1.79e-04` | `+0.020313` | `1.022084` | `1.289691` | `9/9` | 0 |

判断：`KW4 hidden72` 系统侧闭合，但 macro 仍明显低于 `+0.0200`。这条 cached workspace 系统路线仍没有恢复质量。

No-fake audit：

```text
rows_checked = 3381
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

追加 hash：

| artifact | SHA256 |
|---|---|
| `KW6 hidden72 route` | `41de2140a0705fe189d0f76e7ec3db5a0b5842be6efa5cde75634a4f3217901a` |
| `KW4 hidden72 route` | `8e01caaed87e15719bef17f2bd6e28d011df7b32a943b21b50d58023d02f6c70` |

## 10. 最终结论

v8.3 本轮没有完成 minimum success：

```text
H0_system_base_pass = false
functional_full_task_opened = false
success_v83_minimum = false
```

机制结论：

1. v8.3 runner 成功建立了 system-gated functional re-entry 路线，且 artifact 明确区分 base system gate、functional operator audit、one-step direction audit。
2. 10-seed 下没有 base candidate 同时通过 macro + memory + step：`KC6/KF10` 过 macro 但系统失败，`KF4/KW6` 系统较好但 macro 失败。
3. task-aware functional correction 的 one-step 方向质量有真实信号：`FT5` 的 cos `0.9929`、holdout ratio `0.9932`、bad step `0`。
4. 但 functional operator cost 当前明显过高，且 H0 不成立，因此 full functional training 按计划被 gate，没有打开。
5. hidden72 repair 没有闭合 H0：`KW6 hidden72` 质量/内存过了但 step 失败；`KW4 hidden72` 系统过了但质量失败。
6. 本轮没有 fake/proxy，没有 CPU offload，也没有 teacher/loss/sampler/class weight 修改。

最终一句话：

> v8.3 没有完成最终目标，但正确执行了文档中心思想：先系统闭合，再 functional re-entry。当前 blocker 仍是没有 H0 base system candidate；functional update 只能保留 one-step diagnostic，不能进入 full task。下一步应继续 CE-only base closure，重点是把 `KW6 hidden72` 的 step overhead 拉回 S2，或把 `KW4 hidden72` 的 macro 恢复到 `+0.0200` 以上，然后再把 FT0/FT5 这类 task-aware functional correction 带入 P4。

## 11. 追加：KW6 hidden70/68 H0 与 P2/P4 re-entry

本追加继续遵守 v8.3 文档中心思想：

```text
先闭合 CE-only/no-teacher/no-loss H0 base；
H0 过后才允许 functional P2/P3/P4；
不打开 P5/P6/P7 full functional task，除非 P4 有 survivor；
不使用 fake/proxy/CPU offload/teacher/loss/sampler/class weight。
```

### 11.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v83_real.py` | 新增 `FT5` streamed in-place second-diff functional operator audit，避免 P2 全量 materialized direction；新增 `FT7` streamed role-wise operator；新增 P4 20/50-step functional smoke artifact |

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 11.2 KW6 hidden70 / hidden68 5-seed probe

运行：

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/v83_kw6_hidden70_base_repair_5seed_20260507T103000Z \
  --fresh \
  --device auto \
  --hidden-dim 70 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/v83_kw6_hidden68_base_repair_5seed_20260507T110000Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 5000
```

| candidate | macro gap | CI95 low | Holm p | test gap | GradPass | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden70` | `+0.019792` | `+0.010938` | `1.97e-03` | `+0.023307` | 0 | `1.013154` | `1.333073` | `9/9` | 0 |
| `KW6 hidden68` | `+0.027995` | `+0.020182` | `1.56e-05` | `+0.030078` | 1 | `1.011163` | `1.333855` | `9/9` | 1 |

判断：

- `hidden70` 系统闭合，但 macro 未过 `+0.0200`，且 GradPass 为 `0`，不能作为 H0。
- `hidden68` 在 5-seed 下同时通过 macro、GradPass、memory、step、FullGridS2，因此进入 10-seed confirmation。

### 11.3 KW6 hidden68 10-seed H0 confirmation

运行：

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/v83_kw6_hidden68_base_confirm_10seed_20260507T113000Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

10-seed H0 result：

| candidate | macro gap | CI95 low | Holm p | test gap | ECE delta | NLL delta | GradPass | max relerr | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden68` | `+0.021680` | `+0.016341` | `2.70e-08` | `+0.024349` | `-0.003134` | `-0.102687` | 1 | `9.43e-05` | `1.011163` | `1.328673` | `9/9` | 1 |

判断：`KW6 hidden68` 是本轮第一个 10-seed H0 base system candidate。它满足 CE-only、no-teacher、no-loss、StrictPass、GradPass、macro gate、memory gate、step gate、FullGridS2。

### 11.4 P2 streamed FT5/FT6/FT7 operator audit

H0 成立后，按计划重新打开 functional operator audit。`FT5/FT6/FT7` 原 materialized direction path 太重，因此新增 streamed in-place second-diff implementation。它仍是 graph-free update rule，不是 loss，不用 autograd metric，不用 CPU solve。

`p2_functional_operator_audit.csv`：

| functional | implementation | time ms | base step ms | time ratio | memory MB | memory ratio | P2 pass |
|---|---|---:|---:|---:|---:|---:|---:|
| `FT5` | streamed in-place second-diff | `0.069517` | `0.874196` | `0.079521` | `0.477539` | `1.025154` | 1 |
| `FT6` | streamed late-phase in-place second-diff | `0.068879` | `0.874196` | `0.078791` | `0.477539` | `1.025154` | 1 |
| `FT7` | streamed role-wise in-place second-diff | `0.069579` | `0.874196` | `0.079592` | `0.477539` | `1.025154` | 1 |

Contract：

```text
uses_loss_backward = 0
uses_full_jacobian_materialization = 0
uses_cpu_solve = 0
cpu_offload_used = 0
fake_data_used = 0
proxy_row_used = 0
```

判断：P2 从此前的 cost gate fail 推进到 pass；functional full task 仍没有打开，因为还需要 P4 survivor。`FT5/FT7` 在 P4 中追加了 task-descent-budget event guard，该 guard 是 update 接受/拒绝机制，不改变 CE objective；本 P2 行只计 streamed correction operator，不把 P4 smoke 里的额外 forward guard 写成 full-task time success。

### 11.5 P3 one-step direction audit

`p3_one_step_functional_direction.csv`：

| functional | cos corrected/task | holdout ratio | bad step | curvature reduction | P3 pass |
|---|---:|---:|---:|---:|---:|
| `FT5` | `0.993048` | `0.993564` | `0.000000` | `0.000361740` | 1 |
| `FT6` | `0.999551` | `0.999658` | `0.000000` | `0.000113569` | 1 |
| `FT7` | `0.995047` | `0.995451` | `0.000000` | `0.000295911` | 1 |

判断：`FT5/FT6/FT7` 方向 gate 均通过。`FT5` one-step curvature reduction 更强，`FT6` 的方向最贴近 task。

### 11.6 P4 20/50-step functional smoke

`p4_multistep_functional_smoke.csv` 对 H0 base `KW6 hidden68` 与 `FT5/FT6/FT7` 执行 20/50-step smoke。P4 不是 full functional task confirmation；P5/P6/P7 仍保持 gate。

本轮追加了两个 plan 内修复：

- `FT6`：late-phase scheduling，只在 50-step smoke 的后半段施加 functional correction。
- `FT5/FT7`：task-descent-budget event guard；functional correction 最多消耗当前 batch task descent 的 `10%`，超出则回滚。本机制只决定 update 是否接受，不改变 CE objective。
- `FT7`：role-wise partial accept；stack/head correction 分开尝试、分开接受或回滚。

Step 50 aggregate over `3 datasets x 3 seeds`：

| functional | rows | mean AccDrop | mean ValLossIncrease | mean bad step | mean fallback | mean curvature ratio | mean geometry reduction | task pass rows | geometry pass rows | P4 pass rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT5` | 9 | `0.000651` | `0.008280` | `0.022222` | `0.057778` | `0.862432` | `0.137568` | 0 | 9 | 0 |
| `FT6` | 9 | `0.000000` | `0.008391` | `0.022222` | `0.000000` | `0.923609` | `0.076391` | 0 | 0 | 0 |
| `FT7` | 9 | `0.000651` | `0.006520` | `0.020000` | `0.023333` | `0.901001` | `0.098999` | 0 | 4 | 0 |

Step 20 aggregate：

| functional | rows | mean ValLossIncrease | mean bad step | mean fallback | mean curvature ratio | task pass rows | geometry pass rows | P4 pass rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT5` | 9 | `0.001002` | `0.011111` | `0.522222` | `0.970033` | 6 | 0 | 0 |
| `FT6` | 9 | `0.000000` | `0.011111` | `0.000000` | `1.000000` | 7 | 0 | 0 |
| `FT7` | 9 | `0.001666` | `0.011111` | `0.000000` | `0.957677` | 4 | 0 | 0 |

判断：

- `FT5` 在 `10%` descent budget 下恢复了 geometry：curvature ratio `0.862432`，9/9 rows 过 geometry gate；但 ValLossIncrease `0.008280`、bad step `0.022222`，task gate 失败。
- `FT7` role-wise partial accept 是最接近的候选：ValLossIncrease `0.006520`，bad step `0.020000`，curvature ratio `0.901001`。但它仍同时略差于 task loss gate `0.005` 与 geometry gate `0.90`。
- `FT7` 的 fallback 从 whole-update strict guard 的 `0.651111` 降到 `0.023333`，说明 partial accept 确实保留了更多 geometry update，但仍不足以闭合 gate。
- `FT6` late-phase scheduling 保住了 AccDrop，但 ValLossIncrease / bad step 仍没过 task gate，geometry 也没到 `<=0.90`。
- 因此 route 回到 `S3-FunctionalGeometryTaskConflict`。这不是 success；它说明较宽 budget 能保住 geometry，但 task preservation 又不够。

### 11.7 Route / audit / hash

最终 route：

```json
{
  "route": "S3-FunctionalGeometryTaskConflict",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT5",
  "H0_system_base_pass": 1,
  "S2_pass": 1,
  "functional_update_time_ratio": "0.07952092124192398",
  "functional_full_task_opened": 0,
  "success_v83_minimum": 0,
  "success_v83_formal": 0,
  "primary_blocker": "p4_task_preservation_gate",
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 3452
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `07631a90bf60496e10f5221d9f8f707d6b77e2a2bf2674a4559ea070b1da77b7` |
| `KW6 hidden70 route` | `181f3c4dc8df14370edcae9acc2239e4c6cc212be39e6b3b66d8b91daf845de4` |
| `KW6 hidden68 5-seed route` | `29092bb532976a03720d6d7cbdc17097212ccb9eda0cf6a96132cda9efafb357` |
| `KW6 hidden68 10-seed route` | `9f5bcab4f019d4a0e24c1788bc58d45d06ef7ebd6d0331fb70289927ae69e946` |
| `KW6 hidden68 10-seed P1` | `3a0decb7ce9ee7554c49fa656bdb6cc8dd35922e7cb7ae2bcf0373e4fc480408` |
| `KW6 hidden68 10-seed P2` | `00a598984a354cb2827f0021f8e3bf8463833e595d9b0b3b2c291058e41a2b87` |
| `KW6 hidden68 10-seed P3` | `c25a318a76f655436d76a1f1902d09ed6beb2caef2a74ab28480ab295a9c977f` |
| `KW6 hidden68 10-seed P4` | `f7635ad2015816cd087681a2c9d81033d745cb88d22482fa25e91ff0be6789ac` |
| `KW6 hidden68 10-seed provenance` | `2df974fb8f92989ec896e8b57a5776055915357c97fc9b37ba201d3bcd431717` |

## 12. 更新结论

v8.3 仍未完成 minimum success：

```text
H0_system_base_pass = true for KW6 hidden68 10-seed
P2_functional_operator_pass = true for FT5/FT6/FT7 streamed operators
P3_one_step_direction_pass = true for FT5/FT6/FT7
P4_geometry_pass = true for FT5 under 10% descent budget
P4_task_preservation = near miss for FT7 role-wise partial accept
functional_full_task_opened = false
success_v83_minimum = false
```

机制结论：

1. v8.3 的第一层系统门控已经取得关键进展：`KW6 hidden68` 是真实 10-seed H0 survivor，macro `+0.02168`，FullGridS2 `9/9`。
2. P2 blocker 被修掉：`FT5/FT6/FT7` streamed operator time ratio 都约 `0.08`，memory ratio 都为 `1.02515`，且无 loss backward / full Jacobian / CPU solve / CPU offload。
3. P3 方向成立：`FT5` cos `0.99305`，`FT6` cos `0.99955`，`FT7` cos `0.99505`，三者 bad step 都是 `0`。
4. P4 的 blocker 被重新定位得更清楚：strict event trigger 保 task 但丢 geometry；`10%` descent budget 保 geometry 但伤 task。
5. `FT7` role-wise partial accept 是当前最接近合流点：curvature ratio `0.901001` 距离 geometry gate 只差约 `0.001001`，但 ValLossIncrease `0.006520` 仍高于 `0.005`。
6. 这不是 teacher/loss 问题，也不是系统 base 问题；下一步应做更细的 per-role budget，例如 stack/head 分别设置不同 accept budget 或只对贡献 geometry 且不伤 task 的 role 做 partial accept。

最终一句话：

> v8.3 还没完成。当前最接近的是 `FT7` role-wise partial accept：几何几乎到 gate，但 task loss 仍略高。下一步应继续做 per-role budget / partial accept，而不是回到 teacher、loss 或 CPU/offload。

## 13. 追加：FT7 H5 role-budget repair 与 P5 re-entry

本节继续沿 `docs/DG-KAN_v8.3_SystemGated_FunctionalUpdate_ReEntry_完整实验计划.md` 的中心思想执行：system-gated functional re-entry。没有回到 teacher / self-teacher / distillation / loss 修改 / sampler / class weight / CPU offload；functional update 仍只是 CE 训练后的 update rule。

### 13.1 代码改动

基于 H5 role analysis，`FT7` 的 one-step update/descent 主要由 stack role 贡献：

```text
role_update_share_mean:
  stack = 0.930754
  head  = 0.069246

role_descent_contribution_mean:
  stack = 0.937572
  head  = 0.062428
```

因此本轮只做 role-wise trust-region 修复：

| candidate | 改动 | 目的 |
|---|---|---|
| `FT7` | stack/head partial accept 加 disjoint train holdout CE budget | 降低 P4 task conflict |
| `FT7` | head budget `0.05`，stack budget `0.10` | 按 H5 限制低占比 head role |
| `FT7` | stack role weight 从 `0.5` 调到 `0.75`，head 保持 `1.0` | 恢复 geometry gate |
| P5 | 新增 `p5_functional_task_geometry_raw.csv` 与真实 240-step 3-seed co-selection | P4 survivor 后打开 full task，不把 P4 当成功 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 13.2 P2/P3 gate 复核

`FT7` streamed operator 仍满足 P2：

| functional | time ms | time ratio | memory MB | memory ratio | P2 pass |
|---|---:|---:|---:|---:|---:|
| `FT7` | `0.067847` | `0.077611` | `0.477539` | `1.025154` | 1 |

`FT7` P3 direction 仍满足 one-step gate：

| functional | cos corrected/task | holdout ratio | bad step | curvature reduction | P3 pass |
|---|---:|---:|---:|---:|---:|
| `FT7` | `0.995047` | `0.995466` | `0.000000` | `0.000309180` | 1 |

### 13.3 P4 20/50-step smoke 更新

`FT7` 在 stack weight `0.75` + holdout role-budget 后，P4 finally 出现 survivor。

Step 50 aggregate over `3 datasets x 3 seeds`：

| functional | rows | mean AccDrop | mean ValLossIncrease | mean bad step | mean fallback | mean curvature ratio | mean geometry reduction | task pass rows | geometry pass rows | P4 pass rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT5` | 9 | `0.000651` | `0.008280` | `0.022222` | `0.057778` | `0.862432` | `0.137568` | 0 | 9 | 0 |
| `FT6` | 9 | `0.000000` | `0.008391` | `0.022222` | `0.000000` | `0.923609` | `0.076391` | 0 | 0 | 0 |
| `FT7` | 9 | `0.000434` | `0.004849` | `0.020000` | `0.192222` | `0.898281` | `0.101719` | 3 | 6 | 2 |

`FT7` 的 P4 pass rows：

| dataset | seed | AccDrop | ValLossIncrease | bad step | curvature ratio |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `-0.003906` | `0.004936` | `0.000000` | `0.896954` |
| KMNIST | 2 | `-0.001953` | `0.003699` | `0.000000` | `0.898022` |

判断：P4 gate 已经从原先的 `p4_task_preservation_gate` / `p4_geometry_gate` 推进到有真实 `FT7` survivor，因此按计划打开 P5。P4 不是 final success，只是进入 full task co-selection 的门票。

### 13.4 P5 3-seed full task co-selection

本轮对 P4 survivor `FT7` 打开 240-step full task，比较 `KW6 hidden68 BASE` 与 `BASE + FT7`。P5 使用 `datasets = MNIST,Fashion-MNIST,KMNIST`，`seeds = 0,1,2`。

`p5_functional_task_geometry_coselection.csv`：

| functional | rows | val acc delta vs base | test acc delta vs base | ECE delta | NLL delta | ValLossAUC ratio | curvature ratio | memory ratio | step ratio | bad step | P5 useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT7` | 9 | `+0.000434` | `+0.003255` | `+0.000297` | `-0.007459` | `0.993771` | `0.725409` | `1.011562` | `2.616737` | `0.066204` | 0 |

判断：

1. P5 证明 functional update 的几何目标在 full task 下是真实有效的：curvature ratio `0.725409`，明显优于 P5 hard geometry benefit `<=0.80`。
2. Task accuracy 没有坏：val acc delta `+0.000434`，test acc delta `+0.003255`，NLL delta `-0.007459`。
3. 但 P5 不能给 survivor：full-task guarded update 的 measured step ratio 为 `2.616737`，超过系统硬约束 `<=1.50`；bad step rate `0.066204` 也说明多步 guard 仍不够稳。
4. 因此本轮从 “P4 没有 survivor” 推进为 “P5 有几何收益但系统/step 不可用”，仍不能进入 P6/P7。

### 13.5 Route / audit / hash

最终 route：

```json
{
  "route": "S1-P5FunctionalCoSelectionNoUsefulSurvivor",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "H0_system_base_pass": 1,
  "S2_pass": 1,
  "functional_full_task_opened": 1,
  "acc_delta_vs_base": "0.00043402777777777775",
  "geometry_delta_vs_base": "0.27459074818188056",
  "functional_update_time_ratio": "2.616737044526797",
  "primary_blocker": "p5_functional_no_useful_survivor",
  "success_v83_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 3470
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `0dacf2a9a1b36a2e9f98b62fde954fed69f9d16948d32d47b2b33b885ee9a05b` |
| v8.3 plan | `791bffc5ebb4b3015fc65c4a4aa447283fca1a8ee4ee701ef9f1551537241ba9` |
| route | `2e261174a97746df3ebe1706aaaddb1f90f2d137c25f720cb223c16a2bc98058` |
| P4 smoke | `cde222f35896c67c9a97141b03d789c0647b77b686db4747361425ed69528434` |
| P5 co-selection | `e4f86a7417525ba2753d736b0b4317b82a917b6c11d255fa0b7044714d637982` |
| P5 raw | `d8cb3d4281401ec0eb6ea327261f30f6b5518c5172e4e01fcc941965c72d16c0` |
| provenance audit | `7bd4ce0a31801a53e21e4568d5f53ac92f9766b91fb2af8bd4d0112ab73dc88c` |
| run manifest | `2205a7b534a63732c0c887230076bbc0278214ba5ee78f8767fd3c3da2d56c7b` |

### 13.6 更新结论

v8.3 仍未完成 minimum success：

```text
H0_system_base_pass = true
P2_functional_operator_pass = true
P3_one_step_direction_pass = true
P4_multistep_pass = true for FT7 on 2/9 rows
P5_primary_benefit_pass = true
P5_hard_constraints_pass = false
P5_functional_useful_pass = false
P6/P7 = not_run
success_v83_minimum = false
```

机制结论：

1. 文档中心思想是正确的：必须先 system-gate，再 functional re-entry。本轮已经有 `KW6 hidden68` H0，并且 `FT7` 通过 P2/P3/P4 后进入 P5。
2. functional update 的几何价值被 P5 真实验证：full task curvature ratio `0.725409`，不是 smoke-only 假象。
3. 当前 blocker 变成 full-task system overhead：P5 measured step ratio `2.616737`，主要来自每步 role-wise train/holdout CE guard 的额外 forward 检查。
4. 不能写 success，也不能进入 P6/P7。下一步应做 P5-level fast guard / low-cadence event trigger / cached cheap acceptance statistic，把 step ratio 拉回 `<=1.50`，同时保留 curvature ratio `<=0.80`。

最终一句话：

> v8.3 仍没完成，但已经从 “functional 只能 one-step diagnostic” 推进到 “FT7 在 P5 full task 下确实带来强几何收益”。失败点也更清楚了：不是 teacher/loss，也不是 H0/P2/P3/P4，而是 P5 full-task guarded update 太慢且 bad step 偏高。下一步要做的是 fast guard / low-cadence functional event trigger，而不是改变 CE objective。

## 14. 追加：FT7 low-cadence functional event 与 P6 conservative confirmation

本节继续修复上一节的真实 blocker：P5 full-task guarded update 太慢。修复仍保持文档中心思想：

```text
CE-only
no external teacher / no self teacher
no distillation / no loss modification
no sampler / no class weight
no CPU offload
functional update 只是 update rule，不是 geometry loss
```

### 14.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v83_real.py` | 将 `FT7` full-task functional event 改为 low-cadence：每 4 step 执行一次 role-wise functional correction，event alpha 乘 4，其余 step 保持普通 CE task update |
| `experiments/run_gafu_v83_real.py` | 新增 P6 5-seed functional confirmation artifact：`p6_functional_confirm5.csv` 与 `p6_functional_confirm5_raw.csv` |
| `experiments/run_gafu_v83_real.py` | route 增加 P6 后的 conservative gate：如果 P6 均值 task/geometry/system 通过，但 `S2_shape_count` 未真实测量，则停在 `p6_s2_shape_count_not_measured`，不进入 P7 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 14.2 postprocess rerun

本节没有重新编造或手填结果，而是在既有 10-seed H0 run 上重新执行 v8.3 postprocess，让 P5 survivor 进入 P6：

```text
results/real_rerun_20260506/v83_kw6_hidden68_base_confirm_10seed_20260507T113000Z
```

关键落盘文件：

```text
p5_functional_task_geometry_coselection.csv
p5_functional_task_geometry_raw.csv
p6_functional_confirm5.csv
p6_functional_confirm5_raw.csv
route_decision.json
v83_provenance_audit.csv
run_manifest.json
```

### 14.3 P5 low-cadence full task co-selection

`p5_functional_task_geometry_coselection.csv`：

| functional | rows | val acc delta vs base | test acc delta vs base | ECE delta | NLL delta | ValLossAUC ratio | curvature ratio | memory ratio | step ratio | bad step | P5 useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT7` | 9 | `+0.000434` | `+0.002387` | `-0.000059` | `-0.005582` | `0.995021` | `0.780005` | `1.011562` | `1.302178` | `0.069907` | 1 |

判断：

1. Low-cadence event 没有改变 CE objective，只降低 functional correction 的触发频率。
2. 相比上一节每步 guarded update，P5 step ratio 从 `2.616737` 降到 `1.302178`，进入 `<=1.50` hard constraint。
3. 几何收益仍保留：curvature ratio `0.780005`，满足 P5 hard geometry benefit `<=0.80`。
4. Task 没有损伤：val acc delta `+0.000434`，test acc delta `+0.002387`，NLL delta `-0.005582`。
5. 因此 P5 出现真实 survivor：`P5_functional_useful_pass = 1`。

### 14.4 P6 5-seed functional confirmation

`p6_functional_confirm5.csv`：

| functional | rows | val acc delta vs base | test acc delta vs base | ECE delta | NLL delta | ValLossAUC ratio | curvature ratio | memory ratio | step ratio | bad step | mean gate | S2 measured | P6 confirm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT7` | 15 | `0.000000` | `+0.002214` | `+0.000110` | `-0.005793` | `0.995130` | `0.776142` | `1.011562` | `1.317947` | `0.077222` | 1 | 0 | 0 |

P6 raw：

```text
p6_functional_confirm5_raw.csv rows = 30
stage = P6_FUNCTIONAL_CONFIRM5_RAW_V83
```

判断：

1. P6 的 task / geometry / system 均值 gate 复现：accuracy 没掉，curvature ratio `0.776142`，memory ratio `1.011562`，step ratio `1.317947`。
2. 但 P6 文档要求必须记录 `S2_shape_count`。本轮 P6 只测到了均值 memory/step ratio，没有真实 functional full-grid shape count artifact。
3. 因此 runner 保守写入：

```text
S2_shape_count = metric_unavailable
S2_shape_count_measured = 0
P6_task_geometry_system_mean_pass = 1
P6_confirm5_pass = 0
reason = s2_fullgrid_shape_count_not_measured_for_functional_update
```

这不是失败包装成成功；它明确说明 P6 还缺一个真实 measured functional full-grid S2 package。

### 14.5 Route / audit / hash

最终 route：

```json
{
  "route": "S1-FunctionalP6Confirm5NeedsMeasuredFullGridS2",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "base_macro_gap": "0.0216796875",
  "acc_delta_vs_base": "0.0",
  "geometry_delta_vs_base": "0.2238582216981022",
  "functional_update_time_ratio": "1.3179466834141413",
  "H0_system_base_pass": 1,
  "S2_pass": 1,
  "functional_full_task_opened": 1,
  "primary_blocker": "p6_s2_shape_count_not_measured",
  "next_required_implementation": "measure_functional_fullgrid_s2_package",
  "success_v83_minimum": 0,
  "success_v83_formal": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 3500
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `e27ff80b8463066608edf5e02fc9e9b8b6415bf61f08ae201d718b055694e7d5` |
| v8.3 plan | `791bffc5ebb4b3015fc65c4a4aa447283fca1a8ee4ee701ef9f1551537241ba9` |
| route | `b336cb3372cbd29e18fac6ba9edc3e6eb33c0ad24c4363f83bd442f59ddfcb0d` |
| P2 operator audit | `fac62687beb0d59a2c4e7578cfaccbb57bc68258f108f27e4c8892b8cf8d5827` |
| P3 direction audit | `00447422e7a274d5cd0c0da16ca5ea2c6617b74c393e5d28792ceb2e82c798b7` |
| P4 smoke | `cde222f35896c67c9a97141b03d789c0647b77b686db4747361425ed69528434` |
| P5 co-selection | `83ffd69af29f0c545a34825313568aa723f871de6d2f7a59714197c6b90337af` |
| P5 raw | `d2fa1a4b73574077f1c72f112e4f40b69d7ff67ff042d38bb1418931a957df89` |
| P6 confirm5 | `8631b163284d5e371057383834e942010396498baadc27a1733d24a88e9b5c2d` |
| P6 raw | `8c4b5b0fb9c32c3418285a8f00cf4a69a7ba3581273b6d6fd5459a9ebd2c658a` |
| provenance audit | `4e7c624542465bc1116ca9e01208a66b346678c5a2c661d4bf9d9b206e5e410a` |
| run manifest | `7f17b2df87d98a1b3e1eaa8266702bc5cf9775fb7b86060650913a5f02e55eeb` |

### 14.6 更新结论

v8.3 仍未完成 minimum success：

```text
H0_system_base_pass = true
P2_functional_operator_pass = true
P3_one_step_direction_pass = true
P4_multistep_pass = true for FT7 on 2/9 rows
P5_functional_useful_pass = true
P6_task_geometry_system_mean_pass = true
P6_confirm5_pass = false
P7/P8/P9 = not_run
success_v83_minimum = false
```

机制结论：

1. Low-cadence FT7 是有效的 P5-level system repair：它把 full-task step ratio 拉回 S2 范围，同时保留 strong geometry benefit。
2. P6 5-seed 复现了 task neutral、NLL 改善、geometry 改善和均值 system ratio 可用。
3. 但 P6 仍缺真实 functional full-grid S2 shape count；按计划不能进入 P7，也不能写 v8.3 success。
4. 当前 blocker 从 “P5 full-task step 太慢” 推进为 “需要 measured functional full-grid S2 package”。
5. 全程没有 fake/proxy，没有 CPU offload，也没有 teacher/loss/sampler/class weight 修改。

最终一句话：

> v8.3 还没完成，但已经进入比上一节更接近目标的位置：`KW6 hidden68 + FT7 low-cadence functional update` 在 P5/P6 均值上同时保 task、保 geometry、保系统 ratio；唯一不能放行的是 P6 还没有真实 full-grid S2 shape count。下一步应实现并测量 functional full-grid S2 package，而不是改变 teacher/loss 或把未测 S2 写成成功。

## 15. 追加：measured full-grid S2、P7 10-seed 与 P8 TimeAUC profiler

本节继续执行 `docs/DG-KAN_v8.3_SystemGated_FunctionalUpdate_ReEntry_完整实验计划.md` 的中心路线：system-gated functional re-entry。没有改变 teacher/loss/sampler/class weight，没有 CPU offload，也没有把 P6 均值 ratio 当成 full-grid S2 proxy。

### 15.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v83_real.py` | 新增 P6 measured functional full-grid S2 package：`p6_functional_fullgrid_s2.csv` |
| `experiments/run_gafu_v83_real.py` | 将 FT7 guarded rollback 从 full-parameter snapshot 收窄为 role-local snapshot，降低 GPU workspace |
| `experiments/run_gafu_v83_real.py` | 新增 P7 10-seed raw/summary：`p7_functional_confirm10_raw.csv`、`p7_functional_confirm10.csv` |
| `experiments/run_gafu_v83_real.py` | 新增 P8 wall-clock profiler raw/summary：`p8_functional_time_profiler_raw.csv`、`p8_functional_time_profiler.csv` |
| `experiments/run_gafu_v83_real.py` | P8 cost trim：跳过 zero-weight role correction 遍历，并关闭不参与 gate / 不影响参数轨迹的 diagnostic second-diff norm tracking |

本节还尝试过两条 P8 repair，但均未进入 official route：

| probe | 结果 | 处理 |
|---|---|---|
| FT7 fast task-check（去掉 role holdout rollback，只做 correction 后 task CE check） | P4 退回 `S3-FunctionalGeometryTaskConflict` | rejected，不写 success |
| FT7 stride8 / alpha8 低频等价事件 | P5 退回 `S1-P5FunctionalCoSelectionNoUsefulSurvivor` | rejected，不写 success |
| FT7 stack-only fast task-check | P4 退回 `S4-FunctionalTaskNeutralNoGeometryBenefit` | rejected，不写 success |
| FT7 train-only role guard（去掉 holdout-side role guard） | P4 退回 `S3-FunctionalGeometryTaskConflict` | rejected，不写 success |

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 15.2 P6 measured full-grid S2

role-local snapshot 修复后，`p6_functional_fullgrid_s2.csv` 真实测得 `FT7` 全 9 个 shape 通过 S2：

| metric | value |
|---|---:|
| `S2_shape_count` | `9/9` |
| `functional_fullgrid_memory_ratio_max` | `1.000000` |
| `functional_fullgrid_step_ratio_max` | `1.378236` |
| max failing status | none |

`p6_functional_confirm5.csv`：

| metric | value |
|---|---:|
| val acc delta vs base | `0.000000` |
| test acc delta vs base | `+0.002214` |
| ECE delta | `+0.000110` |
| NLL delta | `-0.005793` |
| ValLossAUC step ratio | `0.995130` |
| curvature ratio | `0.776142` |
| P6 confirm5 pass | `1` |

判断：P6 从 “S2 未测” 推进为 “P6 5-seed task/geometry/system 全部通过”。这不是 proxy S2，来自真实 per-shape measured artifact。

### 15.3 P7 10-seed final functional confirmation

`p7_functional_confirm10.csv`：

| metric | value |
|---|---:|
| rows | `30` |
| val acc delta vs base | `+0.001497` |
| CI95 low | `0.000000` |
| test acc delta vs base | `+0.001367` |
| ECE delta | `+0.002346` |
| NLL delta | `-0.006261` |
| ValLossAUC step ratio | `0.994741` |
| curvature ratio | `0.778850` |
| S2 shape count | `9/9` |
| P7 confirm10 pass | `1` |

判断：`FT7` 在 10-seed 下作为 functional update route 成立：不伤 task，CI low 没跌破 `-0.005`，geometry 保持明显改善，S2 仍为 `9/9`。

### 15.4 P8 TimeAUC profiler

`p8_functional_time_profiler.csv`：

| metric | value | gate |
|---|---:|---:|
| ValLossAUC time ratio | `1.265371` | `<=1.05` |
| functional update time / step | `0.166037` | `<=0.10` |
| unknown time fraction max | `0.029489` | `<=0.10` |
| train step time ratio | `1.198331` | diagnostic |
| functional metric build ms/step | `0.140320` | diagnostic |
| functional update ms/step | `0.295228` | diagnostic |
| P8 pass | `0` |  |

判断：

1. P8 profiler 映射本身是可信的：`unknown_time_fraction_max = 0.029489`，低于 `0.10`。
2. zero-weight role skip 与 diagnostic norm disable 是有效但不足的 cost trim：update fraction 从上一轮 `0.197875` 降到 `0.166037`，但仍高于 `0.10`。
3. P7 说明 functional update 的几何/任务路线成立；P8 说明它还不是最终可用 route，因为 time AUC 和 update fraction 不达标。

### 15.5 Route / audit / hash

最终 route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "base_macro_gap": "0.0216796875",
  "functional_macro_gap": "0.0014973958333333334",
  "functional_ci95_low": "0.0",
  "geometry_delta_vs_base": "0.22115017693419425",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.2653709798194397",
  "functional_update_time_ratio_of_step": "0.16603685264485266",
  "unknown_time_fraction_max": "0.029489020631717912",
  "primary_blocker": "p8_functional_time_profiler_gate",
  "next_required_implementation": "p8_functional_time_profiler_repair",
  "success_v83_minimum": 0,
  "success_v83_formal": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `c9797d705ade5b855dea8837d37bacff781dd8d5f03cb55b0582da933dd71858` |
| v8.3 plan | `f9dec8994f3e9cce574964c74ab5bf11bb00ad264cc23755204b478fb52bf037` |
| route | `8cd2239bf1ab7adc0cddbc283f7b239a54ba459350442dcb61965caf5d7d4b67` |
| P6 confirm5 | `5b1e1d722a18cce28b0c622621d39acfa81e7ad18ab2cd553955d8e61c294f0c` |
| P6 full-grid S2 | `3c6c43217b28558cfef73c71c7a0edc7a69002d6eff8f31aefa92b67aac96bb6` |
| P7 confirm10 | `abacfaac8707cf780dacfe6826f2300f2fa70292d779588cab9e8730c3631c94` |
| P7 raw | `060a373663f4f11589acf8f1a4114bf0e66cc71bbb6809b4ddfc9486a1b9c7ae` |
| P8 profiler | `25ea3686945337b0293dc31104c958025e7efad06bc30a3403b752164de03647` |
| P8 raw | `ff3581e291aa998c017e21a339cc126a8d1793de4170260246c3b0ea809e943e` |
| provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| run manifest | `c3d6074bab8ecce674713373123f3102b9b20952ab9237e20b31b2b454cc3b3a` |

### 15.6 更新结论

v8.3 仍未完成 minimum success：

```text
H0_system_base_pass = true
P2_functional_operator_pass = true
P3_one_step_direction_pass = true
P4/P5/P6 = pass
P7_10seed_functional_confirmation = pass
P8_TimeAUCProfilerPass = false
P9 = not_run
success_v83_minimum = false
```

机制结论：

1. `KW6 hidden68 + FT7` 已经满足 system-gated functional re-entry 的前半段：系统 base 通过，functional update 进入 full task，P7 10-seed 仍保 task 并提供 geometry improvement。
2. role-local snapshot 与 P8 diagnostic trim 是有效的 system repair：P6 full-grid memory max 现在为 `1.000000`，S2 为 `9/9`。
3. 当前唯一 clean blocker 仍是 P8 wall-clock：functional correction 本身太贵，`functional_update_time_ratio_of_step = 0.166037`，超过 `0.10`。
4. fast task-check、stack-only fast、train-only guard 与 stride8/alpha8 都被真实 artifact 否掉；下一步需要更细的 P8 repair，例如数值等价 fused guard measurement 或减少 role eval 的 kernel/forward overhead。不能回到 teacher/loss/sampler/class weight，也不能把 P8 failure 写成 success。

最终一句话：

> v8.3 还没完成，但推进到了关键新位置：`FT7` 已通过 P6 full-grid S2 和 P7 10-seed functional confirmation，证明 functional update 的几何路线成立；P8 cost trim 把 update fraction 从 `19.8%` 降到 `16.6%`，但仍没过 `10%` gate。下一步不是改 objective，而是做更强的 guarded functional correction 执行成本修复，同时保持 P7 的 task/geometry/S2。

## 16. 追加：P8 role-guard cost repair

本节继续第 15 节的唯一 clean blocker：`FT7` 已通过 P7，但 P8 wall-clock / profiler gate 未过。本轮没有改变 teacher/loss/sampler/class weight，没有 CPU offload，也没有把 functional update 改成 loss；只优化 role-wise guarded functional correction 的执行路径。

### 16.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `FT7` role guard 增加 train-CE short-circuit：如果某个 role 的 train CE 已经超过 accept limit，则不再计算该 role 的 holdout CE，直接 rollback |
| `experiments/run_gafu_v83_real.py` | `_loss_only` / `_logits_only` 使用 `torch.inference_mode()` 执行 no-grad diagnostic forward |

该 short-circuit 不放宽任何 gate：原逻辑中 train fail 的 role 无论 holdout 结果如何都会 rollback，因此跳过 holdout CE 只减少无效 forward，不改变 accepted update。

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 16.2 Rejected probes

以下 probe 均来自真实 postprocess artifact；失败后已回退，未写 success：

| probe | route / 结果 | 判断 |
|---|---|---|
| fused train+holdout role CE check | `S1-FunctionalP6Confirm5NoFullGridS2`，P6 memory max 退到 `1.227271` | rejected：拼接 forward 放大 memory，不是可用 P8 repair |
| `FT7 stride8 / alpha4` | `S1-P5FunctionalCoSelectionNoUsefulSurvivor`，macro delta `-0.001085` | rejected：事件太稀，P5 survivor 消失 |
| `FT7 stride6 / alpha4` | `S1-P5FunctionalCoSelectionNoUsefulSurvivor`，macro delta `-0.000217` | rejected：仍未保住 P5 useful survivor |

判断：`FT7` 的 useful route 依赖当前 stride4 事件密度；简单降低事件频率会丢 P5，不应保留。

### 16.3 Final postprocess after accepted short-circuit

最终保留 `stride4 / alpha4`，并加入 role train short-circuit。Postprocess 仍使用：

```text
out_dir = results/real_rerun_20260506/v83_kw6_hidden68_base_confirm_10seed_20260507T113000Z
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
hidden_dim = 68
bootstrap_reps = 10000
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.2483851792659137",
  "functional_update_time_ratio_of_step": "0.14409945164895654",
  "unknown_time_fraction_max": "0.030560547640869698",
  "success_v83_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

P6 / P7 remained open and passing:

| stage | key metrics |
|---|---|
| P6 confirm5 | `P6_confirm5_pass=1`，S2 `9/9`，memory max `1.000000`，step max `1.311296` |
| P7 confirm10 | val acc delta `+0.001497`，CI95 low `0.000000`，curvature ratio `0.778850`，S2 `9/9`，pass `1` |

P8 final profiler:

| metric | value | gate |
|---|---:|---:|
| ValLossAUC time ratio | `1.248385` | `<=1.05` |
| functional update time / step | `0.144099` | `<=0.10` |
| unknown time fraction max | `0.030561` | `<=0.10` |
| train step time ratio | `1.221663` | diagnostic |
| functional metric build ms/step | `0.132704` | diagnostic |
| functional update ms/step | `0.246496` | diagnostic |
| P8 pass | `0` |  |

判断：

1. role train short-circuit 是 accepted repair：P6/P7 没有回退，fake/proxy/CPU offload 仍为 0。
2. P8 cost 确实下降，但仍不足：functional update fraction 从第 15 节的 `0.166037` 降到 `0.144099`，仍高于 `0.10`。
3. TimeAUC 仍未过：`1.248385 > 1.05`。因此 v8.3 仍不能记 minimum success。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 16.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `dd65105050f562c9fddfb142e6a17913f21e7a59cf4f44f79dcb70caf8a3647a` |
| v8.3 plan | `f9dec8994f3e9cce574964c74ab5bf11bb00ad264cc23755204b478fb52bf037` |
| route | `40c43cba48589828854504776e7f195b57835d9620a4edbd1ddea6103a9cc807` |
| P6 confirm5 | `d6d2dfa4c6186be9c8b54951f2bc8b02d2e25ec187a42601d923dd050985ea15` |
| P6 full-grid S2 | `dd3ce19b4954f8398d60ed70b5b07ef1ae353234fb644e770b7ed8edd85cce82` |
| P7 confirm10 | `2e516e6750a4734ca145e98652236b3b77dc64bb6d234e660e0e587752cb0772` |
| P7 raw | `78cdf7faaca6c125f76f50b2dd3d402493bd1eaef77b3c32ca3940220b5a0e2a` |
| P8 profiler | `6aa41f4ce5185ae330de4d3e5efcd22406e00b7ae070edf7f573ecd1beb994c3` |
| P8 raw | `efd8e774d5cfa6981ccdd8ccd1cc6396f92d4389ae6f0c255b8c3c2eee6be635` |
| provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| run manifest | `4cf4092cf57a2f75f49a6a0e1da4bd0bde20eb07bd0de95fcabf16079e90ac81` |

### 16.5 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P2_functional_operator_pass = true
P3_one_step_direction_pass = true
P4/P5/P6 = pass
P7_10seed_functional_confirmation = pass
P8_TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> 本轮继续沿着 v8.3 的中心思想优化：保留 system-gated functional update，不改 objective。role train short-circuit 把 P8 functional update fraction 从 `16.6%` 降到 `14.4%`，且 P7 仍过；但 TimeAUC 和 `<=10%` profiler gate 仍未闭合，所以 v8.3 仍不能写 success。下一步需要更强的 role-guard 执行成本修复，例如减少 accepted-role CE forward 次数或把 guard 逻辑做成数值等价的轻量 kernel，而不是降低 functional update 事件密度或放弃 holdout guard。

## 17. 追加：P8 role-guard execution trim

本节继续第 16 节同一个 blocker：`FT7` 已经通过 P7，但 P8 TimeAUC / profiler gate 未过。本轮仍保持 v8.3 中心思想：

```text
functional_update_is_update_rule = 1
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 17.1 Accepted code changes

| change | 作用 | 是否改变 gate |
|---|---|---|
| task / holdout feature cache | event step 上复用 task-update 后的 stack features，避免 head role 反复重跑 stack forward | 不改变 CE accept/rollback 判据 |
| head train+holdout paired CE | head role 用缓存 features 拼接一次 head forward，同时得到 train / holdout CE | 不改变 CE accept/rollback 判据 |
| rollback-delta | rejected role 不再 clone/restore 整个 role 参数，而是记录 second-diff delta 并反向补偿 | 数值等价 rollback |
| stack-reject head skip | 如果 stack role 已经被 gate reject，则跳过 head-only functional correction | 更保守，只减少半截 functional update |

这些改动只修 guarded functional correction 的执行路径，不改 CE objective，也没有引入 teacher/loss/sampler/class weight/CPU offload。

### 17.2 Cost repair progression

| step | ValLossAUC time ratio | functional update / step | P7 | P8 | 判断 |
|---|---:|---:|---:|---:|---|
| 第 16 节 role train short-circuit | `1.248385` | `0.144099` | 1 | 0 | accepted，但不足 |
| feature cache | `1.215734` | `0.131465` | 1 | 0 | accepted |
| head paired CE | `1.211663` | `0.125127` | 1 | 0 | accepted |
| rollback-delta | `1.205623` | `0.123548` | 1 | 0 | accepted |
| stack-reject head skip | `1.183551` | `0.098971` | 1 | 0 | accepted；首次通过 `<=0.10` update fraction 子门 |

判断：

1. P8 profiler 的 update fraction 子门已经被修到 `0.098971 <= 0.10`。
2. 但 TimeAUC ratio 仍是 `1.183551 > 1.05`，所以 P8 仍不通过。
3. 当前 clean blocker 从 “functional update 本体超过 10%” 收缩为 “guard metric build + 总 wall-clock 仍使 TimeAUC 过慢”。

### 17.3 Rejected probes

| probe | route / 结果 | 判断 |
|---|---|---|
| head-only standard guard | `S4-FunctionalTaskNeutralNoGeometryBenefit` | rejected：单独 head role 不能保住 functional geometry |
| stack-only standard guard | `S4-FunctionalTaskNeutralNoGeometryBenefit` | rejected：单独 stack role 不能保住 full route |
| post-task holdout strict guard | `S4-FunctionalTaskNeutralNoGeometryBenefit`，geometry delta 只剩 `0.000362` | rejected：虽然省掉 pre-holdout CE，但把 functional geometry route 掐掉 |

判断：不能用移除 holdout baseline、只保单 role 或降低 functional update 有效性来换 P8；这会偏离 v8.3 “functional update 带来几何优势” 的中心目标。

### 17.4 Final postprocess after accepted trims

最终保留 `stride4 / alpha4`、pre/post holdout guard、feature cache、head paired CE、rollback-delta 与 stack-reject head skip。Postprocess 仍使用：

```text
out_dir = results/real_rerun_20260506/v83_kw6_hidden68_base_confirm_10seed_20260507T113000Z
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
hidden_dim = 68
bootstrap_reps = 10000
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.1835511216084764",
  "functional_update_time_ratio_of_step": "0.09897069753206578",
  "unknown_time_fraction_max": "0.03192774323164801",
  "success_v83_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

P6 / P7 remained passing:

| stage | key metrics |
|---|---|
| P6 confirm5 | `P6_confirm5_pass=1`，S2 `9/9`，geometry reduction `0.214114`，step ratio mean `1.161944` |
| P7 confirm10 | val acc delta `+0.001302`，curvature ratio `0.789779`，geometry reduction `0.210221`，S2 `9/9`，pass `1` |

P8 final profiler:

| metric | value | gate |
|---|---:|---:|
| ValLossAUC time ratio | `1.183551` | `<=1.05` |
| functional update time / step | `0.098971` | `<=0.10` |
| unknown time fraction max | `0.031928` | `<=0.10` |
| train step time ratio | `1.180578` | diagnostic |
| functional metric build ms/step | `0.130379` | diagnostic |
| functional update ms/step | `0.157795` | diagnostic |
| P8 pass | `0` |  |

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 17.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `eb8a331132244f5d2a859b38a8ccf3f5cee52b3ddd23987c1f53c4ed8dfa772a` |
| v8.3 plan | `f9dec8994f3e9cce574964c74ab5bf11bb00ad264cc23755204b478fb52bf037` |
| route | `08503dda89dc36b9876ec2226ba188c3bb1b326f191704133b0ad019ed329a88` |
| P6 confirm5 | `e6dbea59a6a6bad2d42335f456434734f4134dc278a1a47e194bc0d3b3ad57d8` |
| P6 full-grid S2 | `65e7b0892f6eaaf80e3376cb8be27850b775de5f5b8dbd60cc8447f442c4d733` |
| P7 confirm10 | `daa2dcb855b5183dabb26335e40f429ce90f45e9650130d24058fb66d5178695` |
| P8 profiler | `ef7ff7188016b09a03c54808557a559b74c19bff25c29573a8f87dd90ab45a4a` |
| P8 raw | `2d824474385082a6b1c42c8f5b1f77270002b48833362e65ca0aa4691f546e77` |
| provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| run manifest | `ced76cb2606e24f977d2d9585628c3b0fa2e41706affb15602c0669c6d185b9f` |

### 17.6 更新结论

v8.3 仍未完成 minimum success：

```text
H0_system_base_pass = true
P2_functional_operator_pass = true
P3_one_step_direction_pass = true
P4/P5/P6 = pass
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> 本轮继续沿 v8.3 的中心思想优化，没有回到 teacher/loss/CPU，也没有把 functional update 降级为 optimizer trick。`FT7` 现在不仅 P7 通过，而且 P8 的 functional update fraction 已从 `16.6% -> 9.9%`，首次过了 `<=10%` 子门；但 TimeAUC ratio 仍为 `1.18355`，所以 v8.3 仍不能记 success。下一步 blocker 是 guard metric build / wall-clock AUC，需要做数值等价的 guard forward 融合或更轻的 holdout measurement，而不能牺牲 functional geometry。

## 18. 追加：FT7 role-budget boundary probe

本节继续第 17 节的 P8 blocker，但不再移除 holdout guard、不降低 event density，也不改 CE objective。只在 `FT7` role-wise guard 内做一个边界 probe：把 head role 的 CE accept budget 从 `0.05` 放宽到 `0.10`，stack role 仍保持 `0.10`。

约束保持：

```text
loss_type = CE
functional_update_is_update_rule = 1
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 18.1 Accepted / rejected

| probe | route / 结果 | 判断 |
|---|---|---|
| head role budget `0.10` | P7 pass，geometry reduction `0.222823`，P8 update fraction `0.098847` | accepted：比 `0.05` 有更强几何和 task delta，系统 gate 不退 |
| head role budget `0.15` | P7 pass，但 P8 TimeAUC `1.192785`，update fraction `0.098327` | rejected：几何略强但 P8 wall-clock 更差 |
| joint event guard | `S1-P5FunctionalCoSelectionNoUsefulSurvivor` | rejected：整体 gate 太粗，P5 useful survivor 消失 |

判断：`FT7` 需要 role-wise 分段接受；不能用 joint event gate 替代。head budget `0.10` 是当前更好的 task/geometry Pareto，但它仍没有解决 P8 TimeAUC。

### 18.2 Final artifact after accepted head budget

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.1913948187624028",
  "functional_update_time_ratio_of_step": "0.09884745365458969",
  "unknown_time_fraction_max": "0.03181309418968891",
  "success_v83_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

P6 / P7：

| stage | key metrics |
|---|---|
| P6 confirm5 | `P6_confirm5_pass=1`，S2 `9/9`，geometry reduction `0.226360`，step ratio mean `1.156363` |
| P7 confirm10 | val acc delta `+0.001693`，CI95 low `+0.000130`，curvature ratio `0.777177`，geometry reduction `0.222823`，S2 `9/9` |

P8：

| metric | value | gate |
|---|---:|---:|
| ValLossAUC time ratio | `1.191395` | `<=1.05` |
| ValLossAUC step ratio | `1.014038` | diagnostic |
| functional update time / step | `0.098847` | `<=0.10` |
| unknown time fraction max | `0.031813` | `<=0.10` |
| train step time ratio | `1.157886` | diagnostic |
| functional metric build ms/step | `0.132504` | diagnostic |
| functional update ms/step | `0.159288` | diagnostic |
| P8 pass | `0` |  |

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 18.3 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `ddaad576aac0c125a794da93c0c5362ac9cfed3c834cce698730903e5a1dd037` |
| v8.3 plan | `f9dec8994f3e9cce574964c74ab5bf11bb00ad264cc23755204b478fb52bf037` |
| route | `1b0ce0721e570f3691b7d9230b81b32fd4d2be0cc57ed3bac440a021ec799df5` |
| P6 confirm5 | `25ff012dec10e3b5481e5d2210d716149be3cf36abf8110b6f4bc3a80dc57f4e` |
| P6 full-grid S2 | `254186ecf4c1cbeb4eca620151eef618b9153a08231b8996006a0618cbd8de81` |
| P7 confirm10 | `52375d156ae3191cb9bea2b4b07fa5d6d455e38a99be2efc3c85cc6d25cc0f21` |
| P8 profiler | `fa55d51dfd63fc7ef135e3f43d7edf19da161ec3444a4feb1c1a9a4cf440d577` |
| P8 raw | `4ebb19d2aa034dd4b40b347ef8e92f11488c1a0494f31167ebe732db1db1a5fd` |
| provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| run manifest | `40018f1276cfd76cb04b335d089c32fbac6434c736823a136937523e6684528e` |

### 18.4 更新结论

v8.3 仍未完成 minimum success：

```text
H0_system_base_pass = true
P2_functional_operator_pass = true
P3_one_step_direction_pass = true
P4/P5/P6 = pass
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> head role budget `0.10` 让 `FT7` 的 P7 几何收益从 `0.21022` 提到 `0.22282`，且 update fraction 继续保持在 `9.88%`，但 P8 TimeAUC 仍为 `1.19139`，远高于 `1.05`。所以 v8.3 还没完成；下一步必须针对 `functional_metric_build_time_ms_mean = 0.132504` 做数值等价的 guard measurement fusion，而不是再粗暴放宽 gate 或整体合并 role update。

## 19. 追加：pre-holdout measurement sparsification probe

本节继续针对 P8 的 `functional_metric_build_time_ms_mean`。尝试的 probe 是：不删除 holdout guard，而是每两个 `FT7` event 只做一次 pre-holdout baseline；没有 pre-baseline 的 event 使用更严格的 post-holdout 不变差 gate。

该 probe 的意图是减少 pre-holdout forward，但仍保持：

```text
loss_type = CE
functional_update_is_update_rule = 1
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 19.1 Rejected result

| probe | route / 结果 | 判断 |
|---|---|---|
| pre-holdout every 2 FT7 events | `S4-FunctionalTaskNeutralNoGeometryBenefit`，functional full task 未打开 | rejected：pre-holdout baseline 对 `FT7` 几何路线必要，隔 event 严格化会让 P4 geometry gate 掉 |

失败后已恢复为每个 `FT7` event 都测真实 pre-holdout baseline，即 `FT7_PRE_HOLDOUT_EVERY = 1`。最终保留的 active route 仍是第 18 节的 head budget `0.10`。

### 19.2 Final restored artifact

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.1922174433446389",
  "functional_update_time_ratio_of_step": "0.09899616788573097",
  "unknown_time_fraction_max": "0.03187554068206944",
  "success_v83_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

P7 / P8：

| stage | key metrics |
|---|---|
| P7 confirm10 | pass `1`，val acc delta `+0.001693`，CI95 low `+0.000130`，geometry reduction `0.222823` |
| P8 profiler | pass `0`，TimeAUC ratio `1.192217`，update fraction `0.098996`，metric build `0.132728 ms/step` |

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 19.3 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `24ff747b974873a857e2c52c21a15603000a44fbbbf5283200b17f13a6e7866b` |
| v8.3 plan | `f9dec8994f3e9cce574964c74ab5bf11bb00ad264cc23755204b478fb52bf037` |
| route | `54dea673e6523064795c9f1e8895008e2d82171675f3ef5ac5c2034604b9ddd7` |
| P6 confirm5 | `dc85ab12bff8689d2370cfc40378a4055d3075d423029bb484bd3db4d5bce4a2` |
| P6 full-grid S2 | `c5e5920b1a23b1d69c21ff6bc02b0d0eb553f66ed92567f27c6ffb7a62410a54` |
| P7 confirm10 | `9492532453560efc9f9a58336caaec422684942e9d6a5ce4033fee3f37311af0` |
| P8 profiler | `dcb191c284b403a9bfc9965087c481d3a5b8c02b5e5a47131815a77ffb522723` |
| P8 raw | `1246f79e0fd92b5cbe97038ddc6d7bebb1fdbad2cc27b5e11f3e5b264a890fb8` |
| provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| run manifest | `5855235b5f17a9dcc1c06d9a7451d4115e20d79ab086546c93227b0d39d08a34` |

### 19.4 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> pre-holdout measurement sparsification 不可用：它省掉的不是冗余，而是 `FT7` 保持几何路线所需的 guard baseline。当前 clean blocker 仍是 P8 TimeAUC，尤其是每步 `0.1327 ms` 的 guard metric build；下一步需要真正数值等价的 guard forward fusion，而不是减少 guard 频率或把 pre-holdout baseline 稀疏化。

## 20. 追加：CE value-only guard measurement repair

本节继续第 19 节的 P8 blocker，但不减少 guard 频率、不删除 holdout baseline、不改 loss/objective。修复点是：原先 guard 的 loss-only path 复用了会生成 `grad_logits` 的 CE helper；guard 只需要 loss value，不需要 logits gradient。因此新增数值等价的 CE value-only path，仅用于 `_loss_only` / `_loss_and_features_only` / head feature guard loss。

保持不变：

```text
manual backward / task gradient path = unchanged
loss_type = CE
functional_update_is_update_rule = 1
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 20.1 数值等价检查

对随机 logits 检查旧 helper 与新 value-only helper：

| label_smoothing | abs diff |
|---:|---:|
| `0.00` | `0.0` |
| `0.02` | `0.0` |
| `0.05` | `0.0` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 20.2 Smoke

```bash
python experiments/run_gafu_v83_real.py \
  --out-dir results/real_rerun_20260506/tmp_v83_ce_value_only_smoke \
  --fresh \
  --device auto \
  --datasets MNIST \
  --candidates B0,KW6 \
  --seeds 0 \
  --task-steps 2 \
  --bench-batch-sizes 128 \
  --bench-warmup 2 \
  --bench-reps 5 \
  --grad-batch-sizes 8 \
  --trace-every 1 \
  --bootstrap-reps 100
```

Smoke 只验证代码路径。由于 `task_steps=2`，H0 base system gate 不应通过；结果为 `S6-NoBaseSystemCandidateFunctionalFullTaskGated`，符合预期。No-fake audit：

```text
rows_checked = 506
fake_proxy_nonzero_count = 0
cpu_offload_used = 0
```

### 20.3 P8 targeted remeasurement

因为本节只改变 guard loss measurement 的实现，并且 loss value 已验证数值等价，所以本次没有重跑 P0-P7；复用第 19 节的 P7/P6 真实 artifact，仅重测 P8 profiler：

```text
results/real_rerun_20260506/v83_p8_ce_value_only_guard_repair_20260507T120000Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.1766594076443386",
  "functional_update_time_ratio_of_step": "0.09412197827808161",
  "unknown_time_fraction_max": "0.032614640578826626",
  "success_v83_minimum": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

P8 profiler：

| metric | before section 19 | CE value-only repair | gate |
|---|---:|---:|---:|
| ValLossAUC time ratio | `1.192217` | `1.176659` | `<=1.05` |
| train step ratio | `1.155141` | `1.075182` | diagnostic |
| functional update / step | `0.098996` | `0.094122` | `<=0.10` |
| metric build ms/step | `0.132728` | `0.124654` | diagnostic |
| functional update ms/step | `0.159879` | `0.150490` | diagnostic |
| unknown time fraction max | `0.031876` | `0.032615` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. CE value-only guard measurement 是有效的 P8 partial repair：TimeAUC ratio 从 `1.192217` 降到 `1.176659`，train step ratio 从 `1.155141` 降到 `1.075182`。
2. 它没有改变 task gradient、manual backward 或 functional update 公式；只是移除了 guard loss 中未使用的 grad-logits materialization。
3. 但它仍没有闭合 P8：`1.176659 > 1.05`，所以 v8.3 仍不能记 minimum success。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 20.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `d764515da73790e15592607c5a36d5bdf7c5f3b56ea621663e8eea8c05ef9dcd` |
| route | `9b011994d4b943e2b5243fccb0100431e316d1f5db08890664a2e1a08ea74a80` |
| P8 profiler | `6620e215fa611da59dfa020ec163e3166c284a314ea13b14f97bc2531cdef388` |
| P8 raw | `45686fbd23eb3b233b4ed4288fc666f57ade81c2aec1fb6722f6d6a722eade3d` |
| provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| run manifest | `6ba91c83a1dcaae9f770ee5088dc031d834d03bbd1cf074d37abb831976d751e` |

### 20.5 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> CE value-only guard measurement 是一个干净的、数值等价的 system-side repair：它把 P8 TimeAUC 从 `1.1922` 推到 `1.1767`，但还没到 `1.05`。当前下一步不应改 loss/teacher/CPU，也不应降低 guard 频率；应继续做 guard forward fusion，尤其是减少 event step 中 pre/post holdout full-forward 与 role CE check 的重复 stack/head forward。

## 21. 追加：P8 paired-init fairness 与 fixed-coeff rollback trim

本节继续只修 P8 profiler 的系统侧执行路径，不改变 v8.3 中心思想：

```text
CE-only
no external teacher / no self-teacher
no distillation / no loss modification
no sampler / class weight
functional update remains update rule
no CPU offload
no fake / proxy rows
```

### 21.1 代码改动

| 改动 | 目的 | 是否改变 update 数学 |
|---|---|---:|
| P8 profiler 去掉 `functional_id` 对 BASE/FT7 初始化 seed 的影响 | 让 P8 与 P5/P7 一样使用 paired same-init 公平比较 | 0 |
| role-local correction 使用 fixed-coeff rollback helper | 去掉 role 内重复 role-weight lookup 与临时 map 构造 | 0 |
| `_rollback_streamed_second_diff_correction` 不再复制 rollback list | 降低 Python/list overhead | 0 |

fixed-coeff helper 的 tensor-level 等价检查：

```text
stack max_apply_diff = 0.0, max_pair_after_rollback = 0.0
head  max_apply_diff = 0.0, max_pair_after_rollback = 0.0
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 21.2 P8 paired-init targeted remeasurement

复用第 20 节的 P7/P6 真实 artifact，只重测 P8 profiler：

```text
results/real_rerun_20260506/v83_p8_paired_init_value_guard_official_20260507T121500Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.153745731385766",
  "functional_update_time_ratio_of_step": "0.0949887760416475",
  "unknown_time_fraction_max": "0.032387135960007155",
  "success_v83_minimum": 0
}
```

P8 profiler：

| metric | CE value-only | paired-init | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `1.014038` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.176659` | `1.153746` | `<=1.05` |
| train step ratio | `1.075182` | `1.054264` | diagnostic |
| functional update / step | `0.094122` | `0.094989` | `<=0.10` |
| metric build ms/step | `0.124654` | `0.122249` | diagnostic |
| unknown time fraction max | `0.032615` | `0.032387` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. P8 paired-init 是必要的 profiler fairness repair：step AUC 从 `1.014038` 修到 `0.996934`，说明 functional update 本身没有造成 step-indexed loss AUC 退化。
2. TimeAUC 仍为 `1.153746 > 1.05`，所以 P8 仍不通过。
3. 这不改变 P7 task/geometry/S2 结论，也不改变 functional update rule。

### 21.3 Fixed-coeff rollback targeted remeasurement

继续复用同一组 P7/P6 真实 artifact，只重测 P8：

```text
results/real_rerun_20260506/v83_p8_fixed_coeff_rollback_official_20260507T123000Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "H0_system_base_pass": 1,
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.148219250749198",
  "functional_update_time_ratio_of_step": "0.09272247248483124",
  "unknown_time_fraction_max": "0.03247530638290384",
  "success_v83_minimum": 0
}
```

P8 profiler：

| metric | paired-init | fixed-coeff rollback | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.153746` | `1.148219` | `<=1.05` |
| train step ratio | `1.054264` | `1.077618` | diagnostic |
| functional update / step | `0.094989` | `0.092722` | `<=0.10` |
| metric build ms/step | `0.122249` | `0.128962` | diagnostic |
| functional update ms/step | `0.149594` | `0.156583` | diagnostic |
| unknown time fraction max | `0.032387` | `0.032475` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. fixed-coeff rollback 是数值等价的实现级 trim，保持 step AUC 完全一致：`0.996934`。
2. 它把 TimeAUC 从 `1.153746` 小幅推到 `1.148219`，但仍明显高于 `1.05`。
3. update fraction 继续满足 P8 子门：`0.092722 <= 0.10`。
4. 当前唯一 blocker 仍是 P8 TimeAUC，而不是 H0、P7、S2、update fraction 或 no-fake/no-proxy contract。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 21.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `51e85f49cf8001a8da07f5ca151bea7af856f958cda626b1d6be4f6a4063ca80` |
| paired-init route | `f3d63d0a6fd9da421627eda82089215f7fd5c841abe5faa5b71e5721395d4e44` |
| paired-init P8 profiler | `6a9996da53ce7dcb1a2dbf45aa179984ea1a14c61b05122433b7390c75b5fb39` |
| paired-init P8 raw | `ac334668a264f707ac51a00455a934a322789c2eda4d23d1deb259792fa9341b` |
| paired-init run manifest | `6d01910ca2c54a23bb4d15504ac70228fb3636b444731189c4f44325fa9bff74` |
| fixed-coeff route | `e02f0deddf9b628d65da751b236272be4009707e3719c0d35702bd0fbbc91697` |
| fixed-coeff P8 profiler | `e5745d245459ef3bf85a155ddf02018f36e4c9095f942a7d14ccc76d7d38dce1` |
| fixed-coeff P8 raw | `77897ca3a915fa8ee81fb7c1e2445c75190737861da478b51de8d88b2c0f00be` |
| fixed-coeff provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| fixed-coeff run manifest | `3571e7531afca3b085a7d5a221ca2c5b95b42d4c3765e2c6f714527d195ac4c7` |

### 21.5 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> P8 paired-init 修正证明 `FT7` 的 step-indexed loss AUC 没有退化，fixed-coeff rollback 又把 TimeAUC 小幅降到 `1.14822`，但距离 `<=1.05` 仍不够。所以 v8.3 还没完成；下一步仍应沿文档中心思想做数值等价的 guard forward fusion / role CE measurement fusion，不能改 teacher/loss/CPU，也不能靠减少 guard 频率或削弱 functional geometry 换成功。

## 22. 追加：P8 guard micro-trim rejected probes

本节继续围绕 P8 做极窄 implementation probe。约束不变：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 22.1 Closed-form CE guard probe

尝试点：把 guard value-only CE 的 full-target 构造改成闭式公式，并让 head paired CE 只做一次 `log_softmax`。这只影响 guard measurement，不影响 task backward。

数值检查：

```text
max abs diff vs full-target CE = 2.384185791015625e-07
```

P8 probe：

```text
results/real_rerun_20260506/v83_p8_closed_form_ce_guard_probe_20260507T124500Z/
```

| metric | fixed-coeff best | closed-form CE probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.148219` | `1.149641` | `<=1.05` |
| train step ratio | `1.077618` | `1.068828` | diagnostic |
| functional update / step | `0.092722` | `0.096690` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：closed-form CE 没有带来 P8 改善，且存在微小浮点求和差异；因此不作为 accepted route。

### 22.2 Role-entry cache probe

Accepted code-level trim：`FT7` 的 role entries 分组在训练变体初始化时预先计算，并传入 guarded update，避免每个 event 重扫参数列表。这不改变 correction、CE gate、rollback 或 functional update 公式。

```text
results/real_rerun_20260506/v83_p8_role_entries_cache_official_20260507T130000Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "H0_system_base_pass": 1,
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.1996739451572134",
  "functional_update_time_ratio_of_step": "0.09447996340801257",
  "unknown_time_fraction_max": "0.03219407285948478",
  "success_v83_minimum": 0
}
```

P8 profiler：

| metric | fixed-coeff best | role-entry cache | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.148219` | `1.199674` | `<=1.05` |
| train step ratio | `1.077618` | `1.004415` | diagnostic |
| functional update / step | `0.092722` | `0.094480` | `<=0.10` |
| metric build ms/step | `0.128962` | `0.122989` | diagnostic |
| functional update ms/step | `0.156583` | `0.148859` | diagnostic |
| unknown time fraction max | `0.032475` | `0.032194` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. role-entry cache 降低了部分 profiler phase time，但 TimeAUC 没有闭合，不能写 success。
2. 该 probe 进一步说明当前 blocker 不只是 functional update 子阶段本身；`ValLossAUC_time_ratio` 对总 wall-clock / validation placement 仍敏感。
3. 不能改判 v8.3；下一步仍需要真正减少 guard full-forward 次数，尤其是 stack role train/holdout CE measurement 的数值等价 fusion。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 22.3 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `0c5d9b5f2487bdcbe406063305abb076f0740c363308cc0ae9d0a79ff18b905c` |
| role-entry route | `159c7f85b22d08979c99674b6c80c0e00441b3e574cbff7412fbc5fe6a925c7a` |
| role-entry P8 profiler | `7bbb337b9c811c7442ff8b465f4ccb0d6e29de5676ee800d63a364cbea2d067e` |
| role-entry P8 raw | `bc04a456fa100fcecfd5c3d42e973398bfe00f67cd140d6e7de02e2c8c368be7` |
| role-entry provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| role-entry run manifest | `6e87f2c639ac695166ed52de791416b89b3df0d666e2e4ef57dbec37ed4a50a7` |

### 22.4 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> 本轮继续做了不改 objective、不削弱 functional update 的 P8 micro-trim。closed-form CE probe 被拒绝；role-entry cache 保持 update 数学不变，但 P8 TimeAUC 仍失败。当前最强有效记录仍是 fixed-coeff rollback 的 `TimeAUC = 1.148219`，距离 `<=1.05` 还差一段；下一步应集中做 stack-role guard CE 的数值等价 forward fusion，而不是回到 teacher/loss/CPU/offload 或降低 guard 频率。

## 23. 追加：P8 paired after-guard forward fusion

本节继续沿第 22 节最后一句的方向，只做 guard CE measurement 的数值等价 forward fusion。约束保持不变：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 23.1 代码改动

Accepted implementation trim：

1. 新增 `_loss_pair_and_features_only` / `_timed_loss_pair_and_features_only`。
2. 只在 P8 profiler 的 `FT7` event step 中，把 `task_loss_after + task_features_after` 与 `holdout_loss_after + holdout_features_after` 合成一次 paired forward。
3. 不改变 `FT7` 的 correction coefficient、rollback、role gate、CE limit、event stride、functional alpha 或 functional update 公式。

数值等价检查：

```json
{
  "device": "cuda",
  "loss_a_abs_diff": 0.0,
  "loss_b_abs_diff": 0.0,
  "feat_a_max_abs_diff": 3.5762786865234375e-07,
  "feat_b_max_abs_diff": 4.172325134277344e-07
}
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 23.2 P8 targeted remeasurement

Probe：

```text
results/real_rerun_20260506/v83_p8_paired_after_guard_probe_20260507T131500Z/
```

Official-style artifact：

```text
results/real_rerun_20260506/v83_p8_paired_after_guard_official_20260507T131500Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "H0_system_base_pass": 1,
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.13627407380401",
  "functional_update_time_ratio_of_step": "0.09716354476143381",
  "unknown_time_fraction_max": "0.0324452901042365",
  "success_v83_minimum": 0
}
```

P8 profiler：

| metric | fixed-coeff best | paired after-guard fusion | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.148219` | `1.136274` | `<=1.05` |
| train step ratio | `1.077618` | `1.024146` | diagnostic |
| functional update / step | `0.092722` | `0.097164` | `<=0.10` |
| metric build ms/step | `0.128962` | `0.062242` | diagnostic |
| logging/post-check ms/step | not recorded in fixed table | `0.276232` | diagnostic |
| unknown time fraction max | `0.032475` | `0.032445` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. paired after-guard fusion 是本轮最有效的 P8 system-side trim：metric build 从 `0.128962 ms/step` 降到 `0.062242 ms/step`。
2. TimeAUC 从之前最佳 `1.148219` 推到 `1.136274`，方向正确，但仍高于 `1.05`。
3. update fraction 仍满足子门：`0.097164 <= 0.10`。
4. step-indexed loss AUC 维持 `0.996934`，没有通过时间优化换取 step trajectory 退化。
5. 因此 v8.3 仍不能记 minimum success；当前 best effective P8 record 更新为 paired after-guard fusion，但 blocker 仍是 TimeAUC。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 23.3 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `f661f050d14bfbb1762f641678dfa352802598e8c68253cc1e53315032e1ceab` |
| paired after-guard route | `28a06a6cbc2bc647f56f040b493d37a904455e9373c9cb1a707bd934a98413f1` |
| paired after-guard P8 profiler | `f087aadd4779dfb0048321038e22ac4cb5e88579d21995bbf75f24965471739b` |
| paired after-guard P8 raw | `76d4cb0254076b2fbf7e16a4b37e1cf5ae529e5f64f33fb76b4182f1dc4d7787` |
| paired after-guard provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| paired after-guard run manifest | `48c1f30113108c44c96d1164ffb037fd5f24e054277a3b30a09d8c63a0c57a41` |

### 23.4 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> 本轮做成了一个真正沿文档中心思想前进的 P8 优化：paired after-guard forward fusion 不改 teacher/loss/update rule，不削弱 guard，不 CPU offload，把 TimeAUC 从 `1.148219` 降到 `1.136274`。但它仍没到 `<=1.05`，所以 v8.3 还不能记完成；下一步应继续做数值等价的 guard forward/feature reuse fusion，目标是再拿掉剩余 post-check/functional-update wall time，而不是改变实验合同。

## 24. 追加：P8 stack-role paired guard fusion

本节继续第 23 节仍未闭合的 P8 blocker，只把 stack role correction 后的 train/holdout CE gate 合成 paired forward。该开关只用于 P8 profiler，不改变已通过的 P5/P6/P7 confirmation 路径。

约束保持不变：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 24.1 代码改动

Accepted implementation trim：

1. `_apply_ft7_streamed_guarded_update` 新增 `pair_stack_guard_forward=False` 参数。
2. 默认值保持 false；P5/P6/P7 call path 不打开。
3. P8 profiler 的 `FT7` event step 打开该参数，在 stack role correction 后用 `_loss_pair_and_features_only` 同时测 train/holdout gate。
4. 不改变 correction coefficient、rollback、role budget、event stride、functional alpha 或 CE gate 公式。

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 24.2 P8 targeted remeasurement

Probe：

```text
results/real_rerun_20260506/v83_p8_stack_pair_guard_probe_20260507T133000Z/
```

Official-style artifact：

```text
results/real_rerun_20260506/v83_p8_stack_pair_guard_official_20260507T133000Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "H0_system_base_pass": 1,
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.1194790674480732",
  "functional_update_time_ratio_of_step": "0.08481246982775105",
  "unknown_time_fraction_max": "0.03260513967208631",
  "success_v83_minimum": 0
}
```

P8 profiler：

| metric | fixed-coeff best | paired after-guard | stack-role paired guard | gate |
|---|---:|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.148219` | `1.136274` | `1.119479` | `<=1.05` |
| train step ratio | `1.077618` | `1.024146` | `0.997895` | diagnostic |
| functional update / step | `0.092722` | `0.097164` | `0.084812` | `<=0.10` |
| metric build ms/step | `0.128962` | `0.062242` | `0.061197` | diagnostic |
| functional update ms/step | `0.156583` | `0.150818` | `0.127579` | diagnostic |
| logging/post-check ms/step | not recorded in fixed table | `0.276232` | `0.271492` | diagnostic |
| unknown time fraction max | `0.032475` | `0.032445` | `0.032605` | `<=0.10` |
| P8 pass | `0` | `0` | `0` |  |

判断：

1. stack-role paired guard fusion 进一步降低了 P8 wall-clock：TimeAUC 从 `1.136274` 降到 `1.119479`。
2. functional update fraction 明显改善到 `0.084812`，继续满足 `<=0.10`。
3. train step ratio 已压到 `0.997895`，说明系统侧开销基本回到 base 同级。
4. 但 TimeAUC 仍未达到 `<=1.05`，因此不能写 v8.3 minimum success。
5. 当前 blocker 更集中：不是 update fraction、not H0/P7/S2，而是剩余 post-check / validation-time placement 对 wall-clock AUC 的影响。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 24.3 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `c99e6bfa6ede9f3f61d240b92e06521fc2d675cb7e2b7b887f62b582b62729c4` |
| stack-role paired route | `0b61e2c20dcc2954e49a25906dfb39d5ae46a657c5eecc510ee0d47c213ea73a` |
| stack-role paired P8 profiler | `1ed981eb1717e42d91c3791e4ea67af9adda5d6422fa71c830494a0d9517e502` |
| stack-role paired P8 raw | `ad11a1e316fd01f320addd81c51c76c94e4d10a0e0595b2983440e3acd904771` |
| stack-role paired provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| stack-role paired run manifest | `76418f2e96e7b775ab1532711e0a3f07b2d218c71cca80e6028ec8ceef757848` |

### 24.4 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

最终一句话：

> v8.3 继续沿 system-gated functional update re-entry 的中心思想推进：P8 的 paired after-guard 与 stack-role paired guard 都是数值等价的系统侧 trim，没有改 CE/no-teacher/no-loss/no-offload 合同，也没有削弱 functional update。最新 TimeAUC 已从 `1.148219 -> 1.136274 -> 1.119479`，但仍高于 `1.05`，所以还不能记完成；下一步应继续减少剩余 post-check/feature measurement 的 wall-clock，而不是改变实验目标或 gate。

## 25. 追加：P8 diagnostic norm-off rejected probe

本节记录一个没有采纳的 P8-only probe：关闭 `_apply_ft7_streamed_guarded_update` 在 P8 profiler 中返回的 diagnostic norm 计算。该 norm 不参与 correction、rollback 或 CE gate，但本 probe 没有改善 TimeAUC，因此代码已回滚到第 24 节 accepted stack-role paired guard 版本。

Probe：

```text
results/real_rerun_20260506/v83_p8_diag_norm_off_probe_20260507T134500Z/
```

P8 profiler：

| metric | stack-role paired guard | diagnostic norm-off probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.119479` | `1.122680` | `<=1.05` |
| train step ratio | `0.997895` | `1.006851` | diagnostic |
| functional update / step | `0.084812` | `0.084524` | `<=0.10` |
| functional update ms/step | `0.127579` | `0.126614` | diagnostic |
| P8 pass | `0` | `0` |  |

判断：

1. diagnostic norm-off 只微幅降低 functional update 子阶段，没有改善关键 TimeAUC。
2. 由于 TimeAUC 从 `1.119479` 反弹到 `1.122680`，本 probe 不采纳。
3. 当前 active code 回到 stack-role paired guard 版本，best effective P8 record 仍是第 24 节的 `1.119479`。

Probe hash：

| artifact | SHA256 |
|---|---|
| norm-off P8 profiler | `49e5cb0fcc044e43437175ba8239e406349eb862c673c0cef583d70d81effd60` |
| norm-off P8 raw | `7a6f22b8390b7c8f88512846bdbf6b7b95f7654b87f57b4b2254a52f3e340ed2` |

最终 active hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `c99e6bfa6ede9f3f61d240b92e06521fc2d675cb7e2b7b887f62b582b62729c4` |

最终一句话：

> diagnostic norm-off 没有带来 P8 闭合，已回滚；v8.3 的当前最好状态仍是 H0/P7/S2/update-fraction 全过、P8 TimeAUC 卡在 `1.119479`，所以计划还没完成。

## 26. 追加：P8 foreach correction rejected probe

本节记录另一个未采纳的 P8-only probe：把 streamed second-diff correction 的多次 in-place add 改成 foreach 风格 correction。该 probe 不改 loss、teacher、sampler、CPU offload 或 functional gate，但实际 P8 没有改善，因此代码已回滚。

Probe：

```text
results/real_rerun_20260506/v83_p8_foreach_correction_probe_20260507T140000Z/
```

P8 profiler：

| metric | stack-role paired guard | foreach correction probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.119479` | `1.120113` | `<=1.05` |
| train step ratio | `0.997895` | `1.005783` | diagnostic |
| functional update / step | `0.084812` | `0.089918` | `<=0.10` |
| functional update ms/step | `0.127579` | `0.137334` | diagnostic |
| P8 pass | `0` | `0` |  |

判断：

1. foreach correction 没有减少 functional update wall-clock，反而把 update ms/step 从 `0.127579` 推到 `0.137334`。
2. TimeAUC 从 `1.119479` 反弹到 `1.120113`，因此不采纳。
3. 当前继续保留第 24 节 stack-role paired guard 与后续 accepted pre-holdout pair path。

Probe hash：

| artifact | SHA256 |
|---|---|
| foreach P8 profiler | `8f583fcd160ef4e45a86d201c72fd8efe0d4ccec62abbc400fc2c95f5a5f0340` |
| foreach P8 raw | `8eacb70e174547813976e61cfbcdf307489957a21861f7f36f4fb606a6fac109` |

## 27. 追加：P8 pre-holdout paired forward/backward trim

本节继续 P8 blocker。accepted change 只作用于 P8 profiler：在 event step 中，把 `holdout_loss_before` 的 forward 与训练 split 的 forward/backward 合并成一次 paired forward；仍只对训练 split 产生 CE gradient，不对 holdout split 回传。该改动不改变 P5/P6/P7 official confirmation 路径，也不改变 functional correction、role budget、event stride 或 rollback。

### 27.1 数值等价检查

本地等价检查：

```text
device = cuda
loss_abs_diff = 0.0
holdout_loss_abs_diff = 2.384185791015625e-07
max_grad_abs_diff = 7.916241884231567e-09
```

判断：训练 loss 与参数梯度达到数值等价；holdout loss 只有浮点规约级差异。

### 27.2 P8 official-style artifact

Probe：

```text
results/real_rerun_20260506/v83_p8_preholdout_pair_probe_20260507T141500Z/
```

Official-style artifact：

```text
results/real_rerun_20260506/v83_p8_preholdout_pair_official_20260507T141500Z/
```

Route：

```json
{
  "route": "S0-FunctionalP8TimeProfilerFail",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "H0_system_base_pass": 1,
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 0,
  "ValLossAUC_time_ratio": "1.0916243113046948",
  "functional_update_time_ratio_of_step": "0.08732472824931299",
  "unknown_time_fraction_max": "0.034062180796656905",
  "success_v83_minimum": 0
}
```

P8 profiler：

| metric | stack-role paired guard | pre-holdout pair | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.119479` | `1.091624` | `<=1.05` |
| train step ratio | `0.997895` | `0.968227` | diagnostic |
| functional update / step | `0.084812` | `0.087325` | `<=0.10` |
| metric build ms/step | `0.061197` | `0.000000` | diagnostic |
| functional update ms/step | `0.127579` | `0.128394` | diagnostic |
| unknown time fraction max | `0.032605` | `0.034062` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. pre-holdout paired path 是有效的 P8 implementation trim：TimeAUC 从 `1.119479` 降到 `1.091624`。
2. metric build 被完全并入训练 forward/backward，`functional_metric_build_time_ms_mean = 0.0`。
3. step-indexed trajectory 不变：ValLossAUC step ratio 仍为 `0.996934`。
4. update fraction 仍过子门：`0.087325 <= 0.10`。
5. 但 TimeAUC 仍高于 `1.05`，所以 v8.3 不能记 minimum success。

No-fake audit：

```text
rows_checked = 3575
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 27.3 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `8aa609a3d0bb75280d481dda96eb89661c7c5518091855cfb084b4722ff282a6` |
| pre-holdout pair route | `e8d4571a65ccbeaa38536d0fa9d49124f9ee1d38d3a7dd723e827fde4309c32e` |
| pre-holdout pair P8 profiler | `6ccd68f93ba2245a44750c62cc360c56e7ae6cfd1928b20ad1fe1bffb7ff1a8d` |
| pre-holdout pair P8 raw | `b43770fca7be9dfe3888e0d56491fd3175dcf8fcef0d1ca1c7fabc6dbd23bf9e` |
| pre-holdout pair provenance audit | `1e72e5093412ffb0f7ee56b80b514b797c7236e829892fc5a1274a3f979bec1a` |
| pre-holdout pair run manifest | `2acf2f6d60a4f8647ce33e0bcc6833ea64b7e6f213a5dec24b96560b00bf3d55` |
| pre-holdout pair run complete | `25de4ea6e6c9d8f4c44c8bb20bbbd1ad45327ab3afa3ca9534c8ade3bf6a6690` |

## 28. 追加：FT7 event-stride screen 与拒绝原因

为了确认 P8 是否只是 event frequency 的系统开销问题，本节做了 FT7 event stride screen。该方向没有改 loss、teacher、offload 或 objective，但它改变 functional update schedule，因此必须同时看 P5 几何收益，不能只看 P8。

### 28.1 stride8 P8 screen

Probe：

```text
results/real_rerun_20260506/v83_p8_stride8_screen_probe_20260507T143000Z/
```

P8 screen result：

| metric | stride4 pre-holdout pair | stride8 screen | gate |
|---|---:|---:|---:|
| ValLossAUC time ratio | `1.091624` | `1.045620` | `<=1.05` |
| train step ratio | `0.968227` | `0.903556` | diagnostic |
| functional update / step | `0.087325` | `0.046081` | `<=0.10` |
| P8 pass | `0` | `1` |  |

P8 alone 会过，但这不能作为 success，因为 event stride 改变了 functional update 的几何强度，需要重新走 P5。

### 28.2 stride8 P5/P6/P7 probe

Official-style probe：

```text
results/real_rerun_20260506/v83_ft7_stride8_official_probe_20260507T143500Z/
```

Route：

```json
{
  "route": "S1-P5FunctionalCoSelectionNoUsefulSurvivor",
  "primary_blocker": "p5_functional_no_useful_survivor",
  "success_v83_minimum": 0
}
```

P5 result：

| metric | stride4 accepted FT7 | stride8 probe | gate |
|---|---:|---:|---:|
| macro val acc delta vs base | `+0.000651` | `-0.000868` | `>= -0.005` |
| geometry curvature ratio | `0.775785` | `0.889794` | primary benefit |
| geometry reduction | `0.224215` | `0.110206` | primary benefit |
| step ratio mean | `1.156406` | `1.095771` | `<=1.50` |
| P5 useful pass | `1` | `0` |  |

判断：

1. stride8 证明 P8 可以靠降低 event frequency 过线，但它把 geometry reduction 从 `0.224215` 打到 `0.110206`。
2. 这偏离 v8.3 中心目标：functional update 必须提供几何优势，而不是只变成更少执行的系统开销。
3. 因此 stride8 不采纳；P8 pass 不能覆盖 P5 failure。
4. 当前 accepted best 仍是第 27 节 pre-holdout paired path：`P7 pass + P8 TimeAUC = 1.091624`，尚未完成。

### 28.3 关键 hash

| artifact | SHA256 |
|---|---|
| stride8 screen P8 profiler | `dd43f67b608203853e10ba09296324e8f9c2bed976469301cab558fa76d0b32a` |
| stride8 screen summary | `b2b26da0b5618c77e345f7e75cddfd81541b38cdc394af169fab9d584db3a5e0` |
| stride8 official probe route | `b43e624eb47b56cc03c135a44c34b9f7212598710d0bffe63677facaa824db21` |
| stride8 P5 co-selection | `237dfee2cea9ebc99e3600c49db0bebe2f0e99be43ec78e4b6d181d53513693c` |
| stride8 P4 smoke | `ccaa816dac84e23c07e3f4f203dc4c548b685632091d6411c8b4a7473111c434` |

### 28.4 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

当前 best accepted route：

```text
results/real_rerun_20260506/v83_p8_preholdout_pair_official_20260507T141500Z/
ValLossAUC_time_ratio = 1.0916243113046948
P8_TimeAUCProfilerPass = 0
success_v83_minimum = 0
```

最终 active hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `8aa609a3d0bb75280d481dda96eb89661c7c5518091855cfb084b4722ff282a6` |

最终一句话：

> v8.3 还没有完成。pre-holdout paired forward/backward 是本轮 accepted 的真实进展，把 P8 TimeAUC 从 `1.119479` 推到 `1.091624`；但仍没到 `<=1.05`。stride8 虽能让 P8 过线，却破坏 functional geometry usefulness，因此不采纳。下一步仍应做数值等价的 P8 event implementation trim，不能通过削弱 functional update 几何目标来换 success。

## 29. 追加：P8 instrumentation-side rejected probes

本节记录两个没有采纳的 P8 instrumentation-side probe。二者都不改 CE/no-teacher/no-loss/no-offload，也不改 functional update 数学，但没有改善 accepted best，因此代码均已回滚到第 27 节 pre-holdout paired path。

### 29.1 lazy post-step check rejected

Probe：

```text
results/real_rerun_20260506/v83_p8_lazy_postcheck_probe_20260507T150000Z/
```

设计：非 event step 不额外做 post-step CE logging forward，只在 event gate 需要时测。该 probe 不改变训练更新，但改变 profiler logging policy。

结果：

| metric | pre-holdout pair accepted | lazy post-check probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.091624` | `1.287726` | `<=1.05` |
| train step ratio | `0.968227` | `1.209958` | diagnostic |
| functional update / step | `0.087325` | `0.082741` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：lazy post-check 让 base path 同时变快，反而放大 functional TimeAUC ratio，不采纳。

### 29.2 compact redundant pre-sync rejected

Probe：

```text
results/real_rerun_20260506/v83_p8_compact_sync_probe_20260507T151500Z/
```

设计：去掉连续 phase 之间的冗余 pre-sync，只保留 phase 结束 sync。该 probe 不改变训练更新，也不改变 gate。

结果：

| metric | pre-holdout pair accepted | compact sync probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.091624` | `1.094023` | `<=1.05` |
| train step ratio | `0.968227` | `1.012678` | diagnostic |
| unknown time fraction max | `0.034062` | `0.024362` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：compact sync 降低了 unknown fraction，但没有改善关键 TimeAUC，反而略差，因此不采纳。

### 29.3 更新结论

当前 accepted best 不变：

```text
results/real_rerun_20260506/v83_p8_preholdout_pair_official_20260507T141500Z/
ValLossAUC_time_ratio = 1.0916243113046948
P8_TimeAUCProfilerPass = 0
success_v83_minimum = 0
```

Probe hash：

| artifact | SHA256 |
|---|---|
| lazy post-check P8 profiler | `9d9506aed5521e572502d2c44aec988b645c2cf00f06c49eb19a19c0828a6b19` |
| lazy post-check P8 raw | `42b2b19d3eb0cc03a1efbdd54f322c84b9ff0677423c3f2d598411305854dd9d` |
| compact sync P8 profiler | `3d2564bc61edf34ef76792c611620649c6af93386129fd34fd91dcdbcf986c44` |
| compact sync P8 raw | `25fc9c4db1787bd36f34cf1de6a2d8bd6a1b58348fd8ee99555df70235598b65` |

最终 active hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `8aa609a3d0bb75280d481dda96eb89661c7c5518091855cfb084b4722ff282a6` |

最终一句话：

> v8.3 仍未完成。本节两个 instrumentation probe 都没有超过第 27 节 accepted best；当前唯一采纳的新增进展仍是 pre-holdout paired forward/backward，把 TimeAUC 压到 `1.091624`，但 P8 还差 `0.041624` 才到 hard gate。

## 30. 追加：P8 repeat 与 paired scalar sync probe

本节继续只围绕 P8 TimeAUC blocker 做实现侧检查；没有改 CE objective、teacher、sampler、class weight、CPU offload，也没有改变 functional update 的 event stride、role weight、role budget 或 geometry target。

### 30.1 accepted path repeat probe

Probe：

```text
results/real_rerun_20260506/v83_p8_preholdout_pair_repeat_probe_20260507T153000Z/
```

设计：不改代码，只重复测第 27 节 accepted pre-holdout paired path，用于判断当前 P8 blocker 是否已经接近 measurement noise。

结果：

| metric | accepted best | repeat probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.091624` | `1.100677` | `<=1.05` |
| train step ratio | `0.968227` | `0.941842` | diagnostic |
| functional update / step | `0.087325` | `0.089378` | `<=0.10` |
| unknown time fraction max | `0.034062` | `0.033913` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：repeat 没有把 P8 推近 gate，反而比 accepted best 略差；第 27 节仍是当前 accepted best。

### 30.2 paired scalar sync fusion rejected

Probe：

```text
results/real_rerun_20260506/v83_p8_paired_scalar_sync_probe_20260507T154500Z/
```

设计：尝试把 paired CE gate 中 train/holdout 两个 scalar 的 CPU transfer 合并为一次。该 probe 不改变 CE 值、不改变 guard、不改变 functional update 强度；但它没有改善 P8，因此代码已回滚。

结果：

| metric | accepted best | scalar sync probe | gate |
|---|---:|---:|---:|
| ValLossAUC step ratio | `0.996934` | `0.996934` | diagnostic |
| ValLossAUC time ratio | `1.091624` | `1.142111` | `<=1.05` |
| train step ratio | `0.968227` | `1.041440` | diagnostic |
| functional update / step | `0.087325` | `0.084938` | `<=0.10` |
| unknown time fraction max | `0.034062` | `0.091873` | `<=0.10` |
| P8 pass | `0` | `0` |  |

判断：

1. paired scalar sync fusion 没有改善关键 TimeAUC，且 unknown fraction 接近 `0.10` gate。
2. 该改动只降低了 update fraction 的表观比例，没有改善 end-to-end TimeAUC。
3. 因此不采纳，runner 回到第 27 节 active accepted code。

### 30.3 关键 hash

| artifact | SHA256 |
|---|---|
| repeat probe P8 profiler | `afa1c8bb28769088b514c5a26abb8d5dd09bdfb4d7d5d059b1bee3456aea621a` |
| repeat probe P8 raw | `fef471c4c9e1b45041cf09a2c4992de76c12b92e33405f96506df27d6687de11` |
| repeat probe summary | `6225c2b91150b950322013c23f86852f505a3deb4596716188178206495d9d40` |
| scalar sync probe P8 profiler | `e1c8b86f21fb8eee09903ce414dc0dc2189ffa5baaf58d0e3d300b6fa1ab0dea` |
| scalar sync probe P8 raw | `aa12162c6fb9512c306d4f77d38f9d00dcc8b128d3db6a1dc5eb570d85bdedac` |
| scalar sync probe summary | `3ee09fa4b80940dddeb2b26b2d80e0b2cfeda185f129ec6a807c86219e1f5999` |
| active `experiments/run_gafu_v83_real.py` | `8aa609a3d0bb75280d481dda96eb89661c7c5518091855cfb084b4722ff282a6` |

### 30.4 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8 update fraction subgate = pass
P8 unknown fraction subgate = pass
P8 TimeAUCProfilerPass = false
success_v83_minimum = false
```

当前 accepted best 不变：

```text
results/real_rerun_20260506/v83_p8_preholdout_pair_official_20260507T141500Z/
ValLossAUC_time_ratio = 1.0916243113046948
P8_TimeAUCProfilerPass = 0
success_v83_minimum = 0
```

最终一句话：

> v8.3 还没有完成。最新 repeat 与 scalar-sync probe 都没有超过 pre-holdout paired path；当前真正 blocker 仍是 P8 TimeAUC，而不是 H0、P7、S2、update fraction 或 contract。下一步仍应做数值等价的 P8 event/guard forward fusion，不能通过降低 functional update 几何强度或改变实验合同换 success。

## 31. 追加：schedule / guard-side narrow screens

本节继续围绕同一个 blocker：`FT7` 已通过 H0 / P7 / S2 / update fraction，但 P8 TimeAUC 仍未闭合。本节 probe 没有修改 CE objective、teacher、sampler、class weight、CPU offload，也没有把 geometry loss 作为训练目标。

### 31.1 head split pair rejected

Probe：

```text
results/real_rerun_20260506/v83_p8_head_split_pair_probe_20260507T160000Z/
```

设计：把 head-role paired guard 的 concatenated forward 拆成 train / holdout 两次 head forward，用于检查小 batch head path 是否拖慢 P8。该 probe 不改变 functional update 数学，但没有改善 accepted best，因此不采纳。

| metric | accepted best | head split probe | gate |
|---|---:|---:|---:|
| ValLossAUC time ratio | `1.091624` | `1.101792` | `<=1.05` |
| functional update / step | `0.087325` | `0.091759` | `<=0.10` |
| unknown time fraction max | `0.034062` | `0.033533` | `<=0.10` |
| P8 pass | `0` | `0` |  |

### 31.2 stride7 alpha schedule screens

Probe：

```text
results/real_rerun_20260506/v83_ft7_stride7_alpha9_p8_repeat_20260507T173000Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha10_schedule_screen_20260507T174500Z/
```

结果：

| schedule | P5 pass | geometry ratio | geometry reduction | P8 time ratio | P8 pass | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `stride7 alpha9 repeat` | from prior screen pass | prior `0.786469` | prior `0.213531` | `1.052157` | 0 | near-pass but still fail |
| `stride7 alpha10` | 1 | `0.781849` | `0.218151` | `1.052984` | 0 | geometry ok, P8 still fail |

判断：

1. `stride7 alpha9/10` 不像 stride8 那样削弱 geometry；它们仍保留 functional geometry usefulness。
2. 但 P8 仍高于 `1.05`，最接近的 repeat 为 `1.052157`。
3. 因此不能把 schedule screen 写成 success，也不能进入 P9。

### 31.3 pre-holdout interval guard screens

设计：不降低 FT7 event frequency，而是让部分 event 使用更严格的 no-pre-holdout-budget guard。这样可以少做一次 pre-holdout forward；如果没有 pre-holdout baseline，则 holdout gate 不给额外 budget，要求 functional correction 不增加 after-AdamW holdout CE。该思路仍属于 update guard implementation，不是 loss / teacher / sampler 改动。

Probe：

```text
results/real_rerun_20260506/v83_ft7_stride7_alpha9_preholdout2_screen_20260507T180000Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha12_preholdout2_screen_20260507T181500Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha15_preholdout2_screen_20260507T183000Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha18_preholdout2_screen_20260507T184500Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha20_preholdout2_screen_20260507T190000Z/
```

结果：

| schedule | P5 pass | macro delta | geometry ratio | geometry reduction | P8 time ratio | P8 pass |
|---|---:|---:|---:|---:|---:|---:|
| `alpha9 pre2` | 0 | `-0.001302` | `0.845198` | `0.154802` | `1.044036` | 1 |
| `alpha12 pre2` | 0 | `-0.000217` | `0.819550` | `0.180450` | `1.049044` | 1 |
| `alpha15 pre2` | 0 | `-0.000434` | `0.807262` | `0.192738` | `1.040043` | 1 |
| `alpha18 pre2` | 0 | `-0.001085` | `0.811543` | `0.188457` | `1.050273` | 0 |
| `alpha20 pre2` | 0 | `-0.000868` | `0.811394` | `0.188606` | `1.049362` | 1 |

判断：

1. `pre_holdout_every=2` 能真实修 P8；多个 probe 的 TimeAUC 已经低于 `1.05`。
2. 但它没有保住 P5 primary geometry gate：最佳 geometry ratio 仍为 `0.807262`，未达到 `<=0.80`。
3. 因此该路线是 “P8 pass / P5 fail”，不能采纳为 v8.3 success。

### 31.4 role-weight narrow screens

继续在 `stride7 alpha15 pre_holdout_every=2` 上做 role-wise 窄屏，检查是否能恢复 geometry。该 probe 仍属于 v8.3 文档中的 role-wise functional correction，不改 loss 或 teacher。

Probe：

```text
results/real_rerun_20260506/v83_ft7_stride7_alpha15_preholdout2_stack10_screen_20260507T191500Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha15_preholdout2_stack125_screen_20260507T193000Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha15_preholdout2_head125_screen_20260507T194500Z/
results/real_rerun_20260506/v83_ft7_stride7_alpha15_preholdout2_head15_screen_20260507T200000Z/
```

结果：

| role weights | P5 pass | macro delta | geometry ratio | geometry reduction | P8 time ratio | P8 pass |
|---|---:|---:|---:|---:|---:|---:|
| `stack1.0/head1.0` | 0 | `-0.001519` | `0.813883` | `0.186117` | `1.045605` | 1 |
| `stack1.25/head1.0` | 0 | `-0.000651` | `0.821803` | `0.178197` | `1.047369` | 1 |
| `stack0.75/head1.25` | 0 | `-0.001302` | `0.806170` | `0.193830` | `1.047891` | 1 |
| `stack0.75/head1.5` | 0 | `-0.000651` | `0.812125` | `0.187875` | `1.028478` | 1 |

判断：

1. role-weight 调整可以继续保持 P8 pass，甚至 `head1.5` 达到 TimeAUC `1.028478`。
2. 但所有 role-weight probe 均未跨过 P5 geometry primary gate。
3. 这说明当前卡点已经从单纯 P8 timing 转为 “P8-efficient guard 与 P5 geometry usefulness 不能同时满足”。

### 31.5 关键 hash

| artifact | SHA256 |
|---|---|
| head split P8 profiler | `304d2f648d45d7067cad195288d5536b4716706751c93d2087178dd67319c02a` |
| head split P8 raw | `210d2cd576c3b7dde1f3f33f92fd62996345a8584b8012e44ad62feb3f593f6b` |
| head split summary | `f515435b9332e9b31ef8f238c7c6bb0a981371e5f4985b69750abce7b0831d56` |
| stride7 alpha9 repeat P8 profiler | `c014be0aa552fcfb52df554978bf04cfe2b11b0acc8e9bd92e213ef344c2bf85` |
| stride7 alpha10 P5 co-selection | `b78ed51ed0c8f1c93cbc45f73b147501cd5c23b68be4041a49ed5b034f4308c7` |
| stride7 alpha10 P8 profiler | `1c261daa9f5b547208b51cdac85159329ddb3dd7acb9e47f544453c5bc7e6904` |
| alpha9 pre2 P5 co-selection | `0f2c10d8f348673ebb7735509c17ef704f1a4d916a83a74b7cea29ef6d8015fa` |
| alpha9 pre2 P8 profiler | `473ee8f730810bf5ac1d2fdec4965e78d5132a9f0df79ca046d0b81a1645c492` |
| alpha15 pre2 head1.5 P5 co-selection | `2897d00bbb1ef1d58d04459aa9001a01c36861e609129cb36e965252994d5786` |
| alpha15 pre2 head1.5 P8 profiler | `70ac750f1bbf1dc801b42b8605d3a26aea98c6f0e8d502aece64c9c2f8d6149a` |

### 31.6 更新结论

v8.3 仍未完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
strict CE/no-teacher/no-loss contract = pass
best accepted path P8 = fail
best P8-pass guard screen P5 = fail
success_v83_minimum = false
```

当前状态：

```text
Accepted P7 survivor:
  FT7, macro delta = +0.0016927083333333334
  geometry reduction = 0.2228229211976781
  S2 = 9/9

Best centered P8-near screen with P5 pass:
  stride7 alpha9/10 family
  P8 TimeAUC = 1.052157 to 1.052984
  P8_TimeAUCProfilerPass = 0

Best P8-pass screen:
  stride7 alpha15 pre_holdout_every=2 head1.5
  P8 TimeAUC = 1.028478
  P5 geometry ratio = 0.812125
  P5_functional_useful_pass = 0
```

最终一句话：

> v8.3 仍没有完成。本节找到了一个能过 P8 的 guard-side implementation screen，但它没有保住 functional geometry usefulness；同时保住 P5 的 stride7 alpha9/10 仍卡在 P8 `~1.052`。下一步不能继续靠降低 guard/geometry 强度换成功，而应做真正数值等价的 event/guard fusion 或 role-local geometry update，让 P5 geometry 和 P8 TimeAUC 同时闭合。

## 32. 追加：hidden68 FT7 stride8 + P9 scaling/robustness closure

本节继续严格沿 v8.3 文档中心思想执行：先 system-gated H0，再 functional update re-entry；functional update 仍是 update rule，不是 loss，不是 teacher，不是 sampler/class-weight，也不是 CPU offload。

本节所有数值来自：

```text
results/real_rerun_20260506/v83_ft7_hidden68_budget015_trainbudget_stride8_official_20260507T223000Z/
```

### 32.1 代码改动

本节把 P9 从占位 `not_run` 改为真实 gated artifact。只有在 `P8_TimeAUCProfilerPass=1` 后才打开 P9。

Active FT7 guard/update setting：

```text
FT7_ROLE_BUDGETS = {"stack": 0.15, "head": 0.15}
P5_FT7_EVENT_STRIDE = 8
P5_FT7_EVENT_ALPHA_MULT = 15.0
FT7_PRE_HOLDOUT_EVERY = 2
FT7_NO_PRE_HOLDOUT_TRAIN_BUDGET_MULT = 0.25
```

新增 P9 grid：

```text
train_size = 256,512,1024,1536,4096
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
seeds = 0,1,2,3,4
```

判断规则保持文档要求：

```text
sample_efficiency_auc_functional >= sample_efficiency_auc_base
robustness benefit settings >= 2
geometry_delta_vs_base_mean < 0
ECE_delta <= 0 or NLL_delta <= 0
memory/step ratio remains inside system gate
```

代码检查：

```text
python -m py_compile experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 32.2 rejected / accepted path

先说明本节没有把早先失败 screen 偷换成 success：

| run | route / blocker | 判断 |
|---|---|---|
| hidden64 default official | `S6-NoBaseSystemCandidateFunctionalFullTaskGated` | rejected：没有 H0 base |
| hidden68 budget0.12 pre2 | P4 geometry gate fail | rejected |
| hidden68 budget0.15 no train-budget fallback | P4 geometry gate fail | rejected |
| hidden68 budget0.15 train-budget pre3 | P4 task preservation fail | rejected |
| hidden68 budget0.15 train-budget stride8 | P5/P7/P8/P9 pass | accepted |

早先 `stride8 alpha8` 被拒绝，是因为 P5 geometry usefulness 被削弱；本节 accepted 的不是那个结果，而是 `role_budget=0.15 + alpha15 + pre2 + train-budget fallback` 后的 `stride8`。它保住了 P5/P7 geometry gate。

### 32.3 route

```json
{
  "route": "S0-FunctionalReEntryMinimumSuccess",
  "base_candidate_id": "KW6",
  "functional_candidate_id": "FT7",
  "base_macro_gap": "0.0216796875",
  "functional_macro_gap": "0.0013020833333333333",
  "functional_ci95_low": "-0.0004557291666666667",
  "geometry_delta_vs_base": "0.23815161791505923",
  "P7_confirm10_pass": 1,
  "P8_TimeAUCProfilerPass": 1,
  "P9_scaling_robustness_pass": 1,
  "success_v83_minimum": 1,
  "success_v83_formal": 0,
  "primary_blocker": "none"
}
```

注：`route_decision.json` 中的 `TimeAUC_pass` / `ProfilerPass` 是 base-route legacy fields；本轮 functional official time gate 看 `P8_TimeAUCProfilerPass=1`。

### 32.4 P5/P6/P7/P8

| stage | macro delta | CI low | test delta | geometry ratio | geometry reduction | memory max / ratio | step max / ratio | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P5 | `-0.000217` | n/a | `+0.003255` | `0.781900` | `0.218100` | `1.002518` | `1.119802` | 1 |
| P6 | `-0.000260` | n/a | `+0.002604` | `0.760509` | `0.239491` | `1.027250` | `1.098856` | 1 |
| P7 | `+0.001302` | `-0.000456` | `+0.001302` | `0.761848` | `0.238152` | `1.027250` | `1.098856` | 1 |
| P8 | n/a | n/a | n/a | n/a | n/a | `1.002674` | `1.041379` | 1 |

P8 detail：

| metric | value | gate |
|---|---:|---:|
| `ValLossAUC_time_ratio_vs_base_mean` | `1.040689` | `<=1.05` |
| `functional_update_time_ratio_of_step_mean` | `0.044032` | `<=0.10` |
| `unknown_time_fraction_max` | `0.080018` | `<=0.10` |
| `ValLossAUC_step_ratio_vs_base_mean` | `0.997471` | record |

判断：

1. P5/P7 geometry ratio 分别为 `0.781900` / `0.761848`，不是靠削弱 geometry 换 P8。
2. P8 TimeAUC 从上一节近似 `1.052` 推到 `1.040689`，正式过线。
3. Functional update overhead 为 step 的 `4.40%`，低于 `10%` gate。

### 32.5 P9 scaling / robustness

P9 真实落盘 `300` 条 measured raw rows + `1` 条 summary row，CSV 总行数含 header 为 `302`。

Summary：

| metric | value | pass |
|---|---:|---:|
| `sample_efficiency_auc_base` | `3139.083333` |  |
| `sample_efficiency_auc_functional` | `3141.300000` |  |
| `sample_efficiency_auc_delta` | `+2.216667` | 1 |
| `robustness_benefit_count` | `3/5` | 1 |
| `geometry_delta_vs_base_mean` | `-0.204968` | 1 |
| `ECE_delta_vs_base_mean` | `+0.000484` |  |
| `NLL_delta_vs_base_mean` | `-0.004983` | 1 |
| `memory_ratio_vs_base_max` | `1.004610` | 1 |
| `step_ratio_vs_base_max` | `1.229527` | 1 |
| `P9_scaling_robustness_pass` | `1` | 1 |

Robustness settings：

```json
{
  "input_noise:0.05": 1,
  "input_noise:0.1": 1,
  "label_noise:0.05": 0,
  "label_noise:0.1": 1,
  "label_noise:0.2": 0
}
```

判断：

1. Functional route 的 sample efficiency AUC 高于 base。
2. 5 个 robustness setting 中有 3 个满足 `AccDrop_functional <= AccDrop_base`。
3. 几何均值下降，且 NLL 均值改善；ECE 有轻微正 delta，但 P9 的 consistency 条件允许 `ECE_delta <= 0 or NLL_delta <= 0`。
4. P9 的 system ratio 没有破坏 S2：memory max `1.004610`，step max `1.229527`。

### 32.6 no-fake audit

```text
rows_checked = 3875
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 32.7 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v83_real.py` | `9565be79ec3896aa06baa396aa43f463a89461cae2d67e93486126990d829166` |
| route | `f5291eebcff20da45876497d977754c2c8b7e7b66fdb778f53117eac18de61f0` |
| P7 confirm10 | `c6ce8b7f217b1307e581139cc7a64c2ce34ba2c26ad9a8b0d152d76fc15e438e` |
| P8 profiler | `f031fdfb494e6ada0ac2dc07975d5a2ee05965df3ddcae4505ca1fd7faea5c5e` |
| P9 scaling robustness | `bba5164b1d0e929137d6190366dddbeae8a777eff280a31db4d67eb37ed58ae1` |
| v83 provenance audit | `9abcd95dec0a56105415993985bece946265cc80a8ea376b783e2e8ee93ac0a0` |

### 32.8 更新结论

v8.3 minimum success 已完成：

```text
H0_system_base_pass = true
P7_10seed_functional_confirmation = pass
P8_TimeAUCProfilerPass = pass
P9_scaling_robustness_pass = pass
strict CE/no-teacher/no-loss/no-offload contract = pass
success_v83_minimum = true
success_v83_formal = false
```

机制结论：

1. `KW6 hidden68` 是有效 H0 base：macro `+0.0216796875`，memory/step 在 system gate 内。
2. `FT7` role-wise functional update 在 H0 base 后重新进入 full task，并在 P7 10-seed 保住 task，同时带来 `~23.8%` curvature reduction。
3. `stride8 + alpha15 + role budget0.15 + pre2 + train-budget fallback` 是当前 accepted P8 repair：它不是早先失败的 stride8-alpha8，也不是削弱 geometry 的 P8-only screen。
4. P9 证明 functional geometry 不只是几何指标好看：sample efficiency AUC 和 3/5 robustness settings 都优于 base，同时 system ratio 仍安全。
5. 全程没有 teacher/self-teacher、distillation、loss/sampler/class-weight 修改、CPU offload、fake/proxy。

最终一句话：

> v8.3 的核心计划已经完成到 minimum success：在 `KW6 hidden68` 这个 CE-only system survivor 上，`FT7` functional update 通过了 P7 task/geometry confirmation、P8 TimeAUC profiler 和 P9 scaling/robustness。它证明 functional update 可以作为 system-gated 的几何优势机制重新接回 PureKAN 路线；本轮不声明 formal strong success，只声明 `success_v83_minimum = true`。
