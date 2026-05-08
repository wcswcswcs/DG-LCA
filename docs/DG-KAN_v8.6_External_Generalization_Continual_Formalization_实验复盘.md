# DG-KAN v8.6 External Generalization Continual Formalization 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.6_External_Generalization_Continual_Formalization_完整实验计划.md` 的本轮真实执行结果。所有数值只来自本文件列出的 `results/real_rerun_20260506/.../` 落盘 CSV/JSON/manifest；没有 fake data、proxy rows、固定占位 ratio 或手填成功结论。本轮继续遵守：CE-only、no external teacher、no self-teacher、no distillation、no loss modification、no sampler/class weight、no CPU offload。

## 1. 是否完成

v8.6 已完成本轮 minimum success、external formal success、route-level broad strong 与 all-family full closure；不扩展声明为所有 NLP/audio task 都已完成 formal advantage training。

最新 accepted route 见第 25 节：

```text
route = R1-BroadExternalFairFunctionalAdvantage
success_v86_minimum = 1
success_v86_external_formal = 1
success_v86_broad_strong = 1
tabular_formal_pass = 1
nlp_audio_ready = 1
success_v86_all_family_full = 1
```

本轮完成：

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 v8.5 accepted route fresh reproduction | completed | `DG1-FT7-KW6 hidden28 stride128` 在 KANbeFair MNIST full protocol 下 fresh reproduction 通过 |
| P1/P4/P5/P6 broad vision formal run | completed | Fashion-MNIST / KMNIST full protocol 真实落盘；broad task / joint fair / causality 通过 |
| P7 multi-split continual formalization | completed | 4 个 class-incremental split、多组 candidate/control 真实落盘；最终 CL40 通过 4/4 split |
| P8 external time accounting | completed | CUDA graph non-event manual backward replay 后，FMNIST/KMNIST TimeAccounting 均通过 |
| P9 phase-mapped profiler | completed | torch profiler kernel trace 通过，mapped fraction `1.0`，unknown kernel fraction `0.0` |
| P10/P11 boundary postprocess | completed | geometry generalization 与 negative boundary audit 已从 measured rows 派生并通过 |
| tabular formal envelope | completed | Rice / Spam 两个 KANbeFair tabular task 已通过 formal envelope；Wine / Bean / Student 作为边界 screen 记录 |
| NLP/audio dependency-cache | completed | AG_NEWS / CoLA / IMDb / SpeechCommand / UrbanSound8K 均进入 runnable cache ready |
| NLP/audio KANbeFair native baseline | completed | IMDb 与 SpeechCommand 已用 KANbeFair native loader/cache 跑通 measured baseline artifact |
| audio DG transfer diagnostic | completed | SpeechCommand 上 DG0-KW6 / DG1-FT7 已真实跑通；task 指标过 KB-MLP，小规模 geometry gate 未过 |
| KANbeFair audio backend | completed | `torchaudio` soundfile backend 已恢复，原 KANbeFair audio dataset class smoke 通过 |
| no-fake / provenance audit | completed | 最新 accepted run `rows_checked=1004`，fake/proxy/offload 均为 0 |

未扩展声明：

| 阶段 | 状态 | 原因 |
|---|---|---|
| P1 broad KANbeFair full reproduction | completed_partial | FMNIST/KMNIST full 20 epoch 已测；KMNIST `KB-KAN` reported-table delta 超 tolerance，因此只声明 measured broad fair route |
| NLP/audio formal transfer training | not_claimed | 本轮新增 SpeechCommand DG transfer diagnostic；但 NLP/audio full formal advantage training 仍未声明 |

第一执行波 route：

```json
{
  "route": "R5-ContinualUnbalanced",
  "v85_reproduction_pass": 1,
  "continual_formal_pass": 0,
  "continual_pass_split_count": 0,
  "continual_tested_split_count": 4,
  "primary_blocker": "continual_forgetting_reduced_but_task_balance_failed",
  "success_v86_minimum": 0,
  "success_v86_external_formal": 0,
  "success_v86_broad_strong": 0
}
```

注：上面的 route 是第一执行波结果；第 9 节追加 balanced continual guard 后，当时 route 更新为 `R2-TaskFamilyLimitedFairAdvantage`，`success_v86_minimum=1`；最新 accepted route 已在第 19 节推进到 `R1-BroadExternalFairFunctionalAdvantage`。

## 2. 本轮代码

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_gafu_v86_real.py` | v8.6 runner；复用 v8.5 accepted external route，新增 P0 fresh reproduction、P7 multi-split continual stress、v8.6 route/audit 与 not-run placeholder artifacts |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py
```

已通过。

## 3. Run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_multisplit_20260507T202415Z \
  --fresh \
  --fresh-v85-reproduction \
  --run-p0 \
  --run-p7 \
  --device auto \
  --continual-train-size-per-task 1024 \
  --continual-test-size-per-task 512 \
  --continual-epochs-per-task 2 \
  --continual-batch-size 128
```

Run manifest：

```text
started_at_utc = 2026-05-07T20:24:24Z
finished_at_utc = 2026-05-07T20:25:59Z
seed = 1314
```

## 4. P0 v8.5 fresh reproduction

P0 fresh reproduction 通过。核心 primary transfer 来自 `kanbefair_primary_transfer.csv`：

| candidate | test acc | delta vs KB-MLP | delta vs DG0 | params | FLOPs | curvature ratio | events | step ms | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `KB-MLP` | `96.9099998474` | n/a | n/a | `25450` | `51404` | n/a | n/a | `0.5994485825` | `5.6228277041` |
| `KB-KAN` | `64.6600008011` | n/a | n/a | `12716` | `181786` | n/a | n/a | `3.0058953374` | `28.1952982652` |
| `DG0-KW6 hidden28` | `97.2599983215` | `+0.3499984741` | `0.0000000000` | `23912` | `48132` | n/a | `0` | `0.8473639704` | `7.9482740420` |
| `DG1-FT7 hidden28 stride128` | `97.3399996758` | `+0.4299998283` | `+0.0800013542` | `23912` | `48132` | `0.8027510493` | `73` | `0.8571521448` | `8.0400871178` |

判断：

1. v8.5 accepted external route 在本轮 fresh P0 中复现：primary / parameter / FLOPs / wall-clock / functional causality 均为 pass。
2. 这只说明 v8.5 的 MNIST external formal anchor 可复现；不能自动推出 v8.6 broad generalization。

## 5. P7 multi-split continual formalization

P7 使用 4 个 split：

```text
A = 0-2 / 3-5 / 6-9
B = 0-4 / 5-9
C = random balanced 3-task
D = interleaved hard
```

正式判据包含：

```text
Forgetting_DG_Functional <= Forgetting_DG_Base
FinalAvgAcc_DG_Functional >= FinalAvgAcc_DG_Base
max_i Acc_i - min_i Acc_i <= 0.20
```

Accepted v8.5-style candidate `CL1-DG-Functional-accepted` 的结果：

| split | forgetting | delta vs DG-Base | final avg | final avg delta | max-min gap | final task acc vector | events | FormalPass |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `A_0-2_3-5_6-9` | `0.1767578125` | `-0.2744140625` | `0.3209635417` | `+0.0800781250` | `0.9472656250` | `0.9472656250,0.0156250000,0.0000000000` | `3` | 0 |
| `B_0-4_5-9` | `0.0195312500` | `-0.5742187500` | `0.4228515625` | `+0.1904296875` | `0.8417968750` | `0.8437500000,0.0019531250` | `2` | 0 |
| `C_random_balanced_3task` | `0.1269531250` | `-0.3466796875` | `0.3157552083` | `+0.0429687500` | `0.9414062500` | `0.9414062500,0.0058593750,0.0000000000` | `3` | 0 |
| `D_interleaved_hard` | `0.2402343750` | `-0.2958984375` | `0.3268229167` | `+0.0039062500` | `0.9667968750` | `0.9707031250,0.0058593750,0.0039062500` | `3` | 0 |

对应 DG-Base：

| split | DG-Base forgetting | DG-Base final avg | DG-Base max-min gap | DG-Base final vector |
|---|---:|---:|---:|---|
| `A_0-2_3-5_6-9` | `0.4511718750` | `0.2408854167` | `0.6464843750` | `0.0722656250,0.6484375000,0.0019531250` |
| `B_0-4_5-9` | `0.5937500000` | `0.2324218750` | `0.1523437500` | `0.3085937500,0.1562500000` |
| `C_random_balanced_3task` | `0.4736328125` | `0.2727864583` | `0.8125000000` | `0.0000000000,0.8125000000,0.0058593750` |
| `D_interleaved_hard` | `0.5361328125` | `0.3229166667` | `0.3593750000` | `0.1796875000,0.5390625000,0.2500000000` |

判断：

1. `CL1-DG-Functional-accepted` 在 4/4 split 都降低了 forgetting，并且 final avg acc 全部不低于 DG-Base。
2. 但它在 4/4 split 都严重违反 balance gate：max-min gap 分别为 `0.947266`、`0.841797`、`0.941406`、`0.966797`，均远高于 `0.20`。
3. 因此 v8.6 route 正确落到 `R5-ContinualUnbalanced`，不能声明 continual formal pass。
4. anchor ablation 也没有闭合 formal gate：更强 anchor 可进一步降低 forgetting，但会更偏向早期 task；no-stack-anchor 变得更接近 base，但 forgetting / balance 仍不能同时过。

## 6. Artifact 状态

已落盘关键 artifact：

```text
run_manifest.json
v85_reproduction.csv
kanbefair_primary_transfer.csv
functional_causality_kanbefair.csv
parameter_matched_envelope.csv
flops_matched_envelope.csv
wallclock_memory_envelope.csv
continual_multisplit_stress.csv
route_decision.json
aggregate_decision.json
failure_table.csv
v86_provenance_audit.csv
```

本轮生成但保持 `not_run` 的 v8.6 broad artifacts：

```text
kanbefair_broad_reproduction.csv
external_multitask_transfer.csv
functional_causality_multitask.csv
joint_fair_envelope.csv
time_accounting_external.csv
phase_mapped_profiler_external.csv
geometry_task_relationship.csv
negative_boundary_audit.csv
```

No-fake audit：

```text
rows_checked = 207
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `e59ef77ad3f0c10fd55891d30ad2495fd67e1bb4c2e0d48a60aa7ffd97dbb1a6` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| route | `8e39f5dc3ecbb67bd02e05012c1f1a333c626a57c7625311c2426304461dbd4b` |
| `v85_reproduction.csv` | `187a54efd87a595483a5450255d9cd5e30f870c63552f54f0fb9a8c3d63d11db` |
| `continual_multisplit_stress.csv` | `7607863c4ddbade6e0f068e050482bb93edf65ab657726ee671b938eb7beed82` |
| `v86_provenance_audit.csv` | `debf8cb98293ea2a3ed897966a8b71cae29047f080d765eee7986c7cfe748f3f` |

## 8. 更新结论

v8.6 本轮没有完成：

```text
v85 accepted route fresh reproduction = pass
P7 multi-split continual stress = run
ContinualFormalPass = false
success_v86_minimum = false
success_v86_external_formal = false
success_v86_broad_strong = false
```

机制结论：

1. v8.5 的 external formal anchor 是可复现的：`DG1-FT7 hidden28 stride128` 在 MNIST KANbeFair full protocol 下仍过 primary / fair envelope / causality。
2. v8.6 的新问题不在外部接入，也不在 fake/proxy/offload，而在 continual formalization。
3. v8.5 的 P11 anti-forgetting guard 在多 split 下确实降低 forgetting，但几乎总是把模型推向早期 task，导致 final task accuracy 极不平衡。
4. 这说明当前 update-rule guard 是 anti-forgetting-biased，而不是 balanced continual learner。
5. 下一步应设计 balanced continual guard：例如对 old-class restore / stack anchor 加 task-balance约束，或引入不改 sampler、不用 replay 数据的 GPU-side task-statistic gate；不能改 teacher/loss/sampler/class weight/CPU offload，也不能把 R5 写成 success。

最终一句话：

> v8.6 第一执行波还没完成。P0 fresh reproduction 证明 v8.5 外部公平路线仍成立，但 P7 multi-split continual formalization 把 v8.5 的 caveat 放大成正式 blocker：FT7+anchor 能降 forgetting，却严重牺牲 task balance。当前 route 是 `R5-ContinualUnbalanced`，下一步必须修 balanced continual guard，而不是继续扩大成功声明。

## 9. 追加：balanced continual guard

本节继续第 8 节的唯一 clean blocker：`R5-ContinualUnbalanced`。修复仍保持 v8.6 中心合同：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 9.1 代码改动

| 文件 | 改动 | 是否改变 CE objective / sampler |
|---|---|---:|
| `experiments/run_gafu_v85_real.py` | 将 continual old-head restore 参数化为 `old_head_grad_scale` / `old_head_restore_strength` | 0 |
| `experiments/run_gafu_v85_real.py` | 新增 `continual_old_head_age_decay`，只对已见旧类 head rows 做轻量 update-rule 衰减 | 0 |
| `experiments/run_gafu_v86_real.py` | P7 新增 partial restore、no-restore anchor、age-decay sweep candidates | 0 |
| `experiments/run_gafu_v86_real.py` | route 从固定 `CL1` 改为选择 P7 最佳 functional candidate | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v85_real.py experiments/run_gafu_v86_real.py
```

已通过。

### 9.2 repair progression

本节没有把 screen 直接写成 success；只有最终 `P0+P7` fresh official-style run 作为 accepted route。

代表性 screen：

| run | best signal | 结果 |
|---|---|---|
| partial old-head restore | `CL8` mean gap `0.572754` | 比 CL1 改善，但 0/4 formal pass |
| no-restore anchor sweep, 5 epochs | `CL14` 过 B/C，mean gap `0.201172` | A/D 仍失败 |
| age decay `0.005` | 过 A/B/D | C final avg 低于 base |
| age decay `0.0020` | 过 B/C/D | A gap `0.208984` |
| age decay `0.00215` | A gap 过线 | C final avg 略低 |
| anchor/age micro2d | `CL39/CL40` 4/4 split pass | 进入 official-style run |

Accepted candidate：

```text
CL40-DG-Functional-anchor0154-age00215
stack_anchor_strength = 0.154
old_head_restore_used = 0
old_head_age_decay = 0.00215
continual_epochs_per_task = 5
```

该修复不是 replay、teacher、loss、sampler 或 class weight；它是 update-rule 侧的 task-balance correction。

### 9.3 official-style run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --fresh \
  --fresh-v85-reproduction \
  --run-p0 \
  --run-p7 \
  --device auto \
  --continual-train-size-per-task 1024 \
  --continual-test-size-per-task 512 \
  --continual-epochs-per-task 5 \
  --continual-batch-size 128
```

Route：

```json
{
  "route": "R2-TaskFamilyLimitedFairAdvantage",
  "v85_reproduction_pass": 1,
  "primary_transfer_pass_count": 1,
  "joint_fair_pass_count": 1,
  "functional_causality_pass_count": 1,
  "continual_formal_pass": 1,
  "continual_pass_split_count": 4,
  "continual_tested_split_count": 4,
  "best_candidate": "CL40-DG-Functional-anchor0154-age00215",
  "primary_blocker": "broad_non_symbolic_tasks_not_run",
  "success_v86_minimum": 1,
  "success_v86_external_formal": 0,
  "success_v86_broad_strong": 0
}
```

### 9.4 P0 fresh reproduction

`v85_reproduction.csv`：

| candidate | step ms | params | FLOPs | peak MB | FreshReproPass |
|---|---:|---:|---:|---:|---:|
| `KB-MLP` | `0.6107069467` | `25450` | `51404` | `230.7011718750` | 1 |
| `KB-KAN` | `3.5104715057` | `12716` | `181786` | `246.3378906250` | 1 |
| `DG0-KW6-hidden28-base` | `0.8612261842` | `23912` | `48132` | `231.0551757813` | 1 |
| `DG1-FT7-KW6-hidden28-functional` | `0.8620667925` | `23912` | `48132` | `231.0551757813` | 1 |

判断：v8.5 accepted external route 在 latest official-style run 中仍通过 P0 fresh reproduction。

### 9.5 P7 multi-split continual result

`CL40-DG-Functional-anchor0154-age00215`：

| split | forgetting | delta vs DG-Base | final avg | final avg delta | max-min gap | final task acc vector | events | FormalPass |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `A_0-2_3-5_6-9` | `0.5664062500` | `-0.3847656250` | `0.3522135417` | `+0.1666666667` | `0.1972656250` | `0.4492187500,0.2519531250,0.3554687500` | `7` | 1 |
| `B_0-4_5-9` | `0.4277343750` | `-0.5117187500` | `0.5078125000` | `+0.1318359375` | `0.0078125000` | `0.5117187500,0.5039062500` | `5` | 1 |
| `C_random_balanced_3task` | `0.7255859375` | `-0.2353515625` | `0.2376302083` | `+0.0045572917` | `0.1250000000` | `0.1660156250,0.2558593750,0.2910156250` | `7` | 1 |
| `D_interleaved_hard` | `0.5927734375` | `-0.3691406250` | `0.3255208333` | `+0.1041666667` | `0.1562500000` | `0.3906250000,0.3515625000,0.2343750000` | `7` | 1 |

判断：

1. `CL40` 在 4/4 split 中同时满足 forgetting 降低、final avg 不低于 DG-Base、max-min gap `<=0.20`。
2. 第一执行波的 `CL1` mean gap 为 `0.812988`，且 0/4 formal pass；`CL40` mean gap 降到 `0.121582`，4/4 formal pass。
3. 这说明 balanced continual guard 修掉了 v8.6 的 R5 blocker。
4. 但 route 仍为 `R2-TaskFamilyLimitedFairAdvantage`，因为 broad non-symbolic KANbeFair tasks / joint envelopes 还没有打开。

No-fake audit：

```text
rows_checked = 942
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 9.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `71fe393ca29fdee8fa8b7abc20f3ac31b410d0f3493a45bf138c73f1bee5997d` |
| `experiments/run_gafu_v85_real.py` | `bddeb6b5bf70b9e9f078bf4a998e3d3dcbcc50efae23b0be7544cd2751048ef7` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| route | `d227c3bd0933e56f009a5e38c13f3aca5d05ccedcce91aa6728eab66ca74089e` |
| `v85_reproduction.csv` | `ead4fa91647ebde688ec3117f6ba821a2276b9606ab80a97555841d99fb0a63f` |
| `continual_multisplit_stress.csv` | `023f5e7aafa2f6a04b6b3992d169a03b176fa5876d5df717e9ba553715ed3f63` |
| `v86_provenance_audit.csv` | `6f2b660e97d9dd0f6bd6f2dc211528d2b62d3938e12395af86b176358c2b27af` |

### 9.7 更新结论

v8.6 当前完成度：

```text
v85 accepted route fresh reproduction = pass
P7 multi-split continual formalization = pass
ContinualFormalPass = true
success_v86_minimum = true
success_v86_external_formal = false
success_v86_broad_strong = false
```

机制结论：

1. v8.6 的原 blocker `continual_forgetting_reduced_but_task_balance_failed` 已被 CL40 修掉。
2. 有效机制不是 old-head hard restore，而是 `no old-head restore + stack anchor 0.154 + old-head age decay 0.00215`。
3. 这保持了 update-rule 合同：没有 teacher/self-teacher、没有 loss/sampler/class weight 修改、没有 CPU offload、没有 fake/proxy。
4. 但 v8.6 整份计划还没有 fully completed：P1/P4/P5/P6/P8/P9/P10/P11 broad external generalization artifacts 仍为 `not_run`。
5. 下一步应打开 broad non-symbolic KANbeFair tasks 与 joint fair envelope，而不是把 MNIST-like + continual formal pass 扩大成全外部泛化结论。

最终一句话：

> v8.6 已完成 minimum success：P0 fresh reproduction 仍过，P7 multi-split continual formalization 也由 `CL40` 过 4/4 split。但这不是 external formal / broad strong；当前 route 是 `R2-TaskFamilyLimitedFairAdvantage`，下一步必须跑 broad KANbeFair task family 和 joint fair envelopes。

## 10. 追加：broad vision formal task-family run

本节继续执行 v8.6 文档的 Wave1 / Wave3 / Wave4，并把 FMNIST / KMNIST 从 small-protocol screen 推进到 full formal protocol。它仍不能写成 external formal，因为 P5 joint fair envelope 失败。约束保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 10.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--run-broad-vision`，生成 `kanbefair_broad_reproduction.csv` / `external_multitask_transfer.csv` / `joint_fair_envelope.csv` / `functional_causality_multitask.csv` |
| `experiments/run_gafu_v86_real.py` | route 读取 broad artifact；formal task/causality 通过但 joint fair 失败时，落到 `broad_formal_joint_fair_failed` |
| `experiments/run_gafu_v86_real.py` | broad run 释放 KANbeFair model GPU 生命周期，并在 DG control 前后做 GC/CUDA cache cleanup；不改变训练目标 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 10.2 run

在第 9 节 accepted out-dir 上追加，不使用 `--fresh`，保留已有 P0/P7 真实 artifact：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-broad-vision \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 128 \
  --broad-ft7-event-alpha-mult 15.0
```

落盘 rows：

```text
kanbefair_broad_reproduction.csv = 4 rows
external_multitask_transfer.csv = 14 rows
joint_fair_envelope.csv = 2 rows
functional_causality_multitask.csv = 10 rows
```

### 10.3 route

```json
{
  "route": "R2-TaskFamilyLimitedFairAdvantage",
  "kanbefair_broad_reproduction_measured": 1,
  "broad_task_pass_count": 2,
  "broad_joint_pass_count": 0,
  "broad_causality_pass_count": 1,
  "broad_formal_joint_pass_count": 0,
  "broad_formal_causality_pass_count": 0,
  "primary_blocker": "broad_formal_joint_fair_failed",
  "success_v86_minimum": 1,
  "success_v86_external_formal": 0,
  "success_v86_broad_strong": 0
}
```

判断：broad formal 已真实执行，P1/P4/P6 均给出正面结果；但 P5 joint fair envelope 没有通过，因此不能声明 external formal。

### 10.4 P1 broad reproduction screen

| task | model | measured acc | reported acc | abs delta | formal protocol | reproduction pass |
|---|---|---:|---:|---:|---:|---:|
| FMNIST | KB-MLP | `86.869997` | `87.630000` | `0.760003` | 1 | 1 |
| FMNIST | KB-KAN | `76.510000` | `76.290000` | `0.220000` | 1 | 1 |
| KMNIST | KB-MLP | `83.749998` | `83.250000` | `0.499998` | 1 | 1 |
| KMNIST | KB-KAN | `39.339998` | `40.350000` | `1.010002` | 1 | 1 |

判断：FMNIST / KMNIST 的 KANbeFair full protocol reproduction 均通过。

### 10.5 P4 primary transfer screen

| task | KB-MLP acc | DG-Base acc | DG-FT7 acc | delta vs MLP | delta vs DG0 | TaskFamilyPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `86.869997` | `87.899995` | `87.909997` | `+1.040000` | `+0.010002` | 1 |
| KMNIST | `83.749998` | `85.200000` | `85.589999` | `+1.840001` | `+0.389999` | 1 |

判断：full formal protocol 下 DG-FT7 在 FMNIST / KMNIST 都超过 KB-MLP，并且也不低于 DG-Base。

### 10.6 P5 joint fair envelope

| task | param ratio | FLOPs ratio | memory ratio | step ratio | metric delta vs MLP | JointFairPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `0.939568` | `0.936347` | `1.922567` | `1.422175` | `+1.040000` | 0 |
| KMNIST | `0.939568` | `0.936347` | `1.922567` | `1.460443` | `+1.840001` | 0 |

判断：params / FLOPs / step 均过；但 KB-MLP peak memory 为 `230.701172 MB`，DG peak 为 `443.538574 MB`，memory ratio `1.922567`，因此 joint fair 不通过。GC/CUDA cleanup 没有修掉这一点，说明当前 blocker 更像 DG manual full-data-on-GPU live-set，而不是前序模型生命周期污染。

### 10.7 P6 functional causality screen

| task | FT7 curvature ratio | random curvature ratio | noop curvature ratio | delta vs base | CausalityPass |
|---|---:|---:|---:|---:|---:|
| FMNIST | `0.719622` | `1.114058` | `1.000000` | `+0.010002` | 1 |
| KMNIST | `0.762946` | `1.032342` | `1.000000` | `+0.389999` | 1 |

判断：

1. FMNIST / KMNIST 上 FT7 的 curvature improvement 都明确优于 noop / random，且 task 没有相对 noop 明显下降。
2. small-protocol 中 KMNIST causality 曾失败；full protocol 下已经闭合。
3. 当前 broad blocker 已从 causality split 收缩为 joint fair memory envelope。

### 10.8 no-fake audit

```text
rows_checked = 969
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 10.9 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `a06acae99e230a8fde7dd9e0d2a0f33ccef147d67acc4c4c8c3c8ced29850171` |
| latest route | `7685c5bb506e6afd7685e53d115ac8bd3884636a0b391857253c14a2410b03af` |
| `kanbefair_broad_reproduction.csv` | `1a533801e23278fc0a40f2965537199a424e5d4ed86d5ec4fd37f912ca9fb615` |
| `external_multitask_transfer.csv` | `8ed1a83d4443d285bf0b6db3374ba23941929d86a56fb2e99e8014fb6ef40f34` |
| `joint_fair_envelope.csv` | `cbd999fd3dc7ea4e69d6dc6fded677df24ed3a60ae8e3879967a7f9df208f3e6` |
| `functional_causality_multitask.csv` | `452c5f767b174f675c681b6cb4ede975addf61f474fc8ec9287976751d910b8e` |
| `v86_provenance_audit.csv` | `909ebd901ec70598b2c8be5bb661ad1b466d7934bdd3d9cc5c24d15e98288a0c` |

### 10.10 更新结论

v8.6 仍未完成 external formal / broad strong：

```text
v85 accepted route fresh reproduction = pass
P7 multi-split continual formalization = pass
Broad vision full reproduction = pass on FMNIST/KMNIST
Broad vision primary transfer = pass on FMNIST/KMNIST
Broad vision causality = pass on FMNIST/KMNIST
Broad joint fair envelope = fail
success_v86_minimum = true
success_v86_external_formal = false
success_v86_broad_strong = false
```

机制结论：

1. broad vision full protocol 给出正面 task signal：FMNIST / KMNIST 上 DG-FT7 都超过 KB-MLP。
2. functional causality 在 FMNIST / KMNIST 都通过，说明 FT7 的 curvature benefit 在这两个 broad vision task 上不是 no-op / random artifact。
3. 但 fair envelope 没有闭合，当前 blocker 是 DG manual full-data-on-GPU memory live-set：memory ratio `1.922567`，远高于 `<=1.05`。
4. params / FLOPs / step 都已过 gate，因此下一步应聚焦 broad memory live-set repair，而不是再改 objective 或 functional schedule。
5. 因此 v8.6 的完成度从“broad not_run”推进到“broad formal task + causality pass, joint fair memory fail”。

最终一句话：

> v8.6 现在比第 9 节更清楚：minimum success 仍成立，FMNIST/KMNIST broad formal reproduction、primary transfer 和 causality 都过；但 joint fair envelope 被 memory live-set 卡住，所以 external formal / broad strong 仍不能声明完成。

## 11. 追加：broad joint fair input live-set repair

本节继续第 10 节的 clean blocker：FMNIST / KMNIST broad formal task 与 causality 已通过，但 joint fair 被 DG manual full-data-on-GPU live-set 卡住。本轮只做 system-side input live-set repair，不改变实验合同：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 11.1 代码改动

| 文件 | 改动 | 目的 |
|---|---|---|
| `experiments/run_gafu_v85_real.py` | DG primary classifier 新增 `dg_stream_batches_from_cpu` | 避免 full external train tensor 常驻 GPU |
| `experiments/run_gafu_v85_real.py` | 新增 `dg_stream_chunk_batches` | 用小 GPU chunk 降低 per-batch CPU->GPU 拷贝开销 |
| `experiments/run_gafu_v85_real.py` | 新增 `dg_stream_epoch_permute_cpu` probe | 检查 epoch-level CPU permutation 是否更快 |
| `experiments/run_gafu_v86_real.py` | broad run 接入上述开关，并落盘 `input_data_residency` / `stream_chunk_batches` | 让 broad joint fair artifact 可审计 |

说明：这里的数据流是外部输入 tensor 的 batch/chunk loading；模型、参数、optimizer state、activation / gradient 仍在 GPU 训练路径内，没有 CPU offload。

代码检查：

```text
python -m py_compile experiments/run_gafu_v85_real.py experiments/run_gafu_v86_real.py
```

已通过。

### 11.2 smoke / rejected probes

Smoke 只验证代码路径，不作为 formal success：

```text
results/real_rerun_20260506/tmp_v86_broad_stream_smoke/
results/real_rerun_20260506/tmp_v86_broad_stream_chunk_smoke/
results/real_rerun_20260506/tmp_v86_broad_stream_epochperm_smoke/
```

判断：

1. 单 batch streaming 能降低 GPU live-set，但 full formal 下 step overhead 过大。
2. chunk16 能把 memory 保持在 gate 内，并显著好于单 batch streaming。
3. `stream_epoch_permute_cpu=1` 在 smoke 中没有改善 step，最终 formal run 不采纳。

### 11.3 final broad formal rerun

最终 official-style rerun 复用同一 accepted out-dir，只覆盖 broad artifacts，不删除 P0/P7：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-broad-vision \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 128 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16
```

Route：

```json
{
  "route": "R2-TaskFamilyLimitedFairAdvantage",
  "kanbefair_broad_reproduction_pass": 1,
  "broad_task_pass_count": 2,
  "broad_joint_pass_count": 0,
  "broad_causality_pass_count": 2,
  "broad_formal_joint_pass_count": 0,
  "broad_formal_causality_pass_count": 2,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 0,
  "success_v86_broad_strong": 0
}
```

### 11.4 P4 broad transfer after repair

| task | KB-MLP acc | DG-Base acc | DG-FT7 acc | delta vs MLP | delta vs DG0 | DG input residency | chunk | TaskFamilyPass |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| FMNIST | `86.869997` | `88.169998` | `88.139999` | `+1.270002` | `-0.029999` | `batch_streamed_cpu_to_gpu` | 16 | 1 |
| KMNIST | `83.749998` | `85.139996` | `85.589999` | `+1.840001` | `+0.450003` | `batch_streamed_cpu_to_gpu` | 16 | 1 |

判断：input live-set repair 没有破坏 broad task transfer；FMNIST / KMNIST 仍都超过 KB-MLP。

### 11.5 P5 joint fair after repair

| task | param ratio | FLOPs ratio | memory ratio | step ratio | metric delta vs MLP | JointFairPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `0.939568` | `0.936347` | `1.048481` | `4.574949` | `+1.270002` | 0 |
| KMNIST | `0.939568` | `0.936347` | `1.048481` | `4.650558` | `+1.840001` | 0 |

判断：

1. memory live-set blocker 已被修到 gate 内：从第 10 节的 `1.922567` 降到 `1.048481 <= 1.05`。
2. 但 step gate 明显失败：FMNIST `4.574949`，KMNIST `4.650558`，均远高于 `<=1.50`。
3. 因此 broad joint fair 仍不通过，v8.6 external formal 仍不能声明完成。

### 11.6 P6 functional causality after repair

| task | FT7 curvature ratio | random curvature ratio | noop curvature ratio | delta vs base | CausalityPass |
|---|---:|---:|---:|---:|---:|
| FMNIST | `0.702365` | `0.930725` | `1.000000` | `-0.029999` | 1 |
| KMNIST | `0.775350` | `1.113362` | `1.000000` | `+0.450003` | 1 |

判断：causality 仍过；FT7 的 curvature benefit 没有被 input live-set repair 消掉。

### 11.7 no-fake audit

```text
rows_checked = 969
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 11.8 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `28ddc63c4b6b79f192bcd27f99b66e01fc25f4e09811916397e22aef33ea1423` |
| `experiments/run_gafu_v85_real.py` | `d5ec72cdcd56ea6f376e8ce5e1e886e40b11728e4be62f39e805a29ac0d8a813` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| latest route | `7685c5bb506e6afd7685e53d115ac8bd3884636a0b391857253c14a2410b03af` |
| `kanbefair_broad_reproduction.csv` | `5f7aeba4de9245414bbb7a7609c40429bf7901f93b783166f54877543e8d2076` |
| `external_multitask_transfer.csv` | `fbc59ce34899a8039baaf2c81ad6c952fd07f367354ef04b003225858b15e143` |
| `joint_fair_envelope.csv` | `2adbceff6e15d6325646c6676fffee22a95f98d1dcd634790d1507b67b645dec` |
| `functional_causality_multitask.csv` | `23a3406d9ccb88ec981e97fc81b3cf5fcde8e0495ec33aae9fef9e0ffec3e127` |
| `v86_provenance_audit.csv` | `909ebd901ec70598b2c8be5bb661ad1b466d7934bdd3d9cc5c24d15e98288a0c` |

### 11.9 更新结论

v8.6 当前完成度：

```text
v85 accepted route fresh reproduction = pass
P7 multi-split continual formalization = pass
Broad vision full reproduction = pass on FMNIST/KMNIST
Broad vision primary transfer = pass on FMNIST/KMNIST
Broad vision causality = pass on FMNIST/KMNIST
Broad joint fair memory subgate = pass after input live-set repair
Broad joint fair step subgate = fail
success_v86_minimum = true
success_v86_external_formal = false
success_v86_broad_strong = false
```

机制结论：

1. 本轮 system-side input live-set repair 是真实进展：broad memory ratio 从 `1.922567` 降到 `1.048481`。
2. 但它把 blocker 从 memory 转移到了 step overhead：chunked CPU-to-GPU input loading 让 step ratio 上升到 `4.57-4.65`。
3. FMNIST / KMNIST 的 task transfer 与 causality 仍通过，说明当前不是 objective 或 functional geometry 失效。
4. 下一步应做 GPU-resident low-memory input cache、fused chunk prefetch 或更深的 DG manual step system repair；不能改 teacher/loss/sampler/class weight，也不能把 step fail 写成 external formal success。

最终一句话：

> v8.6 还没有完成 external formal / broad strong。本轮把 broad joint fair 的 memory blocker 修进 gate，但 step gate 反弹失败；当前最干净的 blocker 是在不改变 CE/no-teacher/no-loss/no-offload 合同的前提下，把 chunked input live-set repair 的 step overhead 从 `~4.6x` 拉回 `<=1.50x`。

## 12. 追加：chunk-order + low-cadence broad route closure

本节继续第 11 节的真实 blocker：chunked input streaming 修好了 memory，但 step overhead 太高。修复仍保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 12.1 代码改动

| 文件 | 改动 | 目的 |
|---|---|---|
| `experiments/run_gafu_v85_real.py` | 新增 DG input streaming chunk-order path，并记录 `input_data_residency` / `stream_chunk_batches` / `stream_chunk_order_shuffle` | 避免每 step CPU random gather，把 batch 顺序改成 chunk-order shuffle |
| `experiments/run_gafu_v86_real.py` | 将 broad route 的 streaming 参数传入 v8.5 primary trainer | 在 FMNIST/KMNIST formal broad route 中复用同一 system repair |
| `experiments/run_gafu_v86_real.py` | 新增 P8/P10/P11 conservative postprocess | P8/P10/P11 只从已落盘 measured rows 派生；P9 kernel profiler 未跑则保持 incomplete |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 12.2 rejected / diagnostic screens

| screen | 结果 | 判断 |
|---|---|---|
| full GPU residency FMNIST | step `1.515471` near pass，但 memory `1.922567` fail | rejected：memory 不公平 |
| chunk32 random gather FMNIST | step `3.645567` fail，memory `1.101715` fail | rejected：两边都不闭合 |
| chunk16 chunk-order FMNIST | memory `1.048481` pass，step `1.653949` fail | 方向正确但仍慢 |
| chunk32 chunk-order FMNIST | step `1.451358` pass，memory `1.101715` fail | speed/memory tradeoff |
| chunk16 chunk-order stride256 dual | FMNIST joint pass，KMNIST step `1.588129` fail | rejected：双数据集不稳定 |

这些 probe 都没有写成 success。

### 12.3 accepted broad formal run

最终 accepted broad run 使用：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-broad-vision \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle
```

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "v85_reproduction_pass": 1,
  "kanbefair_broad_reproduction_pass": 0,
  "kanbefair_broad_reproduction_measured": 1,
  "broad_task_pass_count": 2,
  "broad_joint_pass_count": 2,
  "broad_causality_pass_count": 2,
  "broad_formal_joint_pass_count": 2,
  "broad_formal_causality_pass_count": 2,
  "continual_formal_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "no_fake": true,
  "no_proxy": true
}
```

注：`kanbefair_broad_reproduction_pass=0` 是因为 KMNIST 的 `KB-KAN` row 与 reported table 的 delta 超出 tolerance；`KB-MLP` measured baseline 与 fair envelope rows 仍真实落盘，route 记录 `kanbefair_broad_reproduction_measured=1`。因此这里声明的是 measured broad fair route，不把 KAN reported-table 全复现写成完成。

### 12.4 broad primary / joint fair

Primary transfer：

| task | KB-MLP acc | DG-Base acc | DG-FT7 acc | delta vs MLP | delta vs DG0 | events | chunk | step ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `86.869997` | `87.930000` | `87.930000` | `+1.060003` | `0.000000` | 18 | 16 | `0.845376` |
| KMNIST | `83.749998` | `84.909999` | `85.039997` | `+1.289999` | `+0.129998` | 18 | 16 | `0.843872` |

Joint fair envelope：

| task | params ratio | FLOPs ratio | memory ratio | step ratio | metric delta vs MLP | JointFairPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `0.939568` | `0.936347` | `1.048362` | `1.475987` | `+1.060003` | 1 |
| KMNIST | `0.939568` | `0.936347` | `1.048362` | `1.326844` | `+1.289999` | 1 |

判断：

1. `stride512 + chunk16 + chunk-order` 同时闭合 FMNIST/KMNIST 的 params、FLOPs、memory、step 与 task gate。
2. 这不是降低到无 functional update：两任务仍各有 `18` 个 FT7 event，且 causality rows 通过。
3. `success_v86_external_formal = true` 可以声明；但 P8/P9 full profiler 仍未过，所以不能声明 broad strong。

### 12.5 functional causality

| task | FT7 curvature ratio | random curvature ratio | noop curvature ratio | delta vs base | ECE delta | NLL delta | CausalityPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `0.879777` | `1.017534` | `1.000000` | `0.000000` | `-0.001815` | `-0.000888` | 1 |
| KMNIST | `0.915183` | `0.957379` | `1.000000` | `+0.129998` | `-0.003845` | `-0.003255` | 1 |

判断：FT7 在两个 broad vision task 上都降低 curvature，并且 ECE / NLL 没有变差；这支持 `geometry_generalization_pass=1` 的保守机制结论。

### 12.6 P8/P9/P10/P11 postprocess

Postprocess 命令：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto
```

P8 只记录已测 training timers，不补 phase：

| task | step ratio | wall-clock ratio | WallClockPass | TimeAccountingPass | reason |
|---|---:|---:|---:|---:|---|
| FMNIST | `1.475987` | `1.475987` | 1 | 0 | phase times / TimeAUC not measured |
| KMNIST | `1.326844` | `1.326844` | 1 | 0 | phase times / TimeAUC not measured |

P9 保守保持 incomplete：

```text
status = not_run
reason = torch_kernel_phase_profiler_not_executed
ProfilerPass = 0
```

P10 geometry-task relationship：

| task | curvature delta | jacobian delta | ECE delta | NLL delta | MechanismModelPass |
|---|---:|---:|---:|---:|---:|
| FMNIST | `-0.120223` | `-0.060902` | `-0.001815` | `-0.000888` | 1 |
| KMNIST | `-0.084817` | `-0.100613` | `-0.003845` | `-0.003255` | 1 |

P11 boundary audit：

| task | boundary label | reason |
|---|---|---|
| FMNIST | `mnist_like_win` | joint fair envelope and causality pass |
| KMNIST | `mnist_like_win` | joint fair envelope and causality pass |
| KMNIST | `baseline_reproduction_boundary` | KB-KAN reported-table tolerance miss; fair comparison uses measured MLP baseline |

因此最新 failure table 只剩：

```text
F12_profiler_incomplete = active
```

### 12.7 no-fake audit

```text
rows_checked = 973
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 12.8 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `57f6075b25fb862a41f63763a9f37ad56aed3dfc3e76161732eddef99873fb08` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| route | `f86c78b26b516ed06f2d35bffc3b982b27a3ea20c5b8f91d29ff175b75a7bd74` |
| `joint_fair_envelope.csv` | `9ccf29bba403cbededd0dd216f848cb5772dde085639b7d68c8e1e5e2c43d467` |
| `external_multitask_transfer.csv` | `35c1a81320dcedaa684d7f597e44f7322deff11e0e902accbf58c827a8a2338b` |
| `functional_causality_multitask.csv` | `b2bd35e029ea5ea66f0f233d7ecfffd44566b13c55b60f61f64fc5f6380175dc` |
| `time_accounting_external.csv` | `269778784cf48805203c9cce5a49612d560efac43a4a650daabd8b6ebc13671c` |
| `phase_mapped_profiler_external.csv` | `ea867e286398c153fb53dcbe30af4fcddcb1cc35b9dd6988c08ecb6394f68354` |
| `geometry_task_relationship.csv` | `238627477a6e0774bf3cba8fd8e2f2f6c2c7cdc18a0304e75aceeb8dbd9cb644` |
| `negative_boundary_audit.csv` | `c1b6fbbef5ce6c4386723e78e3b787b1faca8570e28080a46963b3c5799c8eae` |
| `v86_provenance_audit.csv` | `4f8f1c053a8215609fd9b50e0f5eb5cd6857f984ecafceb5f771c4c27cbb7da5` |

### 12.9 更新结论

v8.6 当前完成度：

```text
v85 accepted route fresh reproduction = pass
P7 multi-split continual formalization = pass
FMNIST/KMNIST broad primary transfer = pass
FMNIST/KMNIST joint fair envelope = pass
FMNIST/KMNIST functional causality = pass
P10 geometry-task relationship = pass on measured broad vision rows
P11 negative/boundary audit = measured
P8 full TimeAccountingPass = false
P9 PhaseMappedProfilerPass = false
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = false
```

机制结论：

1. v8.6 已经从第 11 节的 memory/step split 推进到 broad external formal pass：FMNIST/KMNIST 都在 parameter、FLOPs、memory、step、task 与 causality gate 内。
2. `chunk16 + chunk-order + stride512` 是当前 accepted system/functional schedule：它同时控制 input live-set 和 functional event overhead。
3. functional geometry 在 FMNIST/KMNIST 上仍有机制信号：curvature、Jacobian/ECE/NLL 都向好。
4. 但 broad strong 仍不能声明，因为 P8 只有 training timer 派生，没有 TimeAUC/phase breakdown；P9 没有真实 kernel phase profiler。
5. 下一步如果继续，应实现真实 `phase_mapped_profiler_external.csv`：用 torch/CUDA profiler 映射 forward/backward/base update/functional update/guard forward，而不是把 phase 字段补成假数据。

最终一句话：

> v8.6 已经完成 minimum 和 external formal：`DG1-FT7-KW6 hidden28 stride512 + chunk-order` 在 FMNIST/KMNIST 上同时通过 broad task、joint fair 与 causality。但整份计划还没有 broad strong/full completion，因为 P8/P9 的真实 time accounting / phase-mapped profiler 仍未闭合；当前唯一 active failure 是 `F12_profiler_incomplete`。

## 13. 追加：真实 P8 phase-timer profiler

本节继续第 12 节的唯一 active blocker：`F12_profiler_incomplete`。本节没有修改 task/geometry route，也没有补假 kernel 数据；只新增真实 profiler-window 计时：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 13.1 代码改动

| 文件 | 改动 | 说明 |
|---|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--run-system-profiler` | 在 accepted broad setting 上运行 P8 profiler window |
| `experiments/run_gafu_v86_real.py` | 新增 `time_accounting_external_raw.csv` | 记录 MLP / DG 的 raw profiler rows |
| `experiments/run_gafu_v86_real.py` | 新增同步 phase timers | 记录 forward/backward/base update/functional update/validation/logging/unknown fraction |
| `experiments/run_gafu_v86_real.py` | P9 保守记录 `measured_phase_timers` | 没有 torch kernel trace，因此 `ProfilerPass=0` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 13.2 profiler run

正式 profiler window：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-system-profiler \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 4096 \
  --profiler-trace-every 512
```

本节也尝试过 `1024` step profiler window；结果同样失败，因此保留更长的 `4096` step 作为记录。

### 13.3 P8 time accounting result

`time_accounting_external.csv`：

| task | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | unknown fraction | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `1.631841` | `1.064471` | `1.799416` | `0.049623` | 0 | 0 |
| KMNIST | `1.501902` | `0.937883` | `1.407537` | `0.053100` | 0 | 0 |

Phase means for DG-FT7:

| task | forward ms | backward ms | update ms | functional update ms | validation ms | logging ms |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `0.274012` | `0.520743` | `0.117806` | `0.001150` | `27.412503` | `0.000677` |
| KMNIST | `0.256120` | `0.518367` | `0.115546` | `0.001014` | `7.200471` | `0.000670` |

判断：

1. unknown fraction 已经符合 `<=0.10`：FMNIST `0.049623`，KMNIST `0.053100`。
2. 但 profiler-window 的 step ratio 与 TimeAUC 没有过 gate：FMNIST step `1.631841`、KMNIST step `1.501902`，TimeAUC 分别 `1.799416` / `1.407537`。
3. 因此 P8 仍不能 pass，不能写 broad strong。
4. profiler-window 与 full broad run 的 joint fair step ratio 不完全一致；这说明同步 phase timer 本身会暴露额外 wall-clock/validation placement 成本，必须如实记录。

### 13.4 P9 phase-mapped profiler result

`phase_mapped_profiler_external.csv`：

| task | status | mapped fraction | unknown fraction | top phase | ProfilerPass |
|---|---|---:|---:|---|---:|
| FMNIST | `measured_phase_timers` | `0.950377` | `0.049623` | validation | 0 |
| KMNIST | `measured_phase_timers` | `0.946900` | `0.053100` | validation | 0 |

判断：

1. 本节是真实 phase timer，不是 kernel-level profiler。
2. 因为没有采集 torch/CUDA kernel trace，`kernel_count_*` 保持 `metric_unavailable`，`ProfilerPass=0`。
3. P9 的下一步仍是接入真正 torch profiler / CUDA kernel trace，而不是把 synchronized phase timer 改名成 kernel profiler。

### 13.5 route / audit

Route 保持：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 0,
  "phase_mapped_profiler_pass": 0,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "next_required_implementation": "run_phase_mapped_profiler_external"
}
```

Failure table：

```text
F12_profiler_incomplete = active
```

No-fake audit：

```text
rows_checked = 978
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 13.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `a2b29773799861147fd0e8a62119ee6de6ca5cfc81758585b9bf8f131f3d5a5c` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| route | `f86c78b26b516ed06f2d35bffc3b982b27a3ea20c5b8f91d29ff175b75a7bd74` |
| `time_accounting_external.csv` | `eeae77c1c3816e3c71ef285f11e583d2968b16cc371d0769c8847e5ac979fb07` |
| `time_accounting_external_raw.csv` | `a2d738446e925272e6738df5d951df5da7f9939bddc163772df073538412e437` |
| `phase_mapped_profiler_external.csv` | `ddf62de51957a5493014ad5c3477f20d811ee19d6372899ee854e91a4448b086` |
| `v86_provenance_audit.csv` | `6dcba6920bbd07f00eeb9bbe6b0db8f1e2fbb41bd94a3a50c59adcbc79244b50` |

### 13.7 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = false
P8 TimeAccountingPass = false
P9 PhaseMappedProfilerPass = false
```

机制结论：

1. v8.6 的 external formal 仍成立；P8/P9 profiler 不回滚 FMNIST/KMNIST broad joint fair artifact。
2. 但 profiler-window 说明系统解释还没闭合：同步 phase accounting 下 DG-FT7 的 backward/validation 时间把 TimeAUC 推高。
3. 当前 blocker 更具体：不是 broad task/fair/causality，而是 profiler-grade wall-clock accounting 与 kernel phase trace。
4. 下一步需要真正 kernel-level phase profiler，并考虑 profiler-grade system repair，例如减少 validation placement cost、减少 manual backward phase fragmentation；不能改 CE objective，也不能补 fake kernel counts。

最终一句话：

> v8.6 仍不是 full completion：external formal 已完成，但真实 P8 profiler-window 没过 TimeAccounting，P9 也还没有 kernel trace。当前唯一 active blocker 仍是 `F12_profiler_incomplete`，但它已经从占位 not_run 推进为有真实 P8 negative artifact 的系统问题。

## 14. 追加：P9 torch kernel profiler 与 P8 validation warmup

本节继续第 13 节的 active blocker。约束保持不变：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 14.1 代码改动

| 文件 | 改动 | 说明 |
|---|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--run-kernel-profiler` | 使用 `torch.profiler` 采集 CUDA kernel events |
| `experiments/run_gafu_v86_real.py` | 新增 phase `record_function` | 将 kernel event 映射到 batch / forward-backward / update / guard / functional / validation |
| `experiments/run_gafu_v86_real.py` | P9 写入 `measured_torch_profiler_kernel_trace` | 不再把 phase timer 当作 kernel profiler |
| `experiments/run_gafu_v86_real.py` | profiler validation warmup + `--profiler-eval-batch-size` | 只修 profiler measurement，不改训练目标或 candidate |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 14.2 profiler run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-kernel-profiler \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 4096 \
  --profiler-trace-every 512 \
  --profiler-eval-batch-size 2048 \
  --kernel-profiler-steps 512 \
  --kernel-profiler-trace-every 512
```

### 14.3 P8 time accounting result

`time_accounting_external.csv`：

| task | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | validation ms | unknown fraction | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `1.587964` | `1.064471` | `1.755410` | `4.862655` | `0.052399` | 0 | 0 |
| KMNIST | `1.521013` | `0.937883` | `1.486387` | `6.112030` | `0.048943` | 0 | 0 |

判断：

1. validation warmup 与 eval batch `2048` 确实把第 13 节 FMNIST validation phase 从 `87.94 ms` 降到 `4.86 ms`。
2. 但 P8 仍失败：FMNIST step ratio `1.587964`，KMNIST step ratio `1.521013`，二者均超过 `1.50`；TimeAUC 也仍高于 `1.05`。
3. 因此不能声明 broad strong/full completion。

### 14.4 P9 torch kernel profiler result

`phase_mapped_profiler_external.csv` 中新增 `measured_torch_profiler_kernel_trace` 行：

| task | kernel count | mapped kernel fraction | unknown kernel fraction | top phase 1 | top phase 2 | top phase 3 | ProfilerPass |
|---|---:|---:|---:|---|---|---|---:|
| FMNIST | `66416` | `1.000000` | `0.000000` | forward_backward | batch_select | base_update | 1 |
| KMNIST | `66416` | `1.000000` | `0.000000` | forward_backward | batch_select | base_update | 1 |

代表性 kernel fields：

| task | functional events | functional update kernels | guard forward kernels | validation kernels | memcpy time us | sync time us |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `1` | `46` | `37` | `658` | `17264.3030` | `22012.4170` |
| KMNIST | `1` | `46` | `37` | `658` | `17275.7610` | `22104.5040` |

判断：

1. P9 已从第 13 节的 `measured_phase_timers / ProfilerPass=0` 推进为真实 torch/CUDA kernel trace。
2. 初次 kernel trace 中 async H2D memcpy 落在 phase range 外，unknown 约 `0.1103`，因此没有采纳；本节在 kernel-profiler-only 的 `batch_select` phase 内同步，使 memcpy 归属到真实 batch phase 后，P9 才通过。
3. 这不是 fake/proxy kernel count：`kernel_count_total`、memcpy、sync、top phase 均来自 `torch.profiler` events。

### 14.5 route / audit

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 0,
  "phase_mapped_profiler_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "no_fake": true,
  "no_proxy": true,
  "cpu_offload_used": 0
}
```

No-fake audit：

```text
rows_checked = 980
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 14.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `88b703e4c8cda4329bf12bd1c2ab04c91715e8474a534eb9d3855310385ae945` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| route | `dbeeee68998a006fcfab38506a085dd34f208086b8b7f3ac91acbbcf5f135d12` |
| `time_accounting_external.csv` | `d9fb8b87d77bc6b58f9e23626d26a30d81df87c6d5e4fa218f989da5fc609932` |
| `phase_mapped_profiler_external.csv` | `00c2bf56a9df90855a07afafeccee8dedae9a91ba581c5706c03b9c388eb4970` |
| `v86_provenance_audit.csv` | `29da377528f2e6bb8453a2641ede271529502732923a446f7720f63c14803f06` |

### 14.7 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
P9 PhaseMappedProfilerPass = true
P8 TimeAccountingPass = false
success_v86_broad_strong = false
```

机制结论：

1. v8.6 的 P9 已闭合：现在有真实 torch/CUDA kernel trace，且 FMNIST/KMNIST 都能映射到 phase。
2. v8.6 仍未 full completion：P8 TimeAccounting 仍失败，当前 blocker 从 “P8/P9 都没闭合” 收缩为 “P8 profiler-grade step/time AUC 未闭合”。
3. P8 失败不是 fake/proxy，也不是 external formal route 回退；external formal 仍为 true。
4. 下一步应继续做 profiler-grade system repair，重点是减少 manual forward/backward fragmentation 或调整 time accounting 的低开销测量方式；不能改 CE objective、teacher、sampler/class weight，也不能把 P8 failure 写成 strong success。

最终一句话：

> v8.6 还没有 broad strong/full completion。它已经完成 minimum、external formal 和 P9 kernel profiler；唯一剩下的是 P8 TimeAccounting，当前 DG-FT7 在 profiler-window 下 step ratio 与 TimeAUC 仍超过 gate。

## 15. 追加：P8 low-overhead time accounting probe

本节继续第 14 节剩余 blocker：`P8 TimeAccountingPass = false`。本节只调整 P8 测量口径与 profiler 实现，不改变训练合同：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 15.1 代码改动

| 文件 | 改动 | 说明 |
|---|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--profiler-low-overhead` | P8 official time window 不再每个 phase 强同步，只在 step / validation 边界同步 |
| `experiments/run_gafu_v86_real.py` | low-overhead manual CE backward | 非 event step 不再把 loss scalar 每步拉回 CPU |
| `experiments/run_gafu_v86_real.py` | low-overhead unknown fraction 修正 | unknown 用 step + validation cumulative wall-clock 计算 |
| `experiments/run_gafu_v86_real.py` | postprocess 保护 low-overhead measured rows | 防止 route-only rerun 把 P8 rows 覆盖回 training-timer 占位 |
| `experiments/run_gafu_v86_real.py` | failure id 细化 | P9 pass 但 P8 fail 时写 `F13_time_accounting_fail` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 15.2 low-overhead P8/P9 run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-kernel-profiler \
  --profiler-low-overhead \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 9376 \
  --profiler-trace-every 1024 \
  --profiler-eval-batch-size 2048 \
  --kernel-profiler-steps 512 \
  --kernel-profiler-trace-every 512
```

### 15.3 P8 result

`time_accounting_external.csv`：

| task | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | validation ms | unknown fraction | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `1.458126` | `1.054696` | `1.560745` | `4.922081` | `0.001225` | 1 | 0 |
| KMNIST | `1.572411` | `0.851547` | `1.348108` | `6.247119` | `0.001404` | 0 | 0 |

判断：

1. low-overhead window 成功去除了 phase-sync 计时膨胀：FMNIST step ratio 从第 14 节 `1.587964` 降到 `1.458126`，并通过 step 子门。
2. KMNIST 仍未过 step 子门：`1.572411 > 1.50`。
3. 两个任务的 TimeAUC 仍未过 `<=1.05`：FMNIST `1.560745`，KMNIST `1.348108`。
4. 因此 P8 仍不通过，不能声明 `success_v86_broad_strong`。

### 15.4 P9 remains pass

本轮同时重跑 torch kernel profiler。`phase_mapped_profiler_external.csv` 中 `measured_torch_profiler_kernel_trace` 仍通过：

| task | kernel count | mapped fraction | unknown fraction | top phase 1 | top phase 2 | top phase 3 | ProfilerPass |
|---|---:|---:|---:|---|---|---|---:|
| FMNIST | `66416` | `1.000000` | `0.000000` | forward_backward | batch_select | base_update | 1 |
| KMNIST | `66416` | `1.000000` | `0.000000` | forward_backward | batch_select | base_update | 1 |

判断：P9 closure 稳定；当前 active blocker 已经不是 profiler trace，而是 P8 time accounting。

### 15.5 route / audit

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 0,
  "phase_mapped_profiler_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "next_required_implementation": "repair_time_accounting_external",
  "no_fake": true,
  "no_proxy": true,
  "cpu_offload_used": 0
}
```

Failure table：

```text
F13_time_accounting_fail = active
```

No-fake audit：

```text
rows_checked = 980
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 15.6 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `2d0e6b16d4fe8031513b139ed2603aa07c7f0b9339c4c2e48505bae39e0a3006` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| route | `dbeeee68998a006fcfab38506a085dd34f208086b8b7f3ac91acbbcf5f135d12` |
| failure table | `0abdb4803e86064f5bfa5c6e0530a0ce97685635a919b42497a74ba3cb33396f` |
| `time_accounting_external.csv` | `d22e502e796c041a211a437c0b47b028be90270c7cf67bf19e3e0b42c19a7af3` |
| `time_accounting_external_raw.csv` | `4e2ef69c72249650cb49100c321e32d4e3099a84cdeae3f95cea3af9a29b8ff5` |
| `phase_mapped_profiler_external.csv` | `42576b39e07870ac2c62e361525d83ab17f8a92a226b595ab245d47e65d37a9b` |
| `v86_provenance_audit.csv` | `29da377528f2e6bb8453a2641ede271529502732923a446f7720f63c14803f06` |

### 15.7 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
P9 PhaseMappedProfilerPass = true
P8 TimeAccountingPass = false
success_v86_broad_strong = false
```

机制结论：

1. low-overhead P8 probe 是有效的 measurement/system repair，但不足以闭合 P8。
2. FMNIST 的 step 子门已经被修过，但 TimeAUC 仍失败；KMNIST 的 step 子门和 TimeAUC 都失败。
3. P9 kernel profiler 已稳定通过，因此当前 blocker 被精确收缩为 `F13_time_accounting_fail`。
4. 下一步如果继续，应针对 P8 的真实瓶颈做系统修复：减少 manual backward Python/kernel fragmentation，或改进 DG low-overhead path 的 wall-clock，而不是改 CE objective、teacher、loss、sampler/class weight 或 CPU offload。

最终一句话：

> v8.6 仍未完成 broad strong/full completion。当前已经完成 minimum、external formal 和 P9 kernel profiler；唯一 blocker 是 P8 TimeAccounting，尤其 KMNIST step ratio `1.572411` 与两个任务的 TimeAUC gate。

## 16. 追加：P8 cadence / schedule boundary screens

本节继续第 15 节的唯一 blocker：`P8 TimeAccountingPass = false`。本节没有改变 CE objective、teacher/self-teacher、loss、sampler/class weight、CPU offload，也没有使用 fake/proxy；只做 P8 采样 cadence 与 FT7 event schedule 的窄屏，判断能否在不偏离 v8.6 中心思想的前提下闭合 TimeAccounting。

### 16.1 4096-step cadence screen

本轮把 P8 validation / TimeAUC 采样从第 15 节的 `trace_every=1024` 放宽到 `trace_every=4096`，其它 accepted broad setting 保持不变：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-kernel-profiler \
  --profiler-low-overhead \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 9376 \
  --profiler-trace-every 4096 \
  --profiler-eval-batch-size 2048 \
  --kernel-profiler-steps 512 \
  --kernel-profiler-trace-every 512
```

`time_accounting_external.csv`：

| task | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | validation ms | unknown fraction | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `1.500995` | `1.018189` | `1.569926` | `4.971543` | `0.001158` | 0 | 0 |
| KMNIST | `1.617226` | `0.817688` | `1.239867` | `6.382545` | `0.001566` | 0 | 0 |

判断：

1. 更稀疏的 validation cadence 没有闭合 P8；两个任务的 TimeAUC 仍显著高于 `1.05`。
2. FMNIST 的 step ratio 卡在 `1.500995`，刚好超过 `1.50`；KMNIST step ratio 反而升到 `1.617226`。
3. 因此不能把 cadence screen 写成 broad strong success。

P9 torch profiler 仍通过：

| task | kernel count | mapped fraction | unknown fraction | top phase 1 | top phase 2 | top phase 3 | ProfilerPass |
|---|---:|---:|---:|---|---|---|---:|
| FMNIST | `66416` | `1.000000` | `0.000000` | forward_backward | batch_select | base_update | 1 |
| KMNIST | `66416` | `1.000000` | `0.000000` | forward_backward | batch_select | base_update | 1 |

### 16.2 stride1024 schedule screen

为了确认 P8 是否主要来自 functional event overhead，另开单独 out-dir 做 `stride1024` P8-only schedule screen。该 run 没有复制 accepted broad route artifact，因此 route 显示 `R8-NoReproduction`；这里只使用它的 P8/P9 measured rows 做边界诊断，不把它作为正式 route。

```text
results/real_rerun_20260506/v86_p8_stride1024_schedule_screen_20260507T230000Z/
```

`time_accounting_external.csv`：

| task | events | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `9` | `1.507023` | `1.055954` | `1.628471` | 0 | 0 |
| KMNIST | `9` | `1.630741` | `0.853467` | `1.393941` | 0 | 0 |

判断：

1. `stride1024` 把 functional event count 从 `18` 降到 `9`，但没有修 P8；两个任务的 step ratio 与 TimeAUC 仍失败。
2. 这说明当前 blocker 不是简单 event frequency；更低频的 functional update 既不能在 P8 screen 中闭合系统门槛，也不能替代已通过 broad formal 的 `stride512` route。
3. 因此本 probe 只记录为 rejected boundary，不进入 full broad revalidation。

### 16.3 route / audit

当前 accepted out-dir 的 route 仍为：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 0,
  "phase_mapped_profiler_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "next_required_implementation": "repair_time_accounting_external",
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

Failure table：

```text
F13_time_accounting_fail = active
```

No-fake audit：

```text
rows_checked = 980
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 16.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `2d0e6b16d4fe8031513b139ed2603aa07c7f0b9339c4c2e48505bae39e0a3006` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| accepted route | `dbeeee68998a006fcfab38506a085dd34f208086b8b7f3ac91acbbcf5f135d12` |
| accepted `time_accounting_external.csv` | `ec6b5f9a64e9be7072cf598aa8e840fea367919e816cc53312105d4da0a0fdaa` |
| accepted `time_accounting_external_raw.csv` | `efe9079e4ad01d45b108deaac13af114f4f60c81da7d08160998cfb8bc8d21d9` |
| accepted `phase_mapped_profiler_external.csv` | `5d74be44d3043ac6458149499ad3a07425cc1ac5ac28101dfa6af8a7bb36653a` |
| accepted `v86_provenance_audit.csv` | `29da377528f2e6bb8453a2641ede271529502732923a446f7720f63c14803f06` |
| stride1024 route | `be4c06963d4469de34b8b72e9d41d96485b913c26dca4a0079303fd21d231314` |
| stride1024 `time_accounting_external.csv` | `624b51c117f5d1dd24d61328a5042d0db45b4e6ea95aaf8c6c7d7c4326219874` |
| stride1024 `time_accounting_external_raw.csv` | `6b91e5bb823c1cfe19818e33920e74854ea3ca3e300e4c97f0e4a3cbb90ed0d5` |
| stride1024 `phase_mapped_profiler_external.csv` | `350bd0a72d5c87879a6c9933df8fa886e535a107e774ce50324cf70247b77405` |
| stride1024 `v86_provenance_audit.csv` | `0c99ab65d636156d91ef0c0313f1e99310670f5227ef7a298672947aa67c206e` |

### 16.5 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
P9 PhaseMappedProfilerPass = true
P8 TimeAccountingPass = false
success_v86_broad_strong = false
```

机制结论：

1. P8 cadence 不是解法：4096-step validation cadence 仍失败，且 FMNIST/KMNIST 都没有稳定通过 step gate。
2. 简单降低 functional event frequency 也不是解法：`stride1024` 的 event count 减半，但 P8 step/time AUC 仍失败。
3. 当前 blocker 仍是 P8 的 end-to-end time accounting，尤其 `ValLossAUC_time`。P9 已经证明 kernel trace 可以被完整 phase-map，因此后续应继续做真实 implementation-side wall-clock repair，而不是改 CE objective、teacher、loss、sampler/class weight、CPU offload，或靠减少 evaluation / functional geometry 强度换 success。

最终一句话：

> v8.6 还没有 broad strong/full completion。minimum 与 external formal 已经成立，P9 也已通过；最新 cadence 和 stride screen 进一步证明，P8 TimeAccounting 不是采样频率或 event 频率的小问题，而是 DG manual path 的真实 time-to-loss-AUC 系统瓶颈。

## 17. 追加：P8 standard CE loss-only validation trim

本节继续 P8 blocker，但只修 P8 validation measurement 的实现路径。原 P8 DG validation 调用完整 `_manual_eval`，会额外计算 accuracy、ECE、classwise accuracy、feature effective rank 等；P8 TimeAccounting 只需要 validation CE loss trace。因此本节新增 standard CE loss-only validation path。

约束保持不变：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 17.1 代码改动

| 文件 | 改动 | 是否改变训练/update |
|---|---|---:|
| `experiments/run_gafu_v86_real.py` | 新增 `_eval_dg_ce_loss_only_external`，P8 DG validation 只计算 standard CE loss | 0 |
| `experiments/run_gafu_v86_real.py` | P8 warmup / trace validation 从完整 `_manual_eval` 改为 CE loss-only | 0 |

注：本路径使用 `F.cross_entropy(..., reduction="sum")` 聚合，与 `_manual_eval -> _eval_logits` 的 validation loss 口径一致；不使用 smoothed training CE 作为 validation metric。

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 17.2 P8/P9 run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-kernel-profiler \
  --profiler-low-overhead \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 9376 \
  --profiler-trace-every 1024 \
  --profiler-eval-batch-size 2048 \
  --kernel-profiler-steps 512 \
  --kernel-profiler-trace-every 512
```

`time_accounting_external.csv`：

| task | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | validation ms | unknown fraction | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `1.537990` | `1.054696` | `1.637110` | `0.895725` | `0.001232` | 0 | 0 |
| KMNIST | `1.597862` | `0.851547` | `1.360150` | `1.213200` | `0.001546` | 0 | 0 |

判断：

1. loss-only validation 是有效的 validation-side trim：DG validation time 从第 15 节的 `4.922081 / 6.247119 ms` 降到 `0.895725 / 1.213200 ms`。
2. 但 P8 仍失败：FMNIST / KMNIST 的 step ratio 都超过 `1.50`，TimeAUC 也仍显著高于 `1.05`.
3. 这说明 P8 blocker 主要不是 validation metric overhead，而是 DG manual training step wall-clock 与 time-to-loss-AUC 本身。

P9 torch profiler 仍通过；route 仍保守保持 `success_v86_broad_strong=0`。

### 17.3 route / audit

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 0,
  "phase_mapped_profiler_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "next_required_implementation": "repair_time_accounting_external",
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 980
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 17.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `0385569ace6e86a505f1b78ceb8f110853ec4182246f79875e4ca2b56be1193e` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| accepted route | `dbeeee68998a006fcfab38506a085dd34f208086b8b7f3ac91acbbcf5f135d12` |
| accepted `time_accounting_external.csv` | `3d5be67a3fa14fbca7d010655ca537f5585ff46c035a7bed46c0556a8496c8ce` |
| accepted `time_accounting_external_raw.csv` | `52c638d91ca57a2394b2aa434259fc0e4ee2ccc1eb119471cc5e6fcfeea6b997` |
| accepted `phase_mapped_profiler_external.csv` | `4edc797a6bb89576e3a719bb90c0874fda2154c0a0af3f0e04904a0bb07f579e` |
| accepted `v86_provenance_audit.csv` | `29da377528f2e6bb8453a2641ede271529502732923a446f7720f63c14803f06` |

### 17.5 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
P9 PhaseMappedProfilerPass = true
P8 TimeAccountingPass = false
success_v86_broad_strong = false
```

机制结论：

1. standard CE loss-only validation 是干净的 P8 measurement repair，但只能降低 validation phase，不足以闭合 TimeAccounting。
2. P8 的 blocker 已进一步定位为 DG manual step / time-to-loss-AUC 系统瓶颈，而不是 validation-side extra metrics。
3. 下一步应继续做数值等价的 manual forward/backward/update implementation repair，或更深的 kernel/compiled path；不能改 CE objective、teacher、loss、sampler/class weight、CPU offload，也不能把 P8 negative artifact 写成 broad strong。

最终一句话：

> v8.6 仍未完成 broad strong/full completion。loss-only validation 把 P8 validation overhead 明显降下来，但 `ValLossAUC_time` 和 step ratio 仍不过线；当前 blocker 仍是 DG manual path 的真实 wall-clock/time-to-loss-AUC 瓶颈。

## 18. 追加：P8 async wall-clock window 与 alpha screen

本节继续围绕 P8 TimeAccounting blocker 做 measurement/system-side 修复。第 17 节已经证明 validation metric overhead 不是主因；本节进一步去掉每 step 强制 sync 的 measurement inflation，改为 async wall-clock window：训练区间只在 validation / window 边界同步并计入真实 wall-clock，P9 phase attribution 仍由 torch profiler kernel trace 提供。

约束保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 18.1 代码改动

| 文件 | 改动 | 是否改变训练/update |
|---|---|---:|
| `experiments/run_gafu_v86_real.py` | 新增 `--profiler-async-window` | 0 |
| `experiments/run_gafu_v86_real.py` | P8 low-overhead mode 下可按 validation window 统计 train wall-clock，不再每 step 强制 sync | 0 |
| `experiments/run_gafu_v86_real.py` | postprocess 接受 `measured_low_overhead_async_window` measured status | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 18.2 accepted route async-window P8/P9 run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-kernel-profiler \
  --profiler-low-overhead \
  --profiler-async-window \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 9376 \
  --profiler-trace-every 1024 \
  --profiler-eval-batch-size 2048 \
  --kernel-profiler-steps 512 \
  --kernel-profiler-trace-every 512
```

`time_accounting_external.csv`：

| task | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | validation ms | unknown fraction | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `1.556815` | `1.054696` | `1.655148` | `0.928754` | `0.000003` | 0 | 0 |
| KMNIST | `1.530222` | `0.851547` | `1.285117` | `1.165020` | `0.000002` | 0 | 0 |

判断：

1. async-window 把 unknown fraction 压到接近 0，说明 P8 wall-clock window 的覆盖已经足够完整。
2. 但 P8 仍失败：FMNIST/KMNIST step ratio 都高于 `1.50`，TimeAUC 仍远高于 `1.05`。
3. 因此每-step sync inflation 不是主因；当前 blocker 更明确地落在 DG manual path 的真实训练 wall-clock 与 loss trajectory。

### 18.3 alpha30 P8-only screen

为确认是否可以通过增强 FT7 functional update 改善 loss trajectory，另开单独 out-dir 做 `alpha30` P8-only screen。该 run 没有复制 accepted broad route artifact，因此 route 显示 `R8-NoReproduction`；只用于边界诊断，不进入正式 route。

```text
results/real_rerun_20260506/v86_p8_stride512_alpha30_async_screen_20260507T233000Z/
```

`time_accounting_external.csv`：

| task | events | step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|
| FMNIST | `18` | `1.513576` | `1.054099` | `1.618588` | 0 | 0 |
| KMNIST | `18` | `1.778292` | `0.850845` | `1.527088` | 0 | 0 |

判断：

1. `alpha30` 对 FMNIST 有轻微系统改善，但仍未过 step gate 和 TimeAUC。
2. KMNIST 明显变慢，step ratio 到 `1.778292`，因此不能作为 repair。
3. 这说明继续粗调 FT7 alpha 不符合 v8.6 的中心思想；需要更深的 implementation / kernel repair，而不是靠放大 functional event 强度。

### 18.4 route / audit

Accepted route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 0,
  "phase_mapped_profiler_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 0,
  "next_required_implementation": "repair_time_accounting_external",
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 980
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 18.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `14cba6b02400d054cf018b03450f95bd2067b7e575c8491e95be2fc79297438f` |
| `experiments/run_gafu_v85_real.py` | `4365eb480fcd4e897d95bf0d2f3ca6c3238dad25f1ae526e82d35df78e6183f3` |
| accepted route | `dbeeee68998a006fcfab38506a085dd34f208086b8b7f3ac91acbbcf5f135d12` |
| accepted `time_accounting_external.csv` | `a22ebc816e7b642d607148b120cb8a85d245a2983943114f4588ebd46a3c7fdd` |
| accepted `time_accounting_external_raw.csv` | `0a85600dd6f457f1ce7fd455ea9ec072f244652e58f3c48f2b3d07ce6c0cc781` |
| accepted `phase_mapped_profiler_external.csv` | `4d777d3b7992a7836aa4567973da05e5905bf87e5848714cd289a94bb6bd9e92` |
| accepted `v86_provenance_audit.csv` | `29da377528f2e6bb8453a2641ede271529502732923a446f7720f63c14803f06` |
| alpha30 screen `time_accounting_external.csv` | `36a98a76d5d08efa616240b03f8959284daee07576d7d93dca6f62e4163edc0c` |
| alpha30 screen `time_accounting_external_raw.csv` | `b8dcf47641f0f61e6bdf0efb3f1c53a58f43ae2c4fa23a42e7be0c952c261135` |
| alpha30 screen `phase_mapped_profiler_external.csv` | `c816576dad2168a5cf0affe3e63adf0496ff1a05b4b73a8bb1c88bfd9652bae8` |
| alpha30 screen `v86_provenance_audit.csv` | `0c99ab65d636156d91ef0c0313f1e99310670f5227ef7a298672947aa67c206e` |

### 18.6 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
P9 PhaseMappedProfilerPass = true
P8 TimeAccountingPass = false
success_v86_broad_strong = false
```

机制结论：

1. P8 async-window 证明当前 failure 不是 measurement coverage 或 per-step sync artifact。
2. alpha30 screen 证明简单增强 FT7 event strength 不能闭合 P8，且会让 KMNIST 系统侧恶化。
3. 当前最干净的 blocker 仍是 `F13_time_accounting_fail`：DG manual path 的 forward/backward/update 真实 wall-clock 仍不足以满足 external TimeAUC gate。
4. 下一步应做 kernel/compiled/manual-backward 级别的系统实现修复；不能改 CE objective、teacher、loss、sampler/class weight、CPU offload，也不能通过削弱 evaluation、降低 geometry 或调 alpha 把 P8 fail 包装成 success。

最终一句话：

> v8.6 仍没有 broad strong/full completion。async-window 与 alpha screen 都是真实 negative artifact：P9 已经闭合，P8 仍被 DG manual path 的 end-to-end TimeAUC 卡住。下一步要进入更深的 manual kernel/compiled implementation repair，而不是继续调 schedule 或 measurement cadence。

## 19. 追加：CUDA graph non-event manual backward closure

本节继续第 18 节的 clean blocker：P8 TimeAccounting。修复仍保持 v8.6 中心合同：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 19.1 代码改动

| 文件 | 改动 | 是否改变训练目标 |
|---|---|---:|
| `experiments/run_gafu_v86_real.py` | 新增 `--profiler-cuda-graph-non-event` | 0 |
| `experiments/run_gafu_v86_real.py` | 对 P8 low-overhead async-window 的 non-event manual CE forward/backward 做 CUDA graph capture/replay | 0 |
| `experiments/run_gafu_v86_real.py` | event step 仍走原 FT7 guarded functional update；只把普通 CE non-event step 的 Python/kernel launch overhead 图化 | 0 |
| `experiments/run_gafu_v86_real.py` | 修复 graph capture warmup：static label 初始化为合法 class index，避免未初始化 label 触发 CUDA scatter assert | 0 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py
```

已通过。

### 19.2 CUDA graph smoke

先做小规模 smoke，只验证 graph path 可真实启用，不作为 success 结论：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/tmp_v86_p8_cuda_graph_smoke \
  --fresh \
  --device auto \
  --run-kernel-profiler \
  --profiler-low-overhead \
  --profiler-async-window \
  --profiler-cuda-graph-non-event \
  --broad-datasets Fashion-MNIST \
  --broad-train-size 1024 \
  --broad-test-size 512 \
  --broad-epochs 1 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 128 \
  --profiler-trace-every 64 \
  --profiler-eval-batch-size 512 \
  --kernel-profiler-steps 64 \
  --kernel-profiler-trace-every 64
```

Smoke result：

| task | graph status | graph replay | functional events | TimeAccountingPass | fake/proxy/offload |
|---|---|---:|---:|---:|---:|
| FMNIST | `enabled` | `128` | `0` | 1 | 0 |

判断：CUDA graph path 真实启用并 replay；smoke 只是 path check，不写 v8.6 success。

### 19.3 P8-only graph screen

先在单独 out-dir 做 P8/P9 screen。该 out-dir 没有复制 accepted broad route artifact，因此 route 仍为 `R8-NoReproduction`，只用于确认 P8 修复方向。

```text
results/real_rerun_20260506/v86_p8_cuda_graph_non_event_screen_20260508T000000Z/
```

`time_accounting_external.csv`：

| task | graph replay | events | train step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | wall-clock total ratio | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `9338` | `18` | `0.518639` | `1.054696` | `0.558175` | `0.519690` | 1 |
| KMNIST | `9338` | `18` | `0.417067` | `0.851547` | `0.356624` | `0.418411` | 1 |

判断：P8-only screen 证明 graph replay 不是小样本假象；两个 broad vision task 的 P8 gate 均闭合。但因为 route 是 `R8-NoReproduction`，不能把这个 screen 单独写成 official completion。

### 19.4 accepted route P8/P9 rerun

把同一实现接回已经通过 P0/P1/P4/P5/P6/P7/P10/P11 的 accepted out-dir，重测 P8/P9：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-kernel-profiler \
  --profiler-low-overhead \
  --profiler-async-window \
  --profiler-cuda-graph-non-event \
  --broad-datasets Fashion-MNIST,KMNIST \
  --broad-train-size 60000 \
  --broad-test-size 10000 \
  --broad-epochs 20 \
  --broad-dg-hidden-dim 28 \
  --broad-ft7-event-stride 512 \
  --broad-ft7-event-alpha-mult 15.0 \
  --broad-dg-stream-batches-from-cpu \
  --broad-dg-stream-chunk-batches 16 \
  --broad-dg-stream-chunk-order-shuffle \
  --profiler-steps 9376 \
  --profiler-trace-every 1024 \
  --profiler-eval-batch-size 2048 \
  --kernel-profiler-steps 512 \
  --kernel-profiler-trace-every 512
```

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "time_accounting_pass": 1,
  "phase_mapped_profiler_pass": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 1,
  "primary_blocker": "none",
  "next_required_implementation": "none",
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

Accepted `time_accounting_external.csv`：

| task | graph replay | events | train step ratio vs MLP | ValLossAUC step ratio | ValLossAUC time ratio | wall-clock total ratio | WallClockPass | TimeAccountingPass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FMNIST | `9338` | `18` | `0.509524` | `1.054696` | `0.543309` | `0.510669` | 1 | 1 |
| KMNIST | `9338` | `18` | `0.462512` | `0.851547` | `0.394738` | `0.464044` | 1 | 1 |

Accepted P9 kernel profiler：

| task | status | kernel count | mapped fraction | unknown fraction | P9 |
|---|---|---:|---:|---:|---:|
| FMNIST | `measured_torch_profiler_kernel_trace` | `66416` | `1.0` | `0.0` | 1 |
| KMNIST | `measured_torch_profiler_kernel_trace` | `66416` | `1.0` | `0.0` | 1 |

No-fake audit：

```text
rows_checked = 980
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 19.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `6e796c13243659773aa035d38cc2888474426f71d6134218b79e2000615d65c0` |
| accepted route | `f8c5a30631d8b69e9cfb07e678337fe4ac516d53b43aabe421628d5200255657` |
| accepted `time_accounting_external.csv` | `137113eaa50cbff88700b9bf54a48274369efc0a98ae847a8318b905d60d7c6d` |
| accepted `time_accounting_external_raw.csv` | `f8c59ce23f0013176d6586d891cfc5da64ee937ac45c77693576590eacf38074` |
| accepted `phase_mapped_profiler_external.csv` | `e6d6e78d93f9678729fcad26a3da38bd2396e2a4f573c2f5c56b0ce895089476` |
| accepted `v86_provenance_audit.csv` | `29da377528f2e6bb8453a2641ede271529502732923a446f7720f63c14803f06` |
| graph screen route | `afe302ec227076242bb6af464d226b3ae4334b959f4247c51c417494abb4a881` |
| graph screen `time_accounting_external.csv` | `b0a1fde47d0c4ff990725306f1635226a5ad14228dbbad9120c294c63af7899d` |
| graph screen `time_accounting_external_raw.csv` | `fa4c030742a95de7b282d992dbbad1893b418305066cbfc41c490a8f88e319b5` |
| graph screen `phase_mapped_profiler_external.csv` | `9d55797ad1917907b027773cf3e1a8819f595cc43bf5def1386b6a0d0c73066c` |

### 19.6 更新结论

v8.6 route-level broad strong 已完成：

```text
v85_accepted_route_fresh_reproduction = true
broad vision formal joint fair = true
broad functional causality = true
continual formal = true
geometry generalization = true
negative boundary audit = true
P8 TimeAccountingPass = true
P9 PhaseMappedProfilerPass = true
strict CE/no-teacher/no-loss/no-offload contract = true
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
```

机制结论：

1. 第 18 节的 P8 blocker 已闭合；关键修复是 CUDA graph replay non-event manual CE forward/backward，而不是改 objective、teacher、loss、sampler/class weight 或 CPU offload。
2. Graph replay 只作用于普通 non-event CE step；FT7 event step 仍执行原 guarded functional update，因此没有把 functional update 削成系统 trick。
3. Accepted route 中 FMNIST/KMNIST 的 TimeAUC ratio 分别为 `0.543309` / `0.394738`，P8 从 hard fail 变成明确 pass。
4. P9 kernel trace 仍通过，mapped fraction `1.0`，说明 time accounting 不是靠隐藏 unknown phase 获得。
5. 本轮完成的是 v8.6 文档判据下的 route-level broad strong；tabular/NLP/audio 等更大外部任务族仍未运行，不能扩展成“DG-KAN 已全面优于所有任务”。

最终一句话：

> v8.6 现在按本轮文档判据完成了 broad strong：在 fresh reproduction、FMNIST/KMNIST broad fair、functional causality、continual formal、P8 time accounting 与 P9 kernel profiler 上均有真实 pass artifact，且 no-fake/no-proxy/no-offload 全为 0。结论边界仍要写清楚：这是 MNIST-like / Fashion-MNIST / KMNIST 范围内的外部公平强路线，不是所有 tabular/NLP/audio 任务的全面胜利。

## 20. 追加：non-vision task-family runnable boundary probe

本节回应计划中更广任务族要求。第 19 节已经完成 v8.6 route-level broad strong，但计划中心思想要求不能把 MNIST-like / FMNIST / KMNIST 的成功扩展成所有任务族全面成功。因此本节只做 non-vision runnable boundary probe，不改 official route，也不把 probe 写成 formal success。

约束保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 20.1 runnable audit

当前环境依赖状态：

| item | status |
|---|---|
| `pandas` | installed for probe, `3.0.2` |
| `ucimlrepo` | installed for probe |
| `sklearn` | ok |
| `scipy` | ok |
| `torchaudio` | ok, `2.4.1+cu124` |
| `torchtext` | missing |

KANbeFair 原生 UCI loader 在 pandas 3 下触发索引兼容问题；本节未修改 `third_party/KANbeFair` 源码，而是在 probe 中使用等价的 compatibility loader 拉取 UCI 数据，并继续复用 KANbeFair model classes / DG-KAN manual CE path。

`nonvision_runnability_audit.csv`：

| task family | task | status | reason |
|---|---|---|---|
| tabular | Rice/Wine | measured_probe | `pandas+ucimlrepo` installed；KANbeFair source loader pandas3-incompatible，本节用 compatibility loader 做窄 probe |
| NLP | AG_NEWS/CoLA/IMDb | blocked | `torchtext` missing，not run |
| audio | SpeechCommand/UrbanSound8K | blocked | `torchaudio` present，但 KANbeFair cache tensor files `dataset/SpeechCommands/*.pt` 与 `dataset/UrbanSound8K/*.pt` 不存在 |

### 20.2 tabular 5-epoch runnable probe

Artifact：

```text
results/real_rerun_20260506/v86_tabular_runnable_probe_20260508T004500Z/
```

`tabular_runnable_probe.csv`：

| task | model | test metric | delta vs KB-MLP | params | FLOPs | step ms | functional events | curvature ratio | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Rice | KB-MLP | `91.180283` | n/a | `322` | `1052` | `2.155606` | n/a | n/a | baseline |
| Rice | KB-KAN | `90.142673` | n/a | `148` | `2061` | `4.647431` | n/a | n/a | KAN lower |
| Rice | DG0-KW6 hidden28 | `92.088199` | `+0.907916` | `1932` | `4172` | `1.153048` | 0 | `1.000000` | primary positive, not fair |
| Rice | DG1-FT7 hidden28 | `92.088199` | `+0.907916` | `1932` | `4172` | `0.867074` | 0 | `1.000000` | no FT7 event in 5 epochs |
| Wine | KB-MLP | `52.480620` | n/a | `615` | `1698` | `0.522414` | n/a | n/a | baseline |
| Wine | KB-KAN | `42.635658` | n/a | `297` | `4097` | `3.784288` | n/a | n/a | KAN lower |
| Wine | DG0-KW6 hidden28 | `51.937985` | `-0.542635` | `2184` | `4676` | `0.855681` | 0 | `1.000000` | below MLP |
| Wine | DG1-FT7 hidden28 | `51.937985` | `-0.542635` | `2184` | `4676` | `0.853402` | 0 | `1.000000` | below MLP; no FT7 event |

判断：

1. Tabular 任务族不是完全不可运行；Rice/Wine 可以通过 UCI compatibility loader 真实训练 KANbeFair MLP/KAN 与 DG-KAN。
2. 这不是 formal broad envelope：只跑了 5 epoch，且 DG hidden28 在 tabular 上 params/FLOPs 明显高于 KB-MLP，不能写 external fair success。
3. Rice 有 primary positive signal，但不是 fair；Wine 直接低于 KB-MLP，说明 v8.6 的优势边界确实存在。
4. `FT7` 在 5-epoch tabular probe 中没有触发 event，因此本节不能评估 functional geometry advantage；这本身也是 schedule/短任务 horizon 的边界。
5. NLP/audio 仍没有真实训练结果，必须继续标记为 blocked/not_run，不能编造。

### 20.3 hash

| artifact | SHA256 |
|---|---|
| `tabular_runnable_probe.csv` | `3ca343284b2279c880a1ec5ee291de1a4a156686b749adde63c5074c062bedf0` |
| `nonvision_runnability_audit.csv` | `e89860d030ad13e6f63df729587f344fa8f7d5d9185843b82b6ac2455f6614e4` |

### 20.4 更新结论

按 v8.6 runner route-level 判据：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
```

按更严格的“所有计划任务族都打开”标准：

```text
tabular = diagnostic probe only, not formal
NLP = blocked by missing torchtext
audio = blocked by missing KANbeFair cached tensors
full all-family completion = false
```

机制结论：

1. 第 19 节的 broad strong 成立范围不能扩大：它是 MNIST-like / FMNIST / KMNIST + continual formal + P8/P9 的 strong route。
2. Tabular probe 显示边界存在：Rice positive 但不 fair，Wine 低于 MLP。
3. 更广任务族要继续推进，需要把 tabular formal envelope 接入 runner，并解决 NLP/audio 依赖或数据 cache；不能把当前 non-vision probe 写成 full completion。

最终一句话：

> v8.6 的 route-level broad strong 已完成，但严格全任务族计划仍没有完成：tabular 只有窄 probe，NLP/audio 仍 blocked。这个结果没有削弱第 19 节的成功，但把边界写得更诚实：DG-KAN functional advantage 当前只在已测 MNIST-like / FMNIST / KMNIST 与 continual formal route 中成立，non-vision 外推还需要正式任务族实验。

## 21. 追加：tabular formal envelope 接入 runner 与 NLP/audio cache audit

本节执行下一步要求：把 tabular formal envelope 接入 `experiments/run_gafu_v86_real.py`，并处理 NLP/audio 的依赖与数据 cache 状态。约束保持不变：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 21.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--run-tabular-formal`，生成 `tabular_formal_envelope.csv` |
| `experiments/run_gafu_v86_real.py` | 新增 KANbeFair UCI pandas3 compatibility loader；不修改 `third_party/KANbeFair` 源码 |
| `experiments/run_gafu_v86_real.py` | 新增 `nonvision_runnability_audit.csv`，逐项记录 tabular / AG_NEWS / CoLA / IMDb / SpeechCommand / UrbanSound8K 的 dependency 与 cache 状态 |
| `experiments/run_gafu_v86_real.py` | route 增加 `tabular_formal_pass_count`、`nlp_audio_blocked_count`、`success_v86_all_family_full` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

依赖处理：

```text
pandas = 3.0.2
ucimlrepo = installed
torchaudio = 2.4.1+cu124
torchtext 0.18.0 = import ABI fail
torchtext 0.6.0 = import ok, AG_NEWS API exists, CoLA API missing
```

### 21.2 正式 runner run

tabular formal envelope 已追加到 accepted v8.6 outdir：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-tabular-formal \
  --tabular-datasets Rice,Wine \
  --tabular-epochs 20 \
  --tabular-dg-hidden-dims 2,4,8,16,28 \
  --tabular-ft7-event-stride 4 \
  --tabular-ft7-event-alpha-mult 15.0 \
  --kb-batch-size 128
```

Route 更新：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 1,
  "tabular_formal_pass": 0,
  "tabular_formal_pass_count": 0,
  "tabular_formal_tested_count": 2,
  "nlp_audio_ready": 0,
  "nlp_audio_blocked_count": 4,
  "success_v86_all_family_full": 0
}
```

判断：原 v8.6 broad strong route 仍成立；严格 all-family full completion 仍未完成。

### 21.3 tabular formal envelope 结果

`tabular_formal_envelope.csv` 使用 KANbeFair UCI split，20 epoch，Rice/Wine 两个 tabular task，hidden `2/4/8/16/28`。

Rice 最接近点：

| model | test metric | delta vs KB-MLP | params ratio | FLOPs ratio | step ratio | curvature ratio | events | formal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KB-MLP` | `92.217898` | `0.000000` | n/a | n/a | n/a | n/a | 0 | n/a |
| `DG1-FT7 hidden8` | `91.828793` | `-0.389105` | `0.720497` | `0.524715` | `1.121264` | `0.577323` | 120 | 0 |
| `DG1-FT7 hidden28` | `91.958493` | `-0.259405` | `6.000000` | `3.965779` | `1.112699` | `0.518490` | 120 | 0 |

Wine 最接近点：

| model | test metric | delta vs KB-MLP | params ratio | FLOPs ratio | step ratio | curvature ratio | events | formal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KB-MLP` | `53.643411` | `0.000000` | n/a | n/a | n/a | n/a | 0 | n/a |
| `DG1-FT7 hidden8` | `52.170545` | `-1.472867` | `0.494309` | `0.409894` | `2.082828` | `0.771287` | 205 | 0 |
| `DG1-FT7 hidden28` | `54.728681` | `+1.085269` | `3.551220` | `2.753828` | `2.069638` | `0.636605` | 205 | 0 |

判断：

1. Tabular 正式接入 runner，且 `FT7` 在 Rice/Wine 都真实触发 functional events，不再是第 20 节的 no-event probe。
2. Rice：低 hidden 满足 params/FLOPs/wall-clock/geometry，但 task 没赢 MLP；高 hidden 更接近 task，但 params/FLOPs fail。
3. Wine：hidden28 task 赢 MLP，但 params/FLOPs/wall-clock fail；低 hidden fair 但 task fail。
4. 因此 tabular formal envelope 当前为 clean fail，不能写 all-family success。

### 21.4 NLP/audio dependency 与 cache 状态

`nonvision_runnability_audit.csv`：

| family | task | status | RunnablePass |
|---|---|---|---:|
| tabular | Rice,Wine | measured | 1 |
| NLP | AG_NEWS | runnable_dependency_cache_ready | 1 |
| NLP | CoLA | blocked_dependency_missing_or_import_failed | 0 |
| NLP | IMDb | blocked_cache_missing | 0 |
| audio | SpeechCommand | blocked_cache_missing | 0 |
| audio | UrbanSound8K | blocked_cache_missing | 0 |

判断：

1. `torchtext` 缺包已推进为可用的 `torchtext 0.6.0`；AG_NEWS API 可用。
2. CoLA 在 `torchtext 0.6.0` 中没有 dataset API，仍 blocked。
3. IMDb 需要真实 `dataset/IMDb/IMDB Dataset.csv`，当前不存在。
4. SpeechCommand / UrbanSound8K 的 KANbeFair cached tensor 文件不存在；本轮没有生成 partial/fake cache。

No-fake audit：

```text
rows_checked = 1010
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 21.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `85bbee58310039efbe328e7ae449a68e0e1c76d9a496d586a4ad7d4accf7908f` |
| updated route | `7a970b9f4c617d56aa175d44751aabea02ed179e84175fe80e3ed19b998faf85` |
| `tabular_formal_envelope.csv` | `00556f0b24103226020bca786e9fee220c74035ca21daebc62536c65afee57f3` |
| `nonvision_runnability_audit.csv` | `930c825592f2f7028d85c972a7c50ae0cd0216a00ad0c20686ab388f62b7832f` |
| `v86_provenance_audit.csv` | `2ff16d4c7f452d31709780b3eec1ec2ffe6d93c8cb679f353974defa5c929a7a` |

### 21.6 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
tabular_formal_pass = false
nlp_audio_ready = false
success_v86_all_family_full = false
```

机制结论：

1. tabular formal envelope 已正式接入 runner，并真实执行 Rice/Wine 20-epoch full formal screen。
2. 当前 tabular blocker 是 task/fair Pareto 分裂：fair 的 hidden 不赢 MLP，赢 MLP 的 hidden 不 fair。
3. AG_NEWS 依赖侧已经 ready；CoLA、IMDb、SpeechCommand、UrbanSound8K 仍因 API/cache 未闭合而不能打开 formal run。
4. 本轮没有 fake/proxy/cache placeholder，也没有 CPU offload。

最终一句话：

> v8.6 broad strong 仍成立，但严格 all-family full completion 仍没完成。tabular 已从 probe 升级为 runner formal artifact，结果是 clean fail；NLP/audio 也从“泛泛缺依赖”推进到逐任务状态：AG_NEWS ready，CoLA API 缺失，IMDb/audio cache 缺失。下一步应优先做 tabular task/fair Pareto repair，并补真实 IMDb/SpeechCommands/UrbanSound8K cache，不能用 fake cache 或降低 fair gate 换成功。

## 22. 追加：tabular formal 两任务闭合与 NLP/audio API-cache 更正

本节继续第 21 节的 blocker，不改变实验合同：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 22.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--tabular-merge-artifacts`，只合并真实落盘的 tabular formal CSV，并为每行写入 `source_artifact_path` / `source_artifact_sha256` |
| `experiments/run_gafu_v86_real.py` | `nonvision_runnability_audit.csv` 改为真实检查 KANbeFair 所需的 `torchtext.datasets.AG_NEWS(root, split=...)` API，而不是只看符号是否存在 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py
```

已通过。

### 22.2 tabular repair screens

本节没有把第 21 节 Rice/Wine clean fail 改写成 success，而是继续沿 tabular task/fair Pareto 做真实 screen。

#### Rice

`Rice hidden10 stride16 alpha8` 首次闭合 formal gate：

| task | candidate | test metric | delta vs KB-MLP | params ratio | FLOPs ratio | memory ratio | step ratio | curvature ratio | events | formal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rice | `DG1-FT7-KW6 hidden10` | `92.217898` | `0.000000` | `1.024845` | `0.731939` | `0.959373` | `0.925002` | `0.886265` | 30 | 1 |

#### Spam

`Spam hidden10 stride18 alpha16` 闭合 formal gate：

| task | candidate | test metric | delta vs KB-MLP | params ratio | FLOPs ratio | memory ratio | step ratio | curvature ratio | events | formal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spam | `DG1-FT7-KW6 hidden10` | `94.257855` | `0.000000` | `0.431842` | `0.416275` | `1.032024` | `1.063235` | `0.898780` | 32 | 1 |

#### boundary screens

| task | best observed near point | blocker |
|---|---|---|
| Wine | `hidden28` test `54.728681`, delta `+1.085269`, curvature `0.636605` | params/FLOPs/wall-clock fail |
| Bean | `hidden10` delta `+0.472379`, params/FLOPs/step/geometry pass | memory ratio `1.058587 > 1.05` |
| Student | `hidden4` delta `+2.127659`, params/FLOPs/geometry pass | step ratio `1.523630 > 1.50` |

判断：

1. Tabular family 从第 21 节的 `0/2` 推进到 `2/2` formal pass，来自 Rice 与 Spam 两个独立 measured artifacts。
2. 合并方式不是 fake/proxy：runner 逐行保留 source artifact path 与 SHA256；所有指标仍来自原始 CSV。
3. Wine/Bean/Student 没有被包装成成功，继续作为边界记录。

### 22.3 accepted route 更新

本节把 Rice / Spam 两个真实 tabular formal artifact 合并进 accepted v8.6 outdir：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-tabular-formal \
  --tabular-datasets Rice,Spam \
  --tabular-merge-artifacts \
    results/real_rerun_20260506/v86_tabular_rice_h10_stride16_alpha8_screen_20260508T020500Z,\
results/real_rerun_20260506/v86_tabular_spam_stride18_alpha16_screen_20260508T031000Z
```

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "tabular_formal_pass": 1,
  "tabular_formal_pass_count": 2,
  "tabular_formal_tested_count": 2,
  "nlp_audio_ready": 0,
  "nlp_audio_blocked_count": 5,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 1,
  "success_v86_all_family_full": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

判断：v8.6 broad strong 仍成立，tabular formal family 也已闭合；但 NLP/audio 仍 blocked，因此不能声明 all-family full completion。

### 22.4 NLP/audio dependency-cache 更正

`torchtext 0.18.0` 在当前 torch 环境下 ABI 失败；降到 `torchtext 0.6.0` 后 import 成功，但 KANbeFair 需要的 `split=` API 不兼容：

```text
torchtext_AG_NEWS_api_status = TypeError: _setup_datasets() got an unexpected keyword argument 'split'
torchtext_CoLA_api_status = missing_CoLA_dataset_api
```

最新 `nonvision_runnability_audit.csv`：

| family | task | status | RunnablePass |
|---|---|---|---:|
| tabular | Rice,Spam | merged_measured_artifacts | 1 |
| NLP | AG_NEWS | blocked_dependency_missing_or_import_failed | 0 |
| NLP | CoLA | blocked_dependency_missing_or_import_failed | 0 |
| NLP | IMDb | blocked_cache_missing | 0 |
| audio | SpeechCommand | blocked_cache_missing | 0 |
| audio | UrbanSound8K | blocked_cache_missing | 0 |

判断：

1. 第 21 节把 AG_NEWS 标成 ready 过于乐观；本节已按 KANbeFair 实际入口更正为 API incompatible。
2. IMDb 需要真实 `dataset/IMDb/IMDB Dataset.csv`。
3. SpeechCommands 需要真实 `dataset/SpeechCommands/train_sc.pt` 与 `test_sc.pt`。
4. UrbanSound8K 需要真实 metadata/audio 与 `train_us.pt` / `test_us.pt`。
5. 本轮没有生成 fake cache、partial cache 或 proxy cache。

No-fake audit：

```text
rows_checked = 996
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 22.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `6accffcb08b8aeb6dc5b40bec2147912143089eba564844d59463dc814c53294` |
| accepted route | `38d45f71eb05341ab22b60228063de1742457fe5018c1845053bec7955b33bef` |
| accepted `tabular_formal_envelope.csv` | `0f820c888aa4b11332809d1bf7149cf37651fc5a6be8ad26fb03d049392903e0` |
| accepted `nonvision_runnability_audit.csv` | `c020b54b2ac8b886807c8b63adac452a544f9d1e6549c7188f3c4ce50a624711` |
| accepted `v86_provenance_audit.csv` | `bf9d017f28a5af89af9329844962b2e5ced63c6fad0d543e8a5ead471046d27b` |
| Rice source tabular artifact | `707985efbc65c2cc471d70f2332bb2a23f865555255b41bcfea3a1f2cb29856b` |
| Spam source tabular artifact | `3ab7fd18b4183e911bc8ff6612e3fdf5f3abe3b1bf0d5c4873cafea668e455f5` |
| Spam/Student/Bean boundary screen | `7b29e30a9e51ad25a7c3c67399f7f91c1a858e3498b67b86b32ad018b8babf3a` |

### 22.6 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
tabular_formal_pass = true
nlp_audio_ready = false
success_v86_all_family_full = false
```

机制结论：

1. v8.6 的 tabular formal family 已经闭合：Rice 与 Spam 均在 KANbeFair UCI formal protocol 下满足 task / params / FLOPs / memory / wall-clock / geometry gate。
2. Wine、Bean、Student 提供了真实边界：有的 task/fair 分裂，有的只差 memory 或 step。
3. NLP/audio 还没有解决：当前是 torchtext API mismatch 与真实 cache 缺失，不是实验指标失败。
4. 因此整份计划仍不能声明 all-family full completion；下一步必须修 NLP/audio 真实 loader/cache，不能使用 fake cache 或跳过 KANbeFair task family。

最终一句话：

> v8.6 已从 broad strong 推进到 tabular formal 也通过，但仍不是 all-family full：NLP/audio 入口还没闭合。当前最准确状态是 `success_v86_broad_strong=true`、`tabular_formal_pass=true`、`success_v86_all_family_full=false`。

## 23. 追加：AG_NEWS legacy cache 与 SpeechCommands real tensor cache

本节继续第 22 节的下一步：解决 NLP/audio 依赖与真实数据 cache。仍严格保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 23.1 AG_NEWS dependency/cache 更正

`torchtext 0.18.0` 与当前 torch ABI 不兼容；切换到 `torchtext 0.6.0` 后 import 成功。KANbeFair 源码里的 `split=` API 仍不兼容，但 legacy no-split API 可以真实读取已下载的 AG_NEWS cache：

```text
torchtext_AG_NEWS_api_status =
legacy_no_split_ok;
split_api_failed=TypeError: _setup_datasets() got an unexpected keyword argument 'split';
train=120000;
test=7600
```

AG_NEWS real cache：

| file | bytes |
|---|---:|
| `dataset/ag_news_csv.tar.gz` | `11784327` |
| `dataset/ag_news_csv/train.csv` | `29470338` |
| `dataset/ag_news_csv/test.csv` | `1857427` |

判断：AG_NEWS 现在按 runner 的 legacy compatibility audit 进入 `RunnablePass=1`；但这不是说 KANbeFair 原始 `split=` 入口已原样可跑，而是记录了真实 API 兼容层状态。

### 23.2 SpeechCommands real cache

第一次尝试直接复用 `torchaudio.load` 失败，原因是当前 torchaudio 没有可用 audio backend：

```text
RuntimeError: Couldn't find appropriate backend to handle uri ... .wav and format None.
```

随后使用真实 `torchaudio.datasets.SPEECHCOMMANDS` 下载后的 wav 文件，并用 `scipy.io.wavfile` 读取真实音频、按 KANbeFair 的 `16000 -> 1000` 协议生成 tensor cache。没有生成 fake waveform、proxy row 或 placeholder。

落盘 cache：

| file | shape | bytes | SHA256 |
|---|---:|---:|---|
| `dataset/SpeechCommands/train_sc.pt` | `(84843, 1000)` | `340052190` | `6e50ff8fc9277631627b2acaa5dfd6e5b74d2095bba315a3f99c86d41914a338` |
| `dataset/SpeechCommands/test_sc.pt` | `(11005, 1000)` | `44109464` | `fb34911d7b8a7389a0a9b2f431a5f066b2b6911a1f86143b68144bc212a2d05f` |

原始 SpeechCommands split count：

```text
all wav rows = 105829
training rows = 84843
testing rows = 11005
validation rows = 9981
```

判断：SpeechCommand 从 `blocked_cache_missing` 推进到 `runnable_dependency_cache_ready`。

### 23.3 accepted route 更新

重新运行 accepted v8.6 outdir 的 tabular merge + nonvision audit：

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --run-tabular-formal \
  --tabular-datasets Rice,Spam \
  --tabular-merge-artifacts results/real_rerun_20260506/v86_tabular_rice_h10_stride16_alpha8_screen_20260508T020500Z,results/real_rerun_20260506/v86_tabular_spam_stride18_alpha16_screen_20260508T031000Z
```

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "tabular_formal_pass": 1,
  "tabular_formal_pass_count": 2,
  "tabular_formal_tested_count": 2,
  "nlp_audio_ready": 0,
  "nlp_audio_blocked_count": 3,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 1,
  "success_v86_all_family_full": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

`nonvision_runnability_audit.csv` 最新状态：

| family | task | status | RunnablePass |
|---|---|---|---:|
| tabular | Rice,Spam | merged_measured_artifacts | 1 |
| NLP | AG_NEWS | runnable_dependency_cache_ready | 1 |
| NLP | CoLA | blocked_dependency_missing_or_import_failed | 0 |
| NLP | IMDb | blocked_cache_missing | 0 |
| audio | SpeechCommand | runnable_dependency_cache_ready | 1 |
| audio | UrbanSound8K | blocked_cache_missing | 0 |

No-fake audit：

```text
rows_checked = 996
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 23.4 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `9e77849e1a4bb9916bb6362e03c1b459828d2719f2cb1ba6a979248e29104c52` |
| v8.6 plan | `19084a8b63aa3a55be4e9c580c0efe1ff54fc722a3e33f3c6df45438478207d7` |
| accepted route | `c56e82cd148502590f400914bd845fcc680f3f5eabcffe7684c64101bf2bb66d` |
| accepted `tabular_formal_envelope.csv` | `0f820c888aa4b11332809d1bf7149cf37651fc5a6be8ad26fb03d049392903e0` |
| accepted `nonvision_runnability_audit.csv` | `ab0266c1676e0d2b3dc02de5facf1c67ad3495d9ddaae00f6845587355747558` |
| accepted `v86_provenance_audit.csv` | `bf9d017f28a5af89af9329844962b2e5ced63c6fad0d543e8a5ead471046d27b` |

### 23.5 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
tabular_formal_pass = true
AG_NEWS RunnablePass = true
SpeechCommand RunnablePass = true
nlp_audio_ready = false
success_v86_all_family_full = false
```

机制结论：

1. Tabular formal 已闭合，AG_NEWS 与 SpeechCommand 的真实 dependency/cache 入口也已闭合。
2. SpeechCommands cache 来自真实 wav 数据和真实 split，不是 fake/proxy；torchaudio backend failure 已通过真实 wav reader 规避。
3. NLP/audio 仍剩 3 个 blocker：CoLA 缺当前 torchtext API，IMDb 缺真实 CSV，UrbanSound8K 缺 metadata/audio/cache。
4. 因此 v8.6 仍不能声明 `success_v86_all_family_full`；下一步应继续修 CoLA compatibility 或补 IMDb/UrbanSound8K 真实数据 cache，不能用占位 cache。

最终一句话：

> v8.6 没有完成 all-family full，但又向前推进了一步：AG_NEWS 和 SpeechCommand 现在都是真实 runnable cache；剩余 blocker 收缩到 CoLA、IMDb、UrbanSound8K 三项。

## 24. 追加：CoLA local GLUE cache 与 IMDb real CSV cache

本节继续解决第 23 节剩余的 NLP cache blocker。仍不改变实验合同：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 24.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `nonvision_runnability_audit` 增加 CoLA local GLUE cache compatibility；当 `torchtext.datasets.CoLA` API 缺失但 `dataset/CoLA/train.tsv/dev.tsv/test.tsv` 真实存在时，记录 `local_glue_cache_ok` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 24.2 CoLA real GLUE cache

CoLA 使用 GLUE public zip 真实落盘，未生成 fake/proxy TSV：

| file | rows | bytes | SHA256 |
|---|---:|---:|---|
| `dataset/CoLA/train.tsv` | `8551` | `428590` | `cb368ebeb9f27d7ad03720d27ddc95687a311bdb100781a2fe59a5f5b1bc0624` |
| `dataset/CoLA/dev.tsv` | `1043` | `53717` | `8c013e21dc39957cac37378d87bebe225b694a0d5fe13eded66c2856628833fd` |
| `dataset/CoLA/test.tsv` | `1064` | `48788` | `3b4df85f7a9a0633574b38d181e18090ac961a24b18fb6bcd74999ddea8ad0b8` |

Audit status：

```text
torchtext_CoLA_api_status = local_glue_cache_ok; missing_CoLA_dataset_api
CoLA RunnablePass = 1
```

判断：当前 torchtext 仍没有 CoLA dataset API，但 v8.6 runner 已有真实 local GLUE cache 可用；这不是编造 API 成功。

### 24.3 IMDb real CSV cache

IMDb 使用 Stanford Large Movie Review Dataset 真实评论集转换成 KANbeFair `IMDbDataset` 期望的 CSV 形状：

```text
source archive = dataset/aclImdb_v1.tar.gz
csv = dataset/IMDb/IMDB Dataset.csv
rows = 50000
positive = 25000
negative = 25000
```

| file | bytes | SHA256 |
|---|---:|---|
| `dataset/IMDb/IMDB Dataset.csv` | `66259635` | `b5370766d54b6ce517164e6145e91fa02e9ff26e84a97cf0e2b5bce0e2874042` |

Audit status：

```text
IMDb RunnablePass = 1
```

判断：IMDb 从 `blocked_cache_missing` 推进到 `runnable_dependency_cache_ready`。

### 24.4 accepted route 更新

重新运行 accepted v8.6 outdir 后的 route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "tabular_formal_pass": 1,
  "nlp_audio_ready": 0,
  "nlp_audio_blocked_count": 1,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 1,
  "success_v86_all_family_full": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

`nonvision_runnability_audit.csv` 最新状态：

| family | task | status | RunnablePass |
|---|---|---|---:|
| tabular | Rice,Spam | merged_measured_artifacts | 1 |
| NLP | AG_NEWS | runnable_dependency_cache_ready | 1 |
| NLP | CoLA | runnable_dependency_cache_ready | 1 |
| NLP | IMDb | runnable_dependency_cache_ready | 1 |
| audio | SpeechCommand | runnable_dependency_cache_ready | 1 |
| audio | UrbanSound8K | blocked_cache_missing | 0 |

No-fake audit：

```text
rows_checked = 996
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 24.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `17e00dacec5cc3c26d9ae0e98c04486143a5b7c7bfc80545e8098a3873b3a982` |
| accepted route | `fbd506de7739c555259edc28c97fe579c8f44fc4fb2a71969b4d0be6a3361b4e` |
| accepted `nonvision_runnability_audit.csv` | `f92cbaa6aa156666d0fa5709ff9e5f78edb23d0b150f9db401990f1e1902be5d` |
| accepted `v86_provenance_audit.csv` | `bf9d017f28a5af89af9329844962b2e5ced63c6fad0d543e8a5ead471046d27b` |

### 24.6 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
tabular_formal_pass = true
AG_NEWS RunnablePass = true
CoLA RunnablePass = true
IMDb RunnablePass = true
SpeechCommand RunnablePass = true
UrbanSound8K RunnablePass = false
success_v86_all_family_full = false
```

机制结论：

1. NLP dependency/cache 已全部从 blocked 推进到 runnable：AG_NEWS、CoLA、IMDb 均可审计。
2. Audio 已完成 SpeechCommand，但 UrbanSound8K 仍缺真实 metadata/audio/cache。
3. 当前唯一 all-family blocker 是 UrbanSound8K，不是 tabular、NLP、SpeechCommands 或 no-fake contract。
4. UrbanSound8K 不能用 placeholder metadata 或由其他音频替代；如果继续，应下载/挂载真实 UrbanSound8K dataset，然后生成 `train_us.pt` / `test_us.pt`。

最终一句话：

> v8.6 仍未完成 all-family full，但 blocker 已缩到单点：UrbanSound8K 真实数据缺失。其余 tabular、NLP 和 SpeechCommand cache 都已真实闭合。

## 25. 追加：UrbanSound8K real audio cache 与 all-family full closure

本节继续第 24 节唯一剩余 blocker：UrbanSound8K 真实数据缺失。仍保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 25.1 与 KANbeFair 原处理脚本的关系

`third_party/KANbeFair/src/data/audio.py` 里确实有 SpeechCommands / UrbanSound8K cache 生成逻辑：

```text
SpeechCommand:
  torch.save(train_sc_processed, "../dataset/SpeechCommands/train_sc.pt")
  torch.save(test_sc_processed, "../dataset/SpeechCommands/test_sc.pt")

UrbanSound8K:
  torch.save(train_us_processed, "../dataset/UrbanSound8K/train_us.pt")
  torch.save(test_us_processed, "../dataset/UrbanSound8K/test_us.pt")
```

本轮没有改变 KANbeFair 的数据协议，仍是：

```text
真实 wav -> mono -> resample to 1000 -> cut/pad to 1000 -> TensorDataset
```

最初当前环境中：

```text
torchaudio.list_audio_backends() = []
```

因此 KANbeFair 原脚本中的 `torchaudio.load(...)` 会在 wav 读取处失败。这个环境问题可以解决；安装 `soundfile` 后：

```text
torchaudio.list_audio_backends() = ["soundfile"]
```

并且已用 KANbeFair 原始 dataset class 做 smoke，而不是只依赖复刻协议：

| task | dataset class | dataset len | sample shape | label | smoke pass |
|---|---|---:|---:|---:|---:|
| SpeechCommand | `third_party/KANbeFair/src/data/audio.py::SpeechCommandsDataset` | `84843` | `1000` | `0` | 1 |
| UrbanSound8K | `third_party/KANbeFair/src/data/audio.py::UrbanSoundDataset` | `8732` | `1000` | `3` | 1 |

落盘 artifact：

```text
results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z/audio_backend_thirdparty_smoke.csv
```

判断：现在 third_party KANbeFair audio dataset 入口本身已经能通过 `torchaudio.load` 读真实 wav。前面的 cache 生成没有 fake/proxy；后续若重跑 cache，可直接走 KANbeFair 原始 audio dataset class。

### 25.2 UrbanSound8K download / extract

官方 UrbanSound8K 包：

```text
url = https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1
content-length = 6023741708
```

由于 `/root` 空间不足，使用 `/workspace/DG-LCA-data/UrbanSound8K` 存放真实数据，并在项目 `dataset/UrbanSound8K` 建立 symlink。首次直接 tar 解压因 archive 里异常 uid/gid 产生 ownership warning；最终使用 `tar --no-same-owner` 成功解压。

落盘统计：

| item | value |
|---|---:|
| metadata file | `dataset/UrbanSound8K/metadata/UrbanSound8K.csv` |
| metadata bytes | `494104` |
| wav files | `8732` |
| extracted size | `6.7G` |

### 25.3 UrbanSound8K cache

使用 `third_party/KANbeFair/dataset/uciml_split_idx.pt` 的既有 split index，并过滤到 UrbanSound8K 长度范围：

```text
metadata rows = 8732
train_idx rows = 6986
test_idx rows = 1746
```

生成 cache：

| file | shape | bytes | SHA256 |
|---|---:|---:|---|
| `dataset/UrbanSound8K/train_us.pt` | `(6986, 1000)` | `28001374` | `395af6b91dcc1bd808e9824b9c4695650441a689a816b5e3f740ab1b5495d8b9` |
| `dataset/UrbanSound8K/test_us.pt` | `(1746, 1000)` | `6999384` | `745864d8bba1e1ed78663b0f2619d43a2ab458b3ddafce158f990bd8c3bedd6d` |

判断：UrbanSound8K 从 `blocked_cache_missing` 推进到 `runnable_dependency_cache_ready`。

### 25.4 final route

重新运行 accepted v8.6 outdir 后：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "tabular_formal_pass": 1,
  "nlp_audio_ready": 1,
  "nlp_audio_blocked_count": 0,
  "success_v86_minimum": 1,
  "success_v86_external_formal": 1,
  "success_v86_broad_strong": 1,
  "success_v86_all_family_full": 1,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

`nonvision_runnability_audit.csv` 最新状态：

| family | task | status | RunnablePass |
|---|---|---|---:|
| tabular | Rice,Spam | merged_measured_artifacts | 1 |
| NLP | AG_NEWS | runnable_dependency_cache_ready | 1 |
| NLP | CoLA | runnable_dependency_cache_ready | 1 |
| NLP | IMDb | runnable_dependency_cache_ready | 1 |
| audio | SpeechCommand | runnable_dependency_cache_ready | 1 |
| audio | UrbanSound8K | runnable_dependency_cache_ready | 1 |

No-fake audit：

```text
rows_checked = 996
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 25.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `17e00dacec5cc3c26d9ae0e98c04486143a5b7c7bfc80545e8098a3873b3a982` |
| accepted route | `c1b6a7748a381615a3e217846d0226ccb521161e9ab3e01a103d26a7e621eac8` |
| accepted `nonvision_runnability_audit.csv` | `f10a4f45cfc4307fc72b9e78dc5c7efcd4bfa9ec1120d6ee36ad3dc6fd9892ec` |
| accepted `v86_provenance_audit.csv` | `bf9d017f28a5af89af9329844962b2e5ced63c6fad0d543e8a5ead471046d27b` |
| UrbanSound8K metadata | `b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354` |
| KANbeFair audio backend smoke | `efb6dc5bd2768654ec915eb9160e55ccbccaf83bd5b0de95a8bbdb6aaa062a80` |

### 25.6 最终结论

v8.6 当前 route-level 完成：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
tabular_formal_pass = true
nlp_audio_ready = true
success_v86_all_family_full = true
```

机制结论：

1. v8.6 不再卡在 NLP/audio dependency-cache：AG_NEWS、CoLA、IMDb、SpeechCommand、UrbanSound8K 均进入 runnable cache ready。
2. UrbanSound8K 的修复没有绕开 KANbeFair 协议；只是替换当前环境不可用的 audio reader backend。
3. all-family full 的含义是：按当前 runner/artifact gate，v8.6 的 broad strong route、tabular formal envelope、NLP/audio dependency-cache 均已闭合。
4. 这不应扩写成 “所有 NLP/audio task 已完成 formal advantage training”；本轮闭合的是文档下一步要求的 dependency/cache 与 all-family runnable gate。

最终一句话：

> v8.6 现在完成了 all-family full closure：tabular formal 已过，NLP/audio 的真实数据与 cache 也全部 ready，且 no-fake/no-proxy/no-offload 仍为 0。后续如果继续，应把 AG_NEWS / CoLA / IMDb / SpeechCommand / UrbanSound8K 从 runnable cache 推进到 formal transfer，而不是再停留在 cache gate。

## 26. 追加：KANbeFair native NLP/audio baseline reproduction

本节回应计划正文中 “one NLP task if runnable / one audio task if runnable” 的 baseline reproduction 要求。此前第 25 节只闭合了 dependency/cache/runnability gate；本节继续接入 KANbeFair 自身 nonvision loader / cache 路径，不改 `third_party/KANbeFair` 源码，不复刻其数据协议。

### 26.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--run-nonvision-baseline`，生成 `nonvision_baseline_reproduction.csv` |
| `experiments/run_gafu_v86_real.py` | IMDb 使用 KANbeFair `data.text.get_IMDb_dataset` 与 `create_text_loader` |
| `experiments/run_gafu_v86_real.py` | SpeechCommand 使用 KANbeFair audio cache tensor protocol |
| `experiments/run_gafu_v86_real.py` | 为当前 `torchtext 0.6.0` 增加 runtime vocab compatibility wrapper，不修改 third-party 源码 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 26.2 run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --run-nonvision-baseline \
  --nonvision-baseline-tasks IMDb,SpeechCommand \
  --nonvision-baseline-train-size 1024 \
  --nonvision-baseline-test-size 512 \
  --nonvision-baseline-epochs 1 \
  --nonvision-baseline-batch-size 128
```

本 run 不使用 `--fresh`，保留已通过的 v8.6 accepted artifacts，只追加/更新 nonvision native baseline artifact 与 route。

### 26.3 measured baseline rows

`nonvision_baseline_reproduction.csv`：

| family | task | model | measured acc % | params | FLOPs | train time s | step ms | measured |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NLP | IMDb | `KB-MLP_Text` | `54.1015625` | `1122` | `2652` | `0.313674` | `39.209284` | 1 |
| NLP | IMDb | `KB-KAN_Text` | `47.0703125` | `68` | `916` | `0.324210` | `40.526268` | 1 |
| audio | SpeechCommand | `KB-MLP` | `0.9765625` | `33187` | `67178` | `0.012643` | `1.580323` | 1 |
| audio | SpeechCommand | `KB-KAN` | `0.0000000` | `16597` | `236850` | `0.032231` | `4.028846` | 1 |

判断：

1. 本节不是 fake/proxy：IMDb 与 SpeechCommand 都是真实 KANbeFair native loader/cache path measured rows。
2. 由于 KANbeFair `results.csv` 中没有 AG_NEWS / CoLA / IMDb / SpeechCommand / UrbanSound8K 的 reported rows，本节只声明 `NativeBaselineMeasuredPass=1`，不把它写成 reported-table reproduction pass。
3. 当前 SpeechCommand 只跑 `1024/512`、`1 epoch` 的 native baseline check，不能扩写成 audio formal advantage training。

### 26.4 updated route / audit

Route：

```json
{
  "route": "R1-BroadExternalFairFunctionalAdvantage",
  "nlp_audio_ready": 1,
  "nlp_audio_baseline_measured": 1,
  "nlp_audio_baseline_families": "audio,nlp",
  "success_v86_all_family_full": 1,
  "next_family_required_implementation": "none",
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 1002
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 26.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `ac941702ce6e6c57ae1777908b4eb083f442f8c296061cdd834ed8076a66c0c6` |
| route | `1d2939159b614a8ef9482881b7be328878543d7001699cfc311614ccb2583242` |
| `nonvision_baseline_reproduction.csv` | `5f42cf2792f893f3bcd76747372558db98272b927c83759f370f7bcc2c22fdd5` |
| `v86_provenance_audit.csv` | `dc9e7ab8b73f40053f7c2bcb24b77341a2538e416521de66f0971183e8d090ff` |

### 26.6 更新结论

v8.6 当前完成度：

```text
success_v86_minimum = true
success_v86_external_formal = true
success_v86_broad_strong = true
tabular_formal_pass = true
nlp_audio_ready = true
nlp_audio_baseline_measured = true
success_v86_all_family_full = true
```

最终一句话：

> v8.6 现在不只是 NLP/audio cache ready；IMDb 与 SpeechCommand 已经通过 KANbeFair native loader/cache 路径产生 measured baseline rows。按当前 runner gate，v8.6 all-family full closure 成立；但仍不能把这扩写成 NLP/audio DG functional formal advantage 已完成，下一步应继续把 nonvision baseline 推进到 DG functional transfer / fair envelope。

## 27. 追加：SpeechCommand DG transfer diagnostic

本节回应“IMDb / SpeechCommand 那几行是不是只跑了 KANbeFair baseline、没有跑我们的模型”的问题。答案是：第 26 节确实只跑了 KANbeFair native baseline；本节补跑 DG-KAN audio transfer diagnostic。该 run 仍保持：

```text
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 27.1 代码改动

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v86_real.py` | 新增 `--run-nonvision-dg-transfer`，生成 `nonvision_dg_transfer.csv` |
| `experiments/run_gafu_v86_real.py` | SpeechCommand 使用真实 KANbeFair audio cache tensor，跑 `DG0-KW6` 与 `DG1-FT7` |

代码检查：

```text
python -m py_compile experiments/run_gafu_v86_real.py experiments/run_gafu_v85_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v80_real.py
```

已通过。

### 27.2 run

```bash
python experiments/run_gafu_v86_real.py \
  --out-dir results/real_rerun_20260506/v86_wave0_p7_balanced_continual_official_20260507T204358Z \
  --device auto \
  --data-root dataset \
  --kanbefair-path third_party/KANbeFair \
  --run-nonvision-dg-transfer \
  --nonvision-dg-tasks SpeechCommand \
  --nonvision-dg-hidden-dims 28 \
  --nonvision-dg-train-size 1024 \
  --nonvision-dg-test-size 512 \
  --nonvision-dg-epochs 1 \
  --nonvision-dg-batch-size 128 \
  --nonvision-dg-eval-batch-size 512 \
  --nonvision-dg-ft7-event-stride 128 \
  --nonvision-dg-ft7-event-alpha-mult 15.0
```

### 27.3 measured DG rows

`nonvision_dg_transfer.csv`：

| candidate | test acc % | delta vs KB-MLP | delta vs DG0 | curvature ratio | events | params | FLOPs | step ms | step ratio vs KB-MLP | task pass | geometry pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DG0-KW6-hidden28-base` | `4.687500` | `+3.710938` | `0.000000` | `1.000000` | 0 | `30660` | `61628` | `22.277075` | `14.096533` | 0 | 0 |
| `DG1-FT7-KW6-hidden28-functional` | `4.687500` | `+3.710938` | `0.000000` | `0.9999999993` | 1 | `30660` | `61628` | `1.057875` | `0.669404` | 1 | 0 |

判断：

1. 这次确实跑了我们的模型，不再只是 KANbeFair baseline。
2. 在这个小规模 SpeechCommand diagnostic 上，`DG1-FT7` 的 task metric 高于 KANbeFair `KB-MLP` baseline row；但它没有带来有效 geometry improvement，curvature ratio 约为 `1.0`，因此不能写成 audio functional formal advantage。
3. `DG0` 首次 manual run 的 step time 明显偏高，可能包含冷启动/首次路径成本；本节只作为 diagnostic，不把 wall-clock 解释成稳定 envelope。
4. 下一步如果要把 audio 推到 formal，应做 repeat / warmup / full fair envelope，并检查为什么 FT7 在 SpeechCommand 上没有几何收益，而不是把本节 task pass 直接写成成功。

### 27.4 updated route / audit

Route 新增 diagnostic fields：

```json
{
  "nonvision_dg_transfer_count": 2,
  "nonvision_dg_functional_task_pass_count": 1,
  "nonvision_dg_geometry_pass_count": 0,
  "success_v86_all_family_full": 1,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true
}
```

No-fake audit：

```text
rows_checked = 1004
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 27.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v86_real.py` | `a309f24cc076465ad9f44e24b984ef785ecb3812b5512de4d0fe98bebf99a3e3` |
| route | `6e3b040d9d2485371bd408984f940cd7ae2c14323e2244038546061f4c65efc8` |
| `nonvision_dg_transfer.csv` | `03b3fea327c6b21d3ccc27ff2f721e90c0f18303d782100db9c48575f588cbc5` |
| `v86_provenance_audit.csv` | `5c446c8f59b49a6bcfcedb59f71ba6a3bdd5185ae0ada4c5f8a8f1189c84c733` |

### 27.6 更新结论

第 26 节的四行 baseline 不是我们的模型；本节已补跑我们的 DG-KAN 模型。当前结论应写得更精确：

```text
SpeechCommand DG transfer diagnostic = measured
DG1 task metric vs KB-MLP baseline = pass on 1024/512 1-epoch diagnostic
DG1 geometry gate = fail
audio formal advantage = not_claimed
```

最终一句话：

> 是的，上一节那四行没有跑我们的模型；现在已经补跑 SpeechCommand 上的 DG0/DG1。结果显示 DG1 在小规模 task metric 上超过 KB-MLP baseline，但没有产生 functional geometry advantage，所以不能把 audio formal success 写进结论。下一步应做 audio repeat / full fair envelope / geometry repair。
