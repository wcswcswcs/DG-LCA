# DG-KAN v9.2.7 FC-PureKAN LQ FullPass FunctionalGate 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.7_FC_PureKAN_LQ_FullPass_FunctionalGate_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 Functional / PureKANConv / PureKANFormer 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.7 执行到一个可审计 terminal route：

```text
route = R5-LQFullPassRepairFail
best_candidate = R2-LQ-t2-h256-fanin-output-scale
primitive_family = LinearLiftQuadraticEdgeBasis
success_v927_purekan_equivalence = true
success_v927_p4 = true
success_v927_p5_nearpass = true
success_v927_p5_fullpass = false
success_v927_functional = false
```

最终 artifact：

```text
results/real_rerun_20260506/v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z/
```

核心结论：

1. LQ strict FC-PureKAN equivalence 审计通过：`LQ-t2-h256` 被落成两层 FullEdge composition，而不是普通 MLP hidden activation path。
2. P4 official compact/recompute gate 仍通过：official repeat forward `0.632342`、backward `0.987953`、step `0.636039`、compact memory `0.969501`。
3. 但 P2 的 strict v9.2.6 timing reproduction 没过：official warmed repeat 比 v9.2.6 reference 快很多，超出计划的 `±0.10` 复现窗口；这不是 P4 系统失败，但按计划不能写成 exact reproduction pass。
4. P2 P5 复现成立：三任务三 seed 仍为 `8/9` near-pass，macro delta `-0.004000`，唯一 miss 仍是 KMNIST seed0。
5. P3 robust 10-seed 成立 near-pass 但未 full-pass：`24/30` rows near-pass，macro delta `-0.005767`，CI95 low `-0.008510`，full pass 未达成。
6. Miss rows 全部来自 KMNIST seeds `0/3/4/5/7/9`，主要归因为 class mode、lift conditioning 与 basis underuse。
7. 预注册 repairs 中 R2 fan-in output scale 最好，3-seed macro delta `-0.001667`，但没有达到 promotion 阈值；P6 survivor confirmation 与 P7 functional 均未打开。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v927_fc_purekan_lq_fullpass_functional_gate.py` | v9.2.7 runner；生成 P0-P8 required artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v927_fc_purekan_lq_fullpass_functional_gate.py
```

已通过。

正式运行：

```bash
python experiments/run_v927_fc_purekan_lq_fullpass_functional_gate.py \
  --out-dir results/real_rerun_20260506/v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --lr 0.0005 \
  --official-memory-mode compact \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20 \
  --p4-repeat-measurements 2 \
  --p5-datasets MNIST,Fashion-MNIST,KMNIST \
  --p5-seeds 0,1,2,3,4,5,6,7,8,9 \
  --p5-train-size 9984 \
  --p5-test-size 2000 \
  --p5-epochs 20 \
  --p5-lr 0.0005 \
  --repair-seeds 0,1,2
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R5-LQFullPassRepairFail",
  "best_candidate": "R2-LQ-t2-h256-fanin-output-scale",
  "best_basis": "t2",
  "best_hidden_dim": 256,
  "primitive_family": "LinearLiftQuadraticEdgeBasis",
  "purekan_equivalence_pass": 1,
  "p4_pass": 1,
  "p5_near_pass": 1,
  "p5_pass": 0,
  "p5_macro_delta": -0.005766669909159343,
  "p5_ci95_low": -0.008509995727292864,
  "p5_seed_win_rate": 0.1,
  "compact_memory_ratio": 0.9695007261731864,
  "conservative_memory_ratio": 1.2651127354110616,
  "memory_accounting_pass": 1,
  "p2_reproduction_pass": 0,
  "p3_near_pass_count": 24,
  "p3_near_pass_rate": 0.8,
  "p5_repair_promoted": 0,
  "functional_open_allowed": 0,
  "primary_blocker": "v926_P4_or_P5_reproduction_failed"
}
```

判断：

1. LQ purity、P4 compact gate、P5 robust near-pass 都成立。
2. 本轮没有达到 full pass：macro delta 仍为负，且 repair 没有 promotion。
3. `p2_reproduction_pass = 0` 的原因是 strict timing reproduction 窗口失败：P4 warmed repeat 明显快于 v9.2.6 reference，不是 memory 或 correctness 失败。

## 3. P1 PureKAN equivalence

Artifact：

```text
p1_lq_purekan_equivalence_audit.csv
p1_lq_symbolic_formula.md
```

关键记录：

| item | value |
|---|---:|
| candidate | `EQ0-LQ-current` |
| formula | two-layer FullEdge identity lift + quadratic edge basis |
| layer count | `2` |
| edge basis | `layer0:B0_identity; layer1:B0_identity,T2_quadratic` |
| non-edge params | `0` |
| ordinary MLP hidden path used | `0` |
| node activation used | `0` |
| manual forward / backward / update | `1 / 1 / 1` |
| GradRelErrMax | `6.448128e-06` |
| GradCosMin | `1.000000` |
| full edge equivalence pass | `1` |

负控 `EQ3-LQ-negative-MLP-hidden-control` 明确记录为 ordinary linear + node quadratic activation diagnostic，`eligible_for_route = 0`。

## 4. P2 reproduction and memory accounting

P4 reproduction artifact：

```text
p2_v926_reproduction_memory_accounting.csv
```

P4 repeat rows：

| repeat | official | forward | backward | step | compact memory | conservative memory | P4 pass | strict reproduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | `1.019997` | `0.931829` | `1.686511` | `0.969501` | `1.265113` | 0 | 0 |
| 1 | 1 | `0.632342` | `0.987953` | `0.636039` | `0.969501` | `1.265113` | 1 | 0 |

说明：

1. repeat 0 保留了 first compiled step overhead，step gate 未过。
2. warmed official repeat 1 真实通过 P4 gate。
3. strict reproduction 仍为 0，因为 v9.2.7 warmed timing 与 v9.2.6 reference 的 `forward=1.086593 / backward=0.735494 / step=1.104636` 不在 `±0.10` 窗口内。
4. Conservative memory ratio `1.265113` 继续保留，说明 official P4 pass 依赖 compact/recompute live-set。

P5 reproduction artifact：

```text
p2_p5_reproduction_rows.csv
```

Summary：

| scope | macro delta | near-pass |
|---|---:|---:|
| all 9 rows | `-0.004000` | `8/9` |
| MNIST | `-0.004333` | `3/3` |
| Fashion-MNIST | `-0.001500` | `3/3` |
| KMNIST | `-0.006167` | `2/3` |

唯一 reproduction miss：

| dataset | seed | delta |
|---|---:|---:|
| KMNIST | 0 | `-0.012500` |

## 5. P3 robust 10-seed confirmation

Artifact：

```text
p3_robust_nearpass_confirmation.csv
```

Protocol：

```text
candidate = LQ-t2-h256
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
train_size = 9984
test_size = 2000
epochs = 20
functional_update = off
baseline = same-param MLP-match
```

Result：

| scope | macro delta | near-pass | win rows |
|---|---:|---:|---:|
| all 30 rows | `-0.005767` | `24/30` | `3/30` |
| MNIST | `-0.005400` | `10/10` | - |
| Fashion-MNIST | `-0.000250` | `10/10` | - |
| KMNIST | `-0.011650` | `4/10` | - |

Robust判断：

```text
near_pass_rate = 0.80
macro_delta = -0.0057666699
CI95 low = -0.0085099957
P5 robust near-pass = true
P5 full pass = false
```

Miss rows：

| dataset | seeds | deltas |
|---|---|---|
| KMNIST | `0,3,4,5,7,9` | `-0.012500,-0.014500,-0.010500,-0.024000,-0.022500,-0.011000` |

## 6. P4 miss attribution

Artifact：

```text
p4_miss_row_failure_attribution.csv
```

Miss attribution：

| dataset | seed | worst class | class gap | CEp99 | margin p10 | lift cond | basis entropy | attribution |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| KMNIST | 0 | 7 | `-0.053398` | `8.516067` | `-1.998872` | `9051.991` | `0.685542` | `M2,M3,M4` |
| KMNIST | 3 | 7 | `-0.063107` | `8.204919` | `-1.875799` | `9786.799` | `0.685462` | `M2,M3,M4` |
| KMNIST | 4 | 2 | `-0.030151` | `8.057662` | `-1.982411` | `10420.224` | `0.685896` | `M3,M4` |
| KMNIST | 5 | 3 | `-0.037383` | `9.103600` | `-2.177046` | `9784.308` | `0.677411` | `M3,M4` |
| KMNIST | 7 | 7 | `-0.048544` | `7.971135` | `-1.903205` | `9710.780` | `0.690613` | `M3,M4` |
| KMNIST | 9 | 5 | `-0.054726` | `8.118559` | `-1.994593` | `9990.940` | `0.679734` | `M2,M3,M4` |

判断：

1. Miss 不是 MNIST/Fashion 普遍失败，而是 KMNIST-specific。
2. 主要机制是 lifted feature conditioning 偏高、basis usage entropy 偏低，且部分 seed 出现 class mode 集中失败。
3. Attribution pass 成立，因此允许进入预注册 P5 repair。

## 7. P5 pre-registered repair

Artifact：

```text
p5_fullpass_repair_candidates.csv
```

Repair smoke 使用三任务 seed `0,1,2`，不改 loss、teacher、sampler、class weight 或 functional。

| candidate | macro delta | near-pass | win rows | P4 pass |
|---|---:|---:|---:|---:|
| R1 orthogonal lift init | `-0.004556` | `7/9` | `2/9` | 1 |
| R2 fanin output scale | `-0.001667` | `8/9` | `4/9` | 1 |
| R4 t2t3 h256 | `-0.018611` | `1/9` | `0/9` | 1 |
| R5 legendre23 h256 | `-0.013500` | `4/9` | `2/9` | 1 |
| R6 t2 h224 | `-0.009833` | `6/9` | `0/9` | 0 |
| R7 t2 h288 | `-0.004167` | `7/9` | `3/9` | 1 |
| R8 margin scale init | `-0.002333` | `9/9` | `2/9` | 1 |

判断：

1. R2 是本轮 best repair，3-seed macro delta 达到 `-0.001667`。
2. 但 promotion 阈值要求相对 base repair subset 至少提升 `+0.003`；R2 未达到，因此 `p5_repair_promoted = 0`。
3. R8 达到 `9/9` near-pass，但 macro delta 仍为负，也未满足 full-pass 或 promotion。
4. 因无 promoted survivor，P6 不打开。

## 8. P6/P7 downstream boundary

P6：

```text
p6_survivor_confirmation.csv = not_run
reason = no_P5_repair_candidate_promoted
```

P7：

```text
p7_functional_open_diagnostic.csv = not_run
reason = P6_near_pass_not_available
```

判断：没有用 functional update 倒灌 AdamW-only success；PureKANConv / PureKANFormer 继续 deferred。

## 9. No-fake audit

`v927_provenance_audit.csv`：

```text
rows_checked = 120
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P1/P2/P3/P4/P5 rows 均来自本轮真实 runner。
2. P6/P7 是明确 `not_run`，没有伪装成通过。
3. Conv/Former 只有 deferred registry，没有 measured candidate。

## 10. Hash

| artifact | SHA256 |
|---|---|
| v9.2.7 plan | `ed7c1a5babb4c03b554b218559bd916eb279c4da9cbc2c7dbae2b7d6ea565ba3` |
| `experiments/run_v927_fc_purekan_lq_fullpass_functional_gate.py` | `097283c2867464a3057f769ec079a8210295be26ae55eba492777bbf21cf34d9` |
| route | `865f9df5c7e4dd48f8c6e7415943d8ba0fcd59b070dd91145ef0ad90401b3630` |
| contract audit | `aaf40d4e79716c4622a38a933865da8bd6539376c11a4f825fe955c8bc2ba17c` |
| P1 equivalence audit | `1cee55bdc15281dedd60ae1a8bb717541c7ceff671dd3dfb567c959f6b0aced0` |
| P2 memory accounting | `15ff5d8f496628b09ba62a7f3c8a6f58df1a88b776618d993882b22ab35130ee` |
| P2 P5 reproduction | `fdd823d506e2972e70f194e820b2c1942ca7fc14404d891cd56008561f56b37f` |
| P3 robust confirmation | `641f04b28bf9ddda3f3268b01393f5ba58af7c7996f9d5dd0287daf5dae54bb8` |
| P4 attribution | `27dd6851875b47ed073c52915078834d57d38055c469855f94bf7505abcfa44c` |
| P5 repairs | `829474a3f3a014d3ee230e9950a002675e3c8c893871c834550d3a255204bf96` |
| P6 survivor | `e5ea3ca230799e8d7355a485219b62324c35a8c8a635729003baa4849864cff9` |
| P7 functional | `0405b1744b00b748c1f1c7eb1eb2c8e4daae1d86064dd4c2fe683cab30861efd` |
| provenance audit | `2c5e8f2d840f4525cab8be753246cb13231aa8febe1a16ff788bc407217eb0d2` |

## 11. 最终分析结论

v9.2.7 的真实推进是：

```text
LQ strict FC-PureKAN purity = pass
P4 compact/recompute gate = pass
P5 robust near-pass = pass
P5 full-pass = fail
pre-registered repair promotion = fail
functional / Conv / Former = not opened
```

机制判断：

1. `LinearLiftQuadraticEdgeBasis` 可以作为 strict FC-PureKAN primitive 保留；它不是普通 node activation MLP path。
2. v9.2.6 的 near-pass 不是 3-seed 偶然：10-seed 下仍有 `24/30` near-pass，macro delta `-0.005767`。
3. 但 full-pass 仍没成立；KMNIST 是主要 blocker，`6/10` KMNIST rows 未过 near-pass。
4. P4 compact memory 口径成立，但 strict v9.2.6 timing reproduction 没过，因为 warmed repeat 与旧 timing 差异太大；这需要后续做更稳定的 timing protocol，而不是把本轮写成 exact reproduction。
5. R2/R8 有改善信号，但没有达到预注册 promotion / full-pass 要求，因此不能打开 P6/P7。

最终一句话：

> v9.2.7 真实执行后停在 `R5-LQFullPassRepairFail`：LQ 已通过 strict PureKAN 审计和 P4 gate，并在 10-seed 下保持 robust near-pass；但 full-pass 未达成，预注册 repairs 未 promotion，functional / Conv / Former 仍不能打开。

## 12. 追加：runner 维护性重构

本轮复盘落盘后，针对 runner 代码过重的问题做了维护性拆分；没有重跑正式实验，也没有改变上述实验数据。

新增：

| 文件 | 作用 |
|---|---|
| `dgkan/models/fc_purekan_lq.py` | 持有 `LinearLiftQuadraticEdgeBasis` 的 `LQSpec`、basis eval、manual forward/backward、初始化、logits eval、lift/basis conditioning metrics |

调整：

| 文件 | 改动 |
|---|---|
| `experiments/run_v927_fc_purekan_lq_fullpass_functional_gate.py` | 移除 LQ primitive 数学实现，改为调用 `dgkan.models.fc_purekan_lq`；runner 只保留 P0-P8 编排、artifact 写出、route 判定和实验诊断 |

验证：

```text
python -m py_compile dgkan/models/fc_purekan_lq.py experiments/run_v927_fc_purekan_lq_fullpass_functional_gate.py
```

已通过。另做了一次小规模 smoke run 验证重构后 runner 可执行，smoke artifact 已删除，不计入本复盘正式实验结论。

说明：第 10 节 hash 对应正式实验运行时的脚本版本；本节是实验后的维护性重构记录，不改变第 0-11 节的 route、CSV、JSON 或 no-fake audit 结论。
