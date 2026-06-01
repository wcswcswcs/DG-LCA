# DG-KAN v9.2.17 Signal-Aligned Functional Update 与 Primitive Reset 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.17_SignalAligned_Functional_or_PrimitiveReset_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P4-P6 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.17 执行到一个可审计 terminal route：

```text
route = R1-AdamWParallelIsLRControl
base_candidate = LQ-t2-h256
best_control = OC8-AdamWParallel-TrustRatio-0.03
best_lr_control = OC8-AdamWParallel-TrustRatio-0.03
best_functional_candidate = SF1-GlobalSNRMetricAdamW
success_v9217_signal_functional = false
success_v9217_full_functional = false
success_v9217_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z
```

核心结论：

1. v9.2.16 boundary 被真实复现：source route 为 `R8-NoExtractableFunctionalAdvantage`，A4e/KMNIST/h80 row-level beat 为 `0`，control-derived / control-contrastive 均未打开。
2. P1 autopsy 继续显示 best control 是 `AdamWParallelDirection`，win count 为 `18007`，归因为 `A1-understep,A2-role_signal`。
3. P2 scalar LR / extra AdamW control matrix 已真实执行：best LR control 为 `OC8-AdamWParallel-TrustRatio-0.03`，AdamWParallel gain 为 `-0.000349464`，best LR gain 为 `0.0013544`。
4. P3 signal-aligned functional metric factory 已真实执行；raw paired survivor count 为 `6`，但扣除 LR/extra-AdamW equivalence 后 effective survivor count 为 `0`。
5. 因 P2 判定 LR/extra-AdamW equivalence，P3 的 raw survivor 不能算 independent functional survivor，P4 short-run、P5 full 10-seed、P6 robustness 全部明确 `not_run`。
6. 当前结论是：signal-aligned metric 在当前 LQ/A4/A7c family 中没有提取出超过 AdamWParallel / scalar LR controls 的 functional advantage，应回到 primitive / basis factory。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9217_signal_aligned_functional_or_reset.py` | v9.2.17 runner；生成 P0-P7 artifacts、scalar LR control matrix、signal metric factory、route/no-fake audit |

代码检查：

```bash
python -m py_compile experiments/run_v9217_signal_aligned_functional_or_reset.py
```

正式运行：

```bash
python experiments/run_v9217_signal_aligned_functional_or_reset.py \
  --out-dir results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --lr 0.0005 \
  --batch-size 128 \
  --train-size 9984 \
  --audit-batch-size 128 \
  --eval-size 512 \
  --warmup-steps 36 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --horizons 1,5,20,80 \
  --events E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E5-KMNISTHardModeEvent,E6-ControlDominanceEvent
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R1-AdamWParallelIsLRControl",
  "base_candidate": "LQ-t2-h256",
  "best_functional_candidate": "SF1-GlobalSNRMetricAdamW",
  "best_control": "OC8-AdamWParallel-TrustRatio-0.03",
  "best_lr_control": "OC8-AdamWParallel-TrustRatio-0.03",
  "adamwparallel_dominance_type": "A1-understep,A2-role_signal",
  "lr_equivalence_pass": 1,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_reentry_pass": 0,
  "functional_task_safe": 1,
  "functional_mechanism_pass": 1,
  "functional_control_pass": 1,
  "functional_system_pass": 1,
  "functional_kmnist_repair_pass": 0,
  "noise_robustness_pass": 0,
  "strong_baseline_pass": 0,
  "return_to_primitive_required": 1,
  "external_ready": 0,
  "primary_blocker": "adamwparallel_dominance_equivalent_to_scalar_lr_or_extra_adamw_control",
  "next_required_implementation": "return_to_primitive_basis_factory_and_adamw_only_fullpass_repair",
  "success_v9217_signal_functional": 0,
  "success_v9217_full_functional": 0,
  "success_v9217_external_ready": 0
}
```

判断：本轮达到 failure stop，而不是 minimum success。P0/P1/P2/P3 均执行并落盘，但 P3 没有任何 signal-aligned metric 同时击败 AdamWParallel 和 best LR control。

## 3. P0 / P1

P0 artifact：

```text
p0_v9216_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `R8-NoExtractableFunctionalAdvantage` |
| best control | `AdamWParallelDirection` |
| A4e row-level beat | `0` |
| P0 pass | `1` |

P1 artifact：

```text
p1_adamwparallel_dominance_autopsy.csv
role_snr_metric_trace_v9217.csv
```

P1 判断：

```text
p1_attribution = A1-understep,A2-role_signal
adamwparallel_win_share = 0.5210358796296296
```

这说明 AdamWParallel dominance 仍然是本轮最重要事实，但这本身不能被写成 functional advantage。

## 4. P2 Scalar LR / Extra AdamW Controls

Artifact：

```text
p2_scalar_lr_extra_adamw_control_matrix.csv
```

Summary：

| item | value |
|---|---:|
| rows | `1440` |
| best LR control | `OC8-AdamWParallel-TrustRatio-0.03` |
| best control | `OC8-AdamWParallel-TrustRatio-0.03` |
| AdamWParallel gain | `-0.000349464` |
| best LR gain | `0.0013544` |
| LR equivalence | `1` |

判断：AdamWParallel 的优势仍与 extra AdamW / trust-ratio control 高度纠缠，不能作为独立 functional evidence。

## 5. P3 Signal-Aligned Functional Metric Factory

Artifact：

```text
p3_signal_aligned_functional_metric_factory.csv
functional_metric_event_trace_v9217.csv
```

Summary：

| item | value |
|---|---:|
| rows | `1296` |
| raw paired survivor count | `6` |
| effective control-resistant survivor count | `0` |
| best functional | `SF1-GlobalSNRMetricAdamW` |
| best gain | `0.0622863` |
| best CEp99 delta | `-0.0400701` |
| best margin delta | `0.0242665` |

判断：SF1-SF9 即使出现局部 paired pass，也被 P2 的 LR/extra-AdamW equivalence 截断，不能算控制抗性的 functional survivor。也就是说，把 functional update 改写成 role/SNR/tail/hard-mode signal metric，在当前实现空间里仍没有超过 optimizer controls。

## 6. P4-P7 Boundary

P4/P5/P6 均为：

```text
not_run
reason = P2_lr_equivalence_blocks_independent_functional_open
```

P7：

```text
return_to_primitive_required = 1
next_required_implementation = return_to_primitive_basis_factory_and_adamw_only_fullpass_repair
```

判断：没有用 short-run/full-run/robustness 越过 paired replay gate。

## 7. No-fake audit

`v9217_provenance_audit.csv`：

```text
rows_checked = 40256
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `docs/DG-KAN_v9.2.17_SignalAligned_Functional_or_PrimitiveReset_完整实验计划.md` | `d0eeb83fa92834ee1ea5690fd06655ac6a02ca03ff71ff324a3e7d4209f80868` |
| `experiments/run_v9217_signal_aligned_functional_or_reset.py` | `c7adebf146930a3c434bda97c77c4a98fd0a625faa026ea9b726880ad103559b` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/contract_audit_v9217.csv` | `976c40f6660f56d45ec10ffe39c3cb10f1e39f568b15eac12d63f72140c038bb` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/p0_v9216_boundary_reproduction.csv` | `d3280d7162fec8562e953dc585093e5146ae1586770f2f6b2efbf2c7e37f3920` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/p1_adamwparallel_dominance_autopsy.csv` | `4d35e896942e1e1670f7405b5727bb6757016b667a9692306057a6bef1dfdcf3` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/p2_scalar_lr_extra_adamw_control_matrix.csv` | `f200540d68d8e0758c1f7ff53c6c98628de978977f199eac25a4d3b1c5a9d874` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/p3_signal_aligned_functional_metric_factory.csv` | `ff6bf15455553518df83991910b61a5fe6cc7e8410bdf36d72a144c5326ef1d3` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/p7_primitive_basis_reset_decision.csv` | `32cced78040710abb2019e060045139aa85cb7a376cf72ee58c7080e346c6f3e` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/route_decision.json` | `3a81231c8f95f0408575eefe9a1882e7211c89e913d9db6b49d864ae9905825a` |
| `results/real_rerun_20260506/v9217_signal_aligned_functional_or_reset_first_20260510T050000Z/v9217_provenance_audit.csv` | `584e55f3c9b9500050c2fbcf492024f9ba00da02a4de069dadb31b14e4bdad4f` |

## 9. 最终分析结论

v9.2.17 的真实推进是：

```text
v9.2.16: output target / control-derived target / solver 都没有可提取 functional advantage。
v9.2.17: 改测 signal-aligned optimizer metric，仍没有找到超过 AdamWParallel / LR controls 的 survivor。
```

机制判断：

1. 当前最强信号仍来自 AdamW/optimizer direction，而不是独立 functional geometry。
2. Signal-aligned functional metric 没有把这个事实转化为可归因优势；它要么近似 extra LR/trust-ratio，要么效应太小。
3. 这意味着继续调 SNR、event、step fraction 或现有 actuator target 已经不是高价值路径。
4. 当前应按计划回到 primitive / basis factory，优先修 AdamW-only full-pass 与 stronger baseline challenge，再决定是否重开 functional。

最终一句话：

> v9.2.17 真实执行后停在 `R1-AdamWParallelIsLRControl`：signal-aligned functional metric 没有击败 AdamWParallel / scalar LR controls，当前 LQ/A4/A7c family 下没有可提取 functional advantage，应正式回到 primitive / basis design。
