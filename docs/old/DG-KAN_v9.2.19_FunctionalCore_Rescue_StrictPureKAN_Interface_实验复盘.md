# DG-KAN v9.2.19 FunctionalCore Rescue StrictPureKAN Interface 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.19_FunctionalCore_Rescue_StrictPureKAN_Interface_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 one-step replay 伪装成 20-epoch strong-control replay。

## 0. 最新结论

截至本轮，v9.2.19 执行到一个可审计 terminal route：

```text
route = R7-FunctionalPausedButNotAbandoned
base_candidate = LQ-t2-h256
success_v9219_functional_core_retained = false
success_v9219_strict_purekan_functional = false
success_v9219_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9219_functional_core_rescue_strictpurekan_interface_first_20260510T080000Z/
```

核心结论：

1. 本轮没有只做 v8 source recap；新增了真实 v8 KW4 one-step strong-control replay，覆盖 AdamWParallel / LRScale / NoOp / Random / Shuffled / Inverted controls。
2. 但本轮只完成 one-step replay，不是计划要求的 20-epoch P1 strong-control replay；因此不能写 `v8_strong_control_replay_pass=true`。
3. one-step replay measured rows = `450`；functional 有 row-level / curvature 机制信号，但 CEp99 / margin aggregate 仍由 AdamWParallel / LR controls 领先。
4. P3-P7 均因 full-epoch v8 strong-control replay 未完成而明确 `not_run`；没有用 strict PureKAN interface 或 full functional training 越过 P1。
5. 当前 route 是 `R7-FunctionalPausedButNotAbandoned`：functional 仍作为研究核心保留，但本轮还不能把 functional retained success 写成达成。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9219_functional_core_rescue.py` | v9.2.19 runner；生成 P0-P7 artifacts、v8 one-step strong-control replay、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9219_functional_core_rescue.py
```

正式运行：

```bash
python experiments/run_v9219_functional_core_rescue.py \
  --out-dir results/real_rerun_20260506/v9219_functional_core_rescue_strictpurekan_interface_first_20260510T080000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "basis_factory_functional_pass": 0,
  "external_ready": 0,
  "functional_core_retained": 0,
  "functional_short_full_pass": 0,
  "next_required_implementation": "complete_full_epoch_v8_strong_control_replay_or_promote_measured_one_step_controls_to_50_240_step_short_run",
  "primary_blocker": "one_step_v8_functional_signal_exists_but_full_epoch_strong_control_replay_not_completed",
  "route": "R7-FunctionalPausedButNotAbandoned",
  "success_v9219_external_ready": 0,
  "success_v9219_functional_core_retained": 0,
  "success_v9219_strict_purekan_functional": 0,
  "v8_full_epoch_strong_control_replay_done": 0,
  "v8_lr_equivalent": 0,
  "v8_mechanism_identified": 0,
  "v8_one_step_best_functional_CEp99_delta": -0.05656584103902181,
  "v8_one_step_best_functional_curvature_delta": -0.2520451419614157,
  "v8_one_step_best_functional_margin_delta": 0.04670259952545166,
  "v8_one_step_best_lr_CEp99_delta": -0.061988695462544756,
  "v8_one_step_best_lr_curvature_delta": -0.03994573334441611,
  "v8_one_step_best_lr_margin_delta": 0.051592735449473064,
  "v8_one_step_best_parallel_CEp99_delta": -0.061988695462544756,
  "v8_one_step_best_parallel_curvature_delta": -0.03994573278809715,
  "v8_one_step_best_parallel_margin_delta": 0.051592735449473064,
  "v8_one_step_functional_beats_controls": 1,
  "v8_one_step_functional_curvature_aggregate_pass": 1,
  "v8_one_step_functional_tail_margin_aggregate_pass": 0,
  "v8_one_step_lr_equivalent_signal": 0,
  "v8_one_step_row_count": 450,
  "v8_strong_control_replay_pass": 0,
  "v9_interface_pass": 0
}
```

判断：本轮补上了 v8 strong-control 的真实测量起点，但没有完成 full P1，因此 functional core 不能宣布 retained success。

## 3. P1 v8 one-step strong-control replay

Artifact：

```text
p1_v8_strong_control_replay.csv
p1_v8_strong_control_replay_summary.csv
paired_replay_branch_trace_v9219.csv
```

Top control summary（按 CEp99 delta / margin delta 排序）：

| candidate | CEp99 delta mean | margin delta mean | step q90 | beats AdamWParallel | beats best LR |
|---|---:|---:|---:|---:|---:|
| `V8-AdamWParallelSameNorm` | `-0.0619887` | `0.0515927` | `1.49071` | `0.000` | `0.000` |
| `V8-LRScale-1.10` | `-0.0619887` | `0.0515927` | `1.5611` | `0.000` | `0.000` |
| `V8-AdamWParallelTrustRatio-0.03` | `-0.0582793` | `0.0482266` | `1.45386` | `0.000` | `0.000` |
| `V8-LRScale-1.03` | `-0.0582793` | `0.0482266` | `1.2856` | `0.000` | `0.000` |
| `V8-AdamWParallelTrustRatio-0.01` | `-0.0572127` | `0.0472509` | `1.34301` | `0.000` | `0.000` |
| `V8-LRScale-1.01` | `-0.0572127` | `0.0472509` | `1.30233` | `0.000` | `0.000` |
| `V8-AdamWParallelTrustRatio-0.003` | `-0.0568393` | `0.0469103` | `1.37963` | `0.000` | `0.000` |
| `V8-LRScale-1.003` | `-0.0568393` | `0.0469103` | `1.45318` | `0.000` | `0.000` |

说明：

1. `eval_scope = holdout_microbatch`，不是 20-epoch final test。
2. `memory_ratio = not_measured_one_step`，没有伪造成系统 P4 memory。
3. `test_acc/val_acc = not_measured_one_step_holdout_scope`，没有把 one-step holdout 当作正式 P1 full-run accuracy。
4. Aggregate CEp99/margin 不是 functional 胜出：best functional CEp99 delta `-0.0565658`，best AdamWParallel `-0.0619887`，best LR `-0.0619887`。
5. Functional 的正信号主要在 curvature：best functional curvature delta `-0.252045`，best AdamWParallel `-0.0399457`，best LR `-0.0399457`。

## 4. P2 mechanism attribution

Artifact：

```text
p2_v8_mechanism_attribution.csv
role_mechanism_trace_v9219.csv
```

本轮记录了 role update norm、role SNR、branch ratio、effective derivative scale、cos functional AdamW、CEp99/margin/ECE/NLL/curvature delta。  
但因为 P1 full replay 未完成，P2 只能作为 one-step mechanism trace，不能判定 v8 independent mechanism pass。

## 5. P3-P7 downstream boundary

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p3_purekan_functional_interface_redesign.csv` | `P1_full_epoch_v8_strong_control_replay_not_completed` |
| `p4_basis_factory_functional_actuatability.csv` | same |
| `p5_adamw_only_fullpass_repair.csv` | same |
| `p6_functional_short_full_validation.csv` | same |
| `p7_robustness_external_ready.csv` | same |

## 6. No-fake audit

`v9219_provenance_audit.csv`：

```text
rows_checked = 1400
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

## 7. 最终分析结论

v9.2.19 的真实推进是：

```text
v9.2.18: 只能 source-audit，v8 strong controls 缺失。
v9.2.19: 补上 v8 one-step AdamWParallel / LR strong controls，但还未完成 20-epoch replay。
```

机制判断：

1. 计划方向是正确的：必须先检验 v8 functional 是否能打败 AdamWParallel / LR controls，再谈 strict PureKAN interface。
2. 本轮 one-step replay 是必要但不足的证据；它显示 curvature 上有 functional-like 信号，但 CEp99/margin aggregate 仍被 AdamWParallel / LR controls 解释得更好。
3. 因此不能把 v8 functional success 恢复为当前 strict claim，也不能把它降级为 LR-equivalent final conclusion。
4. 下一步应继续实现 full-epoch v8 strong-control replay，或把 one-step replay中最强控制/functional分支提升到 short-run 50/240 step replay；在这之前不要打开 P3-P7。

最终一句话：

> v9.2.19 本轮真实补测了 v8 one-step strong controls，但诚实停在 `R7-FunctionalPausedButNotAbandoned`：这不是 full P1 strong-control replay，因此 functional core 仍未重新证明，strict PureKAN functional 和 external-ready 都不能打开。
