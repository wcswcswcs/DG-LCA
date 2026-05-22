# DG-KAN v11.0 Function-Space Natural Update 实验复盘

> 本复盘记录 `DG-KAN_v11.0_FunctionSpaceNaturalUpdate_完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.8 真实 artifact、真实 MNIST/Fashion-MNIST/KMNIST 小样本诊断、计划内 synthetic/tabular 诊断与 autograd discovery reference；没有 fake data、proxy rows、占位数据或 CPU offload。graph-free official controller 因手写 JVP/VJP 未就绪保持关闭。

## 0. 最新结论

```text
route = RouteC-MLPFunctionalUpdateFails
primary_blocker = mlp_function_space_update_not_validated
secondary_blocker = short_training_smoke_not_passed
system_legal_controller_pass = 0
generated_route_status = stopped_mlp_functional_update_rebuild
```

最终 artifact：`results/real_rerun_20260506/v1100_function_space_natural_update_full_20260519T000000Z`

核心结论：

1. P0 v10.8 boundary lock pass = `1`；source route = `CaseA-KnownGoodDirectReplayFail`；payload hash match rate = `1.0`，state/optimizer/batch identity match rate = `0.0` / `0.0` / `0.0`。
2. P1 discovery/correctness/official = `1` / `1` / `0`；MLP/LQ discovery = `1` / `1`；max JVP FD error = `3.0439035031691106e-09`，max dot error = `1.0408340855860843e-17`，max direct/amortized cost ratio = `6.118286336735954` / `0.6118286336735954`。
3. P2 weak/strong = `1` / `0`；MLP mean predicted/actual cosine = `0.9804886894881899`；heldout nonharm rate = `0.6666666666666666`；beats random rate = `1.0`。
4. P3 weak/strong = `1` / `0`；mean signal ratio/random = `4.549193720157593` / `0.3283289344206694`；heldout improved rate = `1.0`。
5. P4 weak/strong = `1` / `1`；memory/hard-tail noharm = `1.0` / `1.0`；heldout improve = `1.0`。
6. P5 short-training smoke status = `summary`，task pass = `0` / `6`，weak/strong = `0` / `0`；P6/P7/P8 = `not_run` / `not_run` / `not_run`；No-fake rows checked = `144`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/functional_space_natural_step.py experiments/run_v1100_function_space_natural_update.py
```

```bash
python experiments/run_v1100_function_space_natural_update.py --out-dir results/real_rerun_20260506/v1100_function_space_natural_update_full_20260519T000000Z --fresh --device auto --data-root data --seed 2110 --execution-profile full-gated --reference-size 2
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1100",
  "status": "summary",
  "route": "RouteC-MLPFunctionalUpdateFails",
  "primary_blocker": "mlp_function_space_update_not_validated",
  "secondary_blocker": "short_training_smoke_not_passed",
  "route_explanation": "RouteC-MLPFunctionalUpdateFails: mlp_function_space_update_not_validated; short_training_smoke_not_passed.",
  "source_route_v1080": "CaseA-KnownGoodDirectReplayFail",
  "P0_boundary_pass": 1,
  "P1_discovery_pass": 1,
  "P1_correctness_pass": 1,
  "P1_official_candidate_pass": 0,
  "P2_weak_pass": 1,
  "P2_strong_pass": 0,
  "P3_weak_pass": 1,
  "P3_strong_pass": 0,
  "P4_weak_pass": 1,
  "P4_strong_pass": 1,
  "P5_weak_pass": 0,
  "P5_strong_pass": 0,
  "P6_weak_pass": 0,
  "P6_strong_pass": 0,
  "P7_weak_pass": 0,
  "P7_strong_pass": 0,
  "P8_controller_pass": 0,
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "generated_route_status": "stopped_mlp_functional_update_rebuild",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P0 Boundary

| metric | value |
| --- | ---: |
| source route | `CaseA-KnownGoodDirectReplayFail` |
| known-good direct replay pass | `0` |
| known-good direct FastSlow-like count | `0` |
| state hash match rate | `0.0` |
| optimizer hash match rate | `0.0` |
| batch sequence hash match rate | `0.0` |
| payload hash match rate | `1.0` |

## 4. P1-P5 摘要

```text
P1 correctness pass = 1
P1 discovery pass = 1 (MLP=1, LQ=1)
P2 weak/strong = 1 / 0
P3 weak/strong = 1 / 0 (best repair=larger_reference_memory)
P4 weak/strong = 1 / 1
P5 weak/strong = 0 / 0 (task pass=0/6)
```

解释：P1 的 MLP/LQ autograd reference 只作为 discovery；本轮按 cost blocker 将工程 reference size 降到 `2`，并按计划记录 periodic10 amortized cost repair。JVP/VJP/solve 数值正确性通过，direct cost ratio 仍如实记录；P1 discovery gate 使用 correctness + direct-or-amortized cost gate。`P1_official_candidate_pass=0` 是因为 graph-free edge-function 还没有手写 JVP/VJP。按计划修复后，P3 通过 `larger_reference_memory` repair 打开了 P5；P5 真实短程训练 smoke 执行后 `P5_weak_pass=0`，所以 P6-P8 仍保持 `not_run`。

## 5. 实现思路与伪代码

本轮实现分两层：`functional_space_natural_step.py` 只实现函数空间自然步的最小数学内核；`run_v1100_function_space_natural_update.py` 负责按计划执行 P0-P8、写 CSV/JSON/manifest/recap，并强制区分 discovery reference 与 official graph-free candidate。

### 5.1 P0 旧路线冻结

```text
load v10.8 route_decision
load known-good direct replay summary
load direct replay identity audit

assert source_route == CaseA-KnownGoodDirectReplayFail
record payload/state/optimizer/batch hash match rates
set disable_action_selector_routes = 1
do not open FPO/FPAU/GoodCone/LFRO/generated action routes
```

本轮 P0 的作用不是继续修 known-good action，而是确认历史好动作在当前 replay identity 下不可稳定复现，所以 v11.0 改走 function-space update。

### 5.2 函数空间自然步核心

`FunctionalModuleAdapter` 使用 `torch.func.functional_call` 把模型参数展平成 `theta_flat`，所有 JVP/VJP 都在同一 flat 参数顺序下计算，避免参数块顺序错配。

```text
forward_outputs(model, batch):
  logits = model(x)
  return logits, light_cache

vjp_outputs_to_params(model, batch, u):
  theta = flatten(model.parameters)
  logits = f(theta, x)
  scalar = dot(flatten(logits), flatten(u))
  return grad(scalar, theta)

jvp_params_to_outputs(model, batch, v):
  theta = flatten(model.parameters)
  return jvp(lambda th: f(th, x), theta, v)
```

核心求解器是显式输出空间核版本，只适合小 batch discovery，不写成 official runtime：

```text
solve_functional_step(model, train_batch, reference_batch):
  logits_R = f(theta, x_R)
  r = (softmax(logits_R) - onehot(y_R)) / batch_size

  for each output coordinate e_i:
    g_i = J_R^T e_i               # VJP

  G = stack(g_i)                  # [output_dim * reference_size, parameter_count]
  K = G @ G.T                     # empirical function-space kernel
  alpha = solve(K + lambda I, r)
  delta_theta = -G.T @ alpha
  predicted_delta = J_R delta_theta

  if norm(predicted_delta) > trust_radius:
    scale delta_theta and predicted_delta

  record kernel rank/condition, solve residual, param/function norm, cost
```

P1 correctness 用两条真实数值检查：

```text
finite_difference_jvp_error =
  ||JVP(v) - (f(theta + eps v) - f(theta)) / eps|| / denominator

jvp_vjp_consistency_error =
  |<Jv, u> - <v, J^T u>| / denominator
```

### 5.3 P2 单步正确性

P2 不读 outcome label，不训练 selector，只比较同一个当前 batch 下的真实 loss 变化：

```text
for model_type in [MLP, LQ]:
  for task in real MNIST/Fashion/KMNIST + plan synthetic/tabular:
    build current train / heldout / memory / hard-tail batches
    delta_fsnu = solve_functional_step(...)
    delta_adamw = -grad(CE), rescaled to same parameter norm
    delta_random_param = random vector with same parameter norm
    delta_random_function = random vector rescaled to same JVP norm

    apply each delta to a cloned model
    record train/heldout/memory/hard-tail loss before/after
    record predicted vs actual output delta cosine
    record whether FSNU beats random same-norm perturbation
```

因此 P2 的 `weak=1` 只说明单步函数移动和 loss 变化有弱信号；它不是 controller，也不是未来路径成功。

### 5.4 P3 信号/水库诊断

P3 检查函数空间方向是否跨 split 稳定，而不是只吃当前 batch 噪声：

```text
split train batch into k subsets
for each split s:
  delta_s = solve_functional_step(split_s)
  u_s = J_probe delta_s

mean_u = mean_s(u_s)
signal_ratio = ||mean_u||^2 / mean_s(||u_s - mean_u||^2)

build random directions with matched norm
random_signal_ratio = same formula on random directions

also solve with shuffled labels
require real-label benefit > noise-label benefit
```

本轮不是停在 baseline P3 失败；runner 按计划执行了 repair cycle：

```text
variants = baseline, residual_centering, larger_reference_memory,
           stable_top3_eigen, reservoir_shrink_top2

for variant:
  run split-stability diagnostic
  compare against matched-norm random directions
  run shuffled-label diagnostic
  check memory/hard-tail no-harm
select best variant by weak_count, signal-random gap, heldout improvement
```

实际最佳修复：

```text
best_variant = larger_reference_memory
repair_variant_count = 4
weak_trial_count = 4 / 4
mean_signal_ratio = 4.549193720157593
mean_random_signal_ratio = 0.3283289344206694
heldout_improved_rate = 1.0
```

因此 P3 从 baseline failure 被修到 `P3_weak_pass=1`，但还没有达到 strong gate；这一步允许进入 P5 smoke，但不能写成 official controller。

### 5.5 P4 memory / hard-tail 约束

P4 把 memory/hard-tail 作为 reference set 约束近似，而不是 hard veto：

```text
for solver in candidate_solvers:
  reference = train
  optionally append memory batch one or more times
  optionally append hard-tail batch
  increase ridge / reduce trust radius for safer solvers

  delta = solve_functional_step(train, reference)
  apply delta
  record train / heldout / memory / hard-tail deltas
```

本轮 `memory_hardtail_reference` 通过 weak/strong，是因为小样本下 memory 和 hard-tail no-harm 率为 1.0；P4 只能证明约束求解在诊断样本上没有伤害 memory/hard-tail，不能替代 P5 短程训练。

### 5.6 P5 短程训练 smoke

```text
open P5 if:
  MLP_discovery_pass = 1
  P1_correctness_pass = 1
  P2_weak_pass = 1
  P3_weak_pass = 1
  P4_weak_pass = 1

for task in six MLP diagnostic tasks:
  run AdamW baseline
  run AdamW + FSNU every step
  run AdamW + FSNU every 5 steps
  run AdamW + FSNU every 10 steps
  run AdamW + matched-norm random-function update every 5 steps
  run AdamW + signal-channel-filtered FSNU every 5 steps
  run AdamW + memory-constrained FSNU every 5 steps
  compare validation-loss AUC, memory loss, hard-tail loss, bad steps, and random control
```

本轮：

```text
P5_status = summary
P5_task_pass_count = 0 / 6
P5_weak_pass = 0
P5_strong_pass = 0
```

P5 已经真实运行，不再是 `not_run`；但只有 `0` / `6` 个任务过 smoke gate，低于 weak gate，所以 P6 graph-free migration、P7 future-path verification、P8 official controller/runtime/paired replay 仍保持 `not_run`。

## 6. No-Fake / Hash

```text
rows_checked = 144
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `contract_audit_v1100.csv` | `d33447138f7863cc18bbf1039b893b232631f70ba02960cc6740ae0994427967` |
| `fig_p1_core_correctness_v1100.svg` | `c48b01e5e6658eb674e753d228aceea3927a5c5388a617e99acc578f82bf2276` |
| `fig_p2_single_step_v1100.svg` | `6f224cd1a5cf7b1996df9eebbbcfb7ce1960ae021b2a073bbc2f6ea55d3c3dfc` |
| `fig_p3_signal_reservoir_v1100.svg` | `ac49d98664c2cc97cbc4b18d227b4279815564a91deddf39f117115382890e62` |
| `fig_p4_constraint_pareto_v1100.svg` | `faeba111a210d5976f3a8a52e9e17a7f874a6ffb0a6f146f423288a9450f5ce3` |
| `fig_v1100_route_matrix.svg` | `86b2ab0bddb4e1be4719317d442dc9b84dd12a8b85bf5ca859a5705b9e852747` |
| `no_fake_audit_v1100.csv` | `e1a0cf1671c95054e635c7ce0e47c99010df715046bf43221a621e86e1009d4f` |
| `p0_v1080_boundary_lock_v1100.csv` | `9b6417fde5b626c76cc9c954863ccf783ae3b8938fdcd7883b1b952a31fae306` |
| `p1_function_space_core_v1100.csv` | `efd2e2acea237bf53294d153142ea847e454b43c06baaaaa0cd6ff35ecb105b8` |
| `p2_single_step_functional_correctness_v1100.csv` | `537dc498ff98db3286b868f510796c33166830a820aef2c03aa2ee27424ceeda` |
| `p3_signal_reservoir_diagnostic_v1100.csv` | `5b24429da79d9a17b84d14931351659e54fd5a7a679ba2f813bf635afa6a37a0` |
| `p4_memory_hardtail_constraint_solve_v1100.csv` | `a28b1cbd0406aa5ae37455d353ce97af89b9a5f8072faa75fc1a541a8b4f429d` |
| `p5_short_training_smoke_v1100.csv` | `d382f01e33cd12c22fea8190265d3c8a85c798729d2f35f993836e169a3501ea` |
| `p6_graph_free_edge_migration_v1100.csv` | `8cc914787994f003f2b25cb2cb491daf0f7177a84a4dfd089c84242dcbe087ae` |
| `p7_future_path_verification_v1100.csv` | `72559e9af4e1a918957fb1de84ed9a09d272cb6a668a4ae1122ec13c3db6c345` |
| `p8_controller_runtime_boundary_v1100.csv` | `2c8a96d8c0f148a68a03a0d072960b8f415d5bb5710d235750caa926099deabb` |
| `route_decision_v1100.json` | `9024e003da9ee3014bd684e4def3204edcbc1f27c77d7e01316793bf4bd7c0dc` |
| `plan` | `2c8377fb6b3e9d92079d22c6cdc436d26ff6e7246f06162fd70fd89e20d40311` |
| `runner` | `7ec438f5b36561b9f94f373f6c91e324cc8d448dce8f9557f3348d87b9340832` |
| `functional_space_natural_step` | `70e8bf027c0042ab4418f59d2bc2d1179695ee37614fa683c98cf126546c0e34` |
| `run_manifest_v1100.json` | `481f5dddfb829814da0bde8f7a01250c32e07f1ab9cd555e9eb688af614311cf` |

## 7. 最终分析结论

```text
1. v11.0 按计划停止旧 action-selection / generated action / FastSlow selector 主线，P0 只锁定 v10.8 replay identity blocker。
2. P1 已落地函数空间自然步的 autograd discovery reference，并真实校验 JVP finite difference、JVP/VJP dot consistency 与输出核求解。
3. graph-free official path 没有手写 JVP/VJP，因此没有被写成 official candidate。
4. P3 baseline 失败后已按计划执行 repair cycle，并由 `larger_reference_memory` 修到 weak pass；P5 因此真实打开并运行短程训练 smoke。
5. P5 未通过：短程训练没有在验证损失/随机对照维度证明 MLP function-space update 可推广，因此 P6-P8 显式 not_run。
6. 本轮没有使用 outcome label 训练 controller，也没有生成新 action selector 或 fake/proxy rows。
```

最终一句话：v11.0 真实执行后停在 `RouteC-MLPFunctionalUpdateFails`：RouteC-MLPFunctionalUpdateFails: mlp_function_space_update_not_validated; short_training_smoke_not_passed.
