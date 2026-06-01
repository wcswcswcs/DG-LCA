# DG-KAN v9.2.9 SNR-Gated Functional Update Mainline 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.9_SNR_Gated_Functional_Update_Mainline_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P3-P6 未打开阶段写成通过。

## 0. 最新结论

本文件按执行顺序追加。前面第 1-8 节保留 ghost-batch P1 failure；第 9-12 节保留 low-cost P1 pass；最新结论以第 16 节为准。

截至本轮，v9.2.9 执行到一个可审计 terminal route：

```text
route = R4-SNRGateNotUseful
base_candidate = LQ-t2-h256
success_v929_p1_snr_gate = true
success_v929_functional_opened = false
success_v929_functional_advantage = false
success_v929_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v929_snr_gated_functional_update_p2_warmema_20260509T193500Z/
```

核心结论：

1. P0 functional diagnostic open gate 通过：复用 v9.2.7 真实 artifact，`LQ-t2-h256` 仍满足 PureKAN equivalence、P4 pass、P5 robust near-pass。
2. first-wave ghost-batch SNR 已真实失败：mean overhead ratio `4.061492`，max `5.726611`，不能作为在线 gate。
3. 本轮新增低开销 `ema_role_scalar` estimator：复用正常 AdamW step 的梯度，只维护 role-level scalar EMA SNR；不测 per-param histogram，并在 CSV 写明 `not_measured_role_scalar_ema`。
4. calibrated P1 已真实通过：eligible rows `270`，mean amortized overhead `0.125983`，max `0.130215`，active fraction `0.051099-0.298428`，max role CV `0.206763`。
5. P2 one-step population-risk audit 已真实打开：`F1-SNRGatedTaskProjected` 共 `144` rows；SNR gate 比 no-SNR geometry control 略好，但未达到 safety gate。
6. P2 failure 是 terminal blocker：prediction corr `0.231090 < 0.30`，bad-step rate `0.819444 > 0.05`，holdout non-harm fraction `0.180556 < 0.70`。
7. 因 P2 未过，P3 short-run、P4 full functional re-entry、P5 strong baseline、P6 robustness 全部明确 `not_run`；不能声明 functional update advantage。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `dgkan/functional/snr_gated_lq.py` | LQ role-aware SNR instrumentation：microbatch gradient、role SNR、per-param histogram、overhead timing、not-run row helper |
| `experiments/run_v929_snr_gated_functional_update_mainline.py` | v9.2.9 runner；负责 P0/P1 gate、artifact 写出、route/failure/no-fake audit |

代码检查：

```bash
python -m py_compile DG-LCA/dgkan/functional/snr_gated_lq.py DG-LCA/experiments/run_v929_snr_gated_functional_update_mainline.py
```

已通过。

正式运行：

```bash
python DG-LCA/experiments/run_v929_snr_gated_functional_update_mainline.py \
  --out-dir results/real_rerun_20260506/v929_snr_gated_functional_update_mainline_p1_20260509T181000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --lr 0.0005 \
  --p1-candidates LQ0,LQ1 \
  --p1-datasets MNIST,Fashion-MNIST,KMNIST \
  --p1-seeds 0,1,2 \
  --p1-microbatches 4,8 \
  --p1-steps 3 \
  --p1-train-size 9984
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R6-FunctionalSystemOverheadFail",
  "base_candidate": "LQ-t2-h256",
  "p4_pass": 1,
  "p5_near_pass": 1,
  "p5_full_pass": 0,
  "p1_snr_row_count": 324,
  "p1_snr_stability_pass": 0,
  "p1_snr_overhead_pass": 0,
  "p1_snr_active_fraction_pass": 1,
  "p1_snr_pass": 0,
  "mean_snr_overhead_ratio": 4.06149152972046,
  "max_snr_overhead_ratio": 5.726611334239872,
  "min_active_fraction_tau1": 0.20976562798023224,
  "max_active_fraction_tau1": 0.7515625357627869,
  "max_snr_role_cv": 0.8649414595325146,
  "primary_blocker": "P1_snr_compute_overhead_exceeds_20_percent_step_gate",
  "next_required_implementation": "implement_low_overhead_running_or_subsampled_SNR_estimator_before_P2",
  "success_v929_functional_opened": 0,
  "success_v929_functional_advantage": 0,
  "success_v929_external_ready": 0
}
```

判断：

1. v9.2.9 并没有进入 functional update 训练；只打开了 P1 instrumentation。
2. 当前 blocker 不是 LQ base 不合格，而是计划中的 SNR estimator 以 ghost-batch 方式计算过重，无法作为主线 update gate。
3. 因 P1 未通过，不能执行 P2-P6，也不能声明 functional advantage。

## 3. P0 functional open gate

P0 source artifact：

```text
results/real_rerun_20260506/v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z/
```

`functional_open_gate_v929.csv`：

| candidate | functional diagnostic can open | source |
|---|---:|---|
| `LQ-t2-h256` | 1 | v9.2.7 route / P4 / P5 near-pass |

判断：

1. P0 gate 通过，只说明允许做 gated functional diagnostic。
2. P0 不代表 functional update 已经成功，也不代表 AdamW-only full-pass。

## 4. P1 SNR instrumentation

P1 artifact：

```text
snr_instrumentation_v929.csv
```

总行数：

```text
rows = 324
functional_update_used = 0
loss_type = CE
uses_loss_backward = 0
```

按 candidate：

| candidate | rows | mean overhead | max role CV | active tau1 range | overhead pass | stability pass | active pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LQ0-LQ-t2-h256-AdamW` | 162 | `4.0703` | `0.8649` | `0.2301-0.7516` | `0/162` | `147/162` | `162/162` |
| `LQ1-LQ-t2-h256-fanin-output-scale` | 162 | `4.0527` | `0.7929` | `0.2098-0.7355` | `0/162` | `147/162` | `162/162` |

按 microbatch count：

| microbatches | rows | mean overhead | max role CV | active tau1 range | overhead pass | stability pass | active pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 162 | `2.7590` | `0.8649` | `0.3699-0.7516` | `0/162` | `138/162` | `162/162` |
| 8 | 162 | `5.3640` | `0.5364` | `0.2098-0.6152` | `0/162` | `156/162` | `162/162` |

按 role：

| role | rows | mean overhead | max role CV | active tau1 range | overhead pass | stability pass | active pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lift_identity` | 108 | `4.0615` | `0.4559` | `0.3457-0.6975` | `0/108` | `108/108` | `108/108` |
| `output_linear` | 108 | `4.0615` | `0.5931` | `0.3836-0.7465` | `0/108` | `99/108` | `108/108` |
| `quadratic_coeff` | 108 | `4.0615` | `0.8649` | `0.2098-0.7516` | `0/108` | `87/108` | `108/108` |

判断：

1. `tau=1` 的 active fraction 有信息量，不是全 0 或全 1。
2. 但 ghost-batch SNR 每次要做 4/8 个 microbatch manual backward，实测 overhead 是普通 step 的 `2.76x-5.36x`，不可能通过 `<=0.20` gate。
3. `quadratic_coeff` role 的 SNR CV 最高，说明这个最关键 basis channel 的 signal/noise 估计还不稳定。
4. 因此当前 SNR gate 不能作为在线 functional update 主线使用。

## 5. Downstream boundary

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | status | reason |
|---|---|---|
| `one_step_population_risk_audit_v929.csv` | `not_run` | `P1_snr_instrumentation_failed_so_functional_update_not_opened` |
| `short_run_functional_safety_v929.csv` | `not_run` | `P1_snr_instrumentation_failed_so_functional_update_not_opened` |
| `full_functional_reentry_10seed_v929.csv` | `not_run` | `P1_snr_instrumentation_failed_so_functional_update_not_opened` |
| `strong_baseline_challenge_v929.csv` | `not_run` | `P1_snr_instrumentation_failed_so_functional_update_not_opened` |
| `robustness_noise_diagnostic_v929.csv` | `not_run` | `P1_snr_instrumentation_failed_so_functional_update_not_opened` |
| `functional_event_trace_v929.csv` | `not_run` | `P1_snr_instrumentation_failed_so_functional_update_not_opened` |

判断：没有用 functional update 越过 P1 gate，也没有把 P2-P6 未运行结果写成成功。

## 6. No-fake audit

`v929_provenance_audit.csv`：

```text
rows_checked = 333
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0 是对 v9.2.7 真实 source artifact 的可追溯 recap。
2. P1 是本轮真实 SNR instrumentation。
3. P2-P6 是明确 `not_run`，没有 fake/proxy。

## 7. Hash

| artifact | SHA256 |
|---|---|
| v9.2.9 plan | `51b0679dc36847a7a4a1154129f60059328bfef35adad618e408cf8e84b3fba5` |
| `dgkan/functional/snr_gated_lq.py` | `c3224b5247a33c9ba9df2141d97f1efb1fb294c158d5d6b1ccc948bce7ac5a62` |
| `dgkan/models/fc_purekan_lq.py` | `feb2bb205b0ad11155bca1c74d9f938d60cd6821c38a8c99a30c8db067669f8c` |
| `experiments/run_v929_snr_gated_functional_update_mainline.py` | `ded03f81f8dbe214a98f5b7a66ea66e3192ab1a73208319b7eaffbaf0182d270` |
| route | `a3495afcc54d88ba31b5983ef40755dbfa06ec6cbe9435dd87b02e3d77de6314` |
| `snr_instrumentation_v929.csv` | `c68fcb7ca4d227abc7ab24e72b054311e6e2e6fa0596c3de3d0d82b971607c3a` |
| provenance audit | `6399323ec34bd4036674d71885b7f29f95cd1f2deedb45bc590426b84891011a` |

## 8. 最终分析结论

v9.2.9 本轮没有证明 functional update advantage。它证明的是一个更早、更硬的系统事实：

```text
LQ near-pass base 已允许 functional diagnostic；
role-aware SNR gate 有非塌缩 active fraction；
但 ghost-batch microbatch SNR estimator 太慢，且 quadratic role SNR 稳定性不足；
因此 functional update 主线不能用当前 estimator 打开。
```

机制判断：

1. SNR gate 本身不是完全无信息：active fraction 落在 `0.21-0.75`，说明 `tau=1` 没有退化成全开或全关。
2. 但 SNR 估计方式不适合在线训练：即使 microbatch=4，mean overhead 也达到 `2.759x`，远高于计划的 `0.20x`。
3. microbatch=8 稳定性略好，但 overhead 更高，达到 `5.364x`，因此不能通过“多 microbatch”解决。
4. `quadratic_coeff` 是当前 LQ 的核心 channel，它的 max CV `0.8649`，说明 signal/noise 估计仍不够稳。
5. 下一步不能直接做 P2/P3 functional update；应先实现低开销 SNR estimator，例如 running EMA SNR、低频 subsampled SNR、role-level batch-stat proxy，或离线校准后在线轻量 gate。

最终一句话：

> v9.2.9 已正式打开 SNR-gated functional 主线的第一道 instrumentation gate，但诚实停在 `R6-FunctionalSystemOverheadFail`：SNR active fraction 有信息量，然而当前 ghost-batch estimator overhead 是 step 的 `4.06x` 平均值，无法进入 P2/P4 functional update。

## 9. 追加：低开销 SNR estimator

根据第 8 节 blocker，本轮继续做低开销 SNR estimator，然后回到 P1。实现原则：

```text
不改 loss
不引入 teacher / distillation
不改 sampler / class weight
不打开 functional update
不把 role-level scalar SNR 伪装成 per-parameter SNR
```

代码改动：

| 文件 | 改动 |
|---|---|
| `dgkan/functional/snr_gated_lq.py` | 新增 `EMARoleSNRState`、`ScalarRoleSNRState`、`measure_ema_role_snr_step`、`measure_scalar_ema_role_snr_step` |
| `experiments/run_v929_snr_gated_functional_update_mainline.py` | 新增 `--p1-estimator ema_role / ema_role_scalar`、`--p1-snr-stride`、EMA warmup、smooth role active fraction |

关键实现边界：

```text
ghost_microbatch:
  真实 microbatch gradients，完整但太慢。

ema_role:
  per-parameter EMA SNR，复用 step gradient，但仍需维护大张量 EMA。

ema_role_scalar:
  role-level scalar EMA SNR，只维护每个 role 的 gradient-norm time-series。
  per_param_SNR_histogram = not_measured_role_scalar_ema
  active_fraction_scope = smooth_role_fraction
  SNR_subsample_stride = 12
```

代码检查：

```bash
python -m py_compile DG-LCA/dgkan/functional/snr_gated_lq.py DG-LCA/experiments/run_v929_snr_gated_functional_update_mainline.py
```

已通过。

正式 rerun：

```bash
python DG-LCA/experiments/run_v929_snr_gated_functional_update_mainline.py \
  --out-dir results/real_rerun_20260506/v929_snr_gated_functional_update_ema_role_scalar_p1cal_20260509T184500Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --lr 0.0005 \
  --p1-estimator ema_role_scalar \
  --p1-candidates LQ0,LQ1 \
  --p1-datasets MNIST,Fashion-MNIST,KMNIST \
  --p1-seeds 0,1,2 \
  --p1-steps 96 \
  --p1-ema-warmup-steps 36 \
  --p1-snr-stride 12 \
  --p1-ema-beta 0.97 \
  --snr-tau1 20 \
  --snr-tau2 40 \
  --snr-smooth-temperature 1 \
  --p1-train-size 9984
```

## 10. Updated route

`route_decision.json`：

```json
{
  "route": "R4-SNRGateNotUseful",
  "base_candidate": "LQ-t2-h256",
  "p4_pass": 1,
  "p5_near_pass": 1,
  "p5_full_pass": 0,
  "p1_snr_row_count": 270,
  "p1_snr_stability_pass": 1,
  "p1_snr_overhead_pass": 1,
  "p1_snr_active_fraction_pass": 1,
  "p1_snr_pass": 1,
  "mean_snr_overhead_ratio": 0.12669562642136065,
  "max_snr_overhead_ratio": 0.14291432741990004,
  "min_active_fraction_tau1": 0.05109928175806999,
  "max_active_fraction_tau1": 0.298427939414978,
  "max_snr_role_cv": 0.20676330730044976,
  "primary_blocker": "P2_to_P6_not_executed_after_P1_pass_in_this_runner",
  "next_required_implementation": "open_P2_one_step_population_risk_audit",
  "success_v929_p1_snr_gate": 1,
  "success_v929_functional_opened": 0,
  "success_v929_functional_advantage": 0,
  "success_v929_external_ready": 0
}
```

判断：

1. P1 的三个 gate 已关闭：overhead、active fraction、stability 全部 pass。
2. 这不是 functional success：P2 one-step audit、P3 short-run、P4 full re-entry 都仍是 `not_run`。
3. route 写为 `R4-SNRGateNotUseful` 是因为当前 runner 到 P1 为止，尚未证明 SNR gate 对 holdout / population-risk 有用。

## 11. Low-cost P1 result

P1 artifact：

```text
snr_instrumentation_v929.csv
```

设置：

```text
estimator = ema_role_scalar
ema_beta = 0.97
ema_warmup_steps = 36
SNR_subsample_stride = 12
tau1 = 20
tau2 = 40
smooth_temperature = 1
active_fraction_scope = smooth_role_fraction
per_param_SNR_histogram = not_measured_role_scalar_ema
```

总体：

```text
eligible rows = 270
mean amortized overhead ratio = 0.126696
max amortized overhead ratio = 0.142914
max raw event overhead ratio = 1.7150
active fraction tau1 range = 0.051099 - 0.298428
max role CV = 0.206763
```

按 candidate：

| candidate | rows | max CV | mean overhead | active tau1 range | raw event overhead max |
|---|---:|---:|---:|---:|---:|
| `LQ0-LQ-t2-h256-AdamW` | 135 | `0.1761` | `0.1273` | `0.0511-0.2600` | `1.7150` |
| `LQ1-LQ-t2-h256-fanin-output-scale` | 135 | `0.2068` | `0.1261` | `0.0645-0.2984` | `1.6039` |

按 dataset：

| dataset | rows | max CV | mean overhead | active tau1 range | raw event overhead max |
|---|---:|---:|---:|---:|---:|
| MNIST | 90 | `0.1864` | `0.1264` | `0.0511-0.1640` | `1.5670` |
| Fashion-MNIST | 90 | `0.1874` | `0.1257` | `0.0999-0.1855` | `1.6008` |
| KMNIST | 90 | `0.2068` | `0.1280` | `0.1060-0.2984` | `1.7150` |

按 role：

| role | rows | max CV | mean overhead | active tau1 range |
|---|---:|---:|---:|---:|
| `lift_identity` | 90 | `0.1681` | `0.1267` | `0.0511-0.2984` |
| `output_linear` | 90 | `0.1874` | `0.1267` | `0.0511-0.2984` |
| `quadratic_coeff` | 90 | `0.2068` | `0.1267` | `0.0511-0.2984` |

判断：

1. 相比 ghost-batch，`ema_role_scalar` 把 mean overhead 从 `4.061492` 降到 `0.126696`。
2. `quadratic_coeff` 的稳定性也从 max CV `0.864941` 降到 `0.206763`。
3. 这是低开销 SNR estimator 的真实修复，但它牺牲了 per-parameter SNR：本轮只支持 role-level scalar gate。
4. 因此下一步 P2 只能验证 role-level SNR gate 的 population-risk usefulness，不能把它说成 per-parameter SNR gate。

## 12. No-fake audit / hash / 更新结论

No-fake audit：

```text
rows_checked = 441
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

关键 hash：

| artifact | SHA256 |
|---|---|
| v9.2.9 plan | `51b0679dc36847a7a4a1154129f60059328bfef35adad618e408cf8e84b3fba5` |
| `dgkan/functional/snr_gated_lq.py` | `15ed445d8cd272f0a4ae0ba20d065404dc048e4cc077138ba831a1bb09cf7489` |
| `experiments/run_v929_snr_gated_functional_update_mainline.py` | `1c6782eb546f1e7c33d80492d9a59d7baf4c8107070c6d3ff2a0c9afc163e8df` |
| route | `ff17c4ec051d578d73a55027502345032fd8aa75dde45991b376ef7f2e0b6f5b` |
| `snr_instrumentation_v929.csv` | `9b7f9737bf9a4aada8f2d7304e2bb26ddd11ffdcef2bdfe2ac7bd6e9fbcab7c0` |
| provenance audit | `37acf75bff1860b1d39dcc6ce7d3fffe1be19c3a90755651fcb8ecb238552938` |

更新结论：

```text
P0 functional diagnostic gate = pass
P1 ghost-batch SNR = fail by overhead
P1 ema_role_scalar SNR = pass
P2 one-step population-risk audit = not_run
P3 short-run functional safety = not_run
P4 full functional re-entry = not_run
route = R4-SNRGateNotUseful
```

机制结论：

1. 低开销 SNR estimator 已经闭合 P1：role-level scalar EMA + stride amortization 可以进入系统 envelope。
2. 但这个 estimator 是 role-level，不是 per-parameter；不能把它外推成 F6 ParamSNRGatedFunctional。
3. 低开销 P1 pass 后，真正的下一关是 P2：验证 SNR gate 是否预测 holdout loss delta，并且 task-safe projection 是否必要。
4. 在 P2 之前仍不能执行 P3/P4 functional training，也不能声明 functional advantage。

最终一句话：

> v9.2.9 追加低开销 SNR estimator 后，P1 已被推进：`ema_role_scalar` 的 amortized overhead `0.126696`、active fraction `0.051099-0.298428`、max CV `0.206763` 均过 gate。但 functional update 仍未打开；下一步必须执行 P2 one-step population-risk audit。

## 13. 追加：P2 one-step population-risk audit

根据第 12 节结论，本轮继续打开 P2。为避免 runner 继续膨胀，新增的 functional/SNR 数学工具放在：

| 文件 | 改动 |
|---|---|
| `dgkan/functional/snr_gated_lq.py` | 新增 role-scalar EMA 从已计算梯度更新、functional step norm/dot/projection、quadratic coeff direction、task-safe projection helper |
| `experiments/run_v929_snr_gated_functional_update_mainline.py` | 新增 P2 编排、holdout before/after audit、SNR-gated / no-SNR / noop / random controls、route/failure 更新 |

P2 协议：

```text
train microbatch = 64
holdout microbatch = 64
P2 measured steps = 8
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates = LQ0,LQ1
functional step norm = 0.10 * AdamW-equivalent task step norm
P2 SNR EMA warmup = 36 normal AdamW steps before measured one-step audit
loss_type = CE
label_smoothing = 0
teacher/distillation/geometry_loss/sampler/class_weight/cpu_offload = 0
uses_loss_backward = 0
fake/proxy = 0
```

重要实现边界：

1. P2 使用 warmed `ema_role_scalar`，不是 cold-start SNR；否则第一步 SNR 会虚高，不符合第 11 节 P1 calibrated gate。
2. Holdout microbatch 只用于 one-step causality audit，不用于选择 validation/test success。
3. P3-P6 只有在 P2 safety/usefulness gate 通过后才允许打开。

代码检查：

```bash
python -m py_compile experiments/run_v929_snr_gated_functional_update_mainline.py dgkan/functional/snr_gated_lq.py
```

已通过。

正式 rerun：

```bash
python experiments/run_v929_snr_gated_functional_update_mainline.py \
  --out-dir results/real_rerun_20260506/v929_snr_gated_functional_update_p2_warmema_20260509T193500Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --lr 0.0005 \
  --p1-estimator ema_role_scalar \
  --p1-ema-beta 0.97 \
  --p1-ema-warmup-steps 36 \
  --p1-snr-stride 12 \
  --p1-steps 96 \
  --snr-tau1 20.0 \
  --snr-tau2 40.0 \
  --snr-smooth-temperature 1.0 \
  --p1-candidates LQ0,LQ1 \
  --p1-datasets MNIST,Fashion-MNIST,KMNIST \
  --p1-seeds 0,1,2 \
  --run-p2-one-step \
  --p2-candidates LQ0,LQ1 \
  --p2-datasets MNIST,Fashion-MNIST,KMNIST \
  --p2-seeds 0,1,2 \
  --p2-steps 8 \
  --p2-snr-warmup-steps 36 \
  --p2-microbatch-size 64 \
  --p2-functional-step-fraction 0.10
```

## 14. Updated route

`route_decision.json`：

```json
{
  "route": "R4-SNRGateNotUseful",
  "base_candidate": "LQ-t2-h256",
  "p1_snr_pass": 1,
  "p2_one_step_row_count": 576,
  "p2_snr_row_count": 144,
  "p2_prediction_corr": 0.23108980828056389,
  "p2_prediction_corr_pass": 0,
  "p2_bad_step_rate": 0.8194444444444444,
  "p2_bad_step_pass": 0,
  "p2_holdout_nonharm_rate": 0.18055555555555555,
  "p2_holdout_nonharm_pass": 0,
  "p2_snr_gate_usefulness_pass": 1,
  "p2_one_step_population_risk_pass": 0,
  "primary_blocker": "P2_predicted_population_improvement_did_not_correlate_with_holdout_delta",
  "next_required_implementation": "repair_functional_predictor_or_SNR_gate_before_short_run",
  "success_v929_functional_opened": 0,
  "success_v929_functional_advantage": 0,
  "success_v929_external_ready": 0
}
```

判断：

1. P1 低开销 SNR 复现通过，本轮不是退回 P1 blocker。
2. P2 没过：相关性、bad-step、non-harm 三个 safety/population-risk gate 均未达标。
3. SNR-gated functional 比 no-SNR geometry control 略好，因此 failure 不是 “SNR 完全无信号”，而是信号不足以安全打开 short-run functional。

## 15. P2 result

P1 recap：

```text
eligible rows = 270
mean amortized overhead = 0.125983
max amortized overhead = 0.130215
active fraction tau1 range = 0.051099 - 0.298428
max role CV = 0.206763
P1 pass = 1
```

P2 mode summary：

| mode | rows | bad-step rate | holdout non-harm | mean holdout delta | prediction corr |
|---|---:|---:|---:|---:|---:|
| `F1-SNRGatedTaskProjected` | 144 | `0.819444` | `0.180556` | `+8.309467e-07` | `0.231090` |
| `C1-GeometryOnlyNoSNRTaskProjected` | 144 | `0.840278` | `0.159722` | `+2.406538e-06` | `0.227809` |
| `C2-RandomMatchedNormTaskProjected` | 144 | `0.465278` | `0.534722` | `+3.725290e-09` | `0.035449` |
| `C0-NoOp` | 144 | `0.000000` | `1.000000` | `0.000000` | `0.000000` |

P2 SNR-gated by dataset：

| dataset | rows | bad-step rate | holdout non-harm | mean holdout delta | prediction corr |
|---|---:|---:|---:|---:|---:|
| MNIST | 48 | `0.979167` | `0.020833` | `+7.164975e-07` | `0.123672` |
| Fashion-MNIST | 48 | `0.583333` | `0.416667` | `+2.359351e-08` | `0.134805` |
| KMNIST | 48 | `0.895833` | `0.104167` | `+1.752749e-06` | `0.081764` |

Gate 判断：

| gate | threshold | observed | pass |
|---|---:|---:|---:|
| prediction corr | `>= 0.30` | `0.231090` | 0 |
| bad-step rate | `<= 0.05` | `0.819444` | 0 |
| holdout non-harm fraction | `>= 0.70` | `0.180556` | 0 |
| SNR vs no-SNR usefulness | beat no-SNR on bad-step or delta | delta better: `8.31e-07 < 2.41e-06` | 1 |

判断：

1. Role-level SNR gate 有微弱 usefulness：mean holdout delta 比 no-SNR geometry control 更小，bad-step rate 也略低。
2. 但这个 improvement 远远不足以打开 P3；`0.819444` bad-step rate 说明当前 functional direction / projection / SNR gate 组合不安全。
3. NoOp control 保持 `0` delta，证明 P2 audit 没有把未更新状态误记为 improvement。
4. Random matched-norm control 的 bad-step rate 低于当前 geometry direction，提示当前 quadratic coeff damping direction 本身可能不是合适的 population-risk direction。

## 16. No-fake audit / hash / final conclusion

No-fake audit：

```text
rows_checked = 1016
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

关键 hash：

| artifact | SHA256 |
|---|---|
| v9.2.9 plan | `51b0679dc36847a7a4a1154129f60059328bfef35adad618e408cf8e84b3fba5` |
| `dgkan/functional/snr_gated_lq.py` | `cd04043742c9e82e5e6a0bbb9d71010133bcd4c7753707cbf28e202087295c5e` |
| `dgkan/models/fc_purekan_lq.py` | `feb2bb205b0ad11155bca1c74d9f938d60cd6821c38a8c99a30c8db067669f8c` |
| `experiments/run_v929_snr_gated_functional_update_mainline.py` | `e76ca4ace0e54e9f8245081d39ffc7aa75e007e8a522383222c5ef0b4cf8e5df` |
| route | `1140b8e10b9e59138c99888383a5c57c079984984244c91f100f2cc7c4dc3de9` |
| `snr_instrumentation_v929.csv` | `4c569bf60cfbbe7ed48ed3e2bd0229ea74b86d584bb7d714815198d1a0342a4b` |
| `one_step_population_risk_audit_v929.csv` | `005585a82529ede010a6d943fecc351b4646ec48bba9291911bfa73d6d94ae41` |
| provenance audit | `71c9d930e73a73cbb373e880c9ccc12e607448c7c2f5091762a105f0fd8668b1` |

最终结论：

```text
P0 functional diagnostic gate = pass
P1 low-cost role-scalar SNR = pass
P2 one-step population-risk audit = fail
P3 short-run functional safety = not_run
P4 full functional re-entry = not_run
P5 strong baseline challenge = not_run
P6 robustness = not_run
route = R4-SNRGateNotUseful
success_v929_functional_opened = false
```

机制结论：

1. v9.2.9 已经解决 “SNR estimator 是否太贵” 的问题：role-scalar EMA 在 stride amortization 下通过 P1。
2. 但 warmed P2 显示，当前 SNR gate + quadratic coeff damping direction 并不具备足够 one-step population-risk safety。
3. task-safe projection 只能保证 train microbatch first-order CE 方向，不保证 holdout；这正是 P2 暴露的问题。
4. 下一步不能打开 P3/P4 functional training，应先修 functional predictor / role gate / direction：例如更保守的 holdout-aware calibration、只在高置信 SNR event 更新、或重新定义 geometry direction，而不是继续增大 functional strength。

最终一句话：

> v9.2.9 已推进到 P2 并诚实停下：低开销 SNR gate 通过 P1，但 one-step holdout audit 未过，`F1-SNRGatedTaskProjected` 的 bad-step rate 为 `0.819444`、prediction corr 为 `0.231090`。因此 functional update 仍不能打开，P3-P6 均保持 `not_run`。
