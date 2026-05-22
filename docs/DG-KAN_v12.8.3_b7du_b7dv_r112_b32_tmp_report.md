# DG-KAN v12.8.3 B109 / 经典基函数全家族高效率化 / Functional Geometry 结果复盘

> 本复盘由 `run_v1283_b109_classic_family_functional_geometry.py` 生成。所有结论只来自本轮落盘 CSV/JSON/hash/provenance audit；没有 fake data、proxy row、CPU offload、teacher/distillation、loss modification、sampler/class weight 或 dataset-name branch。

## 0. 最新结论

```text
route = R2-B109AUCTrajectoryStillBlocked
B109_FHQ_any_official_efficiency = 1
B109_F3_official_efficiency = 1
B109_FHQ_efficiency_anchor = B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095 / F2-triton-forward-delta-readout-proj-grad-learnableP
B109_A5_autopsy_pass = 0
B109_A5_best_candidate = B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095
B109_LineC_measured = 1
B109_LineC_nontearing_pass = 0
B109_functional_reentry_measured = 1
B109_functional_beats_controls = 0
official_functional_success = 0
base_qualified = 0
functional_open = 0
next_recommended_action = manual foreach update fixed part of dense-P F3 cost, but AUC/NLL trajectory is still blocked; next try a trajectory primitive that changes learning dynamics without adding direct-grad cost, or a true lower-level projection-gradient/update fusion
artifact = results/v12_8_3_b109_classic_family_functional_geometry/v1283_b109_b257_rational_b7du_b7dv_r112_b32_20260522T004500Z
```

## 1. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `hat_wavelet`、`B5h-HatWaveletKAN-local-K4`、`B3c-ChebyKAN-K3-stream`、`B4b-FourierKAN-lowfreq-K2-stream`、`B136a-lowfreqP`、`B137a-blockfreqP`、`B138b-semiFixedP`、`B139b-pUpdateEvery4`、`B140b-pUpdateEvery2`、`B141b-warm1PUpdateEvery4`、`B142b-warm1PUpdateEvery2`、`B143b-warm2PUpdateEvery4`、`B144b-warm2PUpdateEvery2`、`B145b-activeP64`、`B146b-activeP48`、`B147b-activeP96`、`B148-B197`、`B198-B202 scheduled manual-update cross repairs`、`B203 Triton quad_proj AdamW update repair`、`B204 fused projection-gradient AdamW update repair`、`B210-B213 stop-grad logitnorm repairs` | 对齐 v12.8.3 family plan 与 B109 trajectory repair；B164/B165 将 projection 参数从 `D*H` 降为 `H` 个 scale；B166-B168 将 projection 限制为 `K*H` 个 sparse weights；B188-B190 保留 dense-P trajectory，只替换 update 为可审计 manual foreach AdamW；B194-B197 不加 direct rows，只在 quadratic projection 内做 stop-grad batch RMS / tanh bound；B198-B202 将已有 pUpdateEvery/warm pUpdateEvery 轨迹修复与 manual foreach update 组合；B203 保持 dense-P/B109 数学轨迹，只把 `quad_proj` AdamW update 下沉到 Triton 单 tensor kernel；B204 在同一 projection-gradient kernel 内直接更新 `quad_proj` 与 AdamW 状态；B210/B211 保留 B205 前向 samplewise logitnorm，但把 RMS 分母从反向图中切掉；B212/B213 改为 batch-level stop-grad logit RMS，避免逐样本 confidence 结构被单独重标定。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B205 fused-update samplewise logitnorm` | 在 B204 fused projection-gradient/update 路径上加入 sample-wise logit normalization，并固定 branch/gain 以满足 FHQ analytic backward 支持条件；不改 loss/gate/data，也不按 dataset 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B206 fused-update fixed scalar` | 隔离 B205 的改善来源：保留 B204 fused update，只固定 branch/gain，不启用昂贵 sample-wise logitnorm，检查是否能保持效率并改善 AUC trajectory。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B207 fused-update lower logitnorm` | 保留 B205 结构但把 sample-wise logitnorm target 从 `1.50` 降到 `1.00`，检验能否缓解 ECE/AUC 伤害。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B208/B209 hybrid SparseP logitnorm` | 将 projection 从 dense `D*H` 改为 hybrid local/global `K*H` sparse weights，并叠加 B205 的 logitnorm trajectory primitive，测试能否保留 task 改善同时降低 projection backward/update cost。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B210/B211 stop-grad logitnorm` | B210 用 stop-gradient RMS denominator 保留 samplewise logitnorm 前向轨迹、移除完整 logitnorm backward 的逐样本耦合项；B211 再叠加 warm2/update-every2 projection schedule。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B212/B213 batch stop-grad logitnorm` | 将 B210 的 per-sample RMS 改成 whole-batch RMS 标尺，目标是在保留低成本 backward 的同时减少 per-sample normalization 对 ECE/AUC 的伤害；B213 再叠加 warm2/update-every2 projection schedule。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 active-hidden projection-gradient Triton kernel | `B145/B146/B147` 只对指定 active P hidden columns 计算 projection gradient，其余列梯度清零；这是降低 fused backward/update cost 的模型约束，不改 loss、不用数据集分支。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 pairScaleP sparse q / scale-gradient Triton kernel | `B164/B165` 使用固定 pair 投影方向 + 可学习 per-hidden scale；forward/backward/update 不再需要 dense `D x H` projection 参数。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 sparseK-P sparse q / sparse-weight-gradient Triton kernel | `B166-B168` 每个 projection hidden 只学习 `K` 个固定位置权重，目标是在 PairScaleP 崩塌后保留 single-kernel-friendly 低参数量，同时恢复比 scale-only 更强的 trajectory capacity。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 尝试修复 `Q_TANH + Q_RMS` projection-gradient chain rule，并 guard official support | B196 暴露 fused backward correctness failure；先把 tanh 导数改为使用 RMS 归一化之前的 `tanh(q)`。若 correctness 仍不足，`rmsQ+boundQ` 会显式要求 two-pass q transform kernel，不允许冒充 official。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 narrow Triton AdamW update kernel | B203 仅对 dense `quad_proj` 使用 lower-level AdamW update kernel；其他参数继续用 manual foreach AdamW，便于隔离 `quad_proj` update cost。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 fused projection-gradient + AdamW update kernel | B204 在 projection-gradient accumulation 后直接更新 dense `quad_proj`/AdamW state，不再写回 `quad_proj.grad` 后另跑 update kernel。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 stop-grad logitnorm effective-delta branch | 让 FHQ fused backward/update 与 B210-B213 的 autograd 语义一致，确保 correctness audit 比对同一个数学 primitive。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 v12.8.3 runner，并支持 `--repair-candidate-ids` 多 B109 候选 autopsy 与 full-step profile；支持 projection-only LR smoothing / single-stage and multi-stage schedule；新增 classic family L3 analytic manual CE step measurement | 写 `v1283_*` artifacts；区分 official FHQ、L1 autograd、L2 compile-forward、L3 manual analytic 探索；不把探索写成 official。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 `B109_FHQ_any_official_efficiency` 与 `B109_FHQ_efficiency_anchor` route 字段 | 区分“任一 B109-family FHQ path 过 efficiency”和“learnable-P F3 path 过 efficiency”，避免误读 fixed-P pass。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 修复 scheduled projection-gradient selector 的 regex | 旧写法匹配字面量 `\d`，导致 `pUpdateEvery*` 变体未真正切到 F4；修复后重新落盘结果。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 B109 Line C / Functional re-entry 输出 | 复用 v12.5.2 的 coupling/signal/noise 与 cloned functional strong-control audit，重写为 `v1283_*` artifact；base 开门后仍只按真实 control/task-safe/overhead 证据决定是否 official。 |

## 2. B109 FHQ Full-Step

| candidate_id | implementation_id | step_ratio_q90 | memory_ratio_q90 | B109_special_step_target_pass | B109_special_memory_target_pass | official_efficiency_pass |
| --- | --- | --- | --- | --- | --- | --- |
| `B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095` | `B48a-existing-manual-ce-torch-reduction` | `1.9460976880847014` | `2.719047619047619` | `0` | `0` | `0` |
| `B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095` | `F2-triton-forward-delta-readout-proj-grad-learnableP` | `1.0751843085738877` | `0.5961904761904762` | `0` | `1` | `1` |
| `B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `1.0397783446747146` | `0.1295238095238095` | `0` | `1` | `1` |
| `B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F1-triton-forward-delta-readout-grad-fixedP` | `1.1131460094840684` | `0.13` | `0` | `1` | `0` |
| `B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F4-triton-workspace-forward-delta-readout-grad-fixedP` | `1.1453415437711363` | `0.13` | `0` | `1` | `0` |

## 3. B109 AUC Autopsy

| candidate_id | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok | A5_autopsy_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095` | `0.012369791666666666` | `-0.01953125` | `0.8333333333333334` | `1` | `0` | `1` | `0` |
| `B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `-0.265625` | `-0.36328125` | `0.0` | `0` | `0` | `0` | `0` |

解释：这是本轮真实重跑的短程 autopsy，不继承旧文档数值；若 epochs/seeds/train-size 小于 official protocol，则只能作为 trajectory diagnostic，不写成 official base success。

## 4. 经典基函数家族状态

| family | status | best_L1_step_ratio | best_L3_manual_step_ratio | L2_forward_attempt_measured | L3_manual_analytic_attempt_measured | blocker | official_family_failure_claim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BSpline` | `KernelBlocked` | `2.929218335021113` | `1.8374881445492977` | `True` | `True` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` | `False` |
| `Chebyshev` | `ExpressionBlocked` | `2.10183085153134` | `1.0875685359895506` | `True` | `True` | `family-specific fused L3 efficiency exists; A4/A5 official family promotion not run in this closed follow-up runner` | `False` |
| `Fourier` | `ExpressionBlocked` | `2.0819947972204687` | `1.2383296223858553` | `True` | `True` | `family-specific fused L3 efficiency exists; A4/A5 official family promotion not run in this closed follow-up runner` | `False` |
| `RBF` | `KernelBlocked` | `2.5301619333623053` | `1.7871126931478036` | `True` | `True` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` | `False` |
| `Rational` | `KernelBlocked` | `1.5653128111230044` | `1.4413205498408637` | `True` | `True` | `family-specific fused L3 kernel exists, but full-step efficiency gate failed` | `False` |
| `Wavelet` | `KernelBlocked` | `6.348538054726346` | `3.281886131572026` | `True` | `True` | `L3 manual analytic attempted; fused family-specific backward/update kernel missing` | `False` |

解释：本轮对每个 classic family 至少做了 L1 autograd full-step、L2 `torch.compile` forward attempt，并补充 L3 analytic manual CE step attempt。L3 manual step 不调用 `loss.backward()`，但仍是 torch-reduction path，不是 family-specific fused Triton/CUDA kernel，所以状态仍是 `KernelBlocked`，不是数学失败结论。

## 5. Family Efficiency 摘要

| family | candidate_id | implementation_level | step_ratio_q90 | memory_ratio_q90 | A1_exploratory_pass | official_efficiency_pass | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BSpline` | `B6r-BSpline-order1-stream-K2-repair` | `L1-vectorized-or-stream-autograd` | `2.929218335021113` | `1.48` | `0` | `0` | `measured` |
| `BSpline` | `B6r-BSpline-order1-stream-K2-repair` | `L3-analytic-manual-ce-torch-reduction` | `1.8374881445492977` | `1.6147619047619048` | `0` | `0` | `measured` |
| `Rational` | `B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.667918103583202` | `1.1728571428571428` | `0` | `0` | `measured` |
| `Rational` | `B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.4688878011934927` | `0.8633333333333333` | `0` | `0` | `measured` |
| `Rational` | `B7c-RationalKAT-flashgroup-G16-h80-linearres-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.5653128111230044` | `1.1576190476190475` | `0` | `0` | `measured` |
| `Rational` | `B7c-RationalKAT-flashgroup-G16-h80-linearres-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `2.635296659659433` | `0.8328571428571429` | `0` | `0` | `measured` |
| `Rational` | `B7d-RationalKAT-flashgroup-G16-h64-linearres-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.5964780621821184` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7d-RationalKAT-flashgroup-G16-h64-linearres-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.4413205498408637` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7e-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR64-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.7694540171125022` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7e-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR64-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6181536677806962` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7f-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR128-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.7423428931730898` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7f-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR128-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6581079086883328` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7g-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR96-tritonL3` | `L1-vectorized-or-stream-autograd` | `6.948560861097008` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7g-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR96-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.9093912243850106` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7h-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR112-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.7439037676856086` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7h-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR112-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6446887377588533` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7i-RationalKAT-flashgroup-G16-h64-linearres-paircrossR136-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.8995090735613185` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7i-RationalKAT-flashgroup-G16-h64-linearres-paircrossR136-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6308750288128613` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7j-RationalKAT-flashgroup-G16-h64-linearres-paircrossR96-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.8903420101884716` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7j-RationalKAT-flashgroup-G16-h64-linearres-paircrossR96-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6326373392730174` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7k-RationalKAT-flashgroup-G16-h64-linearres-paircrossR88-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.9722038718794197` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7k-RationalKAT-flashgroup-G16-h64-linearres-paircrossR88-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6612995809888371` | `0.8176190476190476` | `0` | `0` | `measured` |
| `Rational` | `B7l-RationalKAT-flashgroup-G16-h64-linearres-paircrossR80-tritonL3` | `L1-vectorized-or-stream-autograd` | `1.9625184819006447` | `1.15` | `0` | `0` | `measured` |
| `Rational` | `B7l-RationalKAT-flashgroup-G16-h64-linearres-paircrossR80-tritonL3` | `L3-analytic-manual-ce-torch-reduction` | `1.6225507035216533` | `0.8176190476190476` | `0` | `0` | `measured` |

## 6. L2 Component Profile

| family | candidate_id | component | ratio_vs_mlp_forward_q90 | official_l2_kernel | status |
| --- | --- | --- | --- | --- | --- |
| `BSpline` | `B6r-BSpline-order1-stream-K2-repair` | `L2_torch_compile_forward_attempt` | `2.1095569148631275` | `0` | `measured` |
| `Rational` | `B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7c-RationalKAT-flashgroup-G16-h80-linearres-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7d-RationalKAT-flashgroup-G16-h64-linearres-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7e-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR64-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7f-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR128-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7g-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR96-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7h-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR112-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7i-RationalKAT-flashgroup-G16-h64-linearres-paircrossR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7j-RationalKAT-flashgroup-G16-h64-linearres-paircrossR96-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7k-RationalKAT-flashgroup-G16-h64-linearres-paircrossR88-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7l-RationalKAT-flashgroup-G16-h64-linearres-paircrossR80-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7m-RationalKAT-flashgroup-G16-h64-linearres-balancedPairR88-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7n-RationalKAT-flashgroup-G16-h64-linearres-balancedPairR96-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7o-RationalKAT-flashgroup-G16-h64-linearres-transformPairR88-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7p-RationalKAT-flashgroup-G16-h64-linearres-transformPairR96-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7q-RationalKAT-flashgroup-G16-h56-linearres-paircrossR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7r-RationalKAT-flashgroup-G16-h48-linearres-paircrossR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7s-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7t-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7u-RationalKAT-flashgroup-G16-h40-linearres-gridPairR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7v-RationalKAT-flashgroup-G16-h32-linearres-gridPairR136-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7w-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |
| `Rational` | `B7x-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-tritonL3` | `L2_torch_compile_forward_attempt` | `` | `0` | `failed` |

## 7. Provenance / Hash

| artifact | rows_checked | fake_data_used_sum | proxy_row_used_sum | cpu_offload_used_sum | no_fake_pass |
| --- | --- | --- | --- | --- | --- |
| `v1283_b109_auc_attribution.csv` | `14` | `0` | `0` | `0` | `1` |
| `v1283_b109_auc_autopsy_trace.csv` | `54` | `0` | `0` | `0` | `1` |
| `v1283_b109_fullstep_profile.csv` | `6` | `0` | `0` | `0` | `1` |
| `v1283_b109_functional_control_matrix.csv` | `4` | `0` | `0` | `0` | `1` |
| `v1283_b109_functional_direction_audit.csv` | `28` | `0` | `0` | `0` | `1` |
| `v1283_b109_functional_five_step.csv` | `28` | `0` | `0` | `0` | `1` |
| `v1283_b109_functional_one_step.csv` | `28` | `0` | `0` | `0` | `1` |
| `v1283_b109_functional_route.json` | `0` | `0` | `0` | `0` | `1` |
| `v1283_b109_fused_backward_correctness.csv` | `20` | `0` | `0` | `0` | `1` |
| `v1283_b109_linec_coupling.csv` | `5` | `0` | `0` | `0` | `1` |
| `v1283_b109_linec_diagnostics.csv` | `5` | `0` | `0` | `0` | `1` |
| `v1283_b109_linec_functional_reentry_summary.csv` | `1` | `0` | `0` | `0` | `1` |
| `v1283_b109_linec_noise_leak.csv` | `5` | `0` | `0` | `0` | `1` |
| `v1283_b109_linec_signal_reservoir.csv` | `5` | `0` | `0` | `0` | `1` |
| `v1283_b109_task_autopsy_final.csv` | `18` | `0` | `0` | `0` | `1` |
| `v1283_family_component_profile.csv` | `350` | `0` | `0` | `0` | `1` |
| `v1283_family_efficiency.csv` | `351` | `0` | `0` | `0` | `1` |
| `v1283_family_expression.csv` | `175` | `0` | `0` | `0` | `1` |
| `v1283_family_failure_table.csv` | `175` | `0` | `0` | `0` | `1` |
| `v1283_family_functional_diagnostic.csv` | `175` | `0` | `0` | `0` | `1` |

Selected hashes:

| artifact | sha256_prefix |
| --- | --- |
| `v1283_b109_auc_attribution.csv` | `ec19fb47e2cb` |
| `v1283_b109_auc_autopsy_trace.csv` | `bd706b45a3e3` |
| `v1283_b109_fullstep_profile.csv` | `861d0a5f1c3a` |
| `v1283_b109_functional_control_matrix.csv` | `6a9b703bc642` |
| `v1283_b109_functional_direction_audit.csv` | `4a860741bb24` |
| `v1283_b109_functional_five_step.csv` | `6ced6247b5d1` |
| `v1283_b109_functional_one_step.csv` | `421349da65df` |
| `v1283_b109_functional_route.json` | `f682a8a7ede0` |
| `v1283_b109_fused_backward_correctness.csv` | `f8803f1b00d7` |
| `v1283_b109_linec_coupling.csv` | `0d27a1b7cdc9` |
| `v1283_b109_linec_diagnostics.csv` | `6cdc4bc2a4ae` |
| `v1283_b109_linec_functional_reentry_summary.csv` | `6c7d5abc0898` |
| `v1283_b109_linec_noise_leak.csv` | `a03436cf0bab` |
| `v1283_b109_linec_signal_reservoir.csv` | `58cd77c58cad` |
| `v1283_b109_task_autopsy_final.csv` | `fbb76d89727e` |
| `v1283_family_component_profile.csv` | `965a4bd55489` |
| `v1283_family_efficiency.csv` | `d53b4791a30a` |
| `v1283_family_expression.csv` | `7f834ddb73e8` |
| `v1283_family_failure_table.csv` | `96fd9a417ac8` |
| `v1283_family_functional_diagnostic.csv` | `a3a9d7e95151` |

## 8. 分析结论

1. B109/FHQ 仍是主线 anchor；本 runner 用 FHQ path 重新测 full-step，并用短程 autopsy 重新记录 AUC-step/AUC-time，不复用旧结果冒充本轮数据。若 exact candidate 换成 B145 这类 repair diagnostic，不能把 fixed-P pass 冒充为 B109b learnable-P F3 pass。
2. 经典 family 本轮没有被判为数学失败；它们的共同 blocker 是 family-specific fused L3 backward/update kernel 缺失。L1/L2 与新增 L3 manual analytic 结果只能说明当前实现路径状态。
3. Functional official 仍关闭，除非 B109 或某个 family base 同时通过 official efficiency、A4、A5 和 Line C。
4. B136/B137 fixed projection 明显降低 kernel/update cost 但 task trajectory 崩塌；B138 freeze-after-1 与 B139-B144 scheduled projection-gradient repair 用于检验能否保留 B109 trajectory 同时降低多数 step 的 projection backward/update cost。
5. B145 activeP64 是实质 kernel-level repair，但没有完成：F2 step ratio `1.2529631891274053` 接近 1.25 gate，F3 workspace step ratio `1.3654556430637974` 未过；AUC 上 Fashion-MNIST seed 2 达到 `1.0429494332758882`，但 seed 0/1 仍为 `1.0919466232193322` / `1.0856452212776184`，A5 仍失败。F2/F3 fused backward correctness 通过；manual fallback 的 `quad_proj` grad 不适合作为 B145 official path。
6. 最新 B142/B143/B144 说明 warm + scheduled P update 能改善部分 Fashion-MNIST AUC，但不能打开 A5：B144 mean delta `0.003689236111111111`、worst `-0.01953125`、near `0.7777777777777778`、ECE ok，但 AUC-step/AUC-time 仍失败；B109b 仍是 best candidate。
7. B146/B147 继续检验 active projection hidden 的 Pareto 中间点：B146 activeP48 偏效率，B147 activeP96 偏保留 trajectory 容量。若没有同时改善 FHQ step 与 AUC/accuracy，不能写成完成。
8. B148-B151 继续检验完整 P 容量下的 projection-only LR smoothing：只降低 `quad_proj` 参数组学习率，不改 loss、batch、标签权重或数据集分支；B148/B149 是强平滑，B150/B151 是中间点。
9. B152-B159 继续检验 projection LR schedule：前 1 或 2 epoch 使用更高 P LR，之后降到 0.50/0.55/0.60；B158/B159 进一步做 `0.75 -> 0.60 -> 0.50` 两段 schedule，目标是在不使用数据集分支的情况下兼顾 MNIST 早期学习、Fashion accuracy 与后期 AUC 稳定。
10. B160/B161 检验 lowFreqPUpdate-K32/K64：P 方向用低频结构初始化，但只允许前 32/64 个 projection hidden columns 更新；目标是比 fixed lowfreqP 更保留 B109 trajectory，同时实质降低 fused projection-gradient/update cost。
11. B162/B163 检验 blockfreqPUpdate-K64/K96：P 方向用局部块频率结构初始化，同时只更新前 64/96 个 projection hidden columns；这是 lowfreq active 崩塌后的局部化修复尝试，仍不使用数据集分支。
12. B164/B165 检验 pairScaleP：固定 sparse pair projection directions，只学习每个 projection hidden 的 scale；这是 single-kernel-friendly trajectory primitive，目标是实质降低 projection backward/update 参数量。
13. B166-B168 检验 sparseK-P：PairScaleP 若过效率但 task collapse，则改为每个 hidden 只学习 `K=8/16` 个固定 sparse input weights，并对该 sparse projection 参数组使用 projection-only LR smoothing；这仍是 single-kernel-friendly，不回到 dense random pair trajectory。
14. B169-B171 检验 group-abs tail-local signal repair：SparseK-P 若保留效率但 task collapse，则回到 dense learnable-P trajectory，同时只在 direct branch 加少量 fixed group absolute-value signal rows，目标是修 Fashion-MNIST AUC-step/tail NLL，不改 loss、不使用数据集分支。
15. B172-B175 交叉检验 group-abs 与 projection LR schedule：B169 保留较好的 worst row 但 AUC 失败，B171/B155 过 AUC 但 worst/near 仍失败；因此测试 `groupabs4/8` + `projlr075->055/060 after epoch1`，试图同时保留 tail signal 和后期 trajectory 稳定性。
16. B176-B179 检验 group-abs tail signal strength：只把 group-abs direct rows 的初始化倍率改为 `0.25` 或 `0.75`，并分别测试无 schedule 与 `projlr075->050 after epoch1`；这是为了确认 B169 的 tail 修复是否只是强度没调到位。
17. B180-B183 检验 block-local SparseP：继续使用 SparseK fused q / sparse-weight-gradient kernel，但每个 projection hidden 只看一个局部图像块，而不是随机散点；B180/B181 用 `K=8`，B182 用 `K=12`，B183 用更低 projection LR。目标是在实质降低 projection backward/update 参数量的同时，比 B166-B168 随机 SparseK 更保留任务轨迹。
18. B184-B187 检验 hybrid SparseP：block-local SparseP 如果因为缺少全局输入耦合而崩塌，则每个 hidden 使用一半局部块位置、一半全局分散位置；仍然复用 single-kernel SparseP path，并保持 `K*H` projection 参数量。
19. B188-B190 检验 dense-P manual foreach AdamW：不再压缩 projection 表达力，只把 update path 从 `torch.optim.AdamW.step()` 替换成同公式的 foreach update engine，验证是否能实质降低 F3 backward/update cost。
20. B191-B193 检验 norm/mean statistic direct rows：保留 dense-P 与 manual foreach update，只增加 1 到 2 个全局统计信号，测试是否能低成本改善 NLL/AUC trajectory。
21. B194-B197 检验 q-normalized/bounded trajectory primitive：保留 dense-P 与 manual foreach update，不增加 direct branch rows，只在 quadratic hidden `q` 内部做 stop-grad batch RMS 和/或 tanh bound，测试是否能改变 NLL/AUC 轨迹而不增加 direct-grad 成本。
22. B198-B202 检验 scheduled projection update + manual foreach update cross repair：保留 B139-B144 的 warm / update-every-N 轨迹设定，同时使用已审计的 manual foreach AdamW update，测试“早期 dense-P 学习信号 + 多数 step 固定 P + 低 update cost”是否能同时改善 AUC 和 full-step。
23. B203 检验 Triton `quad_proj` AdamW update repair：保持 B189/B109 dense-P trajectory 与 manual foreach 其余参数 update，只把最大的 dense projection 参数 update 下沉到 single-tensor Triton kernel，测试是否能进一步实质降低 fused backward/update cost。
24. B204 检验 fused projection-gradient + AdamW update repair：保持 B189/B109 dense-P trajectory，但在 projection-gradient kernel 内直接更新 `quad_proj`，测试是否能同时省掉 `quad_proj.grad` 写回和单独 update pass。
25. B205 检验 fused-update + sample-wise logitnorm trajectory repair：不增加 direct-gradient rows，不改 loss，只用固定 branch/gain 的 per-sample logit normalization 尝试改善 NLL/AUC trajectory，并保持 B204 的 fused `quad_proj` update。
26. B206 检验 fused-update + fixed scalar trajectory repair：保留 B204 fused `quad_proj` update，只固定 branch/gain，确认 B205 的 task 改善是否来自固定 scalar dynamics，还是来自昂贵 logitnorm。
27. B207 检验 lower-target logitnorm trajectory repair：若 B205 accuracy 改善但 ECE/AUC 失败，则降低 logitnorm target 检验校准/trajectory trade-off。
28. B208/B209 检验 hybrid SparseP + logitnorm trajectory repair：把 projection 参数从 dense `D*H` 降到 `K*H`，用 local/global sparse receptive fields 尝试抵消 B205 的 logitnorm 额外 backward cost，并检查 task trajectory 是否崩塌。
29. B210/B211 检验 stop-gradient logitnorm trajectory repair：B205 的完整可微 logitnorm 改善 accuracy 但太慢且 ECE/AUC 失败；B210 保留前向 normalization、把 RMS denominator 视为常量反传，B211 再叠加 warm2/update-every2 projection schedule，目标是实质降低 fused backward/update cost 而不是继续调 target。
30. B212/B213 检验 batch-level stop-gradient logitnorm trajectory repair：若 B210/B211 证明 per-sample normalization 仍伤 ECE/AUC，则改为 batch-wide RMS 标尺，保留单 scalar backward scale，避免逐样本重标定 confidence 结构。
31. 当前 blocker 已收窄到 Fashion-MNIST NLL/AUC trajectory：不能用 dataset-specific branch 修；下一步应把工程投入放在 classic family L3 kernels，或更细粒度的 block-sparse / fused projection-gradient accumulation；不应靠 dense random pair trajectory 或继续温度/epoch小修。
