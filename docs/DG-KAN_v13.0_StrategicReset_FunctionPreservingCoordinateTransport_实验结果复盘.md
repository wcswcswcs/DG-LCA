# DG-KAN v13.0 StrategicReset FunctionPreservingCoordinateTransport 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/smoke/skipped row 写成 promotion。

## 1. 计划理解

v13.0 是 strategic reset，不继续扩同类 B-RAT / B-FOU / M-J token 网格，而是把 functional update 改成：

```text
function-preserving coordinate transport + future training probe
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 label-informed initialization。
3. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 dataset-name branch。
4. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
5. smoke/diagnostic/skipped row 不能写成 promotion。
6. blocker 后必须按文档 fallback：ridge compensation、smaller transport step、readout-only refit、MLP analog、future probe controls。
```

本轮实现明确区分：

```text
1. synthetic controlled feature-coordinate transport。
2. real-data frozen readout-feature transport。
3. full basis-parameter coordinate surgery。
```

当前新增 runner 只覆盖 1 和 2；如果真实结果有信号，也不能写成 full basis surgery 或 official S5，除非 artifact 明确显示 `full_basis_param_transport_executed=1`。

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v130_function_preserving_transport.py
```

核心内容：

```text
1. Line S：读取 v12.35 substrate artifact，按 v13.0 S1/S2 gate 重算 `v130_substrate_gate_v2.csv`。
2. Line X：执行 X1..X7 synthetic tasks 的 feature-coordinate transport proof。
3. Line G：对 S1 substrate 的真实数据 frozen readout features 执行 function preservation / future probe。
4. Line M：执行 MLP hidden-feature analog transport。
5. Line Z：输出 route、failure table、next hypothesis、figures、manifest、code review packet。
```

审计说明：

```text
1. transport direction 只读取 features 与 current logits。
2. labels 只用于 checkpoint/future training probe，不进入 transport proposal。
3. 当前真实数据 Line G 是 readout-feature coordinate transport，不是完整 basis 参数 transport。
4. 因此即使 feature transport 有 pass，也必须检查 MLP analog；若 MLP analog 同样有效，则 `KAN_specific_claim_allowed=0`。
```

## 3. 初始 smoke blocker 与修复

首次 smoke 在真实 KAN frozen feature 的 function-preserving compensation 阶段失败：

```text
torch.linalg.solve: The solver failed because the input matrix is singular.
```

按 v13.0 文档的 fallback “增大 ridge compensation / more stable compensation solve”，修改：

```text
experiments/run_v130_function_preserving_transport.py
```

具体修复：

```text
1. ridge_solve 增加 feature/logit 的 nan/inf guard。
2. ridge compensation 依次尝试 scale = 1,10,100,1000。
3. 如果 solve 仍失败，fallback 到 pinv/lstsq。
```

审计判断：

```text
这是数值补偿修复，不改变任何 v13.0 gate；
不使用 label/CE/NLL/ECE/CEp99 作为方向源；
只影响 function-preserving readout compensation 的可执行性。
```

第二次 smoke 在 code packet 打包阶段失败：

```text
relative path / absolute ROOT mismatch
```

修复：

```text
experiments/run_v130_function_preserving_transport.py
  out_dir/source_dir 统一 resolve。
```

审计判断：这是 artifact packaging 修复，不改变 gate，不改变任何 metric。

修复后 smoke 结果：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
substrate_s1_count = 14
synthetic_rows = 4
synthetic_mechanism_pass_count = 0
future_training_probe_rows = 40
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
```

解释：smoke 只证明 runner / artifact path 可运行，不能 promotion。

## 4. official v13.0 执行结果

执行命令：

```text
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

最终 route：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 14
substrate_s2_healthy_base_count = 0
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
generic_reparameterization_effect = 0
KAN_specific_claim_allowed = 0
full_basis_param_transport_executed = 0
provenance_violation_count = 0
```

重要边界：

```text
1. 本轮没有达到 S5，也没有达到 synthetic mechanism proof。
2. 只达到 S1-EfficientSubstrate；S1 全部来自 D-RAT。
3. S2 healthy base 为 0。
4. full_basis_param_transport_executed=0，因此不能声称完成完整 basis-parameter coordinate surgery。
5. KAN_specific_claim_allowed=0，不能声称 KAN-specific functional mechanism 成立。
```

## 5. v13.0 substrate gate v2

来源 artifact：

```text
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_substrate_gate_v2.csv
```

按 family 聚合：

| family | rows | S1 efficient substrate | S2 healthy base |
|---|---:|---:|---:|
| D-CHE | 8 | 0 | 0 |
| D-FOU | 8 | 0 | 0 |
| D-RAT | 16 | 14 | 0 |
| D-RBF | 6 | 0 | 0 |
| D-WAV | 6 | 0 | 0 |

解释：

```text
1. v13.0 重新套用更严格的 substrate gate 后，仍只有 Rational 进入 efficient substrate。
2. 没有任何 family 进入 S2 healthy base。
3. Non-RAT 仍停在 substrate health 之前，不能直接进入 functional promotion。
```

## 6. synthetic controlled mechanism proof

来源 artifact：

```text
v130_function_preservation.csv
v130_future_training_probe.csv
v130_synthetic_mechanism_proof.csv
```

执行规模：

```text
families = D-RAT,D-CHE,D-FOU,D-RBF,D-WAV,MLP
synthetic_tasks = X1..X7
seeds = 0
synthetic_mechanism_rows = 42
future_probe_rows_total = 528
```

核心结果：

```text
synthetic_mechanism_pass_count = 0
future_training_probe_pass_rows = 0
```

FamilyTransport function-preservation accept：

| scope | FamilyTransport rows | accepted |
|---|---:|---:|
| synthetic controlled feature space | 42 | 0 |
| real basis frozen readout feature transport | 2 | 0 |
| real MLP hidden feature analog | 1 | 0 |

说明：

```text
NoOp / RandomMatchedTransport / ReadoutOnlyRefit controls 中存在 function-preservation accepted rows；
但真正要验证的 FamilyTransport accepted=0。
因此不能把 control rows 或 identity rows 写成 mechanism pass。
```

最典型 failure pattern：

| scope | family | task | transport | GeomGain | Drift_B | Drift_Q | TangentCondition ratio | accepted |
|---|---|---|---|---:|---:|---:|---:|---:|
| synthetic | D-RAT | X2 | FamilyTransport | 9.58894181982856 | 0.6251489520072937 | 0.6043727397918701 | 0.0010595900037988937 | 0 |
| synthetic | MLP | X6 | FamilyTransport | 9.586332072871386 | 0.8375203013420105 | 0.8399123549461365 | 0.0010083128093220125 | 0 |
| synthetic | D-FOU | X7 | FamilyTransport | 9.543361817099289 | 0.5758734941482544 | 0.527223527431488 | 0.0010737818222386264 | 0 |

解释：

```text
transport 可以显著改善 tangent condition / geometry proxy；
但 function drift 大，FamilyTransport 没有通过 function preservation gate。
这正是 v13.0 要先检验的 blocker：几何坐标搬运不能自动等价于函数保持。
```

## 7. real-data frozen readout-feature transport / MLP analog

真实数据执行对象：

```text
dataset = MNIST
real_basis candidates = D-RAT26-TangentTrustRegionNoCE, D-RAT38-GroupDiversityTransportNoCE
real MLP analog = hidden feature coordinate transport
checkpoint_epochs = 2
future_steps = 50,200
```

结果：

```text
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
generic_reparameterization_effect = 0
KAN_specific_claim_allowed = 0
```

最接近但不能 pass 的 future probe rows：

| scope | family | candidate | K | future proxy | noop proxy | random proxy | ControlGap | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| real basis | D-RAT | D-RAT26 | 200 | 0.008519411462204972 | 0.390074478178694 | 0.3780730194015128 | 0.3695536079393078 | 0 |
| real basis | D-RAT | D-RAT26 | 50 | 0.008270547852003934 | 0.4365968852349128 | 0.37490544612620047 | 0.36663489827419654 | 0 |
| synthetic | D-WAV | synthetic X7 | 1000 | 0.09281829214344921 | 0.4336109480541851 | 0.4256972915477569 | 0.33287899940430765 | 0 |
| synthetic | D-RAT | synthetic X2 | 1000 | 0.04765935634688678 | 0.31399439148215047 | 0.29615276958476017 | 0.24849341323787338 | 0 |

解释：

```text
部分 future proxy 明显优于 controls；
但对应 transport 未通过 function-preservation accept gate。
按 v13.0 规则，future training benefit 不能覆盖 function drift blocker，也不能写成 promotion。
```

## 8. Family route

来源 artifact：

```text
v130_family_decision.csv
```

| family | route |
|---|---|
| D-CHE | NoEfficientSubstrate |
| D-FOU | NoEfficientSubstrate |
| D-RAT | SyntheticMechanismNoGo |
| D-RBF | NoEfficientSubstrate |
| D-WAV | NoEfficientSubstrate |
| MLP | NoEfficientSubstrate |

解释：

```text
1. D-RAT 有 efficient substrate，但 synthetic/real FamilyTransport proof 没有通过。
2. Non-RAT family 仍没有 efficient substrate。
3. MLP analog 没有通过；本轮也没有 generic reparameterization success。
```

## 9. Required artifacts

主要产物：

```text
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_route_decision.json
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_required_artifact_manifest.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_substrate_gate_v2.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_synthetic_mechanism_proof.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_function_preservation.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_future_training_probe.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_transport_controls.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_transport_proposals.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_basis_telemetry_before_after.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_mlp_analog_transport.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_linec_audit.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_failure_table.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_family_decision.csv
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_next_hypothesis_queue.md
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_code_review_packet.zip
```

Required manifest：

```text
manifest_rows = 25
missing_rows = 0
```

## 10. 最终科学结论

v13.0 没有达成 S5，也没有达成 function-preserving coordinate transport 的 synthetic mechanism proof。

已闭合事实：

```text
1. v13.0 重新审计 substrate 后，只有 D-RAT 有 S1 efficient substrate，数量为 14。
2. S2 healthy base 为 0。
3. X1..X7 synthetic controlled feature-coordinate transport 总计 42 rows，synthetic mechanism pass 为 0。
4. FamilyTransport 在 synthetic / real basis / MLP analog 三个 scope 上 accepted 都为 0。
5. future training probe 有 528 rows，pass 为 0；部分 row 有 future proxy improvement，但没有通过 function-preservation gate。
6. 当前实现没有执行 full basis-parameter coordinate surgery；route 明确记录 `full_basis_param_transport_executed=0`。
7. provenance audit 通过，required artifacts 缺失为 0。
```

新增 no-go boundary：

```text
1. 只改善 geometry proxy / tangent condition 不足以证明 function-preserving coordinate transport。
2. readout-feature transport 即使在 future probe 上偶尔优于 controls，也不能绕过 function drift。
3. 当前结果不能支持 KAN-specific functional claim。
4. 下一步若继续，不应继续扩 readout-feature transport token；需要真正的 full basis-parameter coordinate surgery hook，或回到 substrate health 修复。
```

最终合法状态：

```text
R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
```

本轮已按 v13.0 文档执行：substrate gate v2、synthetic controlled mechanism proof、ridge compensation fallback、smaller transport step / readout-only / random controls、real frozen readout-feature transport、MLP analog、future probe、required artifacts 与 provenance 审计。当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我现在不确定继续在 readout-feature transport 的局部 token 上排列变体能形成有效机制；继续推进需要新的 full basis-parameter transport 实现，而不是把当前 feature-space proxy 写成成功。

## 11. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.0 是否达成目标，若未达成则继续。本次重新读取计划文件 stop/go 段与最终 route。

计划文件 `docs/DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md` 的关键规则：

```text
进入真实数据 functional P4 必须满足：
1. S1 EfficientSubstrate pass；
2. Synthetic mechanism proof pass；
3. function-preserving transport pass；
4. future training probe pass；
5. controls pass；
6. provenance pass。

如果某 family synthetic mechanism proof fail，
或 function-preserving transport impossible，
或 future training probe repeatedly no benefit，
或 MLP analog explains all benefit，
则该 family 的 functional transport 暂停。

如果 all families fail synthetic proof，
Codex 必须停止 functional transport，回到 substrate design，不允许继续扩 token。
```

再次读取最终 `v130_route_decision.json`：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 14
substrate_s2_healthy_base_count = 0
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
generic_reparameterization_effect = 0
KAN_specific_claim_allowed = 0
full_basis_param_transport_executed = 0
provenance_violation_count = 0
manifest_rows = 25
missing_rows = 0
```

最终判断仍是：

```text
v13.0 没有达成 S5；
只达到 S1-EfficientSubstrate；
没有达成 synthetic mechanism proof；
没有达成 function-preserving transport pass；
没有达成 future training probe pass；
不允许 promotion；
合法 route 仍是 R2-SyntheticMechanismProofFailed；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV/JSON 指标、没有修改代码或 gate。原因不是轻易放弃，而是计划文件明确规定 all families fail synthetic proof 时必须停止 functional transport、不允许继续扩 token；当前 artifact 的 `synthetic_mechanism_pass_count=0/42` 正好触发该 no-go。继续推进需要新的 substrate design 或真正 full basis-parameter coordinate surgery 机制，而不是在当前 readout-feature transport proxy 上继续排列局部变体。

## 12. 用户再次追问后的 stop-contract 复核 2

用户再次要求“未达成则继续”。再次对照计划文件与最终 artifact，结论仍不变：

```text
计划文件第 754-756 行：
如果 all families fail synthetic proof，
Codex 必须停止 functional transport，回到 substrate design，
不允许继续扩 token。
```

最终 artifact：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
full_basis_param_transport_executed = 0
required_artifact_missing_count = 0
```

最终判断仍是：

```text
v13.0 没有达成 S5；
没有达成 synthetic mechanism proof；
没有达成 function-preserving transport pass；
不允许 promotion；
合法 route 仍是 R2-SyntheticMechanismProofFailed；
允许 final stop。
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。原因：继续扩当前 readout-feature transport token 与计划文件 no-go 明确冲突；同时 `full_basis_param_transport_executed=0` 已诚实记录，不能把未实现的 full basis-parameter surgery 编造成成功。继续推进需要新的 substrate design 或 full basis-parameter coordinate surgery 的机制级实现，我目前不确定如何在现有代码路径内安全完成并满足 function-preservation gate。

## 13. fallback2: affine / block-wise / scale-joint compensation

用户再次要求未达成则继续。本次重新审计 v13.0 runner 后，发现 function preservation fallback 尚未完整覆盖计划第 12 节：

```text
1. block-wise readout compensation；
2. solve readout + scale jointly；
3. local/tangent trust-region style coordinate scaling。
```

因此本次继续推进，修改：

```text
experiments/run_v130_function_preserving_transport.py
```

新增内容：

```text
1. with_bias(phi)：为 readout compensation 增加仿射偏置。
2. blockwise_ridge_solve：按 feature block 迭代拟合 residual。
3. FamilyTransport 同时尝试 global_no_bias / affine_bias / blockwise_affine。
4. FamilyTransport 增加 post_scale search，用 feature RMS penalty 选择更稳定坐标。
5. FamilyTransport 增加更小 ridge 与 alpha 组合。
```

合法性说明：

```text
1. 不改变 gate。
2. 不使用 label/CE/validation/future/dataset-name branch 构造方向。
3. transport 方向只来自 unlabeled features/current logits。
4. label 仍只用于普通 checkpoint 与 future-training probe。
```

### 13.1 fallback2 smoke

执行：

```text
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130_fallback3 --synthetic-tasks X1,X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --synthetic-feature-dim 32 --future-steps 10,20 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 10 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
```

结果：

```text
route = R2-SyntheticMechanismProofFailed
synthetic_rows = 4
synthetic_mechanism_pass_count = 0
required_artifact_missing_count = 0
```

但 function preservation 已被打开：

| scope | family/task | Drift_B | Drift_Q | TangentCondition ratio | accepted | mode |
|---|---|---:|---:|---:|---:|---|
| synthetic | D-RAT X1 | 2.055372192444338e-07 | 3.3872777294163825e-07 | 0.003404450653983507 | 1 | affine_bias |
| synthetic | D-RAT X2 | 1.6937478619638568e-07 | 2.6427758825775527e-07 | 0.0030916034457460196 | 1 | affine_bias |
| synthetic | MLP X1 | 2.295357148796029e-07 | 3.822543135356682e-07 | 0.034907113008173216 | 1 | affine_bias |
| synthetic | MLP X2 | 2.337991702461295e-07 | 3.132911956527096e-07 | 0.0011538760005148478 | 1 | affine_bias |

解释：

```text
fallback2 成功修复了原先的 function drift blocker；
smoke 仍没有 future probe pass，因此不能 promotion。
```

### 13.2 fallback2 official

正式重跑：

```text
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

最终 route 改变为：

```text
route = R3-SyntheticOnlyNoRealTransportBenefit
minimum_success = S2-SyntheticMechanismProof
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_s1_count = 14
substrate_s2_healthy_base_count = 0
synthetic_rows = 42
synthetic_mechanism_pass_count = 1
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
full_basis_param_transport_executed = 0
```

唯一 synthetic mechanism pass：

| family | task | seed | function drift pass | transport accepted | future probe pass | synthetic pass |
|---|---|---:|---:|---:|---:|---:|
| D-FOU | X4 | 0 | 1 | 1 | 1 | 1 |

FamilyTransport accepted 聚合：

| scope/family | accepted |
|---|---:|
| synthetic D-CHE | 7/7 |
| synthetic D-FOU | 7/7 |
| synthetic D-RAT | 7/7 |
| synthetic D-RBF | 7/7 |
| synthetic D-WAV | 7/7 |
| synthetic MLP | 7/7 |
| real MLP analog | 1/1 |
| real D-RAT basis feature transport | 0/2 |

Real-data blocker：

| candidate | Drift_B | Drift_Q | TangentCondition ratio | accepted |
|---|---:|---:|---:|---:|
| D-RAT38-GroupDiversityTransportNoCE | 0.08626401424407959 | 0.9629298448562622 | 1.5338226445729857 | 0 |
| D-RAT26-TangentTrustRegionNoCE | 0.00014294836728367954 | 0.9592770934104919 | 1.543149351170232 | 0 |

解释：

```text
1. fallback2 是真实进展：v13.0 从 R2 推进到 R3。
2. 但唯一 synthetic pass 属于 D-FOU，而 D-FOU 没有 S1 efficient substrate。
3. 有 S1 substrate 的 D-RAT 仍然 synthetic mechanism no-go。
4. real D-RAT feature transport 仍未通过 function-preservation + geometry gate。
5. full_basis_param_transport_executed 仍为 0。
```

## 14. fallback2 后最终结论

v13.0 仍没有达成 S5，但最低成功从 S1 提升到：

```text
minimum_success = S2-SyntheticMechanismProof
```

最终合法 route：

```text
R3-SyntheticOnlyNoRealTransportBenefit
```

已闭合事实：

```text
1. affine / blockwise / scale-joint fallback 修复了 synthetic feature-space function preservation。
2. 所有 synthetic FamilyTransport 都能 function-preserving accepted，但只有 D-FOU X4 获得 future probe pass。
3. D-FOU 没有 S1 substrate，不能进入真实数据 promotion path。
4. D-RAT 有 S1 substrate，但 synthetic mechanism pass 为 0。
5. real D-RAT readout-feature transport 仍然 Drift_Q 接近 0.96 且 tangent condition 变坏。
6. 当前仍没有 full basis-parameter coordinate surgery。
```

新增 no-go boundary：

```text
1. feature-space affine compensation 能解决 synthetic function drift，但不能自动迁移到真实 D-RAT substrate。
2. synthetic-only proof 不能写成 official success。
3. 下一步如果继续，必须进入真正的 basis-parameter coordinate surgery 或 substrate design；继续 readout-feature proxy token 已经低价值。
```

## 15. provenance 修复后的最终可信结果

fallback2 后自查发现：候选选择 score 使用了 `drift_val`，即 probe/query batch drift。probe drift 作为 accept gate/audit 合法，但不能参与 transport direction/selection。因此做了 provenance 修复：

```text
experiments/run_v130_function_preserving_transport.py
  selection score 改为只使用 train-stream drift、feature RMS penalty、train-stream geometry；
  probe drift 仍记录并参与 accept gate，但不参与候选选择；
  provenance audit 新增 probe_batch_used_for_transport_selection=0。
```

修复后重跑 official，最终可信 route：

```text
route = R3-SyntheticOnlyNoRealTransportBenefit
minimum_success = S2-SyntheticMechanismProof
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_s1_count = 14
substrate_s2_healthy_base_count = 0
synthetic_rows = 42
synthetic_mechanism_pass_count = 2
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
full_basis_param_transport_executed = 0
provenance_violation_count = 0
```

Synthetic pass rows：

| family | task | seed | K | ControlGap | synthetic pass |
|---|---|---:|---:|---:|---:|
| D-CHE | X4 | 0 | 1000 | 0.01751038993991528 | 1 |
| D-FOU | X4 | 0 | 1000 | 0.04716192028906989 | 1 |

但 family-level promotion path 仍未打开：

```text
1. D-CHE / D-FOU 有 synthetic pass，但没有 S1 efficient substrate。
2. D-RAT 有 S1 efficient substrate，但 synthetic mechanism pass 为 0。
3. real D-RAT readout-feature transport pass 为 0。
4. full_basis_param_transport_executed 仍为 0。
```

Real D-RAT blocker：

| candidate | Drift_B | Drift_Q | TangentCondition ratio | accepted |
|---|---:|---:|---:|---:|
| D-RAT38-GroupDiversityTransportNoCE | 3.3582368814677466e-06 | 1.4307606220245361 | 1.6770742383008945 | 0 |
| D-RAT26-TangentTrustRegionNoCE | 3.4927684282592963e-06 | 4.8830108642578125 | 0.8352663401283539 | 0 |

最终判断：

```text
v13.0 仍没有达成 S5；
已从原始 R2 推进到 R3；
最低成功为 S2-SyntheticMechanismProof；
不允许 promotion；
合法 route = R3-SyntheticOnlyNoRealTransportBenefit。
```

本次是真实进展：function-preservation fallback 和 no-probe-selection 修复后，synthetic mechanism proof 从 0 增至 2。但它们属于没有 S1 substrate 的 D-CHE/D-FOU；真实 D-RAT transport 仍失败。因此继续推进需要 Non-RAT substrate design 或 full basis-parameter coordinate surgery，而不是继续扩 readout-feature proxy。

## 16. train-stream fit/select split fallback 后复核

用户再次要求继续推进后，重新审计 fallback4 发现一个真实风险：transport readout compensation 可能在 train stream 上过拟合，表现为 train drift 很小但 probe/query drift 很大。为此继续执行计划内 function-preservation fallback，新增 train-stream 内部 fit/select split：

```text
experiments/run_v130_function_preserving_transport.py
  新增 train_stream_fit_select_indices。
  FamilyTransport 的 readout compensation 在 fit split 上求解。
  selection score 只用 select split drift、train drift、RMS penalty、geometry。
  probe/query drift 仍仅作为 audit/gate，不参与方向选择。
```

重跑 default official 后，最终 route 变为：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
substrate_s1_count = 14
substrate_s2_healthy_base_count = 0
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
full_basis_param_transport_executed = 0
provenance_violation_count = 0
```

解释：这个更严格 split 修复后，synthetic function preservation 仍大多能成立，但 future probe 全部为 0；因此之前 R3 不能作为最终可信结论。

## 17. future optimizer sensitivity fallback

default future probe 下观察到：多个 FamilyTransport row 的 `ControlGap_transport` 为正，但 CEp99 tail 明显坏化，gate 因 CEp99 审计约束拒绝。计划要求 future training no-benefit 时先做 K sensitivity / optimizer sensitivity / MLP analog / readout-only 对照；runner 已覆盖 K=50/200/1000、MLP analog、readout-only，本轮追加低 LR future probe sensitivity，且把 probe 超参数写入 CSV：

```text
experiments/run_v130_function_preserving_transport.py
  v130_future_training_probe.csv 新增 future_lr / future_weight_decay。
  provenance audit 新增 future_optimizer_probe_as_direction=0。
```

低 LR diagnostic 结果：

```text
smoke_v130_future_lr003_probe:
route = R3-SyntheticOnlyNoRealTransportBenefit
synthetic_mechanism_pass_count = 1
synthetic pass = D-FOU X4 seed=0 K=1000 ControlGap=0.007501776195555615
real_basis_feature_transport_pass_count = 0
```

低 LR official 结果：

```text
route = R3-SyntheticOnlyNoRealTransportBenefit
minimum_success = S2-SyntheticMechanismProof
synthetic_mechanism_pass_count = 1
synthetic pass = D-RAT X2 seed=0 K=200 ControlGap=0.01872296667919826
future_lr = 0.003
future_weight_decay = 0.001
real_basis_feature_transport_pass_count = 0
full_basis_param_transport_executed = 0
```

解释：低 LR future-probe sensitivity 是真实进展，证明 D-RAT synthetic proof 可以在更稳的 future optimizer probe 下出现。但它仍没有打开真实数据 transport，因此仍不能 promotion。

## 18. local hidden blend fallback 后最终可信结果

由于低 LR 后真实 D-RAT 仍因 function preservation fail 被拒绝，继续执行计划 12 中的 smaller transport step / local hidden reconstruction fallback。本轮将 FamilyTransport 扩展为 hidden-feature blend：

```text
transport_blend in (1.0,0.75,0.50,0.25,0.10,0.05)
```

审计说明：

```text
1. hidden blend 在原 hidden/readout feature 与 whitening transport feature 之间插值。
2. selection 仍只用 train-stream split drift、train drift、RMS penalty、geometry。
3. 不使用 label/CE/probe/query/future outcome 选择 direction。
4. fallback_note 写出 transport_blend，便于复现。
```

smoke 结果显示它有局部改善，但不够过 gate：

| row | Drift_B | Drift_Q | TangentCondition ratio | accepted |
|---|---:|---:|---:|---:|
| D-RAT38 smoke hidden blend | 0.2628291845321655 | 1.0001124143600464 | 0.7170520934977773 | 0 |

最终 hidden-blend official route：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 14
substrate_s2_healthy_base_count = 0
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
generic_reparameterization_effect = 0
KAN_specific_claim_allowed = 0
full_basis_param_transport_executed = 0
provenance_violation_count = 0
```

FamilyTransport accepted 聚合：

| scope/family | accepted | min Drift_B | min Drift_Q | max GeomGain |
|---|---:|---:|---:|---:|
| synthetic D-CHE | 7/7 | 3.2011053008318413e-07 | 4.165758298313449e-07 | 9.519591344585423 |
| synthetic D-FOU | 7/7 | 3.0316644483718846e-07 | 3.4925858471979154e-07 | 9.6107809240638 |
| synthetic D-RAT | 7/7 | 3.3884444405885006e-07 | 4.549314667201543e-07 | 9.56073486846692 |
| synthetic D-RBF | 7/7 | 3.2003416094994463e-07 | 4.5836171125301917e-07 | 9.535620081600397 |
| synthetic D-WAV | 7/7 | 3.276582845046505e-07 | 4.817370040655078e-07 | 9.51451973937376 |
| synthetic MLP | 7/7 | 3.953620932861668e-07 | 4.699540454566886e-07 | 9.230615510408153 |
| real MLP analog | 1/1 | 5.299695544636052e-07 | 7.18850117209513e-07 | 6.43807032828094 |
| real D-RAT | 0/2 | 0.0948987826704979 | 0.9619827270507812 | 4.085259964386785 |

Real D-RAT blocker：

| candidate | Drift_B | Drift_Q | FunctionDrift pass | TangentCondition ratio | accepted |
|---|---:|---:|---:|---:|---:|
| D-RAT38-GroupDiversityTransportNoCE | 0.6267601251602173 | 1.0111536979675293 | 0 | 1.6770742383008945 | 0 |
| D-RAT26-TangentTrustRegionNoCE | 0.0948987826704979 | 0.9619827270507812 | 0 | 0.8352663401283539 | 0 |

最终判断：

```text
v13.0 没有达成 S5；
最新可信 official 只达到 S1-EfficientSubstrate；
synthetic_mechanism_pass_count = 0；
real_basis_feature_transport_pass_count = 0；
不允许 promotion；
合法 route = R2-SyntheticMechanismProofFailed。
```

科学结论：

```text
1. affine / blockwise / scale / train-stream split / low-LR future probe / hidden blend 都已执行。
2. synthetic function preservation 可被打开，但 future proof 在最终不使用 future outcome 选 direction 的 hidden-blend official 下仍为 0。
3. 真实 D-RAT readout-feature transport 仍不能 function-preserving accepted，主要卡在 Drift_B / Drift_Q。
4. MLP analog 可 function-preserving accepted，但没有 future pass；generic_reparameterization_effect=0。
5. 当前 runner 明确 full_basis_param_transport_executed=0，因此不能宣称真正 basis-parameter coordinate surgery。
6. 继续在 readout-feature proxy 上扩 token/网格会变成低价值搜索；下一步需要 full basis-parameter coordinate transport hook 或重新回到 substrate design。
```

## 19. 用户再次追问后的 full-basis hook 源码复核

用户再次要求确认 v13.0 是否达成目标，若未达成则继续。本次重新读取 final route，结论仍未变化：

```text
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 14
synthetic_mechanism_pass_count = 0
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
full_basis_param_transport_executed = 0
provenance_violation_count = 0
```

本次新增的是源码级 full-basis hook 可行性审计，没有新增训练实验。审计对象：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v130_function_preserving_transport.py
```

审计事实：

```text
1. GroupedRationalKATKAN.frozen_readout_features() 拼接的是后验特征：hidden、hidden square/abs/rat、linear residual、projected square/bilinear、cross/PCA/signal、bias。
2. GroupedRationalKATKAN.forward() 对这些分支使用多组独立参数分别读出：w2、hidden_*_readout、linear_readout、proj_*_readout、cross_*_readout 等。
3. 当前 v13.0 runner 只在 frozen readout feature 上做 transport/refit，route_detail 已明确写出 readout_feature_transport_only_not_full_basis_parameter_surgery。
4. build_route() 中 official gate 显式要求 full_basis_param_transport_executed == 1；当前值为 0。
```

对“继续修复”的判断：

```text
1. 现有模型可以做 hidden-channel permutation，但这基本只改变参数排列，不改变 functional geometry，不能作为新的有效机制。
2. 现有模型也可以尝试 hidden near-scaling，但 tanh + hidden_rat_residual + 多分支 readout 使它不再是可验证的 full basis-coordinate transport；做了也很容易变成不可审计的局部 proxy。
3. 给模型外面套 feature-transform wrapper 仍然是 readout-feature proxy，不能把 full_basis_param_transport_executed 写成 1。
4. 计划 12 明确规定 all families fail synthetic proof 时必须停止 functional transport、回到 substrate design，不允许继续扩 token。
```

因此本次不启动新增训练、不新增 CSV 指标、不修改 gate。这个 stop 不是把失败写成成功，而是避免把 proxy 包装成 full basis-parameter coordinate surgery。

最终判断仍是：

```text
v13.0 没有达成 S5；
只达到 S1-EfficientSubstrate；
synthetic_mechanism_pass_count = 0；
real_basis_feature_transport_pass_count = 0；
full_basis_param_transport_executed = 0；
不允许 promotion；
合法 route 仍是 R2-SyntheticMechanismProofFailed。
```

本次复核后重新刷新 code packet / manifest。由于 code packet 包含本复盘与执行日志，任何后续日志编辑都会改变 zip hash；最终 hash 以 `v130_route_decision.json` 为准。本轮需要审计的 packet 状态为：

```text
required_artifact_missing_count = 0
code_review_packet_entries = 8
zip_has_exec_log = True
zip_has_review_log = True
```
