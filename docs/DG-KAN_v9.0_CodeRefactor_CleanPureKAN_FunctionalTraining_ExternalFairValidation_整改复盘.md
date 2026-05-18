# DG-KAN v9.0 CodeRefactor CleanPureKAN 整改复盘

> 本复盘记录 `docs/DG-KAN_v9.0_CodeRefactor_CleanPureKAN_FunctionalTraining_ExternalFairValidation_完整整改计划.md` 的首轮执行结果。所有结论只来自本文件列出的落盘 artifact 与源码审计；没有 fake data、proxy rows、手填训练结果或未跑成功结论。

## 0. 当前结论

> 本文件按执行顺序追加。前面第 1-8 节保留首轮 Phase A/B 审计结论；最新结论以第 14 节为准。

截至第 14 节，v9.0 已继续执行到一个可审计的 terminal route：

```text
route = R3-CleanTransitionalOnlySuccess
success_v90_clean_ce = true
success_v90_contract_hardening = true
success_v90_core_skeleton = true
success_v90_full_edge = false
success_v90_external_fair = false
success_v90_strong = false
```

这不是 Clean FullEdge PureKAN success，也不是 external fair / strong success。真实结论是：

1. CleanCE transitional route 已通过：clean CE、compiled fused CE/head、无 label smoothing 的 transitional DG-KAN 在 MNIST 上保住 task / geometry / wall-clock。
2. FullEdge manual implementation 与 edge-layer gradcheck 通过，functional causality controls 也通过。
3. FullEdge 外部公平失败：三任务 best full-edge functional candidate 均低于 KB-MLP，并且 training compute fair counter 失败。
4. v9.0 当前完成的是“整改执行到边界并诚实停在 R3”，不是把 full-edge 未过的部分包装成成功。

最终 artifact：

```text
results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z/
```

首轮 Phase A/B 完成的是：

```text
Phase A legacy truth audit = measured
Phase B core package skeleton = pass
Phase B contract hardening negative tests = pass
FullEdge minimal edge-layer implementation / gradcheck = pass
CleanCE transitional revalidation = not_run
External fair validation = not_run
success_v90_full_edge = false
success_v90_external_fair = false
success_v90_strong = false
```

Route：

```json
{
  "route": "R0-RefactorAuditInProgress",
  "phase_a_legacy_truth_audit_pass": 1,
  "phase_b_core_package_import_pass": 1,
  "phase_b_core_smoke_pass": 1,
  "phase_d_edge_layer_gradcheck_pass": 1,
  "full_edge_contract_pass": 1,
  "full_edge_task_pass": 0,
  "negative_contract_tests_pass": 1,
  "legacy_selected_route_v90_interpretation": "TransitionalDGKANDiagnostic",
  "legacy_clean_official_claim_allowed": 0,
  "primary_blocker": "full_edge_purekan_clean_ce_revalidation_and_external_validation_not_run"
}
```

## 1. 本轮新增代码

| 路径 | 作用 |
|---|---|
| `dgkan/` | v9 clean core package skeleton；把 specs / contracts / model primitives / optimizer / functional / external / profiling / artifacts 拆出 runner |
| `experiments/run_v90_refactor_audit.py` | v9 Phase A/B 审计 runner；只做 legacy truth audit、module import、contract negative tests、core smoke 与统一 artifact skeleton |

代码检查：

```text
python -m py_compile experiments/run_v90_refactor_audit.py $(find dgkan -name '*.py' | sort)
```

已通过。

## 2. 本轮 artifact

```text
results/real_rerun_20260506/v90_phase_ab_refactor_audit_20260508T230000Z/
```

关键文件：

```text
run_manifest.json
code_audit_legacy_current.csv
candidate_registry_clean.csv
contract_audit_clean.csv
contract_negative_tests.csv
module_import_audit.csv
core_smoke_tests.csv
full_edge_implementation_audit.csv
gradcheck_full_edge.csv
route_decision.json
aggregate_decision.json
failure_table.csv
v90_provenance_audit.csv
artifact_hashes.csv
```

计划要求但本轮未执行的后续 artifact 已以 `status=not_run` 落盘，包括：

```text
legacy_refactor_parity.csv
label_smoothing_removal_audit.csv
clean_transitional_revalidation.csv
full_edge_external_validation.csv
functional_causality_controls.csv
robust_timing_protocols.csv
training_compute_counter.csv
kanbefair_baseline_reproduction.csv
external_fair_envelope.csv
robustness_perturbation.csv
continual_balance_multisplit.csv
boundary_audit.csv
```

## 3. Legacy truth audit

审计对象：

```text
results/real_rerun_20260506/v87_full_chain_fmnist_kmnist_selected_kw4_formal_repeat_20260508T203000Z/
experiments/run_gafu_v87_real.py
experiments/run_gafu_v85_real.py
experiments/run_gafu_v83_real.py
experiments/run_gafu_v72_real.py
```

关键结果：

| 项 | 结果 |
|---|---|
| selected candidate | `KW4` / child `DG1-FT7-KW4-hidden28-functional` |
| v8.7 artifact formal flag | `success_v87_formal = 1` |
| v9 interpretation | `TransitionalDGKANDiagnostic` |
| model level | `transitional` |
| stack type | `linear_silu_packed_or_cached` |
| head type | `poly2_silu_head` |
| loss type in artifact | `CE` |
| selected artifact label smoothing value | `not_recorded` |
| source uses smooth CE helper | `1` |
| legacy source has nonzero smoothing candidates | `1` |
| full-edge PureKAN | `0` |
| clean official claim allowed from legacy artifact | `0` |

判断：

1. v8.7 selected route 可以作为 legacy behavior freeze 目标，但不能在 v9.0 中直接记为 Clean FullEdge PureKAN。
2. 原因不是说 v8.7 数据是假，而是 v9.0 的合同更严格：selected route 是 transitional linear-SiLU stack，不是 full-edge edge-function stack。
3. legacy artifact 没有显式记录 selected `label_smoothing` 数值；旧源码也保留 smooth CE helper 和非零 smoothing 候选。因此 v9.0 不能把旧结果直接继承为 CleanCE official。

Legacy behavior freeze 关键值：

| metric | value |
|---|---:|
| DG functional MNIST acc | `97.18999862670898` |
| delta vs KB-MLP | `+0.279998779296875` |
| curvature ratio vs DG base | `0.7724828819718651` |
| step ratio vs KB-MLP | `0.794290891683765` |
| functional events | `73` |
| FMNIST delta vs MLP | `+1.410001516342163` |
| KMNIST delta vs MLP | `+2.1799981594085693` |

这些数值只作为 legacy freeze，不作为 v9 Clean FullEdge 成功。

## 4. Contract hardening

`contract_negative_tests.csv` 结果：

```text
negative_contract_tests_pass = 1
tests = 13
```

已确认以下违规输入都会触发 contract violation：

```text
label_smoothing != 0
loss_type != CE
external_teacher_used = 1
self_teacher_used = 1
geometry_loss_used = 1
sampler_changed = 1
class_weight_used = 1
cpu_offload_used = 1
uses_loss_backward = 1
fake_data_used = 1
proxy_row_used = 1
model_level=full_edge with linear stack
```

说明：这些 fake/proxy/offload=1 只出现在 deliberate negative contract tests 中，用来证明合同会拒绝违规输入；它们没有被计入真实实验 provenance。

## 5. Core skeleton 与 FullEdge gradcheck

Module import audit：

```text
modules_checked = 11
import_pass = 11/11
```

Core smoke：

| check | value |
|---|---|
| stage | `B1_CORE_SMOKE_EDGE_LAYER` |
| input shape | `(5, 3)` |
| output shape | `(5, 2)` |
| dx shape | `(5, 3)` |
| grad shape | `(3, 2, 6)` |
| manual backward shape pass | `1` |
| uses_loss_backward | `0` |

判断：`dgkan` 包已经可 import，并有最小 full-edge `EdgeFunctionLayer` manual forward/backward shape smoke。

FullEdge edge-layer gradcheck：

| check | value |
|---|---:|
| candidate | `DG-FullEdge-Poly2Silu-edge-layer-smoke` |
| model level | `full_edge` |
| edge basis | `poly2_silu` |
| output max abs diff | `0.0` |
| dx max abs diff | `3.725290298461914e-09` |
| grad max abs diff | `1.1920928955078125e-07` |
| GradPass | `1` |
| uses_loss_backward | `0` |

判断：最小 `EdgeFunctionLayer` 的 manual backward 与 autograd reference 对齐；autograd 只用于 verification，未调用 `loss.backward`，也不是 official training path。Full-edge task training 仍未运行，因此 `success_v90_full_edge = false`。

## 6. No-fake audit

```text
rows_checked = 5
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

注意：contract negative tests 中的违规输入是测试合同拒绝能力，不是实验 evidence rows；本轮 provenance audit 只统计真实审计 / smoke artifact。

## 7. Hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v90_refactor_audit.py` | `7faabb953b392245f1356cc4814a42e8a034482034ff5b6783854d0f21bfddc8` |
| `dgkan/specs.py` | `2456ecd586a3bab067447918c41a62efc67ed8f2fa5c57035162281bec5432c4` |
| `dgkan/contracts.py` | `03d0ce0f729d59505df8713e4b54d168e387946e7cc8ce036f39c644e133c465` |
| `dgkan/models/edge_layers.py` | `ed815553f065713287ce081067c12c998db51af294dcacc07f49fa1d55e7ed4a` |
| route | `b770c8ecaadfa1d2a356e58a3a1cffe6fec4c889a93a73e60d7b7d11107e6cc3` |
| legacy audit | `4e6620713bdf0223ea68e4da9aae23fee3bb2073009c10714900d50e248eccc9` |
| contract audit | `c451be4e83da22e2bf28b6a85ac6e52bd914d53b2ddb2e683e58926967720d8f` |
| core smoke | `5598e16f2dc4d0821b0798478b94b96c9b5b25ffb9543ea05dcb25a378e59d79` |
| full-edge implementation audit | `a173dc87bf9255e6e6447d2a795dc6e7774901df18440953d4d1254d22f4c20b` |
| full-edge gradcheck | `a315c6f1dec90f5634117868fc90d63ac0a126ff2c9d9960581e5f6f596fb116` |
| provenance audit | `9466cb555d6c40794bd4fdef579fa34c4152355e03387bdfc9da19aa2afce496` |

## 8. 下一步

按 v9.0 计划，下一步不能直接做 external success 宣称，应继续：

```text
Step 4: Transitional clean CE revalidation
Step 5: FullEdge-Poly2Silu full model implementation + task smoke
Step 6: functional update migration onto full-edge parameters
Step 7: unified timing / memory / training compute closure
Step 8: KB-MLP / KB-KAN external fair validation
```

当前最重要的边界：

> v9.0 首轮整改完成了审计、合同地基和最小 FullEdge edge-layer gradcheck，但没有完成 CleanCE、FullEdge task training、external fair 或 broad strong。旧 v8.7 结果仍有价值，但在 v9.0 中只能作为 transitional diagnostic / regression target，不能直接继承为 Clean FullEdge PureKAN 成功。

## 9. 追加：继续执行到 CleanCE / FullEdge external terminal route

本节继续执行 v9.0 整改计划，不沿用 v8.7 的 selected success 作为 v9 成功结论。新增 runner 真实执行 CleanCE transitional revalidation、manual FullEdge external validation、functional causality controls、robust timing 与 training compute counter。

新增 / 修改代码：

| 路径 | 作用 |
|---|---|
| `dgkan/models/manual_full_edge.py` | 新增 manual FullEdge classifier / edge layer，支持 `poly2_silu6`、`compact_poly_silu3`、`rbf4`、`spline4` |
| `dgkan/training/manual_full_edge.py` | 新增 manual CE training、manual AdamW、functional smoothing controls、timing helper |
| `dgkan/models/__init__.py` | 导出 `ManualFullEdgeClassifier` / `ManualFullEdgeLayer` |
| `experiments/run_v90_clean_validation.py` | 新增 v9 full validation runner，生成 CleanCE、FullEdge、controls、timing、compute、route、audit artifacts |

代码检查：

```text
python -m py_compile experiments/run_v90_clean_validation.py experiments/run_v90_refactor_audit.py $(find dgkan -name '*.py' | sort)
```

已通过。

最终执行：

```bash
python experiments/run_v90_clean_validation.py \
  --out-dir results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z \
  --fresh \
  --device auto \
  --data-root data \
  --kanbefair-path third_party/KANbeFair \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --train-size 60000 \
  --test-size 10000 \
  --epochs 20 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --seed 1314
```

## 10. Final route

`route_decision.json`：

```json
{
  "route": "R3-CleanTransitionalOnlySuccess",
  "success_v90_clean_ce": 1,
  "success_v90_contract_hardening": 1,
  "success_v90_core_skeleton": 1,
  "success_v90_full_edge": 0,
  "success_v90_external_fair": 0,
  "success_v90_strong": 0,
  "clean_ce_pass": 1,
  "full_edge_contract_pass": 1,
  "full_edge_task_pass": 0,
  "functional_causality_pass": 1,
  "robust_timing_pass": 1,
  "training_compute_fair_pass": 0,
  "primary_blocker": "full_edge_candidates_fail_external_fair_or_compute_gate",
  "next_required_implementation": "redesign_full_edge_parameterization_or_accept_transitional_only_boundary"
}
```

判断：

1. v9.0 继续执行到计划允许的边界 route；没有停在中间状态。
2. CleanCE transitional 成功，但 FullEdge PureKAN 没有过外部公平 / compute gate。
3. 因此本轮不能声明 `success_v90_full_edge`、`success_v90_external_fair` 或 `success_v90_strong`。

## 11. CleanCE transitional revalidation

`clean_transitional_revalidation.csv` summary：

| metric | value |
|---|---:|
| task | `MNIST` |
| protocol | full train `60000` / full test `10000` |
| KB-MLP acc | `0.9690999985` |
| legacy smooth functional acc | `0.9718999863` |
| clean functional acc | `0.9734999537` |
| clean functional delta vs KB-MLP | `+0.0043999553` |
| clean functional step ratio vs KB-MLP | `1.1588365644` |
| clean CE task preservation pass | `1` |
| clean CE geometry pass | `1` |
| clean CE system pass | `1` |
| clean CE external fair pass | `1` |
| clean CE pass | `1` |

判断：

1. 去掉 label smoothing 后，transitional clean CE route 没有 collapse。
2. clean functional acc 高于 KB-MLP，也高于本轮 legacy smooth functional acc。
3. compiled fused CE/head + prewarm 后 step ratio 为 `1.158837`，低于 `1.50`。
4. 这只证明 transitional DG-KAN clean CE route 成立；它不是 FullEdge PureKAN 成功。

## 12. FullEdge external validation

FullEdge candidates：

```text
DG-FullEdge-Poly2Silu6-h5
DG-FullEdge-CompactPolySilu3-h11
DG-FullEdge-CompactPolySilu3-h11x2
DG-FullEdge-RBF4-h8
DG-FullEdge-Spline4-h8
```

Baseline reproduction：

| task | KB-MLP acc | KB-KAN acc |
|---|---:|---:|
| MNIST | `96.989995` | `65.069997` |
| Fashion-MNIST | `86.979997` | `76.830000` |
| KMNIST | `83.190000` | `38.699999` |

Best FullEdge functional rows:

| task | best FullEdge functional | acc | delta vs KB-MLP | params ratio | FLOPs ratio | memory ratio | step ratio | curvature ratio | min pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | `CompactPolySilu3-h11x2+Functional` | `94.949996` | `-2.039999` | `1.043811` | `1.550366` | `1.864129` | `1.228074` | `0.522096` | 0 |
| Fashion-MNIST | `CompactPolySilu3-h11+Functional` | `85.670000` | `-1.309997` | `1.029548` | `1.529181` | `1.856530` | `0.981949` | `0.386515` | 0 |
| KMNIST | `CompactPolySilu3-h11x2+Functional` | `78.679997` | `-4.510003` | `1.043811` | `1.550366` | `1.862592` | `1.106736` | `0.480719` | 0 |

`full_edge_external_summary.csv`：

```text
task_count = 3
full_edge_task_pass_count = 0
full_edge_external_fair_pass = 0
functional_causality_pass = 1
robust_timing_pass = 1
training_compute_fair_pass = 0
q90_step_ratio_vs_KB_MLP = 1.1568331718
max_step_ratio_vs_KB_MLP = 1.2280741993
```

判断：

1. FullEdge functional update 有几何作用，三任务 curvature ratio 都显著低于 base。
2. 但三任务 best FullEdge functional acc 均低于 KB-MLP，且 FLOPs / memory / backward compute estimate 超出公平 envelope。
3. 因此 FullEdge route 的失败是 task + compute 双重失败，不是 timing 或 causality 单项失败。

## 13. Controls / timing / compute

FullEdge implementation audit：

| metric | value |
|---|---:|
| manual forward | `1` |
| manual backward | `1` |
| manual update | `1` |
| poly2_silu6 implemented | `1` |
| compact_poly_silu3 implemented | `1` |
| rbf4 implemented | `1` |
| spline4 implemented | `1` |
| full edge contract pass | `1` |
| GradPass | `1` |

FullEdge edge-layer gradcheck：

| metric | value |
|---|---:|
| output max abs diff | `0.0` |
| dx max abs diff | `3.725290298461914e-09` |
| grad max abs diff | `1.1920928955078125e-07` |
| GradPass | `1` |

Functional causality controls：

| task | Functional curvature ratio | RandomFunc curvature ratio | ShuffledRole curvature ratio | CausalityPass |
|---|---:|---:|---:|---:|
| MNIST | `0.522096` | `0.986738` | `1.063280` | 1 |
| Fashion-MNIST | `0.386515` | `0.999233` | `1.103875` | 1 |
| KMNIST | `0.480719` | `1.002601` | `0.965675` | 1 |

Robust timing:

| metric | value |
|---|---:|
| rows | `15` |
| tasks | `MNIST,Fashion-MNIST,KMNIST` |
| q90 step ratio vs KB-MLP | `1.1568331718` |
| max step ratio vs KB-MLP | `1.2280741993` |
| robust timing pass | `1` |

Training compute counter:

| task | candidate | forward FLOPs ratio | backward estimate ratio | step ratio | pass |
|---|---|---:|---:|---:|---:|
| MNIST | `CompactPolySilu3-h11x2+Functional` | `1.550366` | `3.100731` | `1.228074` | 0 |
| Fashion-MNIST | `CompactPolySilu3-h11+Functional` | `1.529181` | `3.058361` | `0.981949` | 0 |
| KMNIST | `CompactPolySilu3-h11x2+Functional` | `1.550366` | `3.100731` | `1.106736` | 0 |

判断：

1. FullEdge implementation contract 和 gradcheck 是实打实通过。
2. Functional causality controls 也通过，说明 full-edge functional smoothing 不是纯 no-op。
3. Timing pass 说明当前失败不是 wall-clock step ratio。
4. Training compute fail 说明 full-edge 参数化虽然可运行，但还没有在公平 compute envelope 内打过 KB-MLP。

## 14. No-fake audit / hash / 最终结论

No-fake audit：

```text
rows_checked = 71
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `dgkan/models/manual_full_edge.py` | `a46eb944831a3381b6427b0c0d61cf79ced04c0a5103a9b4a6c05fff8d805e4d` |
| `dgkan/training/manual_full_edge.py` | `22e17a5caa4fd005f4f6ad0577d3afa76c613c4b32507c16d0c4fb47157323d2` |
| `experiments/run_v90_clean_validation.py` | `c10bd5a1f535f498756675af3baa896251a23bc14834df1756a2f2ef9c5bc509` |
| route | `4e8d74d47be4f566df8c839a94878b635be2dea222f6c5d1a49a9229ae8e1fa1` |
| CleanCE transitional | `b450358b96db960ac36ff3ba91388ea3b3c7734d4e4d6a909f28ca1468502f61` |
| FullEdge external | `53318991136d5f9f86c8c7d4c704ac1919f73dad4e0f2c00b4d7a2d089684e7f` |
| causality controls | `a1adbce783ef71e4c3ad67226d3114d702c0b7a3fe11861f24424cd2b390c2e0` |
| robust timing | `973b8297691a87388cfb4a614f6a19282e45d47f90677db140d8b8fccaf11bdb` |
| training compute | `e4e7d85f7ec3f9b8e6bcb0b897bc153628f06bb99dfd68352f564a0a1022163a` |
| provenance audit | `be716bc991fab9a3bb34dcbb4c908b32157c28cc4436b1959089690db391316f` |

最终结论：

```text
CleanCE transitional revalidation = pass
FullEdge implementation / GradPass = pass
FullEdge functional causality controls = pass
FullEdge robust timing = pass
FullEdge external task fair = fail
FullEdge training compute fair = fail
route = R3-CleanTransitionalOnlySuccess
success_v90_clean_ce = true
success_v90_full_edge = false
success_v90_external_fair = false
success_v90_strong = false
```

v9.0 本轮已经“继续执行直到完成”到一个真实 terminal route：CleanCE transitional route 成立，但 Clean FullEdge PureKAN external fair validation 没有完成成功。下一步如果继续 v9.x，必须重新设计 full-edge parameterization / compute accounting / compact edge basis；不能把 transitional clean CE success 继承成 FullEdge PureKAN success。
