# DG-KAN v16.1 MultiHypothesisFunctionalDynamics 4GPU 执行日志

生成时间：2026-06-01（Asia/Singapore）

## 1. 关键文件

- plan: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_完整计划.md
- runner: /home/chengshun.wang/DG-LCA/experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py
- shared kernel source: /home/chengshun.wang/DG-LCA/experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py
- all-basis registry: /home/chengshun.wang/DG-LCA/experiments/run_v149_line_d_all_basis_substrate_repair.py
- result dir: results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161
- all-basis line F dir: results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate
- fast horizon readback: 1
- executed method surface: M1..M6 representative mechanisms + matched controls; full R1..R9 ladder was deferred and is not claimed as covered
- recap: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_实验结果复盘.md
- execution log: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_执行日志.md

## 2. 编译检查

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. 四卡并行分片执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines A --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a0 --line-a-methods A-M1-R0-D-CHE-PulseOnce-AdamWRecovery,ACTRL0-D-CHE-AdamW
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines A --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a1 --line-a-methods A-M2-D-CHE-SplitConsensusSignalSubspace,A-M3-D-CHE-PopRiskSNRPreconditioner,ACTRL1-D-CHE-NoOpMatchedOverhead
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines A --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a2 --line-a-methods A-M4-D-CHE-FunctionSpaceProximal,A-M5-D-CHE-OptimizerAlignedFU,ACTRL2-D-CHE-RandomMatchedPulse
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines A --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a3 --line-a-methods A-M6-D-CHE-CarrierReparamGlobalDecay,ACTRL3-D-CHE-AdamWExtraStepsMatchedTime
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines B --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b0 --line-b-methods B-M1-R0-MLP-PulseOnce-AdamWRecovery,BCTRL0-MLP-AdamW
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines B --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b1 --line-b-methods B-M2-MLP-HiddenSplitConsensusSignalSubspace,B-M3-MLP-PopRiskSNRPreconditioner,BCTRL1-MLP-NoOpMatchedOverhead
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines B --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b2 --line-b-methods B-M4-MLP-FunctionSpaceProximal,B-M5-MLP-CautiousAdamW,BCTRL2-MLP-RandomMatchedPulse
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines B,C,D --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b3 --line-b-methods B-M6-MLP-HiddenCarrierDecay,BCTRL3-MLP-AdamWExtraStepsMatchedTime
```

Line F all-basis substrate command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU92-LowFreqIdentityResidualV7,D-FOU93-BandwiseSNRWarmupV7,D-FOU94-PhaseStableBandMixV7,D-FOU95-NoMaterializeLifetimeV7,D-FOU96-HighFrequencyQuarantineV7,D-RBF90-ActiveCenterOccupancyV7,D-RBF91-WidthConditionGuardV7,D-RBF92-CompactBumpNoDenseV7,D-RBF93-GaussianLocalK4TaskHealthV7,D-RBF94-IdentityResidualWidthWarmupV7,D-WAV77-TriangularSupportV7,D-WAV78-ScaleOccupancyV7,D-WAV79-SupportOverlapDampingV7,D-WAV80-LocalTailCoverageAuditV7 --device cuda:3 --data-root data --no-download --train-size 64 --val-size 48 --batch-size 32 --epochs 1
```

Merge / h1600 / finalize:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines MERGE_A,MERGE_B,A1600 --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines B1600 --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py --out-dir results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161 --line-f-out results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate --run-lines FINALIZE --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 64 --val-size 48 --test-size 48 --batch-size 64 --train-steps 60 --horizon-steps 800 --h1600-steps 1600 --rational-steps 50 --lq-steps 50 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --reuse-if-present 1
```

## 4. GPU assignment manifest

| round | gpu | run lines | suffix | rows | artifact exists | role |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | cuda:0 | A | a0 | 18 | 1 | D-CHE M1 representative shard |
| 1 | cuda:1 | A | a1 | 27 | 1 | D-CHE signal/SNR shard |
| 1 | cuda:2 | A | a2 | 27 | 1 | D-CHE proximal/optimizer shard |
| 1 | cuda:3 | A | a3 | 18 | 1 | D-CHE carrier-reparam shard |
| 2 | cuda:0 | B | b0 | 18 | 1 | MLP M1 representative shard |
| 2 | cuda:1 | B | b1 | 27 | 1 | MLP signal/SNR shard |
| 2 | cuda:2 | B | b2 | 27 | 1 | MLP proximal/optimizer shard |
| 2 | cuda:3 | B,C,D | b3 | 18 | 1 | MLP carrier plus LQ/Rational |
| 3 | cuda:3 | F |  | 126 | 1 | all-basis substrate |
| 4 | cuda:0 | MERGE_A,MERGE_B,A1600 |  | 18 | 1 | merge and D-CHE long horizon |
| 4 | cuda:1 | B1600 |  | 18 | 1 | MLP long horizon |
| 5 | cuda:0 | FINALIZE |  |  | 1 | route/docs/audits |

## 5. Artifact inventory

| artifact | exists | rows | bytes |
| --- | --- | --- | --- |
| v161_route_decision.json | 1 |  | 1000 |
| v161_method_surface_manifest.csv | 1 | 20 | 4324 |
| v161_carrier_registry_manifest.csv | 1 | 5 | 384 |
| v161_line_r_audit.csv | 1 | 9 | 346 |
| v161_forbidden_information_audit.csv | 1 | 9 | 346 |
| v161_no_action_search_audit.csv | 1 | 5 | 138 |
| v161_gpu_assignment_manifest.csv | 1 | 12 | 1333 |
| v161_line_g_dynamic_geometry.csv | 1 | 1260 | 870705 |
| v161_line_a_dche_matrix.csv | 1 | 90 | 202795 |
| v161_line_a_horizon_recovery.csv | 1 | 630 | 413521 |
| v161_line_a_h1600_long.csv | 1 | 18 | 43759 |
| v161_line_b_mlp_matrix.csv | 1 | 90 | 197871 |
| v161_line_b_horizon_recovery.csv | 1 | 630 | 360387 |
| v161_line_b_h1600_long.csv | 1 | 18 | 42415 |
| v161_line_c_lq_reanchor.csv | 1 | 63 | 20335 |
| v161_line_c_lq_functional.csv | 1 | 5 | 529 |
| v161_line_d_rational_monitor.csv | 1 | 45 | 46671 |
| v161_line_e_recovery_matrix.csv | 1 | 180 | 39535 |
| v161_line_f_allbasis_results.csv | 1 | 126 | 160159 |
| v161_line_f_allbasis_family_summary.csv | 1 | 3 | 707 |
| v161_carrier_mechanism_summary.csv | 1 | 12 | 2392 |
| v161_line_m_crossline_attribution.csv | 1 | 6 | 1659 |
| v161_failure_taxonomy.csv | 1 | 219 | 26398 |
| v161_budget_exhaustion_certificate.csv | 1 | 5 | 1122 |
| v161_no_go_boundary.csv | 1 | 7 | 763 |
| v161_next_hypothesis_queue.csv | 1 | 3 | 414 |
| v161_code_review_packet.csv | 1 | 5 | 1531 |
| v161_execution_contract_coverage_audit.csv | 1 | 13 | 844 |
| v161_deep_coverage_audit.csv | 1 | 9 | 572 |
| v161_required_artifact_manifest.csv | 1 | 52 | 2379 |
| carrier_mechanism_gate_heatmap.svg | 1 |  | 2578 |
| carrier_mechanism_source_heatmap_h800.svg | 1 |  | 2587 |
| carrier_mechanism_debt_recovery_heatmap_h800.svg | 1 |  | 2585 |
| carrier_mechanism_control_equivalence_heatmap.svg | 1 |  | 2598 |
| source_retention_curve_by_carrier.svg | 1 |  | 5034 |
| tail_debt_curve_by_carrier.svg | 1 |  | 5033 |
| LineC_debt_curve_by_carrier.svg | 1 |  | 5032 |
| calibration_debt_curve_by_carrier.svg | 1 |  | 5044 |
| AUCtime_curve_by_carrier.svg | 1 |  | 5051 |
| recovery_rate_by_mechanism.svg | 1 |  | 5415 |
| D-CHE_vs_MLP_source_retention.svg | 1 |  | 565 |
| D-CHE_vs_MLP_tail_recovery.svg | 1 |  | 561 |
| D-CHE_vs_MLP_LineC_recovery.svg | 1 |  | 560 |
| KAN_specific_delta_heatmap.svg | 1 |  | 1369 |
| generic_vs_kan_specific_route_dashboard.svg | 1 |  | 716 |
| LQ_reanchor_nearpass_heatmap.svg | 1 |  | 5675 |
| LQ_protocol_drift_heatmap.svg | 1 |  | 5684 |
| basis_family_pass_count_heatmap.svg | 1 |  | 4998 |
| basis_step_memory_pareto.svg | 1 |  | 5733 |
| basis_LineC_task_health.svg | 1 |  | 5733 |
| gpu_assignment_timeline.svg | 1 |  | 2298 |
| gpu_rows_executed_by_round.svg | 1 |  | 2320 |
| gpu_idle_reason_table.svg | 1 |  | 2522 |

## 6. Final route snapshot

```text
route = R16_1-FunctionalMechanismMatrixNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
```

## 7. 修复与审计记录

```text
1. 新增 v16.1 runner，使用 v16.0 validated kernels，但单独生成 v161 artifacts/docs/route。
2. 新增 v16.1 substrate-only candidate registry；Line F rows 仍不能 promotion。
3. method_surface_manifest 记录每个方法的内部映射与参数 override，便于后续审计。
4. gpu_assignment_manifest 记录四卡分片；cpu_offload_used=0。
5. runtime blocker 修复：exact per-step full ladder 过慢；正式结果只声明 horizon-target representative surface，不把未执行 ladder 写成 coverage。
```

## 8. 用户再次追问后的闭环复核指令

本节只做计划/route/artifact readback，不新增训练，不改变指标。

```bash
rg -n "S2|S3|S5|No-go|no-go|route|继续|停止|forbidden|G9|controller|reset|action|fallback|h1600|promotion|达成" docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_完整计划.md

python - <<'PY'
import json, csv, pathlib
base = pathlib.Path('results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161')
route = json.loads((base / 'v161_route_decision.json').read_text())
print(json.dumps(route, indent=2, ensure_ascii=False))
for f in [
    'v161_required_artifact_manifest.csv',
    'v161_execution_contract_coverage_audit.csv',
    'v161_deep_coverage_audit.csv',
    'v161_budget_exhaustion_certificate.csv',
    'v161_no_go_boundary.csv',
]:
    rows = list(csv.DictReader((base / f).open()))
    print('\\n', f, len(rows))
    if f == 'v161_required_artifact_manifest.csv':
        print('missing=', sum(int(r.get('missing') or 0) for r in rows))
    if 'coverage' in f:
        print('nonpass=', [r for r in rows if str(r.get('status')) != '1'])
PY
```

复核输出摘要：

```text
route = R16_1-FunctionalMechanismMatrixNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
required_artifact_manifest_rows = 52
required_artifact_missing_rows = 0
execution_contract_nonpass_rows = 0
deep_coverage_nonpass_rows = 0
```

复核结论：

```text
1. v16.1 覆盖执行目标已闭合；promotion/scientific success 未达成。
2. 完整计划第 13 节要求此时写 R16_1-FunctionalMechanismMatrixNoGo，并停止当前 train-stream functional-update family。
3. 当前版本内不允许继续 G9/G10/action bank/controller/reset/audit-directed search。
4. 后续只能作为新预注册计划推进：MechanismSpecificRecoveryDefinitionRevision、MLPGenericDynamicsMechanismFollowup 或 CarrierSubstrateBeforeFU。
```

## 9. 二次追问后的最终闭环复核

本节只记录同一批 artifact 的二次 readback，不新增训练。

```bash
python - <<'PY'
import json, csv, pathlib
base = pathlib.Path('results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161')
route = json.loads((base / 'v161_route_decision.json').read_text())
print('route=', route['route'])
print('promotion_allowed=', route['promotion_allowed'])
print('S2=', route['S2_weak_productive_dynamics_reached'])
print('S3=', route['S3_productive_debt_recovery_reached'])
print('S5=', route['official_s5_reached'])
print('required_missing=', route['required_artifact_missing_count'])
for f in [
    'v161_required_artifact_manifest.csv',
    'v161_execution_contract_coverage_audit.csv',
    'v161_deep_coverage_audit.csv',
    'v161_budget_exhaustion_certificate.csv',
]:
    rows = list(csv.DictReader((base / f).open()))
    print(f, len(rows))
PY
```

输出摘要：

```text
route = R16_1-FunctionalMechanismMatrixNoGo
promotion_allowed = 0
S2 = 0
S3 = 0
S5 = 0
required_missing = 0
v161_required_artifact_manifest.csv = 52 rows
v161_execution_contract_coverage_audit.csv = 13 rows
v161_deep_coverage_audit.csv = 9 rows
v161_budget_exhaustion_certificate.csv = 5 rows
```

最终复核结论：

```text
1. v16.1 内没有剩余合法 continuation。
2. 不继续训练不是放弃，而是遵守完整计划的 no-go / no-action-search 边界。
3. 下一步只能新开预注册计划。
```
