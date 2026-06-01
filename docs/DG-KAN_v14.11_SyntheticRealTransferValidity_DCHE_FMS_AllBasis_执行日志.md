# DG-KAN v14.11 SyntheticRealTransferValidity DCHE FMS AllBasis 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 本轮新增/修改文件

```text
experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_执行日志.md
docs/DG-KAN_v14.11_SyntheticRealTransferValidity_DCHE_FMS_AllBasis_实验结果复盘.md
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py --out-dir results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py --out-dir results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411 --run-split-batch-probe 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v149_line_d_all_basis_substrate_repair.py experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/line_d_substrate_hardening_v1411 --candidates D-FOU27-LowFreqIdentityResidualV2,D-FOU28-BandwiseSNRSafeWarmup,D-FOU29-PhaseStableBandMixNoHighFreq,D-FOU30-NoMaterializeLifetimeV3,D-RBF26-ActiveCenterOccupancyV2,D-RBF27-WidthConditionIdentityResidual,D-RBF28-CompactBumpNoDenseMaterialization,D-RBF29-GaussianLocalK4TaskHealth,D-WAV25-TriangularSupportV3,D-WAV26-ScaleOccupancyNoTailTarget,D-WAV27-LocalSupportOverlapDamping --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --linec-seeds 12319500,12319501,12319502 --compute-budgeted-run 1 --device cuda:0
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1411_synthetic_real_transfer_validity_dche_fms_all_basis.py --out-dir results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411 --run-split-batch-probe 1
```

执行中先发现 Line R 静态扫描把审计规则文本自身误判成 F-CHE8/controller/action-token 证据；
随后修改 `scan_forbidden()`，改为读取 artifact 的 method/flag 字段作为审计来源，并重跑。

## 3. 输入 artifact

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt123_steps200_interval200_v1410
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_composite_steps200_interval200_v1410
results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel
results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/line_d_substrate_hardening_v1411
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143
results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05
```

## 4. 输出 artifact

```text
results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411
```

主要输出：

```text
v1411_route_decision.json
v1411_synthetic_real_alignment.csv
v1411_synthetic_real_predictivity_summary.csv
v1411_train_stream_value_proxy.csv
v1411_train_stream_value_validity.csv
v1411_split_batch_value_probe.csv
v1411_split_batch_value_validity.csv
v1411_split_batch_control_validity.csv
v1411_v145_oracle_reference.csv
v1411_v145_missing_state_reference.csv
v1411_line_d_v1411_hardening_summary.csv
v1411_line_d_v1411_hardening_results.csv
v1411_line_d_v1411_hardening_route.json
v1411_required_artifact_manifest.csv
v1411_code_review_packet.zip
```

## 5. 复现说明

本 runner 是审计/finalizer surface，不执行新的 F-CHE token、controller、action search 或 real training。
Line E 使用既有 synthetic/real artifact 做 predictivity audit。
继续推进时已将 Line E battery 从 D-CHE internal 扩展到 v14.3 Rational K8 synthetic/real 和 v14.4 Rational S4b real reference。
继续推进时补入计划要求的数据源 v14.5 action-bank oracle diagnostics，输出 oracle upper-bound 与 missing-state reference；该数据源 `oracle_uses_audit_metric=1`，因此只作为审计参考，不作为 v14.11 direction。
Line V 使用既有 train-stream/model-state telemetry 字段构造 proxy audit，不使用 validation/test/LineC/tail/AUC/calibration 作为方向源。
继续推进时新增 split-batch probe：只在 synthetic train batch 内拆 B1/B2，比较已有 F-CHE/FB direction 与 AdamW 的 one-step counterfactual U_train；并补充 NoOpMatchedOverhead、RandomMatchedNorm、SameActiveFractionRandomMask control probe。该 probe 只用于 validity audit，不作为方向选择。
继续推进时执行计划 Line D 允许的 D-FOU27..30 / D-RBF26..29 / D-WAV25..27 substrate-only hardening；不执行 Non-RAT FMS proof，不使用 audit metric 生成方向。
Line E 和 Line V 均未通过时，按计划 Case A/B fail-closed，不启动新的 D-CHE real FMS 训练。

## 6. 用户再次追问后的最终复核

执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv,json
from pathlib import Path
out=Path('results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411')
route=json.loads((out/'v1411_route_decision.json').read_text())
print(json.dumps(route, indent=2, sort_keys=True))
with (out/'v1411_required_artifact_manifest.csv').open() as f:
    rows=list(csv.DictReader(f))
print('manifest_missing', sum(int(r['missing']) for r in rows), 'manifest_rows', len(rows))
print('progress')
with (out/'v1411_progress_table.csv').open() as f:
    for r in csv.DictReader(f): print(r)
print('failure_taxonomy')
with (out/'v1411_failure_taxonomy.csv').open() as f:
    for r in csv.DictReader(f): print(r)
PY
```

复核结果：

```text
route = R1-SyntheticGateNotPredictiveForRealTransfer
minimum_success = S3-DCHESyntheticFMSPass
line_e_pass = 0
line_v_pass = 0
line_fche_executed = 0
line_d_v1411_route = R8-NonRATSubstrateStillMissing
line_d_v1411_best_family_dataset_seed_pass_count = 0
real_dataset_seed_pass_count = 2
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
```

本次只做状态复核，不启动新训练。原因是当前计划内可执行分支已经覆盖：

```text
1. Line E synthetic-to-real validity audit；
2. Line V telemetry + split-batch value audit；
3. v14.5 oracle/missing-state audit reference；
4. Line D D-FOU/RBF/WAV substrate-only hardening。
```

继续新增 F-CHE / real FMS / action-controller / dataset-seed pattern repair 会违反 v14.11 stop/go 边界。

## 7. 用户再次追问后的二次复核

执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import json,csv
from pathlib import Path
out=Path('results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411')
route=json.loads((out/'v1411_route_decision.json').read_text())
keys=['route','minimum_success','synthetic_task_family_pass_count','real_dataset_seed_pass_count','line_e_pass','line_v_pass','line_fche_executed','line_d_v1411_route','line_d_v1411_best_family_dataset_seed_pass_count','official_s5_reached','promotion_allowed','required_artifact_missing_count']
for k in keys:
    print(f'{k}={route.get(k)}')
with (out/'v1411_required_artifact_manifest.csv').open() as f:
    rows=list(csv.DictReader(f))
print('manifest_missing_sum=', sum(int(r['missing']) for r in rows))
PY
```

结果：

```text
route=R1-SyntheticGateNotPredictiveForRealTransfer
minimum_success=S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count=5
real_dataset_seed_pass_count=2
line_e_pass=0
line_v_pass=0
line_fche_executed=0
line_d_v1411_route=R8-NonRATSubstrateStillMissing
line_d_v1411_best_family_dataset_seed_pass_count=0
official_s5_reached=0
promotion_allowed=0
required_artifact_missing_count=0
manifest_missing_sum=0
```

本次仍不启动新训练；这是重复状态复核，不产生新实验数据。
