# DG-KAN v14.12.1 FunctionalContinueOpen TransferObservability AllBasis 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md
runner = experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py
out_dir = results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1
recap = docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_实验结果复盘.md
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 120 --real-lite-batch-size 32 --real-lite-fms-update-interval 120 --real-lite-trace-interval 60 --real-lite-linec-seeds 12319500 --force-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 1 --force-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_strength002_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --real-lite-fms-strength 0.02 --force-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_strength010_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --real-lite-fms-strength 0.10 --force-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/line_d_replay_v149_exact_best_v1412_1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU26-NoMaterializeLifetimeAuditV2,D-RBF25-WidthConditionGuardNoTaskBranch --train-size 256 --val-size 128 --batch-size 32 --epochs 20 --linec-seeds 12319500,12319501,12319502 --compute-budgeted-run 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 1 --micro-horizon-max-rows 210 --micro-horizon-steps 1,2,4 --force-micro-horizon-probe 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_d4_h2_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 2 --micro-horizon-max-rows 210 --micro-horizon-steps 1,2,4 --force-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/real_lite_d4_h4_steps200_linec3_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 4 --micro-horizon-max-rows 210 --micro-horizon-steps 1,2,4 --force-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py --out-dir results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --fdiag-micro-horizon-steps 1 --micro-horizon-max-rows 210 --micro-horizon-steps 1,2,4
```

## 3. 主要 artifact

```text
v1412_1_route_decision.json
v1412_1_progress_table.csv
v1412_1_provenance_audit.csv
v1412_1_no_action_search_audit.csv
v1412_1_forbidden_information_audit.csv
v1412_1_method_surface_manifest.csv
v1412_1_synthetic_real_predictivity.csv
v1412_1_train_stream_proxy_expansion.csv
v1412_1_micro_horizon_probe.csv
v1412_1_micro_horizon_validity.csv
v1412_1_real_lite_dche_fms_diagnostic.csv
v1412_1_real_lite_controls.csv
v1412_1_allbasis_reconciliation.csv
v1412_1_d_fou_hardening.csv
v1412_1_d_rbf_hardening.csv
v1412_1_d_wav_monitor.csv
v1412_1_dche_no_regression.csv
v1412_1_mlp_controls.csv
v1412_1_linec_tail_audit.csv
v1412_1_failure_taxonomy.csv
v1412_1_required_manifest.csv
v1412_1_code_review_packet.zip
```

## 4. 复现说明

```text
1. 使用 /home/chengshun.wang/miniconda3/envs/kan/bin/python 执行 runner。
2. 如需重跑 real-lite，请加 --force-real-lite 1。
3. 如需重跑 actual micro-horizon probe，请加 --force-micro-horizon-probe 1。
4. 如果只想复核现有 artifact，可不加 --force-real-lite，runner 会复用 v1412_1_real_lite_dche_fms_diagnostic.csv。
5. 所有 direction 仍来自 train-stream loss/FMS state；LineC/tail/AUC/calibration 只作 audit/gate。
```

## 5. 用户再次追问后的状态复核指令

本次没有新增训练；只读取最新 route / manifest / proxy / real-lite / all-basis reconciliation artifacts，确认计划 Case B/C/D/E 是否还有合法继续分支。

```bash
cat results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_route_decision.json

sed -n '1110,1165p' docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
base = 'results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1'
route = json.load(open(f'{base}/v1412_1_route_decision.json'))
manifest = list(csv.DictReader(open(f'{base}/v1412_1_required_manifest.csv')))
proxy = list(csv.DictReader(open(f'{base}/v1412_1_train_stream_proxy_expansion.csv')))
real = list(csv.DictReader(open(f'{base}/v1412_1_real_lite_dche_fms_diagnostic.csv')))
recon = list(csv.DictReader(open(f'{base}/v1412_1_allbasis_reconciliation.csv')))
print('route', route['route'])
print('manifest_missing_sum', sum(int(float(r.get('missing') or 0)) for r in manifest))
print('proxy_best', max(proxy, key=lambda r: max(float(r.get('auc_to_selected_pass') or 0), float(r.get('auc_to_synthetic_pass') or 0), float(r.get('auc_to_real_pass') or 0)))['proxy_name'])
print('proxy_rows', len(proxy))
print('real_pass', len({(r.get('dataset'), r.get('seed')) for r in real if int(float(r.get('strict_gate_pass') or 0)) == 1}))
print('line_d_best', max((int(float(r.get('v1412_replay_dataset_seed_pass_count') or r.get('v1411_dataset_seed_pass_count') or 0)), r.get('family')) for r in recon if r.get('family') != 'D-CHE'))
PY

tail -90 docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_实验结果复盘.md
```

本次复核日志写入后，刷新 code review packet / required manifest：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from experiments.run_v1412_1_functional_continue_open_transfer_observability_all_basis import ROOT, make_packet, write_required_manifest
out = ROOT / 'results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1'
make_packet(out)
write_required_manifest(out)
make_packet(out)
write_required_manifest(out)
print(out / 'v1412_1_code_review_packet.zip')
print(out / 'v1412_1_required_manifest.csv')
PY
```

## 6. 用户再次追问后的二次状态复核指令

本次没有新增训练；读取计划 Case B/C/D/E 和最新 artifacts，确认当前是否仍有可继续执行且不越界的分支。

```bash
rg -n "Case B|Case C|Case D|Case E|停止|stop|go|promotion|F-CHE|action|controller|reset|Line D|Line V|Line E" docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
base = 'results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1'
route = json.load(open(f'{base}/v1412_1_route_decision.json'))
manifest = list(csv.DictReader(open(f'{base}/v1412_1_required_manifest.csv')))
proxy = list(csv.DictReader(open(f'{base}/v1412_1_train_stream_proxy_expansion.csv')))
real = list(csv.DictReader(open(f'{base}/v1412_1_real_lite_dche_fms_diagnostic.csv')))
recon = list(csv.DictReader(open(f'{base}/v1412_1_allbasis_reconciliation.csv')))
print(json.dumps({
  'route': route.get('route'),
  'minimum_success': route.get('minimum_success'),
  'line_e2_exploration_gate_pass': route.get('line_e2_exploration_gate_pass'),
  'line_v2_best_auc': route.get('line_v2_best_auc'),
  'line_v2_exploration_gate_pass': route.get('line_v2_exploration_gate_pass'),
  'line_v2_micro_horizon_probe_rows': route.get('line_v2_micro_horizon_probe_rows'),
  'real_lite_dataset_seed_pass_count': route.get('real_lite_dataset_seed_pass_count'),
  'line_d_best_non_dche_dataset_seed_pass_count': route.get('line_d_best_non_dche_dataset_seed_pass_count'),
  'official_s5_reached': route.get('official_s5_reached'),
  'promotion_allowed': route.get('promotion_allowed'),
  'required_artifact_missing_count': route.get('required_artifact_missing_count'),
  'forbidden_information_violation_count': route.get('forbidden_information_violation_count'),
  'no_action_search_violation_count': route.get('no_action_search_violation_count'),
  'manifest_missing_sum': sum(int(float(r.get('missing') or 0)) for r in manifest),
  'proxy_rows': len(proxy),
  'real_strict_unique_pass': len({(r.get('dataset'), r.get('seed')) for r in real if int(float(r.get('strict_gate_pass') or 0)) == 1}),
  'best_non_dche_replay': max((int(float(r.get('v1412_replay_dataset_seed_pass_count') or r.get('v1411_dataset_seed_pass_count') or 0)), r.get('family')) for r in recon if r.get('family') != 'D-CHE'),
}, indent=2, sort_keys=True))
PY

tail -80 docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_实验结果复盘.md
tail -60 docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_执行日志.md
```

## 7. 用户再次追问后的三次状态复核指令

本次没有新增训练；重新读取计划原文中 Line F-Diag、Line D、Case B/C/D/E、route definitions，并复核 hardening artifacts 是否覆盖允许候选。

```bash
sed -n '600,760p' docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md
sed -n '1090,1165p' docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md
sed -n '980,1045p' docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md
ls -1 results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1 | sed -n '1,120p'

sed -n '740,860p' docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md
sed -n '1040,1090p' docs/DG-KAN_v14.12.1_FunctionalContinueOpen_TransferObservability_AllBasis_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv
base = 'results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1'
for name in [
    'v1412_1_d_fou_hardening.csv',
    'v1412_1_d_rbf_hardening.csv',
    'v1412_1_d_wav_monitor.csv',
    'v1412_1_allbasis_reconciliation.csv',
    'v1412_1_dche_no_regression.csv',
    'v1412_1_failure_taxonomy.csv',
]:
    path = f'{base}/{name}'
    rows = list(csv.DictReader(open(path)))
    print(name, 'rows', len(rows))
    if name == 'v1412_1_allbasis_reconciliation.csv':
        for r in rows:
            print(r.get('family'), r.get('v1412_replay_dataset_seed_pass_count'), r.get('official_fms_eligibility'))
PY
```

## 8. 用户再次追问后的代码层方法面复核指令

本次没有新增训练；读取 runner 中注册的 FMS-D alias / Line D candidates，并检查 artifact method/candidate 覆盖范围。

```bash
rg -n "FMS-D[1-5]|D-FOU|D-RBF|D-WAV|REAL|F_DIAG|micro_horizon|candidate" experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv
base = 'results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1'
for name in [
    'v1412_1_method_surface_manifest.csv',
    'v1412_1_real_lite_dche_fms_diagnostic.csv',
    'v1412_1_real_lite_controls.csv',
    'v1412_1_d_fou_hardening.csv',
    'v1412_1_d_rbf_hardening.csv',
    'v1412_1_d_wav_monitor.csv',
]:
    rows = list(csv.DictReader(open(f'{base}/{name}')))
    print('\\n' + name, 'rows', len(rows))
    keys = rows[0].keys() if rows else []
    for col in ['method', 'method_name', 'candidate_id', 'family', 'dataset', 'seed',
                'strict_gate_pass', 'promotion_allowed', 'official_fms_proof_executed']:
        if col in keys:
            vals = sorted({r.get(col, '') for r in rows})
            print(col, vals[:20], '... total', len(vals))
PY

cat results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_no_go_boundary.md
cat results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1/v1412_1_next_hypothesis_queue.md
```
