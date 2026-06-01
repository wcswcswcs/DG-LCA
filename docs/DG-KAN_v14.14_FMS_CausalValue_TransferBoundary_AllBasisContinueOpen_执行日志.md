# DG-KAN v14.14 FMS CausalValue TransferBoundary AllBasisContinueOpen 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
runner = experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py
out_dir = results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414
recap = docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_实验结果复盘.md
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py --out-dir results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/smoke_v1414 --device cpu --real-lite-datasets MNIST --real-lite-seeds 0 --real-lite-train-size 64 --real-lite-val-size 32 --real-lite-test-size 32 --real-lite-train-steps 4 --real-lite-batch-size 8 --real-lite-linec-batch-size 8 --real-lite-linec-sketch-dim 4 --force-boundary-real-lite 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py --out-dir results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414 --device cuda:0 --force-boundary-real-lite 1
```

## 3. 主要 artifacts

```text
v1414_route_decision.json
v1414_required_artifact_manifest.csv
v1414_forbidden_information_audit.csv
v1414_no_action_search_audit.csv
v1414_method_surface_manifest.csv
v1414_line_r_audit.csv
v1414_e4_observability_deconfound.csv
v1414_e4_leaveout_summary.csv
v1414_q_fms_vs_matched_controls.csv
v1414_q_control_equivalence_summary.csv
v1414_b_boundary_fms_real_lite.csv
v1414_b_boundary_controls.csv
v1414_d_cross_version_reconciliation.csv
v1414_d_all_basis_substrate_hardening.csv
v1414_linec_tail_audit.csv
v1414_failure_taxonomy.csv
v1414_no_go_boundary.md
v1414_next_hypothesis_queue.md
v1414_code_review_packet.zip
```

## 4. 复现说明

```text
1. 使用 /home/chengshun.wang/miniconda3/envs/kan/bin/python 执行 runner。
2. 如需重跑 boundary real-lite，请加 --force-boundary-real-lite 1。
3. 若不加 --force-boundary-real-lite，runner 会复用 v1414_b_boundary_fms_real_lite.csv。
4. B1/B2/B3 是预注册 boundary mechanism，不是 FMS-M6/M7，也不是 F-CHE8/F-CHE9。
5. 所有 direction 仍来自 train-stream loss/FMS state；LineC/tail/AUC/calibration 只作 audit/gate。
```

## 5. 运行结果索引

```text
smoke route =
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/smoke_v1414/v1414_route_decision.json

official route =
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414/v1414_route_decision.json

official boundary rows =
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414/v1414_b_boundary_fms_real_lite.csv
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414/v1414_b_boundary_controls.csv

official audits =
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414/v1414_required_artifact_manifest.csv
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414/v1414_forbidden_information_audit.csv
  results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414/v1414_no_action_search_audit.csv
```

## 6. 关键结果

```text
route = R4-AllBasisSubstrateBlocked
E4 best feature = E4-K-recovery_lag
E4 raw AUC = 0.9090909090909091
E4 leaveout min = 0.5
Q control_equivalent_fraction = 0.8888888888888888
Boundary real-lite pass = 0/9
Line D best non-D-CHE = D-FOU 2/9
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
promotion_allowed = 0
```

## 7. 用户再次追问后的复核指令

```bash
rg -n "Line B|Line F|Line D|Allowed definitions|Allowed substrate|NeedsFreshHardening|Do not|不允许|必须|Codex must|Gate|route|R4|R5|B1|B2|B3" docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
python - <<'PY'
import csv,json
from pathlib import Path
out=Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
print(json.dumps(json.load((out/'v1414_route_decision.json').open()), indent=2, sort_keys=True))
for name in [
    'v1414_method_surface_manifest.csv',
    'v1414_b_boundary_fms_real_lite.csv',
    'v1414_b_boundary_controls.csv',
    'v1414_d_cross_version_reconciliation.csv',
    'v1414_d_all_basis_substrate_hardening.csv',
    'v1414_required_artifact_manifest.csv',
]:
    rows=list(csv.DictReader((out/name).open()))
    print(name, len(rows))
PY
```

复核结果：

```text
B/F method rows = 27
B/F control rows = 54
Line D substrate hardening rows = 135
required manifest missing sum = 0
未发现 v14.14 计划允许但漏跑的 F-B method / control / Line D candidate。
```

## 8. 用户再次追问后的代码层覆盖复核指令

本次未新增训练，只做代码层 method surface 与 artifact 覆盖核验。

```bash
rg -n "^## |route =|manifest_missing_sum|forbidden|no-action|当前合法结论|停止|覆盖|F-B|G4|G5" docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_实验结果复盘.md
rg -n "BOUNDARY_ALIASES|Q_GROUPS|G4|G5|build_line_d|write_required_manifest|make_packet|route" experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py
sed -n '573,742p' docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
sed -n '742,986p' docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
sed -n '986,1020p' docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
sed -n '80,120p' experiments/run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open.py
```

```bash
python - <<'PY'
from pathlib import Path
import csv, json
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')

def rows(name):
    with (out / name).open(newline='') as f:
        return list(csv.DictReader(f))

def uniq(name, col):
    return sorted({r.get(col, '') for r in rows(name) if r.get(col, '')})

route = json.loads((out / 'v1414_route_decision.json').read_text())
manifest = rows('v1414_required_artifact_manifest.csv')
forbidden = rows('v1414_forbidden_information_audit.csv')
no_action = rows('v1414_no_action_search_audit.csv')
summary = {
    'route': route.get('route'),
    'minimum_success': route.get('minimum_success'),
    'e4_exploration_gate_pass': route.get('e4_exploration_gate_pass'),
    'q_causal_exploration_gate_pass': route.get('q_causal_exploration_gate_pass'),
    'q_route_hint': route.get('q_route_hint'),
    'boundary_real_lite_pass_count': route.get('boundary_real_lite_pass_count'),
    'boundary_mean_source_vs_best_control': route.get('boundary_mean_source_vs_best_control'),
    'line_d_best_non_dche_dataset_seed_pass_count': route.get('line_d_best_non_dche_dataset_seed_pass_count'),
    'line_d_official_fms_eligible_family_count': route.get('line_d_official_fms_eligible_family_count'),
    'official_s5_reached': route.get('official_s5_reached'),
    'promotion_allowed': route.get('promotion_allowed'),
    'manifest_missing_sum': sum(int(float(r.get('missing', 0) or 0)) for r in manifest),
    'forbidden_violation_sum': sum(int(float(r.get('violation_count', 0) or 0)) for r in forbidden),
    'no_action_violation_sum': sum(int(float(r.get('violation_count', 0) or 0)) for r in no_action),
    'method_surface_rows': len(rows('v1414_method_surface_manifest.csv')),
    'method_surface_methods': uniq('v1414_method_surface_manifest.csv', 'method'),
    'q_rows': len(rows('v1414_q_fms_vs_matched_controls.csv')),
    'q_groups': uniq('v1414_q_fms_vs_matched_controls.csv', 'group'),
    'boundary_rows': len(rows('v1414_b_boundary_fms_real_lite.csv')),
    'boundary_methods': uniq('v1414_b_boundary_fms_real_lite.csv', 'method'),
    'boundary_control_rows': len(rows('v1414_b_boundary_controls.csv')),
    'boundary_controls': uniq('v1414_b_boundary_controls.csv', 'method'),
    'line_d_rows': len(rows('v1414_d_all_basis_substrate_hardening.csv')),
    'line_d_candidates': uniq('v1414_d_all_basis_substrate_hardening.csv', 'candidate_id'),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e4_exploration_gate_pass = 0
q_causal_exploration_gate_pass = 0
q_route_hint = R3-FMSDirectionControlEquivalent
boundary_real_lite_pass_count = 0/9
boundary_mean_source_vs_best_control = -0.032401322214691726
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0

v1414_method_surface_manifest.csv rows = 11
v1414_q_fms_vs_matched_controls.csv rows = 189
v1414_b_boundary_fms_real_lite.csv rows = 27
v1414_b_boundary_controls.csv rows = 54
v1414_d_all_basis_substrate_hardening.csv rows = 135
Line D candidate_id count = 15
```

覆盖结论：

```text
F-B1/F-B2/F-B3 已执行。
C0..C5 controls 已执行。
Q groups G0..G8 已输出，其中 G4/G5 是 matched controls。
D-FOU26..31、D-RBF25..29、D-WAV25..28 已合入 Line D hardening artifact。
没有发现当前计划允许但漏跑的分支。
本次不启动新的 v14.14 训练。
```

## 9. 日志更新后的 packet / manifest 刷新

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments import run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open as r
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
r.make_packet(out)
r.write_required_manifest(out)
PY
```

刷新后核验：

```bash
python - <<'PY'
from pathlib import Path
import csv, json
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
route = json.loads((out / 'v1414_route_decision.json').read_text())

def rows(name):
    with (out / name).open(newline='') as f:
        return list(csv.DictReader(f))

print('route=', route.get('route'))
print('official_s5_reached=', route.get('official_s5_reached'))
print('promotion_allowed=', route.get('promotion_allowed'))
print('manifest_missing_sum=', sum(int(float(r.get('missing', 0) or 0)) for r in rows('v1414_required_artifact_manifest.csv')))
print('forbidden_violation_sum=', sum(int(float(r.get('violation_count', 0) or 0)) for r in rows('v1414_forbidden_information_audit.csv')))
print('no_action_violation_sum=', sum(int(float(r.get('violation_count', 0) or 0)) for r in rows('v1414_no_action_search_audit.csv')))
PY
```

结果：

```text
route = R4-AllBasisSubstrateBlocked
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0
```

## 10. 用户再次追问后的二次状态复核指令

本次未新增代码修改，也未启动新训练。

```bash
python - <<'PY'
from pathlib import Path
import csv, json
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
route = json.loads((out / 'v1414_route_decision.json').read_text())

def rows(name):
    with (out / name).open(newline='') as f:
        return list(csv.DictReader(f))

def uniq(name, col):
    return sorted({r.get(col, '') for r in rows(name) if r.get(col, '')})

print(json.dumps({
    'route': route.get('route'),
    'minimum_success': route.get('minimum_success'),
    'e4_best_auc_mean': route.get('e4_best_auc_mean'),
    'e4_best_leaveout_auc_min': route.get('e4_best_leaveout_auc_min'),
    'e4_exploration_gate_pass': route.get('e4_exploration_gate_pass'),
    'q_control_equivalent_fraction': route.get('q_control_equivalent_fraction'),
    'q_causal_exploration_gate_pass': route.get('q_causal_exploration_gate_pass'),
    'boundary_real_lite_pass_count': route.get('boundary_real_lite_pass_count'),
    'boundary_mean_source_vs_best_control': route.get('boundary_mean_source_vs_best_control'),
    'line_d_best_non_dche_dataset_seed_pass_count': route.get('line_d_best_non_dche_dataset_seed_pass_count'),
    'line_d_official_fms_eligible_family_count': route.get('line_d_official_fms_eligible_family_count'),
    'official_s5_reached': route.get('official_s5_reached'),
    'promotion_allowed': route.get('promotion_allowed'),
    'manifest_missing_sum': sum(int(float(r.get('missing', 0) or 0)) for r in rows('v1414_required_artifact_manifest.csv')),
    'forbidden_violation_sum': sum(int(float(r.get('violation_count', 0) or 0)) for r in rows('v1414_forbidden_information_audit.csv')),
    'no_action_violation_sum': sum(int(float(r.get('violation_count', 0) or 0)) for r in rows('v1414_no_action_search_audit.csv')),
    'boundary_methods': uniq('v1414_b_boundary_fms_real_lite.csv', 'method'),
    'boundary_controls': uniq('v1414_b_boundary_controls.csv', 'method'),
    'q_groups': uniq('v1414_q_fms_vs_matched_controls.csv', 'group'),
    'line_d_candidate_id_count': len(uniq('v1414_d_all_basis_substrate_hardening.csv', 'candidate_id')),
    'line_d_candidate_ids': uniq('v1414_d_all_basis_substrate_hardening.csv', 'candidate_id'),
}, indent=2, ensure_ascii=False))
PY

rg -n "Line B 不允许|Allowed definitions|Line D non-D-CHE|Hard stop|Codex must not|B4|FMS-M6|F-CHE8|action token|reset route|S4-lite|S5-Official|R4-AllBasis" docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
```

结果摘要：

```text
route = R4-AllBasisSubstrateBlocked
e4_best_auc_mean = 0.9090909090909091
e4_best_leaveout_auc_min = 0.5
e4_exploration_gate_pass = 0
q_control_equivalent_fraction = 0.8888888888888888
q_causal_exploration_gate_pass = 0
boundary_real_lite_pass_count = 0/9
boundary_mean_source_vs_best_control = -0.032401322214691726
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0

boundary methods = F-B1/F-B2/F-B3
boundary controls = C0..C5
Q groups = G0..G8
Line D candidate_id count = 15
```

二次复核结论：

```text
未发现 v14.14 计划允许但漏跑的预注册 method/control/substrate candidate。
本次不启动新的 v14.14 训练。
```

## 11. 用户再次追问后的 Line Z / no-go 三次复核指令

本次未新增代码修改，也未启动新训练；只读取 route definitions、Line Z artifacts、manifest 与 audit artifacts。

```bash
rg -n "Route definitions|S0-|S3-|S4-lite|S4-Exploration|S5-|R0-|R1-|R2-|R3-|R4-|R5-|Line Z|no-go|next hypothesis|current method family|legal fallback|必须进入下一版|不允许|Codex must not" docs/DG-KAN_v14.14_FMS_CausalValue_TransferBoundary_AllBasisContinueOpen_完整计划.md
```

```bash
python - <<'PY'
from pathlib import Path
import csv, json
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
route = json.loads((out/'v1414_route_decision.json').read_text())

def rows(name):
    with (out/name).open(newline='') as f:
        return list(csv.DictReader(f))

required = rows('v1414_required_artifact_manifest.csv')
print(json.dumps({
  'route': route.get('route'),
  'minimum_success': route.get('minimum_success'),
  'e4_exploration_gate_pass': route.get('e4_exploration_gate_pass'),
  'q_causal_exploration_gate_pass': route.get('q_causal_exploration_gate_pass'),
  'q_route_hint': route.get('q_route_hint'),
  'boundary_exploration_gate_pass': route.get('boundary_exploration_gate_pass'),
  'boundary_meaningful_gate_pass': route.get('boundary_meaningful_gate_pass'),
  'boundary_s4_gate_pass': route.get('boundary_s4_gate_pass'),
  'boundary_real_lite_pass_count': route.get('boundary_real_lite_pass_count'),
  'line_d_best_non_dche_dataset_seed_pass_count': route.get('line_d_best_non_dche_dataset_seed_pass_count'),
  'line_d_official_fms_eligible_family_count': route.get('line_d_official_fms_eligible_family_count'),
  'official_s5_reached': route.get('official_s5_reached'),
  'promotion_allowed': route.get('promotion_allowed'),
  'required_artifact_rows': len(required),
  'required_artifact_missing_count': route.get('required_artifact_missing_count'),
  'manifest_missing_sum': sum(int(float(r.get('missing', 0) or 0)) for r in required),
  'forbidden_violation_sum': sum(int(float(r.get('violation_count', 0) or 0)) for r in rows('v1414_forbidden_information_audit.csv')),
  'no_action_violation_sum': sum(int(float(r.get('violation_count', 0) or 0)) for r in rows('v1414_no_action_search_audit.csv')),
}, indent=2, ensure_ascii=False))
print('no_go_boundary:')
print((out/'v1414_no_go_boundary.md').read_text())
print('next_hypothesis_queue:')
print((out/'v1414_next_hypothesis_queue.md').read_text())
PY
```

```bash
python - <<'PY'
from pathlib import Path
import csv
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
for name, col in [
  ('v1414_method_surface_manifest.csv','method'),
  ('v1414_b_boundary_fms_real_lite.csv','method'),
  ('v1414_b_boundary_controls.csv','method'),
  ('v1414_q_fms_vs_matched_controls.csv','group'),
  ('v1414_d_all_basis_substrate_hardening.csv','candidate_id'),
]:
    with (out/name).open(newline='') as f:
        rows=list(csv.DictReader(f))
    vals=sorted({r.get(col,'') for r in rows if r.get(col,'')})
    print(name, 'rows=', len(rows), col, 'count=', len(vals))
    print('\n'.join('  '+v for v in vals))
PY
```

结果摘要：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e4_exploration_gate_pass = 0
q_causal_exploration_gate_pass = 0
q_route_hint = R3-FMSDirectionControlEquivalent
boundary_exploration_gate_pass = 0
boundary_meaningful_gate_pass = 0
boundary_s4_gate_pass = 0
boundary_real_lite_pass_count = 0/9
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_rows = 29
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0

v1414_method_surface_manifest.csv rows = 11
v1414_b_boundary_fms_real_lite.csv rows = 27
v1414_b_boundary_controls.csv rows = 54
v1414_q_fms_vs_matched_controls.csv rows = 189
v1414_d_all_basis_substrate_hardening.csv rows = 135
Line D candidate_id count = 15
```

三次复核结论：

```text
v14.14 已执行 Line R/E4/Q/B/F/D/C/Z。
所有预注册 fallback 已覆盖。
当前计划内没有可继续执行且不越界的分支。
本次不启动新的 v14.14 训练。
```

刷新 code review packet / manifest：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments import run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open as r
out = Path('results/v14_14_fms_causal_value_transfer_boundary_all_basis_continue_open/official_v1414')
r.make_packet(out)
r.write_required_manifest(out)
PY
```
