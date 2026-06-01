# DG-KAN v13.0 StrategicReset FunctionPreservingCoordinateTransport 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志记录实际执行过的命令、输入 artifact、输出文件和 blocker 修复，方便后续复现。不把 smoke/diagnostic/skipped row 写成 promotion。

## 1. 计划读取

读取计划文件：

```bash
sed -n '1,260p' docs/DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md
sed -n '260,620p' docs/DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md
sed -n '620,860p' docs/DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md
```

同时读取 v12.36 作为前序 context：

```bash
sed -n '1,260p' docs/DG-KAN_v12.36_SubstrateHealth_ChannelColocation_FunctionalRepair_完整计划.md
sed -n '520,1140p' docs/DG-KAN_v12.36_SubstrateHealth_ChannelColocation_FunctionalRepair_完整计划.md
```

结论：本轮按用户显式提到的 v13.0 strategic reset 执行，v12.36/v12.35 仅作为背景和可复用 artifact。v13.0 不继续扩 B-RAT/B-FOU/M-J token，而是执行：

```text
Line S: Substrate Gate v2
Line X: Synthetic controlled mechanism proof
Line G: Function-preserving coordinate transport
Line M: MLP analog transport control
Line C: audit only
Line Z: decision/finalizer
```

## 2. 本轮代码修改

新增：

```text
experiments/run_v130_function_preserving_transport.py
```

实现内容：

```text
1. 从 v12.35 `v1235_basis_substrate_health.csv` 重算 v13.0 S1/S2 substrate gate。
2. 在 synthetic controlled feature space 上执行 function-preserving coordinate transport。
3. 在真实 MNIST 小规模 probe 上执行 frozen readout-feature transport。
4. 同时执行 MLP hidden-feature analog control。
5. 生成 v13.0 要求的 CSV/JSON/SVG/zip artifact。
```

审计边界：

```text
1. transport direction 只使用 unlabeled features 与 current logits，不使用 label/CE/validation/future。
2. label 只用于普通 checkpoint training 与 future-training probe。
3. 当前实现是 readout-feature coordinate transport，不是完整 basis-parameter surgery；route 会显式写 `full_basis_param_transport_executed=0`。
4. CE/NLL/ECE/CEp99 只作为 audit/future probe 坏化约束。
```

## 3. py_compile 与 smoke blocker

语法审计：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
```

结果：pass。

首次 smoke：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130 --synthetic-tasks X1,X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --synthetic-feature-dim 32 --future-steps 10,20 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 10 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
```

结果：失败，真实 KAN frozen feature 的 ridge compensation 在线性求解时触发 singular matrix：

```text
torch._C._LinAlgError: torch.linalg.solve: The solver failed because the input matrix is singular.
```

按 v13.0 fallback “增大 ridge compensation / more stable compensation solve”修复：

```text
experiments/run_v130_function_preserving_transport.py
  ridge_solve() 增加 nan/inf guard；
  依次尝试 ridge scale 1/10/100/1000；
  仍失败时 fallback 到 pinv/lstsq。
```

合理性：这是 function preservation compensation 的数值稳定修复，不改变 gate，不使用 label/CE direction。

第二次 smoke 继续执行到 code packet 阶段后失败：

```text
ValueError: 'results/.../v130_substrate_gate_v2.csv' is not in the subpath of '/home/chengshun.wang/DG-LCA' OR one path is relative and the other is absolute.
```

修复：

```text
experiments/run_v130_function_preserving_transport.py
  run() 中 out_dir/source_dir 统一使用 Path(...).resolve()。
```

合理性：这是 artifact packaging 路径修复，不改变 gate，不改变实验结果。

修复后重跑 py_compile + smoke：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130 --synthetic-tasks X1,X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --synthetic-feature-dim 32 --future-steps 10,20 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 10 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
```

结果：

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

解释：smoke 只验证 runner/artifact 路径，不能 promotion。小规模 smoke 已经显示 synthetic mechanism proof 未打开。

## 4. official v13.0 run

正式运行命令：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

输出目录：

```text
results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130
```

关键 route：

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

artifact 复核：

```bash
python - <<'PY'
import csv, json, pathlib
base = pathlib.Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130')
route = json.loads((base / 'v130_route_decision.json').read_text())
print(route)
rows = list(csv.DictReader((base / 'v130_required_artifact_manifest.csv').open()))
print(len(rows), sum(int(float(r.get('missing', 0) or 0)) for r in rows))
PY
```

复核结果：

```text
required_manifest_rows = 25
missing_rows = 0
```

主要产物：

```text
v130_route_decision.json
v130_required_artifact_manifest.csv
v130_substrate_gate_v2.csv
v130_synthetic_mechanism_proof.csv
v130_function_preservation.csv
v130_future_training_probe.csv
v130_transport_controls.csv
v130_transport_proposals.csv
v130_basis_telemetry_before_after.csv
v130_mlp_analog_transport.csv
v130_linec_audit.csv
v130_failure_table.csv
v130_family_decision.csv
v130_next_hypothesis_queue.md
v130_code_review_packet.zip
fig_substrate_map_by_family.svg
fig_synthetic_mechanism_success_matrix.svg
fig_transport_function_drift_vs_geom_gain.svg
fig_future_auc_transport_vs_controls.svg
fig_mlp_analog_vs_basis_transport.svg
fig_family_go_nogo_dashboard.svg
```

## 5. 聚合审计命令

substrate / transport / future-probe 聚合审计：

```bash
python - <<'PY'
import csv, pathlib
from collections import defaultdict
base = pathlib.Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130')

rows = list(csv.DictReader((base / 'v130_substrate_gate_v2.csv').open()))
for fam in sorted({r['family'] for r in rows}):
    rr = [r for r in rows if r['family'] == fam]
    print(fam, len(rr),
          sum(int(float(r.get('S1_efficient_substrate', 0) or 0)) for r in rr),
          sum(int(float(r.get('S2_healthy_base', 0) or 0)) for r in rr))

rows = list(csv.DictReader((base / 'v130_function_preservation.csv').open()))
by = defaultdict(lambda: [0, 0])
for r in rows:
    key = (r['transport_scope'], r['transport_name'])
    by[key][0] += 1
    by[key][1] += int(float(r['transport_accepted']))
for key, value in sorted(by.items()):
    print(key, value)

rows = list(csv.DictReader((base / 'v130_future_training_probe.csv').open()))
print('future_rows', len(rows),
      'future_pass', sum(int(float(r.get('future_training_probe_pass', 0) or 0)) for r in rows))

rows = list(csv.DictReader((base / 'v130_family_decision.csv').open()))
for r in rows:
    print(r['family'], r['family_route'])
PY
```

结果摘要：

```text
Substrate S1/S2:
D-CHE rows=8 S1=0 S2=0
D-FOU rows=8 S1=0 S2=0
D-RAT rows=16 S1=14 S2=0
D-RBF rows=6 S1=0 S2=0
D-WAV rows=6 S1=0 S2=0

FamilyTransport accepted:
synthetic_controlled_feature_space FamilyTransport rows=42 accepted=0
real_basis_frozen_readout_feature_transport FamilyTransport rows=2 accepted=0
real_mlp_hidden_feature_analog FamilyTransport rows=1 accepted=0

Future probe:
future_training_probe_rows=528
future_training_probe_pass_rows=0

Family decision:
D-CHE NoEfficientSubstrate
D-FOU NoEfficientSubstrate
D-RAT SyntheticMechanismNoGo
D-RBF NoEfficientSubstrate
D-WAV NoEfficientSubstrate
MLP NoEfficientSubstrate
```

## 6. final artifact refresh

为使 `v130_code_review_packet.zip` 收录本执行日志和复盘日志中的 official 结果，日志写入后重跑同一 official 命令刷新 artifact 包：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

预期：同一 deterministic 配置下 route 与核心指标不变；仅 code packet 重新打包当前日志与代码/CSV。

## 7. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.0 是否达成目标，若未达成则继续。本次按计划文件 stop/go 段与最终 route 重新复核。

计划文件定位命令：

```bash
nl -ba docs/DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md | sed -n '630,770p'
```

关键计划规则：

```text
Line Z Go 条件要求：
1. S1 EfficientSubstrate pass；
2. Synthetic mechanism proof pass；
3. function-preserving transport pass；
4. future training probe pass；
5. controls pass；
6. provenance pass。

No-go 条件：
synthetic mechanism proof fail；
或 function-preserving transport impossible；
或 future training probe repeatedly no benefit；
或 MLP analog explains all benefit；
则该 family 的 functional transport 暂停。

Codex failure fallback：
如果 all families fail synthetic proof，Codex 必须停止 functional transport，
回到 substrate design，不允许继续扩 token。
```

最终 route 复核命令：

```bash
python - <<'PY'
import json, csv, pathlib
base = pathlib.Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130')
route = json.loads((base / 'v130_route_decision.json').read_text())
for k in [
    'route','route_detail','minimum_success','official_success_reached',
    'promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted',
    'fallback_all_executed','required_artifact_missing_count',
    'substrate_s1_count','substrate_s2_healthy_base_count',
    'synthetic_rows','synthetic_mechanism_pass_count',
    'future_training_probe_rows','real_basis_feature_transport_pass_count',
    'mlp_analog_transport_pass_count','generic_reparameterization_effect',
    'KAN_specific_claim_allowed','full_basis_param_transport_executed',
    'provenance_violation_count','code_review_packet_sha256',
]:
    print(f'{k}={route.get(k)}')
rows = list(csv.DictReader((base / 'v130_required_artifact_manifest.csv').open()))
print(f"manifest_rows={len(rows)}")
print(f"missing_rows={sum(int(float(r.get('missing', 0) or 0)) for r in rows)}")
PY
```

复核结果：

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

本次不新增训练命令、不新增 CSV/JSON artifact、不修改代码或 gate。原因：v13.0 文档明确规定 all families fail synthetic proof 时必须停止 functional transport，不允许继续扩 token；当前 artifact 已满足该 no-go 条件。继续推进需要新的 substrate design 或真正 full basis-parameter coordinate surgery 机制，不属于继续排列当前 readout-feature transport token。

## 8. 用户再次追问后的 stop-contract 复核 2

用户再次要求“未达成则继续”。本次复核不新增训练命令，复用上一次读取计划与 route 的命令结果：

```bash
nl -ba docs/DG-KAN_v13.0_StrategicReset_FunctionPreservingCoordinateTransport.md | sed -n '630,770p'
python - <<'PY'
import json, pathlib
p = pathlib.Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_route_decision.json')
d = json.loads(p.read_text())
for k in ['route','minimum_success','promotion_allowed','final_stop_allowed',
          'synthetic_rows','synthetic_mechanism_pass_count',
          'full_basis_param_transport_executed','required_artifact_missing_count']:
    print(k, d.get(k))
PY
```

复核事实：

```text
plan_line_754_756 = 如果 all families fail synthetic proof，Codex 必须停止 functional transport，回到 substrate design，不允许继续扩 token。
route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
promotion_allowed = 0
final_stop_allowed = 1
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
full_basis_param_transport_executed = 0
required_artifact_missing_count = 0
```

执行决定：

```text
不启动新增 functional transport 实验；
不扩当前 readout-feature transport token；
不把未执行的 full basis-parameter coordinate surgery 写成已执行；
不改 gate。
```

原因：当前已经触发计划文件明确 no-go 条款。下一步若继续，需要新的 substrate design 或 full basis-parameter coordinate surgery 的可执行计划/实现；我目前不确定如何在现有代码路径内安全完成真正 basis-parameter surgery 并满足 function-preservation gate，因此不做低价值 token 网格搜索。

## 9. fallback2: affine / block-wise / scale-joint compensation

用户再次要求未达成则继续，并强调按计划文件推荐思路修复。本次重新审计代码后发现：上一版已实现 ridge retry / smaller alpha / readout-only / random / MLP analog / K sensitivity，但 `FamilyTransport` 仍缺少 v13.0 第 12 节列出的部分 fallback：

```text
1. block-wise readout compensation；
2. solve readout + scale jointly；
3. 更明确的 coordinate scale / tangent trust-region rescale。
```

代码修改：

```text
experiments/run_v130_function_preserving_transport.py
  新增 with_bias(phi)，用于仿射 readout compensation。
  新增 blockwise_ridge_solve(phi, target, ridge, blocks=4)。
  FamilyTransport 增加 compensation modes:
    global_no_bias
    affine_bias
    blockwise_affine
  FamilyTransport 增加 post_scale search:
    1.0,0.5,0.25,0.10,0.05,rms_match,sqrt(rms_match)
  FamilyTransport 增加更小 ridge:
    1e-8,1e-6,1e-4,1e-3,1e-2,1e-1,1.0
  selection score 加入 feature RMS penalty，避免无界坐标缩放。
```

审计说明：

```text
1. 这是 function-preservation compensation fallback，不改变 gate。
2. transport direction 仍只读取 unlabeled features/current logits。
3. label 仍只用于 checkpoint/future training probe。
4. CE/NLL/ECE/CEp99 仍只作为 future/audit 坏化约束，不进入方向。
```

py_compile + smoke 命令：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130_fallback3 --synthetic-tasks X1,X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --synthetic-feature-dim 32 --future-steps 10,20 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 10 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
```

smoke 结果：

```text
route = R2-SyntheticMechanismProofFailed
synthetic_rows = 4
synthetic_mechanism_pass_count = 0
required_artifact_missing_count = 0

FamilyTransport function preservation:
D-RAT X1 accepted=1, Drift_B=2.055e-07, Drift_Q=3.387e-07
D-RAT X2 accepted=1, Drift_B=1.694e-07, Drift_Q=2.643e-07
MLP X1 accepted=1, Drift_B=2.295e-07, Drift_Q=3.823e-07
MLP X2 accepted=1, Drift_B=2.338e-07, Drift_Q=3.133e-07

Future probe still failed in smoke:
best synthetic ControlGap examples were below gate or tail-constrained.
```

解释：fallback2 真实修复了 function-preservation drift，但 smoke 不能 promotion，且 future probe 仍未过。

## 10. official fallback2 run

正式重跑命令：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

最终 route：

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

通过的 synthetic row：

```text
family = D-FOU
task = X4
seed = 0
FunctionDrift_pass_rows = 1
transport_accepted_rows = 1
future_training_probe_pass_rows = 1
synthetic_mechanism_pass = 1
```

FamilyTransport accepted 聚合：

```text
synthetic D-CHE: 7/7 accepted
synthetic D-FOU: 7/7 accepted
synthetic D-RAT: 7/7 accepted
synthetic D-RBF: 7/7 accepted
synthetic D-WAV: 7/7 accepted
synthetic MLP: 7/7 accepted
real MLP analog: 1/1 accepted
real D-RAT basis feature transport: 0/2 accepted
```

最关键的 real-data blocker：

```text
D-RAT38 real FamilyTransport:
  Drift_B = 0.08626401424407959
  Drift_Q = 0.9629298448562622
  TangentCondition_ratio = 1.5338226445729857
  accepted = 0

D-RAT26 real FamilyTransport:
  Drift_B = 0.00014294836728367954
  Drift_Q = 0.9592770934104919
  TangentCondition_ratio = 1.543149351170232
  accepted = 0
```

解释：fallback2 将 synthetic function preservation 打开，并取得 1 个 synthetic mechanism proof；但真实 D-RAT feature transport 在 probe batch drift 和 tangent condition 上失败，且 `full_basis_param_transport_executed=0`，因此仍不能 promotion。

最终 code packet hash：

```text
code_review_packet_sha256 = 0b4c59f624cc4a266932db1bb9289b2a025d171c8bc11d7ac838ee93766ea4fa
```

## 11. fallback2 后的 final artifact refresh

为使 code packet 收录本次 fallback2 执行日志和复盘日志，日志更新后再次运行同一 official 命令刷新 artifact 包：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

预期：route 保持 `R3-SyntheticOnlyNoRealTransportBenefit`；仅 code packet hash 可能因日志更新而变化。

## 12. provenance repair: probe batch 不参与 transport selection

fallback2 后自查发现：`make_transport()` 的 candidate selection score 使用了 `drift_val`，这等价于使用 probe/query batch 辅助选择 transport。虽然 probe drift 作为 gate/audit 合法，但不能作为 direction/selection source。

修复：

```text
experiments/run_v130_function_preserving_transport.py
  FamilyTransport selection score 改为只使用：
    train-stream drift
    feature RMS penalty
    train-stream geometry
  probe/query drift 仍写入 artifact，并作为 accept gate/audit；
  不参与候选选择。

  provenance audit 新增：
    probe_batch_used_for_transport_selection = 0
```

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130_fallback4_noprobe_select --synthetic-tasks X1,X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --synthetic-feature-dim 32 --future-steps 10,20 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 10 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

最终 official route：

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

```text
D-CHE X4 seed=0, future K=1000, ControlGap=0.01751038993991528
D-FOU X4 seed=0, future K=1000, ControlGap=0.04716192028906989
```

Real D-RAT blocker：

```text
D-RAT38:
  Drift_B = 3.3582368814677466e-06
  Drift_Q = 1.4307606220245361
  TangentCondition_ratio = 1.6770742383008945
  accepted = 0

D-RAT26:
  Drift_B = 3.4927684282592963e-06
  Drift_Q = 4.8830108642578125
  TangentCondition_ratio = 0.8352663401283539
  accepted = 0
```

final code packet refresh command（只刷新 packet / manifest / route hash，不重跑实验）：

```bash
conda run -n kan python - <<'PY'
import json
from pathlib import Path
import experiments.run_v130_function_preserving_transport as r
out = Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130').resolve()
code_manifest, code_sha = r.write_code_packet(out)
_manifest, missing = r.write_manifest(out)
route_path = out / 'v130_route_decision.json'
route = json.loads(route_path.read_text())
route['code_review_packet_entries'] = len(code_manifest)
route['code_review_packet_sha256'] = code_sha
route['required_artifact_missing_count'] = missing
route_path.write_text(json.dumps(route, indent=2, sort_keys=True) + '\\n')
r.write_manifest(out)
print({'code_review_packet_entries': len(code_manifest), 'code_review_packet_sha256': code_sha, 'missing': missing})
PY
```

## 13. train-stream fit/select split fallback

触发原因：fallback4 虽然修复了 probe/query batch 参与 selection 的 provenance 问题，但真实 D-RAT 仍出现 train-stream drift 很小、probe drift 很大的过拟合式 transport。因此继续按计划的 function-preservation fallback 做 train-stream 内部 fit/select split。

代码修改：

```text
experiments/run_v130_function_preserving_transport.py
  新增 train_stream_fit_select_indices()。
  FamilyTransport 的 readout compensation 只在 train-stream fit split 上求解。
  selection score 使用 train-stream select drift、train drift、RMS penalty、geometry。
  probe/query drift 仍作为 artifact/gate 审计，不参与方向选择。
```

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130_fallback5_trainstream_split --synthetic-tasks X1,X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --synthetic-feature-dim 32 --future-steps 10,20 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 10 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download
```

结果：

```text
smoke route = R2-SyntheticMechanismProofFailed
smoke synthetic_rows = 4
smoke synthetic_mechanism_pass_count = 0

official route = R2-SyntheticMechanismProofFailed
minimum_success = S1-EfficientSubstrate
synthetic_rows = 42
synthetic_mechanism_pass_count = 0
future_training_probe_rows = 528
real_basis_feature_transport_pass_count = 0
mlp_analog_transport_pass_count = 0
provenance_violation_count = 0
required_artifact_missing_count = 0
```

解释：train-stream split 后 synthetic FamilyTransport 仍能 function-preserving accepted，但 future probe 全部为 0；这说明之前较宽松 R3 不够稳健。

## 14. future optimizer sensitivity fallback

触发原因：default future probe 下多个 row 的 AUC proxy 明显改善，但 CEp99 tail 爆掉，导致 gate 拒绝；计划 12 要求 future training 不改善时先做 K sensitivity / optimizer state or timing sensitivity / MLP analog / readout-only 对照。本 runner 已覆盖 K=50/200/1000、MLP analog、readout-only；本次追加低 LR future optimizer sensitivity，仍不改变 transport direction。

代码修改：

```text
experiments/run_v130_function_preserving_transport.py
  v130_future_training_probe.csv 新增 future_lr / future_weight_decay 字段。
  provenance audit 新增 future_optimizer_probe_as_direction=0。
```

诊断命令：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130_future_lr003_probe --synthetic-tasks X4,X6 --synthetic-families D-RAT,D-CHE,D-FOU,MLP --synthetic-seeds 0 --synthetic-train-size 256 --synthetic-val-size 128 --synthetic-feature-dim 64 --future-steps 50,200,1000 --future-lr 0.003 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 50 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
```

诊断结果：

```text
route = R3-SyntheticOnlyNoRealTransportBenefit
synthetic_rows = 8
synthetic_mechanism_pass_count = 1
synthetic pass = D-FOU X4 seed=0 K=1000 ControlGap=0.007501776195555615
real_basis_feature_transport_pass_count = 0
```

低 LR official 命令：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download --future-lr 0.003
```

低 LR official 结果：

```text
route = R3-SyntheticOnlyNoRealTransportBenefit
minimum_success = S2-SyntheticMechanismProof
synthetic_mechanism_pass_count = 1
synthetic pass = D-RAT X2 seed=0 K=200 ControlGap=0.01872296667919826
real_basis_feature_transport_pass_count = 0
full_basis_param_transport_executed = 0
```

解释：低 LR future probe 能恢复一个 D-RAT synthetic proof，但真实 D-RAT 仍没有 function-preserving feature transport，因此仍不能 promotion。

## 15. local hidden blend function-preservation fallback

触发原因：低 LR 后 D-RAT synthetic proof 已出现，但真实 D-RAT feature transport 仍因 function preservation fail 被拒绝。计划 12 的 function-preservation fallback 还包括 smaller transport step 与 local hidden reconstruction。本轮将 FamilyTransport 扩展为 hidden-feature blend：在原 hidden/readout feature 与 whitening coordinate transport 之间做预提交插值，再用同一 train-stream fit/select split 选择。

代码修改：

```text
experiments/run_v130_function_preserving_transport.py
  FamilyTransport 新增 transport_blend in (1.0,0.75,0.50,0.25,0.10,0.05)。
  fallback_note 写出 transport_blend。
  选择仍只用 train-stream split drift / train drift / RMS / geometry；
  不使用 probe/query/future/CE 选方向。
```

smoke 命令：

```bash
conda run -n kan python -m py_compile experiments/run_v130_function_preserving_transport.py
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/smoke_v130_hidden_blend --synthetic-tasks X2 --synthetic-families D-RAT,MLP --synthetic-seeds 0 --synthetic-train-size 128 --synthetic-val-size 64 --synthetic-feature-dim 64 --future-steps 50,200 --future-lr 0.003 --real-datasets MNIST --real-seeds 0 --real-train-size 64 --real-val-size 32 --real-checkpoint-epochs 1 --real-max-candidates 1 --real-future-steps 50 --batch-size 32 --mlp-hidden 32 --device cuda:0 --data-root data --no-download
```

smoke 结果：

```text
route = R2-SyntheticMechanismProofFailed
synthetic_rows = 2
real_basis_feature_transport_pass_count = 0
D-RAT38 smoke Drift_B = 0.2628291845321655
D-RAT38 smoke Drift_Q = 1.0001124143600464
D-RAT38 smoke TangentCondition_ratio = 0.7170520934977773
D-RAT38 smoke accepted = 0
```

hidden-blend official 命令：

```bash
conda run -n kan python experiments/run_v130_function_preserving_transport.py --out-dir results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130 --device cuda:0 --data-root data --no-download --future-lr 0.003
```

hidden-blend official 最终结果：

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
full_basis_param_transport_executed = 0
provenance_violation_count = 0
```

解释：hidden blend 是计划内 fallback，但它不能打开真实 D-RAT function preservation；同时在不允许用 future outcome 选 transport 的前提下，最终 synthetic mechanism proof 也回到 0。

final code packet refresh command（只刷新 packet / manifest / route hash，不重跑实验）：

```bash
conda run -n kan python -c "import json; from pathlib import Path; import experiments.run_v130_function_preserving_transport as r; out=Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130').resolve(); code_manifest, code_sha = r.write_code_packet(out); _manifest, missing = r.write_manifest(out); route_path = out / 'v130_route_decision.json'; route = json.loads(route_path.read_text()); route['code_review_packet_entries'] = len(code_manifest); route['code_review_packet_sha256'] = code_sha; route['required_artifact_missing_count'] = missing; route_path.write_text(json.dumps(route, indent=2, sort_keys=True) + '\\n'); r.write_manifest(out); print({'code_review_packet_entries': len(code_manifest), 'code_review_packet_sha256': code_sha, 'missing': missing})"
```

## 16. 用户再次追问后的 full-basis hook 源码复核

触发原因：用户再次要求确认 v13.0 是否达成目标，若未达成继续按计划文件推荐方向修复。本次不继续扩 readout-feature proxy 网格，而是检查是否可以在现有模型 API 中安全实现计划所需的 full basis-parameter coordinate surgery。

复核 route：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130/v130_route_decision.json')
route=json.loads(p.read_text())
keys=['route','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_all_executed','required_artifact_missing_count','substrate_s1_count','synthetic_mechanism_pass_count','real_basis_feature_transport_pass_count','mlp_analog_transport_pass_count','full_basis_param_transport_executed','provenance_violation_count','code_review_packet_sha256']
for k in keys:
    print(f'{k}={route.get(k)}')
PY
```

输出关键值：

```text
route=R2-SyntheticMechanismProofFailed
minimum_success=S1-EfficientSubstrate
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_all_executed=1
required_artifact_missing_count=0
substrate_s1_count=14
synthetic_mechanism_pass_count=0
real_basis_feature_transport_pass_count=0
mlp_analog_transport_pass_count=0
full_basis_param_transport_executed=0
provenance_violation_count=0
```

源码定位命令：

```bash
rg -n "def frozen_readout_features|def forward|class GroupedRationalKATKAN|readout_only_backbone_frozen|linear_readout|cross_readout" dgkan/models/fc_purekan_primitives.py
sed -n '3170,3295p' dgkan/models/fc_purekan_primitives.py
nl -ba dgkan/models/fc_purekan_primitives.py | sed -n '3172,3290p'
nl -ba experiments/run_v130_function_preserving_transport.py | sed -n '936,982p'
```

源码复核结果：

```text
1. GroupedRationalKATKAN.frozen_readout_features() 是后验拼接特征：
   h, hidden_square/abs/rat features, linear residual z, projected square/bilinear, cross/PCA/signal features, optional bias。
2. GroupedRationalKATKAN.forward() 不是统一矩阵读出，而是分别使用 w2、hidden_*_readout、linear_readout、proj_*_readout、cross_*_readout 等分支再相加。
3. experiments/run_v130_function_preserving_transport.py:943 显式 full_basis_param_transport_executed = 0。
4. build_route() 的 official gate 要求 full_basis_param_transport_executed == 1，因此当前 runner 无法 promotion。
```

审计判断：

```text
现有代码可以做 readout-feature proxy，也可以做非常窄的 hidden-channel permutation/near-scaling 尝试；
但 permutation 不改变实质几何，near-scaling 在 tanh + hidden_rat_residual + 多分支 readout 下不是可验证的 full basis-coordinate transport。
如果把 proxy 包成 wrapper，仍不是计划要求的 basis-parameter surgery。
因此本次不启动新增训练，以免把 readout-feature proxy 或不可审计的局部缩放写成 full-basis transport。
```

manifest/code packet 复核命令：

```bash
python - <<'PY'
import csv,json
from pathlib import Path
base=Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130')
for name in ['v130_required_artifact_manifest.csv','v130_code_review_manifest.csv']:
    rows=list(csv.DictReader((base/name).open()))
    print(name, 'rows', len(rows), 'missing', sum(str(r.get('exists','')).lower() in ('0','false') or str(r.get('missing','')).lower() in ('1','true') for r in rows))
route=json.loads((base/'v130_route_decision.json').read_text())
print('route', route['route'], 'sha', route.get('code_review_packet_sha256'))
PY
```

结果：

```text
v130_required_artifact_manifest.csv rows 25 missing 0
v130_code_review_manifest.csv rows 8 missing 0
route R2-SyntheticMechanismProofFailed
```

本次复核日志写入后的 code packet refresh：

```bash
conda run -n kan python -c "import json; from pathlib import Path; import experiments.run_v130_function_preserving_transport as r; out=Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130').resolve(); code_manifest, code_sha = r.write_code_packet(out); _manifest, missing = r.write_manifest(out); route_path = out / 'v130_route_decision.json'; route = json.loads(route_path.read_text()); route['code_review_packet_entries'] = len(code_manifest); route['code_review_packet_sha256'] = code_sha; route['required_artifact_missing_count'] = missing; route_path.write_text(json.dumps(route, indent=2, sort_keys=True) + '\\n'); r.write_manifest(out); print({'code_review_packet_entries': len(code_manifest), 'code_review_packet_sha256': code_sha, 'missing': missing})"
```

输出：

```text
{'code_review_packet_entries': 8, 'code_review_packet_sha256': '7a5f5e8d6d754168a809eae9ed078a177aa2544b150d712016127c13f206263a', 'missing': 0}
```

sanity check（随后若继续编辑本日志，需要重新刷新 code packet；最终 hash 以 route JSON 为准）：

```bash
python - <<'PY'
import json,csv,zipfile
from pathlib import Path
base=Path('results/v13_0_strategic_reset_function_preserving_coordinate_transport/official_v130')
route=json.loads((base/'v130_route_decision.json').read_text())
print('route', route['route'])
print('minimum_success', route['minimum_success'])
print('official_success_reached', route['official_success_reached'])
print('promotion_allowed', route['promotion_allowed'])
print('required_artifact_missing_count', route['required_artifact_missing_count'])
print('code_review_packet_entries', route.get('code_review_packet_entries'))
print('code_review_packet_sha256', route.get('code_review_packet_sha256'))
manifest=list(csv.DictReader((base/'v130_required_artifact_manifest.csv').open()))
print('manifest_rows', len(manifest), 'missing', sum(str(r.get('exists','')).lower() in ('0','false') or str(r.get('missing','')).lower() in ('1','true') for r in manifest))
with zipfile.ZipFile(base/'v130_code_review_packet.zip') as z:
    names=z.namelist()
    print('zip_entries', len(names))
    print('has_exec_log', any('执行日志' in n for n in names))
    print('has_review_log', any('实验结果复盘' in n for n in names))
PY
```

输出：

```text
route R2-SyntheticMechanismProofFailed
minimum_success S1-EfficientSubstrate
official_success_reached 0
promotion_allowed 0
required_artifact_missing_count 0
code_review_packet_entries 8
code_review_packet_sha256 7a5f5e8d6d754168a809eae9ed078a177aa2544b150d712016127c13f206263a
manifest_rows 25 missing 0
zip_entries 8
has_exec_log True
has_review_log True
```
