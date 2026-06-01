# DG-KAN v13.5 DecisiveOracleTarget SubstrateReset 执行日志

生成时间：2026-05-28（Asia/Singapore）

本日志只记录实际执行过的代码修改、命令、artifact 路径与结果摘要；不把 smoke / diagnostic / fallback 写成 promotion。

## 1. 计划文件

计划来源：

```text
docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md
```

v13.5 的核心问题不是继续修 O7/O8/BM/BN，而是拆分：

```text
1. 是否存在有 future advantage 的 oracle basis-channel target。
2. 如果存在，合法 precommit feature 是否能提前看见它。
3. Non-RAT 是否能形成 S1C vertical slice。
4. MLP analog 是否解释同一 oracle 机制。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 label-informed initialization。
3. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
4. oracle target 可作为 diagnostic upper-bound，但不能写成 promotion direction。
5. CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate；不能作为 legal functional success。
6. 只有 oracle upper-bound + legal visibility + Non-RAT/S1C/MLP closure 等 gate 同时满足后，才允许进入 real short-run。
```

## 2. 代码文件

新增 runner：

```text
experiments/run_v135_decisive_oracle_target_substrate_reset.py
```

主要实现：

```text
1. 复用 v13.4 live basis-channel、J_theta->Z projection 与 future probe 工具。
2. 新增 oracle target：
   - O-OR1-FutureTrainingDeltaZ
   - O-OR2-BestControlResidual
   - O-OR3-LineCReleaseOracle
   - O-OR4-TaskFamilySpecific
3. 新增 projection/writeback audit：
   - P4-ConjugateGradientJtJProjection
   - P7-ReachableSubspaceProjection
   - P8-TrustRegionReachableProjection
4. 新增 legal precommit visibility audit。
5. 新增 sequential two-event oracle diagnostic。
6. 新增 MLP oracle controls：
   - M-OR1-MLP-hidden-channel-oracle-DeltaH
   - M-OR2-MLP-hidden-channel-precommit-proxy
   - M-OR3-MLP-hidden-channel-random-reachable-control
7. 新增 Non-RAT S1C vertical slice：
   - FOU-S1C1/2/3
   - CHE-S1C1/2/3
8. 输出 required manifest、figures、route decision、code review packet。
```

审计重点：

```text
1. projection 写回真实 basis parameters。
2. full_basis_param_update=1。
3. readout_feature_proxy_only=0。
4. feature_table_proxy_only=0。
5. oracle rows 均 promotion_allowed=0。
6. forbidden information audit 记录 provenance violation。
```

## 3. 语法检查

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v135_decisive_oracle_target_substrate_reset.py
```

结果：

```text
py_compile pass
```

## 4. Smoke

命令：

```bash
conda run -n kan python experiments/run_v135_decisive_oracle_target_substrate_reset.py \
  --out-dir results/v13_5_decisive_oracle_target_substrate_reset/smoke_v135 \
  --device cuda:0 \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --oracle-types O-OR1-FutureTrainingDeltaZ \
  --synthetic-train-size 48 --synthetic-val-size 24 \
  --operator-batch-size 8 --max-channel-dim 8 --max-jacobian-rows 64 \
  --checkpoint-steps 4 --oracle-future-steps 4 --future-steps 4 \
  --nonrat-actuation-autopsy
```

结果：

```text
route = R1-NoS1C
minimum_success = S1-EfficientSubstrate
required_artifact_missing_count = 0
oracle_rows = 1
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
full_basis_param_update_rows = 1
provenance_violation_count = 0
```

说明：

```text
smoke 只证明 runner / artifact / writeback trace 可执行。
该 smoke 中 O-OR1 future-training DeltaZ norm = 0，不允许写成 S1C 或 oracle success。
```

## 5. Official v13.5

命令：

```bash
conda run -n kan python experiments/run_v135_decisive_oracle_target_substrate_reset.py \
  --out-dir results/v13_5_decisive_oracle_target_substrate_reset/official_v135 \
  --device cuda:0 \
  --synthetic-seeds 0,1 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --oracle-types O-OR1-FutureTrainingDeltaZ,O-OR2-BestControlResidual,O-OR3-LineCReleaseOracle,O-OR4-TaskFamilySpecific \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --oracle-delta-z-scale 0.01 \
  --checkpoint-steps 20 --oracle-future-steps 20 --future-steps 30 \
  --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

主要结果：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s1c_count = 1
nonrat_s1c_count = 0
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
mlp_oracle_pass_count = 0
full_basis_param_update_rows = 56
provenance_violation_count = 0
forbidden_information_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

最接近 oracle row：

```text
task = X4
seed = 0
oracle_type = O-OR2-BestControlResidual
projection_solver = P4-ConjugateGradientJtJProjection
oracle_delta_z_norm = 0.0
oracle_projection_residual = 0.0
oracle_actuation_error = 0.0
oracle_actuation_cosine = 0.0
source_vs_best = 0.0017166185519627633
NoiseSignalLeak_delta = -0.012942790985107422
RealSignalReservoirRatio_delta = -1.6300039291381836
CEp99_delta = -4.755213737487793
synthetic_success = 0
```

失败原因：

```text
source_vs_best < 0.005；
oracle task-family pass = 0/7；
oracle upper-bound 不成立。
最终逐行数值以 official_v135/v135_oracle_synthetic_proof.csv 为准。
```

## 6. Blocker 修复：P8 trust-region / scale fallback

v13.5 计划要求 oracle upper-bound fail 后检查 projection/scale/trust-region 方向。本轮新增并执行：

```text
P8-TrustRegionReachableProjection
oracle_delta_z_scale = 0.03
```

命令：

```bash
conda run -n kan python experiments/run_v135_decisive_oracle_target_substrate_reset.py \
  --out-dir results/v13_5_decisive_oracle_target_substrate_reset/fallback_p8_scale003_v135 \
  --device cuda:0 \
  --synthetic-seeds 0,1 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --oracle-types O-OR1-FutureTrainingDeltaZ,O-OR2-BestControlResidual,O-OR3-LineCReleaseOracle,O-OR4-TaskFamilySpecific \
  --oracle-projection-solver P8-TrustRegionReachableProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --oracle-delta-z-scale 0.03 \
  --checkpoint-steps 20 --oracle-future-steps 20 --future-steps 30 \
  --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

结果：

```text
route = R4-OracleTargetNoUpperBound
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
mlp_oracle_pass_count = 0
nonrat_s1c_count = 0
full_basis_param_update_rows = 56
required_artifact_missing_count = 0
```

最接近 P8 row：

```text
task = X7
seed = 0
oracle_type = O-OR3-LineCReleaseOracle
source_vs_best = 0.0044884306696907
synthetic_success = 0
```

说明：

```text
P8 / larger oracle scale 提高了最接近 row 的 source_vs_best，但仍低于 0.005 gate；
没有打开 oracle upper-bound。
```

## 7. Non-RAT S1C vertical slice

初始 Non-RAT 复用 v13.4-style design 后，本轮按 v13.5 计划补成 explicit vertical slices：

```text
FOU-S1C1-sincos-channel-band-normalized-DeltaZ
FOU-S1C2-frequency-band-block-projection
FOU-S1C3-low-frequency-only-reachable-target
CHE-S1C1-degree-channel-energy-normalized-DeltaZ
CHE-S1C2-low-degree-only-projection
CHE-S1C3-recurrence-stable-channel-projection
```

相关代码修改：

```text
experiments/run_v135_decisive_oracle_target_substrate_reset.py
  build_nonrat_vertical_slice() 输出 v13.5-specific FOU/CHE S1C rows。
```

official_v135 Non-RAT 结果：

```text
D-FOU14 FOU-S1C1: actuation_error=0.058193475008010864, cosine=0.9986006021499634, workspace_incremental_ratio=4.2583056478405314, LineC_pass_rate=0.0, S1C=0
D-FOU14 FOU-S1C2: actuation_error=0.05084142088890076, cosine=0.9990485906600952, workspace_incremental_ratio=4.2583056478405314, LineC_pass_rate=0.0, S1C=0
D-FOU14 FOU-S1C3: actuation_error=0.06653273850679398, cosine=0.9981427192687988, workspace_incremental_ratio=4.2583056478405314, LineC_pass_rate=0.0, S1C=0
D-CHE16 CHE-S1C1: actuation_error=0.2500919997692108, cosine=0.9684257507324219, workspace_incremental_ratio=7.000830564784053, LineC_pass_rate=0.0, S1C=0
D-CHE16 CHE-S1C2: actuation_error=0.05191929265856743, cosine=0.9989519715309143, workspace_incremental_ratio=7.000830564784053, LineC_pass_rate=0.0, S1C=0
D-CHE16 CHE-S1C3: actuation_error=0.02824893593788147, cosine=0.9997985363006592, workspace_incremental_ratio=7.000830564784053, LineC_pass_rate=0.0, S1C=0
```

说明：

```text
Non-RAT random actuation 可通过，但 workspace / LineC gate 失败；
不能把 actuation pass 写成 Non-RAT S1C。
```

## 8. 最终刷新命令

写入本执行日志与实验结果复盘后，重新执行 official 命令一次，使 code review packet 包含最终文档：

```bash
conda run -n kan python experiments/run_v135_decisive_oracle_target_substrate_reset.py \
  --out-dir results/v13_5_decisive_oracle_target_substrate_reset/official_v135 \
  --device cuda:0 \
  --synthetic-seeds 0,1 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --oracle-types O-OR1-FutureTrainingDeltaZ,O-OR2-BestControlResidual,O-OR3-LineCReleaseOracle,O-OR4-TaskFamilySpecific \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --oracle-delta-z-scale 0.01 \
  --checkpoint-steps 20 --oracle-future-steps 20 --future-steps 30 \
  --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

该命令已执行完成；route 仍为：

```text
R4-OracleTargetNoUpperBound
```

随后仅因文档数值补正刷新 code review packet 包装，不重跑训练指标。包装刷新命令：

```bash
python - <<'PY'
import hashlib, json, pathlib, zipfile
root = pathlib.Path('.').resolve()
out = root / 'results/v13_5_decisive_oracle_target_substrate_reset/official_v135'
files = [
    root / 'experiments/run_v135_decisive_oracle_target_substrate_reset.py',
    root / 'docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md',
    root / 'docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_执行日志.md',
    root / 'docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_实验结果复盘.md',
    out / 'v135_route_decision.json',
    out / 'v135_oracle_target_audit.csv',
    out / 'v135_precommit_feature_audit.csv',
    out / 'v135_no_go_boundary.md',
]
def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
manifest = []
zip_path = out / 'v135_code_review_packet.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files:
        if p.exists():
            zf.write(p, p.relative_to(root).as_posix())
            manifest.append({'path': p.relative_to(root).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
(out / 'v135_code_review_manifest.csv').write_text(
    'path,sha256,bytes\n' + ''.join(f"{r['path']},{r['sha256']},{r['bytes']}\n" for r in manifest),
    encoding='utf-8',
)
route = json.loads((out / 'v135_route_decision.json').read_text())
route['code_review_packet_entries'] = len(manifest)
route['code_review_packet_sha256'] = sha(zip_path)
(out / 'v135_route_decision.json').write_text(json.dumps(route, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({'entries': len(manifest), 'sha256': route['code_review_packet_sha256']}, indent=2))
PY
```

最终 artifact 目录：

```text
results/v13_5_decisive_oracle_target_substrate_reset/official_v135
```

## 9. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.5 是否达成目标，若未达成则继续。本次没有新增训练实验；先按计划文件 stop rule、route decision、no-go boundary 与 next hypothesis queue 复核是否仍有计划内必须执行的 fallback。

执行命令：

```bash
rg -n "R4|final stop|stop|NoGo|no-go|OracleTargetNoUpperBound|不得|不能|继续|fallback|Non-RAT|MLP|legal|visibility|upper-bound" \
  docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md

sed -n '949,986p' docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md

python - <<'PY'
import json, pathlib
p = pathlib.Path('results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_route_decision.json')
d = json.loads(p.read_text())
for k in [
    'route','minimum_success','official_success_reached','promotion_allowed',
    'final_stop_allowed','hard_compute_budget_exhausted','fallback_all_executed',
    'required_artifact_missing_count','substrate_s1_count','substrate_s1c_count',
    'nonrat_s1c_count','oracle_rows','oracle_pass_rows',
    'oracle_task_family_pass_count','oracle_upper_bound_pass',
    'precommit_visibility_rows','precommit_visibility_pass',
    'sequential_oracle_rows','sequential_oracle_pass_count',
    'mlp_oracle_rows','mlp_oracle_pass_count',
    'full_basis_param_update_rows','provenance_violation_count',
    'forbidden_information_violation_count','readout_feature_proxy_only',
    'feature_table_proxy_only','code_review_packet_entries','code_review_packet_sha256'
]:
    print(f'{k}={d.get(k)}')
PY

cat results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_no_go_boundary.md
cat results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_next_hypothesis_queue.md
```

复核结果：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
mlp_oracle_pass_count = 0
nonrat_s1c_count = 0
full_basis_param_update_rows = 56
provenance_violation_count = 0
forbidden_information_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

计划 stop rule 对照：

```text
1. v13.5 第 8.4 节：If oracle fails -> route R4, stop functional on current substrate, return to substrate/base architecture。
2. v13.5 第 17.1 节：如果没有 oracle upper-bound，禁止继续 O7/O8/O9、BM/BN、real short-run、LineC/CEp99/NLL/ECE direction、readout-feature proxy、frozen feature-table transport。
3. v135_next_hypothesis_queue.md：If R4, return to substrate/base architecture; do not continue O7/O8/O9 or BM/BN token search。
```

结论：

```text
v13.5 没有达成目标；
但计划内继续条件已经闭合；
继续在当前 v13.5 runner 内加 O-token / BM-BN / trust-region / scale 小修会违反计划 no-go；
下一步应转入新的 substrate/base architecture plan，而不是把 diagnostic 写成 success。
```

本次因只追加日志，未重跑训练指标。随后刷新 code review packet 包装以包含本节日志。

## 10. 用户再次追问后的 substrate/base architecture 边界复核

用户再次要求“未达成则继续”。本次额外检查了当前打开的 v12.36 substrate-health 计划与 v13.5 stop rule 的衔接，确认 v13.5 的下一步确实是“return to substrate/base architecture”，但不是在 v13.5 runner 内继续 O-token / BM-BN / trust-region 小修。

执行命令：

```bash
rg -n "substrate/base|base architecture|return to substrate|architecture|S1C vertical|Non-RAT|FOU|CHE|RBF|Wavelet|must|必须|If oracle|oracle upper-bound fail|17\\.2|7\\.3|14\\.3" \
  docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md \
  docs/DG-KAN_v12.36_SubstrateHealth_ChannelColocation_FunctionalRepair_完整计划.md

sed -n '374,462p' docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md
sed -n '949,986p' docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md
```

检查结果：

```text
v13.5 已执行的 Non-RAT vertical slice 覆盖计划中明确列出的 FOU-S1C1/2/3 与 CHE-S1C1/2/3。
v13.5 第 17.2 节只给出 oracle fail 后的方向：return to substrate/base architecture。
v13.5 第 17.1 节明确禁止继续当前 functional target 小修。
v12.36 包含 D-CHE23/24/25、D-FOU23/24/25、D-RBF20/21/22 等 substrate-health redesign 候选，但这是新的 substrate/base architecture 工作，不是 v13.5 oracle-target runner 内的局部修复。
```

本次判断：

```text
1. v13.5 仍未达成目标。
2. 继续在 v13.5 runner 内增加 O-token / BM-BN / trust-region / scale grid 会违反计划。
3. 直接把 v12.36 的 Non-RAT substrate redesign 塞进 v13.5，并沿用 v13.5 promotion gate，存在把 substrate diagnostic / proxy 写成 oracle-target success 的风险。
4. 我现在不确定如何在当前 v13.5 代码路径内安全实现新的 substrate/base architecture reset，并保证不混淆 v13.5 的 oracle upper-bound route。
```

因此，本次没有新增训练实验、没有新增 CSV、没有修改 gate；只记录边界复核。合法下一步应是单独制定/执行新的 substrate/base architecture reset 计划。

## 11. 用户再次追问后的可执行 runner 复核

用户再次要求“未达成则继续”。本次检查 v12.36 substrate-health 候选是否已有可直接作为 v13.5 fallback 接入的 runner / registry 路径。

执行命令：

```bash
rg -n "D-CHE23|D-CHE24|D-CHE25|D-FOU23|D-FOU24|D-FOU25|D-RBF20|D-RBF21|D-RBF22|Non-RAT task-health|Line N|task-health substrate" docs experiments dgkan | head -n 200

rg -n "def .*nonrat|Non-RAT|D-CHE|D-FOU|D-RBF|task-health|candidate" \
  experiments/run_v1235_finalize_substrate_health_colocation.py \
  experiments/run_v12342_finalize_all_basis_substrate_functional.py \
  experiments/run_v135_decisive_oracle_target_substrate_reset.py \
  experiments/run_v134_operator_level_basis_channel_functional.py | head -n 220

sed -n '890,990p' docs/DG-KAN_v12.36_SubstrateHealth_ChannelColocation_FunctionalRepair_完整计划.md
```

检查结果：

```text
1. D-CHE23/24/25、D-FOU23/24/25、D-RBF20/21/22 目前主要出现在 v12.36 计划文件中。
2. 现有 v13.5 runner 只实现了 v13.5 计划中的 FOU-S1C1/2/3 与 CHE-S1C1/2/3 vertical slice。
3. v1235 finalizer 有 Non-RAT task-health repair audit，但不是 v13.5 oracle-target fallback runner。
4. 未发现可直接把 v12.36 substrate-health redesign 安全挂入 v13.5 oracle route 的现成实现。
```

结论：

```text
v13.5 仍未达成目标；
但我现在不确定如何在当前 v13.5 代码路径内安全实现 v12.36 substrate/base reset，并保证不把 v12.36 substrate diagnostic 混成 v13.5 oracle upper-bound success。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因是缺少明确的 v13.5 内部 runner/route 来承载新的 substrate/base architecture reset。

## 12. 用户再次追问后的最终 stop-contract 复核

用户再次要求“未达成则继续”。本次复核不新增训练实验，不修改 gate，只确认是否仍有 v13.5 计划内未执行的推荐修复方向。

执行命令：

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path('results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_route_decision.json')
d = json.loads(p.read_text())
for k in [
    'route','minimum_success','official_success_reached','promotion_allowed',
    'final_stop_allowed','hard_compute_budget_exhausted','fallback_all_executed',
    'required_artifact_missing_count','oracle_upper_bound_pass',
    'oracle_task_family_pass_count','precommit_visibility_pass',
    'sequential_oracle_pass_count','nonrat_s1c_count','mlp_oracle_pass_count',
    'full_basis_param_update_rows','readout_feature_proxy_only','feature_table_proxy_only'
]:
    print(f'{k}={d.get(k)}')
PY

sed -n '548,558p' docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md
sed -n '949,986p' docs/DG-KAN_v13.5_DecisiveOracleTarget_SubstrateReset_完整计划.md
```

复核结果：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
oracle_upper_bound_pass = 0
oracle_task_family_pass_count = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
nonrat_s1c_count = 0
mlp_oracle_pass_count = 0
full_basis_param_update_rows = 56
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

计划对照：

```text
1. 第 8.4 节：If oracle fails -> route = R4-OracleTargetNoUpperBound；recommendation = stop functional on current substrate; return to substrate/base architecture。
2. 第 17.1 节：如果没有 oracle upper-bound，禁止继续 O7/O8/O9 token 小修、BM/BN metric 回退、response dictionary vN、real short-run、LineC hard target direction、CEp99/NLL/ECE direction、readout-feature proxy、frozen feature-table transport。
3. 第 17.2 节：oracle upper-bound fail 后只能 return to substrate/base architecture。
```

最终执行判断：

```text
v13.5 当前 runner 内没有剩余 plan-authorized 修复方向；
继续当前路径会违反 v13.5 no-go；
下一步应新开 substrate/base architecture reset，而不是继续 v13.5。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因是我已经不确定如何在当前 v13.5 代码路径内安全实现新的 substrate/base architecture reset，并保证不把 substrate diagnostic / proxy 编造成 oracle upper-bound success。
