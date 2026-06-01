# DG-KAN v13.6 PopRiskSNR BasisCoverBoundary 执行日志

生成时间：2026-05-28（Asia/Singapore）

本日志只记录实际执行过的命令、产物路径与可复现信息；不把 smoke / fallback / diagnostic 写成 promotion。

## 1. 计划文件

```text
docs/DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md
```

理解后的执行顺序：

```text
1. MLP parameter-SNR baseline。
2. Rational parameter-SNR baseline。
3. Rational basis-channel SNR。
4. Rational basis-cover boundary。
5. Non-RAT substrate-SNR scouts：Fourier + Chebyshev/RBF。
6. X1-X7 synthetic family proof。
7. MLP analog comparison。
8. final route / no-go / next hypothesis queue。
```

## 2. 代码修改

新增 runner：

```text
experiments/run_v136_poprisk_snr_basis_cover_boundary.py
```

关键实现位置：

```text
loss_interface_cotangent_per_example: line 132
collect_per_example_gradients: line 326
make_snr_update: line 377
run_future_case: line 584
run_substrate_scouts: line 1106
build_route: line 1350
```

实现摘要：

```text
1. 使用 train-stream current batch 的 per-example gradient g_i = J_theta(x_i)^T delta_i。
2. delta_i 来自 generic loss interface：CE / Brier / MSELogit；没有使用 CEp99/NLL/ECE/LineC 生成方向。
3. 实现 parameter SNR、basis-channel SNR、basis-cover boundary。
4. 写回真实 named parameters，并记录 writeback sha。
5. 实现 controls：AdamW、NoOpMatchedOverhead、RandomMatchedNorm、AdamWParallelDirection、ShuffledPerExampleGradient、SameMaskRandomSign、SameActiveFractionRandomMask。
6. 执行 Non-RAT substrate-SNR scouts：D-FOU、D-CHE、D-RBF。
7. 输出 required manifest、forbidden audit、failure table、figures、code packet。
8. 修复 finalize-only 刷新 route 时 control_rows 为空字符串的问题；刷新后 control_rows=1120。
```

## 3. 语法检查

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v136_poprisk_snr_basis_cover_boundary.py
```

结果：

```text
py_compile pass
```

## 4. Smoke

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/smoke_v136 \
  --device cuda:0 \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --synthetic-train-size 32 \
  --synthetic-val-size 16 \
  --checkpoint-steps 2 \
  --future-steps 3 \
  --operator-batch-size 8 \
  --methods MLP-ParameterSNR,KAN-BasisChannelSNR \
  --snr-variants Hard \
  --loss-interfaces CE \
  --controls Functional,AdamW,RandomMatchedNorm,SameActiveFractionRandomMask \
  --mlp-hidden 32
```

结果：

```text
out_dir = results/v13_6_poprisk_snr_basis_cover_boundary/smoke_v136
route = R4-PopRiskSNRNoGo
required_artifact_missing_count = 0
synthetic_rows = 2
control_rows = 8
full_parameter_update_rows = 2
forbidden_information_violation_count = 0
```

解释：smoke 只证明 runner / per-example gradient / writeback / controls / artifact surface 可运行，不允许 promotion。

## 5. Official v13.6

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/official_v136 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 8 \
  --future-steps 10 \
  --operator-batch-size 16 \
  --methods MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary \
  --snr-variants Hard \
  --loss-interfaces CE,Brier \
  --controls Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果：

```text
out_dir = results/v13_6_poprisk_snr_basis_cover_boundary/official_v136
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
required_artifact_missing_count = 0
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_rows = 28
cover_boundary_accept_count = 22
substrate_snr_gate_pass_count = 1
nonrat_substrate_snr_gate_pass_count = 0
full_parameter_update_rows = 134
forbidden_information_violation_count = 0
```

## 6. Official 结果检查命令

执行：

```bash
python - <<'PY'
import csv, json, collections
base='results/v13_6_poprisk_snr_basis_cover_boundary/official_v136'
rows=list(csv.DictReader(open(base+'/v136_synthetic_family_results.csv')))
print('rows',len(rows))
print('official pass',sum(int(float(r['official_synthetic_pass'])) for r in rows))
print('explore',sum(int(float(r['exploration_pass'])) for r in rows))
for method in sorted(set(r['method'] for r in rows)):
    rs=[r for r in rows if r['method']==method]
    print(method, max(float(r['source_vs_best_control']) for r in rs),
          sum(float(r['snr_active_fraction']) for r in rs)/len(rs))
print(collections.Counter(csv.DictReader(open(base+'/v136_failure_table.csv')).__next__() for _ in []))
PY
```

实际用于复盘的读取结果：

```text
official pass = 0
exploration pass = 0
best source_vs_best_control:
  KAN-BasisCoverBoundary = 0.0007192629328190044
  MLP-ParameterSNR = 0.00000020563967318299597
  MLP-HiddenChannelSNR = -0.00010800396597619472
  KAN-BasisChannelSNR = -0.00023018903925793067
  KAN-ParameterSNR = -0.0013982748115428344
```

## 7. Fallback 1：Soft / EMA3RoleNorm

触发原因：

```text
official 没有任何 exploration pass；
按计划尝试 soft q、EMA mu/sigma over 3 windows、role-wise threshold normalization。
```

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_soft_ema_role_v136 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 8 \
  --future-steps 8 \
  --operator-batch-size 16 \
  --methods MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary \
  --snr-variants Soft,EMA3RoleNorm \
  --loss-interfaces CE \
  --controls Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 140
control_rows = 1120
official pass = 0
exploration pass = 0
required_artifact_missing_count = 0
best source_vs_best_control = 0.0004920390084638543
best row = KAN-BasisChannelSNR / EMA3RoleNorm / X2 / seed 0 / CE
```

## 8. Fallback 2：High tau + active cap

触发原因：

```text
official 中 MLP-HiddenChannelSNR active fraction 均值 0.814；
Non-RAT CHE scout active fraction = 1.0；
按计划尝试 increase tau + cap active fraction。
```

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_tau_cap_v136 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 8 \
  --future-steps 8 \
  --operator-batch-size 16 \
  --methods MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary \
  --snr-variants Hard \
  --snr-tau 4.0 \
  --active-fraction-cap 0.25 \
  --loss-interfaces CE \
  --controls Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 70
control_rows = 560
official pass = 0
exploration pass = 0
required_artifact_missing_count = 0
best source_vs_best_control = 0.0002900220635735917
best row = KAN-BasisCoverBoundary / Hard / X4 / seed 0 / CE
```

## 9. Fallback 3：eta / norm sensitivity

触发原因：

```text
official / Soft / cap fallback 都没有 exploration pass；
额外排除只是 functional step 太弱或 norm cap 太紧。
```

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_eta_norm_v136 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 8 \
  --future-steps 8 \
  --operator-batch-size 16 \
  --methods MLP-ParameterSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary \
  --snr-variants Hard \
  --functional-eta 0.05 \
  --max-update-norm-ratio 0.05 \
  --loss-interfaces CE \
  --controls Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 56
control_rows = 448
official pass = 0
exploration pass = 0
required_artifact_missing_count = 0
best source_vs_best_control = 0.00021401959926648298
best row = MLP-ParameterSNR / Hard / X4 / seed 0 / CE
```

## 10. Finalize

写入本文档与复盘后刷新 code packet：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/official_v136 \
  --finalize-only
```

最终 route 仍以：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_route_decision.json
```

为准。

刷新后关键值：

```text
route = R4-PopRiskSNRNoGo
control_rows = 1120
code_review_packet_entries = 7
code_review_packet_sha256 以 v136_route_decision.json 为准
```

## 11. 用户再次追问后的 stop-contract 复核

触发原因：用户再次要求确认 v13.6 是否达成目标，若未达成则继续。本次只复核计划 stop rule 与已生成 artifact；没有新增训练实验、没有新增 CSV 指标、没有修改 gate。

复核计划推荐分支：

```bash
sed -n '570,620p' docs/DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md
sed -n '1170,1245p' docs/DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md
```

复核 official route：

```bash
jq . results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_route_decision.json
```

复核 official 与 fallback 的关键计数：

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path('results/v13_6_poprisk_snr_basis_cover_boundary')
for name in ['official_v136', 'fallback_soft_ema_role_v136', 'fallback_tau_cap_v136', 'fallback_eta_norm_v136']:
    route = json.loads((base / name / 'v136_route_decision.json').read_text())
    print(
        name,
        route.get('route'),
        'synthetic_success', route.get('synthetic_task_success_count'),
        'mlp', route.get('mlp_snr_pass_rows'),
        'kan_param', route.get('kan_parameter_snr_pass_rows'),
        'kan_basis', route.get('kan_basis_snr_pass_rows'),
        'missing', route.get('required_artifact_missing_count'),
    )
PY
```

复核结果：

```text
official_v136 route = R4-PopRiskSNRNoGo
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
synthetic_task_success_count = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
required_artifact_missing_count = 0
full_parameter_update_rows = 134
control_rows = 1120

fallback_soft_ema_role_v136 route = R4-PopRiskSNRNoGo
fallback_tau_cap_v136 route = R4-PopRiskSNRNoGo
fallback_eta_norm_v136 route = R4-PopRiskSNRNoGo
三组 fallback 的 synthetic_success / MLP / KAN parameter / KAN basis pass 均为 0。
```

计划 stop rule 对照：

```text
1. 5.7 要求 active fraction 太低时尝试 soft q / EMA3 / role-wise normalization：已执行 fallback_soft_ema_role_v136。
2. 5.7 要求 active fraction 太高时尝试 higher tau / per-role active cap / SameActiveFractionRandomMask：已执行 fallback_tau_cap_v136。
3. 额外排除 step/norm sensitivity：已执行 fallback_eta_norm_v136。
4. 14.2 写明 parameter-SNR 与 basis-SNR 在 MLP/KAN 上都 fail 时，route = R4-PopRiskSNRNoGo。
5. 15 要求 MLP baseline、Rational parameter-SNR、Rational basis-channel SNR、basis-cover boundary、Non-RAT scouts、MLP analog、final no-go / next queue：均已执行并写入 official artifact。
```

结论：当前 v13.6 计划内继续条件已经闭合；继续在 SNR mask / tau / EMA / norm / cover-boundary 小网格上排列变体不属于计划推荐方向，且会增加把 diagnostic 写成 promotion 的风险。合法下一步应是新的 substrate/base architecture 或重新定义可价值 functional source，而不是继续 v13.6 runner 内部小修。

写入本文档与复盘后刷新 official code packet：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/official_v136 \
  --finalize-only
```

刷新后 route 仍为：

```text
R4-PopRiskSNRNoGo
```

## 12. 用户再次追问后的 Line S substrate/base architecture scout 补跑

触发原因：再次复核计划后发现 v13.6 第 8 节明确包含 `Line S：Substrate / base architecture reset`，并列出了 RAT/CHE/FOU/RBF/WAV-SNR1/2/3 candidate families。此前 official 只覆盖 primary D-RAT26 与 D-FOU14/D-CHE16/D-RBF11 scouts，没有覆盖这些 Line S candidate 名称。因此本次补齐 plan-authorized substrate scout，不把它写成 promotion。

代码修改：

```text
experiments/run_v136_poprisk_snr_basis_cover_boundary.py
  新增 LINE_S_CANDIDATE_SPECS。
  新增 make_line_s_model()。
  新增 line_s_candidate_rows()。
  run_substrate_scouts() 追加 Line S candidates，并记录 mapped_candidate_id / line_s_candidate / line_s_mapping_no_new_kernel_claim。
  code_review_manifest 增加 Line S candidate mapping / model builder / substrate-SNR gate surface。
```

审计说明：

```text
1. Line S 名称是设计级 candidate，不声明新的 exact kernel 成功。
2. 每个 Line S candidate 映射到已有 v12.35/v1235 audited primitive candidate。
3. artifact 中显式写出 mapped_candidate_id 与 line_s_mapping_no_new_kernel_claim=1。
4. 仍按 v13.6 Substrate-SNR gate 判定，不降低 workspace/task/LineC/cover gate。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v136_poprisk_snr_basis_cover_boundary.py
```

Line S scout 补跑：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_line_s_substrate_scout_v136 \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 4 \
  --future-steps 4 \
  --operator-batch-size 8 \
  --methods KAN-BasisChannelSNR \
  --snr-variants Hard \
  --loss-interfaces CE \
  --controls Functional,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果检查：

```bash
jq . results/v13_6_poprisk_snr_basis_cover_boundary/fallback_line_s_substrate_scout_v136/v136_route_decision.json

python - <<'PY'
import csv
from pathlib import Path
p = Path('results/v13_6_poprisk_snr_basis_cover_boundary/fallback_line_s_substrate_scout_v136/v136_substrate_snr_gate.csv')
rows = list(csv.DictReader(p.open()))
print('rows', len(rows))
print('pass_total', sum(int(float(r.get('substrate_snr_gate_pass') or 0)) for r in rows))
print('nonrat_pass', sum(int(float(r.get('substrate_snr_gate_pass') or 0)) for r in rows if r.get('family') != 'D-RAT'))
PY
```

关键结果：

```text
route = R4-PopRiskSNRNoGo
required_artifact_missing_count = 0
synthetic_rows = 1
synthetic_task_success_count = 0
full_parameter_update_rows = 1
substrate_snr_gate_rows = 19
line_s_candidate_rows = 15
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
```

Family summary：

```text
D-RAT rows = 4, pass = 4, status = SubstrateSNRGatePass
D-CHE rows = 4, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
D-FOU rows = 4, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
D-RBF rows = 4, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
D-WAV rows = 3, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
```

判断：Line S 补跑确认 Rational 映射候选可过 substrate-SNR gate，但 Non-RAT 的 Chebyshev / Fourier / RBF / Wavelet 仍没有形成 functional substrate。该结果不改变 official promotion 判断；它只闭合 v13.6 第 8 节的 substrate/base architecture scout 覆盖缺口。

写入本文档与复盘后刷新 fallback / official code packet：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_line_s_substrate_scout_v136 \
  --finalize-only

conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/official_v136 \
  --finalize-only
```

刷新后：

```text
fallback_line_s_substrate_scout_v136 route = R4-PopRiskSNRNoGo
fallback_line_s_substrate_scout_v136 substrate_snr_gate_pass_count = 4
fallback_line_s_substrate_scout_v136 nonrat_substrate_snr_gate_pass_count = 0
official_v136 route = R4-PopRiskSNRNoGo
official_v136 promotion_allowed = 0
```

## 13. Line S 合并后的 official_v136 重跑

触发原因：上一节的 Line S scout 已证明计划第 8 节覆盖缺口，但结果位于 fallback 目录。为了让 official_v136 本体也直接包含 Line S substrate/base architecture scout，本次用同一 official 配置重跑。

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/official_v136 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 8 \
  --future-steps 10 \
  --operator-batch-size 16 \
  --methods MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary \
  --snr-variants Hard \
  --loss-interfaces CE,Brier \
  --controls Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果：

```text
route = R4-PopRiskSNRNoGo
promotion_allowed = 0
final_stop_allowed = 1
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
substrate_snr_gate_rows = 19
line_s_candidate_rows = 15
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
required_artifact_missing_count = 0
```

通过的 substrate-SNR rows：

```text
D-RAT26-TangentTrustRegionNoCE -> D-RAT26-TangentTrustRegionNoCE
RAT-SNR1-denSlopeTelemetrySubstrate -> D-RAT27-DenSlopeGuardNoCE
RAT-SNR2-groupDiversityFloorSubstrate -> D-RAT28-GroupDiversityPreservingRational
RAT-SNR3-readoutRationalDecoupledSubstrate -> D-RAT35-ReadoutRationalDecoupleNoCE
```

Non-RAT 结果：

```text
D-CHE rows = 4, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
D-FOU rows = 4, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
D-RBF rows = 4, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
D-WAV rows = 3, pass = 0, status = WorkspaceOnly_NotFunctionalSubstrate
```

最接近 synthetic row：

```text
method = KAN-BasisCoverBoundary
task = X4
seed = 0
loss = CE
source_vs_best_control = 0.0003762373062765681
CouplingR2_delta = 2.0438709259033203
NoiseSignalLeak_delta = -0.018877089023590088
RealSignalReservoirRatio_delta = -0.34347105026245117
CEp99_delta = -0.95526123046875
official_synthetic_pass = 0
```

判断：official_v136 已经包含 Line S substrate/base architecture scout。结果仍为 R4；Rational 的 substrate-SNR 可见，但没有转化为 synthetic success；Non-RAT 仍没有进入 functional substrate gate。

## 14. Soft / EMA3RoleNorm CE+Brier 补跑

触发原因：再次复核时发现 `fallback_soft_ema_role_v136` 只使用 CE，而 official value-source audit 同时使用 CE/Brier。为了更完整覆盖计划 5.7 的 soft q / EMA3 / role-wise threshold normalization fallback，本次用 CE+Brier 补跑，不改 gate。

执行：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_soft_ema_role_ce_brier_v136 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --synthetic-train-size 64 \
  --synthetic-val-size 32 \
  --checkpoint-steps 8 \
  --future-steps 8 \
  --operator-batch-size 16 \
  --methods MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary \
  --snr-variants Soft,EMA3RoleNorm \
  --loss-interfaces CE,Brier \
  --controls Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask \
  --mlp-hidden 160
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 280
control_rows = 2240
synthetic_task_success_count = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_accept_count = 49
cover_boundary_rows = 56
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
required_artifact_missing_count = 0
```

最接近但不能 pass 的 row：

```text
method = KAN-BasisChannelSNR
task = X2
seed = 0
loss = CE
snr_variant = EMA3RoleNorm
source_vs_best_control = 0.0005674217174649332
CouplingR2_delta = 11.909942626953125
NoiseSignalLeak_delta = -0.013969182968139648
RealSignalReservoirRatio_delta = -0.8789176940917969
CEp99_delta = -0.9672317504882812
exploration_pass = 0
official_synthetic_pass = 0
```

主要 failure pattern：

```text
source;noise = 49
source = 28
source;coupling;noise;reservoir;cep99;nll;ece = 19
source;cep99 = 19
source;coupling;noise;reservoir;cep99;nll = 18
```

判断：CE+Brier 下的 Soft / EMA3RoleNorm 仍没有打开 exploration 或 official pass；主要 blocker 仍是 source_vs_best_control 不足。

写入本文档与复盘后刷新 fallback / official code packet：

```bash
conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/fallback_soft_ema_role_ce_brier_v136 \
  --finalize-only

conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py \
  --out-dir results/v13_6_poprisk_snr_basis_cover_boundary/official_v136 \
  --finalize-only
```

刷新后：

```text
fallback_soft_ema_role_ce_brier_v136 route = R4-PopRiskSNRNoGo
fallback_soft_ema_role_ce_brier_v136 synthetic_task_success_count = 0
official_v136 route = R4-PopRiskSNRNoGo
official_v136 promotion_allowed = 0
```

## 15. 用户再次追问后的 stop-contract 复核

触发原因：用户再次要求确认 v13.6 是否达成目标，若未达成则继续。本次不新增训练实验，先复核 official artifact、计划 stop/go rules、no-go boundary 与 required manifest，确认是否还有计划内未闭合 fallback。

复核 official route：

```bash
jq '{route,minimum_success,official_success_reached,promotion_allowed,final_stop_allowed,hard_compute_budget_exhausted,fallback_all_executed,synthetic_rows,control_rows,synthetic_task_success_count,synthetic_5of7_pass,mlp_snr_pass_rows,kan_parameter_snr_pass_rows,kan_basis_snr_pass_rows,cover_boundary_accept_count,cover_boundary_rows,substrate_snr_gate_pass_count,nonrat_substrate_snr_gate_pass_count,required_artifact_missing_count,provenance_violation_count,forbidden_information_violation_count,readout_feature_proxy_only,feature_table_proxy_only,code_review_packet_sha256}' \
  results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_route_decision.json
```

输出：

```text
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_accept_count = 22
cover_boundary_rows = 28
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
required_artifact_missing_count = 0
provenance_violation_count = 0
forbidden_information_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
code_review_packet_sha256 = 84b4aeb1b3ed76537ac6a9b4c70b0a0155284fd5f667778075d3bf39cea10d12
```

复核计划 fallback / Line S / stop-go 条款：

```bash
sed -n '568,620p' docs/DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md
sed -n '884,1048p' docs/DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md
sed -n '1168,1248p' docs/DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md
```

复核 official no-go boundary / next hypothesis queue：

```bash
cat results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_no_go_boundary.md
cat results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_next_hypothesis_queue.md
```

复核 required manifest：

```bash
python - <<'PY'
import csv
p='results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_required_manifest.csv'
rows=list(csv.DictReader(open(p)))
missing=[r for r in rows if str(r.get('exists','')).lower() not in ('1','true','yes')]
print('manifest_rows', len(rows))
print('missing_required_rows', len(missing))
PY
```

输出：

```text
manifest_rows = 35
missing_required_rows = 0
```

判断：

```text
1. v13.6 仍未达成目标，合法 route 仍为 R4-PopRiskSNRNoGo。
2. 计划 5.7 的 soft q / EMA3 / role-wise threshold normalization 与 tau/cap/norm fallback 已补跑。
3. 计划 Line S 的 RAT/CHE/FOU/RBF/WAV substrate-SNR candidate scout 已补跑并合入 official_v136。
4. MLP-SNR 与 KAN-SNR 都没有 pass；按计划 5.7 与 14.2，当前实现应记录 PopRiskSNR no-go，而不是继续 token/metric 小修。
5. 本次没有新增训练实验、没有修改 gate、没有新增 CSV 指标。
```

## 16. 用户再次追问后的 Line G / Line X 分支复核

触发原因：用户再次要求“未达成则继续”。本次继续检查是否误漏了计划 7.5 的 cover-boundary phase-specific loosening 分支，或计划 9 的 synthetic exploration-pass 分支。

执行：

```bash
python - <<'PY'
import csv, collections
base='results/v13_6_poprisk_snr_basis_cover_boundary/official_v136'
for name in [
    'v136_basis_cover_boundary.csv',
    'v136_synthetic_family_results.csv',
    'v136_snr_parameter_update.csv',
    'v136_snr_basis_channel_update.csv',
    'v136_controls.csv',
    'v136_substrate_snr_gate.csv',
]:
    p=f'{base}/{name}'
    rows=list(csv.DictReader(open(p)))
    print(name, 'rows', len(rows), 'cols', ','.join(rows[0].keys()) if rows else '')

rows=list(csv.DictReader(open(f'{base}/v136_basis_cover_boundary.csv')))
print('boundary_accept', collections.Counter(r['boundary_accept'] for r in rows))

syn=list(csv.DictReader(open(f'{base}/v136_synthetic_family_results.csv')))
print('official_synthetic_pass', collections.Counter(r['official_synthetic_pass'] for r in syn))
print('exploration_pass', collections.Counter(r['exploration_pass'] for r in syn))

def f(r,k):
    try:
        return float(r.get(k,'nan'))
    except Exception:
        return float('nan')
best=max(syn, key=lambda r: f(r,'source_vs_best_control'))
print('best_source_row', {
    k: best.get(k)
    for k in [
        'method','synthetic_task','seed','loss_interface','snr_variant',
        'source_vs_best_control','CouplingR2_delta','NoiseSignalLeak_delta',
        'RealSignalReservoirRatio_delta','CEp99_delta',
        'official_synthetic_pass','exploration_pass'
    ]
})
PY
```

输出摘要：

```text
v136_basis_cover_boundary.csv rows = 28
v136_synthetic_family_results.csv rows = 140
v136_snr_parameter_update.csv rows = 56
v136_snr_basis_channel_update.csv rows = 84
v136_controls.csv rows = 1120
v136_substrate_snr_gate.csv rows = 19
boundary_accept = {'1': 22, '0': 6}
official_synthetic_pass = {'0': 140}
exploration_pass = {'0': 140}
best_source_row:
  method = KAN-BasisCoverBoundary
  synthetic_task = X4
  seed = 0
  loss_interface = CE
  snr_variant = Hard
  source_vs_best_control = 0.0003762373062765681
  CouplingR2_delta = 2.0438709259033203
  NoiseSignalLeak_delta = -0.018877089023590088
  RealSignalReservoirRatio_delta = -0.34347105026245117
  CEp99_delta = -0.95526123046875
  official_synthetic_pass = 0
  exploration_pass = 0
```

判断：

```text
1. cover-boundary 不是全拒绝 no-op：boundary_accept = 22/28。
2. 因此计划 7.5 的 “如果 boundary guard 让所有 update 变成 no-op，则 try phase-specific loosening” 条件没有触发。
3. Line X 的 exploration_pass = 0/140，official_synthetic_pass = 0/140；当前 blocker 仍是 source_vs_best_control 不足，而不是 cover 全拒绝。
4. 本次没有新增训练实验、没有修改 gate、没有新增 CSV 指标。
```
