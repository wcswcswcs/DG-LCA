# DG-KAN v14.13 TransferMechanismClarification FunctionalContinueOpen AllBasisParallel 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
runner = experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
out_dir = results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413
recap = docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_实验结果复盘.md
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py --out-dir results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502
```

## 3. 主要 artifacts

```text
v1413_route_decision.json
v1413_progress_table.csv
v1413_line_r_audit.csv
v1413_forbidden_information_audit.csv
v1413_no_action_search_audit.csv
v1413_e3_transfer_observability_features.csv
v1413_e3_transfer_observability_summary.csv
v1413_e3_leaveout_predictivity.csv
v1413_v3_proxy_to_effect_chain.csv
v1413_v3_breakpoint_summary.csv
v1413_f3_dche_fms_real_lite.csv
v1413_f3_dche_fms_controls.csv
v1413_f3_failure_taxonomy.csv
v1413_d0_cross_version_reconciliation.csv
v1413_d_fou_hardening.csv
v1413_d_rbf_hardening.csv
v1413_d_wav_monitor.csv
v1413_dche_no_regression.csv
v1413_mlp_controls.csv
v1413_linec_tail_audit.csv
v1413_required_artifact_manifest.csv
v1413_code_review_packet.zip
v1413_no_go_boundary.md
v1413_next_hypothesis_queue.md
```

## 4. 复现说明

```text
1. 使用 /home/chengshun.wang/miniconda3/envs/kan/bin/python 执行 runner。
2. 如需重跑 F3 real-lite，请加 --force-f3-real-lite 1。
3. 若不加 --force-f3-real-lite，runner 会复用 v1413_f3_dche_fms_real_lite.csv。
4. 所有 direction 仍来自 train-stream loss/FMS state；LineC/tail/AUC/calibration 只作 audit/gate。
```

## 5. 用户继续要求后的 SameActiveFractionControl 补齐

触发原因：

```text
复核 v14.13 完整计划后发现，F3 control surface 要求包含 SameActiveFractionControl；
初始 runner 只记录 not_executed audit row，没有实际执行该 control。
```

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 C5-D-CHE-AdamW-SameActiveFractionControl。
  C5 使用 current train batch / FMS state 估计 active fraction，
  再随机选择同数量 active keys 执行 matched update control。

experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
  D_CHE_CONTROLS 纳入 C5。
  v1413_method_surface_manifest.csv 不再写 not_executed SameActiveFractionControl row。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
```

重跑 F3 real-lite：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py --out-dir results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502 --force-f3-real-lite 1
```

关键结果：

```text
route = R4-AllBasisSubstrateBlocked
f3_real_lite_pass_count = 1/9
f3_mean_source_vs_best_control = -0.032626233498255414
required_artifact_missing_count = 0
promotion_allowed = 0
```

F3 control surface：

```text
v1413_method_surface_manifest.csv rows = 17
C0..C5 controls executed_in_f3 = 1
FMS-M1..FMS-M5 executed_in_f3 = 1
not_executed SameActiveFractionControl rows = 0
```

## 6. 用户继续要求后的 Line D extra substrate-only hardening

触发原因：

```text
复核 v14.13 完整计划后发现，Line D 推荐方向中 HighFrequencyQuarantine 与
LocalTailCoverageAudit 没有被当前 v14.13 finalizer 的 exact replay 覆盖。
因此按 substrate-only hardening 执行，不进入 official FMS proof。
```

代码修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 D-FOU31-HighFrequencyQuarantine。
  新增 D-WAV28-LocalTailCoverageAudit。

experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
  新增读取 line_d_extra_highfreq_tail_v1413 artifact。
  将 extra D-FOU/D-WAV rows 合入 v1413_d_fou_hardening.csv / v1413_d_wav_monitor.csv。
  v1413_d0_cross_version_reconciliation.csv 新增 extra pass count 与 best replay-or-extra pass count。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v149_line_d_all_basis_substrate_repair.py experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
```

Line D extra 执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/line_d_extra_highfreq_tail_v1413 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU31-HighFrequencyQuarantine,D-WAV28-LocalTailCoverageAudit --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --linec-seeds 12319500,12319501,12319502 --compute-budgeted-run 1
```

Line D extra 结果：

```text
route = R8-NonRATSubstrateStillMissing
candidate_rows = 18
linec_rows = 54
best_family = D-FOU
best_family_dataset_seed_pass_count = 0/9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
promotion_allowed = 0
```

extra candidate summary：

| candidate | rows | pass | best delta vs MLP | best NLL ratio | best LineC | median step ratio |
|---|---:|---:|---:|---:|---:|---:|
| D-FOU31-HighFrequencyQuarantine | 9 | 0 | -0.156250 | 1.564545 | 0.666667 | 0.410670 |
| D-WAV28-LocalTailCoverageAudit | 9 | 0 | -0.187500 | 1.563148 | 0.666667 | 0.521715 |

合入 finalizer：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py --out-dir results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413 --real-lite-datasets MNIST,Fashion-MNIST,KMNIST --real-lite-seeds 0,1,2 --real-lite-train-size 1024 --real-lite-val-size 512 --real-lite-test-size 512 --real-lite-train-steps 200 --real-lite-batch-size 32 --real-lite-fms-update-interval 200 --real-lite-trace-interval 100 --real-lite-linec-seeds 12319500,12319501,12319502
```

合入后关键结果：

```text
v1413_d_fou_hardening.csv rows = 54
v1413_d_wav_monitor.csv rows = 36
D-FOU v1412_replay = 2/9, v1413_extra = 0/9, best = 2/9
D-RBF v1412_replay = 1/9, v1413_extra = 0/9, best = 1/9
D-WAV v1412_replay = missing, v1413_extra = 0/9, best = 0/9
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
```

## 7. 最终刷新 code packet / manifest

执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments.run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel import make_packet, write_required_manifest
out = Path("results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413")
make_packet(out)
rows = write_required_manifest(out)
make_packet(out)
rows = write_required_manifest(out)
print("manifest_rows", len(rows))
print("missing_sum", sum(int(float(r["missing"])) for r in rows))
print("packet_exists", (out / "v1413_code_review_packet.zip").exists())
PY
```

最终核验结果：

```text
manifest_rows = 35
missing_sum = 0
packet_exists = True
```

## 8. 用户再次追问后的状态复核

本次不新增训练，只复核计划 stop/continue 边界与最新 artifact。

读取计划边界：

```bash
rg -n "FMS-M|Fallback ladder|Line D gate|Stop / continue|Do not add methods|不允许扩成|未过 substrate gate" docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
sed -n '638,725p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
sed -n '811,837p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
sed -n '997,1042p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
```

读取 artifact：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out = Path("results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413")
route = json.loads((out / "v1413_route_decision.json").read_text())
for k in [
    "route",
    "minimum_success",
    "e3_exploration_gate_pass",
    "e3_promotion_enabling_gate_pass",
    "v3_dominant_breakpoint",
    "f3_real_lite_pass_count",
    "f3_mean_source_vs_best_control",
    "line_d_best_non_dche_family",
    "line_d_best_non_dche_dataset_seed_pass_count",
    "line_d_official_fms_eligible_family_count",
    "official_s5_reached",
    "promotion_allowed",
    "required_artifact_missing_count",
    "forbidden_information_violation_count",
    "no_action_search_violation_count",
]:
    print(k, route.get(k))
with (out / "v1413_required_artifact_manifest.csv").open() as f:
    rows = list(csv.DictReader(f))
print("manifest_rows", len(rows))
print("missing_sum", sum(int(float(r["missing"])) for r in rows))
PY
```

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e3_exploration_gate_pass = 1
e3_promotion_enabling_gate_pass = 0
v3_dominant_breakpoint = V3-B6-ControlEquivalent
f3_real_lite_pass_count = 1/9
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_family = D-FOU
line_d_best_non_dche_dataset_seed_pass_count = 2/9
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
manifest_rows = 35
missing_sum = 0
```

复核结论：

```text
F3 <3/9 的 fallback 已覆盖到 V3 taxonomy + Line D/M/C/Z。
FMS-M1..M5、C0..C5 controls、Line D v14.9/v14.12 replay、
D-FOU31 与 D-WAV28 extra hardening 均已执行。
Line D best non-D-CHE 仍为 2/9，不能进入 official FMS proof。
继续推进需要新 method/action/controller/reset route 或 audit-direction，违反 v14.13 当前计划。
```

## 9. 用户再次追问后的覆盖矩阵复核

本次不新增训练，只把计划列表与实际 artifact 做矩阵对齐。

执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out = Path("results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413")
route = json.loads((out / "v1413_route_decision.json").read_text())

def read(name):
    with (out / name).open() as f:
        return list(csv.DictReader(f))

f3 = read("v1413_f3_dche_fms_real_lite.csv")
controls = read("v1413_f3_dche_fms_controls.csv")
d_fou = read("v1413_d_fou_hardening.csv")
d_rbf = read("v1413_d_rbf_hardening.csv")
d_wav = read("v1413_d_wav_monitor.csv")
manifest = read("v1413_required_artifact_manifest.csv")

print("route", route.get("route"))
print("minimum_success", route.get("minimum_success"))
print("f3_real_lite_pass_count", route.get("f3_real_lite_pass_count"))
print("f3_mean_source_vs_best_control", route.get("f3_mean_source_vs_best_control"))
print("v3_dominant_breakpoint", route.get("v3_dominant_breakpoint"))
print("line_d_best_non_dche_dataset_seed_pass_count", route.get("line_d_best_non_dche_dataset_seed_pass_count"))
print("official_s5_reached", route.get("official_s5_reached"))
print("promotion_allowed", route.get("promotion_allowed"))
print("manifest_missing_sum", sum(int(float(r["missing"])) for r in manifest))
print("f3_rows", len(f3), "f3_methods", sorted({r.get("method", "") for r in f3}))
print("control_rows", len(controls), "control_prefixes", sorted({r.get("method", "").split("-")[0] for r in controls}))
print("d_fou_rows", len(d_fou), "candidates", sorted({r.get("candidate_id", "") for r in d_fou}))
print("d_rbf_rows", len(d_rbf), "candidates", sorted({r.get("candidate_id", "") for r in d_rbf}))
print("d_wav_rows", len(d_wav), "candidates", sorted({r.get("candidate_id", "") for r in d_wav}))
PY
```

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
f3_real_lite_pass_count = 1
f3_mean_source_vs_best_control = -0.032626233498255414
v3_dominant_breakpoint = V3-B6-ControlEquivalent
line_d_best_non_dche_dataset_seed_pass_count = 2
official_s5_reached = 0
promotion_allowed = 0
manifest_missing_sum = 0

f3_rows = 45
f3_methods =
  FMS-M1-GenericValueOnlyDegreeSafety
  FMS-M2-DegreeContinuousBoundary
  FMS-M3-MicroHorizonGatedBoundary
  FMS-M4-RecoveryLagSuppressedBoundary
  FMS-M5-ProjectionRetentionFloor

control_rows = 54
control_prefixes = C0,C1,C2,C3,C4,C5

d_fou_rows = 54
d_fou candidates =
  D-FOU26-NoMaterializeLifetimeAuditV2
  D-FOU27-LowFreqIdentityResidualV2
  D-FOU28-BandwiseSNRSafeWarmup
  D-FOU29-PhaseStableBandMixNoHighFreq
  D-FOU30-NoMaterializeLifetimeV3
  D-FOU31-HighFrequencyQuarantine

d_rbf_rows = 45
d_rbf candidates =
  D-RBF25-WidthConditionGuardNoTaskBranch
  D-RBF26-ActiveCenterOccupancyV2
  D-RBF27-WidthConditionIdentityResidual
  D-RBF28-CompactBumpNoDenseMaterialization
  D-RBF29-GaussianLocalK4TaskHealth

d_wav_rows = 36
d_wav candidates =
  D-WAV25-TriangularSupportV3
  D-WAV26-ScaleOccupancyNoTailTarget
  D-WAV27-LocalSupportOverlapDamping
  D-WAV28-LocalTailCoverageAudit
```

结论：

```text
v14.13 计划内 F3 method / control 与 Line D candidate 均无漏跑。
没有新训练启动；当前合法结论仍是 R4-AllBasisSubstrateBlocked。
```

## 10. 用户再次追问后的 V3 gate 复核

本次不新增训练；检查是否触发 `ambiguous_rows_fraction > 0.20` 的 V3 diagnostic fallback。

执行指令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out = Path("results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413")
route = json.loads((out / "v1413_route_decision.json").read_text())
print("v3_breakpoint_coverage", route.get("v3_breakpoint_coverage"))
print("v3_ambiguous_rows_fraction", route.get("v3_ambiguous_rows_fraction"))
print("v3_dominant_breakpoint", route.get("v3_dominant_breakpoint"))
for name in ["v1413_v3_proxy_to_effect_chain.csv", "v1413_v3_breakpoint_summary.csv"]:
    with (out / name).open() as f:
        rows = list(csv.DictReader(f))
    print(name, "rows", len(rows))
    if name.endswith("summary.csv"):
        for r in rows:
            print(r)
PY
```

复核结果：

```text
v3_breakpoint_coverage = 1
v3_ambiguous_rows_fraction = 0.0
v3_dominant_breakpoint = V3-B6-ControlEquivalent
v1413_v3_proxy_to_effect_chain.csv rows = 45
v1413_v3_breakpoint_summary.csv rows = 4

V3-B3-EventTooWeak rows = 1, fraction = 0.022222222222222223
V3-B5-MicroHorizonGoodRealBad rows = 7, fraction = 0.15555555555555556
V3-B6-ControlEquivalent rows = 36, fraction = 0.8
V3-Pass rows = 1, fraction = 0.022222222222222223
```

结论：

```text
V3 ambiguous diagnostics fallback 不触发；
主要断点为 ControlEquivalent。
没有新训练启动；当前合法结论仍是 R4-AllBasisSubstrateBlocked。
```

## 11. 用户再次追问后的 Line Z / no-go 复核

本次不新增训练；读取 route definitions、Line Z/no-go artifacts 与 audit/manifest。

执行指令：

```bash
sed -n '953,989p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
sed -n '1031,1042p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md
sed -n '1055,1067p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md

cat results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413/v1413_no_go_boundary.md
cat results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413/v1413_next_hypothesis_queue.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out = Path("results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413")
route = json.loads((out / "v1413_route_decision.json").read_text())
for k in [
    "route",
    "minimum_success",
    "e3_promotion_enabling_gate_pass",
    "v3_breakpoint_coverage",
    "v3_ambiguous_rows_fraction",
    "v3_dominant_breakpoint",
    "f3_real_lite_pass_count",
    "f3_mean_source_vs_best_control",
    "line_d_best_non_dche_dataset_seed_pass_count",
    "line_d_official_fms_eligible_family_count",
    "official_s5_reached",
    "promotion_allowed",
    "required_artifact_missing_count",
    "forbidden_information_violation_count",
    "no_action_search_violation_count",
]:
    print(k, route.get(k))
for name in [
    "v1413_forbidden_information_audit.csv",
    "v1413_no_action_search_audit.csv",
    "v1413_required_artifact_manifest.csv",
]:
    with (out / name).open() as f:
        rows = list(csv.DictReader(f))
    print(name, "rows", len(rows), "viol_or_missing_sum", sum(int(float(r.get("violation", r.get("missing", 0)) or 0)) for r in rows))
PY
```

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
e3_promotion_enabling_gate_pass = 0
v3_breakpoint_coverage = 1
v3_ambiguous_rows_fraction = 0.0
v3_dominant_breakpoint = V3-B6-ControlEquivalent
f3_real_lite_pass_count = 1
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_dataset_seed_pass_count = 2
line_d_official_fms_eligible_family_count = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
v1413_forbidden_information_audit.csv rows = 17, viol_or_missing_sum = 0
v1413_no_action_search_audit.csv rows = 17, viol_or_missing_sum = 0
v1413_required_artifact_manifest.csv rows = 35, viol_or_missing_sum = 0
```

结论：

```text
Line Z 已写明当前 no-go boundary 与下一步 queue。
v14.13 内没有剩余合法训练分支；没有新训练启动。
```

## 12. 用户再次追问后的 route precedence 复核

本次不新增训练；检查当前 `R4-AllBasisSubstrateBlocked` 是否掩盖了计划中的 `R5-GenericFMSConfound`。

执行指令：

```bash
sed -n '953,989p' docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out = Path("results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413")
route = json.loads((out / "v1413_route_decision.json").read_text())
print("route", route.get("route"))
print("f3_real_lite_pass_count", route.get("f3_real_lite_pass_count"))
print("f3_mean_source_vs_best_control", route.get("f3_mean_source_vs_best_control"))
print("line_d_best_non_dche_dataset_seed_pass_count", route.get("line_d_best_non_dche_dataset_seed_pass_count"))
print("v3_dominant_breakpoint", route.get("v3_dominant_breakpoint"))
with (out / "v1413_v3_breakpoint_summary.csv").open() as f:
    for row in csv.DictReader(f):
        if row.get("breakpoint") == "V3-B6-ControlEquivalent":
            print(row)
PY
```

复核结果：

```text
route = R4-AllBasisSubstrateBlocked
f3_real_lite_pass_count = 1
f3_mean_source_vs_best_control = -0.032626233498255414
line_d_best_non_dche_dataset_seed_pass_count = 2
v3_dominant_breakpoint = V3-B6-ControlEquivalent
V3-B6-ControlEquivalent rows = 36/45
V3-B6-ControlEquivalent mean_source_vs_best_control = -0.04305057144827313
```

结论：

```text
ControlEquivalent 已作为 V3 breakpoint 记录；
但 F3 没有整体 observed gains，且 Line D 仍低于 >=6/9 substrate gate。
因此 primary route 保持 R4-AllBasisSubstrateBlocked；
没有新训练启动。
```
