# DG-KAN v13.3 TaskFamilyRobustBasisNaturalFunctional 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志只记录实际执行的命令、文件与 artifact 路径；不把 smoke/diagnostic 写成 promotion。

## 1. 计划读取与代码基线

读取计划文件：

```bash
rg -n "v13\\.3|Task-Family|BM8|low-rank|Non-RAT|Stop|Case|Required artifacts|S3-|R4-|R5-" docs/DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md
sed -n '713,829p' docs/DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md
sed -n '400,620p' docs/DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md
```

读取 v13.2 runner 结构：

```bash
rg -n "def natural_update|class NaturalUpdate|def run_future_case|def run_basis_natural|def build_route|REQUIRED|FIGURES|write_rows|basis_natural_p3|synthetic_mechanism|mlp_analog|real_short" experiments/run_v132_loss_interface_basis_natural_update.py
sed -n '1,140p' experiments/run_v132_loss_interface_basis_natural_update.py
sed -n '380,560p' experiments/run_v132_loss_interface_basis_natural_update.py
sed -n '660,1040p' experiments/run_v132_loss_interface_basis_natural_update.py
sed -n '1040,1215p' experiments/run_v132_loss_interface_basis_natural_update.py
```

工作区状态审计：

```bash
git status --short
```

说明：工作区存在大量历史未跟踪/删除/修改文件。本轮只修改 v13.3 runner 与 v13.3 两份日志，不回滚无关改动。

## 2. v13.3 runner 创建与语法检查

创建 v13.3 runner：

```bash
cp experiments/run_v132_loss_interface_basis_natural_update.py experiments/run_v133_task_family_robust_basis_natural.py
perl -0pi -e 's/v13\\.2/v13.3/g; s/V132/V133/g; s/v132/v133/g; s/v13_2_deep_reset_loss_interface_basis_natural_update/v13_3_task_family_robust_basis_natural_functional/g; s/DeepReset_LossInterfaceFunctional_BasisNaturalUpdate/TaskFamilyRobustBasisNaturalFunctional/g; s/loss-interface basis-natural functional update runner/task-family robust basis-natural functional runner/g' experiments/run_v133_task_family_robust_basis_natural.py
```

随后用 `apply_patch` 修改：

```text
experiments/run_v133_task_family_robust_basis_natural.py
```

已完成的关键修改：

```text
1. v13.3 required artifact / figure 名称对齐计划文件。
2. NaturalUpdate 增加 low-rank/block metric telemetry 字段。
3. natural_update 新增 BM8/BM9/BM10/BM11/BM12/BM13/BM14/BM15 分支。
4. BM8/BM9/BM14/BM15 使用 train-stream per-example basis-parameter gradient 构造 low-rank metric，并用 Woodbury 逆。
5. BM10 使用 block-group diagonal metric。
6. 新增 task-family summary、autopsy、Non-RAT rescue taxonomy、progress table、no-go boundary 输出。
7. v13.3 route 改为按 task-family pass 计数：每个 X family 需要 >=2 substrates 或 >=2 seeds pass。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v133_task_family_robust_basis_natural.py
```

结果：

```text
py_compile pass
```

## 3. v13.3 smoke

执行：

```bash
conda run -n kan python experiments/run_v133_task_family_robust_basis_natural.py --out-dir results/v13_3_task_family_robust_basis_natural_functional/smoke_v133 --device cuda:0 --synthetic-tasks X1,X7 --synthetic-seeds 0 --synthetic-train-size 32 --synthetic-val-size 16 --synthetic-dim 16 --synthetic-classes 3 --loss-interfaces CE --update-candidates BM8-LowRankTangentNatural-r4,BM10-BlockGroupNatural --future-steps 10 --checkpoint-steps 5 --batch-size 16 --future-lr 0.003 --functional-eta 0.01 --metric-rho 0.001 --max-update-norm-ratio 0.01 --max-s1-per-family 1
```

结果：

```text
route = R2-BasisNaturalP3Failed
required_artifact_missing_count = 0
basis_natural_p3_rows = 4
basis_natural_p3_pass_count = 0
lowrank_metric_rows = 4
synthetic_rows = 2
full_basis_param_update_rows = 4
```

## 4. Non-RAT focused rescue

按计划 Non-RAT rescue 要至少执行一次 lifetime repair 与 task-health probe。本轮选择 Chebyshev lifetime repair candidate：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --run-id v133_nonrat_che12_rescue --out-dir results/v13_3_task_family_robust_basis_natural_functional/nonrat_rescue_che12 --artifact-prefix v133_nonrat_che12 --device cuda:0 --data-root data --no-download --datasets MNIST --seeds 0 --candidates D-CHE12-LifetimeRecomputeBackward-K3 --candidate-registry v1235 --train-size 64 --val-size 32 --batch-size 32 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 2 --workspace-profile-steps 5 --linec-batch-size 32 --linec-sketch-dim 32 --linec-seeds 0
```

结果 artifact：

```text
results/v13_3_task_family_robust_basis_natural_functional/nonrat_rescue_che12/v133_nonrat_che12_workspace_truth.csv
results/v13_3_task_family_robust_basis_natural_functional/nonrat_rescue_che12/v133_nonrat_che12_hardening.csv
results/v13_3_task_family_robust_basis_natural_functional/nonrat_rescue_che12/v133_nonrat_che12_linec.csv
```

关键结果：

```text
workspace_rows = 1
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
linec_executed = 0
skip_reason = workspace_gate_fail
raw_memory_ratio_vs_mlp = 1.1833379351891362
incremental_memory_ratio_vs_mlp = 2.83375104427736
step_ratio_vs_mlp = 1.271381889140255
```

解释：D-CHE12 focused rescue 没有打开 workspace gate；task-health/LineC probe 按 runner 规则被显式 skip，不能 promotion。

为把实际 rescue artifact 进入 v13.3 finalizer，随后用 `apply_patch` 修改：

```text
experiments/run_v133_task_family_robust_basis_natural.py
  build_nonrat_rescue() 读取 nonrat_rescue_che12 的 workspace/hardening/linec artifact。
```

再次语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v133_task_family_robust_basis_natural.py
```

结果：

```text
py_compile pass
```

## 5. official_v133: BM8/BM9/BM10/BM14

执行：

```bash
conda run -n kan python experiments/run_v133_task_family_robust_basis_natural.py --out-dir results/v13_3_task_family_robust_basis_natural_functional/official_v133 --device cuda:0 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --synthetic-classes 3 --loss-interfaces CE,Brier --update-candidates BM8-LowRankTangentNatural-r4,BM9-LowRankTangentNatural-r8,BM10-BlockGroupNatural,BM14-TrustRegionLowRankNatural --future-steps 50 --checkpoint-steps 40 --batch-size 32 --future-lr 0.01 --future-weight-decay 0.001 --functional-eta 0.02 --metric-rho 0.001 --max-update-norm-ratio 0.02 --max-s1-per-family 3
```

结果：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
required_artifact_missing_count = 0
basis_natural_p3_rows = 168
basis_natural_p3_pass_count = 0
lowrank_metric_rows = 168
lowrank_metric_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 5
nonrat_rescue_s1_count = 0
real_short_run_open_allowed = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 168
```

## 6. fallback BM11/BM12/BM13/BM15

因为 v13.3 计划 no-go 写明 BM8-BM15，本轮继续补跑 BM11/BM12/BM13/BM15：

```bash
conda run -n kan python experiments/run_v133_task_family_robust_basis_natural.py --out-dir results/v13_3_task_family_robust_basis_natural_functional/fallback_bm11_bm15_v133 --device cuda:0 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --synthetic-train-size 96 --synthetic-val-size 64 --synthetic-dim 16 --synthetic-classes 3 --loss-interfaces CE,Brier --update-candidates BM11-TaskFamilyBalancedMetric,BM12-LeaveFamilyOutMetric,BM13-KroneckerGroupNatural,BM15-ControlResidualizedNatural --future-steps 50 --checkpoint-steps 40 --batch-size 32 --future-lr 0.01 --future-weight-decay 0.001 --functional-eta 0.02 --metric-rho 0.001 --max-update-norm-ratio 0.02 --max-s1-per-family 3
```

结果：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
required_artifact_missing_count = 0
basis_natural_p3_rows = 168
basis_natural_p3_pass_count = 0
lowrank_metric_rows = 168
lowrank_metric_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_s1_count = 0
real_short_run_open_allowed = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 168
```

## 7. route finalizer refresh

route 逻辑修正：当 low-rank rows 已执行且 P3=0、Non-RAT rescue S1=0 时，合法 no-go route 应写为：

```text
R4-DiagonalAndLowRankMetricNoGo
```

修改：

```text
experiments/run_v133_task_family_robust_basis_natural.py
  build_route(): p3<=0 + lowrank_rows + nonrat_s1<=0 -> R4。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v133_task_family_robust_basis_natural.py
```

刷新 official/fallback route、manifest、code packet：

```bash
conda run -n kan bash -lc 'python - <<"PY"
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(".").resolve()))
from experiments.run_v133_task_family_robust_basis_natural import read_rows, build_task_family_summary, build_nonrat_rescue, build_route, write_manifest, write_no_go_boundary, build_progress_table, write_rows, write_code_packet
for name in ["official_v133","fallback_bm11_bm15_v133"]:
    out=Path("results/v13_3_task_family_robust_basis_natural_functional")/name
    substrate=read_rows(out/"v133_substrate_map.csv")
    p3=read_rows(out/"v133_lowrank_metric_rows.csv")
    synthetic=read_rows(out/"v133_synthetic_mechanism_v2.csv")
    family=read_rows(out/"v133_synthetic_family_summary.csv") or build_task_family_summary(synthetic)
    nonrat=read_rows(out/"v133_nonrat_substrate_rescue.csv") or build_nonrat_rescue(substrate)
    real=read_rows(out/"v133_real_short_run.csv")
    mlp=read_rows(out/"v133_mlp_analog_closure.csv")
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    write_no_go_boundary(out, route, family, nonrat)
    write_rows(out/"v133_progress_table.csv", build_progress_table(route))
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    manifest, code_sha=write_code_packet(out)
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,code_sha=code_sha,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    route["code_review_packet_entries"]=len(manifest)
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    write_code_packet(out)
    print(name, route["route"], route["minimum_success"], route["required_artifact_missing_count"])
PY'
```

输出：

```text
official_v133 R4-DiagonalAndLowRankMetricNoGo S1-EfficientControllableSubstrate 0
fallback_bm11_bm15_v133 R4-DiagonalAndLowRankMetricNoGo S1-EfficientControllableSubstrate 0
```

日志写入后刷新 code packet：

```bash
conda run -n kan bash -lc 'python - <<"PY"
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(".").resolve()))
from experiments.run_v133_task_family_robust_basis_natural import read_rows, build_task_family_summary, build_nonrat_rescue, build_route, write_manifest, write_no_go_boundary, build_progress_table, write_rows, write_code_packet
for name in ["official_v133","fallback_bm11_bm15_v133"]:
    out=Path("results/v13_3_task_family_robust_basis_natural_functional")/name
    substrate=read_rows(out/"v133_substrate_map.csv")
    p3=read_rows(out/"v133_lowrank_metric_rows.csv")
    synthetic=read_rows(out/"v133_synthetic_mechanism_v2.csv")
    family=read_rows(out/"v133_synthetic_family_summary.csv") or build_task_family_summary(synthetic)
    nonrat=read_rows(out/"v133_nonrat_substrate_rescue.csv") or build_nonrat_rescue(substrate)
    real=read_rows(out/"v133_real_short_run.csv")
    mlp=read_rows(out/"v133_mlp_analog_closure.csv")
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    write_no_go_boundary(out, route, family, nonrat)
    write_rows(out/"v133_progress_table.csv", build_progress_table(route))
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    manifest, code_sha=write_code_packet(out)
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,code_sha=code_sha,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    route["code_review_packet_entries"]=len(manifest)
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    write_code_packet(out)
    print(name, route["route"], route["code_review_packet_sha256"], route["required_artifact_missing_count"])
PY'
```

最终输出：

```text
official_v133 R4-DiagonalAndLowRankMetricNoGo 310b0812baae2322288622b6a0b99a876fd3d108054f967030f405a46cd0d046 0
fallback_bm11_bm15_v133 R4-DiagonalAndLowRankMetricNoGo e67bdac65e8a62b8b19a9079583f4ac0f3103ac3faf668dcafc274a3d1f4e4c6 0
```

## 8. 用户再次追问后的 Non-RAT task-health rescue 补跑

复核 stop rule 与现有 artifact：

```bash
cat results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_route_decision.json
sed -n '753,789p' docs/DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md
python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v13_3_task_family_robust_basis_natural_functional/official_v133')
for file in ['v133_nonrat_substrate_rescue.csv','v133_synthetic_family_summary.csv','v133_required_artifact_manifest.csv']:
 print('\n==', file)
 rows=list(csv.DictReader((base/file).open()))
 print('rows', len(rows))
 if file=='v133_required_artifact_manifest.csv':
  print('missing', sum(1 for r in rows if r.get('required')=='1' and r.get('exists')!='1'))
  continue
 for r in rows[:10]: print(r)
PY
```

复核发现：D-CHE12 focused run 是 lifetime repair candidate；task-health/LineC probe 因 workspace gate fail 被 skip。为更严格覆盖计划中的 task-health repair，先尝试旧候选名：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --run-id v133_nonrat_che6_taskhealth --out-dir results/v13_3_task_family_robust_basis_natural_functional/nonrat_rescue_che6_taskhealth --artifact-prefix v133_nonrat_che6 --device cuda:0 --data-root data --no-download --datasets MNIST --seeds 0 --candidates D-CHE6-K3DegreeEnergyCap --candidate-registry v1235 --train-size 64 --val-size 32 --batch-size 32 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 2 --workspace-profile-steps 5 --linec-batch-size 32 --linec-sketch-dim 32 --linec-seeds 0
```

结果：

```text
KeyError: 'D-CHE6-K3DegreeEnergyCap'
```

查询 v1235 registry：

```bash
conda run -n kan bash -lc 'python - <<"PY"
from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES
print(type(V1235_BASIS_CANDIDATES), len(V1235_BASIS_CANDIDATES))
seq=V1235_BASIS_CANDIDATES.values()
for c in seq:
    if getattr(c,"family",None)=="D-CHE":
        print(getattr(c,"candidate_id",getattr(c,"id",None)))
PY'
```

实际可用 Chebyshev candidates：

```text
D-CHE12-LifetimeRecomputeBackward-K3
D-CHE13-FusedReadoutGradNoMaterialize-K3
D-CHE14-OptimizerStateLifetimeReuse-K3
D-CHE15-FullStepNoMaterialize-K3
D-CHE16-DegreeEnergyDampingSubstrate
D-CHE17-HighDegreeLateEnableSubstrate
D-CHE18-RoleDegreeEnergyCapSubstrate
D-CHE19-ChebyTangentTrustSubstrate
D-CHE20-DegreeNormalizedReadoutHealthSubstrate
```

补跑 task-health focused candidate：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --run-id v133_nonrat_che20_taskhealth --out-dir results/v13_3_task_family_robust_basis_natural_functional/nonrat_rescue_che20_taskhealth --artifact-prefix v133_nonrat_che20 --device cuda:0 --data-root data --no-download --datasets MNIST --seeds 0 --candidates D-CHE20-DegreeNormalizedReadoutHealthSubstrate --candidate-registry v1235 --train-size 64 --val-size 32 --batch-size 32 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 2 --workspace-profile-steps 5 --linec-batch-size 32 --linec-sketch-dim 32 --linec-seeds 0
```

结果：

```text
workspace_rows = 1
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
linec_executed = 0
skip_reason = workspace_gate_fail
raw_memory_ratio_vs_mlp = 1.2288712066065914
incremental_memory_ratio_vs_mlp = 2.8111946532999164
step_ratio_vs_mlp = 1.1159531205667197
```

修改 finalizer 纳入 D-CHE20 artifact：

```text
experiments/run_v133_task_family_robust_basis_natural.py
  build_nonrat_rescue() 同时读取 nonrat_rescue_che12 与 nonrat_rescue_che20_taskhealth。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v133_task_family_robust_basis_natural.py
```

刷新 official/fallback route、manifest、packet：

```bash
conda run -n kan bash -lc 'python - <<"PY"
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(".").resolve()))
from experiments.run_v133_task_family_robust_basis_natural import read_rows, build_task_family_summary, build_nonrat_rescue, build_route, write_manifest, write_no_go_boundary, build_progress_table, write_rows, write_code_packet
for name in ["official_v133","fallback_bm11_bm15_v133"]:
    out=Path("results/v13_3_task_family_robust_basis_natural_functional")/name
    substrate=read_rows(out/"v133_substrate_map.csv")
    p3=read_rows(out/"v133_lowrank_metric_rows.csv")
    synthetic=read_rows(out/"v133_synthetic_mechanism_v2.csv")
    family=read_rows(out/"v133_synthetic_family_summary.csv") or build_task_family_summary(synthetic)
    nonrat=build_nonrat_rescue(substrate)
    write_rows(out/"v133_nonrat_substrate_rescue.csv", nonrat)
    real=read_rows(out/"v133_real_short_run.csv")
    mlp=read_rows(out/"v133_mlp_analog_closure.csv")
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    write_no_go_boundary(out, route, family, nonrat)
    write_rows(out/"v133_progress_table.csv", build_progress_table(route))
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    manifest, code_sha=write_code_packet(out)
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,code_sha=code_sha,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    route["code_review_packet_entries"]=len(manifest)
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    write_code_packet(out)
    print(name, route["route"], route["nonrat_rescue_rows"], route["nonrat_rescue_s1_count"], route["required_artifact_missing_count"])
PY'
```

输出：

```text
official_v133 R4-DiagonalAndLowRankMetricNoGo 6 0 0
fallback_bm11_bm15_v133 R4-DiagonalAndLowRankMetricNoGo 6 0 0
```

日志写入后最终刷新 code packet：

```bash
conda run -n kan bash -lc 'python - <<"PY"
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(".").resolve()))
from experiments.run_v133_task_family_robust_basis_natural import read_rows, build_task_family_summary, build_nonrat_rescue, build_route, write_manifest, write_no_go_boundary, build_progress_table, write_rows, write_code_packet
for name in ["official_v133","fallback_bm11_bm15_v133"]:
    out=Path("results/v13_3_task_family_robust_basis_natural_functional")/name
    substrate=read_rows(out/"v133_substrate_map.csv")
    p3=read_rows(out/"v133_lowrank_metric_rows.csv")
    synthetic=read_rows(out/"v133_synthetic_mechanism_v2.csv")
    family=read_rows(out/"v133_synthetic_family_summary.csv") or build_task_family_summary(synthetic)
    nonrat=build_nonrat_rescue(substrate)
    write_rows(out/"v133_nonrat_substrate_rescue.csv", nonrat)
    real=read_rows(out/"v133_real_short_run.csv")
    mlp=read_rows(out/"v133_mlp_analog_closure.csv")
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    write_no_go_boundary(out, route, family, nonrat)
    write_rows(out/"v133_progress_table.csv", build_progress_table(route))
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    manifest, code_sha=write_code_packet(out)
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,code_sha=code_sha,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    route["code_review_packet_entries"]=len(manifest)
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    write_code_packet(out)
    print(name, route["route"], route["nonrat_rescue_rows"], route["nonrat_rescue_s1_count"], route["code_review_packet_sha256"], route["required_artifact_missing_count"])
PY'
```

最终输出：

```text
official_v133 R4-DiagonalAndLowRankMetricNoGo 6 0 8469a0a67fb81174caacad0160f13f8405b9cf424ba27da8a956e08cbd6b7122 0
fallback_bm11_bm15_v133 R4-DiagonalAndLowRankMetricNoGo 6 0 3dee62e5c8597932d63066d008114b5fb46fb439f9242b98adfcfa739210e287 0
```

## 9. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次复核 final route、计划 stop rule、no-go boundary 与 required artifact：

```bash
cat results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_route_decision.json
sed -n '753,789p' docs/DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md
cat results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_no_go_boundary.md
python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v13_3_task_family_robust_basis_natural_functional/official_v133')
for file in ['v133_required_artifact_manifest.csv','v133_nonrat_substrate_rescue.csv','v133_synthetic_family_summary.csv','v133_mlp_analog_closure.csv']:
    rows=list(csv.DictReader((base/file).open()))
    print(file, 'rows', len(rows))
    if file == 'v133_required_artifact_manifest.csv':
        print('missing_required', sum(1 for r in rows if r.get('required')=='1' and r.get('exists')!='1'))
    elif file == 'v133_nonrat_substrate_rescue.csv':
        print('s1_pass', sum(int(float(r.get('S1_rescue_pass') or 0)) for r in rows))
        print('focused', [(r.get('candidate_id'), r.get('workspace_ok'), r.get('hardening_executed'), r.get('blocker_type')) for r in rows[:3]])
    elif file == 'v133_synthetic_family_summary.csv':
        print('task_family_pass', sum(int(float(r.get('task_family_pass') or 0)) for r in rows))
    elif file == 'v133_mlp_analog_closure.csv':
        print('mlp_pass', sum(int(float(r.get('mlp_analog_pass') or 0)) for r in rows))
PY
```

复核结果：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
mlp_analog_pass_count = 0
required_artifact_missing_count = 0
manifest_rows = 36
missing_required = 0
```

结论：本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因是 v13.3 计划 stop rule 已满足：BM8-BM15 coverage 没超过 2/7，Non-RAT lifetime/task-health rescue 均未打开 S1，MLP analog 完成，no-go boundary 已写明 pivot。继续在同一 BM family 上扩局部参数会成为低价值搜索。

## 10. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次对照最终 artifact 与计划 stop rule 后，结论仍不变：当前 v13.3 已触发合法 final stop，不再启动新的低价值局部网格实验。

复核结果：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
mlp_analog_pass_count = 0
required_artifact_missing_count = 0
manifest_rows = 36
missing_required = 0
```

再次刷新 official/fallback 的 nonrat rescue 汇总、route、manifest、no-go boundary、progress table 与 code packet：

```bash
conda run -n kan bash -lc 'python - <<"PY"
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(".").resolve()))
from experiments.run_v133_task_family_robust_basis_natural import read_rows, build_task_family_summary, build_nonrat_rescue, build_route, write_manifest, write_no_go_boundary, build_progress_table, write_rows, write_code_packet
for name in ["official_v133","fallback_bm11_bm15_v133"]:
    out=Path("results/v13_3_task_family_robust_basis_natural_functional")/name
    substrate=read_rows(out/"v133_substrate_map.csv")
    p3=read_rows(out/"v133_lowrank_metric_rows.csv")
    synthetic=read_rows(out/"v133_synthetic_mechanism_v2.csv")
    family=read_rows(out/"v133_synthetic_family_summary.csv") or build_task_family_summary(synthetic)
    nonrat=build_nonrat_rescue(substrate)
    write_rows(out/"v133_nonrat_substrate_rescue.csv", nonrat)
    real=read_rows(out/"v133_real_short_run.csv")
    mlp=read_rows(out/"v133_mlp_analog_closure.csv")
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    write_no_go_boundary(out, route, family, nonrat)
    write_rows(out/"v133_progress_table.csv", build_progress_table(route))
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    manifest, code_sha=write_code_packet(out)
    _m, missing=write_manifest(out)
    route=build_route(substrate,p3,synthetic,family,nonrat,real,mlp,missing,code_sha=code_sha,writeback_rows=read_rows(out/"v133_writeback_trace.csv"))
    route["code_review_packet_entries"]=len(manifest)
    (out/"v133_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    write_code_packet(out)
    print(name, route["route"], route["nonrat_rescue_rows"], route["nonrat_rescue_s1_count"], route["code_review_packet_sha256"], route["required_artifact_missing_count"])
PY'
```

说明：code packet 包含执行日志与复盘日志；任何后续日志编辑都会改变 zip hash，因此最终 hash 以刷新后的 `v133_route_decision.json` 为准。

## 11. 用户再次追问后的 stop-contract 复核 3

用户再次要求确认 v13.3 是否达成目标，若未达成则继续。本次继续执行的是 stop-contract 复核，不启动新的训练实验。

执行：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v13_3_task_family_robust_basis_natural_functional/official_v133/v133_route_decision.json')
r=json.loads(p.read_text())
keys=['route','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_all_executed','required_artifact_missing_count','basis_natural_p3_pass_count','lowrank_metric_pass_count','synthetic_task_success_count','synthetic_5of7_pass','nonrat_rescue_rows','nonrat_rescue_s1_count','mlp_analog_pass_count','code_review_packet_sha256']
for k in keys:
    print(f'{k}={r.get(k)}')
PY

sed -n '753,789p' docs/DG-KAN_v13.3_TaskFamilyRobustBasisNaturalFunctional_完整计划.md
```

复核结果：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_natural_p3_pass_count = 0
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_rows = 6
nonrat_rescue_s1_count = 0
mlp_analog_pass_count = 0
```

判断：v13.3 仍未达成 S3/S5，但计划 stop rule 已满足。没有新增训练实验、没有新增 CSV 指标、没有修改 gate。接下来刷新 official/fallback 的 route、manifest、no-go boundary、progress table 与 code packet，使本次复核日志进入审计包；刷新命令同上一节。
