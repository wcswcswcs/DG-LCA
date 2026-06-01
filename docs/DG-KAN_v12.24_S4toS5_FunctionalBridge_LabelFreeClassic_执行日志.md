# DG-KAN v12.24 S4toS5 FunctionalBridge LabelFreeClassic 执行日志

生成时间：2026-05-25（Asia/Singapore）

本日志只记录实际执行的命令、文件和产物路径；不记录未执行数据。代码环境固定使用 `conda env kan`。

## 1. 计划理解

输入计划文档：

```text
docs/DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_独立分析与下一步计划.md
```

本轮目标不是重复 v12.23 的 query-reference audit-only 成功，而是尝试把 S4 FunctionalP3Opened 推进到 S5：

- 替换 I24/I25 的 query-batch direct-logit compensation，测试 train-stream/precommit-safe compensation family。
- 继续 Line A/C：A66-A69 label-free LineC-aware frame repair。
- 继续 Line T/C：T1B online micro-probe 与 response-distillation fallback。
- 继续 Line I/B：I28-I32、policy-aware P3/P4、matched controls、multi-sketch 与 train-shuffle robustness。
- 继续 Line D：Rational/Fourier formal hardening，不再只停 smoke。
- 完成 R0-R12 核心代码审计 manifest、required artifact manifest、figures、route decision 和 code review zip。

关键约束：

```text
no_fake = 1
promotion_allowed 不能凭 audit-only 结果打开
query-reference compensation 不能写成 official S5
遇到 blocker 必须执行计划中 depth-2 fallback
```

## 2. 代码修改记录

### 2.1 注册 A66-A69

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A66-LineCAwareUnlabeledMultiSketchFrame
A67-ReservoirStabilizedResidualFrame
A68-A51TrainProbeCouplingPreservingFrame
A69-A51MultiSketchRMSQBoundQNoReadoutRepair
```

说明：这四个候选均设置 `uses_y_for_stats=0`，使用已有 primitive token 组合实现 label-free residual/RMSQ/boundQ/signal-block repair，不使用 label 或 CE vector 构造 frame。

### 2.2 新增 train-stream bridge 与 classic hardening 脚本

新增文件：

```text
experiments/run_v1224_train_stream_functional_bridge.py
experiments/run_v1224_classic_hardening.py
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1224_train_stream_functional_bridge.py experiments/run_v1224_classic_hardening.py
```

结果：命令退出码 0。

## 3. 并行实验启动记录

GPU 状态检查：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

结果：

```text
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 0 MiB, 97887 MiB, 0 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 0 MiB, 97887 MiB, 0 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 0 MiB, 97887 MiB, 0 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 0 MiB, 97887 MiB, 0 %
```

### 3.1 Line A：A66-A69 hardening

启动命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1224_linea_A66A69_hardening --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/linea/A66A69_hardening --artifact-prefix v1224_linea_A66A69_hardening --result-stage V1224_LINEA_A66A69_HARDENING --summary-stage V1224_LINEA_A66A69_SUMMARY --route-stage V1224_LINEA_A66A69_ROUTE --result-scope v1224_linea_A66A69_hardening --smoke-not-official 0 --official-training-result-available 1 --protocol-note "v12.24 Line A A66-A69 label-free LineC repair hardening" --route-impact "v12.24 Line A exploration evidence" --device cuda:0 --data-root data --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A63-A51RMSQReservoirRepair,A65-A58RMSQSignalBlockRepair,A66-LineCAwareUnlabeledMultiSketchFrame,A67-ReservoirStabilizedResidualFrame,A68-A51TrainProbeCouplingPreservingFrame,A69-A51MultiSketchRMSQBoundQNoReadoutRepair --seed-base 12240000 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 8 --linec-sketch-dim 24 > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/linea_A66A69_hardening.log 2>&1
```

### 3.2 Line I/B：train-stream/precommit bridge

启动命令：

```bash
conda run -n kan python experiments/run_v1224_train_stream_functional_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --compensation-batch 32 --ensemble-count 8 --train-seed-bases 12240400,12241400,12242400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1224_train_stream_functional_bridge > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/train_stream_bridge.log 2>&1
```

### 3.3 Line D：Rational/Fourier formal hardening

启动命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py --run-id v1224_classic_rational_fourier_hardening --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_classic_hardening --device cuda:3 --data-root data --families Rational,Fourier --datasets MNIST,Fashion-MNIST --seeds 0,1,2 --train-size 1024 --val-size 512 --batch-size 128 --epochs 8 --linec-batch-size 64 --linec-sketch-dim 24 --mlp-hidden 160 > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/classic_hardening.log 2>&1
```

## 4. 并行实验结束状态

### 4.1 Line A

日志：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/linea_A66A69_hardening.log
```

产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/linea/A66A69_hardening/v1224_linea_A66A69_hardening_ablation.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/linea/A66A69_hardening/v1224_linea_A66A69_hardening_summary.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/linea/A66A69_hardening/v1224_linea_A66A69_hardening_route.json
```

结束摘要：`rows=99`，`summary_rows=11`，best label-free candidate 为 `A51-StagedUnlabeledAdapt-warm1-r002`。

### 4.2 Line I/B train-stream bridge

日志：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/train_stream_bridge.log
```

产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_train_stream_functional_bridge.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_train_stream_functional_bridge_summary.json
```

结束摘要：12 个 candidate/train-shuffle summary rows，`any_strict_majority_pass=0`，`any_strict_all_pass=0`。

### 4.3 Line D

日志：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/classic_hardening.log
```

产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_hardening.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_hardening_summary.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_hardening_summary.json
```

结束摘要：`rows=12`，`exploration_pass_rows=0`。

## 5. Finalizer / 审计与打包

新增脚本：

```text
experiments/run_v1224_finalize_functional_bridge.py
```

语法检查和最终聚合命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize.log 2>&1
```

发现 `route_decision.json` 首次写入时 `t1b_microprobe_auc=NaN`，不利于标准 JSON 审计。修复：

```text
experiments/run_v1224_finalize_functional_bridge.py
```

修复内容：

```text
新增 json_safe()，将 NaN/inf 转为空字符串；
required manifest 增加 v1224_code_review_packet.zip；
重新运行 finalizer。
```

复跑命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py && conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_rerun_jsonsafe.log 2>&1
```

最终聚合产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_core_code_review_manifest.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_fallback_execution_manifest.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_required_artifact_manifest.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_online_microprobe.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_response_distillation_fallback.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip
```

最终 zip：

```text
path = results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip
entries/sha256 = 以最终 results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json 为准
```

required artifact 审计：

```text
required_artifact_rows = 28
required_artifact_missing_count = 0
```

## 6. 用户要求继续后的补充执行

### 6.1 继续原因

用户指出 v12.24 未达成目标则必须继续。重新核对原计划后，发现上一轮仍有两个 mandatory fallback 未充分执行：

```text
If I28/I29 fail -> I30/T1B-guided compensation.
If P3/P4 decouple -> I32 policy-aware P3.
```

同时 Line D 计划列出了 Rational/Fourier 具体 hardening 候选，而上一轮只跑了 B7b/B4g formal baseline。因此本轮继续补：

```text
I30-T1BGuidedDirectCompensation
I32-PolicyAwareP3QuadProbe
RationalB7me/B7lp/B7lz/B7ma
FourierB4p/B4q/B4v/B4w
```

### 6.2 新增/修改代码

新增：

```text
experiments/run_v1224_i30_i32_policy_bridge.py
```

修改：

```text
experiments/run_v1224_classic_hardening.py
```

修改内容：`FAMILY_SPECS` 增加计划中的 Rational/Fourier hardening 候选。

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1224_classic_hardening.py
```

结果：退出码 0。

### 6.3 I30/I32 bridge 命令

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --compensation-batch 32 --ensemble-count 8 --train-seed-bases 12240400,12241400,12242400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --candidates I30-T1BGuidedDirectCompensation,I32-PolicyAwareP3QuadProbe --i30-scales 0.5,1.0,1.5 --i32-policies quad_only,quad_direct,direct_only --i32-probe-steps 1,3,5 --i32-probe-lr 0.001 --artifact-prefix v1224_i30_i32_policy_bridge > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/i30_i32_policy_bridge.log 2>&1
```

### 6.4 Line D candidate fallback 命令

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py --run-id v1224_classic_candidate_fallback_hardening --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_classic_candidate_fallback_hardening --device cuda:3 --data-root data --families RationalB7me,RationalB7lp,RationalB7lz,RationalB7ma,FourierB4p,FourierB4q,FourierB4v,FourierB4w --datasets MNIST,Fashion-MNIST --seeds 0 --train-size 1024 --val-size 512 --batch-size 128 --epochs 8 --linec-batch-size 64 --linec-sketch-dim 24 --mlp-hidden 160 > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/classic_candidate_fallback_hardening.log 2>&1
```

### 6.5 补充实验完成后的 finalizer 修复

补充实验完成后，发现 finalizer 还没有把新增的 I30/I32 与 Line D candidate fallback 纳入 route、required manifest、fallback manifest、T1B microprobe 和 zip。因此修改：

```text
experiments/run_v1224_finalize_functional_bridge.py
```

修改内容：

```text
1. v1224_t1b_online_microprobe.csv 同时读取 v1224_train_stream_functional_bridge.csv 与 v1224_i30_i32_policy_bridge.csv。
2. core code review 增加 R13：experiments/run_v1224_i30_i32_policy_bridge.py。
3. fallback manifest 增加 I30、I32、LineD candidate fallback 三个触发项。
4. required manifest 增加 I30/I32 script、I30/I32 CSV/JSON、classic candidate fallback CSV/JSON。
5. route 增加 i30_i32_*、classic_candidate_*、combined_bridge_*、combined_classic_* 字段。
6. zip package 增加 experiments/run_v1224_i30_i32_policy_bridge.py。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1224_classic_hardening.py experiments/run_v1224_train_stream_functional_bridge.py
```

结果：退出码 0。

第一次补充 finalizer：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_after_i30_lined_candidate.log 2>&1
```

产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_i30_i32_policy_bridge.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_i30_i32_policy_bridge_summary.json
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_candidate_fallback_hardening.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_candidate_fallback_hardening_summary.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_candidate_fallback_hardening_summary.json
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_required_artifact_manifest.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_fallback_execution_manifest.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip
```

补充 finalizer 后的只读校验命令：

```bash
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json >/tmp/v1224_route_pretty.json && echo json_ok
python -c "import csv; rows=list(csv.DictReader(open('results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_required_artifact_manifest.csv'))); print('rows',len(rows),'missing',sum(int(r['exists'])==0 for r in rows)); [print(r) for r in rows if int(r['exists'])==0]"
python -c "import zipfile; zp='results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip'; zf=zipfile.ZipFile(zp); names=set(zf.namelist()); print('entries',len(names)); [print(n,n in names) for n in ['experiments/run_v1224_i30_i32_policy_bridge.py','results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_i30_i32_policy_bridge.csv','results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_candidate_fallback_hardening.csv']]"
```

只读校验结果：

```text
json_ok
required_artifact_rows = 33
required_artifact_missing_count = 0
zip entries = 56
zip contains experiments/run_v1224_i30_i32_policy_bridge.py = True
zip contains v1224_i30_i32_policy_bridge.csv = True
zip contains v1224_classic_candidate_fallback_hardening.csv = True
```

注意：本执行日志和复盘日志写入后，会再运行一次 finalizer，使最终 zip 包含最新日志。最终 zip sha256 以 `v1224_route_decision.json` 为准，避免在日志中写入会被后续打包动作改变的循环值。

最终补日志后 finalizer：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_after_logs_i30_lined_candidate.log 2>&1
```

最终只读复核命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1224_classic_hardening.py experiments/run_v1224_train_stream_functional_bridge.py
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json >/tmp/v1224_route_final_pretty.json && echo json_ok
python -c "import csv, zipfile; from pathlib import Path; rows=list(csv.DictReader(open('results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_required_artifact_manifest.csv'))); print('manifest_rows',len(rows)); print('manifest_missing',sum(int(r['exists'])==0 for r in rows)); zp='results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip'; zf=zipfile.ZipFile(zp); names=set(zf.namelist()); print('zip_entries',len(names)); [print('contains',n,n in names) for n in ['docs/DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_执行日志.md','docs/DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_实验结果复盘.md','experiments/run_v1224_i30_i32_policy_bridge.py','results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_i30_i32_policy_bridge.csv','results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_candidate_fallback_hardening.csv']]"
```

最终只读复核结果：

```text
py_compile exit code = 0
json_ok
manifest_rows = 33
manifest_missing = 0
zip_entries = 57
zip contains execution log = True
zip contains review log = True
zip contains experiments/run_v1224_i30_i32_policy_bridge.py = True
zip contains v1224_i30_i32_policy_bridge.csv = True
zip contains v1224_classic_candidate_fallback_hardening.csv = True
```

## 7. 用户继续要求后的 transitive code packet 与 T1B response-distilled fallback

用户继续要求确认是否达成目标；复核 v12.24 计划后，确认 S5 仍未达成，且还存在两个必须补齐的审计点：

```text
1. v12.24 final code packet 必须包含 v12.23 后续 P4/multisketch/precommit/LineD 脚本。
2. T2/T3 response-level signal 存在但 T1B fail 时，计划要求继续做 response-distillation feature generation。
```

### 7.1 只读核对当前 zip 缺口

命令：

```bash
python - <<'PY'
import json, zipfile
from pathlib import Path
out=Path('results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5')
route=json.loads((out/'v1224_route_decision.json').read_text())
zp=out/'v1224_code_review_packet.zip'
with zipfile.ZipFile(zp) as z:
    names=set(z.namelist())
    required=[
      'experiments/run_v1223_p4_compensation_modes.py',
      'experiments/run_v1223_p4_official_row_scan.py',
      'experiments/run_v1223_shadowp4_coupling_preserving_blend.py',
      'experiments/run_v1223_p4_trainable_role_scan.py',
      'experiments/run_v1223_p4_trajectory_gate_verifier.py',
      'experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py',
      'experiments/run_v1223_line_d_hardening.py',
      'experiments/summarize_v1223_line_d_hardening.py',
    ]
    for item in required:
        print(item, item in names)
PY
```

只读结论：上述 v12.23 continuation 审计脚本在旧 v12.24 zip 中不完整；这属于 code packet 审计缺口，不是实验成功或失败数据。

### 7.2 代码修改

修改文件：

```text
experiments/run_v1224_finalize_functional_bridge.py
```

修改内容：

```text
1. 新增 TRANSITIVE_CODE_PACKET_MEMBERS，显式打包 v12.23/v12.24 transitive core scripts。
2. 新增 v1224_t1b_response_distilled_features.csv 与 summary JSON 生成逻辑。
3. response-distilled features 只使用 train-stream/unlabeled probe 字段；T2/T3 response artifact 仅作为 teacher boundary 诊断，不作为 deployable feature。
4. fallback manifest 增加 T2_T3_pass_but_T1B_fail -> response_distilled_T1B_unlabeled_feature_generation。
5. required artifact manifest 纳入新增 CSV/JSON 与 transitive code packet。
6. route decision 记录 transitive_code_packet_missing_count 与 response-distilled T1B summary 字段。
7. 修复 summary JSON 初版 NaN 问题，改成 JSON-safe 空字符串和 json_safe 写入。
```

审计说明：这次修改没有降低 gate，没有把 diagnostic 结果写成 promotion；新增 T1B response-distilled artifact 的 `promotion_allowed=0`。

### 7.3 语法检查与第一次 finalizer

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_transitive_packet_response_distill.log 2>&1
```

结果：

```text
py_compile exit code = 0
first finalizer completed
```

第一次 finalizer 暴露问题：`v1224_t1b_response_distilled_features_summary.json` 中存在非标准 JSON 的 `NaN`。这是 JSON 序列化问题，不是实验结果 blocker；随后按 JSON-safe 方向修复。

### 7.4 JSON-safe 修复后 finalizer

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_transitive_packet_response_distill_jsonsafe.log 2>&1
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features_summary.json >/tmp/v1224_t1b_distill.json && echo t1b_json_ok
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json >/tmp/v1224_route_after_distill.json && echo route_json_ok
```

结果：

```text
py_compile exit code = 0
t1b_json_ok
route_json_ok
```

### 7.5 只读复核命令

命令：

```bash
python - <<'PY'
import json, zipfile, hashlib
from pathlib import Path
out=Path('results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5')
route=json.loads((out/'v1224_route_decision.json').read_text())
print('route', route.get('route'))
print('official_success_reached', route.get('official_success_reached'))
print('p4_pass', route.get('p4_pass'))
print('final_stop_allowed', route.get('final_stop_allowed'))
print('required_artifact_missing_count', route.get('required_artifact_missing_count'))
print('fallback_rows', route.get('fallback_rows'))
print('fallback_all_executed', route.get('fallback_all_executed'))
print('t1b_response_distilled_rows', route.get('t1b_response_distilled_rows'))
print('t1b_response_distilled_positive_rows', route.get('t1b_response_distilled_positive_rows'))
print('t1b_response_distilled_class_count', route.get('t1b_response_distilled_class_count'))
print('t1b_response_distilled_auc_joint', route.get('t1b_response_distilled_auc_joint'))
print('transitive_code_packet_missing_count', route.get('transitive_code_packet_missing_count'))
zp=out/'v1224_code_review_packet.zip'
data=zp.read_bytes()
print('zip_sha256', hashlib.sha256(data).hexdigest())
with zipfile.ZipFile(zp) as z:
    names=set(z.namelist())
    print('zip_entries', len(names))
    for item in [
      'experiments/run_v1223_p4_compensation_modes.py',
      'experiments/run_v1223_p4_official_row_scan.py',
      'experiments/run_v1223_shadowp4_coupling_preserving_blend.py',
      'experiments/run_v1223_p4_trainable_role_scan.py',
      'experiments/run_v1223_p4_trajectory_gate_verifier.py',
      'experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py',
      'experiments/run_v1223_line_d_hardening.py',
      'experiments/summarize_v1223_line_d_hardening.py',
      'results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features.csv',
      'results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features_summary.json',
    ]:
        print('zip_has', item, item in names)
PY
```

只读复核结果：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
fallback_rows = 10
fallback_all_executed = 1
t1b_response_distilled_rows = 18
t1b_response_distilled_positive_rows = 0
t1b_response_distilled_class_count = 1
t1b_response_distilled_auc_joint = blank
transitive_code_packet_missing_count = 0
zip_entries = 69
zip contains all listed v12.23 transitive scripts = True
zip contains v1224_t1b_response_distilled_features.csv = True
zip contains v1224_t1b_response_distilled_features_summary.json = True
```

注意：本执行日志和复盘日志写入后，会再运行一次 finalizer，使最终 zip 包含最新日志。最终 zip sha256 以最终只读复核和 `v1224_route_decision.json` 为准，不在日志中手写一个会因日志自引用而变化的值。

### 7.6 最终日志写入后 finalizer 与批量语法复核

日志写入后重新运行 finalizer：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py && conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_after_logs_transitive_response_distill.log 2>&1
```

结果：

```text
exit code = 0
```

最终只读复核命令：

```bash
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_route_decision.json >/tmp/v1224_route_final_transitive.json
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features_summary.json >/tmp/v1224_t1b_final_transitive.json
python - <<'PY'
import csv, json, zipfile
from pathlib import Path
out=Path('results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5')
route=json.loads((out/'v1224_route_decision.json').read_text())
manifest=list(csv.DictReader(open(out/'v1224_required_artifact_manifest.csv')))
print(route['route'], route['official_success_reached'], route['p4_pass'])
print('manifest_missing', sum(int(r['exists'])==0 for r in manifest))
with zipfile.ZipFile(out/'v1224_code_review_packet.zip') as z:
    names=set(z.namelist())
    print('zip_entries', len(names))
    for item in [
      'docs/DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_执行日志.md',
      'docs/DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_实验结果复盘.md',
      'experiments/run_v1223_p4_compensation_modes.py',
      'experiments/run_v1223_p4_official_row_scan.py',
      'experiments/run_v1223_shadowp4_coupling_preserving_blend.py',
      'experiments/run_v1223_p4_trainable_role_scan.py',
      'experiments/run_v1223_p4_trajectory_gate_verifier.py',
      'experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py',
      'experiments/run_v1223_line_d_hardening.py',
      'experiments/summarize_v1223_line_d_hardening.py',
      'results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features.csv',
      'results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features_summary.json',
    ]:
        print(item, item in names)
PY
```

复核结果：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
manifest_missing = 0
zip_entries = 70
zip contains execution log = True
zip contains review log = True
zip contains all listed transitive scripts and response-distilled artifacts = True
```

批量语法复核命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py experiments/run_v1224_train_stream_functional_bridge.py experiments/run_v1224_classic_hardening.py experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1223_p4_compensation_modes.py experiments/run_v1223_p4_official_row_scan.py experiments/run_v1223_shadowp4_coupling_preserving_blend.py experiments/run_v1223_p4_trainable_role_scan.py experiments/run_v1223_p4_trajectory_gate_verifier.py experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py experiments/run_v1223_line_d_hardening.py experiments/summarize_v1223_line_d_hardening.py
```

结果：

```text
exit code = 0
```

注意：本节写入后再次运行 finalizer，最终 zip sha256 仍以最后一次只读复核和最终回复为准；日志中不记录自引用 sha。

## 8. 用户再次要求继续后的 Depth-2 Line D repair 与 plan-contract 审计补齐

用户再次要求“达成目标了吗，没有请继续”。复核 v12.24 计划后，本轮继续点不是重复 I24 query-reference，也不是降低 gate，而是补齐两个 plan-contract 条件并执行一个真实 Depth-2 实验：

```text
1. 合法停止条件需要 hard_budget_exhausted=1 且 fallback_depth>=2。
2. 失败时必须输出 executable next-generation fallback，而不是只写 no-go。
3. Line D candidate fallback 失败后，按 Rational task-good / efficiency-LineC blocker 继续 readscale/crossWarm/pairStd/pairNorm depth-2 repair。
```

### 8.1 代码修改

修改文件：

```text
experiments/run_v1224_classic_hardening.py
experiments/run_v1224_finalize_functional_bridge.py
```

修改内容：

```text
1. 在 run_v1224_classic_hardening.py 增加 RationalB7dq/B7dr/B7ds/B7dt/B7em/B7en/B7eq/B7er 映射。
2. 在 finalizer 中纳入 v1224_classic_depth2_efficiency_hardening.csv/json。
3. route 增加 hard_budget_exhausted=1、fallback_depth=2、classic_depth2_rows、classic_depth2_exploration_pass_rows。
4. fallback manifest 增加 LineD_candidate_fallback_fail -> Rational_readscale_crosswarm_pairnorm_depth2_hardening。
5. 新增 v1224_next_generation_fallback_plan.csv/json/md，记录失败后的可执行下一代 fallback 命令。
6. 强化 v1224_core_code_review_manifest.csv：修复 line range 查找，增加 artifact_fields、gate_relation、review_status 字段。
```

### 8.2 语法检查

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py experiments/run_v1224_classic_hardening.py
```

结果：

```text
exit code = 0
```

### 8.3 Line D Depth-2 实验

命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 \
  --artifact-prefix v1224_classic_depth2_efficiency_hardening \
  --device cuda:0 \
  --families RationalB7dq,RationalB7dr,RationalB7ds,RationalB7dt,RationalB7em,RationalB7en,RationalB7eq,RationalB7er \
  --datasets MNIST,Fashion-MNIST \
  --seeds 0 \
  --train-size 1024 \
  --val-size 512 \
  --epochs 8 \
  --no-download \
  > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/classic_depth2_efficiency_hardening_cuda0.log 2>&1
```

产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_depth2_efficiency_hardening.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_depth2_efficiency_hardening_summary.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_depth2_efficiency_hardening_summary.json
```

只读复核命令：

```bash
python -m json.tool results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_depth2_efficiency_hardening_summary.json
python - <<'PY'
import csv
p='results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_classic_depth2_efficiency_hardening.csv'
rows=list(csv.DictReader(open(p)))
print('rows',len(rows))
print('exploration',sum(int(float(r['exploration_gate_pass'])) for r in rows))
for r in sorted(rows, key=lambda x:(-float(x['mean_delta_vs_MLP']) if x['mean_delta_vs_MLP'] else -999, -float(x['linec_CouplingR2']) if x['linec_CouplingR2'] else -999)):
    print(r['family'], r['dataset'], r['mean_delta_vs_MLP'], r['step_ratio_vs_mlp'], r['memory_ratio_vs_mlp'], r['linec_CouplingR2'], r['linec_pass'], r['exploration_gate_pass'])
PY
```

复核结果：

```text
rows = 16
summary_rows = 8
exploration_pass_rows = 0
promotion_allowed = 0
```

最接近 task gate 的行：

```text
RationalB7en / MNIST: delta_vs_MLP=-0.021484375, step_ratio=1.3622519040172953, memory_ratio=2.1016346983201215, CouplingR2=0.11629072993049316, linec_pass=0
RationalB7em / MNIST: delta_vs_MLP=-0.021484375, step_ratio=1.345775430701884, memory_ratio=2.1016346983201215, CouplingR2=0.11583633984652064, linec_pass=0
RationalB7eq / MNIST: delta_vs_MLP=-0.021484375, step_ratio=1.3481769451810812, memory_ratio=2.1016346983201215, CouplingR2=0.11543581587342444, linec_pass=0
RationalB7er / MNIST: delta_vs_MLP=-0.021484375, step_ratio=1.3394975935980358, memory_ratio=2.1016346983201215, CouplingR2=0.11543569641139528, linec_pass=0
```

最接近 LineC 的强 task 行：

```text
RationalB7en / Fashion-MNIST: delta_vs_MLP=-0.0234375, step_ratio=1.2194536746032019, memory_ratio=2.1016346983201215, CouplingR2=0.17596690876225063, linec_pass=1
RationalB7em / Fashion-MNIST: delta_vs_MLP=-0.0234375, step_ratio=1.223651174537795, memory_ratio=2.1016346983201215, CouplingR2=0.16910627443233062, linec_pass=1
```

### 8.4 finalizer 与 route 复核

命令：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_after_lined_depth2_core_nextgen.log 2>&1
```

只读复核结果：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_budget_exhausted = 1
fallback_depth = 2
fallback_rows = 12
fallback_all_executed = 1
required_artifact_rows = 59
required_artifact_missing_count = 0
classic_depth2_rows = 16
classic_depth2_exploration_pass_rows = 0
combined_classic_exploration_pass_rows = 0
next_generation_fallback_ready = 1
next_generation_fallback_rows = 3
code_semantics_review_pass = 1
```

新增产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_next_generation_fallback_plan.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_next_generation_fallback_plan.json
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_next_generation_fallback_plan.md
```

### 8.5 批量语法复核

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py experiments/run_v1224_classic_hardening.py experiments/run_v1224_train_stream_functional_bridge.py experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1223_p4_compensation_modes.py experiments/run_v1223_p4_official_row_scan.py experiments/run_v1223_shadowp4_coupling_preserving_blend.py experiments/run_v1223_p4_trainable_role_scan.py experiments/run_v1223_p4_trajectory_gate_verifier.py experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py experiments/run_v1223_line_d_hardening.py experiments/summarize_v1223_line_d_hardening.py
```

结果：

```text
exit code = 0
```

注意：本日志写入后仍需再运行一次 finalizer，使最终 zip 包含本节日志；最终 sha256 以后续只读复核为准。

## 9. 用户再次要求继续后的 NG2-NG7 bridge 修复执行

### 9.1 当前状态复核与 finalizer 修复

复核结论：v12.24 没有达成目标，当前仍是：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
```

先修复 finalizer 的审计覆盖范围：

```text
1. 修正 next-generation fallback command：补上 --source-out-dir，把 --candidate-names 改为脚本真实支持的 --candidates。
2. 新增 NG2/NG3/NG4/NG5/NG6/NG7 独立 artifact 前缀纳入 route、fallback manifest、required manifest、T1B response-distilled feature source。
3. train-stream / I30-I32 脚本新增 --label-smoothing 参数，用于公平 post-P3 tail-risk repair；source/noop/control 共享同一 smoothing。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_train_stream_functional_bridge.py experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1224_finalize_functional_bridge.py
```

结果：

```text
exit code = 0
```

### 9.2 NG2 train-stream bridge replay

命令：

```bash
conda run -n kan python experiments/run_v1224_train_stream_functional_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng2_train_stream_bridge_replay --device cuda:1 --candidates I28-TrainStreamEMACompensation,I29-TrainProbeMedianCompensation,I31-NullLogitCompensatedShadowRelease --train-seed-bases 12240400,12241400,12242400 --ensemble-count 16 --compensation-batch 32 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng2_train_stream_bridge_replay_cuda1.log 2>&1
```

结果：

```text
candidate_rows = 9
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
```

best aggregates：

```text
I29-TrainProbeMedianCompensation: best_source_vs_noop=0.01171875, best_source_vs_control=0.0
I31-NullLogitCompensatedShadowRelease: best_source_vs_noop=0.01171875, best_source_vs_control=-0.0078125
I28-TrainStreamEMACompensation: best_source_vs_noop=-0.015625, best_source_vs_control=-0.01171875
```

### 9.3 NG3 policy-aware probe reset

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng3_policy_probe_reset --device cuda:2 --candidates I30-T1BGuidedDirectCompensation,I32-PolicyAwareP3QuadProbe --i30-scales 0.35,0.50,0.75,1.00 --i32-probe-steps 1,3,5,8 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng3_policy_probe_reset_cuda2.log 2>&1
```

结果：

```text
candidate_rows = 6
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
I30 best_source_vs_noop = 0.02734375
I30 best_source_vs_control = -0.0078125
I32 best_source_vs_noop = 0.0
I32 best_source_vs_control = -0.00390625
```

### 9.4 NG4 lower-lr longer-epoch repair

理由：NG3 的 I30 已经能赢 NoOp，但仍输 matched control。因此尝试更长训练、更低 lr、更宽 scale。

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng4_policy_probe_lr_epoch_repair --device cuda:3 --candidates I30-T1BGuidedDirectCompensation,I32-PolicyAwareP3QuadProbe --i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 --i32-probe-steps 2,4,6,10 --epochs 12 --lr 0.0015 --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng4_policy_probe_lr_epoch_repair_cuda3.log 2>&1
```

结果：

```text
candidate_rows = 6
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
I30 best_source_vs_noop = 0.01953125
I30 best_source_vs_control = 0.0078125
```

最接近行：

```text
train_seed_base = 12240400
source_acc = 0.73828125
noop_acc = 0.71875
control_acc = 0.73046875
task_gate_pass = 0
LineC pass = 3/5
source_NLL = 1.106278896331787 <= noop_NLL = 1.138135313987732
source_CEp99 = 10.711200714111328 > noop_CEp99 + 0.05 = 9.37567024230957
```

解释：这是本轮最接近的 train-stream/precommit 点；acc、NLL、time、LineC majority 接近，但 CEp99 tail-risk gate 失败，不能 promotion。

### 9.5 NG5 tail-risk weight decay repair

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng5_tailrisk_weightdecay_repair --device cuda:0 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 --epochs 12 --lr 0.0015 --weight-decay 0.005 --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng5_tailrisk_weightdecay_repair_cuda0.log 2>&1
```

结果：

```text
candidate_rows = 3
any_strict_majority_pass = 0
best_source_vs_noop = 0.01953125
best_source_vs_control = 0.0078125
best source_CEp99 = 10.704131126403809
best noop_CEp99 = 9.319934844970703
```

解释：提高 weight decay 没有修复 CEp99 tail blocker。

### 9.6 NG6 fair label-smoothing tail repair

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng6_label_smoothing_tail_repair --device cuda:1 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 --epochs 12 --lr 0.0015 --label-smoothing 0.05 --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng6_label_smoothing_tail_repair_cuda1.log 2>&1
```

结果：

```text
candidate_rows = 3
any_strict_majority_pass = 0
best_source_vs_noop = 0.0078125
best_source_vs_control = -0.00390625
```

解释：label smoothing 0.05 明显降低 CEp99，但 source 不再赢 matched control，LineC 也没有 robust pass。

### 9.7 NG7 light label-smoothing interpolation

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng7_light_label_smoothing_tail_repair --device cuda:2 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 --epochs 12 --lr 0.0015 --label-smoothing 0.02 --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng7_light_label_smoothing_tail_repair_cuda2.log 2>&1
```

结果：

```text
candidate_rows = 3
any_strict_majority_pass = 0
best_source_vs_noop = 0.00390625
best_source_vs_control = -0.00390625
```

解释：轻度 smoothing 仍然丢掉 source-control advantage；NG4 的 acc/control-positive 点和 NG6/NG7 的 tail-risk 修复不重合。

### 9.8 finalizer 聚合结果

命令：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/finalize_after_ng2_to_ng7.log 2>&1
```

只读复核：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 18
fallback_all_executed = 1
required_artifact_rows = 71
required_artifact_missing_count = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
t1b_response_distilled_rows = 48
t1b_response_distilled_positive_rows = 0
code_review_packet_entries = 101
code_review_packet_sha256 = 984ac0d57241c33795fc6939463d9f22374f11599c34500a2efb2237c16126bd
```

本日志写入后仍需再跑 finalizer，使 zip 包含本节新增日志；最终 sha256 以后续只读复核为准。


## 10. 用户追问权限提示后的 NG8-NG10 继续推进（2026-05-25）

### 10.1 权限提示说明

本轮继续时，普通只读命令和带日志重定向的实验命令都曾触发：

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

因此出现权限确认。该权限提示只与当前沙箱运行 CUDA/日志重定向有关，不是实验 gate、promotion 口径或阈值的人工放行。本轮没有降低 gate，没有把 audit-only 写成 success，也没有编造数据。

### 10.2 代码修改与语法检查

修改内容：

```text
experiments/run_v1224_i30_i32_policy_bridge.py
experiments/run_v1224_finalize_functional_bridge.py
```

新增能力：

```text
1. NG8 tail_safe precommit probe score：只用 train-stream unlabeled logits 的 drift/tail/confidence/logit-abs/entropy observable。
2. NG9 quad_direct role-policy repair：source/noop/control 共享同一 post-P3 role policy，不改变 loss 或 gate。
3. NG10 contrast_tail_safe precommit selector：对 source 与 matched train-stream control 的 unlabeled probe score 做 contrast，仍不使用 label/CE/LineC。
4. finalizer 纳入 NG8/NG9/NG10 artifact、required manifest、fallback manifest、route fields 和 zip。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_i30_i32_policy_bridge.py experiments/run_v1224_finalize_functional_bridge.py
```

结果：

```text
py_compile pass
```

### 10.3 NG8 tail-safe precommit probe

首次沙箱执行因 `bwrap` 失败，随后按系统要求以提权方式重跑同一命令。

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng8_tail_safe_precommit_probe --device cuda:3 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng8_tail_safe_precommit_probe_cuda3.log 2>&1
```

结果：

```text
candidate_rows = 3
selected_actuator = I26-TrainDirectLogitCompensatedQuadRelease
selected_budget = 0.0045
selected_sign = -1.0
selected_comp_mode = median
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
best_source_vs_noop = 0.01953125
best_source_vs_best_control = 0.0078125
best_linec_seed_pass_count = 3/5
```

### 10.4 NG9 tail-safe + quad_direct repair

首次沙箱执行因 `bwrap` 失败，随后按系统要求以提权方式重跑同一命令。

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng9_tail_safe_quaddirect_repair --device cuda:0 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng9_tail_safe_quaddirect_repair_cuda0.log 2>&1
```

结果：

```text
candidate_rows = 3
selected_actuator = I26-TrainDirectLogitCompensatedQuadRelease
selected_budget = 0.0045
selected_sign = -1.0
selected_comp_mode = median
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
best_source_vs_noop = 0.02734375
best_source_vs_best_control = 0.00390625
best_linec_seed_pass_count = 3/5
```

### 10.5 NG10 contrast-tail-safe selector

首次沙箱执行因 `bwrap` 失败，随后按系统要求以提权方式重跑同一命令。

命令：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng10_contrast_tail_safe_probe --device cuda:1 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode contrast_tail_safe --probe-control-contrast-weight 1.00 --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download > results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/logs/ng10_contrast_tail_safe_probe_cuda1.log 2>&1
```

结果：

```text
candidate_rows = 3
selected_actuator = I27-TrainDirectLogitCompensatedShadowRelease
selected_budget = 0.009
selected_sign = -1.0
selected_comp_mode = median
probe_source_raw_score = -0.9014347008615732
probe_control_raw_score = -1.1697777668386697
probe_source_control_score_delta = 0.26834306597709656
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
best_source_vs_noop = 0.0234375
best_source_vs_best_control = 0.0
best_linec_seed_pass_count = 3/5
```

### 10.6 finalizer 命令

写入本节后执行 finalizer，使 NG8-NG10 和本轮日志进入 required artifact/zip 审计：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py
```


### 10.7 finalizer 结果

执行结果：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 18
fallback_all_executed = 1
required_artifact_rows = 77
required_artifact_missing_count = 0
ng8_candidate_rows = 3
ng9_candidate_rows = 3
ng10_candidate_rows = 3
ng8_any_train_shuffle_robust_majority_pass = 0
ng9_any_train_shuffle_robust_majority_pass = 0
ng10_any_train_shuffle_robust_majority_pass = 0
code_review_packet_entries = 111
```

本节写入后再跑一次 finalizer，使本节记录进入 zip；最终 sha256 以后续只读复核为准。


## 11. NG11-NG18 继续推进记录（2026-05-25）

用户再次要求未达成目标继续推进。本节不使用 shell 重定向，避免触发额外权限弹窗；命令均直接通过已批准的 `conda run -n kan` 执行。

### 11.1 finalizer 覆盖面修复

目的：把 NG11-NG18 纳入 route、fallback manifest、required artifact manifest 与 code review packet 审计，不改变任何 promotion gate。

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1224_finalize_functional_bridge.py
```

结果：`py_compile pass`。

### 11.2 NG11 quad_direct + weight_decay 0.003

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng11_tail_safe_quaddirect_wd003_repair --device cuda:2 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --weight-decay 0.003 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download
```

结果：

```text
candidate_rows = 3
selected_actuator = I26-TrainDirectLogitCompensatedQuadRelease
selected_budget = 0.0045
selected_sign = -1.0
selected_comp_mode = median
best_source_vs_noop = 0.02734375
best_source_vs_best_control = 0.00390625
best_linec_seed_pass_count = 3/5
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
```

### 11.3 NG12 quad_direct + weight_decay 0.005

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng12_tail_safe_quaddirect_wd005_repair --device cuda:3 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --weight-decay 0.005 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download
```

结果：与 NG11 基本一致，`best_source_vs_best_control = 0.00390625`，`any_strict_majority_pass = 0`。

### 11.4 NG13/NG14 stronger contrast-tail-safe selector

NG13：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng13_contrast2_tail_safe_quaddirect_probe --device cuda:0 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode contrast_tail_safe --probe-control-contrast-weight 2.00 --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download
```

NG14：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng14_contrast3_tail_safe_quaddirect_probe --device cuda:1 --candidates I30-T1BGuidedDirectCompensation --i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 --probe-score-mode contrast_tail_safe --probe-control-contrast-weight 3.00 --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download
```

结果：

```text
NG13 best_source_vs_noop = 0.0234375
NG13 best_source_vs_best_control = 0.0
NG14 best_source_vs_noop = 0.0234375
NG14 best_source_vs_best_control = 0.0
any_strict_majority_pass = 0
```

### 11.5 NG15-NG18 expanded I32 policy probe

NG15：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng15_i32_expanded_policy_probe_quaddirect --device cuda:2 --candidates I32-PolicyAwareP3QuadProbe --i32-policies direct_gain,direct_branch_gain,quad_direct,freeze_quad --i32-probe-steps 1,3,5,8 --i32-probe-lr 0.001 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download
```

NG16：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng16_i32_expanded_policy_probe_directbranch --device cuda:3 --candidates I32-PolicyAwareP3QuadProbe --i32-policies direct_gain,direct_branch_gain,quad_direct,freeze_quad --i32-probe-steps 1,3,5,8 --i32-probe-lr 0.001 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy direct_branch_gain --train-seed-bases 12240400,12241400,12242400 --no-download
```

NG17：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng17_i32_expanded_policy_probe_directgain --device cuda:0 --candidates I32-PolicyAwareP3QuadProbe --i32-policies direct_gain,direct_branch_gain,quad_direct,freeze_quad --i32-probe-steps 1,3,5,8 --i32-probe-lr 0.001 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy direct_gain --train-seed-bases 12240400,12241400,12242400 --no-download
```

NG18：

```bash
conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 --artifact-prefix v1224_ng18_i32_expanded_policy_probe_directonly --device cuda:1 --candidates I32-PolicyAwareP3QuadProbe --i32-policies direct_gain,direct_branch_gain,quad_direct,freeze_quad --i32-probe-steps 1,3,5,8 --i32-probe-lr 0.001 --probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 --epochs 12 --lr 0.0015 --role-policy direct_only --train-seed-bases 12240400,12241400,12242400 --no-download
```

结果：

```text
NG15 quaddirect: best_source_vs_noop = -0.015625, best_source_vs_best_control = -0.02734375, any_strict_majority_pass = 0
NG16 direct_branch_gain: best_source_vs_noop = 0.015625, best_source_vs_best_control = 0.0078125, linec_pass = 0/5, any_strict_majority_pass = 0
NG17 direct_gain: best_source_vs_noop = 0.01953125, best_source_vs_best_control = 0.01171875, linec_pass = 0/5, any_strict_majority_pass = 0
NG18 direct_only: best_source_vs_noop = 0.01953125, best_source_vs_best_control = 0.01171875, linec_pass = 0/5, any_strict_majority_pass = 0
```

### 11.6 finalizer 命令

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py
```

最终 route 以后续 finalizer 输出为准。


### 11.7 finalizer 结果

执行结果：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 24
fallback_all_executed = 1
required_artifact_rows = 93
required_artifact_missing_count = 0
ng11_candidate_rows = 3
ng12_candidate_rows = 3
ng13_candidate_rows = 3
ng14_candidate_rows = 3
ng15_candidate_rows = 3
ng16_candidate_rows = 3
ng17_candidate_rows = 3
ng18_candidate_rows = 3
ng17_best_source_vs_noop_acc_delta = 0.01953125
ng17_best_source_vs_control_acc_delta = 0.01171875
ng18_best_source_vs_noop_acc_delta = 0.01953125
ng18_best_source_vs_control_acc_delta = 0.01171875
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
code_review_packet_entries = 127
code_review_packet_sha256 = 以 v1224_route_decision.json 当前值为准
transitive_code_packet_missing_count = 0
```

本节写入后再次执行 finalizer 以刷新 zip sha。
