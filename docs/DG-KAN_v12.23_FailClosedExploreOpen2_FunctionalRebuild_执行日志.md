# DG-KAN v12.23 FailClosedExploreOpen2 FunctionalRebuild 执行日志

生成时间：2026-05-25（Asia/Singapore）

本轮目标：按 `docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md` 执行到硬预算完成，产出 v12.23 要求 artifacts、代码审计包、执行日志和实验结果复盘。环境使用 `conda env kan`。本日志只记录实际执行过的命令、修改和观测结果，不补写未执行实验。

## 0. 代码准备与静态检查

修改文件：

- `experiments/run_v1218_b320_label_free_ablation.py`
  - 新增 A43-A50：A43/A44 conservative residual, A45 staged unlabeled adapt, A46 direct-role residual, A47 frozen low-rank residual, A48 branch/gain-only residual, A49 T1B diagnostic, A50 small-label oracle diagnostic。
  - 新增 `adaptframeschedp005` token 解析，实际强度为 0.005。
- `dgkan/models/fc_purekan_primitives.py`
  - 将 `reslowrankpNNN` 改为通用强度解析；`reslowrankp005=0.005`，`reslowrankp010=0.010`。
- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - 新增 v12.23 runner，负责 Line R/A/T/C/I/B/D artifacts、route/final-stop、required manifest 和 zip。

静态检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_label_free_ablation.py \
  experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py \
  dgkan/models/fc_purekan_primitives.py
```

结果：通过。

候选注册检查：

```bash
conda run -n kan python -c "import sys; sys.path[:0]=['.','experiments']; import run_v1218_b320_label_free_ablation as r; specs=r.ablation_specs(784,10); print(','.join([k for k in specs if k.startswith('A4') or k.startswith('A5')][-12:]))"
```

结果：A43-A50 均可被 `ablation_specs` 识别。

## 1. A43 运行时 smoke

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1223_a43_runtime_smoke \
  --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/smoke_a43 \
  --artifact-prefix v1223_a43_runtime_smoke \
  --result-scope v1223_runtime_smoke \
  --smoke-not-official 1 \
  --official-training-result-available 0 \
  --protocol-note "v12.23 A43 runtime smoke before scout" \
  --route-impact "runtime check only" \
  --device cuda:0 --data-root data \
  --datasets MNIST --seeds 0 \
  --ablation-ids A43-ConservativeResidualA1Frame-r005 \
  --train-size 64 --val-size 64 --test-size 64 \
  --batch-size 64 --epochs 1 \
  --task-compile-warmup-steps 1 \
  --measure-linec 1 --linec-batch-size 16 \
  --linec-sketch-batch-size 8 --linec-sketch-dim 4
```

结果：通过，`rows=3`，`summary_rows=3`，A43 无初始化/训练崩溃。

## 2. Line A Level 1 Scout

并行分配：

- GPU0：MNIST
- GPU1：Fashion-MNIST
- GPU2：KMNIST

共同参数：

- train_size=512
- val_size=256
- test_size=256
- epochs=3
- seeds=0,1,2
- measure_linec=1
- linec_batch_size=32
- linec_sketch_dim=8
- candidates：A0/A1/A36/A41/A43/A44/A45/A46/A47/A48/A49/A50

MNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1223_linea_scout_mnist_gpu0 \
  --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/scout/MNIST \
  --artifact-prefix v1223_linea_scout_MNIST \
  --result-stage V1223_LINEA_LEVEL1_SCOUT \
  --summary-stage V1223_LINEA_LEVEL1_SCOUT_SUMMARY \
  --route-stage V1223_LINEA_LEVEL1_SCOUT_ROUTE \
  --result-scope v1223_linea_level1_scout \
  --smoke-not-official 1 \
  --official-training-result-available 0 \
  --protocol-note "v12.23 Line A Level1 scout train512 val/test256 epochs3 seeds0,1,2" \
  --route-impact "feeds v12.23 hardening selection; no final stop" \
  --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 \
  --ablation-ids A0-labelInit,A1-noYForStats,A36-ResidualA1LowRankFrame-labelFree,A41-UnlabeledFrameAdaptSchedule-labelFree,A43-ConservativeResidualA1Frame-r005,A44-ConservativeResidualA1Frame-r010,A45-StagedUnlabeledAdapt-warm1-r005,A46-RoleResidualNoReplace-directOnly,A47-QuadProjLowRankResidualFrozenMain,A48-BranchGainOnlyLabelFreeResidual,A49-A1PlusT1BWeakSignalFrame-diagnostic,A50-SmallLabelOracleMatchedBudget-diagnostic \
  --train-size 512 --val-size 256 --test-size 256 --batch-size 128 --epochs 3 \
  --task-compile-warmup-steps 2 --measure-linec 1 \
  --linec-batch-size 32 --linec-sketch-batch-size 8 --linec-sketch-dim 8
```

Fashion-MNIST 使用同参数，`--device cuda:1 --datasets Fashion-MNIST`，输出目录 `linea/scout/Fashion-MNIST`，prefix `v1223_linea_scout_FashionMNIST`。

KMNIST 使用同参数，`--device cuda:2 --datasets KMNIST`，输出目录 `linea/scout/KMNIST`，prefix `v1223_linea_scout_KMNIST`。

结果：

- MNIST：`rows=42`，best legal/smoke candidate=`A45-StagedUnlabeledAdapt-warm1-r005`，`best_label_free_mean_delta_vs_A0=-0.016927083333333332`。
- Fashion-MNIST：`rows=42`，best candidate=`A49-A1PlusT1BWeakSignalFrame-diagnostic`，`best_label_free_mean_delta_vs_A0=-0.00390625`；注意 A49 是 diagnostic-only，不进入 promotion hardening。
- KMNIST：`rows=42`，best legal/smoke candidate=`A45-StagedUnlabeledAdapt-warm1-r005`，`best_label_free_mean_delta_vs_A0=-0.029947916666666668`。

全局 legal label-free top-2 选择命令：

```bash
conda run -n kan python -c "exec('''import csv,glob,math
paths=glob.glob('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/scout/*/*_summary.csv')
rows=[]
for p in paths:
    for r in csv.DictReader(open(p, newline='', encoding='utf-8')):
        cid=r.get('candidate_id','')
        if cid.startswith('A') and cid not in {'A0-labelInit','A49-A1PlusT1BWeakSignalFrame-diagnostic','A50-SmallLabelOracleMatchedBudget-diagnostic','A42-SmallLabelOracleUpperBound-diagnostic'}:
            rows.append(r)
by={}
for r in rows:
    by.setdefault(r['candidate_id'],[]).append(r)
def f(x, default=float('nan')):
    try: return float(x) if str(x).strip()!='' else default
    except Exception: return default
summary=[]
for cid,rs in by.items():
    vals=[f(r.get('mean_delta_vs_A0_labelInit')) for r in rs if math.isfinite(f(r.get('mean_delta_vs_A0_labelInit')))]
    worst=[f(r.get('worst_delta_vs_A0_labelInit')) for r in rs if math.isfinite(f(r.get('worst_delta_vs_A0_labelInit')))]
    lc=[f(r.get('linec_nontearing_pass_rate')) for r in rs if math.isfinite(f(r.get('linec_nontearing_pass_rate')))]
    summary.append((sum(vals)/len(vals) if vals else -999, min(worst) if worst else -999, sum(lc)/len(lc) if lc else -999, cid))
summary.sort(reverse=True)
for item in summary: print(item[3], item[0], item[1], item[2])
print('TOP2', ','.join([s[3] for s in summary[:2]]))
''')"
```

结果：TOP2=`A45-StagedUnlabeledAdapt-warm1-r005,A41-UnlabeledFrameAdaptSchedule-labelFree`。

## 3. Line A Level 2 Hardening

Hardening 候选：A0/A1/A36/A41/A45。A41 是 top-2 同时也是保留 baseline；A1/A36/A41 按计划保留。

共同参数：

- train_size=1024
- val_size=512
- test_size=512
- epochs=8
- seeds=0,1,2
- LineC batch=64
- LineC sketch_dim=24

MNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1223_linea_hardening_mnist_gpu0 \
  --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/hardening/MNIST \
  --artifact-prefix v1223_linea_hardening_MNIST \
  --result-stage V1223_LINEA_LEVEL2_HARDENING \
  --summary-stage V1223_LINEA_LEVEL2_HARDENING_SUMMARY \
  --route-stage V1223_LINEA_LEVEL2_HARDENING_ROUTE \
  --result-scope v1223_linea_level2_hardening \
  --smoke-not-official 1 \
  --official-training-result-available 0 \
  --protocol-note "v12.23 Line A Level2 top2 hardening train1024 val/test512 epochs8 seeds0,1,2" \
  --route-impact "mandatory hardening; no final stop by itself" \
  --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 \
  --ablation-ids A0-labelInit,A1-noYForStats,A36-ResidualA1LowRankFrame-labelFree,A41-UnlabeledFrameAdaptSchedule-labelFree,A45-StagedUnlabeledAdapt-warm1-r005 \
  --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 \
  --task-compile-warmup-steps 2 --measure-linec 1 \
  --linec-batch-size 64 --linec-sketch-batch-size 24 --linec-sketch-dim 24
```

Fashion-MNIST 使用同参数，`--device cuda:1 --datasets Fashion-MNIST`，输出目录 `linea/hardening/Fashion-MNIST`，prefix `v1223_linea_hardening_FashionMNIST`。

KMNIST 使用同参数，`--device cuda:2 --datasets KMNIST`，输出目录 `linea/hardening/KMNIST`，prefix `v1223_linea_hardening_KMNIST`。

结果：

- MNIST：`rows=21`，best=`A45-StagedUnlabeledAdapt-warm1-r005`，`best_label_free_mean_delta_vs_A0=0.005859375`。
- Fashion-MNIST：`rows=21`，best=`A1-noYForStats`，`best_label_free_mean_delta_vs_A0=-0.005859375`。
- KMNIST：`rows=21`，best=`A36-ResidualA1LowRankFrame-labelFree`，`best_label_free_mean_delta_vs_A0=0.0006510416666666666`。

## 4. v12.23 汇总 runner 与修复记录

最终汇总命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py \
  --run-id official_explore_open2 \
  --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 \
  --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea \
  --data-root data \
  --actuator-device cuda:3 \
  --actuator-datasets MNIST,Fashion-MNIST,KMNIST \
  --actuator-seeds 0,1,2 \
  --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 \
  --actuator-train-size 256 \
  --actuator-val-size 128 \
  --actuator-linec-batch 32 \
  --actuator-sketch-dim 8 \
  --line-d-device cuda:2 \
  --line-d-datasets MNIST \
  --line-d-seeds 0 \
  --line-d-train-size 128 \
  --line-d-val-size 64 \
  --line-d-batch-size 64 \
  --line-d-epochs 1 \
  --line-d-linec-batch 24 \
  --line-d-sketch-dim 8
```

修复 1：

- 首次运行在 `write_label_free_v3` 失败。
- 错误：`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'set'`。
- 修复：将 `int(LINEA_SCOUT_IDS.intersection(...))` 改为 `int(bool(LINEA_SCOUT_IDS.intersection(...)))`。
- 复跑：通过该阶段。

修复 2：

- 复跑后发现 Line D 五个 family 全部 `KernelBlocked`，错误同源：`AttributeError: ... has no attribute 'direct_readout'`。
- 原因：Line D 复用了 Line A/B320 `train_one`，该路径会调用 B320 direct-branch ramp，不适合 PrimitiveKAN/Rational。
- 修复：在 v12.23 runner 中新增 `classic_family_smoke_one`，使用独立轻量 autograd + AdamW smoke，不走 B320 专属 ramp。
- 复跑：Line D 五个 family 均实际产生 task/LineC 指标，`line_d_kernel_blocked_count=0`。

修复 3：

- 复跑后审计发现 runner 会再次读取自身生成的 `v1223_combined_linea_ablation.csv`，导致 Line A 行数重复膨胀。
- 修复：`collect_linea_rows` 跳过 `v1223_combined_linea_ablation.csv`。
- 最终复跑：Line A 输入行数恢复为 `linea_rows=189`，其中 scout=126，hardening=63。

最终 runner 输出摘要：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `minimum_success=Minimum Success F`
- `final_stop_allowed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `required_artifact_missing_count=0`
- `line_d_executed_family_count=5`
- `line_d_kernel_blocked_count=0`

## 5. 关键 artifact 与 zip 验证

验证命令：

```bash
conda run -n kan python -c "exec('''import csv,zipfile,hashlib
from pathlib import Path
out=Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2')
rows=list(csv.DictReader((out/'v1223_required_artifact_manifest.csv').open()))
print('required_missing', [r['artifact'] for r in rows if r.get('exists')!='1'])
z=out/'v1223_code_review_packet.zip'
with zipfile.ZipFile(z) as zz:
    names=zz.namelist()
    print('zip_entries', len(names), 'sha', hashlib.sha256(z.read_bytes()).hexdigest())
    for n in ['experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py','experiments/run_v1218_b320_label_free_ablation.py','dgkan/models/fc_purekan_primitives.py','docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md']:
        print(n, n in names)
''')"
```

结果：

- `required_missing=[]`
- zip 路径：`results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`
- zip sha256：`a4ef59864511fab816a48a746aa517ce85f93f576e7c4de7535fc672b788e31a`
- zip entries：43
- zip 已包含：
  - `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - `experiments/run_v1218_b320_label_free_ablation.py`
  - `dgkan/models/fc_purekan_primitives.py`
  - `docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md`

## 6. 追加只读复核记录（2026-05-25）

用户再次追问 v12.23 是否完成后，执行只读复核；未追加新实验，未修改结果 CSV/JSON。

首次复核命令：

```bash
conda run -n kan python -c "import json, pathlib, zipfile, hashlib; repo=pathlib.Path('/home/chengshun.wang/DG-LCA'); out=repo/'results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'; route=json.loads((out/'v1223_route_decision.json').read_text()); manifest=json.loads((out/'v1223_required_artifact_manifest.json').read_text()); zip_path=out/'v1223_code_review_packet.zip'; names=zipfile.ZipFile(zip_path).namelist(); print('route', route.get('route')); print('final_stop_allowed', route.get('final_stop_allowed')); print('official_success_reached', route.get('official_success_reached')); print('hard_budget_exhausted', route.get('hard_budget_exhausted')); print('mandatory_exploration_levels_executed', route.get('mandatory_exploration_levels_executed')); print('required_artifact_missing_count', route.get('required_artifact_missing_count')); print('manifest_missing_count', len(manifest.get('missing', []))); print('linea_rows', route.get('linea_rows')); print('line_d_executed_family_count', route.get('line_d_executed_family_count')); print('line_d_kernel_blocked_count', route.get('line_d_kernel_blocked_count')); print('actuator_rows', route.get('actuator_rows')); print('zip_exists', zip_path.exists()); print('zip_entries', len(names)); print('zip_sha256', hashlib.sha256(zip_path.read_bytes()).hexdigest()); print('exec_log_exists', (repo/'docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_执行日志.md').exists()); print('review_log_exists', (repo/'docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果复盘.md').exists())"
```

结果：失败。原因是我把实际存在的 `v1223_required_artifact_manifest.csv` 误写成了 `v1223_required_artifact_manifest.json`。错误为：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/home/chengshun.wang/DG-LCA/results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_required_artifact_manifest.json'
```

随后定位真实 artifact/manifest 文件：

```bash
find results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 -maxdepth 2 -type f \( -name '*manifest*' -o -name '*artifact*' -o -name '*route*' -o -name '*audit*' \) | sort
```

确认存在：

- `v1223_required_artifact_manifest.csv`
- `v1223_route_decision.json`
- `v1223_final_stop_audit.json`
- `v1223_hash_manifest.json`
- `v1223_child_run_manifest.csv`
- `v1223_fallback_execution_manifest.csv`
- `v1223_diff_isolation_manifest.csv`
- `v1223_dirty_tree_audit.csv`

修正后复核命令：

```bash
conda run -n kan python -c "import csv,json,pathlib,zipfile,hashlib; repo=pathlib.Path('/home/chengshun.wang/DG-LCA'); out=repo/'results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'; route=json.loads((out/'v1223_route_decision.json').read_text()); rows=list(csv.DictReader((out/'v1223_required_artifact_manifest.csv').open())); missing=[r for r in rows if str(r.get('exists','')).lower() not in {'1','true','yes'}]; zip_path=out/'v1223_code_review_packet.zip'; names=zipfile.ZipFile(zip_path).namelist(); print('route', route.get('route')); print('minimum_success', route.get('minimum_success')); print('final_stop_allowed', route.get('final_stop_allowed')); print('official_success_reached', route.get('official_success_reached')); print('hard_budget_exhausted', route.get('hard_budget_exhausted')); print('mandatory_exploration_levels_executed', route.get('mandatory_exploration_levels_executed')); print('required_artifact_missing_count', route.get('required_artifact_missing_count')); print('manifest_rows', len(rows)); print('manifest_missing_count', len(missing)); print('linea_rows', route.get('linea_rows')); print('line_d_executed_family_count', route.get('line_d_executed_family_count')); print('line_d_kernel_blocked_count', route.get('line_d_kernel_blocked_count')); print('actuator_rows', route.get('actuator_rows')); print('zip_entries', len(names)); print('has_runner', 'experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py' in names); print('has_ablation', 'experiments/run_v1218_b320_label_free_ablation.py' in names); print('has_primitives', 'dgkan/models/fc_purekan_primitives.py' in names); print('zip_sha256', hashlib.sha256(zip_path.read_bytes()).hexdigest()); print('exec_log_exists', (repo/'docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_执行日志.md').exists()); print('review_log_exists', (repo/'docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果复盘.md').exists())"
```

修正后输出：

```text
route R4-FunctionalMechanismNoGoAfterAllFallbacks
minimum_success Minimum Success F
final_stop_allowed 1
official_success_reached 0
hard_budget_exhausted 1
mandatory_exploration_levels_executed 1
required_artifact_missing_count 0
manifest_rows 46
manifest_missing_count 0
linea_rows 189
line_d_executed_family_count 5
line_d_kernel_blocked_count 0
actuator_rows 1512
zip_entries 43
has_runner True
has_ablation True
has_primitives True
zip_sha256 a4ef59864511fab816a48a746aa517ce85f93f576e7c4de7535fc672b788e31a
exec_log_exists True
review_log_exists True
```

## 7. 纠正：继续执行计划中未充分闭合的 fallback（2026-05-25）

用户指出计划要求继续后，重新核对原计划第 9、11、12、19 节，确认此前把 runner 自身写出的 `mandatory_exploration_levels_executed=1` 当成充分证明是不严谨的。计划的 fallback 表还要求 hardening task/LineC fail 后继续 lower residual energy、staged warmup、role-energy/branch-only 修复；T1A/T1B 未打开时，需要 T2 clone-probe upper-bound diagnostic 后才能写更强 no-go。

### 7.1 代码修改

修改文件：

- `experiments/run_v1218_b320_label_free_ablation.py`
- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`

修改内容：

- 新增 Line A Level 3 repair candidates：`A51-StagedUnlabeledAdapt-warm1-r002`、`A52-StagedUnlabeledAdapt-warm1-r001`、`A53-ResidualA1LowRankFrame-r002-fixedP`、`A54-A1FixedPBranchLowQuad020`。
- 在 v12.23 runner 注册 A51-A54 的 frame family 映射。
- 新增 `v1223_t2_clone_probe_visibility_diagnostic.csv`，显式 `promotion_allowed=0`、`clone_probe_only=1`、`precommit_available=0`。
- 将 T2 artifact 加入 required artifacts、fallback manifest 和 no-go proof layer。

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

结果：通过，无输出错误。

### 7.2 A51-A54 runtime smoke

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_level3_repair_smoke_MNIST_seed0 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/level3_repair_smoke/MNIST --artifact-prefix v1223_level3_repair_smoke_MNIST --result-stage V1223_LINEA_LEVEL3_REPAIR_SMOKE --summary-stage V1223_LINEA_LEVEL3_REPAIR_SMOKE_SUMMARY --route-stage V1223_LINEA_LEVEL3_REPAIR_SMOKE_ROUTE --result-scope v1223_level3_repair_smoke --device cuda:0 --data-root data --datasets MNIST --seeds 0 --ablation-ids A0-labelInit,A51-StagedUnlabeledAdapt-warm1-r002,A52-StagedUnlabeledAdapt-warm1-r001,A53-ResidualA1LowRankFrame-r002-fixedP,A54-A1FixedPBranchLowQuad020 --seed-base 12233000 --train-size 256 --val-size 128 --test-size 128 --batch-size 128 --epochs 1 --measure-linec 1 --linec-batch-size 32 --linec-sketch-batch-size 8 --linec-sketch-dim 8 --smoke-not-official 1 --official-training-result-available 0 --protocol-note 'v12.23 continuation smoke for Level3 failure-specific repair candidates A51-A54' --route-impact 'registration/runtime smoke only; no promotion claim'
```

结果：`rows=7`，`summary_rows=7`，`label_free_ablation_available_count=4`，best smoke candidate `A51-StagedUnlabeledAdapt-warm1-r002`，best smoke mean delta vs A0 `-0.2109375`。A51-A54 无 runtime/kernel blocker；smoke 指标很差，不能作为 promotion。

### 7.3 Line A Level 3 repair hardening

MNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_level3_repair_MNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/level3_repair/MNIST --artifact-prefix v1223_level3_repair_MNIST --result-stage V1223_LINEA_LEVEL3_REPAIR --summary-stage V1223_LINEA_LEVEL3_REPAIR_SUMMARY --route-stage V1223_LINEA_LEVEL3_REPAIR_ROUTE --result-scope v1223_level3_repair_hardening --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A45-StagedUnlabeledAdapt-warm1-r005,A51-StagedUnlabeledAdapt-warm1-r002,A52-StagedUnlabeledAdapt-warm1-r001,A53-ResidualA1LowRankFrame-r002-fixedP,A54-A1FixedPBranchLowQuad020 --seed-base 12233100 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 8 --linec-sketch-dim 24 --smoke-not-official 1 --official-training-result-available 0 --protocol-note 'v12.23 continuation Level3 failure-specific repair: lower residual/staged warmup/fixedP branch lowquad' --route-impact 'v12.23 Line A Level3 repair; no promotion claim unless gates pass'
```

Fashion-MNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_level3_repair_FashionMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/level3_repair/Fashion-MNIST --artifact-prefix v1223_level3_repair_FashionMNIST --result-stage V1223_LINEA_LEVEL3_REPAIR --summary-stage V1223_LINEA_LEVEL3_REPAIR_SUMMARY --route-stage V1223_LINEA_LEVEL3_REPAIR_ROUTE --result-scope v1223_level3_repair_hardening --device cuda:1 --data-root data --datasets Fashion-MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A45-StagedUnlabeledAdapt-warm1-r005,A51-StagedUnlabeledAdapt-warm1-r002,A52-StagedUnlabeledAdapt-warm1-r001,A53-ResidualA1LowRankFrame-r002-fixedP,A54-A1FixedPBranchLowQuad020 --seed-base 12233100 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 8 --linec-sketch-dim 24 --smoke-not-official 1 --official-training-result-available 0 --protocol-note 'v12.23 continuation Level3 failure-specific repair: lower residual/staged warmup/fixedP branch lowquad' --route-impact 'v12.23 Line A Level3 repair; no promotion claim unless gates pass'
```

KMNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_level3_repair_KMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/level3_repair/KMNIST --artifact-prefix v1223_level3_repair_KMNIST --result-stage V1223_LINEA_LEVEL3_REPAIR --summary-stage V1223_LINEA_LEVEL3_REPAIR_SUMMARY --route-stage V1223_LINEA_LEVEL3_REPAIR_ROUTE --result-scope v1223_level3_repair_hardening --device cuda:2 --data-root data --datasets KMNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A45-StagedUnlabeledAdapt-warm1-r005,A51-StagedUnlabeledAdapt-warm1-r002,A52-StagedUnlabeledAdapt-warm1-r001,A53-ResidualA1LowRankFrame-r002-fixedP,A54-A1FixedPBranchLowQuad020 --seed-base 12233100 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 8 --linec-sketch-dim 24 --smoke-not-official 1 --official-training-result-available 0 --protocol-note 'v12.23 continuation Level3 failure-specific repair: lower residual/staged warmup/fixedP branch lowquad' --route-impact 'v12.23 Line A Level3 repair; no promotion claim unless gates pass'
```

结果：

- MNIST：`rows=27`，best label-free candidate `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.010416666666666666`。
- Fashion-MNIST：`rows=27`，best label-free candidate `A1-noYForStats`，best mean delta vs A0 `-0.009114583333333334`。
- KMNIST：`rows=27`，best label-free candidate `A1-noYForStats`，best mean delta vs A0 `-0.0006510416666666666`。

关键 aggregate：

- A1：mean delta avg `-0.0013020833333333337`，worst min `-0.017578125`，AUC max `1.1653115504975817`，LineC avg `0.0`
- A45：mean delta avg `-0.003038194444444444`，worst min `-0.029296875`，AUC max `1.1065331974809605`，LineC avg `0.0`
- A51：mean delta avg `-0.0023871527777777775`，worst min `-0.0234375`，AUC max `1.1064237918237636`，LineC avg `0.0`
- A52：mean delta avg `-0.002821180555555556`，worst min `-0.02734375`，AUC max `1.1063864086995177`，LineC avg `0.0`
- A53：mean delta avg `-0.09483506944444446`，worst min `-0.166015625`，AUC max `2.0610389504270414`，LineC avg `0.0`
- A54：mean delta avg `-0.09657118055555554`，worst min `-0.13671875`，AUC max `2.2795644556702754`，LineC avg `0.0`

结论：A51/A52 降低 residual 后 task mean 接近 gate，但 worst 和 LineC 仍失败；A53/A54 fixedP/branch-lowquad 明显伤 task/AUC。没有 exploration pass。

### 7.4 重新运行 v12.23 official runner 纳入 continuation artifacts

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_continued --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

最终 runner 输出摘要：

- `run_id=official_explore_open2_continued`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `minimum_success=Minimum Success F`
- `final_stop_allowed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `linea_rows=277`
- `linea_scout_rows=133`
- `linea_hardening_rows=144`
- `label_free_best_hardening_candidate=A1-noYForStats`
- `label_free_best_hardening_mean_delta_vs_A0=-0.002495659722222222`
- `label_free_exploration_pass_count=0`
- `T1A_auc_joint=0.34141121237227284`
- `T1B_auc_joint=0.5107014636840652`
- `T1B_native_logged_rows=193`
- `T2_clone_probe_auc_joint=0.7762043088840475`
- `T2_clone_probe_support_count=277`
- `T2_clone_probe_positive_count=34`
- `T2_clone_probe_precision_at_k=0.3235294117647059`
- `T2_clone_probe_recall_at_k=0.3235294117647059`
- `T2_clone_probe_promotion_allowed=0`
- `actuator_rows=1512`
- `actuator_role_safe_movement_rows=387`
- `actuator_release_audit_rows=112`
- `actuator_exploratory_release_dataset_seed_count=0`
- `functional_p3_pass_count=0`
- `line_d_executed_family_count=5`
- `line_d_kernel_blocked_count=0`
- `required_artifact_missing_count=0`
- `required_artifact_rows=47`
- `code_review_packet_sha256=b585c681b5af29a3cda13b05352786c49b21e6502ffb145e42269c83b469dff8`

### 7.5 最终只读核验

输出：

```text
route R4-FunctionalMechanismNoGoAfterAllFallbacks
run_id official_explore_open2_continued
final_stop_allowed 1
linea_rows 277
linea_hardening_rows 144
T2_auc 0.7762043088840475
T2_support 277
T2_promotion_allowed 0
missing_count 0
required_rows 47
zip_entries 44
has_t2_artifact True
has_runner True
has_ablation True
zip_sha256 b585c681b5af29a3cda13b05352786c49b21e6502ffb145e42269c83b469dff8
```

## 8. 继续纠正：Line I controls-win fallback 与 I18 bisection（2026-05-25）

用户再次要求按计划继续后，重新审计计划第 13.7 与第 19 节，发现 Line I 仍有未充分闭合的 fallback：

```text
If release and movement exist but controls also win:
  subtract AdamW-parallel component; add matched role-energy controls.

If logit drift exceeds gate:
  norm bisection and split event into smaller low-frequency maintenance events.
```

旧结果中 non-control actuator rows 为 756，safe+release rows 为 66，但 safe+release+control_gap>=0.002 为 0。因此继续执行。

### 8.1 代码修改

修改文件：

- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`

修改内容：

- 新增 `I18-AdamWOrthogonalResidual-diagnostic`。
- I18 对 role 参数构造 residual，并从 residual 中减去 AdamW-parallel component。
- I18 显式标记 `diagnostic_only=1`、`uses_label=1`、`uses_ce_vector=1`，不允许打开 promotion/exploration gate。
- 将 `actuator_release_movement_but_controls_win -> AdamW_orthogonal_residual_plus_matched_role_controls` 写入 fallback manifest。
- 增加 I18 低预算 bisection：`1e-05,2.5e-05,5e-05,0.0001,0.0002`。
- 为同一低预算补跑 matched controls，避免 I18 bisection 缺少同预算 controls。

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

结果：通过，无输出错误。

### 8.2 I18 首轮 official runner

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_continued_i18 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：

- `actuator_rows=1620`
- `actuator_release_audit_rows=124`
- `actuator_exploratory_release_dataset_seed_count=0`
- `I18 rows=108`
- `I18 safe=0`
- `I18 release=12`
- `I18 exploratory=0`
- `I18 gap max=-0.013203956186771393`
- `I18 drift min=0.24471133947372437`

结论：I18 有 release 行，但 drift 全部超过 0.05，且 control gap 不足。

### 8.3 I18 norm bisection 首轮

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_continued_i18_bisect --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：

- `actuator_rows=1710`
- `actuator_release_audit_rows=150`
- `I18 all rows=198`
- `I18 bisection rows=90`
- `I18 bisection safe=0`
- `I18 bisection release=26`
- `I18 bisection exploratory=0`
- `I18 bisection drift min=0.24620485305786133`

问题：低预算 I18 rows 缺少同预算 controls，导致部分 `control_gap` 为空。该问题继续修复。

### 8.4 I18 norm bisection + matched controls

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_continued_i18_bisect_controls --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：

- `run_id=official_explore_open2_continued_i18_bisect_controls`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `final_stop_allowed=1`
- `actuator_rows=2340`
- `actuator_release_audit_rows=150`
- `actuator_exploratory_release_dataset_seed_count=0`
- `I18 all rows=198`
- `I18 safe=0`
- `I18 release=38`
- `I18 exploratory=0`
- `I18 gap max=0.00018125027418136597`
- `I18 drift min=0.24471133947372437`
- `I18 drift <= 0.05 rows=0`
- `I18 bisection rows=90`
- `I18 bisection safe=0`
- `I18 bisection release=26`
- `I18 bisection exploratory=0`
- `I18 bisection gap max=0.00018125027418136597`
- `I18 bisection drift min=0.24620485305786133`
- low-budget matched control rows：630
- `fallback_i18=True`
- `fallback_i18_bisect=True`
- `required_artifact_missing_count=0`
- `zip_sha256=9ca9c4ac8a282bda300649e0686a3cf8816989d450967a34d1c12eb9e94ff420`

最终解释：I18 在低预算下仍无法进入 drift gate，且最大 control gap 只有 `0.00018125027418136597`，低于 exploratory gate `0.002`。因此 Line I controls-win fallback 与 norm-bisection fallback 均执行，但没有打开 actuator exploration。

## 9. 非 I18 release/no-safe drift 补齐：I11/I13/I15/I17 全部低范数二分

### 9.1 继续原因

上一轮只对 I18 做了 norm bisection。补充审计发现非 I18 actuator 中也存在 `release_audit_pass=1` 但 `role_safe_movement_pass=0` 的 drift-failure 行：

- `I11-BranchGainRedistributionRoleSafe`
- `I13-QuadResidualLowDriftRotation`
- `I15-NoiseReleaseReservoirReleaseQP`
- `I17-ShadowPositiveReplayMatchedControls`

这对应计划中的 “Actuator release but no safety -> role-specific movement gate + norm bisection”，不能只对 I18 执行后停止。

### 9.2 代码修改

修改文件：

- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`

修改内容：

- 新增 fallback manifest 行：
  - `non_i18_release_no_safe_due_drift -> I11_I13_I15_I17_norm_bisection_with_matched_controls`
- 将低范数 bisection actuator 集合扩展为：
  - `I11-BranchGainRedistributionRoleSafe`
  - `I13-QuadResidualLowDriftRotation`
  - `I15-NoiseReleaseReservoirReleaseQP`
  - `I17-ShadowPositiveReplayMatchedControls`
  - `I18-AdamWOrthogonalResidual-diagnostic`
- 低预算集合保持：
  - `1e-05,2.5e-05,5e-05,0.0001,0.0002`
- 同步补齐 matched controls 的低预算 rows，避免 `control_gap` 因缺少同预算 control 而为空。

修改理由：

- 该修改直接对应计划的 blocker 修复方向。
- I18 之外的 release/no-safe drift 不能被 I18 diagnostic 代表，必须对触发 actuator 自身做低范数二分。
- matched controls 必须和 actuator 预算对齐，否则 control gap 审计不完整。

### 9.3 编译检查

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py dgkan/models/fc_purekan_primitives.py
```

结果：

- exit code：0
- 无 Python 语法错误输出。

### 9.4 official runner：all drift bisection

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_continued_all_drift_bisect --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果 JSON 摘要：

- `run_id=official_explore_open2_continued_all_drift_bisect`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `final_stop_allowed=1`
- `mandatory_exploration_levels_executed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `actuator_rows=2700`
- `actuator_release_audit_rows=256`
- `actuator_exploratory_release_dataset_seed_count=0`
- `line_d_executed_family_count=5`
- `required_artifact_missing_count=0`
- `code_review_packet_sha256=e8b158c3bc8395e03c6f183aabb17cccd870d4abb3ec211c89a1680248b90ecc`

输出文件：

- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_route_decision.json`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_actuator_safety_roleaware.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_actuator_controls.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_fallback_execution_manifest.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`

### 9.5 落盘审计命令

CSV/JSON 审计命令使用 `conda env kan` 执行，读取：

- `v1223_route_decision.json`
- `v1223_actuator_safety_roleaware.csv`
- `v1223_actuator_controls.csv`
- `v1223_fallback_execution_manifest.csv`
- `v1223_required_artifact_manifest.csv`

关键审计结果：

- fallback manifest rows：10
- fallback manifest all executed：true
- required artifact rows：47
- required artifact missing count：0
- low-budget matched control rows：630
- code review zip entries：44
- code review zip 包含：
  - `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - `experiments/run_v1222_failclosed_explore_open_functional_rebuild.py`
  - `experiments/run_v1221_failclosed_continue2_label_free_functional.py`
  - `experiments/run_v1218_b320_label_free_ablation.py`
  - `experiments/run_v1283_b109_classic_family_functional_geometry.py`
  - `experiments/run_v1252_efficiency_functional_manifold.py`
  - `experiments/run_v124_multibasis_functional_dual.py`
  - `dgkan/models/fc_purekan_primitives.py`
  - `dgkan/kernels/fused_hinge_quadratic.py`
  - `docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md`

### 9.6 单 actuator 低范数二分审计

`I11-BranchGainRedistributionRoleSafe`：

- rows：198
- bisection rows：90
- safe：170
- release：46
- release_no_safe：4
- exploratory：0
- official：0
- gap max：0.0016209781169891357
- drift min：3.8623809814453125e-05
- drift <= 0.05 rows：170
- bisection release：28
- bisection safe：90
- bisection exploratory：0
- bisection gap max：0.0001363307237625122
- bisection drift min：3.8623809814453125e-05

`I13-QuadResidualLowDriftRotation`：

- rows：198
- bisection rows：90
- safe：0
- release：39
- release_no_safe：39
- exploratory：0
- official：0
- gap max：0.04676394909620285
- drift min：0.23476892709732056
- drift <= 0.05 rows：0
- bisection release：26
- bisection safe：0
- bisection exploratory：0
- bisection gap max：-0.0003450736403465271
- bisection drift min：0.24181252717971802

`I15-NoiseReleaseReservoirReleaseQP`：

- rows：198
- bisection rows：90
- safe：0
- release：41
- release_no_safe：41
- exploratory：0
- official：0
- gap max：0.0004449784755706787
- drift min：0.23766690492630005
- drift <= 0.05 rows：0
- bisection release：26
- bisection safe：0
- bisection exploratory：0
- bisection gap max：0.0004449784755706787
- bisection drift min：0.24253308773040771

`I17-ShadowPositiveReplayMatchedControls`：

- rows：198
- bisection rows：90
- safe：0
- release：39
- release_no_safe：39
- exploratory：0
- official：0
- gap max：0.019115328788757324
- drift min：0.2368667721748352
- drift <= 0.05 rows：0
- bisection release：26
- bisection safe：0
- bisection exploratory：0
- bisection gap max：0.00019361823797225952
- bisection drift min：0.2449427843093872

`I18-AdamWOrthogonalResidual-diagnostic`：

- rows：198
- bisection rows：90
- safe：0
- release：38
- release_no_safe：38
- exploratory：0
- official：0
- gap max：0.00018125027418136597
- drift min：0.24471133947372437
- drift <= 0.05 rows：0
- bisection release：26
- bisection safe：0
- bisection exploratory：0
- bisection gap max：0.00018125027418136597
- bisection drift min：0.24620485305786133

## 10. 全线 continuation：Line A/T/I/B/D/R 每条线追加推进

用户指出“每个线都要推进”。本节不是重复确认 route，而是对每条线追加实际动作并重新跑最终 route。

### 10.1 代码修改

修改文件：

- `experiments/run_v1218_b320_label_free_ablation.py`
- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`

修改内容：

- Line A：
  - 修正 `A51-A54` 被误放入 diagnostic 集合的问题；这些是 label-free continuation repair，不应作为 diagnostic-only 排除。
  - 新增 `A55-ResidualA1LowRankFrame-r001-fixedP`。
  - 新增 `A56-A1FixedPBranchLowQuad010`。
- Line T/C：
  - 新增 `v1223_t3_visibility_support_stress.csv`。
  - 对 T2 response/LineC upper-bound diagnostic 做 dataset/seed support stress；仍标记 audit-only，`promotion_allowed=0`。
- Line I：
  - 新增 branch/direct/gain role matched controls：
    - `BranchOnlyRandomControl`
    - `DirectOnlyRandomControl`
    - `GainOnlyRandomControl`
  - 这些 controls 用同预算随机 role perturbation，补齐 role-specific control ambiguity 审计。
- Line B：
  - 新增 `v1223_functional_p3_control_audit.csv`。
  - P3 gate 显式要求 source actuator 同时满足 role-safe、release、exploratory、非 diagnostic、非 label/CE provenance。
- Line D：
  - `write_line_d_classic_smoke` 从只取第一个 dataset/seed 改为遍历 `--line-d-datasets` 与 `--line-d-seeds`。
- Line R：
  - required artifact 增加 T3 与 P3 control audit。
  - fallback manifest 增加 T3、role-control、P3 control audit 三项。
  - zip 自动包含新增 `v1223_*` artifacts。

### 10.2 编译检查

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py dgkan/models/fc_purekan_primitives.py
```

结果：

- exit code：0
- 无 Python 语法错误输出。

### 10.3 Line A：A55/A56 hardening 三数据集并行

MNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_allline_A55A56_hardening_MNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/allline_A55A56_hardening/MNIST --artifact-prefix v1223_allline_A55A56_hardening_MNIST --result-stage V1223_ALLLINE_LINEA_A55A56_HARDENING --summary-stage V1223_ALLLINE_LINEA_A55A56_HARDENING_SUMMARY --route-stage V1223_ALLLINE_LINEA_A55A56_HARDENING_ROUTE --result-scope v1223_allline_A55A56_hardening --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A55-ResidualA1LowRankFrame-r001-fixedP,A56-A1FixedPBranchLowQuad010 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-all-line-continuation-A55A56-hardening --route-impact feeds-v1223-all-line-continuation
```

Fashion-MNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_allline_A55A56_hardening_FashionMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/allline_A55A56_hardening/Fashion-MNIST --artifact-prefix v1223_allline_A55A56_hardening_FashionMNIST --result-stage V1223_ALLLINE_LINEA_A55A56_HARDENING --summary-stage V1223_ALLLINE_LINEA_A55A56_HARDENING_SUMMARY --route-stage V1223_ALLLINE_LINEA_A55A56_HARDENING_ROUTE --result-scope v1223_allline_A55A56_hardening --device cuda:1 --data-root data --datasets Fashion-MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A55-ResidualA1LowRankFrame-r001-fixedP,A56-A1FixedPBranchLowQuad010 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-all-line-continuation-A55A56-hardening --route-impact feeds-v1223-all-line-continuation
```

KMNIST 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_allline_A55A56_hardening_KMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/allline_A55A56_hardening/KMNIST --artifact-prefix v1223_allline_A55A56_hardening_KMNIST --result-stage V1223_ALLLINE_LINEA_A55A56_HARDENING --summary-stage V1223_ALLLINE_LINEA_A55A56_HARDENING_SUMMARY --route-stage V1223_ALLLINE_LINEA_A55A56_HARDENING_ROUTE --result-scope v1223_allline_A55A56_hardening --device cuda:2 --data-root data --datasets KMNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A55-ResidualA1LowRankFrame-r001-fixedP,A56-A1FixedPBranchLowQuad010 --train-size 1024 --val-size 512 --test-size 512 --batch-size 128 --epochs 8 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-all-line-continuation-A55A56-hardening --route-impact feeds-v1223-all-line-continuation
```

Line A 子任务结果：

- MNIST rows：18，summary rows：6，best：`A1-noYForStats`，best mean delta vs A0：`0.0006510416666666666`
- Fashion-MNIST rows：18，summary rows：6，best：`A1-noYForStats`，best mean delta vs A0：`-0.005859375`
- KMNIST rows：18，summary rows：6，best：`A1-noYForStats`，best mean delta vs A0：`-0.009114583333333334`

### 10.4 全线 official runner

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_all_lines_continuation --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST,Fashion-MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果 JSON 摘要：

- `run_id=official_explore_open2_all_lines_continuation`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `final_stop_allowed=1`
- `mandatory_exploration_levels_executed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `linea_rows=331`
- `linea_hardening_rows=198`
- `label_free_candidate_rows=18`
- `label_free_official_pass_count=0`
- `T3_visibility_support_stress_rows=6`
- `actuator_rows=3294`
- `actuator_exploratory_release_dataset_seed_count=0`
- `functional_p3_control_audit_rows=30`
- `functional_p3_pass_count=0`
- `line_d_family_rows=10`
- `line_d_dataset_seed_rows=10`
- `line_d_kernel_blocked_count=0`
- `required_artifact_rows=49`
- `required_artifact_missing_count=0`
- `code_review_packet_sha256=c8d45a8dbf9ace12e32418d7905f9db75680deba1561dacab23156c61b5bd1c7`

### 10.5 逐线审计命令与结果

route/fallback/artifact 审计命令：

```bash
conda run -n kan python -c "import json; d=json.load(open('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_route_decision.json')); keys=['run_id','route','final_stop_allowed','mandatory_exploration_levels_executed','official_success_reached','hard_budget_exhausted','linea_rows','linea_hardening_rows','label_free_candidate_rows','label_free_official_pass_count','T3_visibility_support_stress_rows','actuator_rows','functional_p3_control_audit_rows','line_d_family_rows','line_d_dataset_seed_rows','required_artifact_missing_count','code_review_packet_sha256']; print({k:d.get(k) for k in keys})"
conda run -n kan python -c "import csv; root='results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/'; f=list(csv.DictReader(open(root+'v1223_fallback_execution_manifest.csv'))); r=list(csv.DictReader(open(root+'v1223_required_artifact_manifest.csv'))); print('fallback_rows',len(f),'all_executed',all(x.get('executed')=='1' for x in f),'triggers',[x.get('trigger') for x in f]); print('required_rows',len(r),'missing',[x.get('artifact') for x in r if x.get('exists')!='1'])"
```

结果：

- fallback rows：13
- fallback all executed：true
- required rows：49
- missing：`[]`

Line A 审计结果：

- `A1-noYForStats`：mean delta vs A0 `-0.0029658564814814816`，worst `-0.029296875`，LineC pass rate `0.0`，diagnostic `0`
- `A51-StagedUnlabeledAdapt-warm1-r002`：mean delta vs A0 `-0.002387152777777778`，worst `-0.0234375`，LineC pass rate `0.0`，diagnostic `0`
- `A55-ResidualA1LowRankFrame-r001-fixedP`：mean delta vs A0 `-0.09288194444444445`，worst `-0.15625`，LineC pass rate `0.0`，diagnostic `0`
- `A56-A1FixedPBranchLowQuad010`：mean delta vs A0 `-0.10221354166666667`，worst `-0.150390625`，LineC pass rate `0.0`，diagnostic `0`

Line T/C 审计结果：

- T3 rows：6
- T3 stable rows：6
- T3 AUC min：`0.7436998854524628`
- T3 AUC max：`0.8489583333333334`
- T3 仍为 diagnostic-only，`promotion_allowed=0`

Line I/B 审计结果：

- `BranchOnlyRandomControl` rows：198
- `DirectOnlyRandomControl` rows：198
- `GainOnlyRandomControl` rows：198
- non-control exploratory actuator rows：0
- P3 rows：6
- P3 pass：0
- P3 control audit rows：30
- P3 audit top row：`I11-BranchGainRedistributionRoleSafe`, dataset `MNIST`, seed `1`, budget `0.005`, control gap `0.0016209781169891357`, safe `1`, release `1`, exploratory `0`

Line D 审计结果：

- rows：10
- kernel blocked：0
- datasets：MNIST, Fashion-MNIST
- families：Rational, Chebyshev, Wavelet, RBF/FastKAN, Fourier
- Rational MNIST：val_acc `0.640625`, LineC CouplingR2 `0.1688038217034734`
- Rational Fashion-MNIST：val_acc `0.5625`, LineC CouplingR2 `0.20384892975350133`
- Fourier MNIST：val_acc `0.296875`, LineC CouplingR2 `0.09257388126333477`
- Fourier Fashion-MNIST：val_acc `0.40625`, LineC CouplingR2 `0.088131683128016`

Zip 审计：

- `zip_entries=46`
- `has_t3=True`
- `has_p3_audit=True`
- `has_runner=True`
- `has_linea=True`
- zip path：`results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`

## 11. 2026-05-25 继续执行：target-open 修复、测量口径修复与最终复核

本节补充执行 v12.23 计划时在 section 10 之后继续推进的命令和产物。环境均使用：

```bash
conda run -n kan ...
```

工作目录：

```text
/home/chengshun.wang/DG-LCA
```

主输出目录：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2
```

### 11.1 代码修改与语法检查

修改文件：

- `experiments/run_v1218_b320_label_free_ablation.py`
  - 新增 `A57-A51BoundQLineCRepair`
  - 新增 `A58-A51SignalBlock010LineCRepair`
  - 新增 `A59-A51SignalBroad025Block010BoundQ`
  - 新增 `A60-A51DirectRead125ReservoirRepair`
  - 新增 `A61-A51IdentityAmp150ReservoirRepair`
  - 新增 `A62-A58DirectRead125ReservoirRepair`
  - 新增 `A63-A51RMSQReservoirRepair`
  - 新增 `A64-A51RMSQBoundQReservoirRepair`
  - 新增 `A65-A58RMSQSignalBlockRepair`
- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - 将 A57-A65 纳入 v12.23 candidate mapping。
  - 修复 Line A hardening 汇总的 scope isolation，避免不同 result_scope 的 A0 baseline 混用。
  - 将 hardening gate 计入 `v1223_label_free_*_pass_count`。
  - 新增 T3 support stress、role-specific matched controls、P3 control audit、Line D 多数据集 smoke。
  - 修复 actuator LineC delta：base 与 trial 使用同一个 `metric_seed`，并按 seed 缓存 base metrics。修复后 NoOp control 的 Noise/Reservoir delta 最大绝对和为 `0.0`。

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

结果：退出码 0，无 stdout/stderr 错误。

### 11.2 A57-A59 LineC repair 三数据集并行

MNIST，GPU 0：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A57A59_e12_MNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A57A59_e12/MNIST --artifact-prefix v1223_linec_repair_A57A59_e12_MNIST --result-stage V1223_LINEC_REPAIR_A57A59_E12 --summary-stage V1223_LINEC_REPAIR_A57A59_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A57A59_E12_ROUTE --result-scope v1223_linec_repair_A57A59_e12 --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A57-A51BoundQLineCRepair,A58-A51SignalBlock010LineCRepair,A59-A51SignalBroad025Block010BoundQ --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A57A59-e12 --route-impact feeds-v1223-linec-repair-continuation
```

Fashion-MNIST，GPU 1：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A57A59_e12_FashionMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A57A59_e12/Fashion-MNIST --artifact-prefix v1223_linec_repair_A57A59_e12_FashionMNIST --result-stage V1223_LINEC_REPAIR_A57A59_E12 --summary-stage V1223_LINEC_REPAIR_A57A59_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A57A59_E12_ROUTE --result-scope v1223_linec_repair_A57A59_e12 --device cuda:1 --data-root data --datasets Fashion-MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A57-A51BoundQLineCRepair,A58-A51SignalBlock010LineCRepair,A59-A51SignalBroad025Block010BoundQ --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A57A59-e12 --route-impact feeds-v1223-linec-repair-continuation
```

KMNIST，GPU 2：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A57A59_e12_KMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A57A59_e12/KMNIST --artifact-prefix v1223_linec_repair_A57A59_e12_KMNIST --result-stage V1223_LINEC_REPAIR_A57A59_E12 --summary-stage V1223_LINEC_REPAIR_A57A59_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A57A59_E12_ROUTE --result-scope v1223_linec_repair_A57A59_e12 --device cuda:2 --data-root data --datasets KMNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A57-A51BoundQLineCRepair,A58-A51SignalBlock010LineCRepair,A59-A51SignalBroad025Block010BoundQ --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A57A59-e12 --route-impact feeds-v1223-linec-repair-continuation
```

子任务结果：

- MNIST：rows `24`，summary rows `8`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.006184895833333333`
- Fashion-MNIST：rows `24`，summary rows `8`，best `A59-A51SignalBroad025Block010BoundQ`，best mean delta vs A0 `0.010416666666666666`
- KMNIST：rows `24`，summary rows `8`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.0048828125`

### 11.3 A57-A59 official 汇总

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_linec_repair_A57A59 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.004,0.005,0.006,0.007,0.008,0.009,0.01 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST,Fashion-MNIST --line-d-seeds 0 --line-d-train-size 512 --line-d-val-size 256 --line-d-batch-size 128 --line-d-epochs 3 --line-d-linec-batch 32 --line-d-sketch-dim 16
```

结果：

- route：`R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `label_free_best_hardening_candidate=A51-StagedUnlabeledAdapt-warm1-r002`
- `label_free_best_hardening_mean_delta_vs_A0=0.003146701388888889`
- `label_free_best_linec_pass_rate=0.0`
- `linea_rows=511`
- `linea_hardening_rows=378`
- `actuator_rows=3618`
- `actuator_release_audit_rows=309`
- `actuator_exploratory_release_dataset_seed_count=0`
- `line_d_family_rows=10`
- zip sha：`c83ecb4053fa129371c960659d7ad3630a35f1342cde47e11d43bf24a88aee01`

### 11.4 Actuator LineC delta 同 seed baseline 修复与复核

发现的问题：

```text
write_actuator_roleaware_v4 原先用 seed+7300 计算 base_metrics，用 seed+8400+budget 计算 trial metrics。
这会让 NoOp control 也出现非零 Noise/Reservoir delta，污染 release_audit 和 matched_control_gap。
```

修复方向：

```text
每个 dataset/seed/budget 使用相同 metric_seed 计算 base 与 trial LineC metrics，并缓存 base metrics。
```

复核命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_actuator_seedfix --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.004,0.005,0.006,0.007,0.008,0.009,0.01 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST,Fashion-MNIST --line-d-seeds 0 --line-d-train-size 512 --line-d-val-size 256 --line-d-batch-size 128 --line-d-epochs 3 --line-d-linec-batch 32 --line-d-sketch-dim 16
```

结果：

- route：`R4-FunctionalMechanismNoGoAfterAllFallbacks`
- NoOp control max abs delta：`0.0`
- `actuator_release_audit_rows` 从修复前的 `309` 降为 `48`
- `actuator_exploratory_release_dataset_seed_count=0`
- `functional_p3_pass_count=0`
- zip sha：`bf718486ae706c678ae8076bd65a963c1a8410e6dd5169dd37344c0edb5e42ac`

### 11.5 A60-A62 direct/readout LineC repair 三数据集并行

共同参数：

```text
datasets：MNIST / Fashion-MNIST / KMNIST
seeds：0,1,2
train/val/test：2048 / 1024 / 1024
epochs：12
LineC batch/sketch：64 / 16 / 24
ablation ids：A0,A1,A51,A58,A60,A61,A62
```

等价复现命令，实际执行时将三条命令分别放到 GPU 0/1/2 并行：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A60A62_e12_MNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A60A62_e12/MNIST --artifact-prefix v1223_linec_repair_A60A62_e12_MNIST --result-stage V1223_LINEC_REPAIR_A60A62_E12 --summary-stage V1223_LINEC_REPAIR_A60A62_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A60A62_E12_ROUTE --result-scope v1223_linec_repair_A60A62_e12 --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A58-A51SignalBlock010LineCRepair,A60-A51DirectRead125ReservoirRepair,A61-A51IdentityAmp150ReservoirRepair,A62-A58DirectRead125ReservoirRepair --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A60A62-e12 --route-impact feeds-v1223-linec-direct-readout-repair-continuation
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A60A62_e12_FashionMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A60A62_e12/Fashion-MNIST --artifact-prefix v1223_linec_repair_A60A62_e12_FashionMNIST --result-stage V1223_LINEC_REPAIR_A60A62_E12 --summary-stage V1223_LINEC_REPAIR_A60A62_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A60A62_E12_ROUTE --result-scope v1223_linec_repair_A60A62_e12 --device cuda:1 --data-root data --datasets Fashion-MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A58-A51SignalBlock010LineCRepair,A60-A51DirectRead125ReservoirRepair,A61-A51IdentityAmp150ReservoirRepair,A62-A58DirectRead125ReservoirRepair --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A60A62-e12 --route-impact feeds-v1223-linec-direct-readout-repair-continuation
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A60A62_e12_KMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A60A62_e12/KMNIST --artifact-prefix v1223_linec_repair_A60A62_e12_KMNIST --result-stage V1223_LINEC_REPAIR_A60A62_E12 --summary-stage V1223_LINEC_REPAIR_A60A62_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A60A62_E12_ROUTE --result-scope v1223_linec_repair_A60A62_e12 --device cuda:2 --data-root data --datasets KMNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A58-A51SignalBlock010LineCRepair,A60-A51DirectRead125ReservoirRepair,A61-A51IdentityAmp150ReservoirRepair,A62-A58DirectRead125ReservoirRepair --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A60A62-e12 --route-impact feeds-v1223-linec-direct-readout-repair-continuation
```

MNIST 使用 `cuda:0`，Fashion-MNIST 使用 `cuda:1`，KMNIST 使用 `cuda:2`。完整命令中的 `run_id` 与输出目录分别为：

- `v1223_linec_repair_A60A62_e12_MNIST`
  - `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A60A62_e12/MNIST`
- `v1223_linec_repair_A60A62_e12_FashionMNIST`
  - `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A60A62_e12/Fashion-MNIST`
- `v1223_linec_repair_A60A62_e12_KMNIST`
  - `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A60A62_e12/KMNIST`

子任务结果：

- MNIST：rows `27`，summary rows `9`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.006184895833333333`
- Fashion-MNIST：rows `27`，summary rows `9`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `-0.0016276041666666667`
- KMNIST：rows `27`，summary rows `9`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.0048828125`
- A60/A61/A62 全部 LineC pass rate `0.0`

### 11.6 A63-A65 RMSQ LineC repair 三数据集并行

共同参数：

```text
datasets：MNIST / Fashion-MNIST / KMNIST
seeds：0,1,2
train/val/test：2048 / 1024 / 1024
epochs：12
LineC batch/sketch：64 / 16 / 24
ablation ids：A0,A1,A51,A58,A63,A64,A65
```

等价复现命令，实际执行时将三条命令分别放到 GPU 0/1/2 并行：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A63A65_e12_MNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A63A65_e12/MNIST --artifact-prefix v1223_linec_repair_A63A65_e12_MNIST --result-stage V1223_LINEC_REPAIR_A63A65_E12 --summary-stage V1223_LINEC_REPAIR_A63A65_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A63A65_E12_ROUTE --result-scope v1223_linec_repair_A63A65_e12 --device cuda:0 --data-root data --datasets MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A58-A51SignalBlock010LineCRepair,A63-A51RMSQReservoirRepair,A64-A51RMSQBoundQReservoirRepair,A65-A58RMSQSignalBlockRepair --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A63A65-e12 --route-impact feeds-v1223-linec-rmsq-repair-continuation
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A63A65_e12_FashionMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A63A65_e12/Fashion-MNIST --artifact-prefix v1223_linec_repair_A63A65_e12_FashionMNIST --result-stage V1223_LINEC_REPAIR_A63A65_E12 --summary-stage V1223_LINEC_REPAIR_A63A65_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A63A65_E12_ROUTE --result-scope v1223_linec_repair_A63A65_e12 --device cuda:1 --data-root data --datasets Fashion-MNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A58-A51SignalBlock010LineCRepair,A63-A51RMSQReservoirRepair,A64-A51RMSQBoundQReservoirRepair,A65-A58RMSQSignalBlockRepair --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A63A65-e12 --route-impact feeds-v1223-linec-rmsq-repair-continuation
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id v1223_linec_repair_A63A65_e12_KMNIST --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A63A65_e12/KMNIST --artifact-prefix v1223_linec_repair_A63A65_e12_KMNIST --result-stage V1223_LINEC_REPAIR_A63A65_E12 --summary-stage V1223_LINEC_REPAIR_A63A65_E12_SUMMARY --route-stage V1223_LINEC_REPAIR_A63A65_E12_ROUTE --result-scope v1223_linec_repair_A63A65_e12 --device cuda:2 --data-root data --datasets KMNIST --seeds 0,1,2 --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A58-A51SignalBlock010LineCRepair,A63-A51RMSQReservoirRepair,A64-A51RMSQBoundQReservoirRepair,A65-A58RMSQSignalBlockRepair --train-size 2048 --val-size 1024 --test-size 1024 --batch-size 128 --epochs 12 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 16 --linec-sketch-dim 24 --protocol-note v12.23-linec-repair-A63A65-e12 --route-impact feeds-v1223-linec-rmsq-repair-continuation
```

MNIST 使用 `cuda:0`，Fashion-MNIST 使用 `cuda:1`，KMNIST 使用 `cuda:2`。完整命令中的 `run_id` 与输出目录分别为：

- `v1223_linec_repair_A63A65_e12_MNIST`
  - `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A63A65_e12/MNIST`
- `v1223_linec_repair_A63A65_e12_FashionMNIST`
  - `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A63A65_e12/Fashion-MNIST`
- `v1223_linec_repair_A63A65_e12_KMNIST`
  - `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea/linec_repair_A63A65_e12/KMNIST`

子任务结果：

- MNIST：rows `27`，summary rows `9`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.006184895833333333`
- Fashion-MNIST：rows `27`，summary rows `9`，best `A63-A51RMSQReservoirRepair`，best mean delta vs A0 `0.0022786458333333335`
- KMNIST：rows `27`，summary rows `9`，best `A51-StagedUnlabeledAdapt-warm1-r002`，best mean delta vs A0 `0.004557291666666667`
- A63/A65 在 Fashion-MNIST 上改善 task/AUC，但所有候选 LineC pass rate 仍为 `0.0`

### 11.7 最终 official runner

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_rmsq_seedfix_final --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.004,0.005,0.006,0.007,0.008,0.009,0.01 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST,Fashion-MNIST --line-d-seeds 0 --line-d-train-size 512 --line-d-val-size 256 --line-d-batch-size 128 --line-d-epochs 3 --line-d-linec-batch 32 --line-d-sketch-dim 16
```

最终结果 JSON 摘要：

- `run_id=official_explore_open2_rmsq_seedfix_final`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `fail_reason=all mandatory exploration levels exhausted without promotion`
- `official_success_reached=0`
- `label_free_best_hardening_candidate=A51-StagedUnlabeledAdapt-warm1-r002`
- `label_free_best_hardening_mean_delta_vs_A0=0.003146701388888889`
- `label_free_best_linec_pass_rate=0.0`
- `label_free_candidate_rows=27`
- `linea_rows=673`
- `linea_hardening_rows=540`
- `linec_deployable_rows=427`
- `T1A_auc_joint=0.41019417475728154`
- `T1B_auc_joint=0.5032062807143713`
- `T2_clone_probe_auc_joint=0.7383255633255633`
- `T2_clone_probe_precision_at_k=0.3181818181818182`
- `T2_clone_probe_recall_at_k=0.3181818181818182`
- `T3_visibility_support_auc_min=0.6660482374768089`
- `T3_visibility_support_stable_rows=6`
- `actuator_rows=3618`
- `actuator_release_audit_rows=48`
- `actuator_exploratory_release_dataset_seed_count=0`
- `functional_p3_rows=6`
- `functional_p3_pass_count=0`
- `line_d_family_rows=10`
- `line_d_kernel_blocked_count=0`
- `required_artifact_rows=49`
- `required_artifact_missing_count=0`
- `code_review_packet_sha256=18fa7c952f3e4143b7d35bd9ef439039e38ba317db11693bbea97bfcb4410ff4`

### 11.8 最终审计命令

route 和关键表审计：

```bash
conda run -n kan python -c "import json,csv,pathlib; base=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'); route=json.loads((base/'v1223_route_decision.json').read_text()); print(route['run_id'], route['route'], route['code_review_packet_sha256']); print('required_missing', route['required_artifact_missing_count'])"
```

actuator seed-fix 审计：

```bash
conda run -n kan python -c "import csv,pathlib; base=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'); rows=list(csv.DictReader((base/'v1223_actuator_safety_roleaware.csv').open())); print(max(abs(float(r['NoiseSignalLeak_delta_audit']))+abs(float(r['RealSignalReservoirRatio_delta_audit'])) for r in rows if r['actuator_id']=='NoOpMatchedOverhead'))"
```

zip 审计：

```bash
conda run -n kan python -c "import zipfile,pathlib,hashlib; p=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip'); print(p.exists(), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()); print(zipfile.ZipFile(p).namelist())"
```

zip 结果：

- path：`results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`
- exists：`True`
- size：`1224305`
- sha256：`18fa7c952f3e4143b7d35bd9ef439039e38ba317db11693bbea97bfcb4410ff4`
- entries：`46`
- 包含新代码：
  - `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - `experiments/run_v1218_b320_label_free_ablation.py`
- 包含核心审计结果：
  - `v1223_route_decision.json`
  - `v1223_label_free_hardening.csv`
  - `v1223_label_free_linec.csv`
  - `v1223_actuator_safety_roleaware.csv`
  - `v1223_functional_p3_control_audit.csv`
  - `v1223_classic_family_status.csv`

## 12. 用户要求“未达成目标则继续”后的补充推进

本节记录在用户再次询问 `docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md 达成目标了吗` 后执行的继续推进。前一轮 official 结果仍为 `R4-FunctionalMechanismNoGoAfterAllFallbacks`，因此不能回答“达成”，继续按文档中的推荐方向补 Line I drift-threshold scan 空档与 Line D Rational/Fourier hardening。

### 12.1 代码修改与语法检查

修改文件：

- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - 在 `package_zip()` 固定代码清单中加入：
    - `experiments/run_v1223_line_d_hardening.py`
    - `experiments/summarize_v1223_line_d_hardening.py`
  - 目的：确保本轮新增 Line D hardening driver 与汇总脚本进入 v12.23 code review packet。
- `experiments/run_v1223_line_d_hardening.py`
  - 新增 targeted Line D hardening driver。
  - 复用官方 v12.23 的 `classic_family_smoke_one()`，按 family/dataset/seed/GPU 独立执行，输出顶层 `v1223_line_d_hardening_*` CSV/JSON，便于被 official zip 自动收集。
  - 强制 CUDA；CPU offload 不允许。
- `experiments/summarize_v1223_line_d_hardening.py`
  - 新增 hardening shard 聚合脚本。
  - 输出：
    - `v1223_line_d_hardening_aggregate.csv`
    - `v1223_line_d_hardening_aggregate_summary.csv`
    - `v1223_line_d_hardening_aggregate_summary.json`

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_line_d_hardening.py
```

结果：通过，无输出。

新增汇总脚本后再次语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_line_d_hardening.py experiments/summarize_v1223_line_d_hardening.py
```

结果：通过，无输出。

### 12.2 Line I dense actuator threshold scan

补充目的：前一轮 seed-fix official run 已确认 `safe` 与 `release` 分离，但预算主要集中在 `0.004+`；因此补 `0.0003-0.004` 的 dense scan，验证是否存在 drift-safe release threshold。

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_actuator_threshold_scan --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST,Fashion-MNIST --line-d-seeds 0 --line-d-train-size 512 --line-d-val-size 256 --line-d-batch-size 128 --line-d-epochs 3 --line-d-linec-batch 32 --line-d-sketch-dim 16
```

结果摘要：

- `run_id=official_explore_open2_actuator_threshold_scan`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `official_success_reached=0`
- `actuator_rows=6534`
- `actuator_release_audit_rows=120`
- `actuator_role_safe_movement_rows=1228`
- `actuator_exploratory_release_dataset_seed_count=0`
- `actuator_official_gate_rows=0`
- `functional_p3_pass_count=0`
- `p4_open=0`
- `line_d_family_rows=10`
- `line_d_kernel_blocked_count=0`
- `required_artifact_missing_count=0`
- 当时 zip sha256：`1f02a899dae2f15c6307bc0e1a0373432b2aed3ca362937b6a408637d3335032`

该 scan 没有打开 S3/P3/P4，但给后续判断提供了更密预算证据。

### 12.3 Line D Rational/Fourier hardening 四 GPU 并行

共同参数：

```text
families: Rational / Fourier
datasets: MNIST / Fashion-MNIST
seeds: 0,1,2
train/val: 1024 / 512
epochs: 8
batch size: 128
LineC batch/sketch: 64 / 24
promotion_allowed: 0
```

四条并行命令：

```bash
conda run -n kan python experiments/run_v1223_line_d_hardening.py --run-id v1223_line_d_hardening_rational_mnist_e8 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --artifact-prefix v1223_line_d_hardening_rational_mnist_e8 --device cuda:0 --data-root data --families Rational --datasets MNIST --seeds 0,1,2 --train-size 1024 --val-size 512 --batch-size 128 --epochs 8 --linec-batch-size 64 --linec-sketch-dim 24
conda run -n kan python experiments/run_v1223_line_d_hardening.py --run-id v1223_line_d_hardening_rational_fashion_e8 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --artifact-prefix v1223_line_d_hardening_rational_fashion_e8 --device cuda:1 --data-root data --families Rational --datasets Fashion-MNIST --seeds 0,1,2 --train-size 1024 --val-size 512 --batch-size 128 --epochs 8 --linec-batch-size 64 --linec-sketch-dim 24
conda run -n kan python experiments/run_v1223_line_d_hardening.py --run-id v1223_line_d_hardening_fourier_mnist_e8 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --artifact-prefix v1223_line_d_hardening_fourier_mnist_e8 --device cuda:2 --data-root data --families Fourier --datasets MNIST --seeds 0,1,2 --train-size 1024 --val-size 512 --batch-size 128 --epochs 8 --linec-batch-size 64 --linec-sketch-dim 24
conda run -n kan python experiments/run_v1223_line_d_hardening.py --run-id v1223_line_d_hardening_fourier_fashion_e8 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --artifact-prefix v1223_line_d_hardening_fourier_fashion_e8 --device cuda:3 --data-root data --families Fourier --datasets Fashion-MNIST --seeds 0,1,2 --train-size 1024 --val-size 512 --batch-size 128 --epochs 8 --linec-batch-size 64 --linec-sketch-dim 24
```

四条均正常结束：

- `v1223_line_d_hardening_rational_mnist_e8`: rows `3`, family_near_pass_rows `3`, kernel_blocked_rows `0`
- `v1223_line_d_hardening_rational_fashion_e8`: rows `3`, family_near_pass_rows `3`, kernel_blocked_rows `0`
- `v1223_line_d_hardening_fourier_mnist_e8`: rows `3`, family_near_pass_rows `3`, kernel_blocked_rows `0`
- `v1223_line_d_hardening_fourier_fashion_e8`: rows `3`, family_near_pass_rows `3`, kernel_blocked_rows `0`

### 12.4 Line D hardening 聚合

曾尝试使用 heredoc 直接聚合：

```bash
conda run -n kan python - <<'PY'
...
PY
```

实际结果：命令返回 0 但没有生成 `v1223_line_d_hardening_aggregate.*` 文件。该命令不计入有效聚合结果，已改为可审计脚本执行。

有效聚合命令：

```bash
conda run -n kan python experiments/summarize_v1223_line_d_hardening.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --pattern v1223_line_d_hardening_*_e8.csv --artifact-prefix v1223_line_d_hardening_aggregate
```

有效聚合结果：

- input files：4 个 hardening CSV shard
- aggregate rows：`12`
- summary rows：`4`
- family_near_pass_rows：`12`
- kernel_blocked_rows：`0`
- best_by_val_acc：
  - family：`Rational`
  - dataset：`MNIST`
  - seed：`0`
  - val_acc：`0.701171875`
  - LineC CouplingR2：`0.13238496612899686`
- best_by_linec_CouplingR2：
  - family：`Fourier`
  - dataset：`Fashion-MNIST`
  - seed：`2`
  - val_acc：`0.609375`
  - LineC CouplingR2：`0.17223205548419696`

聚合 summary 表：

| family | dataset | rows | mean_val_acc | min_val_acc | max_val_acc | mean_NLL | mean_linec_CouplingR2 | mean_linec_NoiseSignalLeak | mean_linec_RealSignalReservoirRatio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fourier | Fashion-MNIST | 3 | 0.5846354166666666 | 0.5546875 | 0.609375 | 1.3872931798299153 | 0.12844373058968392 | 0.0521984708805879 | 0.6052201688289642 |
| Fourier | MNIST | 3 | 0.6647135416666666 | 0.650390625 | 0.68359375 | 1.6000018914540608 | 0.12411807126846226 | 0.06800038740038872 | 0.6920690933863322 |
| Rational | Fashion-MNIST | 3 | 0.6588541666666666 | 0.630859375 | 0.67578125 | 1.3111967245737712 | 0.10925706826505344 | 0.05980806487301985 | 0.4520403246084849 |
| Rational | MNIST | 3 | 0.6744791666666666 | 0.6484375 | 0.701171875 | 1.7119425932566326 | 0.13741879786845176 | 0.07816234479347865 | 0.5534500678380331 |

### 12.5 最终 official rerun

目的：在加入 Line D hardening driver、汇总脚本和聚合产物后，重跑 official route 并刷新 code review packet。

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id official_explore_open2_lined_hardening_final --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2 --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:3 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:2 --line-d-datasets MNIST,Fashion-MNIST --line-d-seeds 0 --line-d-train-size 512 --line-d-val-size 256 --line-d-batch-size 128 --line-d-epochs 3 --line-d-linec-batch 32 --line-d-sketch-dim 16
```

最终 route JSON 摘要：

- `run_id=official_explore_open2_lined_hardening_final`
- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `fail_reason=all mandatory exploration levels exhausted without promotion`
- `official_success_reached=0`
- `final_stop_allowed=1`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `required_artifact_missing_count=0`
- `label_free_best_hardening_candidate=A51-StagedUnlabeledAdapt-warm1-r002`
- `label_free_best_hardening_mean_delta_vs_A0=0.003146701388888889`
- `label_free_best_linec_pass_rate=0.0`
- `label_free_official_pass_count=0`
- `label_free_exploration_pass_count=0`
- `T1A_auc_joint=0.41019417475728154`
- `T1B_auc_joint=0.5032062807143713`
- `T2_clone_probe_auc_joint=0.7383255633255633`
- `T2_clone_probe_precision_at_k=0.3181818181818182`
- `T2_clone_probe_recall_at_k=0.3181818181818182`
- `T3_visibility_support_auc_min=0.6660482374768089`
- `actuator_rows=6534`
- `actuator_release_audit_rows=120`
- `actuator_role_safe_movement_rows=1228`
- `actuator_exploratory_release_dataset_seed_count=0`
- `actuator_official_gate_rows=0`
- `functional_p3_pass_count=0`
- `p4_open=0`
- `line_d_family_rows=10`
- `line_d_kernel_blocked_count=0`
- `code_review_packet_sha256=6a65395e8cb2cd851de80821aaac395db48dc80cf82e574d5805ea236f7cd424`

### 12.6 最终审计命令

route 审计：

```bash
conda run -n kan python -c "import json,pathlib; base=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'); route=json.loads((base/'v1223_route_decision.json').read_text()); keys=['run_id','route','fail_reason','official_success_reached','final_stop_allowed','hard_budget_exhausted','mandatory_exploration_levels_executed','required_artifact_missing_count','code_review_packet_sha256','actuator_rows','actuator_release_audit_rows','actuator_role_safe_movement_rows','actuator_exploratory_release_dataset_seed_count','functional_p3_pass_count','p4_open','line_d_family_rows','line_d_kernel_blocked_count']; print(json.dumps({k:route.get(k) for k in keys}, indent=2, ensure_ascii=False, sort_keys=True))"
```

结果确认：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `official_success_reached=0`
- `required_artifact_missing_count=0`
- `code_review_packet_sha256=6a65395e8cb2cd851de80821aaac395db48dc80cf82e574d5805ea236f7cd424`

zip 审计：

```bash
conda run -n kan python -c "import zipfile,pathlib,hashlib,json; p=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip'); names=zipfile.ZipFile(p).namelist(); wanted=['experiments/run_v1223_line_d_hardening.py','experiments/summarize_v1223_line_d_hardening.py','results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate.csv','results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate_summary.csv','results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate_summary.json']; print(json.dumps({'exists':p.exists(),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'entries':len(names),'wanted_present':{w:(w in names) for w in wanted}}, indent=2, ensure_ascii=False, sort_keys=True))"
```

结果：

- path：`results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`
- exists：`true`
- size：`1857914`
- entries：`63`
- sha256：`6a65395e8cb2cd851de80821aaac395db48dc80cf82e574d5805ea236f7cd424`
- 确认入包：
  - `experiments/run_v1223_line_d_hardening.py`
  - `experiments/summarize_v1223_line_d_hardening.py`
  - `v1223_line_d_hardening_aggregate.csv`
  - `v1223_line_d_hardening_aggregate_summary.csv`
  - `v1223_line_d_hardening_aggregate_summary.json`

actuator 审计命令曾有一次失败：

```bash
conda run -n kan python -c "import csv,pathlib,json,math; ... \nfor aid in aids:\n ..."
```

失败原因：`python -c` 字符串中 `\n` 转义导致 `SyntaxError: unexpected character after line continuation character`。该命令未产出数据，不计为实验结果。

修正后的 actuator 审计命令：

```bash
conda run -n kan python -c "import csv,pathlib,json; base=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'); rows=list(csv.DictReader((base/'v1223_actuator_safety_roleaware.csv').open())); aids=['I11-BranchGainRedistributionRoleSafe','I12-DirectRoleScaleShiftRoleSafe','I13-QuadResidualLowDriftRotation','I14-ResponseComposedBranchDirect','I15-NoiseReleaseReservoirReleaseQP','I17-ShadowPositiveReplayMatchedControls','I18-AdamWOrthogonalResidual-diagnostic']; out=[{'actuator_id':aid,'rows':sum(1 for r in rows if r['actuator_id']==aid),'safe_rows':sum(int(float(r['role_safe_movement_pass'])) for r in rows if r['actuator_id']==aid),'release_rows':sum(int(float(r['release_audit_pass'])) for r in rows if r['actuator_id']==aid),'safe_release_rows':sum(int(float(r['role_safe_movement_pass'])) and int(float(r['release_audit_pass'])) for r in rows if r['actuator_id']==aid),'exploratory_rows':sum(int(float(r['exploratory_release_gate'])) for r in rows if r['actuator_id']==aid),'best_control_gap':max([float(r['control_gap']) for r in rows if r['actuator_id']==aid and r['control_gap'] not in ('',None)] or [float('nan')]),'min_drift':min([float(r['logit_max_abs_drift']) for r in rows if r['actuator_id']==aid] or [float('nan')])} for aid in aids]; print(json.dumps(out, indent=2, ensure_ascii=False))"
```

结果：

| actuator | rows | safe_rows | release_rows | safe_release_rows | exploratory_rows | best_control_gap | min_drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| I11 | 378 | 366 | 0 | 0 | 0 | -0.016013488173484802 | 0.000038623809814453125 |
| I12 | 288 | 288 | 0 | 0 | 0 | -0.018504545092582703 | 0.00008755922317504883 |
| I13 | 378 | 0 | 32 | 0 | 0 | 0.010798055678606033 | 0.23476892709732056 |
| I14 | 288 | 288 | 0 | 0 | 0 | -0.0175180584192276 | 0.0006197690963745117 |
| I15 | 378 | 0 | 28 | 0 | 0 | 0.033934324979782104 | 0.22774165868759155 |
| I17 | 378 | 0 | 30 | 0 | 0 | 0.051491498947143555 | 0.2368667721748352 |
| I18 diagnostic | 378 | 0 | 30 | 0 | 0 | 0.014657139778137207 | 0.24330437183380127 |

NoOp seed-fix 审计命令：

```bash
conda run -n kan python -c "import csv,pathlib; base=pathlib.Path('results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2'); rows=list(csv.DictReader((base/'v1223_actuator_safety_roleaware.csv').open())); vals=[abs(float(r['NoiseSignalLeak_delta_audit']))+abs(float(r['RealSignalReservoirRatio_delta_audit'])) for r in rows if r['actuator_id']=='NoOpMatchedOverhead']; print({'noop_rows':len(vals),'max_abs_metric_delta_sum':max(vals) if vals else None})"
```

结果：

- `noop_rows=378`
- `max_abs_metric_delta_sum=0.0`

### 12.7 本节新增/刷新关键产物

- `experiments/run_v1223_line_d_hardening.py`
- `experiments/summarize_v1223_line_d_hardening.py`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_rational_mnist_e8.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_rational_fashion_e8.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_fourier_mnist_e8.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_fourier_fashion_e8.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate_summary.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate_summary.json`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_route_decision.json`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`


## 13. 继续推进记录：I24/I25、control-scope 修复、P4 short-run

本节记录用户要求继续推进后的新增执行。工作目录：/home/chengshun.wang/DG-LCA。运行环境：conda env kan。

### 13.1 环境异常

普通沙箱读文件、apply_patch、一次日志追加曾报：

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

后续使用命令前缀：

```bash
conda run -n kan ...
```

另有一次日志追加命令因为 Markdown 反引号被 shell 展开而失败，已终止该错误会话。该错误没有产生实验数据，不纳入实验结论。

### 13.2 代码变更

修改或新增：

```text
experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
experiments/summarize_v1223_actuator_rows.py
experiments/recompute_v1223_control_scope_and_route.py
experiments/run_v1223_p4_short_from_actuator.py
```

实际做了以下修复：

```text
1. runner 增加 --actuator-ids，便于只跑指定 actuator，controls 自动保留。
2. 增加 I19-I25 与 DirectLogitCompensatedRandomControl 等 controls。
3. 增加 direct_logit_compensation_* 字段，审计 direct_readout compensation 前后 drift。
4. Functional P3 source selector 改为优先选择满足 P3 硬条件的 row，否则才 fallback 到 exploratory/control_gap 排序。
5. matched_control_gap scope 从全 out-dir 同 budget/sign 修为同 dataset/seed/budget/sign，并写 control_scope=dataset_seed_budget_sign。
6. 新增 recompute 脚本，用已有 CSV 重算 local control-scope gate、Functional P3、route、zip。
7. 新增 P4 short-run 脚本，P3 打开后执行 source/control/noop 短训审计，promotion_allowed 保持 0。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/summarize_v1223_actuator_rows.py experiments/recompute_v1223_control_scope_and_route.py experiments/run_v1223_p4_short_from_actuator.py
```

结果：通过，无错误输出。

### 13.3 I19-I25 smoke 结果

I19-I21 low-budget smoke 命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i19_i21_mnist_seed0_smoke --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i19_smoke --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:0 --actuator-datasets MNIST --actuator-seeds 0 --actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:0 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：route=R4，actuator_rows=894，role_safe_movement_rows=240，release_audit_rows=8，exploratory_dataset_seed_count=0。

I19-I21 high-budget smoke 命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i20_i21_highbudget_mnist_seed0_smoke --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i20_highbudget_smoke --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:0 --actuator-datasets MNIST --actuator-seeds 0 --actuator-budgets 0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:0 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：route=R4，actuator_rows=542，release_audit_rows=12，exploratory_dataset_seed_count=0。

I22/I23 mean-comp smoke 命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i22_i23_meancomp_mnist_seed0_smoke --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i22_meancomp_smoke --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:0 --actuator-datasets MNIST --actuator-seeds 0 --actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.03 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:0 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：route=R4，actuator_rows=1170，release_audit_rows=19，role_safe_movement_rows=265，exploratory_dataset_seed_count=0。

I24/I25 MNIST seed0 smoke 命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i24_i25_directcomp_mnist_seed0_smoke --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_smoke --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:0 --actuator-datasets MNIST --actuator-seeds 0 --actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.03 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:0 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：route=R4，actuator_rows=1314，role_safe_movement_rows=361，release_audit_rows=26，exploratory_dataset_seed_count=1，official_gate_rows=4。

代表行：I24 / MNIST seed0 / budget 0.03 / sign +1：

```text
direct_logit_compensation_pre_drift = 1.2538255453109741
direct_logit_compensation_post_drift = 0.015671372413635254
logit_max_abs_drift = 0.015671372413635254
role_safe_movement_pass = 1
release_audit_pass = 1
exploratory_release_gate = 1
official_release_gate = 1
control_gap = 0.021569199860095978
NoiseSignalLeak_delta_audit = -0.08185220509767532
RealSignalReservoirRatio_delta_audit = -0.03331410884857178
```

### 13.4 四 GPU 分片验证

命令分别对应 GPU0 MNIST seed1/2、GPU1 Fashion-MNIST seed0/1/2、GPU2 KMNIST seed0/1、GPU3 KMNIST seed2。共同参数：

```text
--actuator-ids I24-DirectLogitCompensatedQuadRelease,I25-DirectLogitCompensatedShadowRelease
--actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.03
--actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8
```

结果：

| split | route | exploratory dataset-seed count | P3 | P4 open |
|---|---|---:|---:|---:|
| MNIST seed1/2 | R4 | 0 | 0 | 0 |
| Fashion-MNIST seed0/1/2 | R4 | 2 | 0 | 0 |
| KMNIST seed0/1 | S4 | 2 | 6 | 1 |
| KMNIST seed2 | R4 | 0 | 0 | 0 |

### 13.5 合并 targeted official 与 route 修复

合并 run 命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i24_i25_directcomp_all_targeted_official --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --linea-root results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/linea --data-root data --actuator-device cuda:0 --actuator-datasets MNIST,Fashion-MNIST,KMNIST --actuator-seeds 0,1,2 --actuator-ids I24-DirectLogitCompensatedQuadRelease,I25-DirectLogitCompensatedShadowRelease --actuator-budgets 0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.03 --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:0 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

旧 global control scope 初始结果：route=S3，actuator_exploratory_release_dataset_seed_count=4，functional_p3_pass_count=0。

修复重算命令：

```bash
conda run -n kan python experiments/recompute_v1223_control_scope_and_route.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted
```

local control scope 修复后结果：

```text
route = S4-FunctionalP3Opened
minimum_success = Minimum Success D
functional_p3_pass_count = 6
p4_open = 1
p4_pass = 0
actuator_exploratory_release_dataset_seed_count = 6
actuator_official_gate_rows = 13
actuator_exploratory_success = 1
actuator_official_gate_pass = 1
control_scope_fix_applied = 1
required_artifact_missing_count = 0
```

P3 source：

```text
source_actuator = I24-DirectLogitCompensatedQuadRelease
source_dataset = KMNIST
source_seed = 0
norm_budget = 0.009
signed_direction = +1
CouplingR2_delta = 0.05668966566648037
NoiseSignalLeak_delta = -0.02430429309606552
Reservoir_delta = -0.06281167268753052
control_gap = 0.04396222531795502
CEp99_delta = -0.07897424697875977
ECE_delta = -0.031210273504257202
source_selection_reason = p3_hard_gate_candidate
```

### 13.6 P4 short-run 审计

1-epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_short_from_actuator.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:1 --train-size 256 --val-size 128 --batch-size 64 --epochs 1 --linec-batch 32 --linec-sketch-dim 8
```

结果：

```text
source_final_acc = 0.609375
noop_final_acc = 0.5859375
control_final_acc = 0.625
source_vs_noop_acc_delta = +0.0234375
source_vs_control_acc_delta = -0.015625
p4_pass = 0
promotion_allowed = 0
```

3-epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_short_from_actuator.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:1 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8
```

结果：

```text
source_final_acc = 0.703125
noop_final_acc = 0.7421875
control_final_acc = 0.703125
source_vs_noop_acc_delta = -0.0390625
source_vs_control_acc_delta = 0.0
p4_pass = 0
promotion_allowed = 0
```

### 13.7 最终产物

最终 targeted out-dir：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted
```

最终 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
```

zip 审计：

```text
exists = true
size = 2045934
entries = 58
sha256 = f9f0478bd2392f0f1bdc778820e1d1f7813c7757735d7a16eea09d42963ef3b0
```

确认入包：runner、summarizer、recompute 脚本、P4 short 脚本、P4 csv/json、1-epoch P4 summary 备份。


---

# 14. 继续推进：P4 survivor 全量扫描、shadow-P4-to-P3 targeted repair、zip 审计修复

本节记录用户追问“达成目标了吗，没有请继续”后的追加执行。结论先写在执行日志顶部，方便复现者判断：本节没有把 v12.23 推到 S5；它把 P4 fail 的原因进一步收窄为“P3 机制信号与短训任务收益不对齐”。同时，本节发现并修复了一个复现包问题：新写的 P4 audit 脚本已经执行，但最初没有进入 zip。修复后两个最终 zip 均包含新脚本。

## 14.1 P4 compensation-mode audit 脚本与官方 P4 survivor 扫描脚本

新增文件：

```text
experiments/run_v1223_p4_compensation_modes.py
experiments/run_v1223_p4_official_row_scan.py
```

目的：

```text
1. run_v1223_p4_compensation_modes.py：对已通过 P3 的 source，在不同 compensation substrate 下做 P4 短训。
2. run_v1223_p4_official_row_scan.py：不只看 P3 source，而是扫描全部 official_release_gate=1 的 actuator survivor，判断是否存在 audit-only P4 task advantage。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_compensation_modes.py experiments/run_v1223_p4_official_row_scan.py
```

结果：命令退出码 0。

## 14.2 原 targeted official out-dir 的 P4 compensation-mode scan

out-dir：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted
```

1 epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_compensation_modes.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 256 --val-size 128 --batch-size 64 --epochs 1 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_compensation_modes_e1
```

1 epoch 结果：

```text
any_p4_pass = 0
required_artifact_missing_count = 0
per_variant_query: source=0.5703125, noop=0.6171875, control=0.59375, source-noop=-0.046875, source-control=-0.0234375
shared_source_query: source=0.5703125, noop=0.59375, control=0.5859375, source-noop=-0.0234375, source-control=-0.015625
per_variant_train: source=0.578125, noop=0.6171875, control=0.5859375, source-noop=-0.0390625, source-control=-0.0078125
shared_source_train: source=0.578125, noop=0.6015625, control=0.609375, source-noop=-0.0234375, source-control=-0.03125
none: source=0.6015625, noop=0.6171875, control=0.59375, source-noop=-0.015625, source-control=+0.0078125
```

3 epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_compensation_modes.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:1 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_compensation_modes_e3
```

3 epoch 结果：

```text
any_p4_pass = 0
required_artifact_missing_count = 0
per_variant_query: source=0.6796875, noop=0.7109375, control=0.71875, source-noop=-0.03125, source-control=-0.0390625
shared_source_query: source=0.6796875, noop=0.703125, control=0.7265625, source-noop=-0.0234375, source-control=-0.046875
per_variant_train: source=0.6640625, noop=0.7109375, control=0.71875, source-noop=-0.046875, source-control=-0.0546875
shared_source_train: source=0.6640625, noop=0.703125, control=0.75, source-noop=-0.0390625, source-control=-0.0859375
none: source=0.671875, noop=0.7109375, control=0.7109375, source-noop=-0.0390625, source-control=-0.0390625
```

解释：对原 P3 source，换 compensation substrate、换 query/train reference、甚至不做 compensation，都不能让 source 同时胜过 noop/control。

## 14.3 原 targeted official out-dir 的 official survivor P4 row scan

1 epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_official_row_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:2 --train-size 256 --val-size 128 --batch-size 64 --epochs 1 --linec-batch 32 --linec-sketch-dim 8 --modes per_variant_query --top-k 0 --artifact-prefix v1223_functional_p4_official_row_scan_e1
```

1 epoch 结果：

```text
source_rows = 13
summary_rows = 13
any_p4_pass = 0
required_artifact_missing_count = 0
best row by scan order: Fashion-MNIST seed0 norm 0.007 sign +1, source=0.765625, noop=0.7578125, control=0.734375, p4_pass=0
```

3 epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_official_row_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:3 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8 --modes per_variant_query --top-k 0 --artifact-prefix v1223_functional_p4_official_row_scan_e3
```

3 epoch 结果：

```text
source_rows = 13
summary_rows = 13
any_p4_pass = 1
promotion_allowed = 0
required_artifact_missing_count = 0
```

唯一 P4 pass 行：

```text
source_dataset = KMNIST
source_seed = 1
source_actuator = I24-DirectLogitCompensatedQuadRelease
source_norm_budget = 0.03
source_signed_direction = -1
source_CouplingR2_delta = -0.022635042354497537
source_control_gap = 0.04189634323120117
source_final_acc = 0.6796875
noop_final_acc = 0.640625
control_final_acc = 0.6328125
source_vs_noop_acc_delta = +0.0390625
source_vs_control_acc_delta = +0.046875
p4_pass = 1
status = pass_audit_only_no_s5_route
```

为什么不能升 S5：v12.23 文档 14.4/14.5 写明 official P4 只允许在 P3 通过后执行。该行 CouplingR2_delta 为负，不满足 P3 hard gate 的 CouplingR2_delta >= 0.02，因此只能作为 shadow/audit evidence，不能作为 official functional success。

## 14.4 训练批 direct compensation actuator 的失败尝试

为排除 query-batch compensation 造成 P3/P4 不稳定的可能，新增并执行 train-batch compensation actuator：

```text
I26-TrainDirectLogitCompensatedQuadRelease
I27-TrainDirectLogitCompensatedShadowRelease
TrainDirectLogitCompensatedRandomControl
```

修改位置：

```text
experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

修改内容：

```text
1. 在 actuator 字典中加入 I26/I27 与 TrainDirectLogitCompensatedRandomControl。
2. 在 write_actuator_roleaware_v4 中缓存 base_train_logits = model(xb).detach()。
3. 新增 train_direct_logit_compensated_actuators 分支，用 xb/base_train_logits 做 direct_readout compensation。
4. 保持 matched controls 与 promotion gate 不放宽。
```

四 GPU 分片命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i26_traincomp_mnist_s0s2 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i26_traincomp_mnist_s0s2 --data-root data --actuator-device cuda:0 --actuator-datasets MNIST --actuator-seeds 0,1,2 --actuator-ids I26-TrainDirectLogitCompensatedQuadRelease,I27-TrainDirectLogitCompensatedShadowRelease --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8
```

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i26_traincomp_fashion_s0s2 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i26_traincomp_fashion_s0s2 --data-root data --actuator-device cuda:1 --actuator-datasets Fashion-MNIST --actuator-seeds 0,1,2 --actuator-ids I26-TrainDirectLogitCompensatedQuadRelease,I27-TrainDirectLogitCompensatedShadowRelease --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8
```

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i26_traincomp_kmnist_s0s1 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i26_traincomp_kmnist_s0s1 --data-root data --actuator-device cuda:2 --actuator-datasets KMNIST --actuator-seeds 0,1 --actuator-ids I26-TrainDirectLogitCompensatedQuadRelease,I27-TrainDirectLogitCompensatedShadowRelease --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8
```

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_i26_traincomp_kmnist_s2 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i26_traincomp_kmnist_s2 --data-root data --actuator-device cuda:3 --actuator-datasets KMNIST --actuator-seeds 2 --actuator-ids I26-TrainDirectLogitCompensatedQuadRelease,I27-TrainDirectLogitCompensatedShadowRelease --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8
```

结果汇总：

| split | route | actuator_rows | release_audit_rows | release_dataset_seed_count | role_safe_movement_rows | exploratory_release_dataset_seed_count | p4_open |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST s0-s2 | R4 | 2304 | 6 | 2 | 0 | 0 | 0 |
| Fashion s0-s2 | R4 | 2304 | 37 | 3 | 0 | 0 | 0 |
| KMNIST s0-s1 | R4 | 1536 | 16 | 2 | 0 | 0 | 0 |
| KMNIST s2 | R4 | 768 | 13 | 1 | 0 | 0 | 0 |

结论：train-batch compensation 能产生 release rows，但 role_safe_movement_rows 全部为 0，无法进入 P3/P4。

## 14.5 shadow-P4-to-P3 targeted repair

触发原因：14.3 找到一个 P4 pass 的 shadow 行，但它 CouplingR2_delta 为负，不能 official。按文档 failure matrix 的 “Functional P3 fail -> shadow-to-cloned P3 conversion + control audit” 方向，追加邻域修复，扫描同一 dataset/seed/mechanism 附近是否能找到 P3 与 P4 同时闭合的点。

命令：

```bash
conda run -n kan python experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py --run-id v1223_shadowp4_to_p3_repair_kmnist_s1_i24i25 --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --actuator-device cuda:0 --actuator-datasets KMNIST --actuator-seeds 1 --actuator-budgets 0.004,0.006,0.008,0.01,0.012,0.015,0.018,0.021,0.024,0.027,0.03,0.033,0.036,0.04 --actuator-ids I24-DirectLogitCompensatedQuadRelease,I25-DirectLogitCompensatedShadowRelease --actuator-train-size 256 --actuator-val-size 128 --actuator-linec-batch 32 --actuator-sketch-dim 8 --line-d-device cuda:1 --line-d-datasets MNIST --line-d-seeds 0 --line-d-train-size 128 --line-d-val-size 64 --line-d-batch-size 64 --line-d-epochs 1 --line-d-linec-batch 24 --line-d-sketch-dim 8
```

结果：

```text
route = S4-FunctionalP3Opened
functional_p3_pass_count = 6
p4_open = 1
p4_pass = 0
required_artifact_missing_count = 0
```

新 P3 source：

```text
source_actuator = I24-DirectLogitCompensatedQuadRelease
source_dataset = KMNIST
source_seed = 1
norm_budget = 0.027
signed_direction = +1
CouplingR2_delta = 0.0782001797938564
NoiseSignalLeak_delta = -0.05513792112469673
Reservoir_delta = -0.05831503868103027
control_gap = 0.04508481174707413
CEp99_delta = -0.762779951095581
ECE_delta = 0.0001386404037475586
```

该修复成功把 shadow-P4 邻域转成一个很强 P3 source，但还没有证明任务收益。

## 14.6 targeted repair out-dir 的 P4 short-run

1 epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_short_from_actuator.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:2 --train-size 256 --val-size 128 --batch-size 64 --epochs 1 --linec-batch 32 --linec-sketch-dim 8
```

1 epoch 结果：

```text
source_final_acc = 0.4921875
noop_final_acc = 0.5
control_final_acc = 0.5234375
source_vs_noop_acc_delta = -0.0078125
source_vs_control_acc_delta = -0.03125
p4_pass = 0
route = S4-FunctionalP3Opened
```

3 epoch 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_short_from_actuator.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8
```

3 epoch 结果：

```text
source_final_acc = 0.625
noop_final_acc = 0.6875
control_final_acc = 0.671875
source_vs_noop_acc_delta = -0.0625
source_vs_control_acc_delta = -0.046875
p4_pass = 0
route = S4-FunctionalP3Opened
```

## 14.7 targeted repair out-dir 的 P4 compensation modes 与 official row scan

P4 compensation modes 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_compensation_modes.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:0 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_compensation_modes_repair_e3
```

结果：

```text
any_p4_pass = 0
per_variant_query: source=0.6171875, noop=0.640625, control=0.640625, source-noop=-0.0234375, source-control=-0.0234375
shared_source_query: source=0.6171875, noop=0.640625, control=0.625, source-noop=-0.0234375, source-control=-0.0078125
per_variant_train: source=0.625, noop=0.640625, control=0.6328125, source-noop=-0.015625, source-control=-0.0078125
shared_source_train: source=0.625, noop=0.65625, control=0.6328125, source-noop=-0.03125, source-control=-0.0078125
none: source=0.6171875, noop=0.640625, control=0.6328125, source-noop=-0.0234375, source-control=-0.015625
```

Official row scan 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_official_row_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8 --modes per_variant_query --top-k 0 --artifact-prefix v1223_functional_p4_official_row_scan_repair_e3
```

结果：

```text
source_rows = 2
summary_rows = 2
any_p4_pass = 1
promotion_allowed = 0
```

两行对比：

| rank | source | P3 status | source acc | noop acc | control acc | p4_pass |
|---:|---|---|---:|---:|---:|---:|
| 1 | KMNIST seed1 I24 norm0.027 sign+1 | P3 pass | 0.6171875 | 0.640625 | 0.640625 | 0 |
| 2 | KMNIST seed1 I24 norm0.03 sign-1 | P3 fail, CouplingR2 negative | 0.6796875 | 0.640625 | 0.6328125 | 1 |

解释：同一 seed/mechanism 附近出现明确 tradeoff。norm0.027/+1 满足 P3 但没有短训收益；norm0.03/-1 有短训收益但 CouplingR2_delta = -0.022635042354497537，不能过 P3。

## 14.8 复现包修复与最终 zip 审计

问题：新增 P4 audit 脚本已经执行，但最初 package_zip 列表没有包含它们，导致 zip 复现包缺脚本。

先尝试 apply_patch，失败原因是本地 sandbox helper 报错：

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

由于这是可审计的小范围 package list 修复，改用已批准的 conda Python 文本替换。实际修改：

```text
experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py 的 package_zip members 增加：
experiments/run_v1223_p4_compensation_modes.py
experiments/run_v1223_p4_official_row_scan.py
```

刷新 zip 与 manifest 后的最终 zip 审计：

| out-dir | route | official_success | p4_pass | p4_official_row_scan_any_pass | required missing | entries | sha256 | scripts in zip |
|---|---|---:|---:|---:|---:|---:|---|---|
| official_explore_open2_i24_directcomp_all_targeted | S4-FunctionalP3Opened | 0 | 0 | 1 | 0 | 68 | 2ad392c3b87825ca557625d0f066905c1ca3e7f0600414d1830a1b5095c5ff35 | yes |
| official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 | S4-FunctionalP3Opened | 0 | 0 | 1 | 0 | 58 | 3e58b41d6e7b9f82acf0d7377880afaa90940854a100675a851e04756aeea5b2 | yes |

最终主要 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
sha256 = 2ad392c3b87825ca557625d0f066905c1ca3e7f0600414d1830a1b5095c5ff35
```

最终 targeted repair zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
sha256 = 3e58b41d6e7b9f82acf0d7377880afaa90940854a100675a851e04756aeea5b2
```

## 14.9 本节执行结论

本节没有达成 S5。真实推进如下：

```text
1. 原 P3 source 的 P4 fail 不是 compensation-mode 单点问题，5 种 compensation/reference 设置均未通过。
2. 13 个 official release survivors 中存在 1 个 P4 pass，但它 P3 fail，必须保持 promotion_allowed=0。
3. shadow-P4-to-P3 repair 找到新的强 P3 source：KMNIST seed1 / I24 / norm0.027 / sign+1。
4. 新 P3 source 在 1 epoch、3 epoch、5 种 compensation mode 下全部 P4 fail。
5. 同一邻域形成清晰 tradeoff：P3 pass 行无任务收益，P4 pass 行无 P3 几何资格。
6. 复现包缺新脚本的问题已修复，两个 zip 均确认包含新 P4 audit 脚本。
```


---

# 15. 继续推进：Coupling-preserving shadow/P3 blend 诊断

本节是在 14.9 之后继续推进的机制修复尝试。触发原因：14.7 已经确认同一 KMNIST seed1 / I24 邻域里，P3 pass 点与 P4 pass 点分离。因此追加一个显式 blend 诊断：把 P3-pass 的 sign+ 方向与 P4-pass 的 sign- 方向做二段式 actuator，然后统一 direct-logit compensation，测试能否得到 both P3 and P4。

## 15.1 新增诊断脚本

新增文件：

```text
experiments/run_v1223_shadowp4_coupling_preserving_blend.py
```

脚本行为：

```text
1. 基于 A1-noYForStats / KMNIST seed1 构造 base。
2. source_blend 先应用 I24 sign+ p3_budget，再应用 I24 sign- shadow_budget。
3. 对 source_blend 做 direct_readout compensation，保持 logit drift 可控。
4. matched control 使用 DirectLogitCompensatedRandomControl，控制预算为 sqrt(p3_budget^2 + shadow_budget^2)。
5. 对每个组合同时记录 P3 audit 指标与 3 epoch P4 short-run 指标。
6. promotion_allowed 固定为 0，因为这是机制诊断，不直接改 official gate。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_shadowp4_coupling_preserving_blend.py experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

结果：退出码 0。

## 15.2 blend 诊断运行命令

```bash
conda run -n kan python experiments/run_v1223_shadowp4_coupling_preserving_blend.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:2 --dataset KMNIST --seed 1 --p3-budgets 0.018,0.021,0.024,0.027,0.03 --shadow-budgets 0.006,0.012,0.018,0.024,0.03 --train-size 256 --val-size 128 --batch-size 64 --epochs 3 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_shadowp4_coupling_preserving_blend_e3
```

输出文件：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_shadowp4_coupling_preserving_blend_e3.csv
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_shadowp4_coupling_preserving_blend_e3_summary.json
```

汇总结果：

```text
summary_rows = 25
any_p3_pass = 1
any_p4_pass = 0
any_both_p3_p4_pass = 0
required_artifact_missing_count = 0
```

最佳 P3 行：

```text
p3_budget = 0.018
shadow_budget = 0.006
p3_pass = 1
p4_pass = 0
both_p3_p4_pass = 0
CouplingR2_delta = 0.02181613985160702
NoiseSignalLeak_delta_audit = -0.02925848215818405
RealSignalReservoirRatio_delta_audit = -0.11880582571029663
control_gap = 0.02241254597902298
source_final_acc = 0.6640625
noop_final_acc = 0.6875
control_final_acc = 0.6171875
source_vs_noop_acc_delta = -0.0234375
source_vs_control_acc_delta = +0.046875
```

重要解释：blend 能重新造出 P3 pass，而且 source 胜过 random control；但仍输给 NoOp，因此不满足 P4。换句话说，简单线性二段 blend 没有解决 P3/P4 解耦。

## 15.3 zip 刷新中的 pyc 缓存问题与最终修复

第一次把 blend 脚本加入 package_zip 后，zip 审计仍显示：

```text
has_blend = false
```

排查发现源文件 package list 已经包含：

```text
experiments/run_v1223_shadowp4_coupling_preserving_blend.py
```

但 import 后的 package_zip code constants 没有该项，说明 Python 使用了旧 pyc。修复命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

随后重新刷新两个 zip，最终审计：

| out-dir | entries | has_p4_modes | has_p4_official_scan | has_blend | route | official_success | p4_pass | sha256 |
|---|---:|---|---|---|---|---:|---:|---|
| official_explore_open2_i24_directcomp_all_targeted | 69 | true | true | true | S4-FunctionalP3Opened | 0 | 0 | 336b4a3d4c702214b9482f1937fb2b700adfbe2ddf32089fdb8b649f4067d15c |
| official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 | 61 | true | true | true | S4-FunctionalP3Opened | 0 | 0 | 7cf9f90a86366b9490e0996a3c068498d01e45c7110d7ea8bfbf6614ce7897de |

最终主要 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
sha256 = 336b4a3d4c702214b9482f1937fb2b700adfbe2ddf32089fdb8b649f4067d15c
```

最终 repair zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
sha256 = 7cf9f90a86366b9490e0996a3c068498d01e45c7110d7ea8bfbf6614ce7897de
```

## 15.4 本节结论

```text
1. Coupling-preserving blend 小网格没有找到 both P3/P4 pass。
2. 它可以找到 P3 pass 且胜过 random control 的点，但仍输给 NoOp。
3. 当前失败边界进一步收窄：不是单纯 norm grid 不够，也不是简单 source/shadow 线性混合不够；NoOp baseline 仍然吃掉了短训收益。
4. 复现 zip 已最终确认包含 p4_modes、p4_official_scan、shadowp4_blend 三个新增脚本。
```


# 16. 继续推进：P4 optimizer/role/trajectory gate 深挖与 zip 刷新

本节是在 15 节之后继续执行。触发原因：上一轮 blend 后仍未达成 S5，用户明确要求未达成目标不得停止。因此继续沿 18.8/19.5 推荐方向推进：不要继续盲目扩 I24/I25 norm grid，而是测试 post-P3 online update 是否能让 P3 source 不再输给 NoOp，并把 audit pass 进一步拆成 strict trajectory gate。

## 16.1 代码修改与新增脚本

新增/修改文件：

```text
experiments/run_v1223_p4_optimizer_schedule_scan.py
experiments/run_v1223_p4_trainable_role_scan.py
experiments/run_v1223_p4_temperature_calibration_scan.py
experiments/run_v1223_p4_trajectory_gate_verifier.py
experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

修改说明：

```text
1. 新增 optimizer schedule scan：同一 P3 source 下扫描 epochs/lr/weight_decay，source/noop/control 使用同一 optimizer schedule，promotion_allowed=0。
2. 新增 trainable-role scan：source/noop/control 使用同一 role-policy（all/freeze_quad/direct_only/direct_gain/direct_branch_gain/quad_only/quad_direct），测试 post-P3 online update 是否存在 NoOp-resistant task gain，promotion_allowed=0。
3. 新增 temperature calibration scan：同一 temperature 同时作用到 source/noop/control，只做校准诊断，promotion_allowed=0。
4. 新增 trajectory gate verifier：重新训练 audit pass 候选，记录 per-epoch trajectory、AUC-error-time、step-time q90、final NLL/ECE/CEp99、LineC 三项，并给出 strict_trajectory_gate_pass；route 不自动改 S5。
5. package_zip members 增加上述新增脚本，保证复现包包含本节代码。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_optimizer_schedule_scan.py experiments/run_v1223_p4_trainable_role_scan.py experiments/run_v1223_p4_temperature_calibration_scan.py experiments/run_v1223_p4_trajectory_gate_verifier.py
```

结果：退出码 0。

## 16.2 P4 optimizer schedule scan：e3/e5, lr/weight decay

主 out-dir 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_optimizer_schedule_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 256 --val-size 128 --batch-size 64 --epochs-list 3,5 --lrs 0.0005,0.001,0.002,0.004,0.008 --weight-decays 0,0.001 --linec-batch 32 --linec-sketch-dim 8 --modes per_variant_query --artifact-prefix v1223_functional_p4_optimizer_schedule_scan_main_e3e5
```

结果：

```text
summary_rows = 20
any_p4_pass = 0
best observed: epochs=5, lr=0.008, source=noop=control=0.6953125, p4_pass=0
```

repair out-dir 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_optimizer_schedule_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 256 --val-size 128 --batch-size 64 --epochs-list 3,5 --lrs 0.0005,0.001,0.002,0.004,0.008 --weight-decays 0,0.001 --linec-batch 32 --linec-sketch-dim 8 --modes per_variant_query --artifact-prefix v1223_functional_p4_optimizer_schedule_scan_repair_e3e5
```

结果：

```text
summary_rows = 20
any_p4_pass = 0
best observed: epochs=5, lr=0.0005, source=0.5234375, noop=0.53125, control=0.5546875, p4_pass=0
```

## 16.3 Trainable-role scan：e3/e5

主 out-dir 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trainable_role_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:2 --train-size 256 --val-size 128 --batch-size 64 --epochs-list 3,5 --lrs 0.001,0.002 --weight-decays 0.001 --role-policies all,freeze_quad,direct_only,direct_gain,direct_branch_gain,quad_only,quad_direct --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trainable_role_scan_main_e3e5
```

结果：

```text
summary_rows = 28
any_p4_pass = 0
best positive accuracy movement: direct_branch_gain/e5/lr0.002 source=0.421875, noop=0.390625, control=0.390625, but p4_pass=0
```

repair out-dir 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trainable_role_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 256 --val-size 128 --batch-size 64 --epochs-list 3,5 --lrs 0.001,0.002 --weight-decays 0.001 --role-policies all,freeze_quad,direct_only,direct_gain,direct_branch_gain,quad_only,quad_direct --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trainable_role_scan_repair_e3e5
```

结果：

```text
summary_rows = 28
any_p4_pass = 1
pass row = freeze_quad/e5/lr0.001, source=0.3359375, noop=0.3125, control=0.3046875, source-noop=+0.0234375, source-control=+0.03125
status = pass_audit_only_no_s5_route
promotion_allowed = 0
```

## 16.4 repair pass 的 512/256 confirmation 与 temperature calibration

512/256 confirmation 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trainable_role_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs-list 5 --lrs 0.001 --weight-decays 0.001 --role-policies freeze_quad,direct_only,direct_gain,direct_branch_gain --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trainable_role_scan_repair_confirm512_e5
```

结果：

```text
summary_rows = 4
any_p4_pass = 0
direct_only/direct_gain: source=0.359375, noop=0.32421875, control=0.32421875, but source_NLL/CEp99 worse than NoOp
direct_branch_gain: source=0.37890625, noop=0.35546875, control=0.33203125, but source_NLL/CEp99 worse than NoOp
freeze_quad: source=0.50390625, noop=0.51171875, p4_pass=0
```

同 temperature calibration 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_temperature_calibration_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs-list 5 --lrs 0.001 --weight-decays 0.001 --role-policies direct_only,direct_gain,direct_branch_gain,freeze_quad --temperatures 0.75,1.0,1.25,1.5,2.0,3.0,4.0 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_temperature_calibration_scan_confirm512_e5
```

结果：

```text
summary_rows = 28
any_p4_pass = 0
best direct_branch_gain/T=0.75: source=0.3828125, noop=0.3515625, control=0.33203125, but source_NLL=1.9193236827850342 > noop_NLL=1.906362771987915 and source_CEp99=3.38726544380188 > noop_CEp99=3.1597981452941895
```

结论：e5 的小样本 P4 pass 不稳定；同 temperature 不能修复 NLL/CEp99 blocker。

## 16.5 Trainable-role scan：e8/e12, 512/256

主 out-dir 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trainable_role_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs-list 8,12 --lrs 0.001,0.002 --weight-decays 0.001 --role-policies freeze_quad,direct_only,direct_branch_gain,quad_only --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trainable_role_scan_main_e8e12_confirm512
```

结果：

```text
summary_rows = 16
any_p4_pass = 1
pass row = quad_only/e8/lr0.002, source=0.75, noop=0.73828125, control=0.7265625, source-noop=+0.01171875, source-control=+0.0234375
status = pass_audit_only_no_s5_route
promotion_allowed = 0
```

repair out-dir 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trainable_role_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs-list 8,12 --lrs 0.001,0.002 --weight-decays 0.001 --role-policies freeze_quad,direct_only,direct_branch_gain,quad_only --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trainable_role_scan_repair_e8e12_confirm512
```

结果：

```text
summary_rows = 16
any_p4_pass = 1
pass row 1 = quad_only/e8/lr0.002, source=0.80859375, noop=0.78515625, control=0.7890625
pass row 2 = quad_only/e12/lr0.002, source=0.80859375, noop=0.7734375, control=0.78125
status = pass_audit_only_no_s5_route
promotion_allowed = 0
```

## 16.6 Strict trajectory gate verifier

第一次 verifier 使用 seed-shift 复核，目的是测试 audit pass 是否对 train shuffle / LineC sketch 稳定。

主 out-dir seed-shift 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_quadonly_e8_lr002_confirm512
```

结果：

```text
strict_trajectory_gate_pass = 0
source=0.7265625, noop=0.734375, control=0.73828125
status=strict_gate_fail_audit_only
```

repair out-dir seed-shift 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e12_lr002_confirm512
```

结果：

```text
strict_trajectory_gate_pass = 0
source=0.80078125, noop=0.78515625, control=0.7890625
fail details: source_CEp99=10.07345962524414 > noop_CEp99=8.37108039855957; step_time_ratio=1.1070675794406672; NoiseSignalLeak/Reservoir also worse than NoOp
```

随后修复 verifier 的复现口径：增加 `--train-seed-base` 与 `--linec-seed-base`，默认分别为原 trainable-role scan 使用的 `12239400` 与 `12239500`。这不是结果修饰，只是让 exact replay 与 scan 的随机种子一致。

主 out-dir exact replay 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_quadonly_e8_lr002_confirm512_exactseed_v2
```

结果：

```text
strict_trajectory_gate_pass = 1
promotion_allowed = 0
status = strict_gate_pass_audit_only_route_not_mutated
source=0.75, noop=0.73828125, control=0.7265625
source_NLL=1.2607176303863525 <= noop_NLL=1.2737209796905518
source_CEp99=10.830883026123047 <= noop_CEp99=14.015250205993652
source_ECE=0.25108960270881653 <= noop_ECE+0.02
step_time_ratio=0.9758790537510409
auc_error_time_ratio=0.9876783565783177
CouplingR2: source=0.3259254463806682 > noop=0.29090090994226603
NoiseSignalLeak: source=0.13140901923179626 < noop=0.23948727548122406
RealSignalReservoirRatio: source=0.8273053169250488 < noop=0.8910343050956726
```

repair out-dir exact replay 命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e12_lr002_confirm512_exactseed_v2
```

结果：

```text
strict_trajectory_gate_pass = 0
promotion_allowed = 0
source=0.80859375, noop=0.7734375, control=0.78125
source_NLL=0.8303794264793396 <= noop_NLL=0.9941673874855042
source_CEp99=9.003666877746582 <= noop_CEp99=10.317561149597168
source_ECE=0.19563405215740204 <= noop_ECE=0.20045700669288635
step_time_ratio=0.9808645769696466
auc_error_time_ratio=0.8903285087745244
CouplingR2: source=0.265527624440711 > noop=0.25369382153472153
NoiseSignalLeak: source=0.08920775353908539 < noop=0.11743558943271637
fail blocker: RealSignalReservoirRatio source=0.913444995880127 > noop=0.8790651559829712
```

## 16.7 最终 zip 刷新与审计

刷新前语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_optimizer_schedule_scan.py experiments/run_v1223_p4_trainable_role_scan.py experiments/run_v1223_p4_temperature_calibration_scan.py experiments/run_v1223_p4_trajectory_gate_verifier.py
```

刷新 zip 的审计结果：

| out-dir | route | official_success | p4_pass | p4_trainable_any_pass | trajectory_strict_last | promotion_allowed | required missing | entries | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| official_explore_open2_i24_directcomp_all_targeted | S4-FunctionalP3Opened | 0 | 0 | 1 | 1 | 0 | 0 | 85 | 6476520cee8efcbba167a316937538ca108b74a1ccfb8f03ae57b62d50c49a5b |
| official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 | S4-FunctionalP3Opened | 0 | 0 | 1 | 0 | 0 | 0 | 81 | 5d33fadd3e55d4395d056cfcdbc2685ffb8d492ec56fcb6c504c57409e50423f |

最终 zip 路径：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
```

两个 zip 均确认包含新增脚本：

```text
experiments/run_v1223_p4_compensation_modes.py
experiments/run_v1223_p4_official_row_scan.py
experiments/run_v1223_shadowp4_coupling_preserving_blend.py
experiments/run_v1223_p4_optimizer_schedule_scan.py
experiments/run_v1223_p4_trainable_role_scan.py
experiments/run_v1223_p4_temperature_calibration_scan.py
experiments/run_v1223_p4_trajectory_gate_verifier.py
```

## 16.8 本节执行结论

```text
1. 普通 optimizer schedule e3/e5 没有修复 P4。
2. trainable-role 的 quad_only e8/e12 出现明确 audit-only P4 pass，说明 P3 source 不是完全没有可训练任务价值。
3. 512/256 exact replay 下，主 targeted source 通过 strict trajectory gate，但仍保持 promotion_allowed=0，route 不改 S5。
4. repair source 的 e12 exact replay 任务指标更强，但 RealSignalReservoirRatio 相对 NoOp 变差，因此 strict trajectory gate fail。
5. seed-shift 复核显示该现象仍有随机性，不能写成 robust official success。
6. official route 仍是 S4-FunctionalP3Opened；S5 未正式达成。
```


# 17. 继续推进：robustness、train-stream compensation、reservoir-veto checkpoint

本节是在 16 节之后继续执行。触发原因：16 节仍未达成官方 S5；主 targeted source 只有 exact replay 的 audit strict pass，且 `promotion_allowed=0`。本节按 20.9 的推荐顺序继续推进：

```text
A. 对 main strict pass 做 train-shuffle seed robustness。
B. 设计并测试 precommit/train-stream-only direct compensation reference，替代 query-batch reference。
C. 对 repair source 做 reservoir-vetoed quad_only checkpoint 诊断。
D. 仍然保持 promotion fail-closed，不把 audit pass 写成 S5。
```

## 17.1 代码修改

修改/新增文件：

```text
experiments/run_v1223_p4_trajectory_gate_verifier.py
experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py
experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
```

修改说明：

```text
1. trajectory verifier 增加：
   --source-actuator-override
   --control-actuator-override
   --compensation-reference query|train
   --compensation-batch
   目的：复用同一 strict gate 复核逻辑，显式测试 I26/TrainDirectLogitCompensatedRandomControl 的 train-stream-only compensation。

2. 新增 reservoir-veto checkpoint scan：
   每个 epoch 同步记录 source/noop/control 的 acc/NLL/ECE/CEp99 和 LineC 三项；
   只要 source 任务收益、NLL/CEp99/ECE、CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio 同时不劣于 NoOp/control，即记为 checkpoint audit pass；
   promotion_allowed 固定为 0，因为 checkpoint 选择和 per-epoch LineC 是 audit-only。

3. package_zip members 增加：
   experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py
```

语法检查命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_trajectory_gate_verifier.py experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py experiments/run_v1223_p4_optimizer_schedule_scan.py experiments/run_v1223_p4_trainable_role_scan.py experiments/run_v1223_p4_temperature_calibration_scan.py
```

结果：退出码 0。

## 17.2 main strict pass 的 3 个 train-shuffle seed robustness

固定 main source：

```text
out-dir = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted
source = KMNIST seed0 / I24 / norm0.009 / sign+1
role_policy = quad_only
epochs = 8
lr = 0.002
linec_seed_base = 12239500
```

运行命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12240400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_quadonly_e8_lr002_trainseed12240400_linecexact
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12241400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_quadonly_e8_lr002_trainseed12241400_linecexact
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12242400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_quadonly_e8_lr002_trainseed12242400_linecexact
```

结果：

| train_seed_base | source acc | noop acc | control acc | strict pass | 主要失败 |
|---:|---:|---:|---:|---:|---|
| 12240400 | 0.75 | 0.734375 | 0.75 | 0 | source 未超过 control，且 CEp99/NLL 劣于 NoOp |
| 12241400 | 0.72265625 | 0.7421875 | 0.734375 | 0 | source accuracy 输 NoOp/control |
| 12242400 | 0.72265625 | 0.70703125 | 0.7421875 | 0 | source accuracy 输 control |

汇总：

```text
pass_count = 0 / 3
official S5 = 0
promotion_allowed = 0
```

## 17.3 precommit/train-stream-only compensation：I26 batch 32/128/256

目的：替代 I24 query-batch reference，用 train-stream-only reference 做 direct-logit compensation。source 改为：

```text
source_actuator_override = I26-TrainDirectLogitCompensatedQuadRelease
control_actuator_override = TrainDirectLogitCompensatedRandomControl
compensation_reference = train
role_policy = quad_only
epochs = 8
lr = 0.002
train_seed_base = 12239400
linec_seed_base = 12239500
```

运行命令：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --source-actuator-override I26-TrainDirectLogitCompensatedQuadRelease --control-actuator-override TrainDirectLogitCompensatedRandomControl --compensation-reference train --compensation-batch 32 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_i26_traincomp_b32_quadonly_e8_lr002_exactseed
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --source-actuator-override I26-TrainDirectLogitCompensatedQuadRelease --control-actuator-override TrainDirectLogitCompensatedRandomControl --compensation-reference train --compensation-batch 128 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_i26_traincomp_b128_quadonly_e8_lr002_exactseed
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --source-actuator-override I26-TrainDirectLogitCompensatedQuadRelease --control-actuator-override TrainDirectLogitCompensatedRandomControl --compensation-reference train --compensation-batch 256 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_main_i26_traincomp_b256_quadonly_e8_lr002_exactseed
```

结果：

| compensation_batch | source acc | noop acc | control acc | strict pass | 主要失败 |
|---:|---:|---:|---:|---:|---|
| 32 | 0.734375 | 0.73828125 | 0.75 | 0 | source 输 NoOp/control |
| 128 | 0.73046875 | 0.73828125 | 0.75 | 0 | source 输 NoOp/control |
| 256 | 0.73046875 | 0.73828125 | 0.75 | 0 | source 输 NoOp/control |

汇总：

```text
main_train_stream_compensation_any_pass = 0
结论：train-stream-only direct compensation 没有保留 query-reference 下的 task gain。
```

## 17.4 reservoir-veto checkpoint scan：repair e12

目标：repair source 在 e12 exact replay 中 task/NLL/CEp99 很强，但 `RealSignalReservoirRatio` 对 NoOp 变差。追加 per-epoch checkpoint scan，检查是否存在某个 epoch 同时满足 task 与 reservoir gate。

运行命令：

```bash
conda run -n kan python experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --compensation-reference query --train-seed-base 12239400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_reservoir_veto_repair_quadonly_e12_lr002_exactseed
```

结果：

```text
summary_rows = 12
any_reservoir_veto_pass = 1
best_epoch = 11
source_acc = 0.80859375
noop_acc = 0.77734375
control_acc = 0.78515625
source_NLL = 0.8233494758605957 <= noop_NLL = 0.9744518399238586
source_CEp99 = 8.874238967895508 <= noop_CEp99 = 10.221264839172363
source_ECE = 0.1958829164505005 <= noop_ECE = 0.1987992823123932
source_CouplingR2 = 0.27166296495386577 > noop_CouplingR2 = 0.256553534179773
source_NoiseSignalLeak = 0.16414161026477814 < noop_NoiseSignalLeak = 0.2372053563594818
source_RealSignalReservoirRatio = 0.793380856513977 < noop_RealSignalReservoirRatio = 0.8506437540054321
promotion_allowed = 0
```

解释：checkpoint scan 找到了 epoch 11 的 audit-only pass，但因为使用 per-epoch checkpoint 选择和 LineC audit，不能 promotion。

## 17.5 repair e11 exact verifier 与 LineC sketch robustness

为了检查 checkpoint pass 是否只是脚本内 LineC seed 偶然，运行独立 trajectory verifier。

默认 LineC seed：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_exactseed
```

结果：strict pass = 0，失败点是 `source_RealSignalReservoirRatio=0.9068674445152283 > noop=0.8772277235984802`。

checkpoint 相同 LineC seed：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:3 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12240600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_checkpoint_linecseed
```

结果：strict pass = 1。

追加 3 个 LineC sketch seed：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12241600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_linecseed12241600
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12242600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_linecseed12242600
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12243600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_linecseed12243600
```

LineC seed robustness 结果：

```text
pass_count = 3 / 5
failed seeds include default 12239500 and 12242600
结论：repair e11 的 LineC/reservoir 判定仍然 seed-sensitive。
```

## 17.6 repair e11 train-shuffle robustness

固定一个通过的 LineC seed `12241600`，跑 3 个 train-shuffle seed：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12240400 --linec-seed-base 12241600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_trainseed12240400_linec12241600
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12241400 --linec-seed-base 12241600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_trainseed12241400_linec12241600
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12242400 --linec-seed-base 12241600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr002_trainseed12242400_linec12241600
```

结果：

```text
pass_count = 1 / 3
12240400 fail: NoiseSignalLeak source > NoOp
12241400 fail: CouplingR2 source < NoOp and CEp99 source > NoOp+0.05
12242400 pass: source=0.796875, noop=0.77734375, control=0.7890625
```

结论：repair e11 也不是 train-shuffle robust。

## 17.7 lr=0.0015 稳定性修复尝试

为了降低 quad_only update 的波动，追加 lr=0.0015 probe：

```bash
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.0015 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr0015_exactseed_defaultlinec
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.0015 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seed-base 12241600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr0015_exactseed_linec12241600
conda run -n kan python experiments/run_v1223_p4_trajectory_gate_verifier.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.0015 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12242400 --linec-seed-base 12241600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_functional_p4_trajectory_gate_repair_quadonly_e11_lr0015_trainseed12242400_linec12241600
```

结果：

```text
pass_count = 0 / 3
lr=0.0015 没有稳定修复。
```

## 17.8 汇总 artifact 与最终 zip

生成汇总 artifact：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_continuation_p4_robustness_and_repair_summary.json
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_continuation_p4_robustness_and_repair_summary.json
```

最终 zip 刷新命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_trajectory_gate_verifier.py experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py experiments/run_v1223_p4_optimizer_schedule_scan.py experiments/run_v1223_p4_trainable_role_scan.py experiments/run_v1223_p4_temperature_calibration_scan.py
```

最终 zip 审计：

| out-dir | route | official_success | p4_pass | main train-seed pass | train-stream comp pass | repair checkpoint pass | repair LineC pass | repair train-seed pass | required missing | entries | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| official_explore_open2_i24_directcomp_all_targeted | S4-FunctionalP3Opened | 0 | 0 | 0/3 | 0 | 1 | 3/5 | 1/3 | 0 | 99 | 6ba42be4a7bc68a816b2e89e613a503bc867a2d51031fef908642e1016528206 |
| official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 | S4-FunctionalP3Opened | 0 | 0 | 0/3 | 0 | 1 | 3/5 | 1/3 | 0 | 107 | e303966aeb723de1df7973760aed5a5da1e69c36083b67a4ca33e3e081fea4b5 |

最终 zip 路径：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
```

两个 zip 均确认包含新增脚本：

```text
experiments/run_v1223_p4_trajectory_gate_verifier.py
experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py
```

## 17.9 本节结论

```text
1. main exact strict pass 不具备 train-shuffle robustness：0/3。
2. train-stream-only compensation I26 b32/b128/b256 全失败，不能替代 I24 query reference。
3. repair e11 checkpoint 出现强 audit pass，但 LineC sketch robustness 只有 3/5，train-shuffle robustness 只有 1/3。
4. lr=0.0015 稳定性修复失败：0/3。
5. S5 仍未达成；route 必须保持 S4-FunctionalP3Opened，promotion_allowed=0。
```


# 18. 继续推进：train-stream ensemble compensation 结构性修复尝试

本节是在 17 节之后继续执行。原因：单 batch train-stream compensation 失败，但仍存在一个明确结构性修复方向：用多个 train micro-batch 的 direct-readout ridge delta 做 ensemble 平均，避免单一 train reference 偶然性。

## 18.1 新增脚本

新增：

```text
experiments/run_v1223_p4_train_ensemble_compensation_scan.py
```

脚本行为：

```text
1. source 使用 I26-TrainDirectLogitCompensatedQuadRelease。
2. matched control 使用 TrainDirectLogitCompensatedRandomControl。
3. 对多个 train micro-batch 分别计算 direct_readout compensation delta。
4. 对 delta 做 mean ensemble 后一次性写入 direct_readout。
5. source/noop/control 使用同一 quad_only P4 update。
6. promotion_allowed 固定为 0。
```

编译命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_train_ensemble_compensation_scan.py experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py experiments/run_v1223_p4_trajectory_gate_verifier.py
```

结果：退出码 0。

## 18.2 运行命令

```bash
conda run -n kan python experiments/run_v1223_p4_train_ensemble_compensation_scan.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --compensation-batch 32 --ensemble-counts 2,4,8 --train-seed-base 12239400 --linec-seed-base 12239500 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_p4_train_ensemble_compensation_main_b32_counts2_4_8_e8_lr002
```

## 18.3 结果

```text
any_pass = 0
summary_rows = 3
promotion_allowed = 0
```

| ensemble_count | source acc | noop acc | control acc | strict pass | 主要失败 |
|---:|---:|---:|---:|---:|---|
| 2 | 0.73046875 | 0.73828125 | 0.75 | 0 | source 输 NoOp/control |
| 4 | 0.734375 | 0.73828125 | 0.74609375 | 0 | source 输 NoOp/control |
| 8 | 0.734375 | 0.73828125 | 0.74609375 | 0 | source 输 NoOp/control |

结论：train-stream ensemble compensation 没有恢复 I24 query-reference 的 task gain。

## 18.4 zip 刷新

最终 zip：

```text
main zip = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
main sha256 = 6ba42be4a7bc68a816b2e89e613a503bc867a2d51031fef908642e1016528206
entries = 102

repair zip = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
repair sha256 = e303966aeb723de1df7973760aed5a5da1e69c36083b67a4ca33e3e081fea4b5
entries = 108
```

两个 zip 均确认包含：

```text
experiments/run_v1223_p4_train_ensemble_compensation_scan.py
experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py
experiments/run_v1223_p4_trajectory_gate_verifier.py
```

## 18.5 本节结论

```text
1. train-stream single reference fail。
2. train-stream ensemble reference counts 2/4/8 仍 fail。
3. provenance 修复方向仍未打开。
4. S5 仍未达成，route 保持 S4-FunctionalP3Opened。
```


# 19. 继续推进：multi-sketch aggregation 与 precommit proxy checkpoint（2026-05-25）

本节继续检查 v12.23 P4 是否达成 official S5。结论先写明：没有达成，仍保持 official_success_reached=0、p4_pass=0、promotion_allowed=0。本节新增两个脚本并刷新审计 zip。

## 19.1 新增和修改代码

新增文件：
- experiments/run_v1223_p4_multisketch_aggregate_gate.py
- experiments/run_v1223_p4_precommit_proxy_checkpoint.py

修改文件：
- experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py

修改内容：
- 将 multisketch aggregate gate 脚本加入 package_zip。
- 将 precommit proxy checkpoint 脚本加入 package_zip。
- precommit proxy 脚本增加 selection_strategy 参数，支持 best_score、latest_eligible、best_after_warmup。
- precommit proxy 脚本增加 min_selection_epoch，避免初版 best_score 早停偏差。

编译命令：
conda run -n kan python -m py_compile experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py experiments/run_v1223_p4_multisketch_aggregate_gate.py experiments/run_v1223_p4_train_ensemble_compensation_scan.py experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py experiments/run_v1223_p4_trajectory_gate_verifier.py

结果：退出码 0。

修改 precommit proxy 后编译命令：
conda run -n kan python -m py_compile experiments/run_v1223_p4_precommit_proxy_checkpoint.py

结果：退出码 0。

## 19.2 multi-sketch aggregation batch32 dim8

main 命令：
conda run -n kan python experiments/run_v1223_p4_multisketch_aggregate_gate.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_p4_multisketch_main_quadonly_e8_lr002_exactseed

repair 命令：
conda run -n kan python experiments/run_v1223_p4_multisketch_aggregate_gate.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --artifact-prefix v1223_p4_multisketch_repair_quadonly_e11_lr002_exactseed

结果：
- main epoch 8：source acc 0.75，noop acc 0.73828125，control acc 0.7265625，task_gate_pass 1，LineC pass 4/5，strict_all_sketch_pass 0，strict_majority_sketch_pass 1。
- repair epoch 11：source acc 0.80859375，noop acc 0.77734375，control acc 0.78515625，task_gate_pass 1，LineC pass 3/5，strict_all_sketch_pass 0，strict_majority_sketch_pass 1。

## 19.3 precommit proxy best_score

proxy 选择只使用 unlabeled train probe 的 logit distribution，不使用 label、CE/loss、LineC 或 audit target。选完 epoch 后才执行 task 和 LineC 审计。

main 命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --artifact-prefix v1223_p4_precommit_proxy_main_quadonly_e12_lr002_exactseed

repair 命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --artifact-prefix v1223_p4_precommit_proxy_repair_quadonly_e12_lr002_exactseed

结果：
- main best_score selected_epoch 2，source 0.75390625，noop 0.76171875，control 0.734375，task_gate_pass 0，LineC pass 2/5。
- repair best_score selected_epoch 4，source 0.82421875，noop 0.80078125，control 0.78125，task_gate_pass 1，LineC pass 0/5。

结论：best_score proxy 有早停偏差。

## 19.4 precommit proxy latest_eligible min epoch 6

main 命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --selection-strategy latest_eligible --min-selection-epoch 6 --artifact-prefix v1223_p4_precommit_proxy_main_latesteligible_e12_lr002_exactseed

repair 命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --selection-strategy latest_eligible --min-selection-epoch 6 --artifact-prefix v1223_p4_precommit_proxy_repair_latesteligible_e12_lr002_exactseed

结果：
- main selected_epoch 12，source 0.73828125，noop 0.73046875，control 0.7421875，task_gate_pass 0，LineC pass 3/5。
- repair selected_epoch 12，source 0.80859375，noop 0.7734375，control 0.78125，task_gate_pass 1，LineC pass 2/5。

## 19.5 precommit proxy best_after_warmup min epoch 8

main 命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --selection-strategy best_after_warmup --min-selection-epoch 8 --artifact-prefix v1223_p4_precommit_proxy_main_bestafterwarmup8_e12_lr002_exactseed

repair 命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --selection-strategy best_after_warmup --min-selection-epoch 8 --artifact-prefix v1223_p4_precommit_proxy_repair_bestafterwarmup8_e12_lr002_exactseed

结果：
- main selected_epoch 8，source 0.75，noop 0.73828125，control 0.7265625，task_gate_pass 1，LineC pass 4/5，strict_majority_sketch_pass 1，strict_all_sketch_pass 0。
- repair selected_epoch 8，source 0.80859375，noop 0.78515625，control 0.7890625，task_gate_pass 1，LineC pass 0/5。

## 19.6 train-stream compensation provenance check

命令：
conda run -n kan python experiments/run_v1223_p4_precommit_proxy_checkpoint.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:2 --train-size 512 --val-size 256 --batch-size 64 --epochs 12 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --source-actuator-override I26-TrainDirectLogitCompensatedQuadRelease --control-actuator-override TrainDirectLogitCompensatedRandomControl --compensation-reference train --compensation-batch 32 --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 32 --linec-sketch-dim 8 --proxy-batch 128 --selection-strategy best_after_warmup --min-selection-epoch 8 --artifact-prefix v1223_p4_precommit_proxy_traincomp_bestafterwarmup8_e12_lr002_exactseed

结果：selected_epoch 8，source actuator I26，control actuator TrainDirectLogitCompensatedRandomControl，source acc 0.734375，noop acc 0.73828125，control acc 0.75，task_gate_pass 0，LineC pass 0/5。

## 19.7 larger LineC batch and sketch diagnostic

main 命令：
conda run -n kan python experiments/run_v1223_p4_multisketch_aggregate_gate.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted --data-root data --device cuda:0 --train-size 512 --val-size 256 --batch-size 64 --epochs 8 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 64 --linec-sketch-dim 24 --artifact-prefix v1223_p4_multisketch_main_quadonly_e8_lr002_batch64_dim24

repair 命令：
conda run -n kan python experiments/run_v1223_p4_multisketch_aggregate_gate.py --out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25 --data-root data --device cuda:1 --train-size 512 --val-size 256 --batch-size 64 --epochs 11 --lr 0.002 --weight-decay 0.001 --role-policy quad_only --train-seed-base 12239400 --linec-seeds 12239500,12240600,12241600,12242600,12243600 --linec-batch 64 --linec-sketch-dim 24 --artifact-prefix v1223_p4_multisketch_repair_quadonly_e11_lr002_batch64_dim24

结果：
- main batch64 dim24 epoch 8，source 0.74609375，noop 0.73828125，control 0.73046875，task_gate_pass 0，LineC pass 0/5。
- repair batch64 dim24 epoch 11，source 0.80859375，noop 0.77734375，control 0.7890625，task_gate_pass 1，LineC pass 2/5。

## 19.8 zip 刷新

刷新命令：使用 conda run -n kan python 调用 experiments.run_v1223_failclosed_explore_open2_functional_rebuild.package_zip(out)，分别刷新 main 与 repair 目录。

结果：
- main zip results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
- main sha256 f14d33cd3cf9fab2823e7d744c56a83cde3ef35036fdecfaf9b3f11f13889c39
- main entries 116
- repair zip results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
- repair sha256 b403579e714f318b36d7f7ce44fde2e1f98ae75f44078dc3bcb8d5a5f12ea4c4
- repair entries 120

两个 zip 均确认包含：
- experiments/run_v1223_p4_multisketch_aggregate_gate.py
- experiments/run_v1223_p4_precommit_proxy_checkpoint.py
- experiments/run_v1223_p4_train_ensemble_compensation_scan.py

## 19.9 本节结论

本节没有达成 v12.23 official S5。新增证据为：
- query-reference I24 在 batch32 dim8 下可达到 task pass 加 majority LineC，但不能 all-sketch。
- precommit proxy 的 best_score、latest_eligible、best_after_warmup 三种选择均不能打开 strict all-sketch。
- train-stream I26 provenance 仍失败，source 输 NoOp/control 且 LineC 0/5。
- larger LineC batch/sketch 不是简单修复，main/repair 仍不达标。
