# DG-KAN v12.22 FailClosedExploreOpen FunctionalRebuild 执行日志

本日志记录 v12.22 的实际执行过程、命令、修复点和落盘文件。所有训练/审计结果必须来自本机真实命令输出与 artifact；不手填训练指标，不用缺失数据补结论。

## 0. 执行环境

- 工作目录：`/home/chengshun.wang/DG-LCA`
- Python 环境：`conda env kan`
- 计划文档：`docs/DG-KAN_v12.22_FailClosedExploreOpen_FunctionalRebuild_实验结果分析与下一步计划.md`
- 主结果目录：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open`
- Line A 结果目录：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/linea`
- 目标 zip：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_code_review_packet.zip`

GPU 检查命令：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

实际输出显示 GPU 0/1/2/3 均可用，型号均为 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，初始显存占用为 0 MiB。

## 1. 计划理解回读

v12.22 的核心修正是：promotion 仍然 fail-closed，但 exploration 不能 fail-fast。v12.21 把 no-go/next hypothesis 写完就允许 final stop，这一轮必须改成 queue/child-run/budget 驱动：失败线要么执行下一层 child/fallback，要么明确记录预算权衡，不能把“写了假设”当成“执行了实验”。

## 2. 代码修改记录

### 2.1 `experiments/run_v1218_b320_label_free_ablation.py`

新增 A36-A42：

- A36 `ResidualA1LowRankFrame-labelFree`
- A37 `ConvexMultiFrameMixture-labelFree`
- A38 `PersistentDriftCotangentBank-labelFree`
- A39 `RoleConditionedFrame-labelFree`
- A40 `SelfConditionedStopGradFrame-labelFree`
- A41 `UnlabeledFrameAdaptSchedule-labelFree`
- A42 `SmallLabelOracleUpperBound-diagnostic`

新增 T1B temporal / transition / spectrum 字段：

- `t1b_update_half_life_step`
- `t1b_update_half_life_frac`
- `t1b_update_cosine_transition_abs_mean`
- `t1b_update_role_energy_transition_l1`
- `t1b_update_quad_direct_energy_flow_mean`
- `t1b_update_spectrum_entropy_mean`
- `t1b_update_spectrum_entropy_delta`

### 2.2 `dgkan/models/fc_purekan_primitives.py`

新增 v12.22 label-free frame token：

- `reslowrankp`
- `convexmixp`
- `driftcotbankp`
- `rolecondp`
- `selfcondstopgradp`

修复：A37 初次实例化 smoke 出现非有限 projector/logit。按计划中的 NaN 处理方向，修复 `_orthogonal_fill()` 与 convex frame normalize，使 frame 中的 NaN/Inf 被 fail-safe 到有限值，并在残差退化时回退到随机/坐标基向量。

### 2.3 `experiments/run_v1222_failclosed_explore_open_functional_rebuild.py`

新增 v12.22 汇总 runner，负责：

- `v1222_exploration_queue.json`
- `v1222_child_run_manifest.csv`
- `v1222_fallback_execution_manifest.csv`
- `v1222_budget_accounting.csv`
- `v1222_final_stop_audit.json`
- v12.22 Line A/C/T/I/B/D artifact contract
- shadow P4 capacity
- no-go / next hypothesis / implementation readback
- figures / hash manifest / code review zip

关键语义：`next_hypothesis_generator_written` 不再授予 `final_stop_allowed`。

## 3. 语法与实例化检查

语法检查命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_label_free_ablation.py \
  experiments/run_v1222_failclosed_explore_open_functional_rebuild.py \
  dgkan/models/fc_purekan_primitives.py
```

结果：通过。

A36-A42 CUDA 实例化 smoke 命令：

```bash
conda run -n kan python -c "import torch; from experiments import run_v1218_b320_label_free_ablation as r; device=torch.device('cuda:0'); x=torch.randn(64,784,device=device); y=torch.arange(64,device=device)%10; specs=r.ablation_specs(784,10); ids=['A36-ResidualA1LowRankFrame-labelFree','A37-ConvexMultiFrameMixture-labelFree','A38-PersistentDriftCotangentBank-labelFree','A39-RoleConditionedFrame-labelFree','A40-SelfConditionedStopGradFrame-labelFree','A41-UnlabeledFrameAdaptSchedule-labelFree','A42-SmallLabelOracleUpperBound-diagnostic']; \
for cid in ids: \
 item=specs[cid]; m=r.make_model(cid,item['spec'],784,10,x,y,device,123,int(item['uses_y_for_stats']),str(item.get('y_stats_mode','actual'))); out=m(x[:8]); print(cid, bool(torch.isfinite(out).all()), bool(torch.isfinite(m.quad_proj).all()), bool(torch.isfinite(m.quad_feature_std).all()))"
```

结果：A36-A42 的 forward、`quad_proj`、`quad_feature_std` 均为 finite。A37 的非有限 blocker 已通过上述 frame 修复解决。

## 4. Line A 并行执行命令

统一参数：

```bash
--result-stage V1222_LINEA
--summary-stage V1222_LINEA_SUMMARY
--route-stage V1222_LINEA_ROUTE
--result-scope v1222_failclosed_explore_open_e3_seed012
--smoke-not-official 0
--official-training-result-available 1
--protocol-note 'v12.22 fail-closed explore-open A36-A42 queue child run'
--route-impact 'feeds v12.22 exploration queue; no standalone promotion'
--ablation-ids A0-labelInit,A1-noYForStats,A36-ResidualA1LowRankFrame-labelFree,A37-ConvexMultiFrameMixture-labelFree,A38-PersistentDriftCotangentBank-labelFree,A39-RoleConditionedFrame-labelFree,A40-SelfConditionedStopGradFrame-labelFree,A41-UnlabeledFrameAdaptSchedule-labelFree,A42-SmallLabelOracleUpperBound-diagnostic
--train-size 512 --val-size 256 --test-size 256
--batch-size 128 --epochs 3
--measure-linec 1 --linec-batch-size 32 --linec-sketch-batch-size 8 --linec-sketch-dim 8
--task-compile-warmup-steps 1
--no-download
```

MNIST / GPU0：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1222_linea_e3_MNIST_seed012 \
  --out-dir results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/linea/e3_MNIST \
  --artifact-prefix v1222_linea_e3_MNIST \
  --device cuda:0 --datasets MNIST --seeds 0,1,2 [统一参数]
```

Fashion-MNIST / GPU1：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1222_linea_e3_FashionMNIST_seed012 \
  --out-dir results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/linea/e3_FashionMNIST \
  --artifact-prefix v1222_linea_e3_FashionMNIST \
  --device cuda:1 --datasets Fashion-MNIST --seeds 0,1,2 [统一参数]
```

KMNIST / GPU2：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1222_linea_e3_KMNIST_seed012 \
  --out-dir results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/linea/e3_KMNIST \
  --artifact-prefix v1222_linea_e3_KMNIST \
  --device cuda:2 --datasets KMNIST --seeds 0,1,2 [统一参数]
```

实际 Line A route 输出：

```text
MNIST: rows=33, summary_rows=11, best_label_free_candidate=A41-UnlabeledFrameAdaptSchedule-labelFree, best_label_free_mean_delta_vs_A0=-0.020833333333333332
Fashion-MNIST: rows=33, summary_rows=11, best_label_free_candidate=A38-PersistentDriftCotangentBank-labelFree, best_label_free_mean_delta_vs_A0=-0.013020833333333334
KMNIST: rows=33, summary_rows=11, best_label_free_candidate=A41-UnlabeledFrameAdaptSchedule-labelFree, best_label_free_mean_delta_vs_A0=-0.03125
```

## 5. v12.22 aggregate / C-T-I-B-D / zip

第一次 aggregate 命令：

```bash
conda run -n kan python experiments/run_v1222_failclosed_explore_open_functional_rebuild.py \
  --run-id official_explore_open_e3_seed012 \
  --out-dir results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open \
  --linea-root results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/linea \
  --linea-datasets MNIST,Fashion-MNIST,KMNIST \
  --linea-seeds 0,1,2 \
  --linea-epochs 3 \
  --linea-train-size 512 --linea-val-size 256 --linea-test-size 256 --linea-batch-size 128 \
  --actuator-device cuda:3 \
  --actuator-datasets MNIST,Fashion-MNIST,KMNIST \
  --actuator-seeds 0,1,2 \
  --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 \
  --actuator-train-size 128 \
  --actuator-val-size 64 \
  --actuator-linec-batch 16 \
  --actuator-sketch-dim 4 \
  --no-download
```

运行结果：成功落盘 1080 actuator rows，route 为 `R2-LabelFreeSignalFrameMissing`。

### 5.1 Manifest 审计修复

核验时发现 Line D 在 `child_run_manifest` 里容易被误读为 classic candidate 已执行。真实情况是：本轮没有执行 classic family smoke，只按 v12.22 P5 的允许路径记录了 per-family `NotExecuted_HypothesisGenerated` 和预算权衡。

修复：

- 修改 `experiments/run_v1222_failclosed_explore_open_functional_rebuild.py` 的 `write_execution_contract()`。
- Line D 现在记录：
  - `executed_this_version=0`
  - `deferred_with_budget_tradeoff=1`
  - `terminal_for_queue=1`
  - `status=deferred_with_budget_tradeoff`
- `v1222_budget_accounting.csv` 增加：
  - `candidate_executed_jobs`
  - `deferred_with_budget_jobs`

修复后重跑命令：

```bash
conda run -n kan python experiments/run_v1222_failclosed_explore_open_functional_rebuild.py \
  --run-id official_explore_open_e3_seed012_manifest_auditfix \
  --out-dir results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open \
  --linea-root results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/linea \
  --linea-datasets MNIST,Fashion-MNIST,KMNIST \
  --linea-seeds 0,1,2 \
  --linea-epochs 3 \
  --linea-train-size 512 --linea-val-size 256 --linea-test-size 256 --linea-batch-size 128 \
  --actuator-device cuda:3 \
  --actuator-datasets MNIST,Fashion-MNIST,KMNIST \
  --actuator-seeds 0,1,2 \
  --actuator-budgets 0.0005,0.001,0.0025,0.005,0.01,0.02 \
  --actuator-train-size 128 \
  --actuator-val-size 64 \
  --actuator-linec-batch 16 \
  --actuator-sketch-dim 4 \
  --no-download
```

最终 route 摘要：

```json
{
  "route": "R2-LabelFreeSignalFrameMissing",
  "minimum_success": "Minimum Success F",
  "final_stop_allowed": 1,
  "success_f_no_failfast": 1,
  "required_artifact_missing_count": 0,
  "fallback_missing_count": 0,
  "exploration_budget_exhausted": 1,
  "child_run_executed_count": 7,
  "child_run_deferred_with_budget_count": 1,
  "code_review_packet_zip_sha256": "710cab38a4514f2d08612912ea74d9c531abb9ec264fc6fa5acdf4413d5da03e",
  "code_review_packet_zip_entries": 56
}
```

Line D child manifest 核验：

```text
D-classic-status-budget Line D executed_this_version=0 deferred_with_budget_tradeoff=1 terminal_for_queue=1 status=deferred_with_budget_tradeoff
```

## 6. 最终核验命令

```bash
conda run -n kan python -c "import csv,json,zipfile; from pathlib import Path; out=Path('results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open'); route=json.loads((out/'v1222_route_decision.json').read_text()); print(json.dumps({k:route[k] for k in ['route','minimum_success','final_stop_allowed','success_f_no_failfast','required_artifact_missing_count','fallback_missing_count','exploration_budget_exhausted','child_run_executed_count','child_run_deferred_with_budget_count','code_review_packet_zip_sha256','code_review_packet_zip_entries']}, indent=2));"
```

结果：见上方最终 route 摘要。

```bash
find results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open -maxdepth 2 -type f | sort
```

结果：`v1222_*` CSV/JSON/MD、15 个 `fig_v1222_*.svg`、`v1222_code_review_packet.zip` 和 `v1222_hash_manifest.json` 均已落盘。

## 7. 最终关键文件

- route：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_route_decision.json`
- final stop audit：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_final_stop_audit.json`
- queue：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_exploration_queue.json`
- child manifest：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_child_run_manifest.csv`
- fallback manifest：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_fallback_execution_manifest.csv`
- budget accounting：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_budget_accounting.csv`
- Line A：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_label_free_signal_frame.csv`
- Line C/T：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_visibility_scores.csv`
- Line I：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_actuator_safety.csv`
- Line B shadow：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_shadow_p4_capacity.csv`
- Line D：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_classic_family_status.csv`
- zip：`results/v12_22_failclosed_explore_open_functional_rebuild/official_explore_open/v1222_code_review_packet.zip`
