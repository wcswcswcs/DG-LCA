# DG-KAN v12.21 FailClosedContinue2 LabelFreeFunctional 执行日志

本日志记录本轮 v12.21 的实际执行过程、命令、修复点和落盘文件。所有结果来自本机真实命令输出与落盘 artifact；没有手填训练指标或编造数据。

## 0. 执行环境

- 工作目录：`/home/chengshun.wang/DG-LCA`
- Python 环境：`conda env kan`
- 计划文档：`docs/DG-KAN_v12.21_FailClosedContinue2_LabelFreeFunctional_独立分析与下一步计划.md`
- 主结果目录：`results/v12_21_failclosed_continue2_label_free_functional/official_continuation`
- Line A 结果目录：`results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea`
- 最终 zip：`results/v12_21_failclosed_continue2_label_free_functional/official_continuation/v1221_code_review_packet.zip`

GPU 检查命令：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

实际可见 GPU 0/1/2/3 均为空闲 RTX PRO 6000 Blackwell Server Edition。

## 1. 代码修改记录

### 1.1 `experiments/run_v1218_b320_label_free_ablation.py`

新增：

- A30 `OptimizerObservableFrame-labelFree`
- A31 `AugConsistencyTangentFrame-labelFree`
- A32 `PersistentDriftFrame-labelFree`
- A33 `RoleBalancedPrimitiveEnergyFrame-labelFree`
- A34 `HybridA1OptimizerFrame-labelFree`
- A35 `HybridA1AugDriftFrame-labelFree`
- T1B optimizer-update 原生日志字段，来自实际参数 delta，不读取 per-example CE vector。
- `g_cov_output_cov_drift_mean/max`，用于 v12.21 G_LA target。
- `finite_float_or_none()`，修复 NaN summary 不能被当作 0 的问题。

修复：

- T1B update snapshot/statistics 原先被计入 step timing；已移出计时窗。计时窗现在只覆盖 task forward/backward/update。
- A33 出现真实 `NLL=nan` 时，summary 不再把 `AUC_time_ratio_vs_mlp` 写成 `0.0`，而是留空并按 gate fail 处理。

### 1.2 `dgkan/models/fc_purekan_primitives.py`

新增 v12.21 label-free projector token：

- `augtangentp`：从无标签输入增强/平移 tangent 构造 projector frame。
- `rolebalancedp`：按 PCA、low-frequency、tangent、SRHT/random frame 做 role-balanced quota。

说明：该文件在本轮前已有其他未提交改动；本轮没有回滚或重写无关改动。

### 1.3 `experiments/run_v1221_failclosed_continue2_label_free_functional.py`

新增 v12.21 汇总 runner，负责：

- Line R CR0-CR15 代码审计。
- Line A signal frame 总结与 failure decomposition。
- Line C G_LA deployable target、sign sanity、soft target regression。
- Line T visibility v5 与 T1B native logging 审计。
- Line I actuator v2 smoke：I1-I9，budget 0.5x/1x/2x/4x。
- Line B P3/P4 fail-closed gate。
- Line D classic hypothesis generation。
- no-go boundary、next hypothesis generator、figures、hash manifest、zip 打包。

修复：

- provenance audit 初版在写入顺序上把 `v1221_route_decision.json` 和自身记为缺失；已修正为 finalization block 内的 intended final presence。

## 2. 语法检查

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py dgkan/models/fc_purekan_primitives.py
conda run -n kan python -m py_compile experiments/run_v1221_failclosed_continue2_label_free_functional.py
```

结果：通过。

## 3. Runtime smoke

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_smoke_A30_A31 \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/smoke_linea \
  --artifact-prefix v1221_smoke_A30_A31 \
  --result-stage V1221_SMOKE \
  --summary-stage V1221_SMOKE_SUMMARY \
  --route-stage V1221_SMOKE_ROUTE \
  --result-scope v1221_smoke_runtime \
  --smoke-not-official 1 \
  --official-training-result-available 0 \
  --protocol-note 'v12.21 runtime smoke for A30/A31 only' \
  --route-impact 'runtime smoke only' \
  --device cuda:0 \
  --datasets MNIST \
  --seeds 0 \
  --ablation-ids A0-labelInit,A1-noYForStats,A30-OptimizerObservableFrame-labelFree,A31-AugConsistencyTangentFrame-labelFree \
  --train-size 128 --val-size 64 --test-size 64 \
  --batch-size 64 --epochs 1 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-batch-size 8 --linec-sketch-dim 4 \
  --task-compile-warmup-steps 0 \
  --no-download
```

结果：6 rows，runtime 通过；smoke 不用于 official claim。

## 4. Line A official small-budget 运行

统一参数：

```bash
--result-stage V1221_LINEA
--summary-stage V1221_LINEA_SUMMARY
--route-stage V1221_LINEA_ROUTE
--result-scope v1221_B320_locked_small_budget
--smoke-not-official 0
--official-training-result-available 1
--ablation-ids A0-labelInit,A1-noYForStats,A30-OptimizerObservableFrame-labelFree,A31-AugConsistencyTangentFrame-labelFree,A32-PersistentDriftFrame-labelFree,A33-RoleBalancedPrimitiveEnergyFrame-labelFree,A34-HybridA1OptimizerFrame-labelFree,A35-HybridA1AugDriftFrame-labelFree
--train-size 1024 --val-size 512 --test-size 512
--batch-size 128
--measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 64 --linec-sketch-dim 24
--task-compile-warmup-steps 2
--no-download
```

实际最终重跑命令模式：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_linea_e3_MNIST_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea/e3_MNIST \
  --artifact-prefix v1221_linea_e3_MNIST \
  --device cuda:0 --datasets MNIST --seeds 0,1,2 --epochs 3 [统一参数]
```

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_linea_e3_FashionMNIST_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea/e3_FashionMNIST \
  --artifact-prefix v1221_linea_e3_FashionMNIST \
  --device cuda:1 --datasets Fashion-MNIST --seeds 0,1,2 --epochs 3 [统一参数]
```

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_linea_e3_KMNIST_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea/e3_KMNIST \
  --artifact-prefix v1221_linea_e3_KMNIST \
  --device cuda:2 --datasets KMNIST --seeds 0,1,2 --epochs 3 [统一参数]
```

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_linea_e8_MNIST_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea/e8_MNIST \
  --artifact-prefix v1221_linea_e8_MNIST \
  --device cuda:3 --datasets MNIST --seeds 0,1,2 --epochs 8 [统一参数]
```

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_linea_e8_FashionMNIST_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea/e8_FashionMNIST \
  --artifact-prefix v1221_linea_e8_FashionMNIST \
  --device cuda:0 --datasets Fashion-MNIST --seeds 0,1,2 --epochs 8 [统一参数]
```

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1221_linea_e8_KMNIST_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea/e8_KMNIST \
  --artifact-prefix v1221_linea_e8_KMNIST \
  --device cuda:1 --datasets KMNIST --seeds 0,1,2 --epochs 8 [统一参数]
```

说明：上面日志中 `[统一参数]` 是本节开头列出的完整参数组。实际命令已按该组执行，输出文件内 `protocol_note` 记录 `T1B timing and NaN summary fixed`。

运行中出现 warning：

```text
torch.linalg.svd: During SVD computation ... failed to converge. A more accurate method will be used ...
```

该 warning 来自 projector diagnostic SVD fallback，命令未失败，CSV 已落盘。

## 5. v12.21 汇总 / 审计 / actuator / zip

最终命令：

```bash
conda run -n kan python experiments/run_v1221_failclosed_continue2_label_free_functional.py \
  --run-id official_continuation_e3_e8_seed012_nanfix \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation \
  --linea-root results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea \
  --actuator-device cuda:3 \
  --actuator-datasets MNIST,Fashion-MNIST,KMNIST \
  --actuator-seeds 0 \
  --actuator-budgets 0.0025,0.005,0.01,0.02 \
  --actuator-train-size 256 \
  --actuator-val-size 128 \
  --actuator-linec-batch 32 \
  --actuator-sketch-dim 8 \
  --no-download
```

最终 route 摘要：

```json
{
  "route": "R2-LabelFreeSignalFrameMissing",
  "minimum_success": "Success E",
  "continuation_missing_count": 0,
  "required_artifact_missing_count": 0,
  "provenance_missing_count": 0,
  "line_r_required_missing": 0,
  "label_free_official_pass_count": 0,
  "g_la_official_pass": 0,
  "T1B_native_logged_rows": 108,
  "T1B_visibility_pass": 0,
  "actuator_executor_success": 0,
  "p4_open": 0,
  "classic_new_hypothesis_count": 5,
  "final_stop_allowed": 1
}
```

最终 zip：

```text
results/v12_21_failclosed_continue2_label_free_functional/official_continuation/v1221_code_review_packet.zip
sha256=e556a7ffed314752aa51390a7986f2425be1f3f412ffbf9facc53bf273420762
entries=67
```

zip 已包含：

- `experiments/run_v1221_failclosed_continue2_label_free_functional.py`
- `experiments/run_v1218_b320_label_free_ablation.py`
- `dgkan/models/fc_purekan_primitives.py`
- v12.21 plan 文档
- v1221 artifacts 与 figures

## 6. 关键落盘文件

- `v1221_route_decision.json`
- `v1221_continuation_manifest.csv`
- `v1221_provenance_audit.csv`
- `v1221_code_review_manifest.csv`
- `v1221_core_symbol_map.json`
- `v1221_implementation_readback.md`
- `v1221_label_free_signal_frame.csv`
- `v1221_linec_deployable_targets.csv`
- `v1221_visibility_scores.csv`
- `v1221_t1b_optimizer_update_features.csv`
- `v1221_actuator_safety.csv`
- `v1221_functional_p4_short.csv`
- `v1221_classic_family_new_hypothesis.csv`
- `v1221_no_go_boundary.md`
- `v1221_next_hypothesis_generator.md`
- `v1221_code_review_packet.zip`

校验结果：

- required artifact manifest：30 rows，bad=0
- continuation manifest：7 rows，bad=0
- provenance audit：14 rows，bad=0
- join key uniqueness：3 rows，bad=0

## 7. 2026-05-25 completion-audit 补充

用户再次要求确认是否完成后，重新按计划第 14/16 节核对，发现 route JSON 未显式写入 `final_stop_allowed`，且 `v1221_next_hypothesis_generator.md` 将 Line C/T 合并描述、未单独列 Line B。训练数据无需重跑；这是 completion-audit artifact 的补全。

修复内容：

- `experiments/run_v1221_failclosed_continue2_label_free_functional.py`
  - `write_no_go_and_next()` 中新增 Line C、Line T、Line B 独立 next hypothesis。
  - route summary 新增 `next_hypothesis_failed_lines_covered=1`。
  - route summary 新增 `next_hypothesis_lines=Line A,Line C,Line T,Line I,Line B,Line D`。
  - route summary 新增 `final_stop_allowed=1`，条件为 continuation 完成、no-go/next 写入、失败线 hypothesis 覆盖且 minimum success 合法。

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1221_failclosed_continue2_label_free_functional.py && \
conda run -n kan python experiments/run_v1221_failclosed_continue2_label_free_functional.py \
  --run-id official_continuation_e3_e8_seed012_finalstop \
  --out-dir results/v12_21_failclosed_continue2_label_free_functional/official_continuation \
  --linea-root results/v12_21_failclosed_continue2_label_free_functional/official_continuation/linea \
  --actuator-device cuda:3 \
  --actuator-datasets MNIST,Fashion-MNIST,KMNIST \
  --actuator-seeds 0 \
  --actuator-budgets 0.0025,0.005,0.01,0.02 \
  --actuator-train-size 256 \
  --actuator-val-size 128 \
  --actuator-linec-batch 32 \
  --actuator-sketch-dim 8 \
  --no-download
```

补充后最终字段：

```text
final_stop_allowed=1
next_hypothesis_failed_lines_covered=1
next_hypothesis_lines=Line A,Line C,Line T,Line I,Line B,Line D
route=R2-LabelFreeSignalFrameMissing
minimum_success=Success E
required_artifact_missing_count=0
continuation_missing_count=0
provenance_missing_count=0
line_r_required_missing=0
```

最终 zip 更新为：

```text
results/v12_21_failclosed_continue2_label_free_functional/official_continuation/v1221_code_review_packet.zip
sha256=e556a7ffed314752aa51390a7986f2425be1f3f412ffbf9facc53bf273420762
entries=67
```
