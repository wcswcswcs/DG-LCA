# DG-KAN v15.5 SplitConsensusMetricStability AllBasisAcceleration 执行日志

## 1. 关键文件

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.5_SplitConsensusMetricStability_AllBasisAcceleration_完整计划.md
runner = experiments/run_v155_split_consensus_metric_stability_allbasis.py
out_dir = results/v15_5_split_consensus_metric_stability_allbasis/official_v155
line_d_out = results/v15_5_split_consensus_metric_stability_allbasis/line_d_v155_allbasis_substrate
```

## 2. 复现命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v155_split_consensus_metric_stability_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_5_split_consensus_metric_stability_allbasis/line_d_v155_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates D-FOU62-LowFreqIdentityResidualV5,D-FOU63-BandwiseConsensusMetric,D-FOU64-PhaseStableBandMixV2,D-FOU65-NoMaterializeLifetimeV3,D-FOU66-HighFrequencyQuarantineNoAuditDirection,D-RBF60-CompactBumpIdentityResidualV2,D-RBF61-ActiveCenterConsensusOccupancy,D-RBF62-WidthConditionGuardV2,D-RBF63-GaussianLocalK4NoDenseMaterialization,D-RBF64-CenterDropoutNoTaskBranchDiagnostic,D-WAV53-TriangularSupportV5,D-WAV54-ScaleOccupancyConsensus,D-WAV55-SupportOverlapDampingV2,D-WAV56-LocalTailCoverageAuditOnly

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v155_split_consensus_metric_stability_allbasis.py --out-dir results/v15_5_split_consensus_metric_stability_allbasis/official_v155 --line-d-out results/v15_5_split_consensus_metric_stability_allbasis/line_d_v155_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v155_split_consensus_metric_stability_allbasis.py --out-dir results/v15_5_split_consensus_metric_stability_allbasis/official_v155 --line-d-out results/v15_5_split_consensus_metric_stability_allbasis/line_d_v155_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

## 2.1 过程修正

```text
自查发现初版 v15.5 official run 把 minibatch generator 与 method 名字绑定，
会让 G7R/G7S 的比较混入 batch stochasticity。
已修正为 dataset/seed/family matched batch stream，random direction 使用独立 method-specific generator。
同时修正 G7R replay 的 trust_scalar_applied 审计字段：G7R 为 0，G7S/C6/M4 为 1。
修正后使用 --reuse-if-present 0 重跑 official GPU 矩阵，不沿用旧训练指标。
再次反方复核发现计划第 15 节要求 11 张指定 figures，初版 runner 只生成 6 张内部命名 figures。
已修正 FIGURES / write_figures，并用 --reuse-if-present 1 + --device cuda:0 重写 finalizer/docs/manifest。
继续反方复核发现 Line P failure taxonomy 使用内部名 P-B3/P-B5/P-OK；已改为计划第 11.4 节 P1..P6 taxonomy，并保留 P0-OK 表示未触发失败类。
这些修复只补 artifact coverage / taxonomy / 日志，不新增训练、不改变训练指标。
```

## 3. 结果摘要

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StableSplitConsensusSubspaceObservable
line_s_gate_pass = 1
line_b_gate_pass = 0
real_lite_pass_count = 1 / 9
source_vs_best_control_mean = 0.46147918701171875
bad_event_fraction = 1.0
generic_metric_stability_explains = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
promotion_allowed = 0
```

## 4. 用户再次追问后的反方复核

复核/修复指令：

```bash
rg -n "FIGURES|fig_v155|fig_line|fig_g7|fig_bad|fig_source|fig_controls|fig_micro|fig_allbasis|fig_transfer|fig_route" experiments/run_v155_split_consensus_metric_stability_allbasis.py
rg -n "P1-|P2-|P3-|P4-|P5-|P6-|P-B|failure_class" experiments/run_v155_split_consensus_metric_stability_allbasis.py docs/DG-KAN_v15.5_SplitConsensusMetricStability_AllBasisAcceleration_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v155_split_consensus_metric_stability_allbasis.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v155_split_consensus_metric_stability_allbasis.py --out-dir results/v15_5_split_consensus_metric_stability_allbasis/official_v155 --line-d-out results/v15_5_split_consensus_metric_stability_allbasis/line_d_v155_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

复核结果：

```text
figures_present = 11 / 11
required_artifact_manifest_rows = 42, missing = 0
Line P failure classes = P5-ControlEquivalentPath=27, P3-TailProxySpike=9
positive-looking rows matched controls missing = 0
route = R4-LineCOrTailDominated
promotion_allowed = 0
```

结论：

```text
v15.5 没有达成 S2/S3/S4/S5，也没有 promotion。
已修复 figures 覆盖缺口与 Line P taxonomy 命名缺口。
当前 v15.5 内没有仍可合法补跑的分支；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。
```
