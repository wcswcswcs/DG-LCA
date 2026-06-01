# DG-KAN v15.6 StabilityNullspaceSplitConsensusFU AllBasisAcceleration 执行日志

## 1. 关键文件

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.6_StabilityNullspaceSplitConsensusFU_AllBasisAcceleration_完整计划.md
runner = experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py
out_dir = results/v15_6_stability_nullspace_split_consensus_fu_allbasis/official_v156
line_d_out = results/v15_6_stability_nullspace_split_consensus_fu_allbasis/line_d_v156_allbasis_substrate
```

## 2. 复现命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_6_stability_nullspace_split_consensus_fu_allbasis/line_d_v156_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates D-FOU67-LowFreqIdentityResidualV5,D-FOU68-BandwiseConsensusMetricV2,D-FOU69-PhaseStableBandMixV3,D-FOU70-HighFrequencyQuarantineV2,D-FOU71-NoMaterializeLifetimeV4,D-RBF65-CompactBumpIdentityResidualV3,D-RBF66-ActiveCenterOccupancyRepairV3,D-RBF67-WidthConditionGuardV3,D-RBF68-GaussianLocalK4NoDenseMaterializationV2,D-RBF69-CenterSNRWarmupV2,D-WAV57-TriangularSupportV5,D-WAV58-ScaleOccupancyHardeningV3,D-WAV59-SupportOverlapDampingV3,D-WAV60-LocalTailCoverageAuditV2

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py --out-dir results/v15_6_stability_nullspace_split_consensus_fu_allbasis/official_v156 --line-d-out results/v15_6_stability_nullspace_split_consensus_fu_allbasis/line_d_v156_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py --out-dir results/v15_6_stability_nullspace_split_consensus_fu_allbasis/official_v156 --line-d-out results/v15_6_stability_nullspace_split_consensus_fu_allbasis/line_d_v156_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

## 2.1 过程修正

```text
自查发现初版 v15.6 official run 把 minibatch generator 与 method 名字绑定，
会让 N0/NQ 的比较混入 batch stochasticity。
已修正为 dataset/seed/family matched batch stream，random direction 使用独立 method-specific generator。
v15.6 实现从 v15.5 runner 派生，但已将核心语义改为 N/Q hazard-nullspace 与 source-retaining projection。
Line U source-hazard decomposition 只做 readback，Line B proxy/leaveout 只做 audit。
计划要求 11 张指定 figures、Line P P1..P7 taxonomy、D-FOU67..71/D-RBF65..69/D-WAV57..60 surface 已纳入 manifest。
用户再次追问后修复 Q 分支 telemetry：source_retention_after_null 改为实际提交 update 的 retention，full-null retention 单独保留。
同次修复 Line U readback：N0/G7R 不再用空 method-specific basis，而用 combined-hazard basis 读取 source-hazard overlap。
如发生修复，会用 --reuse-if-present 0 重跑 official GPU 矩阵；finalizer-only 修复才用 --reuse-if-present 1。
```

## 3. 结果摘要

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StabilityNullspaceObservable
line_s_gate_pass = 1
line_b_gate_pass = 0
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.46147918701171875
bad_event_fraction = 1.0
generic_metric_stability_explains = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
promotion_allowed = 0
```

## 4. 覆盖反方复核

复核/修复指令：

```bash
rg -n "FIGURES|fig_v156|fig_line|fig_g7|fig_bad|fig_source|fig_controls|fig_micro|fig_allbasis|fig_transfer|fig_route" experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py
rg -n "P1-|P2-|P3-|P4-|P5-|P6-|P-B|failure_class" experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py docs/DG-KAN_v15.6_StabilityNullspaceSplitConsensusFU_AllBasisAcceleration_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v156_stability_nullspace_split_consensus_fu_allbasis.py --out-dir results/v15_6_stability_nullspace_split_consensus_fu_allbasis/official_v156 --line-d-out results/v15_6_stability_nullspace_split_consensus_fu_allbasis/line_d_v156_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

复核结果：

```text
figures_present = 11 / 11
required_artifact_manifest_rows = 43, missing = 0
Line P failure classes = P3-MicroHorizonGoodRealBad=34, P4-ControlEquivalentPath=11
positive-looking rows matched controls missing = 0
Q full-null retention failed rows = 40
Q source-retention repaired rows = 40
Line U source_hazard_colinear_rows = 63 / 63
Line U mean_hazard_overlap = 0.956108654302264
Line U mean_source_retention_after_null = 0.04575427500383249
route = R4-LineCOrTailDominated
promotion_allowed = 0
```

结论：

```text
v15.6 没有达成 S2/S3/S4/S5，也没有 promotion。
required figures、Line P taxonomy、positive-looking matched controls 均通过覆盖复核。
当前 v15.6 内没有仍可合法补跑的分支；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。
```
