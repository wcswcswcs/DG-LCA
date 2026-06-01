# DG-KAN v15.7 SourceHazardFactorization CarrierReparam AllBasisAcceleration 执行日志

## 1. 关键文件

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.7_SourceHazardFactorization_CarrierReparam_AllBasisAcceleration_完整计划.md
runner = experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py
out_dir = results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/official_v157
line_d_out = results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate
```

## 2. 复现命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates D-FOU72-LowFreqIdentityResidualV5,D-FOU73-BandwiseConsensusMetricV2,D-FOU74-PhaseStableBandMixV2,D-FOU75-NoMaterializeLifetimeV4,D-FOU76-HighFrequencyQuarantineV2,D-RBF70-ActiveCenterOccupancyV4,D-RBF71-WidthConditionGuardV4,D-RBF72-CompactBumpNoDenseV4,D-RBF73-GaussianLocalK4TaskHealthV2,D-RBF74-CenterSplitConsensusMetric,D-WAV61-TriangularSupportV5,D-WAV62-ScaleOccupancyV3,D-WAV63-SupportOverlapDampingV3,D-WAV64-LocalTailCoverageAuditV2

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py --out-dir results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/official_v157 --line-d-out results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py --out-dir results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/official_v157 --line-d-out results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

## 2.1 过程修正

```text
自查发现初版 v15.7 official run 把 minibatch generator 与 method 名字绑定，
会让 G7R/G7K 的比较混入 batch stochasticity。
已修正为 dataset/seed/family matched batch stream，random direction 使用独立 method-specific generator。
v15.7 实现从 v15.6 runner 派生，但已将核心语义改为 U2 metric-space validation 与 K carrier reparameterization。
Line U2/B/P 只做 readback/audit。
计划要求 10 张指定 figures、Line P P1..P7 taxonomy、D-FOU72..76/D-RBF70..74/D-WAV61..64 surface 已纳入 manifest。
自查发现 U2 gate 初版会被 U4 random / U5 AdamW control support 打开；已改为只有 U0/G7R 与 U1..U3 Q-source readback eligible。
如发生修复，会用 --reuse-if-present 0 重跑 official GPU 矩阵；finalizer-only 修复才用 --reuse-if-present 1。
```

## 3. 结果摘要

```text
route = R4-StabilizationKillsSource
minimum_success = S1-SourceSignalObservable
line_s_gate_pass = 1
line_b_gate_pass = 0
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.0
bad_event_fraction = 1.0
generic_metric_stability_explains = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
promotion_allowed = 0
```

## 4. 覆盖反方复核

复核/修复指令：

```bash
rg -n "FIGURES|fig_u2|fig_k|fig_g|fig_b|fig_p|fig_d|fig_m|fig_failure|fig_route" experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py
rg -n "P1-|P2-|P3-|P4-|P5-|P6-|P7-|P-B|failure_class" experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py docs/DG-KAN_v15.7_SourceHazardFactorization_CarrierReparam_AllBasisAcceleration_完整计划.md

/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v157_source_hazard_factorization_carrier_reparam_allbasis.py --out-dir results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/official_v157 --line-d-out results/v15_7_source_hazard_factorization_carrier_reparam_allbasis/line_d_v157_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

复核结果：

```text
figures_present = 10 / 10
required_artifact_manifest_rows = 22, missing = 0
Line P failure classes = P3-MicroHorizonGoodRealBad=31, P4-ControlEquivalentPath=14, P2-HazardNotRemoved=9
positive-looking rows matched controls missing = 0
Q full-null retention failed rows = 27
Q source-retention repaired rows = 27
Line U source_hazard_colinear_rows = 275 / 378
Line U mean_hazard_overlap = 0.7798357372044058
Line U mean_source_retention_after_null = 0.2547888235569247
route = R4-StabilizationKillsSource
promotion_allowed = 0
```

结论：

```text
v15.7 没有达成 S2/S3/S4/S5，也没有 promotion。
required figures、Line P taxonomy、positive-looking matched controls 均通过覆盖复核。
当前 v15.7 内没有仍可合法补跑的分支；继续需要下一版 theory-level stability definition 或 substrate/base-architecture 计划。
```
