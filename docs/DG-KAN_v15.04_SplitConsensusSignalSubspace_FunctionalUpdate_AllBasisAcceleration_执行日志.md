# DG-KAN v15.04 SplitConsensusSignalSubspace FunctionalUpdate AllBasisAcceleration 执行日志

## 1. 关键文件

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.04_SplitConsensusSignalSubspace_FunctionalUpdate_AllBasisAcceleration_完整计划.md
runner = experiments/run_v154_split_consensus_signal_subspace_fu_allbasis.py
out_dir = results/v15_04_split_consensus_signal_subspace_fu_allbasis/official_v154_batch256_repair
line_d_out = /home/chengshun.wang/DG-LCA/results/v15_04_split_consensus_signal_subspace_fu_allbasis/line_d_v154_allbasis_substrate
```

## 2. 复现命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v154_split_consensus_signal_subspace_fu_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_04_split_consensus_signal_subspace_fu_allbasis/line_d_v154_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --batch-size 32 --epochs 1 --candidates D-FOU57-LowFreqIdentityResidualV5,D-FOU58-BandwiseSNRWarmupV3,D-FOU59-PhaseStableBandMixV3,D-FOU60-NoMaterializeLifetimeV3,D-FOU61-HighFrequencyQuarantineV3,D-RBF55-CompactBumpIdentityResidualV4,D-RBF56-ActiveCenterOccupancyRepairV4,D-RBF57-WidthConditionGuardV4,D-RBF58-GaussianLocalK4NoDenseV3,D-RBF59-CenterSNRWarmupV2,D-WAV49-TriangularSupportV5,D-WAV50-ScaleOccupancyV4,D-WAV51-SupportOverlapDampingV3,D-WAV52-LocalTailCoverageAuditV3

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v154_split_consensus_signal_subspace_fu_allbasis.py --out-dir results/v15_04_split_consensus_signal_subspace_fu_allbasis/official_v154_batch256_repair --line-d-out results/v15_04_split_consensus_signal_subspace_fu_allbasis/line_d_v154_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 0

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v154_split_consensus_signal_subspace_fu_allbasis.py --out-dir results/v15_04_split_consensus_signal_subspace_fu_allbasis/official_v154_batch256_repair --line-d-out results/v15_04_split_consensus_signal_subspace_fu_allbasis/line_d_v154_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --batch-size 256 --split-count 4 --trace-interval 10 --linec-seeds 0,1,2 --real-linec 1 --reuse-if-present 1
```

## 3. 过程修正

```text
修复 finalizer 自举 artifact 顺序：
首次 official run 的训练 artifact 完整且 required/audit 均为 0，
但 route 在 manifest 自举阶段被错误锁成 R0。
修复后 reuse-if-present=1 只重写 route / gate recompute / manifest / docs，
不重跑训练，不修改任何实验指标。
GPU policy：所有训练/finalizer 命令均显式 --device cuda:0。
补齐 method surface manifest 的 C0..C8 显式 alias 映射；
其中 C8 映射到 M3/M4 MLP SameSplitConsensus control rows。
修正 S2/S3 low-rank signal subspace：使用 A_split=C_split-lambda*N_split
在 split-gradient span 内做正特征子空间投影。
补跑 S-FB1 K=2/K=4/K=8 sensitivity diagnostic；
修正后已用 --reuse-if-present 0 重跑 official GPU 矩阵。
基于 K=2-only signal clue 执行 batch_size=256 micro-split budget repair；
不使用 audit metric 生成方向。
再次复核发现 G7-MetricNoProjection 初版仍对 signal mask 做 soft attenuation，
已修正为真正 no-projection metric update；G3/G4/G5/G6/G8 统一使用
metric base update 后再做 signal projection。
该修复按计划区分 metric-only 与 projection value，不新增 G9/G10。
修复后再次用 --reuse-if-present 0 重跑 batch256 official GPU 矩阵。
最终修复独立 gate/route recompute 的 R4 优先级；
修复后 gate_route_inconsistent_rows = 0。
用户再次追问后补齐 D-CHE/Rational no-regression monitor required coverage；
该修复不新增训练，不改变 route，不写成 promotion。
再次按完整计划做 deep coverage audit，逐项核对 S-FB/G-FB/P/C/D/M/monitor；
确认 deep_coverage_unclosed_rows = 0。
再次补齐 gate semantics audit，逐项重算 weak gate components、R4 条件和 direction provenance；
失败项只来自 weak exploration 的 bad_event component，不作为 artifact 缺口。
用户指出数据/分析边界后，runner 改为只自动写 artifact 数据，并保留手工分析 marker 区。
用户质疑复盘过薄后，补齐完整计划执行对照、fallback ladder、gate recompute、
contract/deep coverage、Line M delta 与 Line D taxonomy 到复盘文件。
```

## 4. 结果摘要

```text
route = R4-LineCOrTailDominated
minimum_success = S1-SplitConsensusSubspaceObservable
line_s_gate_pass = 1
real_lite_pass_count = 3 / 9
source_vs_best_control_mean = 0.4495026138093736
generic_split_consensus_explains = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
contract_unclosed_rows = 0
deep_coverage_unclosed_rows = 0
gate_semantics_failed_components = 2
promotion_allowed = 0
```
