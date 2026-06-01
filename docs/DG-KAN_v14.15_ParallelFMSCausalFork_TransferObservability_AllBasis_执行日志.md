# DG-KAN v14.15 ParallelFMSCausalFork TransferObservability AllBasis 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = docs/DG-KAN_v14.15_ParallelFMSCausalFork_TransferObservability_AllBasis_完整计划.md
runner = experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py
line_d_runner = experiments/run_v149_line_d_all_basis_substrate_repair.py
out_dir = results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/official_v1415
line_d_out_dir = results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration
line_m_out_dir = results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion
```

## 2. 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py experiments/run_v143_nonrat_compact_task_health_probe.py

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration --device cuda:0 --candidates D-FOU32-LowFreqIdentityResidualV3,D-FOU33-BandwiseSNRWarmupV2,D-FOU34-PhaseStableBandMixNoHighFreqV2,D-FOU35-NoMaterializeLifetimeV4,D-FOU36-HighFrequencyQuarantineV2,D-RBF30-ActiveCenterOccupancyV3,D-RBF31-WidthConditionIdentityResidualV2,D-RBF32-CompactBumpNoDenseMaterializationV2,D-RBF33-GaussianLocalK4TaskHealthV2,D-RBF34-CenterOccupancyWarmupNoTaskBranch,D-WAV29-TriangularSupportV4,D-WAV30-ScaleOccupancyNoTailTargetV2,D-WAV31-LocalSupportOverlapDampingV2,D-WAV32-LocalTailCoverageAuditV2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 12319500,12319501,12319502

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py --out-dir results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion --device cuda:0 --methods '' --mlp-methods M5-MLP-NoOpMatchedOverhead,M6-MLP-SameActiveFractionControl,M7-MLP-BoundaryOnly --skip-real 1 --run-fallbacks 0 --train-steps 200 --batch-size 32

/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py --out-dir results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/official_v1415 --line-d-out results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_d_v1415_substrate_acceleration --line-m-out results/v14_15_parallel_fms_causal_fork_transfer_observability_allbasis/line_m_mlp_control_completion
```

## 3. 关键结果

```text
route = R15-ParallelForkNoCausalValueAllBasisBlocked
Line P = R-P-PredictorWeak
Line I = R-I-DirectionControlEquivalent
Line B = R-B-BoundaryIsHarmlessNull
Line F = R-F-RealLiteBelow4
Line D = R-D-SubstrateStillBlocked
Line M = M-ControlsCompleteNoGenericPositive
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 4. 覆盖性修正记录

```text
首次计划复核后发现 Line D D-WAV29..32 属于预注册 low-budget all-basis 分支。
因此将 Line D 命令从 D-FOU/D-RBF 扩展为 D-FOU/D-RBF/D-WAV 全量候选并重跑。
最终 v1415_allbasis_substrate_acceleration.csv candidate rows = 261；
v14.14 baseline rows = 135；
v14.15 new rows = 126；
summary rows = 3；
total rows = 264。
D-WAV 补齐后仍未打开 substrate exploration gate；
line_d_best_non_dche_dataset_seed_pass_count = 2 / 9。
```

## 4.1 Line D D-FOU telemetry 补齐记录

```text
复核 v14.15 计划 8.2 后发现 D-FOU 必须测：
  band_energy_low/mid/high
  phase_drift
  high_freq_ratio
  bandwise_snr
本次在 compact substrate probe 中新增 train-stream layer1 basis channel split telemetry。
telemetry 只写入 artifact，不参与 direction / gate / promotion。
line_d_fou_telemetry_required_rows = 45。
line_d_fou_telemetry_available_rows = 45。
line_d_fou_telemetry_missing_rows = 0。
line_d_fou_telemetry_source = train_stream_layer1_basis_channel_split。
line_d_rbf_residual_required_rows = 45。
line_d_rbf_residual_available_rows = 45。
line_d_rbf_residual_missing_rows = 0。
line_d_rbf_residual_source = no_linear_residual_present,train_stream_logit_residual_norm_ratio。
v14.15 gate = step_ratio<=1.75 & memory_ratio<=1.75 & mean_delta>=-0.05 & worst_delta>=-0.10 & LineC_pass_rate>=0.30。
```

Line D visualization 覆盖：

```text
fig_d_family_pass_heatmap.svg
fig_d_workspace_task_pareto.svg
fig_d_nonrat_telemetry_matrix.svg
fig_d_linec_tail_task_breakdown.svg
```

## 5. Line P leaveout 补齐记录

```text
复核 v14.15 计划 4.3 后，Line P 必须报告六类 split：
  leave-dataset-out
  leave-seed-out
  leave-loss-interface-out
  leave-method-out
  leave-control-out
  leave-synthetic-family-out
本轮将 v1415_predictor_leaveout.csv 规范为 planned split table。
可从 v14.14 E4 artifact 读取的 split 直接写入真实 AUC；
缺失的 method/control split 只记录 unavailable，不填假 AUC。
line_p_leaveout_rows = 54。
line_p_leaveout_available_rows = 36。
line_p_leaveout_unavailable_rows = 18。
```

## 6. Line M control 补齐记录

```text
复核 v14.15 计划 9.2 后发现必须比较 7 类 MLP/generic controls。
旧 artifact 只覆盖：
  MLP-AdamW、MLP-FMS-generic、MLP-RandomMatchedNorm、
  MLP-GenericOptimizerStateControl，以及一个 MLP degree analog。
本次新增并执行：
  MLP-NoOpMatchedOverhead
  MLP-SameActiveFractionControl
  MLP-FMS-boundary-only
最终 line_m_executed_required_control_count = 7 / 7。
line_m_missing_required_control_count = 0。
```

## 7. Line I selector 补齐记录

```text
复核 v14.15 计划 5.2/5.3/5.4 后发现 reduced matrix 缺少：
  S1-E4-predictor-high-score
  S2-inverse-E4-predictor
本次从 v14.13/v14.14 真实 artifact 重建 E4-K recovery_lag selector replay。
line_i_selector_ids = S0-always-on,S1-E4-predictor-high-score,S2-inverse-E4-predictor,S3-random-matched-active-fraction,S4-NoOp-safe-abstention-selector。
line_i_boundary_ids = B0-no-boundary,B1-degree-projection-safety,B2-low-amplitude-boundary,B3-rejection-fraction-cap,B4-value-retention-floor。
line_i_selector_replay_rows = 45。
line_i_value_retention_floor_rows = 9。
strict same-direction random-selector contrast 未执行，
因此只作为 diagnostic coverage，不允许 promotion。
```

## 8. 复现说明

```text
1. 先运行 py_compile。
2. 再运行 Line D substrate-only acceleration；该命令不执行 official FMS proof。
3. 再运行 Line M MLP control completion，补齐 NoOp / SameActiveFraction / boundary-only controls。
4. 最后运行 v14.15 runner 汇总 P/I/B/F/D/M/C/Z；
   runner 会从既有 v14.13/v14.14 artifact 重建 S1/S2 selector replay。
5. 所有 replay 行都在 CSV 中标注 reused/replay/source artifact；不得写成新训练。
```
