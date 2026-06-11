# DG-KAN v22.10 ConstructiveRetainedSource FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-07 19:27:08 +0800

## Route

- route: `R8-KANRetainedSourceOpened`
- promotion_allowed: 0
- blocking_metric: `optimization_loss_agnostic_efficiency_gate;D-CHE_D-FOU_efficiency_reconfirm_gate;D-RAT_D-RBF_robust_efficiency_gate`
- S0.17 pass / CodeRoute: 1 / `R0-CodePacketSelfContained`
- D-CHE/D-FOU S1 pass: 0 / 0
- D-RAT/D-RBF robust pass: 0 / 0
- D-RAT/D-RBF official candidate scan route / groups / rows: `NotRunLossAgnosticContract` / 0 / 0
- D-RAT/D-RBF blockH repair scan route / groups / rows: `NotRunLossAgnosticContract` / 0 / 0
- D-RAT/D-RBF fastK2 repair scan route / groups / rows: `NotRunLossAgnosticContract` / 0 / 0
- D-RAT/D-RBF blockB repair scan route / groups / rows: `NotRunLossAgnosticContract` / 0 / 0
- D-RAT/D-RBF combo repair scan route / groups / rows: `NotRunLossAgnosticContract` / 0 / 0
- optimization loss-agnostic efficiency contract pass / CE rows invalidated: 0 / 1
- loss-agnostic source contract pass / uses_labels / uses_loss: 1 / 0 / 0
- S2 source atom pass rows / selected attempt: 2 / `initial_norm_0p08_split4_all`
- S3 variational pass rows / selected attempt / blocker: 1 / `control_balanced_l1` / ``
- S4/C3/C4 pass rows: 1 / 7 / 7
- S5 PID attempts / C3/C4 / best: 3 / 3 / 3 / `pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0` h3200=0.22984354124227524
- KAN mapping decision: `KANRetainedSourceOpened`
- execution_contract_violation: 0
- minimum_effective_progress: `A-CodeClosure;C-SourceAtomProgress;D-ConstructiveSolveProgress;E-FunctionalProgress;F-KANSourceProgress`
- next_codex_action: implement arbitrary-loss/upstream-gradient S1 efficiency runner before any basis-efficiency repair or promotion

## Executive Summary

- 目标没有达成 official promotion：`promotion_allowed=0`。核心原因不是 source-side 失败，而是用户明确要求 loss-agnostic 后，S1 basis-efficiency 仍没有 arbitrary-loss/upstream-gradient runner；所有 CE-targeted efficiency rows 被 formal invalidated。
- source-side constructive path 已经打开：S2 pass rows=2，S3 pass rows=1，S4 pass rows=1，C3/C4 pass rows=7/7，KAN decision=`KANRetainedSourceOpened`。
- PID 思路已经按 loss-agnostic 约束真实尝试：PID attempts=3，best PID=`pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0`，h3200 margin=0.2298，final projected gain=1.003，相对 best non-PID h3200 margin 改善=0.1179。
- KAN opening 的真实来源是 `D-FOU` / `readout_source_state_replay_lr0` / `slower_replay_100x0p16_repair`，h3200 margin=0.3321，KAN-specific delta=0.1023；basis-estimate rows 本轮没有打开，best basis-estimate delta=-1.274。
- 当前最有价值的结果是：loss-agnostic source geometry、PID functional update 和 KAN source-channel 这条线给出正证据；但 efficiency gate 必须先补任意 loss/上游梯度接口，否则不能 promotion。

## Exploration Process

| stage | question | attempt_or_repair | result | decision |
| --- | --- | --- | --- | --- |
| S0.17 | 代码包是否自洽 | clean unzip compile/import + required CSV vs zip + semantic/kernel audit | pass=1 CodeRoute=R0-CodePacketSelfContained | 进入科学 gate |
| S1 | basis efficiency 是否可正式计入 | loss-agnostic clarified contract | contract_pass=0 CE_invalidated=1 | fail-closed，等待 arbitrary-loss/upstream-gradient runner |
| S2 | 能否生成 loss-agnostic source atoms | 5 attempt groups; best=A11_causal_split_transport_atom | pass_rows=2 best_B2=0.3527 | A10/A11 进入 S3 |
| S3 | atoms 能否组合成低 control projection source | control_balanced_l1 | DDR=11.41 control_projection=0.2062 weights=A10_split_consensus_atom:0;A11_causal_split_transport_atom:0.85 | 进入 S4 commit |
| S4 | metric dynamics commit 是否真实可 actuation | S4.5-readout-all-train-actuation-repair | residual=0.03945 R2=0.9984 B2=0.8971 | 进入 h100-h6400 horizon |
| S5 | source 是否在 horizon 内 retained | FU attempts=15 PID attempts=3 | best_nonPID_h3200=0.1119 best_PID_h3200=0.2298 PID_delta=0.1179 | C3/C4 打开，但仍受 S1 efficiency blocker 限制 |
| S6 | KAN 是否存在同 metric source-channel | rows=7 pass_rows=1 | best=D-FOU/slower_replay_100x0p16_repair h3200=0.3321 delta=0.1023 | KAN source opened，但不能 promotion |


## Data Analysis / Insight

### Loss-Agnostic Boundary

- 用户 clarified hard constraint 后，`loss-agnostic` 被落实为所有优化不得针对 CE：source direction、variational solve、commit、horizon、KAN mapping 都必须不使用 labels 和 supervised loss；basis efficiency 也不能借 manual CE trainpath 做正式 closure。
- S1 当前 formal status：optimization contract pass=0，CE-targeted rows invalidated=1。因此 D-CHE/D-FOU 的旧 S1 readback、D-RAT/D-RBF repair scan 只作为 legacy implementation/audit context，不能用于 promotion。
- 这个 blocker 很关键：v22.10 source-side 已经比 v22.09 多走到 C3/C4/KAN opening，但 route 仍是 R8，而不是 R9，因为 promotion 必须同时满足 source-side 与 efficiency-side official contract。

### Source Atom Search

- S2 一共执行 5 组 source atom attempts：initial/lower-norm attempts 有 pass rows，more-splits 与 block-restricted attempts 是真实负结果。best atom=`A11_causal_split_transport_atom`，family=`causal_split_transport`，B2 transfer=0.3527，DDR=11.41，control projection=0.2062。
- A10 split-consensus 与 A11 causal split transport 只读 train logits 的几何结构，不读 labels、不读 CE。它们的意义不是预测 future label，而是在当前 train-stream 中形成可 replay 的 target displacement source atom。
- lower-norm attempts 仍能保留 pass rows 但 B2 下降，说明 source signal 不是单纯 norm/scale artefact；more-splits 和 block-restricted rows 没打开，说明当前有效 source 需要全 train-stream 的 split/causal geometry，而不是更碎的 split 或单 block 限制。

### Variational Solve

- S3 selected attempt=`control_balanced_l1`，selected atoms=`A10_split_consensus_atom;A11_causal_split_transport_atom`，weights=`A10_split_consensus_atom:0;A11_causal_split_transport_atom:0.85`，DDR=11.41，control projection=0.2062，NDS=3.629。
- default sparse combination 的 DDR 读数很高但 control projection 更高；control-balanced L1 通过 capped DDR 与 control projection 约束，避免把 control-equivalent direction 写成 source。这是 v22.10 从 observer 到 constructive source 的主要语义变化。

### Commit Repair

- S4 selected commit=`S4.5-readout-all-train-actuation-repair` / block_role=`readout_only`，projection residual=0.03945，ActuationR2=0.9984，B2 transfer=0.8971，random matched B2=-2.312。
- readout-only、hidden/readout、高 damping、optimizer-state-only commit attempts 都没有打开；all-train actuation repair 才把 target displacement 以几何 readback 方式写入当前 tiny-model train stream。

### Horizon And PID

- S5 non-PID best=`finite_source_state_replay_400x0p20_stop800_lr0`，h3200 margin=0.1119；PID best=`pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0`，h3200 margin=0.2298，h4800 margin=0.2298，PID improvement=0.1179。
- PID controller 的 error 是 label-free projected gain error：`1 - projected_gain`。它只调节 source update 注入幅度；matched random/sign/corrupt/same-solver controls 使用同一 PID controller，因此没有 source-only controller advantage。best PID mean abs error=0.003937，mean abs injection=0.00101。
- PID 的意义不是追 CE loss，而是把 functional update 线上 target displacement 的投影增益稳定在 1 附近。三个 PID attempts 都 C3/C4 pass，说明用户建议的 PID 思路在 loss-agnostic FU 线上有正证据。

### KAN Mapping

- S6 pass row 是 `D-FOU` slower replay source-channel：h100/h400/h800/h1600/h2400/h3200/h4800/h6400 相对 matched controls 均为正，h3200 row positive count=5，R4800/h3200=0.8751。
- basis-estimate readout commit rows 的 fit cosine 虽高，但 source-channel delta 仍为负；best basis-estimate delta=-1.274。因此不能把本轮 opening 写成 basis-estimate opening，正式结论必须写成 D-FOU slower replay source-channel opening。
- KAN result 与 MLP source-side result 一致地只使用 label-free target-retention score；它证明存在 KAN-side source channel evidence，但不解除 S1 loss-agnostic efficiency blocker。

## Part A Code / Packet Truth Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 28/28 |  |
| compileall_repo | 1 | py_compile | 0 |  |
| import_closure_repo | 1 | import | 0 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 4/4 |  |
| metric_solver_tests | 1 | tests | 33/33 |  |
| source_atom_generator_tests | 1 | tests | 2/2 |  |
| variational_solver_tests | 1 | tests | 2/2 |  |
| constructive_commit_tests | 1 | tests | 1/1 |  |
| loss_agnostic_source_contract_tests | 1 | tests | 3/3 |  |
| profiler_phase_tests | 1 | tests | 5/5 |  |
| kernel_status_consistency | 1 | official_vs_gradcheck | 1 |  |
| semantic_contract | 1 | forbidden+alias | forbidden_pass=1;undeclared_alias_pairs=0 |  |
| csv_claimed_exists_but_zip_missing_count | 1 | zip_required_compare | 0 |  |
| self_contained_compileall | 1 | clean_unzip_compile | 0 |  |
| self_contained_import_check | 1 | clean_unzip_import | 0 |  |


### Clean Unzip Self Test

| packet | unzip_root | compileall_returncode | import_returncode | self_contained_import_check | pass |
| --- | --- | --- | --- | --- | --- |
| /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/_code_packet_unzip | 0 | 0 | 1 | 1 |


### Kernel Gradcheck / Status Consistency

| family | kernel_forward_relerr | kernel_grad_relerr | kernel_grad_cosine | kernel_correctness_official_pass | official_fused_kernel_complete | no_nan_inf |
| --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 0.0 | 0.0 | 1.0 | 1 | 0 | 1 |
| D-FOU | 0.0 | 9.656568435259105e-17 | 0.9999999999999998 | 1 | 0 | 1 |
| D-RAT | 0.0 | 0.0 | 1.0 | 1 | 0 | 1 |
| D-RBF | 0.0 | 0.0 | 0.9999999999999997 | 1 | 0 | 1 |


## Part B Basis Efficiency

- hard constraint: v22.10 formal S1 cannot optimize or profile a CE-targeted trainpath. Existing manual-CE D-RAT/D-RBF scans are retained only as legacy code/audit context and are not executed or promoted in this formal route until an arbitrary-loss/upstream-gradient efficiency runner exists.
| carrier | variant_family | forward_ratio_vs_mlp | step_ratio_vs_mlp | memory_ratio_vs_mlp | functional_runner_same_kernel | fallback_kernel_used | official_fused_kernel_complete | v22_09_S1_pass | formal_status | v22_10_loss_contract | v22_09_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE21-R2/R4 |  |  |  |  |  |  | 0 | blocked_before_CE_targeted_training_loop | CE_targeted_trainpath_not_allowed_for_v22_10_formal_optimization | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |
| D-FOU | FOU21-R3 |  |  |  |  |  |  | 0 | blocked_before_CE_targeted_training_loop | CE_targeted_trainpath_not_allowed_for_v22_10_formal_optimization | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |
| D-RAT | RAT22.10 |  |  |  |  |  |  | 0 | blocked_before_CE_targeted_training_loop | CE_targeted_trainpath_not_allowed_for_v22_10_formal_optimization | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |
| D-RBF | RBF22.10 |  |  |  |  |  |  | 0 | blocked_before_CE_targeted_training_loop | CE_targeted_trainpath_not_allowed_for_v22_10_formal_optimization | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |


### D-RAT / D-RBF Multibatch

| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 0 | 0 | 0 | 0 |  |  |  | S1LossAgnosticEfficiencyNotEstablished | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |
| D-RBF | 0 | 0 | 0 | 0 |  |  |  | S1LossAgnosticEfficiencyNotEstablished | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |


### Telemetry Bridge

| carrier | profile_rows | robust_production_pass_rows | component_telemetry_complete_rows | best_forward_ratio | best_step_ratio | decision | blocker | official_closure_claimed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 0 | 0 | 0 |  |  | S1LossAgnosticEfficiencyNotEstablished | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |  |
| D-RBF | 0 | 0 | 0 |  |  | S1LossAgnosticEfficiencyNotEstablished | optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing |  |


### Official Candidate Repair Scan

- status: not run in the formal v22.10 route because the available official S1 trainpath is CE-targeted. These implementation-side scan functions remain in the code packet, but they cannot produce a promotion claim under the clarified loss-agnostic optimization contract.

_无落盘 rows。_


### Official BlockH Repair Scan

- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.

_无落盘 rows。_


### Official FastK2 Repair Scan

- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.

_无落盘 rows。_


### Official BlockB Repair Scan

- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.

_无落盘 rows。_


### Official Combo Repair Scan

- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.

_无落盘 rows。_


### Focused Low-Hidden Sanity

- status: legacy focused CE-trainpath audit is omitted from the formal recap after the loss-agnostic optimization clarification.

_无落盘 rows。_


_无落盘 rows。_


_无落盘 rows。_


_无落盘 rows。_


## Part C Source Atom Generation

- S2 使用 TinyConstructiveMLP 的当前 train logits；source direction 只由 label-free train-stream logit geometry 构造，不使用 labels、supervised loss、validation/test/future/LineC/ECE/Brier/AUCtime。
| attempt | fallback_step | norm_scale | split_count | block_role | atom_rows | S2_source_atom_pass_rows | control_NDS_median | random_matched_B2_gain | route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_norm_0p08_split4_all | initial | 0.08 | 4 | all | 11 | 2 | 3.7546775341033936 | -0.011488250456750393 | S2-SourceAtomPass |
| fallback_lower_norm_0p04_split4_all | lower_source_atom_norm | 0.04 | 4 | all | 11 | 2 | 3.7546775341033936 | -0.009984939359128475 | S2-SourceAtomPass |
| fallback_lower_norm_0p02_split4_all | lower_source_atom_norm | 0.02 | 4 | all | 11 | 2 | 3.7546775341033936 | -0.009163266979157925 | S2-SourceAtomPass |
| fallback_more_splits_0p02_split6_all | increase_micro_batch_split_count | 0.02 | 6 | all | 11 | 0 | 3.7546775341033936 | 0.043716657906770706 | S2-SourceAtomAttemptBlocked |
| fallback_block_restricted_0p02_split6_readout | block_restricted_atom | 0.02 | 6 | readout_only | 11 | 0 | 3.7546775341033936 | 0.043716657906770706 | S2-SourceAtomAttemptBlocked |


### Source Atom Rows

| attempt | atom_id | atom_family | B2_transfer_gain | random_gap | sign_flip_gap | corrupt_target_gap | commutator_norm_ratio | DDR | NDS | control_projection_fraction | S2_source_atom_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_norm_0p08_split4_all | A1_split_transfer_atom | split_transfer | -0.026317227631807327 | -0.014828977175056934 | -0.05263550579547882 | -0.025640649429988116 | 0.0 | 2.1151351205445437e-14 | 6.637226581573486 | 0.003480230923742056 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A2_drift_diffusion_atom | drift_diffusion | -0.026317225769162178 | -0.014828975312411785 | -0.05263550393283367 | -0.025640647567342967 | 0.0 | 3.388570890897058e-13 | 6.637227535247803 | 0.0034802297595888376 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A3_control_null_residual_atom | control_null_residual | -0.02699876017868519 | -0.015510509721934795 | -0.05331703834235668 | -0.026322181976865977 | 0.0 | 4.9249672883888707e-05 | 6.636465072631836 | 0.0020734320860356092 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A4_low_NDS_atom | low_NDS | -0.044458650052547455 | -0.03297039959579706 | -0.07077692821621895 | -0.043782071850728244 | 0.0 | 0.01996479742228985 | 7.081103801727295 | 0.0021289787255227566 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A5_row_orthogonal_atom | row_orthogonal | -0.026317227631807327 | -0.014828977175056934 | -0.05263550579547882 | -0.025640649429988116 | 0.0 | 2.7497189909134884e-14 | 6.637226581573486 | 0.003480231622233987 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A6_fast_slow_source_state_atom | fast_slow_source_state | -0.026697201654314995 | -0.015208951197564602 | -0.05301547981798649 | -0.026020623452495784 | 1.2303012534746198 | 0.0023193485103547573 | 6.622964382171631 | 0.0035109298769384623 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;commutator_norm_ratio_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A7_train_flow_commutator_atom | train_flow_commutator | -0.0492788702249527 | -0.037790619768202305 | -0.07559714838862419 | -0.048602292023133487 | 0.06826630059341342 | 4.726888707382894e-14 | 6.460636615753174 | 0.0002830489829648286 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A8_low_degree_frequency_atom | low_degree_frequency | -0.006183125078678131 | 0.005305125378072262 | -0.032501403242349625 | -0.00550654687685892 | 0.0 | 1.823487243363467e-13 | 7.8629279136657715 | 0.0029323198832571507 | 0 | sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A9_control_balanced_drift_atom | control_balanced_drift | -0.06400184705853462 | -0.05251359660178423 | -0.09032012522220612 | -0.06332526885671541 | 0.0 | 0.03341406211256981 | 7.132145404815674 | 0.0006138123571872711 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A10_split_consensus_atom | split_consensus | 0.32659875229001045 | 0.33808700274676085 | 0.30028047412633896 | 0.32727533049182966 | 0.0 | 3200001.0 | 3.6238434314727783 | 0.41195592284202576 | 1 |  |
| initial_norm_0p08_split4_all | A11_causal_split_transport_atom | causal_split_transport | 0.3527107620611787 | 0.3641990125179291 | 0.3263924838975072 | 0.3533873402629979 | 0.0 | 11.407050132751465 | 3.62947416305542 | 0.2061547487974167 | 1 |  |
| fallback_lower_norm_0p04_split4_all | A1_split_transfer_atom | split_transfer | -0.026341427117586136 | -0.01635648775845766 | -0.05268309265375137 | -0.025978034071158618 | 0.0 | 2.1151351205445437e-14 | 6.637226581573486 | 0.003480230923742056 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A2_drift_diffusion_atom | drift_diffusion | -0.026341425254940987 | -0.01635648589581251 | -0.052683090791106224 | -0.02597803220851347 | 0.0 | 3.388570890897058e-13 | 6.637227535247803 | 0.0034802297595888376 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A3_control_null_residual_atom | control_null_residual | -0.026943713426589966 | -0.01695877406746149 | -0.0532853789627552 | -0.026580320380162448 | 0.0 | 4.925124449073337e-05 | 6.636464595794678 | 0.002073431620374322 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A4_low_NDS_atom | low_NDS | -0.042237646877765656 | -0.03225270751863718 | -0.06857931241393089 | -0.04187425383133814 | 0.0 | 0.01996479742228985 | 7.081103801727295 | 0.0021289787255227566 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A5_row_orthogonal_atom | row_orthogonal | -0.026341427117586136 | -0.01635648775845766 | -0.05268309265375137 | -0.025978034071158618 | 0.0 | 2.7497189909134884e-14 | 6.637226581573486 | 0.003480230923742056 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A6_fast_slow_source_state_atom | fast_slow_source_state | -0.02667785994708538 | -0.016692920587956905 | -0.05301952548325062 | -0.026314466900657862 | 1.2303012534746198 | 0.0023193485103547573 | 6.622964382171631 | 0.0035109298769384623 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;commutator_norm_ratio_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A7_train_flow_commutator_atom | train_flow_commutator | -0.04567941278219223 | -0.035694473423063755 | -0.07202107831835747 | -0.04531601973576471 | 0.06826630059341342 | 4.726888707382894e-14 | 6.460636615753174 | 0.0002830489829648286 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A8_low_degree_frequency_atom | low_degree_frequency | -0.008456975221633911 | 0.001527964137494564 | -0.03479864075779915 | -0.008093582175206393 | 0.0 | 1.823487243363467e-13 | 7.8629279136657715 | 0.0029323198832571507 | 0 | loss_agnostic_B2_transfer_gain_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A9_control_balanced_drift_atom | control_balanced_drift | -0.05975930392742157 | -0.049774364568293095 | -0.08610096946358681 | -0.05939591088099405 | 0.0 | 0.03341405466198921 | 7.132145404815674 | 0.0006138126482255757 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A10_split_consensus_atom | split_consensus | 0.28747033327817917 | 0.29745527263730764 | 0.26112866774201393 | 0.2878337263246067 | 0.0 | 800000.25 | 3.6238434314727783 | 0.41195592284202576 | 1 |  |
| fallback_lower_norm_0p04_split4_all | A11_causal_split_transport_atom | causal_split_transport | 0.30045974627137184 | 0.3104446856305003 | 0.2741180807352066 | 0.30082313931779936 | 0.0 | 11.407050132751465 | 3.62947416305542 | 0.2061547487974167 | 1 |  |
| fallback_lower_norm_0p02_split4_all | A1_split_transfer_atom | split_transfer | -0.026353564113378525 | -0.0171902971342206 | -0.05270718038082123 | -0.026207847229670733 | 0.0 | 2.1151351205445437e-14 | 6.637226581573486 | 0.003480230923742056 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A2_drift_diffusion_atom | drift_diffusion | -0.026353562250733376 | -0.01719029527157545 | -0.05270717851817608 | -0.026207845367025584 | 0.0 | 3.388570890897058e-13 | 6.637227535247803 | 0.0034802297595888376 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A3_control_null_residual_atom | control_null_residual | -0.02691369503736496 | -0.017750428058207035 | -0.05326731130480766 | -0.026767978153657168 | 0.0 | 4.925124449073337e-05 | 6.636464595794678 | 0.002073431620374322 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A4_low_NDS_atom | low_NDS | -0.0410601943731308 | -0.031896927393972874 | -0.0674138106405735 | -0.04091447748942301 | 0.0 | 0.01996479742228985 | 7.081103801727295 | 0.0021289787255227566 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A5_row_orthogonal_atom | row_orthogonal | -0.026353564113378525 | -0.0171902971342206 | -0.05270718038082123 | -0.026207847229670733 | 0.0 | 2.7497189909134884e-14 | 6.637226581573486 | 0.003480230923742056 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A6_fast_slow_source_state_atom | fast_slow_source_state | -0.026666855439543724 | -0.0175035884603858 | -0.05302047170698643 | -0.026521138555835932 | 1.2303012534746198 | 0.0023193485103547573 | 6.622964382171631 | 0.0035109298769384623 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;commutator_norm_ratio_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A7_train_flow_commutator_atom | train_flow_commutator | -0.04377231001853943 | -0.034609043039381504 | -0.07012592628598213 | -0.04362659313483164 | 0.06826630059341342 | 4.726888707382894e-14 | 6.460636615753174 | 0.0002830489829648286 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A8_low_degree_frequency_atom | low_degree_frequency | -0.00965707004070282 | -0.0004938030615448952 | -0.03601068630814552 | -0.009511353156995028 | 0.0 | 1.823487243363467e-13 | 7.8629279136657715 | 0.0029323198832571507 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |

_仅显示前 30 / 55 rows；完整 CSV 见 artifact。_


## Part D Variational Solve

| attempt | source_combo_atom_count | source_combo_l1_norm | DDR | control_projection_fraction | random_gap | sign_flip_gap | corrupt_gap | metric_energy_Sobolev | control_Sobolev_p50 | NDS | control_NDS_median | selected_atoms | selected_atom_weights | S3_variational_source_solve_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 2 | 1.0 | 3199996.0 | 0.41195550560951233 | 0.33808715362101793 | 0.30028062500059605 | 0.32727548136608675 | 0.006399991922080517 | 0.01566186547279358 | 3.6238434314727783 | 3.7546775341033936 | A10_split_consensus_atom;A11_causal_split_transport_atom | A10_split_consensus_atom:0.99999632;A11_causal_split_transport_atom:3.6785003e-06 | 0 | control_projection_fraction_gate |
| control_balanced_l1 | 2 | 0.85 | 11.407050132751465 | 0.2061547487974167 | 0.34850273933261633 | 0.31069621071219444 | 0.33769106707768515 | 0.004712910857051611 | 0.01566186547279358 | 3.6294751167297363 | 3.7546775341033936 | A10_split_consensus_atom;A11_causal_split_transport_atom | A10_split_consensus_atom:0;A11_causal_split_transport_atom:0.85 | 1 |  |
| sparse_l1 | 2 | 0.65 | 1351998.5 | 0.41195547580718994 | 0.3106729341670871 | 0.2728664055466652 | 0.2998612619121559 | 0.0027039970736950636 | 0.01566186547279358 | 3.6238434314727783 | 3.7546775341033936 | A10_split_consensus_atom;A11_causal_split_transport_atom | A10_split_consensus_atom:0.64999761;A11_causal_split_transport_atom:2.3910252e-06 | 0 | control_projection_fraction_gate |
| low_nds_only | 2 | 0.7500000000000001 | 1799998.125 | 0.4119555354118347 | 0.31849548500031233 | 0.28068895637989044 | 0.30768381274538115 | 0.0035999955143779516 | 0.01566186547279358 | 3.623843193054199 | 3.7546775341033936 | A10_split_consensus_atom;A11_causal_split_transport_atom | A10_split_consensus_atom:0.74999725;A11_causal_split_transport_atom:2.7545951e-06 | 0 | control_projection_fraction_gate |
| control_null_only | 0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  | 0 | no_S2_pass_atoms_for_attempt |
| block_restricted | 0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  | 0 | no_S2_pass_atoms_for_attempt |


## Part E Commit / Horizon / KAN

### S4 Commit Matrix

| solver_level | block_role | projection_residual_Gf | ActuationR2 | B2_transfer_gain | random_matched_B2_transfer_gain | function_displacement_cos_with_target | S4_metric_dynamics_commit_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S4.1-readout-exact | readout_only | 0.7470011288909167 | 0.441989541053772 | -1.602543294429779 | -2.463890627026558 | 0.8238537311553955 | 0 | projection_residual_Gf_gate |
| S4.2-hidden-readout-block | hidden_readout | 0.7470011288909167 | 0.441989541053772 | -1.602543294429779 | -2.463890627026558 | 0.8238537311553955 | 0 | projection_residual_Gf_gate |
| S4.3-readout-higher-damping | readout_only | 0.7505757049457553 | 0.4366362690925598 | -1.6105836033821106 | -2.2087172120809555 | 0.8226601481437683 | 0 | projection_residual_Gf_gate |
| S4.4-optimizer-state-integrated | optimizer_state_only | 1.0 | 0.0 | -1.0 | -1.0 | 0.0 | 0 | projection_residual_Gf_gate;ActuationR2_gate;B2_transfer_gain_vs_random_gate;function_displacement_cos_gate |
| S4.5-readout-all-train-actuation-repair | readout_only | 0.039448339010604516 | 0.9984438419342041 | 0.8971059396862984 | -2.312143538147211 | 0.9992844462394714 | 1 |  |


### S5 FU Attempt Summary

| attempt | pid_control | periodic_stop_step | lr_scale | h100_margin | h800_margin | h3200_margin | h4800_margin | h6400_margin | row_positive_h3200 | control_equivalent_fraction | C3 | C4 | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_commit_adamw |  | 0 | 1.0 | 0.011235208684212017 | 0.010395598758555447 | -0.38484063363765586 | -0.2235505668820008 | 0.028673503441337456 | 1 | 0.09090909090909091 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| slow_source_replay_400x0p08 |  | 0 | 1.0 | 0.011235208684212017 | 0.01674074464909392 | 0.10914645190871897 | 0.10838065706555511 | 0.14479206410081003 | 11 | 0.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| damped_source_replay_400x0p05_lr0p5 |  | 0 | 0.5 | -0.009414347164527936 | -0.02417679850134713 | -0.04190032519124387 | -0.08986921135786885 | -0.010947286386382427 | 10 | 0.09090909090909091 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| source_replay_200x0p04 |  | 0 | 1.0 | 0.011235208684212017 | -0.04788195703255105 | -0.0770961016031928 | 0.02718042915083141 | 0.1660842995582732 | 10 | 0.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| source_state_replay_100x0p10_lr0 |  | 0 | 0.0 | 0.18968539629918557 | 0.0843616698036157 | -1.1515916603076954 | -2.7324464413747753 | -4.313338732182074 | 7 | 0.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| source_state_replay_100x0p15_lr0 |  | 0 | 0.0 | 0.16819365678224274 | 0.07395966349327154 | -2.732440890452976 | -5.103784502252757 | -7.475152830587609 | 7 | 0.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| source_state_replay_50x0p08_lr0 |  | 0 | 0.0 | 0.16443840858531888 | 0.07261655336580519 | -3.048616289646981 | -5.578049048378859 | -8.107477614117066 | 7 | 0.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| source_state_replay_200x0p20_lr0 |  | 0 | 0.0 | 0.2197596523493387 | 0.08436297115140456 | -1.1515908742315668 | -2.7324418553430387 | -4.313331037256018 | 7 | 0.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| finite_source_state_replay_200x0p20_stop800_lr0 |  | 800 | 0.0 | 0.2197596523493387 | 0.08436297115140456 | 0.08436297115140456 | 0.08436297115140456 | 0.08436297115140456 | 11 | 0.0 | 1 | 1 |  |
| finite_source_state_replay_100x0p10_stop800_lr0 |  | 800 | 0.0 | 0.18968539629918557 | 0.0843616698036157 | 0.0843616698036157 | 0.0843616698036157 | 0.0843616698036157 | 11 | 0.0 | 1 | 1 |  |
| finite_source_state_replay_400x0p20_stop800_lr0 |  | 800 | 0.0 | 0.2197596523493387 | 0.11192692578081487 | 0.11192692578081487 | 0.11192692578081487 | 0.11192692578081487 | 11 | 0.0 | 1 | 1 |  |
| finite_source_state_replay_200x0p15_stop1200_lr0 |  | 1200 | 0.0 | 0.2197596523493387 | 0.0941676052705972 | 0.08096271307133196 | 0.08096271307133196 | 0.08096271307133196 | 11 | 0.0 | 1 | 1 |  |
| pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0 | 1 | 0 | 0.0 | 0.22142406300274253 | 0.22740773947010762 | 0.22984354124227524 | 0.22984354124227524 | 0.22984354124227524 | 11 | 0.0 | 1 | 1 |  |
| pid_source_state_replay_200_kp0p20_ki0p01_kd0p05_stop1600_lr0 | 1 | 0 | 0.0 | 0.2197596523493387 | 0.22435859421779347 | 0.22683315957054795 | 0.22683315957054795 | 0.22683315957054795 | 11 | 0.0 | 1 | 1 |  |
| pid_source_state_replay_100_kp0p12_ki0p04_kd0p08_stop800_lr0 | 1 | 0 | 0.0 | 0.22128946749749867 | 0.22916705861691333 | 0.22916705861691333 | 0.22916705861691333 | 0.22916705861691333 | 11 | 0.0 | 1 | 1 |  |


### S5 PID Control Attempts

| attempt | kp | ki | kd | pid_final_projected_gain | pid_abs_error_mean | pid_abs_injection_mean | h100_margin | h3200_margin | h4800_margin | C3 | C4 | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0 | 0.16 | 0.02 | 0.08 | 1.0033196210861206 | 0.0039372555911540985 | 0.001009521186351776 | 0.22142406300274253 | 0.22984354124227524 | 0.22984354124227524 | 1 | 1 |  |
| pid_source_state_replay_200_kp0p20_ki0p01_kd0p05_stop1600_lr0 | 0.2 | 0.01 | 0.05 | 0.999315619468689 | 0.005807846784591675 | 0.0015120847523212434 | 0.2197596523493387 | 0.22683315957054795 | 0.22683315957054795 | 1 | 1 |  |
| pid_source_state_replay_100_kp0p12_ki0p04_kd0p08_stop800_lr0 | 0.12 | 0.04 | 0.08 | 1.002623200416565 | 0.005436047911643982 | 0.0019307926297187806 | 0.22128946749749867 | 0.22916705861691333 | 0.22916705861691333 | 1 | 1 |  |


### S5 Matched Control Sample

| attempt | variant | optimizer | target_retention_score_h100 | target_retention_score_h800 | target_retention_score_h3200 | target_retention_score_h4800 | target_projected_gain_h3200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initial_commit_adamw | AdamW | AdamW | -1.1979696339409052 | -0.9920123689920929 | -1.0997526903846615 | -1.0731870853853986 |  |
| initial_commit_adamw | SGD | SGD | -1.0 | -1.0 | -1.0 | -1.0 |  |
| initial_commit_adamw | Momentum | Momentum | -1.0 | -1.0 | -1.0 | -1.0 |  |
| initial_commit_adamw | NoOpMatchedOverhead | NoOp | -1.0 | -1.0 | -1.0 | -1.0 |  |
| initial_commit_adamw | RandomMatchedNorm | AdamW | -1.1535152209861346 | -0.9729417158041802 | -1.002004643075534 | -0.9612655710902122 |  |
| initial_commit_adamw | StableRandom | AdamW | -1.1178777875265797 | -1.0146870534219619 | -0.9706838391197304 | -0.9669663325330056 |  |
| initial_commit_adamw | SameAtomsRandomWeights | AdamW | -1.1535152209861346 | -0.9729417158041802 | -1.002004643075534 | -0.9612655710902122 |  |
| initial_commit_adamw | SameMetricRandomTarget | AdamW | -1.025727559568161 | -1.0777884737027519 | -1.077136421303783 | -1.0368145409526366 |  |
| initial_commit_adamw | SameSolverRandomTarget | AdamW | -1.025727559568161 | -1.0777884737027519 | -1.077136421303783 | -1.0368145409526366 |  |
| initial_commit_adamw | SignFlipTarget | AdamW | -1.2812612715000076 | -1.0195552451740106 | -1.2874065676521154 | -1.3469981824821275 |  |
| initial_commit_adamw | CorruptTarget | AdamW | -0.7289870888217942 | -0.976119498741748 | -0.7506581442126959 | -0.9023135488050458 |  |
| slow_source_replay_400x0p08 | AdamW | AdamW | -1.1979696339409052 | -0.9920123689920929 | -1.0997526903846615 | -1.0731870853853986 |  |
| slow_source_replay_400x0p08 | SGD | SGD | -1.0 | -1.0 | -1.0 | -1.0 |  |
| slow_source_replay_400x0p08 | Momentum | Momentum | -1.0 | -1.0 | -1.0 | -1.0 |  |
| slow_source_replay_400x0p08 | NoOpMatchedOverhead | NoOp | -1.0 | -1.0 | -1.0 | -1.0 |  |
| slow_source_replay_400x0p08 | RandomMatchedNorm | AdamW | -1.1535152209861346 | -1.2824564161653715 | -1.3345161762105966 | -1.3572878912417168 |  |
| slow_source_replay_400x0p08 | StableRandom | AdamW | -1.1178777875265797 | -1.2616054965657306 | -1.1942613452194233 | -1.1572892100617589 |  |
| slow_source_replay_400x0p08 | SameAtomsRandomWeights | AdamW | -1.1535152209861346 | -1.2824564161653715 | -1.3345161762105966 | -1.3572878912417168 |  |
| slow_source_replay_400x0p08 | SameMetricRandomTarget | AdamW | -1.025727559568161 | -0.9675787341831534 | -0.9790833389865179 | -0.9765607483723759 |  |
| slow_source_replay_400x0p08 | SameSolverRandomTarget | AdamW | -1.025727559568161 | -0.9675787341831534 | -0.9790833389865179 | -0.9765607483723759 |  |
| slow_source_replay_400x0p08 | SignFlipTarget | AdamW | -1.2812612715000076 | -1.3635420416088821 | -1.394593257819462 | -1.3905969880500038 |  |
| slow_source_replay_400x0p08 | CorruptTarget | AdamW | -0.7289870888217942 | -0.6565755360420621 | -0.631315058081784 | -0.6614069301750986 |  |
| damped_source_replay_400x0p05_lr0p5 | AdamW | AdamW | -0.722451439386627 | -1.000664362220945 | -1.0045298412421164 | -1.0134655329726465 |  |
| damped_source_replay_400x0p05_lr0p5 | SGD | SGD | -1.0 | -1.0 | -1.0 | -1.0 |  |

_仅显示前 24 / 165 rows；完整 CSV 见 artifact。_


### S5 Full Horizon Matrix Sample

| attempt | variant | periodic_stop_step | pid_control | pid_kp | pid_ki | pid_kd | pid_final_projected_gain | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h3200 | source_vs_best_control_h4800 | source_vs_best_control_h6400 | row_positive_count_h3200 | control_equivalent_fraction | source_target_retention_area_ratio | AUCtime_ratio | C3_source_formation_pass | C4_terminal_retention_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_commit_adamw | FU | 0 |  |  |  |  |  | 0.011235208684212017 | -0.0025208396411782186 | 0.010395598758555447 | -0.19034109352295925 | -0.38484063363765586 | -0.2235505668820008 | 0.028673503441337456 | 1 | 0.09090909090909091 | 0.0 | 1.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| initial_commit_adamw | AdamW | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | SGD | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | Momentum | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | NoOpMatchedOverhead | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | RandomMatchedNorm | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | StableRandom | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | SameAtomsRandomWeights | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | SameMetricRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | SameSolverRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | SignFlipTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| initial_commit_adamw | CorruptTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | FU | 0 |  |  |  |  |  | 0.011235208684212017 | 0.018172519036990353 | 0.01674074464909392 | 0.020697473319336357 | 0.10914645190871897 | 0.10838065706555511 | 0.14479206410081003 | 11 | 0.0 | 0.0 | 1.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| slow_source_replay_400x0p08 | AdamW | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | SGD | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | Momentum | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | NoOpMatchedOverhead | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | RandomMatchedNorm | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | StableRandom | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | SameAtomsRandomWeights | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | SameMetricRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | SameSolverRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | SignFlipTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| slow_source_replay_400x0p08 | CorruptTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | FU | 0 |  |  |  |  |  | -0.009414347164527936 | -0.005887468310996846 | -0.02417679850134713 | 0.11868899979347936 | -0.04190032519124387 | -0.08986921135786885 | -0.010947286386382427 | 10 | 0.09090909090909091 | 0.0 | 1.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| damped_source_replay_400x0p05_lr0p5 | AdamW | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | SGD | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | Momentum | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | NoOpMatchedOverhead | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | RandomMatchedNorm | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | StableRandom | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | SameAtomsRandomWeights | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | SameMetricRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | SameSolverRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | SignFlipTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| damped_source_replay_400x0p05_lr0p5 | CorruptTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | FU | 0 |  |  |  |  |  | 0.011235208684212017 | 0.034117748811790305 | -0.04788195703255105 | -0.37118170681596374 | -0.0770961016031928 | 0.02718042915083141 | 0.1660842995582732 | 10 | 0.0 | 0.0 | 1.0 | 0 | 0 | source_horizon_or_control_or_debt_gate_failed |
| source_replay_200x0p04 | AdamW | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | SGD | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | Momentum | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | NoOpMatchedOverhead | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | RandomMatchedNorm | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | StableRandom | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | SameAtomsRandomWeights | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | SameMetricRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | SameSolverRandomTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | SignFlipTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| source_replay_200x0p04 | CorruptTarget | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

_仅显示前 48 / 180 rows；完整 CSV 见 artifact。_


### S6 KAN Matrix

| carrier | mechanism | attempt | basis_estimate_fit_cos | basis_estimate_projection_residual | KAN_source_vs_best_control_h100 | KAN_source_vs_best_control_h400 | KAN_source_vs_best_control_h800 | KAN_source_vs_best_control_h1600 | KAN_source_vs_best_control_h3200 | KAN_source_vs_best_control_h4800 | R4800_over_3200 | KAN_specific_delta_vs_MLP_same_metric | row_positive_count_h3200 | KAN_source_channel_decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | readout_source_state_replay_lr0 | baseline_50x0p08 |  |  | -17.556397557258606 | -25.05995872616768 | -35.02914874255657 | -54.926550164818764 | -94.6772990822792 | -134.4127247929573 | 1.4196932749015798 | -94.90714262352148 | 2 | KANSourceChannelMismatchConfirmed | KAN_specific_delta_or_source_horizon_gate_failed |
| D-CHE | readout_source_state_replay_lr0 | slower_replay_100x0p16_repair |  |  | -25.21000088751316 | -35.893466383218765 | -50.086476758122444 | -78.41117595136166 | -134.99434113502502 | -191.55466245114803 | 1.4189829058060281 | -135.2241846762673 | 1 | KANSourceChannelMismatchConfirmed | KAN_specific_delta_or_source_horizon_gate_failed |
| D-FOU | readout_source_state_replay_lr0 | baseline_50x0p08 |  |  | -1.5671654045581818 | -1.5235587060451508 | -1.4252634942531586 | -1.2500203289091587 | -1.5272227600216866 | -1.7928098402917385 | 1.1739019920488094 | -1.7570663012639618 | 3 | KANSourceChannelMismatchConfirmed | KAN_specific_delta_or_source_horizon_gate_failed |
| D-FOU | readout_source_state_replay_lr0 | stronger_replay_50x0p12_repair |  |  | -2.1677739024162292 | -2.269640624523163 | -2.3373308181762695 | -3.080625541508198 | -4.552779771387577 | -6.008887253701687 | 1.3198282270241075 | -4.782623312629852 | 2 | KANSourceChannelMismatchConfirmed | KAN_specific_delta_or_source_horizon_gate_failed |
| D-FOU | readout_source_state_replay_lr0 | slower_replay_100x0p16_repair |  |  | 0.35188810899853706 | 0.38552435487508774 | 0.3853251412510872 | 0.37123068422079086 | 0.33211544156074524 | 0.2906392142176628 | 0.8751150288340439 | 0.10227190031847 | 5 | KANRetainedSourceOpened |  |
| D-CHE | basis_estimate_readout_commit_lr0 | basis_estimate_readout_commit_400x1e-3_replay100x0p16 | 0.9924492239952087 | 0.12510734704750215 | 0.17864945530891418 | 0.1619080901145935 | 0.18887484073638916 | 0.006273061037063599 | -1.0442840084433556 | -1.8978360071778297 | 1.8173561903019153 | -1.2741275496856308 | 5 | KANSourceChannelMismatchConfirmed | KAN_specific_delta_or_source_horizon_gate_failed |
| D-FOU | basis_estimate_readout_commit_lr0 | basis_estimate_readout_commit_400x1e-3_replay100x0p16 | 0.9993331432342529 | 0.03672050540500659 | 0.1524980515241623 | 0.09488344192504883 | 0.09299200773239136 | -0.542513657361269 | -2.0140087520703673 | -2.379581466317177 | 1.1815149581008557 | -2.2438522933126426 | 3 | KANSourceChannelMismatchConfirmed | KAN_specific_delta_or_source_horizon_gate_failed |


## 4GPU Queue

| task_id | line | gpu | command | status | returncode | start_time | end_time | duration_sec | log_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_basis_efficiency_closure | S1 | 3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_basis_efficiency_closure.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:46:10 +0800 | 2026-06-07 17:46:12 +0800 | 2.001213550567627 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S1_basis_efficiency_closure.log |
| S0_17_code_packet_truth_gate | S0.17 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:46:10 +0800 | 2026-06-07 17:46:19 +0800 | 9.003339529037476 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S0_17_code_packet_truth_gate.log |
| S2_source_atom_generation | S2 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_source_atom_generation.py --seed 2210 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:46:19 +0800 | 2026-06-07 17:46:21 +0800 | 2.0016863346099854 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S2_source_atom_generation.log |
| S3_variational_source_solve | S3 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:46:21 +0800 | 2026-06-07 17:46:23 +0800 | 2.001922369003296 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S3_variational_source_solve.log |
| S4_metric_dynamics_commit | S4 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:46:23 +0800 | 2026-06-07 17:46:25 +0800 | 2.001931667327881 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S4_metric_dynamics_commit.log |
| S5_horizon_source_formation | S5 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:46:25 +0800 | 2026-06-07 17:49:31 +0800 | 186.01335430145264 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S5_horizon_source_formation.log |
| S6_kan_mapping_gate | S6 | 2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:49:31 +0800 | 2026-06-07 17:49:36 +0800 | 5.002397298812866 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S6_kan_mapping_gate.log |
| S7_finalize_packet_recap | S7 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | 0 | 2026-06-07 17:49:36 +0800 | 2026-06-07 17:49:37 +0800 | 1.0018246173858643 | /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_packet_recap.log |

| execution_contract_violation | reason | max_idle_gap_sec |
| --- | --- | --- |
| 0 | no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min | 0 |


### Command Journal / Post-Full Reruns

| timestamp | command | status | note |
| --- | --- | --- | --- |
| 2026-06-07 17:46:22 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=S3-VariationalSourceSolvePass pass_rows=1 |
| 2026-06-07 17:46:24 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_metric_dynamics_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=S4-MetricDynamicsCommitPass pass_rows=1 blocker= |
| 2026-06-07 17:49:30 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210 | completed | route=S5-HorizonSourceFormationPass c3=4 c4=4 |
| 2026-06-07 17:49:36 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210 | completed | decision=KANRetainedSourceOpened blocker= |
| 2026-06-07 17:49:37 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=86 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 17:49:37 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_full.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --drat-device cuda:3 --seed 2210 | completed | queue_drained=1 blocked=0 |
| 2026-06-07 17:49:38 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 17:49:38 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | post_queue_manifest_finalize_returncode=0 log=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/logs/S7_finalize_after_queue_manifest.log |
| 2026-06-07 17:52:11 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 17:53:16 +0800 | rm -rf /tmp/v22_10_final_packet_check && mkdir -p /tmp/v22_10_final_packet_check && unzip -q results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip -d /tmp/v22_10_final_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_10_final_packet_check/dgkan /tmp/v22_10_final_packet_check/experiments | completed | final v22_10_code_review_packet.zip clean unzip compileall_rc=0 |
| 2026-06-07 17:53:37 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 17:58:46 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_s017_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | S0.17=1 code_route=R0-CodePacketSelfContained missing_zip=0 clean_import=0 kernel_consistency=1 |
| 2026-06-07 18:02:28 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_horizon_source_formation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210 | completed | route=S5-HorizonSourceFormationPass c3=7 c4=7 |
| 2026-06-07 18:03:14 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 --seed 2210 | completed | decision=KANRetainedSourceOpened blocker= |
| 2026-06-07 18:03:31 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 18:04:17 +0800 | rm -rf /tmp/v22_10_pid_packet_check && mkdir -p /tmp/v22_10_pid_packet_check && unzip -q results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip -d /tmp/v22_10_pid_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_10_pid_packet_check/dgkan /tmp/v22_10_pid_packet_check/experiments | completed | post-PID v22_10_code_review_packet.zip clean unzip compileall_rc=0 |
| 2026-06-07 18:04:18 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 19:25:12 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_10_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10 | completed | route=R8-KANRetainedSourceOpened artifacts=91 bundle=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_results_bundle.zip code_review_packet=/home/chengshun.wang/DG-LCA/results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip |
| 2026-06-07 19:26:31 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v22_10_finalize.py | completed | expanded v22.10 recap generator syntax check returncode=0 |
| 2026-06-07 19:26:52 +0800 | rm -rf /tmp/v22_10_rich_recap_packet_check && mkdir -p /tmp/v22_10_rich_recap_packet_check && unzip -q results/v22_10_constructive_retained_source_functional_update_basis_efficiency_4gpu/official_v22_10/v22_10_code_review_packet.zip -d /tmp/v22_10_rich_recap_packet_check && /home/chengshun.wang/miniconda3/envs/kan/bin/python -m compileall -q /tmp/v22_10_rich_recap_packet_check/dgkan /tmp/v22_10_rich_recap_packet_check/experiments | completed | rich-recap v22_10_code_review_packet.zip clean unzip compileall_rc=0 |


## Failure Taxonomy

| blocker | route | repair_or_next_direction |
| --- | --- | --- |
| optimization_loss_agnostic_efficiency_gate | R8-KANRetainedSourceOpened | do not run or promote CE-targeted trainpath repairs; implement an arbitrary-loss/upstream-gradient efficiency runner before any S1 closure claim |
| D-CHE_D-FOU_efficiency_reconfirm_gate | R8-KANRetainedSourceOpened | implement arbitrary-loss/upstream-gradient S1 efficiency runner before any basis-efficiency repair or promotion |
| D-RAT_D-RBF_robust_efficiency_gate | R8-KANRetainedSourceOpened | after loss-agnostic efficiency runner exists, continue D-RAT numerator/denominator reciprocal fusion, singlelaunch/launch-meta forward repair, D-RBF singlelaunch/local backward, and low-hidden/blockH/fastK2/blockB/combo scans under the same robust gate |


## 修改记录

- 新增并改造 `dgkan/fu/source_atoms.py`：实现 v22.10 train-stream source atom generator、controls、NDS/DDR/commutator/row metrics 与 unit tests；正式 source direction 改为 label-free logit geometry，不使用 labels 或 supervised loss。
- 新增 A10 `split_consensus` 与 A11 `causal_split_transport` loss-agnostic atoms：只用 train logits 的 split-consensus / causal split transport 构造方向，并写入 `loss_agnostic_contract_pass=1`、`uses_labels_for_direction=0`、`uses_loss_for_direction=0`。
- 修复 source atom commutator / NDS proxy：order-invariant atom 不再被样本顺序翻转误罚；loss-agnostic geometry NDS 改为 logit-channel curvature，避免把样本标签顺序当成 source 信息。
- 新增 `dgkan/fu/variational_source_solver.py` loss-agnostic path：实现 S3 sparse/low-NDS/control-null/block-restricted/control-balanced repair ladder，并对 DDR 做 capped scoring，避免高 DDR 但高 control projection 的 atom 覆盖低投影 source。
- 改造 `dgkan/fu/constructive_commit.py`：S4 正式路径以 `y=None` 运行，不计算 supervised loss/CE 梯度；B2 transfer 改为 target displacement readback geometry，与 matched random/corrupt target controls 比较。
- 改造 S5 lightweight train-stream horizon runner：执行 h100-h6400、11 个 matched controls、source replay / damped replay / source-state replay / finite-window source-state replay repair ladder；C3/C4 只看 label-free target-retention score 与 residual/geometry debt。
- 新增 S5 PID functional-update repair ladder：以 label-free target projected-gain error 作为 PID 误差信号，只调节 source update 注入幅度；matched random/sign/corrupt/same-solver controls 使用同一 PID controller，避免 source 独享控制器优势。
- 改造 S6 lightweight KAN source-channel mapping：KAN mapping 使用 label-free target-retention score、basis-estimate fit cosine 和 matched param controls；不使用 CE loss improvement 判定 source success。
- 按用户 clarified hard constraint 改造 `experiments/run_v22_10_basis_efficiency_closure.py`：正式 S1 在进入 CE-targeted trainpath 前 fail-closed，写出 `optimization_loss_agnostic_contract_pass=0` 与 `ce_targeted_efficiency_rows_invalidated=1`；旧 manual-CE D-RAT/D-RBF repair scan 只保留为 legacy code/audit context，不执行、不 promotion。
- CE 时代新增过的 D-RAT/D-RBF K2、singlelaunch、blockH、fastK2、blockB、combo 和 component telemetry probe 代码仍进入 code packet 供审计，但在 v22.10 clarified contract 下不构成有效优化证据；下一步必须实现 arbitrary-loss/upstream-gradient official efficiency runner 后才能重新测这些方向。
- 新增 `experiments/run_v22_10_*` runner：S0.17 truth gate、basis efficiency wrapper、source atom、variational solve、commit、horizon、KAN、dynamic queue、final recap 与 packet/bundle。
- v22.10 S2 fallback 真实执行了 lower norm / more split / block-restricted attempts；S3 fallback 真实执行了 l1 sparse、low-NDS-only、control-null-only、block-restricted 与 balanced-grid attempts。

## 补充分析 / Evidence Chain

- v22.10 的语义从 observer-as-filter 改成 constructive source generation；本轮没有使用 v22.09 的 future audit label 生成方向。
- 用户追加 hard constraint 后，v22.10 的 loss-agnostic 解释为所有优化不得针对 CE：source/C3/KAN route 均必须写出 `loss_agnostic_contract_pass=1` 且 `uses_labels_for_direction=0`、`uses_loss_for_direction=0`；S1 efficiency 也不得运行或提升 CE-targeted trainpath。
- S0.17 必须先闭合 code packet：clean unzip compile/import、required CSV vs zip、semantic forbidden audit、source atom/variational/commit unit tests、kernel status consistency 都落盘；若此处失败，不能写 scientific no-go。
- Basis efficiency 在本次 clarified contract 下必须 fail-closed：当前 official trainpath 仍围绕 CE/manual-CE backward 组织，缺少 arbitrary-loss/upstream-gradient S1 runner，因此 D-CHE/D-FOU 与 D-RAT/D-RBF 都不能计入正式 efficiency closure。
- CE 时代的 D-RAT/D-RBF K2/singlelaunch/blockH/fastK2/blockB/combo/telemetry probe 只能说明代码实现方向，不再是 v22.10 有效优化证据；正式复盘不读取旧 low-hidden 或 scan artifacts 来决定 route，避免旧 CE workload 污染结论。
- 下一步 S1 的真正修复不是继续 CE duplicate-cost/tile/launch scan，而是先把 official efficiency contract 改成任意上游 `grad_logits` / arbitrary scalar loss 的 forward-backward-step runner，再在同一 robust gate 下重测 D-CHE/D-FOU/D-RAT/D-RBF。
- A10/A11 修复了 loss-agnostic S2/S3 的二难：split-consensus atom 有强几何读数但 control projection 高，causal split transport projection 更低；S3 control-balanced L1 在 capped DDR 下选择两者组合，打开 variational source solve。
- S4 的 split_A readout / hidden-readout / higher damping / optimizer-state attempts 都是真实负结果；S4.5 all_train_stream actuation repair 才以 label-free target readback 打开 commit。
- S5 普通 AdamW continuation 没有 retained source：弱 replay 后期可正但早期 margin 不够，持续 source-state replay 早期强但 h2400 后过冲。finite-window source-state replay 直接针对这个 blocker，只在早期写入 source 后停止刷新。
- S5 finite-window source-state replay 是本轮 MLP loss-agnostic functional progress 的关键：`finite_source_state_replay_400x0p20_stop800_lr0` 在 h100/h400/h800/h1600/h3200/h4800/h6400 全部大于 best matched control，h3200 margin 约 0.112，matched control positive count=11，C3/C4 都打开。该结论限定于 v22.10 lightweight train-stream runner，不等价于 S1 efficiency closure。
- S5 PID control repair 已真实执行：PID rows=3，PID C3/C4 pass=3/3，best PID=`pid_source_state_replay_100_kp0p16_ki0p02_kd0p08_stop1600_lr0`，h3200 margin=0.22984354124227524。PID 结果只使用 target displacement 几何误差，不使用 CE 或 labels。
- AUCtime 在本 lightweight runner 中记录同 step budget 时间比，因此为 1.0；source advantage 的面积另列为 `source_target_retention_area_ratio`，不作为 debt blocker。
- S6 readout-only D-CHE 与多数 D-FOU replay rows 是真实负结果；本轮 opening 来自 D-FOU `slower_replay_100x0p16_repair`。basis-estimate/readout-commit rows 虽有高 fit cosine，但 KAN-specific delta 为负，不能写成 basis-estimate opening。
- 最新结论：route=`R8-KANRetainedSourceOpened`，minimum_effective_progress=`A-CodeClosure;C-SourceAtomProgress;D-ConstructiveSolveProgress;E-FunctionalProgress;F-KANSourceProgress`。本轮有效进展是 loss-agnostic A10/A11 source atoms、S4 all-train actuation repair、finite-window 与 PID MLP source-state replay C3/C4、D-FOU slower replay KAN source-channel opening；最终仍不能 official promotion，因为 blocker 仍未清空；若 blocker 是 D-RAT/D-RBF robust efficiency side gate，则只能继续 efficiency side-line repair，不能把 KAN/MLP source success 单独写成 promotion。

_完整 artifact 清单保留在 `v22_10_artifact_index.csv`，不放入复盘正文。_
