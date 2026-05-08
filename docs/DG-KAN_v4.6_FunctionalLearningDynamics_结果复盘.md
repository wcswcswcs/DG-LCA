# DG-KAN v4.6 Functional Learning Dynamics 结果复盘

本轮依据 `docs/DG-KAN_v4.6_FunctionalLearningDynamics_实验计划.md`。目标是验证 PureKAN 是否需要 FLD：functional-coordinate adaptive dynamics + role-wise representation control + phased geometry + Lyapunov restart。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 FLD smoke | 60 | 0 |
| P1 AdamW envelope | 45 | 0 |
| P2 FLD dynamics | 90 | 0 |
| P3 100-step trajectory | 45 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v46.py
  Added FLD-AdamCoord, FLD-Nesterov, FLD-AdanLite, FLD-WinLite,
  FLD-LyapunovRestart, and FLD-TeacherEnvelope probes.
  Added AdamW trajectory envelope construction over seeds 0/1/2 and steps 1/5/20/50/100.
  Added phase-aware P2 gate and 100-step P3 trajectory audit.

experiments/analyze_gafu_v46.py
  Generates AdamW envelope JSON, P2/P3 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Implementation Invariants

| rows | errors | max nonKAN | max roundtrip | max u->a | max rollback | pass |
|---|---|---|---|---|---|---|
| 60 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | true |

P0 verdict: pass. FLD methods keep strict PureKAN trainable parameters, coefficient coverage, rollback, and finite optimizer states.

## P1 AdamW Trajectory Envelope

| dataset | holdout Δ | rank ratio | phi ratio | input share | block share | output share |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 1.7950 | 0.7919 | 1.1583 | 0.6774 | 0.2152 | 0.1074 |
| KMNIST | 1.7909 | 0.7285 | 1.1776 | 0.6796 | 0.2149 | 0.1054 |
| MNIST | 1.9020 | 0.5084 | 1.2107 | 0.6830 | 0.2131 | 0.1039 |

Observation:

```text
AdamW is used as a profile teacher, not an exact displacement target.
The envelope records early task descent, role share, rank/margin movement, and phi budget.
```

## P2 One-Step / 20-Step Dynamics Audit

| dataset | method | runs | hold20 | hold/A | rank | margin | phi | bad | energy | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7351 | 0.6747 | 0.6151 | 0.1636 | 0.9978 | 0.0667 | 1.7270 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8802 | 0.8079 | 0.4651 | 0.2122 | 1.0027 | 0.0833 | 1.5753 | no |
| Fashion-MNIST | FCAdam-dataSob | 3 | 1.5207 | 1.3957 | 0.3345 | 1.4934 | 1.3574 | 0.0000 | 0.8650 | no |
| Fashion-MNIST | FLD-AdamCoord | 3 | 1.5223 | 1.3971 | 0.3343 | 1.5021 | 1.3614 | 0.0000 | 0.8636 | no |
| Fashion-MNIST | FLD-AdanLite | 3 | 1.5464 | 1.4193 | 0.3350 | 1.6167 | 1.3475 | 0.0000 | 0.8388 | no |
| Fashion-MNIST | FLD-LyapunovRestart | 3 | 1.4474 | 1.3284 | 0.3561 | 1.2718 | 1.2961 | 0.0333 | 0.9335 | no |
| Fashion-MNIST | FLD-Nesterov | 3 | 1.5181 | 1.3933 | 0.3108 | 1.3784 | 1.5670 | 0.0000 | 0.8849 | no |
| Fashion-MNIST | FLD-TeacherEnvelope | 3 | 1.5140 | 1.3895 | 0.3345 | 1.4679 | 1.3594 | 0.0000 | 0.8718 | no |
| Fashion-MNIST | FLD-WinLite | 3 | 1.5222 | 1.3971 | 0.3343 | 1.5019 | 1.3612 | 0.0000 | 0.8636 | no |
| Fashion-MNIST | PureKAN-AdamW | 3 | 1.0896 | 1.0000 | 0.1896 | 0.3116 | 1.0506 | 0.0000 | 1.3827 | yes |
| KMNIST | D0-allFullSobolev | 3 | 0.3708 | 0.7117 | 0.7213 | -0.2172 | 0.9978 | 0.1000 | 2.2615 | no |
| KMNIST | D6-allTaskAware | 3 | 0.4430 | 0.8502 | 0.5752 | -0.1031 | 1.0023 | 0.1167 | 2.1347 | no |
| KMNIST | FCAdam-dataSob | 3 | 1.3247 | 2.5427 | 0.3555 | 1.2697 | 1.3863 | 0.0000 | 1.0720 | no |
| KMNIST | FLD-AdamCoord | 3 | 1.3255 | 2.5442 | 0.3560 | 1.2729 | 1.3862 | 0.0000 | 1.0712 | no |
| KMNIST | FLD-AdanLite | 3 | 1.3529 | 2.5968 | 0.3574 | 1.3924 | 1.3747 | 0.0000 | 1.0431 | no |
| KMNIST | FLD-LyapunovRestart | 3 | 1.1355 | 2.1794 | 0.3242 | 0.7756 | 1.3141 | 0.0500 | 1.2685 | no |
| KMNIST | FLD-Nesterov | 3 | 1.2662 | 2.4305 | 0.3052 | 1.0194 | 1.5936 | 0.0000 | 1.1533 | no |
| KMNIST | FLD-TeacherEnvelope | 3 | 1.3148 | 2.5238 | 0.3549 | 1.2453 | 1.3820 | 0.0000 | 1.0818 | no |
| KMNIST | FLD-WinLite | 3 | 1.3254 | 2.5441 | 0.3560 | 1.2728 | 1.3859 | 0.0000 | 1.0712 | no |
| KMNIST | PureKAN-AdamW | 3 | 0.5210 | 1.0000 | 0.1775 | 0.0243 | 1.0419 | 0.0000 | 2.0576 | yes |
| MNIST | D0-allFullSobolev | 3 | 0.3983 | 1.5019 | 0.6701 | -0.0624 | 0.9991 | 0.1333 | 2.1495 | no |
| MNIST | D6-allTaskAware | 3 | 0.2977 | 1.1225 | 0.5569 | -0.3066 | 1.0005 | 0.1333 | 2.3780 | no |
| MNIST | FCAdam-dataSob | 3 | 1.4264 | 5.3783 | 0.3187 | 1.5307 | 1.4314 | 0.0000 | 0.9661 | no |
| MNIST | FLD-AdamCoord | 3 | 1.4267 | 5.3795 | 0.3192 | 1.5319 | 1.4345 | 0.0000 | 0.9659 | no |
| MNIST | FLD-AdanLite | 3 | 1.4673 | 5.5328 | 0.3243 | 1.6747 | 1.4171 | 0.0000 | 0.9234 | no |
| MNIST | FLD-LyapunovRestart | 3 | 1.2339 | 4.6525 | 0.3277 | 1.1394 | 1.3529 | 0.0500 | 1.1537 | no |
| MNIST | FLD-Nesterov | 3 | 1.3943 | 5.2573 | 0.2810 | 1.4385 | 1.6718 | 0.0167 | 1.0268 | no |
| MNIST | FLD-TeacherEnvelope | 3 | 1.4240 | 5.3693 | 0.3187 | 1.5275 | 1.4316 | 0.0000 | 0.9685 | no |
| MNIST | FLD-WinLite | 3 | 1.4266 | 5.3791 | 0.3192 | 1.5316 | 1.4342 | 0.0000 | 0.9660 | no |
| MNIST | PureKAN-AdamW | 3 | 0.2652 | 1.0000 | 0.2771 | -0.0603 | 1.0368 | 0.0167 | 2.3188 | yes |

Best non-Adam points by 20-step holdout descent:

| dataset | best hold20 | hold20 | hold/A | rank | margin | phi |
|---|---|---|---|---|---|---|
| MNIST | FLD-AdanLite | 1.4673 | 5.5328 | 0.3243 | 1.6747 | 1.4171 |
| Fashion-MNIST | FLD-AdanLite | 1.5464 | 1.4193 | 0.3350 | 1.6167 | 1.3475 |
| KMNIST | FLD-AdanLite | 1.3529 | 2.5968 | 0.3574 | 1.3924 | 1.3747 |

P2 survivors:

```text
none
```

P2 verdict:

```text
P2 uses the new phase-aware gate. Cos/R2 are diagnostics only.
Survivors enter P3; if there are no survivors the plan forces FCAdam-dataSob,
FLD-AdanLite, and FLD-LyapunovRestart as diagnostic P3 methods.
```

## P3 100-Step Trajectory Audit

| dataset | method | runs | hold100 | hold/A | energy/D6 | rank | margin | phi | restart | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D6-allTaskAware | 3 | 1.6296 | 0.9001 | 1.0000 | 0.5145 | 2.0161 | 1.0032 | 0.0000 | yes |
| Fashion-MNIST | FCAdam-dataSob | 3 | 1.6750 | 0.9251 | 1.0660 | 0.3733 | 5.7657 | 1.8271 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | 3 | 1.6577 | 0.9156 | 1.0818 | 0.3610 | 5.7757 | 1.7921 | 0.0000 | no |
| Fashion-MNIST | FLD-LyapunovRestart | 3 | 1.7122 | 0.9457 | 0.9333 | 0.3955 | 4.6721 | 1.4836 | 18.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | 3 | 1.8105 | 1.0000 | 0.7777 | 0.3541 | 3.7258 | 1.1606 | 0.0000 | yes |
| KMNIST | D6-allTaskAware | 3 | 1.0744 | 0.5887 | 1.0000 | 0.5718 | 0.6274 | 1.0032 | 0.0000 | yes |
| KMNIST | FCAdam-dataSob | 3 | 1.5877 | 0.8700 | 0.6763 | 0.3509 | 5.1524 | 1.8420 | 0.0000 | no |
| KMNIST | FLD-AdanLite | 3 | 1.5786 | 0.8650 | 0.6771 | 0.3488 | 5.0418 | 1.8032 | 0.0000 | no |
| KMNIST | FLD-LyapunovRestart | 3 | 1.6124 | 0.8835 | 0.6140 | 0.3793 | 4.0932 | 1.5435 | 18.0000 | no |
| KMNIST | PureKAN-AdamW | 3 | 1.8250 | 1.0000 | 0.4247 | 0.4465 | 3.8677 | 1.1876 | 0.0000 | yes |
| MNIST | D6-allTaskAware | 3 | 0.9263 | 0.4749 | 1.0000 | 0.4878 | 0.2327 | 1.0057 | 0.0000 | yes |
| MNIST | FCAdam-dataSob | 3 | 1.8099 | 0.9279 | 0.4167 | 0.3341 | 6.4866 | 1.8379 | 0.0000 | no |
| MNIST | FLD-AdanLite | 3 | 1.7887 | 0.9171 | 0.4256 | 0.3323 | 6.3310 | 1.8034 | 0.0000 | no |
| MNIST | FLD-LyapunovRestart | 3 | 1.8290 | 0.9377 | 0.3725 | 0.3553 | 5.3991 | 1.5785 | 18.0000 | no |
| MNIST | PureKAN-AdamW | 3 | 1.9505 | 1.0000 | 0.2717 | 0.3876 | 4.4335 | 1.1988 | 0.0000 | yes |

P3 survivors:

```text
none
```

## P4-P7 Decision

```text
P4 phase-controller ablation: not run
P5 3-seed short full-budget selection: not run
P6 5-seed confirm: not run
P7 10-seed final confirm: not run

Reason: P3 produced no survivor
```

## P8 Failure Diagnosis

Generated:

```text
p8_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
```

Diagnosis:

```text
P2/P3 failures are classified by role share, trajectory energy, rank formation,
margin formation, geometry budget, momentum drift, restart behavior, and basis occupancy.
The dominant hard blocker in this run is not task descent: FCAdam/FLD descend strongly,
but rank remains below the Phase-I/P3 budget while phi grows far above the delayed
geometry budget by 100 steps.
```

## Required Artifacts

Written under `results/v4_6/`:

```text
p0_fld_invariants.csv
p1_adamw_trajectory_envelope.csv
p1_adamw_target_profile.csv
adamw_trajectory_envelope.json
p2_fld_dynamics_audit.csv
p2_fld_gate_summary.csv
p3_100step_trajectory.csv
p3_trajectory_gate_summary.csv
p4_phase_controller_ablation.csv
p5_short_full_budget_selection.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN Functional Learning Dynamics status:
  stop_after_p3_no_survivor

What improved:
  v4.6 no longer kills candidates solely by AdamW displacement cos/R2.
  It tests task descent, role dynamics, rank/margin formation, and phased phi budget directly.

What failed:
  See P2/P3 gate summaries and failure taxonomy.

Conclusion:
  The key question is whether stateful functional-coordinate dynamics can preserve
  enough task learning while respecting delayed geometry budgets.
```
