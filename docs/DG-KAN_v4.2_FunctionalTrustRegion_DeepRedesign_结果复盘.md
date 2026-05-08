# DG-KAN v4.2 Functional Trust Region 结果复盘

本轮依据 `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_实验计划.md`。核心目标是把 v4.1 的 proposal 直接更新改成 BFT：proposal -> 临时 apply -> train/holdout loss、activation drift、logit drift 验收 -> backtrack / accept / reject。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 24 | 0 |
| P1 proposal | 108 | 0 |
| P2 single-block | 48 | 0 |
| P3 baselines+BFT | 99 | 0 |
| BFT acceptance log | 24372 | 2233 |

## Code / Config Changes

```text
experiments/run_gafu_v42.py
  Added BFT proposal generation using existing raw/Sobolev/D6/FNG/FTF/FC paths.
  Added temporary apply/rollback, train+holdout loss checks, activation/logit trust,
  backtracking, mixed proposal selection, and sequential role orders.

experiments/analyze_gafu_v42.py
  Generates v4.2 required artifacts, gate summaries, traces, figures, and this replay.
```

## P0 Implementation Smoke

```text
P0 rows = 24
errors = 0
rollback max abs error = 0.0000
strict PureKAN nonKAN params = 0
```

P0 通过：proposal、临时 apply、rollback、accept/reject/backtrack 统计均能产生有限记录。

## P1 Proposal Direction Audit

| proposal | rows | accept | bad | holdout+ | median ratio | act p95 | logit p95 | p1_pass |
|---|---|---|---|---|---|---|---|---|
| FC-whitened-Adam-one-step | 12 | 0.7500 | 0.0000 | 3 | 0.9911 | 0.0779 | 0.1362 | no |
| FC-whitened-gradient | 12 | 0.7500 | 0.0000 | 3 | 0.9994 | 0.0704 | 0.1208 | no |
| FNG-leftFullRight | 12 | 0.8333 | 0.0000 | 3 | 0.9935 | 0.0702 | 0.1070 | no |
| FTF-blocks-output | 12 | 1.0000 | 0.0000 | 3 | 0.9891 | 0.0866 | 0.0972 | yes |
| FTF-output-only | 12 | 1.0000 | 0.0000 | 3 | 0.9904 | 0.0865 | 0.0944 | yes |
| Sobolev-full | 12 | 1.0000 | 0.0000 | 3 | 0.9831 | 0.0788 | 0.0884 | yes |
| mixed-best-of-proposals | 12 | 1.0000 | 0.0000 | 3 | 0.9861 | 0.0878 | 0.0963 | yes |
| raw-gradient | 12 | 1.0000 | 0.0000 | 3 | 0.9977 | 0.0242 | 0.0820 | yes |
| task-diag-D6 | 12 | 0.7500 | 0.0000 | 3 | 0.9772 | 0.0948 | 0.1613 | no |

P1 survivors:

```text
FTF-blocks-output, FTF-output-only, Sobolev-full, mixed-best-of-proposals, raw-gradient
```

观察：

```text
BFT acceptance gate 明显压住了 v4.1 的危险方向。
FTF 在 block/output role 上仍有强 one-step descent，但 input role 基本是 no-op 或需要 mixed/raw 接管。
input role 的 FNG/D6/FC 经常因为 logit drift 超过阈值被拒绝。
```

## P2 Single-Block Accepted-Step Audit

| proposal | rows | accept | bad | holdout+ | median ratio | act p95 | logit p95 | p2_pass |
|---|---|---|---|---|---|---|---|---|
| FNG-leftFullRight | 12 | 0.7500 |  |  |  | 0.0771 | 0.1908 | no |
| FTF-blocks-output | 12 | 1.0000 |  |  |  | 0.0912 | 0.0947 | yes |
| mixed-best-of-proposals | 12 | 1.0000 |  |  |  | 0.0771 | 0.0956 | yes |
| raw-gradient | 12 | 1.0000 |  |  |  | 0.0242 | 0.0637 | yes |

P2 survivors:

```text
FTF-blocks-output, mixed-best-of-proposals, raw-gradient
```

观察：

```text
single-block 层面，FTF / mixed / raw 都能在 trust gate 内得到正 holdout descent。
FNG 对 hidden/output 比较安全，但 input FNG 会因为 logit drift 被拒绝。
这说明 BFT 的验收机制确实阻止了 v4.1 的 FTF 长程爆炸第一步。
```

## P3 Sequential BFT Micro-Run

| dataset | method | runs | acc | std | gap vs AdamW | AUC vs D6 | accept | fallback | act p95 | logit p95 | P3 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | BFT-FNG-forward | 3 | 0.7400 | 0.0265 | 0.0800 | -0.1274 | 0.9045 | 0.0955 | 0.0912 | 0.1305 | 0 |
| Fashion-MNIST | BFT-FNG-reverse | 3 | 0.7460 | 0.0306 | 0.0740 | -0.1454 | 0.9219 | 0.0781 | 0.0858 | 0.1248 | 0 |
| Fashion-MNIST | BFT-FTF-forward | 3 | 0.7033 | 0.0182 | 0.1167 | -0.1139 | 0.9080 | 0.0920 | 0.0866 | 0.1054 | 0 |
| Fashion-MNIST | BFT-FTF-reverse | 3 | 0.7263 | 0.0105 | 0.0937 | -0.1053 | 0.9809 | 0.0191 | 0.0827 | 0.0940 | 0 |
| Fashion-MNIST | BFT-mixed-forward | 3 | 0.7297 | 0.0226 | 0.0903 | 0.0890 | 1.0000 | 0.0000 | 0.0824 | 0.0874 | 0 |
| Fashion-MNIST | BFT-mixed-output-first | 3 | 0.7253 | 0.0252 | 0.0947 | 0.0657 | 1.0000 | 0.0000 | 0.0841 | 0.0866 | 0 |
| Fashion-MNIST | BFT-mixed-reverse | 3 | 0.7373 | 0.0184 | 0.0827 | 0.0683 | 1.0000 | 0.0000 | 0.0848 | 0.0876 | 0 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7593 | 0.0296 | 0.0607 | -0.1722 |  |  |  |  | 1 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.7827 | 0.0154 | 0.0373 | 0.0000 |  |  |  |  | 1 |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 0.7930 | 0.0164 | 0.0270 | 0.2413 |  |  |  |  | 1 |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0.8200 | 0.0086 | 0.0000 | 0.2862 |  |  |  |  | 1 |
| KMNIST | BFT-FNG-forward | 3 | 0.3837 | 0.0189 | 0.3407 | -0.3030 | 0.7526 | 0.2474 | 0.0875 | 0.1507 | 0 |
| KMNIST | BFT-FNG-reverse | 3 | 0.4223 | 0.0450 | 0.3020 | -0.2699 | 0.8325 | 0.1675 | 0.0882 | 0.1463 | 0 |
| KMNIST | BFT-FTF-forward | 3 | 0.4127 | 0.0045 | 0.3117 | -0.1222 | 0.8741 | 0.1259 | 0.0904 | 0.1218 | 0 |
| KMNIST | BFT-FTF-reverse | 3 | 0.4250 | 0.0102 | 0.2993 | -0.1106 | 0.9575 | 0.0425 | 0.0868 | 0.0995 | 0 |
| KMNIST | BFT-mixed-forward | 3 | 0.4913 | 0.0137 | 0.2330 | -0.0259 | 1.0000 | 0.0000 | 0.0818 | 0.0939 | 0 |
| KMNIST | BFT-mixed-output-first | 3 | 0.4900 | 0.0115 | 0.2343 | -0.0508 | 1.0000 | 0.0000 | 0.0833 | 0.0925 | 0 |
| KMNIST | BFT-mixed-reverse | 3 | 0.4907 | 0.0103 | 0.2337 | -0.0389 | 1.0000 | 0.0000 | 0.0815 | 0.0932 | 0 |
| KMNIST | D0-allFullSobolev | 3 | 0.5367 | 0.0237 | 0.1877 | -0.0882 |  |  |  |  | 1 |
| KMNIST | D6-allTaskAware | 3 | 0.5383 | 0.0180 | 0.1860 | 0.0000 |  |  |  |  | 1 |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.6630 | 0.0028 | 0.0613 | 0.3186 |  |  |  |  | 1 |
| KMNIST | PureKAN-AdamW | 3 | 0.7243 | 0.0111 | 0.0000 | 0.4216 |  |  |  |  | 1 |
| MNIST | BFT-FNG-forward | 3 | 0.6540 | 0.0022 | 0.2463 | -0.0152 | 0.7500 | 0.2500 | 0.1690 | 0.2146 | 0 |
| MNIST | BFT-FNG-reverse | 3 | 0.6503 | 0.0110 | 0.2500 | -0.0347 | 0.7500 | 0.2500 | 0.1530 | 0.1949 | 0 |
| MNIST | BFT-FTF-forward | 3 | 0.6433 | 0.0155 | 0.2570 | 0.1077 | 0.6988 | 0.3012 | 0.0933 | 0.2033 | 0 |
| MNIST | BFT-FTF-reverse | 3 | 0.6520 | 0.0024 | 0.2483 | 0.1301 | 0.7491 | 0.2509 | 0.0912 | 0.1850 | 0 |
| MNIST | BFT-mixed-forward | 3 | 0.7060 | 0.0237 | 0.1943 | 0.1565 | 1.0000 | 0.0000 | 0.0797 | 0.0922 | 0 |
| MNIST | BFT-mixed-output-first | 3 | 0.7137 | 0.0259 | 0.1867 | 0.1402 | 1.0000 | 0.0000 | 0.0855 | 0.0918 | 0 |
| MNIST | BFT-mixed-reverse | 3 | 0.7277 | 0.0323 | 0.1727 | 0.1610 | 1.0000 | 0.0000 | 0.0811 | 0.0939 | 0 |
| MNIST | D0-allFullSobolev | 3 | 0.6800 | 0.0349 | 0.2203 | 0.0269 |  |  |  |  | 1 |
| MNIST | D6-allTaskAware | 3 | 0.6913 | 0.0300 | 0.2090 | 0.0000 |  |  |  |  | 1 |
| MNIST | F4-FNG-leftFullRight | 3 | 0.8400 | 0.0333 | 0.0603 | 0.4325 |  |  |  |  | 1 |
| MNIST | PureKAN-AdamW | 3 | 0.9003 | 0.0076 | 0.0000 | 0.4301 |  |  |  |  | 1 |

Best BFT points:

```text
MNIST: BFT-mixed-reverse acc=0.7277, gap=0.1727
Fashion: BFT-FNG-reverse acc=0.7460, gap=0.0740
KMNIST: BFT-mixed-forward acc=0.4913, gap=0.2330
```

P3 verdict:

```text
No BFT candidate passed the P3 joint gate.

BFT successfully prevents catastrophic FTF divergence:
  no chance-accuracy collapse like v4.1 FTF
  acceptance/logit/activation traces remain finite

But BFT is too conservative or direction-weak:
  MNIST best BFT remains far below PureKAN-AdamW
  Fashion best BFT remains below PureKAN-AdamW and FNG baseline
  KMNIST best BFT improves over D0/D6 but remains far below PureKAN-AdamW and F4-FNG baseline
```

## P4-P8 Decision

```text
P4 proposal/order ablation: not run.
Reason: P3 produced no survivor.

P5 temporal dynamics: not run.
Reason: no P4 candidate.

P6 3-seed full-budget selection: not run.
Reason: P3 micro-run did not satisfy entry gate.

P7/P8 confirm: not run.
Reason: P6 was not reached.
```

## Required Artifacts

Written under `results/v4_2/`:

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p2_single_block_acceptance.csv
p3_sequential_micro_scorecard.csv
p4_ablation_scorecard.csv
p5_temporal_dynamics_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
bft_acceptance_log.csv
bft_rejection_reason_log.csv
role_update_trace.csv
proposal_choice_trace.csv
activation_drift_trace.csv
logit_drift_trace.csv
metric_norm_trace.csv
eta_backtracking_trace.csv
failure_table.csv
aggregate_decision.json
figures/p3_kmnist_bft_acc.svg
figures/p3_fashion_bft_acc.svg
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.2.

What is confirmed:
  1. BFT acceptance/backtracking prevents FTF-style numerical catastrophe.
  2. Proposal selection is useful: mixed candidates choose FTF for hidden blocks and FNG/raw for safer roles.
  3. The trust protocol gives finite, auditable acceptance/rejection traces.

What is not confirmed:
  1. Stability did not translate into AdamW-level representation learning.
  2. BFT candidates underfit, especially on KMNIST.
  3. P4/P5/P6 expansion is not justified.

Interpretation:
  v4.1 failed because proposal was too strong and unchecked.
  v4.2 shows the opposite side: checked proposals are stable but too weak.
  The next design needs stronger accepted directions, likely better downstream curvature or temporal dynamics,
  but only after improving P3 micro-run accuracy.
```
