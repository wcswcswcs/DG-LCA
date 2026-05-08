# DG-KAN v5.2 Memory-Budgeted Residual Smoothing 结果复盘

本轮依据 `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_实验计划.md`。目标是保留 AB-RBF 作为 PureKAN 默认 edge primitive，同时把 v5.1 的 strong residual smoothing acceptor 降级为低频、轻量、显存预算内的 geometry maintenance。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 memory smoke | 15 | 0 |
| P1 single smoothing | 45 | 0 |
| P2 one-cycle | 12 | 0 |
| P3 event-driven | 12 | 0 |
| P4 selection | 1 | 0 |
| P5 confirm5 | 1 | 0 |
| P6 confirm10 | 1 | 0 |
| P7 CIFAR precheck | 1 | 0 |
| P8 failures | 30 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v52.py
  Added memory/time instrumentation for AB-RBF and residual smoothing events.
  Added light coefficient-only residual smoothing eta search.
  Added P1 single-event, P2 one-cycle, and P3 event-driven multi-cycle probes.
  StrongSmooth is retained only as offline reference.

experiments/analyze_gafu_v52.py
  Generates memory/gate summaries, failure taxonomy, figures, aggregate decision, and this replay.
```

## P0 Memory Smoke

| dataset | method | peak MB | mem ratio | smooth sec | memory pass |
|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 64.2 | 1.0000 | 0.000 | yes |
| Fashion-MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.1179 | 0.038 | yes |
| Fashion-MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.1179 | 0.038 | yes |
| Fashion-MNIST | ABRBF-StrongSmooth-reference-smoke | 241.0 | 3.7551 | 2.681 | no |
| Fashion-MNIST | RBFOnly-AdamW | 59.8 | 0.9323 | 0.000 | yes |
| KMNIST | ABRBF-AdamW | 64.2 | 1.0000 | 0.000 | yes |
| KMNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.1179 | 0.039 | yes |
| KMNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.1179 | 0.039 | yes |
| KMNIST | ABRBF-StrongSmooth-reference-smoke | 241.0 | 3.7551 | 2.842 | no |
| KMNIST | RBFOnly-AdamW | 59.8 | 0.9323 | 0.000 | yes |
| MNIST | ABRBF-AdamW | 54.3 | 1.0000 | 0.000 | yes |
| MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.3206 | 0.047 | no |
| MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.3206 | 0.039 | no |
| MNIST | ABRBF-StrongSmooth-reference-smoke | 241.0 | 4.4360 | 2.761 | no |
| MNIST | RBFOnly-AdamW | 59.8 | 1.1013 | 0.000 | yes |

P0 verdict: core invariants passed. LightSmooth reduced runtime by roughly two orders of magnitude versus StrongSmooth and used far less memory; StrongSmooth remains offline-only. A first CUDA MNIST smoke row is a borderline memory-ratio outlier, but P1/P2 teacher-relative light smoothing stays within budget.

## P1 Single Smoothing Event

| dataset | teacher | proposal | controller | acc drop | KL | logit | phiR red | curvR red | mem | P1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8527 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8558 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8527 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8558 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8399 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8399 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8401 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8401 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0004 | 0.0223 | 0.2273 | 0.3979 | 0.8280 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0004 | 0.0223 | 0.2273 | 0.3979 | 0.8370 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8527 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8558 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8527 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8558 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8399 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8399 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8401 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8401 | yes |

P1 all-dataset survivors:

```text
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|scoreOnly
PureKAN-ABRBF-silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|scoreOnly
```

P1 verdict: light smoothing works as a single event. `S0-laplacian` and `S1-sobolev-grad` with logit/score gates preserve task outputs while reducing residual geometry. StrongSmooth does not pass the memory/task gate.

## P2 One-Cycle Train/Smooth/Refresh

| dataset | method | acc | gap | phiR red | curvR red | recovery | retention | mem | P2 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 0.7949 | 0.0000 |  |  |  |  |  | yes |
| Fashion-MNIST | ABRBF-LightSmooth-logitOnly-oneCycle | 0.7988 | -0.0039 | 0.0520 | 0.5365 | 6.0000 | 0.7691 | 1.0081 | yes |
| Fashion-MNIST | ABRBF-LightSmooth-scoreOnly-oneCycle | 0.7988 | -0.0039 | 0.0520 | 0.5365 | 6.0000 | 0.7691 | 1.0081 | yes |
| Fashion-MNIST | ABRBF-StrongSmooth-reference-oneCycle | 0.7988 | -0.0039 | 0.0520 | 0.5365 | 6.0000 | 0.7691 | 12.6321 | no |
| KMNIST | ABRBF-AdamW | 0.6797 | 0.0000 |  |  |  |  |  | yes |
| KMNIST | ABRBF-LightSmooth-logitOnly-oneCycle | 0.6973 | -0.0176 | 0.0510 | 0.5395 | 1.0000 | 0.7993 | 1.0081 | yes |
| KMNIST | ABRBF-LightSmooth-scoreOnly-oneCycle | 0.6973 | -0.0176 | 0.0510 | 0.5395 | 1.0000 | 0.7993 | 1.0081 | yes |
| KMNIST | ABRBF-StrongSmooth-reference-oneCycle | 0.6973 | -0.0176 | 0.0510 | 0.5395 | 1.0000 | 0.7993 | 12.6321 | no |
| MNIST | ABRBF-AdamW | 0.8770 | 0.0000 |  |  |  |  |  | yes |
| MNIST | ABRBF-LightSmooth-logitOnly-oneCycle | 0.8809 | -0.0039 | 0.0509 | 0.5377 | 1.0000 | 0.8291 | 1.1624 | yes |
| MNIST | ABRBF-LightSmooth-scoreOnly-oneCycle | 0.8809 | -0.0039 | 0.0509 | 0.5377 | 1.0000 | 0.8291 | 1.1624 | yes |
| MNIST | ABRBF-StrongSmooth-reference-oneCycle | 0.8809 | -0.0039 | 0.0774 | 0.6963 | 1.0000 | 0.8854 | 14.5657 | no |

P2 all-dataset survivors:

```text
ABRBF-AdamW
ABRBF-LightSmooth-logitOnly-oneCycle
ABRBF-LightSmooth-scoreOnly-oneCycle
```

P2 verdict: one-cycle light smoothing passes across MNIST, Fashion-MNIST, and KMNIST. Refresh can recover or preserve task performance while keeping around 5% residual phi reduction and large curvature reduction. StrongSmooth remains rejected due memory/time.

## P3 Event-Driven Multi-Cycle

| dataset | method | acc | gap | phiR red | curvR red | events | accepted | time frac | P3 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 0.8008 | 0.0000 | 0.0000 | -0.0994 | 0.0 | 0.0 | 0.0000 | no |
| Fashion-MNIST | ABRBF-LightSmooth-eventDriven | 0.8008 | 0.0000 | 0.0000 | -0.0994 | 0.0 | 0.0 | 0.0000 | no |
| Fashion-MNIST | ABRBF-LightSmooth-eventDriven-refreshStrong | 0.8008 | 0.0000 | 0.0000 | -0.0994 | 0.0 | 0.0 | 0.0000 | no |
| Fashion-MNIST | ABRBF-LightSmooth-fixedInterval | 0.7969 | 0.0039 | 0.1139 | 0.7852 | 3.0 | 3.0 | 0.0149 | yes |
| KMNIST | ABRBF-AdamW | 0.6934 | 0.0000 | 0.0000 | -0.0998 | 0.0 | 0.0 | 0.0000 | no |
| KMNIST | ABRBF-LightSmooth-eventDriven | 0.6934 | 0.0000 | 0.0000 | -0.0998 | 0.0 | 0.0 | 0.0000 | no |
| KMNIST | ABRBF-LightSmooth-eventDriven-refreshStrong | 0.6934 | 0.0000 | 0.0000 | -0.0998 | 0.0 | 0.0 | 0.0000 | no |
| KMNIST | ABRBF-LightSmooth-fixedInterval | 0.6699 | 0.0234 | 0.1233 | 0.7796 | 3.0 | 3.0 | 0.0140 | no |
| MNIST | ABRBF-AdamW | 0.8906 | 0.0000 | 0.0000 | -0.0781 | 0.0 | 0.0 | 0.0000 | no |
| MNIST | ABRBF-LightSmooth-eventDriven | 0.8906 | 0.0000 | 0.0000 | -0.0781 | 0.0 | 0.0 | 0.0000 | no |
| MNIST | ABRBF-LightSmooth-eventDriven-refreshStrong | 0.8906 | 0.0000 | 0.0000 | -0.0781 | 0.0 | 0.0 | 0.0000 | no |
| MNIST | ABRBF-LightSmooth-fixedInterval | 0.8711 | 0.0195 | 0.1000 | 0.7708 | 3.0 | 2.0 | 0.0161 | no |

P3 all-dataset survivors:

```text
none
```

P3 verdict: no multicycle survivor. Fixed-interval smoothing keeps geometry gains but costs too much accuracy on MNIST/KMNIST. Event-driven smoothing with the written plateau trigger does not fire, so it preserves accuracy but gains no geometry.

## P4-P7 Decision

```text
P4 3-seed selection: not run
Reason: P3 produced no all-dataset survivor.

P5/P6 confirm: not run
Reason: P4 was not reached.

P7 CIFAR-small precheck: not run
Reason: no unified light smoothing candidate to scale-check.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
```

Diagnosis:

```text
F1 memory too high:
  StrongSmooth clearly fails. LightSmooth is mostly inside budget, with a borderline P0 smoke outlier.

F3/F4/F5 train-time integration:
  P2 one-cycle succeeds, but P3 multicycle integration fails.

F7 KMNIST-specific failure:
  Fixed-interval P3 loses too much accuracy on KMNIST; MNIST also fails.

Controller diagnosis:
  The smoothing operator is now cheap and effective locally. The remaining blocker is event scheduling / recovery, not AB-RBF or single-event smoothing.
```

## Required Artifacts

Written under `results/v5_2/`:

```text
p0_memory_smoke.csv
p0_memory_gate_summary.csv
edge_param_manifest.csv
p1_single_smoothing_event.csv
p1_single_smoothing_gate_summary.csv
p2_one_cycle_train_smooth_refresh.csv
p2_training_trace.csv
p2_one_cycle_gate_summary.csv
p3_event_driven_multicycle.csv
p3_event_trace.csv
p3_event_gate_summary.csv
p4_candidate_selection3.csv
p5_confirm5.csv
p6_confirm10.csv
p7_cifar_small_memory_precheck.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.2 status:
  stop_after_p3_no_multicycle_survivor

What improved:
  AB-RBF remains the correct PureKAN edge primitive.
  Light residual smoothing replaces v5.1 strong acceptor for single-event and one-cycle use.
  Memory/time cost is dramatically lower than StrongSmooth.

What failed:
  Multicycle integration is not solved.
  Fixed-interval smoothing accumulates task cost.
  Event-driven smoothing did not trigger under the written plateau rule.

Conclusion:
  v5.2 upgrades residual smoothing from too-heavy strong acceptor to practical one-cycle maintenance.
  It should not yet be used as a multicycle training component.
  Next work should redesign event scheduling and refresh policy, not the AB-RBF primitive.
```
