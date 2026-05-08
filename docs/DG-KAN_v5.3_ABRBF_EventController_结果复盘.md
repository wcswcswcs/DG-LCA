# DG-KAN v5.3 AB-RBF Event Controller 结果复盘

本轮依据 `docs/DG-KAN_v5.3_ABRBF_EventController_下一步实验计划.md`。目标是把 v5.2 的 LightSmooth 从单次/one-cycle 工具接入 budgeted event controller：geometry debt + task slack + cheap probe score + cooldown + refresh recovery。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core/memory | 15 | 0 |
| P1 single event | 45 | 0 |
| P2 probe audit | 225 | 0 |
| P3 one-cycle refresh | 21 | 0 |
| P4 event v2 | 1 | 0 |
| P5 selection | 1 | 0 |
| P6 confirm5 | 1 | 0 |
| P7 confirm10 | 1 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v53.py
  Added P2 cheap probe audit without writes.
  Added P3 one-cycle refresh policy comparison: full/base-only/RBF-frozen refresh.
  Added P4 EventV2 controller with score probe, geometry debt, cooldown, eta cap, and recovery accounting.

experiments/run_gafu_v51.py
  Aligned runner AB-RBF helper classes with core dgkan_core.ABRBFDense so P0 can audit core consistency.

experiments/analyze_gafu_v53.py
  Generates v5.3 gate summaries, figures, aggregate_decision.json, recommendation.json, and this replay.
```

## P0 Core / Memory Smoke

| dataset | method | peak MB | mem ratio | smooth sec | core | coverage | memory |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 69.1 | 1.0000 | 0.000 | yes | yes | yes |
| Fashion-MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.0388 | 0.046 | yes | yes | yes |
| Fashion-MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.0388 | 0.046 | yes | yes | yes |
| Fashion-MNIST | ABRBF-StrongSmooth-reference-smoke | 240.5 | 3.4813 | 3.005 | yes | yes | no |
| Fashion-MNIST | RBFOnly-AdamW | 60.8 | 0.8803 | 0.000 | no | no | yes |
| KMNIST | ABRBF-AdamW | 70.2 | 1.0000 | 0.000 | yes | yes | yes |
| KMNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.0218 | 0.046 | yes | yes | yes |
| KMNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.0218 | 0.046 | yes | yes | yes |
| KMNIST | ABRBF-StrongSmooth-reference-smoke | 240.5 | 3.4244 | 2.590 | yes | yes | no |
| KMNIST | RBFOnly-AdamW | 60.8 | 0.8659 | 0.000 | no | no | yes |
| MNIST | ABRBF-AdamW | 59.9 | 1.0000 | 0.000 | yes | yes | yes |
| MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.1983 | 0.055 | yes | yes | yes |
| MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.1983 | 0.046 | yes | yes | yes |
| MNIST | ABRBF-StrongSmooth-reference-smoke | 240.5 | 4.0157 | 2.920 | yes | yes | no |
| MNIST | RBFOnly-AdamW | 59.3 | 0.9898 | 0.000 | no | no | yes |

## P1 Single-Event Verification

| dataset | teacher | proposal | controller | acc drop | KL | logit | phiR | curvR | mem | P1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9186 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9129 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9326 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9129 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.8983 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.9066 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.8983 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.9066 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | StrongSmooth-reference | strongReference | 0.0020 |  | 0.0246 | 0.0632 | 0.5484 | 11.4568 | no |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | StrongSmooth-reference | strongReference | -0.0078 |  | 0.0274 | 0.0650 | 0.5497 | 11.3665 | no |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | StrongSmooth-reference | strongReference | 0.0000 |  | 0.0223 | 0.0448 | 0.4166 | 10.0109 | no |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.9036 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.8980 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.9173 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.8980 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8836 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8918 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8836 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8918 | yes |
| KMNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0064 | 0.0711 | 0.1388 | 0.8808 | yes |

P1 all-dataset survivors:
```text
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|scoreOnly
PureKAN-ABRBF-silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-silu-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-silu-AdamW|S1-sobolev-grad|scoreOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|scoreOnly
```

## P2 Probe Audit

| dataset | seed | probes | max score | new triggers | old triggers | p90 acc drop | max-eta count | P2 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0 | 5 | 0.1287 | 5 | 0 | 0.0000 | 5 | yes |
| Fashion-MNIST | 1 | 5 | 0.1266 | 5 | 0 | 0.0000 | 5 | yes |
| Fashion-MNIST | 2 | 5 | 0.1268 | 5 | 0 | 0.0000 | 5 | yes |
| KMNIST | 0 | 5 | 0.1308 | 5 | 0 | 0.0000 | 5 | yes |
| KMNIST | 1 | 5 | 0.1287 | 5 | 0 | 0.0000 | 5 | yes |
| KMNIST | 2 | 5 | 0.1274 | 5 | 0 | 0.0000 | 5 | yes |
| MNIST | 0 | 5 | 0.1312 | 5 | 0 | 0.0000 | 5 | yes |
| MNIST | 1 | 5 | 0.1307 | 5 | 0 | 0.0000 | 5 | yes |
| MNIST | 2 | 5 | 0.1303 | 4 | 0 | 0.0500 | 4 | no |

P2 verdict: the new controller is considered viable when score/debt probes fire on at least two datasets and probe task cost stays within budget.

## P3 One-Cycle Refresh Policies

| dataset | method | acc | gap | phiR | curvR | recovery | retention | mem | P3 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 0.7812 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
| Fashion-MNIST | oneShot-late-logitOnly-baseOnlyRefresh-r10 | 0.7715 | 0.0098 | 0.0159 | 0.1458 | 1.0000 | 1.2229 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-logitOnly-fullRefresh-r10 | 0.7852 | -0.0039 | 0.0177 | 0.1438 | 1.0000 | 1.3593 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-logitOnly-rbfFrozenRefresh-r10 | 0.7812 | 0.0000 | 0.0122 | 0.1452 | 1.0000 | 0.9373 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-scoreOnly-baseOnlyRefresh-r10 | 0.7715 | 0.0098 | 0.0159 | 0.1458 | 1.0000 | 1.2229 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-scoreOnly-fullRefresh-r10 | 0.7852 | -0.0039 | 0.0177 | 0.1438 | 1.0000 | 1.3593 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-scoreOnly-rbfFrozenRefresh-r10 | 0.7812 | 0.0000 | 0.0122 | 0.1452 | 1.0000 | 0.9373 | 1.0000 | no |
| KMNIST | ABRBF-AdamW | 0.6953 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
| KMNIST | oneShot-late-logitOnly-baseOnlyRefresh-r10 | 0.7031 | -0.0078 | 0.0097 | 0.1472 | 1.0000 | 0.6708 | 1.0000 | no |
| KMNIST | oneShot-late-logitOnly-fullRefresh-r10 | 0.6973 | -0.0020 | 0.0079 | 0.1439 | 1.0000 | 0.5460 | 1.0000 | no |
| KMNIST | oneShot-late-logitOnly-rbfFrozenRefresh-r10 | 0.6914 | 0.0039 | 0.0101 | 0.1461 | 1.0000 | 0.6949 | 1.0000 | no |
| KMNIST | oneShot-late-scoreOnly-baseOnlyRefresh-r10 | 0.7031 | -0.0078 | 0.0097 | 0.1472 | 1.0000 | 0.6708 | 1.0000 | no |
| KMNIST | oneShot-late-scoreOnly-fullRefresh-r10 | 0.6973 | -0.0020 | 0.0079 | 0.1439 | 1.0000 | 0.5460 | 1.0000 | no |
| KMNIST | oneShot-late-scoreOnly-rbfFrozenRefresh-r10 | 0.6914 | 0.0039 | 0.0101 | 0.1461 | 1.0000 | 0.6949 | 1.0000 | no |
| MNIST | ABRBF-AdamW | 0.8535 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
| MNIST | oneShot-late-logitOnly-baseOnlyRefresh-r10 | 0.8711 | -0.0176 | 0.0150 | 0.1459 | 1.0000 | 1.2084 | 1.1439 | no |
| MNIST | oneShot-late-logitOnly-fullRefresh-r10 | 0.8770 | -0.0234 | 0.0129 | 0.1449 | 1.0000 | 1.0434 | 1.1439 | no |
| MNIST | oneShot-late-logitOnly-rbfFrozenRefresh-r10 | 0.8633 | -0.0098 | 0.0153 | 0.1456 | 1.0000 | 1.2351 | 1.1439 | no |
| MNIST | oneShot-late-scoreOnly-baseOnlyRefresh-r10 | 0.8711 | -0.0176 | 0.0150 | 0.1459 | 1.0000 | 1.2084 | 1.1439 | no |
| MNIST | oneShot-late-scoreOnly-fullRefresh-r10 | 0.8770 | -0.0234 | 0.0129 | 0.1449 | 1.0000 | 1.0434 | 1.1439 | no |
| MNIST | oneShot-late-scoreOnly-rbfFrozenRefresh-r10 | 0.8633 | -0.0098 | 0.0153 | 0.1456 | 1.0000 | 1.2351 | 1.1439 | no |

P3 all-dataset survivors:
```text
none
```

## P4 Event-Driven Multi-Cycle V2

| dataset | method | acc | gap | phiR | curvR | events | accepted | mem | time | P4 |
|---|---|---|---|---|---|---|---|---|---|---|

P4 all-dataset survivors:
```text
none
```

## P5-P7 Decision

```text
P5 3-seed selection: not run
Reason: P4 produced no all-dataset survivor.

P6/P7 confirm: not run
Reason: P5 was not reached.
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

## Required Artifacts

Written under `results/v5_3/`:
```text
p0_memory_smoke.csv
p0_core_memory_gate_summary.csv
edge_param_manifest.csv
p1_single_smoothing_event.csv
p1_single_smoothing_gate_summary.csv
p2_event_probe_audit.csv
p2_probe_training_trace.csv
p2_event_probe_gate_summary.csv
p3_one_cycle_controller_variants.csv
p3_refresh_trace.csv
p3_one_cycle_gate_summary.csv
p4_event_driven_multicycle_v2.csv
p4_event_trace.csv
p4_event_v2_gate_summary.csv
p5_candidate_selection3.csv
p6_confirm5.csv
p7_confirm10.csv
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
DG-KAN v5.3 status:
  stop_after_p3_no_one_cycle_survivor

What improved:
  AB-RBF core consistency is now audited in P0.
  P2 directly audits whether the event controller would fire, instead of relying on a plateau proxy.
  The new score/debt trigger fires on most probe points while the old plateau trigger stays at zero.
  P3 separates smoothing damage from refresh recovery and compares base/full/RBF-frozen refresh.

What failed:
  P3 produced no non-baseline one-cycle survivor.
  Refresh recovers task accuracy, but final residual phi reduction is only about 0.8%-1.8%, below the gate.
  Therefore P4/P5/P6/P7 are not justified under the written plan.

Conclusion:
  v5.3 fixes the event-trigger observability problem, but not the geometry-retention problem after refresh.
  LightSmooth remains useful as one-cycle / post-task geometry maintenance, not as a validated train-time multicycle component.
```
