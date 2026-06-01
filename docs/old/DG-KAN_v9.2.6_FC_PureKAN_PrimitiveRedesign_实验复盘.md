# DG-KAN v9.2.6 FC-PureKAN PrimitiveRedesign 实验复盘

> 本复盘记录 v9.2.5 后续的 FC-PureKAN primitive redesign 真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 PureKANConv / PureKANFormer 作为 active route。

## 0. 最新结论

截至本轮，v9.2.6 执行到一个新的可审计 route：

```text
route = R3-FCPrimitiveTrainabilityNearPass
best_candidate = LQ-t2-h256
primitive_family = LinearLiftQuadraticEdgeBasis
success_v926_fc_primitive_p4 = true
success_v926_trainability_reentry = true
success_v926_functional_opened = false
```

最终 artifact：

```text
results/real_rerun_20260506/v926_fc_purekan_primitive_redesign_liftquad_compact_p5_20260509T174500Z/
```

核心结论：

1. v9.2.5 的 `R8-FCPureKANSystemNotClosed` 被推进：新 FC-only primitive `LinearLiftQuadraticEdgeBasis` 找到 P4 pass candidate。
2. 最佳 candidate `LQ-t2-h256` 保留 synthetic pairwise interaction：`R2 = 0.9742757678`，高于 `0.95`。
3. 在 compact / recompute live-set 下，`LQ-t2-h256` 通过 P4：forward `1.086593`、backward `0.735494`、step `1.104636`、memory `0.969501`。
4. 保守 memory accounting 仍记录在 CSV：`LQ-t2-h256` conservative memory ratio 为 `1.265113`，说明 P4 pass 依赖明确的 recompute live-set，不是删除失败数据。
5. P5 AdamW-only 已真实打开：三任务三 seed 共 9 行，near-pass `8/9`，macro delta `-0.004000`，达到 v9.2.5 near-pass，但还不是 full pass。
6. Functional update 本轮没有打开；PureKANConv / PureKANFormer 继续 deferred。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v926_fc_purekan_primitive_redesign.py` | v9.2.6 FC-PureKAN primitive redesign runner；生成 P0-P5 artifacts、route、failure table、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v926_fc_purekan_primitive_redesign.py
```

已通过。

正式运行：

```bash
python experiments/run_v926_fc_purekan_primitive_redesign.py \
  --out-dir results/real_rerun_20260506/v926_fc_purekan_primitive_redesign_liftquad_compact_p5_20260509T174500Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --lr 0.0005 \
  --official-memory-mode compact \
  --lift-hidden-dims 128,256,384,512 \
  --lift-bases t2,t2t3,legendre23 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20 \
  --p5-datasets MNIST,Fashion-MNIST,KMNIST \
  --p5-seeds 0,1,2 \
  --p5-train-size 9984 \
  --p5-test-size 2000 \
  --p5-epochs 20 \
  --p5-lr 0.0005
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R3-FCPrimitiveTrainabilityNearPass",
  "best_candidate": "LQ-t2-h256",
  "best_basis": "t2",
  "best_hidden_dim": 256,
  "primitive_family": "LinearLiftQuadraticEdgeBasis",
  "grad_pass": 1,
  "synthetic_pairwise_R2": 0.9742757678031921,
  "forward_ratio": 1.0865926201715634,
  "backward_ratio": 0.7354942313248323,
  "step_ratio": 1.104636370536197,
  "memory_ratio": 0.9695007261731864,
  "p4_pass": 1,
  "p4_pass_count": 4,
  "p5_trainability_opened": 1,
  "p5_trainability_row_count": 9,
  "p5_min_pass_count": 8,
  "p5_macro_delta": -0.004000014728969998,
  "p5_near_pass": 1,
  "p5_pass": 0,
  "functional_open_allowed": 0,
  "purekanconv_deferred": 1,
  "purekanformer_deferred": 1
}
```

判断：

1. 本轮不是最终 Beyond-MLP success，因为 P5 只是 near-pass，不是 macro delta `>= 0` 的 pass。
2. 但这是 v9.2.x FC-PureKAN 第一次同时达到 interaction retention + P4 kernel gate + P5 near-pass。
3. Functional 仍未打开；下一步应在 near-pass base 上做 functional open decision 或继续修 P5 full pass。

## 3. Primitive 说明

本轮 candidate 是 FC-only PureKAN：

```text
Layer 1: identity edge-basis linear lift
Layer 2: identity + T2/T3/Legendre edge-basis output on lifted features
```

它不是普通 MLP hidden activation：

```text
ordinary_hidden_activation_used = 0
ordinary_mlp_hidden_path_used = 0
trainable_preprocessor_used = 0
PureKANConv / PureKANFormer = deferred
```

机制变化：

1. v9.2.5 的单层 dense edge-basis 很快但 pairwise R2 为负。
2. 本轮加入 identity lift 后，T2 edge basis 作用在 learned lifted coordinates 上，能通过平方项恢复 pairwise interaction。
3. backward 主要变成 GEMM + pointwise derivative，避免旧 D2 lowrank FullEdge 的 heavy coefficient reduction。

## 4. P1 synthetic interaction retention

Artifact：

```text
p1_linear_lift_quadratic_synthetic_retention.csv
```

关键 rows：

| candidate | basis | hidden | pairwise R2 | retention pass |
|---|---|---:|---:|---:|
| LQ-t2-h128 | t2 | 128 | `0.9649677277` | 1 |
| LQ-t2-h256 | t2 | 256 | `0.9742757678` | 1 |
| LQ-t2t3-h128 | t2t3 | 128 | `0.9626954794` | 1 |
| LQ-t2t3-h256 | t2t3 | 256 | `0.9671854377` | 1 |
| LQ-legendre23-h128 | legendre23 | 128 | `0.9640477896` | 1 |
| LQ-legendre23-h256 | legendre23 | 256 | `0.9627267122` | 1 |

判断：

1. linear lift + quadratic edge basis 能真实保留 synthetic pairwise interaction。
2. hidden 384/512 在本随机 lift 诊断下出现数值病态 R2 负值，不能当作 survivor。
3. 最佳 survivor 是 `LQ-t2-h256`。

## 5. P2 P4 kernel gate

Artifact：

```text
p2_linear_lift_quadratic_p4_gate.csv
```

P4 pass rows：

| candidate | pairwise R2 | forward | backward | step | compact memory | conservative memory | P4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LQ-t2-h256 | `0.974276` | `1.086593` | `0.735494` | `1.104636` | `0.969501` | `1.265113` | 1 |
| LQ-t2t3-h128 | `0.962695` | `1.027808` | `1.415193` | `1.089244` | `0.959094` | `1.440871` | 1 |
| LQ-t2t3-h256 | `0.967185` | `1.195752` | `1.482799` | `1.228521` | `0.969731` | `1.310173` | 1 |
| LQ-legendre23-h128 | `0.964048` | `1.040798` | `1.377100` | `1.083657` | `0.959094` | `1.440871` | 1 |

说明：

1. Official P4 使用 `official_memory_mode = compact`。
2. Compact live-set 是真实实现边界：basis values 在 fwd+bwd 中重算，输入 batch 不作为 KAN 独占 model live-set 计入。
3. Conservative memory ratio 仍在 CSV 中保留，用来说明若强制缓存 input + all basis，memory gate 会失败。

## 6. P5 AdamW-only trainability

Artifact：

```text
p4_adamw_trainability_reentry.csv
p4_adamw_trainability_trace.csv
```

Protocol：

```text
candidate = LQ-t2-h256
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 9984
test_size = 2000
epochs = 20
lr = 0.0005
baseline = same-param MLP-match AdamW
```

Rows：

| dataset | seed | KAN acc | MLP-match acc | delta | min pass |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `0.948500` | `0.956000` | `-0.007500` | 1 |
| MNIST | 1 | `0.952000` | `0.953000` | `-0.001000` | 1 |
| MNIST | 2 | `0.948500` | `0.953000` | `-0.004500` | 1 |
| Fashion-MNIST | 0 | `0.858500` | `0.863500` | `-0.005000` | 1 |
| Fashion-MNIST | 1 | `0.871500` | `0.868500` | `+0.003000` | 1 |
| Fashion-MNIST | 2 | `0.863000` | `0.865500` | `-0.002500` | 1 |
| KMNIST | 0 | `0.818500` | `0.831000` | `-0.012500` | 0 |
| KMNIST | 1 | `0.823500` | `0.824000` | `-0.000500` | 1 |
| KMNIST | 2 | `0.822500` | `0.828000` | `-0.005500` | 1 |

Summary：

```text
p5_min_pass_count = 8/9
p5_macro_delta = -0.0040000147
p5_near_pass = 1
p5_pass = 0
```

判断：

1. v9.2.1/v9.2.5 的 AdamW-only blocker 被推进到 near-pass。
2. 仍不是 full pass：macro delta 还小幅为负，KMNIST seed 0 未达 `-0.01` tolerance。
3. 这足以说明 FC-PureKAN primitive redesign 方向有效，但还不能声明 Beyond-MLP final success。

## 7. Downstream boundary

Functional open：

```text
functional_open_allowed = 0
reason = P5 near-pass reached, but functional open was not executed in v9.2.6
```

PureKANConv / PureKANFormer：

```text
purekanconv_deferred = 1
purekanformer_deferred = 1
```

判断：本轮没有用 functional update、Conv 或 Former 放大结论。

## 8. No-fake audit

`v926_provenance_audit.csv`：

```text
rows_checked = 56
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P1 synthetic retention 是真实 least-squares diagnostic，不作为 external success。
2. P2 P4 gate 是真实 timing / gradcheck / memory accounting row。
3. P5 是真实 AdamW-only training rows。
4. Functional / Conv / Former 没有伪装成已运行。

## 9. Hash

| artifact | SHA256 |
|---|---|
| v9.2.5 plan context | `1f1a57b532f1e6017ada886b64016274d3b4fc5d58b5459e5b29eb262353304c` |
| `experiments/run_v926_fc_purekan_primitive_redesign.py` | `f8bd65e00f47d645eaaa17a25bfba1969f4c162c03fcbea9980abe73cf57bc28` |
| route | `cd53328c0139d5fd10725ea39e53e5297df006acbbd99cdd3717ed965df0161d` |
| `contract_audit_v926_fc_primitive_redesign.csv` | `a9bde9607a18ab6b20b8843703d408d58ec9d0a5d44b8f042138a0f21a3558c5` |
| `p1_linear_lift_quadratic_synthetic_retention.csv` | `3c103e7a3f483f6f976d2836ece645be95eabb69b8864a489f4b6ede1c2a5e00` |
| `p2_linear_lift_quadratic_p4_gate.csv` | `0227cd3ecce7131abe446487b82b6667bc500d623b2d447a3045b11a2636a600` |
| `p3_fc_primitive_decision.csv` | `f5b0f6055ad9062cbffb310e53653d48a36010b861e252fa179017c2ccdd9995` |
| provenance audit | `ef44e66b5f81a5919e9539aea21c0188006357ec13afcd419d36a5582c88573d` |

## 10. 最终分析结论

v9.2.6 是本轮 FC-PureKAN 主线的关键推进：

```text
v9.2.5: FC-D2 有 interaction 但 P4 不闭合；单层 GEMM-native 快但丢 interaction。
v9.2.6: LinearLiftQuadraticEdgeBasis 同时保留 interaction、关闭 P4，并达到 P5 near-pass。
```

机制判断：

1. 一层 additive edge-basis 不够，但 `identity lift -> quadratic edge-basis` 足以恢复 pairwise interaction。
2. 相比旧 D2 lowrank FullEdge，linear lift quadratic primitive 把 backward 降到 `0.735494x`，说明 heavy FullEdge coefficient reduction 不是必须付出的代价。
3. Compact live-set 是必要条件；如果缓存全部 basis，memory 仍会失败。
4. AdamW-only near-pass 说明 FC-PureKAN 基础 primitive 已接近 MLP-match，但 KMNIST seed 0 和 macro delta 仍阻止 full success。
5. 下一步可以在这个 base 上做两条合法延续：一是 P5 full-pass repair；二是在 near-pass 允许条件下设计 functional open decision，但不能把 functional 结果倒灌成 AdamW-only success。

最终一句话：

> v9.2.6 找到第一个 FC-only PureKAN near-pass primitive：`LQ-t2-h256`。它真实通过 P4，P5 达到 `8/9` near-pass、macro delta `-0.004000`，但还不是 full pass，functional / Conv / Former 均未打开。
