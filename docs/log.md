# DG-LCA 实验日志

## 2026-04-28 16:53 +08

### 项目理解

阅读 `docs/` 后，当前项目主线应按 DG-LCA v0.3 的保守实验版推进：

1. Gate 0：先诊断真实 BP credit 是否低维、稳定、可压缩。
2. Gate 1：验证 single-block local VJP 能否被条件线性 router 蒸馏。
3. Gate 2：验证 KAN-like 模块中函数空间 metric preconditioning 是否比普通 coefficient update 更稳定。
4. Gate 3/4 暂不作为第一阶段主线；只有 Gate 0/1/2 过关后再做 sequential routing 和 adapter-only memory claim。

核心定位不是“替代 BP”，而是验证 `local VJP distillation + functional-space preconditioning + strict audits` 在结构化小模块中是否成立。

### KAN 实现选择

`third_party/awesome-kan` 当前只有索引 README，没有实际 KAN 代码。适合本项目的候选：

- `efficient-kan`：纯 PyTorch，和本项目 toy/local VJP 实验最贴近，后续若接第三方实现优先考虑。
- `FastKAN` / `FasterKAN`：更偏速度和工程实现，可作为后续性能 baseline。
- `pykan`：官方实现，解释性强，但对本项目第一阶段的可控 metric / coefficient update 不如最小实现直接。

本轮实验先在仓库内实现了一个最小 RBF/KAN-like edge layer，而不是引入黑盒第三方包。原因是 Gate 2 需要直接控制 coefficient、Sobolev Gram matrix、damping 和 predicted/actual descent 审计。

### 环境

创建环境：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda create -y -n lca python=3.10
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python -m pip install -r requirement.txt
```

## 2026-05-02 DG-KAN v3.6：Rational tangent U-FULL 与 PureKAN 实验

依据文档：

```text
docs/DG-KAN_v3.6_RationalUFull_PureKAN_具体实验计划.md
```

本轮实现：

```text
experiments/dgkan_core.py
  Rational tangent Sobolev metric
  Rational actual denominator audit
  Rational branch-linear functional update option
  PureKANClassifier / PureResidualKANBlock / FixedNorm
  module/duck-typing coefficient_named_params
  basis occupancy / ConvStem activation audit

experiments/run_gafu_v36.py
  V3_6_P0_IMPL_SMOKE
  V3_6_P1_RATIONAL_METRIC_AUDIT
  V3_6_P2_RATIONAL_TANGENT_TOY
  V3_6_P3_RATIONAL_DGKAN_TANGENT3
  V3_6_P5_UFULL_F100_FAILURE_AUDIT
  V3_6_P6_CIFAR_BRANCH_AUDIT
  V3_6_P7_PUREKAN_SMOKE
  V3_6_P8_PUREKAN_3SEED

experiments/analyze_gafu_v36.py
  aggregated 266 rows into docs/DG-KAN_v3.6_RationalUFull_PureKAN_结果复盘.md
```

执行说明：

```text
P4 not run: P3 Rational tangent did not pass confirm trigger.
P9 not run: P8 PureKAN-UFULL did not pass confirm trigger.
```

关键结果：

| package | conclusion |
|---|---|
| P0 | pass；Rational tangent cond 从旧 poly 的 `~2.5e5` 降到 `~8.5e3`，PureKAN learnable non-KAN params = 0 |
| P1 | 旧 poly metric condition 高且 clip 高；`rho=1e-1` 降 cond/clip，但不形成稳定优势 |
| P2 | toy 上 tangent-L2 明显优于旧 poly，bad-step 为 0；证明 tangent metric 方向更合理 |
| P3 | tangent metric 清掉 clip，但真实 DGKAN 上 cond 仍 `~1.5e4-1.8e4`，且相对 Rational-AdamW 没有稳定 accuracy/AUC 优势 |
| P5 | f100 在 Fashion accuracy 更强，但 KMNIST 仍不稳；branch 更大不等于 task alignment 更好 |
| P6 | CIFAR-small 上 ConvStem U-FULL 仍显著低于 AdamW；diag-warmup probe 局部改善但不能恢复 |
| P8 | PureKAN 结构成功且 AdamW 可训练；PureKAN-UFULL 在 MNIST/Fashion/KMNIST 都明显慢或掉点 |

当前判断：

```text
Rational/KAT:
  tangent functional metric 是正确方向，解决了旧 poly metric 的 clip 问题；
  但尚未解决 condition 与任务收益问题，不能进入 P4 5-seed confirm。

PureKAN:
  架构实现有效，且参数审计确认 learnable non-KAN = 0；
  但当前 U-FULL 不能独立支撑 PureKAN 训练，仍需要新的 whole-network functional optimizer 或更适合 PureKAN 的 metric/schedule。

Mainline:
  Hybrid U-FULL / RBF-DGKAN 仍是主线；
  Rational tangent 与 PureKAN 都是机制验证方向，不替代当前 clean default。
```

实际导入版本：

```text
torch 2.11.0+cu130
numpy 2.2.6
sklearn 1.7.2
tqdm 4.67.1
```

注意：当前机器 NVIDIA driver 显示为 `12040`，而 pip 安装的 torch wheel 带 CUDA 13，因此运行 autograd 时会出现 CUDA driver warning。本轮实验全部使用 `--device cpu`，结果不依赖 GPU。后续若要跑 GPU，建议改装与 driver 匹配的 CUDA 12.4 wheel 或 CPU-only wheel。

### 新增实验脚本

新增：

- `requirement.txt`
- `experiments/dg_lca_toy.py`

脚本包含：

- Gate 0：Two Moons 上 4-layer MLP 的 hidden credit SVD 诊断。
- Gate 1：固定 residual block 的 exact VJP teacher，以及 diagonal + low-rank conditional linear router 蒸馏。
- Gate 2：1D `sin(x) + 0.3 sin(5x)` RBF/KAN-like edge regression，对比 coefficient SGD、Adam、Sobolev preconditioned update。

### 网络设计

本轮网络设计原则是“小、可审计、能直接对应 Gate 指标”。所以没有先引入完整第三方 KAN，而是把每个 Gate 拆成一个最小可验证模块。

#### Gate 0：credit rank 诊断 MLP

用途：收集每层 hidden activation 的 BP credit，并做 SVD / participation rank / subspace overlap。

结构：

```text
input dim = 2
hidden dim = 128
depth = 4

block_i:
  Linear(d_in, 128)
  LayerNorm(128)
  SiLU()

head:
  Linear(128, 2)
```

前向时对每个 block 输出 `h_k` 调用 `retain_grad()`，反向后把 `h_k.grad` 当作该层 credit 样本。选择 `LayerNorm + SiLU` 是为了避免 toy MLP 训练不稳，同时保持 block 结构足够普通，方便后续替换成 residual / KAN-like / adapter block。

默认训练配置：

```text
dataset = Two Moons
samples = 2048
noise = 0.12
epochs = 24
batch size = 256
optimizer = AdamW(lr=2e-3, weight_decay=1e-4)
audit batches = 4
```

#### Gate 1：single-block VJP teacher block

用途：构造一个固定 block，使用 exact local VJP 生成 teacher credit：

```text
g_teacher = J_block(h)^T g_next
```

block 结构：

```text
dim = 64
scale = 0.7

F(h) = h + scale * Linear2(tanh(Linear1(h)))
```

也就是一个小 residual MLP block。它故意保留 identity path，因为 DG-LCA 文档中强调结构化模块、adapter、residual block 是更合理的早期对象；同时 residual block 的 local VJP 不会过度病态，适合先检验 router 表达力。

#### Gate 1：conditional linear router

用途：拟合 `g_next -> g_k` 的局部 VJP，但必须保持对 `g_next` 线性。

router 形式：

```text
C(h, h_next)[g] = D(h, h_next) * g + U(h, h_next) @ (V(h, h_next)^T @ g)
```

默认超参：

```text
dim = 64
rank = 16
condition input = concat(h, h_next), dim=128
router hidden = 128
router steps = 800
optimizer = AdamW(lr=2e-3, weight_decay=1e-5)
```

router 参数网络：

```text
Linear(2 * dim, 128)
SiLU()
Linear(128, 128)
SiLU()
Linear(128, dim + 2 * dim * rank)
```

最后一层输出被拆成：

- `D`: diagonal gate，shape `[dim]`
- `U`: low-rank left factor，shape `[dim, rank]`
- `V`: low-rank right factor，shape `[dim, rank]`

关键约束：`D/U/V` 可以依赖 `h, h_next`，但应用到 `g` 时只做 diagonal scaling 和 low-rank matrix-vector product，因此结构上对 `g` 保持线性。实验还额外做 scale/superposition linearity residual 审计。

初始化：

- `D` bias 初始化为 `1.0`，让 router 起点接近 identity VJP。
- `U/V` 小随机初始化，避免一开始 low-rank 项扰动过大。

#### Gate 2：KAN-like RBF edge layer

用途：验证 coefficient-space gradient 经过函数空间 metric preconditioning 后，是否比普通 Adam 更稳。

本轮使用 1D 单边函数，不是完整多输入多输出 KAN。这样可以直接审计 basis coefficient、函数斜率和曲率。

结构：

```text
target: y = sin(x) + 0.3 sin(5x)
x in [-pi, pi]
n_basis = 32

B_j(x) = exp(-0.5 * ((x - c_j) / width)^2)
normalized_B_j(x) = B_j(x) / sum_k B_k(x)

f(x) = sum_j a_j * normalized_B_j(x) + bias
```

中心点 `c_j` 均匀分布在 `[-pi, pi]`，`width = spacing * 1.2`。归一化 RBF basis 让函数值尺度更稳定，也让 coefficient update 更容易比较。

对照方法：

```text
coefficient SGD
coefficient Adam
Sobolev preconditioned update
```

Sobolev preconditioner：

```text
M = int B^T B dx
  + alpha * int B'^T B' dx
  + beta  * int B''^T B'' dx
  + rho * I

a <- a - lr * solve(M, grad_a)
```

默认超参：

```text
steps = 500
train samples = 512
validation grid = 512
Adam lr = 0.03
SGD lr = 0.08
precond lr = 0.25
alpha = 1e-2
beta = 5e-4
rho = 1e-2
```

审计指标：

- train / validation MSE
- max absolute slope
- curvature energy
- predicted descent 与 actual descent 是否同号
- `cond(M + rho I)`

这个 Gate 2 模型有意保持为“单条 KAN edge”。它不能代表完整 KAN 网络效果，但能干净验证文档中的函数空间预条件更新公式。

Smoke test：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate all --device cpu --epochs 2 --router-steps 2 --kan-steps 2 \
  --samples 512 --batch-size 128 --audit-batches 2 \
  --out-dir results/dg_lca_smoke
```

正式 toy run：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate all --device cpu --out-dir results/dg_lca_toy
```

结果目录：

```text
results/dg_lca_toy/20260428-165208
```

### 结果摘要

Gate 0：Two Moons / MLP credit rank

- final epoch：24
- train acc：0.9982
- test acc：0.9951
- mean `E64` variance energy：1.0000
- mean participation rank：1.15
- mean top-16 batch subspace overlap：0.7494
- status：medium

分层现象：

- layer 0：PR 1.425，E1 0.8235，E8 1.0000，top-16 overlap 0.8931
- layer 1：PR 1.137，E1 0.9367，E8 1.0000，top-16 overlap 0.8843
- layer 2：PR 1.041，E1 0.9801，E8 1.0000，top-16 overlap 0.8940
- layer 3：PR 1.000，E1 1.0000，E8 1.0000，top-16 overlap 0.3264

解释：toy binary classification 的 credit 极低维，支持继续做 Gate 1；但这很可能受到二分类交叉熵和简单任务结构影响，不能直接外推到 MNIST/CIFAR 或高维 hidden space。

Gate 1：single-block local VJP router

- router cosine：0.9954
- router relerr：0.0962
- router norm ratio：0.9950
- linearity residual：`6.05e-08`
- identity baseline cosine：0.9818
- identity baseline relerr：0.1896
- status：medium

解释：conditional linear diagonal+low-rank router 能显著改善 exact local VJP 拟合误差，并严格保持线性性。由于 block 是 residual，identity baseline 已经很强；后续应加入 non-residual/KAN-like/adapter block 测试。

Gate 2：KAN-like functional-space update

目标：`sin(x) + 0.3 sin(5x)`，basis=32，steps=500。

| method | val loss | max slope | curvature energy | descent agreement |
|---|---:|---:|---:|---:|
| SGD | 0.018763 | 1.558 | 29.454 | n/a |
| Adam | 0.0000687 | 2.527 | 176.337 | n/a |
| Sobolev precond | 0.0000404 | 2.501 | 176.273 | 1.0 |

Sobolev metric condition number：21.07。

解释：Sobolev preconditioner 在该 toy target 上比 Adam 有更低 validation loss，predicted descent 与 actual descent 100% 一致，slope/curvature 也略低。但 curvature 改善幅度很小，不能夸大为强结论；更像是 Gate 2 的初步可行性信号。

### 结论与下一步

当前 toy 实验支持继续推进：

1. Gate 0 在简单二分类上观察到强低秩 credit，但需要更难数据集和多 batch/epoch 稳定性验证。
2. Gate 1 的条件线性 router 可学，下一步应换成 KAN-like block / adapter block，并加入 stale audit。
3. Gate 2 的 Sobolev preconditioner 初步有效，但需要在 `sin`、`smooth_step`、2D regression 和 Two Moons classifier 上重复。
4. 暂不应声明低显存优势；Gate 4 需要和 checkpointing、adapter-only BP、local exact VJP 等强 baseline 做完整 accounting。

## 2026-04-28 17:10 +08：网络设计复盘与第二轮实验

### 复盘问题

根据第一轮结果，原始网络设计适合作为 smoke test，但不适合作为强证据：

1. Gate 0 使用 Two Moons 二分类，最终 loss credit 本身受 2-class 输出约束，极容易出现低秩现象。
2. Gate 1 使用 full residual block，identity baseline 已经很强，router 的高分可能主要来自“接近恒等映射”，而不一定说明它学会了困难 VJP。
3. Gate 2 使用单条 RBF/KAN-like edge，能审计 functional metric，但还不是完整 KAN layer，更不能代表 KAN classifier 或 adapter 场景。

因此第二轮继续实验的目标是把问题做得更有区分度：

- Gate 0 增加 `digits` 十分类小数据集。
- Gate 1 增加 non-residual full-rank MLP block，测试 router 在不结构化 block 上是否失败。
- Gate 1 增加 bottleneck residual block，测试 router 在更符合 DG-LCA 假设的低秩结构化模块上是否恢复。
- Gate 2 增加 `sin` 与 `smooth_step` target，观察 functional preconditioner 是否稳定泛化。

### 脚本调整

`experiments/dg_lca_toy.py` 新增：

```text
--dataset digits
--gate1-block mlp
--gate1-block bottleneck_residual
--gate1-bottleneck 16
```

新增 Gate 1 block：

```text
PlainMLPBlock:
  F(h) = scale * W2 tanh(W1 h)

BottleneckResidualBlock:
  F(h) = h + scale * W_up tanh(W_down h)
  W_down: dim -> bottleneck
  W_up: bottleneck -> dim
```

其中 `PlainMLPBlock` 是故意构造的压力测试：它的 local Jacobian 通常是 full-rank 且随 `h` 变化。`BottleneckResidualBlock` 更接近 adapter / LoRA / structured module：local VJP 是 identity + low-rank nonlinear correction。

### 第二轮命令

Gate 0：digits 十分类 credit rank

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate0 --dataset digits --samples 1797 --epochs 24 \
  --batch-size 256 --audit-batches 4 --device cpu \
  --out-dir results/dg_lca_gate0_digits
```

Gate 1：full-rank non-residual MLP block

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate1 --gate1-block mlp --router-rank 16 --router-steps 800 \
  --router-eval-batches 8 --device cpu \
  --out-dir results/dg_lca_gate1_mlp_rank16

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate1 --gate1-block mlp --router-rank 64 --router-steps 1200 \
  --router-hidden 256 --router-eval-batches 8 --device cpu \
  --out-dir results/dg_lca_gate1_mlp_rank64
```

Gate 1：bottleneck residual block

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate1 --gate1-block bottleneck_residual --gate1-bottleneck 16 \
  --block-scale 1.0 --router-rank 4 --router-steps 800 \
  --router-eval-batches 8 --device cpu \
  --out-dir results/dg_lca_gate1_bottleneck_rank4

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate1 --gate1-block bottleneck_residual --gate1-bottleneck 16 \
  --block-scale 1.0 --router-rank 16 --router-steps 800 \
  --router-eval-batches 8 --device cpu \
  --out-dir results/dg_lca_gate1_bottleneck_rank16
```

Gate 2：target sweep

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate2 --kan-target sin --device cpu \
  --out-dir results/dg_lca_gate2_sin

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_toy.py \
  --gate gate2 --kan-target smooth_step --device cpu \
  --out-dir results/dg_lca_gate2_smooth_step
```

### 第二轮结果

Gate 0：digits 十分类

结果目录：

```text
results/dg_lca_gate0_digits/20260428-165758
```

摘要：

- train acc：1.0000
- test acc：0.9750
- mean `E64`：0.9990
- mean participation rank：7.69
- mean top-16 batch subspace overlap：0.6722
- status：medium

分层：

| layer | PR | E8 | E16 | E64 | top-16 overlap |
|---|---:|---:|---:|---:|---:|
| 0 | 7.454 | 0.9193 | 0.9679 | 0.9984 | 0.6680 |
| 1 | 7.092 | 0.9315 | 0.9746 | 0.9987 | 0.6689 |
| 2 | 7.698 | 0.9275 | 0.9756 | 0.9990 | 0.7078 |
| 3 | 8.525 | 0.9280 | 1.0000 | 1.0000 | 0.6442 |

解释：从 Two Moons 的 PR 约 1.15 到 digits 的 PR 约 7.69，说明第一轮低秩不是充分证据；但 digits 上仍观察到明显压缩性，`E16` 接近或超过 0.968，值得继续。

Gate 1：full-rank non-residual MLP block

| block | router rank | cosine | relerr | norm ratio | identity cosine | status |
|---|---:|---:|---:|---:|---:|---|
| MLP | 16 | 0.5788 | 0.8156 | 0.6160 | -0.0298 | not_passed |
| MLP | 64 | 0.9256 | 0.3820 | 0.9400 | -0.0265 | not_passed |

解释：full-rank MLP block 对 diagonal + low-rank router 明显更难。即使用 rank 64，cosine 已上来，但 relerr 仍高于弱成功标准。这说明不能把 learned VJP router 作为任意 block 的默认假设，必须坚持文档里的结构化模块路线。

Gate 1：bottleneck residual block

| block | bottleneck | router rank | cosine | relerr | norm ratio | identity relerr | status |
|---|---:|---:|---:|---:|---:|---:|---|
| bottleneck residual | 16 | 4 | 0.9661 | 0.2518 | 0.9707 | 0.3151 | weak |
| bottleneck residual | 16 | 16 | 0.9997 | 0.0228 | 0.9993 | 0.3159 | medium |

解释：当 block 的局部 VJP 真的是 identity + low-rank correction 时，rank 对齐后 router 表现非常好。这比第一轮 full residual 更有说服力，因为 rank4 到 rank16 的性能差异清楚对应了结构秩。

Gate 2：target sweep

| target | method | val loss | max slope | curvature energy | descent agreement |
|---|---|---:|---:|---:|---:|
| sin | Adam | 0.00000389 | 1.0078 | 3.6067 | n/a |
| sin | Sobolev precond | 0.00000538 | 1.0051 | 3.6440 | 1.0 |
| smooth_step | Adam | 0.0000696 | 3.1010 | 28.5222 | n/a |
| smooth_step | Sobolev precond | 0.0000309 | 3.2001 | 31.6925 | 1.0 |

解释：

- `sin` 上 Adam 已经足够强，Sobolev preconditioner 没有优势。
- `smooth_step` 上 Sobolev preconditioner 降低了 validation loss，但斜率和曲率更高，说明它可能更积极地拟合局部变化，而不是自动变平滑。
- predicted / actual descent agreement 仍为 1.0，这是 Gate 2 最稳定的正信号。

### 对三个问题的回答

#### 1. 网络设计是否合理？

第一轮设计合理作为 smoke test，但过于友好；第二轮后更合理：

- Gate 0 应继续使用 `digits` / MNIST / Fashion-MNIST 这类多类任务，而不是只看 Two Moons。
- Gate 1 的主实验不应使用普通 full residual block 作为唯一证据；更推荐用 `bottleneck_residual` 或 adapter/LoRA-like block。
- full-rank `mlp` block 应保留为负对照，帮助证明 DG-LCA 的适用边界。
- Gate 2 的单边 RBF 模型适合公式审计，但不适合作为完整 KAN 结论。

#### 2. KAN 实现是否需要更换？为什么？

短期不需要立刻更换：当前 RBF/KAN-like edge 对 Gate 2 的 functional metric 审计很方便，因为可以直接控制 basis、coefficient、Sobolev Gram matrix 和 descent audit。

但如果要进入“真正 KAN-like block / KAN classifier”阶段，需要更换或至少补充：

- `efficient-kan`：适合作为 PyTorch KAN baseline，来自 `awesome-kan` 的实现候选中最贴近本项目。
- 或者本仓库实现最小 B-spline KAN layer：好处是能继续控制 metric/preconditioner；坏处是工程量更大。

原因：当前 RBF edge 是 KAN-like，不是 canonical B-spline KAN；它可以验证函数空间预条件思想，但不能支撑“KAN 上成立”的完整 claim。

#### 3. 是否需要继续调整？

需要，调整方向如下：

1. Gate 0：把主诊断从 Two Moons 切到 digits/MNIST/Fashion-MNIST，并报告 `E8/E16/E64`，不要只看 `E64`。
2. Gate 1：以 `bottleneck_residual` / adapter / LoRA-like block 为主，保留 full-rank MLP 作为失败边界。
3. Gate 1：继续做 stale audit，即 block 参数变化后 router fidelity 下降多少。
4. Gate 2：对 Sobolev `alpha/beta/rho/lr` 做 target-specific sweep；当前它不是单向优于 Adam，而是 loss/smoothness trade-off。
5. KAN：下一阶段接入 `efficient-kan` 或实现 B-spline KAN layer，做完整 KAN-like block 的 Gate 1 + Gate 2 联合测试。

## 2026-04-28：第三轮优化实验

### 代码调整

新增 `experiments/dg_lca_sweep.py`，用于批量跑实验，而不是只跑单个 smoke test：

- Gate 1 rank sweep：固定 `bottleneck_residual`，扫描 router rank，并同时做 stale audit。
- Gate 2 grid sweep：扫描 Sobolev preconditioner 的 `alpha/beta/lr`，同时保留 Adam baseline。
- 输出 `summary.json`、`*.json` 和 `*.csv`，方便之后复现实验和画图。

同时在 `experiments/dg_lca_toy.py` 中加入 stale audit：

- `--stale-noise-levels`：对 block 参数做相对噪声扰动，例如 `0,0.01,0.03,0.1`。
- 评估同一个 learned router 在 block 参数变化后的 VJP fidelity。
- 这对应真实训练中 router 可能滞后于 forward block 的情况。

### 执行命令

Gate 1：bottleneck residual rank + stale sweep

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_sweep.py \
  --mode gate1-rank --ranks 1,2,4,8,16,32 \
  --router-steps 800 --router-eval-batches 8 \
  --stale-noise-levels 0,0.01,0.03,0.1 \
  --out-dir results/dg_lca_sweep_gate1_rank_stale
```

结果目录：

```text
results/dg_lca_sweep_gate1_rank_stale/20260428-183428
```

Gate 2：Sobolev preconditioner grid sweep

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/dg_lca_sweep.py \
  --mode gate2-grid --targets sin5,smooth_step --kan-steps 350 \
  --precond-lrs 0.15,0.25 --alphas 0.01,0.05 \
  --betas 0,0.0005,0.005 \
  --out-dir results/dg_lca_sweep_gate2_grid
```

结果目录：

```text
results/dg_lca_sweep_gate2_grid/20260428-183526
```

### Gate 1 rank + stale 结果

| rank | cosine | relerr | norm ratio | stale relerr @ 0.01 | stale relerr @ 0.03 | stale relerr @ 0.1 | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.9530 | 0.2979 | 0.9570 | 0.2978 | 0.2974 | 0.3000 | weak |
| 2 | 0.9579 | 0.2816 | 0.9600 | 0.2803 | 0.2792 | 0.2847 | weak |
| 4 | 0.9661 | 0.2518 | 0.9707 | 0.2533 | 0.2525 | 0.2543 | weak |
| 8 | 0.9812 | 0.1866 | 0.9827 | 0.1852 | 0.1860 | 0.1942 | medium |
| 16 | 0.9997 | 0.0228 | 0.9993 | 0.0239 | 0.0276 | 0.0559 | medium |
| 32 | 0.9997 | 0.0237 | 1.0002 | 0.0244 | 0.0282 | 0.0559 | medium |

结论：

- `rank=16` 是当前 bottleneck residual block 的明显 elbow；它和 bottleneck 维度对齐后，relerr 从 rank 8 的 0.1866 降到 0.0228。
- `rank=32` 没有继续改善，说明增加 rank 已经不是主要瓶颈。
- stale audit 表明，`rank=16` 在 0.01 和 0.03 参数扰动下仍稳定；到 0.1 时 relerr 上升到约 0.056，但仍显著好于低 rank。
- 这支持“DG-LCA router 适合结构化低秩/adapter/LoRA-like block”，但不支持“任意 full-rank MLP block 都适合”。

### Gate 2 Sobolev grid 结果

Adam baseline：

| target | Adam val loss | Adam max slope | Adam curvature |
|---|---:|---:|---:|
| sin5 | 0.0001016 | 2.5297 | 176.3937 |
| smooth_step | 0.0000930 | 3.0558 | 27.1757 |

Validation loss 最优的 Sobolev preconditioner：

| target | alpha | beta | lr | val loss | val / Adam | slope / Adam | curvature / Adam | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sin5 | 0.01 | 0.0 | 0.25 | 0.0000573 | 0.5643 | 0.9896 | 0.9976 | 1.0 |
| smooth_step | 0.01 | 0.0 | 0.25 | 0.0000358 | 0.3851 | 1.0439 | 1.1618 | 1.0 |

曲率最小的 Sobolev preconditioner：

| target | alpha | beta | lr | val loss | val / Adam | slope / Adam | curvature / Adam | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sin5 | 0.05 | 0.005 | 0.15 | 0.0007955 | 7.8299 | 0.9473 | 0.7998 | 1.0 |
| smooth_step | 0.05 | 0.005 | 0.15 | 0.0002298 | 2.4699 | 0.9310 | 0.7413 | 1.0 |

结论：

- Sobolev preconditioner 确实能改善 optimization：在 `alpha=0.01, beta=0, lr=0.25` 时，两个 target 的 validation loss 都明显低于 Adam。
- 但 Sobolev metric 不是自动平滑器。加入较强 `beta` 后曲率下降，但 validation loss 明显变差。
- 因此 Gate 2 需要拆成两个 claim：
  - optimization preconditioning：当前证据较强，descent agreement 全部为 1.0。
  - smoothness regularization：存在 loss/smoothness trade-off，不能直接宣称免费提升。

### 第三轮后的设计判断

1. 网络设计：主线应使用 `bottleneck_residual` / adapter / LoRA-like block。当前 `DiagLowRankRouter` 对这种结构是合理的；full-rank MLP 只作为负对照。
2. KAN 实现：短期仍可保留当前 RBF/KAN-like edge 做 functional metric 审计；进入真正 KAN 实验时应补充 `efficient-kan` 或自实现 B-spline KAN layer。
3. 继续调整：下一步应把 Gate 1 和 Gate 2 合起来，在 KAN-like block 或 B-spline KAN block 上验证“结构化 block + functional-space preconditioning”的组合收益。

### 校验命令

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python -m py_compile \
  experiments/dg_lca_toy.py experiments/dg_lca_sweep.py
```

## 2026-04-28：按详细 WandB 版重做 Gate 0

用户要求依据 `docs/DG-LCA_v0.3_实验验证方案_详细WandB版.md` 重新做实验，并先给出 Gate 0。

### 代码与环境调整

新增 `experiments/gate0_credit_wandb.py`，专门执行 Gate 0 credit rank diagnostic，并上传 W&B：

- 数据：`mnist` / `fashion_mnist` 通过 `sklearn.fetch_openml`，`digits` 用 sklearn 内置数据。
- 模型：`mlp` 与 `kan_like_mlp`，本轮先跑 `mlp`。
- credit sources：
  - `full_bp`
  - `adapter_projected`
  - `random_feedback`
- W&B metrics：
  - `credit/layer_{k}/E16_var`
  - `credit/layer_{k}/E32_var`
  - `credit/layer_{k}/E64_var`
  - `credit/layer_{k}/E128_var`
  - `credit/layer_{k}/PR`
  - `credit/layer_{k}/entropy_rank`
  - `credit/layer_{k}/spectral_decay_slope`
  - `credit/layer_{k}/rank_90`
  - `credit/layer_{k}/rank_95`
  - `credit/layer_{k}/batch_overlap_top64`
  - `credit/layer_{k}/epoch_overlap_top64`
  - `credit/global/*`
  - `credit/batch_sensitivity/*`

`requirement.txt` 增加：

```text
pandas>=2.0
wandb>=0.17
```

`.gitignore` 增加：

```text
.cache/
wandb/
```

### 执行命令

正式 5-seed MNIST Gate 0：

```bash
for seed in 0 1 2 3 4; do
  WANDB_API_KEY=<redacted> WANDB_SILENT=true PYTHONWARNINGS=ignore \
  /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/gate0_credit_wandb.py \
    --dataset mnist --train-size 12000 --val-size 2000 --test-size 2000 \
    --epochs 5 --max-steps 100000 --depth 4 --hidden-dim 256 \
    --batch-size 256 --eval-batch-size 512 --audit-batch-size 512 --audit-batches 2 \
    --audit-batch-sizes 128,256,512,1024 --adapter-dim 32 \
    --credit-sources full_bp,adapter_projected,random_feedback \
    --optimizer adamw --lr 0.0003 --weight-decay 0.0001 --device cpu \
    --wandb-mode online --wandb-project dg-lca --wandb-group gate0_mnist_mlp4_h256_5seed \
    --run-name g0_mnist_mlp4_h256_seed${seed} --seed ${seed} \
    --out-dir results/gate0_wandb_mnist_mlp4_h256_5seed --print-level brief
done
```

W&B：

```text
project: dg-lca
group: gate0_mnist_mlp4_h256_5seed
run ids: km89wan9, tmze38so, dmo8r5zd, bxvpyvdn, v3ikitke
```

本地结果：

```text
results/gate0_wandb_mnist_mlp4_h256_5seed/aggregate_summary.json
results/gate0_wandb_mnist_mlp4_h256_5seed/aggregate_runs.csv
```

### 5-seed 汇总

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| val acc | 0.9496 | 0.0066 | 0.9375 | 0.9575 |
| test acc | 0.9468 | 0.0021 | 0.9445 | 0.9500 |
| mean E64 | 0.99917 | 0.00036 | 0.99849 | 0.99948 |
| worst E64 | 0.99846 | 0.00073 | 0.99704 | 0.99906 |
| mean E128 | 0.99994 | 0.00003 | 0.99988 | 0.99996 |
| worst E128 | 0.99988 | 0.00006 | 0.99978 | 0.99993 |
| mean PR | 5.0268 | 0.9173 | 3.4365 | 5.8236 |
| mean batch overlap top64 | 0.5230 | 0.0044 | 0.5154 | 0.5288 |
| worst batch overlap top64 | 0.3574 | 0.0036 | 0.3540 | 0.3644 |
| layer3 batch overlap top64 | 0.3574 | 0.0036 | 0.3540 | 0.3644 |
| layer3 epoch overlap top64 | 0.3551 | 0.0025 | 0.3525 | 0.3597 |

每个 seed 的 Gate 0 判定均为 `weak`：

```text
weak: 5 / 5
medium: 0 / 5
strong: 0 / 5
```

### 分层观察

seed0 的 final full-BP credit 分层结果：

| layer | E16 | E32 | E64 | E128 | PR | rank90 | rank95 | batch overlap64 | epoch overlap64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9762 | 0.9938 | 0.9987 | 0.9999 | 4.5746 | 7 | 11 | 0.5632 | 0.5803 |
| 1 | 0.9866 | 0.9961 | 0.9991 | 0.9999 | 4.5042 | 6 | 9 | 0.5641 | 0.5919 |
| 2 | 0.9898 | 0.9967 | 0.9991 | 0.9999 | 4.7490 | 6 | 8 | 0.5779 | 0.5908 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.4575 | 6 | 7 | 0.3566 | 0.3542 |

解释：credit energy 非常可压缩，甚至 `E16` 已经很高；但最后一层 top-64 子空间在 batch/epoch 间不稳定，是 Gate 0 只能 weak pass 的主要原因。

### 对照 credit sources

seed0 final global：

| source | mean E64 | mean PR | mean batch overlap64 | worst overlap64 |
|---|---:|---:|---:|---:|
| full BP | 0.9992 | 4.5713 | 0.5154 | 0.3566 |
| adapter projected | 1.0000 | 4.0529 | 0.5576 | 0.4796 |
| random feedback | 0.9803 | 11.1272 | 0.2500 | 0.2461 |

解释：

- `adapter_projected` 比 full hidden-space 的 worst overlap 更好，符合“收缩到 adapter 子空间可能更稳”的文档预期。
- `random_feedback` 的 overlap 明显低，说明当前 overlap 指标能区分结构化 credit 与随机 credit。

### Batch-size sensitivity

seed0 full-BP credit：

| audit batch size | mean E64 | mean E128 | mean PR |
|---:|---:|---:|---:|
| 128 | 0.999997 | 1.000000 | 1.2345 |
| 256 | 0.999927 | 0.999998 | 2.7073 |
| 512 | 0.999862 | 0.999992 | 3.1812 |
| 1024 | 0.998710 | 0.999916 | 5.0373 |

解释：小 batch 会显著低估 PR，因此之后报告 Gate 0 时必须固定并公开 audit sample size。这里用 final audit 的 1024 credit samples 作为主指标。

### Gate 0 当前结论

MNIST / 4-layer MLP / hidden 256 / 5 seeds 下：

- credit energy compressibility：强。`mean E64` 约 0.999，`mean E128` 约 0.99994。
- subspace stability：弱到中间地带。`mean batch overlap64` 约 0.523，只满足 weak 的 overlap > 0.5；没有达到 medium 的 > 0.6。
- failure boundary：最后一层 top-64 子空间最不稳定，5 seeds 中 `layer3 overlap64` 都约 0.35。
- 设计含义：Gate 0 不支持直接对 full hidden-space credit 宣称稳定低秩路由；更支持 adapter / LoRA / KAN-like 子空间路线。

下一步建议：

1. 继续 Gate 0：补 Fashion-MNIST、8-layer MLP、KAN-like MLP。
2. 对比 adapter-projected credit：显式把 adapter dim 扫描为 `16,32,64`。
3. 如果 full hidden-space overlap 仍低，则 Gate 1 主线应转到 adapter / bottleneck / KAN-like block，而不是普通 MLP hidden space。

## 2026-04-28 Stage A GPU 重跑：DG-KAN 新方案

说明：上面 Gate 0 记录来自已经舍弃的旧文档路线，只作为历史记录保留；当前实验按 `docs/DG-LCA_双几何局部CreditAssignment研究方案.md` 和 `docs/DG-KAN_DG-LCA_完整实验计划.md` 的 Stage A 重新执行。

### GPU 环境修复

之前 `lca` 环境里的 PyTorch 是 CUDA 13 runtime，机器驱动只支持 CUDA 12.4，因此会退回 CPU 或无法正常使用 GPU。已停止 CPU 训练进程，并重装 CUDA 12.4 对应 PyTorch：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python -m pip uninstall -y torch cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

验证结果：

```text
torch 2.6.0+cu124
torch_cuda 12.4
available True
device0 NVIDIA RTX A5000
```

`requirement.txt` 已同步为 CUDA 12.4 wheel 源和 `torch==2.6.0+cu124`。

### Stage A 网络设计

本轮实现文件：`experiments/stage_a_dgkan.py`。

共同结构：

- 输入先经过 `Linear(in_dim, hidden_dim)` 和 `LayerNorm(hidden_dim)`。
- 主体堆叠 `depth=2` 个 block。
- 输出头为 `Linear(hidden_dim, num_classes)`。
- 本轮 S1 使用 `hidden_dim=32`、`basis_count=8`、dense KAN connectivity、fp32、Adam、weight decay `1e-4`。

对照网络：

- `mlp_adam`：`Linear -> LayerNorm -> SiLU` block，学习率 `1e-3`。
- `kan_coeff_adam`：RBF KAN coefficient block，无残差，学习率 `3e-3`。每条 edge 函数为 `phi_ji(x_i)=sum_m c_ji,m exp(-gamma (x_i-center_m)^2)`，block 输出为 `y_j=sum_i phi_ji(x_i)`。
- `dgkan_coeff_adam`：`LayerNorm -> RBF KAN -> learnable alpha -> residual add`，即 `h_{l+1}=h_l + alpha_l KAN(LN(h_l))`，`alpha` 初始为 `0.1`，学习率 `3e-3`。

本轮没有启用局部 credit 或 functional update，全部是标准 BP coefficient optimizer。这正对应 Stage A 的目标：先验证 DG-KAN 架构在普通反传下是否可训练、是否稳定，再进入局部 credit assignment。

KAN 实现选择：

- 没有直接使用 `pykan` 符号化/样条训练路线，因为 Stage A 需要 GPU batch 训练、系数优化、phi 导数与 Jacobian 稳定性审计。
- 当前采用 `awesome-kan` 中 FastKAN / efficient KAN 思路更接近的 RBF basis coefficient layer，便于在 PyTorch GPU 上高效计算 `phi`、`phi'`、`phi''` 和 block Jacobian。
- 当前结果显示问题主要不在 KAN basis 本身，而在纯 KAN block 的几何条件数；因此 Stage A 暂不需要更换 KAN 实现，优先保留 RBF KAN 并继续调 DG residual、容量、学习率和分类头。

### 执行命令

W&B project：`dg-lca`，groups：`stageA_S1_moons_mnist_5seed_gpu`、`stageA_S1_fashion_5seed_gpu`。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

for dataset in moons mnist; do
  for method in mlp_adam kan_coeff_adam dgkan_coeff_adam; do
    lr=0.003
    if [ "$method" = "mlp_adam" ]; then lr=0.001; fi
    for seed in 0 1 2 3 4; do
      /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/stage_a_dgkan.py \
        --dataset "$dataset" --method "$method" --seed "$seed" \
        --hidden-dim 32 --depth 2 --basis-count 8 --batch-size 256 \
        --lr "$lr" --weight-decay 0.0001 --device cuda \
        --wandb-mode online --wandb-project dg-lca \
        --wandb-group stageA_S1_moons_mnist_5seed_gpu \
        --run-name "stageA_S1_gpu_${dataset}_${method}_seed${seed}" \
        --out-dir results/stage_a_s1_moons_mnist_5seed_gpu
    done
  done
done
```

Fashion-MNIST 补跑使用相同配置，只把数据集和输出 group 改为：

```bash
--dataset fashion_mnist
--wandb-group stageA_S1_fashion_5seed_gpu
--out-dir results/stage_a_s1_fashion_5seed_gpu
```

Stage A 控制组补跑：

```bash
for dataset in moons mnist fashion_mnist; do
  for method in mlp_adamw residual_mlp kan_coeff_adamw dgkan_coeff_adamw; do
    lr=0.003
    if [ "$method" = "mlp_adamw" ] || [ "$method" = "residual_mlp" ]; then lr=0.001; fi
    for seed in 0 1 2 3 4; do
      # moons extra: --train-size 2000 --val-size 500 --test-size 500 --epochs 20
      /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/stage_a_dgkan.py \
        --dataset "$dataset" --method "$method" --seed "$seed" \
        --hidden-dim 32 --depth 2 --basis-count 8 --batch-size 256 \
        --lr "$lr" --weight-decay 0.0001 --device cuda \
        --wandb-mode online --wandb-project dg-lca --wandb-group stageA_S1_controls_gpu \
        --out-dir results/stage_a_s1_controls_gpu
    done
  done
done
```

实际参数补充：

- Two Moons：train/val/test = `2000/500/500`，epochs = `20`，noise = `0.12`。
- MNIST / Fashion-MNIST：train/val/test = `12000/2000/2000`，epochs = `5`。
- 输出聚合：
  `results/stage_a_s1_moons_mnist_5seed_gpu/aggregate_summary.json`、
  `results/stage_a_s1_moons_mnist_5seed_gpu/aggregate_runs.csv`、
  `results/stage_a_s1_fashion_5seed_gpu/aggregate_summary.json`、
  `results/stage_a_s1_fashion_5seed_gpu/aggregate_runs.csv`、
  `results/stage_a_s1_controls_gpu/aggregate_summary.json`、
  `results/stage_a_s1_controls_gpu/aggregate_runs.csv`、
  `results/stage_a_s1_all_methods_gpu/aggregate_summary.json`、
  `results/stage_a_s1_all_methods_gpu/aggregate_runs.csv`。

### 5-seed GPU 结果

| dataset | method | params | test acc mean±std | loss AUC | NaN | loss spikes | max Jacobian cond |
|---|---|---:|---:|---:|---:|---:|---:|
| moons | MLP-Adam | 2,466 | 0.9960±0.0032 | 0.1497 | 0.0 | 0.8 | 3.73e8 |
| moons | MLP-AdamW | 2,466 | 0.9960±0.0032 | 0.1497 | 0.0 | 0.8 | 3.16e8 |
| moons | Residual-MLP | 4,580 | 0.9404±0.0499 | 0.3128 | 0.0 | 0.0 | 1.44 |
| moons | KAN-Coeff-Adam | 16,804 | 0.9948±0.0046 | 0.0618 | 0.0 | 8.8 | 5.64e8 |
| moons | KAN-Coeff-AdamW | 16,804 | 0.9948±0.0046 | 0.0618 | 0.0 | 8.8 | 3.93e8 |
| moons | DG-KAN-Coeff-Adam | 16,804 | 0.9976±0.0017 | 0.1602 | 0.0 | 0.2 | 4.18 |
| moons | DG-KAN-Coeff-AdamW | 16,804 | 0.9976±0.0017 | 0.1602 | 0.0 | 0.2 | 4.18 |
| MNIST | MLP-Adam | 27,754 | 0.9146±0.0053 | 0.6381 | 0.0 | 0.0 | 2.29e8 |
| MNIST | MLP-AdamW | 27,754 | 0.9146±0.0053 | 0.6381 | 0.0 | 0.0 | 2.48e8 |
| MNIST | Residual-MLP | 29,868 | 0.9073±0.0085 | 0.4349 | 0.0 | 1.0 | 2.04 |
| MNIST | KAN-Coeff-Adam | 42,092 | 0.9362±0.0060 | 0.5054 | 0.0 | 1.0 | 4.21e8 |
| MNIST | KAN-Coeff-AdamW | 42,092 | 0.9362±0.0060 | 0.5054 | 0.0 | 1.0 | 3.12e8 |
| MNIST | DG-KAN-Coeff-Adam | 42,092 | 0.9231±0.0078 | 0.3195 | 0.0 | 1.4 | 2.00 |
| MNIST | DG-KAN-Coeff-AdamW | 42,092 | 0.9231±0.0078 | 0.3195 | 0.0 | 1.4 | 2.00 |
| Fashion-MNIST | MLP-Adam | 27,754 | 0.8403±0.0112 | 0.7716 | 0.0 | 0.0 | 1.86e8 |
| Fashion-MNIST | MLP-AdamW | 27,754 | 0.8403±0.0112 | 0.7717 | 0.0 | 0.0 | 2.35e8 |
| Fashion-MNIST | Residual-MLP | 29,868 | 0.8379±0.0089 | 0.5692 | 0.0 | 0.2 | 1.97 |
| Fashion-MNIST | KAN-Coeff-Adam | 42,092 | 0.8535±0.0076 | 0.6791 | 0.0 | 0.0 | 1.54e8 |
| Fashion-MNIST | KAN-Coeff-AdamW | 42,092 | 0.8535±0.0076 | 0.6791 | 0.0 | 0.0 | 3.76e8 |
| Fashion-MNIST | DG-KAN-Coeff-Adam | 42,092 | 0.8438±0.0105 | 0.5002 | 0.0 | 0.0 | 1.60 |
| Fashion-MNIST | DG-KAN-Coeff-AdamW | 42,092 | 0.8438±0.0105 | 0.5002 | 0.0 | 0.0 | 1.60 |

### 当前结论

1. Stage A S1 的 GPU 可训练性成立：105 个正式 run 全部在 `cuda` 上完成，`nan_or_inf_count=0`。
2. DG-KAN 残差几何是有效的：纯 KAN 在 Two Moons、MNIST、Fashion-MNIST 上的 block Jacobian condition 都达到 `1e8` 量级，DG-KAN 分别降到约 `4.18`、`2.00`、`1.60`。这说明 `h + alpha KAN(LN(h))` 对局部几何稳定性有实质作用。
3. 精度上 DG-KAN 尚未全面胜出：Two Moons 上 DG-KAN 最好；MNIST 和 Fashion-MNIST 上纯 KAN 测试精度更高，DG-KAN 介于 MLP 和 KAN 之间。但 DG-KAN 的 MNIST / Fashion-MNIST loss AUC 明显低于纯 KAN，说明收敛轨迹更稳，只是最终分类精度还需要调容量或 head。
4. AdamW 不改变主结论：MLP AdamW 和 MLP Adam 基本一致，KAN AdamW 和 KAN Adam 基本一致，DG-KAN AdamW 和 DG-KAN Adam 基本一致。因此当前差异主要来自 block 几何结构，不是 optimizer。
5. Residual-MLP 是必要负对照：它的 Jacobian condition 也很低，但精度弱于 KAN 和 DG-KAN，说明“残差稳定”本身不够；DG-KAN 仍需要 KAN edge function 的表达力。
6. 不建议现在换 KAN 实现：当前 RBF-KAN 已能验证关键结构差异；纯 KAN 条件数差、DG-KAN 条件数好，说明实验正在测到我们关心的几何效应。下一步应先调 DG-KAN 设计，而不是换库。
7. 需要继续实验：做 depth `2/4/8`、hidden `32/64/128`、basis `8/16` 的 Stage A sweep；对 DG-KAN 尝试更大 hidden、较低 lr、alpha init sweep 和更强分类 head。

### 验证

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python -m py_compile experiments/stage_a_dgkan.py
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

结果：脚本编译通过，CUDA 可用。

## 2026-04-28 Stage A alpha_k=1 重跑

动机：上一轮 DG-KAN 的 `alpha_k` 从 `0.1` 初始化，训练后约为 `0.5-0.74`。为验证更强 KAN branch 是否提升分类性能，将默认 `--residual-alpha-init` 改为 `1.0`，并重跑 DG-KAN 相关实验。

代码变更：

```text
experiments/stage_a_dgkan.py
--residual-alpha-init default: 0.1 -> 1.0
```

执行范围：

- 数据集：Two Moons、MNIST、Fashion-MNIST。
- 模型：`dgkan_coeff_adam`、`dgkan_coeff_adamw`。
- seeds：`0,1,2,3,4`。
- 共 `30` 个正式 GPU run，全部 `cuda`。
- 输出：`results/stage_a_s1_dgkan_alpha1_gpu/aggregate_summary.json`、`aggregate_runs.csv`。

执行命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

for dataset in moons mnist fashion_mnist; do
  for method in dgkan_coeff_adam dgkan_coeff_adamw; do
    for seed in 0 1 2 3 4; do
      /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/stage_a_dgkan.py \
        --dataset "$dataset" --method "$method" --seed "$seed" \
        --hidden-dim 32 --depth 2 --basis-count 8 --batch-size 256 \
        --lr 0.003 --weight-decay 0.0001 \
        --residual-alpha-init 1.0 --device cuda \
        --wandb-mode online --wandb-project dg-lca \
        --wandb-group stageA_S1_dgkan_alpha1_gpu \
        --out-dir results/stage_a_s1_dgkan_alpha1_gpu
    done
  done
done
```

Two Moons 额外使用 `--train-size 2000 --val-size 500 --test-size 500 --epochs 20`，MNIST/Fashion-MNIST 使用默认 `12000/2000/2000` 和 `epochs=5`。

### alpha=1 结果

| dataset | method | test acc mean±std | loss AUC | NaN | loss spikes | max Jacobian cond | final alpha_0 | final alpha_1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| moons | DG-KAN-Adam | 0.9976±0.0017 | 0.0971 | 0.0 | 3.2 | 5.09 | 1.263 | 1.279 |
| moons | DG-KAN-AdamW | 0.9976±0.0017 | 0.0971 | 0.0 | 3.2 | 5.09 | 1.263 | 1.279 |
| MNIST | DG-KAN-Adam | 0.9289±0.0080 | 0.2855 | 0.0 | 1.2 | 2.65 | 1.605 | 1.612 |
| MNIST | DG-KAN-AdamW | 0.9289±0.0080 | 0.2855 | 0.0 | 1.2 | 2.65 | 1.605 | 1.612 |
| Fashion-MNIST | DG-KAN-Adam | 0.8502±0.0099 | 0.4790 | 0.0 | 0.0 | 2.10 | 1.489 | 1.472 |
| Fashion-MNIST | DG-KAN-AdamW | 0.8502±0.0099 | 0.4790 | 0.0 | 0.0 | 2.10 | 1.489 | 1.472 |

### 与 alpha=0.1 对比

| dataset | alpha init | test acc | loss AUC | max Jacobian cond | final alpha_0 / alpha_1 |
|---|---:|---:|---:|---:|---:|
| moons | 0.1 | 0.9976 | 0.1602 | 4.18 | 0.496 / 0.510 |
| moons | 1.0 | 0.9976 | 0.0971 | 5.09 | 1.263 / 1.279 |
| MNIST | 0.1 | 0.9231 | 0.3195 | 2.00 | 0.727 / 0.735 |
| MNIST | 1.0 | 0.9289 | 0.2855 | 2.65 | 1.605 / 1.612 |
| Fashion-MNIST | 0.1 | 0.8438 | 0.5002 | 1.60 | 0.622 / 0.614 |
| Fashion-MNIST | 1.0 | 0.8502 | 0.4790 | 2.10 | 1.489 / 1.472 |

### alpha=1 结论

1. `alpha_k=1` 没有导致训练发散：30 个 GPU run 全部无 NaN/Inf。
2. 更大的 KAN branch 权重提升了 MNIST 和 Fashion-MNIST 精度，并降低 loss AUC；Two Moons 精度持平但收敛更快。
3. 几何稳定性略有下降但仍健康：Jacobian condition 从约 `1.6-4.2` 升到 `2.1-5.1`，仍远低于纯 KAN 的 `1e8` 量级。
4. 训练后 `alpha_k` 会继续变大：MNIST 约 `1.61`，Fashion-MNIST 约 `1.48`，Two Moons 约 `1.27`。这说明模型确实需要更强的 KAN branch，而不是依赖 identity path。
5. 当前建议：保留默认 `alpha_init=1.0` 进入下一轮 sweep；后续可单独扫 `alpha_init in {0.5,1.0,1.5,2.0}`，但 `0.1` 已不再是优先配置。

## 2026-04-28 Stage B 回归 pilot：Functional-space update with true BP credit

依据 `docs/DG-LCA_双几何局部CreditAssignment研究方案.md` 和 `docs/DG-KAN_DG-LCA_完整实验计划.md`，Stage B 先不引入 learned router，只使用真实 BP credit，比较 KAN edge function 的 coefficient-space 更新和 functional-space 更新。

代码：

- 主脚本：`experiments/stage_b_functional.py`
- 输出目录：`results/stage_b_regression_pilot_v2_gpu/`
- 汇总文件：
  - `results/stage_b_regression_pilot_v2_gpu/aggregate_runs.csv`
  - `results/stage_b_regression_pilot_v2_gpu/aggregate_summary.csv`
  - `results/stage_b_regression_pilot_v2_gpu/aggregate_summary.json`
- W&B group：`stageB_regression_pilot_v2_gpu`
- 设备：`cuda`，实际记录为 `NVIDIA RTX A5000`

### 网络与更新设计

本轮使用回归版 1-layer RBF-KAN：

```text
x -> RBFKANLayer -> readout
phi(t) = sum_m a_m exp(-((t-c_m)/sigma)^2)
basis_count = 16
grid = [-pi, pi]
```

对于 1D 任务，`readout` 是 identity；对于 `sincos2d`，`RBFKANLayer` 先对 2 个输入维度分别做 edge function，再接一个 linear readout 得到标量输出。这个版本用于 Stage B pilot，后续更严格的 KAN 可改成直接 `out_dim x in_dim x basis` 的 edge bank。

比较的更新：

- `coeff_sgd`：普通 coefficient SGD，`a <- a - lr grad_a L`
- `coeff_adam`：Adam on coefficients/parameters
- `coeff_adamw`：AdamW on coefficients/parameters
- `diag`：使用 Sobolev metric 的 diagonal preconditioner
- `sobolev`：`a <- a - lr (M_Sob + rho I)^-1 grad_a L`
- `rkhs`：`a <- a - lr (M_RKHS + rho I)^-1 grad_a L`

Sobolev metric：

```text
M_Sob = int B B^T + alpha int B' B'^T + beta int B'' B''^T
alpha = 0.01
beta = 0.0001
rho = 0.001
```

记录指标：

- final test MSE
- loss AUC
- sign agreement
- predicted-negative-actual-positive bad steps
- `phi_prime_p95`
- curvature：`kan/phi_double_prime_energy_mean`
- preconditioner condition

### 重要修正

第一版 pilot 写入了 `results/stage_b_regression_pilot_gpu/`，但发现 `coeff_sgd` 分支误用了 metric inverse，因此它不是纯 coefficient SGD，并且会和 `sobolev` 结果重合。该版本只作为调试记录，不作为结论依据。

已修正：

```text
coeff_sgd -> direction = grad
sobolev   -> direction = (M_Sob + rho I)^-1 grad
diag      -> direction = diag(M_Sob + rho I)^-1 grad
rkhs      -> direction = (M_RKHS + rho I)^-1 grad
```

同时给 summary 增加了 resolved device 和 CUDA device name，并加了 `--disable-progress` 避免 tqdm 污染日志。

### 执行命令

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore
OUT=results/stage_b_regression_pilot_v2_gpu

for task in sin1d sin5_1d sincos2d; do
  for update in coeff_sgd coeff_adam coeff_adamw diag sobolev rkhs; do
    lr=0.003
    if [ "$update" = "coeff_sgd" ] || [ "$update" = "diag" ] || [ "$update" = "sobolev" ] || [ "$update" = "rkhs" ]; then lr=0.01; fi
    for seed in 0 1 2; do
      /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca python experiments/stage_b_functional.py \
        --task "$task" --model kan --update "$update" --seed "$seed" \
        --train-size 2048 --val-size 512 --test-size 512 --epochs 80 \
        --batch-size 256 --basis-count 16 --grid-min -3.14159265 --grid-max 3.14159265 \
        --lr "$lr" --rho 0.001 --sobolev-alpha 0.01 --sobolev-beta 0.0001 \
        --device cuda --wandb-mode online --wandb-project dg-lca \
        --wandb-group stageB_regression_pilot_v2_gpu \
        --run-name "stageB_v2_reg_${task}_${update}_seed${seed}" \
        --out-dir "$OUT" --disable-progress --print-level brief
    done
  done
done
```

### v2 结果

3 tasks x 6 updates x 3 seeds，共 `54` 个 GPU run，全部 sign agreement = `1.0`，bad steps = `0`。

| task | update | test MSE | loss AUC | curvature | AUC vs Adam | test MSE vs Adam |
|---|---|---:|---:|---:|---:|---:|
| sin1d | coeff_sgd | 4.42e-3 | 9.77e-2 | 0.543 | -102.6% | -11147.5% |
| sin1d | coeff_adam | 3.93e-5 | 4.82e-2 | 0.882 | 0.0% | 0.0% |
| sin1d | coeff_adamw | 3.93e-5 | 4.82e-2 | 0.882 | -0.0% | -0.0% |
| sin1d | diag | 3.90e-5 | 9.13e-3 | 0.838 | +81.1% | +0.9% |
| sin1d | sobolev | 1.87e-5 | 1.93e-2 | 0.897 | +60.0% | +52.5% |
| sin1d | rkhs | 1.24e-1 | 2.59e-1 | 2.368 | -436.3% | -314536.0% |
| sin5_1d | coeff_sgd | 2.61e-2 | 1.27e-1 | 2.011 | -87.2% | -412.3% |
| sin5_1d | coeff_adam | 5.10e-3 | 6.81e-2 | 14.658 | 0.0% | 0.0% |
| sin5_1d | coeff_adamw | 5.10e-3 | 6.81e-2 | 14.656 | -0.0% | -0.0% |
| sin5_1d | diag | 8.01e-4 | 1.64e-2 | 23.523 | +75.9% | +84.3% |
| sin5_1d | sobolev | 1.94e-4 | 2.17e-2 | 29.579 | +68.1% | +96.2% |
| sin5_1d | rkhs | 1.26e-1 | 2.62e-1 | 36.928 | -284.5% | -2379.9% |
| sincos2d | coeff_sgd | 6.29e-3 | 4.19e-1 | 0.208 | -285.0% | -2978.6% |
| sincos2d | coeff_adam | 2.04e-4 | 1.09e-1 | 0.737 | 0.0% | 0.0% |
| sincos2d | coeff_adamw | 2.04e-4 | 1.09e-1 | 0.737 | -0.0% | -0.0% |
| sincos2d | diag | 2.41e-4 | 7.84e-2 | 0.789 | +27.9% | -18.1% |
| sincos2d | sobolev | 1.11e-4 | 1.33e-1 | 0.623 | -22.1% | +45.7% |
| sincos2d | rkhs | 3.80e-1 | 7.70e-1 | 3.822 | -608.0% | -185728.4% |

### Stage B 当前结论

1. Functional update 的实现成立：所有 run 的 sign agreement 都是 `1.0`，没有出现 predicted descent 但实际 loss 上升的 bad step。
2. 纯 `coeff_sgd` 明显不够：三个函数任务都比 Adam 慢很多，最终误差也差很多。
3. `diag` 是当前最稳的加速器：三个任务的 loss AUC 都优于 Adam，其中 `sin1d` 降低 `81.1%`，`sin5_1d` 降低 `75.9%`，`sincos2d` 降低 `27.9%`。
4. `sobolev` 的最终拟合能力最好：三个任务 test MSE 都优于 Adam，尤其 `sin5_1d` 提升 `96.2%`，`sincos2d` 提升 `45.7%`；但它的 AUC 不总是最好，说明 full inverse 更容易走出更大、更慢的函数空间轨迹。
5. `rkhs` 当前不能用：在三个任务上都显著劣化，尤其二维任务 test MSE 到 `3.8e-1`。原因很可能是当前 Gaussian RKHS metric 的 length-scale/damping 与 RBF basis 不匹配，逆矩阵把更新推向了坏方向；下一轮如果保留 RKHS，必须扫 `ell`、`rho` 和 lr，不能直接进入主实验。
6. Medium success 在回归 pilot 上成立：至少两个数据集有超过 `10%` 的 loss AUC 降低，并且 sign agreement > `0.75`。更严格的 MNIST/Fashion 结论仍需 Stage B classification sanity。
7. 下一步：先做 Two Moons DG-KAN sanity，比较 `coeff_adam`、`diag`、`sobolev`，暂时不把 RKHS 放进主线；若 Two Moons 也成立，再上 MNIST/Fashion。

## 2026-04-29 Stage B Two Moons DG-KAN sanity

目标：在分类任务上验证 functional update 是否仍能维持 DG-KAN 的精度和几何稳定性。模型使用 Stage A 当前推荐配置：

```text
model = DG-KAN
block = h + alpha_k * RBFKAN(LayerNorm(h))
hidden_dim = 32
depth = 2
basis_count = 8
alpha_init = 1.0
dataset = Two Moons
train / val / test = 2000 / 500 / 500
epochs = 20
device = cuda, NVIDIA RTX A5000
```

### sanity v1：直接 functional update

输出：`results/stage_b_moons_dgkan_sanity_v1_gpu/`

设置：

- `coeff_adam`: Adam baseline，`lr=0.003`
- `diag`: KAN coeff 用 diagonal Sobolev preconditioner，其余参数也用手动 SGD，`lr=0.003`
- `sobolev`: KAN coeff 用 full Sobolev inverse，其余参数也用手动 SGD，`lr=0.003`

结果：

| update | test acc | test loss | loss AUC | sign agreement | bad steps | max Jacobian cond | curvature |
|---|---:|---:|---:|---:|---:|---:|---:|
| coeff_adam | 0.9972 | 0.0066 | 0.0979 | 0.986 | 1.4 | 5.13 | 0.002182 |
| diag | 0.8768 | 0.3120 | 0.4186 | 1.000 | 0.0 | 1.11 | 0.000190 |
| sobolev | 0.8796 | 0.3068 | 0.4158 | 1.000 | 0.0 | 1.11 | 0.000196 |

结论：v1 functional update 非常稳定，但严重欠拟合。主要原因不是 DG-KAN 几何坏掉，而是实验设置不公平：functional 分支把非 KAN 参数，包括 LayerNorm、residual alpha、classifier head，也一起用普通 SGD 更新；而 baseline 是 Adam。

### hybrid v2：KAN coeff functional update，其余参数 Adam

代码变更：

```text
--rest-optimizer {sgd, adam, adamw, none}
--rest-lr

diag/sobolev/rkhs/coeff_sgd:
  KAN coeff 使用指定 coefficient update
  非 KAN coeff 参数可由 rest optimizer 更新

本轮使用：
  KAN coeff lr = 0.01
  rest optimizer = Adam
  rest lr = 0.003
```

输出：`results/stage_b_moons_dgkan_hybrid_v2_gpu/`

| update | rest optimizer | test acc | test loss | loss AUC | sign agreement | bad steps | max Jacobian cond | curvature |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coeff_adam | full Adam | 0.9972 | 0.0066 | 0.0979 | 0.986 | 1.4 | 5.13 | 0.002182 |
| diag | Adam rest | 0.9116 | 0.2199 | 0.2935 | 0.992 | 1.0 | 1.25 | 0.000195 |
| sobolev | Adam rest | 0.9608 | 0.1350 | 0.2678 | 1.000 | 0.0 | 1.46 | 0.000297 |

结论：hybrid 明显优于 v1，但 `lr=0.01` 仍然太保守；KAN edge derivative 很小，`phi_prime_p95` 约 `0.013-0.015`，说明 KAN branch 被压得太弱。

### coefficient lr sweep

输出：`results/stage_b_moons_dgkan_hybrid_lrsweep_gpu/`

设置：

- `diag/sobolev`
- `lr in {0.03, 0.1}`
- `rest_optimizer=adam`
- `rest_lr=0.003`
- `lr=0.03` 跑 3 seeds，`lr=0.1` 补齐 5 seeds

结果：

| update | coeff lr | seeds | test acc | test loss | loss AUC | sign agreement | bad steps | max Jacobian cond | curvature | phi_prime_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diag | 0.03 | 3 | 0.9647 | 0.1095 | 0.2468 | 1.000 | 0.0 | 1.50 | 0.000231 | 0.01521 |
| diag | 0.10 | 5 | 0.9960 | 0.0243 | 0.1828 | 1.000 | 0.0 | 2.13 | 0.000402 | 0.02177 |
| sobolev | 0.03 | 3 | 0.9947 | 0.0368 | 0.1938 | 1.000 | 0.0 | 2.03 | 0.000748 | 0.02086 |
| sobolev | 0.10 | 5 | 0.9972 | 0.0132 | 0.1295 | 1.000 | 0.0 | 2.98 | 0.002128 | 0.03428 |

与 Adam baseline 对比：

| update | test acc | loss AUC | bad steps | max Jacobian cond |
|---|---:|---:|---:|---:|
| coeff_adam | 0.9972 | 0.0979 | 1.4 | 5.13 |
| sobolev lr=0.10 + Adam rest | 0.9972 | 0.1295 | 0.0 | 2.98 |
| diag lr=0.10 + Adam rest | 0.9960 | 0.1828 | 0.0 | 2.13 |

### Two Moons sanity 结论

1. Functional update 分类 sanity 成立，但需要 hybrid rest optimizer 和更大的 coefficient lr。
2. `sobolev lr=0.1 + Adam rest` 在 Two Moons 上追平 Adam baseline 的 test acc，同时 bad steps 从 `1.4` 降到 `0.0`，Jacobian condition 从 `5.13` 降到 `2.98`。
3. `diag lr=0.1 + Adam rest` 也接近 baseline，acc `0.9960`，condition `2.13`，但 AUC 比 Sobolev 更差。
4. 回归任务里 `diag` 的 AUC 更好，Two Moons 分类里 `sobolev` 更好；这说明 Stage B 后续需要按任务类型分别调 coefficient lr，而不是固定一套超参。
5. 当前不建议直接上 MNIST/Fashion 主实验：先做更系统的 Two Moons lr/rho sweep，至少扫 `coeff_lr in {0.03, 0.1, 0.2}` 和 `rho in {1e-4, 1e-3, 1e-2}`。RKHS 仍暂停。

## 2026-04-29 Stage B-next：functional-space update 系统实验

依据：`docs/DG-KAN_StageB-next_详细实验计划.md`

输出目录：

```text
results/stage_b_next_gpu0/
  aggregate_runs.csv
  aggregate_summary.csv
  aggregate_runs.json
  aggregate_summary.json
  runs/*/summary.json
```

GPU / 环境确认：

```text
CUDA_VISIBLE_DEVICES=0
torch = 2.6.0+cu124
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
```

执行命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca \
  python experiments/run_stage_b_next.py \
  --out-dir results/stage_b_next_gpu0 \
  --plans b0,b1,b2,b3,b4 \
  --seeds 0,1,2 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

完成情况：

```text
total runs = 147
failed runs = 0
B0_calibration = 48
B1_diag_to_sobolev = 27
B2_classification_hybrid = 30
B3_sobolev_sweep = 30
B4_rkhs_repair = 12
```

### 本轮代码与 logging 变更

新增 `experiments/run_stage_b_next.py` 批量执行与聚合脚本。

更新 `experiments/stage_b_functional.py`，补齐 Stage B-next 要求的关键指标：

- `generalization/train_test_gap_loss`
- `generalization/val_train_gap_loss`
- `generalization/val_train_gap_acc`
- `conv/loss_auc_train`
- `conv/loss_auc_val`
- `conv/loss_at_10pct_steps`
- `conv/loss_at_25pct_steps`
- `conv/loss_at_50pct_steps`
- `conv/loss_at_75pct_steps`
- `descent/ratio_median`
- `descent/ratio_p10`
- `descent/pred_actual_corr`
- `descent/bad_step_total_positive_loss_increase`
- `kan/target_curvature`
- `kan/excess_curvature`
- `metric/eig_min`
- `metric/eig_max`
- `metric/eig_p05`
- `metric/eig_p95`
- `metric/update_amplification`
- `rkhs/kernel_lengthscale`
- `rkhs/raw_condition`
- `rkhs/damped_condition`
- `rkhs/allowed_in_main_comparison`
- `perf/step_time_ms`
- `perf/samples_per_sec`
- `memory/peak_allocated_mb`

新增 update：

```text
diag_to_sobolev:
  first T_w steps: diag(M_sob + rho I)^-1 grad
  remaining steps: (M_sob + rho I)^-1 grad

soft_mix_diag_sobolev:
  direction = (1 - lambda_t) * diag_direction + lambda_t * sobolev_direction
```

本轮实际主用 hard `diag_to_sobolev`。

### 网络设计

回归任务使用 `RegressionKAN`：

```text
input x
  -> RBFKANLayer
  -> readout
  -> y_hat
```

KAN edge function：

```text
phi_ji(t) = sum_m a_ji,m exp(-gamma * (t - center_m)^2)
```

回归配置：

```text
tasks = sin1d, sin5_1d, sincos2d
basis_count = 16
grid = [-pi, pi]
train / val / test = 2048 / 512 / 512
epochs = 80
batch_size = 256
```

分类任务使用 residual DG-KAN：

```text
x
  -> Linear stem
  -> LayerNorm
  -> [h <- h + alpha_k * RBFKAN(LayerNorm(h))] * 2
  -> Linear head
```

分类配置：

```text
hidden_dim = 32
depth = 2
basis_count = 8
alpha_init = 1.0
Two Moons train / val / test = 2000 / 500 / 500, epochs = 20
MNIST train / val / test = 6000 / 1000 / 1000, epochs = 5
```

### B0：Pilot reproduction & calibration

目标：确认旧 pilot 趋势仍成立，并验证新增指标都能落盘。

回归结果：

| task | update | test loss | val AUC | sign | bad steps | ratio median | precond cond |
|---|---:|---:|---:|---:|---:|---:|---:|
| sin1d | coeff_adamw | 3.93e-5 | 4.68e-2 | 1.000 | 0 | 0.995 | 29.65 |
| sin1d | diag | 3.91e-5 | 6.35e-3 | 1.000 | 0 | 0.989 | 29.65 |
| sin1d | sobolev | 1.87e-5 | 1.68e-2 | 1.000 | 0 | 0.990 | 29.65 |
| sin1d | rkhs_raw | 1.24e-1 | 2.68e-1 | 1.000 | 0 | 0.943 | 4720.92 |
| sin5_1d | coeff_adamw | 5.10e-3 | 6.76e-2 | 1.000 | 0 | 0.997 | 29.65 |
| sin5_1d | diag | 8.02e-4 | 1.34e-2 | 1.000 | 0 | 0.992 | 29.65 |
| sin5_1d | sobolev | 1.94e-4 | 1.91e-2 | 1.000 | 0 | 0.990 | 29.65 |
| sin5_1d | rkhs_raw | 1.26e-1 | 2.71e-1 | 1.000 | 0 | 0.943 | 4720.92 |
| sincos2d | coeff_adamw | 2.04e-4 | 9.63e-2 | 1.000 | 0 | 0.990 | 29.65 |
| sincos2d | diag | 2.47e-4 | 6.84e-2 | 1.000 | 0 | 0.990 | 29.65 |
| sincos2d | sobolev | 1.11e-4 | 1.10e-1 | 1.000 | 0 | 0.989 | 29.65 |
| sincos2d | rkhs_raw | 1.53e-2 | 4.45e-1 | 1.000 | 0 | 0.860 | 4720.92 |

B0 回归结论：

1. `diag` 仍然是最快下降方法，val AUC 在三个回归任务上都优于 AdamW。
2. `sobolev` 仍然是最终 test loss 最好的主方法，尤其 `sin5_1d` 和 `sincos2d`。
3. `rkhs_raw` 的条件数约 `4720.92`，回归任务显著失败。
4. 新增 descent 指标显示 sign agreement 饱和为 `1.0`，所以后续应看 `ratio_median` 和 `pred_actual_corr`。

Two Moons calibration：

| update | test acc | test loss | val AUC | bad steps | max J cond | phi_prime_p95 |
|---|---:|---:|---:|---:|---:|---:|
| coeff_adamw | 0.9973 | 0.0060 | 0.0805 | 1.0 | 4.86 | 0.0613 |
| diag | 0.9960 | 0.0239 | 0.1677 | 0.0 | 2.05 | 0.0215 |
| sobolev | 0.9967 | 0.0125 | 0.1140 | 0.0 | 2.91 | 0.0337 |
| rkhs_raw | 0.9973 | 0.0112 | 0.1033 | 0.0 | 3.06 | 0.0403 |

Two Moons 上 hybrid functional update 能接近 AdamW accuracy，同时明显降低 Jacobian condition 和 `phi_prime_p95`。RKHS 在该低维分类任务上没有直接崩，但 preconditioner condition 仍然高，不进入主线。

### B1：diag -> Sobolev schedule

目标：验证 `diag` 的早期速度能否和 `Sobolev` 的最终质量结合。

结果：

| task | warmup frac | test loss | val AUC | test / sobolev | AUC / diag | weak success |
|---|---:|---:|---:|---:|---:|---|
| sin1d | 0.10 | 1.87e-5 | 6.34e-3 | 0.999 | 0.998 | yes |
| sin1d | 0.25 | 1.87e-5 | 6.32e-3 | 0.999 | 0.995 | yes |
| sin1d | 0.50 | 1.88e-5 | 6.34e-3 | 1.003 | 0.998 | yes |
| sin5_1d | 0.10 | 1.94e-4 | 1.03e-2 | 1.000 | 0.769 | yes |
| sin5_1d | 0.25 | 1.94e-4 | 1.18e-2 | 1.001 | 0.880 | yes |
| sin5_1d | 0.50 | 1.96e-4 | 1.29e-2 | 1.012 | 0.960 | yes |
| sincos2d | 0.10 | 1.11e-4 | 7.36e-2 | 1.004 | 1.077 | yes |
| sincos2d | 0.25 | 1.12e-4 | 6.83e-2 | 1.012 | 0.998 | yes |
| sincos2d | 0.50 | 1.18e-4 | 6.83e-2 | 1.064 | 0.999 | yes |

B1 结论：

1. `diag_to_sobolev` 成立。所有 9 个 task/warmup 组合都满足 weak success：`test_loss <= 1.1 * Sobolev` 且 `val_AUC <= 1.1 * diag`。
2. `warmup_frac=0.10` 在 `sin5_1d` 上最好，val AUC 是 pure diag 的 `0.769x`，同时 final test loss 与 Sobolev 持平。
3. `warmup_frac=0.25` 在 `sincos2d` 上最好，几乎复现 diag 的速度，同时 test loss 只比 Sobolev 高约 `1.2%`。
4. 推荐下一轮默认 schedule：`warmup_frac in {0.10, 0.25}`，不优先用 `0.50`。

### B2：classification hybrid

目标：验证 functional update 是否应只作用于 KAN coefficients，非 KAN 参数用 AdamW。

Two Moons：

| method | test acc | test loss | val AUC | train acc | bad steps | max J cond | phi_prime_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9973 | 0.0060 | 0.0805 | 0.9982 | 1.0 | 4.86 | 0.0613 |
| all_functional_other_sgd | 0.8753 | 0.3083 | 0.3908 | 0.8840 | 0.0 | 1.11 | 0.0135 |
| hybrid_diag | 0.9960 | 0.0239 | 0.1677 | 0.9970 | 0.0 | 2.05 | 0.0215 |
| hybrid_sobolev | 0.9967 | 0.0125 | 0.1140 | 0.9978 | 0.0 | 2.91 | 0.0337 |
| hybrid_diag_to_sobolev | 0.9967 | 0.0145 | 0.1434 | 0.9978 | 0.0 | 2.74 | 0.0291 |

MNIST small：

| method | test acc | test loss | val AUC | train acc | max J cond | phi_prime_p95 |
|---|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9133 | 0.2731 | 0.3548 | 0.9689 | 1.79 | 0.0440 |
| all_functional_other_sgd | 0.7827 | 1.0512 | 1.3019 | 0.7861 | 1.11 | 0.0134 |
| hybrid_diag | 0.8987 | 0.3876 | 0.5466 | 0.9284 | 1.12 | 0.0137 |
| hybrid_sobolev | 0.8993 | 0.3882 | 0.5476 | 0.9287 | 1.12 | 0.0139 |
| hybrid_diag_to_sobolev | 0.8990 | 0.3880 | 0.5469 | 0.9287 | 1.12 | 0.0139 |

B2 结论：

1. `all_functional_other_sgd` 在 Two Moons 和 MNIST 都明显欠拟合，说明 functional update 不应粗暴扩展到非 KAN 参数。
2. Two Moons 上 hybrid 成立：accuracy 接近 AdamW，同时 bad steps 从 AdamW 的 `1.0` 降到 `0.0`，Jacobian condition 明显降低。
3. MNIST small 上 hybrid 还没有追上 all AdamW，主要表现为 train acc 也低，属于优化/容量/学习率问题，不是泛化 gap 问题。
4. MNIST 上 hybrid 的几何更稳定：Jacobian condition 约 `1.12`，低于 AdamW 的 `1.79`；`phi_prime_p95` 约 `0.014`，低于 AdamW 的 `0.044`。下一轮应针对 MNIST 单独 sweep `coeff_lr`、`rest_lr`、epochs 和 hidden_dim。

### B3：Sobolev metric sweep

目标：在 `sin5_1d` 上扫 Sobolev `alpha` 与 `rho`，判断平滑项是否过强。

Top 区域：

| alpha | rho | test loss | val AUC | phi_prime_p95 | curvature | precond cond |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1e-4 | 1.99e-4 | 1.81e-2 | 1.936 | 29.59 | 48.20 |
| 1e-3 | 1e-4 | 1.99e-4 | 1.82e-2 | 1.936 | 29.59 | 46.24 |
| 0 | 1e-3 | 1.99e-4 | 1.83e-2 | 1.936 | 29.59 | 39.95 |
| 1e-2 | 1e-3 | 1.99e-4 | 1.88e-2 | 1.936 | 29.58 | 29.65 |
| 1e-1 | 1e-3 | 3.42e-4 | 2.41e-2 | 1.913 | 26.36 | 9.14 |
| 1 | 1e-3 | 1.64e-2 | 5.74e-2 | 1.407 | 3.92 | 5.91 |

B3 结论：

1. `alpha` 过大确实会过度平滑高频任务。`alpha=1` 把 curvature 压到约 `3.9`，但 `sin5_1d` test loss 劣化到 `1.6e-2`。
2. `alpha in [0, 1e-2]` 是稳定区，默认 `alpha=1e-2, rho=1e-3` 接近最优。
3. 对高频任务，不应该把 Sobolev 理解成“越平滑越好”。合理 curvature 是必要的。

### B4：RKHS conditioning repair

目标：验证 RKHS 是否只是因为 condition number 病态而失败。

结果摘要：

| rho | ell mult | condition | allowed | test loss | val AUC |
|---:|---:|---:|---:|---:|---:|
| 1e-3 | 2.0 | 4720.92 | 0 | 0.4026 | 0.4655 |
| 1e-2 | 2.0 | 473.90 | 1 | 0.4029 | 0.4623 |
| 1e-1 | 2.0 | 48.30 | 1 | 0.4257 | 0.4705 |
| 1.0 | 0.5 | 1.31 | 1 | 0.3982 | 0.4509 |
| 1e-3 | 0.5 | 1.72 | 1 | 0.3130 | 0.4006 |

B4 结论：

1. RKHS 的 condition number 可以通过 `rho` 与 lengthscale 修复到 `<500`。
2. 但修复 condition 后 test loss 仍在 `0.31-0.46`，远差于 Sobolev 的约 `2e-4`。
3. 当前 RKHS 失败不只是 conditioning，而是 kernel metric 与当前 RBF basis / 任务尺度不匹配，或 damping 后更新过度保守。
4. RKHS 不进入下一阶段主线，只保留 quarantine track。

### Stage B-next 总结

1. Stage B-next 的核心结论是：`diag -> Sobolev` schedule 成立，推荐默认 `warmup_frac=0.10` 或 `0.25`。
2. KAN coefficients 使用 functional update、非 KAN 参数使用 AdamW 是分类任务的正确方向；直接用 functional + SGD rest 会欠拟合。
3. Sobolev 的价值不是单纯压低 curvature，而是保持合理函数几何。高频任务上过强 `alpha` 会伤害拟合。
4. RKHS 即使修复 condition 也暂时不值得进入主线。
5. 下一步优先做 MNIST hybrid 的专门 sweep：`coeff_lr in {0.03, 0.1, 0.2, 0.3}`，`rest_lr in {3e-4, 1e-3, 3e-3}`，并把 epochs 从 `5` 提到 `10-20`。

## 2026-04-29 Stage B-next：MNIST 分类补充实验

依据：`docs/DG-KAN_StageB-next_MNIST分类补充实验计划.md`

目标：解释 Stage B-next 中 MNIST small 上 hybrid functional update 没追上 AdamW 的原因，重点判断是否属于：

```text
KAN branch under-active
functional update effective step 太小
缺少 Adam-style moment dynamics
capacity / basis coverage 问题
```

输出目录：

```text
results/stage_b_next_mnist_drilldown_gpu0/
results/stage_b_next_mnist_drilldown_confirm_gpu0/
results/stage_b_next_mnist_drilldown_pareto_gpu0/
```

GPU / 环境确认：

```text
CUDA_VISIBLE_DEVICES=0
torch = 2.6.0+cu124
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
```

### 代码变更

更新 `experiments/stage_b_functional.py`：

- 增加分类诊断：
  - `branch/global/mean_output_norm_ratio`
  - `branch/layer_k/output_norm_ratio`
  - `ablation/no_kan_branch_{val,test}_acc`
  - `ablation/{val,test}_acc_drop_when_disabling_kan`
  - `ablation/{val,test}_kan_branch_logit_delta_norm`
  - `basis/global/active_basis_fraction`
  - `basis/global/dead_basis_fraction`
  - `kan_input/layer_k/out_of_grid_fraction`
  - `class/worst_class_acc`
  - `class/confusion_matrix`
  - `update/coeff_delta_over_coeff_norm`
  - `update/function_delta_over_function_norm`
- 增加 functional moment update：
  - `sobolev_momentum`
  - `sobolev_adam_moment`
  - `diag_to_sobolev_adam_moment`
- 增加 `--classification-diagnostics`
- 增加 `--rbf-gamma-scale`

新增 `experiments/run_stage_b_mnist_drilldown.py`：

```text
M0: MNIST failure reproduction with diagnostics
M1: MNIST / Fashion-MNIST branch activity and ablation audit
M2: MNIST coeff_lr / rest_lr sweep
M3: Sobolev + momentum / Adam moment 初筛
M2B: coeff_lr=1.0, rest_lr=0.003 的 5-seed 复核
M2C: coeff_lr=0.3, rest_lr=0.003 的 Pareto 5-seed 复核
```

### 执行命令

主实验：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca \
  python experiments/run_stage_b_mnist_drilldown.py \
  --out-dir results/stage_b_next_mnist_drilldown_gpu0 \
  --phases M0,M1,M2,M3 \
  --seeds 0,1,2 \
  --m2-seeds 0 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

5-seed 高步长复核：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca \
  python experiments/run_stage_b_mnist_drilldown.py \
  --out-dir results/stage_b_next_mnist_drilldown_confirm_gpu0 \
  --phases M2B \
  --seeds 0,1,2,3,4 \
  --m2-seeds 0 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

5-seed Pareto 复核：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca \
  python experiments/run_stage_b_mnist_drilldown.py \
  --out-dir results/stage_b_next_mnist_drilldown_pareto_gpu0 \
  --phases M2C \
  --seeds 0,1,2,3,4 \
  --m2-seeds 0 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

完成情况：

```text
main M0-M3 runs = 99, failed = 0
M2B confirm runs = 20, failed = 0
M2C pareto runs = 15, failed = 0
total non-smoke runs = 134
```

### M0：MNIST failure reproduction with full diagnostics

配置：

```text
dataset = MNIST small
train / val / test = 6000 / 1000 / 1000
hidden_dim = 32
depth = 2
basis_count = 8
epochs = 5
seeds = 0,1,2
```

结果：

| method | train acc | test acc | val AUC | branch ratio | branch / AdamW | no-KAN acc drop | active basis | phi p95 | max J cond | train gap vs AdamW | test gap vs AdamW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9689 | 0.9133 | 0.3548 | 0.2795 | 1.000 | 0.0203 | 0.8125 | 0.0440 | 1.7887 | 0.0000 | 0.0000 |
| all_functional_other_sgd | 0.7861 | 0.7827 | 1.3018 | 0.0325 | 0.116 | 0.0017 | 0.7500 | 0.0134 | 1.1108 | 0.1828 | 0.1307 |
| hybrid_diag | 0.9284 | 0.8987 | 0.5466 | 0.0481 | 0.172 | 0.0037 | 0.7500 | 0.0137 | 1.1219 | 0.0405 | 0.0147 |
| hybrid_sobolev | 0.9287 | 0.8993 | 0.5475 | 0.0449 | 0.161 | 0.0033 | 0.7500 | 0.0139 | 1.1190 | 0.0402 | 0.0140 |
| hybrid_diag_to_sobolev | 0.9287 | 0.8990 | 0.5469 | 0.0463 | 0.166 | 0.0033 | 0.7500 | 0.0139 | 1.1203 | 0.0402 | 0.0143 |

M0 结论：

1. 复现了旧失败：hybrid 的 train acc 比 AdamW 低约 `4.0%`，所以不是泛化失败，而是优化 / 表达不足。
2. branch under-active 证据很强：hybrid branch ratio 只有 AdamW 的 `16%-17%`。
3. no-KAN ablation 证据一致：AdamW 关闭 KAN branch 后 test acc 下降约 `2.03%`，hybrid 只下降约 `0.33%-0.37%`。
4. basis coverage 不是主因：active basis fraction 约 `0.75`，out-of-grid fraction 约 `1e-4 - 8e-4`，没有触发 coverage failure。

### M1：MNIST / Fashion-MNIST branch activity audit

目标：检查 MNIST 现象是否在 Fashion-MNIST 复现。

| dataset | method | train acc | test acc | val AUC | branch ratio | branch / AdamW | no-KAN acc drop | phi p95 | max J cond | train gap | test gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | all_adamw | 0.9689 | 0.9133 | 0.3548 | 0.2795 | 1.000 | 0.0203 | 0.0440 | 1.7887 | 0.0000 | 0.0000 |
| MNIST | hybrid_diag | 0.9284 | 0.8987 | 0.5466 | 0.0481 | 0.172 | 0.0037 | 0.0137 | 1.1219 | 0.0405 | 0.0147 |
| MNIST | hybrid_sobolev | 0.9287 | 0.8993 | 0.5475 | 0.0449 | 0.161 | 0.0033 | 0.0139 | 1.1190 | 0.0402 | 0.0140 |
| Fashion | all_adamw | 0.8807 | 0.8353 | 0.5107 | 0.2497 | 1.000 | 0.0100 | 0.0378 | 1.6518 | 0.0000 | 0.0000 |
| Fashion | hybrid_diag | 0.8564 | 0.8230 | 0.6487 | 0.0509 | 0.204 | 0.0007 | 0.0137 | 1.1158 | 0.0242 | 0.0123 |
| Fashion | hybrid_sobolev | 0.8567 | 0.8227 | 0.6490 | 0.0477 | 0.191 | 0.0003 | 0.0139 | 1.1149 | 0.0239 | 0.0127 |

M1 结论：

Fashion-MNIST 也出现同样模式：hybrid accuracy 接近但低于 AdamW，branch ratio 只有 AdamW 的约 `19%-20%`，关闭 KAN branch 后几乎不掉精度。这说明 branch under-active 不是 MNIST split 的偶然现象，而是当前小模型 + functional update 分类设置的共性。

### M2：effective update strength / learning-rate sweep

目标：判断增大 KAN coefficient learning rate 是否能修复 under-active。

M2 首轮是 seed0 sweep：

```text
hybrid_diag / hybrid_sobolev:
  coeff_lr = 0.03, 0.10, 0.30, 1.00
  rest_lr = 3e-4, 1e-3, 3e-3

hybrid_diag_to_sobolev:
  coeff_lr = 0.10, 0.30, 1.00
  rest_lr = 1e-3, 3e-3
  warmup_frac = 0.05, 0.10, 0.25
```

Top seed0 configs：

| method | coeff lr | rest lr | warmup | train acc | test acc | val AUC | branch / AdamW | phi p95 | max J cond | train gap | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_sobolev | 1.0 | 0.003 | 0.25 | 0.9747 | 0.9220 | 0.3632 | 0.803 | 0.0653 | 2.378 | -0.0070 | -0.0050 |
| hybrid_diag_to_sobolev | 1.0 | 0.003 | 0.05 | 0.9743 | 0.9210 | 0.3646 | 0.802 | 0.0630 | 2.344 | -0.0067 | -0.0040 |
| hybrid_diag | 1.0 | 0.003 | 0.25 | 0.9670 | 0.9190 | 0.3696 | 0.845 | 0.0417 | 1.823 | 0.0007 | -0.0020 |
| hybrid_diag_to_sobolev | 0.3 | 0.003 | 0.25 | 0.9555 | 0.9120 | 0.4001 | 0.377 | 0.0235 | 1.400 | 0.0122 | 0.0050 |

M2 结论：

1. 原 MNIST failure 主要来自 effective update strength 太小。`coeff_lr` 从 `0.03` 增大到 `1.0` 后，train/test acc 能追上甚至超过 AdamW。
2. 但 `coeff_lr=1.0` 的 Sobolev 类方法会牺牲部分几何优势：`phi_prime_p95` 和 Jacobian condition 都高于 AdamW。
3. `coeff_lr=0.3, rest_lr=0.003` 是更好的 Pareto 区域：精度接近 AdamW，同时保留明显更低的 `phi_prime_p95` 和 Jacobian condition。

### M2B：高步长 5-seed 复核

配置：

```text
dataset = MNIST small
seeds = 0,1,2,3,4
coeff_lr = 1.0
rest_lr = 0.003
```

结果：

| method | train acc | test acc | test std | test loss | val AUC | branch / AdamW | no-KAN acc drop | phi p95 | max J cond | train gap | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9677 | 0.9114 | 0.0035 | 0.2911 | 0.3583 | 1.000 | 0.0212 | 0.0440 | 1.7646 | 0.0000 | 0.0000 |
| hybrid_diag | 0.9672 | 0.9102 | 0.0059 | 0.2963 | 0.3553 | 0.888 | 0.0166 | 0.0415 | 1.6879 | 0.0005 | 0.0012 |
| hybrid_sobolev | 0.9762 | 0.9140 | 0.0048 | 0.2834 | 0.3480 | 0.831 | 0.0178 | 0.0648 | 2.1416 | -0.0085 | -0.0026 |
| hybrid_diag_to_sobolev | 0.9757 | 0.9134 | 0.0051 | 0.2848 | 0.3487 | 0.834 | 0.0180 | 0.0621 | 2.0942 | -0.0080 | -0.0020 |

M2B 结论：

高步长确认可以修复分类精度。`hybrid_sobolev` 和 `hybrid_diag_to_sobolev` 平均 test acc 略高于 AdamW，但几何不再更好；`hybrid_diag` 与 AdamW 基本持平，同时保留轻微几何优势。

### M2C：Pareto 5-seed 复核

配置：

```text
dataset = MNIST small
seeds = 0,1,2,3,4
coeff_lr = 0.3
rest_lr = 0.003
warmup_frac = 0.25 for diag_to_sobolev
```

结果：

| method | train acc | test acc | test std | test loss | val AUC | branch / AdamW | no-KAN acc drop | phi p95 | max J cond | train gap | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9677 | 0.9114 | 0.0035 | 0.2911 | 0.3583 | 1.000 | 0.0212 | 0.0440 | 1.7646 | 0.0000 | 0.0000 |
| hybrid_sobolev | 0.9597 | 0.9048 | 0.0049 | 0.3105 | 0.3815 | 0.417 | 0.0086 | 0.0273 | 1.3643 | 0.0079 | 0.0066 |
| hybrid_diag_to_sobolev | 0.9590 | 0.9032 | 0.0062 | 0.3135 | 0.3827 | 0.430 | 0.0084 | 0.0236 | 1.3243 | 0.0087 | 0.0082 |

M2C 结论：

`coeff_lr=0.3, rest_lr=0.003` 是当前最重要的 Pareto 结果：

```text
test acc gap vs AdamW < 0.01
train acc gap vs AdamW < 0.01
Jacobian condition < 0.8 * AdamW
phi_prime_p95 < 0.8 * AdamW
```

这满足补充计划的“方法改进成功”标准。虽然绝对 test acc 没超过 AdamW，但精度 gap 小于 1%，且几何明显更稳。

### M3：Adam-style moment 初筛

配置：

```text
dataset = MNIST small
coeff_lr = 0.3
rest_lr = 1e-3
seeds = 0,1,2
```

结果：

| method | train acc | test acc | val AUC | branch / AdamW | no-KAN acc drop | phi p95 | max J cond | bad steps | train gap | test gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9689 | 0.9133 | 0.3548 | 1.000 | 0.0203 | 0.0440 | 1.7887 | 0.0 | 0.0000 | 0.0000 |
| hybrid_sobolev | 0.9398 | 0.9087 | 0.4887 | 0.488 | 0.0107 | 0.0335 | 1.4068 | 0.0 | 0.0292 | 0.0047 |
| hybrid_sobolev_momentum | 0.9388 | 0.9073 | 0.4979 | 0.478 | 0.0097 | 0.0323 | 1.3915 | 0.0 | 0.0301 | 0.0060 |
| hybrid_sobolev_adam_moment | 0.9854 | 0.8727 | 0.8517 | 45.60 | 0.5700 | 9.9306 | 1197.31 | 4.0 | -0.0165 | 0.0407 |
| hybrid_diag_to_sobolev_adam_moment | 0.9813 | 0.8933 | 0.9458 | 57.92 | 0.7400 | 6.7496 | 1927.06 | 5.3 | -0.0123 | 0.0200 |

M3 结论：

1. 简单 momentum 没有解决问题，和无 momentum 的 Sobolev 基本一致。
2. 当前 Adam-moment + Sobolev 实现过强，导致 branch 爆炸：branch ratio 是 AdamW 的 `45x-58x`，Jacobian condition 到 `1e3` 量级，test acc 反而下降。
3. Adam-style moment 方向不能直接作为默认，需要先加 trust-region / clipping / lr 降低 / moment 后再归一化，否则破坏几何稳定性。

### 失败诊断表

输出：`results/stage_b_next_mnist_drilldown_gpu0/failure_table.csv`

触发统计：

```text
no_kan_ablation_small_drop = 73
kan_branch_underactive = 63
underfit_train_acc_gap = 46
classification_hybrid_not_competitive = 8
```

没有触发 basis coverage failure。当前证据链更支持：

```text
primary cause = functional update effective step too small -> KAN branch under-active
not primary = RBF basis coverage mismatch
not solved by naive Adam moment
```

### MNIST 补充实验结论

1. MNIST small 旧失败不是泛化问题，而是 KAN branch under-active / effective update strength 太小。
2. Fashion-MNIST 复现了同类 branch under-active，因此不是 MNIST 特例。
3. 增大 coefficient learning rate 可以修复精度；`coeff_lr=1.0` 可追平或略超 AdamW，但会削弱几何优势。
4. 最好的当前默认分类候选不是 naive Adam moment，而是：

```text
KAN coeff: Sobolev 或 diag_to_sobolev
coeff_lr: 0.3
non-KAN params: AdamW
rest_lr: 0.003
warmup_frac: 0.25 for diag_to_sobolev
```

5. 这个 Pareto 配置满足补充计划成功标准：test/train acc gap 都小于 `1%`，同时 Jacobian condition 和 `phi_prime_p95` 都低于 `0.8 x AdamW`。
6. 下一步建议做 M4/M6：扩大 hidden_dim/basis_count/epochs，并做小子集 memorization test，确认 `coeff_lr=0.3` 的 Pareto 配置是否能在更大容量和更长训练下继续保持几何优势。

## 2026-04-29 Stage C：Analytic KAN adjoint 与 local credit router

依据：`docs/DG-KAN_StageC_详细实验计划.md`

目标：验证 Stage B 得到的不同 branch activity regime 是否影响 credit transport 稳定性和 local router 可学习性。

输出目录：

```text
results/stage_c_credit_v2_gpu0/
  c0_regime_audit.csv
  c1_core_adjoint.csv
  c2_full_block_adjoint.csv
  c3_credit_geometry.csv
  c4_router_distill.csv
  c5_one_step.csv
  aggregate_*.csv
  aggregate_summary.json
  runs/*/summary.json
  runs/*/model.pt
```

GPU / 环境确认：

```text
CUDA_VISIBLE_DEVICES=0
torch = 2.6.0+cu124
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
```

执行命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca \
  python experiments/run_stage_c_credit.py \
  --out-dir results/stage_c_credit_v2_gpu0 \
  --datasets mnist,fashion_mnist \
  --seeds 0,1,2,3,4 \
  --epochs 5 \
  --audit-batch-size 256 \
  --router-mlp-steps 120 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

完成情况：

```text
C0 checkpoints = 40
C1 core adjoint rows = 400
C2 full block adjoint rows = 400
C3 geometry rows = 80
C4 router rows = 3200
C5 one-step rows = 880
failed runs = 0
```

### 本轮代码变更

更新 `experiments/stage_b_functional.py`：

- 增加 `--save-checkpoint`
- 增加内部 `return_artifacts`，Stage C 可直接拿到训练后的 model 和 train/val/test split

新增 `experiments/run_stage_c_credit.py`：

- C0：四类 regime checkpoint 训练与复核
- C1：core KAN analytic VJP vs autograd VJP
- C2：residual DG-KAN full block hybrid analytic VJP vs autograd VJP
- C3：credit amplification / noise gain / credit rank audit
- C4：local credit router distillation，同 regime 与 cross regime
- C5：hidden-state one-step credit sanity

备注：第一版 C4 发现 ridge / diagonal router 没有按 tiny credit scale 做中心化和自适应正则，导致过拟合/放大。已修正为 centered adaptive ridge 后重跑 v2，以下结论使用 v2。

### Stage C 网络与 regime

分类网络仍使用 residual DG-KAN：

```text
x
  -> Linear stem
  -> LayerNorm
  -> [h <- h + alpha_k * RBFKAN(LayerNorm(h))] * 2
  -> Linear head
```

配置：

```text
datasets = MNIST small, Fashion-MNIST small
train / val / test = 6000 / 1000 / 1000
hidden_dim = 32
depth = 2
basis_count = 8
alpha_init = 1.0
seeds = 0,1,2,3,4
```

四类 checkpoint：

| regime | update | coeff lr | rest lr | warmup |
|---|---|---:|---:|---:|
| all_adamw | coeff_adamw | 0.003 | - | - |
| hybrid_conservative | diag_to_sobolev | 0.1 | 0.003 | 0.25 |
| hybrid_pareto | diag_to_sobolev | 0.3 | 0.003 | 0.25 |
| hybrid_active | diag_to_sobolev | 1.0 | 0.003 | 0.05 |

### C0：Regime checkpoint audit

目标：复核 Stage B 的 conservative / Pareto / active 三个 branch activity regime 是否稳定复现。

| dataset | regime | test acc | gap vs AdamW | branch / AdamW | phi' / AdamW | J cond / AdamW | active basis | max OOG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | all_adamw | 0.9114 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.80 | 0.00313 |
| MNIST | conservative | 0.8990 | 0.0124 | 0.266 | 0.345 | 0.660 | 0.75 | 0.00081 |
| MNIST | pareto | 0.9032 | 0.0082 | 0.430 | 0.536 | 0.751 | 0.75 | 0.00071 |
| MNIST | active | 0.9134 | -0.0020 | 0.834 | 1.413 | 1.186 | 0.75 | 0.00010 |
| Fashion | all_adamw | 0.8324 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.75 | 0.00156 |
| Fashion | conservative | 0.8284 | 0.0040 | 0.298 | 0.395 | 0.711 | 0.75 | 0.00032 |
| Fashion | pareto | 0.8310 | 0.0014 | 0.478 | 0.604 | 0.811 | 0.75 | 0.00032 |
| Fashion | active | 0.8310 | 0.0014 | 0.922 | 1.633 | 1.376 | 0.75 | 0.00029 |

C0 结论：

1. `hybrid_pareto` 在两个数据集都满足 test gap < `1%`。
2. `hybrid_pareto` 的 `phi_prime_p95` 和 Jacobian condition 都显著低于 AdamW。
3. `hybrid_active` 精度追平或略超 AdamW，但几何优势消失，尤其 Fashion 上 `phi' / AdamW = 1.63`、`J cond / AdamW = 1.38`。
4. active basis 正常，out-of-grid fraction 很低，没有 coverage failure。

### C1：Core KAN analytic adjoint correctness

目标：验证 KAN primitive 解析 VJP：

```text
g_x_i = sum_j g_y_j * phi'_ji(x_i)
```

credit types：

```text
task_bp
random_gaussian
mixed_lambda_0.25 / 0.5 / 0.75
```

全局结果：

```text
core/cos mean = 1.0
core/worst_cos min = 0.9999997616
core/relerr mean = 1.50e-7
core/relerr max = 1.93e-7
core/relerr_p95 max = 2.67e-7
analytic/autograd time ratio mean = 0.581
```

C1 结论：解析 KAN adjoint 通过 strict gate。误差远低于 `1e-5`，且平均比 autograd VJP 更快。

### C2：Full residual DG-KAN block adjoint

目标：验证完整 block：

```text
h_out = h + alpha * KAN(LN(h))
g_h = g_out + LN^T(alpha * Phi'(LN(h))^T g_out)
```

实现方式：KAN VJP 用解析公式，LayerNorm VJP 用 autograd，和 full autograd block VJP 对比。

全局结果：

```text
full/cos mean = 1.0
full/worst_cos min = 0.9999997616
full/relerr mean = 3.80e-8
full/relerr max = 7.51e-8
full/relerr_p95 max = 1.01e-7
credit amplification mean = 1.089
backward branch ratio mean = 0.194
```

C2 结论：full residual block hybrid analytic VJP 也通过 strict gate。解析 KAN adjoint 可以安全进入后续 local credit routing 实验。

### C3：Credit geometry across regimes

目标：比较不同 regime 下 credit amplification、noise gain 和 credit rank。

| dataset | regime | amp mean | amp p95 | noise gain sigma=0.1 | PR | rank90 |
|---|---|---:|---:|---:|---:|---:|
| MNIST | all_adamw | 1.247 | 1.351 | 1.0397 | 6.36 | 7.5 |
| MNIST | conservative | 1.033 | 1.056 | 1.0054 | 6.10 | 6.8 |
| MNIST | pareto | 1.092 | 1.150 | 1.0136 | 6.18 | 6.9 |
| MNIST | active | 1.289 | 1.452 | 1.0406 | 7.08 | 9.0 |
| Fashion | all_adamw | 1.184 | 1.301 | 1.0291 | 6.54 | 7.0 |
| Fashion | conservative | 1.026 | 1.053 | 1.0042 | 6.26 | 6.3 |
| Fashion | pareto | 1.078 | 1.144 | 1.0108 | 6.43 | 6.8 |
| Fashion | active | 1.294 | 1.491 | 1.0381 | 7.77 | 10.0 |

C3 结论：

1. conservative / Pareto 明显降低 credit amplification 和 noise gain。
2. active 的 credit amplification 和 credit rank 都升高，说明高 branch activity 会让 credit transport 更复杂。
3. Stage B 的几何收益不是“只好看”，它确实让 backward credit transport 更接近 identity、更稳定。

### C4：Learned local router distillation

目标：用 local router 从 `g_out` 逼近 teacher `g_h`。比较同 regime 和 cross regime。

同 regime 关键结果：

| dataset | regime | identity cos / relerr | static ridge cos / relerr | diagonal cos / relerr |
|---|---|---:|---:|---:|
| MNIST | all_adamw | 0.974 / 0.285 | 0.829 / 0.874 | 0.768 / 1.412 |
| MNIST | conservative | 0.998 / 0.072 | 0.987 / 0.136 | 0.986 / 0.135 |
| MNIST | pareto | 0.991 / 0.153 | 0.944 / 0.313 | 0.930 / 0.357 |
| MNIST | active | 0.953 / 0.352 | 0.762 / 1.252 | 0.708 / 1.827 |
| Fashion | all_adamw | 0.979 / 0.246 | 0.905 / 0.502 | 0.863 / 0.691 |
| Fashion | conservative | 0.998 / 0.071 | 0.991 / 0.116 | 0.992 / 0.108 |
| Fashion | pareto | 0.990 / 0.153 | 0.961 / 0.254 | 0.952 / 0.269 |
| Fashion | active | 0.945 / 0.373 | 0.797 / 0.989 | 0.754 / 1.268 |

C4 结论：

1. residual DG-KAN 的 local VJP 本身非常接近 identity，尤其 conservative / Pareto。
2. Conservative regime 的 router 最容易学：identity、static ridge、diagonal 都超过 weak/medium gate。
3. Pareto regime 仍然较容易：identity 很强，static ridge / diagonal 可用但弱于 conservative。
4. Active / AdamW 下 learned linear router 难度明显上升，说明 branch activity 和 credit amplification 会降低 router 可学习性。
5. 直接 MLP / identity-plus-MLP 在本轮小样本设置下表现差，主要是 credit scale 很小且训练样本少；下一轮如果继续 learned router，应使用 residual/normalized target、更多 credit samples 和更强正则，而不是直接 MLP。

### C5：One-step local credit sanity

目标：用不同 credit source 对 block input hidden state 做 one-step descent sanity，检查 actual loss delta。

关键结果：

| dataset | regime | oracle delta | identity delta | static ridge delta | random delta |
|---|---|---:|---:|---:|---:|
| MNIST | all_adamw | -0.0581 | -0.0570 | -0.0574 | -0.00095 |
| MNIST | conservative | -0.0441 | -0.0440 | -0.0440 | -0.00072 |
| MNIST | pareto | -0.0478 | -0.0475 | -0.0476 | -0.00072 |
| MNIST | active | -0.0610 | -0.0587 | -0.0594 | -0.00054 |
| Fashion | all_adamw | -0.0449 | -0.0438 | -0.0444 | 0.00001 |
| Fashion | conservative | -0.0385 | -0.0384 | -0.0385 | 0.00008 |
| Fashion | pareto | -0.0410 | -0.0406 | -0.0407 | 0.00002 |
| Fashion | active | -0.0534 | -0.0503 | -0.0515 | -0.00053 |

所有非随机有效 credit source 的 teacher sign agreement 都是 `1.0`。随机 credit 的实际下降接近 0，是合理负对照。

### Stage C 总结论

1. C1 / C2 解析 adjoint 都通过 strict correctness gate，可以作为 DG-LCA local credit 的可靠 teacher。
2. Stage B 的 Pareto functional update 确实改善了 credit geometry：更低 amplification、更低 noise gain、更低 Jacobian condition，同时 test gap < `1%`。
3. Router 可学习性和 branch regime 强相关：conservative 最容易，Pareto 次之，AdamW / active 更难。
4. 当前 residual DG-KAN 的 VJP 很接近 identity，因此简单 identity router 已经是很强 baseline。后续任何 learned router 都必须明确打败 identity，而不是只打败 zero/random。
5. Stage D 可以继续推进，但建议优先用 `identity + small learned residual` 或 strongly regularized diagonal/ridge router，而不是 naive MLP router。

## 2026-04-29 Stage C-next：非 Identity correction router 实验

依据：`docs/DG-KAN_StageC-next_非Identity修正Router实验计划.md`

核心目标：上一轮 Stage C 发现 residual DG-KAN 的 full-block VJP 很接近 identity，本轮改为学习非 identity 修正项：

```text
g_h = g_out + Delta g
router target = Delta g = g_h_teacher - g_out
```

本轮不再只看 full-VJP cosine，而是同时看：

```text
correction norm ratio
identity relerr / identity cos
router full relerr
gain over identity
correction cosine / correction relerr
one-step actual descent gain over identity
```

### 环境 / 依赖

按要求补装并验证 `torchvision`：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python -m pip install \
  torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

记录到 `requirement.txt`：

```text
torchvision==0.21.0+cu124
```

GPU 确认：

```text
CUDA_VISIBLE_DEVICES=0
torch = 2.6.0+cu124
torchvision = 0.21.0+cu124
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
```

### 代码变更

新增 / 更新：

- `experiments/run_stage_c_next_correction.py`
  - CN0 checkpoint audit
  - CN1 correction geometry audit
  - CN2 residual correction router distillation
  - CN3 identity stress 聚合
  - CN4 one-step correction usefulness
- `experiments/run_stage_c_next_n5_lite.py`
  - N5-lite：full/BP analytic adjoint vs identity adjoint joint-training precheck
- `experiments/stage_a_dgkan.py`
  - 增加 `kmnist` / `emnist_balanced` dataset support
  - 增加 `KANBlock.adjoint_mode = full | identity`
- `experiments/stage_b_functional.py`
  - 增加 `--adjoint-mode`
  - 记录 `credit/adjoint_mode`

编译检查：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python -m py_compile \
  experiments/stage_a_dgkan.py \
  experiments/stage_b_functional.py \
  experiments/run_stage_c_next_correction.py \
  experiments/run_stage_c_next_n5_lite.py
```

### 执行命令

N1-N4：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n lca \
  python experiments/run_stage_c_next_correction.py \
  --out-dir results/stage_c_next_correction_gpu0 \
  --phases N1,N2,N3,N4 \
  --seeds 0,1,2,3,4 \
  --stress-seeds 0,1,2 \
  --epochs 5 \
  --audit-batch-size 256 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

N5-lite：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_stage_c_next_n5_lite.py \
  --out-dir results/stage_c_next_n5_lite_gpu0 \
  --datasets mnist,fashion_mnist \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

输出目录：

```text
results/stage_c_next_correction_gpu0/
  cn0_checkpoint_audit.csv
  cn1_correction_geometry.csv
  cn2_correction_router.csv
  cn4_one_step_correction.csv
  aggregate_cn0_by_setting.csv
  aggregate_cn1_by_setting.csv
  aggregate_cn2_same_by_router.csv
  aggregate_cn2_cross_by_router.csv
  aggregate_cn3_stress.csv
  aggregate_cn4_by_source.csv
  failure_table.csv
  aggregate_summary.json

results/stage_c_next_n5_lite_gpu0/
  cn5_lite_runs.csv
  cn5_lite_summary.csv
  aggregate_summary.json
```

完成情况：

```text
N1-N4 checkpoint audit runs = 108, failed = 0
CN1 rows = 1680
CN2 rows = 8208
CN4 rows = 3360
N5-lite joint-training runs = 40, failed = 0
```

### N1：Correction geometry audit

目标：确认 `Delta g` 在 AdamW / conservative / Pareto / active regime 下的大小、稳定性和几何复杂度。

Small model：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
depth = 2
hidden = 32
basis = 8
seeds = 0..4
```

| dataset | regime | test acc | test gap vs AdamW | branch / AdamW | phi' / AdamW | J cond / AdamW | identity relerr | correction ratio | amp p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | AdamW | 0.8324 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.2469 | 0.2945 | 1.301 |
| Fashion | conservative | 0.8284 | 0.0040 | 0.298 | 0.395 | 0.711 | 0.0710 | 0.0730 | 1.053 |
| Fashion | Pareto | 0.8310 | 0.0014 | 0.478 | 0.604 | 0.811 | 0.1534 | 0.1661 | 1.144 |
| Fashion | active | 0.8310 | 0.0014 | 0.922 | 1.633 | 1.376 | 0.3747 | 0.4888 | 1.491 |
| KMNIST | AdamW | 0.8176 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.3162 | 0.4013 | 1.376 |
| KMNIST | conservative | 0.7802 | 0.0374 | 0.293 | 0.328 | 0.610 | 0.0799 | 0.0827 | 1.058 |
| KMNIST | Pareto | 0.7896 | 0.0280 | 0.469 | 0.565 | 0.736 | 0.1831 | 0.2025 | 1.163 |
| KMNIST | active | 0.8180 | -0.0004 | 0.935 | 1.631 | 1.459 | 0.4258 | 0.5832 | 1.535 |
| MNIST | AdamW | 0.9114 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.2862 | 0.3590 | 1.351 |
| MNIST | conservative | 0.8990 | 0.0124 | 0.267 | 0.345 | 0.660 | 0.0724 | 0.0749 | 1.056 |
| MNIST | Pareto | 0.9032 | 0.0082 | 0.430 | 0.536 | 0.751 | 0.1537 | 0.1684 | 1.150 |
| MNIST | active | 0.9134 | -0.0020 | 0.834 | 1.413 | 1.186 | 0.3510 | 0.4557 | 1.452 |

N1 结论：

1. `conservative` 的 correction ratio 只有 `0.07-0.08`，identity 太强，不适合证明 learned router 价值。
2. `Pareto` 的 correction ratio 在 `0.16-0.20`，amplification p95 约 `1.14-1.16`，是最合适的 router 测试区。
3. `active` 的 correction ratio 明显放大到 `0.46-0.58`，但 `phi'` 和 Jacobian condition 已经高于 AdamW，属于更难但几何更差的 stress 区。
4. KMNIST 上 Pareto 精度 gap 已经到 `2.8%`，说明当前 Pareto 超参不能直接泛化到更复杂数据。

### N2：Residual correction router

目标：让 router 学习 `Delta g`，检验是否能真正打败 identity。

Router：

```text
zero_correction
scalar_correction
diagonal_correction
ridge_correction
diag_lowrank_correction_{4,8,16,32}
analytic_correction_oracle
```

计划成功标准：

```text
gain over identity > 20%
correction cosine > 0.8
```

同 regime 关键结果：

| dataset | regime | router | full relerr | gain rel vs identity | beat rate | correction cos | correction relerr | correction norm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Fashion | Pareto | zero | 0.1539 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| Fashion | Pareto | scalar | 0.1421 | 0.070 | 0.765 | 0.396 | 0.930 | 0.487 |
| Fashion | Pareto | ridge / lowrank32 | 0.2251 | -0.588 | 0.533 | 0.439 | 1.588 | 1.509 |
| Fashion | active | zero | 0.3761 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| Fashion | active | scalar | 0.3388 | 0.095 | 0.786 | 0.434 | 0.905 | 0.510 |
| Fashion | active | ridge / lowrank32 | 0.9069 | -1.621 | 0.475 | 0.409 | 2.621 | 2.598 |
| MNIST | Pareto | zero | 0.1539 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| MNIST | Pareto | scalar | 0.1356 | 0.114 | 0.878 | 0.478 | 0.886 | 0.552 |
| MNIST | Pareto | ridge / lowrank32 | 0.2636 | -0.818 | 0.414 | 0.385 | 1.818 | 1.753 |
| MNIST | active | zero | 0.3526 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| MNIST | active | scalar | 0.3076 | 0.124 | 0.873 | 0.485 | 0.876 | 0.564 |
| MNIST | active | ridge / lowrank32 | 1.0776 | -2.264 | 0.330 | 0.332 | 3.264 | 3.222 |

N2 结论：

1. 只有 `scalar_correction` 稳定正增益，但增益只有 `7%-12%`，没达到 `>20%` gate。
2. `scalar_correction` 的 correction cosine 只有 `0.40-0.49`，说明主要是在做尺度校准，不是真正学会了 correction direction。
3. `ridge` / `diag_lowrank` 的 direction 有一点信号，但 norm 明显过大，full relerr 反而比 identity 更差。
4. 当前 learned correction router 不应进入 Stage D 主线。

### N3：Depth stress

目标：增加 depth 后，identity baseline 是否变弱。

配置：

```text
datasets = MNIST, Fashion-MNIST
depth = 2, 4, 8
hidden = 64
basis = 16
regime = Pareto
seeds = 0,1,2
```

Pareto 结果：

| dataset | depth | test acc | test gap vs AdamW | branch / AdamW | phi' / AdamW | J cond / AdamW | identity relerr | correction ratio | amp p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | 2 | 0.8317 | 0.0073 | 0.380 | 0.532 | 0.578 | 0.2237 | 0.2437 | 1.173 |
| Fashion | 4 | 0.8347 | 0.0040 | 0.458 | 0.611 | 0.698 | 0.2024 | 0.2230 | 1.174 |
| Fashion | 8 | 0.8410 | 0.0023 | 0.586 | 0.697 | 0.858 | 0.1683 | 0.1844 | 1.154 |
| MNIST | 2 | 0.9063 | 0.0177 | 0.329 | 0.485 | 0.493 | 0.1683 | 0.1787 | 1.104 |
| MNIST | 4 | 0.9120 | 0.0133 | 0.399 | 0.561 | 0.601 | 0.1612 | 0.1728 | 1.114 |
| MNIST | 8 | 0.9187 | 0.0123 | 0.488 | 0.659 | 0.774 | 0.1433 | 0.1535 | 1.111 |

N3 结论：

1. depth 增加没有让 Pareto identity relerr 明显升高，反而在 MNIST / Fashion 上略下降。
2. Pareto 几何仍然稳定：amplification p95 约 `1.10-1.17`，Jacobian condition 仍低于 AdamW。
3. depth stress 没能为 learned router 创造足够强的非 identity 空间。

### N4：Dataset stress

目标：更复杂数据是否让 non-identity correction 更必要。

配置：

```text
datasets = Fashion-MNIST, KMNIST, EMNIST Balanced small
depth = 4
hidden = 64
basis = 16
regime = Pareto
seeds = 0,1,2
```

关键结果：

| dataset | regime | test acc | test gap vs AdamW | branch / AdamW | phi' / AdamW | J cond / AdamW | identity relerr | correction ratio | amp p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMNIST Balanced | AdamW | 0.6780 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.2679 | 0.3197 | 1.267 |
| EMNIST Balanced | Pareto | 0.6417 | 0.0363 | 0.448 | 0.522 | 0.657 | 0.1353 | 0.1422 | 1.082 |
| KMNIST | AdamW | 0.8663 | 0.0000 | 1.000 | 1.000 | 1.000 | 0.3819 | 0.5148 | 1.468 |
| KMNIST | Pareto | 0.8147 | 0.0517 | 0.421 | 0.533 | 0.584 | 0.2111 | 0.2330 | 1.158 |

N4 结论：

1. 更复杂数据让 AdamW 的 correction ratio 更大，但 Pareto functional update 把 correction 又压回较稳定区。
2. 当前 Pareto 超参在 EMNIST / KMNIST 上 accuracy gap 达 `3.6%-5.2%`，没有达到计划里的 `<2%` 成功标准。
3. Dataset stress 不支持“直接上 learned router”，更支持先调分类 Pareto 超参或容量。

### CN4：One-step correction usefulness

目标：检查 correction credit 做 hidden-state one-step update 是否真的比 identity 带来更好的 actual descent。

全局平均：

| credit source | n | actual loss delta | gain rel vs identity | teacher credit cos | correction cos |
|---|---:|---:|---:|---:|---:|
| analytic oracle | 336 | -0.0595 | 0.0248 | 1.000 | 1.000 |
| ridge correction | 336 | -0.0586 | 0.0087 | 0.841 | 0.399 |
| diag-lowrank32 | 336 | -0.0585 | 0.0075 | 0.840 | 0.377 |
| zero / identity | 336 | -0.0581 | 0.0000 | 0.976 | 0.000 |
| scalar correction | 336 | -0.0581 | 0.0000 | 0.976 | 0.441 |
| diagonal correction | 336 | -0.0575 | -0.0079 | 0.759 | 0.260 |
| random credit | 336 | 0.0000 | -1.001 | -0.001 | -0.281 |

CN4 结论：

1. Oracle 确实有额外 one-step gain，但平均只有 `2.5%`，说明 identity 已经很强。
2. ridge / lowrank32 有微弱实际增益，但只有 `<1%` 平均，不足以进入主线。
3. 随机 credit 基本不下降，负对照正常。

### N5-lite：AnalyticAdj vs IdentityAdj joint-training precheck

目标：做轻量 joint-training precheck，判断 Stage D 应优先 analytic adjoint 还是 identity / correction router。

说明：

```text
bp_analytic_full = 当前 full BP 训练；Stage C 已证明 analytic block adjoint 与 autograd VJP 对齐到数值精度，因此这里作为 AnalyticAdj-KAN-Sobolev 的等价 precheck。
identity_adj = forward residual branch 保留，但 branch input detach，使更早层只收到 identity credit。
CorrectionRouter 没有进入 N5-lite，因为 N2 未通过 router gate。
```

配置：

```text
datasets = MNIST, Fashion-MNIST
depth / hidden / basis = 2/32/8, 4/64/16
update = diag_to_sobolev
coeff_lr = 0.3
rest_lr = 0.003
seeds = 0..4
```

结果：

| dataset | model | method | test acc | acc gap vs full | val loss AUC | AUC ratio vs full | branch ratio | phi p95 | max J cond |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion | d2 h32 b8 | full / analytic | 0.8310 | 0.0000 | 0.5362 | 1.000 | 0.1210 | 0.0230 | 1.319 |
| Fashion | d2 h32 b8 | identity adj | 0.8310 | 0.0000 | 0.5395 | 1.006 | 0.1166 | 0.0217 | 1.000 |
| Fashion | d4 h64 b16 | full / analytic | 0.8350 | 0.0000 | 0.4938 | 1.000 | 0.0749 | 0.0227 | 1.474 |
| Fashion | d4 h64 b16 | identity adj | 0.8316 | 0.0034 | 0.5030 | 1.019 | 0.0692 | 0.0219 | 1.000 |
| MNIST | d2 h32 b8 | full / analytic | 0.9032 | 0.0000 | 0.3827 | 1.000 | 0.1209 | 0.0236 | 1.324 |
| MNIST | d2 h32 b8 | identity adj | 0.9010 | 0.0022 | 0.3864 | 1.010 | 0.1155 | 0.0222 | 1.000 |
| MNIST | d4 h64 b16 | full / analytic | 0.9080 | 0.0000 | 0.3379 | 1.000 | 0.0751 | 0.0219 | 1.395 |
| MNIST | d4 h64 b16 | identity adj | 0.9030 | 0.0050 | 0.3456 | 1.023 | 0.0713 | 0.0214 | 1.000 |

N5-lite 结论：

1. IdentityAdj 训练非常接近 full/BP analytic：accuracy gap `0.0%-0.5%`，loss AUC 只差 `0.6%-2.3%`。
2. 这说明当前 residual DG-KAN 的训练 credit 主体确实由 identity path 解释，非 identity correction 对训练效果有贡献但不大。
3. 由于 N2 correction router 未过 gate，而 N5-lite identity 已经很强，Stage D 不应把 learned router 作为主线。

### Failure table

输出：`results/stage_c_next_correction_gpu0/failure_table.csv`

统计：

```text
router_bad_correction_cos = 2264
router_no_gain = 2112
router_norm_miscalibrated = 1678
one_step_no_gain = 1076
```

主要诊断：

```text
router does not learn the residual correction direction
correction router does not improve identity enough
correction norm is miscalibrated
router credit does not improve hidden one-step descent over identity
```

### Stage C-next 总结论

1. `Pareto` regime 确实给出合适的 correction magnitude：`correction ratio ~= 0.16-0.20`，同时 amplification p95 约 `1.14`，几何稳定。
2. 但当前 residual correction router 没有通过计划 gate。`scalar_correction` 只能带来 `7%-12%` full relerr 改善，且 correction cosine 只有 `0.40-0.49`；ridge / lowrank 类方法 norm 过大，反而输给 identity。
3. depth stress 没有让 identity baseline 变弱；在 Pareto 下 depth 增加后 identity relerr 反而略降。
4. dataset stress 暴露了更重要的问题：KMNIST / EMNIST 上 Pareto accuracy gap 已经超过 `2%`，应先做分类容量 / 超参调优，而不是优先上 router。
5. N5-lite 表明 IdentityAdj 已经非常接近 full/BP analytic，accuracy gap 最大约 `0.5%`。
6. Stage D 主线建议：

```text
主线 = AnalyticAdj-KAN-Sobolev
强 baseline = IdentityAdj-KAN-Sobolev
暂不主推 = Learned CorrectionRouter
```

下一步如果继续 router，优先方向不是更复杂 MLP，而是：

```text
norm calibration / trust region
residual correction clipping
更多 correction samples
只在 active 或 stress setting 中作为 ablation
```

## 2026-04-29 Stage C-next v2：span router 与 norm-calibrated correction 实验

依据：`docs/DG-KAN_StageC-next_v2_详细实验计划.md`

核心目标：上一轮证明单层 residual correction 太小，learned router 很难证明价值。本轮改为：

```text
Package A: hard-dataset Pareto search
Package B: span-VJP geometry audit, span = 1,2,4
Package C: norm-calibrated residual correction router
Package D: one-step usefulness
```

核心问题：

```text
更难数据 + 更大 span 是否让 identity baseline 变弱？
norm-calibrated router 是否能稳定打败 identity？
router 的 fidelity gain 是否能转化成 one-step actual descent gain？
```

### 代码变更

新增：

- `experiments/run_stage_c_next_v2.py`
  - A：Fashion-MNIST / KMNIST Pareto search
  - B：span VJP audit
  - C：norm-calibrated / clipped / direction-norm router
  - D：one-step usefulness

修复 / 补充：

- `run_stage_c_next_v2.py` 增加 `--resume`
- 修复 Package A 聚合时 span rows 没有 `warmup_frac` 的 merge 问题
- 补跑 Fashion depth8 alpha=1.0 AdamW baseline，使 active depth8 的相对指标完整
- 补跑新增 router-worthy setting 的 C/D router 与 one-step 结果

编译检查：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python -m py_compile \
  experiments/run_stage_c_next_v2.py
```

### 执行命令

主实验：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_stage_c_next_v2.py \
  --out-dir results/stage_c_next_v2_gpu0 \
  --packages A,B,C,D \
  --datasets fashion_mnist,kmnist \
  --search-seeds 0,1,2 \
  --seeds 0,1,2,3,4 \
  --epochs 5 \
  --audit-batch-size 256 \
  --router-mlp-steps 220 \
  --device cuda \
  --wandb-mode disabled \
  --continue-on-error
```

A 包完成后因聚合字段问题中断，训练结果已落盘。修复后从 A 结果继续：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_stage_c_next_v2.py \
  --out-dir results/stage_c_next_v2_gpu0 \
  --packages B,C,D \
  --datasets fashion_mnist,kmnist \
  --search-seeds 0,1,2 \
  --seeds 0,1,2,3,4 \
  --epochs 5 \
  --audit-batch-size 256 \
  --router-mlp-steps 220 \
  --device cuda \
  --wandb-mode disabled \
  --resume \
  --continue-on-error
```

补充：

```text
Fashion-MNIST depth8 alpha=1.0 AdamW baseline, seeds 0..4
Fashion-MNIST depth8 alpha=1.0 active 新增 router-worthy setting 的 C/D 补跑
```

GPU：

```text
CUDA_VISIBLE_DEVICES=0
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
```

输出目录：

```text
results/stage_c_next_v2_gpu0/
  cnv2_checkpoint_audit.csv
  cnv2_pareto_search_summary.csv
  cnv2_selected_configs.json
  cnv2_span_geometry.csv
  cnv2_router_distill.csv
  cnv2_one_step_usefulness.csv
  aggregate_summary.json
  aggregate_pareto_search_by_config.csv
  aggregate_span_geometry_by_setting.csv
  aggregate_router_by_type.csv
  aggregate_one_step_by_source.csv
  failure_table.csv
```

完成情况：

```text
checkpoints = 173
span geometry rows = 1372
router-worthy settings = 94
router rows = 1410
one-step rows = 376
failed training runs = 0
```

### Package A：Hard-dataset Pareto search

目标：在 Fashion-MNIST / KMNIST 上重新寻找满足 `test gap < 2%`、`0.4 < branch / AdamW < 0.8`、`J cond / AdamW < 1.1`、`phi' / AdamW < 1.0` 的 Pareto checkpoint。

搜索空间：

```text
dataset = Fashion-MNIST, KMNIST
depth = 4
hidden = 64
basis = 16
coeff_lr = 0.3, 0.5, 0.7, 1.0
rest_lr = 0.001, 0.003
alpha = 1.0, 1.5
seeds = 0,1,2
```

严格 Pareto pass：

| dataset | coeff lr | rest lr | alpha | test acc | test gap | branch / AdamW | phi / AdamW | J cond / AdamW | s1 id relerr | amp p95 | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | 0.3 | 0.001 | 1.5 | 0.8380 | -0.0027 | 0.606 | 0.703 | 0.726 | 0.271 | 1.288 | -0.001 |
| Fashion | 0.3 | 0.003 | 1.5 | 0.8370 | -0.0017 | 0.563 | 0.699 | 0.815 | 0.293 | 1.320 | 0.006 |
| Fashion | 0.5 | 0.001 | 1.0 | 0.8350 | 0.0037 | 0.587 | 0.683 | 0.722 | 0.229 | 1.234 | 0.007 |
| Fashion | 0.5 | 0.003 | 1.0 | 0.8370 | 0.0017 | 0.557 | 0.684 | 0.800 | 0.261 | 1.266 | 0.010 |
| Fashion | 0.7 | 0.003 | 1.0 | 0.8360 | 0.0027 | 0.647 | 0.767 | 0.885 | 0.301 | 1.345 | 0.012 |
| Fashion | 0.7 | 0.001 | 1.0 | 0.8393 | -0.0007 | 0.701 | 0.770 | 0.822 | 0.272 | 1.305 | 0.020 |
| Fashion | 0.5 | 0.001 | 1.5 | 0.8443 | -0.0090 | 0.768 | 0.820 | 0.910 | 0.335 | 1.417 | 0.025 |
| KMNIST | 0.5 | 0.003 | 1.5 | 0.8523 | 0.0170 | 0.643 | 0.751 | 0.847 | 0.363 | 1.399 | 0.026 |
| KMNIST | 0.7 | 0.003 | 1.5 | 0.8547 | 0.0147 | 0.741 | 0.858 | 1.004 | 0.399 | 1.477 | 0.063 |

自动选择：

```text
Fashion Pareto: coeff_lr=0.3, rest_lr=0.001, alpha=1.5
Fashion active: coeff_lr=1.0, rest_lr=0.001, alpha=1.0
KMNIST Pareto: coeff_lr=0.5, rest_lr=0.003, alpha=1.5
KMNIST active: coeff_lr=1.0, rest_lr=0.001, alpha=1.5
```

A 结论：

1. v2 成功解决了上一轮 KMNIST Pareto gap 过大的问题：KMNIST strict Pareto 找到 `test gap = 1.7%` 的配置。
2. Fashion 的 Pareto 区很宽，多组配置同时满足 strict gate。
3. 增大 `alpha` 到 `1.5` 是让 correction 变大的有效手段；但 active 区开始出现 `J cond` 和 `phi'` 升高。

### Package B：Span-VJP geometry audit

目标：测试 span 从 `1` 增大到 `2/4` 后，identity baseline 是否变弱。

配置：

```text
dataset = Fashion-MNIST, KMNIST
depth = 4, 8
span = 1, 2, 4
regime = AdamW, Pareto, active
seeds = 0..4
```

Checkpoint 摘要：

| dataset | depth | regime | alpha | test acc | gap | branch / AdamW | phi / AdamW | J cond / AdamW |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Fashion | 4 | active | 1.0 | 0.8380 | -0.0010 | 0.838 | 0.910 | 0.978 |
| Fashion | 4 | Pareto | 1.5 | 0.8366 | -0.0046 | 0.608 | 0.703 | 0.731 |
| Fashion | 8 | active | 1.0 | 0.8304 | 0.0070 | 0.910 | 0.960 | 1.115 |
| Fashion | 8 | Pareto | 1.5 | 0.8368 | -0.0056 | 0.727 | 0.781 | 0.872 |
| KMNIST | 4 | active | 1.5 | 0.8462 | 0.0200 | 0.862 | 1.044 | 1.249 |
| KMNIST | 4 | Pareto | 1.5 | 0.8450 | 0.0212 | 0.644 | 0.753 | 0.847 |
| KMNIST | 8 | active | 1.5 | 0.8488 | 0.0146 | 0.916 | 1.040 | 1.225 |
| KMNIST | 8 | Pareto | 1.5 | 0.8512 | 0.0122 | 0.725 | 0.803 | 0.976 |

Span geometry 关键结果：

| dataset | depth | regime | span | worthy | id relerr | corr ratio | amp p95 | gap | J cond / AdamW |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion | 4 | active | 1 | 12/20 | 0.320 | 0.320 | 1.395 | -0.001 | 0.978 |
| Fashion | 4 | Pareto | 1 | 0/20 | 0.272 | 0.272 | 1.294 | -0.0046 | 0.731 |
| Fashion | 4 | Pareto | 2 | 0/15 | 0.475 | 0.475 | 1.684 | -0.0046 | 0.731 |
| Fashion | 8 | active | 2 | 8/35 | 0.409 | 0.409 | 1.607 | 0.007 | 1.115 |
| Fashion | 8 | AdamW | 2 | 22/70 | 0.401 | 0.401 | 1.562 | 0.000 | 1.000 |
| Fashion | 8 | Pareto | 2 | 15/35 | 0.369 | 0.369 | 1.498 | -0.0056 | 0.872 |
| KMNIST | 4 | Pareto | 1 | 8/20 | 0.364 | 0.364 | 1.397 | 0.0212 | 0.847 |
| KMNIST | 8 | active | 1 | 4/40 | 0.278 | 0.278 | 1.301 | 0.0146 | 1.225 |
| KMNIST | 8 | active | 2 | 4/35 | 0.478 | 0.478 | 1.693 | 0.0146 | 1.225 |
| KMNIST | 8 | Pareto | 2 | 5/35 | 0.438 | 0.438 | 1.582 | 0.0122 | 0.976 |

Router-worthy 判定：

```text
identity_relerr > 0.25
correction_ratio > 0.30
amp p95 < 1.5
test gap < 2%
```

实际找到：

```text
router-worthy settings = 94
```

B 结论：

1. v2 的 span 设计有效：上一轮几乎没有 router-worthy setting，本轮找到 `94` 个。
2. span 从 `1` 到 `2/4` 会显著放大 identity relerr，但 span 太大也迅速推高 amplification。
3. `span=2` 是最有价值的 stress：identity relerr 到 `0.37-0.48`，但部分 setting 的 amp p95 仍接近可控边界。
4. `span=4` identity 明显变弱，但 amplification p95 通常超过 `2.2`，不是稳定主线。

### Package C：Norm-calibrated correction router

目标：在 router-worthy settings 上训练 / 拟合 correction router，验证是否打败 identity。

Router：

```text
zero
scalar
norm_calibrated_scalar_tau{0.2,0.5,1.0}
clipped_ridge_tau{0.2,0.5,1.0}
diag_lowrank_{16,32}_tau{0.5,1.0}
direction_norm_tau{0.5,1.0}
analytic_oracle
```

全局结果：

| router | n | identity relerr | full relerr | gain vs identity | beat rate | correction cos | norm ratio | clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic oracle | 94 | 0.386 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| direction_norm_tau0.5 | 94 | 0.386 | 0.335 | 0.134 | 0.690 | 0.528 | 0.822 | 0.032 |
| direction_norm_tau1.0 | 94 | 0.386 | 0.335 | 0.133 | 0.685 | 0.527 | 0.825 | 0.000 |
| scalar | 94 | 0.386 | 0.364 | 0.057 | 0.737 | 0.327 | 0.428 | 0.000 |
| norm_calibrated_scalar_tau0.2 | 94 | 0.386 | 0.368 | 0.046 | 0.687 | 0.327 | 0.435 | 1.000 |
| clipped_ridge_tau0.2 | 94 | 0.386 | 0.376 | 0.023 | 0.530 | 0.275 | 0.435 | 0.997 |
| zero | 94 | 0.386 | 0.386 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

最好的局部 setting：

| setting | router | gain | beat | correction cos | norm ratio |
|---|---|---:|---:|---:|---:|
| Fashion d8 AdamW span2 | direction_norm_tau0.5 | 0.248 | 0.858 | 0.651 | 0.865 |
| Fashion d8 AdamW span2 | direction_norm_tau1.0 | 0.246 | 0.845 | 0.650 | 0.865 |
| Fashion d4 AdamW span1 | direction_norm_tau1.0 | 0.216 | 0.824 | 0.614 | 0.849 |
| Fashion d4 AdamW span1 | direction_norm_tau0.5 | 0.213 | 0.801 | 0.608 | 0.840 |
| Fashion d4 active span1 | direction_norm_tau1.0 | 0.182 | 0.802 | 0.615 | 0.890 |
| Fashion d8 Pareto span2 | direction_norm_tau1.0 | 0.145 | 0.752 | 0.562 | 0.843 |
| KMNIST d8 active span1 | scalar | 0.165 | 0.981 | 0.557 | 0.582 |
| KMNIST d4 Pareto span1 | scalar | 0.128 | 0.937 | 0.464 | 0.498 |

C 结论：

1. v2 明显优于上一轮：`direction_norm` router 全局 gain 达 `13.3%-13.4%`，beat rate 约 `0.69`，norm ratio 约 `0.82`，不再是 norm 爆炸。
2. 但全局仍未达到计划的中等成功 gate：`gain > 20%` 且 `correction cos > 0.7`。
3. 局部 setting 上有接近成功的结果：Fashion d8 AdamW span2 的 gain 达 `24.8%`，beat `0.858`，但 correction cosine 只有 `0.651`，低于 `0.7/0.8` gate。
4. ridge / lowrank 仍然不好，即使 clipping 后 direction 也弱，说明问题不只是 norm miscalibration。

### Package D：One-step usefulness

目标：检验 fidelity gain 是否转化成 actual descent gain。

全局 one-step：

| credit source | n | actual loss delta | gain vs identity | teacher cos | correction cos |
|---|---:|---:|---:|---:|---:|
| analytic oracle | 94 | -0.0540 | 0.0823 | 1.000 | 1.000 |
| direction_norm_tau0.5 | 28 | -0.0444 | 0.0326 | 0.967 | 0.754 |
| direction_norm_tau1.0 | 28 | -0.0422 | 0.0280 | 0.966 | 0.738 |
| scalar | 34 | -0.0603 | 0.0000 | 0.923 | 0.294 |
| zero / identity | 94 | -0.0503 | 0.0000 | 0.931 | 0.000 |
| random credit | 94 | 0.0000 | -1.001 | -0.0004 | -0.200 |

较好的 one-step setting：

| setting | router | gain vs identity | correction cos |
|---|---|---:|---:|
| Fashion d8 AdamW span2 | direction_norm_tau0.5 | 0.039 | 0.741 |
| Fashion d8 Pareto span2 | direction_norm_tau0.5 | 0.036 | 0.811 |
| Fashion d8 active span2 | direction_norm_tau1.0 | 0.041 | 0.722 |
| KMNIST d8 Pareto span2 | direction_norm_tau0.5 | 0.061 | 0.806 |
| Fashion d4 active span1 | direction_norm_tau0.5 | 0.026 | 0.860 |

D 结论：

1. Router 的 one-step gain 存在，但全局平均只有 `2.8%-3.3%`，未达到计划的 `>5%` gate。
2. 局部 setting 达到或接近 one-step gate，例如 KMNIST d8 Pareto span2 的 `6.1%`。
3. Oracle one-step 平均 gain 是 `8.2%`，说明 correction 本身有实际价值；当前 router 没有充分捕捉 teacher correction。
4. Fidelity gain 不完全等价于 update usefulness，Stage D 不应直接把 router 作为主线训练方法。

### Failure table

输出：`results/stage_c_next_v2_gpu0/failure_table.csv`

统计：

```text
router_bad_correction_cos = 1166
router_no_gain = 1139
router_norm_miscalibrated = 659
one_step_no_gain = 80
pareto_not_found = 20
```

说明：

1. `pareto_not_found=20` 来自 search 中不满足 strict Pareto 的候选配置，不表示最终没有 Pareto；最终 Fashion / KMNIST 都找到了 strict Pareto。
2. router 主要失败仍是 correction direction 不够好，而不是单纯 norm 爆炸。

### Stage C-next v2 总结论

1. v2 的实验设计成功放大了非 identity correction：从上一轮的单层 `0.16-0.20`，提升到 span2 常见 `0.37-0.48`。
2. hard-dataset Pareto search 成功，尤其 KMNIST 从上一轮 `3%-5%` gap 降到 strict Pareto 的 `1.7%`。
3. 找到 `94` 个 router-worthy setting，说明 learned router 的问题不再是“完全没有可训练场景”。
4. `direction_norm` router 是本轮最强 router，证明 direction + norm 分解有价值；它全局 gain 约 `13%`，局部 gain 可到 `25%`。
5. 但是 router 仍未过 Stage D 主线 gate：

```text
global gain < 20%
global correction cosine < 0.7
global one-step gain < 5%
```

6. 当前结论是“router 有探索价值，但还不够进入 Stage D 主线”。

Stage D 建议：

```text
主线继续 = AnalyticAdj / IdentityAdj + functional update + memory accounting
router = Stage D ablation / later work
最值得保留的 router = direction_norm_tau0.5 / tau1.0
最值得继续的 setting = Fashion d8 span2, KMNIST d8 span2
```

## 2026-04-29 Stage C-next v3：Derivative-aware span correction router 实验

依据：`docs/DG-KAN_StageC-next_v3_详细实验计划.md`

核心目标：v2 证明 span2 hard settings 能制造 router-worthy correction，但 direction 学不准。本轮集中测试：

```text
V3-A checkpoint 复核与固定 router dataset
V3-B span correction 一阶 / cross-term 分解
V3-C derivative-aware feature probe
V3-D direction-first correction router
V3-F one-step + micro-training usefulness precheck
```

输出目录：

```text
results/stage_c_next_v3_gpu0/
  cnv3_checkpoint_audit.csv
  cnv3_router_dataset_summary.csv
  cnv3_decomposition_audit.csv
  cnv3_feature_probe.csv
  cnv3_router_distill.csv
  cnv3_one_step_usefulness.csv
  cnv3_micro_training.csv
  aggregate_*.csv
  aggregate_summary.json
  failure_table.csv
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES=0
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
run_failures = 0
```

执行命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_stage_c_next_v3.py \
  --out-dir results/stage_c_next_v3_gpu0 \
  --v2-dir results/stage_c_next_v2_gpu0 \
  --packages A,B,C,D,F \
  --settings F8_A_s2,F8_P_s2,K8_P_s2,K8_A_s2 \
  --checkpoint-seeds 0,1,2,3,4 \
  --primary-seed 0 \
  --cross-seed 1 \
  --router-seeds 0,1,2,3,4 \
  --router-train-samples 8192 \
  --router-eval-samples 2048 \
  --batch-size 128 \
  --router-steps 220 \
  --router-batch-size 512 \
  --micro-steps 50,100 \
  --micro-batch-size 128 \
  --micro-lr 0.001 \
  --device cuda \
  --continue-on-error
```

完成情况：

```text
checkpoint_audit_rows = 20
router_dataset_rows = 57344
router_dataset_splits = 16
decomposition_rows = 68
feature_probe_rows = 32
router_rows = 360
one_step_rows = 96
micro_rows = 40
run_failures = 0
```

代码变更：

- 新增 `experiments/run_stage_c_next_v3.py`
  - 复用 v2 固定 checkpoint 做 V3-A；
  - 构造 span2 router dataset：task / random / mixed / teacher-forced train credit；
  - 计算 `delta0`, `delta1`, `delta_first`, `delta_cross`；
  - 增加 hidden-only / derivative / first-order / layerwise feature probe；
  - 增加 direction-first routers：
    - `scalar`
    - `direction_norm_hidden_only`
    - `direction_norm_derivative`
    - `decomposition_aware`
    - `analytic_oracle`
  - 修正 direction-norm router 的 norm head 初始化，围绕训练集 correction norm 的 log-median；
  - 增加 micro-training precheck：固定 checkpoint，只用 hidden credit 更新 prefix KAN coeffs，optimizer 只包含 KAN coeff / bias。

注意：本轮 micro-training 是 `prefix_hidden_credit_kan_coeff_adamw` 轻量 precheck，不是完整 functional-update micro-training。因此它用于 usefulness 风险判断，不能直接等价于最终 Stage D 训练实现。

### 固定 setting 复核

| setting | dataset | regime | test acc | gap vs AdamW | branch / AdamW | phi' / AdamW | J cond / AdamW |
|---|---|---|---:|---:|---:|---:|---:|
| F8_A_s2 | Fashion | AdamW | 0.8374 | 0.0000 | 1.000 | 1.000 | 1.000 |
| F8_P_s2 | Fashion | Pareto | 0.8368 | -0.0056 | 0.727 | 0.781 | 0.872 |
| K8_P_s2 | KMNIST | Pareto | 0.8512 | 0.0122 | 0.725 | 0.803 | 0.976 |
| K8_A_s2 | KMNIST | active | 0.8488 | 0.0146 | 0.916 | 1.040 | 1.225 |

V3-A router dataset test split：

| setting | identity relerr | correction ratio | amp p95 |
|---|---:|---:|---:|
| F8_A_s2 | 0.313 | 0.313 | 1.506 |
| F8_P_s2 | 0.303 | 0.303 | 1.463 |
| K8_P_s2 | 0.388 | 0.388 | 1.586 |
| K8_A_s2 | 0.426 | 0.426 | 1.685 |

说明：F8_P_s2 是最干净的 stable setting；K8_A_s2 是明确 stress setting；F8_A_s2 / K8_P_s2 在本轮 all-start/all-credit test split 的 `amp p95` 略高于 1.5，属于边界附近。

### V3-B：span correction decomposition

| setting | first cos | first relerr | cross ratio | correction SNR |
|---|---:|---:|---:|---:|
| F8_A_s2 | 0.9982 | 0.0716 | 0.0716 | 14.71 |
| F8_P_s2 | 0.9973 | 0.0805 | 0.0805 | 12.86 |
| K8_P_s2 | 0.9954 | 0.1054 | 0.1054 | 9.84 |
| K8_A_s2 | 0.9952 | 0.1102 | 0.1102 | 9.77 |

结论：span2 correction 几乎完全由一阶 branch correction 解释，cross-term 只有约 `7%-11%`。这说明 v2 失败不是因为高阶项复杂，而是 router 没拿到足够结构化的一阶 derivative sketch。

### V3-C：feature probe

test split linear probe cosine：

| setting | hidden-only | derivative | first-order sketch | layerwise first-order |
|---|---:|---:|---:|---:|
| F8_A_s2 | 0.164 | 0.746 | 0.820 | 0.821 |
| F8_P_s2 | 0.122 | 0.862 | 0.938 | 0.937 |
| K8_P_s2 | 0.046 | 0.764 | 0.851 | 0.864 |
| K8_A_s2 | 0.073 | 0.856 | 0.930 | 0.937 |

结论：hidden-only 基本没有 direction signal；显式 derivative / first-order features 带来巨大提升。V3 的假设“router 需要 derivative-aware features”被支持。

### V3-D：direction-first router

test split 最佳非 oracle router 全部是 `decomposition_aware`：

| setting | gain vs identity | beat rate | correction cos | norm ratio | full relerr |
|---|---:|---:|---:|---:|---:|
| F8_A_s2 | 0.9316 | 1.000 | 0.9982 | 0.988 | 0.0208 |
| F8_P_s2 | 0.9213 | 1.000 | 0.9974 | 0.989 | 0.0232 |
| K8_P_s2 | 0.8792 | 1.000 | 0.9952 | 0.955 | 0.0464 |
| K8_A_s2 | 0.8775 | 1.000 | 0.9951 | 0.955 | 0.0526 |

cross-seed eval 仍然强：

| setting | gain vs identity | correction cos | norm ratio |
|---|---:|---:|---:|
| F8_A_s2 | 0.7909 | 0.9966 | 0.814 |
| F8_P_s2 | 0.7526 | 0.9944 | 0.779 |
| K8_P_s2 | 0.7892 | 0.9942 | 0.823 |
| K8_A_s2 | 0.7904 | 0.9948 | 0.826 |

Gate：

```text
weak pass = yes, 4/4 settings
medium pass = yes, 4/4 settings
strong fidelity pass = yes, 4/4 settings
```

但要注意：这个成功来自 `decomposition_aware` 使用 layerwise first-order correction sketch；它不是 hidden-only learned router，也不是 v2 的 pure direction-norm router。

对照：

- `direction_norm_derivative` 只在 F8_P_s2 / K8_A_s2 有部分正结果：
  - F8_P_s2：gain `0.261`, cos `0.688`
  - K8_A_s2：gain `0.157`, cos `0.740`
- hidden-only direction router 仍然失败。

### V3-F：one-step usefulness

| setting | identity delta | analytic gain | decomposition gain | decomposition correction cos |
|---|---:|---:|---:|---:|
| F8_A_s2 | -0.0366 | 0.177 | 0.169 | 0.998 |
| F8_P_s2 | -0.0309 | 0.177 | 0.171 | 0.997 |
| K8_P_s2 | -0.0525 | 0.186 | 0.175 | 0.995 |
| K8_A_s2 | -0.0407 | 0.243 | 0.227 | 0.995 |

结论：one-step gate 明确通过，`decomposition_aware` 基本贴近 analytic oracle，且显著优于 identity。

### Micro-training precheck

`best_router = learned decomposition_aware`，`first_order_decomp = 直接使用 delta_first`。指标为 final val loss 相对 identity 的 gain。

| setting | steps | best router gain | first-order gain | full analytic gain | best router test acc |
|---|---:|---:|---:|---:|---:|
| F8_A_s2 | 50 | -0.002 | 0.025 | 0.028 | 0.847 |
| F8_A_s2 | 100 | -0.002 | 0.001 | 0.004 | 0.854 |
| F8_P_s2 | 50 | -0.028 | 0.004 | -0.012 | 0.844 |
| F8_P_s2 | 100 | -0.010 | 0.020 | -0.004 | 0.850 |
| K8_P_s2 | 50 | -0.008 | -0.042 | -0.016 | 0.847 |
| K8_P_s2 | 100 | 0.036 | 0.021 | 0.027 | 0.850 |
| K8_A_s2 | 50 | 0.010 | -0.007 | 0.006 | 0.842 |
| K8_A_s2 | 100 | 0.014 | 0.001 | 0.003 | 0.843 |

Micro 结论：

1. fidelity / one-step 的成功没有稳定转化为 micro-training gain。
2. 只有 `K8_P_s2, 100 steps` 的 best router 超过 `3%` micro gain。
3. `F8_A_s2` 的 first-order / analytic 在 50 steps 有小幅收益，但 100 steps 消失。
4. 当前 micro precheck 较 noisy，且不是完整 functional-update micro-training，所以不能用它宣称 RouterAdj 已经满足 Stage D 主线标准。

### Failure table

```text
router_gate_fail = 55
router_bad_direction = 42
router_norm_bad = 35
micro_no_gain = 15
run_failed = 0
```

这些 failure 主要来自 scalar / hidden-only / plain derivative direction router 和 micro-precheck，不影响 `decomposition_aware` 的 fidelity / one-step 成功结论。

### Stage C-next v3 总结论

1. v3 推翻了 v2 中“learned router 暂时无主线资格”的强结论：当 router 使用 layerwise first-order derivative sketch 时，span correction 可以被非常准确地学习。
2. span2 correction 的结构很简单：first-order cos `>0.995`，cross ratio `0.07-0.11`。这说明核心不是高阶复杂性，而是需要显式一阶 derivative 信息。
3. `decomposition_aware` 在 4/4 setting 上通过 weak / medium / strong fidelity gate，并且 cross-seed 仍然稳定。
4. one-step usefulness 也通过：router gain `17%-23%`，接近 analytic oracle。
5. 但是 micro-training precheck 没有稳定通过 `>3%` gate，因此还不能把 RouterAdj-DGKAN 放进 Stage D 主线。

Stage D 建议更新：

```text
主线仍然 = AnalyticAdj / IdentityAdj + functional update + memory accounting
必须加入的 ablation/candidate = DecompositionAwareRouterAdj
不建议继续主推 = hidden-only learned router, plain direction-norm router
下一步若继续 router = 实现真正 functional-update micro-training，并比较 analytic span VJP cost vs first-order decomposition cost
```

## 2026-04-29 Stage C-next v4：functional-update credit-to-update 实验

依据：`docs/DG-KAN_StageC-next_v4_详细实验计划.md`

核心目标：v3 已经证明 `decomposition_aware` router 在 span2 correction fidelity 和 one-step usefulness 上有效。本轮把问题推进到真实 KAN coefficient functional update：

```text
V4-B credit-to-coefficient-gradient / preconditioned-update audit
V4-C true KAN coefficient functional-update micro-training
V4-E cost accounting
V4-F Stage D readiness dry-run
```

本轮最终结论基于完整补充版：

```text
results/stage_c_next_v4_full_gpu0/
  v4_credit_to_update_audit.csv
  v4_functional_micro_training.csv
  v4_cost_accounting.csv
  aggregate_credit_to_update_by_source.csv
  aggregate_micro_by_method.csv
  aggregate_dry_run_by_method.csv
  aggregate_cost_by_method.csv
  aggregate_summary.json
  failure_table.csv
```

早期主跑目录：

```text
results/stage_c_next_v4_gpu0/
```

早期主跑已显示趋势，但缺少 `hidden_router`、`full_bp` micro 对照和 `1000-step` dry-run；因此补齐脚本后以 `stage_c_next_v4_full_gpu0` 作为正式 v4 报告依据。

### 环境 / GPU

```text
CUDA_VISIBLE_DEVICES=0
torch.cuda.is_available() = True
cuda device = NVIDIA RTX A5000
run_failures = 0
```

### 代码变更

新增 / 更新：

- `experiments/run_stage_c_next_v4.py`
  - V4-B：比较 hidden credit、KAN coeff gradient、Sobolev-preconditioned update 与 analytic span teacher 的一致性；
  - V4-C：实现 prefix KAN coefficient functional update，不再使用 v3 的 AdamW precheck；
  - V4-E：记录 credit 构造、functional update step time 和 peak memory；
  - V4-F：500 / 1000 step Stage D readiness dry-run；
  - 补齐 `hidden_router`、`full_bp` micro 对照和 `autograd_span` cost accounting。

注意：本轮实现是 prefix functional-update runner。它先把 span2 credit 运输到 `h_k`，再对 `h_k` 之前的 prefix KAN coefficients 做 diag-to-Sobolev / Sobolev preconditioned update。它比 v3 micro precheck 更接近 Stage D，但还不是完整端到端 Stage D training。

编译检查：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python -m py_compile \
  experiments/run_stage_c_next_v4.py
```

### 执行命令

主跑：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_stage_c_next_v4.py \
  --out-dir results/stage_c_next_v4_gpu0 \
  --v2-dir results/stage_c_next_v2_gpu0 \
  --settings F8_P_s2,K8_P_s2 \
  --seeds 0,1,2,3,4 \
  --packages B,C,E,F \
  --router-train-samples 4096 \
  --router-steps 180 \
  --batch-size 128 \
  --audit-batches 2 \
  --prefix-layers 2,4 \
  --micro-steps 50,100,200 \
  --dry-run-steps 500 \
  --micro-lr 0.001 \
  --device cuda \
  --continue-on-error
```

完整补充版：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_stage_c_next_v4.py \
  --out-dir results/stage_c_next_v4_full_gpu0 \
  --v2-dir results/stage_c_next_v2_gpu0 \
  --settings F8_P_s2,K8_P_s2 \
  --seeds 0,1,2,3,4 \
  --packages B,C,E,F \
  --router-train-samples 4096 \
  --router-steps 180 \
  --batch-size 128 \
  --audit-batches 2 \
  --prefix-layers 2,4 \
  --micro-steps 50,100,200,500 \
  --dry-run-steps 500,1000 \
  --micro-lr 0.001 \
  --device cuda \
  --continue-on-error
```

完成情况：

```text
grad_rows = 240
training_rows = 380
cost_rows = 12
run_failures = 0
failure rows = 160
```

### V4-B：credit-to-update audit

目标：验证 v3 的 hidden-credit fidelity 是否能穿过 KAN coefficient gradient 和 Sobolev preconditioner，仍然保持正确 functional update 方向。

| setting | source | hidden cos | coeff grad cos | precond update cos | precond relerr | update norm ratio |
|---|---|---:|---:|---:|---:|---:|
| F8_P_s2 | identity | 0.9509 | 0.9677 | 0.9664 | 0.3588 | 0.7156 |
| F8_P_s2 | hidden_router | 0.9420 | 0.9425 | 0.9500 | 0.3556 | 0.7850 |
| F8_P_s2 | first_order_decomp | 0.9997 | 0.9997 | 0.9997 | 0.0278 | 0.9853 |
| F8_P_s2 | decomp_router | 0.9997 | 0.9995 | 0.9995 | 0.0334 | 0.9874 |
| K8_P_s2 | identity | 0.9331 | 0.9567 | 0.9571 | 0.4084 | 0.6690 |
| K8_P_s2 | hidden_router | 0.8793 | 0.9521 | 0.9535 | 0.4052 | 0.6830 |
| K8_P_s2 | first_order_decomp | 0.9993 | 0.9994 | 0.9993 | 0.0411 | 0.9792 |
| K8_P_s2 | decomp_router | 0.9991 | 0.9977 | 0.9977 | 0.0826 | 0.9485 |

B 结论：

1. `first_order_decomp` 和 `decomp_router` 都通过 credit-to-update gate。hidden credit 的高 fidelity 没有在 coefficient gradient / preconditioned update 阶段崩掉。
2. `first_order_decomp` 在两个 setting 上都比 learned `decomp_router` 更接近 analytic teacher，尤其 K8 上 precond relerr 是 `0.041` vs `0.083`。
3. `hidden_router` 基本没有超过 identity，确认 v3 的判断：router 必须显式使用一阶 derivative / decomposition 信息。

### V4-C：true functional-update micro-training

目标：检验 credit fidelity 是否能转化为真实 KAN coefficient functional update 的 validation loss / AUC gain。

500-step 关键结果：

| setting | method | val AUC | final val loss | test acc | AUC gain vs identity | final loss gain | bad steps |
|---|---|---:|---:|---:|---:|---:|---:|
| F8_P_s2 | identity | 0.4469 | 0.4435 | 0.8392 | 0.0000 | 0.0000 | 0.0 |
| F8_P_s2 | analytic_span | 0.4457 | 0.4418 | 0.8392 | 0.0028 | 0.0037 | 0.0 |
| F8_P_s2 | first_order_decomp | 0.4457 | 0.4418 | 0.8398 | 0.0028 | 0.0037 | 0.0 |
| F8_P_s2 | decomp_router | 0.4457 | 0.4419 | 0.8392 | 0.0027 | 0.0036 | 0.0 |
| F8_P_s2 | hidden_router | 0.4463 | 0.4428 | 0.8410 | 0.0014 | 0.0015 | 0.0 |
| F8_P_s2 | full_bp | 0.4433 | 0.4388 | 0.8402 | 0.0081 | 0.0104 | 0.0 |
| F8_P_s2 | random | 0.4523 | 0.4522 | 0.8374 | -0.0116 | -0.0193 | 239.2 |
| K8_P_s2 | identity | 0.5455 | 0.5430 | 0.8522 | 0.0000 | 0.0000 | 0.0 |
| K8_P_s2 | analytic_span | 0.5447 | 0.5419 | 0.8524 | 0.0014 | 0.0021 | 0.0 |
| K8_P_s2 | first_order_decomp | 0.5448 | 0.5420 | 0.8526 | 0.0013 | 0.0020 | 0.0 |
| K8_P_s2 | decomp_router | 0.5447 | 0.5420 | 0.8524 | 0.0014 | 0.0020 | 0.0 |
| K8_P_s2 | hidden_router | 0.5452 | 0.5427 | 0.8524 | 0.0004 | 0.0006 | 0.0 |
| K8_P_s2 | full_bp | 0.5428 | 0.5392 | 0.8534 | 0.0050 | 0.0073 | 0.0 |
| K8_P_s2 | random | 0.5495 | 0.5496 | 0.8510 | -0.0075 | -0.0123 | 255.6 |

C 结论：

1. `decomp_router` 非常接近 `analytic_span`，但二者相对 identity 的真实训练收益都很小。
2. 计划 gate 要求 `AUC gain > 5%`、`final val loss gain > 3%`。实际 `decomp_router` 只有约 `0.14%-0.27%` AUC gain，final loss gain 约 `0.20%-0.36%`，明确未通过。
3. `first_order_decomp` 与 `decomp_router` 基本同级，且不需要训练 router。若 Stage D 需要近似 analytic credit，deterministic first-order 比 learned router 更划算。
4. `random` 负对照正常失败，bad steps 约 `239-256 / 500`，说明 functional update pipeline 不是对任意 credit 都有效。

### V4-F：Stage D readiness dry-run

目标：用更长预算检查 `DecompRouter-Func` 是否能超过 `IdentityAdj-Func` 并接近 `AnalyticAdj-Func`。

1000-step 关键结果：

| setting | method | val AUC | final val loss | test acc | AUC gain vs identity | final loss gain | gap vs analytic |
|---|---|---:|---:|---:|---:|---:|---:|
| F8_P_s2 | identity | 0.4437 | 0.4393 | 0.8398 | 0.0000 | 0.0000 | 0.0031 |
| F8_P_s2 | analytic_span | 0.4425 | 0.4379 | 0.8406 | 0.0026 | 0.0031 | 0.0000 |
| F8_P_s2 | first_order_decomp | 0.4425 | 0.4379 | 0.8408 | 0.0026 | 0.0032 | -0.0000 |
| F8_P_s2 | decomp_router | 0.4426 | 0.4379 | 0.8404 | 0.0025 | 0.0031 | 0.0001 |
| F8_P_s2 | full_bp | 0.4406 | 0.4363 | 0.8418 | 0.0070 | 0.0068 | -0.0037 |
| K8_P_s2 | identity | 0.5431 | 0.5397 | 0.8532 | 0.0000 | 0.0000 | 0.0025 |
| K8_P_s2 | analytic_span | 0.5422 | 0.5384 | 0.8538 | 0.0018 | 0.0025 | 0.0000 |
| K8_P_s2 | first_order_decomp | 0.5422 | 0.5384 | 0.8542 | 0.0018 | 0.0025 | -0.0000 |
| K8_P_s2 | decomp_router | 0.5422 | 0.5385 | 0.8538 | 0.0017 | 0.0024 | 0.0001 |
| K8_P_s2 | full_bp | 0.5399 | 0.5358 | 0.8534 | 0.0061 | 0.0074 | -0.0050 |

F 结论：

1. 1000-step 也没有改变结论。`decomp_router` 仍然贴近 analytic，但只比 identity 好 `0.17%-0.25%` AUC，远低于 `>5%` gate。
2. `analytic_span` 自身也只比 identity 好 `0.18%-0.26%` AUC，说明当前 setting 里 identity credit 仍然很强，router 没有足够训练收益空间。
3. `full_bp` 是最强上界，但也只有约 `0.61%-0.70%` AUC gain，说明这个 prefix functional-update setup 的整体收益幅度有限。

### V4-E：cost accounting

| setting | method | total step ms | ratio vs identity | peak memory MB |
|---|---|---:|---:|---:|
| F8_P_s2 | identity | 7.51 | 1.00 | 70.20 |
| F8_P_s2 | analytic_span | 7.91 | 1.05 | 68.68 |
| F8_P_s2 | first_order_decomp | 7.51 | 1.00 | 68.71 |
| F8_P_s2 | decomp_router | 9.06 | 1.21 | 68.71 |
| K8_P_s2 | identity | 7.71 | 1.00 | 70.20 |
| K8_P_s2 | analytic_span | 7.65 | 0.99 | 68.68 |
| K8_P_s2 | first_order_decomp | 7.69 | 1.00 | 68.71 |
| K8_P_s2 | decomp_router | 9.22 | 1.20 | 68.71 |

E 结论：

1. `decomp_router` 的 cost 没有超过 `1.3x identity`，所以不是硬性 cost failure。
2. 但它比 `first_order_decomp` 慢约 `20%`，训练收益又没有超过 first-order，因此没有工程主线优势。
3. `first_order_decomp` 几乎与 identity 同成本，并且 fidelity 接近 analytic，是当前更合理的 approximation candidate。

### Failure table

```text
micro_no_gain = 160
run_failed = 0
hidden_to_coeff_drop = 0
precond_amplifies_error = 0
router_cost_too_high = 0
```

按 phase / method：

```text
V4-C F8_P_s2 decomp_router = 20
V4-C F8_P_s2 first_order_decomp = 20
V4-C F8_P_s2 hidden_router = 20
V4-C K8_P_s2 decomp_router = 20
V4-C K8_P_s2 first_order_decomp = 20
V4-C K8_P_s2 hidden_router = 20
V4-F F8_P_s2 decomp_router = 10
V4-F F8_P_s2 first_order_decomp = 10
V4-F K8_P_s2 decomp_router = 10
V4-F K8_P_s2 first_order_decomp = 10
```

### Stage C-next v4 总结论

1. v4 支持 v3 的正面结论：`decomp_router` 学到的 derivative-aware credit 可以稳定转化为 KAN coefficient gradient 和 Sobolev-preconditioned update，B 包明确通过。
2. v4 同时给出负面训练结论：credit fidelity 和 one-step usefulness 没有转化为足够大的真实 functional-update training gain。C/F 包全部未达到 `>5% AUC gain` gate。
3. 问题不是 router fidelity 在 update 阶段崩掉，也不是 Sobolev preconditioner 把方向破坏了；更像是当前 residual DG-KAN setting 中 identity / analytic 的训练差距本来就太小，router 没有足够收益空间。
4. `hidden_router` 继续失败，`random` 负对照正常失败，说明实验判别力是存在的。
5. `first_order_decomp` 比 learned `decomp_router` 更值得保留：fidelity 更好，cost 更低，不需要 router 训练。

Stage D 建议：

```text
主线 = IdentityAdj / AnalyticAdj + functional update + memory accounting
候选/ablation = FirstOrderDecompAdj
不进入主线 = DecompositionAwareRouterAdj
保留理论结果 = router 可学到 correction，但当前不能证明真实训练收益
下一步重点 = 完整 Stage D 训练、memory accounting、identity vs analytic 的稳定差距
```

## 2026-04-29 DG-KAN 重构计划 Stage D/D0：analytic primitive 与 joint-training 首批实验

依据：`docs/DG-KAN_重构研究计划.md`

本轮定位：重构计划很大，包含 Stage D/E/F/G/H/I。本轮先启动第一阶段 Stage D/D0，验证新的主线：

```text
KAN primitive-adjoint co-design
+ functional-space update
+ analytic local credit
```

本轮没有继续把 learned router 作为主线；也没有做 Stage E sufficient statistics、Stage F/G/H/I。FirstOrderDecompAdj 还没有进入本轮 joint-training，因为它需要单独实现 full joint first-order credit transport；本轮先把 FullBP / Identity / Analytic / AdamW 基线固定下来。

输出目录：

```text
results/refactor_stage_d_d0_gpu4567/
  aggregate_runs.csv
  aggregate_summary.csv
  aggregate_summary.json
  failure_table.csv
  shard_0_runs.csv
  shard_1_runs.csv
  shard_2_runs.csv
  shard_3_runs.csv
  runs/*/summary.json
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES = 4,5,6,7
torch.cuda.is_available() = True
GPU = NVIDIA RTX A5000
run_count = 60
run_failures = 0
```

四卡分工：

```text
GPU 4 -> FullBP-DGKAN
GPU 5 -> IdentityAdj-DGKAN
GPU 6 -> AnalyticAdj-DGKAN
GPU 7 -> DGKAN-AdamW
```

### 代码变更

更新 `experiments/stage_a_dgkan.py`：

- 增加 `AnalyticRBFKANFunction`；
- 对 RBF-KAN primitive 显式实现 analytic backward：

```text
y_j = sum_i phi_ji(x_i)
g_x_i = sum_j g_y_j phi'_ji(x_i)
grad a_jim = sum_n g_j^(n) B_m(x_i^(n))
```

梯度一致性 smoke：

```text
forward diff = 0.0
x grad diff = 1.24e-9
coeff grad diff = 2.98e-8
bias grad diff = 0.0
```

更新 `experiments/stage_b_functional.py`：

- 增加 `--primitive-mode autograd|analytic`；
- `AnalyticAdj-DGKAN` 使用 `primitive_mode=analytic`；
- `FullBP-DGKAN` 使用普通 autograd primitive；
- `IdentityAdj-DGKAN` 使用 `adjoint_mode=identity`。

新增 `experiments/run_refactor_stage_d.py`：

- 支持 shard 并行；
- 支持聚合 `aggregate_runs.csv` / `aggregate_summary.csv` / `failure_table.csv`；
- 记录 Stage D 指标：test acc、val AUC、branch utilization、no-KAN ablation、KAN geometry、Jacobian condition、step time、memory。

### 执行命令

smoke：

```bash
export CUDA_VISIBLE_DEVICES=4
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_d.py \
  --out-dir results/refactor_stage_d_d0_smoke2 \
  --datasets fashion_mnist \
  --methods analytic_func \
  --seeds 0 \
  --epochs 1 \
  --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 16 --depth 2 --basis-count 8 \
  --batch-size 128 --eval-batch-size 256 --audit-batch-size 64 \
  --device cuda \
  --continue-on-error
```

正式四卡实验：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

CUDA_VISIBLE_DEVICES=4 python experiments/run_refactor_stage_d.py \
  --out-dir results/refactor_stage_d_d0_gpu4567 \
  --shard-index 0 --num-shards 4 \
  --datasets mnist,fashion_mnist,kmnist \
  --methods fullbp_func,identity_func,analytic_func,adamw \
  --seeds 0,1,2,3,4 \
  --epochs 8 \
  --device cuda \
  --continue-on-error

CUDA_VISIBLE_DEVICES=5 ... --shard-index 1 ...
CUDA_VISIBLE_DEVICES=6 ... --shard-index 2 ...
CUDA_VISIBLE_DEVICES=7 ... --shard-index 3 ...

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_d.py \
  --out-dir results/refactor_stage_d_d0_gpu4567 \
  --aggregate-only
```

训练配置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2,3,4
model = residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_k init = 1.5
epochs = 8
batch_size = 256
functional update = diag_to_sobolev
non-KAN optimizer = AdamW, rest_lr = 0.003
coeff_lr = 0.3 for MNIST/Fashion, 0.5 for KMNIST
```

### 方法定义

| method | credit / primitive | update |
|---|---|---|
| FullBP-DGKAN | autograd full BP | diag-to-Sobolev |
| IdentityAdj-DGKAN | residual identity credit, branch input detach | diag-to-Sobolev |
| AnalyticAdj-DGKAN | explicit analytic RBF-KAN backward | diag-to-Sobolev |
| DGKAN-AdamW | full BP | coefficient AdamW |

注意：`AnalyticAdj-DGKAN` 的 Jacobian audit 路径由于自定义 backward 触发 PyTorch vmap/einsum batching limitation，原始 run 中 `max_jac_condition` 为 NaN。聚合时用同 seed 的 FullBP-DGKAN audit 值补 analytic Jacobian 条目，因为 analytic primitive 的 forward / gradient 已验证与 FullBP 数值一致。训练指标未补值。

### 关键结果

| dataset | method | test acc | val AUC | gap vs FullBP | branch ratio | no-KAN drop | phi' p95 | max J cond | step ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | FullBP-DGKAN | 0.9184 | 0.3192 | 0.0000 | 0.1224 | 0.0276 | 0.0245 | 1.996 | 12.00 |
| MNIST | AnalyticAdj-DGKAN | 0.9184 | 0.3192 | 0.0000 | 0.1224 | 0.0276 | 0.0245 | 1.996 | 12.16 |
| MNIST | IdentityAdj-DGKAN | 0.9012 | 0.3416 | 0.0172 | 0.1124 | 0.0168 | 0.0234 | 1.000 | 12.20 |
| MNIST | DGKAN-AdamW | 0.9282 | 0.3051 | -0.0098 | 0.2419 | 0.0432 | 0.0390 | 3.096 | 9.06 |
| Fashion | FullBP-DGKAN | 0.8344 | 0.4771 | 0.0000 | 0.1308 | 0.0304 | 0.0271 | 2.709 | 11.43 |
| Fashion | AnalyticAdj-DGKAN | 0.8342 | 0.4771 | 0.0002 | 0.1308 | 0.0302 | 0.0271 | 2.709 | 11.98 |
| Fashion | IdentityAdj-DGKAN | 0.8276 | 0.4843 | 0.0068 | 0.1091 | 0.0208 | 0.0243 | 1.000 | 11.73 |
| Fashion | DGKAN-AdamW | 0.8412 | 0.4857 | -0.0068 | 0.2265 | 0.0468 | 0.0415 | 3.282 | 9.43 |
| KMNIST | FullBP-DGKAN | 0.8512 | 0.5797 | 0.0000 | 0.1682 | 0.1096 | 0.0330 | 3.137 | 11.51 |
| KMNIST | AnalyticAdj-DGKAN | 0.8512 | 0.5797 | 0.0000 | 0.1682 | 0.1096 | 0.0330 | 3.137 | 11.78 |
| KMNIST | IdentityAdj-DGKAN | 0.7842 | 0.7057 | 0.0670 | 0.1383 | 0.0596 | 0.0313 | 1.000 | 11.90 |
| KMNIST | DGKAN-AdamW | 0.8748 | 0.5547 | -0.0236 | 0.2563 | 0.1396 | 0.0446 | 3.835 | 9.27 |

### 结论

1. `AnalyticAdj-DGKAN` 通过 Stage D 的核心 gate：在 MNIST / Fashion / KMNIST 上与 `FullBP-DGKAN` 几乎数值等价。test acc gap 在三个数据集上分别是 `0.00% / 0.02% / 0.00%`，val AUC gap 约 `0`。
2. 这支持重构计划里的 primitive-adjoint co-design：RBF-KAN primitive 的显式 analytic backward 可以替代 autograd backward，至少在首批视觉任务上不损失训练效果。
3. `IdentityAdj-DGKAN` 不是 universally safe。Fashion 平均 gap 约 `0.68%`，勉强在 `<1%` 成功区；MNIST gap `1.72%`，KMNIST gap `6.70%`，说明更难任务仍需要 non-identity / analytic credit。
4. `DGKAN-AdamW` 在本轮 8 epoch D0 上 accuracy 更高：MNIST `+0.98%`、Fashion `+0.68%`、KMNIST `+2.36%` vs FullBP functional update。但它的 branch 更强、`phi' p95` 更高、Jacobian condition 更高，几何稳定性明显更差。
5. Functional-update 主线还需要继续调 learning rate / alpha / schedule。D0 证明 analytic credit 可用，但还没有证明 diag-to-Sobolev 在视觉分类 final accuracy 上优于 AdamW。
6. no-KAN ablation 显示 KAN branch 在所有主方法中都参与了任务；KMNIST 的 no-KAN drop 最大，说明 KAN branch 对难任务更关键。

Failure table：

```text
identity_acc_gap = 12
branch_underactive = 1
run_failed = 0
```

Stage D 下一步：

```text
1. 实现 FirstOrderDecompAdj 的真正 joint-training credit transport；
2. 对 functional update 做 LR / alpha / warmup sweep，目标是追上 AdamW accuracy 同时保持更好 geometry；
3. 增加 Stage E memory accounting：full BP vs checkpointing vs analytic primitive/local sufficient statistics；
4. 保留 DGKAN-AdamW 作为强 accuracy baseline，但主线继续比较 geometry / memory / convergence。
```

## 2026-04-29 DG-KAN 重构计划 Stage D/D1：FirstOrderDecomp joint-training credit transport

依据：`docs/DG-KAN_重构研究计划.md`

本轮目标：接上 D0 的下一步，实现真正 joint-training credit transport，而不是 Stage C 的 prefix micro-training。核心测试对象是：

```text
FirstOrderDecompAdj + functional-space update
```

credit transport 设计：

```text
span = 2 residual KAN blocks
h_{k+2} = F_{k+1}(F_k(h_k))
g_k_first = g_{k+2} + Delta_k(g_{k+2}) + Delta_{k+1}(g_{k+2})
```

其中：

```text
Delta_k(g) = branch-local first-order correction
cross term J_k^T J_{k+1}^T g is intentionally omitted
KAN coefficient update = diag-to-Sobolev functional update
non-KAN update = AdamW
stem / head / LayerNorm / alpha_k all participate in joint training
```

输出目录：

```text
results/refactor_stage_d_d1_firstorder_gpu4567/
  aggregate_runs.csv
  aggregate_summary.csv
  aggregate_summary.json
  failure_table.csv
  shard_0_runs.csv
  shard_1_runs.csv
  shard_2_runs.csv
  shard_3_runs.csv
  runs/*/summary.json
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES = 4,5,6,7
GPU = NVIDIA RTX A5000
run_count = 75
run_failures = 0
```

四卡执行命令：

```bash
rm -rf results/refactor_stage_d_d1_firstorder_gpu4567
rm -f results_refactor_stage_d_d1_gpu{4,5,6,7}.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

CUDA_VISIBLE_DEVICES=4 /mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_d.py \
  --out-dir results/refactor_stage_d_d1_firstorder_gpu4567 \
  --stage-tag D1 \
  --shard-index 0 --num-shards 4 \
  --datasets mnist,fashion_mnist,kmnist \
  --methods fullbp_func,identity_func,analytic_func,first_order_func,adamw \
  --seeds 0,1,2,3,4 \
  --epochs 8 \
  --device cuda \
  --continue-on-error

CUDA_VISIBLE_DEVICES=5 ... --shard-index 1 ...
CUDA_VISIBLE_DEVICES=6 ... --shard-index 2 ...
CUDA_VISIBLE_DEVICES=7 ... --shard-index 3 ...

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_d.py \
  --out-dir results/refactor_stage_d_d1_firstorder_gpu4567 \
  --aggregate-only
```

训练配置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2,3,4
model = residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_k init = 1.5
epochs = 8
batch_size = 256
functional update = diag-to-Sobolev
rest_lr = 0.003
coeff_lr = 0.3 for MNIST/Fashion, 0.5 for KMNIST
```

### 代码变更

更新 `experiments/run_refactor_stage_d.py`：

- 增加 `first_order_func` / `FirstOrderDecomp-DGKAN`；
- 增加 `run_first_order_joint`，在训练循环中显式做 span-2 first-order credit transport；
- KAN coefficients 用 `apply_manual_update` 做 diag-to-Sobolev；
- 非 KAN 参数通过 AdamW 更新；
- audit 时临时切回 autograd primitive，避免 custom analytic backward 与 `vmap/einsum` 的 batching limitation；
- 修复 runner 输出中 stage label 写死为 D0 的问题。

smoke：

```text
dataset = MNIST
method = first_order_func
seed = 0
epochs = 1
train / val / test = 512 / 128 / 128
CUDA_VISIBLE_DEVICES = 4
result = completed 1/1, failed 0
```

### 关键结果

| dataset | method | test acc | val AUC | gap vs Analytic | branch ratio | no-KAN drop | phi' p95 | max J cond | step ms | peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | FullBP-DGKAN | 0.9184 | 0.3192 | 0.0000 | 0.1224 | 0.0276 | 0.0245 | 1.996 | 11.64 | 45.23 |
| MNIST | AnalyticAdj-DGKAN | 0.9184 | 0.3192 | 0.0000 | 0.1224 | 0.0276 | 0.0245 | 1.996 | 11.63 | 45.23 |
| MNIST | FirstOrderDecomp-DGKAN | 0.9170 | 0.3193 | 0.0014 | 0.1212 | 0.0270 | 0.0244 | 1.936 | 11.45 | 86.94 |
| MNIST | IdentityAdj-DGKAN | 0.9012 | 0.3416 | 0.0172 | 0.1124 | 0.0168 | 0.0234 | 1.000 | 11.67 | 45.23 |
| MNIST | DGKAN-AdamW | 0.9282 | 0.3051 | -0.0098 | 0.2419 | 0.0432 | 0.0390 | 3.096 | 8.61 | 47.23 |
| Fashion | FullBP-DGKAN | 0.8344 | 0.4771 | 0.0002 | 0.1308 | 0.0304 | 0.0271 | 2.709 | 11.56 | 45.23 |
| Fashion | AnalyticAdj-DGKAN | 0.8342 | 0.4771 | 0.0000 | 0.1308 | 0.0302 | 0.0271 | 2.709 | 12.07 | 45.23 |
| Fashion | FirstOrderDecomp-DGKAN | 0.8358 | 0.4761 | -0.0016 | 0.1273 | 0.0306 | 0.0266 | 2.503 | 11.23 | 86.94 |
| Fashion | IdentityAdj-DGKAN | 0.8276 | 0.4843 | 0.0066 | 0.1091 | 0.0208 | 0.0243 | 1.000 | 11.14 | 45.23 |
| Fashion | DGKAN-AdamW | 0.8412 | 0.4857 | -0.0070 | 0.2265 | 0.0468 | 0.0415 | 3.282 | 9.10 | 47.23 |
| KMNIST | FullBP-DGKAN | 0.8512 | 0.5797 | 0.0000 | 0.1682 | 0.1096 | 0.0330 | 3.137 | 11.16 | 45.23 |
| KMNIST | AnalyticAdj-DGKAN | 0.8512 | 0.5797 | 0.0000 | 0.1682 | 0.1096 | 0.0330 | 3.137 | 11.38 | 45.23 |
| KMNIST | FirstOrderDecomp-DGKAN | 0.8532 | 0.5808 | -0.0020 | 0.1653 | 0.1096 | 0.0324 | 2.973 | 10.67 | 86.94 |
| KMNIST | IdentityAdj-DGKAN | 0.7842 | 0.7057 | 0.0670 | 0.1383 | 0.0596 | 0.0313 | 1.000 | 11.55 | 45.23 |
| KMNIST | DGKAN-AdamW | 0.8748 | 0.5547 | -0.0236 | 0.2563 | 0.1396 | 0.0446 | 3.835 | 9.39 | 47.23 |

Failure table：

```text
identity_acc_gap = 12
branch_underactive = 1
first_order_acc_gap = 0
run_failed = 0
```

### 结论

1. `FirstOrderDecompAdj` 已经从 Stage C 的 micro precheck 升级成真正 joint training credit transport。
2. FirstOrder 通过 D1 成功标准：相对 AnalyticAdj 的 test acc gap 在 MNIST / Fashion / KMNIST 分别为 `0.14% / -0.16% / -0.20%`，全部小于 `1%`。
3. FirstOrder 与 FullBP / Analytic 的收敛基本一致，在 Fashion 和 KMNIST 上平均 test acc 还略高一点；这说明 span-2 first-order correction 足以保留主要 non-identity credit。
4. IdentityAdj 继续暴露问题：MNIST gap `1.72%`，KMNIST gap `6.70%`，说明不能把 identity credit 作为唯一主线。
5. FirstOrder 的几何指标保持在 functional-update 主线附近，并且略低于 Analytic / FullBP 的 Jacobian condition；AdamW 虽然 accuracy 更高，但 branch ratio、`phi' p95`、Jacobian condition 都更激进。
6. 当前 FirstOrder 实现还不是 memory win：peak memory `86.94 MB` 高于 FullBP / Analytic 的 `45.23 MB`。原因是 D1 用显式 surrogate backward 做完整 joint training，还没有实现 Stage E 的 local sufficient statistics / memory-optimized primitive。

下一步：

```text
1. 继续保留 FirstOrderDecompAdj 作为 Stage D 候选主线；
2. 做 functional update 的 LR / alpha / warmup sweep，目标追近 AdamW accuracy；
3. 启动 Stage E memory accounting，把当前 FirstOrder 从正确性实现推进到低内存实现；
4. 比较 AnalyticAdj vs FirstOrderDecompAdj 的 cost / memory / accuracy trade-off。
```

## 2026-04-29 DG-KAN 重构计划 Stage E/E0：memory accounting 与 sufficient statistics 优化

依据：`docs/DG-KAN_重构研究计划.md` Stage E。

本轮目标：验证 DG-KAN 是否可以不保存完整 KAN activation graph，而通过 analytic local credit 和 edge sufficient statistics 完成训练，并拆分 memory / compute / stats fidelity。

输出目录：

```text
results/refactor_stage_e_gpu4567/
  aggregate_runs.csv
  aggregate_summary.csv
  aggregate_summary.json
  failure_table.csv
  shard_0_runs.csv
  shard_1_runs.csv
  shard_2_runs.csv
  shard_3_runs.csv
  runs/*/summary.json
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES = 4,5,6,7
GPU = NVIDIA RTX A5000
run_count = 45
run_failures = 0
```

### 代码变更

新增 `experiments/run_refactor_stage_e.py`：

- `FullBP-DGKAN`：标准 full activation graph；
- `Checkpointed-DGKAN`：block checkpoint + recompute baseline；
- `AnalyticAdj-DGKAN`：analytic RBF primitive backward；
- `SufficientStats-DGKAN`：exact local credit + RBF edge sufficient statistics；
- `FirstOrderStats-DGKAN`：span-2 first-order approximate credit + edge stats。

`SufficientStats-DGKAN` 的核心实现：

```text
forward:
  no_grad forward, save block interfaces h_k

head backward:
  compute g_T from classifier head only

reverse local block:
  z_k = LN(h_k) locally recomputed
  grad coeff = sum_b g_j^(b) B_m(z_i^(b))
  grad z_i = sum_j g_j phi'_ji(z_i)
  backprop only through LayerNorm
  update KAN coeff immediately with diag-to-Sobolev

stem:
  backprop transported credit through input + input_ln
```

本轮还做了两个实现修正：

1. `stats_fidelity` 的 deepcopy 模型显式释放，避免 measurement 污染；
2. 手动 RBF VJP 从三输入 einsum 改成显式 `gout @ coeff` sketch，减少中间张量和 step time。

### 执行命令

smoke：

```bash
CUDA_VISIBLE_DEVICES=4 /mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_e.py \
  --out-dir results/refactor_stage_e_smoke_gpu4 \
  --stage-tag E_smoke \
  --datasets mnist \
  --methods fullbp,checkpoint,analytic,sufficient_stats,first_order_stats \
  --seeds 0 \
  --epochs 1 \
  --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 32 --depth 2 --basis-count 8 \
  --batch-size 128 --eval-batch-size 256 --audit-batch-size 64 \
  --device cuda \
  --continue-on-error
```

正式四卡实验：

```bash
rm -rf results/refactor_stage_e_gpu4567
rm -f results_refactor_stage_e_gpu{4,5,6,7}.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

CUDA_VISIBLE_DEVICES=4 /mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_e.py \
  --out-dir results/refactor_stage_e_gpu4567 \
  --stage-tag E0 \
  --shard-index 0 --num-shards 4 \
  --datasets mnist,fashion_mnist,kmnist \
  --methods fullbp,checkpoint,analytic,sufficient_stats,first_order_stats \
  --seeds 0,1,2 \
  --epochs 5 \
  --hidden-dim 128 --depth 8 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --device cuda \
  --continue-on-error

CUDA_VISIBLE_DEVICES=5 ... --shard-index 1 ...
CUDA_VISIBLE_DEVICES=6 ... --shard-index 2 ...
CUDA_VISIBLE_DEVICES=7 ... --shard-index 3 ...

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python \
  experiments/run_refactor_stage_e.py \
  --out-dir results/refactor_stage_e_gpu4567 \
  --aggregate-only
```

训练配置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2
model = residual DG-KAN
depth = 8
hidden_dim = 128
basis_count = 16
alpha_k init = 1.5
epochs = 5
batch_size = 256
functional update = diag-to-Sobolev
rest optimizer = AdamW
rest_lr = 0.003
coeff_lr = 0.3 for MNIST/Fashion, 0.5 for KMNIST
```

### Memory accounting

| dataset | method | test acc | val AUC | step ms | peak MB | reduction vs FullBP | grad cos | grad relerr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | FullBP | 0.9207 | 0.3046 | 15.04 | 78.44 | 0.0% | - | - |
| MNIST | Checkpoint | 0.9207 | 0.3046 | 24.31 | 78.44 | 0.0% | - | - |
| MNIST | AnalyticAdj | 0.9207 | 0.3046 | 15.73 | 43.53 | 44.5% | - | - |
| MNIST | SufficientStats | 0.9207 | 0.3046 | 15.92 | 70.44 | 10.2% | 1.0000 | 2.76e-7 |
| MNIST | FirstOrderStats | 0.9203 | 0.3051 | 15.90 | 70.44 | 10.2% | 0.9978 | 6.68e-2 |
| Fashion | FullBP | 0.8400 | 0.4648 | 14.94 | 78.44 | 0.0% | - | - |
| Fashion | Checkpoint | 0.8400 | 0.4648 | 24.33 | 78.44 | 0.0% | - | - |
| Fashion | AnalyticAdj | 0.8400 | 0.4648 | 15.57 | 43.53 | 44.5% | - | - |
| Fashion | SufficientStats | 0.8400 | 0.4648 | 15.44 | 70.44 | 10.2% | 1.0000 | 2.59e-7 |
| Fashion | FirstOrderStats | 0.8403 | 0.4628 | 15.97 | 70.44 | 10.2% | 0.9982 | 5.94e-2 |
| KMNIST | FullBP | 0.8653 | 0.5293 | 13.81 | 78.44 | 0.0% | - | - |
| KMNIST | Checkpoint | 0.8653 | 0.5293 | 24.00 | 78.44 | 0.0% | - | - |
| KMNIST | AnalyticAdj | 0.8653 | 0.5293 | 15.06 | 43.53 | 44.5% | - | - |
| KMNIST | SufficientStats | 0.8653 | 0.5293 | 15.05 | 70.44 | 10.2% | 1.0000 | 2.89e-7 |
| KMNIST | FirstOrderStats | 0.8667 | 0.5295 | 15.55 | 70.44 | 10.2% | 0.9973 | 7.30e-2 |

Memory breakdown for `depth=8, hidden=128, basis=16, batch=256`：

```text
params_mb = 8.40
optimizer_mb = 0.80
FullBP estimated full_activations_mb = 18.0
Checkpoint checkpoints_mb = 1.125
SufficientStats interfaces_mb = 1.125
SufficientStats edge_stats_mb = 8.0
SufficientStats local_recompute_mb = 2.0
```

说明：

1. 当前 block-level checkpoint baseline 没有降低 measured peak：`78.44 MB`，但 step time 增加到约 `1.6x-1.75x`。这说明当前 checkpoint 粒度对 RBF-KAN 的 autograd basis graph 不够有效，后续需要 segment-level / primitive-level checkpoint。
2. `AnalyticAdj-DGKAN` 是本轮最强 memory 方法：peak 从 `78.44 MB` 降到 `43.53 MB`，约 `44.5%` reduction，accuracy 与 FullBP 完全一致。
3. `SufficientStats-DGKAN` 的 edge stats 路径是正确的：coeff gradient cosine `~1.0`，relative error `~2.6e-7 - 2.9e-7`；preconditioned update cosine `~1.0`，relative error `~1e-6`。
4. `SufficientStats-DGKAN` 目前只比 FullBP / checkpoint 降低 `10.2%` peak memory，没有超过 `30%` strong gate，也明显不如 analytic primitive。
5. `FirstOrderStats-DGKAN` 作为近似方法 accuracy 没掉，但 stats gradient relerr 为 `0.059-0.073`，preconditioned update relerr 为 `0.083-0.088`；它可作为 approximation ablation，不能替代 exact stats。

Failure table：

```text
failure_count = 0
run_failed = 0
```

### Stage E/E0 结论

1. Weak / medium success：`SufficientStats-DGKAN` accuracy gap 为 `0`，memory 低于 FullBP 和本轮 checkpoint baseline，且 stats update 与 FullBP 数值等价。
2. Strong success 未通过：memory reduction vs FullBP 只有 `10.2%`，未达到 `>30%`；相对 checkpoint 的 `10.2%` 刚过计划里的 `>10%` 线，但 checkpoint baseline 本身没有成功省显存。
3. 当前最有价值的 low-memory 路线不是 full sufficient stats，而是 `AnalyticAdj-DGKAN`：它在 accuracy 不变的前提下给出 `44.5%` peak memory reduction。
4. Sufficient stats 的主要瓶颈是 edge_stats / local recompute temp / PyTorch tensor allocation，尤其 dense KAN coeff stats 为 `8.0 MB`。如果要让 SufficientStats 成为主低显存 claim，需要继续做：

```text
streaming / chunked edge stats
grouped connectivity stats
half precision stats accumulation
fused RBF stats kernel
primitive-level checkpoint baseline
```

5. Stage F 前建议主线更新为：

```text
主线 low-memory = AnalyticAdj primitive
候选优化 = SufficientStats with chunked/grouped stats
保留 ablation = FirstOrderStats
不应 claim = 当前 dense SufficientStats 已强低显存
```

## 2026-04-30 DG-KAN Stage F：convergence dynamics / optimization efficiency

依据：`docs/DG-KAN_StageF_调整后详细实验计划.md`

本轮目标：系统比较 functional update 的收敛效率、几何稳定性、branch utilization，以及 AnalyticAdj / FirstOrder / Identity credit 的优化差异。

输出目录：

```text
results/stage_f_gpu0/
  stage_f_runs.csv
  stage_f_summary_by_method.csv
  stage_f_summary_by_dataset.csv
  stage_f_target_matched.csv
  f1_search_ranked_configs.csv
  selected_configs.json
  aggregate_summary.json

results/stage_f_confirm_gpu0/
  stage_f_runs.csv
  stage_f_summary_by_method.csv
  stage_f_summary_by_dataset.csv
  stage_f_target_matched.csv
  aggregate_summary.json
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES = 0
GPU = NVIDIA RTX A5000
F0 + F1_search runs = 511, failed = 0
F1_confirm runs = 80, failed = 0
```

### 代码变更

新增 `experiments/run_stage_f.py`：

- F0：AdamW / FullBP functional / AnalyticAdj functional / FirstOrderDecomp functional / IdentityAdj functional；
- F1_search：Fashion-MNIST / KMNIST 上扫 `coeff_lr, rest_lr, alpha, warmup_frac`；
- F1_confirm：对每个数据集 top-3 配置做 `8` 和 `20` epoch、5-seed 复核；
- 聚合 `test_acc`, `val_auc`, `test_acc_gap_vs_adamw`, `val_auc_improvement_vs_adamw`, `phi_prime_reduction_vs_adamw`, `jac_reduction_vs_adamw`, `branch_over_adamw`, target-matched steps/time。

修复 `experiments/stage_b_functional.py`：

```text
Analytic primitive 训练仍使用 explicit analytic backward；
Jacobian audit 临时切回 autograd primitive，避免 PyTorch custom backward + vmap 限制导致 max_jac_condition = NaN。
```

smoke：

```text
analytic Jacobian smoke:
  jac_count = 2
  finite = True
  audit_error = None
```

### 执行命令

F0 + F1_search：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_gpu0 \
  --phases F0,F1_search \
  --seeds 0,1,2,3,4 \
  --search-seeds 0,1 \
  --search-epochs 8 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_gpu0 \
  --aggregate-only
```

F1_confirm：

```bash
cp results/stage_f_gpu0/selected_configs.json results/stage_f_confirm_gpu0/selected_configs.json

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_confirm_gpu0 \
  --phases F1_confirm \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_confirm_gpu0 \
  --aggregate-only
```

训练配置：

```text
train / val / test = 6000 / 1000 / 1000
model = residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_k init = 1.5
batch_size = 256
functional update = diag-to-Sobolev
rest optimizer = AdamW
```

### F0：credit / optimizer baseline

| dataset | method | test acc | val AUC | acc gap vs AdamW | AUC improv | phi' reduction | J reduction | branch / AdamW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | AdamW | 0.9282 | 0.3051 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| MNIST | AnalyticAdj | 0.9184 | 0.3192 | 0.0098 | -0.047 | 0.371 | 0.353 | 0.506 |
| MNIST | FirstOrder | 0.9170 | 0.3193 | 0.0112 | -0.047 | 0.375 | 0.372 | 0.501 |
| MNIST | IdentityAdj | 0.9012 | 0.3416 | 0.0270 | -0.121 | 0.399 | 0.675 | 0.464 |
| Fashion | AdamW | 0.8412 | 0.4857 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| Fashion | AnalyticAdj | 0.8342 | 0.4771 | 0.0070 | 0.017 | 0.347 | 0.168 | 0.578 |
| Fashion | FirstOrder | 0.8358 | 0.4761 | 0.0054 | 0.020 | 0.359 | 0.235 | 0.562 |
| Fashion | IdentityAdj | 0.8276 | 0.4843 | 0.0136 | 0.003 | 0.415 | 0.694 | 0.482 |
| KMNIST | AdamW | 0.8748 | 0.5547 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| KMNIST | AnalyticAdj | 0.8512 | 0.5797 | 0.0236 | -0.045 | 0.259 | 0.178 | 0.656 |
| KMNIST | FirstOrder | 0.8532 | 0.5808 | 0.0216 | -0.047 | 0.273 | 0.221 | 0.645 |
| KMNIST | IdentityAdj | 0.7842 | 0.7057 | 0.0906 | -0.274 | 0.298 | 0.736 | 0.540 |

F0 结论：

1. `AnalyticAdj` 继续与 `FullBP` 数值等价；`FirstOrder` 仍在 Analytic 附近。
2. `IdentityAdj` 不适合做主线 credit，尤其 KMNIST gap 达 `9.06%`。
3. AdamW accuracy 仍最强，但 functional methods 的 `phi'` 和 Jacobian condition 明显更低。
4. Functional update 当前瓶颈不是 credit 正确性，而是 branch/optimization tradeoff：branch 只有 AdamW 的约 `0.50-0.66`。

### F1_search：functional update 粗搜

搜索空间：

```text
dataset = Fashion-MNIST, KMNIST
seeds = 0,1
coeff_lr = 0.3, 0.5, 0.7, 1.0
rest_lr = 0.001, 0.003, 0.006
alpha = 1.0, 1.5, 2.0
warmup_frac = 0.05, 0.10, 0.25
epochs = 8
```

Top configs：

| dataset | config | test acc | acc gap | AUC improv | phi' reduction | J reduction | branch / AdamW |
|---|---|---:|---:|---:|---:|---:|---:|
| Fashion | clr0.3 rlr0.001 a1 tw0.25 | 0.8325 | 0.0145 | -0.003 | 0.400 | 0.509 | 0.390 |
| Fashion | clr0.3 rlr0.001 a1 tw0.10 | 0.8335 | 0.0135 | -0.008 | 0.390 | 0.499 | 0.372 |
| Fashion | clr0.3 rlr0.001 a1 tw0.05 | 0.8355 | 0.0115 | -0.011 | 0.388 | 0.500 | 0.362 |
| KMNIST | clr0.3 rlr0.003 a1.5 tw0.25 | 0.8360 | 0.0365 | -0.089 | 0.359 | 0.300 | 0.553 |
| KMNIST | clr0.3 rlr0.003 a2 tw0.25 | 0.8505 | 0.0220 | -0.025 | 0.359 | 0.135 | 0.669 |
| KMNIST | clr0.3 rlr0.003 a1 tw0.25 | 0.8305 | 0.0420 | -0.188 | 0.380 | 0.429 | 0.421 |

F1_search 结论：

1. Fashion 能找到 geometry 明显优于 AdamW 的配置，但 8-epoch accuracy 仍差 `1.1%-1.5%`，AUC 没有稳定超过 AdamW。
2. KMNIST best search 仍差 `2.2%-3.7%`，没有通过 `acc gap < 2%` 的弱 gate。
3. `coeff_lr=0.3` 在综合排序中胜出；`coeff_lr >= 0.7` 常把 Jacobian condition 拉高，高 branch 不自动等于好优化。

### F1_confirm：top 配置 5-seed 复核

Fashion-MNIST：

| config | epochs | test acc | acc gap | AUC improv | phi' reduction | J reduction | branch / AdamW |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 8 | 0.8412 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| top1 | 8 | 0.8364 | 0.0048 | -0.006 | 0.401 | 0.516 | 0.387 |
| top2 | 8 | 0.8364 | 0.0048 | -0.012 | 0.392 | 0.506 | 0.368 |
| top3 | 8 | 0.8374 | 0.0038 | -0.015 | 0.390 | 0.503 | 0.356 |
| AdamW | 20 | 0.8368 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| top1 | 20 | 0.8296 | 0.0072 | 0.169 | 0.379 | 0.508 | 0.460 |
| top2 | 20 | 0.8240 | 0.0128 | 0.169 | 0.347 | 0.484 | 0.448 |
| top3 | 20 | 0.8268 | 0.0100 | 0.171 | 0.334 | 0.464 | 0.439 |

KMNIST：

| config | epochs | test acc | acc gap | AUC improv | phi' reduction | J reduction | branch / AdamW |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 8 | 0.8748 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| top1 | 8 | 0.8402 | 0.0346 | -0.097 | 0.357 | 0.302 | 0.555 |
| top2 | 8 | 0.8524 | 0.0224 | -0.033 | 0.358 | 0.172 | 0.673 |
| top3 | 8 | 0.8252 | 0.0496 | -0.181 | 0.377 | 0.457 | 0.423 |
| AdamW | 20 | 0.8760 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 |
| top1 | 20 | 0.8526 | 0.0234 | -0.006 | 0.401 | 0.339 | 0.584 |
| top2 | 20 | 0.8614 | 0.0146 | 0.042 | 0.410 | 0.222 | 0.688 |
| top3 | 20 | 0.8402 | 0.0358 | -0.077 | 0.389 | 0.422 | 0.478 |

F1_confirm 结论：

1. Fashion 8-epoch confirm 比 search 更好，top3 的 gap 降到 `0.38%`，并保留约 `39% phi' reduction` 和 `50% J reduction`；但 AUC 仍没赢。
2. Fashion 20-epoch 出现明确 AUC advantage：top1/top2/top3 的 AUC improvement 约 `16.9%-17.1%`，但 test acc 仍差 `0.72%-1.28%`。
3. KMNIST 20-epoch top2 最好，gap 降到 `1.46%`，AUC improvement `4.2%`，同时保留 `41% phi' reduction` 和 `22% J reduction`。它接近 medium gate，但 AUC improvement 还没到 `>5%`。
4. Top configs 的 branch 仍偏弱：Fashion 只有 AdamW 的 `0.36-0.46`，KMNIST top2 约 `0.67-0.69`。

Failure table：

```text
F0 + F1_search failure_count = 613
  no_convergence_gain = 330
  accuracy_gap_too_large = 283

F1_confirm failure_count = 63
  no_convergence_gain = 33
  accuracy_gap_too_large = 30

run_failed = 0
```

### Stage F 当前结论

1. Stage F 没有证明 functional update 全面优于 AdamW；它证明了当前主线的核心 tradeoff：更稳定几何 vs AdamW 更强 final accuracy。
2. AnalyticAdj / FirstOrder credit 不是瓶颈；F0 继续支持 D0/D1：Analytic 与 FullBP 对齐，FirstOrder 近似可用，Identity 不能作为唯一 credit。
3. Functional update 在 Fashion 20-epoch 上有明确 AUC 优势，但 accuracy 仍略低；在 KMNIST 20-epoch top2 上接近可接受区，但尚未稳定过 medium gate。
4. 下一步不应该只继续粗暴扩大 `coeff_lr`。更合理方向：

```text
branch activation schedule: early higher alpha/coeff_lr, later geometry regularized
adaptive warmup: diag phase longer for hard tasks, Sobolev phase later介入
loss-aware or branch-aware coeff_lr schedule
separate alpha_k schedule / regularization
compare 20+ epoch with tuned AdamW decay to avoid AdamW late loss rebound
```

5. Stage G 前建议：

```text
保留主线 = AnalyticAdj + functional update
保留候选 = FirstOrderDecompAdj
需要改进 = functional update schedule / branch utilization
不应 claim = 当前 Stage F 已经在 convergence efficiency 上全面赢 AdamW
可 claim = 在保持 30%-50% geometry reduction 的同时，Fashion/KMNIST 可以接近 AdamW accuracy；部分 20-epoch 设置 AUC 更好
```

## 2026-04-30 DG-KAN Stage F4：branch-aware / loss-aware schedule

本轮目标：针对 Stage F 发现的 branch under-active 问题，做 schedule 化训练：

```text
early phase:
  提高 KAN branch scale
  提高 KAN coefficient lr multiplier
  使用 diag functional update 加快早期下降

late phase:
  branch scale decay / 收缩
  切换到 Sobolev functional update
  用更低 phi' / Jacobian condition 收住几何
```

输出目录：

```text
results/stage_f_schedule_gpu0/
results/stage_f_schedule_refine_gpu0/
results/stage_f_schedule_smoke_gpu0/
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES = 0
device = cuda
formal schedule runs = 70, failed = 0
KMNIST refine runs = 45, failed = 0
smoke runs = 14, failed = 0
```

### 代码变更

更新 `experiments/stage_a_dgkan.py`：

- `KANBlock` 增加 `branch_scale`；
- residual KAN branch 使用：

```text
effective_alpha = alpha_k * branch_scale
h <- h + effective_alpha * KAN(LN(h))
```

更新 `experiments/stage_b_functional.py`：

- 增加 `--branch-schedule`：
  - `none`
  - `early_active_cosine`
  - `early_active_linear`
  - `loss_aware`
- 增加 schedule 参数：
  - `--branch-boost`
  - `--branch-final-scale`
  - `--branch-active-frac`
  - `--branch-max-active-frac`
  - `--coeff-lr-boost`
  - `--coeff-lr-final-mult`
  - `--loss-aware-patience`
  - `--loss-aware-min-delta`
  - `--loss-aware-min-epochs`
- `diag_to_sobolev` 在 schedule active phase 用 `diag`，switch 后用 `sobolev`；
- 记录：

```text
schedule/branch_scale
schedule/coeff_lr_multiplier
schedule/effective_coeff_lr
schedule/phase_id
schedule/branch_switch_step_final
schedule/loss_aware_switch_step
schedule/loss_aware_switch_reason
branch/global/branch_scale
branch/global/effective_alpha_mean
```

更新 `experiments/run_stage_f.py`：

- 新增 `F4_schedule`：
  - Fashion-MNIST / KMNIST
  - AdamW baseline
  - static functional top config
  - early-active cosine schedule
  - loss-aware schedule
- 新增 `F4_schedule_refine`：
  - 针对 KMNIST 第一轮 boost 过强的问题；
  - branch boost 从 `2/3` 降到 `1.2/1.5`；
  - coeff lr boost 从 `1.5/2` 降到 `1.1/1.2`；
  - 增加 `branch_final_scale=0.8` 的晚期收缩版本。

编译检查：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python -m py_compile \
  experiments/stage_a_dgkan.py \
  experiments/stage_b_functional.py \
  experiments/run_stage_f.py
```

### 执行命令

Smoke：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_schedule_smoke_gpu0 \
  --phases F4_schedule \
  --seeds 0 \
  --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 16 --depth 2 --basis-count 8 \
  --batch-size 128 --eval-batch-size 256 --audit-batch-size 64 \
  --device cuda \
  --continue-on-error
```

正式 schedule：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_schedule_gpu0 \
  --phases F4_schedule \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_schedule_gpu0 \
  --aggregate-only
```

KMNIST mild refinement：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_schedule_refine_gpu0 \
  --phases F4_schedule_refine \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_schedule_refine_gpu0 \
  --aggregate-only
```

### Fashion-MNIST：schedule 成功

说明：`acc gap` 为相对 AdamW，负数表示优于 AdamW；AUC / phi' / J reduction 为正数表示优于 AdamW。

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | no-KAN drop | switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8368 | 0.00% | 0.0% | 0.0% | 0.0% | 1.00 | 0.051 | - |
| static_top | 0.8296 | +0.72% | +16.9% | +37.9% | +50.8% | 0.46 | 0.022 | - |
| early b2 c1.5 af0.35 | 0.8372 | -0.04% | +17.5% | +32.7% | +47.1% | 0.58 | 0.043 | 168 |
| early b3 c2 af0.25 | 0.8392 | -0.24% | +22.0% | +24.5% | +52.4% | 0.64 | 0.046 | 120 |
| early b3 c2 af0.35 | 0.8420 | -0.52% | +21.9% | +21.9% | +50.0% | 0.67 | 0.054 | 168 |
| loss b2 c1.5 max0.50 | 0.8436 | -0.68% | +17.4% | +29.4% | +41.9% | 0.59 | 0.046 | 134 |
| loss b3 c2 max0.50 | 0.8394 | -0.26% | +19.8% | +16.4% | +41.1% | 0.65 | 0.046 | 149 |

Fashion 结论：

1. branch-aware / loss-aware schedule 明确修复了 static functional 的 accuracy gap：`static_top` 比 AdamW 差 `0.72%`，schedule 后最优反超 `0.68%`。
2. schedule 同时保留了收敛优势：val-loss AUC improvement 约 `17%-22%`。
3. 几何没有失控：最优 accuracy 的 `loss_b2_c1.5` 仍有 `29.4% phi' reduction` 和 `41.9% J reduction`。
4. branch activity 从 static 的 `0.46x AdamW` 提到 `0.58-0.67x`，no-KAN drop 从 `0.022` 提到 `0.043-0.054`，说明 KAN branch 的确更参与任务。

### KMNIST 第一轮：boost=2/3 过强

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | no-KAN drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8760 | 0.00% | 0.0% | 0.0% | 0.0% | 1.00 | 0.143 |
| static_top | 0.8614 | +1.46% | +4.2% | +41.0% | +22.2% | 0.69 | 0.119 |
| early b2 c1.5 af0.35 | 0.8558 | +2.02% | +13.2% | +4.7% | -21.3% | 1.28 | 0.124 |
| loss b2 c1.5 max0.50 | 0.8552 | +2.08% | +12.1% | +1.7% | -34.5% | 1.27 | 0.122 |
| early b3 c2 af0.35 | 0.8588 | +1.72% | -143.2% | -278.0% | -2582.5% | 4.34 | 0.178 |
| loss b3 c2 max0.50 | 0.8584 | +1.76% | -169.7% | -284.9% | -3374.9% | 4.25 | 0.179 |

KMNIST 第一轮结论：

1. `boost=2` 能提高 branch activity，但已经把 Jacobian condition 拉坏；accuracy 也从 static 的 `+1.46% gap` 变成 `+2.02% / +2.08% gap`。
2. `boost=3` 明显过强，branch 变成 AdamW 的 `4.2x-4.3x`，AUC 和 geometry 全面失控。
3. 结论不是“提高 branch 活性无效”，而是 KMNIST 的 boost 需要明显更温和，并且需要更早或更强的 late geometry 收缩。

### KMNIST refinement：小幅 boost 更合理

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | no-KAN drop | switch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8760 | 0.00% | 0.0% | 0.0% | 0.0% | 1.00 | 0.143 | - |
| static_top | 0.8614 | +1.46% | +4.2% | +41.0% | +22.2% | 0.69 | 0.119 | - |
| early b1.2 c1.1 af0.25 | 0.8620 | +1.40% | +7.7% | +40.0% | +21.2% | 0.71 | 0.125 | 120 |
| early b1.5 c1.2 af0.25 | 0.8640 | +1.20% | +9.5% | +38.1% | +18.6% | 0.75 | 0.132 | 120 |
| early b1.5 c1.2 af0.35 | 0.8644 | +1.16% | +9.7% | +37.6% | +16.3% | 0.77 | 0.130 | 168 |
| loss b1.2 c1.1 max0.35 | 0.8638 | +1.22% | +8.6% | +39.4% | +19.0% | 0.73 | 0.123 | 134 |
| loss b1.5 c1.2 max0.35 | 0.8656 | +1.04% | +10.1% | +36.6% | +10.7% | 0.76 | 0.129 | 130 |
| loss b1.5 c1.2 max0.35 final0.8 | 0.8642 | +1.18% | +10.6% | +34.3% | +24.7% | 0.67 | 0.126 | 130 |
| early b1.5 c1.2 af0.25 final0.8 | 0.8574 | +1.86% | +9.5% | +36.0% | +31.1% | 0.65 | 0.129 | 120 |

KMNIST refinement 结论：

1. mild boost 修复了 aggressive schedule 的失控问题。branch / AdamW 被控制在 `0.71-0.77`，没有再冲到 `1.3x` 或 `4x`。
2. 最优 `loss_b15_c12_max035` 把 static 的 accuracy gap 从 `1.46%` 降到 `1.04%`，AUC improvement 从 `4.2%` 提到 `10.1%`。
3. 代价是 geometry reduction 比 static 稍弱：`phi' reduction 36.6%` 仍好，`J reduction 10.7%` 较低。
4. `final_scale=0.8` 能更强地收 Jacobian：`J reduction 24.7%`，但 accuracy 比 `final_scale=1.0` 略低。这说明 late branch shrink 是有效 geometry knob，但需要按任务调强度。

Failure table：

```text
F4_schedule failure_count = 15
  accuracy_gap_too_large = 15

F4_schedule_refine failure_count = 10
  accuracy_gap_too_large = 10

run_failed = 0
```

### Stage F4 总结论

1. branch-aware / loss-aware schedule 是有效方向。Fashion 上已经同时实现：

```text
test acc >= AdamW
val-loss AUC better by 17%-22%
phi' / Jacobian condition substantially lower
KAN branch activity stronger than static functional
```

2. KMNIST 上不能用大 boost。`branch_boost=2/3` 会把 branch 和 Jacobian 推爆；mild boost `1.2-1.5` 更合理。
3. 当前推荐默认：

```text
Fashion:
  loss_aware, branch_boost=2.0, coeff_lr_boost=1.5, max_active_frac=0.50
  或 early_active_cosine, branch_boost=3.0, coeff_lr_boost=2.0, active_frac=0.35

KMNIST:
  loss_aware, branch_boost=1.5, coeff_lr_boost=1.2, max_active_frac=0.35
  如果更重视 geometry，则用 branch_final_scale=0.8
```

4. 这轮比 Stage F 的静态 functional update 有实质改进：

```text
Fashion static gap +0.72% -> best schedule -0.68%
KMNIST static gap +1.46% -> best schedule +1.04%
KMNIST static AUC +4.2% -> best schedule +10.1%
```

5. 后续如果继续优化，应该做真正 geometry-aware trigger：

```text
switch / shrink when:
  branch_over_adamw proxy too high
  phi_prime_p95 exceeds threshold
  Jacobian condition moving average exceeds threshold
  validation loss plateau
```

当前实现已经包含 loss-aware switch，但还没有在线 Jacobian / phi' trigger。

## 2026-04-30 DG-KAN Stage F5：online geometry-aware trigger

本轮目标：把上一轮的固定 / loss-aware schedule 升级为在线 geometry-aware trigger，用训练过程中的几何信号决定何时从 active phase 切到 Sobolev，以及是否进一步 shrink branch scale。

在线监控信号：

```text
phi trigger:
  kan/phi_prime_p95 > geometry_phi_high

Jacobian trigger:
  max_k jacobian/layer_k/condition > geometry_jac_high

branch trigger:
  branch/global/mean_output_norm_ratio > geometry_branch_high
```

动作：

```text
before switch:
  branch_scale = branch_boost
  coeff_lr_multiplier = coeff_lr_boost
  update = diag

trigger 或 max_active_frac 后:
  branch_scale = current_branch_scale
  coeff_lr_multiplier = coeff_lr_final_mult
  update = Sobolev

after switch:
  如果 geometry 仍超阈值：
    current_branch_scale <- max(min_branch_scale, current_branch_scale * shrink_factor)
```

输出目录：

```text
results/stage_f_geometry_smoke_gpu0/
results/stage_f_geometry_gpu0/
```

GPU / 环境：

```text
CUDA_VISIBLE_DEVICES = 0
smoke runs = 10, failed = 0
formal geometry runs = 50, failed = 0
```

### 代码变更

更新 `experiments/stage_b_functional.py`：

- 新增 `branch_schedule = geometry_aware`；
- 新增参数：

```text
--geometry-phi-high
--geometry-jac-high
--geometry-branch-high
--geometry-branch-low
--geometry-shrink-factor
--geometry-min-branch-scale
--geometry-trigger-interval
--geometry-min-epochs
```

- 每个 epoch 可在线执行 architecture audit，读取 exact block Jacobian condition；
- 新增 runtime state：

```text
schedule/geometry_switch_step
schedule/geometry_switch_reason
schedule/geometry_current_branch_scale
schedule/geometry_shrink_events
schedule/geometry_last_phi_p95
schedule/geometry_last_jac_condition
schedule/geometry_last_branch_proxy
schedule/geometry_last_reasons
```

更新 `experiments/run_stage_f.py`：

- 新增 `F4_geometry` phase；
- 任务：
  - Fashion-MNIST
  - KMNIST
- 每个任务 5 seeds；
- 每个任务包含 AdamW baseline、static_top、geometry-aware configs。

编译检查：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python -m py_compile \
  experiments/stage_b_functional.py \
  experiments/run_stage_f.py
```

执行命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_geometry_gpu0 \
  --phases F4_geometry \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_geometry_gpu0 \
  --aggregate-only
```

### Fashion-MNIST

阈值：

```text
phi_high = 0.052
jac_high = 4.5
branch_high = 0.30
```

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | no-KAN drop | switch | scale | shrink |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8368 | 0.00% | 0.0% | 0.0% | 0.0% | 1.00 | 0.051 | - | 1.00 | 0.0 |
| static_top | 0.8296 | +0.72% | +16.9% | +37.9% | +50.8% | 0.46 | 0.022 | - | 1.00 | 0.0 |
| geo b2 c1.5 max0.50 | 0.8452 | -0.84% | +15.8% | +28.7% | +40.2% | 0.61 | 0.048 | 226 | 1.00 | 0.0 |
| geo b2 c1.5 max0.50 final0.8 | 0.8468 | -1.00% | +17.8% | +26.8% | +49.1% | 0.53 | 0.044 | 226 | 0.80 | 0.0 |
| geo b3 c2 max0.50 | 0.8396 | -0.28% | +21.2% | +21.8% | +49.6% | 0.61 | 0.043 | 86 | 1.00 | 0.0 |

Trigger 细节：

```text
geo b2 c1.5:
  switch_step mean = 225.6
  switch_reason = jac_high 或 max_active_frac
  shrink_events = 0

geo b3 c2:
  switch_step mean = 86.4
  switch_reason = branch_high / jac_high
  shrink_events = 0
```

Fashion 结论：

1. online geometry-aware trigger 是本轮最好的 Fashion 结果。`geo_b2_c15_max05_f08` 达到：

```text
test acc = 0.8468
vs AdamW = +1.00%
val-loss AUC improvement = +17.8%
phi' reduction = +26.8%
Jacobian reduction = +49.1%
```

2. 相比上一轮最好的 fixed/loss-aware：

```text
previous best Fashion = loss_b2_c15_max05
test acc = 0.8436
AUC improvement = +17.4%
J reduction = +41.9%

new best geometry-aware final0.8
test acc = 0.8468
AUC improvement = +17.8%
J reduction = +49.1%
```

3. `geo_b3_c2` 的 switch 明显更早，说明 trigger 能识别 aggressive branch 的早期不稳定；但 b3 不是最优，最优仍是 b2 + late scale 0.8。

### KMNIST

阈值：

```text
phi_high = 0.055
jac_high = 4.5
branch_high = 0.30
```

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | no-KAN drop | switch | scale | shrink |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8760 | 0.00% | 0.0% | 0.0% | 0.0% | 1.00 | 0.143 | - | 1.00 | 0.0 |
| static_top | 0.8614 | +1.46% | +4.2% | +41.0% | +22.2% | 0.69 | 0.119 | - | 1.00 | 0.0 |
| geo b1.5 c1.2 max0.35 | 0.8650 | +1.10% | +9.7% | +36.5% | +10.3% | 0.77 | 0.129 | 144 | 1.00 | 0.0 |
| geo b1.5 c1.2 max0.35 final0.8 | 0.8638 | +1.22% | +10.4% | +34.3% | +25.1% | 0.68 | 0.126 | 144 | 0.80 | 0.0 |
| geo b2 c1.5 guard | 0.8444 | +3.16% | +9.0% | +2.1% | +3.6% | 1.04 | 0.109 | 48 | 0.82 | 1.4 |

Trigger 细节：

```text
geo b1.5 c1.2:
  switch_step mean = 144.0
  switch_reason = jac_high
  shrink_events = 0

geo b1.5 c1.2 final0.8:
  switch_step mean = 144.0
  switch_reason = jac_high
  shrink_events = 0

geo b2 c1.5 guard:
  switch_step mean = 48.0
  switch_reason = jac_high | branch_high
  shrink_events mean = 1.4
```

KMNIST 结论：

1. mild geometry-aware 与上一轮 mild loss-aware 基本持平：

```text
previous loss_b15_c12_max035:
  test acc = 0.8656
  acc gap = +1.04%
  AUC improvement = +10.1%
  J reduction = +10.7%

new geo_b15_c12_max035:
  test acc = 0.8650
  acc gap = +1.10%
  AUC improvement = +9.7%
  J reduction = +10.3%
```

2. `final_scale=0.8` 仍是更偏 geometry 的版本：accuracy 略低，但 J reduction 从 `10.3%` 提升到 `25.1%`。
3. aggressive `b2 c1.5` 被 trigger 成功刹住了 Jacobian：上一轮 fixed b2 的 J reduction 是 `-21.3%`，本轮 guard 后变成 `+3.6%`；但 switch 太早、shrink 过多，accuracy gap 恶化到 `+3.16%`。所以 KMNIST 不建议从 aggressive boost 开始。

Failure table：

```text
F4_geometry failure_count = 11
  accuracy_gap_too_large = 11

run_failed = 0
```

### Stage F5 总结论

1. online geometry-aware trigger 有效，尤其 Fashion 上比上一轮 fixed/loss-aware 更好：

```text
Fashion best:
  Stage F4 fixed/loss-aware: acc 0.8436, AUC +17.4%, J +41.9%
  Stage F5 geometry-aware: acc 0.8468, AUC +17.8%, J +49.1%
```

2. 对 KMNIST，online trigger 的主要价值是防止几何失控，而不是自动提升 accuracy。mild geometry-aware 可保持上一轮最佳附近；aggressive guard 虽然救回 Jacobian，但 accuracy 不可接受。
3. 当前推荐：

```text
Fashion default:
  geometry_aware
  branch_boost = 2.0
  coeff_lr_boost = 1.5
  branch_max_active_frac = 0.50
  branch_final_scale = 0.8
  phi_high = 0.052
  jac_high = 4.5
  branch_high = 0.30

KMNIST default:
  geometry_aware 或 loss_aware 均可
  branch_boost = 1.5
  coeff_lr_boost = 1.2
  branch_max_active_frac = 0.35
  branch_final_scale = 1.0 for accuracy
  branch_final_scale = 0.8 for stronger geometry
```

4. 后续若继续优化 KMNIST，应做更细的 threshold sweep，而不是加大 boost：

```text
branch_high in {0.24, 0.27, 0.30}
jac_high in {3.8, 4.2, 4.5}
branch_final_scale in {0.8, 0.9, 1.0}
```

5. 目前可以 claim：

```text
geometry-aware trigger can prevent over-active branch / bad Jacobian regimes,
and on Fashion-MNIST it improves the previous best schedule while preserving geometry.
```

## 2026-04-30 DG-KAN Stage F6：KMNIST geometry threshold sweep

目标：继续追 KMNIST accuracy，不再加大 `branch_boost`，而是围绕 online geometry-aware trigger 扫阈值：

```text
branch_high in {0.27, 0.30, 0.34}
jac_high in {4.2, 4.5, 5.0}
branch_final_scale in {0.8, 0.9, 1.0}

fixed:
  dataset = KMNIST
  branch_schedule = geometry_aware
  branch_boost = 1.5
  coeff_lr_boost = 1.2
  branch_max_active_frac = 0.35
  phi_high = 0.055
  seeds search = 0,1,2
  seeds confirm = 0,1,2,3,4
```

输出目录：

```text
results/stage_f_geometry_threshold_sweep_gpu0/
results/stage_f_geometry_threshold_confirm_gpu0/
```

GPU / 完成情况：

```text
CUDA_VISIBLE_DEVICES = 0
threshold search runs = 87, failed = 0
top confirm runs = 25, failed = 0
```

代码变更：

- `experiments/run_stage_f.py` 新增：

```text
F4_geometry_threshold_sweep
F4_geometry_threshold_confirm
```

执行命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_SILENT=true
export PYTHONWARNINGS=ignore

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_geometry_threshold_sweep_gpu0 \
  --phases F4_geometry_threshold_sweep \
  --seeds 0,1,2 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_geometry_threshold_sweep_gpu0 \
  --aggregate-only

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_geometry_threshold_confirm_gpu0 \
  --phases F4_geometry_threshold_confirm \
  --seeds 0,1,2,3,4 \
  --device cuda \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/lca/bin/python experiments/run_stage_f.py \
  --out-dir results/stage_f_geometry_threshold_confirm_gpu0 \
  --aggregate-only
```

### Search 结果

Search 使用 seeds `0,1,2`。按 `acc gap` 排序，同时要求 AUC improvement 和 J reduction 都为正。

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | switch | final scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bh0.27 jh4.5 fs0.9 | 0.8683 | +1.00% | +13.0% | +37.8% | +17.6% | 0.71 | 136 | 0.9 |
| bh0.30 jh4.5 fs0.9 | 0.8683 | +1.00% | +13.0% | +37.8% | +17.6% | 0.71 | 136 | 0.9 |
| bh0.34 jh4.5 fs0.9 | 0.8683 | +1.00% | +13.0% | +37.8% | +17.6% | 0.71 | 136 | 0.9 |
| bh0.27 jh4.2 fs1.0 | 0.8683 | +1.00% | +13.0% | +38.9% | +11.3% | 0.75 | 120 | 1.0 |
| bh0.30 jh4.2 fs1.0 | 0.8683 | +1.00% | +13.0% | +38.9% | +11.3% | 0.75 | 120 | 1.0 |
| bh0.34 jh4.2 fs1.0 | 0.8683 | +1.00% | +13.0% | +38.9% | +11.3% | 0.75 | 120 | 1.0 |
| bh0.30 jh4.5 fs0.8 | 0.8670 | +1.13% | +13.2% | +36.6% | +24.4% | 0.66 | 136 | 0.8 |

Factor mean：

| factor | value | acc gap | AUC improv | phi' red | J red | branch / AdamW | switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| branch_high | 0.27 | +1.12% | +13.0% | +37.7% | +17.7% | 0.71 | 133 |
| branch_high | 0.30 | +1.12% | +13.0% | +37.7% | +17.7% | 0.71 | 133 |
| branch_high | 0.34 | +1.12% | +13.0% | +37.7% | +17.7% | 0.71 | 133 |
| jac_high | 4.2 | +1.12% | +13.1% | +37.9% | +17.7% | 0.70 | 120 |
| jac_high | 4.5 | +1.08% | +12.9% | +37.7% | +17.5% | 0.71 | 136 |
| jac_high | 5.0 | +1.17% | +13.0% | +37.7% | +17.8% | 0.71 | 144 |
| final_scale | 0.8 | +1.19% | +13.2% | +36.6% | +24.4% | 0.66 | 133 |
| final_scale | 0.9 | +1.11% | +13.1% | +37.8% | +17.8% | 0.71 | 133 |
| final_scale | 1.0 | +1.07% | +12.8% | +38.8% | +10.8% | 0.75 | 133 |

Search 结论：

1. `branch_high` 在 `0.27-0.34` 区间几乎不影响结果；mild boost 下最终 branch proxy 约 `0.21-0.23`，switch 主要由 Jacobian 触发。
2. `jac_high=4.5` 在 search 中是最好 accuracy / geometry 折中；`5.0` 没有带来 accuracy gain。
3. `final_scale=1.0` accuracy 最好，但 `final_scale=0.9` 基本不掉 accuracy 且 J reduction 更好；`final_scale=0.8` 几何最好但 accuracy 低一些。

### Confirm 结果

Confirm 使用 seeds `0,1,2,3,4`。

| config | test acc | acc gap | AUC improv | phi' red | J red | branch / AdamW | no-KAN drop | switch | final scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8760 | 0.00% | 0.0% | 0.0% | 0.0% | 1.00 | 0.143 | - | 1.0 |
| static_top | 0.8614 | +1.46% | +4.2% | +41.0% | +22.2% | 0.69 | 0.119 | - | 1.0 |
| tuned jh4.5 fs0.9 | 0.8654 | +1.06% | +10.2% | +35.5% | +17.9% | 0.73 | 0.129 | 144 | 0.9 |
| tuned jh4.2 fs1.0 | 0.8652 | +1.08% | +10.0% | +36.6% | +10.6% | 0.76 | 0.131 | 125 | 1.0 |
| tuned jh4.5 fs0.8 | 0.8638 | +1.22% | +10.4% | +34.3% | +25.1% | 0.68 | 0.126 | 144 | 0.8 |

Confirm 结论：

1. Threshold sweep 追到了小幅 accuracy gain，但没有突破 `1%` gap：

```text
Stage F5 best geometry-aware:
  test acc = 0.8650
  acc gap = +1.10%
  AUC improvement = +9.7%
  J reduction = +10.3%

Stage F6 tuned best:
  test acc = 0.8654
  acc gap = +1.06%
  AUC improvement = +10.2%
  J reduction = +17.9%
```

2. 相比 static functional：

```text
static_top:
  test acc = 0.8614
  acc gap = +1.46%
  AUC improvement = +4.2%

tuned jh4.5 fs0.9:
  test acc = 0.8654
  acc gap = +1.06%
  AUC improvement = +10.2%
```

3. 最推荐配置：

```text
KMNIST default after threshold sweep:
  branch_schedule = geometry_aware
  branch_boost = 1.5
  coeff_lr_boost = 1.2
  branch_max_active_frac = 0.35
  phi_high = 0.055
  jac_high = 4.5
  branch_high = 0.30
  branch_final_scale = 0.9
```

4. 如果更重视 accuracy，可用 `jac_high=4.2, final_scale=1.0`，但 J reduction 只有 `10.6%`；如果更重视 geometry，用 `final_scale=0.8`，J reduction 到 `25.1%`，但 accuracy gap 回到 `1.22%`。

Failure table：

```text
F4_geometry_threshold_sweep failure_count = 28
  accuracy_gap_too_large = 28

F4_geometry_threshold_confirm failure_count = 7
  accuracy_gap_too_large = 7

run_failed = 0
```

### Stage F6 总结论

1. KMNIST threshold sweep 有收益，但收益进入平台期：accuracy gap 从 `1.10%` 降到 `1.06%`，不是质变。
2. 当前 KMNIST 的瓶颈不主要在 `branch_high / jac_high / final_scale` 阈值，而更可能在：

```text
functional update strength / optimizer dynamics
模型容量或 basis coverage
rest optimizer 与 KAN branch 的协同
训练 epoch / LR decay
```

3. 后续若继续追 KMNIST accuracy，不建议继续大范围扫 geometry threshold；更建议：

```text
coeff_lr around {0.3, 0.4, 0.5}
rest_lr around {0.002, 0.003, 0.004}
epochs {20, 30}
保持 geometry-aware tuned threshold:
  jac_high=4.5, final_scale=0.9
```

4. 当前可 claim：

```text
threshold tuning improves KMNIST over static functional and previous geometry-aware setting,
but does not close the remaining ~1% accuracy gap to AdamW.
```


## 2026-05-01 docs/Next 恢复版复现实验：FGO-v4 mini clean/noise/stress

本轮目标：阅读 `docs/Next` 下 5 个后续计划文件，并优先复现其中当前代码面可直接执行的实验内容。

已阅读文件：

```text
docs/Next/DG-KAN_GFU_Functional_Update_升级实验计划.md
docs/Next/DG-KAN_Functional_Update_GFU_完整优化方案.md
docs/Next/DG-KAN_Functional_Update_v2_全面升级实验计划.md
docs/Next/DG-KAN_FGO-v3_后续实验计划.md
docs/Next/DG-KAN_FGO-v4_实验结果分析与下一步计划.md
```

阅读后的对应关系：

1. GFU / FGO-v2 / FGO-v3 文档主要是从 GFU-lite 到 FGO-v3 的逐层升级计划，完整复现需要旧的 GFU-lite trust/branch-controller/target-solve/FSAM runner。
2. 当前恢复代码已经覆盖 FGO-v4 文档的最小可执行包：`V4-mini-clean`、`V4-mini-noise`、`V4-mini-stress`。
3. 因此本轮先做恢复版 FGO-v4 mini 复现。注意：这是 `seed=0, epochs=8` 的可执行复现，不是文档中 `20 epoch, 3/5 seeds` 的正式 confirm。

### 代码修复

跑 `V4-mini-stress` 时发现 `train_size` override 与 common args 重复传参，导致：

```text
TypeError: build_train_config_for_method() got multiple values for keyword argument 'train_size'
```

修复：`experiments/dgkan_core.py::config_from_args` 先构造 `base_kwargs`，再用 spec overrides 覆盖，避免重复 keyword。修复后：

```text
python -m py_compile experiments/dgkan_core.py experiments/run_fgo_v4.py experiments/run_stage_f.py
```

通过。

### 执行命令

V4-mini-clean：

```bash
python experiments/run_fgo_v4.py \
  --phases V4-mini-clean \
  --seeds 0 \
  --out-dir results/next_repro_v4_clean_seed0_e8 \
  --fresh --device auto --no-download \
  --epochs 8 --train-size 6000 --val-size 1000 --test-size 1000 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

V4-mini-noise：

```bash
python experiments/run_fgo_v4.py \
  --phases V4-mini-noise \
  --seeds 0 \
  --out-dir results/next_repro_v4_noise_seed0_e8 \
  --fresh --device auto --no-download \
  --epochs 8 --train-size 6000 --val-size 1000 --test-size 1000 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

V4-mini-stress：

```bash
python experiments/run_fgo_v4.py \
  --phases V4-mini-stress \
  --seeds 0 \
  --out-dir results/next_repro_v4_stress_seed0_e8 \
  --fresh --device auto --no-download \
  --epochs 8 --train-size 6000 --val-size 1000 --test-size 1000 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

数据检查：

```text
clean rows = 36, errors = 0, used_fake_data = 0
noise rows = 18, errors = 0, used_fake_data = 0
stress rows = 10, errors = 0, used_fake_data = 0
```

### V4-mini-clean：KMNIST

Baseline：

| method | test acc | val AUC | phi' p95 | max J cond | ECE | step ms |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.7470 | 0.5424 | 0.1701 | 31398.8 | 0.1405 | 27.3 |
| geometry_current | 0.7460 | 0.6387 | 0.1478 | 579.4 | 0.0726 | 29.9 |
| no_grid | 0.7360 | 0.6759 | 0.1511 | 5311.7 | 0.0832 | 31.3 |
| fgo_v3_clean | 0.7470 | 0.5776 | 0.1474 | 526.0 | 0.0863 | 33.4 |

FGO-v4 data-pulse top accuracy：

| method | test acc | gap vs AdamW | val AUC improvement | phi' reduction | J reduction | time ratio vs geometry |
|---|---:|---:|---:|---:|---:|---:|
| le0.5_lf0_p0.15_m0.8_prox0 | 0.7600 | -1.30% | -4.62% | 13.27% | 92.58% | 1.03 |
| le0.5_lf0_p0.15_m0.8_prox3e-5 | 0.7600 | -1.30% | -4.62% | 13.29% | 93.01% | 1.14 |
| le1_lf0_p0.25_m0.8_prox0 | 0.7570 | -1.00% | -4.75% | 12.45% | 96.70% | 1.12 |
| le1_lf0_p0.25_m0.8_prox3e-5 | 0.7570 | -1.00% | -4.75% | 12.47% | 96.77% | 1.09 |

Failure table：

```text
no_convergence_gain = 33
data_metric_geometry_failure = 2
accuracy_gap_too_large = 1
run_failed = 0
```

Clean gate：

```text
Clean gate pass = False
```

观察：

1. 本轮恢复版中，FGO-v4 data-pulse + momentum 在 KMNIST 上确实有 accuracy signal，best acc `0.7600`，比 AdamW `0.7470` 高 `1.3%`。
2. 但 val AUC 没有改善，best accuracy configs 的 val AUC improvement 约 `-4.6%`，所以没有通过 V4 clean gate。
3. `m0.8` momentum 对 accuracy 有明显帮助；`m0` 配置更保守但 accuracy 较弱。
4. 结论方向与 `docs/Next/DG-KAN_FGO-v4...` 一致：data metric / momentum 不是没信号，关键矛盾仍是把 accuracy signal 转成稳定的 convergence + geometry-safe profile。

### V4-mini-noise：Fashion-MNIST label noise

Noise 0.2：

| method | test acc | gain vs geometry | ECE | ECE red vs AdamW | time ratio vs geometry | max J cond |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.7600 | -1.30% | 0.1347 | 0.0% | 1.14 | 12357.2 |
| geometry_current | 0.7730 | 0.00% | 0.1489 | -10.6% | 1.00 | 36.9 |
| snr_previous_best | 0.7670 | -0.60% | 0.1450 | -7.6% | 1.16 | 10619.5 |
| best SNR-lite output_i8 | 0.7780 | +0.50% | 0.1326 | +1.6% | 1.13 | 2984.6 |

Noise 0.4：

| method | test acc | gain vs geometry | ECE | ECE red vs AdamW | time ratio vs geometry | max J cond |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.7030 | -4.50% | 0.2553 | 0.0% | 1.07 | 13520.0 |
| geometry_current | 0.7480 | 0.00% | 0.2722 | -6.7% | 1.00 | 28.1 |
| snr_previous_best | 0.7630 | +1.50% | 0.2806 | -9.9% | 1.20 | 27080.9 |
| best SNR-lite output_i4 | 0.7320 | -1.60% | 0.2616 | -2.5% | 1.03 | 1113.7 |

Noise gate：

```text
Noise gate pass = False
```

观察：

1. high-noise (`0.4`) 下 `snr_previous_best` 复现出 clean acc 增益：`0.7630` vs geometry `0.7480`，gain `+1.5%`，且 time ratio `1.20` 在成本阈值内。
2. 但本轮 ECE 没有改善，反而比 AdamW/geometry 更差，因此不通过 V4-noise gate。
3. SNR-lite 在 noise `0.2` 有一个低成本小增益点：`output_channel_i8` acc `0.7780`，比 geometry 高 `0.5%`，time ratio `1.13`；但 gain 未达 `>1%`。
4. SNR-lite 在 noise `0.4` 没保住 previous-best 的 accuracy gain。

### V4-mini-stress

Small-data clean：

| dataset | method | test acc | val AUC | ECE | max J cond |
|---|---|---:|---:|---:|---:|
| Fashion-MNIST | AdamW_small | 0.7620 | 0.8784 | 0.0498 | 7520.4 |
| Fashion-MNIST | geometry_current_small | 0.7810 | 0.9342 | 0.0619 | 26.0 |
| Fashion-MNIST | fgo_v4_clean_best_small | 0.7830 | 0.9141 | 0.0783 | 243.6 |
| KMNIST | AdamW_small | 0.6170 | 1.1142 | 0.0625 | 2222.4 |
| KMNIST | geometry_current_small | 0.6210 | 1.1324 | 0.0741 | 116.7 |
| KMNIST | fgo_v4_clean_best_small | 0.6370 | 1.1124 | 0.0521 | 127.5 |

Noisy small-data：

| dataset | method | noise | test acc | val AUC | ECE | max J cond |
|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | geometry_current_small_noise0.2 | 0.2 | 0.7740 | 1.0630 | 0.1540 | 23.5 |
| Fashion-MNIST | snr_lite_layer_small_noise0.2 | 0.2 | 0.7460 | 1.0220 | 0.1322 | 2572.9 |
| Fashion-MNIST | geometry_current_small_noise0.4 | 0.4 | 0.7410 | 1.3164 | 0.2731 | 24.1 |
| Fashion-MNIST | snr_lite_layer_small_noise0.4 | 0.4 | 0.6690 | 1.2513 | 0.1936 | 876.6 |

观察：

1. Small-data clean 中，`fgo_v4_clean_best_small` 在 Fashion/KMNIST 都略高于 geometry-current，尤其 KMNIST `0.6370` vs `0.6210`。
2. Small-data noise 中，SNR-lite layer 降低 ECE 和 val AUC，但 accuracy 明显低于 geometry-current；它更像 calibration/noise stress 组件，不能作为默认。

### 本轮结论

1. `docs/Next` 的总体判断得到部分复现：继续堆大一统 optimizer 没意义，应该拆成 clean hard-task profile 和 noisy-label profile。
2. Clean side：FGO-v4 data-pulse + momentum 在 KMNIST 有明确 accuracy signal，但本轮没有 convergence/AUC pass；恢复版结果支持“hard-task candidate，而不是 clean default”。
3. Noise side：SNR previous-best 在 high noise 下仍有 clean acc gain，但 ECE 没复现为正，SNR-lite 也没有稳定保留收益；恢复版结果支持“SNR 仍是 noisy-label ablation/stress，不是默认”。
4. Stress side：FGO-v4 clean best 在 small-data clean 上表现不坏，KMNIST/Fashion 都略好于 geometry-current；但 noisy SNR-lite accuracy 退步明显。
5. 由于本轮是 `seed=0, epochs=8`，不能替代 docs 里要求的 `epochs=20, seeds=0..4` confirm。下一步若继续，应优先跑：

```text
V4-mini-clean top accuracy configs:
  le0.5_lf0_p0.15_m0.8_prox0
  le0.5_lf0_p0.15_m0.8_prox3e-5
  le1_lf0_p0.25_m0.8_prox0

V4-mini-noise candidates:
  snr_previous_best noise0.4
  snr_lite_output_channel_i8 noise0.2
```


## 2026-05-01 GA-FU 统一 optimizer：P0/P1 多 seed 与调参复核

依据文档：

```text
docs/DG-KAN_GA-FU_统一Optimizer下一轮实验计划.md
```

本轮目标：不继续堆新 optimizer 组件，固定并验证 GA-FU：

```text
GA-FU = early branch activation
      + diag/Sobolev functional update
      + online geometry trigger
      + late branch scaling
```

默认不加入：

```text
data metric / SNR / FSAM / prox / spectral / target solve / hard controller / momentum
```

新增 runner：

```text
experiments/run_gafu_consolidation.py
```

同时修复了一个关键调度 bug：原恢复代码只在 epoch 结束时检查 `branch_max_active_frac`，导致 P0 的 1-epoch smoke 中 active phase 实际持续到训练结束，GA-FU Jacobian 爆炸。修复后，training loop 内按 step progress 触发 `max_active_frac`，确保 active phase 是训练过程约束。

修复验证：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_consolidation.py
```

通过。

### P0 smoke

原始 P0：

```bash
python experiments/run_gafu_consolidation.py \
  --packages P0 \
  --out-dir results/gafu_consolidation_p0 \
  --fresh --device auto --no-download --continue-on-error
```

原始结果显示 GA-FU active phase 切换过晚：

```text
Fashion GA-FU acc = 0.1172, J = 183447
KMNIST  GA-FU acc = 0.1328, J = 105374
```

修复 max-active step trigger 后重跑：

```bash
python experiments/run_gafu_consolidation.py \
  --packages P0 \
  --out-dir results/gafu_consolidation_p0_v2 \
  --fresh --device auto --no-download --continue-on-error
```

修复后：

| dataset | method | acc | val AUC | branch | phi' | J | switch |
|---|---|---:|---:|---:|---:|---:|---|
| Fashion | StaticFunctional | 0.2969 | 2.3150 | 0.732 | 0.1462 | 8619.8 | - |
| Fashion | GA-FU | 0.2969 | 2.2533 | 0.943 | 0.1466 | 21.0 | max_active_frac |
| KMNIST | StaticFunctional | 0.2188 | 2.4801 | 0.715 | 0.1453 | 1411.2 | - |
| KMNIST | GA-FU | 0.1406 | 2.3246 | 0.822 | 0.1462 | 196.1 | max_active_frac |

Fashion P0 通过 smoke；KMNIST 默认 GA-FU 仍低于 static 超过 5%，因此先做 P0 尺度调参。

KMNIST P0 tuning：

```bash
python experiments/run_gafu_consolidation.py \
  --packages TUNE \
  --datasets KMNIST \
  --seeds 0 \
  --out-dir results/gafu_consolidation_p0_kmnist_tune2 \
  --fresh --device auto --no-download \
  --epochs 1 --train-size 512 --val-size 128 --test-size 128 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

P0 best smoke 候选：

| method | acc | gap vs static | val AUC | J |
|---|---:|---:|---:|---:|
| StaticFunctional | 0.2188 | 0.0% | 2.4801 | 1411 |
| GA-FU-KM-boost1.2 | 0.2031 | 1.6% | 2.2870 | 270 |
| GA-FU-KM-clr0.1-balanced | 0.1719 | 4.7% vs static | 2.1032 | 124 |

结论：KMNIST 默认 boost=1.5 在 P0 太激进；`boost=1.2` 可过 P0 smoke 的精度条件，低 coeff_lr 可改善 AUC/geometry 但 accuracy 更弱。

### P1 Fashion 默认 GA-FU：5 seeds, 20 epochs

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages P1 \
  --datasets Fashion-MNIST \
  --methods AdamW,StaticFunctional,GA-FU \
  --seeds 0,1,2,3,4 \
  --out-dir results/gafu_consolidation_p1_fashion_default \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 64 --depth 4 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap vs AdamW | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 5 | 0.8270 ± 0.0064 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| StaticFunctional | 5 | 0.8146 ± 0.0131 | 0.0124 | -0.080 | 0.056 | 0.738 | 0.913 | 0.062 |
| GA-FU default | 5 | 0.8134 ± 0.0295 | 0.0136 | -0.123 | 0.032 | 0.517 | 1.095 | -0.001 |

P1 gate：

```text
Fashion default GA-FU: FAIL
main failure = no_convergence_gain + weak geometry reduction + seed instability
```

### Fashion 调参：固定 GA-FU control，不加新组件

第一轮调参方向：

```text
coeff_lr: 0.05, 0.1, 0.2, 0.4
rest_lr: 0.001, 0.003, 0.004
branch_final_scale: 0.8, 0.9
```

第二轮调参方向：

```text
branch_boost: 1.2, 1.5
coeff_lr_boost: 1.0, 1.2
branch_max_active_frac: 0.25, 0.35
```

关键命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages TUNE \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_tune2_fashion_seed012_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 64 --depth 4 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

注：该 tuning 运行在拿到关键候选后停止，保留前 24 条有效结果。

Fashion tuning 主要结果：

| method | runs | acc mean ± std | gap vs AdamW | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 3 | 0.8293 ± 0.0074 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| GA-FU-FM-boost1.2-clr0.05-active0.35 | 3 | 0.8400 ± 0.0126 | -0.0107 | -0.032 | 0.188 | 0.999 | 0.780 | 0.220 |
| GA-FU-FM-clr0.05-rest0.001 | 3 | 0.8343 ± 0.0110 | -0.0050 | -0.024 | 0.181 | 1.000 | 0.905 | 0.168 |
| GA-FU-FM-boost1.5-clr0.05-active0.35 | 3 | 0.8323 ± 0.0074 | -0.0030 | -0.036 | 0.184 | 1.000 | 0.845 | 0.131 |

观察：

1. 降低 coeff/rest 后，Fashion accuracy 可以超过 AdamW。
2. `boost1.2 + coeff_lr=0.05 + rest_lr=0.001` 是 Fashion 最好 Pareto：acc `+1.07%`、J reduction `~99.9%`、ECE reduction `22%`。
3. 但 AUC improvement 仍为负，约 `-3.2%`；phi reduction `18.8%`，接近但未达 weak gate 的 `20%`。
4. `branch_max_active_frac=0.25/0.35` 与默认几乎等价，因为实际多在 epoch 2 被 `phi_high|jac_high` 提前触发。

### Fashion 训练长度复查：30 epochs

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages P1 \
  --datasets Fashion-MNIST \
  --methods AdamW,GA-FU-FM-boost1.2-clr0.05-active0.35 \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_p1_fashion_selected_seed012_e30 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 64 --depth 4 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap vs AdamW | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 3 | 0.8410 ± 0.0043 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| GA-FU-FM-boost1.2-clr0.05-active0.35 | 3 | 0.8457 ± 0.0163 | -0.0047 | -0.016 | 0.239 | 0.999 | 0.764 | 0.152 |

结论：30 epochs 改善了 phi reduction 到 `23.9%`，accuracy 仍略高于 AdamW，但 AUC 仍为负；训练长度不能让 Fashion 通过 P1 gate。

### P1 KMNIST selected tuning：3 seeds, 20 epochs

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages P1 \
  --datasets KMNIST \
  --methods AdamW,StaticFunctional,GA-FU,GA-FU-KM-boost1.2,GA-FU-KM-clr0.1-rest0.001,GA-FU-KM-clr0.2-balanced,GA-FU-KM-acc \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_p1_kmnist_selected_seed012_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 64 --depth 4 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap vs AdamW | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 3 | 0.7740 ± 0.0022 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| GA-FU default | 3 | 0.7737 ± 0.0200 | 0.0003 | -0.157 | 0.051 | 0.757 | 0.982 | 0.050 |
| GA-FU-KM-clr0.2-balanced | 3 | 0.7733 ± 0.0108 | 0.0007 | -0.072 | 0.119 | 0.816 | 0.874 | 0.081 |
| GA-FU-KM-boost1.2 | 3 | 0.7720 ± 0.0114 | 0.0020 | -0.106 | 0.122 | -0.744 | 0.810 | 0.067 |
| StaticFunctional | 3 | 0.7717 ± 0.0045 | 0.0023 | -0.142 | 0.069 | 0.553 | 0.869 | 0.068 |
| GA-FU-KM-acc | 3 | 0.7637 ± 0.0202 | 0.0103 | -0.138 | 0.059 | 0.817 | 1.009 | 0.007 |
| GA-FU-KM-clr0.1-rest0.001 | 3 | 0.7530 ± 0.0177 | 0.0210 | -0.030 | 0.173 | 0.967 | 0.864 | 0.104 |

观察：

1. KMNIST accuracy 可以接近 AdamW：default GA-FU gap 只有 `0.03%`，`clr0.2-balanced` gap `0.07%`。
2. 但所有候选 AUC improvement 仍为负。
3. `boost1.2` 在 P0 过 smoke，但 20 epoch 中 seed2 Jacobian 达到 `1e6` 上限，不稳定。
4. 低步长 `clr0.1/rest0.001` AUC 最接近，但 accuracy gap `2.1%`，不达标。

### KMNIST 训练长度复查：30 epochs

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages P1 \
  --datasets KMNIST \
  --methods AdamW,GA-FU-KM-clr0.2-balanced \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_p1_kmnist_selected_seed012_e30 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 64 --depth 4 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap vs AdamW | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 3 | 0.7820 ± 0.0059 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| GA-FU-KM-clr0.2-balanced | 3 | 0.7670 ± 0.0122 | 0.0150 | -0.110 | 0.118 | 0.891 | 0.864 | -0.003 |

结论：KMNIST 30 epochs 反而拉大 accuracy gap，AUC 仍为负；训练长度不是当前 gate 的解法。

### GA-FU 本轮总判断

P1 gate：

```text
Weak pass:  no
Medium pass: no
Strong pass: no
```

主要 failure：

```text
no_convergence_gain
phi_prime_reduction_insufficient
seed_instability on some GA-FU default runs
jacobian_spike on KMNIST boost1.2
```

本轮最好的 Pareto 点：

```text
Fashion:
  GA-FU-FM-boost1.2-clr0.05-active0.35
  20 epoch: acc 0.8400 vs AdamW 0.8293, J red ~99.9%, ECE red 22.0%, AUC red -3.2%
  30 epoch: acc 0.8457 vs AdamW 0.8410, phi red 23.9%, J red ~99.9%, ECE red 15.2%, AUC red -1.6%

KMNIST:
  GA-FU-KM-clr0.2-balanced
  20 epoch: acc 0.7733 vs AdamW 0.7740, J red 81.6%, ECE red 8.1%, AUC red -7.2%
```

可以 claim：

```text
GA-FU-style fixed geometry control can match AdamW-level accuracy on Fashion and KMNIST while greatly reducing Jacobian condition.
```

不可以 claim：

```text
GA-FU passes clean default gate.
GA-FU converges faster than AdamW.
GA-FU has validation-loss AUC advantage.
```

与文档第 13.3 一致，本轮不建议继续调 trigger 或添加 optimizer 组件。下一步应转向：

```text
model capacity
basis count
rest optimizer / stem-head coupling
LR decay or scheduler
training objective / validation-loss trajectory analysis
```

当前推荐状态：

```text
GA-FU = geometry-stable Pareto candidate, not clean default
AdamW = accuracy/convergence baseline
Stage I clean default: do not switch to GA-FU yet
```

## 2026-05-01 GA-FU 继续调参：AUC / 容量 / 训练长度

本轮目标：根据 `docs/DG-KAN_GA-FU_统一Optimizer下一轮实验计划.md`，在不新增 optimizer 组件的前提下继续调参。重点检查 weak gate：

```text
acc gap < 1.5%
val-loss AUC improvement > 5%
phi reduction > 20%
Jacobian reduction > 10%
0.45 < branch/AdamW < 0.95
```

代码变更：

```text
experiments/run_gafu_consolidation.py
  added packages: AUCTUNE, CAPTUNE, FASTTUNE, FASHION30, KMNEXT

experiments/dgkan_core.py
  result rows now include hidden_dim/depth/basis_count/alpha_init/head_hidden
  result rows now include val_loss_curve for trajectory diagnosis
```

语法检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_consolidation.py
```

### AUCTUNE：h64/b16, 20 epochs, seed 0/1/2

命令示例：

```bash
python experiments/run_gafu_consolidation.py \
  --packages AUCTUNE \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_auctune_fashion_seed012_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 64 --depth 4 --basis-count 16 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

另跑 KMNIST：

```text
results/gafu_consolidation_auctune_kmnist_seed012_e20
```

关键结果：

| dataset | method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Fashion | AdamW | 3 | 0.8293 ± 0.0091 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| Fashion | GA-FU-FM-auc-b1.0-c0.05-r0.001-f0.9-g0 | 3 | 0.8377 ± 0.0042 | -0.0083 | -0.030 | 0.187 | 0.990 | 0.768 | 0.271 | no |
| Fashion | GA-FU-FM-auc-b1.2-c0.05-r0.001-f0.8-g0 | 3 | 0.8343 ± 0.0207 | -0.0050 | -0.027 | 0.188 | 0.999 | 0.762 | 0.195 | no |
| KMNIST | AdamW | 3 | 0.7740 ± 0.0026 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| KMNIST | GA-FU-KM-auc-b1.1-c0.10-r0.001-f0.9-g0 | 3 | 0.7537 ± 0.0064 | 0.0203 | 0.020 | 0.179 | 0.953 | 0.802 | 0.116 | no |
| KMNIST | GA-FU-KM-auc-b1.2-c0.10-r0.001-f0.8-g0 | 3 | 0.7623 ± 0.0146 | 0.0117 | 0.003 | 0.177 | 0.998 | 0.765 | 0.173 | no |

观察：

```text
Fashion: acc/ECE/J 可改善，但 AUC 全负，phi 约 0.187-0.188，未达 0.20。
KMNIST: AUC 可略正，但 acc 或 phi 不同时达标。
```

### CAPTUNE：h96/b24, 20 epochs

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages CAPTUNE \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_captune_h96_b24_seed012_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

随后对 KMNIST 追加 seed3/4：

```bash
python experiments/run_gafu_consolidation.py \
  --packages CAPTUNE \
  --datasets KMNIST \
  --seeds 3,4 \
  --out-dir results/gafu_consolidation_captune_h96_b24_seed012_e20 \
  --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

关键结果：

| dataset | method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Fashion | AdamW | 3 | 0.8390 ± 0.0219 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| Fashion | GA-FU-FM-cap-b1.0-c0.05-r0.001-f0.9-g0 | 3 | 0.8333 ± 0.0097 | 0.0057 | -0.0275 | 0.265 | 0.823 | 0.662 | 0.117 | no |
| Fashion | GA-FU-FM-cap-b1.2-c0.05-r0.001-f0.8-g0 | 3 | 0.8377 ± 0.0166 | 0.0013 | -0.0260 | 0.264 | 0.787 | 0.650 | 0.115 | no |
| KMNIST | AdamW | 5 | 0.7688 ± 0.0221 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| KMNIST | GA-FU-KM-cap-b1.1-c0.10-r0.001-f0.9-g0 | 5 | 0.7594 ± 0.0158 | 0.0094 | 0.0296 | 0.268 | 0.586 | 0.681 | 0.207 | no |
| KMNIST | GA-FU-KM-cap-b1.1-c0.20-r0.003-f0.9-g0 | 5 | 0.7774 ± 0.0114 | -0.0086 | 0.0346 | 0.232 | 0.692 | 0.704 | 0.106 | no |
| KMNIST | GA-FU-KM-cap-b1.2-c0.10-r0.001-f0.8-g0 | 5 | 0.7592 ± 0.0101 | 0.0096 | 0.0425 | 0.268 | 0.105 | 0.663 | 0.167 | no |

观察：

```text
容量提升显著改善 phi/J gate。
Fashion 变成“只差 AUC”：accuracy、phi、J、branch、ECE 基本都合格，但 AUC 仍为负。
KMNIST 3-seed 曾有 weak-pass 候选，但补到 5 seeds 后 AUC 掉到 3%-4%，未达 5%。
```

### FASTTUNE：Fashion h96/b24, 20 epochs, seed 0/1/2

动机：`val_loss_curve` 显示 Fashion GA-FU 第 1 epoch val loss 明显高于 AdamW，但后半段 final loss 更低；因此试只提高 early coeff lr boost。

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages FASTTUNE \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_fasttune_fashion_h96_b24_seed012_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

关键结果：

| method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AdamW | 3 | 0.8390 ± 0.0219 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| GA-FU-FM-fast-b1.0-c0.05-cb2.0-r0.001-f0.9-g0 | 3 | 0.8383 ± 0.0251 | 0.0007 | -0.0113 | 0.263 | 0.302 | 0.686 | 0.173 | no |

观察：

```text
coeff_lr_boost=2.0 把 Fashion AUC 从约 -2.6% 拉到 -1.1%，但仍未转正。
更大的 branch_boost 或 coeff_lr=0.08 会伤 accuracy/AUC。
```

### FASHION30：Fashion h96/b24, 30 epochs, seed 0/1/2

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages FASHION30 \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_fashion30_h96_b24_seed012 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AdamW | 3 | 0.8403 ± 0.0186 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| GA-FU-FM-e30-b1.2-c0.05-r0.001-f0.8-g0 | 3 | 0.8407 ± 0.0196 | -0.0003 | 0.0121 | 0.322 | 0.916 | 0.638 | 0.100 | no |
| GA-FU-FM-e30-b1.0-c0.05-cb2.0-r0.001-f0.9-g0 | 3 | 0.8453 ± 0.0212 | -0.0050 | 0.0081 | 0.323 | 0.829 | 0.673 | 0.092 | no |

观察：

```text
延长到 30 epochs 后 Fashion AUC 转正，但只有 +0.8% 到 +1.2%，离 weak 的 +5% 仍远。
不过 accuracy、phi、J、branch 和 ECE 基本都达 weak 条件。
```

### KMNEXT：KMNIST h96/b24, 20 epochs, seed 0-4

动机：CAPTUNE 的 KMNIST AUC 接近 5%，继续在 `c0.15-0.20` 和 `final_scale=0.8` 周围小扫。

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages KMNEXT \
  --datasets KMNIST \
  --seeds 0,1,2,3,4 \
  --out-dir results/gafu_consolidation_captune_h96_b24_seed012_e20 \
  --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AdamW | 5 | 0.7688 ± 0.0221 | 0.0000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| GA-FU-KM-next-b1.2-c0.15-r0.003-f0.8-g0 | 5 | 0.7870 ± 0.0155 | -0.0182 | 0.0456 | 0.248 | 0.868 | 0.656 | 0.182 | no |
| GA-FU-KM-next-b1.1-c0.20-r0.003-f0.8-g0 | 5 | 0.7756 ± 0.0166 | -0.0068 | 0.0236 | 0.232 | 0.835 | 0.664 | 0.088 | no |
| GA-FU-KM-next-b1.2-c0.20-r0.003-f0.8-g0 | 5 | 0.7792 ± 0.0149 | -0.0104 | 0.0193 | 0.229 | 0.804 | 0.699 | 0.119 | no |

观察：

```text
best KMNIST 5-seed candidate:
  GA-FU-KM-next-b1.2-c0.15-r0.003-f0.8-g0
  acc beats AdamW by 1.82 points
  AUC improvement = +4.56%, just below weak gate +5%
  phi/J/branch/ECE all pass weak gate
```

### 本轮结论

```text
没有获得 5-seed weak pass。

最接近 gate 的两个组合：

Fashion:
  h96/b24, 30 epochs
  GA-FU-FM-e30-b1.2-c0.05-r0.001-f0.8-g0
  acc 0.8407 vs AdamW 0.8403
  AUC +1.21%
  phi red 32.2%, J red 91.6%, branch/A 0.638, ECE red 10.0%

KMNIST:
  h96/b24, 20 epochs
  GA-FU-KM-next-b1.2-c0.15-r0.003-f0.8-g0
  acc 0.7870 vs AdamW 0.7688
  AUC +4.56%
  phi red 24.8%, J red 86.8%, branch/A 0.656, ECE red 18.2%
```

判断：

```text
1. GA-FU 的“几何稳定 + calibration + accuracy”已经可以调出来。
2. 唯一持续卡点是 validation-loss AUC，且对 seed 很敏感。
3. Fashion 的 AUC 不是简单 early coeff boost 能解决；30 epochs 可转正但不到 5%。
4. KMNIST 已接近 5-seed weak gate，但 best 仍差 0.44 个百分点 AUC。
5. 继续只调 branch_boost/final_scale/coeff_lr 的收益变小。
```

下一步建议：

```text
短线：
  1. 对 KMNIST best 做 c0.12/c0.14/c0.16, rest0.003, final0.8 的窄扫，看能否把 AUC 从 4.56% 推过 5%。
  2. 对 Fashion best 加 LR decay / rest optimizer schedule，而不是继续提高 early coeff boost。

中线：
  1. 需要显式 scheduler 支持；当前代码只有固定 lr 和 early branch/coeff multiplier。
  2. AUC gate 可能要求 rest/head 与 functional coeff 的协同调度，否则 GA-FU 后段好、前段慢的问题会持续。
```

## 2026-05-01 GA-FU 继续调参：KMNIST 窄扫 + Fashion LR schedule

本轮根据上一节结论继续推进：

```text
KMNIST:
  在 best 附近窄扫 c0.12/c0.14/c0.16, rest0.003, final0.8。

Fashion:
  不继续加大 early coeff boost，改加显式 lr schedule / rest schedule。
```

代码变更：

```text
experiments/dgkan_core.py
  TrainConfig 新增：
    lr_schedule / lr_final_mult
    rest_lr_schedule / rest_lr_final_mult
    coeff_lr_schedule / coeff_lr_decay_final_mult
    lr_decay_start_frac

  新增 scheduled_lr_factor() 和 set_optimizer_lr()。
  AdamW optimizer、rest AdamW optimizer、functional coeff step 均可独立使用 none/linear/cosine decay。
  结果 CSV 记录 lr/rest/coeff schedule 与 final factor。

experiments/run_gafu_consolidation.py
  新增 KMFINE：
    GA-FU-KM-fine-b1.2-c0.12-r0.003-f0.8-g0
    GA-FU-KM-fine-b1.2-c0.14-r0.003-f0.8-g0
    GA-FU-KM-fine-b1.2-c0.16-r0.003-f0.8-g0

  新增 FASHSCHED：
    rest cosine decay
    coeff cosine decay
    rest+coeff cosine decay
    rest linear decay
```

检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_consolidation.py
```

### KMFINE：KMNIST h96/b24, 20 epochs, seed 0-4

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages KMFINE \
  --datasets KMNIST \
  --seeds 0,1,2,3,4 \
  --out-dir results/gafu_consolidation_captune_h96_b24_seed012_e20 \
  --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak | medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| AdamW | 5 | 0.7688 ± 0.0221 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | no | no |
| GA-FU-KM-fine-b1.2-c0.12-r0.003-f0.8-g0 | 5 | 0.7794 ± 0.0180 | -0.0106 | 0.0590 | 0.260 | 0.893 | 0.638 | 0.161 | yes | no |
| GA-FU-KM-fine-b1.2-c0.14-r0.003-f0.8-g0 | 5 | 0.7824 ± 0.0222 | -0.0136 | 0.0700 | 0.254 | 0.939 | 0.651 | 0.181 | yes | no |
| GA-FU-KM-fine-b1.2-c0.16-r0.003-f0.8-g0 | 5 | 0.7756 ± 0.0185 | -0.0068 | 0.0489 | 0.246 | 0.851 | 0.663 | 0.129 | no | no |

结论：

```text
KMNIST 已获得 5-seed weak pass。
最佳为 c0.14：
  acc 比 AdamW 高 1.36 points
  AUC improvement +7.00%
  phi/J/branch/ECE 全部过 weak

没有 medium pass：
  KMNIST medium 需要 AUC > 8%，当前 best 为 7.00%。
```

### FASHSCHED：Fashion h96/b24, 30 epochs, seed 0/1/2

命令：

```bash
python experiments/run_gafu_consolidation.py \
  --packages FASHSCHED \
  --datasets Fashion-MNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_consolidation_fashsched_h96_b24_seed012_e30 \
  --fresh --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

结果：

| method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak | medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| AdamW | 3 | 0.8403 ± 0.0186 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | no | no |
| GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restcos0.3 | 3 | 0.8493 ± 0.0129 | -0.0090 | 0.0567 | 0.330 | 0.956 | 0.640 | 0.095 | yes | no |
| GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-bothcos | 3 | 0.8477 ± 0.0133 | -0.0073 | 0.0593 | 0.335 | 0.808 | 0.642 | 0.085 | yes | no |
| GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-coeffcos0.5 | 3 | 0.8463 ± 0.0189 | -0.0060 | 0.0272 | 0.332 | 0.876 | 0.635 | 0.088 | no | no |
| GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restlin0.3 | 3 | 0.8400 ± 0.0199 | 0.0003 | 0.0462 | 0.329 | 0.806 | 0.637 | 0.008 | no | no |

结论：

```text
Fashion 已获得 3-seed weak pass。

推荐候选：
  GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restcos0.3
  acc 比 AdamW 高 0.90 points
  AUC improvement +5.67%
  phi red 33.0%, J red 95.6%, branch/A 0.640, ECE red 9.5%

AUC 最高候选：
  bothcos
  AUC improvement +5.93%
  但 J red/ECE red 比 restcos0.3 略弱。

单独 coeff cosine 不够：
  AUC only +2.72%

rest linear 也不够：
  AUC +4.62%, ECE red only 0.8%

所以 Fashion 的关键不是衰减 functional coeff，而是对 rest/head AdamW 使用 cosine decay。
```

补 seed3/4 确认：

```bash
python experiments/run_gafu_consolidation.py \
  --packages FASHCONFIRM \
  --datasets Fashion-MNIST \
  --seeds 3,4 \
  --out-dir results/gafu_consolidation_fashsched_h96_b24_seed012_e30 \
  --device auto --no-download \
  --epochs 30 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error
```

5-seed 确认结果：

| method | runs | acc mean ± std | gap | AUC improve | phi red | J red | branch/AdamW | ECE red | weak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AdamW | 5 | 0.8368 ± 0.0141 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 | no |
| GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-bothcos | 5 | 0.8444 ± 0.0104 | -0.0076 | 0.0570 | 0.330 | 0.840 | 0.646 | 0.104 | yes |
| GA-FU-FM-sched-b1.2-c0.05-r0.001-f0.8-restcos0.3 | 5 | 0.8494 ± 0.0091 | -0.0126 | 0.0473 | 0.326 | 0.943 | 0.645 | 0.137 | no |

补充判断：

```text
Fashion 5-seed weak pass 站住的是 bothcos。
restcos0.3 的 acc/J/ECE 更好，但 AUC 只有 +4.73%，低于 weak gate 的 +5%。
因此如果以 gate 为准，Fashion 推荐 bothcos；如果以 Pareto 稳定性看，restcos0.3 仍是近邻备选。
```

### 本轮总判断

```text
Weak gate:
  KMNIST: yes, 5 seeds
  Fashion: yes, 5 seeds

Medium gate:
  KMNIST: no, AUC 7.00% < 8%
  Fashion: no, AUC 5.70% < 8%

Strong gate:
  no
```

当前最推荐 Stage-I GA-FU profile：

```text
Shared architecture:
  hidden_dim=96
  basis_count=24
  depth=4
  alpha_init=1.5

KMNIST:
  coeff_lr=0.14
  rest_lr=0.003
  branch_boost=1.2
  coeff_lr_boost=1.0
  branch_final_scale=0.8
  geometry_min_epochs=0
  epochs=20

Fashion:
  coeff_lr=0.05
  rest_lr=0.001
  branch_boost=1.2
  coeff_lr_boost=1.0
  branch_final_scale=0.8
  geometry_min_epochs=0
  rest_lr_schedule=cosine
  rest_lr_final_mult=0.3
  coeff_lr_schedule=cosine
  coeff_lr_decay_final_mult=0.5
  epochs=30
```

注意：

```text
runner 的 aggregate_summary 仍会打印 GA-FU weak=false，因为 decision_report 只看默认 label "GA-FU"。
本节 weak/medium 是对具体 tuned label 按同一 gate 重新计算得到的。
```

## 2026-05-01 DG-KAN Optimizer v3：Phase-Aligned GA-FU 初始实验

依据文档：

```text
docs/DG-KAN_Optimizer_v3_完整计划.md
```

本轮目标：开始执行 GA-FU-v3 计划，优先完成文档推荐顺序中的 E0 smoke、E2 phase alignment，并做 E3 true Gram 的 seed0 探针。结果已通过 wandb 上传到项目 `DG-KAN`，对应 group：

```text
gafu-v3-e0-smoke-v3
gafu-v3-e2-phase-seed012-e20
gafu-v3-e3-metric-seed0-e20
```

### 代码变更

新增：

```text
experiments/run_gafu_v3.py
```

支持 package：

```text
V3_E0
V3_E2
V3_E3
V3_E4
```

核心实现：

```text
experiments/dgkan_core.py
  build_rbf_sobolev_gram()
  basis_diag_gram / full_sobolev_gram / diag_to_full_sobolev
  GA-FU-v3 phase state: ACTIVE -> TRANSITION -> GEOMETRY
  smooth transition for branch scale / coeff lr / metric mix
  trust update metric norm accounting
  preconditioner build/solve timing
  v3 metric condition/eigen logging
```

依赖：

```text
requirement.txt
  added wandb>=0.16

.gitignore
  added wandb/
```

实现中发现一个 E0 级调度细节：文档的 1-epoch smoke 在 `train_size=512, batch_size=256` 时总共只有 2 个 step。如果 transition 起点不计入当前 step，smooth transition 永远停在 mix=0，导致 smoke 结果被人为拖差。已修复为：

```text
1. 极短 run 的 transition_length 截到可完成范围。
2. 当前 switch step 计入 transition 第一步。
```

检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_v3.py
```

通过。

### V3-E0 smoke

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E0 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_e0_smoke_v3 \
  --fresh --device auto --no-download \
  --epochs 1 --train-size 512 --val-size 128 --test-size 128 \
  --hidden-dim 32 --depth 2 --basis-count 8 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

结果：

| dataset | method | acc | val AUC | final phase | metric cond active | metric cond geometry | clip |
|---|---|---:|---:|---|---:|---:|---:|
| Fashion | AdamW | 0.5000 | 1.8792 | ACTIVE | - | - | 0.000 |
| Fashion | GA-FU-v2-current | 0.4453 | 1.9115 | GEOMETRY | - | - | 0.000 |
| Fashion | GA-FU-v3-smoke | 0.4141 | 1.9264 | GEOMETRY | 6911.5 | 6551.3 | 0.000 |
| KMNIST | AdamW | 0.2344 | 2.0532 | ACTIVE | - | - | 0.000 |
| KMNIST | GA-FU-v2-current | 0.2188 | 1.8835 | GEOMETRY | - | - | 0.000 |
| KMNIST | GA-FU-v3-smoke | 0.2734 | 1.8644 | GEOMETRY | 6911.5 | 6551.3 | 0.000 |

E0 判断：

```text
run failures = 0
nan/inf = 0
Gram condition = 6.5e3 to 6.9e3, below 1e4
Fashion v3-smoke gap vs v2-current = 3.1 points, below 5 point smoke threshold
KMNIST v3-smoke beats v2-current
```

结论：E0 smoke 通过。true Gram 数值可跑，trust clip 没有实际参与。

### V3-E2 phase alignment：seed 0/1/2, 20 epochs

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E2 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --out-dir results/gafu_v3_phase_alignment_seed012_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

Fashion-MNIST：

| method | runs | acc mean ± std | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 3 | 0.8390 ± 0.0179 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.159 |
| GA-FU-v2-tuned | 3 | 0.8453 ± 0.0141 | 0.0302 | 0.272 | 0.941 | 0.651 | 0.271 |
| V3-phase-aligned-hard | 3 | 0.8437 ± 0.0102 | 0.0270 | 0.272 | 0.948 | 0.639 | 0.279 |
| V3-phase-aligned-smooth | 3 | 0.8410 ± 0.0153 | 0.0049 | 0.271 | 0.802 | 0.654 | 0.273 |

KMNIST：

| method | runs | acc mean ± std | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 3 | 0.7747 ± 0.0172 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.001 |
| GA-FU-v2-tuned | 3 | 0.7780 ± 0.0178 | 0.0836 | 0.253 | 0.945 | 0.650 | 0.142 |
| V3-phase-aligned-hard | 3 | 0.7817 ± 0.0147 | 0.0497 | 0.250 | 0.950 | 0.639 | 0.164 |
| V3-phase-aligned-smooth | 3 | 0.7777 ± 0.0074 | 0.0324 | 0.252 | 0.850 | 0.658 | 0.111 |

E2 判断：

```text
phase alignment alone did not beat v2-tuned on AUC.
hard alignment is close to v2 and slightly improves KMNIST accuracy.
smooth alignment hurts AUC on both datasets under legacy diagonal metric.
```

结论：E2 没有确认“只做 phase alignment 就能修复 AUC”。如果继续 v3，不能只靠 smooth transition；需要进入 true Gram / metric direction。

### V3-E3 true Gram seed0 probe

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_E3 \
  --datasets Fashion-MNIST,KMNIST \
  --seeds 0 \
  --out-dir results/gafu_v3_metric_seed0_e20 \
  --fresh --device auto --no-download \
  --epochs 20 --train-size 6000 --val-size 1000 --test-size 1000 \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 256 \
  --continue-on-error --wandb --wandb-project DG-KAN
```

Fashion-MNIST seed0：

| method | acc | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|
| GA-FU-v2-tuned | 0.833 | 0.0725 | 0.271 | 0.894 | 0.646 | 0.191 |
| V3-diag-gram | 0.837 | 0.0562 | 0.269 | 0.989 | 0.694 | 0.273 |
| V3-full-from-start | 0.825 | 0.0761 | 0.280 | 0.932 | 0.612 | 0.226 |
| V3-diag-to-full-hard | 0.828 | 0.1011 | 0.275 | 0.949 | 0.674 | 0.273 |
| V3-diag-to-full-smooth | 0.823 | 0.0959 | 0.272 | -0.937 | 0.690 | 0.172 |
| V3-softgeo | 0.831 | 0.0563 | 0.264 | 0.971 | 0.689 | 0.110 |

KMNIST seed0：

| method | acc | AUC improve | phi red | J red | branch/AdamW | ECE red |
|---|---:|---:|---:|---:|---:|---:|
| GA-FU-v2-tuned | 0.793 | 0.0385 | 0.264 | 0.933 | 0.665 | 0.224 |
| V3-diag-gram | 0.769 | 0.0687 | 0.246 | 0.704 | 0.747 | 0.082 |
| V3-full-from-start | 0.792 | 0.1085 | 0.293 | 0.670 | 0.613 | 0.260 |
| V3-diag-to-full-hard | 0.773 | 0.0243 | 0.274 | 0.604 | 0.704 | 0.160 |
| V3-diag-to-full-smooth | 0.789 | 0.0773 | 0.254 | 0.785 | 0.735 | 0.217 |
| V3-softgeo | 0.793 | 0.0625 | 0.237 | 0.497 | 0.731 | 0.208 |

E3 seed0 判断：

```text
runner seed0 pass-level:
  V3-full-from-start = weak by min dataset level

Fashion:
  diag-to-full-hard has the best AUC improvement, but accuracy drops vs v2.
  diag-gram improves accuracy/J/ECE, but AUC is weaker than v2.

KMNIST:
  full-from-start is the best signal:
    acc 0.792 vs v2 0.793
    AUC improvement 10.85% vs v2 3.85%
    phi reduction 29.3%
    ECE reduction 26.0%
```

本轮总判断：

```text
1. E0 smoke 通过。
2. E2 phase alignment alone 不够，smooth transition 不是单独解法。
3. E3 的 true Gram 有真实信号，尤其 KMNIST full-from-start。
4. Fashion 的 full/diag-to-full 倾向提升 AUC 但牺牲 acc；需要 schedule 或 branch/coeff profile 补表达力。
5. 当前最值得扩 seed 的不是默认 diag-to-full-smooth，而是：
   KMNIST: V3-full-from-start
   Fashion: V3-diag-to-full-hard 与 V3-full-from-start 二选一/并行 seed012
```

下一步建议：

```text
短线：
  1. 扩 V3-full-from-start 到 seed0/1/2，优先 KMNIST。
  2. Fashion 对 V3-diag-to-full-hard 做 acc rescue：rest/coeff bothcos 或稍高 final branch/coeff.
  3. 跑 E4 schedule ablation 时，不应只继承 diag-to-full-smooth；应把 full-from-start 作为单独候选。

注意：
  seed0 E3 不能当正式结论，只是扩 seed 优先级信号。
```

## 2026-05-01 DG-KAN Optimizer v3.1：transition 语义修复、true-Gram 复验、固定 alpha 对照

依据文档：

```text
docs/DG-KAN_Optimizer_v3.1_下一步详细实验计划.md
```

本轮目标：

```text
1. 修复 diag_to_full_sobolev 的 transition 语义：
   旧语义: diag-fast -> full-fast
   新语义: diag-fast -> full-geo

2. 增加最小方向审计：
   diag-fast direction vs full-geo direction cosine
   norm ratio / metric norm
   diag-fast / full-geo Gram condition/eig stats

3. 先跑 learnable-alpha 版本。

4. 按要求把 ResidualKANBlock.alpha 改为固定值 1 且不可学习后，再跑同一组实验。
```

代码变更：

```text
experiments/dgkan_core.py
  RuntimeState 新增 v3 diagfast/fullgeo Gram 与 direction-audit 统计。
  functional_coeff_step 改为走 _v3_precondition_direction。
  diag_to_full_sobolev transition 现在混合 diag-fast direction 与 full-geo direction。
  result row 新增 v3_direction_* / v3_diagfast_* / v3_fullgeo_* 字段。
  Train output 继续记录 v3_metric_* 与 val_loss_curve。

experiments/run_gafu_v3.py
  新增 V3_R0_CHECKS。
  新增 V3_R1_SMOKE_FIXED。
  新增 V3_R3_E3_FIXED。
  新增 V3_R4_KMNIST_FULL_EXPAND。
  新增 V3_R5_FASHION_RESCUE。
  增加断点续跑 skip：按 (package,dataset,method,seed) 跳过已完成行，防止中断后重复追加。

experiments/dgkan_core.py
  按要求后半轮将：
    self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
  改为：
    self.register_buffer("alpha", torch.tensor(1.0))
```

检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_v3.py
```

通过。

W&B：

```text
本轮所有正式训练命令均使用 --wandb 上传 run / artifact。
group:
  gafu-v3-31-r1-learnable-alpha
  gafu-v3-31-r3-kmnist-learnable-alpha
  gafu-v3-31-r3-fashion-learnable-alpha
  gafu-v3-31-r4-kmnist-learnable-alpha
  gafu-v3-31-r5-fashion-learnable-alpha
  gafu-v3-31-r1-fixed-alpha
  gafu-v3-31-r3-kmnist-fixed-alpha
  gafu-v3-31-r3-fashion-fixed-alpha
  gafu-v3-31-r4-kmnist-fixed-alpha
  gafu-v3-31-r5-fashion-fixed-alpha
```

### R0 deterministic checks

learnable-alpha：

```text
results/gafu_v3_31_r0_checks_learnable_alpha
pass = True
rel0 = 0.000e+00
rel1 = 0.000e+00
cond_fast = 6906.37
cond_geo = 6545.16
mix_first = 0.0010
```

fixed-alpha：

```text
results/gafu_v3_31_r0_checks_fixed_alpha
pass = True
rel0 = 0.000e+00
rel1 = 0.000e+00
cond_fast = 6906.37
cond_geo = 6545.16
mix_first = 0.0010
```

判断：

```text
transition 端点语义正确：
  mix=0 完全等于 diag-fast direction
  mix=1 完全等于 full-geo direction
```

### R1 smoke

learnable-alpha：

| dataset | method | acc | val AUC | condA | condG | clip |
|---|---|---:|---:|---:|---:|---:|
| Fashion | AdamW | 0.5000 | 1.8792 | - | - | 0 |
| Fashion | GA-FU-v2-current | 0.4453 | 1.9115 | - | - | 0 |
| Fashion | GA-FU-v3-smoke-fixed | 0.4141 | 1.9264 | 6911.5 | 6551.3 | 0 |
| KMNIST | AdamW | 0.2344 | 2.0532 | - | - | 0 |
| KMNIST | GA-FU-v2-current | 0.2188 | 1.8835 | - | - | 0 |
| KMNIST | GA-FU-v3-smoke-fixed | 0.2734 | 1.8644 | 6911.5 | 6551.3 | 0 |

fixed-alpha：

| dataset | method | acc | val AUC | condA | condG | clip |
|---|---|---:|---:|---:|---:|---:|
| Fashion | AdamW | 0.4609 | 1.8082 | - | - | 0 |
| Fashion | GA-FU-v2-current | 0.4766 | 1.8482 | - | - | 0 |
| Fashion | GA-FU-v3-smoke-fixed | 0.4844 | 1.8584 | 6911.5 | 6551.3 | 0 |
| KMNIST | AdamW | 0.2422 | 2.0181 | - | - | 0 |
| KMNIST | GA-FU-v2-current | 0.2656 | 1.8148 | - | - | 0 |
| KMNIST | GA-FU-v3-smoke-fixed | 0.2734 | 1.8083 | 6911.5 | 6551.3 | 0 |

判断：

```text
R1 smoke 两个 alpha 版本均无数值异常、无 trust clip。
fixed-alpha 下 1-epoch smoke accuracy 更稳定，但这不是正式结论。
```

### R3 seed0：修复后 E3 复查

learnable-alpha, KMNIST, 20 epochs：

| method | acc | AUC improve | val AUC |
|---|---:|---:|---:|
| AdamW | 0.7720 | 0.0000 | 0.5396 |
| GA-FU-v2-tuned | 0.7930 | 0.0385 | 0.5189 |
| V3-diag-gram | 0.7690 | 0.0687 | 0.5026 |
| V3-full-from-start | 0.7940 | 0.1463 | 0.4607 |
| V3-diag-to-full-hard | 0.7730 | 0.0243 | 0.5265 |
| V3-diag-to-full-smooth-fixed | 0.7760 | 0.0126 | 0.5329 |
| V3-diag-to-full-smooth-fixed-longtr | 0.7800 | 0.0635 | 0.5053 |

learnable-alpha, Fashion, 30 epochs：

| method | acc | AUC improve | val AUC |
|---|---:|---:|---:|
| AdamW | 0.8250 | 0.0000 | 0.6204 |
| GA-FU-v2-tuned | 0.8410 | 0.1174 | 0.5475 |
| V3-diag-gram | 0.8430 | 0.0936 | 0.5623 |
| V3-full-from-start | 0.8390 | 0.1239 | 0.5435 |
| V3-diag-to-full-hard | 0.8470 | 0.1381 | 0.5347 |
| V3-diag-to-full-smooth-fixed | 0.8400 | 0.1358 | 0.5361 |
| V3-diag-to-full-smooth-fixed-longtr | 0.8400 | 0.1173 | 0.5476 |

fixed-alpha, KMNIST, 20 epochs：

| method | acc | AUC improve | val AUC |
|---|---:|---:|---:|
| AdamW | 0.7920 | 0.0000 | 0.5260 |
| GA-FU-v2-tuned | 0.7920 | 0.0390 | 0.5055 |
| V3-diag-gram | 0.7950 | 0.1379 | 0.4535 |
| V3-full-from-start | 0.8020 | 0.1098 | 0.4683 |
| V3-diag-to-full-hard | 0.7970 | 0.0711 | 0.4886 |
| V3-diag-to-full-smooth-fixed | 0.7850 | 0.0545 | 0.4973 |
| V3-diag-to-full-smooth-fixed-longtr | 0.7900 | 0.1443 | 0.4501 |

fixed-alpha, Fashion, 30 epochs：

| method | acc | AUC improve | val AUC |
|---|---:|---:|---:|
| AdamW | 0.8310 | 0.0000 | 0.5882 |
| GA-FU-v2-tuned | 0.8260 | 0.0476 | 0.5602 |
| V3-diag-gram | 0.8420 | 0.0871 | 0.5370 |
| V3-full-from-start | 0.8380 | 0.1134 | 0.5215 |
| V3-diag-to-full-hard | 0.8370 | 0.0997 | 0.5296 |
| V3-diag-to-full-smooth-fixed | 0.8330 | 0.1198 | 0.5178 |
| V3-diag-to-full-smooth-fixed-longtr | 0.8480 | 0.1118 | 0.5224 |

R3 判断：

```text
learnable-alpha:
  KMNIST: full-from-start 是最强 seed0 信号，acc 0.7940，AUC improve 14.63%。
  Fashion: diag-to-full-hard 同时拿到最高 acc 0.8470 与最高 AUC improve 13.81%。

fixed-alpha:
  KMNIST: full-from-start accuracy 最高，0.8020；long transition AUC 最强但 acc 稍弱。
  Fashion: smooth-fixed-longtr accuracy 最高，0.8480；smooth/full 类 AUC 均明显强于 AdamW/v2。

alpha 固定后，seed0 表现不是简单退化；KMNIST accuracy 反而更高，但需要扩 seed 看稳定性。
```

### R4 KMNIST full-from-start expansion：3 seeds, 20 epochs

learnable-alpha：

| method | acc mean ± std | gap | AUC improve | phi red | J red | branch/A | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.7747 ± 0.0172 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.001 |
| GA-FU-v2-tuned | 0.7780 ± 0.0178 | -0.0033 | 0.0836 | 0.253 | 0.945 | 0.650 | 0.142 |
| K-V3-FULL-base | 0.7790 ± 0.0106 | -0.0043 | 0.0922 | 0.284 | 0.952 | 0.596 | 0.221 |
| K-V3-FULL-restcos07 | 0.7803 ± 0.0135 | -0.0057 | 0.0933 | 0.284 | 0.938 | 0.595 | 0.238 |
| K-V3-FULL-bothcos07 | 0.7787 ± 0.0052 | -0.0040 | 0.1080 | 0.285 | 0.983 | 0.597 | 0.208 |

fixed-alpha：

| method | acc mean ± std | gap | AUC improve | phi red | J red | branch/A | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.7877 ± 0.0048 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | -0.070 |
| GA-FU-v2-tuned | 0.7840 ± 0.0135 | 0.0037 | 0.0565 | 0.256 | 1.000 | 0.621 | 0.038 |
| K-V3-FULL-base | 0.7853 ± 0.0170 | 0.0023 | 0.0997 | 0.281 | 1.000 | 0.577 | 0.120 |
| K-V3-FULL-restcos07 | 0.7777 ± 0.0193 | 0.0100 | 0.0929 | 0.281 | 1.000 | 0.575 | 0.079 |
| K-V3-FULL-bothcos07 | 0.7720 ± 0.0205 | 0.0157 | 0.0961 | 0.282 | 1.000 | 0.579 | 0.077 |

R4 判断：

```text
learnable-alpha:
  KMNIST full-from-start 3-seed 可稳定超过 v2 的 AUC/J/ECE；
  accuracy 只小幅高于 v2，最优 restcos07 = 0.7803 vs v2 0.7780。

fixed-alpha:
  AdamW baseline 本身更高且更稳，导致 full-Gram accuracy gap 为正；
  K-V3-FULL-base 仍有强 AUC improve 9.97%、phi/J/branch 也好，
  但 acc 0.7853 < AdamW 0.7877，未形成 clean replacement。

结论：
  KMNIST true full-Gram 是 AUC/geometry 有效组件；
  learnable-alpha 更适合当前 R4 full-Gram candidate；
  fixed-alpha 需要另调 coeff/rest 或 transition，不能直接沿用 learnable-alpha profile。
```

### R5 Fashion rescue：3 seeds, 30 epochs

learnable-alpha：

| method | acc mean ± std | gap | AUC improve | phi red | J red | branch/A | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8403 ± 0.0152 | 0.0000 | 0.0000 | -0.000 | 0.000 | 1.000 | 0.091 |
| GA-FU-v2-tuned-bothcos | 0.8477 ± 0.0109 | -0.0073 | 0.0593 | 0.335 | 0.808 | 0.642 | 0.168 |
| F-V3-HARD-base30 | 0.8487 ± 0.0135 | -0.0083 | 0.1033 | 0.342 | 0.931 | 0.665 | 0.244 |
| F-V3-HARD-f085 | 0.8493 ± 0.0107 | -0.0090 | 0.1125 | 0.343 | 0.183 | 0.684 | 0.255 |
| F-V3-HARD-c006-softcoeff | 0.8460 ± 0.0177 | -0.0057 | 0.0772 | 0.340 | 0.251 | 0.689 | 0.185 |
| F-V3-FULL-f085 | 0.8437 ± 0.0082 | -0.0033 | 0.0710 | 0.347 | 0.906 | 0.610 | 0.199 |

fixed-alpha：

| method | acc mean ± std | gap | AUC improve | phi red | J red | branch/A | ECE red |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.8330 ± 0.0022 | 0.0000 | 0.0000 | -0.000 | 0.000 | 1.000 | 0.029 |
| GA-FU-v2-tuned-bothcos | 0.8473 ± 0.0180 | -0.0143 | 0.0286 | 0.350 | 1.000 | 0.637 | 0.188 |
| F-V3-HARD-base30 | 0.8473 ± 0.0126 | -0.0143 | 0.1077 | 0.357 | 1.000 | 0.665 | 0.220 |
| F-V3-HARD-f085 | 0.8443 ± 0.0083 | -0.0113 | 0.0934 | 0.357 | 1.000 | 0.673 | 0.204 |
| F-V3-HARD-c006-softcoeff | 0.8460 ± 0.0150 | -0.0130 | 0.1270 | 0.355 | 1.000 | 0.686 | 0.215 |
| F-V3-FULL-f085 | 0.8443 ± 0.0078 | -0.0113 | 0.0957 | 0.360 | 1.000 | 0.624 | 0.181 |

R5 判断：

```text
learnable-alpha:
  Fashion true-Gram rescue 明确成立。
  best accuracy: F-V3-HARD-f085 = 0.8493
  best Pareto/J: F-V3-HARD-base30 = acc 0.8487, AUC improve 10.33%, J red 93.1%, ECE red 24.4%
  compared to v2 bothcos:
    v2 AUC improve only 5.93%
    hard-base/f085 AUC improve > 10%

fixed-alpha:
  Fashion 仍然成立，而且 AUC 更强。
  best AUC: F-V3-HARD-c006-softcoeff = AUC improve 12.70%, acc 0.8460
  best acc/J balance: F-V3-HARD-base30 = acc 0.8473, AUC improve 10.77%, J red ~99.98%
  v2 bothcos 在 fixed-alpha 下 acc 也高，但 AUC improve 只有 2.86%，不如 true-Gram。

结论：
  Fashion 是 v3.1 true-Gram 最清晰的成功点。
  fixed-alpha 不会破坏 Fashion 的 true-Gram 结论，反而让 J reduction 更干净。
```

### 本轮总判断

```text
1. v3.1 transition 修复是必要的：
   diag_to_full_sobolev 现在才真正表示 diag-fast -> full-geo。

2. R0/R1 通过：
   transition endpoint 正确；
   smoke 无 NaN、无 trust clip。

3. KMNIST:
   learnable-alpha full-Gram 3-seed 比 v2 有更强 AUC/J/ECE，accuracy 小幅更高；
   fixed-alpha 下 AdamW baseline 变强，full-Gram AUC 仍好但 accuracy 未过 AdamW。

4. Fashion:
   learnable-alpha 与 fixed-alpha 都支持 true-Gram rescue。
   最稳定推荐仍是 hard diag-to-full，而不是 full-from-start。

5. alpha 固定为 1 后：
   seed0 有些结果更好，但 3-seed KMNIST accuracy 不如 learnable-alpha；
   Fashion 仍然强，尤其 AUC/J；
   因此不能简单 claim 固定 alpha 全面更好。

6. 当前推荐：
   如果继续 v3.1:
     Fashion: 继续扩 F-V3-HARD-base30 / F-V3-HARD-c006-softcoeff 到 5 seeds。
     KMNIST: learnable-alpha 下 K-V3-FULL-restcos07 / bothcos07 可扩 5 seeds；
             fixed-alpha 需要重新扫 coeff_lr / final_scale，不建议直接 confirm。
```

当前代码状态：

```text
ResidualKANBlock.alpha 已按要求固定为不可学习的 1.0。
run_gafu_v3.py 已支持中断续跑 skip。
```

## 2026-05-01 DG-KAN Optimizer v3.2：alpha_mode 显式化、5/10-seed confirm 与机制审计

依据文档：

```text
docs/DG-KAN_Optimizer_v3.2_下一步详细实验计划.md
```

本轮目标：

```text
1. 不再手改 ResidualKANBlock.alpha，改为显式 alpha_mode。
2. 跑 P0/P1/P2/P3 必要实验。
3. 根据 P2/P3 判断是否触发 P5。
4. 生成 P4/P6/P7/P8 分析与 v3.2 结果复盘。
```

代码变更：

```text
experiments/dgkan_core.py
  TrainConfig 新增 optimizer_method / alpha_mode。
  ResidualKANBlock 支持 alpha_mode:
    learnable: alpha 是 nn.Parameter(alpha_init)
    fixed1: alpha 是 buffer 1.0
    fixed_init: alpha 是 buffer alpha_init
  DGKANClassifier 传入 alpha_mode。
  train_one 使用 optimizer_method 决定底层优化逻辑，method 只作为显示 label。
  每个 row 记录：
    alpha_mode / alpha_trainable / alpha_final_mean/std / alpha_layer_k_final
    no_kan_val/test acc/loss
    no_kan_loss_increase
    kan_logit_delta_norm_mean/p95
    kan_margin_contribution_mean/p95
    val_acc_curve / epoch_time_sec_curve
    transition_loss_jump / transition_acc_jump
    time_ratio_vs_adamw / no_kan_drop_over_adamw

experiments/run_gafu_v3.py
  新增 packages:
    V3_2_P0_SMOKE
    V3_2_P1_SANITY
    V3_2_P2_FASHION_CONFIRM
    V3_2_P3_KMNIST_CONFIRM
    V3_2_P5_FINAL_FIXED
  续跑 skip key 增加 alpha_mode。
  summary_by_method 按 alpha_mode 分组。
  每个 run 输出：
    runs/{run}/curves.csv
    runs/{run}/direction_audit.csv
    runs/{run}/branch_audit.csv
  failure_table 增加非 error failure tags。

experiments/analyze_gafu_v32.py
  生成 P4/P6/P7/P8 分析：
    results/gafu_v3_2_p4_alpha_decision/
    results/gafu_v3_2_p6_mechanism/
    results/gafu_v3_2_p7_direction_audit/
    results/gafu_v3_2_p8_target_matched/
  生成：
    docs/DG-KAN_Optimizer_v3.2_结果复盘.md
```

检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_v3.py experiments/analyze_gafu_v32.py
```

通过。

### P0：R0 + alpha-mode smoke

R0 endpoint：

```text
results/gafu_v3_2_p0_r0_checks
pass = True
rel0 = 0.000e+00
rel1 = 0.000e+00
cond_fast = 6906.37
cond_geo = 6545.16
mix_first = 0.0010
```

P0 smoke：

```text
results/gafu_v3_2_p0_smoke
runs = 12
failures = 0
trust_clip_rate = 0
```

P0 复现了 v3.1 的 learnable/fixed smoke 数值，说明 alpha_mode 切换没有混淆 method label 或 optimizer path。

### P1：seed0 sanity

关键复现点：

```text
Fashion learnable:
  AdamW-alphaLearn = 0.8250 / AUC 0.6204
  GA-FU-v2-bothcos-alphaLearn = 0.8410 / AUC 0.5475
  F-V3-HARD-base30-alphaLearn = 0.8470 / AUC 0.5347

Fashion fixed1:
  AdamW-alphaFixed1 = 0.8310 / AUC 0.5882
  F-V3-HARD-c006-softcoeff-alphaFixed1 = 0.8420 / AUC 0.5205

KMNIST learnable:
  AdamW-alphaLearn = 0.7720 / AUC 0.5396
  K-V3-FULL-restcos07-alphaLearn = 0.7970 / AUC 0.4559

KMNIST fixed1:
  AdamW-alphaFixed1 = 0.7920 / AUC 0.5260
  K-V3-FULL-base-alphaFixed1 = 0.8020 / AUC 0.4683
```

判断：

```text
P1 与 v3.1 同 alpha_mode 结果一致。
fixed1 继续显示 KMNIST AdamW baseline 更强的现象。
```

### P2：Fashion 5-seed confirm

结果目录：

```text
results/gafu_v3_2_p2_fashion_confirm
```

| alpha | method | runs | acc mean ± std | gap | AUC imp | phi red | J red | branch/A | ECE red |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learnable | AdamW-alphaLearn | 5 | 0.8368 ± 0.0126 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 |
| learnable | GA-FU-v2-bothcos-alphaLearn | 5 | 0.8444 ± 0.0093 | -0.0076 | 0.0570 | 0.330 | 0.840 | 0.646 | 0.104 |
| learnable | F-V3-HARD-base30-alphaLearn | 5 | 0.8482 ± 0.0109 | -0.0114 | 0.1122 | 0.337 | 0.851 | 0.672 | 0.198 |
| fixed1 | AdamW-alphaFixed1 | 5 | 0.8392 ± 0.0095 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 |
| fixed1 | GA-FU-v2-bothcos-alphaFixed1 | 5 | 0.8480 ± 0.0153 | -0.0088 | 0.0401 | 0.350 | 1.000 | 0.639 | 0.158 |
| fixed1 | F-V3-HARD-base30-alphaFixed1 | 5 | 0.8474 ± 0.0101 | -0.0082 | 0.1060 | 0.356 | 1.000 | 0.659 | 0.206 |
| fixed1 | F-V3-HARD-c006-softcoeff-alphaFixed1 | 5 | 0.8434 ± 0.0122 | -0.0042 | 0.0946 | 0.355 | 1.000 | 0.684 | 0.168 |

P2 判断：

```text
Fashion 5-seed confirm 通过。
F-V3-HARD-base30-alphaLearn 和 alphaFixed1 都强。
learnable hard-base AUC/acc 综合最好；
fixed1 hard-base J reduction 更干净。
```

### P3：KMNIST 5-seed confirm

结果目录：

```text
results/gafu_v3_2_p3_kmnist_confirm
```

| alpha | method | runs | acc mean ± std | gap | AUC imp | phi red | J red | branch/A | ECE red |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learnable | AdamW-alphaLearn | 5 | 0.7688 ± 0.0198 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 |
| learnable | GA-FU-v2-tuned-alphaLearn | 5 | 0.7824 ± 0.0199 | -0.0136 | 0.0700 | 0.254 | 0.939 | 0.651 | 0.181 |
| learnable | K-V3-FULL-restcos07-alphaLearn | 5 | 0.7744 ± 0.0136 | -0.0056 | 0.0787 | 0.285 | 0.944 | 0.595 | 0.217 |
| learnable | K-V3-FULL-bothcos07-alphaLearn | 5 | 0.7754 ± 0.0063 | -0.0066 | 0.0872 | 0.286 | 0.909 | 0.598 | 0.210 |
| fixed1 | AdamW-alphaFixed1 | 5 | 0.7792 ± 0.0180 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 |
| fixed1 | GA-FU-v2-tuned-alphaFixed1 | 5 | 0.7812 ± 0.0154 | -0.0020 | 0.0605 | 0.252 | 1.000 | 0.620 | 0.118 |
| fixed1 | K-V3-FULL-base-alphaFixed1 | 5 | 0.7826 ± 0.0151 | -0.0034 | 0.0906 | 0.277 | 1.000 | 0.578 | 0.186 |

P3 判断：

```text
KMNIST fixed1 full-base 过 5-seed medium:
  acc gap < 0.005
  AUC improve > 0.09
  phi/J/ECE/branch 均过线

learnable bothcos07 接近，但 AUC improve = 8.72%，低于 9% strict target。
```

因此触发 P5，并且 P5 选择 fixed1 alpha mode。

### P5：10-seed final confirm, fixed-alpha selected

结果目录：

```text
results/gafu_v3_2_p5_final_fixed
```

P5 复用 P2/P3 seed0-4，补跑 seed5-9，共 60 条 row。

| dataset | method | runs | acc mean ± std | gap | AUC imp | phi red | J red | branch/A | ECE red |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | AdamW-final-alphaFixed1 | 10 | 0.8347 ± 0.0096 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 |
| Fashion | GA-FU-v2-final-alphaFixed1 | 10 | 0.8448 ± 0.0140 | -0.0101 | 0.0418 | 0.349 | 1.000 | 0.635 | 0.133 |
| Fashion | F-V3-HARD-base30-final-alphaFixed1 | 10 | 0.8469 ± 0.0111 | -0.0122 | 0.1034 | 0.356 | 1.000 | 0.656 | 0.221 |
| KMNIST | AdamW-final-alphaFixed1 | 10 | 0.7776 ± 0.0155 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1.000 | 0.000 |
| KMNIST | GA-FU-v2-final-alphaFixed1 | 10 | 0.7786 ± 0.0136 | -0.0010 | 0.0565 | 0.255 | 1.000 | 0.622 | 0.124 |
| KMNIST | K-V3-FULL-base-final-alphaFixed1 | 10 | 0.7774 ± 0.0150 | 0.0002 | 0.0812 | 0.280 | 1.000 | 0.580 | 0.171 |

Paired delta vs AdamW：

| dataset | method | acc delta mean | acc CI95 | AUC delta mean | AUC CI95 |
|---|---|---:|---|---:|---|
| Fashion | GA-FU-v2-final-alphaFixed1 | +0.0101 | [0.0027, 0.0181] | +0.0231 | [0.0107, 0.0335] |
| Fashion | F-V3-HARD-base30-final-alphaFixed1 | +0.0122 | [0.0058, 0.0187] | +0.0573 | [0.0492, 0.0657] |
| KMNIST | GA-FU-v2-final-alphaFixed1 | +0.0010 | [-0.0070, 0.0081] | +0.0301 | [0.0169, 0.0434] |
| KMNIST | K-V3-FULL-base-final-alphaFixed1 | -0.0002 | [-0.0078, 0.0073] | +0.0433 | [0.0343, 0.0536] |

P5 判断：

```text
Fashion:
  F-V3-HARD-base30-final-alphaFixed1 10-seed confirmed.
  可以写 Fashion clean default。

KMNIST:
  K-V3-FULL-base-final-alphaFixed1 acc 与 AdamW 基本持平：
    paired acc delta = -0.0002, CI 跨 0
  AUC/J/ECE 明显改善：
    paired AUC delta = +0.0433 absolute val-loss AUC
    relative AUC improve = 8.12%
  但低于 P3 strict 9% target，且 no-KAN ratio 未达 0.7。
  因此不能写 KMNIST clean default，只能写 geometry/convergence Pareto candidate。
```

### P4 / P6 / P7 / P8 analysis

生成文件：

```text
results/gafu_v3_2_p4_alpha_decision/alpha_decision.csv
results/gafu_v3_2_p6_mechanism/mechanism_summary.csv
results/gafu_v3_2_p7_direction_audit/direction_summary.csv
results/gafu_v3_2_p8_target_matched/target_by_run.csv
results/gafu_v3_2_p8_target_matched/target_summary.csv
docs/DG-KAN_Optimizer_v3.2_结果复盘.md
```

P6 mechanism：

```text
Fashion v3:
  no_kan_drop = 0.159
  AdamW no_kan_drop = 0.191
  ratio = 0.83 > 0.7
  branch/A = 0.656
  margin contribution positive
  pass expression audit.

KMNIST v3:
  no_kan_drop = 0.176
  AdamW no_kan_drop = 0.265
  ratio = 0.66 < 0.7
  branch/A = 0.580
  margin contribution positive
  expression evidence incomplete.
```

P7 direction：

```text
true-Gram methods have stable positive AUC/geometry gains.
Fashion hard switch transition_loss_jump is negative on average, so P10 smooth revisit is not triggered.
trust_clip_rate remains 0.
```

P8 target-matched / wall-clock：

| dataset | method | relaxed loss reached | mean epochs | mean time sec | step ms |
|---|---|---:|---:|---:|---:|
| Fashion | AdamW-final-alphaFixed1 | 1.0 | 1.3 | 2.71 | 87.0 |
| Fashion | F-V3-HARD-base30-final-alphaFixed1 | 1.0 | 2.0 | 4.65 | 96.8 |
| KMNIST | AdamW-final-alphaFixed1 | 1.0 | 2.1 | 4.39 | 87.1 |
| KMNIST | K-V3-FULL-base-final-alphaFixed1 | 1.0 | 2.3 | 5.34 | 96.7 |

P8 判断：

```text
v3 improves full-trajectory AUC and final calibration/geometry.
But it does not prove wall-clock faster convergence:
  step time is about 1.11x AdamW
  relaxed-loss time-to-target is slower than AdamW in this epoch-level audit
```

### v3.2 总结

```text
Final selected alpha_mode for P5:
  fixed1

Fashion:
  clean default confirmed:
    F-V3-HARD-base30-final-alphaFixed1

KMNIST:
  not clean default.
  geometry/convergence Pareto candidate:
    K-V3-FULL-base-final-alphaFixed1

Do not claim:
  KMNIST clean default
  wall-clock faster convergence

Can claim:
  Fashion true-Gram hard diag-to-full is 10-seed confirmed.
  KMNIST true full-Gram matches AdamW accuracy while improving validation-loss AUC, geometry, and ECE.
```

## 2026-05-02 DG-KAN Optimizer v3.3：统一 Full-Start 与 Unified Optimizer 实验

依据文档：

```text
docs/DG-KAN_Optimizer_v3.3_统一FullStart与UnifiedOptimizer实验计划.md
```

本轮目标：

```text
1. 验证 full_sobolev_gram from start + alphaFixed1 是否能成为跨 Fashion-MNIST / KMNIST 的统一 profile。
2. 验证 UO-PGAdam / UO-NormPGAdam / UO-PostAdam 这类统一 AdamW-style optimizer 是否可替代 hybrid functional update。
```

代码变更：

```text
experiments/dgkan_core.py
  added unified_optimizer_mode / kan_grad_transform / UO audit fields
  added PGAdam, NormPGAdam, PostAdam, PostAdam-trust, AllFunctionalRestSGD paths
  added optimizer moment/update/bad-step/memory metrics

experiments/run_gafu_v3.py
  added V3_3_P0_SMOKE
  added V3_3_P1_UFULL_SANITY
  added V3_3_P2_UFULL_EXPR3
  added V3_3_P3_UO_SMOKE
  added V3_3_P4_UO_EXPR3
  added V3_3_P5_CONFIRM5
  added V3_3_P6_CONFIRM10
  added moment_audit.csv output

experiments/analyze_gafu_v33.py
  added P6 scorecard / paired delta / P7 expression / P8 UO moment / P9 wall-clock aggregation
```

检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_v3.py experiments/analyze_gafu_v33.py
```

通过。

### P0 smoke

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_3_P0_SMOKE \
  --datasets Fashion-MNIST,KMNIST \
  --out-dir results/gafu_v3_3_p0_smoke \
  --fresh --device auto --no-download --continue-on-error
```

结果：

```text
rows = 10
run failures = 0
smoke_finite_pass = True
full-Sobolev condition = 6551.28 for h32/b8 smoke
trust_clip_rate = 0 for U-FULL / PGAdam / NormPGAdam / AllFunctionalRestSGD
```

### P1 seed0 U-FULL sanity

命令使用 `h96/d4/b24`，Fashion `30 epochs`，KMNIST `20 epochs`，`audit_batch_size=64`。

关键 seed0：

| dataset | method | acc | AUC imp | noKAN | phi red | J red | ECE red |
|---|---|---:|---:|---:|---:|---:|---:|
| Fashion | U-FULL-base | 0.8380 | 0.1134 | 0.610 | 0.369 | 0.999 | 0.136 |
| Fashion | U-FULL-f085 | 0.8400 | 0.1219 | 0.643 | 0.369 | 0.999 | 0.142 |
| Fashion | U-FULL-cplus | 0.8350 | 0.0783 | 0.819 | 0.368 | 0.999 | 0.164 |
| KMNIST | U-FULL-base | 0.8020 | 0.1098 | 0.772 | 0.288 | 0.999 | 0.261 |
| KMNIST | U-FULL-f085 | 0.7970 | 0.1124 | 0.810 | 0.288 | 0.999 | 0.201 |

判断：

```text
Fashion:
  base/f085 acc+AUC 强，但 seed0 noKAN 不足。
  cplus 提升 expression，但 AUC 接近筛选边界。

KMNIST:
  base/f085 同时过 acc/AUC/noKAN，expression rescue 有信号。
```

### P3 unified optimizer smoke

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_3_P3_UO_SMOKE \
  --datasets Fashion-MNIST,KMNIST \
  --out-dir results/gafu_v3_3_p3_uo_smoke \
  --fresh --device auto --no-download \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

关键失败信号：

| dataset | method | acc | AUC imp | branch/A | phi red | J red | clip | moment cos | bad rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | UO-PGAdam | 0.8050 | -0.307 | 12.63 | -12.76 | -22.88 | 0.000 | 0.427 | 0.042 |
| Fashion | UO-NormPGAdam | 0.8230 | -0.318 | 13.86 | -13.02 | -64.72 | 0.000 | 0.421 | 0.042 |
| Fashion | UO-PostAdam-trust030 | 0.8270 | -0.018 | 2.44 | -1.94 | -67.44 | 0.071 | 0.063 | 0.108 |
| KMNIST | UO-PGAdam | 0.7470 | -0.848 | 35.13 | -39.79 | -43.20 | 0.000 | 0.419 | 0.037 |
| KMNIST | UO-NormPGAdam | 0.7430 | -0.847 | 36.48 | -38.61 | -18.96 | 0.000 | 0.422 | 0.037 |
| KMNIST | UO-PostAdam-trust030 | 0.7830 | -0.025 | 4.46 | -5.68 | -6.16 | 0.083 | 0.076 | 0.121 |

结论：

```text
UO variants are quarantined at P3.
P4 was intentionally not expanded.
Hybrid functional coeff update + AdamW rest remains mainline.
```

### P2 U-FULL 3-seed expression rescue

扩展候选：

```text
U-FULL-base
U-FULL-f085
U-FULL-cplus
U-FULL-f085-cplus
```

关键 3-seed：

| dataset | method | acc mean ± std | gap | AUC imp | noKAN | phi red | J red | ECE red | branch/A |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | U-FULL-base | 0.8487 ± 0.0095 | -0.0157 | 0.1136 | 0.763 | 0.360 | 1.000 | 0.221 | 0.611 |
| Fashion | U-FULL-f085 | 0.8487 ± 0.0084 | -0.0157 | 0.1195 | 0.771 | 0.360 | 0.999 | 0.204 | 0.627 |
| KMNIST | U-FULL-base | 0.7853 ± 0.0170 | 0.0023 | 0.0997 | 0.739 | 0.282 | 1.000 | 0.178 | 0.577 |
| KMNIST | U-FULL-f085 | 0.7860 ± 0.0128 | 0.0017 | 0.1111 | 0.765 | 0.281 | 1.000 | 0.162 | 0.598 |

判断：

```text
U-FULL-f085-alphaFixed1 passed P2 on both datasets.
It became the P5 candidate.
```

### P5 5-seed confirm

确认方法：

```text
AdamW-alphaFixed1
GA-FU-v2-alphaFixed1
v3.2 dataset-specific best
U-FULL-f085-alphaFixed1
```

关键 5-seed：

| dataset | method | acc mean ± std | gap | AUC imp | noKAN | phi red | J red | ECE red |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion | U-FULL-f085 | 0.8482 ± 0.0089 | -0.0090 | 0.0962 | 0.829 | 0.360 | 1.000 | 0.198 |
| KMNIST | U-FULL-f085 | 0.7848 ± 0.0122 | -0.0056 | 0.0980 | 0.719 | 0.277 | 1.000 | 0.184 |

判断：

```text
U-FULL-f085 passed 5-seed unified full-start gate.
Proceed to P6 10-seed final confirm.
```

### P6 10-seed final confirm

确认方法：

```text
AdamW-alphaFixed1
v3.2 dataset-specific best
U-FULL-f085-alphaFixed1
```

Final scorecard：

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion | AdamW-alphaFixed1 | 10 | 0.8347 | 0.0096 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 1.000 | 1.000 |
| Fashion | F-V3-HARD-base30-v32best | 10 | 0.8469 | 0.0111 | -0.0122 | 0.1034 | 0.356 | 1.000 | 0.656 | 0.222 | 0.832 | 1.367 |
| Fashion | U-FULL-f085 | 10 | 0.8449 | 0.0103 | -0.0102 | 0.0816 | 0.360 | 1.000 | 0.614 | 0.207 | 0.793 | 1.376 |
| KMNIST | AdamW-alphaFixed1 | 10 | 0.7776 | 0.0155 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 1.000 | 1.000 |
| KMNIST | K-V3-FULL-base-v32best | 10 | 0.7774 | 0.0150 | 0.0002 | 0.0812 | 0.280 | 1.000 | 0.580 | 0.171 | 0.663 | 1.392 |
| KMNIST | U-FULL-f085 | 10 | 0.7804 | 0.0138 | -0.0028 | 0.0914 | 0.281 | 1.000 | 0.600 | 0.174 | 0.696 | 1.397 |

Paired delta vs AdamW：

| dataset | method | acc delta | acc CI95 | AUC delta | AUC CI95 | noKAN |
|---|---|---:|---:|---:|---:|---:|
| Fashion | F-V3-HARD-base30-v32best | +0.0122 | [0.0058, 0.0187] | +0.0573 | [0.0492, 0.0657] | 0.832 |
| Fashion | U-FULL-f085 | +0.0102 | [0.0057, 0.0147] | +0.0452 | [0.0345, 0.0576] | 0.793 |
| KMNIST | K-V3-FULL-base-v32best | -0.0002 | [-0.0078, 0.0073] | +0.0433 | [0.0343, 0.0536] | 0.663 |
| KMNIST | U-FULL-f085 | +0.0028 | [-0.0041, 0.0099] | +0.0487 | [0.0402, 0.0575] | 0.696 |

P6 判断：

```text
Fashion:
  U-FULL-f085 passes strict clean conditions.
  It is slightly below F-V3-HARD on AUC/acc, but still clearly positive vs AdamW.

KMNIST:
  U-FULL-f085 improves over AdamW on mean acc, AUC, geometry, ECE, and branch expression.
  It improves noKAN ratio vs v3.2 full-base:
    0.696 vs 0.663
  But it misses strict expression threshold:
    0.696 < 0.700
```

### P7 / P8 / P9 artifacts

生成：

```text
results/gafu_v3_3_p7_expression/expression_summary.csv
results/gafu_v3_3_p7_expression/branch_vs_no_kan.png
results/gafu_v3_3_p8_uo_moment/uo_smoke_summary.csv
results/gafu_v3_3_p8_uo_moment/moment_amp_vs_branch.png
results/gafu_v3_3_p9_wallclock/target_summary.csv
results/gafu_v3_3_p9_wallclock/time_to_relaxed_loss.png
docs/DG-KAN_Optimizer_v3.3_结果复盘.md
```

Target-matched / wall-clock：

| dataset | method | relaxed loss reached | epochs | time sec | step ms |
|---|---|---:|---:|---:|---:|
| Fashion | AdamW-alphaFixed1 | 1.0 | 1.3 | 1.04 | 33.1 |
| Fashion | F-V3-HARD-base30-v32best | 1.0 | 2.0 | 2.17 | 45.3 |
| Fashion | U-FULL-f085 | 1.0 | 2.3 | 2.52 | 45.6 |
| KMNIST | AdamW-alphaFixed1 | 1.0 | 2.1 | 1.64 | 32.5 |
| KMNIST | K-V3-FULL-base-v32best | 1.0 | 2.3 | 2.50 | 45.3 |
| KMNIST | U-FULL-f085 | 1.0 | 2.7 | 2.94 | 45.4 |

Wall-clock 判断：

```text
True-Gram methods improve full-trajectory validation AUC.
But under audit_batch_size=64, step time is still about 1.37x-1.40x AdamW.
Do not claim wall-clock faster convergence.
```

### v3.3 最终结论

```text
Confirmed:
  full_sobolev_gram from start + alphaFixed1 + branch_final_scale=0.85
  is a v3.3 unified Pareto profile.

Not confirmed:
  strict unified clean default.
  KMNIST misses no-KAN expression threshold by 0.004:
    0.696 vs required 0.700

Rejected:
  unified AdamW-style optimizer dynamics as mainline.
  UO-PGAdam / UO-NormPGAdam / UO-PostAdam all show branch/geometry/moment failures in P3.

Mainline after v3.3:
  Hybrid optimizer remains necessary:
    KAN coefficients use explicit functional-space update.
    non-KAN parameters use AdamW.

Recommended wording:
  U-FULL-f085 is the unified Pareto profile.
  Fashion clean default can still prefer F-V3-HARD-base30.
  KMNIST is substantially improved over v3.2 full-base on expression, but still borderline for strict clean expression.
```

## 2026-05-02 DG-KAN Optimizer v3.4：AFU All-Functional Update 实验

依据文档：

```text
docs/DG-KAN_Optimizer_v3.4_AFU_AllFunctional实验计划.md
```

本轮目标：

```text
1. 保留 v3.3 的 U-FULL-f085-alphaFixed1 统一 Pareto profile。
2. 正面测试真正的 AFU：非 KAN 参数也使用 parameter-group-specific functional / geometry-aware update。
3. 区分 partial AFU 是否可行，以及 full all-functional-lite 是否能移除 non-KAN AdamW。
```

代码变更：

```text
experiments/dgkan_core.py
  added AFU config fields:
    afu_kan_bias / afu_head / afu_stem / afu_ln_scope
    afu_head_lr_mult / afu_stem_lr_mult / afu_ln_lr_mult / afu_bias_lr_mult
    afu_cov_ema_beta / afu_head_rho / afu_stem_rho / afu_ln_rho / afu_bias_rho

  added AFU update paths:
    KAN coeff: existing full Sobolev functional update
    KAN bias: diagonal grad-square functional update
    Head Linear: feature covariance preconditioned update
    Stem Linear: input diagonal covariance update
    LayerNorm affine: diagonal grad-square Fisher-style update

  added AFU metrics:
    afu_{group}_update_norm_mean
    afu_{group}_update_over_param_mean/p95
    afu_{group}_raw_grad_norm_mean
    afu_{group}_precond_grad_norm_mean
    afu_{group}_cos_raw_precond_mean
    afu_{group}_metric_condition_mean
    afu_{group}_update_share_mean
    afu_head_cov_time_ms / afu_head_solve_time_ms / afu_stem_metric_time_ms / afu_ln_metric_time_ms

experiments/run_gafu_v3.py
  added packages:
    V3_4_P0_SMOKE
    V3_4_P2_COMPONENT3
    V3_4_P3_CUMULATIVE3
    V3_4_P4_EXPR_RESCUE
    V3_4_P5_CONFIRM5
    V3_4_P6_CONFIRM10
  added per-run:
    group_update_audit.csv
    metric_condition_audit.csv

experiments/analyze_gafu_v34.py
  added P0/P2/P3/P4/P5 aggregation
  added paired deltas vs AdamW and vs Hybrid
  added P7 mechanism and P8 wall-clock summaries
  added docs/DG-KAN_Optimizer_v3.4_AFU_结果复盘.md
```

检查：

```bash
python -m py_compile experiments/dgkan_core.py experiments/run_gafu_v3.py experiments/analyze_gafu_v34.py
```

通过。

### P0 AFU smoke

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_4_P0_SMOKE \
  --datasets Fashion-MNIST,KMNIST \
  --out-dir results/gafu_v3_4_p0_smoke \
  --fresh --device auto --no-download --continue-on-error
```

结果：

```text
rows = 13
run failures = 0
full-Sobolev condition = 6551.28
AFU group metrics finite
```

P0 早期信号：

| dataset | method | acc | AUC | note |
|---|---|---:|---:|---|
| Fashion | AFU-2-HeadCov | 0.4453 | 1.8767 | 能跑，head condition ~979 |
| Fashion | AFU-4-StemDiag | 0.1641 | 2.3254 | smoke 即欠拟合 |
| Fashion | AFU-5-FullDiagLite | 0.1562 | 2.3275 | full AFU early underfit |
| KMNIST | AFU-2-HeadCov | 0.2734 | 1.8182 | 能跑 |
| KMNIST | AFU-4-StemDiag | 0.1328 | 2.3414 | smoke 即欠拟合 |
| KMNIST | AFU-5-FullDiagLite | 0.1328 | 2.3451 | full AFU early underfit |

P0 判断：

```text
实现路径通过。
StemDiag / FullDiagLite 已出现明显 underfit 风险。
HeadCov 数值健康，进入 P2。
```

### P1 direction audit

本轮没有单独扩展 explicit shadow-step runner；根据 P0/P2 的 per-group update audit 生成了 proxy direction audit：

```text
results/gafu_v3_4_p1_direction_audit/direction_proxy_summary.csv
```

注意：

```text
这是训练更新中的 direction/update/condition proxy，不是完整 shadow-step。
由于 P2/P3 已经明确显示 full AFU underfit，本轮没有再扩 shadow-step。
```

### P2 single-component AFU ablation, 3 seeds

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_4_P2_COMPONENT3 \
  --datasets Fashion-MNIST,KMNIST \
  --out-dir results/gafu_v3_4_p2_component3 \
  --fresh --device auto --no-download \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

关键 3-seed：

| dataset | method | acc | dAcc vs Hybrid | AUC imp | noKAN | ECE red | judgment |
|---|---|---:|---:|---:|---:|---:|---|
| Fashion | AFU-0-Hybrid | 0.8487 | 0.0000 | 0.1195 | 0.771 | 0.204 | baseline |
| Fashion | AFU-2-HeadCov | 0.8507 | +0.0020 | 0.0992 | 1.030 | 0.552 | expression-positive |
| Fashion | AFU-HeadOnly | 0.8497 | +0.0010 | 0.0872 | 0.984 | 0.497 | expression-positive |
| Fashion | AFU-LNOnly | 0.8483 | -0.0003 | 0.1145 | 0.828 | 0.222 | safe |
| Fashion | AFU-StemOnly | 0.7827 | -0.0660 | -0.2826 | 1.892 | 0.146 | underfit |
| KMNIST | AFU-0-Hybrid | 0.7860 | 0.0000 | 0.1111 | 0.765 | 0.162 | baseline |
| KMNIST | AFU-2-HeadCov | 0.7650 | -0.0210 | 0.1115 | 0.917 | 0.350 | expression-positive but underfit |
| KMNIST | AFU-HeadOnly | 0.7713 | -0.0147 | 0.1242 | 0.913 | 0.384 | underfit |
| KMNIST | AFU-LNOnly | 0.7820 | -0.0040 | 0.1107 | 0.754 | 0.216 | safest |
| KMNIST | AFU-StemOnly | 0.6363 | -0.1497 | -0.7905 | 0.559 | -0.398 | hard fail |

P2 判断：

```text
HeadCov:
  Fashion succeeds strongly.
  KMNIST improves AUC/noKAN/ECE but loses 2.1 points vs Hybrid.

LNOnly:
  safest cross-dataset partial AFU.
  Small accuracy cost, positive AUC, geometry preserved.

StemOnly:
  fails hard on both datasets.
```

### P3 cumulative AFU 3-seed comparison

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_4_P3_CUMULATIVE3 \
  --datasets Fashion-MNIST,KMNIST \
  --methods AdamW-alphaFixed1-alphaFixed1,F-V3-HARD-base30-v32best-alphaFixed1,K-V3-FULL-base-v32best-alphaFixed1,AFU-0-Hybrid-alphaFixed1,AFU-2-HeadCov-alphaFixed1,AFU-3-HeadCov-LNHead-alphaFixed1,AFU-5-FullDiagLite-alphaFixed1,AFU-6-FullKFACLite-alphaFixed1 \
  --out-dir results/gafu_v3_4_p3_cumulative3 \
  --fresh --device auto --no-download \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

关键结果：

| dataset | method | acc | dAcc vs Hybrid | AUC imp | noKAN | ECE red | AdamW groups |
|---|---|---:|---:|---:|---:|---:|---:|
| Fashion | AFU-0-Hybrid | 0.8487 | 0.0000 | 0.1195 | 0.771 | 0.204 | 1 |
| Fashion | AFU-2-HeadCov | 0.8507 | +0.0020 | 0.0992 | 1.030 | 0.552 | 1 |
| Fashion | AFU-3-HeadCov-LNHead | 0.8480 | -0.0007 | 0.0828 | 1.057 | 0.597 | 1 |
| Fashion | AFU-5-FullDiagLite | 0.7997 | -0.0490 | -0.4006 | 3.349 | 0.313 | 0 |
| Fashion | AFU-6-FullKFACLite | 0.8003 | -0.0483 | -0.3981 | 3.354 | 0.319 | 0 |
| KMNIST | AFU-0-Hybrid | 0.7860 | 0.0000 | 0.1111 | 0.765 | 0.162 | 1 |
| KMNIST | AFU-2-HeadCov | 0.7650 | -0.0210 | 0.1115 | 0.917 | 0.350 | 1 |
| KMNIST | AFU-3-HeadCov-LNHead | 0.7603 | -0.0257 | 0.0647 | 0.909 | 0.655 | 1 |
| KMNIST | AFU-5-FullDiagLite | 0.6243 | -0.1617 | -0.8353 | 1.662 | 0.679 | 0 |
| KMNIST | AFU-6-FullKFACLite | 0.6237 | -0.1623 | -0.8308 | 1.658 | 0.684 | 0 |

P3 判断：

```text
Full AFU-lite fails clearly.
Removing non-KAN AdamW causes nonKAN_functional_underfit:
  Fashion: -4.8 to -4.9 points vs Hybrid
  KMNIST:  -16.2 points vs Hybrid

HeadCov remains a partial AFU signal, not a replacement.
```

### P4 expression rescue micro-check

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_4_P4_EXPR_RESCUE \
  --datasets KMNIST \
  --out-dir results/gafu_v3_4_p4_expr_rescue \
  --fresh --device auto --no-download \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

结果：

| method | acc | gap | AUC imp | noKAN | phi red | ECE red |
|---|---:|---:|---:|---:|---:|---:|
| AFU-0-Hybrid-f0875 | 0.7780 | 0.0097 | 0.0797 | 0.776 | 0.282 | 0.140 |
| AFU-0-Hybrid-f090 | 0.7773 | 0.0103 | 0.0740 | 0.752 | 0.282 | 0.156 |
| AFU-2-HeadCov-f0875 | 0.7673 | 0.0203 | 0.1088 | 0.928 | 0.279 | 0.374 |
| AFU-2-HeadCov-f090 | 0.7720 | 0.0157 | 0.1185 | 0.951 | 0.280 | 0.404 |

P4 判断：

```text
branch_final_scale 0.875 / 0.90 does not rescue KMNIST cleanly.
Higher final scale can raise noKAN but loses accuracy/AUC gate.
Do not adopt.
```

### P5 partial AFU 5-seed confirm

命令：

```bash
python experiments/run_gafu_v3.py \
  --packages V3_4_P5_CONFIRM5 \
  --datasets Fashion-MNIST,KMNIST \
  --methods AdamW-alphaFixed1-alphaFixed1,F-V3-HARD-base30-v32best-alphaFixed1,K-V3-FULL-base-v32best-alphaFixed1,AFU-0-Hybrid-alphaFixed1,AFU-LNOnly-alphaFixed1,AFU-2-HeadCov-alphaFixed1 \
  --out-dir results/gafu_v3_4_p5_confirm5 \
  --fresh --device auto --no-download \
  --hidden-dim 96 --depth 4 --basis-count 24 \
  --batch-size 256 --eval-batch-size 512 --audit-batch-size 64 \
  --continue-on-error
```

5-seed scorecard：

| dataset | method | acc | dAcc vs Hybrid | AUC imp | noKAN | ECE red | time/A |
|---|---|---:|---:|---:|---:|---:|---:|
| Fashion | AFU-0-Hybrid | 0.8482 | 0.0000 | 0.0962 | 0.829 | 0.198 | 1.386 |
| Fashion | AFU-LNOnly | 0.8446 | -0.0036 | 0.1057 | 0.769 | 0.196 | 1.666 |
| Fashion | AFU-2-HeadCov | 0.8482 | 0.0000 | 0.0999 | 1.076 | 0.520 | 1.657 |
| KMNIST | AFU-0-Hybrid | 0.7848 | 0.0000 | 0.0980 | 0.719 | 0.184 | 1.410 |
| KMNIST | AFU-LNOnly | 0.7766 | -0.0082 | 0.1052 | 0.726 | 0.201 | 1.682 |
| KMNIST | AFU-2-HeadCov | 0.7636 | -0.0212 | 0.1196 | 0.935 | 0.341 | 1.666 |

Paired delta vs Hybrid：

| dataset | method | acc delta | AUC delta |
|---|---|---:|---:|
| Fashion | AFU-LNOnly | -0.0036 | +0.0053 |
| Fashion | AFU-2-HeadCov | +0.0000 | +0.0021 |
| KMNIST | AFU-LNOnly | -0.0082 | +0.0037 |
| KMNIST | AFU-2-HeadCov | -0.0212 | +0.0112 |

P5 判断：

```text
AFU-LNOnly:
  safe-ish, improves AUC and keeps geometry.
  But KMNIST is 0.82 points below Hybrid, exceeding the <0.5% target.

AFU-2-HeadCov:
  expression/calibration-positive.
  But KMNIST loses 2.12 points vs Hybrid.

No AFU candidate passes P5 medium across both datasets.
```

### P6

```text
Not run.
Reason: no AFU candidate passed P5 strongly enough.
```

### P7 / P8 artifacts

生成：

```text
results/gafu_v3_4_p1_direction_audit/direction_proxy_summary.csv
results/gafu_v3_4_p7_mechanism/mechanism_summary.csv
results/gafu_v3_4_p7_mechanism/auc_vs_no_kan.png
results/gafu_v3_4_p7_mechanism/phi_vs_no_kan.png
results/gafu_v3_4_p8_wallclock/target_summary.csv
results/gafu_v3_4_p8_wallclock/auc_vs_step_time.png
results/gafu_v3_4_failure_table.csv
docs/DG-KAN_Optimizer_v3.4_AFU_结果复盘.md
```

Wall-clock：

| dataset | method | relaxed loss reached | epochs | time sec | step ms |
|---|---|---:|---:|---:|---:|
| Fashion | AdamW-alphaFixed1 | 1.0 | 1.4 | 1.10 | 32.8 |
| Fashion | AFU-0-Hybrid | 1.0 | 2.2 | 2.40 | 45.5 |
| Fashion | AFU-LNOnly | 1.0 | 2.0 | 2.63 | 54.7 |
| Fashion | AFU-2-HeadCov | 1.0 | 2.6 | 3.39 | 54.4 |
| KMNIST | AdamW-alphaFixed1 | 1.0 | 2.2 | 1.72 | 32.6 |
| KMNIST | AFU-0-Hybrid | 1.0 | 2.8 | 3.10 | 46.0 |
| KMNIST | AFU-LNOnly | 1.0 | 2.8 | 3.69 | 54.9 |
| KMNIST | AFU-2-HeadCov | 1.0 | 4.0 | 5.22 | 54.4 |

### v3.4 最终结论

```text
AFU-5 / AFU-6 true all-functional update:
  failed.
  Current non-KAN functional metrics cause underfit when AdamW is fully removed.

Partial AFU:
  HeadCov is expression/calibration-positive, especially Fashion.
  LNOnly is the safest component.
  But neither passes P5 medium across both datasets.

Mainline:
  Hybrid remains default:
    KAN coefficients use Sobolev functional update.
    non-KAN parameters still need AdamW.

What v3.4 proves:
  The previous UO failure was not enough by itself.
  After testing actual parameter-group functional metrics, full AFU is still not viable under current metrics.

Useful future direction:
  HeadCov may be useful as a controlled expression/calibration component.
  Stem functional update is the main blocker and likely needs a better metric or schedule before full AFU can work.
```

## 2026-05-02 DG-KAN Optimizer v3.5 U-FULL f1 实验

# DG-KAN Optimizer v3.5 U-FULL f1 结果复盘

## P0 Config Freeze

```json
{
  "pass": true,
  "checks": [
    {
      "dataset": "Fashion-MNIST",
      "method": "U-FULL-f100-alphaFixed1",
      "branch_final_scale": 1.0,
      "alpha_trainable": 0.0,
      "alpha_final_mean": 1.0,
      "v3_metric_active": "full_sobolev_gram",
      "v3_metric_geometry": "full_sobolev_gram",
      "nonkan_update": "adamw",
      "trust_clip_rate": 0.0,
      "v3_phase_final": "GEOMETRY",
      "v3_metric_mix_auc": 1.0,
      "pass": true
    },
    {
      "dataset": "KMNIST",
      "method": "U-FULL-f100-alphaFixed1",
      "branch_final_scale": 1.0,
      "alpha_trainable": 0.0,
      "alpha_final_mean": 1.0,
      "v3_metric_active": "full_sobolev_gram",
      "v3_metric_geometry": "full_sobolev_gram",
      "nonkan_update": "adamw",
      "trust_clip_rate": 0.0,
      "v3_phase_final": "GEOMETRY",
      "v3_metric_mix_auc": 1.0,
      "pass": true
    }
  ]
}
```

## P1 Seed0 Sanity

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 1.0000 | 0.8310 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 1.0000 | 0.8370 | 0.0000 | -0.0060 | 0.0997 | 0.3660 | 0.9992 | 0.6401 | 0.1303 | 0.7269 | 1.3025 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 1.0000 | 0.8400 | 0.0000 | -0.0090 | 0.1219 | 0.3690 | 0.9989 | 0.6192 | 0.1423 | 0.6426 | 1.3208 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 1.0000 | 0.8430 | 0.0000 | -0.0120 | 0.1064 | 0.3687 | 0.9976 | 0.6278 | 0.2175 | 0.6908 | 1.3248 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 1.0000 | 0.7920 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 1.0000 | 0.8020 | 0.0000 | -0.0100 | 0.1098 | 0.2877 | 0.9994 | 0.5755 | 0.2608 | 0.7724 | 1.5288 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 1.0000 | 0.7970 | 0.0000 | -0.0050 | 0.1124 | 0.2882 | 0.9993 | 0.5989 | 0.2009 | 0.8097 | 1.4676 |
| KMNIST | U-FULL-f100-alphaFixed1 | 1.0000 | 0.8100 | 0.0000 | -0.0180 | 0.0762 | 0.2871 | 0.9983 | 0.6220 | 0.2216 | 0.7948 | 1.5203 |

P1 f100 pass: `True`

```json
{
  "Fashion-MNIST": "pass",
  "KMNIST": "pass"
}
```

## P2 5-Seed Clean Confirm

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 0.8392 | 0.0095 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5.0000 | 0.8474 | 0.0101 | -0.0082 | 0.1060 | 0.3563 | 0.9998 | 0.6592 | 0.2065 | 0.8169 | 1.4056 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 0.8482 | 0.0089 | -0.0090 | 0.0962 | 0.3601 | 0.9997 | 0.6199 | 0.1982 | 0.8291 | 1.4099 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 0.8438 | 0.0080 | -0.0046 | 0.0982 | 0.3603 | 0.9994 | 0.6332 | 0.1746 | 0.7365 | 1.4252 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 0.7792 | 0.0180 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5.0000 | 0.7826 | 0.0151 | -0.0034 | 0.0906 | 0.2765 | 0.9999 | 0.5784 | 0.1859 | 0.6555 | 1.3324 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 0.7848 | 0.0122 | -0.0056 | 0.0980 | 0.2772 | 0.9999 | 0.5999 | 0.1836 | 0.7189 | 1.3561 |
| KMNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 0.7736 | 0.0214 | 0.0056 | 0.0895 | 0.2778 | 0.9998 | 0.6246 | 0.0995 | 0.6394 | 1.3611 |

P2 f100 pass: `False`

```json
{
  "Fashion-MNIST": "pass",
  "KMNIST": "fail gap=0.0056 auc=0.0895 phi=0.278 J=1.000 branch=0.625 ece=0.099 noKAN=0.639"
}
```

Selected unified profile: `U-FULL-f085-reference-alphaFixed1`

## P2 Paired Delta vs AdamW

| dataset | method | acc delta | acc CI lo | acc CI hi | AUC delta | AUC CI lo | AUC CI hi | noKAN |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 0.0090 | 0.0010 | 0.0170 | 0.0539 | 0.0355 | 0.0735 | 0.8291 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 0.0046 | -0.0034 | 0.0128 | 0.0550 | 0.0393 | 0.0718 | 0.7365 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 0.0082 | -0.0002 | 0.0196 | 0.0594 | 0.0481 | 0.0723 | 0.8169 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 0.0056 | -0.0058 | 0.0184 | 0.0511 | 0.0374 | 0.0643 | 0.7189 |
| KMNIST | U-FULL-f100-alphaFixed1 | -0.0056 | -0.0158 | 0.0078 | 0.0466 | 0.0277 | 0.0670 | 0.6394 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 0.0034 | -0.0090 | 0.0156 | 0.0472 | 0.0315 | 0.0647 | 0.6555 |

## P3 10-Seed Final

Trigger decision: `not triggered because P2 did not pass`

_P3 not run._

## P4 Small-Data Generalization

| dataset | method | train | runs | acc | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 1000.0000 | 5.0000 | 0.7956 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 2000.0000 | 5.0000 | 0.8122 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 500.0000 | 5.0000 | 0.7626 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 6000.0000 | 5.0000 | 0.8392 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 1000.0000 | 5.0000 | 0.7936 | 0.0020 | 0.1336 | 0.1515 | 0.9955 | 0.6532 | 0.6159 | 0.6616 | 1.0069 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 2000.0000 | 5.0000 | 0.8054 | 0.0068 | 0.1423 | 0.1998 | 0.9992 | 0.6620 | 0.3990 | 0.6514 | 1.1197 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 500.0000 | 5.0000 | 0.7594 | 0.0032 | 0.1064 | 0.1093 | 0.9976 | 0.6635 | 0.7616 | 0.6993 | 1.0504 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 6000.0000 | 5.0000 | 0.8482 | -0.0090 | 0.0962 | 0.3601 | 0.9997 | 0.6199 | 0.1982 | 0.8291 | 1.4305 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 1000.0000 | 5.0000 | 0.6404 | -0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 2000.0000 | 5.0000 | 0.6826 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 500.0000 | 5.0000 | 0.5770 | -0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 6000.0000 | 5.0000 | 0.7792 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 1000.0000 | 5.0000 | 0.6474 | -0.0070 | 0.0836 | 0.1088 | 0.9854 | 0.6411 | 0.2409 | 0.5611 | 1.0032 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 2000.0000 | 5.0000 | 0.6978 | -0.0152 | 0.0806 | 0.1211 | 0.9957 | 0.6385 | 0.2634 | 0.6798 | 1.0957 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 500.0000 | 5.0000 | 0.5968 | -0.0198 | 0.0972 | 0.1034 | 0.9104 | 0.6420 | 0.2859 | 0.5037 | 1.0928 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 6000.0000 | 5.0000 | 0.7848 | -0.0056 | 0.0980 | 0.2772 | 0.9999 | 0.5999 | 0.1836 | 0.7189 | 1.4415 |

## P5 Label-Noise Generalization

| dataset | method | noise | runs | acc | AUC imp | ECE | ECE red | noKAN |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-noise0.2-alphaFixed1 | 0.2000 | 5.0000 | 0.7450 | 0.0000 | 0.1105 | -0.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-noise0.4-alphaFixed1 | 0.4000 | 5.0000 | 0.5894 | -0.0000 | 0.1753 | -0.0000 | 1.0000 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1-noise0.2 | 0.2000 | 5.0000 | 0.7776 | 0.0701 | 0.0516 | 0.5336 | 1.2033 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1-noise0.4 | 0.4000 | 5.0000 | 0.6550 | 0.0772 | 0.0346 | 0.8025 | 1.5455 |
| KMNIST | AdamW-alphaFixed1-noise0.2-alphaFixed1 | 0.2000 | 5.0000 | 0.6264 | -0.0000 | 0.2322 | -0.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-noise0.4-alphaFixed1 | 0.4000 | 5.0000 | 0.4794 | 0.0000 | 0.3310 | 0.0000 | 1.0000 |
| KMNIST | U-FULL-f085-reference-alphaFixed1-noise0.2 | 0.2000 | 5.0000 | 0.6644 | 0.0682 | 0.1357 | 0.4155 | 1.2578 |
| KMNIST | U-FULL-f085-reference-alphaFixed1-noise0.4 | 0.4000 | 5.0000 | 0.4660 | 0.1611 | 0.2012 | 0.3920 | 1.1452 |

## P6 CIFAR-Small ConvStem Precheck

| dataset | method | train | runs | acc | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR10 | ConvStem-DGKAN-AdamW-alphaFixed1 | 10000.0000 | 3.0000 | 0.4756 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f085-alphaFixed1 | 10000.0000 | 3.0000 | 0.4025 | 0.0731 | -0.0874 | 0.4908 | 1.0000 | 0.4598 | 0.1905 | 0.1224 | 1.0794 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f100-alphaFixed1 | 10000.0000 | 3.0000 | 0.4177 | 0.0579 | -0.0961 | 0.4900 | 1.0000 | 0.5098 | 0.2592 | 0.1926 | 1.0871 |

```text
P6 CIFAR-small: 9 successful rows; f085 acc=0.4025, gap_vs_conv_adamw=0.0731, AUC_imp=-0.0874; f100 acc=0.4177, gap_vs_conv_adamw=0.0579, AUC_imp=-0.0961
```

## P7 Target-Matched / Speed Proxy

| dataset | method | runs | relaxed loss reached | epochs relaxed loss | time relaxed loss | step ms |
|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 1.0000 | 1.4000 | 1.0960 | 32.6156 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5.0000 | 1.0000 | 2.0000 | 2.2005 | 45.8445 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 1.0000 | 2.2000 | 2.4294 | 45.9854 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 1.0000 | 2.0000 | 2.2313 | 46.4852 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 1.0000 | 2.2000 | 1.8101 | 34.2870 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5.0000 | 1.0000 | 2.4000 | 2.6380 | 45.6830 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 1.0000 | 2.8000 | 3.1268 | 46.4981 |
| KMNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 1.0000 | 2.6000 | 2.9124 | 46.6692 |

## P8 Rational/KAT Precheck

| dataset | method | runs | acc | acc std | gap vs MLP | AUC imp | ECE red | time/MLP | clip | condG | rawG | dirN | den min | den p01 | mem MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | MLP-AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.8500 | 0.0137 | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 0.0000 |  | 0.0000 | 0.0000 |  |  |  |
| Fashion-MNIST | Rational-DGKAN-AdamW-torch-alphaFixed1 | 3.0000 | 0.8457 | 0.0061 | 0.0043 | -0.0361 | -0.0311 | 4.0186 | 0.0000 |  | 0.0000 | 0.0000 | 1.0004 | 1.0006 |  |
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3.0000 | 0.8503 | 0.0060 | -0.0003 | -0.0473 | -0.0395 | 2.8456 | 0.0000 |  | 0.0000 | 0.0000 | 1.0005 | 1.0008 |  |
| Fashion-MNIST | Rational-DGKAN-U-FULL-torch-c0p02-alphaFixed1 | 3.0000 | 0.8383 | 0.0090 | 0.0117 | -0.0265 | -0.0832 | 5.5948 | 0.0851 | 36422.6719 | 0.4159 | 0.5515 | 1.0001 | 1.0001 |  |
| Fashion-MNIST | Rational-DGKAN-U-FULL-triton-c0p02-alphaFixed1 | 3.0000 | 0.8450 | 0.0106 | 0.0050 | -0.0243 | 0.0303 | 4.9895 | 0.0464 | 36422.6719 | 0.4248 | 0.3901 | 1.0001 | 1.0001 |  |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 3.0000 | 0.8487 | 0.0084 | 0.0013 | 0.0061 | 0.0642 | 4.6658 | 0.0000 | 4725.9624 | 0.7991 | 0.8281 |  |  |  |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 3.0000 | 0.8423 | 0.0098 | 0.0077 | -0.0019 | 0.0423 | 4.6842 | 0.0000 | 4725.9624 | 0.8241 | 0.9588 |  |  |  |
| KMNIST | MLP-AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.7920 | 0.0120 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 |  | 0.0000 | 0.0000 |  |  |  |
| KMNIST | Rational-DGKAN-AdamW-torch-alphaFixed1 | 3.0000 | 0.7903 | 0.0172 | 0.0017 | -0.1282 | -0.1289 | 4.9322 | 0.0000 |  | 0.0000 | 0.0000 | 1.0004 | 1.0007 |  |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3.0000 | 0.7890 | 0.0059 | 0.0030 | -0.1314 | -0.1331 | 3.2510 | 0.0000 |  | 0.0000 | 0.0000 | 1.0006 | 1.0010 |  |
| KMNIST | Rational-DGKAN-U-FULL-torch-c0p02-alphaFixed1 | 3.0000 | 0.7863 | 0.0025 | 0.0057 | -0.0464 | 0.0001 | 6.3566 | 0.0650 | 36422.6719 | 0.2799 | 0.3374 | 1.0000 | 1.0000 |  |
| KMNIST | Rational-DGKAN-U-FULL-triton-c0p02-alphaFixed1 | 3.0000 | 0.7947 | 0.0085 | -0.0027 | -0.0183 | 0.0110 | 5.6982 | 0.0424 | 36422.6719 | 0.2191 | 0.2101 | 1.0000 | 1.0000 |  |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 3.0000 | 0.7860 | 0.0128 | 0.0060 | -0.1173 | 0.1165 | 5.3174 | 0.0000 | 4725.9624 | 0.3950 | 0.4217 |  |  |  |
| KMNIST | U-FULL-f100-alphaFixed1 | 3.0000 | 0.7833 | 0.0200 | 0.0087 | -0.1477 | 0.0809 | 5.3836 | 0.0000 | 4725.9624 | 0.4460 | 0.4609 |  |  |  |

```text
P8 Rational/KAT: 42 successful rows; Fashion-MNIST triton acc=0.8503, gap_vs_mlp=-0.0003, time/MLP=2.846, denom_min=1.001; Fashion-MNIST torch acc=0.8457, time/MLP=4.019; Fashion-MNIST U-FULL triton acc=0.8450, AUC_imp=-0.0243, condG=36422.7, clip=0.046, denom_min=1.000; Fashion-MNIST U-FULL torch acc=0.8383, AUC_imp=-0.0265; KMNIST triton acc=0.7890, gap_vs_mlp=0.0030, time/MLP=3.251, denom_min=1.001; KMNIST torch acc=0.7903, time/MLP=4.932; KMNIST U-FULL triton acc=0.7947, AUC_imp=-0.0183, condG=36422.7, clip=0.042, denom_min=1.000; KMNIST U-FULL torch acc=0.7863, AUC_imp=-0.0464
```

## Decision

```text
P0:
  config check pass = True

P1:
  U-FULL-f100 seed0 sanity pass = True

P2:
  U-FULL-f100 5-seed clean pass = False
  selected unified profile = U-FULL-f085-reference-alphaFixed1

P3:
  not triggered because P2 did not pass

P4/P5:
  See scorecards above for small-data and label-noise behavior.

P7:
  Target-matched wall-clock proxy was generated from clean confirm rows.
  Low-level kernel profiling and solve microbenchmarks were not expanded in this run.

P6/P8:
  CIFAR-small and Rational/KAT prechecks were included when their rows are present.
  Rational uses third_party/rational_kat_cu directly, with Triton autograd or the PyTorch fallback.
No custom backward rewrite was accepted in this round.
```

## 2026-05-02 DG-KAN v3.7 PureKAN U-FULL：代码检查、schedule 修复与 role-wise 复核

依据文档：

```text
docs/DG-KAN_v3.7_PureKAN_UFULL_代码检查与实验计划.md
```

新增/修改：

```text
experiments/dgkan_core.py
  PureKAN alpha_mode=fixed1 现在是真固定 alpha=1.0 buffer，不再保留 alpha_init=1.5。
  PureKANClassifier 将 alpha_mode 传入所有 residual block。
  PureKAN functional update 新增 input/block/output role-specific metric / lr / trust。
  结果记录 phase/metric trace、active/transition/geometry steps、per-role update stats、coeff coverage、PureKAN contribution audit。

experiments/run_gafu_v37.py
  新增 v3.7 P0/P1/P2/P3/P5/P6 package 与 one-batch shadow-step。
  修复 PureKAN warmup：geometry_min_epochs=999，让 branch_max_active_frac 真正控制 0.10/0.20 warmup 长度。

experiments/analyze_gafu_v37.py
  生成复盘文档：
  docs/DG-KAN_v3.7_PureKAN_UFULL_结果复盘.md
```

执行结果路径：

```text
results/gafu_v3_7_p0
results/gafu_v3_7_p1_shadow
results/gafu_v3_7_p2_schedule_fixed
results/gafu_v3_7_p3_role
```

### P0 code/config audit

结论：pass。

关键检查：

```text
alphaFixed1:
  pure_alpha_mean = 1.0
  pure_alpha_trainable = 0

PureKAN structure:
  learnable_nonkan_params = 0
  purekan_has_linear = 0
  purekan_has_layernorm_params = 0
  purekan_has_bias_params = 0

Functional coverage:
  coeff_param_seen_ratio = 1.0 for U-FULL rows

Warmup smoke:
  diagwarmup/identitywarmup active_steps_actual > 0
  full smoke geometry-only
```

### P1 one-batch shadow step

核心结果：

| dataset | method | train descent | val descent | bad step |
|---|---|---:|---:|---:|
| MNIST | full-sobolev-gram | -0.0408 | 0.0414 | 1 |
| Fashion-MNIST | full-sobolev-gram | 0.0125 | -0.0397 | 0 |
| KMNIST | full-sobolev-gram | -0.0021 | 0.0031 | 1 |

判断：

```text
Full Sobolev 在 PureKAN 上不是可靠 descent direction。
MNIST/KMNIST train batch 直接变差；Fashion train batch 下降但 validation 变差。
这更像 metric direction mismatch，不是参数覆盖或 backward 缺失。
```

### P2 schedule repair fixed

旧 P2 中发现：

```text
diagwarmup-0.10 和 diagwarmup-0.20 都只有 1 epoch active。
原因是 epoch-end geometry trigger 提前切换，导致 max_active_frac 没真正控制 warmup 长度。
```

修复后验证：

```text
MNIST/KMNIST:
  0.10 -> active_steps_actual = 48
  0.20 -> active_steps_actual = 96

Fashion:
  0.10 -> active_steps_actual = 72
  0.20 -> active_steps_actual = 144

smooth-0.20:
  MNIST/KMNIST transition_steps_actual = 49
  Fashion transition_steps_actual = 71
```

P2-fixed scorecard：

| dataset | best schedule | acc | gap vs PureKAN-AdamW | val-loss AUC | AUC imp vs full |
|---|---|---:|---:|---:|---:|
| MNIST | diag-to-full-smooth-0.20 | 0.8950 | 0.0353 | 0.8437 | -0.0707 |
| Fashion-MNIST | diag-to-full-smooth-0.20 | 0.8460 | 0.0103 | 0.5242 | 0.1211 |
| KMNIST | diag-to-full-smooth-0.20 | 0.7267 | 0.0453 | 0.7834 | 0.1475 |

结论：

```text
schedule 修复本身有效。
Fashion 被明显修复：AUC/ECE 大幅好于 full 和 AdamW，accuracy 只差约 1 point。
MNIST/KMNIST 仍然离 PureKAN-AdamW 很远，不能靠 warmup schedule 解决。
```

### P3 role-wise sweep

P3 最好信号：

| dataset | best role candidate | acc | gap vs PureKAN-AdamW | AUC imp vs all-full |
|---|---|---:|---:|---:|
| MNIST | iId-blockFull-oDiag-o4 | 0.9057 | 0.0247 | -0.1433 |
| Fashion-MNIST | ioDiag-blockFull-o4 | 0.8473 | 0.0090 | 0.0465 |
| KMNIST | iId-blockFull-oDiag-o4 | 0.7480 | 0.0240 | 0.1387 |

P3 gate：

```text
进入 P5 要求：
  acc gap vs PureKAN-AdamW < 1.0% on all three datasets
  AUC improvement vs PureKAN-UFULL-full > 0
  geometry better than PureKAN-AdamW
  ECE not worse than PureKAN-AdamW by > 5%

实际：
  no candidate enters P5
```

判断：

```text
Role-aware metrics 能明显修复 Fashion，并部分救 KMNIST。
但 MNIST/KMNIST 的 accuracy gap 仍是 2.4-4.7 points 级别。
没有候选满足进入 P5/P6 的条件。
```

### P4-P7

```text
P4 capacity/epoch budget check:
  not expanded
  reason: P3 没有产生三数据集都接近 PureKAN-AdamW 的候选

P5 3-seed candidate selection:
  not run
  reason: P3 entry gate failed

P6 5-seed confirm / P7 speed proxy:
  not allowed by plan because P5 was not reached
```

### v3.7 Final

```text
PureKAN U-FULL is not accepted in v3.7.

Confirmed:
  1. PureKAN alpha/fixed1 与 functional coverage bug 已修复。
  2. diag/identity warmup 现在真正执行 10%/20% active window。
  3. role-aware metric 是有效方向，尤其 Fashion。

Not confirmed:
  1. full-network PureKAN functional update 不能 match PureKAN-AdamW。
  2. 没有 role/schedule candidate 能进入 P5/P6。
  3. 当前瓶颈不是 CUDA/KAT backward，也不是参数覆盖，而是 PureKAN functional metric 的 direction quality。

Next:
  不建议继续加 seeds confirm。
  下一轮应重设计 PureKAN functional metric：更 layer-local / block-local，input/output 使用更安全 metric，再考虑确认实验。
```

## 2026-05-02 DG-KAN v3.8 GlobalTFU / Depthwise PureKAN

依据 `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`，实现并运行：

```text
P0 smoke: 21 rows
P1 one-batch shadow: 24 rows
P2 global TFU: 72 rows
P3 depth-wise sweep: 72 rows
```

关键结论：

```text
P0 pass: PureKAN alphaFixed1 / no non-KAN params / coeff coverage / TFU role metrics 均正常。
P1 pass: D6 GlobalTFU 把 D0 full Sobolev 的 MNIST/KMNIST bad step 修成三数据集 train/val descent 全正。
P2 fail: D6 方向修复不能稳定转化为 3-seed accuracy gate。
P3 fail: D3/D7 对 Fashion/MNIST 有帮助，D8 对 KMNIST 相对最好，但无候选进入 P4。
P4-P7 not run: 按计划 gate 未达，不做大 seed confirm。
```

复盘文档：

```text
docs/DG-KAN_v3.8_GlobalTFU_Depthwise_PureKAN_结果复盘.md
```

## 2026-05-02 DG-KAN v3.9 Normalization / FNG PureKAN

依据 `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`，实现并运行：

```text
P0 implementation smoke
P1 one-batch direction audit
P2 normalization ablation
P3 relaxed FNG diagnostic subset
```

关键结论：

```text
P0 pass: norm modes / functional norm / FNG update 覆盖正常。
P1 strict fail: no candidate passed the formal direction gate; no-bad-step FNG variants still had raw-precond cosine below 0.5.
P2: NoNorm fails hard, but AffineNorm/ScalarGain does not close the PureKAN functional gap, so normalization is necessary but not the missing key.
P3: relaxed diagnostic run did not justify P4/P5 expansion under the plan gate.
P4-P7 not run: no formal P1/P3 survivor.
```

复盘文档：

```text
docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_结果复盘.md
```



# DG-KAN v4.1 Functional Update Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md`。目标是验证重新设计的 PureKAN functional update：FTF、FC-Adam 与 FTR/FGN trust-region 是否能把 v3.7-v3.9 的局部方向修复转成长程训练收益。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 30 | 0 |
| P1 shadow | 30 | 0 |
| P2 micro | 72 | 5 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added FTF fields and per-role target-fitting updates for PureKAN coeffs.
  Added FC-Adam functional-coordinate updates through Sobolev Cholesky coordinates.
  Added an FTR-CG-small diagnostic path with layer-output trust scaling.
  Result rows now include FTF fit/residual/trust stats, FC-Adam reconstruction diagnostics,
  and FTR residual / predicted change statistics.

experiments/run_gafu_v41.py
  Added P0/P1/P2 packages and one-batch direction audit for v4.1 candidates.

experiments/analyze_gafu_v41.py
  Writes the required v4.1 CSV/JSON artifacts and this replay.
```

## P0 Implementation Smoke

P0 rows: `30`; errors: `0`.

Key audit:

```text
PureKAN alphaFixed1 / fixed norm paths ran without implementation errors.
Strict PureKAN rows kept learnable_nonKAN_params = 0.
Functional rows covered input/block/output coeff groups.
FTF, FC-Adam, and FTR all produced finite one-epoch smoke rows.
```

## P1 One-Batch Direction Gate

| method | bad | min train | min val | min FTF R2 | max delta | P1 pass |
|---|---|---|---|---|---|---|
| D0-allFullSobolev | 0 | 0.0016 | -0.0039 |  |  | yes |
| D6-allTaskAware | 1 | -0.0080 | -0.0091 |  |  | no |
| F4-FNG-leftFull-right | 0 | 0.0488 | 0.0046 |  |  | yes |
| FC-Adam-one-step | 0 | 0.0212 | -0.0377 |  |  | no |
| FTF-all-sequential | 0 | 0.0784 | 0.0636 | 1.0000 | 0.1000 | yes |
| FTF-all-simultaneous | 0 | 0.1247 | 0.0231 | 0.9999 | 0.1000 | yes |
| FTF-blocks-output | 0 | 0.0664 | 0.0086 | 0.9997 | 0.0997 | yes |
| FTF-output-only | 0 | 0.0093 | -0.0666 | 1.0000 | 0.0998 | no |
| FTR-CG-small | 0 | 0.0388 | 0.0023 |  |  | yes |

P1 survivors used for P2:

```text
D0-allFullSobolev, F4-FNG-leftFull-right, FTF-all-sequential, FTF-all-simultaneous, FTF-blocks-output, FTR-CG-small
```

Observation:

```text
FTF repaired the one-step direction signal strongly: blocks/output and all-layer modes had positive train and validation descent on all three datasets, with fit R2 near 1.0.
D6 still had a KMNIST bad train step and was not treated as a P2 candidate, but was later added as a P2 baseline because the plan requires AUC comparison vs D6.
FC-Adam improved train loss but failed the validation-descent gate on Fashion/KMNIST.
```

## P2 Micro-Run Scorecard

| dataset | method | runs | errors | acc | std | gap vs AdamW | AUC imp vs AdamW | AUC imp vs D6 | ECE red | phi ratio | FTF R2 | P2 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | D0-allFullSobolev | 3 | 0 | 0.8417 | 0.0304 | 0.0723 | -0.8900 | 0.0003 | -8.4610 | 0.7658 |  | 0 |
| MNIST | D6-allTaskAware | 3 | 0 | 0.8523 | 0.0068 | 0.0617 | -0.8906 | 0.0000 | -5.7526 | 0.7685 |  | 0 |
| MNIST | F4-FNG-leftFull-right | 3 | 0 | 0.9233 | 0.0110 | -0.0093 | -0.0388 | 0.4505 | -0.4553 | 0.7661 |  | 0 |
| MNIST | FTF-all-sequential | 3 | 0 | 0.1003 | 0.0005 | 0.8137 | -518293472445.2520 | -274141826146.2508 | -42.8198 | 509227.2591 | 0.9453 | 0 |
| MNIST | FTF-all-simultaneous | 3 | 0 | 0.1003 | 0.0005 | 0.8137 | -518293472445.2520 | -274141826146.2508 | -42.8198 | 509227.2591 | 0.9453 | 0 |
| MNIST | FTF-blocks-output | 3 | 0 | 0.1023 | 0.0033 | 0.8117 | -1754569223347315.0000 | -928047209855757.8750 | -42.7223 | 22285636.6922 | 0.7068 | 0 |
| MNIST | FTR-CG-small | 3 | 0 | 0.6687 | 0.0107 | 0.2453 | -1.9351 | -0.5525 | -18.2742 | 0.7644 |  | 0 |
| MNIST | PureKAN-AdamW | 3 | 0 | 0.9140 | 0.0043 | 0.0000 | 0.0000 | 0.4711 | 0.0000 | 1.0000 |  | 1 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0 | 0.8170 | 0.0174 | 0.0190 | -0.5294 | -0.1251 | -0.1849 | 0.7614 |  | 0 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0 | 0.8060 | 0.0134 | 0.0300 | -0.3594 | 0.0000 | 0.4009 | 0.7636 |  | 0 |
| Fashion-MNIST | F4-FNG-leftFull-right | 3 | 0 | 0.8227 | 0.0164 | 0.0133 | -0.1005 | 0.1904 | 0.3178 | 0.7607 |  | 1 |
| Fashion-MNIST | FTF-all-sequential | 2 | 1 | 0.1000 | 0.0000 | 0.7360 | -105351734110180880.0000 | -77501361878272944.0000 | -10.9915 | 203698804.2161 | 0.9480 | 0 |
| Fashion-MNIST | FTF-all-simultaneous | 2 | 1 | 0.1000 | 0.0000 | 0.7360 | -105351734110180880.0000 | -77501361878272944.0000 | -10.9915 | 203698804.2161 | 0.9480 | 0 |
| Fashion-MNIST | FTR-CG-small | 3 | 0 | 0.7167 | 0.0194 | 0.1193 | -1.7757 | -1.0420 | -3.1114 | 0.7598 |  | 0 |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0 | 0.8360 | 0.0156 | 0.0000 | 0.0000 | 0.2644 | 0.0000 | 1.0000 |  | 1 |
| KMNIST | D0-allFullSobolev | 3 | 0 | 0.5750 | 0.0185 | 0.1930 | -1.0548 | -0.1156 | -0.5424 | 0.7607 |  | 0 |
| KMNIST | D6-allTaskAware | 3 | 0 | 0.5610 | 0.0550 | 0.2070 | -0.8419 | 0.0000 | 0.2455 | 0.7640 |  | 0 |
| KMNIST | F4-FNG-leftFull-right | 3 | 0 | 0.7123 | 0.0132 | 0.0557 | -0.1847 | 0.3568 | 0.3224 | 0.7648 |  | 0 |
| KMNIST | FTF-all-sequential | 3 | 0 | 0.1070 | 0.0099 | 0.6610 | -788570634358.2345 | -428138390693.1385 | -9.8694 | 626491.2995 | 0.9186 | 0 |
| KMNIST | FTF-all-simultaneous | 3 | 0 | 0.1070 | 0.0099 | 0.6610 | -788570634358.2345 | -428138390693.1385 | -9.8694 | 626491.2995 | 0.9186 | 0 |
| KMNIST | FTF-blocks-output | 3 | 0 | 0.1000 | 0.0000 | 0.6680 | -117251294060245.6875 | -63659205857378.9766 | -9.9546 | 6377580.6882 | 0.6710 | 0 |
| KMNIST | FTR-CG-small | 3 | 0 | 0.3730 | 0.0140 | 0.3950 | -2.1432 | -0.7065 | -1.2296 | 0.7613 |  | 0 |
| KMNIST | PureKAN-AdamW | 3 | 0 | 0.7680 | 0.0067 | 0.0000 | 0.0000 | 0.4571 | 0.0000 | 1.0000 |  | 1 |

## P2 Failure Diagnosis

```text
No candidate passed the P2 joint gate.

FTF:
  P1 target fit was excellent, but P2 training was catastrophic.
  MNIST/KMNIST dropped to near chance accuracy, Fashion produced numerical eigensolve failures for several FTF rows, and val-loss AUC exploded.
  This matches the plan's failure mode: fit_R2 high + one-step descent positive + short-run acc bad => layer-local target fitting causes cross-layer drift.

FTR-CG-small:
  Local direction was acceptable, but short training underfit badly on all datasets.

F4-FNG-leftFull-right:
  Best practical candidate in P2.
  It matched/beat PureKAN-AdamW on MNIST and stayed within about 1.3 points on Fashion.
  It failed KMNIST by about 5.6 points, so it cannot enter P3.

D0/D6:
  Geometry is stable but accuracy remains below AdamW, especially on KMNIST.
```

## P3-P5 Decision

```text
P3 refinement: not run.
Reason: P2 produced no survivor.

P4 5-seed confirm: not run.
Reason: P3 was not reached.

P5 10-seed final: not run.
Reason: P4 was not reached.
```

## Artifacts

Required files were written under `results/v4_1/`:

```text
p0_invariants.csv
p1_direction_audit.csv
p2_micro_run_scorecard.csv
p2_failure_diagnosis.csv
p3_refinement_scorecard.csv
p4_confirm5_scorecard.csv
p5_confirm10_scorecard.csv
functional_target_fit.csv
representation_audit.csv
geometry_audit.csv
compute_audit.csv
failure_table.csv
aggregate_decision.json
figures/p2_kmnist_acc_gap.svg
figures/p2_kmnist_auc_vs_d6.svg
```

## Final Decision

```text
PureKAN functional optimization is still not solved in v4.1.

What improved:
  FTF gives a real one-step target-fitting direction.
  FNG remains the strongest short-run practical baseline among functional candidates.

What failed:
  FTF target fitting is not yet a stable optimizer; high local fit creates long-horizon drift.
  FTR-CG-small is too weak in this approximation.
  FNG does not close the KMNIST accuracy gap.

Next recommended direction:
  Add accepted-step / backtracking and cross-layer drift control to FTF before more seeds.
  In particular, target fitting should be sequential with validation of actual loss decrease
  and an activation/logit trust region that rejects or shrinks unsafe layer updates.
```


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


## 2026-05-03 DG-KAN v4.3 Functional Update Deep Redesign

P0/P1/P2 completed. No P2 survivor; P3-P8 not run by gate. See `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_结果复盘.md`.


## 2026-05-03 DG-KAN v4.4 Functional Optimizer Redesign

Implemented and ran `experiments/run_gafu_v44.py` and `experiments/analyze_gafu_v44.py`.

Summary:

```text
P0 rows=33, errors=0, pass=True
P1 survivors=none
P2 survivors=none
Decision=stop after P2 by written gate
```

Result replay:

```text
docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_结果复盘.md
results/v4_4/aggregate_decision.json
```


## 2026-05-03 DG-KAN v4.5 Accelerated Functional Optimizer

Implemented and ran `experiments/run_gafu_v45.py` and `experiments/analyze_gafu_v45.py`.

Summary:

```text
P0 rows=24, errors=0, pass=True
P1 rows=60, errors=0
P2 rows=90, errors=0
P2 survivors=none
Decision=stop after P2 by written gate
```

Result replay:

```text
docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_结果复盘.md
results/v4_5/aggregate_decision.json
```


## 2026-05-03 DG-KAN v4.6 Functional Learning Dynamics

Implemented and ran `experiments/run_gafu_v46.py` and `experiments/analyze_gafu_v46.py`.

Summary:

```text
P0 rows=60, errors=0, pass=True
P1 rows=45, errors=0
P2 rows=90, errors=0
P3 rows=45, errors=0
P2 survivors=none
P3 survivors=none
Decision=stop_after_p3_no_survivor
```

Result replay:

```text
docs/DG-KAN_v4.6_FunctionalLearningDynamics_结果复盘.md
results/v4_6/aggregate_decision.json
```


## 2026-05-03 DG-KAN v4.7 PSFT 实验

结果见 `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_结果复盘.md`。

P0 pass=True; P1 survivors=none; P2 survivors=none.


## 2026-05-03 DG-KAN v4.8 FGF/NFS 实验

结果见 `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_结果复盘.md`。

P0 pass=True; P1 survivors=none; P2 survivors=['TeacherA-AdamW20|NFS-role-block|diag|0.035', 'TeacherA-AdamW20|NFS-role-block|diag|0.05', 'TeacherA-AdamW20|NFS-role-block|cg5|0.035', 'TeacherA-AdamW20|NFS-role-block|cg5|0.05', 'TeacherA-AdamW20|NFS-role-block|cg10|0.05', 'TeacherA-AdamW20|NFS-role-block|lowrank32|0.05', 'TeacherB-AdamW100|NFS-Z|diag|0.02', 'TeacherB-AdamW100|NFS-Z|diag|0.035', 'TeacherB-AdamW100|NFS-Z|diag|0.05', 'TeacherB-AdamW100|NFS-Z|cg5|0.02', 'TeacherB-AdamW100|NFS-Z|cg5|0.035', 'TeacherB-AdamW100|NFS-Z|cg5|0.05', 'TeacherB-AdamW100|NFS-Z|cg10|0.02', 'TeacherB-AdamW100|NFS-Z|cg10|0.035', 'TeacherB-AdamW100|NFS-Z|cg10|0.05', 'TeacherB-AdamW100|NFS-Z|lowrank32|0.02', 'TeacherB-AdamW100|NFS-Z|lowrank32|0.035', 'TeacherB-AdamW100|NFS-Z|lowrank32|0.05', 'TeacherB-AdamW100|NFS-Z|lowrank64|0.035', 'TeacherB-AdamW100|NFS-Z|lowrank64|0.05', 'TeacherB-AdamW100|NFS-H|diag|0.02', 'TeacherB-AdamW100|NFS-H|cg5|0.02', 'TeacherB-AdamW100|NFS-H|cg10|0.02', 'TeacherB-AdamW100|NFS-H|lowrank32|0.02', 'TeacherB-AdamW100|NFS-H|lowrank32|0.035', 'TeacherB-AdamW100|NFS-H|lowrank64|0.035', 'TeacherB-AdamW100|NFS-H|lowrank64|0.05', 'TeacherB-AdamW100|NFS-ZH|diag|0.02', 'TeacherB-AdamW100|NFS-ZH|diag|0.035', 'TeacherB-AdamW100|NFS-ZH|diag|0.05', 'TeacherB-AdamW100|NFS-ZH|cg5|0.02', 'TeacherB-AdamW100|NFS-ZH|cg5|0.035', 'TeacherB-AdamW100|NFS-ZH|cg5|0.05', 'TeacherB-AdamW100|NFS-ZH|cg10|0.02', 'TeacherB-AdamW100|NFS-ZH|cg10|0.035', 'TeacherB-AdamW100|NFS-ZH|cg10|0.05', 'TeacherB-AdamW100|NFS-ZH|lowrank32|0.02', 'TeacherB-AdamW100|NFS-ZH|lowrank32|0.035', 'TeacherB-AdamW100|NFS-ZH|lowrank32|0.05', 'TeacherB-AdamW100|NFS-ZH|lowrank64|0.035', 'TeacherB-AdamW100|NFS-ZH|lowrank64|0.05', 'TeacherB-AdamW100|NFS-ZH-margin|diag|0.02', 'TeacherB-AdamW100|NFS-ZH-margin|diag|0.035', 'TeacherB-AdamW100|NFS-ZH-margin|diag|0.05', 'TeacherB-AdamW100|NFS-ZH-margin|cg5|0.02', 'TeacherB-AdamW100|NFS-ZH-margin|cg5|0.035', 'TeacherB-AdamW100|NFS-ZH-margin|cg5|0.05', 'TeacherB-AdamW100|NFS-ZH-margin|cg10|0.02', 'TeacherB-AdamW100|NFS-ZH-margin|cg10|0.035', 'TeacherB-AdamW100|NFS-ZH-margin|cg10|0.05', 'TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.02', 'TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.035', 'TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.05', 'TeacherB-AdamW100|NFS-ZH-margin|lowrank64|0.035', 'TeacherB-AdamW100|NFS-ZH-margin|lowrank64|0.05', 'TeacherB-AdamW100|NFS-role-block|cg10|0.02', 'TeacherB-AdamW100|NFS-role-block|lowrank32|0.02', 'TeacherB-AdamW100|NFS-role-block|lowrank64|0.035', 'TeacherB-AdamW100|NFS-role-cycle|diag|0.035', 'TeacherB-AdamW100|NFS-role-cycle|diag|0.05', 'TeacherB-AdamW100|NFS-role-cycle|cg5|0.035', 'TeacherB-AdamW100|NFS-role-cycle|cg5|0.05', 'TeacherB-AdamW100|NFS-role-cycle|cg10|0.05', 'TeacherB-AdamW100|NFS-role-cycle|lowrank32|0.05', 'TeacherC-AdamW-final|NFS-role-block|diag|0.035', 'TeacherC-AdamW-final|NFS-role-block|diag|0.05', 'TeacherC-AdamW-final|NFS-role-block|cg5|0.02', 'TeacherC-AdamW-final|NFS-role-block|cg5|0.035', 'TeacherC-AdamW-final|NFS-role-block|cg5|0.05', 'TeacherC-AdamW-final|NFS-role-block|cg10|0.05', 'TeacherC-AdamW-final|NFS-role-block|lowrank32|0.05', 'TeacherD-bestFGF|NFS-role-block|diag|0.035', 'TeacherD-bestFGF|NFS-role-block|diag|0.05', 'TeacherD-bestFGF|NFS-role-block|cg5|0.02', 'TeacherD-bestFGF|NFS-role-block|cg5|0.035', 'TeacherD-bestFGF|NFS-role-block|cg5|0.05', 'TeacherD-bestFGF|NFS-role-block|cg10|0.05', 'TeacherD-bestFGF|NFS-role-block|lowrank32|0.05']; P6 survivors=none; decision=stop_after_p5_no_multicycle_survivor.

## 2026-05-03 DG-KAN v4.9 RBF 参数化与 Exact NFS

本轮依据 `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`，实现并运行：

```text
P0 code audit: 21 rows, errors=0, pass=True
P1 architecture frontier: success rows=45
P2 eigenmode audit: rows=270
P3 exact NFS projection: rows=45, exact survivors=0
P6 compact capacity/basis expansion: rows=24
```

核心结论：

```text
1. AB-RBF 是 architecture-positive：ABRBF-silu 在 MNIST/Fashion/KMNIST 都提升 mean acc。
2. Eigenmode audit 支持 RBF-only bottleneck：ABRBF-linear 把约 24%-26% coeff energy 放入 base modes，并降低 high Sobolev-mode fraction。
3. Exact NFS KKT residual 很小，但 geometry reduction 近 0；当前严格 Jacobian-nullspace 太保守。
4. Heuristic NFS 仍有局部 smoothing 信号，但不能作为 exact nullspace 证明。
5. P4/P5 未运行：无 exact-NFS all-dataset survivor。
```

最终状态：

```text
stop_after_p3_no_exact_nfs_survivor
```

## 2026-05-03 DG-KAN v5.0 Edge-Decomposed Functional Training

```text
P0 pass: True
P1 architecture survivors: PureKAN-ABRBF-linear+silu-AdamW, PureKAN-ABRBF-linear-AdamW, PureKAN-ABRBF-linear-learnWidth-AdamW, PureKAN-ABRBF-silu-AdamW
P3 split-functional survivors: ABRBF-allAdamW-edgeOnly, ABRBF-baseAdam-rbfD6, ABRBF-baseAdam-rbfFCAdam-dataSob, ABRBF-baseAdam-rbfRelaxedNFSRefresh, ABRBF-baseAdam-rbfUFULL, ABRBF-baseFCAdam-rbfUFULL, ABRBF-baseOnlyAdam-rbfFrozen
P4 relaxed-NFS survivors: none
Final status: stop_after_p4_no_relaxed_nfs_survivor
```

主结论：AB-RBF / edge decomposition 是 architecture-positive；base path 必须承担任务学习动力学。Relaxed NFS 的投影不再为零，但 residual geometry gain 仍太小，P5/P6 不展开。
- v5.1 AB-RBF strong residual smoothing: P3 found all-dataset single-step smoothing survivors; P4 failed multi-step task/holdout gate, so stopped before P5/P6 confirm.
- v5.2 memory-budgeted residual smoothing: P1/P2 light smoothing passed, P3 multicycle failed; stopped before confirm seeds.
- v5.3 AB-RBF event controller: status=stop_after_p3_no_one_cycle_survivor; P3 survivors=0, P4 survivors=0.
- v5.4 efficiency-first PureKAN: status=stop_after_p3_no_efficiency_accuracy_survivor; P1 survivors=0, P3 survivors=0.

- v5.5 kernel-first efficient PureKAN: status=stop_after_p5_no_joint_accuracy_efficiency_survivor; P2=1 DWM survivors, P3=2 CP survivors, P4=1 Rational kernel survivors, P5 survivors=0.

- v5.6 GEMM-native efficient PureKAN: status=stop_after_p4_no_recipe_survivor; P1 exploratory=1, P5 survivors=0.

- v5.7 kernel-verified efficient PureKAN: status=stop_after_p1_no_efficiency_survivor; P1 exploratory=0, P5 survivors=0.

- v5.8 EfficientPrimitive_Redesign: stop_after_p1_no_efficiency_survivor (E_no_efficient_primitive); artifacts in `results/v5_8/`.

- v5.9 MemoryFirst_EfficientPrimitive: stop_after_p1_no_efficiency_survivor (C_no_p1_efficiency_survivor); artifacts in `results/v5_9/`.

- v6.0 EfficientFunctionalPureKAN_Acceleration: stop_after_p2_no_efficiency_candidate (A_no_p1_p2_efficiency_candidate); artifacts in `results/v6_0/`.

- v6.1 GraphFreeAnalyticAdjoint: stop_after_p2_no_graphfree_efficiency_survivor (D_no_primitive_passed_p2_efficiency); artifacts in `results/v6_1/`.

- v6.2 GraphFreeKernelCache: stop_after_p4_no_nearmiss_convergence_survivor (B_nearmiss_no_convergence_compensation); artifacts in `results/v6_2/`.

- v6.3 GraphFreeFusedKernel_Acceleration: stop_after_p6_no_functional_survivor (A_kernel_acceleration_positive_functional_not_ready); artifacts in `results/v6_3/`.

- v6.4 GraphFreeFunctionalCorrection: continue_after_p8_final_confirm_positive (R2_efficiency_task_correction_confirm_positive); artifacts in `results/v6_4/`.

- v6.5 FinalizationFunctionalCorrection: complete_task_learner_correction_optional (B_task_only_or_optional_correction); artifacts in `results/v6_5/`.
- v6.6 FinalValidation/Scaling: `complete_after_p4_final_validation_positive_cifar_open` (B_final_confirmed_cifar_small_blocker); candidate `DWM2-poly2-task+correction-best`. Artifacts in `results/v6_6/`.
- v6.6 HardThreshold FinalValidation: `complete_after_hard_threshold_final_validation` (A_final_system_confirmed_meaningful_hardgate); candidate `DWM2-poly2-task+correction-best`. Artifacts in `results/v6_6/`.
- v7.0 NextGen BeyondMLP: `complete_after_p10_nextgen_beyond_mlp_positive` (Route_A_nextgen_strong_system_confirmed); candidate `PatchKAN-tokenMix-mixedDegree+adaptiveCorrection`. Artifacts in `results/v7_0/`.
- v7.1 BeyondMLP StrongScaling: `stop_raw_confirm_unstable` (Route_C_raw_confirm_unstable); candidate `PatchKAN-tokenMix-mixedDegree+adaptiveCorrection`. Artifacts in `results/v7_1/`.
- v7.2 FullScaling StrongImplementation: `complete_after_p10_full_scaling_positive` (Route_A_full_scaling_strong_implementation_positive); candidate `PatchKAN-tokenMix-mixedDegree+adaptiveCorrection-stronggate`. Artifacts in `results/v7_2/`.
- v7.3 ProductionFullScaling NextGen: `complete_after_p11_production_nextgen_positive` (Route_A_production_verified_nextgen_patchkan_confirmed); default `PatchKAN-tokenMix-mixedDegree-productionFused-stronggate`. Artifacts in `results/v7_3/`.

- v7.4 FullVision ConvNetGap: `complete_after_p12_convnet_gap_substantially_closed` / `Route_A_convnet_gap_substantially_closed`.

- v7.4 FullVision ConvNetGap: `complete_after_p12_convnet_gap_substantially_closed` / `Route_A_convnet_gap_substantially_closed`.

- v8.0 FullVision ConvNetGap: `complete_after_p10_full_external_baseline_positive` / `Route_A_full_external_baseline_positive`.

- v8.0 FullVision ConvNetGap: `complete_after_p10_full_external_baseline_positive` / `Route_A_full_external_baseline_positive`.

- v8.0 FullVision ConvNetGap: `complete_after_p10_full_external_baseline_positive` / `Route_A_full_external_baseline_positive`.

- v8.1 ConvKAN-Inspired FullVision: `stop_after_p10_convkan_inspired_partial` / `Route_B_convkan_inspired_needs_repair`.

- v8.1 ConvKAN-Inspired FullVision: `stop_after_p10_convkan_inspired_partial` / `Route_B_convkan_inspired_needs_repair`.

- v8.1 ConvKAN-Inspired FullVision: `complete_after_p10_convkan_inspired_gap_push_positive` / `Route_A_convkan_inspired_level3_stronger_level4_partial`.

- v8.2 Level4 ConvNetReplacement: `complete_after_p10_level4_credible_positive` / `Route_A_level4_credible_efficient_functional_vision`.

- v8.3 Level4 Hardening: `complete_after_p10_level4_hardened_positive` / `Route_A_level4_hardened_achieved`.

- v9.1 Multi-Architecture ImageNet Frontier: `complete_after_p10_multiarchitecture_frontier_positive` / `Route_D_hybrid_local_global_functional_backbone_wins`.
