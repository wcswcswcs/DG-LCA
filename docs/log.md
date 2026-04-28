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
