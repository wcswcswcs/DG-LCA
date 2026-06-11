# DG-KAN v17.0.1 实验结果复盘

生成时间：2026-06-02 17:09:10 +08

## Route

- route: `R8-CarrierSpecificPartialPositive`
- route_detail: fresh h100/h800/h1600 screen found partial positive source, but S4 real-transfer and S5 official gates were not executed/passed
- S0 pass: 1
- measured mechanism jobs: 945/945
- best h100 source vs best control: 0.11567223072052002
- best h800 source vs best control: 0.6183267831802368
- best h1600 source vs best control: 0.3805924654006958
- h800 valid summary rows: 105
- h1600 valid summary rows: 105
- measured horizon rows: 945
- 注：route 中 best source 字段是单个 fresh row 的最大值；h800/h1600 表格是 carrier x mechanism 分组后的 9-row 均值排行。

## 计划覆盖边界

- 已完成：S0 import/LineC/update/kernel correctness gate、full carrier x mechanism h100 screen、full 945-row h800/h1600 extension、mandatory exploration efficiency census、required/forbidden/no-action audits、v17.0.1 code review packet、4GPU shard execution记录。
- 已修复/补齐：v17.0.1 runner compatibility shims、`dgkan.diagnostics.linec` 稳定入口、FU update semantics shims、efficiency package shims、`loss_interface`、manual optimizer、matched controls；M2 SGD/Momentum branch 使用 `.step_gradient()`；basis repair 从 alias reference path 改为 family-specific torch repair path。
- 未完成/不可冒充：official fused CUDA/C++ basis kernels、S4 real-transfer exploration、S5 official success gate、真正动态 work-stealing queue。本次 4GPU 为 shard queue drain，并记录 idle/dashboard；不能写成 promotion-ready official result。

## 关键实验数据

| carrier | mechanism | dataset | seed | source_h100 | final_val_loss | LineC valid | CouplingR2 |
|---|---|---|---:|---:|---:|---:|---:|
| MLP | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 0 | 0.11567223072052002 | 1.1136894226074219 | 1 | 0.09645487360629446 |
| MLP | M1-AdamWPrimaryFUResidual | MNIST | 0 | 0.06858468055725098 | 1.3802064657211304 | 1 | 0.08413871879275892 |
| D-RAT | M1-AdamWPrimaryFUResidual | KMNIST | 2 | 0.04729962348937988 | 1.975417137145996 | 1 | 0.09557361833714939 |
| D-CHE | M1-AdamWPrimaryFUResidual | MNIST | 2 | 0.041661977767944336 | 1.7515848875045776 | 1 | 0.0 |
| D-FOU | M1-AdamWPrimaryFUResidual | MNIST | 2 | 0.03851354122161865 | 1.7809066772460938 | 1 | 0.0 |
| D-RBF | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 2 | 0.03650557994842529 | 1.9294363260269165 | 1 | 0.006222984902326312 |
| LQ | M1-AdamWPrimaryFUResidual | MNIST | 2 | 0.03245246410369873 | 1.9229087829589844 | 1 | 0.0 |
| D-RAT | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 1 | 0.028737664222717285 | 1.6809548139572144 | 1 | 0.0 |
| LQ | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 1 | 0.026569724082946777 | 1.5633560419082642 | 1 | 0.0 |
| D-RBF | M1-AdamWPrimaryFUResidual | Fashion-MNIST | 1 | 0.02381765842437744 | 1.7538894414901733 | 1 | 0.0002541850130547557 |
| D-CHE | M1-AdamWPrimaryFUResidual | MNIST | 0 | 0.023011207580566406 | 1.8814640045166016 | 1 | 0.12088389164179958 |
| D-RBF | M1-AdamWPrimaryFUResidual | KMNIST | 2 | 0.01926732063293457 | 2.0813286304473877 | 1 | 0.005831970768654993 |

## h800 / h1600 Fresh Extension

| horizon | carrier | mechanism | valid_rows | source_vs_best_control | route |
|---|---|---|---:|---:|---|
| h800 | MLP | M2-SGDMomentumPrimaryFU | 9 | 0.1751598914464315 | measured |
| h800 | D-CHE | M1-AdamWPrimaryFUResidual | 9 | -0.002686844931708442 | measured |
| h800 | D-RBF | CTRL-RecoveryOnly | 9 | -0.007917450533972846 | measured |
| h800 | D-WAV | CTRL-AdamW | 9 | -0.009836514790852865 | measured |
| h800 | D-CHE | CTRL-AdamW | 9 | -0.012133015526665581 | measured |
| h800 | LQ | CTRL-RecoveryOnly | 9 | -0.013121240668826632 | measured |
| h800 | D-WAV | CTRL-RecoveryOnly | 9 | -0.0134863191180759 | measured |
| h800 | D-RBF | CTRL-AdamW | 9 | -0.013663775391048856 | measured |
| h800 | D-FOU | CTRL-RecoveryOnly | 9 | -0.014129738012949625 | measured |
| h800 | D-FOU | M1-AdamWPrimaryFUResidual | 9 | -0.01462852292590671 | measured |
| h1600 | D-CHE | M1-AdamWPrimaryFUResidual | 9 | 0.0032123857074313695 | measured |
| h1600 | D-RBF | M1-AdamWPrimaryFUResidual | 9 | -0.00665075249142117 | measured |
| h1600 | D-WAV | CTRL-RecoveryOnly | 9 | -0.010287218623691134 | measured |
| h1600 | LQ | CTRL-RecoveryOnly | 9 | -0.011402944723765055 | measured |
| h1600 | LQ | M1-AdamWPrimaryFUResidual | 9 | -0.012305027908749051 | measured |
| h1600 | D-WAV | CTRL-AdamW | 9 | -0.013830310768551297 | measured |
| h1600 | MLP | CTRL-SGD | 9 | -0.0138778289159139 | measured |
| h1600 | D-FOU | CTRL-RecoveryOnly | 9 | -0.01632106304168701 | measured |
| h1600 | D-RBF | CTRL-AdamW | 9 | -0.016780740684933133 | measured |
| h1600 | D-WAV | M1-AdamWPrimaryFUResidual | 9 | -0.017111447122361925 | measured |

## Efficiency 证据

| carrier | batch | forward_ratio | backward_ratio | update_ratio | step_ratio | memory_ratio | exploration |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 32 | 1.2782815884103762 | 0.7116976171039081 | 1.6367665536866336 | 0.7517668432683838 | 1.0 | 1 |
| MLP | 8 | 1.3482550101430362 | 0.836662996082678 | 1.493936120283379 | 0.8671211975189186 | 1.0 | 1 |
| MLP | 64 | 1.265791855950648 | 0.8422979313040029 | 1.6268588809944173 | 0.8788694220829236 | 1.0 | 1 |
| D-FOU | 64 | 5.918944339046272 | 1.2701542608646275 | 0.9265370485894503 | 1.4379371061544173 | 1.0079650716856452 | 0 |
| LQ | 64 | 5.6211909471574995 | 1.3413882840627298 | 0.886082237847753 | 1.4889874674247376 | 1.004395539559856 | 0 |
| D-CHE | 32 | 4.667403280363164 | 1.477187405191999 | 0.8715280993419363 | 1.569783574310247 | 1.002246327550025 | 0 |
| LQ | 32 | 5.718961310106349 | 1.563883569865485 | 0.7699964333726577 | 1.6941717463158998 | 1.002246327550025 | 0 |
| D-CHE | 64 | 5.865595452323246 | 1.6248073917668941 | 1.1510945766843261 | 1.7751344087640657 | 1.004395539559856 | 0 |
| LQ | 8 | 6.225778565887176 | 1.6134137375245337 | 1.0844540923823438 | 1.7877074857040114 | 1.001449318228874 | 0 |
| D-RAT | 32 | 6.51242794498445 | 1.6619120689529976 | 0.8895182351143692 | 1.8205419420139273 | 1.0032808205006947 | 0 |
| D-CHE | 8 | 6.25402677255324 | 1.7172356670945113 | 1.463114931511085 | 1.8874998822100781 | 1.001449318228874 | 0 |
| D-RBF | 64 | 9.7627226645097 | 1.9250483018999813 | 0.9867151259789458 | 2.1956891685585793 | 1.0043660392943536 | 0 |

## Kernel 证据

| family | forward_relerr | grad_relerr | grad_cosine | dense_materialized | exploration_pass |
|---|---:|---:|---:|---:|---:|
| D-CHE | 0.0 | 0.0 | 0.9999999999999999 | 1 | 1 |
| D-FOU | 0.0 | 1.0432822587251365e-16 | 0.9999999999999998 | 1 | 1 |
| LQ | 0.0 | 0.0 | 1.0 | 1 | 1 |
| D-RAT | 0.0 | 0.0 | 1.0 | 1 | 1 |
| D-RBF | 0.0 | 0.0 | 0.9999999999999999 | 0 | 1 |
| D-WAV | 0.0 | 8.432204087278424e-17 | 1.0 | 0 | 1 |

## 修改记录（便于审计）

- 新增 `dgkan/metrics/linec.py`：LineC measurement invalid 独立 route，不把异常当 geometry fail。
- 新增 `dgkan/fu/core.py` 与 `dgkan/fu/mechanisms.py`：统一 UpdateTensor kind/sign/space/source，并实现 AdamW-free、slow-state、matrix-block、PopRisk/SNR、role-partition smoke。
- 新增 `dgkan/fu/loss_interface.py`：显式记录 CE/Brier train-stream cotangent readback，禁止 audit metrics 进入 direction source。
- 新增 `dgkan/fu/optimizers.py` 与 `dgkan/fu/controls.py`：manual SGD/Momentum optimizer 与 matched NoOp/Random controls；修复 M2 branch 的 `.step_gradient()` 调用。
- 新增 `dgkan/profiling/efficiency_v17.py`：每个 phase 使用 cloned state，training cost 与 audit cost 分离。
- 新增/修改 `dgkan/kernels/v17_*`：family kernel correctness/readback 入口；corrective pass 将 repaired path 改为 torch family-specific repair，但 `official_fused_kernel_complete=0`。
- 新增 `experiments/run_v17_common.py` 及 v170/v171/v172/v173/finalize 包装入口。

## 分析与 insight

- S0 的价值在于先确认 import closure、LineC golden、update sign、kernel gradcheck 是否可信；若这里失败，不能写 scientific no-go。
- corrective pass 的机制矩阵与 h800/h1600 extension 都是 fresh real-dataset rows；这解决了之前 low-budget/top-only 违背计划覆盖的问题。
- AdamW overwrite diagnostic 提供了 FU 是否被 AdamW 抵消的证据链：`cos_FU_AdamW` 与 `optimizer_overwrite_projection` 为后续 AdamWWashesFU/IntrinsicNoSource 分类服务。
- Efficiency rows 分离了 `step_training_only_ms` 和 `linec_audit_ms`，避免把 audit readback 误计入 carrier 训练效率。
- 由于 S4/S5 未执行且 official fused kernel 未完成，route 最高只能是 partial/exploration 级别；任何 promotion_allowed=1 都必须等 real-transfer official rows 和 official efficiency/kernel gates 真实通过。
