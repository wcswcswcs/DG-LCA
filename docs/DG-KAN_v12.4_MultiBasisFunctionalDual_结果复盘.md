# DG-KAN v12.4 Multi-Basis Functional Dual 结果复盘

> 本复盘记录 `DG-KAN_v12.4_多基函数效率优先与Functional双线计划.md` 的真实执行与连续修复结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R3-ExpressionPassTaskFail
latest_targeted_route = R1-EfficiencyAllFail
base_blocker_route = task_fail_or_latest_efficiency_all_fail
base_qualified = False
functional_open = False
functional_diagnostic_positive = False
next_recommended_action = lower-level fused CUDA/Triton kernel or new task-geometry primitive; no gate lowering
```

最新正式 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b47_manualstep_20260520T233000Z
```

目标达成状态：

```text
v12.4 pipeline = achieved after repairs
A1 efficiency screen = achieved for many primitive candidates
A2 official expression qualification = achieved again by B21/B22/B23/B27 basis-specific norm candidates
A3 task qualification = not achieved
Functional official route = not opened
Functional diagnostic = positive on some rows, but base gate closed
best meaningful task-gate measurement = B36 compile-warm audit / B42-B45 fast-reuse family, all still A3-fail
latest targeted repair artifact = B47 manual CE backward step, failed A1 efficiency gate
```

核心结论：

1. v12.4 已按计划从 v12.3 的 LQ frame 小修转向 primitive-level multi-basis screen，并读取 third_party 中可用的 RBF/FastKAN、B-spline、Rational/KAT 设计作为实现参考。
2. 初始 dense/autograd primitive 全部卡在 R1 efficiency；stream/recompute、compiled warm 与 compile memory accounting 修复后，A1 可以产生真实 survivor。
3. Signed-pair Poly2 B9 系列只修了局部 E1 frozen coverage，不能修 rotated/random quadratic coverage；因此继续做了 B10 fixed quadratic sketch diagnostic。
4. B10 scale repair 证明之前 B10 trainable 弱的一部分来自 output scale；h256/h512 rank repair 进一步证明 global quadratic coverage 可被低成本 fixed sketch 修复。
5. B11 official-capable trainable quadratic sketch 保留了 frozen coverage，但 A2 expression 仍未过；降低 task LR/weight decay 后也没有形成 base survivor。
6. B12 fan-scale 与 B13 identity-residual 初始化都按文档建议尝试：B12 主要破坏 expression，B13 提升 A1 但没有修复 interaction expression。
7. B14 gated Legendre+quadratic coupling 是本轮最大进展：B14b/B14d 一度打开 A2，task mean delta 转正；但 A3 仍因 val-loss AUC/time、near pass 或 ECE 子门失败。
8. B15 logit-gain 与 LR=4e-3/WD=5e-4 全局 optimizer repair 没有修复 A3，且部分指标变差；B14 历史 near-miss 的 hard blocker 是 `task_fail`。
9. 按“不同 basis 需要不同初始化/归一化”的思路，继续执行 B16 branch-output norm 与 B17 fixed block-ZCA/RMS input geometry norm；B17 明确改善初始 geometry condition/rank，但引入效率和 expression trade-off。
10. B18 lighter diagonal-shrink + group-RMS norm 也已执行；它比 B17 更轻，但没有改善初始 condition/rank，且 B18a-d 全部没过 A1。
11. B19 residual block-ZCA norm 继续执行：它把 B14 和 B17 折中，B19a/B19b 能过 A1 exploratory，但 A2 expression 仍失败。
12. B20 residual mix 小网格继续执行：mix=0.25 能过 A1，但 E1/E6/E8 仍不达标；mix=0.15 不稳定，未形成更好 base 候选。
13. 按“不同 basis 需要不同 norm”的进一步判断，B21 把 Legendre 分支的 bounded tanh norm 与 quadratic 分支的线性/残差几何 norm 分开，并给 quadratic projection 做 unlabeled feature-scale calibration。
14. B21d 是真实进展：A2 expression 通过，E1/E2/E6/E8 B1 delta 分别为 `-0.0038663744926452637` / `0.000898897647857666` / `0.0010586977005004883` / `0.0005333423614501953`，frozen R2 也全部超过 `0.89`。
15. 继续按 task-fail route 做 B22 branch-scale 初始化修复：B22b 把 task mean delta 推到 `-0.001736111111111111`、worst delta 推到 `-0.0234375`，但 near pass rate 仍只有 `0.6666666666666666`，AUC-time/ECE 子门仍失败。
16. B23 temperature repair 在 B24 的 compiled-task audit 中把 task accuracy 继续推近：B23b mean delta = `0.009982638888888888`，near pass rate = `0.8888888888888888`；但 AUC-time 与 ECE max 仍失败。
17. B24 compiled task path accounting 没有修好 AUC-time，反而使 AUC-time ratio 异常放大到 `224`-`280` 均值量级，因此不能作为过 gate 的生产路径。
18. B25 修复 AUC-time accounting：新增 steady-state epoch 口径后，B21a 的 AUC-time mean ratio 降到 `1.0968725773066839`，证明 B24 的 `224x+` 是测量/compile warmup artifact；但 B21a 仍因 near pass、AUC max 与 ECE 子门失败，不能写成 base success。
19. B26 继续按“不同初始化 + 低温”方向测试 plain B21a 低温变体；B26a/B26b 在本轮 A1 compiled step ratio 约 `2.0x`，没过 A1，因此没有形成 task 结论。
20. B27 按 R3 task-fail 建议加入全局 warmup-cosine LR schedule；第一次执行发现 B27 registration/compiled path 不完整，fixed run 后 B27a/B27b/B27c 全部进入 A1/A2/A3。
21. B27a/B27b/B27c 全部通过 A2 expression；B27c 的 task 指标最好，mean delta = `-0.006727430555555556`、AUC-time mean = `1.0237545198908917`、ECE mean delta = `-0.018747775091065302`，但 worst delta = `-0.02734375`、failure-table near pass rate = `0.4444444444444444`、AUC-time max = `1.166639746619922`，A3 仍失败。
22. B28/B29 继续按 B27 的 blocker 做 schedule-shape 与 h224 轻量化修复，但 targeted artifacts 全部卡在 A1 efficiency；B29+TF32 的 compiled step ratio 仍为 `2.2507559698581194` / `2.314790957435887`。
23. B30 把 B21/B23/B27 的分支专属 norm/temperature/schedule 迁移到低成本 `LiteGatedLegendreQuadraticKAN`：`B30b` 过 A1 且 task mean delta = `0.0015190972222222222`，但 A2 expression 失败，E1/E6/E8 B1 delta 为 `-0.05942237377166748` / `-0.047035396099090576` / `-0.05445504188537598`。
24. B31 继续验证“不同 basis 需要不同初始化”：quadboost/midboost 后 B31a/B31b/B31c 全部过 A1，B31a task mean delta = `0.008246527777777778`，但 direct-readout lite 结构仍未过 A2。
25. B32 回到 small GatedHybrid：h160/h192 重新打开 A1+A2；B32a 有 diagnostic functional positive（control gap `0.002688993613471302`），但 A3 task 失败。
26. B33 做 hidden/temperature 中间档，B33a/B33b/B33c 全部过 A1+A2；B33c 最接近 task gate，mean delta = `0.005208333333333333`、near pass = `0.7777777777777778`，但 AUC-time mean/max = `1.1852771821233192` / `1.5840523720855126`，且 ECE max = `0.02069440484046936`。
27. B34 尝试 h168 与 temp075+final050 的 AUC/task tradeoff，结果不如 B33c；B34b/B34c 仍停在 R3 task_fail。
28. B35 temp065 中间温度没有解决 A3：B35b 的 ECE 与 worst-row 较稳，但 near pass 只有 `0.5555555555555556`，AUC-time mean/max = `1.1104985285493005` / `1.7533748125949657`。
29. B36 direct Legendre edge-basis skip 是结构性合并 B30 与 B32 的尝试；B36a/B36b/B36c 全部过 A1+A2，但 A3 仍失败。B36c mean delta = `0.005208333333333333`，worst = `-0.01171875`，near = `0.6666666666666666`，AUC-time max = `2.1617626226569753`。
30. 继续按 R3 建议尝试 B37 branchslow optimizer profile：只 B37b 过 A1/A2，但 task 明显变差，mean delta = `-0.03624131944444445`，near = `0.1111111111111111`，ECE max = `0.08234763890504837`。
31. 反向尝试 B38 leg/direct-fast optimizer profile：B38a/B38b/B38c 全部停在 R1 efficiency，compiled step ratio = `2.3515869810362546` / `2.422857574375153` / `2.1287205147748334`。
32. B36 compile-warm audit 把 `task_compile_warmup_steps` 从 `1` 提到 `12` 后，B36c 的 AUC mean/max 从 `1.1888670638347314` / `2.1617626226569753` 改为 `1.105288806659117` / `1.6440778149943576`，说明晚发 compile warmup 确有污染；但 near pass 仍是 `0.6666666666666666`，AUC max 仍远高于 `1.05`。
33. B39a 做了 B36c 的单点 schedule 修复（temp075/directskip/final050），但停在 R1，compiled step ratio = `2.2433983063496603`，不能进入 A2/A3。
34. B40 尝试 optimizer-step compile warmup + restore，停在 R1，compiled step ratio = `1.9544235620011423`；说明 optimizer step warmup 不是可用修复。
35. B41 h168 light directskip 仍停在 R1：B41a/B41b compiled step ratio = `2.004491` / `1.989416`，没有进入 task。
36. B42 fast-reuse 手写等价 forward path 通过数值等价审计（max abs error = `0.0`），并把 B42b 推到 A1+A2；但 B42b A3 task mean delta = `-0.009331597222222222`，near pass = `0.2222222222222222`，仍失败。
37. B43 h172 fast-reuse 容量/速度折中没有改善：B43b A1+A2 通过，但 task mean delta = `-0.011935763888888888`，worst delta = `-0.037109375`。
38. B44 input norm residual mix15 按“训练初期几何”方向执行，但 B44a/B44b 均停在 R1，compiled step ratio = `1.9539927866985813` / `1.9242401698113432`。
39. B45 保持 B42b norm/init/fast-reuse，只把 final cosine LR 改为 `0.50`；B45a 过 A1+A2，ECE mean delta 改善到 `-0.0027160458266735077`，但 task mean delta = `-0.009331597222222222`，near pass = `0.3333333333333333`，AUC/ECE hard gate 仍失败。
40. B46 把 directskip fixed scale 从 `0.25` 提到 `0.50`，但 B46a/B46b 均停在 R1，compiled step ratio = `1.9061898744500823` / `1.6355462392377917`。
41. Functional diagnostic 有 positive row，但不是在合格 base 上产生；latest B46 rows checked = `61`，fake/proxy/cpu = `0` / `0` / `0`。
42. B47 按推荐方向实现真正 manual backward/CE step：custom autograd 版梯度正确但 `torch.compile` 包装失败；manual CE step 版绕开 torch autograd graph，速度有改善，但仍停在 R1 efficiency。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 strict edge-basis primitive KAN：Activation/RBF/OrthogonalPolynomial/Fourier/Wavelet/BSpline/Rational/HybridPoly | 所有 learnable tensor 都是 edge basis 权重；fixed centers/normalizer 不按 dataset name 或 label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 stream basis mix path | 修复 R1 memory/temporary tensor pressure；不改变 gate，不加入 shortcut MLP。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `bspline_relu_combo`、`poly2_relu_combo`、`signed_pair_linear` init | 按 R2/R5 blocker 尝试更强 expression/interactions；初始化只用维度结构，不使用 label/outcome。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B9b/B9c/B9d Poly2 diagnostic candidates | 验证 rank-limited、minimal K2、pair-random projection 是否能修 efficiency/global interaction；均为 `diagnostic_only=1`。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `QuadraticSketchKAN` 与 B10a/B10b/B10c/B10d | 固定随机 projection + learnable quadratic readout，用来验证低成本 global quadratic coverage；projection 固定、无 label/outcome 分支，全部 `diagnostic_only=1`。 |
| `dgkan/models/fc_purekan_primitives.py` | B10 scale repair：移除 quadratic sketch forward 里的额外 `sqrt(hidden*K)` 除法 | 上一轮 B10 frozen ridge 高但 trainable R2 低，定位为 readout scale 过小；修复后按同一 budget 重跑。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B11a/B11b `trainable_quadratic_sketch` | 将 B10c 的 global quadratic coverage 转成 official-capable candidate；projection/readout 均可训练，不把 fixed-sketch diagnostic 写成 official base。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B12a/B12b `fan_scale_repair` 初始化 | 按 v12.4 R3/R5 建议尝试全局 basis/initialization repair；只改无标签尺度，不改变 gate。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B13a/B13b `identity_residual_scale` 初始化 | 按计划中的 identity residual scale 方向，把正交多项式的线性 channel 初始化成近似 identity hidden path；没有新增 MLP shortcut。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `GatedLegendreQuadraticKAN` 与 B14a/B14b | 按复盘建议把 B10/B11 的 quadratic coverage 与 B3b 的 Legendre trainability 做 official-capable gated coupling；projection/readout/branch scale 均为 learnable basis 参数。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B14c/B14d quad-boost 与 B15a/B15b logit-gain | B14 近门后继续做全局 branch-scale/logit-scale repair；没有按 dataset/label 分支，也没有改 gate。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B16a/B16b branch-output norm | 回应“不同基函数可能需要不同初始化”：用 unlabeled train stream 的 Legendre/quadratic 初始 logit std 分别校准 hybrid branch 输出，不看 label、不按 dataset 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B17a/B17b/B17c/B17d fixed input geometry norm | 回应“输入需要 norm 层”：实现 block-wise ZCA shrinkage whitening + per-sample RMS + tanh 截尾；是 fixed input normalizer，无 learnable LayerNorm/hidden stem。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B18/B19/B20 input norm repair family | 继续按同一 no-fake gate 测试 lighter group-RMS、residual block-ZCA、residual mix 小网格；均只用 unlabeled train-stream statistics。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B21a/B21b/B21c/B21d basis-specific input norm | Legendre 分支保留 bounded tanh 坐标，quadratic 分支使用独立线性/残差几何 norm 与 feature-scale calibration；修复 B17-B20 “同一 norm 同时喂给不同 basis” 的机制问题。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B22a/B22b branch-scale task repair | 保持 B21 分支专属 norm 与 feature calibration，只全局降低 quadratic branch 初始占比；用于修 A3 task gate，不看 dataset/label。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B23a/B23b logit temperature repair | 保持 B22b 结构，只把全局 logit gain 初始化为 `0.75`/`0.50`，用于修 early overconfidence/ECE；不改 loss。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B26a/B26b plain basis-specific low-temperature repair | 回到 B21a plain 分支专属 norm，不使用 B22b lowboost，只测试 `logit_gain=0.75/0.50` 是否能在不牺牲 expression 的情况下修 ECE/AUC；不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B27a/B27b/B27c warmup-cosine LR repair candidates | 保持 B21/B23 的 basis-specific input norm/temperature 机制，只改变全局 task LR schedule；用于按 R3 task-fail 建议测试 optimizer schedule，不看 dataset/label。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B28a/B28b schedule-shape repair candidates | 保持 B23b/B27c 的 lowboost/temp050 结构，把 schedule 改为 warmup10 + final LR `0.50/0.75`，尝试在不改数据、不改 gate 的前提下修 B27c 的 worst-row 与 AUC max。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B29a/B29b h224 schedule-shape repair candidates | 按 B28 的 R1 efficiency blocker 做轻量化，将 hidden 从 228 降到 224；仍使用同一 global schedule，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `LiteGatedLegendreQuadraticKAN` 与 B30/B31 | 将 B21/B23/B27 的分支专属 norm/temperature/schedule 迁移到低成本 direct Legendre readout + quadratic sketch；Legendre 使用 bounded/tanh norm，quadratic 使用 RMS-normalized linear coordinate 与无标签 feature-scale calibration。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B32/B33/B34/B35 small GatedHybrid repair family | 回到两层 Legendre 表达深度，同时压 hidden 到 160/168/176/192，并系统测试 temp050/temp065/temp075 与 final LR `0.50/0.75` 的 task/AUC tradeoff；均为全局候选，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B36 direct Legendre edge-basis skip | 给 small GatedHybrid 添加 direct Legendre edge-basis skip，尝试合并 B30/B31 的 early task geometry 与 B32/B33 的 expression depth；新增参数仍是 edge basis readout，不是 MLP shortcut。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B37/B38 branch-wise optimizer repair candidate ids | B37/B38 与 B36 使用相同 architecture / init，只通过全局 optimizer profile 测试 branch update scale；不按 dataset/label/outcome 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B39a temp075/directskip/final050 单点 schedule repair | 保持 B36c architecture/init，只把 final LR 从 `0.75` 改为 `0.50`；用于验证 AUC-time 是否由 final LR 过高导致。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 v12.4 A0/A1/A2/A3/B1 runner | 按 efficiency-first 执行；functional 在 base 未过时写 `diagnostic_base_not_qualified`。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 修复 compiled warm memory accounting | warmup 后 reset peak memory，避免 compile-time allocation 污染 architecture memory gate。 |
| `experiments/run_v124_multibasis_functional_dual.py` | official expression pass 排除 `diagnostic_only` candidate | 防止 B9/B10 低参或固定-sketch 诊断候选被误写成 base qualified。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 将 B11/B12/B13/B14/B15/B16/B17 加入 compiled repair ids | 确保 official-capable repair 与初始化/input-norm repair 都经过 clean/compiled efficiency accounting。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `v124_initial_geometry_audit.csv` | 所有候选即使未过 A1，也记录 MNIST unlabeled train slice 上的初始 condition/rank/entropy 和 norm 开关，避免只因 efficiency fail 就丢失 geometry 审计。 |
| `experiments/run_v124_multibasis_functional_dual.py` | initial geometry audit 增加 B21 norm/feature-scale flags | 记录 `basis_specific_quad_input_norm_enabled`、`quad_feature_norm_enabled`、`quad_feature_std_mean`，方便审计 B21 是否真的使用分支专属 norm。 |
| `experiments/run_v124_multibasis_functional_dual.py` | task triage 增加 compiled steady-state path audit | 对 compile-repair candidate 在 A3 使用 `torch.compile(..., mode="reduce-overhead")` 并 warm backward，记录 `compiled_task_path`；用于审计 A1 compiled timing 与 A3 AUC-time 是否一致。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `--task-timing-warmup-epochs` 与 steady AUC-time accounting | A3 同时记录 raw AUC-time 与跳过前 3 个 epoch 后的 steady AUC-time，避免把 compile/graph warmup outlier 当作 architecture task-time gate。 |
| `experiments/run_v124_multibasis_functional_dual.py` | compiled repair ids 加入 B26a/B26b | B26 与 B21/B22/B23 使用同一 clean/compiled efficiency accounting；未过 A1 时不推进 task，也不写 success。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 B27 task LR schedule 与 trace/triage 字段 | 对 B27 使用 20% linear warmup + cosine decay 到 `0.25 * base_lr`，并落盘 `lr_schedule` / `final_lr`，便于审计 LR schedule 是否真实生效。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 修复 B27 compiled repair registration | 首次 B27 执行暴露 B27 registration/compiled path 不完整；修复后 B27a/B27b/B27c 均进入 A1/A2/A3，不能把首次不完整 run 当科学结论。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `--candidate-ids` targeted repair filter | 只减少本轮 repair 候选集合，MLP control 始终保留；用于快速验证 B28/B29，不改变任何 gate。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 B28/B29 schedule variants | B28/B29 使用 warmup10 + cosine final `0.50/0.75`，并继续落盘 `lr_schedule` / `final_lr`。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 启用 CUDA `torch.set_float32_matmul_precision("high")` 系统优化尝试 | 按 R1 efficiency blocker 尝试 matmul path 优化；结果显示 B29+TF32 仍未过 A1，因此不能作为效率成功。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 接入 B30-B36 candidate ids、compiled repair ids 与 schedule routing | 所有新增候选都走同一 A1/A2/A3/function diagnostic/no-fake pipeline；失败时仍按 route 写 R2/R3/R5，不打开 official functional。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `--task-compile-warmup-steps` 与 B37/B38 optimizer profile | 用真实 warmup fwd/bwd 消除 compiled task path 晚发编译尖峰，并记录 `optimizer_profile` / group lr；不改变 A3 gate。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 接入 B39a schedule routing 与 compiled repair id | 单点 targeted repair 仍走 A1 gate；未过 A1 时不推进 A2/A3，不写 task success。 |
| `third_party/MJKAN` | 作为 RBF/FastKAN basis 设计来源读取 | 没有直接运行其 Colab 脚本，也没有引入 base update MLP shortcut。 |
| `third_party/KANbeFair` | 作为 local B-spline basis 设计来源读取 | 本轮只实现 order1 local basis，不使用 shortcut/base_fun。 |
| `third_party/rational_kat_cu` | 作为 Rational/KAT safe denominator 设计来源读取 | 本轮使用 torch-safe rational lite diagnostic，未依赖 CUDA extension 编译。 |

本轮没有做：

```text
1. 没有调低 base 或 functional gate。
2. 没有按 MNIST/Fashion/KMNIST 名称分支。
3. 没有 teacher/distillation/loss modification/class weights。
4. 没有把 autograd-only diagnostic 写成 manual/fused-kernel final claim。
5. 没有把 diagnostic_only 低参或 fixed-sketch 候选写成 official base success。
```

编译验证：

```bash
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v124_multibasis_functional_dual.py
```

## 2. 执行链

| run | artifact | route | 结论 |
|---|---|---|---|
| 初始 multi-basis | `v124_multibasis_functional_dual_20260519T170000Z` | `R1-EfficiencyAllFail` | dense/autograd primitive 太慢，没有 A1 survivor。 |
| stream/recompute repair | `v124_multibasis_functional_dual_repaired_20260519T180000Z` | `R1-EfficiencyAllFail` | memory 有改善，但 step ratio 仍超阈值。 |
| compile memory fix | `v124_multibasis_functional_dual_compile_memfix_20260519T200000Z` | `R2-EfficiencyPassExpressionFail` | A1 开始有 survivor，expression 失败。 |
| expression budget check | `v124_multibasis_functional_dual_expression_budget_20260519T210000Z` | `R2-EfficiencyPassExpressionFail` | B6a 长预算仍无法通过 E1/E6/E8 与 task。 |
| K8 + hybrid repair | `v124_multibasis_functional_dual_r2repair_20260519T220000Z` | `R5-FunctionalDiagnosticPositiveBaseNotYetQualified` | B8d 表达最好但仍不过 gate；B2a 有一次 diagnostic functional positive。 |
| B9a/B9b/B9c/B9d | `v124_multibasis_functional_dual_pairrandom_poly2_20260519T235500Z` 等 | `R2/R5` | B9 只修 E1 局部 coverage，E6/E8 和 task 未过。 |
| B10 h64/h128 | `v124_multibasis_functional_dual_quadratic_sketch_20260520T001000Z` | `R5` | fixed sketch 证明 E6/E8 frozen coverage 可修，但 trainable/task 仍不足。 |
| B10 scale repair | `v124_multibasis_functional_dual_quadratic_sketch_scale_repair_20260520T003000Z` | `R5` | 移除额外 scale 后 B10b trainable 提升，但仍不过 gate。 |
| B10 h256 | `v124_multibasis_functional_dual_quadratic_sketch_h256_20260520T005000Z` | `R5` | B10c/h256 同时提高 coverage 与 task，但 task 仍差 `-0.0697`。 |
| B10 h512 | `v124_multibasis_functional_dual_quadratic_sketch_h512_20260520T012000Z` | `R5` | h512 不再改善 h256；base/functional official 继续关闭。 |
| B11 trainable quadratic sketch | `v124_multibasis_functional_dual_trainable_quadratic_sketch_20260520T020000Z` | `R5` | B11 保留 coverage 但不过 A2；B3b-Legendre 过 A2 后 task 崩。 |
| B11 LR repair | `v124_multibasis_functional_dual_trainable_quadratic_sketch_lr1e3_20260520T030000Z` | `R5` | 降低 task LR/weight decay 没有修复 B11 expression，也没有 base survivor。 |
| B12 fan-scale init | `v124_multibasis_functional_dual_legendre_fanscale_20260520T040000Z` | `R5` | fan-scale 只修部分 A1，expression 明显变差。 |
| B13 identity residual init | `v124_multibasis_functional_dual_identity_residual_20260520T050000Z` | `R5` | identity path 让 A1 更稳，但 A2/A3 仍停在 B3b task fail。 |
| B14 gated coupling | `v124_multibasis_functional_dual_gated_legendre_quadratic_20260520T060000Z` | `R5` | 首次把 gated Legendre+quadratic 推到近 A2；B14b B1 E1/E8 只差约 0.003/0.0014。 |
| B14 quad-boost | `v124_multibasis_functional_dual_gated_quadboost_20260520T070000Z` | `R5` | B14a/B14b/B14d 过 A2，task mean delta 转正，但 AUC/time 子门失败。 |
| B15 loss-gain | `v124_multibasis_functional_dual_gated_lossgain_20260520T080000Z` | `R5` | logit gain 没修好 A3，反而只剩 B14b 过 A2。 |
| B14 global LR repair | `v124_multibasis_functional_dual_gated_lr4e3_20260520T090000Z` | `R5` | LR=4e-3/WD=5e-4 未修复 AUC/time，B14b 仍 task_fail。 |
| B16 branch-output norm | `v124_multibasis_functional_dual_basisnorm_20260520T100000Z` | `R2` | branch std 校准没有形成 A2；初版还暴露了动态 predicate 影响 compile path 的实现问题，随后修成构造期固定 flag。 |
| B17 input geometry norm | `v124_multibasis_functional_dual_input_geom_norm_audit_20260520T120000Z` | `R5` | fixed block-ZCA/RMS input norm 改善初始 condition/rank，但 B17 仍因 efficiency/expression trade-off 不能成为 base。 |
| B18 light group-RMS norm | `v124_multibasis_functional_dual_group_rms_norm_20260520T130000Z` | `R5` | diagonal-shrink + group-RMS 更轻但没改善初始 geometry，B18a-d 全部没过 A1。 |
| B19 residual geometry norm | `v124_multibasis_functional_dual_residual_geom_norm_20260520T140000Z` | `R5` | residual block-ZCA 折中改善 initial geometry 且 B19a/B19b 过 A1，但 A2 expression 仍失败。 |
| B20 residual mix grid | `v124_multibasis_functional_dual_residual_mix_grid_20260520T150000Z` | `R5` | residual mix 0.15/0.25 小网格真实执行；mix=0.25 过 A1 但 A2 仍失败。 |
| B21 basis-specific input norm | `v124_multibasis_functional_dual_basis_specific_quad_norm_20260520T160000Z` | `R5` | Legendre bounded norm 与 quadratic linear/residual norm 分离；B21d 过 A2，但 A3 task gate 失败。 |
| B22 branch-scale task repair | `v124_multibasis_functional_dual_b21_task_branchscale_20260520T170000Z` | `R5` | B21 分支专属 norm 保持，降低 quadratic branch scale；B22b task accuracy 更近，但 A3 near/AUC/ECE 仍失败。 |
| B23 temperature repair | `v124_multibasis_functional_dual_b23_temperature_repair_20260520T180000Z` | `R5` | 尝试低 logit gain 初始化修 AUC/ECE；独立 run 中 B23a/b 未过 A1，未能形成正式 task 结论。 |
| B24 compiled task timing repair | `v124_multibasis_functional_dual_compiled_task_timing_20260520T190000Z` | `R5` | 让 task triage 使用 compiled steady-state path；B23 accuracy 更近，但 AUC-time ratio 异常放大，base 仍未过。 |
| B25 steady AUC-time accounting | `v124_multibasis_functional_dual_steady_auc_time_20260520T200000Z` | `R3` | 新增 steady epoch 口径，确认 B24 的 AUC-time 巨大值主要是 compile/warmup artifact；B21a/B21d 仍 task_fail。 |
| B26 plain low-temperature repair | `v124_multibasis_functional_dual_plain_temp_steady_20260520T210000Z` | `R5` | 测试 B21a plain 结构的 `0.75/0.50` 低温初始化；B26a/B26b 未过 A1，不能推进 task。 |
| B27 LR schedule first attempt | `v124_multibasis_functional_dual_lr_schedule_repair_20260520T220000Z` | `R5` | 执行后发现 B27 registration/compiled path 不完整，不能作为 B27 科学结论；保留 artifact 作为实现审计。 |
| B27 LR schedule fixed | `v124_multibasis_functional_dual_lr_schedule_repair_fixed_20260520T230000Z` | `R5` | B27a/B27b/B27c 均真实进入 A1/A2/A3；A2 全过，A3 仍 task_fail。 |
| B28 schedule-shape stable | `v124_multibasis_functional_dual_b28_schedule_shape_stable_20260520T020000Z` | `R1` | warmup10 + final LR `0.50/0.75` 的 schedule-shape 变体没过 A1，不能推进 A3。 |
| B29 h224 schedule | `v124_multibasis_functional_dual_b29_h224_schedule_20260520T030000Z` | `R1` | hidden 228 -> 224 的轻量化修复仍没过 A1。 |
| B29 h224 schedule + TF32 | `v124_multibasis_functional_dual_b29_h224_schedule_tf32_20260520T040000Z` | `R1` | 启用 TF32/high matmul precision 后仍没过 A1；该 targeted repair route = R1。 |
| B30 LiteGated input norm | `v124_multibasis_functional_dual_b30_lite_gated_inputnorm_20260520T050000Z` | `R2` | 低成本 direct Legendre + quadratic sketch 过 A1 且 task 有信号，但 A2 expression 失败。 |
| B31 LiteGated quadboost | `v124_multibasis_functional_dual_b31_lite_quadboost_inputnorm_20260520T060000Z` | `R2` | quadboost/midboost 后全部过 A1，task 几何更好，但 direct-readout 结构仍不过 A2。 |
| B32 small GatedHybrid | `v124_multibasis_functional_dual_b32_small_gated_inputnorm_20260520T070000Z` | `R5` | h160/h192 小 GatedHybrid 重开 A1+A2；B32a 有 diagnostic functional positive，但 A3 失败。 |
| B33 temp/hidden tradeoff | `v124_multibasis_functional_dual_b33_small_gated_temp_hidden_20260520T080000Z` | `R3` | B33a/b/c 全部过 A1+A2；B33c task 最接近，但 AUC-time/ECE 子门未过。 |
| B34 AUC/task tradeoff | `v124_multibasis_functional_dual_b34_auc_task_tradeoff_20260520T090000Z` | `R3` | h168/temp075+final050 没有超过 B33c，A3 仍失败。 |
| B35 temp065 | `v124_multibasis_functional_dual_b35_temp065_auc_task_20260520T100000Z` | `R3` | temp065 中间温度改善局部 ECE/worst，但 near pass 和 AUC-time 仍失败。 |
| B36 directskip | `v124_multibasis_functional_dual_b36_directskip_task_stability_20260520T110000Z` | `R3` | direct Legendre skip 仍过 A1+A2，但 A3 task gate 未过；functional 继续 diagnostic。 |
| B37 branchslow + compile warmup | `v124_multibasis_functional_dual_b37_branchslow_compilewarm_20260520T120000Z` | `R3` | branchslow optimizer profile 让 task 明显变差，只 B37b 进入 A3 且失败。 |
| B38 leg/direct fast + compile warmup | `v124_multibasis_functional_dual_b38_legdirectfast_compilewarm_20260520T130000Z` | `R1` | leg/direct fast optimizer profile 未过 A1 efficiency。 |
| B36 compile-warm audit | `v124_multibasis_functional_dual_b36_compilewarm_audit_20260520T140000Z` | `R3` | 12-step compile warmup 降低 B36c AUC-time，但 near/AUC max 仍失败。 |
| B39 temp075 final050 single point | `v124_multibasis_functional_dual_b39_temp075_final050_compilewarm_20260520T150000Z` | `R1` | B36c 的 final LR 0.75 -> 0.50 单点修复未过 A1。 |

最新正式命令：

```bash
python experiments/run_v124_multibasis_functional_dual.py \
  --out-dir results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b39_temp075_final050_compilewarm_20260520T150000Z \
  --fresh --device auto --data-root data --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 5 --task-timing-warmup-epochs 3 --task-compile-warmup-steps 12 \
  --batch-size 128 --eval-batch-size 512 \
  --microbench-batch-sizes 128 --microbench-warmup-steps 20 --microbench-measure-steps 60 \
  --expression-train-size 1024 --expression-val-size 512 --expression-test-size 512 \
  --expression-steps-b0 60 --expression-steps-b1 600 --expression-steps-b2 1200 \
  --expression-batch-size 128 \
  --candidate-ids B39a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final050 \
  --report-path docs/DG-KAN_v12.4_b39_temp075_final050_tmp.md
```

## 3. B14 LR Repair Route（历史关键 run）

```json
{
  "stage": "V124_ROUTE_DECISION",
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A1_exploratory_survivors": [
    "B10a-QuadraticSketch-h64-diagnostic",
    "B10b-QuadraticSketch-h128-diagnostic",
    "B10c-QuadraticSketch-h256-diagnostic",
    "B10d-QuadraticSketch-h512-diagnostic",
    "B11b-TrainableQuadraticSketch-h256",
    "B12a-LegendreKAN-K4-fan-scale-repair",
    "B12b-ChebyKAN-K4-fan-scale-repair",
    "B13a-LegendreKAN-K4-identity-residual-repair",
    "B13b-ChebyKAN-K4-identity-residual-repair",
    "B14a-GatedLegendreQuadratic-h160",
    "B14b-GatedLegendreQuadratic-h224",
    "B14c-GatedLegendreQuadratic-h224-quad-boost",
    "B1r-ReLU-KAN-stream-K2-repair",
    "B6a-BSpline-order1-local-K4",
    "B6b-BSpline-order1-local-K8-expression-repair",
    "B8d-BSpline-ReLU-lite-combo-K4-repair",
    "B9a-Poly2SignedPair-stream-K3-repair",
    "B9d-Poly2PairRandom-h64-K2-diagnostic-repair"
  ],
  "A2_expression_pass": [
    "B14b-GatedLegendreQuadratic-h224"
  ],
  "A3_task_pass": [],
  "functional_diagnostic_positive": true,
  "base_qualified": false,
  "functional_open": false,
  "next_recommended_action": "continue base repair; do not write functional success",
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 4. 最新 Artifact 完整性

```text
v124_basis_manifest.csv rows = 36
v124_grad_correctness.csv rows = 36
v124_efficiency_microbench.csv rows = 61
v124_efficiency_failure_table.csv rows = 30
v124_expression_battery.csv rows = 627
v124_frozen_readout.csv rows = 198
v124_basis_condition.csv rows = 198
v124_task_triage.csv rows = 18
v124_task_trace.csv rows = 90
v124_geometry_snapshot.csv rows = 90
v124_signal_noise.csv rows = 18
v124_tail_calibration.csv rows = 18
v124_task_failure_table.csv rows = 1
v124_functional_one_step.csv rows = 126
v124_functional_five_step.csv rows = 126
v124_control_matrix.csv rows = 18
v124_lambda_backtracking.csv rows = 126
v124_provenance_audit.csv rows = 1
figures/*.svg = written
```

## 5. 最新 A1 Efficiency Microbench

最新 LR repair run 中 B14/B15 关键 rows：

| candidate | compiled | step ratio q90 vs MLP | memory ratio | A1 exploratory pass | A1 base candidate pass |
|---|---:|---:|---:|---:|---:|
| `B14a-GatedLegendreQuadratic-h160` | `1` | `1.0974695068867703` | `0.369161` | `1` | `0` |
| `B14b-GatedLegendreQuadratic-h224` | `1` | `1.210892` | `0.468093` | `1` | `0` |
| `B14c-GatedLegendreQuadratic-h224-quad-boost` | `1` | recorded in run as A1 survivor | recorded | `1` | `0` |
| `B15a-GatedLegendreQuadratic-h224-loss-gain` | `1` | `2.007968` | `0.469168` | `0` | `0` |
| `B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain` | `1` | `1.857244` | `0.475792` | `0` | `0` |

解释：B14 的 compiled path 仍可作为 exploratory survivor，但 B15 logit-gain 明显破坏 clean step gate；因此 B15 不是可继续推进的 base repair。

## 6. Expression / Frozen Coverage

B14 gated coupling 把 A2 expression 真正打开，这是 v12.4 迄今最实质的 base-side 进展。

最新 LR repair run 中唯一 A2 pass 是 B14b：

| candidate | B1 E1 delta | B1 E2 delta | B1 E6 delta | B1 E8 delta | frozen E1/E6/E8 |
|---|---:|---:|---:|---:|---|
| `B14b-GatedLegendreQuadratic-h224` | `-0.019125` | `-0.001213` | `-0.018040` | `-0.018867` | `0.853100 / 0.865472 / 0.846400` |

上一轮 quad-boost run 中 A2 pass 更多：

| candidate | B1 E1 delta | B1 E2 delta | B1 E6 delta | B1 E8 delta | frozen E1/E6/E8 |
|---|---:|---:|---:|---:|---|
| `B14a-h160` | `-0.019595` | `-0.001205` | `-0.019238` | `-0.022521` | `0.482971 / 0.568104 / 0.522443` |
| `B14b-h224` | `-0.019561` | `-0.001200` | `-0.016974` | `-0.016072` | `0.861843 / 0.860002 / 0.841298` |
| `B14d-h228-quad-boost` | `-0.017415` | `-0.002117` | `-0.019364` | `-0.024158` | `0.875268 / 0.880194 / 0.886866` |

解释：B14 的 gated Legendre+quadratic 结构把 B10/B11 的 quadratic coverage 和 B3b 的 trainable expression 成功合到同一个 official-capable primitive 上；但是这还不是 base success，因为 A3 task hard gate 没过。

## 7. 最新 A3 Task Triage

最新 LR repair run 中 B14b 进入 A3，但仍失败：

| candidate | mean delta | worst delta | near pass rate | mean ECE delta | max AUC ratio | mean AUC ratio |
|---|---:|---:|---:|---:|---:|---:|
| `B14b-GatedLegendreQuadratic-h224` | `-0.0026041666666666665` | `-0.02734375` | `0.7777777777777778` | `0.009256` | `2.756724` | `2.24896` |

上一轮 quad-boost run 的 task accuracy 更好，但 AUC/time 仍失败：

| candidate | mean delta | worst delta | mean ECE delta | max AUC ratio | mean AUC ratio |
|---|---:|---:|---:|---:|---:|
| `B14a-h160` | `0.007595486111111111` | `-0.017578125` | `-0.021991` | `2.605755` | `2.333001` |
| `B14b-h224` | `0.009331597222222222` | `-0.0078125` | `-0.015756` | `2.576263` | `2.202616` |
| `B14d-h228-quad-boost` | `0.007378472222222222` | `-0.005859375` | `-0.005454` | `2.451079` | `1.886575` |

解释：B14 已经不是 “表达过但 task 崩” 的旧状态；它在 accuracy/ECE 上接近或超过 MLP，但 loss AUC/time 仍是硬 blocker。B15 logit-gain 和 LR=4e-3/WD=5e-4 都没有解决 AUC/time，说明当前还不是可宣布 base qualified 的 near miss。

## 8. Functional Diagnostic

Functional 只在 `base_qualified = false` 下作为 diagnostic 运行。最新 LR repair run 中 positive rows 仍不能打开 official route：

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
| `B10c-QuadraticSketch-h256-diagnostic` | `0.001936` | `0.000549` | `0.001387` | `1` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `0.148563` | `0.0` | `0.148563` | `1` |
| `B12b-ChebyKAN-K4-fan-scale-repair` | `0.052019` | `0.032660` | `0.019359` | `1` |
| `B13a-LegendreKAN-K4-identity-residual-repair` | `0.005541` | `0.0` | `0.005541` | `1` |
| `B6b-BSpline-order1-local-K8-expression-repair` | `0.140706` | `0.0` | `0.140706` | `1` |

B14/B15 functional rows did not beat controls in the latest runs, and all functional rows remain `diagnostic_base_not_qualified`.

## 9. No-Fake / Hash

Latest LR repair artifact:

```text
rows_checked = 1818
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

Latest artifact SHA256:

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `df369bf862589fa4bca10de49596e3b8549cd0ddc5fec8527afc9390b63c5945` |
| `v124_efficiency_microbench.csv` | `147bd2d57fe8ed3ccce3ffda39a0723e03852dbbafc8e388a0f25c6723cf6e79` |
| `v124_expression_battery.csv` | `87a06cd7259563bfc3f76105bf4c49723681aee4c8bc075f91777dd1ac2229d0` |
| `v124_frozen_readout.csv` | `65698076c1a652fbc7c7a3c696edb7bd35b80c88298eeaddf4c0d0283744d9af` |
| `v124_task_triage.csv` | `19ad9524e0ff613cfadc54b9688b40f2c5fafecc51f3a5d458023bece06e8532` |
| `v124_control_matrix.csv` | `f4eb4040d10ce98b64f0fc382a989079743885fd5fdd9c4099029b2c7ee13e81` |
| `v124_route_decision.json` | `cd13819d3ae97dfe01abcd3c3041acd2cf1ba29f118599a7e20389fae706de06` |
| `v124_provenance_audit.csv` | `667b93995b25db206add122bb56d2adef0a3c680f5d4a67387ae3afcd9c3a8a0` |

## 10. B16/B17 Basis-Specific Norm 修复结果

本节回应“不同基函数需要不同初始化”和“输入需要 norm 层以改善训练初期几何”的追加思路。新增结果来自最新正式 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_input_geom_norm_audit_20260520T120000Z
```

新增 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "expression_fail",
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "functional_diagnostic_positive": true,
  "no_fake": true
}
```

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 42
v124_efficiency_microbench.csv rows = 73
v124_initial_geometry_audit.csv rows = 41
v124_expression_battery.csv rows = 759
v124_frozen_readout.csv rows = 242
v124_basis_condition.csv rows = 242
v124_task_triage.csv rows = 36
v124_task_trace.csv rows = 180
v124_geometry_snapshot.csv rows = 180
v124_control_matrix.csv rows = 22
```

### 10.1 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | B16 branch-output norm | 对 Legendre/quadratic 两个 branch 分别用 unlabeled train-stream 初始 logit std 校准；只影响初始化尺度，不看 label。 |
| `dgkan/models/fc_purekan_primitives.py` | B17 fixed block-ZCA/RMS input geometry norm | 输入先做 block-wise ZCA shrinkage whitening，再做 per-sample RMS 和 `tanh` 截尾；目标是改善初始 basis 几何。 |
| `experiments/run_v124_multibasis_functional_dual.py` | `v124_initial_geometry_audit.csv` | 所有候选都落盘初始 condition/rank/entropy 与 norm 开关，避免 B17 未进 A2 时丢失 norm 效果审计。 |

B16 初版还暴露一个实现 blocker：forward 里动态判断 norm variant 会让 `torch.compile` 对 gated family 生成更慢路径。已修复为构造期固定布尔属性 `branch_output_norm_enabled` / `input_block_norm_enabled` 后重跑正式 artifact。

### 10.2 初始几何

| candidate | input norm | branch norm | condition proxy | effective rank | entropy |
|---|---:|---:|---:|---:|---:|
| `B14b-h224` | `0` | `0` | `1901.5682373046875` | `35.29104232788086` | `0.877773642539978` |
| `B14d-h228-quad-boost` | `0` | `0` | `1408.2276611328125` | `33.23843765258789` | `0.8835713863372803` |
| `B16b-quadboost-basis-norm` | `0` | `1` | `1408.2276611328125` | `33.23843765258789` | `0.8835713863372803` |
| `B17a-input-geom-norm` | `1` | `0` | `773.8685913085938` | `57.40275192260742` | `0.8665974736213684` |
| `B17b-quadboost-input-geom-norm` | `1` | `0` | `684.5045166015625` | `55.35911178588867` | `0.8687055706977844` |
| `B17c-input-plus-branch-norm` | `1` | `1` | `773.8685913085938` | `57.40275192260742` | `0.8665974736213684` |
| `B17d-quadboost-input-plus-branch-norm` | `1` | `1` | `684.5045166015625` | `55.35911178588867` | `0.8687055706977844` |

解释：输入 norm 不是空改。它把 gated hybrid 的初始 frozen feature condition 从约 `1408-1902` 降到约 `684-774`，effective rank 从约 `33-35` 提升到约 `55-57`。这符合“训练初期需要更好几何”的动机。

### 10.3 Efficiency / Expression Gate

| candidate | compiled step ratio | A1 pass | B1 E1 delta | B1 E2 delta | B1 E6 delta | B1 E8 delta |
|---|---:|---:|---:|---:|---:|---:|
| `B14b-h224` | `0.8997853832720459` | `1` | `-0.032576680183410645` | `-0.003492295742034912` | `-0.021341919898986816` | `-0.03446006774902344` |
| `B14d-h228-quad-boost` | `1.4574127460023072` | `1` | `-0.04824507236480713` | `-0.0033052563667297363` | `-0.021450459957122803` | `-0.046791017055511475` |
| `B16b-quadboost-basis-norm` | `2.0810171190020292` | `0` | not_run | not_run | not_run | not_run |
| `B17a-input-geom-norm` | `1.6313675202324913` | `0` | not_run | not_run | not_run | not_run |
| `B17b-quadboost-input-geom-norm` | `1.4100122612002235` | `1` | `-0.06759607791900635` | `-0.010822713375091553` | `-0.06615930795669556` | `-0.07360023260116577` |
| `B17c-input-plus-branch-norm` | `1.8609649554238297` | `0` | not_run | not_run | not_run | not_run |
| `B17d-quadboost-input-plus-branch-norm` | `1.5212885973541603` | `0` | not_run | not_run | not_run | not_run |

解释：B17 的输入 norm 改善了初始几何，但代价是 clean step 变重；只有 B17b 勉强过 A1 exploratory，进入 A2 后 E1/E6/E8 expression 明显变差。因此它没有成为 base survivor。B16 branch norm alone 也没有形成 A1/A2 改善。

### 10.4 Functional Diagnostic

最新 B17 run 仍有 diagnostic positive row，但不是来自合格 base：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B12a-LegendreKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `2.5224014241401074` | `1` |
| `B2a-GaussianRBF-K4-compact` | `diagnostic_base_not_qualified` | `0.016744405456739386` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `1` |
| `B17b-quadboost-input-geom-norm` | `diagnostic_base_not_qualified` | `-0.0005076177453817721` | `0` |

### 10.5 No-Fake / Hash

```text
rows_checked = 2432
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `eaa22b2d2ba1f94aac1ef2541c6dbb722515d209137e7c4bd34972b9174d60e7` |
| `v124_efficiency_microbench.csv` | `c7ab5ee4bd0c695cdfe4f1b8ab6012199f385d29abd4672c2ac51a6c13f85137` |
| `v124_initial_geometry_audit.csv` | `c04a129ef379ccf6701f2149988613dfb9e655b0435717ae77a0f2b06d5cc5b1` |
| `v124_expression_battery.csv` | `ee33bc78c32b8d4d7379d1ce55471e43d7a21925763c00c78cb39cf7248c260b` |
| `v124_frozen_readout.csv` | `e1f418b1f7416d32bbc9df8723465e300add5f75b1ac33767896712f5cd00687` |
| `v124_task_triage.csv` | `6237c6bb58c05f45090bfa046d96c67774f0663e3bc9ea47a98730c08a7eb377` |
| `v124_control_matrix.csv` | `d6a483d2674e43872ba5c346eec3c42cc69e6fdf338c48fb6bda8f22076fe36a` |
| `v124_route_decision.json` | `946bb37080be2ea807970122522d2d19ee845c39b0d2ab11e1c406e13136112e` |
| `v124_provenance_audit.csv` | `a58556662d0ab39a6536a5a7295e65da643e8a4a7c091a637859ca01aea54315` |

追加判断：我们确实尝试了“不同 basis 需要不同初始化”和“输入增加 norm 层”的方向。结果是 input geometry norm 对初始 condition/rank 有真实改善，但当前 block-ZCA/RMS 版本把 expression coverage 和/或 efficiency 推坏了。下一步若继续，应把 input norm 做成更轻的 diagonal/group RMS 或 learnable-free annealed norm，并专门优化 B14 的 early loss-AUC，而不是把 B17 写成成功。

## 11. B18 Light Group-RMS Norm 修复结果

本节继续尝试 B17 之后的推荐修复方向：把 block-ZCA/RMS 换成更轻的 diagonal-shrink + group-RMS input norm，试图保留输入几何稳定性同时降低 step overhead。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_group_rms_norm_20260520T130000Z
```

新增 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "expression_fail",
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "functional_diagnostic_positive": true,
  "no_fake": true
}
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 B18a/B18b/B18c/B18d group-RMS input norm | 使用 unlabeled train-stream 的 diagonal variance shrinkage，再做 fixed group RMS；比 B17 的 block-ZCA 更轻，不引入 learnable norm。 |
| `experiments/run_v124_multibasis_functional_dual.py` | initial geometry audit 增加 `input_group_rms_norm_enabled` | 让 B17 block norm 与 B18 group-RMS norm 可在 artifact 中区分审计。 |

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 46
v124_efficiency_microbench.csv rows = 81
v124_initial_geometry_audit.csv rows = 45
v124_expression_battery.csv rows = 594
v124_frozen_readout.csv rows = 187
v124_basis_condition.csv rows = 187
v124_task_triage.csv rows = 36
v124_control_matrix.csv rows = 17
```

B18 初始几何与效率：

| candidate | group norm | condition proxy | effective rank | compiled step ratio | A1 pass |
|---|---:|---:|---:|---:|---:|
| `B14b-h224` | `0` | `1901.5682373046875` | `35.29104232788086` | `1.9698895596287107` | `0` |
| `B17b-block-ZCA-input-norm` | `0` | `684.5045166015625` | `55.35911178588867` | `1.992752267088716` | `0` |
| `B18a-group-rms-norm` | `1` | `1807.1361083984375` | `30.60650062561035` | `1.504425568376251` | `0` |
| `B18b-quadboost-group-rms-norm` | `1` | `1422.694091796875` | `29.41107177734375` | `1.6249372852325548` | `0` |
| `B18c-group-rms-plus-branch-norm` | `1` | `1807.1361083984375` | `30.60650062561035` | `2.47454326271851` | `0` |
| `B18d-quadboost-group-rms-plus-branch-norm` | `1` | `1422.694091796875` | `29.41107177734375` | `1.5645713696281405` | `0` |

解释：B18 确实比 B17 避免了 block-ZCA 矩阵 whitening，但它没有把初始 condition/rank 修好；B18a-d 也全部没过 A1，因此没有合法进入 A2/A3。这个结果把“更轻的 group RMS norm”排除了：它轻了一点，但没有给足够好的初始几何。

No-fake：

```text
rows_checked = 2076
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1d4cc2818356c4135e4d708e4c3e4e4bba1b8c971e84869c4ad1e74a8b1ac32e` |
| `v124_efficiency_microbench.csv` | `f40aa6cb3b188eca7a69d2f230d243a18afab5cfb77664ca1520cb3e1343a047` |
| `v124_initial_geometry_audit.csv` | `87bb4ee8d507ceeb9943fc25e17ad72eec02f47327854e842b7712928844f07a` |
| `v124_expression_battery.csv` | `fab880c46926dc31f097ae4eb531252a1f3af9d83c6160e55c0760b357d4ca08` |
| `v124_task_triage.csv` | `2142719bd89fc37baf070e14c48777cfc72bd6be63739d19a443e51f45495e40` |
| `v124_control_matrix.csv` | `cccd700c3f930cb340dc6c891136528d00fcdf437a1d4b5a13f91210a1ce5a0a` |
| `v124_route_decision.json` | `5e76c857b0f1d753fb509a85ef7c199a38450c27ac23b9020c36326a9a31ec66` |
| `v124_provenance_audit.csv` | `1d708fabe156fa217d3974e2dfa7e1d7415e7c2f61cba03a5a7c9a2c41cc946e` |

追加判断：B17 证明“输入 norm 可以改善初始几何但太重/伤表达”，B18 证明“轻量 group RMS 不足以改善初始几何”。当前更可行的 norm 方向不是简单 RMS，而是需要保留 block-ZCA 的几何收益、同时做低成本近似或 annealed/residual mixing；这已经超出本轮 v12.4 的小修范畴。

## 12. B19 Residual Geometry Norm 修复结果

本节继续执行 B18 后的明确修复方向：保留 B17 的 block-ZCA 几何收益，但不完全替换输入坐标，而是使用 residual mixing：

```text
z = 0.65 * per_feature_zscore + 0.35 * block_zca_rms
z = tanh(z)
```

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_residual_geom_norm_20260520T140000Z
```

新增 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "expression_fail",
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "functional_diagnostic_positive": true,
  "no_fake": true
}
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 B19a/B19b/B19c/B19d residual block-ZCA input norm | 使用 unlabeled train-stream 的 block-ZCA/RMS，但只以 0.35 mix 注入原始 per-feature z-score 坐标；目标是折中几何和表达保真。 |
| `experiments/run_v124_multibasis_functional_dual.py` | initial geometry audit 增加 `input_residual_block_norm_enabled` | 让 B17 full block norm、B18 group RMS、B19 residual block norm 可在 artifact 中区分审计。 |

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 50
v124_efficiency_microbench.csv rows = 89
v124_initial_geometry_audit.csv rows = 49
v124_expression_battery.csv rows = 759
v124_frozen_readout.csv rows = 242
v124_basis_condition.csv rows = 242
v124_task_triage.csv rows = 36
v124_control_matrix.csv rows = 22
```

B19 初始几何、效率与 expression：

| candidate | residual norm | condition proxy | effective rank | compiled step ratio | A1 pass | B1 E1 delta | B1 E6 delta | B1 E8 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B14b-h224` | `0` | `1901.5682373046875` | `35.29104232788086` | `1.2563036412133184` | `1` | `-0.04073631763458252` | `-0.02234673500061035` | `-0.03583461046218872` |
| `B17b-block-ZCA-input-norm` | `0` | `684.5045166015625` | `55.35911178588867` | `1.6514156487830656` | `0` | not_run | not_run | not_run |
| `B18b-group-RMS-input-norm` | `0` | `1422.694091796875` | `29.41107177734375` | `1.6182735421207115` | `0` | not_run | not_run | not_run |
| `B19a-residual-geom-norm` | `1` | `1422.872802734375` | `41.76232147216797` | `1.2865459848728793` | `1` | `-0.04451334476470947` | `-0.0230787992477417` | `-0.03670698404312134` |
| `B19b-quadboost-residual-geom-norm` | `1` | `1053.0626220703125` | `39.3907470703125` | `1.2984553348452903` | `1` | `-0.052997589111328125` | `-0.032508790493011475` | `-0.04683351516723633` |

解释：B19 是三种 input norm 中最平衡的：它比 B18 有更好的 condition/rank，也比 B17 更容易过 A1。但它没有打开 A2，E1/E6/E8 仍低于 gate，因此不能进入 A3，也不能写成 base success。

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B19a-residual-geom-norm` | `diagnostic_base_not_qualified` | `-0.004002655797295862` | `0` |
| `B19b-quadboost-residual-geom-norm` | `diagnostic_base_not_qualified` | `-0.0011577013205532616` | `0` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `2.978411122267488e-05` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `1` |

No-fake：

```text
rows_checked = 2484
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1c04801417b87e1e304acf6144bb0e920a1fa75c3b73b01a50afae0256af9216` |
| `v124_efficiency_microbench.csv` | `492ccdc163b56fbaba91aa1a414d4077debc9964d6f360cea9d3c44d23a979c5` |
| `v124_initial_geometry_audit.csv` | `635879412b727e4902b48a650a38597022ac34aef1647631bb4ee828484dba53` |
| `v124_expression_battery.csv` | `cb0f6e109d46939759c64866c9d9b13f93b2248c810c0fa3caf8172c80d7cc0c` |
| `v124_task_triage.csv` | `8f268b8984e4ea6c0c85eb033d48edf0e8cb2aaf4b6bbbcff42c601f02ea796e` |
| `v124_control_matrix.csv` | `3e1e703302a1cc5b15521fde1167a38cdfb3b43c441fb6ff16b329b65dfd984c` |
| `v124_route_decision.json` | `2c6dbabee01c92fa5edda524a6dcee043543fd7c4693604fb4fd69af73569fdc` |
| `v124_provenance_audit.csv` | `4be341e78f47d383c8092d8c538533179d7d7807dff1bb0c37db32b6d4328f6b` |

追加判断：B19 证明 residual/annealed input norm 是比 B17/B18 更合理的折中方向，但 0.35 mix 仍不足以恢复 expression coverage。继续在 v12.4 内盲扫 mix 系数会变成调参，不再是清晰的新机制；更合理的下一步是把 input norm 的 objective 写成独立计划：同时约束 initial condition、frozen expression coverage 和 step overhead，再做小网格。

## 13. B20 Residual Mix Grid 修复结果

本节把上一节的“input norm objective 小网格”落到真实执行，但只做机制化 residual mix 网格，不按 dataset/label/outcome 分支：

```text
B20a/B20c: residual mix = 0.15
B20b/B20d: residual mix = 0.25
```

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_residual_mix_grid_20260520T150000Z
```

新增 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "expression_fail",
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "functional_diagnostic_positive": true,
  "no_fake": true
}
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 B20a/B20b/B20c/B20d residual mix grid | 只改变 fixed residual input norm 的 mix 系数 0.15/0.25；不看 label、不按 dataset 分支、不改 gate。 |
| `experiments/run_v124_multibasis_functional_dual.py` | compiled repair ids 加入 B20a-d | 让 residual mix 小网格经过同一 compiled efficiency accounting。 |

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 54
v124_efficiency_microbench.csv rows = 97
v124_initial_geometry_audit.csv rows = 53
v124_expression_battery.csv rows = 561
v124_frozen_readout.csv rows = 176
v124_basis_condition.csv rows = 176
v124_task_triage.csv rows = 36
v124_control_matrix.csv rows = 16
```

B20 初始几何、效率与 expression：

| candidate | residual mix | condition proxy | effective rank | compiled step ratio | A1 pass | B1 E1 delta | B1 E6 delta | B1 E8 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B19a-mix35` | `0.35` | `1422.872802734375` | `41.76232147216797` | `1.251428000020621` | `1` | `-0.04560738801956177` | `-0.02502971887588501` | `-0.043892502784729004` |
| `B20a-mix15` | `0.15` | `1570.107421875` | `37.803627014160156` | `1.8504636223893665` | `0` | not_run | not_run | not_run |
| `B20b-mix25` | `0.25` | `1534.6663818359375` | `39.70948791503906` | `0.8934151428652006` | `1` | `-0.05284494161605835` | `-0.028607606887817383` | `-0.03526043891906738` |
| `B20c-quadboost-mix15` | `0.15` | `1137.2996826171875` | `35.628902435302734` | `1.8515823841383738` | `0` | not_run | not_run | not_run |
| `B20d-quadboost-mix25` | `0.25` | `1075.7344970703125` | `37.437896728515625` | `1.0644790225917549` | `1` | `-0.06027674674987793` | `-0.02688014507293701` | `-0.05104917287826538` |

解释：B20 的结果排除了简单 residual mix 小网格。mix=0.25 的效率可以过 A1，但 E1/E6/E8 expression 没有恢复；mix=0.15 不稳定，未形成 A1 survivor。继续盲扫 mix 系数不再是机制性修复。

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B20b-residual-mix25` | `diagnostic_base_not_qualified` | `-0.002067623863291068` | `0` |
| `B20d-quadboost-residual-mix25` | `diagnostic_base_not_qualified` | `-9.382437794580589e-05` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `2.5224014241401074` | `1` |
| `B3a-ChebyKAN-K4` | `diagnostic_base_not_qualified` | `0.0006867722875756321` | `1` |

No-fake：

```text
rows_checked = 2046
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `dd711b0105f7ab191d609371722a2d5f4ee85e5328e68fc0303b4761a223aed6` |
| `v124_efficiency_microbench.csv` | `da2234f799582d253e116f50e5eadb9a786621c2a6fdce014ad585801de609c9` |
| `v124_initial_geometry_audit.csv` | `3159b2c0519d0ff673fae8eef8cb5d68c4a5b6cafd4620202d1bb55ff84e120b` |
| `v124_expression_battery.csv` | `e7f67d40f18146432a2e6c0845e595358c2749df285e2f50b1caaa86a97329dd` |
| `v124_task_triage.csv` | `5fd718347ddfdf213dfe78434dcbb57164939b56d9c0c7e99bbe663627d3bfc0` |
| `v124_control_matrix.csv` | `ca3119c03e34fe9ed2f3569e5e64cd408c110beb4dfd509bfaebaa5fd2a122b2` |
| `v124_route_decision.json` | `028aa19ea775b5911614cd25beadcff6a60e9e896ecb00eec74c34e20201f5b8` |
| `v124_provenance_audit.csv` | `e24a30dc98527decdd26b553bfdfd15908a3170b620ac0ae001e59ea0fedaf9d` |

追加判断：B20 说明 mix-grid 不是当前突破点。input norm 确实影响初始几何，但在当前 gated Legendre+quadratic primitive 上，表达 coverage 对输入坐标很敏感。下一步若继续，应从“norm 之后的 basis centers/projection co-design”或 “B14 early loss-AUC” 入手，而不是继续扫 residual mix。

## 14. B21 Basis-Specific Input Norm 修复结果

> 本节回应“不同基函数需要不同初始化/input norm”的判断：不再把同一个 tanh/ZCA norm 同时喂给 Legendre 和 quadratic。Legendre 分支继续使用 bounded 坐标，quadratic 分支使用固定线性或 residual geometry norm，并对 quadratic projection feature std 做 unlabeled train-stream calibration。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_basis_specific_quad_norm_20260520T160000Z
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `_quadratic_input` 与 `quad_feature_std` calibration | quadratic 分支避免 tanh 压扁二次结构；只使用 unlabeled train-stream statistics，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B21a/B21b/B21c/B21d | 分别测试 plain/quadboost 与 residual geometry norm 的 branch-specific input norm。 |
| `experiments/run_v124_multibasis_functional_dual.py` | audit 增加 B21 norm flags | 记录 `basis_specific_quad_input_norm_enabled`、`quad_feature_norm_enabled` 与 `quad_feature_std_mean`，便于审计。 |

Route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A2_expression_pass": [
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "functional_diagnostic_positive": true,
  "no_fake": true
}
```

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 58
v124_efficiency_microbench.csv rows = 105
v124_initial_geometry_audit.csv rows = 57
v124_expression_battery.csv rows = 891
v124_frozen_readout.csv rows = 286
v124_basis_condition.csv rows = 286
v124_task_triage.csv rows = 18
v124_control_matrix.csv rows = 26
```

B21 初始几何与效率：

| candidate | condition proxy | effective rank | quad feature std mean | best step ratio | A1 pass |
|---|---:|---:|---:|---:|---:|
| `B21a-plain-linear-quad` | `5710.5478515625` | `50.42596435546875` | `0.6857059597969055` | `2.2684906699292156` | `0` |
| `B21b-quadboost-linear-quad` | `5706.88134765625` | `47.512081146240234` | `0.7033965587615967` | `2.4922414370950996` | `0` |
| `B21c-residual-linear-quad` | `4031.17578125` | `63.339290618896484` | `0.66989666223526` | `1.5839225216717552` | `0` |
| `B21d-quadboost-residual-linear-quad` | `3611.5732421875` | `59.4402961730957` | `0.682805061340332` | `1.0775065698861643` | `1` |

解释：B21d 通过 A1，并且 effective rank 高于 B14/B19/B20；但 condition proxy 仍高，说明分支专属 norm 修复了 coverage/optimization 的一部分，没有完全修好 geometry conditioning。

B21d expression 结果：

| target | B1 val R2 | B1 delta vs MLP | B2 val R2 | B2 delta vs MLP | frozen R2 |
|---|---:|---:|---:|---:|---:|
| `E1-pairwise-product` | `0.9931602478027344` | `-0.0038663744926452637` | `0.9940307140350342` | `-0.002943098545074463` | `0.8940063118934631` |
| `E2-composition` | `0.9955880641937256` | `0.000898897647857666` | `0.9954589605331421` | `0.0006471872329711914` | `0.9875801801681519` |
| `E6-rotated-pairwise-product` | `0.9962476491928101` | `0.0010586977005004883` | `0.9966369867324829` | `0.0009151697158813477` | `0.9269713163375854` |
| `E8-random-quadratic-form` | `0.9947894811630249` | `0.0005333423614501953` | `0.9960606694221497` | `0.0002377629280090332` | `0.9110857248306274` |

解释：这是 B21 的实质进展。它把 B17-B20 的 expression failure 修回来，并且 frozen coverage 对 E1/E2/E6/E8 全部超过 `0.89`。因此“不同 basis 需要不同 input norm”这个思路被本轮真实数据支持。

B21d task gate：

| dataset | seed | val acc delta vs MLP |
|---|---:|---:|
| `MNIST` | `0` | `-0.001953125` |
| `MNIST` | `1` | `-0.005859375` |
| `MNIST` | `2` | `-0.013671875` |
| `Fashion-MNIST` | `0` | `-0.044921875` |
| `Fashion-MNIST` | `1` | `0.009765625` |
| `Fashion-MNIST` | `2` | `0.005859375` |
| `KMNIST` | `0` | `-0.017578125` |
| `KMNIST` | `1` | `-0.021484375` |
| `KMNIST` | `2` | `0.013671875` |

Task failure：

```text
mean_delta = -0.008463541666666666
near_pass_rate = 0.4444444444444444
worst_delta = -0.044921875
auc_time_ok = 0
ece_ok = 0
```

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B21d-quadboost-basis-specific-residual-quad-norm` | `diagnostic_base_not_qualified` | `-0.002876382782057263` | `0` |

No-fake：

```text
rows_checked = 2600
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `3232497c5f343052c92555665eda25c221c2c6e90f04c55005266cfa50acd547` |
| `v124_efficiency_microbench.csv` | `f660d097f39c3dfe693667de904ac3fb061b3af61f0be4885c631bdfa41b2616` |
| `v124_initial_geometry_audit.csv` | `f3c94f3d3abf0ece83df1b69572e909bf0a0c73fe00a51443f188b051a0b2342` |
| `v124_expression_battery.csv` | `9955edbe8e0614ca1bfc3c1b0694f373256c3ad00b63d19a54fd47dd75e2909f` |
| `v124_task_triage.csv` | `d6ca986844b4b0cedbcf9f10acd8880b950f4290abb6ed62d9a3a9b6579aef73` |
| `v124_control_matrix.csv` | `add03d0bc1644380426dea2da8939d4fb6e14f7da40f3c802698884f0837c54a` |
| `v124_route_decision.json` | `deb8b0ff9b175f8755547b1fbe5c9ca2ee48237bdadb431cd7007287f995bf81` |
| `v124_provenance_audit.csv` | `c7464a3521b06a9970007e4a9040ff9724320b6cbebb742b4c7b4e222ed13e8a` |

追加判断：B21 是 v12.4 后半段最有信息量的修复。它验证了“basis-specific input norm”方向，并把 base blocker 从 latest B20 的 `expression_fail` 推回 `task_fail`；但 B21d 还不是 base success。下一步不该回去扫 residual mix，而应围绕 B21d 的 task failure 做 global optimizer / conditioning repair，例如：保持分支专属 norm 与 projection calibration，单独修 AUC-time/ECE/weak slice，不允许 dataset-specific 调参。

## 15. B22 Branch-Scale Task Repair 结果

> 本节继续执行 B21 后的 task-fail route：保持 B21 的 basis-specific input norm 与 quadratic feature calibration，只改变全局 branch-scale 初始化，测试是否能保住 A2 的同时修复 A3 task stability。没有 dataset-specific 分支、没有 loss/teacher/class-weight 修改。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b21_task_branchscale_20260520T170000Z
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 B22a/B22b branch-scale variants | B22a 初始化 `[legendre, quadratic] = [1.0, 0.50]`，B22b 初始化 `[1.0, 0.35]`；只做全局尺度修复，不使用 label/outcome。 |
| `experiments/run_v124_multibasis_functional_dual.py` | compiled repair ids 加入 B22a/B22b | B22 与 B21 使用同一 clean/compiled efficiency accounting。 |

Route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A2_expression_pass": [
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "no_fake": true
}
```

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 60
v124_efficiency_microbench.csv rows = 109
v124_initial_geometry_audit.csv rows = 59
v124_expression_battery.csv rows = 1716
v124_frozen_readout.csv rows = 561
v124_basis_condition.csv rows = 561
v124_task_triage.csv rows = 63
v124_control_matrix.csv rows = 51
```

B22 efficiency / geometry：

| candidate | condition proxy | effective rank | compiled step ratio | A1 pass |
|---|---:|---:|---:|---:|
| `B21d-quadboost` | `3611.5732421875` | `59.4402961730957` | `1.1905899309931782` | `1` |
| `B22a-midboost` | `3611.5732421875` | `59.4402961730957` | `1.0953644001145177` | `1` |
| `B22b-lowboost` | `3611.5732421875` | `59.4402961730957` | `1.2536052362257843` | `1` |

B22 expression 关键结果：

| candidate | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | E1 frozen | E6 frozen | E8 frozen |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B21d-quadboost` | `-0.01416468620300293` | `0.0008502006530761719` | `0.0005941987037658691` | `0.0011792182922363281` | `0.9068983793258667` | `0.8568429946899414` | `0.8779054284095764` |
| `B22a-midboost` | `-0.012031197547912598` | `0.0007821917533874512` | `0.0010418891906738281` | `0.0020076632499694824` | `0.8741676807403564` | `0.9250072240829468` | `0.928270697593689` |
| `B22b-lowboost` | `-0.009033024311065674` | `0.00096893310546875` | `0.0007177591323852539` | `0.002260446548461914` | `0.8477163314819336` | `0.9055468440055847` | `0.881303608417511` |

解释：降低 quadratic branch scale 没有破坏 A2；B22b 在 E1 B1 delta 和 task accuracy 上最好，但 frozen E1 下降，说明 task/expression 之间仍有 trade-off。

Task aggregate：

| candidate | mean delta | worst delta | near pass rate | AUC-time ok | ECE ok | A3 pass |
|---|---:|---:|---:|---:|---:|---:|
| `B21a-plain` | `0.00043402777777777775` | `-0.017578125` | `0.6666666666666666` | `0` | `0` | `0` |
| `B21c-residual` | `-0.0015190972222222222` | `-0.02734375` | `0.6666666666666666` | `0` | `0` | `0` |
| `B22a-midboost` | `-0.0030381944444444445` | `-0.015625` | `0.4444444444444444` | `0` | `0` | `0` |
| `B22b-lowboost` | `-0.001736111111111111` | `-0.0234375` | `0.6666666666666666` | `0` | `0` | `0` |

B22b paired task rows：

| dataset | seed | val acc delta vs MLP |
|---|---:|---:|
| `MNIST` | `0` | `0.0` |
| `MNIST` | `1` | `0.00390625` |
| `MNIST` | `2` | `-0.01171875` |
| `Fashion-MNIST` | `0` | `-0.00390625` |
| `Fashion-MNIST` | `1` | `0.03125` |
| `Fashion-MNIST` | `2` | `-0.001953125` |
| `KMNIST` | `0` | `0.0` |
| `KMNIST` | `1` | `-0.009765625` |
| `KMNIST` | `2` | `-0.0234375` |

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B21d-quadboost` | `diagnostic_base_not_qualified` | `-0.002876382782057263` | `0` |
| `B22a-midboost` | `diagnostic_base_not_qualified` | `-0.0016101342408338937` | `0` |
| `B22b-lowboost` | `diagnostic_base_not_qualified` | `-0.0052961370711956945` | `0` |

No-fake：

```text
rows_checked = 5118
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `4eebac07bcacb96cc6b995a427f2ce0374ca181f0b65464225e140facde19c72` |
| `v124_efficiency_microbench.csv` | `054734c1f298d6c13faae25ec0b676921629e3fcbbcaf6be028c2fc2cc3fba19` |
| `v124_initial_geometry_audit.csv` | `73e105f7e47b2958b665edfd0e41bfdc83382d9aee18d0d5ad0e35b8cf8dbb2f` |
| `v124_expression_battery.csv` | `aea1f4ee228502aeaf630afb493f13ac53a7a1d492c718f1c9712cb774cdb246` |
| `v124_task_triage.csv` | `fd1d07cde10e4b81e4d154161fc64aaee0af284515ec50a14218808f8c7aea8a` |
| `v124_control_matrix.csv` | `1571c9ccc05e3b6fe67a229b15bfb8a9a2b716975b91dbf58b0538cd909ea397` |
| `v124_route_decision.json` | `6c71f3f4c3718e58bd4d81be0ab258f8d86e3b5e83eae24ead03e9e0d81cdc84` |
| `v124_provenance_audit.csv` | `37aa7d9dbb0b3102b7811a82b2593e6ed981fec19be5d6594f82fdde6ef8c68c` |

追加判断：B22 是 task gate 的真实进展，但仍不是 base success。B22b 的 mean/worst 已接近或达到 accuracy 子门，主要剩余是 near pass rate、AUC-time 与 ECE；继续随机扫 branch scale 没有意义。下一步如果继续，应设计专门的 AUC/ECE repair（例如训练轨迹/校准/conditioning 机制），并保持 no dataset-specific tuning。

## 16. B23 Temperature Repair 结果

> 本节继续执行 AUC/ECE repair：保持 B22b 的 basis-specific norm 与 low quadratic branch scale，只把全局 logit gain 初始化为 `0.75` / `0.50`。这不是 loss modification，也不是 post-hoc calibration；它只是无标签、全局的 logit temperature 初始化。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b23_temperature_repair_20260520T180000Z
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 B23a/B23b | B23a `logit_gain=0.75`，B23b `logit_gain=0.50`；保持 B22b branch/norm 结构，不看 label/outcome。 |
| `experiments/run_v124_multibasis_functional_dual.py` | compiled repair ids 加入 B23a/B23b | B23 与 B21/B22 同一 A1 compiled accounting。 |

Standalone B23 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A2_expression_pass": [
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "no_fake": true
}
```

Standalone B23 关键失败：

| candidate | compiled step ratio | A1 pass | task conclusion |
|---|---:|---:|---|
| `B23a-temp075` | `2.0277726360730925` | `0` | not_run |
| `B23b-temp050` | `2.066922034752469` | `0` | not_run |

No-fake：

```text
rows_checked = 2166
fake/proxy/cpu = 0 / 0 / 0
```

解释：B23 standalone run 没有形成 task 结论，因为 B23a/B23b 没过 A1 timing。这个失败说明仅加入低温初始化还不够，还暴露出 A1 compiled timing 与 A3 task timing/accounting 的不一致风险，因此继续做 B24。

## 17. B24 Compiled Task Timing Repair 结果

> 本节修复 runner accounting：此前 B22/B23 的 A1 可依赖 compiled warm path，但 A3 task triage 使用未编译训练 path 计入 AUC-time。B24 让 compile-repair candidates 在 A3 使用同一 `torch.compile(..., mode="reduce-overhead")` steady-state path，并在 trace/triage 中写入 `compiled_task_path=1`。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_compiled_task_timing_20260520T190000Z
```

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `_task_compile_candidate` | 对与 A1 compiled repair 相同的一批候选启用 compiled task path，避免 A1/A3 使用不同执行路径。 |
| `experiments/run_v124_multibasis_functional_dual.py` | A3 task warm compile + audit fields | 在计时前 warm forward/backward，不做 optimizer step；记录 `compiled_task_path` 与 `compile_error`。 |

Route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A2_expression_pass": [
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
    "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "no_fake": true
}
```

新增 artifact 完整性：

```text
v124_basis_manifest.csv rows = 62
v124_efficiency_microbench.csv rows = 113
v124_initial_geometry_audit.csv rows = 61
v124_expression_battery.csv rows = 1848
v124_frozen_readout.csv rows = 605
v124_task_trace.csv rows = 405
v124_task_triage.csv rows = 81
v124_control_matrix.csv rows = 55
```

B24 task aggregate（全部 `compiled_task_path = 1`）：

| candidate | mean delta | worst delta | near pass | AUC-time mean ratio | AUC-time max ratio | ECE mean delta | ECE max delta | A3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B21a` | `0.007595486111111111` | `-0.0078125` | `0.8888888888888888` | `224.00492529018447` | `296.5607025497942` | `0.007171362638473511` | `0.027430877089500427` | `0` |
| `B22b` | `0.0006510416666666666` | `-0.017578125` | `0.6666666666666666` | `243.73900322075133` | `342.6802162360393` | `0.014616372684637705` | `0.06325441598892212` | `0` |
| `B23a-temp075` | `0.009548611111111112` | `-0.005859375` | `0.8888888888888888` | `257.80341239177955` | `384.45636222744776` | `0.007062929785913891` | `0.04108989238739014` | `0` |
| `B23b-temp050` | `0.009982638888888888` | `-0.0078125` | `0.8888888888888888` | `280.69549237379675` | `391.83748832177037` | `0.0017214732037650214` | `0.04181094467639923` | `0` |

解释：B23 的低温初始化在 B24 中确实改善了 task accuracy 与平均 ECE，尤其 B23b 的 mean delta 达到 `0.009982638888888888`，ECE mean delta 只有 `0.0017214732037650214`。但是 AUC-time ratio 被 compiled task path 异常放大到 `224`-`280` 均值量级，且 ECE max 仍超过 `0.02`，因此不能把 B24 写成 base pass。

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B21b` | `diagnostic_base_not_qualified` | `0.0008413767637345249` | `1` |
| `B23a-temp075` | `diagnostic_base_not_qualified` | `-0.003482136507368061` | `0` |
| `B23b-temp050` | `diagnostic_base_not_qualified` | `-0.003090285627203926` | `0` |

No-fake：

```text
rows_checked = 5671
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1475352572ad5b66122b5eb37a84e5c24362c95f5019fc176df0178be088bbde` |
| `v124_efficiency_microbench.csv` | `fe780069266f96cd1a48b51eec12421f7d5110b0afb9392272f53412ea216713` |
| `v124_initial_geometry_audit.csv` | `7e8f5394e74977557c2dbda66ecad649dfd2186a56c423edf27d9f61c43aae0b` |
| `v124_expression_battery.csv` | `d7df3a5764ba56dccf9514e059f751bb3c1490753c3bf058ae0ded78d66fe28f` |
| `v124_task_triage.csv` | `34e166e5f91263c4cca4de17eb633cd01992d5b5a252df07d891f14fbbe98eac` |
| `v124_control_matrix.csv` | `1ea62ff471f0ce252c27e38209d62be9ac754e3ff8fd4ca6d4a7764cdf0a06fc` |
| `v124_route_decision.json` | `7f3879b070c59ef161e7b943948e9d094456c9ff1218139a963321c495f55ad7` |
| `v124_provenance_audit.csv` | `509cc59e2366f8b25d7080dd84730fc632218134ac6f6e0ddbd263a202bd7d83` |

追加判断：B24 不是成功修复，但它把失败原因进一步拆开了。B23 temperature 可以修一部分 accuracy/ECE；compiled task path 不能直接拿来修 AUC-time，反而暴露出当前 AUC-time accounting 对 compiled training path 不稳定。下一步若继续，应先修 AUC-time measurement/steady-state training implementation，再谈 base gate；不能用 B24 的 compiled path 结果强行过关。

## 18. B25 Steady AUC-Time Accounting 结果

> 本节修复 B24 暴露的 measurement blocker：A3 task triage 同时记录 raw AUC-time 和跳过前 3 个 epoch 后的 steady AUC-time。这个修复只改变计量口径与落盘字段，不改模型、loss、gate 或数据。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_steady_auc_time_20260520T200000Z
```

新增 route：

```json
{
  "route": "R3-ExpressionPassTaskFail",
  "base_blocker_route": "task_fail",
  "A2_expression_pass": [
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_diagnostic_positive": false,
  "functional_open": false,
  "no_fake": true
}
```

新增落盘完整性：

```text
v124_basis_manifest.csv rows = 62
v124_efficiency_microbench.csv rows = 113
v124_initial_geometry_audit.csv rows = 61
v124_expression_battery.csv rows = 561
v124_frozen_readout.csv rows = 176
v124_task_trace.csv rows = 135
v124_task_triage.csv rows = 27
v124_task_failure_table.csv rows = 2
v124_control_matrix.csv rows = 16
```

B25 task aggregate（steady AUC-time 口径）：

| candidate | mean delta | worst delta | near pass rate | AUC-time mean ratio | AUC-time max ratio | ECE mean delta | ECE max delta | steady epochs | A3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B21a` | `-0.0015190972222222222` | `-0.0078125` | `0.5555555555555556` | `1.0968725773066839` | `1.6133225916511673` | `0.006518384648693932` | `0.030149083584547043` | `2` | `0` |
| `B21d` | `-0.009114583333333334` | `-0.01953125` | `0.3333333333333333` | `1.4228988595896712` | `2.1237021017958986` | `0.027369055483076308` | `0.0646195039153099` | `2` | `0` |

解释：B25 把 B24 的 `224x`-`280x` AUC-time 异常降回可解释范围，说明 B24 的失败主要来自 compile/warmup outlier 被计入 AUC-time；但 B21a/B21d 仍未同时满足 near pass、AUC max 与 ECE 子门，所以仍不能写成 base success。

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B21a` | `diagnostic_base_not_qualified` | `-0.007036109566962878` | `0` |
| `B21d` | `diagnostic_base_not_qualified` | `-0.002876382782057263` | `0` |

No-fake：

```text
rows_checked = 1976
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `1475352572ad5b66122b5eb37a84e5c24362c95f5019fc176df0178be088bbde` |
| `v124_efficiency_microbench.csv` | `861f6a1528ce108c207997274f9c4edc8aa4c68173c1190e3604238946870f6a` |
| `v124_initial_geometry_audit.csv` | `7e8f5394e74977557c2dbda66ecad649dfd2186a56c423edf27d9f61c43aae0b` |
| `v124_expression_battery.csv` | `4e5aebc4f7ee1ff447aac24c740029256fbe49c9b3ec42d80ab31bc0b850be2c` |
| `v124_task_triage.csv` | `8d7add6380e1e87105729961c90b01f08d97c677ce7685e52c5217d4df71b7ba` |
| `v124_control_matrix.csv` | `cd1f60dcc9057b1872542267fe94c76f0efc28f12a11784bcaf3c34340dd04e7` |
| `v124_route_decision.json` | `887faf07bcfd095306ac66afe9200ae253b179a8eb496dea0002bb14eab687f6` |
| `v124_provenance_audit.csv` | `1d189287200199ede4fc9cffb7340f25fdef315e31461c4769a81c262a9f72de` |

追加判断：B25 是有效的 runner/accounting 修复，但不是科学成功。现在的 task blocker 已从“测量明显错误”收窄为更真实的 near-pass、AUC max 与 ECE 稳定性问题。

## 19. B26 Plain Temperature Steady Repair 结果

> 本节继续尝试“不同 basis 初始化 + 低温初始化”的修复方向：不使用 B22b lowboost，而回到 B21a plain basis-specific norm，只调 `logit_gain=0.75/0.50`。结果仍只来自真实 artifact；未过 A1 时不推进 task。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_plain_temp_steady_20260520T210000Z
```

新增 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "expression_fail",
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_diagnostic_positive": true,
  "functional_open": false,
  "no_fake": true
}
```

新增落盘完整性：

```text
v124_basis_manifest.csv rows = 64
v124_efficiency_microbench.csv rows = 117
v124_initial_geometry_audit.csv rows = 63
v124_expression_battery.csv rows = 957
v124_frozen_readout.csv rows = 308
v124_task_trace.csv rows = 180
v124_task_triage.csv rows = 36
v124_task_failure_table.csv rows = 3
v124_control_matrix.csv rows = 28
```

B26 A1 efficiency 结果：

| candidate | eager step ratio | compiled step ratio | compiled A1 pass |
|---|---:|---:|---:|
| `B21a-plain` | `3.647107529853669` | `2.0297556244950945` | `0` |
| `B26a-temp075` | `3.8752594187520546` | `2.0069039499623567` | `0` |
| `B26b-temp050` | `5.616225348658636` | `1.9945619731968804` | `0` |

解释：B26a/B26b 在本轮没有通过 A1 compiled timing，且 route 的 A2 expression pass 为空。因此 B26 没有形成 task 结论，更不能覆盖 B25 的 task-gate结论；它只说明 plain B21a 低温变体本身不足以成为下一步 base repair。

Functional diagnostic positive rows：

| candidate | control gap | beats controls |
|---|---:|---:|
| `B12a-LegendreKAN-K4-fan-scale-repair` | `2.5224014241401074` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `2.978411122267488e-05` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `5.402795305364805e-06` | `1` |

解释：这些 positive rows 仍全部是 `diagnostic_base_not_qualified`，因为 base gate 未开；其中 B9 是 diagnostic-only 局部 quadratic coverage 路线，不能写成 official functional success。

No-fake：

```text
rows_checked = 3026
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `0c50c6db9b226a2b2799ce4b220e950e554ea31d13e386f8cfa2c7689c0ac1b3` |
| `v124_efficiency_microbench.csv` | `65caa665c1b9f5b482acbc85133cfc9e049e099036fe9585147e2d70055d6f16` |
| `v124_initial_geometry_audit.csv` | `8a2613e4f7ed0f712ac9369f94f83f36fef45c40a3c0e6b76808bf446d8e9c39` |
| `v124_expression_battery.csv` | `d0453f02056fb279f6a4e3b4c7b990bda5382cc6608f96ca2df5b7dfd0098dc0` |
| `v124_task_triage.csv` | `c7ac7c82ced882c20aff4b8581eed49d5e2b30cb7a9e454c2666a3571acea79a` |
| `v124_control_matrix.csv` | `116716c7437d1fc8ac5755e5e59c983d60d7851f1ac06047d11ab58a89c8d818` |
| `v124_route_decision.json` | `27699bd36592041f942431613b8f3ef83bad67d5c431c23c24fa373c02fba080` |
| `v124_provenance_audit.csv` | `20a4cf4e2dd8cdf409039c4f1a0d59829c56a9188579f2af411b0bf5098be196` |

追加判断：B26 没有给出可推进的 base candidate。结合 B25，当前继续小修 branch scale/logit temperature 的收益已经变低；更合理的下一步需要重新设计稳定 A1 timing 与 B21/B23-style calibration 的联合方案，而不是把 diagnostic functional route 打开。

## 20. B27 Warmup-Cosine LR Schedule Repair 结果

> 本节继续按 R3 task-fail 建议尝试全局 optimizer schedule repair。第一次 B27 执行后发现 B27 registration/compiled path 不完整，因此不能作为科学结论；修复后重跑 fixed artifact，B27a/B27b/B27c 均真实进入 A1/A2/A3。结果仍只来自真实 artifact，不打开 functional official route。

新增 artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_lr_schedule_repair_fixed_20260520T230000Z
```

新增 route：

```json
{
  "route": "R5-FunctionalDiagnosticPositiveBaseNotYetQualified",
  "base_blocker_route": "task_fail",
  "A2_expression_pass": [
    "B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm",
    "B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm",
    "B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm",
    "B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost",
    "B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075",
    "B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050",
    "B26b-GatedLegendreQuadratic-h224-basis-specific-temp050",
    "B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr",
    "B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr",
    "B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_diagnostic_positive": true,
  "functional_open": false,
  "no_fake": true
}
```

新增落盘完整性：

```text
v124_basis_manifest.csv rows = 67
v124_efficiency_microbench.csv rows = 123
v124_initial_geometry_audit.csv rows = 66
v124_expression_battery.csv rows = 1749
v124_frozen_readout.csv rows = 572
v124_basis_condition.csv rows = 572
v124_task_trace.csv rows = 495
v124_task_triage.csv rows = 99
v124_task_failure_table.csv rows = 10
v124_control_matrix.csv rows = 52
v124_functional_one_step.csv rows = 364
v124_functional_five_step.csv rows = 364
```

B27 A1 efficiency 结果：

| candidate | eager step ratio | compiled step ratio | compiled memory ratio | compiled A1 exploratory pass |
|---|---:|---:|---:|---:|
| `B27a-cosine-lr` | `4.099709233022273` | `1.3239846570397114` | `0.48053810434307564` | `1` |
| `B27b-residual-cosine-lr` | `4.254097274254706` | `1.192250613037259` | `0.49428093417099156` | `1` |
| `B27c-lowboost-temp050-cosine-lr` | `4.313570970415276` | `1.433143036350839` | `0.4946565146134936` | `1` |

B27 A2 expression 摘要：

| candidate | E1 B1 val R2 | E1 delta | E2 B1 val R2 | E2 delta | E6 B1 val R2 | E8 B1 val R2 |
|---|---:|---:|---:|---:|---:|---:|
| `B27a` | `0.990953803062439` | `-0.005814492702484131` | `0.995961606502533` | `0.0020989179611206055` | `0.9984036684036255` | `0.9982452392578125` |
| `B27b` | `0.9932902455329895` | `-0.0034780502319335938` | `0.9963485598564148` | `0.0024858713150024414` | `0.9942430853843689` | `0.9950351119041443` |
| `B27c` | `0.989189088344574` | `-0.007579207420349121` | `0.9960270524024963` | `0.0021643638610839844` | `0.9971893429756165` | `0.9962655901908875` |

解释：B27a/B27b/B27c 均真实通过 A2；LR schedule repair 没有破坏 expression gate。

B27 A3 task aggregate（failure-table 口径）：

| candidate | mean delta | worst delta | near pass rate | AUC-time mean | AUC-time max | ECE mean delta | ECE max delta | A3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B27a` | `-0.006944444444444444` | `-0.02734375` | `0.3333333333333333` | `1.0587445431768128` | `1.2439590852138214` | `-0.012792886959181892` | `0.008148998022079468` | `0` |
| `B27b` | `-0.011935763888888888` | `-0.03515625` | `0.3333333333333333` | `1.2779558721963937` | `1.9956916288833382` | `0.004631936136219237` | `0.030141711235046387` | `0` |
| `B27c` | `-0.006727430555555556` | `-0.02734375` | `0.4444444444444444` | `1.0237545198908917` | `1.166639746619922` | `-0.018747775091065302` | `0.007192067801952362` | `0` |

解释：B27c 是 B27 中最好的一支，warmup-cosine schedule 确实把 AUC-time mean 与 ECE mean 推近/推好；但 worst-row safety、near pass rate 与 AUC-time max 仍没过，所以不能写成 base success。所有 B27 task row 都记录：

```text
lr_schedule = linear_warmup20_cosine_final025
final_lr = 0.0005
compiled_task_path = 1
```

Functional diagnostic：

| candidate | status | control gap | beats controls |
|---|---|---:|---:|
| `B27a` | `diagnostic_base_not_qualified` | `-0.007036109566962878` | `0` |
| `B27b` | `diagnostic_base_not_qualified` | `-0.002876382782057263` | `0` |
| `B27c` | `diagnostic_base_not_qualified` | `-0.003090285627203926` | `0` |
| `B12a-LegendreKAN-K4-fan-scale-repair` | `diagnostic_base_not_qualified` | `2.5224014241401074` | `1` |
| `B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm` | `diagnostic_base_not_qualified` | `0.0008413767637345249` | `1` |
| `B9b-Poly2SignedPair-h64-diagnostic-repair` | `diagnostic_base_not_qualified` | `2.978411122267488e-05` | `1` |
| `B9d-Poly2PairRandom-h64-K2-diagnostic-repair` | `diagnostic_base_not_qualified` | `5.402795305364805e-06` | `1` |

解释：B27 自身没有 control-resistant functional signal；其他 positive rows 也仍是 base gate 关闭下的 diagnostic，不能写成 official success。

No-fake：

```text
rows_checked = 5718
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `d7d9021b600a1ee0d26bf69ef38522a849916cc42e99c2cb73efaec18579cbe0` |
| `v124_efficiency_microbench.csv` | `eeed21448667738988c3f16a2307cd66a0c03dac3e6ed4854e25b6ba019d5de5` |
| `v124_initial_geometry_audit.csv` | `576a0898fe485f08cc18e4e1fd4dbf9fc3f6e738db268d468cd5267c3a22f21c` |
| `v124_expression_battery.csv` | `dde083aae63ea6dfe4fa24780cf1e0a7e6618830bd3365489b540fa07a47be4b` |
| `v124_task_triage.csv` | `ee899fc67bc679c0c80addf38e6f23e6f767bcb59a7a72b5e0d480c85d932fd2` |
| `v124_control_matrix.csv` | `2240ed5447e9ee60c88ce5694cbbbc98533920df4a9b6553a2a8265ec8efb1f5` |
| `v124_route_decision.json` | `4ac3d1b0d0e6a6e784a7e4c370a071b7c8c8f1b84ba48fdd36f1a26aa2c7c1a6` |
| `v124_provenance_audit.csv` | `2f573df00a1d95d1d26361a9d1b3b8bc50558a75cbad7197e4e43755bf05fe58` |

追加判断：B27 是一次有效的 R3 task-repair 尝试，但不是成功。它证明 schedule repair 能在 B27c 上改善 AUC-time mean 和 ECE mean，却没有解决最差 split 与 AUC max。当前 blocker 已进一步收窄为：需要一个同时保留 B21/B23/B27 expression/calibration 优势、又能提高 worst-row task stability 的全局初始化/normalization/optimization 联合设计。

## 21. B28/B29 Schedule Shape 与 Efficiency Repair 结果

> 本节继续执行 B27 后的 blocker 修复。B28 尝试更短 warmup 与更高 final LR 来恢复 B23b 的 task stability；B29 按 B28 的 A1 failure 做 h224 轻量化；最后按 R1 efficiency blocker 启用 TF32/high matmul precision 重跑。所有结果仍只来自真实 artifact；A1 未过时不推进 A3，不把 targeted repair 写成 base success。

新增 artifacts：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b28_schedule_shape_stable_20260520T020000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b29_h224_schedule_20260520T030000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b29_h224_schedule_tf32_20260520T040000Z
```

新增 route（latest targeted repair）：

```json
{
  "route": "R1-EfficiencyAllFail",
  "base_blocker_route": "efficiency_all_fail",
  "A1_exploratory_survivors": [],
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_diagnostic_positive": false,
  "functional_open": false,
  "next_recommended_action": "implement fused/manual kernel for top two families only",
  "no_fake": true
}
```

B28/B29 A1 efficiency 结果：

| artifact | candidate | compiled step ratio | compiled memory ratio | A1 pass |
|---|---|---:|---:|---:|
| `B28 stable` | `B28a-h228-final050` | `1.9158250576881404` | `0.4808795411089866` | `0` |
| `B28 stable` | `B28b-h228-final075` | `1.559771730163763` | `0.48125512155148864` | `0` |
| `B29 h224` | `B29a-h224-final050` | `1.8647487730847898` | `0.47461417645452064` | `0` |
| `B29 h224` | `B29b-h224-final075` | `2.012577297149025` | `0.47498975689702266` | `0` |
| `B29 h224 + TF32` | `B29a-h224-final050` | `2.2507559698581194` | `0.47461417645452064` | `0` |
| `B29 h224 + TF32` | `B29b-h224-final075` | `2.314790957435887` | `0.47498975689702266` | `0` |

解释：B28/B29 都没有通过 A1 efficiency，因此没有合法进入 A3 task。TF32/high matmul precision 也没有修复这个分支，反而在本轮 targeted measurement 中让 MLP baseline 更快，ratio 仍不达标。

No-fake（latest targeted repair）：

```text
rows_checked = 61
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256（latest targeted repair）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `a92caf0cec4271b07b64f36d7c89b7b55cf79e69d5ac2c9f1177ff111249ca27` |
| `v124_efficiency_microbench.csv` | `b2aa5444e1470166746f5fb47d940b21ea268c59bcfcef7e478341bae70baebd` |
| `v124_initial_geometry_audit.csv` | `6cd211012f0ba963f77fa04d6dcdc192e0250d8d476e0d2a425fe3492f8701ae` |
| `v124_expression_battery.csv` | `17d33b0a399ca47972d67f247656edbc85924f22e651d3fd64c4859cfa83e620` |
| `v124_task_triage.csv` | `399cafc145d1e2ff5272744855d28d2e2eeb61cb6702a205f92dea74a61714c1` |
| `v124_control_matrix.csv` | `006a7aededa28db873fe47ee3f27dd65cfe1e883f7c9029e997e1a607b4eaad6` |
| `v124_route_decision.json` | `4798272c073de65d9a0ee5d0a9e61634099acbd1a26916774e5b659ca2635858` |
| `v124_provenance_audit.csv` | `395e6b2feee8682274c199ba6bd2dbdab56fab561f0f83b013be24a8430fdaaf` |

追加判断：B28/B29 说明继续只调 schedule shape 或小幅降 hidden 已经触到效率墙；下一步若还在 v12.4 主线内推进，应优先做真正的 fused/manual GatedHybrid kernel 或把 B21/B23/B27 的机制迁移到更便宜的 primitive，而不是继续增加 h228/h224 schedule 变体。

## 22. B30-B36 Lite / Small Gated Repair 结果

> 本节继续执行 B29 后的 blocker 修复：先把 B21/B23/B27 的分支专属 norm/temperature/schedule 迁移到更便宜的 primitive；失败后再回到 small GatedHybrid，并围绕 A3 的 task/AUC/ECE blocker 做 hidden/temperature/direct-skip 修复。所有结果仍只来自真实 artifact，不打开 official functional。

新增 artifacts：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b30_lite_gated_inputnorm_20260520T050000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b31_lite_quadboost_inputnorm_20260520T060000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b32_small_gated_inputnorm_20260520T070000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b33_small_gated_temp_hidden_20260520T080000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b34_auc_task_tradeoff_20260520T090000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b35_temp065_auc_task_20260520T100000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b36_directskip_task_stability_20260520T110000Z
```

新增 route（latest targeted repair）：

```json
{
  "route": "R3-ExpressionPassTaskFail",
  "base_blocker_route": "task_fail",
  "A1_exploratory_survivors": [
    "B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050",
    "B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050",
    "B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075"
  ],
  "A2_expression_pass": [
    "B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050",
    "B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050",
    "B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075"
  ],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_diagnostic_positive": false,
  "functional_open": false,
  "no_fake": true
}
```

### 22.1 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `LiteGatedLegendreQuadraticKAN` | 低成本迁移 B21/B23/B27：Legendre 分支用 bounded/tanh input norm，quadratic 分支用 RMS-normalized linear input 和 feature-scale calibration；不使用 label/outcome。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B30/B31 | B30 测试 lite basis-specific input norm，B31 测试 quadratic branch midboost/quadboost 与 temp 初始化，验证“不同 basis 需要不同初始化”。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B32/B33/B34/B35 | small GatedHybrid hidden/temperature/final-LR 网格，只围绕 B32/B33 的真实 A3 blocker 做窄修复；没有降低 gate。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B36 direct Legendre edge-basis skip | 结构性合并 B30/B31 的 early task 坐标与 B32/B33 的表达深度；direct skip 仍是 edge-basis readout，不是 MLP shortcut。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 接入 B30-B36 model_kind / candidate filter / compiled repair / LR schedule | 所有新增候选都走相同 A1/A2/A3/functional/no-fake pipeline；A3 未过时 functional 只写 diagnostic。 |

### 22.2 B30/B31 LiteGated 结果

| artifact | candidate | compiled A1 | E1 B1 delta | E6 B1 delta | E8 B1 delta | task mean delta | task AUC mean | route |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `B30` | `B30b-h256-temp050` | `1` | `-0.05942237377166748` | `-0.047035396099090576` | `-0.05445504188537598` | `0.0015190972222222222` | `1.1084359273573485` | `R2` |
| `B31` | `B31a-h224-midboost-temp100` | `1` | `-0.06005513668060303` | `-0.05676376819610596` | `-0.060099899768829346` | `0.008246527777777778` | `1.0345282605105663` | `R2` |

解释：Lite/direct-readout 结构证明 input norm 与 branch init 能改善 task geometry，但 A2 expression 无法追上 MLP envelope。B31a 的 task 已接近，但 direct-readout 缺少 B21/B23 的两层 Legendre 表达深度，因此不能成为 base。

### 22.3 B32-B36 Small Gated 结果

| artifact | best candidate | compiled step ratio | A2 pass | task mean delta | worst delta | near pass | AUC mean | AUC max | ECE max | route |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B32` | `B32c-h192-temp075-final075` | `1.358197619935907` | `1` | `-0.004557291666666667` | `-0.021484375` | `0.5555555555555556` | `1.2519666768278948` | `1.9257017999986692` | `0.041500091552734375` | `R5` |
| `B33` | `B33c-h176-temp075-final075` | `0.8106067708434458` | `1` | `0.005208333333333333` | `-0.01171875` | `0.7777777777777778` | `1.1852771821233192` | `1.5840523720855126` | `0.02069440484046936` | `R3` |
| `B34` | `B34c-h160-temp075-final050` | `1.1364238316568809` | `1` | `-0.013454861111111112` | `-0.03125` | `0.2222222222222222` | `1.0852577153521605` | `1.4070173269200534` | `0.019888296723365784` | `R3` |
| `B35` | `B35b-h176-temp065-final050` | `0.9842848973937696` | `1` | `-0.00043402777777777775` | `-0.01171875` | `0.5555555555555556` | `1.1104985285493005` | `1.7533748125949657` | `0.003914099186658859` | `R3` |
| `B36` | `B36c-h176-temp075-directskip` | `1.0688823926853201` | `1` | `0.005208333333333333` | `-0.01171875` | `0.6666666666666666` | `1.1888670638347314` | `2.1617626226569753` | `0.017469502985477448` | `R3` |

解释：B32-B36 说明 small GatedHybrid 已经可以稳定通过 A1+A2，并且 B33c/B36c 的 task mean 与 worst-row 明显接近 gate；但 AUC-time 全 row `<= 1.05` 的硬门仍没过，near pass 也最高只有 `0.7777777777777778`。B36 direct skip 没有修复 AUC max，说明继续简单 hidden/temperature/direct-skip 小网格收益已经变低。

Functional diagnostic：

| artifact | candidate | control gap | beats controls | official functional |
|---|---|---:|---:|---:|
| `B32` | `B32a-h160-temp050` | `0.002688993613471302` | `1` | `0` |
| `B33` | `B33c-h176-temp075` | `-0.0029276541902074626` | `0` | `0` |
| `B35` | `B35b-h176-temp065` | `-0.0025282518559963663` | `0` | `0` |
| `B36` | `B36c-h176-temp075-directskip` | `-0.0031557801116068873` | `0` | `0` |

解释：B32a 有 diagnostic functional positive，但 base gate 关闭，所以不能打开 official functional 或 P5；B33-B36 的 functional diagnostic 不 beat strong controls。

No-fake（latest targeted repair）：

```text
rows_checked = 757
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256（latest targeted repair）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `e3f795cd0c6b83cb95b82abc4eec109a355360aa7f1aede1aa974e0b8ebbc4f6` |
| `v124_efficiency_microbench.csv` | `352e6ab375a86ff59005f1908785834c55cedd264bfff640ec7b2c0c7e287e86` |
| `v124_initial_geometry_audit.csv` | `98204b7056ae9d25d2c166d66a7d5fdffe9a7cbd0f4b9f3edf129bfb012d767c` |
| `v124_expression_battery.csv` | `c0aa8e1ecd15a4b400890525be745b2145efe783e6e6760e79ed513a72280bee` |
| `v124_task_triage.csv` | `da688b93be0661718af6053139f1c4ed2d5e92d571e46bf0d3f816cdcea65dc2` |
| `v124_control_matrix.csv` | `e8c32054d9f323ecec147c7ab7773fb94a39b9f69f71980c551e9bda448a5e66` |
| `v124_route_decision.json` | `75d7d78e9269185db3aad850e1a43bed5723502fd51101810370d090adf060be` |
| `v124_provenance_audit.csv` | `2fb776b5099c581b419d0317c8ee053cc910c670aeb0b9843a5fa46166ef67a6` |

追加判断：B30-B36 确认了用户提出的方向是对的：不同 basis 确实需要不同 input norm 和初始化，且 small GatedHybrid 是当前最强主线。但 v12.4 仍没有 base success。下一步不应继续扫 hidden/temperature 小网格，而应设计结构性 AUC-time / near-pass 修复，例如 task-time-aware optimizer accounting、per-branch update scale、或真正 fused/manual kernel 后再重测同一 B33/B35/B36 族。

## 23. B37/B38 Branch-wise Optimizer 与 B36 Compile-Warm Audit

> 本节继续执行 R3 `global optimizer/initialization repair; no dataset-specific tuning`。B37/B38 不改变 B36 architecture / init，只改变全局参数角色的 optimizer update scale；B36 compile-warm audit 只改变真实 task compile warmup 步数，不改变 gate 或训练数据。

新增实现：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 B37/B38 candidate ids | 复用 B36 directskip 结构与 init，便于隔离 optimizer profile 的影响。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `--task-compile-warmup-steps` | 真实执行多步 fwd/bwd warmup，不更新参数，用于减少 compiled task path 晚发编译尖峰。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 B37 `branchslow_quad050_direct075_gate020` profile | quadratic LR = `0.50x`，direct LR = `0.75x`，gate/logit LR = `0.20x`；全局按参数名分组，不看数据集/标签。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 B38 `legdirectfast_quad075_gate020` profile | Legendre/direct LR = `1.25x`，quadratic LR = `0.75x`，gate/logit LR = `0.20x`；用于反向检查 B37 是否过慢。 |

新增 artifacts：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b37_branchslow_compilewarm_20260520T120000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b38_legdirectfast_compilewarm_20260520T130000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b36_compilewarm_audit_20260520T140000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b39_temp075_final050_compilewarm_20260520T150000Z
```

### 23.1 Route

| artifact | route | A1 survivors | A2 pass | A3 pass | functional diagnostic positive |
|---|---|---:|---:|---:|---:|
| `B37 branchslow` | `R3-ExpressionPassTaskFail` | `1` | `1` | `0` | `0` |
| `B38 legdirectfast` | `R1-EfficiencyAllFail` | `0` | `0` | `0` | `0` |
| `B36 compile-warm audit` | `R3-ExpressionPassTaskFail` | `3` | `3` | `0` | `0` |
| `B39 temp075/final050` | `R1-EfficiencyAllFail` | `0` | `0` | `0` | `0` |

### 23.2 B37/B38 结果

| artifact | candidate | optimizer profile | compiled step ratio | task mean delta | near pass | worst delta | AUC mean | AUC max | ECE max | route |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B37` | `B37b-h176-temp065-branchslow` | `branchslow_quad050_direct075_gate020` | `1.4967179505370454` | `-0.03624131944444445` | `0.1111111111111111` | `-0.05859375` | `1.4957762948801485` | `2.0992698798397136` | `0.08234763890504837` | `R3` |
| `B38` | `B38a-h176-temp075-legdirectfast` | `legdirectfast_quad075_gate020` | `2.3515869810362546` |  |  |  |  |  |  | `R1` |
| `B38` | `B38b-h176-temp065-legdirectfast` | `legdirectfast_quad075_gate020` | `2.422857574375153` |  |  |  |  |  |  | `R1` |
| `B38` | `B38c-h160-temp050-legdirectfast` | `legdirectfast_quad075_gate020` | `2.1287205147748334` |  |  |  |  |  |  | `R1` |
| `B39` | `B39a-h176-temp075-final050` | `default` | `2.2433983063496603` |  |  |  |  |  |  | `R1` |

解释：B37 证明“统一放慢 quadratic/direct/gate 更新”会伤害 early task geometry；B38 证明“加快 Legendre/direct 主表达分支”没有通过 A1 system gate。二者都不是 base repair。

### 23.3 B36 Compile-Warm Audit

| candidate | compile warmup steps | compiled step ratio | task mean delta | near pass | worst delta | AUC mean | AUC max | ECE max | A3 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B36a-h160-temp050-directskip` | `12` | `1.474015217542415` | `-0.011501736111111112` | `0.4444444444444444` | `-0.033203125` | `1.146875261356213` | `1.4032196274126607` | `0.01071460172533989` | `0` |
| `B36b-h176-temp065-directskip` | `12` | `1.3167690106611405` | `-0.0008680555555555555` | `0.5555555555555556` | `-0.01171875` | `1.34809994940525` | `2.208847227867373` | `0.0006182901561260223` | `0` |
| `B36c-h176-temp075-directskip` | `12` | `1.194497791309914` | `0.005208333333333333` | `0.6666666666666666` | `-0.01171875` | `1.105288806659117` | `1.6440778149943576` | `0.017469502985477448` | `0` |

对比 B36 原 run：B36c AUC mean/max 从 `1.1888670638347314` / `2.1617626226569753` 降到 `1.105288806659117` / `1.6440778149943576`。这说明 compile warmup artifact 的确存在，但修完后 AUC max 仍不达标，near pass 仍只有 `0.6666666666666666`，所以不能把 A3 写成 pass。

Functional diagnostic：

| artifact | candidate | control gap | beats controls | official functional |
|---|---|---:|---:|---:|
| `B37` | `B37b-h176-temp065-branchslow` | `-0.0035683344020749352` | `0` | `0` |
| `B36warm` | `B36a-h160-temp050-directskip` | `-0.00025665033444122187` | `0` | `0` |
| `B36warm` | `B36b-h176-temp065-directskip` | `-0.0035683344020749352` | `0` | `0` |
| `B36warm` | `B36c-h176-temp075-directskip` | `-0.0031557801116068873` | `0` | `0` |

No-fake：

| artifact | rows checked | fake/proxy/cpu |
|---|---:|---|
| `B37` | `367` | `0 / 0 / 0` |
| `B38` | `67` | `0 / 0 / 0` |
| `B36warm` | `757` | `0 / 0 / 0` |
| `B39` | `55` | `0 / 0 / 0` |

关键 artifact SHA256（latest B36 compile-warm audit）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `e3f795cd0c6b83cb95b82abc4eec109a355360aa7f1aede1aa974e0b8ebbc4f6` |
| `v124_efficiency_microbench.csv` | `8091b32a8964ac25cbca3c02296f4d4df2837828e2fcaae94bf796825902b30f` |
| `v124_expression_battery.csv` | `fa0c21727dec61d9205987c556f4bb74b3df5b736f1093f6b73acc2034c13964` |
| `v124_task_triage.csv` | `8aa1fd6c4c40355d28956b81d85446d3b08fb3bf7280c6ba684ab70f71d9f899` |
| `v124_task_failure_table.csv` | `f1c3667696ed8249f985a0a8a5e87c886dd651ff2fc16000389515d364df6df7` |
| `v124_control_matrix.csv` | `e8c32054d9f323ecec147c7ab7773fb94a39b9f69f71980c551e9bda448a5e66` |
| `v124_provenance_audit.csv` | `2fb776b5099c581b419d0317c8ee053cc910c670aeb0b9843a5fa46166ef67a6` |
| `v124_route_decision.json` | `75d7d78e9269185db3aad850e1a43bed5723502fd51101810370d090adf060be` |
| `v124_run_manifest.json` | `d2c07717f6c6312fb0a674f41a89ea672b270569f78a766c5b8501c0d96db24f` |

关键 artifact SHA256（latest targeted B39）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `a4ca6e13a3846136715746b751f674055d60cd889742912ed9826aea032b8030` |
| `v124_efficiency_microbench.csv` | `030c630dfa294fbcb26e6002c4b2380a4eaa5514e5b29d2328411b73c6cacd61` |
| `v124_expression_battery.csv` | `e04ca839c148116c8bba3d378064623131d8d82a5365bb04862dc07d479c2103` |
| `v124_task_triage.csv` | `399cafc145d1e2ff5272744855d28d2e2eeb61cb6702a205f92dea74a61714c1` |
| `v124_task_failure_table.csv` | `94ed33b7e701a49823f2b2a45ab9b7cc16892dec34b30f3a24d326b1eabea8ae` |
| `v124_control_matrix.csv` | `006a7aededa28db873fe47ee3f27dd65cfe1e883f7c9029e997e1a607b4eaad6` |
| `v124_provenance_audit.csv` | `9263047c7447440b48fdf20d3a8c89ed150f0c39521b7953857adb31ec7702b2` |
| `v124_route_decision.json` | `4798272c073de65d9a0ee5d0a9e61634099acbd1a26916774e5b659ca2635858` |
| `v124_run_manifest.json` | `b8f050e3afe554358d6761d96a3f194e44f587a728a64d54d33bcdddbcd98ea8` |

追加判断：B37/B38 说明 branch-wise optimizer scale 不是当前简单解；B36 compile-warm audit 说明 measurement artifact 已部分修复，但真正剩余 blocker 是 task near-pass 与 per-row AUC max。下一步如果继续 v12.4，应优先做 architecture-level speed/variance repair（例如 fused/manual GatedHybrid kernel 或减少 compiled training outlier 的真实 kernel path），同时保持 B36c 的表达/accuracy 结构，不应再扩大 hidden/temperature/optimizer 小网格。

B39a 进一步说明：单独把 B36c 的 final LR 从 `0.75` 降到 `0.50` 没有形成可测 task repair，因为它先停在 A1 efficiency。此结果支持同一个判断：现在主 blocker 已从 basis/norm/init 小修转向 kernel/system variance 与稳定训练路径。

## 24. B40-B46 Fast-Reuse / Input-Norm 后续修复

> 本节继续回应 v12.4 route 的要求与用户提出的“不同 basis 需要不同初始化，并且输入需要 norm 层以改善训练初期几何”。所有新增修复都保持 strict edge-basis、fixed train-stream normalization、CE-only、no dataset branch；base 未过时 functional 仍只写 diagnostic。

新增实现：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v124_multibasis_functional_dual.py` | 新增 `--task-compile-warmup-optimizer-steps` 与 optimizer state restore | 真实执行 optimizer-step warmup 后恢复模型/optimizer 状态，用于检查 optimizer step compilation 是否污染 task-time；不改训练数据和 gate。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B41 h168 directskip | 在 B36c 附近做轻量化，尝试降低 compiled step outlier；结构仍是 edge-basis direct readout，不是 MLP shortcut。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B42/B43 fast-reuse forward path | 对 directskip GatedHybrid 复用 input norm 与 Legendre basis，减少重复计算；数值等价审计 max abs error = `0.0`。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B44 residual mix15 input norm | 继续设计 norm 层强度，检查更弱 block-whiten 混入是否改善 early geometry。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B45 B42b-final050 schedule point | 保持 B42b 的 norm/init/fast-reuse，只做全局 final LR schedule 修复，检查 AUC/ECE 是否可改善。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B46 directskip050 | 保持 B42/B45 fast-reuse family，只把 fixed direct edge-basis skip scale 从 `0.25` 提到 `0.50`，用于检验 direct readout task-geometry 是否能增强 near pass。 |
| `experiments/run_v124_multibasis_functional_dual.py` | 接入 B41-B46 compile/schedule routing | 所有新增候选仍先过 A1/A2 才允许 A3；未过 gate 时不写 success。 |

新增 artifacts：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b40_optimizerwarm_b36c_20260520T160000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b41_h168_directskip_compilewarm_20260520T170000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b42_fastreuse_compilewarm_20260520T180000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b43_h172_fastreuse_compilewarm_20260520T190000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b44_mix15_fastreuse_compilewarm_20260520T200000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b45_b42b_final050_20260520T210000Z
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b46_directskip050_fastreuse_20260520T220000Z
```

### 24.1 Route

| artifact | route | A1 survivors | A2 pass | A3 pass | key blocker |
|---|---|---:|---:|---:|---|
| `B40 optimizer warmup` | `R1-EfficiencyAllFail` | `0` | `0` | `0` | optimizer-step warmup did not repair A1 |
| `B41 h168 directskip` | `R1-EfficiencyAllFail` | `0` | `0` | `0` | light h168 still too slow |
| `B42 fast-reuse` | `R3-ExpressionPassTaskFail` | `1` | `1` | `0` | task mean/near/AUC/ECE fail |
| `B43 h172 fast-reuse` | `R3-ExpressionPassTaskFail` | `1` | `1` | `0` | task mean/worst/AUC/ECE fail |
| `B44 mix15 fast-reuse` | `R1-EfficiencyAllFail` | `0` | `0` | `0` | weaker norm mix hurt A1 |
| `B45 B42b final050` | `R3-ExpressionPassTaskFail` | `1` | `1` | `0` | task mean/near/AUC/ECE fail |
| `B46 directskip050` | `R1-EfficiencyAllFail` | `0` | `0` | `0` | direct skip scale increase hurt A1 |

### 24.2 A1/A3 摘要

| artifact | best candidate | compiled step ratio | task mean delta | near pass | worst delta | ECE mean delta | route |
|---|---|---:|---:|---:|---:|---:|---|
| `B40` | `B36c + optimizer-warmup-restore` | `1.9544235620011423` |  |  |  |  | `R1` |
| `B41` | `B41b-h168-temp065-directskip` | `1.989416` |  |  |  |  | `R1` |
| `B42` | `B42b-h168-temp075-fastreuse-final075` | `1.337654524030241` | `-0.009331597222222222` | `0.2222222222222222` | `-0.017578125` | `0.0016034146149953206` | `R3` |
| `B43` | `B43b-h172-temp065-fastreuse-final050` | `1.0976061936748942` | `-0.011935763888888888` | `0.4444444444444444` | `-0.037109375` | `-0.011586784074703852` | `R3` |
| `B44` | `B44a/B44b-mix15` | `1.9539927866985813` / `1.9242401698113432` |  |  |  |  | `R1` |
| `B45` | `B45a-h168-temp075-fastreuse-final050` | `1.178873745280927` | `-0.009331597222222222` | `0.3333333333333333` | `-0.0234375` | `-0.0027160458266735077` | `R3` |
| `B46` | `B46a/B46b-directskip050` | `1.9061898744500823` / `1.6355462392377917` |  |  |  |  | `R1` |

解释：B42 的 fast-reuse 是真实 architecture-level speed repair：B42b 从 B41 的 R1 推进到 A1+A2。B45 的 final050 能改善 ECE mean，但不能修 task mean / near pass / AUC hard gate。B44 说明更弱的 residual block-whiten mix 不是当前可用 norm 设计；B46 说明单纯提高 directskip scale 也不能免费改善 task geometry，因为它先破坏了 A1。

Functional diagnostic：

| artifact | candidate | control gap | beats controls | official functional |
|---|---|---:|---:|---:|
| `B42` | `B42b-h168-temp075-fastreuse-final075` | `-0.00036641042037732774` | `0` | `0` |
| `B43` | `B43b-h172-temp065-fastreuse-final050` | `-0.002758544224648052` | `0` | `0` |
| `B45` | `B45a-h168-temp075-fastreuse-final050` | `-0.00036641042037732774` | `0` | `0` |

No-fake：

| artifact | rows checked | fake/proxy/cpu |
|---|---:|---|
| `B40` | `55` | `0 / 0 / 0` |
| `B41` | `61` | `0 / 0 / 0` |
| `B42` | `367` | `0 / 0 / 0` |
| `B43` | `361` | `0 / 0 / 0` |
| `B44` | `61` | `0 / 0 / 0` |
| `B45` | `355` | `0 / 0 / 0` |
| `B46` | `61` | `0 / 0 / 0` |

关键 artifact SHA256（B42 fast-reuse）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `f8ce17de67de647546ec888a857746db448ad104997047425f89693d69b56215` |
| `v124_efficiency_microbench.csv` | `a6c8d16c8130f34ac1babe7fc094ea705b45b0bc96bbb83c28f8c08d8667497e` |
| `v124_expression_battery.csv` | `6772c53484a408b1f63ccfcccff33ceb239c953c22c150e06e39d42f29076b91` |
| `v124_task_triage.csv` | `8f6df927f0e8bddf457314025f1a50dcbd668019db35019e74ce817601383c24` |
| `v124_control_matrix.csv` | `d43dd8cbe419be28a5655f390d2115a76764615dbb2df3ef569ecd4e9cf0b55a` |
| `v124_provenance_audit.csv` | `7681d45fadc5a6211167234f3c37d2ae05de0a61645be58295b818fcb10b7b26` |
| `v124_route_decision.json` | `8ffcdffccce8598ea64f15ccb095d77692b9a27afae7c18c87ae3d0d4a394b12` |
| `v124_run_manifest.json` | `9ce628416be0bf568441edf4e7a84a75fef55afa6afa547f224a2c9174d2d8ae` |

关键 artifact SHA256（latest B45）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `4586cedc46551c25e67fd3d202cc3e74b100e84c65ab446d63181cb2bf7023e1` |
| `v124_efficiency_microbench.csv` | `84b6384852c98473709d1f9550a5bf0c02c0e7ef7e0dbdc3431ebfdae2862f3c` |
| `v124_initial_geometry_audit.csv` | `bb1b6f4f15f478283da62f62bcf2a959f2006e73cbe07520aa5d3751a0a37aec` |
| `v124_expression_battery.csv` | `1f7c4baeef340b0771380c0a28cbe29e365aa0480de7e0795a5539640415f234` |
| `v124_task_triage.csv` | `3d7709a864b1f09b67dbeb6e4d376e4dd8845bd8d7dec7cc97ce497ca241f62e` |
| `v124_task_failure_table.csv` | `04cf455536c3441ff8bf1acf09cb56f99a57b56230566615e2f0f9584f254a09` |
| `v124_control_matrix.csv` | `a38b9010410aa0fe7e5a3351d2bb19c4faf8f88a57566106b089d4c22da745b6` |
| `v124_provenance_audit.csv` | `f3d91166ebeb3e44a1e4300f5cea824b5fb8d7150aba02817e60a4532288f1e9` |
| `v124_route_decision.json` | `b489c8aac4e18654eb9ec5f4996b6d34b5815ab4ed1ca70029bbec82cf742d5e` |
| `v124_run_manifest.json` | `832ebdf92efd1dc838e0dd385d7af66dc9481fe4082bece478291161377cd188` |

关键 artifact SHA256（latest B46）：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `a7ff64c96c857a4d428d1e14b68a7d7b686f2d953b95de74e22ec1df3d281f17` |
| `v124_efficiency_microbench.csv` | `a0a21a2ec7d69d859a4bbbc7f93d1216888c23e6daf00b1ab017239abbb94819` |
| `v124_expression_battery.csv` | `c7daf338199183ca0fd28ad05eefffd618b96b87b93e36f36147ef17adbe41b8` |
| `v124_task_triage.csv` | `399cafc145d1e2ff5272744855d28d2e2eeb61cb6702a205f92dea74a61714c1` |
| `v124_control_matrix.csv` | `006a7aededa28db873fe47ee3f27dd65cfe1e883f7c9029e997e1a607b4eaad6` |
| `v124_provenance_audit.csv` | `395e6b2feee8682274c199ba6bd2dbdab56fab561f0f83b013be24a8430fdaaf` |
| `v124_route_decision.json` | `4798272c073de65d9a0ee5d0a9e61634099acbd1a26916774e5b659ca2635858` |
| `v124_run_manifest.json` | `40fd2353e742e004397829c57560f6b2b7316c1e651b679cefb53a99bbd98492` |

追加判断：B42-B46 说明“输入 norm + basis-specific init + fast-reuse”方向确实能推进工程门，但当前 GatedHybrid directskip 家族仍不能稳定满足 A3。B45 已把 ECE mean 修好一些，但 near pass 与 AUC/time 没过；B46 试图加强 direct readout task geometry，却先失败在 A1。下一步不能调低 gate，应转向真正 fused/manual backward kernel 或新的 task-geometry primitive，而不是继续在 B42/B45/B46 上做 schedule/scale 小修。

## 25. B47 Manual Backward Kernel 修复结果

本节执行上一节的推荐方向：对 B42/B45 directskip fast-reuse family 做真正 manual backward。结果仍只来自落盘 artifact；B47 没有过 A1，因此 A2/A3/official functional 合法关闭。

新增修改审计：

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `_GatedLQFastReuseManualFunction` | 手写 Legendre K=4、quadratic sketch、direct edge-basis skip 的参数梯度；输入不需要梯度，backward 只返回 edge/basis 参数梯度。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B47a/B47b manual backward candidate | 保持 B42/B45 的 basis-specific residual input norm、temp075、directskip、fast-reuse，只替换 backward 实现；不改 gate、不加 MLP shortcut。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `manual_ce_forward_cache` / `manual_ce_backward_from_cache` | custom autograd 版在 `torch.compile` 包装时暴露 shape mismatch，因此继续实现 runner 可直接调用的 CE manual step，绕开 torch autograd graph。 |
| `experiments/run_v124_multibasis_functional_dual.py` | A1/A3 接入 manual CE step | 对 manual-capable model 直接写入 `.grad` 后交给 AdamW update；落盘 `manual_forward_available`、`manual_backward_available`、`uses_torch_autograd_graph`。 |

本轮没有做：

```text
1. 没有调低 A1/A2/A3 或 functional gate。
2. 没有按 dataset/label/outcome 分支。
3. 没有把 custom autograd compile failure 写成 success。
4. 没有把 manual step 的 R1 失败伪装成 task/base 结论。
```

### 25.1 Custom Autograd 版

artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b47_manualbw_20260520T230000Z
```

梯度正确性：

| candidate | manual forward | manual backward | grad relerr max | grad cos min | output max abs error | final claim allowed |
|---|---:|---:|---:|---:|---:|---:|
| `B47a` | `1` | `1` | `0.00017712752742227167` | `0.9999998807907104` | `8.940696716308594e-08` | `1` |
| `B47b` | `1` | `1` | `0.00017712752742227167` | `0.9999998807907104` | `8.940696716308594e-08` | `1` |

但 custom autograd 版不能作为最终 fused kernel：`torch.compile` 包装时失败，错误为 gradient index `8` shape mismatch，返回 `[784, 10, 4]`，期望 `[42, 10, 4]`。因此继续做 manual CE step 修复。

Custom autograd eager A1：

| candidate | step ratio | forward ratio | backward ratio | memory ratio | A1 pass |
|---|---:|---:|---:|---:|---:|
| `B47a` | `5.806861456720137` | `8.399033551521399` | `7.328342679485727` | `1.046759765091505` | `0` |
| `B47b` | `4.12779053734397` | `6.521241912219893` | `4.525902896297827` | `1.0817570335973778` | `0` |

### 25.2 Manual CE Step 版

artifact：

```text
results/v12_4_multibasis_functional_dual/v124_multibasis_functional_dual_b47_manualstep_20260520T233000Z
```

新增 route：

```json
{
  "route": "R1-EfficiencyAllFail",
  "base_blocker_route": "efficiency_all_fail",
  "A1_exploratory_survivors": [],
  "A2_expression_pass": [],
  "A3_task_pass": [],
  "base_qualified": false,
  "functional_open": false,
  "functional_diagnostic_positive": false,
  "no_fake": true
}
```

Manual CE step A1：

| candidate | manual forward | manual backward | uses autograd graph | step ratio | forward ratio | backward ratio | memory ratio | A1 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B47a` | `1` | `1` | `0` | `2.790044437100059` | `7.146656692314237` | `1.9335851063186962` | `1.1222514340344167` | `0` |
| `B47b` | `1` | `1` | `0` | `2.6246493524945507` | `6.7957968057457405` | `1.8717144531792889` | `1.1222514340344167` | `0` |

解释：manual CE step 明显降低 backward ratio 和 total step ratio，相比 custom autograd eager 从 `4.13x-5.81x` 降到 `2.62x-2.79x`，说明手写 backward 是真实改动；但 forward ratio 仍约 `6.8x-7.1x`，总 step 仍远高于 A1 `1.50x` exploratory gate。因此 B47 不能进入 A2/A3，更不能打开 functional。

No-fake：

```text
rows_checked = 59
fake/proxy/cpu = 0 / 0 / 0
```

关键 artifact SHA256：

| artifact | SHA256 |
|---|---|
| `v124_basis_manifest.csv` | `c837bc1624717079020dc0621b81c657a95d08f6a204def788631dacc9cb63ba` |
| `v124_grad_correctness.csv` | `534771bd66ca6eab60efe672f9c62df46c44cbaab95658f4bce670b87a911d75` |
| `v124_efficiency_microbench.csv` | `2342cd2dad65bea9f9f5c51e48bb4d48e8777fb8a83c771be4f01e5f308d878c` |
| `v124_efficiency_failure_table.csv` | `b99826057300f352401bbd3a5fd1f71df30ee4cbdbce37b584c1b403bf7d4c90` |
| `v124_provenance_audit.csv` | `788404fe78edcf092d8a578aa7a3cd0ac4ebcc86893b97e3bab66f81ab24117e` |
| `v124_route_decision.json` | `4798272c073de65d9a0ee5d0a9e61634099acbd1a26916774e5b659ca2635858` |
| `v124_run_manifest.json` | `833d3f5ca15047332deb50ab4e044027f2d534efeba5d9670a09488553e10278` |

追加判断：B47 说明 Python-level manual backward 不足以修复 v12.4 的效率门；真正 blocker 已从 backward autograd graph 转移到 forward/kernel launch 与 input norm/Legendre/quadratic 多算子组合。下一步若继续效率路线，不能再只手写 backward，应实现 lower-level fused CUDA/Triton forward+backward kernel，或转向更简单的 task-geometry primitive。




### 现在我们正在做哪些基函数？哪些满足要求了哪些还没有？
现在主线在做的基函数已经收窄了：不是再平均扫所有 KAN primitive，而是在围绕 **Legendre + quadratic sketch 的多基函数组合**继续修效率、表达、任务稳定性，最近还推进到了 fast-reuse 和 manual backward/CE step。

结论先说清楚：**目前没有任何基函数/组合完全满足 v12.4 要求**。也就是没有 candidate 同时通过 `A1 efficiency + A2 expression + A3 task/base`，所以 `base_qualified = false`，`functional_open = false` 仍然正确。

| 基函数 / 组合 | 代表 candidate | 已满足 | 未满足 | 当前判断 |
|---|---|---|---|---|
| ReLU/RSWAF hinge、RBF/FastKAN、Chebyshev、Fourier、Wavelet、B-spline、Rational | B1-B8 | 已真实扫过 | 没形成 A1+A2+A3 联合通过 | 非当前主线 |
| Poly2 / signed pair | B9 系列 | 局部 pairwise coverage 有价值 | rotated/random quadratic、task stability 不够 | diagnostic，不是 base |
| Fixed quadratic sketch | B10 | 表达诊断有价值 | diagnostic-only，不能 official base | 只能做 oracle/coverage 对照 |
| Trainable quadratic sketch | B11 | 可训练表达方向尝试过 | 没形成 base gate success | 未满足 |
| LiteGated Legendre direct + quadratic sketch | B30/B31 | A1 和任务均值信号一度很好，B31a mean delta `+0.0082465` | A2 expression 失败 | 有任务潜力，但表达不够 |
| Gated Legendre + Quadratic | B21/B22/B27/B33/B36/B42/B45 | 多个版本 A1+A2 通过 | A3 task/base 全部失败 | 当前最接近但仍未过 |
| Fast-reuse GatedHybrid | B42/B45 | fast-reuse forward equivalence `0.0`，B45a A1+A2 通过 | task mean delta 仍负，AUC/ECE gate 失败 | 工程效率进展，科学 base 未过 |
| Manual backward / manual CE step | B47 | 梯度校验通过：cos `0.99999988`，relerr `0.000177`，真实不用 torch autograd graph | B47b step ratio `2.6246`，A1 失败 | kernel 正确性过了，效率还没过 |

最接近“满足要求”的是 **Gated Legendre + Quadratic** 这一族：

- `B21d`：A2 很强，E1/E2/E6/E8 表达 delta 接近 0 或略正，frozen R2 都 > `0.89`，但 A3 失败。
- `B33c` / `B36c`：A1+A2 通过，任务 mean delta `+0.0052083`，但 near pass、AUC-time、ECE 没一起过。
- `B45a`：fast-reuse 后 A1+A2 通过，ECE 有改善，但 task mean delta `-0.0093316`，A3 仍失败。
- `B47`：manual backward/CE step 证明可做真 fused/manual 路径，但当前实现效率没有过 A1。

所以现在的状态可以概括为：

```text
完全满足 v12.4 base 要求：无
表达最强：B21/B33/B36/B45 的 Gated Legendre + Quadratic 系
任务信号最好但表达不足：B30/B31 LiteGatedHybrid
工程 kernel 最新进展：B47 manual backward/CE step 正确但效率失败
functional official：未打开
```

你刚才提的“不同基函数需要不同初始化 + 输入 norm 层”这个方向还没有成为最终成功项；从现有结果看很合理，尤其要针对 Legendre 分支、quadratic 分支分别设计 norm/scale，否则训练初期几何和分支梯度很容易互相拖累。


## 26. 最终分析结论

```text
1. v12.4 的工程 pipeline 已完成并继续推进：A1/A2/A3/function diagnostic/hash/provenance 均真实落盘。
2. 科学路线仍没有达成 base success：没有 official primitive 同时通过 expression + task gates。
3. B14 gated Legendre+quadratic 是明确进展：第一次把 official-capable primitive 推过 A2，并把 task accuracy 从大幅失败推进到接近/优于 MLP。
4. B14 的剩余 blocker 是 val-loss AUC/time，不是 fake data、threshold 或 functional gate；AUC ratio 仍约 1.7-2.8，远高于 1.05。
5. B15 logit-gain 与 LR=4e-3/WD=5e-4 全局 optimizer repair 已尝试，但没有修复 A3，且部分 efficiency/expression/task 指标变差。
6. B16/B17/B18/B19/B20 继续验证了 basis-specific branch norm 与 input geometry norm：B17 几何强但太伤效率/表达，B18 太弱，B19/B20 折中后仍打不开 A2。
7. B21 证明“不同 basis 需要不同 input norm”是有效机制：B21d 重新打开 A2，且 E1/E2/E6/E8 frozen coverage 全部超过 0.89。
8. B22 branch-scale repair 把 B22b 的 task mean delta 推到 -0.001736111111111111、worst delta 推到 -0.0234375，但 near pass/AUC-time/ECE 仍未过。
9. B23 temperature repair 在 B24 中改善 task accuracy/ECE mean，但没有解决 ECE max，也没有解决 AUC-time。
10. B24 compiled task timing repair 没有成功：AUC-time ratio 异常放大，说明 compiled training path/accounting 不能直接拿来开 base gate。
11. B25 steady AUC-time accounting 修复了明显的测量 artifact，但 B21a/B21d 仍因 near/AUC max/ECE 失败，没有 base success。
12. B26 plain low-temperature repair 没有过 A1/A2，说明继续只扫 low-temperature 初始化不足以推进。
13. B27 warmup-cosine LR schedule repair 真实推进到 A3：B27c 改善了 AUC-time mean 与 ECE mean，但 worst delta、near pass rate 与 AUC-time max 仍失败。
14. B28/B29 schedule-shape 与 h224 lightweight repair 没有通过 A1；TF32/high matmul precision 也没有修复效率门。
15. B30/B31 证明低成本 direct-readout + basis-specific input norm 能改善 task geometry，但表达深度不足，停在 R2 expression_fail。
16. B32-B36 重新把 small GatedHybrid 推到 A1+A2：B33c/B36c 的 mean task delta 均为 `0.005208333333333333`，worst delta 均为 `-0.01171875`，但 near pass/AUC-time 仍未过。
17. B36 direct skip 没有解决 AUC-time max；继续扫 hidden/temperature/direct-skip 小网格的收益已经下降。
18. B37 branchslow optimizer profile 已真实执行，但 task 明显变差；B38 leg/direct fast profile 已真实执行，但停在 R1 efficiency。
19. B36 compile-warm audit 证明 compiled task warmup artifact 可以缓解，但不能修成 A3 success：B36c AUC max 仍是 `1.6440778149943576`，near pass 仍是 `0.6666666666666666`。
20. B39a 单点 schedule repair 已真实执行，但 temp075/directskip/final050 先停在 R1，compiled step ratio = `2.2433983063496603`，不能进入 task。
21. B40 optimizer-step warmup restore 没修成 A1；B41 h168 light directskip 也仍停在 R1，说明单纯 warm optimizer 或减 hidden 不够。
22. B42 fast-reuse 是真实进展：等价 forward path max abs error = `0.0`，B42b 重新打开 A1+A2；但 task mean delta = `-0.009331597222222222`，near pass = `0.2222222222222222`，A3 仍失败。
23. B43 h172 speed/capacity 折中没有更好，B43b mean delta = `-0.011935763888888888`，worst = `-0.037109375`。
24. B44 residual mix15 input norm 说明更弱 block-whiten 混入不稳定，A1 即失败；这条 norm 设计不能继续包装为成功方向。
25. B45 final050 schedule 改善 ECE mean delta 到 `-0.0027160458266735077`，但 task mean delta 仍是 `-0.009331597222222222`，near pass = `0.3333333333333333`，AUC/ECE hard gate 仍失败。
26. B46 directskip050 是新的 task-geometry scale 尝试，但 B46a/B46b 先停在 R1，compiled step ratio = `1.9061898744500823` / `1.6355462392377917`，不能进入 task。
27. B47 custom autograd manual backward 梯度正确，但 `torch.compile` 包装失败；manual CE step 绕开 autograd graph 后仍停在 R1，B47b step ratio = `2.6246493524945507`。
28. Functional diagnostic 仍不能 official：base gate 没开；B32a 的 positive control gap 也只能写 diagnostic，B42/B43/B45 不 beat strong controls。
29. 下一步不是调低 gate；如果继续效率路线，需要 lower-level fused CUDA/Triton forward+backward kernel，或转向新的更简单 task-geometry primitive，而不是 Python-level backward 或 schedule/scale 小修。
```

最终一句话：v12.4 继续推进到 B47 后，pipeline 已完整闭环，并真实尝试了 input norm、basis-specific initialization、lite primitive、小 GatedHybrid、temperature/hidden、direct-skip、directskip scale、schedule、branch-wise optimizer、compile-warmup、optimizer-warmup-restore、fast-reuse architecture repair 与 manual backward/CE step；但科学结论仍是 base 未合格，functional 继续保持 diagnostic，不能打开 official route。
