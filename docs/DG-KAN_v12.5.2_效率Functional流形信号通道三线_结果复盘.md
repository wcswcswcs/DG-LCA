# DG-KAN v12.5.2 效率 / Functional / 流形信号通道三线结果复盘

> 本复盘记录 `DG-KAN_v12.5.2_效率Functional流形信号通道三线完整计划.md` 的真实执行与连续修复结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R3-ExpressionPassTaskFail
latest_targeted_route = R1-FusedEfficiencyFail
base_qualified = False
functional_open = False
best_current_candidate = B48a-SimpleFastTaskGeometry-h128-temp075, epochs8 task trajectory
best_current_artifact = results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_epochs8_20260521T023000Z
latest_targeted_artifact = results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_foreachadamw_20260520T112000Z
next_recommended_action = lower-level fused simple hinge/quadratic kernel or new low-cost task-geometry primitive; do not open functional official route
```

目标达成状态：

```text
A1/A3 efficiency = achieved by B48a/B48d/B48e compiled SimpleFastTaskGeometry in steady runs
A4 expression = achieved by B48a/B48d/B48e
A5 task/base = not achieved
Functional official = not opened
Line C = diagnostic only
no fake/proxy/cpu = pass in all listed artifacts
```

核心结论：

1. v12.5.2 不是完成态：B47 manual backward 仍太慢，B48 SimpleFastTaskGeometry 打开了效率与 expression，但 task AUC-time 仍是 hard blocker。
2. `B48a` 是本轮最大进展：compiled full-step `step_ratio_q90 = 0.8748996687637653`、`memory_ratio_q90 = 0.4033043955666884`，A4 expression pass。
3. `B48a` 5 epoch task 已接近但失败：mean delta `-0.00021701388888888888`、worst `-0.021484375`、ECE ok，但 near pass `0.5555555555555556`、AUC-time fail。
4. 训练轨迹修复 `B48a` epochs8 是当前最好 task 结果：mean delta `0.003472222222222222`、worst `-0.015625`、near `0.8888888888888888`、ECE ok；但 AUC-time 仍失败，所以不能写成 base success。
5. `B48d` temp050 与 `B48e` temp065 都过 A1/A4，但 near pass 比 B48a 更差，低温只能改善部分 AUC/ECE trade-off，不能修成 A5。
6. `B48b` h192/quad050 与 `B49a` sqdiag direct channel 都停在 R1 efficiency，不能合法进入 A4/A5。
7. `lr=3e-3`、epochs12、constant LR 都已尝试：`lr=3e-3` task 更差；epochs12 与 constant LR 在该次测量中未过 full-step official efficiency，因此不能用于 task 结论。
8. B50 后续修复继续失败：B50a temp100 过 A1/A4 但 A5 near/AUC 失败；B50b two-hinge h96 过 A1 但 A4 expression 失败；B50c/B50d two-hinge 加容量后回到 R1。
9. optimizer-step timing repair 已尝试：`fused_adamw` / `foreach_adamw` 都没修成 official efficiency，A4/A5 合法关闭。
10. Functional diagnostic 仍不能 official：base gate 没开，所有 functional/control rows 只能作为 diagnostic。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 v12.5.2 三线 runner | 独立输出 `v125_*` artifacts；复用 v12.4 strict PureKAN model，不降低 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Triton F2/F3 component kernels | 真实执行 residual input norm + Legendre basis / hidden preactivation component fusion；记录 correctness/timing，不冒充 full fused layer。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 修复 Triton `tl.tanh` blocker | 当前 Triton 3.0 无 `tl.tanh`，改为等价 `2/(1+exp(-2x))-1`；失败前结果不写成 kernel success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 F4 hybrid full-forward repair | 复用 F2/F3 component kernel 后接 torch readout；用于检查 component fusion 组合收益，不冒充 single-call fused kernel。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `SimpleFastTaskGeometryKAN` 与 B48a/B48b/B48c | 按 R1/R6 推荐方向，从 full GatedLQ 转向 hinge/direct edge basis + minimal quadratic sketch；fixed train-stream norm，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | B48 fast rewrite + manual CE step | 合并 direct hinge 三路 readout 为单 matmul，quadratic readout 改为扁平 matmul，并新增 manual CE backward；只减少算子/反向开销，不改 loss/gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 将 K3 SimpleFastTaskGeometry 接入 forward/full-step/Line C/Functional diagnostic | 用同一 A1/A3/Line C/Functional artifact 审计，不把 efficiency 或 diagnostic 自动写成 base success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 修复 full-step peak memory warmup accounting | formal B48 compiled step 暴露 compile/warmup allocation 污染 memory ratio；warmup 后 reset peak memory 再计 steady measurement。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 A4 expression qualification | 复用 v12.4 expression battery 的真实训练/冻结读出逻辑，按 v12.5.2 gate 计算 B1 key deltas、frozen R2 与 dead basis；未过 A4 不打开 A5。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 A5 task triage | A4 通过后才运行 MNIST/Fashion-MNIST/KMNIST × seeds 0/1/2；记录 mean/worst/near/ECE/AUC-time，不用单个 accuracy 替代 task gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 修复 route 判定使用 official full-step gate | A4 只有 `A3_efficiency_official_pass=1` 才允许打开；exploratory pass 不再被误判成 expression-ready。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48d temp050 | 保持 B48a h128/quad030，只把全局 logit gain 初始化为 `0.50`；用于修 AUC/near-pass，不改 loss、不做 post-hoc calibration。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48e temp065 | 在 temp075 与 temp050 之间验证 AUC/accuracy trade-off；仍是全局初始化，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B49a sqdiag direct channel | 在 direct edge basis 中加入 fixed-centered `z^2` diagonal quadratic channel，并把 sketch hidden 降到 h96 控制效率；用于验证更强 task geometry 是否可行。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B50a/B50b/B50c/B50d | B50a 测试高 logit gain；B50b/B50c/B50d 测试 fixed two-hinge direct channel 与 h96/h128/h112 表达-效率折中；均不按 dataset/label 分支。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 `--optimizer-impl` | 支持 `adamw` / `foreach_adamw` / `fused_adamw`，用于同协议检验 optimizer-step timing repair；MLP 与 K3 使用同一 optimizer impl，B47 manual 对照在 fused 不兼容时回退并落盘 `update_impl`。 |

本轮没有做：

```text
1. 没有调低 A1/A4/A5/Line C/functional gate。
2. 没有按 MNIST/Fashion-MNIST/KMNIST 名称分支。
3. 没有 teacher / distillation / modified loss / sampler / class weight。
4. 没有把 Triton component kernel 写成 full fused kernel success。
5. 没有把 diagnostic functional 写成 official success。
6. 没有把 exploratory efficiency pass 当成 official A4/A5 前置条件。
```

编译验证：

```bash
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1252_efficiency_functional_manifold.py
```

## 2. 执行链

| run | artifact | route | 结论 |
|---|---|---|---|
| F4 baseline | `v1252_efficiency_functional_manifold_f4_20260520T235800Z` | `R1-FusedEfficiencyFail` | B47 manual CE step ratio `2.0938924309965254`，F4 forward ratio `12.676035006742387`，full fused GatedLQ 未达标。 |
| B48a memfix | `v1252_efficiency_functional_manifold_b48a_memfix_20260521T004000Z` | `R2` | 首次看到 B48a compiled full-step official pass，但当时 A4/A5 还未接入。 |
| B48a A4/A5 | `v1252_efficiency_functional_manifold_b48a_steadya1_20260521T012000Z` | `R3` | A1/A4 通过，5 epoch A5 task 失败，near/AUC blocker。 |
| B48b task geometry | `v1252_efficiency_functional_manifold_b48b_taskgeom_20260521T013000Z` | `R1` | h192/quad050 太重，compiled step `1.9362694236037328`，不能进入 A4/A5。 |
| B48d temp050 | `v1252_efficiency_functional_manifold_b48d_temp050_20260521T015000Z` | `R3` | A1/A4 通过，但 near pass 降到 `0.3333333333333333`，AUC 仍失败。 |
| B48e temp065 | `v1252_efficiency_functional_manifold_b48e_temp065_20260521T020000Z` | `R3` | A1/A4 通过，near pass `0.4444444444444444`，不如 B48a。 |
| B49a sqdiag | `v1252_efficiency_functional_manifold_b49a_sqdiag_20260521T021000Z` | `R1` | diagonal quadratic direct channel 太慢，compiled step `1.8004856915469725`。 |
| B48a lr3e-3 | `v1252_efficiency_functional_manifold_b48a_lr3e3_20260521T022000Z` | `R3` | A1/A4 通过，但 task AUC 更差，worst `-0.029296875`。 |
| B48a epochs8 | `v1252_efficiency_functional_manifold_b48a_epochs8_20260521T023000Z` | `R3` | 当前最佳 task：mean/worst/near/ECE 均过，AUC-time 失败。 |
| B48a epochs12 | `v1252_efficiency_functional_manifold_b48a_epochs12_20260521T024000Z` | `R1` | compiled step `1.2750147965715033`，official efficiency 未过，不能作为 task 结论。 |
| B48a constant LR | `v1252_efficiency_functional_manifold_b48a_constantlr_20260521T025000Z` | `R1` | compiled step `1.3186464476668396`，official efficiency 未过。 |
| B50a temp100 | `v1252_efficiency_functional_manifold_b50a_temp100_20260520T101000Z` | `R3` | 高 logit gain 过 A1/A4，但 task near `0.6666666666666666`、AUC-time 失败。 |
| B50b twohinge h96 | `v1252_efficiency_functional_manifold_b50b_twohinge_20260520T102000Z` | `R2` | compiled step `0.742302331583089` 很快，但 E1 delta `-0.01132267713546753`，A4 expression 失败。 |
| B50c twohinge h128 | `v1252_efficiency_functional_manifold_b50c_twohinge_h128_20260520T103000Z` | `R1` | 增加 coverage 后 compiled step `1.7765248143263377`，A4/A5 合法关闭。 |
| B50d twohinge h112 | `v1252_efficiency_functional_manifold_b50d_twohinge_h112_20260520T104000Z` | `R1` | h112 中间点仍停在 R1，compiled step `1.8841033909191922`。 |
| B48a epochs10 | `v1252_efficiency_functional_manifold_b48a_epochs10_20260520T105000Z` | `R1` | 8/12 epoch 中间点未过 official efficiency，compiled step `1.7360781263110465`。 |
| B48a fused AdamW | `v1252_efficiency_functional_manifold_b48a_fusedadamw_fix_20260520T111000Z` | `R1` | fused optimizer 修复未过 official efficiency；compiled step `1.2935073872778775`，manual memory ratio `1.14477506199079`。 |
| B48a foreach AdamW | `v1252_efficiency_functional_manifold_b48a_foreachadamw_20260520T112000Z` | `R1` | foreach optimizer 更差，compiled step `2.214660506282393`。 |

最新完整主结论命令：

```bash
python experiments/run_v1252_efficiency_functional_manifold.py \
  --out-dir results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_epochs8_20260521T023000Z \
  --fresh --device auto --data-root data --no-download \
  --train-size 1024 --val-size 512 --test-size 512 \
  --batch-size 128 --kernel-warmup-steps 20 --kernel-measure-steps 60 \
  --coupling-batch-size 32 --functional-batch-size 32 --sketch-batch-size 8 --sketch-dim 8 \
  --expression-steps-b0 60 --expression-steps-b1 600 --expression-steps-b2 1200 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --epochs 8 --task-timing-warmup-epochs 4 --task-compile-warmup-steps 12 \
  --k3-candidate-id B48a-SimpleFastTaskGeometry-h128-temp075 \
  --report-path docs/DG-KAN_v12.5.2_效率Functional流形信号通道三线_结果复盘.md
```

## 3. A1/A3 Efficiency

| candidate / run | compiled step ratio | memory ratio | official pass | 判断 |
|---|---:|---:|---:|---|
| `B48a` 5ep | `0.8748996687637653` | `0.4033043955666884` | `1` | 通过 efficiency。 |
| `B48a` epochs8 | `1.232843215410771` | `0.40344768439108064` | `1` | 通过，但接近 1.25 step gate。 |
| `B48d` temp050 | `1.176040233647194` | `0.4033043955666884` | `1` | 通过。 |
| `B48e` temp065 | `1.1109911378645096` | `0.4033043955666884` | `1` | 通过。 |
| `B48b` h192/quad050 | `1.9362694236037328` | `0.43586856038322547` | `0` | R1，不能进入 A4。 |
| `B49a` sqdiag | `1.8004856915469725` | `0.39172871283707916` | `0` | R1，direct `z^2` channel 太贵。 |
| `B48a` epochs12 | `1.2750147965715033` | `0.40344768439108064` | `0` | official efficiency 未过，不能推进 A5。 |
| `B48a` constant LR | `1.3186464476668396` | `0.40344768439108064` | `0` | official efficiency 未过。 |

解释：B48a/B48d/B48e 的 compiled SimpleFastTaskGeometry 是真实 efficiency 进展；但 efficiency 不等于 base success。B48b/B49a 说明增强 task geometry 立刻撞到 R1。

## 4. A4 Expression

| candidate / run | E1 B1 delta | E2 B1 delta | E6 B1 delta | E8 B1 delta | frozen E1/E6/E8 | A4 pass |
|---|---:|---:|---:|---:|---|---:|
| `B48a` 5ep | `-0.008967041969299316` | `-0.0004900097846984863` | `0.0014042258262634277` | `0.0021126866340637207` | `0.9738487601280212 / 0.9880223870277405 / 0.9832057356834412` | `1` |
| `B48a` epochs8 | `-0.009019792079925537` | recorded pass | recorded pass | recorded pass | E6 frozen `0.9731121063232422` | `1` |
| `B48d` temp050 | `-0.010804891586303711` | `0.0005067586898803711` | `0.0036560893058776855` | `0.0017172694206237793` | `0.9707274436950684 / 0.9785473942756653 / 0.9950762987136841` | `1` |
| `B48e` temp065 | `-0.009357333183288574` | `-0.0011475086212158203` | `0.002190113067626953` | `0.0003685951232910156` | `0.9487748742103577 / 0.9581416249275208 / 0.9668599367141724` | `1` |

解释：B48 family 的 expression gate 主要由 frozen readout 和 key target B1 delta 支撑。B48d 的 E1 trainable delta 稍低于 `-0.01`，但 frozen key R2 全部超过 `0.89` 且 dead basis 为 `0.0`，因此 A4 仍合法通过。

## 5. A5 Task

| run | mean delta | worst delta | near pass | ECE ok | AUC-time ok | A5 pass |
|---|---:|---:|---:|---:|---:|---:|
| `B48a` 5ep | `-0.00021701388888888888` | `-0.021484375` | `0.5555555555555556` | `1` | `0` | `0` |
| `B48d` temp050 | `-0.001953125` | `-0.01953125` | `0.3333333333333333` | `1` | `0` | `0` |
| `B48e` temp065 | `-0.001736111111111111` | `-0.017578125` | `0.4444444444444444` | `1` | `0` | `0` |
| `B48a` lr3e-3 | `-0.0008680555555555555` | `-0.029296875` | `0.5555555555555556` | `1` | `0` | `0` |
| `B48a` epochs8 | `0.003472222222222222` | `-0.015625` | `0.8888888888888888` | `1` | `0` | `0` |

Best B48a epochs8 per-row task deltas:

| dataset | seed | val acc delta | AUC-time ratio | ECE delta |
|---|---:|---:|---:|---:|
| MNIST | 0 | `0.0` | `1.0495108892074518` | `0.001935124397277832` |
| MNIST | 1 | `0.009765625` | `1.1988236130206844` | `-0.004259243607521057` |
| MNIST | 2 | `-0.001953125` | `1.3858639395449737` | `-0.012461572885513306` |
| Fashion-MNIST | 0 | `0.001953125` | `1.3199949954799823` | `0.002098873257637024` |
| Fashion-MNIST | 1 | `0.017578125` | `0.861503545490614` | `0.0032938122749328613` |
| Fashion-MNIST | 2 | `-0.015625` | `1.0220184872732687` | `-0.008221253752708435` |
| KMNIST | 0 | `0.0` | `0.9563873792172763` | `0.005738511681556702` |
| KMNIST | 1 | `0.009765625` | `0.7083443264052757` | `-0.0073184967041015625` |
| KMNIST | 2 | `0.009765625` | `0.6498531456128085` | `-0.002658471465110779` |

解释：epochs8 已经修好 accuracy mean/worst/near 与 ECE，但 AUC-time max 仍远高于 `1.05`，尤其 MNIST seed 1/2 与 Fashion seed 0。不能因为 accuracy 达标就宣布 base。

## 6. Line C / Functional

Line C 与 Functional 继续只作为 diagnostic。Base 未过 A5，`functional_open = false`。

最新有效主线中没有 official functional success；B48 family 的 functional/control gap 未被写成 success。后续如果 A5 打开，才允许执行 official short-run。

## 7. No-Fake / Hash

No-fake summary：

| artifact | rows checked | fake/proxy/cpu |
|---|---:|---|
| `B48a 5ep` | `417` | `0 / 0 / 0` |
| `B48a epochs8` | `474` | `0 / 0 / 0` |
| `B48e temp065` | `419` | `0 / 0 / 0` |
| `B49a sqdiag` | `187` | `0 / 0 / 0` |
| `B48a constant LR` | `186` | `0 / 0 / 0` |

Selected SHA256 prefixes from audited artifacts:

| run | artifact | SHA256 prefix |
|---|---|---|
| `B48a 5ep` | `v125_full_step_efficiency.csv` | `0416a97dd046` |
| `B48a 5ep` | `v125_expression_summary.csv` | `45ef0d1c02f1` |
| `B48a 5ep` | `v125_task_failure_table.csv` | `2c3dad035c9e` |
| `B48a epochs8` | `v125_full_step_efficiency.csv` | `2469dc7cdb61` |
| `B48a epochs8` | `v125_expression_summary.csv` | `08b897b9bff9` |
| `B48a epochs8` | `v125_task_failure_table.csv` | `ea7bbaf37a62` |
| `B48d temp050` | `v125_full_step_efficiency.csv` | `ca2e949f5352` |
| `B48e temp065` | `v125_full_step_efficiency.csv` | `99dc96d07d82` |
| `B49a sqdiag` | `v125_full_step_efficiency.csv` | `386e2a1a59aa` |

## 8. B50 / Optimizer-Step 后续修复

> 本节继续执行最新推荐方向：不是降低 gate，而是分别尝试更高 logit gain、低成本 two-hinge task geometry、B48a 训练轨迹中间点，以及 optimizer-step timing repair。所有结果都来自新增 artifact；A1/A4 未开时 A5 合法关闭。

新增 artifacts：

```text
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b50a_temp100_20260520T101000Z
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b50b_twohinge_20260520T102000Z
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b50c_twohinge_h128_20260520T103000Z
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b50d_twohinge_h112_20260520T104000Z
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_epochs10_20260520T105000Z
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_fusedadamw_fix_20260520T111000Z
results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_b48a_foreachadamw_20260520T112000Z
```

Route 摘要：

| run | route | 关键结论 |
|---|---|---|
| `B50a-temp100` | `R3-ExpressionPassTaskFail` | 高 logit gain 过 A1/A4，但 A5 仍因 near/AUC 失败。 |
| `B50b-twohinge-h96` | `R2-EfficiencyPassExpressionFail` | compiled step 很快，但 expression 差一点，task 合法关闭。 |
| `B50c-twohinge-h128` | `R1-FusedEfficiencyFail` | 补容量后 compiled path 太慢。 |
| `B50d-twohinge-h112` | `R1-FusedEfficiencyFail` | 中间点仍太慢。 |
| `B48a-epochs10` | `R1-FusedEfficiencyFail` | 8/12 epoch 中间点未过 official full-step gate。 |
| `B48a-fused-adamw` | `R1-FusedEfficiencyFail` | 初次 fused AdamW 在 B47 manual 对照上因 dtype/device/layout 不兼容崩溃；修复为 B47 fallback 后，B48a 仍未过 official efficiency。 |
| `B48a-foreach-adamw` | `R1-FusedEfficiencyFail` | foreach optimizer 没有修复 step ratio。 |

### 8.1 B50 Task Geometry

| candidate | compiled step ratio | memory ratio | A4 pass | task mean delta | near pass | AUC ok | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `B50a-temp100` | `1.1263100225094536` | `0.40344768439108064` | `1` | `0.0` | `0.6666666666666666` | `0` | 高温改善 worst，但 near/AUC 未过。 |
| `B50b-twohinge-h96` | `0.742302331583089` | `0.3969276340715737` | `0` | not_run | not_run | not_run | two-hinge 很快但表达不足。 |
| `B50c-twohinge-h128` | `1.7765248143263377` | `0.41348923622355926` | `0` | not_run | not_run | not_run | 补容量后 R1。 |
| `B50d-twohinge-h112` | `1.8841033909191922` | `0.40530075896451884` | `0` | not_run | not_run | not_run | 中间点仍 R1。 |

B50a expression：

| target | B1 delta vs MLP | frozen R2 |
|---|---:|---:|
| `E1-pairwise-product` | `-0.007124185562133789` | `0.9710924625396729` |
| `E2-composition` | `-0.0017309784889221191` | `-1.1302626132965088` |
| `E6-rotated-pairwise-product` | `0.0010032057762145996` | `0.9446777105331421` |
| `E8-random-quadratic-form` | `-0.000929415225982666` | `0.8675657510757446` |

B50b expression failure：

```text
E1 B1 delta = -0.01132267713546753
E6 B1 delta = 0.001184225082397461
E8 B1 delta = -0.001538991928100586
A4_expression_pass = 0
```

解释：B50b 证明 two-hinge direct channel 有真实效率余量，但 h96/quad020 表达略薄；B50c/B50d 证明直接加 hidden 容量会触发 compiled/full-step overhead，不能合法进入 task。B50a 证明高 logit gain 不是 AUC/NLL 解，Fashion/KMNIST 部分行的 AUC 仍高。

### 8.2 Optimizer-Step Timing Repair

新增 `--optimizer-impl` 后，`MLP-same-param-AdamW` 与 K3 candidate 使用同一个 optimizer implementation；B47 manual 对照在 `fused_adamw` 不兼容时回退到普通 AdamW，并在 `update_impl` 中落盘，不参与 base gate。

| run | K3 update impl | manual step ratio / memory | compiled step ratio / memory | route |
|---|---|---:|---:|---|
| `B48a-fused-adamw` | `fused_adamw` | `1.2458835579570569` / `1.14477506199079` | `1.2935073872778775` / `0.39969890187743534` | `R1` |
| `B48a-foreach-adamw` | `foreach_adamw` | `1.3211055463753936` / `1.108542024013722` | `2.214660506282393` / `0.40344768439108064` | `R1` |

解释：fused AdamW 把 manual step ratio 推到非常接近 `1.25`，但 memory ratio 超 `1.05`；compiled path 仍超 step gate。foreach AdamW 更差。因此 optimizer-step timing repair 不能打开 A4/A5。

### 8.3 No-Fake / Hash

| run | rows checked | fake/proxy/cpu | full-step hash prefix | route hash prefix |
|---|---:|---|---|---|
| `B50a-temp100` | `476` | `0 / 0 / 0` | `0a3cb2ce7aba` | `bd76e4c20baf` |
| `B50b-twohinge-h96` | `316` | `0 / 0 / 0` | `67a41a29af75` | `3a37654ff559` |
| `B50c-twohinge-h128` | `190` | `0 / 0 / 0` | `8f54aeca0432` | `5c7d46e49202` |
| `B50d-twohinge-h112` | `191` | `0 / 0 / 0` | `e97de3b25761` | `37cfaf680925` |
| `B48a-epochs10` | `191` | `0 / 0 / 0` | `7e99b8e972a8` | `37cfaf680925` |
| `B48a-fused-adamw` | `190` | `0 / 0 / 0` | `fad96b1fa38d` | `638bf21b6d02` |
| `B48a-foreach-adamw` | `191` | `0 / 0 / 0` | `c92c02e826ea` | `37cfaf680925` |

追加判断：B50 和 optimizer repair 都没有改变 v12.5.2 的科学状态。当前 best 仍是 `B48a epochs8`：A1/A4 通过，A5 accuracy/ECE 近门但 AUC-time 失败。继续推进时，不应再做 temperature/epoch/optimizer 小修；需要真正 fused simple hinge/quadratic forward+backward kernel，或重新设计更低成本且不伤 expression 的 task-geometry primitive。

## 9. 最终分析结论

```text
1. v12.5.2 已按计划继续推进，不是停在 v12.4 B47；新增 B48 SimpleFastTaskGeometry 后，效率与 expression gate 被真实打开。
2. B48a 是当前主线：A1/A3 official efficiency pass，A4 expression pass；但 A5 task 未过。
3. B48a epochs8 是当前最接近 base 的 artifact：accuracy 与 ECE 子门已达标，只剩 AUC-time hard gate。
4. 低温 B48d/B48e 不能修好 task：AUC/accuracy trade-off 变差，near pass 下降。
5. 增强 quadratic branch 的 B48b 与 diagonal quadratic direct channel B49a 都失败在 efficiency，说明直接增加 task geometry 会迅速撞到 R1。
6. lr=3e-3 使 AUC 与 worst-row 变差；epochs12 与 constant LR 在该次测量中未过 official efficiency，不能作为 task success 路线。
7. B50a 高 logit gain 不是解：虽然过 A1/A4，mean delta 到 `0.0`、worst 到 `-0.009765625`，但 near `0.6666666666666666` 且 AUC-time 失败。
8. B50b/B50c/B50d two-hinge 支线被排除：h96 过效率但 A4 expression 失败；h112/h128 补容量后都回到 R1。
9. B48a epochs10、fused AdamW、foreach AdamW 都没有打开 A4/A5：这些 targeted repair 均停在 R1 official efficiency。
10. Functional official 仍关闭；任何 Line C/functional positive 都不能越过 base gate。
11. 下一步不应调低 gate，也不应继续 temperature/epoch/optimizer 小修。更合理方向是：
   a. lower-level fused simple hinge/quadratic forward+backward kernel，专门修 official full-step 与 task AUC-time；
   b. 新的低成本 task-geometry primitive，必须先证明不伤 A1/A4；
   c. 若继续 B48a，先做 kernel-level timing，而不是继续靠 PyTorch compile/optimizer wrapper。
```
