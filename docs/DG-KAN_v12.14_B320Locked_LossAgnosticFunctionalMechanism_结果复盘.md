# DG-KAN v12.14 B320Locked LossAgnosticFunctionalMechanism 结果复盘

生成时间：`2026-05-23T23:28:41Z`

## 1. 结论

```text
route = R3-FunctionalMechanismPartialOnly
P4_open = 0
bm2_full_3x3_pass_candidate_count = 0
bm2_partial_pass_rows = 0
```

本轮结果只统计 loss-agnostic official 候选。CE-specific projector 仍只作为 v12.13 诊断，不进入 v12.14 official 聚合。

## 2. 关键数据

| method | batch | window | rows | CouplingR2_delta_mean | NoiseSignalLeak_delta_mean | RealSignalReservoirRatio_delta_mean | pass_rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BM2aa-ClassMeanFreeCotangentK16` | `32` | `3` | `27` | `0.07183199248345783` | `0.0002452238566345639` | `-0.0002231349547704061` | `0` |
| `BM2aa-ClassMeanFreeCotangentK16` | `32` | `5` | `27` | `0.11768511756435046` | `-7.207389851963078e-05` | `-0.0002396366110554448` | `0` |
| `BM2aa-ClassMeanFreeCotangentK16` | `32` | `10` | `27` | `0.19346410763428382` | `0.0002910693292506039` | `0.0006129824452930026` | `0` |
| `BM2ac-LogitWhitenedCotangentK16` | `32` | `3` | `27` | `0.059976459125139166` | `0.00011259219091799523` | `-0.00020228878215507225` | `0` |
| `BM2ac-LogitWhitenedCotangentK16` | `32` | `5` | `27` | `0.09815373304859633` | `0.00010959114189501162` | `-0.000905786951382955` | `0` |
| `BM2ac-LogitWhitenedCotangentK16` | `32` | `10` | `27` | `0.16798029424264335` | `0.00017505237957704122` | `-0.0005740109417173597` | `0` |
| `BM2ae-OrthogonalRademacherCotangentK32` | `32` | `3` | `27` | `0.06325259257677708` | `-0.0001707945915835875` | `8.852890244236699e-05` | `0` |
| `BM2ae-OrthogonalRademacherCotangentK32` | `32` | `5` | `27` | `0.1067869924872215` | `0.000652287017416071` | `-0.00023884188245843957` | `0` |
| `BM2ae-OrthogonalRademacherCotangentK32` | `32` | `10` | `27` | `0.18179366386081253` | `-0.00010091172535558817` | `0.0012619059394907068` | `0` |
| `BM2ba-DirectRoleCotangentSpectral` | `32` | `3` | `27` | `0.0024276686789616443` | `2.184841367933485e-05` | `9.108472753454137e-06` | `0` |
| `BM2ba-DirectRoleCotangentSpectral` | `32` | `5` | `27` | `0.006200344719700102` | `-5.707067127029101e-05` | `1.210526183799461e-05` | `0` |
| `BM2ba-DirectRoleCotangentSpectral` | `32` | `10` | `27` | `0.01906259440058667` | `-9.069960624738424e-05` | `-7.150846498983878e-05` | `0` |
| `BM2bb-QuadRoleCotangentSpectral` | `32` | `3` | `27` | `0.06434488145420306` | `0.00026695839025908045` | `-0.00013367500570085313` | `0` |
| `BM2bb-QuadRoleCotangentSpectral` | `32` | `5` | `27` | `0.10061957120496677` | `1.3729712615410486e-05` | `-0.0007546268127582691` | `0` |
| `BM2bb-QuadRoleCotangentSpectral` | `32` | `10` | `27` | `0.16905705096127424` | `-0.00028491647659662975` | `1.3413252653899016e-05` | `0` |
| `BM2bc-BranchRoleCotangentSpectral` | `32` | `3` | `27` | `0.02837852440365187` | `-2.7762898416430864e-06` | `-7.929790903020787e-05` | `0` |
| `BM2bc-BranchRoleCotangentSpectral` | `32` | `5` | `27` | `0.05656211829458682` | `1.912445036901368e-05` | `-2.2250193136709708e-05` | `0` |
| `BM2bc-BranchRoleCotangentSpectral` | `32` | `10` | `27` | `0.11347222088674869` | `1.8123806781928849e-06` | `0.0001459237602021959` | `0` |
| `BM2bd-DirectQuadRoleMixed` | `32` | `3` | `27` | `0.06722172685976861` | `2.0153613554106818e-05` | `-7.529446372279415e-05` | `0` |
| `BM2bd-DirectQuadRoleMixed` | `32` | `5` | `27` | `0.11039199886665216` | `0.00036064302548766136` | `0.0002026414429699933` | `0` |
| `BM2bd-DirectQuadRoleMixed` | `32` | `10` | `27` | `0.18028557092235642` | `-0.00024878853798361014` | `-0.00043075945642259385` | `0` |
| `BM2bg-RoleAdaptiveCotangentNoLabelOrth` | `32` | `3` | `27` | `0.06674420198513167` | `-0.00025743101206090715` | `-2.4488678684941044e-05` | `0` |
| `BM2bg-RoleAdaptiveCotangentNoLabelOrth` | `32` | `5` | `27` | `0.10925419835743605` | `0.0004423893298263903` | `5.941589673360189e-05` | `0` |
| `BM2bg-RoleAdaptiveCotangentNoLabelOrth` | `32` | `10` | `27` | `0.18052077774494998` | `-0.0007516772376321671` | `-0.0007385291435100414` | `0` |
| `BM2ca-DirectPrimitiveSketchWhiten` | `32` | `3` | `27` | `0.002059384639134313` | `1.9988941925543325e-07` | `4.2861258542096174e-05` | `0` |
| `BM2ca-DirectPrimitiveSketchWhiten` | `32` | `5` | `27` | `0.005361372845455813` | `-6.467778304660762e-05` | `-9.842934431853118e-05` | `0` |
| `BM2ca-DirectPrimitiveSketchWhiten` | `32` | `10` | `27` | `0.017146910628983263` | `-8.29842077412953e-05` | `-4.642760312115705e-05` | `0` |
| `BM2cb-QuadPrimitiveSketchWhiten` | `32` | `3` | `27` | `0.06346854087679282` | `-9.242008888611087e-05` | `-0.00022852475996370668` | `0` |
| `BM2cb-QuadPrimitiveSketchWhiten` | `32` | `5` | `27` | `0.10368206381984714` | `-0.00014565988547272154` | `0.0005100926867237797` | `0` |
| `BM2cb-QuadPrimitiveSketchWhiten` | `32` | `10` | `27` | `0.17245544228954882` | `0.0005549597877284719` | `-0.0006644532636359885` | `0` |
| `BM2cc-BranchPrimitiveSketchWhiten` | `32` | `3` | `27` | `0.034604168316836575` | `-3.691088339244878e-05` | `-5.180637041727702e-06` | `0` |
| `BM2cc-BranchPrimitiveSketchWhiten` | `32` | `5` | `27` | `0.06717820018960259` | `-5.0713791063538304e-05` | `1.0625631720931441e-05` | `0` |
| `BM2cc-BranchPrimitiveSketchWhiten` | `32` | `10` | `27` | `0.1293447347146301` | `-2.7497314106397055e-05` | `0.00026314622826046415` | `0` |
| `BM2cd-DirectQuadSketchIsotropy` | `32` | `3` | `27` | `0.06392573407081094` | `0.00010678915444899488` | `-5.7985385258992515e-05` | `0` |
| `BM2cd-DirectQuadSketchIsotropy` | `32` | `5` | `27` | `0.10428225021468522` | `-0.00014536848498715294` | `-0.00041332454593093307` | `0` |
| `BM2cd-DirectQuadSketchIsotropy` | `32` | `10` | `27` | `0.17814336883038936` | `0.0002487281841846804` | `0.0009686946868896484` | `0` |
| `BM2ce-BranchQuadSketchIsotropy` | `32` | `3` | `27` | `0.0697984480880104` | `8.03345055491836e-05` | `-0.00028827786445617676` | `0` |
| `BM2ce-BranchQuadSketchIsotropy` | `32` | `5` | `27` | `0.1117999690864696` | `6.435628704450748e-06` | `-1.0044486434371383e-07` | `0` |
| `BM2ce-BranchQuadSketchIsotropy` | `32` | `10` | `27` | `0.17808705326581495` | `-0.00049499147334481` | `0.0007694683693073414` | `0` |
| `BM2cf-SignalReservoirSketchSpread` | `32` | `3` | `27` | `0.07780339258273238` | `1.182157063373813e-05` | `1.892281903160943e-05` | `0` |
| `BM2cf-SignalReservoirSketchSpread` | `32` | `5` | `27` | `0.12358835865182646` | `0.0004172398432813309` | `0.000295366402025576` | `0` |
| `BM2cf-SignalReservoirSketchSpread` | `32` | `10` | `27` | `0.20064144521565677` | `0.00017824647322952473` | `0.001347697995327137` | `0` |
| `BM2cg-NoiseNullSketchSpread` | `32` | `3` | `27` | `0.06286554657920097` | `-5.938788806950605e-05` | `-7.167348155268917e-05` | `0` |
| `BM2cg-NoiseNullSketchSpread` | `32` | `5` | `27` | `0.10654129685524766` | `-0.00013718315986571488` | `-0.00048334730996025936` | `0` |
| `BM2cg-NoiseNullSketchSpread` | `32` | `10` | `27` | `0.18235398771955053` | `-0.00013527866664204608` | `-0.000606324937608507` | `0` |
| `BM4a-BranchMomentTransport` | `32` | `3` | `27` | `0.0` | `0.0` | `0.0` | `0` |
| `BM4a-BranchMomentTransport` | `32` | `5` | `27` | `0.0` | `0.0` | `0.0` | `0` |
| `BM4a-BranchMomentTransport` | `32` | `10` | `27` | `0.0` | `0.0` | `0.0` | `0` |
| `BM4b-QuadMomentTransport` | `32` | `3` | `27` | `0.0015666729475797487` | `-5.328889798235011e-07` | `4.472004042731391e-06` | `0` |
| `BM4b-QuadMomentTransport` | `32` | `5` | `27` | `0.0042754940322199475` | `2.4811985592047375e-05` | `-3.251212614553946e-06` | `0` |
| `BM4b-QuadMomentTransport` | `32` | `10` | `27` | `0.01589629212501881` | `-2.7321224075017704e-05` | `-1.3057832364682798e-05` | `0` |
| `BM4c-DirectQuadCovTransport` | `32` | `3` | `27` | `0.0013043130290019708` | `3.2279812903315934e-06` | `1.7658979804427536e-05` | `0` |
| `BM4c-DirectQuadCovTransport` | `32` | `5` | `27` | `0.0034919636810865034` | `1.1112885894598784e-05` | `3.065224047060366e-05` | `0` |
| `BM4c-DirectQuadCovTransport` | `32` | `10` | `27` | `0.012185332715411442` | `-2.1179342487206063e-06` | `6.596578492058648e-05` | `0` |
| `BM4d-RoleCovarianceEqualization` | `32` | `3` | `27` | `0.0015666729475797487` | `-5.328889798235011e-07` | `4.472004042731391e-06` | `0` |
| `BM4d-RoleCovarianceEqualization` | `32` | `5` | `27` | `0.0042754940322199475` | `2.4811985592047375e-05` | `-3.251212614553946e-06` | `0` |
| `BM4d-RoleCovarianceEqualization` | `32` | `10` | `27` | `0.01589629212501881` | `-2.7321224075017704e-05` | `-1.3057832364682798e-05` | `0` |
| `BM4e-LogitJacobianMomentTransport` | `32` | `3` | `27` | `0.001563079687214668` | `-1.2578028771612378e-07` | `7.2022279103597e-06` | `0` |
| `BM4e-LogitJacobianMomentTransport` | `32` | `5` | `27` | `0.0042664926091875366` | `2.475929687972422e-05` | `-3.863264013219763e-08` | `0` |
| `BM4e-LogitJacobianMomentTransport` | `32` | `10` | `27` | `0.015863558931509897` | `-2.269693154462234e-05` | `-2.3971001307169597e-05` | `0` |
| `BM4f-TailSafeMomentTransport` | `32` | `3` | `27` | `0.003258137588326497` | `-2.2924850108446897e-06` | `2.8212313298825865e-05` | `0` |
| `BM4f-TailSafeMomentTransport` | `32` | `5` | `27` | `0.008672868964622082` | `3.83307708910218e-05` | `2.5317624763206198e-05` | `0` |
| `BM4f-TailSafeMomentTransport` | `32` | `10` | `27` | `0.029613903317866088` | `-1.1737187850047593e-05` | `2.9252635108100043e-05` | `0` |
| `NoOp` | `32` | `3` | `54` | `0.0` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `5` | `54` | `0.0` | `0.0` | `0.0` | `0` |
| `NoOp` | `32` | `10` | `54` | `0.0` | `0.0` | `0.0` | `0` |
| `RandomMatchedNorm` | `32` | `3` | `54` | `0.00022342914843694102` | `1.446999333522938e-07` | `-7.597384629426179e-06` | `0` |
| `RandomMatchedNorm` | `32` | `5` | `54` | `0.0006195782387597334` | `4.288412768531728e-06` | `-8.351824901722096e-06` | `0` |
| `RandomMatchedNorm` | `32` | `10` | `54` | `0.0024553669653704173` | `8.318701508903393e-06` | `-4.106097751193576e-06` | `0` |
| `TaskOnlyAdamW` | `32` | `3` | `54` | `0.0938851333687909` | `3.7144748838963334e-05` | `-0.0008477447209534821` | `0` |
| `TaskOnlyAdamW` | `32` | `5` | `54` | `0.1440669945754887` | `-9.028955052296321e-05` | `-0.0006972407853161847` | `0` |
| `TaskOnlyAdamW` | `32` | `10` | `54` | `0.2123571976013605` | `-0.0011593224637046525` | `0.0013591382238599989` | `0` |

## 3. 当前 blocker

```text
B1/B2/B3/B4 all failed P3; build a stronger loss-agnostic signal-channel estimator or deeper primitive instrumentation before P4
```

若没有 P3 survivor，P4 按计划保持关闭。

## 4. 修改与审计说明

本轮新增 `experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py`，用于独立生成 v12.14 artifact contract。实现内容包括：B1 multi-cotangent spectral ensemble、B2 role-conditioned cotangent shaping、B3 primitive/role sketch low-eigen or isotropy shaping、B4 loss-agnostic role moment transport、loss-agnostic audit、P3 controls、P3 failure table、P4 gate、hash manifest 和两个复盘日志。

所有 official functional candidate 的方向生成字段均为：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
```

`CEProjectorDiagnostic` 只写入 `v1214_loss_agnostic_audit.csv` 作为反例审计行，`official_eligible=0`，没有进入 P3 official 聚合。

## 5. 结果解释

Line C calibration 仍可信：`NoOp` 和 `RandomMatchedNorm` 的 false positive 都是 0。失败不是 metric 误报，而是当前 loss-agnostic 方向不能把 `NoiseSignalLeak_delta` 和 `RealSignalReservoirRatio_delta` 稳定推到 `-0.01` 以下。

主失败计数为：

```text
NoiseSignalLeak_delta>-0.01 = 2186
RealSignalReservoirRatio_delta>-0.01 = 2178
control_gap<0.005 = 1940
CouplingR2_delta<0.02 = 926
```

B1/B2/B3 可以稳定打开 CouplingR2，但 noise/reservoir 改变量通常只有 `1e-4` 到 `1e-3` 量级，远低于 P3 hard gate。B4 moment transport 更接近 no-op，说明当前 role parameter moment transport 不是可用 value source。

因此 v12.14 的正式结论是：没有 loss-agnostic P3 survivor，不能打开 P4；当前 blocker 已从“cotangent ensemble 太弱”推进为“B1/B2/B3/B4 当前实现都无法提供足够强的 loss-agnostic signal-channel estimator / primitive instrumentation”。
