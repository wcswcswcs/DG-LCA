# DG-KAN v9.9.2 Natural Generator Distribution Repair / FuturePathOperator 实验复盘

> 本复盘记录 `DG-KAN_v9.9.2_自然生成器分布修复_FuturePathOperator_四线并行完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 和真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。修复候选未通过分布 gate 时，后续 full density panel 显式 `not_run`。

## 0. 最新结论

```text
route = R2-DistributionFidelityPassDensityPending
primary_blocker = natural_density_full_panel_pending
secondary_blocker = future_operator_sketch_not_opened
system_legal_controller_pass = 0
generated_route_status = stopped_density_full_panel_pending
```

最终 artifact：`results/real_rerun_20260506/v9920_natural_generator_distribution_repair_futurepathoperator_full_20260517T030000Z`

核心结论：

1. P0 复现 v9.9.1 boundary：source route = `R-P1-NaturalGeneratorDistributionMismatch`，PSI max = `13.018683555518681`，missing major groups = `17`。
2. P1 定位原 generator 分布错误：weak/strong = `0` / `0`，failure class = `F1-cursor-collapse`。
3. P2 候选矩阵最佳候选 = `G5-hybrid-quota-random-generator`，action count = `1024`，weak/strong = `1` / `1`。
4. 最佳候选 PSI/JS/max-share/entropy-min = `0.016435209021362127` / `0.045276968538283786` / `0.2626953125` / `0.9355692755194838`。
5. 最佳候选 CoreLike count/LCB/UCB = `4` / `0.0015200565834699157` / `0.01000078476017681`；PathGood count/LCB/UCB = `3` / `0.000996829459082154` / `0.008578186769099308`。
6. P3 density sufficient/insufficient/inconclusive = `0` / `0` / `1`；5000/10000/20000 reason = `full_density_panels_deferred_set_run_full_density_panels_to_open`。
7. P5 FuturePathOperator sketch weak/strong = `0` / `0`。
8. P6 controller = `not_run`；P7 generated sandbox allowed = `0`。
9. No-fake audit：rows checked = `3804`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.2 generator repair profiles；G0 保留原 v9.9.1 行为，G1-G5 生成新 payload 并记录 quota/cursor provenance。 |
| `experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py` | v9.9.2 runner；执行 P0/P1/P2 repair matrix/P3 boundary/P5-P8 gate。 |

```text
python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py
```

```bash
python experiments/run_v9920_natural_generator_distribution_repair_futurepathoperator.py --out-dir results/real_rerun_20260506/v9920_natural_generator_distribution_repair_futurepathoperator_full_20260517T030000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9920",
  "status": "summary",
  "route": "R2-DistributionFidelityPassDensityPending",
  "primary_blocker": "natural_density_full_panel_pending",
  "secondary_blocker": "future_operator_sketch_not_opened",
  "source_route_v9910": "R-P1-NaturalGeneratorDistributionMismatch",
  "P0_boundary_pass": 1,
  "P1_distribution_weak_pass": 0,
  "P1_distribution_strong_pass": 0,
  "P1_PSI_major_max": 25.56018339816044,
  "P1_missing_major_group_count": 41,
  "best_generator_id": "G5-hybrid-quota-random-generator",
  "best_generator_distribution_weak_pass": 1,
  "best_generator_distribution_strong_pass": 1,
  "best_generator_enters_P3": 1,
  "best_PSI_major_max": 0.016435209021362127,
  "best_JS_major_max": 0.045276968538283786,
  "best_max_group_share": 0.2626953125,
  "best_entropy_ratio_min": 0.9355692755194838,
  "P3_density_sufficient": 0,
  "P3_density_insufficient": 0,
  "P3_density_inconclusive": 1,
  "P5_future_operator_sketch_weak_pass": 0,
  "P5_future_operator_sketch_strong_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_density_full_panel_pending",
  "P8_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Distribution Localization

```text
generator = G0-current-v9910-generator
axis_count = 18
PSI_major_max = 25.56018339816044
JS_major_max = 0.8325546111576977
max_group_share_pilot = 1.0
entropy_ratio_min = -0.0
missing_major_group_count = 41
failure_class = F1-cursor-collapse
```

## 4. P2 Candidate Repair

```text
best_generator_id = G5-hybrid-quota-random-generator
best_action_count = 1024
best_branch_rows = 30720 / 30720
best_distribution_weak_pass = 1
best_candidate_enters_P3 = 1
```

判断：P2 没有复用旧 payload；候选 action_id/payload_hash 都来自新 materializer。未通过 256-action gate 的候选没有被硬推到 1024/full panel。

## 5. P3 Density Boundary

```text
selected_generator = G5-hybrid-quota-random-generator
panel_size = 1024
CoreLike count/LCB/UCB = 4 / 0.0015200565834699157 / 0.01000078476017681
PathGood count/LCB/UCB = 3 / 0.000996829459082154 / 0.008578186769099308
density sufficient/insufficient/inconclusive = 0 / 0 / 1
```

## 6. No-Fake / Contract / Failure

```text
rows_checked = 3804
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9920.csv` | `9aa65a11772ce4df31f3749da936c96029bdd4761150985ee4395885791aaddc` |
| `failure_taxonomy_v9920.csv` | `2479a7d35f6e2a5e9e77f63b755caf9025d8b5e0993757373c283febc218b15d` |
| `fig_p1_distribution_error_axes_v9920.svg` | `eed4a780d3a084f7615cb23abb1f1bf24055349a508222bfce113d2afca37e5d` |
| `fig_p2_candidate_gate_matrix_v9920.svg` | `091e5a8f8dcb28f19cbd1e0e8056bb80ec327df713afff558f578dc2c780c1c0` |
| `fig_p3_density_decision_v9920.svg` | `788a64ead4ba5f31556264f8ff5f43bba851224090b86dd3b3fc9c2d7b537b15` |
| `materializer` | `ee7dd19c0a54caf333c5dcb408c8b84ba0d66a0233eb431b59c1583d673e42d7` |
| `no_fake_audit_v9920.csv` | `9dc4b6919c424c5c6ece935de9c080500c94744c088c61b938e7329ac50e8298` |
| `p0_v9910_boundary_reproduction.csv` | `2667fecfad10d97e00e3f224b878543816279fb28687aea46f756c78d00c2902` |
| `p1_distribution_fidelity_axes.csv` | `04a92dfbfb0e6dc56de13c22ad89cb1f5d1bf872108fe9ba49bbfed9a6923838` |
| `p1_generator_cursor_trace.csv` | `de3b9c4ea8515f8783332777eba96793c6ba97ec1ba49229ef32b12f4f09ff9c` |
| `p1_missing_major_groups.csv` | `cffe585a80887da1f1b2932198abaa0ccb18cba2a464535af2d53f663df71ea5` |
| `p2_G1-round-robin-source-cursor_16_action_smoke.csv` | `5f26a353f101210863210002cbcf9d8375684ad6b87d5b56d6a886f1d5270cf1` |
| `p2_G1-round-robin-source-cursor_16_distribution_axes.csv` | `a1dd18b642994017139ff93ade156c0cd4fcb7c1606cd49da0b2842bfc45cd4b` |
| `p2_G1-round-robin-source-cursor_1_action_smoke.csv` | `0120fefa51e69d496cece8e8e0399d975947b58f2518294af38e1f794dcb3031` |
| `p2_G1-round-robin-source-cursor_1_distribution_axes.csv` | `bb48ff296bb408de9a92e740c45acb4645ce4cc7e80d36804f93315358167d96` |
| `p2_G1-round-robin-source-cursor_256_action_smoke.csv` | `fb02a67d3ae30b12e8ba908e10ccb9b4ad26e8afe745c287a0bf179e23d39689` |
| `p2_G1-round-robin-source-cursor_256_distribution_axes.csv` | `7a590b3b8e61c628865ef8f4533153514e7dda41844897dc1f49349cb112e410` |
| `p2_G2-stratified-reference-quota-generator_1024_action_fidelity_pilot.csv` | `53196b9b4b017a415a889dd88a24dd15c475547ece764eff246ee67d8d7dfb31` |
| `p2_G2-stratified-reference-quota-generator_1024_action_labels.csv` | `fb3a46db77f32b8d164c394d0cecb30c47abf3bf9d1e6ba0ed548d3872994125` |
| `p2_G2-stratified-reference-quota-generator_1024_distribution_axes.csv` | `8b3a1836a30d9bddd082a4957a4c2f6a164b0564469e1030e69498cdef3ea156` |
| `p2_G2-stratified-reference-quota-generator_16_action_smoke.csv` | `08aed155b3c11f6fef41975b177bd77ea1c7b379c25661a21a02afd9c79ae569` |
| `p2_G2-stratified-reference-quota-generator_16_distribution_axes.csv` | `b96bfb7f3e7ea6a0ee2eb4e0390ea868937f72449852e7a4f0cc70079d5a17d7` |
| `p2_G2-stratified-reference-quota-generator_1_action_smoke.csv` | `e00165722fde0b067c47268824181f2a5024e48875b514565876aa8935a49882` |
| `p2_G2-stratified-reference-quota-generator_1_distribution_axes.csv` | `7a244c11b69dac18eaf385efd7bb007a71d9c6d8a6fd7480da1dd20e5f92c599` |
| `p2_G2-stratified-reference-quota-generator_256_action_smoke.csv` | `8ec91babb381ae5d7a878f9b1c713cbc51ce182064ed163d1c1e9e5d1f7ce64c` |
| `p2_G2-stratified-reference-quota-generator_256_distribution_axes.csv` | `67105032a92b7c96e066d7a92c5369c4d627778cdee41712be79d79fdb4a1484` |
| `p2_G3-recipe-complete-generator_16_action_smoke.csv` | `42d85e7e01af01703fd2122691ea3062d399e3e475bc2ee5ce7202e2916c04a3` |
| `p2_G3-recipe-complete-generator_16_distribution_axes.csv` | `7f9e98b62ec0c9c8defaac09fe78b4c4e8c32878cc890bfa29a7da24948846cc` |
| `p2_G3-recipe-complete-generator_1_action_smoke.csv` | `53269a1e067efde99d69a071e19c577e3daec67e095087c87434188e886320f0` |
| `p2_G3-recipe-complete-generator_1_distribution_axes.csv` | `99db7fab890c15cfd93d7e3946d6f310015708613ce373791bb63d6a2ff693c5` |
| `p2_G3-recipe-complete-generator_256_action_smoke.csv` | `353ea47f0b1af446fd03149abd2fe369354919b2cbd937c851270cbf8e035417` |
| `p2_G3-recipe-complete-generator_256_distribution_axes.csv` | `8d8dbe72cd75bf24700ad7edbe5d6b468507708c4b25bd2db0d3457c22ef4cf4` |
| `p2_G4-randomized-trainstream-window-generator_1024_action_fidelity_pilot.csv` | `7a1c0072dd46c67a7bc7cc728d0e5d388b64a66853a4fc2e775d054cec73b1e3` |
| `p2_G4-randomized-trainstream-window-generator_1024_action_labels.csv` | `a2f50f6d4ea7f6719cbfa8f93003b10720b3282106dfbe587ea18cf5672db4d8` |
| `p2_G4-randomized-trainstream-window-generator_1024_distribution_axes.csv` | `3fbb89e43809a6853b00e246a6d2fedc07089de5fc13315efa8956d7ebdeed67` |
| `p2_G4-randomized-trainstream-window-generator_16_action_smoke.csv` | `90d7404ba36ce5567be5bf350f7f5158beb7938a21f7e2e20016d7a587214568` |
| `p2_G4-randomized-trainstream-window-generator_16_distribution_axes.csv` | `599ad1924ae41837c6732517212706f775a7604dda277020eafee072d237fef4` |
| `p2_G4-randomized-trainstream-window-generator_1_action_smoke.csv` | `7daadd335cdc028aa5cc15f3db2b5b141c99061e8026367cdd43dcbe4d6aa1e2` |
| `p2_G4-randomized-trainstream-window-generator_1_distribution_axes.csv` | `facba175ec99c346ba879074f690b453e70378e51f416cec78840cddee6f488a` |
| `p2_G4-randomized-trainstream-window-generator_256_action_smoke.csv` | `ac23af6643aef4ca91e81dcf75d43abcb14eb691fd0020beca3fb94e151eea3a` |
| `p2_G4-randomized-trainstream-window-generator_256_distribution_axes.csv` | `d4304fa035ea250e67a9c36a0744acb51a47cfdd69eb10eacd0f9fc9fc306996` |
| `p2_G5-hybrid-quota-random-generator_1024_action_fidelity_pilot.csv` | `042b730097d3ffa826377cef2ff4bbda5cecf70b09714cde5877b57a8b1cec4f` |
| `p2_G5-hybrid-quota-random-generator_1024_action_labels.csv` | `aee20dd651c972eac2b54642a6569baa4308146f67f3b39f2c97b0a7fee2ec7d` |
| `p2_G5-hybrid-quota-random-generator_1024_distribution_axes.csv` | `3f619e9bf7137cbbd4523843b91550fdfe2ffd8991b8bf62f95ed2311ce151c2` |
| `p2_G5-hybrid-quota-random-generator_16_action_smoke.csv` | `71029db99326abbb463f958c23f8ee951d3de1cf3101bad87552ec8b5f088309` |
| `p2_G5-hybrid-quota-random-generator_16_distribution_axes.csv` | `39267c0c00e7abfd1248feb98d58d8517297203c03c641a07e4137d56d13da2b` |
| `p2_G5-hybrid-quota-random-generator_1_action_smoke.csv` | `273e8596566817b9be1f72db512cdbe6a690291a59654d62465db8fd8dd1a9df` |
| `p2_G5-hybrid-quota-random-generator_1_distribution_axes.csv` | `02f6e838138544c4af7ce37c4b180d008a18bb53d651cc7a8eb588c2ffcca660` |
| `p2_G5-hybrid-quota-random-generator_256_action_smoke.csv` | `1ae868c8c33bd3cd5eb6aef460d03117671d6c7c5fdb610dd7f9137da7c0a726` |
| `p2_G5-hybrid-quota-random-generator_256_distribution_axes.csv` | `7b9c0802d1285771ba76289bf858cedc3deb3624aad0be80b7ea13fe62f0adb4` |
| `p2_generator_candidate_matrix.csv` | `5cee71b561306715b99be0fe3e4392bd53ca8f5a662e7c894ff384a9ef890c12` |
| `p2_generator_repair_failure_matrix.csv` | `1c5f430bc56e451aefb2b393e196f44a0ba1fce1d2d8285cc061393ae596be2c` |
| `p3_density_decision.json` | `7ca20e8bc07377bca9fc348366c3683e51dde28a68bb79c79591b71289ceed6e` |
| `p3_density_panel_10000.csv` | `74853e2c044ab9c805c78beacc480182594bea4c597d0b24bbcc48b182b51476` |
| `p3_density_panel_1024.csv` | `c3b6e1a3f7fd65d8f01136e08c51bec42900e998f225b2fc04f4e4071e1ca247` |
| `p3_density_panel_20000.csv` | `d985e46e2772fb7b97ea5f5c54c6affafd9cb4211115000dad18ec71158f4c2b` |
| `p3_density_panel_5000.csv` | `f408f922d3489e5ec8251655e526d1101bd9f89848fc9c67d32ea6c6105c966e` |
| `p4_future_path_curves_by_group.csv` | `7a10dec60ba774e94074ce9e96cda9fee873933389b24345edb182dbef32ee23` |
| `p4_future_path_types.csv` | `7a10dec60ba774e94074ce9e96cda9fee873933389b24345edb182dbef32ee23` |
| `p5_future_operator_sketch_v4.csv` | `d9c076d8ed35f89209e622052ca6ef27c0a0bf797a7de462c3e84d60e6a25cc2` |
| `p6_existing_action_controller_gate.csv` | `a8bea37cab00a4471818a1ff4c69e4e1ad296ceff9455ffa9ff35a8c03804d7b` |
| `p7_generated_sandbox_gate.csv` | `fefecba66fdddb6972e9dd4bad3e659d286ce77e1f2aea7e69542cd6c020c95a` |
| `p8_paired_replay_boundary.csv` | `b42d4d412ffb6be2c31d1ba468625f887dc409cc1fab9b0337f1d57c08e5750c` |
| `p8_selected_runtime_boundary.csv` | `c017ecd6243460f8ffa5fce0691ee641bbac868450cbd26700a4581e37185bd9` |
| `plan` | `c090daab609fc1b22d75ec26b519dd96ef2bcb56ad02ba5c423112371176d516` |
| `route_decision_v9920.json` | `89947cfdddbbde86e1988da083bd6c49872853dddbffcca717f056bb26ad3153` |
| `run_manifest_v9920.json` | `a4d52b81b14853a7cceeb8c064b2c6f21eb8eb52ef0dad6691fbe0afef4a8581` |
| `runner` | `eecd626d2690f3348d735faafbcb3bc4b8756e5920106c8aa2bf7584877bb6ab` |

## 8. 最终分析结论

```text
1. v9.9.2 没有把 v9.9.1 的低密度 pilot 写成 natural AP0 density fail；它先做分布修复门。
2. 原始 G0 的分布塌缩被复现并定位到 step/payload/cursor coverage。
3. 修复候选使用真实 materializer 生成新 payload、真实 action apply、真实 branch-horizon replay；没有 fake/proxy rows。
4. 只有通过分布 weak gate 的候选才允许进入 P3；否则 full 5000/10000/20000 继续关闭。
5. controller/generated/runtime/paired replay 仍只按 gate 打开，未通过时保持 not_run。
```

最终一句话：

> v9.9.2 真实执行后停在 `R2-DistributionFidelityPassDensityPending`：本轮把自然生成器问题从“发现分布失真”推进到候选修复审计；但没有满足 official controller/runtime/generated gate，因此 strict PureKAN functional 仍未成功。
