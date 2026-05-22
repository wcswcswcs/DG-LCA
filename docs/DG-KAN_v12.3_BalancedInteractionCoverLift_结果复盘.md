# DG-KAN v12.3 Balanced Interaction-Cover Lift 结果复盘

> 本复盘记录 `DG-KAN_v12.3_实验结果独立分析与下一步总计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Functional 只在 base gate 通过后才允许打开。

## 0. 最新结论

```text
route = R6-LQFrameFamilyExhausted
base_qualified = False
functional_open = False
next_recommended_action = pivot to primitive-level design after C11-C17 failed joint gates
```

最终 artifact：

```text
results/v12_3_balanced_interaction_cover_lift/v123_balanced_interaction_cover_lift_20260519T160000Z
```

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_lq.py` | 新增 `balanced_signed_pair_whiten`、`srht_quadratic_frame`、`block_orthogonal_pair_cover`、`low_coherence_random_frame`、`two_subframe_lq`、`condition_normalized_lift` 初始化 | 均只用 unlabeled train-stream stats 或固定随机种子，不按 dataset name/label 分支；用于 v12.3 C11-C16 frame family。 |
| `experiments/run_v123_balanced_interaction_cover_lift.py` | 新增 v12.3 Batch 1-5 runner | 独立输出 v123 artifact，functional 只在 base pass 后打开；没有改低 gate。 |
| `experiments/run_v123_balanced_interaction_cover_lift.py` | 执行中修复：D 线由 top-K 表达候选改为 C0/C8/C10/C11-C17 全量 triage | 第一版正式 run 只推进表达 top-K，漏掉 C11/C15/C17 vision gate；这不符合 v12.3 C-track 完整执行，因此修复后重跑正式 artifact。 |
| `experiments/run_v123_balanced_interaction_cover_lift.py` | 执行中修复：Track A timing 加入 C0 clean row 与 A7 hook path row | 第一版 C0 scorecard timing 缺失；修复后 architecture clean timing 与 diagnostic hook timing 分开落盘，避免混淆。 |
| `experiments/run_v123_balanced_interaction_cover_lift.py` | 执行中修复：补落 `v123_functional_route.json` | v12.3 计划要求 functional route artifact；base 未通过时写真实 `not_run/base_not_qualified`，不写 success。 |

本轮没有做：

```text
1. 没有调低 functional gate。
2. 没有按 MNIST/Fashion/KMNIST 名称分支。
3. 没有使用 teacher/distillation/loss modification/class weights。
4. 没有把 diagnostic/not_run row 写成 success。
```

## 2. Route

```json
{
  "stage": "V123_FINAL_BASE_ROUTE",
  "route": "R6-LQFrameFamilyExhausted",
  "base_qualified": false,
  "functional_open": false,
  "promoted_candidates": [],
  "any_expression_gate_pass": false,
  "any_condition_gate_pass": false,
  "any_task_gate_pass": false,
  "next_recommended_action": "pivot to primitive-level design after C11-C17 failed joint gates",
  "failure_count": 10,
  "no_fake": true,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. Artifact 完整性

Required artifact 均已落盘：

```text
v123_frame_manifest.csv rows = 10
v123_timing_accounting.csv rows = 12
v123_expression_capacity.csv rows = 90
v123_matrix_span_audit.csv rows = 50
v123_frame_failure_table.csv rows = 20
v123_base_triage.csv rows = 108
v123_base_task_trace.csv rows = 648
v123_base_geometry_snapshot.csv rows = 648
v123_system_profile.csv rows = 108
v123_triage_scorecard.csv rows = 10
v123_geometry_trajectory.csv rows = 162
v123_signal_noise_battery.csv rows = 162
v123_tail_calibration.csv rows = 162
v123_geometry_certificate.csv rows = 3
v123_base_confirm.csv rows = 1
v123_functional_route.csv rows = 1
v123_functional_route.json = written
```

## 4. Track A Timing Accounting

| method | step q90 ms | architecture ratio | online/hook ratio | geometry hook q90 ms |
|---|---:|---:|---:|---:|
| `B0-MLP-same-param-AdamW` | `0.4465301986783743` | `1.0` |  | `0.0` |
| `C0-R2-LQ-current` | `0.5067245103418827` | `1.134804570534467` |  | `0.0` |
| `C8-signed-pair-cover-lift` | `0.5085375159978867` | `1.1388647789176178` |  | `0.0` |
| `C10-data-pca-whiten-h64` | `0.5019913427531719` | `1.1242046881464003` |  | `0.0` |
| `C11-balanced-signed-pair-whiten` | `0.5064050201326609` | `1.1340890753447408` |  | `0.0` |
| `C12-srht-quadratic-frame` | `0.5376184359192848` | `1.2039912138316973` |  | `0.0` |
| `C13-block-orthogonal-pair-cover` | `0.6702290382236243` | `1.5009713569369925` |  | `0.0` |
| `C14-low-coherence-random-frame` | `0.5699978675693274` | `1.2765046334075247` |  | `0.0` |
| `C15-two-subframe-lq` | `0.5268998444080353` | `1.1799870332791298` |  | `0.0` |
| `C16-condition-normalized-lift` | `0.526323402300477` | `1.1786960968334776` |  | `0.0` |
| `C17-t2t3-composition-diagnostic` | `0.6664276123046875` | `1.492458100879086` |  | `0.0` |
| `A7-C0-runner-profile-with-hooks` | `85.65710880793632` | `1.372788740789994` | `191.82825497908422` | `85.09282195009293` |

解释：C0/C8/C10/C11/C12/C15/C16 clean step 在 `1.25x` 附近或以内；C13/C14/C17 超过 clean system gate。A7 hook path 再次证明 diagnostic hook 不能混进 architecture step timing。

## 5. Track B/C/D Triage Scorecard

| candidate | mean delta vs MLP | E1 LS | E6 LS | E8 LS | condition ratio vs C0 | promote |
|---|---:|---:|---:|---:|---:|---:|
| `C0-R2-LQ-current` | `-0.006076388888888889` | `0.42912566661834717` | `0.42785489559173584` | `0.3477097749710083` | `1.0` | `0` |
| `C14-low-coherence-random-frame` | `-0.006727430555555556` | `0.5108539462089539` | `0.43677234649658203` | `0.39776474237442017` | `1.030776216782355` | `0` |
| `C12-srht-quadratic-frame` | `-0.009548611111111112` | `0.4701780676841736` | `0.2512388229370117` | `0.46192288398742676` | `1.0510252799231112` | `0` |
| `C16-condition-normalized-lift` | `-0.009982638888888888` | `0.5180319547653198` | `0.36144816875457764` | `0.4174286127090454` | `0.9928802504820333` | `0` |
| `C17-t2t3-composition-diagnostic` | `-0.011935763888888888` | `0.21324002742767334` | `0.1571124792098999` | `0.24458271265029907` | `1.5440031113101984` | `0` |
| `C10-data-pca-whiten-h64` | `-0.018229166666666668` | `0.379244863986969` | `0.42895132303237915` | `0.32619500160217285` | `0.18423722174162988` | `0` |
| `C8-signed-pair-cover-lift` | `-0.019314236111111112` | `0.9923796057701111` | `0.1526818871498108` | `0.4375520944595337` | `1.2696238702058613` | `0` |
| `C13-block-orthogonal-pair-cover` | `-0.019314236111111112` | `0.9922178387641907` | `0.2403741478919983` | `0.4780576229095459` | `1.2696238702058613` | `0` |
| `C15-two-subframe-lq` | `-0.022135416666666668` | `0.3869255781173706` | `0.2065664529800415` | `0.23273998498916626` | `1.6209712847284223` | `0` |
| `C11-balanced-signed-pair-whiten` | `-0.024305555555555556` | `0.19958257675170898` | `0.15792137384414673` | `0.1699211597442627` | `1.6915665497361987` | `0` |

所有 candidate 的 promotion gate 均为 `0`。没有任何 candidate 同时满足 task near、worst-row safety、expression coverage、condition drop 与 system gate。

## 6. C11-C17 Expression 摘要

| candidate | E1 LS | E1 trainable R2 | E2 trainable delta vs MLP | E6 LS | E8 LS |
|---|---:|---:|---:|---:|---:|
| `C11-balanced-signed-pair-whiten` | `0.19958257675170898` | `0.9875513911247253` | `-0.012648165225982666` | `0.15792137384414673` | `0.1699211597442627` |
| `C12-srht-quadratic-frame` | `0.4701780676841736` | `0.9949496388435364` | `-0.007343769073486328` | `0.2512388229370117` | `0.46192288398742676` |
| `C13-block-orthogonal-pair-cover` | `0.9922178387641907` | `0.996874988079071` | `-0.003355264663696289` | `0.2403741478919983` | `0.4780576229095459` |
| `C14-low-coherence-random-frame` | `0.5108539462089539` | `0.9952221512794495` | `-0.005067408084869385` | `0.43677234649658203` | `0.39776474237442017` |
| `C15-two-subframe-lq` | `0.3869255781173706` | `0.9935275912284851` | `-0.004896283149719238` | `0.2065664529800415` | `0.23273998498916626` |
| `C16-condition-normalized-lift` | `0.5180319547653198` | `0.9919909834861755` | `-0.00640106201171875` | `0.36144816875457764` | `0.4174286127090454` |
| `C17-t2t3-composition-diagnostic` | `0.21324002742767334` | `0.991262674331665` | `-0.009348809719085693` | `0.1571124792098999` | `0.24458271265029907` |

解释：trainable pairwise R2 多数接近 `0.99`，但 frozen/global matrix-span coverage 仍低；`C13` 继承了 signed-pair 局部 E1 优势，却没有修复 rotated/random quadratic coverage，也没有修复 task stability 或 condition。

## 7. Gate Closure

```text
Track F base confirmation = not_run, reason = no_candidate_passed_trackD_E_gates
Track G functional complementarity = not_run, reason = base_not_qualified
functional_open = false
```

本轮没有把 not_run 写成 pass；也没有因为 clean timing 局部通过而打开 functional。

## 8. No-Fake / Hash

```text
rows_checked = 2203
fake/proxy/cpu = 0 / 0 / 0
```

| artifact | SHA256 |
|---|---|
| `v123_base_triage.csv` | `2db4c1e8bed0d382753f081409fec7f2b3d98bc1deca3bb934869493d7baf4e6` |
| `v123_expression_capacity.csv` | `13fae1ef7b0b42e9e0b4a823a2f92d68855eaa66bdbc1d74586432b2772d50ef` |
| `v123_final_base_route.json` | `81614f77fe219b5d9e5eeee3790126e0871769ce8eca82cddd983a3696d52995` |
| `v123_frame_manifest.csv` | `678d14ea6d3177b3a1d0a07a904f23d45d86146e31552a1c8bad44e05dade340` |
| `v123_geometry_certificate.csv` | `92371801cef269148c0d98c422f00e43f0bc0b47a2d637c564bc7e7dfc34ebe5` |
| `v123_matrix_span_audit.csv` | `7ffad830826635b84c12c8a78c7a0bdc6cf146708d10c78967e520e6187d63a1` |
| `v123_provenance_audit.csv` | `0024ddccf7fcb54243e80783569c668973050c7596346215f74e8efaf8bd1d44` |
| `v123_timing_accounting.csv` | `6fc917fa1809587dc7168ccfddc81e9ae56ec198c129e73666d7ffc1feff77ad` |
| `v123_triage_scorecard.csv` | `c8d1cccab20993246196ffc225f63bbac8ae40915e77c77c009ce6c4feeaccd2` |

## 9. 最终分析结论

```text
1. v12.3 已全量尝试 C11-C17 balanced frame family，并完成 Batch 1/2/3；Batch 4/5 因 base gate 失败合法关闭。
2. Clean timing 不再是主 blocker，但 C13/C14/C17 的 system gate 仍失败；A7 hook path 继续极慢，必须保持与 clean architecture timing 分离。
3. C13/C8 的 E1 局部 coverage 接近 1.0，但 rotated/random quadratic coverage、condition 与 task stability 没有同时满足。
4. C10 仍是 condition diagnostic 的最好方向之一，condition ratio = 0.18423722174162988，但 task 与 expression gate 都没过。
5. route = R6-LQFrameFamilyExhausted：下一步应按 v12.3 route 建议转向 primitive-level design，而不是继续小修同一 LQ frame 或打开 functional。
```
