# DG-KAN v22.14 Basis-State Loss-Interface Operator FU 执行日志

生成时间：2026-06-09 01:17:11 +0800

记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。

## 2026-06-09 01:19:51 +0800 s0_construct

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage s0_construct --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12
```

- gpu: n/a
- status: completed
- note: exit=0

## 2026-06-09 01:19:51 +0800 efficiency

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage efficiency --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 2 --warmup 1
```

- gpu: cuda:0
- status: running
- note: launched by v22.14 dynamic queue runner

## 2026-06-09 01:19:51 +0800 f1_anchor_split

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage f1 --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:1 --f1-seeds 2213 --f1-adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter,Delta-PreferenceAdapter-smoke,Delta-StableRandom-control,Delta-RandomMatched-control --f1-modes operator_only,periodic_source_state,auxiliary_loss_anchor,optimizer_prox_anchor,source_state_projector
```

- gpu: cuda:1
- status: running
- note: launched by v22.14 dynamic queue runner

## 2026-06-09 01:19:51 +0800 kan_basis

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage kan --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:2
```

- gpu: cuda:2
- status: running
- note: launched by v22.14 dynamic queue runner

## 2026-06-09 01:19:57 +0800 efficiency

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage efficiency --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 2 --warmup 1
```

- gpu: cuda:0
- status: completed
- note: exit=0

## 2026-06-09 02:05:38 +0800 f1_anchor_split

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage f1 --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:1 --f1-seeds 2213 --f1-adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter,Delta-PreferenceAdapter-smoke,Delta-StableRandom-control,Delta-RandomMatched-control --f1-modes operator_only,periodic_source_state,auxiliary_loss_anchor,optimizer_prox_anchor,source_state_projector
```

- gpu: cuda:1
- status: completed
- note: exit=0

## 2026-06-09 02:05:38 +0800 kan_basis

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage kan --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:2
```

- gpu: cuda:2
- status: completed
- note: exit=0

## 2026-06-09 02:05:40 +0800 task_gate

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage task_gate --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:3
```

- gpu: cuda:3
- status: completed
- note: exit=0

## 2026-06-09 02:07:55 +0800 finalize_rerun

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_14_basis_state_operator_fu.py --stage finalize --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12
```

- gpu: n/a
- status: completed
- note: manual rerun after fixing artifact_index relative path bug; no experiment rows rerun

## 2026-06-09 03:15:48 +0800 efficiency_repair

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_14_repair_continuation.py --stage efficiency_repair --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --device-efficiency cuda:0 --batch-sizes 128,256,512,1024 --hidden 128 --repeats 5 --warmup 3
```

- gpu: cuda:0
- status: completed
- note: route=E1Repair-VariantRobustEfficiencyPass profile_rows=240 robust=4 blocker=

## 2026-06-09 03:43:06 +0800 f1_repair

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_14_repair_continuation.py --stage f1_repair --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --device-f1 cuda:1 --f1-modes optimizer_prox_jacobian_rank16,optimizer_prox_jacobian_rank32,source_state_projector_jacobian_rank32
```

- gpu: cuda:1
- status: completed
- note: route=F1Repair-JacobianSourceStateNoGo rows=18 blocker=JacobianSourceStateNoGo

## 2026-06-09 03:57:34 +0800 finalize_repair

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_14_repair_continuation.py --stage finalize_repair --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213
```

- gpu: n/a
- status: completed
- note: route=R5-OptimizerProxAnchorNoGo blocker=JacobianSourceStateNoGo;R7-KANBasisTangentCoverageNoGo
