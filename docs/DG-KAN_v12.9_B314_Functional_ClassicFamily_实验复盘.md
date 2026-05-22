# DG-KAN v12.9 B314 Functional ClassicFamily 实验复盘入口

本文件只作为入口，避免 runner 自动生成的 v12.8.3 模板复盘覆盖 v12.9 的真实审计结论。

完整复盘、关键实验数据、修复尝试、拒绝理由和更新后的下一步结论记录在：

```text
docs/DG-KAN_v12.9_B314_Functional_ClassicFamily_执行复盘.md
```

当前结论：

```text
B314 old-gate = pass, but B314 v12.9 strict AUC = fail / near-pass
B320/B321 global-gain repair = A1 10-seed strict task/ECE/AUC pass
B320 = current strict-efficiency anchor candidate; B320-only Line C reconfirm pass
Functional official success = fail; best observed B320 branch-damping gap = -0.00014220178469248612
Classic family pass count = 0; B3y/B3z repaired Chebyshev paircrossR32 L3 efficiency, but B3aa-B3am input-cross / projection-square / localrot2 brackets still failed A4 expression or failed L3 efficiency
Rational RAT-B4 manual AdamW temporary-allocation repair can reopen L3/A4 in focused runs; B7kf reaches task mean/worst/near/ECE subgate pass but still fails KMNIST AUC_step/AUC_time. B7ki/B7kj no-hidden lower-cap failed L3; B7fe/B7ga/B7ge low-rank recheck passed L3 but failed A4 expression; B7kk/B7kl middle-rank bracket remained just over L3 gate; B7km/B7kn opened L3 but failed A4 expression; B7ko/B7kp/B7kq and B7kr-B7kv pair-coverage variants opened L3 except B7kt, but still failed A4; B7kw/B7kx balanced-dominant mix failed L3. B7ky/B7kz/B7la R32 projection sidecars opened L3 but failed A4; B7lb-B7lg R48/R64 projection sidecars failed L3; B7lh/B7li/B7lj low-cost pair-feature fallback left B7li/B7lj at L3 pass but A4 fail and B7lh at L3 fail. B7lk/B7ll/B7lm random-projection inputcross R64/R96/R112 and B7ln/B7lo low-rank R32/R48 passed analytic gradcheck but all failed L3. New Rational denominator/derivative diagnostics were added; B7kc autopsy reconfirmed L3 pass + A4 pass + A5 fail, with den_p01 = 1.0000253915786743 and status = TaskBlocked. B7en pairNorm improved E1/E6/E8 expression deltas while preserving L3/A4, but still failed A5. B7lp combined pairNorm + hiddenRatResidual005 and reached L3 step_ratio = 0.7347600905789334 with A4 pass, but A5 still failed. Family Line C autopsy was added and showed B7kc/B7en/B7lp all have coupling_collapse. B7lq/B7lr pairNorm + stop-gradient batch cap kept L3/A4 but still showed coupling_collapse and A5 fail. B7ls/B7lt freezeRational/trainable-w1 kept L3/A4 but also coupling_collapse and raised RealSignalReservoirRatio to about 0.827/0.828, a negative reservoir-trapping result. B7lu/B7lv shared pair-signal readout was implemented and audited; B7lu failed L3, B7lv passed L3 but failed A4 and both stayed coupling_collapse with RealSignalReservoirRatio about 0.218. B7kd/B7ke low linear-residual gain Line C first-pass also stayed coupling_collapse and increased RealSignalReservoirRatio to 0.248/0.300 while worsening A5. B7lw/B7lx PCA whitening was implemented and audited; B7lx passed L3/A4 but Line C CouplingR2 fell to about 0.130 and CEp99 exploded above 248, a clear negative result. B7ly/B7lz removed whitening; B7lz passed L3/A4 with A5 mean_delta = 0.000977 and near_pass_rate = 0.75, but still failed A5 and Line C stayed coupling_collapse with CouplingR2 about 0.240 and RealSignalReservoirRatio about 0.219. B7ma-B7mh output-scale geometry was implemented and audited; samplewise B7ma/B7mb improved Line C CouplingR2 to about 0.297/0.300 but failed L3, batch B7mc/B7md/B7me/B7mf passed L3/A4 but failed A5 badly, and residual mix B7mg/B7mh failed L3. Best new Line C value was B7me CouplingR2 = 0.3113662178976532, still below MLP 0.3356805192996951 and paired with A5 failure. Rational remains not family success.
```

不得从 runner 自动生成模板中的旧字段推断 v12.9 official success；所有 v12.9 结论以 `执行复盘.md` 和对应 `results/v12_9_b314_functional_classic_family/*` artifact 为准。
