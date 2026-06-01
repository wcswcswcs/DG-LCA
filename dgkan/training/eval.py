"""Evaluation helpers used by DG-KAN runners and diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def metric_unavailable() -> str:
    return "metric_unavailable"


def classification_basic(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        correct = pred.eq(y)
        ce = F.cross_entropy(logits, y, reduction="none")
        true = probs.gather(1, y.view(-1, 1)).squeeze(1)
        top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else true
        ece = torch.abs(conf - correct.float()).mean()
        return {
            "acc": float(correct.float().mean().item()),
            "NLL": float(loss.item()),
            "ECE": float(ece.item()),
            "CEp99": float(torch.quantile(ce.detach().float(), 0.99).item()),
            "margin_p10": float(torch.quantile(margin.detach().float(), 0.10).item()),
        }
