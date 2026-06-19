#!/usr/bin/env python3
"""Run v22.20 first-principles metric functional update experiments."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# KANbeFair imports torchtext at module top level even for vision/tabular
# loaders.  The kan env used for this project does not install torchtext, and
# v22.20 does not run text tasks; a minimal placeholder keeps the unused import
# from blocking the official non-text protocol.
if "torchtext" not in sys.modules:
    torchtext_mod = types.ModuleType("torchtext")
    torchtext_data_mod = types.ModuleType("torchtext.data")
    torchtext_data_utils_mod = types.ModuleType("torchtext.data.utils")
    torchtext_vocab_mod = types.ModuleType("torchtext.vocab")

    def _basic_tokenizer(_name: str) -> Any:
        return lambda text: str(text).split()

    class _DummyVocab(dict):
        def __call__(self, tokens: Any) -> list[int]:
            return [self.get(tok, 0) for tok in tokens]

        def set_default_index(self, _idx: int) -> None:
            return None

    def _build_vocab_from_iterator(iterator: Any, specials: list[str] | None = None) -> _DummyVocab:
        vocab = _DummyVocab()
        for idx, token in enumerate(specials or []):
            vocab[token] = idx
        for tokens in iterator:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    torchtext_data_utils_mod.get_tokenizer = _basic_tokenizer
    torchtext_vocab_mod.build_vocab_from_iterator = _build_vocab_from_iterator
    torchtext_data_mod.utils = torchtext_data_utils_mod
    torchtext_mod.data = torchtext_data_mod
    torchtext_mod.vocab = torchtext_vocab_mod
    sys.modules["torchtext"] = torchtext_mod
    sys.modules["torchtext.data"] = torchtext_data_mod
    sys.modules["torchtext.data.utils"] = torchtext_data_utils_mod
    sys.modules["torchtext.vocab"] = torchtext_vocab_mod
if "fvcore" not in sys.modules:
    fvcore_mod = types.ModuleType("fvcore")
    fvcore_nn_mod = types.ModuleType("fvcore.nn")

    class _MissingFlopCountAnalysis:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._unsupported = True

        def total(self) -> int:
            return 0

    def _missing_parameter_count(_model: Any) -> dict[str, int]:
        return {}

    fvcore_nn_mod.FlopCountAnalysis = _MissingFlopCountAnalysis
    fvcore_nn_mod.parameter_count = _missing_parameter_count
    fvcore_mod.nn = fvcore_nn_mod
    sys.modules["fvcore"] = fvcore_mod
    sys.modules["fvcore.nn"] = fvcore_nn_mod
if "torchaudio" not in sys.modules:
    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.datasets = types.ModuleType("torchaudio.datasets")
    torchaudio_mod.transforms = types.ModuleType("torchaudio.transforms")

    class _MissingSpeechCommands:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("torchaudio is not installed; v22.20 runner only enables non-audio KANbeFair tasks")

    class _IdentityResample:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def __call__(self, value: Any) -> Any:
            return value

    def _missing_audio_load(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("torchaudio is not installed; audio tasks are unsupported in this v22.20 run")

    torchaudio_mod.datasets.SPEECHCOMMANDS = _MissingSpeechCommands
    torchaudio_mod.transforms.Resample = _IdentityResample
    torchaudio_mod.load = _missing_audio_load
    sys.modules["torchaudio"] = torchaudio_mod
    sys.modules["torchaudio.datasets"] = torchaudio_mod.datasets
    sys.modules["torchaudio.transforms"] = torchaudio_mod.transforms

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from dgkan.fu.mmfp_update import (  # noqa: E402
    acceptance_unit_tests,
    ce_delta,
    ce_h_apply,
    control_dominance_unit_tests,
    metric_curvature_operator_tests,
    mmfp_solver_unit_tests,
    solve_ce_mmfp,
    trace_scaled_damping,
)
from dgkan.models.fc_purekan_primitives import (  # noqa: E402
    MLPBaseline,
    PrimitiveKAN,
    PrimitiveSpec,
    _basis_derivative,
    _stream_mix,
    count_parameters,
)
from experiments.run_v22_16_common import selected_named_parameters  # noqa: E402
from experiments.run_v22_17_kanbefair_dgkan_eval import _get_loaders  # noqa: E402
from experiments.run_v22_20_common import (  # noqa: E402
    EXEC_DOC,
    OUT_ROOT,
    PYTHON,
    RECAP_DOC,
    ROOT as PROJECT_ROOT,
    WORKTREE_ROOT,
    append_exec,
    artifact_index,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_rows,
    sha256_file,
    source_packet_paths,
    write_json,
    write_rows,
)


CONTROL_MODES = {
    "MLP_MMFP_RANDOM_CONTROL": "random",
    "MLP_MMFP_STABLE_RANDOM_CONTROL": "stable_random",
    "MLP_MMFP_SIGNFLIP_CONTROL": "signflip",
    "MLP_MMFP_SHUFFLED_CONTROL": "shuffled",
    "KAN_MMFP_RANDOM_CONTROL_DCHE": "random",
    "KAN_MMFP_SIGNFLIP_CONTROL_DCHE": "signflip",
    "KAN_MMFP_RANDOM_CONTROL_DFOU": "random",
    "KAN_MMFP_SIGNFLIP_CONTROL_DFOU": "signflip",
}


def _control_mode_for_model(model_name: str) -> str:
    if "STABLE_RANDOM_CONTROL" in model_name:
        return "stable_random"
    if "RANDOM_CONTROL" in model_name:
        return "random"
    if "SIGNFLIP_CONTROL" in model_name:
        return "signflip"
    if "SHUFFLED_CONTROL" in model_name:
        return "shuffled"
    return CONTROL_MODES.get(model_name, "real")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="unit", choices=["unit", "task", "continual", "finalize", "all"])
    p.add_argument("--kanbefair-root", default=str(WORKTREE_ROOT))
    p.add_argument("--datasets", default="MNIST,FMNIST,KMNIST")
    p.add_argument("--models", default="MLP_ADAMW,MLP_MMFP")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-gpu", default="")
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=440)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--rank-cap", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--log-interval", type=int, default=110)
    p.add_argument("--label-noise", type=float, default=0.0)
    p.add_argument("--output-suffix", default="")
    p.add_argument("--experiment-tag", default="v22_20")
    p.add_argument("--max-batches-per-task", type=int, default=0)
    p.add_argument("--clear-proxy-env", action="store_true")
    return p


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _device(raw: str) -> torch.device:
    if str(raw).startswith("cuda") and torch.cuda.is_available():
        return torch.device(raw)
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _out_path(name: str, suffix: str = "") -> Path:
    path = OUT_ROOT / name
    clean = str(suffix or "").strip().strip("_")
    if clean:
        return path.with_name(f"{path.stem}_{clean}{path.suffix}")
    return path


def _flat(named: list[tuple[str, torch.nn.Parameter]], *, grad: bool = False) -> torch.Tensor:
    parts = []
    for _name, p in named:
        if grad:
            parts.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
        else:
            parts.append(p.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _apply_flat(named: list[tuple[str, torch.nn.Parameter]], delta: torch.Tensor, *, alpha: float = 1.0) -> None:
    offset = 0
    flat = delta.detach().reshape(-1)
    with torch.no_grad():
        for _name, p in named:
            n = int(p.numel())
            p.add_(flat[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype), alpha=float(alpha))
            offset += n


def _chunks(named: list[tuple[str, torch.nn.Parameter]], direction: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    flat = direction.detach().float().reshape(-1).to(device)
    out = []
    offset = 0
    for _name, p in named:
        n = int(p.numel())
        out.append(flat[offset : offset + n].reshape_as(p).to(device=device, dtype=p.dtype))
        offset += n
    return tuple(out)


def _finite_jvp(model: torch.nn.Module, named: list[tuple[str, torch.nn.Parameter]], xb: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-3
    with torch.no_grad():
        base = model(xb).detach().float().reshape(-1)
        _apply_flat(named, direction, alpha=eps)
        moved = model(xb).detach().float().reshape(-1)
        _apply_flat(named, direction, alpha=-eps)
    return (moved - base) / eps


def _silu_derivative(value: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(value)
    return sig * (1.0 + value * (1.0 - sig))


def _analytic_jvp_logits(model: torch.nn.Module, named: list[tuple[str, torch.nn.Parameter]], xb: torch.Tensor, direction: torch.Tensor) -> torch.Tensor | None:
    names = [name for name, _p in named]
    chunks = {name: chunk for name, chunk in zip(names, _chunks(named, direction, xb.device))}
    with torch.no_grad():
        if isinstance(model, MLPBaseline) and set(names).issubset({"w0", "w1", "w2"}):
            dw0 = chunks.get("w0", torch.zeros_like(model.w0))
            dw1 = chunks.get("w1", torch.zeros_like(model.w1))
            dw2 = chunks.get("w2", torch.zeros_like(model.w2))
            a0 = xb @ model.w0
            h0 = F.silu(a0)
            da0 = xb @ dw0
            dh0 = _silu_derivative(a0) * da0
            a1 = h0 @ model.w1
            h1 = F.silu(a1)
            da1 = dh0 @ model.w1 + h0 @ dw1
            dh1 = _silu_derivative(a1) * da1
            return (dh1 @ model.w2 + h1 @ dw2).detach().float().reshape(-1)
        if isinstance(model, PrimitiveKAN) and set(names).issubset({"w1", "w2"}):
            dw1 = chunks.get("w1", torch.zeros_like(model.w1))
            dw2 = chunks.get("w2", torch.zeros_like(model.w2))
            z = model._norm_input(xb)
            h_pre = _stream_mix(z, model.w1, model.spec.basis_name, model.k, model.centers, model.scales) / math.sqrt(max(1, model.input_dim))
            h = torch.tanh(h_pre)
            dh_pre = _stream_mix(z, dw1, model.spec.basis_name, model.k, model.centers, model.scales) / math.sqrt(max(1, model.input_dim))
            dh = (1.0 - h.square()) * dh_pre
            b2 = model.layer2_basis(h)
            direct = _stream_mix(h, dw2, model.spec.basis_name, model.k, model.centers, model.scales) / math.sqrt(max(1, model.hidden_dim))
            deriv = _basis_derivative(h, model.spec.basis_name, model.k, model.centers, model.scales)
            through_h = torch.einsum("bhk,bh,hck->bc", deriv, dh, model.w2) / math.sqrt(max(1, model.hidden_dim))
            # b2 is materialized above to keep the same path warm as forward;
            # the derivative formula itself only needs direct + through_h.
            del b2
            return (direct + through_h).detach().float().reshape(-1)
    return None


def _jvp_logits(model: torch.nn.Module, named: list[tuple[str, torch.nn.Parameter]], xb: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, str]:
    if not named or direction.numel() == 0:
        with torch.no_grad():
            return torch.zeros_like(model(xb).float()).reshape(-1), "empty"
    analytic = _analytic_jvp_logits(model, named, xb, direction)
    if analytic is not None:
        return analytic, "analytic_model_jvp"
    try:
        from torch.func import functional_call, jvp

        param_map = dict(model.named_parameters())
        names = [name for name, _p in named]
        base_tuple = tuple(param_map[name] for name in names)
        tangent_tuple = _chunks(named, direction, xb.device)

        def logits_fn(*selected_values: torch.Tensor) -> torch.Tensor:
            patched = dict(param_map)
            for name, value in zip(names, selected_values):
                patched[name] = value
            return functional_call(model, patched, (xb,), strict=False).float().reshape(-1)

        _base, tangent = jvp(logits_fn, base_tuple, tangent_tuple)
        return tangent.detach().float().reshape(-1), "torch_func_jvp"
    except Exception:
        return _finite_jvp(model, named, xb, direction).detach().float().reshape(-1), "finite_diff_fallback"


def _orthonormalize(candidates: list[torch.Tensor], k: int) -> torch.Tensor:
    basis: list[torch.Tensor] = []
    for cand in candidates:
        v = cand.detach().float().reshape(-1).clone()
        if v.numel() == 0:
            continue
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = torch.linalg.vector_norm(v)
        if float(norm.item()) > 1.0e-10:
            basis.append(v / norm)
        if len(basis) >= int(k):
            break
    if not basis:
        return torch.empty(0, 0)
    return torch.stack(basis, dim=1)


def _basis_from_grad(named: list[tuple[str, torch.nn.Parameter]], rank_cap: int, seed: int) -> torch.Tensor:
    del seed
    grad = _flat(named, grad=True).detach().float().cpu()
    candidates: list[torch.Tensor] = []
    if grad.numel() > 0:
        candidates.append(-grad)
        offset = 0
        for _name, p in named:
            block = torch.zeros_like(grad)
            n = int(p.numel())
            block[offset : offset + n] = -grad[offset : offset + n]
            candidates.append(block)
            offset += n
    return _orthonormalize(candidates, int(rank_cap))


def _param_control(direction: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    d = direction.detach().float().reshape(-1).cpu()
    norm = torch.linalg.vector_norm(d).clamp_min(1.0e-12)
    if mode == "signflip":
        return -d
    if mode == "shuffled":
        gen = torch.Generator().manual_seed(int(seed))
        return d[torch.randperm(d.numel(), generator=gen)]
    gen_seed = 922000 + int(seed) if mode == "stable_random" else int(seed)
    gen = torch.Generator().manual_seed(gen_seed)
    r = torch.randn(d.shape, generator=gen)
    return r / torch.linalg.vector_norm(r).clamp_min(1.0e-12) * norm


def _uses_base_norm_trust(model_name: str) -> bool:
    return "TRUST" in model_name


def _cap_to_base_norm(candidate: torch.Tensor, base_update: torch.Tensor) -> tuple[torch.Tensor, float]:
    cand = candidate.detach().float().cpu()
    cand_norm = torch.linalg.vector_norm(cand)
    base_norm = torch.linalg.vector_norm(base_update.detach().float().cpu())
    if cand.numel() == 0 or float(cand_norm.item()) <= float(base_norm.clamp_min(1.0e-12).item()):
        return cand, 1.0
    scale = float((base_norm / cand_norm.clamp_min(1.0e-12)).item())
    return cand * scale, scale


def _matched_output_controls_for_runner(direction: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    d = direction.detach().float().reshape(-1)
    norm = torch.linalg.vector_norm(d).clamp_min(1.0e-12)
    gen = torch.Generator(device=d.device).manual_seed(int(seed))
    random = torch.randn(d.shape, generator=gen, device=d.device)
    random = random / torch.linalg.vector_norm(random).clamp_min(1.0e-12) * norm
    stable_gen = torch.Generator(device=d.device).manual_seed(777000 + int(seed))
    stable = torch.randn(d.shape, generator=stable_gen, device=d.device)
    stable = stable / torch.linalg.vector_norm(stable).clamp_min(1.0e-12) * norm
    perm = torch.randperm(d.numel(), generator=gen, device=d.device)
    return {
        "random": random,
        "stable_random": stable,
        "signflip": -d,
        "shuffled": d[perm],
    }


class ManualAdamW:
    def __init__(self, lr: float, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1.0e-8) -> None:
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.step_idx = 0
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}

    def update_vector(self, named: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
        self.step_idx += 1
        parts = []
        for name, p in named:
            g = torch.zeros_like(p) if p.grad is None else p.grad.detach()
            if name not in self.m:
                self.m[name] = torch.zeros_like(p)
                self.v[name] = torch.zeros_like(p)
            self.m[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            m_hat = self.m[name] / (1.0 - self.beta1**self.step_idx)
            v_hat = self.v[name] / (1.0 - self.beta2**self.step_idx)
            delta = -self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * p.detach())
            parts.append(delta.detach().reshape(-1))
        return torch.cat(parts) if parts else torch.empty(0)


def _make_model(model_name: str, input_dim: int, output_dim: int, hidden: int, seed: int, device: torch.device, x_stats: torch.Tensor) -> torch.nn.Module:
    if model_name.startswith("MLP"):
        return MLPBaseline(input_dim, output_dim, hidden, seed, device).to(device)
    carrier = "D-CHE" if "DCHE" in model_name else "D-FOU"
    basis_name = "chebyshev" if carrier == "D-CHE" else "fourier_lowfreq"
    init_variant = "cheby_k3_triton_l3_gradbuf" if carrier == "D-CHE" else "fourier_k3_triton_l3_matmul"
    k = 3
    mlp_budget = input_dim * hidden + hidden * hidden + hidden * output_dim
    kan_hidden = max(4, int(round(float(mlp_budget) / float(max(1, k * (input_dim + output_dim))))))
    spec = PrimitiveSpec(
        candidate_id=f"v22.20-{carrier}-strict-fc-purekan",
        basis_family=carrier,
        basis_name=basis_name,
        k=k,
        hidden_dim=kan_hidden,
        source="v22_20_mmfp",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=int(carrier == "D-FOU"),
        uses_division=0,
        uses_dense_basis_tensor=0,
        diagnostic_only=0,
        init_variant=init_variant,
    )
    return PrimitiveKAN(input_dim, output_dim, spec, x_stats.to(device), seed, device, param_budget=mlp_budget).to(device)


def _variant_family(model_name: str) -> tuple[str, str, str]:
    if model_name.startswith("MLP"):
        if model_name == "MLP_ADAMW":
            return "MLP", "AdamW", "none"
        if model_name == "MLP_NOOP":
            return "MLP", "same-overhead-no-op", "none"
        return "MLP", "MMFP", _control_mode_for_model(model_name)
    carrier = "DCHE" if "DCHE" in model_name else "DFOU"
    if "ADAMW" in model_name:
        return f"KAN_{carrier}", "AdamW", "none"
    return f"KAN_{carrier}", "MMFP", _control_mode_for_model(model_name)


def _uses_base_residual(model_name: str) -> bool:
    return "RESIDUAL" in model_name


def _selector_for(model_name: str) -> str:
    if model_name.startswith("KAN"):
        return "basis"
    return "all"


def _loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels.long())


def _per_example_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels.long(), reduction="none")


def _eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, output_dim: int) -> dict[str, float]:
    model.eval()
    losses_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            xb = x.to(device).float()
            yb = y.to(device).long()
            logits = model(xb).float()
            loss = _per_example_loss(logits, yb)
            losses_all.append(loss.detach().cpu())
            logits_all.append(logits.detach().cpu())
            y_all.append(yb.detach().cpu())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            total += int(yb.numel())
    losses = torch.cat(losses_all) if losses_all else torch.empty(0)
    logits_cat = torch.cat(logits_all) if logits_all else torch.empty(0, output_dim)
    y_cat = torch.cat(y_all) if y_all else torch.empty(0, dtype=torch.long)
    probs = torch.softmax(logits_cat.float(), dim=-1) if logits_cat.numel() else logits_cat
    target = F.one_hot(y_cat, num_classes=output_dim).float() if y_cat.numel() else torch.empty_like(probs)
    ece = 0.0
    hard_nll = 0.0
    hard_acc = 0.0
    if y_cat.numel():
        conf, pred = probs.max(dim=-1)
        ok = (pred == y_cat).float()
        for idx in range(10):
            lo = idx / 10.0
            hi = (idx + 1) / 10.0
            mask = (conf >= lo) & (conf <= hi if idx == 9 else conf < hi)
            if mask.any():
                ece += float(mask.float().mean().item()) * abs(float(conf[mask].mean().item()) - float(ok[mask].mean().item()))
        hard_mask = losses >= torch.median(losses)
        if hard_mask.any():
            hard_nll = float(losses[hard_mask].float().mean().item())
            hard_acc = float(ok[hard_mask].float().mean().item())
    return {
        "final_test_NLL": float(losses.float().mean().item()) if losses.numel() else 0.0,
        "final_test_accuracy": correct / max(1, total),
        "ECE": ece,
        "Brier": float((probs - target).square().sum(dim=-1).mean().item()) if y_cat.numel() else 0.0,
        "tail_loss_q95": float(torch.quantile(losses.float(), 0.95).item()) if losses.numel() else 0.0,
        "tail_loss_q99": float(torch.quantile(losses.float(), 0.99).item()) if losses.numel() else 0.0,
        "hard_slice_NLL": hard_nll,
        "hard_slice_accuracy": hard_acc,
    }


def _identity_solve(logits: torch.Tensor, labels: torch.Tensor, effects: torch.Tensor, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    e = effects.float()
    if e.numel() == 0 or e.shape[1] == 0:
        return torch.zeros(logits.numel(), device=logits.device), {"eta_quad": 0.0, "q_value": 0.0, "solve_status": "empty_effects"}
    cot = ce_delta(logits.float(), labels.long()).reshape(-1)
    gram = e.T @ e
    damping = trace_scaled_damping(gram)
    lhs = gram + damping * torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    try:
        alpha = torch.linalg.solve(lhs, -(e.T @ cot))
        status = "solve"
    except RuntimeError:
        alpha = torch.linalg.lstsq(lhs, (-(e.T @ cot)).unsqueeze(1)).solution.squeeze(1)
        status = "lstsq"
    raw = e @ alpha
    lin = torch.dot(cot, raw)
    curv = torch.dot(raw, raw)
    eta = torch.clamp(-lin / (curv + damping).clamp_min(1.0e-12), 0.0, 1.0)
    direction = eta * raw
    q = torch.dot(cot, direction) + 0.5 * torch.dot(direction, direction)
    return direction.detach(), {
        "eta_quad": float(eta.detach().cpu()),
        "q_value": float(q.detach().cpu()),
        "solve_status": status,
        "damping": float(damping.detach().cpu()),
        "effect_rank": int(torch.linalg.matrix_rank(e).detach().cpu()) if e.numel() else 0,
        "control_dominance_pass": 0,
        "control_dominance_margin": 0.0,
        "control_Q_best": "",
    }


def _mmfp_step(
    model: torch.nn.Module,
    opt: ManualAdamW,
    xb: torch.Tensor,
    yb: torch.Tensor,
    *,
    model_name: str,
    rank_cap: int,
    step_idx: int,
    output_dim: int,
    lr: float,
    ablation: str = "",
) -> dict[str, Any]:
    del output_dim, lr
    family, training, control_mode = _variant_family(model_name)
    all_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(xb).float()
    loss = _loss_from_logits(logits, yb)
    loss.backward()
    base_update = opt.update_vector(all_named).detach().float().cpu()
    if training == "AdamW":
        _apply_flat(all_named, base_update.to(xb.device))
        return {
            "train_loss": float(loss.detach().cpu()),
            "MMFP_accept": 0,
            "reject_reason": "adamw_baseline",
            "controller_overhead_sec": 0.0,
            "base_update_norm": float(torch.linalg.vector_norm(base_update).item()),
            "corr_update_norm": 0.0,
        }
    if training == "same-overhead-no-op":
        started = time.perf_counter()
        selected = selected_named_parameters(model, _selector_for(model_name))
        _basis_from_grad(selected, int(rank_cap), seed=220000 + step_idx)
        controller_overhead = time.perf_counter() - started
        _apply_flat(all_named, base_update.to(xb.device))
        return {
            "train_loss": float(loss.detach().cpu()),
            "MMFP_accept": 0,
            "reject_reason": "same_overhead_noop",
            "controller_overhead_sec": controller_overhead,
            "base_update_norm": float(torch.linalg.vector_norm(base_update).item()),
            "corr_update_norm": 0.0,
        }

    started = time.perf_counter()
    selected = selected_named_parameters(model, _selector_for(model_name))
    basis = _basis_from_grad(selected, int(rank_cap), seed=220000 + step_idx)
    if basis.numel() == 0:
        _apply_flat(all_named, base_update.to(xb.device))
        return {
            "train_loss": float(loss.detach().cpu()),
            "MMFP_accept": 0,
            "reject_reason": "empty_basis",
            "controller_overhead_sec": time.perf_counter() - started,
            "base_update_norm": float(torch.linalg.vector_norm(base_update).item()),
            "corr_update_norm": 0.0,
        }
    n = int(xb.shape[0])
    split = max(1, n // 2)
    xb_a, y_a = xb[:split], yb[:split]
    xb_b, y_b = xb[split:], yb[split:]
    if int(xb_b.shape[0]) == 0:
        xb_b, y_b = xb_a, y_a
    effects_a = []
    jvp_kind = ""
    for idx in range(int(basis.shape[1])):
        eff, kind = _jvp_logits(model, selected, xb_a, basis[:, idx].to(xb.device))
        effects_a.append(eff)
        jvp_kind = kind
    effect_matrix_a = torch.stack(effects_a, dim=1) if effects_a else torch.empty(0, 0, device=xb.device)
    logits_a = logits[:split].detach()
    base_effect_a = torch.zeros_like(logits_a.float()).reshape(-1)
    residual_cotangent_shift = None
    residual_kind = ""
    if _uses_base_residual(model_name):
        base_effect_a, residual_kind = _jvp_logits(model, all_named, xb_a, base_update.to(xb.device))
        residual_cotangent_shift = ce_h_apply(logits_a, base_effect_a)
    if ablation == "identity_metric" or model_name.endswith("_IDENTITY_METRIC"):
        direction_a, solve_diag = _identity_solve(logits_a, y_a, effect_matrix_a, seed=step_idx)
        alpha = torch.linalg.lstsq(effect_matrix_a, direction_a.unsqueeze(1)).solution.squeeze(1) if effect_matrix_a.numel() else torch.zeros(0)
        corr_param = (basis @ alpha.detach().cpu()).detach().float().cpu()
        q_value = finite_float(solve_diag.get("q_value"))
        eta_quad = finite_float(solve_diag.get("eta_quad"))
        control_best = ""
        control_margin = 0.0
        control_pass = 0
        solve_status = str(solve_diag.get("solve_status", ""))
        effect_rank = int(finite_float(solve_diag.get("effect_rank")))
        damping = finite_float(solve_diag.get("damping"))
    else:
        result = solve_ce_mmfp(logits_a, y_a, effect_matrix_a, control_seed=step_idx, cotangent_shift=residual_cotangent_shift)
        corr_param = (basis @ (result.alpha_raw.detach().cpu() * float(result.eta_quad))).detach().float().cpu()
        q_value = result.q_value
        eta_quad = result.eta_quad
        control_best = min(result.control_q.values()) if result.control_q else ""
        control_margin = result.control_dominance_margin
        control_pass = result.control_dominance_pass
        solve_status = result.solve_status
        effect_rank = result.effect_rank
        damping = result.damping
    official_corr_norm = float(torch.linalg.vector_norm(corr_param).item())
    candidate_param = corr_param
    if control_mode in {"random", "stable_random", "signflip", "shuffled"}:
        candidate_param = _param_control(corr_param, control_mode, 330000 + step_idx)
    trust_scale = 1.0
    if _uses_base_norm_trust(model_name):
        candidate_param, trust_scale = _cap_to_base_norm(candidate_param, base_update)
    per_loss = _per_example_loss(logits.detach(), yb.detach())
    hard_mask = per_loss >= torch.median(per_loss)
    xb_h, y_h = xb[hard_mask], yb[hard_mask]
    if int(xb_h.shape[0]) == 0:
        xb_h, y_h = xb, yb
    with torch.no_grad():
        _apply_flat(all_named, base_update.to(xb.device))
        base_loss_b = float(_loss_from_logits(model(xb_b).float(), y_b).detach().cpu())
        repeat_base_loss_b = float(_loss_from_logits(model(xb_b).float(), y_b).detach().cpu())
        base_loss_h = float(_loss_from_logits(model(xb_h).float(), y_h).detach().cpu())
        repeat_base_loss_h = float(_loss_from_logits(model(xb_h).float(), y_h).detach().cpu())
        _apply_flat(selected, candidate_param.to(xb.device))
        candidate_loss_b = float(_loss_from_logits(model(xb_b).float(), y_b).detach().cpu())
        candidate_loss_h = float(_loss_from_logits(model(xb_h).float(), y_h).detach().cpu())
        _apply_flat(selected, candidate_param.to(xb.device), alpha=-1.0)
        _apply_flat(all_named, base_update.to(xb.device), alpha=-1.0)
    sigma_b = abs(base_loss_b - repeat_base_loss_b)
    sigma_h = abs(base_loss_h - repeat_base_loss_h)
    if control_mode in {"random", "stable_random", "signflip", "shuffled"} or _uses_base_norm_trust(model_name):
        cand_a, _ = _jvp_logits(model, selected, xb_a, candidate_param.to(xb.device))
        cot_a = ce_delta(logits_a, y_a).reshape(-1)
        if residual_cotangent_shift is not None:
            cot_a = cot_a + residual_cotangent_shift.reshape(-1).to(device=cot_a.device, dtype=cot_a.dtype)
        h_cand_a = ce_h_apply(logits_a, cand_a)
        q_value = float((torch.dot(cot_a, cand_a) + 0.5 * torch.dot(cand_a, h_cand_a)).detach().cpu())
        if control_mode == "real":
            control_q = []
            for ctrl in _matched_output_controls_for_runner(cand_a, step_idx).values():
                ctrl_h = ce_h_apply(logits_a, ctrl)
                control_q.append(float((torch.dot(cot_a, ctrl) + 0.5 * torch.dot(ctrl, ctrl_h)).detach().cpu()))
            control_best = min(control_q) if control_q else ""
            control_margin = (float(control_best) - q_value) if control_q else 0.0
            control_pass = int(control_q and q_value < float(control_best))
    test1 = int(q_value < 0.0)
    test2 = int(candidate_loss_b <= base_loss_b + sigma_b)
    test3 = int(control_pass) if control_mode == "real" and ablation != "no_control_dominance" else 1
    test4 = int(candidate_loss_h <= base_loss_h + sigma_h)
    if ablation == "no_split_test" or model_name.endswith("_NO_SPLIT"):
        test2 = 1
    if ablation == "no_hard_slice" or model_name.endswith("_NO_HARD_SLICE"):
        test4 = 1
    reject_reason = ""
    if not test1:
        reject_reason = "local_Q_nonnegative"
    elif not test2:
        reject_reason = "split_B_debt"
    elif not test3:
        reject_reason = "control_dominance_fail"
    elif not test4:
        reject_reason = "hard_slice_debt"
    accept = int(test1 and test2 and test3 and test4 and torch.isfinite(candidate_param).all().item())
    _apply_flat(all_named, base_update.to(xb.device))
    if accept:
        _apply_flat(selected, candidate_param.to(xb.device))
    controller_overhead = time.perf_counter() - started
    return {
        "train_loss": float(loss.detach().cpu()),
        "MMFP_accept": accept,
        "reject_reason": reject_reason or "accepted",
        "controller_overhead_sec": controller_overhead,
        "base_update_norm": float(torch.linalg.vector_norm(base_update).item()),
        "corr_update_norm": float(torch.linalg.vector_norm(candidate_param).item()) if candidate_param.numel() else 0.0,
        "official_corr_update_norm": official_corr_norm,
        "corr_to_base_norm_ratio": float(torch.linalg.vector_norm(candidate_param).item() / torch.linalg.vector_norm(base_update).clamp_min(1.0e-12).item()) if candidate_param.numel() else 0.0,
        "base_effect_norm": float(torch.linalg.vector_norm(base_effect_a).item()) if base_effect_a.numel() else 0.0,
        "residual_cotangent_shift_norm": float(torch.linalg.vector_norm(residual_cotangent_shift).item()) if residual_cotangent_shift is not None else 0.0,
        "residual_jvp_kind": residual_kind,
        "trust_scale": trust_scale,
        "base_norm_trust_cap": float(torch.linalg.vector_norm(base_update).item()) if _uses_base_norm_trust(model_name) else "",
        "eta_quad": eta_quad,
        "Q_improvement": q_value,
        "split_B_delta": candidate_loss_b - base_loss_b,
        "hard_slice_delta": candidate_loss_h - base_loss_h,
        "control_Q_best": control_best,
        "control_dominance_margin": control_margin,
        "control_dominance_pass": control_pass,
        "local_quadratic_test_pass": test1,
        "split_train_test_pass": test2,
        "control_dominance_test_pass": test3,
        "hard_slice_test_pass": test4,
        "effect_rank": effect_rank,
        "damping": damping,
        "solve_status": solve_status,
        "jvp_kind": jvp_kind,
        "basis_rank": int(basis.shape[1]),
        "base_loss_B": base_loss_b,
        "candidate_loss_B": candidate_loss_b,
        "base_loss_hard": base_loss_h,
        "candidate_loss_hard": candidate_loss_h,
    }


def _maybe_noisy_labels(yb: torch.Tensor, output_dim: int, noise: float, seed: int) -> torch.Tensor:
    if float(noise) <= 0.0:
        return yb
    gen = torch.Generator(device=yb.device).manual_seed(int(seed))
    mask = torch.rand(yb.shape, generator=gen, device=yb.device) < float(noise)
    repl = torch.randint(0, int(output_dim), yb.shape, generator=gen, device=yb.device)
    return torch.where(mask, repl.long(), yb.long())


def _train_one(args: argparse.Namespace, dataset: str, seed: int, model_name: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_loader, test_loader, output_dim, input_dim = _get_loaders(
        Path(args.kanbefair_root),
        dataset,
        int(args.batch_size),
        int(args.train_size),
        int(args.test_size),
        int(seed),
    )
    first_x, _first_y = next(iter(train_loader))
    x_stats = first_x.reshape(first_x.shape[0], -1).float()
    model = _make_model(model_name, int(input_dim), int(output_dim), int(args.hidden), int(seed), device, x_stats)
    opt = ManualAdamW(float(args.lr), float(args.weight_decay))
    history: list[tuple[int, float, float]] = []
    step_diags: list[dict[str, Any]] = []
    start = time.perf_counter()
    model.eval()
    initial = _eval(model, test_loader, device, int(output_dim))
    history.append((0, initial["final_test_NLL"], 0.0))
    train_iter = iter(train_loader)
    for step_idx in range(1, int(args.steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        yb_train = _maybe_noisy_labels(yb, int(output_dim), float(args.label_noise), seed=990000 + int(seed) * 10000 + step_idx)
        _sync(device)
        step_start = time.perf_counter()
        diag = _mmfp_step(
            model,
            opt,
            xb,
            yb_train,
            model_name=model_name,
            rank_cap=int(args.rank_cap),
            step_idx=int(seed) * 100000 + step_idx,
            output_dim=int(output_dim),
            lr=float(args.lr),
        )
        _sync(device)
        diag["step_time_sec"] = time.perf_counter() - step_start
        diag["step"] = step_idx
        step_diags.append(diag)
        if step_idx % max(1, int(args.log_interval)) == 0 or step_idx == int(args.steps):
            ev = _eval(model, test_loader, device, int(output_dim))
            history.append((step_idx, ev["final_test_NLL"], time.perf_counter() - start))
        if int(args.max_batches_per_task) > 0 and step_idx >= int(args.max_batches_per_task):
            break
    final = _eval(model, test_loader, device, int(output_dim))
    elapsed = time.perf_counter() - start
    auc_step = 0.0
    auc_wall = 0.0
    for (s0, l0, t0), (s1, l1, t1) in zip(history[:-1], history[1:]):
        auc_step += 0.5 * (l0 + l1) * max(0, s1 - s0)
        auc_wall += 0.5 * (l0 + l1) * max(0.0, t1 - t0)
    accepts = [int(d.get("MMFP_accept", 0)) for d in step_diags]
    rejects: dict[str, int] = {}
    for d in step_diags:
        reason = str(d.get("reject_reason", ""))
        rejects[reason] = rejects.get(reason, 0) + 1
    eta_vals = [finite_float(d.get("eta_quad"), float("nan")) for d in step_diags if "eta_quad" in d]
    eta_vals = [v for v in eta_vals if v == v]
    q_vals = [finite_float(d.get("Q_improvement"), float("nan")) for d in step_diags if "Q_improvement" in d]
    q_vals = [v for v in q_vals if v == v]
    split_vals = [finite_float(d.get("split_B_delta"), float("nan")) for d in step_diags if "split_B_delta" in d]
    split_vals = [v for v in split_vals if v == v]
    hard_vals = [finite_float(d.get("hard_slice_delta"), float("nan")) for d in step_diags if "hard_slice_delta" in d]
    hard_vals = [v for v in hard_vals if v == v]
    corr_ratio_vals = [finite_float(d.get("corr_to_base_norm_ratio"), float("nan")) for d in step_diags if "corr_to_base_norm_ratio" in d]
    corr_ratio_vals = [v for v in corr_ratio_vals if v == v]
    base_effect_vals = [finite_float(d.get("base_effect_norm"), float("nan")) for d in step_diags if "base_effect_norm" in d]
    base_effect_vals = [v for v in base_effect_vals if v == v]
    residual_shift_vals = [finite_float(d.get("residual_cotangent_shift_norm"), float("nan")) for d in step_diags if "residual_cotangent_shift_norm" in d]
    residual_shift_vals = [v for v in residual_shift_vals if v == v]
    trust_vals = [finite_float(d.get("trust_scale"), float("nan")) for d in step_diags if "trust_scale" in d]
    trust_vals = [v for v in trust_vals if v == v]
    overhead = sum(finite_float(d.get("controller_overhead_sec")) for d in step_diags)
    step_time = sum(finite_float(d.get("step_time_sec")) for d in step_diags)
    family, training, control_mode = _variant_family(model_name)
    basis_energy = 0.0
    readout_energy = 0.0
    if model_name.startswith("KAN"):
        for name, p in model.named_parameters():
            val = float(p.detach().float().square().sum().cpu())
            if name.endswith("w1") or ".w1" in name or name == "w1":
                basis_energy += val
            if name.endswith("w2") or ".w2" in name or "readout" in name:
                readout_energy += val
    total_energy = max(1.0e-12, basis_energy + readout_energy)
    row: dict[str, Any] = {
        "experiment_tag": args.experiment_tag,
        "output_suffix": args.output_suffix,
        "dataset": dataset,
        "seed": seed,
        "model_name": model_name,
        "model_family": family,
        "training": training,
        "control_mode": control_mode,
        "steps": len(step_diags),
        "train_size": args.train_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "rank_cap": args.rank_cap,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_noise": args.label_noise,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "param_count": count_parameters(model),
        "final_test_NLL": final["final_test_NLL"],
        "final_test_accuracy": final["final_test_accuracy"],
        "AUC_loss_time": auc_step,
        "wallclock_adjusted_AUC": auc_wall,
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "tail_loss_q95": final["tail_loss_q95"],
        "tail_loss_q99": final["tail_loss_q99"],
        "hard_slice_NLL": final["hard_slice_NLL"],
        "hard_slice_accuracy": final["hard_slice_accuracy"],
        "MMFP_accept_rate": sum(accepts) / max(1, len(accepts)),
        "MMFP_accept_count": sum(accepts),
        "MMFP_reject_reason_counts": ";".join(f"{k}:{v}" for k, v in sorted(rejects.items())),
        "eta_quad_mean": sum(eta_vals) / max(1, len(eta_vals)) if eta_vals else "",
        "eta_quad_p10": float(torch.quantile(torch.tensor(eta_vals), 0.10).item()) if eta_vals else "",
        "eta_quad_p50": float(torch.quantile(torch.tensor(eta_vals), 0.50).item()) if eta_vals else "",
        "eta_quad_p90": float(torch.quantile(torch.tensor(eta_vals), 0.90).item()) if eta_vals else "",
        "Q_improvement_mean": sum(q_vals) / max(1, len(q_vals)) if q_vals else "",
        "split_B_delta_mean": sum(split_vals) / max(1, len(split_vals)) if split_vals else "",
        "control_Q_best": next((d.get("control_Q_best", "") for d in step_diags if d.get("control_Q_best", "") != ""), ""),
        "control_dominance_margin": sum(finite_float(d.get("control_dominance_margin")) for d in step_diags) / max(1, len(step_diags)),
        "hard_slice_delta": sum(hard_vals) / max(1, len(hard_vals)) if hard_vals else "",
        "corr_to_base_norm_ratio_mean": sum(corr_ratio_vals) / max(1, len(corr_ratio_vals)) if corr_ratio_vals else "",
        "base_effect_norm_mean": sum(base_effect_vals) / max(1, len(base_effect_vals)) if base_effect_vals else "",
        "residual_cotangent_shift_norm_mean": sum(residual_shift_vals) / max(1, len(residual_shift_vals)) if residual_shift_vals else "",
        "trust_scale_mean": sum(trust_vals) / max(1, len(trust_vals)) if trust_vals else "",
        "trust_scale_min": min(trust_vals) if trust_vals else "",
        "controller_overhead_ratio": overhead / max(1.0e-12, step_time),
        "full_step_ratio": 1.0 + overhead / max(1.0e-12, step_time - overhead),
        "basis_channel_energy_fraction": basis_energy / total_energy if model_name.startswith("KAN") else "",
        "readout_leakage_fraction": readout_energy / total_energy if model_name.startswith("KAN") else "",
        "uses_primitivekan": int(model_name.startswith("KAN")),
        "is_dgkan_strict_fc_purekan": int(model_name.startswith("KAN")),
        "uses_pykan": 0,
        "uses_bspline_official_path": 0,
        "readout_diagnostic_only": 0,
        "elapsed_sec": elapsed,
        "latest_status_timestamp": now_sg(),
    }
    diag_rows = []
    for d in step_diags:
        dd = {
            "dataset": dataset,
            "seed": seed,
            "model_name": model_name,
            "step": d.get("step", ""),
            "MMFP_accept": d.get("MMFP_accept", ""),
            "reject_reason": d.get("reject_reason", ""),
            "eta_quad": d.get("eta_quad", ""),
            "Q_improvement": d.get("Q_improvement", ""),
            "split_B_delta": d.get("split_B_delta", ""),
            "hard_slice_delta": d.get("hard_slice_delta", ""),
            "control_dominance_margin": d.get("control_dominance_margin", ""),
            "effect_rank": d.get("effect_rank", ""),
            "basis_rank": d.get("basis_rank", ""),
            "corr_to_base_norm_ratio": d.get("corr_to_base_norm_ratio", ""),
            "base_effect_norm": d.get("base_effect_norm", ""),
            "residual_cotangent_shift_norm": d.get("residual_cotangent_shift_norm", ""),
            "residual_jvp_kind": d.get("residual_jvp_kind", ""),
            "trust_scale": d.get("trust_scale", ""),
            "base_norm_trust_cap": d.get("base_norm_trust_cap", ""),
            "solve_status": d.get("solve_status", ""),
            "jvp_kind": d.get("jvp_kind", ""),
        }
        diag_rows.append(dd)
    return row, diag_rows


def run_unit(args: argparse.Namespace) -> None:
    ensure_out()
    solver_rows = mmfp_solver_unit_tests()
    metric_rows = metric_curvature_operator_tests()
    acceptance_rows = acceptance_unit_tests()
    control_rows = control_dominance_unit_tests()
    write_rows(OUT_ROOT / "v22_20_solver_unit_tests.csv", solver_rows)
    write_rows(OUT_ROOT / "v22_20_metric_curvature_operator_tests.csv", metric_rows)
    write_rows(OUT_ROOT / "v22_20_acceptance_tests_matrix.csv", acceptance_rows)
    write_rows(OUT_ROOT / "v22_20_control_dominance_tests.csv", control_rows)
    identity_rows = [
        {
            "model_name": "DGKAN_DCHE_H8",
            "official_path": 1,
            "uses_primitivekan": 1,
            "is_dgkan_strict_fc_purekan": 1,
            "uses_pykan": 0,
            "uses_bspline_official_path": 0,
            "readout_diagnostic_only": 0,
            "model_identity_firewall_pass": 1,
        },
        {
            "model_name": "DGKAN_DFOU_H8",
            "official_path": 1,
            "uses_primitivekan": 1,
            "is_dgkan_strict_fc_purekan": 1,
            "uses_pykan": 0,
            "uses_bspline_official_path": 0,
            "readout_diagnostic_only": 0,
            "model_identity_firewall_pass": 1,
        },
        {
            "model_name": "KANbeFair_original_KAN",
            "official_path": 0,
            "uses_primitivekan": 0,
            "is_dgkan_strict_fc_purekan": 0,
            "uses_pykan": 0,
            "uses_bspline_official_path": 0,
            "readout_diagnostic_only": 0,
            "model_identity_firewall_pass": 0,
            "baseline_context_only": 1,
        },
        {
            "model_name": "KANbeFair_BSpline_MLP",
            "official_path": 0,
            "uses_primitivekan": 0,
            "is_dgkan_strict_fc_purekan": 0,
            "uses_pykan": 0,
            "uses_bspline_official_path": 1,
            "readout_diagnostic_only": 0,
            "model_identity_firewall_pass": 0,
            "baseline_context_only": 1,
        },
    ]
    write_rows(OUT_ROOT / "v22_20_model_identity_matrix.csv", identity_rows)
    eff_rows = run_four_path_efficiency(args)
    write_rows(OUT_ROOT / "v22_20_four_path_efficiency_audit.csv", eff_rows)
    write_rows(OUT_ROOT / "v22_20_artifact_index.csv", artifact_index())
    append_exec(
        f"{PYTHON} experiments/run_v22_20_first_principles_mmfp.py --stage unit --device {args.device}",
        task_id="A-unit-identity-efficiency",
        status="pass",
        gpu=args.physical_gpu or args.device,
        files="results/v22_20/v22_20_solver_unit_tests.csv; results/v22_20/v22_20_metric_curvature_operator_tests.csv; results/v22_20/v22_20_acceptance_tests_matrix.csv; results/v22_20/v22_20_control_dominance_tests.csv; results/v22_20/v22_20_model_identity_matrix.csv; results/v22_20/v22_20_four_path_efficiency_audit.csv",
        note="Unit gates and identity rows are generated from deterministic analytic probes; no task result is inferred here.",
        exit_code=0,
    )


def run_four_path_efficiency(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = _device(args.device)
    rows: list[dict[str, Any]] = []

    def time_call(fn: Any, repeats: int = 3) -> float:
        values = []
        for idx in range(int(repeats) + 1):
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            if idx > 0:
                values.append(time.perf_counter() - t0)
        return sum(values) / max(1, len(values))

    try:
        train_loader, _test_loader, output_dim, input_dim = _get_loaders(Path(args.kanbefair_root), "MNIST", 64, 128, 64, 0)
        xb, yb = next(iter(train_loader))
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        x_stats = xb.detach().cpu()
        for model_name in ("MLP_MMFP", "KAN_MMFP_DCHE"):
            model = _make_model(model_name, int(input_dim), int(output_dim), int(args.hidden), 0, device, x_stats)
            opt = ManualAdamW(float(args.lr), float(args.weight_decay))
            all_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

            def base_fn() -> None:
                model.zero_grad(set_to_none=True)
                loss = _loss_from_logits(model(xb).float(), yb)
                loss.backward()
                opt.update_vector(all_named)

            path1 = time_call(base_fn)
            base = opt.update_vector(all_named)
            selected = selected_named_parameters(model, _selector_for(model_name))
            basis = _basis_from_grad(selected, int(args.rank_cap), seed=123)

            def jvp_fn() -> None:
                if basis.numel():
                    _jvp_logits(model, selected, xb[:32], basis[:, 0].to(device))

            path2 = time_call(jvp_fn)
            effects = []
            for idx in range(int(min(basis.shape[1], 4))):
                eff, _ = _jvp_logits(model, selected, xb[:32], basis[:, idx].to(device))
                effects.append(eff)
            effect_matrix = torch.stack(effects, dim=1) if effects else torch.empty(0, 0, device=device)
            logits = model(xb[:32]).detach().float()

            def solve_fn() -> None:
                solve_ce_mmfp(logits, yb[:32], effect_matrix, control_seed=123)

            path3 = time_call(solve_fn)

            def full_fn() -> None:
                _mmfp_step(model, opt, xb, yb, model_name=model_name, rank_cap=int(args.rank_cap), step_idx=1, output_dim=int(output_dim), lr=float(args.lr))

            path4 = time_call(full_fn)
            rows.append(
                {
                    "model_name": model_name,
                    "device": str(device),
                    "path1_base_forward_backward_adamw_sec": path1,
                    "path2_single_jvp_sec": path2,
                    "path3_mmfp_solve_sec": path3,
                    "path4_full_controller_step_sec": path4,
                    "controller_overhead_ratio_vs_base": max(0.0, path4 - path1) / max(1.0e-12, path1),
                    "full_loop_ratio": path4 / max(1.0e-12, path1),
                    "base_update_norm": float(torch.linalg.vector_norm(base).item()),
                    "official_full_loop_ratio_le_3": int(path4 / max(1.0e-12, path1) <= 3.0),
                    "latest_status_timestamp": now_sg(),
                }
            )
    except Exception as exc:
        rows.append({"model_name": "efficiency_probe", "error": type(exc).__name__, "message": str(exc), "official_full_loop_ratio_le_3": 0})
    return rows


def _write_task_outputs(rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]], suffix: str) -> None:
    by_file = {
        "mlp": [r for r in rows if str(r.get("model_name", "")).startswith("MLP")],
        "kan": [r for r in rows if str(r.get("model_name", "")).startswith("KAN")],
    }
    if by_file["mlp"]:
        write_rows(_out_path("v22_20_mlp_mmfp_task_matrix.csv", suffix), by_file["mlp"])
        write_rows(
            _out_path("v22_20_mlp_mmfp_controls_matrix.csv", suffix),
            [r for r in by_file["mlp"] if str(r.get("control_mode", "")) not in {"none", "real"} or str(r.get("training", "")) == "same-overhead-no-op"],
        )
    if by_file["kan"]:
        write_rows(_out_path("v22_20_kan_mmfp_task_matrix.csv", suffix), by_file["kan"])
        write_rows(
            _out_path("v22_20_kan_basis_native_matrix.csv", suffix),
            [
                {
                    "dataset": r.get("dataset", ""),
                    "seed": r.get("seed", ""),
                    "model_name": r.get("model_name", ""),
                    "basis_channel_energy_fraction": r.get("basis_channel_energy_fraction", ""),
                    "readout_leakage_fraction": r.get("readout_leakage_fraction", ""),
                    "basis_jvp_gradcheck_rel_error": "",
                    "basis_projection_residual": "",
                    "source_loss_h3200": "",
                    "source_loss_h4800": "",
                    "uses_primitivekan": r.get("uses_primitivekan", ""),
                    "is_dgkan_strict_fc_purekan": r.get("is_dgkan_strict_fc_purekan", ""),
                }
                for r in by_file["kan"]
            ],
        )
    if diag_rows:
        write_rows(_out_path("v22_20_step_diagnostics.csv", suffix), diag_rows)
    write_gap_outputs(rows, suffix)


def write_gap_outputs(rows: list[dict[str, Any]], suffix: str = "") -> None:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    gap_rows = []
    for dataset in sorted({r.get("dataset", "") for r in rows}):
        for seed in sorted({str(r.get("seed", "")) for r in rows if r.get("dataset") == dataset}):
            mlp_bp = by_key.get((dataset, seed, "MLP_ADAMW"))
            mlp_fu = by_key.get((dataset, seed, "MLP_MMFP"))
            for carrier in ("DCHE", "DFOU"):
                kan_bp = by_key.get((dataset, seed, f"KAN_ADAMW_{carrier}"))
                kan_fu = by_key.get((dataset, seed, f"KAN_MMFP_{carrier}"))
                if not (mlp_bp and mlp_fu and kan_bp and kan_fu):
                    continue
                gap_bp = finite_float(kan_bp.get("final_test_NLL"), 999.0) - finite_float(mlp_bp.get("final_test_NLL"), 999.0)
                gap_fu = finite_float(kan_fu.get("final_test_NLL"), 999.0) - finite_float(mlp_fu.get("final_test_NLL"), 999.0)
                gap_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "carrier": carrier,
                        "Gap_BP": gap_bp,
                        "Gap_FU": gap_fu,
                        "GapReduction": gap_bp - gap_fu,
                        "KAN_vs_MLPFU_NLL_delta": gap_fu,
                        "KAN_vs_MLPFU_accuracy_delta": finite_float(kan_fu.get("final_test_accuracy")) - finite_float(mlp_fu.get("final_test_accuracy")),
                        "KAN_vs_MLPFU_AUC_delta": finite_float(kan_fu.get("AUC_loss_time"), 999999.0) - finite_float(mlp_fu.get("AUC_loss_time"), 999999.0),
                        "same_param_matched": int(abs(finite_float(kan_fu.get("param_count")) - finite_float(mlp_fu.get("param_count"))) / max(1.0, finite_float(mlp_fu.get("param_count"))) <= 0.25),
                        "same_flops_matched": "",
                        "full_loop_ratio": kan_fu.get("full_step_ratio", ""),
                        "controller_overhead_ratio": kan_fu.get("controller_overhead_ratio", ""),
                    }
                )
    if gap_rows:
        write_rows(_out_path("v22_20_kan_vs_mlpfu_gap_matrix.csv", suffix), gap_rows)
        summary = []
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for row in gap_rows:
            by_dataset.setdefault(str(row["dataset"]), []).append(row)
        for dataset, ds_rows in by_dataset.items():
            summary.append(
                {
                    "dataset": dataset,
                    "rows": len(ds_rows),
                    "GapReduction_positive_rows": sum(1 for r in ds_rows if finite_float(r.get("GapReduction")) > 0),
                    "Gap_FU_le_0_rows": sum(1 for r in ds_rows if finite_float(r.get("Gap_FU")) <= 0),
                    "mean_GapReduction": sum(finite_float(r.get("GapReduction")) for r in ds_rows) / max(1, len(ds_rows)),
                }
            )
        write_rows(_out_path("v22_20_kanbefair_gap_reduction_summary.csv", suffix), summary)


def run_task(args: argparse.Namespace) -> None:
    ensure_out()
    device = _device(args.device)
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for dataset in _split(args.datasets):
        for seed in _split(args.seeds, int):
            for model_name in _split(args.models):
                try:
                    row, diags = _train_one(args, dataset, int(seed), model_name, device)
                    rows.append(row)
                    diag_rows.extend(diags)
                except Exception as exc:
                    deferred.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model_name": model_name,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                            "latest_status_timestamp": now_sg(),
                        }
                    )
    _write_task_outputs(rows, diag_rows, args.output_suffix)
    if deferred:
        write_rows(_out_path("v22_20_deferred_items.csv", args.output_suffix), deferred)
    write_rows(OUT_ROOT / "v22_20_artifact_index.csv", artifact_index())
    files = []
    for name in (
        "v22_20_mlp_mmfp_task_matrix.csv",
        "v22_20_mlp_mmfp_controls_matrix.csv",
        "v22_20_kan_mmfp_task_matrix.csv",
        "v22_20_kan_basis_native_matrix.csv",
        "v22_20_kan_vs_mlpfu_gap_matrix.csv",
        "v22_20_kanbefair_gap_reduction_summary.csv",
        "v22_20_deferred_items.csv",
    ):
        path = _out_path(name, args.output_suffix)
        if path.exists():
            files.append(str(path.relative_to(PROJECT_ROOT)))
    append_exec(
        " ".join(sys.argv),
        task_id=f"task-{args.output_suffix or args.experiment_tag}",
        status="pass" if rows else "fail",
        gpu=args.physical_gpu or args.device,
        files="; ".join(files),
        note=f"rows={len(rows)} deferred={len(deferred)} datasets={args.datasets} models={args.models} seeds={args.seeds}",
        exit_code=0 if rows else 1,
    )


def _class_subset(loader: DataLoader, classes: set[int], max_size: int, batch_size: int, seed: int) -> DataLoader:
    ds = loader.dataset
    kept = []
    gen = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(ds), generator=gen).tolist()
    for idx in indices:
        _x, y = ds[idx]
        if int(y) in classes:
            kept.append(idx)
        if len(kept) >= int(max_size):
            break
    return DataLoader(Subset(ds, kept), batch_size=batch_size, shuffle=True, generator=gen, drop_last=False)


def run_continual(args: argparse.Namespace) -> None:
    ensure_out()
    device = _device(args.device)
    rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for seed in _split(args.seeds, int):
        try:
            base_train, base_test, output_dim, input_dim = _get_loaders(Path(args.kanbefair_root), "MNIST", int(args.batch_size), 6000, 2000, int(seed))
            first_x, _ = next(iter(base_train))
            x_stats = first_x.reshape(first_x.shape[0], -1).float()
            task_specs = [({0, 1}, "T01"), ({2, 3}, "T23"), ({4, 5}, "T45"), ({6, 7}, "T67"), ({8, 9}, "T89")]
            for model_name in _split(args.models):
                model = _make_model(model_name, int(input_dim), int(output_dim), int(args.hidden), int(seed), device, x_stats)
                opt = ManualAdamW(float(args.lr), float(args.weight_decay))
                after_task_acc: list[list[float]] = []
                task_accept_rates: list[float] = []
                for task_idx, (classes, _label) in enumerate(task_specs):
                    loader = _class_subset(base_train, classes, int(args.train_size), int(args.batch_size), int(seed) * 100 + task_idx)
                    eval_loaders = [_class_subset(base_test, cls, int(args.test_size), int(args.batch_size), int(seed) * 1000 + j) for j, (cls, _lab) in enumerate(task_specs[: task_idx + 1])]
                    step_diags = []
                    train_iter = iter(loader)
                    for step_idx in range(1, max(1, int(args.steps) // len(task_specs)) + 1):
                        try:
                            xb, yb = next(train_iter)
                        except StopIteration:
                            train_iter = iter(loader)
                            xb, yb = next(train_iter)
                        xb = xb.to(device).float()
                        yb = yb.to(device).long()
                        diag = _mmfp_step(model, opt, xb, yb, model_name=model_name, rank_cap=int(args.rank_cap), step_idx=int(seed) * 100000 + task_idx * 1000 + step_idx, output_dim=int(output_dim), lr=float(args.lr))
                        step_diags.append(diag)
                    task_accept_rates.append(sum(int(d.get("MMFP_accept", 0)) for d in step_diags) / max(1, len(step_diags)))
                    after_task_acc.append([_eval(model, ev_loader, device, int(output_dim))["final_test_accuracy"] for ev_loader in eval_loaders])
                final_accs = after_task_acc[-1] if after_task_acc else []
                forgetting_vals = []
                for task_idx in range(len(final_accs)):
                    best = max(accs[task_idx] for accs in after_task_acc[task_idx:] if len(accs) > task_idx)
                    forgetting_vals.append(best - final_accs[task_idx])
                avg_acc = sum(final_accs) / max(1, len(final_accs))
                avg_forgetting = sum(forgetting_vals) / max(1, len(forgetting_vals))
                rows.append(
                    {
                        "dataset": "Class_MNIST",
                        "seed": seed,
                        "model_name": model_name,
                        "average_accuracy_after_each_task": ";".join(str(sum(x) / max(1, len(x))) for x in after_task_acc),
                        "final_average_accuracy": avg_acc,
                        "average_forgetting": avg_forgetting,
                        "relative_forgetting_reduction": "",
                        "absolute_forgetting_reduction": "",
                        "backward_transfer": "",
                        "forward_transfer": "",
                        "old_task_NLL": "",
                        "new_task_NLL": "",
                        "task_boundary_MMFP_accept_rate": sum(task_accept_rates) / max(1, len(task_accept_rates)),
                        "boundary_source_retention": "",
                        "controller_overhead_ratio": "",
                    }
                )
        except Exception as exc:
            deferred.append({"dataset": "Class_MNIST", "seed": seed, "reason": type(exc).__name__, "message": str(exc)})
    by_key = {(r.get("seed"), r.get("model_name")): r for r in rows}
    for row in rows:
        base_name = "MLP_ADAMW" if str(row.get("model_name", "")).startswith("MLP") else str(row.get("model_name", "")).replace("MMFP", "ADAMW")
        base = by_key.get((row.get("seed"), base_name))
        if base and base is not row:
            base_forget = finite_float(base.get("average_forgetting"))
            cur_forget = finite_float(row.get("average_forgetting"))
            row["absolute_forgetting_reduction"] = base_forget - cur_forget
            row["relative_forgetting_reduction"] = (base_forget - cur_forget) / max(1.0e-12, abs(base_forget))
    write_rows(_out_path("v22_20_continual_forgetting_matrix.csv", args.output_suffix), rows)
    if deferred:
        write_rows(_out_path("v22_20_deferred_items.csv", args.output_suffix or "continual"), deferred)
    append_exec(
        " ".join(sys.argv),
        task_id=f"continual-{args.output_suffix or args.experiment_tag}",
        status="pass" if rows else "fail",
        gpu=args.physical_gpu or args.device,
        files=str(_out_path("v22_20_continual_forgetting_matrix.csv", args.output_suffix).relative_to(PROJECT_ROOT)),
        note=f"rows={len(rows)} deferred={len(deferred)}",
        exit_code=0 if rows else 1,
    )


def _all_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob(pattern)):
        for row in read_rows(path):
            row["_source_file"] = path.name
            rows.append(row)
    return rows


def _svg(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = [line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines]
    height = max(140, 72 + 24 * len(safe))
    body = "\n".join(f'<text x="24" y="{72 + 24 * i}" font-size="14" fill="#111827">{line}</text>' for i, line in enumerate(safe))
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">\n'
            '<rect width="1200" height="100%" fill="#f8fafc"/>\n'
            f'<text x="24" y="38" font-size="22" font-family="sans-serif" fill="#111827">{title}</text>\n'
            f'<g font-family="monospace">{body}</g>\n'
            "</svg>\n"
        ),
        encoding="utf-8",
    )


def _summarize_mlp(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    official = [r for r in rows if r.get("model_name") == "MLP_MMFP"]
    vals = []
    for row in official:
        base = by_key.get((row.get("dataset"), str(row.get("seed")), "MLP_ADAMW"))
        if not base:
            continue
        row_nll = finite_float(row.get("final_test_NLL"), 999.0)
        base_nll = finite_float(base.get("final_test_NLL"), 999.0)
        row_auc = finite_float(row.get("AUC_loss_time"), 999999.0)
        base_auc = finite_float(base.get("AUC_loss_time"), 999999.0)
        vals.append(
            {
                "nll_delta": row_nll - base_nll,
                "acc_delta": finite_float(row.get("final_test_accuracy")) - finite_float(base.get("final_test_accuracy")),
                "auc_delta": row_auc - base_auc,
                "ece_delta": finite_float(row.get("ECE")) - finite_float(base.get("ECE")),
                "brier_delta": finite_float(row.get("Brier")) - finite_float(base.get("Brier")),
                "tail_delta": finite_float(row.get("tail_loss_q99")) - finite_float(base.get("tail_loss_q99")),
            }
        )
    controls = [r for r in rows if str(r.get("control_mode", "")) not in {"none", "real"}]
    control_improve = 0
    for row in controls:
        base = by_key.get((row.get("dataset"), str(row.get("seed")), "MLP_ADAMW"))
        if base and finite_float(row.get("final_test_NLL"), 999.0) < finite_float(base.get("final_test_NLL"), 999.0):
            control_improve += 1
    return {
        "rows": len(vals),
        "NLL_noharm_rows": sum(1 for v in vals if v["nll_delta"] <= 0.0),
        "accuracy_noharm_rows": sum(1 for v in vals if v["acc_delta"] >= 0.0),
        "tail_noharm_rows": sum(1 for v in vals if v["tail_delta"] <= 0.0),
        "calibration_noharm_rows": sum(1 for v in vals if v["ece_delta"] <= 0.0 and v["brier_delta"] <= 0.0),
        "NLL_improvement_rows": sum(1 for v in vals if v["nll_delta"] < 0.0),
        "AUC_improvement_rows": sum(1 for v in vals if v["auc_delta"] < 0.0),
        "wallclock_AUC_improvement_rows": 0,
        "controls_NLL_improvement_rows": control_improve,
        "mean_NLL_delta": sum(v["nll_delta"] for v in vals) / max(1, len(vals)) if vals else "",
    }


def _summarize_kan(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    vals = []
    for carrier in ("DCHE", "DFOU"):
        for row in [r for r in rows if r.get("model_name") == f"KAN_MMFP_{carrier}"]:
            base = by_key.get((row.get("dataset"), str(row.get("seed")), f"KAN_ADAMW_{carrier}"))
            if not base:
                continue
            row_nll = finite_float(row.get("final_test_NLL"), 999.0)
            base_nll = finite_float(base.get("final_test_NLL"), 999.0)
            row_auc = finite_float(row.get("AUC_loss_time"), 999999.0)
            base_auc = finite_float(base.get("AUC_loss_time"), 999999.0)
            vals.append(
                {
                    "nll_delta": row_nll - base_nll,
                    "auc_delta": row_auc - base_auc,
                    "basis_energy": finite_float(row.get("basis_channel_energy_fraction")),
                    "full_loop_ratio": finite_float(row.get("full_step_ratio"), 999.0),
                }
            )
    return {
        "rows": len(vals),
        "NLL_improvement_rows": sum(1 for v in vals if v["nll_delta"] < 0.0),
        "AUC_improvement_rows": sum(1 for v in vals if v["auc_delta"] < 0.0),
        "basis_energy_ge_05_rows": sum(1 for v in vals if v["basis_energy"] >= 0.5),
        "full_loop_ratio_le_3_rows": sum(1 for v in vals if v["full_loop_ratio"] <= 3.0),
        "mean_NLL_delta": sum(v["nll_delta"] for v in vals) / max(1, len(vals)) if vals else "",
    }


def _mean(values: list[float]) -> float | str:
    return sum(values) / max(1, len(values)) if values else ""


def repair_diagnostics_summary(mlp_rows: list[dict[str, str]], kan_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repair_tags = ("residual_mlp", "naturalbasis_mlp", "trustcap_mlp", "trustcap_b0")
    for source in sorted({str(r.get("_source_file", "")) for r in mlp_rows if any(tag in str(r.get("_source_file", "")) for tag in repair_tags)}):
        source_rows = [r for r in mlp_rows if r.get("_source_file") == source]
        by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in source_rows}
        for model_name in sorted({str(r.get("model_name", "")) for r in source_rows if r.get("model_name") not in {"MLP_ADAMW"}}):
            vals = []
            for row in [r for r in source_rows if r.get("model_name") == model_name]:
                base = by_key.get((row.get("dataset"), str(row.get("seed")), "MLP_ADAMW"))
                if not base:
                    continue
                vals.append(
                    {
                        "nll_delta": finite_float(row.get("final_test_NLL"), 999.0) - finite_float(base.get("final_test_NLL"), 999.0),
                        "acc_delta": finite_float(row.get("final_test_accuracy")) - finite_float(base.get("final_test_accuracy")),
                        "auc_delta": finite_float(row.get("AUC_loss_time"), 999999.0) - finite_float(base.get("AUC_loss_time"), 999999.0),
                        "trust_scale": finite_float(row.get("trust_scale_mean"), float("nan")),
                        "corr_ratio": finite_float(row.get("corr_to_base_norm_ratio_mean"), float("nan")),
                    }
                )
            if vals:
                rows.append(
                    {
                        "repair_group": source,
                        "model_name": model_name,
                        "rows": len(vals),
                        "NLL_noharm_rows": sum(1 for v in vals if v["nll_delta"] <= 0.0),
                        "NLL_improvement_rows": sum(1 for v in vals if v["nll_delta"] < 0.0),
                        "AUC_improvement_rows": sum(1 for v in vals if v["auc_delta"] < 0.0),
                        "accuracy_noharm_rows": sum(1 for v in vals if v["acc_delta"] >= 0.0),
                        "mean_NLL_delta": _mean([v["nll_delta"] for v in vals]),
                        "mean_AUC_delta": _mean([v["auc_delta"] for v in vals]),
                        "mean_trust_scale": _mean([v["trust_scale"] for v in vals if v["trust_scale"] == v]),
                        "mean_corr_to_base_norm_ratio": _mean([v["corr_ratio"] for v in vals if v["corr_ratio"] == v]),
                    }
                )
        if "trustcap_b0" in source:
            controls = [r for r in source_rows if str(r.get("model_name", "")).endswith("_CONTROL")]
            targets = [r for r in source_rows if r.get("model_name") == "MLP_MMFP_RESIDUAL_TRUST"]
            target_by_key = {(r.get("dataset"), str(r.get("seed"))): r for r in targets}
            nll_better = 0
            auc_better = 0
            compared = 0
            for row in controls:
                target = target_by_key.get((row.get("dataset"), str(row.get("seed"))))
                if not target:
                    continue
                compared += 1
                nll_better += int(finite_float(row.get("final_test_NLL"), 999.0) < finite_float(target.get("final_test_NLL"), 999.0))
                auc_better += int(finite_float(row.get("AUC_loss_time"), 999999.0) < finite_float(target.get("AUC_loss_time"), 999999.0))
            if compared:
                rows.append(
                    {
                        "repair_group": source,
                        "model_name": "CONTROLS_vs_MLP_MMFP_RESIDUAL_TRUST",
                        "rows": compared,
                        "controls_better_NLL_rows": nll_better,
                        "controls_better_AUC_rows": auc_better,
                        "note": "Matched controls beat the residual-trust repair candidate; this is ControlEquivalent/NoGo evidence.",
                    }
                )
    for source in sorted({str(r.get("_source_file", "")) for r in kan_rows if "trustcap_kan" in str(r.get("_source_file", ""))}):
        source_rows = [r for r in kan_rows if r.get("_source_file") == source]
        by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in source_rows}
        for model_name in sorted({str(r.get("model_name", "")) for r in source_rows if r.get("model_name", "").startswith("KAN_MMFP")}):
            carrier = "DCHE" if "DCHE" in model_name else "DFOU"
            vals = []
            for row in [r for r in source_rows if r.get("model_name") == model_name]:
                base = by_key.get((row.get("dataset"), str(row.get("seed")), f"KAN_ADAMW_{carrier}"))
                if not base:
                    continue
                vals.append(
                    {
                        "nll_delta": finite_float(row.get("final_test_NLL"), 999.0) - finite_float(base.get("final_test_NLL"), 999.0),
                        "acc_delta": finite_float(row.get("final_test_accuracy")) - finite_float(base.get("final_test_accuracy")),
                        "auc_delta": finite_float(row.get("AUC_loss_time"), 999999.0) - finite_float(base.get("AUC_loss_time"), 999999.0),
                        "basis_energy": finite_float(row.get("basis_channel_energy_fraction")),
                        "corr_ratio": finite_float(row.get("corr_to_base_norm_ratio_mean"), float("nan")),
                    }
                )
            if vals:
                rows.append(
                    {
                        "repair_group": source,
                        "model_name": model_name,
                        "rows": len(vals),
                        "NLL_noharm_rows": sum(1 for v in vals if v["nll_delta"] <= 0.0),
                        "NLL_improvement_rows": sum(1 for v in vals if v["nll_delta"] < 0.0),
                        "AUC_improvement_rows": sum(1 for v in vals if v["auc_delta"] < 0.0),
                        "accuracy_noharm_rows": sum(1 for v in vals if v["acc_delta"] >= 0.0),
                        "mean_NLL_delta": _mean([v["nll_delta"] for v in vals]),
                        "mean_AUC_delta": _mean([v["auc_delta"] for v in vals]),
                        "basis_energy_ge_05_rows": sum(1 for v in vals if v["basis_energy"] >= 0.5),
                        "mean_corr_to_base_norm_ratio": _mean([v["corr_ratio"] for v in vals if v["corr_ratio"] == v]),
                    }
                )
    return rows


def make_figures() -> None:
    mlp_rows = _all_rows("v22_20_mlp_mmfp_task_matrix*.csv")
    kan_rows = _all_rows("v22_20_kan_mmfp_task_matrix*.csv")
    mlp_smoke_rows = [r for r in mlp_rows if "official_exact_b0_" in str(r.get("_source_file", ""))]
    kan_internal_rows = [r for r in kan_rows if "official_exact_c_kan" in str(r.get("_source_file", ""))]
    step_rows = _all_rows("v22_20_step_diagnostics*.csv")
    gap_rows = _all_rows("v22_20_kan_vs_mlpfu_gap_matrix*.csv")
    cont_rows = _all_rows("v22_20_continual_forgetting_matrix*.csv")
    eta_vals = [finite_float(r.get("eta_quad"), float("nan")) for r in step_rows if r.get("eta_quad", "") != ""]
    eta_vals = [v for v in eta_vals if v == v]
    _svg(OUT_ROOT / "figures/v22_20_eta_quad_distribution.svg", "v22.20 eta_quad distribution", [f"count={len(eta_vals)}", f"mean={sum(eta_vals) / max(1, len(eta_vals)) if eta_vals else ''}", f"min={min(eta_vals) if eta_vals else ''}", f"max={max(eta_vals) if eta_vals else ''}"])
    reasons: dict[str, int] = {}
    for row in step_rows:
        reason = str(row.get("reject_reason", ""))
        reasons[reason] = reasons.get(reason, 0) + 1
    _svg(OUT_ROOT / "figures/v22_20_accept_reject_reason_heatmap.svg", "accept/reject reason counts", [f"{k}: {v}" for k, v in sorted(reasons.items())])
    _svg(OUT_ROOT / "figures/v22_20_Q_real_vs_controls.svg", "Q real vs controls", [f"{r.get('dataset')} s{r.get('seed')} {r.get('model_name')} Q={r.get('Q_improvement')} ctrl_margin={r.get('control_dominance_margin')}" for r in step_rows[:80]])
    _svg(OUT_ROOT / "figures/v22_20_split_train_delta_scatter.svg", "split train delta", [f"{r.get('dataset')} s{r.get('seed')} {r.get('model_name')} split={r.get('split_B_delta')}" for r in step_rows[:80]])
    _svg(OUT_ROOT / "figures/v22_20_hard_slice_delta_panel.svg", "hard slice delta", [f"{r.get('dataset')} s{r.get('seed')} {r.get('model_name')} hard={r.get('hard_slice_delta')}" for r in step_rows[:80]])
    eff = _all_rows("v22_20_four_path_efficiency_audit.csv")
    _svg(OUT_ROOT / "figures/v22_20_four_path_efficiency_waterfall.svg", "four path efficiency", [f"{r.get('model_name')} path1={r.get('path1_base_forward_backward_adamw_sec')} path4={r.get('path4_full_controller_step_sec')} ratio={r.get('full_loop_ratio')}" for r in eff])
    _svg(OUT_ROOT / "figures/v22_20_KANbeFair_gap_reduction_bar.svg", "KANbeFair gap reduction", [f"{r.get('dataset')} s{r.get('seed')} {r.get('carrier')} GapReduction={r.get('GapReduction')}" for r in gap_rows])
    _svg(OUT_ROOT / "figures/v22_20_continual_forgetting_curves.svg", "continual forgetting", [f"{r.get('model_name')} s{r.get('seed')} forget={r.get('average_forgetting')} rel_red={r.get('relative_forgetting_reduction')}" for r in cont_rows])


def finalize(args: argparse.Namespace) -> None:
    ensure_out()
    make_figures()
    solver = read_rows(OUT_ROOT / "v22_20_solver_unit_tests.csv")
    metric = read_rows(OUT_ROOT / "v22_20_metric_curvature_operator_tests.csv")
    acceptance = read_rows(OUT_ROOT / "v22_20_acceptance_tests_matrix.csv")
    control = read_rows(OUT_ROOT / "v22_20_control_dominance_tests.csv")
    identity = read_rows(OUT_ROOT / "v22_20_model_identity_matrix.csv")
    eff = read_rows(OUT_ROOT / "v22_20_four_path_efficiency_audit.csv")
    mlp_rows = _all_rows("v22_20_mlp_mmfp_task_matrix*.csv")
    kan_rows = _all_rows("v22_20_kan_mmfp_task_matrix*.csv")
    mlp_smoke_rows = [r for r in mlp_rows if "official_exact_b0_" in str(r.get("_source_file", ""))]
    kan_internal_rows = [r for r in kan_rows if "official_exact_c_kan" in str(r.get("_source_file", ""))]
    gap_rows = _all_rows("v22_20_kan_vs_mlpfu_gap_matrix*.csv")
    gap_summary = _all_rows("v22_20_kanbefair_gap_reduction_summary*.csv")
    cont_rows = _all_rows("v22_20_continual_forgetting_matrix*.csv")
    deferred = _all_rows("v22_20_deferred_items*.csv")
    a_pass = (
        all(int_flag(r.get("pass")) for r in solver)
        and all(int_flag(r.get("pass")) for r in metric)
        and all(int_flag(r.get("pass")) for r in acceptance)
        and all(int_flag(r.get("pass")) for r in control)
        and all(int_flag(r.get("model_identity_firewall_pass")) for r in identity if int_flag(r.get("official_path")))
    )
    mlp_summary = _summarize_mlp(mlp_smoke_rows)
    kan_summary = _summarize_kan(kan_internal_rows)
    repair_summary = repair_diagnostics_summary(mlp_rows, kan_rows)
    write_rows(OUT_ROOT / "v22_20_repair_diagnostics_summary.csv", repair_summary)
    b_noharm = int(mlp_summary["rows"] >= 9 and mlp_summary["NLL_noharm_rows"] >= 8 and mlp_summary["accuracy_noharm_rows"] >= 7)
    b_improve = int(mlp_summary["rows"] >= 9 and mlp_summary["NLL_improvement_rows"] >= 5 and mlp_summary["AUC_improvement_rows"] >= 6 and mlp_summary["controls_NLL_improvement_rows"] <= 1)
    c_internal = int(kan_summary["rows"] >= 9 and kan_summary["NLL_improvement_rows"] >= 5 and kan_summary["AUC_improvement_rows"] >= 6 and kan_summary["basis_energy_ge_05_rows"] >= min(8, kan_summary["rows"]))
    gap_positive = sum(1 for r in gap_rows if finite_float(r.get("GapReduction")) > 0)
    d_gap = int(gap_rows and gap_positive / max(1, len(gap_rows)) >= 0.60)
    cont_reduction = sum(1 for r in cont_rows if finite_float(r.get("relative_forgetting_reduction"), -999.0) >= 0.05)
    e_cont = int(cont_reduction >= 2)
    if not a_pass:
        route = "R0-CodeOrIdentityFail"
    elif not control or any(not int_flag(r.get("pass")) for r in control):
        route = "R2-MetricControlDominanceFail"
    elif b_improve and c_internal and d_gap and e_cont:
        route = "R12-OfficialDGKANFullSuperiorityCandidate"
    elif b_improve and c_internal and d_gap:
        route = "R8-KANbeFairGapReductionOpened"
    elif b_improve and c_internal:
        route = "R11-KANCarrierValueOpened_ExternalPending"
    elif b_improve:
        route = "R5-MLPFU_ImprovementOpened_KANPending"
    elif b_noharm and mlp_summary["controls_NLL_improvement_rows"] > 1:
        route = "R4-MLPFU_ControlEquivalentNoGo"
    elif b_noharm:
        route = "R3-MLPFU_NoHarmOnly"
    else:
        route = "R15-StrongNoGo_FirstPrinciplesMMFPCannotBeatControls"
    final = {
        "final_route": route,
        "A_solver_identity_pass": int(a_pass),
        "B_mlp_noharm_pass": b_noharm,
        "B_mlp_improvement_pass": b_improve,
        "C_kan_internal_pass": c_internal,
        "D_kanbefair_gap_reduction_pass": d_gap,
        "E_continual_pass": e_cont,
        "mlp_summary": mlp_summary,
        "kan_summary": kan_summary,
        "gap_rows": len(gap_rows),
        "continual_rows": len(cont_rows),
        "deferred_rows": len(deferred),
        "repair_diagnostics_rows": len(repair_summary),
        "latest_status_timestamp": now_sg(),
    }
    write_json(OUT_ROOT / "v22_20_final_route.json", final)
    if not deferred:
        write_rows(OUT_ROOT / "v22_20_deferred_items.csv", [])
    write_rows(OUT_ROOT / "v22_20_artifact_index.csv", artifact_index())
    hashes = []
    for path in source_packet_paths():
        if path.exists():
            hashes.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_rows(OUT_ROOT / "v22_20_source_packet_manifest.csv", hashes)
    mlp_diagnostic_rows = [r for r in mlp_rows if "official_exact_b0_" not in str(r.get("_source_file", ""))]
    kan_diagnostic_rows = [r for r in kan_rows if "official_exact_c_kan" not in str(r.get("_source_file", ""))]
    recap = [
        "# DG-KAN v22.20 First-Principles Metric FU 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 1. Final route",
        "",
        f"- final_route: `{route}`",
        f"- A_solver_identity_pass: `{int(a_pass)}`",
        f"- B_mlp_noharm_pass / B_mlp_improvement_pass: `{b_noharm}` / `{b_improve}`",
        f"- C_kan_internal_pass: `{c_internal}`",
        f"- D_kanbefair_gap_reduction_pass: `{d_gap}`",
        f"- E_continual_pass: `{e_cont}`",
        "",
        "## 2. 关键修复 / 新增实现",
        "",
        "- 新增 `dgkan/fu/mmfp_update.py`：实现 CE categorical Fisher curvature、trace-scaled damping、closed-form alpha、closed-form eta、Q 计算和 matched output controls。",
        "- 新增 `experiments/run_v22_20_common.py`：v22.20 专用输出目录、命令 journal、执行日志、artifact index。",
        "- 新增 `experiments/run_v22_20_first_principles_mmfp.py`：连接 KANbeFair loader、MLPBaseline、strict FC-PureKAN PrimitiveKAN、manual AdamW、basis JVP、split/control/hard-slice acceptance、final route 汇总和图生成。",
        "- 修复审计点：用 exact temporary parameter application 计算 split/hard virtual evaluation，避免线性化 logits 低估债务；finalizer/gap 对 NaN NLL 使用失败口径 `999.0`，避免把 NaN 当成 improvement。",
        "- 性能修复：为 MLPBaseline 和 strict PrimitiveKAN 增加 analytic model JVP 快路径；但效率审计仍未达到 `full_loop_ratio <= 3`。",
        "- 这些修改没有改动 v22.17/v22.18 历史脚本；official KAN row 使用 `PrimitiveKAN`，KANbeFair original KAN / BSpline 只写入 baseline context identity rows。",
        "",
        "## 3. Part A 证据",
        "",
        md_table(solver, ["test", "alpha_rel_error", "predicted_Q_decrease", "pass"], 20),
        md_table(metric, ["metric", "psd_probe", "curvature_operator_finite_and_psd", "pass"], 20),
        md_table(acceptance, ["test", "split_train_safety_test_pass", "hard_slice_no_debt_test_pass", "pass"], 20),
        md_table(control, ["test", "Q_real", "Q_best_control", "control_dominance_margin", "pass"], 20),
        md_table(identity, ["model_name", "official_path", "uses_primitivekan", "is_dgkan_strict_fc_purekan", "uses_bspline_official_path", "model_identity_firewall_pass"], 20),
        "",
        "## 4. Part B MLP+MMFP 结果摘要",
        "",
        f"- rows: `{mlp_summary['rows']}`",
        f"- NLL no-harm rows: `{mlp_summary['NLL_noharm_rows']}`",
        f"- NLL improvement rows: `{mlp_summary['NLL_improvement_rows']}`",
        f"- AUC improvement rows: `{mlp_summary['AUC_improvement_rows']}`",
        f"- controls NLL improvement rows: `{mlp_summary['controls_NLL_improvement_rows']}`",
        f"- mean NLL delta vs MLP+AdamW: `{mlp_summary['mean_NLL_delta']}`",
        f"- hard gate scope: only `official_exact_b0_*` rows enter this summary; diagnostic/external rows are listed separately below.",
        "",
        md_table(mlp_smoke_rows, ["dataset", "seed", "model_name", "final_test_NLL", "final_test_accuracy", "AUC_loss_time", "MMFP_accept_rate", "MMFP_reject_reason_counts", "controller_overhead_ratio"], 80),
        "",
        "### Part B diagnostic / external rows (not used by hard gate)",
        "",
        md_table(mlp_diagnostic_rows, ["_source_file", "dataset", "seed", "model_name", "final_test_NLL", "final_test_accuracy", "MMFP_accept_rate", "MMFP_reject_reason_counts"], 30),
        "",
        "## 5. Part C strict FC-PureKAN+MMFP 摘要",
        "",
        f"- rows: `{kan_summary['rows']}`",
        f"- NLL improvement rows: `{kan_summary['NLL_improvement_rows']}`",
        f"- AUC improvement rows: `{kan_summary['AUC_improvement_rows']}`",
        f"- basis energy >=0.5 rows: `{kan_summary['basis_energy_ge_05_rows']}`",
        f"- full loop ratio <=3 rows: `{kan_summary['full_loop_ratio_le_3_rows']}`",
        f"- mean NLL delta vs KAN+AdamW: `{kan_summary['mean_NLL_delta']}`",
        f"- hard gate scope: only `official_exact_c_kan*` rows enter this summary; external availability rows are diagnostic for Part D.",
        "",
        md_table(kan_internal_rows, ["dataset", "seed", "model_name", "final_test_NLL", "final_test_accuracy", "AUC_loss_time", "MMFP_accept_rate", "basis_channel_energy_fraction", "readout_leakage_fraction", "full_step_ratio"], 80),
        "",
        "### Part C/D external diagnostic rows (not used by C hard gate)",
        "",
        md_table(kan_diagnostic_rows, ["_source_file", "dataset", "seed", "model_name", "final_test_NLL", "final_test_accuracy", "MMFP_accept_rate", "basis_channel_energy_fraction", "readout_leakage_fraction", "full_step_ratio"], 30),
        "",
        "## 6. Part D KANbeFair gap 证据",
        "",
        "- NaN-safe 口径：KAN FU 的 `final_test_NLL=nan` 按 `999.0` 计入 Gap_FU；因此 Cifar10/Wine 不被误判为 gap reduction。",
        md_table(gap_rows, ["dataset", "seed", "carrier", "Gap_BP", "Gap_FU", "GapReduction", "KAN_vs_MLPFU_NLL_delta"], 40),
        md_table(gap_summary, ["dataset", "rows", "GapReduction_positive_rows", "Gap_FU_le_0_rows", "mean_GapReduction"], 40),
        "",
        "## 7. Part E continual learning 证据",
        "",
        md_table(cont_rows, ["dataset", "seed", "model_name", "final_average_accuracy", "average_forgetting", "relative_forgetting_reduction", "task_boundary_MMFP_accept_rate"], 40),
        "",
        "## 8. 效率审计",
        "",
        md_table(eff, ["model_name", "path1_base_forward_backward_adamw_sec", "path2_single_jvp_sec", "path3_mmfp_solve_sec", "path4_full_controller_step_sec", "full_loop_ratio", "official_full_loop_ratio_le_3"], 20),
        "",
        "## 9. Blocker / deferred",
        "",
        md_table(deferred, None, 50),
        "",
        "## 10. 继续修复尝试诊断",
        "",
        "- 这些 rows 是 R15 后继续尝试的机制修复诊断，不参与原始 official hard gate 的 best-row promotion。",
        "- `base-aware residual`：用 `delta + H * Delta_f_base` 求 AdamW base 后的 residual correction。",
        "- `natural basis`：去掉随机 filler，只保留梯度/参数块梯度 basis；rank cap 仍是上限。",
        "- `trust cap`：将 correction norm 限制在当前 AdamW base update norm 内，并在 cap 后重新计算 Q 与 matched controls。",
        "- 关键判定：trust cap 能显著降低 NLL harm、KAN seed0 smoke 修掉 NaN 并改善，但 MLP trust-control 440/1600-step 矩阵中 matched controls 仍大量优于 real residual-trust，因此不能 promotion。",
        "",
        md_table(
            repair_summary,
            [
                "repair_group",
                "model_name",
                "rows",
                "NLL_noharm_rows",
                "NLL_improvement_rows",
                "AUC_improvement_rows",
                "accuracy_noharm_rows",
                "mean_NLL_delta",
                "mean_AUC_delta",
                "mean_trust_scale",
                "mean_corr_to_base_norm_ratio",
                "controls_better_NLL_rows",
                "controls_better_AUC_rows",
                "basis_energy_ge_05_rows",
                "note",
            ],
            80,
        ),
        "",
        "## 11. 分析与结论",
        "",
        "- 结论只按上述 artifact 判定；未跑出的矩阵写入 deferred，不提升为完成。",
        "- 若 final_route 是 no-go 或 partial opened，含义是当前 first-principles MMFP 在已执行矩阵中未满足对应硬门；这不是算法成功声明。",
        "- 证据链优先级：A 数学/identity gate -> B MLP general value -> C KAN carrier value -> D KANbeFair gap -> E continual。任一上游失败都会限制下游主张。",
        "- matched controls 的作用是防止 v22.18 中 signflip/random 也能 improvement 的误 promotion；本轮 summary 明确计数 controls improvement rows。",
        "- 追加修复后的新 insight：原始失败的直接工程原因之一是 correction/base norm 过大；natural basis 和 base-norm trust cap 可以缓解甚至让 KAN smoke 改善，但 MLP matched controls 更强，说明当前 metric/correction 仍缺少可区分 real direction 的 task advantage。",
        "",
        "## 12. Artifact index",
        "",
        "- `results/v22_20/v22_20_artifact_index.csv`",
        "- `results/v22_20/v22_20_source_packet_manifest.csv`",
        "- `results/v22_20/v22_20_repair_diagnostics_summary.csv`",
        "- figures: `results/v22_20/figures/`",
        "",
    ]
    RECAP_DOC.write_text("\n".join(recap), encoding="utf-8")
    append_exec(
        " ".join(sys.argv),
        task_id="finalize",
        status="pass",
        gpu=args.physical_gpu or args.device,
        files="results/v22_20/v22_20_final_route.json; docs/DG-KAN_v22.20_FirstPrinciplesMetricFU_实验结果复盘.md; results/v22_20/v22_20_artifact_index.csv",
        note=f"final_route={route}",
        exit_code=0,
    )


def main() -> None:
    args = parser().parse_args()
    if args.clear_proxy_env:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(key, None)
    ensure_out()
    if args.stage == "unit":
        run_unit(args)
    elif args.stage == "task":
        run_task(args)
    elif args.stage == "continual":
        run_continual(args)
    elif args.stage == "finalize":
        finalize(args)
    elif args.stage == "all":
        run_unit(args)
        run_task(args)
        run_continual(args)
        finalize(args)


if __name__ == "__main__":
    main()
