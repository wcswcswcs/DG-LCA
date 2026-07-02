"""Edge-Sobolev population-flow optimizers for DG-KAN v23.00R."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from dgkan.fu.debt_conflict_veto import conflict_veto_gate
from dgkan.fu.edge_sobolev_metrics import flatten_param_mode_weights, functional_edge_gram, make_mode_weights, mode_band_slices
from dgkan.fu.population_risk_gate import block_snr_gate, diagonal_snr_gate


@dataclass
class EdgeSobolevStepStats:
    step: int = 0
    gate_density_mean: float = 1.0
    gate_density_median: float = 1.0
    gate_density_p10: float = 1.0
    gate_density_p90: float = 1.0
    veto_density_mean: float = 0.0
    preserved_task_energy_fraction: float = 1.0
    mode_weight_condition_max: float = 1.0
    edge_params_updated: int = 0
    non_edge_params_seen: int = 0


class EdgeSobolevPopulationFlow(torch.optim.Optimizer):
    """Minimal optimizer that updates only KAN edge coefficient tensors.

    The optimizer expects parameters to carry ``_kan_is_edge_coeff=True`` and
    basis metadata. It can run as Edge-Sobolev AdamW without a population gate,
    or use pre-observed per-example gradients for a diagonal SNR gate.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        sobolev_exponent: float = 1.0,
        edge_metric_type: str = "mode_diag",
        edge_weight_normalization: str = "median",
        edge_weight_ridge: float = 0.0,
        functional_gram_quadrature_points: int = 257,
        gate_beta: float = 2.0,
        gate_family: str = "diagonal",
        gate_floor: float = 0.0,
        gate_input_side: int = 0,
        gate_patch_size: int = 2,
        stat_warmup_steps: int = 0,
        use_population_gate: bool = False,
        use_debt_veto: bool = False,
        strict_edge_params: bool = True,
    ) -> None:
        self.strict_edge_params = bool(strict_edge_params)
        defaults = dict(
            lr=float(lr),
            betas=tuple(float(x) for x in betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            sobolev_exponent=float(sobolev_exponent),
            edge_metric_type=str(edge_metric_type),
            edge_weight_normalization=str(edge_weight_normalization),
            edge_weight_ridge=float(edge_weight_ridge),
            functional_gram_quadrature_points=int(functional_gram_quadrature_points),
            gate_beta=float(gate_beta),
            gate_family=str(gate_family),
            gate_floor=float(gate_floor),
            gate_input_side=int(gate_input_side),
            gate_patch_size=int(gate_patch_size),
            stat_warmup_steps=int(stat_warmup_steps),
            use_population_gate=bool(use_population_gate),
            use_debt_veto=bool(use_debt_veto),
            debt_veto_conflict_mode="legacy_positive",
        )
        super().__init__(list(params), defaults)
        self._observed_grads: dict[int, torch.Tensor] = {}
        self._observed_debt_grads: dict[int, dict[str, torch.Tensor]] = {}
        self._gram_cache: dict[tuple[str, int, float, int, str, float, str, torch.dtype], tuple[torch.Tensor, torch.Tensor, float]] = {}
        self.last_stats = EdgeSobolevStepStats()
        self._validate_params()

    def _validate_params(self) -> None:
        bad = []
        for group in self.param_groups:
            for param in group["params"]:
                if not getattr(param, "_kan_is_edge_coeff", False):
                    bad.append(tuple(param.shape))
        if bad and self.strict_edge_params:
            raise ValueError(f"EdgeSobolevPopulationFlow received non-edge parameters: {bad[:4]}")

    def observe_per_example_gradients(self, param: torch.nn.Parameter, grads: torch.Tensor) -> None:
        self._observed_grads[id(param)] = grads.detach().to(device=param.device, dtype=param.dtype)

    def observe_debt_per_example_gradients(self, component: str, param: torch.nn.Parameter, grads: torch.Tensor) -> None:
        by_component = self._observed_debt_grads.setdefault(id(param), {})
        by_component[str(component)] = grads.detach().to(device=param.device, dtype=param.dtype)

    def clear_observed_gradients(self) -> None:
        self._observed_grads.clear()
        self._observed_debt_grads.clear()

    def _weights_for_param(self, param: torch.nn.Parameter, group: dict) -> torch.Tensor:
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        axis = int(getattr(param, "_kan_mode_axis", -1))
        mode_weights = make_mode_weights(
            str(basis_key),
            k,
            exponent=float(group["sobolev_exponent"]),
            normalization=str(group.get("edge_weight_normalization", "median")),
            ridge=float(group.get("edge_weight_ridge", 0.0)),
            device=param.device,
            dtype=torch.float64,
        )
        return flatten_param_mode_weights(param, mode_weights, mode_axis=axis).to(device=param.device, dtype=param.dtype)

    def _functional_gram_for_param(self, param: torch.nn.Parameter, group: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
        basis_key = str(getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5")))
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        cache_key = (
            basis_key,
            k,
            float(group["sobolev_exponent"]),
            int(group.get("functional_gram_quadrature_points", 257)),
            str(group.get("edge_weight_normalization", "trace")),
            float(group.get("edge_weight_ridge", 1.0e-6)),
            str(param.device),
            param.dtype,
        )
        cached = self._gram_cache.get(cache_key)
        if cached is not None:
            return cached
        gram = functional_edge_gram(
            basis_key,
            k,
            sobolev_order=float(group["sobolev_exponent"]),
            quadrature_points=int(group.get("functional_gram_quadrature_points", 257)),
            normalization=str(group.get("edge_weight_normalization", "trace")),
            ridge=float(group.get("edge_weight_ridge", 1.0e-6)),
            device=param.device,
            dtype=param.dtype,
        )
        vals, basis = torch.linalg.eigh(0.5 * (gram + gram.T))
        eps = float(group["eps"])
        inv_sqrt = (basis * vals.clamp_min(eps).rsqrt().reshape(1, -1)) @ basis.T
        cond = float((vals.max().clamp_min(eps) / vals.min().clamp_min(eps)).detach().cpu().item()) if int(vals.numel()) else 1.0
        cached = (gram, inv_sqrt, cond)
        self._gram_cache[cache_key] = cached
        return cached

    def _apply_metric_inv_sqrt(self, tensor: torch.Tensor, param: torch.nn.Parameter, group: dict) -> torch.Tensor:
        if str(group.get("edge_metric_type", "mode_diag")) != "functional_gram":
            weights = self._weights_for_param(param, group).reshape_as(param).clamp_min(float(group["eps"]))
            if tuple(tensor.shape) == tuple(param.shape):
                return tensor / weights.sqrt()
            return tensor / weights.reshape((1,) + tuple(param.shape)).sqrt()
        _, inv_sqrt, _ = self._functional_gram_for_param(param, group)
        axis = int(getattr(param, "_kan_mode_axis", -1))
        if axis < 0:
            axis += param.ndim
        leading_example = int(tensor.ndim) == int(param.ndim) + 1
        work = tensor.movedim(axis + (1 if leading_example else 0), -1)
        orig_shape = tuple(work.shape)
        flat = work.reshape(-1, int(inv_sqrt.shape[0]))
        out = flat @ inv_sqrt.T.to(device=tensor.device, dtype=tensor.dtype)
        return out.reshape(orig_shape).movedim(-1, axis + (1 if leading_example else 0))

    def _blocks_for_param(self, param: torch.nn.Parameter, group: dict | None = None) -> list[list[int]]:
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        if int(param.numel()) <= 0 or k <= 0:
            return []
        family = str((group or {}).get("gate_family", "block")).lower()
        if "corner_checker_hybrid" in family or "cornercheckerhybrid" in family:
            blocks = self._corner_checker_hybrid_blocks_for_param(param, group or {}, degree_bands="degree" in family)
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "visual_pattern" in family or "visualpattern" in family:
            blocks = [
                *self._diagonal_pattern_blocks_for_param(param, group or {}, degree_bands="degree" in family),
                *self._checker_patch_blocks_for_param(param, group or {}, degree_bands="degree" in family),
            ]
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "checker_patch" in family or "checkerpatch" in family or "local_patch" in family or "localpatch" in family:
            blocks = self._checker_patch_blocks_for_param(param, group or {}, degree_bands="degree" in family)
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "edgebank" in family:
            rows = max(1, int(param.numel()) // k)
            if "degree" not in family:
                return [list(range(row * k, min((row + 1) * k, int(param.numel())))) for row in range(rows)]
            bands = mode_band_slices(k, str(basis_key))
            blocks: list[list[int]] = []
            for row in range(rows):
                base = row * k
                for mode_ids in bands.values():
                    block = [base + mode for mode in mode_ids if base + mode < int(param.numel())]
                    if block:
                        blocks.append(block)
            return blocks
        blocks: list[list[int]] = []
        bands = mode_band_slices(k, str(basis_key))
        for mode_ids in bands.values():
            block = [idx for idx in range(int(param.numel())) if idx % k in mode_ids]
            if block:
                blocks.append(block)
        return blocks

    def _edgebank_blocks_for_param(
        self,
        param: torch.nn.Parameter,
        basis_key: str,
        k: int,
        *,
        degree_bands: bool,
        exclude_rows: set[int] | None = None,
    ) -> list[list[int]]:
        rows = max(1, int(param.numel()) // k)
        excluded = exclude_rows or set()
        if not degree_bands:
            return [
                list(range(row * k, min((row + 1) * k, int(param.numel()))))
                for row in range(rows)
                if row not in excluded
            ]
        bands = mode_band_slices(k, str(basis_key))
        blocks: list[list[int]] = []
        for row in range(rows):
            if row in excluded:
                continue
            base = row * k
            for mode_ids in bands.values():
                block = [base + mode for mode in mode_ids if base + mode < int(param.numel())]
                if block:
                    blocks.append(block)
        return blocks

    def _corner_checker_hybrid_blocks_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[int]]:
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
            k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
            return self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands)
        corner_blocks = self._checker_patch_blocks_for_param(param, {**group, "gate_checker_anchor_mode": "corners"}, degree_bands=degree_bands)
        if not corner_blocks:
            return []
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        covered_rows = {int(idx) // max(1, k) for block in corner_blocks for idx in block}
        return [
            *corner_blocks,
            *self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands, exclude_rows=covered_rows),
        ]

    def _image_edge_row_groups_for_coords(self, param: torch.nn.Parameter, group: dict, coords: Iterable[tuple[int, int]]) -> list[int]:
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        input_dim = int(param.shape[0]) if int(param.ndim) >= 1 else 0
        side = int(group.get("gate_input_side", 0) or 0)
        if side <= 0:
            root = int(round(math.sqrt(max(1, input_dim))))
            side = root if root * root == input_dim else 0
        if side <= 0 or side * side != input_dim or k <= 0:
            return []
        row_count = max(1, int(param.numel()) // k)
        rows_per_input = row_count // max(1, input_dim)
        if rows_per_input <= 0:
            return []
        edge_rows: list[int] = []
        for row, col in coords:
            if 0 <= int(row) < side and 0 <= int(col) < side:
                input_idx = int(row) * side + int(col)
                start = input_idx * rows_per_input
                edge_rows.extend(range(start, min(start + rows_per_input, row_count)))
        return edge_rows

    def _blocks_from_edge_rows(self, param: torch.nn.Parameter, group: dict, edge_rows: list[int], *, degree_bands: bool) -> list[list[int]]:
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        mode_groups = mode_band_slices(k, str(basis_key)).values() if degree_bands else [list(range(k))]
        blocks: list[list[int]] = []
        for mode_ids in mode_groups:
            block = [edge_row * k + mode for edge_row in edge_rows for mode in mode_ids if edge_row * k + mode < int(param.numel())]
            if block:
                blocks.append(block)
        return blocks

    def _diagonal_pattern_blocks_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[int]]:
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            return []
        input_dim = int(param.shape[0])
        side = int(group.get("gate_input_side", 0) or 0)
        if side <= 0:
            root = int(round(math.sqrt(max(1, input_dim))))
            side = root if root * root == input_dim else 0
        if side <= 0 or side * side != input_dim:
            return []
        diag_rows = self._image_edge_row_groups_for_coords(param, group, ((idx, idx) for idx in range(side)))
        anti_rows = self._image_edge_row_groups_for_coords(param, group, ((idx, side - 1 - idx) for idx in range(side)))
        return [
            *self._blocks_from_edge_rows(param, group, diag_rows, degree_bands=degree_bands),
            *self._blocks_from_edge_rows(param, group, anti_rows, degree_bands=degree_bands),
        ]

    def _checker_patch_blocks_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[int]]:
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            return []
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        if k <= 0 or int(param.shape[-1]) != k:
            return []
        input_dim = int(param.shape[0])
        side = int(group.get("gate_input_side", 0) or 0)
        if side <= 0:
            root = int(round(math.sqrt(max(1, input_dim))))
            side = root if root * root == input_dim else 0
        patch_size = max(2, int(group.get("gate_patch_size", 2) or 2))
        if side <= 0 or side * side != input_dim or side < patch_size + 1:
            return []
        row_count = max(1, int(param.numel()) // k)
        rows_per_input = row_count // max(1, input_dim)
        if rows_per_input <= 0:
            return []
        if str(group.get("gate_checker_anchor_mode", "")).lower() == "corners":
            anchors = [1, side - patch_size - 1]
        else:
            anchors = list(range(1, max(2, side - patch_size), patch_size))
        if not anchors:
            anchors = [0]
        max_anchor = side - patch_size
        anchors = sorted({min(max(0, anchor), max_anchor) for anchor in anchors})
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        mode_groups = mode_band_slices(k, str(basis_key)).values() if degree_bands else [list(range(k))]
        blocks: list[list[int]] = []
        for top in anchors:
            for left in anchors:
                polarities = (
                    ((top, left), (top + 1, left + 1)),
                    ((top, left + 1), (top + 1, left)),
                )
                for coords in polarities:
                    edge_rows = self._image_edge_row_groups_for_coords(param, group, coords)
                    blocks.extend(self._blocks_from_edge_rows(param, group, edge_rows, degree_bands=degree_bands))
        return blocks

    @staticmethod
    def _base_family_for_random_match(family: str) -> str:
        lower = str(family).lower()
        if "random_matched" not in lower:
            return lower
        base = lower.replace("_random_matched", "").replace("random_matched_", "")
        return base or "diagonal"

    @staticmethod
    def _permute_block_gate(gate: torch.Tensor, blocks: list[list[int]]) -> torch.Tensor:
        clean = [sorted({int(i) for i in block if 0 <= int(i) < int(gate.numel())}) for block in blocks]
        clean = [block for block in clean if block]
        if len(clean) <= 1:
            return gate
        vals = [gate[block[0]].detach().clone() for block in clean]
        perm = torch.randperm(len(clean), device=gate.device).tolist()
        out = gate.clone()
        for dst, src in enumerate(perm):
            out[torch.tensor(clean[dst], dtype=torch.long, device=gate.device)] = vals[int(src)]
        return out

    def _gate_from_observed(self, observed: torch.Tensor, weights: torch.Tensor, param: torch.nn.Parameter, group: dict) -> tuple[torch.Tensor, torch.Tensor]:
        white_tensor = self._apply_metric_inv_sqrt(observed, param, group)
        white_obs = white_tensor.reshape(int(observed.shape[0]), -1)
        family = str(group.get("gate_family", "diagonal")).lower()
        random_matched = "random_matched" in family
        base_family = self._base_family_for_random_match(family)
        block_group = dict(group)
        block_group["gate_family"] = base_family
        if base_family == "block" or "edgebank" in base_family or "checker_patch" in base_family or "checkerpatch" in base_family or "local_patch" in base_family or "localpatch" in base_family or "visual_pattern" in base_family or "visualpattern" in base_family:
            blocks = self._blocks_for_param(param, block_group)
            gres = block_snr_gate(white_obs.detach().cpu(), blocks, beta=float(group["gate_beta"]))
        else:
            gres = diagonal_snr_gate(white_obs.detach().cpu(), beta=float(group["gate_beta"]))
        gate = gres.gate.to(device=param.device, dtype=param.dtype)
        if random_matched and int(gate.numel()) > 0:
            if base_family == "block" or "edgebank" in base_family or "checker_patch" in base_family or "checkerpatch" in base_family or "local_patch" in base_family or "localpatch" in base_family or "visual_pattern" in base_family or "visualpattern" in base_family:
                gate = self._permute_block_gate(gate, blocks)
            else:
                perm = torch.randperm(int(gate.numel()), device=param.device)
                gate = gate[perm]
        if float(group.get("gate_floor", 0.0)) > 0.0:
            floor = float(group["gate_floor"])
            gate = gate * (1.0 - floor) + floor
        return gate.reshape_as(param), gres.mean.to(device=param.device, dtype=param.dtype).reshape(-1)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gate_density: list[float] = []
        gate_values: list[torch.Tensor] = []
        veto_density: list[float] = []
        preserved_energy: list[float] = []
        conds: list[float] = []
        updated = 0
        non_edge = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if not getattr(param, "_kan_is_edge_coeff", False):
                    non_edge += 1
                    if self.strict_edge_params:
                        raise ValueError("non-edge parameter reached optimizer step")
                    continue
                grad = param.grad.detach()
                weights = self._weights_for_param(param, group)
                if str(group.get("edge_metric_type", "mode_diag")) == "functional_gram":
                    _, _, cond = self._functional_gram_for_param(param, group)
                    conds.append(cond)
                else:
                    conds.append(float((weights.max() / weights.min().clamp_min(float(group["eps"]))).detach().cpu().item()))
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                state["step"] += 1
                m = state["m"]
                v = state["v"]
                grad_white = self._apply_metric_inv_sqrt(grad, param, group)
                m.mul_(beta1).add_(grad_white, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad_white, grad_white, value=1.0 - beta2)
                bias_correction1 = 1.0 - float(beta1) ** int(state["step"])
                bias_correction2 = 1.0 - float(beta2) ** int(state["step"])
                m_hat = m / max(bias_correction1, float(group["eps"]))
                v_hat = v / max(bias_correction2, float(group["eps"]))
                update_white = m_hat / (v_hat.sqrt() + float(group["eps"]))
                gate = torch.ones_like(update_white)
                task_mu = grad_white.detach().reshape(-1)
                if bool(group["use_population_gate"]):
                    observed = self._observed_grads.get(id(param))
                    warmup = int(state["step"]) <= int(group.get("stat_warmup_steps", 0))
                    if observed is not None and observed.ndim >= 2 and not warmup:
                        gate, task_mu = self._gate_from_observed(observed, weights, param, group)
                        if bool(group.get("use_debt_veto", False)):
                            debt_gates: dict[str, torch.Tensor] = {}
                            debt_mus: dict[str, torch.Tensor] = {}
                            for component, debt_obs in self._observed_debt_grads.get(id(param), {}).items():
                                if debt_obs is None or debt_obs.ndim < 2:
                                    continue
                                dgate, dmu = self._gate_from_observed(debt_obs, weights, param, group)
                                debt_gates[component] = dgate.reshape(-1)
                                debt_mus[component] = dmu.reshape(-1)
                            if debt_gates:
                                flat_gate, veto_summary = conflict_veto_gate(
                                    gate.reshape(-1),
                                    task_mu.reshape(-1),
                                    debt_gates,
                                    debt_mus,
                                    conflict_mode=str(group.get("debt_veto_conflict_mode", "legacy_positive")),
                                )
                                gate = flat_gate.to(device=param.device, dtype=param.dtype).reshape_as(param)
                                veto_density.append(float(veto_summary.get("veto_density_mean", 0.0)))
                                preserved_energy.append(float(veto_summary.get("preserved_task_energy_fraction", 1.0)))
                gate_density.append(float(gate.detach().mean().cpu().item()))
                gate_values.append(gate.detach().reshape(-1).to(dtype=torch.float64).cpu())
                update_raw = self._apply_metric_inv_sqrt(gate * update_white, param, group)
                if float(group["weight_decay"]) != 0.0:
                    param.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
                param.add_(update_raw, alpha=-float(group["lr"]))
                updated += 1
        all_gate = torch.cat(gate_values) if gate_values else torch.zeros(0, dtype=torch.float64)
        self.last_stats = EdgeSobolevStepStats(
            step=max([int(self.state[p].get("step", 0)) for group in self.param_groups for p in group["params"] if p in self.state] or [0]),
            gate_density_mean=float(sum(gate_density) / len(gate_density)) if gate_density else 0.0,
            gate_density_median=float(torch.quantile(all_gate, 0.5).item()) if int(all_gate.numel()) else 0.0,
            gate_density_p10=float(torch.quantile(all_gate, 0.1).item()) if int(all_gate.numel()) else 0.0,
            gate_density_p90=float(torch.quantile(all_gate, 0.9).item()) if int(all_gate.numel()) else 0.0,
            veto_density_mean=float(sum(veto_density) / len(veto_density)) if veto_density else 0.0,
            preserved_task_energy_fraction=float(sum(preserved_energy) / len(preserved_energy)) if preserved_energy else 1.0,
            mode_weight_condition_max=max(conds or [1.0]),
            edge_params_updated=updated,
            non_edge_params_seen=non_edge,
        )
        self.clear_observed_gradients()
        return loss


class EdgeSobolevAdamW(EdgeSobolevPopulationFlow):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        kwargs["use_population_gate"] = False
        super().__init__(params, **kwargs)


class EdgeSobolevSNRFU(EdgeSobolevPopulationFlow):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        kwargs["use_population_gate"] = True
        super().__init__(params, **kwargs)


class EdgeSobolevSNRFUWithDebtVeto(EdgeSobolevSNRFU):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs) -> None:
        kwargs["use_debt_veto"] = True
        super().__init__(params, **kwargs)


def mark_kan_edge_params(model: torch.nn.Module, *, basis_key: str) -> None:
    coeffs = list(getattr(model, "coeffs", []))
    for layer_id, param in enumerate(coeffs):
        param._kan_layer_id = int(layer_id)  # type: ignore[attr-defined]
        param._kan_basis_key = str(basis_key)  # type: ignore[attr-defined]
        param._kan_basis_order_or_freq_count = int(param.shape[-1]) if param.ndim else 1  # type: ignore[attr-defined]
        param._kan_mode_axis = -1  # type: ignore[attr-defined]
        param._kan_edge_axis = 0  # type: ignore[attr-defined]
        param._kan_is_edge_coeff = True  # type: ignore[attr-defined]


def optimizer_smoke_test(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, *, basis_key: str) -> dict[str, float | int]:
    mark_kan_edge_params(model, basis_key=basis_key)
    params = list(getattr(model, "coeffs", []))
    opt = EdgeSobolevAdamW(params, lr=1.0e-3, sobolev_exponent=1.0)
    before = [p.detach().clone() for p in params]
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x).float(), y.long())
    loss.backward()
    opt.step()
    changed = [int(not torch.allclose(a, p.detach())) for a, p in zip(before, params)]
    return {
        "standard_forward_backward_optimizer_loop": 1,
        "changed_edge_coefficients": int(any(changed)),
        "changed_mlp_tensors": 0,
        "edge_params_updated": opt.last_stats.edge_params_updated,
        "gate_density_mean": opt.last_stats.gate_density_mean,
        "mode_weight_condition_max": opt.last_stats.mode_weight_condition_max,
    }
