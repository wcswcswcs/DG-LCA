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
        gate_floor_mode: str = "final",
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
            gate_floor_mode=str(gate_floor_mode),
            gate_input_side=int(gate_input_side),
            gate_patch_size=int(gate_patch_size),
            stat_warmup_steps=int(stat_warmup_steps),
            use_population_gate=bool(use_population_gate),
            use_debt_veto=bool(use_debt_veto),
            debt_veto_conflict_mode="legacy_positive",
        )
        super().__init__(list(params), defaults)
        self._observed_grads: dict[int, torch.Tensor] = {}
        self._observed_labels: dict[int, torch.Tensor] = {}
        self._observed_structure_gates: dict[int, torch.Tensor] = {}
        self._observed_structure_update_modifiers: dict[int, torch.Tensor] = {}
        self._observed_structure_update_projectors: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
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

    def observe_per_example_gradients(self, param: torch.nn.Parameter, grads: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        self._observed_grads[id(param)] = grads.detach().to(device=param.device, dtype=param.dtype)
        if labels is not None:
            self._observed_labels[id(param)] = labels.detach().reshape(-1).to(device=param.device)

    def observe_debt_per_example_gradients(self, component: str, param: torch.nn.Parameter, grads: torch.Tensor) -> None:
        by_component = self._observed_debt_grads.setdefault(id(param), {})
        by_component[str(component)] = grads.detach().to(device=param.device, dtype=param.dtype)

    def observe_structure_gate(self, param: torch.nn.Parameter, gate: torch.Tensor) -> None:
        self._observed_structure_gates[id(param)] = gate.detach().to(device=param.device, dtype=param.dtype)

    def observe_structure_update_modifier(self, param: torch.nn.Parameter, modifier: torch.Tensor) -> None:
        self._observed_structure_update_modifiers[id(param)] = modifier.detach().to(device=param.device, dtype=param.dtype)

    def observe_structure_update_projector(self, param: torch.nn.Parameter, block_directions: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        clean: list[tuple[torch.Tensor, torch.Tensor]] = []
        numel = int(param.numel())
        for idx, direction in block_directions:
            idx_t = idx.detach().reshape(-1).to(device=param.device, dtype=torch.long)
            dir_t = direction.detach().reshape(-1).to(device=param.device, dtype=param.dtype)
            valid = idx_t.numel() > 0 and idx_t.numel() == dir_t.numel()
            if valid and int(idx_t.min().item()) >= 0 and int(idx_t.max().item()) < numel:
                clean.append((idx_t, dir_t))
        self._observed_structure_update_projectors[id(param)] = clean

    def clear_observed_gradients(self) -> None:
        self._observed_grads.clear()
        self._observed_labels.clear()
        self._observed_structure_gates.clear()
        self._observed_structure_update_modifiers.clear()
        self._observed_structure_update_projectors.clear()
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
        if family in {"layer", "layer_block", "block_layer"}:
            return [list(range(int(param.numel())))]
        if "corner_checker_contrast" in family or "cornercheckercontrast" in family:
            if int(getattr(param, "_kan_layer_id", -1)) == 0 and int(param.ndim) >= 3:
                pairs = self._corner_checker_contrast_pairs_for_param(param, group or {}, degree_bands="degree" in family)
                blocks = [block for pair in pairs for block in pair]
                if blocks:
                    return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "corner_checker_focus" in family or "cornercheckerfocus" in family:
            blocks = self._corner_checker_focus_blocks_for_param(param, group or {}, degree_bands="degree" in family)
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "corner_checker_multiring" in family or "cornercheckermultiring" in family:
            blocks = self._corner_checker_multiring_hybrid_blocks_for_param(param, group or {}, degree_bands="degree" in family)
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "corner_checker_diagonal_union" in family or "cornercheckerdiagonalunion" in family:
            block_sets = self._corner_checker_diagonal_union_block_sets_for_param(param, group or {}, degree_bands="degree" in family)
            blocks = [block for component in block_sets for block in component]
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
        if "corner_checker_diagonal_hybrid" in family or "cornercheckerdiagonalhybrid" in family:
            blocks = self._corner_checker_diagonal_hybrid_blocks_for_param(param, group or {}, degree_bands="degree" in family)
            if blocks:
                return blocks
            fallback = dict(group or {})
            fallback["gate_family"] = "degree_edgebank" if "degree" in family else "edgebank"
            return self._blocks_for_param(param, fallback)
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

    @staticmethod
    def _dedupe_blocks(blocks: list[list[int]]) -> list[list[int]]:
        seen: set[tuple[int, ...]] = set()
        out: list[list[int]] = []
        for block in blocks:
            clean = tuple(sorted({int(idx) for idx in block}))
            if not clean or clean in seen:
                continue
            seen.add(clean)
            out.append(list(clean))
        return out

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

    def _corner_checker_contrast_pairs_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[tuple[list[int], list[int]]]:
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
        anchors = sorted({min(max(0, anchor), side - patch_size) for anchor in [1, side - patch_size - 1]})
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        mode_groups = mode_band_slices(k, str(basis_key)).values() if degree_bands else [list(range(k))]
        pairs: list[tuple[list[int], list[int]]] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for top in anchors:
            for left in anchors:
                edge_rows_a = self._image_edge_row_groups_for_coords(param, group, ((top, left), (top + 1, left + 1)))
                edge_rows_b = self._image_edge_row_groups_for_coords(param, group, ((top, left + 1), (top + 1, left)))
                for mode_ids in mode_groups:
                    block_a = [edge_row * k + mode for edge_row in edge_rows_a for mode in mode_ids if edge_row * k + mode < int(param.numel())]
                    block_b = [edge_row * k + mode for edge_row in edge_rows_b for mode in mode_ids if edge_row * k + mode < int(param.numel())]
                    clean_a = tuple(sorted({int(idx) for idx in block_a}))
                    clean_b = tuple(sorted({int(idx) for idx in block_b}))
                    key = (clean_a, clean_b)
                    if not clean_a or not clean_b or key in seen:
                        continue
                    seen.add(key)
                    pairs.append((list(clean_a), list(clean_b)))
        return pairs

    def _corner_checker_focus_blocks_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[int]]:
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
            k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
            return self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands)
        return self._dedupe_blocks(
            self._checker_patch_blocks_for_param(param, {**group, "gate_checker_anchor_mode": "corners"}, degree_bands=degree_bands)
        )

    def _corner_checker_multiring_hybrid_blocks_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[int]]:
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
            k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
            return self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands)
        corner_blocks = self._checker_patch_blocks_for_param(
            param,
            {**group, "gate_checker_anchor_mode": "inner_outer_corners"},
            degree_bands=degree_bands,
        )
        corner_blocks = self._dedupe_blocks(corner_blocks)
        if not corner_blocks:
            return []
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        covered_rows = {int(idx) // max(1, k) for block in corner_blocks for idx in block}
        return [
            *corner_blocks,
            *self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands, exclude_rows=covered_rows),
        ]

    def _corner_checker_diagonal_hybrid_blocks_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[int]]:
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
            k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
            return self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands)
        corner_blocks = self._checker_patch_blocks_for_param(param, {**group, "gate_checker_anchor_mode": "corners"}, degree_bands=degree_bands)
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        covered_rows = {int(idx) // max(1, k) for block in corner_blocks for idx in block}
        input_dim = int(param.shape[0])
        side = int(group.get("gate_input_side", 0) or 0)
        if side <= 0:
            root = int(round(math.sqrt(max(1, input_dim))))
            side = root if root * root == input_dim else 0
        diagonal_blocks: list[list[int]] = []
        if side > 0 and side * side == input_dim:
            diag_rows = self._image_edge_row_groups_for_coords(param, group, ((idx, idx) for idx in range(side)))
            anti_rows = self._image_edge_row_groups_for_coords(param, group, ((idx, side - 1 - idx) for idx in range(side)))
            diag_rows = [row for row in dict.fromkeys(diag_rows) if row not in covered_rows]
            anti_rows = [row for row in dict.fromkeys(anti_rows) if row not in covered_rows]
            diagonal_blocks.extend(self._blocks_from_edge_rows(param, group, diag_rows, degree_bands=degree_bands))
            diagonal_blocks.extend(self._blocks_from_edge_rows(param, group, anti_rows, degree_bands=degree_bands))
            covered_rows.update(diag_rows)
            covered_rows.update(anti_rows)
        if not corner_blocks and not diagonal_blocks:
            return []
        return [
            *corner_blocks,
            *diagonal_blocks,
            *self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands, exclude_rows=covered_rows),
        ]

    def _corner_checker_diagonal_union_block_sets_for_param(self, param: torch.nn.Parameter, group: dict, *, degree_bands: bool) -> list[list[list[int]]]:
        basis_key = getattr(param, "_kan_basis_key", getattr(param, "_kan_basis_name", "dche_k5"))
        k = int(getattr(param, "_kan_basis_order_or_freq_count", param.shape[-1] if param.ndim else 1))
        if int(getattr(param, "_kan_layer_id", -1)) != 0 or int(param.ndim) < 3:
            blocks = self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands)
            return [blocks] if blocks else []
        components = [
            self._checker_patch_blocks_for_param(param, {**group, "gate_checker_anchor_mode": "corners"}, degree_bands=degree_bands),
            self._diagonal_pattern_blocks_for_param(param, group, degree_bands=degree_bands),
            self._edgebank_blocks_for_param(param, str(basis_key), k, degree_bands=degree_bands),
        ]
        return [blocks for blocks in components if blocks]

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
        anchor_mode = str(group.get("gate_checker_anchor_mode", "")).lower()
        if anchor_mode in {"outer_corners", "outer-corners", "outer"}:
            anchors = [0, side - patch_size]
        elif anchor_mode in {"inner_outer_corners", "inner-outer-corners", "multiring_corners", "corner_multiring"}:
            anchors = [0, 1, side - patch_size - 1, side - patch_size]
        elif anchor_mode == "corners":
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
    def _base_family_for_class_conditional(family: str) -> str:
        lower = str(family).lower()
        base = lower.replace("_class_conditional", "").replace("class_conditional_", "")
        base = base.replace("_classconditional", "").replace("classconditional_", "")
        return base or "diagonal"

    @staticmethod
    def _base_family_for_class_cohort(family: str) -> str:
        lower = str(family).lower()
        base = lower.replace("_class_cohort", "").replace("class_cohort_", "")
        base = base.replace("_classcohort", "").replace("classcohort_", "")
        return base or "diagonal"

    @staticmethod
    def _uses_block_snr_family(family: str) -> bool:
        text = str(family).lower()
        return (
            text in {"block", "layer", "degree", "layer_block", "block_layer"}
            or "edgebank" in text
            or "checker_patch" in text
            or "checkerpatch" in text
            or "local_patch" in text
            or "localpatch" in text
            or "visual_pattern" in text
            or "visualpattern" in text
            or "corner_checker" in text
            or "cornerchecker" in text
        )

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

    @staticmethod
    def _topk_block_gate(gate: torch.Tensor, blocks: list[list[int]], *, fraction: float = 0.35) -> torch.Tensor:
        clean = [sorted({int(i) for i in block if 0 <= int(i) < int(gate.numel())}) for block in blocks]
        clean = [block for block in clean if block]
        if not clean:
            return gate
        scores = []
        for block in clean:
            idx = torch.tensor(block, dtype=torch.long, device=gate.device)
            scores.append(gate[idx].mean())
        score = torch.stack(scores)
        keep = max(1, min(len(clean), int(math.ceil(float(fraction) * len(clean)))))
        top_ids = set(int(i) for i in torch.topk(score, k=keep).indices.detach().cpu().tolist())
        out = torch.zeros_like(gate)
        for block_idx, block in enumerate(clean):
            idx = torch.tensor(block, dtype=torch.long, device=gate.device)
            if block_idx in top_ids:
                out[idx] = gate[idx]
        return out

    @staticmethod
    def _topk_fraction_for_family(family: str) -> float:
        text = str(family).lower()
        if "topk25" in text or "top_k25" in text:
            return 0.25
        if "topk50" in text or "top_k50" in text:
            return 0.50
        return 0.35

    def _gate_from_observed(self, observed: torch.Tensor, weights: torch.Tensor, param: torch.nn.Parameter, group: dict) -> tuple[torch.Tensor, torch.Tensor]:
        white_tensor = self._apply_metric_inv_sqrt(observed, param, group)
        white_obs = white_tensor.reshape(int(observed.shape[0]), -1)
        family = str(group.get("gate_family", "diagonal")).lower()
        random_matched = "random_matched" in family
        base_family = self._base_family_for_random_match(family)
        if "corner_checker_contrast" in base_family or "cornercheckercontrast" in base_family:
            pairs = self._corner_checker_contrast_pairs_for_param(param, group, degree_bands="degree" in base_family)
            if pairs:
                work = white_obs.detach().cpu().to(dtype=torch.float64)
                b, n = int(work.shape[0]), int(work.shape[1])
                mu = work.mean(dim=0)
                var = (work - mu.unsqueeze(0)).square().mean(dim=0)
                gate_cpu = torch.zeros(n, dtype=torch.float64)
                snr_eps = 1.0e-8
                beta = float(group["gate_beta"])
                threshold_log = math.log(1.0)
                for block_a, block_b in pairs:
                    idx_a = torch.tensor(block_a, dtype=torch.long)
                    idx_b = torch.tensor(block_b, dtype=torch.long)
                    snr_a = mu[idx_a].square().sum() / (var[idx_a].sum() / max(1, b - 1) + snr_eps)
                    snr_b = mu[idx_b].square().sum() / (var[idx_b].sum() / max(1, b - 1) + snr_eps)
                    log_a = torch.log(snr_a + snr_eps)
                    log_b = torch.log(snr_b + snr_eps)
                    base_a = torch.sigmoid(beta * (log_a - threshold_log))
                    base_b = torch.sigmoid(beta * (log_b - threshold_log))
                    advantage_a = torch.sigmoid(beta * (log_a - log_b))
                    gate_cpu[idx_a] = base_a * advantage_a
                    gate_cpu[idx_b] = base_b * (1.0 - advantage_a)
                gate = gate_cpu.to(device=param.device, dtype=param.dtype)
                if random_matched and int(gate.numel()) > 0:
                    gate = self._permute_block_gate(gate, [block for pair in pairs for block in pair])
                if float(group.get("gate_floor", 0.0)) > 0.0:
                    floor = float(group["gate_floor"])
                    gate = gate * (1.0 - floor) + floor
                return gate.reshape_as(param), white_obs.detach().mean(dim=0).to(device=param.device, dtype=param.dtype).reshape(-1)
        class_cohort = "class_cohort" in base_family or "classcohort" in base_family
        if class_cohort:
            class_base_family = self._base_family_for_class_cohort(base_family)
            class_group = dict(group)
            class_group["gate_family"] = class_base_family
            labels = self._observed_labels.get(id(param))
            if labels is not None and int(labels.numel()) >= 2:
                label_vec = labels.to(device=white_obs.device).reshape(-1)[: int(white_obs.shape[0])]
                cohort_rows = []
                for label in torch.unique(label_vec.detach()):
                    mask = label_vec == label
                    if int(mask.sum().item()) >= 1:
                        cohort_rows.append(white_obs[mask].mean(dim=0))
                if len(cohort_rows) >= 2:
                    cohort_obs = torch.stack(cohort_rows, dim=0)
                    if self._uses_block_snr_family(class_base_family):
                        blocks = self._blocks_for_param(param, class_group)
                        gres = block_snr_gate(cohort_obs.detach().cpu(), blocks, beta=float(group["gate_beta"]))
                        gate = gres.gate.to(device=param.device, dtype=param.dtype)
                        if random_matched and int(gate.numel()) > 0:
                            gate = self._permute_block_gate(gate, blocks)
                    else:
                        gres = diagonal_snr_gate(cohort_obs.detach().cpu(), beta=float(group["gate_beta"]))
                        gate = gres.gate.to(device=param.device, dtype=param.dtype)
                        if random_matched and int(gate.numel()) > 0:
                            perm = torch.randperm(int(gate.numel()), device=param.device)
                            gate = gate[perm]
                    if float(group.get("gate_floor", 0.0)) > 0.0:
                        floor = float(group["gate_floor"])
                        gate = gate * (1.0 - floor) + floor
                    return gate.reshape_as(param), white_obs.detach().mean(dim=0).to(device=param.device, dtype=param.dtype).reshape(-1)
        class_conditional = "class_conditional" in base_family or "classconditional" in base_family
        if class_conditional:
            class_base_family = self._base_family_for_class_conditional(base_family)
            class_group = dict(group)
            class_group["gate_family"] = class_base_family
            labels = self._observed_labels.get(id(param))
            if labels is not None and int(labels.numel()) >= 2:
                blocks = self._blocks_for_param(param, class_group)

                def class_gate(obs_rows: torch.Tensor) -> torch.Tensor | None:
                    if int(obs_rows.shape[0]) < 2:
                        return None
                    if self._uses_block_snr_family(class_base_family):
                        component = block_snr_gate(obs_rows.detach().cpu(), blocks, beta=float(group["gate_beta"]))
                        gate_component = component.gate.to(device=param.device, dtype=param.dtype)
                        if random_matched and int(gate_component.numel()) > 0:
                            gate_component = self._permute_block_gate(gate_component, blocks)
                        return gate_component
                    component = diagonal_snr_gate(obs_rows.detach().cpu(), beta=float(group["gate_beta"]))
                    gate_component = component.gate.to(device=param.device, dtype=param.dtype)
                    if random_matched and int(gate_component.numel()) > 0:
                        perm = torch.randperm(int(gate_component.numel()), device=param.device)
                        gate_component = gate_component[perm]
                    return gate_component

                label_vec = labels.to(device=white_obs.device).reshape(-1)[: int(white_obs.shape[0])]
                component_gates = [g for g in [class_gate(white_obs)] if g is not None]
                for label in torch.unique(label_vec.detach()):
                    mask = label_vec == label
                    gate_component = class_gate(white_obs[mask])
                    if gate_component is not None:
                        component_gates.append(gate_component)
                if component_gates:
                    gate = torch.stack(component_gates, dim=0).mean(dim=0)
                    if float(group.get("gate_floor", 0.0)) > 0.0:
                        floor = float(group["gate_floor"])
                        gate = gate * (1.0 - floor) + floor
                    return gate.reshape_as(param), white_obs.detach().mean(dim=0).to(device=param.device, dtype=param.dtype).reshape(-1)
        block_group = dict(group)
        block_group["gate_family"] = base_family
        if "corner_checker_diagonal_union" in base_family or "cornercheckerdiagonalunion" in base_family:
            block_sets = self._corner_checker_diagonal_union_block_sets_for_param(param, block_group, degree_bands="degree" in base_family)
            component_gates: list[torch.Tensor] = []
            gres = None
            for blocks in block_sets:
                component = block_snr_gate(white_obs.detach().cpu(), blocks, beta=float(group["gate_beta"]))
                gate_component = component.gate.to(device=param.device, dtype=param.dtype)
                if random_matched and int(gate_component.numel()) > 0:
                    gate_component = self._permute_block_gate(gate_component, blocks)
                component_gates.append(gate_component)
                if gres is None:
                    gres = component
            if component_gates:
                gate = torch.stack(component_gates, dim=0).amax(dim=0)
                mean = gres.mean if gres is not None else white_obs.detach().cpu().mean(dim=0)
                if "topk" in base_family:
                    merged_blocks = [block for blocks in block_sets for block in blocks]
                    gate = self._topk_block_gate(gate, merged_blocks, fraction=self._topk_fraction_for_family(base_family))
            else:
                gres = diagonal_snr_gate(white_obs.detach().cpu(), beta=float(group["gate_beta"]))
                gate = gres.gate.to(device=param.device, dtype=param.dtype)
                mean = gres.mean
        elif self._uses_block_snr_family(base_family):
            blocks = self._blocks_for_param(param, block_group)
            gres = block_snr_gate(white_obs.detach().cpu(), blocks, beta=float(group["gate_beta"]))
            gate = gres.gate.to(device=param.device, dtype=param.dtype)
            if "topk" in base_family:
                gate = self._topk_block_gate(gate, blocks, fraction=self._topk_fraction_for_family(base_family))
            if random_matched and int(gate.numel()) > 0:
                gate = self._permute_block_gate(gate, blocks)
            mean = gres.mean
        else:
            gres = diagonal_snr_gate(white_obs.detach().cpu(), beta=float(group["gate_beta"]))
            gate = gres.gate.to(device=param.device, dtype=param.dtype)
            if random_matched and int(gate.numel()) > 0:
                perm = torch.randperm(int(gate.numel()), device=param.device)
                gate = gate[perm]
            mean = gres.mean
        if float(group.get("gate_floor", 0.0)) > 0.0:
            floor = float(group["gate_floor"])
            gate = gate * (1.0 - floor) + floor
        return gate.reshape_as(param), mean.to(device=param.device, dtype=param.dtype).reshape(-1)

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
                        structure_gate = self._observed_structure_gates.get(id(param))
                        saved_floor = float(group.get("gate_floor", 0.0))
                        floor_mode = str(group.get("gate_floor_mode", "final"))
                        if structure_gate is not None:
                            group["gate_floor"] = 0.0
                        gate, task_mu = self._gate_from_observed(observed, weights, param, group)
                        if structure_gate is not None:
                            group["gate_floor"] = saved_floor
                            if saved_floor > 0.0 and floor_mode == "population":
                                gate = gate * (1.0 - saved_floor) + saved_floor
                            gate = gate * structure_gate.reshape_as(param).to(device=param.device, dtype=param.dtype)
                            if saved_floor > 0.0 and floor_mode != "population":
                                gate = gate * (1.0 - saved_floor) + saved_floor
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
                modifier = self._observed_structure_update_modifiers.get(id(param))
                if modifier is not None:
                    update_white = update_white * modifier.reshape_as(param).to(device=param.device, dtype=param.dtype)
                projectors = self._observed_structure_update_projectors.get(id(param))
                if projectors:
                    flat_update = update_white.reshape(-1)
                    projected = flat_update.clone()
                    accum = torch.zeros_like(flat_update)
                    covered = torch.zeros_like(flat_update, dtype=torch.bool)
                    for idx, direction in projectors:
                        idx_t = idx.to(device=param.device, dtype=torch.long)
                        dir_t = direction.to(device=param.device, dtype=param.dtype).reshape(-1)
                        if int(idx_t.numel()) <= 0 or int(idx_t.numel()) != int(dir_t.numel()):
                            continue
                        denom = torch.dot(dir_t, dir_t).clamp_min(float(group["eps"]))
                        coeff = torch.dot(flat_update[idx_t], dir_t).div(denom)
                        accum[idx_t] += coeff * dir_t
                        covered[idx_t] = True
                    projected[covered] = accum[covered]
                    update_white = projected.reshape_as(update_white)
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
