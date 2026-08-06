"""DD-KAC-compatible patient graph calibration encoder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentation import GraphViewAugmenter
from .calibration import CalibrationResult, ChebyshevInverseCalibrator
from .losses import (
    cosine_variance_consistency,
    gate_consistency,
    squared_tangent_cosine,
)
from .spectral import ReferenceSpectralOperator


def _as_batch(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.unsqueeze(0) if inputs.ndim == 1 else inputs


class PCCMKADDKACPathway(nn.Module):
    """One functional group with Word-aligned graph calibration and DD-KAC."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adjacency: torch.Tensor,
        config: dict[str, Any],
        pathway_index: int,
    ) -> None:
        super().__init__()
        spectral = config["spectral"]
        self.operator = ReferenceSpectralOperator.from_adjacency(
            adjacency,
            spectral_bound=spectral.get("spectral_bound"),
            max_log_deformation=float(config["solver"]["rho_max"]),
            normalization=str(spectral.get("normalization", "reference")),
            eps=float(spectral["epsilon"]),
        )
        self.calibrator = ChebyshevInverseCalibrator(input_dim, self.operator, config)
        self.augmenter = GraphViewAugmenter(self.operator, self.calibrator, config)
        self.order = int(spectral["moment_order"])
        self.pathway_index = int(pathway_index)
        self.probe_mode = str(config.get("probe_mode", "functional"))
        self.shared_route = bool(config["augmentation"].get("shared_route", True))
        self.ssl_mode = str(config["ssl"].get("mode", "structure_fusion"))
        self.identifiability_mode = str(config["identifiability"].get("mode", "off"))
        self.identifiability_probes = int(
            config["identifiability"].get("random_probes", 1)
        )
        self.loss_weights = dict(config["loss"])

        self.value_linear = nn.Parameter(torch.tensor(1.0))
        self.value_coefficients = nn.Parameter(torch.zeros(8))
        self.register_buffer("value_centers", torch.linspace(-1.0, 1.0, 8))
        generator = torch.Generator().manual_seed(9173 + pathway_index)
        random_probe = torch.randn(input_dim, generator=generator)
        self.register_buffer("random_probe", F.normalize(random_probe, dim=0))

        self.value_projection = nn.Linear(input_dim, output_dim)
        self.structure_projection = nn.Linear(input_dim, output_dim)
        self.residual_projection = nn.Linear(input_dim, output_dim)
        self.router = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, self.order + 1)
        )
        self.gate = nn.Linear(5, 1)
        self.norm = nn.LayerNorm(output_dim)
        self.auxiliary_loss = torch.tensor(0.0)
        self.loss_components: dict[str, torch.Tensor] = {}
        self.diagnostics: dict[str, Any] = {}

    def _probe(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.probe_mode == "functional":
            # Within one signature, m_g is the all-one mask by construction.
            return F.normalize(inputs, p=2, dim=1, eps=1e-6)
        if self.probe_mode == "random":
            return self.random_probe.unsqueeze(0).expand(inputs.shape[0], -1)
        raise ValueError(f"unknown probe_mode: {self.probe_mode}")

    def _value_branch(self, inputs: torch.Tensor) -> torch.Tensor:
        widths = 2.0 / max(self.value_centers.numel() - 1, 1)
        basis = torch.exp(
            -0.5 * ((inputs.unsqueeze(-1) - self.value_centers) / widths).square()
        )
        transformed = self.value_linear * inputs + torch.einsum(
            "bik,k->bi", basis, self.value_coefficients
        )
        return self.value_projection(transformed)

    def _graph_stats(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        centered = inputs - inputs.mean(dim=1, keepdim=True)
        filtered = self.operator.apply_operator(weights, centered)
        energy = (centered * filtered).sum(dim=1) / centered.square().sum(
            dim=1
        ).clamp_min(1e-6)
        sparsity = (inputs.abs() < 1e-6).float().mean(dim=1)
        burden = (inputs.abs() > 0.5).float().mean(dim=1)
        return torch.stack(
            (
                inputs.mean(dim=1),
                inputs.std(dim=1, unbiased=False),
                sparsity,
                energy,
                burden,
            ),
            dim=1,
        )

    def _structure_branch(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centered = inputs - inputs.mean(dim=1, keepdim=True)
        responses = self.operator.chebyshev_responses(weights, centered, self.order)
        route = torch.softmax(route_logits, dim=1)
        filtered = sum(
            route[:, index : index + 1] * response
            for index, response in enumerate(responses)
        )
        return self.structure_projection(filtered), route

    def _fuse(
        self,
        value: torch.Tensor,
        structure: torch.Tensor,
        inputs: torch.Tensor,
        stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(stats))
        fused = self.norm(
            gate * value
            + (1.0 - gate) * structure
            + self.residual_projection(inputs)
        )
        return fused, gate

    def _identifiability_loss(
        self,
        inputs: torch.Tensor,
        calibration: CalibrationResult,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.identifiability_mode == "off" or not self.training:
            return inputs.new_zeros(())
        coefficients = calibration.coefficients
        if not coefficients.requires_grad:
            coefficients = coefficients.detach().requires_grad_(True)
        if not route_logits.requires_grad:
            route_logits = route_logits.detach().requires_grad_(True)

        def from_coefficients(values: torch.Tensor) -> torch.Tensor:
            weights, _ = self.calibrator.weights_from_coefficients(values)
            return self._structure_branch(inputs, weights, route_logits)[0]

        def from_route(values: torch.Tensor) -> torch.Tensor:
            return self._structure_branch(inputs, calibration.weights, values)[0]

        losses = []
        for _ in range(self.identifiability_probes):
            direction_a = F.normalize(torch.randn_like(coefficients), dim=1)
            direction_theta = F.normalize(torch.randn_like(route_logits), dim=1)
            tangent_a = torch.autograd.functional.jvp(
                from_coefficients,
                coefficients,
                direction_a,
                create_graph=True,
            )[1]
            tangent_theta = torch.autograd.functional.jvp(
                from_route,
                route_logits,
                direction_theta,
                create_graph=True,
            )[1]
            losses.append(squared_tangent_cosine(tangent_a, tangent_theta))
        return torch.stack(losses).mean()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = _as_batch(inputs).float()
        probes = self._probe(inputs)
        calibration = self.calibrator(inputs, probes)
        value = self._value_branch(inputs)
        base_stats = self._graph_stats(inputs, calibration.weights)
        base_logits = self.router(base_stats)
        structure, base_route = self._structure_branch(
            inputs, calibration.weights, base_logits
        )
        fused, base_gate = self._fuse(value, structure, inputs, base_stats)

        views = self.augmenter(calibration, probes)
        ssl_structure = inputs.new_zeros(())
        ssl_fusion = inputs.new_zeros(())
        ssl_gate = inputs.new_zeros(())
        positive_route = base_route
        negative_route = base_route
        positive_gate = base_gate
        negative_gate = base_gate
        if self.training and str(self.augmenter.settings.get("mode", "off")) != "off":
            positive_stats = self._graph_stats(inputs, views.positive)
            negative_stats = self._graph_stats(inputs, views.negative)
            if self.shared_route:
                positive_logits = negative_logits = base_logits
            else:
                positive_logits = self.router(positive_stats)
                negative_logits = self.router(negative_stats)
            positive_structure, positive_route = self._structure_branch(
                inputs, views.positive, positive_logits
            )
            negative_structure, negative_route = self._structure_branch(
                inputs, views.negative, negative_logits
            )
            positive_fusion, positive_gate = self._fuse(
                value, positive_structure, inputs, positive_stats
            )
            negative_fusion, negative_gate = self._fuse(
                value, negative_structure, inputs, negative_stats
            )
            ssl_structure = cosine_variance_consistency(
                positive_structure, negative_structure
            )
            ssl_fusion = cosine_variance_consistency(
                positive_fusion, negative_fusion
            )
            ssl_gate = gate_consistency(positive_gate, negative_gate)

        if self.ssl_mode == "off":
            ssl = inputs.new_zeros(())
        elif self.ssl_mode == "structure":
            ssl = float(self.loss_weights["lambda_structure"]) * ssl_structure
        elif self.ssl_mode == "fusion":
            ssl = float(self.loss_weights["lambda_fusion"]) * ssl_fusion
        elif self.ssl_mode in {"structure_fusion", "structure_fusion_gate"}:
            ssl = (
                float(self.loss_weights["lambda_structure"]) * ssl_structure
                + float(self.loss_weights["lambda_fusion"]) * ssl_fusion
            )
            if self.ssl_mode == "structure_fusion_gate":
                ssl = ssl + float(self.loss_weights["lambda_gate"]) * ssl_gate
        else:
            raise ValueError(f"unknown SSL mode: {self.ssl_mode}")

        identifiability = self._identifiability_loss(
            inputs, calibration, base_logits
        )
        components = {
            "moment": calibration.moment_loss,
            "trust": calibration.trust_loss,
            "ssl": ssl,
            "identifiability": identifiability,
            "dictionary": calibration.dictionary_loss,
        }
        self.loss_components = components
        self.auxiliary_loss = (
            float(self.loss_weights["lambda_moment"]) * components["moment"]
            + float(self.loss_weights["lambda_trust"]) * components["trust"]
            + float(self.loss_weights["lambda_ssl"]) * components["ssl"]
            + float(self.loss_weights["lambda_id"]) * components["identifiability"]
            + float(self.loss_weights["lambda_dictionary"]) * components["dictionary"]
        )

        prior = self.operator.prior_weights.unsqueeze(0)
        log_change = torch.log(calibration.weights / prior.clamp_min(1e-8))
        top_count = min(10, log_change.shape[1])
        increase = torch.topk(log_change, top_count, dim=1)
        decrease = torch.topk(-log_change, top_count, dim=1)
        self.diagnostics = {
            "rho": calibration.rho.detach(),
            "target_moments": calibration.target_moments.detach(),
            "actual_moments": calibration.actual_moments.detach(),
            "moment_residual": (
                calibration.actual_moments - calibration.target_moments
            ).detach(),
            "precision": calibration.precision.detach(),
            "solver_convergence": calibration.convergence.detach(),
            "solve_seconds": calibration.rho.new_tensor(calibration.solve_seconds),
            "log_edge_change_mean": log_change.mean(dim=1).detach(),
            "log_edge_change_std": log_change.std(dim=1, unbiased=False).detach(),
            "top_increase_edge": self.operator.edge_index[:, increase.indices[0]].detach(),
            "top_increase_value": increase.values.detach(),
            "top_decrease_edge": self.operator.edge_index[:, decrease.indices[0]].detach(),
            "top_decrease_value": (-decrease.values).detach(),
            "hessian": views.hessian_summary.detach(),
            "positive_edge_delta": views.positive_delta.detach(),
            "negative_edge_delta": views.negative_delta.detach(),
            "antithetic": views.is_antithetic,
            "augmentation_scale": views.scale.detach(),
            "krylov_before": views.krylov_before.detach(),
            "krylov_after": views.krylov_after.detach(),
            "route_base": base_route.detach(),
            "route_positive": positive_route.detach(),
            "route_negative": negative_route.detach(),
            "gate_base": base_gate.detach(),
            "gate_positive": positive_gate.detach(),
            "gate_negative": negative_gate.detach(),
            "tangent_correlation": identifiability.detach(),
            "loss_moment": components["moment"].detach(),
            "loss_trust": components["trust"].detach(),
            "loss_ssl": components["ssl"].detach(),
            "loss_identifiability": components["identifiability"].detach(),
            "loss_dictionary": components["dictionary"].detach(),
        }
        return fused


class PCCMKADDKACEncoder(nn.Module):
    """Six-group PC-CMKA-DDKAC encoder preserving the MRePath token API."""

    def __init__(
        self,
        input_dims: Sequence[int],
        gene_graphs: Sequence[torch.Tensor],
        config: dict[str, Any],
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(input_dims) != 6:
            raise ValueError("PC-CMKA-DDKAC requires exactly six functional groups")
        if gene_graphs is None or len(gene_graphs) != len(input_dims):
            raise ValueError("one fixed prior graph is required per functional group")
        self.pathways = nn.ModuleList(
            PCCMKADDKACPathway(
                int(size), output_dim, graph, deepcopy(config), index
            )
            for index, (size, graph) in enumerate(zip(input_dims, gene_graphs))
        )
        self.auxiliary_loss = torch.tensor(0.0)
        self.loss_components: dict[str, torch.Tensor] = {}
        self.diagnostics: dict[str, Any] = {}

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(pathways) != 6:
            raise ValueError("six pathway tensors are required")
        outputs = [module(values) for module, values in zip(self.pathways, pathways)]
        stacked = torch.stack(outputs, dim=1)
        names = ("moment", "trust", "ssl", "identifiability", "dictionary")
        self.loss_components = {
            name: torch.stack([module.loss_components[name] for module in self.pathways]).mean()
            for name in names
        }
        self.auxiliary_loss = torch.stack(
            [module.auxiliary_loss for module in self.pathways]
        ).mean()
        self.diagnostics = {
            f"group_{index}": module.diagnostics
            for index, module in enumerate(self.pathways)
        }
        self.diagnostics["loss_components"] = {
            name: value.detach() for name, value in self.loss_components.items()
        }
        return stacked
