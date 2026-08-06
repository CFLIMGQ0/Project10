"""Five pathway-wise KAN-like genomic encoders from the design document."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ENCODER_NAMES = (
    "pb_tamlu",
    "do_la",
    "jc_moa",
    "tc_rbf_kan",
    "dd_kac",
)


def _as_batch(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.unsqueeze(0) if inputs.ndim == 1 else inputs


def _activation_stack(inputs: torch.Tensor, include_identity: bool) -> torch.Tensor:
    values = [
        F.elu(inputs),
        F.gelu(inputs),
        F.silu(inputs),
        torch.tanh(inputs),
    ]
    if include_identity:
        values.append(inputs)
    return torch.stack(values, dim=-1)


class GenomicEncoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.auxiliary_loss = torch.tensor(0.0)
        self.diagnostics: dict[str, torch.Tensor] = {}

    def _zero(self, reference: torch.Tensor) -> None:
        self.auxiliary_loss = reference.new_zeros(())


class PBTAMLUEncoder(GenomicEncoderBase):
    """Pathway-budget coupled TAMLU encoder."""

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.25,
        budget: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.first = nn.ModuleList(
            nn.Linear(size, hidden_dim) for size in input_dims
        )
        self.second = nn.ModuleList(
            nn.Linear(hidden_dim, output_dim) for _ in input_dims
        )
        self.norm_first = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in input_dims
        )
        self.norm_second = nn.ModuleList(
            nn.LayerNorm(output_dim) for _ in input_dims
        )
        shape = (len(input_dims), 2)
        self.budget_logits = nn.Parameter(torch.zeros(shape))
        self.direction = nn.Parameter(torch.zeros(shape))
        alpha_initial = math.log((1.0 - 0.25) / (4.0 - 1.0))
        self.alpha_raw = nn.Parameter(torch.full(shape, alpha_initial))
        self.dropout = nn.Dropout(dropout)
        self.budget = budget
        self.temperature = temperature

    def _activate(
        self,
        values: torch.Tensor,
        pathway: int,
        layer: int,
        allocation: torch.Tensor,
    ) -> torch.Tensor:
        beta = (
            self.budget
            * allocation[pathway, layer]
            * torch.tanh(self.direction[pathway, layer])
        )
        alpha = 0.25 + 3.75 * torch.sigmoid(
            self.alpha_raw[pathway, layer]
        )
        return values + beta * values * torch.tanh(alpha * values)

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        allocation = torch.softmax(
            self.budget_logits.flatten() / self.temperature, dim=0
        ).reshape_as(self.budget_logits)
        outputs = []
        for index, inputs in enumerate(pathways):
            hidden = self.norm_first[index](self.first[index](_as_batch(inputs)))
            hidden = self.dropout(self._activate(hidden, index, 0, allocation))
            output = self.norm_second[index](self.second[index](hidden))
            outputs.append(self._activate(output, index, 1, allocation))
        stacked = torch.stack(outputs, dim=1)
        entropy = -(allocation * allocation.clamp_min(1e-8).log()).sum()
        self.auxiliary_loss = -1e-4 * entropy
        beta = self.budget * allocation * torch.tanh(self.direction)
        alpha = 0.25 + 3.75 * torch.sigmoid(self.alpha_raw)
        self.diagnostics = {
            "budget": allocation.detach(),
            "pathway_budget": allocation.detach().sum(dim=1),
            "beta": beta.detach(),
            "alpha": alpha.detach(),
        }
        return stacked


class DistributionOrthogonalActivation(nn.Module):
    """EMA distribution-orthogonalized learnable activation."""

    def __init__(self, ema: float = 0.97, update_every: int = 5) -> None:
        super().__init__()
        experts = 4
        self.logits = nn.Parameter(torch.zeros(experts))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.delta = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("running_mean", torch.zeros(experts))
        self.register_buffer("running_gram", torch.eye(experts))
        self.register_buffer("whitening", torch.eye(experts))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.ema = ema
        self.update_every = update_every
        self.scale_loss = torch.tensor(0.0)

    @torch.no_grad()
    def _update(self, experts: torch.Tensor) -> None:
        flattened = experts.detach().reshape(-1, experts.shape[-1])
        batch_mean = flattened.mean(dim=0)
        centered = flattened - batch_mean
        gram = centered.t().matmul(centered) / max(centered.shape[0], 1)
        gram = gram + torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        ) * 1e-4
        self.running_mean.mul_(self.ema).add_(batch_mean, alpha=1.0 - self.ema)
        self.running_gram.mul_(self.ema).add_(gram, alpha=1.0 - self.ema)
        self.steps.add_(1)
        if self.steps.item() >= 5 and self.steps.item() % self.update_every == 0:
            eigenvalues, eigenvectors = torch.linalg.eigh(self.running_gram)
            inverse_root = eigenvectors @ torch.diag(
                eigenvalues.clamp_min(1e-4).rsqrt()
            ) @ eigenvectors.t()
            self.whitening.copy_(inverse_root)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        experts = _activation_stack(inputs, include_identity=False)
        if self.training:
            self._update(experts)
        centered = experts - self.running_mean
        orthogonal = torch.einsum("...k,kl->...l", centered, self.whitening)
        mixture = torch.softmax(self.logits, dim=0)
        output = self.gamma * torch.einsum("...k,k->...", orthogonal, mixture)
        output = output + self.delta * inputs
        self.scale_loss = (output.var(unbiased=False) - 1.0).square()
        return output


class DOLAEncoder(GenomicEncoderBase):
    """Distribution-orthogonalized learnable activation encoder."""

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.first = nn.ModuleList(nn.Linear(size, hidden_dim) for size in input_dims)
        self.second = nn.ModuleList(nn.Linear(hidden_dim, output_dim) for _ in input_dims)
        self.norm_first = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in input_dims)
        self.norm_second = nn.ModuleList(nn.LayerNorm(output_dim) for _ in input_dims)
        self.activations = nn.ModuleList(
            DistributionOrthogonalActivation() for _ in range(len(input_dims) * 2)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        outputs = []
        losses = []
        for index, inputs in enumerate(pathways):
            first_activation = self.activations[index * 2]
            second_activation = self.activations[index * 2 + 1]
            hidden = first_activation(
                self.norm_first[index](self.first[index](_as_batch(inputs)))
            )
            hidden = self.dropout(hidden)
            output = second_activation(
                self.norm_second[index](self.second[index](hidden))
            )
            outputs.append(output)
            losses.extend((first_activation.scale_loss, second_activation.scale_loss))
        stacked = torch.stack(outputs, dim=1)
        self.auxiliary_loss = 1e-4 * torch.stack(losses).mean()
        self.diagnostics = {
            "mixture_weights": torch.stack(
                [torch.softmax(module.logits.detach(), dim=0) for module in self.activations]
            ),
            "gram_condition": torch.stack(
                [torch.linalg.cond(module.running_gram.detach()) for module in self.activations]
            ),
        }
        return stacked


class JCMoAEncoder(GenomicEncoderBase):
    """Patient-conditioned mixture of activations with Jacobian calibration."""

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.25,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        count = len(input_dims)
        self.first = nn.ModuleList(nn.Linear(size, hidden_dim) for size in input_dims)
        self.second = nn.ModuleList(nn.Linear(hidden_dim, output_dim) for _ in input_dims)
        self.norm_first = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in input_dims)
        self.norm_second = nn.ModuleList(nn.LayerNorm(output_dim) for _ in input_dims)
        self.pathway_embedding = nn.Embedding(count, embedding_dim)
        self.static_logits = nn.Parameter(torch.zeros(count, 2, 5))
        self.routers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(5 + embedding_dim, 32),
                nn.GELU(),
                nn.Linear(32, 5),
            )
            for _ in range(2)
        )
        self.dropout = nn.Dropout(dropout)
        self._route_weights: list[torch.Tensor] = []
        self._jacobian_losses: list[torch.Tensor] = []
        self._prior_losses: list[torch.Tensor] = []

    @staticmethod
    def _stats(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values.mean(dim=1),
                values.std(dim=1, unbiased=False),
                values.abs().mean(dim=1),
                values.max(dim=1).values,
                values.min(dim=1).values,
            ),
            dim=1,
        )

    @staticmethod
    def _derivatives(values: torch.Tensor) -> torch.Tensor:
        elu = torch.where(values > 0, torch.ones_like(values), values.exp())
        cdf = 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))
        density = values.mul(-0.5).mul(values).exp() / math.sqrt(2.0 * math.pi)
        gelu = cdf + values * density
        sigmoid = torch.sigmoid(values)
        silu = sigmoid * (1.0 + values * (1.0 - sigmoid))
        tanh = 1.0 - torch.tanh(values).square()
        identity = torch.ones_like(values)
        return torch.stack((elu, gelu, silu, tanh, identity), dim=-1)

    def _activate(self, values: torch.Tensor, pathway: int, layer: int) -> torch.Tensor:
        batch = values.shape[0]
        path_ids = torch.full(
            (batch,), pathway, device=values.device, dtype=torch.long
        )
        router_input = torch.cat(
            (self._stats(values), self.pathway_embedding(path_ids)), dim=1
        )
        prior_logits = self.static_logits[pathway, layer]
        logits = prior_logits.unsqueeze(0) + self.routers[layer](router_input)
        route = torch.softmax(logits, dim=1)
        experts = _activation_stack(values, include_identity=True)
        output = torch.einsum("bdk,bk->bd", experts, route)

        derivatives = self._derivatives(values)
        mixed_derivative = torch.einsum("bdk,bk->bd", derivatives, route.detach())
        reference = derivatives[..., 0]
        jacobian_loss = (
            mixed_derivative.square().mean(dim=1)
            - reference.square().mean(dim=1)
        ).square().mean()
        prior = torch.softmax(prior_logits, dim=0).unsqueeze(0)
        prior_loss = (
            route * (route.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())
        ).sum(dim=1).mean()
        self._route_weights.append(route)
        self._jacobian_losses.append(jacobian_loss)
        self._prior_losses.append(prior_loss)
        return output

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        self._route_weights = []
        self._jacobian_losses = []
        self._prior_losses = []
        outputs = []
        for index, inputs in enumerate(pathways):
            hidden = self.norm_first[index](self.first[index](_as_batch(inputs)))
            hidden = self.dropout(self._activate(hidden, index, 0))
            output = self.norm_second[index](self.second[index](hidden))
            outputs.append(self._activate(output, index, 1))
        stacked = torch.stack(outputs, dim=1)
        jacobian = torch.stack(self._jacobian_losses).mean()
        prior = torch.stack(self._prior_losses).mean()
        self.auxiliary_loss = 1e-3 * jacobian + 1e-4 * prior
        routes = torch.stack(self._route_weights, dim=1)
        entropy = -(routes * routes.clamp_min(1e-8).log()).sum(dim=-1)
        self.diagnostics = {
            "route_weights": routes.detach(),
            "route_entropy": entropy.detach(),
            "jacobian_energy_loss": jacobian.detach(),
        }
        return stacked


class TCRBFLinear(nn.Module):
    """Low-rank RBF edge functions on an ordered transport grid."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        centers: int = 12,
        rank: int = 16,
        nearest: int = 3,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.output_factor = nn.Parameter(torch.empty(out_features, rank))
        self.input_factor = nn.Parameter(torch.empty(in_features, rank))
        self.templates = nn.Parameter(torch.empty(rank, centers))
        self.transport_raw = nn.Parameter(torch.zeros(centers))
        self.width_raw = nn.Parameter(torch.zeros(centers))
        self.centers_count = centers
        self.nearest = nearest
        self.coverage_loss = torch.tensor(0.0)
        nn.init.normal_(self.output_factor, std=0.02)
        nn.init.normal_(self.input_factor, std=0.02)
        nn.init.normal_(self.templates, std=0.02)

    def grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        increments = F.softplus(self.transport_raw) + 1e-4
        centers = -1.0 + 2.0 * increments.cumsum(0) / increments.sum()
        left = torch.cat((centers[:1], centers[:-1]))
        right = torch.cat((centers[1:], centers[-1:]))
        spacing = (right - left).abs().clamp_min(0.05) * 0.5
        kappa = 0.5 + 1.5 * torch.sigmoid(self.width_raw)
        return centers, spacing * kappa

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        centers, widths = self.grid()
        distances = (inputs.unsqueeze(-1) - centers).abs()
        nearest_indices = distances.topk(
            min(self.nearest, self.centers_count), dim=-1, largest=False
        ).indices
        mask = torch.zeros_like(distances).scatter_(-1, nearest_indices, 1.0)
        basis = torch.exp(
            -0.5 * ((inputs.unsqueeze(-1) - centers) / widths.clamp_min(1e-4)).square()
        ) * mask
        rank_features = torch.einsum(
            "bik,ir,rk->br", basis, self.input_factor, self.templates
        )
        nonlinear = rank_features.matmul(self.output_factor.t())
        normalized_distance = distances / widths.clamp_min(1e-4)
        self.coverage_loss = normalized_distance.min(dim=-1).values.square().mean()
        return self.linear(inputs) + nonlinear


class TCRBFKANEncoder(GenomicEncoderBase):
    """Transport-constrained Free-RBF-KAN pathway encoder."""

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.rbf = nn.ModuleList(
            TCRBFLinear(size, hidden_dim) for size in input_dims
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in input_dims)
        self.outputs = nn.ModuleList(
            nn.Linear(hidden_dim, output_dim) for _ in input_dims
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for index, inputs in enumerate(pathways):
            hidden = self.rbf[index](_as_batch(inputs))
            hidden = self.dropout(self.norms[index](hidden))
            outputs.append(self.outputs[index](hidden))
        stacked = torch.stack(outputs, dim=1)
        coverage = torch.stack([module.coverage_loss for module in self.rbf])
        self.auxiliary_loss = 1e-5 * coverage.mean()
        grids = [module.grid() for module in self.rbf]
        self.diagnostics = {
            "min_center_spacing": torch.stack(
                [torch.diff(center).min() for center, _ in grids]
            ).detach(),
            "mean_width": torch.stack(
                [width.mean() for _, width in grids]
            ).detach(),
            "coverage": coverage.detach(),
        }
        return stacked


class DDKACPathway(nn.Module):
    """Value-domain KAN plus patient-adaptive Chebyshev graph filtering."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adjacency: torch.Tensor,
        order: int = 2,
    ) -> None:
        super().__init__()
        adjacency = adjacency.float().clone()
        adjacency.fill_diagonal_(1.0)
        degree = adjacency.sum(dim=1).clamp_min(1e-6)
        normalized = adjacency / degree.sqrt().unsqueeze(0) / degree.sqrt().unsqueeze(1)
        laplacian = torch.eye(input_dim) - normalized
        self.register_buffer("laplacian", laplacian)
        self.register_buffer("scaled_laplacian", laplacian - torch.eye(input_dim))
        self.value_linear = nn.Parameter(torch.tensor(1.0))
        self.value_coefficients = nn.Parameter(torch.zeros(8))
        self.register_buffer("value_centers", torch.linspace(-1.0, 1.0, 8))
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.structure_projection = nn.Linear(input_dim, output_dim)
        self.residual_projection = nn.Linear(input_dim, output_dim)
        self.router = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, order + 1))
        self.gate = nn.Linear(5, 1)
        self.norm = nn.LayerNorm(output_dim)
        self.order = order
        self.last_gate = torch.tensor(0.0)
        self.last_filter = torch.tensor(0.0)
        self.consistency_loss = torch.tensor(0.0)

    def _stats(self, inputs: torch.Tensor) -> torch.Tensor:
        centered = inputs - inputs.mean(dim=1, keepdim=True)
        filtered = centered.matmul(self.laplacian.t())
        energy = (centered * filtered).sum(dim=1) / centered.square().sum(dim=1).clamp_min(1e-6)
        sparsity = (inputs.abs() < 1e-6).float().mean(dim=1)
        burden = (inputs.abs() > 0.5).float().mean(dim=1)
        return torch.stack(
            (inputs.mean(dim=1), inputs.std(dim=1, unbiased=False), sparsity, energy, burden),
            dim=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stats = self._stats(inputs)
        widths = 2.0 / (self.value_centers.numel() - 1)
        basis = torch.exp(
            -0.5 * ((inputs.unsqueeze(-1) - self.value_centers) / widths).square()
        )
        transformed = self.value_linear * inputs + torch.einsum(
            "bik,k->bi", basis, self.value_coefficients
        )
        value_features = self.value_projection(transformed)

        centered = inputs - inputs.mean(dim=1, keepdim=True)
        responses = [centered]
        if self.order >= 1:
            responses.append(centered.matmul(self.scaled_laplacian.t()))
        for _ in range(2, self.order + 1):
            responses.append(
                2.0 * responses[-1].matmul(self.scaled_laplacian.t()) - responses[-2]
            )
        filter_weights = torch.softmax(self.router(stats), dim=1)
        filtered = sum(
            filter_weights[:, index : index + 1] * response
            for index, response in enumerate(responses)
        )
        structure_features = self.structure_projection(filtered)
        gate = torch.sigmoid(self.gate(stats))
        output = self.norm(
            gate * value_features
            + (1.0 - gate) * structure_features
            + self.residual_projection(inputs)
        )
        self.consistency_loss = (1.0 - F.cosine_similarity(
            value_features, structure_features, dim=1
        )).mean()
        self.last_gate = gate.detach()
        self.last_filter = filter_weights.detach()
        return output


class DDKACEncoder(GenomicEncoderBase):
    """Dual-domain gene KAC using fold-specific gene-relation graphs."""

    def __init__(
        self,
        input_dims: Sequence[int],
        gene_graphs: Sequence[torch.Tensor],
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        if gene_graphs is None or len(gene_graphs) != len(input_dims):
            raise ValueError("DD-KAC requires one training-fold gene graph per pathway")
        self.pathways = nn.ModuleList(
            DDKACPathway(size, output_dim, graph)
            for size, graph in zip(input_dims, gene_graphs)
        )

    def forward(self, pathways: Sequence[torch.Tensor]) -> torch.Tensor:
        outputs = [
            module(_as_batch(inputs))
            for module, inputs in zip(self.pathways, pathways)
        ]
        stacked = torch.stack(outputs, dim=1)
        consistency = torch.stack(
            [module.consistency_loss for module in self.pathways]
        )
        self.auxiliary_loss = 1e-4 * consistency.mean()
        self.diagnostics = {
            "domain_gate": torch.cat(
                [module.last_gate for module in self.pathways], dim=1
            ),
            "filter_weights": torch.stack(
                [module.last_filter for module in self.pathways], dim=1
            ),
            "consistency": consistency.detach(),
        }
        return stacked


def build_genomic_encoder(
    name: str,
    input_dims: Sequence[int],
    hidden_dim: int = 1024,
    output_dim: int = 256,
    dropout: float = 0.25,
    gene_graphs: Sequence[torch.Tensor] | None = None,
    pc_cmka_config: dict | None = None,
) -> GenomicEncoderBase:
    if name == "pb_tamlu":
        return PBTAMLUEncoder(input_dims, hidden_dim, output_dim, dropout)
    if name == "do_la":
        return DOLAEncoder(input_dims, hidden_dim, output_dim, dropout)
    if name == "jc_moa":
        return JCMoAEncoder(input_dims, hidden_dim, output_dim, dropout)
    if name == "tc_rbf_kan":
        return TCRBFKANEncoder(input_dims, hidden_dim, output_dim, dropout)
    if name == "dd_kac":
        return DDKACEncoder(input_dims, gene_graphs, output_dim)
    if name == "pc_cmka_ddkac":
        if pc_cmka_config is None:
            raise ValueError("PC-CMKA-DDKAC requires a resolved configuration")
        from models.layers.pc_cmka import PCCMKADDKACEncoder

        return PCCMKADDKACEncoder(
            input_dims=input_dims,
            gene_graphs=gene_graphs,
            config=pc_cmka_config,
            output_dim=output_dim,
        )
    raise ValueError(f"Unknown genomic encoder: {name}")
