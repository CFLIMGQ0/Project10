"""High-level six-group PC-CMKA-DDKAC encoder."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.pc_cmka_augmentation import (
    AntitheticAugmentationResult,
    CalibrationUncertaintyAugmentor,
    ControlGraphAugmentor,
)
from models.layers.pc_cmka_calibration import (
    CalibrationResult,
    DifferentiableMomentSolver,
    DirectPatientEdgeGate,
    EdgeDeformationDictionary,
    MomentTargetNetwork,
)
from models.layers.pc_cmka_ddkac_core import (
    IdentifiabilityTangentRegularizer,
    SharedRouteDDKACPathway,
    SharedRoutePathwayResult,
    negative_free_consistency_loss,
)
from models.layers.pc_cmka_spectral import (
    ChebyshevMomentResponse,
    ReferenceSpectralOperator,
)

class PCCMKAPathwayEncoder(nn.Module):
    """All PC-CMKA-DDKAC operations for one functional gene group."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_index: torch.Tensor,
        prior_weights: torch.Tensor,
        *,
        shared_lambda: float,
        moment_order: int = 2,
        probe_epsilon: float = 1e-8,
        dictionary_rank: int = 8,
        dictionary_trainable: bool = True,
        target_hidden_dim: int = 64,
        target_max_offset: float = 0.5,
        target_min_precision: float = 1e-4,
        solver_iterations: int = 5,
        solver_step_size: float = 0.1,
        solver_beta: float = 1e-2,
        solver_lambda_rho: float = 1e-3,
        solver_gradient_clip: float = 10.0,
        rho_max: float = 0.5,
        calibration_mode: str = "joint",
        augmentation_mode: str = "hessian_antithetic",
        hessian_mode: str = "diagonal",
        hessian_low_rank: int = 2,
        hessian_damping: float = 1e-5,
        xi_max: float = 0.2,
        krylov_eta: float = 0.1,
        krylov_enabled: bool = True,
        shared_route: bool = True,
        random_drop_probability: float = 0.1,
        identifiability_mode: str = "randomized",
        identifiability_probes: int = 1,
    ) -> None:
        super().__init__()
        if calibration_mode not in (
            DifferentiableMomentSolver.VALID_MODES | {"direct_edge_gate"}
        ):
            raise ValueError("invalid calibration mode")
        valid_augmentation = {
            "off",
            "hessian_antithetic",
            *ControlGraphAugmentor.VALID_MODES,
        }
        if augmentation_mode not in valid_augmentation:
            raise ValueError("invalid augmentation mode")
        operator = ReferenceSpectralOperator(
            edge_index,
            prior_weights,
            input_dim,
            lambda_max=shared_lambda,
            max_log_deformation=rho_max,
        )
        self.moment_response = ChebyshevMomentResponse(
            operator,
            order=moment_order,
            probe_epsilon=probe_epsilon,
        )
        self.dictionary = EdgeDeformationDictionary(
            prior_weights,
            rank=min(dictionary_rank, prior_weights.numel()),
            trainable=dictionary_trainable,
        )
        self.target_network = MomentTargetNetwork(
            input_dim,
            moment_order=moment_order,
            hidden_dim=target_hidden_dim,
            max_offset=target_max_offset,
            min_precision=target_min_precision,
        )
        self.direct_edge_gate = DirectPatientEdgeGate(
            input_dim,
            prior_weights.numel(),
            rho_max=rho_max,
        )
        self.solver = DifferentiableMomentSolver(
            self.moment_response,
            self.dictionary,
            iterations=solver_iterations,
            step_size=solver_step_size,
            beta=solver_beta,
            lambda_rho=solver_lambda_rho,
            rho_max=rho_max,
            gradient_clip=solver_gradient_clip,
        )
        self.augmentor = CalibrationUncertaintyAugmentor(
            self.moment_response,
            self.dictionary,
            beta=solver_beta,
            damping=hessian_damping,
            mode=hessian_mode,
            low_rank=hessian_low_rank,
            xi_max=xi_max,
            krylov_order=moment_order,
            krylov_eta=krylov_eta,
            use_krylov_safety=krylov_enabled,
        )
        if augmentation_mode in ControlGraphAugmentor.VALID_MODES:
            self.control_augmentor = ControlGraphAugmentor(
                operator,
                mode=augmentation_mode,
                drop_probability=random_drop_probability,
            )
        else:
            self.control_augmentor = None
        self.ddkac = SharedRouteDDKACPathway(
            input_dim,
            output_dim,
            operator,
            order=moment_order,
        )
        self.identifiability = IdentifiabilityTangentRegularizer(
            self.moment_response,
            self.dictionary,
            self.ddkac,
            mode=identifiability_mode,
            random_probes=identifiability_probes,
        )
        self.calibration_mode = calibration_mode
        self.augmentation_mode = augmentation_mode
        self.shared_route = bool(shared_route)
        if calibration_mode == "target_only":
            self.dictionary.matrix.requires_grad_(False)
            for parameter in self.direct_edge_gate.parameters():
                parameter.requires_grad_(False)

    def _direct_calibration(
        self, expression: torch.Tensor
    ) -> CalibrationResult:
        log_deformation = self.direct_edge_gate(expression)
        weights = self.moment_response.operator.patient_weights(log_deformation)
        prior_moments = self.moment_response(expression)
        actual_moments = self.moment_response(expression, weights)
        precision = torch.ones_like(actual_moments)
        coefficients = expression.new_zeros(
            expression.shape[0], self.dictionary.rank
        )
        rho = log_deformation.abs().amax(dim=1)
        zeros = actual_moments.new_zeros(actual_moments.shape)
        return CalibrationResult(
            coefficients=coefficients,
            log_deformation=log_deformation,
            weights=weights,
            prior_moments=prior_moments,
            target_moments=actual_moments,
            actual_moments=actual_moments,
            precision=precision,
            residual=zeros,
            rho=rho,
            coefficient_norm=coefficients.norm(dim=1),
            objective_history=zeros.new_zeros(1, expression.shape[0]),
            moment_loss=zeros.mean(),
            trust_loss=rho.mean(),
            dictionary_loss=self.dictionary.orthogonality_loss(),
            finite=torch.stack(
                [torch.isfinite(weights).all(), torch.isfinite(actual_moments).all()]
            ).all(),
        )

    def forward(
        self,
        expression: torch.Tensor,
        *,
        enable_augmentation: bool,
        enable_identifiability: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[
        SharedRoutePathwayResult,
        CalibrationResult,
        AntitheticAugmentationResult,
        torch.Tensor,
    ]:
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)
        if self.calibration_mode == "direct_edge_gate":
            calibration = self._direct_calibration(expression)
        else:
            offset, precision = self.target_network(expression)
            calibration = self.solver(
                expression,
                offset,
                precision,
                mode=self.calibration_mode,
            )
        if enable_augmentation:
            if self.augmentation_mode == "hessian_antithetic":
                augmentation = self.augmentor(
                    expression, calibration, generator=generator
                )
            elif self.control_augmentor is not None:
                augmentation = self.control_augmentor(
                    calibration.weights, generator=generator
                )
            else:
                was_training = self.augmentor.training
                self.augmentor.eval()
                augmentation = self.augmentor(expression, calibration)
                self.augmentor.train(was_training)
        else:
            was_training = self.augmentor.training
            self.augmentor.eval()
            augmentation = self.augmentor(expression, calibration)
            self.augmentor.train(was_training)
        pathway = self.ddkac(
            expression,
            calibration.weights,
            augmentation.positive_weights,
            augmentation.negative_weights,
            shared_route=self.shared_route,
        )
        if enable_identifiability:
            identifiability_loss = self.identifiability(
                expression, calibration.coefficients, pathway.route_logits
            )
        else:
            identifiability_loss = expression.new_zeros(())
        return pathway, calibration, augmentation, identifiability_loss


class PCCMKADDKACEncoder(nn.Module):
    """Six-group PC-CMKA-DDKAC encoder preserving the ``[B, 6, d]`` API."""

    def __init__(
        self,
        input_dims: list[int] | tuple[int, ...],
        graph_priors,
        output_dim: int = 256,
        *,
        moment_order: int = 2,
        probe_epsilon: float = 1e-8,
        dictionary_rank: int = 8,
        dictionary_trainable: bool = True,
        target_hidden_dim: int = 64,
        target_max_offset: float = 0.5,
        target_min_precision: float = 1e-4,
        solver_iterations: int = 5,
        solver_step_size: float = 0.1,
        solver_beta: float = 1e-2,
        solver_lambda_rho: float = 1e-3,
        solver_gradient_clip: float = 10.0,
        rho_max: float = 0.5,
        calibration_mode: str = "joint",
        augmentation_mode: str = "hessian_antithetic",
        hessian_mode: str = "diagonal",
        hessian_low_rank: int = 2,
        hessian_damping: float = 1e-5,
        xi_max: float = 0.2,
        krylov_eta: float = 0.1,
        krylov_enabled: bool = True,
        shared_route: bool = True,
        random_drop_probability: float = 0.1,
        identifiability_mode: str = "randomized",
        identifiability_probes: int = 1,
        lambda_moment: float = 1.0,
        lambda_trust: float = 1e-3,
        lambda_ssl: float = 0.1,
        lambda_id: float = 1e-3,
        lambda_dict: float = 1e-3,
        lambda_ddkac_consistency: float = 1e-4,
        ssl_variance_weight: float = 0.0,
        ssl_covariance_weight: float = 0.0,
        ssl_structure_weight: float = 1.0,
        ssl_fusion_weight: float = 1.0,
        ssl_gate_weight: float = 1.0,
        edge_histogram_bins: int = 20,
        top_edges: int = 10,
    ) -> None:
        super().__init__()
        if len(input_dims) != 6 or len(graph_priors) != 6:
            raise ValueError("PC-CMKA-DDKAC requires exactly six graph priors")
        loss_weights = (
            lambda_moment,
            lambda_trust,
            lambda_ssl,
            lambda_id,
            lambda_dict,
            lambda_ddkac_consistency,
            ssl_variance_weight,
            ssl_covariance_weight,
            ssl_structure_weight,
            ssl_fusion_weight,
            ssl_gate_weight,
        )
        if min(loss_weights) < 0:
            raise ValueError("all PC-CMKA loss weights must be non-negative")
        if edge_histogram_bins < 2:
            raise ValueError("edge_histogram_bins must be at least two")
        if top_edges < 1:
            raise ValueError("top_edges must be positive")
        shared_lambda = ReferenceSpectralOperator.theoretical_upper_bound(
            rho_max
        )
        pathways = []
        prior_sources = []
        for input_dim, prior in zip(input_dims, graph_priors):
            edge_index = prior.edge_index if hasattr(prior, "edge_index") else prior[0]
            prior_weights = (
                prior.prior_weights if hasattr(prior, "prior_weights") else prior[1]
            )
            num_nodes = prior.num_nodes if hasattr(prior, "num_nodes") else input_dim
            if int(num_nodes) != int(input_dim):
                raise ValueError("graph prior and pathway input dimensions differ")
            prior_sources.append(getattr(prior, "source", "unspecified"))
            pathways.append(
                PCCMKAPathwayEncoder(
                    input_dim,
                    output_dim,
                    edge_index,
                    prior_weights,
                    shared_lambda=shared_lambda,
                    moment_order=moment_order,
                    probe_epsilon=probe_epsilon,
                    dictionary_rank=dictionary_rank,
                    dictionary_trainable=dictionary_trainable,
                    target_hidden_dim=target_hidden_dim,
                    target_max_offset=target_max_offset,
                    target_min_precision=target_min_precision,
                    solver_iterations=solver_iterations,
                    solver_step_size=solver_step_size,
                    solver_beta=solver_beta,
                    solver_lambda_rho=solver_lambda_rho,
                    solver_gradient_clip=solver_gradient_clip,
                    rho_max=rho_max,
                    calibration_mode=calibration_mode,
                    augmentation_mode=augmentation_mode,
                    hessian_mode=hessian_mode,
                    hessian_low_rank=hessian_low_rank,
                    hessian_damping=hessian_damping,
                    xi_max=xi_max,
                    krylov_eta=krylov_eta,
                    krylov_enabled=krylov_enabled,
                    shared_route=shared_route,
                    random_drop_probability=random_drop_probability,
                    identifiability_mode=identifiability_mode,
                    identifiability_probes=identifiability_probes,
                )
            )
        self.pathways = nn.ModuleList(pathways)
        self.prior_sources = tuple(prior_sources)
        self.shared_lambda = float(shared_lambda)
        self.lambda_moment = float(lambda_moment)
        self.lambda_trust = float(lambda_trust)
        self.lambda_ssl = float(lambda_ssl)
        self.lambda_id = float(lambda_id)
        self.lambda_dict = float(lambda_dict)
        self.lambda_ddkac_consistency = float(lambda_ddkac_consistency)
        self.ssl_variance_weight = float(ssl_variance_weight)
        self.ssl_covariance_weight = float(ssl_covariance_weight)
        self.ssl_structure_weight = float(ssl_structure_weight)
        self.ssl_fusion_weight = float(ssl_fusion_weight)
        self.ssl_gate_weight = float(ssl_gate_weight)
        self.edge_histogram_bins = int(edge_histogram_bins)
        self.rho_max = float(rho_max)
        self.top_edges = int(top_edges)
        self.auxiliary_loss = torch.tensor(0.0)
        self.auxiliary_losses: dict[str, torch.Tensor] = {}
        self.diagnostics: dict[str, torch.Tensor] = {}
        self.last_positive_tokens: torch.Tensor | None = None
        self.last_negative_tokens: torch.Tensor | None = None
        self.last_edge_diagnostics: list[dict[str, torch.Tensor]] = []

    def reset_conservative_initialization(self) -> None:
        """Restore zero-offset priors after a parent model-wide initializer."""

        for pathway in self.pathways:
            nn.init.zeros_(pathway.target_network.offset_head.weight)
            nn.init.zeros_(pathway.target_network.offset_head.bias)
            nn.init.zeros_(pathway.target_network.precision_head.weight)
            nn.init.zeros_(pathway.target_network.precision_head.bias)
            nn.init.zeros_(pathway.direct_edge_gate.network[-1].weight)
            nn.init.zeros_(pathway.direct_edge_gate.network[-1].bias)

    def forward(
        self,
        pathways: list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if len(pathways) != 6:
            raise ValueError("expected six functional-group tensors")
        start = time.perf_counter()
        encoded = []
        positive = []
        negative = []
        calibrations = []
        augmentations = []
        route_results = []
        identifiability_losses = []
        enable_augmentation = self.training and self.lambda_ssl > 0
        enable_identifiability = self.training and self.lambda_id > 0
        for module, inputs in zip(self.pathways, pathways):
            result, calibration, augmentation, identifiability = module(
                inputs,
                enable_augmentation=enable_augmentation,
                enable_identifiability=enable_identifiability,
                generator=generator,
            )
            encoded.append(result.base_token)
            positive.append(result.positive_token)
            negative.append(result.negative_token)
            route_results.append(result)
            calibrations.append(calibration)
            augmentations.append(augmentation)
            identifiability_losses.append(identifiability)

        tokens = torch.stack(encoded, dim=1)
        positive_tokens = torch.stack(positive, dim=1)
        negative_tokens = torch.stack(negative, dim=1)
        positive_structure = torch.stack(
            [item.positive_structure for item in route_results], dim=1
        )
        negative_structure = torch.stack(
            [item.negative_structure for item in route_results], dim=1
        )
        structure_ssl = negative_free_consistency_loss(
            positive_structure,
            negative_structure,
            variance_weight=self.ssl_variance_weight,
            covariance_weight=self.ssl_covariance_weight,
        )
        fusion_ssl = negative_free_consistency_loss(
            positive_tokens, negative_tokens
        )
        # Frequency routes are shared in the full method, while the two graph
        # views retain separate value/structure gates as required by L_gate.
        gate_ssl = torch.stack(
            [
                (item.positive_gate - item.negative_gate).square().mean()
                for item in route_results
            ]
        ).mean()
        ddkac_consistency = torch.stack(
            [
                (
                    1.0
                    - F.cosine_similarity(
                        item.base_token, item.base_structure, dim=1
                    )
                ).mean()
                for item in route_results
            ]
        ).mean()
        moment_loss = torch.stack(
            [item.moment_loss for item in calibrations]
        ).mean()
        trust_loss = torch.stack(
            [item.trust_loss for item in calibrations]
        ).mean()
        dictionary_loss = torch.stack(
            [item.dictionary_loss for item in calibrations]
        ).mean()
        identifiability_loss = torch.stack(identifiability_losses).mean()
        ssl_normalizer = max(
            self.ssl_structure_weight
            + self.ssl_fusion_weight
            + self.ssl_gate_weight,
            1e-12,
        )
        ssl_loss = (
            self.ssl_structure_weight * structure_ssl
            + self.ssl_fusion_weight * fusion_ssl
            + self.ssl_gate_weight * gate_ssl
        ) / ssl_normalizer
        self.auxiliary_losses = {
            "moment": moment_loss,
            "trust": trust_loss,
            "ssl": ssl_loss,
            "ssl_structure": structure_ssl,
            "ssl_fusion": fusion_ssl,
            "ssl_gate": gate_ssl,
            "identifiability": identifiability_loss,
            "dictionary": dictionary_loss,
            "ddkac_consistency": ddkac_consistency,
        }
        self.auxiliary_loss = (
            self.lambda_moment * moment_loss
            + self.lambda_trust * trust_loss
            + self.lambda_ssl * ssl_loss
            + self.lambda_id * identifiability_loss
            + self.lambda_dict * dictionary_loss
            + self.lambda_ddkac_consistency * ddkac_consistency
        )
        self.last_positive_tokens = positive_tokens
        self.last_negative_tokens = negative_tokens

        edge_quantiles = []
        edge_histograms = []
        top_indices = []
        top_values = []
        top_increase_indices = []
        top_increase_values = []
        top_decrease_indices = []
        top_decrease_values = []
        self.last_edge_diagnostics = []
        shared_top_count = min(
            self.top_edges,
            *(item.log_deformation.shape[1] for item in calibrations),
        )
        for calibration in calibrations:
            absolute = calibration.log_deformation.detach().abs()
            quantiles = torch.quantile(
                calibration.log_deformation.detach(),
                calibration.log_deformation.new_tensor(
                    [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                ),
                dim=1,
            ).transpose(0, 1)
            values, indices = torch.topk(
                absolute, shared_top_count, dim=1
            )
            increase_values, increase_indices = torch.topk(
                calibration.log_deformation.detach(),
                shared_top_count,
                dim=1,
            )
            decrease_values, decrease_indices = torch.topk(
                -calibration.log_deformation.detach(),
                shared_top_count,
                dim=1,
            )
            histograms = torch.stack(
                [
                    torch.histc(
                        patient,
                        bins=self.edge_histogram_bins,
                        min=-self.rho_max,
                        max=self.rho_max,
                    )
                    for patient in calibration.log_deformation.detach()
                ]
            )
            edge_quantiles.append(quantiles)
            edge_histograms.append(histograms)
            top_indices.append(indices)
            top_values.append(values)
            top_increase_indices.append(increase_indices)
            top_increase_values.append(increase_values)
            top_decrease_indices.append(decrease_indices)
            top_decrease_values.append(-decrease_values)
            self.last_edge_diagnostics.append(
                {
                    "log_deformation": calibration.log_deformation.detach(),
                    "weights": calibration.weights.detach(),
                    "edge_index": self.pathways[
                        len(self.last_edge_diagnostics)
                    ].moment_response.operator.edge_index.detach().unsqueeze(0),
                }
            )

        self.diagnostics = {
            "rho": torch.stack([item.rho for item in calibrations], dim=1).detach(),
            "target_moments": torch.stack(
                [item.target_moments for item in calibrations], dim=1
            ).detach(),
            "actual_moments": torch.stack(
                [item.actual_moments for item in calibrations], dim=1
            ).detach(),
            "moment_residual": torch.stack(
                [item.residual for item in calibrations], dim=1
            ).detach(),
            "target_offset": torch.stack(
                [
                    item.target_moments - item.prior_moments
                    for item in calibrations
                ],
                dim=1,
            ).detach(),
            "precision": torch.stack(
                [item.precision for item in calibrations], dim=1
            ).detach(),
            "coefficient_norm": torch.stack(
                [item.coefficient_norm for item in calibrations], dim=1
            ).detach(),
            "solver_objective": torch.stack(
                [item.objective_history for item in calibrations], dim=2
            ).detach(),
            "hessian_diagonal": torch.stack(
                [item.hessian_diagonal for item in augmentations], dim=1
            ).detach(),
            "hessian_eigenvalues": torch.stack(
                [item.hessian_eigenvalues for item in augmentations], dim=1
            ).detach(),
            "infinity_scale": torch.stack(
                [item.infinity_scale for item in augmentations], dim=1
            ).detach(),
            "krylov_scale": torch.stack(
                [item.krylov_scale for item in augmentations], dim=1
            ).detach(),
            "krylov_error": torch.stack(
                [item.krylov_error for item in augmentations], dim=1
            ).detach(),
            "routes": torch.stack(
                [item.route for item in route_results], dim=1
            ).detach(),
            "positive_routes": torch.stack(
                [item.positive_route for item in route_results], dim=1
            ).detach(),
            "negative_routes": torch.stack(
                [item.negative_route for item in route_results], dim=1
            ).detach(),
            "gates": torch.cat(
                [item.gate for item in route_results], dim=1
            ).detach(),
            "tangent_correlation": torch.stack(
                [item.detach() for item in identifiability_losses]
            ),
            "edge_quantiles": torch.stack(edge_quantiles, dim=1),
            "top_edge_indices": torch.stack(top_indices, dim=1),
            "top_edge_abs_log_change": torch.stack(top_values, dim=1),
            "edge_histogram_counts": torch.stack(
                edge_histograms, dim=1
            ),
            "edge_histogram_boundaries": torch.linspace(
                -self.rho_max,
                self.rho_max,
                self.edge_histogram_bins + 1,
                device=tokens.device,
                dtype=tokens.dtype,
            ),
            "top_increase_indices": torch.stack(
                top_increase_indices, dim=1
            ),
            "top_increase_log_change": torch.stack(
                top_increase_values, dim=1
            ),
            "top_decrease_indices": torch.stack(
                top_decrease_indices, dim=1
            ),
            "top_decrease_log_change": torch.stack(
                top_decrease_values, dim=1
            ),
            "perturbation_max_after": torch.stack(
                [
                    item.relative_perturbation.abs().amax(dim=1)
                    for item in augmentations
                ],
                dim=1,
            ).detach(),
            "perturbation_max_before_krylov": torch.stack(
                [
                    item.relative_perturbation.abs().amax(dim=1)
                    / item.krylov_scale.clamp_min(1e-12)
                    for item in augmentations
                ],
                dim=1,
            ).detach(),
            "structure_view_distance": torch.stack(
                [
                    (item.positive_structure - item.negative_structure)
                    .norm(dim=1)
                    for item in route_results
                ],
                dim=1,
            ).detach(),
            "calibration_finite": torch.stack(
                [item.finite for item in calibrations]
            ).detach(),
            "augmentation_finite": torch.stack(
                [item.finite for item in augmentations]
            ).detach(),
            "forward_seconds": tokens.new_tensor(time.perf_counter() - start),
            "shared_lambda": tokens.new_tensor(self.shared_lambda),
        }
        return tokens
