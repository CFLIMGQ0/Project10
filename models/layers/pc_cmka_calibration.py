"""Low-rank dictionaries and differentiable moment calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.pc_cmka_spectral import ChebyshevMomentResponse

class EdgeDeformationDictionary(nn.Module):
    """Learnable low-rank edge dictionary with weighted-orthogonal init."""

    def __init__(
        self,
        prior_weights: torch.Tensor,
        rank: int,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        prior_weights = torch.as_tensor(prior_weights).float().flatten()
        if rank < 1 or rank > prior_weights.numel():
            raise ValueError("dictionary rank must be in [1, num_edges]")
        if bool((prior_weights <= 0).any()) or not bool(
            torch.isfinite(prior_weights).all()
        ):
            raise ValueError("prior_weights must be finite and positive")
        weighted_random = torch.randn(prior_weights.numel(), rank)
        orthonormal, _ = torch.linalg.qr(weighted_random, mode="reduced")
        initial = orthonormal / prior_weights.sqrt().unsqueeze(1)
        self.matrix = nn.Parameter(initial, requires_grad=trainable)
        self.register_buffer("prior_weights", prior_weights)

    @property
    def rank(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def num_edges(self) -> int:
        return int(self.matrix.shape[0])

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.rank:
            raise ValueError("coefficients have the wrong dictionary rank")
        return coefficients @ self.matrix.transpose(0, 1)

    def weighted_gram(self) -> torch.Tensor:
        return self.matrix.transpose(0, 1) @ (
            self.prior_weights[:, None] * self.matrix
        )
    def orthogonality_loss(self) -> torch.Tensor:
        identity = torch.eye(
            self.rank, device=self.matrix.device, dtype=self.matrix.dtype
        )
        return (self.weighted_gram() - identity).square().mean()


class MomentTargetNetwork(nn.Module):
    """Predict target response offsets and positive precisions, never edges."""

    def __init__(
        self,
        input_dim: int,
        moment_order: int,
        hidden_dim: int = 64,
        *,
        max_offset: float = 0.5,
        min_precision: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(input_dim, moment_order, hidden_dim) < 1:
            raise ValueError("network dimensions must be positive")
        if max_offset <= 0 or min_precision <= 0:
            raise ValueError("offset and precision bounds must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.offset_head = nn.Linear(hidden_dim, moment_order)
        self.precision_head = nn.Linear(hidden_dim, moment_order)
        self.max_offset = float(max_offset)
        self.min_precision = float(min_precision)

        # The conservative initial state asks the inverse problem to recover
        # the prior graph.  This makes Delta=0 an explicit tested fallback.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.precision_head.weight)
        nn.init.zeros_(self.precision_head.bias)

    def forward(self, expression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)
        hidden = self.encoder(expression)
        offset = self.max_offset * torch.tanh(self.offset_head(hidden))
        precision = F.softplus(self.precision_head(hidden)) + self.min_precision
        return offset, precision


class DirectPatientEdgeGate(nn.Module):
    """Direct full-edge prediction used only by the required A2 control."""

    def __init__(
        self,
        input_dim: int,
        num_edges: int,
        *,
        hidden_dim: int = 64,
        rho_max: float = 0.5,
    ) -> None:
        super().__init__()
        if min(input_dim, num_edges, hidden_dim) < 1 or rho_max <= 0:
            raise ValueError("invalid direct edge gate settings")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_edges),
        )
        self.rho_max = float(rho_max)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)
        return self.rho_max * torch.tanh(self.network(expression))


@dataclass
class CalibrationResult:
    coefficients: torch.Tensor
    log_deformation: torch.Tensor
    weights: torch.Tensor
    prior_moments: torch.Tensor
    target_moments: torch.Tensor
    actual_moments: torch.Tensor
    precision: torch.Tensor
    residual: torch.Tensor
    rho: torch.Tensor
    coefficient_norm: torch.Tensor
    objective_history: torch.Tensor
    moment_loss: torch.Tensor
    trust_loss: torch.Tensor
    dictionary_loss: torch.Tensor
    finite: torch.Tensor


class DifferentiableMomentSolver(nn.Module):
    """Fixed-iteration proximal-gradient inverse moment calibration.

    Supported modes:

    - ``fixed_graph``: use the prior graph and skip inverse updates;
    - ``detach_solver``: solve, then detach coefficients from the solver graph;
    - ``target_only``: same detached solve; the caller freezes all parameters
      except the target network and optimizes the explicit moment loss;
    - ``joint``: retain the complete unrolled optimization graph.

    With a positive linear penalty on ``rho`` and no other dependence on rho,
    its minimum for fixed ``a`` is ``rho=||Pa||_inf``.  We therefore compute
    this minimal evidence radius directly and project it to ``rho_max`` rather
    than predicting rho with another network.
    """

    VALID_MODES = {"fixed_graph", "detach_solver", "target_only", "joint"}

    def __init__(
        self,
        moment_response: ChebyshevMomentResponse,
        dictionary: EdgeDeformationDictionary,
        *,
        iterations: int = 5,
        step_size: float = 0.1,
        beta: float = 1e-2,
        lambda_rho: float = 1e-3,
        rho_max: float = 0.5,
        gradient_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if step_size <= 0 or beta < 0 or lambda_rho < 0 or rho_max <= 0:
            raise ValueError("invalid solver hyperparameters")
        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if dictionary.num_edges != moment_response.operator.num_edges:
            raise ValueError("dictionary and graph edge dimensions differ")
        if rho_max > moment_response.operator.max_log_deformation + 1e-12:
            raise ValueError(
                "rho_max cannot exceed the operator's max_log_deformation"
            )
        # These are references to modules owned by the enclosing pathway
        # encoder.  Avoid registering aliases here, otherwise checkpoints would
        # serialize the same operator/dictionary several times.
        object.__setattr__(self, "moment_response", moment_response)
        object.__setattr__(self, "dictionary", dictionary)
        self.iterations = int(iterations)
        self.step_size = float(step_size)
        self.beta = float(beta)
        self.lambda_rho = float(lambda_rho)
        self.rho_max = float(rho_max)
        self.gradient_clip = float(gradient_clip)

    @staticmethod
    def _as_batch(expression: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if expression.ndim == 1:
            return expression.unsqueeze(0), True
        if expression.ndim == 2:
            return expression, False
        raise ValueError("expression must be [nodes] or [batch, nodes]")

    def _project_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        log_deformation = self.dictionary(coefficients)
        radius = log_deformation.abs().amax(dim=1, keepdim=True)
        scale = torch.clamp(
            self.rho_max / radius.clamp_min(1e-12), max=1.0
        )
        return coefficients * scale

    def _per_patient_objective(
        self,
        expression: torch.Tensor,
        coefficients: torch.Tensor,
        prior_moments: torch.Tensor,
        target_moments: torch.Tensor,
        precision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_deformation = self.dictionary(coefficients)
        weights = self.moment_response.operator.patient_weights(log_deformation)
        actual = self.moment_response(expression, weights)
        residual = actual - target_moments
        rho = log_deformation.abs().amax(dim=1)
        objective = (
            (precision * residual.square()).sum(dim=1)
            + self.beta * coefficients.square().sum(dim=1)
            + self.lambda_rho * rho
        )
        return objective, actual, residual, rho

    def forward(
        self,
        expression: torch.Tensor,
        target_offset: torch.Tensor,
        precision: torch.Tensor,
        *,
        mode: str = "joint",
    ) -> CalibrationResult:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown calibration mode: {mode}")
        expression, _ = self._as_batch(expression)
        if target_offset.ndim == 1:
            target_offset = target_offset.unsqueeze(0)
        if precision.ndim == 1:
            precision = precision.unsqueeze(0)
        expected = (expression.shape[0], self.moment_response.order)
        if target_offset.shape != expected or precision.shape != expected:
            raise ValueError("target_offset and precision must be [batch, order]")
        if bool((precision <= 0).any()) or not bool(torch.isfinite(precision).all()):
            raise ValueError("precision must be finite and positive")

        prior_moments = self.moment_response(expression)
        target_moments = prior_moments + target_offset
        coefficients = expression.new_zeros(
            expression.shape[0], self.dictionary.rank, requires_grad=True
        )
        history: list[torch.Tensor] = []
        caller_grad_enabled = torch.is_grad_enabled()
        fully_differentiable = mode == "joint" and caller_grad_enabled

        if mode != "fixed_graph":
            # Non-joint modes intentionally solve against detached targets.
            solve_expression = expression if fully_differentiable else expression.detach()
            solve_prior = prior_moments if fully_differentiable else prior_moments.detach()
            solve_target = target_moments if fully_differentiable else target_moments.detach()
            solve_precision = precision if fully_differentiable else precision.detach()
            # Validation runs under torch.no_grad(), but solving for a still
            # needs first-order local derivatives.  Re-enable them only inside
            # the fixed iteration block and detach the result afterwards.
            with torch.enable_grad():
                for _ in range(self.iterations):
                    objective, _, _, _ = self._per_patient_objective(
                        solve_expression,
                        coefficients,
                        solve_prior,
                        solve_target,
                        solve_precision,
                    )
                    history.append(objective.detach())
                    gradient = torch.autograd.grad(
                        objective.sum(),
                        coefficients,
                        create_graph=fully_differentiable,
                        retain_graph=fully_differentiable,
                    )[0]
                    gradient_norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    gradient = gradient * torch.clamp(
                        self.gradient_clip / gradient_norm, max=1.0
                    )
                    coefficients = self._project_coefficients(
                        coefficients - self.step_size * gradient
                    )
            if not fully_differentiable:
                coefficients = coefficients.detach()
        else:
            coefficients = coefficients.detach()

        coefficients = self._project_coefficients(coefficients)
        objective, actual_moments, residual, rho = self._per_patient_objective(
            expression,
            coefficients,
            prior_moments,
            target_moments,
            precision,
        )
        history.append(objective.detach())
        log_deformation = self.dictionary(coefficients)
        weights = self.moment_response.operator.patient_weights(log_deformation)
        moment_loss = (precision * residual.square()).mean()
        trust_loss = rho.mean()
        dictionary_loss = self.dictionary.orthogonality_loss()
        finite = torch.stack(
            [
                torch.isfinite(coefficients).all(),
                torch.isfinite(weights).all(),
                torch.isfinite(actual_moments).all(),
                torch.isfinite(objective).all(),
            ]
        ).all()
        return CalibrationResult(
            coefficients=coefficients,
            log_deformation=log_deformation,
            weights=weights,
            prior_moments=prior_moments,
            target_moments=target_moments,
            actual_moments=actual_moments,
            precision=precision,
            residual=residual,
            rho=rho,
            coefficient_norm=coefficients.norm(dim=1),
            objective_history=torch.stack(history, dim=0),
            moment_loss=moment_loss,
            trust_loss=trust_loss,
            dictionary_loss=dictionary_loss,
            finite=finite,
        )
