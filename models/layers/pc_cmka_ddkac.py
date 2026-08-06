"""Backward-compatible public API for the modular PC-CMKA-DDKAC package."""

from models.layers.pc_cmka_spectral import (
    ChebyshevMomentResponse,
    ReferenceSpectralOperator,
    chebyshev_recurrence,
    normalized_patient_probe,
)
from models.layers.pc_cmka_calibration import (
    CalibrationResult,
    DifferentiableMomentSolver,
    DirectPatientEdgeGate,
    EdgeDeformationDictionary,
    MomentTargetNetwork,
)
from models.layers.pc_cmka_augmentation import (
    AntitheticAugmentationResult,
    CalibrationUncertaintyAugmentor,
    ControlGraphAugmentor,
    build_krylov_basis,
    krylov_operator_error,
)
from models.layers.pc_cmka_ddkac_core import (
    IdentifiabilityTangentRegularizer,
    SharedRouteDDKACPathway,
    SharedRoutePathwayResult,
    negative_free_consistency_loss,
)
from models.layers.pc_cmka_encoder import (
    PCCMKADDKACEncoder,
    PCCMKAPathwayEncoder,
)

__all__ = [
    "AntitheticAugmentationResult",
    "CalibrationResult",
    "CalibrationUncertaintyAugmentor",
    "ChebyshevMomentResponse",
    "ControlGraphAugmentor",
    "DifferentiableMomentSolver",
    "DirectPatientEdgeGate",
    "EdgeDeformationDictionary",
    "IdentifiabilityTangentRegularizer",
    "MomentTargetNetwork",
    "PCCMKADDKACEncoder",
    "PCCMKAPathwayEncoder",
    "ReferenceSpectralOperator",
    "SharedRouteDDKACPathway",
    "SharedRoutePathwayResult",
    "build_krylov_basis",
    "chebyshev_recurrence",
    "krylov_operator_error",
    "negative_free_consistency_loss",
    "normalized_patient_probe",
]
