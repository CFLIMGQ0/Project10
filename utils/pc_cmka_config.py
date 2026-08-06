"""Load and resolve PC-CMKA-DDKAC default/ablation configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pc_cmka_ddkac.json"


def load_pc_cmka_config(
    path: str | Path | None = None,
    ablation: str = "full_pc_cmka_ddkac",
) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = copy.deepcopy(raw)
    ablations = base.pop("ablations")
    selected = next(
        (item for item in ablations if item["name"] == ablation), None
    )
    if selected is None:
        names = ", ".join(item["name"] for item in ablations)
        raise ValueError(f"unknown PC-CMKA ablation {ablation!r}; choose from {names}")
    if selected.get("encoder") == "dd_kac":
        raise ValueError(
            "original_ddkac_fixed_graph uses --mrepath_genomic_encoder dd_kac, "
            "not the PC-CMKA encoder"
        )

    solver = base["solver"]
    augmentation = base["augmentation"]
    ssl = base["ssl"]
    identifiability = base["identifiability"]
    loss = base["loss"]
    if "calibration" in selected:
        solver["mode"] = selected["calibration"]
        if selected["calibration"] in {"fixed_graph", "direct_edge_gate"}:
            loss["lambda_moment"] = 0.0
        if selected["calibration"] == "direct_edge_gate":
            loss["lambda_dict"] = 0.0
    if "augmentation" in selected:
        augmentation["mode"] = selected["augmentation"]
    if "krylov_enabled" in selected:
        augmentation["krylov_enabled"] = bool(selected["krylov_enabled"])
    if "shared_route" in selected:
        augmentation["shared_route"] = bool(selected["shared_route"])
    if selected.get("ssl") is False:
        loss["lambda_ssl"] = 0.0
    elif selected.get("ssl") == "structure":
        ssl["structure_weight"] = 1.0
        ssl["fusion_weight"] = 0.0
        ssl["gate_weight"] = 0.0
    if selected.get("identifiability") == "off":
        identifiability["mode"] = "off"
        loss["lambda_id"] = 0.0
    elif "identifiability" in selected:
        identifiability["mode"] = selected["identifiability"]
    base["selected_ablation"] = selected["name"]
    base["source_label"] = selected["source_label"]
    base["config_path"] = str(config_path.resolve())
    return base


def encoder_kwargs(config: dict) -> dict:
    spectral = config["spectral"]
    dictionary = config["dictionary"]
    target = config["target"]
    solver = config["solver"]
    augmentation = config["augmentation"]
    ssl = config["ssl"]
    identifiability = config["identifiability"]
    loss = config["loss"]
    diagnostics = config["diagnostics"]
    return {
        "moment_order": spectral["moment_order"],
        "probe_epsilon": spectral["probe_epsilon"],
        "dictionary_rank": dictionary["rank"],
        "dictionary_trainable": dictionary["trainable"],
        "target_hidden_dim": target["hidden_dim"],
        "target_max_offset": target["max_offset"],
        "target_min_precision": target["min_precision"],
        "solver_iterations": solver["iterations"],
        "solver_step_size": solver["step_size"],
        "solver_beta": solver["beta"],
        "solver_lambda_rho": solver["lambda_rho"],
        "solver_gradient_clip": solver["gradient_clip"],
        "rho_max": spectral["rho_max"],
        "calibration_mode": solver["mode"],
        "augmentation_mode": augmentation["mode"],
        "hessian_mode": augmentation["hessian_mode"],
        "hessian_low_rank": augmentation["hessian_low_rank"],
        "hessian_damping": augmentation["damping"],
        "xi_max": augmentation["xi_max"],
        "krylov_eta": augmentation["krylov_eta"],
        "krylov_enabled": augmentation["krylov_enabled"],
        "shared_route": augmentation["shared_route"],
        "random_drop_probability": augmentation.get("drop_probability", 0.1),
        "identifiability_mode": identifiability["mode"],
        "identifiability_probes": identifiability["random_probes"],
        "lambda_moment": loss["lambda_moment"],
        "lambda_trust": loss["lambda_trust"],
        "lambda_ssl": loss["lambda_ssl"],
        "lambda_id": loss["lambda_id"],
        "lambda_dict": loss["lambda_dict"],
        "lambda_ddkac_consistency": loss["lambda_ddkac_consistency"],
        "ssl_variance_weight": ssl["variance_weight"],
        "ssl_covariance_weight": ssl["covariance_weight"],
        "ssl_structure_weight": ssl["structure_weight"],
        "ssl_fusion_weight": ssl["fusion_weight"],
        "ssl_gate_weight": ssl["gate_weight"],
        "edge_histogram_bins": diagnostics["edge_histogram_bins"],
        "top_edges": diagnostics["top_edges"],
    }
