"""
Mixture-CDF utilities for the hybrid sampling distribution.

Shared by ParametricTrajectoryRouter, MaskGenerator, and PathPriorLoss.

The training noise schedule samples ln(σ) from a mixture:
    p_train(ln σ) = (1-α)·N(ln σ | p_mean, p_std²) + α·U(ln σ | ln σ_min, ln σ_max)

with α = extreme_prob (default 0.2). All CDF computations must use this
mixture CDF — NOT the pure Gaussian CDF — to avoid exponential compression
in the Gaussian tail (σ > 10) that makes the router blind to high-noise
distinctions.

Fix 4 from the Information Geometry Analysis.
"""

import math
import torch


def mixture_cdf(
    log_sigma: torch.Tensor,
    p_mean: float,
    p_std: float,
    log_sigma_min: float,
    log_sigma_max: float,
    alpha_ext: float = 0.2,
) -> torch.Tensor:
    """
    CDF of the hybrid sampling distribution used by sample_sigma_hybrid.

    F_mix(σ) = (1 - α)·Φ((ln σ - p_mean) / p_std) + α·(ln σ - ln σ_min) / (ln σ_max - ln σ_min)

    The derivative f_mix(ln σ) ≥ α / (ln σ_max - ln σ_min) > 0 everywhere,
    so the CDF never saturates in the tails.

    Args:
        log_sigma:     ln(σ) values, any shape.
        p_mean:        Mean of the log-normal component.
        p_std:         Std of the log-normal component.
        log_sigma_min: ln(σ_min), lower bound of the uniform component.
        log_sigma_max: ln(σ_max), upper bound of the uniform component.
        alpha_ext:     Weight of the uniform (extreme) component. Default 0.2.

    Returns:
        Mixture CDF values in [0.001, 0.999], same shape as log_sigma.
    """
    gaussian_cdf = 0.5 * (1.0 + torch.erf(
        (log_sigma - p_mean) / (p_std * math.sqrt(2))
    ))
    log_range = log_sigma_max - log_sigma_min
    if log_range < 1e-8:
        uniform_cdf = torch.zeros_like(log_sigma)
    else:
        uniform_cdf = (log_sigma - log_sigma_min) / log_range

    return ((1.0 - alpha_ext) * gaussian_cdf + alpha_ext * uniform_cdf).clamp(0.001, 0.999)


def branch_local_quantile_centers(
    num_experts: int,
    sigma_low: float,
    sigma_high: float,
    p_mean: float,
    p_std: float,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    alpha_ext: float = 0.2,
) -> torch.Tensor:
    """
    Place E expert centers equidistantly in branch-local [0, 1] quantile space,
    then map back to global mixture-CDF coordinates.

    Each expert spans an identical width 1/E in local quantile space, ensuring
    equal training sample count per expert regardless of the nonlinear CDF mapping.

    Args:
        num_experts: Number of experts in this branch (E).
        sigma_low:   Lower bound of this branch's noise range.
        sigma_high:  Upper bound of this branch's noise range.
        p_mean:      Mean of the log-normal sampling component.
        p_std:       Std of the log-normal sampling component.
        sigma_min:   Global minimum sigma.
        sigma_max:   Global maximum sigma.
        alpha_ext:   Uniform component weight.

    Returns:
        Tensor of shape (E,) with expert center positions in global CDF space.
    """
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)

    cdf_low = mixture_cdf(
        torch.tensor(math.log(sigma_low)), p_mean, p_std, log_min, log_max, alpha_ext
    ).item()
    cdf_high = mixture_cdf(
        torch.tensor(math.log(sigma_high)), p_mean, p_std, log_min, log_max, alpha_ext
    ).item()

    # Equidistant in local [0, 1] quantile space: c_i = (i + 0.5) / E
    local = torch.linspace(0.5 / num_experts, 1.0 - 0.5 / num_experts, num_experts)

    # Affine map from local [0, 1] to global [cdf_low, cdf_high]
    return (cdf_low + local * (cdf_high - cdf_low)).clamp(0.001, 0.999)

