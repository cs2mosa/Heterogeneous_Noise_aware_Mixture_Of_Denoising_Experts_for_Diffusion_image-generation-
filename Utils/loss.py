import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Utils.cdf_utils import branch_cdf_bounds, mixture_cdf
except ImportError:
    from cdf_utils import branch_cdf_bounds, mixture_cdf


# 1. Helper Function
def laplacian_decomposition(image: torch.Tensor, kernel_size: int = 9, sigma: float = 1.5):
    """
    Splits image into orthogonal Low-Freq and High-Freq components via unitary 2D rFFT
    so <low_freq, high_freq> == 0 and ||low_freq||^2 + ||high_freq||^2 == ||image||^2.
    """
    del kernel_size
    h, w = image.shape[-2:]
    cutoff_sq = math.log(2.0) / (4.0 * (math.pi ** 2) * max(float(sigma), 1e-6) ** 2)

    x_fp32 = image.float().contiguous()
    fy = torch.fft.fftfreq(h, device=image.device, dtype=torch.float32).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=image.device, dtype=torch.float32).view(1, -1)
    low_mask = ((fy * fy + fx * fx) <= cutoff_sq).to(torch.float32)

    spectrum = torch.fft.rfft2(x_fp32, norm="ortho")
    low_freq = torch.fft.irfft2(spectrum * low_mask, s=(h, w), norm="ortho").to(image.dtype)
    high_freq = image - low_freq
    return low_freq, high_freq


class PathPriorLoss(nn.Module):
    def __init__(self,
                 expert_centers: torch.Tensor,
                 p_mean: float = -0.4,  # IMPORTANT: Must match sampling p_mean
                 p_std: float = 1.0,  # IMPORTANT: Must match sampling p_std
                 sharpness: float = 25.0,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 alpha_ext: float = 0.2,
                 noise_range: Optional[tuple[float, float]] = None):
        super().__init__()
        self.register_buffer('expert_centers', expert_centers)
        self.p_mean = p_mean
        self.p_std = p_std
        self.sharpness = sharpness
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.alpha_ext = alpha_ext
        if noise_range is not None:
            low_cdf, high_cdf = branch_cdf_bounds(
                noise_range[0], noise_range[1],
                p_mean, p_std, sigma_min, sigma_max, alpha_ext,
            )
            span = max(float(high_cdf - low_cdf), 1e-4)
        else:
            span = 1.0
        self.register_buffer('branch_span', torch.tensor(span, dtype=torch.float32))
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, gate_probs: torch.Tensor, sigma: torch.Tensor):
        device = sigma.device
        log_sigma = torch.log(sigma.flatten().clamp(min=1e-8))

        # Match the mixture CDF and branch-local span normalization used by the routers
        sigma_percentile = mixture_cdf(
            log_sigma, self.p_mean, self.p_std,
            math.log(self.sigma_min), math.log(self.sigma_max), self.alpha_ext,
        ).view(-1, 1)

        dist_sq = ((sigma_percentile - self.expert_centers.to(device).view(1, -1)) / self.branch_span.to(device)) ** 2
        target_logits = -dist_sq * self.sharpness
        target_dist = F.softmax(target_logits, dim=-1)

        log_probs = torch.log(gate_probs + 1e-8)
        return self.kl_div(log_probs, target_dist)

class EDM_LOSS_NOD(nn.Module):
    def __init__(self,
                 expert_centers_vit,
                 expert_centers_Unet,
                 num_experts: int,
                 sigma_data: float = 0.5,
                 Unet_bal: float = 0.0005,
                 vit_bal: float = 0.0005,
                 z_bal: float = 0.0001,
                 prior_bal: float = 0.001,
                 p_mean: float = 0.4,
                 p_std: float = 1.6,
                 sharpness: float = 2.0,
                 transition_sigma: float = 3.0,
                 overlap_width: float = 1.5,
                 alpha_blend: float = 0.2,
                 alpha_unet: float = 0.4,
                 alpha_vit: float = 0.4,
                 lap_kernel: int = 9,
                 lap_sigma: float = 1.5,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 alpha_ext: float = 0.2,
                 vit_noise_range: Optional[tuple[float, float]] = None,
                 unet_noise_range: Optional[tuple[float, float]] = None,
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.sigma_data = sigma_data
        self.Unet_lambda = Unet_bal
        self.vit_lambda = vit_bal
        self.z_bal = z_bal
        self.prior_bal = prior_bal

        # Trajectory decomposition params
        self.transition_sigma = transition_sigma
        self.overlap_width = overlap_width
        self.alpha_blend = alpha_blend
        self.alpha_unet = alpha_unet
        self.alpha_vit = alpha_vit

        # Fix 2: Frequency decomposition parameters for orthogonal branch losses
        self.lap_kernel = lap_kernel
        self.lap_sigma = lap_sigma

        vit_range = vit_noise_range if vit_noise_range is not None else (
            (sigma_min, transition_sigma) if transition_sigma > sigma_min else None
        )
        unet_range = unet_noise_range if unet_noise_range is not None else (
            (transition_sigma, sigma_max) if sigma_max > transition_sigma else None
        )

        self.prior_path_loss_vit = PathPriorLoss(
            p_mean=p_mean, p_std=p_std,
            expert_centers=expert_centers_vit,
            sharpness=sharpness,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            alpha_ext=alpha_ext,
            noise_range=vit_range,
        )
        self.prior_path_loss_Unet = PathPriorLoss(
            p_mean=p_mean, p_std=p_std,
            expert_centers=expert_centers_Unet,
            sharpness=sharpness,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            alpha_ext=alpha_ext,
            noise_range=unet_range,
        )

    def __call__(self,
                 x: torch.Tensor,
                 noisy_x: torch.Tensor,
                 sigma: torch.Tensor,
                 out_model: dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:

        # 1. EDM Weighting (Corrected Clamping)
        sigma = sigma.reshape(-1, 1, 1, 1)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # This ensures Loss exactly matches Model behavior
        wx_vit = out_model['w_vit']  # (B, 1, 1, 1)
        wx_unet = out_model['w_unet']  # (B, 1, 1, 1)

        # Keep Gaussian NLL gradient d/d(log_var) exclusively on the blended prediction
        if out_model["log_var"] is not None:
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            uncertainty_scale = 1.0 / log_var.exp()
            branch_uncertainty_scale = uncertainty_scale.detach()
        else:
            log_var = None
            uncertainty_scale = 1.0
            branch_uncertainty_scale = 1.0

        # 3. Global Blend Loss (The "Truth")
        diff_blend = out_model["denoised"] - x
        pure_loss = torch.mean(weight * uncertainty_scale * (diff_blend ** 2))
        if log_var is not None:
            pure_loss = pure_loss + torch.mean(log_var)

        # 4. UNet Branch Loss — LOW-FREQUENCY projection only (Fix 2)
        unet_residual = out_model['unet_denoised'] - x
        unet_low, _ = laplacian_decomposition(unet_residual, self.lap_kernel, self.lap_sigma)
        unet_loss = torch.mean(weight * branch_uncertainty_scale * wx_unet.detach() * (unet_low ** 2))

        # 5. ViT Branch Loss — HIGH-FREQUENCY projection only (Fix 2)
        vit_residual = out_model['vit_denoised'] - x
        _, vit_high = laplacian_decomposition(vit_residual, self.lap_kernel, self.lap_sigma)
        vit_loss = torch.mean(weight * branch_uncertainty_scale * wx_vit.detach() * (vit_high ** 2))

        # 6. Total Loss Aggregation
        trajectory_loss = (
                self.alpha_blend * pure_loss +
                self.alpha_unet * unet_loss +
                self.alpha_vit * vit_loss
        )

        # Aux losses
        denoising_loss = torch.mean(diff_blend ** 2)
        router_loss = (
                self.Unet_lambda * self.load_balance(out_model["Unet_router_loss"], self.num_experts) +
                self.vit_lambda * self.load_balance(out_model["vit_router_loss"], self.num_experts)
        ).clamp(max=50)

        # Path prior (scaling)
        scaling_loss = (
                self.prior_bal * (
                self.prior_path_loss_vit(out_model["vit_router_loss"], sigma) +
                self.prior_path_loss_Unet(out_model["Unet_router_loss"], sigma)
        )
        ).clamp(max=50)

        z_loss = (
                self.z_bal * self.z_loss(out_model["Unet_raw"]) +
                self.z_bal * self.z_loss(out_model["vit_raw"])
        ).clamp(max=50)

        total_loss = (
                trajectory_loss +
                router_loss +
                scaling_loss +
                z_loss
        )

        return {
            "loss": total_loss,
            "loss_texture": vit_loss,       # ViT is now doing Low-Noise Textures
            "loss_structure": unet_loss,
            "denoising": denoising_loss,
            "balance": router_loss,
            "z_loss": z_loss,
            "entropy": scaling_loss,
            "pure_loss": pure_loss,
        }
    @staticmethod
    def load_balance(gate_probs, num_experts):
        P = gate_probs.mean(dim=0)
        return num_experts * torch.sum(P ** 2)

    @staticmethod
    def z_loss(logits):
        # Penalize large logits to keep router stable
        logits = logits.clamp(min=-50, max=50)
        z = torch.logsumexp(logits, dim=-1) ** 2
        return torch.mean(z)


