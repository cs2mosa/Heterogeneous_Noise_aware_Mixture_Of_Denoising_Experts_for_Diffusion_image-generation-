import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
import math

# 1. Helper Function
def laplacian_decomposition(image, kernel_size=9, sigma=1.5):
    """
    Splits image into Low-Freq (Gaussian) and High-Freq (Laplacian).
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0: kernel_size += 1

    # Gaussian Blur (Low Freq)
    # We detach because we don't want to backprop through the target generation
    low_freq = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    # Laplacian (High Freq / Residual)
    high_freq = image - low_freq

    return low_freq, high_freq


class PathPriorLoss(nn.Module):
    def __init__(self,
                 expert_centers: torch.Tensor,
                 p_mean: float = -0.4,  # IMPORTANT: Must match sampling p_mean
                 p_std: float = 1.0,    # IMPORTANT: Must match sampling p_std
                 sharpness: float = 25.0):
        super().__init__()
        self.register_buffer('expert_centers', expert_centers)
        self.p_mean = p_mean
        self.p_std = p_std
        self.sharpness = sharpness
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, gate_probs: torch.Tensor, sigma: torch.Tensor):
        device = sigma.device
        log_sigma = torch.log(sigma.flatten().clamp(min=1e-8))
        
        # THE FIX: Mathematically identical to MaskGenerator's CDF projection
        sigma_percentile = 0.5 * (1.0 + torch.erf((log_sigma - self.p_mean) / (self.p_std * math.sqrt(2))))
        sigma_percentile = sigma_percentile.clamp(0.001, 0.999).view(-1, 1)

        dist_sq = (sigma_percentile - self.expert_centers.to(device).view(1, -1)) ** 2
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

        self.prior_path_loss_vit = PathPriorLoss(
            p_mean=p_mean, p_std=p_std,
            expert_centers=expert_centers_vit,
            sharpness=sharpness
        )
        self.prior_path_loss_Unet = PathPriorLoss(
            p_mean=p_mean, p_std=p_std,
            expert_centers=expert_centers_Unet,
            sharpness=sharpness
        )

    def _compute_trajectory_masks(self, sigma: torch.Tensor):
        # U-Net targets High Noise (> transition)
        # ViT targets Low Noise (< transition)
        sigma_flat = sigma.flatten()
        
        # Active when sigma is ABOVE the transition boundary (minus overlap)
        unet_mask = sigma_flat > (self.transition_sigma - self.overlap_width)
        
        # Active when sigma is BELOW the transition boundary (plus overlap)
        vit_mask = sigma_flat < (self.transition_sigma + self.overlap_width)
        
        return unet_mask, vit_mask

    def __call__(self,
                 x: torch.Tensor,
                 noisy_x: torch.Tensor,
                 sigma: torch.Tensor,
                 out_model: dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:

        # 1. EDM Weighting (Corrected Clamping)
        sigma = sigma.reshape(-1, 1, 1, 1)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # 2. Trajectory Decomposition Setup
        unet_mask, vit_mask = self._compute_trajectory_masks(sigma)

        # This ensures Loss exactly matches Model behavior
        wx_vit = out_model['w_vit']  # (B, 1, 1, 1)
        wx_unet = out_model['w_unet']  # (B, 1, 1, 1)

        # 3. Global Blend Loss (The "Truth")
        diff_blend = out_model["denoised"] - x
        if out_model["log_var"] is None:
            pure_loss = torch.mean(weight * (diff_blend ** 2))
        else:
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            pure_loss = torch.mean((weight * (diff_blend ** 2)) / log_var.exp() + log_var)

        # 4. UNet Branch Loss (Masked Reduction)
        # We weight the error by 'wx_unet'. This means as the model naturally transitions
        # away from UNet, we reduce supervision, preventing "Teacher Forcing" on noise.
        m_unet = unet_mask.view(-1, 1, 1, 1).to(dtype=out_model['unet_denoised'].dtype)
        unet_sq_error = (out_model['unet_denoised'] - x) ** 2

        # Weight = EDM_Weight * Blend_Weight * Mask
        weighted_err_unet = weight * wx_unet * unet_sq_error * m_unet

        num_active_elements_u = m_unet.sum() * x.shape[1] * x.shape[2] * x.shape[3]
        unet_loss = weighted_err_unet.sum() / (num_active_elements_u + 1e-6)

        # 5. ViT Branch Loss (Masked Reduction)
        m_vit = vit_mask.view(-1, 1, 1, 1).to(dtype=out_model['vit_denoised'].dtype)
        vit_sq_error = (out_model['vit_denoised'] - x) ** 2

        weighted_err_vit = weight * wx_vit * vit_sq_error * m_vit

        num_active_elements_v = m_vit.sum() * x.shape[1] * x.shape[2] * x.shape[3]
        vit_loss = weighted_err_vit.sum() / (num_active_elements_v + 1e-6)

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
            # NEW: Statistics for monitoring
            "unet_training_samples": unet_mask.sum().item(),
            "vit_training_samples": vit_mask.sum().item(),
            "overlap_samples": (unet_mask & vit_mask).sum().item(),
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


class EDM_LOSS_LPD(nn.Module):
    def __init__(self,
                 expert_centers_vit: torch.Tensor,
                 expert_centers_Unet: torch.Tensor,
                 num_experts: int,
                 sigma_data: float = 0.5,
                 Unet_bal: float = 0.0005,
                 vit_bal: float = 0.0005,
                 z_bal: float = 0.0001,
                 prior_bal: float = 0.001,
                 p_mean: float = -0.4,  # Adjusted default as discussed
                 p_std: float = 1.6,
                 sharpness: float = 2.0,
                 # LPD Configuration
                 laplacian_sigma: float = 1.5,
                 laplacian_kernel: int = 9
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.sigma_data = sigma_data
        self.Unet_lambda = Unet_bal
        self.vit_lambda = vit_bal
        self.z_bal = z_bal
        self.prior_bal = prior_bal

        # Laplacian Params
        self.laplacian_sigma = laplacian_sigma
        self.laplacian_kernel = laplacian_kernel

        # Path Priors (Keep existing logic)
        self.prior_path_loss_vit = PathPriorLoss(p_mean=p_mean, p_std=p_std,
                                                 expert_centers=expert_centers_vit, sharpness=sharpness)
        self.prior_path_loss_Unet = PathPriorLoss(p_mean=p_mean, p_std=p_std,
                                                  expert_centers=expert_centers_Unet, sharpness=sharpness)

    def __call__(self,
                 x: torch.Tensor,  # Clean Image (Target)
                 noisy_x: torch.Tensor,  # Noisy Input (x + n)
                 sigma: torch.Tensor,
                 out_model: dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:

        # 1. EDM Constants Calculation
        sigma = sigma.reshape(-1, 1, 1, 1)
        # Standard EDM Preconditioning coefficients
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        # c_in is handled inside the model, but needed for target derivation logic if we had raw inputs.
        # Since 'noisy_x' is already the raw input, we use it directly.

        # 2. Calculate the Ideal "F" Target
        # In EDM, D(x) = c_skip * x_in + c_out * F(x_in).
        # We want to train F(x_in) to predict: (x_clean - c_skip * x_in) / c_out
        target_F = (x - c_skip * noisy_x) / c_out

        # 3. Laplacian Decomposition of the TARGET F
        # We decompose the *feature target*, not just the pixel target.
        # This forces the network branches to specialize in the preconditioned space.
        target_F_low, target_F_high = laplacian_decomposition(
            image=target_F,
            kernel_size=self.laplacian_kernel,
            sigma=self.laplacian_sigma
        )

        # 4. Retrieve Raw Branch Outputs
        # Note: These keys must exist in your HDMOEM return dict
        # These are the outputs of output_proj_vit and output_proj_unet *before* blending
        pred_vit_F = out_model['vit_denoised']
        pred_unet_F = out_model['unet_denoised']

        # 5. EDM Weighting (Critical for stability)
        # This weight balances the loss magnitude across different noise levels
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # 6. Calculate Orthogonal Losses
        # ViT Branch Loss (Structure)
        vit_loss = torch.mean(weight * (pred_vit_F - target_F_low) ** 2)

        # UNet Branch Loss (Texture)
        # This is the "Texture Forcing" term
        unet_loss = torch.mean(weight * (pred_unet_F - target_F_high) ** 2)

        # 7. Global Consistency Loss
        # We also compute loss on the final blended 'denoised' output vs clean image
        # This ensures the summation A+B is correct.
        diff = out_model["denoised"] - x

        if out_model["log_var"] is None:
            pure_loss = torch.mean(weight * (diff ** 2))
        else:
            # Uncertainty-aware loss (if enabled)
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            pure_loss = torch.mean((weight * (diff ** 2)) / log_var.exp() + log_var)

        denoising_loss = torch.mean(diff ** 2)  # Raw MSE for logging

        # 8. Auxiliary Losses (Router & Z-Loss)
        router_loss = (self.Unet_lambda * self.load_balance(out_model["Unet_router_loss"], self.num_experts) +
                       self.vit_lambda * self.load_balance(out_model["vit_router_loss"], self.num_experts)).clamp(
            max=50)

        scaling_loss = (self.prior_bal * (self.prior_path_loss_vit(out_model["vit_router_loss"], sigma) +
                                          self.prior_path_loss_Unet(out_model["Unet_router_loss"], sigma))).clamp(max=50)

        z_los = (self.z_bal * self.z_loss(out_model["Unet_raw"]) +
                 self.z_bal * self.z_loss(out_model["vit_raw"])).clamp(max=50)

        # 9. Total Loss Summation
        total_loss = (pure_loss + vit_loss + unet_loss + z_los + router_loss + scaling_loss).clamp(max=50)

        return {
            "loss": total_loss,
            "loss_texture": unet_loss,
            "loss_structure": vit_loss,
            "denoising": denoising_loss,
            "balance": router_loss,
            "z_loss": z_los,
            "entropy": scaling_loss,
            "pure_loss": pure_loss
        }

    @staticmethod
    def load_balance(gate_probs, num_experts):
        P = gate_probs.mean(dim=0)
        return num_experts * torch.sum(P ** 2)

    @staticmethod
    def z_loss(logits):
        logits = logits.clamp(min=-50, max=50)
        z = torch.logsumexp(logits, dim=-1) ** 2
        return torch.mean(z)


class EDM_LOSS(nn.Module):
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
                 sharpness: float = 2.0
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.sigma_data = sigma_data
        self.Unet_lambda = Unet_bal
        self.vit_lambda = vit_bal
        self.z_bal = z_bal
        self.prior_bal = prior_bal
        self.prior_path_loss_vit = PathPriorLoss(p_mean=p_mean, p_std=p_std, expert_centers=expert_centers_vit,
                                                 sharpness=sharpness)
        self.prior_path_loss_Unet = PathPriorLoss(p_mean=p_mean, p_std=p_std, expert_centers=expert_centers_Unet,
                                                  sharpness=sharpness)

    def __call__(self,
                 x: torch.Tensor,
                 sigma: torch.Tensor,
                 out_model: dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:


        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Main Denoising Loss
        diff = out_model["denoised"] - x
        if out_model["log_var"] is None:
            pure_loss = torch.mean(weight * (diff ** 2))
        else:
            # Uncertainty-aware loss
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            pure_loss = torch.mean((weight * (diff ** 2)) / log_var.exp() + log_var)

        denoising_loss = torch.mean(diff ** 2)  # Raw MSE for logging

        # Aux Losses
        router_loss = (self.Unet_lambda * self.load_balance(out_model["Unet_router_loss"], self.num_experts) +
                       self.vit_lambda * self.load_balance(out_model["vit_router_loss"], self.num_experts)).clamp(max=50)

        scaling_loss = (self.prior_bal * (self.prior_path_loss_vit(out_model["Unet_router_loss"], sigma) +
                                          self.prior_path_loss_Unet(out_model["vit_router_loss"], sigma))).clamp(max=50)

        z_los = (self.z_bal * self.z_loss(out_model["Unet_raw"]) +
                 self.z_bal * self.z_loss(out_model["vit_raw"])).clamp(max=50)

        total_loss = (pure_loss + z_los + router_loss + scaling_loss).clamp(max=50)

        return {
            "loss": total_loss,
            "denoising": denoising_loss,
            "balance": router_loss,
            "z_loss": z_los,
            "entropy": scaling_loss,
            "pure_loss": pure_loss
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

