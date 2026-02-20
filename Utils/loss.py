import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF


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
                 p_mean: float = 0.0,
                 p_std: float = 1.4,
                 sharpness: float = 5.0):
        super().__init__()
        self.register_buffer('expert_centers', expert_centers)
        self.p_mean = p_mean
        self.p_std = p_std
        self.sharpness = sharpness
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, gate_probs: torch.Tensor, sigma: torch.Tensor):
        device = sigma.device
        s = sigma.flatten()
        log_sigma = torch.log(s + 1e-8)
        sigma_percentile = 0.5 * (1 + torch.erf((log_sigma - self.p_mean) / (self.p_std * np.sqrt(2))))
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
                 # NEW: Trajectory decomposition params
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

    def compute_trajectory_masks(self, sigma: torch.Tensor):
        """
        Compute masks for trajectory decomposition.

        UNet trains: σ < transition + overlap
        ViT trains:  σ > transition - overlap
        Overlap region: both train

        Returns:
            unet_mask: bool tensor (B,) indicating UNet training samples
            vit_mask: bool tensor (B,) indicating ViT training samples
        """
        sigma_flat = sigma.flatten()

        # UNet: low noise region
        unet_mask = sigma_flat < (self.transition_sigma + self.overlap_width)

        # ViT: high noise region
        vit_mask = sigma_flat > (self.transition_sigma - self.overlap_width)

        return unet_mask, vit_mask

    def __call__(self,
                 x: torch.Tensor,
                 noisy_x: torch.Tensor,
                 sigma: torch.Tensor,
                 out_model: dict[str, torch.Tensor]
                 ) -> dict[str, torch.Tensor]:
        """
        EDM loss with trajectory decomposition.

        Key change: Each branch trains only on its designated noise range.
        """
        # Standard EDM preconditioning
        sigma = sigma.reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

        # Target in denoised space
        target = (x - c_skip * noisy_x) / c_out

        # EDM loss weighting
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        weight = weight.clamp(max=100.0)

        # === TRAJECTORY DECOMPOSITION ===
        # Get masks for which branch should train on which samples
        unet_mask, vit_mask = self.compute_trajectory_masks(sigma)

        # 1. Blended output loss (train on ALL samples)
        diff_blend = out_model["denoised"] - target

        if out_model["log_var"] is None:
            pure_loss = torch.mean(weight * (diff_blend ** 2))
        else:
            # Uncertainty-aware loss
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            pure_loss = torch.mean(
                (weight * (diff_blend ** 2)) / log_var.exp() + log_var
            )
        pure_loss = pure_loss.clamp(max=50)

        # 2. UNet branch loss (ONLY on low-noise samples)
        unet_loss = torch.tensor(0.0, device=x.device)
        if unet_mask.any():
            # Extract samples where UNet should train
            w_unet = weight[unet_mask]
            pred_unet = out_model['unet_denoised'][unet_mask]
            target_unet = target[unet_mask]

            unet_loss = torch.mean(w_unet * (pred_unet - target_unet) ** 2)
            unet_loss = unet_loss.clamp(max=50)

        # 3. ViT branch loss (ONLY on high-noise samples)
        vit_loss = torch.tensor(0.0, device=x.device)
        if vit_mask.any():
            # Extract samples where ViT should train
            w_vit = weight[vit_mask]
            pred_vit = out_model['vit_denoised'][vit_mask]
            target_vit = target[vit_mask]

            vit_loss = torch.mean(w_vit * (pred_vit - target_vit) ** 2)
            vit_loss = vit_loss.clamp(max=50)

        # Weighted combination of trajectory losses
        trajectory_loss = (
                self.alpha_blend * pure_loss +
                self.alpha_unet * unet_loss +
                self.alpha_vit * vit_loss
        )

        # === AUXILIARY LOSSES ===
        # Denoising loss (for logging)
        denoising_loss = torch.mean((out_model["denoised"] - x) ** 2)

        # Router load balance
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

        # Z-loss (router logit regularization)
        z_loss = (
                self.z_bal * self.z_loss(out_model["Unet_raw"]) +
                self.z_bal * self.z_loss(out_model["vit_raw"])
        ).clamp(max=50)

        # === TOTAL LOSS ===
        total_loss = (
                trajectory_loss +
                router_loss +
                scaling_loss +
                z_loss
        ).clamp(max=50)

        return {
            "loss": total_loss,
            "loss_texture": unet_loss,  # Now: UNet on low-noise
            "loss_structure": vit_loss,  # Now: ViT on high-noise
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
        weight = weight.clamp(max=100.0)

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

        pure_loss = pure_loss.clamp(max=50)
        denoising_loss = torch.mean(diff ** 2)  # Raw MSE for logging

        # 8. Auxiliary Losses (Router & Z-Loss)
        router_loss = (self.Unet_lambda * self.load_balance(out_model["Unet_router_loss"], self.num_experts) +
                       self.vit_lambda * self.load_balance(out_model["vit_router_loss"], self.num_experts)).clamp(
            max=50)

        scaling_loss = (self.prior_bal * (self.prior_path_loss_vit(out_model["Unet_router_loss"], sigma) +
                                          self.prior_path_loss_Unet(out_model["vit_router_loss"], sigma))).clamp(max=50)

        z_los = (self.z_bal * self.z_loss(out_model["Unet_raw"]) +
                 self.z_bal * self.z_loss(out_model["vit_raw"])).clamp(max=50)

        # 9. Total Loss Summation
        total_loss = (pure_loss + vit_loss + unet_loss + z_los + router_loss + scaling_loss).clamp(max=50)

        return {
            "loss": total_loss,
            "loss_texture": unet_loss,  # Check this in logs. Should behave differently than before.
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

        # --- FIX 1: Correct EDM Weighting (Karras et al. 2022) ---
        # Weight minimizes visual error, not just MSE.
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Clamp weight to prevent instability at very low sigma (0.002)
        weight = weight.clamp(max=100.0)

        # Main Denoising Loss
        diff = out_model["denoised"] - x
        if out_model["log_var"] is None:
            pure_loss = torch.mean(weight * (diff ** 2))
        else:
            # Uncertainty-aware loss
            log_var = out_model["log_var"].clamp(min=-10, max=10)
            pure_loss = torch.mean((weight * (diff ** 2)) / log_var.exp() + log_var)

        pure_loss = pure_loss.clamp(max=50)  # Safety clip
        denoising_loss = torch.mean(diff ** 2)  # Raw MSE for logging

        # Aux Losses
        router_loss = (self.Unet_lambda * self.load_balance(out_model["Unet_router_loss"], self.num_experts) +
                       self.vit_lambda * self.load_balance(out_model["vit_router_loss"], self.num_experts)).clamp(max=50)

        scaling_loss = (self.prior_bal * (self.prior_path_loss_vit(out_model["Unet_router_loss"], sigma) +
                                          self.prior_path_loss_Unet(out_model["vit_router_loss"], sigma))).clamp(max=50)

        # Router Z-Loss to prevent logit explosion (ST-MoE)
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

