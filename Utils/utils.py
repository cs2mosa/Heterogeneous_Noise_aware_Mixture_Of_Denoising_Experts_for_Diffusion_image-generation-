from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def sample_sigma_normal(batch_size, device,
                            p_mean=15.0, p_std=20.0,
                            sigma_min=0.002, sigma_max=80.0):
    """
    Sample sigma from truncated normal distribution.

    Designed for MoE: centers distribution higher than EDM
    to give high-noise experts more training.

    Args:
        sigma_max:
        device:
        sigma_min:
        batch_size:
        p_mean: Center of distribution (default: 15.0)
              - EDM effective mean ≈ 0.5 (too low for MoE!)
              - We use 15.0 to balance all experts
        p_std: Spread of distribution (default: 20.0)
              - Large std ensures coverage of full range
        sigma_min, sigma_max: Clamp range

    Returns:
        Tensor [batch_size, 1, 1, 1] with sigma values
    """
    # Sample from normal
    sigma = torch.randn(batch_size, 1, 1, 1, device=device)
    sigma = sigma * p_std + p_mean

    # Clamp to valid range
    sigma = torch.clamp(sigma, sigma_min, sigma_max)

    return sigma

def sample_sigma(batch_size: int,
                 uniform: bool = False,
                 p_mean: float = -0.4,
                 p_std: float = 1.0,
                 sigma_max: Optional[float] = 80.0,
                 sigma_min: Optional[float] = 0.002
                 ) -> torch.Tensor:
    if uniform:
        low = torch.log(torch.tensor(sigma_min))
        high = torch.log(torch.tensor(sigma_max))
        rnd_uniform = torch.rand([batch_size, 1, 1, 1])
        sigma_vec = (rnd_uniform * (high - low) + low).exp()
    else:
        sigma_vec = torch.randn([batch_size, 1, 1, 1])
        sigma_vec = (sigma_vec * p_std + p_mean).exp()
        sigma_vec = sigma_vec.clamp(min=sigma_min, max=sigma_max)
    return sigma_vec


def sample_sigma_hybrid(batch_size,
                        sigma_min=0.002,
                        sigma_max=80.0,
                        p_mean=-0.4,
                        p_std=1.0,
                        extreme_prob=0.2,
                        device='cuda'):
    n_lognormal = int(batch_size * (1 - extreme_prob))
    n_uniform = batch_size - n_lognormal

    rnd_normal = torch.randn([n_lognormal, 1, 1, 1], device=device)
    sigma_lognormal = (rnd_normal * p_std + p_mean).exp()

    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    u = torch.rand([n_uniform, 1, 1, 1], device=device)
    sigma_uniform = (u * (log_max - log_min) + log_min).exp()

    sigma = torch.cat([sigma_lognormal, sigma_uniform], dim=0)
    sigma = sigma.clamp(sigma_min, sigma_max)
    perm = torch.randperm(batch_size, device=device)
    return sigma[perm]


class GradBooster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, multiplier: int):
        ctx.multiplier = multiplier
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output * ctx.multiplier
        return grad_x, None


def boost(x, multiplier=100.0):
    return GradBooster.apply(x, multiplier)


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


class Lr_Scheduler:
    """
    EDM2 Learning Rate Scheduler with support for multiple parameter groups.
    Each group maintains its own base LR but follows the same schedule.
    """
    def __init__(
            self,
            optimizer,
            total_steps: int,
            batch_size: int,
            ref_batches: float = 70e3,
            rampup_Mimg: float = 10,
            dataset_size: int = 8200,
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.ref_batches = ref_batches
        self.rampup_Mimg = rampup_Mimg
        self.dataset_size = dataset_size
        self.total_steps = total_steps

        # Store base LR for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Track current number of images seen
        self.cur_nimg = 0
        self._last_lrs = self.base_lrs.copy()

        print(f"📊 EDM2 Scheduler initialized with {len(self.base_lrs)} parameter groups:")
        for i, lr in enumerate(self.base_lrs):
            print(f"   Group {i}: base_lr = {lr:.6f}")

    def step(self):
        """Call this after optimizer.step()"""
        # Compute the LR multiplier (same for all groups)
        lr_multiplier = self._compute_lr_multiplier(self.cur_nimg)

        # Apply multiplier to each group's base LR
        new_lrs = []
        for i, (param_group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            new_lr = base_lr * lr_multiplier
            param_group['lr'] = new_lr
            new_lrs.append(new_lr)

        # Update image counter
        self.cur_nimg += self.batch_size
        self._last_lrs = new_lrs

        return new_lrs

    def _compute_lr_multiplier(self, cur_nimg):
        """
        Compute the learning rate multiplier based on images seen.
        This multiplier is applied to ALL base learning rates.
        """
        multiplier = 1.0

        # Inverse sqrt decay
        if self.ref_batches > 0:
            multiplier /= np.sqrt(max(cur_nimg / (self.ref_batches * self.batch_size), 1))

        # Linear warmup
        if self.rampup_Mimg > 0:
            warmup_multiplier = min(cur_nimg / (self.rampup_Mimg * 1e6), 1)
            multiplier *= warmup_multiplier

        return multiplier

    def get_last_lr(self):
        """Returns list of current LRs for all parameter groups"""
        return self._last_lrs

    def get_lr_multiplier(self):
        """Returns the current LR multiplier (for logging)"""
        return self._compute_lr_multiplier(self.cur_nimg)

    def state_dict(self):
        """For checkpoint saving"""
        return {
            'cur_nimg': self.cur_nimg,
            'base_lrs': self.base_lrs,
            '_last_lrs': self._last_lrs,
        }

    def load_state_dict(self, state_dict):
        """For checkpoint loading"""
        self.cur_nimg = state_dict['cur_nimg']
        self.base_lrs = state_dict['base_lrs']
        self._last_lrs = state_dict['_last_lrs']


class ZetaScheduler:
    def __init__(self, total_steps: int, max_zeta: float, min_zeta: float = 0.0, strategy: str = 'cos',
                 alpha: float = 4.0, warmup_ratio: float = 0.05):
        self.total_steps = total_steps
        self.max_zeta = max_zeta
        self.min_zeta = min_zeta
        self.strategy = strategy
        self.alpha = alpha
        self.warmup_steps = int(total_steps * warmup_ratio)

    def get_zeta(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_zeta

        if step >= self.total_steps:
            return self.min_zeta

        curr_step = step - self.warmup_steps
        decay_total = self.total_steps - self.warmup_steps

        if self.strategy == 'cos':
            cosine_val = 0.5 * (1 + np.cos(np.pi * curr_step / decay_total))
            zeta = self.min_zeta + (self.max_zeta - self.min_zeta) * cosine_val

        elif self.strategy == 'exp':
            term = -self.alpha * (curr_step - (self.max_zeta / decay_total))
            #Safety clamp for exp to prevent overflow if alpha is huge
            term = max(min(term, 10), -10)
            decay_factor = np.exp(term)
            zeta = (self.max_zeta - self.min_zeta) * decay_factor + self.min_zeta
            zeta = max(min(zeta, self.max_zeta), self.min_zeta)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return float(zeta)


class MaskGenerator(nn.Module):
    def __init__(
            self,
            expert_attributes: list,  # List of kernel sizes or patch sizes
            p_mean: float = -0.4,
            p_std: float = 1.0,
            bandwidth: float = 0.3,
            max_bandwidth: float = 0.9,
            min_active: int = 1,
            total_steps: int = 5000,
            step_size: float = 0.1,
            noise_range: tuple = (0.0, 1.0),
            strat_band: str = "step",
    ):
        """
        Attributes-Aware Mask Generator (Rank-Based).
        Maps experts to noise levels based on their RANK of physical scale.
        Ensures equidistant spacing even if attributes are identical.
        """
        super().__init__()
        self.strat_band = strat_band
        self.total_steps = total_steps
        self.max_bw = max_bandwidth
        self.step_size = step_size
        self.p_mean = p_mean
        self.p_std = p_std
        self.bandwidth = bandwidth
        self.min_active = min_active

        attrs = torch.tensor(expert_attributes, dtype=torch.float32)
        num_experts = len(attrs)
        _, sorted_indices = torch.sort(attrs, stable=True)
        min_range, max_range = noise_range
        equidistant_points = torch.linspace(min_range, max_range, steps=num_experts)
        final_centers = torch.zeros_like(attrs)
        final_centers[sorted_indices] = equidistant_points
        self.register_buffer('expert_centers', final_centers)

    @torch.no_grad()
    def __call__(self, sigma: torch.Tensor, step: int) -> torch.Tensor:
        device = sigma.device
        s = sigma.flatten()
        log_sigma = s #torch.log(s) we will try using the normal distribution
        sigma_percentile = 0.5 * (1 + torch.erf((log_sigma - self.p_mean) / (self.p_std * np.sqrt(2))))
        sigma_percentile = sigma_percentile.clamp(0, 1)
        sp_reshaped = sigma_percentile.view(-1, 1)
        experts_reshaped = self.expert_centers.to(device).view(1, -1)
        dist = torch.abs(sp_reshaped - experts_reshaped)
        current_bw = self.bandwidth_scheduler(step)
        mask = (dist <= current_bw).float()
        _, top_indices = torch.topk(-dist, k=self.min_active, dim=-1)
        mask.scatter_(1, top_indices, 1.0)
        return mask

    def bandwidth_scheduler(self, step: int) -> float:
        if step < self.total_steps * 0.1:  # First 10%
            return 0.9  # Start WIDE - let all experts see data
        elif step < self.total_steps * 0.5:  # Next 40%
            # Anneal quickly
            progress = (step - 0.1 * self.total_steps) / (0.4 * self.total_steps)
            return 0.9 - 0.7 * progress  # 0.9 → 0.2
        else:
            return 0.2