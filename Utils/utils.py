from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import math
from copy import deepcopy
import os

def save_checkpoint(model,
                    optimizer,
                    step,
                    mse_score,
                    configs,
                    filename,
                    ema_instance = None
                    ):
    if "save_dir" in configs:
        save_path = configs["save_dir"]
    elif "model_configs" in configs and "save_dir" in configs["model_configs"]:
        save_path = configs["model_configs"]["save_dir"]
    else:
        save_path = "./checkpoints"

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'mse': mse_score,
        'config': configs
    }
    if ema_instance is not None:
        # We save the underlying model inside the EMA class
        checkpoint['ema_model_state_dict'] = ema_instance.ema_model.state_dict()

    torch.save(checkpoint, str(full_path))
    print(f"   [Save] Checkpoint saved: {full_path}")

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


def register_gradient_scalers(model):
    """
    Registers hooks to boost gradients for Expert layers specifically.
    This helps overcome the "stiffness" of MP-initialized weights.
    """
    hooks = []

    # Target the experts
    target_modules = [model.net.Unet_experts,
                      model.net.VIT_experts,
                      model.net.input_proj,  # <--- ADD THIS
                      model.net.input_time_proj
                      ]

    for module in target_modules:
        for name, param in module.named_parameters():
            if param.requires_grad:
                # We define a lambda that multiplies the grad by 100
                # logic: 1e-7 (current) * 100 -> 1e-5 (healthy start)
                h = param.register_hook(lambda grad: grad * 1000.0)
                hooks.append(h)

    print(f"Registered Gradient Boosters (x100) on {len(hooks)} expert parameters.")
    return hooks

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
            cur_nimg: int = 0
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
        self.cur_nimg = cur_nimg
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
            bandwidth: float = 0.9,
            min_bandwidth: float = 0.3,
            min_active: int = 1,
            total_steps: int = 5000,
            step_size: float = 0.1,
            noise_range: tuple = (0.002, 80.0), # Example: (2.5, 80.0) for UNet
            strat_band: str = "step",
            sigma_min: float = 0.002,
            sigma_max: float = 80.0
    ):
        """
        Attributes-Aware Mask Generator (Rank-Based).
        Maps experts to noise levels based on their RANK of physical scale.
        Ensures equidistant spacing even if attributes are identical.
        """
        super().__init__()
        self.strat_band = strat_band
        self.total_steps = total_steps
        self.min_bw = min_bandwidth
        self.step_size = step_size
        self.p_mean = p_mean
        self.p_std = p_std
        self.bandwidth = bandwidth
        self.min_active = min_active
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        attrs = torch.tensor(expert_attributes, dtype=torch.float32)
        num_experts = len(attrs)
        _, sorted_indices = torch.sort(attrs, stable=True)
        
        # Fix 4: Use mixture-CDF quantile seeding instead of pure Gaussian CDF
        from Utils.cdf_utils import branch_local_quantile_centers
        min_r, max_r = noise_range
        cdf_points = branch_local_quantile_centers(
            num_experts, min_r, max_r,
            self.p_mean, self.p_std, self.sigma_min, self.sigma_max,
        )
        
        final_centers = torch.zeros_like(attrs)
        final_centers[sorted_indices] = cdf_points
        self.register_buffer('expert_centers', final_centers)

    @torch.no_grad()
    def __call__(self, sigma: torch.Tensor, step: int) -> torch.Tensor:
        device = sigma.device
        log_sigma = torch.log(sigma.flatten().clamp(min=1e-8))
        
        # Fix 4: Use mixture CDF to match quantile-seeded centers
        from Utils.cdf_utils import mixture_cdf
        sigma_percentile = mixture_cdf(
            log_sigma, self.p_mean, self.p_std,
            math.log(self.sigma_min), math.log(self.sigma_max),
        )
        sp_reshaped = sigma_percentile.view(-1, 1)
        
        experts_reshaped = self.expert_centers.to(device).view(1, -1)
        dist = torch.abs(sp_reshaped - experts_reshaped)
        current_bw = self.bandwidth_scheduler(step)
        mask = (dist <= current_bw).float()
        _, top_indices = torch.topk(-dist, k=self.min_active, dim=-1)
        mask.scatter_(1, top_indices, 1.0)
        return mask

    def bandwidth_scheduler(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_bw  # Changed from hardcoded 1.0 to max_bw

        if self.strat_band == 'linear':
            prog = step / float(self.total_steps)
            return self.bandwidth + (self.min_bw - self.bandwidth) * prog

        elif self.strat_band == 'step':
            # Calculate interval based on total steps / num_intervals (or step_size fraction)
            # Using step_size as a fraction (e.g. 0.1 for 10%)
            interval_size = self.total_steps * self.step_size
            current_interval = int(step / interval_size)

            # Max progress is 1.0
            total_intervals = int(1.0 / self.step_size)
            progress = min(current_interval / total_intervals, 1.0)

            return self.bandwidth + (self.min_bw - self.bandwidth) * progress
        else:
            return self.bandwidth

#to be revised against EDM's PostHoc EMA implementation
class EMA:
    """
    Maintains a shadow copy of the model parameters using Exponential Moving Average.
    Implements the 'Power Function' warmup to prevent zero-bias at the start.
    """

    def __init__(self,
                 model,
                 beta: float=0.999,
                 warmup_power:float = 2 / 3,
                 min_warmup_step=0
                 ):
        self.beta = beta
        self.warmup_power = warmup_power
        self.min_warmup_step = min_warmup_step
        self.step = 0

        # Create a shadow copy of the model
        # We turn off gradients for the shadow model to save memory/compute
        self.ema_model = deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # Map parameters for fast updates
        self.source_dict = dict(model.named_parameters())
        self.ema_dict = dict(self.ema_model.named_parameters())

    def update(self):
        self.step += 1

        # 1. Calculate the Effective Decay (Power Function Warmup)
        # This prevents the EMA from staying at 0.0 in the early steps.
        # Logic: decay starts small (fast updates) and grows to self.beta (slow smoothing).
        cur_beta = self.beta
        if self.step < self.min_warmup_step:  # Standard "warmup" period
            # A simple power-law approximation often used in EDM implementations
            # Effective beta = 1 - (1/step) roughly
            cur_beta = min(self.beta, (1 + self.step) / (10 + self.step))

        # 2. Update Shadow Parameters
        for name, source_param in self.source_dict.items():
            if name in self.ema_dict:
                ema_param = self.ema_dict[name]

                # Apply EMA formula: Shadow = Beta * Shadow + (1-Beta) * Current
                ema_param.data.mul_(cur_beta).add_(source_param.data, alpha=1.0 - cur_beta)

    def copy_to(self, target_model):
        """Copies the EMA weights into a target model (usually for saving inference)"""
        m_param = dict(target_model.named_parameters())
        for name, param in self.ema_dict.items():
            if name in m_param:
                m_param[name].data.copy_(param.data)