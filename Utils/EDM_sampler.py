from typing import Optional
import numpy as np
import torch.nn as nn
import torch


class EDM_Sampler:
    def __init__(self,
                 model: nn.Module,
                 num_solve_steps: int = 32,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 rho: float = 7.0,
                 S_churn: float = 0.0,
                 S_min: float = 0.0,
                 S_max: float = float('inf'),
                 S_noise: float = 1.0,
                 guidance: float = 1.0,
                 dtype=torch.float32
                 ):
        self.model = model
        self.num_steps = num_solve_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s_churn = S_churn
        self.s_min = S_min
        self.s_max = S_max
        self.s_noise = S_noise
        self.guide = guidance
        self.dtype = dtype

    def denoise(self, x, sigma, text_emb,uncond_text_emb=None):
        """
        Denoise helper that handles Classifier-Free Guidance internally.
        """

        out_model = self.model(
            x=x,
            sigma=sigma,
            text_emb=text_emb,
            Unet_router_mask=None,
            Vit_router_mask=None,
            zeta=0,
        )
        D_x = out_model["denoised"].to(self.dtype)
        if self.guide == 1.0:
            return D_x

        emb_for_guide = uncond_text_emb if uncond_text_emb is not None else text_emb
        out_guide = self.model(
            x=x,
            sigma=sigma,
            text_emb=emb_for_guide,
            Unet_router_mask=None,
            Vit_router_mask=None,
            zeta=0,
        )
        ref_D_x = out_guide["denoised"].to(self.dtype)

        return ref_D_x.lerp(D_x, self.guide)

    @torch.no_grad()
    def sample(self,
               noise: torch.Tensor,
               text_emb: Optional[torch.Tensor] = None,
               uncond_text_emb: torch.Tensor = None
               ) -> torch.Tensor:

        device = noise.device
        step_indices = torch.arange(self.num_steps, dtype=self.dtype, device=device)

        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) *
                   (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho

        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        x_next = noise.to(self.dtype) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            if self.s_churn > 0 and self.s_min <= t_cur <= self.s_max:
                gamma = min(self.s_churn / self.num_steps, np.sqrt(2) - 1)
            else:
                gamma = 0

            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * torch.randn_like(x_cur)
            denoised = self.denoise(x_hat, t_hat, text_emb, uncond_text_emb)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                denoised_prime = self.denoise(x_next, t_next, text_emb, uncond_text_emb)
                d_prime = (x_next - denoised_prime) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next