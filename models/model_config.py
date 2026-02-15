"""
Fixed HDMOEM Architecture

Key Changes:
1. REMOVED: Cross-attention (was suppressing ViT)
2. REMOVED: Learned 3-way gate (was causing instability)
3. ADDED: Deterministic noise-scheduled blending
4. ADDED: Separate output paths without interference

This implements your TRUE hypothesis:
- High noise (σ > 3): ViT handles coarse structure (large receptive field)
- Low noise (σ < 3): UNet handles fine details (small kernels)
- Transition: Smooth sigmoid blend
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from models.model_internals import MP_Conv, MP_Fourier, mp_silu
from models.model_components import Router, Unet_expert, Vit_expert


def router_to_experts(x: torch.Tensor,
                      experts: nn.ModuleList,
                      out_router: torch.Tensor,
                      time_emb: torch.Tensor,
                      text_emb: torch.Tensor,
                      ) -> torch.Tensor:
    """Routes inputs to experts - FIXED denominator handling"""
    # FIX: Ensure denominator never gets too small
    denom = out_router.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=0.1)  # Never divide by less than 0.1
    denom = denom.view(-1, 1, 1, 1)

    if text_emb is not None and text_emb.ndim == 3:
        text_processed = text_emb.mean(dim=1)
    else:
        text_processed = text_emb

    output = torch.zeros_like(x)

    for i, expert in enumerate(experts):
        weights = out_router[:, i]
        mask = weights > 1e-3  # Small threshold to avoid zero grads

        if not mask.any():
            continue

        x_expert = x[mask]
        time_expert = time_emb[mask]
        text_expert = text_processed[mask] if text_processed is not None else None

        out_exp = expert(x=x_expert, time_emb=time_expert, text_emb=text_expert)
        w = weights[mask].view(-1, 1, 1, 1)
        output[mask] += out_exp * w

    return output / denom


class HDMOEM(nn.Module):
    """
    Fixed Hybrid Diffusion MoE Model
    
    Changes from original:
    - Removed cross-attention
    - Removed learned gate
    - Added deterministic noise-based blending
    """

    def __init__(self,
                 IN_in_channels: int,
                 IN_img_resolution: int,
                 internal_channels: int,
                 time_emb_dim: int,
                 text_emb_dim: int,
                 num_experts: int,
                 top_k: int,
                 Fourier_bandwidth: float,
                 VIT_num_blocks: int,
                 VIT_patch_sizes: List[int],
                 VIT_num_groups: int,
                 VIT_num_heads: int,
                 VIT_emb_size: int,
                 Unet_num_blocks: int,
                 Unet_channel_mult: list[int],
                 Unet_kernel_sizes: List[Tuple[int, int]],
                 Unet_model_channels: Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance: Optional[float] = 0.5,
                 Unet_concat_balance: Optional[float] = 0.5,
                 transition_sigma: float = 3.0,  # NEW: Handoff point
                 transition_sharpness: float = 1.5,  # NEW: Blend smoothness
                 ):
        super().__init__()
        self.internal_channels = internal_channels
        self.num_experts = num_experts
        self.transition_sigma = transition_sigma
        self.transition_sharpness = transition_sharpness

        # 1. Input Projection
        self.input_proj = MP_Conv(
            in_channels=IN_in_channels,
            out_channels=self.internal_channels,
            kernel=(3, 3)
        )

        # 2. Fourier Time Embedding
        self.Fourier_emb = MP_Fourier(num_channels=time_emb_dim // 2, bandwidth=Fourier_bandwidth)
        self.out_fourier1 = MP_Conv(in_channels=time_emb_dim // 2, out_channels=time_emb_dim * 2, kernel=())
        self.out_fourier2 = MP_Conv(in_channels=time_emb_dim * 2, out_channels=time_emb_dim, kernel=())

        # 3. Routers
        self.Unet_router = Router(time_dim=time_emb_dim, top_k=top_k, num_experts=num_experts)
        self.vit_router = Router(time_dim=time_emb_dim, top_k=top_k, num_experts=num_experts)

        # 4. UNet Experts
        self.Unet_experts = nn.ModuleList()
        for i in range(num_experts):
            self.Unet_experts.append(Unet_expert(
                img_resolution=IN_img_resolution,
                img_channels=self.internal_channels,
                time_emb_dim=time_emb_dim,
                text_emb_dim=text_emb_dim,
                num_blocks=Unet_num_blocks,
                channel_mult=Unet_channel_mult,
                kernel_size=Unet_kernel_sizes[i],
                label_balance=Unet_label_balance,
                concat_balance=Unet_concat_balance,
                model_channels=Unet_model_channels,
                channel_mult_emb=Unet_channel_mult_emb
            ))

        # 5. ViT Experts
        self.VIT_experts = nn.ModuleList()
        for i in range(num_experts):
            self.VIT_experts.append(Vit_expert(
                num_heads=VIT_num_heads,
                num_groups=VIT_num_groups,
                in_channels=self.internal_channels,
                seq_ln=math.ceil(IN_img_resolution / VIT_patch_sizes[i]) ** 2,
                emb_dim=VIT_emb_size,
                num_blocks=VIT_num_blocks,
                patch_size=VIT_patch_sizes[i],
                text_dim=text_emb_dim,
                time_dim=time_emb_dim
            ))

        # 6. REMOVED: Cross-attention (was suppressing ViT)
        # 7. REMOVED: gate_mlp (using deterministic blending)

        # 8. Separate Output Projections (NO shared fusion)
        self.output_proj_vit = MP_Conv(
            in_channels=self.internal_channels, 
            out_channels=IN_in_channels,
            kernel=(5, 5)  # Larger kernel for ViT (coarse features)
        )
        self.output_proj_unet = MP_Conv(
            in_channels=self.internal_channels, 
            out_channels=IN_in_channels,
            kernel=(3, 3)  # Small kernel for UNet (fine details)
        )

    def compute_blend_weights(self, sigma: torch.Tensor) -> tuple:
        """
        Compute deterministic blend weights based on noise level.
        
        High noise (σ > transition): w_vit ≈ 1, w_unet ≈ 0
        Low noise (σ < transition):  w_vit ≈ 0, w_unet ≈ 1
        
        Returns:
            (w_vit, w_unet): Normalized weights that sum to 1
        """
        # Compute blend factor using sigmoid
        # x > 0 → w_vit high, x < 0 → w_unet high
        x = (sigma.flatten() - self.transition_sigma) / self.transition_sharpness
        w_vit = torch.sigmoid(x).view(-1, 1, 1, 1)
        w_unet = 1.0 - w_vit  # Ensures sum = 1
        
        return w_vit, w_unet

    def forward(self,
                x: torch.Tensor,
                time_vec: torch.Tensor,
                text_emb: torch.Tensor,
                Unet_router_mask: torch.Tensor,
                Vit_router_mask: torch.Tensor,
                zeta: float,
                ) -> tuple:

        # 1. Embed Time
        time_embed = self.Fourier_emb(x=time_vec)
        time_embed = self.out_fourier1(x=time_embed)
        time_embed = self.out_fourier2(x=mp_silu(x=time_embed))

        # 2. Initial Features
        x_feats = self.input_proj(x)

        # 3. Run Routers
        out_vit_router, Vit_gate_probs, vit_raw = self.vit_router(
            time_emb=time_embed, zeta=zeta, mask=Vit_router_mask
        )
        out_unet_router, Unet_gate_probs, Unet_raw = self.Unet_router(
            time_emb=time_embed, zeta=zeta, mask=Unet_router_mask
        )

        # 4. Run Experts (in parallel, independently)
        out_Unet_expert = router_to_experts(
            x=x_feats, experts=self.Unet_experts, out_router=out_unet_router,
            time_emb=time_embed, text_emb=text_emb
        )
        out_Vit_expert = router_to_experts(
            x=x_feats, experts=self.VIT_experts, out_router=out_vit_router,
            time_emb=time_embed, text_emb=text_emb
        )

        # 5. Project to output space (independently, no fusion)
        vit_out = self.output_proj_vit(out_Vit_expert)
        unet_out = self.output_proj_unet(out_Unet_expert)

        # 6. Compute deterministic blend weights based on noise
        # Note: We need sigma, not time_vec. Reconstruct it from time_vec.
        # time_vec = ln(sigma) / 4 → sigma = exp(4 * time_vec)
        sigma_reconstructed = torch.exp(4 * time_vec).view(-1, 1, 1, 1)
        w_vit, w_unet = self.compute_blend_weights(sigma_reconstructed)

        # 7. Blend outputs (simple weighted sum, no normalization needed since sum=1)
        out = w_vit * vit_out + w_unet * unet_out

        # 8. Return compatible output (dummy values for removed components)
        dummy_scaling = torch.stack([w_unet.flatten(), w_vit.flatten(), 
                                     torch.zeros_like(w_vit.flatten())], dim=1)
        dummy_gate = torch.stack([w_unet.flatten(), w_vit.flatten(), 
                                  torch.zeros_like(w_vit.flatten())], dim=1)
        
        return (out, 
                Unet_gate_probs,  # Real UNet router probs
                Unet_raw,          # Real UNet raw logits
                Vit_gate_probs,    # Real ViT router probs
                vit_raw,           # Real ViT raw logits
                dummy_scaling,     # Blend weights (for logging)
                dummy_gate)        # Same (for compatibility)


class preconditioned_HDMOEM(nn.Module):
    """EDM Preconditioning Wrapper for Fixed HDMOEM"""
    
    def __init__(self,
                 IN_in_channels: int,
                 IN_img_resolution: int,
                 internal_channels: int,
                 time_emb_dim: int,
                 text_emb_dim: int,
                 num_experts: int,
                 top_k: int,
                 Fourier_bandwidth: float,
                 VIT_num_blocks: int,
                 VIT_patch_sizes: List[int],
                 VIT_num_groups: int,
                 VIT_num_heads: int,
                 VIT_emb_size: int,
                 Unet_num_blocks: int,
                 Unet_channel_mult: list[int],
                 Unet_kernel_sizes: List[Tuple[int, int]],
                 Unet_model_channels: Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance: Optional[float] = 0.5,
                 Unet_concat_balance: Optional[float] = 0.5,
                 sigma_data: Optional[float] = 0.5,
                 log_var_channels: Optional[int] = 128,
                 transition_sigma: float = 3.0,
                 transition_sharpness: float = 1.5,
                 ):
        super().__init__()
        self.sigma_data = sigma_data
        self.log_var_channels = log_var_channels
        self.log_var_fourier = MP_Fourier(num_channels=self.log_var_channels)
        self.log_var_linear = MP_Conv(in_channels=self.log_var_channels, out_channels=1, kernel=())

        self.net = HDMOEM(
            IN_in_channels=IN_in_channels,
            IN_img_resolution=IN_img_resolution,
            internal_channels=internal_channels,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            num_experts=num_experts,
            top_k=top_k,
            Fourier_bandwidth=Fourier_bandwidth,
            VIT_num_blocks=VIT_num_blocks,
            VIT_patch_sizes=VIT_patch_sizes,
            VIT_num_groups=VIT_num_groups,
            VIT_num_heads=VIT_num_heads,
            VIT_emb_size=VIT_emb_size,
            Unet_num_blocks=Unet_num_blocks,
            Unet_channel_mult=Unet_channel_mult,
            Unet_kernel_sizes=Unet_kernel_sizes,
            Unet_model_channels=Unet_model_channels,
            Unet_channel_mult_emb=Unet_channel_mult_emb,
            Unet_label_balance=Unet_label_balance,
            Unet_concat_balance=Unet_concat_balance,
            transition_sigma=transition_sigma,
            transition_sharpness=transition_sharpness,
        )

    def forward(self,
                x: torch.Tensor,
                sigma: torch.Tensor,
                text_emb: torch.Tensor,
                Unet_router_mask: torch.Tensor,
                Vit_router_mask: torch.Tensor,
                zeta: float,
                return_log_var: bool = False
                ) -> dict:
        
        # EDM Preconditioning
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0], -1, -1, -1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        #c_noise = sigma.flatten().log() / 4
        c_noise = sigma.flatten() / 4
        x_in = x * c_in

        # Forward
        out_net, Unet_probs, Unet_raw, Vit_probs, Vit_raw, scaling, gate = self.net(
            x=x_in,
            text_emb=text_emb,
            time_vec=c_noise,
            Unet_router_mask=Unet_router_mask,
            Vit_router_mask=Vit_router_mask,
            zeta=zeta
        )

        # Output Scaling (EDM skip connection)
        D_x = c_skip * x + c_out * out_net

        result = {
            "denoised": D_x,
            "Unet_router_loss": Unet_probs,
            "Unet_raw": Unet_raw,
            "vit_router_loss": Vit_probs,
            "vit_raw": Vit_raw,
            "scaling_net_out": scaling,
            "out_gate": gate,
            "log_var": None
        }

        if return_log_var:
            log_var = self.log_var_linear(self.log_var_fourier(c_noise)).reshape(-1, 1, 1, 1)
            result["log_var"] = log_var

        return result
