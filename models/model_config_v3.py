import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from models.model_internals import (
    MP_Conv, MP_Fourier, PooledCrossAttn,
    mp_silu, normalize,
)
from models.model_components import ParametricTrajectoryRouter, Unet_expert, Vit_expert

def router_to_experts(
    x:          torch.Tensor,      # (B, C, H, W)
    experts:    nn.ModuleList,
    out_router: torch.Tensor,      # (B, E)  soft routing weights
    time_emb:   torch.Tensor,      # (B, time_emb_dim)
    text_emb:   Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Weighted sum of expert outputs.

    Denominator is clamped to 0.1 so a sparse router that only activates
    a subset of experts on a given sample never divides by near-zero.
    """
    denom = out_router.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=0.1).view(-1, 1, 1, 1)

    if text_emb is not None and text_emb.ndim == 3:
        text_processed = text_emb.mean(dim=1)   # (B, text_dim)
    else:
        text_processed = text_emb

    output = torch.zeros_like(x)

    for i, expert in enumerate(experts):
        weights = out_router[:, i]           # (B,)
        mask    = weights > 1e-3             # skip truly zero-weighted samples

        if not mask.any():
            continue

        out_exp = expert(
            x        = x[mask],
            time_emb = time_emb[mask],
            text_emb = text_processed[mask] if text_processed is not None else None,
        )
        output[mask] += out_exp * weights[mask].view(-1, 1, 1, 1)

    return output / denom

class HDMOEM_v3(nn.Module):
    """
    HDMOEM v3 — Trajectory-Gated Cross-Attention.

    Forward data flow
    ──────────────────
    x_in (preconditioned noisy latent)
        │
        ▼
    input_proj  →  x_feats (B, C, H, W)
        │
        ├─ input_time_proj(time_emb) → gain  → x_feats = normalize(x_feats × (1+gain))
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
    Unet experts (router-gated)                      ViT experts (router-gated)
    out_Unet_expert (B, C, H, W)                     out_Vit_expert (B, C, H, W)
        │                                                  │
        ▼                                                  ▼
    output_proj_unet (3×3, zero-init)          output_proj_vit (5×5, xavier)
    unet_out (B, in_ch, H, W)                         vit_out (B, in_ch, H, W)
        │                                                  │
        └──────────────┬───────────────────────────────────┘
                       ▼
    compute_blend_weights(σ)  →  w_vit, w_unet, gate_strength
        │
        ├─ blend_residual(time_emb) → residual correction on w_vit
        │
        ├─ PooledCrossAttn(out_Unet_expert, out_Vit_expert, gate_strength)
        │  → attn_correction (B, C, H, W)   [zero at step 0]
        │
        ▼
    out = w_vit × vit_out + w_unet × unet_out
        + gate_strength × output_proj_attn(attn_correction)

    Parameters
    ──────────────────────────────────────────────────────────────────────────
    All v2 parameters are retained unchanged.  New v3-only parameters:

    pool_size (int):
        Spatial size of the pooled grid for PooledCrossAttn.
        Default 4 → 16 tokens per branch (cost = O(256) per head).
        Larger values add spatial resolution to the attention but increase cost.

    attn_pool_balance (float):
        attn_balance passed to the MP_Attention inside PooledCrossAttn.
        Controls how strongly the cross-attention residual modifies the query.
        Default 0.5 (equal blend with original query).
    """

    def __init__(
        self,
        IN_in_channels:       int,
        IN_img_resolution:    int,
        internal_channels:    int,
        time_emb_dim:         int,
        text_emb_dim:         int,
        num_experts:          int,
        top_k:                int,
        Fourier_bandwidth:    float,
        VIT_num_blocks:       int,
        VIT_patch_sizes:      List[int],
        VIT_num_groups:       int,
        VIT_num_heads:        int,
        VIT_emb_size:         int,
        VIT_pos_emb_balance:  List[float],
        VIT_emb_balance:      List[float],
        Unet_num_blocks:      int,
        Unet_channel_mult:    List[int],
        Unet_kernel_sizes:    List[Tuple[int, int]],
        Unet_model_channels:  Optional[int]          = 192,
        Unet_channel_mult_emb: Optional[int]         = None,
        Unet_label_balance:   Optional[float]        = 0.5,
        Unet_concat_balance:  Optional[tuple]        = None,
        transition_sigma:     float                  = 3.0,
        transition_sharpness: float                  = 1.5,
        pool_size:            int                    = 4,
        attn_pool_balance:    float                  = 0.5,
    ):
        super().__init__()
        self.internal_channels = internal_channels
        self.num_experts        = num_experts
        self.transition_sigma   = transition_sigma
        self.transition_sharpness = transition_sharpness

        self.input_proj = MP_Conv(
            in_channels  = IN_in_channels,
            out_channels = internal_channels,
            kernel       = (3, 3),
        )

        self.Fourier_emb  = MP_Fourier(num_channels=time_emb_dim // 2,
                                       bandwidth=Fourier_bandwidth)
        self.out_fourier1 = MP_Conv(in_channels=time_emb_dim // 2,
                                   out_channels=time_emb_dim * 2, kernel=())
        self.out_fourier2 = MP_Conv(in_channels=time_emb_dim * 2,
                                   out_channels=time_emb_dim,     kernel=())

        self.input_time_proj = MP_Conv(
            in_channels  = time_emb_dim,
            out_channels = internal_channels,
            kernel       = (),
        )
        nn.init.constant_(self.input_time_proj.weights, 0.0)

        global_sigma_min = 0.002
        global_sigma_max = 80.0
        
        # 💥 SWAPPED: U-Net is now fenced to HIGH NOISE
        self.Unet_router = ParametricTrajectoryRouter(
            num_experts=num_experts, 
            min_sigma=transition_sigma, 
            max_sigma=global_sigma_max,
            p_mean=-0.2,   # MUST MATCH CONFIG
            p_std=1.2      # MUST MATCH CONFIG  
        )
        
        # 💥 SWAPPED: ViT is now fenced to LOW NOISE
        self.vit_router = ParametricTrajectoryRouter(
            num_experts=num_experts, 
            min_sigma=global_sigma_min, 
            max_sigma=transition_sigma,
            p_mean=-0.2,   # MUST MATCH CONFIG
            p_std=1.2      # MUST MATCH CONFIG
        )

        if Unet_concat_balance is None:
            balances = [0.5] * num_experts
        elif isinstance(Unet_concat_balance, (float, int)):
            balances = [float(Unet_concat_balance)] * num_experts
        else:
            assert len(Unet_concat_balance) == num_experts, (
                f"Unet_concat_balance has {len(Unet_concat_balance)} entries "
                f"but num_experts = {num_experts}."
            )
            balances = list(Unet_concat_balance)

        self.Unet_experts = nn.ModuleList([
            Unet_expert(
                img_resolution  = IN_img_resolution,
                img_channels    = internal_channels,
                time_emb_dim    = time_emb_dim,
                text_emb_dim    = text_emb_dim,
                num_blocks      = Unet_num_blocks,
                channel_mult    = Unet_channel_mult,
                kernel_size     = Unet_kernel_sizes[i],
                label_balance   = Unet_label_balance,
                concat_balance  = balances[i],
                model_channels  = Unet_model_channels,
                channel_mult_emb= Unet_channel_mult_emb,
            )
            for i in range(num_experts)
        ])

        self.VIT_experts = nn.ModuleList([
            Vit_expert(
                num_heads       = VIT_num_heads,
                num_groups      = VIT_num_groups,
                in_channels     = internal_channels,
                seq_ln          = math.ceil(IN_img_resolution / VIT_patch_sizes[i]) ** 2,
                emb_dim         = VIT_emb_size,
                num_blocks      = VIT_num_blocks,
                patch_size      = VIT_patch_sizes[i],
                text_dim        = text_emb_dim,
                time_dim        = time_emb_dim,
                emb_balance     = VIT_emb_balance[i],
                pos_emb_balance = VIT_pos_emb_balance[i],
            )
            for i in range(num_experts)
        ])

        self.output_proj_vit = nn.Conv2d(
            in_channels  = internal_channels,
            out_channels = IN_in_channels,
            kernel_size  = 1,
            padding      = 0,
        )
        self.output_proj_unet = nn.Conv2d(
            in_channels  = internal_channels,
            out_channels = IN_in_channels,
            kernel_size  = 3,
            padding      = 1,
        )
        nn.init.xavier_normal_(self.output_proj_unet.weight)
        nn.init.constant_(self.output_proj_vit.weight, 0.0)
        nn.init.constant_(self.output_proj_vit.bias,   0.0)

        self.blend_residual = nn.Sequential(
            MP_Conv(in_channels=time_emb_dim, out_channels=time_emb_dim // 2, kernel=()),
            MP_Conv(in_channels=time_emb_dim // 2, out_channels=1, kernel=()),
        )
        nn.init.constant_(self.blend_residual[-1].weights, 0.0)

        if internal_channels % VIT_num_heads == 0:
            self.pooled_cross_attn = PooledCrossAttn(
                channels     = internal_channels,
                num_heads    = VIT_num_heads,
                pool_size    = pool_size,
                attn_balance = attn_pool_balance,
            )
            self._cross_attn_enabled = True
        else:
            self.pooled_cross_attn   = None
            self._cross_attn_enabled = False
            print(
                f"[HDMOEM_v3] WARNING: internal_channels ({internal_channels}) "
                f"is not divisible by VIT_num_heads ({VIT_num_heads}). "
                f"PooledCrossAttn is DISABLED. To enable it, ensure "
                f"internal_channels % VIT_num_heads == 0."
            )

        if self._cross_attn_enabled:
            self.output_proj_attn = MP_Conv(
                in_channels  = internal_channels,
                out_channels = IN_in_channels,
                kernel       = (1, 1),
            )
            nn.init.constant_(self.output_proj_attn.weights, 0.0)

    def compute_blend_weights(
            self,
            sigma:    torch.Tensor,    # (B,) — *reconstructed* σ values
            time_emb: torch.Tensor,    # (B, time_emb_dim) — for learned residual
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            
        log_sigma = torch.log(sigma + 1e-8)
        log_transition = math.log(self.transition_sigma)
        
        # FIX: We want ViT at LOW noise (sigma < transition_sigma) -> w_vit -> 1
        # When sigma < transition_sigma, (log_transition - log_sigma) is POSITIVE.
        x_prior = (log_transition - log_sigma) / self.transition_sharpness
        x_prior  = x_prior.view(-1, 1, 1, 1)                        # (B,1,1,1)

        h        = self.blend_residual[0](time_emb)                  
        h        = mp_silu(h)
        residual = self.blend_residual[1](h).view(-1, 1, 1, 1)       

        # w_vit -> 1 at low noise, w_unet -> 1 at high noise
        w_vit    = torch.sigmoid(x_prior + residual)
        w_unet   = 1.0 - w_vit

        gate_strength = w_vit * w_unet                               # ∈ [0, 0.25]

        return w_vit, w_unet, gate_strength

    def forward(
        self,
        x:               torch.Tensor,    # (B, C_in, H, W) — preconditioned input
        time_vec:        torch.Tensor,    # (B,)             — c_noise = log(σ)/4
        text_emb:        Optional[torch.Tensor],
        Unet_router_mask: torch.Tensor,   # (B, num_experts)
        Vit_router_mask:  torch.Tensor,   # (B, num_experts)
        zeta:             float,
    ) -> dict:
        """
        Forward pass of HDMOEM v3.

        Returns a dict with keys matching the v2 return signature so the
        existing loss function and logger work without modification:

            out          : (B, C_in, H, W)  final denoised features
            vit_out      : (B, C_in, H, W)  ViT branch prediction (for branch loss)
            unet_out     : (B, C_in, H, W)  UNet branch prediction (for branch loss)
            w_vit        : (B, 1, 1, 1)     learned blend weight for ViT
            w_unet       : (B, 1, 1, 1)     learned blend weight for UNet
            gate_strength: (B, 1, 1, 1)     transition gate (for logging)
            Unet_probs   : (B, num_experts)  router soft probabilities
            Unet_raw     : (B, num_experts)  router raw logits
            Vit_probs    : (B, num_experts)  router soft probabilities
            Vit_raw      : (B, num_experts)  router raw logits
        """
        # ── Step 1: Time embedding ────────────────────────────────────────────
        time_embed = self.Fourier_emb(x=time_vec)
        time_embed = self.out_fourier1(x=time_embed)
        time_embed = self.out_fourier2(x=mp_silu(x=time_embed))   # (B, time_emb_dim)

        x_feats   = self.input_proj(x)                              # (B, C, H, W)
        time_gain = self.input_time_proj(time_embed)                # (B, C)
        time_gain = time_gain.view(-1, self.internal_channels, 1, 1)
        x_feats   = normalize(x_feats) * (1.0 + time_gain)      # magnitude-safe

        # Extract the pure 1D log_sigma scalar
        log_sigma = time_vec.view(-1, 1) * 4.0 

        out_vit_router,  Vit_probs,  Vit_raw  = self.vit_router(
            log_sigma=log_sigma, zeta=zeta, mask=Vit_router_mask)
        out_unet_router, Unet_probs, Unet_raw = self.Unet_router(
            log_sigma=log_sigma, zeta=zeta, mask=Unet_router_mask)

        # Apply the first output (out_unet_router / out_vit_router) to the experts
        out_Unet_expert = router_to_experts(
            x=x_feats, experts=self.Unet_experts, out_router=out_unet_router,
            time_emb=time_embed, text_emb=text_emb,
        )
        
        out_Vit_expert = router_to_experts(
            x=x_feats, experts=self.VIT_experts, out_router=out_vit_router,
            time_emb=time_embed, text_emb=text_emb,
        )
        
        unet_out = self.output_proj_unet(out_Unet_expert)           # (B, C_in, H, W)
        vit_out  = self.output_proj_vit(out_Vit_expert)             # (B, C_in, H, W)
        sigma_reconstructed = torch.exp(4.0 * time_vec)             # (B,)
        w_vit, w_unet, gate_strength = self.compute_blend_weights(
            sigma    = sigma_reconstructed,
            time_emb = time_embed,
        )                         # each (B, 1, 1, 1)
        attn_correction = torch.zeros_like(unet_out)
        if self._cross_attn_enabled:
            raw_correction = self.pooled_cross_attn(
                unet_feat    = out_Unet_expert,    # (B, C, H, W)
                vit_feat     = out_Vit_expert,     # (B, C, H, W)
                gate_strength= gate_strength,      # (B, 1, 1, 1) — already in [0, 0.25]
            )                                                        # (B, C, H, W)
            attn_correction = self.output_proj_attn(raw_correction)  # (B, C_in, H, W)

        out = w_vit * vit_out + w_unet * unet_out + attn_correction

        return {
            "out":           out,
            "vit_out":       vit_out,
            "unet_out":      unet_out,
            "w_vit":         w_vit,
            "w_unet":        w_unet,
            "gate_strength": gate_strength,
            "Unet_probs":    Unet_probs,
            "Unet_raw":      Unet_raw,
            "Vit_probs":     Vit_probs,
            "Vit_raw":       Vit_raw,
        }

class preconditioned_HDMOEM_v3(nn.Module):
    """
    EDM Preconditioning wrapper for HDMOEM v3.

    Identical in structure to the v2 wrapper.  All new v3 parameters are
    forwarded to HDMOEM_v3.  The output dict keys are updated to expose the
    new signals (w_vit, w_unet, gate_strength) that the logger and loss
    function can consume directly — no dummy tensors.

    Return dict keys
    ─────────────────
        denoised      : Final D(x) after EDM skip connection.
        vit_denoised  : D(x) computed from the ViT branch only (for branch loss).
        unet_denoised : D(x) computed from the UNet branch only (for branch loss).
        w_vit         : Blend weight for ViT branch (after learned residual).
        w_unet        : Blend weight for UNet branch.
        gate_strength : Transition gate w_vit × w_unet (logging).
        Unet_router_loss: UNet router soft probabilities (for balance loss).
        Unet_raw      : UNet router raw logits.
        vit_router_loss : ViT router soft probabilities.
        vit_raw       : ViT router raw logits.
        log_var       : Learned log-variance (None unless return_log_var=True).
    """

    def __init__(
        self,
        IN_in_channels:        int,
        IN_img_resolution:     int,
        internal_channels:     int,
        time_emb_dim:          int,
        text_emb_dim:          int,
        num_experts:           int,
        top_k:                 int,
        Fourier_bandwidth:     float,
        VIT_num_blocks:        int,
        VIT_patch_sizes:       List[int],
        VIT_num_groups:        int,
        VIT_num_heads:         int,
        VIT_emb_size:          int,
        VIT_pos_emb_balance:   List[float],
        VIT_emb_balance:       List[float],
        Unet_num_blocks:       int,
        Unet_channel_mult:     List[int],
        Unet_kernel_sizes:     List[Tuple[int, int]],
        Unet_model_channels:   Optional[int]   = 192,
        Unet_channel_mult_emb: Optional[int]   = None,
        Unet_label_balance:    Optional[float] = 0.5,
        Unet_concat_balance:   Optional[tuple] = None,
        sigma_data:            Optional[float] = 0.5,
        log_var_channels:      Optional[int]   = 128,
        transition_sigma:      float           = 3.0,
        transition_sharpness:  float           = 1.5,
        # ── v3-only ──────────────────────────────────────────────────────────
        pool_size:             int             = 4,
        attn_pool_balance:     float           = 0.5,
    ):
        super().__init__()
        self.sigma_data       = sigma_data
        self.log_var_channels = log_var_channels
        self.log_var_fourier = MP_Fourier(num_channels=log_var_channels)
        self.log_var_linear  = MP_Conv(in_channels=log_var_channels,
                                       out_channels=1, kernel=())

        self.net = HDMOEM_v3(
            IN_in_channels        = IN_in_channels,
            IN_img_resolution     = IN_img_resolution,
            internal_channels     = internal_channels,
            time_emb_dim          = time_emb_dim,
            text_emb_dim          = text_emb_dim,
            num_experts           = num_experts,
            top_k                 = top_k,
            Fourier_bandwidth     = Fourier_bandwidth,
            VIT_num_blocks        = VIT_num_blocks,
            VIT_patch_sizes       = VIT_patch_sizes,
            VIT_num_groups        = VIT_num_groups,
            VIT_num_heads         = VIT_num_heads,
            VIT_emb_size          = VIT_emb_size,
            VIT_pos_emb_balance   = VIT_pos_emb_balance,
            VIT_emb_balance       = VIT_emb_balance,
            Unet_num_blocks       = Unet_num_blocks,
            Unet_channel_mult     = Unet_channel_mult,
            Unet_kernel_sizes     = Unet_kernel_sizes,
            Unet_model_channels   = Unet_model_channels,
            Unet_channel_mult_emb = Unet_channel_mult_emb,
            Unet_label_balance    = Unet_label_balance,
            Unet_concat_balance   = Unet_concat_balance,
            transition_sigma      = transition_sigma,
            transition_sharpness  = transition_sharpness,
            pool_size             = pool_size,
            attn_pool_balance     = attn_pool_balance,
        )

    def forward(
        self,
        x:                torch.Tensor,
        sigma:            torch.Tensor,
        text_emb:         Optional[torch.Tensor],
        Unet_router_mask: torch.Tensor,
        Vit_router_mask:  torch.Tensor,
        zeta:             float,
        return_log_var:   bool = False,
    ) -> dict:
        """
        EDM-preconditioned forward pass.

        sigma shape: (B,) or (B, 1, 1, 1).  Reshaped internally.
        """
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0], -1, -1, -1)

        c_skip  = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out   = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1.0 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4.0

        net_out = self.net(
            x                = x * c_in,
            time_vec         = c_noise,
            text_emb         = text_emb,
            Unet_router_mask = Unet_router_mask,
            Vit_router_mask  = Vit_router_mask,
            zeta             = zeta,
        )

        D_x      = c_skip * x + c_out * net_out["out"]
        D_x_vit  = c_skip * x + c_out * net_out["vit_out"]
        D_x_unet = c_skip * x + c_out * net_out["unet_out"]

        result = {
            "denoised":         D_x,
            "vit_denoised":     D_x_vit,
            "unet_denoised":    D_x_unet,
            "w_vit":            net_out["w_vit"],
            "w_unet":           net_out["w_unet"],
            "gate_strength":    net_out["gate_strength"],
            "Unet_router_loss": net_out["Unet_probs"],
            "Unet_raw":         net_out["Unet_raw"],
            "vit_router_loss":  net_out["Vit_probs"],
            "vit_raw":          net_out["Vit_raw"],
            "log_var":          None,
        }

        if return_log_var:
            log_var = self.log_var_linear(
                                    self.log_var_fourier(c_noise)
                                ).reshape(-1, 1, 1, 1)
            result["log_var"] = log_var

        return result
