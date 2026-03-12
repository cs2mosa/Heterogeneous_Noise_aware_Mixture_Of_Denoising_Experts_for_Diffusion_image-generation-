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
    Hybrid Diffusion Mixture of Experts Model (HDMOEM).

    This architecture integrates a U-Net path and a Vision Transformer (ViT) path,
    both utilizing Sparse Mixture of Experts (MoE) routing. The model processes
    noisy inputs through both paths in parallel, fuses them via simple addition
    (where U-Net features query ViT features), and applies a final learned gating
    mechanism (now implemented as a static sigmoid for testing)to produce the output.

    Key Features:
    1. **Dual Routing**: Independent routers for U-Net (local/spatial) and ViT (global/semantic) experts.
    2. **Adaptive Scaling**: A `scaling_net` modulates the input magnitude for both paths based on noise level. (omitted for testing)
    3. **Hybrid Fusion**: U-Net experts provide structural details, while ViT experts provide
       global context. These are fused via Magnitude-Preserving Cross-Attention.
    4. **Soft Gating**: A final MLP blends the pure U-Net output with the Attended output. (omitted for testing)

    Attributes:
        Unet_router (Router): Sparse Top-K router for U-Net experts.
        vit_router (Router): Sparse Top-K router for ViT experts.
        Unet_experts (nn.ModuleList): Collection of `Unet_expert` modules.
        VIT_experts (nn.ModuleList): Collection of `Vit_expert` modules.

    forward argument shapes:
        x (torch.Tensor) -> Shape: (Batch, In_Channels, Height, Width).
        time_vec (torch.Tensor) -> Shape: (Batch,).
        text_emb (torch.Tensor) -> Shape: (Batch, Seq_Len, Text_Emb_Dim) -> 3D Tensor.
        Unet_router_mask (torch.Tensor) -> Shape: (Batch, Num_Experts).
        Vit_router_mask (torch.Tensor)-> Shape: (Batch, Num_Experts).
        zeta (float): Exploration noise magnitude for the Router logits.
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
                 VIT_pos_emb_balance:list[float],
                 VIT_emb_balance:list[float],
                 Unet_num_blocks: int,
                 Unet_channel_mult: list[int],
                 Unet_kernel_sizes: List[Tuple[int, int]],
                 Unet_model_channels: Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance: Optional[float] = 0.5,
                 Unet_concat_balance: Optional[tuple[float]] = None,
                 transition_sigma: float = 3.0,  # NEW: Handoff point
                 transition_sharpness: float = 1.5,  # NEW: Blend smoothness
                 ):
        """

        :param IN_in_channels: input image channels (typically 4 for a VAE latent diffusion)
        :param IN_img_resolution: input image resolution (we deal with square images only in this implementation -> height = width)
        :param time_emb_dim: expected embedding dimensions of time vector
        :param text_emb_dim: expected embedding dimensions of time vector
        :param num_experts: number of expert per path
        :param top_k: router's topk -> 1 = switch router
        :param Fourier_bandwidth: default 1.0
        :param VIT_num_blocks: number of vision transformer blocks per VIT expert
        :param VIT_patch_sizes: list of patch sizes -> higher patch_size for high noise and vice versa
        :param VIT_num_groups: number of groups for group normalizations in VIT experts
        :param VIT_num_heads: number of attention heads per VIT expert
        :param VIT_emb_size: internal embedding dimension for each VIT expert
        :param Unet_num_blocks: number of Unet blocks per Unet expert
        :param Unet_channel_mult: list of channel multipliers for progressive increase in channels -> going in depth of each Unet expert
        :param Unet_kernel_sizes: list of kernel sizes (one kernel for each expert). typically going higher in kernel sizes while progressing the list
        :param Unet_model_channels: treated as embedding size of the VIT. typically 192 channels is enough
        :param Unet_channel_mult_emb: list of embedding channel multipliers for each unet expert
        :param Unet_label_balance: text balance of Unet
        :param Unet_concat_balance: concatenation balance of Unet
        """
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

        if Unet_concat_balance is None:
            # Default fallback
            balances = [0.5] * num_experts
        elif isinstance(Unet_concat_balance, (float, int)):
            # User passed a single value; broadcast it to all experts
            balances = [float(Unet_concat_balance)] * num_experts
        else:
            # User passed a list/tuple; validate length
            assert len(Unet_concat_balance) == num_experts, \
                f"Config Mismatch: 'Unet_concat_balance' has {len(Unet_concat_balance)} elements, but 'num_experts' is {num_experts}."
            balances = Unet_concat_balance
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
                concat_balance=balances[i],
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
                time_dim=time_emb_dim,
                emb_balance=VIT_emb_balance[i],
                pos_emb_balance= VIT_pos_emb_balance[i]
            ))

        # 8. Separate Output Projections (NO shared fusion)
        self.output_proj_unet = nn.Conv2d(
            in_channels=self.internal_channels,
            out_channels=IN_in_channels,
            kernel_size=3,
            padding=1
        )
        # CRITICAL: Zero-Initialization
        nn.init.normal_(self.output_proj_unet.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_proj_unet.bias, 0.0)

        self.output_proj_vit = nn.Conv2d(
            in_channels=self.internal_channels,
            out_channels=IN_in_channels,
            kernel_size=5,
            padding=2
        )
        # Initialize ViT to small values (not zero, to keep gradients flowing for structure)
        nn.init.xavier_normal_(self.output_proj_vit.weight)

        """self.blend_residual = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim // 2, 1)
        )
        nn.init.constant_(self.blend_residual[-1].weight, 0.0)
        nn.init.constant_(self.blend_residual[-1].bias, 0.0)"""

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
        x = (sigma - self.transition_sigma) / self.transition_sharpness
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
        """
        Forward pass of the Hybrid MoE.

        Args:
            :param x : Input noisy image/latent.
                              Shape: (Batch, In_Channels, Height, Width).
            :param time_vec: Raw time steps or noise levels (sigma).
                                     Shape: (Batch,).
            :param text_emb : Conditional text embeddings.
                                     Shape: (Batch, Seq_Len, Text_Emb_Dim) -> 3D Tensor.
            :param Unet_router_mask : Binary mask to disable specific U-Net experts.
                                             Shape: (Batch, Num_Experts).
            :param Vit_router_mask : Binary mask to disable specific ViT experts.
                                            Shape: (Batch, Num_Experts).
            :param zeta : Exploration noise magnitude for the Router logits.

        :returns tuple:
                - out_gated_attn (torch.Tensor): Final denoised output. Shape: (Batch, C, H, W).
                - Unet_gate_probs (torch.Tensor): Softmax probabilities from U-Net router. Shape: (Batch, Experts).
                - Vit_gate_probs (torch.Tensor): Softmax probabilities from ViT router. Shape: (Batch, Experts).
                - other values are for logging/ enhancing loss function control
        """
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
        sigma_reconstructed = torch.exp(4 * time_vec)
        w_vit, w_unet = self.compute_blend_weights(sigma_reconstructed)

        # 7. Blend outputs (simple weighted sum, no normalization needed since sum=1)
        #out = (vit_out + w_unet * unet_out)/ torch.sqrt(1+w_unet**2) for testing LPD
        out = w_vit * vit_out + w_unet * unet_out

        # 8. Return compatible output (dummy values for removed components)
        dummy_scaling = torch.stack([w_unet.flatten(), w_vit.flatten(), 
                                     torch.zeros_like(w_vit.flatten())], dim=1)
        dummy_gate = torch.stack([w_unet.flatten(), w_vit.flatten(), 
                                  torch.zeros_like(w_vit.flatten())], dim=1)
        
        return (out,
                vit_out,
                unet_out,
                Unet_gate_probs,
                Unet_raw,
                Vit_gate_probs,
                vit_raw,
                dummy_scaling,
                dummy_gate,
                w_vit,   # <--- Added
                w_unet)     # Same (for compatibility)


class preconditioned_HDMOEM(nn.Module):
    """
    EDM Preconditioning Wrapper for the HDMOEM.

    This class implements the "Elucidated Diffusion Models" (Karras et al.) preconditioning
    framework. It handles input scaling c_in, noise embedding c_noise,
    skip scaling c_skip, and output scaling c_out to ensure the neural network
    predicts the denoised signal D_X with unit variance targets.

    It optionally learns a log-variance parameter for improved likelihood estimation.

    Attributes:
        net (HDMOEM): The core denoising backbone.
        sigma_data (float): Expected standard deviation of the training data (usually 0.5).
        log_var_fourier (MP_Fourier): Fourier features for the learnable variance prediction.
        log_var_linear (MP_Conv): Projection for the learnable variance.

    forward argument shapes:
        x (torch.Tensor) -> Shape: (Batch, In_Channels, Height, Width).
        time_vec (torch.Tensor) -> Shape: (Batch,).
        text_emb (torch.Tensor) -> Shape: (Batch, Seq_Len, Text_Emb_Dim) for text conditioned generation. or (Batch, Text_Emb_Dim) for class conditioned generation.
        Unet_router_mask (torch.Tensor) -> Shape: (Batch, Num_Experts).
        Vit_router_mask (torch.Tensor)-> Shape: (Batch, Num_Experts).
        zeta (float): Exploration noise magnitude for the Router logits.
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
                 VIT_pos_emb_balance: list[float],
                 VIT_emb_balance: list[float],
                 Unet_num_blocks: int,
                 Unet_channel_mult: list[int],
                 Unet_kernel_sizes: List[Tuple[int, int]],
                 Unet_model_channels: Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance: Optional[float] = 0.5,
                 Unet_concat_balance: Optional[tuple[float]] = None,
                 sigma_data: Optional[float] = 0.5,
                 log_var_channels: Optional[int] = 128,
                 transition_sigma: float = 3.0,
                 transition_sharpness: float = 1.5,
                 ):
        """
        Args:
            ... (Same args as HDMOEM) ...
            sigma_data (float, optional): The standard deviation of the dataset (EDM hyperparam). Defaults to 0.5.
            log_var_channels (int, optional): Hidden dimension for the variance prediction network. Defaults to 128.
        """
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
            VIT_emb_balance=VIT_emb_balance,
            VIT_pos_emb_balance=VIT_pos_emb_balance,
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
        """
        Forward pass with EDM Preconditioning.

        Calculates: C_NOISE, C_IN, C_OUT,C_SKIP given the standard deviation of the data and the noise conditioning vector

        Args:
            x : Input noisy image/latent.
                              Shape: (Batch, In_Channels, Height, Width).
            sigma: Raw time steps or noise levels.
                                     Shape: (Batch,).
            text_emb : Conditional text embeddings.
                                     Shape: (Batch, Seq_Len, Text_Emb_Dim) for text conditioned generation. or (Batch, Text_Emb_Dim) for class conditioned generation
            Unet_router_mask : Binary mask to disable specific U-Net experts.
                                             Shape: (Batch, Num_Experts).
            Vit_router_mask : Binary mask to disable specific ViT experts.
                                            Shape: (Batch, Num_Experts).
            zeta : Exploration noise magnitude for the Router logits.

            return_log_var : If True, returns the learned log variance
                                             for likelihood weighting. Defaults to False.

        Returns:
            tuple of :
                - D_x (torch.Tensor): The estimated clean data (denoised). Shape: (Batch, C, H, W).
                - Unet_gate_probs (torch.Tensor): Load balancing probabilities for U-Net.
                - Vit_gate_probs (torch.Tensor): Load balancing probabilities for ViT.
                - log_var (torch.Tensor | None): Learned log-variance (if requested).
                - other return values are used for logging and for loss function computations
        """
        # EDM Preconditioning
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0], -1, -1, -1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        x_in = x * c_in

        # Forward
        out_net, vit_out, unet_out, Unet_probs, Unet_raw, Vit_probs, Vit_raw, scaling, gate, w_vit, w_unet = self.net(
            x=x_in,
            text_emb=text_emb,
            time_vec=c_noise,
            Unet_router_mask=Unet_router_mask,
            Vit_router_mask=Vit_router_mask,
            zeta=zeta
        )

        # 3. Output Scaling (EDM skip connection)
        # Global Output
        D_x = c_skip * x + c_out * out_net
        D_vit = c_skip * x + c_out * vit_out
        D_unet = c_skip * x + c_out * unet_out

        result = {
            "denoised": D_x,
            "vit_denoised": D_vit,  # Transformed to image space
            "unet_denoised": D_unet,  # Transformed to image space
            "w_vit": w_vit,  # Passed for loss weighting
            "w_unet": w_unet,  # Passed for loss weighting
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