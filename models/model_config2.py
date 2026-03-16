import math
from Utils.utils import boost
import torch
import torch.nn as nn
import model_internals as util
import model_components as m
import torch.nn.functional as F
from typing import Optional ,List,Tuple

#helper to abstract the router running operation
def router_to_unet_experts(x: torch.Tensor,
                      experts: nn.ModuleList,
                      out_router: torch.Tensor,
                      time_emb: torch.Tensor,
                      text_emb: torch.Tensor,
                      ) -> torch.Tensor:
    denom = torch.sqrt((out_router**2).sum(dim=-1, keepdim=True) + 1e-8)
    denom = denom.view(-1, 1, 1, 1)
    if text_emb is not None and text_emb.ndim == 3:
        # Average over the sequence dimension (dim 1)
        text_processed = text_emb.mean(dim=1)
    else:
        text_processed = text_emb
    output = torch.zeros_like(x)

    for i, expert in enumerate(experts):
        mask = out_router[:, i] > 0

        if not mask.any():
            continue

        x_expert = x[mask]
        time_expert = time_emb[mask]
        text_expert = text_processed[mask] if text_processed is not None else None
        out_exp = expert(x=x_expert, time_emb=time_expert, text_emb=text_expert)
        router_weights = out_router[mask, i]
        router_weights = router_weights.view(-1, 1, 1, 1)
        output[mask] += out_exp * router_weights

    return output / denom

#adding gating MLP network after the attention module with 2 units output and softmax activation -> linear interpolation with the 2 units across the out_unet and Out_VIT
class HDMOEM (nn.Module):
    """
    Hybrid Diffusion Mixture of Experts Model (HDMOEM).

    This architecture integrates a U-Net path and a Vision Transformer (ViT) path,
    both utilizing Sparse Mixture of Experts (MoE) routing. The model processes
    noisy inputs through both paths in parallel, fuses them via Cross-Attention
    (where U-Net features query ViT features), and applies a final learned gating
    mechanism to produce the output.

    Key Features:
    1. **Dual Routing**: Independent routers for U-Net (local/spatial) and ViT (global/semantic) experts.
    2. **Adaptive Scaling**: A `scaling_net` modulates the input magnitude for both paths based on noise level.
    3. **Hybrid Fusion**: U-Net experts provide structural details, while ViT experts provide
       global context. These are fused via Magnitude-Preserving Cross-Attention.
    4. **Soft Gating**: A final MLP blends the pure U-Net output with the Attended output.

    Attributes:
        Unet_router (Router): Sparse Top-K router for U-Net experts.
        vit_router (Router): Sparse Top-K router for ViT experts.
        Unet_experts (nn.ModuleList): Collection of `Unet_expert` modules.
        VIT_experts (nn.ModuleList): Collection of `Vit_expert` modules.
        cross_attn (MP_Attention): Fuses U-Net features (Query) and ViT features (Context).

    forward argument shapes:
        x (torch.Tensor) -> Shape: (Batch, In_Channels, Height, Width).
        time_vec (torch.Tensor) -> Shape: (Batch,).
        text_emb (torch.Tensor) -> Shape: (Batch, Seq_Len, Text_Emb_Dim) -> 3D Tensor.
        Unet_router_mask (torch.Tensor) -> Shape: (Batch, Num_Experts).
        Vit_router_mask (torch.Tensor)-> Shape: (Batch, Num_Experts).
        zeta (float): Exploration noise magnitude for the Router logits.
    """
    def __init__(self,
                 IN_in_channels       : int,  # in_channels == Out_channels in VIT experts and Unet experts
                 IN_img_resolution    : int, # seq_ln  = img_res ** 2
                 internal_channels    : int,
                 time_emb_dim         : int,
                 text_emb_dim         : int,
                 num_experts          : int,
                 top_k                : int,
                 Fourier_bandwidth    : float,
                 VIT_num_blocks       : int,
                 VIT_patch_sizes      : List[int], #patches are square small pieces of image
                 VIT_num_groups       : int,
                 VIT_num_heads        : int,
                 VIT_emb_size         : int,  # internal VIT embedding size
                 Unet_num_blocks      : int,
                 Unet_channel_mult    : list[int], #shape of (num_experts , levels)
                 Unet_kernel_sizes    : List[Tuple[int, int]],
                 Unet_model_channels  : Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance   : Optional[float] = 0.5,
                 Unet_concat_balance  : Optional[float] = 0.5,
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
        #assert IN_img_resolution ** 2 == VIT_seq_ln
        self.internal_channels = internal_channels
        self.input_proj = util.MP_Conv(
            in_channels=IN_in_channels,
            out_channels=self.internal_channels,
            kernel=(3, 3)
        )
        self.Fourier_emb = util.MP_Fourier(num_channels=time_emb_dim // 2,
                                           bandwidth=Fourier_bandwidth)

        self.out_fourier1 = util.MP_Conv(in_channels= time_emb_dim // 2,
                                         out_channels= time_emb_dim * 2,
                                         kernel=())

        self.out_fourier2 = util.MP_Conv(in_channels=time_emb_dim * 2,
                                         out_channels=time_emb_dim,
                                         kernel=())

        self.Unet_router = m.Router(time_dim= time_emb_dim,
                                    top_k= top_k,
                                    num_experts= num_experts)

        self.vit_router = m.Router(time_dim=time_emb_dim,
                                    top_k=top_k,
                                    num_experts= num_experts)

        self.alpha_txt = nn.Parameter(torch.tensor(0.0))
        self.Unet_experts= nn.ModuleList()
        for i in range(num_experts):
            self.Unet_experts.append(m.Unet_expert(img_resolution= IN_img_resolution,
                                                   img_channels= self.internal_channels,
                                                   time_emb_dim= time_emb_dim,
                                                   text_emb_dim= text_emb_dim,
                                                   num_blocks=Unet_num_blocks,
                                                   channel_mult= Unet_channel_mult,
                                                   kernel_size=Unet_kernel_sizes[i],
                                                   label_balance=Unet_label_balance,
                                                   concat_balance=Unet_concat_balance,
                                                   model_channels=Unet_model_channels,
                                                   channel_mult_emb=Unet_channel_mult_emb
                                                   ))
        self.VIT_experts = nn.ModuleList()
        for i in range(num_experts):
            self.VIT_experts.append(m.Vit_expert(num_heads= VIT_num_heads,
                                                  num_groups=VIT_num_groups,
                                                  in_channels=self.internal_channels,
                                                  seq_ln= math.ceil(IN_img_resolution / VIT_patch_sizes[i]) ** 2,
                                                  emb_dim=VIT_emb_size,
                                                  num_blocks= VIT_num_blocks,
                                                  patch_size= VIT_patch_sizes[i],
                                                  text_dim= text_emb_dim,
                                                  time_dim= time_emb_dim
                                                  ))

        self.cross_attn = util.MP_Attention(num_heads= VIT_num_heads,
                                                  emb_dim= self.internal_channels,
                                                  seq_ln= IN_img_resolution ** 2,
                                                  context_dim= self.internal_channels,
                                                  attn_balance= 0.5,
                                                  is_cross_attn= True)

        """self.cross_attn_text = util.MP_Attention(
            num_heads=VIT_num_heads,
            emb_dim=self.internal_channels,
            seq_ln=IN_img_resolution ** 2,
            context_dim=text_emb_dim,
            attn_balance=0.2,
            is_cross_attn=True
        )"""

        self.gate1 = util.MP_Conv(in_channels= self.internal_channels * 2,
                                  out_channels=self.internal_channels ,
                                  kernel=(1,1)
                                  )
        self.gate2 = util.MP_Conv(in_channels= self.internal_channels ,
                                  out_channels = 2,
                                  kernel=(1,1)
                                  )
        self.output_proj_vit= util.MP_Conv(
            in_channels=self.internal_channels,
            out_channels=IN_in_channels,
            kernel=(3, 3)
        )
        self.output_proj_unet = util.MP_Conv(
            in_channels=self.internal_channels,
            out_channels=IN_in_channels,
            kernel=(3, 3)
        )
        self.output_proj_fusion = util.MP_Conv(
            in_channels=self.internal_channels,
            out_channels=IN_in_channels,
            kernel=(3, 3)
        )

    def forward(self,
                x               : torch.Tensor, #shape -> (batch,in_channels,height, width)
                time_vec        : torch.Tensor, #sigma in literature shape ->(batch,)
                text_emb        : torch.Tensor, #shape -> (batch,text_seq_ln,text_emb_dim)
                Unet_router_mask: torch.Tensor, #shape-> (batch,num_experts)
                Vit_router_mask : torch.Tensor, #shape-> (batch,num_experts)
                zeta            : float,
                transition_point:float,
                softness :float
                )-> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Forward pass of the Hybrid MoE.

        Args:
            :param x : Input noisy image/latent. Shape: (Batch, In_Channels, Height, Width).
            :param softness: used for the model's scaling criteria
            :param transition_point: used for the model's scaling criteria

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
        """
        B,C,H,W = x.shape
        time_embed = self.Fourier_emb(x = time_vec)
        time_embed = self.out_fourier1(x = time_embed)
        time_embed = self.out_fourier2(x = util.mp_silu(x= time_embed))
        x_feats = self.input_proj(x)
        log_sigma = time_vec * 4
        vit_weight = torch.sigmoid((log_sigma - transition_point) / softness)
        vit_weight = vit_weight.view(-1, 1, 1, 1)
        scaling_vit = vit_weight + 1e-6
        scaling_unet = 1.0 - vit_weight + 1e-6
        total_scale = scaling_unet + scaling_vit
        scaling_vit = scaling_vit / total_scale
        scaling_unet = scaling_unet / total_scale
        scaling_factors = torch.cat([scaling_vit, scaling_unet], dim=1).view(-1, 2)
        in_unet_router =  x_feats
        in_vit_router =  x_feats
        out_vit_router,Vit_gate_probs,vit_raw = self.vit_router(
                                         time_emb = time_embed,
                                         zeta= zeta,
                                         mask = Vit_router_mask)

        out_unet_router,Unet_gate_probs,Unet_raw = self.Unet_router(
                                           time_emb = time_embed,
                                           zeta = zeta,
                                           mask = Unet_router_mask)

        out_Unet_expert = router_to_unet_experts(x = in_unet_router,
                                            experts=self.Unet_experts,
                                            time_emb= time_embed,
                                            text_emb= text_emb,
                                            out_router=out_unet_router
                                            )

        out_Vit_expert = router_to_unet_experts(x = in_vit_router,
                                           experts=self.VIT_experts,
                                           time_emb= time_embed,
                                           text_emb= text_emb,
                                           out_router=out_vit_router
                                            )
        out_Vit_expert = boost(out_Vit_expert, 5.0)
        out_Unet_flat = out_Unet_expert.flatten(2).transpose(1, 2)
        out_Vit_flat = out_Vit_expert.flatten(2).transpose(1, 2)

        out_final_attn = self.cross_attn(query = out_Unet_flat,
                                          context = out_Vit_flat,
                                          gain_s = 1.0,
                                          gain_t = 1.0,
                                         )
        out_final_attn = boost(out_final_attn, 5.0)
        """final_feats = self.cross_attn_text(
            query=out_final_attn,
            context=text_emb,
            gain_s=1.0, gain_t=1.0
        )
        final_feats = util.mp_sum(out_final_attn, final_feats, t=self.alpha_txt.clamp(min=0.1, max=0.5).item())
        """

        out_final_attn_img = out_final_attn.transpose(1, 2).view(B, self.internal_channels, H, W)
        in_gate = util.mp_cat(out_Unet_expert,out_final_attn_img,dim = 1)
        out_gate = self.gate1(in_gate)
        out_gate = self.gate2(util.mp_silu(out_gate))
        out_gate = F.softmax(out_gate,dim = 1)
        Wx = out_gate[:, 0:1]
        Wa = out_gate[:, 1:2]
        Wx_suppressed = Wx* 0.3

        vit_out = self.output_proj_vit(out_Vit_expert)
        vit_out = boost(vit_out, 5.0)
        unet_out = self.output_proj_unet(out_Unet_expert)
        unet_out = boost(unet_out, 5.0)
        fusion_out = self.output_proj_fusion(out_final_attn_img)
        fusion_out = boost(fusion_out, 5.0)

        w_f = 0.4
        w_v = scaling_vit * (1 - w_f)
        w_u = scaling_unet * (1 - w_f)
        out_raw = (w_v * vit_out) + (w_u * unet_out) + (w_f * fusion_out)
        mag = torch.sqrt(w_v ** 2 + w_u ** 2 + w_f ** 2 + 1e-8)
        out = out_raw / mag.detach()
        return out, Unet_gate_probs, Unet_raw, Vit_gate_probs, vit_raw, scaling_factors, out_gate


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
                 IN_in_channels: int,  # in_channels == Out_channels in VIT experts and Unet experts
                 IN_img_resolution: int,
                 internal_channels:int,# seq_ln  = img_res ** 2
                 time_emb_dim: int,
                 text_emb_dim: int,
                 num_experts: int,
                 top_k: int,
                 Fourier_bandwidth: float,
                 VIT_num_blocks: int,
                 VIT_patch_sizes: List[int],  # patches are square small pieces of image
                 VIT_num_groups: int,
                 VIT_num_heads: int,
                 VIT_emb_size: int,  # internal VIT embedding size
                 Unet_num_blocks: int,
                 Unet_channel_mult: list[int],  # shape of (num_experts , levels)
                 Unet_kernel_sizes: List[Tuple[int, int]],
                 Unet_model_channels: Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance: Optional[float] = 0.5,
                 Unet_concat_balance: Optional[float] = 0.5,
                 sigma_data: Optional[float] = 0.5, #expected standard deviation of the data
                 log_var_channels: Optional[int] = 128
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
        self.num_experts = num_experts
        self.log_var_fourier = util.MP_Fourier(num_channels= self.log_var_channels)
        self.log_var_linear = util.MP_Conv(in_channels= self.log_var_channels, out_channels=1, kernel=())
        self.net = HDMOEM(IN_in_channels = IN_in_channels,
                          IN_img_resolution = IN_img_resolution,
                          internal_channels=internal_channels,# seq_ln  = img_res ** 2
                          time_emb_dim= time_emb_dim,
                          text_emb_dim =text_emb_dim,
                          num_experts =num_experts,
                          top_k = top_k,
                          Fourier_bandwidth =Fourier_bandwidth,
                          VIT_num_blocks = VIT_num_blocks,
                          VIT_patch_sizes=VIT_patch_sizes,  # patches are square small pieces of image
                          VIT_num_groups=VIT_num_groups,
                          VIT_num_heads=VIT_num_heads,
                          VIT_emb_size =VIT_emb_size,  # internal VIT embedding size
                          Unet_num_blocks= Unet_num_blocks,
                          Unet_channel_mult=Unet_channel_mult,  # shape of (num_experts , levels)
                          Unet_kernel_sizes =Unet_kernel_sizes,
                          Unet_model_channels =Unet_model_channels,
                          Unet_channel_mult_emb = Unet_channel_mult_emb,
                          Unet_label_balance= Unet_label_balance,
                          Unet_concat_balance = Unet_concat_balance
                          )

    def forward(self,
                x: torch.Tensor,  # shape -> (batch,in_channels,height, width)
                sigma: torch.Tensor,  # sigma in literature shape ->(batch,1,1,1)
                text_emb: torch.Tensor,  # shape -> (batch,text_seq_ln,text_emb_dim)
                Unet_router_mask: torch.Tensor,  # shape-> (batch,num_experts)
                Vit_router_mask: torch.Tensor,  # shape-> (batch,num_experts)
                zeta: float,
                transition_point:float,
                softness :float,
                return_log_var:bool = False
                ) -> dict[str,torch.Tensor]:
        """
        Forward pass with EDM Preconditioning.

        Calculates: C_NOISE, C_IN, C_OUT,C_SKIP given the standard deviation of the data and the noise conditioning vector

        Args:
            softness:
            transition_point:
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
        """
        # Preconditioning weights.
        sigma = sigma.to(torch.float32)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out =  sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        batch_size = x.shape[0]
        if c_noise.shape[0] == 1 and batch_size > 1:
            c_noise = c_noise.expand(batch_size)

        x = x * c_in
        out_net,Unet_gate_probs,Unet_raw, Vit_gate_probs,vit_raw ,scaling_factors,out_gate= self.net(x = x,
                           text_emb = text_emb,
                           time_vec = c_noise ,
                           Unet_router_mask = Unet_router_mask,
                           Vit_router_mask = Vit_router_mask,
                           zeta = zeta,
                           transition_point=transition_point ,
                           softness = softness)
        D_x = c_skip * x + c_out* out_net
        if return_log_var:
            log_var = self.log_var_linear(self.log_var_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return {"denoised":D_x,
                    "Unet_router_loss":Unet_gate_probs,
                    "Unet_raw" :Unet_raw,
                    "vit_router_loss": Vit_gate_probs,
                    "vit_raw":vit_raw,
                    "scaling_net_out": scaling_factors,
                    "out_gate": out_gate,
                    "log_var":log_var}
        else:
            return {"denoised":D_x,
                    "Unet_router_loss":Unet_gate_probs,
                    "Unet_raw" :Unet_raw,
                    "vit_router_loss": Vit_gate_probs,
                    "vit_raw":vit_raw,
                    "scaling_net_out": scaling_factors,
                    "out_gate": out_gate,
                    "log_var":None}