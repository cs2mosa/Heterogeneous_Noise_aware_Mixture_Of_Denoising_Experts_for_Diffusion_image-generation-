from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_internals as m

class Scaling_router(nn.Module):
    """
    A Residual-Scaling Router.

    Robustness Principle:
    The output is centered around 1.0. The network predicts a deviation factor.
    This ensures that at initialization (when weights are small/zero),
    the experts are FULLY ACTIVE, forcing the backbone to learn immediately.

    Formula: Scale = 1.0 + tanh(Net(x)) * range_factor
    Range: [1 - range, 1 + range]
    """

    def __init__(self,
                 emb_dim: int = 3,
                 num_experts: int = 2,
                 scale_range: float = 0.9  # Allows scaling from 0.0 to 2.0
                 ):
        super().__init__()
        self.scale_range = scale_range

        # 1. Deeper, Narrower MLP for stability
        # Using LayerNorm (via normalize) is crucial to keep logits bounded
        self.net = nn.Sequential(
            m.MP_Conv(in_channels=emb_dim, out_channels=emb_dim * 2, kernel=()),
            nn.SiLU(),  # Use standard SiLU for router internals
            m.MP_Conv(in_channels=emb_dim * 2, out_channels=emb_dim, kernel=()),
            nn.SiLU(),
            m.MP_Conv(in_channels=emb_dim, out_channels=num_experts, kernel=())
        )

        # 2. Zero Initialization for the last layer
        # This guarantees that at Step 0, Net(x) = 0
        nn.init.constant_(self.net[-1].weights, 0.0)

    def forward(self,
                x: torch.Tensor,
                zeta: Optional[float] = 0.0
                ) -> torch.Tensor:

        if x.ndim == 3:
            x = x.squeeze(1)

        # 1. Get Logits (Deviation from baseline)
        deviation_logits = self.net(x)

        # 2. Add exploration noise (Zeta) to logits, not output
        if self.training and zeta > 0:
            deviation_logits = deviation_logits + torch.randn_like(deviation_logits) * zeta

        # 3. Compute Scaling Factor
        # Tanh maps to [-1, 1].
        # Output maps to [1.0 - range, 1.0 + range]
        # At Init (logits=0): Output = 1.0 + 0 = 1.0 (Unit Gain)
        scaling_output = 1.0 + torch.tanh(deviation_logits) * self.scale_range

        # 4. Safety Clamp (Prevent negative scaling if range > 1.0)
        return scaling_output.clamp(min=0.01)

class Router(nn.Module):
    """
    A Pure Noise-Dependent Trajectory Router.

    This module maps the noise conditioning (time_emb) directly to expert selection.
    It ignores the image content 'x' to ensure routing is a pure function of
    the ODE trajectory stage.
    """

    def __init__(self,
                 time_dim: Optional[int] = 256,  # Input: dimension of noise vector
                 top_k: Optional[int] = 1,
                 num_experts: Optional[int] = 4,
                 dropout: Optional[float] = 0.2
                 ):
        super().__init__()
        self.mlp_route = nn.Sequential(
            m.MP_Conv(in_channels=time_dim, out_channels=time_dim * 2, kernel=()),
            nn.SiLU(),
            m.MP_Conv(in_channels=time_dim * 2, out_channels=time_dim, kernel=()),
            nn.SiLU(),
            nn.Dropout(dropout),
            m.MP_Conv(in_channels=time_dim, out_channels=num_experts, kernel=()),
            nn.SiLU()
        )
        self.k = top_k

    def forward(self,
                time_emb: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                zeta: Optional[float] = 1e-2
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            time_emb: [B, time_dim] or [B, 1, time_dim] noise embedding
        Output:
            Standard tuple to maintain interface with HDMOEM.
        """
        if time_emb.ndim == 3:
            time_emb = time_emb.squeeze(1)

        logits = self.mlp_route(time_emb)
        if self.training and zeta > 0:
            logits = logits + torch.randn_like(logits) * zeta

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        gate_probs = F.softmax(logits, dim=-1)
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=-1)
        if self.k == 1:
            gating_weights = torch.ones_like(topk_vals)
        else:
            gating_weights = F.softmax(topk_vals, dim=-1)

        sparse_gate_weights = torch.zeros_like(logits).scatter(-1, topk_indices, gating_weights)
        return sparse_gate_weights, gate_probs, logits


class Unet_block(nn.Module):
    """
    the building block of the Unet experts in the Unet pathway , according to Nvidia's paper
        Note: we made the kernel size a variable input to enforce architectural differences between each expert in the Unet path
    """
    def __init__(self,
                 in_channels     : int ,
                 out_channels    : int ,
                 kernel          : tuple,
                 emb_size        : int ,
                 resample        : Optional[str] = 'keep',
                 Type            : Optional[str] = 'enc',
                 residual_balance: Optional[float] = 0.5,
                 Dropout         : Optional[float] = 0.0,
                 emb_gain        : Optional[float] = 1.0,
                 conv_gain       : Optional[float] = 1.0
                 ):
        """
        Args:
            :param in_channels: number of input channels
            :param out_channels: number of desired output channels
            :param kernel: kernel size -> this is variable for every expert ... the only difference between this block and Nvidia's block
            :param emb_size: embedding dimensions
            :param resample:
                    keep: returns the same input (identity function)
                    up: upsamples the input by (2 x 2)
                    down: downsamples the input by (2 x 2)
            :param Type: encoder or decoder -> enc | dec
                Note: choosing 'enc' implies choosing 'down' in type ... to be consistent with Nvidia's diagram
            :param residual_balance: balance between the main branch and the residual branch of the block
            :param Dropout: usually 0.2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_size = emb_size
        self.residual_balance= residual_balance
        self.type = Type
        self.resample = resample
        self.kernel = kernel
        self.dropout =  Dropout
        self.emb_gain = emb_gain
        self.conv_gain1 = conv_gain
        self.conv_gain2 = conv_gain
        self.conv_skip = m.MP_Conv(in_channels = in_channels, out_channels=out_channels,kernel=(1, 1)) if in_channels != out_channels else None
        self.emb_layer = m.MP_Conv(in_channels = emb_size, out_channels = out_channels,kernel=())
        self.conv_res1 = m.MP_Conv(in_channels = out_channels if self.type == 'enc' else in_channels, out_channels= out_channels, kernel=self.kernel)
        self.conv_res2 = m.MP_Conv(in_channels = out_channels, out_channels= out_channels, kernel=self.kernel)

    def forward(self,
                x        :torch.Tensor,
                embedding:torch.Tensor
                ) -> torch.Tensor:
        """
        the forward method of the Unet block
        Args:
            :param x: the input image tensor of dimensions -> (batch , in_channels, height, width)
            :param embedding: embedding tensor for text and noise merged before the entry , dimensions -> (batch, embedding_size)
                Note: concatenation with the skip connection must be done in the Full Unet class
            :return: processed output tensor of dimensions -> (batch , out_channels, new_height, new_width)
        """
        embedding = 1 + self.emb_layer(embedding,gain = self.emb_gain)
        #up for decoders : down for encoders
        x = m.resample(x, mode=self.resample)
        if self.type == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = m.normalize(x,dim = [1])

        main_branch = self.conv_res1(m.mp_silu(x),gain = self.conv_gain1)
        #embedding dimensions (batch,embedding_size) -> (batch, intermediate_channels,1,1) for broadcasting
        main_branch = main_branch * embedding.unsqueeze(2).unsqueeze(3).to(x.dtype)
        main_branch = m.normalize(main_branch)#ensure magnitude preservation
        main_branch = m.mp_silu(main_branch)
        #dropout only in training mode
        """if self.training and self.dropout != 0:
            main_branch = F.dropout(main_branch, p= self.dropout)"""

        main_branch = self.conv_res2(main_branch,gain = self.conv_gain2)
        #conv_skip for decoders in the residual branch
        if self.type == 'dec' and self.conv_skip is not None :
            x = self.conv_skip(x)

        return m.mp_sum(x,main_branch,t = self.residual_balance)

class Unet_expert(nn.Module):
    """
    A Magnitude-Preserving U-Net Expert for Diffusion Models.

    This architecture implements a symmetric Encoder-Decoder network with skip connections,
    specifically designed for diffusion noise prediction. It features Magnitude Preserving (MP)
    layers to maintain signal variance stability and flexible conditioning mechanisms.

    Key Architectural Features:
    1. **Input Augmentation**: Appends a constant channel of ones to the input to aid in
       bias learning/signal reference.
    2. **Conditioning**: Projects and mixes Time and Text embeddings using a learnable
       balance factor (`label_balance`).
    3. **Skip Connections**: Utilizes magnitude-preserving concatenation (`mp_cat`) to merge
       encoder features with decoder features, controlled by `concat_balance`.
    4. **Hierarchical Resolution**: Downsamples and Upsamples features according to
       `channel_mult` factors.

    Attributes:
        map_noise (MP_Conv): Projector for time embeddings.
        map_text (MP_Conv, optional): Projector for text embeddings.
        encoders (nn.ModuleDict): Dictionary of encoder blocks (Conv, Downsample, ResBlocks).
        decoders (nn.ModuleDict): Dictionary of decoder blocks (Upsample, ResBlocks).
        out_conv (MP_Conv): Final projection layer to image space.
        #out_gain (nn.Parameter): Learnable scalar gain for the final output.
    """
    def __init__(self,
                 img_resolution  : int,
                 img_channels    : int,
                 time_emb_dim    : int,
                 text_emb_dim    : int,
                 channel_mult    : list,
                 model_channels  : Optional[int] = 192,
                 channel_mult_emb: Optional[int] = None,
                 num_blocks      : Optional[int] = 3,
                 kernel_size     : Optional[tuple] = (3, 3),
                 label_balance   : Optional[float] = 0.5,
                 concat_balance  : Optional[float] = 0.5,
                 ):
        """
        Initializes the U-Net Expert.

        Args:
            img_resolution (int): Spatial resolution of the input image (Height/Width).
            img_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            time_emb_dim (int): Dimension of the input time/noise embedding vector.
            text_emb_dim (int): Dimension of the input text embedding vector.
            channel_mult (list[int]): List of multipliers for `model_channels` at each resolution level.
                                      Example: [1, 2, 2, 4] defines the channel depth at each stage.
            model_channels (int, optional): Base number of feature channels. Defaults to 192.
            channel_mult_emb (int, optional): Multiplier for the internal embedding dimension relative
                                              to `model_channels`. If None, uses max channel size.
            num_blocks (int, optional): Number of residual blocks per resolution level. Defaults to 3.
            kernel_size (tuple, optional): Spatial convolution kernel size. Defaults to (3, 3).
            label_balance (float, optional): Interpolation weight 't' for mixing Time and Text
                                             embeddings via `mp_sum`. Defaults to 0.5.
            concat_balance (float, optional): Interpolation weight 't' for merging skip connections
                                             via `mp_cat`. Defaults to 0.5.
        """
        super().__init__()
        self.block_channels = [model_channels * i for i in channel_mult]
        self.emb_size = model_channels * channel_mult_emb if channel_mult_emb is not None else max(self.block_channels)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        #self.out_gain = nn.Parameter(torch.ones([]) * 0.02)
        self.map_noise = m.MP_Conv(in_channels=time_emb_dim, out_channels=self.emb_size, kernel=())
        self.map_text = m.MP_Conv(in_channels=text_emb_dim, out_channels=self.emb_size, kernel=()) if text_emb_dim > 0 else None
        self.encoders = nn.ModuleDict()
        self.out_channels = img_channels + 1
        # _______________________________________________________________________________________________#
        for level , channel in enumerate(self.block_channels):
            res = img_resolution >> level
            if level == 0:
                in_channel = self.out_channels
                self.out_channels = channel
                self.encoders[f'{res}x{res}_conv'] = m.MP_Conv(in_channels= in_channel,
                                                               out_channels= self.out_channels,
                                                               kernel= kernel_size)
            else:
                self.encoders[f'{res}x{res}_down'] = Unet_block(in_channels= self.out_channels,
                                                                out_channels=self.out_channels,
                                                                kernel=kernel_size,
                                                                Type='enc',
                                                                resample= 'down',
                                                                emb_size= self.emb_size
                                                                )
            for i in range(num_blocks):
                in_channel = self.out_channels
                self.out_channels = channel
                self.encoders[f'{res}x{res}_block{i}'] = Unet_block(in_channels=in_channel,
                                                                    out_channels= self.out_channels,
                                                                    emb_size= self.emb_size,
                                                                    Type= 'enc',
                                                                    resample='keep',
                                                                    kernel= kernel_size)
        #_______________________________________________________________________________________________#
        self.decoders = nn.ModuleDict()
        skips = [block.out_channels for name, block in self.encoders.items()]
        for level, channel in reversed(list(enumerate(self.block_channels))):
            res = img_resolution >> level
            if level == len(self.block_channels) - 1:
                self.decoders[f'{res}x{res}_in0'] = Unet_block(in_channels=self.out_channels,
                                                                    out_channels= self.out_channels,
                                                                    emb_size= self.emb_size,
                                                                    Type= 'dec',
                                                                    resample='keep',
                                                                    kernel= kernel_size)

                self.decoders[f'{res}x{res}_in1'] = Unet_block(in_channels=self.out_channels,
                                                                out_channels=self.out_channels,
                                                                emb_size=self.emb_size,
                                                                Type='dec',
                                                                resample='keep',
                                                                kernel=kernel_size)
            else:
                self.decoders[f'{res}x{res}_up'] = Unet_block(in_channels=self.out_channels,
                                                                out_channels=self.out_channels,
                                                                emb_size=self.emb_size,
                                                                Type='dec',
                                                                resample='up',
                                                                kernel=kernel_size)

            for i in range (num_blocks + 1):
                in_channel = self.out_channels + skips.pop()
                self.out_channels = channel
                self.decoders[f'{res}x{res}_block{i}'] = Unet_block(in_channels=in_channel,
                                                               out_channels=self.out_channels,
                                                               emb_size=self.emb_size,
                                                               Type='dec',
                                                               resample='keep',
                                                               kernel=kernel_size)
        # _______________________________________________________________________________________________#
        self.out_conv = m.MP_Conv(in_channels=self.out_channels, out_channels=img_channels, kernel=kernel_size)

    def forward(self,
                x        : torch.Tensor ,
                time_emb : torch.Tensor ,
                text_emb :torch.Tensor
                )-> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Noisy input image tensor. Shape (Batch, img_channels, H, W).
            time_emb (torch.Tensor): Time/Noise embedding vector. Shape (Batch, time_emb_dim).
            text_emb (torch.Tensor): Text embedding vector. Shape (Batch, seq_ln ,text_emb_dim). or (Batch, text_emb_dim)

        Returns:
            torch.Tensor: Predicted output (e.g., noise or denoised image).
                          Shape (Batch, img_channels, H, W).
        """
        emb = self.map_noise(time_emb)
        if self.map_text is not None and text_emb is not None:
            # txt_emb Shape transforms to -> (batch,text_emb_dim) if ndims = 3
            if text_emb.ndim == 3:
                text_emb = text_emb.mean(dim=1)

            txt = self.map_text(text_emb)
            emb = m.mp_sum(emb,txt,t = self.label_balance)

        emb = m.mp_silu(emb)
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.encoders.items():
            if 'conv' in name:
                x = block(x)
            else:
                x = block(x, embedding=emb)
            skips.append(x)

        for name, block in self.decoders.items():
            if 'block' in name:
                skip_x = skips.pop()
                x = m.mp_cat(x, skip_x, t=self.concat_balance)

            x = block(x, embedding=emb)

        x = self.out_conv(x, gain=1.0)
        return x

class Vit_block(nn.Module):
    """
    A Diffusion Vision Transformer (DiffiT) Block adapted for High-Noise/Latent processing.

    This block implements a hybrid architecture that combines a local feature projection
    (using GroupNorm and Linear layers) with a global Time-Dependent Multi-Head Self-Attention
    (TMSA) mechanism. It is designed to operate within a Magnitude-Preserving (MP) framework,
    ensuring variance stability across deep networks.

    Architectural Changes from original DiffiT:
        - Replaces the standard 3x3 Convolutional wrapper with a 1x1 Linear Projection.
          This increases expressive power for high-noise/latent tokens where spatial
          adjacency is less correlated than in pixel space.

    Attributes:
        linear1 (MP_Conv): Input projection layer (Input Dim -> Emb Dim).
        TMSA (MP_Attention): Time-Dependent Self-Attention core.
        linear2, linear3 (MP_Conv): Feed-forward MLP layers (expansion factor of 4).
        skip_proj (MP_Conv, optional): Linear projection for the residual connection
                                       if input_dim != emb_dim.
    """
    def __init__(self,
                 num_heads    : int ,
                 num_groups   : int,
                 num_channels : int ,
                 seq_ln       : int ,
                 emb_dim      : int ,
                 resample     : Optional[str] = 'keep',
                 time_dim     : Optional[int] = 0 ,
                 res_balance  : Optional[float] = 0.5 ,
                 attn_balance : Optional[float] = 0.5 ,
                 gain_s       : Optional[float] = 1.0,
                 gain_t       : Optional[float] = 1.0,
                 ):
        """
       Initializes the ViT Block.

       Args:
           num_heads (int): Number of attention heads.
           num_groups (int): Number of groups for GroupNormalization.
           num_channels (int): Input feature dimension (C_in).
           seq_ln (int): Sequence length (Total number of tokens, H*W).
           emb_dim (int): Internal embedding dimension for the Transformer (D).
           time_dim (int, optional): Dimension of the time embedding vector. Defaults to 0.
           res_balance (float, optional): Residual balance factor 't' for mp_sum. Defaults to 0.5.
           attn_balance (float, optional): Attention balance factor. Defaults to 0.5.
           gain_s (float, optional): Magnitude scaling factor for spatial signals. Defaults to 1.0.
           gain_t (float, optional): Magnitude scaling factor for temporal signals. Defaults to 1.0.
       """
        super().__init__()
        self.res_balance = res_balance
        self.gain_s = gain_s
        self.gain_t = gain_t
        self.emb_dim = emb_dim
        self.resample = resample
        #self.GN = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        #instead of conv 3X3 we use linear layer before the block
        self.skip_proj = m.MP_Conv(num_channels, emb_dim, kernel=()) if num_channels != emb_dim else None
        self.linear1 = m.MP_Conv(num_channels,emb_dim,kernel = ())
        #self.norm1 = nn.LayerNorm(emb_dim)
        #self.norm2 = nn.LayerNorm(emb_dim)
        self.TMSA = m.MP_Attention(num_heads= num_heads, emb_dim= emb_dim ,seq_ln= seq_ln ,time_dim= time_dim, attn_balance= attn_balance)
        #MLP used after the TMSA attention
        self.linear2 = m.MP_Conv(emb_dim,emb_dim*4,kernel = ())
        self.linear3 = m.MP_Conv(emb_dim*4,emb_dim,kernel = ())

    def forward(self,
                x             : torch.Tensor,
                time_embedding: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass of the block.

        The flow consists of:
        1. Input Norm & Projection (GN -> SiLU -> Linear).
        2. Time-Dependent Attention (TMSA) with residual connection.
        3. Pointwise MLP with residual connection.
        4. Final Skip Connection (Projecting original input if necessary).

        Args:
            x (torch.Tensor): Input sequence tensor.
                              Shape: (Batch, Seq_Len, Input_Channels)
            time_embedding (torch.Tensor, optional): Time step embedding vector added with the label embedding vector.
                                                     Shape: (Batch, Time_Dim) or (Batch, 1, Time_Dim).

        Returns:
            torch.Tensor: Processed tensor.
                          Shape: (Batch, Seq_Len, Emb_Dim)
        """
        x = m.resample(x, mode=self.resample)
        batch_size, seq_ln, input_channels = x.shape
        res_main = x
        #linear projection instead of conv 3X3 in the paper DIFFIT
        x = x.transpose(1, 2)
        x = m.mp_silu(m.normalize(x=x,dim=[1]))
        x = x.transpose(1, 2)
        x = x.reshape(batch_size * seq_ln, input_channels)
        x = self.linear1(x, gain=self.gain_s)

        #core tmsa block
        res_attn = x
        y = m.normalize(x)
        y = y.reshape(batch_size, seq_ln, self.emb_dim)

        if time_embedding is not None and time_embedding.ndim == 2:
            time_embedding = time_embedding[:, None, :]  # (B, 1, C)

        y = self.TMSA(y, time_embedding=time_embedding, gain_s=self.gain_s, gain_t=self.gain_t)
        y = y.reshape(batch_size * seq_ln, self.emb_dim)
        y = m.mp_sum(y, res_attn, t=self.res_balance)
        x = m.normalize(y)
        x = m.mp_silu(self.linear2(x, gain=self.gain_s))
        x = self.linear3(x, gain=self.gain_s)
        x = m.mp_sum(x, y, t=self.res_balance)

        # Reshape back to 3D
        x = x.reshape(batch_size, seq_ln, self.emb_dim)

        if self.skip_proj is not None:
            # project original residual to emb_dim (MP_Conv in linear mode)
            res_proj = res_main.reshape(batch_size * seq_ln, input_channels)
            res_proj = self.skip_proj(res_proj, gain=self.gain_s)
            res_proj = res_proj.reshape(batch_size, seq_ln, self.emb_dim)
            return m.mp_sum(res_proj, x, t=self.res_balance)
        else:
            # Dimensions match. Add Skip Connection.
            return m.mp_sum(res_main, x, t=self.res_balance)

class Vit_expert(nn.Module):
    """
    A Diffusion Vision Transformer (ViT) Expert.

    This module implements an isotropic Transformer backbone designed for generative diffusion tasks.
    It handles image tokenization via strided convolution, processes sequences using DiffiT-style blocks,
    and reconstructs the image/latent space using PixelShuffle. It supports multi-modal conditioning
    (Time + Text) via magnitude-preserving summation.

    Key Features:
    1. **Patchify/Unpatchify**: Converts images to sequences using Conv2d and reconstructs them
       using Linear Projection + PixelShuffle.
    2. **Isotropic Backbone**: Maintains a constant embedding dimension throughout the depth of the network.
    3. **Conditioning Mixing**: Linearly interpolates between Time and Text embeddings based on
       `emb_balance` to guide the denoising process.

    Attributes:
        patch (nn.Conv2d): Tokenizer layer (Image -> Patches).
        map_txt (MP_Conv, optional): Projector to align text embedding dimensions with time embeddings.
        pos_emb (nn.Parameter): Learnable absolute positional embeddings added to the sequence.
        diffit (nn.ModuleList): Stack of `Vit_block` transformer layers.
        unpatch_proj (MP_Conv): Linear projection to expand channels for PixelShuffle.
        unpatch (nn.PixelShuffle): Reconstructs spatial resolution from channel depth.
    """
    def __init__(self,
                 num_heads   : int,
                 num_groups  : int,
                 in_channels : int,
                 seq_ln      : int,
                 emb_dim     : int,
                 num_blocks  : int,
                 patch_size  : int,
                 time_dim    : Optional[int] = 0,
                 text_dim    : Optional[int] = 0,
                 res_balance : Optional[float] = 0.5 ,
                 attn_balance: Optional[float] = 0.5 ,
                 emb_balance : Optional[float] = 0.5,
                 gain_s      : Optional[float] = 1.0,
                 gain_t      : Optional[float] = 1.0,
                 ):
        """
        Initializes the ViT Expert.

        Args:
            num_heads (int): Number of attention heads in the transformer blocks.
            num_groups (int): Number of groups for GroupNormalization.
            in_channels (int): Number of channels in the input image/latent.
            seq_ln (int): Expected sequence length (Total number of patches = H/P * W/P).
            emb_dim (int): Internal hidden dimension of the transformer.
            num_blocks (int): Depth of the network (number of layers).
            patch_size (int): Spatial size of the patches used for tokenization.
            time_dim (int, optional): Dimension of the input time embedding. Defaults to 0.
            text_dim (int, optional): Dimension of the input text embedding. Defaults to 0.
            res_balance (float, optional): Residual connection weight 't' for internal blocks. Defaults to 0.5.
            attn_balance (float, optional): Attention output weight 't' for internal blocks. Defaults to 0.5.
            emb_balance (float, optional): Interpolation weight 't' for mixing Time vs Text.
                                           Final Emb = Time*(1-t) + Text*(t). Defaults to 0.5.
            gain_s (float, optional): Magnitude gain for spatial signal path. Defaults to 1.0.
            gain_t (float, optional): Magnitude gain for temporal/conditioning signal path. Defaults to 1.0.
        """
        super().__init__()
        self.seq_ln = seq_ln
        self.emb_balance = emb_balance
        self.emb_dim = emb_dim
        self.patch = m.MP_Conv(in_channels=in_channels + 1,out_channels=emb_dim,kernel=(patch_size,patch_size),stride=patch_size)
        self.map_txt = m.MP_Conv(in_channels=text_dim,out_channels=time_dim,kernel=()) if text_dim != time_dim and text_dim != 0 else None
        self.pos_emb = nn.Parameter(torch.randn(1,seq_ln,emb_dim)*0.02)
        self.diffit = nn.ModuleList()
        for i in range(num_blocks):
            self.diffit.append(Vit_block(num_heads= num_heads,
                                       num_groups=num_groups,
                                       num_channels=emb_dim,
                                       seq_ln=seq_ln,emb_dim=emb_dim,
                                       resample='keep',
                                       time_dim= time_dim,
                                       res_balance=res_balance,
                                       attn_balance=attn_balance,
                                       gain_s=gain_s,
                                       gain_t=gain_t))

        #self.norm = nn.LayerNorm(emb_dim)
        self.unpatch_proj = m.MP_Conv(in_channels= emb_dim,out_channels=in_channels*(patch_size**2),kernel=())
        self.unpatch = nn.PixelShuffle(upscale_factor=patch_size)

    def forward(self,
                x       : torch.Tensor,
                time_emb: torch.Tensor = None,
                text_emb: Optional[torch.Tensor] = None
                )->torch.Tensor:
        """
        Forward pass of the ViT Expert.

        Args:
            x (torch.Tensor): Input image/latent tensor. Shape (Batch, In_Channels, H, W).
            time_emb (torch.Tensor, optional): Time embedding vector. Shape (Batch, Time_Dim).
            text_emb (torch.Tensor, optional): Text embedding vector. Shape (Batch, Text_Dim). or shape (Batch, seq_ln, Text_dim)

        Returns:
            torch.Tensor: Output tensor (e.g., predicted noise). Shape (Batch, In_Channels, H, W).
        """
        # Store original dimensions
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        batch, in_channels, orig_h, orig_w = x.shape
        patch_size = self.patch.kernel[0]

        # Calculate and apply padding
        pad_h = (patch_size - orig_h % patch_size) % patch_size
        pad_w = (patch_size - orig_w % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        x = self.patch(x)
        batch,c,h_patch,w_patch = x.shape
        assert h_patch*w_patch == self.seq_ln, f"Sequence length mismatch: Got {h_patch * w_patch}, expected {self.seq_ln}, shape: {x.shape}"
        x = x.flatten(2).transpose(1, 2)
        x = m.mp_sum(x, self.pos_emb, t=0.2)
        if text_emb is not None:
            if self.map_txt is not None:
                #shape transforms to -> (batch, text_emb_size) for the linear layer if ndims = 3
                if text_emb.ndim == 3:
                    text_emb = text_emb.mean(dim = 1)

                text_emb = self.map_txt(text_emb)

            time_emb = m.mp_sum(a=time_emb, b=text_emb, t=self.emb_balance)

        for block in self.diffit:
            x = block(x,time_embedding = time_emb)

        x = m.normalize(x)
        x = x.reshape(batch * self.seq_ln, self.emb_dim)
        x = self.unpatch_proj(x)

        c_expanded = x.shape[-1]
        x = x.reshape(batch, self.seq_ln, c_expanded)
        x = x.transpose(1, 2).view(batch, c_expanded, h_patch,w_patch)
        x = self.unpatch(x)
        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]

        return x
