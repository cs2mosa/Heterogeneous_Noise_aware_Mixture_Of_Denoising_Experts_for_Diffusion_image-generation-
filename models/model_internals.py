from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def normalize(x: torch.Tensor, dim: Optional[list[int]] = None, eps: float = 1e-4)-> torch.Tensor:
    """
        Normalizes the input tensor `x` to have unit variance (RMS Normalization).

        This technique is used to force the input to lie on a hypersphere, ensuring
        consistent signal magnitude regardless of the specific values.

        Args:
            x (torch.Tensor): Input tensor of any shape (usually [N, C, ...]).
            dim (list[int], optional): Dimensions to reduce over. If None, reduces
                                       over all dimensions except the first (batch).
                                       Defaults to None.
            eps (float, optional): A small constant for numerical stability to avoid
                                   division by zero. Defaults to 1e-4.

        Returns:
            torch.Tensor: The normalized tensor with the same shape as `x`.
        """
    if dim is None:
        dim = list(range(1, x.ndim))
    w = torch.linalg.vector_norm(x,dim = dim, keepdim = True, dtype = torch.float32)
    w = torch.add(eps,w, alpha= np.sqrt(w.numel() / x.numel()))
    return x / w.to(x.dtype)

#magnitude preserving silu as in EDM2 paper
def mp_silu(x: torch.Tensor)-> torch.Tensor:
    """
        Applies the SiLU (Swish) activation function, scaled to preserve variance.

        Standard SiLU reduces the standard deviation of the signal. Dividing by
        0.596 restores the variance to 1 (assuming the input variance was 1).
        This is specific to the EDM2 (Elucidating the Design Space of Diffusion-Based Generative Models) architecture.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor with preserved magnitude.
        """
    return F.silu(x) / 0.596

#magnitude preserving sum as in EDM2 paper
def mp_sum(a: torch.Tensor,b: torch.Tensor,t: float = 0.5)-> torch.Tensor:
    """
        Computes a weighted sum of tensors `a` and `b` that preserves variance.

        It performs linear interpolation followed by a normalization factor.
        Formula: result = ((1-t)*a + t*b) / sqrt((1-t)^2 + t^2)

        Args:
            a (torch.Tensor): First input tensor.
            b (torch.Tensor): Second input tensor.
            t (float, optional): Interpolation weight. 0.0 is all `a`, 1.0 is all `b`.
                                 Defaults to 0.5.

        Returns:
            torch.Tensor: The combined tensor with unit variance (assuming inputs were unit variance).
        """
    return a.lerp(b,t) / np.sqrt((1 - t)**2 + t**2)

#magnitude preserving concatenation as in EDM2 paper
def mp_cat(a: torch.Tensor,b:torch.Tensor,dim: int = 1,t: float = 0.5)-> torch.Tensor :
    """
        Concatenates two tensors `a` and `b` along a dimension while scaling them
        to preserve the overall magnitude/variance of the resulting feature vector.

        Instead of a simple concatenation where energy adds up, this weights `a`
        and `b` based on `t` so the resulting tensor maintains a stable magnitude statistic.

        Args:
            a (torch.Tensor): First input tensor.
            b (torch.Tensor): Second input tensor.
            dim (int, optional): Dimension along which to concatenate. Defaults to 1 (Channel dim).
            t (float, optional): Weight factor. A higher `t` gives more variance weight
                                 to `b` and less to `a`. Defaults to 0.5.

        Returns:
            torch.Tensor: The concatenated and scaled tensor.
        """
    Na = a.shape[dim]
    Nb = b.shape[dim]
    c1 = np.sqrt((Na+Nb)/((1-t)**2 + t**2))
    Wa = (c1* (1-t)) / np.sqrt(Na)
    Wb = (c1* t) / np.sqrt(Nb)
    return torch.cat([Wa*a, Wb*b],dim = dim)


def resample(x, f:Optional[list] = (1, 1), mode: Optional[str]='keep'):
    """
    Resamples a 4D tensor (Batch, Channels, Height, Width) using a separable filter.

    Args:
        x: Input tensor.
        f: Filter kernel (default bilinear [1,1]). Must be 1D and even length.
        mode: 'keep', 'down', or 'up'.
            keep: returns the input as it is.
            up: uses conv_transpose to upsample the input
            down: uses conv_ to downsample the input
    """
    if mode == 'keep':
        return x

    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32)

    assert f.ndim == 1 and f.shape[0] % 2 == 0
    pad = (f.shape[0] - 1) // 2
    f = f / f.sum()
    f = torch.outer(f, f)
    f = f.unsqueeze(0).unsqueeze(0)
    f = f.to(device=x.device, dtype=x.dtype)
    c = x.shape[1]
    kernel = f.repeat(c, 1, 1, 1)
    if mode == 'down':
        return F.conv2d(x, kernel, stride=2, groups=c, padding=pad)

    if mode == 'up':
        return F.conv_transpose2d(x, kernel * 4, stride=2, groups=c, padding=pad)

    raise ValueError(f"Invalid mode: {mode}")

#magnitude preserving fourier features as in EDM2 paper
class MP_Fourier(nn.Module):
    """
    Magnitude-Preserving Fourier Feature Mapping.

    This module maps low-dimensional scalar inputs (like noise levels sigma or timesteps t)
    into high-dimensional embedding vectors using random Fourier features.

    It implements the mapping:
        y = sqrt(2) * cos(2 * pi * (x * frequencies + phases))

    This projection allows (MLPs) to learn high-frequency
    functions in low-dimensional domains (e.g., learning the noise schedule).
    The sqrt(2) factor ensures the output preserves unit variance (Magnitude Preserving),
    assuming the input and weights are properly distributed.
    """
    def __init__(self,num_channels: int, bandwidth: float = 1.0 ):
        """
        Args:
            num_channels (int): The dimension of the output embedding vector.
                                Effectively the number of random frequencies sampled.
            bandwidth (float):  Controls the spread (standard deviation) of the random frequencies.
                                - Higher bandwidth = Higher frequency components (better for fine details).
                                - Lower bandwidth = Lower frequency components (smoother functions).
                                Default is 1.0.
        """
        super().__init__()
        self.register_buffer('freqs', 2 * torch.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * torch.pi * torch.rand(num_channels))
    def forward(self,x: torch.Tensor)-> torch.Tensor:
        """
        Performs the Fourier mapping using an outer product.

        Args:
            x (torch.Tensor): Input scalar values (e.g., noise levels).
                              **Expected Shape:** (Batch_Size,) -> strictly 1D tensor.
                              Note: This cannot be (Batch_Size, 1).

        Returns:
            torch.Tensor: High-dimensional embedding vectors.
                          **Output Shape:** (Batch_Size, Num_Channels).
        """
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#not used right now but can be applied later
class MP_Pos_encoding_2D(nn.Module):
    """
    Magnitude-Preserving 2D Sinusoidal Positional Encoding for Vision Transformers.
    Generates fixed geometric coordinates so the ViT understands spatial layout
    without needing millions of images to learn a 2D grid from scratch.
    """

    def __init__(self, emb_dim: int, seq_ln: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_ln = seq_ln

        # Calculate grid dimensions (assuming square images)
        grid_size = int(np.sqrt(seq_ln))
        assert grid_size * grid_size == seq_ln, "Sequence length must be a perfect square."

        # 1. Create 2D grid coordinates (X and Y)
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        grid_y = grid_y.flatten()
        grid_x = grid_x.flatten()

        # 2. Calculate frequencies (half of emb_dim is for X, half for Y)
        # We divide by 4 because each dimension gets Sine and Cosine (4 parts total)
        dim_half = emb_dim // 4
        omega = torch.exp(-math.log(10000.0) * torch.arange(dim_half) / dim_half)

        # 3. Apply frequencies to grid
        emb_y = grid_y[:, None] * omega[None, :]
        emb_x = grid_x[:, None] * omega[None, :]

        # 4. Generate Sine and Cosine waves
        emb_y = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=1)
        emb_x = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=1)

        # 5. Concatenate Y and X embeddings
        pos_emb = torch.cat([emb_y, emb_x], dim=1)

        # Pad if emb_dim is not perfectly divisible by 4
        if pos_emb.shape[1] < emb_dim:
            pad_size = emb_dim - pos_emb.shape[1]
            pos_emb = torch.cat([pos_emb, torch.zeros(pos_emb.shape[0], pad_size)], dim=1)

        # 6. Apply Magnitude Preserving Normalization
        # This is CRITICAL. It forces the variance to 1.0 so mp_sum doesn't break.
        pos_emb = normalize(pos_emb)

        # Register as a non-learnable fixed buffer of shape (1, seq_ln, emb_dim)
        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))

    def forward(self, x: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """
        Blends the input sequence with the 2D positional embeddings using mp_sum.
        """
        # Ensure the buffer is cast to the correct dtype (e.g., float16/float32) dynamically
        pe = self.pos_emb.to(x.dtype)
        return mp_sum(x, pe, t=t)

#magnitude preserving convolution layer as in EDM2 paper
class MP_Conv(nn.Module):
    """
    Magnitude Preserving Convolution/Linear Layer.

    This layer forces the weights to have unit norm and dynamically scales them
    based on the number of input elements (fan-in). It ensures that the variance
    of the output signal remains consistent with the input signal, preventing
    signal explosion or vanishing gradients.

    It handles both 4D inputs (Convolution) and 2D inputs (Linear/Dense).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 stride: int = 1
                 ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel (tuple): Kernel size, e.g., (3,3) for conv or () for linear.
                            Note: Code logic implies square/odd kernels for correct padding.
        """
        super().__init__()
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(out_channels,in_channels,*kernel))
        assert self.weights.numel() != 0
        self.kernel = kernel
        self.stride = stride

    def forward(self,x: torch.Tensor, gain: float= 1.0)-> torch.Tensor:
        """
        Forward pass with dynamic weight normalization and scaling.

        Args:
            x (torch.Tensor): Input tensor.
                              Shape (N, C) for Linear mode, or (N, C, H, W) for Conv mode.
            gain (float, optional): Scaling factor for the output magnitude.
                                    Defaults to 1.0.

        Returns:
            torch.Tensor: The convolved or projected output.
        """
        w = self.weights.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weights.copy_(normalize(w))

        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if x.ndim == 2:
            # Flatten the weight from (Out, In, 1, 1) to (Out, In)
            w_linear = w.view(self.out_channels, -1)
            return F.linear(x, w_linear)

        assert x.ndim == 4
        k = w.shape[-1]  # kernel size (assuming square)
        total_pad = max(k - self.stride, 0)

        # Optimization: If padding is symmetric (even total_pad), use optimized F.conv2d padding
        if total_pad % 2 == 0:
            pad_val = total_pad // 2
            return F.conv2d(x, w, padding=pad_val, stride=self.stride)
        # Asymmetric Padding (Required for Even kernels with Stride 1, etc.)
        # We pad more on the Right/Bottom to follow PyTorch/TensorFlow convention
        else:
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            # F.pad format: (left, right, top, bottom)
            x = F.pad(x, (pad_left, pad_right, pad_left, pad_right))
            return F.conv2d(x, w, padding=0, stride=self.stride)

#magnitude preserving attention (single and multi_head) layer as in EDM2 paper
#note: we will try to use gated attention mechanism for FUSING the output of the 2 main paths
class MP_Attention(nn.Module):
    """
    Magnitude Preserving Multi-Head Self/Cross Attention (Spatial/Token Mixing).
    Includes Relative Positional Bias (DiffiT Style).
    """
    def __init__(self,
                 num_heads: int,
                 emb_dim: int,
                 seq_ln: int,
                 time_dim: Optional[int] = 0,
                 context_dim: Optional[int] = None,
                 attn_balance: Optional[float] = 0.5,
                 is_cross_attn: Optional[bool] = False):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        self.seq_ln = seq_ln
        self.time_emb = time_dim
        assert emb_dim % num_heads == 0, f"Emb_dim {emb_dim} must be divisible by num_heads {num_heads}"

        self.is_cross = is_cross_attn
        self.attn_balance = attn_balance
        self.time_dependent = True if time_dim > 0 else False

        if context_dim is None:
            context_dim = emb_dim

        # --- Projections ---
        self.q_proj = MP_Conv(in_channels=emb_dim, out_channels=emb_dim, kernel=())
        self.k_proj = MP_Conv(in_channels=context_dim, out_channels=emb_dim, kernel=())
        self.v_proj = MP_Conv(in_channels=context_dim, out_channels=emb_dim, kernel=())

        if self.time_dependent:
            self.q_time = MP_Conv(in_channels=time_dim, out_channels=emb_dim, kernel=())
            if not self.is_cross:
                self.k_time = MP_Conv(in_channels=time_dim, out_channels=emb_dim, kernel=())
                self.v_time = MP_Conv(in_channels=time_dim, out_channels=emb_dim, kernel=())

        self.out_proj = MP_Conv(in_channels=emb_dim, out_channels=emb_dim, kernel=())

        # --- Relative Positional Bias ---
        # Only for Self-Attention. Cross-Attention usually relies on context alignment.
        if not self.is_cross:
            # Shape: (Heads, Seq_Len, Seq_Len)
            # This is a learnable lookup table for attention scores.
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, seq_ln, seq_ln))

            # Initialize with small variance to not disrupt MP start
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self,
                query: torch.Tensor,
                gain_s: float,
                gain_t: float,
                context: Optional[torch.Tensor] = None,
                time_embedding: Optional[torch.Tensor] = None
                )-> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Input sequence. Shape (Batch, Seq_Len, Emb_Dim).
            gain_s (float): Magnitude scaling factor passed to MP_Conv for spatial features.
            gain_t (float): Magnitude scaling factor passed to MP_Conv for temporal features.
            context (torch.Tensor, optional): Context for cross-attention.
                                              Shape (Batch, Seq_Len_Ctx, Context_Dim).
            time_embedding (torch.Tensor, optional):  The diffusion time step embedding used in TMSA for VIT experts
        """
        batch_size, seq_len_q, _ = query.shape

        if context is None:
            context = query
        batch_size, seq_len_kv, _ = context.shape

        # Flatten for MP_Conv (Linear Mode)
        q_flat = query.reshape(batch_size * seq_len_q, -1)
        k_flat = context.reshape(batch_size * seq_len_kv, -1)
        v_flat = context.reshape(batch_size * seq_len_kv, -1)

        # Project
        q = self.q_proj(q_flat, gain=gain_s).view(batch_size, seq_len_q, self.emb_dim)
        k = self.k_proj(k_flat, gain=gain_s).view(batch_size, seq_len_kv, self.emb_dim)
        v = self.v_proj(v_flat, gain=gain_s).view(batch_size, seq_len_kv, self.emb_dim)

        # Time Injection
        if self.time_dependent and time_embedding is not None:
            t_emb = time_embedding.view(batch_size, -1)
            q_t = self.q_time(t_emb, gain=gain_t).unsqueeze(1)
            q = mp_sum(q, q_t.expand(-1, seq_len_q, -1), t=0.3)

            if not self.is_cross:
                k_t = self.k_time(t_emb, gain=gain_t).unsqueeze(1)
                v_t = self.v_time(t_emb, gain=gain_t).unsqueeze(1)
                k = mp_sum(k, k_t.expand(-1, seq_len_kv, -1), t=0.3)
                v = mp_sum(v, v_t.expand(-1, seq_len_kv, -1), t=0.3)

        # Reshape to (B, Heads, Seq, Dim)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / np.sqrt(self.head_dim)

        # --- Add Relative Positional Bias ---
        if not self.is_cross:
            bias = self.rel_pos_bias

            # Interpolation logic for variable sequence lengths (e.g. different patch sizes)
            if seq_len_q != bias.shape[1] or seq_len_kv != bias.shape[2]:
                # Reshape to (1, Heads, S, S) for interpolation
                bias_in = bias.unsqueeze(0)

                # Interpolate to current sequence length
                bias = F.interpolate(
                    bias_in,
                    size=(seq_len_q, seq_len_kv),
                    mode='bilinear',
                    align_corners=False
                )
                bias = bias.squeeze(0)  # (Heads, Sq, Skv)

            # Add bias (Broadcasting over Batch dimension)
            attn_scores = attn_scores + bias.unsqueeze(0)

        # Softmax & Output
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)

        # Reshape & Normalize
        out = out.transpose(1, 2).contiguous().reshape(batch_size * seq_len_q, self.emb_dim)
        out = normalize(out)
        out = self.out_proj(out, gain=gain_s)
        out = out.view(batch_size, seq_len_q, self.emb_dim)

        return mp_sum(query, out, t=self.attn_balance)


class PooledCrossAttn(nn.Module):
    """
    Cheap inter-branch communication module.

    Why pooled instead of full spatial?
    ─────────────────────────────────────
    Full spatial cross-attention at resolution H×W costs O((HW)²) per head.
    For a 32×32 latent that is ~1M ops per head; for 64×64 it becomes ~16M.
    By pooling both branches to a P×P grid (P=4 by default) the cost drops
    to O(P⁴) = 256 ops — independent of input resolution.

    The coarse grid is sufficient because the module's role is to give UNet
    a sense of the *global* structure produced by ViT so it can preserve
    large-scale coherence while refining local detail.  Fine spatial detail
    does not need to cross the branch boundary; only coarse layout does.

    How the transition gate enforces trajectory alignment
    ──────────────────────────────────────────────────────
    The correction tensor is multiplied by gate_strength = w_vit × w_unet.
    This product is 0 when either branch is fully suppressed (σ extremes)
    and reaches its maximum of 0.25 at σ = transition_sigma.

    At high σ (ViT dominant): gate_strength → 0  → no UNet influence on ViT
    At low  σ (UNet dominant): gate_strength → 0  → no ViT  influence on UNet
    At transition σ:           gate_strength = 0.25 → maximal communication

    Initialisation
    ───────────────
    out_proj is zero-initialised.  At step 0 the module outputs all-zeros,
    making v3 identical to v2 at initialisation.  The module grows its
    contribution organically as out_proj learns non-zero weights.

    Args:
        channels    : C — number of feature channels (= internal_channels).
        num_heads   : Number of attention heads in MP_Attention.
        attn_balance: Residual blend factor inside MP_Attention.
    """

    def __init__(
            self,
            channels: int,
            num_heads: int,
            img_resolution: int = 64,
            pool_size: int = 16,
            attn_balance: float = 0.5,
    ):
        super().__init__()

        assert channels % num_heads == 0, "Channels must be divisible by num_heads."

        self.pool_size = pool_size
        seq_ln = img_resolution * img_resolution
        pool_seq = pool_size * pool_size

        # Fix 6a: Positional encodings for spatial anchoring
        self.pos_enc_q = MP_Pos_encoding_2D(emb_dim=channels, seq_ln=seq_ln)
        self.pos_enc_kv = MP_Pos_encoding_2D(emb_dim=channels, seq_ln=pool_seq)

        # Fix 6b: Pure cross-attention (attn_balance=1.0), no query self-echo
        self.cross_attn = MP_Attention(
            num_heads=num_heads,
            emb_dim=channels,
            seq_ln=0,
            context_dim=channels,
            attn_balance=1.0,
            is_cross_attn=True,
        )

        self.out_proj = MP_Conv(
            in_channels=channels,
            out_channels=channels,
            kernel=(1, 1),
        )
        # Fix 3: Keep weights on the unit sphere (default init).
        # Gate output with a learnable scalar initialized to 0 instead.
        self.out_proj_gate = nn.Parameter(torch.zeros(1))

    def forward(
            self,
            unet_feat: torch.Tensor,  # (B, C, H, W)
            vit_feat: torch.Tensor,  # (B, C, H, W)
            gate_strength: torch.Tensor,  # (B, 1, 1, 1)
    ) -> torch.Tensor:
        B, C, H, W = unet_feat.shape

        # 1. Query: flatten ViT features to sequence and add positional encoding
        vit_seq = vit_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        vit_seq = self.pos_enc_q(vit_seq, t=0.3)       # Fix 6a: spatial anchoring

        # 2. K/V: Pool U-Net features to fixed grid, then flatten + pos encode
        # Fix 6c: Reduces cost from O((HW)^2) to O(HW * pool_size^2)
        unet_pooled = F.adaptive_avg_pool2d(unet_feat, self.pool_size)  # (B, C, P, P)
        unet_seq = unet_pooled.flatten(2).transpose(1, 2)  # (B, P*P, C)
        unet_seq = self.pos_enc_kv(unet_seq, t=0.3)   # Fix 6a: spatial anchoring

        # 3. Cross-attention: ViT queries pooled U-Net
        attn_out = self.cross_attn(
            query=vit_seq,
            context=unet_seq,
            gain_s=1.0,
            gain_t=1.0,
        )

        # 4. Reshape directly back to spatial grid
        attn_full = attn_out.transpose(1, 2).view(B, C, H, W)

        # 5. Project and gate
        correction = self.out_proj(attn_full) * self.out_proj_gate

        return correction * gate_strength