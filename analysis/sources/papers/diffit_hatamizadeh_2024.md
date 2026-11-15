# Paper Summary: *DiffiT: Diffusion Vision Transformers for Image Generation*

- **Authors:** Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat (NVIDIA)
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\scientifc papers\DiffiT, Diffusion Vision Transformers for Image Generation.pdf`
- **Sections Read:** Main Sections 1–3.2 (Time-dependent Multihead Self-Attention / TMSA).

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Time-Dependent Multihead Self-Attention (TMSA) (Sec. 1–3.2)
- Standard DiT injects timestep conditioning only via adaptive LayerNorm scale/shift (adaLN), whereas **DiffiT** dynamically conditions self-attention projections ($Q, K, V$) directly on timestep/noise embedding $x_t$ alongside spatial tokens $x_s$:
  $$q_t = x_s W_{qs} + x_t W_{qt}, \quad k_t = x_s W_{ks} + x_t W_{kt}, \quad v_t = x_s W_{vs} + x_t W_{vt}.$$
- **Architectural premise of DiffiT:** A single ViT/U-shaped Transformer applies TMSA **across the entire denoising trajectory** (full-spectrum generation with smooth spatial downsampling/upsampling stages—e.g. U-shaped multi-scale DiffiT for high resolution without raw patch-only final reconstruction), adapting receptive/attention behavior with noise level $\sigma$ rather than partitioning $\sigma$ or FFT bands hard across disconnected sub-networks.

---

## 2. Direct Relevance to `HDMOEM_v3`
- `HDMOEM_v3`'s `Vit_block` uses `MP_Attention` (with time-dependent $Q/K/V$ projections inspired by TMSA) followed by `tmsa_gate` and `mlp_gate` (both zero-initialized!).
- Unlike U-shaped DiffiT which blends multi-resolution convolutional/tokenizer stages seamlessly with non-zero residual identity propagation, `HDMOEM_v3`'s ViT branch operates at patch tokenization `[2, 2]` (`16x16` tokens on a `32x32` latent) and reconstructs back to `32x32` via a single `PixelShuffle(2)` + `output_proj_vit` (`3x3` in `v3`, `1x1` in earlier variants), creating an asymmetric spatial bottleneck when forced by `vit_loss` to carry high-frequency detail (`P_high` near Nyquist!).
