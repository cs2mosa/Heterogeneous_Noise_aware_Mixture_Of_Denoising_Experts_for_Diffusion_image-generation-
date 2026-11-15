# Phase 0: Model Identification and Mathematical Subject Map

## 1. Model Family & Architecture Identification

- **System Name:** `HDMOEM_v3` wrapped in `preconditioned_HDMOEM_v3` (`models/model_config_v3.py`), trained with `EDM_LOSS_NOD` (`Utils/loss.py`) and sampled with `EDM_sampler` (`Utils/EDM_sampler.py`), configured for latent text-to-image diffusion (`32x32x4` VAE latents corresponding to `256x256` RGB images, CLIP `768`-d text conditioning, `sigma_data = 0.8`).
- **Model Family:** Continuous-time Score-Based / Elucidated Diffusion Model (EDM) with an **EDM2-style Magnitude-Preserving (MP) Backbone** and a **Dual-Branch Heterogeneous Mixture-of-Experts (U-Net + ViT) denoiser** fused along the noise trajectory $\sigma \in [\sigma_{\min}, \sigma_{\max}] = [0.002, 80.0]$.
- **Component Breakdown (directly observed in workspace code):**
  1. **Preconditioning & Score-to-Denoiser Parametrization (`preconditioned_HDMOEM_v3`, `models/model_config_v3.py:459-620`):**
     - $D_\theta(x; \sigma, c) = c_{\text{skip}}(\sigma) x + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma) x; c_{\text{noise}}(\sigma), c)$
     - $c_{\text{skip}}(\sigma) = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}$, $c_{\text{out}}(\sigma) = \frac{\sigma \sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}$, $c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_d^2}}$, $c_{\text{noise}}(\sigma) = \frac{1}{4}\ln \sigma$, with $\sigma_d = 0.8$ (calibrated in notebook) or `0.5`/`0.8` in configs.
     - Uncertainty estimation head `log_var(c_noise)` trained inside `pure_loss` (`Utils/loss.py:168-181`).
  2. **Magnitude-Preserving Weight-Normalized Primitives (`models/model_internals.py`):**
     - `MP_Conv`: weights normalized to unit L2 norm over fan-in (`weight / ||weight||_2 * sqrt(N)`) with forced weight normalization on forward (`normalize(w)`).
     - `normalize(x)`: L2 activation normalization along channels/hidden dim (`x / ||x||_2 * sqrt(C)`), projecting activations onto the scaled hypersphere $\sqrt{d}\,\mathbb{S}^{d-1}$.
     - `mp_silu`, `mp_sum`, `mp_gated_sum`, `mp_cat`, `resample`, `MP_Attention`, `PooledCrossAttn`.
  3. **Trajectory-Conditioned Heterogeneous MoE (`HDMOEM_v3`, `models/model_config_v3.py:100-456`):**
     - Shared spatial input stem (`MP_Conv`, `MP_Fourier`, `time_proj_gate`), feeding TWO parallel MoE banks:
       - **UNet MoE bank** (`num_experts=2`, `top_k=1`, `ParametricTrajectoryRouter`): convolutional multi-scale encoders/decoders intended for high-$\sigma$ global structure.
       - **ViT MoE bank** (`num_experts=2`, `top_k=1`, `ParametricTrajectoryRouter`): patch-tokenized self-attention (`VIT_patch_sizes=[2,2]` or `[1,2]`, `PixelShuffle(2)` upsampling + `output_proj_vit`), intended for low-$\sigma$ detail/texture.
     - Cross-branch `PooledCrossAttn` (`pool_size=16` or spatial cross-attn) gated by `gate_strength = w_vit * w_unet` (`in [0, 0.25]`), plus zero-initialized scalar residual gates (`time_proj_gate`, `unet_proj_gate`, `vit_proj_gate`, `blend_gate`, `res_gate`, `tmsa_gate`, `mlp_gate`, `out_proj_gate`).
     - Trajectory blend (`spherical_blend` on `out_gain_vit * vit_raw_out` and `out_gain_unet * unet_raw_out` with `w_vit + w_unet = 1`) + additive `attn_correction`.
  4. **Objective Function (`EDM_LOSS_NOD`, `Utils/loss.py:81-248`):**
     - `pure_loss`: uncertainty-weighted EDM denoising MSE $\mathbb{E}[\lambda(\sigma) e^{-u} \|D_{\text{blend}} - x\|_2^2 + u]$.
     - `unet_loss`: low-pass FFT projection $\mathbb{E}[\lambda(\sigma) e^{-u.\text{detach}()} w_{\text{unet}}.\text{detach}() \|\mathcal{P}_{\text{low}}(D_{\text{unet}} - x)\|_2^2]$.
     - `vit_loss`: high-pass FFT projection $\mathbb{E}[\lambda(\sigma) e^{-u.\text{detach}()} w_{\text{vit}}.\text{detach}() \|\mathcal{P}_{\text{high}}(D_{\text{vit}} - x)\|_2^2]$.
     - Auxiliary router losses: MoE load-balancing (`PathPriorLoss` with CDF bounds), router Z-loss, router entropy.
  5. **Training Curriculum & Noise Sampling (`Utils/utils.py`, `colab_text_guided_training.ipynb`):**
     - Hybrid log-normal + uniform $\ln \sigma$ sampling (`sample_sigma_hybrid`), `ZetaScheduler` (mixing uniform soft routing with trajectory-guided routing during warmup), `MaskGenerator` (probabilistic branch bandwidth masking), Adam with component LRs (`Cosine` in v2 plot / `WSD` in v3 notebook, `total_steps=8000`), EMA.

---

## 2. Observed Failure Symptoms (from `hdmoem_v3_nature50k_256_v2` Diagnostic Plots)

1. **Reconstruction MSE Plateau (Plots 3a, 7b):** `Train MSE = 0.21340` and `Val Probe MSE = 0.21567` plateau early (~step 1000–2000) with zero train-val gap (neither overfitting nor memorization; classic optimization/capacity/objective bottleneck).
2. **ViT High-Frequency Loss Inversion (Plot 7a):** While `Pure EDM Loss` drops from `~1.53` to `~0.58`, `ViT Branch (High-Freq)` loss **rises** from `~0.35` (step 0–500) to `~0.65` (step 2000) and settles **above** `Pure EDM Loss`. `UNet Branch (Low-Freq)` stays tiny (`~0.08`).
3. **Lipschitz Velocity Collapse at Low $\sigma$ & Branch Redundancy Post-Skip (Plot 1):** `||(x - D_x)/sigma||_RMS` collapses from `~1.01` ($\sigma \ge 3$) down to `~0.05` ($\sigma \le 0.02$), while `cos_sim(D_vit, D_unet) -> 1.00` for $\sigma < 0.3$ and generated latents show excess mid/high-frequency power in RAPSD (`cycles 2–12`).
4. **Router Health is Clean, Update Ratios Decay Late (Plots 5, 6):** Routers have normalized entropy `~0.95–1.0`, Gini `< 0.25`, `0` dead experts, clean 2-expert specialization across $\sigma$. `blend_residual` has the lowest update ratio (`~2e-4 -> 1.5e-5`), and all components fall below `1e-4` after step ~6000 under cosine LR decay.
5. **Sample Artifacts (Plots 2, 8):** Dark/under-structured high-$\sigma$ predictions ($\sigma=80 \to 16.59$) and harsh, patchy/high-frequency artifacts at $\sigma \to 0$.

---

## 3. Relevant Mathematical Subjects (for Grounding & Derivations)

| Code Mechanism / Symptom | Required Mathematical Discipline | Specific Mathematical Objects & Questions |
| :--- | :--- | :--- |
| `MP_Conv`, `normalize(x)`, `spherical_blend`, `nGPT`-style hyperspherical constraints | **Riemannian Geometry & Optimization on Manifolds** ($\mathbb{S}^{d-1}$) | Tangent space projection $(I - \hat{w}\hat{w}^\top)\nabla$, Adam / radial-vs-tangential weight scaling (`||w|| = sqrt(N)`), geodesic/SLERP interpolation vs scaled-vector addition (`w_1 v_1 + w_2 v_2` vs $\sin/\cos$ norm collapse). |
| Score-based diffusion, $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}$, `log_var`, `sample_sigma_hybrid` | **Stochastic Processes, Diffusion Theory & Statistical Decision Theory** | Tweedie's formula, MMSE / Wiener optimal denoiser $\mathbb{E}[x_0 \mid x_\sigma]$, SNR scaling, ELBO / Fisher-divergence weighting $\lambda(\sigma) = \frac{\sigma^2+\sigma_d^2}{\sigma^2\sigma_d^2}$, heteroscedastic likelihood $e^{-u}\|r\|^2 + u$. |
| `laplacian_decomposition`, FFT orthogonal projectors $(\mathcal{P}_{\text{low}}, \mathcal{P}_{\text{high}})$, `PixelShuffle(2)` + `1x1` conv, RAPSD | **Harmonic / Fourier Analysis & Signal Processing** | Parseval's identity, $1/f^\alpha$ natural latent PSD vs white-noise PSD, exact energy partition of $\mathcal{P}_{\text{low}}/\mathcal{P}_{\text{high}}$, Wiener shrinkage $H(k,\sigma) = \frac{S(k)}{S(k)+\sigma^2}$, Bayes-risk floor vs trivial $F=0$ baseline. |
| Nested zero-initialized scalar gates (`res_gate`, `tmsa_gate`, `mlp_gate`, `vit_proj_gate`, `unet_proj_gate`) + `mp_sum` vs `mp_gated_sum` | **Dynamical Systems, Saddle Points & Gradient Flow** | Linearization around symmetric fixed point ($g=0$), bilinear Jacobian rank-deficiency ($\partial \mathcal{L}/\partial g = \langle \nabla_y \mathcal{L}, y_0 \rangle$ vs $\partial \mathcal{L}/\partial W = g \cdots$), escape timescale from zero saddle under Adam and finite cosine LR budget, variance contraction in un-gated vs `mp_sum` chains. |
| `EDM_LOSS_NOD`, gradient interference between `pure_loss`, `vit_loss`, `unet_loss`, `uncertainty_scale.detach()`, `w.detach()`, `ZetaScheduler` | **Information Geometry & Multi-Objective Optimization** | Pullback Fisher metric on statistical manifolds, gradient conflict / Pareto stationarity of competing projections ($\alpha_{\text{blend}} I + \alpha_{\text{vit}} \mathcal{P}_{\text{high}}$ vs shared parameters), stop-gradient biases in heteroscedastic & MoE weights. |
| `ParametricTrajectoryRouter`, `PathPriorLoss` (`cdf_utils`), Top-1 routing | **Probability, Optimal Transport & Mixture-of-Experts Theory** | Empirical CDF matching / 1D OT Wasserstein balance loss, entropy vs load-balance tension, conditional variance across $\sigma$ bins. |
