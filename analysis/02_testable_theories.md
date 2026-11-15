# Phase 3a: Formal Testable Theories (Architecture vs. Loss vs. Under-Convergence)

Based on the grounded mathematical sources in [01_source_registry.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/01_source_registry.md), the observed code mechanics, and the `hdmoem_v3_nature50k_256_v2` training diagnostics, we formulate **six explicit, falsifiable hypotheses** separating **Architectural**, **Loss-Related**, and **Under-Convergence** failure mechanisms (and distinguishing pre-`f945711` `v2` bugs from persistent `f945711` / `v3` properties).

---

## Summary Hypothesis Matrix

| ID | Category | Mechanism Under Test | Primary Observed Symptom Explained |
| :--- | :--- | :--- | :--- |
| **T1** | Architectural | Nested zero-init gates (`0 * 0` eliminating singularities) + pre-`f945711` `mp_sum` variance contraction & `resample` scaling | Slow/stalled early learning, tiny `blend_residual` update ratio (`~2e-4`), dark/under-structured high-$\sigma$ predictions |
| **T2** | Loss + Baseline Math | `EDM` trivial $F_\theta=0$ baseline & Wiener MMSE risk vs. low-$\sigma$ velocity collapse (`c_skip`-only trap) | `Train MSE = 0.2134` / `Val Probe MSE = 0.2157` flat plateau, `||(x-D_x)/\sigma||_RMS -> 0.05` & `cos_sim -> 1.00` as $\sigma \to 0.002$ |
| **T3** | Loss-Related | `laplacian_decomposition` isotropic white-noise skew (`>90%` in `P_high`) + `ZetaScheduler` / `MaskGenerator` coupling | `ViT Branch (High-Freq)` loss **rising** `0.35 -> 0.65` above `Pure EDM Loss` (`0.58`), while `UNet Branch (Low-Freq)` stays `~0.08` |
| **T4** | Loss + Architectural | `EDM_LOSS_NOD` gradient conflict (`pure_loss` vs `vit_loss`/`unet_loss`) & ViT `[2,2]` + `PixelShuffle(2)` spatial aliasing | Generated RAPSD excess spectral power at `cycles 2–12` and harsh high-frequency patchy sample artifacts |
| **T5** | Architectural | `spherical_blend` geometry vs. true $S^n(R)$ geodesic SLERP & branch disagreement at high $\sigma$ (`cos_sim ~ 0.71–0.76`) | High-$\sigma$ branch disagreement without structural synthesis; potential norm/gradient distortion across blend zone |
| **T6** | Under-Convergence | 8,000-step horizon + Cosine LR starvation below $10^{-4}$ vs. `MP_Conv` tangent-step radius $\sqrt{N_j}$ | All component update ratios collapsing below `1e-4` by step ~6000; distinguishing true step-budget insufficiency vs. structural floor |

---

## T1: Nested Zero-Init Eliminating Singularities & Pre-`f945711` Signal Contraction (Architectural)

- **Mathematical Grounding:** Amari (`[B1]`, Ch. 12.2.1–12.2.5) proves that parameters multiplied by zero-valued output gates ($v_i = 0$) lie on **eliminating singularities ($R_e$)** with degenerate Fisher metric ($a^\top G a = 0$) and cubic plateau drift ($du/dt = O(|u|^3)$). Strogatz (`[B2]`, Ch. 2.4–2.7, 6.3) shows saddle escape requires $\tau \sim 1/\lambda_+$ integrated step time. EDM2 (`[P1]`, App. B.7 Eq. 104) uses only a **single** zero-init scalar gain at the final network output, whereas `HDMOEM_v3` nests zero-init gates in series (`res_gate`, `tmsa_gate`, `mlp_gate`, `out_proj_gate` $\to$ `unet_proj_gate`/`vit_proj_gate` $\to$ `blend_gate`). Furthermore, Calin (`[B3]`, Ch. 6.2.6) and EDM2 (`[P1]`, Eq. 88) show that pre-`f945711` `mp_sum(x, branch * 0, balance)` strictly contracts activation variance by $\frac{(1-t)^2}{(1-t)^2+t^2}$ at every block!
- **Falsifiable Predictions:**
  1. At initialization (step 0) on `preconditioned_HDMOEM_v3`, exact backward gradients $\partial \mathcal{L}/\partial W$ on **all** inner `Unet_block` conv weights, `Vit_block` TMSA/MLP weights, and `PooledCrossAttn` weights are **identically `0.0`**, while outer gates must absorb multiple Adam steps before gradients penetrate inside branches.
  2. Under pre-`f945711` `mp_sum(x, 0, balance)`, forward signal RMS shrinks exponentially across UNet/ViT depth (verified directly vs. `f945711` `mp_gated_sum`).
  3. Because `blend_gate` and inner gates start at `0`, `blend_residual` suffers severe early gradient suppression and stalls at update ratio `~2e-4` before cosine LR decays it to `1.5e-5`.

---

## T2: Skip-Connection Degeneration at Low $\sigma$ & The `0.213–0.216` MSE Floor (Loss + Noise Geometry)

- **Mathematical Grounding:** In EDM (`[P2]`, App. B.6) and DSM / Tweedie's formula (`[B5]` Ch. 4.1–4.3; `[B7]` Murphy Eq. 24.31–24.36), $D_\theta(x; \sigma) = c_{\text{skip}}(\sigma)x + c_{\text{out}}(\sigma)F_\theta(\dots)$. At $\sigma \ll \sigma_d$, $c_{\text{skip}} \to 1$ and $x - c_{\text{skip}} x = \frac{\sigma^2}{\sigma^2 + \sigma_d^2} x = O(\sigma^2)$, whereas true data-minus-noisy displacement is $-\sigma \epsilon = O(\sigma)$. Consequently, a Bayes-optimal denoiser maintains normalized velocity RMS $\|(x_\sigma - D^*)/\sigma\|_{\text{RMS}} \ge 1$ even as $\sigma \to 0.002$ (and in fact equal to $1$ for zero error, $\sqrt{1 + \text{norm\_err}^2}$ otherwise), whereas if $F_\theta \approx 0$ at low $\sigma$, $\|(x - D)/\sigma\|_{\text{RMS}} = \frac{\sigma}{\sigma^2+\sigma_d^2}\|x\|_{\text{RMS}} \to 0$! Meanwhile, at $F_\theta \equiv 0$, EDM's weighted loss equals $1.0$ when $\operatorname{Var}(x_0) = \sigma_d^2$, and unweighted reconstruction MSE depends critically on the sampled $\sigma$ distribution (`sample_sigma_hybrid` vs. fixed probe bins).
- **Falsifiable Predictions:**
  1. Computing $\|(x - c_{\text{skip}}x)/\sigma\|_{\text{RMS}}$ analytically at $\sigma = 0.002\text{--}80$ with $\sigma_d = 0.8$ will match Plot 1's qualitative collapse as $\sigma \to 0$ (**proving Plot 1's `0.05` floor at low $\sigma$ is the model outputting virtually $F_\theta \approx 0$ / dead $D_{\text{vit}}$ & $D_{\text{unet}}$ at low noise** — which simultaneously causes `cos_sim(D_vit, D_unet) = 1.00` because both equal $c_{\text{skip}} x$!).
  2. Evaluating exact **trivial $F_\theta = 0$ MSE** and **Wiener MMSE** (`[B7]` Eq. 194, $H_W(k) = S_{xx}(k)/(S_{xx}(k) + \sigma^2)$) across $\sigma$ on $32\times 32\times 4$ natural-$1/f^2$ latents will quantify precisely what fraction of `MSE = 0.21567` comes from high-$\sigma$ Bayes uncertainty vs. high-$\sigma$/low-$\sigma$ optimization failure!

---

## T3: Fourier `laplacian_decomposition` White-Noise Skew & `vit_loss` Rising Inversion (Loss-Related)

- **Mathematical Grounding:** By Parseval's theorem (`[B7]` p. 90; `[B4]` Ch. 3), `laplacian_decomposition` splits frequency space into radial low-pass mask $\mathcal{P}_{\text{low}}$ and complementary high-pass mask $\mathcal{P}_{\text{high}} = I - \mathcal{P}_{\text{low}}$ with Gaussian radial cutoff `cutoff_sq = ln(2)/(4 pi^2 sigma_lap^2)`. Additive white noise $n \sim \mathcal{N}(0, \sigma^2 I)$ has flat PSD $S_{nn}(k) = \sigma^2$ across all discrete $32\times 32$ frequencies, while natural latents concentrate energy in low $k$.
- **Falsifiable Predictions:**
  1. On a $32\times 32$ DFT grid, for both `laplacian_std = 1.0` (used in earlier/v2 configs) and `0.5` (`v3` notebook), $\mathcal{P}_{\text{high}}$ captures a strongly skewed majority of white Gaussian noise variance (e.g., $\gg 85\%$ at $\sigma_{\text{lap}}=1.0$ vs. smaller but still majority at $0.5$), explaining why at $F_\theta = 0$ weighted `unet_loss` ($\mathcal{P}_{\text{low}}$) is tiny (`~0.08`) while `vit_loss` ($\mathcal{P}_{\text{high}}$) carries the bulk of high-frequency noise energy (`~0.35–0.9+` depending on `w_vit` and `laplacian_std`)!
  2. Tracing the exact formula in `EDM_LOSS_NOD` (`Utils/loss.py`) with `ZetaScheduler` annealing ($1 \to 0$ over steps `0–2400`) and `MaskGenerator` (`log_var`, `w_vit`, `w_unet`, `zeta`) will show **how `vit_loss` rises from `0.35` to `0.65` during steps `0–2000`**: specifically, whether `MaskGenerator` / `Zeta` weight annealing or learned variance $e^{-u}$ shift (`Pure EDM Loss` drops `1.53 -> 0.58` as $u(\sigma)$ grows/adapts, changing $e^{-u.\text{detach}()}$ or ViT noise predictions!) causes `vit_loss` to climb above `pure_loss` (`0.65 > 0.58`)! (Note: we will verify exact mechanism in a probe!)

---

## T4: Multi-Objective Gradient Conflict & ViT `PixelShuffle(2)` Spatial Aliasing (Loss + Architectural)

- **Mathematical Grounding:** FreeU (`[P4]`, Sec. 1–2) demonstrates that decoupling low-frequency structural backbone from high-frequency skip details must preserve coherent phase reconstruction; Shao et al. (`[P7]`) show pullback metrics must remain well-conditioned across generators; DiffiT (`[P5]`) avoids feeding raw `PixelShuffle(2)` outputs directly into a Nyquist-band high-pass loss without multi-scale mixing. Meanwhile, optimizing shared stem parameters under $\alpha_{\text{blend}}\mathcal{L}_{\text{pure}} + \alpha_{\text{vit}}\mathcal{L}_{\text{vit}} + \alpha_{\text{unet}}\mathcal{L}_{\text{unet}}$ exposes shared layers to potentially conflicting projections ($I$ vs. $\mathcal{P}_{\text{high}}$ vs. $\mathcal{P}_{\text{low}}$).
- **Falsifiable Predictions:**
  1. Inside `preconditioned_HDMOEM_v3`, gradients from `vit_loss` and `unet_loss` wrt shared parameters (`input_proj`, `PooledCrossAttn`, `time_proj`) exhibit measurable cosine conflict or severe magnitude imbalance against `pure_loss`.
  2. Patch-wise processing `[2,2]` $\to$ `PixelShuffle(2)` + `1x1` conv (or `3x3` conv with zero/random weights) introduces periodic spectral peaks / excess energy in the radial power spectrum (RAPSD `cycles 2–12`), directly matching Plot 1's excess generated RAPSD over real `nature_50k` latents.

---

## T5: `spherical_blend` Non-Geodesic Distortion on $S^n(R)$ (Architectural)

- **Mathematical Grounding:** Lee (`[B6]`, Prop. 5.27) proves that true constant-speed geodesics on the sphere $S^n(R)$ obey $\gamma(t) = (\cos at)p + (\sin at)\tilde{v}$ (preserving norm $\|\gamma(t)\| = R$), whereas naive convex combinations $(1-t)p + t q$ or unnormalized angle approximations shrink norm along chords when $\theta > 0$. Notice Plot 1 shows `cos_sim(D_vit, D_unet) ~ 0.71–0.76` at high $\sigma$, meaning $\theta \approx 40^\circ\text{--}45^\circ$!
- **Falsifiable Predictions:**
  1. Testing `spherical_blend` from `models/model_config_v3.py` (or `model_components.py`) directly across angles $\theta \in [0, \pi]$ and mismatched magnitudes $\|v_{\text{vit}}\| \neq \|v_{\text{unet}}\|$ will reveal exact norm error $\Delta \|\cdot\|$ and Jacobian gradient attenuation $\partial \text{blend}/\partial v_{\text{vit}}$, $\partial \text{blend}/\partial v_{\text{unet}}$ vs. linear blend and true SLERP.

---

## T6: Premature Cosine LR Decay & Finite 8,000-Step Horizon (Under-Convergence vs. Ceiling)

- **Mathematical Grounding:** nGPT (`[P3]`, Eq. 22) and EDM2 (`[P1]`, Eq. 59, 66, inverse-sqrt schedule) show that Adam's effective angular step size on `MP_Conv` spheres $\|w\|_2 = \sqrt{N_j}$ scales inversely with radius $\sqrt{N_j}$ and decays proportionally to $\alpha(t)$, while EDM2 trains for hundreds of millions of images (tens/hundreds of thousands of steps) with $\alpha(t) \propto 1/\sqrt{\max(t/t_{\text{ref}}, 1)}$. In `v2`, `total_steps = 8000` with cosine decay drops component update ratios below $10^{-4}$ after step ~6000 (Plot 6a), while `blend_residual` never exceeds `2.5e-4`.
- **Falsifiable Predictions:**
  1. Measuring per-component parameter norm vs. theoretical Adam tangential rotation $\Delta \theta \approx \int_0^{8000} \alpha(t) / \sqrt{N_j}\, dt$ will quantify the maximum possible representation change in `8,000` cosine steps vs. `8,000` WSD steps vs. `70,000` EDM2 steps.
  2. Combined with **T1–T5**, we will determine rigorously whether extending steps alone to 50k–70k (fixing under-convergence) would converge cleanly OR remain capped by the loss/architectural bottlenecks (`vit_loss` spectral skew, `EDM_LOSS_NOD` objective interference, nested zero-gates, low-$\sigma$ `c_skip` deadness).
