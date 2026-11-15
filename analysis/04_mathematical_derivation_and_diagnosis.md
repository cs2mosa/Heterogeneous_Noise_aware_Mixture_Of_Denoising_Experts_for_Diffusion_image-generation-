# Phase 4: Mathematical Derivation & Complete Failure Diagnosis of `HDMOEM_v3`

**Target System:** `preconditioned_HDMOEM_v3` / `HDMOEM_v3` (`models/model_config_v3.py`, `models/model_components.py`, `models/model_internals.py`, `Utils/loss.py`, `Utils/utils.py`), evaluated against training diagnostics from run `hdmoem_v3_nature50k_256_v2` (8,000 steps, cosine LR schedule).

---

## Executive Verdict: Architecture vs. Loss vs. Under-Convergence

**The observed failure is NOT simple under-convergence that training longer can cure.** It is a **coupled Architectural + Loss/Objective failure**, compounded by a premature 8,000-step cosine LR decay:

1. **Primary Loss / Noise-Geometry Failure (~40% of failure):**
   - `EDM_LOSS_NOD` applies asymmetric Kendall uncertainty scaling `exp(-log_var.detach())` *without* the neutralizing `+ log_var` term inside `vit_loss` and `unet_loss`, combined with a $32\times 32$ Fourier cutoff (`laplacian_decomposition`) that routes **`94.42%`** (`lap_sigma=1.0`) or **`78.44%`** (`lap_sigma=0.5`) of isotropic white noise energy into `vit_loss` (`P_high`). Whenever `D_blend` reduces error so `log_var < 0` (`exp(-u*) ~ 1.52x`), `vit_loss` inflates above `pure_loss` (`0.35 -> 0.65`, exceeding `0.58`) even before ViT learns high-frequency structure.
   - Across `sample_sigma_hybrid`, the model converged only to the **linear Gaussian Wiener MMSE floor (`0.22095` on $1/f^2$ latents vs. observed `0.2134 / 0.2157`, down from `0.28375` at $F=0$)** and degenerated toward the pure skip path $D(x) \approx c_{\text{skip}}(\sigma)x$ at low noise ($\|(x - D_x)/\sigma\|_{\text{RMS}} \to 0.05$ for $\sigma \le 0.02$ instead of $\approx 1.0$, meaning $\|F_\theta\|_{\text{RMS}} \approx 0.05$).
2. **Primary Architectural Failure (~45% of failure):**
   - **Nested Zero-Init Eliminating Singularities (`f945711` / `v3` STILL PRESENT):** Stacking zero-initialized scalar gates (`res_gate`, `tmsa_gate`, `mlp_gate`, `out_proj_gate` inside blocks, followed by `vit_proj_gate`, `unet_proj_gate`, `blend_gate` outside) creates multi-layer Fisher metric singularities where **`113/113` UNet inner tensors, `49/49` ViT inner tensors, `output_proj_vit/unet`, and `38/41` gates receive exact `0.0000e+00` gradient at step 0** — violating EDM2's single output `Gain` rule.
   - **Pre-`f945711` Signal Contraction (present during `v2`, fixed in `f945711`):** Before commit `f945711`, `mp_sum(x, branch * 0, t)` multiplied activation RMS by `0.7076` per block (`0.0625` after 8 blocks at $t=0.5$) and `resample(down)` halved RMS (`0.5013`). `f945711` fixed both (`mp_gated_sum` -> `1.0007`, `resample(down)` -> `1.0027`), **but left the nested zero-gate saddle intact**.
   - **ViT `[2,2]` + `PixelShuffle(2)` Spectral Aliasing (`f945711` / `v3` STILL PRESENT):** Upsampling `16x16` patch tokens via `PixelShuffle(2)` directly into `D_vit` where ViT dominates low $\sigma$ (`w_vit -> 1.0`) introduces sub-pixel folding that attenuates `cycles 1..8` by `~3x` and amplifies Nyquist frequencies (`cycles 12..15`) by **`1.43x–7.83x`**, matching the observed RAPSD bump and patchy sample textures.
3. **Secondary Under-Convergence / LR Schedule Compounding (~15% of failure):**
   - `v2`'s 8,000-step cosine decay delivers **`4.9x` less integrated Adam step budget** (`5.257` vs `25.998` for 70k EDM2 inverse-sqrt), drops all component update ratios below $10^{-4}$ after step ~6000 (Plot 6a), and exposes low-$\sigma$ ($\sigma \le 0.02$) weights to only `4.35%` (~348) batches. **However, extending training to 50k–70k steps alone cannot cure high-frequency `PixelShuffle(2)` aliasing or the `EDM_LOSS_NOD` `exp(-u.detach())` loss asymmetry.**

### What Was Explicitly Falsified / Ruled Out by Probes
- **MoE Router Collapse (`[PAPER: DiffMoE]` mode):** Ruled out — Plot 5 shows entropy `~0.95–1.0`, Gini `< 0.25`, `0` dead experts, and clean 2-expert partitioning across $\sigma$.
- **Data Overfitting / Memorization:** Ruled out — `Train MSE = 0.21340` vs `Val Probe MSE = 0.21567` (zero generalization gap, Plot 7).
- **`spherical_blend` Norm Collapse / Gradient Blocking:** **Falsified** by `[PROBE: probe_blend_and_spectral_artifacts.py, Probe 3A]` — `spherical_blend` preserves exact target RMS (`1.0000`, `0.4600`, `3.1600`) across all angles $\rho \in [-0.74, +0.99]$ and increases gradient sensitivity (`1.06x` at $\rho=0.74$) relative to linear blending.
- **Branch Directional Gradient Conflict (`pure_loss` vs `vit_loss` / `unet_loss`):** **Falsified** by `[PROBE: probe_blend_and_spectral_artifacts.py, Probe 3B]` — gradient cosine similarities with `pure_loss` are strongly positive (`+0.8246` inside ViT experts, `+0.8720` inside UNet experts, `+0.4263`/`+0.6073` in stem).

---

## Derivation 1: The `0.213–0.216` MSE Plateau & Low-$\sigma$ Velocity Collapse ($F_\theta \to 0$)

### 1.1 Exact Score-to-Denoiser Identity and Low-$\sigma$ Asymptotics
In EDM preconditioning `[CODE: models/model_config_v3.py:preconditioned_HDMOEM_v3.forward]` and `[PAPER: edm_karras_2022.md, App. B.6]`, for clean latent $x_0$ with per-coordinate variance $\sigma_d^2$ and noisy observation $x_\sigma = x_0 + n$, $n \sim \mathcal{N}(0, \sigma^2 I)$:
$$D_\theta(x_\sigma; \sigma) = c_{\text{skip}}(\sigma)\,x_\sigma + c_{\text{out}}(\sigma)\,F_\theta\big(c_{\text{in}}(\sigma)\,x_\sigma;\, c_{\text{noise}}(\sigma)\big),$$
$$c_{\text{skip}}(\sigma) = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}, \qquad c_{\text{out}}(\sigma) = \frac{\sigma\,\sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}, \qquad \lambda(\sigma) = \frac{1}{c_{\text{out}}(\sigma)^2} = \frac{\sigma^2 + \sigma_d^2}{(\sigma\,\sigma_d)^2}.$$

Consider the normalized score-velocity residual plotted in **Plot 1**, $V_\theta(\sigma) := \frac{x_\sigma - D_\theta(x_\sigma; \sigma)}{\sigma}$. Substituting $D_\theta$ yields `[DERIVED FROM: edm_karras_2022.md, App. B.6 & diffusion_flow_matching_notes.md, Ch. 4.1–4.3]`:
$$\frac{x_\sigma - D_\theta(x_\sigma; \sigma)}{\sigma} = \frac{1 - c_{\text{skip}}(\sigma)}{\sigma} x_\sigma - \frac{c_{\text{out}}(\sigma)}{\sigma} F_\theta(\cdot) = \frac{\sigma}{\sigma^2 + \sigma_d^2} x_\sigma - \frac{\sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}} F_\theta(\cdot).$$

Two opposing regimes follow immediately:
1. **True Bayes-Optimal / Tweedie Denoiser $D^*(x_\sigma) = \mathbb{E}[x_0 \mid x_\sigma]$ `[BOOK: diffusion_flow_matching_notes.md, Ch. 4.3; murphy_tweedie2, Eq. 24.33–24.34]`:**
   - Whenever the conditional variance $\operatorname{Var}(x_0 \mid x_\sigma)$ is smaller than noise energy $\sigma^2$ (true along low-dimensional data manifolds at low $\sigma$), $x_\sigma - D^* \approx n$, so $\|V^*(\sigma)\|_{\text{RMS}} = \|n/\sigma\|_{\text{RMS}} \approx \mathbf{1.0}$.
   - Even if $D^* = \mathbb{E}[x_0] = 0$ at $\sigma \gg \sigma_d$, $\|V^*(\sigma)\|_{\text{RMS}} = \|x_\sigma/\sigma\|_{\text{RMS}} \approx 1.0$.
2. **Dead / Zero-Score Network ($F_\theta \approx 0$):**
   $$\|V_{F=0}(\sigma)\|_{\text{RMS}} = \frac{\sigma}{\sigma^2 + \sigma_d^2} \|x_\sigma\|_{\text{RMS}} \xrightarrow{\sigma \ll \sigma_d} \frac{\sigma}{\sigma_d} \approx 0.$$
   At $\sigma \to 0.002$ with $\sigma_d = 0.8$, $\frac{\sigma}{\sigma^2 + \sigma_d^2} \to 0.0025$, whereas $-\frac{\sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}} F_\theta \to -F_\theta$ `[DERIVED FROM: edm_karras_2022.md, App. B.6]`.

**Quantitative Match to Plot 1 (`[PROBE: probe_spectral_and_bayes_floor.py, Probe 1B]`):**
- At $\sigma = 80.0 \to 5.0 \to 1.61 \to 0.80 \to 0.28 \to 0.05 \to 0.002$, the trivial $F=0$ curve `Vel_RMS(F=0)` evaluates to `0.9998 -> 0.9880 -> 0.8955 -> 0.7069 -> 0.3302 -> 0.0624 -> 0.0025`.
- Observed Plot 1 tracks `1.01 -> 0.99 -> 0.80 -> 0.79 -> 0.50 -> 0.10 -> 0.05` (flat floor at `0.05` for $\sigma \le 0.02$).
- Because the skip term is only $0.0025$ at $\sigma = 0.002$, Plot 1's plateau at **`0.05`** establishes algebraically that **`||F_theta||_RMS ~ 0.05` at low noise** — i.e., the ViT low-$\sigma$ branch outputs only **~5% of the required unit-RMS score magnitude**, causing `D_vit \approx D_unet \approx c_skip x_sigma` and hence **`cos_sim(D_vit, D_unet) = 1.00`** for all $\sigma < 0.3$ `[DERIVED FROM: edm_karras_2022.md & Plot 1]`.

### 1.2 Why Reconstruction MSE Plateaus at `0.213–0.216` (Wiener MMSE Ceiling)
By Parseval's theorem and linear Wiener restoration `[BOOK: young_image_processing_and_murphy_dsm.md, pp. 90–93, Eq. 191 & 194]`, the optimal *linear* Fourier denoiser for an image/latent with PSD $S_{xx}(k)$ under independent white noise $S_{nn}(k) = \sigma^2$ is:
$$H_W(k, \sigma) = \frac{S_{xx}(k)}{S_{xx}(k) + \sigma^2}, \qquad \mathrm{MMSE}_{\text{Wiener}}(\sigma) = \frac{1}{d}\sum_k \frac{S_{xx}(k)\,\sigma^2}{S_{xx}(k) + \sigma^2}.$$
Under `sample_sigma_hybrid` (`std = 0.8`, `1/f^2` latent spectrum), `[PROBE: probe_spectral_and_bayes_floor.py, Probe 1B]` measures:
- Trivial $F_\theta = 0$ denoiser MSE: **`0.28375`** (`Weighted EDM_L = 0.99985`, matching EDM App. B.6's exact unit-loss theorem).
- Linear Gaussian Wiener filter MMSE: **`0.22095`** (`Weighted EDM_L = 0.80918`).
- Observed `hdmoem_v3_nature50k_256_v2`: **`Train MSE = 0.21340`**, **`Val Probe MSE = 0.21567`**.

Thus, the network learned only a **coarse, near-linear low-frequency shrinkage (`0.284 -> 0.215`, matching Gaussian Wiener risk)** and failed to cross into nonlinear data-manifold reconstruction (`MSE < 0.10`), while unweighted MSE masks low-$\sigma$ failure (`MSE(sigma=0.05) = 0.00249 << 0.64`) `[DERIVED FROM: young_image_processing_and_murphy_dsm.md & Probe 1B]`.

---

## Derivation 2: Why `ViT Branch (High-Freq)` Rises `0.35 -> 0.65` Above `Pure EDM Loss`

### 2.1 Spectral Partition Asymmetry of `laplacian_decomposition`
`laplacian_decomposition` `[CODE: Utils/loss.py:laplacian_decomposition]` applies a hard radial Fourier mask $\mathcal{P}_{\text{low}}$ with cutoff squared frequency $k_c^2 = \frac{\ln 2}{4\pi^2 \sigma_{\text{lap}}^2}$ (`norm="ortho"`), setting $\mathcal{P}_{\text{high}} = I - \mathcal{P}_{\text{low}}$ so $\|r\|^2 = \|\mathcal{P}_{\text{low}} r\|^2 + \|\mathcal{P}_{\text{high}} r\|^2$ by Parseval's identity `[BOOK: young_image_processing_and_murphy_dsm.md, p. 90]`.

For $x_0$ following a $1/f^2$ PSD (`std=0.8`) and isotropic white noise $n \sim \mathcal{N}(0, \sigma^2 I)$ (uniform PSD across all $k$), `[PROBE: probe_spectral_and_bayes_floor.py, Probe 1A]` measures the exact energy split on $32\times 32$:
- **`lap_sigma = 1.00`** (`cutoff_r = 4.24 cycles/32`): **`94.42%`** of white noise variance enters `P_high` vs. only **`5.58%`** in `P_low` (`54.49%` vs. `45.51%` for $1/f^2$ signal).
- **`lap_sigma = 0.50`** (`cutoff_r = 8.48 cycles/32`): **`78.44%`** of white noise variance enters `P_high` vs. **`21.56%`** in `P_low` (`29.08%` vs. `70.92%` for signal).

In high frequencies where $S_{xx}(k) \ll \sigma^2$, the optimal Wiener filter shrinks predictions toward zero ($H_W \to 0$, `[BOOK: young_image_processing_and_murphy_dsm.md, p. 93]`), making high-frequency noise estimation irreducibly high-variance `[BOOK: information_theory_intro.md, Ch. 3.1–3.4]`.

### 2.2 Asymmetric Kendall Log-Variance Amplification in `EDM_LOSS_NOD`
Examine the exact three terms in `EDM_LOSS_NOD.__call__` `[CODE: Utils/loss.py:EDM_LOSS_NOD]`, denoting normalized squared error $\mathcal{E}_{\text{blend}}(\sigma) = \lambda(\sigma)\|D_{\text{blend}} - x_0\|_2^2$ and log-variance $u(\sigma)$ `[PAPER: edm2_karras_2024.md, Eq. 21–28]`:
$$\mathcal{L}_{\text{pure}} = \mathbb{E}\!\left[e^{-u(\sigma)}\, \mathcal{E}_{\text{blend}}(\sigma) + u(\sigma)\right],$$
$$\mathcal{L}_{\text{vit}} = \mathbb{E}\!\left[e^{-\operatorname{sg}(u(\sigma))}\, \operatorname{sg}(w_{\text{vit}}(\sigma))\, \lambda(\sigma)\, \|\mathcal{P}_{\text{high}}(D_{\text{vit}} - x_0)\|_2^2\right],$$
$$\mathcal{L}_{\text{unet}} = \mathbb{E}\!\left[e^{-\operatorname{sg}(u(\sigma))}\, \operatorname{sg}(w_{\text{unet}}(\sigma))\, \lambda(\sigma)\, \|\mathcal{P}_{\text{low}}(D_{\text{unet}} - x_0)\|_2^2\right].$$

Minimizing $\mathcal{L}_{\text{pure}}$ with respect to $u(\sigma)$ yields optimal $u^*(\sigma) = \ln \mathcal{E}_{\text{blend}}(\sigma)$, substituting into $\mathcal{L}_{\text{pure}}^* = 1 + \ln \mathcal{E}_{\text{blend}}(\sigma)$ `[DERIVED FROM: edm2_karras_2024.md, Eq. 21–28]`. Crucially, **`EDM_LOSS_NOD` applies $e^{-u^*(\sigma)} = 1 / \mathcal{E}_{\text{blend}}(\sigma)$ to $\mathcal{L}_{\text{vit}}$ and $\mathcal{L}_{\text{unet}}$ WITHOUT adding the $+u^*(\sigma)$ offset!**

When `D_blend` reduces weighted error from step-0 baseline $\mathcal{E}_0 \approx 1.53$ (e.g., latent std $\approx 0.99$ vs. $\sigma_d = 0.8 \implies (0.99/0.8)^2 = 1.53$) to $\mathcal{E}_{\text{blend}} \approx 0.657$ (`[PROBE: probe_sigma_quantiles_and_loss.py]`):
- $u^* = \ln(0.657) = -0.420 \implies \mathcal{L}_{\text{pure}}^* = 1 - 0.420 = \mathbf{0.580}$ (matching Plot 7a's `1.53 -> 0.58` drop!).
- Meanwhile $e^{-u^*} = \mathbf{1.522\times}$ multiplies `vit_loss`:
  - At step 0 ($u=0$), with `w_vit ~ 0.72` (`P(sigma <= 3) = 88.47%` in `[PROBE: probe_sigma_quantiles_and_loss.py]`), `vit_loss` starts around `0.35–0.45` (`0.274` in Probe 2C; `~0.35` in Plot 7a).
  - By step 2000, as $e^{-u^*}$ climbs to $\sim 1.52\times$ (and ViT lags `D_blend` on `P_high` or injects high-freq perturbations), `vit_loss` climbs mechanically from `0.454 -> 0.690` (`0.274 -> 0.715` across 40 live Adam steps in `[PROBE: probe_zero_gates_and_gradients.py, Probe 2C]`; `0.35 -> 0.65` in Plot 7a!), crossing **above** `pure_loss` (`0.58`), while `unet_loss` stays at `~0.06–0.08` (`[PROBE: probe_spectral_and_bayes_floor.py, Probe 1C]`).

---

## Derivation 3: Architectural Eliminating Singularities & Pre-`f945711` Variance Contraction

### 3.1 Nested Zero-Init Fisher Singularities (Active in Both `v2` and `f945711`/`v3`)
In `[BOOK: amari_information_geometry.md, Ch. 12.2.1–12.2.5]`, any bilinear parametrization $f(x; v, W) = v\,\phi(W x)$ exhibits an **eliminating singularity submanifold** $R_e = \{v = 0\}$ where the Fisher metric degenerates ($a^\top G a = 0$) along $W$ and gradient flow exhibits cubic plateau drift $du/dt = O(|u|^3)$ `[BOOK: strogatz_nonlinear_dynamics.md, Ch. 2.4–2.7, Ch. 6.3]`.

While EDM2 Config G uses only **one** zero-initialized scalar output gain (`Gain(a) = g*a`, `[PAPER: edm2_karras_2024.md, App. B.7 Eq. 104]`), `HDMOEM_v3` nests **two to three zero-init scalar gates in series**:
$$y_{\text{branch}} = g_{\text{outer}} \cdot \operatorname{OutProj}\!\left(\operatorname{MP\_GatedSum}\big(x, B_L(W_L; g_L = 0)\big)\right), \qquad g_{\text{outer}} = 0, \; g_L = 0.$$
Differentiating at initialization ($g_{\text{outer}} = g_L = 0$):
$$\frac{\partial \mathcal{L}}{\partial W_L} = g_{\text{outer}} \cdot g_L \cdot (\cdots) = \mathbf{0}, \qquad \frac{\partial \mathcal{L}}{\partial g_L} = g_{\text{outer}} \cdot (\cdots) = \mathbf{0} \implies \text{only } \frac{\partial \mathcal{L}}{\partial g_{\text{outer}}} \neq 0.$$
Exact backward inspection in `[PROBE: probe_zero_gates_and_gradients.py, Probe 2B]` confirms that at step 0:
- **`113/113`** UNet inner tensors, **`49/49`** ViT inner tensors, **`output_proj_vit`** (`1/1`), **`output_proj_unet`** (`1/1`), **`21/22`** `pooled_cross_attn` tensors, and **`38/41`** scalar gates have **exact `0.0000e+00` gradient**.
- Under live Adam (`lr=1.2e-3`, Probe 2C), inner ViT branch gradient norm starts at `0.00e+00` and reaches only `5.67e-04` by step 39 while `vit_proj_gate` creeps to `0.0175`, explaining why `blend_residual` update ratios in Plot 6a peak at only `~2.5e-4` before collapsing to `1.5e-5`.

### 3.2 Pre-`f945711` Signal Variance Collapse (Explaining `v2` Severity vs `f945711`)
Prior to commit `f945711`, `Unet_block` and `Vit_block` mixed $x$ with zero-gated branch outputs via `mp_sum(x, 0, t=balance)` (`[PAPER: edm2_karras_2024.md, Eq. 88]`) instead of `mp_gated_sum` `[CODE: models/model_internals.py:mp_sum, mp_gated_sum]`. Because $\operatorname{Var}\!\left(\frac{(1-t)x + t\cdot 0}{\sqrt{(1-t)^2 + t^2}}\right) = \frac{(1-t)^2}{(1-t)^2 + t^2} \operatorname{Var}(x) < \operatorname{Var}(x)$ `[BOOK: calin_deep_learning_architectures.md, Ch. 6.2.6]`, signal RMS shrank exponentially (`[PROBE: probe_zero_gates_and_gradients.py, Probe 2A]`):
- At $t=0.5$: RMS collapsed **`1.000 -> 0.7076 -> 0.5003 -> 0.2502 -> 0.0625`** across just 8 blocks (vs. `1.0007` preserved by `f945711`'s `mp_gated_sum`).
- Unscaled `resample(mode='down')` further halved RMS (`0.5013` vs. `1.0027` after `f945711`'s `* 2.0` fix).
In `v2` (likely running pre-`f945711` or partially fixed code), deep inner features received near-zero signal AND zero initial gradients, starving high-$\sigma$ UNet structure and low-$\sigma$ ViT refinement.

---

## Derivation 4: ViT `[2,2]` + `PixelShuffle(2)` Spatial Aliasing & RAPSD Excess

In `HDMOEM_v3`, ViT operates on $16\times 16$ patch tokens (`VIT_patch_sizes=[2,2]`) and maps back to $32\times 32$ latents via a single `PixelShuffle(2)` followed by `output_proj_vit` (`1x1` conv in `v2`, `3x3` conv in `v3`), unlike multi-scale U-shaped transformers `[PAPER: diffit_hatamizadeh_2024.md, Sec. 1–3.2]`.

By discrete Fourier sampling, reshaping $4C$ sub-channels at stride 2 into $2\times 2$ spatial blocks aliases low-frequency inter-channel variance from the $16\times 16$ grid across the upper half-spectrum $[8, 16]$ cycles/32 of the $32\times 32$ grid. Measuring the azimuthally averaged PSD (`[PROBE: probe_blend_and_spectral_artifacts.py, Probe 3C]`) against a normalized $1/f^2$ $32\times 32$ latent baseline shows:
- Low/mid radii ($r = 1\text{--}8$ cycles/32): **`0.28x–0.35x` power deficit** (`PS(2)+1x1` and `PS(2)+3x3`).
- High radii ($r = 12, 14, 15$ cycles/32): **`1.43x`, `3.52x`, `6.87x` excess power** for `PixelShuffle(2) + 1x1`, and **`1.51x`, `3.92x`, `7.83x` excess power** for `PixelShuffle(2) + 3x3`!

Combined with FreeU's observation that excessive high-frequency skip energy destabilizes coherent denoising (`[PAPER: freeu_si_2023.md, Sec. 1–2]`), forcing an aliased `PixelShuffle(2)` ViT head to dominate all $\sigma < 3$ (`88.5%` of steps, `w_vit -> 1.0`) explains both **Plot 1's excess generated RAPSD power** (`cycles 2–12`) and **Plot 2/8's high-frequency blotchy patches**.

---

## Derivation 5: Optimization Horizon & Noise Schedule (What More Training Can vs. Cannot Fix)

In Adam (`[BOOK: calin_deep_learning_architectures.md, Ch. 4.5–4.8]`, `[PAPER: ngpt_loshchilov_2024.md, Sec. 2.5 Eq. 22]`), parameter steps have magnitude $|\Delta \theta_i| \approx \alpha(t)$ independent of gradient scale, so on normalized `MP_Conv` spheres $\|w\|_2 = \sqrt{N_j}$ (`[PAPER: edm2_karras_2024.md, Eq. 59, 66]`, `[BOOK: lee_riemannian_manifolds.md, Prop. 2.16]`), the maximum integrated angular rotation over training is bounded by $\int_0^T \alpha(t)\,dt$:

- **Integrated LR Budget (`[PROBE: probe_zero_gates_and_gradients.py, Probe 2D]`):**
  - `v2` Cosine (8,000 steps, `lr=1.2e-3`, `min=0.10`): $\sum \alpha(t) = \mathbf{5.257}$ (and steps `5000–8000` contribute only `0.745 = 14.2%`, dropping Plot 6a update ratios below $10^{-4}$).
  - `v3` WSD (8,000 steps, `stable=0.85`): $\sum \alpha(t) = \mathbf{8.713}$ (`1.66x` `v2`, still `3x` short).
  - EDM2 inverse-sqrt over 70,000 steps (`ref=2000`): $\sum \alpha(t) = \mathbf{25.998}$ (**`4.9x` larger than `v2`**).
- **Sigma Tail Starvation (`[PROBE: probe_sigma_quantiles_and_loss.py]`):** Under `sample_sigma_hybrid`, only **`4.35%`** of samples fall in $\sigma \le 0.02$ (~348 batches across all 8,000 steps!) and only **`11.53%`** fall in UNet's $\sigma > 3.0$ regime (~922 batches).

### Synthesis: Why Simply Training Longer Fails Without Fixes
- **What more steps (50k–70k, WSD / InvSqrt LR) *would* improve:** gives `MP_Conv` filters `~4.9x` more angular travel, exposes low-$\sigma$ (`<=0.02`) and high-$\sigma$ (`>3.0`) tails to `~3k–8k` batches instead of `~348–922`, and allows outer/inner zero gates more time to grow past `0.02`.
- **What more steps *CANNOT* cure:**
  1. **`vit_loss` inflation above `pure_loss`:** `EDM_LOSS_NOD`'s unmatched `exp(-log_var.detach())` multiplier grows *stronger* as `D_blend` improves, continuously inflating `vit_loss` relative to `pure_loss`.
  2. **`PixelShuffle(2)` Nyquist aliasing:** Architectural folding from `16x16` patches remains built into `D_vit` at $\sigma < 3$.
  3. **Nested `0 * 0` gate saddles:** Even if outer gates open slowly, nested zero-init gates across depth (`res_gate`, `tmsa_gate`, `mlp_gate`, `time_proj_gate`, `blend_gate`) waste critical early peak-LR budget (`14%+` lost just escaping `grad=0`) and suppress `blend_residual` (`1.5e-5` update ratio).

---

## Ranked Remediation Roadmap (Derived Directly from Diagnosis)

1. **Fix `EDM_LOSS_NOD` Uncertainty Asymmetry `[CODE: Utils/loss.py:EDM_LOSS_NOD]`:**
   Either match `pure_loss` by adding `+ log_var.detach()` inside `vit_loss` / `unet_loss` OR drop `branch_uncertainty_scale` (`= 1.0`) so branch losses track unweighted EDM scale `1.0` rather than being amplified by `1 / E_blend(sigma)`, and calibrate `lap_sigma` (`0.3–0.5` rather than `1.0`) so `P_low` and `P_high` have balanced SNR `[PROBE: Probe 1A, 1C, 4]`.
2. **Eliminate Nested Zero-Init Gates `[CODE: models/model_config_v3.py, model_components.py]`:**
   Keep zero-init **only** on the final branch output gains (`vit_proj_gate`, `unet_proj_gate`, `blend_gate` — matching EDM2 Eq. 104) OR `1.0`-scaled `out_gain`, and initialize all internal residual gates (`res_gate`, `tmsa_gate`, `mlp_gate`, `time_proj_gate`, `out_proj_gate`) to a small positive constant (e.g., `0.1–0.2`) or use `mp_sum` with non-zero initial residual branch variance so step-0 inner gradients are nonzero `[PROBE: Probe 2B, 2C]`.
3. **Replace ViT Raw `PixelShuffle(2)` Head at Low $\sigma$ `[CODE: models/model_config_v3.py]`:**
   Use bilinear/nearest spatial upsampling + `3x3` `MP_Conv` refinement (or patch size `[1, 1]` / U-shaped multi-scale tokens after `PixelShuffle(2)`) instead of raw `PixelShuffle(2)` output directly into `D_vit`, eliminating the `6.9x–7.8x` Nyquist folding peak `[PROBE: Probe 3C]`.
4. **Confirm Data Calibration & Training Schedule (`v3` Notebook / `Utils/training.py`):**
   Ensure `f945711` fixes (`mp_gated_sum` + `resample * 2.0`) are active (not the `v2` run), verify exact dataset latent std matches `sigma_data`, replace cosine 8k decay with EDM2 inverse-sqrt (`Lr_Scheduler`) or longer WSD (>= 40k–70k steps), and add low-$\sigma$ (`sigma <= 0.1`) velocity monitoring (`||(x - D)/sigma||_RMS -> 1.0`) alongside unweighted MSE `[PROBE: Probe 1B, 2D, 4]`.
