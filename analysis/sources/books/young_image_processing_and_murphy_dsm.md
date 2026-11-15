# Reference Summary: Image Restoration (Wiener Filter) & Denoising Score Matching

1. **Book A:** *Image Processing Fundamentals* (`C:\Users\Maha Mamdouh\Desktop\ML learning phase\image processing guide.pdf`), pp. 90–94 (Sections 10.2–10.2.3).
2. **Book B:** *Probabilistic Machine Learning: Advanced Topics*, Kevin P. Murphy (`C:\Users\Maha Mamdouh\Desktop\ML learning phase\Probabilistic ML advanced topics.pdf`), pp. 889–893 (book pp. 855–859, Section 24.3: Score Matching & Denoising Score Matching).

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Wiener MMSE Filter & Parseval Energy Equivalence (*Image Processing Fundamentals*, pp. 90–93)
- **MSE & Parseval's Theorem (p. 90, Eq. 191):** MSE corresponds directly to total error signal energy and preserves equivalence between spatial and Fourier domains via Parseval's theorem.
- **Fourier-Domain Wiener Optimal Filter (pp. 91–92, Eq. 194):** For an image $a[m,n]$ with power spectral density $S_{aa}(u,v)$ corrupted by independent additive noise with PSD $S_{nn}(u,v)$, the minimum MSE (and minimum RMS) linear estimator is:
  $$H_W(u,v) = \frac{S_{aa}(u,v)}{S_{aa}(u,v) + S_{nn}(u,v)}.$$
- **Asymptotic Signal-vs-Noise Regimes (p. 93):**
  - Low-frequency / signal-dominated band ($S_{aa}(u,v) \gg S_{nn}(u,v)$): $H_W(u,v) \to 1$ (signal passes nearly unattenuated).
  - High-frequency / noise-dominated band ($S_{aa}(u,v) \ll S_{nn}(u,v)$): $H_W(u,v) \to \frac{S_{aa}(u,v)}{S_{nn}(u,v)} \ll 1$ (heavy shrinkage; noise cannot be recovered without distorting signal).
- **Laplacian High-Frequency Edge Extraction (p. 90, Eq. 192):** The discrete Laplacian isolates high-frequency edge variations ($\hat{a} = a - k\nabla^2 a$).

### 1.2 Denoising Score Matching (DSM), Tweedie's Formula & Small-$\sigma$ Variance Blowup (Murphy, pp. 891–892)
- **DSM Equivalence (p. 891, Eq. 24.31–24.34):** Minimizing Fisher divergence to the smoothed density $q(\tilde{x})$ matches the score $s_\theta(\tilde{x})$ to conditional score $-\frac{\tilde{x} - x}{\sigma^2}$, traceable directly to **Tweedie's formula** and Stein's unbiased risk estimation (SURE).
- **Variance Explosion at Small $\sigma$ (p. 892, Eq. 24.35–24.36):** Expanding the unweighted DSM Monte Carlo estimator near $\sigma \to 0$ yields terms scaling as $\frac{2}{\sigma} z^\top \nabla_x \log p_\theta(x) + \frac{\|z\|_2^2}{\sigma^2}$, whose Monte Carlo variance grows unboundedly ($O(1/\sigma^2)$) unless weighted precisely (as in EDM's $\lambda(\sigma)$ and $c_{\text{out}}(\sigma)$ preconditioning).

---

## 2. Direct Relevance to `HDMOEM_v3`
- **Analytical Bayes / Wiener Risk Floor:** Using the power spectrum $S_{aa}(k)$ of natural images/VAE latents and isotropic white noise spectrum $S_{nn}(k) = \sigma^2$, the Wiener filter gives an exact closed-form upper bound on linear MMSE distortion per frequency band and noise level $\sigma$. Specifically, in `EDM_LOSS_NOD`, `vit_loss` evaluates **high-frequency residuals** $\mathcal{P}_{\text{high}}(D_{\text{vit}} - x)$ multiplied by $\lambda(\sigma) = \frac{\sigma^2 + \sigma_d^2}{\sigma^2 \sigma_d^2}$: at moderate-to-high $\sigma$, $\mathcal{P}_{\text{high}}(x)$ has tiny signal energy while $\mathcal{P}_{\text{high}}(\tilde{x})$ carries the bulk of additive white noise, forcing any denoiser toward zero output ($H_W \to 0$) in high frequencies and creating an irreducible baseline MSE even for the Bayes-optimal estimator!
