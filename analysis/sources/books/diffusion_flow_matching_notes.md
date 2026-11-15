# Book Summary: *The Math Behind Flow Matching and Diffusion Models*

- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\math behind flow matching and diffusion models.pdf`
- **Sections Read:** Ch. 3.3 (Continuity equation & Fokker-Planck equation), Ch. 4.1–4.3 (Marginal vs. conditional score matching, Gaussian score linearity)

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Continuity Equation & Fokker-Planck Dynamics (Ch. 3.3)
- Deterministic probability flow ODEs $\dot{x} = u_t(x)$ preserve probability mass via the continuity equation $\partial_t p_t + \nabla \cdot (p_t u_t) = 0$.
- Stochastic differential equations $dX_t = f_t(X_t)dt + g_t dW_t$ evolve densities under the **Fokker-Planck equation**, with score $\nabla_x \log p_t(x)$ linking the probability flow ODE velocity directly to forward transition noise.

### 1.2 Marginal vs. Conditional Score Matching (Ch. 4.1–4.3)
- Matching the intractable marginal score $\nabla_x \log p_t(x)$ over perturbed data $x_t \sim \mathcal{N}(x_0, \sigma^2 I)$ is mathematically equivalent up to an additive parameter-independent constant to **conditional score matching**:
  $$\mathbb{E}_{x_0, x_t}\left[\left\|s_\theta(x_t, \sigma) - \nabla_{x_t} \log p(x_t \mid x_0)\right\|_2^2\right],$$
  whose conditional score is linear in the added noise: $\nabla_{x_t}\log p(x_t \mid x_0) = -\frac{x_t - x_0}{\sigma^2}$.
- **Minimum MSE / Posterior Expectation Solution:** Because MSE is minimized by conditional expectation, the unique global minimizer for denoiser $D(x_\sigma; \sigma)$ is the **Bayes-optimal posterior mean** $\mathbb{E}[x_0 \mid x_\sigma]$, tied to the marginal score via Tweedie's relation:
  $$\nabla_{x_\sigma} \log p_\sigma(x_\sigma) = \frac{\mathbb{E}[x_0 \mid x_\sigma] - x_\sigma}{\sigma^2}.$$

---

## 2. Direct Relevance to `HDMOEM_v3`
- **Lipschitz / Normalized Score Velocity Diagnostics:** Plot 1 plots $\|(x_\sigma - D_\theta(x_\sigma; \sigma))/\sigma\|_{\text{RMS}}$. At high $\sigma \gg \sigma_{\text{data}}$, $D^*(x_\sigma) \approx \mathbb{E}[x_0] = 0$, so $(x_\sigma - D)/\sigma = (\sigma \epsilon + x_0)/\sigma \to \epsilon$ has expected RMS $\approx 1$. At low $\sigma \ll \sigma_{\text{data}}$, the **true posterior mean displacement** $x_0 - x_\sigma = -\sigma \epsilon$ also scales linearly with $\sigma$, so an accurate denoiser maintains $\|(x_\sigma - D)/\sigma\|_{\text{RMS}} = \|(x_\sigma - x_0 + x_0 - D)/\sigma\|_{\text{RMS}} \approx O(1)$ (specifically $\ge 1$, since any estimation error adds orthogonally to the irreducible noise residual!). Collapse of $\|(x_\sigma - D)/\sigma\|_{\text{RMS}} \to 0.05$ at $\sigma \le 0.02$ proves mathematically that $D_\theta(x_\sigma; \sigma) \approx x_\sigma$ identically (the skip connection $c_{\text{skip}} \to 1$ dominates and $F_\theta$ predicts zero score!).
- **Conditional Variance Floor:** Unconditional / noisy targets leave an irreducible Bayes MMSE equal to $\mathbb{E}[\operatorname{Var}(x_0 \mid x_\sigma, c)]$, which must be quantified numerically under `sample_sigma_hybrid` before interpreting `0.213–0.215` MSE!
