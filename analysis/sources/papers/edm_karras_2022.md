# Paper Summary: *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM)

- **Authors:** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (NVIDIA)
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\scientifc papers\Elucidating the Design Space of Diffusion-Based Generative Models.pdf`
- **Pages Read:** Main Section 5 (pp. 8–10); Appendix B.3, B.5–B.6 (Closed-form empirical denoiser, Fokker-Planck/heat PDE, derivation of $c_{\text{in}}, c_{\text{out}}, c_{\text{skip}}, \lambda(\sigma)$).

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Preconditioning Derivation for Unit Input/Target/Output Variance (App. B.6)
- Let clean data $y$ have variance $\sigma_{\text{data}}^2$ per coordinate and noisy sample $x = y + n$ with $n \sim \mathcal{N}(0, \sigma^2 I)$, parameterized via denoiser:
  $$D_\theta(x; \sigma) = c_{\text{skip}}(\sigma)\, x + c_{\text{out}}(\sigma)\, F_\theta\big(c_{\text{in}}(\sigma)\, x;\, c_{\text{noise}}(\sigma)\big).$$
- **Unit-variance input scaling:** Setting $\operatorname{Var}(c_{\text{in}}(\sigma) x) = 1$ yields:
  $$c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}.$$
- **Effective training target of $F_\theta$:** Rewriting $\lambda(\sigma)\|D_\theta - y\|_2^2 = \lambda(\sigma) c_{\text{out}}(\sigma)^2 \|F_\theta - F_{\text{target}}\|_2^2$ where $F_{\text{target}} = \frac{1}{c_{\text{out}}(\sigma)}(y - c_{\text{skip}}(\sigma)x)$.
- **Minimizing $F_{\text{target}}$ variance** with respect to $c_{\text{skip}}(\sigma)$ yields the optimal Wiener-like linear skip coefficient:
  $$c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2},$$
  and forcing unit target variance ($\operatorname{Var}(F_{\text{target}}) = 1$) gives:
  $$c_{\text{out}}(\sigma) = \frac{\sigma \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}.$$
- **Unit loss normalization:** Choosing $\lambda(\sigma) = \frac{1}{c_{\text{out}}(\sigma)^2} = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \sigma_{\text{data}})^2}$ ensures that a trivial network predicting $F_\theta \equiv 0$ achieves **exact expected weighted loss equal to $1.0$** across all $\sigma$ when $\operatorname{Var}(y) = \sigma_{\text{data}}^2$!
  $$\mathbb{E}[\lambda(\sigma)\|c_{\text{skip}}(\sigma)(y+n) - y\|_2^2] = 1.$$

### 1.2 Empirical Closed-Form Optimal Denoiser & Low-$\sigma$ Velocity (Sec. 5, App. B.3, B.5)
- For empirical dataset $\{y_i\}_{i=1}^N$, the exact MMSE denoiser is the Boltzmann-weighted data mixture $D^*(x; \sigma) = \frac{\sum_i \mathcal{N}(x; y_i, \sigma^2 I)\, y_i}{\sum_i \mathcal{N}(x; y_i, \sigma^2 I)}$.
- Across $\sigma \in [\sigma_{\min}, \sigma_{\max}]$:
  - At high $\sigma \gg \sigma_{\text{data}}$, $D^*(x; \sigma) \approx \mathbb{E}[y]$, so $(x - D^*)/\sigma \approx (n + y - \mathbb{E}[y])/\sigma \implies$ normalized velocity RMS $\approx 1$.
  - At intermediate/low $\sigma$, the denoiser must cancel $(1 - c_{\text{skip}})x + c_{\text{out}}F_\theta \approx y$. Specifically, since $c_{\text{skip}}(\sigma) \to 1$ as $\sigma \to 0$, $x - c_{\text{skip}}(\sigma)x = \frac{\sigma^2}{\sigma^2 + \sigma_d^2}x = O(\sigma^2)$, while the true denoising displacement is $x - y = n = O(\sigma)$! Thus **the $c_{\text{out}}(\sigma)F_\theta \approx -n$ term carries almost 100% of the velocity $(x - D)/\sigma = n/\sigma \implies \text{RMS} = 1$ at low $\sigma$!** If $F_\theta \to 0$ at low $\sigma$, then $(x - D)/\sigma \to \frac{\sigma}{\sigma^2+\sigma_d^2} x \to 0$!

---

## 2. Direct Relevance to `HDMOEM_v3`
- **Explains Plot 1 (Lipschitz Velocity Collapse to $0.05$ as $\sigma \to 0.002$):** By EDM's exact formula $(x - c_{\text{skip}}(\sigma) x)/\sigma = \frac{\sigma}{\sigma^2 + \sigma_d^2} x \to 0$ as $\sigma \to 0$, seeing $\|(x - D_x)/\sigma\|_{\text{RMS}} \to 0.05$ proves algebraically that $F_\theta$ is nearly silent at low $\sigma$, forcing `D_x` to degenerate to the pure skip path $c_{\text{skip}}(\sigma)x$ (which also forces `cos_sim(D_vit, D_unet) -> 1.00` because both equal $c_{\text{skip}}(\sigma)x$!).
- **Provides the exact $F_\theta = 0$ Baseline MSE & Weighted Loss:** Under `EDM_LOSS_NOD`, $F_\theta = 0$ already achieves unit expected weighted loss ($1.0$) before any learning, giving an exact quantitative benchmark for `Train MSE = 0.2134` / `Val MSE = 0.2157`.
