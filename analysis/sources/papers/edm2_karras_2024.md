# Paper Summary: *Analyzing and Improving the Training Dynamics of Diffusion Models* (EDM2)

- **Authors:** Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine (NVIDIA)
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\scientifc papers\EDM_2 Analyzing and Improving the Training Dynamics of Diffusion Models.pdf`
- **Pages Read:** Main paper pp. 1–9; Appendices B.1–B.7 & Config G architecture pp. 24–31.

---

## 1. Key Mathematical Concepts & Equations Extracted

### 1.1 Heteroscedastic Uncertainty Weighting (Main pp. 2–4; Eq. 21–28)
- To equalize gradient magnitudes across noise levels $\sigma$ without tuning manual schedules, EDM2 predicts a per-noise-level log-variance $u(\sigma)$ and minimizes:
  $$\mathcal{L}(D_\theta, u; \sigma) = \frac{e^{-u(\sigma)}}{\sigma_d^2 + \sigma^2 \dots} \dots \quad \text{specifically: } \mathbb{E}\left[\frac{1}{e^{u(\sigma)}} \|D_\theta(x+\epsilon;\sigma) - x\|_2^2 \cdot \lambda(\sigma) + u(\sigma)\right]$$
  (Eq. 21–28 form $e^{-u}\lambda(\sigma)\|D_\theta - x\|^2 + u$ normalized by $e^{u}$ so at optimum $e^{u^*(\sigma)} = \mathbb{E}[\lambda(\sigma)\|D_\theta - x\|^2]$, rendering expected weighted training loss constant $\approx 1$ across $\sigma$).

### 1.2 Magnitude-Preserving Architecture & Forced Weight Normalization (App. B.1–B.7, Eq. 46–104)
- **Unit-variance principle:** Every activation tensor and every convolutional/linear weight vector is designed to maintain expected unit per-channel second moment ($\mathbb{E}[a^2] = 1$) without LayerNorm/GroupNorm shifts.
- **`MP-Conv` with Forced Weight Normalization (Eq. 46–66):**
  - During every forward pass (and after parameter updates), each filter $w$ is projected/normalized onto the sphere $\|w\|_2 = \sqrt{N_j}$ (Eq. 66), with learnable scalar gains decoupled from filter direction.
  - Because $w / \|w\|_2$ is scale-invariant, the loss gradient $\nabla_w \mathcal{L}$ satisfies $\langle w, \nabla_w \mathcal{L} \rangle = 0$ (orthogonal projection onto the tangent space, Eq. 59).
  - Under Adam, gradient updates occur orthogonal to $w$, increasing $\|w\|^2$ monotonically by Pythagorean theorem if not re-normalized (hence forced re-normalization), and effective step size on the angular direction is controlled by $\alpha(t)$ relative to $\|w\| = \sqrt{N_j}$.
- **Magnitude-Preserving Primitives:**
  - **`MP-SiLU`** & **`MP-Fourier`**: Scaled to preserve unit output variance under standard normal inputs.
  - **`MP-Sum` (Eq. 88):** $\operatorname{MP\text{-}Sum}(a, b; t) = \frac{(1-t)a + t b}{\sqrt{(1-t)^2 + t^2}}$, preserving unit variance *assuming* $a$ and $b$ are independent and **both have unit variance**. If $b \equiv 0$ (e.g., from a zero-initialized gate), $\operatorname{Var}(\operatorname{MP\text{-}Sum}(a, 0; t)) = \frac{(1-t)^2}{(1-t)^2 + t^2} < 1$, **strictly shrinking signal variance!**
  - **`MP-Cat` (Eq. 103):** Re-scales concatenated tensors according to their blend parameter $t$ and channel counts so concatenated outputs preserve unit variance.
- **Zero-Initialized Final Output Gain Only (Eq. 104, p. 31):** EDM2 Config G employs a *single* learned scalar gain at the network output (`Gain(a) = g * a`) initialized near zero — **not** nested chains of zero-initialized scalar gates inside every residual/attention block.

### 1.3 Learning Rate Schedule, Training Budget & Post-Hoc EMA (Main pp. 5–8)
- **Inverse square-root LR schedule (Config G):** $\alpha(t) = \frac{\alpha_{\text{ref}}}{\sqrt{\max(t/t_{\text{ref}}, 1)}}$ (eliminates premature LR death inherent to short cosine schedules).
- **Optimal EMA length vs. LR:** EDM2 finds optimal relative EMA width scales inversely with step budget / learning rate ($\sigma_{\text{rel}} \propto 1/(\alpha_{\text{ref}}^2 t_{\text{ref}})$, p. 8) and uses multi-hundred-thousand-step / megabatches (e.g., 268M–2684M images in Fig. 5c) for ImageNet-512.

---

## 2. Direct Relevance to `HDMOEM_v3`
1. **Why `mp_sum` with `res_gate=0` failed pre-`f945711`:** Applying Eq. (88) `MP-Sum(x, 0, t)` across dozens of UNet and ViT blocks geometrically destroys forward activation variance ($\left(\frac{(1-t)^2}{(1-t)^2+t^2}\right)^{L/2} \ll 1$), explaining why `f945711` had to introduce `mp_gated_sum`.
2. **Deviation from Eq. (104):** `HDMOEM_v3` stacks **multiple** zero-initialized scalar gates in series (`res_gate`, `tmsa_gate`, `mlp_gate` $\to$ `vit_proj_gate` / `unet_proj_gate` $\to$ `blend_gate`), violating EDM2's single output `Gain` rule and creating multi-order eliminating singularities.
3. **8,000 steps + Cosine LR vs. EDM2 inverse-sqrt schedule:** Cosine LR to $10^{-4}$ over 8,000 steps freezes residual updates just as outer/inner gates finish opening.
