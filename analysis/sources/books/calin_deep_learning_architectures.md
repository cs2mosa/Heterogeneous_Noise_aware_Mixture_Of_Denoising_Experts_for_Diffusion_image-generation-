# Book Summary: *Deep Learning Architectures: A Mathematical Approach*

- **Author:** Ovidiu Calin
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\deep learning math books\Deep-Learning-Architectures.pdf`
- **Sections Read:** Ch. 4.5–4.8 (AdaGrad, RMSProp, Adam moments & step normalization), Ch. 6.2.6–6.3 (Vanishing gradients, plateaus, weight-initialization variance propagation)

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Adaptive Moment Estimation (AdaGrad, RMSProp, Adam) (Ch. 4.5–4.8)
- **AdaGrad & RMSProp:** Rescale coordinate-wise gradient steps inversely by accumulated or exponentially moving squared gradient magnitudes ($\sqrt{v_t} + \epsilon$).
- **Adam:** Tracks biased/bias-corrected first moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ and second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, updating parameters via:
  $$\Delta \theta_t = -\alpha_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$
- **Key scaling property:** For any coordinate whose gradient has consistent sign across steps ($|\hat{m}_t| / \sqrt{\hat{v}_t} \approx 1$), the step size is approximately $\pm \alpha_t$, largely **independent of both absolute gradient magnitude and current parameter magnitude $|\theta_t|$** (until gradients hit the numerical floor $\epsilon$ or vanish identically). Conversely, if gradients are zero identically along a nested zero-gate chain at step 0, $\hat{m}_0 = 0$ and weights remain frozen until outer gates open.

### 1.2 Vanishing Gradients, Plateaus, and Variance Propagation (Ch. 6.2.6–6.3)
- Forward and backward signal propagation across $L$ residual/feedforward layers preserves second moments only when layer gains satisfy variance conservation ($\prod_{l} \gamma_l^2 \approx 1$).
- **Variance contraction / vanishing gradients:** If every block multiplies residual features or skips by a scalar factor $< 1$ without unit-variance compensation, signal energy decays exponentially with depth ($c^L$), starving early layers of signal or shrinking backpropagated gradients.
- **Weight initialization & plateau formation:** Improper variance scaling across depth pushes activations or backpropagated Jacobians into low-curvature plateaus where optimization stagnates.

---

## 2. Direct Relevance to `HDMOEM_v3`
- **Pre-`f945711` `mp_sum` with `res_gate=0` (`mp_sum(x, branch * 0, t=balance)`):** Mixing a unit-variance signal $x$ with an identically zero branch via `mp_sum(x, 0, t)` multiplies the variance by $(1-t)^2 / [(1-t)^2 + t^2]$ or linear shrinkage factor $(1-t)$, **contracting signal variance across every UNet and ViT block** until `mp_gated_sum` was introduced in commit `f945711`.
- **Adam on scalar gates vs. normalized `MP_Conv` weights:** Adam applies steps of size $O(\alpha_t)$ regardless of weight norm $\|w\|_2$, creating drastically different angular update ratios for scalar gains vs. high-dimensional weights constrained to $\sqrt{N_j}\,\mathbb{S}^{N_j-1}$.
