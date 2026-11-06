# Information-Geometric & Mathematical Analysis of HDMOEM: Weaknesses, Failure Modes, and Formal Remedies

## 1. The Mathematical & Information-Geometric Lens on HDMOEM

To analyze **HDMOEM** rigorously, we view the diffusion process and the network's internal representations as objects on three coupled geometric spaces:

1. **The Statistical Manifold of Noisy Marginals $(\mathcal{M}, g_F, W_2)$**:
   In the Variance-Exploding (VE) / EDM formulation, corrupting clean latent data $x_0 \sim p_{\text{data}}$ with Gaussian noise of scale $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ defines a 1-parameter curve of probability distributions:
   $$p_\sigma(x) = \int p_{\text{data}}(x_0) \, \mathcal{N}(x \mid x_0, \sigma^2 I) \, dx_0$$
   Moving from $\sigma_{\max} \to \sigma_{\min}$ via the Probability Flow ODE ([`EDM_Sampler`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/EDM_sampler.py#L79-L97)):
   $$\frac{dx}{d\sigma} = -\sigma \nabla_x \log p_\sigma(x) = \frac{x - D^*(x, \sigma)}{\sigma}$$
   is a **Wasserstein-2 ($W_2$) gradient flow** transporting the high-entropy Gaussian prior toward the low-dimensional data manifold $\mathcal{M}_{\text{data}}$. By **Tweedie's Formula**, the optimal denoiser $D^*(x, \sigma) = \mathbb{E}[x_0 \mid x, \sigma]$ is an affine reparameterization of the **Fisher score vector** $s^*(x, \sigma) = \nabla_x \log p_\sigma(x)$:
   $$D^*(x, \sigma) = x + \sigma^2 \nabla_x \log p_\sigma(x)$$

2. **Dual Affine Connections ($m$-Geodesics vs. $e$-Geodesics) in Mixture Models**:
   In Amari's Information Geometry, combining probability distributions on $\mathcal{M}$ admits two dual projections:
   * **$m$-Geodesic (Mixture / Linear Pooling)**: $p^{(m)}(x_0 \mid x) = \sum_k w_k p_k(x_0 \mid x)$, under which the posterior mean is linear: $\mathbb{E}_{p^{(m)}}[x_0 \mid x] = \sum_k w_k D_k(x, \sigma)$.
   * **$e$-Geodesic (Exponential / Log-Linear / Product-of-Experts Pooling)**: $p^{(e)}(x_0 \mid x) \propto \prod_k p_k(x_0 \mid x)^{w_k}$, under which **scores and tangent vectors** add linearly: $\nabla_{x_0} \log p^{(e)}(x_0 \mid x) = \sum_k w_k \nabla_{x_0} \log p_k(x_0 \mid x)$.

3. **The EDM2 Riemannian Hypersphere $(\mathbb{S}^{d-1})$ of Internal Activations and Weights**:
   Following Karras et al. (EDM2), [`models/model_internals.py`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py) constrains weights $w \in \mathbb{S}^{d-1}$ and feature activations $h \in \mathbb{S}^{C-1}$ to have unit RMS variance ($\frac{1}{d}\|v\|_2^2 = 1$). Every valid operation on this manifold must be an **isometry or variance-preserving spherical interpolation** so that the preconditioned network output $F_\theta(x, \sigma)$ satisfies $\operatorname{Var}[F_\theta(x, \sigma)] = 1$ for all $\sigma$.

---

## 2. Six Mathematical & Architectural Weaknesses (and Their Failure Modes)

When we inspect the codebase through these three geometric lenses, **six concrete mathematical vulnerabilities** emerge.

---

### Weakness 1: Euclidean $\ell_1$ Blending Violates EDM2 Spherical Variance Preservation ("The Handoff Variance Collapse")

#### Where it occurs in code
* **Inter-branch blending** in [`HDMOEM_v3.forward` (line 367)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L367):
  ```python
  out = w_vit * vit_out + w_unet * unet_out + attn_correction
  ```
  where $w_{\text{vit}} + w_{\text{unet}} = 1$ ([`compute_blend_weights` line 281](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L281)).
* **Intra-branch expert routing** in [`router_to_experts` (lines 24–43)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L24-L43):
  ```python
  denom = out_router.sum(dim=-1, keepdim=True)
  ...
  output[mask] += out_exp * weights[mask].view(-1, 1, 1, 1)
  return output / denom
  ```

#### Mathematical Diagnosis
In EDM, the output scaling factor $c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$ is derived under the strict precondition that the raw network output $F_\theta(x, \sigma)$ has **unit variance ($\operatorname{Var}[F_\theta] = 1$)** at every noise level $\sigma$.

Suppose $F_{\text{vit}}$ and $F_{\text{unet}}$ both have variance $\sigma_F^2 = 1$ and Pearson correlation $\rho(\sigma) = \frac{\operatorname{Cov}(F_{\text{vit}}, F_{\text{unet}})}{\sqrt{\operatorname{Var}(F_{\text{vit}})\operatorname{Var}(F_{\text{unet}})}} \in [-1, 1]$. Under the Euclidean convex combination $F_{\text{blend}} = w_{\text{vit}} F_{\text{vit}} + (1 - w_{\text{vit}}) F_{\text{unet}}$, the variance is:
$$\operatorname{Var}[F_{\text{blend}}] = w_{\text{vit}}^2 + (1 - w_{\text{vit}})^2 + 2\rho(\sigma) w_{\text{vit}}(1 - w_{\text{vit}}) = 1 - 2(1 - \rho(\sigma))\, w_{\text{vit}}(1 - w_{\text{vit}})$$
Notice the quadratic penalty proportional to the transition gate $g(\sigma) = w_{\text{vit}}(1 - w_{\text{vit}})$:
* If the two branches specialize in **orthogonal features** ($\rho \to 0$, which [`logger2.py` (line 428)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/graphs/logger2.py#L428) actively encourages by monitoring `cosine_sim` $\to 0$), then at the handoff $\sigma = \sigma_{\text{trans}}$ ($w_{\text{vit}} = w_{\text{unet}} = 0.5$):
  $$\operatorname{Var}[F_{\text{blend}}] = 0.5^2 + 0.5^2 = 0.50 \implies \operatorname{Std}[F_{\text{blend}}] = \frac{1}{\sqrt{2}} \approx 0.707$$
* The exact same $50\%$ variance drop happens **inside** [`router_to_experts`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L24-L43) whenever two experts in the same branch have equal routing probability $p_1 = p_2 = 0.5$, because `out_router` is normalized by its $\ell_1$ sum ($\sum_i p_i = 1$) rather than its $\ell_2$ norm ($\sqrt{\sum_i p_i^2}$)! (Compare this to [`model_config2.py` (line 17)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config2.py#L17), which previously used $\ell_2$ normalization `denom = torch.sqrt((out_router**2).sum(dim=-1, keepdim=True) + 1e-8)`, before `model_config_v3.py` reverted to $\ell_1$ sum!)

#### Resulting Failure Mode
1. **Mid-Trajectory Signal Attenuation**: Whenever the ODE solver passes through an expert handoff or the branch transition $\sigma_{\text{trans}}$, the magnitude of the predicted update $c_{\text{out}}(\sigma) F_\theta(x, \sigma)$ shrinks by up to $\sim 30\%$ per handoff (and up to $50\%$ if both intra-branch and inter-branch handoffs coincide).
2. **Compensatory Weight Distortion**: To fight this variance dip in the MSE loss, `blend_residual` is incentivized to push $w_{\text{vit}}$ away from $0.5$ (making the transition overly sharp/binary), or the experts are forced to become **positively correlated ($\rho \to 1$)** so that $\operatorname{Var}[F_{\text{blend}}] \to 1$—which directly causes **branch collapse**!

---

### Weakness 2: The Homogenization Paradox in `EDM_LOSS_NOD` (Multi-Task Target Conflict)

#### Where it occurs in code
* [`EDM_LOSS_NOD.__call__` (lines 116–140)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/loss.py#L116-L140):
  ```python
  diff_blend = out_model["denoised"] - x
  pure_loss = torch.mean(weight * (diff_blend ** 2))
  unet_sq_error = (out_model['unet_denoised'] - x) ** 2
  unet_loss = torch.mean(weight * wx_unet * unet_sq_error)
  vit_sq_error = (out_model['vit_denoised'] - x) ** 2
  vit_loss = torch.mean(weight * wx_vit * vit_sq_error)
  trajectory_loss = (
      self.alpha_blend * pure_loss +
      self.alpha_unet * unet_loss +
      self.alpha_vit * vit_loss
  )
  ```
  *(Note also a runtime bug on [line 110 of `Utils/loss.py`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/loss.py#L110): `self._compute_trajectory_masks(sigma)` is called, but `_compute_trajectory_masks` is not defined on `EDM_LOSS_NOD`.)*

#### Mathematical Diagnosis
Let $D_{\text{vit}}(x), D_{\text{unet}}(x)$ be the two branch predictions, and let $D(x) = w_{\text{vit}} D_{\text{vit}}(x) + w_{\text{unet}} D_{\text{unet}}(x) + \Delta_{\text{attn}}(x)$ where $\Delta_{\text{attn}} = c_{\text{out}} F_{\text{attn}}$.
Consider the pointwise risk at noise level $\sigma$ in `EDM_LOSS_NOD`, taking expectation over $p(x_0 \mid x, \sigma)$. By the bias-variance decomposition $\mathbb{E}[\|D - x_0\|^2 \mid x] = \|D - \mu^*(x)\|^2 + \operatorname{Tr}(\Sigma_{x_0 \mid x})$ where $\mu^*(x) = \mathbb{E}[x_0 \mid x, \sigma]$ is the true posterior mean (Tweedie's formula), the excess functional risk is:
$$\mathcal{R}(D_{\text{unet}}, D_{\text{vit}}, \Delta_{\text{attn}}) = \alpha_{\text{blend}} \big\| w_{\text{vit}} D_{\text{vit}} + w_{\text{unet}} D_{\text{unet}} + \Delta_{\text{attn}} - \mu^*(x) \big\|^2 + \alpha_{\text{unet}} w_{\text{unet}} \big\| D_{\text{unet}} - \mu^*(x) \big\|^2 + \alpha_{\text{vit}} w_{\text{vit}} \big\| D_{\text{vit}} - \mu^*(x) \big\|^2$$

Taking the Fréchet derivatives with respect to $D_{\text{unet}}, D_{\text{vit}}, \Delta_{\text{attn}}$ for any $\sigma$ in the overlap region ($w_{\text{vit}}, w_{\text{unet}} > 0$) and setting them to zero yields the **unique stationary point**:
$$D_{\text{unet}}^*(x, \sigma) = \mu^*(x), \qquad D_{\text{vit}}^*(x, \sigma) = \mu^*(x), \qquad \Delta_{\text{attn}}^*(x, \sigma) = 0$$

Look at what this mathematical identity proves:
1. **`EDM_LOSS_NOD` actively punishes branch specialization in the overlap zone**: Because both `unet_loss` and `vit_loss` use the **same unsplit clean target $x$**, the loss function strictly penalizes $D_{\text{unet}} \neq D_{\text{vit}}$. In other words, `EDM_LOSS_NOD` forces both branches to learn the **exact same function** $\mu^*(x) = \mathbb{E}[x_0 \mid x, \sigma]$ wherever $w_{\text{unet}}, w_{\text{vit}} > 0$!
2. **It starves `PooledCrossAttn` ($\Delta_{\text{attn}}$) of residual signal**: Because $w_{\text{vit}} D_{\text{vit}}^* + w_{\text{unet}} D_{\text{unet}}^* = (w_{\text{vit}} + w_{\text{unet}})\mu^*(x) = \mu^*(x)$, the optimal cross-attention correction $\Delta_{\text{attn}}^*$ at the global minimum of `EDM_LOSS_NOD` is **identically zero**!
3. **Asymmetric Uncertainty Scaling (`log_var`)**: In [`EDM_LOSS_NOD` (lines 118–133)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/loss.py#L118-L133), `pure_loss` is weighted by $e^{-\text{log\_var}(\sigma)}$, while `unet_loss` and `vit_loss` are **not**. Since EDM's learned `log_var` becomes large and negative at low $\sigma$ and positive at high $\sigma$, the effective ratio $\frac{\alpha_{\text{blend}} e^{-\text{log\_var}}}{\alpha_{\text{unet}}}$ varies by orders of magnitude across $\sigma$, distorting the multi-task balance.

#### Resulting Failure Mode
* **Branch Homogenization**: In the overlap region, `l2_divergence` between ViT and U-Net is driven down by the loss itself, wasting the capacity of having two distinct architectures.
* **Dead Cross-Attention Module**: Even if `PooledCrossAttn` has non-zero gradients initially, as $D_{\text{vit}}$ and $D_{\text{unet}}$ converge toward $\mu^*(x)$, `PooledCrossAttn` is pushed toward zero rather than learning a meaningful structural-to-textural bridge.

---

### Weakness 3: Double-Zero-Initialization Saddle Point & The Hypersphere Projection Singularity

#### Where it occurs in code
1. **Series Zero-Init Deadlock in Cross-Attention**:
   * [`PooledCrossAttn.__init__` (line 514 of `models/model_internals.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L514):
     ```python
     nn.init.constant_(self.out_proj.weights, 0.0)
     ```
   * [`HDMOEM_v3.__init__` (line 260 of `models/model_config_v3.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L260):
     ```python
     nn.init.constant_(self.output_proj_attn.weights, 0.0)
     ```
2. **Singularity of `MP_Conv` at $W = 0$**:
   * [`MP_Conv.forward` (lines 280–286 of `models/model_internals.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L280-L286):
     ```python
     w = self.weights.to(torch.float32)
     if self.training:
         with torch.no_grad():
             self.weights.copy_(normalize(w))
     w = normalize(w)
     w = w * (gain / np.sqrt(w[0].numel()))
     ```
   Used with `nn.init.constant_(..., 0.0)` in `input_time_proj` ([line 152](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L152)), `output_proj_unet` ([line 225](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L225)), `output_proj_vit` ([line 226](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L226)), `blend_residual[-1]` ([line 234](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L234)), `Unet_block.conv_res2` ([line 278](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L278)), `Vit_block.linear3` ([line 575](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L575)), and `Vit_block.TMSA.out_proj` ([line 576](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L576)).

#### Mathematical Diagnosis
* **Part A — Exact Zero-Gradient Saddle Point in `PooledCrossAttn`**:
  In `HDMOEM_v3`, the cross-attention contribution is:
  $$\Delta_{\text{attn}} = W_{\text{output\_proj\_attn}} \Big( g(\sigma) \cdot W_{\text{out\_proj}} \, \text{MP\_Attention}(F_{\text{vit}}, F_{\text{unet}}) \Big)$$
  Because **both** $W_{\text{output\_proj\_attn}} = 0$ and $W_{\text{out\_proj}} = 0$ at initialization, the Jacobian of $\Delta_{\text{attn}}$ with respect to either weight matrix is proportional to the other weight matrix:
  $$\nabla_{W_{\text{output\_proj\_attn}}} \mathcal{L} \propto W_{\text{out\_proj}} = 0, \qquad \nabla_{W_{\text{out\_proj}}} \mathcal{L} \propto W_{\text{output\_proj\_attn}}^\top = 0, \qquad \nabla_{\theta_{\text{cross\_attn}}} \mathcal{L} \propto W_{\text{out\_proj}}^\top W_{\text{output\_proj\_attn}}^\top = 0$$
  Thus, **the entire `PooledCrossAttn` module in `HDMOEM_v3` is permanently deadlocked at $0$ for all training steps $t \ge 0$!** (Notice that [`tests/files/README.md` (lines 91–98)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/tests/files/README.md#L91-L98) explicitly documented this bug, but the fix was never applied to `models/model_internals.py` line 514 or `models/model_config_v3.py` line 226!)

* **Part B — The Radial Projection Singularity at the Origin of $\mathbb{R}^d$**:
  In EDM2, `MP_Conv` lives on the Riemannian manifold of row-wise unit-RMS matrices $(\mathbb{S}^{d-1})^{C_{\text{out}}}$, enforced via radial projection:
  $$\pi_\epsilon(w) = \frac{w}{\frac{1}{\sqrt{d}}\|w\|_2 + \epsilon}, \quad \epsilon = 10^{-4}$$
  The origin $w = 0$ is **not on the sphere $\mathbb{S}^{d-1}$**—it is the **topological singularity** of the radial projection $\pi_\epsilon$!
  Look at what happens when an `MP_Conv` weight is initialized to $w^{(0)} = 0$:
  1. At Step 0 forward pass, $\pi_\epsilon(0) = 0$. However, the Jacobian of $\pi_\epsilon(w)$ at $w = 0$ is:
     $$\left.\frac{\partial \pi_\epsilon(w)}{\partial w}\right|_{w=0} = \frac{1}{\epsilon} I = 10^{4} \, I$$
     **The backward pass through `w = normalize(w)` at $w=0$ amplifies gradients by a factor of $\frac{1}{\epsilon} = 10,000\times$!**
  2. At Step 0 optimizer step, Adam normalizes the gradient so every coordinate of $w^{(1)}$ becomes $\pm \text{lr}$. Thus $\frac{1}{\sqrt{d}}\|w^{(1)}\|_2 = \text{lr} \approx 10^{-3} \gg \epsilon$.
  3. At Step 1 forward pass, `self.weights.copy_(normalize(w))` projects $w^{(1)}$ onto the unit sphere:
     $$\frac{1}{\sqrt{d}}\|\pi_\epsilon(w^{(1)})\|_2 = \frac{\text{lr}}{\text{lr} + 10^{-4}} \approx 0.91 \approx 1.0$$
     **In a single optimizer step, every zero-initialized `MP_Conv` layer discontinuously jumps from 0.0 to ~91% of full unit-sphere magnitude**, pointing in the noisy direction of the very first mini-batch's gradient $\operatorname{sign}(-g_0)$!

#### Resulting Failure Mode
1. `PooledCrossAttn` never trains at all in the current code because two zero-init `MP_Conv` layers are stacked in series.
2. Every other zero-init `MP_Conv` layer (`input_time_proj`, `blend_residual[-1]`, `conv_res2`, `linear3`, `TMSA.out_proj`) experiences a $10^4\times$ gradient spike at step 0 followed by a discontinuous step-1 jump onto the unit sphere in the direction of a single noisy batch, causing severe early-training instability. (In EDM2, Karras et al. never zero-initialize `MP_Conv` weights; instead, they initialize `MP_Conv` weights from $\mathcal{N}(0, I)$ on the sphere and use a **learnable scalar gain parameter initialized to 0** or a small constant!)

---

### Weakness 4: Fisher-Rao / Quantile Distortion in the Tail of `ParametricTrajectoryRouter` & `MaskGenerator`

#### Where it occurs in code
* [`ParametricTrajectoryRouter.__init__` (lines 33–40 of `models/model_components.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L33-L40) and [`MaskGenerator.__init__` (lines 296–308 of `Utils/utils.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/utils.py#L296-L308):
  ```python
  log_min = math.log(min_sigma)
  log_max = math.log(max_sigma)
  log_equidistant = torch.linspace(log_min, log_max, steps=num_experts)
  cdf_points = 0.5 * (1.0 + torch.erf((log_equidistant - p_mean) / (p_std * math.sqrt(2))))
  ```

#### Mathematical Diagnosis
In Information Geometry, if we want $E$ experts to partition a branch's noise interval $[\sigma_{\text{low}}, \sigma_{\text{high}}]$ equally with respect to the training data density $p(\ln \sigma)$ (or the Fisher information metric along the noise trajectory), we must place expert centers **equidistantly in CDF (quantile) space**, NOT place them equidistantly in $\ln \sigma$ and then pass them through the nonlinear Gaussian CDF $\Phi\left(\frac{\ln \sigma - p_{\text{mean}}}{p_{\text{std}}}\right)$!

Look at what happens quantitatively with the config in [`Utils/configs.py`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/configs.py#L94-L105) (`p_mean = -0.2, p_std = 1.2`, U-Net range `[2.5, 80.0]`):
1. **Exponential Compression in the Gaussian Tail**:
   * At $\sigma = 2.5$: $\ln(2.5) = 0.916 \implies z = \frac{0.916 - (-0.2)}{1.2} = 0.930 \implies \Phi(z) = 0.8238$.
   * At $\sigma = 80.0$: $\ln(80.0) = 4.382 \implies z = \frac{4.382 - (-0.2)}{1.2} = 3.818 \implies \Phi(z) = 0.99993 \to \text{clamped to } 0.999$.
   * Because the derivative of the CDF is the Gaussian PDF $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$, which decays exponentially for $z > 1$, **any $\sigma \in [10.0, 80.0]$ maps to $\Phi(z) \in [0.979, 0.999]$—a width of only $0.02$ in CDF space!**
2. **Vanishing RBF Metric Sensitivity at High Noise**:
   In [`ParametricTrajectoryRouter.forward` (lines 67–71)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L67-L71):
   $$\text{logit}_i(\sigma) = -\gamma_i \big(\Phi(\sigma) - c_i\big)^2$$
   The effective sensitivity of the router to changes in $\ln \sigma$ is governed by the pullback metric:
   $$\left|\frac{\partial \text{logit}_i}{\partial \ln \sigma}\right| = \frac{2\gamma_i}{p_{\text{std}}} \big|\Phi(\sigma) - c_i\big| \, \phi\!\left(\frac{\ln \sigma - p_{\text{mean}}}{p_{\text{std}}}\right)$$
   At $\sigma \in [15, 80]$, $\phi(z) \in [10^{-4}, 10^{-2}]$, so $\frac{\partial \text{logit}_i}{\partial \ln \sigma} \approx 0$!
   With `init_sharpness` $\gamma = 25.0$ and U-Net centers separated by $\Delta c = 0.999 - 0.824 = 0.175$, the maximum logit difference between Expert 0 and Expert 1 across the entire $[2.5, 80]$ range is:
   $$\Delta \text{logit}_{\max} = 25.0 \times (0.175)^2 = 0.765 \implies \text{softmax}([0, -0.765]) = [0.682, 0.318]$$
   Without hard masks (i.e., **during inference in [`EDM_Sampler`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/EDM_sampler.py#L42-L43) where `Unet_router_mask=None`**), U-Net Expert 0 and Expert 1 never achieve sharp specialization—they remain blended at ~68% / 32% at the extremes and ~50% / 50% in the middle!
3. **Mismatch Between Router CDF and `sample_sigma_hybrid`**:
   In [`Utils/training.py` (line 258)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/training.py#L258), $\sigma$ is sampled from [`sample_sigma_hybrid`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/utils.py#L64-L85), which is a **mixture distribution**:
   $$p_{\text{train}}(\ln \sigma) = (1 - \alpha_{\text{ext}}) \, \mathcal{N}(\ln \sigma \mid p_{\text{mean}}, p_{\text{std}}^2) + \alpha_{\text{ext}} \, \mathcal{U}(\ln \sigma \mid \ln \sigma_{\min}, \ln \sigma_{\max}), \quad \alpha_{\text{ext}} = 0.2$$
   Yet `ParametricTrajectoryRouter`, `MaskGenerator`, and `PathPriorLoss` all use the **unmixed** Gaussian CDF ($0.8$ component only), ignoring the $20\%$ log-uniform component that dominates the high-noise tail!

#### Resulting Failure Mode
* **Train-Inference Mismatch**: During training, `MaskGenerator` forces hard binary masks (`-1e9` logits) so only 1 expert runs at high noise. At inference time (`mask=None`), the unmasked `ParametricTrajectoryRouter` has compressed CDF distances in the high-noise region ($\Delta \text{logit} \le 0.765$), causing **both U-Net experts to co-activate (~65/35) on inputs they were trained to handle in isolation**!

---

### Weakness 5: ODE Vector-Field Jacobian Spike at $\sigma_{\text{trans}}$ (Lipschitz Violation in Heun Integration)

#### Where it occurs in code
* [`HDMOEM_v3.compute_blend_weights` (lines 262–283 of `models/model_config_v3.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L262-L283) and [`Utils/configs.py` (lines 51–52)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/configs.py#L51-L52):
  ```python
  'transition_sigma': 2.5,
  'transition_sharpness': 0.3,
  ```

#### Mathematical Diagnosis
The Probability Flow ODE integrated by [`EDM_Sampler`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/EDM_sampler.py) is:
$$\frac{dx}{d\sigma} = \frac{x - D_\theta(x, \sigma)}{\sigma}$$
Differentiating the blended denoiser $D_\theta(x, \sigma) = w_{\text{vit}}(\sigma) D_{\text{vit}}(x, \sigma) + (1 - w_{\text{vit}}(\sigma)) D_{\text{unet}}(x, \sigma) + \Delta_{\text{attn}}(x, \sigma)$ with respect to $\ln \sigma$ gives:
$$\frac{\partial D_\theta}{\partial \ln \sigma} = \underbrace{\frac{\partial w_{\text{vit}}}{\partial \ln \sigma} \big(D_{\text{vit}}(x, \sigma) - D_{\text{unet}}(x, \sigma)\big)}_{\text{Handoff Transition Velocity}} + w_{\text{vit}} \frac{\partial D_{\text{vit}}}{\partial \ln \sigma} + w_{\text{unet}} \frac{\partial D_{\text{unet}}}{\partial \ln \sigma} + \frac{\partial \Delta_{\text{attn}}}{\partial \ln \sigma}$$
For $w_{\text{vit}}(\sigma) = \sigma_{\text{sigmoid}}\!\left(\frac{\ln \sigma_{\text{trans}} - \ln \sigma}{\tau} + r_\theta(\sigma)\right)$, at $\sigma = \sigma_{\text{trans}}$:
$$\left|\frac{\partial w_{\text{vit}}}{\partial \ln \sigma}\right| = \frac{1}{4}\left|\frac{1}{\tau} - \frac{\partial r_\theta}{\partial \ln \sigma}\right|$$
When $\tau = 0.3$ (as in `configs.py`), the transition from $w_{\text{vit}} = 0.1$ to $w_{\text{vit}} = 0.9$ happens over a narrow interval $\Delta(\ln \sigma) = 2 \tau \ln(9) \approx 1.32$ ($\sigma \in [1.29, 4.84]$). If `blend_residual` $r_\theta(\sigma)$ sharpens this further, $\left|\frac{\partial D_\theta}{\partial \sigma}\right|$ exhibits a sharp impulse at $\sigma_{\text{trans}}$ proportional to the inter-branch divergence $\|D_{\text{vit}} - D_{\text{unet}}\|_2$.

#### Resulting Failure Mode
* In a 30–50 step Heun solve ([`EDM_Sampler`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/EDM_sampler.py)), only 2–3 discrete solver steps land inside $\sigma \in [1.29, 4.84]$. Because Heun's method approximates the integral of $\frac{dx}{d\sigma}$ by a trapezoid between $(x_i, \sigma_i)$ and $(x_{i+1}, \sigma_{i+1})$, a steep sigmoid handoff creates **high second-order curvature $\frac{\partial^2 x}{\partial \sigma^2}$**, causing the trapezoidal corrector to overshoot and producing **structural ghosting or texture-layout misalignment** at the handoff step.

---

### Weakness 6: Permutation Equivariance & Self-Echo in `PooledCrossAttn`

#### Where it occurs in code
* [`PooledCrossAttn.forward` (lines 516–543 of `models/model_internals.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L516-L543) and [`MP_Attention.forward` (line 447)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L447).

#### Mathematical Diagnosis
1. **Zero Spatial / Geometric Inducement (Permutation Equivariance)**:
   In [`PooledCrossAttn.__init__`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L500-L507), `MP_Attention` is instantiated with `is_cross_attn=True` (so `rel_pos_bias` is disabled), and in `PooledCrossAttn.forward`, `unet_feat` and `vit_feat` are flattened directly into sequences `(B, H*W, C)` **without adding 2D positional encodings** (unlike [`SpatialCrossAttn`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L222), which explicitly calls `self.pos_encoder(x_seq, t=0.5)`).
   Consequently, for any spatial permutation matrix $\Pi \in \{0, 1\}^{(HW)\times(HW)}$:
   $$\text{CrossAttn}(Q_{\text{vit}}, \Pi K_{\text{unet}}, \Pi V_{\text{unet}}) = \text{CrossAttn}(Q_{\text{vit}}, K_{\text{unet}}, V_{\text{unet}})$$
   The cross-attention operation is **completely blind to 2D spatial coordinates**! A ViT token at top-left pixel $(0, 0)$ attends equally to a U-Net feature at bottom-right pixel $(63, 63)$ if their channel vectors happen to align.
2. **Unintended `vit_seq` Self-Echo via `MP_Attention` Residual**:
   Inside [`MP_Attention.forward` (line 447)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L447), the module returns:
   $$\text{MP\_Attention}(Q, K, V) = \operatorname{mp\_sum}(Q, \text{AttnOut}, t=\text{attn\_balance})$$
   With `attn_balance = 0.5`, $1/\sqrt{2} \approx 70.7\%$ of the output of `PooledCrossAttn.cross_attn` is just the **unmodified query `vit_seq` itself**, which is already present in `w_vit * vit_out`!
3. **Unpooled $\mathcal{O}((HW)^2)$ Memory Scaling**:
   Despite the class name `PooledCrossAttn` and its docstring describing $4\times 4$ spatial pooling, `forward` flattens the full $H\times W$ grid and uses `torch.matmul(q, k.transpose(-2, -1))` ([line 413](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L413)). At $64\times 64$ latent resolution ($HW=4096$, batch size $128$, $12$ heads), the attention matrix alone requires $128 \times 12 \times 4096^2 \times 4\text{ bytes} \approx \mathbf{103\text{ GB}}$ of VRAM!

---

## 3. Summary Matrix of Weaknesses & Failure Modes

| # | Subsystem | Mathematical / Geometric Flaw | Concrete Failure Mode |
| :--- | :--- | :--- | :--- |
| **1** | [`HDMOEM_v3.forward`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L367) & [`router_to_experts`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L24-L43) | Euclidean $\ell_1$ convex combination ($\sum w_i = 1$) instead of spherical variance-preserving combination on $\mathbb{S}^{d-1}$. | **Up to $50\%$ variance drop** ($\operatorname{Std} \to 0.707$) at $\sigma_{\text{trans}}$ and expert handoffs when branches are orthogonal ($\rho \to 0$), causing under-denoising or forcing branches to correlate (collapse). |
| **2** | [`EDM_LOSS_NOD`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/loss.py#L116-L140) | Both `unet_denoised` and `vit_denoised` are regressed against the **same full target $x$** in the overlap zone. | **Homogenization Paradox**: Unique minimizer is $D_{\text{unet}}^* = D_{\text{vit}}^* = \mathbb{E}[x_0\mid x]$ and $\Delta_{\text{attn}}^* = 0$, penalizing branch specialization and starving cross-attention of residual signal. |
| **3** | [`PooledCrossAttn`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L514) & [`MP_Conv`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L280-L286) | (a) Two zero-init layers in series (`out_proj` + `output_proj_attn`). (b) $W=0$ is a singularity of `normalize(w)` with Jacobian $10^4 I$. | (a) **`PooledCrossAttn` is permanently deadlocked at 0 gradient.** (b) Zero-init `MP_Conv` layers suffer a $10^4\times$ step-0 gradient spike and snap discontinuously from $0$ to unit norm at step 1. |
| **4** | [`ParametricTrajectoryRouter`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_components.py#L36-L40) & [`MaskGenerator`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/utils.py#L301-L305) | Log-spaced centers mapped through the saturating Gaussian tail $\operatorname{erf}(\cdot)$, ignoring the $20\%$ uniform mixture in `sample_sigma_hybrid`. | High-noise U-Net centers are crushed into $[0.824, 0.999]$; at inference (`mask=None`), router logits differ by $\le 0.765$, causing **unintended ~65/35 co-activation (train-test shift)**. |
| **5** | [`compute_blend_weights`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L262-L283) | Steep transition sharpness ($\tau = 0.3$) + unconstrained derivative of `blend_residual` across $\sigma_{\text{trans}}$. | **ODE Velocity / Curvature Spike** $\left\|\frac{\partial D_\theta}{\partial \sigma}\right\|_2 \propto \frac{1}{\tau}\|D_{\text{vit}} - D_{\text{unet}}\|_2$, breaking Heun's $\mathcal{O}(\Delta\sigma^3)$ local error bound at handoff. |
| **6** | [`PooledCrossAttn`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L524-L543) | Missing 2D positional encoding (permutation-equivariant), $70.7\%$ `vit_seq` self-echo inside `MP_Attention`, and unpooled $\mathcal{O}((HW)^2)$ matmul. | Spatial scrambling across distant patches, redundant ViT feature echo, and **OOM / quadratic bottleneck** at $64\times 64$ latent resolution. |

---

## 4. Formal Mathematical Remedies & Proofs of Improvement

Below are the exact mathematical formulations to resolve these six issues while **provably preserving or improving generation quality**.

### Remedy 1: Covariance-Aware Spherical Blending on the EDM2 Manifold
To preserve $\operatorname{Var}[F_\theta(x, \sigma)] = 1$ across all $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ regardless of whether the branches are correlated ($\rho \approx 1$) or orthogonal ($\rho \approx 0$):
1. **Inside [`router_to_experts`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L24-L43)**: Normalize the active expert combination by the empirical or $\ell_2$ norm of the routing weights, or apply `normalize()` before the output projection:
   $$F_{\text{branch}}(x) = \frac{\sum_{i=1}^E p_i(\sigma) F_{\text{expert}, i}(x)}{\sqrt{\sum_{i=1}^E p_i(\sigma)^2 + 2 \sum_{i < j} p_i(\sigma) p_j(\sigma) \hat{\rho}_{ij}}}$$
2. **At the Inter-Branch Handoff ([`HDMOEM_v3.forward`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_config_v3.py#L367))**:
   Given branch outputs $F_{\text{vit}}, F_{\text{unet}}$ with empirical per-sample spatial-channel cosine similarity $\hat{\rho} = \frac{\langle F_{\text{vit}}, F_{\text{unet}} \rangle}{\|F_{\text{vit}}\|_2 \|F_{\text{unet}}\|_2}$, define the **Riemannian / Covariance-Preserving Blend**:
   $$F_{\text{blend}} = \frac{w_{\text{vit}} F_{\text{vit}} + w_{\text{unet}} F_{\text{unet}}}{\sqrt{w_{\text{vit}}^2 + w_{\text{unet}}^2 + 2 \hat{\rho} \, w_{\text{vit}} w_{\text{unet}} + \epsilon}}$$
   * **Proof of Variance Preservation**:
     When $\operatorname{Var}(F_{\text{vit}}) = \operatorname{Var}(F_{\text{unet}}) = 1$ and $\operatorname{Corr}(F_{\text{vit}}, F_{\text{unet}}) = \hat{\rho}$, the variance of the numerator is $w_{\text{vit}}^2 + w_{\text{unet}}^2 + 2\hat{\rho} w_{\text{vit}} w_{\text{unet}}$. Dividing by its square root guarantees $\operatorname{Var}[F_{\text{blend}}] \equiv 1$ for **all** $w_{\text{vit}} \in [0, 1]$ and **all** $\hat{\rho} \in (-1, 1]$!
     * Special case $\hat{\rho} = 1$ (identical predictions): denominator $= \sqrt{(w_{\text{vit}} + w_{\text{unet}})^2} = 1$ (standard convex combination).
     * Special case $\hat{\rho} = 0$ (orthogonal predictions): denominator $= \sqrt{w_{\text{vit}}^2 + w_{\text{unet}}^2}$ (EDM2's exact [`mp_sum`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L50-L66) formula!).

---

### Remedy 2: Orthogonal / Residual Decomposition Loss (Replacing `EDM_LOSS_NOD`'s Redundant Penalty)
Instead of forcing both $D_{\text{unet}}$ and $D_{\text{vit}}$ to predict the full $x$ inside the overlap zone (which forces $D_{\text{unet}} = D_{\text{vit}}$ and kills `PooledCrossAttn`), we can formulate the loss so that **each branch is only supervised as a standalone full denoiser in its pure regime**, while in the **handoff zone**, the incoming branch is trained as an **orthogonal frequency/detail refinement** on top of the outgoing branch:

Let $P_{\text{low}}(\cdot)$ be a resolution-adaptive Gaussian low-pass filter (from [`laplacian_decomposition`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/loss.py#L8-L22) or FFT mask) and $P_{\text{high}}(\cdot) = I - P_{\text{low}}(\cdot)$ be its complementary high-pass projection ($P_{\text{low}} + P_{\text{high}} = I$).
Define the **Scale-Consistent Trajectory Loss**:
$$\mathcal{L}_{\text{SC}} = \lambda(\sigma) e^{-\hat{s}(\sigma)} \Big[ \big\| D(x, \sigma) - x_0 \big\|_2^2 + \alpha_{\text{unet}} w_{\text{unet}}(\sigma) \big\| P_{\text{low}}\big(D_{\text{unet}}(x, \sigma) - x_0\big) \big\|_2^2 + \alpha_{\text{vit}} w_{\text{vit}}(\sigma) \big\| P_{\text{high}}\big(D_{\text{vit}}(x, \sigma) - x_0\big) \big\|_2^2 \Big] + \hat{s}(\sigma)$$
where $\hat{s}(\sigma) = \text{log\_var}(\sigma)$ weights **all** terms consistently.

* **Proof of Non-Conflicting Optimality**:
  Because $P_{\text{low}}$ and $P_{\text{high}}$ project onto orthogonal subspaces in $L^2(\Omega)$ (or complementary spectral bands), minimizing the branch terms requires:
  $$P_{\text{low}} D_{\text{unet}}^*(x, \sigma) = P_{\text{low}} \mathbb{E}[x_0 \mid x, \sigma], \qquad P_{\text{high}} D_{\text{vit}}^*(x, \sigma) = P_{\text{high}} \mathbb{E}[x_0 \mid x, \sigma]$$
  Notice that:
  1. $D_{\text{unet}}^*$ and $D_{\text{vit}}^*$ are **no longer forced to be equal** in the overlap zone ($D_{\text{unet}}^*$ is free on the high-frequency subspace; $D_{\text{vit}}^*$ is free on the low-frequency subspace).
  2. At the transition $\sigma_{\text{trans}}$ ($w_{\text{unet}} = w_{\text{vit}} = 0.5$), the linear blend $0.5 D_{\text{unet}}^* + 0.5 D_{\text{vit}}^*$ has a **non-zero residual** $\mu^*(x) - (0.5 D_{\text{unet}}^* + 0.5 D_{\text{vit}}^*)$, which **directly provides a non-zero target for `PooledCrossAttn` ($\Delta_{\text{attn}}^*$) to learn**!

---

### Remedy 3: Eliminate the Double-Zero-Init Deadlock & Replace Zero-Weight `MP_Conv` with Scalar Gain Gating
1. **Fix the `PooledCrossAttn` Deadlock Immediately**:
   Remove `nn.init.constant_(self.out_proj.weights, 0.0)` from [`PooledCrossAttn.__init__` (line 514 of `models/model_internals.py`)](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L514) so that only `output_proj_attn` (or a scalar gate) is zero-initialized—never two layers in series!
2. **Replace `nn.init.constant_(mp_conv.weights, 0.0)` with Sphere-Initialized Weights + Zero-Initialized Scalar Gain**:
   For any `MP_Conv` layer that must output `0.0` at Step 0:
   * Keep `self.weights` initialized to $\mathcal{N}(0, I)$ (so `normalize(w)` is well-conditioned on the unit sphere $\mathbb{S}^{d-1}$ with Jacobian $\mathcal{O}(1)$ instead of $\frac{1}{\epsilon} = 10^4$).
   * Multiply the layer output by a learnable scalar parameter $\gamma_{\text{gate}} \in \mathbb{R}$ initialized to `0.0` (or pass `gain=self.gate_param` into `MP_Conv.forward(x, gain=self.gate_param)`):
     $$y = \gamma_{\text{gate}} \cdot \text{MP\_Conv}(x, \text{gain}=1.0), \quad \gamma_{\text{gate}}^{(0)} = 0$$
   * **Proof of Smooth Bootstrapping**:
     At Step 0, $y = 0$, so the EDM skip connection $D(x) = c_{\text{skip}} x$ is preserved exactly.
     In the backward pass at Step 0:
     $$\frac{\partial \mathcal{L}}{\partial \gamma_{\text{gate}}} = \left\langle \frac{\partial \mathcal{L}}{\partial y}, \text{MP\_Conv}(x) \right\rangle \neq 0$$
     Because $\text{MP\_Conv}(x)$ has unit variance ($\neq 0$), $\gamma_{\text{gate}}$ receives a clean $\mathcal{O}(1)$ gradient at Step 0 and grows smoothly ($\gamma_{\text{gate}}^{(1)} = \pm \text{lr} \approx 10^{-3}$ at Step 1, $2\times 10^{-3}$ at Step 2, etc.) **without snapping from 0 to 1 in a single step**!

---

### Remedy 4: Exact Mixture-CDF Quantile Seeding & Branch-Conditional Normalization in `ParametricTrajectoryRouter`
1. **Match the True Hybrid Sampling CDF**:
   Since training samples $\ln \sigma$ from the mixture [`sample_sigma_hybrid`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/Utils/utils.py#L64-L85), the true Cumulative Distribution Function $F_{\text{mix}}(\sigma)$ is:
   $$F_{\text{mix}}(\sigma) = (1 - \alpha_{\text{ext}}) \Phi\!\left(\frac{\ln \sigma - p_{\text{mean}}}{p_{\text{std}}}\right) + \alpha_{\text{ext}} \frac{\ln \sigma - \ln \sigma_{\min}}{\ln \sigma_{\max} - \ln \sigma_{\min}}$$
   Using $F_{\text{mix}}(\sigma)$ in `ParametricTrajectoryRouter`, `MaskGenerator`, and `PathPriorLoss` ensures that the CDF derivative $\frac{d F_{\text{mix}}}{d \ln \sigma} \ge \frac{\alpha_{\text{ext}}}{\ln(\sigma_{\max}/\sigma_{\min})} > 0$ **never vanishes in the high-noise tail**!
2. **Branch-Local CDF Normalization ($[0, 1]$ per Branch)**:
   For a branch operating on $[\sigma_{\text{low}}, \sigma_{\text{high}}]$, normalize its local percentile coordinate to $[0, 1]$:
   $$\tilde{u}(\sigma) = \operatorname{clamp}\!\left(\frac{F_{\text{mix}}(\sigma) - F_{\text{mix}}(\sigma_{\text{low}})}{F_{\text{mix}}(\sigma_{\text{high}}) - F_{\text{mix}}(\sigma_{\text{low}})}, 0, 1\right) \in [0, 1]$$
   and initialize the $E$ expert centers **uniformly in local quantile space**:
   $$c_i = \frac{i + 0.5}{E}, \quad i \in \{0, \dots, E-1\}$$
   * **Why this fixes the U-Net Router**: Every expert in both the U-Net and ViT branches now spans an identical width $\frac{1}{E}$ on $[0, 1]$ with equal training sample count, and the sharpness $\gamma_i = 25$ produces the exact same routing contrast ($\Delta \text{logit} = 6.25$ for $E=2$, i.e. $99.8\%$ confidence) in the U-Net branch as in the ViT branch—eliminating the train-test mask mismatch!

---

### Remedy 5: Spatially-Anchored, Pooled, Pure-Residual Cross-Attention
In [`PooledCrossAttn`](file:///f:/Heterogeneous-MOE-for-Diffusion-models/models/model_internals.py#L450-L543):
1. **Inject 2D Sinusoidal Positional Encodings (`MP_Pos_encoding_2D`)** into both `vit_seq` and `unet_seq` before computing queries and keys so that cross-attention is spatially anchored (breaking permutation equivariance).
2. **Restore Adaptive Spatial Pooling for $K, V$ (or $Q, K, V$)** when $H \times W > 16\times 16$ (e.g., pooling U-Net context to $16\times 16$ tokens while keeping ViT queries at full resolution, reducing cost from $\mathcal{O}((HW)^2)$ to $\mathcal{O}(HW \cdot P^2)$ while preserving 1:1 query resolution!).
3. **Remove the Redundant Query Residual in `PooledCrossAttn`**: Set `attn_balance=1.0` inside `PooledCrossAttn`'s `MP_Attention` call so that `PooledCrossAttn` outputs the **pure cross-attended U-Net context** rather than echoing $70.7\%$ of `vit_seq`.
