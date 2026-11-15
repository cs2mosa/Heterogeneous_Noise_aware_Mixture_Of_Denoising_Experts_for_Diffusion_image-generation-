# Paper Summary: *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*

- **Authors:** Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg (NVIDIA)
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\scientifc papers\NGPT (normalized transformers).pdf`
- **Sections Read:** Main Section 2 (pp. 1–6) & Appendix A.1–A.6 (pp. 10–15).

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Hyperspherical Representation Updates (SLERP / LERP on $\mathbb{S}^{d-1}$) (Sec. 2)
- Every hidden state $h$, embedding, and normalized linear block matrix row/column lies on the unit hypersphere $\mathbb{S}^{d-1}$.
- Residual updates move along the hypersphere using normalized convex / eigenlearning-rate combinations:
  $$h \leftarrow \operatorname{Norm}\big(h + \alpha_M (\operatorname{Norm}(h_M) - h)\big),$$
  approximating geodesic SLERP interpolation where step gain $\alpha$ directly controls angular displacement along $\mathbb{S}^{d-1}$ without norm drift or collapse.

### 1.2 Adam Effective Learning Rate vs. Parameter Magnitude (Sec. 2.5, Eq. 22; App. A)
- Under Adam, parameter increments satisfy $|\Delta \theta_i| \approx \alpha$ independently of current parameter scale $|\theta_i|$ (or initial scalar magnitude `s_init`).
- Consequently, for normalized weights or hyperspherically coupled scalars trained with Adam, the **effective relative/angular step size** scales inversely with the restored/stored scale magnitude (Eq. 22): initializing scalars with tiny `s_init` vs. `s_restore = 1` alters effective learning rate by factor `s_restore / s_init`, requiring careful `alpha_scale` decoupling (`sa,scale` trick) so small eigenlearning rates neither overshoot at early steps nor stall when global LR decays.
- Tangent-space Riemannian projection of gradients (Eq. 24–26) removes radial components orthogonal to the hypersphere, leaving pure angular steps on the manifold.

---

## 2. Direct Relevance to `HDMOEM_v3`
- **`normalize(x)` & `spherical_blend`:** `HDMOEM_v3` adopts nGPT/EDM2 hyperspherical activation normalization (`normalize`) and attempts a spherical interpolation (`spherical_blend(vit_out, unet_out, w_vit, w_unet)`). Checking whether `spherical_blend` follows true geodesic normalization ($\operatorname{Norm}$ / $1/\sin\theta$) or suffers chordal norm shrinkage when $D_{\text{vit}}$ and $D_{\text{unet}}$ diverge is critical.
- **Zero-init scalar gates under Adam:** Unlike nGPT's positive-initialized learnable step scales (`alpha_init = 1/L` scaled by `s_init/s_restore`), `HDMOEM_v3` initializes scalar gates at **exact 0** while using per-component LR multipliers (`0.1x–1.0x`), altering the effective Adam escape time relative to `MP_Conv` normalized weights ($\|w\|_2 = \sqrt{N_j}$).
