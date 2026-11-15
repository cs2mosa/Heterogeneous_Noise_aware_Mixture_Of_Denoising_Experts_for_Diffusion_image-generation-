# Paper Summary: *DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers*

- **Authors:** Minglei Shi, Ziyang Yuan, Haotian Yang, Xintao Wang, Mingwu Zheng, Xin Tao, Wenliang Zhao, Wenzmin Wang, Jie Zhou, Li Yuan, Pengfei Wan
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\scientifc papers\DiffMoE.pdf`
- **Sections Read:** Main Sections 1–2.2 (Batch-level global token pool & capacity-aware dynamic routing).

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Batch-Level Global Token Pool Routing vs. Fixed Per-Sample Allocation (Sec. 1–2.2)
- Standard token-level MoE enforces fixed capacity per sample or uniform expert assignment, ignoring that different noise levels $t$ and spatial tokens have vastly different complexity.
- **DiffMoE** introduces a **batch-level global token pool** where all tokens across a batch (containing heterogeneous noise levels and spatial regions) compete dynamically under capacity predictors, allowing complex/noisy samples or regions to allocate more compute than trivial steps without expert starvation.
- DiffMoE relies on standard DiT residual paths (non-zero initialized transformer blocks) and unified diffusion loss ($\epsilon$ or $v$-prediction) without frequency splitting or nested zero-init branch gates.

---

## 2. Direct Relevance to `HDMOEM_v3`
- **Contrast with `ParametricTrajectoryRouter` + `PathPriorLoss`:** `HDMOEM_v3` performs **macro-level, deterministic/soft trajectory routing strictly as a function of $\ln \sigma$** (`c_noise`), selecting `top_k=1` out of 2 experts inside UNet and ViT banks (verified healthy in Plot 5: entropy `~0.95–1.0`, Gini `< 0.25`, `0` dead experts, clean specialization across $\sigma$).
- Therefore, DiffMoE's failure mode (expert collapse / token starvation) is **explicitly NOT occurring** in `HDMOEM_v3`; the failure lies elsewhere (signal propagation, optimization horizon, loss decomposition, and low-$\sigma$ gradient/velocity collapse).
