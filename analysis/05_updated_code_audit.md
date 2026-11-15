# Second-Pass Mathematical & Architectural Audit (`text_guided` Branch, Post-`f945711`)

**Target Codebase:** `f:\Heterogeneous-MOE-for-Diffusion-models` (`text_guided` branch at commit `f945711`), auditing `models/model_internals.py`, `models/model_components.py`, `models/model_config_v3.py`, `Utils/loss.py`, `Utils/utils.py`, `Utils/configs.py`, `Utils/training.py`, `Utils/EDM_sampler.py`, `Utils/cdf_utils.py`, `Utils/generate_test.py`, `data_augmentation/data_collector.py`, `graphs/logger2.py`, and `graphs/plotter2.py`.

**Read-Only Diagnostic Probes & Test Verification:**
- [probe_post_fix_audit.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_post_fix_audit.py) (`PROBE A`–`PROBE E`, exit code `0`)
- [probe_post_fix_dynamics_and_sampler.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_post_fix_dynamics_and_sampler.py) (`PROBE 1`–`PROBE 3`, exit code `0`)
- [test_info_geom_fixes.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/tests/files/test_info_geom_fixes.py) (`23` unit tests passed in `7.810s`, exit code `0`)

---

## 1. Executive Summary

Relative to the baseline failure diagnosis in [04_mathematical_derivation_and_diagnosis.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/04_mathematical_derivation_and_diagnosis.md), commits `31d5375` and `f945711` **successfully fixed the four largest baseline failures**:
1. Internal residual variance collapse (`mp_gated_sum` + `res_gate=0.1`, `tmsa_gate=0.1`, `mlp_gate=0.1`, `resample(down)*2.0`),
2. ViT `PixelShuffle(2)` Nyquist spectral aliasing (`bilinear` + `unpatch_skip` + `unpatch_refine`),
3. `EDM_LOSS_NOD`'s `laplacian_decomposition` noise skew and `exp(-log_var.detach())` loss inflation, and
4. Premature 8k cosine LR starvation (replaced by 200k EDM2 inverse-sqrt `Lr_Scheduler`) plus mixture-CDF router quantile seeding (`Utils/cdf_utils.py`).

However, **five material mathematical, architectural, and training/inference bugs remain or were newly exposed** on `text_guided`:

| Rank | Severity | Component | Core Mechanism |
| :--- | :--- | :--- | :--- |
| **1** | **CRITICAL** | Outer Zero-Init Gates & Warmup (`model_config_v3.py`, `configs.py`) | `vit_proj_gate=0`, `unet_proj_gate=0`, `time_proj_gate=0`, `blend_gate=0`, `out_proj_gate=0` freeze `619+` parameter tensors at `grad=0` at step 0 (`100%` of experts, projections, and `blend_residual`), while `lr_gates=5e-4` + `rampup_Mimg=0.25` keeps inner expert gradients near/below Adam `eps=1e-8` early and caps `vit_g/unet_g` at `~0.012` after 250 steps. |
| **2** | **HIGH** | `Unet_router` + `MaskGenerator` (`configs.py`, `model_components.py`) | Under `p_mean=-0.2, p_std=1.2, transition_sigma=2.5`, ViT spans `79.37%` of $\sigma$ samples and UNet only `20.63%`; evaluating `Unet_router` across the whole batch gives `[0.8953, 0.1047]` usage (`ent_norm=0.4838 < 0.85`, `gini=0.3953 > 0.25`), starving `Unet_experts[1]` (`5x5`) by `8.55x` while running `Unet_experts[0]` on `79.4%` low-$\sigma$ samples where `grad_unet_exp` drops `16,600x`. |
| **3** | **HIGH** | Un-gated `Vit_block.cross_attn` & Unconditional Inference (`model_components.py`, `generate_test.py`) | `Vit_block.cross_attn` mixes `normalize(out)` at un-gated `t=0.5` in all 10 blocks; empty CFG prompts (`""`, repeated EOS/PAD tokens) drive spatial token cosine similarity to `+0.8933` by block 9, while `generate_test.py` passes `text_emb=None` (skipping `cross_attn` & `bottleneck_attn`) causing an `83.7%` relative L2 (`cos=0.6653`) train-inference mismatch. |
| **4** | **HIGH** | `EMA` Resume Reset (`training.py`, `utils.py`) | `ema.step` is neither saved in `save_checkpoint` nor restored on resume; resuming at step 26,000 with `ema_min_warmup_step=40000` resets `cur_beta` from `0.99886` to `0.37004`, overwriting `63%` of loaded EMA weights on the first resumed step. |
| **5** | **MEDIUM** | Inference `p_mean` Drift & Residual Hygiene (`generate_test.py`, `loss.py`, `logger2.py`) | `generate_test.py` uses `preconditioned_HDMOEM_v3` default `p_mean=-0.4` instead of training `p_mean=-0.2` (shifting UNet router probs by `0.3814` at $\sigma=5$); `laplacian_decomposition` is dead code; `Logger` appends across fresh runs and logs static LRs; `final_model1.pt` omits EMA; `lr_vit=8e-3` is `16x` larger than `lr_gates`. |

---

## 2. Confirmed Fixed Since Baseline Audit

1. **Forward Signal Variance Preservation Across UNet & ViT Depth (`[CODE: models/model_internals.py#L68-L85, L140-L144]`, `[CODE: models/model_components.py#L180, L216, L482-L483]`):**
   - Previously, `mp_sum(x, 0, t=0.5)` contracted RMS to `0.0625` in 8 blocks and `resample(down)` halved RMS (`0.5013`). Now `mp_gated_sum` + `res_gate=0.1` / `tmsa_gate=0.1` / `mlp_gate=0.1` and `resample(down)` `kernel * 2.0` preserve unit variance `[PAPER: edm2_karras_2024.md, Eq. 88]` `[BOOK: calin_deep_learning_architectures.md, Ch. 6.2.6]`.
   - `[PROBE: probe_post_fix_audit.py, PROBE A]` measures layer-by-layer RMS on 10-block `Vit_expert` (`post_patch+pos=0.9946 -> diffit.0=1.0619 -> diffit.4=1.0073 -> diffit.9=1.0024 -> post_bilinear=0.9495 -> post_refine=0.9810`) and 3-level `Unet_expert` (`32x32_conv=0.9616 -> 8x8_block0=1.0026 -> bottleneck_attn=1.0492 -> 32x32_up=1.0047 -> out_conv=0.9458`).

2. **Elimination of ViT `PixelShuffle(2)` Spatial Aliasing (`[CODE: models/model_components.py#L648-L656, L694-L712]`):**
   - Replacing `PixelShuffle(2)` with `unpatch_proj` + `F.interpolate(..., mode='bilinear', align_corners=False)` + `normalize` + `unpatch_skip` + `unpatch_refine` removes the `16x16 -> 32x32` stride-2 folding artifact `[BOOK: young_image_processing_and_murphy_dsm.md, p. 90]`.
   - `[PROBE: probe_post_fix_audit.py, PROBE C]` compares RAPSD against `1/f^2` latent baseline: old `PixelShuffle(2)` swings from `0.062x` (`r=1`) to `5.070x` (`r=15`), whereas the new `Vit_expert` head is flat across all radial frequencies (`r=1: 0.874x`, `r=2: 0.925x`, `r=4: 0.806x`, `r=8: 0.864x`, `r=12: 0.834x`, `r=14: 0.837x`, `r=15: 0.835x`).

3. **`EDM_LOSS_NOD` Spectral Skew & `log_var` Bias Fix (`[CODE: Utils/loss.py#L168-L194]`):**
   - Removing `laplacian_decomposition` and `exp(-log_var)` from `unet_loss` and `vit_loss` and detaching `wx_unet`/`wx_vit` eliminates both the `94.4%` isotropic white-noise skew into `vit_loss` and the `exp(-u*)` inflation (`vit_loss` rising above `pure_loss` solely from `D_blend` progress) `[PAPER: edm2_karras_2024.md, Eq. 21-28]`.
   - Verified by `[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 1]` (`vit_L` remains bounded `0.5804–0.8268` vs `pure_L 1.0641–1.6227`) and `[PROBE: test_info_geom_fixes.py, TestFix2LossAndDecomposition]` (`d(loss)/d(log_var) = 0` at unit error, and `d(loss)/d(w_vit)` unskewed).

4. **`spherical_blend`, `router_to_experts`, and Mixture-CDF Quantile Seeding (`[CODE: models/model_config_v3.py#L12-L98]`, `[CODE: Utils/cdf_utils.py#L21-L123]`):**
   - Analytic gradients match finite differences (`rel_err < 1e-3`), `branch_local_quantile_centers` prevents tail compression in `mixture_cdf`, and `ZetaScheduler` exponential continuity and `EMA` `warmup_power` exponent are corrected (`[PROBE: test_info_geom_fixes.py, TestPostAuditMathFixes]`).

---

## 3. Remaining & Newly Exposed Problems (Ranked by Severity)

### Problem 1 [CRITICAL]: Residual Outer Zero-Init Gates Create Step-0 Eliminating Singularities & Early Adam `eps` Stagnation

**Code Locations:**
- `time_proj_gate = 0` (`[CODE: models/model_config_v3.py#L213, L389-L391]`)
- `unet_proj_gate = 0`, `vit_proj_gate = 0` (`[CODE: models/model_config_v3.py#L293-L294, L419-L423]`)
- `blend_gate = 0` (`[CODE: models/model_config_v3.py#L303, L350]`)
- `PooledCrossAttn.out_proj_gate = 0` (`[CODE: models/model_internals.py#L611, L645]`)
- Redundant multiplicative scalar gains `out_gain_vit = 1`, `out_gain_unet = 1` (`[CODE: models/model_config_v3.py#L296-L297, L422-L423]`)
- `lr_gates = 5e-4`, `rampup_Mimg = 0.25`, `batch_size = 128` (`[CODE: Utils/configs.py#L46, L84-L89]`, `[CODE: Utils/training.py#L96-L141]`)

**Mathematical Derivation:**
By `[BOOK: amari_information_geometry.md, Ch. 12.2.1-12.2.5]`, any bilinear parameterization $y = g_{\text{gate}} \cdot H_\theta(x)$ places the manifold $\{g_{\text{gate}} = 0\}$ on an **eliminating singularity** $R_e$ where the pullback Fisher metric vanishes along $\theta$ ($a^\top G a = 0$) and gradient flow exhibits algebraic slowdown near the saddle `[BOOK: strogatz_nonlinear_dynamics.md, Ch. 2.4-2.7, Ch. 6.3]`. Even though internal gates (`res_gate`, `tmsa_gate`, `mlp_gate`) were moved to `0.1`, placing `unet_proj_gate = 0` and `vit_proj_gate = 0` at the branch outputs sets `unet_out == 0` and `vit_out == 0` identically at step 0, and sets `raw_correction == 0` via `out_proj_gate = 0`. Consequently:
1. **Complete inner-network step-0 blackout:** Every parameter upstream of `unet_proj_gate`, `vit_proj_gate`, and `out_proj_gate` has exact zero gradient:
   $$\left.\frac{\partial \mathcal{L}}{\partial \theta_{\text{inner}}}\right|_{t=0} = 0, \qquad \left.\frac{\partial \mathcal{L}}{\partial g_{\text{time}}}\right|_{t=0} = 0, \qquad \left.\frac{\partial \mathcal{L}}{\partial g_{\text{blend}}}\right|_{t=0} = 0, \qquad \left.\frac{\partial \mathcal{L}}{\partial \text{out\_gain}}\right|_{t=0} = 0$$
   `[DERIVED FROM: amari_information_geometry.md, Ch. 12.2 + models/model_config_v3.py#L213, L293-L303, L419-L443]`. Notably, `blend_gate = 0` and `time_proj_gate = 0` are **second-order (nested) singularities** behind `vit_proj_gate = 0` and `unet_proj_gate = 0` because `d(spherical_blend(0, 0, w_v, w_u))/d(w_v) = 0`, violating `[PAPER: edm2_karras_2024.md, App. B.7 Eq. 104]`'s single final-output gain rule.
2. **Adam `eps` attenuation + slow warmup escape:** Under `Lr_Scheduler` (`rampup_Mimg = 0.25`, `batch_size = 128`), linear warmup takes $250{,}000 / 128 = 1{,}953.125$ steps, so `lr_gates(0) = 2.56e-7`. Under Adam $\Delta \theta_t = -\alpha_t \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ `[BOOK: calin_deep_learning_architectures.md, Ch. 4.5-4.8]`, while `|vit_proj_gate| ~ 1e-5`, inner expert gradients scale linearly with `|vit_proj_gate|` and fall to $O(10^{-9}\text{--}10^{-8})$ — comparable to `eps = 1e-8`, damping early expert updates and keeping the score network nearly silent (`|g| ~ 0.012`) across the first ~250 steps `[DERIVED FROM: calin_deep_learning_architectures.md, Ch. 4.5-4.8 + Utils/configs.py#L84-L89]`.

**Empirical Verification:**
- `[PROBE: probe_post_fix_audit.py, PROBE A]`: At step 0: `input_proj+time_stem`: `4/4` exact zero grad (`0.0000e+00`); `Unet_experts_all`: **`258/258`** exact zero grad; `VIT_experts_all`: **`348/348`** exact zero grad; `output_proj_unet`: `1/1`; `output_proj_vit`: `1/1`; `output_proj_attn`: `1/1`; `blend_residual`: `2/2`; `pooled_cross_attn`: `5/6` exact zero grad; `time_proj_gate`, `blend_gate`, `out_gain_vit`, `out_gain_unet`: all `0.0000e+00`.
- `[PROBE: probe_post_fix_audit.py, PROBE B]` & `[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 1]`: At step 1, `g_unp_skip = 1.72e-09` and `g_diffit_blk0 = 1.56e-09` (both `< eps = 1e-8`!). By step 40, `vit_g = +0.00013`, `unet_g = +0.00014`, `time_g = -0.00012`, `blend_g = +0.00006`. Even by **step 250**, `vit_g = +0.01230`, `unet_g = +0.01209`, `time_g = +0.00948`, `blend_g = +0.00539`, `attn_g = +0.01066`.

---

### Problem 2 [HIGH]: `Unet_router` Batch-Wide `90/10` Imbalance, `Unet_experts[1]` Starvation & `79%` Wasted UNet Compute

**Code Locations:**
- `Utils/configs.py#L52, L95-L106` (`p_mean=-0.2, p_std=1.2, transition_sigma=2.5, extreme_prob=0.2, min_BW_u=0.05`)
- `Utils/cdf_utils.py#L21-L123` (`mixture_cdf`, `branch_local_quantile_centers`)
- `models/model_components.py#L40-L104` (`ParametricTrajectoryRouter`)
- `models/model_config_v3.py#L34-L50, L402-L405` (`router_to_experts` runs every sample with `weight > 1e-5`)
- `graphs/logger2.py#L23-L24, L243-L248` (`MIN_HEALTHY_ENTROPY_RATIO=0.85`, `MAX_HEALTHY_GINI=0.25`)

**Mathematical Derivation:**
Under `sample_sigma_hybrid` (`[CODE: Utils/utils.py#L46-L67]`) with `p_mean=-0.2`, `p_std=1.2`, `sigma_min=0.002`, `sigma_max=80.0`, `alpha_ext=0.2`, the mixture CDF at `transition_sigma = 2.5` is:
$$F_{\text{mix}}(2.5) = 0.8\,\Phi\!\left(\frac{\ln 2.5 - (-0.2)}{1.2}\right) + 0.2\,\frac{\ln 2.5 - \ln 0.002}{\ln 80 - \ln 0.002} = 0.8(0.8239) + 0.2(0.6729) = 0.7937$$
`[DERIVED FROM: Utils/cdf_utils.py#L48-L57 + Utils/configs.py#L52, L95-L96]`. Thus ViT covers `79.37%` of training samples while UNet covers only `20.63%`.
1. Because `Unet_router` receives the **entire batch** (`[CODE: models/model_config_v3.py#L398-L405]`) with branch-local centers `[0.8450, 0.9477]`, every sample with $\sigma < 2.5$ (`79.37%` of the batch) is closer to expert 0 (`0.8450`) and receives `Unet_probs[:, 0] ≈ 1.0`, while only the upper half of UNet's own regime (`~10.3%`) routes to `Unet_experts[1]` (`5x5` kernel).
2. Consequently, expected batch usage is $\bar{p}_{\text{UNet}} \approx [0.897, 0.103]$, giving normalized entropy $H / \ln 2 \approx 0.48 < 0.85$ and Gini $G \approx 0.40 > 0.25$ `[DERIVED FROM: graphs/logger2.py#L36-L73]` — guaranteeing `unet_healthy = False` in `Logger`.
3. `router_to_experts` executes `Unet_experts[0]` on `89.5%–99.2%` of batch samples (`weights > 1e-5`) even though for $\sigma \ll 2.5$, `w_unet` is tiny (`w_unet(0.002) = 0.0001`), causing UNet expert gradient norm to drop by **`16,600x`** from $\sigma=80$ to $\sigma=0.002$ (`[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 2]`), while `Unet_experts[1]` receives **`8.55x` fewer** samples (`~13` per 128-batch).
4. Furthermore, at late `MaskGenerator` steps (`min_BW_u = 0.05` vs center spacing `0.9477 - 0.8450 = 0.1027 > 2 * 0.05`), the two UNet masks have **zero overlap** (`[0.8950, 0.8977]` gap covered only by `min_active=1` argmin), creating a hard routing boundary inside `[2.5, 80]` unless `MaskGenerator` is also passed at inference.

**Empirical Verification (`[PROBE: probe_post_fix_audit.py, PROBE D]` & `[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 2]`):**
- `F_mix(2.5) = 0.7937` (`ViT = 79.37%`, `UNet = 20.63%`); `UNet centers = [0.8450, 0.9477]`, `ViT centers = [0.1992, 0.5955]`.
- `UNet avg_usage = [0.8953, 0.1047]`, `ent_norm = 0.4838`, `gini = 0.3953` (`PASS_unet_router = False`).
- Late-stage 2-expert overlap fraction: `UNet = 0.0000`, `ViT = 0.1038`.
- `grad_unet_exp` across $\sigma$: `2.2978e-01` ($\sigma=80$) vs `1.3822e-05` ($\sigma=0.002$).

---

### Problem 3 [HIGH]: Un-gated `Vit_block.cross_attn` Spatial Rank Collapse Under Empty CFG Prompts & `83.7%` Train/Inference Mismatch

**Code Locations:**
- `models/model_components.py#L469-L472, L540-L542` (`Vit_block.cross_attn`), `#L107-L125, L392-L395` (`Unet_expert.bottleneck_attn`)
- `models/model_internals.py#L480-L533` (`MP_Attention.forward`)
- `Utils/training.py#L286-L298` (CFG replaces dropped prompts with `""` via `clip.encode_text`)
- `Utils/generate_test.py#L57, L254, L319` (`text_emb=None` for unguided evaluation)

**Mathematical Derivation:**
1. In `Vit_block.forward` (`#L540-L542`), whenever `text_embedding is not None`, `cross_attn` runs `MP_Attention(..., attn_balance=0.5, is_cross_attn=True)` **without a learnable gate** (unlike `tmsa_gate` and `mlp_gate`):
   $$y \leftarrow \operatorname{mp\_sum}\!\left(y,\; W_{\text{out}}\operatorname{normalize}\!\left(\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V(c)\right),\; t=0.5\right).$$
   Because `normalize(out)` rescales every spatial token's cross-attention output to unit RMS regardless of key/value variance (`[CODE: models/model_internals.py#L527]`), and CLIP's empty prompt `""` produces 75+ identical `<|endoftext|>` padding tokens (with **no** padding attention mask in `MP_Attention`), `softmax(...) @ V(c)` becomes nearly identical across all $H \times W$ spatial query positions. Repeating `0.707\,y + 0.707\,u_{\text{const}}` across **10 consecutive `Vit_block`s** contracts spatial variation and drives token representations toward rank-1 alignment `[PAPER: ngpt_loshchilov_2024.md, Sec. 2]`.
2. **Train-vs-Inference Structural Mismatch:** During training (`cfg_drop_prob = 0.10`, `Utils/training.py#L288-L298`), unconditional samples receive `clip("")` (a `(B, 77, 768)` tensor), so `cross_attn` in all 10 `Vit_block`s and `bottleneck_attn` in `Unet_expert` **execute** (`mp_sum(..., t=0.5)`). Calling the model with `text_emb=None` (as in `Utils/generate_test.py#L57, L254, L319`) skips those 11 `mp_sum` layers altogether!

**Empirical Verification (`[PROBE: probe_post_fix_audit.py, PROBE C]`):**
- Mean pairwise off-diagonal spatial token cosine similarity across 10 `Vit_block`s (`32x32` -> `256` tokens):
  - `text_emb=None`: `blk0=+0.4655 -> blk3=+0.5704 -> blk6=+0.5804 -> blk9=+0.5126` (`out_std=1.0148`)
  - `text_emb=varied`: `blk0=+0.4545 -> blk3=+0.5658 -> blk6=+0.6464 -> blk9=+0.6310` (`out_std=0.9778`)
  - `text_emb=clip("")` (BOS + 76 identical EOS/PAD): `blk0=+0.5713 -> blk3=+0.8166 -> blk6=+0.8324 -> blk9=+0.8933` (`out_std=0.9624`) — severe spatial collapse!
- Train-vs-inference mismatch (`text_emb=None` vs `text_emb=clip("")` on `Vit_expert`): `cos_sim = 0.6653`, `rel_L2 = 0.8366`.

---

### Problem 4 [HIGH]: `EMA.step` Reset on Checkpoint Resume Corrupts Loaded EMA Weights by `63%`

**Code Locations:**
- `Utils/utils.py#L31-L40` (`save_checkpoint` omits `ema.step` and `scheduler`)
- `Utils/utils.py#L313, L331-L342` (`EMA.__init__` sets `self.step = 0`; `update` computes `cur_beta = min(beta, 1 - (1 + step)^(-warmup_power))`)
- `Utils/training.py#L244-L247, L369, L403-L407, L417` (`resume_path="./ckpt_26000.pt"`), `Utils/configs.py#L58` (`ema_min_warmup_step = 40000`)

**Mathematical Derivation:**
Upon resuming from `ckpt_26000.pt` (`start_step = 26001 < ema_min_warmup_step = 40000`), `EMA` is constructed fresh with `ema.step = 0`, and only `ema.ema_model.load_state_dict(...)` is restored. On the first resumed step (`ema.step = 1`), `EMA.update()` computes:
$$\beta_{\text{actual}} = 1 - (1 + 1)^{-2/3} = 0.37004 \quad \text{instead of} \quad \beta_{\text{expected}}(26001) = 1 - (26002)^{-2/3} = 0.99886,$$
replacing $1 - 0.37004 = \mathbf{63.0\%}$ of the smoothed EMA shadow parameters with the unsmoothed online weights in one step `[DERIVED FROM: Utils/utils.py#L331-L342 + Utils/configs.py#L58]`. Additionally, `final_model1.pt` (`Utils/training.py#L403-L407`) does not pass `ema_instance=ema` to `save_checkpoint`.

**Empirical Verification (`[PROBE: probe_post_fix_audit.py, PROBE E]`):**
- Resuming with `ema_model.weight = 0.5` and `src.weight = 1.0`: expected EMA weight after 1 step at `26,001` is `0.50057`; actual weight produced by `training.py` resume logic is **`0.81498`**.

---

### Problem 5 [MEDIUM]: Inference `p_mean` Mismatch (`generate_test.py`) + Optimizer LR Asymmetry + Logger/Dead-Code Hazards

1. **`generate_test.py` `p_mean` mismatch (`[CODE: Utils/generate_test.py#L452-L482]`, `[CODE: models/model_config_v3.py#L514]`, `[CODE: Utils/configs.py#L95]`):**
   - `sample_and_save` instantiates `preconditioned_HDMOEM_v3` without passing `p_mean`/`p_std`, defaulting to `p_mean = -0.4` while `training.py` uses `mask_configs['p_mean'] = -0.2`. Because `self.p_mean` is not in `state_dict`, `mixture_cdf` shifts relative to loaded `centers`, altering UNet router probabilities at $\sigma = 5.0$ from `[0.5398, 0.4602]` to `[0.1584, 0.8416]` (`delta = 0.3814`, `[PROBE: probe_post_fix_audit.py, PROBE E]`).
2. **Adam Step-Size Asymmetry on `MP_Conv` vs Scalar Gates (`[CODE: models/model_internals.py#L26-L30, L314-L320]`, `[CODE: Utils/configs.py#L76-L86]`):**
   - Because `MP_Conv` normalizes each filter to $\|w_j\|_2 = \sqrt{N_j}$ (`[PROBE: probe_post_fix_audit.py, PROBE B]` confirms `||w_j||_2 = 23.9976` for `N_j = 576`), every entry of `w_j` has unit RMS (`O(1)`), so Adam's angular step on the sphere $\mathbb{S}^{N_j-1}(\sqrt{N_j})$ is $\Delta\theta \approx \text{lr}$ (`[PAPER: edm2_karras_2024.md, Eq. 59, 66]`, `[BOOK: lee_riemannian_manifolds.md, Prop. 2.16]`, `[PAPER: ngpt_loshchilov_2024.md, Eq. 22]`). Thus `lr_vit = 8e-3` (`0.008` rad/step) and `lr_unet = 5e-3` are **`16x` and `10x` larger** than `lr_gates = 5e-4` and `LR_DEFAULT = 5e-4`, amplifying early instability once gates open while starving gate escape during warmup.
3. **`Vit_expert` `unpatch_skip` conditioning bypass (`[CODE: models/model_components.py#L651, L710-L711]`):**
   - `detail_skip = unpatch_skip(normalize(x_in))` contributes `~50%` of `Vit_expert` variance (`cos(full, skip_only) = 0.6870` vs `cos(full, vit_only) = 0.7280`, `[PROBE: probe_post_fix_audit.py, PROBE C]`), bypassing all 10 `Vit_block`s (time & text conditioning). However, `diffit[0]` still receives comparable gradient to `unpatch_skip` (`g_skip / g_blk0 = 0.57x–1.76x`, `[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 1]`), so this is a secondary conditioning-leakage risk rather than a gradient blocker.
4. **Dead code, logging, & fragile ops:**
   - `laplacian_decomposition` (`[CODE: Utils/loss.py#L15-L32]`) and `lap_kernel`/`lap_sigma` (`[CODE: Utils/loss.py#L99-L100]`, `[CODE: Utils/configs.py#L66-L67]`) are unused in `EDM_LOSS_NOD.__call__`.
   - `Utils/training.py#L256-L261` removes `training_stats.jsonl` while `Logger` appends (`'a'`) to `hdmoem_flower102_v1_*.jsonl` (`[CODE: graphs/logger2.py#L560]`), and `Logger.log_gradients` uses unscheduled static `lr_per_component` (`[CODE: Utils/training.py#L41-L49, L361-L365]`) while `Logger._sigma_percentile` uses pure Gaussian CDF instead of `mixture_cdf` (`[CODE: graphs/logger2.py#L130-L135]`).
   - `Vit_block.forward` calls `m.resample(x, mode=self.resample)` on a 3D `(B, Seq, C)` tensor (`[CODE: models/model_components.py#L510]`), safe only while `resample='keep'`.
   - `model_configs` mixes LHQ paths/stats (`data_dir`, `raw_mean/std`) with `"fixed_prompt": "a photo of a flower"` and `run_name="hdmoem_flower102_v1"`, and `ImageFolder` (`[CODE: data_augmentation/data_collector.py#L23-L33, L90-L97]`) assigns folder-name prompts if uncaptioned.

---

## 4. Explicitly Falsified / Confirmed Healthy Mechanisms

- **ViT `[2,2]` Head Spatial Aliasing:** **Falsified on current `text_guided` code** — replaced `PixelShuffle(2)` with `bilinear + unpatch_skip + unpatch_refine`, yielding flat RAPSD `0.806x–0.925x` (`[PROBE: probe_post_fix_audit.py, PROBE C]`).
- **`unpatch_skip` Starving `Vit_block` Gradients:** **Falsified** — once gates open (`steps 50–250`), `g_unpatch_skip / g_diffit_blk0` stays balanced at `0.57x–1.76x` (`[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 1]`).
- **Pre-`f945711` Block Variance Contraction & Downsample Attenuation:** **Falsified on current code** — both `Unet_expert` and `Vit_expert` maintain `RMS ~0.95–1.06` across full depth (`[PROBE: probe_post_fix_audit.py, PROBE A]`).
- **`EDM_LOSS_NOD` `log_var` Bias & Branch Weight Gradient Skew:** **Falsified on current code** — `23/23` unit tests in `test_info_geom_fixes.py` pass (`[PROBE: test_info_geom_fixes.py]`).
- **`EDM_Sampler` Instability / NaN Drift under CFG + Masks:** **Falsified** — `g=1.0` unmasked and `g=5.0` masked/unmasked with `clip("")` produce finite, bounded latents (`std=0.9428` and `1.3848–1.3861`, `[PROBE: probe_post_fix_dynamics_and_sampler.py, PROBE 3]`).

---

## 5. Mathematically Grounded Remedies (Prioritized)

1. **Break the Outer Gate Singularity & Remove Redundant Gains (`[CODE: models/model_config_v3.py#L213, L293-L303]`, `[CODE: models/model_internals.py#L611]`, `[CODE: Utils/configs.py#L77-L86]`):**
   - Keep **at most one** zero-initialized scalar output gain per parallel residual output to the denoiser (`vit_proj_gate = 0` or `out_gain_vit`, `unet_proj_gate = 0` or `out_gain_unet`, `out_proj_gate = 0` — matching `[PAPER: edm2_karras_2024.md, Eq. 104]`), **or** initialize them to a small positive value (`0.05–0.1`) so `100%` of expert/stem gradients are non-zero at step 0 (`[DERIVED FROM: amari_information_geometry.md, Ch. 12.2]`).
   - Initialize `time_proj_gate = 0.1` and `blend_gate = 0.1` (no longer behind a `0 * 0` saddle), and remove the redundant `out_gain_vit` / `out_gain_unet` scalar pair.
   - Increase `lr_gates` to match `LR_DEFAULT` / branch scale (e.g., `2e-3` or `s_init/s_restore` decoupling per `[PAPER: ngpt_loshchilov_2024.md, Eq. 22]`) and moderate `lr_vit`/`lr_unet` (`1e-3–3e-3` rather than `8e-3/5e-3`).

2. **Fix `Unet_router` Low-$\sigma$ Waste & `Unet_experts[1]` Starvation (`[CODE: models/model_config_v3.py#L36, L398-L405]`, `[CODE: Utils/configs.py#L95-L106]`):**
   - Skip `Unet_experts` execution (and exclude $\sigma < \text{transition\_sigma}$ samples from `Unet_router` health/balance statistics) whenever `w_unet` is negligible or mask UNet execution cleanly outside its active handoff window (e.g. `w_unet < 1e-3`), saving `~70–79%` of UNet FLOPs and eliminating the artificial `[0.90, 0.10]` router skew.
   - Raise `min_BW_u` from `0.05` to `>= 0.08–0.10` (or keep `max_BW_u = 0.25 -> min_BW_u = 0.10`) so UNet experts maintain smooth overlap near the midpoint `0.8964`, and pass `unet_mask_gen`/`vit_mask_gen` in `generate_test.py`.

3. **Gate `Vit_block.cross_attn`, Support Attention Padding Mask & Fix Train/Inference Unconditional Parity (`[CODE: models/model_components.py#L469-L472, L540-L542]`, `[CODE: models/model_internals.py#L480-L523]`, `[CODE: Utils/generate_test.py#L57, L254, L319]`):**
   - Replace un-gated `mp_sum(y, out, t=0.5)` in `Vit_block.cross_attn` (and `SpatialCrossAttn`) with `mp_gated_sum(..., gate=self.ca_gate)` (`ca_gate` initialized to `0.1`), and pass CLIP's `attention_mask` (`-inf` before softmax on padding tokens) so `""` padding tokens do not collapse spatial rank to `+0.893`.
   - Always pass `clip.encode_text([""] * B)` (never `None`) for unconditional inference in `generate_test.py` and `EDM_Sampler` so the forward graph matches training's `cfg_drop_prob=0.10`.

4. **Persist `ema.step`, `scheduler.state_dict()`, and `p_mean` in Checkpoints (`[CODE: Utils/utils.py#L31-L40]`, `[CODE: Utils/training.py#L244-L247, L403-L407]`, `[CODE: Utils/generate_test.py#L452-L482]`):**
   - Save and restore `ema.step` and `scheduler.state_dict()` in `save_checkpoint` / `training_HDMOE`, pass `ema_instance=ema` to `final_model1.pt`, and register `p_mean`/`p_std` as buffers in `ParametricTrajectoryRouter` (and pass `mask_configs['p_mean']` in `generate_test.py`).
   - Clean up unused `laplacian_decomposition` / `lap_kernel` / `lap_sigma` in `EDM_LOSS_NOD`, clear `{run_name}_*.jsonl` on fresh training starts, and pass live scheduled LRs to `logger.log_gradients`.

---

## 6. Source & Probe Audit Matrix

| Claim / Mechanism | Grounded Book / Paper Tag | Code Locator | Empirical Probe |
| :--- | :--- | :--- | :--- |
| Internal `mp_gated_sum` & `resample(down)` fix | `[PAPER: edm2_karras_2024.md, Eq. 88]` `[BOOK: calin_deep_learning_architectures.md, Ch. 6.2.6]` | `models/model_internals.py#L68-L85, L140-L144` | `probe_post_fix_audit.py` (`PROBE A`) |
| `Vit_expert` `bilinear+skip+refine` removes aliasing | `[BOOK: young_image_processing_and_murphy_dsm.md, p. 90]` | `models/model_components.py#L648-L656, L694-L712` | `probe_post_fix_audit.py` (`PROBE C`) |
| `EDM_LOSS_NOD` `log_var` & `wx.detach()` fix | `[PAPER: edm2_karras_2024.md, Eq. 21-28]` | `Utils/loss.py#L168-L194` | `test_info_geom_fixes.py` (`TestFix2`, `TestPostAudit`) |
| Outer zero-init gates & Adam `eps` warmup stall | `[BOOK: amari_information_geometry.md, Ch. 12.2]` `[BOOK: strogatz_nonlinear_dynamics.md, Ch. 2.4-2.7, 6.3]` `[BOOK: calin_deep_learning_architectures.md, Ch. 4.5-4.8]` `[PAPER: edm2_karras_2024.md, Eq. 104]` | `models/model_config_v3.py#L213, L293-L303` `Utils/configs.py#L84-L89` | `probe_post_fix_audit.py` (`PROBE A, B`), `probe_post_fix_dynamics_and_sampler.py` (`PROBE 1`) |
| `MP_Conv` sphere step size vs `lr_vit=8e-3` | `[PAPER: edm2_karras_2024.md, Eq. 59, 66]` `[BOOK: lee_riemannian_manifolds.md, Prop. 2.16]` `[PAPER: ngpt_loshchilov_2024.md, Eq. 22]` | `models/model_internals.py#L26-L30, L314-L320` `Utils/configs.py#L76-L86` | `probe_post_fix_audit.py` (`PROBE B`) |
| `Unet_router` `90/10` skew & expert-1 starvation | `[PAPER: diffmoe_shi_2025.md, Sec. 1-2.2]` | `Utils/configs.py#L95-L106` `Utils/cdf_utils.py#L21-L123` `graphs/logger2.py#L23-L73` | `probe_post_fix_audit.py` (`PROBE D`), `probe_post_fix_dynamics_and_sampler.py` (`PROBE 2`) |
| `Vit_block.cross_attn` collapse & CFG mismatch | `[PAPER: ngpt_loshchilov_2024.md, Sec. 2]` `[PAPER: freeu_si_2023.md, Sec. 1-2]` | `models/model_components.py#L540-L542` `models/model_internals.py#L527, L533` `Utils/training.py#L286-L298` `Utils/generate_test.py#L57` | `probe_post_fix_audit.py` (`PROBE C`) |
| `EMA` resume reset & `generate_test.py` `p_mean` | — (`Utils` state bug & CDF parameter drift) | `Utils/utils.py#L31-L40, L313-L342` `Utils/training.py#L244-L247` `Utils/generate_test.py#L452-L482` | `probe_post_fix_audit.py` (`PROBE E`) |
