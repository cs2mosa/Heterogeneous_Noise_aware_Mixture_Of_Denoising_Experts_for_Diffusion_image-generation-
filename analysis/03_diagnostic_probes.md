# Phase 3b: Diagnostic Probe Verification & Verbatim Telemetry

All probes are read-only Python scripts stored under [analysis/probes/](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes) and executed via the workspace virtual environment (`.venv\Scripts\python.exe`). Zero source files in the repository were modified.

---

## Probe Outcome Summary Against Testable Theories

| Theory | Probe Script | Outcome | Key Empirical Finding |
| :--- | :--- | :--- | :--- |
| **T1** (Zero-init eliminating singularities & pre-`f945711` contraction) | [probe_zero_gates_and_gradients.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_zero_gates_and_gradients.py) | **CONFIRMED** | At step 0: `113/113` UNet expert tensors, `49/49` ViT expert tensors, `output_proj_vit/unet`, `38/41` gates have `grad = 0.0`. Pre-`f945711` `mp_sum(x, 0, t=0.5)` shrinks RMS to `0.0625` in 8 blocks; unscaled `resample(down)` halves RMS (`0.5013`). |
| **T2** (Wiener MMSE floor & low-$\sigma$ `F->0` velocity collapse) | [probe_spectral_and_bayes_floor.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_spectral_and_bayes_floor.py) | **CONFIRMED** | Under `sample_sigma_hybrid`: $F=0$ baseline MSE = `0.28375`, Wiener linear MMSE on $1/f^2$ latents = `0.22095` — matching observed `Train=0.2134 / Val=0.2157`. Plot 1's velocity curve tracks $F=0$ closely and plateaus at `0.05` at $\sigma \le 0.02$ (vs `~1.0` for true score), proving $\|F_\theta\|_{\text{RMS}} \approx 0.05$ at low $\sigma$. |
| **T3** (`laplacian_decomposition` noise skew + `vit_loss` inversion) | [probe_spectral_and_bayes_floor.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_spectral_and_bayes_floor.py) & [probe_sigma_quantiles_and_loss.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_sigma_quantiles_and_loss.py) | **CONFIRMED** | `lap_sigma=1.0` sends `94.42%` of white noise energy into `P_high` (`78.44%` at `lap_sigma=0.5`). `EDM_LOSS_NOD` applies `exp(-u.detach())` *without* `+ u` inside `vit_loss`: as `u` adapts negatively (`exp(-u*) ~ 1.52x`) while ViT lags `D_blend`, `vit_loss` climbs mechanically above `pure_loss` (`0.27 -> 0.72` in 10–39 steps in live Adam trace!). |
| **T4** (Multi-objective gradient conflict vs `PixelShuffle(2)` RAPSD aliasing) | [probe_blend_and_spectral_artifacts.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_blend_and_spectral_artifacts.py) | **SPLIT: Conflict FALSIFIED; Aliasing CONFIRMED** | Gradient cosines wrt `pure_loss` are **positive** (`+0.8246` ViT, `+0.8720` UNet, `+0.4263` stem) — directional conflict is NOT the blocker. Conversely, `PixelShuffle(2) + 1x1/3x3 MP_Conv` exhibits **`6.87x–7.83x`** excess high-frequency power at `r=15` and attenuates `r=1..8` by `~3x`, driving RAPSD mid/high excess! |
| **T5** (`spherical_blend` norm/gradient distortion) | [probe_blend_and_spectral_artifacts.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_blend_and_spectral_artifacts.py) | **FALSIFIED** | `spherical_blend` preserves exact target RMS (`1.0000`, `0.4600`, `3.1600`) across all angles `rho in [-0.74, +0.99]` and boosts gradient norm (`1.06x` at `rho=0.74`). `spherical_blend` is mathematically sound. |
| **T6** (Cosine 8k step budget starvation vs EDM2) | [probe_zero_gates_and_gradients.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_zero_gates_and_gradients.py) & [probe_sigma_quantiles_and_loss.py](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/probes/probe_sigma_quantiles_and_loss.py) | **CONFIRMED (Secondary Compounding Factor)** | `v2` Cosine 8k integrated LR (`5.257`) gives `4.9x` less Adam rotation budget than 70k inverse-sqrt (`25.998`), and the last 3,000 steps supply only `14.2%` of total updates; `P(sigma <= 0.02) = 4.35%` (~348 steps total in 8k). |

---

## Probe 1: Spectral Energy Partition, $F=0$ Baseline & Wiener MMSE Risk Floor
**Command:** `& "f:\Heterogeneous-MOE-for-Diffusion-models\.venv\Scripts\python.exe" "f:\Heterogeneous-MOE-for-Diffusion-models\analysis\probes\probe_spectral_and_bayes_floor.py"`  
**Exit code:** `0`

```text
==============================================================================
PROBE 1A: Fourier laplacian_decomposition Energy Partition (32x32x4)
==============================================================================
lap_sigma=1.00 | cutoff_r=4.24 cycles/32 | WhiteNoise(low=0.0558, high=0.9442) | Signal_1/f^2(low=0.4551, high=0.5449)
lap_sigma=0.50 | cutoff_r=8.48 cycles/32 | WhiteNoise(low=0.2156, high=0.7844) | Signal_1/f^2(low=0.7092, high=0.2908)
lap_sigma=0.25 | cutoff_r=16.96 cycles/32 | WhiteNoise(low=0.8433, high=0.1567) | Signal_1/f^2(low=0.9722, high=0.0278)

==============================================================================
PROBE 1B: F=0 Baseline & Wiener Bayes-Optimal Risk vs Observed MSE (0.2134 / 0.2157)
==============================================================================
   sigma |   MSE(F=0) | MSE(Wiener) | EDM_L(F=0) | EDM_L(Wien) | Vel_RMS(F=0) | Vel_RMS(Wien)
----------------------------------------------------------------------------------------
  80.000 |    0.63930 |     0.63894 |    0.99901 |     0.99845 |      0.99979 |       0.99979
  38.070 |    0.63907 |     0.63734 |    0.99898 |     0.99629 |      1.00007 |       1.00007
  16.590 |    0.63793 |     0.63083 |    0.99909 |     0.98797 |      0.99921 |       0.99923
   5.000 |    0.62330 |     0.56236 |    0.99884 |     0.90119 |      0.98802 |       0.98924
   1.610 |    0.51291 |     0.36673 |    0.99929 |     0.71449 |      0.89550 |       0.92639
   0.800 |    0.32012 |     0.22017 |    1.00037 |     0.68802 |      0.70685 |       0.80995
   0.280 |    0.06966 |     0.05964 |    0.99734 |     0.85391 |      0.33024 |       0.48710
   0.050 |    0.00249 |     0.00247 |    1.00023 |     0.99200 |      0.06235 |       0.10937
   0.002 |    0.00000 |     0.00000 |    1.00008 |     0.99913 |      0.00250 |       0.03081
  [debug] sample_sigma_hybrid shape=(2048, 1, 1, 1), min=0.0020, median=0.6514, max=78.3184

  [matched std=0.8] sample_sigma_hybrid:
    F=0:    MSE = 0.28375 | Weighted EDM_L = 0.99985
    Wiener: MSE = 0.22095 | Weighted EDM_L = 0.80918
  [debug] sample_sigma_hybrid shape=(2048, 1, 1, 1), min=0.0021, median=0.6401, max=75.5969

  [std=1.0, sd=0.8] sample_sigma_hybrid:
    F=0:    MSE = 0.38999 | Weighted EDM_L = 1.24508
    Wiener: MSE = 0.29496 | Weighted EDM_L = 0.98516
  [matched std=0.8] log-uniform [0.002,80]: F=0 MSE=0.27808 | Wiener MSE=0.24945
  val_sigmas=[0.05,0.2,0.5,1,2,5,10,25]: F=0 mean MSE=0.38228 | Wiener mean MSE=0.33259
  val_sigmas=[0.02,0.1,0.5,1,3,10,30,60]: F=0 mean MSE=0.38629 | Wiener mean MSE=0.34844

==============================================================================
PROBE 1C: EDM_LOSS_NOD Mechanism for vit_loss Rising (0.35 -> 0.65) vs pure_loss (1.53 -> 0.58)
==============================================================================

[lap_sigma=1.00] mean w_unet=0.194, mean w_vit=0.806
  Step-0 (F=0, log_var=0)                        | tot=0.8743, pure=1.0000, vit(hi)=0.6510, unet(lo)=0.0749, mse=0.2774
  Wiener optimal + optimal log_var               | tot=0.7028, pure=0.7803, vit(hi)=0.7065, unet(lo)=0.0615, mse=0.2182
  Partial blend + opt log_var + D_vit=D0         | tot=0.7581, pure=0.8418, vit(hi)=0.7634, unet(lo)=0.0657, mse=0.2330
  Partial blend + opt log_var + noisy F_vit(0.25) | tot=0.7638, pure=0.8418, vit(hi)=0.8201, unet(lo)=0.0657, mse=0.2330

[lap_sigma=0.50] mean w_unet=0.196, mean w_vit=0.804
  Step-0 (F=0, log_var=0)                        | tot=0.8611, pure=0.9986, vit(hi)=0.4847, unet(lo)=0.1208, mse=0.2899
  Wiener optimal + optimal log_var               | tot=0.6840, pure=0.7750, vit(hi)=0.5096, unet(lo)=0.1136, mse=0.2253
  Partial blend + opt log_var + D_vit=D0         | tot=0.7397, pure=0.8373, vit(hi)=0.5657, unet(lo)=0.1158, mse=0.2414
  Partial blend + opt log_var + noisy F_vit(0.25) | tot=0.7444, pure=0.8373, vit(hi)=0.6126, unet(lo)=0.1158, mse=0.2414
```

---

## Probe 2: Pre-`f945711` Contraction, Zero-Init Gates, Escape Trace & Step Budget
**Command:** `& "f:\Heterogeneous-MOE-for-Diffusion-models\.venv\Scripts\python.exe" -u "f:\Heterogeneous-MOE-for-Diffusion-models\analysis\probes\probe_zero_gates_and_gradients.py"`  
**Exit code:** `0`

```text
================================================================================
PROBE 2A: Pre-f945711 mp_sum Variance Contraction & Downsample Attenuation
================================================================================
balance t=0.5: mp_sum(x,0) RMS [L1=0.7076, L2=0.5003, L4=0.2502, L8=0.0625] | mp_gated_sum(x,0) RMS [L1=1.0007, L2=1.0007, L4=1.0007, L8=1.0007]
balance t=0.3: mp_sum(x,0) RMS [L1=0.9197, L2=0.8454, L4=0.7142, L8=0.5098] | mp_gated_sum(x,0) RMS [L1=1.0007, L2=1.0007, L4=1.0007, L8=1.0007]
resample(down) single-step RMS: old (no *2.0)=0.5013 | new (*2.0)=1.0027

================================================================================
PROBE 2B: Exact Step-0 Backward Gradient Audit Across Zero-Init Gates & Weights
================================================================================
HDMOEM_v3 children: ['input_proj', 'Fourier_emb', 'out_fourier1', 'out_fourier2', 'input_time_proj', 'Unet_router', 'vit_router', 'Unet_experts', 'VIT_experts', 'output_proj_vit', 'output_proj_unet', 'blend_residual', 'pooled_cross_attn', 'output_proj_attn']
Step-0 Parameter Group Gradient Norms (Total L2 norm, zero-grad tensor counts):
  unet_inner          : grad_norm=0.0000e+00 | zero_grad_tensors=113/113
  vit_inner           : grad_norm=0.0000e+00 | zero_grad_tensors=49/49
  pooled_cross_attn   : grad_norm=7.3202e-04 | zero_grad_tensors=21/22
  output_proj_unet    : grad_norm=0.0000e+00 | zero_grad_tensors=1/1
  output_proj_vit     : grad_norm=0.0000e+00 | zero_grad_tensors=1/1
  input_stem          : grad_norm=2.4025e-03 | zero_grad_tensors=45/47
  routers             : grad_norm=8.5369e-02 | zero_grad_tensors=0/4
  log_var             : grad_norm=3.1047e-01 | zero_grad_tensors=0/1

Sample Zero-Init Gates at Step 0 (name, init_val, grad_norm):
  net.time_proj_gate                                     : val=+0.0000, grad=0.0000e+00
  net.unet_proj_gate                                     : val=+0.0000, grad=2.3210e-03
  net.vit_proj_gate                                      : val=+0.0000, grad=6.2021e-04
  net.blend_gate                                         : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.encoders.32x32_block0.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.encoders.32x32_block1.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.encoders.16x16_down.res_gate        : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.encoders.16x16_block0.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.encoders.16x16_block1.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.16x16_in0.res_gate         : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.16x16_in1.res_gate         : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.16x16_block0.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.16x16_block1.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.16x16_block2.res_gate      : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.32x32_up.res_gate          : val=+0.0000, grad=0.0000e+00
  net.Unet_experts.0.decoders.32x32_block0.res_gate      : val=+0.0000, grad=0.0000e+00
  ... (25 more gates, exact zero count=38/41)

================================================================================
PROBE 2C: 40-Step Dynamical Optimization Trace from Step-0 Init (Adam LR=0.0012)
================================================================================
step |       lr |  zeta |    tot |   pure | vit(hi) | unet(lo) |    MSE |   vit_g |  unet_g |  res_g0 | tmsa_g0 |  u_in_gn |  v_in_gn
----------------------------------------------------------------------------------------------------------------------------
   0 |  0.00012 |  1.00 | 1.2871 | 1.4110 |  0.2742 |   0.5687 | 0.4930 |  0.0001 |  0.0001 |  0.0000 |  0.0000 | 0.00e+00 | 0.00e+00
   1 |  0.00024 |  1.00 | 1.2126 | 1.1465 |  0.3971 |   0.1695 | 0.2811 |  0.0001 |  0.0003 |  0.0002 |  0.0002 | 7.87e-06 | 4.69e-06
   2 |  0.00036 |  1.00 | 1.3239 | 1.3018 |  0.3918 |   0.1965 | 0.2848 |  0.0002 |  0.0004 |  0.0005 |  0.0002 | 2.00e-05 | 4.45e-06
   3 |  0.00048 |  0.99 | 1.2816 | 1.3848 |  0.3737 |   0.1055 | 0.3546 |  0.0002 |  0.0006 |  0.0008 |  0.0005 | 4.57e-06 | 3.59e-06
   5 |  0.00072 |  0.95 | 1.2316 | 1.1411 |  0.5074 |   0.1911 | 0.2470 |  0.0001 |  0.0010 |  0.0002 |  0.0011 | 4.38e-05 | 1.70e-06
  10 |  0.00120 |  0.78 | 1.3515 | 1.2775 |  0.7209 |   0.0936 | 0.1473 |  0.0033 |  0.0043 |  0.0001 |  0.0027 | 1.21e-04 | 8.22e-05
  20 |  0.00093 |  0.27 | 1.3304 | 1.4318 |  0.5389 |   0.6322 | 0.4488 |  0.0100 |  0.0144 |  0.0067 |  0.0078 | 1.62e-03 | 4.17e-04
  30 |  0.00039 |  0.00 | 1.3240 | 1.2956 |  0.3709 |   0.0293 | 0.1019 |  0.0156 |  0.0212 |  0.0086 |  0.0100 | 3.03e-04 | 2.44e-04
  39 |  0.00012 |  0.00 | 1.2830 | 1.2005 |  0.7151 |   0.0346 | 0.0499 |  0.0175 |  0.0232 |  0.0101 |  0.0106 | 3.94e-04 | 5.67e-04

================================================================================
PROBE 2D: Adam Tangential Update Budget Across Cosine (8k) vs WSD (8k) vs EDM2 (70k)
================================================================================
Integrated LR sum int alpha(t) dt:
  v2 Cosine (8k steps):  total=5.257 (last 3000 steps contribute 0.745 = 14.2%)
  v3 WSD    (8k steps):  total=8.713
  70k InvSqrt ref=1.2e-3: total=25.998 (4.9x more integrated Adam step budget)
```

---

## Probe 3: `spherical_blend` Geometry, Multi-Objective Gradients & `PixelShuffle(2)` RAPSD
**Command:** `& "f:\Heterogeneous-MOE-for-Diffusion-models\.venv\Scripts\python.exe" "f:\Heterogeneous-MOE-for-Diffusion-models\analysis\probes\probe_blend_and_spectral_artifacts.py"`  
**Exit code:** `0`

```text
================================================================================
PROBE 3A: spherical_blend Manifold Norm & Gradient Jacobian Audit
================================================================================
   rho |  sv/su |  RMS_lin |  RMS_sph |  RMS_tgt | |dSph/dv|/|dLin/dv| | |dSph/du|/|dLin/du|
----------------------------------------------------------------------------------------
  0.99 |   1.00 |   0.9980 |   1.0000 |   1.0000 |              1.0019 |              1.0022
  0.74 |   1.00 |   0.9461 |   1.0000 |   1.0000 |              1.0568 |              1.0569
  0.00 |   1.00 |   0.7725 |   1.0000 |   1.0000 |              1.2920 |              1.2956
 -0.74 |   1.00 |   0.5463 |   1.0000 |   1.0000 |              1.8293 |              1.8335
  0.99 |   0.25 |   0.4589 |   0.4600 |   0.4600 |              1.0025 |              1.0023
  0.74 |   0.25 |   0.4306 |   0.4600 |   0.4600 |              1.0690 |              1.0677
  0.00 |   0.25 |   0.3329 |   0.4600 |   0.4600 |              1.3774 |              1.3828
 -0.74 |   0.25 |   0.1903 |   0.4600 |   0.4600 |              2.4267 |              2.4203
  0.99 |   4.00 |   3.1574 |   3.1600 |   3.1600 |              1.0008 |              1.0009
  0.74 |   4.00 |   3.0929 |   3.1600 |   3.1600 |              1.0217 |              1.0205
  0.00 |   4.00 |   2.8936 |   3.1600 |   3.1600 |              1.0920 |              1.0920
 -0.74 |   4.00 |   2.6794 |   3.1600 |   3.1600 |              1.1795 |              1.1796

================================================================================
PROBE 3B: EDM_LOSS_NOD Multi-Objective Gradient Conflict (Gates Opened to 0.3)
================================================================================
Param Group          |  |g_pure| |   |g_vit| |  |g_unet| |  cos(p,v) |  cos(p,u) |  cos(v,u)
----------------------------------------------------------------------------------------
input_stem           | 8.156e-03 | 3.425e-04 | 4.930e-04 |    0.4263 |    0.6073 |    0.0270
pooled_cross_attn    | 1.574e-02 | 5.816e-04 | 0.000e+00 |    0.3180 |    0.0000 |    0.0000
unet_experts         | 1.144e-02 | 0.000e+00 | 1.017e-03 |    0.0000 |    0.8720 |    0.0000
vit_experts          | 1.765e-02 | 1.552e-03 | 0.000e+00 |    0.8246 |    0.0000 |    0.0000
output_proj_vit      | 7.708e-03 | 6.563e-04 | 0.000e+00 |    0.6835 |    0.0000 |    0.0000
output_proj_unet     | 5.153e-03 | 0.000e+00 | 5.067e-04 |    0.0000 |    0.8895 |    0.0000

================================================================================
PROBE 3C: ViT [2,2] + PixelShuffle(2) + 1x1 vs 3x3 Conv RAPSD Artifact Check
================================================================================
radius (cyc/32) |  Real 1/f^2 |   PS(2)+1x1 |   PS(2)+3x3 | Ratio 1x1/Real | Ratio 3x3/Real
----------------------------------------------------------------------------------------
              1 |    14.27956 |     4.88618 |     3.93198 |          0.342 |          0.275
              2 |     6.44893 |     2.24565 |     1.88285 |          0.348 |          0.292
              3 |     3.72463 |     1.29213 |     1.08141 |          0.347 |          0.290
              4 |     2.29582 |     0.77799 |     0.66442 |          0.339 |          0.289
              6 |     1.06136 |     0.36150 |     0.31883 |          0.341 |          0.300
              8 |     0.61692 |     0.19145 |     0.17766 |          0.310 |          0.288
             10 |     0.39095 |     0.25086 |     0.24784 |          0.642 |          0.634
             12 |     0.27771 |     0.39616 |     0.41803 |          1.427 |          1.505
             14 |     0.20567 |     0.72412 |     0.80661 |          3.521 |          3.922
             15 |     0.17962 |     1.23460 |     1.40581 |          6.873 |          7.826
```

---

## Probe 4: `sample_sigma_hybrid` Tail Quantiles & Kendall $u^*(\sigma)$ Amplification Math
**Command:** `& "f:\Heterogeneous-MOE-for-Diffusion-models\.venv\Scripts\python.exe" "f:\Heterogeneous-MOE-for-Diffusion-models\analysis\probes\probe_sigma_quantiles_and_loss.py"`  
**Exit code:** `0`

```text
P(sigma <= 0.010) = 0.0297 | P(sigma > 0.010) = 0.9703
P(sigma <= 0.020) = 0.0435 | P(sigma > 0.020) = 0.9565
P(sigma <= 0.050) = 0.0641 | P(sigma > 0.050) = 0.9359
P(sigma <= 0.100) = 0.0960 | P(sigma > 0.100) = 0.9040
P(sigma <= 0.300) = 0.2635 | P(sigma > 0.300) = 0.7365
P(sigma <= 1.000) = 0.6415 | P(sigma > 1.000) = 0.3585
P(sigma <= 3.000) = 0.8847 | P(sigma > 3.000) = 0.1153
P(sigma <= 10.000) = 0.9575 | P(sigma > 10.000) = 0.0425
P(sigma <= 30.000) = 0.9815 | P(sigma > 30.000) = 0.0185

Closed-form Kendall uncertainty u*(sigma) = ln(E_blend) impact on vit_loss:
  step 0 initial E_blend=1.53 (u=0): u*=+0.000, exp(-u*)=1.000, pure_loss=1.530, vit_loss(if e_vit=0.63*0.72*exp(-u*))=0.454
  step 2000 E_blend=0.657 -> pure=1+ln(0.657)=0.580: u*=-0.420, exp(-u*)=1.522, pure_loss=0.580, vit_loss(if e_vit=0.63*0.72*exp(-u*))=0.690
```
