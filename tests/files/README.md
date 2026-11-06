# HDMOEM v3 test suite

## How to run

Place this `tests/` folder at the root of the repo, next to your `models/`
package (the same layout `model_config_v3.py`'s own `from models.model_components
import ...` already assumes):

```
your_repo/
  models/
    __init__.py
    model_components.py
    model_config_v3.py
    model_internals.py
  tests/
    __init__.py
    model_factory.py
    test_edm_preconditioning.py
    test_magnitude_preservation.py
    test_router.py
    test_tgff.py
    test_gradient_flow.py
    test_numerical_stability.py
    test_train_inference_modes.py
```

Run the whole suite from the repo root:

```
python -m unittest discover -s tests -v
```

Run one file:

```
python -m unittest tests.test_gradient_flow -v
```

All tests use a shared tiny config (`tests/model_factory.py`) — 32 internal
channels, 2 experts per branch, 16×16 latents — chosen only for speed
(the full suite runs in ~5-6 seconds on CPU). It is not a stand-in for
testing your real training-size config; treat this as the fast suite you
run on every change, and periodically re-run the gradient-flow file
against your actual training config before a real run.

## Run this file first, always

**`test_gradient_flow.py`** — before starting any training run, on any
architecture change. This is the file that catches a parameter group
receiving permanently zero gradient (e.g. two zero-initialized layers
stacked in series with no alternate path). It found exactly that bug in
the ViT branch and the pooled cross-attention branch when run against the
pre-fix code; verified clean against the fixed code. If you change any
`nn.init.constant_(..., 0.0)` call anywhere in the model, re-run this file.

## Coverage matrix

| File | What it locks down |
|---|---|
| `test_edm_preconditioning.py` | c_skip/c_out/c_in/c_noise match the closed-form EDM (Karras et al. 2022) equations exactly; correct behavior at the sigma→0 and sigma→∞ limits; published hyperparameter defaults (sigma_data=0.5, sigma_min=0.002, sigma_max=80) |
| `test_magnitude_preservation.py` | EDM2 (Karras et al. 2024) magnitude-preserving primitives (`mp_silu`, `mp_sum`, `mp_cat`, `normalize`, `MP_Conv`) hold unit variance in isolation; `E_unified` stays unit-variance across the full sigma range; `x_feats` scales with `c_in(sigma)` as EDM specifies; resolution-adaptive frequency cutoff behaves correctly at multiple resolutions; no NaN/Inf anywhere in a forward pass |
| `test_router.py` | `ParametricTrajectoryRouter` outputs a valid probability simplex; masking zeroes the intended experts exactly; UNet-router and ViT-router centers are seeded in disjoint CDF ranges (the mechanism the noise-level specialization depends on); `zeta` exploration noise is inert outside `.train()` mode; sharpness stays positive under gradient pressure |
| `test_tgff.py` | Low/high-pass masks are exact complements; the resolution-adaptive cutoff fix; `zero_conv` truly gates the ViT contribution to exactly 0 at init (independent of what ViT produces); `eta` stays in [0, 1]; a genuinely low-frequency signal survives the FFT round trip with high fidelity |
| `test_gradient_flow.py` | **The critical file.** Zero-init layers start at exact 0; model output equals the pure EDM skip term at init; no NaN/Inf gradients; every named architectural region receives nonzero gradient within a small step budget (the general regression test); the ViT branch and pooled-cross-attention branch specifically are not deadlocked (named regression tests for the exact bug found) |
| `test_numerical_stability.py` | Batch size 1; sigma_min/sigma_max boundary values; a 20-point log-spaced sweep across the full noise range; per-sample coefficient isolation in a mixed-sigma batch; an all-experts-masked-out edge case doesn't produce NaN; `return_log_var` path is finite |
| `test_train_inference_modes.py` | `.eval()` is fully deterministic even if `zeta > 0` is passed; `.train()` with `zeta > 0` is (correctly) stochastic across seeds; `zeta = 0.0` is deterministic in both modes; switching `.train()` → `.eval()` doesn't corrupt state |

## What this suite deliberately does NOT test

- **Whether the router *learns* to specialize by noise level.** That's an
  outcome of training data and loss, not something a pre-training unit
  test can assert. What's tested is that the router's *mechanism* (CDF
  seeding, masking, probability normalization) is correct, so that if
  training does produce specialization, nothing here is silently broken.
- **Sample quality / FID / perceptual metrics.** Out of scope for a unit
  test suite — these need an eval harness with a trained checkpoint.
- **Your real training-size config.** The tiny config here is for speed.
  Re-run at least `test_gradient_flow.py` and `test_magnitude_preservation.py`
  against your actual `IN_img_resolution` / `internal_channels` / `num_experts`
  before a real run, since some issues (e.g. the ~0.5 std UNet branch
  finding) could behave differently at a different scale.

## Verification

This suite was run against both the pre-fix and post-fix versions of the
three model files as a sanity check on the suite itself, not just the
model:

```
BEFORE (as uploaded, output_proj_vit and PooledCrossAttn.out_proj both
zero-initialized): 46 passed, 3 failed — all 3 failures in
test_gradient_flow.py, correctly identifying the ViT branch and the
pooled cross-attention branch as deadlocked.

AFTER (output_proj_vit and PooledCrossAttn.out_proj restored to normal
init, tgff.zero_conv and output_proj_attn remaining as the sole zero-init
gates): 49 passed, 0 failed.
```
