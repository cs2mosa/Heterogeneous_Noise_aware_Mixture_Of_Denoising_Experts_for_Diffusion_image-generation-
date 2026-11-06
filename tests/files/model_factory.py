"""
Shared test fixtures for the HDMOEM v3 test suite.

Every test module builds its model through `build_tiny_model()` rather than
instantiating `preconditioned_HDMOEM_v3` directly. This keeps all tests
using one canonical small-but-representative configuration, so a passing
suite means something consistent, and keeps the suite fast enough to run
on every commit (CPU, sub-second per test).

Tiny-model dimensions are chosen to satisfy every architectural constraint
in the codebase simultaneously:
    - internal_channels (32) % VIT_num_heads (4) == 0   -> PooledCrossAttn enabled
    - VIT_emb_size (32) % VIT_num_heads (4) == 0
    - IN_img_resolution (16) is divisible by every VIT_patch_size ([2, 4])
    - Unet_channel_mult has 2 levels -> resolution goes 16 -> 8, stays integer
"""
import torch
from models.model_config_v3 import preconditioned_HDMOEM_v3, HDMOEM_v3

DEFAULT_CFG = dict(
    IN_in_channels=4,
    IN_img_resolution=16,
    internal_channels=32,
    time_emb_dim=64,
    text_emb_dim=32,
    num_experts=2,
    top_k=1,
    Fourier_bandwidth=1.0,
    VIT_num_blocks=2,
    VIT_patch_sizes=[2, 4],
    VIT_num_groups=4,
    VIT_num_heads=4,
    VIT_emb_size=32,
    VIT_pos_emb_balance=[0.5, 0.5],
    VIT_emb_balance=[0.5, 0.5],
    Unet_num_blocks=1,
    Unet_channel_mult=[1, 2],
    Unet_kernel_sizes=[(3, 3), (3, 3)],
    Unet_model_channels=16,
    transition_sigma=3.0,
    transition_sharpness=1.5,
    pool_size=2,
    attn_pool_balance=0.5,
    p_mean=-0.4,
    p_std=1.2,
    sigma_min=0.002,
    sigma_max=80.0,
)

# Canonical sigma probes used across many tests: low (deep ViT regime),
# transition handoff point, high (deep UNet regime), plus the EDM
# schedule's exact boundary values.
SIGMA_PROBES = {
    "sigma_min_boundary": DEFAULT_CFG["sigma_min"],
    "low_vit_regime": 0.05,
    "transition": DEFAULT_CFG["transition_sigma"],
    "high_unet_regime": 30.0,
    "sigma_max_boundary": DEFAULT_CFG["sigma_max"],
}


def build_tiny_model(seed=0, **overrides):
    """Instantiate preconditioned_HDMOEM_v3 with the tiny test config."""
    torch.manual_seed(seed)
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    return preconditioned_HDMOEM_v3(**cfg), cfg


def build_tiny_net(seed=0, **overrides):
    """Instantiate the inner HDMOEM_v3 (no EDM preconditioning wrapper)."""
    torch.manual_seed(seed)
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    return HDMOEM_v3(**cfg), cfg


def random_batch(cfg, batch_size=2, sigma_value=10.0, seed=1, all_experts_unmasked=True):
    """
    Build a self-consistent (x, sigma, text_emb, masks) tuple for a forward pass.

    sigma_value may be a float (broadcast to the whole batch) or a 1D
    tensor/list of length batch_size (per-sample sigma).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, cfg["IN_in_channels"],
                     cfg["IN_img_resolution"], cfg["IN_img_resolution"], generator=g)
    if isinstance(sigma_value, (int, float)):
        sigma = torch.full((batch_size,), float(sigma_value))
    else:
        sigma = torch.as_tensor(sigma_value, dtype=torch.float32)
        assert sigma.shape[0] == batch_size
    text_emb = torch.randn(batch_size, 6, cfg["text_emb_dim"], generator=g)
    if all_experts_unmasked:
        Umask = torch.ones(batch_size, cfg["num_experts"])
        Vmask = torch.ones(batch_size, cfg["num_experts"])
    else:
        Umask = None
        Vmask = None
    return x, sigma, text_emb, Umask, Vmask


def named_param_groups(model):
    """
    Returns {name: module_or_parameter} for every architecturally distinct
    piece of the network. Used by gradient-flow tests to check each region
    of the graph independently rather than only checking model.parameters()
    in aggregate (which would hide a single dead branch inside a healthy sum).
    """
    net = model.net
    groups = {
        "input_proj": net.input_proj,
        "input_time_proj": net.input_time_proj,
        "Unet_router": net.Unet_router,
        "vit_router": net.vit_router,
        "Unet_experts": net.Unet_experts,
        "VIT_experts": net.VIT_experts,
        "output_proj_unet": net.output_proj_unet,
        "output_proj_vit": net.output_proj_vit,
        "blend_residual": net.blend_residual,
        "out_gain_unet": net.out_gain_unet,
        "out_gain_vit": net.out_gain_vit,
    }
    if net._cross_attn_enabled:
        groups["pooled_cross_attn.cross_attn"] = net.pooled_cross_attn.cross_attn
        groups["pooled_cross_attn.out_proj"] = net.pooled_cross_attn.out_proj
        groups["output_proj_attn"] = net.output_proj_attn
    return groups


def grad_abs_sum(module_or_param):
    """Sum of |gradient| over a module's parameters, or a single Parameter."""
    if hasattr(module_or_param, "parameters"):
        params = list(module_or_param.parameters())
    else:
        params = [module_or_param]
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().abs().sum().item()
    return total
