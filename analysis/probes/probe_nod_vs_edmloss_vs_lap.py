"""Dynamic comparison probe: Fixed NOD (no Laplacian) vs Laplacian NOD (text_guided style) vs Pure EDM_LOSS (blend-only)."""
import copy
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_config_v3 import preconditioned_HDMOEM_v3


def rfft_laplacian_decomp(image: torch.Tensor, sigma: float = 1.0):
    h, w = image.shape[-2:]
    cutoff_sq = math.log(2.0) / (4.0 * (math.pi ** 2) * max(float(sigma), 1e-6) ** 2)
    fy = torch.fft.fftfreq(h, device=image.device, dtype=torch.float32).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=image.device, dtype=torch.float32).view(1, -1)
    low_mask = ((fy * fy + fx * fx) <= cutoff_sq).to(torch.float32)
    low_freq = torch.fft.irfft2(torch.fft.rfft2(image.float(), norm="ortho") * low_mask, s=(h, w), norm="ortho")
    return low_freq, image - low_freq


def make_1f2_latents(n: int, c: int = 4, h: int = 16, w: int = 16, sigma_data: float = 0.8) -> torch.Tensor:
    fx = torch.fft.fftfreq(h)
    fy = torch.fft.rfftfreq(w)
    ky, kx = torch.meshgrid(fx, fy, indexing="ij")
    amp = (kx ** 2 + ky ** 2 + (1.0 / h) ** 2) ** (-0.5)
    amp[0, 0] = 0.0
    spec = torch.complex(torch.randn(n, c, h, fy.shape[0]), torch.randn(n, c, h, fy.shape[0])) * amp
    x0 = torch.fft.irfft2(spec, s=(h, w), norm="ortho")
    x0 = x0 / (x0.std(dim=(-2, -1), keepdim=True) + 1e-8) * sigma_data
    return x0


def sample_hybrid_sigma(b: int, p_mean: float = -0.4, p_std: float = 1.2, sigma_min: float = 0.002, sigma_max: float = 80.0):
    u = torch.rand(b)
    log_sig_unif = math.log(sigma_min) + torch.rand(b) * (math.log(sigma_max) - math.log(sigma_min))
    log_sig_norm = p_mean + p_std * torch.randn(b)
    sig = torch.where(u < 0.2, torch.exp(log_sig_unif), torch.exp(log_sig_norm)).clamp(sigma_min, sigma_max)
    return sig


def compute_loss(mode: str, x: torch.Tensor, sigma: torch.Tensor, out: dict, sigma_data: float = 0.8):
    sig4 = sigma.reshape(-1, 1, 1, 1)
    weight = (sig4 ** 2 + sigma_data ** 2) / (sig4 * sigma_data) ** 2
    diff_blend = out["denoised"] - x
    log_var = out["log_var"].clamp(-10, 10)
    pure_loss = torch.mean((weight * (diff_blend ** 2)) / log_var.exp() + log_var)

    if mode == "edm_loss_blend_only":
        return pure_loss, pure_loss.item(), (weight * (diff_blend ** 2)).mean().item()

    w_vit = out["w_vit"].detach()
    w_unet = out["w_unet"].detach()

    if mode == "nod_no_laplacian_fixed":
        unet_loss = torch.mean(weight * w_unet * ((out["unet_denoised"] - x) ** 2))
        vit_loss = torch.mean(weight * w_vit * ((out["vit_denoised"] - x) ** 2))
        total = 0.6 * pure_loss + 0.2 * unet_loss + 0.2 * vit_loss
        return total, pure_loss.item(), (weight * (diff_blend ** 2)).mean().item()

    if mode == "nod_laplacian_text_guided":
        u_scale = (1.0 / log_var.exp()).detach()
        unet_low, _ = rfft_laplacian_decomp(out["unet_denoised"] - x, sigma=1.0)
        _, vit_high = rfft_laplacian_decomp(out["vit_denoised"] - x, sigma=1.0)
        unet_loss = torch.mean(weight * u_scale * w_unet * (unet_low ** 2))
        vit_loss = torch.mean(weight * u_scale * w_vit * (vit_high ** 2))
        total = 0.6 * pure_loss + 0.2 * unet_loss + 0.2 * vit_loss
        return total, pure_loss.item(), (weight * (diff_blend ** 2)).mean().item()

    raise ValueError(mode)


def main():
    torch.manual_seed(42)
    sigma_data = 0.8
    base_model = preconditioned_HDMOEM_v3(
        IN_in_channels=4,
        IN_img_resolution=16,
        internal_channels=32,
        time_emb_dim=32,
        text_emb_dim=32,
        num_experts=2,
        top_k=1,
        Fourier_bandwidth=1.2,
        VIT_num_blocks=2,
        VIT_patch_sizes=[2, 2],
        VIT_num_groups=4,
        VIT_num_heads=4,
        VIT_emb_size=64,
        VIT_pos_emb_balance=[0.5, 0.5],
        VIT_emb_balance=[0.4, 0.4],
        Unet_num_blocks=1,
        Unet_channel_mult=[1, 2],
        Unet_kernel_sizes=[(3, 3), (5, 5)],
        Unet_model_channels=32,
        Unet_channel_mult_emb=4,
        Unet_label_balance=0.5,
        Unet_concat_balance=[0.7, 0.3],
        sigma_data=sigma_data,
        log_var_channels=16,
        transition_sigma=3.0,
        transition_sharpness=0.5,
        pool_size=4,
        attn_pool_balance=0.5,
        p_mean=-0.4,
        p_std=1.2,
        sigma_min=0.002,
        sigma_max=80.0,
    )

    # Pre-generate fixed training stream & fixed validation probes
    n_steps = 80
    B = 16
    train_batches = []
    for s in range(n_steps):
        g = torch.Generator().manual_seed(1000 + s)
        x0 = make_1f2_latents(B, 4, 16, 16, sigma_data)
        sig = sample_hybrid_sigma(B)
        eps = torch.randn(x0.shape, generator=g)
        txt = torch.randn(B, 4, 32, generator=g)
        train_batches.append((x0, sig, eps, txt))

    val_x0 = make_1f2_latents(32, 4, 16, 16, sigma_data)
    val_txt = torch.randn(32, 4, 32)
    val_eps = torch.randn_like(val_x0)
    mask_all = torch.ones(B, 2)
    val_mask = torch.ones(32, 2)

    modes = ["nod_no_laplacian_fixed", "nod_laplacian_text_guided", "edm_loss_blend_only"]
    results = {}

    for mode in modes:
        model = copy.deepcopy(base_model)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0)
        for step, (x0, sig, eps, txt) in enumerate(train_batches):
            sig4 = sig.reshape(-1, 1, 1, 1)
            xn = x0 + sig4 * eps
            out = model(xn, sig, txt, mask_all, mask_all, zeta=0.0, return_log_var=True)
            loss, p_val, mse_val = compute_loss(mode, x0, sig, out, sigma_data)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            metrics = {}
            for label, s_val in [("low_0.05", 0.05), ("mid_0.8", 0.8), ("overlap_3.0", 3.0), ("high_20.0", 20.0)]:
                sv = torch.full((32,), s_val)
                sv4 = sv.reshape(-1, 1, 1, 1)
                xn = val_x0 + sv4 * val_eps
                out = model(xn, sv, val_txt, val_mask, val_mask, zeta=0.0, return_log_var=True)
                lam = (sv4 ** 2 + sigma_data ** 2) / (sv4 * sigma_data) ** 2
                wmse_blend = (lam * (out["denoised"] - val_x0) ** 2).mean().item()
                wmse_vit = (lam * (out["vit_denoised"] - val_x0) ** 2).mean().item()
                wmse_unet = (lam * (out["unet_denoised"] - val_x0) ** 2).mean().item()
                vel_rms = ((xn - out["denoised"]) / sv4).pow(2).mean().sqrt().item()
                c_skip = sigma_data ** 2 / (sv4 ** 2 + sigma_data ** 2)
                dv = (out["vit_denoised"] - c_skip * xn).flatten(1)
                du = (out["unet_denoised"] - c_skip * xn).flatten(1)
                cos = F.cosine_similarity(dv, du, dim=1).mean().item()
                metrics[label] = (wmse_blend, wmse_vit, wmse_unet, vel_rms, cos)
            results[mode] = metrics

    print("=== 80-step Fixed-Init Dynamic Comparison (1/f^2 latents, HDMOEM_v3) ===")
    for mode in modes:
        print(f"\n[{mode}]")
        for label, (wb, wv, wu, vrms, cos) in results[mode].items():
            print(f"  {label:11s} | blend_wMSE={wb:.4f} | vit_wMSE={wv:.4f} | unet_wMSE={wu:.4f} | vel_RMS={vrms:.4f} | branch_cos={cos:.4f}")


if __name__ == "__main__":
    main()
