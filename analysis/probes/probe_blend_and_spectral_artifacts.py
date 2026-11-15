"""Read-only diagnostic probe 3: spherical_blend geometry, EDM_LOSS_NOD gradient conflict, and ViT PixelShuffle RAPSD."""
import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_config_v3 import preconditioned_HDMOEM_v3, spherical_blend
from models.model_internals import MP_Conv
from Utils.loss import EDM_LOSS_NOD
from Utils.utils import sample_sigma_hybrid


def make_1f_latents(batch: int, channels: int, size: int, alpha: float = 2.0, target_std: float = 0.8, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    fx = torch.fft.fftfreq(size)
    fy = torch.fft.rfftfreq(size)
    ky, kx = torch.meshgrid(fx, fy, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k0_sq = (1.0 / size) ** 2
    amp = (k2 + k0_sq) ** (-alpha / 4.0)
    amp[0, 0] = 0.0
    spec = torch.complex(
        torch.randn(batch, channels, size, fy.shape[0], generator=g),
        torch.randn(batch, channels, size, fy.shape[0], generator=g),
    ) * amp
    x = torch.fft.irfft2(spec, s=(size, size), norm="ortho")
    x = x / (x.std(dim=(-2, -1), keepdim=True) + 1e-8) * target_std
    return x


def compute_rapsd(img: torch.Tensor) -> torch.Tensor:
    """Compute azimuthally averaged 2D power spectrum at radii 0..H//2-1."""
    B, C, H, W = img.shape
    fft2 = torch.fft.fftshift(torch.fft.fft2(img, norm="ortho"), dim=(-2, -1))
    psd2 = fft2.abs().pow(2).mean(dim=(0, 1))
    cy, cx = H // 2, W // 2
    y = torch.arange(H) - cy
    x = torch.arange(W) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rr = torch.sqrt(xx.float() ** 2 + yy.float() ** 2).round().long()
    out = []
    for r in range(H // 2):
        mask = (rr == r)
        out.append(psd2[mask].mean().item() if mask.any() else 0.0)
    return torch.tensor(out)


def build_compact_model(vit_patch=None, sigma_data=0.8):
    if vit_patch is None:
        vit_patch = [2, 2]
    return preconditioned_HDMOEM_v3(
        IN_in_channels=4, IN_img_resolution=32, internal_channels=32,
        time_emb_dim=128, text_emb_dim=64, num_experts=2, top_k=1,
        Fourier_bandwidth=1.0, VIT_num_blocks=2, VIT_patch_sizes=vit_patch,
        VIT_num_groups=4, VIT_num_heads=4, VIT_emb_size=64,
        VIT_pos_emb_balance=[0.1, 0.1], VIT_emb_balance=[0.3, 0.3],
        Unet_num_blocks=2, Unet_channel_mult=[1, 2], Unet_kernel_sizes=[(3, 3), (3, 3)],
        Unet_model_channels=32, Unet_channel_mult_emb=2, Unet_label_balance=0.5,
        sigma_data=sigma_data, log_var_channels=64, transition_sigma=3.0,
        transition_sharpness=1.5, pool_size=4, attn_pool_balance=1.0,
        p_mean=-0.4, p_std=1.2,
    )


def get_group_grad_vec(model, group_fn):
    vecs = []
    for n, p in model.named_parameters():
        if group_fn(n):
            g = p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p.data.flatten())
            vecs.append(g)
    return torch.cat(vecs) if vecs else torch.zeros(1)


def main():
    torch.manual_seed(0)
    print("=" * 80)
    print("PROBE 3A: spherical_blend Manifold Norm & Gradient Jacobian Audit")
    print("=" * 80)
    # Construct unit/orthogonal fields of dim 4*32*32=4096
    e1 = torch.randn(4, 4, 32, 32)
    e1 = e1 / e1.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
    e2_raw = torch.randn(4, 4, 32, 32)
    # Orthogonalize e2_raw wrt e1 per batch sample
    dot = (e1 * e2_raw).sum(dim=(1, 2, 3), keepdim=True) / (e1 * e1).sum(dim=(1, 2, 3), keepdim=True)
    e2 = e2_raw - dot * e1
    e2 = e2 / e2.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()

    wv = torch.full((4, 1, 1, 1), 0.72)
    wu = torch.full((4, 1, 1, 1), 0.28)
    print(f"{'rho':>6} | {'sv/su':>6} | {'RMS_lin':>8} | {'RMS_sph':>8} | {'RMS_tgt':>8} | {'|dSph/dv|/|dLin/dv|':>19} | {'|dSph/du|/|dLin/du|':>19}")
    print("-" * 88)
    for ratio in [1.0, 0.25, 4.0]:
        for rho in [0.99, 0.74, 0.0, -0.74]:
            v = (ratio * e1).clone().detach().requires_grad_(True)
            u_vec = (rho * e1 + math.sqrt(max(0.0, 1.0 - rho ** 2)) * e2).clone().detach().requires_grad_(True)
            target = torch.randn_like(v)

            lin = wv * v + wu * u_vec
            l_lin = (lin - target).pow(2).sum()
            gv_lin, gu_lin = torch.autograd.grad(l_lin, [v, u_vec])

            sph = spherical_blend(v, u_vec, wv, wu)
            l_sph = (sph - target).pow(2).sum()
            gv_sph, gu_sph = torch.autograd.grad(l_sph, [v, u_vec])

            sv = v.pow(2).mean().sqrt().item()
            su = u_vec.pow(2).mean().sqrt().item()
            rms_tgt = (wv * sv + wu * su).mean().item()
            rms_lin = lin.pow(2).mean().sqrt().item()
            rms_sph = sph.pow(2).mean().sqrt().item()
            rv = gv_sph.norm().item() / (gv_lin.norm().item() + 1e-12)
            ru = gu_sph.norm().item() / (gu_lin.norm().item() + 1e-12)
            print(f"{rho:6.2f} | {ratio:6.2f} | {rms_lin:8.4f} | {rms_sph:8.4f} | {rms_tgt:8.4f} | {rv:19.4f} | {ru:19.4f}")

    print("\n" + "=" * 80)
    print("PROBE 3B: EDM_LOSS_NOD Multi-Objective Gradient Conflict (Gates Opened to 0.3)")
    print("=" * 80)
    model = build_compact_model(vit_patch=[2, 2], sigma_data=0.8)
    model.train()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "gate" in n.lower():
                p.fill_(0.3)

    vit_router = getattr(model.net, "vit_router", None) or getattr(model.net, "VIT_router", None)
    unet_router = getattr(model.net, "Unet_router", None) or getattr(model.net, "unet_router", None)
    loss_fn = EDM_LOSS_NOD(
        expert_centers_vit=vit_router.centers.detach().clone(),
        expert_centers_Unet=unet_router.centers.detach().clone(),
        num_experts=2, sigma_data=0.8, alpha_blend=0.8, alpha_unet=0.1, alpha_vit=0.1,
        lap_sigma=0.5, transition_sigma=3.0,
    )

    x0 = make_1f_latents(32, 4, 32, alpha=2.0, target_std=0.8, seed=5)
    sig = sample_sigma_hybrid(32, device=torch.device("cpu"))
    xn = x0 + sig.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
    txt = torch.randn(32, 64)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    m_u = torch.ones(32, 2)
    m_v = torch.ones(32, 2)

    groups = {
        "input_stem": lambda n: any(k in n for k in ["input_proj", "Fourier_emb", "out_fourier", "input_time_proj"]),
        "pooled_cross_attn": lambda n: "cross_attn" in n.lower() or "output_proj_attn" in n.lower(),
        "unet_experts": lambda n: "unet_experts" in n.lower(),
        "vit_experts": lambda n: "vit_experts" in n.lower(),
        "output_proj_vit": lambda n: "output_proj_vit" in n.lower(),
        "output_proj_unet": lambda n: "output_proj_unet" in n.lower(),
    }

    # Evaluate separately for pure_loss, vit_loss (scaled 0.1 vs 1.0), unet_loss
    grads = {obj: {} for obj in ["pure", "vit", "unet"]}
    for obj_key in ["pure", "vit", "unet"]:
        model.zero_grad()
        out = model(xn, sig, txt, m_u, m_v, zeta=0.0, return_log_var=True)
        ldict = loss_fn(x0, xn, sig, out)
        loss_map = {"pure": 0.8 * ldict["pure_loss"], "vit": 0.1 * ldict["loss_texture"], "unet": 0.1 * ldict["loss_structure"]}
        loss_map[obj_key].backward()
        for gname, gfn in groups.items():
            grads[obj_key][gname] = get_group_grad_vec(model, gfn)

    print(f"{'Param Group':20s} | {'|g_pure|':>9} | {'|g_vit|':>9} | {'|g_unet|':>9} | {'cos(p,v)':>9} | {'cos(p,u)':>9} | {'cos(v,u)':>9}")
    print("-" * 88)
    for gname in groups:
        gp, gv, gu = grads["pure"][gname], grads["vit"][gname], grads["unet"][gname]
        np_, nv, nu = gp.norm().item(), gv.norm().item(), gu.norm().item()
        cpv = F.cosine_similarity(gp.unsqueeze(0), gv.unsqueeze(0)).item() if (np_ > 0 and nv > 0) else 0.0
        cpu = F.cosine_similarity(gp.unsqueeze(0), gu.unsqueeze(0)).item() if (np_ > 0 and nu > 0) else 0.0
        cvu = F.cosine_similarity(gv.unsqueeze(0), gu.unsqueeze(0)).item() if (nv > 0 and nu > 0) else 0.0
        print(f"{gname:20s} | {np_:9.3e} | {nv:9.3e} | {nu:9.3e} | {cpv:9.4f} | {cpu:9.4f} | {cvu:9.4f}")

    print("\n" + "=" * 80)
    print("PROBE 3C: ViT [2,2] + PixelShuffle(2) + 1x1 vs 3x3 Conv RAPSD Artifact Check")
    print("=" * 80)
    # Create smooth correlated latent patch features [B, 4*C_tok, 16, 16] vs correlated spatial 32x32 -> patchified
    real_32 = make_1f_latents(256, 4, 32, alpha=2.0, target_std=0.8, seed=10)
    # Smooth 16x16 features projected via PixelShuffle(2) + 1x1 vs 3x3 MP_Conv
    feats_16 = make_1f_latents(256, 64, 16, alpha=2.0, target_std=1.0, seed=20)
    ps = nn.PixelShuffle(2)
    y_ps = ps(feats_16)  # [B, 16, 32, 32]
    conv1 = MP_Conv(16, 4, kernel=(1, 1))
    conv3 = MP_Conv(16, 4, kernel=(3, 3))
    with torch.no_grad():
        out_1x1 = conv1(y_ps)
        out_3x3 = conv3(y_ps)
    # Normalize total variance to real_32 variance so RAPSD shapes compare directly across radii 1..15
    target_var = real_32.var().item()
    out_1x1 = out_1x1 * math.sqrt(target_var / (out_1x1.var().item() + 1e-12))
    out_3x3 = out_3x3 * math.sqrt(target_var / (out_3x3.var().item() + 1e-12))

    r_real = compute_rapsd(real_32)
    r_1x1 = compute_rapsd(out_1x1)
    r_3x3 = compute_rapsd(out_3x3)
    print(f"{'radius (cyc/32)':>15} | {'Real 1/f^2':>11} | {'PS(2)+1x1':>11} | {'PS(2)+3x3':>11} | {'Ratio 1x1/Real':>14} | {'Ratio 3x3/Real':>14}")
    print("-" * 88)
    for r in [1, 2, 3, 4, 6, 8, 10, 12, 14, 15]:
        print(f"{r:15d} | {r_real[r]:11.5f} | {r_1x1[r]:11.5f} | {r_3x3[r]:11.5f} | {r_1x1[r]/(r_real[r]+1e-12):14.3f} | {r_3x3[r]/(r_real[r]+1e-12):14.3f}")


if __name__ == "__main__":
    main()
