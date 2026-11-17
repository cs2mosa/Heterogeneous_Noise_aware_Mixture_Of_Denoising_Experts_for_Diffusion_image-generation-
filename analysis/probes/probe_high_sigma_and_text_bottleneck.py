"""Diagnose the two bottlenecks visible in the Colab hybrid_300m_p12 overfitting plots:
1. High-sigma (2.5, 10, 40) MSE plateau at 0.23-0.31:
   Measure text-conditioning sensitivity of Unet_expert vs Vit_expert at sigma=10, 40,
   and test adding pooled-text conditioning to Unet_expert's emb path (where self.label_balance
   was defined in __init__ but unused in forward!).
2. Low-sigma (0.1) 1000-step delay + RAPSD high-freq attenuation (r=8..15):
   Measure gradient share across sigma bins in EDM_LOSS_NOD and check linear vs spherical
   RAPSD under 64-image overfitting.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_components import Unet_expert, Vit_expert
from models.model_config_v3 import preconditioned_HDMOEM_v3
from models.model_internals import MP_Conv, mp_silu, mp_sum, normalize
from Utils.configs import model_configs, mask_configs, loss_configs
from Utils.utils import sample_sigma_hybrid, MaskGenerator
from Utils.loss import EDM_LOSS_NOD

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def probe_1_text_conditioning_sensitivity():
    print("=" * 78)
    print("PROBE 1: Text-Conditioning Sensitivity Inside Unet_expert vs Vit_expert")
    print("=" * 78)
    torch.manual_seed(0)
    B, C, H, W = 4, 128, 32, 32
    time_dim, text_dim = 512, 768

    unet = Unet_expert(
        img_resolution=H, img_channels=C, time_emb_dim=time_dim, text_emb_dim=text_dim,
        channel_mult=[1, 2, 4], model_channels=96, channel_mult_emb=4, num_blocks=2,
        kernel_size=(3, 3), label_balance=0.5, concat_balance=0.7,
    ).to(DEV).eval()

    vit = Vit_expert(
        num_heads=8, num_groups=4, in_channels=C, seq_ln=H * W, emb_dim=256,
        num_blocks=6, patch_size=1, time_dim=time_dim, text_dim=text_dim,
    ).to(DEV).eval()

    x = torch.randn(B, C, H, W, device=DEV)
    t_emb = torch.randn(B, time_dim, device=DEV)
    txt1 = torch.randn(B, 77, text_dim, device=DEV)
    txt2 = torch.randn(B, 77, text_dim, device=DEV)

    with torch.no_grad():
        u1 = unet(x, t_emb, txt1)
        u2 = unet(x, t_emb, txt2)
        v1 = vit(x, t_emb, txt1)
        v2 = vit(x, t_emb, txt2)

    u_rel = (u1 - u2).norm() / u1.norm()
    v_rel = (v1 - v2).norm() / v1.norm()
    u_cos = F.cosine_similarity(u1.flatten(1), u2.flatten(1), dim=1).mean().item()
    v_cos = F.cosine_similarity(v1.flatten(1), v2.flatten(1), dim=1).mean().item()
    print(f"  Unet_expert (ca_gate=0.1, bottleneck-only): rel_diff(txt1,txt2) = {u_rel.item():.4f}, cos = {u_cos:.4f}")
    print(f"  Vit_expert  (6x cross_attn, ca_gate=0.1):   rel_diff(txt1,txt2) = {v_rel.item():.4f}, cos = {v_cos:.4f}")

    # Check gradient norm to bottleneck_attn vs encoders at sigma=10
    unet.train()
    u_out = unet(x, t_emb, txt1)
    u_out.pow(2).mean().backward()
    g_attn = math.sqrt(sum(p.grad.pow(2).sum().item() for p in unet.bottleneck_attn.parameters() if p.grad is not None))
    g_enc0 = math.sqrt(sum(p.grad.pow(2).sum().item() for k, m in unet.encoders.items() if "32x32" in k for p in m.parameters() if p.grad is not None))
    print(f"  Unet grad norms: bottleneck_attn={g_attn:.4f}, 32x32_encoders={g_enc0:.4f}")


def probe_2_sigma_gradient_share():
    print("\n" + "=" * 78)
    print("PROBE 2: Training Sample Fraction & Gradient Share per Sigma Band")
    print("=" * 78)
    torch.manual_seed(0)
    sigma = sample_sigma_hybrid(
        8192, sigma_min=0.002, sigma_max=80.0,
        p_mean=-0.2, p_std=1.2, extreme_prob=0.2, device="cpu",
    ).flatten()
    bins = [(0.002, 0.05), (0.05, 0.25), (0.25, 1.0), (1.0, 2.5), (2.5, 10.0), (10.0, 80.0)]
    for lo, hi in bins:
        m = (sigma >= lo) & (sigma < hi)
        frac = m.float().mean().item()
        print(f"  sigma in [{lo:5.3f}, {hi:5.2f}): {frac:6.2%} of training batch  (~{frac*32:4.1f} imgs per 32-batch)")


def probe_3_high_sigma_memorization_ablation():
    """Compare Unet_expert memorization at sigma=10..40:
       (A) Current Unet_expert (text only at 8x8 SpatialCrossAttn with ca_gate=0.1)
       (B) Unet_expert + pooled text embedding mixed into time_emb via label_balance (standard EDM2/ADM)
    """
    print("\n" + "=" * 78)
    print("PROBE 3: High-Sigma (sigma in [2.5, 40]) 64-Image Memorization on Unet_expert")
    print("         Current (bottleneck cross-attn only) vs + Pooled Text-in-Emb")
    print("=" * 78)
    torch.manual_seed(42)
    N, C, H, W = 64, 4, 32, 32
    sigma_data = 0.8
    # 64 distinct structured latents + distinct text prompts (like CLIP 77x256)
    x0 = torch.randn(N, C, H, W, device=DEV)
    x0 = F.avg_pool2d(F.pad(x0, (1, 1, 1, 1), mode="circular"), 3, stride=1)  # smooth 1/f-like structure
    x0 = x0 / x0.std() * sigma_data
    txt = torch.randn(N, 16, 256, device=DEV)
    txt = txt / txt.norm(dim=-1, keepdim=True) * math.sqrt(256)

    class TextConditionedUnetWrapper(nn.Module):
        def __init__(self, use_text_in_emb: bool, concat_balance: float = 0.7):
            super().__init__()
            self.use_text_in_emb = use_text_in_emb
            self.in_proj = MP_Conv(4, 64, (3, 3))
            self.t_f = MP_Conv(64, 128, ())
            self.t_p = MP_Conv(128, 128, ())
            if use_text_in_emb:
                self.txt_pool = MP_Conv(256, 128, ())
            self.unet = Unet_expert(
                img_resolution=32, img_channels=64, time_emb_dim=128, text_emb_dim=256,
                channel_mult=[1, 2, 4], model_channels=48, channel_mult_emb=4, num_blocks=2,
                kernel_size=(3, 3), label_balance=0.5, concat_balance=concat_balance,
            )
            self.out_proj = MP_Conv(65, 4, (3, 3))
            self.out_gate = nn.Parameter(torch.zeros(1))
            self.freqs = torch.randn(64, device=DEV) * 1.2 * 2 * math.pi

        def forward(self, x, sigma, text_emb):
            s = sigma.view(-1, 1, 1, 1)
            c_skip = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
            c_out = s * sigma_data / (s ** 2 + sigma_data ** 2).sqrt()
            c_in = 1.0 / (s ** 2 + sigma_data ** 2).sqrt()
            c_noise = sigma.flatten().log() / 4.0
            f = (c_noise[:, None] * self.freqs[None, :]).cos() * math.sqrt(2)
            temb = self.t_p(mp_silu(self.t_f(f)))
            if self.use_text_in_emb:
                pooled = normalize(text_emb.mean(dim=1))
                temb = mp_sum(temb, self.txt_pool(pooled), t=0.5)
            h = normalize(self.in_proj(x * c_in))
            u = self.unet(h, temb, text_emb)
            u = torch.cat([u, torch.ones_like(u[:, :1])], dim=1)
            F_out = self.out_proj(u) * self.out_gate
            return c_skip * x + c_out * F_out

    for name, flag, cb in [
        ("Current Unet (CA only, concat_bal=0.7)", False, 0.7),
        ("+ Pooled Text in Emb (concat_bal=0.7)", True, 0.7),
        ("+ Pooled Text in Emb (concat_bal=0.3)", True, 0.3),
        ("CA only (concat_bal=0.3)", False, 0.3),
    ]:
        torch.manual_seed(42)
        m = TextConditionedUnetWrapper(flag, cb).to(DEV)
        opt = torch.optim.Adam([
            {"params": [p for n, p in m.named_parameters() if not n.endswith("gate")], "lr": 1.5e-3},
            {"params": [p for n, p in m.named_parameters() if n.endswith("gate")], "lr": 3.0e-3},
        ], betas=(0.9, 0.99))
        lmin, lmax = math.log(2.5), math.log(45.0)
        for step in range(400):
            idx = torch.randint(0, N, (32,), device=DEV)
            xb, tb = x0[idx], txt[idx]
            sig = torch.exp(lmin + (lmax - lmin) * torch.rand(32, device=DEV))
            nx = xb + torch.randn_like(xb) * sig.view(-1, 1, 1, 1)
            d = m(nx, sig, tb)
            s = sig.view(-1, 1, 1, 1)
            lam = (s ** 2 + sigma_data ** 2) / (s * sigma_data) ** 2
            loss = (lam * (d - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            mses = {}
            for sv in (2.5, 10.0, 40.0):
                sig = torch.full((N,), sv, device=DEV)
                nx = x0 + torch.randn_like(x0) * sv
                d = m(nx, sig, txt)
                mses[sv] = (d - x0).pow(2).mean().item()
        print(f"  {name:42s} -> s=2.5: {mses[2.5]:.4f} | s=10.0: {mses[10.0]:.4f} | s=40.0: {mses[40.0]:.4f}")


if __name__ == "__main__":
    probe_1_text_conditioning_sensitivity()
    probe_2_sigma_gradient_share()
    probe_3_high_sigma_memorization_ablation()
