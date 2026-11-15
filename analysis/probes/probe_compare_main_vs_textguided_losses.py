"""Read-only probe comparing main vs text_guided loss designs (NOD without Laplacian vs Laplacian NOD vs EDM_LOSS)."""
import sys, math
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import Utils.loss as main_loss

print("=== 1. Verifying main:Utils/loss.py exact bug check ===")
c = torch.tensor([0.25, 0.75])
nod_main = main_loss.EDM_LOSS_NOD(expert_centers_vit=c, expert_centers_Unet=c, num_experts=2, sigma_data=0.8)
B = 4
fake_x = torch.randn(B, 4, 32, 32)
fake_sig = torch.ones(B)
fake_out = {
    "denoised": fake_x.clone(),
    "unet_denoised": fake_x.clone(),
    "vit_denoised": fake_x.clone(),
    "w_vit": torch.full((B, 1, 1, 1), 0.5, requires_grad=True),
    "w_unet": torch.full((B, 1, 1, 1), 0.5, requires_grad=True),
    "log_var": torch.zeros(B, 1, 1, 1),
    "Unet_router_loss": torch.full((B, 2), 0.5),
    "vit_router_loss": torch.full((B, 2), 0.5),
    "Unet_raw": torch.zeros(B, 2),
    "vit_raw": torch.zeros(B, 2),
}
try:
    nod_main(fake_x, fake_x, fake_sig, fake_out)
    print("main EDM_LOSS_NOD forward: OK")
except Exception as e:
    print(f"main EDM_LOSS_NOD forward ERROR: {type(e).__name__}: {e}")

# Verify EDM_LOSS_LPD target assignment bug on main
lpd_main = main_loss.EDM_LOSS_LPD(expert_centers_vit=c, expert_centers_Unet=c, num_experts=2, sigma_data=0.8)
out_lpd = lpd_main(fake_x, fake_x + 0.1 * torch.randn_like(fake_x), fake_sig, {
    **fake_out,
    "vit_denoised": torch.zeros_like(fake_x),
    "unet_denoised": torch.zeros_like(fake_x),
})
print(f"main EDM_LOSS_LPD output keys mapping: loss_structure(actually vit->low)={out_lpd['loss_structure']:.4f}, "
      f"loss_texture(actually unet->high)={out_lpd['loss_texture']:.4f}")

print("\n=== 2. Comparing Loss Formulations on Real $1/f^2$ Latents + Wiener Spectral Breakdown ===")
# Closed-form comparison at step 0 vs converged log_var (u* < 0) across sigmas
def rfft_laplacian_decomp(image: torch.Tensor, sigma: float = 0.5):
    h, w = image.shape[-2:]
    cutoff_sq = math.log(2.0) / (4.0 * (math.pi ** 2) * max(float(sigma), 1e-6) ** 2)
    fy = torch.fft.fftfreq(h, dtype=torch.float32).view(-1, 1)
    fx = torch.fft.rfftfreq(w, dtype=torch.float32).view(1, -1)
    low_mask = ((fy * fy + fx * fx) <= cutoff_sq).to(torch.float32)
    low_freq = torch.fft.irfft2(torch.fft.rfft2(image.float(), norm="ortho") * low_mask, s=(h, w), norm="ortho")
    return low_freq, image - low_freq

torch.manual_seed(0)
N, C, H, W = 512, 4, 32, 32
sigma_data = 0.8
fx = torch.fft.fftfreq(H); fy = torch.fft.rfftfreq(W)
ky, kx = torch.meshgrid(fx, fy, indexing="ij")
amp = (kx ** 2 + ky ** 2 + (1.0 / H) ** 2) ** (-2.0 / 4.0); amp[0, 0] = 0.0
spec = torch.complex(torch.randn(N, C, H, fy.shape[0]), torch.randn(N, C, H, fy.shape[0])) * amp
x0 = torch.fft.irfft2(spec, s=(H, W), norm="ortho")
x0 = x0 / (x0.std(dim=(-2, -1), keepdim=True) + 1e-8) * sigma_data

# Sample hybrid sigmas explicitly (matching Utils.utils log-normal + uniform tail mixture)
# p_mean=-0.4, p_std=1.2, alpha_ext=0.2, [0.002, 80]
u_mix = torch.rand(N)
sig = torch.where(
    u_mix < 0.2,
    torch.exp(math.log(0.002) + torch.rand(N) * (math.log(80.0) - math.log(0.002))),
    torch.exp(-0.4 + 1.2 * torch.randn(N)).clamp(0.002, 80.0),
)
sig4 = sig.reshape(-1, 1, 1, 1)
xn = x0 + sig4 * torch.randn_like(x0)
c_skip = sigma_data ** 2 / (sig4 ** 2 + sigma_data ** 2)
lam = (sig4 ** 2 + sigma_data ** 2) / ((sig4 * sigma_data) ** 2)
D_0 = c_skip * xn

# Wiener optimal full-spectrum estimator
S_aa = torch.fft.rfft2(x0, norm="ortho").abs().pow(2).mean(dim=(0, 1))
S_nn = torch.fft.rfft2(torch.randn_like(x0), norm="ortho").abs().pow(2).mean(dim=(0, 1))
H_W = S_aa[None, None, :, :] / (S_aa[None, None, :, :] + (sig4 ** 2) * S_nn[None, None, :, :] + 1e-12)
D_W = torch.fft.irfft2(torch.fft.rfft2(xn, norm="ortho") * H_W, s=(H, W), norm="ortho")

# Simulate trajectory weights w_unet, w_vit with transition=3.0, sharpness=1.5
w_unet = torch.sigmoid(1.5 * (torch.log(sig4) - math.log(3.0)))
w_vit = 1.0 - w_unet
err_W = (lam * (D_W - x0).pow(2)).mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
u_opt = torch.log(err_W)  # converged negative u* where D_blend = D_W

print(f"F=0 vs Wiener Optimal Losses (transition=3.0, mean w_vit={w_vit.mean():.3f}, mean w_unet={w_unet.mean():.3f}):")
for state_name, D_br, u_cur in [
    ("Step-0 (F=0, u=0)", D_0, torch.zeros_like(u_opt)),
    ("Converged (D=Wiener, u=u*)", D_W, u_opt),
    ("Partial (blend=Wiener, vit=F0, u=u*)", D_0, u_opt),
]:
    D_bl = D_0 if "Step-0" in state_name else D_W
    pure_l = (lam * (D_bl - x0).pow(2) / u_cur.exp() + u_cur).mean().item()

    # 1) main NOD (no Laplacian, no exp(-u.detach())):
    vit_nod_nolu = (lam * w_vit * (D_br - x0).pow(2)).mean().item()
    unet_nod_nolu = (lam * w_unet * (D_bl - x0).pow(2)).mean().item()

    # 2) text_guided NOD (with rFFT Laplacian lap_sigma=0.5 & 1.0, WITH exp(-u.detach())):
    _, vh05 = rfft_laplacian_decomp(D_br - x0, 0.5)
    ul05, _ = rfft_laplacian_decomp(D_bl - x0, 0.5)
    vit_lap05 = (lam * (1.0 / u_cur.exp()) * w_vit * vh05.pow(2)).mean().item()
    unet_lap05 = (lam * (1.0 / u_cur.exp()) * w_unet * ul05.pow(2)).mean().item()

    _, vh10 = rfft_laplacian_decomp(D_br - x0, 1.0)
    ul10, _ = rfft_laplacian_decomp(D_bl - x0, 1.0)
    vit_lap10 = (lam * (1.0 / u_cur.exp()) * w_vit * vh10.pow(2)).mean().item()
    unet_lap10 = (lam * (1.0 / u_cur.exp()) * w_unet * ul10.pow(2)).mean().item()

    print(f"  [{state_name}] pure={pure_l:.4f}")
    print(f"    NOD (no Laplacian, main style):     vit={vit_nod_nolu:.4f}, unet={unet_nod_nolu:.4f}")
    print(f"    Laplacian NOD (sigma=0.5, exp(-u)): vit={vit_lap05:.4f}, unet={unet_lap05:.4f}")
    print(f"    Laplacian NOD (sigma=1.0, exp(-u)): vit={vit_lap10:.4f}, unet={unet_lap10:.4f}")
