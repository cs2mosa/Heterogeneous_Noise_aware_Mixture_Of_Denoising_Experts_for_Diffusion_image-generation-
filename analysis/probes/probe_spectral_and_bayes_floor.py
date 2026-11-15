"""Read-only diagnostic probe 1: Spectral partition, F=0 baseline, Wiener Bayes floor, and EDM_LOSS_NOD inversion."""
import sys
import math
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Utils.loss import EDM_LOSS_NOD, laplacian_decomposition
from Utils.utils import sample_sigma_hybrid, ZetaScheduler, MaskGenerator


def make_1f_latents(batch: int, channels: int, size: int, alpha: float = 2.0, target_std: float = 0.8, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    fx = torch.fft.fftfreq(size)
    fy = torch.fft.rfftfreq(size)
    ky, kx = torch.meshgrid(fx, fy, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k0_sq = (1.0 / size) ** 2
    amp = (k2 + k0_sq) ** (-alpha / 4.0)
    amp[0, 0] = 0.0
    noise_r = torch.randn(batch, channels, size, fy.shape[0], generator=g)
    noise_i = torch.randn(batch, channels, size, fy.shape[0], generator=g)
    spec = torch.complex(noise_r, noise_i) * amp
    x = torch.fft.irfft2(spec, s=(size, size), norm="ortho")
    x = x / (x.std(dim=(-2, -1), keepdim=True) + 1e-8) * target_std
    return x


def main():
    torch.manual_seed(0)
    B, C, H, W = 512, 4, 32, 32
    sigma_data = 0.8

    print("=" * 78)
    print("PROBE 1A: Fourier laplacian_decomposition Energy Partition (32x32x4)")
    print("=" * 78)
    white_noise = torch.randn(B, C, H, W)
    latents_a2 = make_1f_latents(B, C, H, alpha=2.0, target_std=sigma_data)

    for lap_std in [1.0, 0.5, 0.25]:
        low_n, high_n = laplacian_decomposition(white_noise, sigma=lap_std)
        e_n = white_noise.pow(2).mean().item()
        frac_low_n = low_n.pow(2).mean().item() / e_n
        frac_high_n = high_n.pow(2).mean().item() / e_n

        low_x, high_x = laplacian_decomposition(latents_a2, sigma=lap_std)
        e_x = latents_a2.pow(2).mean().item()
        frac_low_x = low_x.pow(2).mean().item() / e_x
        frac_high_x = high_x.pow(2).mean().item() / e_x

        cutoff_sq = math.log(2.0) / (4.0 * math.pi ** 2 * lap_std ** 2)
        cutoff_cycles = math.sqrt(cutoff_sq) * H
        print(f"lap_sigma={lap_std:.2f} | cutoff_r={cutoff_cycles:.2f} cycles/32 | "
              f"WhiteNoise(low={frac_low_n:.4f}, high={frac_high_n:.4f}) | "
              f"Signal_1/f^2(low={frac_low_x:.4f}, high={frac_high_x:.4f})")

    print("\n" + "=" * 78)
    print("PROBE 1B: F=0 Baseline & Wiener Bayes-Optimal Risk vs Observed MSE (0.2134 / 0.2157)")
    print("=" * 78)
    lat_fft = torch.fft.rfft2(latents_a2, norm="ortho")
    S_aa = lat_fft.abs().pow(2).mean(dim=(0, 1))
    unit_noise_fft = torch.fft.rfft2(torch.randn(B, C, H, W), norm="ortho")
    S_nn_unit = unit_noise_fft.abs().pow(2).mean(dim=(0, 1))

    sigmas_probe = [80.0, 38.07, 16.59, 5.0, 1.61, 0.8, 0.28, 0.05, 0.002]
    print(f"{'sigma':>8} | {'MSE(F=0)':>10} | {'MSE(Wiener)':>11} | {'EDM_L(F=0)':>10} | {'EDM_L(Wien)':>11} | {'Vel_RMS(F=0)':>12} | {'Vel_RMS(Wien)':>13}")
    print("-" * 88)
    for sig in sigmas_probe:
        n = sig * torch.randn_like(latents_a2)
        x_noisy = latents_a2 + n
        c_skip = sigma_data ** 2 / (sig ** 2 + sigma_data ** 2)
        lam = (sig ** 2 + sigma_data ** 2) / ((sig * sigma_data) ** 2)

        D_0 = c_skip * x_noisy
        mse_0 = (D_0 - latents_a2).pow(2).mean().item()
        edm_l_0 = lam * mse_0
        vel_0 = ((x_noisy - D_0) / sig).pow(2).mean().sqrt().item()

        H_W = S_aa / (S_aa + (sig ** 2) * S_nn_unit + 1e-12)
        D_W = torch.fft.irfft2(torch.fft.rfft2(x_noisy, norm="ortho") * H_W, s=(H, W), norm="ortho")
        mse_W = (D_W - latents_a2).pow(2).mean().item()
        edm_l_W = lam * mse_W
        vel_W = ((x_noisy - D_W) / sig).pow(2).mean().sqrt().item()
        print(f"{sig:8.3f} | {mse_0:10.5f} | {mse_W:11.5f} | {edm_l_0:10.5f} | {edm_l_W:11.5f} | {vel_0:12.5f} | {vel_W:13.5f}")

    # Test with real data std != sigma_data (common mismatch in v2 if data_std=1.0 while sigma_data=0.8 or 0.5)
    for test_std, label in [(0.8, "matched std=0.8"), (1.0, "std=1.0, sd=0.8")]:
        x_eval = make_1f_latents(2048, C, H, alpha=2.0, target_std=test_std, seed=99)
        sig_samples = sample_sigma_hybrid(2048, device=torch.device("cpu"))
        print(f"  [debug] sample_sigma_hybrid shape={tuple(sig_samples.shape)}, min={sig_samples.min():.4f}, median={sig_samples.median():.4f}, max={sig_samples.max():.4f}")
        sig_grid = sig_samples.reshape(-1, 1, 1, 1)
        x_n = x_eval + sig_grid * torch.randn_like(x_eval)
        c_skip_s = sigma_data ** 2 / (sig_grid ** 2 + sigma_data ** 2)
        lam_s = (sig_grid ** 2 + sigma_data ** 2) / ((sig_grid * sigma_data) ** 2)
        D0_s = c_skip_s * x_n
        mse_f0_hybrid = (D0_s - x_eval).pow(2).mean().item()
        edml_f0_hybrid = (lam_s * (D0_s - x_eval).pow(2)).mean().item()

        S_aa_t = torch.fft.rfft2(x_eval[:512], norm="ortho").abs().pow(2).mean(dim=(0, 1))
        H_W_s = S_aa_t[None, None, :, :] / (S_aa_t[None, None, :, :] + (sig_grid ** 2) * S_nn_unit[None, None, :, :] + 1e-12)
        DW_s = torch.fft.irfft2(torch.fft.rfft2(x_n, norm="ortho") * H_W_s, s=(H, W), norm="ortho")
        mse_W_hybrid = (DW_s - x_eval).pow(2).mean().item()
        edml_W_hybrid = (lam_s * (DW_s - x_eval).pow(2)).mean().item()
        print(f"\n  [{label}] sample_sigma_hybrid:")
        print(f"    F=0:    MSE = {mse_f0_hybrid:.5f} | Weighted EDM_L = {edml_f0_hybrid:.5f}")
        print(f"    Wiener: MSE = {mse_W_hybrid:.5f} | Weighted EDM_L = {edml_W_hybrid:.5f}")

    # Test fixed probe bins used in validation probes (e.g., sigma=[0.05..10] vs log-uniform [0.002..80])
    x_e2 = make_1f_latents(512, C, H, alpha=2.0, target_std=sigma_data, seed=99)
    log_s_uni = torch.linspace(math.log(0.002), math.log(80.0), 512)[:, None, None, None].exp()
    x_nu = x_e2 + log_s_uni * torch.randn_like(x_e2)
    cs_u = sigma_data ** 2 / (log_s_uni ** 2 + sigma_data ** 2)
    mse_f0_lu = (cs_u * x_nu - x_e2).pow(2).mean().item()
    hw_u = S_aa[None, None, :, :] / (S_aa[None, None, :, :] + (log_s_uni ** 2) * S_nn_unit[None, None, :, :] + 1e-12)
    mse_W_lu = (torch.fft.irfft2(torch.fft.rfft2(x_nu, norm="ortho") * hw_u, s=(H, W), norm="ortho") - x_e2).pow(2).mean().item()
    print(f"  [matched std=0.8] log-uniform [0.002,80]: F=0 MSE={mse_f0_lu:.5f} | Wiener MSE={mse_W_lu:.5f}")

    # Check common val_probe fixed sigma sets
    for pname, ps in [("val_sigmas=[0.05,0.2,0.5,1,2,5,10,25]", [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]),
                      ("val_sigmas=[0.02,0.1,0.5,1,3,10,30,60]", [0.02, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 60.0])]:
        m0s, mWs = [], []
        for s in ps:
            xn = x_e2 + s * torch.randn_like(x_e2)
            cs = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
            m0s.append((cs * xn - x_e2).pow(2).mean().item())
            hw = S_aa / (S_aa + (s ** 2) * S_nn_unit + 1e-12)
            mWs.append((torch.fft.irfft2(torch.fft.rfft2(xn, norm="ortho") * hw, s=(H, W), norm="ortho") - x_e2).pow(2).mean().item())
        print(f"  {pname}: F=0 mean MSE={np.mean(m0s):.5f} | Wiener mean MSE={np.mean(mWs):.5f}")

    print("\n" + "=" * 78)
    print("PROBE 1C: EDM_LOSS_NOD Mechanism for vit_loss Rising (0.35 -> 0.65) vs pure_loss (1.53 -> 0.58)")
    print("=" * 78)
    expert_centers = torch.tensor([0.25, 0.75])
    for lap_std in [1.0, 0.5]:
        loss_fn = EDM_LOSS_NOD(
            expert_centers_vit=expert_centers,
            expert_centers_Unet=expert_centers,
            num_experts=2,
            sigma_data=sigma_data,
            lap_sigma=lap_std,
            alpha_blend=0.8,
            alpha_unet=0.1,
            alpha_vit=0.1,
            transition_sigma=3.0,
        )
        x0 = make_1f_latents(512, C, H, alpha=2.0, target_std=sigma_data, seed=123)
        sig = sample_sigma_hybrid(512, device=torch.device("cpu"))
        sig_4d = sig.reshape(-1, 1, 1, 1)
        x_noisy = x0 + sig_4d * torch.randn_like(x0)
        c_skip = sigma_data ** 2 / (sig_4d ** 2 + sigma_data ** 2)
        c_out = sig_4d * sigma_data / torch.sqrt(sig_4d ** 2 + sigma_data ** 2)
        lam = (sig_4d ** 2 + sigma_data ** 2) / ((sig_4d * sigma_data) ** 2)

        # Trajectory router blend weights calibrated to match Plot 7c/7d (mean w_vit~0.72, w_unet~0.28)
        w_unet = torch.sigmoid(1.5 * (torch.log(sig_4d) - math.log(3.0)))
        w_vit = 1.0 - w_unet
        r_probs = torch.full((512, 2), 0.5)
        r_raw = torch.zeros(512, 2)

        # Compare Case A (Step 0: F=0, log_var=0) vs Case B (Trained: Wiener/partial denoiser + converged optimal log_var)
        # Note optimal Gaussian NLL log_var^*(sig) = log(E[weight * ||D_blend - x||^2]) per sample/sigma!
        D0 = c_skip * x_noisy
        out_step0 = {
            "denoised": D0, "unet_denoised": D0, "vit_denoised": D0,
            "w_vit": w_vit, "w_unet": w_unet, "log_var": torch.zeros(512, 1, 1, 1),
            "Unet_router_loss": r_probs, "vit_router_loss": r_probs,
            "Unet_raw": r_raw, "vit_raw": r_raw,
        }
        res0 = loss_fn(x0, x_noisy, sig, out_step0)

        # Case B: Suppose model learned D_W (or 50% mix of D_W and D_0) and log_var adapts to local weighted error!
        # Wait: at Wiener solution, compute per-sigma expected weighted error E(sig) -> log_var*(sig) = log(E(sig))
        H_W_4d = S_aa[None, None, :, :] / (S_aa[None, None, :, :] + (sig_4d ** 2) * S_nn_unit[None, None, :, :] + 1e-12)
        D_W = torch.fft.irfft2(torch.fft.rfft2(x_noisy, norm="ortho") * H_W_4d, s=(H, W), norm="ortho")
        err_blend_W = (lam * (D_W - x0).pow(2)).mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
        log_var_W = torch.log(err_blend_W).squeeze(-1).squeeze(-1)

        # What if blend is partially trained (e.g. D_blend = 0.5 D_W + 0.5 D_0), while log_var adapts to D_blend AND D_vit stays closer to D_0 (low-sigma skip collapse)?
        D_partial = 0.5 * D_W + 0.5 * D0
        err_part = (lam * (D_partial - x0).pow(2)).mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
        log_var_part = torch.log(err_part)

        out_Wiener = {
            "denoised": D_W, "unet_denoised": D_W, "vit_denoised": D_W,
            "w_vit": w_vit, "w_unet": w_unet, "log_var": torch.log(err_blend_W),
            "Unet_router_loss": r_probs, "vit_router_loss": r_probs,
            "Unet_raw": r_raw, "vit_raw": r_raw,
        }
        resW = loss_fn(x0, x_noisy, sig, out_Wiener)

        # Case C: D_blend = D_partial, log_var = log_var_part, D_unet = D_partial, D_vit = D_0 (frozen ViT at low sigma!)
        out_vit_frozen = {
            "denoised": D_partial, "unet_denoised": D_partial, "vit_denoised": D0,
            "w_vit": w_vit, "w_unet": w_unet, "log_var": log_var_part,
            "Unet_router_loss": r_probs, "vit_router_loss": r_probs,
            "Unet_raw": r_raw, "vit_raw": r_raw,
        }
        res_vf = loss_fn(x0, x_noisy, sig, out_vit_frozen)

        # Case D: Same as Case C, PLUS D_vit outputs un-attenuated high-freq noise / F_vit perturbation (std=0.25)
        D_vit_noisy = D0 + c_out * (0.25 * torch.randn_like(x0))
        out_vit_noisy = {
            "denoised": D_partial, "unet_denoised": D_partial, "vit_denoised": D_vit_noisy,
            "w_vit": w_vit, "w_unet": w_unet, "log_var": log_var_part,
            "Unet_router_loss": r_probs, "vit_router_loss": r_probs,
            "Unet_raw": r_raw, "vit_raw": r_raw,
        }
        res_vn = loss_fn(x0, x_noisy, sig, out_vit_noisy)

        print(f"\n[lap_sigma={lap_std:.2f}] mean w_unet={w_unet.mean():.3f}, mean w_vit={w_vit.mean():.3f}")
        for tag, r in [
            ("Step-0 (F=0, log_var=0)", res0),
            ("Wiener optimal + optimal log_var", resW),
            ("Partial blend + opt log_var + D_vit=D0", res_vf),
            ("Partial blend + opt log_var + noisy F_vit(0.25)", res_vn),
        ]:
            print(f"  {tag:46s} | tot={r['loss']:.4f}, pure={r['pure_loss']:.4f}, "
                  f"vit(hi)={r['loss_texture']:.4f}, unet(lo)={r['loss_structure']:.4f}, mse={r['denoising']:.4f}")


if __name__ == "__main__":
    main()
