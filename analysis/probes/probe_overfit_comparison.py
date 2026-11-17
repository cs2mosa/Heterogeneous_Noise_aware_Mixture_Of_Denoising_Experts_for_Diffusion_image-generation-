"""Controlled HDMOEM_v3 small-subset overfitting test (mirroring colab Cell 5b).

Uses the real repository stack:
  preconditioned_HDMOEM_v3, EDM_LOSS_NOD, build_optimizer_param_groups,
  WSDScheduler, EMA, ZetaScheduler, MaskGenerator, sample_sigma_hybrid, EDM_Sampler.

Runs on 64 structured sharp-edged 32x32x4 latents (scaled to sigma_data=0.8)
with paired conditioning embeddings so we can directly compare Before vs. After
architectural changes on:
  - Per-sigma memorization MSE (0.02, 0.05, 0.1, 0.5, 2.5, 10.0, 40.0)
  - Low-sigma score velocity ||(x - D)/sigma||_RMS (target ~ 1.0)
  - Inner-band vs outer-band residual noise ratios (r_inner, r_outer)
  - EDM_Sampler latent RAPSD ratio vs ground-truth latents (CFG=1.0 & CFG=3.0)
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_config_v3 import preconditioned_HDMOEM_v3
from Utils.configs import (
    loss_configs as base_loss_configs,
    mask_configs as base_mask_configs,
    model_configs as base_model_configs,
    optim_configs as base_optim_configs,
    zeta_configs as base_zeta_configs,
)
from Utils.EDM_sampler import EDM_Sampler
from Utils.loss import EDM_LOSS_NOD
from Utils.training import build_optimizer_param_groups
from Utils.utils import EMA, MaskGenerator, WSDScheduler, ZetaScheduler, sample_sigma_hybrid


def make_structured_latents(n: int = 64, res: int = 32, sigma_data: float = 0.8, seed: int = 42, device: str = "cuda"):
    """Generate 64 distinct sharp-edged occlusion + texture 32x32x4 latents with paired embeddings."""
    g = torch.Generator(device=device).manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(res, device=device), torch.arange(res, device=device), indexing="ij")
    imgs = torch.zeros(n, 4, res, res, device=device)

    # Layer sharp-edged geometric shapes & textures per sample so high-freq structure is deterministic
    for _ in range(20):
        r = 3.0 + (res / 2.5) * torch.rand(n, generator=g, device=device)
        cy = torch.rand(n, generator=g, device=device) * res
        cx = torch.rand(n, generator=g, device=device) * res
        col = torch.randn(n, 4, generator=g, device=device)
        mask = ((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2) < r[:, None, None] ** 2
        imgs = torch.where(mask[:, None], col[:, :, None, None], imgs)

    # Add fine 1-2px checkerboard/stripe edges inside random crops to test outer-band high-freq detail
    for i in range(n):
        y0 = int(torch.randint(0, res - 8, (1,), generator=g, device=device).item())
        x0 = int(torch.randint(0, res - 8, (1,), generator=g, device=device).item())
        freq = 1.5 if (i % 2 == 0) else 2.5
        pat = torch.sin((xx[y0:y0 + 8, x0:x0 + 8] * freq + yy[y0:y0 + 8, x0:x0 + 8] * 0.7)).unsqueeze(0)
        col = torch.randn(4, 1, 1, generator=g, device=device) * 0.8
        imgs[i, :, y0:y0 + 8, x0:x0 + 8] += col * pat

    imgs = imgs - imgs.mean(dim=(0, 2, 3), keepdim=True)
    imgs = imgs / imgs.std() * sigma_data

    # Paired conditioning tokens (N, 16, 128) + null prompt embedding (1, 16, 128)
    txt = torch.randn(n, 16, 128, generator=g, device=device)
    txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-8) * math.sqrt(128)
    null_txt = torch.zeros(1, 16, 128, device=device)
    null_txt[:, 0, :] = 1.0  # BOS-like constant token, matching empty prompt structure
    return imgs, txt, null_txt


def radial_spectrum(tensor_4d: torch.Tensor) -> np.ndarray:
    x = tensor_4d.float() - tensor_4d.float().mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(x, norm="ortho")
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    power = (fft_shift.real ** 2 + fft_shift.imag ** 2).mean(dim=(0, 1)).cpu().numpy()
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    return (tbin / np.maximum(nr, 1))[: h // 2]


def band_masks(res: int, device: str):
    f = torch.fft.fftfreq(res, device=device).abs()
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    inner = (fx <= 0.25) & (fy <= 0.25)
    return inner, ~inner


def band_energy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    X = torch.fft.fft2(x.float(), norm="ortho")
    return (X.abs() ** 2 * mask).sum(dim=(-2, -1)).sum(dim=-1)


def run_overfit(tag: str, patch_sizes: list[int], total_steps: int, micro_bs: int, grad_accum: int, n_overfit: int, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    res = 32
    sigma_data = 0.8
    overfit_x0, overfit_text_emb, cached_null_emb = make_structured_latents(
        n=n_overfit, res=res, sigma_data=sigma_data, seed=seed, device=device,
    )

    eff_bs = micro_bs * grad_accum
    warmup_steps = max(20, min(200, int(0.05 * total_steps)))

    m_cfg = copy.deepcopy(base_model_configs)
    m_cfg.update({
        "device": device,
        "dataset_size": n_overfit,
        "img_channels": 4,
        "internal_channels": 48,
        "data_img_res": 256,
        "img_resolution": res,
        "time_emb_dim": 128,
        "text_emb_dim": 128,
        "Unet_model_channels": 32,
        "Unet_channel_mult": [1, 2, 4],
        "Unet_num_blocks": 1,
        "Unet_kernel_sizes": [(3, 3), (5, 5)],
        "VIT_num_blocks": 2,
        "VIT_patch_sizes": patch_sizes,
        "VIT_num_heads": 4,
        "VIT_emb_size": 128,
        "VIT_num_groups": 4,
        "cfg_drop_prob": 0.10,
        "pool_size": 8,
        "sigma_data": sigma_data,
        "batch_size": eff_bs,
        "total_steps": total_steps,
        "max_blend_shift": 0.40,
        "blend_anchor_weight": 0.02,
        "blend_mode": "spherical",
        "ema_beta": 0.995,
        "ema_warmup_power": 2.0 / 3.0,
        "ema_min_warmup_step": min(150, max(50, int(0.10 * total_steps))),
    })

    o_cfg = copy.deepcopy(base_optim_configs)
    o_cfg.update({
        "lr_schedule": "wsd",
        "lr_unet": 1.5e-3,
        "lr_vit": 1.5e-3,
        "lr_router": 1.0e-3,
        "lr_out_proj_vit": 1.5e-3,
        "lr_out_proj_unet": 1.5e-3,
        "lr_pooled_attn": 1.0e-3,
        "lr_blend_residual": 3.0e-4,
        "lr_gates": 3.0e-3,
        "LR_DEFAULT": 1.0e-3,
        "warmup_steps": warmup_steps,
        "stable_until_frac": 0.85,
        "min_lr_mult": 0.10,
    })

    l_cfg = copy.deepcopy(base_loss_configs)
    l_cfg.update({"laplacian_std": 0.5, "alpha_blend": 0.8, "alpha_unet": 0.1, "alpha_vit": 0.1})
    mk_cfg = copy.deepcopy(base_mask_configs)
    mk_cfg["total_steps"] = total_steps
    z_cfg = copy.deepcopy(base_zeta_configs)
    z_cfg["max_zeta"] = 1.0
    z_cfg["warmup_ratio"] = 0.10
    z_cfg["total_schedule_steps"] = max(20, int(0.30 * total_steps))

    model = preconditioned_HDMOEM_v3(
        IN_in_channels=m_cfg["img_channels"],
        IN_img_resolution=m_cfg["img_resolution"],
        internal_channels=m_cfg["internal_channels"],
        time_emb_dim=m_cfg["time_emb_dim"],
        text_emb_dim=m_cfg["text_emb_dim"],
        num_experts=m_cfg["num_experts"],
        top_k=m_cfg["top_k"],
        Fourier_bandwidth=m_cfg["fourier_bandwidth"],
        VIT_num_blocks=m_cfg["VIT_num_blocks"],
        VIT_patch_sizes=m_cfg["VIT_patch_sizes"],
        VIT_num_groups=m_cfg["VIT_num_groups"],
        VIT_num_heads=m_cfg["VIT_num_heads"],
        VIT_emb_size=m_cfg["VIT_emb_size"],
        VIT_emb_balance=m_cfg["VIT_emb_balances"],
        VIT_pos_emb_balance=m_cfg["VIT_pos_emb_balances"],
        Unet_num_blocks=m_cfg["Unet_num_blocks"],
        Unet_channel_mult=m_cfg["Unet_channel_mult"],
        Unet_channel_mult_emb=m_cfg["Unet_channel_mult_emb"],
        Unet_kernel_sizes=m_cfg["Unet_kernel_sizes"],
        Unet_model_channels=m_cfg["Unet_model_channels"],
        Unet_label_balance=m_cfg["Unet_label_balance"],
        Unet_concat_balance=m_cfg["Unet_concat_balances"],
        sigma_data=m_cfg["sigma_data"],
        log_var_channels=m_cfg["log_var_channels"],
        transition_sigma=m_cfg["transition_sigma"],
        transition_sharpness=m_cfg["transition_sharpness"],
        pool_size=m_cfg["pool_size"],
        attn_pool_balance=m_cfg["attn_pool_balance"],
        p_mean=mk_cfg["p_mean"],
        p_std=mk_cfg["p_std"],
        sigma_min=m_cfg["sigma_min"],
        sigma_max=m_cfg["sigma_max"],
        alpha_ext=m_cfg["extreme_prob"],
        max_blend_shift=m_cfg["max_blend_shift"],
        blend_mode=m_cfg["blend_mode"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_vit_params = sum(p.numel() for p in model.net.VIT_experts.parameters())
    print(f"\n=== [{tag}] patch_sizes={patch_sizes} | total_params={n_params/1e6:.2f}M | vit_params={n_vit_params/1e6:.2f}M ===")

    groups, _ = build_optimizer_param_groups(model, o_cfg)
    opt = torch.optim.Adam(groups, betas=o_cfg["betas"], weight_decay=o_cfg["weight_decay"], eps=o_cfg["eps"])
    sched = WSDScheduler(
        optimizer=opt, total_steps=total_steps, batch_size=eff_bs,
        warmup_steps=warmup_steps, stable_until_frac=0.85, min_lr_mult=0.10, dataset_size=n_overfit,
    )
    ema = EMA(model, beta=m_cfg["ema_beta"], warmup_power=m_cfg["ema_warmup_power"], min_warmup_step=m_cfg["ema_min_warmup_step"])
    zeta_sched = ZetaScheduler(
        total_steps=z_cfg["total_schedule_steps"], max_zeta=z_cfg["max_zeta"],
        min_zeta=z_cfg["min_zeta"], strategy=z_cfg["strategy"], warmup_ratio=z_cfg["warmup_ratio"],
    )
    u_gen = MaskGenerator(
        expert_attributes=mk_cfg["unet_attr"], p_mean=mk_cfg["p_mean"], p_std=mk_cfg["p_std"],
        total_steps=total_steps, min_active=mk_cfg["min_active"], step_size=mk_cfg["step_size"],
        min_bandwidth=mk_cfg["min_BW_u"], bandwidth=mk_cfg["max_BW_u"], strat_band=mk_cfg["strat_band"],
        noise_range=mk_cfg["unet_noise_range"], sigma_max=m_cfg["sigma_max"], sigma_min=m_cfg["sigma_min"], alpha_ext=m_cfg["extreme_prob"],
    )
    v_gen = MaskGenerator(
        expert_attributes=mk_cfg["vit_attr"], p_mean=mk_cfg["p_mean"], p_std=mk_cfg["p_std"],
        total_steps=total_steps, min_active=mk_cfg["min_active"], step_size=mk_cfg["step_size"],
        min_bandwidth=mk_cfg["min_BW_v"], bandwidth=mk_cfg["max_BW_v"], strat_band=mk_cfg["strat_band"],
        noise_range=mk_cfg["vit_noise_range"], sigma_max=m_cfg["sigma_max"], sigma_min=m_cfg["sigma_min"], alpha_ext=m_cfg["extreme_prob"],
    )
    crit = EDM_LOSS_NOD(
        expert_centers_vit=v_gen.expert_centers, expert_centers_Unet=u_gen.expert_centers,
        num_experts=m_cfg["num_experts"], sigma_data=m_cfg["sigma_data"], Unet_bal=l_cfg["unet_bal"], vit_bal=l_cfg["vit_bal"],
        z_bal=l_cfg["z_bal"], prior_bal=l_cfg["prior_bal"], p_mean=mk_cfg["p_mean"], p_std=mk_cfg["p_std"],
        sharpness=l_cfg["sharpness"], transition_sigma=m_cfg["transition_sigma"], overlap_width=m_cfg["transition_sharpness"],
        alpha_vit=l_cfg["alpha_vit"], alpha_unet=l_cfg["alpha_unet"], alpha_blend=l_cfg["alpha_blend"],
        lap_kernel=l_cfg["laplacian_kernel"], lap_sigma=l_cfg["laplacian_std"], sigma_min=m_cfg["sigma_min"],
        sigma_max=m_cfg["sigma_max"], alpha_ext=m_cfg["extreme_prob"],
        vit_noise_range=mk_cfg["vit_noise_range"], unet_noise_range=mk_cfg["unet_noise_range"],
    )

    from contextlib import nullcontext
    amp_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16))
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else nullcontext
    )

    eval_sigmas = (0.02, 0.05, 0.1, 0.5, 2.5, 10.0, 40.0)
    n_probe = 32
    probe_gen = torch.Generator(device="cpu").manual_seed(1234)
    probe_eps = torch.randn(overfit_x0[:n_probe].shape, generator=probe_gen, dtype=torch.float32).to(device)
    inner_mask, outer_mask = band_masks(res, device)

    @torch.no_grad()
    def eval_bins(eval_m):
        was_train = eval_m.training
        eval_m.eval()
        out_stats = {}
        try:
            for sv in eval_sigmas:
                sig_t = torch.full((n_probe, 1, 1, 1), float(sv), device=device)
                n_tensor = probe_eps * sig_t
                noisy = overfit_x0[:n_probe] + n_tensor
                um = u_gen(sigma=sig_t, step=total_steps)
                vm = v_gen(sigma=sig_t, step=total_steps)
                with amp_ctx():
                    om = eval_m(x=noisy, sigma=sig_t, text_emb=overfit_text_emb[:n_probe], Unet_router_mask=um, Vit_router_mask=vm, zeta=0.0)
                dx = om["denoised"].float()
                err = dx - overfit_x0[:n_probe]
                mse = float((err ** 2).mean().item())
                vel = float(((noisy - dx) / sv).pow(2).mean().sqrt().item())
                r_in = float((band_energy(err, inner_mask).sum() / band_energy(n_tensor, inner_mask).sum()).item())
                r_out = float((band_energy(err, outer_mask).sum() / band_energy(n_tensor, outer_mask).sum()).item())
                out_stats[sv] = {"mse": mse, "vel": vel, "r_in": r_in, "r_out": r_out}
        finally:
            if was_train:
                eval_m.train()
        return out_stats

    perm, perm_pos = None, 0

    def next_batch(bs: int):
        nonlocal perm, perm_pos
        if perm is None or perm_pos + bs > n_overfit:
            perm = torch.randperm(n_overfit, device=device)
            perm_pos = 0
        idx = perm[perm_pos:perm_pos + bs]
        perm_pos += bs
        return overfit_x0[idx], overfit_text_emb[idx]

    init_stats = eval_bins(model)
    print("  [Init] " + " | ".join(f"s={s}: mse={init_stats[s]['mse']:.5f}" for s in eval_sigmas))

    t0 = time.perf_counter()
    model.train()
    eval_every = max(1, total_steps // 4)
    log_every = max(1, min(25, eval_every))
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        cur_zeta = zeta_sched.get_zeta(step=step)
        acc_loss, acc_vit, acc_unet = 0.0, 0.0, 0.0
        for _ in range(grad_accum):
            x0_b, txt_b = next_batch(micro_bs)
            drop_mask = torch.rand(x0_b.shape[0], 1, 1, device=device) < 0.10
            txt_b = torch.where(drop_mask, cached_null_emb.expand_as(txt_b), txt_b)
            sigma = sample_sigma_hybrid(
                batch_size=x0_b.shape[0], sigma_max=m_cfg["sigma_max"], sigma_min=m_cfg["sigma_min"],
                p_mean=mk_cfg["p_mean"], p_std=mk_cfg["p_std"], extreme_prob=m_cfg["extreme_prob"], device=device,
            )
            noisy_x = x0_b + torch.randn_like(x0_b) * sigma
            um = u_gen(sigma=sigma, step=step)
            vm = v_gen(sigma=sigma, step=step)
            with amp_ctx():
                om = model(x=noisy_x, sigma=sigma, text_emb=txt_b, Unet_router_mask=um, Vit_router_mask=vm, zeta=cur_zeta, return_log_var=True)
                ld = crit(x=x0_b, noisy_x=noisy_x, sigma=sigma, out_model=om)
                loss = ld["loss"]
                bres = getattr(model.net, "_last_bounded_residual", None)
                if bres is not None:
                    loss = loss + 0.02 * bres.float().pow(2).mean()
                model.net._last_bounded_residual = None
                (loss / grad_accum).backward()
            acc_loss += float(ld["loss"].detach().item()) / grad_accum
            acc_vit += float(ld["loss_texture"].detach().item()) / grad_accum
            acc_unet += float(ld["loss_structure"].detach().item()) / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        ema.update()

        if (step + 1) % log_every == 0 and (step + 1) % eval_every != 0 and step != total_steps - 1:
            print(f"  step {step+1:4d}/{total_steps} ({time.perf_counter()-t0:5.1f}s) | loss={acc_loss:.4f} ViT={acc_vit:.4f} UNet={acc_unet:.4f}")
        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            st = eval_bins(ema.ema_model)
            dt = time.perf_counter() - t0
            print(
                f"  step {step+1:4d}/{total_steps} ({dt:5.1f}s) | loss={acc_loss:.4f} ViT={acc_vit:.4f} UNet={acc_unet:.4f} "
                f"| s=0.02 r_out={st[0.02]['r_out']:.3f} vel={st[0.02]['vel']:.3f} "
                f"| s=0.1 mse={st[0.1]['mse']:.5f} | s=0.5 mse={st[0.5]['mse']:.5f} | s=10 mse={st[10.0]['mse']:.4f}"
            )

    ema_final = eval_bins(ema.ema_model)
    raw_final = eval_bins(model)
    print("\n  [Final Per-Sigma Metrics (EMA / Raw)]")
    print("   sigma |   EMA_MSE  |   Raw_MSE  | EMA_r_in | EMA_r_out | EMA_vel_RMS | Raw_vel_RMS")
    for s in eval_sigmas:
        ef, rf = ema_final[s], raw_final[s]
        print(f"  {s:6.2f} | {ef['mse']:10.6f} | {rf['mse']:10.6f} | {ef['r_in']:8.4f} | {ef['r_out']:9.4f} | {ef['vel']:11.4f} | {rf['vel']:11.4f}")

    # Sampler check (CFG=1.0 and CFG=3.0) using EMA
    eval_m = ema.ema_model.eval()
    n_vis = 8
    gen_noise = torch.randn(n_vis, 4, res, res, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    null_b = cached_null_emb.expand(n_vis, -1, -1)
    with torch.no_grad():
        smp1 = EDM_Sampler(eval_m, num_solve_steps=32, guidance=1.0, unet_mask_gen=u_gen, vit_mask_gen=v_gen, mask_step=total_steps)
        lats1 = smp1.sample(gen_noise.clone(), text_emb=overfit_text_emb[:n_vis], uncond_text_emb=null_b)
        smp3 = EDM_Sampler(eval_m, num_solve_steps=32, guidance=3.0, unet_mask_gen=u_gen, vit_mask_gen=v_gen, mask_step=total_steps)
        lats3 = smp3.sample(gen_noise.clone(), text_emb=overfit_text_emb[:n_vis], uncond_text_emb=null_b)
    spec_gt = radial_spectrum(overfit_x0[:n_vis])
    spec_1 = radial_spectrum(lats1)
    spec_3 = radial_spectrum(lats3)
    mid1 = float(np.mean(spec_1[2:9] / np.maximum(spec_gt[2:9], 1e-12)))
    hi1 = float(np.mean(spec_1[9:16] / np.maximum(spec_gt[9:16], 1e-12)))
    mid3 = float(np.mean(spec_3[2:9] / np.maximum(spec_gt[2:9], 1e-12)))
    hi3 = float(np.mean(spec_3[9:16] / np.maximum(spec_gt[9:16], 1e-12)))
    # Nearest-neighbor latent distance to GT overfit set (measures memorization fidelity)
    d1 = torch.cdist(lats1.flatten(1), overfit_x0.flatten(1)).min(dim=1).values.mean().item()
    d3 = torch.cdist(lats3.flatten(1), overfit_x0.flatten(1)).min(dim=1).values.mean().item()
    print(f"  [Sampler] CFG=1.0: mid_RAPSD={mid1:.2f}x, high_RAPSD={hi1:.2f}x, NN_L2_to_GT={d1:.2f}")
    print(f"  [Sampler] CFG=3.0: mid_RAPSD={mid3:.2f}x, high_RAPSD={hi3:.2f}x, NN_L2_to_GT={d3:.2f}")
    print(f"  [Gates]   vit_proj_gate={model.net.vit_proj_gate.item():.4f}, unet_proj_gate={model.net.unet_proj_gate.item():.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--patch-sizes", type=int, nargs=2, default=[2, 2])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--micro-bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--n-overfit", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_overfit(args.tag, args.patch_sizes, args.steps, args.micro_bs, args.grad_accum, args.n_overfit, args.seed)
