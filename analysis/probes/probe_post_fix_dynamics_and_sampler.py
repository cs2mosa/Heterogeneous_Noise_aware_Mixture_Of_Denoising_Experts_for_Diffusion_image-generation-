"""Read-only follow-up probe: 250-step dynamics, per-sigma gradient profile, cross-attn rank, and sampler check."""
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_config_v3 import preconditioned_HDMOEM_v3
from models.model_components import Vit_expert
from models.model_internals import MP_Conv, normalize
from Utils.configs import model_configs, loss_configs, optim_configs, mask_configs, zeta_configs
from Utils.loss import EDM_LOSS_NOD
from Utils.utils import sample_sigma_hybrid, ZetaScheduler, MaskGenerator, Lr_Scheduler
from Utils.EDM_sampler import EDM_Sampler
from analysis.probes.probe_post_fix_audit import (
    build_realistic_scaled_model,
    build_criterion_from_configs,
    make_1f_latents,
)


def probe_1_250_step_warmup_and_branch_learning():
    print("=" * 84)
    print("PROBE 1: 250-Step Warmup Escape & MP_Conv vs Gate Dynamics (optim_configs)")
    print("=" * 84)
    torch.manual_seed(10)
    model = build_realistic_scaled_model(vit_blocks=4, unet_blocks=2)
    model.train()

    gate_params = [
        p for n, p in model.named_parameters()
        if n.split(".")[-1].endswith("_gate") or n.split(".")[-1].startswith("out_gain_")
    ]
    gate_param_set = set(gate_params)
    unet_expert_params = [p for p in model.net.Unet_experts.parameters() if p not in gate_param_set]
    vit_expert_params = [p for p in model.net.VIT_experts.parameters() if p not in gate_param_set]
    special_params = set(gate_params) | set(unet_expert_params) | set(vit_expert_params)
    special_params.update(model.net.vit_router.parameters())
    special_params.update(model.net.Unet_router.parameters())
    special_params.update(model.net.output_proj_vit.parameters())
    special_params.update(model.net.output_proj_unet.parameters())
    special_params.update(model.net.blend_residual.parameters())
    attn_params = [p for p in model.net.pooled_cross_attn.parameters() if p not in gate_param_set]
    attn_params += [p for p in model.net.output_proj_attn.parameters() if p not in gate_param_set]
    special_params.update(attn_params)
    default_params = [p for p in model.parameters() if p not in special_params]

    param_groups = [
        {'params': unet_expert_params, 'lr': optim_configs['lr_unet']},
        {'params': vit_expert_params, 'lr': optim_configs['lr_vit']},
        {'params': list(model.net.vit_router.parameters()), 'lr': optim_configs['lr_router']},
        {'params': list(model.net.Unet_router.parameters()), 'lr': optim_configs['lr_router']},
        {'params': list(model.net.output_proj_vit.parameters()), 'lr': optim_configs['lr_out_proj_vit']},
        {'params': list(model.net.output_proj_unet.parameters()), 'lr': optim_configs['lr_out_proj_unet']},
        {'params': list(model.net.blend_residual.parameters()), 'lr': optim_configs['lr_blend_residual']},
        {'params': gate_params, 'lr': optim_configs['lr_gates']},
        {'params': default_params, 'lr': optim_configs['LR_DEFAULT']},
        {'params': attn_params, 'lr': optim_configs['lr_pooled_attn']},
    ]
    opt = torch.optim.Adam(param_groups, betas=optim_configs['betas'], weight_decay=0.0, eps=optim_configs['eps'])
    sched = Lr_Scheduler(
        opt, total_steps=model_configs['total_steps'], batch_size=model_configs['batch_size'],
        ref_batches=optim_configs['ref_batches'], rampup_Mimg=optim_configs['rampup_Mimg'],
        dataset_size=model_configs['dataset_size'], cur_nimg=0,
    )
    u_gen = MaskGenerator(
        mask_configs["unet_attr"], mask_configs["p_mean"], mask_configs["p_std"],
        bandwidth=mask_configs["max_BW_u"], min_bandwidth=mask_configs["min_BW_u"],
        total_steps=mask_configs["total_steps"], noise_range=mask_configs["unet_noise_range"],
        strat_band=mask_configs["strat_band"], sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"], alpha_ext=model_configs["extreme_prob"],
    )
    v_gen = MaskGenerator(
        mask_configs["vit_attr"], mask_configs["p_mean"], mask_configs["p_std"],
        bandwidth=mask_configs["max_BW_v"], min_bandwidth=mask_configs["min_BW_v"],
        total_steps=mask_configs["total_steps"], noise_range=mask_configs["vit_noise_range"],
        strat_band=mask_configs["strat_band"], sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"], alpha_ext=model_configs["extreme_prob"],
    )
    crit = build_criterion_from_configs(u_gen, v_gen)
    z_sched = ZetaScheduler(
        total_steps=zeta_configs["total_schedule_steps"], max_zeta=zeta_configs["max_zeta"],
        min_zeta=zeta_configs["min_zeta"], strategy=zeta_configs["strategy"], warmup_ratio=zeta_configs["warmup_ratio"],
    )

    B = 16
    print("  step | lr_gates |   vit_g |  unet_g |  time_g | blend_g |  attn_g |   MSE  | pure_L |  vit_L | unet_L | g_skip/g_blk0")
    for step in range(251):
        x0 = make_1f_latents(B, 4, 32, target_std=0.8, seed=200 + step)
        sigma = sample_sigma_hybrid(
            B, sigma_min=model_configs["sigma_min"], sigma_max=model_configs["sigma_max"],
            p_mean=mask_configs["p_mean"], p_std=mask_configs["p_std"],
            extreme_prob=model_configs["extreme_prob"], device="cpu",
        )
        noisy_x = x0 + torch.randn_like(x0) * sigma
        txt = torch.randn(B, 77, 64)
        out = model(noisy_x, sigma, txt, u_gen(sigma, step), v_gen(sigma, step), zeta=z_sched.get_zeta(step), return_log_var=True)
        ld = crit(x0, noisy_x, sigma, out)
        opt.zero_grad()
        ld["loss"].backward()

        g_skip = model.net.VIT_experts[0].unpatch_skip.weights.grad.norm().item()
        g_blk0 = math.sqrt(sum((p.grad.norm().item() ** 2) for p in model.net.VIT_experts[0].diffit[0].parameters() if p.grad is not None))
        ratio = g_skip / max(g_blk0, 1e-12)
        if step in (0, 50, 100, 150, 200, 250):
            print(
                f"  {step:4d} | {opt.param_groups[7]['lr']:.2e} | {model.net.vit_proj_gate.item():+.5f} | "
                f"{model.net.unet_proj_gate.item():+.5f} | {model.net.time_proj_gate.item():+.5f} | "
                f"{model.net.blend_gate.item():+.5f} | {model.net.pooled_cross_attn.out_proj_gate.item():+.5f} | "
                f"{ld['denoising'].item():.4f} | {ld['pure_loss'].item():.4f} | {ld['loss_texture'].item():.4f} | "
                f"{ld['loss_structure'].item():.4f} | {ratio:.2f}x"
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        sched.step()


def probe_2_per_sigma_gradient_and_output_scale():
    print("\n" + "=" * 84)
    print("PROBE 2: Per-Sigma Gradient Distribution & Unet/ViT Output Scale (Post-Init)")
    print("=" * 84)
    torch.manual_seed(11)
    model = build_realistic_scaled_model(vit_blocks=4, unet_blocks=2)
    with torch.no_grad():
        # Simulate gates opened to 0.5
        model.net.vit_proj_gate.fill_(0.5)
        model.net.unet_proj_gate.fill_(0.5)
        model.net.time_proj_gate.fill_(0.1)
        model.net.blend_gate.fill_(0.1)
        model.net.pooled_cross_attn.out_proj_gate.fill_(0.1)
    model.train()

    u_gen = MaskGenerator(
        mask_configs["unet_attr"], mask_configs["p_mean"], mask_configs["p_std"],
        bandwidth=mask_configs["max_BW_u"], min_bandwidth=mask_configs["min_BW_u"],
        total_steps=mask_configs["total_steps"], noise_range=mask_configs["unet_noise_range"],
        strat_band=mask_configs["strat_band"], sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"], alpha_ext=model_configs["extreme_prob"],
    )
    v_gen = MaskGenerator(
        mask_configs["vit_attr"], mask_configs["p_mean"], mask_configs["p_std"],
        bandwidth=mask_configs["max_BW_v"], min_bandwidth=mask_configs["min_BW_v"],
        total_steps=mask_configs["total_steps"], noise_range=mask_configs["vit_noise_range"],
        strat_band=mask_configs["strat_band"], sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"], alpha_ext=model_configs["extreme_prob"],
    )
    crit = build_criterion_from_configs(u_gen, v_gen)

    x0 = make_1f_latents(8, 4, 32, target_std=0.8, seed=99)
    txt = torch.randn(8, 77, 64)
    print("  sigma  | w_vit  | w_unet | F_vit_rms | F_unet_rms | grad_vit_exp | grad_unet_exp | vel_rms")
    for s in [0.002, 0.02, 0.1, 0.5, 2.5, 10.0, 40.0, 80.0]:
        model.zero_grad()
        sig = torch.full((8, 1, 1, 1), s)
        noisy = x0 + torch.randn_like(x0) * sig
        out = model(noisy, sig, txt, u_gen(sig, 0), v_gen(sig, 0), zeta=0.0, return_log_var=True)
        ld = crit(x0, noisy, sig, out)
        ld["loss"].backward()
        gv = math.sqrt(sum((p.grad.norm().item() ** 2) for p in model.net.VIT_experts.parameters() if p.grad is not None))
        gu = math.sqrt(sum((p.grad.norm().item() ** 2) for p in model.net.Unet_experts.parameters() if p.grad is not None))
        # Recover F_vit and F_unet RMS via c_out
        c_skip = (0.8 ** 2) / (s ** 2 + 0.8 ** 2)
        c_out = (s * 0.8) / math.sqrt(s ** 2 + 0.8 ** 2)
        f_v = ((out["vit_denoised"] - c_skip * noisy) / c_out).pow(2).mean().sqrt().item()
        f_u = ((out["unet_denoised"] - c_skip * noisy) / c_out).pow(2).mean().sqrt().item()
        vel = ((noisy - out["denoised"]) / s).pow(2).mean().sqrt().item()
        print(f"  {s:6.3f} | {out['w_vit'].mean().item():.4f} | {out['w_unet'].mean().item():.4f} |   {f_v:.4f}  |   {f_u:.4f}   |   {gv:.4e} |   {gu:.4e}  | {vel:.4f}")


def probe_3_sampler_and_logger_checks():
    print("\n" + "=" * 84)
    print("PROBE 3: EDM_Sampler Stability (Guided vs Unmasked) & Logger Overwrite Check")
    print("=" * 84)
    torch.manual_seed(12)
    model = build_realistic_scaled_model(vit_blocks=2, unet_blocks=1)
    with torch.no_grad():
        model.net.vit_proj_gate.fill_(0.5)
        model.net.unet_proj_gate.fill_(0.5)
    model.eval()

    u_gen = MaskGenerator(
        mask_configs["unet_attr"], mask_configs["p_mean"], mask_configs["p_std"],
        bandwidth=mask_configs["max_BW_u"], min_bandwidth=mask_configs["min_BW_u"],
        total_steps=mask_configs["total_steps"], noise_range=mask_configs["unet_noise_range"],
        strat_band=mask_configs["strat_band"], sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"], alpha_ext=model_configs["extreme_prob"],
    )
    v_gen = MaskGenerator(
        mask_configs["vit_attr"], mask_configs["p_mean"], mask_configs["p_std"],
        bandwidth=mask_configs["max_BW_v"], min_bandwidth=mask_configs["min_BW_v"],
        total_steps=mask_configs["total_steps"], noise_range=mask_configs["vit_noise_range"],
        strat_band=mask_configs["strat_band"], sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"], alpha_ext=model_configs["extreme_prob"],
    )

    noise = torch.randn(2, 4, 32, 32)
    txt = torch.randn(2, 77, 64)
    bos = torch.randn(1, 1, 64)
    eos = torch.randn(1, 1, 64)
    uncond_clip = torch.cat([bos, eos.expand(1, 76, 64)], dim=1).expand(2, 77, 64).clone()

    for label, g, ug, vg, u_txt in [
        ("g=1.0, unmasked, cond", 1.0, None, None, None),
        ("g=5.0, unmasked, clip('')", 5.0, None, None, uncond_clip),
        ("g=5.0, masked, clip('')", 5.0, u_gen, v_gen, uncond_clip),
    ]:
        sampler = EDM_Sampler(
            model, num_solve_steps=16, sigma_min=0.002, sigma_max=80.0,
            guidance=g, unet_mask_gen=ug, vit_mask_gen=vg,
        )
        out = sampler.sample(noise.clone(), text_emb=txt, uncond_text_emb=u_txt)
        print(f"  {label:28s}: finite={torch.isfinite(out).all().item()}, mean={out.mean().item():+.4f}, std={out.std().item():.4f}")

    # Verify unused laplacian_decomposition vs EDM_LOSS_NOD source
    crit = build_criterion_from_configs(u_gen, v_gen)
    print(f"  EDM_LOSS_NOD lap_sigma={crit.lap_sigma}, lap_kernel={crit.lap_kernel} (unused in __call__!)")


if __name__ == "__main__":
    probe_1_250_step_warmup_and_branch_learning()
    probe_2_per_sigma_gradient_and_output_scale()
    probe_3_sampler_and_logger_checks()
