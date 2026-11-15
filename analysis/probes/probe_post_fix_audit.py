"""Read-only second-pass diagnostic audit of the updated `text_guided` HDMOEM_v3 codebase."""
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_config_v3 import preconditioned_HDMOEM_v3
from models.model_components import Vit_expert, Unet_expert, ParametricTrajectoryRouter
from models.model_internals import MP_Conv, MP_Attention, mp_sum, normalize
from Utils.configs import model_configs, loss_configs, optim_configs, mask_configs, zeta_configs
from Utils.loss import EDM_LOSS_NOD, PathPriorLoss
from Utils.utils import sample_sigma_hybrid, ZetaScheduler, MaskGenerator, Lr_Scheduler, EMA
from Utils.cdf_utils import mixture_cdf, branch_cdf_bounds, branch_local_quantile_centers
from Utils.EDM_sampler import EDM_Sampler
from graphs.logger2 import _entropy, _max_entropy, _gini


def compute_rapsd(x: torch.Tensor) -> torch.Tensor:
    fft2 = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    power = (fft2.abs() ** 2).mean(dim=(0, 1))
    H, W = power.shape
    yy, xx = torch.meshgrid(torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing="ij")
    r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2).round().int()
    max_r = H // 2
    out = torch.zeros(max_r + 1)
    for k in range(max_r + 1):
        m = (r == k)
        if m.any():
            out[k] = power[m].mean()
    return out


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


def build_realistic_scaled_model(
    img_res: int = 32,
    internal_channels: int = 48,
    vit_emb: int = 96,
    unet_ch: int = 32,
    vit_blocks: int = 10,
    unet_blocks: int = 3,
    text_dim: int = 64,
    time_dim: int = 128,
):
    return preconditioned_HDMOEM_v3(
        IN_in_channels=4,
        IN_img_resolution=img_res,
        internal_channels=internal_channels,
        time_emb_dim=time_dim,
        text_emb_dim=text_dim,
        num_experts=2,
        top_k=1,
        Fourier_bandwidth=model_configs["fourier_bandwidth"],
        VIT_num_blocks=vit_blocks,
        VIT_patch_sizes=[2, 2],
        VIT_num_groups=4,
        VIT_num_heads=4,
        VIT_emb_size=vit_emb,
        VIT_pos_emb_balance=model_configs["VIT_pos_emb_balances"],
        VIT_emb_balance=model_configs["VIT_emb_balances"],
        Unet_num_blocks=unet_blocks,
        Unet_channel_mult=model_configs["Unet_channel_mult"],
        Unet_kernel_sizes=model_configs["Unet_kernel_sizes"],
        Unet_model_channels=unet_ch,
        Unet_channel_mult_emb=model_configs["Unet_channel_mult_emb"],
        Unet_label_balance=model_configs["Unet_label_balance"],
        Unet_concat_balance=model_configs["Unet_concat_balances"],
        sigma_data=model_configs["sigma_data"],
        log_var_channels=model_configs["log_var_channels"],
        transition_sigma=model_configs["transition_sigma"],
        transition_sharpness=model_configs["transition_sharpness"],
        pool_size=8,
        attn_pool_balance=model_configs["attn_pool_balance"],
        p_mean=mask_configs["p_mean"],
        p_std=mask_configs["p_std"],
        sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"],
        alpha_ext=model_configs["extreme_prob"],
    )


def build_criterion_from_configs(u_mask_gen, v_mask_gen):
    return EDM_LOSS_NOD(
        expert_centers_vit=v_mask_gen.expert_centers,
        expert_centers_Unet=u_mask_gen.expert_centers,
        num_experts=model_configs["num_experts"],
        sigma_data=model_configs["sigma_data"],
        Unet_bal=loss_configs["unet_bal"],
        vit_bal=loss_configs["vit_bal"],
        z_bal=loss_configs["z_bal"],
        prior_bal=loss_configs["prior_bal"],
        p_mean=mask_configs["p_mean"],
        p_std=mask_configs["p_std"],
        sharpness=loss_configs["sharpness"],
        transition_sigma=model_configs["transition_sigma"],
        overlap_width=model_configs["transition_sharpness"],
        alpha_blend=loss_configs["alpha_blend"],
        alpha_unet=loss_configs["alpha_unet"],
        alpha_vit=loss_configs["alpha_vit"],
        lap_kernel=loss_configs["laplacian_kernel"],
        lap_sigma=loss_configs["laplacian_std"],
        sigma_min=model_configs["sigma_min"],
        sigma_max=model_configs["sigma_max"],
        alpha_ext=model_configs["extreme_prob"],
        vit_noise_range=mask_configs["vit_noise_range"],
        unet_noise_range=mask_configs["unet_noise_range"],
    )


def probe_a_step0_gradients_and_internal_rms():
    print("=" * 84)
    print("PROBE A: Step-0 Gradients, Remaining Eliminating Singularities & Internal RMS")
    print("=" * 84)
    torch.manual_seed(0)
    model = build_realistic_scaled_model()
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

    B = 16
    x0 = make_1f_latents(B, 4, 32, target_std=0.8)
    sigma = sample_sigma_hybrid(
        B, sigma_min=model_configs["sigma_min"], sigma_max=model_configs["sigma_max"],
        p_mean=mask_configs["p_mean"], p_std=mask_configs["p_std"],
        extreme_prob=model_configs["extreme_prob"], device="cpu",
    )
    noisy_x = x0 + torch.randn_like(x0) * sigma
    txt = torch.randn(B, 77, 64)
    u_mask = u_gen(sigma, step=0)
    v_mask = v_gen(sigma, step=0)

    out = model(noisy_x, sigma, txt, u_mask, v_mask, zeta=0.0, return_log_var=True)
    loss_dict = crit(x0, noisy_x, sigma, out)
    loss_dict["loss"].backward()

    groups = {
        "input_proj+time_stem": [p for n, p in model.net.named_parameters() if n.startswith(("input_proj", "input_time_proj", "Fourier_emb", "out_fourier"))],
        "Unet_experts_all": [p for p in model.net.Unet_experts.parameters()],
        "VIT_experts_all": [p for p in model.net.VIT_experts.parameters()],
        "output_proj_unet": [p for p in model.net.output_proj_unet.parameters()],
        "output_proj_vit": [p for p in model.net.output_proj_vit.parameters()],
        "pooled_cross_attn": [p for p in model.net.pooled_cross_attn.parameters()],
        "output_proj_attn": [p for p in model.net.output_proj_attn.parameters()],
        "blend_residual": [p for p in model.net.blend_residual.parameters()],
        "routers": [p for p in model.net.Unet_router.parameters()] + [p for p in model.net.vit_router.parameters()],
        "log_var_linear": [p for p in model.log_var_linear.parameters()],
    }
    for gname, plist in groups.items():
        zero_cnt = sum(1 for p in plist if p.grad is None or p.grad.abs().max().item() == 0.0)
        gnorm = math.sqrt(sum((p.grad.norm().item() ** 2) for p in plist if p.grad is not None))
        print(f"  {gname:22s}: tensors={len(plist):3d}, exact_zero_grad={zero_cnt:3d}, total_grad_norm={gnorm:.4e}")

    print("  Scalar gates & gains at step 0:")
    for n, p in model.net.named_parameters():
        leaf = n.split(".")[-1]
        if leaf.endswith("_gate") or leaf.startswith("out_gain_"):
            if "Unet_experts.1" in n or "VIT_experts.1" in n:
                continue
            if "Unet_experts.0" in n and not any(k in n for k in ["32x32_block0", "16x16_down", "8x8_in0", "32x32_up"]):
                continue
            if "VIT_experts.0" in n and not any(k in n for k in ["diffit.0.", "diffit.9."]):
                continue
            gval = p.grad.norm().item() if p.grad is not None else -1.0
            print(f"    {n:52s}: val={p.item():+.4f}, grad={gval:.4e}")

    # Trace forward activation RMS inside Unet_expert and Vit_expert
    model.eval()
    with torch.no_grad():
        c_in = 1.0 / (model.sigma_data ** 2 + 1.0 ** 2) ** 0.5
        t_vec = torch.full((4,), math.log(1.0) / 4.0)
        t_emb = model.net.out_fourier2(F.silu(model.net.out_fourier1(model.net.Fourier_emb(t_vec))) / 0.596)
        x_in = torch.randn(4, 4, 32, 32)
        xf = normalize(model.net.input_proj(x_in))
        txt4 = torch.randn(4, 77, 64)

        # UNet forward RMS trace
        u_exp = model.net.Unet_experts[0]
        emb_u = F.silu(u_exp.map_noise(t_emb)) / 0.596
        xu = torch.cat([xf, torch.ones_like(xf[:, :1])], dim=1)
        skips = []
        u_rms = []
        for name, blk in u_exp.encoders.items():
            xu = blk(xu) if "conv" in name else blk(xu, embedding=emb_u)
            skips.append(xu)
            u_rms.append((name, xu.pow(2).mean().sqrt().item()))
        if u_exp.bottleneck_attn is not None:
            xu = u_exp.bottleneck_attn(xu, txt4)
            u_rms.append(("bottleneck_attn", xu.pow(2).mean().sqrt().item()))
        for name, blk in u_exp.decoders.items():
            if "block" in name:
                xu = models_mp_cat(xu, skips.pop(), t=u_exp.concat_balance)
            xu = blk(xu, embedding=emb_u)
            u_rms.append((name, xu.pow(2).mean().sqrt().item()))
        xu = u_exp.out_conv(xu, gain=1.0)
        u_rms.append(("out_conv", xu.pow(2).mean().sqrt().item()))
        print("  Unet_expert[0] layer-by-layer RMS (sampled):")
        for nm, rv in u_rms[::3] + [u_rms[-1]]:
            print(f"    {nm:20s}: rms={rv:.4f}")

        # ViT forward RMS trace
        v_exp = model.net.VIT_experts[0]
        xv = torch.cat([xf, torch.ones_like(xf[:, :1])], dim=1)
        xv = v_exp.patch(xv)
        xv = v_exp.pos_encoder(xv.flatten(2).transpose(1, 2), t=v_exp.pos_emb_balance)
        v_rms = [("post_patch+pos", xv.pow(2).mean().sqrt().item())]
        for bi, blk in enumerate(v_exp.diffit):
            xv = blk(xv, time_embedding=t_emb, text_embedding=txt4)
            if bi in (0, 4, 9):
                v_rms.append((f"diffit.{bi}", xv.pow(2).mean().sqrt().item()))
        xv_norm = normalize(xv)
        xv_up = v_exp.unpatch_proj(xv_norm.reshape(4 * v_exp.seq_ln, v_exp.emb_dim))
        xv_up = xv_up.reshape(4, v_exp.seq_ln, v_exp.in_channels).transpose(1, 2).view(4, v_exp.in_channels, 16, 16)
        xv_interp = F.interpolate(xv_up, size=(32, 32), mode="bilinear", align_corners=False)
        v_rms.append(("post_bilinear", xv_interp.pow(2).mean().sqrt().item()))
        xv_ref = v_exp.unpatch_refine(mp_sum(normalize(xv_interp), v_exp.unpatch_skip(normalize(xf)), t=0.5))
        v_rms.append(("post_refine", xv_ref.pow(2).mean().sqrt().item()))
        print("  Vit_expert[0] layer-by-layer RMS:")
        for nm, rv in v_rms:
            print(f"    {nm:20s}: rms={rv:.4f}")


def models_mp_cat(a, b, t):
    from models.model_internals import mp_cat
    return mp_cat(a, b, t=t)


def probe_b_gate_warmup_and_unpatch_skip_capture():
    print("\n" + "=" * 84)
    print("PROBE B: Early-Step Gate Escape Under Exact optim_configs & unpatch_skip Dominance")
    print("=" * 84)
    torch.manual_seed(1)
    model = build_realistic_scaled_model(vit_blocks=4, unet_blocks=2)
    model.train()

    # Exact parameter grouping from Utils/training.py lines 96-135
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
    print("  step | lr_gates |   vit_g |  unet_g |  time_g | blend_g |  attn_g | g_unp_skip | g_diffit_blk0 | pure_L | vit_L | unet_L")
    for step in range(41):
        x0 = make_1f_latents(B, 4, 32, target_std=0.8, seed=100 + step)
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
        lr_g = opt.param_groups[7]["lr"]

        if step in (0, 1, 2, 5, 10, 20, 30, 40):
            print(
                f"  {step:4d} | {lr_g:.2e} | {model.net.vit_proj_gate.item():+.5f} | "
                f"{model.net.unet_proj_gate.item():+.5f} | {model.net.time_proj_gate.item():+.5f} | "
                f"{model.net.blend_gate.item():+.5f} | {model.net.pooled_cross_attn.out_proj_gate.item():+.5f} | "
                f"  {g_skip:.2e} |      {g_blk0:.2e} | {ld['pure_loss'].item():.4f} | {ld['loss_texture'].item():.4f} | {ld['loss_structure'].item():.4f}"
            )
        opt.step()
        sched.step()

    # Check effective angular step on MP_Conv vs scalar gate at peak LR (step 1953)
    w_unet = model.net.Unet_experts[0].encoders["16x16_block0"].conv_res1.weights
    N_j = w_unet[0].numel()
    filter_norm = w_unet.detach()[0].norm().item()
    print(f"\n  MP_Conv filter check: fan_in N_j={N_j}, ||w_j||_2={filter_norm:.4f} (expected sqrt(N_j)={math.sqrt(N_j):.4f})")
    print(f"  Peak Adam step (step ~1953): lr_vit={optim_configs['lr_vit']:.4f} rad/step, lr_unet={optim_configs['lr_unet']:.4f} rad/step, lr_gates={optim_configs['lr_gates']:.4f}/step")


def probe_c_vit_spectrum_and_cross_attn_text_rank():
    print("\n" + "=" * 84)
    print("PROBE C: New Vit_expert RAPSD Spectrum & Cross-Attn Text-Rank / CFG Mismatch")
    print("=" * 84)
    torch.manual_seed(2)
    # 1. RAPSD of new Vit_expert (patch=2, bilinear + unpatch_skip + unpatch_refine) vs PixelShuffle(2)
    x_lat = make_1f_latents(16, 4, 32, target_std=0.8, seed=7)
    x_feats = normalize(MP_Conv(4, 48, (3, 3))(x_lat))
    v_exp = Vit_expert(
        num_heads=4, num_groups=4, in_channels=48, seq_ln=256, emb_dim=96,
        num_blocks=6, patch_size=2, time_dim=128, text_dim=64,
    )
    v_exp.eval()
    out_proj = MP_Conv(49, 4, (3, 3))
    t_emb = torch.randn(16, 128)
    txt_varied = torch.randn(16, 77, 64)
    # Simulated CLIP "" empty string: 1 BOS + 1 EOS + 75 identical PAD tokens
    bos = torch.randn(1, 1, 64)
    eos = torch.randn(1, 1, 64)
    txt_empty = torch.cat([bos, eos.expand(1, 76, 64)], dim=1).expand(16, 77, 64).clone()

    with torch.no_grad():
        v_out_raw = v_exp(x_feats, time_emb=t_emb, text_emb=txt_varied)
        v_out_proj = out_proj(torch.cat([v_out_raw, torch.ones_like(v_out_raw[:, :1])], dim=1))
        ps = torch.nn.PixelShuffle(2)
        ps_in = MP_Conv(96, 48 * 4, ())(torch.randn(16 * 256, 96)).view(16, 16, 16, 192).permute(0, 3, 1, 2)
        ps_out = out_proj(torch.cat([ps(ps_in), torch.ones_like(x_feats[:, :1])], dim=1))

        # Test bypass sensitivity: compare Vit_expert output when diffit blocks are zeroed vs normal
        v_no_txt = v_exp(x_feats, time_emb=t_emb, text_emb=None)
        v_empty_txt = v_exp(x_feats, time_emb=t_emb, text_emb=txt_empty)

    p_ref = compute_rapsd(x_lat)
    p_new = compute_rapsd(v_out_proj)
    p_ps = compute_rapsd(ps_out)
    print("  RAPSD ratio vs 1/f^2 latent baseline (r=1..15 cycles/32):")
    for r in [1, 2, 4, 8, 12, 14, 15]:
        print(f"    r={r:2d} | new_Vit_head/ref={p_new[r]/p_ref[r]:.3f}x | old_PixelShuffle/ref={p_ps[r]/p_ref[r]:.3f}x")

    # 2. Measure spatial uniformity / token cosine similarity inside Vit_expert across 10 blocks
    v10 = Vit_expert(
        num_heads=4, num_groups=4, in_channels=48, seq_ln=256, emb_dim=96,
        num_blocks=10, patch_size=2, time_dim=128, text_dim=64,
    )
    v10.eval()
    with torch.no_grad():
        for label, tx in [("text_emb=None (generate_test)", None), ("text_emb=clip('') (CFG train)", txt_empty), ("text_emb=varied", txt_varied)]:
            h = torch.cat([x_feats, torch.ones_like(x_feats[:, :1])], dim=1)
            h = v10.pos_encoder(v10.patch(h).flatten(2).transpose(1, 2), t=v10.pos_emb_balance)
            cos_means = []
            for bi, blk in enumerate(v10.diffit):
                h = blk(h, time_embedding=t_emb, text_embedding=tx)
                if bi in (0, 3, 6, 9):
                    hn = F.normalize(h, dim=-1)
                    # Mean pairwise cosine similarity between spatial tokens (excluding diagonal)
                    gram = torch.bmm(hn, hn.transpose(1, 2))
                    N = h.shape[1]
                    off_diag = (gram.sum(dim=(1, 2)) - N) / (N * (N - 1))
                    cos_means.append((bi, off_diag.mean().item()))
            out_branch = v10(x_feats, time_emb=t_emb, text_emb=tx)
            s_str = ", ".join([f"blk{bi}:{c:+.4f}" for bi, c in cos_means])
            print(f"  {label:30s} | token_cos=[{s_str}] | out_std={out_branch.std().item():.4f}")

        # Train-vs-inference mismatch between text_emb=None and text_emb=txt_empty
        out_none = v10(x_feats, time_emb=t_emb, text_emb=None)
        out_emp = v10(x_feats, time_emb=t_emb, text_emb=txt_empty)
        cos_mismatch = F.cosine_similarity(out_none.flatten(1), out_emp.flatten(1), dim=1).mean().item()
        rel_l2 = (out_none - out_emp).norm() / out_emp.norm()
        print(f"  Train/Infer Uncond Mismatch (None vs clip('')): cos_sim={cos_mismatch:.4f}, rel_L2={rel_l2:.4f}")

    # 3. Check how much of Vit_expert output variance comes from unpatch_skip vs 10 diffit blocks
    with torch.no_grad():
        h = torch.cat([x_feats, torch.ones_like(x_feats[:, :1])], dim=1)
        h = v10.pos_encoder(v10.patch(h).flatten(2).transpose(1, 2), t=v10.pos_emb_balance)
        for blk in v10.diffit:
            h = blk(h, time_embedding=t_emb, text_embedding=txt_varied)
        h_up = v10.unpatch_proj(normalize(h).reshape(16 * 256, 96)).reshape(16, 256, 48).transpose(1, 2).view(16, 48, 16, 16)
        h_up = normalize(F.interpolate(h_up, size=(32, 32), mode="bilinear", align_corners=False))
        d_skip = v10.unpatch_skip(normalize(x_feats))
        out_full = v10.unpatch_refine(mp_sum(h_up, d_skip, t=0.5))
        out_skip_only = v10.unpatch_refine(mp_sum(torch.zeros_like(h_up), d_skip, t=0.5))
        out_vit_only = v10.unpatch_refine(mp_sum(h_up, torch.zeros_like(d_skip), t=0.5))
        print(f"  Vit_expert branch share: cos(full, skip_only)={F.cosine_similarity(out_full.flatten(1), out_skip_only.flatten(1), dim=1).mean().item():.4f}, "
              f"cos(full, vit_only)={F.cosine_similarity(out_full.flatten(1), out_vit_only.flatten(1), dim=1).mean().item():.4f}")


def probe_d_router_starvation_and_mask_gap():
    print("\n" + "=" * 84)
    print("PROBE D: Exact Config Router Balance, UNet Expert-1 Starvation & MaskGenerator Gap")
    print("=" * 84)
    torch.manual_seed(3)
    p_mean = mask_configs["p_mean"]
    p_std = mask_configs["p_std"]
    s_min = model_configs["sigma_min"]
    s_max = model_configs["sigma_max"]
    t_sig = model_configs["transition_sigma"]
    a_ext = model_configs["extreme_prob"]

    cdf_trans, cdf_max = branch_cdf_bounds(t_sig, s_max, p_mean, p_std, s_min, s_max, a_ext)
    cdf_min, _ = branch_cdf_bounds(s_min, t_sig, p_mean, p_std, s_min, s_max, a_ext)
    u_centers = branch_local_quantile_centers(2, t_sig, s_max, p_mean, p_std, s_min, s_max, a_ext)
    v_centers = branch_local_quantile_centers(2, s_min, t_sig, p_mean, p_std, s_min, s_max, a_ext)
    print(f"  Exact configs: p_mean={p_mean}, p_std={p_std}, transition_sigma={t_sig}, alpha_ext={a_ext}")
    print(f"  Mixture CDF bounds: F_mix(sigma_min)={cdf_min:.4f}, F_mix(2.5)={cdf_trans:.4f}, F_mix(80)={cdf_max:.4f}")
    print(f"  ViT fraction (<2.5)={cdf_trans:.4f} | UNet fraction (>2.5)={1.0 - cdf_trans:.4f}")
    print(f"  ViT centers={v_centers.tolist()} | UNet centers={u_centers.tolist()}")

    u_router = ParametricTrajectoryRouter(
        2, t_sig, s_max, p_mean, p_std, init_sharpness=loss_configs["sharpness"],
        sigma_min=s_min, sigma_max=s_max, alpha_ext=a_ext,
    )
    v_router = ParametricTrajectoryRouter(
        2, s_min, t_sig, p_mean, p_std, init_sharpness=loss_configs["sharpness"],
        sigma_min=s_min, sigma_max=s_max, alpha_ext=a_ext,
    )
    u_router.eval()
    v_router.eval()

    u_gen = MaskGenerator(
        mask_configs["unet_attr"], p_mean, p_std, bandwidth=mask_configs["max_BW_u"],
        min_bandwidth=mask_configs["min_BW_u"], total_steps=mask_configs["total_steps"],
        noise_range=mask_configs["unet_noise_range"], strat_band=mask_configs["strat_band"],
        sigma_min=s_min, sigma_max=s_max, alpha_ext=a_ext,
    )
    v_gen = MaskGenerator(
        mask_configs["vit_attr"], p_mean, p_std, bandwidth=mask_configs["max_BW_v"],
        min_bandwidth=mask_configs["min_BW_v"], total_steps=mask_configs["total_steps"],
        noise_range=mask_configs["vit_noise_range"], strat_band=mask_configs["strat_band"],
        sigma_min=s_min, sigma_max=s_max, alpha_ext=a_ext,
    )

    sigmas = sample_sigma_hybrid(20000, s_min, s_max, p_mean, p_std, a_ext, device="cpu")
    log_s = torch.log(sigmas.view(-1, 1))
    with torch.no_grad():
        _, u_probs_nomask, _ = u_router(log_s, mask=None, zeta=0.0)
        _, v_probs_nomask, _ = v_router(log_s, mask=None, zeta=0.0)
        u_m_late = u_gen(sigmas, step=mask_configs["total_steps"])
        v_m_late = v_gen(sigmas, step=mask_configs["total_steps"])
        _, u_probs_late, _ = u_router(log_s, mask=u_m_late, zeta=0.0)
        _, v_probs_late, _ = v_router(log_s, mask=v_m_late, zeta=0.0)

    for tag, up, vp in [("No Mask", u_probs_nomask, v_probs_nomask), ("Late Mask (min_BW)", u_probs_late, v_probs_late)]:
        u_mean = up.mean(dim=0).tolist()
        v_mean = vp.mean(dim=0).tolist()
        u_ent = _entropy(up) / _max_entropy(2)
        v_ent = _entropy(vp) / _max_entropy(2)
        u_gini = _gini(up)
        v_gini = _gini(vp)
        u_exec = (up > 1e-5).float().mean(dim=0).tolist()
        v_exec = (vp > 1e-5).float().mean(dim=0).tolist()
        print(f"  [{tag}]")
        print(f"    UNet avg_usage={[round(x,4) for x in u_mean]} | exec_frac={[round(x,4) for x in u_exec]} | ent_norm={u_ent:.4f} (healthy>0.85) | gini={u_gini:.4f} (healthy<0.25)")
        print(f"    ViT  avg_usage={[round(x,4) for x in v_mean]} | exec_frac={[round(x,4) for x in v_exec]} | ent_norm={v_ent:.4f} (healthy>0.85) | gini={v_gini:.4f} (healthy<0.25)")

    # Check MaskGenerator late-stage gap in UNet:
    # u_centers = [0.8450, 0.9477], min_BW_u = 0.05 -> gap between 0.8950 and 0.8977
    u_both_active = (u_m_late.sum(dim=-1) == 2).float().mean().item()
    v_both_active = (v_m_late.sum(dim=-1) == 2).float().mean().item()
    print(f"  Late-stage 2-expert overlap fraction: UNet={u_both_active:.4f}, ViT={v_both_active:.4f}")


def probe_e_ema_resume_and_inference_pmean_drift():
    print("\n" + "=" * 84)
    print("PROBE E: EMA Resume Corruption Bug & generate_test.py p_mean Mismatch")
    print("=" * 84)
    # 1. EMA resume bug
    src = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        src.weight.fill_(1.0)
    ema_loaded = EMA(src, beta=model_configs["ema_beta"], warmup_power=model_configs["ema_warmup_power"], min_warmup_step=model_configs["ema_min_warmup_step"])
    # Simulate checkpoint ema_model having smoothed weight 0.5, online model perturbed to 1.0, resume at step 26000
    with torch.no_grad():
        ema_loaded.ema_model.weight.fill_(0.5)
    # Note: Utils/training.py loads ema_model_state_dict but NEVER restores ema.step (stays 0)
    expected_beta_at_26001 = min(model_configs["ema_beta"], 1.0 - (1.0 + 26001) ** (-model_configs["ema_warmup_power"]))
    ema_loaded.update()
    actual_beta_used = 1.0 - (1.0 + 1) ** (-model_configs["ema_warmup_power"])
    print(f"  EMA resume at step 26,000 (min_warmup_step={model_configs['ema_min_warmup_step']}):")
    print(f"    Expected cur_beta={expected_beta_at_26001:.5f} -> expected weight={0.5*expected_beta_at_26001 + 1.0*(1-expected_beta_at_26001):.5f}")
    print(f"    Actual   cur_beta={actual_beta_used:.5f} (ema.step reset to 0!) -> actual weight={ema_loaded.ema_model.weight.mean().item():.5f} (corrupted!)")

    # 2. generate_test.py p_mean=-0.4 vs training p_mean=-0.2 CDF mismatch
    s_grid = torch.tensor([0.05, 0.5, 2.5, 5.0, 15.0, 40.0])
    train_router = ParametricTrajectoryRouter(2, 2.5, 80.0, p_mean=-0.2, p_std=1.2, init_sharpness=25.0, sigma_min=0.002, sigma_max=80.0, alpha_ext=0.2)
    infer_router = ParametricTrajectoryRouter(2, 2.5, 80.0, p_mean=-0.4, p_std=1.2, init_sharpness=25.0, sigma_min=0.002, sigma_max=80.0, alpha_ext=0.2)
    # Simulate loading checkpoint weights (centers, log_sharpness, branch_span) into infer_router (p_mean stays -0.4!)
    infer_router.load_state_dict(train_router.state_dict())
    train_router.eval()
    infer_router.eval()
    with torch.no_grad():
        _, p_train, _ = train_router(torch.log(s_grid).view(-1, 1))
        _, p_infer, _ = infer_router(torch.log(s_grid).view(-1, 1))
    print("  generate_test.py p_mean mismatch (-0.4 infer vs -0.2 train) UNet router prob shifts:")
    for s, pt, pi in zip(s_grid.tolist(), p_train.tolist(), p_infer.tolist()):
        print(f"    sigma={s:6.3f} | train_p={[round(v,4) for v in pt]} | infer_p={[round(v,4) for v in pi]} | delta={abs(pt[0]-pi[0]):.4f}")


if __name__ == "__main__":
    probe_a_step0_gradients_and_internal_rms()
    probe_b_gate_warmup_and_unpatch_skip_capture()
    probe_c_vit_spectrum_and_cross_attn_text_rank()
    probe_d_router_starvation_and_mask_gap()
    probe_e_ema_resume_and_inference_pmean_drift()
