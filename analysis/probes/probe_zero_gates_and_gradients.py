"""Read-only diagnostic probe 2: Nested zero-init gates, pre-f945711 mp_sum contraction, and 150-step Adam dynamical evolution."""
import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.model_config_v3 import preconditioned_HDMOEM_v3
from models.model_internals import mp_sum, mp_gated_sum, resample
from Utils.loss import EDM_LOSS_NOD
from Utils.utils import sample_sigma_hybrid, ZetaScheduler, MaskGenerator, Lr_Scheduler


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


def build_compact_model(vit_patch=None, sigma_data=0.8):
    if vit_patch is None:
        vit_patch = [2, 2]
    model = preconditioned_HDMOEM_v3(
        IN_in_channels=4,
        IN_img_resolution=32,
        internal_channels=32,
        time_emb_dim=128,
        text_emb_dim=64,
        num_experts=2,
        top_k=1,
        Fourier_bandwidth=1.0,
        VIT_num_blocks=2,
        VIT_patch_sizes=vit_patch,
        VIT_num_groups=4,
        VIT_num_heads=4,
        VIT_emb_size=64,
        VIT_pos_emb_balance=[0.1, 0.1],
        VIT_emb_balance=[0.3, 0.3],
        Unet_num_blocks=2,
        Unet_channel_mult=[1, 2],
        Unet_kernel_sizes=[(3, 3), (3, 3)],
        Unet_model_channels=32,
        Unet_channel_mult_emb=2,
        Unet_label_balance=0.5,
        sigma_data=sigma_data,
        log_var_channels=64,
        transition_sigma=3.0,
        transition_sharpness=1.5,
        pool_size=4,
        attn_pool_balance=1.0,
        p_mean=-0.4,
        p_std=1.2,
    )
    return model


def main():
    torch.manual_seed(7)
    print("=" * 80)
    print("PROBE 2A: Pre-f945711 mp_sum Variance Contraction & Downsample Attenuation")
    print("=" * 80)
    x_unit = torch.randn(16, 64, 32, 32)
    z_zero = torch.zeros_like(x_unit)
    for t_bal in [0.5, 0.3]:
        s_old = x_unit.clone()
        s_new = x_unit.clone()
        rms_old, rms_new = [], []
        for l in range(1, 9):
            s_old = mp_sum(s_old, z_zero, t=t_bal)
            s_new = mp_gated_sum(s_new, x_unit, t=t_bal, gate=torch.zeros(()))
            if l in [1, 2, 4, 8]:
                rms_old.append(f"L{l}={s_old.pow(2).mean().sqrt().item():.4f}")
                rms_new.append(f"L{l}={s_new.pow(2).mean().sqrt().item():.4f}")
        print(f"balance t={t_bal}: mp_sum(x,0) RMS [{', '.join(rms_old)}] | mp_gated_sum(x,0) RMS [{', '.join(rms_new)}]")

    d_new = resample(x_unit, mode="down")
    d_old = d_new / 2.0  # pre-f945711 lacked * 2.0
    print(f"resample(down) single-step RMS: old (no *2.0)={d_old.pow(2).mean().sqrt().item():.4f} | new (*2.0)={d_new.pow(2).mean().sqrt().item():.4f}")

    print("\n" + "=" * 80)
    print("PROBE 2B: Exact Step-0 Backward Gradient Audit Across Zero-Init Gates & Weights")
    print("=" * 80)
    model = build_compact_model(vit_patch=[2, 2], sigma_data=0.8)
    model.train()
    print("HDMOEM_v3 children:", [n for n, _ in model.net.named_children()])
    vit_router = getattr(model.net, "vit_router", None) or getattr(model.net, "VIT_router", None)
    unet_router = getattr(model.net, "Unet_router", None) or getattr(model.net, "unet_router", None)
    centers_vit = vit_router.centers.detach().clone()
    centers_unet = unet_router.centers.detach().clone()
    loss_fn = EDM_LOSS_NOD(
        expert_centers_vit=centers_vit,
        expert_centers_Unet=centers_unet,
        num_experts=2,
        sigma_data=0.8,
        alpha_blend=0.8,
        alpha_unet=0.1,
        alpha_vit=0.1,
        lap_sigma=0.5,
        transition_sigma=3.0,
    )

    x0 = make_1f_latents(16, 4, 32, alpha=2.0, target_std=0.8, seed=1)
    sig = sample_sigma_hybrid(16, device=torch.device("cpu"))
    xn = x0 + sig.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
    text_emb = torch.randn(16, 64)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    m_unet = torch.ones(16, 2)
    m_vit = torch.ones(16, 2)

    out = model(xn, sig, text_emb, m_unet, m_vit, zeta=1.0, return_log_var=True)
    losses = loss_fn(x0, xn, sig, out)
    losses["loss"].backward()

    # Summarize gradient norms grouped by module role & exact zero count
    gate_stats = []
    zero_grad_groups = {"unet_inner": [0, 0], "vit_inner": [0, 0], "pooled_cross_attn": [0, 0],
                        "output_proj_unet": [0, 0], "output_proj_vit": [0, 0],
                        "input_stem": [0, 0], "routers": [0, 0], "log_var": [0, 0]}
    group_gnorms = {k: 0.0 for k in zero_grad_groups}

    for name, p in model.named_parameters():
        nl = name.lower()
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        if "gate" in nl:
            gate_stats.append((name, p.data.mean().item(), gn))
        if "output_proj_unet" in nl:
            key = "output_proj_unet"
        elif "output_proj_vit" in nl:
            key = "output_proj_vit"
        elif "router" in nl:
            key = "routers"
        elif "log_var" in nl:
            key = "log_var"
        elif "cross_attn" in nl:
            key = "pooled_cross_attn"
        elif "unet" in nl and "gate" not in nl:
            key = "unet_inner"
        elif "vit" in nl and "gate" not in nl:
            key = "vit_inner"
        else:
            key = "input_stem"
        zero_grad_groups[key][0] += 1
        if gn == 0.0:
            zero_grad_groups[key][1] += 1
        group_gnorms[key] += gn ** 2

    print("Step-0 Parameter Group Gradient Norms (Total L2 norm, zero-grad tensor counts):")
    for k, (tot_n, zero_n) in zero_grad_groups.items():
        print(f"  {k:20s}: grad_norm={math.sqrt(group_gnorms[k]):.4e} | zero_grad_tensors={zero_n}/{tot_n}")

    print("\nSample Zero-Init Gates at Step 0 (name, init_val, grad_norm):")
    for name, val, gn in gate_stats[:16]:
        print(f"  {name:55s}: val={val:+.4f}, grad={gn:.4e}")
    if len(gate_stats) > 16:
        print(f"  ... ({len(gate_stats) - 16} more gates, exact zero count={sum(1 for _,_,g in gate_stats if g==0)}/{len(gate_stats)})")

    print("\n" + "=" * 80)
    print("PROBE 2C: 150-Step Dynamical Optimization Trace from Step-0 Init (Adam LR=0.0012)")
    print("=" * 80)
    # Re-create fresh model to run clean Adam loop with Lr_Scheduler + ZetaScheduler
    torch.manual_seed(11)
    model2 = build_compact_model(vit_patch=[2, 2], sigma_data=0.8)
    model2.train()
    opt = torch.optim.AdamW(model2.parameters(), lr=1.2e-3, betas=(0.9, 0.99), weight_decay=0.0)
    def cosine_lr(step, peak_lr=1.2e-3, warmup=40, total=150, min_mult=0.10):
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return peak_lr * (min_mult + 0.5 * (1.0 - min_mult) * (1.0 + math.cos(math.pi * prog)))

    def wsd_lr(step, peak_lr=1.2e-3, warmup=400, total=8000, stable_frac=0.85, min_mult=0.10):
        if step < warmup:
            return peak_lr * (step + 1) / warmup
        stable_end = int(total * stable_frac)
        if step < stable_end:
            return peak_lr
        prog = (step - stable_end) / max(1, total - stable_end)
        return peak_lr * (1.0 - (1.0 - min_mult) * prog)

    lr_sched_total = 40
    z_sched = ZetaScheduler(total_steps=30, max_zeta=1.0, min_zeta=0.0, strategy="cos")
    vit_router2 = getattr(model2.net, "vit_router", None) or getattr(model2.net, "VIT_router", None)
    unet_router2 = getattr(model2.net, "Unet_router", None) or getattr(model2.net, "unet_router", None)
    loss_fn2 = EDM_LOSS_NOD(
        expert_centers_vit=vit_router2.centers.detach().clone(),
        expert_centers_Unet=unet_router2.centers.detach().clone(),
        num_experts=2, sigma_data=0.8, alpha_blend=0.8, alpha_unet=0.1, alpha_vit=0.1,
        lap_sigma=0.5, transition_sigma=3.0,
    )

    fixed_x = make_1f_latents(32, 4, 32, alpha=2.0, target_std=0.8, seed=77)
    fixed_txt = torch.randn(32, 64)
    fixed_txt = fixed_txt / fixed_txt.norm(dim=-1, keepdim=True)

    print(f"{'step':>4} | {'lr':>8} | {'zeta':>5} | {'tot':>6} | {'pure':>6} | {'vit(hi)':>7} | {'unet(lo)':>8} | {'MSE':>6} | "
          f"{'vit_g':>7} | {'unet_g':>7} | {'res_g0':>7} | {'tmsa_g0':>7} | {'u_in_gn':>8} | {'v_in_gn':>8}", flush=True)
    print("-" * 124, flush=True)

    for step in range(40):
        idx = torch.randint(0, 32, (4,))
        xb = fixed_x[idx]
        tb = fixed_txt[idx]
        sigb = sample_sigma_hybrid(4, device=torch.device("cpu"))
        xnb = xb + sigb.reshape(-1, 1, 1, 1) * torch.randn_like(xb)
        zeta_v = z_sched.get_zeta(step)
        lr_v = cosine_lr(step, peak_lr=1.2e-3, warmup=10, total=40, min_mult=0.10)
        for pg in opt.param_groups:
            pg["lr"] = lr_v

        opt.zero_grad()
        outb = model2(xnb, sigb, tb, torch.ones(4, 2), torch.ones(4, 2), zeta=zeta_v, return_log_var=True)
        ldict = loss_fn2(xb, xnb, sigb, outb)
        ldict["loss"].backward()

        # Measure inner branch grad norms before step
        u_in_gn = math.sqrt(sum(p.grad.pow(2).sum().item() for n, p in model2.named_parameters()
                                if "unet" in n.lower() and "expert" in n.lower() and "gate" not in n.lower() and p.grad is not None))
        v_in_gn = math.sqrt(sum(p.grad.pow(2).sum().item() for n, p in model2.named_parameters()
                                if "vit" in n.lower() and "expert" in n.lower() and "gate" not in n.lower() and p.grad is not None))

        opt.step()

        if step in [0, 1, 2, 3, 5, 10, 20, 30, 39]:
            vg = model2.net.vit_proj_gate.abs().mean().item()
            ug = model2.net.unet_proj_gate.abs().mean().item()
            rg0 = next(p.abs().mean().item() for n, p in model2.named_parameters() if "unet" in n.lower() and "res_gate" in n)
            tg0 = next(p.abs().mean().item() for n, p in model2.named_parameters() if "vit" in n.lower() and "tmsa_gate" in n)
            print(f"{step:4d} | {lr_v:8.5f} | {zeta_v:5.2f} | {ldict['loss']:6.4f} | {ldict['pure_loss']:6.4f} | "
                  f"{ldict['loss_texture']:7.4f} | {ldict['loss_structure']:8.4f} | {ldict['denoising']:6.4f} | "
                  f"{vg:7.4f} | {ug:7.4f} | {rg0:7.4f} | {tg0:7.4f} | {u_in_gn:8.2e} | {v_in_gn:8.2e}", flush=True)

    print("\n" + "=" * 80)
    print("PROBE 2D: Adam Tangential Update Budget Across Cosine (8k) vs WSD (8k) vs EDM2 (70k)")
    print("=" * 80)
    # Effective max angular rotation on normalized MP_Conv weight of radius sqrt(N_j)
    # Under Adam, |dw_i| ~ alpha(t), so ||dw||_2 ~ alpha(t)*sqrt(N_j) -> angular step dtheta = ||dw||/||w|| = alpha(t) * sqrt(N_j) / sqrt(N_j) = alpha(t) per direction?
    # Wait: EDM2 eq 66 weight is normalized to ||w||=sqrt(N_j) -> random unit sign step of +-alpha gives ||dw|| = alpha*sqrt(N_j), so dtheta ~ alpha(t)!
    # Whereas scalar gain with magnitude g (or init 0) changes by +-alpha(t) linearly.
    # Let's compute exact integrated LR sum(alpha(t)) over steps 0..T
    sum_cos = sum(cosine_lr(t, peak_lr=1.2e-3, warmup=400, total=8000, min_mult=0.10) for t in range(8000))
    sum_cos_tail = sum(cosine_lr(t, peak_lr=1.2e-3, warmup=400, total=8000, min_mult=0.10) for t in range(5000, 8000))
    sum_wsd = sum(wsd_lr(t, peak_lr=1.2e-3, warmup=400, total=8000, stable_frac=0.85, min_mult=0.10) for t in range(8000))
    sum_edm2_70k = sum(1.2e-3 / math.sqrt(max(t / 2000.0, 1.0)) for t in range(70000))
    print(f"Integrated LR sum int alpha(t) dt:")
    print(f"  v2 Cosine (8k steps):  total={sum_cos:.3f} (last 3000 steps contribute {sum_cos_tail:.3f} = {100*sum_cos_tail/sum_cos:.1f}%)")
    print(f"  v3 WSD    (8k steps):  total={sum_wsd:.3f}")
    print(f"  70k InvSqrt ref=1.2e-3: total={sum_edm2_70k:.3f} ({sum_edm2_70k/sum_cos:.1f}x more integrated Adam step budget)")


if __name__ == "__main__":
    main()
