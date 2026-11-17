"""Low-sigma information-bottleneck probe for the HDMOEM_v3 ViT (low-noise) branch.

Part A  (exact / structural, no training):
  A1  measured transfer of F.interpolate(bilinear, x2, align_corners=False) vs cos^3(w/2)
  A2  share of the low-sigma target (white noise) energy per token-path gain level
  A3  sigma below which the UNet branch is skipped, and how often that regime is hit
  A4  census of nonlinear ops by spatial resolution inside the real HDMOEM_v3 forward
  A5  numerical linearity check of the full-resolution path of Vit_expert

Part B  (controlled training experiment, low-sigma denoising only):
  Variants built from the repo's own classes, differing only in the full-res head.
  Reports per-band residual-noise ratio r_band = E|P_b(D-x0)|^2 / E|P_b n|^2.

Usage:  python probe_low_sigma_info_bottleneck.py [A|B|AB] [steps]
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import models.model_internals as mi  # noqa: E402
from models.model_internals import MP_Conv, MP_Fourier, normalize, mp_silu, mp_sum  # noqa: E402
from models.model_components import Vit_expert, Unet_block  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 64          # latent resolution in Utils/configs.py (img_resolution=64)
PATCH = 2         # VIT_patch_sizes=[2, 2]
SIGMA_D = 0.8


# ----------------------------------------------------------------------------
# Spectral helpers
# ----------------------------------------------------------------------------
def band_masks(res: int, device=DEV):
    """inner = token-grid baseband (|wx|,|wy| <= pi/2); outer = everything else."""
    f = torch.fft.fftfreq(res, device=device).abs()  # cycles/pixel in [0, 0.5]
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    inner = (fx <= 0.25) & (fy <= 0.25)
    return inner, ~inner


def band_energy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    X = torch.fft.fft2(x.float(), norm="ortho")
    return (X.abs() ** 2 * mask).sum(dim=(-2, -1)).sum(dim=-1)  # (B,)


# ----------------------------------------------------------------------------
# PART A
# ----------------------------------------------------------------------------
def part_a():
    print("=" * 78)
    print("PART A1: bilinear x2 (align_corners=False) transfer, measured vs cos^3(w/2)")
    print("=" * 78)
    n_tok = 32
    print(f"{'token f (cyc/tok)':>18} {'out f':>7} {'base gain':>10} {'cos^3':>8} "
          f"{'image f':>8} {'image gain':>11} {'cos^3':>8}")
    for k in [1, 4, 8, 12, 15]:
        t = torch.arange(n_tok, dtype=torch.float64)
        sig = torch.cos(2 * math.pi * k * t / n_tok)[None, None, None, :].repeat(1, 1, 1, 1)
        sig2d = sig.expand(1, 1, 4, n_tok).contiguous()
        up = F.interpolate(sig2d, scale_factor=(1, 2), mode="bilinear", align_corners=False)[0, 0, 0]
        U = torch.fft.fft(up) / up.numel()
        base = 2 * U[k].abs().item()            # baseband at k cycles / 64 px
        img = 2 * U[n_tok - k].abs().item()      # spectral image at (32 - k) cycles / 64 px
        wb = 2 * math.pi * k / (2 * n_tok)
        wi = 2 * math.pi * (n_tok - k) / (2 * n_tok)
        print(f"{k:>18d} {k:>7d} {base:>10.4f} {abs(math.cos(wb / 2)) ** 3:>8.4f} "
              f"{n_tok - k:>8d} {img:>11.4f} {abs(math.sin(wb / 2)) ** 3:>8.4f}")
    print("  -> bilinear x2 == binomial [1,3,3,1]/4 filter: H(w) = cos^3(w/2), "
          "3rd-order zero at Nyquist.")

    print("\n" + "=" * 78)
    print("PART A2: low-sigma target F* ~ -eps (white). Energy share by token-path gain")
    print("=" * 78)
    f = torch.fft.fftfreq(RES, dtype=torch.float64)
    w = 2 * math.pi * f
    g1 = torch.cos(w / 2).abs() ** 3
    G = g1[:, None] * g1[None, :]                      # separable 2D gain
    inner, outer = band_masks(RES, device="cpu")
    tot = G.numel()
    print(f"  outer band (unreachable from token baseband) : {outer.sum().item() / tot:6.2%} of white energy")
    for thr in [0.354, 0.125, 0.1, 0.01, 0.001]:
        print(f"  token-path gain < {thr:<6}                  : {(G < thr).sum().item() / tot:6.2%}")
    print(f"  max token-path gain anywhere in outer band   : {G[outer].max().item():.4f}")
    print(f"  mean token-path power gain in outer band     : {(G[outer] ** 2).mean().item():.5f}")

    print("\n" + "=" * 78)
    print("PART A3: regime where the UNet branch is switched off (w_unet < 0.05)")
    print("=" * 78)
    ts, sharp = 2.5, 0.8
    sigma_off = ts * math.exp(-sharp * math.log(0.95 / 0.05))
    print(f"  sigma_off = 2.5 * exp(-0.8 * logit(0.95)) = {sigma_off:.4f}")
    pm, ps, a = -0.2, 1.2, 0.2
    lmin, lmax = math.log(0.002), math.log(80.0)
    lo = math.log(sigma_off)
    p_train = (1 - a) * 0.5 * (1 + math.erf((lo - pm) / (ps * math.sqrt(2)))) + a * (lo - lmin) / (lmax - lmin)
    print(f"  fraction of TRAINING samples with sigma < sigma_off : {p_train:6.2%}")
    for N in [50, 100]:
        i = torch.arange(N, dtype=torch.float64)
        t = (80 ** (1 / 7) + i / (N - 1) * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
        print(f"  fraction of SAMPLER steps (N={N}, rho=7) below sigma_off: {(t < sigma_off).double().mean().item():6.2%}")

    print("\n" + "=" * 78)
    print("PART A4: census of nonlinear ops by resolution in the real HDMOEM_v3 forward")
    print("=" * 78)
    census_nonlinear_ops()

    print("\n" + "=" * 78)
    print("PART A5: linearity of the full-res path inside Vit_expert (skip -> refine)")
    print("=" * 78)
    torch.manual_seed(0)
    vit = Vit_expert(num_heads=4, num_groups=4, in_channels=32, seq_ln=(RES // PATCH) ** 2, emb_dim=64,
                     num_blocks=2, patch_size=PATCH, time_dim=64).to(DEV).eval()
    x1 = torch.randn(4, 32, RES, RES, device=DEV)
    x2 = torch.randn(4, 32, RES, RES, device=DEV)
    L = lambda z: vit.unpatch_refine(vit.unpatch_skip(z))  # what the skip contributes, pre-normalize
    lin_err = (L(x1 + x2) - L(x1) - L(x2)).norm() / L(x1 + x2).norm()
    sh = lambda z: torch.roll(z, shifts=(3, 5), dims=(-2, -1))
    # circular-shift equivariance holds away from zero-padded borders
    crop = (slice(None), slice(None), slice(8, -8), slice(8, -8))
    eqv_err = (L(sh(x1))[crop] - sh(L(x1))[crop]).norm() / L(x1)[crop].norm()
    print("  additivity error  |L(a+b)-L(a)-L(b)|/|L(a+b)| = {:.2e}".format(lin_err.item()))
    print("  shift-equivariance error (interior)           = {:.2e}".format(eqv_err.item()))
    print("  -> skip path = one linear shift-invariant filter (3x3 o 3x3, then 3x3 output_proj,"
          " and 3x3 input_proj before it: 9x9 effective).")

    print("\n" + "=" * 78)
    print("PART A6: Wiener impulse-response spatial support vs sigma (kernel-size test)")
    print("=" * 78)
    f = torch.fft.fftfreq(RES, dtype=torch.float64)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    fr2 = (fx ** 2 + fy ** 2).clamp(min=(1.0 / RES) ** 2)
    Sxx = 1.0 / fr2
    Sxx = Sxx / Sxx.mean() * (SIGMA_D ** 2)  # mean PSD = sigma_d^2 (no artificial DC hole)
    coords = torch.fft.fftfreq(RES, dtype=torch.float64) * RES  # integer pixel offsets in [-32, 31]
    cy, cx = torch.meshgrid(coords, coords, indexing="ij")
    max_cheb = torch.maximum(cy.abs(), cx.abs())  # L_inf radius: <=0 is 1x1, <=1 is 3x3, <=2 is 5x5, <=4 is 9x9
    print(f"{'sigma':>8} | {'1x1 (r=0)':>11} | {'3x3 (r<=1)':>12} | {'5x5 (r<=2)':>12} | {'9x9 (4x 3x3)':>14} | {'Wiener r_out':>12}")
    _, outer_cpu = band_masks(RES, device="cpu")
    for s in [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 2.0, 10.0]:
        c_skip = (SIGMA_D ** 2) / (s ** 2 + SIGMA_D ** 2)
        Hw = Sxx / (Sxx + s ** 2)
        # Score-branch linear filter G_W(k) = H_W(k) - c_skip(sigma) that F_theta must learn
        gw_spatial = torch.fft.ifft2(Hw - c_skip).real
        tot_e = (gw_spatial ** 2).sum().item()
        e1 = (gw_spatial[max_cheb == 0] ** 2).sum().item() / tot_e
        e3 = (gw_spatial[max_cheb <= 1] ** 2).sum().item() / tot_e
        e5 = (gw_spatial[max_cheb <= 2] ** 2).sum().item() / tot_e
        e9 = (gw_spatial[max_cheb <= 4] ** 2).sum().item() / tot_e
        # Exact Wiener outer-band residual-noise ratio E|P_out(D_W - x0)|^2 / E|P_out n|^2 = mean_{k in outer} Hw(k)
        r_w_out = Hw[outer_cpu].mean().item()
        print(f"{s:>8.3f} | {e1:10.2%} | {e3:11.2%} | {e5:11.2%} | {e9:13.2%} | {r_w_out:12.4f}")
    print("  -> At low sigma (<=0.2), >96-99.9% of optimal linear filter energy already fits inside a 3x3-9x9 support!")
    print("     Enlarging kernels from 3x3 (effective 9x9 over 4 layers) to 5x5 cannot break the linear Wiener ceiling.")


def census_nonlinear_ops():
    from models.model_config_v3 import HDMOEM_v3
    torch.manual_seed(0)
    res = RES
    net = HDMOEM_v3(
        IN_in_channels=4, IN_img_resolution=res, internal_channels=48, time_emb_dim=64, text_emb_dim=0,
        num_experts=2, top_k=1, Fourier_bandwidth=1.2, VIT_num_blocks=2, VIT_patch_sizes=[2, 2],
        VIT_num_groups=4, VIT_num_heads=4, VIT_emb_size=64, VIT_pos_emb_balance=[0.5, 0.5],
        VIT_emb_balance=[0.4, 0.4], Unet_num_blocks=1, Unet_channel_mult=[1, 2],
        Unet_kernel_sizes=[(3, 3), (5, 5)], Unet_model_channels=32, Unet_channel_mult_emb=2,
        Unet_concat_balance=[0.7, 0.3], transition_sigma=2.5, transition_sharpness=0.8,
        pool_size=16, p_mean=-0.2, p_std=1.2,
    ).to(DEV).eval()
    B = 2
    records = []
    orig_silu, orig_softmax = F.silu, F.softmax

    def classify(shape, kind):
        if kind == "silu" and len(shape) == 4:
            return shape[-1]
        if kind == "silu" and len(shape) == 2:
            rows = shape[0] // B
            side = int(math.isqrt(rows)) if rows > 1 else 0
            return side if side * side == rows and rows > 1 else "vector"
        if kind == "softmax" and len(shape) == 4:
            lq = shape[2]
            side = int(math.isqrt(lq))
            return side if side * side == lq and lq > 1 else "vector"
        return "vector"

    def silu_rec(x, *a, **k):
        records.append(("silu", classify(tuple(x.shape), "silu")))
        return orig_silu(x, *a, **k)

    def softmax_rec(x, *a, **k):
        records.append(("softmax", classify(tuple(x.shape), "softmax")))
        return orig_softmax(x, *a, **k)

    F.silu, F.softmax = silu_rec, softmax_rec
    try:
        for sigma in [0.05, 0.2, 1.0, 10.0]:
            records.clear()
            x = torch.randn(B, 4, res, res, device=DEV)
            sig = torch.full((B,), sigma, device=DEV)
            masks = torch.ones(B, 2, device=DEV)
            with torch.no_grad():
                out = net(x=x, time_vec=sig.log() / 4, text_emb=None,
                          Unet_router_mask=masks, Vit_router_mask=masks, zeta=0.0)
            counts = {}
            for kind, r in records:
                if kind == "softmax" and r == res:
                    key = f"{kind}@{r}x{r} (pooled cross-attn query, x gate {out['gate_strength'].mean().item():.3f})"
                else:
                    key = f"{kind}@{r}x{r}" if r != "vector" else f"{kind}@vector"
                counts[key] = counts.get(key, 0) + 1
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            print(f"  sigma={sigma:<5}  w_unet={out['w_unet'].mean().item():.3f}  ->  {summary}")
    finally:
        F.silu, F.softmax = orig_silu, orig_softmax
    print(f"  (full latent res = {res}x{res}; token grid = {res // 2}x{res // 2})")


# ----------------------------------------------------------------------------
# PART B: controlled low-sigma denoising experiment
# ----------------------------------------------------------------------------
def dead_leaves(n: int, res: int, seed: int, n_disks: int = 28) -> torch.Tensor:
    """Occlusion 'dead leaves' model: piecewise-constant regions with sharp step edges (non-Gaussian)."""
    g_mix = torch.Generator(device=DEV).manual_seed(0)
    mix = torch.randn(4, 4, generator=g_mix, device=DEV) * 0.25 + torch.eye(4, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    imgs = torch.zeros(n, 4, res, res, device=DEV)
    yy, xx = torch.meshgrid(torch.arange(res, device=DEV), torch.arange(res, device=DEV), indexing="ij")
    rmin, rmax = 3.0, res / 2.0
    for _ in range(n_disks):
        u = torch.rand(n, generator=g, device=DEV)
        r = (rmin ** -2 + u * (rmax ** -2 - rmin ** -2)) ** -0.5   # p(r) ~ r^-3
        cy = torch.rand(n, generator=g, device=DEV) * res
        cx = torch.rand(n, generator=g, device=DEV) * res
        col = torch.randn(n, 4, generator=g, device=DEV) @ mix.T
        m = ((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2) < r[:, None, None] ** 2
        imgs = torch.where(m[:, None], col[:, :, None, None], imgs)
    imgs = imgs - imgs.mean(dim=(0, 2, 3), keepdim=True)
    return imgs / imgs.std() * SIGMA_D


def gaussian_1f(n: int, res: int, seed: int) -> torch.Tensor:
    """Gaussian field with the same 1/f^2 PSD: linear Wiener is Bayes-optimal here (control)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    f = torch.fft.fftfreq(res, device=DEV)
    fr = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).clamp(min=1.0 / res)
    amp = 1.0 / fr
    z = torch.randn(n, 4, res, res, generator=g, device=DEV)
    x = torch.fft.ifft2(torch.fft.fft2(z) * amp).real
    x = x - x.mean(dim=(0, 2, 3), keepdim=True)
    return x / x.std() * SIGMA_D


class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = MP_Fourier(dim // 2, bandwidth=1.2)
        self.l1 = MP_Conv(dim // 2, dim * 2, kernel=())
        self.l2 = MP_Conv(dim * 2, dim, kernel=())

    def forward(self, c_noise):
        return self.l2(mp_silu(self.l1(self.f(c_noise))))


class LowSigmaDenoiser(nn.Module):
    """Replica of the sigma < 0.237 path of HDMOEM_v3: stem -> Vit_expert -> output_proj_vit.

    head: 'tokens_only'  ViT tokens (patch=2) + bilinear x2 + 3x3 refine (NO full-res skip)
          'linear'       full-res linear conv chain only (3x3 stem -> 3x3 skip -> 3x3 refine -> 3x3 out, NO tokens/nonlinearity)
          'current'      repo Vit_expert unchanged (bilinear + 3x3 skip + 3x3 refine + 3x3 out)
          'current_k5'   same as current but 5x5 kernels on skip, refine, and output_proj
          'fullres_nl'   current + 2 full-res nonlinear Unet_blocks (mp_silu, 3x3) before output_proj
          'patch1'       Vit_expert with patch_size=1 (full-res tokens & nonlinearities)
    """

    def __init__(self, head: str, res: int = 32, C=48, emb=128, heads=4, blocks=3, tdim=128):
        super().__init__()
        self.head = head
        self.res = res
        self.tdim = tdim
        p = 1 if head == "patch1" else PATCH
        self.temb = TimeEmb(tdim)
        self.input_proj = MP_Conv(4, C, kernel=(3, 3))
        self.input_time_proj = MP_Conv(tdim, C, kernel=())
        self.time_proj_gate = nn.Parameter(torch.full((1,), 0.1))
        self.vit = Vit_expert(num_heads=heads, num_groups=4, in_channels=C, seq_ln=(res // p) ** 2,
                              emb_dim=emb, num_blocks=blocks, patch_size=p, time_dim=tdim)
        k = (5, 5) if head == "current_k5" else (3, 3)
        if head == "current_k5":
            self.vit.unpatch_skip = MP_Conv(C, C, kernel=k)
            self.vit.unpatch_refine = MP_Conv(C, C, kernel=k)
        if head == "fullres_nl":
            self.nl = nn.ModuleList([Unet_block(C, C, kernel=(3, 3), emb_size=tdim, Type="dec", resample="keep")
                                     for _ in range(2)])
        self.output_proj = MP_Conv(C + 1, 4, kernel=k)
        # Initialize out_gain to 0.05 (near F=0 baseline so random init doesn't inject +100% noise variance,
        # while avoiding the exact 0 saddle).
        self.out_gain = nn.Parameter(torch.full((1,), 0.05))

    def features(self, x_in, temb):
        if self.head == "linear":
            return self.vit.unpatch_refine(self.vit.unpatch_skip(normalize(x_in)))
        if self.head == "tokens_only":
            v = self.vit
            z = torch.cat([x_in, torch.ones_like(x_in[:, :1])], dim=1)
            z = v.patch(z)
            b, _, hp, wp = z.shape
            z = v.pos_encoder(z.flatten(2).transpose(1, 2), t=v.pos_emb_balance)
            for blk in v.diffit:
                z = blk(z, time_embedding=temb, text_embedding=None)
            z = v.unpatch_proj(normalize(z).reshape(b * v.seq_ln, v.emb_dim))
            z = z.reshape(b, v.seq_ln, v.in_channels).transpose(1, 2).reshape(b, v.in_channels, hp, wp)
            z = normalize(F.interpolate(z, scale_factor=PATCH, mode="bilinear", align_corners=False))
            return v.unpatch_refine(z)
        h = self.vit(x_in, time_emb=temb)
        if self.head == "fullres_nl":
            for blk in self.nl:
                h = blk(h, temb)
        return h

    def F_theta(self, x_scaled, c_noise):
        temb = self.temb(c_noise)
        h = self.input_proj(x_scaled)
        gain = (self.input_time_proj(temb) * self.time_proj_gate).view(-1, h.shape[1], 1, 1)
        h = normalize(h) * (1.0 + gain)
        f = self.features(h, temb)
        f = torch.cat([f, torch.ones_like(f[:, :1])], dim=1)
        return self.output_proj(f) * self.out_gain

    def forward(self, x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        c_skip = SIGMA_D ** 2 / (s ** 2 + SIGMA_D ** 2)
        c_out = s * SIGMA_D / (s ** 2 + SIGMA_D ** 2).sqrt()
        c_in = 1 / (s ** 2 + SIGMA_D ** 2).sqrt()
        return c_skip * x + c_out * self.F_theta(x * c_in, sigma.log() / 4)


@torch.no_grad()
def token_vs_skip_share(model: LowSigmaDenoiser, x, sigma):
    """Outer-band energy of D_vit split into token-path and skip-path parts (both linear after tokens)."""
    v = model.vit
    s = sigma.view(-1, 1, 1, 1)
    c_in = 1 / (s ** 2 + SIGMA_D ** 2).sqrt()
    temb = model.temb(sigma.log() / 4)
    h = model.input_proj(x * c_in)
    gain = (model.input_time_proj(temb) * model.time_proj_gate).view(-1, h.shape[1], 1, 1)
    x_in = normalize(h) * (1.0 + gain)
    z = torch.cat([x_in, torch.ones_like(x_in[:, :1])], dim=1)
    z = v.patch(z)
    b, _, hp, wp = z.shape
    z = v.pos_encoder(z.flatten(2).transpose(1, 2), t=v.pos_emb_balance)
    for blk in v.diffit:
        z = blk(z, time_embedding=temb, text_embedding=None)
    z = v.unpatch_proj(normalize(z).reshape(b * v.seq_ln, v.emb_dim))
    z = z.reshape(b, v.seq_ln, v.in_channels).transpose(1, 2).reshape(b, v.in_channels, hp, wp)
    z = normalize(F.interpolate(z, scale_factor=PATCH, mode="bilinear", align_corners=False))
    sk = v.unpatch_skip(normalize(x_in))
    tok = v.unpatch_refine(z / math.sqrt(2))
    skp = v.unpatch_refine(sk / math.sqrt(2))
    zeros = torch.zeros_like(tok[:, :1])
    tok_o = model.output_proj(torch.cat([tok, zeros], 1)) * model.out_gain
    skp_o = model.output_proj(torch.cat([skp, zeros], 1)) * model.out_gain
    inner, outer = band_masks(model.res)
    et, es = band_energy(tok_o, outer).mean().item(), band_energy(skp_o, outer).mean().item()
    it, is_ = band_energy(tok_o, inner).mean().item(), band_energy(skp_o, inner).mean().item()
    return et / (et + es), it / (it + is_)


def evaluate(model, x0, sigmas, seed=123):
    model.eval()
    inner, outer = band_masks(x0.shape[-1])
    rows = []
    with torch.no_grad():
        for s in sigmas:
            g = torch.Generator(device=DEV).manual_seed(seed)
            n = torch.randn(x0.shape, generator=g, device=DEV) * s
            sig = torch.full((x0.shape[0],), s, device=DEV)
            D = torch.cat([model(x0[i:i + 32] + n[i:i + 32], sig[i:i + 32]) for i in range(0, len(x0), 32)])
            err = D - x0
            r_in = band_energy(err, inner).sum() / band_energy(n, inner).sum()
            r_out = band_energy(err, outer).sum() / band_energy(n, outer).sum()
            vel = ((n - err) / s).pow(2).mean().sqrt()   # ||(x - D)/sigma||_RMS ; ideal ~ 1
            rows.append((s, r_in.item(), r_out.item(), vel.item()))
    model.train()
    return rows


def train_variant(head, x_train, steps, bs=32, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    res_b = x_train.shape[-1]
    model = LowSigmaDenoiser(head, res=res_b).to(DEV)
    other_params = [p for n, p in model.named_parameters() if n != "out_gain"]
    opt = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": [model.out_gain], "lr": lr * 8.0},
    ], betas=(0.9, 0.99))
    lmin, lmax = math.log(0.01), math.log(0.25)
    t0 = time.time()
    for it in range(steps):
        mult = min(1.0, (it + 1) / 50) * (1.0 if it < 0.7 * steps else max(0.1, (steps - it) / (0.3 * steps)))
        opt.param_groups[0]["lr"] = lr * mult
        opt.param_groups[1]["lr"] = lr * 8.0 * mult
        idx = torch.randint(0, len(x_train), (bs,), device=DEV)
        x0 = x_train[idx]
        if torch.rand(1).item() < 0.5:
            x0 = x0.flip(-1)
        x0 = torch.roll(x0, shifts=(int(torch.randint(0, res_b, (1,))), int(torch.randint(0, res_b, (1,)))), dims=(-2, -1))
        sigma = torch.exp(lmin + (lmax - lmin) * torch.rand(bs, device=DEV))
        n = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
        D = model(x0 + n, sigma)
        s = sigma.view(-1, 1, 1, 1)
        lam = (s ** 2 + SIGMA_D ** 2) / (s * SIGMA_D) ** 2
        loss = (lam * (D - x0) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % max(1, steps // 4) == 0:
            print(f"    [{head:>11}] step {it + 1:5d}/{steps}  loss(F-space)={loss.item():.4f}  gain={model.out_gain.item():.3f} ({time.time() - t0:.0f}s)")
    return model


def part_b(steps: int):
    res_b = 32
    sigmas = [0.02, 0.05, 0.1, 0.2]
    nparams = {}
    for dname, gen in [("dead_leaves (non-Gaussian, sharp edges)", dead_leaves), ("gaussian 1/f^2 (linear-optimal control)", gaussian_1f)]:
        print("\n" + "=" * 78)
        print(f"PART B: dataset = {dname}, res={res_b}, patch={PATCH}, steps={steps}")
        print("=" * 78)
        x_train = gen(4000, res_b, seed=1)
        x_test = gen(512, res_b, seed=999)
        results = {}
        for head in ["tokens_only", "linear", "current", "current_k5", "fullres_nl"]:
            m = train_variant(head, x_train, steps)
            nparams[head] = sum(p.numel() for p in m.parameters())
            results[head] = evaluate(m, x_test, sigmas)
            if head in ("current", "current_k5", "fullres_nl"):
                sig = torch.full((32,), 0.05, device=DEV)
                xs = x_test[:32] + torch.randn_like(x_test[:32]) * 0.05
                to, ti = token_vs_skip_share(m, xs, sig)
                print(f"    [{head:>11}] token-path share of D_vit energy @sigma=0.05: "
                      f"outer band {to:6.2%}, inner band {ti:6.2%}")
            del m
            torch.cuda.empty_cache()
        print("\n  residual-noise ratio r = E|P(D-x0)|^2 / E|P n|^2   (1.000 = F=0 skip-only, 0 = perfect)")
        print(f"  {'sigma':>6} | {'head':>11} | {'r_inner':>8} | {'r_outer':>8} | {'vel_RMS':>7}")
        for i, s in enumerate(sigmas):
            for head, rows in results.items():
                _, ri, ro, v = rows[i]
                print(f"  {s:>6} | {head:>11} | {ri:8.4f} | {ro:8.4f} | {v:7.4f}")
            print("  " + "-" * 52)
    print("  params:", {k: f"{v / 1e6:.2f}M" for k, v in nparams.items()})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "AB"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    if "A" in mode:
        part_a()
    if "B" in mode:
        part_b(steps)
