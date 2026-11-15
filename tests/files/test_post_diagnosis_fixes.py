"""Regression tests verifying resolution of remaining training, spectral, and gate-singularity issues."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from torch.utils.data import Dataset

from Utils.cdf_utils import mixture_cdf
from Utils.configs import loss_configs, mask_configs, model_configs, optim_configs, zeta_configs
from Utils.loss import EDM_LOSS_NOD
from Utils.training import apply_activation_checkpointing, build_optimizer_param_groups, training_HDMOE
from Utils.utils import EMA, Lr_Scheduler, MaskGenerator, WSDScheduler, save_checkpoint
from graphs.logger2 import Logger
from models.model_components import Unet_block, Vit_block, Vit_expert
from models.model_internals import MP_Attention, mp_cat, mp_sum
from tests.files.model_factory import build_tiny_model, random_batch


def _radial_psd(tensor: torch.Tensor) -> torch.Tensor:
    """Azimuthally averaged 2D power spectrum per radius bin k in [0, H//2]."""
    _, _, h, w = tensor.shape
    fft2 = torch.fft.fft2(tensor.float(), norm="ortho")
    psd2 = (fft2.real ** 2 + fft2.imag ** 2).mean(dim=(0, 1))
    fy = torch.fft.fftfreq(h, d=1.0 / h)
    fx = torch.fft.fftfreq(w, d=1.0 / w)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(kx ** 2 + ky ** 2).round().long()
    max_r = h // 2
    out = torch.zeros(max_r, dtype=torch.float32)
    for k in range(1, max_r):
        mask = r == k
        if mask.any():
            out[k] = psd2[mask].mean()
    return out


class TestPostDiagnosisFixes(unittest.TestCase):

    def test_nod_loss_has_no_laplacian_or_kendall_branch_inflation(self):
        torch.manual_seed(0)
        c = torch.tensor([0.25, 0.75])
        crit = EDM_LOSS_NOD(
            expert_centers_vit=c,
            expert_centers_Unet=c,
            num_experts=2,
            sigma_data=0.8,
            Unet_bal=0.0,
            vit_bal=0.0,
            z_bal=0.0,
            prior_bal=0.0,
            alpha_blend=0.6,
            alpha_unet=0.2,
            alpha_vit=0.2,
        )
        B = 8
        x = torch.randn(B, 4, 32, 32)
        sigma = torch.ones(B)
        sig4 = sigma.view(-1, 1, 1, 1)
        weight = (sig4 ** 2 + 0.8 ** 2) / ((sig4 * 0.8) ** 2)
        vit_pred = x + 0.3 * torch.randn_like(x)
        unet_pred = x + 0.3 * torch.randn_like(x)
        blend_pred = 0.5 * vit_pred + 0.5 * unet_pred
        w_vit = torch.full((B, 1, 1, 1), 0.5)
        w_unet = torch.full((B, 1, 1, 1), 0.5)

        out_u0 = {
            "denoised": blend_pred,
            "vit_denoised": vit_pred,
            "unet_denoised": unet_pred,
            "w_vit": w_vit,
            "w_unet": w_unet,
            "log_var": torch.zeros(B, 1, 1, 1),
            "Unet_router_loss": torch.full((B, 2), 0.5),
            "vit_router_loss": torch.full((B, 2), 0.5),
            "Unet_raw": torch.zeros(B, 2),
            "vit_raw": torch.zeros(B, 2),
        }
        out_u_neg = {**out_u0, "log_var": torch.full((B, 1, 1, 1), -2.0)}

        loss0 = crit(x, x, sigma, out_u0)
        loss_neg = crit(x, x, sigma, out_u_neg)

        expected_vit = torch.mean(weight * w_vit * ((vit_pred - x) ** 2))
        expected_unet = torch.mean(weight * w_unet * ((unet_pred - x) ** 2))
        torch.testing.assert_close(loss0["loss_texture"], expected_vit)
        torch.testing.assert_close(loss0["loss_structure"], expected_unet)
        # Branch losses must not inflate when log_var decreases below zero.
        torch.testing.assert_close(loss_neg["loss_texture"], loss0["loss_texture"])
        torch.testing.assert_close(loss_neg["loss_structure"], loss0["loss_structure"])

    def test_no_nested_zero_gate_saddle_and_step1_inner_wakeup(self):
        model, cfg = build_tiny_model()
        # Internal block gates must start open while outer branch gates start at 0
        for mod in model.net.modules():
            if isinstance(mod, Unet_block):
                self.assertGreater(mod.res_gate.abs().item(), 0.05)
            elif isinstance(mod, Vit_block):
                self.assertGreater(mod.tmsa_gate.abs().item(), 0.05)
                self.assertGreater(mod.mlp_gate.abs().item(), 0.05)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Step 0: outer projections themselves already receive nonzero gradients if inner gates are open,
        # wait: unet_proj_gate=0 gives output_proj_unet grad on step 0, and opens unet_proj_gate on step 0.
        # By step 1 (after ONE optimizer update), 100% of Unet_experts and VIT_experts parameters must have nonzero grad.
        for step in range(2):
            x, sigma, text_emb, Umask, Vmask = random_batch(
                cfg, batch_size=4, sigma_value=[0.05, 1.0, 4.0, 20.0], seed=step
            )
            out = model(x=x, sigma=sigma, text_emb=text_emb, Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
            loss = (out["denoised"] - torch.randn_like(out["denoised"])).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            if step == 1:
                zero_unet = [n for n, p in model.net.Unet_experts.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)]
                zero_vit = [n for n, p in model.net.VIT_experts.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)]
                self.assertEqual(zero_unet, [], f"UNet tensors still at zero grad after 1 step: {zero_unet[:5]}")
                self.assertEqual(zero_vit, [], f"ViT tensors still at zero grad after 1 step: {zero_vit[:5]}")
            opt.step()

    def test_vit_expert_unpatch_eliminates_nyquist_aliasing_spike(self):
        torch.manual_seed(0)
        h = w = 32
        fx = torch.fft.fftfreq(h)
        fy = torch.fft.rfftfreq(w)
        ky, kx = torch.meshgrid(fx, fy, indexing="ij")
        amp = (kx ** 2 + ky ** 2 + (1.0 / h) ** 2) ** (-0.5)
        amp[0, 0] = 0.0
        spec = torch.complex(torch.randn(16, 4, h, fy.shape[0]), torch.randn(16, 4, h, fy.shape[0])) * amp
        x_in = torch.fft.irfft2(spec, s=(h, w), norm="ortho")
        x_in = x_in / (x_in.std(dim=(-2, -1), keepdim=True) + 1e-8)

        vit = Vit_expert(
            num_heads=4,
            num_groups=4,
            in_channels=4,
            seq_ln=(h // 2) * (w // 2),
            emb_dim=64,
            num_blocks=2,
            patch_size=2,
            time_dim=32,
            text_dim=0,
        )
        vit.eval()
        with torch.no_grad():
            y = vit(x_in, time_emb=torch.randn(16, 32))

        self.assertEqual(y.shape, x_in.shape)
        psd_in = _radial_psd(x_in)
        psd_out = _radial_psd(y)
        # Compare normalized spectral shape against the 1/f^2 input;
        # PixelShuffle(2) + 1x1 previously caused a 6.87x excess spike at Nyquist r=15.
        norm_in = psd_in / psd_in[1:16].sum()
        norm_out = psd_out / psd_out[1:16].sum()
        nyquist_ratio = (norm_out[12:16] / (norm_in[12:16] + 1e-12)).max().item()
        self.assertLess(nyquist_ratio, 2.5, f"Nyquist spectral aliasing spike remains too large: {nyquist_ratio:.2f}x")

    def test_lr_scheduler_has_nonzero_monotonic_warmup(self):
        p = torch.nn.Parameter(torch.zeros(2, 2))
        opt = torch.optim.Adam([p], lr=1e-3)
        sched = Lr_Scheduler(
            optimizer=opt,
            total_steps=100,
            batch_size=128,
            ref_batches=20000,
            rampup_Mimg=0.25,
            dataset_size=90000,
            cur_nimg=0,
        )
        lr0 = opt.param_groups[0]["lr"]
        lrs = [lr0] + [sched.step()[0] for _ in range(10)]
        self.assertGreater(lr0, 0.0, "Step-0 LR should be nonzero")
        for i in range(len(lrs) - 1):
            self.assertGreater(lrs[i + 1], lrs[i], f"LR failed monotonic warmup at step {i}->{i+1}: {lrs[i]} -> {lrs[i+1]}")

    def test_plotter2_branch_labels_match_loss_keys(self):
        plotter_src = (Path(__file__).resolve().parents[2] / "graphs" / "plotter2.py").read_text(encoding="utf-8")
        tex_line = [ln for ln in plotter_src.splitlines() if "'loss_texture'" in ln and "ax.plot" in ln][0]
        str_line = [ln for ln in plotter_src.splitlines() if "'loss_structure'" in ln and "ax.plot" in ln][0]
        self.assertIn("ViT Branch", tex_line)
        self.assertIn("UNet Branch", str_line)

    def test_outer_gates_no_second_order_singularity_and_lr_alignment(self):
        model, cfg = build_tiny_model()
        net = model.net
        # Outer time & blend gates must start open (0.1) to eliminate nested 0*0 saddles
        self.assertAlmostEqual(net.time_proj_gate.item(), 0.1, places=6)
        self.assertAlmostEqual(net.blend_gate.item(), 0.1, places=6)
        # Denoiser output gates still start at 0 so D(x) == c_skip * x at init
        self.assertEqual(net.vit_proj_gate.item(), 0.0)
        self.assertEqual(net.unet_proj_gate.item(), 0.0)
        # Gate LR should match/lead branch MP_Conv angular LR so gates escape Adam eps quickly
        self.assertGreaterEqual(optim_configs["lr_gates"], 1e-3)
        self.assertLessEqual(optim_configs["lr_vit"], optim_configs["lr_gates"])
        self.assertLessEqual(optim_configs["lr_unet"], optim_configs["lr_gates"])

    def test_unet_late_bandwidth_overlap_and_low_sigma_skip(self):
        unet_mg = MaskGenerator(
            expert_attributes=mask_configs["unet_attr"],
            p_mean=mask_configs["p_mean"],
            p_std=mask_configs["p_std"],
            bandwidth=mask_configs["max_BW_u"],
            min_bandwidth=mask_configs["min_BW_u"],
            min_active=mask_configs["min_active"],
            total_steps=mask_configs["total_steps"],
            strat_band=mask_configs["strat_band"],
            noise_range=mask_configs["unet_noise_range"],
            sigma_min=model_configs["sigma_min"],
            sigma_max=model_configs["sigma_max"],
            alpha_ext=model_configs["extreme_prob"],
        )
        sigmas = torch.logspace(
            math.log10(mask_configs["unet_noise_range"][0]),
            math.log10(mask_configs["unet_noise_range"][1]),
            200,
        )
        late_mask = unet_mg(sigmas, step=mask_configs["total_steps"])
        overlap_frac = (late_mask.sum(dim=-1) >= 2).float().mean().item()
        self.assertGreater(overlap_frac, 0.05, f"Late-stage UNet experts have zero overlap: {overlap_frac:.4f}")

        # Verify HDMOEM_v3 skips executing Unet_experts on deep low-noise samples where w_unet < 0.05
        model, cfg = build_tiny_model()
        call_count = {"n": 0}
        orig_fwd = model.net.Unet_experts[0].forward

        def counted_fwd(*args, **kwargs):
            call_count["n"] += 1
            return orig_fwd(*args, **kwargs)

        model.net.Unet_experts[0].forward = counted_fwd
        x_lo, s_lo, t_lo, u_lo, v_lo = random_batch(cfg, batch_size=3, sigma_value=[0.002, 0.005, 0.01], seed=10)
        with torch.no_grad():
            out_lo = model(x=x_lo, sigma=s_lo, text_emb=t_lo, Unet_router_mask=u_lo, Vit_router_mask=v_lo, zeta=0.0)
        self.assertLess(out_lo["w_unet"].max().item(), 0.05)
        self.assertEqual(call_count["n"], 0, "Unet_experts executed despite w_unet < 0.05 on low-sigma batch")

        x_hi, s_hi, t_hi, u_hi, v_hi = random_batch(cfg, batch_size=2, sigma_value=[1.0, 10.0], seed=11)
        with torch.no_grad():
            model(x=x_hi, sigma=s_hi, text_emb=t_hi, Unet_router_mask=u_hi, Vit_router_mask=v_hi, zeta=0.0)
        self.assertGreater(call_count["n"], 0, "Unet_experts did not execute on active medium/high-sigma batch")

    def test_logger_in_domain_router_health_mixture_cdf_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = Logger(
                log_dir=tmpdir, run_name="test_run", log_interval=1,
                num_unet_experts=2, num_vit_experts=2,
                transition_sigma=model_configs["transition_sigma"],
                sigma_min=model_configs["sigma_min"],
                sigma_max=model_configs["sigma_max"],
                p_mean=mask_configs["p_mean"],
                p_std=mask_configs["p_std"],
                alpha_ext=model_configs["extreme_prob"],
            )
            test_sigmas = torch.tensor([0.05, 1.0, 2.5, 20.0])
            expected_pct = mixture_cdf(
                torch.log(test_sigmas),
                mask_configs["p_mean"],
                mask_configs["p_std"],
                math.log(model_configs["sigma_min"]),
                math.log(model_configs["sigma_max"]),
                model_configs["extreme_prob"],
            )
            torch.testing.assert_close(logger1._sigma_percentile(test_sigmas), expected_pct)

            # Build mixed batch with 80% low-sigma and 20% high-sigma (realistic mixture distribution)
            sigma = torch.cat([torch.full((16,), 0.2), torch.tensor([3.0, 4.0, 20.0, 50.0])])
            unet_probs = torch.zeros(20, 2)
            unet_probs[:16, 0] = 1.0
            unet_probs[16:18, :] = torch.tensor([0.9, 0.1])
            unet_probs[18:, :] = torch.tensor([0.1, 0.9])
            vit_probs = torch.full((20, 2), 0.5)
            logger1.log_router_statistics(step=1, unet_probs=unet_probs, vit_probs=vit_probs, sigma=sigma)

            import json
            router_lines = (Path(tmpdir) / "test_run_router.jsonl").read_text().strip().splitlines()
            rec = json.loads(router_lines[-1])
            self.assertTrue(rec["unet_healthy"], f"False UNet imbalance reported: {rec}")
            self.assertGreater(rec["unet_entropy_norm"], 0.85)
            self.assertLess(rec["unet_gini"], 0.25)

            # Verify overwrite=True clears prior log lines on a fresh training run
            (Path(tmpdir) / "test_run_training.jsonl").write_text('{"step": 999}\n')
            logger2 = Logger(
                log_dir=tmpdir, run_name="test_run", log_interval=1,
                overwrite=True,
            )
            self.assertFalse((Path(tmpdir) / "test_run_training.jsonl").exists())
            self.assertFalse((Path(tmpdir) / "test_run_router.jsonl").exists())

    def test_vit_cross_attn_gated_and_context_mask_prevent_collapse(self):
        torch.manual_seed(0)
        vit = Vit_expert(
            num_heads=4, num_groups=4, in_channels=4, seq_ln=64,
            emb_dim=64, num_blocks=10, patch_size=2, time_dim=32, text_dim=64,
        )
        vit.eval()
        for blk in vit.diffit:
            self.assertAlmostEqual(blk.ca_gate.item(), 0.1, places=6)

        x = torch.randn(2, 4, 16, 16)
        t_emb = torch.randn(2, 32)
        const_pad_txt = torch.ones(2, 77, 64)
        with torch.no_grad():
            y_gated = vit(x, time_emb=t_emb, text_emb=const_pad_txt)
            # Compare against un-gated ca_gate=1.0 control
            for blk in vit.diffit:
                blk.ca_gate.fill_(1.0)
            y_ungated = vit(x, time_emb=t_emb, text_emb=const_pad_txt)

        def mean_spatial_cos(feat: torch.Tensor) -> float:
            tok = feat.flatten(2).transpose(1, 2)  # (B, S, C)
            tok = tok / (tok.norm(dim=-1, keepdim=True) + 1e-8)
            sim = torch.bmm(tok, tok.transpose(1, 2))
            s = sim.shape[1]
            off = ~torch.eye(s, dtype=torch.bool).unsqueeze(0)
            return sim.masked_select(off).mean().item()

        cos_gated = mean_spatial_cos(y_gated)
        cos_ungated = mean_spatial_cos(y_ungated)
        self.assertLess(cos_gated, cos_ungated, f"ca_gate=0.1 did not reduce constant-token collapse: {cos_gated:.4f} vs {cos_ungated:.4f}")

        # Verify MP_Attention respects context_mask by ignoring masked key/value positions
        attn = MP_Attention(num_heads=4, emb_dim=32, seq_ln=10, context_dim=16, is_cross_attn=True)
        q = torch.randn(2, 8, 32)
        ctx1 = torch.randn(2, 10, 16)
        ctx2 = ctx1.clone()
        ctx2[:, 3:, :] = torch.randn(2, 7, 16) * 10.0
        ctx_mask = torch.zeros(2, 10, dtype=torch.bool)
        ctx_mask[:, :3] = True
        with torch.no_grad():
            o1 = attn(q, gain_s=1.0, gain_t=1.0, context=ctx1, context_mask=ctx_mask)
            o2 = attn(q, gain_s=1.0, gain_t=1.0, context=ctx2, context_mask=ctx_mask)
        torch.testing.assert_close(o1, o2, atol=1e-5, rtol=1e-5)

    def test_checkpoint_save_and_resume_preserves_ema_step_and_scheduler(self):
        model, _ = build_tiny_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = Lr_Scheduler(opt, total_steps=100, batch_size=128, ref_batches=20000, rampup_Mimg=0.25, dataset_size=90000, cur_nimg=0)
        for _ in range(5):
            sched.step()
        ema = EMA(model, beta=0.9999, warmup_power=2 / 3, min_warmup_step=40000)
        ema.step = 26000
        # Diverge online model from shadow model so EMA reset would visibly overwrite shadow weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        shadow_before = next(ema.ema_model.parameters()).detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                model, opt, step=26000, mse_score=0.2,
                configs={"save_dir": tmpdir}, filename="ckpt_test.pt",
                ema_instance=ema, scheduler_instance=sched,
            )
            ckpt = torch.load(Path(tmpdir) / "ckpt_test.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["ema_step"], 26000)
            self.assertIn("scheduler_state_dict", ckpt)

            ema2 = EMA(model, beta=0.9999, warmup_power=2 / 3, min_warmup_step=40000)
            ema2.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
            ema2.step = ckpt.get("ema_step", ckpt.get("step", 0))
            sched2 = Lr_Scheduler(opt, total_steps=100, batch_size=128, ref_batches=20000, rampup_Mimg=0.25, dataset_size=90000, cur_nimg=0)
            sched2.load_state_dict(ckpt["scheduler_state_dict"])
            self.assertEqual(sched2.cur_nimg, sched.cur_nimg)

            ema2.update()
            shadow_after = next(ema2.ema_model.parameters()).detach()
            max_drift = (shadow_after - shadow_before).abs().max().item()
            self.assertLess(max_drift, 0.01, f"EMA weights corrupted on first resumed step: max_drift={max_drift:.4f}")

    def test_generate_test_and_training_wiring(self):
        root = Path(__file__).resolve().parents[2]
        gen_src = (root / "Utils" / "generate_test.py").read_text(encoding="utf-8")
        train_src = (root / "Utils" / "training.py").read_text(encoding="utf-8")
        self.assertIn("p_mean=mask_configs['p_mean']", gen_src)
        self.assertIn("p_std=mask_configs['p_std']", gen_src)
        self.assertIn("uncond_text_emb=uncond_text_emb", gen_src)
        self.assertIn("ema.step = checkpoint.get('ema_step'", train_src)
        self.assertIn('final_ckpt_name = model_config.get("final_ckpt_name", "final_model1.pt")', train_src)

    def test_mp_sum_and_mp_cat_promote_mixed_bf16_fp32_dtypes(self):
        a = torch.randn(2, 8, 4, 4, dtype=torch.bfloat16)
        b = torch.randn(2, 8, 4, 4, dtype=torch.float32)
        s = mp_sum(a, b, t=0.5)
        c = mp_cat(a, b, dim=1, t=0.5)
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(c.dtype, torch.float32)
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue(torch.isfinite(c).all())

    def test_compute_blend_weights_bounds_shift_and_records_residual_in_train_only(self):
        model, _ = build_tiny_model(max_blend_shift=0.40)
        sigma = torch.tensor([2.5, 2.5])
        t_emb = torch.randn(2, 64)
        model.train()
        with torch.enable_grad():
            model.net.compute_blend_weights(sigma, t_emb)
            res = model.net._last_bounded_residual
            self.assertIsNotNone(res)
            self.assertLessEqual(res.abs().max().item(), 0.40 + 1e-6)
        model.eval()
        with torch.no_grad():
            model.net.compute_blend_weights(sigma, t_emb)
            self.assertIsNone(model.net._last_bounded_residual)

    def test_wsd_scheduler_phases_and_state_roundtrip(self):
        p = torch.nn.Parameter(torch.zeros(2, 2))
        opt = torch.optim.Adam([p], lr=1e-3)
        sched = WSDScheduler(
            optimizer=opt,
            total_steps=20,
            batch_size=8,
            warmup_steps=4,
            stable_until_frac=0.5,
            min_lr_mult=0.10,
            dataset_size=64,
        )
        self.assertAlmostEqual(sched.get_lr_multiplier(), 0.25, places=5)
        for _ in range(4):
            sched.step()
        self.assertAlmostEqual(sched.get_lr_multiplier(), 1.0, places=5)
        for _ in range(6):
            sched.step()
        self.assertAlmostEqual(sched.get_lr_multiplier(), 1.0, places=5)
        for _ in range(10):
            sched.step()
        self.assertAlmostEqual(sched.get_lr_multiplier(), 0.10, places=5)

        sd = sched.state_dict()
        sched2 = WSDScheduler(optimizer=opt, total_steps=20, batch_size=8, warmup_steps=4, stable_until_frac=0.5, min_lr_mult=0.10)
        sched2.load_state_dict(sd)
        self.assertEqual(sched2.cur_nimg, sched.cur_nimg)
        self.assertAlmostEqual(sched2.get_lr_multiplier(), sched.get_lr_multiplier(), places=6)

    def test_training_hdmoe_supports_dataset_wsd_grad_accum_val_probe_and_callbacks(self):
        class _DummyPairDataset(Dataset):
            def __init__(self, n=12):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                return {"pixel_values": torch.zeros(3, 16, 16), "prompts": f"nature scene {idx}"}

        class _StubVAE:
            def init(self, device):
                self.device = device

            def encode(self, imgs):
                b = imgs.shape[0]
                return torch.randn(b, 4, 16, 16, device=imgs.device) * 0.5

        class _StubCLIP:
            def init(self):
                pass

            def encode_text(self, prompts):
                return torch.randn(len(prompts), 4, 32)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            log_dir = Path(tmpdir) / "logs"
            callbacks = []

            mc = copy.deepcopy(model_configs)
            mc.update({
                "save_dir": str(ckpt_dir),
                "save_dir_stats": str(log_dir),
                "run_name": "unit_wsd_test",
                "final_ckpt_name": "final_model.pt",
                "device": "cpu",
                "num_workers": 0,
                "dataset_size": 12,
                "img_channels": 4,
                "internal_channels": 32,
                "time_emb_dim": 64,
                "img_resolution": 16,
                "text_emb_dim": 32,
                "num_experts": 2,
                "top_k": 1,
                "fourier_bandwidth": 1.0,
                "VIT_num_blocks": 2,
                "VIT_patch_sizes": [2, 4],
                "VIT_num_groups": 4,
                "VIT_num_heads": 4,
                "VIT_emb_size": 32,
                "VIT_pos_emb_balances": [0.5, 0.5],
                "VIT_emb_balances": [0.5, 0.5],
                "Unet_num_blocks": 1,
                "Unet_channel_mult": [1, 2],
                "Unet_kernel_sizes": [(3, 3), (3, 3)],
                "Unet_model_channels": 16,
                "pool_size": 2,
                "batch_size": 4,
                "micro_batch_size": 2,
                "grad_accum_steps": 2,
                "use_grad_checkpointing": True,
                "total_steps": 3,
                "save_interval": 1,
                "log_interval": 1,
                "max_blend_shift": 0.40,
                "blend_anchor_weight": 0.02,
            })
            oc = copy.deepcopy(optim_configs)
            oc.update({"lr_schedule": "wsd", "warmup_steps": 2, "stable_until_frac": 0.6, "min_lr_mult": 0.1, "grad_clip": 5.0})
            mkc = copy.deepcopy(mask_configs)
            mkc["total_steps"] = 3
            zc = copy.deepcopy(zeta_configs)
            zc["total_schedule_steps"] = 3

            model, ema, out_cfg, logger = training_HDMOE(
                mc, oc, copy.deepcopy(loss_configs), mkc, zc,
                resume_path=None,
                dataset=_DummyPairDataset(12),
                val_dataset=_DummyPairDataset(4),
                vae=_StubVAE(),
                clip=_StubCLIP(),
                keep_last_ckpts=1,
                val_probe_samples=4,
                post_save_callback=lambda sd, ld, tag: callbacks.append(tag),
            )
            self.assertIsNotNone(model)
            self.assertEqual(ema.step, 3)
            self.assertEqual(out_cfg["run_name"], "unit_wsd_test")
            self.assertTrue((ckpt_dir / "final_model.pt").exists())
            kept_ckpts = list(ckpt_dir.glob("ckpt_*.pt"))
            self.assertEqual(len(kept_ckpts), 1)
            self.assertEqual(kept_ckpts[0].name, "ckpt_2.pt")
            self.assertEqual(callbacks, ["step_1", "step_2", "final"])
            ckpt_lines = (log_dir / "unit_wsd_test_checkpoints.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(ckpt_lines), 2)
            self.assertIn("val_probe_mse", json.loads(ckpt_lines[-1]))

    def test_colab_notebook_uses_repo_training_and_no_monkey_patches(self):
        nb_path = Path(__file__).resolve().parents[2] / "colab_text_guided_training.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_src = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("training_HDMOE", code_src)
        self.assertIn("from graphs.plotter2 import Plotter", code_src)
        self.assertIn("WSDScheduler", code_src)
        self.assertIn("nature_50k", code_src)
        for forbidden in (
            "apply_in_notebook_patches",
            "ColabCosineLrScheduler",
            "PatchedLogger",
            "run_colab_training",
            "_standalone_fixed_overview",
            "_ensure_eval_blend_patch",
        ):
            self.assertNotIn(forbidden, code_src)


if __name__ == "__main__":
    unittest.main()
