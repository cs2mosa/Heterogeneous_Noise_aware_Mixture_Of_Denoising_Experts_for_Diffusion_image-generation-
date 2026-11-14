"""
Regression tests for the mathematical, information-geometric, and wiring bugs
identified in the post-fix audit (Fixes 1–6).
"""
import copy
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.model_components import ParametricTrajectoryRouter, Unet_block, Vit_block
from models.model_config_v3 import router_to_experts
from models.model_internals import MP_Conv, PooledCrossAttn
from Utils.configs import loss_configs, mask_configs, model_configs, optim_configs, zeta_configs
from Utils.loss import EDM_LOSS_NOD, PathPriorLoss, laplacian_decomposition
from Utils.utils import MaskGenerator
from tests.files.model_factory import build_tiny_model, build_tiny_net, random_batch


def _rms(t: torch.Tensor) -> torch.Tensor:
    return t.pow(2).mean(dim=(1, 2, 3)).sqrt()


class _ConstProj(torch.nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = value

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return self.value


class TestFix1VariancePreservation(unittest.TestCase):

    def test_spherical_blend_preserves_interpolated_rms_for_unequal_branches(self):
        """
        With unequal branch magnitudes (sigma_vit != sigma_unet) and orthogonal
        branch outputs at transition (w_vit = w_unet = 0.5), the blended output
        RMS must equal the weighted target RMS = w_vit * sigma_v + w_unet * sigma_u
        (not inflate/distort via an unweighted unit-variance denominator).
        """
        net, cfg = build_tiny_net(seed=0)
        net.eval()
        net._cross_attn_enabled = False
        with torch.no_grad():
            net.vit_proj_gate.fill_(1.0)
            net.unet_proj_gate.fill_(1.0)
            net.out_gain_vit.fill_(1.0)
            net.out_gain_unet.fill_(1.0)
            net.blend_gate.fill_(0.0)

        torch.manual_seed(7)
        fv = torch.randn(2, cfg["IN_in_channels"], cfg["IN_img_resolution"], cfg["IN_img_resolution"]) * 2.0
        fu = torch.randn_like(fv) * 0.2
        net.output_proj_vit = _ConstProj(fv)
        net.output_proj_unet = _ConstProj(fu)

        x, _, text_emb, Umask, Vmask = random_batch(cfg, batch_size=2, sigma_value=cfg["transition_sigma"])
        time_vec = torch.full((2,), math.log(cfg["transition_sigma"]) / 4.0)
        with torch.no_grad():
            out = net(
                x=x,
                time_vec=time_vec,
                text_emb=text_emb,
                Unet_router_mask=Umask,
                Vit_router_mask=Vmask,
                zeta=0.0,
            )

        target_rms = out["w_vit"].squeeze() * _rms(out["vit_out"]) + out["w_unet"].squeeze() * _rms(out["unet_out"])
        actual_rms = _rms(out["out"])
        torch.testing.assert_close(actual_rms, target_rms, rtol=0.02, atol=1e-4)

    def test_spherical_blend_does_not_explode_for_antiparallel_unequal_branches(self):
        """
        At rho = -1 and w_vit = w_unet = 0.5, the old formula's denominator
        sqrt(w_v^2 + w_u^2 + 2*rho*w_v*w_u) drops to sqrt(1e-6)=1e-3, inflating
        unequal-norm anti-parallel branches by ~500x.
        """
        net, cfg = build_tiny_net(seed=0)
        net.eval()
        net._cross_attn_enabled = False
        with torch.no_grad():
            net.vit_proj_gate.fill_(1.0)
            net.unet_proj_gate.fill_(1.0)
            net.out_gain_vit.fill_(2.0)
            net.out_gain_unet.fill_(1.0)
            net.blend_gate.fill_(0.0)

        torch.manual_seed(11)
        base = torch.randn(2, cfg["IN_in_channels"], cfg["IN_img_resolution"], cfg["IN_img_resolution"])
        net.output_proj_vit = _ConstProj(base)
        net.output_proj_unet = _ConstProj(-base)

        x, _, text_emb, Umask, Vmask = random_batch(cfg, batch_size=2, sigma_value=cfg["transition_sigma"])
        time_vec = torch.full((2,), math.log(cfg["transition_sigma"]) / 4.0)
        with torch.no_grad():
            out = net(
                x=x,
                time_vec=time_vec,
                text_emb=text_emb,
                Unet_router_mask=Umask,
                Vit_router_mask=Vmask,
                zeta=0.0,
            )

        max_branch_rms = torch.maximum(_rms(out["vit_out"]), _rms(out["unet_out"]))
        self.assertTrue(
            (_rms(out["out"]) <= 2.5 * max_branch_rms).all(),
            f"Anti-parallel spherical blend exploded: actual_rms={_rms(out['out']).tolist()} "
            f"vs max_branch_rms={max_branch_rms.tolist()}",
        )

    def test_router_to_experts_preserves_variance_when_experts_are_correlated(self):
        """
        If intra-branch experts produce correlated/identical features under
        weights [0.5, 0.5], dividing by L2 norm sqrt(0.5^2 + 0.5^2) inflates
        output std by sqrt(2). Combined variance should match expert std.
        """
        class _IdentityExpert(torch.nn.Module):
            def forward(self, x, time_emb, text_emb):
                return x

        torch.manual_seed(0)
        x = torch.randn(4, 16, 16, 16)
        experts = torch.nn.ModuleList([_IdentityExpert(), _IdentityExpert()])
        weights = torch.full((4, 2), 0.5)
        routed = router_to_experts(
            x=x,
            experts=experts,
            out_router=weights,
            time_emb=torch.zeros(4, 32),
            text_emb=None,
        )
        self.assertAlmostEqual(
            routed.std().item(),
            x.std().item(),
            delta=0.05,
            msg=f"Correlated experts inflated output std from {x.std().item():.4f} to {routed.std().item():.4f}",
        )


class TestFix2LossAndDecomposition(unittest.TestCase):

    def test_edm_loss_nod_log_var_is_unbiased_at_unit_errors(self):
        """
        For calibrated unit normalized errors (weighted MSE == 1.0 on blend and
        branch terms), d(loss)/d(log_var) at log_var == 0 must be 0 so log_var
        converges to log(1.0)=0 instead of inflating by sum(alpha)/alpha_blend.
        """
        torch.manual_seed(0)
        sigma_val = 1.0
        sigma_data = 1.0
        sigma = torch.full((2,), sigma_val)
        w_edm = (sigma_val ** 2 + sigma_data ** 2) / (sigma_val * sigma_data) ** 2

        x = torch.zeros(2, 4, 32, 32)
        noisy_x = torch.randn_like(x)
        wx_u = torch.full((2, 1, 1, 1), 0.5)
        wx_v = torch.full((2, 1, 1, 1), 0.5)

        # Build pure low-freq UNet residual (constant) and pure high-freq ViT residual
        # (checkerboard) and normalize each so weighted MSE == 1.0 exactly.
        denoised = torch.ones_like(x) / math.sqrt(w_edm)
        unet_base = torch.ones_like(x)
        unet_low0, _ = laplacian_decomposition(unet_base, kernel_size=9, sigma=1.0)
        scale_u = math.sqrt((w_edm * 0.5 * (unet_low0 ** 2).mean()).item())
        unet_denoised = unet_base / scale_u

        grid_y, grid_x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        checker = (((grid_y + grid_x) % 2) * 2 - 1).float().view(1, 1, 32, 32).expand_as(x)
        _, vit_high0 = laplacian_decomposition(checker, kernel_size=9, sigma=1.0)
        scale_v = math.sqrt((w_edm * 0.5 * (vit_high0 ** 2).mean()).item())
        vit_denoised = checker / scale_v

        log_var = torch.zeros(2, 1, 1, 1, requires_grad=True)
        out_model = {
            "denoised": denoised,
            "unet_denoised": unet_denoised,
            "vit_denoised": vit_denoised,
            "w_unet": wx_u,
            "w_vit": wx_v,
            "log_var": log_var,
            "Unet_router_loss": torch.full((2, 2), 0.5),
            "vit_router_loss": torch.full((2, 2), 0.5),
            "Unet_raw": torch.zeros(2, 2),
            "vit_raw": torch.zeros(2, 2),
        }

        crit = EDM_LOSS_NOD(
            expert_centers_vit=torch.tensor([0.25, 0.75]),
            expert_centers_Unet=torch.tensor([0.25, 0.75]),
            num_experts=2,
            sigma_data=sigma_data,
            Unet_bal=0.0,
            vit_bal=0.0,
            z_bal=0.0,
            prior_bal=0.0,
            alpha_blend=0.6,
            alpha_unet=0.2,
            alpha_vit=0.2,
            lap_kernel=9,
            lap_sigma=1.0,
        )
        loss_dict = crit(x, noisy_x, sigma, out_model)
        loss_dict["loss"].backward()

        self.assertLess(
            log_var.grad.abs().max().item(),
            1e-4,
            f"log_var gradient is biased away from 0 at unit errors: {log_var.grad.flatten().tolist()}",
        )

    def test_laplacian_decomposition_is_orthogonal_and_energy_preserving(self):
        """
        Low-pass and high-pass projections in EDM_LOSS_NOD must be orthogonal
        (<low, high> == 0) and conserve energy (||low||^2 + ||high||^2 == ||r||^2).
        """
        torch.manual_seed(0)
        res = torch.randn(4, 4, 64, 64)
        low, high = laplacian_decomposition(res, kernel_size=9, sigma=1.0)
        inner = (low * high).mean().item()
        energy_diff = abs((low.pow(2).mean() + high.pow(2).mean() - res.pow(2).mean()).item())
        self.assertLess(abs(inner), 1e-4, f"low and high bands are non-orthogonal (<low, high>={inner:.6f})")
        self.assertLess(energy_diff, 1e-4, f"decomposition does not conserve energy (diff={energy_diff:.6f})")

    def test_edm_loss_nod_accepts_and_forwards_sigma_bounds_and_alpha_ext(self):
        crit = EDM_LOSS_NOD(
            expert_centers_vit=torch.tensor([0.25, 0.75]),
            expert_centers_Unet=torch.tensor([0.25, 0.75]),
            num_experts=2,
            sigma_min=0.01,
            sigma_max=50.0,
            alpha_ext=0.05,
        )
        self.assertAlmostEqual(crit.prior_path_loss_vit.sigma_min, 0.01)
        self.assertAlmostEqual(crit.prior_path_loss_vit.sigma_max, 50.0)
        self.assertAlmostEqual(crit.prior_path_loss_vit.alpha_ext, 0.05)


class TestFix3NoZeroInitMPConv(unittest.TestCase):

    def test_no_zero_initialized_mp_conv_weights_in_model(self):
        """
        Zero-initialized MP_Conv tensors trigger a 1/eps (=1e4) gradient spike and
        unit-sphere snap at step 1. Every MP_Conv in HDMOEM_v3 must initialize its
        weights on the sphere (use scalar gates for safe identity start).
        """
        model, _ = build_tiny_model()
        zero_mp_convs = [
            name for name, mod in model.named_modules()
            if isinstance(mod, MP_Conv) and mod.weights.detach().abs().max().item() == 0.0
        ]
        self.assertEqual(
            zero_mp_convs,
            [],
            f"Found {len(zero_mp_convs)} zero-initialized MP_Conv layers: {zero_mp_convs[:5]}",
        )

    def test_unet_block_has_no_step1_gradient_spike(self):
        torch.manual_seed(0)
        ub = Unet_block(32, 32, (3, 3), emb_size=64)
        self.assertGreater(ub.conv_res2.weights.detach().norm().item(), 0.1)

        xu = torch.randn(2, 32, 16, 16)
        tu = torch.randn(2, 64)
        target = torch.randn_like(xu)

        (ub(xu, tu) - target).pow(2).mean().backward()
        grad0 = ub.conv_res2.weights.grad.norm().item()
        self.assertLess(grad0, 20.0, f"Step-0 gradient spike in Unet_block.conv_res2: {grad0:.2f}")

        opt = torch.optim.SGD(ub.parameters(), lr=1e-3)
        opt.step()
        opt.zero_grad()
        (ub(xu, tu) - target).pow(2).mean().backward()
        grad1 = ub.conv_res2.weights.grad.norm().item()
        if grad1 > 1e-8:
            self.assertLess(grad0 / grad1, 10.0, f"Gradient collapsed {grad0 / grad1:.1f}x across step 0->1")

    def test_vit_block_has_no_zero_init_mp_conv_or_grad_spike(self):
        torch.manual_seed(0)
        vb = Vit_block(num_heads=4, num_groups=4, num_channels=32, seq_ln=16, emb_dim=32, text_dim=32, time_dim=32)
        self.assertGreater(vb.linear3.weights.detach().norm().item(), 0.1)
        self.assertGreater(vb.TMSA.out_proj.weights.detach().norm().item(), 0.1)

        xv = torch.randn(2, 16, 32)
        tv = torch.randn(2, 32)
        (vb(xv, tv) - torch.randn_like(xv)).pow(2).mean().backward()
        self.assertLess(vb.linear3.weights.grad.norm().item(), 20.0)
        self.assertLess(vb.TMSA.out_proj.weights.grad.norm().item(), 20.0)

    def test_output_branch_gains_do_not_damp_step0_gradients_100x(self):
        model, _ = build_tiny_model()
        self.assertGreaterEqual(model.net.out_gain_vit.item(), 0.5)
        self.assertGreaterEqual(model.net.out_gain_unet.item(), 0.5)


class TestFix4BranchLocalRoutingAndConfigWiring(unittest.TestCase):

    def test_unet_router_crisp_specialization_under_real_config(self):
        """
        Under Utils/configs.py (p_mean=-0.2, p_std=1.2, transition_sigma=2.5,
        sigma_max=80.0, extreme_prob=0.2), U-Net experts must specialize across
        [2.5, 80.0] (>0.90 probability at opposite ends), not stay co-activated at 63/37.
        """
        u_router = ParametricTrajectoryRouter(
            num_experts=model_configs["num_experts"],
            min_sigma=model_configs["transition_sigma"],
            max_sigma=model_configs["sigma_max"],
            p_mean=mask_configs["p_mean"],
            p_std=mask_configs["p_std"],
            sigma_min=model_configs["sigma_min"],
            sigma_max=model_configs["sigma_max"],
            alpha_ext=model_configs["extreme_prob"],
            init_sharpness=loss_configs["sharpness"],
        )
        u_router.eval()
        _, probs_low, _ = u_router(torch.log(torch.tensor([[model_configs["transition_sigma"]]])))
        _, probs_high, _ = u_router(torch.log(torch.tensor([[float(model_configs["sigma_max"])]])))
        self.assertGreater(probs_low[0, 0].item(), 0.90, f"UNet probs at sigma=2.5 collapsed to {probs_low[0].tolist()}")
        self.assertGreater(probs_high[0, 1].item(), 0.90, f"UNet probs at sigma=80.0 collapsed to {probs_high[0].tolist()}")

    def test_path_prior_loss_matches_crisp_branch_local_targets(self):
        from Utils.cdf_utils import branch_local_quantile_centers

        centers_u = branch_local_quantile_centers(
            2, model_configs["transition_sigma"], model_configs["sigma_max"],
            mask_configs["p_mean"], mask_configs["p_std"],
            model_configs["sigma_min"], model_configs["sigma_max"], model_configs["extreme_prob"],
        )
        prior_u = PathPriorLoss(
            expert_centers=centers_u,
            p_mean=mask_configs["p_mean"],
            p_std=mask_configs["p_std"],
            sharpness=loss_configs["sharpness"],
            sigma_min=model_configs["sigma_min"],
            sigma_max=model_configs["sigma_max"],
            alpha_ext=model_configs["extreme_prob"],
            noise_range=(model_configs["transition_sigma"], model_configs["sigma_max"]),
        )
        # Feed expert-0 one-hot at sigma=2.5 and expert-1 one-hot at sigma=80.0;
        # crisp target distribution implies very low KL divergence (<0.05).
        kl_low = prior_u(torch.tensor([[0.999, 0.001]]), torch.tensor([model_configs["transition_sigma"]])).item()
        kl_high = prior_u(torch.tensor([[0.001, 0.999]]), torch.tensor([float(model_configs["sigma_max"])])).item()
        self.assertLess(kl_low, 0.05, f"PathPriorLoss at sigma=2.5 favors blurred co-activation (KL={kl_low:.4f})")
        self.assertLess(kl_high, 0.05, f"PathPriorLoss at sigma=80.0 favors blurred co-activation (KL={kl_high:.4f})")

    def test_alpha_ext_propagates_through_model_and_mask_generator(self):
        model, _ = build_tiny_model(alpha_ext=0.05)
        self.assertAlmostEqual(model.net.Unet_router.alpha_ext, 0.05)
        self.assertAlmostEqual(model.net.vit_router.alpha_ext, 0.05)
        mg = MaskGenerator(
            expert_attributes=[3, 5],
            p_mean=-0.2,
            p_std=1.2,
            noise_range=(2.5, 80.0),
            sigma_min=0.002,
            sigma_max=80.0,
            alpha_ext=0.05,
        )
        self.assertAlmostEqual(mg.alpha_ext, 0.05)

    def test_training_hdmoe_wires_configs_without_keyerror(self):
        """
        Calling training_HDMOE with Utils/configs.py dicts must not fail with
        KeyError: 'BW' on mask_configs['BW'] and should construct MaskGenerators
        and EDM_LOSS_NOD cleanly.
        """
        utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Utils"))
        added = False
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
            added = True
        try:
            import Utils.training as tr
        finally:
            if added and utils_dir in sys.path:
                sys.path.remove(utils_dir)

        mc = copy.deepcopy(model_configs)
        mc["device"] = "cpu"
        mc["img_resolution"] = 16
        mc["internal_channels"] = 16
        mc["time_emb_dim"] = 32
        mc["text_emb_dim"] = 32
        mc["VIT_num_blocks"] = 1
        mc["VIT_patch_sizes"] = [2, 2]
        mc["VIT_num_heads"] = 4
        mc["VIT_emb_size"] = 16
        mc["Unet_num_blocks"] = 1
        mc["Unet_channel_mult"] = [1, 2]
        mc["Unet_model_channels"] = 16
        mc["total_steps"] = 0
        mc["batch_size"] = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            mc["save_dir_stats"] = tmpdir
            mc["save_dir"] = tmpdir

            class _DummyVAE:
                def __init__(self, *a, **kw): pass
                def init(self, *a, **kw): pass

            class _DummyCLIP:
                def __init__(self, *a, **kw): pass
                def init(self, *a, **kw): pass

            class _DummyLogger:
                def __init__(self, *a, **kw): pass

            with patch.object(tr, "get_dataloader_path", return_value=[]), \
                 patch.object(tr, "StabilityVAE", _DummyVAE), \
                 patch.object(tr, "CLIP_EMBED", _DummyCLIP), \
                 patch.object(tr, "Logger", _DummyLogger), \
                 patch.object(tr, "save_checkpoint"):
                tr.training_HDMOE(mc, optim_configs, loss_configs, mask_configs, zeta_configs)


class TestFix6PooledCrossAttn(unittest.TestCase):

    def test_pooled_cross_attn_handles_arbitrary_spatial_resolutions(self):
        """
        PooledCrossAttn configured for img_resolution=64 must also work at 32x32
        (used by generate_test.py) and smaller grids without crashing on fixed pos_enc_q.
        """
        pca = PooledCrossAttn(channels=32, num_heads=4, img_resolution=64, pool_size=16)
        gate = torch.full((2, 1, 1, 1), 0.25)
        for res in (64, 32, 8):
            with self.subTest(res=res):
                u = torch.randn(2, 32, res, res)
                v = torch.randn(2, 32, res, res)
                out = pca(unet_feat=u, vit_feat=v, gate_strength=gate)
                self.assertEqual(out.shape, u.shape)
                self.assertTrue(torch.isfinite(out).all())

    def test_pooled_cross_attn_respects_attn_balance_and_defaults_pure(self):
        pca_default = PooledCrossAttn(channels=32, num_heads=4)
        self.assertEqual(pca_default.cross_attn.attn_balance, 1.0)
        pca_custom = PooledCrossAttn(channels=32, num_heads=4, attn_balance=0.7)
        self.assertEqual(pca_custom.cross_attn.attn_balance, 0.7)
        self.assertEqual(model_configs["attn_pool_balance"], 1.0)


if __name__ == "__main__":
    unittest.main()
