"""
Validates the EDM2 (Karras et al. 2024) magnitude-preserving claim: every
primitive op (mp_silu, mp_sum, mp_cat, normalize, MP_Conv) should map
unit-variance input to approximately unit-variance output.

Part 1 tests the primitives in isolation (fast, no model needed) so a
failure points at one specific function, not "somewhere in the network".
Part 2 checks integration-level statistics on the actual tiny model,
with tolerances wide enough to allow real architectural attenuation
(e.g. UNet's low-pass-filtered skips) without allowing true regressions
(silent explosion/collapse) to pass.
"""
import unittest
import torch
from models import model_internals as m
from tests.files.model_factory import build_tiny_model, random_batch, SIGMA_PROBES


def std_of(x):
    return x.detach().float().std().item()


class TestMPPrimitives(unittest.TestCase):
    """Part 1: pure-function checks, no model instantiation."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4096)
        self.y = torch.randn(4096)

    def test_normalize_forces_unit_variance(self):
        out = m.normalize(self.x.unsqueeze(0), dim=[1])
        # normalize() targets unit *norm scaled by sqrt(numel)*, i.e. unit std
        self.assertAlmostEqual(std_of(out), 1.0, delta=0.05)

    def test_mp_silu_preserves_variance_on_unit_input(self):
        out = m.mp_silu(self.x)
        self.assertAlmostEqual(std_of(out), 1.0, delta=0.1)

    def test_mp_sum_preserves_variance_for_various_t(self):
        for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
            with self.subTest(t=t):
                out = m.mp_sum(self.x, self.y, t=t)
                self.assertAlmostEqual(std_of(out), 1.0, delta=0.1)

    def test_mp_cat_preserves_variance(self):
        a = torch.randn(8, 16, 4, 4)
        b = torch.randn(8, 32, 4, 4)
        out = m.mp_cat(a, b, dim=1, t=0.5)
        self.assertAlmostEqual(std_of(out), 1.0, delta=0.15)
        self.assertEqual(out.shape[1], 16 + 32)

    def test_mp_conv_linear_mode_preserves_variance(self):
        layer = m.MP_Conv(in_channels=64, out_channels=64, kernel=())
        x = torch.randn(256, 64)
        out = layer(x, gain=1.0)
        self.assertAlmostEqual(std_of(out), 1.0, delta=0.15)

    def test_mp_conv_spatial_mode_preserves_variance(self):
        layer = m.MP_Conv(in_channels=8, out_channels=8, kernel=(3, 3))
        x = torch.randn(4, 8, 16, 16)
        out = layer(x, gain=1.0)
        self.assertAlmostEqual(std_of(out), 1.0, delta=0.2)

    def test_mp_conv_gain_scales_output_std_linearly(self):
        layer = m.MP_Conv(in_channels=32, out_channels=32, kernel=())
        x = torch.randn(512, 32)
        std_gain1 = std_of(layer(x, gain=1.0))
        std_gain2 = std_of(layer(x, gain=2.0))
        self.assertAlmostEqual(std_gain2 / std_gain1, 2.0, delta=0.1)

    def test_mp_conv_zero_init_produces_exact_zero_output(self):
        layer = m.MP_Conv(in_channels=16, out_channels=16, kernel=(3, 3))
        torch.nn.init.constant_(layer.weights, 0.0)
        x = torch.randn(2, 16, 8, 8)
        out = layer(x)
        self.assertTrue(torch.all(out == 0.0))

    def test_gaussian_freq_mask_is_resolution_adaptive(self):
        """
        Same cutoff_fraction at different (H, W) should pass approximately
        the same *fraction* of total frequency-domain energy, not the same
        absolute pixel radius. This is the regression test for the fix
        that replaced the fixed sigma_g=4.0 cutoff.
        """
        fractions = []
        for size in [8, 16, 32, 64]:
            mask = m.get_gaussian_freq_mask(size, size, cutoff_fraction=0.125,
                                             device=torch.device("cpu"))
            fractions.append((mask ** 2).mean().item())
        # All four resolutions should retain a similar energy fraction
        # (within a loose band -- exact equality isn't expected because
        # discretization effects differ at very small grids).
        self.assertLess(max(fractions) - min(fractions), 0.15,
                         msg=f"low-pass energy fraction should be roughly "
                             f"resolution-invariant, got {fractions}")


class TestMPIntegration(unittest.TestCase):
    """Part 2: end-to-end activation statistics on the real tiny model."""

    def setUp(self):
        self.model, self.cfg = build_tiny_model()

    def test_E_unified_is_unit_variance_across_sigma_range(self):
        for name, sigma_val in SIGMA_PROBES.items():
            with self.subTest(sigma=name):
                x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=sigma_val)
                time_vec = (sigma.log() / 4.0)
                with torch.no_grad():
                    time_embed = self.model.net.Fourier_emb(x=time_vec)
                    time_embed = self.model.net.out_fourier1(x=time_embed)
                    time_embed = self.model.net.out_fourier2(x=m.mp_silu(x=time_embed))
                    E_unified = self.model.net.merge_op(time_emb=time_embed, text_seq=text_emb)
                self.assertAlmostEqual(std_of(E_unified), 1.0, delta=0.15,
                                       msg=f"E_unified std drifted at sigma={name}")

    def test_x_feats_scales_with_c_in_as_edm_specifies(self):
        """
        x_feats (post Cin, pre time-gain) should scale in proportion to
        c_in(sigma) = 1/sqrt(sigma_data^2 + sigma^2) -- this is an EDM
        design requirement, not a magnitude-preservation one, so the
        tolerance here is on the *ratio* between two sigmas, not on std=1.
        """
        x1, sigma1, text1, U1, V1 = random_batch(self.cfg, sigma_value=0.002, seed=5)
        x2, sigma2, text2, U2, V2 = random_batch(self.cfg, sigma_value=80.0, seed=5)
        c_in1 = 1.0 / (self.model.sigma_data ** 2 + 0.002 ** 2) ** 0.5
        c_in2 = 1.0 / (self.model.sigma_data ** 2 + 80.0 ** 2) ** 0.5
        expected_ratio = c_in1 / c_in2

        with torch.no_grad():
            feats1 = self.model.net.input_proj(x1 * c_in1)
            feats2 = self.model.net.input_proj(x2 * c_in2)
        observed_ratio = std_of(feats1) / std_of(feats2)
        # Loose tolerance: this checks order-of-magnitude correctness, not
        # exact equality, since input_proj itself adds some variance noise.
        self.assertGreater(observed_ratio, expected_ratio * 0.3)
        self.assertLess(observed_ratio, expected_ratio * 3.0)

    def test_unet_and_vit_expert_outputs_within_reasonable_bounds(self):
        """
        Loose sanity bound, not a tight magnitude-preservation assertion:
        catches true regressions (silent collapse to ~0 or explosion to
        >>10) while tolerating the known ~0.5 std UNet branch finding.
        """
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=10.0)
        net = self.model.net
        with torch.no_grad():
            time_vec = sigma.log() / 4.0
            time_embed = net.out_fourier2(m.mp_silu(net.out_fourier1(net.Fourier_emb(time_vec))))
            E_unified = net.merge_op(time_emb=time_embed, text_seq=text_emb)
            e_global = m.normalize(E_unified.mean(dim=1), dim=[1])
            x_feats = m.normalize(net.input_proj(x))
            for expert in net.Unet_experts:
                out = expert(x=x_feats, time_emb=e_global, text_emb=E_unified)
                self.assertGreater(std_of(out), 0.05, "UNet expert output collapsed toward 0")
                self.assertLess(std_of(out), 10.0, "UNet expert output exploded")
            for expert in net.VIT_experts:
                out = expert(x=x_feats, time_emb=e_global, text_emb=E_unified)
                self.assertGreater(std_of(out), 0.05, "ViT expert output collapsed toward 0")
                self.assertLess(std_of(out), 10.0, "ViT expert output exploded")

    def test_no_nan_or_inf_anywhere_in_forward_pass(self):
        for name, sigma_val in SIGMA_PROBES.items():
            with self.subTest(sigma=name):
                x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=sigma_val)
                with torch.no_grad():
                    out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                                      Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
                self.assertTrue(torch.isfinite(out["denoised"]).all(),
                                f"non-finite values at sigma={name}")


if __name__ == "__main__":
    unittest.main()
