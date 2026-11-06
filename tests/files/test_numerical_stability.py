"""
Edge cases that don't show up with "one clean batch at a mid-range sigma"
but reliably show up once real training/sampling starts: batch size 1,
the exact sigma_min/sigma_max boundary values, a log-spaced sweep across
the full noise range, and a router mask that zeroes out every expert for
some sample in the batch.
"""
import unittest
import torch
import numpy as np
from tests.files.model_factory import build_tiny_model, random_batch


class TestNumericalStability(unittest.TestCase):

    def setUp(self):
        self.model, self.cfg = build_tiny_model()

    def test_batch_size_one_does_not_break_reshape_logic(self):
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, batch_size=1, sigma_value=5.0)
        with torch.no_grad():
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        self.assertEqual(out["denoised"].shape, x.shape)
        self.assertTrue(torch.isfinite(out["denoised"]).all())

    def test_sigma_min_and_sigma_max_boundary_values(self):
        for sigma_val in [self.cfg["sigma_min"], self.cfg["sigma_max"]]:
            with self.subTest(sigma=sigma_val):
                x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=sigma_val)
                with torch.no_grad():
                    out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                                      Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
                self.assertTrue(torch.isfinite(out["denoised"]).all(),
                                 f"non-finite output at boundary sigma={sigma_val}")

    def test_log_spaced_sweep_across_full_noise_range(self):
        """20 log-spaced sigma values from sigma_min to sigma_max -- the
        actual range the model will see during real training/sampling."""
        sigmas = np.geomspace(self.cfg["sigma_min"], self.cfg["sigma_max"], num=20)
        x, _, text_emb, Umask, Vmask = random_batch(self.cfg, batch_size=len(sigmas))
        sigma = torch.tensor(sigmas, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        self.assertTrue(torch.isfinite(out["denoised"]).all(),
                         "non-finite output somewhere in the log-spaced sigma sweep")

    def test_mixed_sigma_batch_each_sample_gets_its_own_coefficients(self):
        """A batch with wildly different sigma per sample should not let
        one sample's coefficients leak onto another's output."""
        sigmas = torch.tensor([0.01, 50.0])
        x, _, text_emb, Umask, Vmask = random_batch(self.cfg, batch_size=2)
        with torch.no_grad():
            out_batched = self.model(x=x, sigma=sigmas, text_emb=text_emb,
                                       Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
            out_single_0 = self.model(x=x[:1], sigma=sigmas[:1], text_emb=text_emb[:1],
                                        Unet_router_mask=Umask[:1], Vit_router_mask=Vmask[:1], zeta=0.0)
        # Sample 0 processed alone should match sample 0 processed in the batch
        # (loose tolerance: floating-point batching order can shift things slightly).
        torch.testing.assert_close(
            out_batched["denoised"][0], out_single_0["denoised"][0], atol=1e-3, rtol=1e-2,
        )

    def test_all_experts_masked_for_one_sample_does_not_produce_nan(self):
        """
        router_to_experts clamps its denominator to 0.1 specifically to
        survive this case. Verify that guard actually works end to end,
        not just in isolation.
        """
        x, sigma, text_emb, _, _ = random_batch(self.cfg, batch_size=2, sigma_value=5.0)
        Umask = torch.ones(2, self.cfg["num_experts"])
        Umask[0] = 0.0  # sample 0 has every UNet expert masked out
        Vmask = torch.ones(2, self.cfg["num_experts"])
        with torch.no_grad():
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        self.assertTrue(torch.isfinite(out["denoised"]).all(),
                         "all-experts-masked edge case produced NaN/Inf")

    def test_return_log_var_produces_finite_values(self):
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=10.0)
        with torch.no_grad():
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0,
                              return_log_var=True)
        self.assertIsNotNone(out["log_var"])
        self.assertTrue(torch.isfinite(out["log_var"]).all())


if __name__ == "__main__":
    unittest.main()
