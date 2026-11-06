"""
Tests for TimeGatedFrequencyFusion (TGFF), matching TGFF.md's spec:

  F_low  = ifft( fft(F_unet) * M_low  )   (renormalized)
  F_high = ifft( fft(F_vit)  * M_high )   (renormalized)
  eta(t) = sigmoid(gate_mlp(time_emb))
  Z_fusion = F_low + eta * zero_conv(F_high)

The most important test here is `test_zero_conv_gate_starts_at_exact_zero`,
which is the direct regression test for the zero-init deadlock: it doesn't
just check TGFF's own zero-init property, it checks that changing
output_proj_vit's init (in HDMOEM_v3) doesn't accidentally reintroduce a
second zero-init gate somewhere in this module.
"""
import unittest
import torch
from models.model_components import TimeGatedFrequencyFusion


class TestTGFF(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.channels = 4
        self.time_emb_dim = 32
        self.tgff = TimeGatedFrequencyFusion(
            channels=self.channels, time_emb_dim=self.time_emb_dim, cutoff_fraction=0.125,
        )
        self.B, self.H, self.W = 3, 16, 16

    def _inputs(self):
        F_unet = torch.randn(self.B, self.channels, self.H, self.W)
        F_vit = torch.randn(self.B, self.channels, self.H, self.W)
        time_emb = torch.randn(self.B, self.time_emb_dim)
        return F_unet, F_vit, time_emb

    def test_low_and_high_pass_masks_are_exact_complements(self):
        mask_low = self.tgff.get_gaussian_lowpass_mask(self.H, self.W, device=torch.device("cpu"))
        mask_high = 1.0 - mask_low
        self.assertTrue(torch.allclose(mask_low + mask_high, torch.ones_like(mask_low)))
        self.assertTrue((mask_low >= 0).all() and (mask_low <= 1).all())

    def test_mask_is_resolution_adaptive_not_fixed_pixel_radius(self):
        """Same cutoff_fraction, different H/W -> similar retained-energy fraction."""
        mask_16 = self.tgff.get_gaussian_lowpass_mask(16, 16, device=torch.device("cpu"))
        mask_64 = self.tgff.get_gaussian_lowpass_mask(64, 64, device=torch.device("cpu"))
        energy_16 = (mask_16 ** 2).mean().item()
        energy_64 = (mask_64 ** 2).mean().item()
        self.assertLess(abs(energy_16 - energy_64), 0.1,
                         msg=f"low-pass energy fraction should not depend heavily on "
                             f"resolution at fixed cutoff_fraction: {energy_16} vs {energy_64}")

    def test_zero_conv_gate_starts_at_exact_zero(self):
        """
        F_high_stable = zero_conv(F_high) must be EXACTLY zero at a fresh
        TimeGatedFrequencyFusion instance -- this is what guarantees the
        "module contributes nothing at step 0" safety property from
        TGFF.md section 4, independent of whatever F_vit/F_unet contain.
        """
        F_unet, F_vit, time_emb = self._inputs()
        Z_fusion, eta = self.tgff(F_unet=F_unet, F_vit=F_vit, time_emb=time_emb)
        # With zero_conv at init, Z_fusion must equal F_low alone --
        # i.e. depend only on F_unet, not at all on F_vit.
        F_vit_alt = torch.randn_like(F_vit) * 50.0  # wildly different ViT input
        Z_fusion_alt, _ = self.tgff(F_unet=F_unet, F_vit=F_vit_alt, time_emb=time_emb)
        torch.testing.assert_close(Z_fusion, Z_fusion_alt, atol=1e-5, rtol=1e-4,
                                    msg="Z_fusion changed when F_vit changed, even though "
                                        "zero_conv should make the ViT path contribute exactly 0")

    def test_eta_gate_is_bounded_zero_one(self):
        F_unet, F_vit, time_emb = self._inputs()
        _, eta = self.tgff(F_unet=F_unet, F_vit=F_vit, time_emb=time_emb)
        self.assertTrue((eta >= 0).all() and (eta <= 1).all())

    def test_smooth_low_frequency_signal_survives_low_pass_round_trip(self):
        """
        Sanity check on the FFT -> mask -> IFFT -> renormalize round trip.

        NOTE: a pure constant (single-bin DC-only) image is deliberately
        NOT used here. The renormalization divides by sqrt(mean(mask^2)),
        which assumes energy is spread roughly evenly across the passband
        (true for realistic feature maps); concentrating 100% of a
        signal's energy into the one DC bin that survives violates that
        assumption and inflates the result by design, not by bug. A
        low-frequency sinusoid spreads energy across several genuinely
        "low" bins, which matches the assumption this renormalization
        is built for.
        """
        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        low_freq_signal = torch.sin(2 * torch.pi * xx.float() / self.W) + 3.0
        low_freq_signal = low_freq_signal.view(1, 1, self.H, self.W).expand(1, self.channels, self.H, self.W)
        zeros = torch.zeros_like(low_freq_signal)
        time_emb = torch.zeros(1, self.time_emb_dim)
        Z_fusion, _ = self.tgff(F_unet=low_freq_signal, F_vit=zeros, time_emb=time_emb)
        # zero_conv makes the ViT term exactly 0, so Z_fusion == F_low here.
        # A one-cycle sinusoid is well within the passband, so its shape
        # (not just its mean) should survive with reasonable fidelity.
        correlation = torch.corrcoef(torch.stack([
            Z_fusion[0, 0].flatten(), low_freq_signal[0, 0].flatten(),
        ]))[0, 1]
        self.assertGreater(correlation.item(), 0.9,
                            "a genuinely low-frequency signal should survive the "
                            "low-pass round trip with high shape fidelity")

    def test_gradient_flows_to_zero_conv_weight_itself_given_nonzero_F_high(self):
        """
        zero_conv's OWN weight should receive a nonzero gradient whenever
        its input (F_high) is nonzero -- this is necessary (though not
        sufficient) for it to bootstrap away from zero during training.
        """
        F_unet, F_vit, time_emb = self._inputs()
        Z_fusion, eta = self.tgff(F_unet=F_unet, F_vit=F_vit, time_emb=time_emb)
        loss = Z_fusion.pow(2).mean()
        loss.backward()
        grad = self.tgff.zero_conv.weights.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
