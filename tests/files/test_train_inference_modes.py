"""
Behavioral contract between model.train() and model.eval():

  - In eval() mode, the same input must produce the exact same output
    every time (no stochastic exploration noise leaking through).
  - In train() mode with zeta > 0, repeated calls are allowed (expected)
    to differ.
  - Switching modes back and forth should not corrupt running state
    (there is no BatchNorm here, but this guards against any future
    stateful layer added without updating this test).
"""
import unittest
import torch
from tests.files.model_factory import build_tiny_model, random_batch


class TestTrainInferenceModes(unittest.TestCase):

    def setUp(self):
        self.model, self.cfg = build_tiny_model()

    def test_eval_mode_is_deterministic_even_with_zeta_requested(self):
        self.model.eval()
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=7.0)
        with torch.no_grad():
            out1 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                               Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=5.0)
            out2 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                               Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=5.0)
        torch.testing.assert_close(out1["denoised"], out2["denoised"], atol=0.0, rtol=0.0)

    def test_repeated_eval_calls_give_identical_router_probs(self):
        self.model.eval()
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=7.0)
        with torch.no_grad():
            out1 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                               Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=1.0)
            out2 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                               Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=1.0)
        torch.testing.assert_close(out1["Unet_router_loss"], out2["Unet_router_loss"])
        torch.testing.assert_close(out1["vit_router_loss"], out2["vit_router_loss"])

    def test_train_mode_with_zeta_can_differ_across_calls(self):
        self.model.train()
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=7.0)
        torch.manual_seed(1)
        out1 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                           Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=5.0)
        torch.manual_seed(2)
        out2 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                           Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=5.0)
        differs = not torch.allclose(out1["Unet_router_loss"], out2["Unet_router_loss"])
        self.assertTrue(differs, "zeta=5.0 in train() mode produced identical router "
                                  "outputs across two different random seeds")

    def test_zeta_zero_is_deterministic_in_train_mode_too(self):
        """zeta=0.0 should be deterministic regardless of train/eval, since
        the noise term itself is scaled to exactly 0."""
        self.model.train()
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=7.0)
        torch.manual_seed(1)
        out1 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                           Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        torch.manual_seed(2)
        out2 = self.model(x=x, sigma=sigma, text_emb=text_emb,
                           Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        torch.testing.assert_close(out1["denoised"], out2["denoised"])

    def test_mode_switch_does_not_change_eval_output(self):
        """train() then eval() should give the same eval output as a model
        that was only ever in eval() -- catches any accidental
        stateful/running-average layer introduced later."""
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=7.0)
        self.model.eval()
        with torch.no_grad():
            baseline = self.model(x=x, sigma=sigma, text_emb=text_emb,
                                   Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        self.model.train()
        _ = self.model(x=x, sigma=sigma, text_emb=text_emb,
                        Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        self.model.eval()
        with torch.no_grad():
            after_switch = self.model(x=x, sigma=sigma, text_emb=text_emb,
                                       Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        torch.testing.assert_close(baseline["denoised"], after_switch["denoised"])


if __name__ == "__main__":
    unittest.main()
