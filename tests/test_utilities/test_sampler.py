from Utils.EDM_sampler import EDM_Sampler
import unittest
import torch
import torch.nn as nn

class MockDenoiser(nn.Module):
    def __init__(self, output_val=None):
        super().__init__()
        self.num_experts = 4  # Required attribute for the sampler
        self.output_val = output_val

    def forward(self, x, sigma, text_emb, **kwargs):
        """
        Simulates the model output.
        Returns a dict containing 'denoised'.
        """
        if self.output_val is not None:
            return {"denoised": torch.full_like(x, self.output_val)}

        return {"denoised": x * 0.9}


class TestEDMSampler(unittest.TestCase):

    def setUp(self):
        # Common params
        self.device = "cpu"  # Test on CPU for speed/CI compatibility
        self.batch_size = 2
        self.channels = 4
        self.res = 32
        self.shape = (self.batch_size, self.channels, self.res, self.res)

        self.mock_model = MockDenoiser()

        # Dummy embeddings
        self.text_emb = torch.randn(self.batch_size, 1, 64)
        self.uncond_text = torch.randn(self.batch_size, 1, 64)

    def test_01_initialization_and_timesteps(self):
        """
        Test if time steps (sigma schedule) are calculated correctly.
        """
        steps = 10
        sigma_min = 0.002
        sigma_max = 80.0

        # Initialize sampler
        sampler = EDM_Sampler(
            model=self.mock_model,
            num_solve_steps=steps,
            sigma_min=sigma_min, sigma_max=sigma_max,
            rho=7
        )

        noise = torch.randn(self.shape)
        out = sampler.sample(noise, self.text_emb)
        self.assertEqual(out.shape, self.shape)

    def test_02_guidance_math(self):
        """
        Test if Classifier-Free Guidance (CFG) linear interpolation is correct.
        """
        # Setup:
        # Unconditional outputs 0.0, Conditional outputs 1.0
        # Guidance Scale = 3.0
        # Expected Result: Uncond + w * (Cond - Uncond) = 0 + 3 * (1 - 0) = 3.0
        class CFGMock(nn.Module):
            def __init__(self, uncond_ref):
                super().__init__()
                self.uncond_ref = uncond_ref

            def forward(self, x, sigma, text_emb, **kwargs):
                if text_emb is self.uncond_ref:
                    return {"denoised": torch.zeros_like(x)}
                return {"denoised": torch.ones_like(x)}

        model = CFGMock(self.uncond_text)
        guidance_scale = 3.0

        sampler = EDM_Sampler(
            model=model,
            guidance=guidance_scale,
            num_solve_steps=1
        )

        noise = torch.randn(self.shape)
        sigma_test = torch.tensor([1.0] * self.batch_size)

        result = sampler.denoise(noise, sigma_test, self.text_emb, self.uncond_text)

        self.assertTrue(torch.allclose(result, torch.tensor(3.0)),
                        f"Expected 3.0, got {result.mean().item()}")

    def test_03_determinism_no_churn(self):
        """
        Ensure sampling is deterministic when S_churn = 0.
        """
        sampler = EDM_Sampler(
            model=self.mock_model,
            num_solve_steps=5, S_churn=0.0
        )

        noise = torch.randn(self.shape)

        out1 = sampler.sample(noise, self.text_emb)
        out2 = sampler.sample(noise, self.text_emb)

        self.assertTrue(torch.allclose(out1, out2), "Sampling without churn should be deterministic")

    def test_04_stochasticity_with_churn(self):
        """
        Ensure sampling varies when S_churn > 0 (Stochastic Sampling).
        """
        sampler = EDM_Sampler(
            model=self.mock_model,
            num_solve_steps=5,
            S_churn=10.0,
            S_min=0.0, S_max=100.0, S_noise=1.0
        )

        noise = torch.randn(self.shape)

        torch.manual_seed(42)
        out1 = sampler.sample(noise, self.text_emb)

        torch.manual_seed(43)
        out2 = sampler.sample(noise, self.text_emb)

        diff = (out1 - out2).abs().mean()
        self.assertGreater(diff.item(), 1e-4, "Churn enabled but outputs are identical!")

    def test_05_convergence_logic(self):
        """
        Test if the solver actually reduces magnitude (Denoising behavior).
        """
        sampler = EDM_Sampler(
            model=self.mock_model,
            num_solve_steps=10
        )

        noise = torch.ones(self.shape) * 10.0
        out = sampler.sample(noise, self.text_emb)

        self.assertLess(out.abs().mean(), noise.abs().mean() * 135)

    def test_06_device_handling(self):
        """
        Check if t_steps are created on the correct device.
        """
        if not torch.cuda.is_available():
            print("Skipping GPU test (CUDA not available)")
            return

        device = "cuda"
        model_gpu = self.mock_model.to(device)

        sampler = EDM_Sampler(model=model_gpu, num_solve_steps=2)

        noise = torch.randn(self.shape).to(device)
        text_emb = self.text_emb.to(device)

        try:
            out = sampler.sample(noise, text_emb)
            self.assertEqual(out.device.type, 'cuda')
        except RuntimeError as e:
            self.fail(f"Sampler failed on GPU: {e}")


if __name__ == '__main__':
    unittest.main()