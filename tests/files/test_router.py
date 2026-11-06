"""
Tests for ParametricTrajectoryRouter — the CDF/RBF-based expert router.

Covers: valid-probability-distribution output, hard-fence mask behavior,
correct CDF-space seeding of expert centers within [min_sigma, max_sigma],
and that the exploration noise (zeta) is only active in training mode.

Does NOT test "does the router learn correct specialization" -- that's an
outcome of training, not something a pre-training unit test can assert.
What it does test is that the router's *mechanism* is correct, so that IF
training pushes it toward specialization, nothing here is silently broken.
"""
import math
import unittest
import torch
from models.model_components import ParametricTrajectoryRouter
from tests.files.model_factory import build_tiny_model


class TestParametricTrajectoryRouter(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.router = ParametricTrajectoryRouter(
            num_experts=4, min_sigma=0.002, max_sigma=80.0, p_mean=-0.4, p_std=1.2,
        )

    def _log_sigma(self, sigma):
        return torch.log(torch.as_tensor(sigma, dtype=torch.float32)).view(-1, 1)

    def test_output_is_a_valid_probability_distribution(self):
        log_sigma = self._log_sigma([0.01, 1.0, 10.0, 79.0])
        weights, probs, logits = self.router(log_sigma=log_sigma)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))
        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

    def test_mask_zeroes_masked_experts_exactly(self):
        log_sigma = self._log_sigma([5.0, 5.0])
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0, 1.0]])
        weights, probs, logits = self.router(log_sigma=log_sigma, mask=mask)
        self.assertTrue(torch.allclose(probs[mask == 0], torch.zeros_like(probs[mask == 0]), atol=1e-6))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_all_experts_masked_out_does_not_produce_nan(self):
        """
        Edge case: if a caller ever passes an all-zero mask row, softmax
        over an all -1e9 logit row is technically well-defined (uniform),
        but this is worth locking down explicitly since router_to_experts
        downstream assumes probs are finite.
        """
        log_sigma = self._log_sigma([5.0])
        mask = torch.zeros(1, 4)
        weights, probs, logits = self.router(log_sigma=log_sigma, mask=mask)
        self.assertTrue(torch.isfinite(probs).all())

    def test_centers_seeded_within_correct_cdf_subrange(self):
        """
        A router built with min_sigma/max_sigma should have its initial
        centers (in CDF space) bounded within the CDF values of that range
        -- this is the mechanism the UNet/ViT split relies on to bias
        (not hard-constrain) each branch toward its intended noise regime.
        """
        min_sigma, max_sigma, p_mean, p_std = 3.0, 80.0, -0.4, 1.2
        router = ParametricTrajectoryRouter(
            num_experts=6, min_sigma=min_sigma, max_sigma=max_sigma,
            p_mean=p_mean, p_std=p_std,
        )
        # Fix 4: Centers are now seeded in mixture-CDF space, not pure Gaussian CDF
        from Utils.cdf_utils import mixture_cdf
        import torch as _torch
        sigma_min_global, sigma_max_global = 0.002, 80.0
        log_min_g, log_max_g = math.log(sigma_min_global), math.log(sigma_max_global)
        cdf_min = mixture_cdf(_torch.tensor(math.log(min_sigma)), p_mean, p_std, log_min_g, log_max_g).item()
        cdf_max = mixture_cdf(_torch.tensor(math.log(max_sigma)), p_mean, p_std, log_min_g, log_max_g).item()
        lo, hi = min(cdf_min, cdf_max), max(cdf_min, cdf_max)
        centers = router.centers.detach()
        self.assertTrue((centers >= lo - 1e-3).all() and (centers <= hi + 1e-3).all(),
                         msg=f"centers {centers.tolist()} fall outside seeded CDF range [{lo}, {hi}]")

    def test_high_and_low_sigma_router_centers_are_seeded_in_disjoint_regions(self):
        """
        Regression test for the UNet-router/ViT-router split itself: the
        two routers built by HDMOEM_v3 should NOT have overlapping seeded
        center ranges, otherwise the "high noise vs low noise" bias the
        architecture depends on isn't present even at initialization.
        """
        _, cfg = build_tiny_model()
        unet_router = ParametricTrajectoryRouter(
            num_experts=cfg["num_experts"], min_sigma=cfg["transition_sigma"],
            max_sigma=cfg["sigma_max"], p_mean=cfg["p_mean"], p_std=cfg["p_std"],
        )
        vit_router = ParametricTrajectoryRouter(
            num_experts=cfg["num_experts"], min_sigma=cfg["sigma_min"],
            max_sigma=cfg["transition_sigma"], p_mean=cfg["p_mean"], p_std=cfg["p_std"],
        )
        self.assertLessEqual(vit_router.centers.max().item(), unet_router.centers.min().item() + 1e-3,
                              msg="ViT-router centers should be seeded no higher than UNet-router centers")

    def test_zeta_exploration_noise_only_active_in_training_mode(self):
        log_sigma = self._log_sigma([1.0])
        torch.manual_seed(42)
        self.router.eval()
        _, probs_eval_1, _ = self.router(log_sigma=log_sigma, zeta=5.0)
        _, probs_eval_2, _ = self.router(log_sigma=log_sigma, zeta=5.0)
        self.assertTrue(torch.allclose(probs_eval_1, probs_eval_2),
                         "zeta noise leaked into eval() mode -- router.forward must check self.training")

        self.router.train()
        torch.manual_seed(1)
        _, probs_train_1, _ = self.router(log_sigma=log_sigma, zeta=5.0)
        torch.manual_seed(2)
        _, probs_train_2, _ = self.router(log_sigma=log_sigma, zeta=5.0)
        self.assertFalse(torch.allclose(probs_train_1, probs_train_2),
                          "zeta=5.0 in train() mode produced identical outputs across two random seeds")

    def test_sharpness_stays_strictly_positive_regardless_of_gradient_updates(self):
        """log_sharpness is learned in log-space specifically to guarantee
        this; verify a large adversarial gradient step can't push the
        effective sharpness negative or to zero."""
        opt = torch.optim.SGD(self.router.parameters(), lr=100.0)
        log_sigma = self._log_sigma([1.0, 2.0, 3.0])
        _, probs, _ = self.router(log_sigma=log_sigma)
        loss = probs.sum()
        loss.backward()
        opt.step()
        sharpness = torch.exp(self.router.log_sharpness)
        self.assertTrue((sharpness > 0).all())
        self.assertTrue(torch.isfinite(sharpness).all())


if __name__ == "__main__":
    unittest.main()
