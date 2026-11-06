"""
Gradient-flow invariants for HDMOEM v3.

This is the single most important file in the suite to run before starting
training. It directly encodes the zero-init deadlock finding: two
zero-initialized layers stacked in series on the same branch (with no
alternate ungated path) produce a PERMANENT zero gradient -- not slow
convergence, an exact mathematical fixed point that persists for any
number of optimizer steps.

`test_every_parameter_group_receives_gradient_within_n_steps` is the
regression test. It is deliberately generic (loops over every named
parameter group from model_factory.named_param_groups) rather than
hardcoding "output_proj_vit" and "tgff.zero_conv" by name, so it will also
catch a *new* accidental double-zero-init introduced anywhere else in the
graph later, not just a reintroduction of this exact bug.
"""
import unittest
import torch
from tests.files.model_factory import (
    build_tiny_model, random_batch, named_param_groups, grad_abs_sum,
)

# How many optimizer steps a healthy zero-init bootstrap chain is allowed
# to take before every parameter group must show nonzero gradient. A
# single zero-init gate should wake up within 1-2 steps; this budget is
# deliberately generous (5) to avoid false positives from slow-but-healthy
# bootstrapping, while still being tight enough to catch a true deadlock
# (which would still show exactly 0.0 after 30+ steps, let alone 5).
BOOTSTRAP_STEP_BUDGET = 5


class TestGradientFlow(unittest.TestCase):

    def setUp(self):
        self.model, self.cfg = build_tiny_model()

    def test_zero_init_layers_start_at_exact_zero(self):
        """Sanity precondition for every other test in this file."""
        net = self.model.net
        zero_init_layers = {
            "unet_proj_gate": net.unet_proj_gate,
            "vit_proj_gate": net.vit_proj_gate,
            "blend_gate": net.blend_gate,
            "time_proj_gate": net.time_proj_gate,
        }
        if net._cross_attn_enabled:
            zero_init_layers["out_proj_gate"] = net.pooled_cross_attn.out_proj_gate
        for name, param in zero_init_layers.items():
            with self.subTest(layer=name):
                self.assertTrue(torch.all(param.data == 0.0),
                                 f"{name} expected to start at exact zero")

    def test_model_output_equals_pure_skip_term_at_init(self):
        """
        With every gate at zero, D(x) should equal c_skip * x exactly
        (the net's own contribution is exactly 0). This is the "safe
        start" property the whole zero-init scheme exists to guarantee.
        """
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=10.0)
        with torch.no_grad():
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        sigma_data = self.model.sigma_data
        c_skip = sigma_data ** 2 / (10.0 ** 2 + sigma_data ** 2)
        expected = c_skip * x
        torch.testing.assert_close(out["denoised"], expected, atol=1e-4, rtol=1e-3)

    def test_no_nan_or_inf_in_any_gradient_after_first_backward(self):
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=10.0)
        out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                          Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        loss = out["denoised"].pow(2).mean()
        loss.backward()
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), f"non-finite gradient in {name}")

    def test_every_parameter_group_receives_gradient_within_n_steps(self):
        """
        THE regression test for the zero-init deadlock. Runs real Adam
        optimizer steps from a fresh init with no manual intervention and
        asserts that every named architectural region has accumulated a
        nonzero gradient sum by step BOOTSTRAP_STEP_BUDGET. A permanently
        deadlocked branch (two zero-init layers in series, no other path)
        will show EXACTLY 0.0 here even after far more steps than this.
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        groups = named_param_groups(self.model)

        for step in range(BOOTSTRAP_STEP_BUDGET):
            x, sigma, text_emb, Umask, Vmask = random_batch(
                self.cfg, sigma_value=float(5 + step), seed=step)
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
            target = torch.randn_like(out["denoised"])
            loss = (out["denoised"] - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # One more forward/backward to read final-state gradients.
        x, sigma, text_emb, Umask, Vmask = random_batch(self.cfg, sigma_value=8.0, seed=99)
        out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                          Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
        loss = (out["denoised"] - torch.randn_like(out["denoised"])).pow(2).mean()
        self.model.zero_grad()
        loss.backward()

        dead_groups = []
        for name, module in groups.items():
            if grad_abs_sum(module) == 0.0:
                dead_groups.append(name)

        self.assertEqual(
            dead_groups, [],
            msg=(f"The following parameter groups received EXACTLY zero gradient "
                 f"after {BOOTSTRAP_STEP_BUDGET} optimizer steps: {dead_groups}. "
                 f"This means at least two zero-initialized layers are stacked in "
                 f"series on this path with no alternate ungated route -- see the "
                 f"output_proj_vit / tgff.zero_conv and pooled_cross_attn.out_proj / "
                 f"output_proj_attn deadlock pattern. Fix: exactly one zero-init "
                 f"gate per branch, never two in series.")
        )

    def test_vit_branch_specifically_is_not_deadlocked(self):
        """
        Narrower, more explicit version of the test above, scoped to the
        exact chain that was previously broken. Kept as a separate test
        (rather than relying only on the generic loop) so a CI failure
        message names this specific historical bug directly.
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for step in range(BOOTSTRAP_STEP_BUDGET):
            x, sigma, text_emb, Umask, Vmask = random_batch(
                self.cfg, sigma_value=float(2 + step), seed=100 + step)
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
            loss = (out["denoised"] - torch.randn_like(out["denoised"])).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        vit_expert_param = next(self.model.net.VIT_experts[0].parameters())
        self.assertFalse(
            torch.all(vit_expert_param == vit_expert_param.detach().clone()).item() and
            grad_abs_sum(self.model.net.VIT_experts) == 0.0,
            "VIT_experts received zero cumulative gradient -- ViT branch is deadlocked",
        )
        self.assertGreater(grad_abs_sum(self.model.net.output_proj_vit), 0.0,
                            "output_proj_vit itself never bootstrapped away from zero")
        self.assertGreater(grad_abs_sum(self.model.net.blend_residual), 0.0,
                            "blend_residual never bootstrapped away from zero")

    def test_pooled_cross_attn_branch_specifically_is_not_deadlocked(self):
        if not self.model.net._cross_attn_enabled:
            self.skipTest("PooledCrossAttn disabled for this config "
                           "(internal_channels not divisible by VIT_num_heads)")
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for step in range(BOOTSTRAP_STEP_BUDGET):
            x, sigma, text_emb, Umask, Vmask = random_batch(
                self.cfg, sigma_value=float(2 + step), seed=200 + step)
            out = self.model(x=x, sigma=sigma, text_emb=text_emb,
                              Unet_router_mask=Umask, Vit_router_mask=Vmask, zeta=0.0)
            loss = (out["denoised"] - torch.randn_like(out["denoised"])).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.assertGreater(grad_abs_sum(self.model.net.pooled_cross_attn.out_proj), 0.0,
                            "pooled_cross_attn.out_proj never bootstrapped away from zero")
        self.assertGreater(grad_abs_sum(self.model.net.output_proj_attn), 0.0,
                            "output_proj_attn never bootstrapped away from zero")
        self.assertGreater(grad_abs_sum(self.model.net.pooled_cross_attn.cross_attn), 0.0,
                            "gradient never reached the cross-attention internals")


if __name__ == "__main__":
    unittest.main()
