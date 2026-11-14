import torch
import unittest
from models import model_internals as m

class TestVITAttention(unittest.TestCase):

    def setUp(self):
        """Set up common parameters before every test."""
        self.batch_size = 2
        self.seq_len = 32
        self.emb_dim = 64
        self.time_dim = 128
        self.num_heads = 4

        self.B, self.H, self.D = 2, 4, 64
        self.T_dim = 128
        self.seq_img = 32
        self.seq_txt = 77
        self.d_ctx = 128  # Different from D to force Cross-Attn checks
        # Initialize model
        self.model = m.MP_Attention(
            num_heads=self.num_heads,
            emb_dim=self.emb_dim,
            seq_ln=self.seq_len,
            time_dim=self.time_dim
        )

        # Standard Inputs
        self.x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        self.t = torch.randn(self.batch_size, self.time_dim)

    def test_output_shape(self):
        """Test 1: Does the forward pass return the correct shape?"""
        out = self.model(self.x, gain_s=1.0, gain_t=1.0, time_embedding=self.t)
        expected_shape = (self.batch_size, self.seq_len, self.emb_dim)
        self.assertEqual(out.shape, expected_shape, "Output shape mismatch")

    def test_tmsa_functionality(self):
        """Test 2: Does changing the time embedding actually change the output?"""
        t1 = torch.randn(self.batch_size, self.time_dim)
        t2 = torch.randn(self.batch_size, self.time_dim)

        # Ensure t1 and t2 are different
        while torch.allclose(t1, t2):
            t2 = torch.randn(self.batch_size, self.time_dim)

        out1 = self.model(self.x, gain_s=1.0, gain_t=1.0, time_embedding=t1)
        out2 = self.model(self.x, gain_s=1.0, gain_t=1.0, time_embedding=t2)

        # The outputs should be significantly different
        diff = (out1 - out2).abs().sum()
        self.assertGreater(diff.item(), 1e-4, "TMSA failed: Different times produced identical outputs")

    def test_gain_zero_disables_time(self):
        """Test 3: Does setting gain_t=0 make the model ignore the time embedding?"""
        t1 = torch.randn(self.batch_size, self.time_dim)
        t2 = torch.randn(self.batch_size, self.time_dim)

        # With gain_t = 0, time projection should be zeroed out
        out1 = self.model(self.x, gain_s=1.0, gain_t=0.0, time_embedding=t1)
        out2 = self.model(self.x, gain_s=1.0, gain_t=0.0, time_embedding=t2)

        # Outputs should be nearly identical (floating point errors allowed)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5),
                        "gain_t=0 failed: Model still reacted to time changes")

    def test_sequence_robustness(self):
        """Test 4: Can the model handle a sequence length smaller than initialization?"""
        short_seq = self.seq_len // 2
        x_short = torch.randn(self.batch_size, short_seq, self.emb_dim)

        try:
            out = self.model(x_short, gain_s=1.0, gain_t=1.0, time_embedding=self.t)
            self.assertEqual(out.shape, (self.batch_size, short_seq, self.emb_dim))
        except Exception as e:
            self.fail(f"Model crashed on smaller sequence length: {e}")

    def test_sequence_overflow(self):
        """Test 5: Does the model interpolate correctly for high-res inputs?"""
        long_seq = self.seq_len * 2  # Double the resolution
        x_long = torch.randn(self.batch_size, long_seq, self.emb_dim)

        # This should NO LONGER fail. It should run and return the correct shape.
        try:
            out = self.model(x_long, gain_s=1.0, gain_t=1.0, time_embedding=self.t)
            self.assertEqual(out.shape, (self.batch_size, long_seq, self.emb_dim))
        except Exception as e:
            self.fail(f"Interpolation failed on long sequence: {e}")

    def test_gradient_flow(self):
        """General: Ensure gradients flow to inputs, context, and time."""
        model = m.MP_Attention(self.H, self.D, self.seq_img, time_dim=self.T_dim,
                             context_dim=self.d_ctx, is_cross_attn=True)

        q = torch.randn(self.B, self.seq_img, self.D, requires_grad=True)
        ctx = torch.randn(self.B, self.seq_txt, self.d_ctx, requires_grad=True)
        t = torch.randn(self.B, self.T_dim, requires_grad=True)

        out = model(q, 1.0, 1.0, context=ctx, time_embedding=t)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(q.grad, "Input Grad Missing")
        self.assertIsNotNone(ctx.grad, "Context Grad Missing")
        self.assertIsNotNone(t.grad, "Time Grad Missing")
        grad_q_mag = q.grad.abs().sum().item()
        grad_ctx_mag = ctx.grad.abs().sum().item()
        grad_t_mag = t.grad.abs().sum().item()

        self.assertGreater(grad_q_mag, 1e-6, f"Query gradient is effectively zero: {grad_q_mag}")
        self.assertGreater(grad_ctx_mag, 1e-6, f"Context gradient is effectively zero: {grad_ctx_mag}")
        self.assertGreater(grad_t_mag, 1e-6, f"Time gradient is effectively zero: {grad_t_mag}")

    def test_no_time_config(self):
        """Test 7: Does the model work if initialized without time dependence?"""
        model_static = m.MP_Attention(
            num_heads=self.num_heads,
            emb_dim=self.emb_dim,
            seq_ln=self.seq_len,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        # Should run without crashing and without time_embedding arg
        out = model_static(x, gain_s=1.0, gain_t=0.0, time_embedding=None)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.emb_dim))
        self.assertFalse(hasattr(model_static, 'q_time'))

    def test_ca_asymmetric_shapes(self):
        """CA: Query (Image) and Context (Text) have different lengths."""
        model = m.MP_Attention(self.H, self.D, self.seq_img, time_dim=self.T_dim,
                             context_dim=self.d_ctx, is_cross_attn=True)

        q = torch.randn(self.B, self.seq_img, self.D)
        k = torch.randn(self.B, self.seq_txt, self.d_ctx)  # Different Len, Different Dim
        t = torch.randn(self.B, self.T_dim)

        # If logic tries to add Square Bias to Rectangular Map, this crashes
        try:
            out = model(q, 1.0, 1.0, context=k, time_embedding=t)
            self.assertEqual(out.shape, (self.B, self.seq_img, self.D))
        except RuntimeError as e:
            self.fail(f"Cross Attn crashed on asymmetric shapes: {e}")

    def test_ca_time_injection_logic(self):
        """CA: Time should affect output (via Q) even in Cross Attn."""
        model = m.MP_Attention(self.H, self.D, self.seq_img, time_dim=self.T_dim,
                             context_dim=self.d_ctx, is_cross_attn=True)

        q = torch.randn(self.B, self.seq_img, self.D)
        k = torch.randn(self.B, self.seq_txt, self.d_ctx)
        t1 = torch.randn(self.B, self.T_dim)
        t2 = torch.randn(self.B, self.T_dim)

        # Even though K/V are static (text), Q is dynamic (image + time), so output changes
        out1 = model(q, 1.0, 1.0, context=k, time_embedding=t1)
        out2 = model(q, 1.0, 1.0, context=k, time_embedding=t2)
        self.assertFalse(torch.allclose(out1, out2), "CA Time Injection failed on Q")

    def test_ca_init_logic(self):
        """CA: Ensure self.k_time and rel_pos_bias are None in Cross Mode."""
        model = m.MP_Attention(self.H, self.D, self.seq_img, time_dim=self.T_dim,
                             context_dim=self.d_ctx, is_cross_attn=True)
        self.assertFalse(hasattr(model, 'rel_pos_bias'))
        self.assertFalse(hasattr(model, 'k_time'))
        self.assertFalse(hasattr(model, 'v_time'))
        self.assertTrue(hasattr(model, 'q_time'))  # Q time must exist

if __name__ == '__main__':
    unittest.main()