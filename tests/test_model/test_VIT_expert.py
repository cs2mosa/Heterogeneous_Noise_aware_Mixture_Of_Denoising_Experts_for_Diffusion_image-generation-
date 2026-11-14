import math
import torch
import models.model_components as m
import unittest

class TestVitExpert(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Standard Configuration
        self.B = 2
        self.C = 4
        self.H = 32
        self.W = 32
        self.Patch = 6

        # Derived Config (with padding to patch multiples)
        self.H_latent = math.ceil(self.H / self.Patch)
        self.W_latent = math.ceil(self.W / self.Patch)
        self.SeqLen = self.H_latent * self.W_latent
        self.EmbDim = 64
        self.TimeDim = 64

        self.model = m.Vit_expert(
            num_heads=4, num_groups=4, in_channels=self.C,
            seq_ln=self.SeqLen, emb_dim=self.EmbDim, num_blocks=2,
            patch_size=self.Patch, time_dim=self.TimeDim, text_dim=self.TimeDim
        ).to(self.device)

    def test_01_standard_forward_shape(self):
        """Test 1: Input Shape == Output Shape (Isotropic Autoencoder logic)"""
        x = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
        t = torch.randn(self.B, self.TimeDim).to(self.device)

        out = self.model(x, time_emb=t)
        self.assertEqual(out.shape, x.shape, "Output shape must restore Input shape")

    def test_02_gradient_flow_full(self):
        """Test 2: Do gradients reach Input, Time, Text, and Weights?"""
        # DiffiT blocks zero-initialize tmsa_gate for identity residual init; open the gate for grad verification.
        for block in self.model.diffit:
            with torch.no_grad():
                block.tmsa_gate.fill_(1.0)

        x = torch.randn(self.B, self.C, self.H, self.W, requires_grad=True, device=self.device)
        t = torch.randn(self.B, self.TimeDim, requires_grad=True, device=self.device)
        txt = torch.randn(self.B, self.TimeDim, requires_grad=True, device=self.device)

        out = self.model(x, time_emb=t, text_emb=txt)
        loss = out.mean()
        loss.backward()

        self.assertGreater(x.grad.abs().sum().item(), 1e-6, "Input x grad blocked")
        self.assertGreater(t.grad.abs().sum().item(), 1e-6, "Time grad blocked")
        self.assertGreater(txt.grad.abs().sum().item(), 1e-6, "Text grad blocked")
        self.assertGreater(self.model.patch.weights.grad.abs().sum().item(), 1e-6, "Patch grad blocked")

    def test_03_unconditional_safety(self):
        """Test 3: Does it crash if text_emb is None? (Validating conditional logic)"""
        x = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
        t = torch.randn(self.B, self.TimeDim).to(self.device)

        try:
            out = self.model(x, time_emb=t, text_emb=None)
        except AttributeError as e:
            self.fail(f"Unconditional forward pass crashed: {e}")
        except Exception as e:
            self.fail(f"Unconditional forward pass crashed with generic error: {e}")

    def test_04_sequence_length_assertion(self):
        """Test 4: Does it catch dimension mismatches between Image and SeqLen?"""
        # Initialize model expecting 32x32 image (SeqLen=36)
        # Pass 16x16 image (SeqLen=9 for patch=6: ceil(16/6)=3, 3*3=9) -> Should Fail
        x_wrong = torch.randn(self.B, self.C, 16, 16).to(self.device)
        t = torch.randn(self.B, self.TimeDim).to(self.device)

        with self.assertRaises(AssertionError) as context:
            self.model(x_wrong, time_emb=t)
        self.assertIn("Sequence length mismatch", str(context.exception))

    def test_05_patch_size_variations(self):
        """Test 5: Does the Patchify/PixelShuffle logic hold for different patch sizes?"""
        for p in [1, 4]:
            h_latent = self.H // p
            seq_len = h_latent ** 2

            m_patch = m.Vit_expert(
                num_heads=4, num_groups=4, in_channels=self.C,
                seq_ln=seq_len, emb_dim=32, num_blocks=1,
                patch_size=p, time_dim=32
            ).to(self.device)

            x = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
            t = torch.randn(self.B, 32).to(self.device)

            out = m_patch(x, t)
            self.assertEqual(out.shape, x.shape, f"Failed for patch size {p}")

    def test_06_batch_size_one(self):
        """Test 6: Corner case B=1. Reshapes often break here."""
        x = torch.randn(1, self.C, self.H, self.W).to(self.device)
        t = torch.randn(1, self.TimeDim).to(self.device)
        out = self.model(x, time_emb=t)
        self.assertEqual(out.shape, (1, self.C, self.H, self.W))

    def test_07_text_projection_mismatch(self):
        """Test 7: Ensure cross-attn works when Text Dim != Time Dim"""
        txt_dim = 128
        time_dim = 64
        patch_size = 2
        seq_len = (self.H // patch_size) * (self.W // patch_size)

        m_proj = m.Vit_expert(
            num_heads=4, num_groups=4, in_channels=self.C,
            seq_ln=seq_len, emb_dim=64, num_blocks=1,
            patch_size=patch_size, time_dim=time_dim, text_dim=txt_dim
        ).to(self.device)

        x = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
        t = torch.randn(self.B, time_dim).to(self.device)
        txt = torch.randn(self.B, txt_dim).to(self.device)

        try:
            m_proj(x, time_emb=t, text_emb=txt)
        except RuntimeError as e:
            self.fail(f"Projection layer failed dimension matching: {e}")

    def test_08_embedding_balance_logic(self):
        """Test 8: If attn_balance=0.0, changing Text should not affect output."""
        patch_size = 2
        seq_len = (self.H // patch_size) * (self.W // patch_size)
        m_bias = m.Vit_expert(
            num_heads=4, num_groups=4, in_channels=self.C,
            seq_ln=seq_len, emb_dim=64, num_blocks=1,
            patch_size=patch_size, time_dim=32, text_dim=32,
            attn_balance=0.0
        ).to(self.device)

        x = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
        t = torch.randn(self.B, 32).to(self.device)

        txt1 = torch.randn(self.B, 32).to(self.device)
        txt2 = torch.randn(self.B, 32).to(self.device)  # Different text

        m_bias.eval()
        out1 = m_bias(x, time_emb=t, text_emb=txt1)
        out2 = m_bias(x, time_emb=t, text_emb=txt2)

        # Outputs should be identical because attn_balance is 0.0
        diff = (out1 - out2).abs().sum().item()
        self.assertLess(diff, 1e-5, "Balance 0.0 should ignore text input")

    def test_09_position_embedding_broadcast(self):
        """Test 9: Verify Pos Embed adds correctly (Broadcasting check)."""
        self.assertEqual(self.model.pos_encoder.pos_emb.shape[0], 1)
        self.assertEqual(self.model.pos_encoder.pos_emb.ndim, 3)

        x = torch.randn(10, self.C, self.H, self.W).to(self.device)  # Batch 10
        t = torch.randn(10, self.TimeDim).to(self.device)
        out = self.model(x, time_emb=t)
        self.assertEqual(out.shape[0], 10)

    def test_10_large_depth_connectivity(self):
        """Test 10: Connectivity check for deeper networks (ensure loop logic holds)."""
        patch_size = 2
        seq_len = (self.H // patch_size) * (self.W // patch_size)
        deep_model = m.Vit_expert(
            num_heads=4, num_groups=4, in_channels=self.C,
            seq_ln=seq_len, emb_dim=64, num_blocks=5,  # 5 Blocks
            patch_size=patch_size, time_dim=32, text_dim=32
        ).to(self.device)

        x = torch.randn(self.B, self.C, self.H, self.W, requires_grad=True, device=self.device)
        t = torch.randn(self.B, 32, requires_grad=True, device=self.device)

        out = deep_model(x, time_emb=t)
        loss = out.mean()
        loss.backward()

        # Check first layer gradients (Patch) to ensure signal traveled all the way back
        self.assertGreater(deep_model.patch.weights.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()