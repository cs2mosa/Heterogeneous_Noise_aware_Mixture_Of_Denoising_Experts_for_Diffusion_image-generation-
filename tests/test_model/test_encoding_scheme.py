import torch
import unittest
from models import model_internals as m
import torch.nn as nn


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8

        # --- Config for MP_Pos_encoding_2D ---
        self.pos_emb_dim = 128
        self.pos_seq_ln = 64  # 8x8 perfect square
        self.pos_model = m.MP_Pos_encoding_2D(
            emb_dim=self.pos_emb_dim,
            seq_ln=self.pos_seq_ln
        ).to(self.device)

        # --- Config for MP_Fourier ---
        self.four_channels = 256
        self.fourier_model = m.MP_Fourier(
            num_channels=self.four_channels,
            bandwidth=1.5
        ).to(self.device)

    # ==========================================
    # TESTS FOR MP_POS_ENCODING_2D
    # ==========================================

    def test_pos_output_shape(self):
        """[MP_Pos_encoding_2D] Output shape match"""
        x = torch.randn(self.batch_size, self.pos_seq_ln, self.pos_emb_dim, device=self.device)
        out = self.pos_model(x)
        self.assertEqual(out.shape, (self.batch_size, self.pos_seq_ln, self.pos_emb_dim))

    def test_pos_buffer_persistence(self):
        """[MP_Pos_encoding_2D] pos_emb should be a buffer, not a parameter"""
        self.assertIn('pos_emb', dict(self.pos_model.named_buffers()))
        self.assertEqual(len(list(self.pos_model.parameters())), 0)

    def test_pos_preserves_variance(self):
        """[MP_Pos_encoding_2D] Blending with position encoding should preserve unit variance"""
        x = torch.randn(self.batch_size, self.pos_seq_ln, self.pos_emb_dim, device=self.device)
        out = self.pos_model(x, t=0.5)
        self.assertAlmostEqual(out.std().item(), 1.0, delta=0.15)

    # ==========================================
    # TESTS FOR MP_FOURIER
    # ==========================================

    def test_four_output_shape(self):
        """[MP_Fourier] Output shape match"""
        x = torch.randn(self.batch_size).to(self.device)
        out = self.fourier_model(x)
        self.assertEqual(out.shape, (self.batch_size, self.four_channels))

    def test_four_determinism(self):
        """[MP_Fourier] Deterministic behavior (Fixed random basis)"""
        x = torch.randn(self.batch_size).to(self.device)
        out1 = self.fourier_model(x)
        out2 = self.fourier_model(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_four_gradient_input(self):
        """[MP_Fourier] Differentiable w.r.t input X"""
        x = torch.randn(self.batch_size, device=self.device, requires_grad=True)
        out = self.fourier_model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_four_dtype_support(self):
        """[MP_Fourier] Supports FP16/Half precision"""
        if self.device.type == 'cpu':
            return

        x = torch.randn(self.batch_size, device=self.device, dtype=torch.float16)
        out = self.fourier_model(x)
        self.assertEqual(out.dtype, torch.float16)

    def test_four_input_constraint(self):
        """[MP_Fourier] Should reject 2D inputs (requires 1D for ger)"""
        x_2d = torch.randn(self.batch_size, 1).to(self.device)
        with self.assertRaises(RuntimeError):
            self.fourier_model(x_2d)


if __name__ == '__main__':
    unittest.main()
