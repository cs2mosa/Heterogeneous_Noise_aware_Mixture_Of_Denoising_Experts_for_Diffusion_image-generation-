import torch
import models.model_components as m
import unittest


class TestUnetExpert(unittest.TestCase):

    def setUp(self):
        # Default common parameters for tests
        self.img_res = 64
        self.img_ch = 3
        self.time_dim = 128
        self.text_dim = 256
        self.batch_size = 2
        self.common_args = {
            'img_resolution': self.img_res,
            'img_channels': self.img_ch,
            'time_emb_dim': self.time_dim,
            'text_emb_dim': self.text_dim,
            'channel_mult': [1, 2],  # Keep small for speed
            'model_channels': 32,
            'num_blocks': 1,
            'kernel_size': (5,5)
        }

    def test_01_standard_forward(self):
        """Standard Forward Pass: Checks input/output shapes match."""
        model = m.Unet_expert(**self.common_args)
        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res)
        t = torch.randn(self.batch_size, self.time_dim)
        txt = torch.randn(self.batch_size, self.text_dim)

        out = model(x, t, txt)
        self.assertEqual(out.shape, x.shape, "Output shape must match input shape")

    def test_02_no_text_embedding(self):
        """Corner Case: Unconditional generation (text_dim=0, text_emb=None)."""
        args = self.common_args.copy()
        args['text_emb_dim'] = 0
        model = m.Unet_expert(**args)

        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res)
        t = torch.randn(self.batch_size, self.time_dim)

        # Pass None for text_emb
        out = model(x, t, None)
        self.assertEqual(out.shape, x.shape)
        # Ensure bottleneck_attn is actually None
        self.assertIsNone(model.bottleneck_attn)

    def test_03_deep_architecture(self):
        """Performance Case: deeper network (3 levels). Checks tensor contraction logic."""
        args = self.common_args.copy()
        args['channel_mult'] = [1, 2, 4]  # 3 levels
        model = m.Unet_expert(**args)

        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res)
        t = torch.randn(self.batch_size, self.time_dim)
        txt = torch.randn(self.batch_size, self.text_dim)

        out = model(x, t, txt)
        self.assertEqual(out.shape, x.shape)

    def test_04_odd_kernel_size(self):
        """Corner Case: Using 5x5 kernels instead of 3x3. Checks padding/shape logic."""
        args = self.common_args.copy()
        args['kernel_size'] = (5,5)
        model = m.Unet_expert(**args)

        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res)
        t = torch.randn(self.batch_size, self.time_dim)
        txt = torch.randn(self.batch_size, self.text_dim)

        out = model(x, t, txt)
        self.assertEqual(out.shape, x.shape)

    def test_05_gradient_propagation(self):
        """Performance Case: Verifies that gradients flow back to input and params."""
        model = m.Unet_expert(**self.common_args)
        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res, requires_grad=True)
        t = torch.randn(self.batch_size, self.time_dim)
        txt = torch.randn(self.batch_size, self.text_dim)

        out = model(x, t, txt)
        loss = out.mean()
        loss.backward()

        # Check input gradient
        self.assertIsNotNone(x.grad, "Input image should have gradients")
        # Check parameter gradient (e.g., out_conv weights)
        self.assertIsNotNone(model.out_conv.weights.grad, "Model weights should have gradients")

    def test_06_non_square_aspect_ratio(self):
        """Corner Case: Rectangular images (64x128). Checks up/downsampling logic without spatial cross-attn."""
        args = self.common_args.copy()
        args['text_emb_dim'] = 0
        model = m.Unet_expert(**args)
        h, w = 64, 128
        x = torch.randn(self.batch_size, self.img_ch, h, w)
        t = torch.randn(self.batch_size, self.time_dim)

        out = model(x, t, None)
        self.assertEqual(out.shape, (self.batch_size, self.img_ch, h, w))

    def test_07_fp16_compatibility(self):
        """Performance Case: Run in Half Precision (simulating production training)."""
        model = m.Unet_expert(**self.common_args).half()
        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res).half()
        t = torch.randn(self.batch_size, self.time_dim).half()
        txt = torch.randn(self.batch_size, self.text_dim).half()

        out = model(x, t, txt)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, x.shape)

    def test_08_embedding_override(self):
        """Corner Case: Explicitly setting channel_mult_emb."""
        args = self.common_args.copy()
        # Normally max(32, 64) = 64. Forcing it to 10 (32*10 = 320)
        args['channel_mult_emb'] = 10
        model = m.Unet_expert(**args)

        # Check if embedding size logic works
        expected_emb_size = 32 * 10
        self.assertEqual(model.emb_size, expected_emb_size)

        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res)
        t = torch.randn(self.batch_size, self.time_dim)
        txt = torch.randn(self.batch_size, self.text_dim)
        out = model(x, t, txt)
        self.assertEqual(out.shape, x.shape)

    def test_09_3d_text_embedding(self):
        """Verify that Unet_expert supports 3D sequence text embeddings (Batch, Seq, Dim)."""
        model = m.Unet_expert(**self.common_args)
        x = torch.randn(self.batch_size, self.img_ch, self.img_res, self.img_res)
        t = torch.randn(self.batch_size, self.time_dim)
        txt = torch.randn(self.batch_size, 77, self.text_dim)

        out = model(x, t, txt)
        self.assertEqual(out.shape, x.shape)


if __name__ == '__main__':
    unittest.main()