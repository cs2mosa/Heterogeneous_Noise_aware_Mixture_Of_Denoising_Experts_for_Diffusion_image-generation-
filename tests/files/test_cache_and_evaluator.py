import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from models.model_config_v3 import preconditioned_HDMOEM_v3
from Utils.configs import loss_configs, mask_configs, model_configs, optim_configs, zeta_configs
from Utils.embedding_cache import (
    CachedEmbeddingDataset,
    build_or_load_embedding_cache,
    compute_stats_from_cache,
    compute_whitening_params,
    whiten_posterior_latents,
)
from Utils.evaluator import (
    RunningFeatureStats,
    compute_fid_from_sources,
    compute_frechet_distance,
    compute_or_load_real_stats,
    evaluate_model_fid,
    normalize_to_uint8_rgb,
)
from Utils.training import training_HDMOE
from Utils.VAE_CLIP import StabilityVAE


class _SyntheticMultiCaptionDataset(Dataset):
    """Mimics COCO2017CaptionDataset (`self.records` + [-1, 1] pixel_values)."""

    def __init__(self, num_items: int = 10, img_res: int = 16, captions_per_img: int = 3):
        self.img_res = img_res
        self.records = []
        self._images = []
        for idx in range(num_items):
            # Distinct per-image channel intensities so VAE posterior mean encodes item identity
            val = -0.8 + 1.6 * (idx / max(1, num_items - 1))
            img = torch.full((3, img_res, img_res), val, dtype=torch.float32)
            self._images.append(img)
            caps = [f"img_{idx:03d}_cap_{c}" for c in range(captions_per_img)]
            self.records.append((f"coco_{idx:05d}.jpg", caps))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return {
            "pixel_values": self._images[idx],
            "prompts": self.records[idx][1][0],
        }


class _DeterministicStubVAE:
    vae_name = "stub-vae-v1"

    def __init__(self, latent_res: int = 2, raw_mean=(0.5, -0.5, 0.25, 0.0), raw_std=(1.0, 2.0, 1.5, 0.5), final_std=0.8):
        self.latent_res = latent_res
        self.scale, self.bias = compute_whitening_params(raw_mean, raw_std, final_mean=0.0, final_std=final_std)

    def init(self, device):
        pass

    def _run_vae_encoder_on_batch(self, pixel_batch: torch.Tensor):
        b = pixel_batch.shape[0]
        # Encode spatial average RGB into channel-distinct raw posterior mean and fixed positive std
        base = pixel_batch.mean(dim=(2, 3))[:, :1].unsqueeze(-1).unsqueeze(-1)
        mean = base.expand(b, 4, self.latent_res, self.latent_res).clone()
        mean[:, 1] += 1.0
        mean[:, 2] -= 0.5
        std = torch.full_like(mean, 0.25)
        return mean, std

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # Map whitened (B, 4, r, r) cleanly back to uint8 [0, 255] RGB (B, 3, H, W)
        img_res = self.latent_res * 8
        rgb = torch.nn.functional.interpolate(latents[:, :3].float(), size=(img_res, img_res), mode="nearest")
        return ((rgb.clamp(-2.0, 2.0) + 2.0) / 4.0 * 255.0).round().clamp(0, 255).to(torch.uint8)


class _DeterministicStubCLIP:
    embed_name = "stub-clip-v1"

    def __init__(self, seq_len: int = 4, dim: int = 8):
        self.seq_len = seq_len
        self.dim = dim

    def init(self):
        pass

    def encode_text(self, text_list):
        embs = []
        for txt in text_list:
            if txt == "":
                embs.append(torch.zeros((self.seq_len, self.dim), dtype=torch.float32))
            else:
                # Encode deterministic unique float values from string UTF-8 bytes
                raw = [float(b) for b in txt.encode("utf-8")]
                vals = torch.tensor(raw[: self.dim] + [0.0] * max(0, self.dim - len(raw)), dtype=torch.float32)
                t = vals.unsqueeze(0).expand(self.seq_len, self.dim).clone() / 128.0
                embs.append(t)
        return torch.stack(embs, dim=0)


def _tiny_configs(tmpdir: str, total_steps: int = 2) -> tuple[dict, dict, dict, dict, dict]:
    m_cfg = dict(model_configs)
    m_cfg.update({
        "run_name": "unit_cache_test",
        "save_dir": str(Path(tmpdir) / "ckpt"),
        "save_dir_stats": str(Path(tmpdir) / "logs"),
        "device": "cpu",
        "use_amp": False,
        "num_workers": 0,
        "dataset_size": 8,
        "data_img_res": 32,
        "img_resolution": 4,
        "img_channels": 4,
        "internal_channels": 8,
        "time_emb_dim": 16,
        "text_emb_dim": 8,
        "Unet_model_channels": 8,
        "Unet_channel_mult": [1],
        "Unet_num_blocks": 1,
        "Unet_kernel_sizes": [(3, 3), (3, 3)],
        "num_Unet_experts": 2,
        "VIT_num_blocks": 1,
        "VIT_patch_sizes": [1, 2],
        "VIT_num_heads": 2,
        "VIT_emb_size": 8,
        "VIT_num_groups": 2,
        "num_vit_experts": 2,
        "num_experts": 2,
        "pool_size": 4,
        "batch_size": 4,
        "micro_batch_size": 4,
        "grad_accum_steps": 1,
        "total_steps": total_steps,
        "save_interval": 1,
        "log_interval": 1,
        "raw_mean": (0.0, 1.0, -0.5, 0.0),
        "raw_std": (1.0, 1.0, 1.0, 1.0),
        "mean_data": 0.0,
        "sigma_data": 0.8,
        "cfg_drop_prob": 0.5,
    })
    o_cfg = dict(optim_configs)
    o_cfg.update({"lr_schedule": "wsd", "warmup_steps": 1, "LR_DEFAULT": 1e-4, "grad_clip": 10.0})
    l_cfg = dict(loss_configs)
    mk_cfg = dict(mask_configs)
    mk_cfg.update({"unet_attr": [3, 5], "vit_attr": [1, 1], "total_steps": total_steps})
    z_cfg = dict(zeta_configs)
    z_cfg.update({"total_schedule_steps": total_steps})
    return m_cfg, o_cfg, l_cfg, mk_cfg, z_cfg


class TestEmbeddingCacheAndEvaluator(unittest.TestCase):
    def test_cached_vae_and_clip_alignment_and_mismatch_detection(self):
        src_ds = _SyntheticMultiCaptionDataset(num_items=7, img_res=16, captions_per_img=3)
        vae = _DeterministicStubVAE(latent_res=2)
        clip = _DeterministicStubCLIP(seq_len=4, dim=8)

        with tempfile.TemporaryDirectory() as tmp:
            drive_dir = Path(tmp) / "drive_cache"
            local_dir = Path(tmp) / "local_cache"

            cached_ds = build_or_load_embedding_cache(
                cache_dir=drive_dir,
                dataset=src_ds,
                vae=vae,
                clip=clip,
                dataset_name="synthetic_coco",
                split="train",
                img_res=16,
                cache_clip=True,
                max_captions_per_image=3,
                shard_size=3,
                encode_batch_size=2,
                clip_batch_size=4,
                device="cpu",
                use_amp=False,
                local_cache_dir=local_dir,
                random_caption=True,
                seed=7,
            )

            self.assertEqual(len(cached_ds), 7)
            self.assertTrue(cached_ds.has_clip_cache)
            self.assertIsNotNone(cached_ds.null_text_emb)
            self.assertTrue((local_dir / "manifest.json").exists())

            # Verify shuffled DataLoader batches keep (vae_mean, prompt, clip_embs, sample_key) 100% aligned
            loader = DataLoader(cached_ds, batch_size=3, shuffle=True)
            seen_keys = set()
            for batch in loader:
                for i, prompt in enumerate(batch["prompts"]):
                    key = batch["sample_key"][i]
                    seen_keys.add(key)
                    idx_from_key = int(key.split(":")[0])
                    exp_mean, exp_std = vae._run_vae_encoder_on_batch(src_ds[idx_from_key]["pixel_values"].unsqueeze(0))
                    torch.testing.assert_close(batch["vae_mean"][i].float(), exp_mean[0], atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(batch["vae_std"][i].float(), exp_std[0], atol=2e-3, rtol=2e-3)
                    exp_clip = clip.encode_text([prompt])[0]
                    torch.testing.assert_close(batch["clip_embs"][i].float(), exp_clip, atol=2e-3, rtol=2e-3)
                    self.assertIn(prompt, src_ds.records[idx_from_key][1])
            self.assertEqual(len(seen_keys), 7)

            # Verify analytical stats match manual mean & E[mu^2 + sigma^2]
            raw_mean, raw_std = compute_stats_from_cache(local_dir)
            self.assertAlmostEqual(raw_mean[0], 0.0, places=2)
            self.assertAlmostEqual(raw_mean[1], 1.0, places=2)
            self.assertEqual(len(raw_std), 4)

            # Verify stochastic vs deterministic whitening parity
            scale, bias = compute_whitening_params(raw_mean, raw_std, final_mean=0.0, final_std=0.8)
            sample0 = cached_ds[0]
            det1 = whiten_posterior_latents(sample0["vae_mean"].unsqueeze(0), sample0["vae_std"].unsqueeze(0), scale, bias, deterministic=True)
            det2 = whiten_posterior_latents(sample0["vae_mean"].unsqueeze(0), sample0["vae_std"].unsqueeze(0), scale, bias, deterministic=True)
            torch.testing.assert_close(det1, det2)
            sto1 = whiten_posterior_latents(sample0["vae_mean"].unsqueeze(0), sample0["vae_std"].unsqueeze(0), scale, bias, deterministic=False)
            sto2 = whiten_posterior_latents(sample0["vae_mean"].unsqueeze(0), sample0["vae_std"].unsqueeze(0), scale, bias, deterministic=False)
            self.assertFalse(torch.allclose(sto1, sto2))

            # Verify StabilityVAE.scale/bias formula matches compute_whitening_params
            ref_vae = StabilityVAE(raw_mean=raw_mean, raw_std=raw_std, final_mean=0.0, final_std=0.8)
            torch.testing.assert_close(scale, ref_vae.scale)
            torch.testing.assert_close(bias, ref_vae.bias)

            # Verify mismatch rejections (resolution / dataset size / vae_id / clip requirement)
            with self.assertRaises(ValueError):
                CachedEmbeddingDataset(
                    local_dir,
                    expected_signature={
                        "dataset_name": "synthetic_coco", "split": "train",
                        "num_images": 7, "img_res": 512, "vae_id": "stub-vae-v1",
                    },
                )
            with self.assertRaises(ValueError):
                CachedEmbeddingDataset(
                    local_dir,
                    expected_signature={
                        "dataset_name": "synthetic_coco", "split": "train",
                        "num_images": 8, "img_res": 16, "vae_id": "stub-vae-v1",
                    },
                )

    def test_cache_resume_and_drift_detection(self):
        src_ds = _SyntheticMultiCaptionDataset(num_items=6, img_res=16, captions_per_img=1)
        vae = _DeterministicStubVAE(latent_res=2)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vae_only"
            build_or_load_embedding_cache(
                cache_dir=cache_dir,
                dataset=src_ds,
                vae=vae,
                clip=None,
                dataset_name="vae_only_ds",
                split="train",
                img_res=16,
                cache_clip=False,
                shard_size=2,
                device="cpu",
                use_amp=False,
            )

            # Simulate interrupted run at shard 2 by deleting last shard and marking complete=False
            manifest_path = cache_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["complete"] = False
            manifest["shards"] = manifest["shards"][:-1]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (cache_dir / "shard_00002.pt").unlink()

            # Drift in underlying dataset captions must raise ValueError instead of silent corruption
            drifted_ds = _SyntheticMultiCaptionDataset(num_items=6, img_res=16, captions_per_img=1)
            drifted_ds.records[0] = ("coco_00000.jpg", ["tampered caption"])
            with self.assertRaises(ValueError):
                build_or_load_embedding_cache(
                    cache_dir=cache_dir, dataset=drifted_ds, vae=vae, clip=None,
                    dataset_name="vae_only_ds", split="train", img_res=16, cache_clip=False, shard_size=2, device="cpu", use_amp=False,
                )

            # Matching source resumes cleanly and completes
            resumed = build_or_load_embedding_cache(
                cache_dir=cache_dir, dataset=src_ds, vae=vae, clip=None,
                dataset_name="vae_only_ds", split="train", img_res=16, cache_clip=False, shard_size=2,
                device="cpu", use_amp=False, preload_vae_to_ram=True,
            )
            self.assertEqual(len(resumed), 6)
            self.assertFalse(resumed.has_clip_cache)
            with self.assertRaises(ValueError):
                CachedEmbeddingDataset(cache_dir, require_clip_cache=True)

    def test_training_hdmoe_with_cached_vae_and_clip(self):
        src_ds = _SyntheticMultiCaptionDataset(num_items=8, img_res=32, captions_per_img=2)
        vae = _DeterministicStubVAE(latent_res=4)
        clip = _DeterministicStubCLIP(seq_len=4, dim=8)

        with tempfile.TemporaryDirectory() as tmp:
            full_cache = build_or_load_embedding_cache(
                cache_dir=Path(tmp) / "cache_both",
                dataset=src_ds, vae=vae, clip=clip,
                dataset_name="unit_ds", split="train", img_res=32,
                cache_clip=True, max_captions_per_image=2, shard_size=4, device="cpu", use_amp=False,
            )
            m_cfg, o_cfg, l_cfg, mk_cfg, z_cfg = _tiny_configs(tmp, total_steps=2)
            # Run with BOTH vae=None and clip=None because full_cache supplies both + clip_null_emb
            model, ema, _, _ = training_HDMOE(
                m_cfg, o_cfg, l_cfg, mk_cfg, z_cfg,
                dataset=full_cache,
                val_dataset=full_cache,
                vae=None,
                clip=None,
                val_probe_samples=4,
            )
            self.assertIsNotNone(ema)
            self.assertTrue((Path(tmp) / "ckpt" / "final_model1.pt").exists())

            # Run VAE-only cache with clip online
            vae_only_cache = build_or_load_embedding_cache(
                cache_dir=Path(tmp) / "cache_vae",
                dataset=src_ds, vae=vae, clip=None,
                dataset_name="unit_ds", split="train", img_res=32,
                cache_clip=False, shard_size=4, device="cpu", use_amp=False,
            )
            m_cfg2, o_cfg2, l_cfg2, mk_cfg2, z_cfg2 = _tiny_configs(tmp + "_v2", total_steps=2)
            model2, _, _, _ = training_HDMOE(
                m_cfg2, o_cfg2, l_cfg2, mk_cfg2, z_cfg2,
                dataset=vae_only_cache,
                val_dataset=vae_only_cache,
                vae=None,
                clip=clip,
                val_probe_samples=4,
            )
            self.assertIsNotNone(model2)

    def test_fid_math_stats_caching_and_model_evaluation(self):
        # 1. Exact FID identity & shift sensitivity
        rng = np.random.default_rng(0)
        x = rng.normal(loc=0.0, scale=1.0, size=(128, 32))
        st1 = RunningFeatureStats()
        st1.update(x[:64])
        st1.update(x[64:])
        mu1, sig1 = st1.finalize()
        self.assertAlmostEqual(compute_frechet_distance(mu1, sig1, mu1, sig1), 0.0, places=5)

        y = rng.normal(loc=1.5, scale=1.0, size=(128, 32))
        st2 = RunningFeatureStats()
        st2.update(y)
        mu2, sig2 = st2.finalize()
        fid_shifted = compute_frechet_distance(mu1, sig1, mu2, sig2)
        self.assertGreater(fid_shifted, 30.0)

        # 2. normalize_to_uint8_rgb checks
        m11 = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1)
        out_m11 = normalize_to_uint8_rgb(m11, pixel_range="auto")
        self.assertEqual(out_m11.dtype, torch.uint8)
        self.assertEqual(out_m11.flatten().tolist(), [0, 128, 255])

        # 3. Real stats caching & mismatch guard + directory comparison
        def _fake_extractor(uint8_imgs: torch.Tensor) -> torch.Tensor:
            f = uint8_imgs.float().mean(dim=(2, 3))
            extra = uint8_imgs[:, :, :8, :].float().mean(dim=(2, 3))
            return torch.cat([f, extra], dim=1)

        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp) / "real"
            gen_dir = Path(tmp) / "gen"
            real_dir.mkdir()
            gen_dir.mkdir()
            for i in range(6):
                arr = np.full((16, 16, 3), 40 + i * 20, dtype=np.uint8)
                Image.fromarray(arr).save(real_dir / f"{i:03d}.png")
                Image.fromarray(np.clip(arr + 15, 0, 255).astype(np.uint8)).save(gen_dir / f"{i:03d}.png")

            same = compute_fid_from_sources(
                real_dir, real_dir, device="cpu", batch_size=3, img_res=16, feature_extractor=_fake_extractor
            )
            self.assertAlmostEqual(same["fid"], 0.0, places=4)
            diff = compute_fid_from_sources(
                real_dir, gen_dir, device="cpu", batch_size=3, img_res=16, feature_extractor=_fake_extractor
            )
            self.assertGreater(diff["fid"], 1.0)

            stats_file = Path(tmp) / "real_stats.pt"
            mu_r, sig_r, cnt = compute_or_load_real_stats(
                real_source=real_dir, feature_extractor=_fake_extractor, backend_name="custom",
                dataset_name="toy_coco", img_res=16, stats_cache_path=stats_file, batch_size=3,
            )
            self.assertEqual(cnt, 6)
            mu_r2, sig_r2, cnt2 = compute_or_load_real_stats(
                real_source=None, feature_extractor=_fake_extractor, backend_name="custom",
                dataset_name="toy_coco", img_res=16, stats_cache_path=stats_file,
            )
            self.assertEqual(cnt2, 6)
            np.testing.assert_allclose(mu_r, mu_r2)
            with self.assertRaises(ValueError):
                compute_or_load_real_stats(
                    real_source=None, feature_extractor=_fake_extractor, backend_name="custom",
                    dataset_name="wrong_dataset", img_res=16, stats_cache_path=stats_file,
                )

            # 4. End-to-end evaluate_model_fid on tiny HDMOEM_v3 + EDM_Sampler + PNG saving + CLIPScore
            m_cfg, _, _, mk_cfg, _ = _tiny_configs(tmp, total_steps=4)
            eval_model = preconditioned_HDMOEM_v3(
                IN_in_channels=m_cfg["img_channels"],
                IN_img_resolution=m_cfg["img_resolution"],
                internal_channels=m_cfg["internal_channels"],
                time_emb_dim=m_cfg["time_emb_dim"],
                text_emb_dim=m_cfg["text_emb_dim"],
                num_experts=m_cfg["num_experts"],
                top_k=m_cfg["top_k"],
                Fourier_bandwidth=m_cfg["fourier_bandwidth"],
                VIT_num_blocks=m_cfg["VIT_num_blocks"],
                VIT_patch_sizes=m_cfg["VIT_patch_sizes"],
                VIT_num_groups=m_cfg["VIT_num_groups"],
                VIT_num_heads=m_cfg["VIT_num_heads"],
                VIT_emb_size=m_cfg["VIT_emb_size"],
                VIT_emb_balance=m_cfg["VIT_emb_balances"],
                VIT_pos_emb_balance=m_cfg["VIT_pos_emb_balances"],
                Unet_num_blocks=m_cfg["Unet_num_blocks"],
                Unet_channel_mult=m_cfg["Unet_channel_mult"],
                Unet_channel_mult_emb=m_cfg["Unet_channel_mult_emb"],
                Unet_kernel_sizes=m_cfg["Unet_kernel_sizes"],
                Unet_model_channels=m_cfg["Unet_model_channels"],
                Unet_label_balance=m_cfg["Unet_label_balance"],
                Unet_concat_balance=m_cfg["Unet_concat_balances"],
                sigma_data=m_cfg["sigma_data"],
                log_var_channels=m_cfg["log_var_channels"],
                transition_sigma=m_cfg["transition_sigma"],
                transition_sharpness=m_cfg["transition_sharpness"],
                pool_size=m_cfg["pool_size"],
                attn_pool_balance=m_cfg["attn_pool_balance"],
                p_mean=mk_cfg["p_mean"],
                p_std=mk_cfg["p_std"],
                sigma_min=m_cfg["sigma_min"],
                sigma_max=m_cfg["sigma_max"],
                alpha_ext=m_cfg["extreme_prob"],
            )
            val_ds = _SyntheticMultiCaptionDataset(num_items=6, img_res=32, captions_per_img=1)
            stub_vae = _DeterministicStubVAE(latent_res=4)
            stub_clip = _DeterministicStubCLIP(seq_len=4, dim=8)

            def _stub_clip_scorer(imgs: torch.Tensor, prompts: list[str]) -> torch.Tensor:
                return torch.full((len(prompts),), 27.5)

            gen_out_dir = Path(tmp) / "eval_pngs"
            report = evaluate_model_fid(
                model=eval_model,
                vae=stub_vae,
                clip=stub_clip,
                val_dataset=val_ds,
                model_config=m_cfg,
                mask_config=mk_cfg,
                dataset_name="toy_val",
                guidance_scales=(1.0, 2.0),
                num_eval_samples=4,
                num_solve_steps=2,
                batch_size=2,
                seed=99,
                save_generated_dir=gen_out_dir,
                stats_cache_path=Path(tmp) / "eval_real_stats.pt",
                device="cpu",
                use_amp=False,
                compute_clip_score=True,
                feature_extractor=_fake_extractor,
                clip_scorer=_stub_clip_scorer,
            )
            self.assertIn(1.0, report["results"])
            self.assertIn(2.0, report["results"])
            self.assertTrue(np.isfinite(report["results"][1.0]["fid"]))
            self.assertAlmostEqual(report["results"][2.0]["clip_score"], 27.5, places=4)
            self.assertEqual(len(list((gen_out_dir / "cfg_1.00").glob("*.png"))), 4)
            self.assertEqual(len(list((gen_out_dir / "cfg_2.00").glob("*.png"))), 4)

    def test_colab_notebook_cells_parse_and_reference_cache_and_evaluator(self):
        nb_path = Path(__file__).resolve().parents[2] / "colab_text_guided_training.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        all_code = []
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                ast.parse(src, filename=f"colab_cell_{idx}.py")
                all_code.append(src)
        combined = "\n".join(all_code)
        self.assertIn("build_or_load_embedding_cache", combined)
        self.assertIn("evaluate_model_fid", combined)
        self.assertIn("clean-fid", combined)


if __name__ == "__main__":
    unittest.main()
