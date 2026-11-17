from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import warnings

import numpy as np
from PIL import Image
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

try:
    from Utils.configs import mask_configs as _default_mask_configs
    from Utils.EDM_sampler import EDM_Sampler
    from Utils.utils import MaskGenerator
    from Utils.VAE_CLIP import CLIP_EMBED, StabilityVAE
except ImportError:
    from configs import mask_configs as _default_mask_configs
    from EDM_sampler import EDM_Sampler
    from utils import MaskGenerator
    from VAE_CLIP import CLIP_EMBED, StabilityVAE


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class RunningFeatureStats:
    """Exact double-precision accumulator for mean and sample covariance of feature vectors."""

    def __init__(self, feature_dim: Optional[int] = None):
        self.count = 0
        self.feature_dim = feature_dim
        self._sum: Optional[np.ndarray] = None
        self._sum_outer: Optional[np.ndarray] = None

    def update(self, features: torch.Tensor | np.ndarray) -> None:
        arr = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(f"Expected non-empty 2D feature array (N, D), got shape {arr.shape}.")
        dim = int(arr.shape[1])
        if self._sum is None:
            self.feature_dim = dim
            self._sum = np.zeros(dim, dtype=np.float64)
            self._sum_outer = np.zeros((dim, dim), dtype=np.float64)
        elif dim != self.feature_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}, got {dim}.")
        self._sum += arr.sum(axis=0)
        self._sum_outer += arr.T @ arr
        self.count += int(arr.shape[0])

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2 or self._sum is None or self._sum_outer is None:
            raise ValueError(f"Need at least 2 samples to compute covariance, got count={self.count}.")
        mu = self._sum / float(self.count)
        n = float(self.count)
        cov = (self._sum_outer - n * np.outer(mu, mu)) / float(self.count - 1)
        cov = 0.5 * (cov + cov.T)
        return mu, cov


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Frechet distance between two multivariate Gaussians N(mu1, sigma1) and N(mu2, sigma2)."""
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vector shape mismatch: {mu1.shape} vs {mu2.shape}.")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance matrix shape mismatch: {sigma1.shape} vs {sigma2.shape}.")

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0.0, atol=1e-3):
            max_imag = float(np.max(np.abs(covmean.imag)))
            raise ValueError(f"Imaginary component {max_imag:.6e} in matrix square root exceeds tolerance.")
        covmean = covmean.real

    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    if fid < 0.0 and fid > -1e-6:
        fid = 0.0
    return fid


def normalize_to_uint8_rgb(images: torch.Tensor, pixel_range: str = "auto") -> torch.Tensor:
    """Convert image batch `(B, 3, H, W)` from uint8 / [-1, 1] / [0, 1] / [0, 255] to uint8 [0, 255]."""
    if images.dtype == torch.uint8:
        return images
    img_f = images.float()
    if pixel_range == "0_255":
        return img_f.round().clamp(0, 255).to(torch.uint8)
    if pixel_range == "minus1_1":
        return ((img_f + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    if pixel_range == "0_1":
        return (img_f * 255.0).round().clamp(0, 255).to(torch.uint8)
    if pixel_range != "auto":
        raise ValueError(f"Unknown pixel_range={pixel_range!r}; expected auto, minus1_1, 0_1, or 0_255.")

    vmax = float(img_f.max().item())
    vmin = float(img_f.min().item())
    if vmax > 1.5:
        return img_f.round().clamp(0, 255).to(torch.uint8)
    if vmin < -0.05:
        return ((img_f + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return (img_f * 255.0).round().clamp(0, 255).to(torch.uint8)


class _TorchvisionInceptionExtractor(nn.Module):
    """Fallback 2048-d Inception-v3 pool3 extractor using torchvision weights."""

    def __init__(self, device: torch.device | str):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        net.fc = nn.Identity()
        net.eval().requires_grad_(False).to(device)
        self.net = net
        self.device = torch.device(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, uint8_images: torch.Tensor) -> torch.Tensor:
        x = uint8_images.to(device=self.device, dtype=torch.float32) / 255.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


def build_inception_feature_extractor(
    backend: str = "auto",
    device: str | torch.device = "cuda",
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """Return `(feature_fn, resolved_backend_name)` accepting uint8 RGB `(B, 3, H, W)` -> `(B, 2048)`."""
    dev = torch.device(device)
    has_cleanfid = importlib.util.find_spec("cleanfid") is not None

    if backend == "auto":
        resolved = "cleanfid" if has_cleanfid else "torchvision"
    elif backend in ("cleanfid", "torchvision"):
        resolved = backend
    else:
        raise ValueError(f"Unsupported fid_backend={backend!r}. Use 'auto', 'cleanfid', or 'torchvision'.")

    if resolved == "cleanfid":
        from cleanfid.features import build_feature_extractor

        feat_model = build_feature_extractor(mode="clean", device=dev)

        @torch.no_grad()
        def _cleanfid_extract(uint8_images: torch.Tensor) -> torch.Tensor:
            # cleanfid expects uint8 or float [0, 255] NCHW resized carefully to 299x299;
            # cleanfid feature model handles Tensor [0, 255] directly via forward() or get_batch_features.
            x = uint8_images.to(device=dev, dtype=torch.float32)
            if x.shape[-2:] != (299, 299):
                x = F.interpolate(x, size=(299, 299), mode="bicubic", align_corners=False).clamp(0.0, 255.0)
            return feat_model(x)

        return _cleanfid_extract, "cleanfid"

    if backend == "auto":
        warnings.warn(
            "`clean-fid` is not installed; falling back to torchvision Inception-v3 weights. "
            "Install `clean-fid` for standard benchmark comparability.",
            UserWarning,
            stacklevel=2,
        )
    extractor = _TorchvisionInceptionExtractor(device=dev)
    return extractor, "torchvision"


class ImageFolderFlatDataset(Dataset):
    """Recursive flat dataset reading images from a directory as uint8 `(3, H, W)` tensors."""

    def __init__(self, root_dir: str | Path, img_res: Optional[int] = None, max_samples: Optional[int] = None):
        self.root_dir = Path(root_dir)
        files = sorted(p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
        if max_samples is not None:
            files = files[: int(max_samples)]
        if not files:
            raise FileNotFoundError(f"No images found in {self.root_dir}.")
        self.files = files
        tf_ops = []
        if img_res is not None:
            tf_ops.extend([
                transforms.Resize(img_res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_res),
            ])
        tf_ops.append(transforms.PILToTensor())
        self.transform = transforms.Compose(tf_ops)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with Image.open(self.files[idx]) as im:
            tensor = self.transform(im.convert("RGB"))
        return {"pixel_values": tensor}


def _extract_images_from_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        return batch["pixel_values"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    raise TypeError(f"Unsupported image batch type for FID extraction: {type(batch).__name__}")


def _extract_prompts_from_dataset(
    dataset: Dataset,
    indices: Sequence[int],
) -> list[str]:
    records = getattr(dataset, "records", None)
    if records is not None:
        prompts = []
        for idx in indices:
            _, raw_caps = records[idx][0], records[idx][1]
            if isinstance(raw_caps, (list, tuple)):
                prompts.append(str(raw_caps[0]))
            else:
                prompts.append(str(raw_caps))
        return prompts

    prompts = []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, dict):
            p = item.get("prompts", item.get("prompt", item.get("caption", "")))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            p = item[1]
        else:
            p = ""
        prompts.append(str(p[0] if isinstance(p, (list, tuple)) else p))
    return prompts


@torch.no_grad()
def compute_stats_from_image_source(
    source: Dataset | torch.Tensor | str | Path,
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int = 64,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    img_res: Optional[int] = None,
    pixel_range: str = "auto",
    desc: str = "Extracting Inception features",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute `(mu, sigma, count)` over an image directory, tensor batch, or Dataset."""
    stats = RunningFeatureStats()

    if isinstance(source, (str, Path)):
        ds: Dataset = ImageFolderFlatDataset(source, img_res=img_res, max_samples=max_samples)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        iterator = dl
        eff_range = "0_255"
    elif isinstance(source, torch.Tensor):
        tens = source[:max_samples] if max_samples is not None else source
        iterator = tens.split(batch_size)
        eff_range = pixel_range
    elif isinstance(source, Dataset):
        n_total = len(source) if max_samples is None else min(len(source), int(max_samples))
        sub = Subset(source, range(n_total)) if n_total < len(source) else source
        iterator = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        eff_range = pixel_range
    else:
        raise TypeError(f"Unsupported image source type: {type(source).__name__}")

    for batch in tqdm(iterator, desc=desc, leave=False):
        imgs = normalize_to_uint8_rgb(_extract_images_from_batch(batch), pixel_range=eff_range)
        feats = feature_extractor(imgs)
        stats.update(feats)

    mu, sigma = stats.finalize()
    return mu, sigma, stats.count


def compute_or_load_real_stats(
    real_source: Optional[Dataset | torch.Tensor | str | Path],
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    backend_name: str,
    dataset_name: str = "custom",
    img_res: int = 256,
    max_samples: Optional[int] = None,
    stats_cache_path: Optional[str | Path] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pixel_range: str = "auto",
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load validated real Inception statistics from cache or compute & persist them."""
    cache_file = Path(stats_cache_path) if stats_cache_path is not None else None
    if cache_file is not None and cache_file.exists() and not overwrite:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        mismatches = []
        if payload.get("backend") != backend_name:
            mismatches.append(f"backend: {payload.get('backend')!r} vs {backend_name!r}")
        if payload.get("dataset_name") != dataset_name:
            mismatches.append(f"dataset_name: {payload.get('dataset_name')!r} vs {dataset_name!r}")
        if int(payload.get("img_res", -1)) != int(img_res):
            mismatches.append(f"img_res: {payload.get('img_res')} vs {img_res}")
        if max_samples is not None and int(payload.get("count", -1)) < int(max_samples):
            mismatches.append(f"count: {payload.get('count')} < {max_samples}")
        if mismatches:
            raise ValueError("Cached real FID stats mismatch: " + "; ".join(mismatches))
        return np.asarray(payload["mu"], dtype=np.float64), np.asarray(payload["sigma"], dtype=np.float64), int(payload["count"])

    if real_source is None:
        raise ValueError("Either an existing `stats_cache_path` or a `real_source` dataset/directory is required.")

    mu, sigma, count = compute_stats_from_image_source(
        real_source,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        img_res=img_res,
        pixel_range=pixel_range,
        desc=f"Real Inception stats ({dataset_name})",
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_file.with_name(cache_file.name + ".tmp")
        torch.save(
            {
                "mu": mu,
                "sigma": sigma,
                "count": int(count),
                "backend": backend_name,
                "dataset_name": dataset_name,
                "img_res": int(img_res),
            },
            tmp_path,
        )
        tmp_path.replace(cache_file)
    return mu, sigma, count


def compute_fid_from_sources(
    real_source: Dataset | torch.Tensor | str | Path,
    gen_source: Dataset | torch.Tensor | str | Path,
    fid_backend: str = "auto",
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    img_res: Optional[int] = 256,
    real_pixel_range: str = "auto",
    gen_pixel_range: str = "auto",
    feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> dict[str, Any]:
    """Compute FID directly between any two image sources (folders, datasets, or tensors)."""
    if feature_extractor is None:
        extractor, resolved_backend = build_inception_feature_extractor(backend=fid_backend, device=device)
    else:
        extractor, resolved_backend = feature_extractor, "custom"

    mu_r, sig_r, n_r = compute_stats_from_image_source(
        real_source,
        extractor,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        img_res=img_res,
        pixel_range=real_pixel_range,
        desc="Real Inception features",
    )
    mu_g, sig_g, n_g = compute_stats_from_image_source(
        gen_source,
        extractor,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        img_res=img_res,
        pixel_range=gen_pixel_range,
        desc="Generated Inception features",
    )
    fid = compute_frechet_distance(mu_r, sig_r, mu_g, sig_g)
    return {"fid": fid, "backend": resolved_backend, "num_real": n_r, "num_generated": n_g}


def _resolve_clip_score_scorer(device: torch.device, clip_model_name: str = "openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    @torch.no_grad()
    def _score_batch(uint8_images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        pil_imgs = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in uint8_images]
        inputs = processor(text=prompts, images=pil_imgs, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        img_emb = F.normalize(outputs.image_embeds, dim=-1)
        txt_emb = F.normalize(outputs.text_embeds, dim=-1)
        return (img_emb * txt_emb).sum(dim=-1) * 100.0

    return _score_batch


@torch.no_grad()
def evaluate_model_fid(
    model: nn.Module,
    vae: StabilityVAE,
    clip: CLIP_EMBED,
    val_dataset: Dataset,
    model_config: dict[str, Any],
    mask_config: dict[str, Any],
    real_dataset: Optional[Dataset | str | Path] = None,
    dataset_name: str = "custom",
    guidance_scales: Sequence[float] = (1.0, 3.0),
    num_eval_samples: int = 5000,
    num_solve_steps: int = 32,
    batch_size: int = 32,
    seed: int = 1234,
    save_generated_dir: Optional[str | Path] = None,
    stats_cache_path: Optional[str | Path] = None,
    fid_backend: str = "auto",
    device: str | torch.device = "cuda",
    use_amp: bool = True,
    compute_clip_score: bool = False,
    clip_score_model_name: str = "openai/clip-vit-base-patch32",
    feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    clip_scorer: Optional[Callable[[torch.Tensor, list[str]], torch.Tensor]] = None,
    overwrite_stats_cache: bool = False,
) -> dict[str, Any]:
    """Generate seeded validation samples across CFG scales and compute FID (and optional CLIP score)."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    if hasattr(vae, "init"):
        vae.init(dev)
    if clip is not None and hasattr(clip, "init"):
        clip.init()

    if feature_extractor is None:
        extractor, resolved_backend = build_inception_feature_extractor(backend=fid_backend, device=dev)
    else:
        extractor, resolved_backend = feature_extractor, "custom"

    img_res = int(model_config.get("data_img_res", 256))
    latent_res = int(model_config.get("img_resolution", img_res // 8))
    latent_ch = int(model_config.get("img_channels", 4))

    ref_source = real_dataset if real_dataset is not None else val_dataset
    real_pixel_range = "0_255" if isinstance(ref_source, (str, Path)) else "minus1_1"

    n_eval = min(int(num_eval_samples), len(val_dataset))
    if n_eval < 2:
        raise ValueError(f"Need at least 2 evaluation samples for FID, got {n_eval}.")

    mu_real, sigma_real, num_real = compute_or_load_real_stats(
        real_source=ref_source,
        feature_extractor=extractor,
        backend_name=resolved_backend,
        dataset_name=dataset_name,
        img_res=img_res,
        max_samples=n_eval if real_dataset is None else None,
        stats_cache_path=stats_cache_path,
        batch_size=max(batch_size, 32),
        pixel_range=real_pixel_range,
        overwrite=overwrite_stats_cache,
    )

    indices = list(range(n_eval))
    if hasattr(val_dataset, "has_clip_cache") and getattr(val_dataset, "has_clip_cache", False):
        prev_random = getattr(val_dataset, "random_caption", True)
        val_dataset.random_caption = False
        try:
            items = [val_dataset[i] for i in indices]
        finally:
            val_dataset.random_caption = prev_random
        prompts = [str(it["prompts"]) for it in items]
        cached_clip_embs = torch.stack([it["clip_embs"] for it in items], dim=0)
        cached_null = getattr(val_dataset, "null_text_emb", None)
    else:
        prompts = _extract_prompts_from_dataset(val_dataset, indices)
        cached_clip_embs = None
        cached_null = None

    if cached_null is not None:
        null_emb_single = cached_null.to(dev)
    elif clip is not None:
        null_emb_single = clip.encode_text([""]).detach().to(dev)
    else:
        raise ValueError("Cannot determine unconditional null text embedding without `clip` or cached `null_text_emb`.")

    mk_cfg = {**_default_mask_configs, **(mask_config or {})}
    mask_total_steps = int(mk_cfg.get("total_steps", model_config["total_steps"]))
    t_sigma = float(model_config.get("transition_sigma", 2.5))
    s_min = float(model_config.get("sigma_min", 0.002))
    s_max = float(model_config.get("sigma_max", 80.0))
    a_ext = float(model_config.get("extreme_prob", 0.05))
    unet_mask_gen = MaskGenerator(
        expert_attributes=mk_cfg["unet_attr"],
        p_mean=mk_cfg["p_mean"],
        p_std=mk_cfg["p_std"],
        total_steps=mask_total_steps,
        min_active=mk_cfg["min_active"],
        step_size=mk_cfg["step_size"],
        min_bandwidth=mk_cfg["min_BW_u"],
        bandwidth=mk_cfg["max_BW_u"],
        strat_band=mk_cfg["strat_band"],
        noise_range=mk_cfg.get("unet_noise_range", (t_sigma, s_max)),
        sigma_max=s_max,
        sigma_min=s_min,
        alpha_ext=a_ext,
    )
    vit_mask_gen = MaskGenerator(
        expert_attributes=mk_cfg["vit_attr"],
        p_mean=mk_cfg["p_mean"],
        p_std=mk_cfg["p_std"],
        total_steps=mask_total_steps,
        min_active=mk_cfg["min_active"],
        step_size=mk_cfg["step_size"],
        min_bandwidth=mk_cfg["min_BW_v"],
        bandwidth=mk_cfg["max_BW_v"],
        strat_band=mk_cfg["strat_band"],
        noise_range=mk_cfg.get("vit_noise_range", (s_min, t_sigma)),
        sigma_max=s_max,
        sigma_min=s_min,
        alpha_ext=a_ext,
    )
    mask_step = mask_total_steps

    clip_fn = None
    if compute_clip_score:
        clip_fn = clip_scorer if clip_scorer is not None else _resolve_clip_score_scorer(dev, clip_score_model_name)

    amp_enabled = bool(use_amp) and dev.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()

    results: dict[float, dict[str, Any]] = {}
    base_save_dir = Path(save_generated_dir) if save_generated_dir is not None else None

    for guidance in guidance_scales:
        g_val = float(guidance)
        gen_rng = torch.Generator(device="cpu").manual_seed(int(seed))
        gen_stats = RunningFeatureStats()
        clip_scores: list[torch.Tensor] = []

        out_cfg_dir = (base_save_dir / f"cfg_{g_val:.2f}") if base_save_dir is not None else None
        if out_cfg_dir is not None:
            out_cfg_dir.mkdir(parents=True, exist_ok=True)

        sampler = EDM_Sampler(
            model=model,
            num_solve_steps=num_solve_steps,
            sigma_min=float(model_config.get("sigma_min", 0.002)),
            sigma_max=float(model_config.get("sigma_max", 80.0)),
            rho=float(model_config.get("rho", 7.0)),
            guidance=g_val,
            dtype=torch.float32,
            unet_mask_gen=unet_mask_gen,
            vit_mask_gen=vit_mask_gen,
            mask_step=mask_step,
        )

        for start in tqdm(range(0, n_eval, batch_size), desc=f"Eval FID CFG={g_val:.2f}", unit="batch"):
            end = min(n_eval, start + batch_size)
            cur_bs = end - start
            batch_prompts = prompts[start:end]
            noise = torch.randn((cur_bs, latent_ch, latent_res, latent_res), generator=gen_rng, dtype=torch.float32).to(dev)

            if cached_clip_embs is not None:
                cond_emb = cached_clip_embs[start:end].to(device=dev, dtype=torch.float32)
            else:
                with amp_ctx:
                    cond_emb = clip.encode_text(batch_prompts)

            uncond_emb = null_emb_single.expand(cur_bs, *null_emb_single.shape[1:])
            with amp_ctx:
                latents = sampler.sample(noise=noise, text_emb=cond_emb, uncond_text_emb=uncond_emb)

            uint8_imgs = vae.decode(latents.float())
            feats = extractor(uint8_imgs)
            gen_stats.update(feats)

            if clip_fn is not None:
                clip_scores.append(clip_fn(uint8_imgs, batch_prompts).detach().cpu())

            if out_cfg_dir is not None:
                for offset, img_t in enumerate(uint8_imgs.cpu()):
                    img_idx = start + offset
                    Image.fromarray(img_t.permute(1, 2, 0).numpy()).save(out_cfg_dir / f"{img_idx:06d}.png")

        mu_gen, sigma_gen = gen_stats.finalize()
        fid_val = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        entry: dict[str, Any] = {
            "fid": fid_val,
            "num_generated": gen_stats.count,
            "num_real": num_real,
        }
        if clip_scores:
            entry["clip_score"] = float(torch.cat(clip_scores, dim=0).mean().item())
        results[g_val] = entry

        summary_msg = f"[Evaluator] CFG={g_val:.2f} | FID={fid_val:.4f} (n_gen={gen_stats.count}, n_real={num_real}, backend={resolved_backend})"
        if "clip_score" in entry:
            summary_msg += f" | CLIPScore={entry['clip_score']:.3f}"
        print(summary_msg)

        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "dataset_name": dataset_name,
        "backend": resolved_backend,
        "num_eval_samples": n_eval,
        "num_real_samples": num_real,
        "num_solve_steps": num_solve_steps,
        "results": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute FID between real and generated image folders.")
    parser.add_argument("--real-dir", required=True, help="Directory of reference validation images.")
    parser.add_argument("--gen-dir", required=True, help="Directory of generated images.")
    parser.add_argument("--backend", default="auto", choices=["auto", "cleanfid", "torchvision"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-res", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    out = compute_fid_from_sources(
        real_source=Path(args.real_dir),
        gen_source=Path(args.gen_dir),
        fid_backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        img_res=args.img_res,
    )
    print(f"FID ({out['backend']}): {out['fid']:.4f} | real={out['num_real']} | gen={out['num_generated']}")


if __name__ == "__main__":
    main()
