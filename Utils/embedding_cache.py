from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


EMBEDDING_CACHE_VERSION = 1


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def compute_whitening_params(
    raw_mean: Sequence[float] | torch.Tensor,
    raw_std: Sequence[float] | torch.Tensor,
    final_mean: float = 0.0,
    final_std: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `[1, C, 1, 1]` affine scale and bias matching StabilityVAE.encode."""
    mean_arr = np.asarray(raw_mean, dtype=np.float32)
    std_arr = np.asarray(raw_std, dtype=np.float32)
    scale = np.asarray(final_std, dtype=np.float32) / std_arr
    bias = np.asarray(final_mean, dtype=np.float32) - mean_arr * scale
    return (
        torch.tensor(scale, dtype=torch.float32).reshape(1, -1, 1, 1),
        torch.tensor(bias, dtype=torch.float32).reshape(1, -1, 1, 1),
    )


def whiten_posterior_latents(
    vae_mean: torch.Tensor,
    vae_std: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    deterministic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from N(mean, std^2) (or use mean) and apply EDM channel whitening."""
    mean_f32 = vae_mean.float()
    std_f32 = vae_std.float()
    if deterministic:
        raw_latents = mean_f32
    else:
        if generator is not None:
            eps = torch.randn(mean_f32.shape, dtype=mean_f32.dtype, device=mean_f32.device, generator=generator)
        else:
            eps = torch.randn_like(mean_f32)
        raw_latents = mean_f32 + eps * std_f32

    scale_t = scale.to(device=raw_latents.device, dtype=raw_latents.dtype)
    bias_t = bias.to(device=raw_latents.device, dtype=raw_latents.dtype)
    return raw_latents * scale_t + bias_t


def _pixel_batch_to_vae_input(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.dtype == torch.uint8 or float(pixels.max().item()) > 1.5:
        return (pixels.to(torch.float32) / 127.5) - 1.0
    return pixels.to(torch.float32).clamp(-1.0, 1.0)


def _extract_item_pixels_and_prompt(item: Any) -> tuple[torch.Tensor, list[str]]:
    if isinstance(item, dict):
        pixels = item["pixel_values"]
        raw_p = item.get("prompts", item.get("prompt", item.get("caption", "")))
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        pixels = item[0]
        raw_p = item[1]
    else:
        raise TypeError(f"Unsupported dataset item type for embedding cache: {type(item).__name__}")

    if isinstance(raw_p, (list, tuple)):
        caps = [str(c) for c in raw_p]
    else:
        caps = [str(raw_p)]
    return pixels, caps


def extract_sample_captions_and_key(
    dataset: Any,
    idx: int,
    fallback_item: Optional[Any] = None,
    max_captions_per_image: int = 0,
) -> tuple[list[str], str]:
    """Return canonical caption list and stable alignment key for `dataset[idx]`."""
    records = getattr(dataset, "records", None)
    caps: Optional[list[str]] = None
    key: Optional[str] = None

    if records is not None and 0 <= idx < len(records):
        rec = records[idx]
        if isinstance(rec, (tuple, list)) and len(rec) >= 2:
            raw_key, raw_caps = rec[0], rec[1]
            caps = [str(c) for c in raw_caps] if isinstance(raw_caps, (list, tuple)) else [str(raw_caps)]
            name_part = os.path.basename(str(raw_key))
            key = f"{idx}:{name_part}"

    if caps is None:
        item = fallback_item if fallback_item is not None else dataset[idx]
        if isinstance(item, dict) and "sample_key" in item:
            key = str(item["sample_key"])
        _, caps = _extract_item_pixels_and_prompt(item)
        if key is None:
            digest = hashlib.sha1(caps[0].encode("utf-8")).hexdigest()[:12]
            key = f"{idx}:{digest}"

    if max_captions_per_image > 0:
        caps = caps[:max_captions_per_image]
    if not caps:
        caps = [""]
    return caps, key


class _IndexedCacheSourceDataset(Dataset):
    """Wrap source dataset so DataLoader yields global index, pixels, all captions, and alignment key."""

    def __init__(self, base_dataset: Dataset, start_idx: int, end_idx: int, max_captions_per_image: int = 0):
        self.base_dataset = base_dataset
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.max_captions_per_image = int(max_captions_per_image)

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, rel_idx: int) -> dict[str, Any]:
        global_idx = self.start_idx + int(rel_idx)
        item = self.base_dataset[global_idx]
        pixels, _ = _extract_item_pixels_and_prompt(item)
        captions, sample_key = extract_sample_captions_and_key(
            self.base_dataset,
            global_idx,
            fallback_item=item,
            max_captions_per_image=self.max_captions_per_image,
        )
        return {
            "global_idx": global_idx,
            "pixel_values": pixels,
            "captions": captions,
            "sample_key": sample_key,
        }


def _indexed_source_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "global_idx": [int(x["global_idx"]) for x in batch],
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "captions": [list(x["captions"]) for x in batch],
        "sample_key": [str(x["sample_key"]) for x in batch],
    }


def _default_vae_id(vae: Any) -> str:
    return str(getattr(vae, "vae_name", "custom_vae"))


def _default_clip_id(clip: Any) -> str:
    return str(getattr(clip, "embed_name", getattr(clip, "model_name", "custom_clip")))


def _build_signature(
    dataset_name: str,
    split: str,
    num_images: int,
    img_res: int,
    vae_id: str,
    clip_id: Optional[str],
    cache_clip: bool,
    max_captions_per_image: int,
) -> dict[str, Any]:
    return {
        "dataset_name": str(dataset_name),
        "split": str(split),
        "num_images": int(num_images),
        "img_res": int(img_res),
        "vae_id": str(vae_id),
        "clip_id": str(clip_id) if (cache_clip and clip_id is not None) else None,
        "cache_clip": bool(cache_clip),
        "max_captions_per_image": int(max_captions_per_image),
    }


def validate_manifest_signature(
    manifest: dict[str, Any],
    expected_signature: dict[str, Any],
    require_clip_cache: Optional[bool] = None,
) -> None:
    """Verify cache manifest metadata matches caller expectations to prevent dataset/model mismatches."""
    if int(manifest.get("version", -1)) != EMBEDDING_CACHE_VERSION:
        raise ValueError(
            f"Unsupported cache version {manifest.get('version')} (expected {EMBEDDING_CACHE_VERSION})."
        )
    sig = manifest.get("dataset_signature", {})
    check_fields = ["dataset_name", "split", "num_images", "img_res", "vae_id"]
    if expected_signature.get("cache_clip", False) or require_clip_cache:
        check_fields.extend(["cache_clip", "max_captions_per_image"])
        if expected_signature.get("clip_id") is not None:
            check_fields.append("clip_id")

    mismatches = []
    for field in check_fields:
        exp_val = expected_signature.get(field)
        got_val = sig.get(field)
        if exp_val is not None and got_val != exp_val:
            mismatches.append(f"{field}: cache={got_val!r} vs expected={exp_val!r}")
    if require_clip_cache is True and not bool(sig.get("cache_clip", False)):
        mismatches.append("cache_clip: cache=False vs required=True")
    if mismatches:
        raise ValueError("Embedding cache mismatch detected: " + "; ".join(mismatches))


def _load_manifest(cache_dir: Path) -> Optional[dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_stats_from_cache(cache_dir: str | Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute exact analytical per-channel raw_mean and raw_std over all cached (mean, std) shards."""
    cache_path = Path(cache_dir)
    manifest = _load_manifest(cache_path)
    if manifest is None or not manifest.get("shards"):
        raise FileNotFoundError(f"No shards/manifest found in cache directory: {cache_path}")

    sum_mean: Optional[torch.Tensor] = None
    sum_second_moment: Optional[torch.Tensor] = None
    total_elems = 0

    for entry in manifest["shards"]:
        shard = torch.load(cache_path / entry["filename"], map_location="cpu", weights_only=False)
        m = shard["vae_mean"].to(torch.float64)
        s = shard["vae_std"].to(torch.float64)
        if m.ndim != 4:
            raise ValueError(f"Expected 4D vae_mean in {entry['filename']}, got {tuple(m.shape)}")
        n_spatial = m.shape[0] * m.shape[2] * m.shape[3]
        batch_sum_m = m.sum(dim=(0, 2, 3))
        # E[(mu + sigma * eps)^2] = mu^2 + sigma^2 since E[eps]=0, Var(eps)=1
        batch_sum_m2 = (m * m + s * s).sum(dim=(0, 2, 3))
        if sum_mean is None:
            sum_mean = batch_sum_m
            sum_second_moment = batch_sum_m2
        else:
            sum_mean += batch_sum_m
            sum_second_moment += batch_sum_m2
        total_elems += n_spatial

    assert sum_mean is not None and sum_second_moment is not None and total_elems > 0
    chan_mean = sum_mean / total_elems
    chan_var = (sum_second_moment / total_elems - chan_mean * chan_mean).clamp(min=1e-12)
    chan_std = torch.sqrt(chan_var)
    return (
        tuple(round(float(x), 6) for x in chan_mean.tolist()),
        tuple(round(float(x), 6) for x in chan_std.tolist()),
    )


def sync_cache_to_local(source_dir: str | Path, local_dir: str | Path) -> Path:
    """Copy manifest, null embedding, and completed shards from Drive cache to faster local NVMe."""
    src = Path(source_dir)
    dst = Path(local_dir)
    if src.resolve() == dst.resolve():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("manifest.json", "clip_null_emb.pt"):
        sfile = src / fname
        if sfile.exists():
            dfile = dst / fname
            if not dfile.exists() or dfile.stat().st_size != sfile.stat().st_size or fname == "manifest.json":
                shutil.copy2(sfile, dfile)
    manifest = _load_manifest(src)
    if manifest is not None:
        for entry in tqdm(manifest.get("shards", []), desc=f"Syncing cache -> {dst}", leave=False):
            sfile = src / entry["filename"]
            dfile = dst / entry["filename"]
            if sfile.exists() and (not dfile.exists() or dfile.stat().st_size != sfile.stat().st_size):
                shutil.copy2(sfile, dfile)
    return dst


def _verify_existing_shard(
    shard_path: Path,
    entry: dict[str, Any],
    base_dataset: Optional[Dataset] = None,
    max_captions_per_image: int = 0,
    require_clip: bool = False,
) -> bool:
    if not shard_path.exists():
        return False
    try:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    except Exception:
        return False

    start_idx = int(entry["start_idx"])
    end_idx = int(entry["end_idx"])
    expected_count = end_idx - start_idx
    if shard.get("start_idx") != start_idx or shard.get("end_idx") != end_idx:
        return False
    if shard["vae_mean"].shape[0] != expected_count or shard["vae_std"].shape != shard["vae_mean"].shape:
        return False
    if len(shard.get("sample_keys", [])) != expected_count or len(shard.get("prompts", [])) != expected_count:
        return False
    if require_clip and ("clip_embs" not in shard or "clip_offsets" not in shard):
        return False
    if "clip_offsets" in shard and len(shard["clip_offsets"]) != expected_count + 1:
        return False

    if base_dataset is not None and expected_count > 0:
        for rel_check in (0, expected_count - 1):
            g_idx = start_idx + rel_check
            exp_caps, exp_key = extract_sample_captions_and_key(
                base_dataset, g_idx, max_captions_per_image=max_captions_per_image
            )
            if shard["sample_keys"][rel_check] != exp_key or shard["prompts"][rel_check] != exp_caps:
                raise ValueError(
                    f"Existing shard {shard_path.name} failed key/caption alignment at global_idx={g_idx}: "
                    f"shard=({shard['sample_keys'][rel_check]!r}) vs dataset=({exp_key!r})."
                )
    return True


class CachedEmbeddingDataset(Dataset):
    """Aligned sharded dataset serving cached VAE (mean, std) and aligned prompts / CLIP embeddings."""

    def __init__(
        self,
        cache_dir: str | Path,
        expected_signature: Optional[dict[str, Any]] = None,
        require_clip_cache: Optional[bool] = None,
        random_caption: bool = True,
        preload_vae_to_ram: bool = False,
        max_cached_shards: int = 4,
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        manifest = _load_manifest(self.cache_dir)
        if manifest is None:
            raise FileNotFoundError(f"manifest.json not found in {self.cache_dir}")
        if not manifest.get("complete", False):
            raise ValueError(f"Embedding cache in {self.cache_dir} is incomplete (complete=False).")
        if expected_signature is not None:
            validate_manifest_signature(manifest, expected_signature, require_clip_cache=require_clip_cache)
        elif require_clip_cache is True and not manifest.get("dataset_signature", {}).get("cache_clip", False):
            raise ValueError(f"Embedding cache in {self.cache_dir} does not contain CLIP embeddings.")

        self.manifest = manifest
        self.signature = dict(manifest["dataset_signature"])
        self.has_clip_cache = bool(self.signature.get("cache_clip", False))
        self.raw_mean = tuple(float(x) for x in manifest["raw_mean"])
        self.raw_std = tuple(float(x) for x in manifest["raw_std"])
        self.random_caption = bool(random_caption)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._seed = int(seed)
        self._rng: Optional[random.Random] = None
        self._shard_lru: OrderedDict[int, dict[str, Any]] = OrderedDict()

        null_path = self.cache_dir / "clip_null_emb.pt"
        self.null_text_emb: Optional[torch.Tensor] = (
            torch.load(null_path, map_location="cpu", weights_only=False) if null_path.exists() else None
        )

        self.shards = list(manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"Manifest in {self.cache_dir} lists zero shards.")

        cursor = 0
        self._shard_ends: list[int] = []
        for entry in self.shards:
            s_idx = int(entry["start_idx"])
            e_idx = int(entry["end_idx"])
            cnt = int(entry["count"])
            shard_file = self.cache_dir / entry["filename"]
            if not shard_file.exists():
                raise FileNotFoundError(f"Missing cache shard: {shard_file}")
            if s_idx != cursor or (e_idx - s_idx) != cnt or cnt <= 0:
                raise ValueError(
                    f"Non-contiguous shard table in {self.cache_dir}: {entry} (expected start_idx={cursor})."
                )
            cursor = e_idx
            self._shard_ends.append(e_idx)

        expected_n = int(self.signature.get("num_images", cursor))
        if cursor != expected_n:
            raise ValueError(f"Shard coverage {cursor} != manifest num_images {expected_n}.")
        self.num_images = cursor

        self._ram_vae_mean: Optional[torch.Tensor] = None
        self._ram_vae_std: Optional[torch.Tensor] = None
        self._ram_prompts: Optional[list[list[str]]] = None
        self._ram_keys: Optional[list[str]] = None
        if preload_vae_to_ram and not self.has_clip_cache:
            self._preload_vae_and_prompts()

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            offset = worker_info.id if worker_info is not None else 0
            self._rng = random.Random(self._seed + offset)
        return self._rng

    def _preload_vae_and_prompts(self) -> None:
        means, stds, prompts, keys = [], [], [], []
        for entry in self.shards:
            shard = torch.load(self.cache_dir / entry["filename"], map_location="cpu", weights_only=False)
            means.append(shard["vae_mean"])
            stds.append(shard["vae_std"])
            prompts.extend(shard["prompts"])
            keys.extend(shard["sample_keys"])
        self._ram_vae_mean = torch.cat(means, dim=0)
        self._ram_vae_std = torch.cat(stds, dim=0)
        self._ram_prompts = prompts
        self._ram_keys = keys

    def _get_shard(self, shard_idx: int) -> dict[str, Any]:
        cached = self._shard_lru.get(shard_idx)
        if cached is not None:
            self._shard_lru.move_to_end(shard_idx)
            return cached
        shard = torch.load(self.cache_dir / self.shards[shard_idx]["filename"], map_location="cpu", weights_only=False)
        self._shard_lru[shard_idx] = shard
        while len(self._shard_lru) > self.max_cached_shards:
            self._shard_lru.popitem(last=False)
        return shard

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self.num_images
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images}).")

        shard_idx = bisect_right(self._shard_ends, idx)
        shard_start = 0 if shard_idx == 0 else self._shard_ends[shard_idx - 1]
        local_idx = idx - shard_start

        if self._ram_vae_mean is not None and not self.has_clip_cache:
            vae_mean = self._ram_vae_mean[idx]
            vae_std = self._ram_vae_std[idx]
            caps = self._ram_prompts[idx]
            sample_key = self._ram_keys[idx]
            shard = None
        else:
            shard = self._get_shard(shard_idx)
            vae_mean = shard["vae_mean"][local_idx]
            vae_std = shard["vae_std"][local_idx]
            caps = shard["prompts"][local_idx]
            sample_key = shard["sample_keys"][local_idx]

        cap_idx = self._get_rng().randrange(len(caps)) if (self.random_caption and len(caps) > 1) else 0
        out: dict[str, Any] = {
            "vae_mean": vae_mean,
            "vae_std": vae_std,
            "prompts": caps[cap_idx],
            "sample_key": sample_key,
            "caption_idx": cap_idx,
        }

        if self.has_clip_cache:
            assert shard is not None
            offsets = shard["clip_offsets"]
            c_start = int(offsets[local_idx].item())
            c_end = int(offsets[local_idx + 1].item())
            if (c_end - c_start) != len(caps):
                raise RuntimeError(
                    f"Aligned caption/CLIP offset mismatch at global_idx={idx}: "
                    f"offsets span {c_end - c_start}, prompts len {len(caps)}."
                )
            out["clip_embs"] = shard["clip_embs"][c_start + cap_idx]

        return out


@torch.no_grad()
def _encode_clip_list(clip: Any, texts: list[str], clip_batch_size: int, amp_ctx) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for i in range(0, len(texts), clip_batch_size):
        sub = texts[i : i + clip_batch_size]
        with amp_ctx:
            emb = clip.encode_text(sub)
        chunks.append(emb.detach().to(device="cpu", dtype=torch.float16))
    return torch.cat(chunks, dim=0)


def build_or_load_embedding_cache(
    cache_dir: str | Path,
    dataset: Optional[Dataset] = None,
    vae: Any = None,
    clip: Any = None,
    dataset_name: str = "custom",
    split: str = "train",
    img_res: int = 256,
    vae_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    cache_clip: bool = True,
    max_captions_per_image: int = 0,
    shard_size: Optional[int] = None,
    encode_batch_size: int = 32,
    clip_batch_size: int = 128,
    num_workers: int = 0,
    device: str | torch.device = "cuda",
    use_amp: bool = True,
    overwrite: bool = False,
    local_cache_dir: Optional[str | Path] = None,
    preload_vae_to_ram: bool = False,
    max_cached_shards: int = 4,
    require_clip_cache: Optional[bool] = None,
    random_caption: bool = True,
    seed: int = 42,
) -> CachedEmbeddingDataset:
    """Build (or resume / validate and load) a sharded aligned VAE + optional CLIP cache on Drive."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    resolved_vae_id = vae_id or _default_vae_id(vae)
    resolved_clip_id = (clip_id or _default_clip_id(clip)) if cache_clip else None

    if overwrite and cache_path.exists():
        for p in cache_path.glob("shard_*.pt*"):
            p.unlink(missing_ok=True)
        (cache_path / "manifest.json").unlink(missing_ok=True)

    existing_manifest = _load_manifest(cache_path)
    if existing_manifest is not None and existing_manifest.get("complete", False) and not overwrite:
        exp_sig = {
            "dataset_name": dataset_name,
            "split": split,
            "num_images": len(dataset) if dataset is not None else existing_manifest["dataset_signature"]["num_images"],
            "img_res": img_res,
            "vae_id": resolved_vae_id if vae is not None or vae_id is not None else existing_manifest["dataset_signature"]["vae_id"],
            "clip_id": resolved_clip_id if (cache_clip and (clip is not None or clip_id is not None)) else None,
            "cache_clip": cache_clip if require_clip_cache is None else require_clip_cache,
            "max_captions_per_image": max_captions_per_image,
        }
        validate_manifest_signature(existing_manifest, exp_sig, require_clip_cache=require_clip_cache)
        read_dir = sync_cache_to_local(cache_path, local_cache_dir) if local_cache_dir else cache_path
        return CachedEmbeddingDataset(
            cache_dir=read_dir,
            expected_signature=exp_sig,
            require_clip_cache=require_clip_cache,
            random_caption=random_caption,
            preload_vae_to_ram=preload_vae_to_ram,
            max_cached_shards=max_cached_shards,
            seed=seed,
        )

    if dataset is None:
        raise FileNotFoundError(
            f"Complete cache not found in {cache_path} and no source dataset was supplied to build it."
        )
    if vae is None:
        raise ValueError("Building an embedding cache requires a `vae` encoder instance.")
    if cache_clip and clip is None:
        raise ValueError("`cache_clip=True` requires a `clip` text encoder instance.")

    num_images = len(dataset)
    if num_images == 0:
        raise ValueError("Cannot build embedding cache from an empty dataset.")

    sig = _build_signature(
        dataset_name=dataset_name,
        split=split,
        num_images=num_images,
        img_res=img_res,
        vae_id=resolved_vae_id,
        clip_id=resolved_clip_id,
        cache_clip=cache_clip,
        max_captions_per_image=max_captions_per_image,
    )

    eff_shard_size = int(shard_size) if shard_size is not None else (512 if cache_clip else 1024)
    eff_shard_size = max(1, eff_shard_size)

    if existing_manifest is not None and not overwrite:
        validate_manifest_signature(existing_manifest, sig, require_clip_cache=cache_clip)
        existing_by_range = {
            (int(e["start_idx"]), int(e["end_idx"])): e for e in existing_manifest.get("shards", [])
        }
    else:
        existing_by_range = {}

    dev = torch.device(device)
    amp_enabled = bool(use_amp) and dev.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()

    if hasattr(vae, "init"):
        vae.init(dev)
    if clip is not None and hasattr(clip, "init"):
        clip.init()

    null_emb_path = cache_path / "clip_null_emb.pt"
    if clip is not None and not null_emb_path.exists():
        with torch.no_grad():
            null_emb = clip.encode_text([""]).detach().cpu().to(torch.float32)
        _atomic_torch_save(null_emb, null_emb_path)

    total_shards = math.ceil(num_images / eff_shard_size)
    shard_entries: list[dict[str, Any]] = []

    for shard_idx in tqdm(range(total_shards), desc=f"Caching {dataset_name}/{split}", unit="shard"):
        start_idx = shard_idx * eff_shard_size
        end_idx = min(num_images, start_idx + eff_shard_size)
        fname = f"shard_{shard_idx:05d}.pt"
        shard_path = cache_path / fname

        prev_entry = existing_by_range.get((start_idx, end_idx))
        if prev_entry is not None and _verify_existing_shard(
            shard_path, prev_entry, base_dataset=dataset,
            max_captions_per_image=max_captions_per_image, require_clip=cache_clip,
        ):
            shard_entries.append(prev_entry)
            continue

        sub_ds = _IndexedCacheSourceDataset(
            dataset, start_idx=start_idx, end_idx=end_idx, max_captions_per_image=max_captions_per_image
        )
        sub_dl = DataLoader(
            sub_ds,
            batch_size=encode_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
            collate_fn=_indexed_source_collate,
        )

        shard_means: list[torch.Tensor] = []
        shard_stds: list[torch.Tensor] = []
        shard_prompts: list[list[str]] = []
        shard_keys: list[str] = []
        expected_cursor = start_idx

        with torch.no_grad():
            for batch in sub_dl:
                g_indices = batch["global_idx"]
                expected_range = list(range(expected_cursor, expected_cursor + len(g_indices)))
                if g_indices != expected_range:
                    raise RuntimeError(
                        f"Non-sequential ordering detected while building cache: {g_indices} vs {expected_range}"
                    )
                expected_cursor += len(g_indices)

                pix = _pixel_batch_to_vae_input(batch["pixel_values"].to(dev))
                with amp_ctx:
                    if hasattr(vae, "_run_vae_encoder_on_batch"):
                        b_mean, b_std = vae._run_vae_encoder_on_batch(pix)
                    else:
                        raise AttributeError(
                            f"{type(vae).__name__} must implement `_run_vae_encoder_on_batch(pixel_batch)`."
                        )
                shard_means.append(b_mean.detach().to(device="cpu", dtype=torch.float16))
                shard_stds.append(b_std.detach().to(device="cpu", dtype=torch.float16))
                shard_prompts.extend(batch["captions"])
                shard_keys.extend(batch["sample_key"])

        vae_mean_t = torch.cat(shard_means, dim=0)
        vae_std_t = torch.cat(shard_stds, dim=0)
        count = end_idx - start_idx
        if vae_mean_t.shape[0] != count or len(shard_keys) != count or len(shard_prompts) != count:
            raise RuntimeError(
                f"Shard count mismatch in {fname}: mean={vae_mean_t.shape[0]}, keys={len(shard_keys)}, expected={count}"
            )

        shard_payload: dict[str, Any] = {
            "version": EMBEDDING_CACHE_VERSION,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "sample_keys": shard_keys,
            "prompts": shard_prompts,
            "vae_mean": vae_mean_t,
            "vae_std": vae_std_t,
        }

        if cache_clip:
            flat_captions: list[str] = []
            offsets = [0]
            for caps in shard_prompts:
                flat_captions.extend(caps)
                offsets.append(len(flat_captions))
            clip_embs_t = _encode_clip_list(clip, flat_captions, clip_batch_size=clip_batch_size, amp_ctx=amp_ctx)
            if clip_embs_t.shape[0] != len(flat_captions):
                raise RuntimeError(
                    f"CLIP embedding count mismatch in {fname}: {clip_embs_t.shape[0]} vs {len(flat_captions)}"
                )
            shard_payload["clip_embs"] = clip_embs_t
            shard_payload["clip_offsets"] = torch.tensor(offsets, dtype=torch.int64)

        _atomic_torch_save(shard_payload, shard_path)
        entry = {
            "filename": fname,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "count": count,
            "has_clip": bool(cache_clip),
            "sample_keys_head": shard_keys[:2],
            "sample_keys_tail": shard_keys[-2:],
        }
        shard_entries.append(entry)

        _atomic_write_json(
            cache_path / "manifest.json",
            {
                "version": EMBEDDING_CACHE_VERSION,
                "complete": False,
                "dataset_signature": sig,
                "shards": shard_entries,
            },
        )

    # Save temporary complete=False manifest first, compute exact raw_mean/raw_std across all shards
    raw_mean, raw_std = compute_stats_from_cache(cache_path)
    _atomic_write_json(
        cache_path / "manifest.json",
        {
            "version": EMBEDDING_CACHE_VERSION,
            "complete": True,
            "dataset_signature": sig,
            "raw_mean": list(raw_mean),
            "raw_std": list(raw_std),
            "shards": shard_entries,
        },
    )

    read_dir = sync_cache_to_local(cache_path, local_cache_dir) if local_cache_dir else cache_path
    return CachedEmbeddingDataset(
        cache_dir=read_dir,
        expected_signature=sig,
        require_clip_cache=require_clip_cache,
        random_caption=random_caption,
        preload_vae_to_ram=preload_vae_to_ram,
        max_cached_shards=max_cached_shards,
        seed=seed,
    )
