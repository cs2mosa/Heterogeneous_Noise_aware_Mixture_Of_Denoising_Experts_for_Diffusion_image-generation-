from contextlib import nullcontext
from glob import glob
import os
from typing import Any, Callable, Optional
import torch
from torch.utils.data import DataLoader
import torch.utils.checkpoint as ckpt_utils
from models.model_config_v3 import preconditioned_HDMOEM_v3
from models.model_components import Unet_block, Vit_block
try:
    from Utils.loss import EDM_LOSS_NOD
    from Utils.utils import (
        sample_sigma_hybrid, ZetaScheduler, MaskGenerator,
        Lr_Scheduler, WSDScheduler, save_checkpoint, EMA,
    )
    from Utils.VAE_CLIP import StabilityVAE, CLIP_EMBED
    from Utils.configs import model_configs, mask_configs, zeta_configs, loss_configs, optim_configs
except ImportError:
    from loss import EDM_LOSS_NOD
    from utils import (
        sample_sigma_hybrid, ZetaScheduler, MaskGenerator,
        Lr_Scheduler, WSDScheduler, save_checkpoint, EMA,
    )
    from VAE_CLIP import StabilityVAE, CLIP_EMBED
    from configs import model_configs, mask_configs, zeta_configs, loss_configs, optim_configs
from graphs.logger2 import Logger
from data_augmentation.data_collector import get_dataloader_path, _ImageLabelDictCollate


def apply_activation_checkpointing(model: torch.nn.Module) -> int:
    """Wrap Vit_block and Unet_block forward passes with activation checkpointing."""
    wrapped = 0
    for mod in model.modules():
        if isinstance(mod, (Vit_block, Unet_block)) and not getattr(mod, "_ckpt_wrapped", False):
            orig_fwd = mod.forward

            def _make_ckpt_forward(fn):
                def _ckpt_forward(*args, **kwargs):
                    if mod.training and torch.is_grad_enabled():
                        return ckpt_utils.checkpoint(fn, *args, use_reentrant=False, **kwargs)
                    return fn(*args, **kwargs)
                return _ckpt_forward

            mod.forward = _make_ckpt_forward(orig_fwd)
            mod._ckpt_wrapped = True
            wrapped += 1
    return wrapped


def build_optimizer_param_groups(
    model: torch.nn.Module,
    Optim_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Partition model parameters into disjoint component optimizer groups and LR map."""
    gate_params = [
        p for n, p in model.named_parameters()
        if n.split(".")[-1].endswith("_gate") or n.split(".")[-1].startswith("out_gain_")
    ]
    gate_param_set = set(gate_params)

    unet_expert_params = [p for p in model.net.Unet_experts.parameters() if p not in gate_param_set]
    vit_expert_params = [p for p in model.net.VIT_experts.parameters() if p not in gate_param_set]
    vit_router_params = list(model.net.vit_router.parameters())
    unet_router_params = list(model.net.Unet_router.parameters())
    out_proj_vit_params = list(model.net.output_proj_vit.parameters())
    out_proj_unet_params = list(model.net.output_proj_unet.parameters())
    blend_residual_params = list(model.net.blend_residual.parameters())

    special_params = set(gate_params)
    for group in (
        unet_expert_params,
        vit_expert_params,
        vit_router_params,
        unet_router_params,
        out_proj_vit_params,
        out_proj_unet_params,
        blend_residual_params,
    ):
        special_params.update(group)

    attn_params = []
    if getattr(model.net, "_cross_attn_enabled", False) and model.net.pooled_cross_attn is not None:
        attn_params.extend([p for p in model.net.pooled_cross_attn.parameters() if p not in gate_param_set])
        attn_params.extend([p for p in model.net.output_proj_attn.parameters() if p not in gate_param_set])
        special_params.update(attn_params)

    input_proj_params = [p for p in model.net.input_proj.parameters() if p not in special_params]
    special_params.update(input_proj_params)
    input_time_proj_params = [p for p in model.net.input_time_proj.parameters() if p not in special_params]
    special_params.update(input_time_proj_params)

    default_params = [p for p in model.parameters() if p not in special_params]

    param_groups = [
        {'params': unet_expert_params, 'lr': Optim_config['lr_unet']},
        {'params': vit_expert_params, 'lr': Optim_config['lr_vit']},
        {'params': vit_router_params, 'lr': Optim_config['lr_router']},
        {'params': unet_router_params, 'lr': Optim_config['lr_router']},
        {'params': out_proj_vit_params, 'lr': Optim_config['lr_out_proj_vit']},
        {'params': out_proj_unet_params, 'lr': Optim_config['lr_out_proj_unet']},
        {'params': blend_residual_params, 'lr': Optim_config['lr_blend_residual']},
        {'params': gate_params, 'lr': Optim_config.get('lr_gates', Optim_config['LR_DEFAULT'])},
        {'params': input_proj_params, 'lr': Optim_config['LR_DEFAULT']},
        {'params': input_time_proj_params, 'lr': Optim_config['LR_DEFAULT']},
        {'params': default_params, 'lr': Optim_config['LR_DEFAULT']},
    ]
    if attn_params:
        param_groups.append({'params': attn_params, 'lr': Optim_config['lr_pooled_attn']})

    lr_per_component = {
        'Unet_experts': Optim_config['lr_unet'],
        'VIT_experts': Optim_config['lr_vit'],
        'Unet_router': Optim_config['lr_router'],
        'vit_router': Optim_config['lr_router'],
        'output_proj_vit': Optim_config['lr_out_proj_vit'],
        'output_proj_unet': Optim_config['lr_out_proj_unet'],
        'blend_residual': Optim_config['lr_blend_residual'],
        'input_proj': Optim_config['LR_DEFAULT'],
        'input_time_proj': Optim_config['LR_DEFAULT'],
        'gate_params': Optim_config.get('lr_gates', Optim_config['LR_DEFAULT']),
        'default': Optim_config['LR_DEFAULT'],
    }
    if attn_params:
        lr_per_component['pooled_cross_attn'] = Optim_config['lr_pooled_attn']

    return param_groups, lr_per_component


def _unpack_batch(batch) -> tuple[torch.Tensor, list[str]]:
    if isinstance(batch, dict):
        return batch["pixel_values"], list(batch["prompts"])
    images_rgb, prompts = batch
    return images_rgb, [str(p) for p in prompts]


def _scale_pixels_to_255(images_rgb: torch.Tensor) -> torch.Tensor:
    if images_rgb.max().item() > 1.5:
        return images_rgb.clamp(0.0, 255.0)
    return ((images_rgb + 1.0) * 127.5).clamp(0.0, 255.0)


def _prune_local_checkpoints(save_dir: str, keep_last: int):
    if keep_last <= 0 or not os.path.isdir(save_dir):
        return
    ckpts = glob(os.path.join(save_dir, "ckpt_*.pt"))
    if len(ckpts) <= keep_last:
        return
    ckpts_sorted = sorted(
        ckpts,
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    for old_path in ckpts_sorted[:-keep_last]:
        try:
            os.remove(old_path)
        except OSError:
            pass


def _build_val_probe_batch(
    val_dataset,
    model_config: dict[str, Any],
    mask_config: dict[str, Any],
    vae,
    clip,
    unet_mask_gen: MaskGenerator,
    vit_mask_gen: MaskGenerator,
    num_samples: int = 16,
    seed: int = 12345,
) -> Optional[dict[str, torch.Tensor]]:
    if val_dataset is None or len(val_dataset) == 0:
        return None
    n = min(int(num_samples), len(val_dataset))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(val_dataset), generator=gen)[:n].tolist()
    items = [val_dataset[i] for i in indices]
    collated = _ImageLabelDictCollate()(items)
    imgs_rgb, prompts = _unpack_batch(collated)

    device = model_config["device"]
    imgs_255 = _scale_pixels_to_255(imgs_rgb.to(device))
    with torch.no_grad():
        latents = vae.encode(imgs_255).float()
        text_embs = clip.encode_text(prompts)
        rnd_norm = torch.randn([n, 1, 1, 1], generator=gen).to(device=device, dtype=torch.float32)
        sigma = (rnd_norm * mask_config["p_std"] + mask_config["p_mean"]).exp().clamp(
            min=model_config["sigma_min"], max=model_config["sigma_max"]
        )
        noise = torch.randn(latents.shape, generator=gen, dtype=torch.float32).to(device) * sigma
        u_mask = unet_mask_gen(sigma=sigma, step=model_config["total_steps"])
        v_mask = vit_mask_gen(sigma=sigma, step=model_config["total_steps"])
    return {
        "latents": latents,
        "noisy": latents + noise,
        "sigma": sigma,
        "text_embs": text_embs,
        "u_mask": u_mask,
        "v_mask": v_mask,
    }


@torch.no_grad()
def _eval_val_probe_mse(
    ema_model: torch.nn.Module,
    probe: dict[str, torch.Tensor],
    zeta_val: float,
    use_amp: bool = False,
) -> float:
    was_training = ema_model.training
    ema_model.eval()
    device_type = probe["latents"].device.type
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if (use_amp and device_type == "cuda") else nullcontext()
    try:
        with ctx:
            out = ema_model(
                x=probe["noisy"],
                sigma=probe["sigma"],
                text_emb=probe["text_embs"],
                Unet_router_mask=probe.get("u_mask"),
                Vit_router_mask=probe.get("v_mask"),
                zeta=zeta_val,
                return_log_var=False,
            )
        return float(torch.mean((out["denoised"].float() - probe["latents"]) ** 2).item())
    finally:
        if was_training:
            ema_model.train()


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    Optim_config: dict[str, Any],
    start_step: int,
):
    sched_kind = str(Optim_config.get("lr_schedule", "edm2")).lower()
    if sched_kind == "wsd":
        return WSDScheduler(
            optimizer=optimizer,
            total_steps=model_config["total_steps"],
            batch_size=model_config["batch_size"],
            warmup_steps=int(Optim_config.get("warmup_steps", 300)),
            stable_until_frac=float(Optim_config.get("stable_until_frac", 0.85)),
            min_lr_mult=float(Optim_config.get("min_lr_mult", 0.10)),
            dataset_size=model_config["dataset_size"],
            cur_nimg=start_step * model_config["batch_size"],
        )
    return Lr_Scheduler(
        optimizer=optimizer,
        total_steps=model_config["total_steps"],
        batch_size=model_config["batch_size"],
        ref_batches=Optim_config["ref_batches"],
        rampup_Mimg=Optim_config["rampup_Mimg"],
        dataset_size=model_config["dataset_size"],
        cur_nimg=start_step * model_config["batch_size"],
    )


def training_HDMOE(
    model_config: dict[str, Any],
    Optim_config: dict[str, Any],
    loss_config: dict[str, Any],
    mask_config: dict[str, Any],
    zeta_config: dict[str, Any],
    resume_path: str = None,
    dataset=None,
    dataloader=None,
    val_dataset=None,
    vae=None,
    clip=None,
    run_name: Optional[str] = None,
    use_amp: Optional[bool] = None,
    grad_accum_steps: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    use_grad_checkpointing: Optional[bool] = None,
    keep_last_ckpts: int = 0,
    val_probe_samples: int = 16,
    post_save_callback: Optional[Callable[[str, str, str], None]] = None,
):
    device = model_config["device"]
    device_str = str(device)
    global_bs = int(model_config["batch_size"])
    micro_bs = int(micro_batch_size or model_config.get("micro_batch_size", global_bs))
    accum_steps = int(
        grad_accum_steps or model_config.get("grad_accum_steps", max(1, global_bs // max(1, micro_bs)))
    )

    if dataloader is not None:
        active_dataloader = dataloader
    elif dataset is not None:
        num_workers = int(model_config.get("num_workers", 2))
        dl_kwargs = {
            "batch_size": micro_bs,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": device_str.startswith("cuda"),
            "drop_last": len(dataset) >= micro_bs,
            "collate_fn": _ImageLabelDictCollate(),
        }
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = True
        active_dataloader = DataLoader(dataset, **dl_kwargs)
    else:
        active_dataloader = get_dataloader_path(model_config, data_dir=model_config["data_dir"])
    data_iter = iter(active_dataloader)

    if vae is None:
        vae = StabilityVAE(
            batch_size=micro_bs,
            raw_mean=model_config["raw_mean"],
            raw_std=model_config["raw_std"],
            final_mean=model_config["mean_data"],
            final_std=model_config["sigma_data"],
        )
        vae.init(device=device)
    elif hasattr(vae, "vae") and getattr(vae, "vae", None) is None:
        vae.init(device=device)

    if clip is None:
        clip = CLIP_EMBED(device=device)
        clip.init()
    elif hasattr(clip, "model") and getattr(clip, "model", None) is None:
        clip.init()

    model = preconditioned_HDMOEM_v3(
        IN_in_channels=model_config["img_channels"],
        IN_img_resolution=model_config["img_resolution"],
        internal_channels=model_config["internal_channels"],
        time_emb_dim=model_config["time_emb_dim"],
        text_emb_dim=model_config["text_emb_dim"],
        num_experts=model_config["num_experts"],
        top_k=model_config["top_k"],
        Fourier_bandwidth=model_config["fourier_bandwidth"],
        VIT_num_blocks=model_config["VIT_num_blocks"],
        VIT_patch_sizes=model_config["VIT_patch_sizes"],
        VIT_num_groups=model_config["VIT_num_groups"],
        VIT_num_heads=model_config["VIT_num_heads"],
        VIT_emb_size=model_config["VIT_emb_size"],
        VIT_emb_balance=model_config["VIT_emb_balances"],
        VIT_pos_emb_balance=model_config["VIT_pos_emb_balances"],
        Unet_num_blocks=model_config["Unet_num_blocks"],
        Unet_channel_mult=model_config["Unet_channel_mult"],
        Unet_channel_mult_emb=model_config["Unet_channel_mult_emb"],
        Unet_kernel_sizes=model_config["Unet_kernel_sizes"],
        Unet_model_channels=model_config["Unet_model_channels"],
        Unet_label_balance=model_config["Unet_label_balance"],
        Unet_concat_balance=model_config["Unet_concat_balances"],
        sigma_data=model_config["sigma_data"],
        log_var_channels=model_config["log_var_channels"],
        transition_sigma=model_config["transition_sigma"],
        transition_sharpness=model_config["transition_sharpness"],
        pool_size=model_config["pool_size"],
        attn_pool_balance=model_config["attn_pool_balance"],
        p_mean=mask_config["p_mean"],
        p_std=mask_config["p_std"],
        sigma_min=model_config["sigma_min"],
        sigma_max=model_config["sigma_max"],
        alpha_ext=model_config["extreme_prob"],
        max_blend_shift=model_config.get("max_blend_shift", 1.5),
        blend_mode=model_config.get("blend_mode", "spherical"),
    ).to(device)

    ckpt_requested = model_config.get("use_grad_checkpointing", False) if use_grad_checkpointing is None else use_grad_checkpointing
    if ckpt_requested:
        wrapped = apply_activation_checkpointing(model)
        print(f"   Activation checkpointing wrapped on {wrapped} blocks.")

    param_groups, lr_per_component = build_optimizer_param_groups(model, Optim_config)

    optimizer = torch.optim.Adam(
        param_groups,
        betas=Optim_config["betas"],
        weight_decay=Optim_config["weight_decay"],
        eps=Optim_config["eps"],
    )
    # EMA for testing
    ema = EMA(
        model,
        beta=model_config["ema_beta"],
        warmup_power=model_config["ema_warmup_power"],
        min_warmup_step=model_config["ema_min_warmup_step"],
    )

    zeta_sched = ZetaScheduler(total_steps=zeta_config["total_schedule_steps"],
                               max_zeta=zeta_config["max_zeta"],
                               min_zeta=zeta_config["min_zeta"],
                               strategy=zeta_config["strategy"],
                               warmup_ratio=zeta_config["warmup_ratio"],
                               )

    Unet_mask_gen = MaskGenerator(expert_attributes=mask_config["unet_attr"],
                                  p_mean=mask_config["p_mean"],
                                  p_std=mask_config["p_std"],
                                  total_steps=mask_config.get('total_steps', model_config['total_steps']),
                                  min_active=mask_config['min_active'],
                                  step_size=mask_config['step_size'],
                                  min_bandwidth=mask_config['min_BW_u'],
                                  bandwidth=mask_config['max_BW_u'],
                                  strat_band=mask_config['strat_band'],
                                  noise_range=mask_config['unet_noise_range'],
                                  sigma_max=model_config['sigma_max'],
                                  sigma_min=model_config['sigma_min'],
                                  alpha_ext=model_config['extreme_prob'],
                                  )

    vit_mask_gen = MaskGenerator(expert_attributes=mask_config["vit_attr"],
                                 p_mean=mask_config["p_mean"],
                                 p_std=mask_config["p_std"],
                                 total_steps=mask_config.get('total_steps', model_config['total_steps']),
                                 min_active=mask_config['min_active'],
                                 step_size=mask_config['step_size'],
                                 min_bandwidth=mask_config['min_BW_v'],
                                 bandwidth=mask_config['max_BW_v'],
                                 strat_band=mask_config['strat_band'],
                                 noise_range=mask_config['vit_noise_range'],
                                 sigma_max=model_config['sigma_max'],
                                 sigma_min=model_config['sigma_min'],
                                 alpha_ext=model_config['extreme_prob'],
                                 )

    criterion = EDM_LOSS_NOD(expert_centers_vit=vit_mask_gen.expert_centers,
                             expert_centers_Unet=Unet_mask_gen.expert_centers,
                             num_experts=model_config["num_experts"],
                             sigma_data=model_config["sigma_data"],
                             Unet_bal=loss_config["unet_bal"],
                             vit_bal=loss_config['vit_bal'],
                             z_bal=loss_config['z_bal'],
                             prior_bal=loss_config['prior_bal'],
                             p_mean=mask_config['p_mean'],
                             p_std=mask_config['p_std'],
                             sharpness=loss_config['sharpness'],
                             transition_sigma=model_config['transition_sigma'],
                             overlap_width=model_config['transition_sharpness'],
                             alpha_vit=loss_config['alpha_vit'],
                             alpha_unet=loss_config['alpha_unet'],
                             alpha_blend=loss_config['alpha_blend'],
                             lap_kernel=loss_config['laplacian_kernel'],
                             lap_sigma=loss_config['laplacian_std'],
                             sigma_min=model_config['sigma_min'],
                             sigma_max=model_config['sigma_max'],
                             alpha_ext=model_config['extreme_prob'],
                             vit_noise_range=mask_config['vit_noise_range'],
                             unet_noise_range=mask_config['unet_noise_range'],
                             )

    log_dir = model_config.get("save_dir_stats", "./logs")
    save_dir = model_config.get("save_dir", "./checkpoints")
    active_run_name = run_name or model_config.get("run_name", "hdmoem_flower102_v1")
    json_log_path = os.path.join(log_dir, f"{active_run_name}_training.jsonl")
    legacy_log_path = os.path.join(log_dir, "training_stats.jsonl")
    print(f"Logging metrics to: {json_log_path}")
    start_step = 0

    # Check if we should resume
    if resume_path:
        checkpoint_file = None

        # Option A: Auto-find latest
        if resume_path == "latest":
            # Look for ckpt_*.pt files
            list_of_files = glob(os.path.join(save_dir, "ckpt_*.pt"))
            if list_of_files:
                # Parse filenames to find the highest step number
                checkpoint_file = max(list_of_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        # Option B: Specific path
        elif os.path.exists(resume_path):
            checkpoint_file = resume_path

        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"🔄 Resuming training from: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

            # 1. Load Step
            start_step = checkpoint["step"] + 1  # Start from next step

            # 2. Load Model & Optimizer
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # 3. Load EMA (including step counter so power-function warmup does not reset to step 0)
            if ema is not None and "ema_model_state_dict" in checkpoint:
                print("   Loading EMA state...")
                ema.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
                ema.step = checkpoint.get('ema_step', checkpoint.get('step', 0))
            # 4. Handle Log file
            if os.path.exists(json_log_path):
                print(f"   Appending to existing log: {json_log_path}")

            print(f"✅ Successfully loaded. Resuming at step {start_step}")
        else:
            checkpoint = None
            print(f"⚠️ Checkpoint not found at {resume_path}. Starting from scratch.")
            if os.path.exists(legacy_log_path):
                os.remove(legacy_log_path)
    else:
        checkpoint = None
        if os.path.exists(legacy_log_path):
            os.remove(legacy_log_path)

    mse_pass_threshold = float(
        model_config.get("mse_pass_threshold", 0.20 * (float(model_config["sigma_data"]) / 0.8) ** 2)
    )
    logger = Logger(
        log_dir=log_dir,
        run_name=active_run_name,
        log_interval=int(model_config.get("log_interval", 10)),
        num_unet_experts=model_config["num_experts"],
        num_vit_experts=model_config["num_experts"],
        transition_sigma=model_config["transition_sigma"],
        sigma_min=model_config["sigma_min"],
        sigma_max=model_config["sigma_max"],
        p_mean=mask_config["p_mean"],
        p_std=mask_config["p_std"],
        alpha_ext=model_config["extreme_prob"],
        adam_eps=float(Optim_config.get("eps", 1e-8)),
        mse_pass_threshold=mse_pass_threshold,
        overwrite=(start_step == 0),
    )

    scheduler = _build_lr_scheduler(optimizer, model_config, Optim_config, start_step)
    if checkpoint is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        apply_nimg = scheduler.cur_nimg if isinstance(scheduler, WSDScheduler) else (scheduler.cur_nimg + scheduler.batch_size)
        scheduler._apply_lr(apply_nimg)

    amp_requested = model_config.get("use_amp", False) if use_amp is None else use_amp
    amp_enabled = bool(amp_requested) and device_str.startswith("cuda")
    blend_anchor_weight = float(model_config.get("blend_anchor_weight", 0.0))
    cfg_drop_prob = float(model_config.get("cfg_drop_prob", 0.10))
    grad_clip = float(Optim_config.get("grad_clip", 100.0))

    cached_null_emb = None
    val_probe_batch = None
    if model_config["total_steps"] > 0:
        if cfg_drop_prob > 0.0 and hasattr(clip, "encode_text"):
            with torch.no_grad():
                cached_null_emb = clip.encode_text([""]).detach()
        if val_dataset is not None:
            val_probe_batch = _build_val_probe_batch(
                val_dataset, model_config, mask_config, vae, clip,
                Unet_mask_gen, vit_mask_gen, num_samples=val_probe_samples,
            )

    model.train()
    current_mse = float("inf")
    config_bundle = {
        "model_configs": model_config,
        "Optim_config": Optim_config,
        "loss_config": loss_config,
        "mask_config": mask_config,
        "zeta_config": zeta_config,
    }
    final_ckpt_name = model_config.get("final_ckpt_name", "final_model1.pt")

    for step in range(start_step, model_config["total_steps"]):
        optimizer.zero_grad(set_to_none=True)
        cur_zeta = zeta_sched.get_zeta(step=step)
        loss_sums: dict[str, torch.Tensor] = {}
        log_var_sum = 0.0
        last_out_model = None
        last_sigma = None

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(active_dataloader)
                batch = next(data_iter)

            images_rgb, prompts = _unpack_batch(batch)
            images = _scale_pixels_to_255(images_rgb.to(device))
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()

            with torch.no_grad():
                with amp_ctx:
                    latent_images = vae.encode(images).float()
                    text_embeddings = clip.encode_text(prompts)
                if cfg_drop_prob > 0.0:
                    cur_bs_txt = text_embeddings.shape[0]
                    drop_mask = torch.rand(cur_bs_txt, device=device) < cfg_drop_prob
                    if drop_mask.any():
                        if cached_null_emb is None:
                            cached_null_emb = clip.encode_text([""]).detach()
                        text_embeddings = text_embeddings.clone()
                        text_embeddings[drop_mask] = cached_null_emb.to(
                            device=text_embeddings.device, dtype=text_embeddings.dtype
                        ).expand(int(drop_mask.sum().item()), *cached_null_emb.shape[1:])

            cur_bs = latent_images.shape[0]
            sigma = sample_sigma_hybrid(
                batch_size=cur_bs,
                sigma_max=model_config["sigma_max"],
                sigma_min=model_config["sigma_min"],
                p_mean=mask_config["p_mean"],
                p_std=mask_config["p_std"],
                extreme_prob=model_config["extreme_prob"],
                device=device,
            )

            noise = torch.randn_like(latent_images) * sigma
            images_noised = latent_images + noise
            Unet_mask = Unet_mask_gen(sigma=sigma, step=step)
            vit_mask = vit_mask_gen(sigma=sigma, step=step)

            with amp_ctx:
                out_model = model(
                    x=images_noised,
                    sigma=sigma,
                    text_emb=text_embeddings,
                    Unet_router_mask=Unet_mask,
                    Vit_router_mask=vit_mask,
                    zeta=cur_zeta,
                    return_log_var=True,
                )
                loss_micro = criterion(
                    x=latent_images,
                    noisy_x=images_noised,
                    sigma=sigma,
                    out_model=out_model,
                )
                total_micro = loss_micro["loss"]
                if blend_anchor_weight > 0.0:
                    bounded_res = getattr(model.net, "_last_bounded_residual", None)
                    if bounded_res is not None:
                        total_micro = total_micro + blend_anchor_weight * (bounded_res.float() ** 2).mean()
                model.net._last_bounded_residual = None

            (total_micro / float(accum_steps)).backward()

            for k, v in loss_micro.items():
                if torch.is_tensor(v):
                    v_det = v.detach().float()
                    loss_sums[k] = v_det if k not in loss_sums else (loss_sums[k] + v_det)
            if out_model.get("log_var") is not None:
                log_var_sum += float(out_model["log_var"].detach().mean().item())
            last_out_model = out_model
            last_sigma = sigma

        avg_loss = {k: (v / float(accum_steps)) for k, v in loss_sums.items()}
        avg_log_var = log_var_sum / float(accum_steps)

        logger.log_training_step(
            step=step,
            loss_dict=avg_loss,
            sigma=last_sigma,
            zeta=cur_zeta,
            log_var=avg_log_var,
            lr=optimizer.param_groups[0]["lr"],
            w_unet=last_out_model["w_unet"],
            w_vit=last_out_model["w_vit"],
        )
        logger.log_router_statistics(
            step=step,
            unet_probs=last_out_model["Unet_router_loss"],
            vit_probs=last_out_model["vit_router_loss"],
            sigma=last_sigma,
        )
        logger.log_branch_specialization(
            step=step,
            vit_out=last_out_model["vit_denoised"],
            unet_out=last_out_model["unet_denoised"],
            sigma=last_sigma,
        )

        lr_mult = scheduler.get_lr_multiplier()
        live_lrs = {k: v * lr_mult for k, v in lr_per_component.items()}
        logger.log_gradients(step=step, model=model.net, lr_per_component=live_lrs)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update()
        current_mse = float(avg_loss["denoising"].item())

        if step % 100 == 0 or step == start_step:
            print(
                f"Step {step}/{model_config['total_steps']} | "
                f"total_Loss: {avg_loss['loss']:.4f} | "
                f"MSE: {avg_loss['denoising']:.4f} | "
                f"Z_loss: {avg_loss['z_loss']:.4f} | "
                f"Routers_loss: {avg_loss['balance']:.4f} | "
                f"entropy: {avg_loss['entropy']:.4f} | "
                f"pure_loss: {avg_loss['pure_loss']:.4f} | "
                f"LogVar: {avg_log_var:.3f}"
            )

        if step % model_config.get("save_interval", 1000) == 0 and step > 0:
            val_probe_mse = None
            if val_probe_batch is not None:
                val_probe_mse = _eval_val_probe_mse(
                    ema.ema_model, val_probe_batch, cur_zeta, use_amp=amp_enabled
                )
            ckpt_mse = val_probe_mse if val_probe_mse is not None else current_mse
            extra_info = {"train_batch_mse": round(current_mse, 6)}
            if val_probe_mse is not None:
                extra_info["val_probe_mse"] = round(val_probe_mse, 6)
            logger.log_checkpoint_summary(
                step=step,
                mse=ckpt_mse,
                model=model.net,
                unet_probs=last_out_model["Unet_router_loss"],
                vit_probs=last_out_model["vit_router_loss"],
                lr_per_component=live_lrs,
                extra=extra_info,
                sigma=last_sigma,
            )

            save_checkpoint(
                model,
                optimizer,
                step,
                ckpt_mse,
                config_bundle,
                f"ckpt_{step}.pt",
                ema_instance=ema,
                scheduler_instance=scheduler,
            )
            _prune_local_checkpoints(save_dir, keep_last_ckpts)
            if post_save_callback is not None:
                post_save_callback(save_dir, log_dir, f"step_{step}")

    final_val_mse = None
    if val_probe_batch is not None:
        final_val_mse = _eval_val_probe_mse(
            ema.ema_model, val_probe_batch, zeta_sched.get_zeta(step=model_config["total_steps"]), use_amp=amp_enabled
        )
        print(f"[Final Eval] Deterministic Val Probe MSE (EMA): {final_val_mse:.5f}")
    final_eval_mse = final_val_mse if final_val_mse is not None else current_mse
    save_checkpoint(
        model, optimizer, model_config["total_steps"], final_eval_mse,
        config_bundle, final_ckpt_name,
        ema_instance=ema,
        scheduler_instance=scheduler,
    )
    if post_save_callback is not None:
        post_save_callback(save_dir, log_dir, "final")

    print("Training Complete.")
    return model, ema, model_config, logger

if __name__ == '__main__':
    training_HDMOE(model_configs,
                   optim_configs,
                   loss_configs,
                   mask_configs,
                   zeta_configs,
                   resume_path=r"./ckpt_26000.pt"
                   )