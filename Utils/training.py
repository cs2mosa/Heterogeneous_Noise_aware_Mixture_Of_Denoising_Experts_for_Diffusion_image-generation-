import os
import torch
from typing import Any
from models.model_config_v3 import preconditioned_HDMOEM_v3
from loss import EDM_LOSS_NOD
from utils import sample_sigma_hybrid, ZetaScheduler, MaskGenerator, Lr_Scheduler, save_checkpoint, EMA, \
    register_gradient_scalers
from VAE_CLIP import StabilityVAE
from configs import model_configs, mask_configs, zeta_configs, loss_configs, optim_configs
from graphs.logger2 import Logger
from data_augmentation.data_collector import get_dataloader
from glob import glob


def training_HDMOE(model_config: dict[str, Any],
                   Optim_config: dict[str, Any],
                   loss_config: dict[str, Any],
                   mask_config: dict[str, Any],
                   zeta_config: dict[str, Any],
                   resume_path: str = None
                   ):
    dataloader = get_dataloader(model_config)
    data_iter = iter(dataloader)
    logger = Logger(
        log_dir="./logs",
        run_name="hdmoem_flower102_v1",
        log_interval=10,
        num_unet_experts=model_config["num_experts"],
        num_vit_experts=model_config["num_experts"],
        transition_sigma=model_configs['transition_sigma'],
        sigma_min=model_config['sigma_min'],
        sigma_max=model_config['sigma_max'],
        p_mean=mask_config["p_mean"],
        p_std=mask_config["p_std"]
    )
    lr_per_component = {
        'Unet_experts': Optim_config['lr_unet'],
        'VIT_experts': Optim_config['lr_vit'],
        'Unet_router': Optim_config['lr_router'],
        'vit_router': Optim_config['lr_router'],
        'output_proj_vit': Optim_config['lr_out_proj_vit'],
        'output_proj_unet': Optim_config['lr_out_proj_unet'],
        'default': Optim_config['LR_DEFAULT']
    }
    vae = StabilityVAE(batch_size=model_config["batch_size"],
                       raw_mean=model_config['raw_mean'],
                       raw_std=model_config['raw_std'],
                       final_mean=model_config['mean_data'],
                       final_std=model_config['sigma_data']
                       )

    vae.init(device=model_config["device"])
    model = preconditioned_HDMOEM_v3(
        IN_in_channels=model_config["img_channels"],
        IN_img_resolution=model_config['img_resolution'],
        internal_channels=model_config['internal_channels'],
        time_emb_dim=model_config["time_emb_dim"],
        text_emb_dim=model_config["text_emb_dim"],
        num_experts=model_config["num_experts"],
        top_k=model_config["top_k"],
        Fourier_bandwidth=model_config["fourier_bandwidth"],
        VIT_num_blocks=model_config['VIT_num_blocks'],  # Shallow ViT for demo
        VIT_patch_sizes=model_config['VIT_patch_sizes'],  # 4x4 patches for 32x32 img
        VIT_num_groups=model_config['VIT_num_groups'],
        VIT_num_heads=model_config['VIT_num_heads'],
        VIT_emb_size=model_config['VIT_emb_size'],
        VIT_emb_balance=model_config['VIT_emb_balances'],
        VIT_pos_emb_balance=model_config['VIT_pos_emb_balances'],
        Unet_num_blocks=model_config['Unet_num_blocks'],
        Unet_channel_mult=model_config['Unet_channel_mult'],
        Unet_channel_mult_emb=model_config['Unet_channel_mult_emb'],  # 2 levels for 32x32 -> 16->8
        Unet_kernel_sizes=model_config['Unet_kernel_sizes'],
        Unet_model_channels=model_config['Unet_model_channels'],
        Unet_label_balance=model_config['Unet_label_balance'],
        Unet_concat_balance=model_config['Unet_concat_balances'],
        sigma_data=model_config["sigma_data"],
        log_var_channels=model_config["log_var_channels"],
        transition_sigma=model_configs['transition_sigma'],
        transition_sharpness=model_configs['transition_sharpness'],
        pool_size=model_configs['pool_size'],
        attn_pool_balance=model_configs['attn_pool_balance']
    ).to(model_config["device"])

    #_ = register_gradient_scalers(model)
    special_params = set()
    special_params.update(model.net.Unet_experts.parameters())
    special_params.update(model.net.VIT_experts.parameters())
    special_params.update(model.net.vit_router.parameters())
    special_params.update(model.net.Unet_router.parameters())
    special_params.update(model.net.output_proj_vit.parameters())
    special_params.update(model.net.output_proj_unet.parameters())
    special_params.update(model.net.blend_residual.parameters())
    special_params.update(model.net.pooled_cross_attn.parameters())

    default_params = [p for p in model.parameters() if p not in special_params]

    optimizer = torch.optim.Adam([
        {'params': model.net.Unet_experts.parameters(), 'lr': Optim_config['lr_unet']},
        {'params': model.net.VIT_experts.parameters(), 'lr': Optim_config['lr_vit']},
        {'params': model.net.vit_router.parameters(), 'lr': Optim_config['lr_router']},
        {'params': model.net.Unet_router.parameters(), 'lr': Optim_config['lr_router']},
        {'params': model.net.output_proj_vit.parameters(), 'lr': Optim_config['lr_out_proj_vit']},
        {'params': model.net.output_proj_unet.parameters(), 'lr': Optim_config['lr_out_proj_unet']},
        {'params': model.net.blend_residual.parameters(), 'lr': Optim_config['lr_blend_residual']},
        {'params': model.net.pooled_cross_attn.parameters(), 'lr': Optim_config['lr_pooled_attn']},
        {'params': default_params, 'lr': Optim_config['LR_DEFAULT']}  # <--- THE MISSING LINK
    ],
        betas=Optim_config['betas'],
        weight_decay=Optim_config['weight_decay'],
        eps=Optim_config['eps']
    )
    # EMA for testing
    ema = EMA(model,
              beta=model_config['ema_beta'],
              warmup_power=model_config['ema_warmup_power'],
              min_warmup_step=model_config['ema_min_warmup_step']
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
                                  total_steps=model_config['total_steps'],
                                  min_active=mask_config['min_active'],
                                  step_size=mask_config['step_size'],
                                  min_bandwidth=mask_config['min_BW_u'],
                                  bandwidth=mask_config['BW'],
                                  strat_band=mask_config['strat_band'],
                                  noise_range=mask_config['unet_noise_range'],
                                  sigma_max=model_config['sigma_max'],
                                  sigma_min=model_config['sigma_min'],
                                  )

    vit_mask_gen = MaskGenerator(expert_attributes=mask_config["vit_attr"],
                                 p_mean=mask_config["p_mean"],
                                 p_std=mask_config["p_std"],
                                 total_steps=model_config['total_steps'],
                                 min_active=mask_config['min_active'],
                                 step_size=mask_config['step_size'],
                                 min_bandwidth=mask_config['min_BW_v'],
                                 bandwidth=mask_config['BW'],
                                 strat_band=mask_config['strat_band'],
                                 noise_range=mask_config['vit_noise_range'],
                                 sigma_max=model_config['sigma_max'],
                                 sigma_min=model_config['sigma_min'],
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
                             transition_sigma=model_configs['transition_sigma'],
                             overlap_width=model_configs['transition_sharpness'],
                             alpha_vit=loss_config['alpha_vit'],
                             alpha_unet=loss_config['alpha_unet'],
                             alpha_blend=loss_config['alpha_blend']
                             )

    json_log_path = os.path.join(model_config["save_dir_stats"], "training_stats.jsonl")
    print(f"Logging metrics to: {json_log_path}")
    start_step = 0

    # Check if we should resume
    if resume_path:
        checkpoint_file = None

        # Option A: Auto-find latest
        if resume_path == "latest":
            save_dir = model_configs.get("save_dir", "./checkpoints")
            # Look for ckpt_*.pt files
            list_of_files = glob(os.path.join(save_dir, "ckpt_*.pt"))
            if list_of_files:
                # Parse filenames to find the highest step number
                checkpoint_file = max(list_of_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # Option B: Specific path
        elif os.path.exists(resume_path):
            checkpoint_file = resume_path

        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"🔄 Resuming training from: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=model_config["device"], weights_only=False)

            # 1. Load Step
            start_step = checkpoint['step'] + 1  # Start from next step

            # 2. Load Model & Optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 3. Load EMA (If available in checkpoint and EMA is initialized)
            # Checking how save_checkpoint usually stores it. Assuming key 'ema' or 'model_ema'
            if ema is not None and 'ema_model_state_dict' in checkpoint:
                print("   Loading EMA state...")
                # Load the state dict directly into the internal shadow model
                ema.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                # 4. Handle Log file
            # If resuming, we probably shouldn't delete the old log file
            if os.path.exists(json_log_path):
                print(f"   Appending to existing log: {json_log_path}")

            print(f"✅ Successfully loaded. Resuming at step {start_step}")
        else:
            print(f"⚠️ Checkpoint not found at {resume_path}. Starting from scratch.")
            if os.path.exists(json_log_path):
                os.remove(json_log_path)  # Only remove log if starting fresh
    else:
        # Not resuming, clean old log
        if os.path.exists(json_log_path):
            os.remove(json_log_path)

    scheduler = Lr_Scheduler(
        optimizer=optimizer,
        total_steps=model_config["total_steps"],
        batch_size=model_config['batch_size'],
        ref_batches=Optim_config['ref_batches'],  # Adjusted for Flowers102
        rampup_Mimg=Optim_config['rampup_Mimg'],  # Warmup over 2M images
        dataset_size=model_config['dataset_size'],
        cur_nimg=start_step * model_config['batch_size']  # Flowers102 size
    )
    model.train()
    current_mse = float('inf')

    for step in range(start_step, model_config["total_steps"]):

        try:
            images_rgb, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images_rgb, _ = next(data_iter)

        images = images_rgb.to(model_config["device"])
        images = (images + 1) / 2
        images = images * 255
        latent_images = vae.encode(images.clamp(0, 255))

        sigma = sample_sigma_hybrid(batch_size=model_config['batch_size'],
                                    sigma_max=model_config['sigma_max'],
                                    sigma_min=model_config['sigma_min'],
                                    p_mean=mask_config['p_mean'],
                                    p_std=mask_config['p_std'],
                                    extreme_prob=model_config['extreme_prob'],
                                    device=model_config['device']
                                    )

        noise = torch.randn_like(latent_images).to(model_config["device"]) * sigma
        images_noised = latent_images + noise
        cur_zeta = zeta_sched.get_zeta(step=step)
        Unet_mask = Unet_mask_gen(sigma=sigma,
                                  step=step)

        vit_mask = vit_mask_gen(sigma=sigma,
                                step=step)

        out_model = model(
            x=images_noised,
            sigma=sigma,
            text_emb=None,
            Unet_router_mask=Unet_mask,
            Vit_router_mask=vit_mask,
            zeta=cur_zeta,
            return_log_var=True
        )

        loss = criterion(
            x=latent_images,
            noisy_x=images_noised,
            sigma=sigma,
            out_model=out_model
        )

        logger.log_training_step(
            step=step,
            loss_dict=loss,
            sigma=sigma,
            zeta=cur_zeta,
            log_var=out_model['log_var'].mean().item() if out_model["log_var"] is not None else 0.0,
            lr=optimizer.param_groups[0]['lr'],
            w_unet=out_model['w_unet'],
            w_vit=out_model['w_vit']
        )
        # Log router statistics
        logger.log_router_statistics(
            step=step,
            unet_probs=out_model["Unet_router_loss"],
            vit_probs=out_model["vit_router_loss"],
            sigma=sigma
        )
        logger.log_branch_specialization(
            step=step,
            vit_out=out_model["vit_denoised"],
            unet_out=out_model["unet_denoised"],
            sigma=sigma
        )
        optimizer.zero_grad()
        loss["loss"].backward()
        # loggers
        logger.log_gradients(
            step=step,
            model=model.net,
            lr_per_component=lr_per_component
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        current_mse = loss["denoising"].item()

        if step % 100 == 0 or step == start_step:
            print(f"Step {step}/{model_config['total_steps']} | "
                  f"total_Loss: {loss["loss"]:.4f} | "
                  f"MSE: {loss["denoising"]:.4f} | "
                  f"Z_loss: {loss["z_loss"]:.4f} | "
                  f"Routers_loss: {loss["balance"]:.4f} | "
                  f"entropy: {loss["entropy"]:.4f} | "
                  f"pure_loss: {loss['pure_loss']:.4f} | "
                  f"LogVar: {out_model["log_var"].mean().item() if out_model["log_var"] is not None else 0.0 :.3f}")

        if step % model_config.get("save_interval", 1000) == 0 and step > 0:
            # Create checkpoint summary log
            logger.log_checkpoint_summary(
                step=step,
                mse=current_mse,
                model=model.net,
                unet_probs=out_model["Unet_router_loss"],
                vit_probs=out_model["vit_router_loss"],
                lr_per_component=lr_per_component
            )

            save_checkpoint(
                model, optimizer, step, current_mse,
                {"model_configs": model_config,
                 "Optim_config": Optim_config,
                 "loss_config": loss_config,
                 "mask_config": mask_config,
                 "zeta_config": zeta_config},
                f"ckpt_{step}.pt",
                ema_instance=ema
            )
    save_checkpoint(model, optimizer, model_config["total_steps"], current_mse,
                    {"model_configs": model_config, "Optim_config": Optim_config,
                     "loss_config": loss_config,
                     "mask_config": mask_config,
                     "zeta_config": zeta_config}, "final_model1.pt")

    print("Training Complete.")

if __name__ == '__main__':
    training_HDMOE(model_configs,
                   optim_configs,
                   loss_configs,
                   mask_configs,
                   zeta_configs,
                   #resume_path=r"/kaggle/input/datasets/mosa1r/hdmoe-checkpoints/ckpt_26000.pt"
                   )