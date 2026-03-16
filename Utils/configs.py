import torch

model_configs = {
    "save_dir": r"/kaggle/working/checkpoints",
    "save_dir_stats": r"/kaggle/working/logs",
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'dataset_size': 8189, #change to new dataset's size
    'img_channels': 4,
    'internal_channels': 192,
    "data_img_res": 256,
    'img_resolution': 32,
    'time_emb_dim': 256,
    'text_emb_dim': 0, # add size (typically 768 for CLIP text embedding) if working with conditional image generation
    'num_experts': 2,
    'top_k': 1,
    'fourier_bandwidth': 0.9,

    # --- ViT Settings ---
    'VIT_num_blocks': 12, #high capacity VITs
    'VIT_patch_sizes': [1, 2],  # [CRITICAL] 1024 tokens & 256 tokens.
    'VIT_num_groups': 4,
    'VIT_num_heads': 8,
    'VIT_emb_size': 512,
    'VIT_pos_emb_balances': [0.5, 0.5],
    'VIT_emb_balances': [0.4, 0.4],

    # --- U-Net Settings ---
    'Unet_num_blocks': 6, #high capacity Unets
    'Unet_channel_mult': [1, 2, 2, 4, 4, 4],
    'Unet_kernel_sizes': [(3, 3), (5, 5)],
    'Unet_model_channels': 192,
    'Unet_channel_mult_emb': 2,
    'Unet_label_balance': 0.5,
    'Unet_concat_balances': [0.7, 0.3],

    # --- v3 Cross-Attention ---
    'pool_size': 4,
    'attn_pool_balance': 0.5,

    # --- Physics & Data ---
    'sigma_data': 0.8, #can be changed for the new dataset
    'mean_data': 0.0, #can be changed for the new dataset
    'raw_mean': (1.476, -1.671, 0.725, 2.396), #critical: needs to be recalculated for the new dataset (latent mean)
    'raw_std': (4.47, 5.284, 4.838, 4.514), #critical: needs to be recalculated for the new dataset (latent std)
    'log_var_channels': 128,
    'batch_size': 2048, #can be lowered but be sure to use gradient accumulation for a reasonable effective batch_size (2048 to 4096)
    'total_steps': 500000,
    'sigma_min': 0.002,
    'sigma_max': 80,
    "fixed_prompt": "a photo of a flower", #make it dynamic for conditional generation
    'extreme_prob': 0.2,
    'transition_sigma': 1.5,
    'transition_sharpness': 1.0,

    # --- EMA ---
    'ema_beta': 0.999,
    'ema_warmup_power': 2 / 3,
    'ema_min_warmup_step': 2000
}

loss_configs = {
    'unet_bal': 0.05, #be sure to adjust if it is strong
    'vit_bal': 0.05, #be sure to adjust if it is strong
    'z_bal': 0.002,
    'prior_bal': 0.02, #be sure to adjust if it is strong
    'sharpness': 3,
    'laplacian_kernel': 9,
    'laplacian_std': 1.0,
    'transition_sigma': 1.5,
    'transition_sharpness': 1.0,
    'overlap_width': 1.0,
    'alpha_blend': 0.5,  # Balanced blend
    'alpha_unet': 0.4,
    'alpha_vit': 0.4,
}

optim_configs = {
    'LR_DEFAULT': 10e-4,
    'lr_unet': 100e-4,
    'lr_vit': 100e-4,
    'lr_router': 10e-4,
    'lr_out_proj_vit': 10e-4,
    'lr_out_proj_unet': 10e-4,
    'lr_blend_residual': 10e-4,
    'lr_pooled_attn': 10e-4,
    'ref_batches': 400e3,
    'rampup_Mimg': 2.5,
    'betas': (0.9, 0.99),
    'weight_decay': 0.0,
    'eps': 1e-6
}

mask_configs = {
    'unet_attr': [3, 5],
    'vit_attr': [1, 2],  # Matches model_configs patch sizes
    'p_mean': -0.4, #can be adjusted for the new dataset
    'p_std': 1.2,#can be adjusted for the new dataset
    'BW': 0.5,
    'min_BW_u': 0.30,
    'min_BW_v': 0.12,
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'linear',

    # [CRITICAL] Safely overlaps the 62nd percentile transition point (sigma = 1.5)
    'unet_noise_range': (0.10, 0.70),
    'vit_noise_range': (0.50, 0.98),
}

zeta_configs = {
    'min_zeta': 0.001,
    'max_zeta': 2.0,
    'warmup_ratio': 0.2,
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 100e3
}
