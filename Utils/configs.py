import torch

model_configs = {
    "save_dir": r"./checkpoints",
    "save_dir_stats": r"./logs",
    "data_dir": "/kaggle/input/datasets/dimensi0n/lhq-1024/dataset", #add your dataset path 
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'dataset_size': 90000, # make sure to match the dataset size 
    'img_channels': 4,
    'internal_channels': 192,
    "data_img_res": 256,
    'img_resolution':32,
    'time_emb_dim': 128,
    'text_emb_dim': 0, #add 768 if going with text conditioning
    'num_experts': 2,
    'top_k': 1,
    'fourier_bandwidth': 0.9,

    # --- ViT Settings ---
    'VIT_num_blocks': 10,
    'VIT_patch_sizes': [1, 1],
    'VIT_num_groups': 4,
    'VIT_num_heads': 12,
    'VIT_emb_size': 768,
    'VIT_pos_emb_balances': [0.5, 0.5],
    'VIT_emb_balances': [0.4, 0.4],

    # --- U-Net Settings ---
    'Unet_num_blocks': 3,
    'Unet_channel_mult': [1, 2, 4],
    'Unet_kernel_sizes': [(3, 3),  (5, 5)],
    'Unet_model_channels': 128,
    'Unet_channel_mult_emb': 2,
    'Unet_label_balance': 0.5,
    'Unet_concat_balances': [0.7, 0.3],

    # --- v3 Cross-Attention ---
    'pool_size': 8,
    'attn_pool_balance': 0.5,

    # --- Physics & Data ---
    'sigma_data': 0.8,
    'mean_data': 0.0,
    'raw_mean': (0.566, -0.886, 1.47, -0.413),
    'raw_std':  (5.215, 5.469, 4.044, 3.988),
    'log_var_channels': 32,
    'batch_size': 128,
    'total_steps': 200000,
    'sigma_min': 0.002,
    'sigma_max': 80,
    "fixed_prompt": "a photo of a flower",
    'extreme_prob': 0.2,
    'transition_sigma': 2.5,  #log_scale 
    'transition_sharpness': 0.8,

    # --- EMA ---
    'ema_beta': 0.9999,
    'ema_warmup_power': 2 / 3,
    'ema_min_warmup_step': 40000
}
loss_configs = {
    'unet_bal': 0.0,
    'vit_bal': 0.0,
    'z_bal': 0.0,
    'prior_bal': 0.001,     
    'sharpness': 25,
    'laplacian_kernel': 9,
    'laplacian_std': 1.0,
    'transition_sigma': 2.5,
    'transition_sharpness': 0.8,
    'overlap_width': 0.8,
    'alpha_blend': 0.6,
    'alpha_unet': 0.2,   
    'alpha_vit': 0.2,
}

optim_configs = {
    'LR_DEFAULT': 5e-4,
    'lr_unet': 50e-4,         
    'lr_vit': 80e-4,         
    'lr_router': 10e-4,
    'lr_out_proj_vit': 5e-5,
    'lr_out_proj_unet': 5e-5,
    'lr_blend_residual': 10e-4,
    'lr_pooled_attn': 10e-4,
    'ref_batches': 70e3,
    'rampup_Mimg': 0.8,
    'betas': (0.9, 0.99),
    'weight_decay': 0.0,  
    'eps': 1e-8 
}

mask_configs = {
    'unet_attr': [3, 5],
    'vit_attr': [1, 1],
    'p_mean': -0.2,
    'p_std': 1.2,
    'max_BW_u': 0.15,   # U-Net occupies 18% of CDF. Start at 15% radius.
    'min_BW_u': 0.05,   # Shrink to 5% for strict high-noise specialization
    'max_BW_v': 0.60,   # ViT occupies 82% of CDF. Start at 60% radius.
    'min_BW_v': 0.25,   # Shrink to 25% for strict low-noise specialization
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'linear',
    'unet_noise_range': (2.5, 80.0),
    'vit_noise_range':  (0.002, 2.5),
    'total_steps': 100000,
}
zeta_configs = {
    'min_zeta': 0.001,
    'max_zeta': 2.0,
    'warmup_ratio': 0.3,
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 40000
}