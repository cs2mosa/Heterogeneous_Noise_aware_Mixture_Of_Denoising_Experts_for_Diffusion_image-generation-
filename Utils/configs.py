"""
Updated Configs for Fixed Architecture

Key changes:
1. Stronger load balancing (10x increase)
2. Weaker z-loss (was too aggressive)
3. Stronger path prior
4. REVERSED masking curriculum (start wide, end narrow)
5. Added transition_sigma config
"""

import torch

model_configs = {
        "save_dir": "./checkpoints/flower_run_fixed",
        "save_dir_stats": r"./stats_analysis",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'img_channels': 4,
        'internal_channels': 192,
        "data_img_res": 256,
        'img_resolution': 32,
        'time_emb_dim': 128,
        'text_emb_dim': 768,
        'num_experts': 2,
        'top_k': 1,
        'fourier_bandwidth': 0.9,
        'VIT_num_blocks': 8,
        'VIT_patch_sizes': [4,8],  # Keep your large patches for ViT
        'VIT_num_groups': 4,
        'VIT_num_heads': 6,
        'VIT_emb_size': 192,
        'Unet_num_blocks': 5,
        'Unet_channel_mult': [1, 2, 4, 4, 8],
        'Unet_kernel_sizes': [(3, 3), (5, 5)],  # Keep small kernels for UNet
        'Unet_model_channels': 64,
        'Unet_channel_mult_emb': 2,
        'Unet_label_balance': 0.5,
        'Unet_concat_balance': 0.5,
        'sigma_data': 0.5, #can be tuned for each dataset's std
        'mean_data': 0.0,
        'raw_mean': (1.476, -1.671, 0.725, 2.396), #mean of 4 latent channels for Flower102 dataset , need to be calculated for each dataset used
        'raw_std': (4.47, 5.284, 4.838, 4.514), #std of 4 latent channels for Flower102 dataset , need to be calculated for each dataset used
        'log_var_channels': 32,
        'batch_size': 64,
        'total_steps': 100000,
        'sigma_min': 0.002,
        'sigma_max': 80,
        "fixed_prompt": "a photo of a flower", #will be derived from the dataset, now we use unconditional generation
        
        # NEW: Trajectory decomposition settings
        'transition_sigma': 3.0,       # Handoff point: σ > 3 → ViT, σ < 3 → UNet
        'transition_sharpness': 1.5,   # Blend smoothness (higher = sharper transition)
}

loss_configs = {
    'unet_bal': 0.03,
    'vit_bal': 0.03,
    'z_bal': 0.002,
    'prior_bal': 0.0,
    'sharpness': 3,
    'laplacian_kernel': 9,
    'laplacian_std': 1.2, #standard deviation of gaussian blurring in LPD (higher -> harder work for Unet)
    'transition_sigma': 3.0,  # Match model config
    'overlap_width': 1.5,  # Smooth transition region
    'alpha_blend': 0.4,  # Blended output weight -> only used for EDM_LOSS_NOD
    'alpha_unet': 0.3,  # UNet trajectory weight -> only used for EDM_LOSS_NOD
    'alpha_vit': 0.3,  # ViT trajectory weight -> only used for EDM_LOSS_NOD
}

optim_configs = {
    'lr_unet': 50e-4,           # 0.01 - EDM2 baseline
    'lr_vit': 50e-4,             # 0.01 - ViT experts (5x smaller)
    'lr_router': 100e-4,          # 0.005 - Routers (need faster adaptation)
    'lr_out_proj_vit': 50e-4,   # 0.01 - Output projections
    'lr_out_proj_unet': 50e-4,  # 0.01
    'lr_out_proj_fusion': 50e-4,
    'ref_batches':8e3,  # Decay starts after ~6k batches (steps)
    'rampup_Mimg': 0.15, # increasing the LR over the course of (ramp up_img->0.15e6 / batch_size->64) = 2344 steps
}

mask_configs = {
    'unet_attr': [3, 5],
    'vit_attr': [4, 8],
    'p_mean': 0.0,  # should be tuned accordingly
    'p_std': 1.3,  # should be tuned accordingly
    'BW': 0.9, #full coverage in beginning of training
    'max_BW_u': 0.25,# narrow for fine-tuning the weight dist. of Unet experts with slight overlap
    'max_BW_v': 0.45,# narrow for fine-tuning the weight dist. of Vit experts with slight overlap
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'step', # try linear
    'unet_noise_range': (0.0 , 0.4),
    'vit_noise_range': (0.2 , 1.0)
}

zeta_configs = {
    'min_zeta': 0.001,
    'max_zeta': 2.0,
    'warmup_ratio': 0.2,
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 2000 # steepness calculation depends on this
}
