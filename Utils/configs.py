import torch

model_configs = {
        "save_dir": "./checkpoints/flower_run",
        "save_dir_stats": r"C:\Users\Maha Mamdouh\PycharmProjects\PythonProject1\Utils",
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
        'VIT_patch_sizes': [4, 8],
        'VIT_num_groups': 4,
        'VIT_num_heads': 8,
        'VIT_emb_size': 256,
        'Unet_num_blocks': 4,
        'Unet_channel_mult': [1, 2, 4, 8],
        'Unet_kernel_sizes': [(3, 3), (5, 5)],
        'Unet_model_channels': 64,
        'Unet_channel_mult_emb': 2,
        'Unet_label_balance': 0.5,
        'Unet_concat_balance': 0.5,
        'sigma_data': 0.5,
        'mean_data': 0.0,
        'raw_mean':(1.476,-1.671,0.725,2.396),
        'raw_std':(4.47,5.284,4.838,4.514),
        'log_var_channels': 32,
        'batch_size': 64,
        'total_steps': 20000,
        'sigma_min': 0.002,
        'sigma_max': 80,
        'transition_sigma': 3.0,
        'transition_sharpness': 1.0,
        "fixed_prompt": "a photo of a flower"
}
loss_configs = {
    'unet_bal': 0.04,
    'vit_bal': 0.04,
    'z_bal': 0.002,
    'prior_bal': 0.0,
    'sharpness': 3
}
optim_configs = {
    'lr_unet': 100e-4,           # 0.01 - EDM2 baseline
    'lr_vit': 100e-4,             # 0.01 - ViT experts (5x smaller)
    'lr_router': 100e-4,          # 0.005 - Routers (need faster adaptation)
    'lr_out_proj_vit': 100e-4,   # 0.01 - Output projections
    'lr_out_proj_unet': 100e-4,  # 0.01
    'lr_out_proj_fusion': 100e-4,
    'ref_batches': 12e3,  # Decay starts after ~12k batches
    'rampup_Mimg': 0.1,
}
mask_configs ={
    'unet_attr': [3, 5],
    'vit_attr': [4, 8],
    'p_mean': 0.0,  # should be tuned
    'p_std': 1.2,  # should be tuned
    'BW': 0.9,
    'max_BW': 0.3,# should be tuned
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'step',
    'unet_noise_range': (0.0 , 0.4),
    'vit_noise_range': (0.2 , 1.0)
}
zeta_configs= {
    'min_zeta': 0.001,
    'max_zeta': 2.0,
    'warmup_ratio': 0.2,
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 3000 # steepness calculation depends on this
}