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
        'VIT_patch_sizes': [4,8],  # Keep your large patches for ViT
        'VIT_num_groups': 4,
        'VIT_num_heads': 8,
        'VIT_emb_size': 256,
        'Unet_num_blocks': 5,
        'Unet_channel_mult': [1, 2, 4, 4, 8],
        'Unet_kernel_sizes': [(3, 3), (5, 5)],  # Keep small kernels for UNet
        'Unet_model_channels': 64,
        'Unet_channel_mult_emb': 2,
        'Unet_label_balance': 0.5,
        'Unet_concat_balance': 0.5,
        'sigma_data': 0.5,
        'mean_data': 0.0,
        'raw_mean': (1.476, -1.671, 0.725, 2.396),
        'raw_std': (4.47, 5.284, 4.838, 4.514),
        'log_var_channels': 32,
        'batch_size': 256,
        'total_steps': 100000,
        'sigma_min': 0.002,
        'sigma_max': 80,
        "fixed_prompt": "a photo of a flower",
        
        # NEW: Trajectory decomposition settings
        'transition_sigma': 3.0,       # Handoff point: σ > 3 → ViT, σ < 3 → UNet
        'transition_sharpness': 1.5,   # Blend smoothness (higher = sharper transition)
}

loss_configs = {
    'unet_bal': 0.05,      # 10x increase (was 0.005) - PREVENT COLLAPSE
    'vit_bal': 0.05,       # 10x increase
    'z_bal': 0.001,        # 20x decrease (was 0.02) - Was too aggressive
    'prior_bal': 0.0,      # 3x increase (was 0.03) - Guide specialization
    'sharpness': 3.0       # Decrease (was 5) - Softer priors
}

optim_configs = {
    'lr_unet': 100e-4,           
    'lr_vit': 100e-4,             
    'lr_attn': 100e-4,          # Not used (no cross-attention)
    'lr_router': 100e-4,          
    'lr_gate': 100e-4,           # Not used (deterministic blending)
    'lr_out_proj_vit': 100e-4,   
    'lr_out_proj_unet': 100e-4,  
    'lr_out_proj_fusion': 100e-4, # Not used
    'ref_batches': 12e3,  
    'rampup_Mimg': 0.1,
}

mask_configs = {
    'unet_attr': [3, 5],
    'vit_attr': [4, 8],
    'p_mean': 0.0,  
    'p_std': 1.2,  
    
    # REVERSED CURRICULUM: Start wide, end narrow
    'BW': 0.9,              # Start WIDE (was 0.05) - Let experts see everything
    'max_BW': 0.3,          # End narrow (was 0.2) - Lock in specialization
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'step',   # Step-wise annealing
    
    # Noise ranges (for expert center placement)
    'unet_noise_range': (0.0, 0.4),   # UNet experts centered in low-noise region
    'vit_noise_range': (0.2, 1.0)     # ViT experts centered in high-noise region
}

zeta_configs = {
    'min_zeta': 0.001,
    'max_zeta': 2.0,
    'warmup_ratio': 0.2,
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 5000
}
