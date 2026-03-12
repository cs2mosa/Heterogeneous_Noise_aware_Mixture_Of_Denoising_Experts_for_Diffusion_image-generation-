import torch

model_configs = {
    "save_dir":          r"./checkpoints/",
    "save_dir_stats":    r"./logs",
    'device':            "cuda" if torch.cuda.is_available() else "cpu",
    'dataset_size':      1281167,       # ImageNet-1K Training Set size
    "data_img_res":      512,           # Raw Image Resolution
    'img_resolution':    64,            # Latent space (512 / 8 = 64)
    'img_channels':      4,             # VAE latent channels
    'internal_channels': 256,           # Increased from 128 to handle 1000 classes
    'time_emb_dim':      512,           # Increased time embedding capacity
    'text_emb_dim':      768,           # CRITICAL: ImageNet requires class conditioning! (CLIP ViT-L dimension)
    'num_experts':       2,
    'top_k':             1,
    'fourier_bandwidth': 0.9,
    'VIT_num_blocks':    12,            # Up from 4. 12 is standard for "Base" ViTs
    'VIT_patch_sizes':   [2, 4],        #[CRITICAL] 64/2 = 32^2 = 1024 tokens (Exp0). 64/4 = 16^2 = 256 tokens (Exp1).
    'VIT_num_groups':    32,            # Standard for GroupNorm
    'VIT_num_heads':     12,            # Standard for 768 emb_size
    'VIT_emb_size':      768,           # Massive capacity jump to learn global ImageNet layouts
    'VIT_pos_emb_balances':[0.4, 0.1],
    'VIT_emb_balances':[0.4, 0.2],
    'Unet_num_blocks':      4,          # Keep at 3 ResBlocks per resolution
    'Unet_channel_mult':[1, 2, 4, 4], # [CRITICAL] Added a 4th level! 64->32->16->8
    'Unet_kernel_sizes':[(3, 3), (5, 5)],
    'Unet_model_channels':  192,        # Standard ADM base channels
    'Unet_channel_mult_emb': 4,         # Internal embedding expansion
    'Unet_label_balance':    0.5,
    'Unet_concat_balances':[0.7, 0.3],
    'pool_size':         8,             # [CRITICAL] Increased from 4. A 64x64 latent needs an 8x8 coarse grid (64 tokens) for the UNet/ViT to communicate.
    'attn_pool_balance': 0.5,
    'sigma_data':  0.8,                 # VAE standard deviation
    'mean_data':   0.0,
    'raw_mean':    (1.476, -1.671, 0.725, 2.396),
    'raw_std':     (4.47, 5.284, 4.838, 4.514),
    'log_var_channels': 64,
    'batch_size':  256,                 # [WARNING] You MUST use Gradient Accumulation to hit 256. (e.g., Micro-batch 32 * 4 Accumulations * 2 GPUs).
    'total_steps': 500000,              # ImageNet takes a LONG time. 500k is the minimum for state-of-the-art.
    'sigma_min':   0.002,
    'sigma_max':   80,
    "fixed_prompt": "A high resolution photo of an object", # Or use class names dynamically!
    'extreme_prob': 0.2,
    'transition_sigma':     1.5,        # Keep the exact same physics transition
    'transition_sharpness': 1.0,
}
optim_configs = {
    'LR_DEFAULT': 2e-4,  # Lowered slightly for the massive model size to prevent divergence
    'lr_unet': 1e-4,  # Safe start with GradBooster
    'lr_vit': 1e-4,
    'lr_router': 2e-4,
    'lr_out_proj_vit': 1e-4,
    'lr_out_proj_unet': 1e-4,
    'lr_blend_residual': 1e-4,
    'lr_pooled_attn': 1e-4,
    'ref_batches': 200e3,  # Slows down the inverse-sqrt decay (lasts until step 200k)
    'rampup_Mimg': 10.0,  # Warmup over 10 Million images (Wait 39k steps before hitting max LR)

    'betas': (0.9, 0.99),
    'weight_decay': 0,
    'eps': 1e-6  # Keeps MP gradients stable
}
zeta_configs = {
    'min_zeta': 0.001,
    'max_zeta': 2.0,
    'warmup_ratio': 0.1,               # Pure random exploration for the first 50,000 steps
    "strategy": 'cos',
    'alpha': 4.0,
    'total_schedule_steps': 100000     # Slowly decay exploration from step 50k to 150k. Deterministic after 150k.
}
mask_configs = {
    'unet_attr': [3, 5],
    'vit_attr': [1, 2],                 # Patch sizes updated to match model configs
    'p_mean': -0.4,
    'p_std': 1.2,
    'BW': 0.5,
    'min_BW_u': 0.30,
    'min_BW_v': 0.12,
    'min_active': 1,
    'step_size': 0.1,
    'strat_band': 'linear',
    'unet_noise_range': (0.10, 0.70),   # KEEP THESE. They perfectly wrap transition_sigma=1.5
    'vit_noise_range':  (0.50, 0.98),
}
loss_configs = {
    'unet_bal':  0.08,                  # Increased slightly to ensure strict load balancing across 500k steps
    'vit_bal':   0.08,
    'z_bal':     0.002,
    'prior_bal': 0.0,
    'sharpness': 3,
    'transition_sigma': 1.5,
    'transition_sharpness': 1.0,
    'overlap_width': 1.0,
    'alpha_blend': 0.5,
    'alpha_unet':  0.4,
    'alpha_vit':   0.4,
}