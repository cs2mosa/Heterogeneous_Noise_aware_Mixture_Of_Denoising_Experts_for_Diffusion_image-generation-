import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from Utils.EDM_sampler import EDM_Sampler
from Utils.VAE_CLIP import StabilityVAE ,CLIP_EMBED
from models.model_config import preconditioned_HDMOEM
from graphs.plotter import Plotter
from Utils.configs import model_configs as model_config
import os

def test_diffusion_pipeline(
        model,
        vae,
        clip,
        test_image_path: str,
        output_dir: str = "./pipeline_test",
        device: str = "cuda",
        sigma_test: float =1.0,
        sampler_sigma_min: float = 0.002,
        sampler_sigma_max: float = 80.0,
        num_sampling_steps: int = 30,
):
    """
    Complete end-to-end test of the diffusion pipeline.

    Args:
        model: Your diffusion model
        vae: Your VAE (StabilityVAE instance)
        clip: Your CLIP embedder
        test_image_path: Path to a test flower image
        output_dir: Where to save results
        device: 'cuda' or 'cpu'
        sigma_test: Noise level to test (default 1.0 = medium noise)
        sampler_sigma_min: Minimum sigma for full sampler (default 0.002)
        sampler_sigma_max: Maximum sigma for full sampler (default 80.0)
        num_sampling_steps: Number of sampling steps for full diffusion (default 50)

    Returns:
        dict with diagnostic info
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure models are in eval mode
    model.eval()
    vae.init(device)
    clip.init()

    results = {}

    print("\n" + "=" * 80)
    print("DIFFUSION PIPELINE DIAGNOSTIC TEST")
    print("=" * 80)

    # ===== STEP 1: LOAD ORIGINAL IMAGE =====
    print("\n[1/6] Loading original image...")
    from PIL import Image
    import torchvision.transforms as T

    img = Image.open(test_image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    img_tensor = (transform(img).unsqueeze(0) * 255.0).to(device)

    save_image(img_tensor / 255.0, output_dir / "step1_original.png")

    results['original'] = {
        'shape': img_tensor.shape,
        'min': img_tensor.min().item(),
        'max': img_tensor.max().item(),
        'mean': img_tensor.mean().item(),
    }

    print(f"  ✓ Original image: shape={img_tensor.shape}")
    print(f"    Range: [{img_tensor.min():.1f}, {img_tensor.max():.1f}]")
    print(f"    Saved: step1_original.png")

    # ===== STEP 2: VAE ENCODE =====
    print("\n[2/6] Encoding with VAE...")

    with torch.no_grad():
        latent = vae.encode(img_tensor)

    results['encoded_latent'] = {
        'shape': latent.shape,
        'mean': latent.mean().item(),
        'std': latent.std().item(),
        'min': latent.min().item(),
        'max': latent.max().item(),
    }

    print(f"  ✓ Encoded latent: shape={latent.shape}")
    print(f"    Mean: {latent.mean():.3f}, Std: {latent.std():.3f}")
    print(f"    Expected: shape=(1,4,32,32), mean≈0, std≈0.5")

    # Check if latent stats are reasonable
    if abs(latent.mean().item()) > 1.0:
        print(f"  ⚠️  WARNING: Latent mean {latent.mean():.3f} is far from 0!")
    if latent.std().item() < 0.1 or latent.std().item() > 2.0:
        print(f"  ⚠️  WARNING: Latent std {latent.std():.3f} is unusual!")

    # Visualize latent channels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(latent[0, i].cpu(), cmap='viridis')
        axes[i].set_title(f'Latent Channel {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'step2_latent_channels.png', dpi=150)
    plt.close()
    print(f"    Saved: step2_latent_channels.png")

    # ===== STEP 3: VAE DECODE (No diffusion) =====
    print("\n[3/6] Decoding latent (testing VAE round-trip)...")

    with torch.no_grad():
        reconstructed = vae.decode(latent)

    # Handle uint8 vs float
    if reconstructed.dtype == torch.uint8:
        reconstructed_float = reconstructed.float() / 255.0
    else:
        reconstructed_float = reconstructed.clamp(0, 1)

    save_image(reconstructed_float, output_dir / "step3_vae_reconstruction.png")

    results['vae_reconstruction'] = {
        'dtype': str(reconstructed.dtype),
        'min': reconstructed_float.min().item(),
        'max': reconstructed_float.max().item(),
    }

    print(f"  ✓ VAE reconstruction: dtype={reconstructed.dtype}")
    print(f"    Range: [{reconstructed_float.min():.3f}, {reconstructed_float.max():.3f}]")
    print(f"    Saved: step3_vae_reconstruction.png")
    print(f"  ⚠️  CRITICAL CHECK: Does step3_vae_reconstruction.png have grid artifacts?")
    print(f"      If YES → VAE decoder is broken")
    print(f"      If NO → Problem is elsewhere")

    # ===== STEP 4: ADD NOISE =====
    print(f"\n[4/6] Adding noise (σ={sigma_test})...")

    noise = torch.randn_like(latent) * sigma_test
    noisy_latent = latent + noise

    results['noisy_latent'] = {
        'mean': noisy_latent.mean().item(),
        'std': noisy_latent.std().item(),
    }

    print(f"  ✓ Noisy latent: mean={noisy_latent.mean():.3f}, std={noisy_latent.std():.3f}")

    # Decode noisy version for comparison
    with torch.no_grad():
        noisy_decoded = vae.decode(noisy_latent)
    if noisy_decoded.dtype == torch.uint8:
        noisy_decoded = noisy_decoded.float() / 255.0
    else:
        noisy_decoded = noisy_decoded.clamp(0, 1)

    save_image(noisy_decoded, output_dir / "step4_noisy.png")
    print(f"    Saved: step4_noisy.png (should look corrupted)")

    # ===== STEP 5: MODEL DENOISE =====
    print("\n[5/6] Denoising with trained model...")

    # Prepare inputs
    sigma_tensor = torch.full((1, 1, 1, 1), sigma_test, device=device)
    text_emb = clip.encode_text(["a photo of a flower"])

    with torch.no_grad():
        out = model(
            x=noisy_latent,
            sigma=sigma_tensor,
            text_emb=text_emb,
            Unet_router_mask=None,
            Vit_router_mask=None,
            zeta=0.0,
        )
        denoised_latent = out["denoised"]

    results['denoised_latent'] = {
        'mean': denoised_latent.mean().item(),
        'std': denoised_latent.std().item(),
        'min': denoised_latent.min().item(),
        'max': denoised_latent.max().item(),
    }

    print(f"  ✓ Denoised latent: mean={denoised_latent.mean():.3f}, std={denoised_latent.std():.3f}")

    # Check if denoising moved toward original
    dist_noisy_to_original = (noisy_latent - latent).pow(2).mean().sqrt().item()
    dist_denoised_to_original = (denoised_latent - latent).pow(2).mean().sqrt().item()

    results['denoising_quality'] = {
        'noisy_to_original_dist': dist_noisy_to_original,
        'denoised_to_original_dist': dist_denoised_to_original,
        'improvement': dist_noisy_to_original - dist_denoised_to_original,
    }

    print(f"    Distance to original:")
    print(f"      Noisy:    {dist_noisy_to_original:.4f}")
    print(f"      Denoised: {dist_denoised_to_original:.4f}")

    if dist_denoised_to_original < dist_noisy_to_original:
        improvement = ((dist_noisy_to_original - dist_denoised_to_original) / dist_noisy_to_original) * 100
        print(f"    ✓ Model improved by {improvement:.1f}%")
    else:
        print(f"    ❌ WARNING: Model made it WORSE! Not learning to denoise!")

    # Decode denoised
    with torch.no_grad():
        denoised_img = vae.decode(denoised_latent)
    if denoised_img.dtype == torch.uint8:
        denoised_img = denoised_img.float() / 255.0
    else:
        denoised_img = denoised_img.clamp(0, 1)

    save_image(denoised_img, output_dir / "step5_denoised.png")
    print(f"    Saved: step5_denoised.png")

    # ===== STEP 6: FULL SAMPLER TEST =====
    print(f"\n[6/6] Testing full sampler (from pure noise)...")
    print(f"    Sampler config: σ_min={sampler_sigma_min}, σ_max={sampler_sigma_max}, steps={num_sampling_steps}")

    sampler = EDM_Sampler(
        model=model,
        num_solve_steps=num_sampling_steps,
        sigma_min=sampler_sigma_min,
        sigma_max=sampler_sigma_max,
        guidance=1.0,
    )

    # Generate from pure noise
    pure_noise = torch.randn(1, 4, 32, 32, device=device)

    with torch.no_grad():
        sampled_latent = sampler.sample(
            noise=pure_noise,
            text_emb=text_emb,
            uncond_text_emb=None,
        )

    results['sampled_latent'] = {
        'mean': sampled_latent.mean().item(),
        'std': sampled_latent.std().item(),
    }

    print(f"  ✓ Sampled latent: mean={sampled_latent.mean():.3f}, std={sampled_latent.std():.3f}")
    print(f"    Expected: mean≈0, std≈0.4")

    if abs(sampled_latent.mean().item()) > 0.5:
        print(f"    ⚠️  WARNING: Sampled mean {sampled_latent.mean():.3f} is unusual!")
    if sampled_latent.std().item() > 2.0:
        print(f"    ⚠️  WARNING: Sampled std {sampled_latent.std():.3f} is too high!")

    # Decode sampled
    with torch.no_grad():
        sampled_img = vae.decode(sampled_latent)
    if sampled_img.dtype == torch.uint8:
        sampled_img = sampled_img.float() / 255.0
    else:
        sampled_img = sampled_img.clamp(0, 1)

    save_image(sampled_img, output_dir / "step6_sampled_from_noise.png")
    print(f"    Saved: step6_sampled_from_noise.png")
    print(f"  ⚠️  This should resemble a flower (even if blurry)")

    # ===== CREATE COMPARISON FIGURE =====
    print("\n[7/7] Creating comparison figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_tensor[0].cpu().permute(1, 2, 0) / 255.0)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reconstructed_float[0].cpu().permute(1, 2, 0))
    axes[0, 1].set_title('VAE Reconstruction\n(No Diffusion)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(noisy_decoded[0].cpu().permute(1, 2, 0))
    axes[0, 2].set_title(f'Noisy (σ={sigma_test})', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(denoised_img[0].cpu().permute(1, 2, 0))
    axes[1, 0].set_title('Model Denoised\n(Single Step)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sampled_img[0].cpu().permute(1, 2, 0))
    axes[1, 1].set_title('Sampled from Noise\n(Full Diffusion)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Add metrics text
    metrics_text = f"""LATENT STATISTICS
                        Original: μ={results['encoded_latent']['mean']:.3f}, σ={results['encoded_latent']['std']:.3f}
                        Noisy: μ={results['noisy_latent']['mean']:.3f}, σ={results['noisy_latent']['std']:.3f}
                        Denoised: μ={results['denoised_latent']['mean']:.3f}, σ={results['denoised_latent']['std']:.3f}
                        Sampled: μ={results['sampled_latent']['mean']:.3f}, σ={results['sampled_latent']['std']:.3f}
                        
                        DENOISING QUALITY
                        Distance (noisy→original): {results['denoising_quality']['noisy_to_original_dist']:.4f}
                        Distance (denoised→original): {results['denoising_quality']['denoised_to_original_dist']:.4f}
                        Improvement: {results['denoising_quality']['improvement']:.4f}
                        
                        SAMPLER CONFIG
                        σ_min: {sampler_sigma_min}
                        σ_max: {sampler_sigma_max}
                        steps: {num_sampling_steps}
                    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'full_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: full_comparison.png")

    # ===== FINAL REPORT =====
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print("\n✓ FILES SAVED:")
    print(f"  • step1_original.png")
    print(f"  • step2_latent_channels.png")
    print(f"  • step3_vae_reconstruction.png  ← CHECK FOR GRID ARTIFACTS")
    print(f"  • step4_noisy.png")
    print(f"  • step5_denoised.png")
    print(f"  • step6_sampled_from_noise.png  ← CHECK IF LOOKS LIKE FLOWER")
    print(f"  • full_comparison.png")

    print("\n⚠️  CRITICAL CHECKS:")
    print(f"  1. VAE Reconstruction (step3):")
    if abs(results['encoded_latent']['mean']) > 1.0:
        print(f"     ❌ Latent mean {results['encoded_latent']['mean']:.3f} is far from 0")
    else:
        print(f"     ✓ Latent statistics look reasonable")
    print(f"     → Does step3_vae_reconstruction.png have grid artifacts?")
    print(f"       If YES: VAE decoder is broken")
    print(f"       If NO: Problem is in diffusion model or data")

    print(f"\n  2. Model Denoising (step5):")
    if results['denoising_quality']['improvement'] > 0:
        print(f"     ✓ Model reduced distance by {results['denoising_quality']['improvement']:.4f}")
    else:
        print(f"     ❌ Model INCREASED distance (not learning!)")

    print(f"\n  3. Full Sampling (step6):")
    if abs(results['sampled_latent']['mean']) < 0.5 and results['sampled_latent']['std'] < 1.0:
        print(f"     ✓ Sampled latent statistics look reasonable")
    else:
        print(f"     ❌ Sampled latent statistics are unusual")
        print(f"        Mean={results['sampled_latent']['mean']:.3f}, Std={results['sampled_latent']['std']:.3f}")
        print(f"        This usually means sampler diverged")
    print(f"     → Does step6_sampled_from_noise.png look like a flower?")
    print(f"       If YES: Model is working!")
    print(f"       If NO: Check if σ_min/σ_max match training range")

    print("\n" + "=" * 80)

    return results

def sample_and_save(model_configs):
    """
    Initializes model from configs, loads specific checkpoint, and generates samples.
    """
    device = model_configs["device"]

    # Load Model using imported model_configs
    print("Loading model structure...")
    modeli = preconditioned_HDMOEM(
        IN_in_channels=model_configs["img_channels"],
        IN_img_resolution=model_configs['img_resolution'],
        internal_channels=model_configs['internal_channels'],
        time_emb_dim=model_configs["time_emb_dim"],
        text_emb_dim=model_configs["text_emb_dim"],
        num_experts=model_configs["num_experts"],
        top_k=model_configs["top_k"],
        Fourier_bandwidth=model_configs['fourier_bandwidth'],
        VIT_num_blocks=model_configs['VIT_num_blocks'],
        VIT_patch_sizes=model_configs['VIT_patch_sizes'],
        VIT_num_groups=model_configs['VIT_num_groups'],
        VIT_num_heads=model_configs['VIT_num_heads'],
        VIT_emb_size=model_configs['VIT_emb_size'],
        Unet_num_blocks=model_configs['Unet_num_blocks'],
        Unet_channel_mult=model_configs['Unet_channel_mult'],
        Unet_channel_mult_emb=model_configs['Unet_channel_mult_emb'],
        Unet_kernel_sizes=model_configs['Unet_kernel_sizes'],
        Unet_model_channels=model_configs['Unet_model_channels'],
        sigma_data=model_configs["sigma_data"],
        log_var_channels=model_configs['log_var_channels']
    ).to(device)

    # Note: Using absolute path from original snippet.
    # Ideally this path should also be in configs or passed as argument.
    # I'm using here my absolute path just for testing,it needs to be changed
    checkpoint_path = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\ckpt_8000.pt"

    print(f"Loading weights from {checkpoint_path}...")
    if os.path.exists(checkpoint_path):
        ckeck = torch.load(map_location=device, f=checkpoint_path,weights_only=False)
        modeli.load_state_dict(ckeck['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")

    modeli.eval()

    clip = CLIP_EMBED()
    clip.init()

    vae = StabilityVAE(batch_size=model_configs["batch_size"])
    vae.init(device=device)

    # Initialize sampler with config parameters
    SAMPLER = EDM_Sampler(
        modeli,
        guidance=1.0,
        num_solve_steps=500,  # or use a value from configs if available
        sigma_min=model_configs['sigma_min'],
        sigma_max=model_configs['sigma_max'],
        rho=15.0
    )

    # Prepare conditioning
    fixed_prompts = [model_configs["fixed_prompt"]] * model_configs["batch_size"]
    with torch.no_grad():
        base_text_emb = clip.encode_text(fixed_prompts)

    # Generate noise
    latent_noise = torch.randn(
        (model_configs["batch_size"], model_configs["img_channels"], model_configs["img_resolution"],
         model_configs["img_resolution"]),
        device=device
    )

    print("Sampling...")
    with torch.no_grad():
        # Corrected: usage of transition_point and softness from model_configs
        latents_denoised = SAMPLER.sample(
            noise=latent_noise,
            text_emb=base_text_emb,
        )
        out_images = vae.decode(latents_denoised)

    # Save
    print("Saving images...")
    output_folder = "generated_samples_unguided"
    os.makedirs(output_folder, exist_ok=True)

    images_normalized = out_images.float() / 255.0
    images_normalized = images_normalized.clamp(0, 1)

    # using an absolute path on current os for testing, needs to be changed
    test_img_path = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\data\flowers-102\jpg\image_00002.jpg"

    # Check if test image exists before running test
    if os.path.exists(test_img_path):
        """test_diffusion_pipeline(
            model=modeli,
            vae=vae,
            clip=clip,
            test_image_path=test_img_path,
            device=device,
            num_sampling_steps=500,
            # Pass explicit config values to test function if needed,
            # otherwise it defaults to model_configs inside the function
        )"""
    else:
        print(f"Skipping pipeline test: Image not found at {test_img_path}")

    # Save grid
    save_image(images_normalized, f"{output_folder}/grid_unguided.png", nrow=4)

    # Initialize Plotter (paths kept from original)
    plotter = Plotter(run_name="test_run")

    # needs to be changed , just for testing we will keep it like this
    base_log_path = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs"
    train_log = os.path.join(base_log_path, "hdmoem_flower102_v1_training.jsonl")
    grad_log = os.path.join(base_log_path, "hdmoem_flower102_v1_gradients.jsonl")
    route_log = os.path.join(base_log_path, "hdmoem_flower102_v1_router_stats.jsonl")

    plotter.plot_training_overview(train_log)
    plotter.plot_scaling_gating_mechanisms(train_log)
    plotter.plot_gradient_health(grad_log)
    plotter.plot_router_stats(route_log)
    plotter.visualize_expert_landscape(model=modeli,device='cuda',num_points=400)

    # Save individual images
    for i, img in enumerate(images_normalized):
        save_image(img, f"{output_folder}/sample_{i}.png")

    print("Done.")

def sample_sigma_phased(step:int,
                        phases:tuple[int,int],
                        batch_size:int,
                        low_period: tuple[float, float] = (0.002, 15),
                        high_period: tuple[float, float] = (15, 80),
                        p_mean: float = -0.4,
                        p_std: float = -1.0,
                        extreme_prob :float= 0.2,
                        device:str='cuda'
                        ):
    # Phase 1: 0 -> 1000 (Structural Emphasis)
    if step < phases[0]:
        # Sample only from the high-noise "tail" of the log-uniform
        # Focus on sigma [5.0, 80.0]
        log_min, log_max = math.log(high_period[0]), math.log(high_period[1])
        u = torch.rand([batch_size, 1, 1, 1], device=device)
        sigma = (u * (log_max - log_min) + log_min).exp()

    # Phase 2: 1000 -> 2000 (Detail Emphasis)
    elif step < phases[1]:
        # Focus on sigma [0.002, 1.0]
        log_min, log_max = math.log(low_period[0]), math.log(low_period[1])
        u = torch.rand([batch_size, 1, 1, 1], device=device)
        sigma = (u * (log_max - log_min) + log_min).exp()

    # Phase 3: 2000+ (Full Coordination)
    else:
        # Back to your Hybrid Sampler (80% Log-Normal, 20% Uniform)
        sigma = sample_sigma_hybrid(batch_size,
                                    sigma_min=low_period[0],
                                    sigma_max=high_period[1],
                                    p_mean=p_mean,  # Adjusted for Latent Space
                                    p_std=p_std,  # Adjusted for Latent Space
                                    extreme_prob=extreme_prob,
                                    device=device)

    return sigma
# --- 1. Your Function (Pasted Here) ---
def sample_sigma_hybrid(batch_size,
                        sigma_min=0.002,
                        sigma_max=80.0,
                        p_mean=-0.4,
                        p_std=1.0,
                        extreme_prob=0.2,
                        device='cpu'):  # Changed default to CPU for plotting
    """
    Hybrid Sampler: Ensures 'Core' training via Log-Normal (80%)
    and 'Expert Coverage' via Log-Uniform (20%).
    """
    n_lognormal = int(batch_size * (1 - extreme_prob))
    n_uniform = batch_size - n_lognormal

    # Log-Normal Samples
    rnd_normal = torch.randn([n_lognormal, 1, 1, 1], device=device)
    sigma_lognormal = (rnd_normal * p_std + p_mean).exp()

    # Log-Uniform Samples
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    u = torch.rand([n_uniform, 1, 1, 1], device=device)
    sigma_uniform = (u * (log_max - log_min) + log_min).exp()

    # Combine
    sigma = torch.cat([sigma_lognormal, sigma_uniform], dim=0)
    sigma = sigma.clamp(sigma_min, sigma_max)
    perm = torch.randperm(batch_size, device=device)
    return sigma[perm]


# --- 2. Visualization Logic ---

def visualize_sampler():
    # Settings
    BATCH_SIZE = 10000  # Large batch to see the distribution clearly
    SIGMA_MIN = 0.002
    SIGMA_MAX = 80.0
    P_MEAN = 0.0
    P_STD = 1.4
    EXTREME_PROB = 0.5

    # Generate Data
    print(f"Generating {BATCH_SIZE} samples...")
    sigmas_tensor = sample_sigma_hybrid(BATCH_SIZE, SIGMA_MIN, SIGMA_MAX,
                                        P_MEAN, P_STD, EXTREME_PROB, device='cpu')
    #sigmas_tensor = sample_sigma_phased(3150,(1000,2000),BATCH_SIZE,(0.002,5.0),(20,80),P_MEAN,P_STD,EXTREME_PROB,'cpu')
    # Flatten and convert to numpy
    sigmas = sigmas_tensor.flatten().numpy()

    # Calculate Log values for better visualization
    log_sigmas = np.log(sigmas)

    # --- PLOTTING ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Hybrid Sampler Analysis (Batch N={BATCH_SIZE})', fontsize=16)

    # Plot 1: Log-Space Histogram (The most important view)
    # This reveals the Gaussian Curve + The Uniform Floor
    sns.histplot(log_sigmas, bins=100, kde=True, ax=axes[0, 0], color='teal', stat='density')
    axes[0, 0].set_title('Log-Space Distribution (Natural Log)', fontsize=14)
    axes[0, 0].set_xlabel('ln(sigma)')
    axes[0, 0].axvline(np.log(SIGMA_MIN), color='r', linestyle='--', label='Min Limit')
    axes[0, 0].axvline(np.log(SIGMA_MAX), color='r', linestyle='--', label='Max Limit')
    axes[0, 0].axvline(P_MEAN, color='gold', linestyle='-', linewidth=2, label='Log-Normal Mean')
    axes[0, 0].legend()
    axes[0, 0].text(0.05, 0.95, "Look for: Bell curve riding\non top of a flat floor",
                    transform=axes[0, 0].transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Plot 2: Actual Values (Log Scale Axis)
    # This shows the values as the network sees them, but spaced logarithmically
    sns.histplot(sigmas, bins=100, ax=axes[0, 1], color='purple', log_scale=True)
    axes[0, 1].set_title('Actual Sigma Values (Log Scale X-Axis)', fontsize=14)
    axes[0, 1].set_xlabel('Sigma')

    # Plot 3: Scatter Plot of First 300 samples (Verify Shuffling)
    subset = 300
    axes[1, 0].scatter(range(subset), sigmas[:subset], alpha=0.6, s=15)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title(f'Sequence Mix Check (First {subset} samples)', fontsize=14)
    axes[1, 0].set_ylabel('Sigma (Log Scale)')
    axes[1, 0].set_xlabel('Batch Index')
    axes[1, 0].text(0.05, 0.9, "Should look like random noise\n(No patterns)",
                    transform=axes[1, 0].transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Plot 4: Cumulative Distribution Function (CDF)
    sns.ecdfplot(sigmas, ax=axes[1, 1], color='darkblue', log_scale=True)
    axes[1, 1].set_title('CDF (Coverage Probability)', fontsize=14)
    axes[1, 1].set_xlabel('Sigma')
    axes[1, 1].grid(True, which="both", ls="-", alpha=0.2)

    # Add statistics text
    stats_text = (
        f"Stats:\n"
        f"Mean: {np.mean(sigmas):.4f}\n"
        f"Median: {np.median(sigmas):.4f}\n"
        f"Min: {np.min(sigmas):.4f}\n"
        f"Max: {np.max(sigmas):.4f}\n"
        f"Std Dev: {np.std(sigmas):.4f}"
    )
    fig.text(0.92, 0.5, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=(0, 0, 0.9, 0.95))
    plt.show()


"""sigma = sample_sigma_phased(step=step,
                            phases=(2000, 3000),
                            batch_size=model_config['batch_size'],
                            low_period=(model_config['sigma_min'], 5),
                            high_period=(20, model_config['sigma_max']),
                            p_mean=mask_config['p_mean'],
                            p_std=mask_config['p_std'],
                            extreme_prob=model_config['sigma_extreme_prob'],
                            device=model_config["device"]
                            )"""

if __name__ == "__main__":
    #visualize_sampler()
    sample_and_save(model_config)