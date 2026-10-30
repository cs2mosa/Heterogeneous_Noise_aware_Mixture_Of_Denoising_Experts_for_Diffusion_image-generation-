import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image
from Utils.EDM_sampler import EDM_Sampler
from Utils.VAE_CLIP import StabilityVAE, CLIP_EMBED
from models.model_config_v3 import preconditioned_HDMOEM_v3
from graphs.plotter2 import Plotter
from graphs.plotter import Plotter2
from Utils.configs import model_configs as model_config
import os

def debug_trajectory_sampler(
        model,
        vae,
        device,
        output_dir="./pipeline_test",
        num_steps=50,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0
):
    """
    Simulates the EDM sampling trajectory and saves the model's
    clean image prediction (D_x) at every step.
    """
    out_path = Path(output_dir) / "trajectory_frames"
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. EDM Time Schedule
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Add t=0 at the end

    # 2. Initial Pure Noise
    x = torch.randn(1, 4, 32, 32, device=device) * t_steps[0]

    vae.init(device)
    model.eval()

    print("\n[DEBUG] Generating trajectory frames...")

    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]

        sigma_tensor = torch.full((1, 1, 1, 1), t_cur.item(), device=device)
        dummy_mask = torch.ones((1, model.net.num_experts), device=device)

        with torch.no_grad():
            # Get model prediction
            out = model(
                x=x,
                sigma=sigma_tensor,
                text_emb=None,
                Unet_router_mask=dummy_mask,
                Vit_router_mask=dummy_mask,
                zeta=0.0
            )
            denoised_latent = out["denoised"]

            # --- VISUALIZATION: Decode D_x ---
            # We decode what the model THINKS the clean image is at this step
            decoded_img = vae.decode(denoised_latent)
            if decoded_img.dtype == torch.uint8:
                decoded_img = decoded_img.float() / 255.0
            else:
                decoded_img = decoded_img.clamp(0, 1)

            # Save the frame
            filename = out_path / f"frame_{i:02d}_sigma_{t_cur.item():.2f}.png"
            save_image(decoded_img, filename)

        # Euler Integration Step
        d_i = (x - denoised_latent) / t_cur
        x = x + (t_next - t_cur) * d_i

    print(f"  ✓ Saved {num_steps} frames to {out_path}")
    print(f"  ✓ Open the folder and view them in order to watch the network 'think'.")

    return x

def test_diffusion_pipeline(
        model,
        vae,
        #clip,
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
        #clip: Your CLIP embedder
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

    with torch.no_grad():
        bs = noisy_latent.shape[0]
        num_experts = model.net.num_experts
        dummy_mask = torch.ones((bs, num_experts), device=device)
        out = model(
            x=noisy_latent,
            sigma=sigma_tensor,
            text_emb=None,
            Unet_router_mask=dummy_mask,
            Vit_router_mask=dummy_mask,
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
            text_emb=None,
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
    modeli = preconditioned_HDMOEM_v3(
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

    # 2. Load Weights
    # TODO: Update this path to match your actual checkpoint location relative to the script
    checkpoint_path = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\ckpt_113000.pt"

    print(f"Loading weights from {checkpoint_path}...")
    if os.path.exists(checkpoint_path):
        ckeck = torch.load(map_location=device, f=checkpoint_path,weights_only=False)
        modeli.load_state_dict(ckeck['ema_model_state_dict'])
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        total_params_vit = sum(p.numel() for p in modeli.net.VIT_experts.parameters())
        total_params_unet = sum(p.numel() for p in modeli.net.Unet_experts.parameters())
        print(f"Trainable Parameters_vit: {total_params_vit / 1e6:.2f}M")
        print(f"Trainable Parameters_unet: {total_params_unet / 1e6:.2f}M")
        return

    modeli.eval()
    total_params = sum(p.numel() for p in modeli.parameters())
    print(f"Trainable Parameters: {total_params / 1e6:.2f}M")
    # 3. Setup Sampler
    vae = StabilityVAE(batch_size=model_config["batch_size"],
                       raw_mean=model_config['raw_mean'],
                       raw_std=model_config['raw_std'],
                       final_mean=model_config['mean_data'],
                       final_std=model_config['sigma_data']
                       )
    vae.init(device=device)


    clip_bus = CLIP_EMBED(device=device)
    clip_bus.init()

    # Update Sampler: If you want to use guidance (e.g. 7.5), change guidance=1.0 to a higher value
    SAMPLER = EDM_Sampler(
        modeli,
        guidance=5.0,  # <<-- CHANGE: Increase for stronger prompt adherence
        num_solve_steps=100,
        sigma_min=model_configs['sigma_min'],
        sigma_max=model_configs['sigma_max'],
        rho=7.0
    )

    # 4. Generate Text Embeddings
    # <<-- ADDITION 2: Define your custom prompt and encode it -->>
    custom_prompt = "A natural landscape with a tree at the shore of a blue lake"
    print(f"Generating embeddings for: {custom_prompt}")

    # We repeat the prompt for the whole batch
    prompt_list = [custom_prompt] * model_configs["batch_size"]
    text_emb = clip_bus.encode_text(prompt_list)

    # If using guidance > 1.0, we also need an unconditional (empty) embedding
    uncond_prompt_list = [""] * model_configs["batch_size"]
    uncond_text_emb = clip_bus.encode_text(uncond_prompt_list)

    # 5. Generate Noise and Sample
    latent_noise = torch.randn(
        (model_configs["batch_size"], model_configs["img_channels"], model_configs["img_resolution"],
         model_configs["img_resolution"]),
        device=device
    )

    print("Sampling with text guidance...")
    with torch.no_grad():
        # <<-- ADDITION 3: Pass text_emb and uncond_text_emb to the sampler -->>
        latents_denoised = SAMPLER.sample(
            noise=latent_noise,
            text_emb=text_emb,              # Guided embedding
            uncond_text_emb=uncond_text_emb, # Null embedding for CFG
        )
        out_images = vae.decode(latents_denoised)

    print("Saving images...")
    output_folder = "./generated_samples_unguided"
    os.makedirs(output_folder, exist_ok=True)

    images_normalized = out_images.float() / 255.0
    images_normalized = images_normalized.clamp(0, 1)

    # using an absolute path on current os for testing, needs to be changed
    test_img_path = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\00001.jpg"

    # Check if test image exists before running test
    if os.path.exists(test_img_path):
        test_diffusion_pipeline(
            model=modeli,
            vae=vae,
            test_image_path=test_img_path,
            device=device,
            num_sampling_steps=50,
            sigma_test= 1.0
            # Pass explicit config values to test function if needed,
            # otherwise it defaults to model_configs inside the function
        )
        debug_trajectory_sampler(
            model=modeli,
            vae=vae,
            device=model_config['device'],
            num_steps= 100
        )
    else:
        print(f"Skipping pipeline test: Image not found at {test_img_path}")

    # Save grid
    save_image(images_normalized, f"{output_folder}/grid_unguided.png", nrow=4)

    # Save individual images
    for i, img in enumerate(images_normalized):
        save_image(img, f"{output_folder}/sample_{i}.png")

    # -------------------------------------------------------------------------
    # 6. UPDATED PLOTTING SECTION
    # -------------------------------------------------------------------------
    print("Generating training diagnostics...")

    # Define where the logs are located.
    # Must match the directory where 'hdmoem_flower102_v1_training.jsonl' resides.
    log_dir = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs"

    # Must match the 'run_name' used in Logger during training
    run_name = "hdmoem_flower102_v1"

    # Initialize the new Research Plotter
    plotter = Plotter(
        log_dir=log_dir,
        run_name=run_name,
        out_dir=os.path.join(output_folder, "analysis"),  # Save plots inside output folder
        smooth_sigma=5.0
    )
    slotter = Plotter2(run_name="test_run",)
    slotter.visualize_expert_landscape(model=modeli,num_points=500)
    # Generate all plots (Training, Router, Gradients, Specialization, Checkpoints)
    # Note: 'visualize_expert_landscape' is now handled automatically inside plot_router_health
    # utilizing the histogram data logged during training.
    plotter.plot_all()

    print("Done.")

if __name__ == "__main__":
    sample_and_save(model_config)