import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.fft


class Diagnostic_EDM_Sampler:
    """
    Wraps the EDM_Sampler to record the ODE trajectory and spectral evolution.
    """

    def __init__(self, base_sampler):
        self.sampler = base_sampler
        self.trajectory_history = []  # Stores (sigma, D_x)

    @torch.no_grad()
    def sample_with_diagnostics(self, noise, text_emb, uncond_text_emb=None):
        self.trajectory_history = []
        device = noise.device
        dtype = self.sampler.dtype

        step_indices = torch.arange(self.sampler.num_steps, dtype=dtype, device=device)
        t_steps = (self.sampler.sigma_max ** (1 / self.sampler.rho) + step_indices / (self.sampler.num_steps - 1) *
                   (self.sampler.sigma_min ** (1 / self.sampler.rho) - self.sampler.sigma_max ** (
                               1 / self.sampler.rho))) ** self.sampler.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        x_next = noise.to(dtype) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(self.sampler.s_churn / self.sampler.num_steps, np.sqrt(
                2) - 1) if self.sampler.s_churn > 0 and self.sampler.s_min <= t_cur <= self.sampler.s_max else 0

            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.sampler.s_noise * torch.randn_like(x_cur)

            # --- DIAGNOSTIC INTERCEPT ---
            # D_x is the network's current guess of the FINAL clean image (\hat{x}_0)
            D_x = self.sampler.denoise(x_hat, t_hat, text_emb, uncond_text_emb)
            self.trajectory_history.append((t_hat.item(), D_x.detach().cpu()))
            # ----------------------------

            d_cur = (x_hat - D_x) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.sampler.num_steps - 1:
                D_x_prime = self.sampler.denoise(x_next, t_next, text_emb, uncond_text_emb)
                d_prime = (x_next - D_x_prime) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next, self.trajectory_history


def compute_radial_spectrum(latents):
    """
    Computes the Radially Averaged Power Spectral Density (RAPSD).
    latents: (B, C, H, W)
    Returns: 1D array of power across frequency bins
    """
    B, C, H, W = latents.shape

    # 1. 2D Fast Fourier Transform
    fft_2d = torch.fft.fft2(latents.float())
    # 2. Shift zero-frequency to the center
    fft_shifted = torch.fft.fftshift(fft_2d, dim=(-2, -1))
    # 3. Compute Power Spectrum (Magnitude Squared)
    power_spectrum = torch.abs(fft_shifted) ** 2
    # 4. Average over channels and batch
    mean_power = power_spectrum.mean(dim=(0, 1))  # (H, W)

    # 5. Radial Binning
    y, x = torch.meshgrid(
        torch.arange(H) - H // 2,
        torch.arange(W) - W // 2,
        indexing='ij'
    )
    r = torch.sqrt(x ** 2 + y ** 2)
    r = r.round().int()

    # Calculate mean power for each radius
    max_radius = int(r.max().item())
    radial_power = torch.zeros(max_radius + 1)

    for i in range(max_radius + 1):
        mask = (r == i)
        if mask.any():
            radial_power[i] = mean_power[mask].mean()

    return radial_power.numpy()


def analyze_trajectory(trajectory_history, transition_sigma=1.0):
    """
    trajectory_history: List of (sigma, D_x) from Diagnostic_EDM_Sampler
    Generates the diagnostic plot.
    """
    sigmas = []
    velocities = []
    spectra = []

    prev_D_x = None

    for sigma, D_x in trajectory_history:
        sigmas.append(sigma)
        spectra.append(compute_radial_spectrum(D_x))

        # Calculate L2 Velocity of the \hat{x}_0 prediction
        # If D_\theta jumps violently, this spikes.
        if prev_D_x is not None:
            velocity = torch.norm(D_x - prev_D_x, p=2).item() / (prev_D_x.shape[0])
            velocities.append(velocity)
        else:
            velocities.append(0.0)

        prev_D_x = D_x

    sigmas = np.array(sigmas)
    velocities = np.array(velocities)
    spectra = np.array(spectra)  # (Steps, Frequencies)

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    # Plot 1: Spectrogram (Frequency Evolution over Sigma)
    ax1 = fig.add_subplot(gs[0, :])
    # Normalize spectra row-wise for visibility
    spectra_norm = spectra / (spectra.max(axis=1, keepdims=True) + 1e-8)
    im = ax1.imshow(spectra_norm.T, aspect='auto', origin='lower',
                    cmap='magma', extent=[len(sigmas) - 1, 0, 0, spectra.shape[1]])

    ax1.set_xticks(np.arange(len(sigmas))[::5])
    ax1.set_xticklabels([f"{s:.2f}" for s in sigmas[::-1][::5]])  # High to Low
    ax1.set_xlabel("Noise Level ($\sigma$) [High $\to$ Low]")
    ax1.set_ylabel("Spatial Frequency (0 = Layout, Max = Texture)")
    ax1.set_title("Spectral Evolution of $\hat{x}_0$ Prediction over ODE Integration")

    # Mark the Router Handoff
    handoff_step = np.argmin(np.abs(sigmas[::-1] - transition_sigma))
    ax1.axvline(x=handoff_step, color='cyan', linestyle='--', label=f'MoE Handoff ($\sigma={transition_sigma}$)')
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Normalized Power')

    # Plot 2: ODE Trajectory Velocity (Lipschitz Monitor)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(sigmas[::-1], velocities[::-1], color='red', marker='o')
    ax2.axvline(x=transition_sigma, color='cyan', linestyle='--')
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.set_xlabel("Noise Level ($\sigma$)")
    ax2.set_ylabel("$L_2$ Velocity of $\hat{x}_0$")
    ax2.set_title("Prediction Stability (Watch for spikes at Handoff!)")

    # Plot 3: Low vs High Frequency Trackers
    ax3 = fig.add_subplot(gs[1, 1])
    low_freq_power = spectra[:, :5].mean(axis=1)  # Structural features
    high_freq_power = spectra[:, -15:].mean(axis=1)  # Textural features

    ax3.plot(sigmas[::-1], low_freq_power[::-1], color='blue', label='Low Freq (Structure)')
    ax3.plot(sigmas[::-1], high_freq_power[::-1], color='orange', label='High Freq (Texture)')
    ax3.axvline(x=transition_sigma, color='cyan', linestyle='--')
    ax3.set_xscale('log')
    ax3.invert_xaxis()
    ax3.set_yscale('log')
    ax3.set_xlabel("Noise Level ($\sigma$)")
    ax3.set_ylabel("Absolute Power")
    ax3.set_title("Frequency Band Assembly")
    ax3.legend()

    plt.savefig("trajectory_diagnostics.png", dpi=300, bbox_inches='tight')
    print("Saved trajectory_diagnostics.png")