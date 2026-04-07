"""
Professional Analysis & Visualization Toolkit
=============================================
Compatible with Enhanced Logger & HD-MoE Configs
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from typing import Optional
import warnings
import math
from Utils.configs import model_configs

warnings.filterwarnings('ignore')


class Plotter2:
    """
    Research-grade visualization toolkit for Hybrid Diffusion MoE models.
    """

    def __init__(self, run_name: str = "experiment", output_dir: str = "./analysis_results"):
        self.run_name = run_name
        self.output_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set publication-quality plotting style
        sns.set_theme(style="whitegrid", context="paper")
        sns.set_palette("husl")

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })

        print(f"✓ Initialized Plotter for: {run_name}")
        print(f"✓ Output directory: {self.output_dir}")

    # =========================================================
    # 1. TRAINING OVERVIEW (Losses, LR, Zeta)
    # =========================================================
    def plot_training_overview(self, log_file_path: str, smooth_window: int = 50):
        """
        Visualizes the fundamental training metrics: Loss, LR, Zeta, and Reconstruction.
        """
        df = self._parse_jsonl(log_file_path)
        if df is None: return

        # Rename columns to standard readable names if they exist
        rename_map = {
            'step': 'Step',
            'loss': 'Total Loss',
            'denoising': 'MSE',
            'pure_loss': 'EDM Loss',
            'balance': 'Aux: Balance',
            'z_loss': 'Aux: Z-Loss',
            'entropy': 'Aux: Entropy',
            'lr': 'Learning Rate',
            'zeta': 'Zeta',
            'log_var': 'Log Var'
        }
        df.rename(columns=rename_map, inplace=True)

        # Apply smoothing
        metrics_to_smooth = ['Total Loss', 'MSE', 'EDM Loss', 'Aux: Balance', 'Log Var']
        df_smooth = df.copy()

        # Adaptive window size based on data length
        actual_window = min(smooth_window, len(df) // 10) if len(df) > 100 else 1

        for m in metrics_to_smooth:
            if m in df.columns:
                df_smooth[m] = gaussian_filter1d(df[m], sigma=max(1, actual_window // 5))

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)

        # --- 1. Main Loss Components ---
        ax1 = fig.add_subplot(gs[0, :2])
        if 'Total Loss' in df.columns:
            ax1.plot(df['Step'], df['Total Loss'], alpha=0.2, color='gray')
            ax1.plot(df['Step'], df_smooth['Total Loss'], label='Total Loss', color='black', linewidth=2)
        if 'EDM Loss' in df.columns:
            ax1.plot(df['Step'], df_smooth['EDM Loss'], label='Pure EDM Loss', color='tab:blue', linestyle='--')
        ax1.set_title('Global Loss Evolution')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # --- 2. Reconstruction (MSE) ---
        ax2 = fig.add_subplot(gs[0, 2])
        if 'MSE' in df.columns:
            ax2.semilogy(df['Step'], df['MSE'], alpha=0.2, color='tab:green')
            ax2.semilogy(df['Step'], df_smooth['MSE'], color='tab:green', label='MSE (Recon)')
            ax2.set_title('Reconstruction Error (Log Scale)')
            ax2.legend()

        # --- 3. MoE Aux Losses ---
        ax3 = fig.add_subplot(gs[1, 0])
        aux_cols = ['Aux: Balance', 'Aux: Z-Loss', 'Aux: Entropy']
        for col in aux_cols:
            if col in df.columns:
                ax3.plot(df['Step'], df[col], label=col, alpha=0.8)
        ax3.set_title('Router Auxiliary Losses')
        ax3.set_yscale('log')
        ax3.legend()

        # --- 4. Uncertainty (Log Var) ---
        ax4 = fig.add_subplot(gs[1, 1])
        if 'Log Var' in df.columns:
            ax4.plot(df['Step'], df['Log Var'], color='tab:purple', alpha=0.3)
            ax4.plot(df['Step'], df_smooth['Log Var'], color='tab:purple', linewidth=2)
            ax4.set_title('Uncertainty (Predicted Log Variance)')

        # --- 5. Zeta (Exploration) ---
        ax5 = fig.add_subplot(gs[1, 2])
        if 'Zeta' in df.columns:
            ax5.plot(df['Step'], df['Zeta'], color='tab:orange', linewidth=2)
            ax5.fill_between(df['Step'], 0, df['Zeta'], color='tab:orange', alpha=0.2)
            ax5.set_title('Router Exploration Noise (Zeta)')

        # --- 6. Learning Rate ---
        ax6 = fig.add_subplot(gs[2, 0])
        if 'Learning Rate' in df.columns:
            ax6.plot(df['Step'], df['Learning Rate'], color='tab:red')
            ax6.set_title('Learning Rate Schedule')

        # --- 7. Data Distribution (Noise Sampling) ---
        ax7 = fig.add_subplot(gs[2, 1])
        if 'avg_sigma_percentile' in df.columns:
            ax7.hist(df['avg_sigma_percentile'], bins=30, color='teal', alpha=0.7)
            ax7.set_title('Sigma Sampling Distribution (Avg Percentile)')
            ax7.set_xlabel('Percentile (0=Clean, 1=Pure Noise)')

        # --- 8. Stats Summary ---
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        final_row = df.iloc[-1]
        stats = [
            ['Metric', 'Final Value'],
            ['Loss', f"{final_row.get('Total Loss', 0):.4f}"],
            ['MSE', f"{final_row.get('MSE', 0):.5f}"],
            ['Zeta', f"{final_row.get('Zeta', 0):.4f}"]
        ]
        table = ax8.table(cellText=stats, loc='center', cellLoc='center', bbox=[0.1, 0.3, 0.8, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table[0, 0].set_facecolor('#d1d1d1')
        table[0, 1].set_facecolor('#d1d1d1')

        plt.suptitle(f"Training Dynamics: {self.run_name}", fontsize=16, fontweight='bold')
        plt.savefig(f"{self.output_dir}/01_training_overview.png")
        print(f"✓ Saved: 01_training_overview.png")
        plt.close()

    # =========================================================
    # 2. SCALING & GATING MECHANISMS (HD-MoE Specific)
    # =========================================================
    def plot_scaling_gating_mechanisms(self, log_file_path: str):
        """
        Visualizes how the model scales UNet vs ViT branches based on noise levels.
        """
        print("Analyzing Scaling & Gating...")
        df = self._parse_jsonl(log_file_path)
        if df is None: return

        required_keys = ['sample_sigma_percentile', 'sample_vit_scale', 'sample_unet_scale']
        if not all(k in df.columns for k in required_keys):
            print(f"⚠️ Missing keys for scaling analysis. Found: {df.columns.tolist()}")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)

        # --- 1. Scaling Evolution (Global Means) ---
        ax1 = fig.add_subplot(gs[0, :])
        if 'scaling_vit_mean' in df.columns:
            ax1.plot(df['step'], df['scaling_vit_mean'], label='ViT Branch Scale', color='#1f77b4')
            ax1.plot(df['step'], df['scaling_unet_mean'], label='UNet Branch Scale', color='#ff7f0e')
            ax1.plot(df['step'], df['scaling_fusion_mean'], label='Fusion Scale', color='#2ca02c', linestyle=':')
            ax1.set_title('Global Path Scaling Factors (Moving Average)')
            ax1.legend()
            ax1.set_ylim(0, 1.2)

        # --- 2. UNet Scaling vs Noise Level ---
        ax2 = fig.add_subplot(gs[1, 0])
        # Scatter plot colored by training step to show convergence
        sc1 = ax2.scatter(df['sample_sigma_percentile'], df['sample_unet_scale'],
                          c=df['step'], cmap='magma_r', s=10, alpha=0.5)
        ax2.set_xlabel('Noise Percentile (0=Clean, 1=Noise)')
        ax2.set_ylabel('UNet Scaling Factor')
        ax2.set_title('UNet Specialization\n(Should be high at Low Noise)')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.1, 1.1)
        plt.colorbar(sc1, ax=ax2, label='Training Step')

        # --- 3. ViT Scaling vs Noise Level ---
        ax3 = fig.add_subplot(gs[1, 1])
        sc2 = ax3.scatter(df['sample_sigma_percentile'], df['sample_vit_scale'],
                          c=df['step'], cmap='viridis', s=10, alpha=0.5)
        ax3.set_xlabel('Noise Percentile (0=Clean, 1=Noise)')
        ax3.set_ylabel('ViT Scaling Factor')
        ax3.set_title('ViT Specialization\n(Should be high at High Noise)')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(-0.1, 1.1)
        plt.colorbar(sc2, ax=ax3, label='Training Step')

        # --- 4. Gating Weight Correlation ---
        ax4 = fig.add_subplot(gs[1, 2])
        if 'sample_gate_wx' in df.columns:
            ax4.scatter(df['sample_unet_scale'], df['sample_gate_wx'],
                        label='UNet Gate', alpha=0.3, s=10, color='orange')
            ax4.scatter(df['sample_vit_scale'], df['sample_gate_wa'],
                        label='ViT Gate', alpha=0.3, s=10, color='blue')
            ax4.set_xlabel('Scaling Factor Input')
            ax4.set_ylabel('Learned Gate Weight')
            ax4.set_title('Consistency: Scaling vs Gating')
            ax4.legend()

        plt.suptitle(f"Hybrid Mechanism Analysis: {self.run_name}", fontsize=16, fontweight='bold')
        plt.savefig(f"{self.output_dir}/02_scaling_mechanisms.png")
        print(f"✓ Saved: 02_scaling_mechanisms.png")
        plt.close()

    # =========================================================
    # 3. ROUTER STATISTICS (From Logs)
    # =========================================================
    def plot_router_stats(self, router_log_path: str):
        """
        Visualizes expert usage, collapse metrics (Gini), and entropy from the router logs.
        """
        print("Analyzing Router Statistics...")
        df = self._parse_jsonl(router_log_path)
        if df is None: return

        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35)

        # --- 1. Entropy (Diversity) ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['step'], df['unet_entropy'], label='UNet Entropy', color='orange')
        ax1.plot(df['step'], df['vit_entropy'], label='ViT Entropy', color='blue')
        # Ideal entropy line (ln(num_experts))
        num_experts = model_configs.get('num_experts', 2)
        max_ent = np.log(num_experts)
        ax1.axhline(max_ent, color='red', linestyle='--', label='Max Entropy')
        ax1.set_title('Router Entropy (Higher is Better)')
        ax1.legend()

        # --- 2. Gini Coefficient (Collapse) ---
        ax2 = fig.add_subplot(gs[0, 1])
        # Gini 0 = perfect balance, 1 = total collapse
        ax2.plot(df['step'], df['unet_gini'], label='UNet Gini', color='orange')
        ax2.plot(df['step'], df['vit_gini'], label='ViT Gini', color='blue')
        ax2.set_title('Gini Coefficient (Lower is Better)')
        ax2.set_ylim(0, 1)
        ax2.legend()

        # --- 3. Dead Experts ---
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(df['step'], df['unet_dead_experts'], label='UNet Dead', linestyle=':', color='orange')
        ax3.plot(df['step'], df['vit_dead_experts'], label='ViT Dead', linestyle=':', color='blue')
        ax3.set_title('Count of Dead Experts')
        ax3.set_yticks(range(num_experts + 1))
        ax3.legend()

        # --- 4. Expert Usage Distribution (Latest) ---
        ax4 = fig.add_subplot(gs[1, :])

        # Get latest usage
        if 'unet_expert_usage' in df.columns:
            try:
                latest_unet = df.iloc[-1]['unet_expert_usage']
                latest_vit = df.iloc[-1]['vit_expert_usage']

                # Ensure they are lists
                if isinstance(latest_unet, str): latest_unet = json.loads(latest_unet)
                if isinstance(latest_vit, str): latest_vit = json.loads(latest_vit)

                x = np.arange(len(latest_unet))
                width = 0.35

                ax4.bar(x - width / 2, latest_unet, width, label='UNet Experts', color='orange', alpha=0.7)
                ax4.bar(x + width / 2, latest_vit, width, label='ViT Experts', color='blue', alpha=0.7)

                ax4.set_title(f'Final Expert Utilization (Step {df.iloc[-1]["step"]})')
                ax4.set_xticks(x)
                ax4.set_xticklabels([f"Exp {i}" for i in x])
                ax4.set_ylabel('Avg Probability')
                ax4.legend()
            except Exception as e:
                ax4.text(0.5, 0.5, f"Error parsing usage data: {str(e)}", ha='center')

        plt.suptitle(f"Router Health Analysis: {self.run_name}", fontsize=16, fontweight='bold')
        plt.savefig(f"{self.output_dir}/03_router_stats.png")
        print(f"✓ Saved: 03_router_stats.png")
        plt.close()

    # =========================================================
    # 4. GRADIENT & UPDATE HEALTH
    # =========================================================
    def plot_gradient_health(self, grad_log_path: str):
        """
        Visualizes Gradient Norms and Update Ratios.
        """
        print("Analyzing Gradient Health...")
        df = self._parse_jsonl(grad_log_path)
        if df is None: return

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3)

        # --- 1. Gradient Norms (Log Scale) ---
        ax1 = fig.add_subplot(gs[0, 0])
        # Identify norm columns
        norm_cols = [c for c in df.columns if 'grad_norm' in c]

        # Group by component type for cleaner plotting
        unet_cols = [c for c in norm_cols if 'Unet' in c]
        vit_cols = [c for c in norm_cols if 'VIT' in c or 'vit' in c]

        # Plot mean norms if too many columns
        if unet_cols:
            ax1.semilogy(df['step'], df[unet_cols].mean(axis=1), label='UNet Avg Norm', color='orange')
        if vit_cols:
            ax1.semilogy(df['step'], df[vit_cols].mean(axis=1), label='ViT Avg Norm', color='blue')

        ax1.set_title('Average Gradient Norms')
        ax1.legend()

        # --- 2. Router Specific Gradients ---
        ax2 = fig.add_subplot(gs[0, 1])
        router_cols = [c for c in norm_cols if 'router' in c]
        for col in router_cols:
            ax2.semilogy(df['step'], df[col], label=col.replace('_grad_norm', ''), alpha=0.8)
        ax2.set_title('Router Gradient Norms')
        ax2.legend()

        # --- 3. Update Ratios (The Goldilocks Plot) ---
        ax3 = fig.add_subplot(gs[1, :])
        ratio_cols = [c for c in df.columns if 'update_ratio' in c]

        for col in ratio_cols:
            # Smooth lines for clarity
            smoothed = gaussian_filter1d(df[col], sigma=5)
            # Shorten name
            label = col.replace('_update_ratio', '').replace('_experts', '')
            ax3.semilogy(df['step'], smoothed, label=label, alpha=0.6)

        # The Goldilocks Zone (1e-3)
        ax3.axhline(1e-3, color='green', linestyle='--', linewidth=2, label='Ideal (1e-3)')
        ax3.axhline(1e-4, color='red', linestyle=':', alpha=0.5)
        ax3.axhline(1e-2, color='red', linestyle=':', alpha=0.5)

        ax3.set_title('Update Ratios (Step Size / Weight Magnitude)')
        ax3.set_ylabel('Ratio (Log Scale)')
        ax3.legend(loc='upper right', ncol=2)

        plt.suptitle(f"Training Health (Gradients): {self.run_name}", fontsize=16, fontweight='bold')
        plt.savefig(f"{self.output_dir}/04_gradient_health.png")
        print(f"✓ Saved: 04_gradient_health.png")
        plt.close()

    # =========================================================
    # 5. EXPERT LANDSCAPE (Inference based, using Configs)
    # =========================================================
    def visualize_expert_landscape(self, model, device='cuda', num_points=500):
        """
        Runs a batched forward pass on Parametric Routers to visualize
        the Gaussian RBF activation landscape.
        """
        print("Generating Expert Landscape Visualization...")
        model.eval()

        # Handle unwrapping of DDP or EMA models
        net = model.net if hasattr(model, 'net') else model
        net = net.module if hasattr(net, 'module') else net

        num_experts = model_configs.get('num_experts', 2)
        sigma_min = model_configs.get('sigma_min', 0.002)
        sigma_max = model_configs.get('sigma_max', 80.0)

        # 1. Generate a smooth logarithmic spectrum of sigma values
        sigmas = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), num_points).to(device)

        # 2. Extract Log Sigma for the Parametric Router
        # Shape: (num_points, 1)
        log_sigmas = torch.log(sigmas).view(-1, 1)

        # 3. Batched Forward Pass (No Fourier needed anymore!)
        with torch.no_grad():
            # Returns: sparse_weights, probs, logits
            _, unet_probs, _ = net.Unet_router(log_sigma=log_sigmas, mask=None, zeta=0.0)
            _, vit_probs, _ = net.vit_router(log_sigma=log_sigmas, mask=None, zeta=0.0)

        # Move to CPU for plotting
        unet_act = unet_probs.cpu().numpy()  # Shape: (num_points, num_experts)
        vit_act = vit_probs.cpu().numpy()  # Shape: (num_points, num_experts)
        sigmas_np = sigmas.cpu().numpy()

        # 4. Plotting (Research-Grade Line Plots to view Gaussian intersections)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        colors = ['#E07B39', '#4A90D9', '#2ECC71', '#9B59B6']  # Expandable if you use >2 experts

        # UNet Router Plot
        for i in range(num_experts):
            axes[0].plot(sigmas_np, unet_act[:, i], label=f'UNet Expert {i}',
                         color=colors[i % len(colors)], linewidth=2.5)
            # Fill under the curve for better readability
            axes[0].fill_between(sigmas_np, 0, unet_act[:, i], color=colors[i % len(colors)], alpha=0.2)

        axes[0].set_title('UNet Router: Temporal Expert Specialization (High-Noise Fenced)', fontsize=14,
                          fontweight='bold')
        axes[0].set_ylabel('Routing Probability', fontsize=12)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, which="both", ls="--", alpha=0.5)
        axes[0].legend(loc='upper right')

        # ViT Router Plot
        for i in range(num_experts):
            axes[1].plot(sigmas_np, vit_act[:, i], label=f'ViT Expert {i}',
                         color=colors[i % len(colors)], linewidth=2.5)
            axes[1].fill_between(sigmas_np, 0, vit_act[:, i], color=colors[i % len(colors)], alpha=0.2)

        axes[1].set_title('ViT Router: Temporal Expert Specialization (Low-Noise Fenced)', fontsize=14,
                          fontweight='bold')
        axes[1].set_ylabel('Routing Probability', fontsize=12)
        axes[1].set_xlabel('Noise Level (\u03c3)', fontsize=12)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, which="both", ls="--", alpha=0.5)
        axes[1].legend(loc='upper right')

        # Apply true Log Scale to the shared X-axis
        plt.xscale('log')
        plt.xlim(sigma_min, sigma_max)

        # Invert X-axis (Optional, but standard for Diffusion as generation goes High -> Low sigma)
        plt.gca().invert_xaxis()

        plt.tight_layout()

        # Ensure output directory exists (adapt to your exact class paths)
        out_path = f"{self.out_dir}/05_expert_landscape.png" if hasattr(self, 'out_dir') else "05_expert_landscape.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"✓ Saved: {out_path}")
    # =========================================================
    # UTILS
    # =========================================================
    @staticmethod
    def _parse_jsonl(file_path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return pd.DataFrame(data)
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None


if __name__ == "__main__":
    # Test block
    # Assuming standard paths based on configs
    run = "hdmoem_flower102_v1"  # example

    plotter = Plotter2(run_name="test_run")

    # You can point these to your actual log files
    plotter.plot_training_overview(r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs\hdmoem_flower102_v1_training.jsonl")
    plotter.plot_scaling_gating_mechanisms(r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs\hdmoem_flower102_v1_training.jsonl")
    plotter.plot_gradient_health(r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs\hdmoem_flower102_v1_gradients.jsonl")
    plotter.plot_router_stats(r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs\hdmoem_flower102_v1_router_stats.jsonl")