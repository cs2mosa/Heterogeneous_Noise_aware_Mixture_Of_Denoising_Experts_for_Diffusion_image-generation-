"""
HDMOEM Research Plotter
=======================
Research-grade visualization toolkit aligned to HDMOEMLogger output files.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'lines.linewidth':  1.5,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
})

# ── Constants ──────────────────────────────────────────────────────────────────
IDEAL_UPDATE_RATIO   = 1e-3
UPDATE_WARN_HIGH     = 1e-2
UPDATE_WARN_LOW      = 1e-4
MIN_ENTROPY_NORM     = 0.85
MAX_GINI             = 0.25
SIGMA_BIN_EDGES      = [0.002, 0.02, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 80.0]
SIGMA_BIN_LABELS     = ['0.002–0.02', '0.02–0.1', '0.1–0.5', '0.5–1',
                        '1–3', '3–10', '10–30', '30–80']

C_UNET   = '#E07B39'
C_VIT    = '#4A90D9'
C_BLEND  = '#7B68EE'
C_TOTAL  = '#2C2C2C'
C_GOOD   = '#2ECC71'
C_WARN   = '#E74C3C'
C_NEUTRAL= '#95A5A6'


def _smooth(series: pd.Series, sigma: float = 5.0) -> np.ndarray:
    """
    Gaussian smooth.
    FIXED: Interpolates NaNs instead of dropping them to preserve array length.
    Ensures x and y dimensions match during plotting.
    """
    if series.empty:
        return np.array([])
    # Convert to numeric, replace non-numeric with NaN
    s = pd.to_numeric(series, errors='coerce')
    # Interpolate to fill gaps, keeping the original index structure
    s_filled = s.interpolate(method='linear', limit_direction='both').fillna(0.0)
    arr = s_filled.values.astype(float)
    if len(arr) < 4:
        return arr
    return gaussian_filter1d(arr, sigma=max(1.0, sigma))


def _parse_jsonl(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"  [skip] file not found: {path}")
        return None
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        print(f"  [skip] empty file: {path}")
        return None
    return pd.DataFrame(rows)


def _add_health_band(ax, y_low, y_high, label=None, alpha=0.08):
    ax.axhspan(y_low, y_high, color=C_GOOD, alpha=alpha, label=label, zorder=0)


class Plotter:
    def __init__(self,
                 log_dir:  str,
                 run_name: str,
                 out_dir:  str = "./analysis",
                 smooth_sigma: float = 8.0,
                 ):
        self.log_dir    = Path(log_dir)
        self.run_name   = run_name
        self.out_dir    = Path(out_dir) / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.smooth     = smooth_sigma

        self._logs = {
            "training":       str(self.log_dir / f"{run_name}_training.jsonl"),
            "router":         str(self.log_dir / f"{run_name}_router.jsonl"),
            "gradients":      str(self.log_dir / f"{run_name}_gradients.jsonl"),
            "specialization": str(self.log_dir / f"{run_name}_specialization.jsonl"),
            "checkpoints":    str(self.log_dir / f"{run_name}_checkpoints.jsonl"),
        }

        print(f"[HDMOEMPlotter] run: {run_name}")
        print(f"  logs:   {self.log_dir}")
        print(f"  output: {self.out_dir}")

    def _save(self, fig: plt.Figure, filename: str):
        path = self.out_dir / filename
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ saved: {path.name}")

    def plot_training_overview(self):
        print("\n[1] Training Overview")
        df = _parse_jsonl(self._logs["training"])
        if df is None:
            return

        steps = df['step']
        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30)

        # (a) Loss
        ax = fig.add_subplot(gs[0, :2])
        if 'loss' in df.columns:
            ax.plot(steps, df['loss'], color=C_NEUTRAL, alpha=0.25, linewidth=0.8)
            ax.plot(steps, _smooth(df['loss'], self.smooth), color=C_TOTAL, linewidth=2.0, label='Total Loss')
        if 'pure_loss' in df.columns:
            ax.plot(steps, _smooth(df['pure_loss'], self.smooth), color=C_VIT, linestyle='--', label='Pure EDM Loss')
        if 'loss_texture' in df.columns:
            ax.plot(steps, _smooth(df['loss_texture'], self.smooth), color=C_UNET, linestyle=':', alpha=0.8, label='UNet Branch')
        if 'loss_structure' in df.columns:
            ax.plot(steps, _smooth(df['loss_structure'], self.smooth), color=C_VIT, linestyle=':', alpha=0.8, label='ViT Branch')
        ax.set_title('(a) Loss Decomposition')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)

        # (b) MSE
        ax = fig.add_subplot(gs[0, 2])
        if 'denoising' in df.columns:
            ax.semilogy(steps, df['denoising'], color=C_NEUTRAL, alpha=0.3, linewidth=0.8)
            ax.semilogy(steps, _smooth(df['denoising'], self.smooth), color=C_GOOD, linewidth=2.0, label='MSE')
            ax.axhline(0.10, color=C_WARN, linestyle=':', linewidth=1, label='Target ~0.10')
        ax.set_title('(b) Reconstruction Error (log scale)')
        ax.set_ylabel('MSE')
        ax.legend(fontsize=9)

        # (c) Zones
        ax = fig.add_subplot(gs[1, :2])
        zone_cols = {
            'zone_vit_dominant':  (C_VIT,   'ViT-dominant'),
            'zone_overlap':       (C_BLEND,  'Overlap zone'),
            'zone_unet_dominant': (C_UNET,   'UNet-dominant'),
        }
        plotted = False
        for col, (color, label) in zone_cols.items():
            if col in df.columns:
                ax.plot(steps, _smooth(df[col], self.smooth), color=color, label=label, linewidth=1.5)
                plotted = True
        if plotted:
            ax.set_ylim(0, 1)
            ax.set_ylabel('Fraction of Batch')
            ax.set_title('(c) Trajectory Zone Coverage per Batch')
            ax.legend(fontsize=9)
        else:
            ax.set_title('(c) Trajectory Zone Coverage (Missing data)')

        # (d) Blend Weights
        ax = fig.add_subplot(gs[1, 2])
        if 'blend_w_unet_mean' in df.columns:
            ax.plot(steps, _smooth(df['blend_w_unet_mean'], self.smooth), color=C_UNET, label='Mean w_unet')
        if 'blend_w_vit_mean' in df.columns:
            ax.plot(steps, _smooth(df['blend_w_vit_mean'], self.smooth), color=C_VIT, label='Mean w_vit')
        ax.axhline(0.5, color=C_NEUTRAL, linestyle=':', linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title('(d) Blend Weight Evolution')
        ax.legend(fontsize=9)

        # (e) Aux Losses
        ax = fig.add_subplot(gs[2, 0])
        for col, label, color in [('balance', 'Balance', C_WARN), ('z_loss', 'Z-Loss', C_UNET), ('entropy', 'Entropy', C_VIT)]:
            if col in df.columns:
                ax.semilogy(steps, _smooth(df[col], self.smooth), label=label, color=color)
        ax.set_title('(e) Auxiliary Losses (log scale)')
        ax.legend(fontsize=9)

        # (f) Schedule
        ax = fig.add_subplot(gs[2, 1])
        if 'lr' in df.columns:
            ax.plot(steps, df['lr'], color=C_WARN, label='LR')
            ax.set_ylabel('Learning Rate', color=C_WARN)
            ax.tick_params(axis='y', labelcolor=C_WARN)
        ax2 = ax.twinx()
        if 'zeta' in df.columns:
            ax2.plot(steps, df['zeta'], color=C_UNET, linestyle='--', alpha=0.7, label='Zeta')
            ax2.set_ylabel('Zeta', color=C_UNET)
            ax2.tick_params(axis='y', labelcolor=C_UNET)
        ax.set_title('(f) LR and Zeta Schedule')

        # Summary Table
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        last = df.iloc[-1]
        rows = [['Metric', 'Final'], ['Total Loss', f"{last.get('loss', float('nan')):.4f}"], ['MSE', f"{last.get('denoising', float('nan')):.5f}"]]
        tbl = ax.table(cellText=rows, loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.8])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        fig.suptitle(f"Training Overview — {self.run_name}", fontsize=15, fontweight='bold')
        self._save(fig, "01_training_overview.png")

    def plot_router_health(self):
        print("\n[2] Router Health")
        df = _parse_jsonl(self._logs["router"])
        if df is None:
            return

        steps = df['step']
        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.30)

        # (a) Entropy
        ax = fig.add_subplot(gs[0, :2])
        for col, label, color in [('unet_entropy_norm', 'UNet', C_UNET), ('vit_entropy_norm', 'ViT', C_VIT)]:
            if col in df.columns:
                ax.plot(steps, df[col], color=color, alpha=0.25, linewidth=0.8)
                ax.plot(steps, _smooth(df[col], self.smooth), color=color, label=label, linewidth=2.0)
        ax.axhline(MIN_ENTROPY_NORM, color=C_WARN, linestyle='--', linewidth=1.5)
        _add_health_band(ax, MIN_ENTROPY_NORM, 1.0, 'Healthy region')
        ax.set_ylim(0, 1.05)
        ax.set_title('(a) Router Entropy (Normalized)')
        ax.legend(fontsize=9)

        # (b) Gini
        ax = fig.add_subplot(gs[0, 2])
        for col, label, color in [('unet_gini', 'UNet', C_UNET), ('vit_gini', 'ViT', C_VIT)]:
            if col in df.columns:
                ax.plot(steps, df[col], color=color, alpha=0.3, linewidth=0.8)
                ax.plot(steps, _smooth(df[col], self.smooth), color=color, label=label, linewidth=2.0)
        ax.axhline(MAX_GINI, color=C_WARN, linestyle='--', linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_title('(b) Gini Coefficient')
        ax.legend(fontsize=9)

        # (c) Dead
        ax = fig.add_subplot(gs[1, 0])
        for col, label, color in [('unet_dead_experts', 'UNet Dead', C_UNET), ('vit_dead_experts', 'ViT Dead', C_VIT)]:
            if col in df.columns:
                ax.step(steps, df[col], where='post', color=color, label=label)
        ax.set_title('(c) Dead Expert Count')
        ax.legend(fontsize=9)

        # (d) Utilization
        ax = fig.add_subplot(gs[1, 1:])
        last = df.iloc[-1]
        def _parse_list(val): return val if isinstance(val, list) else (json.loads(val) if isinstance(val, str) else [])
        unet_usage = _parse_list(last.get('unet_usage', []))
        vit_usage = _parse_list(last.get('vit_usage', []))
        n_experts = max(len(unet_usage), len(vit_usage), 1)
        x = np.arange(n_experts)
        width = 0.35
        if unet_usage: ax.bar(x - width/2, unet_usage, width, color=C_UNET, alpha=0.8, label='UNet')
        if vit_usage: ax.bar(x + width/2, vit_usage, width, color=C_VIT, alpha=0.8, label='ViT')
        ax.axhline(1.0/n_experts, color=C_GOOD, linestyle='--')
        ax.set_xticks(x)
        ax.set_title(f'(d) Expert Utilization at Step {int(last.get("step", 0))}')
        ax.legend(fontsize=9)

        # (e) Heatmaps
        last_with_hist = df.dropna(subset=['activation_hist_unet']).iloc[-1] if 'activation_hist_unet' in df.columns else None
        # FIXED: Use mpl.colormaps instead of deprecated plt.cm.get_cmap
        for panel_idx, (hist_key, cmap_name, title_suffix) in enumerate([('activation_hist_unet', 'plasma', 'UNet'), ('activation_hist_vit', 'viridis', 'ViT')]):
            ax = fig.add_subplot(gs[2, panel_idx])
            if last_with_hist is not None and hist_key in last_with_hist:
                hist = last_with_hist[hist_key]
                if isinstance(hist, str): hist = json.loads(hist)
                hist = np.array(hist)
                n_bins, n_exp = hist.shape
                im = ax.imshow(hist.T, aspect='auto', cmap=mpl.colormaps[cmap_name], vmin=0, vmax=1, extent=[-0.5, n_bins-0.5, n_exp-0.5, -0.5])
                ax.set_xticks(range(n_bins))
                ax.set_xticklabels(SIGMA_BIN_LABELS[:n_bins], rotation=35, ha='right', fontsize=7)
                ax.set_title(f'(e) {title_suffix} Activation Landscape')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, 'Histogram not logged', ha='center', va='center')

        fig.suptitle(f"Router Health — {self.run_name}", fontsize=15, fontweight='bold')
        self._save(fig, "02_router_health.png")

    def plot_gradient_health(self):
        print("\n[3] Gradient Health")
        df = _parse_jsonl(self._logs["gradients"])
        if df is None: return

        steps = df['step']
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

        # (a) Update Ratios
        ax = fig.add_subplot(gs[0, :])
        ratio_cols = [c for c in df.columns if c.endswith('_update_ratio') and not c.endswith('_healthy')]
        colors = mpl.colormaps['tab10'](np.linspace(0, 1, max(len(ratio_cols), 1)))
        for col, color in zip(ratio_cols, colors):
            label = col.replace('_update_ratio', '').replace('_experts', '')
            vals = df[col].replace(0, np.nan).dropna()
            if len(vals) < 2: continue
            # Handle index alignment since we dropped NaNs
            aligned_steps = steps[vals.index]
            ax.semilogy(aligned_steps, gaussian_filter1d(vals.values, sigma=5), label=label, color=color, linewidth=1.5)
        _add_health_band(ax, UPDATE_WARN_LOW, UPDATE_WARN_HIGH, 'Healthy zone', alpha=0.10)
        ax.axhline(IDEAL_UPDATE_RATIO, color=C_GOOD, linestyle='--', label='Ideal 1e-3')
        ax.set_title('(a) Update Ratios per Component')
        ax.legend(fontsize=8, ncol=3, loc='lower left')

        # (b) Norms
        ax = fig.add_subplot(gs[1, 0])
        norm_cols = [c for c in df.columns if c.endswith('_grad_norm') and not any(x in c for x in ['enc_', 'dec_', 'early_', 'late_'])]
        unet_norms = [c for c in norm_cols if 'Unet' in c or 'unet' in c]
        vit_norms = [c for c in norm_cols if 'VIT' in c or 'vit_exp' in c]
        router_norms = [c for c in norm_cols if 'router' in c]
        for cols, color, label in [(unet_norms, C_UNET, 'UNet experts'), (vit_norms, C_VIT, 'ViT experts'), (router_norms, C_BLEND, 'Routers')]:
            if not cols: continue
            vals = df[cols].mean(axis=1).replace(0, np.nan)
            ax.semilogy(steps, _smooth(vals, 5), color=color, label=label, linewidth=2.0)
        ax.set_title('(b) Gradient Norms')
        ax.legend(fontsize=9)

        # (c) Depth Ratio
        ax = fig.add_subplot(gs[1, 1])
        depth_cols = [c for c in df.columns if 'depth_ratio' in c]
        if depth_cols:
            for col in depth_cols:
                color = C_VIT if 'vit' in col else C_UNET
                ax.plot(steps, _smooth(df[col], 5), color=color, label=col.replace('_depth_ratio', ''), linewidth=1.5)
            ax.axhline(1.0, color=C_NEUTRAL, linestyle='--')
            ax.axhline(0.5, color=C_WARN, linestyle=':')
            ax.set_title('(c) ViT Depth Gradient Ratio (< 0.5 = vanishing)')
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'depth_ratio not logged', ha='center', va='center')

        fig.suptitle(f"Gradient Health — {self.run_name}", fontsize=15, fontweight='bold')
        self._save(fig, "03_gradient_health.png")

    def plot_branch_specialization(self):
        print("\n[4] Branch Specialization")
        df = _parse_jsonl(self._logs["specialization"])
        if df is None: return

        steps = df['step']
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
        n_bins = len(SIGMA_BIN_LABELS)

        def _extract_bin_series(df, col):
            rows = []
            for val in df[col]:
                if isinstance(val, str): val = json.loads(val)
                if isinstance(val, list) and len(val) == n_bins: rows.append(val)
                else: rows.append([np.nan]*n_bins)
            return np.array(rows)

        # (a) L2
        ax = fig.add_subplot(gs[0, 0])
        if 'l2_divergence_per_bin' in df.columns:
            mat = _extract_bin_series(df, 'l2_divergence_per_bin')
            cmap = mpl.colormaps['plasma']
            for b in range(n_bins):
                col_vals = mat[:, b]
                # Filter NaNs for plotting to avoid smoothing artifacts
                mask = ~np.isnan(col_vals)
                if mask.any():
                    # We can't easily use _smooth here because we have a numpy array column
                    # Basic smoothing on valid elements
                    smoothed = gaussian_filter1d(col_vals[mask], sigma=5)
                    ax.plot(steps[mask], smoothed, color=cmap(b/n_bins), label=SIGMA_BIN_LABELS[b], linewidth=1.5)
            ax.set_title('(a) Branch L2 Divergence per σ Bin')
            ax.legend(fontsize=7, ncol=2)

        # (d) Mean L2
        ax = fig.add_subplot(gs[1, 1])
        if 'mean_l2_divergence' in df.columns:
            ax.plot(steps, _smooth(df['mean_l2_divergence'], self.smooth), color=C_BLEND, linewidth=2.5, label='Mean L2')
            ax.axhline(0.05, color=C_WARN, linestyle='--', label='Collapse')
            ax.set_title('(d) Mean Divergence Over Time')
            ax.legend(fontsize=9)

        fig.suptitle(f"Branch Specialization — {self.run_name}", fontsize=15, fontweight='bold')
        self._save(fig, "04_branch_specialization.png")

    def plot_checkpoint_timeline(self):
        print("\n[5] Checkpoint Timeline")
        df = _parse_jsonl(self._logs["checkpoints"])
        if df is None: return

        steps = df['step']
        n_ckpt = len(df)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

        # (a) MSE
        ax = fig.add_subplot(gs[0, 0])
        if 'mse' in df.columns:
            colors = [C_GOOD if p else C_WARN for p in df.get('PASS_mse', [False]*n_ckpt)]
            ax.scatter(steps, df['mse'], c=colors, s=80, zorder=5)
            ax.plot(steps, df['mse'], color=C_NEUTRAL, alpha=0.4, linewidth=1)
            ax.axhline(0.10, color=C_WARN, linestyle='--', label='Target 0.10')
            ax.set_title('(a) Checkpoint MSE')
            ax.legend(fontsize=9)

        # (d) Heatmap
        ax = fig.add_subplot(gs[1, 1])
        pass_cols = [c for c in df.columns if c.startswith('PASS_')]
        if pass_cols:
            data = df[pass_cols].astype(float).values.T
            cmap = mcolors.ListedColormap([C_WARN, C_GOOD])
            ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=[-0.5, n_ckpt-0.5, len(pass_cols)-0.5, -0.5])
            ax.set_xticks(range(n_ckpt))
            ax.set_xticklabels([str(int(s)) for s in steps], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(pass_cols)))
            ax.set_yticklabels([c.replace('PASS_', '') for c in pass_cols], fontsize=9)
            ax.set_title('(d) Pass/Fail Heatmap')

        fig.suptitle(f"Checkpoint Health — {self.run_name}", fontsize=15, fontweight='bold')
        self._save(fig, "05_checkpoint_timeline.png")

    def plot_all(self):
        print(f"\n{'='*60}\n  HDMOEM Diagnostic Plots — {self.run_name}\n{'='*60}")
        self.plot_training_overview()
        self.plot_router_health()
        self.plot_gradient_health()
        self.plot_branch_specialization()
        self.plot_checkpoint_timeline()
        print(f"\n✓ All plots saved to: {self.out_dir}")


if __name__ == "__main__":
    log_dir = r"C:\Users\Maha Mamdouh\PycharmProjects\Heterogeneous-MOE-for-Diffusion-models\Utils\logs"

    # Must match the 'run_name' used in Logger during training
    run_name = "hdmoem_flower102_v1"
    output_folder = "generated_samples_unguided"
    plotter = Plotter(
        log_dir=log_dir,
        run_name=run_name,
        out_dir=os.path.join(output_folder, "analysis"),  # Save plots inside output folder
        smooth_sigma=5.0
    )
    plotter.plot_all()
