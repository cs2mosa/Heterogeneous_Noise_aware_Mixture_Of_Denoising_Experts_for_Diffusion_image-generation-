"""
Enhanced Training Logger for Hybrid Diffusion MoE Models
=========================================================

This module provides comprehensive logging utilities for tracking all critical
metrics during training. Compatible with the ProfessionalDiffusionPlotter.

Based on best practices from:
- EDM2 (NVIDIA)
- MoE literature (Switch Transformer, DeepSeek)
- Recent diffusion model papers
"""
import math
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict


class Logger:
    """
    Comprehensive training logger for diffusion MoE models.
    
    Tracks:
    - Core losses (EDM, MSE, uncertainty)
    - MoE auxiliary losses (balance, z-loss, entropy)
    - Router statistics (usage, entropy, collapse metrics)
    - Scaling/gating behavior
    - Gradient norms
    - Weight statistics
    - Learning rate schedule
    """
    
    def __init__(self, 
                 log_dir: str = "./training_logs",
                 run_name: str = "experiment",
                 log_interval: int = 10):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            run_name: Name of this training run
            log_interval: Log every N steps (reduce overhead)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_name = run_name
        self.log_interval = log_interval
        
        # File paths
        self.main_log_file = self.log_dir / f"{run_name}_training.jsonl"
        self.router_log_file = self.log_dir / f"{run_name}_router_stats.jsonl"
        self.gradient_log_file = self.log_dir / f"{run_name}_gradients.jsonl"
        self.weight_log_file = self.log_dir / f"{run_name}_weights.jsonl"
        
        # Accumulators for averaging over interval
        self.accumulators = defaultdict(list)
        
        print(f"✓ Initialized Enhanced Logger: {run_name}")
        print(f"  Main log: {self.main_log_file}")
        print(f"  Router stats: {self.router_log_file}")
        print(f"  Gradients: {self.gradient_log_file}")
        print(f"  Weights: {self.weight_log_file}")
    
    def log_training_step(self,
                         step: int,
                         loss_dict: Dict[str, torch.Tensor],
                         zeta: float,
                         log_var: float,
                         lr: float,
                         p_mean:float,
                         p_std: float,
                         sigma: Optional[torch.Tensor] = None):
        """
        Log main training metrics.
        
        Args:
            step: Current training step
            loss_dict: Dictionary with keys:
                - 'loss': total loss
                - 'denoising': raw MSE/denoising loss
                - 'pure_loss': pure EDM loss (before auxiliary)
                - 'balance': load balancing loss
                - 'z_loss': router z-loss
                - 'entropy': entropy regularization
            zeta: Router exploration noise
            log_var: Predicted log variance
            lr: Current learning rate
            sigma: Noise levels (optional, for percentile tracking)
        """
        # Accumulate for interval averaging
        self.accumulators['step'].append(step)
        
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.accumulators[key].append(value.item())
            else:
                self.accumulators[key].append(float(value))
        self.accumulators['zeta'].append(float(zeta))
        self.accumulators['log_var'].append(float(log_var))
        self.accumulators['lr'].append(float(lr))
        
        if sigma is not None:
            # Track average sigma percentile
            log_sigma = torch.log(sigma)
            sigma_pct = 0.5 * (1 + torch.erf(
                (log_sigma - p_mean) / (p_std * np.sqrt(2))
            ))
            self.accumulators['avg_sigma_percentile'].append(
                sigma_pct.mean().item()
            )
        
        # Write to file every log_interval steps
        if step % self.log_interval == 0 and len(self.accumulators['step']) > 0:
            self._flush_training_log()
    
    def log_router_statistics(self,
                              step: int,
                              unet_probs: torch.Tensor,
                              vit_probs: torch.Tensor,
                              p_mean: float,
                              p_std: float,
                              sigma: torch.Tensor):
        """
        Log detailed router statistics.
        
        Args:
            p_std:
            p_mean:
            step: Current training step
            unet_probs: UNet router probabilities (B, num_experts)
            vit_probs: ViT router probabilities (B, num_experts)
            sigma: Noise levels (B,)
        """
        if step % self.log_interval != 0:
            return
        
        with torch.no_grad():
            # Convert sigma to percentile
            log_sigma = torch.log(sigma)
            sigma_pct = 0.5 * (1 + torch.erf(
                (log_sigma - p_mean) / (p_std * np.sqrt(2))
            ))
            
            # Calculate entropy (diversity metric)
            def calc_entropy(probs):
                # Average across batch
                avg_probs = probs.mean(dim=0)
                avg_probs = avg_probs / (avg_probs.sum() + 1e-10)
                entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
                return entropy.item()
            
            # Calculate Gini coefficient (collapse metric)
            def calc_gini(probs):
                # Average usage per expert
                usage = probs.mean(dim=0).cpu().numpy()
                usage = np.sort(usage)
                n = len(usage)
                cumsum = np.cumsum(usage)
                return (2 * np.sum((np.arange(1, n+1) * usage))) / (n * cumsum[-1]) - (n+1)/n
            
            # Expert usage statistics
            unet_usage = unet_probs.mean(dim=0)  # (num_experts,)
            vit_usage = vit_probs.mean(dim=0)
            
            record = {
                "step": step,
                "avg_sigma_percentile": sigma_pct.mean().item(),
                
                # UNet router stats
                "unet_entropy": calc_entropy(unet_probs),
                "unet_gini": calc_gini(unet_probs),
                "unet_max_usage": unet_usage.max().item(),
                "unet_min_usage": unet_usage.min().item(),
                "unet_dead_experts": int((unet_usage < 0.01).sum().item()),
                "unet_usage_std": unet_usage.std().item(),
                
                # ViT router stats
                "vit_entropy": calc_entropy(vit_probs),
                "vit_gini": calc_gini(vit_probs),
                "vit_max_usage": vit_usage.max().item(),
                "vit_min_usage": vit_usage.min().item(),
                "vit_dead_experts": int((vit_usage < 0.01).sum().item()),
                "vit_usage_std": vit_usage.std().item(),
                
                # Per-expert usage (for detailed analysis)
                "unet_expert_usage": unet_usage.cpu().tolist(),
                "vit_expert_usage": vit_usage.cpu().tolist(),
            }
            
            self._write_jsonl(self.router_log_file, record)

    def log_scaling_gating(self,
                           scaling_factors: torch.Tensor,
                           gate_weights: torch.Tensor,
                           sigma: torch.Tensor,
                           p_mean: float = -0.4,
                           p_std: float = 1.0):
        # 1. Calculate Individual Percentiles
        s = sigma.flatten()
        log_sigma = torch.log(s)
        sigma_pcts = 0.5 * (1 + torch.erf((log_sigma - p_mean) / (p_std * math.sqrt(2))))
        sigma_pcts = sigma_pcts.clamp(0, 1)

        # --- FOR THE LINE GRAPHS (Evolution over time) ---
        # These need to be MEANS to show the stable trend of the model
        self.accumulators['scaling_vit_mean'].append(scaling_factors[:, 1].mean().item())
        self.accumulators['scaling_unet_mean'].append(scaling_factors[:, 0].mean().item())
        self.accumulators['scaling_fusion_mean'].append(scaling_factors[:, 2].mean().item())
        self.accumulators['gate_wx_mean'].append(gate_weights[:, 0].mean().item())
        self.accumulators['gate_wa_mean'].append(gate_weights[:, 1].mean().item())
        self.accumulators['gate_wf_mean'].append(gate_weights[:, 2].mean().item())
        # --- FOR THE SCATTER PLOTS (Inductive Bias / X-Pattern) ---
        # These MUST be from the same index [0] to maintain correlation
        self.accumulators['sample_sigma_percentile'].append(sigma_pcts[0].item())
        self.accumulators['sample_vit_scale'].append(scaling_factors[0, 1].item())
        self.accumulators['sample_unet_scale'].append(scaling_factors[0, 0].item())
        self.accumulators['sample_fusion_scale'].append(scaling_factors[0, 2].item())
        self.accumulators['sample_gate_wx'].append(gate_weights[0, 0].mean().item())
        self.accumulators['sample_gate_wa'].append(gate_weights[0, 1].mean().item())
        self.accumulators['sample_gate_wf'].append(gate_weights[0, 2].mean().item())
        # --- FOR BOUNDARY CHECKING ---
        self.accumulators['noise_pct_min'].append(sigma_pcts.min().item())
        self.accumulators['noise_pct_max'].append(sigma_pcts.max().item())
        self.accumulators['noise_level_min'].append(sigma.min().item())
        self.accumulators['noise_level_max'].append(sigma.max().item())

    def log_gradients(self,
                      step: int,
                      model: torch.nn.Module,

                      current_lr: float,  # <--- Added Argument
                      component_names: Optional[List[str]] = None):
        """
        Logs gradient norms and the Update Ratio (Grad * LR / Weight).
        Target Update Ratio is typically around 1e-3.
        """
        if step % self.log_interval != 0:
            return

        if component_names is None:
            # Adjust these names based on your exact attribute names in the model wrapper
            component_names = ['Unet_experts', 'VIT_experts', 'Unet_router',
                               'vit_router', 'gate_mlp', 'cross_attn',
                               'output_proj_unet', 'output_proj_vit']

        record: dict[str, Any] = {"step": step}

        # Helper to handle DataParallel or Preconditioned wrappers
        def get_submodule(root, name_):
            if hasattr(root, 'net') and hasattr(root.net, name_):
                return getattr(root.net, name_)
            return getattr(root, name_, None)

        for name in component_names:
            component = get_submodule(model, name)

            if component is None:
                continue

            total_grad_sq = 0.0
            total_param_sq = 0.0
            param_count = 0

            # 1. Accumulate squared norms (Frobenius Norm logic)
            for p in component.parameters():
                if p.grad is not None:
                    # Detach to prevent memory leaks during logging
                    total_grad_sq += p.grad.detach().data.norm(2).item() ** 2
                    total_param_sq += p.detach().data.norm(2).item() ** 2
                    param_count += 1

            if param_count > 0:
                # 2. Square root to get actual Norm
                grad_norm = total_grad_sq ** 0.5
                param_norm = total_param_sq ** 0.5

                record[f"{name}_grad_norm"] = grad_norm

                # 3. Calculate Update Ratio.
                # Ratio = (Gradient_Norm * Learning_Rate) / Weight_Norm
                if param_norm > 1e-9:  # Avoid division by zero
                    update_ratio = (grad_norm * current_lr) / param_norm
                    record[f"{name}_update_ratio"] = update_ratio
                else:
                    record[f"{name}_update_ratio"] = 0.0
            else:
                record[f"{name}_grad_norm"] = 0.0
                record[f"{name}_update_ratio"] = 0.0

        self._write_jsonl(self.gradient_log_file, record)

    def log_weight_statistics(self, step: int, model: torch.nn.Module):
        """
        Logs weight statistics (Mean, Std, Min, Max).
        FIXED: Removed torch.quantile and torch.cat to prevent RuntimeError on large models.
        """
        # Log much less frequently (e.g., every 50th log interval)
        if step % (self.log_interval * 50) != 0:
            return

        component_names = ['Unet_experts', 'VIT_experts']
        record: dict[str, Any] = {"step": step}

        with torch.no_grad():
            for name in component_names:
                if hasattr(model, name):
                    component = getattr(model, name)

                    # Accumulators for iterative calculation
                    total_sum = 0.0
                    total_sq_sum = 0.0  # Sum of squares for Std Dev
                    total_count = 0
                    max_val = -float('inf')
                    min_val = float('inf')

                    found_weights = False

                    # Iterate over parameters one by one (Memory Safe)
                    for param in component.parameters():
                        if param.requires_grad and param.ndim > 1:
                            found_weights = True

                            # Get min/max of this specific tensor
                            p_min = param.min().item()
                            p_max = param.max().item()

                            # Update global min/max
                            if p_max > max_val: max_val = p_max
                            if p_min < min_val: min_val = p_min

                            # Update sums for Mean/Std
                            total_sum += param.sum().item()
                            total_sq_sum += param.pow(2).sum().item()
                            total_count += param.numel()

                    if found_weights and total_count > 0:
                        # Calculate statistics from accumulators
                        mean = total_sum / total_count

                        # Variance = E[X^2] - (E[X])^2
                        var = (total_sq_sum / total_count) - (mean ** 2)
                        std = np.sqrt(max(0, var))  # Clamp to 0 to avoid fp errors

                        record[f"{name}_weight_mean"] = round(mean, 6)
                        record[f"{name}_weight_std"] = round(std, 6)
                        record[f"{name}_weight_max"] = round(max_val, 6)
                        record[f"{name}_weight_min"] = round(min_val, 6)
                    else:
                        record[f"{name}_weight_mean"] = None

        self._write_jsonl(self.weight_log_file, record)
    
    def _flush_training_log(self):
        """Write accumulated training metrics to file."""
        if len(self.accumulators['step']) == 0:
            return
        
        # Average over the interval
        record: dict[str, Any] = {"step": int(self.accumulators['step'][-1])}
        
        for key, values in self.accumulators.items():
            if key == 'step':
                continue
            if len(values) > 0:
                record[key] = round(float(np.mean(values)), 6)
        
        self._write_jsonl(self.main_log_file, record)
        
        # Clear accumulators
        self.accumulators.clear()
    @staticmethod
    def _write_jsonl(filepath: Path, record: Dict[str, Any]):
        """Append JSON record to file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    @staticmethod
    def _compute_grad_norm(parameters) -> float:
        """Compute gradient norm for a set of parameters."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
