"""
HDMOEM Research Logger
======================
Professional, research-grade training logger for the Hybrid Diffusion MoE Model.
"""

import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
SIGMA_BINS = np.array([0.002, 0.02, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 80.0])
N_SIGMA_BINS = len(SIGMA_BINS) - 1
IDEAL_UPDATE_RATIO = 1e-3
UPDATE_RATIO_WARN_HIGH = 1e-2
UPDATE_RATIO_WARN_LOW = 1e-4
MIN_HEALTHY_ENTROPY_RATIO = 0.85  # entropy / max_entropy > this is healthy
MAX_HEALTHY_GINI = 0.25  # Gini < this is healthy


def _sigma_to_bin(sigma_flat: torch.Tensor) -> np.ndarray:
    """Map sigma values to histogram bin indices. Returns (B,) int array."""
    s = sigma_flat.detach().float().cpu().numpy().flatten()
    # digitize returns 1-based indices for bins, we adjust to 0-based
    # indicies will be 0..N_SIGMA_BINS-1 provided sigma is within range
    indices = np.digitize(s, SIGMA_BINS) - 1
    return np.clip(indices, 0, N_SIGMA_BINS - 1)


def _entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of average routing distribution. probs: (B, E)."""
    avg = probs.mean(dim=0)
    avg = avg / (avg.sum() + 1e-10)
    return float(-torch.sum(avg * torch.log(avg + 1e-10)).item())


def _max_entropy(num_experts: int) -> float:
    return math.log(max(1, num_experts))


def _gini(probs: torch.Tensor) -> float:
    """
    Robust Gini coefficient for discrete data (brown's formula).
    instead of using lorentz formula as the num_experts is low.
    Invariant to N (number of experts).
    Range: [0, 1] always.
    """
    # 1. Get average usage per expert
    usage = probs.mean(dim=0).detach().cpu().numpy()  # Shape (N_experts,)

    # 2. Handle edge case (empty or zero sum)
    if len(usage) == 0 or np.sum(usage) < 1e-9:
        return 0.0

    # 3. Sort values (smallest to largest)
    usage = np.sort(usage)
    n = len(usage)

    # 4. Correct Discrete Gini Formula
    # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    # where i is 1-based index

    index = np.arange(1, n + 1)
    numerator = np.sum((2 * index - n - 1) * usage)
    denominator = n * np.sum(usage)

    return float(numerator / denominator)

class Logger:
    """
    Research-grade logger for the HDMOEM training loop.
    """

    def __init__(self,
                 log_dir: str,
                 run_name: str,
                 log_interval: int = 10,
                 num_unet_experts: int = 2,
                 num_vit_experts: int = 2,
                 transition_sigma: float = 3.0,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 p_mean: float = 0.0,
                 p_std: float = 1.3,
                 ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.log_interval = log_interval
        self.num_unet_experts = num_unet_experts
        self.num_vit_experts = num_vit_experts
        self.transition_sigma = transition_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std

        # File handles
        self._files = {
            "training": self.log_dir / f"{run_name}_training.jsonl",
            "router": self.log_dir / f"{run_name}_router.jsonl",
            "gradients": self.log_dir / f"{run_name}_gradients.jsonl",
            "specialization": self.log_dir / f"{run_name}_specialization.jsonl",
            "checkpoints": self.log_dir / f"{run_name}_checkpoints.jsonl",
        }

        # Single shared accumulator
        self._acc: Dict[str, List[float]] = defaultdict(list)

        # Online expert activation histogram
        self._unet_act_hist = np.zeros((N_SIGMA_BINS, num_unet_experts), dtype=np.float64)
        self._vit_act_hist = np.zeros((N_SIGMA_BINS, num_vit_experts), dtype=np.float64)
        self._hist_counts = np.zeros(N_SIGMA_BINS, dtype=np.float64)

        self._step_count = 0
        self._start_time = time.time()

        print(f"[HDMOEMLogger] Initialized run: {run_name}")
        for k, v in self._files.items():
            print(f"  {k:20s} → {v}")

    # ── Sigma Utilities ───────────────────────────────────────────────────────

    def _sigma_percentile(self, sigma: torch.Tensor) -> torch.Tensor:
        log_s = torch.log(sigma.float().flatten().clamp(min=1e-8))
        pct = 0.5 * (1.0 + torch.erf(
            (log_s - self.p_mean) / (self.p_std * math.sqrt(2))
        ))
        return pct.clamp(0.0, 1.0)

    def _trajectory_zones(self, sigma: torch.Tensor) -> Dict[str, float]:
        s = sigma.float().flatten()
        t = self.transition_sigma
        n = s.numel()
        if n == 0:
            return {"vit_dominant": 0.0, "overlap": 0.0, "unet_dominant": 0.0}
        vit_dom = (s > t * 1.5).float().sum().item() / n
        unet_dom = (s <= t * 0.5).float().sum().item() / n
        overlap = 1.0 - vit_dom - unet_dom
        return {
            "zone_vit_dominant": round(vit_dom, 4),
            "zone_overlap": round(overlap, 4),
            "zone_unet_dominant": round(unet_dom, 4),
        }

    # ── Primary Logging Methods ───────────────────────────────────────────────

    def log_training_step(self,
                          step: int,
                          loss_dict: Dict[str, Any],
                          sigma: torch.Tensor,
                          zeta: float,
                          log_var: float,
                          lr: float,
                          w_unet: Optional[torch.Tensor] = None,
                          w_vit: Optional[torch.Tensor] = None,
                          ):
        self._step_count += 1

        # ── Core losses
        for key, val in loss_dict.items():
            if isinstance(val, (int, float)):
                self._acc[key].append(float(val))
            elif isinstance(val, torch.Tensor):
                self._acc[key].append(float(val.item()))

        # ── Schedule state
        self._acc["zeta"].append(float(zeta))
        self._acc["log_var"].append(float(log_var))
        self._acc["lr"].append(float(lr))

        # ── Sigma health
        with torch.no_grad():
            pct = self._sigma_percentile(sigma)
            self._acc["sigma_pct_mean"].append(float(pct.mean().item()))
            self._acc["sigma_pct_std"].append(float(pct.std().item()))
            zones = self._trajectory_zones(sigma)
            for k, v in zones.items():
                self._acc[k].append(v)

        # ── Blend weight health
        if w_unet is not None and w_vit is not None:
            with torch.no_grad():
                wu = w_unet.float().flatten()
                wv = w_vit.float().flatten()
                self._acc["blend_w_unet_mean"].append(float(wu.mean().item()))
                self._acc["blend_w_vit_mean"].append(float(wv.mean().item()))

                # Blend entropy
                safe_wu = wu + 1e-8
                safe_wv = wv + 1e-8
                blend_ent = -(wu * safe_wu.log() + wv * safe_wv.log()).mean()
                self._acc["blend_entropy"].append(float(blend_ent.item()))

        # ── Flush
        if step % self.log_interval == 0 and self._step_count > 0:
            self._flush_training(step)

    def log_router_statistics(self,
                              step: int,
                              unet_probs: torch.Tensor,
                              vit_probs: torch.Tensor,
                              sigma: torch.Tensor,
                              ):
        if step % self.log_interval != 0:
            return

        with torch.no_grad():
            # ── Entropy and Gini
            unet_ent = _entropy(unet_probs)
            vit_ent = _entropy(vit_probs)
            unet_gini = _gini(unet_probs)
            vit_gini = _gini(vit_probs)

            unet_max_ent = _max_entropy(self.num_unet_experts)
            vit_max_ent = _max_entropy(self.num_vit_experts)

            unet_usage = unet_probs.mean(dim=0).cpu()
            vit_usage = vit_probs.mean(dim=0).cpu()

            # Dead expert threshold: < 1% of uniform allocation
            unet_dead = int((unet_usage < (1.0 / max(1, self.num_unet_experts) * 0.01)).sum().item())
            vit_dead = int((vit_usage < (1.0 / max(1, self.num_vit_experts) * 0.01)).sum().item())

            record = {
                "step": step,
                "unet_entropy_norm": round(unet_ent / unet_max_ent, 4),
                "vit_entropy_norm": round(vit_ent / vit_max_ent, 4),
                "unet_entropy_raw": round(unet_ent, 4),
                "vit_entropy_raw": round(vit_ent, 4),
                "unet_gini": round(unet_gini, 4),
                "vit_gini": round(vit_gini, 4),
                "unet_dead_experts": unet_dead,
                "vit_dead_experts": vit_dead,
                "unet_usage": [round(float(x), 4) for x in unet_usage.tolist()],
                "vit_usage": [round(float(x), 4) for x in vit_usage.tolist()],
                "unet_healthy": unet_ent / unet_max_ent > MIN_HEALTHY_ENTROPY_RATIO
                                and unet_gini < MAX_HEALTHY_GINI
                                and unet_dead == 0,
                "vit_healthy": vit_ent / vit_max_ent > MIN_HEALTHY_ENTROPY_RATIO
                               and vit_gini < MAX_HEALTHY_GINI
                               and vit_dead == 0,
            }

            # ── Online activation histogram
            sigma_bins = _sigma_to_bin(sigma)  # (B,) int
            unet_p = unet_probs.detach().cpu().numpy()  # (B, E)
            vit_p = vit_probs.detach().cpu().numpy()  # (B, E)

            for b, sbin in enumerate(sigma_bins):
                # Ensure bin index is within bounds
                if 0 <= sbin < N_SIGMA_BINS:
                    self._unet_act_hist[sbin] += unet_p[b]
                    self._vit_act_hist[sbin] += vit_p[b]
                    self._hist_counts[sbin] += 1.0

            # Normalize histogram
            hist_counts_safe = self._hist_counts[:, None] + 1e-10
            unet_hist_norm = (self._unet_act_hist / hist_counts_safe).tolist()
            vit_hist_norm = (self._vit_act_hist / hist_counts_safe).tolist()

            record["activation_hist_unet"] = [[round(x, 4) for x in row]
                                              for row in unet_hist_norm]
            record["activation_hist_vit"] = [[round(x, 4) for x in row]
                                             for row in vit_hist_norm]
            record["sigma_bin_edges"] = SIGMA_BINS.tolist()

            self._write_jsonl(self._files["router"], record)

    def log_gradients(self,
                      step: int,
                      model: nn.Module,
                      lr_per_component: Dict[str, float],
                      ):
        if step % self.log_interval != 0:
            return

        record: Dict[str, Any] = {"step": step}

        def _component_stats(module: nn.Module, lr: float) -> Dict[str, float]:
            grad_sq_sum = 0.0
            param_sq_sum = 0.0

            # Accumulators for "Mean of Layer Ratios"
            ratios = []

            total_params = 0
            params_with_grad = 0

            for p in module.parameters():
                total_params += 1
                p_norm = p.detach().norm(2).item()
                param_sq_sum += p_norm ** 2

                if p.grad is not None:
                    g_norm = p.grad.detach().norm(2).item()
                    grad_sq_sum += g_norm ** 2
                    params_with_grad += 1

                    # --- THE FIX: Calculate ratio per-tensor ---
                    # For Adam, we estimate the effective step is roughly proportional to LR.
                    # But sticking to standard definition: Ratio = (Grad * LR) / Weight
                    if p_norm > 1e-9:
                        ratios.append((g_norm * lr) / p_norm)
                else:
                    # If no grad, ratio is 0 for this tensor
                    if p_norm > 1e-9:
                        ratios.append(0.0)

            if total_params == 0:
                return {}

            # Global Norms (Keep for reference)
            global_grad_norm = grad_sq_sum ** 0.5
            global_param_norm = param_sq_sum ** 0.5
            coverage = params_with_grad / total_params

            result = {
                "grad_norm": round(global_grad_norm, 8),
                "param_norm": round(global_param_norm, 6),
                "grad_coverage": round(coverage, 4),
            }

            # --- THE FIX: Robust Update Ratio ---
            # If we have ratios, use the MEAN (or Median/Max)
            # This prevents massive layers from drowning out active ones
            if len(ratios) > 0:
                avg_ratio = sum(ratios) / len(ratios)
                max_ratio = max(ratios)

                # We log the average as the primary "update_ratio"
                result["update_ratio"] = round(avg_ratio, 10)

                # Optional: Log max to see if *any* layer is moving fast
                result["update_ratio_max"] = round(max_ratio, 10)

                result["update_ratio_healthy"] = UPDATE_RATIO_WARN_LOW < avg_ratio < UPDATE_RATIO_WARN_HIGH
            else:
                result["update_ratio"] = 0.0
                result["update_ratio_healthy"] = False

            return result

        # ── Top-level components
        component_attrs = [
            'Unet_experts', 'VIT_experts', 'Unet_router', 'vit_router',
            'output_proj_unet', 'output_proj_vit',
            'input_proj', 'input_time_proj', 'blend_residual',
        ]
        for name in component_attrs:
            component = getattr(model, name, None)
            if component is None:
                continue
            # Get specific LR or default
            lr = lr_per_component.get(name, lr_per_component.get('default', 1e-3))

            stats = _component_stats(component, lr)
            for k, v in stats.items():
                record[f"{name}_{k}"] = v

        # ── Depth-resolved: UNet (Unchanged logic, just ensure detach is used)
        if hasattr(model, 'Unet_experts'):
            for expert_idx, expert in enumerate(model.Unet_experts):
                prefix = f"unet_exp{expert_idx}"
                enc_params = list(expert.encoders.parameters())
                dec_params = list(expert.decoders.parameters())

                def _group_norm(params):
                    sq = sum(p.grad.detach().norm(2).item() ** 2
                             for p in params if p.grad is not None)
                    return sq ** 0.5

                record[f"{prefix}_enc_grad_norm"] = round(_group_norm(enc_params), 8)
                record[f"{prefix}_dec_grad_norm"] = round(_group_norm(dec_params), 8)

        # ── Depth-resolved: ViT (Unchanged logic)
        if hasattr(model, 'VIT_experts'):
            for expert_idx, expert in enumerate(model.VIT_experts):
                prefix = f"vit_exp{expert_idx}"
                blocks = list(expert.diffit)
                mid = len(blocks) // 2
                early_params = [p for b in blocks[:mid] for p in b.parameters()]
                late_params = [p for b in blocks[mid:] for p in b.parameters()]

                def _group_norm(params):
                    sq = sum(p.grad.detach().norm(2).item() ** 2
                             for p in params if p.grad is not None)
                    return sq ** 0.5

                early_n = _group_norm(early_params)
                late_n = _group_norm(late_params)
                record[f"{prefix}_early_grad_norm"] = round(early_n, 8)
                record[f"{prefix}_late_grad_norm"] = round(late_n, 8)
                # Avoid division by zero
                denom = early_n + 1e-10
                record[f"{prefix}_depth_ratio"] = round(late_n / denom, 4)

        self._write_jsonl(self._files["gradients"], record)

    def log_branch_specialization(self,
                                  step: int,
                                  vit_out: torch.Tensor,
                                  unet_out: torch.Tensor,
                                  sigma: torch.Tensor,
                                  ):
        if step % self.log_interval != 0:
            return

        with torch.no_grad():
            sigma_bins = _sigma_to_bin(sigma)
            B = vit_out.shape[0]

            # L2 divergence
            diff = (vit_out - unet_out).float()
            denom = (vit_out.float().norm(2, dim=(1, 2, 3)) +
                     unet_out.float().norm(2, dim=(1, 2, 3)) + 1e-8)
            l2_div = diff.norm(2, dim=(1, 2, 3)) / denom

            # Cosine similarity
            v_flat = vit_out.float().reshape(B, -1)
            u_flat = unet_out.float().reshape(B, -1)
            cos_sim = torch.nn.functional.cosine_similarity(v_flat, u_flat, dim=1)

            # Spatial variance
            vit_var = vit_out.float().var(dim=(2, 3)).mean(dim=1)
            unet_var = unet_out.float().var(dim=(2, 3)).mean(dim=1)

            # Binning
            bin_l2 = np.zeros(N_SIGMA_BINS)
            bin_cos = np.zeros(N_SIGMA_BINS)
            bin_vvar = np.zeros(N_SIGMA_BINS)
            bin_uvar = np.zeros(N_SIGMA_BINS)
            bin_count = np.zeros(N_SIGMA_BINS)

            l2_np = l2_div.cpu().numpy()
            cos_np = cos_sim.cpu().numpy()
            vvar_np = vit_var.cpu().numpy()
            uvar_np = unet_var.cpu().numpy()

            for b, sbin in enumerate(sigma_bins):
                if 0 <= sbin < N_SIGMA_BINS:
                    bin_l2[sbin] += l2_np[b]
                    bin_cos[sbin] += cos_np[b]
                    bin_vvar[sbin] += vvar_np[b]
                    bin_uvar[sbin] += uvar_np[b]
                    bin_count[sbin] += 1.0

            safe_count = bin_count + 1e-10
            record = {
                "step": step,
                "sigma_bin_edges": SIGMA_BINS.tolist(),
                "l2_divergence_per_bin": [round(x, 6) for x in (bin_l2 / safe_count).tolist()],
                "cosine_sim_per_bin": [round(x, 6) for x in (bin_cos / safe_count).tolist()],
                "vit_spatial_var_per_bin": [round(x, 6) for x in (bin_vvar / safe_count).tolist()],
                "unet_spatial_var_per_bin": [round(x, 6) for x in (bin_uvar / safe_count).tolist()],
                "mean_l2_divergence": round(float(l2_div.mean().item()), 6),
                "mean_cosine_sim": round(float(cos_sim.mean().item()), 6),
                "mean_vit_spatial_var": round(float(vit_var.mean().item()), 6),
                "mean_unet_spatial_var": round(float(unet_var.mean().item()), 6),
                "branch_collapse_warning": float(l2_div.mean().item()) < 0.05,
            }

            self._write_jsonl(self._files["specialization"], record)

    def log_checkpoint_summary(self,
                               step: int,
                               mse: float,
                               model: nn.Module,
                               unet_probs: torch.Tensor,
                               vit_probs: torch.Tensor,
                               lr_per_component: Dict[str, float],
                               extra: Optional[Dict[str, Any]] = None,
                               ):
        with torch.no_grad():
            unet_ent_norm = _entropy(unet_probs) / _max_entropy(self.num_unet_experts)
            vit_ent_norm = _entropy(vit_probs) / _max_entropy(self.num_vit_experts)
            unet_gini = _gini(unet_probs)
            vit_gini = _gini(vit_probs)
            unet_usage = unet_probs.mean(dim=0).cpu().tolist()
            vit_usage = vit_probs.mean(dim=0).cpu().tolist()

        def _update_ratio(module, lr):
            g2, p2 = 0.0, 0.0
            for p in module.parameters():
                p2 += p.detach().norm(2).item() ** 2
                if p.grad is not None:
                    g2 += p.grad.detach().norm(2).item() ** 2
            pn = p2 ** 0.5
            return (g2 ** 0.5 * lr) / pn if pn > 1e-9 else 0.0

        unet_ur = _update_ratio(model.Unet_experts, lr_per_component.get('Unet_experts', 1e-3))
        vit_ur = _update_ratio(model.VIT_experts, lr_per_component.get('VIT_experts', 1e-3))

        elapsed = time.time() - self._start_time

        summary = {
            "step": step,
            "elapsed_seconds": round(elapsed, 1),
            "mse": round(mse, 6),
            "unet_entropy_norm": round(unet_ent_norm, 4),
            "vit_entropy_norm": round(vit_ent_norm, 4),
            "unet_gini": round(unet_gini, 4),
            "vit_gini": round(vit_gini, 4),
            "unet_usage": [round(x, 4) for x in unet_usage],
            "vit_usage": [round(x, 4) for x in vit_usage],
            "unet_update_ratio": round(unet_ur, 10),
            "vit_update_ratio": round(vit_ur, 10),
            "PASS_mse": mse < 0.15,
            "PASS_unet_router": unet_ent_norm > MIN_HEALTHY_ENTROPY_RATIO and unet_gini < MAX_HEALTHY_GINI,
            "PASS_vit_router": vit_ent_norm > MIN_HEALTHY_ENTROPY_RATIO and vit_gini < MAX_HEALTHY_GINI,
            "PASS_unet_training": UPDATE_RATIO_WARN_LOW < unet_ur < UPDATE_RATIO_WARN_HIGH,
            "PASS_vit_training": UPDATE_RATIO_WARN_LOW < vit_ur < UPDATE_RATIO_WARN_HIGH,
        }

        if extra:
            summary.update(extra)

        self._write_jsonl(self._files["checkpoints"], summary)
        print(f"\n[Checkpoint @ step {step}]")
        print(f"  MSE:              {mse:.5f}  {'✓' if summary['PASS_mse'] else '✗'}")
        print(f"  UNet router:      {unet_ent_norm:.3f}/{unet_gini:.3f} {'✓' if summary['PASS_unet_router'] else '✗'}")
        print(f"  ViT  router:      {vit_ent_norm:.3f}/{vit_gini:.3f} {'✓' if summary['PASS_vit_router'] else '✗'}")

    def _flush_training(self, step: int):
        if not self._acc:
            return

        record: Dict[str, Any] = {
            "step": step,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
        }
        for key, values in self._acc.items():
            if values:
                record[key] = round(float(np.mean(values)), 7)

        self._write_jsonl(self._files["training"], record)

        self._acc.clear()
        self._unet_act_hist[:] = 0.0
        self._vit_act_hist[:] = 0.0
        self._hist_counts[:] = 0.0
        self._step_count = 0

    @staticmethod
    def _write_jsonl(filepath: Path, record: Dict[str, Any]):
        # FIXED: Use custom lambda for default to handle numpy types safely
        def json_default(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, 'a') as f:
            f.write(json.dumps(record, default=json_default) + '\n')