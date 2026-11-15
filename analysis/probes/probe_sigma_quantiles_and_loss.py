"""Read-only diagnostic probe 4: sample_sigma_hybrid tail quantiles & log_var amplification math."""
import sys, math
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from Utils.utils import sample_sigma_hybrid

torch.manual_seed(0)
s = sample_sigma_hybrid(200000, device=torch.device("cpu")).flatten()
for thr in [0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
    print(f"P(sigma <= {thr:5.3f}) = {(s <= thr).float().mean().item():.4f} | P(sigma > {thr:5.3f}) = {(s > thr).float().mean().item():.4f}")

# Check exact EDM2 u(sigma) closed-form optimum effect on pure_loss vs vit_loss:
# If raw normalized error E_blend(sigma) = lambda(sigma) * MSE_blend(sigma):
# Initial E_0 ~ 1.53 (e.g. if data_std=1.0 & sigma_data=0.8: E_0 ~ 1.25-1.53; or step-0 perturbation)
# Optimal log_var u* = ln(E_blend) -> pure_loss* = E_blend * exp(-u*) + u* = 1.0 + ln(E_blend)
# Solve 1.0 + ln(E_blend) = 0.58 => E_blend = exp(-0.42) = 0.657, exp(-u*) = 1 / 0.657 = 1.522x!
print("\nClosed-form Kendall uncertainty u*(sigma) = ln(E_blend) impact on vit_loss:")
for e_blend, label in [(1.53, "step 0 initial E_blend=1.53 (u=0)"),
                       (0.657, "step 2000 E_blend=0.657 -> pure=1+ln(0.657)=0.580")]:
    u_opt = 0.0 if e_blend > 1.0 else math.log(e_blend)
    inv_exp_u = math.exp(-u_opt)
    pure = e_blend * inv_exp_u + u_opt
    # Suppose P_high baseline weighted error with lap_sigma=1.0 is ~0.944 * e_vit, multiplied by mean(w_vit)=0.72:
    # At step 0 (before w_vit sharpens or with uniform 0.5/0.5?):
    vit_step0_uncorr = 0.944 * 0.50 * 0.74  # or at u=0, w_vit=0.72 -> 0.944 * 0.72 * e_vit
    print(f"  {label}: u*={u_opt:+.3f}, exp(-u*)={inv_exp_u:.3f}, pure_loss={pure:.3f}, "
          f"vit_loss(if e_vit=0.63*0.72*exp(-u*))={0.63*0.72*inv_exp_u:.3f}")
