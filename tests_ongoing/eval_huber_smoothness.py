"""Empirical smoothness comparison: pvre vs pvre_squared vs pvre_huber.

The adaptive Gauss–Kronrod quadrature feeds on g(t) = (∂ℓ/∂θ)(t)
along the path. Cost scales with how rough g(t) is — sign-flip jumps
force indefinite refinement (pvre), smooth g costs almost nothing
(pvre_squared), corners are in between.

We build a partly-trained Müller-Brown path (a few iters of
pvre_squared so s(t) = v·F isn't trivially zero everywhere), then on
a dense t-grid evaluate:

  s(t)               = v · F
  (∂ℓ/∂s)(s(t))      for each of {pvre, pvre_squared, pvre_huber}
  ℓ(t)               for each
  Σ |Δ(∂ℓ/∂s)|       discrete total variation of the gradient
                     integrand-shape on the grid (the quadrature-cost
                     proxy: bigger = harder for adaptive GK)
  #zero crossings    of s(t) (where pvre's sign() jumps)
  #δ crossings       of |s(t)|=δ (where Huber's slope corners are)

A jump in (∂ℓ/∂s) shows up as a single-step ΔTV equal to the jump
size, while a continuous corner only contributes O(grid step)
ΔTV — this directly separates "smooth", "cornered", and "jumpy"
integrands without having to instrument the quadrature.
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator
from popcornn.tools.integrand import PATH_INTEGRANDS


DELTA = 1.0
N_T = 5000          # t-grid density
WARMUP_ITERS = 50   # warm path so s(t) crosses zero a few times


def main():
    torch.manual_seed(0)

    # Müller-Brown two-image setup, same as examples/configs/muller_brown.yaml
    init_params = {
        'images': [[-0.558, 1.442], [0.623, 0.028]],
        'path_params': {'name': 'mlp', 'n_embed': 8, 'depth': 4, 'activation': 'gelu'},
    }
    mep = Popcornn(**init_params)
    pot = get_potential(images=mep.images, name='muller_brown',
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)

    # Warm-up with pvre_squared (50 steps, lr=1e-2). Gives a non-trivial
    # path that has both crossings of s=0 (saddle-condition zeros) and
    # crossings of |s|=δ (so all three regimes are exercised).
    integ = PathIntegrator(method='gk21', path_integrand_names='pvre_squared',
                           rtol=1e-1, atol=1e-7,
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path,
                         optimizer={'name': 'adam', 'lr': 1.0e-2},
                         threshold=0.0,
                         device=mep.device, dtype=mep.dtype)
    for _ in range(WARMUP_ITERS):
        optr.optimization_step(mep.path, integ)

    # Dense t-grid; evaluate s(t) = v·F once and reuse for all three losses.
    t = torch.linspace(0.0, 1.0, N_T, device=mep.device, dtype=mep.dtype).unsqueeze(-1)
    po = mep.path(t, return_velocities=True, return_energies=True, return_forces=True)
    v = po.velocities.detach()
    F = po.forces.detach()
    s = (v * F).sum(dim=-1, keepdim=True)            # [N_T, 1]
    abs_s = s.abs()

    s_np = s.cpu().numpy().reshape(-1)
    abs_s_np = abs_s.cpu().numpy().reshape(-1)

    # Counts of features that drive quadrature cost
    n_zero = int(np.sum(np.diff(np.sign(s_np)) != 0))           # pvre kinks
    n_delta = int(np.sum(np.diff(np.sign(abs_s_np - DELTA)) != 0))  # huber corners
    s_min, s_max = float(s_np.min()), float(s_np.max())
    abs_s_max = float(abs_s_np.max())

    # Gradient integrand shape g(t) = (∂ℓ/∂s)(s(t)) — what GK quadratures
    g_pvre = np.sign(s_np)                              # ±1, jumps at zeros
    g_pvre_sq = 2 * s_np                                # smooth
    g_huber = np.clip(s_np, -DELTA, DELTA)              # continuous, corners at ±δ

    # Loss integrand ℓ(t)
    l_pvre = abs_s_np
    l_pvre_sq = s_np ** 2
    l_huber = np.where(abs_s_np <= DELTA,
                       0.5 * s_np ** 2,
                       DELTA * (abs_s_np - 0.5 * DELTA))

    # Total variation of the gradient integrand on the grid — direct
    # proxy for quadrature cost. Jumps contribute their full size in one
    # grid step; smooth and cornered contributions are O(grid step).
    def tv(arr):
        return float(np.sum(np.abs(np.diff(arr))))

    tv_pvre = tv(g_pvre)
    tv_pvre_sq = tv(g_pvre_sq)
    tv_huber = tv(g_huber)

    # Smooth-baseline: how much TV is "intrinsic" to s(t) itself
    tv_s = tv(s_np)

    print(f'\n=== smoothness audit on partly-trained MB path '
          f'(warmup={WARMUP_ITERS} iters of pvre_squared, N_t={N_T}) ===')
    print(f's(t) range: [{s_min:.4f}, {s_max:.4f}]   |s|_max = {abs_s_max:.4f}')
    print(f'crossings: s=0 (pvre kinks)        : {n_zero}')
    print(f'           |s|={DELTA} (huber corners) : {n_delta}')
    print()
    print(f'TV of (∂ℓ/∂s)(s(t)) on grid — quadrature-cost proxy:')
    print(f'  pvre         (sign s)        : {tv_pvre:>10.3f}')
    print(f'  pvre_squared (2s)            : {tv_pvre_sq:>10.3f}')
    print(f'  pvre_huber   (clamp s,±{DELTA}) : {tv_huber:>10.3f}')
    print(f'  reference TV(s) on grid     : {tv_s:>10.3f}')
    print()

    # The headline: how does pvre_huber compare to the two extremes?
    # If TV(huber) ≈ TV(pvre_squared bounded), it's smooth-equivalent.
    # If TV(huber) >> TV(pvre_squared bounded), the corners matter.
    bounded_smooth_tv = tv(np.clip(2 * s_np, -2 * DELTA, 2 * DELTA))
    print(f'For reference, TV of clamp(2s, ±2δ) (smooth bounded): {bounded_smooth_tv:.3f}')

    # Per-step jump check — pvre jumps are O(1) at every zero crossing;
    # huber corners are O(grid step); pvre_sq has no jumps.
    max_jump_pvre = float(np.max(np.abs(np.diff(g_pvre))))
    max_jump_pvre_sq = float(np.max(np.abs(np.diff(g_pvre_sq))))
    max_jump_huber = float(np.max(np.abs(np.diff(g_huber))))
    print()
    print(f'Largest single-step Δ(∂ℓ/∂s) on grid (jump indicator):')
    print(f'  pvre         : {max_jump_pvre:.4e}   ← O(1) flip, every zero crossing')
    print(f'  pvre_squared : {max_jump_pvre_sq:.4e}   ← O(grid step), smooth')
    print(f'  pvre_huber   : {max_jump_huber:.4e}   ← O(grid step) (corners, no jumps)')

    print()
    print('Verdict:')
    print(f'  pvre has {n_zero} sign-flip jumps; pvre_huber has 0 jumps')
    print(f'  (only continuous corners at {n_delta} ±δ crossings).')
    print(f'  TV ratio pvre/pvre_huber = {tv_pvre / max(1e-9, tv_huber):.1f}x   '
          f'(higher = harder quadrature for pvre)')
    print(f'  TV ratio pvre_huber/pvre_squared = '
          f'{tv_huber / max(1e-9, tv_pvre_sq):.2f}x   '
          f'(close to bounded-clip ratio = {bounded_smooth_tv / max(1e-9, tv_pvre_sq):.2f}x)')


if __name__ == '__main__':
    main()
