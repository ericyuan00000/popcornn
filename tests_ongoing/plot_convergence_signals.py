"""Track loss, ‖∫∇L dt‖_2, ‖∫∇L dt‖_∞ over a tight-integration Adam run on
Müller-Brown so the convergence-signal behavior is visible side by side.

Tight integration (rtol=1e-5, atol=1e-7) so the recorded gradient-norm
trajectory reflects optimizer dynamics, not quadrature noise. lr=1e-3 because
that's the lr at which Adam reaches the lowest |g| plateau in this setup.
"""
import json
import os
import time as time_mod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from popcornn import Popcornn
from popcornn.optimization.path_optimizer import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, 'examples', 'configs', 'muller_brown.yaml')
OUT_DIR = '/global/homes/e/ericyuan/scratch/temp/popcornn_sweep'

RTOL = 1e-5
ATOL = 1e-7
LR = 1e-3
N_STEPS = 500


def main():
    base_cfg = import_run_config(CONFIG)
    leg = base_cfg['optimization_params'][0]
    leg['integrator_params']['rtol'] = RTOL
    leg['integrator_params']['atol'] = ATOL
    leg['integrator_params']['track_loss'] = True
    leg['optimizer_params']['optimizer']['lr'] = LR
    # Strip any convergence trigger so the run goes the full N_STEPS.
    leg['optimizer_params'].pop('threshold', None)
    leg['optimizer_params'].pop('patience', None)

    mep = Popcornn(**base_cfg.get('initialization_params', {}))
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg['integrator_params'],
                          device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    iters, losses, g2, ginf = [], [], [], []
    print(f'rtol={RTOL:.0e}, atol={ATOL:.0e}, lr={LR:.0e}, steps={N_STEPS}')
    print(f'{"iter":>6s} {"loss":>11s} {"|g|_2":>11s} {"|g|_inf":>11s}')
    t0 = time_mod.perf_counter()
    for step in range(N_STEPS):
        out = optr.optimization_step(mep.path, integ)
        flat = out.grad_integral.detach()
        iters.append(step)
        losses.append(float(out.loss[0].item()))
        g2.append(float(flat.norm().item()))
        ginf.append(float(flat.abs().max().item()))
        if step < 10 or step % 25 == 0:
            print(f'{step:>6d} {losses[-1]:>11.4e} {g2[-1]:>11.4e} {ginf[-1]:>11.4e}')
    elapsed = time_mod.perf_counter() - t0
    print(f'\nelapsed: {elapsed:.1f}s ({elapsed / N_STEPS * 1000:.1f} ms/step)')

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, 'convergence_signals.json')
    with open(out_json, 'w') as f:
        json.dump({'iters': iters, 'loss': losses, 'g2': g2, 'ginf': ginf,
                   'rtol': RTOL, 'atol': ATOL, 'lr': LR, 'n_steps': N_STEPS}, f)
    print(f'data: {out_json}')

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].semilogy(iters, losses, color='C0', lw=1.0)
    axes[0].set_ylabel(r'$\int L\,dt$  (path loss)')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].semilogy(iters, g2, color='C1', lw=1.0)
    axes[1].set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_2$')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[2].semilogy(iters, ginf, color='C2', lw=1.0)
    axes[2].set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_\infty$')
    axes[2].set_xlabel('Adam iteration')
    axes[2].grid(True, which='both', alpha=0.3)
    fig.suptitle(f'Convergence signals on Müller-Brown  '
                 f'(rtol={RTOL:.0e}, atol={ATOL:.0e}, lr={LR:.0e}, '
                 f'D=610 MLP params)')
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'convergence_signals.png')
    fig.savefig(out_png, dpi=120)
    print(f'plot: {out_png}')


if __name__ == '__main__':
    main()
