"""Compare convergence signals across MLP sizes on Müller-Brown.

Same tight integration (rtol=1e-5, atol=1e-7) and lr=1e-3 across configs. Each
run trains the path-MLP for N_STEPS Adam steps and records loss, ‖g‖_2, ‖g‖_∞
per step. All trajectories overlaid on a 3-panel comparison plot so the user
can read off whether the convergence-signal shape — descent / peak / crash /
plateau — is invariant under MLP size.
"""
import copy
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
N_STEPS = 400

# (n_embed, depth) — span small / current / big in both width and depth.
MLP_CONFIGS = [
    (4, 4),
    (8, 4),    # current default
    (16, 4),
    (8, 2),
    (8, 8),
]


def run_mlp(n_embed, depth, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg['initialization_params']['path_params']['n_embed'] = n_embed
    cfg['initialization_params']['path_params']['depth'] = depth
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['rtol'] = RTOL
    leg['integrator_params']['atol'] = ATOL
    leg['integrator_params']['track_loss'] = True
    leg['optimizer_params']['optimizer']['lr'] = LR
    leg['optimizer_params'].pop('threshold', None)
    leg['optimizer_params'].pop('patience', None)

    mep = Popcornn(**cfg.get('initialization_params', {}))
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg['integrator_params'],
                          device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)
    n_params = sum(p.numel() for p in mep.path.parameters())
    print(f'\n=== n_embed={n_embed}, depth={depth} (D={n_params}) ===')
    print(f'{"iter":>6s} {"loss":>11s} {"|g|_2":>11s} {"|g|_inf":>11s}')

    losses, g2, ginf = [], [], []
    t0 = time_mod.perf_counter()
    for step in range(N_STEPS):
        out = optr.optimization_step(mep.path, integ)
        flat = out.grad_integral.detach()
        losses.append(float(out.loss[0].item()))
        g2.append(float(flat.norm().item()))
        ginf.append(float(flat.abs().max().item()))
        if step in (0, 25, 50, 75, 100, 200, 300, N_STEPS - 1):
            print(f'{step:>6d} {losses[-1]:>11.4e} {g2[-1]:>11.4e} {ginf[-1]:>11.4e}')
    elapsed = time_mod.perf_counter() - t0
    print(f'elapsed: {elapsed:.1f}s ({elapsed / N_STEPS * 1000:.1f} ms/step)')
    return {'n_embed': n_embed, 'depth': depth, 'D': n_params,
            'loss': losses, 'g2': g2, 'ginf': ginf}


def main():
    base_cfg = import_run_config(CONFIG)
    results = [run_mlp(ne, d, base_cfg) for ne, d in MLP_CONFIGS]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, 'convergence_signals_mlp_sweep.json')
    with open(out_json, 'w') as f:
        json.dump({'rtol': RTOL, 'atol': ATOL, 'lr': LR, 'n_steps': N_STEPS,
                   'results': results}, f)
    print(f'\ndata: {out_json}')

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    iters = list(range(N_STEPS))
    for r in results:
        label = f'n_embed={r["n_embed"]}, depth={r["depth"]}, D={r["D"]}'
        axes[0].semilogy(iters, r['loss'], lw=1.0, label=label)
        axes[1].semilogy(iters, r['g2'], lw=1.0, label=label)
        axes[2].semilogy(iters, r['ginf'], lw=1.0, label=label)
    axes[0].set_ylabel(r'$\int L\,dt$  (path loss)')
    axes[1].set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_2$')
    axes[2].set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_\infty$')
    axes[2].set_xlabel('Adam iteration')
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Convergence signals across MLP sizes  '
                 f'(rtol={RTOL:.0e}, atol={ATOL:.0e}, lr={LR:.0e})')
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'convergence_signals_mlp_sweep.png')
    fig.savefig(out_png, dpi=120)
    print(f'plot: {out_png}')


if __name__ == '__main__':
    main()
