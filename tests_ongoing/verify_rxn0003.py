"""Verify the new YAML picks transfer to a real-system run (rxn0003 + UMA).

Stage 1 (repel + geodesic, lr=1e-1, 1000 iters): warmup, unchanged.
Stage 2 (UMA + projected_variational_reaction_energy): apply the picks
derived from the Müller-Brown analysis:

    rtol = atol = 1e-2          (was 1e-5/1e-7)
    lr = 1e-3                   (unchanged)
    grad_norm_tol = 1.0         (L∞)
    grad_norm_patience = 5
    num_optimizer_iterations = 500

MLP unchanged from the YAML (n_embed=1, depth=2 — tiny by design for this
example). The convergence trigger inside PathOptimizer is left active so we
record when it would fire under production semantics, but we don't break
the outer loop on it — the script continues to the iter cap so the
post-trigger trajectory is visible on the plot. That's how we tell whether
the trigger cut us off too early.
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
from popcornn.tools import ODEintegrator, import_run_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, 'examples', 'configs', 'rxn0003.yaml')
OUT_DIR = '/global/homes/e/ericyuan/scratch/temp/popcornn_sweep'

STAGE2_OVERRIDES = {
    'rtol': 1e-2,
    'atol': 1e-2,
    'lr': 1e-3,
    'grad_norm_tol': 1.0,
    'grad_norm_patience': 5,
    'num_optimizer_iterations': 1000,
}


def run_stage(mep, idx, leg, instrument):
    leg = copy.deepcopy(leg)
    leg.setdefault('integrator_params', {})['track_loss'] = instrument
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = ODEintegrator(**leg['integrator_params'],
                          device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    n_iters = leg['num_optimizer_iterations']
    n_params = sum(p.numel() for p in mep.path.parameters())
    print(f'\n=== Stage {idx}: D={n_params}, n_iters={n_iters} '
          f'({"instrumented" if instrument else "warmup"}) ===')

    losses, g2, ginf, walls, n_nodes = [], [], [], [], []
    converged_at = None
    if instrument:
        print(f'{"iter":>6s} {"loss":>11s} {"|g|_2":>11s} {"|g|_inf":>11s} '
              f'{"nodes":>6s} {"sec/it":>9s}')
    t0 = time_mod.perf_counter()
    for step in range(n_iters):
        ts = time_mod.perf_counter()
        out = optr.optimization_step(mep.path, integ)
        wall = time_mod.perf_counter() - ts
        if not instrument:
            continue
        flat = out.integral.detach()
        gi = float(flat.abs().max().item())
        g2v = float(flat.norm().item())
        lv = float(out.loss_integral[0].item())
        losses.append(lv); g2.append(g2v); ginf.append(gi)
        walls.append(wall); n_nodes.append(int(out.t.shape[0]))
        if optr.converged and converged_at is None:
            converged_at = step
            print(f'  *** convergence trigger fired at step {step} ***')
        if (step < 10 or step % 20 == 0 or step == n_iters - 1
                or step == converged_at):
            print(f'{step:>6d} {lv:>11.4e} {g2v:>11.4e} {gi:>11.4e} '
                  f'{n_nodes[-1]:>6d} {wall:>8.3f}s')
    elapsed = time_mod.perf_counter() - t0
    print(f'stage {idx} elapsed: {elapsed:.1f}s')
    return {
        'losses': losses, 'g2': g2, 'ginf': ginf, 'walls': walls,
        'n_nodes': n_nodes, 'converged_at': converged_at,
        'elapsed': elapsed, 'D': n_params,
    }


def main():
    base_cfg = import_run_config(CONFIG)
    init_params = base_cfg.get('initialization_params', {})
    legs = base_cfg.get('optimization_params', [])

    # Apply stage-2 overrides.
    legs[1]['integrator_params']['rtol'] = STAGE2_OVERRIDES['rtol']
    legs[1]['integrator_params']['atol'] = STAGE2_OVERRIDES['atol']
    legs[1]['optimizer_params']['optimizer']['lr'] = STAGE2_OVERRIDES['lr']
    legs[1]['optimizer_params']['grad_norm_tol'] = STAGE2_OVERRIDES['grad_norm_tol']
    legs[1]['optimizer_params']['grad_norm_patience'] = STAGE2_OVERRIDES['grad_norm_patience']
    legs[1]['num_optimizer_iterations'] = STAGE2_OVERRIDES['num_optimizer_iterations']

    mep = Popcornn(**init_params)
    print(f'rxn0003 verification on device={mep.device}, dtype={mep.dtype}')

    s1 = run_stage(mep, 1, legs[0], instrument=False)
    s2 = run_stage(mep, 2, legs[1], instrument=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, 'rxn0003_verify.json')
    with open(out_json, 'w') as f:
        json.dump({'stage2_overrides': STAGE2_OVERRIDES,
                   'stage1_elapsed': s1['elapsed'], 'stage2': s2}, f)
    print(f'\ndata: {out_json}')

    iters = list(range(len(s2['ginf'])))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].semilogy(iters, s2['losses'], color='C0', lw=1.0)
    axes[0].set_ylabel(r'$\int L\,dt$')
    axes[1].semilogy(iters, s2['g2'], color='C1', lw=1.0)
    axes[1].set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_2$')
    axes[2].semilogy(iters, s2['ginf'], color='C2', lw=1.0)
    axes[2].axhline(STAGE2_OVERRIDES['grad_norm_tol'], color='gray', ls='--',
                    alpha=0.6, label=f'grad_norm_tol={STAGE2_OVERRIDES["grad_norm_tol"]:.0e}')
    axes[2].set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_\infty$')
    axes[2].set_xlabel('Stage 2 iteration')
    axes[2].legend(loc='best', fontsize=8)
    if s2['converged_at'] is not None:
        for ax in axes:
            ax.axvline(s2['converged_at'], color='green', ls=':', alpha=0.6)
        axes[0].text(s2['converged_at'], axes[0].get_ylim()[1],
                     f'  trigger@{s2["converged_at"]}',
                     color='green', fontsize=9, va='top')
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle(f'rxn0003 stage-2 verification  '
                 f'(rtol=atol={STAGE2_OVERRIDES["rtol"]:.0e}, '
                 f'lr={STAGE2_OVERRIDES["lr"]:.0e}, D={s2["D"]})')
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'rxn0003_verify.png')
    fig.savefig(out_png, dpi=120)
    print(f'plot: {out_png}')


if __name__ == '__main__':
    main()
