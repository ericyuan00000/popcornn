"""3-seed validation of the rxn0003 stage-1 recipe (lj13 transfer).

Recipe (transferred verbatim from lj13.yaml):
  MLP n_embed=4 / depth=2 / gelu
  potential = repel; integrand = geodesic
  rtol=1e-1, atol=1e-4
  optimizer = adam, lr=1e-3
  threshold = 1e-3, patience = 1   (single-shot, no extra patience)

For each seed:
  1. Build Popcornn fresh.
  2. Run stage-1 until the |g|_∞ < threshold trigger fires
     (or hit MAX_ITER). This uses popcornn's native trigger; no
     UMA in the loop — keeps the optimization cheap.
  3. After convergence, evaluate UMA densely (M=101) on the path:
       barrier = max_E - max(E_R, E_P)
       max_fmax = max over t of (max over atoms of |f_atom|)
  4. Compare to target: barrier < 3.6 eV (dense plateau per user note;
     the M=11 sparse pilot underestimates by ~0.1 eV).

UMA is loaded once and reused across seeds (atomic_numbers / cell /
pbc are the same since the source xyz doesn't change).
"""
import json
import os
import time

import torch

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config

REPO = '/global/u2/e/ericyuan/GitHub/Popcornn'
CONFIG = f'{REPO}/examples/configs/rxn0003.yaml'
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_validate'

SEEDS = [0, 1, 2]
MAX_ITER = 1000
M_DENSE = 101

RTOL = 1.0e-1
ATOL = 1.0e-4
LR = 1.0e-3
THR = 1.0e-3
PATIENCE = 1
N_EMBED = 4
DEPTH = 2

TARGET_BARRIER = 3.6


def build_mep(seed: int):
    cfg = import_run_config(CONFIG)
    init = dict(cfg['initialization_params'])
    if isinstance(init.get('images'), str) and not os.path.isabs(init['images']):
        init['images'] = os.path.join(REPO, 'examples', init['images'])
    init['path_params'] = {'name': 'mlp', 'n_embed': N_EMBED, 'depth': DEPTH, 'activation': 'gelu'}
    init['device'] = 'cuda'
    init['seed'] = seed
    return Popcornn(**init), cfg


def setup_stage1(mep):
    pot = get_potential(images=mep.images, name='repel', device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(
        path_integrand_names='geodesic',
        rtol=RTOL,
        atol=ATOL,
        device=mep.device,
        dtype=mep.dtype,
    )
    integ.save_samples = False
    optr = PathOptimizer(
        path=mep.path,
        optimizer={'name': 'adam', 'lr': LR},
        threshold=THR,
        patience=PATIENCE,
        find_ts=False,
        device=mep.device,
        dtype=mep.dtype,
    )
    return integ, optr


def setup_uma(mep, base_cfg):
    leg2 = base_cfg['optimization_params'][1]
    return get_potential(images=mep.images, **leg2['potential_params'],
                         device=mep.device, dtype=mep.dtype)


@torch.no_grad()
def dense_uma_eval(path, uma, M):
    device = path.initial_position.device
    dtype = path.initial_position.dtype
    times = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype).unsqueeze(-1)
    positions = path.get_positions(times)
    out = uma(positions)
    energies = out.energies.detach().cpu().squeeze(-1).tolist()
    forces = out.forces.detach()
    f_atom = forces.view(forces.shape[0], -1, 3)
    fmax_per_t = f_atom.norm(dim=-1).amax(dim=-1).cpu().tolist()
    return energies, fmax_per_t


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)
    print(f'Recipe: rtol={RTOL:.1e} atol={ATOL:.1e} lr={LR:.1e} thr={THR:.1e} '
          f'patience={PATIENCE} n_embed={N_EMBED} depth={DEPTH} max_iter={MAX_ITER}', flush=True)
    print(f'Target: dense barrier < {TARGET_BARRIER} eV', flush=True)

    print('=== Loading UMA (once, reused across seeds) ===', flush=True)
    t0 = time.perf_counter()
    seed0_mep, cfg = build_mep(seed=SEEDS[0])
    uma = setup_uma(seed0_mep, cfg)
    print(f'UMA loaded in {time.perf_counter()-t0:.1f}s', flush=True)

    results = []
    for seed in SEEDS:
        print(f'\n=== seed={seed} ===', flush=True)
        if seed == SEEDS[0]:
            mep = seed0_mep
        else:
            mep, _ = build_mep(seed=seed)
        integ, optr = setup_stage1(mep)

        t_start = time.perf_counter()
        stop_iter = None
        last_g = None
        for it in range(MAX_ITER):
            out = optr.optimization_step(mep.path, integ)
            torch.cuda.synchronize()
            last_g = float(out.grad_norm.item())
            if optr.converged and stop_iter is None:
                stop_iter = it + 1
                break
            if (it + 1) % 50 == 0:
                print(f'  iter={it+1:4d}  |g|_inf={last_g:.4e}', flush=True)
        wall = time.perf_counter() - t_start

        if stop_iter is None:
            print(f'  ! did not converge within MAX_ITER={MAX_ITER}; last |g|_inf={last_g:.4e}', flush=True)
            stop_iter = MAX_ITER

        print(f'  stopped iter={stop_iter}  |g|_inf={last_g:.4e}  wall={wall:.2f}s', flush=True)
        print(f'  dense UMA eval (M={M_DENSE})...', flush=True)
        t_eval = time.perf_counter()
        energies, fmax_per_t = dense_uma_eval(mep.path, uma, M_DENSE)
        torch.cuda.synchronize()
        eval_wall = time.perf_counter() - t_eval

        E_R, E_P = energies[0], energies[-1]
        E_max = max(energies)
        barrier = E_max - max(E_R, E_P)
        max_fmax = max(fmax_per_t)
        i_max = energies.index(E_max)
        t_max = i_max / (M_DENSE - 1)
        passed = barrier < TARGET_BARRIER

        print(f'  barrier (dense) = {barrier:.4f} eV   {"✓ PASS" if passed else "✗ FAIL"}', flush=True)
        print(f'  max_fmax = {max_fmax:.3f} eV/Å   E_max @ t={t_max:.3f}', flush=True)
        print(f'  UMA eval wall = {eval_wall:.2f}s', flush=True)

        results.append({
            'seed': seed,
            'stop_iter': stop_iter,
            'last_g_inf': last_g,
            'wall_s': wall,
            'eval_wall_s': eval_wall,
            'barrier_eV': barrier,
            'max_fmax_eV_per_A': max_fmax,
            'E_max': E_max,
            'E_R': E_R,
            'E_P': E_P,
            't_argmax': t_max,
            'passed': passed,
        })

    print('\n=== SUMMARY ===', flush=True)
    print(f'{"seed":>4} {"stop_iter":>10} {"|g|_inf":>10} {"barrier(eV)":>12} {"max_fmax":>10} {"wall(s)":>8} {"PASS?":>6}', flush=True)
    for r in results:
        print(f'{r["seed"]:>4} {r["stop_iter"]:>10} {r["last_g_inf"]:>10.3e} '
              f'{r["barrier_eV"]:>12.4f} {r["max_fmax_eV_per_A"]:>10.3f} '
              f'{r["wall_s"]:>8.2f} {"PASS" if r["passed"] else "FAIL":>6}', flush=True)

    n_pass = sum(r['passed'] for r in results)
    print(f'\n{n_pass}/{len(SEEDS)} seeds satisfy barrier < {TARGET_BARRIER} eV', flush=True)
    out = {
        'recipe': {'rtol': RTOL, 'atol': ATOL, 'lr': LR, 'thr': THR, 'patience': PATIENCE,
                   'n_embed': N_EMBED, 'depth': DEPTH, 'max_iter': MAX_ITER},
        'target_barrier': TARGET_BARRIER,
        'M_dense': M_DENSE,
        'results': results,
    }
    with open(f'{OUT_DIR}/results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {OUT_DIR}/results.json', flush=True)


if __name__ == '__main__':
    main()
