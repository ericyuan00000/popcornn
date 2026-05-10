"""Pilot trace for 4 repel-potential variants on rxn0003 stage-1.

Same recipe as the shipped rxn0003 stage-1 (n4d2, lr=1e-3, rtol=1e-1,
atol=1e-4) but with thr=0 (1000-iter cap) and UMA probes every 10
iters. Goal: characterize per-variant convergence so we can pick
per-variant (atol, thr) that fires at each variant's own plateau.

Variants (α=1, β=1 — user-specified, NOT the upstream defaults
α=1.7, β=0.01):

   1. exp_covalent   α=1, β=0, r₀=covalent
   2. invr_covalent  α=0, β=1, r₀=covalent
   3. exp_r0_1.0     α=1, β=0, r₀=1.0 Å (constant)
   4. invr_r0_1.0    α=0, β=1, r₀=1.0 Å (constant)

Constant r₀=1.0 is achieved by monkey-patching `pot.radii.fill_(0.5)`
so each pair's `r0 = self.radii[i] + self.radii[j] = 1.0`. RepelPotential
itself only takes (alpha, beta, cutoff) kwargs; the May-era `r0` kwarg
was a fork-only addition not upstream.
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
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_variants_pilot'

VARIANTS = [
    dict(name='exp_covalent',  alpha=1.0, beta=0.0, r0=None),
    dict(name='invr_covalent', alpha=0.0, beta=1.0, r0=None),
    dict(name='exp_r0_1.0',    alpha=1.0, beta=0.0, r0=1.0),
    dict(name='invr_r0_1.0',   alpha=0.0, beta=1.0, r0=1.0),
]

SEED = 0
N_ITER = 1000
M_PROBE = 11
K_PROBE = 10
M_DENSE_FINAL = 101

RTOL = 1.0e-1
ATOL = 1.0e-4
LR = 1.0e-3
N_EMBED = 4
DEPTH = 2


def build_mep(seed: int):
    cfg = import_run_config(CONFIG)
    init = dict(cfg['initialization_params'])
    if isinstance(init.get('images'), str) and not os.path.isabs(init['images']):
        init['images'] = os.path.join(REPO, 'examples', init['images'])
    init['path_params'] = {'name': 'mlp', 'n_embed': N_EMBED, 'depth': DEPTH, 'activation': 'gelu'}
    init['device'] = 'cuda'
    init['seed'] = seed
    return Popcornn(**init), cfg


def setup_stage1_variant(mep, alpha, beta, r0):
    pot = get_potential(images=mep.images, name='repel', alpha=alpha, beta=beta,
                        device=mep.device, dtype=mep.dtype)
    if r0 is not None:
        # Each pair's r₀ = self.radii[i] + self.radii[j]; setting all radii to r0/2
        # gives r0_pair = r0 for every edge.
        pot.radii.fill_(r0 / 2.0)
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
        threshold=None,
        patience=1,
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
def probe_uma(path, uma, M):
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


def run_variant(v, uma, base_cfg):
    print(f'\n=== variant {v["name"]} (α={v["alpha"]}, β={v["beta"]}, r₀={v["r0"] or "covalent"}) ===', flush=True)
    mep, _ = build_mep(seed=SEED)
    integ, optr = setup_stage1_variant(mep, v['alpha'], v['beta'], v['r0'])

    pilot = {
        'iter': [], 'wall_s': [], 'g_inf': [], 'loss': [],
        'probe_iter': [], 'probe_max_E': [], 'probe_E_per_t': [],
        'probe_fmax_per_t': [], 'probe_max_fmax': [], 'probe_wall_s': [],
        'dense_E_final': [], 'dense_fmax_final': [],
    }

    # Initial probe
    tp = time.perf_counter()
    e, fm = probe_uma(mep.path, uma, M_PROBE)
    torch.cuda.synchronize()
    pilot['probe_iter'].append(0)
    pilot['probe_max_E'].append(max(e))
    pilot['probe_E_per_t'].append(e)
    pilot['probe_fmax_per_t'].append(fm)
    pilot['probe_max_fmax'].append(max(fm))
    pilot['probe_wall_s'].append(time.perf_counter() - tp)
    barrier0 = max(e) - max(e[0], e[-1])
    print(f'  probe iter=0  barrier={barrier0:.4f} eV  max_fmax={max(fm):.3f}', flush=True)

    t_run = time.perf_counter()
    for it in range(N_ITER):
        ts = time.perf_counter()
        out = optr.optimization_step(mep.path, integ)
        torch.cuda.synchronize()
        wall = time.perf_counter() - ts
        gi = float(out.grad_norm.item())
        try:
            lv = float(out.loss[0].item())
        except Exception:
            lv = None
        pilot['iter'].append(it)
        pilot['wall_s'].append(wall)
        pilot['g_inf'].append(gi)
        pilot['loss'].append(lv)

        if (it + 1) % K_PROBE == 0:
            tp = time.perf_counter()
            e, fm = probe_uma(mep.path, uma, M_PROBE)
            torch.cuda.synchronize()
            wp = time.perf_counter() - tp
            pilot['probe_iter'].append(it + 1)
            pilot['probe_max_E'].append(max(e))
            pilot['probe_E_per_t'].append(e)
            pilot['probe_fmax_per_t'].append(fm)
            pilot['probe_max_fmax'].append(max(fm))
            pilot['probe_wall_s'].append(wp)

        if (it + 1) % 100 == 0:
            b = pilot['probe_max_E'][-1] - max(pilot['probe_E_per_t'][-1][0], pilot['probe_E_per_t'][-1][-1])
            print(f'  iter={it+1:4d}  |g|_inf={gi:.4e}  sparse_barrier={b:.4f}  '
                  f'max_fmax={pilot["probe_max_fmax"][-1]:.3f}', flush=True)

    elapsed = time.perf_counter() - t_run
    print(f'  total iter wall={elapsed:.1f}s', flush=True)

    # Final dense eval (M=101) for plateau-quality readout
    print(f'  dense final eval (M={M_DENSE_FINAL})...', flush=True)
    e_dense, fm_dense = probe_uma(mep.path, uma, M_DENSE_FINAL)
    torch.cuda.synchronize()
    pilot['dense_E_final'] = e_dense
    pilot['dense_fmax_final'] = fm_dense
    barrier_dense = max(e_dense) - max(e_dense[0], e_dense[-1])
    print(f'  dense barrier (final) = {barrier_dense:.4f} eV  max_fmax_dense = {max(fm_dense):.3f}', flush=True)

    return pilot


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)
    print(f'Recipe: rtol={RTOL:.1e} atol={ATOL:.1e} lr={LR:.1e} thr=None n_embed={N_EMBED} depth={DEPTH}', flush=True)
    print(f'Pilot: {N_ITER} iters, UMA probe (M={M_PROBE}) every {K_PROBE}, dense final (M={M_DENSE_FINAL})', flush=True)

    print('=== Loading UMA (once, reused across variants) ===', flush=True)
    t0 = time.perf_counter()
    seed_mep, cfg = build_mep(seed=SEED)
    uma = setup_uma(seed_mep, cfg)
    print(f'UMA loaded in {time.perf_counter()-t0:.1f}s', flush=True)
    del seed_mep

    all_pilots = {}
    for v in VARIANTS:
        pilot = run_variant(v, uma, cfg)
        all_pilots[v['name']] = pilot
        with open(f'{OUT_DIR}/pilot_{v["name"]}.json', 'w') as f:
            json.dump(pilot, f)
        print(f'  wrote {OUT_DIR}/pilot_{v["name"]}.json', flush=True)

    # Cross-variant summary
    print('\n=== SUMMARY (final dense barrier per variant) ===', flush=True)
    print(f'{"variant":>20} {"dense barrier":>14} {"max_fmax":>10} {"final |g|_inf":>14}', flush=True)
    for v in VARIANTS:
        p = all_pilots[v['name']]
        e_dense = p['dense_E_final']
        b = max(e_dense) - max(e_dense[0], e_dense[-1])
        fm = max(p['dense_fmax_final'])
        gi = p['g_inf'][-1]
        print(f'{v["name"]:>20} {b:>14.4f} {fm:>10.3f} {gi:>14.4e}', flush=True)


if __name__ == '__main__':
    main()
