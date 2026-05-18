"""Full ablation: 9 repel variants × 2 regimes (non-rotated, rotated) × 3 seeds.
1000 iters per run, n4d2/lr=1e-3/rtol=1e-1/atol=1e-4/thr=None recipe.
UMA evaluated densely (M=101) only at end of each run.

Variants (adds `mix` to the prior 8):
   1. exp_covalent      α=1.0, β=0,    r₀=covalent
   2. invr_covalent     α=0,   β=1.0,  r₀=covalent
   3. exp_r0_1.0        α=1.0, β=0,    r₀=1.0
   4. invr_r0_1.0       α=0,   β=1.0,  r₀=1.0
   5. gauss_covalent    α=1.0, gauss,  r₀=covalent
   6. gauss_r0_1.0      α=1.0, gauss,  r₀=1.0
   7. exp_a1p7_covalent α=1.7, β=0,    r₀=covalent
   8. exp_a1p7_r0_1.0   α=1.7, β=0,    r₀=1.0
   9. mix               α=1.7, β=0.01, r₀=covalent  (upstream default)

Saves per-run JSON to OUT_DIR/pilot_<regime>_<variant>_seed<n>.json.
Prints a 9×2 mean±std summary at end.
"""
import json
import os
import time
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import torch
from ase.io import read
from torch_geometric.utils import to_dense_batch
from ase.data import covalent_radii

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.potentials.base_potential import BasePotential, PotentialOutput
from popcornn.tools import PathIntegrator, import_run_config, radius_graph

REPO = '/global/u2/e/ericyuan/GitHub/Popcornn'
XYZ = f'{REPO}/examples/configs/rxn0003.xyz'
CONFIG = f'{REPO}/examples/configs/rxn0003.yaml'
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_full_pilot'

VARIANTS = [
    dict(name='exp_covalent',      kind='repel',    alpha=1.0, beta=0.0,  r0=None),
    dict(name='invr_covalent',     kind='repel',    alpha=0.0, beta=1.0,  r0=None),
    dict(name='exp_r0_1.0',        kind='repel',    alpha=1.0, beta=0.0,  r0=1.0),
    dict(name='invr_r0_1.0',       kind='repel',    alpha=0.0, beta=1.0,  r0=1.0),
    dict(name='gauss_covalent',    kind='gaussian', alpha=1.0, beta=None, r0=None),
    dict(name='gauss_r0_1.0',      kind='gaussian', alpha=1.0, beta=None, r0=1.0),
    dict(name='exp_a1p7_covalent', kind='repel',    alpha=1.7, beta=0.0,  r0=None),
    dict(name='exp_a1p7_r0_1.0',   kind='repel',    alpha=1.7, beta=0.0,  r0=1.0),
    dict(name='mix',               kind='repel',    alpha=1.7, beta=0.01, r0=None),
]

REGIMES = ['non_rot', 'rot']
SEEDS = [0, 1, 2]
N_ITER = 1000
M_DENSE_FINAL = 101

RTOL = 1.0e-1
ATOL = 1.0e-4
LR = 1.0e-3
N_EMBED = 4
DEPTH = 2


class GaussianRepelPotential(BasePotential):
    def __init__(self, alpha=1.0, cutoff=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.cutoff = cutoff
        self.radii = torch.tensor(
            [covalent_radii[n] for n in self.atomic_numbers],
            device=self.device, dtype=self.dtype,
        )

    def forward(self, positions):
        positions_3d = positions.view(-1, self.n_atoms, 3)
        n_data, n_atoms, _ = positions_3d.shape
        graph_dict = radius_graph(positions=positions_3d, cell=self.cell, pbc=self.pbc, cutoff=self.cutoff, max_neighbors=-1)
        r = graph_dict['edge_distance']
        v = graph_dict['edge_distance_vec']
        r0 = self.radii[graph_dict['edge_index'] % n_atoms].sum(dim=0)
        e = torch.exp(-self.alpha * (r / r0) ** 2)
        if self.cutoff is not None:
            e -= torch.exp(-self.alpha * (self.cutoff / r0) ** 2)
        energies_decomposed, _ = to_dense_batch(e, batch=graph_dict['edge_index'][1] // n_atoms)
        energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)
        f = (-2.0 * self.alpha / r0 ** 2 * e).unsqueeze(-1) * v
        forces_decomposed = torch.zeros(len(f), n_atoms, 3, device=self.device, dtype=self.dtype)
        forces_decomposed[torch.arange(len(f), device=self.device), graph_dict['edge_index'][0] % n_atoms] = -f
        forces_decomposed[torch.arange(len(f), device=self.device), graph_dict['edge_index'][1] % n_atoms] = f
        forces_decomposed, _ = to_dense_batch(forces_decomposed, batch=graph_dict['edge_index'][1] // n_atoms)
        forces_decomposed = forces_decomposed.view(*forces_decomposed.shape[:-2], -1)
        forces = torch.sum(forces_decomposed, dim=-2, keepdim=False)
        return PotentialOutput(energies=energies, energies_decomposed=energies_decomposed,
                               forces=forces, forces_decomposed=forces_decomposed)


def load_images(rotated: bool):
    atoms = read(XYZ, index=':')
    reactant = atoms[0].copy()
    product = atoms[-1].copy()
    if rotated:
        com = product.get_center_of_mass()
        pos = product.get_positions() - com
        R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        product.set_positions(pos @ R.T + com)
    return [reactant, product]


def build_mep(seed: int, images):
    return Popcornn(
        images=images,
        path_params={'name': 'mlp', 'n_embed': N_EMBED, 'depth': DEPTH, 'activation': 'gelu'},
        num_record_points=101,
        device='cuda',
        seed=seed,
    )


def setup_stage1_variant(mep, v):
    if v['kind'] == 'repel':
        pot = get_potential(images=mep.images, name='repel', alpha=v['alpha'], beta=v['beta'],
                            device=mep.device, dtype=mep.dtype)
    else:
        pot = GaussianRepelPotential(images=mep.images, alpha=v['alpha'],
                                     device=mep.device, dtype=mep.dtype)
    if v['r0'] is not None:
        pot.radii.fill_(v['r0'] / 2.0)
    mep.path.set_potential(pot)
    integ = PathIntegrator(path_integrand_names='geodesic', rtol=RTOL, atol=ATOL,
                           device=mep.device, dtype=mep.dtype)
    integ.save_samples = False
    optr = PathOptimizer(path=mep.path, optimizer={'name': 'adam', 'lr': LR},
                         threshold=None, patience=1, find_ts=False,
                         device=mep.device, dtype=mep.dtype)
    return integ, optr


def setup_uma(mep):
    cfg = import_run_config(CONFIG)
    leg2 = cfg['optimization_params'][1]
    return get_potential(images=mep.images, **leg2['potential_params'],
                         device=mep.device, dtype=mep.dtype)


@torch.no_grad()
def dense_eval(path, uma, M):
    device = path.initial_position.device
    dtype = path.initial_position.dtype
    times = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype).unsqueeze(-1)
    positions = path.get_positions(times)
    out = uma(positions)
    energies = out.energies.detach().cpu().squeeze(-1).tolist()
    f_atom = out.forces.detach().view(out.forces.shape[0], -1, 3)
    fmax_per_t = f_atom.norm(dim=-1).amax(dim=-1).cpu().tolist()
    return energies, fmax_per_t


def run_one(v, regime, seed, uma, images):
    label = f'{regime}/{v["name"]}/seed{seed}'
    mep = build_mep(seed=seed, images=images)
    integ, optr = setup_stage1_variant(mep, v)

    t_start = time.perf_counter()
    g_inf_trace = []
    for it in range(N_ITER):
        out = optr.optimization_step(mep.path, integ)
        torch.cuda.synchronize()
        g_inf_trace.append(float(out.grad_norm.item()))
    wall = time.perf_counter() - t_start

    e_dense, fm_dense = dense_eval(mep.path, uma, M_DENSE_FINAL)
    torch.cuda.synchronize()
    barrier = max(e_dense) - max(e_dense[0], e_dense[-1])
    fmax = max(fm_dense)
    g_final = g_inf_trace[-1]

    print(f'  {label:>40}  barrier={barrier:>7.3f}  fmax={fmax:>6.3f}  '
          f'|g|_inf={g_final:>9.2e}  wall={wall:>5.1f}s', flush=True)
    return dict(barrier=barrier, fmax=fmax, g_final=g_final, wall=wall,
                e_dense=e_dense, fm_dense=fm_dense, g_inf_trace=g_inf_trace)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)
    print(f'9 variants × 2 regimes × 3 seeds = 54 runs', flush=True)
    print(f'Recipe: rtol={RTOL:.1e} atol={ATOL:.1e} lr={LR:.1e} thr=None n_embed={N_EMBED} depth={DEPTH}\n', flush=True)

    print('=== Loading UMA ===', flush=True)
    t0 = time.perf_counter()
    images_init = load_images(rotated=False)
    seed_mep = build_mep(seed=0, images=images_init)
    uma = setup_uma(seed_mep)
    print(f'UMA loaded in {time.perf_counter()-t0:.1f}s\n', flush=True)
    del seed_mep
    torch.cuda.empty_cache()

    results = defaultdict(list)  # (regime, variant) -> [seed-result, ...]
    for regime in REGIMES:
        images = load_images(rotated=(regime == 'rot'))
        # Initial barrier reference
        ref_mep = build_mep(seed=0, images=images)
        e0, _ = dense_eval(ref_mep.path, uma, M_DENSE_FINAL)
        init_b = max(e0) - max(e0[0], e0[-1])
        print(f'\n#### regime={regime}  init dense barrier = {init_b:.2f} eV ####\n', flush=True)
        del ref_mep
        torch.cuda.empty_cache()

        for v in VARIANTS:
            for seed in SEEDS:
                r = run_one(v, regime, seed, uma, images)
                results[(regime, v['name'])].append(dict(seed=seed, **r))
                with open(f'{OUT_DIR}/pilot_{regime}_{v["name"]}_seed{seed}.json', 'w') as f:
                    json.dump({'seed': seed, 'regime': regime, 'variant': v, **r}, f)

    # === Summary ===
    print('\n\n========== 3-seed mean±std summary ==========', flush=True)
    for regime in REGIMES:
        print(f'\n--- regime: {regime} ---', flush=True)
        print(f'{"variant":>22}  {"barrier (mean±std)":>22}  {"fmax (mean±std)":>22}  {"wall_s mean":>12}', flush=True)
        # Sort by mean barrier
        items = []
        for v in VARIANTS:
            seeds = results[(regime, v['name'])]
            bs = [s['barrier'] for s in seeds]
            fs = [s['fmax'] for s in seeds]
            ws = [s['wall'] for s in seeds]
            bm, bs_ = mean(bs), stdev(bs) if len(bs) > 1 else 0.0
            fm, fs_ = mean(fs), stdev(fs) if len(fs) > 1 else 0.0
            wm = mean(ws)
            items.append((v['name'], bm, bs_, fm, fs_, wm, bs))
        items.sort(key=lambda x: x[1])
        for name, bm, bs_, fm, fs_, wm, bs in items:
            print(f'{name:>22}  {bm:>10.3f} ± {bs_:>6.3f}    {fm:>9.3f} ± {fs_:>5.3f}      {wm:>10.2f}', flush=True)

    summary = {regime: {v['name']: results[(regime, v['name'])] for v in VARIANTS} for regime in REGIMES}
    with open(f'{OUT_DIR}/full_summary.json', 'w') as f:
        json.dump(summary, f)
    print(f'\nwrote {OUT_DIR}/full_summary.json', flush=True)


if __name__ == '__main__':
    main()
