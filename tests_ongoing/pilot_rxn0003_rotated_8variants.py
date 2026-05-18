"""Rotated-product stress test: all 8 repel variants on rxn0003 with
the product rotated 180° around z about its COM. Same recipe as the
non-rotated set (n4d2, lr=1e-3, rtol=1e-1, atol=1e-4, thr=None,
1000 iters, seed=0) so results are directly comparable.

Variants:
   1. exp_covalent      α=1.0, β=0,    r₀=covalent
   2. invr_covalent     α=0,   β=1.0,  r₀=covalent
   3. exp_r0_1.0        α=1.0, β=0,    r₀=1.0
   4. invr_r0_1.0       α=0,   β=1.0,  r₀=1.0
   5. gauss_covalent    α=1.0, gauss,  r₀=covalent
   6. gauss_r0_1.0      α=1.0, gauss,  r₀=1.0
   7. exp_a1p7_covalent α=1.7, β=0,    r₀=covalent
   8. exp_a1p7_r0_1.0   α=1.7, β=0,    r₀=1.0

The rotation: R = diag(-1, -1, 1) (180° about z); applied to product
positions in COM frame, then COM added back. Reactant unchanged.
"""
import json
import os
import time

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
CONFIG = f'{REPO}/examples/configs/rxn0003.yaml'   # for UMA setup metadata
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_rotated_pilot'

VARIANTS = [
    dict(name='exp_covalent',      kind='repel',    alpha=1.0, beta=0.0, r0=None),
    dict(name='invr_covalent',     kind='repel',    alpha=0.0, beta=1.0, r0=None),
    dict(name='exp_r0_1.0',        kind='repel',    alpha=1.0, beta=0.0, r0=1.0),
    dict(name='invr_r0_1.0',       kind='repel',    alpha=0.0, beta=1.0, r0=1.0),
    dict(name='gauss_covalent',    kind='gaussian', alpha=1.0, beta=None, r0=None),
    dict(name='gauss_r0_1.0',      kind='gaussian', alpha=1.0, beta=None, r0=1.0),
    dict(name='exp_a1p7_covalent', kind='repel',    alpha=1.7, beta=0.0, r0=None),
    dict(name='exp_a1p7_r0_1.0',   kind='repel',    alpha=1.7, beta=0.0, r0=1.0),
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


class GaussianRepelPotential(BasePotential):
    """E_ij = exp(-alpha (r/r0)^2). Same shape conventions as RepelPotential."""

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
        graph_dict = radius_graph(
            positions=positions_3d,
            cell=self.cell,
            pbc=self.pbc,
            cutoff=self.cutoff,
            max_neighbors=-1,
        )
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

        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=forces,
            forces_decomposed=forces_decomposed,
        )


def load_rotated_images():
    atoms = read(XYZ, index=':')
    print(f'  loaded {len(atoms)} frames from {XYZ}', flush=True)
    reactant = atoms[0].copy()
    product = atoms[-1].copy()

    # Rotate product 180° around z about its COM
    com = product.get_center_of_mass()
    pos = product.get_positions() - com
    R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    pos = pos @ R.T
    product.set_positions(pos + com)

    com_check = product.get_center_of_mass()
    print(f'  rotated product COM ({com_check[0]:.4f}, {com_check[1]:.4f}, {com_check[2]:.4f}) — should match original ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})', flush=True)
    return [reactant, product]


def build_mep(seed: int, images):
    return Popcornn(
        images=images,
        path_params={'name': 'mlp', 'n_embed': N_EMBED, 'depth': DEPTH, 'activation': 'gelu'},
        num_record_points=101,
        device='cuda',
        seed=seed,
    ), None


def setup_stage1_variant(mep, v):
    if v['kind'] == 'repel':
        pot = get_potential(images=mep.images, name='repel', alpha=v['alpha'], beta=v['beta'],
                            device=mep.device, dtype=mep.dtype)
    elif v['kind'] == 'gaussian':
        pot = GaussianRepelPotential(images=mep.images, alpha=v['alpha'],
                                     device=mep.device, dtype=mep.dtype)
    else:
        raise ValueError(f'unknown kind: {v["kind"]}')
    if v['r0'] is not None:
        pot.radii.fill_(v['r0'] / 2.0)
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


def setup_uma(mep):
    cfg = import_run_config(CONFIG)
    leg2 = cfg['optimization_params'][1]
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


def run_variant(v, uma, images):
    print(f'\n=== variant {v["name"]} (kind={v["kind"]}, α={v["alpha"]}, β={v["beta"]}, r₀={v["r0"] or "covalent"}) ===', flush=True)
    mep, _ = build_mep(seed=SEED, images=images)
    integ, optr = setup_stage1_variant(mep, v)

    pilot = {
        'iter': [], 'wall_s': [], 'g_inf': [], 'loss': [],
        'probe_iter': [], 'probe_max_E': [], 'probe_E_per_t': [],
        'probe_fmax_per_t': [], 'probe_max_fmax': [], 'probe_wall_s': [],
        'dense_E_final': [], 'dense_fmax_final': [],
    }

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

    print('=== Building rotated product ===', flush=True)
    images = load_rotated_images()

    print('=== Loading UMA (once, reused across variants) ===', flush=True)
    t0 = time.perf_counter()
    seed_mep, _ = build_mep(seed=SEED, images=images)
    uma = setup_uma(seed_mep)
    print(f'UMA loaded in {time.perf_counter()-t0:.1f}s', flush=True)

    # Initial-path probe (linear interp on rotated endpoints — should be very high)
    e0, fm0 = probe_uma(seed_mep.path, uma, M_DENSE_FINAL)
    init_barrier = max(e0) - max(e0[0], e0[-1])
    print(f'  rotated linear-init barrier (dense) = {init_barrier:.2f} eV  '
          f'max_fmax = {max(fm0):.2f} eV/Å', flush=True)

    del seed_mep
    torch.cuda.empty_cache()

    all_pilots = {}
    for v in VARIANTS:
        pilot = run_variant(v, uma, images)
        all_pilots[v['name']] = pilot
        with open(f'{OUT_DIR}/pilot_{v["name"]}.json', 'w') as f:
            json.dump(pilot, f)
        print(f'  wrote {OUT_DIR}/pilot_{v["name"]}.json', flush=True)

    print('\n=== SUMMARY (rotated rxn0003, all 8 variants) ===', flush=True)
    print(f'{"variant":>22} {"dense barrier":>14} {"max_fmax":>10} {"final |g|_inf":>14}', flush=True)
    for v in VARIANTS:
        p = all_pilots[v['name']]
        e_dense = p['dense_E_final']
        b = max(e_dense) - max(e_dense[0], e_dense[-1])
        fm = max(p['dense_fmax_final'])
        gi = p['g_inf'][-1]
        print(f'{v["name"]:>22} {b:>14.4f} {fm:>10.3f} {gi:>14.4e}', flush=True)


if __name__ == '__main__':
    main()
