"""Capacity sweep for rxn0003 stage 2 with NewtonNet (Phase 1A+1B unified).

Tests MLP capacity for the path: n4d2 (baseline, exhibits rebound) vs n4d4
vs n8d4 vs n16d4. Same shipped LJ13 stage-2 recipe otherwise. Single seed.

NewtonNet model is loaded once and reused across configs (cold load ~50s).
Stage 1 (geodesic warm-up) re-runs per config since the path's MLP shape
differs. Stage 2 captures per-iter trace + per-iter dense probe.

Goal: identify the smallest capacity that eliminates the n4d2 rebound on
rxn0003 + NewtonNet, mirroring the MB finding that n8d4 = production ref
for pvre.
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
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_stage2_capsweep_newtonnet'
NEWTONNET_MODEL = '/pscratch/sd/e/ericyuan/20240322_Geodesics/20241025_GG/newtonnet/training_35/models/best_model.pt'

SEED = 0

S1_RTOL = 1.0e-1
S1_ATOL = 1.0e-4
S1_LR = 1.0e-3
S1_THR = 1.0e-3
S1_PATIENCE = 1
S1_MAX_ITER = 1000

S2_RTOL = 1.0e-1
S2_ATOL = 1.0e-4
S2_LR = 1.0e-3
S2_THR = 0.0
S2_PATIENCE = 1

# (n_embed, depth, stage2_max_iter)
CAPACITIES = [
    (4, 2, 300),
    (4, 4, 300),
    (8, 4, 300),
    (16, 4, 200),
]

M_DENSE = 51
CKPT_EVERY = 20


def build_mep(seed: int, n_embed: int, depth: int):
    cfg = import_run_config(CONFIG)
    init = dict(cfg['initialization_params'])
    if isinstance(init.get('images'), str) and not os.path.isabs(init['images']):
        init['images'] = os.path.join(REPO, 'examples', init['images'])
    init['path_params'] = {'name': 'mlp', 'n_embed': n_embed, 'depth': depth, 'activation': 'gelu'}
    init['device'] = 'cuda'
    init['seed'] = seed
    return Popcornn(**init), cfg


def setup_stage1(mep):
    pot = get_potential(images=mep.images, name='repel',
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(
        path_integrand_names='geodesic',
        rtol=S1_RTOL, atol=S1_ATOL,
        device=mep.device, dtype=mep.dtype,
    )
    integ.save_samples = False
    optr = PathOptimizer(
        path=mep.path,
        optimizer={'name': 'adam', 'lr': S1_LR},
        threshold=S1_THR, patience=S1_PATIENCE, find_ts=False,
        device=mep.device, dtype=mep.dtype,
    )
    return integ, optr


def setup_stage2(mep, pot):
    mep.path.set_potential(pot)
    integ = PathIntegrator(
        path_integrand_names='pvre',
        rtol=S2_RTOL, atol=S2_ATOL,
        track_loss=True,
        device=mep.device, dtype=mep.dtype,
    )
    integ.save_samples = False
    optr = PathOptimizer(
        path=mep.path,
        optimizer={'name': 'adam', 'lr': S2_LR},
        threshold=S2_THR, patience=S2_PATIENCE, find_ts=False,
        device=mep.device, dtype=mep.dtype,
    )
    return integ, optr


def probe(path, pot, M):
    device = path.initial_position.device
    dtype = path.initial_position.dtype
    times = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype).unsqueeze(-1)
    positions = path.get_positions(times)
    out = pot(positions)
    energies = out.energies.detach().cpu().squeeze(-1).tolist()
    f = out.forces.detach()
    f_atom = f.view(f.shape[0], -1, 3)
    fmax_per_t = f_atom.norm(dim=-1).amax(dim=-1).cpu().tolist()
    E_R, E_P = energies[0], energies[-1]
    E_max = max(energies)
    barrier = E_max - max(E_R, E_P)
    return {
        'barrier': barrier,
        'max_fmax': max(fmax_per_t),
        'i_argmax_E': energies.index(E_max),
        'E_per_t': energies,
        'fmax_per_t': fmax_per_t,
    }


def write_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_config(seed, n_embed, depth, max_iter, pot_loader):
    tag = f'n{n_embed}d{depth}'
    out_path = f'{OUT_DIR}/trace_{tag}.json'
    if os.path.exists(out_path):
        with open(out_path) as f:
            cached = json.load(f)
        print(f'\n=== {tag} (skipped, already done) ===', flush=True)
        return cached

    print(f'\n=== {tag}: build mep + load NewtonNet (or reuse) ===', flush=True)
    mep, cfg = build_mep(seed=seed, n_embed=n_embed, depth=depth)
    pot2 = pot_loader(mep)
    print(f'  n_params (MLP) = {sum(p.numel() for p in mep.path.parameters())}', flush=True)

    print(f'  --- Stage 1: geodesic warm-up ---', flush=True)
    integ1, optr1 = setup_stage1(mep)
    t_s1 = time.perf_counter()
    s1_stop = None
    for it in range(S1_MAX_ITER):
        out = optr1.optimization_step(mep.path, integ1)
        if optr1.converged:
            s1_stop = it + 1
            break
    torch.cuda.synchronize()
    s1_wall = time.perf_counter() - t_s1
    if s1_stop is None:
        s1_stop = S1_MAX_ITER
    print(f'  stage 1 stop iter={s1_stop} wall={s1_wall:.2f}s '
          f'|g|_inf={float(out.grad_norm.item()):.4e}', flush=True)

    pr0 = probe(mep.path, pot2, M_DENSE)
    fmax_ts0 = pr0['fmax_per_t'][pr0['i_argmax_E']]
    print(f'  post-S1 probe: barrier={pr0["barrier"]:.4f}  fmax_TS={fmax_ts0:.3f}', flush=True)

    print(f'  --- Stage 2: NewtonNet + pvre (max_iter={max_iter}) ---', flush=True)
    integ2, optr2 = setup_stage2(mep, pot2)
    trace = {
        'tag': tag, 'n_embed': n_embed, 'depth': depth,
        'recipe_stage1': {'rtol': S1_RTOL, 'atol': S1_ATOL, 'lr': S1_LR,
                          'thr': S1_THR, 'patience': S1_PATIENCE,
                          'stop_iter': s1_stop, 'wall_s': s1_wall},
        'recipe_stage2': {'rtol': S2_RTOL, 'atol': S2_ATOL, 'lr': S2_LR,
                          'thr': S2_THR, 'patience': S2_PATIENCE, 'max_iter': max_iter},
        'seed': seed, 'M_dense': M_DENSE,
        'iter': [], 'wall_s': [], 'g_inf': [], 'loss': [],
        'probe_iter': [0], 'probe_barrier': [pr0['barrier']],
        'probe_max_fmax': [pr0['max_fmax']], 'probe_i_argmax_E': [pr0['i_argmax_E']],
        'probe_fmax_TS': [fmax_ts0],
        'probe_E_per_t': [pr0['E_per_t']], 'probe_fmax_per_t': [pr0['fmax_per_t']],
    }

    t_s2 = time.perf_counter()
    for it in range(max_iter):
        ts = time.perf_counter()
        out = optr2.optimization_step(mep.path, integ2)
        torch.cuda.synchronize()
        w_step = time.perf_counter() - ts
        gi = float(out.grad_norm.item())
        try:
            lv = float(out.loss[0].item())
        except Exception:
            lv = None
        trace['iter'].append(it + 1)
        trace['wall_s'].append(w_step)
        trace['g_inf'].append(gi)
        trace['loss'].append(lv)

        pr = probe(mep.path, pot2, M_DENSE)
        fmax_ts = pr['fmax_per_t'][pr['i_argmax_E']]
        trace['probe_iter'].append(it + 1)
        trace['probe_barrier'].append(pr['barrier'])
        trace['probe_max_fmax'].append(pr['max_fmax'])
        trace['probe_i_argmax_E'].append(pr['i_argmax_E'])
        trace['probe_fmax_TS'].append(fmax_ts)
        trace['probe_E_per_t'].append(pr['E_per_t'])
        trace['probe_fmax_per_t'].append(pr['fmax_per_t'])

        if (it + 1) % 25 == 0 or it < 3:
            print(f'  iter={it+1:4d}  loss={lv:.4e}  |g|_inf={gi:.4e}  '
                  f'barrier={pr["barrier"]:.4f}  fmax_TS={fmax_ts:.3f}  '
                  f'fmax_max={pr["max_fmax"]:.3f}  step={w_step:.2f}s', flush=True)
        if (it + 1) % CKPT_EVERY == 0:
            write_json(trace, out_path)

    trace['stage2_total_wall_s'] = time.perf_counter() - t_s2
    write_json(trace, out_path)

    min_idx = trace['probe_fmax_TS'].index(min(trace['probe_fmax_TS']))
    print(f'  done {max_iter} iters in {trace["stage2_total_wall_s"]:.1f}s '
          f'({trace["stage2_total_wall_s"]/max_iter:.2f} s/iter avg)', flush=True)
    print(f'  min fmax_TS = {trace["probe_fmax_TS"][min_idx]:.4f} @ iter '
          f'{trace["probe_iter"][min_idx]}', flush=True)
    print(f'  final fmax_TS = {trace["probe_fmax_TS"][-1]:.4f}', flush=True)
    print(f'  final barrier = {trace["probe_barrier"][-1]:.4f}', flush=True)
    return trace


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)
    print(f'seed = {SEED}', flush=True)

    # Load NewtonNet once, reuse across configs (per-config rebuild of mep
    # would otherwise reload the model 4× at ~50s each).
    cached_pot = {'pot': None}
    def pot_loader(mep):
        if cached_pot['pot'] is None:
            t0 = time.perf_counter()
            cached_pot['pot'] = get_potential(
                images=mep.images, name='newtonnet',
                model_path=NEWTONNET_MODEL,
                device=mep.device, dtype=mep.dtype,
            )
            print(f'  NewtonNet loaded ({time.perf_counter()-t0:.1f}s)', flush=True)
        else:
            # Re-bind chemistry metadata (atomic_numbers etc. unchanged across configs
            # since same xyz, but the BasePotential attrs reference the new mep.images)
            cached_pot['pot'].atomic_numbers = mep.images.atomic_numbers
            cached_pot['pot'].n_atoms = mep.images.n_atoms if hasattr(mep.images, 'n_atoms') else len(mep.images.atomic_numbers)
        return cached_pot['pot']

    summary = []
    for n_embed, depth, max_iter in CAPACITIES:
        tr = run_config(SEED, n_embed, depth, max_iter, pot_loader)
        summary.append({
            'tag': tr['tag'],
            'n_embed': tr['n_embed'],
            'depth': tr['depth'],
            'min_fmax_TS': min(tr['probe_fmax_TS']),
            'min_fmax_TS_iter': tr['probe_iter'][tr['probe_fmax_TS'].index(min(tr['probe_fmax_TS']))],
            'final_fmax_TS': tr['probe_fmax_TS'][-1],
            'final_barrier': tr['probe_barrier'][-1],
            'stage2_wall_s': tr.get('stage2_total_wall_s'),
            'stage2_max_iter': tr['recipe_stage2']['max_iter'],
        })

    print('\n=== SUMMARY ===', flush=True)
    hdr = f'{"tag":>8} {"params":>8} {"min_fmax":>10} {"min_at":>8} {"final_fmax":>10} {"barrier":>10} {"wall_s":>8}'
    print(hdr, flush=True)
    for s in summary:
        print(f'{s["tag"]:>8} {"":>8} {s["min_fmax_TS"]:>10.4f} {s["min_fmax_TS_iter"]:>8d} '
              f'{s["final_fmax_TS"]:>10.4f} {s["final_barrier"]:>10.4f} {s["stage2_wall_s"]:>8.1f}', flush=True)

    write_json(summary, f'{OUT_DIR}/summary.json')
    print(f'wrote {OUT_DIR}/summary.json', flush=True)


if __name__ == '__main__':
    main()
