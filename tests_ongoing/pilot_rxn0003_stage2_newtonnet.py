"""Phase 0 pilot for rxn0003 stage 2 with NewtonNet (parallel to LJ13 / UMA pilot).

Same recipe as pilot_rxn0003_stage2_lj13.py, only the potential differs:
  potential = NewtonNet (training_35 checkpoint)
  integrand = pvre
  rest verbatim: n4d2, lr=1e-3, rtol=1e-1, atol=1e-4, thr=0, patience=1,
  max_iter=200, M=51 dense probe every iter.

Stage 1 (geodesic warm-up) unchanged: shipped repel recipe.
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
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_stage2_pilot_newtonnet'
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
S2_MAX_ITER = 200

N_EMBED = 4
DEPTH = 2

M_DENSE = 51
CKPT_EVERY = 10


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


def setup_newtonnet(mep):
    return get_potential(images=mep.images, name='newtonnet',
                         model_path=NEWTONNET_MODEL,
                         device=mep.device, dtype=mep.dtype)


@torch.no_grad()
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


def write_ckpt(trace, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(trace, f, indent=2)
    os.replace(tmp, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)
    print(f'NewtonNet model = {NEWTONNET_MODEL}', flush=True)
    print(f'recipe stage 2: rtol={S2_RTOL:.1e} atol={S2_ATOL:.1e} '
          f'lr={S2_LR:.1e} thr={S2_THR} patience={S2_PATIENCE} '
          f'n_embed={N_EMBED} depth={DEPTH} max_iter={S2_MAX_ITER}', flush=True)
    print(f'seed = {SEED}', flush=True)

    print('=== Build mep + load NewtonNet ===', flush=True)
    t0 = time.perf_counter()
    mep, cfg = build_mep(seed=SEED)
    print(f'  mep built ({time.perf_counter()-t0:.1f}s)', flush=True)
    t0 = time.perf_counter()
    pot2 = setup_newtonnet(mep)
    print(f'  NewtonNet loaded ({time.perf_counter()-t0:.1f}s)', flush=True)

    print('\n=== Stage 1: geodesic warm-up (shipped) ===', flush=True)
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
    print(f'  stage 1 stopped iter={s1_stop} wall={s1_wall:.2f}s '
          f'|g|_inf={float(out.grad_norm.item()):.4e}', flush=True)

    print('\n=== Probe after stage 1 (initial state for stage 2) ===', flush=True)
    pr0 = probe(mep.path, pot2, M_DENSE)
    print(f'  barrier={pr0["barrier"]:.4f} eV  max_fmax={pr0["max_fmax"]:.3f} eV/A  '
          f'fmax_TS={pr0["fmax_per_t"][pr0["i_argmax_E"]]:.3f}  '
          f'i_argmax_E={pr0["i_argmax_E"]}/{M_DENSE-1}', flush=True)

    print('\n=== Stage 2: NewtonNet + pvre (LJ13 settings, thr=0) ===', flush=True)
    integ2, optr2 = setup_stage2(mep, pot2)
    trace = {
        'potential': 'newtonnet', 'model_path': NEWTONNET_MODEL,
        'recipe_stage1': {'rtol': S1_RTOL, 'atol': S1_ATOL, 'lr': S1_LR,
                          'thr': S1_THR, 'patience': S1_PATIENCE,
                          'max_iter': S1_MAX_ITER, 'stop_iter': s1_stop, 'wall_s': s1_wall},
        'recipe_stage2': {'rtol': S2_RTOL, 'atol': S2_ATOL, 'lr': S2_LR,
                          'thr': S2_THR, 'patience': S2_PATIENCE, 'max_iter': S2_MAX_ITER,
                          'n_embed': N_EMBED, 'depth': DEPTH},
        'seed': SEED, 'M_dense': M_DENSE,
        'iter': [], 'wall_s': [], 'g_inf': [], 'loss': [],
        'probe_iter': [0], 'probe_barrier': [pr0['barrier']],
        'probe_max_fmax': [pr0['max_fmax']], 'probe_i_argmax_E': [pr0['i_argmax_E']],
        'probe_E_per_t': [pr0['E_per_t']], 'probe_fmax_per_t': [pr0['fmax_per_t']],
        'probe_wall_s': [0.0],
    }
    ckpt_path = f'{OUT_DIR}/pilot_trace.json'
    write_ckpt(trace, ckpt_path)

    t_s2 = time.perf_counter()
    for it in range(S2_MAX_ITER):
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

        tp = time.perf_counter()
        pr = probe(mep.path, pot2, M_DENSE)
        torch.cuda.synchronize()
        w_probe = time.perf_counter() - tp
        trace['probe_iter'].append(it + 1)
        trace['probe_barrier'].append(pr['barrier'])
        trace['probe_max_fmax'].append(pr['max_fmax'])
        trace['probe_i_argmax_E'].append(pr['i_argmax_E'])
        trace['probe_E_per_t'].append(pr['E_per_t'])
        trace['probe_fmax_per_t'].append(pr['fmax_per_t'])
        trace['probe_wall_s'].append(w_probe)

        fmax_ts = pr['fmax_per_t'][pr['i_argmax_E']]
        print(f'  iter={it+1:4d}  loss={lv:.4e}  |g|_inf={gi:.4e}  '
              f'barrier={pr["barrier"]:.4f}  fmax_TS={fmax_ts:.3f}  '
              f'fmax_max={pr["max_fmax"]:.3f}  step={w_step:.2f}s  probe={w_probe:.2f}s', flush=True)

        if (it + 1) % CKPT_EVERY == 0:
            write_ckpt(trace, ckpt_path)

    s2_wall = time.perf_counter() - t_s2
    trace['stage2_total_wall_s'] = s2_wall
    write_ckpt(trace, ckpt_path)

    fmax_ts_traj = [trace['probe_fmax_per_t'][k][trace['probe_i_argmax_E'][k]]
                    for k in range(len(trace['probe_iter']))]
    print(f'\n=== Stage 2 done: {S2_MAX_ITER} iters in {s2_wall:.1f}s '
          f'({s2_wall/S2_MAX_ITER:.2f} s/iter avg) ===', flush=True)
    print(f'  final |g|_inf  = {trace["g_inf"][-1]:.4e}', flush=True)
    print(f'  final loss     = {trace["loss"][-1]:.4e}', flush=True)
    print(f'  final barrier  = {trace["probe_barrier"][-1]:.4f} eV', flush=True)
    print(f'  final fmax_TS  = {fmax_ts_traj[-1]:.4f} eV/A', flush=True)
    print(f'  final fmax_max = {trace["probe_max_fmax"][-1]:.4f} eV/A', flush=True)
    print(f'  min fmax_TS over trace = {min(fmax_ts_traj):.4f} @ iter '
          f'{trace["probe_iter"][fmax_ts_traj.index(min(fmax_ts_traj))]}', flush=True)
    print(f'  wrote {ckpt_path}', flush=True)


if __name__ == '__main__':
    main()
