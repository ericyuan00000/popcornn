"""Re-run rxn0003 stage-2 n4d2 NewtonNet with M=1001 dense probe.

Tests the user's hypothesis that M=51 undersampling causes the apparent
fmax_TS rebound — the true saddle's t-coordinate may move within a single
[t_i, t_{i+1}] interval (resolution 0.02 at M=51), so the reported
fmax_per_t[i_TS] doesn't track the actual peak. M=1001 gives t-resolution
5e-4 — fine enough to follow the saddle continuously.

Only n4d2 (best from capacity sweep), single seed, same shipped LJ13
stage-2 recipe (thr=0, max_iter=300). Per-iter probe at M=1001.
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
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_stage2_n4d2_M1001_newtonnet'
NEWTONNET_MODEL = '/pscratch/sd/e/ericyuan/20240322_Geodesics/20241025_GG/newtonnet/training_35/models/best_model.pt'

SEED = 0

S1_RTOL, S1_ATOL, S1_LR, S1_THR, S1_PATIENCE, S1_MAX_ITER = 1.0e-1, 1.0e-4, 1.0e-3, 1.0e-3, 1, 1000
S2_RTOL, S2_ATOL, S2_LR, S2_THR, S2_PATIENCE, S2_MAX_ITER = 1.0e-1, 1.0e-4, 1.0e-3, 0.0, 1, 300

N_EMBED, DEPTH = 4, 2
M_DENSE = 1001
CKPT_EVERY = 20


def build_mep(seed):
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
    integ = PathIntegrator(path_integrand_names='geodesic', rtol=S1_RTOL, atol=S1_ATOL,
                           device=mep.device, dtype=mep.dtype)
    integ.save_samples = False
    optr = PathOptimizer(path=mep.path, optimizer={'name': 'adam', 'lr': S1_LR},
                         threshold=S1_THR, patience=S1_PATIENCE, find_ts=False,
                         device=mep.device, dtype=mep.dtype)
    return integ, optr


def setup_stage2(mep, pot):
    mep.path.set_potential(pot)
    integ = PathIntegrator(path_integrand_names='pvre', rtol=S2_RTOL, atol=S2_ATOL,
                           track_loss=True, device=mep.device, dtype=mep.dtype)
    integ.save_samples = False
    optr = PathOptimizer(path=mep.path, optimizer={'name': 'adam', 'lr': S2_LR},
                         threshold=S2_THR, patience=S2_PATIENCE, find_ts=False,
                         device=mep.device, dtype=mep.dtype)
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
    i_TS = energies.index(E_max)
    return {
        'barrier': E_max - max(E_R, E_P),
        'i_TS': i_TS,
        'fmax_TS': fmax_per_t[i_TS],
        'max_fmax': max(fmax_per_t),
    }


def write_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)
    print(f'M_DENSE = {M_DENSE}  (t resolution = {1.0/(M_DENSE-1):.2e})', flush=True)

    t0 = time.perf_counter()
    mep, cfg = build_mep(seed=SEED)
    pot = get_potential(images=mep.images, name='newtonnet', model_path=NEWTONNET_MODEL,
                        device=mep.device, dtype=mep.dtype)
    print(f'NewtonNet loaded ({time.perf_counter()-t0:.1f}s)', flush=True)

    integ1, optr1 = setup_stage1(mep)
    t_s1 = time.perf_counter()
    for it in range(S1_MAX_ITER):
        out = optr1.optimization_step(mep.path, integ1)
        if optr1.converged:
            s1_stop = it + 1
            break
    else:
        s1_stop = S1_MAX_ITER
    torch.cuda.synchronize()
    print(f'stage 1 stop iter={s1_stop} wall={time.perf_counter()-t_s1:.2f}s', flush=True)

    pr0 = probe(mep.path, pot, M_DENSE)
    print(f'post-S1 probe: barrier={pr0["barrier"]:.4f}  fmax_TS={pr0["fmax_TS"]:.4f}  '
          f'i_TS={pr0["i_TS"]}/{M_DENSE-1}', flush=True)

    integ2, optr2 = setup_stage2(mep, pot)
    trace = {
        'tag': f'n{N_EMBED}d{DEPTH}_M{M_DENSE}',
        'seed': SEED, 'M_dense': M_DENSE,
        'recipe_stage2': {'rtol': S2_RTOL, 'atol': S2_ATOL, 'lr': S2_LR,
                          'thr': S2_THR, 'patience': S2_PATIENCE, 'max_iter': S2_MAX_ITER,
                          'n_embed': N_EMBED, 'depth': DEPTH},
        'iter': [], 'wall_s': [], 'probe_wall_s': [], 'g_inf': [], 'loss': [],
        'probe_iter': [0], 'probe_barrier': [pr0['barrier']],
        'probe_i_TS': [pr0['i_TS']], 'probe_fmax_TS': [pr0['fmax_TS']],
        'probe_max_fmax': [pr0['max_fmax']],
    }
    out_path = f'{OUT_DIR}/trace.json'

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
        pr = probe(mep.path, pot, M_DENSE)
        torch.cuda.synchronize()
        w_probe = time.perf_counter() - tp
        trace['probe_iter'].append(it + 1)
        trace['probe_barrier'].append(pr['barrier'])
        trace['probe_i_TS'].append(pr['i_TS'])
        trace['probe_fmax_TS'].append(pr['fmax_TS'])
        trace['probe_max_fmax'].append(pr['max_fmax'])
        trace['probe_wall_s'].append(w_probe)

        if (it + 1) % 25 == 0 or it < 3:
            print(f'  iter={it+1:4d}  loss={lv:.4e}  |g|_inf={gi:.4e}  '
                  f'i_TS={pr["i_TS"]:4d}  barrier={pr["barrier"]:.4f}  '
                  f'fmax_TS={pr["fmax_TS"]:.4f}  step={w_step:.2f}s  probe={w_probe:.2f}s', flush=True)
        if (it + 1) % CKPT_EVERY == 0:
            write_json(trace, out_path)

    trace['stage2_total_wall_s'] = time.perf_counter() - t_s2
    write_json(trace, out_path)

    min_i = trace['probe_fmax_TS'].index(min(trace['probe_fmax_TS']))
    print(f'\ndone {S2_MAX_ITER} iters in {trace["stage2_total_wall_s"]:.1f}s '
          f'({trace["stage2_total_wall_s"]/S2_MAX_ITER:.2f} s/iter avg)', flush=True)
    print(f'min fmax_TS = {trace["probe_fmax_TS"][min_i]:.4f} @ iter '
          f'{trace["probe_iter"][min_i]} (i_TS={trace["probe_i_TS"][min_i]})', flush=True)
    print(f'final fmax_TS = {trace["probe_fmax_TS"][-1]:.4f}', flush=True)
    print(f'final barrier = {trace["probe_barrier"][-1]:.4f}', flush=True)
    print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
