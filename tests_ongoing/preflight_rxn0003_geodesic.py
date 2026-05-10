"""Pre-flight for rxn0003 stage-1 (geodesic on repel) with UMA-probe plateau monitor.

Two phases in one srun (amortizes the UMA cold-start):

  PHASE A — cost bench
    Time t_step (one geodesic optimization_step) and t_UMA(M) (one UMA
    forward on M evenly-spaced path positions) on the lj13-style stage-1
    settings. Compute K_rec = ceil(t_UMA / (0.1 * t_step)).

  PHASE B — pilot trace
    Rebuild a fresh Popcornn (so phase-A's perturbed path doesn't bias
    the trace), run 1000 iters with thr disabled, probe UMA every
    K_PILOT iters. Save per-iter (|g|_inf, loss, wall) and per-probe
    (max_E, mean_E, max_fmax, E-per-t) to JSON for picking ε / J.

Settings transferred from lj13.yaml:
  MLP n_embed=4, depth=2; lr=1e-3; rtol=1e-1, atol=1e-4; thr disabled.
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
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_rxn0003_pilot'

N_WARMUP = 5
N_BENCH_STEP = 20
N_BENCH_UMA = 10
M = 11
K_PILOT = 10
N_ITER = 1000

RTOL = 1.0e-1
ATOL = 1.0e-4
LR = 1.0e-3
N_EMBED = 4
DEPTH = 2


def build_mep(seed: int = 0):
    cfg = import_run_config(CONFIG)
    init = dict(cfg['initialization_params'])
    # yaml stores images as 'configs/rxn0003.xyz' (relative to repo root);
    # absolutize so srun's cwd-not-repo doesn't bite.
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
    if out.forces is not None:
        f = out.forces.detach()
        f_atom = f.view(f.shape[0], -1, 3)
        fmax_per_t = f_atom.norm(dim=-1).amax(dim=-1).cpu().tolist()
    else:
        fmax_per_t = [None] * len(energies)
    return energies, fmax_per_t


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'OUT_DIR = {OUT_DIR}', flush=True)

    print('=== Phase A: build mep + setup stage-1 ===', flush=True)
    mep, cfg = build_mep(seed=0)
    integ, optr = setup_stage1(mep)
    print(f'device={mep.device}, dtype={mep.dtype}', flush=True)
    print(f'n_atoms={mep.images.atomic_numbers.shape[0]}', flush=True)
    print(f'n_params(MLP)={sum(p.numel() for p in mep.path.parameters())}', flush=True)

    print('=== Loading UMA ===', flush=True)
    t0 = time.perf_counter()
    uma = setup_uma(mep, cfg)
    print(f'UMA loaded in {time.perf_counter()-t0:.1f}s', flush=True)

    print(f'=== Bench: optimization_step (warmup={N_WARMUP}, time {N_BENCH_STEP}) ===', flush=True)
    for _ in range(N_WARMUP):
        optr.optimization_step(mep.path, integ)
    torch.cuda.synchronize()
    step_times = []
    for _ in range(N_BENCH_STEP):
        ts = time.perf_counter()
        optr.optimization_step(mep.path, integ)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - ts)
    t_step = sum(step_times) / len(step_times)
    print(f't_step mean = {t_step:.4f}s   (min {min(step_times):.4f}, max {max(step_times):.4f})', flush=True)

    print(f'=== Bench: UMA probe M={M} (warmup={N_WARMUP}, time {N_BENCH_UMA}) ===', flush=True)
    for _ in range(N_WARMUP):
        probe_uma(mep.path, uma, M)
    torch.cuda.synchronize()
    uma_times = []
    for _ in range(N_BENCH_UMA):
        ts = time.perf_counter()
        probe_uma(mep.path, uma, M)
        torch.cuda.synchronize()
        uma_times.append(time.perf_counter() - ts)
    t_uma = sum(uma_times) / len(uma_times)
    print(f't_UMA mean = {t_uma:.4f}s   (min {min(uma_times):.4f}, max {max(uma_times):.4f})', flush=True)

    K_rec = max(1, int((t_uma / (0.1 * t_step)) + 0.999))
    overhead_at_K = t_uma / (K_rec * t_step) * 100.0
    print(f'\nK_rec = ceil(t_UMA / (0.1*t_step)) = {K_rec}   '
          f'(overhead {overhead_at_K:.1f}%)', flush=True)

    bench = {
        't_step_s': step_times,
        't_uma_s': uma_times,
        't_step_mean_s': t_step,
        't_uma_mean_s': t_uma,
        'M': M,
        'K_rec': K_rec,
    }
    with open(f'{OUT_DIR}/bench.json', 'w') as f:
        json.dump(bench, f, indent=2)
    print(f'wrote {OUT_DIR}/bench.json\n', flush=True)

    print(f'=== Phase B: fresh mep, pilot trace {N_ITER} iters, probe every {K_PILOT} ===', flush=True)
    del mep, integ, optr, uma
    torch.cuda.empty_cache()

    mep, _ = build_mep(seed=0)
    integ, optr = setup_stage1(mep)
    uma = setup_uma(mep, cfg)

    pilot = {
        'iter': [], 'wall_s': [], 'g_inf': [], 'loss': [],
        'probe_iter': [], 'probe_max_E': [], 'probe_mean_E': [],
        'probe_E_per_t': [], 'probe_fmax_per_t': [], 'probe_max_fmax': [],
        'probe_wall_s': [],
    }

    print('probe iter=0 (initial path)', flush=True)
    tp = time.perf_counter()
    e, fm = probe_uma(mep.path, uma, M)
    torch.cuda.synchronize()
    pilot['probe_iter'].append(0)
    pilot['probe_max_E'].append(max(e))
    pilot['probe_mean_E'].append(sum(e) / len(e))
    pilot['probe_E_per_t'].append(e)
    pilot['probe_fmax_per_t'].append(fm)
    pilot['probe_max_fmax'].append(max(x for x in fm if x is not None) if any(x is not None for x in fm) else None)
    pilot['probe_wall_s'].append(time.perf_counter() - tp)
    print(f'  iter=0  max_E={max(e):.4e}  mean_E={sum(e)/len(e):.4e}  max_fmax={pilot["probe_max_fmax"][-1]}', flush=True)

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

        if (it + 1) % K_PILOT == 0:
            tp = time.perf_counter()
            e, fm = probe_uma(mep.path, uma, M)
            torch.cuda.synchronize()
            wp = time.perf_counter() - tp
            pilot['probe_iter'].append(it + 1)
            pilot['probe_max_E'].append(max(e))
            pilot['probe_mean_E'].append(sum(e) / len(e))
            pilot['probe_E_per_t'].append(e)
            pilot['probe_fmax_per_t'].append(fm)
            pilot['probe_max_fmax'].append(
                max(x for x in fm if x is not None) if any(x is not None for x in fm) else None
            )
            pilot['probe_wall_s'].append(wp)

        if (it + 1) % 50 == 0 or it < 5:
            last_max = pilot['probe_max_E'][-1]
            last_fmax = pilot['probe_max_fmax'][-1]
            print(f'iter={it+1:4d}  |g|_inf={gi:.4e}  loss={lv}  '
                  f'last_max_E={last_max:.4e}  last_max_fmax={last_fmax}', flush=True)

    elapsed = time.perf_counter() - t_run
    print(f'pilot total wall={elapsed:.1f}s  ({elapsed/N_ITER*1000:.1f} ms/iter)', flush=True)

    with open(f'{OUT_DIR}/pilot_trace.json', 'w') as f:
        json.dump(pilot, f, indent=2)
    print(f'wrote {OUT_DIR}/pilot_trace.json', flush=True)


if __name__ == '__main__':
    main()
