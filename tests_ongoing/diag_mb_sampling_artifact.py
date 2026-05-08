"""Per-iter F_TS at four resolutions to disentangle sampling vs real
geometry drift on MB-small-MLP runs.

Resolutions: g101 / g1001 / g10001 (uniform t-grids) + parab
(parabolic refine around the g10001 argmax).

Configurable via env:
  INTEGRAND  pvre | pvre_pseudo_huber          (default: pvre_pseudo_huber)
  DELTA      δ for pvre_pseudo_huber           (default: 1.0e+2; ignored otherwise)
  LR         optimizer lr                      (default: 1.0e-3)
  N_EMBED    MLP n_embed                       (default: 2)
  DEPTH      MLP depth                         (default: 2)
  RTOL       integrator rtol                   (default: 1.0e-2)
  SEED                                          (default: 0)
  MAX_ITER                                      (default: 1000)
  DIAG_OUT   override output dir               (default: derived from above)

Other knobs are pinned (threshold=0, patience=10, activation=gelu).

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclude=nid001208 --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   INTEGRAND=pvre python tests_ongoing/diag_mb_sampling_artifact.py"

Output: $DIAG_OUT/trace.json
"""
import copy
import json
import math
import os
import time as time_mod

import numpy as np
import torch
import yaml

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml')

INTEGRAND = os.environ.get('INTEGRAND', 'pvre_pseudo_huber')
LR = float(os.environ.get('LR', '1.0e-3'))
DELTA = float(os.environ.get('DELTA', '1.0e+2')) if INTEGRAND == 'pvre_pseudo_huber' else None
N_EMBED = int(os.environ.get('N_EMBED', '2'))
DEPTH = int(os.environ.get('DEPTH', '2'))
RTOL = float(os.environ.get('RTOL', '1.0e-2'))
SEED = int(os.environ.get('SEED', '0'))
MAX_ITER = int(os.environ.get('MAX_ITER', '1000'))
GRID_SIZES = (101, 1001, 10001)


def _fmt_pow10(x):
    return f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')


_TAG = f'pseudo_d{_fmt_pow10(DELTA)}' if INTEGRAND == 'pvre_pseudo_huber' else INTEGRAND
OUT = os.environ.get(
    'DIAG_OUT',
    f'/pscratch/sd/e/ericyuan/temp/popcornn_sampling_diag/'
    f'lr{_fmt_pow10(LR)}_{_TAG}_n{N_EMBED}d{DEPTH}_rtol{_fmt_pow10(RTOL)}_s{SEED}',
)


def quality_multi(mep, grids):
    """F_TS at each grid + parabolic refinement around the densest argmax."""
    out = {}
    cache = {}
    for label, tg in grids.items():
        po = mep.path(tg, return_velocities=False,
                      return_energies=True, return_forces=True)
        e = po.energies.detach().cpu().numpy().reshape(-1)
        f = po.forces.detach().cpu().numpy()
        if f.ndim == 3:
            f = f.reshape(f.shape[0], -1)
        cache[label] = (e, f)
        ts = int(e.argmax())
        out[label] = {
            'barrier': float(e.max() - e[0]),
            'ts_t': float(tg[ts].item()),
            'ts_idx': ts,
            'f_inf_ts': float(np.max(np.abs(f[ts]))),
        }

    dense_label = max(grids.keys(), key=lambda k: grids[k].numel())
    dense_tg = grids[dense_label]
    e, _ = cache[dense_label]
    ts = out[dense_label]['ts_idx']
    n = dense_tg.numel()
    parab = {'ts_t': None, 'f_inf_ts': None, 'e_at_ts': None}
    if 0 < ts < n - 1:
        t0_, t1_, t2_ = (float(dense_tg[ts - 1]), float(dense_tg[ts]), float(dense_tg[ts + 1]))
        e0, e1, e2 = float(e[ts - 1]), float(e[ts]), float(e[ts + 1])
        denom = e2 - 2.0 * e1 + e0
        if denom < 0.0:  # concave-down → valid maximum
            h = t1_ - t0_
            t_star = t1_ - 0.5 * h * (e2 - e0) / denom
            t_star = max(min(t_star, t2_), t0_)
            t_eval = torch.tensor([t_star], device=dense_tg.device, dtype=dense_tg.dtype)
            po = mep.path(t_eval, return_velocities=False,
                          return_energies=True, return_forces=True)
            f_star = po.forces.detach().cpu().numpy().reshape(-1)
            parab = {
                'ts_t': float(t_star),
                'f_inf_ts': float(np.max(np.abs(f_star))),
                'e_at_ts': float(po.energies.detach().cpu().item()),
            }
    out['parab'] = parab
    return out


def build_cfg(base):
    cfg = copy.deepcopy(base)
    cfg['initialization_params']['path_params']['n_embed'] = N_EMBED
    cfg['initialization_params']['path_params']['depth'] = DEPTH
    cfg['initialization_params']['device'] = 'cuda'
    cfg['initialization_params']['seed'] = SEED
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_names'] = INTEGRAND
    if DELTA is not None:
        leg['integrator_params']['path_integrand_kwargs'] = {
            INTEGRAND: {'delta': DELTA},
        }
    else:
        leg['integrator_params'].pop('path_integrand_kwargs', None)
    leg['integrator_params']['rtol'] = RTOL
    leg['integrator_params']['track_loss'] = True
    leg['optimizer_params']['optimizer'] = {'name': 'adam', 'lr': LR}
    leg['optimizer_params']['threshold'] = 0.0
    leg['optimizer_params']['patience'] = 10
    leg['num_optimizer_iterations'] = MAX_ITER
    return cfg


def main():
    os.makedirs(OUT, exist_ok=True)
    base = yaml.safe_load(open(BASE_CFG))
    cfg = build_cfg(base)
    with open(os.path.join(OUT, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    init_params = cfg['initialization_params']
    init_params.pop('output_dir', None)
    mep = Popcornn(**init_params)
    leg = cfg['optimization_params'][0]
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg['integrator_params'],
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    t_init = mep.path.t_init.item()
    t_final = mep.path.t_final.item()
    grids = {
        f'g{n}': torch.linspace(t_init, t_final, n,
                                device=mep.device, dtype=mep.dtype)
        for n in GRID_SIZES
    }
    grid_labels = list(grids.keys()) + ['parab']

    print(f'integrand={INTEGRAND}  lr={LR}  delta={DELTA}  '
          f'n_embed={N_EMBED}  depth={DEPTH}  rtol={RTOL}  '
          f'seed={SEED}  max_iter={MAX_ITER}', flush=True)
    print(f'grids={list(grids.keys())} + parab; out={OUT}', flush=True)
    cols = f'{"iter":>6s} {"loss":>11s} {"|g|_inf":>11s}'
    for lab in grid_labels:
        cols += f' {"F_TS@" + lab:>11s}'
    print(cols, flush=True)

    losses, ginfs, walls = [], [], []
    barrier_g101 = []
    f_ts = {lab: [] for lab in grid_labels}
    ts_t = {lab: [] for lab in grid_labels}

    t0 = time_mod.perf_counter()
    last_print = -1
    for step in range(MAX_ITER):
        s0 = time_mod.perf_counter()
        out = optr.optimization_step(mep.path, integ)
        flat = out.grad_integral.detach()
        loss = float(out.loss[0].item()) if getattr(out, 'loss', None) is not None else None
        ginf = float(flat.abs().max().item())
        q = quality_multi(mep, grids)
        s1 = time_mod.perf_counter()

        losses.append(loss)
        ginfs.append(ginf)
        walls.append(s1 - s0)
        barrier_g101.append(q['g101']['barrier'])
        for lab in grid_labels:
            f_ts[lab].append(q[lab]['f_inf_ts'])
            ts_t[lab].append(q[lab]['ts_t'])

        if step in (0, 5, 25, 100, 200) or (step + 1) % 50 == 0 or step == MAX_ITER - 1:
            row = f'{step:>6d} ' + (f'{loss:11.4e}' if loss is not None else f'{"--":>11s}')
            row += f' {ginf:>11.4e}'
            for lab in grid_labels:
                v = q[lab]['f_inf_ts']
                row += f' {(v if v is not None else float("nan")):>11.3e}'
            print(row, flush=True)
            last_print = step

        if not math.isfinite(ginf) or (loss is not None and not math.isfinite(loss)):
            print(f'  → non-finite at step {step}; aborting', flush=True)
            break

    elapsed = time_mod.perf_counter() - t0
    print(f'  total elapsed: {elapsed:.1f}s ({len(walls)} iters, '
          f'{1000 * elapsed / max(1, len(walls)):.1f} ms/iter)', flush=True)

    print('\n=== summary ===', flush=True)
    for lab in grid_labels:
        arr = [v for v in f_ts[lab] if v is not None]
        if not arr:
            print(f'{lab}: all None', flush=True)
            continue
        a = np.asarray(arr)
        idx = int(a.argmin())
        print(f'{lab:>6s}: argmin iter={idx:>4d}  F_TS_min={a[idx]:.3e}  '
              f'F_TS_final={f_ts[lab][-1]:.3e}', flush=True)

    trace = {
        'integrand': INTEGRAND,
        'lr': LR, 'delta': DELTA, 'n_embed': N_EMBED, 'depth': DEPTH,
        'rtol': RTOL,
        'seed': SEED, 'max_iter': MAX_ITER, 'n_iter': len(walls),
        'elapsed_s': elapsed,
        'grid_sizes': list(GRID_SIZES),
        'loss': losses, 'ginf': ginfs,
        'barrier_g101': barrier_g101,
        'f_inf_ts': f_ts, 'ts_t': ts_t,
        'wall_per_step': walls,
    }
    with open(os.path.join(OUT, 'trace.json'), 'w') as f:
        json.dump(trace, f)
    print(f'trace: {os.path.join(OUT, "trace.json")}', flush=True)


if __name__ == '__main__':
    main()
