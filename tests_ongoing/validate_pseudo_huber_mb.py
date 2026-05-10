"""Multi-seed validation of pvre_pseudo_huber on Müller-Brown.

Apples-to-apples vs the huber_winner already in mb_validate/results.json:
swap `pvre_huber` -> `pvre_pseudo_huber`, hold every other knob constant.

  δ = 1.0
  lr = 1e-2
  (n_embed=2, depth=2)  — 18 trainable params
  threshold = 1e-2
  num_optimizer_iterations = 600
  patience = 10

Speed (per-call wall + eval-point count) is covered by
``tests_ongoing/eval_huber_speed.py``; this script measures end-to-end
optimization wall + path-quality (TS force).

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/validate_pseudo_huber_mb.py"

Output: appends `pseudo_huber_winner_s{0,1,2}` rows to
        /pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_validate/results.json
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_validate'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')
HUBER_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml')

CANDIDATES = [
    {'tag': 'pseudo_huber_winner', 'lr': 1.0e-2, 'n_embed': 2, 'depth': 2,
     'delta': 1.0, 'n_iter': 600, 'threshold': 1.0e-2, 'patience': 10},
]
SEEDS = [0, 1, 2]


def write_pseudo_cfg(base, c, dst):
    cfg = copy.deepcopy(base)
    cfg['initialization_params']['path_params']['n_embed'] = c['n_embed']
    cfg['initialization_params']['path_params']['depth'] = c['depth']
    cfg['initialization_params']['device'] = 'cuda'
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_names'] = 'pvre_pseudo_huber'
    leg['integrator_params']['path_integrand_kwargs'] = {
        'pvre_pseudo_huber': {'delta': c['delta']},
    }
    leg['optimizer_params']['optimizer']['lr'] = c['lr']
    leg['optimizer_params']['threshold'] = c['threshold']
    leg['optimizer_params']['patience'] = c['patience']
    leg['num_optimizer_iterations'] = c['n_iter']
    with open(dst, 'w') as f:
        yaml.dump(cfg, f)


def run_one(cfg_path, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, RUNNER,
           '--config', cfg_path,
           '--out', out_dir,
           '--seed', str(seed),
           '--monitor-every', '5']
    print(f'\n>>> {os.path.basename(out_dir)}', flush=True)
    rc = subprocess.run(cmd).returncode
    return rc == 0


def analyze(out_dir):
    tr = json.load(open(os.path.join(out_dir, 'trace.json')))
    qiter, fts, fperp, ginf, bar = [], [], [], [], []
    offset = 0
    for s in tr['stages']:
        for it, f_ts, fp, b in zip(s['q_iter'], s['f_inf_ts'],
                                    s['fperp_inf_ts'], s['barrier']):
            qiter.append(offset + it)
            fts.append(f_ts); fperp.append(fp); bar.append(b)
            ginf.append(s['ginf'][it])
        offset += s['n_iter']
    qiter = np.array(qiter); fts = np.array(fts); fperp = np.array(fperp)
    ginf = np.array(ginf); bar = np.array(bar)
    bidx = int(np.argmin(fts))
    return {
        'best_f_inf_ts': float(fts[bidx]),
        'best_iter': int(qiter[bidx]),
        'best_ginf': float(ginf[bidx]),
        'best_barrier': float(bar[bidx]),
        'best_fperp_inf_ts': float(fperp[bidx]),
        'final_f_inf_ts': float(fts[-1]),
        'final_fperp_inf_ts': float(fperp[-1]),
        'final_barrier': float(bar[-1]),
        'wall_s': float(sum(s['elapsed_s'] for s in tr['stages'])),
    }


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    results_path = os.path.join(OUT_BASE, 'results.json')
    if os.path.exists(results_path):
        results = json.load(open(results_path))
        # Drop any prior pseudo_huber rows so a re-run replaces them.
        results = [r for r in results
                   if not r.get('config', '').startswith('pseudo_huber')]
    else:
        results = []

    base = yaml.safe_load(open(HUBER_CFG))
    for c in CANDIDATES:
        for seed in SEEDS:
            tag = f'{c["tag"]}_s{seed}'
            out_dir = os.path.join(OUT_BASE, tag)
            cfg_path = os.path.join(out_dir, 'config.yaml')
            os.makedirs(out_dir, exist_ok=True)
            write_pseudo_cfg(base, c, cfg_path)
            if run_one(cfg_path, seed, out_dir):
                r = analyze(out_dir)
                r.update({'config': c['tag'], 'seed': seed,
                          'lr': c['lr'], 'n_embed': c['n_embed'],
                          'depth': c['depth'], 'delta': c['delta'],
                          'n_iter': c['n_iter']})
                results.append(r)
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

    # Aggregate across all configs in results.json (baseline, huber_winner,
    # pseudo_huber_winner). Quality metric = |F|_∞ @TS, best across the
    # trajectory; |F_⊥|_∞ @TS at the same iter alongside it.
    print(f'\n=== mb_validate aggregate (3 seeds, all configs) ===', flush=True)
    print(f'{"config":<22s} {"F_TS_mean":>11s} {"F_TS_std":>10s} '
          f'{"best_iter_mean":>15s} {"|g|@best_mean":>14s} '
          f'{"wall_mean":>10s} {"Fp_TS_mean":>11s}')
    by_cfg = {}
    for r in results:
        by_cfg.setdefault(r['config'], []).append(r)
    for cfg, rs in sorted(by_cfg.items(),
                          key=lambda kv: np.mean([r['best_f_inf_ts'] for r in kv[1]])):
        bts = np.array([r['best_f_inf_ts'] for r in rs])
        bis = np.array([r['best_iter'] for r in rs])
        bgs = np.array([r['best_ginf'] for r in rs])
        ws = np.array([r['wall_s'] for r in rs])
        bps = np.array([r['best_fperp_inf_ts'] for r in rs])
        print(f'{cfg:<22s} {bts.mean():>11.4e} {bts.std():>10.4e} '
              f'{bis.mean():>15.1f} {bgs.mean():>14.4e} '
              f'{ws.mean():>10.1f} {bps.mean():>11.4e}')


if __name__ == '__main__':
    main()
