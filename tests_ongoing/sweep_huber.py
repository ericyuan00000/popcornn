"""δ-sweep for pvre_huber, head-to-head against the two-stage baseline.

For each system in {muller_brown, lj13}:
- Run the shipped two-stage config once as baseline
  (`pvre_squared → pvre`, hand-tuned per system).
- Run the single-stage `pvre_huber` config across δ ∈ {1e-2, 1e-1,
  1e+0, 1e+1, 1e+2} (round powers of 10, per the user's preference).

Per run we record: total iters, wall time, and the path-intrinsic
quality metrics from the per-iter trace — barrier, |F|_∞ at TS, and
|F_⊥|_∞ at TS. The last is the headline MEP-quality metric: the
two-stage schedule beats single-stage `pvre_squared` on LJ-13 by
driving `fperp_inf_ts` from ~0.22 to ~1.6e-3.

Each individual config is driven by `run_lj13_traced.py` (system-
agnostic — takes a YAML and an output dir), so this script is just a
sweep harness that subprocesses out and aggregates results.json.

Usage (Müller-Brown, login node):
    python tests_ongoing/sweep_huber.py --systems muller_brown --seeds 0

Usage (LJ-13, GPU interactive):
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/sweep_huber.py --systems lj13 --seeds 0"

Output: /pscratch/sd/e/ericyuan/temp/popcornn_huber/{system}/results.json
        plus per-config subdirs with each run's trace.json + xyz.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_huber'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')

DELTAS = [1.0e-2, 1.0e-1, 1.0e+0, 1.0e+1, 1.0e+2]

SYSTEMS = {
    'muller_brown': {
        'baseline_cfg': os.path.join(REPO_ROOT, 'examples/configs/muller_brown.yaml'),
        'huber_cfg': os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml'),
        'monitor_every': 5,
    },
    'lj13': {
        'baseline_cfg': os.path.join(REPO_ROOT, 'examples/configs/lj13.yaml'),
        'huber_cfg': os.path.join(REPO_ROOT, 'examples/configs/lj13_huber.yaml'),
        'monitor_every': 5,
    },
}


def write_huber_with_delta(base_cfg, delta, dst):
    """Clone the single-stage Huber config and override delta in place."""
    cfg = copy.deepcopy(base_cfg)
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_kwargs'] = {
        'pvre_huber': {'delta': float(delta)},
    }
    with open(dst, 'w') as f:
        yaml.dump(cfg, f)


def run_one(cfg_path, seed, out_dir, monitor_every):
    """Subprocess into run_lj13_traced.py for one (config, seed)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, RUNNER,
           '--config', cfg_path,
           '--out', out_dir,
           '--seed', str(seed),
           '--monitor-every', str(monitor_every)]
    print(f'\n>>> {" ".join(cmd)}', flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f'  FAILED rc={rc}', flush=True)
        return None
    trace = json.load(open(os.path.join(out_dir, 'trace.json')))
    return summarize(trace)


def summarize(trace):
    """Pull total iters, wall, and final path quality from a trace.json."""
    stages = trace['stages']
    total_iter = sum(
        s['converged_at'] + 1 if s['converged_at'] is not None else s['n_iter']
        for s in stages
    )
    wall_s = sum(s['elapsed_s'] for s in stages)
    last = stages[-1]
    return {
        'total_iter': int(total_iter),
        'wall_s': float(wall_s),
        'barrier_final': float(last['barrier'][-1]) if last['barrier'] else None,
        'f_inf_ts_final': float(last['f_inf_ts'][-1]) if last['f_inf_ts'] else None,
        'fperp_inf_ts_final': float(last['fperp_inf_ts'][-1]) if last['fperp_inf_ts'] else None,
        'stage_iters': [
            (s['converged_at'] + 1 if s['converged_at'] is not None else s['n_iter'])
            for s in stages
        ],
        'stage_elapsed_s': [float(s['elapsed_s']) for s in stages],
    }


def sweep_system(system, seeds):
    spec = SYSTEMS[system]
    sys_out = os.path.join(OUT_BASE, system)
    os.makedirs(sys_out, exist_ok=True)
    huber_base = yaml.safe_load(open(spec['huber_cfg']))

    results = []
    for seed in seeds:
        # Baseline two-stage.
        tag = f'baseline_s{seed}'
        out_dir = os.path.join(sys_out, tag)
        r = run_one(spec['baseline_cfg'], seed, out_dir, spec['monitor_every'])
        if r is not None:
            r.update({'system': system, 'config': 'baseline_two_stage',
                      'seed': seed, 'delta': None})
            results.append(r)
            with open(os.path.join(sys_out, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)

        # δ-sweep on single-stage Huber.
        for delta in DELTAS:
            tag = f'huber_d{delta:.0e}_s{seed}'
            out_dir = os.path.join(sys_out, tag)
            cfg_path = os.path.join(out_dir, 'config.yaml')
            os.makedirs(out_dir, exist_ok=True)
            write_huber_with_delta(huber_base, delta, cfg_path)
            r = run_one(cfg_path, seed, out_dir, spec['monitor_every'])
            if r is not None:
                r.update({'system': system, 'config': f'huber_delta_{delta:.0e}',
                          'seed': seed, 'delta': float(delta)})
                results.append(r)
                with open(os.path.join(sys_out, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    print_summary(system, results)


def print_summary(system, results):
    print(f'\n=== {system} summary ===', flush=True)
    print(f'{"config":<22s} {"seed":>4s} {"iters":>6s} {"wall_s":>8s} '
          f'{"barrier":>9s} {"fperp_TS":>11s}')
    for r in results:
        bar = f'{r["barrier_final"]:.4f}' if r['barrier_final'] is not None else '   nan'
        fp = f'{r["fperp_inf_ts_final"]:.4e}' if r['fperp_inf_ts_final'] is not None else '   nan'
        print(f'{r["config"]:<22s} {r["seed"]:>4d} {r["total_iter"]:>6d} '
              f'{r["wall_s"]:>8.1f} {bar:>9s} {fp:>11s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=list(SYSTEMS.keys()),
                    choices=list(SYSTEMS.keys()),
                    help='Subset of systems to sweep.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = ap.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)
    for system in args.systems:
        sweep_system(system, args.seeds)


if __name__ == '__main__':
    main()
