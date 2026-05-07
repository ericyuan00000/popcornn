"""Sweep (rtol, atol, lr) on Müller-Brown to recommend rough/tight stage settings.

Two signals, two purposes:

  - **Descent** — does this cell make progress? — comes from ∫L dt, the loss
    integral. Adam should drive it monotonically down. ‖∫∇L dt‖ is the
    *gradient* of the loss; it wiggles even on healthy cells, so don't use it
    to judge progress.
  - **Noise floor** — where does ‖∫∇L dt‖ plateau? — sets threshold. Read
    the late-window median of ‖grad‖ off long runs, since the floor is a
    stationary distribution, not a transient.

Sweep A (32 cells, 50 steps each): cross product of (rtol, atol, lr). Picks
rough (rtol, atol, lr).

Sweep B (3 cells, 200 steps each, lr=1e-3): long runs at tight tolerances to
expose the ‖grad‖ floor. Picks tight (rtol, atol, threshold).

Loss measurement inherits the gradient (rtol, atol). At loss magnitude ~ O(100)
on Müller-Brown, even rtol=1e-1 leaves enough SNR to read off a ~2× drop.

Run from the Popcornn repo root:

    python tests_ongoing/sweep_tolerance.py

CPU is fine; Müller-Brown is 2D.
"""
import copy
import json
import math
import os
import time as time_mod
from itertools import product
from statistics import median

from popcornn import Popcornn
from popcornn.optimization.path_optimizer import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, 'examples', 'configs', 'muller_brown.yaml')

# Sweep A: short runs, descent-speed map.
A_RTOLS = [1e-1, 1e-2, 1e-3, 1e-5]
A_ATOLS = [1e-2, 1e-7]            # rtol dominates; two anchors
A_LRS = [1e-1, 1e-2, 1e-3, 1e-4]
A_STEPS = 50

# Sweep B: long tight runs, noise-floor probe. Adaptive: run until the loss
# plateaus (rolling-window check) or B_MAX_STEPS, whichever comes first.
B_CELLS = [(1e-3, 1e-3), (1e-5, 1e-7), (1e-7, 1e-7)]
B_LR = 1e-3
B_MAX_STEPS = 2000
B_PLATEAU_CHECK_EVERY = 50  # steps; need ≥ 2 windows of WINDOW_LEN before checking
B_PLATEAU_WINDOW = 50       # rolling-median window size
B_PLATEAU_TOL = 0.02        # |L_recent - L_prior| / |L_prior| ≤ 2% → plateau
B_MIN_STEPS = 200           # never stop before this — early dynamics are noisy

# Late-window for descent/floor stats.
LATE_FRAC = 0.2          # last 20% of any trajectory
PLATEAU_FRAC_TOL = 0.05  # |L_late - L_pre| / |L_pre| ≤ 5% means plateaued

# Loss measurement inherits the gradient (rtol, atol) — see PathIntegrator
# defaults at popcornn/tools/integrator.py:48. At observed L ~ O(100), even
# rtol=1e-1 leaves enough SNR to read off a 2× drop, so paying for tight loss
# quadrature on every cell wastes wall time.


def run_one(rtol, atol, lr, base_cfg, n_steps, track_loss=True,
            stop_on_plateau=False, plateau_window=B_PLATEAU_WINDOW,
            plateau_tol=B_PLATEAU_TOL, plateau_check_every=B_PLATEAU_CHECK_EVERY,
            min_steps=B_MIN_STEPS):
    cfg = copy.deepcopy(base_cfg)
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['rtol'] = rtol
    leg['integrator_params']['atol'] = atol
    leg['integrator_params']['track_loss'] = track_loss
    leg['optimizer_params']['optimizer']['lr'] = lr

    mep = Popcornn(**cfg.get('initialization_params', {}))
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg['integrator_params'],
                          device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    grad_norms, losses, n_nodes, wall = [], [], [], []
    plateaued = False
    for step in range(n_steps):
        t0 = time_mod.perf_counter()
        try:
            out = optr.optimization_step(mep.path, integ)
        except Exception as exc:
            return {
                'rtol': rtol, 'atol': atol, 'lr': lr,
                'grad_norms': grad_norms, 'losses': losses,
                'n_nodes': n_nodes, 'wall_per_step': wall,
                'failed': True, 'error': str(exc), 'plateaued': False,
            }
        wall.append(time_mod.perf_counter() - t0)
        grad_norms.append(float(out.grad_norm.item()))
        n_nodes.append(int(out.t.shape[0]))
        if track_loss:
            losses.append(float(out.loss[0].item()))
        if not math.isfinite(grad_norms[-1]):
            break
        if track_loss and not math.isfinite(losses[-1]):
            break
        # Adaptive plateau early-stop: only after min_steps and at every
        # plateau_check_every. Compares rolling-median of last plateau_window
        # against the previous plateau_window.
        if (stop_on_plateau and track_loss
                and step + 1 >= max(min_steps, 2 * plateau_window)
                and (step + 1) % plateau_check_every == 0):
            recent = window_median(losses,
                                   len(losses) - plateau_window,
                                   len(losses))
            prior = window_median(losses,
                                  len(losses) - 2 * plateau_window,
                                  len(losses) - plateau_window)
            if (recent is not None and prior is not None and prior != 0
                    and abs(recent - prior) <= plateau_tol * abs(prior)):
                plateaued = True
                break
    return {
        'rtol': rtol, 'atol': atol, 'lr': lr,
        'grad_norms': grad_norms, 'losses': losses,
        'n_nodes': n_nodes, 'wall_per_step': wall,
        'failed': False, 'plateaued': plateaued,
    }


def window_median(xs, lo, hi):
    if not xs:
        return None
    sub = [x for x in xs[lo:hi] if math.isfinite(x)]
    return median(sub) if sub else None


def late_window(xs):
    if not xs:
        return (0, 0)
    n = len(xs)
    lo = max(0, n - max(1, int(round(LATE_FRAC * n))))
    return (lo, n)


def is_descending(losses):
    if len(losses) < 20:
        return False
    L_early = window_median(losses, 0, 10)
    lo, hi = late_window(losses)
    L_late = window_median(losses, lo, hi)
    if L_early is None or L_late is None:
        return False
    return L_late <= 0.5 * L_early


def is_diverging(losses):
    if any(not math.isfinite(x) for x in losses):
        return True
    if len(losses) < 20:
        return False
    L_early = window_median(losses, 0, 10)
    lo, hi = late_window(losses)
    L_late = window_median(losses, lo, hi)
    if L_early is None or L_late is None:
        return False
    return L_late > 1.5 * L_early


def round_up_pow10(x):
    if x is None or x <= 0 or not math.isfinite(x):
        return None
    return 10.0 ** math.ceil(math.log10(x))


def fmt(x):
    return f'{x:.0e}' if isinstance(x, float) and math.isfinite(x) else str(x)


def run_sweep_a(base_cfg):
    print('=' * 80)
    print(f'Sweep A: (rtol, atol) × lr at {A_STEPS} steps — descent map')
    print('=' * 80)
    header = (f'{"rtol":>8s} {"atol":>8s} {"lr":>8s} '
              f'{"L_early":>10s} {"L_late":>10s} {"L_ratio":>8s} '
              f'{"|g|_late":>10s} {"nodes":>6s} {"sec/it":>8s} {"verdict":>8s}')
    print(header)
    print('-' * len(header))
    rows = []
    for rtol, atol, lr in product(A_RTOLS, A_ATOLS, A_LRS):
        r = run_one(rtol, atol, lr, base_cfg, A_STEPS, track_loss=True)
        rows.append(r)
        if r['failed'] or not r['losses']:
            err = r.get('error', 'empty trajectory')
            print(f'{rtol:>8.0e} {atol:>8.0e} {lr:>8.0e}  failed: {err}')
            continue
        L, G = r['losses'], r['grad_norms']
        L_early = window_median(L, 0, 10)
        lo, hi = late_window(L)
        L_late = window_median(L, lo, hi)
        L_ratio = (L_late / L_early) if (L_early and L_early != 0) else float('nan')
        G_late = window_median(G, lo, hi)
        nodes_med = sorted(r['n_nodes'])[len(r['n_nodes']) // 2]
        sec = sum(r['wall_per_step']) / len(r['wall_per_step'])
        if is_diverging(L):
            verdict = 'DIVERGE'
        elif is_descending(L):
            verdict = 'DESCEND'
        else:
            verdict = 'flat'
        print(f'{rtol:>8.0e} {atol:>8.0e} {lr:>8.0e} '
              f'{L_early:>10.3e} {L_late:>10.3e} {L_ratio:>8.3f} '
              f'{G_late:>10.3e} {nodes_med:>6d} {sec:>7.4f}s {verdict:>8s}')
    return rows


def run_sweep_b(base_cfg):
    print()
    print('=' * 80)
    print(f'Sweep B: long tight runs at lr={B_LR:.0e}, ≤{B_MAX_STEPS} steps '
          f'(early-stop on plateau) — noise floor')
    print('=' * 80)
    header = (f'{"rtol":>8s} {"atol":>8s} {"steps":>6s} {"L_late":>10s} '
              f'{"L_pre":>10s} {"plateau":>8s} {"|g|_late":>10s} '
              f'{"|g|_min":>10s} {"|g|_max":>10s} {"nodes":>6s} {"sec/it":>8s}')
    print(header)
    print('-' * len(header))
    rows = []
    for rtol, atol in B_CELLS:
        r = run_one(rtol, atol, B_LR, base_cfg, B_MAX_STEPS, track_loss=True,
                    stop_on_plateau=True)
        rows.append(r)
        if r['failed'] or not r['losses']:
            err = r.get('error', 'empty trajectory')
            print(f'{rtol:>8.0e} {atol:>8.0e}  failed: {err}')
            continue
        L, G = r['losses'], r['grad_norms']
        n = len(L)
        win = max(1, int(round(LATE_FRAC * n)))
        L_late = window_median(L, n - win, n)
        L_pre = window_median(L, max(0, n - 2 * win), n - win)
        plateau = r.get('plateaued', False)
        G_window = G[n - win:]
        G_late = median(G_window)
        G_min, G_max = min(G_window), max(G_window)
        nodes_med = sorted(r['n_nodes'])[len(r['n_nodes']) // 2]
        sec = sum(r['wall_per_step']) / len(r['wall_per_step'])
        plateau_str = 'yes' if plateau else f'no@{n}'
        print(f'{rtol:>8.0e} {atol:>8.0e} {n:>6d} {L_late:>10.3e} {L_pre:>10.3e} '
              f'{plateau_str:>8s} '
              f'{G_late:>10.3e} {G_min:>10.3e} {G_max:>10.3e} '
              f'{nodes_med:>6d} {sec:>7.4f}s')
    return rows


def pick_rough(a_rows):
    """Loosest descending Sweep A cell, prefer largest lr.

    Rough threshold = 10 × late-window ‖grad‖ median, then rounded UP to
    the next power of ten — sits one order above the observed floor so
    patience=3 actually triggers convergence.
    """
    descending = [r for r in a_rows
                  if not r['failed'] and is_descending(r['losses'])]
    if not descending:
        return None
    # Loosest tolerance: largest rtol, tiebreak largest atol.
    rtol_max = max(r['rtol'] for r in descending)
    at_rtol = [r for r in descending if r['rtol'] == rtol_max]
    atol_max = max(r['atol'] for r in at_rtol)
    at_tol = [r for r in at_rtol if r['atol'] == atol_max]
    lr_max = max(r['lr'] for r in at_tol)
    chosen = next(r for r in at_tol if r['lr'] == lr_max)

    G = chosen['grad_norms']
    lo, hi = late_window(G)
    G_late = window_median(G, lo, hi)
    threshold = round_up_pow10(10 * G_late) if G_late else None
    return {
        'rtol': chosen['rtol'], 'atol': chosen['atol'], 'lr': chosen['lr'],
        'threshold': threshold,
        'g_floor_observed': G_late,
    }


def pick_tight(b_rows, a_rows):
    """Loosest Sweep B cell with |g|_late < 1e-3 AND loss plateaued.

    Tight threshold = late-window ‖grad‖ median rounded UP to the next
    power of ten. Hard-floor at 1e-4 (fp32 limit on a 2D potential).
    """
    candidates = []
    for r in b_rows:
        if r['failed'] or not r['losses']:
            continue
        if not r.get('plateaued', False):
            continue  # ran to B_MAX_STEPS without plateauing — not at the floor
        L, G = r['losses'], r['grad_norms']
        n = len(L)
        win = max(1, int(round(LATE_FRAC * n)))
        G_late = median(G[n - win:])
        if G_late >= 1e-3:
            continue
        candidates.append((r, G_late))
    if not candidates:
        return None
    rtol_max = max(r['rtol'] for r, _ in candidates)
    at_rtol = [(r, g) for r, g in candidates if r['rtol'] == rtol_max]
    atol_max = max(r['atol'] for r, _ in at_rtol)
    chosen, G_late = next((r, g) for r, g in at_rtol if r['atol'] == atol_max)

    # lr: 1e-3 if Sweep A's matching cell descended; else 1e-4.
    lr = 1e-3
    a_match = [r for r in a_rows
               if r['rtol'] == chosen['rtol']
               and r['atol'] == chosen['atol']
               and r['lr'] == 1e-3]
    if not a_match or not is_descending(a_match[0]['losses']):
        lr = 1e-4

    threshold = round_up_pow10(G_late)
    if threshold is None or threshold < 1e-4:
        threshold = 1e-4
    return {
        'rtol': chosen['rtol'], 'atol': chosen['atol'], 'lr': lr,
        'threshold': threshold,
        'g_floor_observed': G_late,
    }


def print_recommendation(rough, tight):
    print()
    print('=' * 80)
    print('RECOMMENDATION')
    print('=' * 80)
    if rough is None:
        print('rough: no descending cell — extend A_STEPS or revisit lr grid')
    else:
        print(f'rough: rtol={fmt(rough["rtol"])} '
              f'atol={fmt(rough["atol"])} '
              f'lr={fmt(rough["lr"])} '
              f'threshold={fmt(rough["threshold"])}  '
              f'(observed |g| floor ≈ {rough["g_floor_observed"]:.2e})')
    if tight is None:
        print('tight: no cell met |g|_late < 1e-3 with plateau — extend B_STEPS')
    else:
        print(f'tight: rtol={fmt(tight["rtol"])} '
              f'atol={fmt(tight["atol"])} '
              f'lr={fmt(tight["lr"])} '
              f'threshold={fmt(tight["threshold"])}  '
              f'(observed |g| floor ≈ {tight["g_floor_observed"]:.2e})')


def main():
    base_cfg = import_run_config(CONFIG)
    print(f'config: {CONFIG}')
    print('loss measurement: inherits gradient (rtol, atol)')
    print()
    a_rows = run_sweep_a(base_cfg)
    b_rows = run_sweep_b(base_cfg)
    rough = pick_rough(a_rows)
    tight = pick_tight(b_rows, a_rows)
    print_recommendation(rough, tight)

    out_json = os.path.join(REPO_ROOT, 'tests_ongoing', 'sweep_optim_settings.json')
    with open(out_json, 'w') as f:
        json.dump({'sweep_a': a_rows, 'sweep_b': b_rows,
                   'rough': rough, 'tight': tight}, f)
    print(f'\nfull trajectories: {out_json}')


if __name__ == '__main__':
    main()
