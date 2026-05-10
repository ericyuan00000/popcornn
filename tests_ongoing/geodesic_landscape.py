"""Unified geodesic-loss landscape harness — no UMA, just optimization.

Tests the user's hypothesis that rigid-body transformations (rotation,
translation) of one endpoint should add zero geodesic loss in
principle, so the rotated minimum should equal the non-rotated minimum.
The single metric is the integrated geodesic loss along the optimized
path.

Usage:
    python geodesic_landscape.py STAGE_NAME
    where STAGE_NAME ∈ {smoke, translation, angle_sweep, iter_scaling,
                        lr_capacity, procrustes, trans_x_rot,
                        optimizer, many_seed}

Output: /pscratch/sd/e/ericyuan/temp/popcornn_geodesic_landscape/<STAGE>/
        per-config JSON files plus a summary.json.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
from ase.io import read

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator

REPO = '/global/u2/e/ericyuan/GitHub/Popcornn'
XYZ = f'{REPO}/examples/configs/rxn0003.xyz'
ROOT_OUT = '/pscratch/sd/e/ericyuan/temp/popcornn_geodesic_landscape'

# Default mix recipe = the shipped rxn0003 stage-1 default
DEFAULT = dict(
    rtol=1.0e-1,
    atol=1.0e-4,
    n_embed=4,
    depth=2,
    activation='gelu',
    lr=1.0e-3,
    n_iter=1000,
    optimizer='adam',
    alpha=1.7,
    beta=0.01,
    r0=None,           # None = covalent
    rotation_deg=0.0,  # 0 = non-rot
    rotation_axis='z',
    translation=(0.0, 0.0, 0.0),
    align_endpoints=False,
)


@dataclass
class Config:
    name: str
    seed: int = 0
    # All overrideable knobs (default = the shipped mix recipe)
    rtol: float = 1.0e-1
    atol: float = 1.0e-4
    n_embed: int = 4
    depth: int = 2
    activation: str = 'gelu'
    lr: float = 1.0e-3
    n_iter: int = 1000
    optimizer: str = 'adam'
    alpha: float = 1.7
    beta: float = 0.01
    r0: Optional[float] = None
    rotation_deg: float = 0.0
    rotation_axis: str = 'z'
    translation: tuple = (0.0, 0.0, 0.0)
    align_endpoints: bool = False


# ---- Image preprocessing ---------------------------------------------------

_AXIS_VEC = {'x': np.array([1, 0, 0]), 'y': np.array([0, 1, 0]), 'z': np.array([0, 0, 1])}


def rotation_matrix(angle_deg: float, axis: str = 'z') -> np.ndarray:
    """Rodrigues rotation matrix."""
    if abs(angle_deg) < 1e-12:
        return np.eye(3)
    theta = np.deg2rad(angle_deg)
    k = _AXIS_VEC[axis].astype(float)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=float)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def procrustes_align(target_pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """Find R minimizing ||R @ ref - target||² via SVD; return R-aligned ref.

    Both positions must be centered first (caller's responsibility) or COMs
    must be removed. We center both then SVD then add back ref's COM.
    """
    ref_com = ref_pos.mean(axis=0)
    target_com = target_pos.mean(axis=0)
    A = ref_pos - ref_com           # source
    B = target_pos - target_com     # target
    # We want R s.t. R @ A ≈ B
    # SVD of B^T A
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    aligned_target = (B @ R) + ref_com
    return aligned_target


def load_images(cfg: Config):
    atoms = read(XYZ, index=':')
    reactant = atoms[0].copy()
    product = atoms[-1].copy()

    if cfg.rotation_deg != 0.0:
        com = product.get_center_of_mass()
        pos = product.get_positions() - com
        R = rotation_matrix(cfg.rotation_deg, cfg.rotation_axis)
        pos = pos @ R.T
        product.set_positions(pos + com)

    if any(t != 0.0 for t in cfg.translation):
        product.set_positions(product.get_positions() + np.array(cfg.translation))

    if cfg.align_endpoints:
        # Procrustes-align product to reactant
        aligned = procrustes_align(reactant.get_positions(), product.get_positions())
        product.set_positions(aligned)

    return [reactant, product]


# ---- Run one config --------------------------------------------------------

def build_mep(cfg: Config, images):
    return Popcornn(
        images=images,
        path_params={
            'name': 'mlp',
            'n_embed': cfg.n_embed,
            'depth': cfg.depth,
            'activation': cfg.activation,
        },
        num_record_points=101,
        device='cuda',
        seed=cfg.seed,
    )


def setup_stage(mep, cfg: Config):
    pot = get_potential(images=mep.images, name='repel',
                        alpha=cfg.alpha, beta=cfg.beta,
                        device=mep.device, dtype=mep.dtype)
    if cfg.r0 is not None:
        pot.radii.fill_(cfg.r0 / 2.0)
    mep.path.set_potential(pot)

    integ = PathIntegrator(
        path_integrand_names='geodesic',
        rtol=cfg.rtol,
        atol=cfg.atol,
        track_loss=True,    # populate IntegralOutput.loss; ~2x cost per step
        device=mep.device,
        dtype=mep.dtype,
    )
    integ.save_samples = False

    opt_kwargs = {'name': cfg.optimizer, 'lr': cfg.lr}
    if cfg.optimizer == 'sgd':
        opt_kwargs['momentum'] = 0.9
    optr = PathOptimizer(
        path=mep.path,
        optimizer=opt_kwargs,
        threshold=None,
        patience=1,
        find_ts=False,
        device=mep.device,
        dtype=mep.dtype,
    )
    return integ, optr


def run_one(cfg: Config) -> dict:
    images = load_images(cfg)
    mep = build_mep(cfg, images)
    integ, optr = setup_stage(mep, cfg)

    loss_trace = []
    g_inf_trace = []

    t_start = time.perf_counter()
    for it in range(cfg.n_iter):
        out = optr.optimization_step(mep.path, integ)
        torch.cuda.synchronize()
        loss_v = float(out.loss[0].item()) if (hasattr(out, 'loss') and out.loss is not None) else float('nan')
        gi = float(out.grad_norm.item())
        loss_trace.append(loss_v)
        g_inf_trace.append(gi)
    wall = time.perf_counter() - t_start

    return dict(
        cfg=asdict(cfg),
        final_loss=loss_trace[-1],
        final_g_inf=g_inf_trace[-1],
        loss_trace=loss_trace,
        g_inf_trace=g_inf_trace,
        wall_s=wall,
    )


# ---- Stage configs ---------------------------------------------------------

def stage_smoke():
    out = []
    for regime, rotdeg in [('non_rot', 0.0), ('rot', 180.0)]:
        for seed in (0, 1, 2):
            out.append(Config(name=f'mix_{regime}_seed{seed}', seed=seed,
                              rotation_deg=rotdeg))
    return out


def stage_translation():
    out = []
    for regime, trans in [('non_rot', (0, 0, 0)), ('trans_z5', (0, 0, 5))]:
        for seed in (0, 1, 2):
            out.append(Config(name=f'mix_{regime}_seed{seed}', seed=seed,
                              translation=trans))
    return out


def stage_angle_sweep():
    out = []
    for ang in (0, 30, 60, 90, 135, 180):
        for seed in (0, 1, 2):
            out.append(Config(name=f'mix_rot{ang}_seed{seed}', seed=seed,
                              rotation_deg=float(ang)))
    return out


def stage_iter_scaling():
    out = []
    for n in (1000, 3000, 10000):
        for regime, rotdeg in [('non_rot', 0.0), ('rot', 180.0)]:
            for seed in (0, 1, 2):
                out.append(Config(name=f'mix_{regime}_iter{n}_seed{seed}',
                                  seed=seed, rotation_deg=rotdeg, n_iter=n))
    return out


def stage_lr_capacity():
    out = []
    for lr in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        for n_embed in (2, 4, 8, 16):
            for depth in (2, 4):
                for seed in (0, 1, 2):
                    out.append(Config(
                        name=f'mix_rot_lr{lr:.0e}_n{n_embed}d{depth}_seed{seed}',
                        seed=seed, rotation_deg=180.0,
                        lr=lr, n_embed=n_embed, depth=depth,
                    ))
    return out


def stage_procrustes():
    out = []
    for align in (False, True):
        tag = 'aligned' if align else 'noalign'
        for seed in (0, 1, 2):
            out.append(Config(name=f'mix_rot_{tag}_seed{seed}', seed=seed,
                              rotation_deg=180.0, align_endpoints=align))
    # Non-rot baseline (no transformations) for direct comparison
    for seed in (0, 1, 2):
        out.append(Config(name=f'mix_nonrot_baseline_seed{seed}', seed=seed))
    return out


def stage_trans_x_rot():
    out = []
    for trans_z in (0.0, 2.5, 5.0):
        for rotdeg in (0, 90, 180):
            for seed in (0, 1, 2):
                out.append(Config(
                    name=f'mix_t{trans_z}_r{rotdeg}_seed{seed}',
                    seed=seed,
                    rotation_deg=float(rotdeg),
                    translation=(0.0, 0.0, trans_z),
                ))
    return out


def stage_optimizer():
    """Adam vs SGD-momentum at two recipes:
       (A) default (n4d2, lr=1e-3) — the failing config; can SGD close the gap?
       (B) winning (n16d4, lr=3e-3) — does optimizer matter at the working spot?
    Skip LBFGS: PathOptimizer's loop doesn't pass a closure, so LBFGS won't work.
    """
    out = []
    for opt in ('adam', 'sgd'):
        for seed in (0, 1, 2):
            # default config
            out.append(Config(name=f'default_{opt}_seed{seed}',
                              seed=seed, rotation_deg=180.0, optimizer=opt))
            # winning config
            out.append(Config(name=f'best_{opt}_seed{seed}',
                              seed=seed, rotation_deg=180.0, optimizer=opt,
                              lr=3e-3, n_embed=16, depth=4))
    return out


def stage_many_seed():
    """20 seeds at the BEST (lr, capacity) from Stage 5, plus 20 at the default
    for direct comparison. Tests whether the variance comes down with the
    better config (it should, per Stage 5's std=0.039 at the winner vs 0.380 at
    the default)."""
    out = []
    # Default recipe (lr=1e-3, n4d2) — already have 3 seeds, fill 4-19
    for seed in range(20):
        out.append(Config(name=f'default_rot_seed{seed}', seed=seed, rotation_deg=180.0))
    # Stage 5 winner: lr=3e-3, n16d4
    for seed in range(20):
        out.append(Config(name=f'best_rot_seed{seed}', seed=seed, rotation_deg=180.0,
                          lr=3e-3, n_embed=16, depth=4))
    return out


STAGES = {
    'smoke': stage_smoke,
    'translation': stage_translation,
    'angle_sweep': stage_angle_sweep,
    'iter_scaling': stage_iter_scaling,
    'lr_capacity': stage_lr_capacity,
    'procrustes': stage_procrustes,
    'trans_x_rot': stage_trans_x_rot,
    'optimizer': stage_optimizer,
    'many_seed': stage_many_seed,
}


# ---- Driver ----------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print('usage: python geodesic_landscape.py STAGE', flush=True)
        print(f'STAGE ∈ {list(STAGES.keys())}', flush=True)
        sys.exit(1)
    stage = sys.argv[1]
    out_dir = f'{ROOT_OUT}/{stage}'
    os.makedirs(out_dir, exist_ok=True)

    configs = STAGES[stage]()
    print(f'Stage {stage}: {len(configs)} configs', flush=True)
    print(f'Output: {out_dir}', flush=True)

    summary = []
    t0 = time.perf_counter()
    for i, cfg in enumerate(configs):
        json_path = f'{out_dir}/{cfg.name}.json'
        if os.path.exists(json_path):
            # Resume support: skip configs already done. Load existing data
            # so the summary stays complete after a restart.
            with open(json_path) as f:
                result = json.load(f)
            summary.append({
                'name': cfg.name,
                'final_loss': result['final_loss'],
                'final_g_inf': result['final_g_inf'],
                'wall_s': result['wall_s'],
                **{k: v for k, v in asdict(cfg).items() if k != 'name'},
            })
            print(f'  [{i+1:>3d}/{len(configs)}] {cfg.name:>50}  '
                  f'(skipped, already done) final_loss={result["final_loss"]:.6e}', flush=True)
            continue
        t_cfg = time.perf_counter()
        result = run_one(cfg)
        wall = time.perf_counter() - t_cfg
        summary.append({
            'name': cfg.name,
            'final_loss': result['final_loss'],
            'final_g_inf': result['final_g_inf'],
            'wall_s': result['wall_s'],
            **{k: v for k, v in asdict(cfg).items() if k != 'name'},
        })
        with open(json_path, 'w') as f:
            json.dump(result, f)
        print(f'  [{i+1:>3d}/{len(configs)}] {cfg.name:>50}  '
              f'final_loss={result["final_loss"]:.6e}  '
              f'final_|g|={result["final_g_inf"]:.3e}  '
              f'wall={wall:.1f}s', flush=True)

    elapsed = time.perf_counter() - t0
    print(f'\nstage {stage} done: {len(configs)} runs in {elapsed/60:.1f} min', flush=True)

    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump({'stage': stage, 'n_configs': len(configs),
                   'total_wall_s': elapsed, 'configs': summary}, f, indent=2)
    print(f'wrote {out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
