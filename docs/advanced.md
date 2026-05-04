# Advanced

Topics beyond the first-time user path.

## Multi-leg optimization

`Popcornn.optimize_path` accepts an arbitrary number of leg dicts.
Each leg picks up the path from where the previous one left off, then
swaps in its own potential, loss, and optimizer settings. The
canonical pattern is **clash-resolution then real run**:

```python
final_images, ts_image = mep.optimize_path(
    {
        # leg 1: cheap repulsive geodesic interpolation
        "potential_params": {"potential": "repel"},
        "integrator_params": {"path_integrand_names": "geodesic"},
        "optimizer_params": {"optimizer": {"name": "adam", "lr": 1.0e-1}},
        "num_optimizer_iterations": 1000,
    },
    {
        # leg 2: MLIP-driven TS search
        "potential_params": {
            "potential": "uma",
            "model_name": "uma-s-1p1",
            "task_name": "omol",
        },
        "integrator_params": {
            "path_integrand_names": "pvre",
            "rtol": 1.0e-2,
            "atol": 1.0e-2,
        },
        "optimizer_params": {
            "optimizer": {"name": "adam", "lr": 1.0e-3},
            "threshold": 1.0e-1,
        },
        "num_optimizer_iterations": 1000,
    },
)
```

You can chain more legs — for example, switch loss functions
mid-optimization, or step the learning rate down across legs. The
path's network parameters are persistent state on the `Popcornn`
instance.

### Smooth-then-sharp loss schedule (`pvre_squared → pvre`)

The other canonical multi-leg pattern uses two losses with the same
saddle-point physics but different optimization dynamics:

- **`pvre_squared`** has a $C^\infty$-smooth integrand, so adaptive
  Gauss–Kronrod converges in one or two passes per step (~5× cheaper
  than `pvre`). It drives the path most of the way to the MEP, but its
  gradient $\partial \ell / \partial \theta \propto 2(v\!\cdot\!F)$
  vanishes near the saddle ridge, so it plateaus before pinpointing
  the TS.
- **`pvre`**'s sign-driven gradient keeps pushing once warm-started,
  snapping the path onto the saddle in a small number of iterations.

`examples/configs/muller_brown.yaml` ships this schedule and is ~3.5×
faster than the equivalent single-leg `pvre` config with marginally
better TS recovery. The Stage-2 learning rate is typically ~1/10 of a
single-leg `pvre` rate because the path is already close — Adam's small
steps refine more precisely than they recover. Stage-1 threshold
follows the [convergence recipe](convergence.md) on the warm-up's own
gradient scale (initial $g_\infty / \sim\!30$).

`examples/configs/lj13.yaml` ships the same schedule on a 39-dim
atomistic system (13-atom Lennard-Jones cluster, permutation/inversion
saddle). On this system the two-stage benefit shows up in path
geometry rather than wall time: across three seeds, the perpendicular
force at the saddle $|F_\perp|_\mathrm{TS}$ drops from ~0.22 (single-stage
`pvre_squared` at 600 iters) to ~0.021 (two-stage 300+300 iters), while
the energy barrier is reached by either schedule to within 0.04%. The
`pvre` fine-tune is what tightens the path geometry; running
`pvre_squared` longer is no substitute. Learning rates are 10× lower
than Müller–Brown's because LJ-reduced-units gradients are O(1) rather
than O(1e4).

## Schedulers

Three independent scheduler families are available per leg. Each is a
dict whose keys name what's being scheduled and whose values
configure the scheduler.

### `lr_scheduler` — learning rate

Any class from `torch.optim.lr_scheduler` (case-insensitive). Steps
once per optimization iteration.

```yaml
optimizer_params:
  optimizer:
    name: adam
    lr: 1.0e-3
  lr_scheduler:
    name: cosineannealinglr
    T_max: 1000
    eta_min: 1.0e-5
```

### `path_integrand_schedulers` — per-loss-term weights

Multiplies entries of `path_integrand_scales` (in `integrator_params`) by a
schedule. Useful for ramping one term down while another ramps up.

```yaml
integrator_params:
  path_integrand_names: ['pvre', 'vre']
  path_integrand_scales: [1.0, 0.1]

optimizer_params:
  path_integrand_schedulers:
    pvre:
      value: 1.0
      name: cosine
      start_value: 1.0
      end_value: 0.0
      last_step: 99
    vre:
      value: 1.0
      name: cosine
      start_value: 0.0
      end_value: 1.0
      last_step: 99
```

This config ramps the pVRE term from full weight to zero, and VRE from
zero to full weight, over the first 100 iterations.

Available scheduler types:

| `name` | Behavior |
| --- | --- |
| `linear` | Linear interpolation from `start_value` to `end_value` over `last_step`. |
| `cosine` | Cosine-anneal from `start_value` to `end_value` over `last_step`. |

## Transition-state losses

Two extra loss types apply specifically at or near the predicted
transition state:

- `ts_time_loss_names` / `ts_time_loss_scales` — applied at a single
  time, the predicted TS time.
- `ts_region_loss_names` / `ts_region_loss_scales` — applied across a
  small time window around the predicted TS.

These are useful for, e.g., minimizing the force magnitude at the TS
(`F_mag` as a TS-time loss) or maximizing the TS energy (`E_mean` as
a TS-region loss). Each can also be scheduled with
`ts_time_loss_schedulers` / `ts_region_loss_schedulers`.

The TS itself is picked by `BasePath.ts_search`: an `argmax` over the
per-quadrature-point energy cache that the integrator already collects
during the gradient pass. There is no separate TS optimization — the
saddle's resolution is set by the integrator's `rtol` / `atol`. Tighten
those if `ts_image` looks coarse.

## Logging per-iteration state

If you set `output_dir` on the `Popcornn` constructor, each leg
writes one JSON file per iteration to
`<output_dir>/opt_<leg-index>/logs/output_<iter>.json`. Each file
contains:

- `time`, `positions`, `energies`, `velocities`, `forces` — the path
  evaluated at the integrator's quadrature times.
- `loss_evals` — per-time loss evaluations.
- `integral` — the scalar integral value.
- `grad_norm` — the L∞ norm of the path-integrated gradient (the
  convergence signal).
- `ts_time`, `ts_positions`, `ts_energies`, `ts_velocities`,
  `ts_forces` — the predicted TS at this iteration.
- `loss_integral` — only present if `track_loss: true`.

This is a lot of data. Don't enable it for production runs unless
you're debugging.

## Custom path integrands

To add a new per-point loss term, edit `popcornn/tools/integrand.py`,
write a subclass of `PathIntegrand`, and register it in
`PATH_INTEGRANDS`:

```python
class MyLoss(PathIntegrand):
    requires = ('forces', 'velocities')   # cache keys this term consumes

    def evaluate(self, variables) -> torch.Tensor:
        # return shape [N, 1]
        return ...

PATH_INTEGRANDS['my_loss'] = MyLoss
```

Then reference it from any config as `path_integrand_names: my_loss`.

## Custom path / potential classes

See [Paths](paths.md) and [Potentials](potentials.md).
