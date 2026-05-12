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

A canonical multi-leg pattern uses two losses with the same
saddle-point physics but different optimization dynamics:

- **`pvre_squared`** has a $C^\infty$-smooth integrand, so adaptive
  Gauss–Kronrod converges in one or two passes per step (~5× cheaper
  than `pvre`). It drives the path most of the way to the MEP, but its
  gradient $\partial \ell / \partial \theta \propto 2(v\!\cdot\!F)$
  vanishes near the saddle ridge, so it plateaus before pinpointing
  the TS.
- **`pvre`**'s sign-driven gradient keeps pushing once warm-started,
  snapping the path onto the saddle in a small number of iterations.

`examples/configs/lj13_pvre_two_stage.yaml` is a worked two-stage
example you can use as a template. The current shipped
`muller_brown.yaml` and `lj13.yaml` use a different design point —
**single-stage pvre on a small MLP (n_embed=4, depth=2) with a
deterministic patience=1 trigger**. That recipe is described in
[Convergence](convergence.md) and gives consistent wall + path quality
across seeds with the strict `atol/threshold = 0.1` noise-floor rule.

When to use which design:

- **Single-stage pvre + n4d2** (the shipped default) — easiest to
  reason about, deterministic stop, no mid-run loss switch. Best for
  callers that just want a path and don't need the very last decade
  of TS-force quality.
- **Two-stage pvre² → pvre** — wins when the per-iter integrator cost
  of `pvre` dominates and `pvre_squared`'s 5× cheaper integration can
  amortize a warm-up. Tune the stage-1 threshold from a pilot of the
  warm-up's own gradient scale (typically ~$g_\infty / 30$). If
  stage-1's $|F_\perp|_\mathrm{TS}$ is non-monotonic (LJ-13 case
  observed: descends then oscillates back up 44×), set stage-1
  threshold *before* the rebound; otherwise the path arrives at stage
  2 worse than where it bottomed.
- **Two-stage pvre² → pseudo-Huber δ** — useful when stage-2 needs to
  bridge between the smooth `pvre_squared` warm-up and the sharp
  `pvre` ridge without picking up `pvre`'s integrator cost; small δ
  (≤ 0.01 on chemistry units) keeps the gradient sign-driven near the
  saddle while staying smooth elsewhere.

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

`ts_time_loss_names` / `ts_time_loss_scales` apply an extra loss at
the predicted TS time, useful e.g. for minimizing the force magnitude
at the TS (`F_mag` as a TS-time loss). The scales can be scheduled
with `ts_time_loss_schedulers`.

The TS itself is picked by `BasePath.ts_search`: an `argmax` over the
per-quadrature-point energy cache that the integrator already collects
during the gradient pass. There is no separate TS optimization — the
saddle's resolution is set by the integrator's `rtol` / `atol`. Tighten
those if `ts_image` looks coarse.

## Logging per-iteration state

Every run prints a sparse per-iter table to stdout (header at leg
start, rows at iters 0/5/10/25/50/75/100/150/200/250 then every 50,
plus the last; columns iter / loss / |g|_∞ / |g|_2 / step_s).
This is on by default — no setup needed.

For a programmatic record of the same scalar metrics, pass
`metrics_log_path=<dir>` to `optimize_path`. Each leg writes one
JSONL file `<dir>/opt_<leg-index>.jsonl` with one row per iteration:
`iter`, `loss`, `grad_norm_inf`, `grad_norm_2`, `lr`, `step_s`,
`wall_s`, `converged`. Rows are flushed each iteration so a killed
run still leaves a valid file.

When `output_dir` is set on the `Popcornn` constructor and you
*don't* pass `metrics_log_path`, the JSONL log defaults to
`<output_dir>/metrics/` so it lands next to the heavy per-iter dump
described next.

For full per-iter state — the path itself, energies, forces — set
`output_dir`. Each leg then also writes one JSON file per iteration
to `<output_dir>/opt_<leg-index>/logs/output_<iter>.json`, with:

- `time`, `positions`, `energies`, `velocities`, `forces` — the path
  evaluated at the integrator's quadrature times.
- `loss_evals` — per-time loss evaluations.
- `grad_norm` — the L∞ norm of the path-integrated gradient (the
  convergence signal).
- `ts_time`, `ts_positions`, `ts_energies`, `ts_velocities`,
  `ts_forces` — the predicted TS at this iteration.
- `loss` — the scalar loss integral $\int \mathcal{L}\,\mathrm{d}t$;
  only present if `track_loss: true`.

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
