# Configuration reference

Popcornn reads a YAML config split into two top-level sections:

```yaml
initialization_params:
  # ... how to set up the path
optimization_params:
  - # ... first optimization leg
  - # ... second optimization leg
  - # ...
```

The `examples/run.py` driver passes `initialization_params` to
`Popcornn(...)` and unpacks `optimization_params` (a list) into
`Popcornn.optimize_path(*params)`. You can mirror this from your own
Python code; see [Getting Started](getting-started.md#4-use-the-python-api-directly).

## `initialization_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `images` | `str` or list | required | Path to an `xyz`/`traj`/`json`/`npy`/`pt` file, or a list of ASE `Atoms` objects, or a 2D coordinate list. First frame is the reactant, last is the product. Intermediate frames are non-fixed guesses the path is fitted through. |
| `unwrap_positions` | `bool` | `True` | For periodic systems, unwrap the product against the reactant using the minimum-image convention. Disable if any atom moves more than half a cell. |
| `path_params` | `dict` | `{}` | See below. |
| `num_record_points` | `int` | `101` | Number of frames sampled along the optimized path when returning/saving. |
| `output_dir` | `str` or `None` | `None` | Where to dump per-iteration JSON logs. `None` skips logging. |
| `device` | `str` or `None` | auto | `cuda` or `cpu`. `None` picks `cuda` if available. |
| `dtype` | `"float32"` or `"float64"` | `"float32"` | PyTorch dtype. |
| `seed` | `int` or `None` | `0` | RNG seed for reproducibility. `None` to skip seeding. |

### `path_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `"mlp"` or `"linear"` | required | Path representation. See [Paths](paths.md). |
| `n_embed` | `int` | `1` | (`mlp` only) Width multiplier on hidden layers. |
| `depth` | `int` | `2` | (`mlp` only) Number of MLP layers. `depth=2` is enough for simple reactions; deeper for more complex ones. |
| `activation` | `str` | `"gelu"` | Any activation in `torch.nn` (case-insensitive). |

## `optimization_params`

`optimization_params` is a **list of dicts**. Each dict is one leg of
optimization. Common pattern:

1. A cheap repulsive leg to resolve atom clashes (geodesic
   interpolation).
2. The expensive MLIP leg that finds the actual transition state.

Each leg has four sub-blocks:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `potential_params` | `dict` | `{}` | Energy model. See below. |
| `integrator_params` | `dict` | `{}` | Path integral and loss settings. See below. |
| `optimizer_params` | `dict` | `{}` | Adam / scheduler / convergence settings. See below. |
| `num_optimizer_iterations` | `int` | `1000` | Hard iteration cap. The actual run usually stops earlier when the convergence trigger fires. |

### `potential_params`

| Key | Description |
| --- | --- |
| `name` | One of: `wolfe_schlegel`, `muller_brown`, `schwefel`, `constant`, `sphere`, `lennard_jones`, `repel`, `morse`, `harmonic`, `mace`, `newtonnet`, `ani`, `leftnet`, `orb`, `escaip`, `uma`. See [Potentials](potentials.md) for the full list and per-MLIP install notes. |

Each potential takes its own extra keys (e.g. UMA needs `model_name`
and `task_name`; MACE needs a checkpoint path). The
[Potentials](potentials.md) page documents these per-potential.

### `integrator_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `path_integrand_names` | `str` or list of str | `None` | Per-point quantity (or list of them) integrated along the path. Options: `geodesic`, `pvre`, `vre`, `vre_variational_error`, `pvre_mag`, `E`, `E_mean`, `F_mag`. See [Loss functions](loss-functions.md) for what each one means. |
| `path_integrand_scales` | float or list | `1.0` | Per-term weighting when `path_integrand_names` is a list. |
| `method` | `str` | `"gk21"` | Adaptive-quadrature rule. `gk21` is Gauss–Kronrod 21-point. |
| `rtol` | `float` | `1.0e-6` | Relative tolerance for the adaptive integrator. |
| `atol` | `float` | `1.0e-7` | Absolute tolerance. |
| `max_batch` | `int` or `None` | `None` | Max batch size at any single quadrature step. `None` lets torchpathint auto-size against free GPU memory. |
| `track_loss` | `bool` | `False` | Run a separate detached integral of the loss itself per iteration so you can monitor `∫L dt`. Costs an extra pass but is debug-only. |
| `loss_rtol`, `loss_atol` | float | `rtol`, `atol` | Tolerances for the detached loss integral. Looser is fine since it's not used for gradients. |
| `total_mem_usage` | `float` | `0.9` | Fraction of free GPU memory torchpathint may allocate per batch. Lower if you hit persistent OOMs. See [Memory & OOM](memory-and-oom.md). |

### `optimizer_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `dict` | required | `{"name": "<torch.optim class>", ...kwargs}`. Names are case-insensitive (`adam`, `sgd`, `lbfgs`, …). All extra keys forward to the optimizer constructor (`lr`, `weight_decay`, …). |
| `lr_scheduler` | `dict` or `None` | `None` | `{"name": "<torch.optim.lr_scheduler class>", ...kwargs}`. Same convention. |
| `threshold` | `float` or `None` | `None` | Convergence trigger on `‖∫∇L dt‖_∞`. `None` disables the trigger and always runs the full `num_optimizer_iterations`. See [Convergence](convergence.md) for tuning. |
| `patience` | `int` | `5` | Number of consecutive iterations the trigger must hold before the leg exits. |
| `path_integrand_schedulers` | `dict` | `None` | Schedulers on `path_integrand_scales`. Useful for ramping a geodesic term down as a pVRE term ramps up. See [Advanced](advanced.md). |
| `find_ts` | `bool` or `None` | `None` (auto) | Force-enable / force-disable transition-state extraction. `None` inherits from the path's own flag (default `True`). When active, `BasePath.ts_search` runs every iteration on the integrator's sample cache and `ts_image` is returned by `optimize_path`. |
| `ts_time_loss_names`, `ts_time_loss_scales`, `ts_time_loss_schedulers` | various | `None` | Optional losses applied at the predicted transition-state time. See [Advanced](advanced.md). |
| `ts_region_loss_names`, `ts_region_loss_scales`, `ts_region_loss_schedulers` | various | `None` | Same, but applied across a small time window around the predicted TS. |

#### Scheduler dict

A scheduler entry has the form:

```yaml
path_integrand_schedulers:
  pvre:
    value: 1.0
    name: cosine          # or 'linear'
    start_value: 1.0
    end_value: 0.0
    last_step: 99
```

The `name` selects the scheduler class (`linear` or `cosine`); the
remaining keys are passed to its constructor. Schedulers step once
per optimization iteration.

## A complete example

The `examples/configs/rxn0003.yaml` config:

```yaml
initialization_params:
  images: configs/rxn0003.xyz
  path_params:
    name: mlp
    n_embed: 1
    depth: 2
    activation: gelu
  num_record_points: 101
  device: cuda
  seed: 0

optimization_params:
  - potential_params:
      name: repel
    integrator_params:
      path_integrand_names: geodesic
    optimizer_params:
      optimizer:
        name: adam
        lr: 1.0e-1
    num_optimizer_iterations: 1000

  - potential_params:
      name: uma
      model_name: uma-s-1p1
      task_name: omol
    integrator_params:
      path_integrand_names: pvre
      rtol: 1.0e-2
      atol: 1.0e-2
    optimizer_params:
      optimizer:
        name: adam
        lr: 1.0e-3
      threshold: 1.0e-1
    num_optimizer_iterations: 1000
```

Two legs: a fast repulsive geodesic pre-step (no MLIP, no convergence
trigger), then the UMA-driven step that does the real work.
