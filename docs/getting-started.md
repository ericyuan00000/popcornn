# Getting started

This page walks from a clean machine to a custom reaction.

## 1. Install

```bash
conda create --name popcornn python=3.12
conda activate popcornn

git clone https://github.com/khegazy/popcornn.git
pip install -e ./popcornn
```

`pip install -e` is the editable install — when you pull updates from
the repo, you don't need to reinstall.

If you want to run a machine-learned interatomic potential (MLIP), you
also need to install that potential's package — see
[Potentials](potentials.md).

## 2. Run a built-in example

Two analytic examples ship with the repo, both in 2D so they run on
CPU in seconds:

```bash
cd examples
python run.py --config configs/wolfe.yaml
python run.py --config configs/muller_brown.yaml
```

A real-system example using the UMA MLIP also ships, but takes longer
and needs a GPU plus the UMA install:

```bash
python run.py --config configs/rxn0003.yaml
```

When `run.py` finishes on an **atomistic** input (anything that came
from an ASE `Atoms` list — including the `rxn0003` example) it writes:

- `popcornn.xyz` — the optimized path as a sequence of frames.
- `popcornn_ts.xyz` — the single frame at the predicted transition
  state.

You can open both in any standard molecular viewer (ASE GUI, VMD,
Avogadro, OVITO).

The 2D toy examples (`wolfe.yaml`, `muller_brown.yaml`) don't write
xyz — `run.py` only emits xyz when the input is `Atoms`. They still
return the optimized path through the Python API, so they're useful
for sanity-checking the optimization machinery.

## 3. Run on your own reaction

The shortest possible custom config:

```yaml
initialization_params:
  images: my_reaction.xyz       # at least the reactant and product
  path_params:
    name: mlp                   # neural-network path
    n_embed: 1
    depth: 2

optimization_params:
  - potential_params:
      name: uma                 # or mace, orb, leftnet, ...
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

Point `images:` at an `xyz` or `traj` file. The first frame is treated
as the reactant, the last frame as the product. Any frames in between
are intermediate guesses — popcornn will fit the path through them but
won't pin them.

Before you hand atoms to popcornn:

- The reactant and product must use the **same atom ordering**. Atom 1
  in the reactant is the same atom as atom 1 in the product.
- The two structures must be **rotationally and translationally
  aligned**. The path doesn't include rigid-body motion — it's
  configuration-space only.
- For periodic systems, the product should be **unwrapped** with
  respect to the reactant. By default popcornn does this for you using
  the minimum-image convention, but if any atom moves more than half a
  cell during the reaction, unwrap manually and pass
  `unwrap_positions: false` in `path_params`.

## 4. Use the Python API directly

The `run.py` script is a thin wrapper. If you want full control,
import the `Popcornn` class:

```python
from ase.io import read, write
from popcornn import Popcornn

images = read("my_reaction.xyz", index=":")
for image in images:
    image.info = {"charge": 0, "spin": 1}  # if your MLIP needs it

mep = Popcornn(
    images=images,
    path_params={"name": "mlp", "n_embed": 1, "depth": 2},
)

final_images, ts_image = mep.optimize_path(
    {
        "potential_params": {"potential": "repel"},
        "integrator_params": {"path_integrand_names": "geodesic"},
        "optimizer_params": {"optimizer": {"name": "adam", "lr": 1.0e-1}},
        "num_optimizer_iterations": 1000,
    },
    {
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

write("popcornn.xyz", final_images)
write("popcornn_ts.xyz", ts_image)
```

This is the same recipe the YAML driver uses. Each dict you pass to
`optimize_path` is one **leg** of optimization — you can chain as many
as you like. The example above runs a cheap repulsive pre-step
(geodesic interpolation, fixes atom clashes) before the expensive
MLIP-driven step.

## 5. What happens during optimization

Each leg:

1. Picks up the path from where the previous leg left it.
2. Picks up the potential and the loss you specified.
3. Repeatedly evaluates a path integral of the loss along the path,
   gets a gradient with respect to the path's neural-network
   parameters, and steps Adam.
4. Stops when the integrated gradient norm has been below `threshold`
   for `patience` consecutive iterations, or when
   `num_optimizer_iterations` is reached.

For more on the convergence criterion and how to tune it, see
[Convergence](convergence.md).

## 6. Where to go next

- [Concepts](concepts.md) for the physics background.
- [Configuration reference](configuration.md) for every key in the
  YAML config.
- [Potentials](potentials.md) to install a particular MLIP.
- [Memory & OOM](memory-and-oom.md) if your run crashes with CUDA OOM.
