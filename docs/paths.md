# Paths

A *path* in popcornn is a smooth, differentiable mapping from a scalar
time $t \in [0, 1]$ to a configuration $x(t)$, with $x(0) = $ reactant
and $x(1) = $ product.

You select a path representation with `path_params.name`:

```yaml
initialization_params:
  path_params:
    name: mlp        # or 'linear'
    n_embed: 1
    depth: 2
    activation: gelu
```

## Available representations

### `linear`

The simplest possible path:

$$x(t) = x_0 + t \, (x_1 - x_0)$$

A straight line in configuration space between reactant ($x_0$) and
product ($x_1$). Useful only as a starting reference — it doesn't
have any trainable parameters, so there's nothing to optimize.

### `mlp` (default)

A small multi-layer perceptron $\Delta(t; \theta)$ added on top of
the linear path:

$$x(t) = \underbrace{x_0 + t\,(x_1 - x_0)}_{\text{linear base}}
       + \underbrace{(1 - t)\, t \cdot \Delta(t; \theta)}_{\text{MLP correction}}$$

The factor $(1 - t)\, t$ vanishes at $t = 0$ and $t = 1$, so the path
is **pinned** to the reactant and product at the endpoints regardless
of what the MLP outputs. The MLP only ever moves intermediate points.

| Key | Default | Effect |
| --- | --- | --- |
| `n_embed` | `1` | Width multiplier on the hidden layers. Larger = more expressive. |
| `depth` | `2` | Number of layers. `depth=2` is one input layer + one output layer (no hidden). `depth=4` adds two hidden layers. |
| `activation` | `"gelu"` | Any nonlinearity in `torch.nn` (case-insensitive). |

For simple reactions, `depth: 2, n_embed: 1` is enough. For more
complicated reactions (concerted multi-bond rearrangements, large
configurational changes), bump `depth` up; `depth: 4, n_embed: 8` is
what the `wolfe.yaml` and `loss_example.yaml` configs use.

## Adding your own path representation

Subclass `BasePath` and implement `get_positions(time)`:

```python
from popcornn.paths.base_path import BasePath

class MyPath(BasePath):
    def __init__(self, my_param=1.0, **kwargs):
        super().__init__(**kwargs)
        # ... build whatever trainable params you need
        # use self.initial_position and self.final_position from BasePath

    def get_positions(self, time):
        # time has shape [N, 1] in [0, 1].
        # Return positions of shape [N, D] where D = len(initial_position).
        # Make sure x(0) = self.initial_position and x(1) = self.final_position.
        ...
```

Then register in `popcornn/paths/__init__.py`:

```python
path_dict = {
    "mlp" : MLPpath,
    "linear" : LinearPath,
    "mine" : MyPath,         # <--
}
```

`BasePath` handles the rest: velocity computation via autograd,
output reshaping, periodic-boundary wrapping, fixed-atom masking, and
the `forward` interface popcornn calls.

## Why parameterize the path?

The neural-network path is what makes popcornn end-to-end
differentiable. Once you have a function $x(t; \theta)$, you can:

- Evaluate $x$ at any $t$ — no fixed mesh, no interpolation between
  images.
- Compute $\dot{x}(t)$ analytically via autograd.
- Define a loss that's an integral over $t$ and backpropagate to
  $\theta$ in one shot.

Methods like NEB or string method instead carry around a finite chain
of replicas and have to project forces along/perpendicular to the
chain. They work, but they're discrete and stiff. Popcornn's
continuous representation just trains.

See [Concepts](concepts.md) for more on why this matters.
