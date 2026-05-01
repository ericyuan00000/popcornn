# Loss functions

Popcornn optimizes by integrating a per-point quantity along the path
and minimizing the result:

$$\text{loss}(\theta) = \int_0^1 \ell\big(x(t; \theta)\big) \, \mathrm{d}t$$

The per-point quantity $\ell$ is whatever you put in
`integrator_params.path_integrand_names`. You can pass a single name (the
common case) or a list of names with `path_integrand_scales` weights (for
weighted combinations).

```yaml
integrator_params:
  path_integrand_names: pvre
  rtol: 1.0e-2
  atol: 1.0e-2
```

```yaml
integrator_params:
  path_integrand_names: ['pvre', 'vre']
  path_integrand_scales: [1.0, 0.1]
```

## Available terms

All terms below take the path-evaluated quantities — positions $x$,
velocities $\mathbf{v} = \dot{x}$, energy $E$, forces $\mathbf{F}$ —
as needed. Popcornn fetches whichever fields are required and reuses
cached evaluations when it can.

### `pvre` (pVRE)

$$\ell_{\text{pVRE}} = \big| \mathbf{v}(t) \cdot \mathbf{F}(t) \big|$$

Drives configurations where the force is **perpendicular** to the
path direction — the saddle-point condition. **This is the default
for reaction-path optimization.**

### `pvre_squared` (pVRE²)

$$\ell = \big( \mathbf{v}(t) \cdot \mathbf{F}(t) \big)^2$$

Same saddle-point physics as `pvre` (zero iff $\mathbf{v} \perp
\mathbf{F}$), but the integrand is $C^\infty$-smooth in $t$. The plain
`pvre` integrand has a kink wherever $\mathbf{v}\cdot\mathbf{F}$
crosses zero — i.e. exactly the points the loss is trying to reach —
so its gradient $\partial\mathcal{L}/\partial\theta$ has jump
discontinuities along the path. Adaptive Gauss–Kronrod quadrature has
to refine indefinitely around each crossing, which is the dominant
cost of an iteration. Squaring removes the kink and gk21 typically
converges in one pass.

In practice `pvre_squared`'s gradient $\propto 2(\mathbf{v}\!\cdot\!\mathbf{F})$
vanishes near the saddle ridge, so it drives the path most of the way
to the MEP cheaply but plateaus before pinpointing the TS. Pair it with
`pvre` as a second leg for a smooth-then-sharp schedule —
`examples/configs/muller_brown.yaml` ships this pattern and is ~3.5×
faster than single-leg `pvre`. See [Advanced](advanced.md) for the
multi-leg recipe.

### `pvre_mag`

$$\ell = \big\| \mathbf{v}(t) \odot \mathbf{F}(t) \big\|_2$$

Per-component product, then norm. A geometry-aware variant of
pVRE.

### `vre` (VRE)

$$\ell_{\text{VRE}} = \|\mathbf{F}\|_2 \cdot \|\mathbf{v}\|_2$$

A magnitude-only product. Used in combination with pVRE so that the
difference $\ell_{\text{VRE}} - \ell_{\text{pVRE}}$ is a soft penalty
on the angle between force and velocity (zero when they're parallel).

### `vre_error`

$$\ell = \ell_{\text{VRE}} - \ell_{\text{pVRE}}$$

Force-velocity angular mismatch. Approaches zero on a true MEP (where
forces are tangent to the path).

### `geodesic`

$$\ell_{\text{geo}} = \big\| \mathbf{F}_{\text{decomposed}} \cdot \mathbf{v} \big\|_2$$

Path-length in a force-decomposed metric. With
`potential_params.name: repel`, this is the geodesic interpolation of
[Zhu et al. 2019](https://pubs.aip.org/aip/jcp/article/150/16/164103/198363/Geodesic-interpolation-for-reaction-pathways).
Use it as a pre-step to fix atom clashes.

### `F_mag`

$$\ell = \|\mathbf{F}\|_2$$

Force magnitude. Useful as a transition-state region loss.

### `E`

$$\ell = E(x(t))$$

Raw energy. Integrating energy along the path gives a (poor) measure
of "how steep" the path is.

### `E_mean`

$$\ell = \overline{E}$$

Mean energy across whatever batched dimension the integrator hands
back. Sometimes useful as a TS-time loss.

## Picking a loss

For a typical reaction:

| What you want | Loss |
| --- | --- |
| Resolve atom clashes (pre-step) | `geodesic` with `potential_params.name: repel` |
| Find the minimum-energy path | `pvre`, or a `pvre_squared → pvre` schedule for ~3.5× speedup (see `examples/configs/muller_brown.yaml`) |
| Find the path *and* keep it short | combine pVRE + VRE with scales (see `examples/configs/loss_example.yaml`) |
| Maximize the TS energy | apply `E_mean` as a TS-region loss (see [Advanced](advanced.md)) |
| Minimize the TS force magnitude | apply `F_mag` as a TS-time loss |

If you're not sure, start with pVRE — alone, or as the second stage
of a `pvre_squared → pvre` schedule when integration cost matters.
It's the recommended default for the saddle-finding step.

## Combining terms with schedulers

The `loss_example.yaml` config ramps the geodesic-style term down and
the pVRE term up over the first 100 iterations using cosine
schedulers, so the path first untangles itself and then targets the
TS. See [Advanced](advanced.md) for the syntax.
