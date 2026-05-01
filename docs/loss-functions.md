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
  path_integrand_names: ['pvre', 'variable_reaction_energy']
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

### `projected_variational_reaction_energy_mag`

$$\ell = \big\| \mathbf{v}(t) \odot \mathbf{F}(t) \big\|_2$$

Per-component product, then norm. A geometry-aware variant of
pVRE.

### `variable_reaction_energy` (VRE)

$$\ell_{\text{VRE}} = \|\mathbf{F}\|_2 \cdot \|\mathbf{v}\|_2$$

A magnitude-only product. Used in combination with pVRE so that the
difference $\ell_{\text{VRE}} - \ell_{\text{pVRE}}$ is a soft penalty
on the angle between force and velocity (zero when they're parallel).

### `vre_variational_error`

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
| Find the minimum-energy path | `pvre` |
| Find the path *and* keep it short | combine pVRE + VRE with scales (see `examples/configs/loss_example.yaml`) |
| Maximize the TS energy | apply `E_mean` as a TS-region loss (see [Advanced](advanced.md)) |
| Minimize the TS force magnitude | apply `F_mag` as a TS-time loss |

If you're not sure, start with pVRE alone. It's the
recommended default and is what every example except `loss_example`
uses.

## Combining terms with schedulers

The `loss_example.yaml` config ramps the geodesic-style term down and
the pVRE term up over the first 100 iterations using cosine
schedulers, so the path first untangles itself and then targets the
TS. See [Advanced](advanced.md) for the syntax.
