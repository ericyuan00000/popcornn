# Convergence

Each `optimization_params` leg exits when one of two things happens:

1. The L∞ norm of the path-integrated gradient
   $\big\| \int \nabla_\theta \mathcal{L} \, \mathrm{d}t \big\|_\infty$
   stays below `threshold` for `patience` consecutive iterations.
2. `num_optimizer_iterations` is reached.

Whichever fires first.

## What gets compared to `threshold`

Popcornn integrates the **gradient** of the loss with respect to the
path's neural-network parameters $\theta$ along the path. The result
is a vector of the same shape as $\theta$. The convergence check uses
its L∞ (per-component max) norm:

$$g_\infty = \max_i \left| \int_0^1 \frac{\partial \mathcal{L}}{\partial \theta_i}\, \mathrm{d}t \right|$$

Why L∞ rather than L2? The L2 norm scales with $\sqrt{D}$ where $D$
is the parameter count, so the same `threshold` doesn't transfer
across path-network sizes. L∞ is closer to size-independent.

## How to pick `threshold`

`threshold` is **system-dependent**. Gradient magnitudes differ by
orders of magnitude between toy potentials and real MLIP-driven runs,
and even between MLIPs.

The recipe:

1. **Pilot run.** Set `threshold: null` (or just omit it) and run
   for, say, 50 iterations. Watch the per-iteration $g_\infty$ in
   the integrator output.
2. **Read the early value.** Look at iterations ~5–20. This is your
   "starting" gradient norm before the optimizer has had time to
   make progress.
3. **Drop one order of magnitude.** Set `threshold` to roughly that
   early value divided by 10.

| System | Initial $g_\infty$ | Reasonable `threshold` |
| --- | --- | --- |
| Wolfe (2D analytic) | ~20 | `1.0` |
| UMA-driven `gg3` | ~1.5 | `1.0e-3` |
| Müller–Brown | ~10 | `1.0e-1` |
| LJ-13 cluster | ~75 | `1.0e-3` |

The shipped recipes use a single **on-rule** `(atol, threshold) =
(thr/10, thr)` pair so the gradient noise floor sits an order of
magnitude below the trigger. Stage-2 chemistry recipes all ship at the
same numbers: pseudo-Huber + `atol=1e-4, thr=1e-3`.

- `wolfe.yaml` — `threshold: 1.0`.
- `gg3.yaml` — two-stage. Warm-up (repel + geodesic) uses
  `(rtol, atol, threshold) = (1e-1, 1e-4, 1e-3)`, `patience: 1`;
  UMA stage 2 uses pseudo-Huber δ=0.1 at the same on-rule pair
  `(rtol, atol, threshold) = (1e-1, 1e-4, 1e-3)`.
- `muller_brown.yaml` — single-stage pseudo-Huber δ=1.0 + n4d2 +
  lr=1e-3 + `(rtol, atol, threshold) = (1e-1, 1e-2, 1e-1)`,
  `patience: 1`. Same `atol/thr = 0.1` ratio at MB's higher force
  scale (δ scaled 10× from chemistry's δ=0.1).
- `lj13.yaml` — single-stage pseudo-Huber δ=0.1 + n4d2 + lr=1e-3 +
  `(rtol, atol, threshold) = (1e-1, 1e-4, 1e-3)`, `patience: 1`.
  Same loss family and on-rule pair as `gg3.yaml` stage 2.

Beyond pilot-and-divide, the noise-floor rule deserves its own note:
when the optimizer's `threshold` won't fire on a sweep that the loss
clearly converges, the issue is usually that the integrator's atol
sets a $g_\infty$ floor too close to (or above) `threshold`. Setting
`atol = threshold / 10` (with `rtol` low enough that
`rtol · g_typical_at_stop` is also `≤ threshold / 10`) is the
mechanical fix. See the docstrings in
`tests_ongoing/sweep_mb_lr1em3_tol_thr.py` /
`sweep_lj13_n4d2_lr1em3.py` for the empirical sweep that derived
the shipped (rtol, atol, threshold) triples.

The pilot-and-divide recipe applies per stage: each leg gets its own
threshold from its own initial $g_\infty$. `pvre_squared` gradients are
roughly $2|v\!\cdot\!F|$ times larger than `pvre` gradients, so a
`pvre_squared` warm-up stage typically needs a threshold ~10³× larger
than a `pvre` fine-tune stage on the same system.

## How to pick `patience`

Default is `5`. Adam exhibits a damped-oscillation phase as it
settles, and adaptive quadrature adds its own stochastic wiggle on
top. `patience` absorbs single-iteration dips below `threshold` so
the trigger only fires when the loss has actually flattened.

Override only if you have a specific reason — usually you don't.

- **Lower (1–3)** if you need fast turnaround and don't mind
  occasional false positives.
- **Higher (10+)** for noisy MLIPs where the gradient norm wobbles a
  lot near convergence.

## Disabling the trigger

`threshold: null` (the default) skips the convergence check entirely
and always runs the full `num_optimizer_iterations`. Use this for:

- The initial pilot pass (you don't yet know the gradient scale).
- Cheap legs where you want a fixed-iteration budget.

## Monitoring the loss itself

Convergence is driven by the gradient norm, not the loss value. If
you also want to watch the loss integral $\int \mathcal{L}\,\mathrm{d}t$
per iteration (for plotting or human sanity-checking), set:

```yaml
integrator_params:
  track_loss: true
```

That runs a separate detached quadrature with looser tolerances
(`loss_rtol`, `loss_atol`, defaulting to `rtol`/`atol`) so it doesn't
dominate runtime. The result lands on `integral_output.loss`.
