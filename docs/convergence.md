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
| UMA-driven `rxn0003` | ~1.5 | `1.0e-1` |
| Müller–Brown | ~10 | `1.0` |
| LJ-13 cluster | ~6.2e+2 (stage 1) | `null` — see note below |

The shipped `examples/configs/wolfe.yaml` uses `threshold: 1.0`;
`rxn0003.yaml` uses `threshold: 1.0e-1`. `muller_brown.yaml` ships a
two-stage `pvre_squared → pvre` schedule (see [Advanced](advanced.md)
for the pattern) with thresholds `1.0e+3` for the warm-up stage
(initial $g_\infty \approx 3.6\!\times\!10^4$) and `1.0` for the
fine-tune stage (initial $g_\infty \approx 5$ once warm-started).
These are calibrated to their respective gradient scales, not chosen
by guessing.

`lj13.yaml` deliberately ships with both stages' `threshold: null`
because the recipe above breaks on this system in two ways:

- **Stage 1 ($g_\infty$ decays too steeply.)** Stage-1
  $g_\infty$ drops from ~6.2e+02 to ~58 in the first 5 iters (one OOM
  in 5 steps), then continues falling for another two OOMs while the
  loss is still descending. The "iterations 5–20" reading would set
  `threshold` ~6.0, which fires around step 51 with the loss still
  ~16% above its eventual floor.
- **Stage 2 ($g_\infty$ is a poor proxy for path quality.)** Across
  three seeds, stage-2 $g_\infty$ tightens 5 OOM (from ~1.1e-1 to
  ~6e-6) while the perpendicular force at the saddle, $|F_\perp|_\mathrm{TS}$,
  only tightens ~5x (from ~0.11 to ~0.021). Early-stopping on
  $g_\infty$ alone risks ending with loose path geometry even though
  the gradient norm has nominally converged. The benefit of stage 2
  on LJ-13 is the 10x reduction in $|F_\perp|_\mathrm{TS}$ relative to
  a stage-1-only run; preserving that benefit requires running stage 2
  to its full iteration budget.

The takeaway: the 1-OOM-below-initial recipe assumes (a) $g_\infty$
decays at a rate comparable to the loss, and (b) $g_\infty$ correlates
with the physically-meaningful quality metric. When either assumption
breaks, fall back to fixed iteration counts and verify quality with a
path-intrinsic metric such as $|F_\perp|_\mathrm{TS}$.

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
dominate runtime. The result lands on `path_integral.loss_integral`.
