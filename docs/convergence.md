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
| LJ-13 cluster | ~6.2e+2 (stage 1) | `1.0` (stage 1), `1.0e-3` (stage 2) — see note below |

The shipped `examples/configs/wolfe.yaml` uses `threshold: 1.0`;
`rxn0003.yaml` uses `threshold: 1.0e-1`. `muller_brown.yaml` ships a
two-stage `pvre_squared → pvre` schedule (see [Advanced](advanced.md)
for the pattern) with thresholds `1.0e+3` for the warm-up stage
(initial $g_\infty \approx 3.6\!\times\!10^4$) and `1.0` for the
fine-tune stage (initial $g_\infty \approx 5$ once warm-started).
These are calibrated to their respective gradient scales, not chosen
by guessing.

`lj13.yaml`'s thresholds are calibrated differently because the
recipe above breaks on this system: stage-1 $g_\infty$ decays 3 OOM
in the first 25 iters (faster than the loss bends over) and stage-1
path quality is non-monotonic — under pvre_squared,
$|F_\perp|_\mathrm{TS}$ reaches a minimum of ~0.02 around iter 160
then **oscillates back up 44×** (to ~0.84) over the remaining
iterations as the optimizer sloshes around in pvre_squared's
$C^\infty$-flat basin near the saddle ridge.

The thresholds were derived from a 600+600-iter pilot (saved at
`tests_ongoing/run_lj13_traced.py`) instrumenting per-iter loss,
$g_\infty$, barrier, $|F|_\mathrm{TS}$, and $|F_\perp|_\mathrm{TS}$:

- **Stage 1 (`threshold: 1.0`)** fires at iter 84-124 across three
  seeds, near the $|F_\perp|_\mathrm{TS}$ minimum and *before* the
  late oscillation begins. This is structurally different from
  Müller-Brown's stage-1 threshold — there pvre_squared descent is
  monotonic, here it isn't.
- **Stage 2 (`threshold: 1.0e-3`)** fires at iter 81-99, where
  $|F_\perp|_\mathrm{TS}$ has settled within 1% of its 600-iter
  asymptote. Tighter thresholds (e.g., `1e-4`) waste compute without
  meaningfully improving path geometry.

3-seed validation: with these thresholds the example runs in ~110s
and reaches $|F_\perp|_\mathrm{TS} \approx 0.0016$ — **5× faster and
13× tighter** than the no-threshold 300+300 baseline (562s,
$|F_\perp|_\mathrm{TS} \approx 0.021$). The improvement comes from
stopping stage 1 before late oscillation degrades the path, so stage
2 starts from a cleaner warm-up.

The shipped path network is `n_embed=8, depth=6` (~400k params).
That choice came out of a 10-config (n_embed × depth) sweep with the
threshold-driven schedule — see [Advanced](advanced.md) for the
full result. The headline finding: **depth=4 is on a cliff**. At
depth=4 the same threshold trigger lands stage 2 in different
basins across seeds, so $|F_\perp|_\mathrm{TS}$ varies by an order
of magnitude (e.g. (8,4) seed-range [0.006, 0.031]). Depth=6
removes that variance — (8,6) sits in [0.0016, 0.0018] across the
same seeds. Larger MLPs also let stage 2's adaptive quadrature
finish in fewer evaluations per step (the path is smoother), so
(8,6) is *faster* than (8,4) despite having 2× the parameters.

The takeaway: when $g_\infty$ doesn't decay monotonically alongside
the loss, the early-iter "1-OOM-below-initial" reading misses the
real settle point. Run an instrumented pilot, plot all four metrics
(loss, $g_\infty$, barrier, $|F_\perp|_\mathrm{TS}$) together, and
read the threshold off the $g_\infty$ value at which the
quality-of-interest metric first stabilizes.

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
