# Memory and OOM errors

Popcornn evaluates the path on an **adaptive mesh** of times — denser
where the loss is changing fast, sparser where it's flat. The number
of points in a single quadrature step is therefore not fixed; it
varies from one call to the next, and on big systems with MLIPs, that
variability is what triggers most CUDA out-of-memory errors.

`popcornn/popcornn.py` already sets

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

at import time, which solves most OOM cases. If you're still hitting
OOM, work through the steps below in order.

## 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

PyTorch's default memory allocator is optimized for one batch size
seen many times. The adaptive integrator doesn't behave that way: it
might evaluate 16 points one iteration, 64 the next. Without
expandable segments, PyTorch fragments memory across these resizes
and runs out even though the live working set fits.

`popcornn` sets this environment variable at the top of
`popcornn/popcornn.py`, so it's already on for the YAML driver and
for any code that does `from popcornn import Popcornn` early. If
you're using popcornn as a library and importing other things first,
make sure to set it before any CUDA allocator calls:

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

at the very top of your entry-point file.

## 2. `total_mem_usage`

If you still hit OOM, the integrator's auto-batching is overestimating
how much memory you have free. Cap it manually:

```yaml
integrator_params:
  path_integrand_names: pvre
  rtol: 1.0e-2
  atol: 1.0e-2
  total_mem_usage: 0.75      # default is 0.9
```

`total_mem_usage` is the fraction of currently-free GPU memory the
adaptive batcher is allowed to fill on its next batch. Lower it
until OOMs stop. `0.75` and `0.5` are reasonable retries.

## 3. `max_batch`

If even `total_mem_usage: 0.5` still OOMs, hard-cap the batch:

```yaml
integrator_params:
  path_integrand_names: pvre
  max_batch: 32
```

This forces every quadrature call to evaluate at most 32 points.
Slower (more sequential calls), but bounded memory.

## 4. Float precision

Switching from `float32` to `float64` doubles the memory footprint,
so:

```yaml
initialization_params:
  dtype: float32       # default
```

is preferred unless you specifically need `float64` for numerical
sensitivity (you almost never do for path optimization).

## 5. Smaller path network

A bigger MLP doesn't just cost more compute — every parameter
contributes to the gradient that gets propagated through the
integrator. Drop `n_embed` and `depth` in `path_params` if memory is
tight:

```yaml
path_params:
  name: mlp
  n_embed: 1
  depth: 2
```

This is also the default and is enough for most reactions.

## 6. CPU offload

You can run the whole optimization on CPU by setting
`device: cpu` in `initialization_params`. Slow, but it never OOMs.
This is mostly only practical for the analytic toy potentials.

## What to do if it's still OOM

If you've worked through 1–6 and still see OOM, file a GitHub issue
with:

- The full traceback, including the actual CUDA allocator message.
- `nvidia-smi` output showing how much memory the GPU actually has.
- The full config file you're running.
- The popcornn and torchpathint commit hashes
  (`pip show popcornn torchpathint | grep -i 'name\|version'`).
