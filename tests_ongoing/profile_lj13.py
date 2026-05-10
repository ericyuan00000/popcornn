"""Profile one optimizer step of LJ-13 stage 1 to confirm where time goes.

Hypothesis (see plans/what-is-the-slowest-effervescent-flame.md):
the dominant cost is the per-sample Jacobian via
torch.autograd.grad(grad_outputs=eye(N), is_grads_batched=True)
inside PathIntegrator.integrate_path. Confirm by reading top ops
sorted by self-CUDA time and self-CPU time.

Also reports per-step wall time so we have a baseline to compare against
post-refactor.
"""
import argparse
import copy
import time

import torch

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config


def build(stage_idx):
    cfg = import_run_config("examples/configs/lj13.yaml")
    init = cfg["initialization_params"]
    init.pop("output_dir", None)
    mep = Popcornn(**init)

    leg = copy.deepcopy(cfg["optimization_params"][stage_idx])
    pot = get_potential(images=mep.images, **leg["potential_params"],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg["integrator_params"],
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg["optimizer_params"],
                         device=mep.device, dtype=mep.dtype)
    return mep, integ, optr


def time_steps(mep, integ, optr, n_warmup, n_runs):
    for _ in range(n_warmup):
        optr.optimization_step(mep.path, integ)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        optr.optimization_step(mep.path, integ)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs


def profile_steps(mep, integ, optr, n_runs):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=False, with_stack=False,
    ) as prof:
        for _ in range(n_runs):
            optr.optimization_step(mep.path, integ)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0,
                    help="0=pvre_squared (smooth), 1=pvre (kinky)")
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-time", type=int, default=10,
                    help="iters timed without profiler overhead")
    ap.add_argument("--n-prof", type=int, default=5,
                    help="iters captured under torch.profiler")
    ap.add_argument("--rows", type=int, default=25)
    args = ap.parse_args()

    mep, integ, optr = build(args.stage)
    n_params = sum(p.numel() for p in mep.path.parameters())
    print(f"# device={mep.device} dtype={mep.dtype} stage={args.stage} "
          f"D={n_params}")
    if torch.cuda.is_available():
        print(f"# GPU={torch.cuda.get_device_name()}")

    # Step 1: clean wall-time baseline (no profiler overhead).
    ms = time_steps(mep, integ, optr, args.n_warmup, args.n_time) * 1e3
    print(f"\nClean wall time per optimizer step: {ms:.2f} ms "
          f"(avg over {args.n_time} iters, {args.n_warmup} warmup)")

    # Step 2: profiler trace.
    prof = profile_steps(mep, integ, optr, args.n_prof)

    sort_keys = ["self_cuda_time_total", "self_cpu_time_total"]
    if not torch.cuda.is_available():
        sort_keys = ["self_cpu_time_total"]
    for key in sort_keys:
        print(f"\n=== top {args.rows} by {key} ===")
        print(prof.key_averages().table(sort_by=key, row_limit=args.rows))


if __name__ == "__main__":
    main()
