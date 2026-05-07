"""UMA smoke test for the gradient-of-loss integrand pipeline.

Run via NERSC interactive node:
    srun -A m2834 -q interactive -C gpu -t 15:00 --exclusive --ntasks=1 \
        python tests_ongoing/test_uma_smoke.py
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from torchpathint import path_integral
from popcornn import Popcornn
from popcornn.tools import import_run_config, PathIntegrator, evaluate_integrand_sum
from popcornn.optimization.optimizer import PathOptimizer
from popcornn.potentials import get_potential


REPO_ROOT = '/global/homes/e/ericyuan/GitHub/Popcornn'


def main():
    os.chdir(os.path.join(REPO_ROOT, 'examples'))
    config = import_run_config('configs/rxn0003.yaml')
    mep = Popcornn(**config.get('initialization_params', {}))
    uma_leg = config.get('optimization_params', [])[1]
    uma_leg['potential_params']['model_name'] = 'uma-s-1p2'
    # Loose tolerance — this is a smoke test, not a convergence study.
    # Chunking is handled automatically by torchpathint's OOM-shrink path;
    # nothing to set here.
    uma_leg['integrator_params']['rtol'] = 1e-2
    uma_leg['integrator_params']['atol'] = 1e-2

    pot = get_potential(images=mep.images, **uma_leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**uma_leg['integrator_params'],
                          device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **uma_leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    def measure_L():
        def fval(t_flat):
            l, _ = evaluate_integrand_sum(
                integ._terms, t_flat.unsqueeze(-1), mep.path,
            )
            return l.reshape(t_flat.shape[0], -1).detach()
        out = path_integral(
            fval,
            torch.tensor(0., dtype=mep.dtype, device=mep.device),
            torch.tensor(1., dtype=mep.dtype, device=mep.device),
            method='gl15', dtype=mep.dtype, device=mep.device,
        )
        return out.integral[0].item()

    print(f'L0  = {measure_L():.6e}')
    for it in range(20):
        out = optr.optimization_step(mep.path, integ)
        if it % 2 == 0 or it == 19:
            print(f'iter {it:3d}  ||grad||={out.grad_norm.item():.3e}  '
                  f'L={measure_L():.6e}  rgrad={out.grad_integral.requires_grad}')


if __name__ == '__main__':
    main()
