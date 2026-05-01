"""Tests for ``PathIntegrator.save_samples=True``.

Exercises the side-buffer + byte-keyed stitch that lets the
transition-state finder consume the integrator's quadrature samples
without paying for any extra path forwards.
"""

import pytest
import torch

from popcornn.paths import get_path
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, SamplesCache, process_images


@pytest.fixture
def muller_brown_setup():
    torch.manual_seed(0)
    device = torch.device('cpu')
    dtype = torch.float64
    images = process_images(
        'tests/images/muller_brown.json', device=device, dtype=dtype
    )
    path = get_path(
        'mlp', images=images, n_embed=4, depth=2,
        activation='gelu', device=device, dtype=dtype,
    )
    potential = get_potential(
        'muller_brown', images=images, device=device, dtype=dtype,
    )
    path.set_potential(potential)
    return path, device, dtype


def test_save_samples_aligned_with_quadrature_mesh(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        save_samples=True,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)

    assert isinstance(out.samples, SamplesCache)
    expected_n = out.t.flatten().shape[0]
    assert out.samples.time.shape == (expected_n,)
    # forces are flattened atomic dof; for Muller-Brown D = 2.
    assert out.samples.forces.shape[0] == expected_n
    assert out.samples.forces.shape[1] == 2
    # energies may be shape [N, 1] or [N, K]; just assert leading axis.
    assert out.samples.energies.shape[0] == expected_n

    # sample times round-trip to the integrator's accepted mesh exactly.
    assert torch.allclose(
        out.samples.time, out.t.flatten().to(out.samples.time.device)
    )
    # nothing nan / inf — energies and forces actually came from the potential.
    assert torch.isfinite(out.samples.energies).all()
    assert torch.isfinite(out.samples.forces).all()


def test_save_samples_off_yields_none(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        save_samples=False,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)
    assert out.samples is None
