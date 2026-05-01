import torch
from popcornn.tools import (
    PATH_INTEGRANDS,
    build_integrand_terms,
    evaluate_integrand_sum,
)


def test_single_integrand_outputs_match_per_class():
    """Single-term ``evaluate_integrand_sum`` reproduces the per-class
    ``PathIntegrand.evaluate`` output (sanity-checks the registry
    plumbing and the cache helper). Skips ``geodesic`` since it needs
    forces_decomposed which isn't part of this synthetic harness."""
    T = 100
    N_atoms = 17
    torch.manual_seed(0)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energies = torch.randn(T, N_atoms) * 50 + 1400
    velocities = torch.rand((T, N_atoms * 3)) * 5 + 3
    forces = torch.rand((T, N_atoms * 3)) * 20 + 10
    cache = {
        'time': time,
        'energies': energies,
        'velocities': velocities,
        'forces': forces,
    }

    for name, cls in PATH_INTEGRANDS.items():
        if name == 'geodesic':
            continue
        terms = build_integrand_terms([name])
        composite, _ = evaluate_integrand_sum(terms, time, path=None, cache=cache)
        direct = cls().evaluate(cache)
        assert torch.allclose(composite, direct), \
            f"{name}: composite sum diverged from direct evaluate"


def test_weighted_sum_matches_linear_combination():
    """Two-term weighted sum equals scale_a * a + scale_b * b."""
    T = 100
    N_atoms = 17
    torch.manual_seed(1)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energies = torch.randn(T, N_atoms) * 50 + 1400
    velocities = torch.rand((T, N_atoms * 3)) * 5 + 3
    forces = torch.rand((T, N_atoms * 3)) * 20 + 10
    cache = {
        'time': time,
        'energies': energies,
        'velocities': velocities,
        'forces': forces,
    }

    scales = [17.68, 11.45]
    pairs = [
        ('projected_variational_reaction_energy', 'variable_reaction_energy'),
        ('projected_variational_reaction_energy', 'F_mag'),
        ('E_mean', 'F_mag'),
    ]
    for name_a, name_b in pairs:
        terms = build_integrand_terms([name_a, name_b], scales)
        combined, _ = evaluate_integrand_sum(terms, time, path=None, cache=cache)
        direct = (
            scales[0] * PATH_INTEGRANDS[name_a]().evaluate(cache)
            + scales[1] * PATH_INTEGRANDS[name_b]().evaluate(cache)
        )
        assert torch.allclose(combined, direct), \
            f"weighted sum {name_a} + {name_b} did not match"


def test_also_resolve_returns_requested_fields():
    """``also_resolve`` forces extra fields into the resolved variables dict
    even when no integrand declares them in ``requires``."""
    T = 50
    N_atoms = 5
    torch.manual_seed(2)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energies = torch.randn(T, N_atoms)
    velocities = torch.rand(T, N_atoms * 3)
    forces = torch.rand(T, N_atoms * 3)
    cache = {
        'time': time,
        'energies': energies,
        'velocities': velocities,
        'forces': forces,
    }

    # F_mag only declares 'forces'; without also_resolve, energies wouldn't
    # be required. With it, the resolver still threads them through.
    terms = build_integrand_terms(['F_mag'])
    _, variables = evaluate_integrand_sum(
        terms, time, path=None, cache=cache, also_resolve=('energies',),
    )
    assert variables['energies'] is not None
    assert torch.equal(variables['energies'], energies)


def test_unknown_name_raises():
    try:
        build_integrand_terms(['not_a_real_integrand'])
    except ValueError as e:
        assert 'Unknown integrand' in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown integrand name")


def test_duplicate_name_raises():
    try:
        build_integrand_terms(['F_mag', 'F_mag'])
    except ValueError as e:
        assert 'twice' in str(e)
    else:
        raise AssertionError("Expected ValueError for duplicate integrand name")
