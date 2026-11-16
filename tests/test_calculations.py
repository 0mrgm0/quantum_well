"""Integration tests for high-level quantum-well calculations."""

import numpy as np

from qw.calculations import run_single_quantum_well


def test_run_single_quantum_well_returns_consistent_shapes():
    """The high-level calculation should return mutually consistent arrays."""

    n_grid_points = 301
    n_states = 3

    result = run_single_quantum_well(
        z_min=-150.0,
        z_max=150.0,
        n_grid_points=n_grid_points,
        well_width=100.0,
        well_depth=0.3,
        mass=0.067,
        electric_field=0.0,
        charge=-1.0,
        n_states=n_states,
        d2_stencil_points=3,
        solver="auto",
        well_center=0.0,
    )

    assert result.z.shape == (n_grid_points,)
    assert result.potential.shape == (n_grid_points,)
    assert result.field_potential.shape == (n_grid_points,)
    assert result.total_potential.shape == (n_grid_points,)
    assert result.wavefunctions.shape == (
        n_grid_points,
        n_states,
    )
    assert result.energies.shape == (n_states,)


def test_zero_field_total_potential_equals_well_potential():
    """At zero electric field, the total and confinement potentials coincide."""

    result = run_single_quantum_well(
        z_min=-150.0,
        z_max=150.0,
        n_grid_points=301,
        well_width=100.0,
        well_depth=0.3,
        mass=0.067,
        electric_field=0.0,
        charge=-1.0,
        n_states=3,
    )

    np.testing.assert_allclose(
        result.field_potential,
        0.0,
    )

    np.testing.assert_allclose(
        result.total_potential,
        result.potential,
    )


def test_finite_well_lowest_state_is_bound():
    """A sufficiently deep finite well should have a negative ground-state energy."""

    result = run_single_quantum_well(
        z_min=-200.0,
        z_max=200.0,
        n_grid_points=401,
        well_width=100.0,
        well_depth=0.3,
        mass=0.067,
        electric_field=0.0,
        charge=-1.0,
        n_states=1,
    )

    print(
        f"\nFinite-well ground-state energy: "
        f"{result.energies[0]:.10f} eV"
    )

    assert result.energies[0] < 0.0
