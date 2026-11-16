"""Physics validation against the analytic infinite square well."""

import numpy as np

from qw.constants import KINETIC_FACTOR
from qw.grid import make_grid
from qw.solver import solve_schrodinger


def infinite_well_energies(
    width,
    mass,
    n_states,
):
    """Return analytic energies for a 1D infinite square well."""

    quantum_numbers = np.arange(
        1,
        n_states + 1,
        dtype=float,
    )

    return (
        KINETIC_FACTOR
        / mass
        * (
            quantum_numbers
            * np.pi
            / width
        ) ** 2
    )


def test_numerical_infinite_well_matches_analytic_energies():
    """The numerical Dirichlet box should reproduce the analytic spectrum."""

    width = 100.0
    mass = 0.067
    n_states = 4

    z, dz = make_grid(
        z_min=0.0,
        z_max=width,
        n_grid_points=2001,
    )

    potential = np.zeros_like(z)

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=potential,
        electric_field=0.0,
        charge=-1.0,
        n_states=n_states,
        d2_stencil_points=3,
        solver="tridiagonal",
    )

    expected = infinite_well_energies(
        width=width,
        mass=mass,
        n_states=n_states,
    )

    absolute_error = np.abs(result.energies - expected)
    relative_error = absolute_error / expected

    print("\nInfinite square-well validation")
    print("------------------------------------------")
    print(f"Width: {width:.1f} Angstrom")
    print(f"Effective mass: {mass:.6f} m0")
    print(f"Grid spacing: {dz:.6f} Angstrom")
    print("\nState   Analytic (eV)   Numerical (eV)   Rel. error")

    for state in range(n_states):
        print(
            f"{state:5d}   "
            f"{expected[state]: .10f}   "
            f"{result.energies[state]: .10f}   "
            f"{relative_error[state]:.3e}"
        )

    np.testing.assert_allclose(
        result.energies,
        expected,
        rtol=5.0e-6,
        atol=1.0e-10,
    )


def test_infinite_well_energy_ratios_follow_n_squared():
    """Infinite-well energies should scale as n^2."""

    width = 100.0
    mass = 0.067

    z, _ = make_grid(
        z_min=0.0,
        z_max=width,
        n_grid_points=1601,
    )

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=np.zeros_like(z),
        n_states=4,
        d2_stencil_points=3,
        solver="tridiagonal",
    )

    numerical_ratios = (
        result.energies
        / result.energies[0]
    )

    expected_ratios = np.array(
        [1.0, 4.0, 9.0, 16.0]
    )

    print("\nInfinite-well E_n / E_1 ratios")
    print("------------------------------------------")
    for state, (numerical, expected) in enumerate(
        zip(numerical_ratios, expected_ratios),
        start=1,
    ):
        print(
            f"n={state}: numerical={numerical:.8f}, "
            f"expected={expected:.8f}"
        )

    np.testing.assert_allclose(
        numerical_ratios,
        expected_ratios,
        rtol=1.0e-5,
        atol=1.0e-8,
    )
