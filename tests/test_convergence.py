"""Numerical convergence tests for the finite-difference solver."""

import numpy as np
import pytest

from qw.constants import KINETIC_FACTOR
from qw.grid import make_grid
from qw.solver import solve_schrodinger


def analytic_ground_state_energy(
    width,
    mass,
):
    """Ground-state energy of a 1D infinite square well."""

    return (
        KINETIC_FACTOR
        / mass
        * (np.pi / width) ** 2
    )


def numerical_ground_state_energy(
    width,
    mass,
    n_grid_points,
    d2_stencil_points,
):
    """Calculate the numerical infinite-well ground-state energy."""

    z, dz = make_grid(
        z_min=0.0,
        z_max=width,
        n_grid_points=n_grid_points,
    )

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=np.zeros_like(z),
        n_states=1,
        d2_stencil_points=d2_stencil_points,
        solver="auto",
    )

    return result.energies[0], dz


def test_three_point_grid_refinement_reduces_error():
    """Refining dz should reduce the 3-point finite-difference error."""

    width = 100.0
    mass = 0.067

    exact = analytic_ground_state_energy(
        width=width,
        mass=mass,
    )

    results = []

    for n_grid_points in [101, 201, 401]:
        energy, dz = numerical_ground_state_energy(
            width=width,
            mass=mass,
            n_grid_points=n_grid_points,
            d2_stencil_points=3,
        )

        error = abs(energy - exact)

        results.append(
            (
                n_grid_points,
                dz,
                energy,
                error,
            )
        )

    print("\n3-point finite-difference convergence")
    print("------------------------------------------")
    print(f"Analytic E0 = {exact:.12f} eV")
    print("Grid points   dz (A)       E0 (eV)         Abs. error (eV)")

    for n_grid_points, dz, energy, error in results:
        print(
            f"{n_grid_points:11d}   "
            f"{dz:8.5f}   "
            f"{energy:.12f}   "
            f"{error:.3e}"
        )

    error_coarse = results[0][3]
    error_medium = results[1][3]
    error_fine = results[2][3]

    ratio_coarse_medium = error_coarse / error_medium
    ratio_medium_fine = error_medium / error_fine

    print(
        f"Error reduction 101 -> 201 points: "
        f"{ratio_coarse_medium:.4f}"
    )
    print(
        f"Error reduction 201 -> 401 points: "
        f"{ratio_medium_fine:.4f}"
    )

    assert error_medium < error_coarse
    assert error_fine < error_medium

    # A centered 3-point second derivative has O(dz^2) error.
    # Halving dz should therefore reduce the error by about 4.
    assert ratio_coarse_medium == pytest.approx(
        4.0,
        rel=0.05,
    )

    assert ratio_medium_fine == pytest.approx(
        4.0,
        rel=0.05,
    )


@pytest.mark.parametrize(
    "d2_stencil_points",
    [3, 5, 7],
)
def test_each_stencil_converges_under_grid_refinement(
    d2_stencil_points,
):
    """Each supported stencil should approach the analytic energy as dz decreases."""

    width = 100.0
    mass = 0.067

    exact = analytic_ground_state_energy(
        width=width,
        mass=mass,
    )

    coarse, dz_coarse = numerical_ground_state_energy(
        width=width,
        mass=mass,
        n_grid_points=101,
        d2_stencil_points=d2_stencil_points,
    )

    fine, dz_fine = numerical_ground_state_energy(
        width=width,
        mass=mass,
        n_grid_points=401,
        d2_stencil_points=d2_stencil_points,
    )

    error_coarse = abs(coarse - exact)
    error_fine = abs(fine - exact)

    print(
        f"\n{d2_stencil_points}-point stencil: "
        f"dz {dz_coarse:.4f} -> {dz_fine:.4f} A, "
        f"error {error_coarse:.3e} -> {error_fine:.3e} eV"
    )

    assert error_fine < error_coarse
