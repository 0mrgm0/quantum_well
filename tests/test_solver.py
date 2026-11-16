"""Tests for the finite-difference Schrodinger solver."""

import numpy as np
import pytest

from qw.grid import make_grid
from qw.solver import (
    build_hamiltonian,
    electric_field_potential,
    second_derivative_rule,
    solve_schrodinger,
)
from qw.well import square_well_potential


def make_symmetric_well_problem(
    n_grid_points=401,
    n_states=4,
    d2_stencil_points=3,
    solver="auto",
):
    """Return a standard zero-field symmetric-well calculation."""

    z, _ = make_grid(
        z_min=-200.0,
        z_max=200.0,
        n_grid_points=n_grid_points,
    )

    potential = square_well_potential(
        z=z,
        well_width=100.0,
        well_depth=0.3,
        center=0.0,
    )

    return solve_schrodinger(
        z=z,
        mass=0.067,
        potential=potential,
        electric_field=0.0,
        charge=-1.0,
        n_states=n_states,
        d2_stencil_points=d2_stencil_points,
        solver=solver,
    )


def test_three_point_second_derivative_rule():
    """The 3-point rule should reproduce the standard stencil."""

    offsets, coefficients = second_derivative_rule(3)

    np.testing.assert_array_equal(
        offsets,
        np.array([-1, 0, 1]),
    )

    np.testing.assert_allclose(
        coefficients,
        np.array([1.0, -2.0, 1.0]),
        atol=1.0e-14,
    )


def test_five_point_second_derivative_rule():
    """The 5-point rule should reproduce the standard stencil."""

    offsets, coefficients = second_derivative_rule(5)

    np.testing.assert_array_equal(
        offsets,
        np.array([-2, -1, 0, 1, 2]),
    )

    np.testing.assert_allclose(
        coefficients,
        np.array(
            [
                -1.0 / 12.0,
                4.0 / 3.0,
                -5.0 / 2.0,
                4.0 / 3.0,
                -1.0 / 12.0,
            ]
        ),
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    "d2_stencil_points",
    [3, 5, 7, 9],
)
def test_second_derivative_rule_moment_conditions(
    d2_stencil_points,
):
    """Generated stencil coefficients should satisfy polynomial moments."""

    offsets, coefficients = second_derivative_rule(
        d2_stencil_points
    )

    for power in range(d2_stencil_points):
        moment = np.sum(
            coefficients * offsets**power
        )

        expected = 2.0 if power == 2 else 0.0

        assert moment == pytest.approx(
            expected,
            abs=1.0e-10,
        )


@pytest.mark.parametrize(
    "invalid_stencil",
    [1, 2, 4, 6],
)
def test_second_derivative_rule_rejects_invalid_stencils(
    invalid_stencil,
):
    """Only odd centered stencils with at least three points are allowed."""

    with pytest.raises(ValueError):
        second_derivative_rule(
            invalid_stencil
        )


def test_second_derivative_rule_rejects_noninteger_stencil():
    """The stencil size must be an integer."""

    with pytest.raises(TypeError):
        second_derivative_rule(5.0)


def test_electric_field_is_zero_at_z_zero():
    """The field potential is defined to be zero at z = 0."""

    z = np.array([-10.0, 0.0, 10.0])

    field_potential = electric_field_potential(
        z=z,
        electric_field=20.0,
        charge=-1.0,
    )

    assert field_potential[1] == pytest.approx(0.0)


def test_electron_field_potential_sign():
    """For an electron and F > 0, U_F should increase with z."""

    z = np.array([-10.0, 0.0, 10.0])

    field_potential = electric_field_potential(
        z=z,
        electric_field=20.0,
        charge=-1.0,
    )

    assert field_potential[0] < 0.0
    assert field_potential[1] == pytest.approx(0.0)
    assert field_potential[2] > 0.0


@pytest.mark.parametrize(
    "d2_stencil_points",
    [3, 5, 7],
)
def test_hamiltonian_is_symmetric(
    d2_stencil_points,
):
    """The finite-difference Hamiltonian should be real symmetric."""

    z, _ = make_grid(
        z_min=-50.0,
        z_max=50.0,
        n_grid_points=101,
    )

    potential = square_well_potential(
        z=z,
        well_width=40.0,
        well_depth=0.3,
    )

    hamiltonian, _, _ = build_hamiltonian(
        z=z,
        mass=0.067,
        potential=potential,
        electric_field=5.0,
        charge=-1.0,
        d2_stencil_points=d2_stencil_points,
    )

    difference = (
        hamiltonian
        - hamiltonian.T
    )

    if difference.nnz > 0:
        max_asymmetry = np.max(
            np.abs(difference.data)
        )
    else:
        max_asymmetry = 0.0

    print(
        f"\n{d2_stencil_points}-point Hamiltonian: "
        f"maximum |H-H^T| = {max_asymmetry:.3e}"
    )

    assert max_asymmetry < 1.0e-12


def test_dirichlet_boundaries_are_zero():
    """All returned wavefunctions should vanish at both box boundaries."""

    result = make_symmetric_well_problem()

    np.testing.assert_allclose(
        result.wavefunctions[0, :],
        0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        result.wavefunctions[-1, :],
        0.0,
        atol=0.0,
    )


def test_wavefunctions_are_normalized():
    """Each state should satisfy sum |psi|^2 dz = 1."""

    result = make_symmetric_well_problem()

    norms = np.sum(
        np.abs(result.wavefunctions) ** 2,
        axis=0,
    ) * result.dz

    print("\nWavefunction normalization")
    print("------------------------------------------")
    for state, norm in enumerate(norms):
        print(f"State {state}: integral |psi|^2 dz = {norm:.12f}")

    np.testing.assert_allclose(
        norms,
        np.ones_like(norms),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_eigenenergies_are_ordered():
    """The returned eigenenergies should increase with state index."""

    result = make_symmetric_well_problem()

    print("\nCalculated finite-well energies")
    print("------------------------------------------")
    for state, energy in enumerate(result.energies):
        print(f"State {state}: {energy:.10f} eV")

    assert np.all(
        np.diff(result.energies) > 0.0
    )


def test_tridiagonal_and_sparse_solvers_agree():
    """Both eigensolvers should give the same low-energy spectrum."""

    result_tridiagonal = make_symmetric_well_problem(
        solver="tridiagonal",
    )

    result_sparse = make_symmetric_well_problem(
        solver="sparse",
    )

    differences = np.abs(
        result_tridiagonal.energies
        - result_sparse.energies
    )

    print("\nTridiagonal vs sparse eigensolver")
    print("------------------------------------------")
    print("State   Tridiagonal (eV)   Sparse (eV)      |Difference|")
    for state in range(len(differences)):
        print(
            f"{state:5d}   "
            f"{result_tridiagonal.energies[state]: .10f}   "
            f"{result_sparse.energies[state]: .10f}   "
            f"{differences[state]:.3e}"
        )

    np.testing.assert_allclose(
        result_tridiagonal.energies,
        result_sparse.energies,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_zero_field_density_is_symmetric():
    """A centered symmetric well should have symmetric probability densities."""

    result = make_symmetric_well_problem()

    densities = (
        np.abs(result.wavefunctions) ** 2
    )

    np.testing.assert_allclose(
        densities,
        densities[::-1, :],
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_zero_field_states_have_alternating_parity():
    """The lowest states of a symmetric 1D well alternate even and odd parity."""

    result = make_symmetric_well_problem(
        n_states=4,
        solver="tridiagonal",
    )

    for state in range(4):
        psi = result.wavefunctions[:, state]
        reflected = psi[::-1]

        if state % 2 == 0:
            np.testing.assert_allclose(
                psi,
                reflected,
                rtol=1.0e-7,
                atol=1.0e-9,
            )
        else:
            np.testing.assert_allclose(
                psi,
                -reflected,
                rtol=1.0e-7,
                atol=1.0e-9,
            )
