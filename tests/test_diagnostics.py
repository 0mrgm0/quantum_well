"""Visual numerical diagnostics for the quantum-well solver."""

import matplotlib

# Use a non-interactive backend so diagnostics work locally and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qw.constants import KINETIC_FACTOR
from qw.grid import make_grid
from qw.solver import solve_schrodinger


def analytic_infinite_well_energies(
    width,
    mass,
    n_states,
):
    """Return analytic infinite-well energies."""

    n = np.arange(
        1,
        n_states + 1,
        dtype=float,
    )

    return (
        KINETIC_FACTOR
        / mass
        * (n * np.pi / width) ** 2
    )


def numerical_ground_state_energy(
    width,
    mass,
    n_grid_points,
    d2_stencil_points,
):
    """Return numerical E0 and dz for a Dirichlet box."""

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


def test_save_infinite_well_energy_comparison(
    diagnostic_plot_dir,
):
    """Save a plot comparing numerical and analytic infinite-well energies."""

    width = 100.0
    mass = 0.067
    n_states = 5

    z, _ = make_grid(
        z_min=0.0,
        z_max=width,
        n_grid_points=1601,
    )

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=np.zeros_like(z),
        n_states=n_states,
        d2_stencil_points=3,
        solver="tridiagonal",
    )

    analytic = analytic_infinite_well_energies(
        width=width,
        mass=mass,
        n_states=n_states,
    )

    states = np.arange(n_states)

    fig, ax = plt.subplots()

    ax.plot(
        states,
        analytic,
        "o-",
        label="Analytic",
    )

    ax.plot(
        states,
        result.energies,
        "x--",
        label="Numerical",
    )

    ax.set_xlabel("State index")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Infinite square-well validation")
    ax.legend()

    fig.tight_layout()

    output_path = (
        diagnostic_plot_dir
        / "infinite_well_energy_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    max_relative_error = np.max(
        np.abs(result.energies - analytic)
        / analytic
    )

    print(
        f"\nSaved infinite-well comparison: {output_path}"
    )
    print(
        f"Maximum relative energy error: "
        f"{max_relative_error:.3e}"
    )

    assert output_path.exists()
    assert max_relative_error < 1.0e-5

    plt.close(fig)


def test_save_grid_convergence_plot(
    diagnostic_plot_dir,
):
    """Save the ground-state energy error versus grid spacing."""

    width = 100.0
    mass = 0.067

    exact = analytic_infinite_well_energies(
        width=width,
        mass=mass,
        n_states=1,
    )[0]

    grid_sizes = [51, 101, 201, 401]
    stencils = [3, 5, 7]

    fig, ax = plt.subplots()

    for stencil in stencils:

        dz_values = []
        errors = []

        for n_grid_points in grid_sizes:
            energy, dz = numerical_ground_state_energy(
                width=width,
                mass=mass,
                n_grid_points=n_grid_points,
                d2_stencil_points=stencil,
            )

            dz_values.append(dz)
            errors.append(abs(energy - exact))

        dz_values = np.asarray(dz_values)
        errors = np.asarray(errors)

        ax.loglog(
            dz_values,
            errors,
            "o-",
            label=f"{stencil}-point stencil",
        )

        print(
            f"\n{stencil}-point stencil convergence:"
        )

        for dz, error in zip(
            dz_values,
            errors,
        ):
            print(
                f"  dz={dz:.6f} A   error={error:.3e} eV"
            )

        assert errors[-1] < errors[0]

    ax.set_xlabel(r"Grid spacing $\Delta z$ ($\AA$)")
    ax.set_ylabel("Ground-state absolute error (eV)")
    ax.set_title("Finite-difference grid convergence")
    ax.legend()

    fig.tight_layout()

    output_path = (
        diagnostic_plot_dir
        / "grid_convergence.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    print(
        f"\nSaved convergence diagnostic plot: "
        f"{output_path}"
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)
