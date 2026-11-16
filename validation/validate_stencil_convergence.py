"""Study convergence of the finite-difference second-derivative stencils."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qw.constants import KINETIC_FACTOR
from qw.grid import make_grid
from qw.solver import solve_schrodinger

from common import (
    add_header,
    create_validation_directory,
    write_log,
)


def analytic_ground_state_energy(width, mass):
    """Ground-state energy of a one-dimensional infinite square well."""

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
    """Calculate the numerical ground-state energy of a Dirichlet box."""

    z, dz = make_grid(
        z_min=0.0,
        z_max=width,
        n_grid_points=n_grid_points,
    )

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=np.zeros_like(z),
        electric_field=0.0,
        charge=-1.0,
        n_states=1,
        d2_stencil_points=d2_stencil_points,
        solver="auto",
    )

    return dz, result.energies[0]


def run(output_dir, log_lines):
    """Run the stencil-convergence study and save its figures."""

    add_header(
        log_lines,
        "Finite-difference stencil convergence",
    )

    width = 100.0       # Angstrom
    mass = 0.067        # m0

    grid_points = np.array(
        [101, 201, 401, 801],
        dtype=int,
    )

    stencil_points = [3, 5, 7, 9]

    exact_energy = analytic_ground_state_energy(
        width=width,
        mass=mass,
    )

    log_lines.append(f"Infinite-well width: {width:.6f} Angstrom")
    log_lines.append(f"Effective mass:      {mass:.6f} m0")
    log_lines.append(f"Exact E0:            {exact_energy:.12f} eV")
    log_lines.append("")

    convergence_data = {}

    for stencil in stencil_points:

        dz_values = []
        errors = []
        energies = []

        for n_grid_points in grid_points:

            dz, energy = numerical_ground_state_energy(
                width=width,
                mass=mass,
                n_grid_points=int(n_grid_points),
                d2_stencil_points=stencil,
            )

            error = abs(
                energy
                - exact_energy
            )

            dz_values.append(dz)
            energies.append(energy)
            errors.append(error)

        dz_values = np.asarray(dz_values)
        energies = np.asarray(energies)
        errors = np.asarray(errors)

        # Fit only the four finest grids so the estimate reflects
        # the asymptotic convergence regime as closely as possible.
        fit = np.polyfit(
            np.log(dz_values[-4:]),
            np.log(errors[-4:]),
            1,
        )

        observed_order = fit[0]
        nominal_order = stencil - 1

        convergence_data[stencil] = {
            "dz": dz_values,
            "energies": energies,
            "errors": errors,
            "observed_order": observed_order,
            "nominal_order": nominal_order,
        }

        log_lines.append(
            f"{stencil}-point stencil"
        )
        log_lines.append(
            f"  Nominal interior order:  {nominal_order:d}"
        )
        log_lines.append(
            f"  Observed global order:   {observed_order:.6f}"
        )

        for n_grid_points, dz, energy, error in zip(
            grid_points,
            dz_values,
            energies,
            errors,
        ):
            log_lines.append(
                f"  N = {n_grid_points:4d}   "
                f"dz = {dz:10.6f} Angstrom   "
                f"E0 = {energy:.12f} eV   "
                f"error = {error:.6e} eV"
            )

        log_lines.append("")

    # --------------------------------------------------------
    # Convergence plot
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    for stencil in stencil_points:

        data = convergence_data[stencil]

        ax.loglog(
            data["dz"],
            data["errors"],
            marker="o",
            label=(
                f"{stencil}-point "
                f"(p = {data['observed_order']:.2f})"
            ),
        )

    ax.set_xlabel(r"Grid spacing $\Delta z$ ($\AA$)")
    ax.set_ylabel(r"$|E_0-E_0^{\mathrm{exact}}|$ (eV)")
    ax.set_title("Finite-difference stencil convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "stencil_convergence.png",
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Observed-order plot
    # --------------------------------------------------------

    observed_orders = [
        convergence_data[stencil]["observed_order"]
        for stencil in stencil_points
    ]

    nominal_orders = [
        convergence_data[stencil]["nominal_order"]
        for stencil in stencil_points
    ]

    x = np.arange(len(stencil_points))

    fig, ax = plt.subplots()

    ax.plot(
        x,
        nominal_orders,
        marker="o",
        label="Nominal interior order",
    )

    ax.plot(
        x,
        observed_orders,
        marker="o",
        label="Observed global order",
    )

    ax.set_xticks(
        x,
        [f"{stencil}-point" for stencil in stencil_points],
    )

    ax.set_ylabel("Convergence order")
    ax.set_title("Nominal and observed convergence orders")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "stencil_convergence_order.png",
        dpi=200,
    )
    plt.close(fig)

    log_lines.append(
        "Interpretation: the 3-point stencil retains its expected "
        "second-order global convergence. Wider stencils currently "
        "lose their nominal high-order behavior because the centered "
        "stencil is truncated near the Dirichlet boundaries."
    )


if __name__ == "__main__":

    output_directory = create_validation_directory()
    lines = []

    run(
        output_dir=output_directory,
        log_lines=lines,
    )

    log_file = write_log(
        output_dir=output_directory,
        lines=lines,
    )

    print(f"Validation output: {output_directory}")
    print(f"Log: {log_file}")
