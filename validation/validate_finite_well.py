"""Validate the numerical solver against the analytic finite square well."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from qw.constants import KINETIC_FACTOR
from qw.grid import make_grid
from qw.solver import solve_schrodinger
from qw.well import square_well_potential

from common import (
    add_header,
    create_validation_directory,
    write_log,
)


def analytic_finite_well_states(
    well_width,
    well_depth,
    mass,
):
    """
    Calculate bound-state energies of a symmetric finite square well.

    The potential is

        V(z) = -well_depth    for |z| <= well_width / 2
        V(z) = 0              outside the well.

    The mass is assumed constant everywhere, matching the numerical model.
    """

    half_width = well_width / 2.0

    kinetic_coefficient = (
        KINETIC_FACTOR
        / mass
    )

    rho = (
        half_width
        * np.sqrt(
            well_depth
            / kinetic_coefficient
        )
    )

    states = []

    # Small offset used to stay away from tan/cot singularities
    # and from the continuum threshold x = rho.
    epsilon = 1.0e-10

    interval = 0

    while interval * np.pi < rho:

        # ----------------------------------------------------
        # Even states
        # x tan(x) = sqrt(rho^2 - x^2)
        # ----------------------------------------------------

        lower = (
            interval * np.pi
            + epsilon
        )

        upper = min(
            interval * np.pi
            + np.pi / 2.0
            - epsilon,
            rho - epsilon,
        )

        if upper > lower:

            def even_equation(x):
                return (
                    x * np.tan(x)
                    - np.sqrt(rho**2 - x**2)
                )

            if (
                even_equation(lower)
                * even_equation(upper)
                < 0.0
            ):

                x_root = brentq(
                    even_equation,
                    lower,
                    upper,
                )

                energy = (
                    kinetic_coefficient
                    * (x_root / half_width) ** 2
                    - well_depth
                )

                states.append(
                    {
                        "energy": energy,
                        "parity": "even",
                    }
                )

        # ----------------------------------------------------
        # Odd states
        # -x cot(x) = sqrt(rho^2 - x^2)
        # ----------------------------------------------------

        lower = (
            interval * np.pi
            + np.pi / 2.0
            + epsilon
        )

        upper = min(
            (interval + 1) * np.pi
            - epsilon,
            rho - epsilon,
        )

        if upper > lower:

            def odd_equation(x):
                return (
                    -x / np.tan(x)
                    - np.sqrt(rho**2 - x**2)
                )

            if (
                odd_equation(lower)
                * odd_equation(upper)
                < 0.0
            ):

                x_root = brentq(
                    odd_equation,
                    lower,
                    upper,
                )

                energy = (
                    kinetic_coefficient
                    * (x_root / half_width) ** 2
                    - well_depth
                )

                states.append(
                    {
                        "energy": energy,
                        "parity": "odd",
                    }
                )

        interval += 1

    states.sort(
        key=lambda state: state["energy"]
    )

    return states


def analytic_wavefunction(
    z,
    energy,
    parity,
    well_width,
    well_depth,
    mass,
):
    """Return a normalized analytic finite-well bound-state wavefunction."""

    half_width = well_width / 2.0

    kinetic_coefficient = (
        KINETIC_FACTOR
        / mass
    )

    energy_from_bottom = (
        energy
        + well_depth
    )

    k = np.sqrt(
        energy_from_bottom
        / kinetic_coefficient
    )

    kappa = np.sqrt(
        -energy
        / kinetic_coefficient
    )

    psi = np.empty_like(
        z,
        dtype=float,
    )

    inside = (
        np.abs(z)
        <= half_width
    )

    outside = ~inside

    if parity == "even":

        psi[inside] = np.cos(
            k * z[inside]
        )

        psi[outside] = (
            np.cos(k * half_width)
            * np.exp(
                -kappa
                * (
                    np.abs(z[outside])
                    - half_width
                )
            )
        )

    elif parity == "odd":

        psi[inside] = np.sin(
            k * z[inside]
        )

        psi[outside] = (
            np.sign(z[outside])
            * np.sin(k * half_width)
            * np.exp(
                -kappa
                * (
                    np.abs(z[outside])
                    - half_width
                )
            )
        )

    else:
        raise ValueError(
            "parity must be 'even' or 'odd'."
        )

    dz = z[1] - z[0]

    normalization = np.sqrt(
        np.sum(np.abs(psi) ** 2)
        * dz
    )

    psi /= normalization

    return psi


def run(output_dir, log_lines):
    """Run the analytic finite-square-well validation."""

    add_header(
        log_lines,
        "Finite square well: numerical vs analytic solution",
    )

    well_width = 100.0       # Angstrom
    well_depth = 0.300       # eV
    mass = 0.067             # m0

    z_min = -500.0           # Angstrom
    z_max = 500.0            # Angstrom
    n_grid_points = 16001

    analytic_states = analytic_finite_well_states(
        well_width=well_width,
        well_depth=well_depth,
        mass=mass,
    )

    n_bound_states = len(
        analytic_states
    )

    z, dz = make_grid(
        z_min=z_min,
        z_max=z_max,
        n_grid_points=n_grid_points,
    )

    potential = square_well_potential(
        z=z,
        well_width=well_width,
        well_depth=well_depth,
        center=0.0,
    )

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=potential,
        electric_field=0.0,
        charge=-1.0,
        n_states=n_bound_states,
        d2_stencil_points=3,
        solver="tridiagonal",
    )

    log_lines.append(f"Well width:       {well_width:.6f} Angstrom")
    log_lines.append(f"Well depth:       {well_depth:.6f} eV")
    log_lines.append(f"Effective mass:   {mass:.6f} m0")
    log_lines.append(f"Box:              {z_min:.2f} to {z_max:.2f} Angstrom")
    log_lines.append(f"Grid points:      {n_grid_points}")
    log_lines.append(f"Grid spacing:     {dz:.6f} Angstrom")
    log_lines.append(f"Bound states:     {n_bound_states}")
    log_lines.append("")

    analytic_energies = np.array(
        [state["energy"] for state in analytic_states]
    )

    numerical_energies = result.energies

    # --------------------------------------------------------
    # Energy comparison
    # --------------------------------------------------------

    log_lines.append(
        "State   Parity      Analytic (eV)      Numerical (eV)     "
        "Abs. error (meV)"
    )

    log_lines.append(
        "-----   ------      -------------      --------------     "
        "----------------"
    )

    for state_index, state in enumerate(
        analytic_states
    ):

        analytic_energy = state["energy"]
        numerical_energy = numerical_energies[state_index]

        error_meV = (
            abs(
                numerical_energy
                - analytic_energy
            )
            * 1000.0
        )

        log_lines.append(
            f"{state_index:5d}   "
            f"{state['parity']:>6s}      "
            f"{analytic_energy: .12f}      "
            f"{numerical_energy: .12f}      "
            f"{error_meV: .6f}"
        )

    # --------------------------------------------------------
    # Wavefunction comparison
    # --------------------------------------------------------

    log_lines.append("")
    log_lines.append("Wavefunction comparison")

    analytic_wavefunctions = []

    for state_index, state in enumerate(
        analytic_states
    ):

        psi_analytic = analytic_wavefunction(
            z=z,
            energy=state["energy"],
            parity=state["parity"],
            well_width=well_width,
            well_depth=well_depth,
            mass=mass,
        )

        psi_numerical = result.wavefunctions[
            :,
            state_index,
        ]

        overlap = (
            np.sum(
                psi_analytic
                * psi_numerical
            )
            * dz
        )

        # Eigenvectors have arbitrary overall sign.
        if overlap < 0.0:
            psi_analytic *= -1.0
            overlap *= -1.0

        l2_error = np.sqrt(
            np.sum(
                (
                    psi_numerical
                    - psi_analytic
                ) ** 2
            )
            * dz
        )

        analytic_wavefunctions.append(
            psi_analytic
        )

        log_lines.append(
            f"State {state_index}: "
            f"overlap = {overlap:.12f}, "
            f"L2 error = {l2_error:.6e}"
        )

    # --------------------------------------------------------
    # Energy figure
    # --------------------------------------------------------

    states = np.arange(
        n_bound_states
    )

    fig, ax = plt.subplots()

    ax.plot(
        states,
        analytic_energies,
        marker="o",
        linestyle="none",
        markersize=8,
        label="Analytic",
    )

    ax.plot(
        states,
        numerical_energies,
        marker="x",
        linestyle="none",
        markersize=8,
        label="Numerical",
    )

    ax.set_xlabel("State index")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Finite square well: bound-state energies")
    ax.set_xticks(states)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "finite_well_energies.png",
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Wavefunction figure
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    for state_index in range(
        n_bound_states
    ):

        ax.plot(
            z,
            result.wavefunctions[:, state_index],
            label=f"Numerical state {state_index}",
        )

        ax.plot(
            z,
            analytic_wavefunctions[state_index],
            linestyle="--",
            label=f"Analytic state {state_index}",
        )

    ax.set_xlim(
        -150.0,
        150.0,
    )

    ax.set_xlabel(r"$z$ ($\AA$)")
    ax.set_ylabel(r"$\psi(z)$ ($\AA^{-1/2}$)")
    ax.set_title("Finite square well: wavefunctions")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "finite_well_wavefunctions.png",
        dpi=200,
    )
    plt.close(fig)


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
