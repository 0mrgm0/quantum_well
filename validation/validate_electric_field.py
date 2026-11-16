"""Validate the electric-field response of a symmetric quantum well."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from qw.grid import make_grid
from qw.solver import solve_schrodinger
from qw.well import square_well_potential

from common import (
    add_header,
    create_validation_directory,
    write_log,
)


def run(output_dir, log_lines):
    """Validate symmetry and low-field Stark behavior."""

    add_header(
        log_lines,
        "Electric-field validation: Stark symmetry",
    )

    # Symmetric finite square well
    well_width = 100.0       # Angstrom
    well_depth = 0.300       # eV
    mass = 0.067             # m0
    charge = -1.0            # electron

    z_min = -400.0           # Angstrom
    z_max = 400.0            # Angstrom
    n_grid_points = 4001

    n_states = 3

    # Small fields keep the system in the perturbative Stark regime.
    electric_fields = np.linspace(
        -5.0,
        5.0,
        21,
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

    energies = []

    for electric_field in electric_fields:

        result = solve_schrodinger(
            z=z,
            mass=mass,
            potential=potential,
            electric_field=electric_field,
            charge=charge,
            n_states=n_states,
            d2_stencil_points=3,
            solver="tridiagonal",
        )

        energies.append(
            result.energies
        )

    energies = np.asarray(
        energies
    )

    zero_index = np.argmin(
        np.abs(electric_fields)
    )

    zero_field_energies = energies[
        zero_index
    ]

    # Recalculate F = 0 so that <z> can also be checked directly.
    zero_field_result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=potential,
        electric_field=0.0,
        charge=charge,
        n_states=n_states,
        d2_stencil_points=3,
        solver="tridiagonal",
    )

    log_lines.append(f"Well width:       {well_width:.6f} Angstrom")
    log_lines.append(f"Well depth:       {well_depth:.6f} eV")
    log_lines.append(f"Effective mass:   {mass:.6f} m0")
    log_lines.append(f"Charge:           {charge:.1f} e")
    log_lines.append(f"Field range:      {electric_fields[0]:.2f} to {electric_fields[-1]:.2f} kV/cm")
    log_lines.append(f"Grid spacing:     {dz:.6f} Angstrom")
    log_lines.append("")

    log_lines.append(
        "For a symmetric well, parity transforms F -> -F, so the "
        "spectrum must satisfy E_n(F) = E_n(-F). At F = 0, each "
        "nondegenerate eigenstate also has <z> = 0, so the first-order "
        "Stark shift vanishes."
    )
    log_lines.append("")

    quadratic_coefficients = []

    for state in range(
        n_states
    ):

        energy_curve = energies[
            :,
            state,
        ]

        reversed_curve = energy_curve[
            ::-1
        ]

        symmetry_error = np.max(
            np.abs(
                energy_curve
                - reversed_curve
            )
        )

        psi_zero = zero_field_result.wavefunctions[
            :,
            state,
        ]

        expectation_z = (
            np.sum(
                z
                * np.abs(psi_zero) ** 2
            )
            * dz
        )

        energy_shift = (
            energy_curve
            - zero_field_energies[state]
        )

        field_squared = (
            electric_fields**2
        )

        # Fit Delta E = a F^2 through the origin.
        quadratic_coefficient = (
            np.dot(
                field_squared,
                energy_shift,
            )
            / np.dot(
                field_squared,
                field_squared,
            )
        )

        quadratic_coefficients.append(
            quadratic_coefficient
        )

        predicted_shift = (
            quadratic_coefficient
            * field_squared
        )

        residual_sum = np.sum(
            (
                energy_shift
                - predicted_shift
            ) ** 2
        )

        total_sum = np.sum(
            (
                energy_shift
                - np.mean(energy_shift)
            ) ** 2
        )

        if total_sum > 0.0:
            r_squared = (
                1.0
                - residual_sum / total_sum
            )
        else:
            r_squared = 1.0

        log_lines.append(
            f"State {state}"
        )
        log_lines.append(
            f"  E(0):                         "
            f"{zero_field_energies[state]: .12f} eV"
        )
        log_lines.append(
            f"  <z> at F = 0:                 "
            f"{expectation_z: .6e} Angstrom"
        )
        log_lines.append(
            f"  max |E(F)-E(-F)|:             "
            f"{symmetry_error:.6e} eV"
        )
        log_lines.append(
            f"  quadratic coefficient a:      "
            f"{quadratic_coefficient:.6e} "
            f"eV/(kV/cm)^2"
        )
        log_lines.append(
            f"  quadratic-fit R^2:             "
            f"{r_squared:.9f}"
        )
        log_lines.append("")

    # --------------------------------------------------------
    # Stark-shift plot
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    for state in range(
        n_states
    ):

        energy_shift_meV = (
            energies[:, state]
            - zero_field_energies[state]
        ) * 1000.0

        fitted_shift_meV = (
            quadratic_coefficients[state]
            * electric_fields**2
            * 1000.0
        )

        ax.plot(
            electric_fields,
            energy_shift_meV,
            marker="o",
            label=f"State {state}",
        )

        ax.plot(
            electric_fields,
            fitted_shift_meV,
            linestyle="--",
            label=f"State {state} quadratic fit",
        )

    ax.axvline(
        0.0,
        linewidth=0.8,
        linestyle=":",
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle=":",
    )

    ax.set_xlabel("Electric field (kV/cm)")
    ax.set_ylabel(r"$E_n(F)-E_n(0)$ (meV)")
    ax.set_title("Low-field Stark shift in a symmetric finite well")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "stark_shift.png",
        dpi=200,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Explicit F -> -F symmetry error
    # --------------------------------------------------------

    positive_mask = (
        electric_fields >= 0.0
    )

    positive_fields = electric_fields[
        positive_mask
    ]

    fig, ax = plt.subplots()

    for state in range(
        n_states
    ):

        symmetry_difference = np.abs(
            energies[positive_mask, state]
            - energies[::-1][positive_mask, state]
        )

        ax.semilogy(
            positive_fields,
            np.maximum(
                symmetry_difference,
                np.finfo(float).eps,
            ),
            marker="o",
            label=f"State {state}",
        )

    ax.set_xlabel(r"$|F|$ (kV/cm)")
    ax.set_ylabel(r"$|E_n(F)-E_n(-F)|$ (eV)")
    ax.set_title("Electric-field inversion symmetry")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "stark_symmetry.png",
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
