# scripts/run_qw.py

"""Run a one-dimensional finite quantum-well calculation."""

from contextlib import redirect_stdout
from datetime import datetime
import sys

from qw import parameters

from qw.calculations import (
    run_single_quantum_well,
)

from qw.outputs import (
    Tee,
    create_output_directory,
    print_input_parameters,
    print_results,
)

from qw.plots import (
    plot_wavefunctions,
    plot_densities,
)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        create_output_directory()
    )

    log_path = (
        output_dir
        / "calculation.log"
    )

    terminal = sys.stdout

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        # Write simultaneously to the terminal and the log file.
        with redirect_stdout(
            Tee(
                terminal,
                log_file,
            )
        ):

            calculation_start = (
                datetime.now()
            )

            print(
                "\n=========================================="
            )
            print(
                "Quantum-well calculation"
            )
            print(
                "=========================================="
            )

            print(
                f"Started: "
                f"{calculation_start:%Y-%m-%d %H:%M:%S}"
            )

            print(
                f"Output directory: "
                f"{output_dir}"
            )


            # ====================================================
            # Input parameters
            # ====================================================

            print_input_parameters(
                z_min=parameters.z_min,
                z_max=parameters.z_max,
                n_grid_points=parameters.n_grid_points,
                well_width=parameters.well_width,
                well_depth=parameters.well_depth,
                mass=parameters.mass,
                charge=parameters.charge,
                electric_field=parameters.electric_field,
                n_states=parameters.n_states,
                d2_stencil_points=parameters.d2_stencil_points,
                solver=parameters.solver,
                well_center=parameters.well_center,
            )


            # ====================================================
            # Calculation
            # ====================================================

            result = run_single_quantum_well(
                z_min=parameters.z_min,
                z_max=parameters.z_max,
                n_grid_points=parameters.n_grid_points,
                well_width=parameters.well_width,
                well_depth=parameters.well_depth,
                mass=parameters.mass,
                electric_field=parameters.electric_field,
                charge=parameters.charge,
                n_states=parameters.n_states,
                d2_stencil_points=parameters.d2_stencil_points,
                solver=parameters.solver,
                well_center=parameters.well_center,
            )


            # ====================================================
            # Results
            # ====================================================

            print_results(
                result
            )


            # ====================================================
            # Plots
            # ====================================================

            print(
                "\nPlots"
            )

            print(
                "=========================================="
            )

            plot_wavefunctions(
                result,
                wavefunction_scale=(
                    parameters.wavefunction_scale
                ),
                output_dir=output_dir,
            )

            plot_densities(
                result,
                density_scale=(
                    parameters.density_scale
                ),
                output_dir=output_dir,
            )


            # ====================================================
            # Timing
            # ====================================================

            calculation_end = (
                datetime.now()
            )

            elapsed = (
                calculation_end
                - calculation_start
            )

            print(
                "\n=========================================="
            )

            print(
                f"Ended:   "
                f"{calculation_end:%Y-%m-%d %H:%M:%S}"
            )

            print(
                f"Elapsed: "
                f"{elapsed}"
            )

            print(
                "==========================================\n"
            )

    print(
        f"Calculation log: {log_path}"
    )


if __name__ == "__main__":
    main()