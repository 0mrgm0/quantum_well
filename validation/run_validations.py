"""Run all scientific validation studies."""

from datetime import datetime

from common import (
    add_header,
    create_validation_directory,
    write_log,
)
from validate_electric_field import run as run_electric_field_validation
from validate_finite_well import run as run_finite_well_validation
from validate_stencil_convergence import run as run_stencil_convergence


def main():

    start = datetime.now()

    output_dir = create_validation_directory()
    log_lines = []

    add_header(
        log_lines,
        "Quantum-well solver validation",
    )

    log_lines.append(
        f"Started: {start:%Y-%m-%d %H:%M:%S}"
    )

    run_stencil_convergence(
        output_dir=output_dir,
        log_lines=log_lines,
    )

    run_finite_well_validation(
        output_dir=output_dir,
        log_lines=log_lines,
    )

    run_electric_field_validation(
        output_dir=output_dir,
        log_lines=log_lines,
    )

    end = datetime.now()

    add_header(
        log_lines,
        "Validation completed",
    )

    log_lines.append(
        f"Ended:   {end:%Y-%m-%d %H:%M:%S}"
    )
    log_lines.append(
        f"Elapsed: {end - start}"
    )

    log_file = write_log(
        output_dir=output_dir,
        lines=log_lines,
    )

    print("Validation completed successfully.")
    print(f"Output directory: {output_dir}")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    main()
