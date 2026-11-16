# src/qw/outputs.py

"""Output functions for quantum-well calculations."""

from datetime import datetime
from pathlib import Path


# ============================================================
# Output directory
# ============================================================

def create_output_directory():
    """
    Create a timestamped output directory in the working directory.

    Returns
    -------
    output_dir : pathlib.Path
        Path to the newly created output directory.

    Notes
    -----
    The directory name has the form

        outputs_YYYY_MM_DD_HH_MM_SS
    """

    timestamp = datetime.now().strftime(
        "%Y_%m_%d_%H_%M_%S"
    )

    output_dir = (
        Path.cwd()
        / f"outputs_{timestamp}"
    )

    output_dir.mkdir(
        parents=False,
        exist_ok=False,
    )

    return output_dir


# ============================================================
# Terminal and log output
# ============================================================

class Tee:
    """
    Write text simultaneously to multiple output streams.

    This is used to display calculation output in the terminal
    while also saving the same text to the calculation log.
    """

    def __init__(
        self,
        *streams,
    ):
        self.streams = streams

    def write(
        self,
        text,
    ):
        for stream in self.streams:
            stream.write(
                text
            )

    def flush(
        self,
    ):
        for stream in self.streams:
            stream.flush()


# ============================================================
# Input parameters
# ============================================================

def print_input_parameters(
    z_min,
    z_max,
    n_grid_points,
    well_width,
    well_depth,
    mass,
    charge,
    electric_field,
    n_states,
    d2_stencil_points,
    solver,
    well_center=0.0,
):
    """Print the input parameters of the calculation."""

    print("\nInput parameters")
    print("==========================================")

    print(
        f"Box limits:             "
        f"{z_min:.2f} to {z_max:.2f} Angstrom"
    )

    print(
        f"Grid points:            "
        f"{n_grid_points}"
    )

    print(
        f"Well center:            "
        f"{well_center:.2f} Angstrom"
    )

    print(
        f"Well width:             "
        f"{well_width:.2f} Angstrom"
    )

    print(
        f"Well depth:             "
        f"{well_depth:.6f} eV"
    )

    print(
        f"Effective mass:         "
        f"{mass:.6f} m0"
    )

    print(
        f"Charge:                 "
        f"{charge:.1f} e"
    )

    print(
        f"Electric field:         "
        f"{electric_field:.6f} kV/cm"
    )

    print(
        f"Number of states:       "
        f"{n_states}"
    )

    print(
        f"d2 stencil points:      "
        f"{d2_stencil_points}"
    )

    print(
        f"Solver:                 "
        f"{solver}"
    )


# ============================================================
# Calculation results
# ============================================================

def print_results(
    result,
):
    """Print the calculated eigenenergies."""

    print("\nEigenenergies")
    print("==========================================")

    for state, energy in enumerate(
        result.energies
    ):

        print(
            f"State {state:2d}: "
            f"{energy: .8f} eV"
            f"    "
            f"{energy * 1000.0: .4f} meV"
        )