# src/qw/calculations.py

"""High-level quantum-well calculations."""

from .grid import make_grid
from .well import square_well_potential
from .solver import solve_schrodinger


def run_single_quantum_well(
    z_min,
    z_max,
    n_grid_points,
    well_width,
    well_depth,
    mass,
    electric_field=0.0,
    charge=-1.0,
    n_states=5,
    d2_stencil_points=3,
    solver="auto",
    well_center=0.0,
):
    """
    Run a complete finite single-quantum-well calculation.

    Parameters
    ----------
    z_min : float
        Left boundary of the numerical box in Angstrom.

    z_max : float
        Right boundary of the numerical box in Angstrom.

    n_grid_points : int
        Number of spatial grid points, including the boundaries.

    well_width : float
        Width of the quantum well in Angstrom.

    well_depth : float
        Depth of the quantum well in eV.
        Must be specified as a positive number.

    mass : float
        Constant effective mass in units of m0.

    electric_field : float, optional
        Electric field in kV/cm.

    charge : float, optional
        Particle charge in units of e.

        electron : -1
        hole     : +1

    n_states : int, optional
        Number of eigenstates to calculate.

    d2_stencil_points : int, optional
        Number of points in the finite-difference stencil
        for the second derivative.

    solver : {"auto", "tridiagonal", "sparse"}, optional
        Eigensolver.

    well_center : float, optional
        Center of the quantum well in Angstrom.

    Returns
    -------
    SolverResult
        Energies, wavefunctions, potentials, and grid information.
    """

    # --------------------------------------------------------
    # Spatial grid
    # --------------------------------------------------------

    z, _ = make_grid(
        z_min=z_min,
        z_max=z_max,
        n_grid_points=n_grid_points,
    )

    # --------------------------------------------------------
    # Quantum-well potential
    # --------------------------------------------------------

    potential = square_well_potential(
        z=z,
        well_width=well_width,
        well_depth=well_depth,
        center=well_center,
    )

    # --------------------------------------------------------
    # Schrodinger equation
    # --------------------------------------------------------

    result = solve_schrodinger(
        z=z,
        mass=mass,
        potential=potential,
        electric_field=electric_field,
        charge=charge,
        n_states=n_states,
        d2_stencil_points=d2_stencil_points,
        solver=solver,
    )

    return result