# src/qw/solver.py

"""Finite-difference solver for the one-dimensional Schrodinger equation."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from .constants import KINETIC_FACTOR, FIELD_FACTOR


# ============================================================
# Result container
# ============================================================

@dataclass
class SolverResult:
    """Results of a one-dimensional Schrodinger calculation."""

    energies: np.ndarray
    wavefunctions: np.ndarray

    potential: np.ndarray
    field_potential: np.ndarray
    total_potential: np.ndarray

    z: np.ndarray
    dz: float


# ============================================================
# Finite-difference point rule
# ============================================================

def second_derivative_rule(d2_stencil_points):
    """
    Return a centered finite-difference rule for the second derivative.

    Parameters
    ----------
    d2_stencil_points : int
        Number of points in the finite-difference stencil for the
        second derivative.

        Must be an odd integer >= 3.

    Returns
    -------
    offsets : ndarray
        Integer grid offsets relative to the central point.

    coefficients : ndarray
        Dimensionless coefficients such that

            d2psi/dz2
            ~= sum_j coefficients[j]
               * psi[i + offsets[j]] / dz**2

    Examples
    --------
    3-point rule:

        offsets      = [-1, 0, 1]
        coefficients = [1, -2, 1]

    5-point rule:

        offsets      = [-2, -1, 0, 1, 2]
        coefficients = [-1/12, 4/3, -5/2, 4/3, -1/12]
    """

    if not isinstance(d2_stencil_points, (int, np.integer)):
        raise TypeError(
            "d2_stencil_points must be an integer."
        )

    if d2_stencil_points < 3:
        raise ValueError(
            "d2_stencil_points must be at least 3."
        )

    if d2_stencil_points % 2 == 0:
        raise ValueError(
            "d2_stencil_points must be odd."
        )

    radius = d2_stencil_points // 2

    offsets = np.arange(
        -radius,
        radius + 1,
        dtype=float,
    )

    matrix = np.array(
        [
            offsets**power
            for power in range(d2_stencil_points)
        ],
        dtype=float,
    )

    rhs = np.zeros(
        d2_stencil_points,
        dtype=float,
    )

    # d^2(x^2)/dx^2 = 2
    rhs[2] = 2.0

    coefficients = np.linalg.solve(
        matrix,
        rhs,
    )

    return offsets.astype(int), coefficients


# ============================================================
# Electric-field potential
# ============================================================

def electric_field_potential(
    z,
    electric_field,
    charge=-1.0,
):
    """
    Calculate the potential energy from a uniform electric field.

    Parameters
    ----------
    z : array_like
        Position grid in Angstrom.

    electric_field : float
        Electric field in kV/cm.

        A positive electric field points along +z.

    charge : float, optional
        Particle charge in units of the elementary charge e.

        electron : -1
        hole     : +1

    Returns
    -------
    field_potential : ndarray
        Electric-field potential energy in eV.

    Notes
    -----
    The electric-field potential energy is defined to be zero
    at z = 0.

    For a uniform electric field F pointing along +z,

        phi(z) = -F z

    and therefore

        U(z) = q phi(z)
             = -q F z.
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    field_potential = (
        -charge
        * FIELD_FACTOR
        * electric_field
        * z
    )

    return field_potential


# ============================================================
# Hamiltonian
# ============================================================

def build_hamiltonian(
    z,
    mass,
    potential,
    electric_field=0.0,
    charge=-1.0,
    d2_stencil_points=3,
):
    """
    Construct the finite-difference Hamiltonian.

    Parameters
    ----------
    z : array_like
        Uniform spatial grid in Angstrom.

    mass : float
        Constant effective mass in units of the free-electron mass m0.

    potential : array_like
        Confinement potential in eV.

    electric_field : float, optional
        Uniform electric field in kV/cm.

    charge : float, optional
        Particle charge in units of e.

        Default is -1 for an electron.

    d2_stencil_points : int, optional
        Number of points in the centered finite-difference stencil
        for the second derivative.

        Examples: 3, 5, 7.

    Returns
    -------
    hamiltonian : scipy.sparse.csr_matrix
        Hamiltonian matrix acting on the interior grid points.

    field_potential : ndarray
        Electric-field potential on the full grid in eV.

    total_potential : ndarray
        Total potential on the full grid in eV.

    Notes
    -----
    Dirichlet boundary conditions are imposed:

        psi(z_min) = psi(z_max) = 0.

    The two boundary values are therefore not included as unknowns
    in the Hamiltonian.
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    potential = np.asarray(
        potential,
        dtype=float,
    )

    if z.ndim != 1:
        raise ValueError(
            "z must be a one-dimensional array."
        )

    if potential.shape != z.shape:
        raise ValueError(
            "potential must have the same shape as z."
        )

    if mass <= 0.0:
        raise ValueError(
            "mass must be positive."
        )

    if len(z) < 3:
        raise ValueError(
            "The spatial grid must contain at least 3 points."
        )

    dz_values = np.diff(z)

    if not np.allclose(
        dz_values,
        dz_values[0],
    ):
        raise ValueError(
            "The spatial grid must be uniform."
        )

    dz = dz_values[0]

    # --------------------------------------------------------
    # Potential energy
    # --------------------------------------------------------

    field_potential = electric_field_potential(
        z=z,
        electric_field=electric_field,
        charge=charge,
    )

    total_potential = (
        potential
        + field_potential
    )

    # --------------------------------------------------------
    # Interior grid
    # --------------------------------------------------------

    # Dirichlet boundary conditions:
    #
    # psi(z_min) = psi(z_max) = 0
    #
    # Only the interior points are unknown.
    n_interior = len(z) - 2

    offsets, coefficients = second_derivative_rule(
        d2_stencil_points
    )

    if n_interior < d2_stencil_points:
        raise ValueError(
            "The grid is too small for the requested "
            "finite-difference stencil."
        )

    # --------------------------------------------------------
    # Kinetic-energy operator
    # --------------------------------------------------------

    # T = -(hbar^2 / 2m*) d^2/dz^2
    #
    # mass is supplied in units of m0.
    kinetic_prefactor = (
        -KINETIC_FACTOR
        / (mass * dz**2)
    )

    diagonals = []
    diagonal_offsets = []

    for offset, coefficient in zip(
        offsets,
        coefficients,
    ):

        diagonal_length = (
            n_interior
            - abs(offset)
        )

        if diagonal_length <= 0:
            continue

        diagonal = np.full(
            diagonal_length,
            kinetic_prefactor * coefficient,
            dtype=float,
        )

        diagonals.append(
            diagonal
        )

        diagonal_offsets.append(
            offset
        )

    hamiltonian = diags(
        diagonals,
        diagonal_offsets,
        shape=(
            n_interior,
            n_interior,
        ),
        format="csr",
    )

    # --------------------------------------------------------
    # Potential-energy operator
    # --------------------------------------------------------

    hamiltonian += diags(
        total_potential[1:-1],
        offsets=0,
        format="csr",
    )

    return (
        hamiltonian,
        field_potential,
        total_potential,
    )


# ============================================================
# Eigenvalue solver
# ============================================================

def solve_hamiltonian(
    hamiltonian,
    n_states,
    d2_stencil_points=3,
    solver="auto",
):
    """
    Solve the Hamiltonian eigenvalue problem.

    Parameters
    ----------
    hamiltonian : scipy sparse matrix
        Hamiltonian matrix.

    n_states : int
        Number of lowest-energy eigenstates to calculate.

    d2_stencil_points : int, optional
        Number of points in the finite-difference stencil used
        for the second derivative.

    solver : {"auto", "tridiagonal", "sparse"}, optional
        Eigensolver.

    Returns
    -------
    energies : ndarray
        Eigenenergies in eV.

    eigenvectors : ndarray
        Eigenvectors on the interior spatial grid.
    """

    matrix_size = hamiltonian.shape[0]

    if n_states < 1:
        raise ValueError(
            "n_states must be at least 1."
        )

    if n_states >= matrix_size:
        raise ValueError(
            "n_states must be smaller than "
            "the Hamiltonian dimension."
        )

    # --------------------------------------------------------
    # Automatic solver selection
    # --------------------------------------------------------

    if solver == "auto":

        if d2_stencil_points == 3:
            solver = "tridiagonal"

        else:
            solver = "sparse"

    # --------------------------------------------------------
    # Tridiagonal solver
    # --------------------------------------------------------

    if solver == "tridiagonal":

        if d2_stencil_points != 3:
            raise ValueError(
                "The tridiagonal solver can only be used with "
                "the 3-point finite-difference stencil."
            )

        main_diagonal = hamiltonian.diagonal(
            0
        )

        off_diagonal = hamiltonian.diagonal(
            1
        )

        energies, eigenvectors = eigh_tridiagonal(
            main_diagonal,
            off_diagonal,
            select="i",
            select_range=(
                0,
                n_states - 1,
            ),
        )

    # --------------------------------------------------------
    # Sparse solver
    # --------------------------------------------------------

    elif solver == "sparse":

        energies, eigenvectors = eigsh(
            hamiltonian,
            k=n_states,
            which="SA",
        )

        order = np.argsort(
            energies
        )

        energies = energies[
            order
        ]

        eigenvectors = eigenvectors[
            :,
            order,
        ]

    else:
        raise ValueError(
            f"Unknown solver '{solver}'. "
            "Use 'auto', 'tridiagonal', or 'sparse'."
        )

    return energies, eigenvectors


# ============================================================
# Complete solver
# ============================================================

def solve_schrodinger(
    z,
    mass,
    potential,
    electric_field=0.0,
    charge=-1.0,
    n_states=5,
    d2_stencil_points=3,
    solver="auto",
):
    """
    Build and solve the one-dimensional Schrodinger equation.

    Parameters
    ----------
    z : array_like
        Uniform spatial grid in Angstrom.

    mass : float
        Constant effective mass in units of m0.

    potential : array_like
        Confinement potential in eV.

    electric_field : float, optional
        Electric field in kV/cm.

    charge : float, optional
        Particle charge in units of e.

    n_states : int, optional
        Number of eigenstates to calculate.

    d2_stencil_points : int, optional
        Number of points in the finite-difference stencil
        for the second derivative.

    solver : {"auto", "tridiagonal", "sparse"}, optional
        Eigensolver.

    Returns
    -------
    SolverResult
        Calculation results.
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    potential = np.asarray(
        potential,
        dtype=float,
    )

    (
        hamiltonian,
        field_potential,
        total_potential,
    ) = build_hamiltonian(
        z=z,
        mass=mass,
        potential=potential,
        electric_field=electric_field,
        charge=charge,
        d2_stencil_points=d2_stencil_points,
    )

    energies, eigenvectors = solve_hamiltonian(
        hamiltonian=hamiltonian,
        n_states=n_states,
        d2_stencil_points=d2_stencil_points,
        solver=solver,
    )

    dz = z[1] - z[0]

    # Convert eigensolver normalization:
    #
    # sum_i |psi_i|^2 = 1
    #
    # to
    #
    # sum_i |psi_i|^2 dz = 1.
    eigenvectors = (
        eigenvectors
        / np.sqrt(dz)
    )

    # Restore the two fixed Dirichlet boundary values.
    wavefunctions = np.zeros(
        (
            len(z),
            n_states,
        ),
        dtype=float,
    )

    wavefunctions[
        1:-1,
        :,
    ] = eigenvectors

    return SolverResult(
        energies=energies,
        wavefunctions=wavefunctions,
        potential=potential,
        field_potential=field_potential,
        total_potential=total_potential,
        z=z,
        dz=dz,
    )