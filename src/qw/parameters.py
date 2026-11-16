# src/qw/parameters.py

"""Input parameters for quantum-well calculations."""


# ============================================================
# Numerical grid
# ============================================================

# Numerical box [Angstrom]
z_min = -500.0
z_max = 500.0

# Number of spatial grid points, including both boundaries
n_grid_points = 2001


# ============================================================
# Quantum well
# ============================================================

# Center of the quantum well [Angstrom]
well_center = 0.0

# Well width [Angstrom]
well_width = 200.0

# Well depth [eV]
# The barrier top is defined as 0 eV.
# The potential inside the well is -well_depth.
well_depth = 0.300


# ============================================================
# Particle
# ============================================================

# Effective mass in units of the free-electron mass m0
mass = 0.067

# Charge in units of the elementary charge e
# electron = -1
# hole     = +1
charge = -1.0


# ============================================================
# Electric field
# ============================================================

# Electric field [kV/cm]
# Positive field points along +z.
#
# The electric-field potential energy is defined to be zero
# at z = 0.
electric_field = 0.0


# ============================================================
# Eigenstates
# ============================================================

# Number of eigenstates to calculate
n_states = 5


# ============================================================
# Finite-difference discretization
# ============================================================

# Number of points in the centered finite-difference stencil
# for the second derivative.
#
# Examples:
# 3 -> 3-point stencil
# 5 -> 5-point stencil
# 7 -> 7-point stencil
d2_stencil_points = 3


# ============================================================
# Eigensolver
# ============================================================

# "auto"
#     3-point stencil -> tridiagonal solver
#     wider stencil  -> sparse solver
#
# "tridiagonal"
#     scipy.linalg.eigh_tridiagonal
#
# "sparse"
#     scipy.sparse.linalg.eigsh
solver = "auto"


# ============================================================
# Plotting
# ============================================================

# Vertical plotting scale for normalized wavefunctions [eV]
wavefunction_scale = 0.02

# Vertical plotting scale for normalized probability densities [eV]
density_scale = 0.02