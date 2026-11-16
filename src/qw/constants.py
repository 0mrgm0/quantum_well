# constants.py

"""Physical constants for calculations in eV, Angstrom, and atomic units."""

import numpy as np


# ============================================================
# Atomic units
# ============================================================

# Electron rest mass [atomic units]
ELECTRON_MASS = 1.0

# Elementary charge magnitude [atomic units]
ELEMENTARY_CHARGE = 1.0

# Reduced Planck constant [atomic units]
HBAR = 1.0

# Planck constant [atomic units]
PLANCK_CONSTANT = 2.0 * np.pi


# ============================================================
# Unit conversions
# ============================================================

# Bohr radius [Angstrom]
BOHR_RADIUS = 0.529177210903

# Hartree energy [eV]
HARTREE = 27.211386245988


# ============================================================
# Convenient factors for calculations in eV and Angstrom
# ============================================================

# hbar^2 / (2 m0) [eV Angstrom^2]
KINETIC_FACTOR = 3.80998212

# Electric-field potential-energy conversion:
#
# 1 kV/cm = 1e-5 V/Angstrom
#
# For a particle with charge magnitude |q| = e,
# 1 V corresponds numerically to 1 eV of potential energy.
FIELD_FACTOR = 1.0e-5  # eV / (Angstrom * kV/cm)
