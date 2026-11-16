"""Potential profiles for one-dimensional quantum wells."""

import numpy as np


def square_well_potential(z, well_width, well_depth, center=0.0):
    """
    Construct a finite square quantum-well potential.

    Parameters
    ----------
    z : array_like
        Spatial grid in Angstrom.

    well_width : float
        Width of the quantum well in Angstrom.

    well_depth : float
        Depth of the quantum well in eV.
        This value is specified as a positive number.

    center : float, optional
        Center position of the quantum well in Angstrom.

    Returns
    -------
    potential : ndarray
        Quantum-well potential in eV.

    Notes
    -----
    The top of the barrier is defined as zero:

        V(z) = -well_depth    inside the well
        V(z) = 0              outside the well
    """

    z = np.asarray(z, dtype=float)

    if well_width <= 0.0:
        raise ValueError("well_width must be positive.")

    if well_depth <= 0.0:
        raise ValueError("well_depth must be positive.")

    left_interface = center - well_width / 2.0
    right_interface = center + well_width / 2.0

    potential = np.zeros_like(z)

    inside_well = (
        (z >= left_interface)
        & (z <= right_interface)
    )

    potential[inside_well] = -well_depth

    return potential
