"""Spatial grids for one-dimensional quantum-well calculations."""

import numpy as np


def make_grid(z_min, z_max, n_grid_points):
    """
    Create a uniform one-dimensional spatial grid.

    Parameters
    ----------
    z_min : float
        Left boundary of the numerical box in Angstrom.

    z_max : float
        Right boundary of the numerical box in Angstrom.

    n_grid_points : int
        Total number of grid points, including both boundaries.

    Returns
    -------
    z : ndarray
        Uniform spatial grid in Angstrom.

    dz : float
        Grid spacing in Angstrom.
    """

    if z_max <= z_min:
        raise ValueError("z_max must be larger than z_min.")

    if not isinstance(n_grid_points, (int, np.integer)):
        raise TypeError("n_grid_points must be an integer.")

    if n_grid_points < 3:
        raise ValueError("n_grid_points must be at least 3.")

    z = np.linspace(z_min, z_max, n_grid_points, dtype=float)

    dz = z[1] - z[0]

    return z, dz
