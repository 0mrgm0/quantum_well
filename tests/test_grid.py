"""Tests for spatial-grid construction."""

import numpy as np
import pytest

from qw.grid import make_grid


def test_make_grid_endpoints_size_and_spacing():
    """The grid should contain the requested endpoints and spacing."""

    z_min = -10.0
    z_max = 20.0
    n_grid_points = 61

    z, dz = make_grid(
        z_min=z_min,
        z_max=z_max,
        n_grid_points=n_grid_points,
    )

    assert len(z) == n_grid_points
    assert z[0] == pytest.approx(z_min)
    assert z[-1] == pytest.approx(z_max)
    assert dz == pytest.approx(
        (z_max - z_min) / (n_grid_points - 1)
    )


def test_make_grid_is_uniform():
    """All neighboring grid points should have the same separation."""

    z, dz = make_grid(
        z_min=-25.0,
        z_max=35.0,
        n_grid_points=121,
    )

    np.testing.assert_allclose(
        np.diff(z),
        dz,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_make_grid_rejects_invalid_limits():
    """The right boundary must be larger than the left boundary."""

    with pytest.raises(ValueError):
        make_grid(
            z_min=1.0,
            z_max=1.0,
            n_grid_points=101,
        )

    with pytest.raises(ValueError):
        make_grid(
            z_min=2.0,
            z_max=1.0,
            n_grid_points=101,
        )


def test_make_grid_rejects_invalid_number_of_points():
    """The grid needs an integer number of at least three points."""

    with pytest.raises(TypeError):
        make_grid(
            z_min=-1.0,
            z_max=1.0,
            n_grid_points=100.5,
        )

    with pytest.raises(ValueError):
        make_grid(
            z_min=-1.0,
            z_max=1.0,
            n_grid_points=2,
        )
