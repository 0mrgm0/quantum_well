"""Tests for finite square-well potentials."""

import numpy as np
import pytest

from qw.well import square_well_potential


def test_square_well_potential_values():
    """The well should be -depth inside and zero outside."""

    z = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    potential = square_well_potential(
        z=z,
        well_width=2.0,
        well_depth=0.3,
        center=0.0,
    )

    expected = np.array(
        [0.0, -0.3, -0.3, -0.3, 0.0]
    )

    np.testing.assert_allclose(
        potential,
        expected,
    )


def test_square_well_center_shift():
    """Changing the center should translate the well correctly."""

    z = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    )

    potential = square_well_potential(
        z=z,
        well_width=2.0,
        well_depth=0.5,
        center=1.0,
    )

    expected = np.array(
        [0.0, 0.0, -0.5, -0.5, -0.5, 0.0]
    )

    np.testing.assert_allclose(
        potential,
        expected,
    )


def test_square_well_rejects_nonpositive_width():
    """The well width must be positive."""

    z = np.linspace(-1.0, 1.0, 11)

    with pytest.raises(ValueError):
        square_well_potential(
            z=z,
            well_width=0.0,
            well_depth=0.3,
        )


def test_square_well_rejects_nonpositive_depth():
    """The well depth must be supplied as a positive number."""

    z = np.linspace(-1.0, 1.0, 11)

    with pytest.raises(ValueError):
        square_well_potential(
            z=z,
            well_width=1.0,
            well_depth=0.0,
        )
