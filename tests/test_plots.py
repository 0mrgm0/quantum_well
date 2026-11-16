# tests/test_plots.py

"""Tests for quantum-well plotting functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import qw.plots as plots

from qw.grid import make_grid
from qw.solver import solve_schrodinger
from qw.well import square_well_potential


# ============================================================
# Test calculation
# ============================================================

def make_plot_result():
    """Create a small quantum-well result for plotting tests."""

    z, _ = make_grid(
        z_min=-400.0,
        z_max=400.0,
        n_grid_points=1601,
    )

    potential = square_well_potential(
        z=z,
        well_width=200.0,
        well_depth=0.300,
        center=0.0,
    )

    result = solve_schrodinger(
        z=z,
        mass=0.067,
        potential=potential,
        electric_field=0.0,
        charge=-1.0,
        n_states=3,
        d2_stencil_points=3,
        solver="tridiagonal",
    )

    return result


# ============================================================
# Wavefunction plot
# ============================================================

def test_wavefunction_plot_is_created_and_saved(
    diagnostic_plot_dir,
    monkeypatch,
):
    """The wavefunction plotting function should save a valid figure."""

    # Force save mode regardless of the user's setting in plots.py.
    monkeypatch.setattr(
        plots,
        "save",
        True,
    )

    result = make_plot_result()

    fig, ax = plots.plot_wavefunctions(
        result,
        wavefunction_scale=0.02,
        output_dir=diagnostic_plot_dir,
    )

    plot_path = (
        diagnostic_plot_dir
        / "wavefunctions.png"
    )

    assert fig is not None
    assert ax is not None

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


# ============================================================
# Density plot
# ============================================================

def test_density_plot_is_created_and_saved(
    diagnostic_plot_dir,
    monkeypatch,
):
    """The density plotting function should save a valid figure."""

    # Force save mode regardless of the user's setting in plots.py.
    monkeypatch.setattr(
        plots,
        "save",
        True,
    )

    result = make_plot_result()

    fig, ax = plots.plot_densities(
        result,
        density_scale=0.02,
        output_dir=diagnostic_plot_dir,
    )

    plot_path = (
        diagnostic_plot_dir
        / "densities.png"
    )

    assert fig is not None
    assert ax is not None

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


# ============================================================
# Display mode
# ============================================================

def test_plots_can_be_shown_without_saving(
    tmp_path,
    monkeypatch,
):
    """With save=False, plots should be displayed instead of saved."""

    monkeypatch.setattr(
        plots,
        "save",
        False,
    )

    show_calls = []

    def fake_show():
        show_calls.append(
            True
        )

    monkeypatch.setattr(
        plt,
        "show",
        fake_show,
    )

    result = make_plot_result()

    plots.plot_wavefunctions(
        result,
        wavefunction_scale=0.02,
        output_dir=tmp_path,
    )

    plots.plot_densities(
        result,
        density_scale=0.02,
        output_dir=tmp_path,
    )

    assert len(show_calls) == 2

    assert not (
        tmp_path
        / "wavefunctions.png"
    ).exists()

    assert not (
        tmp_path
        / "densities.png"
    ).exists()