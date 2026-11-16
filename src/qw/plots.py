# src/qw/plots.py

"""Plotting functions for quantum-well calculations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Plot settings
# ============================================================

# If True:
#     save plots to the calculation output directory.
#
# If False:
#     display plots on screen and do not save them.
save = True

# Resolution of saved figures
dpi = 200


# ============================================================
# Plot handling
# ============================================================

def _finish_plot(
    fig,
    filename,
    output_dir=None,
):
    """
    Save or display a figure depending on the global save setting.
    """

    fig.tight_layout()

    if save:

        if output_dir is None:
            raise ValueError(
                "output_dir must be provided when save=True."
            )

        output_dir = Path(
            output_dir
        )

        figure_path = (
            output_dir
            / filename
        )

        fig.savefig(
            figure_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        print(
            f"Saved plot: "
            f"{figure_path}"
        )

    else:

        plt.show()


# ============================================================
# Wavefunctions
# ============================================================

def plot_wavefunctions(
    result,
    wavefunction_scale=0.02,
    output_dir=None,
):
    """
    Plot the total potential, eigenenergies, and wavefunctions.

    Each wavefunction is normalized to its maximum amplitude for
    visualization and shifted vertically to its eigenenergy.

    Parameters
    ----------
    result : SolverResult
        Quantum-well calculation result.

    wavefunction_scale : float, optional
        Vertical plotting scale in eV.

    output_dir : path_like, optional
        Directory where the figure is saved when save=True.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Total potential
    # --------------------------------------------------------

    ax.plot(
        result.z,
        result.total_potential,
        label="Potential",
    )

    # --------------------------------------------------------
    # Eigenstates
    # --------------------------------------------------------

    for state, energy in enumerate(
        result.energies
    ):

        psi = (
            result.wavefunctions[
                :,
                state,
            ]
        )

        psi_max = np.max(
            np.abs(
                psi
            )
        )

        if psi_max > 0.0:

            psi_plot = (
                psi
                / psi_max
            )

        else:

            psi_plot = psi

        ax.plot(
            result.z,
            energy
            + wavefunction_scale
            * psi_plot,
            label=rf"$\psi_{state}$",
        )

        ax.axhline(
            energy,
            linewidth=0.8,
            linestyle="--",
        )

    ax.set_xlabel(
        r"$z$ ($\AA$)"
    )

    ax.set_ylabel(
        "Energy (eV)"
    )

    ax.set_title(
        "Quantum well wavefunctions"
    )

    # Place legend outside the plotting area
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    _finish_plot(
        fig=fig,
        filename="wavefunctions.png",
        output_dir=output_dir,
    )

    return fig, ax


# ============================================================
# Probability densities
# ============================================================

def plot_densities(
    result,
    density_scale=0.02,
    output_dir=None,
):
    """
    Plot the total potential and probability densities.

    Each probability density is normalized to its maximum value
    for visualization and shifted vertically to its eigenenergy.

    Parameters
    ----------
    result : SolverResult
        Quantum-well calculation result.

    density_scale : float, optional
        Vertical plotting scale in eV.

    output_dir : path_like, optional
        Directory where the figure is saved when save=True.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Total potential
    # --------------------------------------------------------

    ax.plot(
        result.z,
        result.total_potential,
        label="Potential",
    )

    # --------------------------------------------------------
    # Probability densities
    # --------------------------------------------------------

    for state, energy in enumerate(
        result.energies
    ):

        psi = (
            result.wavefunctions[
                :,
                state,
            ]
        )

        density = (
            np.abs(
                psi
            ) ** 2
        )

        density_max = np.max(
            density
        )

        if density_max > 0.0:

            density_plot = (
                density
                / density_max
            )

        else:

            density_plot = density

        ax.plot(
            result.z,
            energy
            + density_scale
            * density_plot,
            label=rf"$|\psi_{state}|^2$",
        )

        ax.axhline(
            energy,
            linewidth=0.8,
            linestyle="--",
        )

    ax.set_xlabel(
        r"$z$ ($\AA$)"
    )

    ax.set_ylabel(
        "Energy (eV)"
    )

    ax.set_title(
        "Quantum well probability densities"
    )

    # Place legend outside the plotting area
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    _finish_plot(
        fig=fig,
        filename="densities.png",
        output_dir=output_dir,
    )

    return fig, ax