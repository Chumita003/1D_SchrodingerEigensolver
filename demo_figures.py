"""
Regenerates every figure used in the README:

  figures/wavefunctions_harmonic.png   - QHO eigenfunctions over V(x)
  figures/energy_levels_harmonic.png   - QHO energy-level diagram
  figures/wavefunctions_doublewell.png - quartic double well eigenfunctions over V(x)
  figures/energy_levels_doublewell.png - double well energy-level diagram (tunneling
                                          doublets)

Run: python demo_figures.py
"""

import matplotlib.pyplot as plt
from functools import partial

from Eigensolver_1Dimension import (
    Schrodinger_solver,
    V_HarmonicOscillator,
    V_DoubleWell,
    plot_wavefunctions,
    plot_energy_levels,
)

if __name__ == "__main__":
    # --- Harmonic oscillator ---
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=partial(V_HarmonicOscillator, omega=1.0, m=1.0),
        x_min=-8.0, x_max=8.0, N=2000, num_eigvals=6,
    )

    plot_wavefunctions(
        x, eigvals, eigvecs, partial(V_HarmonicOscillator, omega=1.0, m=1.0),
        n_states=6, scale=1.0, x_range=(-5, 5), y_range=(-0.5, 6.5),
        title="Harmonic oscillator: eigenfunctions over V(x)",
    )
    plt.savefig("figures/wavefunctions_harmonic.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_energy_levels(eigvals, n_states=6, title="Harmonic oscillator: energy levels")
    plt.savefig("figures/energy_levels_harmonic.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Quartic double well ---
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=partial(V_DoubleWell, a=1.5, V0=7.0),
        x_min=-4.0, x_max=4.0, N=3000, num_eigvals=6,
    )

    ax = plot_wavefunctions(
        x, eigvals, eigvecs, partial(V_DoubleWell, a=1.5, V0=7.0),
        n_states=6, scale=3.0, x_range=(-3.2, 3.2), y_range=(-2, 29),
        title="Quartic double well: eigenfunctions over V(x)",
    )
    ax.legend(loc="lower right", fontsize=8)
    plt.savefig("figures/wavefunctions_doublewell.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_energy_levels(eigvals, n_states=6, title="Double well: energy levels")
    plt.savefig("figures/energy_levels_doublewell.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved 4 figures to figures/")
    print(f"Tunneling splitting n=0,1: {eigvals[1]-eigvals[0]:.3e}")
    print(f"Tunneling splitting n=2,3: {eigvals[3]-eigvals[2]:.3e}")
