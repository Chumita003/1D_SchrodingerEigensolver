"""
Validation for Eigensolver_1Dimension.py

Compares the numerical eigenvalues against the two potentials that have a closed-form
spectrum, and reports the relative error per level:

  - Infinite square well:   E_n = n^2 pi^2 hbar^2 / (2 m L^2),   n = 1, 2, 3, ...
  - Harmonic oscillator:    E_n = hbar * omega * (n + 1/2),      n = 0, 1, 2, ...

It also reproduces the convergence-order plot referenced in the README: for the
infinite square well, the ground-state relative error is measured as a function of the
number of grid points N and compared against an O(1/N) reference line. This is the
direct numerical evidence for the boundary-stencil limitation documented in
d2dx2_matrix's docstring in Eigensolver_1Dimension.py: the two rows adjacent to each
Dirichlet boundary use a truncated (not one-sided-corrected) 5-point stencil to keep the
Hamiltonian exactly Hermitian, which locally reduces accuracy from O(dx^4) to O(dx^2) at
those two rows and shows up as O(dx) global convergence for any state with non-negligible
amplitude/slope at the boundary. The harmonic oscillator is essentially unaffected
because its eigenfunctions have decayed to ~0 well before the domain edge.

Run: python validate_1d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from Eigensolver_1Dimension import (
    Schrodinger_solver,
    V_HarmonicOscillator,
    V_InfiniteSquareWell,
)


def validate_infinite_square_well(L=10.0, N=2000, num_eigvals=6, hbar=1.0, m=1.0):
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=V_InfiniteSquareWell,
        L=L, well_centered=False,
        N=N, hbar=hbar, m=m,
        num_eigvals=num_eigvals,
    )
    ns = np.arange(1, num_eigvals + 1)
    analytic = (ns ** 2 * np.pi ** 2 * hbar ** 2) / (2 * m * L ** 2)
    rel_err = np.abs(eigvals - analytic) / analytic
    return ns, eigvals, analytic, rel_err


def validate_harmonic_oscillator(omega=1.0, m=1.0, hbar=1.0, x_min=-8.0, x_max=8.0,
                                  N=2000, num_eigvals=6):
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=partial(V_HarmonicOscillator, omega=omega, m=m),
        x_min=x_min, x_max=x_max,
        N=N, hbar=hbar, m=m,
        num_eigvals=num_eigvals,
    )
    ns = np.arange(0, num_eigvals)
    analytic = hbar * omega * (ns + 0.5)
    rel_err = np.abs(eigvals - analytic) / analytic
    return ns, eigvals, analytic, rel_err


def print_table(name, ns, numeric, analytic, rel_err):
    print(f"\n{name}")
    print(f"{'n':>3} {'numeric':>14} {'analytic':>14} {'rel. error':>12}")
    for n, num, an, err in zip(ns, numeric, analytic, rel_err):
        print(f"{n:>3} {num:>14.8f} {an:>14.8f} {err:>12.3e}")


def convergence_study_isw(L=10.0, Ns=(200, 400, 800, 1600, 3200), hbar=1.0, m=1.0):
    errs = []
    analytic_gs = (np.pi ** 2 * hbar ** 2) / (2 * m * L ** 2)
    for N in Ns:
        _, eigvals, _ = Schrodinger_solver(
            V_pot=V_InfiniteSquareWell, L=L, N=N, hbar=hbar, m=m, num_eigvals=1,
        )
        errs.append(abs(eigvals[0] - analytic_gs) / analytic_gs)
    return np.array(Ns), np.array(errs)


if __name__ == "__main__":
    ns, num, an, err = validate_infinite_square_well()
    print_table("Infinite square well (L=10, hbar=m=1)", ns, num, an, err)

    ns, num, an, err = validate_harmonic_oscillator()
    print_table("Harmonic oscillator (omega=m=hbar=1)", ns, num, an, err)

    Ns, errs = convergence_study_isw()
    print("\nInfinite square well, ground-state convergence vs N:")
    for N, e in zip(Ns, errs):
        print(f"N={N:>5}   rel. error={e:.3e}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(Ns, errs, "o-", label="ground state, measured")
    ax.loglog(Ns, errs[0] * Ns[0] / Ns, "--", color="gray", label=r"$O(1/N)$ reference")
    ax.set_xlabel("N (grid points)")
    ax.set_ylabel("relative error")
    ax.set_title("Infinite square well: convergence order")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/convergence_isw.png", dpi=150)
    print("\nSaved figures/convergence_isw.png")
