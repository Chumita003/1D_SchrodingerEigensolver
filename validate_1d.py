"""
Validation for Eigensolver_1Dimension.py

Compares the numerical eigenvalues against potentials that have a closed-form or
semi-analytic spectrum, and reports the relative error per level:

  - Infinite square well:   E_n = n^2 pi^2 hbar^2 / (2 m L^2),   n = 1, 2, 3, ...
  - Harmonic oscillator:    E_n = hbar * omega * (n + 1/2),      n = 0, 1, 2, ...
  - Finite square well:     semi-analytic, roots of k*tan(kL/2)=kappa (even) and
                            k*cot(kL/2)=-kappa (odd), solved here by bisection.
  - Discrete delta well:    E = -m*alpha^2 / (2*hbar^2), the one exact bound state.

The finite well and delta well each carry their own accuracy caveat, unrelated to the
boundary closure described below - see the README section "Two more precision notes".

It also reproduces the convergence-order plot referenced in the README: for the
infinite square well, the ground-state relative error is measured as a function of the
number of grid points N and compared against an O(1/N^4) reference line. This is the
direct numerical evidence for the boundary closure documented in d2dx2_matrix's docstring
in Eigensolver_1Dimension.py: the two rows adjacent to each Dirichlet boundary close the
5-point stencil with the odd extension psi_{-1} = -psi_1, which is a diagonal-only
correction (so H stays exactly Hermitian) and restores the design order of the stencil.

Historical note, kept because the README references it: before that closure the two
boundary rows simply dropped the out-of-domain term, which locally degraded accuracy to
O(dx^2) and showed up as O(dx) *global* convergence for any state with non-negligible
amplitude/slope at the boundary. The infinite square well sat at ~1e-3 relative error and
p = 1.00; it now sits at ~1e-11 with p ~ 4.0. The harmonic oscillator was, and still is,
unaffected either way because its eigenfunctions decay to ~0 well before the domain edge.

What is still NOT fixed by the boundary closure, and is visible in the tables below:
  - the finite square well stays at ~8e-4 with roughly first-order convergence, because
    the jump in V is sampled pointwise onto the grid. That is a potential-sampling error,
    not a stencil error.
  - the discrete delta well stays at ~3e-2 for the reason described in validate_delta_well.

Run: python validate_1d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from Eigensolver_1Dimension import (
    Schrodinger_solver,
    V_HarmonicOscillator,
    V_InfiniteSquareWell,
    V_FiniteSquareWell,
    V_DeltaDiscrete,
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


def _bisect(f, a, b, tol=1e-12, maxit=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    for _ in range(maxit):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or (b - a) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def _finite_well_analytic_levels(L, V0, m=1.0, hbar=1.0, n_scan=200000):
    '''
    Semi-analytic bound-state energies for the symmetric finite square well, solving the
    even/odd transcendental equations by bisection: k*tan(k*L/2) = kappa (even states),
    k*cot(k*L/2) = -kappa (odd states), with k = sqrt(2mE)/hbar, kappa = sqrt(2m(V0-E))/hbar.
    No scipy.optimize dependency, just a hand-rolled bisection sign-change scan.
    '''
    def k_(E): return np.sqrt(2 * m * E) / hbar
    def kap_(E): return np.sqrt(2 * m * (V0 - E)) / hbar

    def f_even(E):
        k, kap = k_(E), kap_(E)
        return k * np.tan(k * L / 2) - kap

    def f_odd(E):
        k, kap = k_(E), kap_(E)
        return k / np.tan(k * L / 2) + kap

    eps = 1e-6
    Es = np.linspace(eps, V0 - eps, n_scan)
    roots = []
    for f in (f_even, f_odd):
        vals = f(Es)
        for i in range(len(Es) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if np.isfinite(v0) and np.isfinite(v1) and v0 * v1 < 0 and abs(v0) < 50 and abs(v1) < 50:
                r = _bisect(f, Es[i], Es[i + 1])
                if r is not None:
                    roots.append(r)
    return np.array(sorted(roots))


def validate_finite_square_well(L=10.0, V0=50.0, N=2500, num_eigvals=None, hbar=1.0, m=1.0):
    analytic = _finite_well_analytic_levels(L, V0, m=m, hbar=hbar)
    if num_eigvals is None:
        num_eigvals = len(analytic)
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=partial(V_FiniteSquareWell, L=L, V0=V0, well_centered=True),
        x_min=-10.0, x_max=10.0, N=N, hbar=hbar, m=m, num_eigvals=num_eigvals,
    )
    analytic = analytic[:num_eigvals]
    rel_err = np.abs(eigvals - analytic) / analytic
    return np.arange(num_eigvals), eigvals, analytic, rel_err


def validate_delta_well(alpha=5.0, N=2000, hbar=1.0, m=1.0):
    '''
    Exact single bound state of the continuum delta well: E = -m*alpha^2 / (2*hbar^2).
    The discrete implementation (V_DeltaDiscrete) represents the delta as a single grid
    spike, which is a much coarser approximation than the finite-difference stencil
    itself - see the README for why this one converges slower than everything else here.
    '''
    analytic = -m * alpha ** 2 / (2 * hbar ** 2)
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=partial(V_DeltaDiscrete, alpha=alpha, x0=0.0),
        x_min=-20.0, x_max=20.0, N=N, hbar=hbar, m=m, num_eigvals=1,
    )
    rel_err = abs(eigvals[0] - analytic) / abs(analytic)
    return eigvals[0], analytic, rel_err


def print_table(name, ns, numeric, analytic, rel_err):
    print(f"\n{name}")
    print(f"{'n':>3} {'numeric':>14} {'analytic':>14} {'rel. error':>12}")
    for n, num, an, err in zip(ns, numeric, analytic, rel_err):
        print(f"{n:>3} {num:>14.8f} {an:>14.8f} {err:>12.3e}")


def convergence_study_isw(L=10.0, Ns=(20, 30, 45, 65, 100, 150, 220), hbar=1.0, m=1.0):
    '''
    Ground state of the infinite square well vs grid size.

    The N values are deliberately small. With the odd-extension boundary closure in place
    the error reaches ~5e-10 by N = 220 and then flattens out around 1e-10 to 1e-12, where
    the shift-invert eigensolve - not the discretization - is the accuracy floor. Pushing
    to N = 3200 therefore measures ARPACK, not the stencil. In the range used here the
    measured local slope is 4.17, 4.11, 4.08, 4.05, 4.03, 4.01, converging to the expected
    p = 4.
    '''
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

    ns, num, an, err = validate_finite_square_well()
    print_table("Finite square well (L=10, V0=50, centered)", ns, num, an, err)

    num, an, err = validate_delta_well()
    print(f"\nDiscrete delta well (alpha=5): numeric={num:.6f}  analytic={an:.6f}  rel_err={err:.3e}")

    Ns, errs = convergence_study_isw()
    slopes = -np.diff(np.log(errs)) / np.diff(np.log(Ns))
    print("\nInfinite square well, ground-state convergence vs N:")
    for i, (N, e) in enumerate(zip(Ns, errs)):
        slope = f"   local slope p={slopes[i-1]:.2f}" if i > 0 else ""
        print(f"N={N:>5}   rel. error={e:.3e}{slope}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(Ns, errs, "o-", label="ground state, measured")
    ax.loglog(Ns, errs[0] * (Ns[0] / Ns) ** 4, "--", color="gray", label=r"$O(1/N^4)$ reference")
    ax.set_xlabel("N (grid points)")
    ax.set_ylabel("relative error")
    ax.set_title("Infinite square well: convergence order")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/convergence_isw.png", dpi=150)
    print("\nSaved figures/convergence_isw.png")
