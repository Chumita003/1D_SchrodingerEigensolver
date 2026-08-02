"""
Regression tests for Eigensolver_1Dimension.py.

These don't prove the solver is bug-free, they just pin down the accuracy levels
documented in the README (and verified by hand in validate_1d.py) so that a future
change to d2dx2_matrix, Schrodinger_solver, or any of the potentials gets caught
immediately instead of silently drifting. Thresholds are set with headroom above the
measured errors, not at the theoretical best case - see the README for what each number
actually means and why the finite well / delta well thresholds are looser than the
harmonic oscillator's.

Run: pytest
"""

import numpy as np
import pytest
from functools import partial
from scipy.sparse import diags

from Eigensolver_1Dimension import (
    Schrodinger_solver,
    d2dx2_matrix,
    V_SingleWell,
    V_HarmonicOscillator,
    V_AnharmonicOscillator,
    V_InfiniteSquareWell,
    V_FiniteSquareWell_CellAveraged,
    V_LinearPotential,
    V_SoftCoulomb,
    V_DoubleWell,
    V_DeltaDiscrete,
)
from validate_1d import (
    validate_infinite_square_well,
    validate_harmonic_oscillator,
    validate_finite_square_well,
    validate_finite_square_well_cellaveraged,
    convergence_study_finite_well,
    validate_delta_well,
)


def test_infinite_square_well_matches_analytic():
    # Measured ~1e-11 at N=2000, limited by the shift-invert eigensolve rather than by the
    # discretization. Before the odd-extension boundary closure this sat at ~1.5e-4, so the
    # threshold here is deliberately tight: it is the regression guard for that closure.
    _, _, _, err = validate_infinite_square_well(N=2000)
    assert np.all(err < 1e-8)


def test_infinite_square_well_converges_at_fourth_order():
    # Direct check that the boundary closure restores the design order of the stencil.
    # Measured local slopes: 4.17, 4.11, 4.08, 4.05, 4.03, 4.01. Without the closure these
    # would all sit at ~1.0. Bounds are loose enough to absorb the usual preasymptotic
    # wobble but far away from the first-order failure mode.
    from validate_1d import convergence_study_isw
    Ns, errs = convergence_study_isw()
    slopes = -np.log(errs[1:] / errs[:-1]) / np.log(Ns[1:] / Ns[:-1])
    assert np.all(slopes > 3.5)
    assert np.all(slopes < 4.6)


def test_harmonic_oscillator_matches_analytic():
    # Measured ~1e-9 to 1e-10, essentially machine precision for this problem.
    _, _, _, err = validate_harmonic_oscillator(N=2000)
    assert np.all(err < 1e-8)


def test_finite_square_well_matches_semianalytic():
    # Measured ~8e-4 (grid resolution near the V(x) discontinuity, see README). Note this
    # one is NOT improved by the boundary closure: the error comes from sampling the step
    # in V pointwise onto the grid, not from the stencil, so it stays near first order.
    _, _, _, err = validate_finite_square_well(N=2500)
    assert np.all(err < 2e-3)


def test_cell_averaged_finite_well_beats_pointwise():
    # Same well, same solver, only the sampling of V changes. Cell averaging keeps the
    # sub-grid position of the wall instead of snapping it to the nearest grid point,
    # which is worth roughly a factor of 30 at this resolution.
    _, _, _, err_point = validate_finite_square_well(N=2500)
    _, _, _, err_cell = validate_finite_square_well_cellaveraged(N=2500)
    assert np.all(err_cell < err_point)
    assert np.all(err_cell < 1e-4)


def test_cell_averaged_finite_well_converges_at_second_order():
    # The point of the whole exercise: pointwise sampling caps the finite well near p = 1,
    # cell averaging lifts it to p = 2. Measured slopes are 1.25/1.15/1.08/1.04 and
    # 2.39/2.23/2.13/2.07 respectively, both drifting toward their asymptote from above.
    # Second order is the ceiling because psi is only C^1 at the wall, so no amount of
    # better sampling recovers the stencil's fourth order there.
    Ns, e_point, e_cell = convergence_study_finite_well()
    r = np.log(Ns[1:] / Ns[:-1])
    p_point = np.log(e_point[:-1] / e_point[1:]) / r
    p_cell = np.log(e_cell[:-1] / e_cell[1:]) / r
    assert np.all(p_point < 1.4)
    assert np.all(p_cell > 1.9)
    assert np.all(p_cell < 2.6)


def test_delta_well_matches_analytic():
    # Measured ~2.8e-2 at N=2000 (discrete delta is a coarse stand-in, see README).
    # Loose threshold on purpose - this potential is not meant to be precise.
    _, _, err = validate_delta_well(N=2000)
    assert err < 5e-2


def test_eigenvalues_are_sorted_and_nondegenerate_ordering():
    _, eigvals, _, _ = validate_harmonic_oscillator(N=2000)
    assert np.all(np.diff(eigvals) > 0)


def test_nth_eigenfunction_has_n_nodes():
    # Sturm-Liouville oscillation theorem: the n-th excited state has exactly n nodes.
    # Checked here on a potential with no closed-form spectrum (quartic single well),
    # so this is purely a structural sanity check, not an accuracy check.
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=partial(V_SingleWell, V0=2.0),
        x_min=-6.0, x_max=6.0, N=1500, num_eigvals=6,
    )
    for n in range(6):
        psi = eigvecs[:, n]
        significant = psi[np.abs(psi) > 1e-3 * np.max(np.abs(psi))]
        nodes = int(np.sum(np.sign(significant)[1:] != np.sign(significant)[:-1]))
        assert nodes == n


# ---------------------------------------------------------------------------
# Structural checks: properties the solver must satisfy for *any* potential,
# including the five that have no closed-form spectrum to compare against.
# ---------------------------------------------------------------------------

def _rebuild_H(V_fun, x_min, x_max, N, hbar=1.0, m=1.0):
    '''
    Reassemble the interior Hamiltonian the same way Schrodinger_solver does.

    Note this is a *consistency* check, not an independent verification: if the
    assembly itself were wrong, this helper would reproduce the same error and the
    residual test below would still pass. Independent verification is what the
    analytic comparisons above provide. What this catches is everything that happens
    to the eigenpairs *after* eigsh returns them - the sorting, the zero padding at
    the boundaries, the renormalization and the sign fix - none of which ARPACK saw.
    '''
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    x_int = x[1:-1]
    H = -(hbar ** 2 / (2 * m)) * d2dx2_matrix(x_int.size, dx)
    return H + diags(np.asarray(V_fun(x_int), dtype=float), 0, format='csr'), dx


# One entry per potential, including the ones with no analytic spectrum, which is
# the whole point: these are the only quantitative checks those five ever get.
_STRUCTURAL_CASES = [
    ("harmonic", partial(V_HarmonicOscillator, omega=1.0, m=1.0), -8.0, 8.0, 600),
    ("anharmonic", partial(V_AnharmonicOscillator, a=1.0, b=0.05), -8.0, 8.0, 600),
    ("infinite well", V_InfiniteSquareWell, 0.0, 10.0, 600),
    ("finite well", partial(V_FiniteSquareWell_CellAveraged, L=6.0, V0=30.0,
                            well_centered=True), -8.0, 8.0, 600),
    ("linear", partial(V_LinearPotential, F=1.0), -10.0, 10.0, 600),
    ("soft Coulomb", partial(V_SoftCoulomb, Z=1.0, eps=0.5), -20.0, 20.0, 800),
    ("quartic single well", partial(V_SingleWell, V0=2.0), -6.0, 6.0, 600),
    ("quartic double well", partial(V_DoubleWell, a=1.5, V0=5.0), -6.0, 6.0, 600),
    ("discrete delta", partial(V_DeltaDiscrete, alpha=5.0, x0=0.0), -20.0, 20.0, 800),
]


@pytest.mark.parametrize("name,V_fun,x_min,x_max,N", _STRUCTURAL_CASES)
def test_hamiltonian_is_symmetric(name, V_fun, x_min, x_max, N):
    # The regression guard for the entire boundary-closure story. The odd extension
    # was chosen over the one-sided 4th-order rows precisely because it only touches
    # the diagonal and therefore cannot break H = H^dagger. If a future change
    # reintroduces asymmetric boundary rows, eigsh would silently start returning
    # complex or unordered eigenvalues; this fires first.
    H, _ = _rebuild_H(V_fun, x_min, x_max, N)
    asymmetry = abs(H - H.T).max()
    assert asymmetry == 0.0, f"{name}: H is not symmetric, max |H - H^T| = {asymmetry}"


@pytest.mark.parametrize("name,V_fun,x_min,x_max,N", _STRUCTURAL_CASES)
def test_eigenpairs_satisfy_the_eigenvalue_equation(name, V_fun, x_min, x_max, N):
    # ||H psi - E psi|| / ||psi||, checked on the arrays the solver actually hands
    # back rather than on eigsh's raw output. Catches a mismatch between eigvals and
    # eigvecs after the argsort, and any damage done by the padding, renormalization
    # or sign convention.
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=V_fun, x_min=x_min, x_max=x_max, N=N, num_eigvals=5,
    )
    H, _ = _rebuild_H(V_fun, x_min, x_max, N)

    for n in range(eigvals.size):
        psi = eigvecs[1:-1, n]                      # strip the padded boundary zeros
        residual = np.linalg.norm(H @ psi - eigvals[n] * psi) / np.linalg.norm(psi)
        scale = max(abs(eigvals[n]), 1.0)           # relative to the eigenvalue scale
        assert residual / scale < 1e-8, (
            f"{name}, n={n}: residual {residual:.3e} is too large for E={eigvals[n]:.6f}"
        )


@pytest.mark.parametrize("name,V_fun,x_min,x_max,N", _STRUCTURAL_CASES)
def test_eigenfunctions_are_orthonormal(name, V_fun, x_min, x_max, N):
    # <psi_m|psi_n> = delta_mn under the same discrete inner product the solver uses
    # to normalize, sum(psi_m psi_n) * dx. Orthogonality is not something the solver
    # enforces by hand, it is inherited from H being symmetric, so this is really a
    # second, independent way of detecting a broken Hamiltonian.
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot=V_fun, x_min=x_min, x_max=x_max, N=N, num_eigvals=5,
    )
    dx = x[1] - x[0]
    gram = (eigvecs.T @ eigvecs) * dx
    assert np.allclose(gram, np.eye(eigvals.size), atol=1e-8), (
        f"{name}: max deviation from identity = {np.abs(gram - np.eye(eigvals.size)).max():.3e}"
    )
