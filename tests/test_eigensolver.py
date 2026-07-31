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
from functools import partial

from Eigensolver_1Dimension import Schrodinger_solver, V_SingleWell
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
