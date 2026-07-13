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
    validate_delta_well,
)


def test_infinite_square_well_matches_analytic():
    # Measured ~1.5e-4 (boundary-stencil limitation, see README). Generous margin.
    _, _, _, err = validate_infinite_square_well(N=2000)
    assert np.all(err < 2e-4)


def test_harmonic_oscillator_matches_analytic():
    # Measured ~1e-9 to 1e-10, essentially machine precision for this problem.
    _, _, _, err = validate_harmonic_oscillator(N=2000)
    assert np.all(err < 1e-8)


def test_finite_square_well_matches_semianalytic():
    # Measured ~8e-4 (grid resolution near the V(x) discontinuity, see README).
    _, _, _, err = validate_finite_square_well(N=2500)
    assert np.all(err < 2e-3)


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
        V_pot=partial(V_SingleWell, a=1.0, V0=2.0),
        x_min=-6.0, x_max=6.0, N=1500, num_eigvals=6,
    )
    for n in range(6):
        psi = eigvecs[:, n]
        significant = psi[np.abs(psi) > 1e-3 * np.max(np.abs(psi))]
        nodes = int(np.sum(np.sign(significant)[1:] != np.sign(significant)[:-1]))
        assert nodes == n
