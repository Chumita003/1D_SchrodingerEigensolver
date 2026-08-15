import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from functools import partial
import matplotlib.pyplot as plt

def d2dx2_matrix(N, dx):

    '''
    This function approximates to (d^2/dx^2) using a 5-point central finite-difference on
    interior points. Dirichlet boundary conditions were assumed in order to construct the
    Hamiltonian matrix Hermitian.
    '''

    '''
    Boundary treatment (odd/antisymmetric extension).

    Label the interior points x_1 ... x_N, with Dirichlet boundaries at x_0 and x_{N+1}
    where psi = 0. The row for i = 1 reads

        (-psi_{-1} + 16 psi_0 - 30 psi_1 + 16 psi_2 - psi_3) / (12 dx^2),

    and while psi_0 = 0 is legitimate (Dirichlet), psi_{-1} sits at x_0 - dx, one point
    *outside* the domain, and is undefined. Setting it to zero - the naive reading of the
    uniform stencil - is wrong and is what used to cap this solver at O(dx) globally.

    The correct closure comes from the equation itself, not from a one-sided formula.
    Since psi'' = (2m/hbar^2)(V - E) psi and psi(x_0) = 0, we get psi''(x_0) = 0 as well:
    the low-order even derivatives vanish at a Dirichlet boundary, so psi is odd about
    x_0 and psi_{-1} = -psi_1. Substituting, the term -psi_{-1} becomes +psi_1, which
    adds +1/(12 dx^2) to the *diagonal* entry of that row (i.e. -30 -> -29 in units of
    1/(12 dx^2)). The same argument applies at i = N.

    Because the correction touches only the diagonal, the matrix stays exactly symmetric,
    so H = H^dagger is preserved and eigsh still returns real, ordered eigenvalues. This
    is what makes it usable, unlike the one-sided 4th-order rows I originally derived by
    hand (see Eigensolver_1Dimensional.pdf), which do break the symmetry: the coefficient
    a one-sided row assigns to its neighbor does not match the coefficient that neighbor's
    central row assigns back.

    Measured effect on the infinite square well (L = 10, ground state, relative error):
    1.6e-3 -> 6.9e-10 at N = 200, and the observed convergence order goes from p = 1.00 to
    p ~ 4.0, until the shift-invert solve itself becomes the accuracy floor around 1e-10
    to 1e-12. The harmonic oscillator is unchanged to machine precision (its
    eigenfunctions have decayed to ~0 well before the domain edge, so psi_{-1} ~ 0 either
    way). See validate_1d.py.

    Residual limitation: the odd extension is exact only when V'(x_0) = 0. Differentiating
    psi'' = (2m/hbar^2)(V - E) psi twice and evaluating at a Dirichlet boundary (where
    psi = psi'' = 0) leaves the 4th derivative d4psi/dx4 = (4m/hbar^2) V'(x_0) psi'(x_0),
    which is the first even derivative that need not vanish. With V'(x_0) != 0 a local
    O(dx^2) truncation error therefore survives in those two rows. In practice this is
    harmless: if V has a non-negligible slope at the domain edge, the domain is too small
    anyway.

    Separately, and unrelated to the boundary: potentials with a jump discontinuity in the
    interior are limited to ~O(dx) if V is sampled pointwise, which is a property of the
    sampling and not of this stencil. V_FiniteSquareWell_CellAveraged removes that term
    and reaches O(dx^2); second order is then the ceiling, since psi is only C^1 at the
    jump and no sampling scheme can supply derivatives the solution does not have.
    '''

    '''
    N: number of grid points
    dx: distance between grid points
    '''

    if N < 5:
        raise ValueError("N must be at least 5 interior points for the 5-point stencil.")
    if dx <= 0:
        raise ValueError("dx must be positive.")

    # Constructing the second derivative matrix coefficients
    coeffs = np.array([-1.00, 16.00, -30.00, 16.00, -1.00]) / (12.00 * (dx**2))
    offsets = np.array([-2, -1, 0, 1, 2])

    # Odd-extension closure at the two Dirichlet boundaries: psi_{-1} = -psi_1 turns the
    # -psi_{-1} term into +psi_1, i.e. -30 -> -29 in units of 1/(12 dx^2). Diagonal-only,
    # so the matrix stays symmetric. See the docstring above for the derivation.
    main = coeffs[2] * np.ones(N)
    main[0] += 1.00 / (12.00 * (dx**2))
    main[-1] += 1.00 / (12.00 * (dx**2))

    d2_matrix = diags(
        diagonals=[
            coeffs[0] * np.ones(N - 2),
            coeffs[1] * np.ones(N - 1),
            main,
            coeffs[3] * np.ones(N - 1),
            coeffs[4] * np.ones(N - 2),
        ],
        offsets=offsets,
        shape=(N, N),
        format='csr'
    )

    return d2_matrix

def Schrodinger_solver(
    V_pot,
    x_min = -10.0,
    x_max = 10.0,
    L = None, 
    well_centered = False,
    N = 1800,
    hbar = 1.0,
    m = 1.0,
    num_eigvals = 10,
    ):

    '''
    This function solves the time-independent Schrodinger equation for a given potential V(x) using
    the finite difference method.
    '''

    '''
    V_pot: potential function
    If L is provided, the domain is set from L:
      - if well_centered: x in [-L/2, L/2]
      - else:            x in [0, L]
    Otherwise uses x_min, x_max as given.
    x_min: minimum x value (default: -10.0)
    x_max: maximum x value (default: 10.0)
    N: number of grid points (default: 1800)
    hbar: reduced Planck's constant (default: 1.0)
    m: mass of the particle (default: 1.0)
    num_eigvals: number of eigenvalues and eigenvectors to compute (default: 10)

    Interface: V_pot must be callable as V_pot(x) and return an array shaped like x.
    For parameterized potentials use `functools.partial` or `lambda` when calling the solver.
    '''

    if N < 5:
        raise ValueError("N must be at least 5 total grid points.")
    if num_eigvals <= 0:
        raise ValueError("num_eigvals must be a positive integer.")

    if L is not None:
        if well_centered:
            x_min = -0.5 * L
            x_max =  0.5 * L
        else:
            x_min = 0.0
            x_max = float(L)
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")

    # Create spatial grid
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0] # distance between grid points

    x_interior = x[1:-1] # interior points for potential evaluation
    N_int = x_interior.size # updating N to reflect interior points
    if N_int < 5:
        raise ValueError("Need at least 5 interior points.")
    if num_eigvals >= N_int:
        raise ValueError(f"num_eigvals must be smaller than N-2 (got {num_eigvals} >= {N_int}).")

    # Construct the kinetic energy matrix calling the second derivative approximation
    d2dx2 = d2dx2_matrix(N_int, dx)
    T = -(hbar**2 / (2 * m)) * d2dx2

    # Construct the potential energy matrix as a diagonal matrix.
    if not callable(V_pot):
        raise ValueError("V_pot must be callable as V_pot(x). For parameterized potentials use functools.partial or a lambda to bind parameters.")

    V_values = np.asarray(V_pot(x_interior), dtype=float)
    if V_values.shape != x_interior.shape:
        raise ValueError(
            "V_pot(x) must return an array with the same shape as x. ",
            f"Received shape {V_values.shape}, expected {x_interior.shape}.",
        )

    V = diags(V_values, offsets = 0, format = 'csr')

    # Construct the Hamiltonian matrix
    H = T + V

    # Computing the eigenvalues and eigenvectors of the Hamiltonian.
    # Shift-invert: target eigenvalues near sigma instead of running plain Lanczos on
    # the full spectrum (which is slow here because T's spectral range is huge). sigma
    # must sit safely below every possible eigenvalue: since H = T + V with T >= 0
    # (the discretized kinetic energy is positive semi-definite), E_0 >= min(V) always,
    # so any sigma < min(V) is guaranteed safe, and "closest to sigma" becomes exactly
    # "the k smallest".
    sigma = float(V_values.min()) - 1.0
    eigvals, eigvecs = eigsh(H, k = num_eigvals, sigma = sigma, which = 'LM')

    # Sorting the eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Normalizing the eigenvectors
    normalized_eigvecs = np.zeros((N, num_eigvals))
    normalized_eigvecs[1:-1, :] = eigvecs

    for i in range(num_eigvals):
        norm = np.sqrt(np.sum(np.abs(normalized_eigvecs[:, i])**2) * dx)
        normalized_eigvecs[:, i] /= norm

        # Fix the arbitrary overall sign eigsh returns (psi and -psi are equally valid).
        # Convention: the point of largest |psi| is positive. Purely cosmetic, makes
        # plots and repeated runs consistent, has no physical meaning.
        #
        # The tie break matters. An antisymmetric state has two extrema of equal magnitude
        # and opposite sign, so |psi| has two global maxima that differ only by rounding -
        # measured separation ~7e-15 relative for n=1 of the harmonic oscillator. Plain
        # argmax picks whichever happens to come out larger, which is not guaranteed to be
        # the same one on the next run. Selecting the lowest index among all near-maximal
        # points removes the ambiguity, because position is exact where the magnitudes are
        # not. This has not been observed to flip in 1D, but it does in 2D on the same
        # construction, so the rule here is written to be safe rather than lucky.
        mag = np.abs(normalized_eigvecs[:, i])
        near_max = np.flatnonzero(mag >= mag.max() * (1.0 - 1e-9))
        if normalized_eigvecs[near_max[0], i] < 0:
            normalized_eigvecs[:, i] *= -1

    return x, eigvals, normalized_eigvecs

## List of potential functions

def V_HarmonicOscillator(x, omega = 1.0, m = 1.0):
    return 0.5 * (m * (omega**2) * (x**2))

def V_AnharmonicOscillator(x, a = 1.0, b = 0.1):
    return 0.5 * ((a * (x**2)) + (b * (x**4)))

def V_InfiniteSquareWell(x):
    return np.zeros_like(x, dtype = float)

def V_FiniteSquareWell(x, L = 10.0, V0 = 10.0, well_centered = False):
    '''
    Finite square well, sampled pointwise: V = 0 inside, V0 outside.

    Accuracy note: because V jumps, evaluating it at the grid points puts the effective
    wall wherever the nearest grid point happens to fall, up to dx away from the true
    position. That caps convergence near first order regardless of the stencil - see
    V_FiniteSquareWell_CellAveraged for the fix and the README for the measurement.
    '''
    if well_centered:
        return np.where(np.abs(x) <= 0.5 * L, 0.0, V0)
    else:
        return np.where((x >= 0.0) & (x <= L), 0.0, V0)

def V_FiniteSquareWell_CellAveraged(x, L = 10.0, V0 = 10.0, well_centered = False):
    '''
    Like V_DeltaDiscrete, this potential is not smooth and so is put on the grid by cell
    average rather than by evaluating it at the grid points:

        V_i = (1/dx) * integral over [x_i - dx/2, x_i + dx/2] of V(x') dx'

    For a step the integral is exact and elementary: every cell is either fully inside (0)
    or fully outside (V0) except the one straddling a wall, which gets V0 times the
    fraction of itself lying outside. That single fractional value is what encodes the
    sub-grid position of the wall, which pointwise sampling throws away.

    The contrast with the delta is worth keeping in mind. There the cell average is
    forced, since a delta cannot be evaluated pointwise at all. Here pointwise evaluation
    returns a perfectly finite, plausible-looking number, so nothing signals that anything
    is wrong - which is precisely why this error stayed hidden until it was measured
    against the semi-analytic spectrum.

    Why it matters: with pointwise sampling the effective wall snaps to the nearest grid
    point, so the well's width is wrong by up to dx and, since dE/dL is not zero, the
    energies inherit an O(dx) error. Cell averaging makes the encoded width *exact* - the
    measured total measure where V < V0 matches L to 1e-14 at every N, against an O(dx)
    error for pointwise sampling that also jitters non-monotonically with N depending on
    where the grid happens to fall. The ground state of the L=4, V0=40 well goes from
    1.40e-2 to 2.59e-3 at N=200, and from 6.09e-4 to 5.75e-6 at N=3200, with the observed
    order rising from p = 1.04 to p = 2.07.

    Second order, not fourth, is the ceiling, and the residual is NOT a shortcoming of the
    averaging. Since V jumps and psi is nonzero at the wall, psi'' jumps too, so psi is C^1
    but not C^2 there, and the stencil rows spanning that point expand in a Taylor series
    that assumes four continuous derivatives when only one exists. A symmetric stencil
    across a C^1 point gives second order regardless of how V is sampled.

    That attribution is measured, not assumed. Choosing N so the wall falls exactly on a
    cell boundary (N = 8m+5 on the domain [-8, 8] with L=4) makes the step representation
    exact, with no cell straddling it. If the residual came from the averaging inside that
    cell it would collapse; instead it stays at p = 2.00 and is slightly *larger*
    (1.76e-5 against 5.75e-6 at N ~ 3200). So what survives is the stencil crossing the
    kink, which no sampling scheme can remove.

    Note this reads dx off the grid, the same pattern V_DeltaDiscrete uses, so it must be
    called with the full grid array rather than a single point.
    '''
    x = np.asarray(x, dtype = float)
    if x.size < 2:
        raise ValueError("V_FiniteSquareWell_CellAveraged needs the grid array to infer dx.")
    dx = x[1] - x[0]

    if well_centered:
        left, right = -0.5 * L, 0.5 * L
    else:
        left, right = 0.0, float(L)

    # Overlap of each cell [x_i - dx/2, x_i + dx/2] with the well interval [left, right].
    lo = np.maximum(x - 0.5 * dx, left)
    hi = np.minimum(x + 0.5 * dx, right)
    fraction_inside = np.maximum(0.0, hi - lo) / dx

    return V0 * (1.0 - fraction_inside)

def V_LinearPotential(x, F = 1.0):
    # Linear potential: V(x) = F * x
    return F * x

def V_SoftCoulomb(x, Z = 1.0, eps = 1e-3):
    # Regularized 1D Coulomb: require eps provided or use a fixed default; do NOT infer eps from dx
    return -Z / np.sqrt(x**2 + eps**2)

def V_SingleWell(x, V0 = 5.0):
    # Simple quartic single well: V(x) = V0 * x^4 (one minimum, at the origin).
    # For the double-well quartic V0*(x^2 - a^2)^2 use V_DoubleWell instead.
    return V0 * (x**4)

def V_DeltaDiscrete(x, alpha = 8.0, x0 = 0.0):
    """
    Attractive delta well, V(x) = -alpha * delta(x - x0) with alpha > 0.

    Like V_FiniteSquareWell_CellAveraged, this potential is not smooth and so is put on
    the grid by cell average rather than by evaluating it at the grid points:

        V_i = (1/dx) * integral over [x_i - dx/2, x_i + dx/2] of V(x') dx'

    For a delta the integral picks out whichever cell contains x0 and gives -alpha/dx
    there, zero everywhere else. That is the implementation below: the cell containing x0
    is the one whose center is nearest to it, hence the argmin.

    Here the cell average is not a refinement, it is the only option. A delta cannot be
    evaluated pointwise at all, so there is no alternative scheme to compare against. The
    finite square well is the opposite case: evaluating it pointwise returns a perfectly
    finite, plausible-looking number, which is exactly why its sampling error stayed
    hidden until it was measured.

    Accuracy caveat, unrelated to the sampling: representing a delta as a single spike of
    height alpha/dx is a coarse stand-in for the continuum object no matter how the cell
    integral is done. It converges to the exact bound state E = -m*alpha^2/(2*hbar^2) only
    as dx -> 0, and needs a much finer grid than every other potential here. See the
    README.
    """
    dx = x[1] - x[0] if x.size > 1 else 1.0
    Varr = np.zeros_like(x, dtype = float)
    i0 = int(np.argmin(np.abs(x - x0)))
    Varr[i0] = -alpha / dx
    return Varr

def V_DoubleWell(x, a = 1.5, V0 = 7.0):
    # Simple quartic double well: V0*(x^2 - a^2)^2
    return V0 * (x**2 - a**2)**2

## Plotting

def plot_wavefunctions(
    x, eigvals, eigvecs, V_pot,
    n_states = None,
    scale = 1.0,
    x_range = None,
    y_range = None,
    ax = None,
    title = "Eigenfunctions over V(x)",
    ):
    '''
    Plots the first n_states eigenfunctions psi_n(x), each shifted vertically by its own
    eigenvalue E_n, superimposed on the potential V(x). This is the standard textbook way
    to visualize a 1D spectrum: psi_n "sits" on the energy line E_n, and its shape/nodes
    show the quantum number n directly.

    x, eigvals, eigvecs: outputs of Schrodinger_solver.
    V_pot: same potential function passed to Schrodinger_solver (for plotting V(x); can
    be a functools.partial to bind parameters).
    n_states: how many eigenfunctions to draw (default: all available).
    scale: vertical scale factor for psi_n so it is readable next to V(x). Purely
    cosmetic - it does not change eigvals/eigvecs, only how tall the wiggles are drawn.
    x_range, y_range: optional (min, max) tuples to zoom the plot.
    ax: existing matplotlib Axes to draw on (creates a new figure if None).

    Returns the Axes used, so the caller can further customize or save the figure.
    '''
    if n_states is None:
        n_states = eigvals.size
    n_states = min(n_states, eigvals.size)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if x_range is not None:
        mask = (x >= x_range[0]) & (x <= x_range[1])
    else:
        mask = np.ones_like(x, dtype=bool)

    V_values = np.asarray(V_pot(x), dtype=float)
    ax.plot(x[mask], V_values[mask], color="black", lw=1.8, label="V(x)")

    cmap = plt.get_cmap("viridis", max(n_states, 1))
    for n in range(n_states):
        psi = eigvecs[:, n]
        ax.plot(x[mask], eigvals[n] + scale * psi[mask], color=cmap(n), lw=1.4)
        ax.axhline(eigvals[n], color=cmap(n), lw=0.5, ls="--", alpha=0.5)
        ax.text(
            x[mask][-1], eigvals[n], f"  n={n}",
            va="center", ha="left", fontsize=8, color=cmap(n),
        )

    ax.set_xlabel("x")
    ax.set_ylabel(r"$E_n$   /   $E_n + \mathrm{scale} \cdot \psi_n(x)$")
    ax.set_title(title)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.legend(loc="best", fontsize=8)
    return ax

def plot_energy_levels(eigvals, n_states = None, ax = None, title = "Energy levels"):
    '''
    Draws a simple energy-level diagram: one horizontal line per E_n, labeled with n and
    its numeric value. Near-degenerate pairs (|E_n - E_{n-1}| much smaller than the
    overall level spacing, e.g. tunneling doublets in a double well) get their splitting
    Delta E printed explicitly, since the two lines otherwise overlap visually.

    eigvals: output of Schrodinger_solver (assumed sorted ascending).
    n_states: how many levels to draw (default: all available).
    ax: existing matplotlib Axes to draw on (creates a new figure if None).

    Returns the Axes used.
    '''
    if n_states is None:
        n_states = eigvals.size
    n_states = min(n_states, eigvals.size)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 5))

    cmap = plt.get_cmap("viridis", max(n_states, 1))
    yrange = eigvals[n_states - 1] - eigvals[0]
    min_gap = 0.045 * max(yrange, 1e-9)

    placed = []
    for n in range(n_states):
        ax.hlines(eigvals[n], 0, 1, color=cmap(n), lw=2.2)

        y_label = eigvals[n]
        if placed and (y_label - placed[-1]) < min_gap:
            y_label = placed[-1] + min_gap
        placed.append(y_label)

        split = ""
        if n > 0 and abs(eigvals[n] - eigvals[n - 1]) < 1e-3 * max(abs(eigvals[n]), 1.0):
            split = f"  (Δ={eigvals[n] - eigvals[n - 1]:.2e})"

        ax.annotate(
            f"n={n}:  E={eigvals[n]:.4f}{split}",
            xy=(1.0, eigvals[n]), xytext=(1.05, y_label),
            fontsize=8, va="center", color=cmap(n),
            arrowprops=dict(arrowstyle="-", color=cmap(n), lw=0.6),
        )

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Energy")
    ax.set_title(title)
    pad = 0.08 * max(yrange, 1.0)
    ax.set_ylim(eigvals[0] - pad, max(eigvals[n_states - 1], placed[-1]) + pad)
    return ax

'''
 ---------------------------- USAGE RECIPES ----------------------------------
 ------------------------- 1) Harmonic Oscillator ---------------------------
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot=partial(V_HarmonicOscillator, omega = 1.0, m = 1.0),
     x_min = -8.0, x_max = 8.0,
     N = 2000,
     num_eigvals = 10
 )

 ----------------------- 2) Anharmonic Oscillator ---------------------------
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot=partial(V_AnharmonicOscillator, a = 1.0, b = 0.05),
     x_min = -8.0, x_max = 8.0,
     N = 2500,
     num_eigvals = 10
 )

 -------------------------- 3) Infinite Square Well -------------------------
 # Infinite walls are enforced by the domain + Dirichlet BC (psi=0 at edges).
 # Choose L and whether centered:
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = V_InfiniteSquareWell,
     L=10.0, well_centered = False,
     N = 2000,
     num_eigvals = 10
 )

 --------------------------- 4) Finite Square Well --------------------------
 # Two sampling schemes for the same physical well. V jumps at the walls, so how the
 # step is put on the grid matters: pointwise snaps each wall to the nearest grid
 # point (error O(dx), and erratic in N), while the cell average keeps the wall's
 # sub-grid position (error O(dx^2)). Prefer the cell-averaged one for anything
 # quantitative; the pointwise one is kept so the two can be compared directly.

 # 4a) pointwise sampling
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_FiniteSquareWell, L = 10.0, V0 = 50.0, well_centered = True),
     x_min=-10.0, x_max=10.0,
     N = 2500,
     num_eigvals = 10
 )

 # 4b) cell-averaged sampling (recommended)
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_FiniteSquareWell_CellAveraged, L = 10.0, V0 = 50.0, well_centered = True),
     x_min=-10.0, x_max=10.0,
     N = 2500,
     num_eigvals = 10
 )
 # Same call, ~30x smaller error. See convergence_study_finite_well in validate_1d.py.

 ----------------------------- 5) Linear Potential --------------------------
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_LinearPotential, F = 1.0),
     x_min = -10.0, x_max = 10.0,
     N = 3000,
     num_eigvals = 10
 )

 ---------------------------- 6) Soft-Coulomb (1D) --------------------------
 # eps sets the "softening length"; smaller eps => deeper/narrower well.
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot=partial(V_SoftCoulomb, Z = 1.0, eps = 0.1),
     x_min = -80.0, x_max = 80.0,
     N = 8000,
     num_eigvals = 10
 )

 ----------------------------- 7) Quartic Single Well -----------------------
 # V(x) = V0 * x^4, a single minimum at the origin.
 # For the double-well quartic V0*(x^2 - a^2)^2 see recipe 9.
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_SingleWell, V0 = 2.0),
     x_min = -6.0, x_max = 6.0,
     N = 3000,
     num_eigvals = 10
 )

 ------------------------------ 8) Discrete Delta Well ----------------------
 # V(x) = -alpha*delta(x-x0). Like the finite well in recipe 4b this potential is not
 # smooth, so it is put on the grid by cell average: the cell integral of a delta is
 # -alpha/dx in whichever cell contains x0 and zero elsewhere. The difference is that
 # here the cell average is forced - a delta has no pointwise value to sample.
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_DeltaDiscrete, alpha = 5.0, x0 = 0.0),
     x_min = -20.0, x_max = 20.0,
     N = 6000,
     num_eigvals = 6
 )
 # Converges to E = -m*alpha^2/(2*hbar^2), but slowly: a single spike of height
 # alpha/dx is a coarse stand-in for a delta and needs a much finer grid than
 # everything else here. See the README.

 ------------------------------- 9) Quartic Double Well ---------------------
 # V(x) = V0 * (x^2 - a^2)^2  (classic symmetric double well for a>0, V0>0)
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_DoubleWell, a = 1.5, V0 = 5.0),
     x_min = -6.0, x_max = 6.0,
     N = 4000,
     num_eigvals = 10
 )
'''
## Running it

if __name__ == "__main__":
    x, eigvals, eigvecs = Schrodinger_solver(
        V_pot = partial(V_LinearPotential, F = 1.0),
        x_min = -10.0, x_max = 10.0,
        N = 3000,
        num_eigvals = 10
    )

    print("Lowest energies:")
    for n, En in enumerate(eigvals):
        print(f"n={n}, E = {En:.6f}")

    # Quick visual check: eigenfunctions over V(x) + energy-level diagram.
    plot_wavefunctions(
        x, eigvals, eigvecs, partial(V_LinearPotential, F=1.0),
        n_states=6, scale=1.0, x_range=(-5, 5), y_range=(-0.5, 6.5),
        title="Linear Potential: eigenfunctions over V(x)",
    )
    plot_energy_levels(eigvals, n_states=6, title="Line: energy levels")
    plt.show()