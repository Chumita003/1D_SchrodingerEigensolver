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
    Known limitation: the two outermost interior rows (adjacent to each Dirichlet
    boundary) are missing a term of the 5-point stencil that reaches one point past
    x_min/x_max. That point is not the boundary itself (which is legitimately 0 by the
    Dirichlet condition) - it is undefined. I derived the correct one-sided 4th-order
    formulas for those two rows by hand (see Eigensolver_1Dimensional.pdf), but using
    them here would make the matrix asymmetric: the off-diagonal coefficient a
    one-sided row assigns to its neighbor does not match the coefficient that neighbor's
    (central) row assigns back, so H would stop satisfying H = H^dagger and eigsh could
    return complex or unordered eigenvalues. I kept the uniform central stencil on
    purpose to guarantee a Hermitian H. The cost is that local truncation error at those
    two rows drops from O(dx^4) to O(dx^2), which shows up as O(dx) (not O(dx^4)) global
    convergence for states with non-negligible amplitude/slope at the boundary - e.g. the
    infinite square well (see validate_1d.py). States that decay to ~0 well before
    x_min/x_max (e.g. the harmonic oscillator with a wide enough domain) are essentially
    unaffected. A proper fix exists (summation-by-parts boundary operators with a
    matching non-uniform quadrature) but is out of scope here.
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

    d2_matrix = diags(
        diagonals=[
            coeffs[0] * np.ones(N - 2),
            coeffs[1] * np.ones(N - 1),
            coeffs[2] * np.ones(N),
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

    # Computing the eigenvalues and eigenvectors of the Hamiltonian
    eigvals, eigvecs = eigsh(H, k = num_eigvals, which = 'SA')

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
        peak = np.argmax(np.abs(normalized_eigvecs[:, i]))
        if normalized_eigvecs[peak, i] < 0:
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
    if well_centered:
        return np.where(np.abs(x) <= 0.5 * L, 0.0, V0)
    else:
        return np.where((x >= 0.0) & (x <= L), 0.0, V0)

def V_LinearPotential(x, F = 1.0):
    # Linear potential: V(x) = F * x
    return F * x

def V_SoftCoulomb(x, Z = 1.0, eps = 1e-3):
    # Regularized 1D Coulomb: require eps provided or use a fixed default; do NOT infer eps from dx
    return -Z / np.sqrt(x**2 + eps**2)

def V_SingleWell(x, a = 1.0, V0 = 5.0):
    # Simple quartic single well: V0 * (x^2 - a^2)^2
    return V0 * (x**4)

def V_DeltaDiscrete(x, alpha = 8.0, x0 = 0.0):
    """
    Attractive delta: V(x) = -alpha * delta(x - x0), alpha>0
    Discrete implementation on grid: V_i = -alpha/dx at the grid point nearest x0, else 0
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
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_FiniteSquareWell, L = 10.0, V0 = 50.0, well_centered = True),
     x_min=-10.0, x_max=10.0,
     N = 2500,
     num_eigvals = 10
 )

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
 # V(x) = V0 * (x^2 - a^2)^2  (one/two wells depending on parameters)
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_SingleWell, a = 1.0, V0 = 2.0),
     x_min = -6.0, x_max = 6.0,
     N = 3000,
     num_eigvals = 10
 )

 ------------------------------ 8) Discrete Delta Well ----------------------
 # V(x) = -alpha*delta(x-x0)  implemented as V[i0] = -alpha/dx at nearest gridpoint
 x, eigvals, eigvecs = Schrodinger_solver(
     V_pot = partial(V_DeltaDiscrete, alpha = 5.0, x0 = 0.0),
     x_min = -20.0, x_max = 20.0,
     N = 6000,
     num_eigvals = 6
 )

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
        V_pot=partial(V_HarmonicOscillator, omega = 1.0, m = 1.0),
        x_min = -8.0, x_max = 8.0,
        N = 2000,
        num_eigvals = 10
    )

    print("Lowest energies:")
    for n, En in enumerate(eigvals):
        print(f"n={n}, E = {En:.6f}")

    # Quick visual check: eigenfunctions over V(x) + energy-level diagram.
    plot_wavefunctions(
        x, eigvals, eigvecs, partial(V_HarmonicOscillator, omega=1.0, m=1.0),
        n_states=6, scale=1.0, x_range=(-5, 5), y_range=(-0.5, 6.5),
        title="Harmonic oscillator: eigenfunctions over V(x)",
    )
    plot_energy_levels(eigvals, n_states=6, title="Harmonic oscillator: energy levels")
    plt.show()