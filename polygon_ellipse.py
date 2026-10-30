"""
Shared code for reproducing the figures and tables in

    A. N. Elmachtoub and C. F. Van Loan,
    "From Random Polygon to Ellipse: An Eigenanalysis",
    SIAM Review 52(1), pp. 151-170, 2010.  doi:10.1137/090746707

Notation follows the paper.  A polygon with n vertices is stored as two real
vectors x, y in R^n holding the x- and y-coordinates of the vertices in order;
P(x, y) denotes that polygon.  The "averaging" step replaces each vertex by
the midpoint of the edge that follows it, i.e. x_hat = M_n x, y_hat = M_n y,
where M_n = (I_n + S_n)/2 and S_n is the upshift matrix.

Only numpy is required here; matplotlib is used by the figure scripts.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Section 1: the averaging operation and the three algorithms
# --------------------------------------------------------------------------- #


def upshift(n: int) -> np.ndarray:
    """The upshift matrix S_n = [e_n | e_1 | ... | e_{n-1}], so (S_n w)_i = w_{i+1}
    with the cyclic convention w_{n+1} = w_1."""
    return np.roll(np.eye(n), -1, axis=0)


def averaging_matrix(n: int) -> np.ndarray:
    """M_n = (I_n + S_n) / 2."""
    return 0.5 * (np.eye(n) + upshift(n))


def average(w: np.ndarray) -> np.ndarray:
    """Apply M_n to a vector without forming the matrix:  (M_n w)_i = (w_i + w_{i+1}) / 2."""
    return 0.5 * (w + np.roll(w, -1))


def algorithm1(x0: np.ndarray, y0: np.ndarray, kmax: int):
    """Algorithm 1 (repeated averaging, no normalization).

    Returns the list [(x^(0), y^(0)), (x^(1), y^(1)), ..., (x^(kmax), y^(kmax))].
    The polygons shrink toward the (preserved) centroid.
    """
    xs, ys = [np.asarray(x0, float)], [np.asarray(y0, float)]
    for _ in range(kmax):
        xs.append(average(xs[-1]))
        ys.append(average(ys[-1]))
    return list(zip(xs, ys))


def algorithm2(x0: np.ndarray, y0: np.ndarray, kmax: int, recenter: bool = False):
    """Algorithm 2 (repeated averaging with 2-norm normalization).

    Inputs are unit 2-norm vectors whose components sum to zero.  Each step
    computes f = M_n x, g = M_n y and normalizes:  x <- f/||f||_2, y <- g/||g||_2.

    recenter=False reproduces the algorithm exactly as written in the paper.
    recenter=True additionally subtracts the mean at every step.  In exact
    arithmetic this changes nothing (e^T M_n = e^T), but in floating point the
    unnormalized eigenvalue-1 mode is seeded by roundoff and amplified by a
    factor 1/cos(pi/n) per step, so for very long runs (k in the thousands for
    n ~ 10-20) the as-written iteration collapses to a single point.  The
    paper's experiments use k <= ~650, where the effect is invisible.
    """
    xs, ys = [np.asarray(x0, float)], [np.asarray(y0, float)]
    for _ in range(kmax):
        f, g = average(xs[-1]), average(ys[-1])
        if recenter:
            f, g = f - f.mean(), g - g.mean()
        xs.append(f / np.linalg.norm(f))
        ys.append(g / np.linalg.norm(g))
    return list(zip(xs, ys))


def cs_vectors(n: int, delta: float = 0.0):
    """The vectors c(tau + delta), s(tau + delta) of (2.10) and (3.5)-(3.6):

        c_j = sqrt(2/n) cos(tau_j + delta),  s_j = sqrt(2/n) sin(tau_j + delta),
        tau_j = 2 (j-1) pi / n,  j = 1..n.

    For delta = 0 they form an orthonormal basis of the invariant subspace D_2.
    """
    tau = 2.0 * np.arange(n) * np.pi / n
    return np.sqrt(2.0 / n) * np.cos(tau + delta), np.sqrt(2.0 / n) * np.sin(tau + delta)


def algorithm3(n: int, theta_u: float, theta_v: float, kmax: int):
    """Algorithm 3: Algorithm 2 started inside D_2,

        u^(0) = cos(theta_u) c + sin(theta_u) s,   v^(0) = cos(theta_v) c + sin(theta_v) s.

    Returns [(u^(0), v^(0)), ..., (u^(kmax), v^(kmax))].
    """
    c, s = cs_vectors(n)
    u0 = np.cos(theta_u) * c + np.sin(theta_u) * s
    v0 = np.cos(theta_v) * c + np.sin(theta_v) * s
    return algorithm2(u0, v0, kmax)


def random_vertex_vectors(n: int, rng: np.random.Generator, zero_sum_unit: bool):
    """Random x^(0), y^(0) in the spirit of the paper, which used MATLAB's rand
    (components uniform on [0, 1]).

    zero_sum_unit=False : raw uniform vectors (inputs to Algorithm 1).
    zero_sum_unit=True  : centered and scaled to unit 2-norm (inputs to Algorithm 2).
    """
    x, y = rng.random(n), rng.random(n)
    if zero_sum_unit:
        x, y = x - x.mean(), y - y.mean()
        x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    return x, y


# --------------------------------------------------------------------------- #
# Section 2: eigenanalysis of M_n
# --------------------------------------------------------------------------- #


def eigenvalues(n: int) -> np.ndarray:
    """The eigenvalues lambda_1, ..., lambda_n of M_n in the paper's ordering:
    lambda_1 = 1, then conjugate pairs ordered by decreasing modulus,
    lambda_{2j} = (1 + omega_j)/2, lambda_{2j+1} = conj(lambda_{2j}),  omega_j = exp(2 pi i j / n).
    (For even n the last eigenvalue is the unpaired (1 + omega_{n/2})/2 = 0.)
    """
    lam = [1.0 + 0j]
    for j in range(1, n // 2 + 1):
        lj = (1.0 + np.exp(2j * np.pi * j / n)) / 2.0
        lam.append(lj)
        if 2 * j != n:          # j and n-j are distinct indices -> conjugate pair
            lam.append(np.conj(lj))
    return np.array(lam)


def damping_factor(n: int) -> float:
    """rho_n = |lambda_4 / lambda_2| = cos(2 pi / n) / cos(pi / n)  (Section 2.4), n >= 5."""
    return np.cos(2.0 * np.pi / n) / np.cos(np.pi / n)


def damping_factor_asymptotic(n: int) -> float:
    """The expansion rho_n = 1 - (3/2) (pi/n)^2 + O(n^-4)."""
    return 1.0 - 1.5 * (np.pi / n) ** 2


def distance_to_D2(w: np.ndarray) -> float:
    """||P_n w||_2, the norm of the component of w orthogonal to D_2 = span{c, s}."""
    c, s = cs_vectors(len(w))
    return float(np.linalg.norm(w - (c @ w) * c - (s @ w) * s))


# --------------------------------------------------------------------------- #
# Section 4: the limiting ellipse
# --------------------------------------------------------------------------- #


def thetas_from_vectors(x0: np.ndarray, y0: np.ndarray):
    """Section 4.3: theta_u, theta_v from the D_2 components of x^(0), y^(0),
        cos(theta_u) = c^T x0 / r,  sin(theta_u) = s^T x0 / r,  r = sqrt((c^T x0)^2 + (s^T x0)^2),
    and likewise for theta_v with y0."""
    c, s = cs_vectors(len(x0))
    return float(np.arctan2(s @ x0, c @ x0)), float(np.arctan2(s @ y0, c @ y0))


def limiting_ellipse(n: int, theta_u: float, theta_v: float):
    """Equation (4.4) / Theorem 4.1: the vertices of the limiting polygons lie on an
    origin-centered ellipse with tilt pi/4 and semiaxes

        sigma_1 = (2/sqrt(n)) cos((theta_v - theta_u)/2),
        sigma_2 = (2/sqrt(n)) sin((theta_v - theta_u)/2).

    Returns (tilt, sigma_1, sigma_2); the semiaxis lengths are |sigma_1|, |sigma_2|.
    """
    b = 0.5 * (theta_v - theta_u)
    return np.pi / 4, 2.0 / np.sqrt(n) * np.cos(b), 2.0 / np.sqrt(n) * np.sin(b)


def ellipse_curve(tilt: float, sigma1: float, sigma2: float, npts: int = 400):
    """Points on the ellipse obtained by rotating (u/sigma1)^2 + (v/sigma2)^2 = 1
    counterclockwise by `tilt` radians (Section 4.1)."""
    t = np.linspace(0.0, 2.0 * np.pi, npts)
    u, v = sigma1 * np.cos(t), sigma2 * np.sin(t)
    ct, st = np.cos(tilt), np.sin(tilt)
    return ct * u - st * v, st * u + ct * v


def fitted_ellipse(x: np.ndarray, y: np.ndarray):
    """Tilt (radians, in [0, pi)) and semiaxes of the ellipse through n equally
    spaced (in parameter) points A [cos t, sin t]^T, read off from the 2x2 matrix
    [x y]^T [x y] = (n/2) U Sigma^2 U^T.  Used only to check results."""
    C = np.column_stack([x, y]).T @ np.column_stack([x, y])
    w, V = np.linalg.eigh(C)
    tilt = np.arctan2(V[1, 1], V[0, 1]) % np.pi
    sig = np.sqrt(np.maximum(2.0 * w[::-1] / len(x), 0.0))
    return float(tilt), float(sig[0]), float(sig[1])


# --------------------------------------------------------------------------- #
# Edge crossings (Table 1.2, Figure 1.3)
# --------------------------------------------------------------------------- #


def edges_cross(x: np.ndarray, y: np.ndarray) -> bool:
    """True if any two non-adjacent edges of the closed polygon P(x, y) properly
    intersect.  Vectorized O(n^2) test based on segment orientations."""
    n = len(x)
    P = np.column_stack([x, y])
    Q = np.roll(P, -1, axis=0)                 # edge i goes from P[i] to Q[i]
    d = Q - P                                  # edge direction vectors

    def cross(a, b):                           # 2-D cross product, broadcastable
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # orientation of edge j's endpoints relative to edge i (and vice versa)
    o1 = cross(d[:, None, :], P[None, :, :] - P[:, None, :])   # P_j   rel. edge i
    o2 = cross(d[:, None, :], Q[None, :, :] - P[:, None, :])   # Q_j   rel. edge i
    o3 = cross(d[None, :, :], P[:, None, :] - P[None, :, :])   # P_i   rel. edge j
    o4 = cross(d[None, :, :], Q[:, None, :] - P[None, :, :])   # Q_i   rel. edge j
    proper = (o1 * o2 < 0) & (o3 * o4 < 0)

    i, j = np.indices((n, n))
    gap = np.abs(i - j)
    adjacent = (gap == 0) | (gap == 1) | (gap == n - 1)   # edges sharing a vertex
    return bool(np.any(proper & ~adjacent))


def k_uncross(x0: np.ndarray, y0: np.ndarray, kmax: int = 100_000) -> int:
    """k_u: the smallest k such that the edges of P(x^(k), y^(k)) produced by
    Algorithm 2 do not cross."""
    x, y = np.asarray(x0, float), np.asarray(y0, float)
    for k in range(kmax + 1):
        if not edges_cross(x, y):
            return k
        f, g = average(x), average(y)
        x, y = f / np.linalg.norm(f), g / np.linalg.norm(g)
    raise RuntimeError("edges still cross after kmax iterations")


def k_delta(x0: np.ndarray, y0: np.ndarray, delta: float = 1e-3, kmax: int = 10_000_000) -> int:
    """k_delta (Table 1.1): the smallest k such that, under Algorithm 1, all vertices
    of P_k lie within delta of the centroid in the root-sum-of-squares sense,
        sqrt( sum_i (x_i^(k) - xbar)^2 + (y_i^(k) - ybar)^2 ) <= delta."""
    x, y = np.asarray(x0, float), np.asarray(y0, float)
    xbar, ybar = x.mean(), y.mean()          # preserved by M_n
    for k in range(kmax + 1):
        if np.sqrt(np.sum((x - xbar) ** 2) + np.sum((y - ybar) ** 2)) <= delta:
            return k
        x, y = average(x), average(y)
    raise RuntimeError("did not reach delta within kmax iterations")


# --------------------------------------------------------------------------- #
# Plotting helpers (matplotlib imported lazily so the module stays numpy-only)
# --------------------------------------------------------------------------- #


def plot_polygon(ax, x, y, *, color="C0", lw=1.2, marker="o", ms=3.5, alpha=1.0, label=None, ls="-"):
    """Draw the closed polygon P(x, y) on a matplotlib axis."""
    xc, yc = np.append(x, x[0]), np.append(y, y[0])
    ax.plot(xc, yc, ls=ls, color=color, lw=lw, alpha=alpha, label=label)
    if marker:
        ax.plot(x, y, marker, color=color, ms=ms, alpha=alpha)


def tidy(ax, title=None):
    """Equal aspect, no ticks, optional title."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
