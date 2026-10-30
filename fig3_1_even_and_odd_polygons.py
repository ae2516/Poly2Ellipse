"""
Figure 3.1 -- The vertices of P_even (red) and P_odd (blue) produced by
Algorithm 3, i.e. Algorithm 2 started inside the invariant subspace D_2 with
    u^(0) = cos(theta_u) c + sin(theta_u) s,   v^(0) = cos(theta_v) c + sin(theta_v) s.
Section 3 shows u^(k+2) = S_n u^(k): all even-indexed polygons have the same
vertex set and so do all odd-indexed ones, and both vertex sets lie on the
ellipse of Theorem 4.1 / eq. (4.4) (drawn dotted).

The paper's caption gives n = 12 (the surrounding text says n = 5); theta_u and
theta_v are not stated, so arbitrary values are used.  The script verifies
numerically that P_{2k} = P_0 and P_{2k+1} = P_1 up to a cyclic relabeling.

Usage:  python fig3_1_even_and_odd_polygons.py [--n 12] [--theta_u 0.4] [--theta_v 1.7]
Output: figures/fig3_1.pdf, figures/fig3_1.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polygon_ellipse import algorithm3, ellipse_curve, limiting_ellipse, plot_polygon, tidy


def relabel_distance(x, y, xr, yr):
    """min over cyclic shifts r of || (x,y) - shift_r(xr,yr) ||."""
    return min(np.linalg.norm(np.c_[x, y] - np.roll(np.c_[xr, yr], r, axis=0)) for r in range(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--theta_u", type=float, default=0.4)
    ap.add_argument("--theta_v", type=float, default=1.7)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    n = args.n

    its = algorithm3(n, args.theta_u, args.theta_v, 2 * n + 1)
    (u0, v0), (u1, v1) = its[0], its[1]
    # Conjecture 2 / eq. (3.11): even iterates coincide with P_0, odd with P_1 (as vertex sets)
    even_err = max(relabel_distance(u0, v0, *its[k]) for k in range(0, 2 * n + 1, 2))
    odd_err = max(relabel_distance(u1, v1, *its[k]) for k in range(1, 2 * n + 1, 2))

    tilt, s1, s2 = limiting_ellipse(n, args.theta_u, args.theta_v)
    ex, ey = ellipse_curve(tilt, s1, s2)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot(ex, ey, ":", color="0.5", lw=1, label="ellipse of eq. (4.4)")
    plot_polygon(ax, u0, v0, color="red", lw=0.6, alpha=0.9, ms=5, label=r"$\mathcal{P}_{\rm even}$")
    plot_polygon(ax, u1, v1, color="blue", lw=0.6, alpha=0.9, ms=5, label=r"$\mathcal{P}_{\rm odd}$")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    tidy(ax, rf"$n={n}$, $\theta_u={args.theta_u}$, $\theta_v={args.theta_v}$")
    r = 1.15 * max(abs(s1), abs(s2))
    ax.set_xlim(-r, r), ax.set_ylim(-r, r)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig3_1.{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {args.outdir}/fig3_1.pdf and .png  (n={n}); "
          f"max |P_2k - P_0| = {even_err:.1e}, max |P_2k+1 - P_1| = {odd_err:.1e} (after relabeling); "
          f"semiaxes |sigma_1|={abs(s1):.4f}, |sigma_2|={abs(s2):.4f}, tilt 45 deg")


if __name__ == "__main__":
    main()
