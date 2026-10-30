"""
Figure 1.3 -- Algorithm 2 (averaging with normalization) for an n = 20 example:
the progression P_0 -> P_18 -> P_200.  The edges eventually "uncross" and the
vertices settle onto an ellipse tilted 45 degrees from the coordinate axes.

The script also reports k_u (the first iteration with no crossing edges) for
the example drawn, and draws the limiting ellipse predicted by (4.4) as a
dotted curve in the last panel.

Usage:  python fig1_3_normalized_iteration.py [--seed 3] [--n 20] [--ks 0 18 200]
Output: figures/fig1_3.pdf, figures/fig1_3.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polygon_ellipse import (algorithm2, edges_cross, ellipse_curve, k_uncross,
                             limiting_ellipse, plot_polygon, random_vertex_vectors,
                             thetas_from_vectors, tidy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 18, 200])
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x0, y0 = random_vertex_vectors(args.n, rng, zero_sum_unit=True)
    iterates = algorithm2(x0, y0, max(args.ks))
    ku = k_uncross(x0, y0)
    th_u, th_v = thetas_from_vectors(x0, y0)
    tilt, s1, s2 = limiting_ellipse(args.n, th_u, th_v)

    fig, axes = plt.subplots(1, len(args.ks), figsize=(3.2 * len(args.ks), 3.4))
    lim = 2.2 / np.sqrt(args.n)             # all vertices satisfy |x_i|,|y_i| <= 1, but are O(1/sqrt n)
    for ax, k in zip(np.atleast_1d(axes), args.ks):
        xk, yk = iterates[k]
        plot_polygon(ax, xk, yk, color="k")
        crossing = "edges cross" if edges_cross(xk, yk) else "no crossings"
        tidy(ax, rf"$\mathcal{{P}}_{{{k}}}$  ({crossing})")
        r = max(lim, 1.05 * np.max(np.abs(np.r_[xk, yk])))
        ax.set_xlim(-r, r), ax.set_ylim(-r, r)
    ex, ey = ellipse_curve(tilt, s1, s2)
    np.atleast_1d(axes)[-1].plot(ex, ey, ":", color="C3", lw=1.2, label="limiting ellipse, eq. (4.4)")
    np.atleast_1d(axes)[-1].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(f"Algorithm 2 (normalized), n = {args.n};  edges first stop crossing at k = {ku}", fontsize=10)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig1_3.{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {args.outdir}/fig1_3.pdf and .png  (n={args.n}, k={args.ks}, seed={args.seed}); "
          f"k_u = {ku}; predicted semiaxes |sigma_1|={abs(s1):.4f}, |sigma_2|={abs(s2):.4f}")


if __name__ == "__main__":
    main()
