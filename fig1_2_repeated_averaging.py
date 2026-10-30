"""
Figure 1.2 -- Repeated averaging (Algorithm 1, no normalization) for a typical
n = 15 example: the polygons P_0, P_5, P_20, P_100 shrink toward the centroid.

All four panels share the same axes so the shrinkage is visible; the centroid
(which the averaging step preserves) is marked with a cross.

Usage:  python fig1_2_repeated_averaging.py [--seed 2] [--n 15] [--ks 0 5 20 100]
Output: figures/fig1_2.pdf, figures/fig1_2.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polygon_ellipse import algorithm1, plot_polygon, tidy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 5, 20, 100])
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x0, y0 = rng.random(args.n), rng.random(args.n)
    iterates = algorithm1(x0, y0, max(args.ks))
    xbar, ybar = x0.mean(), y0.mean()

    fig, axes = plt.subplots(1, len(args.ks), figsize=(3.0 * len(args.ks), 3.2))
    for ax, k in zip(np.atleast_1d(axes), args.ks):
        xk, yk = iterates[k]
        plot_polygon(ax, xk, yk, color="k")
        ax.plot(xbar, ybar, "+", color="C3", ms=9, mew=1.5)
        ax.set_xlim(-0.05, 1.05), ax.set_ylim(-0.05, 1.05)
        tidy(ax, rf"$\mathcal{{P}}_{{{k}}}$")
    fig.suptitle(f"Algorithm 1 (no normalization), n = {args.n}", fontsize=10)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig1_2.{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {args.outdir}/fig1_2.pdf and .png  (n={args.n}, k={args.ks}, seed={args.seed})")


if __name__ == "__main__":
    main()
