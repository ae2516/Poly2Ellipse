"""
Figure 1.1 -- A random pentagon P (left) and its average P^ (right), obtained by
connecting the midpoints of the edges of P.

Usage:  python fig1_1_pentagon_and_its_average.py [--seed 1] [--n 5]
Output: figures/fig1_1.pdf, figures/fig1_1.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polygon_ellipse import average, plot_polygon, tidy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x, y = rng.random(args.n), rng.random(args.n)          # MATLAB-rand style vertices
    xh, yh = average(x), average(y)                        # the averaged polygon

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    plot_polygon(axes[0], x, y, color="k")
    tidy(axes[0], r"$\mathcal{P}$")
    plot_polygon(axes[1], x, y, color="0.75", marker=None, lw=0.8, ls="--")   # original, for reference
    plot_polygon(axes[1], xh, yh, color="C3")
    tidy(axes[1], r"$\hat{\mathcal{P}}$ (midpoints of the edges of $\mathcal{P}$)")
    for ax in axes:
        ax.set_xlim(-0.05, 1.05), ax.set_ylim(-0.05, 1.05)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig1_1.{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {args.outdir}/fig1_1.pdf and .png  (n={args.n}, seed={args.seed})")


if __name__ == "__main__":
    main()
