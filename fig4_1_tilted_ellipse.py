"""
Figure 4.1 -- Schematic of a tilted ellipse (Section 4.1): the ellipse
(u/sigma_1)^2 + (v/sigma_2)^2 = 1 rotated counterclockwise by phi radians, with
center (0, 0), tilt angle phi and semiaxes |sigma_1|, |sigma_2| labeled.

Usage:  python fig4_1_tilted_ellipse.py [--phi_deg 30] [--sigma1 1.0] [--sigma2 0.55]
Output: figures/fig4_1.pdf, figures/fig4_1.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polygon_ellipse import ellipse_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_deg", type=float, default=30.0)
    ap.add_argument("--sigma1", type=float, default=1.0)
    ap.add_argument("--sigma2", type=float, default=0.55)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    phi = np.radians(args.phi_deg)
    s1, s2 = args.sigma1, args.sigma2

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ex, ey = ellipse_curve(phi, s1, s2)
    ax.plot(ex, ey, color="k", lw=1.6)
    ax.axhline(0, color="0.7", lw=0.8), ax.axvline(0, color="0.7", lw=0.8)
    # principal axes
    c, s = np.cos(phi), np.sin(phi)
    ax.plot([-s1 * c, s1 * c], [-s1 * s, s1 * s], "--", color="C0", lw=1)
    ax.plot([s2 * s, -s2 * s], [-s2 * c, s2 * c], "--", color="C3", lw=1)
    ax.annotate(r"$|\sigma_1|$", (0.8 * s1 * c, 0.8 * s1 * s), xytext=(6, -14),
                textcoords="offset points", color="C0", fontsize=11)
    ax.annotate(r"$|\sigma_2|$", (-0.6 * s2 * s, 0.6 * s2 * c), xytext=(-26, 4),
                textcoords="offset points", color="C3", fontsize=11)
    # tilt angle arc
    r = 0.3 * s1
    t = np.linspace(0, phi, 50)
    ax.plot(r * np.cos(t), r * np.sin(t), color="C2", lw=1.2)
    ax.annotate(r"$\phi$", (1.2 * r * np.cos(phi / 2), 1.2 * r * np.sin(phi / 2)), xytext=(2, -2),
                textcoords="offset points", color="C2", fontsize=12)
    ax.plot(0, 0, "ko", ms=3)
    ax.set_aspect("equal")
    lim = 1.15 * s1
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(rf"A tilted ellipse: tilt $\phi={args.phi_deg:g}^\circ$, semiaxes $|\sigma_1|={s1:g}$, $|\sigma_2|={s2:g}$", fontsize=10)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig4_1.{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {args.outdir}/fig4_1.pdf and .png")


if __name__ == "__main__":
    main()
