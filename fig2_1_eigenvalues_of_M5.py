"""
Figure 2.1 -- The eigenvalues of M_5 = (I_5 + S_5)/2 in the complex plane.
They lie on the circle with center (1/2, 0) and diameter 1, since
lambda = (1 + omega)/2 with |omega| = 1.  Labels follow the paper's ordering:
lambda_1 = 1, lambda_2 = (1 + omega_1)/2, lambda_3 = conj(lambda_2),
lambda_4 = (1 + omega_2)/2, lambda_5 = conj(lambda_4).

Usage:  python fig2_1_eigenvalues_of_M5.py [--n 5]
Output: figures/fig2_1.pdf, figures/fig2_1.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polygon_ellipse import averaging_matrix, eigenvalues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    n = args.n

    lam = eigenvalues(n)
    # sanity check against a numerical eigen-decomposition
    num = np.linalg.eigvals(averaging_matrix(n))
    assert np.allclose(np.sort_complex(num), np.sort_complex(lam)), "eigenvalue formula mismatch"

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0.5 + 0.5 * np.cos(t), 0.5 * np.sin(t), color="0.6", lw=1)
    ax.axhline(0, color="0.85", lw=0.8), ax.axvline(0, color="0.85", lw=0.8)
    ax.plot(lam.real, lam.imag, "o", color="C0", ms=6)
    for j, l in enumerate(lam, start=1):
        dx = 0.04 if l.real >= 0.5 else -0.04
        ax.annotate(rf"$\lambda_{{{j}}}$", (l.real, l.imag), xytext=(dx, 0.03),
                    textcoords="offset fontsize", ha="left" if dx > 0 else "right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, 1.15), ax.set_ylim(-0.65, 0.65)
    ax.set_xlabel("Re"), ax.set_ylabel("Im")
    ax.set_title(rf"Eigenvalues of $M_{{{n}}}$ (circle: center $(1/2,0)$, diameter 1)", fontsize=10)

    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig2_1.{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {args.outdir}/fig2_1.pdf and .png;  eigenvalues: {np.round(lam, 4)}")


if __name__ == "__main__":
    main()
