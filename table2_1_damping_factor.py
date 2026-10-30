"""
Table 2.1 -- The damping factor rho_n = |lambda_4 / lambda_2| = cos(2 pi/n) / cos(pi/n)
for n = 5, 10, 20, 30, 40, 50, 100, together with the asymptotic expansion
rho_n = 1 - (3/2)(pi/n)^2 + O(n^-4) and a check against the actual eigenvalues of M_n.

Paper values: 0.3820, 0.8507, 0.9629, 0.9835, 0.9907, 0.9941, 0.9985.

Usage:  python table2_1_damping_factor.py [--ns 5 10 20 30 40 50 100]
Output: tables/table2_1.csv and a printout
"""
import argparse
import csv
import os

import numpy as np

from polygon_ellipse import averaging_matrix, damping_factor, damping_factor_asymptotic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50, 100])
    ap.add_argument("--outdir", default="tables")
    args = ap.parse_args()

    rows = []
    print(f"{'n':>4} {'rho_n':>8} {'from eigenvalues':>17} {'1-1.5(pi/n)^2':>14}")
    for n in args.ns:
        rho = damping_factor(n)
        mods = np.sort(np.abs(np.linalg.eigvals(averaging_matrix(n))))[::-1]   # |lambda_1| >= |lambda_2| >= ...
        rho_eig = mods[3] / mods[1]                                            # |lambda_4| / |lambda_2|
        rows.append((n, rho, rho_eig, damping_factor_asymptotic(n)))
        print(f"{n:>4} {rho:>8.4f} {rho_eig:>17.4f} {damping_factor_asymptotic(n):>14.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "table2_1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "rho_n", "rho_n_from_eigenvalues", "asymptotic"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
    print(f"wrote {args.outdir}/table2_1.csv")


if __name__ == "__main__":
    main()
