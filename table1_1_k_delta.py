"""
Table 1.1 -- Average k_delta for Algorithm 1 (no normalization), delta = 0.001.

k_delta is the smallest k such that every vertex of P_k is within delta of the
centroid, measured as sqrt( sum_i (x_i^(k) - xbar)^2 + (y_i^(k) - ybar)^2 ).
The paper reports averages over 100 trials:  n = 10 -> 127,  n = 20 -> 485.

The paper says the initial vectors were generated with MATLAB's rand, while
Algorithm 1 is stated for unit 2-norm inputs.  Because k_delta depends on the
scale of the initial polygon, the script reports three readings side by side:
    rand        components uniform on [0,1], no scaling
    rand-unit   uniform on [0,1], then scaled to unit 2-norm
    direction   random unit vector (Gaussian, normalized)  <- matches 127 / 485 best
Use --init to pick a single scheme.

Usage:  python table1_1_k_delta.py [--trials 100] [--delta 1e-3] [--ns 10 20] [--seed 11]
                                   [--init all|rand|rand-unit|direction]
Output: tables/table1_1.csv and a printout
"""
import argparse
import csv
import os

import numpy as np

from polygon_ellipse import k_delta

PAPER = {10: 127, 20: 485}
SCHEMES = ("rand", "rand-unit", "direction")


def initial_vectors(n, rng, scheme):
    if scheme == "rand":
        return rng.random(n), rng.random(n)
    if scheme == "rand-unit":
        x, y = rng.random(n), rng.random(n)
        return x / np.linalg.norm(x), y / np.linalg.norm(y)
    if scheme == "direction":
        x, y = rng.standard_normal(n), rng.standard_normal(n)
        return x / np.linalg.norm(x), y / np.linalg.norm(y)
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--init", default="all", choices=("all",) + SCHEMES)
    ap.add_argument("--outdir", default="tables")
    args = ap.parse_args()

    schemes = SCHEMES if args.init == "all" else (args.init,)
    rows = []
    print(f"Table 1.1: average k_delta, delta = {args.delta:g}, {args.trials} trials per entry")
    print(f"{'n':>4} {'init scheme':>12} {'mean k_delta':>13} {'std':>7} {'paper':>7}")
    for n in args.ns:
        for scheme in schemes:
            rng = np.random.default_rng(args.seed)          # same seed for every scheme
            ks = [k_delta(*initial_vectors(n, rng, scheme), delta=args.delta) for _ in range(args.trials)]
            rows.append((n, scheme, np.mean(ks), np.std(ks), PAPER.get(n, "")))
            print(f"{n:>4} {scheme:>12} {np.mean(ks):>13.1f} {np.std(ks):>7.1f} {PAPER.get(n, ''):>7}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "table1_1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "init_scheme", "mean_k_delta", "std_k_delta", "paper_value", "trials", "delta", "seed"])
        for n, scheme, m, s, p in rows:
            w.writerow([n, scheme, f"{m:.2f}", f"{s:.2f}", p, args.trials, args.delta, args.seed])
    print(f"wrote {args.outdir}/table1_1.csv")


if __name__ == "__main__":
    main()
