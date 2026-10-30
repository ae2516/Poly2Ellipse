"""
Table 1.2 -- Average k_u for Algorithm 2 (averaging with normalization).

k_u is the smallest k such that the edges of P(x^(k), y^(k)) do not cross.
The paper reports averages over 100 random examples:
    n = 10 -> 8.7,   n = 20 -> 38.1,   n = 40 -> 163.0,   n = 80 -> 647.5.
Initial vectors here are uniform-random (MATLAB-rand style), then centered and
scaled to unit 2-norm as Algorithm 2 requires.

Usage:  python table1_2_k_uncross.py [--trials 100] [--ns 10 20 40 80] [--seed 12]
Output: tables/table1_2.csv and a printout   (the n = 80 row takes ~20 s)
"""
import argparse
import csv
import os

import numpy as np

from polygon_ellipse import k_uncross, random_vertex_vectors

PAPER = {10: 8.7, 20: 38.1, 40: 163.0, 80: 647.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--outdir", default="tables")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"Table 1.2: average k_u (first iteration with no crossing edges), {args.trials} trials")
    print(f"{'n':>4} {'mean k_u':>9} {'std':>7} {'paper':>7}")
    for n in args.ns:
        ks = [k_uncross(*random_vertex_vectors(n, rng, zero_sum_unit=True)) for _ in range(args.trials)]
        rows.append((n, np.mean(ks), np.std(ks), PAPER.get(n, "")))
        print(f"{n:>4} {np.mean(ks):>9.1f} {np.std(ks):>7.1f} {PAPER.get(n, ''):>7}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "table1_2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "mean_k_u", "std_k_u", "paper_value", "trials", "seed"])
        for n, m, s, p in rows:
            w.writerow([n, f"{m:.2f}", f"{s:.2f}", p, args.trials, args.seed])
    print(f"wrote {args.outdir}/table1_2.csv")


if __name__ == "__main__":
    main()
