# From Random Polygon to Ellipse — reproducibility code

Code to regenerate every figure and table in

> A. N. Elmachtoub and C. F. Van Loan, **"From Random Polygon to Ellipse: An Eigenanalysis"**,
> *SIAM Review* 52(1), pp. 151–170, 2010. [doi:10.1137/090746707](https://doi.org/10.1137/090746707)

Repeatedly replacing the vertices of a polygon by the midpoints of its edges (and renormalizing)
drives any random polygon to a polygon whose vertices lie on an ellipse tilted 45° from the axes.
The paper explains this through the eigenstructure of the circulant averaging matrix
`M_n = (I_n + S_n)/2`.  Each script below regenerates one figure or table; shared code lives in
`polygon_ellipse.py`.

## Setup

```bash
pip install -r requirements.txt      # numpy, matplotlib
python run_all.py                    # everything (about half a minute; --quick skips n=80 in Table 1.2)
```

Outputs are written to `figures/` (PDF + PNG) and `tables/` (CSV). Every script has `--help`.

## One file per figure / table

| File | Reproduces | What it does |
|---|---|---|
| `fig1_1_pentagon_and_its_average.py` | Figure 1.1 | A random pentagon and the polygon of its edge midpoints. |
| `fig1_2_repeated_averaging.py` | Figure 1.2 | Algorithm 1 (no normalization), n = 15: P₀, P₅, P₂₀, P₁₀₀ shrinking to the centroid. |
| `fig1_3_normalized_iteration.py` | Figure 1.3 | Algorithm 2 (normalized), n = 20: P₀, P₁₈, P₂₀₀; marks crossings, overlays the limiting ellipse of eq. (4.4), prints k_u. |
| `fig2_1_eigenvalues_of_M5.py` | Figure 2.1 | Eigenvalues of M₅ on the circle with center (½, 0) and diameter 1, labeled in the paper's ordering. |
| `fig3_1_even_and_odd_polygons.py` | Figure 3.1 | Algorithm 3: the vertices of P_even (red) and P_odd (blue) on the ellipse of Theorem 4.1; verifies u⁽ᵏ⁺²⁾ = Sₙu⁽ᵏ⁾ numerically. |
| `fig4_1_tilted_ellipse.py` | Figure 4.1 | Schematic of a tilted ellipse with tilt φ and semiaxes labeled. |
| `table1_1_k_delta.py` | Table 1.1 | Average k_δ (iterations for Algorithm 1 to collapse within δ = 0.001 of the centroid), n = 10, 20. |
| `table1_2_k_uncross.py` | Table 1.2 | Average k_u (first Algorithm 2 iterate with no crossing edges), n = 10, 20, 40, 80. |
| `table2_1_damping_factor.py` | Table 2.1 | ρₙ = cos(2π/n)/cos(π/n), checked against the eigenvalues of Mₙ and the expansion 1 − (3/2)(π/n)². |
| `check_theorems.py` | — | Numerical checks of Theorem 2.1, the Section 3 identities, eq. (3.11), Theorem 4.1 and eq. (4.4). |
| `polygon_ellipse.py` | — | Shared module: Sₙ, Mₙ, Algorithms 1–3, the c/s basis of D₂, ellipse formulas, crossing test. |
| `run_all.py` | — | Runs everything above in order. |

## Reproducibility notes

* **Randomness.** The paper's random polygons were drawn with MATLAB's `rand` and no seed is
  recorded, so the figures reproduce the *phenomena* (not the identical polygons) and the tables
  reproduce the paper's averages up to Monte-Carlo noise. All scripts use fixed NumPy seeds
  (`--seed`), so this repository's outputs are themselves exactly reproducible.
* **Table 1.1 initialization.** The text says the initial vectors came from `rand`, while
  Algorithm 1 is stated for unit 2-norm inputs; k_δ depends on the initial scale, so the script
  reports three readings. Random unit directions give 126 / 485 (paper: 127 / 485); raw
  `rand` gives ≈124 / 499; `rand` rescaled to unit norm gives ≈114 / 424. With 100 trials the
  standard error is about 1 (n = 10) and 3 (n = 20).
* **Table 1.2.** Averages reproduce to within sampling error (k_u has a large spread: its standard
  deviation is comparable to its mean), e.g. 8.2 / 36.3 / 166 / ~690 vs. the paper's
  8.7 / 38.1 / 163.0 / 647.5 with 100 trials.
* **Figure 3.1.** The caption says n = 12 (the surrounding text says n = 5); θ_u, θ_v are not
  stated. The script defaults to n = 12, θ_u = 0.4, θ_v = 1.7 and takes all three as arguments.
* **Algorithm 2 and long runs.** `algorithm2(..., recenter=False)` is the algorithm exactly as
  written. Because the eigenvalue-1 mode (the centroid) is never projected out, floating-point
  roundoff seeds it and the normalization amplifies it by 1/cos(π/n) per step; for runs far
  longer than those in the paper (k ≈ 1000 at n = 10, k ≈ 3000 at n = 20) the iterates collapse to
  a point. `recenter=True` subtracts the mean each step, which is a no-op in exact arithmetic
  and keeps the iterates in D₂ indefinitely. `check_theorems.py` uses it.

## Citation

If you use this code, please cite the paper above.

## License

MIT — see `LICENSE`.
