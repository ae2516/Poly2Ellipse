"""
Numerical sanity checks of the paper's main results (not a figure or table in the
paper, but useful for anyone modifying the code):

  * Theorem 2.1   ||P_n w^(k)||_2 <= rho_n^k sqrt(1 - mu^2) / mu
  * Section 3     M_n c(tau + D) = cos(pi/n) c(tau + D + pi/n)  (and the same for s)
  * Eq. (3.11)    u^(k+2) = S_n u^(k)
  * Theorem 4.1   U^T A V = Sigma with a 45-degree U and Sigma = sqrt(2) diag(cos b, sin b)
  * Eq. (4.4)     Algorithm 2 converges to the predicted ellipse (tilt 45 deg, semiaxes)

Run:  python check_theorems.py
"""
import numpy as np

from polygon_ellipse import (algorithm2, averaging_matrix, cs_vectors, damping_factor,
                             distance_to_D2, fitted_ellipse, limiting_ellipse,
                             random_vertex_vectors, thetas_from_vectors, upshift)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def main():
    rng = np.random.default_rng(0)
    all_ok = True

    # Theorem 2.1 ---------------------------------------------------------------
    # The theorem is a statement about exact arithmetic, so the iteration is run with
    # re-centering (see algorithm2's docstring) and the bound is checked only while it
    # is above the roundoff floor (rhs >= 1e-8).
    for n in (7, 12, 25):
        x0, _ = random_vertex_vectors(n, rng, zero_sum_unit=True)
        c, s = cs_vectors(n)
        mu = np.hypot(c @ x0, s @ x0)
        its = algorithm2(x0, x0, 400, recenter=True)
        ok, checked = True, 0
        for k in range(1, 401):
            rhs = damping_factor(n) ** k * np.sqrt(1 - mu**2) / mu
            if rhs < 1e-8:
                break
            ok &= distance_to_D2(its[k][0]) <= rhs * (1 + 1e-9)
            checked = k
        all_ok &= check(f"Theorem 2.1 bound holds for k=1..{checked}, n={n}", ok)

    # Section 3 identity and (3.11) ----------------------------------------------
    for n in (5, 8, 13):
        D = 0.37
        c0, s0 = cs_vectors(n, D)
        c1, s1 = cs_vectors(n, D + np.pi / n)
        M = averaging_matrix(n)
        err = max(np.linalg.norm(M @ c0 - np.cos(np.pi / n) * c1), np.linalg.norm(M @ s0 - np.cos(np.pi / n) * s1))
        all_ok &= check(f"M_n c(tau+D) = cos(pi/n) c(tau+D+pi/n), n={n}", err < 1e-12, f"err={err:.1e}")
        c, s = cs_vectors(n)
        u0 = np.cos(0.4) * c + np.sin(0.4) * s
        its = algorithm2(u0, u0, 2)
        err2 = np.linalg.norm(its[2][0] - upshift(n) @ its[0][0])
        all_ok &= check(f"u^(k+2) = S_n u^(k), n={n}", err2 < 1e-12, f"err={err2:.1e}")

    # Theorem 4.1 ---------------------------------------------------------------
    for th_u, th_v in ((0.3, 1.9), (-1.2, 0.4), (2.0, 2.5)):
        A = np.array([[np.cos(th_u), np.sin(th_u)], [np.cos(th_v), np.sin(th_v)]])
        a, b = 0.5 * (th_v + th_u), 0.5 * (th_v - th_u)
        U = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
        V = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        Sigma = np.diag([np.sqrt(2) * np.cos(b), np.sqrt(2) * np.sin(b)])
        err = np.linalg.norm(U.T @ A @ V - Sigma)
        all_ok &= check(f"Theorem 4.1: U^T A V = Sigma for (theta_u, theta_v)=({th_u}, {th_v})", err < 1e-12, f"err={err:.1e}")

    # Eq. (4.4): the ellipse Algorithm 2 actually converges to -----------------------
    for n in (9, 16, 40):
        x0, y0 = random_vertex_vectors(n, rng, zero_sum_unit=True)
        th_u, th_v = thetas_from_vectors(x0, y0)
        _, s1, s2 = limiting_ellipse(n, th_u, th_v)
        xk, yk = algorithm2(x0, y0, 3000, recenter=True)[-1]
        tilt, f1, f2 = fitted_ellipse(xk, yk)
        pred = sorted([abs(s1), abs(s2)], reverse=True)
        ok = (abs(np.degrees(tilt) % 90 - 45) < 1e-6) and np.allclose([f1, f2], pred, atol=1e-6)
        all_ok &= check(f"Eq. (4.4): tilt 45 deg and semiaxes match, n={n}", ok,
                        f"fit={f1:.5f},{f2:.5f} pred={pred[0]:.5f},{pred[1]:.5f}, tilt={np.degrees(tilt):.3f}")

    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
