#!/usr/bin/env python3
"""
route1_fredholm_attempt.py
==========================
Attempt at Route 1 (A8'): Can we construct a nuclear operator L_s
such that det(I - L_s) = xi(s)/xi(0)?

Approach: Use the Hadamard product over zeros.
xi(s)/xi(0) = prod_rho (1 - s/rho)  [with appropriate convergence factors]

The zeros rho of xi satisfy sum 1/|rho|^2 < infty (Hilbert-Schmidt),
but sum 1/|rho| = infty (NOT trace class).

We test:
1. Whether the regularized determinant det_2(I - sR) works
   (R = diag(1/rho), Hilbert-Schmidt operator on l^2(zeros))
2. The Selberg-vs-Riemann comparison: geodesic density vs prime density
3. Whether including the archimedean factor changes trace-class behavior

Uses the first N zeros from mpmath's zetazero().
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from mpmath import mp, mpf, zetazero, pi, gamma, log, sqrt, fabs, re, im, zeta, fsum
import time

mp.dps = 30

def xi_value(s):
    """Compute xi(s) = 1/2 * s * (s-1) * pi^(-s/2) * gamma(s/2) * zeta(s)"""
    return mpf('0.5') * s * (s - 1) * pi**(-s/2) * gamma(s/2) * zeta(s)


def main():
    N = 50  # number of zeros to use (first N positive imaginary parts)

    print("=" * 90)
    print("  Route 1 Attempt: Fredholm Determinant for xi(s)")
    print(f"  Using first {N} non-trivial zeros of zeta")
    print("=" * 90)

    # Step 1: Compute zeros
    print("\n--- Step 1: Computing zeros ---")
    t0 = time.time()
    zeros = []
    for n in range(1, N + 1):
        rho = zetazero(n)  # returns mpc with real=0.5, imag=gamma_n
        zeros.append(rho)
        if n <= 5 or n == N:
            print(f"  rho_{n} = {float(re(rho)):.1f} + {float(im(rho)):.6f}i")
    print(f"  ({time.time()-t0:.1f}s)")

    # Step 2: Trace-class analysis
    print("\n--- Step 2: Trace-class analysis of R = diag(1/rho) ---")

    sum_1_rho = fsum(1/fabs(rho) for rho in zeros)
    sum_1_rho2 = fsum(1/fabs(rho)**2 for rho in zeros)

    print(f"  sum_{{j=1}}^{{{N}}} 1/|rho_j|     = {float(sum_1_rho):.6f}  (trace norm, should diverge)")
    print(f"  sum_{{j=1}}^{{{N}}} 1/|rho_j|^2   = {float(sum_1_rho2):.6f}  (Hilbert-Schmidt norm, should converge)")

    # Asymptotic: gamma_n ~ 2*pi*n / log(n) for large n
    # So 1/|rho_n| ~ 1/gamma_n ~ log(n)/(2*pi*n)
    # sum 1/|rho_n| ~ sum log(n)/n = divergent (like log^2(N)/2)
    # sum 1/|rho_n|^2 ~ sum (log(n)/n)^2 = convergent

    import math
    asymptotic_trace = sum(math.log(n+1)/(2*math.pi*(n+1)) for n in range(N))
    print(f"  Asymptotic estimate (sum log(n)/2pi*n): {asymptotic_trace:.6f}")
    print(f"  Ratio actual/asymptotic: {float(sum_1_rho)/asymptotic_trace:.4f}")
    print(f"  --> R is Hilbert-Schmidt but NOT trace class (diverges as log^2(N)/2)")

    # Step 3: Regularized determinant det_2(I - sR)
    print("\n--- Step 3: Regularized Fredholm determinant ---")
    print("  det_2(I - sR) = prod_j (1 - s/rho_j) * exp(s/rho_j)")
    print("  xi(s)/xi(0) = exp(Bs) * det_2(I - sR)  where B = -xi'(0)/xi(0)")

    # Compute xi(0)
    xi_0 = xi_value(mpf('0.5'))  # xi(1/2) -- wait, xi(0) = xi(1) = 1/2
    # Actually xi(0) = 1/2 * 0 * (-1) * ... = 0. Use xi(1/2) instead.
    # The Hadamard product: xi(s) = xi(1/2) * prod_{gamma>0} (1 - (s-1/2)^2/((rho-1/2)^2))
    # Simpler: use the symmetric form around s=1/2

    # Let's test: does the product over N zeros approximate xi?
    print("\n--- Step 4: Product approximation test ---")
    print("  Testing: prod_{j=1}^N (1 - s/rho_j)(1 - s/(1-rho_j)) vs xi(s)/[normalization]")

    test_points = [mpf('2'), mpf('3'), mpf('5'), mpf('0.5') + 3j, mpf('0.5') + 10j]
    test_labels = ["s=2", "s=3", "s=5", "s=0.5+3i", "s=0.5+10i"]

    for s_val, label in zip(test_points, test_labels):
        # Paired product
        prod_val = mpf(1)
        for rho in zeros:
            rho_bar = 1 - rho  # conjugate zero
            factor = (1 - s_val/rho) * (1 - s_val/rho_bar)
            prod_val *= factor

        xi_actual = xi_value(s_val)
        # Normalization: xi(1/2) should give prod = 1 (or close)
        xi_half = xi_value(mpf('0.5'))

        if abs(xi_actual) > 1e-30:
            ratio = prod_val / (xi_actual / xi_half)
            print(f"  {label:15s}: |xi|={float(abs(xi_actual)):.6e}  |prod|={float(abs(prod_val)):.6e}  |ratio|={float(abs(ratio)):.6f}")
        else:
            print(f"  {label:15s}: |xi|={float(abs(xi_actual)):.6e}  |prod|={float(abs(prod_val)):.6e}")

    # Step 5: Near a zero - does the product vanish?
    print("\n--- Step 5: Behavior near first zero ---")
    gamma1 = float(im(zeros[0]))
    for dt in [0.1, 0.01, 0.001, 0.0]:
        s_val = mpf('0.5') + 1j * (gamma1 + dt)
        prod_val = mpf(1)
        for rho in zeros:
            rho_bar = 1 - rho
            prod_val *= (1 - s_val/rho) * (1 - s_val/rho_bar)
        xi_actual = xi_value(s_val)
        print(f"  t = gamma1 + {dt:.3f}: |xi| = {float(abs(xi_actual)):.6e}, |prod| = {float(abs(prod_val)):.6e}")

    # Step 6: The fundamental obstruction
    print("\n--- Step 6: Fundamental Obstruction Summary ---")
    print(f"  Operator R = diag(1/rho_j) on l^2(zeros):")
    print(f"    Trace norm ||R||_1 = sum 1/|rho_j| ~ log^2(N)/2 --> DIVERGES")
    print(f"    HS norm ||R||_2^2 = sum 1/|rho_j|^2 = {float(sum_1_rho2):.6f} --> CONVERGES")
    print(f"  ")
    print(f"  Consequence:")
    print(f"    det_1(I-sR) is NOT defined (R not trace class)")
    print(f"    det_2(I-sR) IS defined (R is Hilbert-Schmidt)")
    print(f"    xi(s) = [scalar factor] * det_2(I-sR)")
    print(f"  ")
    print(f"  For A8' we need det_1(I-L_s) = xi(s). This requires a DIFFERENT")
    print(f"  operator L_s that IS trace class on the critical line.")
    print(f"  The Euler product operator diag(p^{{-s}}) fails because sum p^{{-1/2}} = inf.")
    print(f"  The Hadamard product operator diag(1/rho) fails because sum 1/|rho| = inf.")
    print(f"  ")
    print(f"  Both the 'prime basis' and the 'zero basis' give Hilbert-Schmidt")
    print(f"  but NOT trace-class operators. This is the SAME obstruction from")
    print(f"  two different perspectives.")
    print(f"  ")
    print(f"  Possible resolution: incorporate the archimedean factor (Gamma, pi)")
    print(f"  into the operator to improve convergence from HS to trace class.")
    print(f"  This is Deninger's / Connes' program.")


if __name__ == "__main__":
    main()
