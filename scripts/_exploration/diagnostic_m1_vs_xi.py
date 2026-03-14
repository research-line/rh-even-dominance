"""
Diagnostic: Compare m_1(sigma) from FST-RH formula vs direct xi(s) computation.
================================================================================
Goal: Verify that lim_{sigma->1+} m_1(sigma) = lambda_1 = 0.02310...

Approach 1: Compute xi(s) = 1/2 * s * (s-1) * pi^{-s/2} * Gamma(s/2) * zeta(s)
            Then lambda_1 = [d/ds log xi(s)]_{s=1}
            Approximate via finite difference at sigma close to 1.

Approach 2: FST-RH formula m_1(sigma) = E_mu[X_{1,sigma}] - h_1(sigma)
            with X_{1,sigma}(p) = -log(p)/(p^sigma - 1)
            and h_1(sigma) = 1/sigma - log(pi)/2 + psi(sigma/2)/2

Compare both to identify any discrepancy.
"""

import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"

from mpmath import (mp, mpf, log, power, pi, euler, gamma as mpgamma,
                    digamma, zeta, loggamma, diff, fsum, inf)
import sympy

mp.dps = 50

def generate_primes(n):
    primes = []
    p = 2
    while len(primes) < n:
        primes.append(p)
        p = sympy.nextprime(p)
    return primes


def log_xi(s):
    """Compute log xi(s) where xi(s) = 1/2 * s * (s-1) * pi^{-s/2} * Gamma(s/2) * zeta(s)"""
    return (log(mpf(1)/2) + log(s) + log(s - 1)
            - (s/2) * log(pi)
            + loggamma(s/2)
            + log(zeta(s)))


def lambda_n_from_xi(n):
    """Compute lambda_n = (1/(n-1)!) * d^n/ds^n [s^{n-1} log xi(s)] at s=1"""
    if n == 1:
        # lambda_1 = d/ds [log xi(s)] at s=1
        # Use mpmath's numerical differentiation
        return diff(log_xi, mpf(1))
    else:
        raise NotImplementedError


def compute_m1_fst(primes, sigma):
    """Compute m_1(sigma) using FST-RH formula."""
    s = mpf(sigma)

    # Prime sums
    A_terms = []
    P_terms = []
    for p in primes:
        p_mp = mpf(p)
        ps = power(p_mp, s)
        logp = log(p_mp)
        inv_ps = mpf(1) / ps
        ps_m1 = ps - 1

        P_terms.append(inv_ps)
        A_terms.append(logp * inv_ps / ps_m1)

    P = fsum(P_terms)
    A = fsum(A_terms)

    E_X = -A / P

    # h_1(sigma) = 1/sigma - log(pi)/2 + psi(sigma/2)/2
    h1 = 1/s - log(pi)/2 + digamma(s/2)/2

    m1 = E_X - h1

    return float(m1), float(E_X), float(h1), float(P), float(A)


def compute_xi_derivative(sigma):
    """Compute d/ds [log xi(s)] at s=sigma numerically."""
    s = mpf(sigma)
    return float(diff(log_xi, s))


def main():
    print("=" * 70)
    print("Diagnostic: m_1(sigma) vs xi-function derivative")
    print("=" * 70)

    # Exact lambda_1
    lambda1_BL = float(1 + euler/2 - log(4*pi)/2)
    print(f"\nlambda_1 (Bombieri-Lagarias) = {lambda1_BL:.12f}")

    # lambda_1 from xi directly
    lambda1_xi = lambda_n_from_xi(1)
    print(f"lambda_1 (mpmath xi diff)    = {float(lambda1_xi.real):.12f}")

    # Generate primes
    N = 10000
    primes = generate_primes(N)
    print(f"\nUsing N = {N} primes (p_max = {primes[-1]})")

    # Compare at various sigma
    print()
    print("-" * 110)
    print(f"{'sigma':>10}  {'m1_FST':>14}  {'xi_deriv':>14}  {'diff':>14}  "
          f"{'E[X]':>14}  {'h1':>14}  {'P':>10}")
    print("-" * 110)

    test_sigmas = [1.001, 1.002, 1.005, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5,
                   2.0, 3.0, 5.0, 10.0]

    for sigma in test_sigmas:
        m1, E_X, h1, P, A = compute_m1_fst(primes, sigma)
        xi_d = compute_xi_derivative(sigma)
        d = m1 - xi_d

        print(f"  {sigma:>8.3f}  {m1:>14.8f}  {xi_d:>14.8f}  {d:>14.8e}  "
              f"{E_X:>14.8f}  {h1:>14.8f}  {P:>10.4f}")

    print("-" * 110)

    # Now check: what SHOULD the FST-RH m_1 converge to?
    print()
    print("Analysis:")
    print(f"  h_1(1) = {float(1 - log(pi)/2 + digamma(mpf(1)/2)/2):.8f}")
    print(f"  psi(1/2) = {float(digamma(mpf(1)/2)):.8f}")
    print(f"  gamma (Euler) = {float(euler):.8f}")

    # The issue: log xi = log(archimedean) + log((s-1)*zeta(s))
    # h_1 captures d/ds log(archimedean)
    # E[X] should capture d/ds log((s-1)*zeta(s))
    # But E[X] -> 0 as sigma -> 1+, while d/ds log((s-1)*zeta(s))|_{s=1} = gamma

    # Check what d/ds log xi actually gives at various sigma
    print()
    print("xi-derivative profile:")
    for sigma in [1.001, 1.01, 1.05, 1.1, 1.5, 2.0, 5.0]:
        xd = compute_xi_derivative(sigma)
        print(f"  (d/ds log xi)({sigma:.3f}) = {xd:.8f}")


if __name__ == "__main__":
    main()
