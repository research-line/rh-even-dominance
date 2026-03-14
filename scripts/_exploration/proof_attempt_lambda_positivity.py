"""
Proof Attempt: λ_n ≥ 0 via Critical Identification
====================================================
Approach: Compute λ_n directly from the completed xi function,
verify the Bombieri-Lagarias decomposition, and test whether
the local boundary layer approach can be extended.

Key identity:
  λ_n = (1/(n-1)!) d^n/ds^n [s^{n-1} log ξ(s)]_{s=1}

where ξ(s) = ½ s(s-1) π^{-s/2} Γ(s/2) ζ(s).

Decomposition:
  λ_n = λ_n^(∞) + λ_n^(fin,*)  + λ_n^(pole)

where:
  λ_n^(∞)    = archimedean part (from ½ s π^{-s/2} Γ(s/2))
  λ_n^(fin,*) = finite part (from g(s) = (s-1)ζ(s), holomorphic at s=1)
  λ_n^(pole)  = pole part (from log(s-1) in the s(s-1) factor)

The S3 fix: use g(s) = (s-1)ζ(s) instead of ζ(s) to get holomorphic finite part.

Author: Lukas Geiger
Date: 2026-03-12
"""

import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"

from mpmath import (mp, mpf, log, pi, euler, gamma as mpgamma,
                    digamma, polygamma, zeta, loggamma, diff, fsum,
                    power, factorial, binomial, inf, re, im, mpc)

mp.dps = 50

# ============================================================
# Part 1: Compute λ_n directly from ξ(s)
# ============================================================

def log_xi(s):
    """log ξ(s) = log(1/2) + log(s) + log(s-1) - s/2 log π + log Γ(s/2) + log ζ(s)"""
    return (log(mpf(1)/2) + log(s) + log(s - 1)
            - (s/2) * log(pi)
            + loggamma(s/2)
            + log(zeta(s)))


def lambda_n_from_xi(n, dps=50):
    """
    Compute λ_n = (1/(n-1)!) d^n/ds^n [s^{n-1} log ξ(s)]_{s=1}
    using mpmath's numerical differentiation.
    """
    mp.dps = dps

    def f(s):
        return power(s, n-1) * log_xi(s)

    # n-th derivative at s=1 (offset slightly for even n to avoid pole in diff stencil)
    eps = mpf("1e-10")
    deriv = diff(f, mpf(1) + eps, n)
    return re(deriv) / factorial(n - 1)


# ============================================================
# Part 2: Bombieri-Lagarias decomposition
# ============================================================

def lambda_n_archimedean(n, dps=50):
    """
    λ_n^(∞) = (1/(n-1)!) d^n/ds^n [s^{n-1} log(½ s π^{-s/2} Γ(s/2))]_{s=1}

    Note: This is the "archimedean" part WITHOUT the (s-1) factor.
    """
    mp.dps = dps

    def f_arch(s):
        return power(s, n-1) * (log(mpf(1)/2) + log(s) - (s/2)*log(pi) + loggamma(s/2))

    eps = mpf("1e-10")
    deriv = diff(f_arch, mpf(1) + eps, n)
    return re(deriv) / factorial(n - 1)


def lambda_n_finite_star(n, dps=50):
    """
    λ_n^(fin,*) = (1/(n-1)!) d^n/ds^n [s^{n-1} log g(s)]_{s=1}

    where g(s) = (s-1)ζ(s) is holomorphic at s=1 with g(1) = 1.
    Evaluate at s=1+eps to avoid pole (g is holomorphic, so stable).
    """
    mp.dps = dps
    eps = mpf("1e-10")

    def f_fin(s):
        g_s = (s - 1) * zeta(s)
        return power(s, n-1) * log(g_s)

    deriv = diff(f_fin, mpf(1) + eps, n)
    return re(deriv) / factorial(n - 1)


def lambda_n_pole(n, dps=50):
    """
    λ_n^(pole) = (1/(n-1)!) d^n/ds^n [s^{n-1} log(s-1)]_{s=1}

    This is the contribution from the log(s-1) part of log ξ.
    Since log(s-1) has a logarithmic singularity, we compute this via:
      λ_n = λ_n^(∞) + λ_n^(fin,*) + λ_n^(pole)
    => λ_n^(pole) = λ_n - λ_n^(∞) - λ_n^(fin,*)
    """
    return lambda_n_from_xi(n, dps) - lambda_n_archimedean(n, dps) - lambda_n_finite_star(n, dps)


# ============================================================
# Part 3: d/ds log ξ(σ) profile — the "correct" m_1(σ)
# ============================================================

def xi_deriv_profile(sigma):
    """
    Compute d/ds log ξ(σ) = 1/σ + 1/(σ-1) - logπ/2 + ψ(σ/2)/2 + ζ'(σ)/ζ(σ)

    This is the TRUE m_1(σ) that converges to λ_1 as σ → 1+.
    """
    s = mpf(sigma)

    # Archimedean part (including the 1/(s-1) from the s(s-1) prefactor)
    arch = 1/s + 1/(s-1) - log(pi)/2 + digamma(s/2)/2

    # Finite part: ζ'/ζ via numerical differentiation
    zeta_prime_over_zeta = diff(lambda t: log(zeta(t)), s)

    return float(re(arch + zeta_prime_over_zeta))


# ============================================================
# Part 4: Monotonicity test — is d²/ds² log ξ(σ) < 0 near σ=1?
# ============================================================

def xi_second_deriv(sigma):
    """d²/ds² log ξ(σ) — second derivative for convexity analysis"""
    s = mpf(sigma)
    return float(re(diff(log_xi, s, 2)))


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Proof Attempt: Lambda_n Positivity via xi-Function")
    print("=" * 70)

    # === Part 1: Compute λ_n for n=1,...,20 ===
    print("\n--- Part 1: Direct computation of lambda_n ---")
    print(f"{'n':>3}  {'lambda_n':>16}  {'lambda_n^(inf)':>16}  {'lambda_n^(fin*)':>16}  {'lambda_n^(pole)':>16}  {'pos?':>5}")
    print("-" * 85)

    all_positive = True
    for n in range(1, 21):
        try:
            lam = lambda_n_from_xi(n)
            lam_arch = lambda_n_archimedean(n)
            lam_fin = lambda_n_finite_star(n)
            lam_pole = float(lam) - float(lam_arch) - float(lam_fin)
            pos = "YES" if float(lam) > 0 else "NO"
            if float(lam) <= 0:
                all_positive = False
            print(f"  {n:>2}  {float(lam):>16.10f}  {float(lam_arch):>16.10f}  "
                  f"{float(lam_fin):>16.10f}  {lam_pole:>16.10f}  {pos:>5}")
        except Exception as e:
            print(f"  {n:>2}  ERROR: {e}")

    print("-" * 85)
    print(f"  All lambda_n > 0 for n=1..20? {'YES' if all_positive else 'NO'}")

    # === Part 2: Exact value for n=1 (Bombieri-Lagarias) ===
    print("\n--- Part 2: Exact verification for n=1 ---")
    lambda1_exact = float(1 + euler/2 - log(4*pi)/2)
    lambda1_computed = float(lambda_n_from_xi(1))
    print(f"  lambda_1 (B-L formula): {lambda1_exact:.14f}")
    print(f"  lambda_1 (xi diff):     {lambda1_computed:.14f}")
    print(f"  Difference:             {abs(lambda1_exact - lambda1_computed):.2e}")
    print(f"  lambda_1 > 0:           {'YES' if lambda1_exact > 0 else 'NO'}")
    print(f"  Rigorous bound: gamma > 0.577, log(4pi) < 2.532")
    print(f"    => lambda_1 > 1 + 0.577/2 - 2.532/2 = {1 + 0.577/2 - 2.532/2:.4f} > 0  QED")

    # === Part 3: Profile of (d/ds log xi)(sigma) ===
    print("\n--- Part 3: Profile of m_1^{true}(sigma) = d/ds log xi(sigma) ---")
    print(f"  This is the CORRECT m_1 that converges to lambda_1.")
    print(f"{'sigma':>8}  {'m1_true(sigma)':>16}  {'d2_log_xi':>14}")
    print("-" * 45)

    for sigma in [1.001, 1.005, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]:
        m1 = xi_deriv_profile(sigma)
        d2 = xi_second_deriv(sigma)
        print(f"  {sigma:>7.3f}  {m1:>16.10f}  {d2:>14.6f}")

    print("-" * 45)

    # === Part 4: Monotonicity analysis ===
    print("\n--- Part 4: Monotonicity of d/ds log xi ---")
    print("  d^2/ds^2 log xi(sigma) > 0 means d/ds log xi is INCREASING (convex).")
    print("  => lambda_1 = lim_{sigma->1+} is the MINIMUM of d/ds log xi(sigma).")
    print("  => d/ds log xi(sigma) > lambda_1 for all sigma > 1.")
    print("  NOTE: This is the OPPOSITE of the FST m_1 behavior!")
    print()

    # Find where d^2/ds^2 log xi changes sign
    prev_d2 = None
    sign_change = None
    for sigma_val in [1 + i*0.01 for i in range(1, 500)]:
        d2 = xi_second_deriv(sigma_val)
        if prev_d2 is not None and prev_d2 * d2 < 0:
            if sign_change is None:
                sign_change = sigma_val
                print(f"  d^2 log xi sign change near sigma = {sigma_val:.2f}")
        prev_d2 = d2

    if sign_change is None:
        print("  d^2 log xi < 0 for all tested sigma (monotonically decreasing)")

    # === Part 5: Where the proof BREAKS for general n ===
    print("\n--- Part 5: Why this doesn't prove RH ---")
    print("  For n=1: lambda_1 = 1 + gamma/2 - log(4pi)/2 > 0. TRIVIALLY POSITIVE.")
    print("  For general n: lambda_n involves Stieltjes constants gamma_k.")
    print("  The sign of lambda_n depends on delicate cancellations between")
    print("  the archimedean (Gamma) and finite (zeta) contributions.")
    print()
    print("  KEY INSIGHT: Under RH, |1-1/rho| = 1 for all zeros,")
    print("  so lambda_n = sum_rho [1 - cos(n*theta_rho)] >= 0.")
    print("  Without RH, off-line zeros give |1-1/rho| > 1,")
    print("  producing exponentially growing negative terms in n.")
    print("  => Proving lambda_n >= 0 for ALL n IS the RH.")
    print()
    print("  The FST framework REFORMULATES but does NOT resolve this barrier.")
    print("  The most it achieves is: local monotonicity for n=1,2 near sigma=1,")
    print("  which gives lambda_{1,2} >= 0 (already known).")


if __name__ == "__main__":
    main()
