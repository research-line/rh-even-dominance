"""
FST-RH Strategy B (corrected): Direct computation of m_1(sigma)
================================================================
Computes m_1(sigma) = E_mu[X_{1,sigma}] - h_1(sigma) directly.

The critical identification says: lim_{sigma->1+} m_1(sigma) = lambda_1.
If m_1(sigma) >= 0 for all sigma > 1 in a neighborhood of 1, then lambda_1 >= 0.

Also computes m_1'(sigma) with the FULL formula including h_1'(sigma).

Definitions:
  X_{1,sigma}(p) = -log(p) / (p^sigma - 1)
  E_mu[X] = -(1/P) * sum_p log(p) * p^{-sigma} / (p^sigma - 1) = -A(sigma)/P(sigma)
  h_1(sigma) = 1/sigma - log(pi)/2 + psi(sigma/2)/2
  m_1(sigma) = -A/P - h_1(sigma)

  m_1'(sigma) = -[A*B - P*(C+D)] / P^2 - h_1'(sigma)
  h_1'(sigma) = psi'(sigma/2)/4 - 1/sigma^2

Author: Lukas Geiger
Date: 2026-03-12
"""

import os, sys, time, json
os.environ["PYTHONIOENCODING"] = "utf-8"

from mpmath import (mp, mpf, log, power, fsum, pi, euler,
                    digamma, polygamma, inf)
import sympy

mp.dps = 50

N_PRIMES = 10000

def generate_primes(n):
    primes = []
    p = 2
    while len(primes) < n:
        primes.append(p)
        p = sympy.nextprime(p)
    return primes

def compute_all(primes, sigma):
    """Compute m_1(sigma), m_1'(sigma), and all components."""
    s = mpf(sigma)

    P_terms, A_terms, B_terms, C_terms, D_terms = [], [], [], [], []

    for p in primes:
        p_mp = mpf(p)
        ps = power(p_mp, s)
        logp = log(p_mp)
        logp2 = logp * logp
        inv_ps = mpf(1) / ps
        ps_m1 = ps - 1

        P_terms.append(inv_ps)
        A_terms.append(logp * inv_ps / ps_m1)
        B_terms.append(logp * inv_ps)
        C_terms.append(logp2 * inv_ps / ps_m1)
        D_terms.append(logp2 / (ps_m1 * ps_m1))

    P = fsum(P_terms)
    A = fsum(A_terms)
    B = fsum(B_terms)
    C = fsum(C_terms)
    D = fsum(D_terms)

    # Archimedean terms
    # h_1(sigma) = 1/sigma - log(pi)/2 + psi(sigma/2)/2
    psi_half_s = digamma(s / 2)
    h1 = 1/s - log(pi)/2 + psi_half_s/2

    # h_1'(sigma) = psi'(sigma/2)/4 - 1/sigma^2
    # psi'(x) = polygamma(1, x) = trigamma
    psi1_half_s = polygamma(1, s / 2)
    h1_prime = psi1_half_s / 4 - 1/(s*s)

    # m_1(sigma) = -A/P - h_1
    E_X = -A / P
    m1 = E_X - h1

    # m_1'(sigma) = -[AB - P(C+D)] / P^2 - h_1'(sigma)
    F_arith = A * B - P * (C + D)  # arithmetic part
    m1_prime_arith = -F_arith / (P * P)  # arithmetic derivative
    m1_prime = m1_prime_arith - h1_prime  # full derivative

    # Also compute lambda_1 exactly (Bombieri-Lagarias)
    # lambda_1 = 1 + gamma/2 - log(4*pi)/2
    lambda1_exact = 1 + euler/2 - log(4*pi)/2

    return {
        "sigma": float(s),
        "m1": float(m1),
        "m1_prime": float(m1_prime),
        "m1_prime_arith": float(m1_prime_arith),
        "h1_prime": float(h1_prime),
        "F_arith": float(F_arith),
        "E_X": float(E_X),
        "h1": float(h1),
        "P": float(P),
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D),
        "lambda1_exact": float(lambda1_exact),
    }


def main():
    print("=" * 70)
    print("FST-RH: Direct Computation of m_1(sigma)")
    print("=" * 70)

    # Generate primes
    print(f"\nGenerating {N_PRIMES} primes...")
    t0 = time.time()
    primes = generate_primes(N_PRIMES)
    print(f"  Done in {time.time()-t0:.2f}s. Largest: {primes[-1]}")

    # Compute lambda_1 exactly
    mp.dps = 50
    lambda1 = float(1 + euler/2 - log(4*pi)/2)
    print(f"\n  lambda_1 (exact) = {lambda1:.10f}")

    # Build sigma grid - dense near 1, sparser further out
    sigmas = []
    # Very dense near 1
    for i in range(1, 100):
        sigmas.append(1 + i * 0.001)
    # Medium density
    for i in range(10, 200):
        sigmas.append(1 + i * 0.01)
    # Coarse
    for i in range(30, 101):
        sigmas.append(1 + i * 0.1)
    sigmas = sorted(set([round(s, 4) for s in sigmas if s > 1.0]))

    print(f"\nEvaluating at {len(sigmas)} sigma values [{sigmas[0]:.3f}, {sigmas[-1]:.1f}]")
    print()

    results = []
    min_m1 = float("inf")
    min_m1_sigma = None
    m1_sign_changes = []
    m1prime_sign_changes = []

    t_start = time.time()
    prev_result = None

    for idx, sigma in enumerate(sigmas):
        r = compute_all(primes, sigma)
        results.append(r)

        if r["m1"] < min_m1:
            min_m1 = r["m1"]
            min_m1_sigma = sigma

        # Track sign changes
        if prev_result is not None:
            if prev_result["m1"] * r["m1"] < 0:
                m1_sign_changes.append((prev_result["sigma"], sigma))
            if prev_result["m1_prime"] * r["m1_prime"] < 0:
                m1prime_sign_changes.append((prev_result["sigma"], sigma))

        prev_result = r

        # Progress
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 1
            eta = (len(sigmas) - idx - 1) / rate
            print(f"  [{idx+1}/{len(sigmas)}] sigma={sigma:.4f}, "
                  f"m1={r['m1']:.8e}, m1'={r['m1_prime']:.6e}  "
                  f"({elapsed:.1f}s, ~{eta:.0f}s left)")

    t_total = time.time() - t_start

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  N = {N_PRIMES}, p_max = {primes[-1]}, precision = {mp.dps} digits")
    print(f"  Total time: {t_total:.1f}s")
    print(f"  lambda_1 (exact) = {lambda1:.10f}")
    print()

    # m_1 near sigma=1 (convergence check)
    print("  Convergence of m_1(sigma) -> lambda_1:")
    for r in results:
        if r["sigma"] in [1.001, 1.002, 1.005, 1.01, 1.02, 1.05, 1.1]:
            diff = r["m1"] - lambda1
            print(f"    m_1({r['sigma']:.3f}) = {r['m1']:.8f}  "
                  f"(diff from lambda_1: {diff:+.6e})")

    print()
    print(f"  min m_1(sigma):      {min_m1:.10e}  at sigma = {min_m1_sigma:.4f}")
    print(f"  min m_1 > 0?         {'YES' if min_m1 > 0 else 'NO'}")

    print()
    if m1_sign_changes:
        print(f"  m_1 sign changes at: {m1_sign_changes}")
    else:
        if min_m1 > 0:
            print("  m_1(sigma) > 0 for ALL tested sigma (stronger than needed!)")
        else:
            print("  m_1(sigma) < 0 for some sigma (normal, only limit matters)")

    print()
    if m1prime_sign_changes:
        print(f"  m_1' sign changes (monotonicity transitions):")
        for a, b in m1prime_sign_changes:
            print(f"    between sigma = {a:.4f} and {b:.4f}")

    # Detailed table
    print()
    print("Detailed profile:")
    print("-" * 100)
    print(f"{'sigma':>8}  {'m_1(sigma)':>14}  {'m_1_prime':>14}  {'m1_arith':>14}  "
          f"{'h1_prime':>12}  {'F_arith':>14}")
    print("-" * 100)

    shown = set()
    for r in results:
        s = r["sigma"]
        # Show selected values
        show = False
        for target in [1.001, 1.005, 1.01, 1.02, 1.05, 1.1, 1.15, 1.17, 1.2,
                       1.3, 1.5, 2.0, 3.0, 5.0, 10.0]:
            if abs(s - target) < 0.002 and target not in shown:
                show = True
                shown.add(target)
        if s == min_m1_sigma:
            show = True

        if show:
            marker = " <-- min" if s == min_m1_sigma else ""
            print(f"  {s:>7.3f}  {r['m1']:>14.8e}  {r['m1_prime']:>14.8e}  "
                  f"{r['m1_prime_arith']:>14.8e}  {r['h1_prime']:>12.8f}  "
                  f"{r['F_arith']:>14.8e}{marker}")

    print("-" * 100)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "m1_direct_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "N_primes": N_PRIMES,
                "p_max": primes[-1],
                "precision": mp.dps,
            },
            "lambda1_exact": lambda1,
            "summary": {
                "min_m1": min_m1,
                "min_m1_sigma": min_m1_sigma,
                "m1_sign_changes": m1_sign_changes,
                "m1prime_sign_changes": m1prime_sign_changes,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
