#!/usr/bin/env python3
"""
lambda_n_large_scale.py
=======================
Compute lambda_n for n=1..100 using mpmath high-precision arithmetic.

lambda_n = (1/(n-1)!) * d^n/ds^n [ s^{n-1} log xi(s) ] |_{s=1}

where xi(s) = 1/2 * s * (s-1) * pi^{-s/2} * Gamma(s/2) * zeta(s)

Decomposes into archimedean and finite parts, analyzes generating
function structure, complete monotonicity, and growth rates.
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import traceback
from mpmath import (
    mp, mpf, log, pi, gamma, zeta, diff, factorial, fsum, power, inf
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def log_xi(s):
    """log xi(s) = log(1/2) + log(s) + log(s-1) + (-s/2)*log(pi) + loggamma(s/2) + log(zeta(s))
    We use loggamma to avoid overflow issues with Gamma."""
    from mpmath import loggamma, log as mlog
    return (mlog(mpf('0.5')) + mlog(s) + mlog(s - 1)
            + (-s / 2) * mlog(pi) + loggamma(s / 2) + mlog(zeta(s)))


def log_arch(s):
    """Archimedean part: log( 1/2 * s * pi^{-s/2} * Gamma(s/2) )"""
    from mpmath import loggamma, log as mlog
    return mlog(mpf('0.5')) + mlog(s) + (-s / 2) * mlog(pi) + loggamma(s / 2)


def log_fin(s):
    """Finite part: log( (s-1)*zeta(s) ) = log(g(s))
    g(s) = (s-1)*zeta(s) is entire and g(1)=1."""
    from mpmath import log as mlog
    return mlog((s - 1) * zeta(s))


def compute_lambda_n(n, func, s0, dps_use):
    """
    Compute lambda_n = (1/(n-1)!) * [d^n/ds^n (s^{n-1} * func(s))] at s=s0

    func is one of log_xi, log_arch, log_fin.
    """
    old_dps = mp.dps
    mp.dps = dps_use

    try:
        def integrand(s):
            return power(s, n - 1) * func(s)

        # n-th derivative at s0
        deriv_n = diff(integrand, s0, n)
        lam = deriv_n / factorial(n - 1)
        result = mpf(lam)
    except Exception:
        result = None
    finally:
        mp.dps = old_dps

    return result


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main():
    N_MAX = 100
    s0 = mpf('1') + mpf('1e-10')  # small offset from the pole at s=1

    print("=" * 100)
    print(f"  lambda_n Large-Scale Computation  (n = 1 .. {N_MAX})")
    print(f"  Evaluation point: s0 = 1 + 1e-10")
    print("=" * 100)
    print()

    # Storage
    results = []  # list of dicts
    t_start = time.time()
    TIME_LIMIT = 290  # 4 min 50 sec safety margin

    for n in range(1, N_MAX + 1):
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT:
            print(f"\n*** Time limit reached after {elapsed:.0f}s at n={n-1}. Printing results so far. ***\n")
            break

        dps_use = max(100, 2 * n + 20)  # extra margin

        try:
            lam_full = compute_lambda_n(n, log_xi, s0, dps_use)
            lam_arch = compute_lambda_n(n, log_arch, s0, dps_use)
            lam_fin = compute_lambda_n(n, log_fin, s0, dps_use)

            if lam_full is not None:
                c_n = lam_full / n
                from mpmath import log as mlog
                mp.dps = 50
                if n >= 2:
                    ratio = lam_full / (mpf(n) / 2 * mlog(n))
                else:
                    ratio = None
                mp.dps = 15
            else:
                c_n = None
                ratio = None

            results.append({
                'n': n,
                'lam': lam_full,
                'lam_arch': lam_arch,
                'lam_fin': lam_fin,
                'c_n': c_n,
                'ratio': ratio,
            })

            # Progress indicator every 5
            if n % 5 == 0 or n <= 5 or n == N_MAX:
                lam_str = f"{float(lam_full):.10e}" if lam_full is not None else "FAILED"
                dt = time.time() - t_start
                print(f"  n={n:3d}  lambda_n={lam_str:>22s}  ({dt:.1f}s elapsed)")

        except Exception as e:
            results.append({
                'n': n, 'lam': None, 'lam_arch': None,
                'lam_fin': None, 'c_n': None, 'ratio': None,
            })
            print(f"  n={n:3d}  ERROR: {e}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 130)
    print(f"{'n':>4s} | {'lambda_n':>20s} | {'lambda_n^(arch)':>20s} | {'lambda_n^(fin)':>20s} | {'c_n=lambda_n/n':>20s} | {'lam/(n*ln(n)/2)':>18s}")
    print("-" * 130)

    for r in results:
        n = r['n']
        def fmt(x):
            if x is None:
                return "FAILED"
            v = float(x)
            if abs(v) < 1e-6 or abs(v) > 1e6:
                return f"{v:.10e}"
            else:
                return f"{v:.12f}"

        print(f"{n:4d} | {fmt(r['lam']):>20s} | {fmt(r['lam_arch']):>20s} | {fmt(r['lam_fin']):>20s} | {fmt(r['c_n']):>20s} | {fmt(r['ratio']) if r['ratio'] is not None else 'N/A':>18s}")

    # -----------------------------------------------------------------------
    # Analysis: c_n positivity
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  ANALYSIS 1: Positivity of c_n = lambda_n / n")
    print("=" * 80)

    c_vals = [(r['n'], r['c_n']) for r in results if r['c_n'] is not None]
    all_positive = True
    first_negative = None
    for n_val, c_val in c_vals:
        if float(c_val) <= 0:
            all_positive = False
            if first_negative is None:
                first_negative = (n_val, float(c_val))

    if all_positive:
        print(f"  ALL c_n > 0 for n=1..{len(c_vals)}  --> Generating function has positive coefficients!")
    else:
        print(f"  NEGATIVE c_n found! First at n={first_negative[0]}: c_n = {first_negative[1]:.6e}")

    # -----------------------------------------------------------------------
    # Analysis: Complete monotonicity of c_n
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  ANALYSIS 2: Complete Monotonicity of c_n (Path D2)")
    print("=" * 80)
    print("  Check: (-1)^k Delta^k c_n >= 0 for all k, n")

    c_list = [float(r['c_n']) for r in results if r['c_n'] is not None]
    N_cm = len(c_list)

    # Forward differences: Delta^k c_n = sum_{j=0}^{k} (-1)^{k-j} C(k,j) c_{n+j}
    # Complete monotone: (-1)^k Delta^k c_n >= 0
    max_k = min(10, N_cm - 1)
    cm_ok = True
    cm_violations = []

    for k in range(0, max_k + 1):
        for start in range(N_cm - k):
            # Delta^k c_{start}
            val = 0.0
            from math import comb
            for j in range(k + 1):
                val += ((-1) ** (k - j)) * comb(k, j) * c_list[start + j]
            # (-1)^k * Delta^k should be >= 0
            test_val = ((-1) ** k) * val
            if test_val < -1e-20:  # small tolerance
                cm_ok = False
                cm_violations.append((k, start + 1, test_val))  # n is 1-indexed

    if cm_ok:
        print(f"  COMPLETE MONOTONICITY HOLDS for k=0..{max_k} across all n!")
        print("  --> c_n is completely monotone, Stieltjes transform representation exists.")
    else:
        print(f"  Complete monotonicity VIOLATED at {len(cm_violations)} point(s).")
        for k, n_val, tv in cm_violations[:10]:
            print(f"    k={k}, n={n_val}: (-1)^k Delta^k c_n = {tv:.6e}")

    # -----------------------------------------------------------------------
    # Analysis: Growth rate
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  ANALYSIS 3: Growth Rate")
    print("=" * 80)
    print(f"  Under RH: lambda_n ~ (n/2) * log(n)")
    print()
    print(f"  {'n':>4s} | {'lambda_n':>16s} | {'lambda_n/n':>16s} | {'lam/(n*ln(n)/2)':>18s} | {'log(lam)/n':>14s}")
    print(f"  " + "-" * 80)

    import math
    for r in results:
        n = r['n']
        lam = r['lam']
        if lam is None:
            continue
        lam_f = float(lam)
        lam_over_n = lam_f / n if n > 0 else 0
        if n >= 2:
            ratio_val = lam_f / (n * math.log(n) / 2)
        else:
            ratio_val = float('nan')
        if lam_f > 0:
            log_lam_over_n = math.log(lam_f) / n
        else:
            log_lam_over_n = float('nan')

        if n <= 20 or n % 10 == 0:
            print(f"  {n:4d} | {lam_f:16.8f} | {lam_over_n:16.8f} | {ratio_val:18.8f} | {log_lam_over_n:14.8f}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)

    valid = [r for r in results if r['lam'] is not None]
    positive_lam = [r for r in valid if float(r['lam']) > 0]
    print(f"  Computed: n=1..{len(results)}")
    print(f"  Valid results: {len(valid)}")
    print(f"  lambda_n > 0: {len(positive_lam)} / {len(valid)}")
    if len(positive_lam) == len(valid) and len(valid) > 0:
        print(f"  --> ALL lambda_n POSITIVE (consistent with Li's criterion / RH)")
    print(f"  c_n all positive: {all_positive}")
    print(f"  Complete monotonicity (k<=10): {'YES' if cm_ok else 'NO'}")

    total_time = time.time() - t_start
    print(f"\n  Total computation time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
