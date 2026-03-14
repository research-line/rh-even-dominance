#!/usr/bin/env python3
"""
weg2_server_gap.py
==================
Server-Berechnung: Gap(lambda) mit konvergiertem N.

Fuer jeden lambda-Wert wird N so gewaehlt, dass die Luecke konvergiert:
  N_min = max(50, 3 * ceil(log(lambda)))

Zusaetzlich: N-Konvergenz-Check bei jedem lambda.
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time
import json

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

def make_basis_grid(N, t_grid, L):
    phi = np.zeros((N, len(t_grid)))
    phi[0, :] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N):
        phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi

def make_shifted(N, t_grid, L, shift):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    phi[0, mask] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N):
        phi[n, mask] = np.cos(n * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi

def build_QW(lam, N, primes, M_terms=12, n_quad=800, n_int=500):
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = make_basis_grid(N, t_grid, L)

    W = LOG4PI_GAMMA * np.eye(N)
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        k = 1.0 / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_shifted(N, t_grid, L, s).T) * dt
        Sm = (phi @ make_shifted(N, t_grid, L, -s).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s/2) * np.eye(N)) * k * ds

    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_shifted(N, t_grid, L, sign * shift).T) * dt
                W += coeff * S

    return W

def compute_gap_converged(lam, primes, N_start=30, N_max=120, N_step=10,
                          tol=0.05, n_quad=800, n_int=500):
    """Berechne gap mit N-Konvergenz-Check."""
    L = np.log(lam)
    results = []

    prev_gap = None
    for N in range(N_start, N_max + 1, N_step):
        QW = build_QW(lam, N, primes, n_quad=n_quad, n_int=n_int)
        evals = np.sort(eigh(QW, eigvals_only=True))
        gap = evals[1] - evals[0]
        lmin = evals[0]
        l2 = evals[1]

        results.append({'N': N, 'gap': gap, 'lmin': lmin, 'l2': l2})

        if prev_gap is not None:
            rel_change = abs(gap - prev_gap) / max(abs(gap), 1e-10)
            converged = rel_change < tol
        else:
            converged = False

        prev_gap = gap

        if converged and N >= N_start + 2 * N_step:
            break

    return results

if __name__ == "__main__":
    print("=" * 75)
    print("SERVER-BERECHNUNG: GAP(LAMBDA) MIT N-KONVERGENZ")
    print("=" * 75)

    all_primes = list(primerange(2, 500))

    lambdas = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 170, 200]

    all_results = {}

    for lam in lambdas:
        t0 = time.time()
        L = np.log(lam)
        primes_used = [p for p in all_primes if p <= max(lam, 47)]

        N_start = max(30, int(2.5 * L))
        N_max = max(80, int(5 * L))
        N_step = 10

        print(f"\n{'='*60}")
        print(f"  lambda={lam}, L={2*L:.3f}, N_range=[{N_start}, {N_max}], "
              f"primes={len(primes_used)}")

        results = compute_gap_converged(
            lam, primes_used, N_start=N_start, N_max=N_max, N_step=N_step
        )

        for r in results:
            print(f"    N={r['N']:3d}: gap={r['gap']:.6e}, "
                  f"lmin={r['lmin']:+.6e}, l2={r['l2']:+.6e}")

        elapsed = time.time() - t0
        final = results[-1]
        print(f"  => Konvergiert bei N={final['N']}, gap={final['gap']:.6e} ({elapsed:.1f}s)")

        all_results[lam] = {
            'L': 2*L, 'n_primes': len(primes_used),
            'N_final': final['N'], 'gap': final['gap'],
            'lmin': final['lmin'], 'l2': final['l2'],
            'all_N': results
        }

    # Zusammenfassung
    print(f"\n{'='*75}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*75}")
    print(f"\n  {'lambda':>6} | {'L':>6} | {'N_conv':>6} | {'gap':>12} | "
          f"{'lmin':>12} | {'l2':>12} | {'gap/L':>8}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    for lam in lambdas:
        r = all_results[lam]
        gap_L = r['gap'] / r['L'] if r['L'] > 0 else 0
        print(f"  {lam:6d} | {r['L']:6.3f} | {r['N_final']:6d} | {r['gap']:12.6e} | "
              f"{r['lmin']:+12.6e} | {r['l2']:+12.6e} | {gap_L:8.4f}")

    # Speichere Ergebnisse
    outfile = "gap_results_server.json"
    with open(outfile, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\n  Ergebnisse gespeichert: {outfile}")
