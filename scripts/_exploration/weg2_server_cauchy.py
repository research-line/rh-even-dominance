#!/usr/bin/env python3
"""
weg2_server_cauchy.py
=====================
Server-Berechnung: Cauchy-Interlacing Gap-Schranke mit N-Konvergenz.

Fuer jeden lambda-Wert:
  1. Berechne QW mit wachsendem N bis Gap konvergiert
  2. Bestimme dominante Mode n* des Grundzustands
  3. Berechne Cauchy-Schranke: gap >= lambda_1(QW_red) - lambda_1(QW)
  4. Verifiziere: Cauchy-Schranke > 0

Zusaetzlich: |v0(n*)|^2 als Funktion von N (konvergiert das?)
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

def cauchy_analysis(QW, N):
    """Vollstaendige Cauchy-Interlacing-Analyse."""
    evals, evecs = eigh(QW)
    v0 = evecs[:, 0]
    gap = evals[1] - evals[0]

    # Dominante Mode
    n_star = np.argmax(np.abs(v0))
    weight_star = v0[n_star]**2

    # Top-3 Moden
    top3 = np.argsort(np.abs(v0))[::-1][:3]
    top3_weights = v0[top3]**2

    # Cauchy-Schranke: Streiche n_star
    mask = np.ones(N, dtype=bool)
    mask[n_star] = False
    QW_red = QW[np.ix_(mask, mask)]
    evals_red = np.sort(eigh(QW_red, eigvals_only=True))
    cauchy_bound = evals_red[0] - evals[0]

    # Auch Streiche top-2
    mask2 = np.ones(N, dtype=bool)
    mask2[top3[0]] = False
    mask2[top3[1]] = False
    QW_red2 = QW[np.ix_(mask2, mask2)]
    evals_red2 = np.sort(eigh(QW_red2, eigvals_only=True))
    cauchy2 = evals_red2[0] - evals[0]

    return {
        'gap': gap,
        'lmin': evals[0],
        'l2': evals[1],
        'n_star': int(n_star),
        'weight_star': float(weight_star),
        'top3_modes': [int(x) for x in top3],
        'top3_weights': [float(x) for x in top3_weights],
        'cauchy_bound': float(cauchy_bound),
        'cauchy2_bound': float(cauchy2),
        'lmin_red': float(evals_red[0]),
    }

def compute_converged(lam, primes, N_start=40, N_max=120, N_step=10,
                      tol=0.05, n_quad=800, n_int=500):
    """Berechne Cauchy-Schranke mit N-Konvergenz."""
    results = []
    prev_cauchy = None

    for N in range(N_start, N_max + 1, N_step):
        QW = build_QW(lam, N, primes, n_quad=n_quad, n_int=n_int)
        analysis = cauchy_analysis(QW, N)
        analysis['N'] = N
        results.append(analysis)

        cauchy = analysis['cauchy_bound']
        if prev_cauchy is not None:
            rel_change = abs(cauchy - prev_cauchy) / max(abs(cauchy), 1e-10)
            analysis['rel_change'] = float(rel_change)
            converged = rel_change < tol
        else:
            analysis['rel_change'] = None
            converged = False

        prev_cauchy = cauchy

        if converged and N >= N_start + 2 * N_step:
            break

    return results

if __name__ == "__main__":
    print("=" * 75)
    print("SERVER: CAUCHY-INTERLACING MIT N-KONVERGENZ")
    print("=" * 75)

    all_primes = list(primerange(2, 500))

    lambdas = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 170, 200]

    all_results = {}

    for lam in lambdas:
        t0 = time.time()
        L = np.log(lam)
        primes_used = [p for p in all_primes if p <= max(lam, 47)]

        N_start = max(40, int(3 * L))
        N_max = max(100, int(6 * L))
        N_step = 10

        print(f"\n{'='*60}")
        print(f"  lambda={lam}, L={2*L:.3f}, N_range=[{N_start}, {N_max}], "
              f"primes={len(primes_used)}")

        results = compute_converged(
            lam, primes_used, N_start=N_start, N_max=N_max, N_step=N_step
        )

        for r in results:
            rc = f"rc={r['rel_change']:.3f}" if r['rel_change'] is not None else "rc=---"
            print(f"    N={r['N']:3d}: gap={r['gap']:.6e}, cauchy={r['cauchy_bound']:.6e}, "
                  f"|v0(n*)|^2={r['weight_star']:.4f}, n*={r['n_star']}, {rc}")

        elapsed = time.time() - t0
        final = results[-1]
        print(f"  => N={final['N']}, gap={final['gap']:.6e}, "
              f"cauchy={final['cauchy_bound']:.6e}, |v0|^2={final['weight_star']:.4f} "
              f"({elapsed:.1f}s)")

        all_results[lam] = {
            'L': 2*L, 'n_primes': len(primes_used),
            'N_final': final['N'],
            'gap': final['gap'], 'cauchy_bound': final['cauchy_bound'],
            'lmin': final['lmin'], 'l2': final['l2'],
            'n_star': final['n_star'], 'weight_star': final['weight_star'],
            'top3_modes': final['top3_modes'],
            'top3_weights': final['top3_weights'],
            'cauchy2_bound': final['cauchy2_bound'],
            'all_N': results
        }

    # Zusammenfassung
    print(f"\n{'='*75}")
    print(f"ZUSAMMENFASSUNG: CAUCHY-INTERLACING")
    print(f"{'='*75}")
    print(f"\n  {'lam':>5} | {'L':>6} | {'N':>3} | {'gap':>12} | {'cauchy':>12} | "
          f"{'ratio':>6} | {'n*':>4} | {'|v0|^2':>8} | {'cauchy>0':>8}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-"
          f"{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}")

    all_positive = True
    for lam in lambdas:
        r = all_results[lam]
        ratio = r['cauchy_bound'] / r['gap'] if r['gap'] > 0 else 0
        positive = r['cauchy_bound'] > 0
        if not positive:
            all_positive = False
        print(f"  {lam:5d} | {r['L']:6.3f} | {r['N_final']:3d} | {r['gap']:12.6e} | "
              f"{r['cauchy_bound']:+12.6e} | {ratio:6.2f} | {r['n_star']:4d} | "
              f"{r['weight_star']:8.4f} | {'JA' if positive else 'NEIN':>8}")

    print(f"\n  ALLE CAUCHY-SCHRANKEN POSITIV: {'JA' if all_positive else 'NEIN'}")

    # Min weight analysis
    min_weight = min(r['weight_star'] for r in all_results.values())
    min_cauchy = min(r['cauchy_bound'] for r in all_results.values())
    print(f"  Minimales |v0(n*)|^2: {min_weight:.4f}")
    print(f"  Minimale Cauchy-Schranke: {min_cauchy:.6e}")

    # Speichere
    outfile = "cauchy_results_server.json"
    with open(outfile, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\n  Ergebnisse gespeichert: {outfile}")
