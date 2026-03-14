#!/usr/bin/env python3
"""
weg2_even_odd_corrected.py
===========================
Korrigierte Even/Odd-Separation fuer Connes Theorem 6.1.

BUG-FIX: Die f(1)-Terme (Wert bei t=0 in additiver Koordinate) muessen als
RANG-1-Matrix behandelt werden, nicht als Identitaet:

  W_R(f) = (log 4pi + gamma) * f(0) + integral[1/(2sinh(u)) *
            (f(u) + f(-u) - 2*e^{-u/2}*f(0)) du]

Fuer die Bilinearform QW(phi_i, phi_j):
  (log 4pi + gamma)-Term:  (log 4pi + gamma) * phi_i(0) * phi_j(0)  [RANG 1]
  Integral-Korrektur:      -2*e^{-u/2}/(2sinh(u)) * phi_i(0) * phi_j(0) * du [RANG 1]

Fuer den Sinus-Sektor: phi_n(0) = sin(0) = 0 => ALLE f(0)-Terme = 0
Fuer den Kosinus-Sektor: phi_n(0) != 0 => korrekte Rang-1-Matrix noetig
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

def make_cos_basis(N, t_grid, L):
    phi = np.zeros((N, len(t_grid)))
    phi[0, :] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N):
        phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi

def make_cos_shifted(N, t_grid, L, shift):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    phi[0, mask] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N):
        phi[n, mask] = np.cos(n * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi

def make_sin_basis(N, t_grid, L):
    phi = np.zeros((N, len(t_grid)))
    for n in range(N):
        phi[n, :] = np.sin((n+1) * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi

def make_sin_shifted(N, t_grid, L, shift):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    for n in range(N):
        phi[n, mask] = np.sin((n+1) * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi


def build_QW_corrected(lam, N, primes, M_terms=12, n_quad=800, n_int=500, basis='cos'):
    """Korrigierter QW-Aufbau mit Rang-1 f(0)-Termen."""
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    if basis == 'cos':
        phi = make_cos_basis(N, t_grid, L)
        make_sh = make_cos_shifted
        # Basis-Werte bei t=0
        v0 = np.zeros(N)
        v0[0] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            v0[n] = 1.0 / np.sqrt(L)  # cos(0) = 1
    else:
        phi = make_sin_basis(N, t_grid, L)
        make_sh = make_sin_shifted
        # sin(0) = 0 fuer ALLE Basis-Funktionen
        v0 = np.zeros(N)

    # ===== KORRIGIERT: Rang-1 statt Identitaet =====
    # (log 4pi + gamma) * phi_i(0) * phi_j(0)
    W = LOG4PI_GAMMA * np.outer(v0, v0)

    # Archimedischer Integral-Term
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        k = 1.0 / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_sh(N, t_grid, L, s).T) * dt
        Sm = (phi @ make_sh(N, t_grid, L, -s).T) * dt
        # ===== KORRIGIERT: -2e^{-s/2} * f(0) als Rang-1 =====
        correction = 2.0 * np.exp(-s / 2) * np.outer(v0, v0)
        W += (Sp + Sm - correction) * k * ds

    # Primzahl-Terme (keine f(0)-Abhaengigkeit)
    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_sh(N, t_grid, L, sign * shift).T) * dt
                W += coeff * S

    return W


def run_even_odd_comparison(lambdas, primes, N=50):
    """Vergleiche Even und Odd Sektor mit korrektem QW."""
    print(f"{'='*75}")
    print(f"KORRIGIERTE EVEN/ODD-SEPARATION")
    print(f"BUG-FIX: f(0)-Terme als Rang-1-Matrix (nicht Identitaet)")
    print(f"{'='*75}")

    results = []

    for lam in lambdas:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]
        N_used = max(N, int(3 * L))

        t0 = time.time()

        # Even-Sektor (Cosinus)
        QW_even = build_QW_corrected(lam, N_used, primes_used, basis='cos')
        evals_even, evecs_even = eigh(QW_even)

        # Odd-Sektor (Sinus)
        QW_odd = build_QW_corrected(lam, N_used, primes_used, basis='sin')
        evals_odd = np.sort(eigh(QW_odd, eigvals_only=True))

        elapsed = time.time() - t0

        # Globale Analyse
        if evals_even[0] < evals_odd[0]:
            sector = "EVEN"
            global_gap = min(evals_even[1], evals_odd[0]) - evals_even[0]
        else:
            sector = "ODD"
            global_gap = min(evals_odd[1], evals_even[0]) - evals_odd[0]

        sector_gap = abs(evals_even[0] - evals_odd[0])

        r = {
            'lam': lam, 'L': 2*L, 'N': N_used,
            'l1_even': evals_even[0], 'l2_even': evals_even[1],
            'gap_even': evals_even[1] - evals_even[0],
            'l1_odd': evals_odd[0], 'l2_odd': evals_odd[1],
            'gap_odd': evals_odd[1] - evals_odd[0],
            'sector': sector, 'sector_gap': sector_gap,
            'global_gap': global_gap,
            'thm61': sector == 'EVEN' and global_gap > 0,
            'elapsed': elapsed
        }
        results.append(r)

        print(f"\n  lambda={lam:4d} (L={2*L:.2f}, N={N_used}, {elapsed:.1f}s):")
        print(f"    Even:  l1={evals_even[0]:+.6e}, l2={evals_even[1]:+.6e}, "
              f"gap={evals_even[1]-evals_even[0]:.4e}")
        print(f"    Odd:   l1={evals_odd[0]:+.6e}, l2={evals_odd[1]:+.6e}, "
              f"gap={evals_odd[1]-evals_odd[0]:.4e}")
        print(f"    Sektor-Abstand: {sector_gap:.4e} ({sector} ist tiefer)")
        print(f"    Globaler Gap: {global_gap:.4e}")
        print(f"    Theorem 6.1: {'ANWENDBAR' if r['thm61'] else 'NICHT ANWENDBAR'}")

    # Zusammenfassung
    print(f"\n{'='*75}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*75}")
    print(f"\n  {'lam':>5} | {'L':>6} | {'N':>3} | {'l1_even':>12} | {'l1_odd':>12} | "
          f"{'Sektor':>6} | {'Sek.Gap':>10} | {'Gl.Gap':>10} | {'Thm 6.1':>8}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-"
          f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    all_even = True
    for r in results:
        if r['sector'] != 'EVEN':
            all_even = False
        print(f"  {r['lam']:5d} | {r['L']:6.2f} | {r['N']:3d} | {r['l1_even']:+12.6e} | "
              f"{r['l1_odd']:+12.6e} | {r['sector']:>6} | {r['sector_gap']:10.4e} | "
              f"{r['global_gap']:10.4e} | {'JA' if r['thm61'] else 'NEIN':>8}")

    print(f"\n  ALLE GRUNDZUSTAENDE EVEN: {'JA' if all_even else 'NEIN'}")
    n_thm = sum(1 for r in results if r['thm61'])
    print(f"  Theorem 6.1 anwendbar: {n_thm}/{len(results)}")

    if not all_even:
        odd_cases = [r for r in results if r['sector'] == 'ODD']
        print(f"\n  WARNUNG: {len(odd_cases)} Faelle mit ODD-Grundzustand:")
        for r in odd_cases:
            print(f"    lambda={r['lam']}: l1_even={r['l1_even']:+.4e}, "
                  f"l1_odd={r['l1_odd']:+.4e}, Differenz={r['l1_even']-r['l1_odd']:+.4e}")

    # Vergleich mit ALTEM (buggigem) Code
    print(f"\n{'='*75}")
    print(f"VERGLEICH: BUGGY (I) vs KORREKT (vv^T)")
    print(f"{'='*75}")

    for lam in [10, 20, 50]:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]
        N_used = max(N, int(3 * L))

        # Buggy Version (Identitaet)
        QW_even_buggy = build_QW_buggy(lam, N_used, primes_used, basis='cos')
        QW_odd_buggy = build_QW_buggy(lam, N_used, primes_used, basis='sin')
        ev_even_b = np.sort(eigh(QW_even_buggy, eigvals_only=True))
        ev_odd_b = np.sort(eigh(QW_odd_buggy, eigvals_only=True))

        # Korrekte Version
        QW_even_corr = build_QW_corrected(lam, N_used, primes_used, basis='cos')
        QW_odd_corr = build_QW_corrected(lam, N_used, primes_used, basis='sin')
        ev_even_c = np.sort(eigh(QW_even_corr, eigvals_only=True))
        ev_odd_c = np.sort(eigh(QW_odd_corr, eigvals_only=True))

        print(f"\n  lambda={lam}:")
        print(f"    Buggy:   l1_even={ev_even_b[0]:+.4e}, l1_odd={ev_odd_b[0]:+.4e} "
              f"=> {'EVEN' if ev_even_b[0] < ev_odd_b[0] else 'ODD'}")
        print(f"    Korrekt: l1_even={ev_even_c[0]:+.4e}, l1_odd={ev_odd_c[0]:+.4e} "
              f"=> {'EVEN' if ev_even_c[0] < ev_odd_c[0] else 'ODD'}")
        print(f"    Shift Even: {ev_even_c[0] - ev_even_b[0]:+.4e}")
        print(f"    Shift Odd:  {ev_odd_c[0] - ev_odd_b[0]:+.4e}")

    return results


def build_QW_buggy(lam, N, primes, M_terms=12, n_quad=800, n_int=500, basis='cos'):
    """ALTE (buggy) Version mit I statt vv^T -- zum Vergleich."""
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    if basis == 'cos':
        phi = make_cos_basis(N, t_grid, L)
        make_sh = make_cos_shifted
    else:
        phi = make_sin_basis(N, t_grid, L)
        make_sh = make_sin_shifted

    W = LOG4PI_GAMMA * np.eye(N)
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        k = 1.0 / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_sh(N, t_grid, L, s).T) * dt
        Sm = (phi @ make_sh(N, t_grid, L, -s).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * k * ds

    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_sh(N, t_grid, L, sign * shift).T) * dt
                W += coeff * S

    return W


if __name__ == "__main__":
    print("=" * 75)
    print("PHASE 14b: KORRIGIERTE EVEN/ODD-SEPARATION")
    print("=" * 75)

    primes = list(primerange(2, 200))

    lambdas = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 80, 100, 200]

    results = run_even_odd_comparison(lambdas, primes, N=50)
