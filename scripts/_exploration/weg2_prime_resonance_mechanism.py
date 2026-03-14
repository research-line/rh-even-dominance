#!/usr/bin/env python3
"""
weg2_prime_resonance_mechanism.py
==================================
Untersuche den GENAUEN Mechanismus der Prim-Resonanz:

Warum bevorzugt der Prim-Operator bei grossem lambda den Even-Sektor?

Die Prim-Beitraege sind:
  W_prime = sum_p sum_m log(p) * p^{-m/2} * (S_{m*logp} + S_{-m*logp})

Shift-Matrix: <phi_n, S_s phi_k> = int_{-L}^{L} phi_n(t) phi_k(t-s) dt
  (wobei phi_k(t-s) = 0 wenn |t-s| > L)

Fuer cos: phi_n(t) = cos(n*pi*t/(2L))/sqrt(L)
Fuer sin: phi_n(t) = sin((n+1)*pi*t/(2L))/sqrt(L)

Die Shift-Matrixelemente haben die Struktur:
  cos-cos: Dirac-foermig auf Diagonale + langsam abklingend
  sin-sin: Aendert Vorzeichen -> mehr Cancellation

THESE: cos-Moden haben positivere Shift-Overlap bei grossen Shifts
weil cos(0) = 1 (Maximum am Rand), sin(0) = 0 (Null am Rand).
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
import time


def shift_overlap_trace(N, L, shift, basis='cos'):
    """Berechne Tr(S_s) = sum_n <phi_n, S_s phi_n> analytisch/numerisch."""
    n_quad = 2000
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    trace = 0.0
    for n in range(N):
        if basis == 'cos':
            if n == 0:
                phi_t = np.ones_like(t_grid) / np.sqrt(2 * L)
            else:
                phi_t = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
            ts = t_grid - shift
            mask = np.abs(ts) <= L
            phi_s = np.zeros_like(t_grid)
            if n == 0:
                phi_s[mask] = 1.0 / np.sqrt(2 * L)
            else:
                phi_s[mask] = np.cos(n * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
        else:
            phi_t = np.sin((n + 1) * np.pi * t_grid / (2 * L)) / np.sqrt(L)
            ts = t_grid - shift
            mask = np.abs(ts) <= L
            phi_s = np.zeros_like(t_grid)
            phi_s[mask] = np.sin((n + 1) * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)

        trace += np.sum(phi_t * phi_s) * dt

    return trace


def analyze_prime_mechanism():
    """Analysiere wie einzelne Prim-Shifts wirken."""
    primes = list(primerange(2, 100))

    print("=" * 80)
    print("PRIM-RESONANZ-MECHANISMUS")
    print("=" * 80)

    N = 40

    for lam in [20, 30, 50, 100]:
        L = np.log(lam)
        print(f"\n{'='*60}")
        print(f"lambda={lam}, L={L:.3f}, 2L={2*L:.3f}")
        print(f"{'='*60}")

        print(f"\n  {'p':>3} {'m':>2} {'shift':>7} {'s/2L':>5} "
              f"{'coeff':>8} {'Tr_cos':>10} {'Tr_sin':>10} {'diff':>10} "
              f"{'eff_cos':>10} {'eff_sin':>10}")
        print(f"  {'-'*3} {'-'*2} {'-'*7} {'-'*5} "
              f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} "
              f"{'-'*10} {'-'*10}")

        total_eff_cos = 0.0
        total_eff_sin = 0.0

        for p in primes:
            logp = np.log(p)
            for m in range(1, 8):
                shift = m * logp
                if shift >= 2 * L:
                    break
                coeff = logp * p**(-m / 2.0)
                if coeff < 1e-6:
                    continue

                # Tr(S_s + S_{-s}) fuer cos und sin
                tr_cos_p = shift_overlap_trace(N, L, shift, 'cos')
                tr_cos_m = shift_overlap_trace(N, L, -shift, 'cos')
                tr_sin_p = shift_overlap_trace(N, L, shift, 'sin')
                tr_sin_m = shift_overlap_trace(N, L, -shift, 'sin')

                tr_cos = tr_cos_p + tr_cos_m
                tr_sin = tr_sin_p + tr_sin_m

                eff_cos = coeff * tr_cos
                eff_sin = coeff * tr_sin
                total_eff_cos += eff_cos
                total_eff_sin += eff_sin

                ratio = shift / (2 * L)
                print(f"  {p:3d} {m:2d} {shift:7.3f} {ratio:5.2f} "
                      f"{coeff:8.4f} {tr_cos:+10.4f} {tr_sin:+10.4f} "
                      f"{tr_cos-tr_sin:+10.4f} "
                      f"{eff_cos:+10.4f} {eff_sin:+10.4f}")

        print(f"\n  SUMME: eff_cos={total_eff_cos:+.4f}, eff_sin={total_eff_sin:+.4f}, "
              f"diff={total_eff_cos-total_eff_sin:+.4f}")


def trace_vs_shift_curve():
    """Plotte Tr(S_s) als Funktion von s fuer cos vs sin."""
    print("\n" + "=" * 80)
    print("TRACE(S_s) VS SHIFT-KURVE")
    print("=" * 80)

    N = 40

    for lam in [30, 100]:
        L = np.log(lam)
        print(f"\nlambda={lam}, L={L:.3f}:")
        print(f"  {'s':>6} {'s/2L':>5} {'Tr_cos':>10} {'Tr_sin':>10} {'diff':>10}")
        print(f"  {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")

        for s in np.arange(0.1, 2 * L, 0.3):
            tr_c = shift_overlap_trace(N, L, s, 'cos')
            tr_s = shift_overlap_trace(N, L, s, 'sin')
            print(f"  {s:6.2f} {s/(2*L):5.2f} {tr_c:+10.4f} {tr_s:+10.4f} "
                  f"{tr_c-tr_s:+10.4f}")


if __name__ == "__main__":
    t0 = time.time()
    trace_vs_shift_curve()
    analyze_prime_mechanism()
    print(f"\nTotal: {time.time()-t0:.1f}s")
