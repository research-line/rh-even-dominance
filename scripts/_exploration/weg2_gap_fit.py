#!/usr/bin/env python3
"""
weg2_gap_fit.py
===============
Fit-Analyse: gap(lambda) = a + b*log(lambda) + ... ?

Nutzt die LOKALEN Daten (lambda=3..50, N=30) plus erweiterte Berechnung
bis lambda=80 (N=35, lokal machbar).

Fit-Modelle:
  (A) gap = a + b * log(lambda)
  (B) gap = a + b * log(lambda) + c * log(lambda)^2
  (C) gap = a * log(lambda)^b  (Potenzgesetz)
  (D) gap = a + b * sqrt(log(lambda))
  (E) gap = a * (1 - exp(-b * log(lambda)))  (Saettigung)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

# ===========================================================================
# Operator-Bausteine (aus v2, kompakt)
# ===========================================================================

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

def build_QW(lam, N, primes, M_terms=10, n_quad=600, n_int=350):
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = make_basis_grid(N, t_grid, L)

    # Archimedisch
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

    # Primzahlen
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

def compute_gap(lam, N, primes, **kwargs):
    """Berechne Spektralluecke fuer gegebenes lambda."""
    QW = build_QW(lam, N, primes, **kwargs)
    evals = np.sort(eigh(QW, eigvals_only=True))
    gap = evals[1] - evals[0]
    return gap, evals[0], evals[1]

# ===========================================================================
# Daten sammeln
# ===========================================================================

def collect_data():
    """Sammle gap(lambda) Daten."""
    print("Sammle Spektralluecken-Daten...")

    all_primes = list(primerange(2, 200))
    N = 35

    lambdas = [3, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 18, 20, 22, 25,
               28, 30, 35, 40, 45, 50, 55, 60, 70, 80]

    print(f"\n  {'lambda':>6} | {'L':>6} | {'gap':>12} | {'lam_min':>12} | {'lam_2':>12} | {'Zeit':>5}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*5}")

    data = []
    for lam in lambdas:
        t0 = time.time()
        primes_used = [p for p in all_primes if p <= max(lam, 47)]
        gap, lmin, l2 = compute_gap(lam, N, primes_used)
        elapsed = time.time() - t0

        data.append((lam, np.log(lam), gap, lmin, l2))
        print(f"  {lam:6d} | {2*np.log(lam):6.3f} | {gap:12.6e} | {lmin:+12.6e} | {l2:+12.6e} | {elapsed:5.1f}s")

    return np.array(data)

# ===========================================================================
# Fit-Modelle
# ===========================================================================

def fit_models(data):
    """Fitte verschiedene Modelle an gap(lambda)."""
    print(f"\n{'='*75}")
    print(f"FIT-ANALYSE")
    print(f"{'='*75}")

    lam_vals = data[:, 0]
    log_lam = data[:, 1]
    gaps = data[:, 2]

    # Nur lambda >= 6 verwenden (darunter ist gap ~ 0)
    mask = lam_vals >= 6
    x = log_lam[mask]
    y = gaps[mask]

    # Modell A: gap = a + b * log(lambda)
    def model_A(x, a, b):
        return a + b * x

    # Modell B: gap = a + b * log(lambda) + c * log(lambda)^2
    def model_B(x, a, b, c):
        return a + b * x + c * x**2

    # Modell C: gap = a * log(lambda)^b
    def model_C(x, a, b):
        return a * x**b

    # Modell D: gap = a + b * sqrt(log(lambda))
    def model_D(x, a, b):
        return a + b * np.sqrt(x)

    # Modell E: gap = a * (1 - exp(-b * log(lambda)))
    def model_E(x, a, b):
        return a * (1 - np.exp(-b * x))

    models = [
        ("A: a + b*log(lam)", model_A, [0, 0.5], 2),
        ("B: a + b*L + c*L^2", model_B, [0, 0, 0], 3),
        ("C: a * log(lam)^b", model_C, [0.1, 1.5], 2),
        ("D: a + b*sqrt(L)", model_D, [-1, 1], 2),
        ("E: a*(1-exp(-b*L))", model_E, [3, 0.3], 2),
    ]

    print(f"\n  Daten: {len(x)} Punkte (lambda >= 6)")
    print(f"\n  {'Modell':>25} | {'RSS':>12} | {'AIC':>10} | Parameter")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+----------")

    best_rss = float('inf')
    best_model = None

    for name, func, p0, n_params in models:
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
            y_pred = func(x, *popt)
            rss = np.sum((y - y_pred)**2)
            n = len(x)
            aic = n * np.log(rss / n) + 2 * n_params

            param_str = ", ".join(f"{p:.6f}" for p in popt)
            print(f"  {name:>25} | {rss:12.6e} | {aic:10.4f} | {param_str}")

            if rss < best_rss:
                best_rss = rss
                best_model = (name, func, popt)
        except Exception as e:
            print(f"  {name:>25} | FEHLER: {e}")

    if best_model:
        name, func, popt = best_model
        print(f"\n  BESTES MODELL: {name}")
        print(f"  Parameter: {popt}")

        # Extrapolation
        print(f"\n  EXTRAPOLATION:")
        for lam_ext in [100, 200, 500, 1000, 10000]:
            gap_pred = func(np.log(lam_ext), *popt)
            print(f"    lambda={lam_ext:>6}: gap = {gap_pred:.4f}")

    # Spezialanalyse: gap / log(lambda)
    print(f"\n  GAP / LOG(LAMBDA) Verhaeltnis:")
    for i in range(len(lam_vals)):
        if lam_vals[i] >= 6:
            ratio = gaps[i] / log_lam[i]
            print(f"    lambda={lam_vals[i]:>4.0f}: gap/L = {ratio:.4f}")

    return best_model

# ===========================================================================
# Konsistenz-Check: N-Abhaengigkeit
# ===========================================================================

def check_N_dependence(lam=20):
    """Ist die Luecke stabil unter Erhoehung von N?"""
    print(f"\n{'='*75}")
    print(f"N-ABHAENGIGKEITS-CHECK (lambda={lam})")
    print(f"{'='*75}")

    all_primes = list(primerange(2, 200))
    primes_used = [p for p in all_primes if p <= max(lam, 47)]

    N_vals = [15, 20, 25, 30, 35, 40]

    print(f"\n  {'N':>5} | {'gap':>12} | {'lam_min':>12} | {'lam_2':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for N in N_vals:
        gap, lmin, l2 = compute_gap(lam, N, primes_used)
        print(f"  {N:5d} | {gap:12.6e} | {lmin:+12.6e} | {l2:+12.6e}")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("PHASE 9: FIT-ANALYSE DER SPEKTRALLUECKE")
    print("  gap(lambda) = f(log(lambda)) ?")
    print("=" * 75)

    # Daten sammeln
    data = collect_data()

    # Fit
    best = fit_models(data)

    # N-Konsistenz
    check_N_dependence(lam=20)
    check_N_dependence(lam=40)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: FIT-ANALYSE")
    print(f"{'='*75}")
    print(f"""
  ZIEL: Bestimme funktionale Abhaengigkeit gap(lambda).

  Wenn gap ~ c * log(lambda):
    => gap -> inf fuer lambda -> inf
    => Theorem 6.1 Voraussetzung fuer ALLE lambda erfuellt
    => RH folgt aus Connes' Framework

  Wenn gap -> Konstante (Saettigung):
    => Immer noch gap > 0 fuer alle lambda
    => Theorem 6.1 erfuellt, aber weniger robust

  Wenn gap -> 0:
    => Theorem 6.1 koennte scheitern
    => Braucht schaerfere Analyse
""")
