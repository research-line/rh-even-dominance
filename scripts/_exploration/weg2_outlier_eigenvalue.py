#!/usr/bin/env python3
"""
weg2_outlier_eigenvalue.py
==========================
Untersuchung: Warum taucht bei N~30 ein "Outlier-Eigenwert" auf?

Bei lambda=20, N<=25: alle Eigenwerte im Cluster [-4.381, ...]
Bei lambda=20, N>=30: lambda_min springt auf -5.35 (weit unter Cluster)

Hypothesen:
  (H1) Hochfrequenz-Mode: Die n~25. Cosinus-Basis-Funktion hat Frequenz
       omega_25 = 25*pi/(2*L). Bei L=3.0: omega_25 ~ 13.1 ~ gamma_1!
       => Diese Basisfunktion "sieht" die erste Zeta-Nullstelle.

  (H2) Numerischer Artefakt: Die Quadratur ist zu grob fuer hohe Frequenzen.

  (H3) W_p Resonanz: log(p) ist mit omega_n kommensurabel fuer bestimmte n.

Tests:
  1. Welche Mode dominiert den Outlier-Eigenvektor?
  2. Was ist die Frequenz dieser Mode?
  3. Verschwindet der Outlier bei feinerer Quadratur?
  4. Kommt der Outlier von W_arch oder W_prime?
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi

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

def build_parts(lam, N, primes, M_terms=12, n_quad=800, n_int=500):
    """Baue W_arch und W_prime getrennt."""
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = make_basis_grid(N, t_grid, L)

    # Archimedisch
    W_arch = LOG4PI_GAMMA * np.eye(N)
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        k = 1.0 / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_shifted(N, t_grid, L, s).T) * dt
        Sm = (phi @ make_shifted(N, t_grid, L, -s).T) * dt
        W_arch += (Sp + Sm - 2.0 * np.exp(-s/2) * np.eye(N)) * k * ds

    # Primzahlen
    W_prime = np.zeros((N, N))
    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_shifted(N, t_grid, L, sign * shift).T) * dt
                W_prime += coeff * S

    return W_arch, W_prime

# ===========================================================================
# TEST 1: Outlier-Eigenvektor-Analyse
# ===========================================================================

def test_outlier(lam=20):
    """Analysiere den Outlier-Eigenvektor."""
    print(f"\n{'='*75}")
    print(f"OUTLIER-EIGENVEKTOR-ANALYSE (lambda={lam})")
    print(f"{'='*75}")

    L = np.log(lam)
    primes = list(primerange(2, max(int(lam) + 1, 48)))

    for N in [25, 30, 35, 40]:
        W_arch, W_prime = build_parts(lam, N, primes)
        QW = W_arch + W_prime

        evals, evecs = eigh(QW)
        v0 = evecs[:, 0]

        # Dominante Moden
        idx_sorted = np.argsort(np.abs(v0))[::-1]
        top_modes = idx_sorted[:5]
        top_vals = np.abs(v0[top_modes])

        # Frequenz der dominanten Mode
        freqs = [n * np.pi / (2 * L) for n in top_modes]

        print(f"\n  N={N}:")
        print(f"    lambda_min = {evals[0]:+.8e}")
        print(f"    lambda_2   = {evals[1]:+.8e}")
        print(f"    gap        = {evals[1]-evals[0]:.8e}")
        print(f"    Top-Moden:  {top_modes}")
        print(f"    |Koeff|:    {np.array2string(top_vals, precision=4)}")
        print(f"    Frequenzen: {[f'{f:.3f}' for f in freqs]}")
        print(f"    gamma_1 = 14.135, omega_n = {[f'{n*np.pi/(2*L):.3f}' for n in top_modes[:3]]}")

        # Ist der Outlier-EV hochfrequent?
        mean_mode = np.average(np.arange(N), weights=v0**2)
        print(f"    Mittlere Mode: {mean_mode:.1f} (max={N-1})")

        # Zerlegung: Wie viel kommt von W_arch vs W_prime?
        arch_contrib = v0 @ W_arch @ v0
        prime_contrib = v0 @ W_prime @ v0
        print(f"    <v0|W_arch|v0> = {arch_contrib:+.6e}")
        print(f"    <v0|W_prime|v0> = {prime_contrib:+.6e}")
        print(f"    Summe = {arch_contrib + prime_contrib:+.6e} (sollte = {evals[0]:+.6e})")

# ===========================================================================
# TEST 2: Quadratur-Abhaengigkeit
# ===========================================================================

def test_quadrature(lam=20, N=35):
    """Verschwindet der Outlier bei feinerer Quadratur?"""
    print(f"\n{'='*75}")
    print(f"QUADRATUR-ABHAENGIGKEIT (lambda={lam}, N={N})")
    print(f"{'='*75}")

    primes = list(primerange(2, max(int(lam) + 1, 48)))

    for n_quad, n_int in [(400, 200), (600, 350), (800, 500), (1200, 700), (1600, 1000)]:
        W_arch, W_prime = build_parts(lam, N, primes, n_quad=n_quad, n_int=n_int)
        QW = W_arch + W_prime
        evals = np.sort(eigh(QW, eigvals_only=True))
        gap = evals[1] - evals[0]

        print(f"  n_quad={n_quad:5d}, n_int={n_int:4d}: "
              f"lmin={evals[0]:+.8e}, l2={evals[1]:+.8e}, gap={gap:.6e}")

# ===========================================================================
# TEST 3: Frequenz-Resonanz
# ===========================================================================

def test_frequency_resonance(lam=20):
    """Vergleiche Basis-Frequenzen mit log(p) und gamma_k."""
    print(f"\n{'='*75}")
    print(f"FREQUENZ-RESONANZ (lambda={lam})")
    print(f"{'='*75}")

    L = np.log(lam)
    from mpmath import mp, im, zetazero
    mp.dps = 25
    gammas = [float(im(zetazero(k))) for k in range(1, 11)]

    print(f"\n  Basis-Frequenzen omega_n = n*pi/(2*L), L = {L:.4f}")
    print(f"  Zeta-Nullstellen gamma_k:")
    print(f"  Primzahl-Shifts log(p):")

    print(f"\n  {'n':>4} | {'omega_n':>10} | {'naechstes gamma':>15} | {'Delta':>8} | "
          f"{'naechstes log(p)':>15} | {'Delta':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*15}-+-{'-'*8}-+-{'-'*15}-+-{'-'*8}")

    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    logp = [np.log(p) for p in primes]

    for n in range(1, 45):
        omega = n * np.pi / (2 * L)

        # Naechstes gamma
        dists_g = [abs(omega - g) for g in gammas]
        best_g = min(dists_g)
        best_g_idx = dists_g.index(best_g)

        # Naechstes log(p)
        dists_p = [abs(omega - lp) for lp in logp]
        best_p = min(dists_p)
        best_p_idx = dists_p.index(best_p)

        marker = " <== RESONANZ" if best_g < 0.5 else ""
        print(f"  {n:4d} | {omega:10.4f} | gamma_{best_g_idx+1}={gammas[best_g_idx]:8.4f} | "
              f"{best_g:8.4f} | log({primes[best_p_idx]})={logp[best_p_idx]:8.4f} | "
              f"{best_p:8.4f}{marker}")

        if n > 35 and best_g > 1:
            break

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("OUTLIER-EIGENWERT-UNTERSUCHUNG")
    print("=" * 75)

    # Test 1: Outlier-Eigenvektor
    test_outlier(lam=20)

    # Test 2: Quadratur-Abhaengigkeit
    test_quadrature(lam=20, N=35)

    # Test 3: Frequenz-Resonanz
    test_frequency_resonance(lam=20)

    print(f"\n{'='*75}")
    print("FAZIT")
    print(f"{'='*75}")
