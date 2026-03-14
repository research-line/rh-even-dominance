#!/usr/bin/env python3
"""
weg2_einfachheit_test.py
========================
Teste ob No-Coordination die Einfachheit des kleinsten EW erzwingt.

Connes' Theorem 6.1 braucht: lambda_min ist EINFACH (nicht entartet).

Hypothese: Die arithmetische Unabhaengigkeit log p / log q notin Q
(No-Coordination Lemma) verhindert Entartung, weil:
- W_p verschiebt Eigenvektoren um log(p) (auf [-L,L])
- Zwei verschiedene Primzahlen p,q verschieben um inkommensurable Betraege
- Entartung wuerde erfordern, dass ZWEI verschiedene Funktionen gleichzeitig
  Eigenvektoren von ALLEN W_p sind -> unmoeglich bei generischer Nicht-Kommensurabilit

Tests:
1. Spektralluecke: Wie haengt sie von der Primzahlmenge ab?
2. Entartungsaufhebung: Fuege Primzahlen einzeln hinzu, beobachte Luecke
3. Nur-Arch vs. Arch+Primes: Ist W_arch allein entartet?
4. Perturbationstheorie: delta_gap / ||W_p|| Verhaeltnis
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
import time

# ===========================================================================
# Basis (identisch mit v2)
# ===========================================================================

def make_basis_grid(N_basis, t_grid, L):
    phi = np.zeros((N_basis, len(t_grid)))
    phi[0, :] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N_basis):
        phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi

def make_shifted_basis(N_basis, t_grid, L, shift):
    t_shifted = t_grid - shift
    mask = np.abs(t_shifted) <= L
    phi = np.zeros((N_basis, len(t_grid)))
    phi[0, mask] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N_basis):
        phi[n, mask] = np.cos(n * np.pi * t_shifted[mask] / (2 * L)) / np.sqrt(L)
    return phi

# ===========================================================================
# Operator-Bausteine
# ===========================================================================

from mpmath import euler as mp_euler, log as mplog, pi as mppi
LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

def build_W_arch(N_basis, lam, n_quad=600, n_int=400):
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi_grid = make_basis_grid(N_basis, t_grid, L)

    W = LOG4PI_GAMMA * np.eye(N_basis)
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        kernel = 1.0 / (2.0 * np.sinh(s))
        if kernel < 1e-15:
            continue
        phi_plus = make_shifted_basis(N_basis, t_grid, L, s)
        phi_minus = make_shifted_basis(N_basis, t_grid, L, -s)
        S_plus = (phi_grid @ phi_plus.T) * dt
        S_minus = (phi_grid @ phi_minus.T) * dt
        W += (S_plus + S_minus - 2.0 * np.exp(-s/2) * np.eye(N_basis)) * kernel * ds

    return W, phi_grid, t_grid, dt

def build_W_p(p, N_basis, phi_grid, t_grid, L, dt, M_terms=10):
    logp = np.log(p)
    W = np.zeros((N_basis, N_basis))
    for m in range(1, M_terms + 1):
        coeff = logp * p**(-m / 2.0)
        shift = m * logp
        if shift >= 2 * L:
            break
        for sign in [1.0, -1.0]:
            phi_s = make_shifted_basis(N_basis, t_grid, L, sign * shift)
            S = (phi_grid @ phi_s.T) * dt
            W += coeff * S
    return W

# ===========================================================================
# TEST 1: Entartungsaufhebung durch einzelne Primzahlen
# ===========================================================================

def test_degeneracy_lifting(lam, N_basis):
    """Fuege Primzahlen einzeln hinzu, beobachte Spektralluecke."""
    print(f"\n{'='*75}")
    print(f"TEST 1: ENTARTUNGSAUFHEBUNG (lambda={lam}, N={N_basis})")
    print(f"{'='*75}")

    L = np.log(lam)
    W_arch, phi_grid, t_grid, dt = build_W_arch(N_basis, lam)

    # Nur archimedisch
    evals_arch = np.sort(eigh(W_arch, eigvals_only=True))
    gap_arch = evals_arch[1] - evals_arch[0]
    print(f"\n  NUR ARCHIMEDISCH:")
    print(f"    lambda_min = {evals_arch[0]:+.8e}")
    print(f"    lambda_2   = {evals_arch[1]:+.8e}")
    print(f"    Luecke     = {gap_arch:.8e}")
    print(f"    Erste 6 EW: {np.array2string(evals_arch[:6], precision=6)}")

    # Fuege Primzahlen einzeln hinzu
    primes = list(primerange(2, 100))

    print(f"\n  PRIMSUMME INKREMENTELL:")
    print(f"  {'Primes':>20} | {'lam_min':>12} | {'lam_2':>12} | {'Luecke':>12} | "
          f"{'||W_p||':>10} | {'dGap/||Wp||':>12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")

    QW = W_arch.copy()
    prev_gap = gap_arch

    for i, p in enumerate(primes[:20]):
        W_p = build_W_p(p, N_basis, phi_grid, t_grid, L, dt, M_terms=10)
        QW = QW + W_p
        evals = np.sort(eigh(QW, eigvals_only=True))
        gap = evals[1] - evals[0]
        norm_Wp = np.linalg.norm(W_p, 'fro')
        d_gap = gap - prev_gap
        ratio = d_gap / norm_Wp if norm_Wp > 1e-15 else 0

        primes_str = ",".join(str(q) for q in primes[:i+1])
        if len(primes_str) > 20:
            primes_str = f"2..{p} ({i+1} primes)"

        print(f"  {primes_str:>20} | {evals[0]:+12.6e} | {evals[1]:+12.6e} | "
              f"{gap:12.6e} | {norm_Wp:10.4e} | {ratio:+12.6e}")

        prev_gap = gap

# ===========================================================================
# TEST 2: Kommensurabilität und Entartung
# ===========================================================================

def test_commensurability(lam, N_basis):
    """Vergleiche: Primzahlen (inkommensurabel) vs. Primzahlpotenzen (kommensurabel)."""
    print(f"\n{'='*75}")
    print(f"TEST 2: KOMMENSURABILIAET (lambda={lam}, N={N_basis})")
    print(f"{'='*75}")

    L = np.log(lam)
    W_arch, phi_grid, t_grid, dt = build_W_arch(N_basis, lam)

    # Fall A: Verschiedene Primzahlen (log p / log q irrational)
    primes_diverse = [2, 3, 5, 7, 11, 13]
    QW_A = W_arch.copy()
    for p in primes_diverse:
        QW_A += build_W_p(p, N_basis, phi_grid, t_grid, L, dt)
    evals_A = np.sort(eigh(QW_A, eigvals_only=True))
    gap_A = evals_A[1] - evals_A[0]

    # Fall B: Nur Potenzen von 2 (log 2^k / log 2^m = k/m rational!)
    # Simuliere: Verwende nur p=2 mit vielen M_terms
    QW_B = W_arch.copy()
    QW_B += build_W_p(2, N_basis, phi_grid, t_grid, L, dt, M_terms=30)
    evals_B = np.sort(eigh(QW_B, eigvals_only=True))
    gap_B = evals_B[1] - evals_B[0]

    # Fall C: Nur Potenzen von 2 und 3 (inkommensurabel!)
    QW_C = W_arch.copy()
    QW_C += build_W_p(2, N_basis, phi_grid, t_grid, L, dt, M_terms=30)
    QW_C += build_W_p(3, N_basis, phi_grid, t_grid, L, dt, M_terms=30)
    evals_C = np.sort(eigh(QW_C, eigvals_only=True))
    gap_C = evals_C[1] - evals_C[0]

    print(f"\n  Fall A: 6 verschiedene Primzahlen {{2,3,5,7,11,13}}")
    print(f"    Luecke = {gap_A:.8e}")
    print(f"    EW: {np.array2string(evals_A[:4], precision=6)}")

    print(f"\n  Fall B: NUR p=2, 30 Potenzen (alles kommensurabel!)")
    print(f"    Luecke = {gap_B:.8e}")
    print(f"    EW: {np.array2string(evals_B[:4], precision=6)}")

    print(f"\n  Fall C: p=2 UND p=3, je 30 Potenzen (log2/log3 irrational)")
    print(f"    Luecke = {gap_C:.8e}")
    print(f"    EW: {np.array2string(evals_C[:4], precision=6)}")

    print(f"\n  VERGLEICH:")
    print(f"    Luecke(diverse) / Luecke(nur_2) = {gap_A / gap_B:.4f}" if gap_B > 0 else
          f"    Luecke(nur_2) = 0!")
    print(f"    Luecke(2+3) / Luecke(nur_2) = {gap_C / gap_B:.4f}" if gap_B > 0 else "")
    print(f"\n  => Erwartet: Inkommensurable Primzahlen => GROESSERE Luecke")
    print(f"  => No-Coordination: log p / log q irrational verhindert Entartung")

# ===========================================================================
# TEST 3: Luecke vs. Lambda (systematisch)
# ===========================================================================

def test_gap_vs_lambda(N_basis):
    """Systematische Spektralluecke als Funktion von lambda."""
    print(f"\n{'='*75}")
    print(f"TEST 3: SPEKTRALLUECKE vs. LAMBDA (N={N_basis})")
    print(f"{'='*75}")

    lambdas = [3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50]
    all_primes = list(primerange(2, 200))

    print(f"\n  {'lambda':>6} | {'L':>6} | {'n_primes':>8} | {'lam_min':>12} | "
          f"{'lam_2':>12} | {'Luecke':>12} | {'Luecke/L':>10}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for lam in lambdas:
        L = np.log(lam)
        primes_used = [p for p in all_primes if p <= max(lam, 47)]

        W_arch, phi_grid, t_grid, dt = build_W_arch(N_basis, lam)
        QW = W_arch.copy()
        for p in primes_used:
            QW += build_W_p(p, N_basis, phi_grid, t_grid, L, dt)

        evals = np.sort(eigh(QW, eigvals_only=True))
        gap = evals[1] - evals[0]
        gap_norm = gap / L if L > 0 else 0

        print(f"  {lam:6d} | {2*L:6.3f} | {len(primes_used):8d} | {evals[0]:+12.6e} | "
              f"{evals[1]:+12.6e} | {gap:12.6e} | {gap_norm:10.6f}")

# ===========================================================================
# TEST 4: Eigenvektor-Struktur des minimalen EW
# ===========================================================================

def test_evec_structure(lam, N_basis):
    """Analysiere die Struktur des minimalen Eigenvektors."""
    print(f"\n{'='*75}")
    print(f"TEST 4: EIGENVEKTOR-STRUKTUR (lambda={lam}, N={N_basis})")
    print(f"{'='*75}")

    L = np.log(lam)
    all_primes = list(primerange(2, max(int(lam) + 1, 48)))

    W_arch, phi_grid, t_grid, dt = build_W_arch(N_basis, lam)
    QW = W_arch.copy()
    for p in all_primes:
        QW += build_W_p(p, N_basis, phi_grid, t_grid, L, dt)

    evals, evecs = eigh(QW)

    v0 = evecs[:, 0]  # Kleinster EW
    v1 = evecs[:, 1]  # Zweitkleinster

    print(f"\n  lambda_0 = {evals[0]:+.8e}")
    print(f"  lambda_1 = {evals[1]:+.8e}")
    print(f"  Luecke   = {evals[1]-evals[0]:.8e}")

    # Entropie der Koeffizienten (Lokalisierung)
    p0 = v0**2
    p1 = v1**2
    S0 = -np.sum(p0 * np.log(p0 + 1e-30))
    S1 = -np.sum(p1 * np.log(p1 + 1e-30))
    S_max = np.log(N_basis)

    print(f"\n  Entropie: S(v0) = {S0:.4f}, S(v1) = {S1:.4f}, S_max = {S_max:.4f}")
    print(f"  Lokalisierung: v0 = {S0/S_max:.4f}, v1 = {S1/S_max:.4f}")

    # Dominante Koeffizienten
    idx0 = np.argsort(np.abs(v0))[::-1]
    idx1 = np.argsort(np.abs(v1))[::-1]
    print(f"\n  v0 dominante Moden: {idx0[:5]}, Betraege: {np.abs(v0[idx0[:5]])}")
    print(f"  v1 dominante Moden: {idx1[:5]}, Betraege: {np.abs(v1[idx1[:5]])}")

    # Overlap: Wie aehnlich sind v0 und v1?
    overlap = abs(np.dot(v0, v1))
    print(f"\n  |<v0|v1>| = {overlap:.2e} (sollte 0 sein)")

    # Symmetrie-Check: Gerade/Ungerade Anteile
    # Cosinus-Basis ist schon gerade -> alle Koeffizienten tragen bei
    even_energy = np.sum(v0[::2]**2)
    odd_energy = np.sum(v0[1::2]**2)
    print(f"\n  v0: even-index energy = {even_energy:.4f}, odd-index energy = {odd_energy:.4f}")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEG 2: EINFACHHEITS-TEST")
    print("Erzwingt No-Coordination die Einfachheit des kleinsten EW?")
    print("=" * 75)

    N_BASIS = 30
    LAM = 13

    # Test 1: Entartungsaufhebung
    test_degeneracy_lifting(LAM, N_BASIS)

    # Test 2: Kommensurabilität
    test_commensurability(LAM, N_BASIS)

    # Test 3: Luecke vs. Lambda
    test_gap_vs_lambda(N_BASIS)

    # Test 4: Eigenvektor-Struktur
    test_evec_structure(LAM, N_BASIS)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: EINFACHHEITS-TEST")
    print(f"{'='*75}")
    print(f"""
  NO-COORDINATION HYPOTHESE:

  log(p) / log(q) notin Q fuer p != q (verschiedene Primzahlen)

  Konsequenz fuer den Operator A_lambda:
  - W_p verschiebt Eigenfunktionen um log(p) auf [-L, L]
  - Die Verschiebungen sind INKOMMENSURABEL
  - Zwei Eigenfunktionen koennen nicht gleichzeitig Eigenvektoren
    von ALLEN W_p sein (sonst waere log p / log q rational)
  - Also: Entartung wird AUFGEHOBEN durch die Primsumme

  WENN Luecke(diverse Primzahlen) >> Luecke(nur eine Primzahl):
  => No-Coordination ist der Mechanismus fuer Einfachheit
  => Das waere der FST-RH Beitrag zu Connes' Programm!
""")
